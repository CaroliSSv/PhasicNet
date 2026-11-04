#optuna_sweep - Optuna around PhasicNet.train(): one trial = one parallel.Job in its own process (parallel.run_in_process), the study in runs/optuna/<study>/study.db.
#
#   python -c "import experiments, optuna_sweep as o; o.sweep('hlsp_w_in', o.hlsp_w_in, n_trials=20, workers=3)"
#   -> runs/optuna/<study>/: study.db (Optuna's own file), trials.csv (one row per trial), trial_0000/ ... (each trial's run folder)
#
#space(trial) -> Job: suggests the parameters and returns the Job for them (an importable function, hlsp_w_in below is the example).
#n_trials counts FINISHED trials (COMPLETE + PRUNED): calling sweep() again (after a reboot) only runs the missing ones; a trial cut off by the
# reboot is marked FAIL. metric = the key of run.report the study optimizes. One machine per study.db (SQLite on a synced folder breaks).
#
##optuna pruning (stopping a trial that is doing badly, OPTIONAL, off unless a pruner is given):
#   sweep(..., pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2), report=("epoch", "score"))
#   While a trial runs, the parent reads the run's results.jsonl every poll_every seconds. Each line whose message is report[0] and that
#   carries the key report[1] is one intermediate value (step 1, 2, 3 ...), handed to Optuna (trial.report). If Optuna's pruner says the trial
#   is below the others at that step (MedianPruner: below the median of the finished trials), the child process is killed, the trial is marked
#   PRUNED and pruned.json is written in its run folder. The run_id and the job files do not change: the pruner lives here only.
#   Lines in results.jsonl: "epoch" + "score" = ops.run_fit, i.e. the protocols of protocols.py (D&C Holdout / CrossValidation: the validation
#   separability per epoch); "rstdp" + "val_decision_acc" = microcol_mozafari (the validation accuracy per R-STDP pass). The LSM builds (HLSP,
#   NALSM, Deckers, Krenzer, Zhang) and microcol_hlsp write no number during the run (one pass, no epochs): a pruner never stops them.
#   The intermediate value must go the same way as the metric (direction): a higher score = better when the metric is maximized.
#   The early stop (patience in ops.run_fit) is a different thing: a run against its own history. It stays as it is.
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path
import optuna
from optuna.trial import TrialState
from paths import RUNS_DIR
from parallel import Job, THREAD_VARS, run_in_process
import hlsp_lsm_2026 as hlsp

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)     ##heartbeat is marked experimental; it is what recovers a reboot


class _Reports:
    """##optuna pruning: reads NEW lines of one run's results.jsonl and reports the chosen number to Optuna. Called by run_in_process every poll_every
    seconds (as poll()); returns True when Optuna says: stop this trial."""
    def __init__(self, out_dir, trial, record: str, key: str):
        self.path, self.trial, self.record, self.key = (None if out_dir is None else Path(out_dir) / "results.jsonl"), trial, record, key
        self.offset, self.step, self.pruned, self.last = 0, 0, False, None

    def __call__(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        with open(self.path, "rb") as f:                                     ##only the bytes written since the last look
            f.seek(self.offset)
            chunk = f.read()
        lines = chunk.split(b"\n")
        self.offset += len(chunk) - len(lines[-1])                           ##a half-written last line is read next time
        for line in lines[:-1]:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("message") != self.record or d.get(self.key) is None:
                continue
            self.step += 1
            self.last = float(d[self.key])
            self.trial.report(self.last, self.step)
            if self.trial.should_prune():
                self.pruned = True
                return True
        return False


def sweep(study: str, space, n_trials: int, *, metric: str = "val_acc", direction: str = "maximize", workers: int = 1, sampler_seed: int = 0,
          pruner=None, report: tuple[str, str] = ("epoch", "score"), poll_every: float = 30.0):
    """Runs the study until it has n_trials finished (COMPLETE + PRUNED) trials; returns the optuna Study (study.trials_dataframe() for the table).
    pruner = an optuna.pruners.* object (header) or None = no pruning; report = (results.jsonl message, key) of the number reported while a trial runs."""
    ##the study: its folder and its SQLite file; load_if_exists continues an older study of the same name (after a reboot)
    folder = RUNS_DIR / "optuna" / study
    folder.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{(folder / 'study.db').as_posix()}", engine_kwargs={"connect_args": {"timeout": 60}},
                                         heartbeat_interval=60, grace_period=300)     ##heartbeat: a trial silent for 5 min (reboot) becomes FAIL
    st = optuna.create_study(study_name=study, storage=storage, direction=direction, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=sampler_seed),
                             pruner=pruner if pruner is not None else optuna.pruners.NopPruner())     ##optuna pruning: NopPruner = never prune (Optuna's default would be MedianPruner)
    optuna.storages.fail_stale_trials(st)
    st.sampler = optuna.samplers.TPESampler(seed=sampler_seed + len(st.trials))     ##a continued study draws NEW random values: the same seed would repeat the first trials' values
    os.environ.update({k: "1" for k in THREAD_VARS})      ##set once here: the trial threads then never race over parallel._pinned's restore

    def objective(trial):
        ##one trial: the space gives the Job, the Job runs in a child process, the metric comes back in its report
        job = space(trial)
        out_dir = job.train_kw.get("out_dir")
        watch = _Reports(out_dir, trial, *report) if pruner is not None else None          ##optuna pruning: the poll, only with a pruner
        r = run_in_process(job, poll=watch, every=poll_every)
        trial.set_user_attr("out_dir", r.out_dir)
        if watch is not None and watch.pruned:                                             ##optuna pruning: the child was killed at step watch.step
            if out_dir is not None:
                (Path(out_dir) / "pruned.json").write_text(json.dumps({"step": watch.step, "value": watch.last, "record": report[0], "key": report[1]}, indent=1))
            raise optuna.TrialPruned(f"pruned at step {watch.step} ({report[1]} = {watch.last})")
        if not r.ok:
            raise RuntimeError(r.error)
        trial.set_user_attr("run_id", r.run_id)
        for key, v in (r.report or {}).items():                                            ##every plain number of the report: the dashboard's table
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                trial.set_user_attr(key, v)
        value = (r.report or {}).get(metric)
        if value is None:
            raise RuntimeError(f"run.report has no {metric!r}; keys: {sorted(r.report or {})}")
        return value

    ##how many trials are still missing, then Optuna runs them (workers at a time, each a thread of this process holding one child process)
    left = n_trials - len(st.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED)))
    if left > 0:
        st.optimize(objective, n_trials=left, n_jobs=workers, catch=(RuntimeError,))
    st.trials_dataframe().to_csv(folder / "trials.csv", index=False)
    return st


def hlsp_w_in(trial, *, n_train: int = 2000, n_test: int = 500, seed: int = 1) -> Job:
    """EXAMPLE space: HLSP on MNIST, only the input weight searched. Copy it for your own parameters."""
    w_in = trial.suggest_float("w_in", 0.25, 1.0)
    out = RUNS_DIR / "optuna" / trial.study.study_name / f"trial_{trial.number:04d}"
    return Job(build=hlsp.build, load=hlsp.load_dataset, runcfg=hlsp.runcfg(n_train=n_train, n_test=n_test, partitions=hlsp.PARTITIONS["mnist"]),
               build_kw=dict(w_in=w_in, seed=seed), load_kw=dict(name="mnist", seed=seed),
               train_kw=dict(out_dir=out, run_name=f"trial_{trial.number:04d}", dataset_name="mnist", save_run=False))
