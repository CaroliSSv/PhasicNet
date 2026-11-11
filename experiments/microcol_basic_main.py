#microcol_basic_main - RUN the back-to-basics microcolumn experiment. Execution only; the experiment is:
#   microcol_basic_cfg.yaml       ExperimentCfg = dynamics (LIF, fixed thresholds + refractory + escape noise, soft-bounded STDP / R-STDP) + run
#   microcol_basic_network.yaml   NetSpec: input -> L1 -> L2 -> L3, one exc + one inh group per layer, set_columns(4), fixed inhibition and
#                                 feedback inhibition as synapse groups
#   experiment_dynamics/microcol_basic.py   the runtime modules
#   microcol_basic_protocol.py    untrained responses -> STDP L1, L2 -> R-STDP L3 -> scores + the sweep records (its header lists them)
#
#RUN (repo root, conda activate PhasicNet):
#   python -c "import experiments, microcol_basic_main as m; m.main(n_train=600, n_test=200, tag='t1')"
#   values = any cfg field by its dotted path:  m.main(values={"dynamics.threshold.beta": 0.15, "run.protocol.rstdp_epochs": 5})
#   weights / connectivity = builder.set_weights / set_connectivity on top of the network file:  m.main(weights={"L1_ie": -2.0, "fb_L2_L1": 0.0})
#   resume="auto" continues the newest snapshot in the run folder (needs every=N or run.checkpoint.every)
#SEARCHES (optuna_sweep.sweep): space_rates = the untrained network's firing only (no training, short); space_learn = the whole pipeline;
# replicate(...) = the best trials of a study again with other seeds. params_job(...) is the one place that turns search values into a Job.
# sweep_batch(...) = a queue job's share of a study: trials for a number of hours, its run folders and its own trials.csv in one batch folder.
#DATA: N-MNIST through tonic (RAW_DIR; downloaded on first use, ~1 GB): events binned at dt, any event = one spike, the first t_ms of every
# sample (100 ms = the first of the three saccades). ModelInput(column_blocks=(2, 2)) gives every column one quadrant of the sensor.
from __future__ import annotations
import shutil
import time
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
from phasicnet import PhasicNet
from runcfg import ExperimentCfg
from parallel import Job, run_parallel
from paths import RUNS_DIR, RAW_DIR
from checkpoint import snapshots
from modelinput import ModelInput
from microcol_basic import MICROCOL_BASIC
from microcol_basic_protocol import BasicProtocol

HERE = Path(__file__).resolve().parent
CFG_FILE, NET_FILE = HERE / "microcol_basic_cfg.yaml", HERE / "microcol_basic_network.yaml"
SEED = 1
COLUMN_BLOCKS = (2, 2)          ##image blocks (rows, cols) = the network file's 4 columns
EXC = ("L1_exc", "L2_exc", "L3_exc")


def build(*, seed: int = SEED, values: dict | None = None, weights: dict | None = None, connectivity: dict | None = None) -> PhasicNet:
    """The two files -> a PhasicNet ready for train(): cfg (+ values), network spec (+ weights / connectivity) realized, the runtime pack."""
    pnet = PhasicNet(seed)
    exp = pnet.load_experiment(CFG_FILE).override(values)
    pnet.cfg, pnet.runcfg = exp.dynamics, exp.run
    pnet._instantiate_builder()
    pnet.builder.load_specs(NET_FILE)
    if weights:
        pnet.builder.set_weights(weights)
    if connectivity:
        pnet.builder.set_connectivity(connectivity)
    pnet.builder.realize(network_name="microcol_basic", post_offset=True, allow_autapse=False)
    pnet.set_runtime_classes(replace(MICROCOL_BASIC, protocol_classes=(BasicProtocol,)))
    return pnet


@contextmanager
def _one_at_a_time(lock: Path, stale_s: float = 3 * 3600):
    """Jobs started together: one of them at a time is inside (load: only the first downloads and unpacks N-MNIST, the others wait).
    The lock is an empty folder; one older than stale_s (a job killed inside) is taken over."""
    while True:
        try:
            lock.mkdir(parents=True)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:
                    break
            except FileNotFoundError:
                continue
            time.sleep(5)
    try:
        yield
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def load(*, dataset: str = "nmnist", t_ms: float = 100.0, dt_ms: float = 1.0):
    """(train, test) ModelInputs: N-MNIST (header). t_ms must cover the protocol's t_train_ms / t_state_ms (job() and main() pass it)."""
    if dataset != "nmnist":
        raise ValueError(f"Unknown dataset {dataset!r}: nmnist (the network file has its 2 x 34 x 34 inputs)")
    import tonic
    from tonic_helpers import ClampBinary, FixedLength, US_TO_MS
    frame = tonic.transforms.Compose([tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=int(dt_ms * US_TO_MS)),
                                      ClampBinary(dtype=np.bool_), FixedLength(int(round(t_ms / dt_ms)))])
    both = lambda: [tonic.datasets.NMNIST(save_to=str(RAW_DIR), train=flag, transform=frame) for flag in (True, False)]
    ready = RAW_DIR / "NMNIST" / "phasicnet_ready"                  ##written after the first complete load: from then on nobody queues up
    if ready.exists():
        tr, te = both()
    else:
        with _one_at_a_time(RAW_DIR / "nmnist_in_use"):
            tr, te = both()
            ready.touch()
    return (ModelInput.from_tonic(tr, axis_labels="TPHW", column_blocks=COLUMN_BLOCKS),
            ModelInput.from_tonic(te, axis_labels="TPHW", column_blocks=COLUMN_BLOCKS))


def _out(dataset, seed, tag) -> Path:
    return RUNS_DIR / f"microcol_basic_{dataset}_s{seed}{('_' + tag) if tag else ''}"


def _resume(out: Path, resume: str | None) -> str | None:
    """resume="auto" -> the newest snapshot in the run folder (None if there is none); any other value is used as is."""
    if resume != "auto":
        return resume
    found = snapshots(out)
    return str(found[-1]) if found else None


def _values(values, n_train, n_test, every) -> dict:
    extra = {"run.data.n_train": n_train, "run.data.n_test": n_test, "run.checkpoint.every": every}
    return {**{k: v for k, v in extra.items() if v is not None}, **(values or {})}


def _load_kw(exp: ExperimentCfg, dataset: str) -> dict:
    return dict(dataset=dataset, t_ms=max(exp.run.protocol.t_train_ms, exp.run.protocol.t_state_ms), dt_ms=exp.dynamics.simulation.dt)


def job(*, values: dict | None = None, weights: dict | None = None, connectivity: dict | None = None, seed: int = SEED, dataset: str = "nmnist",
        n_train: int | None = None, n_test: int | None = None, every: int | None = None, resume: str | None = None, tag: str = "",
        out: str | Path | None = None, save_history: bool = False) -> Job:
    """ONE run as a parallel.Job (each in its own process): the parent resolves the RunCfg and the snapshot to resume (resume="auto"), the child
    builds the network and loads the data itself."""
    values = _values(values, n_train, n_test, every)
    out = Path(out) if out else _out(dataset, seed, tag)
    exp = ExperimentCfg.load_file(CFG_FILE).override(values)
    return Job(build=build, load=load, runcfg=exp.run, build_kw=dict(seed=seed, values=values, weights=weights, connectivity=connectivity),
               load_kw=_load_kw(exp, dataset),
               train_kw=dict(out_dir=out, resume_from=_resume(out, resume), run_name=out.name, cfg_name=exp.cfg_name, dataset_name=dataset, save_run=False,
                             save_history=save_history))


   ######################################################################################
##########################   Searches   ###################################################
   ######################################################################################

def params_job(p: dict, *, out, seed: int = SEED, n_train: int = 3000, n_test: int = 500, train: bool = True, values: dict | None = None,
               every: int | None = None) -> Job:
    """Search values -> the Job. Every key is optional (missing = the files' value):
        th_L1, th_L2, th_L3   threshold of the exc group of each layer            beta       escape-noise width (fraction of the threshold gap)
        th_inh                threshold of the inh groups (cfg v_th)
        w_ie                  inhibition inh -> exc of every layer (stored as -w_ie)   w_fb   feedback inhibition L2 -> L1 inh, L3 -> L2 inh (0 = off)
        A_ltp, ltd_ratio      STDP rate, and A_ltd = ltd_ratio * A_ltp            reward     x the four R-STDP rates of the cfg file
        normalize             per-neuron normalisation on / off
    train False = no training pass at all (the untrained network's responses only). every = snapshots (None = none: an Optuna trial that is cut
    off is not continued, a new trial is drawn); with every, a run found in `out` is continued."""
    base = ExperimentCfg.load_file(CFG_FILE)
    th, pl = dict(base.dynamics.threshold.group_v_th), base.dynamics.plasticity
    th.update({g: p[k] for g, k in zip(EXC, ("th_L1", "th_L2", "th_L3")) if k in p})
    v = {"dynamics.threshold.group_v_th": th, "run.protocol.trajectory_every": None}
    for key, path in (("beta", "dynamics.threshold.beta"), ("th_inh", "dynamics.threshold.v_th"), ("A_ltp", "dynamics.plasticity.A_ltp"),
                      ("normalize", "dynamics.plasticity.normalize")):
        if key in p:
            v[path] = p[key]
    if "ltd_ratio" in p:
        v["dynamics.plasticity.A_ltd"] = p["ltd_ratio"] * p.get("A_ltp", pl.A_ltp)
    if "reward" in p:
        v.update({f"dynamics.plasticity.{n}": p["reward"] * getattr(pl, n) for n in ("Ar_ltp", "Ar_ltd", "Ap_ltp", "Ap_ltd")})
    if not train:
        v.update({"run.protocol.stage_epochs": [0, 0], "run.protocol.rstdp_epochs": 0})
    weights = {}
    if "w_ie" in p:
        weights.update({f"L{k}_ie": -abs(p["w_ie"]) for k in (1, 2, 3)})
    if "w_fb" in p:
        weights.update({"fb_L2_L1": p["w_fb"], "fb_L3_L2": p["w_fb"]})
    return job(values={**v, **(values or {})}, weights=weights or None, seed=seed, n_train=n_train, n_test=n_test, every=every,
               resume="auto" if every else None, out=out)


def _trial_out(trial, batch: str = "") -> Path:
    """The trial's run folder: runs/optuna/<study>/[<batch>/]trial_<number>. The batch is also a trial attribute (trials.csv user_attrs_batch)."""
    trial.set_user_attr("batch", batch)
    return RUNS_DIR / "optuna" / trial.study.study_name / batch / f"trial_{trial.number:04d}"


def space_rates(trial, batch: str = "") -> Job:
    """Optuna space 1, NO training: how the network as built fires. Thresholds (exc per layer, inh), noise, inhibition and feedback inhibition (the
    last three include 0);
    600 training samples (100 of them validation) + 300 test samples, a minute or two per trial.
    sweep(study, space_rates, metric="val_ami"): val_ami = the decision layer's validation AMI; every rate and flag is a trial attribute."""
    p = {"th_L1": trial.suggest_float("th_L1", 2.0, 60.0, log=True), "th_L2": trial.suggest_float("th_L2", 2.0, 60.0, log=True),
         "th_L3": trial.suggest_float("th_L3", 2.0, 60.0, log=True), "th_inh": trial.suggest_float("th_inh", 2.0, 40.0, log=True),
         "beta": trial.suggest_categorical("beta", [0.0, 0.05, 0.1, 0.15, 0.2]),
         "w_ie": trial.suggest_categorical("w_ie", [0.0, 0.05, 0.25, 1.0]), "w_fb": trial.suggest_categorical("w_fb", [0.0, 0.5, 2.0])}
    return params_job(p, out=_trial_out(trial, batch), n_train=600, n_test=300, train=False)


def space_learn(trial, batch: str = "") -> Job:
    """Optuna space 2, the WHOLE pipeline on 3000 training samples (500 of them validation) + 500 test samples: 1 + 1 STDP epochs, 3 R-STDP passes.
    sweep(study, space_learn, metric="val_ami")           the primary measure (decision layer AMI on validation)
    sweep(study, space_learn, metric="val_decision_acc", pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
          report=("rstdp", "val_decision_acc"))            the R-STDP decision; one number per R-STDP pass for the pruner"""
    p = {"th_L1": trial.suggest_float("th_L1", 3.0, 40.0, log=True), "th_L2": trial.suggest_float("th_L2", 3.0, 40.0, log=True),
         "th_L3": trial.suggest_float("th_L3", 3.0, 40.0, log=True), "th_inh": trial.suggest_float("th_inh", 2.0, 40.0, log=True),
         "beta": trial.suggest_categorical("beta", [0.0, 0.05, 0.1, 0.15, 0.2]),
         "w_ie": trial.suggest_categorical("w_ie", [0.0, 0.05, 0.25, 1.0]), "w_fb": trial.suggest_categorical("w_fb", [0.0, 0.5, 2.0]),
         "A_ltp": trial.suggest_float("A_ltp", 0.0005, 0.05, log=True), "ltd_ratio": trial.suggest_float("ltd_ratio", 0.2, 1.5),
         "reward": trial.suggest_float("reward", 0.1, 30.0, log=True), "normalize": trial.suggest_categorical("normalize", [True, False])}
    return params_job(p, out=_trial_out(trial, batch))


def sweep_batch(study: str, space, batch: str, *, hours: float, max_trials: int = 100000, **sweep_kw):
    """ONE QUEUE JOB's share of an Optuna study: new trials of `study`, one at a time, until `hours` have passed (the trial running then is
    finished) or this batch has max_trials finished trials. Every batch of a study shares its study.db (the sampler learns from all of them);
    the batch's run folders and its own trials.csv (its rows only) are in runs/optuna/<study>/<batch>/, which is all the job has to collect.
    space = space_rates / space_learn; sweep_kw = optuna_sweep.sweep's (metric, pruner, report, ...). Called again (a retry) it continues.
    Five failed trials in a row stop the batch with an error (the job then shows as failed instead of failing trials for hours)."""
    import optuna_sweep as o
    from optuna.trial import TrialState
    ended = (TrialState.COMPLETE, TrialState.PRUNED)
    mine = lambda st: sum(t.user_attrs.get("batch") == batch for t in st.get_trials(states=ended))
    sp, t0 = partial(space, batch=batch), time.time()
    with _one_at_a_time(RUNS_DIR / "optuna" / study / "opening", stale_s=300):       ##batches started together: one at a time creates / opens study.db
        st = o.sweep(study, sp, n_trials=0, **sweep_kw)
    while time.time() - t0 < hours * 3600 and mine(st) < max_trials:
        st = o.sweep(study, sp, n_trials=len(st.get_trials(states=ended)) + 1, **sweep_kw)
        last = [t.state for t in st.get_trials(deepcopy=False) if t.user_attrs.get("batch") == batch][-5:]
        if len(last) == 5 and all(state == TrialState.FAIL for state in last):
            raise RuntimeError(f"study {study} batch {batch}: 5 trials in a row failed (their errors are printed above)")
    df = st.trials_dataframe()
    folder = RUNS_DIR / "optuna" / study / batch
    folder.mkdir(parents=True, exist_ok=True)
    if "user_attrs_batch" in df:
        df[df["user_attrs_batch"] == batch].to_csv(folder / "trials.csv", index=False)
    print(f"study {study} batch {batch}: {mine(st)} finished trials of {len(st.get_trials(states=ended))} in the study, {(time.time() - t0) / 3600:.2f} h", flush=True)
    return st


def replicate(study: str, *, top: int = 3, seeds=(1, 2, 3, 4, 5), n_train: int = 6000, n_test: int = 1000, workers: int = 1, untrained: bool = True) -> list:
    """The `top` best finished trials of an Optuna study (runs/optuna/<study>/study.db) again with other seeds and more samples, each also once
    untrained per seed: how much the result moves between seeds. Run folders runs/optuna/<study>/replicate/top<rank>_s<seed>[_untrained];
    a finished run is skipped and a cut-off run continued when called again. Also written there: chosen.json (the trials taken), the whole
    study's trials.csv, a copy of study.db, and replicate.csv (one row per run: rank, seed, trained, the plain numbers of its results).
    Returns the parallel Results."""
    import json
    import optuna
    import pandas as pd
    folder = RUNS_DIR / "optuna" / study
    rep = folder / "replicate"
    rep.mkdir(parents=True, exist_ok=True)
    st = optuna.load_study(study_name=study, storage=f"sqlite:///{(folder / 'study.db').as_posix()}")
    chosen = rep / "chosen.json"
    if not chosen.exists():             ##the trials are chosen once: a second call (a retry, more seeds) keeps them although the study went on
        done = sorted((t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None), key=lambda t: t.value,
                      reverse=st.direction == optuna.study.StudyDirection.MAXIMIZE)
        chosen.write_text(json.dumps([{"rank": k, "trial": t.number, "value": t.value, "params": t.params} for k, t in enumerate(done[:top])], indent=1))
    picks = json.loads(chosen.read_text())
    st.trials_dataframe().to_csv(rep / "study_trials.csv", index=False)
    shutil.copy2(folder / "study.db", rep / "study.db")
    runs, jobs = [], []
    for pick in picks:
        for seed in seeds:
            for train in (True, False) if untrained else (True,):
                out = rep / f"top{pick['rank']}_s{seed}{'' if train else '_untrained'}"
                runs.append((pick, seed, train, out))
                if not (out / "results.json").exists():
                    jobs.append(params_job(pick["params"], out=out, seed=seed, n_train=n_train, n_test=n_test, train=train, every=2000,
                                           values={"run.protocol.trajectory_every": 1000} if train else None))
    results = run_parallel(jobs, workers=workers)
    rows = []
    for pick, seed, train, out in runs:
        if (out / "results.json").exists():
            r = json.loads((out / "results.json").read_text())
            rows.append({"rank": pick["rank"], "trial": pick["trial"], "seed": seed, "trained": int(train),
                         **{k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}})
    pd.DataFrame(rows).to_csv(rep / "replicate.csv", index=False)
    return results


def main(*, dataset: str = "nmnist", seed: int = SEED, n_train: int | None = None, n_test: int | None = None, values: dict | None = None,
         weights: dict | None = None, connectivity: dict | None = None, every: int | None = None, resume: str | None = None, tag: str = "",
         out: str | None = None, save_history: bool = False):
    """ONE run in this process. Returns the Run: run.report = the protocol's results (also <out>/results.json, run.jsonl, results.jsonl, status.json).
    save_history = write the History (spike raster, if run.history.record_spikes) into the run folder, for plots.raster / plots.all_plots."""
    pnet = build(seed=seed, values=_values(values, n_train, n_test, every), weights=weights, connectivity=connectivity)
    train, test = load(**_load_kw(ExperimentCfg(dynamics=pnet.cfg, run=pnet.runcfg), dataset))
    out = Path(out) if out else _out(dataset, seed, tag)
    return pnet.train(train, test_data=test, out_dir=out, resume_from=_resume(out, resume), run_name=out.name, cfg_name=pnet.cfg_name, dataset_name=dataset, save_run=False,
                      save_history=save_history)
