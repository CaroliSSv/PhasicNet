#microcol_mozafari_main - RUN the layered microcolumn experiment with Mozafari et al. 2019's values (variant B). Execution only; the experiment is:
#   microcol_mozafari_cfg.yaml       ExperimentCfg = dynamics (paper Tables 1-3 + MozafariDeep.py) + run (data, checkpoint, MozafariCfg protocol)
#   microcol_mozafari_network.yaml   NetSpec: input (32, 32, 6) -> L1 (14, 14, 30) -> L2 (4, 4, 250) -> L3 (1, 1, 200), "column" kernels, N(0.8, 0.05)-like start
#   experiment_dynamics/microcol_mozafari.py   the runtime modules (IF neurons, per-layer thresholds + column competition, STDP / R-STDP)
#   microcol_mozafari_protocol.py    STDP layer by layer -> R-STDP decision layer -> decision accuracy + variant A's readout-free scores
#
#RUN (repo root, conda activate PhasicNet):
#   python -c "import experiments, microcol_mozafari_main as m; m.main(n_train=3000, n_test=500, tag='t1')"
#   values = any cfg field by its dotted path:  m.main(values={"run.protocol.rstdp_epochs": 5, "dynamics.threshold.group_v_th": {"L1_exc": 12.0, "L2_exc": 10.0, "L3_exc": None}})
#   weights / connectivity = builder.set_weights / set_connectivity on top of the network file:  m.main(weights={"L1_L2": (0.7, 0.9)})
#   resume="auto" continues the newest snapshot in the run folder (needs every=N or run.checkpoint.every)
#SEARCHES: job(...) -> a parallel.Job (parallel.run_parallel, the queue); space(trial) -> the Optuna example for optuna_sweep.sweep (guide: docs).
#DATA: mnist | fmnist, preprocessing.DoGLatencyDataset (the authors' DoG filters, local normalisation, 15 latency bins, 2 px border) + 2 silent
# steps: 17 steps per sample, one step per bin. The authors trained on all 60k and picked the best R-STDP epoch on the test set; here 1/6 of the
# training set is the validation block (val_split) and the test set is scored once.
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

from phasicnet import PhasicNet
from runcfg import ExperimentCfg
from parallel import Job
from paths import RUNS_DIR, RAW_DIR
from checkpoint import snapshots
from modelinput import ModelInput
from preprocessing import load_mnist, DoGLatencyDataset
from microcol_mozafari import MICROCOL_MOZAFARI
from microcol_mozafari_protocol import MozafariProtocol

HERE = Path(__file__).resolve().parent
CFG_FILE, NET_FILE = HERE / "microcol_mozafari_cfg.yaml", HERE / "microcol_mozafari_network.yaml"
SEED = 1


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
    pnet.builder.realize(network_name="microcol_mozafari", post_offset=True, allow_autapse=False)
    pnet.set_runtime_classes(replace(MICROCOL_MOZAFARI, protocol_classes=(MozafariProtocol,)))
    return pnet


def load(*, dataset: str = "mnist", **encoder):
    """(train, test) ModelInputs: RAW_DIR/<dataset>.npz (0..255 images) through the authors' encoder; encoder = DoGLatencyDataset keywords."""
    x, y, xt, yt = load_mnist(RAW_DIR / f"{dataset}.npz")
    return (ModelInput(DoGLatencyDataset(x, **encoder), y, axis_labels="TX"),     ##deterministic: no seed
            ModelInput(DoGLatencyDataset(xt, **encoder), yt, axis_labels="TX"))


def _out(dataset, seed, tag) -> Path:
    return RUNS_DIR / f"microcol_mozafari_{dataset}_s{seed}{('_' + tag) if tag else ''}"


def _resume(out: Path, resume: str | None) -> str | None:
    """resume="auto" -> the newest snapshot in the run folder (None if there is none); any other value is used as is."""
    if resume != "auto":
        return resume
    found = snapshots(out)
    return str(found[-1]) if found else None


def _values(values, n_train, n_test, every) -> dict:
    extra = {"run.data.n_train": n_train, "run.data.n_test": n_test, "run.checkpoint.every": every}
    return {**{k: v for k, v in extra.items() if v is not None}, **(values or {})}


def job(*, values: dict | None = None, weights: dict | None = None, connectivity: dict | None = None, seed: int = SEED, dataset: str = "mnist",
        n_train: int | None = None, n_test: int | None = None, every: int | None = None, resume: str | None = None, tag: str = "",
        out: str | Path | None = None, save_history: bool = False) -> Job:
    """ONE run as a parallel.Job (each in its own process): the parent resolves the RunCfg and the snapshot to resume (resume="auto"), the child
    builds the network and loads the data itself."""
    values = _values(values, n_train, n_test, every)
    out = Path(out) if out else _out(dataset, seed, tag)
    exp = ExperimentCfg.load_file(CFG_FILE).override(values)
    return Job(build=build, load=load, runcfg=exp.run, build_kw=dict(seed=seed, values=values, weights=weights, connectivity=connectivity),
               load_kw=dict(dataset=dataset),
               train_kw=dict(out_dir=out, resume_from=_resume(out, resume), run_name=out.name, cfg_name=exp.cfg_name, dataset_name=dataset, save_run=False,
                         save_history=save_history))


def space(trial) -> Job:
    """EXAMPLE Optuna space for optuna_sweep.sweep(study, space, metric="val_decision_acc"): the L1 / L2 thresholds and the R-STDP reward rate.
    ##optuna pruning: sweep(..., pruner=optuna.pruners.MedianPruner(n_startup_trials=5), report=("rstdp", "val_decision_acc"))  (one number per R-STDP pass)."""
    th1 = trial.suggest_float("th_L1", 8.0, 25.0)
    th2 = trial.suggest_float("th_L2", 5.0, 20.0)
    ar = trial.suggest_float("Ar_ltp", 0.001, 0.02, log=True)
    values = {"dynamics.threshold.group_v_th": {"L1_exc": th1, "L2_exc": th2, "L3_exc": None}, "dynamics.plasticity.Ar_ltp": ar,
              "run.protocol.rstdp_epochs": 3}
    return job(values=values, n_train=6000, n_test=1000, out=RUNS_DIR / "optuna" / trial.study.study_name / f"trial_{trial.number:04d}")


def space_k(trial) -> Job:
    """Optuna space of the FIRST panther sweep: how many winners update per sample (the authors' k 5 / 8 with radius 3 / 1 vs more, up to every
    column: ##authors choice> mozafari is the k = 5 corner) and the two thresholds; 6000 samples, 1 + 1 STDP epochs, 3 R-STDP passes.
    sweep(..., metric="val_decision_acc", pruner=optuna.pruners.MedianPruner(n_startup_trials=6), report=("rstdp", "val_decision_acc"))"""
    k1 = trial.suggest_categorical("k_L1", [5, 20, 60, 196])
    r1 = trial.suggest_categorical("r_L1", [0, 1, 3])
    k2 = trial.suggest_categorical("k_L2", [8, 16])
    th1 = trial.suggest_float("th_L1", 8.0, 25.0)
    th2 = trial.suggest_float("th_L2", 5.0, 20.0)
    values = {"dynamics.plasticity.kwta": {"L1_exc": [k1, r1], "L2_exc": [k2, 0 if k2 == 16 else 1]},
              "dynamics.threshold.group_v_th": {"L1_exc": th1, "L2_exc": th2, "L3_exc": None},
              "run.protocol.stage_epochs": [1, 1], "run.protocol.rstdp_epochs": 3, "run.protocol.trajectory_every": 2500}
    return job(values=values, n_train=6000, n_test=1000, every=2000, resume="auto", out=RUNS_DIR / "optuna" / trial.study.study_name / f"trial_{trial.number:04d}")


def main(*, dataset: str = "mnist", seed: int = SEED, n_train: int | None = None, n_test: int | None = None, values: dict | None = None,
         weights: dict | None = None, connectivity: dict | None = None, every: int | None = None, resume: str | None = None, tag: str = "",
         out: str | None = None, save_history: bool = False):
    """ONE run in this process. Returns the Run: run.report = the protocol's results (also <out>/results.json, run.jsonl, results.jsonl, status.json).
    save_history = write the History (spike raster, if run.history.record_spikes) into the run folder, for plots.raster / plots.all_plots."""
    pnet = build(seed=seed, values=_values(values, n_train, n_test, every), weights=weights, connectivity=connectivity)
    train, test = load(dataset=dataset)
    out = Path(out) if out else _out(dataset, seed, tag)
    return pnet.train(train, test_data=test, out_dir=out, resume_from=_resume(out, resume), run_name=out.name, cfg_name=pnet.cfg_name, dataset_name=dataset, save_run=False,
                      save_history=save_history)
