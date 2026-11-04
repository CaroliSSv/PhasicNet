#microcol_hlsp_main - RUN the layered microcolumn experiment with HLSP values (variant A). Execution only; the experiment is the four other files:
#   microcol_hlsp_cfg.yaml       ExperimentCfg = dynamics (HLSP Tables 2-3) + run (data, checkpoint, LayerwiseCfg protocol)
#   microcol_hlsp_network.yaml   NetSpec: groups, column grids, the "column" kernel per synapse group, initial weights
#   experiment_dynamics/microcol_hlsp.py   the runtime modules (hlsp2026.py unchanged)
#   microcol_hlsp_protocol.py    layer-wise stages -> val / test responses -> ARI / AMI / vote per layer + the input control
#
#RUN (repo root, conda activate PhasicNet):
#   python -c "import experiments, microcol_hlsp_main as m; m.main(n_train=2000, n_test=500, tag='t1')"
#   values = any cfg field by its dotted path:  m.main(values={"dynamics.plasticity.A_ltp": 0.1, "run.protocol.stage_epochs": [1, 1, 2]})
#   weights / connectivity = builder.set_weights / set_connectivity on top of the network file:  m.main(weights={"in_L1e": {"values": [-3.0, 3.0]}})
#   resume="auto" continues the newest snapshot in the run folder (needs every=N or run.checkpoint.every)
#SEARCHES: job(...) -> a parallel.Job (parallel.run_parallel, the queue); space(trial) -> the Optuna example for optuna_sweep.sweep (guide: docs).
#DATA: mnist | fmnist through hlsp_lsm_2026.load_dataset (HLSP's own Poisson stimulus, 250 steps per sample); the network file is 28 x 28 inputs.
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

from phasicnet import PhasicNet
from runcfg import ExperimentCfg
from parallel import Job
from paths import RUNS_DIR
from checkpoint import snapshots
import hlsp_lsm_2026 as hlsp
from microcol_hlsp import MICROCOL_HLSP_SPARSE, MICROCOL_HLSP_DENSE
from microcol_hlsp_protocol import LayerwiseProtocol

HERE = Path(__file__).resolve().parent
CFG_FILE, NET_FILE = HERE / "microcol_hlsp_cfg.yaml", HERE / "microcol_hlsp_network.yaml"
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
    pnet.builder.realize(network_name="microcol_hlsp", post_offset=True, allow_autapse=False)
    pack = MICROCOL_HLSP_SPARSE if pnet.cfg.simulation.sparse else MICROCOL_HLSP_DENSE
    pnet.set_runtime_classes(replace(pack, protocol_classes=(LayerwiseProtocol,)))
    return pnet


def load(*, dataset: str = "mnist", seed: int = SEED):
    """(train, test) ModelInputs: HLSP's stimulus unchanged (Poisson, PEAK_HZ, the authors' image scaling)."""
    train, test, _ = hlsp.load_dataset(dataset, seed=seed)
    return train, test


def _out(dataset, seed, tag) -> Path:
    return RUNS_DIR / f"microcol_hlsp_{dataset}_s{seed}{('_' + tag) if tag else ''}"


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
               load_kw=dict(dataset=dataset, seed=seed),
               train_kw=dict(out_dir=out, resume_from=_resume(out, resume), run_name=out.name, cfg_name=exp.cfg_name, dataset_name=dataset, save_run=False,
                         save_history=save_history))


def space(trial) -> Job:
    """EXAMPLE Optuna space for optuna_sweep.sweep(study, space, metric="val_ami"): the input weight and the E-/I-STDP intensities, 5000 samples."""
    w_in = trial.suggest_float("w_in", 1.0, 6.0)
    a_e = trial.suggest_float("A_E", 0.02, 0.3, log=True)
    a_i = trial.suggest_float("A_I", 0.02, 0.3, log=True)
    values = {"dynamics.plasticity.A_ltp": a_e, "dynamics.plasticity.A_ltd": a_e, "dynamics.plasticity.A_I": a_i}
    return job(values=values, weights={"in_L1e": {"values": [-w_in, w_in]}, "in_L1i": {"values": [-w_in, w_in]}}, n_train=5000, n_test=1000,
               out=RUNS_DIR / "optuna" / trial.study.study_name / f"trial_{trial.number:04d}")


def main(*, dataset: str = "mnist", seed: int = SEED, n_train: int | None = None, n_test: int | None = None, values: dict | None = None,
         weights: dict | None = None, connectivity: dict | None = None, every: int | None = None, resume: str | None = None, tag: str = "",
         out: str | None = None, save_history: bool = False):
    """ONE run in this process. Returns the Run: run.report = the protocol's results (also <out>/results.json, run.jsonl, results.jsonl, status.json).
    save_history = write the History (spike raster, if run.history.record_spikes) into the run folder, for plots.raster / plots.all_plots."""
    pnet = build(seed=seed, values=_values(values, n_train, n_test, every), weights=weights, connectivity=connectivity)
    train, test = load(dataset=dataset, seed=seed)
    out = Path(out) if out else _out(dataset, seed, tag)
    return pnet.train(train, test_data=test, out_dir=out, resume_from=_resume(out, resume), run_name=out.name, cfg_name=pnet.cfg_name, dataset_name=dataset, save_run=False,
                      save_history=save_history)
