#dc_reproduce - Diehl & Cook (2015) PROTOCOL: train, label neurons, test, report ACCURACY -- the paper's own evaluation, so the number is comparable
# to the paper. The network and physics come unchanged from diehl_cook_2015.build() + diehlcook.py; every step below follows the authors' Brian 1
# script (Diehl&Cook_spiking_MNIST.py) and their evaluation script line by line:
#
#   TRAIN      MNIST training set in its own order (example j = image j % 60000), one fresh network (diehl_cook_2015.fresh_network: V = v_rest - 40 mV).
#              Each presentation: normalise the input weights (diehlcook.dc_normalise -- before EVERY presentation, re-presentations included),
#              350 ms Poisson at pixel/8*intensity Hz (intensity 2 -> max 63.75 Hz), then 150 ms silence.
#              The script's count for a presentation = exc spikes since the END of the previous presentation (its rest period + this presentation).
#              If that count is < 5: intensity + 1 and the example is shown again; the stored response is the count of the presentation that reached 5.
#   ASSIGN     a NEW network as the script's test mode builds it: learning off, theta loaded and fixed, V = v_rest - 40 mV, its delays re-drawn from
#              the training stream over the trained W's nonzero weights (dc_input_delays); the
#              first n_assign training images (default 60,000 = the paper/readme: "one presentation of the training set"), same
#              presentation rule. Labels: the evaluation script's get_new_assignments (scorer.dc_assign_labels, -1 = never fired).
#   TEST       a NEW network again (same delays as ASSIGN: the script re-seeds np.random.seed(0) and draws them the same way), the official test set,
#              test_passes consecutive passes in ONE run (default 10: "averaged over ten presentations"). Prediction: the script's
#              get_recognized_number_ranking (scorer.dc_rank_predict); accuracy per pass, mean and np.std as the evaluation script prints them.
#   NOTE       the PUBLISHED scripts cannot run ASSIGN on 60,000 images or 10 test passes as they stand; the 60,000 / 10-pass protocol here
#              follows the paper's text, and how the 10 passes were run (one run, as here, or ten runs) cannot be checked.
#   TRAIN ACC  the training script's own running performance (Fig. 2C): at every 10,000th example the labels are recomputed from the previous
#              10,000 responses (unassigned -> class 0, as in the training script), each example is predicted with the labels of its block.
#
#The steps are dc_protocol.DCProtocol. This file is the CLI around it: settings -> DCProtocolCfg, MNIST -> DCPoissonImages, protocol.json,
# ONE PhasicNet.train(...) call.
#
#OUTPUT goes to runs/DC_<tag>/ and NOTHING to stdout: run.jsonl (progress) + results.jsonl (the numbers), network/cfg/cfg_derived/runcfg/runmeta,
# protocol.json (settings + git state), snap_*.npz, TrainedModel_<run_id>.npz/.json, results.json, responses.npz (exc spike counts + labels per
# example). analysis.protocol_scores(run_dir) recomputes the accuracy from those files; load_trained(run_dir) rebuilds the trained network.
#   Watch progress (PowerShell):  Get-Content runs\DC_<tag>\run.jsonl -Wait -Tail 5
#
#LONG RUNS: ~0.5 s per example on a laptop CPU; the full protocol takes roughly a day (n_assign=10000, test_passes=1 gives a quick, non-paper
# evaluation). A snapshot is written every save_every examples; from_snapshot= continues from one exactly (inputs are keyed per example).
#
#Run (from the repo root, PowerShell):  python -c "import experiments, dc_reproduce; dc_reproduce.main(tag='e100_bench', n_train=1000, n_assign=1000, n_test=1000, test_passes=1)"
#                                      main(tag="e100_40k")                                                   (the paper's 100-neuron experiment)
from __future__ import annotations
import logging, subprocess
from pathlib import Path
import numpy as np
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg
from paths import RAW_DIR, RUNS_DIR, ROOT_DIR
from preprocessing import load_mnist
from checkpoint import load_trained_model, restore_snapshot
from utils import _write_json_yaml, _read_json_yaml
from modelinput import ModelInput
from diehl_cook_2015 import build, fresh_network, DT_MS, T_PRESENT_MS, T_REST_MS, INPUT_INTENSITY, SEED, DCPoissonImages, mnist_input
from dc_protocol import DCProtocol, DCProtocolCfg                   ##the protocol itself (TRAIN / ASSIGN / TEST / REPORT)

logger = logging.getLogger("dc_reproduce")


   ######################################################################################
##########################   Code state   #################################################
   ######################################################################################

def git_state() -> dict:
    """Which code produced this result: git_commit (hash of HEAD) and git_status ("dirty" = uncommitted edits at start, "clean").
    Written flat into protocol.json."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, capture_output=True, text=True, timeout=10).stdout.strip())
        return {"git_commit": head or None, "git_status": "dirty" if dirty else "clean"}
    except Exception:
        return {"git_commit": None, "git_status": None}


   ######################################################################################
##########################   Stimulus (user-side preprocessing)   ##########################
   ######################################################################################

def mnist_dataset(*, npz=RAW_DIR / "mnist.npz", seed: int = SEED, **kw) -> tuple[ModelInput, ModelInput]:
    """(train, test) = MNIST's official split as two re-presenting ModelInputs (test keyed seed + 1) -- what main() runs on and
    what a parallel.Job(load=mnist_dataset, load_kw={...}) child loads. kw = present_ms / rest_ms (Zhang's shorter window)."""
    x_tr, y_tr, x_te, y_te = load_mnist(Path(npz))
    return mnist_input(x_tr, y_tr, seed=seed, **kw), mnist_input(x_te, y_te, seed=seed + 1, **kw)


   ######################################################################################
##########################   Run settings + log file   ####################################
   ######################################################################################

def _runcfg(*, save_every: int | None = None, n_test: int | None = None, **proto) -> RunCfg:
    """The RunCfg every protocol run uses: no test split inside Run (the protocol uses the official MNIST test set), no spike raster (responses come
    from history.Counts), no automatic TrainedModel writes on improvement (the protocol saves the final model itself).
    proto = DCProtocolCfg fields (n_train, n_assign, ...); nothing given (load_trained) = the protocol's defaults and no snapshots.
    n_test -> DataCfg.n_test (a stratified pick; None = the whole official test set)."""
    return RunCfg(data=DataCfg(test_fraction=0.0, n_test=n_test), checkpoint=CheckpointCfg(to_disk=False, every=save_every),
                  scorer=ScorerCfg(rate_hz_min=0.0), history=HistoryCfg(record_spikes=False), protocol=DCProtocolCfg(**proto))


def load_trained(run_dir: str | Path, *, freeze_theta: bool = True):
    """Rebuild a finished protocol run's network and load its trained model (TrainedModel_*.npz) -> the Run, ready for new
    experiments (e.g. a pruning phase). Same build() + seed, so the network is identical; checked against the saved network.json id.
    Theta is frozen by default (the paper's test state)."""
    run_dir = Path(run_dir)
    proto = _read_json_yaml(run_dir / "protocol.json")
    pnet = build(784, proto["n_exc"], seed=proto["seed"], norm_target=78.0)
    run = pnet._prep_run(mnist_input(np.zeros((4, 28, 28), np.uint8), np.arange(4) % 2), _runcfg())   ##four blank digits: the Run only needs data to assemble
    saved_id = _read_json_yaml(run_dir / "runmeta.json")["network_id"]
    if run.runmeta.network_id != saved_id:
        raise RuntimeError(f"rebuilt network {run.runmeta.network_id} != saved {saved_id}: build() or its defaults changed since the run")
    restore_snapshot(run.ctx, load_trained_model(next(run_dir.glob("TrainedModel_*.npz"))), run.modules)
    run.threshold.freeze(freeze_theta)
    fresh_network(run, redraw_delays=True)        ##the state the script's test mode starts from (new network, its own delays)
    return run


   ######################################################################################
##########################   The run   ####################################################
   ######################################################################################

def main(*, tag: str, n_exc: int = 100, n_train: int = 40_000, n_assign: int = 60_000, n_test: int | None = None, test_passes: int = 10, save_every: int = 5_000,
         report_every: int = 500, max_repeats: int = 100, seed: int = 7, mnist=RAW_DIR / "mnist.npz", from_snapshot: str | None = None):
    """ONE protocol run into runs/DC_<tag>/ (a NEW folder, unless from_snapshot continues one). n_train: paper 40,000 for 100 exc neurons;
    n_assign 60,000 and test_passes 10 = the paper's protocol; n_test: None = the whole official test set (the paper's 10,000), a smaller number
    is a stratified pick; max_repeats = the cap on re-presentations, the paper has none; save_every = snapshot interval.
    Returns run.report (= DCProtocol's results, also results.json). Nothing goes to stdout."""
    out = RUNS_DIR / f"DC_{tag}"
    if out.exists() and from_snapshot is None:
        raise FileExistsError(f"{out} exists. Use a new tag, or from_snapshot= to continue.")
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(tag=tag, n_exc=n_exc, n_train=n_train, n_assign=n_assign, n_test=n_test, test_passes=test_passes, save_every=save_every,
                    report_every=report_every, max_repeats=max_repeats, seed=seed, mnist=str(mnist), from_snapshot=from_snapshot)
    log = logger.info
    log("=" * 72); log(f"DIEHL & COOK PROTOCOL  tag={tag}  settings={settings}")
    code = git_state()
    log(f"git_commit={code['git_commit']} git_status={code['git_status']}  (all modules already imported: editing source files now does not affect this run)")

    data, test = mnist_dataset(npz=mnist, seed=seed)

    ##build exactly as diehl_cook_2015.build(); the protocol class is registered so RunCfg.protocol (DCProtocolCfg, "DC2015") can name it
    pnet = build(784, n_exc, seed=seed, norm_target=78.0)
    pnet.set_runtime_classes(protocol_classes=(DCProtocol,))
    rc = _runcfg(save_every=save_every, n_test=n_test, n_train=n_train, n_assign=n_assign, test_passes=test_passes, max_repeats=max_repeats, report_every=report_every)
    _write_json_yaml(out / "protocol", {**settings, "dt_ms": DT_MS, "present_ms": T_PRESENT_MS, "rest_ms": T_REST_MS, "input_intensity": INPUT_INTENSITY,
                                        "min_spikes": rc.protocol.min_spikes, "block": rc.protocol.block, **code}, JSON=True, YAML=False, log_label="Protocol")

    ##ONE call: _prep_run (no split) -> network/cfg/cfg_derived/runcfg/runmeta into `out` -> resume if asked -> DCProtocol.execute.
    ## save_run=False: the protocol writes the TrainedModel itself right after TRAIN.
    run = pnet.train(data, rc, cfg_name="diehl_cook_2015_cfg", run_name=f"DC_{tag}", dataset_name="mnist_train",
                     test_data=test, out_dir=out, resume_from=from_snapshot, save_run=False, save_history=False)
    return run.report


if __name__ == "__main__":
    main(tag="e100_40k")
