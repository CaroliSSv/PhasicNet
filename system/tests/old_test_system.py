#test_system -- the invariants whose failure is SILENT. Every test here guards something that produces a plausible
# number when broken: leakage, wrong aggregation, non-independent fits, unrecorded clamps, identity fragmentation,
# and dense/sparse/event disagreement. Replaces the seven pre-refactor suites, which tested the old API.
# Run: pytest tests/test_system.py -q     (from the repo root)
from __future__ import annotations
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main"))
import numpy as np
import pytest

from smoke_test import build, make_data
from runcfg import RunCfg, DataCfg, CheckpointCfg
from system.dynamicscfg import SimulationCfg
from rnghandler import RNGHandler
from paths import TRAIN_TEST_DIR
from run import Run


def fresh_split():
    shutil.rmtree(TRAIN_TEST_DIR, ignore_errors=True); TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)

def trained(fmt="dense", update="timestep", **rc):
    fresh_split()
    cfg = dict(epochs=2, k_fold=2, patience=3, data=DataCfg(test_fraction=0.25)); cfg.update(rc)
    return build(fmt, update).train(make_data(), RunCfg(**cfg), save_run=False)


# ------------------------------------------------------------------ equivalence
def test_sparse_timestep_equals_sparse_event_bit_for_bit():
    """Event mode skips only edges whose dw is EXACTLY zero, so it must be the identical simulation."""
    a, b = trained("sparse", "timestep"), trained("sparse", "event")
    assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W)
    assert np.array_equal(a.history.spike_i(), b.history.spike_i())

def test_dense_and_sparse_agree_on_a_short_run():
    """Short run: BLAS float32 and bincount float64 round identically. Long runs may drift; that is float order,
    not a bug -- so this is pinned on a SHORT run only."""
    a, b = trained("dense"), trained("sparse")
    assert a.report["score"] == pytest.approx(b.report["score"], rel=1e-5)

def test_dense_plus_event_is_rejected_not_ignored():
    with pytest.raises(ValueError):
        SimulationCfg(weights_format="dense", update_on="event")


# ------------------------------------------------------------------ leakage & ownership
def test_test_pass_does_not_train():
    run = trained()
    W = run.ctx.syn.W.copy(); traces = run.plasticity.Pre_trace.copy()
    run.test()
    assert np.array_equal(W, run.ctx.syn.W) and np.array_equal(traces, run.plasticity.Pre_trace)

def test_train_and_test_never_overlap():
    run = trained()
    assert not (set(run.train_set.index.tolist()) & set(run.test_set.index.tolist()))

def test_phasicnet_retains_no_run_state():
    """The assembler is stateless w.r.t. runs: cleanup-by-design."""
    fresh_split(); p = build(); run = p.train(make_data(), RunCfg(epochs=1, k_fold=2, data=DataCfg(test_fraction=0.25)), save_run=False)
    for attr in ("ctx", "history", "trial", "run", "synapse", "plasticity"):
        assert getattr(p, attr, None) is None, f"PhasicNet kept .{attr}"
    assert run.ctx is not None


# ------------------------------------------------------------------ aggregation & fits
def test_trial_score_is_mean_over_fits_not_max():
    r = trained(k_fold=3).report
    assert len(r["fit_scores"]) == 3
    assert r["score"] == pytest.approx(float(np.mean(r["fit_scores"])))
    if max(r["fit_scores"]) != min(r["fit_scores"]):
        assert r["score"] < max(r["fit_scores"])

def test_each_fit_starts_from_the_as_built_model():
    """reset_model restores W AND every module's state_dict -- fits must be independent."""
    run = trained()
    W0 = run._W0.copy(); tr0 = run._init_state["plasticity"]["Pre_trace"].copy()
    assert not np.array_equal(W0, run.ctx.syn.W)
    run.reset_model()
    assert np.array_equal(W0, run.ctx.syn.W) and np.array_equal(tr0, run.plasticity.Pre_trace)

def test_fold_split_is_keyed_on_dataset_not_run():
    """Two runs differing only in hyperparameters must see the SAME folds, or Optuna trials are incomparable."""
    a = trained(); b = trained(epochs=1)
    fa = a.datahandler.fold_split(a.train_set, k_fold=2, rng=a.ctx.rng.rng("data", f"folds:{a.runmeta.dataset_id}"))
    fb = b.datahandler.fold_split(b.train_set, k_fold=2, rng=b.ctx.rng.rng("data", f"folds:{b.runmeta.dataset_id}"))
    assert all(np.array_equal(x[1].index, y[1].index) for x, y in zip(fa, fb))


# ------------------------------------------------------------------ derived cfg & identity
def test_derived_cfg_records_the_safety_clamp():
    run = trained()
    assert run.history.derived_cfg.plasticity.w_min == -np.inf          ##entered None, used -inf
    assert run.runmeta.cfg_id != run.runmeta.derived_cfg_id

def test_naming_contract_resolves_time_constants():
    """Modules store each consumed cfg field under its OWN name, so the derived cfg finds tau_mem/tau_th."""
    run = trained()
    assert run.history.derived_cfg.integrator.tau_mem == 20.0
    assert run.history.derived_cfg.threshold.tau_th == 30.0

def test_hash_exclusions_do_not_fragment_identity():
    base = RunCfg(epochs=2, k_fold=2)
    assert base.hashable() == RunCfg(epochs=2, k_fold=2, progress=True).hashable()
    assert base.hashable() == RunCfg(epochs=2, k_fold=2, checkpoint=CheckpointCfg(to_disk=False)).hashable()
    assert base.hashable() != RunCfg(epochs=3, k_fold=2).hashable()
    assert "progress" not in base.hashable() and "to_disk" not in base.hashable()["checkpoint"]

def test_no_names_in_any_hash():
    """Renaming a run must not change its identity."""
    fresh_split(); p = build(); p.run_name = "A"
    a = p.train(make_data(), RunCfg(epochs=1, data=DataCfg(test_fraction=0.25)), save_run=False).runmeta
    fresh_split(); p = build(); p.run_name = "B"
    b = p.train(make_data(), RunCfg(epochs=1, data=DataCfg(test_fraction=0.25)), save_run=False).runmeta
    assert a.run_id == b.run_id and a.cfg_id == b.cfg_id


# ------------------------------------------------------------------ rng & modules
def test_rng_is_keyed_and_order_independent():
    r = RNGHandler(7)
    assert r.rng("fit", 2).random() == RNGHandler(7).rng("fit", 2).random()
    assert r.rng("fit", 1).random() != r.rng("fit", 2).random()
    assert RNGHandler.from_rng_spec(r.rng_spec).seed == 7

def test_params_and_state_dict_split_by_naming_convention():
    run = trained()
    p, sd = run.plasticity.params, run.plasticity.state_dict
    assert "tau_pretrace" in p and "A_ltp" in p and "Pre_trace" not in p        ##lowercase static -> params
    assert set(sd) == {"Pre_trace", "Post_trace"} and "_ltd" not in sd            ##Capitalised arrays -> state_dict
    assert run.threshold.state_dict == {} or "Refrac_count" in run.threshold.state_dict
