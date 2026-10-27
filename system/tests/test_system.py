#test_system -- the invariants whose failure is SILENT. Every test here guards something that produces a plausible
# number when broken: leakage, wrong aggregation, non-independent fits, unrecorded clamps, identity fragmentation,
# and dense/sparse/event disagreement.
# Run: pytest system/tests/test_system.py -q     (from the repo root)
from __future__ import annotations
##system/tests/conftest.py sets sys.path for pytest
import numpy as np
from modelinput import ModelInput
import pytest

from phasicnet import PhasicNet
from runcfg import RunCfg, DataCfg, CheckpointCfg, HistoryCfg, ScorerCfg
from test_sparse import protocol_for
from dynamicscfg import DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, NoiseCfg, PhaseCfg
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, AdaptiveThreshold, TraceSTDP, SparseTraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase
from rnghandler import RNGHandler
from run import Run
from paths import RUNS_DIR
from states import RunContext
from history import RunHistory

SEED, N_INPUT, N_EXC, N_INH = 7, 12, 8, 4

##tiny random binary input: 16 samples x 30 ms, 2 classes
def make_data(n=16, T=30, n_input=N_INPUT):
    X = np.random.default_rng(0).random((n * T, n_input)) < 0.3
    return ModelInput.from_tonic([(X[i * T:(i + 1) * T], i % 2) for i in range(n)])

def build(fmt="dense", update="timestep", post_offset=False):
    ##modelled on main.py build(); w_min/w_max stay None (see test_derived_cfg_records_the_safety_clamp)
    pnet = PhasicNet(SEED)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", N_INPUT, True), ("exc", N_EXC), ("inh", N_INH, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc=0.6, exc_exc=0.3, exc_inh=0.4, inh_exc=0.4)
    pnet.builder.set_weights({"in_exc": (0.0, 2.0), "exc_exc": 1.0, "exc_inh": 3.0, "inh_exc": 3.0})
    pnet.builder.spec_name = "system_spec"
    pnet.builder.realize(network_name="system_net", post_offset=post_offset, save=False)
    sparse = fmt == "sparse"
    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on=update),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=10.0),
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.005, A_ltd=0.006, eta_exc=1e-3, eta_inh=1e-3),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=DeltaCurrentSparse if sparse else DeltaCurrentDense, integrator_cls=LIFexact, threshold_cls=AdaptiveThreshold,
                             plasticity_cls=SparseTraceSTDP if sparse else TraceSTDP, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    return pnet


def trained(fmt="dense", update="timestep", *, epochs=2, val_split=0.25, **rc):
    ##HoldoutValidation with val_split=0.25 by default (val_split=None = the plain holdout)
    cfg = dict(protocol=protocol_for(epochs=epochs, val_split=val_split), data=DataCfg(test_fraction=0.25)); cfg.update(rc)
    return build(fmt, update).train(make_data(), RunCfg(**cfg), save_run=False)

##train(save_run=True) writes TrainedModel_<run_id>.npz into the REAL runs/ directory: remove what a test wrote there
@pytest.fixture(autouse=True)
def _no_trained_model_files_left_behind():
    before = set(RUNS_DIR.glob("TrainedModel_*"))
    yield
    for f in set(RUNS_DIR.glob("TrainedModel_*")) - before:
        f.unlink()


# ------------------------------------------------------------------ equivalence
def test_sparse_timestep_equals_sparse_hybrid_event_bit_for_bit():
    """hybrid_event skips only edges whose dw is EXACTLY zero, so it must be the identical simulation."""
    a, b = trained("sparse", "timestep"), trained("sparse", "hybrid_event")
    assert a.history.spike_i().size > 0, "no spikes at all: the comparison proves nothing"
    assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W)
    assert np.array_equal(a.history.spike_i(), b.history.spike_i())

def test_sparse_event_equals_sparse_timestep_to_tolerance():
    """"event" decays per-synapse traces lazily (synapticindex.py): the same simulation up to float rounding."""
    ##the event modules stamp T_e with sim.train_step, which advances in mode "train" only
    a, b = trained("sparse", "timestep"), trained("sparse", "event")
    assert a.history.spike_i().size > 0, "no spikes at all: the comparison proves nothing"
    assert a.ctx.sim.current_runstep > a.ctx.sim.train_step > 0 and b.ctx.sim.train_step == a.ctx.sim.train_step     ##eval steps ran (the val passes) and did not tick the train clock
    assert np.allclose(a.ctx.syn.W, b.ctx.syn.W, rtol=1e-5, atol=1e-6)
    assert np.array_equal(a.history.spike_i(), b.history.spike_i())

def test_dense_and_sparse_agree_on_a_short_run():
    """Dense and sparse agree on a SHORT run (long runs may drift by float summation order, not a bug)."""
    a, b = trained("dense"), trained("sparse")
    assert a.history.spike_i().size > 0, "no spikes at all: the comparison proves nothing"
    assert a.report["score"] == pytest.approx(b.report["score"], rel=1e-5)

def test_dense_plus_event_is_rejected_not_ignored():
    with pytest.raises(ValueError):
        SimulationCfg(weights_format="dense", update_on="event")


# ------------------------------------------------------------------ leakage & ownership
def test_test_pass_does_not_train():
    run = trained()
    W = run.ctx.syn.W.copy(); traces = run.plasticity.Pre_trace.copy()
    assert run.test()["n_test"] > 0                      ##the test pass really ran
    assert np.array_equal(W, run.ctx.syn.W) and np.array_equal(traces, run.plasticity.Pre_trace)

def test_train_and_test_never_overlap():
    run = trained()
    assert not (set(run.train_set.index.tolist()) & set(run.test_set.index.tolist()))


def test_stored_split_is_reused_only_under_the_policy_it_was_made_with(tmp_path):
    """A stored split is reused only under the same test_fraction / stratified; a change raises naming both values and the spec file."""
    from datahandler import DataHandler
    dh, data = DataHandler(), make_data()
    kw = dict(dataset_id="a7", save_dir=tmp_path)
    tr, te = dh.split_test_train(data, data_cfg=DataCfg(test_fraction=0.25, stratify=True), **kw)
    tr2, te2 = dh.split_test_train(data, data_cfg=DataCfg(test_fraction=0.25, stratify=True), **kw)        ##the same policy: reused verbatim
    assert np.array_equal(tr.index, tr2.index) and np.array_equal(te.index, te2.index) and len(te) == 4 and (tmp_path / "split_a7.json").exists()
    with pytest.raises(ValueError, match=r"split_a7\.json.*test_fraction=0\.25.*test_fraction=0\.5") as e:
        dh.split_test_train(data, data_cfg=DataCfg(test_fraction=0.5, stratify=True), **kw)
    assert "delete" in str(e.value).lower()
    with pytest.raises(ValueError, match=r"stratified=True.*stratified=False"):
        dh.split_test_train(data, data_cfg=DataCfg(test_fraction=0.25, stratify=False), **kw)
    assert np.array_equal(dh.split_test_train(data, data_cfg=DataCfg(test_fraction=0.25, stratify=True), **kw)[1].index, te.index)   ##nothing was overwritten

def test_every_test_splits_into_its_own_store():
    """conftest redirects datahandler.TRAIN_TEST_DIR to a fresh dir per test, so no test reads or writes datasets/train_test/."""
    import datahandler, paths
    from datahandler import DataHandler
    store = datahandler.TRAIN_TEST_DIR
    assert store != paths.TRAIN_TEST_DIR and paths.ROOT_DIR not in store.parents and not list(store.iterdir())   ##outside the tree, and empty at the start of a test
    DataHandler().split_test_train(make_data(), dataset_id="ownstore", data_cfg=DataCfg(test_fraction=0.25))
    assert (store / "split_ownstore.json").exists() and not (paths.TRAIN_TEST_DIR / "split_ownstore.json").exists()

def test_test_split_is_the_same_on_every_computer(tmp_path):
    """The split uses SPLIT_SEED, not the run's seed: made twice from scratch -> the same test set; RunMeta.split_id names it."""
    from datahandler import DataHandler
    dh, data, dc = DataHandler(), make_data(), DataCfg(test_fraction=0.25)
    a, b = (dh.split_test_train(data, dataset_id="s", data_cfg=dc, save_dir=tmp_path / d)[1].index for d in ("a", "b"))
    assert np.array_equal(a, b)

    assert trained().runmeta.split_id == trained(epochs=1).runmeta.split_id is not None      ##the stored split is reused: same id

def test_phasicnet_retains_no_run_state():
    """The assembler is stateless w.r.t. runs: cleanup-by-design."""
    p = build(); run = p.train(make_data(), RunCfg(protocol=protocol_for(val_split=0.25), data=DataCfg(test_fraction=0.25)), save_run=False)
    for attr in ("ctx", "history", "trial", "run", "synapse", "plasticity"):
        assert getattr(p, attr, None) is None, f"PhasicNet kept .{attr}"
    ##the current Run-owned state, checked by name AND by type
    for attr in ("integrator", "threshold", "noise", "phases", "datahandler", "report", "checkpoints", "train_set", "test_set"):
        assert getattr(p, attr, None) is None, f"PhasicNet kept .{attr}"
    assert not any(isinstance(v, (Run, RunContext, RunHistory)) for v in vars(p).values()), "PhasicNet holds a Run, a RunContext or a history"
    assert run.ctx is not None


# ------------------------------------------------------------------ aggregation & fits
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
    fa = a.datahandler.fold_split(a.train_set, val_split=0.25, rng=a.ctx.rng.rng("data", f"folds:{a.runmeta.dataset_id}"))
    fb = b.datahandler.fold_split(b.train_set, val_split=0.25, rng=b.ctx.rng.rng("data", f"folds:{b.runmeta.dataset_id}"))
    assert len(fa) == len(fb) == 1 and fa[0][1] is not None and len(fa[0][1]) > 0      ##non-vacuity guard (zip over empty lists passes)
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
    cv = protocol_for(epochs=2, val_split=0.25)
    base = RunCfg(protocol=cv)
    assert base.hashable() == RunCfg(protocol=cv, history=HistoryCfg(include_input=True)).hashable()
    assert base.hashable() != RunCfg(protocol=cv, checkpoint=CheckpointCfg(select="best")).hashable()   ##which state becomes the model DOES change the result
    assert base.hashable() == RunCfg(protocol=cv, checkpoint=CheckpointCfg(to_disk=False)).hashable()
    assert base.hashable() != RunCfg(protocol=protocol_for(epochs=3, val_split=0.25)).hashable()
    assert base.hashable() == RunCfg(protocol=cv, checkpoint=CheckpointCfg(keep_last=None)).hashable()     ##keep_last = IO only
    assert "history" not in base.hashable() and "to_disk" not in base.hashable()["checkpoint"]

def test_no_names_in_any_hash():
    """Renaming a run must not change its identity."""
    ##both names are varied; the recorded names prove they were used
    a = build().train(make_data(), RunCfg(data=DataCfg(test_fraction=0.25)), cfg_name="cA", run_name="A", save_run=False).runmeta
    b = build().train(make_data(), RunCfg(data=DataCfg(test_fraction=0.25)), cfg_name="cB", run_name="B", save_run=False).runmeta
    assert (a.run_name, b.run_name, a.cfg_name, b.cfg_name) == ("A", "B", "cA", "cB")
    assert a.run_id == b.run_id and a.cfg_id == b.cfg_id and a.derived_cfg_id == b.derived_cfg_id


# ------------------------------------------------------------------ rng & modules
def test_rng_is_keyed_and_order_independent():
    r = RNGHandler(7)
    assert r.rng("fit", 2).random() == RNGHandler(7).rng("fit", 2).random()
    assert r.rng("fit", 1).random() != r.rng("fit", 2).random()
    assert RNGHandler.from_rng_spec(r.rng_spec).seed == 7

def test_params_and_state_dict_split_by_type():
    """By TYPE, not name case: params = every public non-array, state_dict = every public ndarray, a leading underscore = in neither."""
    run = trained()
    p, sd = run.plasticity.params, run.plasticity.state_dict
    assert "tau_pretrace" in p and "A_ltp" in p and "Pre_trace" not in p
    assert set(sd) == {"Pre_trace", "Post_trace", "eta_pre"} and "_ltd" not in sd
    assert set(run.threshold.state_dict) == {"V_th"} and "_dV_th" not in run.threshold.state_dict     ##AdaptiveThreshold's module-owned threshold



# ------------------------------------------------------------------ end-of-life / final-model protocol
import gc, weakref
from runcfg import CheckpointCfg
from checkpoint import load_trained_model
from protocols import HoldoutValidation
from paths import RUNS_DIR

def test_input_neurons_receive_no_synapses():
    """Input neurons are drive, not neurons: no synapse may target them, before or after training (dense and sparse)."""
    ##post_offset False AND True, dense W rows / sparse edge targets, before AND after training
    for fmt in ("dense", "sparse"):
        for off in (False, True):
            run = build(fmt, post_offset=off)._prep_run(make_data(), RunCfg(data=DataCfg(test_fraction=0.25)))
            net = run.ctx.net; n_in = net.n_input; shift = net.n_total - net.n_post
            assert bool(shift) == off and np.all(np.asarray(net.post_idx) + shift >= n_in)
            for when in ("before", "after"):
                if when == "after":
                    HoldoutValidation(None, run).execute()
                w = run.ctx.syn.weights
                if fmt == "dense":
                    assert np.count_nonzero(w.W) > 0, "no synapses at all: the check proves nothing"
                    assert not w.W[:max(n_in - shift, 0)].any(), f"an input neuron's W row has weight ({when} training, post_offset={off})"
                else:
                    assert w.post.size > 0 and np.all(w.post + shift >= n_in), f"a sparse edge targets an input neuron ({when} training, post_offset={off})"

def test_holdout_run_has_one_fold_and_no_refit():
    """A holdout run has ONE fold and no "refit" key; the fold's score is the run's."""
    run = trained(val_split=0.25, epochs=2)
    r, m = run.report, run.runmeta
    assert "refit" not in r and len(r["folds"]) == 1 and r["fold_scores"] == [r["folds"][0]["score"]] and r["score"] == r["fold_scores"][0]
    assert not hasattr(m, "refit")                        ##protocol facts, not identity: the report carries them

def test_select_final_never_restores_and_scores_the_last_epoch():
    run = trained(val_split=0.25, epochs=3, checkpoint=CheckpointCfg(select="final"))
    fold = run.report["folds"][0]
    assert run.checkpoints.best is None                   ##nothing was even snapshotted on improvement
    assert fold["score"] == fold["last_score"]

def test_select_best_restores_and_scores_the_best_epoch():
    run = trained(val_split=0.25, epochs=3, checkpoint=CheckpointCfg(select="best", to_disk=False))
    fold = run.report["folds"][0]
    assert fold["score"] == fold["best_score"]
    assert np.array_equal(run.ctx.syn.W, run.checkpoints.best.W)   ##the best epoch is what is loaded

def test_collapse_is_recorded_not_rolled_back():
    run = trained(epochs=2)
    for fold in run.report["folds"]:
        assert fold["collapsed"] == (fold["stopped"] == "collapse")
        assert (fold["collapse_epoch"] is not None) == fold["collapsed"]

def test_forced_collapse_is_recorded_and_the_collapsed_state_is_kept():
    ##rate_hz_min=1e6 makes every epoch "silent" (is_collapsed is rate-only), so a collapse is forced
    run = trained(val_split=0.25, epochs=3, scorer=ScorerCfg(rate_hz_min=1e6), checkpoint=CheckpointCfg(select="final"))
    fold = run.report["folds"][0]
    assert fold["collapsed"] and fold["stopped"] == "collapse" and fold["collapse_epoch"] == 0 and fold["epochs_run"] == 1   ##stopped at once, not after the 3 epochs asked for
    assert run.checkpoints.collapse is not None and run.checkpoints.collapse.reason == "collapse"
    assert np.array_equal(run.ctx.syn.W, run.checkpoints.collapse.W)      ##recorded, NOT rolled back: the model is the state the probe flagged

def test_load_trained_model_restores_module_state():
    """load_trained_model reads module state (mod__ keys) back, not only W."""
    run = build().train(make_data(), RunCfg(data=DataCfg(test_fraction=0.25)), save_run=True)
    snap = load_trained_model(RUNS_DIR / f"TrainedModel_{run.runmeta.run_id}.npz")
    assert np.array_equal(snap.W, run.ctx.syn.W)
    assert np.array_equal(snap.modules["plasticity"]["Pre_trace"], run.plasticity.Pre_trace)
    assert snap.runstep == run.ctx.sim.current_runstep > 0 and snap.train_step == run.ctx.sim.train_step == snap.runstep     ##both clocks are saved (equal here: no eval pass)

def test_content_weights_saves_only_the_synaptic_configuration():
    run = build().train(make_data(), RunCfg(checkpoint=CheckpointCfg(content="weights"), data=DataCfg(test_fraction=0.25)), save_run=True)
    keys = np.load(RUNS_DIR / f"TrainedModel_{run.runmeta.run_id}.npz").files
    assert keys == ["W"], keys

def test_runmeta_is_one_object_on_history():
    run = trained(epochs=1, val_split=None)
    assert run.runmeta is run.history.runmeta

def test_released_run_still_tests_but_cannot_train():
    run = trained(epochs=1, val_split=None)
    run.release_training_state()
    assert run.test()["n_test"] > 0
    with pytest.raises(RuntimeError):
        run.set_mode("train")          ##a released run cannot enter "train"

def test_run_is_freed_by_refcount_alone():
    """Dropping the last reference frees a Run without gc.collect(); kept apart from pytest.raises, whose traceback holds the Run."""
    run = trained(epochs=1, val_split=None)
    run.release_training_state()
    run.test()
    gc.disable()
    try:
        ref = weakref.ref(run); del run
        assert ref() is None, "Run survived del: a reference cycle is back"
    finally:
        gc.enable()


# --------------------------------------------------------------------- error names, RunMeta types, the scratch scripts
import os, re, subprocess, sys
from paths import ROOT_DIR


def test_refractory_abs_names_itself_and_runmeta_counts_samples_as_int():
    """RefractoryAbs's error names RefractoryAbs, and RunMeta.dataset_id_n_samples is annotated int."""
    from original_dynamics import RefractoryAbs
    from meta import RunMeta
    with pytest.raises(ValueError, match="^RefractoryAbs requires t_refrac"):
        RefractoryAbs(ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0, t_refrac=None), SimulationCfg(dt=1.0))
    assert RunMeta.__dataclass_fields__["dataset_id_n_samples"].type in (int, "int")


@pytest.mark.parametrize("script", ["smoke_test.py", "brian2_stdp_method.py", "brian2_stdp_method2.py"])
def test_scratch_scripts_import_under_the_new_layout(script):
    """The scratch scripts import under the current layout (non-__main__ name, so nothing runs); their bodies are not checked."""
    path = ROOT_DIR / "experiments" / script
    code = f"import runpy; runpy.run_path({str(path)!r}, run_name='import_check')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env={**os.environ, "MPLBACKEND": "Agg"})
    assert r.returncode == 0, r.stderr[-2000:]


def test_grid_positions_enter_network_id_and_survive_save_load(tmp_path, caplog):
    """Grid positions are realized output like pre_idx / W0, so they are hashed into network_id; a grid-less network hashes as before."""
    from phasicnet import PhasicNet
    from network import Network
    def build(share_seed):
        p = PhasicNet(share_seed); p._instantiate_builder(); b = p.builder
        b.set_neurons(("input", 2, True), ("exc", 6), ("inh", 2, False, True))
        b.set_positions(exc=(2, 2, 2), inh=(2, 2, 2), share="liquid")
        b.set_synapses(("in_exc", "input", "exc"), ("exc_inh", "exc", "inh")); b.set_connectivity(in_exc=1.0, exc_inh=1.0)
        b.set_weights({"in_exc": 1.0, "exc_inh": 1.0})
        return b.realize(network_name="grid", save=False)
    a, b = build(0), build(1)
    assert np.array_equal(a.pre_idx, b.pre_idx) and np.array_equal(a.W0, b.W0)      ##same edges and weights (p = 1, constant W) ...
    assert not all(np.array_equal(a.positions[g], b.positions[g]) for g in a.positions)   ##... different placement
    assert a.meta.network_id != b.meta.network_id
    a.save_net(network_name="grid", spec_name="grid_spec", JSON=True, YAML=False, save_to=tmp_path)
    js = sorted(tmp_path.rglob("*.json"))
    assert len(js) == 1
    from phasicnet import PhasicNet as P
    p = P(0); p.load_net(str(js[0]))
    assert p.network.meta.network_id == a.meta.network_id
    assert all(np.array_equal(p.network.positions[g], a.positions[g]) for g in a.positions)
    with caplog.at_level("WARNING"):
        p.network.rehash_network()
    assert "inconsistent" not in caplog.text
