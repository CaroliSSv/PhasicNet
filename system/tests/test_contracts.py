#test_contracts -- the two invariants the architecture RESTS on, both of which fail SILENTLY:
#   1. COORDINATE SPACES: arrays are NEURON space (n_total) or post side (n_post); under post_offset=False the two are EQUAL, so a slip hides.
#   2. WRITE SETS: a runtime module writes ONLY the UpdateState arrays it declares in `writes`, never NeuronalState / SynapticState.
# Run: pytest system/tests/test_contracts.py -q     (from the repo root)
from __future__ import annotations
import dataclasses
import numpy as np
from modelinput import ModelInput
import pytest

from phasicnet import PhasicNet
from dynamicscfg import (DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg,
                 PlasticityCfg, NoiseCfg, PhaseCfg)
from runcfg import RunCfg, DataCfg
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, LIFeuler, AdaptiveThreshold, StaticThreshold, TraceSTDP, SparseTraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase
from states import OPTIONAL_STATE_ARRAYS
from dynamics import Dynamics, _configured_for_network

SEED = 31337
N_INPUT, N_EXC, N_INH = 5, 9, 3

##Which SPACE each state array lives in. This table IS the contract; the code must match it.
##post side = run.ctx.net.n_post; the traces and the ltd scratch (_ltd) are module-owned
NEURON_SPACE = ("Spikes", "pre_spike", "Pre_trace")
POST_SIDE = ("V", "V_th", "post_spike", "I2V", "dV", "rec_spikes", "_dV_th", "neur_noise", "Post_trace")
POST_BY_NEURON = ("W", "dW", "_ltd")          ##2-D: rows post side, columns NEURON space


##tiny random binary input: _prep_run needs a dataset to split
def _data(n_samples=12, T=40):
    X = np.random.default_rng(0).random((n_samples * T, N_INPUT)) < 0.3
    return ModelInput.from_tonic([(X[i * T:(i + 1) * T], i % 2) for i in range(n_samples)])


def build(post_offset: bool = False, *, threshold_cls=AdaptiveThreshold,
          integrator_cls=LIFexact, fmt: str = "dense", update: str = "timestep"):
    ##returns the RUN; fmt/update select the dense or sparse module pair
    pnet = PhasicNet(SEED)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", N_INPUT, True), ("exc", N_EXC), ("inh", N_INH, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"),
                              ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc=0.6, exc_exc=0.3, exc_inh=0.4, inh_exc=0.4)
    pnet.builder.set_weights({"in_exc": 1.2, "exc_exc": 2.2, "exc_inh": 1.1, "inh_exc": 2.4})
    pnet.builder.spec_name = "contract_spec"
    pnet.builder.realize(network_name="contract_net", post_offset=post_offset, save=False)

    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on=update),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=10.0),
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.005, A_ltd=0.006,
                                 eta_exc=1e-3, eta_inh=1e-3, w_min=-10.0, w_max=10.0),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))

    sparse = fmt == "sparse"
    pnet.set_runtime_classes(synapse_cls=DeltaCurrentSparse if sparse else DeltaCurrentDense, integrator_cls=integrator_cls,
                             threshold_cls=threshold_cls, plasticity_cls=SparseTraceSTDP if sparse else TraceSTDP,
                             noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    return pnet._prep_run(_data(), RunCfg(data=DataCfg(test_fraction=0.25)))


def warm(run, T=60):      ##fewer steps never lift V to threshold: the probes would run on silence
    """A few real steps, so buffers hold non-trivial values before anything is probed."""
    X = (np.random.default_rng(1).random((T, run.ctx.net.n_input)) < 0.5)
    run.run_sample(X)                       ##mode "train" is the default
    return run


def _arrays(state):
    """name -> ndarray for every array a state object holds; descends into state.weights, or the write-set tests pass vacuously."""
    out = {}
    for f in dataclasses.fields(state):
        value = getattr(state, f.name)
        if f.name == "weights":
            out.update(_arrays(value))
        elif isinstance(value, np.ndarray):
            out[f.name] = value
    return out


def snapshot(state):
    return {name: array.copy() for name, array in _arrays(state).items()}


def changed(state, before):
    out = set()
    now = _arrays(state)
    for name, old in before.items():
        new = now.get(name)
        if old is None or new is None:
            continue
        if not np.array_equal(old, new):
            out.add(name)
    return out


def _modules(run):
    return (run.synapse, run.integrator, run.threshold, run.plasticity)


##the execution modes SimulationCfg can select (dense+event is rejected)
MODES = (("dense", "timestep"), ("sparse", "timestep"), ("sparse", "event"), ("sparse", "hybrid_event"))


# --------------------------------------------------------------------- 1. coordinate spaces

def _find(run, name):
    """Locate a state array by name across the three state objects. `or` is not usable here:
    truth-testing a numpy array raises."""
    ctx = run.ctx
    ##the modules too: the traces and the ltd scratch are module-owned
    for state in (ctx.neur, ctx.upd, ctx.syn, *run.modules.values()):
        array = getattr(state, name, None)
        if array is not None:
            return array
    return None


def test_every_state_array_is_in_its_declared_space():
    for post_offset in (False, True):
        run = build(post_offset)
        net, ctx = run.ctx.net, run.ctx
        n_total, n_post = net.n_total, net.n_post
        found = set()

        for name in NEURON_SPACE:
            array = _find(run, name)
            if array is None:
                continue
            found.add(name)
            assert array.shape == (n_total,), f"{name} must be NEURON space (post_offset={post_offset})"

        for name in POST_SIDE:
            array = _find(run, name)
            if array is None:
                continue
            found.add(name)
            assert array.shape == (n_post,), f"{name} must be post side (post_offset={post_offset})"

        for name in POST_BY_NEURON:
            array = _find(run, name)
            if array is None:
                continue
            found.add(name)
            assert array.shape == (n_post, n_total), f"{name} must be (n_post, n_total)"

        ##EVERY name in the tables is allocated here; presence is asserted so a RENAMED array cannot fall through the `continue`s
        assert found == set(NEURON_SPACE + POST_SIDE + POST_BY_NEURON), f"arrays missing from the state: {set(NEURON_SPACE + POST_SIDE + POST_BY_NEURON) - found}"

        assert ctx.syn.syn_mask.shape == (n_post, n_total)
        ##step_input is NEITHER space: it is the input nodes only, straight from the dataloader.
        assert ctx.upd.step_input.shape == (net.n_input,)


def test_sparse_edge_coordinates_are_in_their_declared_space():
    """Sparse edges: `post` is a W ROW index (post side), `pre` a true neuron id (neuron space, never shifted)."""
    for post_offset in (False, True):
        run = build(post_offset, fmt="sparse")
        net, syn = run.ctx.net, run.ctx.syn
        assert syn.W.shape == syn.pre.shape == syn.post.shape == (net.n_synapses,)
        assert syn.weights.n_post == net.n_post
        assert 0 <= syn.post.min() and syn.post.max() < net.n_post, f"post must be a ROW index (post_offset={post_offset})"
        assert 0 <= syn.pre.min() and syn.pre.max() < net.n_total, f"pre must be a neuron id (post_offset={post_offset})"
        assert (syn.post + (net.n_total - net.n_post)).min() >= net.n_input, "row index + offset must land on a RECURRENT neuron: input nodes receive nothing"
        index = syn.weights.index
        assert index.post_ptr.shape == (net.n_post + 1,) and index.pre_ptr.shape == (net.n_total + 1,)
        with pytest.raises(RuntimeError):
            syn.syn_mask                      ##a sparse W has no non-edges, so there is no dense mask to be in either space


def test_no_offset_spaces_coincide_exactly():
    """With post_offset=False the two spaces are the SAME numbers (why space bugs hide), asserted as a stated fact."""
    net = build(False).ctx.net
    assert net.n_post == net.n_total
    assert net.rec_slice == slice(net.n_input, net.n_total)


def test_offset_spaces_differ_by_exactly_the_dropped_rows():
    net = build(True).ctx.net
    assert net.n_post == net.n_rec
    assert net.n_total - net.n_post == net.n_input
    assert net.rec_slice == slice(0, net.n_post)


def test_rec_slice_is_a_slice_not_a_mask():
    """Run.step writes THROUGH neur.V[net.rec_slice]: a basic slice is a view; a mask would copy and the reset would be a silent no-op."""
    for post_offset in (False, True):
        net = build(post_offset).ctx.net
        assert isinstance(net.rec_slice, slice)
        assert net.rec_slice.step in (None, 1)
        ##and it must select exactly the recurrent neurons
        assert net.rec_slice.stop - net.rec_slice.start == net.n_rec


def test_input_nodes_never_get_a_postsynaptic_trace():
    """Input nodes receive nothing: UNCOMPRESSED their post_spike / Post_trace rows must stay zero."""
    run = warm(build(False))
    n_input = run.ctx.net.n_input
    assert not run.ctx.neur.post_spike[:n_input].any(), "input nodes gained a post_spike"
    assert not run.plasticity.Post_trace[:n_input].any(), "input nodes gained a post_trace"
    assert run.plasticity.Post_trace[n_input:].any(), "no recurrent neuron fired during warm(), so this proves nothing"


def test_input_rows_of_W_stay_empty_uncompressed():
    """The rows compression drops must hold nothing, before AND after learning."""
    run = build(False)
    n_input = run.ctx.net.n_input
    W0 = run.ctx.syn.W.copy()
    assert not run.ctx.syn.W[:n_input].any()
    warm(run)
    assert not np.array_equal(W0, run.ctx.syn.W), "warm() did not learn, so this proves nothing"
    assert not run.ctx.syn.W[:n_input].any(), "learning wrote into rows that do not exist compressed"


# --------------------------------------------------------------------- 2. write sets

def test_the_snapshot_probe_sees_weight_writes():
    """Power check: the snapshot probe the write-set tests use sees a write to W (in syn.weights)."""
    ctx = warm(build(False)).ctx
    before = snapshot(ctx.syn)
    assert "W" in before
    ctx.syn.W[0, 0] += 1.0
    assert changed(ctx.syn, before) == {"W"}


def test_modules_write_only_what_they_declare():
    """Each module changes only the UpdateState arrays it declares in `writes`, in every execution mode."""
    for fmt, update in MODES:
        run = warm(build(False, fmt=fmt, update=update))
        ctx = run.ctx
        for module in _modules(run):
            before = snapshot(ctx.upd)
            out = module.step(ctx)
            touched = changed(ctx.upd, before)
            declared = set(module.writes)
            assert touched <= declared, (f"{type(module).__name__} wrote {sorted(touched - declared)} "
                                         f"but declares writes={sorted(declared)} ({fmt}/{update})")
            ##a sparse module RETURNS its output: post side for the current, one per edge for dw
            if fmt == "sparse" and module is run.synapse:
                assert out.shape == (ctx.net.n_post,)
            if fmt == "sparse" and module is run.plasticity:
                if update == "timestep":
                    assert out.shape == (ctx.net.n_synapses,)
                else:     ##the event modes return the pair (k, dw_k) of touched plastic edges
                    k, dw_k = out
                    assert k.shape == dw_k.shape and k.ndim == 1 and (k.size == 0 or (k.min() >= 0 and k.max() < ctx.net.n_synapses))


def test_modules_do_not_mutate_neuronal_or_synaptic_state():
    """Modules VIEW NeuronalState / SynapticState and never mutate them; only the orchestrator commits back."""
    for fmt, update in MODES:
        run = warm(build(False, fmt=fmt, update=update))
        ctx = run.ctx
        for module in _modules(run):
            neur_before, syn_before = snapshot(ctx.neur), snapshot(ctx.syn)
            module.step(ctx)
            assert not changed(ctx.neur, neur_before), f"{type(module).__name__} mutated NeuronalState ({fmt}/{update})"
            assert not changed(ctx.syn, syn_before), f"{type(module).__name__} mutated SynapticState ({fmt}/{update})"


def test_declared_requires_are_all_allocated():
    """Every array a module `requires` is a known optional array and is allocated (else None and a TypeError mid-run)."""
    run = build(False)
    for module in _modules(run):
        for name in module.requires:
            assert name in OPTIONAL_STATE_ARRAYS, f"{name} is not a known optional array"
            assert _find(run, name) is not None, f"{type(module).__name__} requires {name}, which was not allocated"


def test_unrequired_optional_arrays_are_not_allocated():
    """The other half: StaticThreshold needs no V_th/dV_th, so nothing should be allocated for it (V_th is module-owned, never shared)."""
    run = build(False, threshold_cls=StaticThreshold)
    assert getattr(run.threshold, "V_th", None) is None
    assert not hasattr(run.ctx.neur, "V_th") and not hasattr(run.ctx.upd, "dV_th")


def test_configured_for_network_reports_the_subclass_not_the_base():
    """Only modules that OVERRIDE configure_for_network report as configured."""
    run = build(False)
    assert _configured_for_network(run.plasticity)      # TraceSTDP defines it (eta_pre)
    assert not _configured_for_network(run.synapse)     # DeltaCurrentDense inherits the base no-op
    assert not _configured_for_network(run.integrator)
    assert Dynamics.configure_for_network(run.synapse, run.ctx.net) is None


def test_swapping_a_module_needs_no_orchestrator_change():
    """Plug-and-play: with the alternate integrator and threshold the run completes with finite state and the right shapes."""
    run = warm(build(False, integrator_cls=LIFeuler, threshold_cls=StaticThreshold))
    assert np.all(np.isfinite(run.ctx.neur.V))
    assert run.ctx.upd.rec_spikes.shape == (run.ctx.net.n_post,)


# --------------------------------------------------------------------- unused cfg fields

def test_unused_t_refrac_is_warned_about_by_static_and_adaptive_threshold_only(caplog):
    """StaticThreshold / AdaptiveThreshold warn when t_refrac is set (only RefractoryAbs reads it); RefractoryAbs does not warn."""
    import logging
    from original_dynamics import RefractoryAbs
    sim = SimulationCfg(dt=1.0)
    with caplog.at_level(logging.WARNING, logger="original_dynamics"):
        StaticThreshold(ThresholdCfg(v_reset=-70.0, v_th=-52.0, t_refrac=2.0), sim)
        AdaptiveThreshold(ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0, t_refrac=2.0), sim)
        RefractoryAbs(ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0, t_refrac=2.0), sim)
        StaticThreshold(ThresholdCfg(v_reset=-70.0, v_th=-52.0), sim)
        AdaptiveThreshold(ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0), sim)
    msgs = [r.getMessage() for r in caplog.records if "t_refrac" in r.getMessage()]
    assert msgs == ["StaticThreshold will ignore inputs in t_refrac in ThresholdCfg.", "AdaptiveThreshold will ignore inputs in t_refrac in ThresholdCfg."]


def test_noise_module_has_no_weight_noise_and_the_base_hook_stays():
    """TestNoise has no weight-noise fields or "weights" stream; noise.on_weights hits the base no-op."""
    from dynamics import Noise
    run = build(False)
    noise = run.noise
    assert not hasattr(noise, "sigma_weights") and not hasattr(noise, "mu_weights") and set(noise.noise_rng) == {"synapse", "membrane", "threshold"}
    assert "sigma_weights" not in noise.params and noise.sigma_mem == 0.0   ## TestNoise is a __slots__ class: params (vars()) carries no slot attribute
    assert type(noise).on_weights is Noise.on_weights and noise.on_weights(run.ctx, run.phases["DEFAULT"]) is None
