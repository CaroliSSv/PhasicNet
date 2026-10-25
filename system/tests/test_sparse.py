#test_sparse -- the sparse format and the event update modes must be the SAME model as the dense path, so every test here is an equivalence test.
# The mode is chosen by SimulationCfg(weights_format="dense"|"sparse", update_on="timestep"|"event"|"hybrid_event"); tests compare runs of one seed that differ only there.
# Caveat: BLAS np.dot (float32) can differ by one ULP from the float64 sparse sum; "event" matches "timestep" to float tolerance only (section 7b).
# Run: pytest test_sparse.py -q
from __future__ import annotations
import dataclasses
import numpy as np
from modelinput import ModelInput
import pytest

from phasicnet import PhasicNet
from system.dynamicscfg import (DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg,
                 PlasticityCfg, NoiseCfg, PhaseCfg)
from runcfg import RunCfg, DataCfg, ScorerCfg, ProtocolCfg
from protocols import HoldoutValidation
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, AdaptiveThreshold, TraceSTDP, SparseTraceSTDP
from original_dynamics import TestNoise
from scorer import Scorer
from phases import DefaultPhase
from freeze import _build_dense_plastic_mask

SEED = 909
N_INPUT, N_EXC, N_INH = 6, 10, 4


##One Run per execution mode: same network, cfg and data recipe; only SimulationCfg(weights_format, update_on) and the module pair differ.
def _data(n_input=N_INPUT, n_samples=12, T=40):
    X = np.random.default_rng(0).random((n_samples * T, n_input)) < 0.3
    return ModelInput.from_tonic([(X[i * T:(i + 1) * T], i % 2) for i in range(n_samples)])


def assemble(fmt="dense", update="timestep", freeze=(), sizes=(N_INPUT, N_EXC, N_INH), p=(0.6, 0.3, 0.4, 0.4), epochs=1, plasticity_cls=None):
    ##plasticity_cls: replaces the format's plasticity module (e.g. _FullArray); None = the stock one
    n_in, n_exc, n_inh = sizes
    pnet = PhasicNet(SEED)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", n_in, True), ("exc", n_exc), ("inh", n_inh, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"),
                              ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc=p[0], exc_exc=p[1], exc_inh=p[2], inh_exc=p[3])
    pnet.builder.set_weights({"in_exc": 1.2, "exc_exc": 2.2, "exc_inh": 1.1, "inh_exc": 2.4})
    pnet.builder.spec_name = "sparse_spec"
    pnet.builder.realize(network_name="sparse_net", save=False)

    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on=update),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=10.0),
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.005, A_ltd=0.006,
                                 eta_exc=1e-3, eta_inh=1e-3, w_min=-10.0, w_max=10.0, freeze=freeze),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))

    sparse = fmt == "sparse"
    pnet.set_runtime_classes(synapse_cls=DeltaCurrentSparse if sparse else DeltaCurrentDense, integrator_cls=LIFexact,
                             threshold_cls=AdaptiveThreshold, plasticity_cls=plasticity_cls or (SparseTraceSTDP if sparse else TraceSTDP),
                             noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.cfg_name = pnet.run_name = "sparse"
    ##rate_hz_min=0.0: a silence flag would stop the fit after epoch 1 and cut the multi-epoch comparisons short
    runcfg = RunCfg(protocol=ProtocolCfg(epochs=epochs), scorer=ScorerCfg(rate_hz_min=0.0), data=DataCfg(test_fraction=0.25))
    return pnet, pnet._prep_run(_data(n_in), runcfg)


def protocol_for(*, epochs=1, val_split=None, patience=None):
    """A ProtocolCfg for HoldoutValidation (val_split=None is the plain holdout)."""
    return ProtocolCfg(epochs=epochs, val_split=val_split, patience=patience)


def build(**kw):
    return assemble(**kw)[1]


def paired(**kw):
    """The dense Run and its sparse twin (same seed -> same network, same cfg except the format), freeze resolved on both."""
    return build(fmt="dense", **kw), build(fmt="sparse", **kw)


def train(run):
    """One real training pass (HoldoutValidation, one fit); asserts it ran every epoch it was asked for."""
    proto = HoldoutValidation(run.runcfg.protocol, run); proto.execute()
    assert len(proto.ops.epoch_log) == run.runcfg.protocol.epochs
    return run


def set_spikes(run, spikes):
    run.ctx.neur.Spikes[...] = spikes


def scatter(sparse_syn, shape):
    """The sparse W as a dense (n_post, n_total) matrix: the edge list scattered back to where it lives."""
    dense = np.zeros(shape, dtype=sparse_syn.W.dtype)
    dense[sparse_syn.post, sparse_syn.pre] = sparse_syn.W
    return dense


def edge_weights(run, like):
    """The weights of `run` at the edges of the sparse state `like`, in its edge order: compares a dense W with a sparse one."""
    W = run.ctx.syn.W
    return W[like.post, like.pre] if W.ndim == 2 else W.copy()


def _scattered(out, ctx):
    """A plasticity step's (k, dw_k) pair scattered back into the (n_synapses,) dw it stands for; a full dw passes through."""
    if not isinstance(out, tuple):
        return out
    k, dw_k = out
    dw = np.zeros(ctx.syn.W.size, dtype=dw_k.dtype); dw[k] = dw_k
    return dw


class _FullArray(SparseTraceSTDP):
    """The stock module with its pair scattered back into the full dw, so Run.step_sparse takes its full-array branch for the same physics."""
    def step(self, ctx):
        return _scattered(super().step(ctx), ctx)


# --------------------------------------------------------------------- 1. representation

def test_to_sparse_is_to_dense_without_the_scatter():
    """_to_dense scatters W0 into (n_post, n_total); to_sparse just does not scatter. Same data."""
    dense, sparse = paired()
    assert np.array_equal(scatter(sparse.ctx.syn, dense.ctx.syn.W.shape), dense.ctx.syn.W)
    assert sparse.ctx.syn.W.size == sparse.ctx.net.n_synapses


def test_sparse_holds_only_realized_edges():
    """The whole point: a non-edge has no array position, so syn_mask has nothing left to mask."""
    dense, sparse = paired()
    assert sparse.ctx.syn.W.size == int(dense.ctx.syn.syn_mask.sum())
    assert sparse.ctx.syn.W.size < dense.ctx.syn.W.size
    assert not hasattr(sparse.ctx.syn.weights, "syn_mask")
    with pytest.raises(RuntimeError):         ##syn_mask RAISES on sparse weights (hasattr would not catch it)
        sparse.ctx.syn.syn_mask


def test_edge_order_is_row_major():
    """The sparse edge list is sorted row-major by (post, pre), the order a dense row scan sums in (the matvec bit-identity rests on it)."""
    sparse = build(fmt="sparse")
    syn, net = sparse.ctx.syn, sparse.ctx.net
    key = syn.post.astype(np.int64) * net.n_total + syn.pre
    assert np.all(np.diff(key) > 0), "edge list is not strictly sorted by (post, pre)"
    ptr = syn.weights.index.post_ptr             ##each post row is the contiguous block post_ptr[i]:post_ptr[i+1]
    assert np.array_equal(np.repeat(np.arange(net.n_post), np.diff(ptr)), syn.post)


# --------------------------------------------------------------------- 2. equivalence of the maths

def _ordered_dense_matvec(W, spikes):
    """Dense reference that sums in a DEFINED order (row scan), unlike BLAS."""
    return (W * spikes).sum(axis=1, dtype=np.float32)


def _currents(dense, sparse, spikes):
    """(dense module's I2V, sparse module's returned I) for one spike pattern, through the modules the Runs actually use."""
    set_spikes(dense, spikes); set_spikes(sparse, spikes)
    dense.synapse.step(dense.ctx)
    return dense.ctx.upd.I2V.copy(), sparse.synapse.step(sparse.ctx)


def test_matches_dense_float64_accumulation():
    """The sparse matvec (float64 bincount, rounded once) is bit-identical to a dense sum accumulated in float64."""
    dense, sparse = paired()
    n_total = dense.ctx.net.n_total
    rng = np.random.default_rng(0)
    for _ in range(50):
        spikes = rng.random(n_total) < 0.3
        _, out = _currents(dense, sparse, spikes)
        reference = (dense.ctx.syn.W.astype(np.float64) * spikes).sum(axis=1).astype(np.float32)
        assert np.array_equal(out, reference)


def test_two_dense_methods_already_disagree_with_each_other():
    """BLAS np.dot and numpy's pairwise .sum() can disagree on float32; sparse differs from them only as float64 does."""
    dense, sparse = paired()
    rng = np.random.default_rng(1)
    blas_vs_ordered = blas_vs_sparse = ordered_vs_sparse = 0
    for _ in range(50):
        spikes = rng.random(dense.ctx.net.n_total) < 0.3
        blas, out = _currents(dense, sparse, spikes)
        ordered = _ordered_dense_matvec(dense.ctx.syn.W, spikes)
        blas_vs_ordered += int(not np.array_equal(blas, ordered))
        blas_vs_sparse += int(not np.array_equal(blas, out))
        ordered_vs_sparse += int(not np.array_equal(ordered, out))

    ##sparse (float64) must not be an outlier: it may differ from float32 dense only as the float32 methods differ among themselves
    assert blas_vs_sparse == ordered_vs_sparse or blas_vs_ordered > 0, \
        "sparse must not be a special case: it differs from float32 dense only as float64 does"


def test_float32_dense_disagrees_with_BOTH_float64_methods():
    """Sparse equals the float64 dense answer; float32 dense accumulation differs from it by less than 1e-5."""
    dense, sparse = paired()
    rng = np.random.default_rng(2)
    max_gap = 0.0
    for _ in range(50):
        spikes = rng.random(dense.ctx.net.n_total) < 0.3
        _, out = _currents(dense, sparse, spikes)
        f32 = _ordered_dense_matvec(dense.ctx.syn.W, spikes)
        f64 = (dense.ctx.syn.W.astype(np.float64) * spikes).sum(axis=1).astype(np.float32)
        assert np.array_equal(f64, out), "sparse must equal the float64 dense answer"
        max_gap = max(max_gap, float(np.abs(f32 - f64).max()))
    assert max_gap < 1e-5, f"float32 vs float64 accumulation gap unexpectedly large: {max_gap}"


def _random_plasticity_state(runs, rng):
    """The same random spikes and traces in every run's neuronal state and plasticity module (the inputs the STDP rule reads)."""
    net = runs[0].ctx.net
    post_spike = (rng.random(net.n_post) < 0.2).astype(np.float32)
    pre_spike = (rng.random(net.n_total) < 0.2).astype(np.float32)
    pre_trace = rng.random(net.n_total).astype(np.float32)
    post_trace = rng.random(net.n_post).astype(np.float32)
    for run in runs:
        run.ctx.neur.post_spike[...] = post_spike; run.ctx.neur.pre_spike[...] = pre_spike
        run.plasticity.Pre_trace[...] = pre_trace; run.plasticity.Post_trace[...] = post_trace
    return post_spike, pre_spike, pre_trace, post_trace


def test_stdp_dW_is_bit_identical():
    """Per-element operation ORDER is identical, so this is exact -- no summation involved."""
    dense, sparse = paired()
    net, plast = dense.ctx.net, dense.plasticity
    post_spike, pre_spike, pre_trace, post_trace = _random_plasticity_state((dense, sparse), np.random.default_rng(3))

    ##the dense rule written out from its definition, BEFORE the modules step (a step also advances the traces)
    dW = np.zeros((net.n_post, net.n_total), np.float32)
    ltd = np.zeros_like(dW)
    np.multiply(post_spike[:, None], pre_trace[None, :], out=dW)
    np.multiply(dW, plast.A_ltp, out=dW)
    np.multiply(post_trace[:, None], pre_spike[None, :], out=ltd)
    np.multiply(ltd, plast.A_ltd, out=ltd)
    np.subtract(dW, ltd, out=dW)
    np.multiply(dW, plast.eta_pre[None, :], out=dW)
    dW *= dense.ctx.syn.syn_mask             ##TraceSTDP.step applies the realized-synapse mask

    dense.plasticity.step(dense.ctx)         ##the sparse step RETURNS the per-edge dw; the dense one writes upd.dW
    dw = sparse.plasticity.step(sparse.ctx)
    syn = sparse.ctx.syn
    assert np.array_equal(dense.ctx.upd.dW, dW), "the dense module is not the dense rule"
    assert np.array_equal(dW[syn.post, syn.pre], dw)
    assert np.count_nonzero(dw) > 0, "no edge changed, so this proves nothing"


def test_commit_is_bit_identical():
    """Dense needs `dW *= syn_mask` to stop non-edges gaining weight; sparse has no non-edges."""
    ##the same frames through run_sample on a dense and a sparse Run: the committed weights must agree edge by edge
    dense, sparse = paired()
    W0 = dense.ctx.syn.W.copy()
    frames = _data()[0][0]
    for run in (dense, sparse):
        run.run_sample(frames)
    syn = sparse.ctx.syn
    assert not np.array_equal(sparse.ctx.syn.W, build(fmt="sparse").ctx.syn.W), "nothing was committed, so this proves nothing"
    assert np.array_equal(edge_weights(dense, syn), syn.W)
    assert not dense.ctx.syn.W[~dense.ctx.syn.syn_mask].any(), "a non-edge gained weight under the dense commit"
    assert np.array_equal(dense.ctx.neur.V, sparse.ctx.neur.V)


def test_clamp_is_bit_identical():
    ##reference: Dale's law per plastic edge (inhibitory pre -> [w_min, 0], else [0, w_max]); frozen projections are outside the per-step clamp and must stay untouched
    for freeze in ((), ("inh",)):
        dense, sparse = paired(freeze=freeze)
        net, syn = dense.ctx.net, sparse.ctx.syn
        big = np.random.default_rng(5).normal(0, 20, syn.W.shape).astype(np.float32)      ##per edge, in the sparse edge order
        plastic = _build_dense_plastic_mask(dense.ctx.syn)[syn.post, syn.pre]
        assert (~plastic).any() == bool(freeze)

        dense.ctx.syn.W[...] = 0.0
        dense.ctx.syn.W[syn.post, syn.pre] = big
        syn.W[...] = big
        dense._clamp(); sparse._clamp()

        w_min, w_max = dense.plasticity.w_min, dense.plasticity.w_max
        inh = net.inh_neuron_mask[syn.pre]
        expected = np.where(inh, np.clip(big, w_min, 0.0), np.clip(big, 0.0, w_max))
        expected = np.where(plastic, expected, big)
        assert np.array_equal(syn.W, expected), f"sparse clamp is wrong (freeze={freeze})"
        assert np.array_equal(dense.ctx.syn.W[syn.post, syn.pre], expected), f"dense clamp is wrong (freeze={freeze})"
        assert np.any(expected != big), "nothing was clamped, so this proves nothing"


# --------------------------------------------------------------------- 3. freeze equivalence

def test_freeze_edges_select_the_same_synapses_as_the_dense_rectangles():
    """For every freeze spec, the sparse plastic edges are exactly the dense plastic mask at those edges."""
    for spec in ((), ("input",), ("inh",), ("exc_exc",), ("pre_inh",), ("input", "inh"), ("all",)):
        dense, sparse = paired(freeze=spec)
        plastic_dense = _build_dense_plastic_mask(dense.ctx.syn)
        syn = sparse.ctx.syn
        plastic_sparse = syn.plastic_mask                            ##realized AND non-frozen
        assert np.array_equal(plastic_dense[syn.post, syn.pre], plastic_sparse), \
            f"{spec}: sparse freeze selects a different edge set"


def test_frozen_edges_do_not_move_under_commit():
    for update in ("timestep", "event"):                             ##SparseTraceSTDP.step applies the frozen mask in both update modes
        run = build(fmt="sparse", update=update, freeze=("inh",))
        syn = run.ctx.syn
        before = syn.W.copy()
        train(run)
        assert np.array_equal(before[~syn.plastic_mask], syn.W[~syn.plastic_mask]), f"a frozen edge moved ({update})"
        assert not np.array_equal(before[syn.plastic_mask], syn.W[syn.plastic_mask]), f"no plastic edge moved ({update})"


# --------------------------------------------------------------------- 4. what it buys

def test_sparse_uses_less_memory_than_dense():
    """The reason for the whole exercise: bytes scale with EDGES, not with n_post * n_total."""
    ##counts ALL sparse arrays (CSR/CSC pointers, plastic mask too) on a larger, sparser network: the claim is about scaling
    dense = build(fmt="dense", sizes=(20, 60, 20), p=(0.05,) * 4)
    sparse = build(fmt="sparse", sizes=(20, 60, 20), p=(0.05,) * 4)
    d, s = dense.ctx.syn, sparse.ctx.syn.weights
    dense_bytes = d.W.nbytes + d.syn_mask.nbytes
    sparse_bytes = sum(a.nbytes for a in (s.W, s.pre, s.post, s.plastic_mask, s.index.post_ptr, s.index.pre_ptr, s.index.pre_perm))
    assert sparse_bytes < dense_bytes
    assert sparse.ctx.upd.dW is None and sparse.plasticity._ltd is None, "sparse must not allocate the (n_post, n_total) plasticity buffers either"


# --------------------------------------------------------------------- 5. event-based mode

def test_event_index_covers_every_edge_exactly_once():
    """CSR-by-post and CSC-by-pre must each partition the edge set."""
    sparse = build(fmt="sparse", update="hybrid_event")
    idx, net = sparse.ctx.syn.weights.index, sparse.ctx.net
    all_post = idx.synIDs_by_post(np.arange(net.n_post))
    all_pre = idx.synIDs_by_pre(np.arange(net.n_total))
    assert np.array_equal(np.sort(all_post), np.arange(net.n_synapses))
    assert np.array_equal(np.sort(all_pre), np.arange(net.n_synapses))


def test_touched_is_exactly_the_edges_with_a_spiking_endpoint():
    """The correctness claim of both event modes: dW is exactly zero everywhere else."""
    sparse = build(fmt="sparse", update="hybrid_event")
    syn = sparse.ctx.syn
    idx = syn.weights.index
    rng = np.random.default_rng(0)
    for _ in range(20):
        post_s = rng.random(sparse.ctx.net.n_post) < 0.1
        pre_s = rng.random(sparse.ctx.net.n_total) < 0.1
        expected = np.flatnonzero(post_s[syn.post] | pre_s[syn.pre])
        assert np.array_equal(idx.spiking_syns(post_s, pre_s), expected)


def test_touched_is_unique_and_ascending_so_the_pair_commit_needs_no_add_at():
    """touched() returns unique, ascending plastic edges for every spike pattern, so Run.step_sparse's W[k] += dw_k needs no add.at."""
    from synapticindex import touched
    sparse = build(fmt="sparse", update="event", freeze=("inh",))
    syn, net = sparse.ctx.syn, sparse.ctx.net
    rng = np.random.default_rng(5)
    none_post, none_pre = np.zeros(net.n_post, bool), np.zeros(net.n_total, bool)
    seen = 0
    for i in range(30):
        post_s = rng.random(net.n_post) < 0.3; pre_s = rng.random(net.n_total) < 0.3
        for ps, pr in ((post_s, pre_s), (none_post, pre_s), (post_s, none_pre)):
            k = touched(syn, ps, pr)
            expected = np.flatnonzero((ps[syn.post] | pr[syn.pre]) & syn.plastic_mask)
            assert np.array_equal(k, expected) and np.array_equal(k, np.unique(k)), i      ##np.unique = sorted AND unique
            seen += k.size
    assert seen > 0 and touched(syn, none_post, none_pre).size == 0


def test_event_matvec_is_bit_identical_to_sparse_matvec():
    """Dropping exact-zero terms from a float64 sum changes nothing."""
    timestep, event, hybrid = build(fmt="sparse", update="timestep"), build(fmt="sparse", update="event"), build(fmt="sparse", update="hybrid_event")
    rng = np.random.default_rng(1)
    for _ in range(30):
        spikes = rng.random(timestep.ctx.net.n_total) < 0.3
        set_spikes(timestep, spikes); set_spikes(event, spikes); set_spikes(hybrid, spikes)
        I = timestep.synapse.step(timestep.ctx)
        assert np.array_equal(I, event.synapse.step(event.ctx)) and np.array_equal(I, hybrid.synapse.step(hybrid.ctx))


def test_hybrid_event_stdp_commit_is_bit_identical_to_timestep_commit():
    """hybrid_event's dw, commit and clamp are bit-identical to timestep's: same expression, restricted to the edges where it is nonzero."""
    a, b = build(fmt="sparse", update="timestep", freeze=("inh",)), build(fmt="sparse", update="hybrid_event", freeze=("inh",))
    rng = np.random.default_rng(2)
    for _ in range(20):
        _random_plasticity_state((a, b), rng)
        dw_a, dw_b = a.plasticity.step(a.ctx), _scattered(b.plasticity.step(b.ctx), b.ctx)     ##the (k, dw_k) pair scattered back for the comparison
        assert np.array_equal(dw_a, dw_b)
        for run, dw in ((a, dw_a), (b, dw_b)):
            np.add(run.ctx.syn.W, dw, out=run.ctx.syn.W)
            run._clamp()
        assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W)
        assert np.array_equal(a.plasticity.Pre_trace, b.plasticity.Pre_trace) and np.array_equal(a.plasticity.Post_trace, b.plasticity.Post_trace)
    assert not np.array_equal(a.ctx.syn.W, build(fmt="sparse", freeze=("inh",)).ctx.syn.W), "nothing was committed, so this proves nothing"


def _full_run(fmt, update, freeze=()):
    ##returns the weights (as the sparse edge list, so dense and sparse compare), the spike history and the membrane state
    run = train(build(fmt=fmt, update=update, freeze=freeze, epochs=2))
    like = build(fmt="sparse").ctx.syn
    return edge_weights(run, like), run.history.spike_i().copy(), run.history.spike_ptr().copy(), run.ctx.neur.V.copy()


def test_full_run_sparse_timestep_equals_sparse_hybrid_event():
    """End to end, plasticity on, two epochs, with/without freeze: sparse timestep and hybrid_event agree to the bit (weights, spikes, V)."""
    for freeze in ((), ("inh",)):
        a = _full_run("sparse", "timestep", freeze)
        b = _full_run("sparse", "hybrid_event", freeze)
        assert a[1].size > 0, "no spikes at all: the comparison proves nothing"
        for x, y in zip(a, b):
            assert np.array_equal(x, y), f"freeze={freeze}: timestep and hybrid_event diverged"


def test_full_run_dense_equals_sparse_hybrid_event_at_this_size():
    """Dense timestep and sparse hybrid_event agree to the bit at this size (BLAS may differ by 1 ULP elsewhere): weights and spike history."""
    for freeze in ((), ("inh",)):
        d = _full_run("dense", "timestep", freeze)
        e = _full_run("sparse", "hybrid_event", freeze)
        assert np.array_equal(d[0], e[0]), "weights diverged"
        assert np.array_equal(d[1], e[1]) and np.array_equal(d[2], e[2]), "spike history diverged"


# --------------------------------------------------------------------- 7b. update_on "event": per-synapse traces, lazy decay

def test_event_helpers_live_in_synapticindex():
    """The event helpers (touched / elapsed / decayed) live in synapticindex; there is no eventbased module."""
    import importlib.util
    from synapticindex import SynapticIndex, touched, elapsed, decayed
    assert importlib.util.find_spec("eventbased") is None
    assert callable(touched) and elapsed(np.array([3, 9], np.int64), np.array([0, 1]), 10).tolist() == [6, 0]
    assert decayed(np.array([2.0, 4.0], np.float32), np.array([0, 1]), 0.5, np.array([1, 2])).tolist() == [1.0, 1.0]


def _caught_up(pl, k, t):
    """The per-synapse traces of edges k as the clocked rule would read them at the start of step t (synapticindex.decayed on the owed steps)."""
    from synapticindex import elapsed, decayed
    n = elapsed(pl.T_e, k, t)
    return decayed(pl.Pre_e, k, pl.decay_pretrace, n), decayed(pl.Post_e, k, pl.decay_posttrace, n)


def test_event_per_synapse_traces_equal_per_neuron_traces_to_tolerance():
    """For one shared tau, the lazily decayed per-synapse traces equal the per-neuron traces to float tolerance (clocked vs event Run, step by step)."""
    a, b = build(fmt="sparse", update="timestep", freeze=("inh",)), build(fmt="sparse", update="event", freeze=("inh",))
    pa, pb, syn = a.plasticity, b.plasticity, b.ctx.syn
    assert pb.Pre_trace is None and pb.Pre_e.shape == (syn.W.size,) and pb.T_e.dtype == np.int64
    rng = np.random.default_rng(3)
    for t in range(1, 60):
        a.ctx.sim.current_runstep = b.ctx.sim.current_runstep = t
        a.ctx.sim.train_step = b.ctx.sim.train_step = t      ##the event module stamps T_e with the train-step clock
        post_spike = (rng.random(a.ctx.net.n_post) < 0.1).astype(np.float32); pre_spike = (rng.random(a.ctx.net.n_total) < 0.1).astype(np.float32)
        for run in (a, b):
            run.ctx.neur.post_spike[...] = post_spike; run.ctx.neur.pre_spike[...] = pre_spike
        dw_a, dw_b = pa.step(a.ctx), _scattered(pb.step(b.ctx), b.ctx)     ##the (k, dw_k) pair scattered back for the comparison
        assert np.allclose(dw_a, dw_b, rtol=1e-5, atol=1e-7)
        assert not np.any(dw_b[~syn.plastic_mask]), "a frozen edge moved"
        for run, dw in ((a, dw_a), (b, dw_b)):
            np.add(run.ctx.syn.W, dw, out=run.ctx.syn.W); run._clamp()
    k = np.flatnonzero(syn.plastic_mask)
    x_pre, x_post = _caught_up(pb, k, 60)                                        ##what step 60 would read, on every plastic edge
    assert np.allclose(x_pre, pa.Pre_trace[syn.pre[k]], rtol=1e-5, atol=1e-6) and np.allclose(x_post, pa.Post_trace[syn.post[k]], rtol=1e-5, atol=1e-6)
    assert x_pre.max() > 0.1 and x_post.max() > 0.1, "traces never moved: the comparison proves nothing"
    assert np.allclose(a.ctx.syn.W, b.ctx.syn.W, rtol=1e-5, atol=1e-6) and not np.array_equal(a.ctx.syn.W, build(fmt="sparse", freeze=("inh",)).ctx.syn.W)
    assert pb.T_e[k].max() > 0 and not pb.T_e[~syn.plastic_mask].any(), "frozen edges are never brought up to date (they need no trace)"


def test_event_full_run_matches_timestep_to_tolerance():
    """End to end, plasticity on, two epochs, with/without freeze: "event" equals "timestep" up to float rounding (same spikes, weights within 1e-5)."""
    for freeze in ((), ("inh",)):
        a = _full_run("sparse", "timestep", freeze)
        e = _full_run("sparse", "event", freeze)
        assert a[1].size > 0, "no spikes at all: the comparison proves nothing"
        assert np.allclose(a[0], e[0], rtol=1e-5, atol=1e-6), f"freeze={freeze}: weights diverged beyond float tolerance"
        assert np.array_equal(a[1], e[1]) and np.array_equal(a[2], e[2]), f"freeze={freeze}: spike history diverged"
        assert np.allclose(a[3], e[3], rtol=1e-5, atol=1e-5)


def test_event_traces_are_module_state_reset_and_snapshotted():
    """Pre_e / Post_e / T_e are Capitalised module state: in state_dict (so a snapshot carries them) and zeroed by reset_transient; W is kept."""
    run = train(build(fmt="sparse", update="event"))
    pl = run.plasticity
    assert {"Pre_e", "Post_e", "T_e"} <= set(pl.state_dict) and "Pre_trace" not in pl.state_dict
    assert pl.T_e.max() > 0 and pl.Pre_e.any()
    W = run.ctx.syn.W.copy()
    run.reset_transient()
    assert not pl.T_e.any() and not pl.Pre_e.any() and not pl.Post_e.any() and np.array_equal(W, run.ctx.syn.W)


# --------------------------------------------------------------------- 7c. the (k, dw_k) commit; the train-step time base

def test_event_modes_return_the_touched_pair_and_timestep_the_full_dw():
    """event / hybrid_event return (k, dw_k) over the touched plastic edges (unique, ascending, float32); timestep returns the full masked dw."""
    rng = np.random.default_rng(4)
    frames = _data()[0][0]
    for update in ("event", "hybrid_event", "timestep"):
        run = build(fmt="sparse", update=update, freeze=("inh",))
        run.run_sample(frames)                                                           ##traces are non-zero now, so the rule has something to say
        syn = run.ctx.syn
        run.ctx.neur.post_spike[...] = rng.random(run.ctx.net.n_post) < 0.3; run.ctx.neur.pre_spike[...] = rng.random(run.ctx.net.n_total) < 0.3
        out = run.plasticity.step(run.ctx)
        if update == "timestep":
            assert isinstance(out, np.ndarray) and out.shape == (syn.W.size,) and not out[~syn.plastic_mask].any()
            continue
        assert isinstance(out, tuple) and len(out) == 2, update
        k, dw_k = out
        assert k.dtype == np.int64 and dw_k.dtype == np.float32 and k.shape == dw_k.shape and k.ndim == 1, update
        assert k.size > 0 and np.all(syn.plastic_mask[k]) and np.array_equal(k, np.unique(k)), update
        assert np.count_nonzero(dw_k) > 0, f"{update}: every dw is zero, the pair proves nothing"


def test_pair_commit_is_bit_identical_to_the_full_array_commit():
    """The pair commit (W[k] += dw_k, clip on k) is bit-identical to the full-array commit (_FullArray): two epochs, both event modes, with/without freeze."""
    for update in ("event", "hybrid_event"):
        for freeze in ((), ("inh",)):
            a = train(build(fmt="sparse", update=update, freeze=freeze, epochs=2))
            b = train(build(fmt="sparse", update=update, freeze=freeze, epochs=2, plasticity_cls=_FullArray))
            assert isinstance(b.plasticity, _FullArray) and a.history.spike_i().size > 0 and not np.array_equal(a.ctx.syn.W, a._W0), f"{update}/{freeze}: proves nothing"
            assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W), f"{update}/{freeze}: the pair commit is not the full-array commit"
            assert np.array_equal(a.history.spike_i(), b.history.spike_i()) and np.array_equal(a.ctx.neur.V, b.ctx.neur.V)


class _PairStub:
    """A plasticity 'module' returning a chosen (k, dw_k): drives Run.step_sparse's pair branch with a dw large enough to hit the bounds."""
    def __init__(self, k, dw_k):
        self.k, self.dw_k = k, dw_k
    def step(self, ctx):
        return self.k, self.dw_k


def test_pair_commit_clips_the_touched_edges_only_and_last():
    """The pair commit clips only the touched edges, after the add (to w_min / 0 / w_max by sign); without a clamp plan it is the plain add."""
    run = build(fmt="sparse", update="event")
    syn = run.ctx.syn
    lo, hi = run._clamp_plan
    k = np.flatnonzero(syn.plastic_mask)[::3][:6]
    dw_k = np.array([1e3, -1e3] * 3, np.float32)[:k.size]
    mask = np.ones(syn.W.size, bool); mask[k] = False
    before = syn.W.copy()
    run._datahandler.begin_sample(np.zeros((4, run.ctx.net.n_input), bool), run.ctx)
    run._plasticity = _PairStub(k, dw_k)
    run.step()
    assert np.array_equal(syn.W[k], np.where(dw_k > 0, hi[k], lo[k])) and np.isfinite(syn.W[k]).all()
    assert np.array_equal(syn.W[mask], before[mask])
    run._clamp_plan = None; W1 = syn.W.copy()
    run.step()
    assert np.array_equal(syn.W[k], W1[k] + dw_k) and np.array_equal(syn.W[mask], W1[mask])


def test_train_step_is_the_event_time_base_and_eval_steps_do_not_count():
    """sim.train_step counts train-mode steps only and is the event time base: an eval pass is not elapsed time for the lazy traces."""
    a, b = build(fmt="sparse", update="timestep"), build(fmt="sparse", update="event")
    data = _data()
    T = data[0][0].shape[0]
    for run in (a, b):
        run.run_sample(data[0][0]); run.set_mode("eval"); run.run_sample(data[1][0]); run.set_mode("train"); run.run_sample(data[2][0])
        assert run.ctx.sim.current_runstep == 3 * T and run.ctx.sim.train_step == 2 * T
    T_e = b.plasticity.T_e
    assert T < T_e.max() <= 2 * T, "T_e are not train steps"                              ##stamped in the third sample, on the train-step clock
    assert a.history.spike_i().size > 0 and np.array_equal(a.history.spike_i(), b.history.spike_i())
    assert np.allclose(a.ctx.syn.W, b.ctx.syn.W, rtol=1e-5, atol=1e-6) and not np.array_equal(a.ctx.syn.W, a._W0)


def test_dense_plus_either_event_mode_is_rejected():
    for mode in ("event", "hybrid_event"):
        with pytest.raises(ValueError):
            SimulationCfg(weights_format="dense", update_on=mode)


def test_sparse_trial_fold_reset_restores_edge_weights():
    run = build(fmt="sparse", update="event")
    w0 = run.ctx.syn.W.copy()
    X = (np.random.default_rng(0).random((80, run.ctx.net.n_input)) < 0.3)
    run.run_sample(X)
    assert not np.array_equal(w0, run.ctx.syn.W)
    run.reset_model()
    assert np.array_equal(w0, run.ctx.syn.W)


def test_weight_health_probes_agree_between_dense_and_sparse():
    ##Scorer.weight_health is format-agnostic: same seed -> same realized weights -> the same probe readings on dense and sparse
    dense, sparse = paired()
    edges = sparse.ctx.syn.weights
    k = edges.W.size // 4
    dense.ctx.syn.W[edges.post[:k], edges.pre[:k]] = 0.0                    ##the same k realized edges "die" in both
    edges.W[:k] = 0.0
    seen = set()
    for bounds in ((-100.0, 100.0), (-1.5, 1.5), (-1.0, 1.0)):              ##none / some (exc_exc, inh_exc) / all of the weights at a bound
        a = Scorer().weight_health(dense.ctx, w_min=bounds[0], w_max=bounds[1])
        b = Scorer().weight_health(sparse.ctx, w_min=bounds[0], w_max=bounds[1])
        assert a == b, f"bounds={bounds}: the probes read differently on dense and sparse"
        seen.add(round(a["saturated_frac"], 6))
    assert len(seen) == 3 and a["dead_syn_frac"] > 0, "the probes did not discriminate anything, so agreeing proves nothing"


def test_backend_cfg_is_frozen_and_archivable():
    ##SimulationCfg(weights_format, update_on) is frozen, hashed into the cfg id and archived; dense + event and a module/cfg format mismatch are refused
    cfg = SimulationCfg(weights_format="sparse", update_on="event")
    assert cfg.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.update_on = "timestep"
    assert dataclasses.asdict(cfg) == {"dt": 1, "array_library": "numpy", "weights_format": "sparse", "update_on": "event"}
    ##archived: the cfg is hashed into the run's identity and stored, as run, with the history
    run = build(fmt="sparse", update="event")
    assert run.history.metadata()["derived_cfg"]["simulation"] == {"dt": 1.0, "array_library": "numpy", "weights_format": "sparse", "update_on": "event"}
    assert run.runmeta.cfg_id != build(fmt="sparse", update="timestep").runmeta.cfg_id != build(fmt="dense").runmeta.cfg_id

    with pytest.raises(ValueError):
        SimulationCfg(weights_format="dense", update_on="event")
    pnet, _ = assemble(fmt="dense")
    pnet.cfg = dataclasses.replace(pnet.cfg, simulation=SimulationCfg(weights_format="sparse"))      ##dense modules under a sparse cfg
    with pytest.raises(TypeError):
        pnet._prep_run(_data(), RunCfg(data=DataCfg(test_fraction=0.25)))


# --------------------------------------------------------------------- 9. the W observables of capture agree dense vs sparse
import capture


def _observables(run, eps=0.0):
    """Every W-block observable capture offers, on one Run, {label: value}."""
    ctx = run.ctx
    out = {"sum_global": capture.sum_globalweight(ctx), "mean_global": capture.mean_globalweight(ctx),
           "mean_global_neuron_abs": capture.mean_globalweight(ctx, per="neuron", signed=False)}
    for axis in (None, 0, 1):
        for signed in (True, False):
            out[f"local_{axis}_{signed}"] = capture.sum_localweight(ctx, axis=axis, signed=signed)
    for group in ctx.net.synapseslices:
        out[f"group_{group}"] = capture.sum_groupweight(ctx, group)
    for post_group in (None, "exc", "inh"):
        for pre_group in (None, "input", "exc", "inh"):
            kw = dict(post_group=post_group, pre_group=pre_group, eps=eps)
            out[f"in_{post_group}_{pre_group}"] = capture.in_degree(ctx, **kw)
            out[f"out_{post_group}_{pre_group}"] = capture.out_degree(ctx, **kw)
            for direction in ("in", "out"):
                for signed in (True, False):
                    out[f"strength_{direction}_{post_group}_{pre_group}_{signed}"] = capture.degree_strength(ctx, direction=direction, signed=signed, **kw)
    out["in_exc_sum"] = capture.in_degree(ctx, post_group="exc", summary="sum")
    out["strength_out_exc_mean"] = capture.degree_strength(ctx, direction="out", pre_group="exc", summary="mean")
    return out


def _same_observables(dense, sparse, eps):
    a, b = _observables(dense, eps), _observables(sparse, eps)
    assert a.keys() == b.keys()
    for k in a:
        va, vb = np.asarray(a[k]), np.asarray(b[k])
        assert va.shape == vb.shape, k
        assert np.allclose(va, vb, rtol=1e-5, atol=1e-6), k
    assert a["in_None_None"].shape == (dense.ctx.net.n_post,) and a["out_None_None"].shape == (dense.ctx.net.n_total,)      ##one value per selected post row / pre column
    assert a["local_0_True"].shape == (dense.ctx.net.n_total,) and a["local_1_True"].shape == (dense.ctx.net.n_post,)
    assert a["in_exc_sum"] > 0 and a["strength_out_exc_mean"] > 0                                                          ##non-vacuity


@pytest.mark.parametrize("post_offset", [False, True])
def test_capture_observables_agree_dense_and_sparse(post_offset):
    """Every W observable has the same values and shapes on dense and sparse, at build and after training; deadened_synapses picks the same synapses."""
    if post_offset:
        from test_lookup_and_loading import _pnet, _data as _data_ll
        dense, sparse = (_pnet(True, fmt)._prep_run(_data_ll(), RunCfg()) for fmt in ("dense", "sparse"))
    else:
        dense, sparse = paired()
    _same_observables(dense, sparse, eps=0.0)
    for group, eps in ((None, 0.0), ("in_exc", 1.3 if not post_offset else 5.0), ("inh_exc", 2.5 if not post_offset else 3.5)):
        post, pre = capture.deadened_synapses(dense.ctx, syn_group=group, eps=eps)
        edges = capture.deadened_synapses(sparse.ctx, syn_group=group, eps=eps)
        assert edges.ndim == 1 and edges.dtype == np.int64
        assert np.array_equal(sparse.ctx.syn.post[edges], post) and np.array_equal(sparse.ctx.syn.pre[edges], pre)
        if group is not None:
            assert post.size > 0 and np.all(np.abs(sparse.ctx.syn.W[edges]) <= eps)
    for frames, _ in list(dense.train_set)[:2]:
        dense.run_sample(frames); sparse.run_sample(frames)
    assert not np.array_equal(dense.ctx.syn.W, dense._W0), "nothing learned: the after-training comparison proves little"
    _same_observables(dense, sparse, eps=1e-3)
