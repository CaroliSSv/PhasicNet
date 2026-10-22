#test_compression - post_offset (level 1, dense "rect") must be a PURE optimization: dropping structurally dead post rows
# must not change any OBSERVABLE.
# Run: pytest system/tests/test_compression.py -q
from __future__ import annotations
import numpy as np
from modelinput import ModelInput   ##train() / _prep_run require a ModelInput
import pytest

from phasicnet import PhasicNet
from cfg import (Cfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg,
                 PlasticityCfg, NoiseCfg, PhaseCfg)
from runcfg import RunCfg, DataCfg, ScorerCfg, ProtocolCfg
from protocols import HoldoutValidation
from original_dynamics import DeltaCurrentDense, LIFexact, AdaptiveThreshold, TraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase
import capture

SEED = 8181
N_INPUT, N_EXC, N_INH = 8, 12, 5


##_prep_run needs a dataset to split: tiny random binary input.
def sample_data(n_samples=12, T=40):
    X = (np.random.default_rng(0).random((n_samples * T, N_INPUT)) < 0.3)
    return ModelInput.from_tonic([(X[i * T:(i + 1) * T], i % 2) for i in range(n_samples)])


def build(post_offset: bool = False, seed: int = SEED, sigma_mem: float = 0.0, epochs: int = 1):
    ##returns the Run; epochs is a protocol setting, so it is given here
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", N_INPUT, True), ("exc", N_EXC), ("inh", N_INH, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"),
                              ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc=0.6, exc_exc=0.3, exc_inh=0.4, inh_exc=0.4)
    pnet.builder.set_weights({"in_exc": 1.2, "exc_exc": 2.2, "exc_inh": 1.1, "inh_exc": 2.4})
    pnet.builder.spec_name = "compress_spec"
    pnet.network = pnet.builder.realize(network_name="compress_net", post_offset=post_offset, save=False)

    pnet.cfg = Cfg(
        simulation=SimulationCfg(dt=1.0),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=10.0),
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.005, A_ltd=0.006,
                                 eta_exc=1e-3, eta_inh=1e-3, w_min=-10.0, w_max=10.0),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=sigma_mem),
        phases=(PhaseCfg(phase_type="DEFAULT"),))

    pnet.set_runtime_classes(synapse_cls=DeltaCurrentDense, integrator_cls=LIFexact,
                             threshold_cls=AdaptiveThreshold, plasticity_cls=TraceSTDP,
                             noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    ##rate_hz_min=0.0: a silence flag would stop a fit after epoch 1 and make the multi-epoch tests single-epoch
    return pnet._prep_run(sample_data(), RunCfg(protocol=ProtocolCfg(epochs=epochs), scorer=ScorerCfg(rate_hz_min=0.0), data=DataCfg(test_fraction=0.25)))


def run(run_):
    ##ONE fit over the train split, shuffled per epoch by the keyed fold rng, identical in both modes
    proto = HoldoutValidation(run_.runcfg.protocol, run_); proto.execute()
    assert len(proto.ops.epoch_log) == run_.runcfg.protocol.epochs, "a fit stopped early: the comparison would no longer cover the epochs it claims to"
    return run_


# --------------------------------------------------------------------- namespace round-trips

def test_post_offset_equals_n_input():
    """For this topology (only input never receives) offset IS n_input -- the whole point."""
    comp, plain = build(post_offset=True).ctx.net, build(post_offset=False).ctx.net
    assert (comp.n_total - comp.n_post) == plain.n_input
    assert (plain.n_total - plain.n_post) == 0   # uncompressed: nothing dropped


def test_compression_is_baked_into_the_realized_network():
    """post_idx and post_slices are ALREADY remapped: the Network describes itself, so no runtime module needs an offset."""
    plain, comp = build().ctx.net, build(post_offset=True).ctx.net
    assert comp.post_offset and not plain.post_offset
    assert comp.post_idx.min() == plain.post_idx.min() - (comp.n_total - comp.n_post)
    assert comp.n_post == plain.n_total - plain.n_input
        # synapseslices POST half is remapped, PRE half untouched
    ##WeightProjection: positional unpack is still (post, pre), and named access says so.
    proj = comp.synapseslices["in_exc"]
    post, pre = proj
    assert (post, pre) == (proj.post, proj.pre)
    assert post.start == plain.neuronslices["exc"].start - (comp.n_total - comp.n_post)
    assert pre == comp.neuronslices["input"]


def test_compressed_rows_cover_all_realized_synapses():
    """The dropped band must contain NO realized synapse -- otherwise compression loses edges."""
    ##the dropped band is the COMPRESSED net's offset: no uncompressed edge ends inside it, and every edge is kept
    plain, comp = build().ctx.net, build(post_offset=True).ctx.net
    offset = comp.n_total - comp.n_post
    assert offset > 0
    assert plain.post_idx.min() >= offset
    assert comp.n_synapses == plain.n_synapses
    assert np.array_equal(comp.post_idx + offset, plain.post_idx)


def test_dead_rows_are_actually_empty_uncompressed():
    """The rows compression removes are all-zero in the UNCOMPRESSED build: compression is lossless."""
    plain = build(post_offset=False)
    offset = (lambda m: m.n_total - m.n_post)(build(post_offset=True).ctx.net)
    assert offset > 0
    assert not plain.ctx.syn.W[:offset].any()
    assert not plain.ctx.syn.syn_mask[:offset].any()


# --------------------------------------------------------------------- shapes

def test_shapes_shrink_only_in_rows():
    a, b = build(False), build(True)
    offset = b.ctx.net.n_total - b.ctx.net.n_post
    n_total = a.ctx.net.n_total

    assert a.ctx.syn.W.shape == (n_total, n_total)
    assert b.ctx.syn.W.shape == (n_total - offset, n_total)
    ##ltd is the TraceSTDP scratch (plasticity._ltd, shape (n_post, n_total))
    for array in (b.ctx.upd.dW, b.plasticity._ltd):
        assert array.shape == (n_total - offset, n_total)
    assert b.ctx.net.n_total - b.ctx.net.n_post == offset
    assert a.ctx.net.n_total - a.ctx.net.n_post == 0

    ##Only PRE-side vectors stay NEURON space; anything with a membrane is POST space.
    assert b.ctx.neur.Spikes.shape == (n_total,)
    for array in (b.ctx.neur.V, b.ctx.upd.I2V, b.ctx.upd.dV, b.ctx.upd.rec_spikes):
        assert array.shape == (n_total - offset,)


# --------------------------------------------------------------------- bit-identical runs

def test_run_is_bit_identical():
    """THE test for the DETERMINISTIC maths: compressed changes nothing (sigma_mem=0, see test_noise_consumption_differs_between_modes)."""
    a = run(build(post_offset=False))
    b = run(build(post_offset=True))
    offset = b.ctx.net.n_total - b.ctx.net.n_post

    sa, sb = {"spike_i": a.history.spike_i()}, {"spike_i": b.history.spike_i()}
    assert sa["spike_i"].size > 0, "no spikes at all: the comparison below proves nothing"
    assert np.array_equal(sa["spike_i"], sb["spike_i"]), "spike neuron indices differ"
    assert np.array_equal(a.history.spike_ptr(), b.history.spike_ptr()), "spike timing differs"

    assert not np.array_equal(a.ctx.syn.W, a._W0), "no learning happened, so the weight comparison proves little"
    assert np.array_equal(a.ctx.syn.W[offset:], b.ctx.syn.W), "learned weights differ"
    assert np.array_equal(a.ctx.neur.V[offset:], b.ctx.neur.V), "membrane potentials differ"
    assert np.array_equal(a.ctx.neur.V_th[offset:], b.ctx.neur.V_th), "thresholds differ"


def test_multi_epoch_bit_identical():
    a = run(build(post_offset=False, epochs=3))
    b = run(build(post_offset=True, epochs=3))
    assert np.array_equal(a.history.spike_i(),
                          b.history.spike_i())
    assert np.array_equal(a.ctx.syn.W[b.ctx.net.n_total - b.ctx.net.n_post:], b.ctx.syn.W)


def test_no_dead_head_exists():
    """I2V is POST space, so there is no region of it that nothing writes (it stays finite)."""
    b = run(build(post_offset=True))
    assert b.ctx.upd.I2V.shape == (b.ctx.net.n_post,)
    assert np.all(np.isfinite(b.ctx.upd.I2V))
    assert np.all(np.isfinite(b.ctx.neur.V))


# --------------------------------------------------------------------- capture agreement

def test_capture_methods_agree():
    """Capture indexes W by true_neuronID post slices; compression must not silently shift them."""
    a, b = run(build(False)), run(build(True))
    offset = b.ctx.net.n_total - b.ctx.net.n_post

    assert np.isclose(capture.sum_globalweight(a.ctx), capture.sum_globalweight(b.ctx))

    for group in ("in_exc", "exc_exc", "exc_inh", "inh_exc"):
        assert np.isclose(capture.sum_groupweight(a.ctx, group),
                          capture.sum_groupweight(b.ctx, group)), f"{group} weight sum differs"

    for post_group in ("exc", "inh"):
        assert np.isclose(capture.in_degree(a.ctx, post_group=post_group, summary="sum"),
                          capture.in_degree(b.ctx, post_group=post_group, summary="sum"))
        assert np.isclose(capture.degree_strength(a.ctx, post_group=post_group, summary="sum"),
                          capture.degree_strength(b.ctx, post_group=post_group, summary="sum"))


def test_deadened_synapse_coords_map_back_to_true_ids():
    """Coordinates are consumed by SleepPhase to index W, so they must be namespace-stable."""
    ##the pair indexes W directly (post = W row, pre = true neuron id): row + offset is the true post id in both modes
    ##eps=1.3 (> w0=1.2 of in_exc) makes every non-potentiated in_exc synapse count as deadened, so the sets are non-empty
    a, b = run(build(False)), run(build(True))
    offset = b.ctx.net.n_total - b.ctx.net.n_post
    post_a, pre_a = capture.deadened_synapses(a.ctx, syn_group="in_exc", eps=1.3)
    post_b, pre_b = capture.deadened_synapses(b.ctx, syn_group="in_exc", eps=1.3)
    assert post_a.size > 0
    assert np.array_equal(post_a, post_b + offset)
    assert np.array_equal(pre_a, pre_b)
    assert np.array_equal(a.ctx.syn.W[post_a, pre_a], b.ctx.syn.W[post_b, pre_b])
    post_a, pre_a = capture.deadened_synapses(a.ctx, syn_group="inh_exc")           ##the default eps=0.0 too: the same agreement on whatever it finds (possibly nothing)
    post_b, pre_b = capture.deadened_synapses(b.ctx, syn_group="inh_exc")
    assert np.array_equal(post_a, post_b + offset)
    assert np.array_equal(pre_a, pre_b)


# --------------------------------------------------------------------- opt-in / safety

def test_default_is_uncompressed():
    assert build().ctx.syn.W.shape[0] == build().ctx.net.n_total


def test_incompressible_network_falls_back():
    """An input group WITH an incoming projection: post_offset must fall back, not drop rows that hold real synapses."""
    pnet = PhasicNet(SEED)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", 6, True), ("exc", 6, False))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_in", "exc", "input"))
    pnet.builder.set_connectivity(in_exc=0.5, exc_in=0.5)
    pnet.builder.set_weights({"in_exc": 1.0, "exc_in": 1.0})
    pnet.builder.spec_name = "s"
    pnet.network = pnet.builder.realize(network_name="n", post_offset=True, save=False)

    assert pnet.network.post_offset is False      # falls back, does not raise
    ##offset_fallback=False turns the silent fallback into a refusal
    with pytest.raises(RuntimeError):
        pnet.builder.realize(network_name="n", post_offset=True, offset_fallback=False, save=False)


def test_noise_consumption_differs_between_modes():
    """DOCUMENTED CONSEQUENCE: compressed noise draws n_post values per step (not n_total), so from the same seed the two modes are different runs once noise is on."""
    plain, comp = build(sigma_mem=0.5), build(post_offset=True, sigma_mem=0.5)
    assert plain.ctx.upd.dV.shape[0] == plain.ctx.net.n_total
    assert comp.ctx.upd.dV.shape[0] == comp.ctx.net.n_post
    assert comp.ctx.upd.dV.shape[0] < plain.ctx.upd.dV.shape[0]

    # deterministic maths still matches when noise is off -- that is what makes the above a
    # stream-consumption difference rather than a maths difference
    a = run(build(sigma_mem=0.0))
    b = run(build(post_offset=True, sigma_mem=0.0))
    assert np.array_equal(a.history.spike_i(),
                          b.history.spike_i())


def test_state_arrays_use_the_right_coordinate_space():
    """NEURON space (n_total) for pre-side, POST space (n_post) for anything with a membrane."""
    net_ = build(post_offset=True)
    n_total, n_post = net_.ctx.net.n_total, net_.ctx.net.n_post
    assert n_post < n_total

    ##the traces are module-owned (plasticity.Pre_trace / Post_trace)
    for array in (net_.ctx.neur.Spikes, net_.ctx.neur.pre_spike, net_.plasticity.Pre_trace):
        assert array.shape == (n_total,), "pre-side array must stay NEURON space"
    ##step_input is neither space: it holds ONLY the input nodes, straight from the dataloader.
    assert net_.ctx.upd.step_input.shape == (net_.ctx.net.n_input,)

    for array in (net_.ctx.neur.V, net_.ctx.neur.V_th, net_.ctx.upd.I2V, net_.ctx.upd.dV,
                  net_.ctx.upd.rec_spikes, net_.ctx.upd.dV_th,
                  net_.ctx.neur.post_spike, net_.plasticity.Post_trace):
        assert array.shape == (n_post,), "post-side array must be POST space"


def test_rec_slice_maps_between_spaces():
    plain, comp = build().ctx.net, build(post_offset=True).ctx.net
    assert comp.rec_slice == slice(0, comp.n_post)
    assert plain.rec_slice == slice(plain.n_input, plain.n_total)
    assert (comp.rec_slice.stop - comp.rec_slice.start) == \
           (plain.rec_slice.stop - plain.rec_slice.start) == plain.n_rec
