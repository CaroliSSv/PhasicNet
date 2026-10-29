#test_freeze -- freeze is the one cfg knob whose failure is SILENT: a wrong freeze trains what you meant to hold fixed (or the reverse).
# WHAT IS TESTED:
#   1. resolution   -- keywords and group names select the right PROJECTIONS
#   2. equivalence  -- the slice set selects exactly the synapses the elementwise mask would have
#   3. behaviour    -- frozen weights really do not move across a real run
#   4. safety       -- typos raise, overlapping targets are harmless, "all" learns nothing
#   5. orthogonality-- freeze composes with post_offset and never touches network identity
#   6. grammar      -- pre_/post_ prefixes select ONE side of a projection
# Run: pytest system/tests/test_freeze.py -q
from __future__ import annotations
import dataclasses
import numpy as np
from modelinput import ModelInput   ##train() / _prep_run require a ModelInput
import pytest

from phasicnet import PhasicNet
from dynamicscfg import (DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg,
                 PlasticityCfg, NoiseCfg, PhaseCfg)
from runcfg import RunCfg, DataCfg
from protocols import HoldoutValidation
from original_dynamics import DeltaCurrentDense, LIFexact, AdaptiveThreshold, TraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase
from freeze import _build_freeze_slices, _build_dense_plastic_mask, FREEZE_KEYWORDS, _NEURON_ROLES


SEED = 909
N_INPUT, N_EXC, N_INH = 6, 10, 4
GROUPS = ("exc_exc", "exc_inh", "in_exc", "inh_exc")   ##_order_synspecs sorts by name


##PhasicNet is the recipe and holds no state: assemble() returns (pnet, Run), build() returns the Run.
def _data(n_samples=12, T=40):
    X = np.random.default_rng(0).random((n_samples * T, N_INPUT)) < 0.3
    return ModelInput.from_tonic([(X[i * T:(i + 1) * T], i % 2) for i in range(n_samples)])


def assemble(freeze=(), post_offset: bool = False, seed: int = SEED):
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", N_INPUT, True), ("exc", N_EXC), ("inh", N_INH, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"),
                              ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc=0.6, exc_exc=0.3, exc_inh=0.4, inh_exc=0.4)
    pnet.builder.set_weights({"in_exc": 1.2, "exc_exc": 2.2, "exc_inh": 1.1, "inh_exc": 2.4})
    pnet.builder.spec_name = "freeze_spec"
    pnet.builder.realize(network_name="freeze_net", post_offset=post_offset, save=False)


    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=1.0),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=10.0),
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.005, A_ltd=0.006,
                                 eta_exc=1e-3, eta_inh=1e-3, w_min=-10.0, w_max=10.0, freeze=freeze),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))

    pnet.set_runtime_classes(synapse_cls=DeltaCurrentDense, integrator_cls=LIFexact,
                             threshold_cls=AdaptiveThreshold, plasticity_cls=TraceSTDP,
                             noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.cfg_name = pnet.run_name = "freeze"
    return pnet, pnet._prep_run(_data(), RunCfg(data=DataCfg(test_fraction=0.25)))


def build(freeze=(), post_offset: bool = False, seed: int = SEED):
    return assemble(freeze, post_offset, seed)[1]


def train_briefly(run):
    """One real fit; the final state is kept (select="final" restores nothing)."""
    HoldoutValidation(None, run).execute()
    return run


def plastic_names(net, freeze):
    """Recover which GROUPS a freeze spec leaves plastic, by matching blocks back to names."""
    blocks = _build_freeze_slices(net, freeze)
    return {name for name, blk in net.synapseslices.items() if blk in blocks}


# --------------------------------------------------------------------- 1. resolution

def frozen_names(net, freeze):
    """Recover WHICH groups a freeze spec freezes, by matching returned blocks back to names."""
    blocks = _build_freeze_slices(net, freeze)
    return {name for name, blk in net.synapseslices.items() if blk in blocks}


def test_no_freeze_freezes_nothing():
    """() is the common case AND the fast path: zero Python iterations in the commit."""
    net = build().ctx.net
    assert _build_freeze_slices(net, ()) == ()
    assert _build_freeze_slices(net, None) == ()


def test_freeze_all_covers_the_whole_matrix():
    """One rectangle spanning everything, so the commit adds zero everywhere."""
    blocks = _build_freeze_slices(build().ctx.net, ("all",))
    assert blocks == ((slice(None), slice(None)),)


def test_freeze_input_selects_projections_touching_input_nodes():
    """The keyword resolves through the PRE/POST neuron GROUPS, not through a column band."""
    assert frozen_names(build().ctx.net, ("input",)) == {"in_exc"}


def test_freeze_inh_selects_both_sides():
    """inh as PRE (inh->exc) AND inh as POST (exc->inh). Both, or the lateral circuit half-trains."""
    assert frozen_names(build().ctx.net, ("inh",)) == {"exc_inh", "inh_exc"}


def test_freeze_named_group():
    assert frozen_names(build().ctx.net, ("exc_exc",)) == {"exc_exc"}


def test_freeze_targets_combine():
    assert frozen_names(build().ctx.net, ("input", "inh")) == {"in_exc", "exc_inh", "inh_exc"}


def test_block_order_follows_synapsespecs():
    """Deterministic order: reproducibility depends on the commit visiting blocks the same way."""
    net = build().ctx.net
    blocks = _build_freeze_slices(net, ("inh",))
    expected = [net.synapseslices[n] for n in net.synapsespecs
                if n in ("exc_inh", "inh_exc")]
    assert list(blocks) == expected


# --------------------------------------------------------------------- 2. prefix grammar

def test_pre_prefix_selects_only_the_presynaptic_side():
    """pre_inh = inh->exc only. Without the prefix, exc->inh would be frozen too."""
    assert frozen_names(build().ctx.net, ("pre_inh",)) == {"inh_exc"}


def test_post_prefix_selects_only_the_postsynaptic_side():
    assert frozen_names(build().ctx.net, ("post_inh",)) == {"exc_inh"}


def test_bare_role_equals_pre_or_post():
    net = build().ctx.net
    assert frozen_names(net, ("inh",)) == frozen_names(net, ("pre_inh",)) | frozen_names(net, ("post_inh",))


def test_exc_means_not_inhibitory():
    """exc means 'not inh', so a non-inhibitory INPUT group counts as exc (pre_/post_ or a group name tell them apart)."""
    assert frozen_names(build().ctx.net, ("post_exc",)) == {"in_exc", "exc_exc", "inh_exc"}


def test_group_name_wins_over_keyword_parsing():
    """Group names with underscores ('in_exc') are matched FIRST, before side_role keyword parsing."""
    net = build().ctx.net
    assert frozen_names(net, ("in_exc",)) == {"in_exc"}


# --------------------------------------------------------------------- 3. equivalence with a mask

def test_frozen_rectangles_select_the_intended_synapses():
    """The representation is slices; the SET of plastic synapses must still be exactly right."""
    for freeze in ((), ("input",), ("inh",), ("exc_exc",), ("pre_inh",), ("input", "inh"), ("all",)):
        run = build(freeze=freeze)
        syn_mask = run.ctx.syn.syn_mask
        plastic = _build_dense_plastic_mask(run.ctx.syn)
        ##A plastic synapse must be a REALIZED synapse: rectangles are supersets of the graph.
        assert np.array_equal(plastic, plastic & syn_mask), f"{freeze}: plastic escapes syn_mask"

        expected = syn_mask.copy()
        for name, spec in run.ctx.net.synapsespecs.items():
            pre = run.ctx.net.neuronspecs[spec.pre_neuron]
            post = run.ctx.net.neuronspecs[spec.post_neuron]
            frozen = ("all" in freeze or name in freeze
                      or ("input" in freeze and (pre.input or post.input))
                      or ("inh" in freeze and (pre.inh or post.inh))
                      or ("pre_inh" in freeze and pre.inh))
            if frozen:
                rows, cols = run.ctx.net.synapseslices[name]
                expected[rows, cols] = False
        if "all" in freeze:
            expected[...] = False
        assert np.array_equal(plastic, expected), f"{freeze}: block set != expected synapse set"


def test_rectangles_are_supersets_not_the_graph():
    """Rectangles contain UNREALIZED pairs, so a commit without syn_mask would create synapses."""
    run = build()
    rows, cols = run.ctx.net.synapseslices["exc_exc"]
    block_area = (rows.stop - rows.start) * (cols.stop - cols.start)
    assert int(run.ctx.syn.syn_mask[rows, cols].sum()) < block_area


# --------------------------------------------------------------------- 4. behaviour under a run

def test_frozen_projection_weights_do_not_move():
    """THE test. Everything else is bookkeeping; this is the claim freeze actually makes."""
    run = build(freeze=("inh",))
    before = run.ctx.syn.W.copy()
    train_briefly(run)
    after = run.ctx.syn.W

    for name in ("exc_inh", "inh_exc"):
        rows, cols = run.ctx.net.synapseslices[name]
        assert np.array_equal(before[rows, cols], after[rows, cols]), f"{name} was frozen but moved"
    for name in ("in_exc", "exc_exc"):
        rows, cols = run.ctx.net.synapseslices[name]
        assert not np.array_equal(before[rows, cols], after[rows, cols]), f"{name} should have learned"


def test_freeze_all_learns_nothing():
    run = build(freeze=("all",))
    before = run.ctx.syn.W.copy()
    train_briefly(run)
    assert np.array_equal(before, run.ctx.syn.W)


def test_overlapping_freeze_targets_are_harmless():
    """Zeroing dW is idempotent, so naming the same projection twice (directly and via a keyword) is safe."""
    a = build(freeze=("inh",))
    b = build(freeze=("inh", "inh_exc", "pre_inh"))     ##inh_exc named three different ways
    train_briefly(a); train_briefly(b)
    assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W)


def test_unrealized_synapses_never_gain_weight():
    """syn_mask still does its separate job under the new commit."""
    run = build(freeze=("input",))
    mask = run.ctx.syn.syn_mask
    train_briefly(run)
    assert not run.ctx.syn.W[~mask].any(), "weight appeared on a synapse that was never realized"


def test_masked_commit_matches_a_where_commit():
    """`dW *= syn_mask; W += dW` is BIT-IDENTICAL to `np.add(where=syn_mask)`."""
    W = np.arange(12, dtype=np.float32).reshape(3, 4)
    dW = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)
    mask = np.array([[1, 0, 1, 1], [0, 0, 1, 0], [1, 1, 0, 1]], dtype=bool)

    a = W.copy(); np.add(a, dW, where=mask, out=a)
    b = W.copy(); d = dW.copy(); np.multiply(d, mask, out=d); np.add(b, d, out=b)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------- 5. safety valves

def test_unknown_freeze_target_raises():
    """A typo must NOT quietly train what you meant to hold fixed."""
    net = build().ctx.net
    with pytest.raises(KeyError):
        _build_freeze_slices(net, ("exc_exec",))        # transposed characters
    with pytest.raises(KeyError):
        _build_freeze_slices(net, ("inhibitory",))      # old attribute name
    with pytest.raises(KeyError):
        _build_freeze_slices(net, ("pre_banana",))      # good prefix, bad role
    with pytest.raises(KeyError):
        _build_freeze_slices(net, ("mid_inh",))         # bad prefix, good role
    with pytest.raises(KeyError):
        _build_freeze_slices(net, ("in_exc", "nope"))   # one good, one bad -> still raises


def test_every_keyword_and_group_name_is_accepted():
    net = build().ctx.net
    for keyword in FREEZE_KEYWORDS:
        _build_freeze_slices(net, (keyword,))
    for group in net.synapsespecs:
        _build_freeze_slices(net, (group,))


def test_freeze_slices_unresolved_raises_rather_than_defaulting():
    """Skipping _implement_plastic must not silently mean 'nothing frozen'."""
    run = build()
    run.ctx.syn.frozen = None
    with pytest.raises(RuntimeError):
        run.ctx.syn.freeze_slices


# --------------------------------------------------------------------- 6. orthogonality

def test_freeze_composes_with_post_offset():
    """freeze composes with post_offset: blocks come from synapseslices, which the builder already translated."""
    plain, comp = build(freeze=("inh",)), build(freeze=("inh",), post_offset=True)
    assert len(plain.ctx.syn.freeze_slices) == len(comp.ctx.syn.freeze_slices)

    off = comp.ctx.net.n_total - comp.ctx.net.n_post
    for (p_rows, p_cols), (c_rows, c_cols) in zip(plain.ctx.syn.freeze_slices, comp.ctx.syn.freeze_slices):
        assert p_cols == c_cols                                  # columns never shift
        assert (p_rows.start - off, p_rows.stop - off) == (c_rows.start, c_rows.stop)

    train_briefly(plain); train_briefly(comp)
    assert np.array_equal(plain.ctx.syn.W[off:], comp.ctx.syn.W)


def test_freeze_leaves_network_identity_untouched():
    """freeze is cfg, not network state: same topology, same seed -> same network_id."""
    assert build(freeze=()).ctx.net.network_id == build(freeze=("inh", "input")).ctx.net.network_id


def test_refreezing_the_same_network_needs_no_rebuild():
    """_implement_freeze is re-runnable: change the cfg, re-resolve on the SAME ctx, keep the same weights."""
    pnet, run = assemble(freeze=())
    W = run.ctx.syn.W.copy()
    assert run.ctx.syn.freeze_slices == ()

    pnet.cfg = dataclasses.replace(
        pnet.cfg, plasticity=dataclasses.replace(pnet.cfg.plasticity, freeze=("all",)))
    pnet._implement_freeze(run.ctx)
    assert len(run.ctx.syn.freeze_slices) == 1
    assert np.array_equal(W, run.ctx.syn.W), "re-freezing must not disturb learned weights"


def test_freeze_allocates_no_second_dense_array():
    """The point of the representation: no N*N companion to W anywhere in SynapticState."""
    run = build(freeze=("inh",))
    arrays = [v for v in vars(run.ctx.syn.weights).values() if isinstance(v, np.ndarray)]
    assert len(arrays) == 2, f"expected only W and syn_mask, got {len(arrays)} dense arrays"
    for rows, cols in run.ctx.syn.freeze_slices:
        assert isinstance(rows, slice) and isinstance(cols, slice)
