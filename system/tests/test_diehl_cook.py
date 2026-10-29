#test_diehl_cook -- the Diehl & Cook (2015) baseline: the builder patterns, and that the D&C modules do what the authors' Brian 1 script
# does, step by step (checked against dc_brian1_reference.py, an independent transcription of the script and Brian 1's update order).
# Run: pytest system/tests/test_diehl_cook.py -q     (from the repo root)
from __future__ import annotations
##system/tests/conftest.py sets sys.path for pytest
from types import SimpleNamespace
import numpy as np
from modelinput import ModelInput
import pytest

from phasicnet import PhasicNet
from builder import CONNECTIVITY_PATTERNS
from dynamicscfg import SimulationCfg
from diehl_cook_2015 import build, runcfg, dc_append_rest, DT_MS
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg

N_IN, N_EXC = 32, 6
NORM = 78.0 * N_IN / 784


def _net(n_in=N_IN, n_exc=N_EXC, seed=7):
    return build(n_in, n_exc, seed=seed, norm_target=NORM).network


def _samples(n=12, T=40, p=0.15, seed=0):
    rng = np.random.default_rng(seed)
    return ModelInput.from_tonic([(dc_append_rest(rng.random((T, N_IN)) < p, rest_ms=10.0), i % 2) for i in range(n)])


def _run(epochs=1):
    p = build(N_IN, N_EXC, norm_target=NORM)
    rc = RunCfg(protocol=ProtocolCfg(epochs=epochs), data=DataCfg(test_fraction=0.25), checkpoint=CheckpointCfg(to_disk=False),
                scorer=ScorerCfg(rate_hz_min=0.0), history=HistoryCfg(include_input=False))
    return p.train(_samples(), rc, save_run=False)


# ------------------------------------------------------------------ builder patterns
def test_patterns_realize_the_dc_wiring():
    net = _net()
    s, ids = net.synapseslices, (net.pre_idx, net.post_idx + net.n_input)      ##post_offset: rows -> true ids
    ex, ih, inp = net.neuronslices["exc"], net.neuronslices["inh"], net.neuronslices["input"]
    by = {k: (net.syntype == v) for k, v in net.type_key.items()}
    ##all_to_all: every input -> every exc
    assert by["in_exc"].sum() == N_IN * N_EXC
    ##one_to_one: exactly n edges, local i -> i
    pre, post = ids[0][by["exc_inh"]], ids[1][by["exc_inh"]]
    assert pre.size == N_EXC and np.array_equal(pre - ex.start, post - ih.start)
    ##all_but_self: n(n-1) edges, never i -> i
    pre, post = ids[0][by["inh_exc"]], ids[1][by["inh_exc"]]
    assert pre.size == N_EXC * (N_EXC - 1) and not np.any((pre - ih.start) == (post - ex.start))
    ##fixed weights carry Dale's sign
    assert np.all(net.W0[by["exc_inh"]] == np.float32(10.4)) and np.all(net.W0[by["inh_exc"]] == np.float32(-17.0))
    assert net.W0[by["in_exc"]].min() >= np.float32(0.003) and net.W0[by["in_exc"]].max() <= np.float32(0.303)     ##(rand + 0.01) * 0.3

def test_square_patterns_reject_unequal_groups():
    p = PhasicNet(1); p._instantiate_builder(); b = p.builder
    b.set_neurons(("input", 4, True), ("exc", 5), ("inh", 3, False, True))
    b.set_synapses(("in_exc", "input", "exc"), ("exc_inh", "exc", "inh"))
    b.set_connectivity(in_exc="all_to_all", exc_inh="one_to_one"); b.set_weights(in_exc=0.1, exc_inh=1.0)
    with pytest.raises(ValueError):
        b.realize(save=False)
    with pytest.raises(ValueError):
        b.set_connectivity(in_exc="nearest_neighbour")

def test_pattern_edges_do_not_depend_on_the_seed_and_survive_a_spec_round_trip(tmp_path=None):
    a, b = _net(seed=1), _net(seed=2)
    ##in_exc WEIGHTS are drawn (differ by seed); the EDGE LISTS of all three patterns are deterministic
    assert np.array_equal(a.pre_idx, b.pre_idx) and np.array_equal(a.post_idx, b.post_idx) and np.array_equal(a.syntype, b.syntype)
    assert not np.array_equal(a.W0, b.W0)
    ##str p_connection round-trips through the canonical spec dict
    from network import _canonicalize_netspecs, _canon_to_specs
    import json
    raw = json.loads(json.dumps({"spec_name": "x"} | _canonicalize_netspecs(a.neuronspecs, a.synapsespecs)))
    _, _, specs = _canon_to_specs(raw)
    assert {k: v.p_connection for k, v in specs.items()} == {"exc_inh": "one_to_one", "in_exc": "all_to_all", "inh_exc": "all_but_self"}


# ------------------------------------------------------------------ the assembled model
def test_only_input_to_exc_learns_and_stays_in_bounds():
    run = _run(epochs=2)
    syn, net = run.ctx.syn, run.ctx.net
    W0 = np.zeros_like(syn.W); W0[net.post_idx, net.pre_idx] = net.W0
    for fixed in ("exc_inh", "inh_exc"):
        pr = net.synapseslices[fixed]
        assert np.array_equal(syn.W[pr.post, pr.pre], W0[pr.post, pr.pre]), f"{fixed} changed"
    pr = net.synapseslices["in_exc"]
    blk = syn.W[pr.post, pr.pre]
    assert not np.array_equal(blk, W0[pr.post, pr.pre]), "input -> exc did not learn"
    assert blk.min() >= 0.0 and blk.max() <= 1.0 + 1e-6
    assert run.history.spike_i().size > 0

def test_dc_normalise_is_the_scripts_normalize_weights():
    """dc_normalise scales each exc neuron's input weights to the target sum (nothing else, no clipping); the protocol calls it only in training."""
    from diehlcook import dc_normalise
    run = build(N_IN, N_EXC, norm_target=NORM)._prep_run(_samples(), runcfg())
    syn, pr = run.ctx.syn, run.ctx.net.synapseslices["in_exc"]
    W = syn.W.copy(); blk = W[pr.post, pr.pre].astype(np.float64)
    dc_normalise(syn.W, pr, NORM)
    assert np.allclose(syn.W[pr.post, pr.pre], blk * (NORM / blk.sum(axis=1))[:, None], rtol=1e-6)
    assert np.allclose(syn.W[pr.post, pr.pre].sum(axis=1), NORM, rtol=1e-6)
    mask = np.ones_like(W, dtype=bool); mask[pr.post, pr.pre] = False
    assert np.array_equal(syn.W[mask], W[mask])
    ##the protocol normalises before EVERY training presentation, re-presentations included, and never in evaluation
    ##theta is persistent: freeze it before "eval" (the mode contract guards it)
    from diehl_cook_2015 import mnist_input
    from dc_protocol import DCProtocol, DCProtocolCfg
    import dc_protocol
    img = np.zeros((28, 28), np.uint8); img[12:14, 12:14] = 30      ##too weak: several presentations
    data = mnist_input(np.stack([img] * 4), np.arange(4) % 2)
    run = build(784, N_EXC, norm_target=78.0)._prep_run(data, runcfg())
    proto = DCProtocol(DCProtocolCfg(n_train=4, max_repeats=3), run)
    calls = []
    orig = dc_protocol.dc_normalise
    dc_protocol.dc_normalise = lambda *a: (calls.append(1), orig(*a))
    try:
        _, reps = proto.present_example(data, 0, "n")
        assert len(calls) == reps and reps > 1
        run.threshold.freeze(True); run.set_mode("eval")
        calls.clear(); proto.present_example(data, 0, "m")
        assert not calls
    finally:
        dc_protocol.dc_normalise = orig


# ------------------------------------------------------------------ the modules against the independent Brian 1 transcription
from dc_brian1_reference import Brian1DC
from diehlcook import dc_input_delays, dc_normalise
from diehl_cook_2015 import fresh_network
from phases import DefaultPhase


class _Rec(DefaultPhase):
    phase_type = "DEFAULT"
    def post_threshold(self, ctx): self.log.append(ctx.upd.rec_spikes.copy())


def _f64_run(n_in, n_exc, *, f64=True, a_ltd=None):
    """A built D&C Run with the shared arrays switched to float64 (the reference computes in float64, as Brian), recording every step's spikes."""
    p = build(n_in, n_exc, seed=3, norm_target=78.0 * n_in / 784); p.set_runtime_classes(phase_classes=(_Rec,))
    run = p._prep_run(ModelInput.from_tonic([(np.zeros((2, n_in), bool), 0)] * 4), runcfg())
    ctx, s = run.ctx, run.ctx.net.synapseslices
    if f64:
        ctx.syn.weights.W = ctx.syn.W.astype(np.float64); ctx.neur.V = ctx.neur.V.astype(np.float64)
        for k in ("I2V", "dV", "dW"):
            setattr(ctx.upd, k, getattr(ctx.upd, k).astype(np.float64))
        blk = ctx.syn.W[s["exc_inh"].post, s["exc_inh"].pre]; blk[np.eye(n_exc, dtype=bool)] = 10.4      ##the script's 10.4 exactly (W0 is float32)
    if a_ltd is not None:
        run.plasticity.A_ltd = a_ltd
    run.phases["DEFAULT"].log = []
    fresh_network(run)
    return run


def _rows(net, group):
    return slice(net.neuronslices[group].start - net.n_input, net.neuronslices[group].stop - net.n_input)


def _compare_training(*, n_in=60, n_exc=8, nsamp=6, pmax=0.3, target=None, seed=11, f64=True, a_ltd=None, break_ref=None):
    run = _f64_run(n_in, n_exc, f64=f64, a_ltd=a_ltd); ctx, net = run.ctx, run.ctx.net; s = net.synapseslices
    d = dc_input_delays(ctx.rng, ctx.syn.W[s["in_exc"].post, s["in_exc"].pre], 0.0, 10.0, 0.5)[0]
    ref = Brian1DC(ctx.syn.W[s["in_exc"].post, s["in_exc"].pre].astype(np.float64).T.copy(), d.T, theta0=np.full(n_exc, 20.0),
                   nu_pre=1e-4 if a_ltd is None else a_ltd)
    if break_ref is not None:
        break_ref(ref)
    rng = np.random.default_rng(seed); target = target or 78.0 * n_in / 784
    ex, ih = _rows(net, "exc"), _rows(net, "inh")
    first, n_e, n_i, blocked = None, 0, 0, 0
    for k in range(nsamp):
        p_in = rng.random(n_in) * pmax
        frames = np.zeros((1000, n_in), bool); frames[:700] = rng.random((700, n_in)) < p_in
        dc_normalise(ctx.syn.W, s["in_exc"], target); ref.normalize_weights(target)
        start = len(run.phases["DEFAULT"].log)
        run.run_sample(frames)          ##mode "train" is the default
        for t in range(1000):
            e, i = ref.step(frames[t]); rs = run.phases["DEFAULT"].log[start + t]
            n_e += int(e.sum()); n_i += int(i.sum()); blocked += int(((ref.v_e > ref.theta - 72.0) & ~e).sum())
            if first is None and (not np.array_equal(rs[ex], e) or not np.array_equal(rs[ih], i)):
                first = (k, t)
    Wm = ctx.syn.W[s["in_exc"].post, s["in_exc"].pre].T
    return dict(first_mismatch=first, exc=n_e, inh=n_i, blocked=blocked, dW=float(np.abs(Wm - ref.W).max()),
                dV=float(max(np.abs(ctx.neur.V[ex] - ref.v_e).max(), np.abs(ctx.neur.V[ih] - ref.v_i).max())),
                dtheta=float(np.abs(run.threshold.Theta[ex] - ref.theta).max()), w0=float((ref.W == 0).mean()), w1=float((ref.W >= 1).mean()))


def test_training_matches_the_brian1_transcription_step_by_step():
    """Same spikes on EVERY step and the same V, theta and W (float64) as the Brian 1 transcription, clipping at 0 and wmax included."""
    r = _compare_training(pmax=0.3)
    assert r["first_mismatch"] is None and r["exc"] > 200 and r["inh"] > 200 and r["blocked"] > 0, r
    assert r["dW"] < 1e-12 and r["dV"] < 1e-9 and r["dtheta"] < 1e-12, r
    r = _compare_training(pmax=0.3, target=50.0)                       ##weights pushed to wmax: clipping at 1
    assert r["first_mismatch"] is None and r["w1"] > 0.01 and r["dW"] < 1e-12, r
    r = _compare_training(pmax=0.3, a_ltd=0.05)                        ##strong depression: clipping at 0
    assert r["first_mismatch"] is None and r["w0"] > 0.5 and r["dW"] < 1e-12, r


def test_the_comparison_detects_a_wrong_rule():
    """Power check: break one Brian rule in the reference at a time -- the step-by-step comparison must notice."""
    def no_inh_clamp(ref):
        orig = ref.step
        def st(x):
            ref.LS_i = ref.LS_i[-1:]; return orig(x)
        ref.step = st
    def no_delays(ref): ref.delay[:] = 0.0
    def held_exc(ref):                                                 ##v held at reset during exc refractoriness
        orig = ref.step
        def st(x):
            out = orig(x); held = ref.next_e > ref.t - ref.dt; ref.v_e[held] = -65.0; return out
        ref.step = st
    def exact_post_decay(ref): ref._num_decay = lambda tau: np.exp(-ref.dt / tau)          ##Brian's post traces are NOT exact
    def untransposed_ltp(ref): ref._ltp_delay = lambda i, j: ref.delay[i, j]          ##Brian's LTP lag index is transposed
    for brk in (no_inh_clamp, no_delays, held_exc, exact_post_decay, untransposed_ltp):
        r = _compare_training(pmax=0.3, nsamp=3, break_ref=brk)
        assert r["first_mismatch"] is not None or r["dW"] > 1e-12, brk.__name__


def test_test_mode_matches_the_brian1_transcription():
    """The script's test mode: a new network (V = v_rest - 40, its own delays), theta loaded and fixed, no STDP, no normalisation."""
    n_in, n_exc = 60, 8
    run = _f64_run(n_in, n_exc); ctx, net = run.ctx, run.ctx.net; s = net.synapseslices
    rng = np.random.default_rng(5)
    blk = ctx.syn.W[s["in_exc"].post, s["in_exc"].pre]; blk[...] = rng.random(blk.shape) * 0.3; blk[rng.random(blk.shape) < 0.2] = 0.0
    ex, ih = _rows(net, "exc"), _rows(net, "inh")
    run.threshold.Theta[ex] = rng.random(n_exc) * 30 + 20
    run.threshold.freeze(True); fresh_network(run, redraw_delays=True)
    run.set_mode("eval")
    d = dc_input_delays(ctx.rng, blk, 0.0, 10.0, 0.5)[0]
    ref = Brian1DC(blk.astype(np.float64).T.copy(), d.T, theta0=run.threshold.Theta[ex].copy(), test_mode=True)
    w = ctx.syn.W.copy(); n_e = 0
    for k in range(5):
        p_in = rng.random(n_in) * 0.3
        frames = np.zeros((1000, n_in), bool); frames[:700] = rng.random((700, n_in)) < p_in
        start = len(run.phases["DEFAULT"].log); run.run_sample(frames)
        for t in range(1000):
            e, i = ref.step(frames[t]); rs = run.phases["DEFAULT"].log[start + t]; n_e += int(e.sum())
            assert np.array_equal(rs[ex], e) and np.array_equal(rs[ih], i), (k, t)
    assert n_e > 50 and np.array_equal(w, ctx.syn.W) and np.array_equal(run.threshold.Theta[ex], ref.theta)


def test_float32_arrays_stay_on_the_reference_spike_train():
    """With the real float32 shared arrays (deviation 1) the spike trains still match the float64 transcription on this scale; W within 1e-6."""
    r = _compare_training(pmax=0.3, nsamp=8, f64=False)
    assert r["first_mismatch"] is None and r["dW"] < 1e-6, r


def test_delays_are_the_scripts_draw():
    """Synapse and plasticity share the script's delay draw; a redraw over a W with zeros shifts the stream past each zero; LTP reads it transposed."""
    run = _f64_run(40, 5); ctx, pr = run.ctx, run.ctx.net.synapseslices["in_exc"]
    run.run_sample(np.zeros((3, 40), bool))          ##mode "train" is the default
    d, K, LTD, KLTP, R = dc_input_delays(ctx.rng, ctx.syn.W[pr.post, pr.pre], 0.0, 10.0, 0.5)
    assert run.synapse.Delay_drawn[0] and run.plasticity.Delay_drawn[0] and R == 21
    assert np.array_equal(run.synapse.K, K) and np.array_equal(run.plasticity.LTD, LTD) and np.array_equal(run.plasticity.KLTP, KLTP)
    assert 0 <= K.min() and K.max() <= 19 and np.all(LTD == 20 - K)
    Db = d.T                                                                       ##Brian layout (n_pre, n_post)
    assert all(KLTP[j, i] == int(2000.0 * Db.reshape(-1)[i + j * 5]) for j in range(5) for i in range(40))
    blk = ctx.syn.W[pr.post, pr.pre]; blk[2, 7] = 0.0                              ##exc 2 <- input 7 trained to exactly 0
    fresh_network(run, redraw_delays=True)
    run.threshold.freeze(True); run.set_mode("eval"); run.run_sample(np.zeros((3, 40), bool))     ##the test state: frozen theta + mode "eval"
    d2 = dc_input_delays(ctx.rng, blk, 0.0, 10.0, 0.5)[0].T                        ##Brian layout
    flat1, flat2 = Db.reshape(-1), d2.reshape(-1); cut = 7 * 5 + 2
    assert np.array_equal(flat2[:cut], flat1[:cut]) and flat2[cut] == 0.0 and np.array_equal(flat2[cut + 1:], flat1[cut:-1])
    assert np.array_equal(run.synapse.K, np.array(2000.0 * d2, dtype=np.int64).T)


# ------------------------------------------------------------------ the protocol (dc_reproduce.py): theta freeze, readout, re-presentation, resume
def test_frozen_theta_neither_decays_nor_adapts():
    run = build(N_IN, N_EXC, norm_target=NORM)._prep_run(_samples(), runcfg())
    run.threshold.freeze(True)
    theta = run.threshold.Theta.copy()
    run._datahandler.begin_sample(np.ones((60, N_IN), dtype=bool), run.ctx)
    run.set_mode("eval")
    for _ in range(60):
        run.step()
    assert np.array_equal(theta, run.threshold.Theta)        ##full drive: unfrozen, theta would both decay and jump on spikes

def test_readout_assigns_by_mean_response_and_predicts_by_class_mean():
    from scorer import assign_neuron_labels as dc_assign, vote_predict as dc_predict      ##the readout lives in scorer
    labels = np.array([0, 0, 1, 1, 2, 2])
    resp = np.array([[5, 0, 0, 1], [3, 0, 0, 1], [0, 4, 0, 1], [0, 6, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4]])
    assign = dc_assign(resp, labels, n_classes=3)
    assert assign.tolist() == [0, 1, -1, 2]                  ##neuron 2 never fired -> unassigned, not class 0
    pred = dc_predict(np.array([[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 0, 9], [0, 0, 7, 0]]), assign, n_classes=3)
    assert pred.tolist()[:3] == [0, 1, 2]
    assert pred[3] == 0                                      ##all assigned neurons silent: first class with a neuron, never an empty class

def test_dc_readout_is_the_authors_code():
    """scorer.dc_assign_labels / dc_rank_predict equal literal copies of the authors' get_new_assignments / get_recognized_number_ranking."""
    from scorer import dc_assign_labels, dc_rank_predict
    def gna(rm, inp, n_e, init):
        a = np.ones(n_e) * init; mx = [0] * n_e
        for j in range(10):
            n = len(np.where(inp == j)[0])
            if n > 0: rate = np.sum(rm[inp == j], axis=0) / n
            for i in range(n_e):
                if rate[i] > mx[i]: mx[i] = rate[i]; a[i] = j
        return a
    def rank(a, r):
        s = [0] * 10; na = [0] * 10
        for i in range(10):
            na[i] = len(np.where(a == i)[0])
            if na[i] > 0: s[i] = np.sum(r[a == i]) / na[i]
        return np.argsort(s, kind="stable")[::-1]
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_e = int(rng.integers(3, 15)); rm = rng.integers(0, 3, (60, n_e)).astype(float); lab = rng.permutation(np.arange(60) % 10)
        for init in (-1, 0):
            assert np.array_equal(gna(rm, lab, n_e, init), dc_assign_labels(rm, lab, unassigned=init))
        a = dc_assign_labels(rm, lab); te = rng.integers(0, 3, (40, n_e)).astype(float)
        assert np.array_equal([rank(a, r)[0] for r in te], dc_rank_predict(te, a))

def test_weak_example_is_re_presented_until_five_exc_spikes_in_one_window():
    """A weak example is re-presented until >= 5 exc spikes in one window (the script's rule); the stored response is that window only."""
    ##a window = exc spikes of the presentation + the carried rest count, recorded from Counts around each run_sample call
    from diehl_cook_2015 import mnist_input
    from dc_protocol import DCProtocol, DCProtocolCfg
    img = np.zeros((28, 28), np.uint8); img[12:16, 12:16] = 60                        ##16 dim pixels: far too weak for 5 spikes at intensity 2
    data = mnist_input(np.stack([img] * 4), np.arange(4) % 2)
    run = build(784, N_EXC, norm_target=78.0)._prep_run(data, runcfg()); fresh_network(run)
    proto = DCProtocol(DCProtocolCfg(n_train=4, max_repeats=60), run)
    run.threshold.freeze(True); run.set_mode("eval")
    windows, ex = [], run.ctx.net.neuronslices["exc"]
    orig = run.run_sample
    def rs(frames, target=None, *, until=None):                                       ##record each window as the protocol forms it
        before = run.history.Counts.copy(); out = orig(frames, target, until=until)
        windows.append(int((run.history.Counts - before)[ex].sum() + proto.Carry.sum())); return out
    run.run_sample = rs
    resp, reps = proto.present_example(data, 0, "t")
    assert reps > 1 and (resp.sum() >= 5 or reps == 60)
    assert all(w < 5 for w in windows[:-1]) and resp.sum() == windows[-1]             ##earlier presentations failed on their OWN window

def test_protocol_refuses_reset_between_samples():
    """DCProtocol refuses reset_between_samples=True (the script never resets between presentations)."""
    from diehl_cook_2015 import mnist_input
    from dc_protocol import DCProtocol, DCProtocolCfg
    run = build(784, N_EXC, norm_target=78.0)._prep_run(mnist_input(np.zeros((4, 28, 28), np.uint8), np.arange(4) % 2), runcfg())
    with pytest.raises(ValueError, match="reset_between_samples"):
        DCProtocol(DCProtocolCfg(n_train=4, reset_between_samples=True), run)
    DCProtocol(DCProtocolCfg(n_train=4), run)                                           ##the default (False) builds

def test_protocol_resume_is_bit_identical(monkeypatch):
    import shutil
    from dc_reproduce import main as dc_main, RUNS_DIR
    from checkpoint import CheckpointPolicy, snapshots
    ##needs real MNIST at RAW_DIR/mnist.npz
    ##run b is killed after its first snapshot (train 3) and resumed; a and b must match at the end of training and of the protocol
    common = dict(n_exc=8, n_train=6, n_assign=4, n_test=4, save_every=3, report_every=3)
    orig = CheckpointPolicy.on_sample_end
    def kill_after_first(policy, **kw):
        path = orig(policy, **kw)
        if path is not None:
            raise KeyboardInterrupt
        return path
    a, b = RUNS_DIR / "DC_pytest_dc_a", RUNS_DIR / "DC_pytest_dc_b"
    try:
        dc_main(tag="pytest_dc_a", **common)
        monkeypatch.setattr(CheckpointPolicy, "on_sample_end", kill_after_first)
        with pytest.raises(KeyboardInterrupt):
            dc_main(tag="pytest_dc_b", **common)
        monkeypatch.undo()
        assert [p.name for p in snapshots(b)] == ["snap_s00_i00000003.npz"]
        dc_main(tag="pytest_dc_b", **common, from_snapshot=str(b / "snap_s00_i00000003.npz"))
        for fa, fb in ((next(a.glob("TrainedModel_*.npz")), next(b.glob("TrainedModel_*.npz"))), (snapshots(a)[-1], snapshots(b)[-1])):
            za, zb = np.load(fa), np.load(fb)
            assert fa.name == fb.name and all(np.array_equal(za[k], zb[k]) for k in za.files if k != "meta"), fa.name
    finally:
        for t in ("pytest_dc_a", "pytest_dc_b"):
            shutil.rmtree(RUNS_DIR / f"DC_{t}", ignore_errors=True)

def test_protocol_saves_everything_and_scores_match_the_stored_responses():
    """A finished protocol run saves its files, protocol_scores recomputes the logged accuracy, load_trained restores the weights; no stdout."""
    import io, json, shutil, contextlib
    from dc_reproduce import main as dc_main, RUNS_DIR, load_trained
    from analysis import protocol_scores
    tag = "pytest_dc_save"
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = dc_main(tag=tag, n_exc=8, n_train=6, n_assign=6, n_test=6, test_passes=2)   ##needs real MNIST
        assert buf.getvalue() == "", "protocol printed to stdout"
        d = RUNS_DIR / f"DC_{tag}"
        for f in ("run.jsonl", "results.jsonl", "status.json", "network.json", "network.npz", "cfg.json", "cfg_derived.json", "runcfg.json", "runmeta.json", "protocol.json",
                  "results.json", "responses.npz"):
            assert (d / f).exists(), f
        assert list(d.glob("TrainedModel_*.npz")) and "RUN COMPLETE" in (d / "run.jsonl").read_text()
        assert json.loads((d / "status.json").read_text())["status"] == "completed"
        sc = protocol_scores(d)
        assert sc["accuracy_per_pass"] == res["accuracy_per_pass"] and "ch_test_per_pass" not in sc and "ch_test_per_pass" not in res
        run = load_trained(d)
        from checkpoint import snapshots
        final = np.load(snapshots(d)[-1])      ##the newest by (stage, position)
        assert np.array_equal(run.ctx.syn.W, final["W"]) and np.array_equal(run.threshold.Theta, final["mod__threshold__Theta"])
    finally:
        shutil.rmtree(RUNS_DIR / f"DC_{tag}", ignore_errors=True)
