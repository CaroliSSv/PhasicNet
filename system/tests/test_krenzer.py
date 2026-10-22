#test_krenzer -- Krenzer & Bogdan 2025 RLSM (krenzer2025.py, experiments/krenzer_2025.py): the wiring and the alpha-scaled initial W_ex, the R-STDP rule step by
# step against a hand recursion (pre trace, Eq. 4 with mu, Eq. 5 scaling, eligibility decay, Eq. 8/9 feedback, auto k), dense == sparse, and the protocol
# end to end on synthetic input.
# Run: pytest system/tests/test_krenzer.py -q     (from the repo root)
from __future__ import annotations
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
from modelinput import ModelInput   ##train() / _prep_run require a ModelInput
import pytest

from krenzer2025 import RSTDPCfg, RSTDPEligibility, RSTDPEligibilitySparse, scale_rows_to
from krenzer_2025 import build, runcfg, ALPHA, ETA, X_TAR, W_MAX, TAU_PRE, TAU_TRACE

N_IN, N_LIQ, N_WTA, N_WTA_INH = 10, 30, 4, 2


def _build(**kw):
    kw.setdefault("n_input", N_IN); kw.setdefault("n_liquid", N_LIQ); kw.setdefault("n_wta", N_WTA); kw.setdefault("n_wta_inh", N_WTA_INH)
    return build(**kw)


def _samples(n=8, T=60, seed=0):
    r = np.random.default_rng(seed)
    return ModelInput.from_tonic([(r.random((T, N_IN)) < 0.1, k % 2) for k in range(n)])


def _run(pnet, data=None):
    data = _samples() if data is None else data
    return pnet._prep_run(data, runcfg(n_train=4, n_test=4), test_data=data.subset(np.arange(4)))


def _wex(run):
    """The R-STDP block (n_wta, n_exc1) as float64, dense or sparse."""
    net, syn = run.ctx.net, run.ctx.syn
    proj = net.synapseslices["exc1_wta"]
    if syn.W.ndim == 2:
        return syn.W[proj.post, proj.pre].astype(np.float64)
    m = (syn.post >= proj.post.start) & (syn.post < proj.post.stop) & (syn.pre >= proj.pre.start) & (syn.pre < proj.pre.stop)
    out = np.zeros((proj.post.stop - proj.post.start, proj.pre.stop - proj.pre.start))
    out[syn.post[m] - proj.post.start, syn.pre[m] - proj.pre.start] = syn.W[m]
    return out


def _E(run):
    net, syn, pl = run.ctx.net, run.ctx.syn, run.plasticity
    proj = net.synapseslices["exc1_wta"]
    if isinstance(pl, RSTDPEligibilitySparse):
        m = (syn.post >= proj.post.start) & (syn.post < proj.post.stop) & (syn.pre >= proj.pre.start) & (syn.pre < proj.pre.stop)
        out = np.zeros((proj.post.stop - proj.post.start, proj.pre.stop - proj.pre.start))
        out[syn.post[m] - proj.post.start, syn.pre[m] - proj.pre.start] = pl.E[m]
        return out
    return pl.E.copy()


def _drive(run, frames, force=None):
    """Step through one sample by hand. force = {step: wta_local_rows}: V of those WTA rows is set to 20 (>= v_th 15) before the step -> a spike in that step.
    Per step: copies of pre_spike, the WTA spikes, W_ex, E, the WTA block of dW (dense only)."""
    ctx, net = run.ctx, run.ctx.net
    proj = net.synapseslices["exc1_wta"]
    run._datahandler.begin_sample(np.asarray(frames, bool), ctx)
    out = []
    for m in range(len(frames)):
        for r in (force or {}).get(m, ()):
            ctx.neur.V[proj.post.start + r] = 20.0
        run.step()
        out.append(dict(pre=ctx.neur.pre_spike.copy(), S=ctx.neur.Spikes[net.n_input:][proj.post].copy(), W=_wex(run), E=_E(run), X=run.plasticity.X.copy()))
    return out


# ------------------------------------------------------------------ wiring
def test_build_wires_two_liquids_a_wta_and_scales_wex_to_alpha():
    pnet = _build(); net = pnet.network
    n_exc, n_inh = 24, 6
    for g, n in (("exc1", n_exc), ("inh1", n_inh), ("wta", N_WTA), ("wta_inh", N_WTA_INH), ("exc2", n_exc), ("inh2", n_inh), ("input", N_IN)):
        assert net.neuronslices[g].stop - net.neuronslices[g].start == n, g
    assert net.synapse_counts["n_exc1_wta"] == n_exc * N_WTA and net.synapse_counts["n_wta_wtainh"] == N_WTA * N_WTA_INH and net.synapse_counts["n_wtainh_wta"] == N_WTA * N_WTA_INH
    by = {k: net.syntype == v for k, v in net.type_key.items()}
    assert (net.W0[by["wtainh_wta"]] < 0).all() and (net.W0[by["inh1_exc1"]] < 0).all() and (net.W0[by["exc1_wta"]] >= 0).all()
    rows = net.post_idx[by["exc1_wta"]] - net.synapseslices["exc1_wta"].post.start
    assert np.allclose(np.bincount(rows, weights=net.W0[by["exc1_wta"]].astype(np.float64)), ALPHA, atol=1e-3)     ##Eq. 5 at initialisation
    net.rehash_network()
    assert net.neuronspecs["exc1"].grid == (15, 3, 3) and net.neuronspecs["exc1"].grid_share == "liquid1" and net.neuronspecs["exc2"].grid_share == "liquid2"
    ##only the R-STDP projection learns
    run = _run(pnet)
    assert set(run.plasticity.freeze) == set(net.synapsespecs) - {"exc1_wta"}
    assert run.ctx.syn.plastic_mask.sum() == n_exc * N_WTA
    assert isinstance(run.plasticity, RSTDPEligibilitySparse)
    run_d = _run(_build(weights_format="dense"))
    assert isinstance(run_d.plasticity, RSTDPEligibility) and not isinstance(run_d.plasticity, RSTDPEligibilitySparse)
    assert np.allclose(_wex(run_d).sum(1), ALPHA, atol=1e-3)


def test_scale_rows_to_and_bad_cfgs():
    W = np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
    scale_rows_to(W, 15.0)
    assert np.allclose(W, [[3.75, 11.25], [0.0, 0.0], [7.5, 7.5]])
    for bad in (dict(feedback="maybe"), dict(mu=-1.0), dict(alpha=0.0), dict(tau_trace=0.0)):
        with pytest.raises(ValueError):
            _run(_build(**bad))


# ------------------------------------------------------------------ the rule, step by step (dense)
def _hand(out, run, mu, force_steps, net):
    """Replays the module's bookkeeping from the recorded pre spikes: X_t = X_{t-1} d + pre_t ; on a forced WTA spike at t: Eq. 4 with X_t, clip, Eq. 5 ; E decays, E += dloc."""
    proj = net.synapseslices["exc1_wta"]
    dt = 1.0
    d_pre, d_tr = np.exp(-dt / TAU_PRE), np.exp(-dt / TAU_TRACE)
    X = np.zeros(net.n_total); E = np.zeros_like(out[0]["E"]); W = None
    for t, o in enumerate(out):
        X = X * d_pre + (o["pre"] > 0)
        E = E * d_tr
        if W is None:
            W = o["W"].copy() if t not in force_steps else None
        assert np.allclose(o["X"], X, atol=1e-12), t
        if t in force_steps:
            rows = force_steps[t]
            x = X[proj.pre]
            dloc = ETA * (x - X_TAR)[None, :] * np.maximum(W_MAX - W_prev[rows], 0.0) ** mu
            new = np.minimum(np.maximum(W_prev[rows] + dloc, 0.0), W_MAX)
            scale_rows_to(new, ALPHA); new = np.minimum(new, W_MAX)                  ##clip, scale, hard bound
            assert np.allclose(o["W"][rows], new, atol=2e-6), t                      ##W after the engine's commit (float32)
            E[rows] += dloc
        assert np.allclose(o["E"], E, atol=1e-9), t
        W_prev = o["W"]
    return X, E

def test_rstdp_hand_recursion_dense():
    for mu in (0.0, 0.1, 1.0):
        run = _run(_build(weights_format="dense", mu=mu, w_in=1e-3))               ##w_in ~0: the liquid is silent, only the FORCED WTA spikes update W
        net = run.ctx.net
        frames = np.random.default_rng(3).random((40, N_IN)) < 0.3
        force = {5: [0], 12: [1, 2], 30: [0]}
        out = _drive(run, frames, force)
        assert [int(o["S"].sum()) for o in out] == [len(force.get(t, ())) for t in range(40)]
        W_prev = out[0]["W"]
        X, E = _hand(out, run, mu, force, net)
        assert (out[-1]["W"].sum(1)[[0, 1, 2]] <= ALPHA + 1e-4).all() and np.allclose(out[-1]["W"][3], out[0]["W"][3])   ##rows that spiked are rescaled (<= alpha: the hard bound can cut), row 3 untouched
        assert (E[[0, 1, 2]] != 0).any() and not E[3].any()
        ##X on the input columns has seen only the input spikes; the trace is below x_tar everywhere -> every local change was a depression
        assert X[:N_IN].max() < X_TAR and (run.plasticity.E[[0, 1, 2]] <= 0).all()

def test_feedback_forgiving_strict_and_auto_k():
    run = _run(_build(weights_format="dense", w_in=1e-3, feedback="forgiving", k=None))       ##k None: the literal auto scaling
    pl, ctx = run.plasticity, run.ctx
    frames = np.random.default_rng(3).random((20, N_IN)) < 0.3
    _drive(run, frames, {4: [0, 1]})
    W0, E0 = _wex(run), pl.E.copy()
    assert np.abs(E0).max() > 0
    ##wrong + forgiving: nothing
    info = pl.apply_feedback(ctx, correct=False)
    assert info["k"] == 0.0 and np.array_equal(_wex(run), W0)
    ##correct: W += k E with k = w_max / max|E| (auto, k=None), clipped to [0, w_max], NOT rescaled (scale_after_feedback False)
    info = pl.apply_feedback(ctx, correct=True)
    k = W_MAX / np.abs(E0).max()
    assert info["k"] == pytest.approx(k)
    assert np.allclose(_wex(run), np.minimum(np.maximum(W0 + k * E0, 0.0), W_MAX), atol=2e-6)
    assert np.array_equal(pl.E, E0)                                                 ##e is not reset by the feedback
    ##strict + wrong: W -= gamma W
    run = _run(_build(weights_format="dense", w_in=1e-3, feedback="strict", gamma=0.01))
    _drive(run, frames, {4: [0, 1]})
    W0 = _wex(run)
    run.plasticity.apply_feedback(run.ctx, correct=False)
    assert np.allclose(_wex(run), W0 * 0.99, atol=2e-6)
    ##fixed k, and scale_after_feedback puts the rows back to alpha
    run = _run(_build(weights_format="dense", w_in=1e-3, feedback="strict", k=0.5, scale_after_feedback=True))
    _drive(run, frames, {4: [0, 1]})
    W0, E0 = _wex(run), run.plasticity.E.copy()
    info = run.plasticity.apply_feedback(run.ctx, correct=True)
    exp = np.minimum(np.maximum(W0 + 0.5 * E0, 0.0), W_MAX); scale_rows_to(exp, ALPHA); exp = np.minimum(exp, W_MAX)
    assert info["k"] == 0.5 and np.allclose(_wex(run), exp, atol=2e-6) and (_wex(run).sum(1) <= ALPHA + 1e-4).all()
    ##feedback "none": nothing ever
    run = _run(_build(weights_format="dense", w_in=1e-3, feedback="none"))
    _drive(run, frames, {4: [0, 1]}); W0 = _wex(run)
    run.plasticity.apply_feedback(run.ctx, correct=True); run.plasticity.apply_feedback(run.ctx, correct=False)
    assert np.array_equal(_wex(run), W0)


def test_reset_transient_clears_traces_and_keeps_wex():
    run = _run(_build(weights_format="dense", w_in=1e-3))
    _drive(run, np.random.default_rng(3).random((20, N_IN)) < 0.3, {4: [0, 1]})
    W0 = _wex(run)
    assert run.plasticity.X.any() and run.plasticity.E.any()
    run.reset_transient()
    assert not run.plasticity.X.any() and not run.plasticity.E.any() and np.array_equal(_wex(run), W0)


# ------------------------------------------------------------------ dense == sparse
def test_dense_equals_sparse_under_the_full_dynamics():
    frames = np.random.default_rng(7).random((300, N_IN)) < 0.4
    runs = [_run(_build(weights_format=f, seed=11)) for f in ("dense", "sparse")]
    counts = []
    for run in runs:
        run.reset_transient()
        m0 = run.history.mark(); run.run_sample(frames); c = run.history.since(m0)
        counts.append(c)
        run.plasticity.apply_feedback(run.ctx, correct=True)
    net = runs[0].ctx.net
    assert counts[0][net.neuronslices["wta"]].sum() > 0, "the WTA never fired: the test has no plasticity to compare"
    assert np.array_equal(counts[0], counts[1])
    assert np.allclose(_wex(runs[0]), _wex(runs[1]), atol=1e-4) and np.allclose(_E(runs[0]), _E(runs[1]), atol=1e-6)
    assert not np.allclose(_wex(runs[0]), _wex(_run(_build(weights_format="dense", seed=11))), atol=1e-6)      ##something was learned


# ------------------------------------------------------------------ protocol
def test_protocol_end_to_end(tmp_path):
    data = _samples(n=12, T=80, seed=1)
    pnet = _build(seed=2)
    rc = runcfg(n_train=8, n_test=4, feedback_start=0.5, refit_every=1, refit_epochs=5, readout_epochs=20, report_every=4)
    run = pnet.train(data.subset(np.arange(8)), rc, test_data=data.subset(np.arange(8, 12)), out_dir=tmp_path, save_run=False, save_history=False)
    r = np.load(tmp_path / "responses.npz")
    assert r["resp_train"].shape == (8, 24) and r["resp_test"].shape == (4, 24) and (tmp_path / "results.json").exists()
    assert (r["pred_train"][:4] == -1).all() and (r["pred_train"][4:] >= 0).all()         ##feedback from sample 4 on, two classes seen by then
    assert r["wex"].size == 24 * N_WTA
    run.check_persistent_unchanged()
    ##no feedback: no predictions, W_ex only moved by the local rule; final_pass fills Resp_final
    pnet = _build(seed=2, feedback="none")
    rc = runcfg(n_train=8, n_test=4, refit_every=1, refit_epochs=5, readout_epochs=20, final_fit="final_pass", features="exc1+exc2", report_every=4)
    run = pnet.train(data.subset(np.arange(8)), rc, test_data=data.subset(np.arange(8, 12)), out_dir=tmp_path / "none", save_run=False, save_history=False)
    r = np.load(tmp_path / "none" / "responses.npz")
    assert (r["pred_train"] == -1).all() and r["resp_final"].shape == (8, 48) and r["resp_train"].shape == (8, 48)


def test_no_where_in_the_new_files():
    for f in (_ROOT / "experiment_dynamics" / "krenzer2025.py", _ROOT / "experiments" / "krenzer_2025.py"):
        assert "where=" not in f.read_text(), f
