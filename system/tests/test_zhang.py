#test_zhang -- Zhang et al. 2025 as add-ons to the D&C baseline (zhang2025.py, zhang_2025.py): the shared exc -> inh wiring, the Eq. 7 SOM-graded
# inhibition, the dendritic current (Eq. 4-6) step by step against a hand recursion, the one-spike threshold, the proportion-weighted readout, and
# that with every switch off the build IS diehl_cook_2015.build(). Dense only (D&C's modules are dense).
# Run: pytest system/tests/test_zhang.py -q     (from the repo root)
from __future__ import annotations
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
##system/tests/conftest.py sets sys.path for pytest
from dataclasses import asdict
import numpy as np
from modelinput import ModelInput
import pytest

from system.dynamicscfg import DynamicsCfg
from builder import CONNECTIVITY_PATTERNS
from zhang2025 import ZhangSynapseCfg, ZhangDendriticSynapse, ZhangThresholdCfg, ZhangOneSpikeThreshold
from diehlcook import DCConductanceSynapse, DCAdaptiveThreshold
from zhang_2025 import build, runcfg, psc_groups, register_psc_patterns, exc_grid, som_inhibition_weights, LAMBDA, BETA_S
from scorer import proportion_weighting
from diehl_cook_2015 import fresh_network, build as dc_build

N_IN, N_EXC = 32, 16
NORM = 78.0 * N_IN / 784


def _build(n_in=N_IN, n_exc=N_EXC, **kw):
    return build(n_in, n_exc, norm_target=78.0 * n_in / 784, **kw)


def _run(pnet, n_in=N_IN):
    """A prepared Run in the state of a fresh D&C network (V = v_rest - 40, delays drawn at the first step), eval mode + theta frozen: W stays put."""
    run = pnet._prep_run(ModelInput.from_tonic([(np.zeros((2, n_in), bool), 0)] * 4), runcfg(n_train=2, n_assign=2, n_test=2))
    fresh_network(run)
    run.threshold.freeze(True); run.set_mode("eval")
    return run


def _rows(net, group):
    return slice(net.neuronslices[group].start - net.row_offset, net.neuronslices[group].stop - net.row_offset)


def _drive(run, frames):
    """Step through ONE sample by hand; per step: copies of I2V, Dend, the rec spikes."""
    ctx = run.ctx
    run._datahandler.begin_sample(np.asarray(frames, bool), ctx)
    out = []
    for _ in range(len(frames)):
        run.step()
        out.append(dict(I=ctx.upd.I2V.astype(np.float64).copy(), D=run.synapse.Dend.copy(), S=ctx.neur.Spikes[ctx.net.n_input:].copy(), V=ctx.neur.V.astype(np.float64).copy()))
    return out


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ------------------------------------------------------------------ partially shared connection (§3.4)
def test_psc_groups_share_the_stated_number_of_neurons():
    for shared, per_exc in ((2, 2), (0, 1)):
        src = psc_groups(16, 4, shared)
        assert src.shape == (16 // (4 - shared), 4)
        assert np.all(np.bincount(src.ravel(), minlength=16) == per_exc), shared      ##shared 2: every exc neuron feeds exactly two inh neurons; 0: one
        for k in range(len(src) - 1):
            assert len(set(src[k]) & set(src[k + 1])) == shared                         ##adjacent groups share `shared` neurons
    src = psc_groups(12, 4, 1)                                                          ##stride 3: 4 groups, the last wraps (9, 10, 11, 0)
    assert src.shape == (4, 4) and list(src[-1]) == [9, 10, 11, 0] and len(set(src[0]) & set(src[1])) == 1
    for bad in ((16, 4, 4), (16, 4, -1), (3, 4, 2)):
        with pytest.raises(ValueError):
            psc_groups(*bad)

def test_psc_grid_layout_keeps_every_group_on_one_tile():
    """Grid layout = 2 x 2 tiles, so an inh neuron's position (mean of its sources) is never more than one cell from any source."""
    _, pos = exc_grid(100)
    for shared, n_inh in ((2, 45), (1, 41), (0, 25)):
        src = psc_groups(100, 4, shared, (10, 10))
        assert src.shape == (n_inh, 4), shared
        spread = np.linalg.norm(pos[src] - pos[src].mean(axis=1, keepdims=True), axis=2).max()
        assert spread < 0.75                                                            ##a 2 x 2 tile: every source sqrt(0.5) from the tile centre
        counts = np.bincount(src.ravel(), minlength=100)
        assert counts.max() == (2 if shared else 1) and counts.min() == 1
    ##the index rule puts 10 of 50 groups across a row end, far from their sources
    idx = psc_groups(100, 4, 2)
    spread = np.linalg.norm(pos[idx] - pos[idx].mean(axis=1, keepdims=True), axis=2).max(axis=1)
    assert (spread > 1.5).sum() == 10
    src = psc_groups(16, 4, 2, (4, 4))
    assert src.shape == (6, 4) and list(src[0]) == [0, 1, 4, 5] and list(src[1]) == [1, 2, 5, 6] and list(src[3]) == [8, 9, 12, 13]
    for bad in (dict(n_exc=15, grid=(3, 5)), dict(n_exc=16, grid=(4, 4), group_size=3), dict(n_exc=16, grid=(2, 4))):
        with pytest.raises(ValueError):
            psc_groups(bad["n_exc"], bad.get("group_size", 4), 2, bad["grid"])

def test_psc_patterns_wire_groups_and_exclude_own_sources():
    fwd, back = register_psc_patterns(4, 2)
    assert fwd in CONNECTIVITY_PATTERNS and back in CONNECTIVITY_PATTERNS
    m = CONNECTIVITY_PATTERNS[fwd](16, 8)
    assert m.shape == (16, 8) and m.sum() == 32 and np.array_equal(np.flatnonzero(m[:, 3]), psc_groups(16, 4, 2)[3])
    b = CONNECTIVITY_PATTERNS[back](8, 16)
    assert b.shape == (8, 16) and b.sum() == 8 * 12 and np.array_equal(b, ~m.T)
    with pytest.raises(ValueError):
        CONNECTIVITY_PATTERNS[fwd](16, 16)                                             ##16 exc form 8 groups, not 16
    ##in a built network (default psc_layout "grid": 4 x 4 exc grid -> 6 tiles)
    net = _build(psc=True).network
    ids = (net.pre_idx, net.post_idx + net.row_offset)
    by = {k: net.syntype == v for k, v in net.type_key.items()}
    ex, ih = net.neuronslices["exc"], net.neuronslices["inh"]
    n_inh = 6
    assert ih.stop - ih.start == n_inh and by["exc_inh"].sum() == 4 * n_inh and by["inh_exc"].sum() == n_inh * (N_EXC - 4)
    src = psc_groups(N_EXC, 4, 2, (4, 4))
    for k in range(n_inh):
        pre = ids[0][by["exc_inh"] & (ids[1] == ih.start + k)] - ex.start
        assert sorted(pre) == sorted(src[k])
        tgt = ids[1][by["inh_exc"] & (ids[0] == ih.start + k)] - ex.start
        assert not (set(tgt) & set(src[k])) and len(tgt) == N_EXC - 4
    assert np.all(net.W0[by["exc_inh"]] == np.float32(10.4))
    ##the index layout is still available
    net = _build(psc=True, psc_layout="index").network
    assert net.neuronslices["inh"].stop - net.neuronslices["inh"].start == N_EXC // 2 and net.synapsespecs["exc_inh"].p_connection == "psc_g4_s2"
    ##psc off: D&C's one_to_one / all_but_self with n_inh = n_exc
    net = _build(psc=False).network
    assert net.neuronslices["inh"].stop - net.neuronslices["inh"].start == N_EXC and net.synapsespecs["exc_inh"].p_connection == "one_to_one"
    assert net.synapsespecs["inh_exc"].p_connection == "all_but_self"


# ------------------------------------------------------------------ adaptive self-organising inhibition (§3.3, Eq. 7)
def test_som_weights_are_eq7_on_the_exc_grid():
    grid, pos = exc_grid(16)
    assert grid == (4, 4) and pos.shape == (16, 2) and list(pos[5]) == [1.0, 1.0]
    with pytest.raises(ValueError):
        exc_grid(12)
    assert exc_grid(12, grid=(3, 4))[0] == (3, 4)
    ##one_to_one: inh k sits at exc k; the weight to exc i is -(lam * d(k, i) / dmax + beta_s), dmax over all realized edges = the grid diagonal
    net = _build(psc=False, som_inhibition=True).network
    by = net.syntype == net.type_key["inh_exc"]
    pre = net.pre_idx[by] - net.neuronslices["inh"].start
    post = net.post_idx[by] + net.row_offset - net.neuronslices["exc"].start
    d = np.linalg.norm(pos[post] - pos[pre], axis=1)
    expect = -(LAMBDA * d / np.sqrt(18.0) + BETA_S)
    assert np.allclose(net.W0[by], expect, atol=1e-4) and np.isclose(net.W0[by].max(), -(LAMBDA / np.sqrt(18.0) + BETA_S), atol=1e-4)
    assert np.isclose(net.W0[by].min(), -(LAMBDA + BETA_S), atol=1e-4)
    assert np.allclose(som_inhibition_weights(pos, pos, pre, post), -expect)
    ##sqrt_distance (BindsNET's grading): the same lam + beta_s at the farthest pair, every nearer pair inhibited MORE than Eq. 7
    sq = som_inhibition_weights(pos, pos, pre, post, sqrt_distance=True)
    assert np.allclose(sq, LAMBDA * np.sqrt(d) / np.sqrt(np.sqrt(18.0)) + BETA_S) and sq.max() == pytest.approx(LAMBDA + BETA_S)
    assert np.all(sq[d < np.sqrt(18.0)] > -expect[d < np.sqrt(18.0)]) and sq.min() == pytest.approx(LAMBDA / np.sqrt(np.sqrt(18.0)) + BETA_S)
    ##psc: the inh neuron's position is the mean of its sources
    net = _build(psc=True, som_inhibition=True).network
    by = net.syntype == net.type_key["inh_exc"]
    pre = net.pre_idx[by] - net.neuronslices["inh"].start
    post = net.post_idx[by] + net.row_offset - net.neuronslices["exc"].start
    pos_inh = pos[psc_groups(N_EXC, 4, 2, (4, 4))].mean(axis=1)
    assert np.allclose(pos_inh[0], [0.5, 0.5])                                        ##tile 0 = exc 0, 1, 4, 5: (0,0),(0,1),(1,0),(1,1) -> mean (0.5, 0.5)
    d = np.linalg.norm(pos[post] - pos_inh[pre], axis=1)
    assert np.allclose(net.W0[by], -(LAMBDA * d / d.max() + BETA_S), atol=1e-4)
    ##off: D&C's -17 everywhere; the network ids differ
    off = _build(psc=True, som_inhibition=False).network
    assert np.all(off.W0[off.syntype == off.type_key["inh_exc"]] == np.float32(-17.0)) and off.network_id != net.network_id
    net.rehash_network()                                                                ##the regenerated meta is consistent with the edited W0

def test_all_switches_off_is_diehl_cook():
    z, d = _build(dendrites=False, som_inhibition=False, psc=False, one_spike=False, triplet=True), dc_build(N_IN, N_EXC, norm_target=NORM)
    zn, dn = z.network, d.network
    assert np.array_equal(zn.pre_idx, dn.pre_idx) and np.array_equal(zn.post_idx, dn.post_idx) and np.array_equal(zn.syntype, dn.syntype) and np.array_equal(zn.W0, dn.W0)
    for sec in ("simulation", "synapse", "integrator", "threshold", "plasticity", "noise"):
        assert asdict(getattr(z.cfg, sec)) == asdict(getattr(d.cfg, sec)), sec
    assert z.synapse_cls is DCConductanceSynapse and z.threshold_cls is DCAdaptiveThreshold
    full = _build()
    assert full.synapse_cls is ZhangDendriticSynapse and full.threshold_cls is ZhangOneSpikeThreshold
    ##the Zhang default is the paper's PAIR rule; D&C keeps the triplet rule
    assert full.cfg.plasticity.triplet is False and d.cfg.plasticity.triplet is True and full.network.network_name.endswith("_pair")

def test_cfg_round_trip_finds_the_zhang_sections():
    cfg = _build().cfg
    back = DynamicsCfg.load(asdict(cfg))
    assert isinstance(back.synapse, ZhangSynapseCfg) and isinstance(back.threshold, ZhangThresholdCfg) and back == cfg


# ------------------------------------------------------------------ dendrites (§3.1, Eq. 4-6), step by step
def _one_spike_frames(T, i):
    f = np.zeros((T, N_IN), bool); f[0, i] = True
    return f

def test_dendritic_current_arrives_with_dc_timing_and_gates_the_soma():
    """One input spike raises Dend at D&C's delayed arrival step, gates the soma current through the dendritic sigmoids, then leaks (Euler)."""
    run = _run(_build(theta_f=(0.05, 0.1, 0.2)))                                     ##thresholds at the scale of ONE weight, so the gates move
    ctx, syn, net = run.ctx, run.synapse, run.ctx.net
    ex = _rows(net, "exc")
    i, T = 3, 30
    out = _drive(run, _one_spike_frames(T, i))
    K, R = syn.K[:, i], syn._R
    assert syn.Delay_drawn[0] and R == 21 and syn.Dend.shape == (1, N_EXC) and syn.Buf.shape == (R, 1, N_EXC)
    w = ctx.syn.W[net.synapseslices["in_exc"].post, net.synapseslices["in_exc"].pre][:, i].astype(np.float64)
    dt, tau, gamma, th = 0.5, 10.0, 0.5, np.array([0.05, 0.1, 0.2])
    V0 = lambda t, r: out[t - 1]["V"][ex][r] if t else -105.0                          ##V at the start of step t (fresh network: v_rest - 40; no spikes: nothing resets it)
    for r in range(N_EXC):
        t0 = 1 + int(K[r])
        for t in range(t0):                                                           ##before the arrival: Dend 0, the soma gets the resting gate value gamma * sum sigmoid(-theta)
            assert out[t]["D"][0, r] == 0.0 and out[t]["I"][ex][r] == pytest.approx(gamma * _sig(-th).sum() * (0.0 - V0(t, r)), rel=1e-6), (r, t)
        D = w[r]
        for t in range(t0, T):
            y = gamma * _sig(D - th).sum()
            assert out[t]["I"][ex][r] == pytest.approx(y * (0.0 - V0(t, r)), rel=1e-6, abs=1e-9), (r, t)
            D = D + (-D / tau) * dt
            assert out[t]["D"][0, r] == pytest.approx(D, rel=1e-9), (r, t)
    assert not any(o["S"].any() for o in out)                                        ##one weak spike: nobody fires

def test_split_assignment_routes_each_input_to_one_dendrite():
    run = _run(_build(dendrite_assign="split"))
    syn = run.synapse
    assert syn.Dend.shape == (3, N_EXC) and syn.Buf.shape == (21, 3, N_EXC) and np.array_equal(syn._dend_of_pre, np.arange(N_IN) % 3)
    out = _drive(run, _one_spike_frames(25, 4))                                      ##input 4 -> dendrite 1
    D = out[-1]["D"]
    assert D[1].min() > 0 and not D[0].any() and not D[2].any()
    ##the gate of dendrite 1 moves, the other two sit at sigmoid(-theta_f)
    gamma, th = 0.5, np.array([1.0, 5.0, 10.0])
    r = 0
    y = gamma * (_sig(out[-2]["D"][1, r] - th[1]) + _sig(-th[0]) + _sig(-th[2]))
    assert out[-1]["I"][_rows(run.ctx.net, "exc")][r] == pytest.approx(y * (0.0 - out[-2]["V"][_rows(run.ctx.net, "exc")][r]), rel=1e-6)

def test_retain_rule_transmits_the_current_then_keeps_its_sigmoid():
    run = _run(_build(dendrite_rule="retain", theta_f=(0.05, 0.1, 0.2)))
    ctx, syn, net = run.ctx, run.synapse, run.ctx.net
    ex, i = _rows(net, "exc"), 3
    out = _drive(run, _one_spike_frames(25, i))
    assert syn.Dend.shape == (3, N_EXC)                                               ##"retain" needs every dendrite's own current
    w = ctx.syn.W[net.synapseslices["in_exc"].post, net.synapseslices["in_exc"].pre][:, i].astype(np.float64)
    dt, tau, gamma, th = 0.5, 10.0, 0.5, np.array([0.05, 0.1, 0.2])
    for r in range(N_EXC):
        t0, D = 1 + int(syn.K[:, i][r]), np.zeros(3)
        for t in range(len(out)):                                                     ##per step: arrival -> transmit the currents (Eq. 5) -> retain their sigmoid (Eq. 6) -> leak (Eq. 4)
            if t == t0:
                D = D + w[r]
            y = gamma * D.sum()
            V = out[t - 1]["V"][ex][r] if t else -105.0
            assert out[t]["I"][ex][r] == pytest.approx(y * (0.0 - V), rel=1e-6, abs=1e-9), (r, t)
            D = _sig(D - th) * (1 - dt / tau)
            assert np.allclose(out[t]["D"][:, r], D, rtol=1e-9), (r, t)

def test_current_soma_input_adds_gain_times_y():
    run = _run(_build(soma_input="current", soma_gain=2.5, theta_f=(0.05, 0.1, 0.2)))
    ctx, syn, net = run.ctx, run.synapse, run.ctx.net
    ex, i = _rows(net, "exc"), 3
    out = _drive(run, _one_spike_frames(25, i))
    w = ctx.syn.W[net.synapseslices["in_exc"].post, net.synapseslices["in_exc"].pre][:, i].astype(np.float64)
    th = np.array([0.05, 0.1, 0.2])
    for r in range(N_EXC):
        t0 = 1 + int(syn.K[:, i][r])
        assert out[t0]["I"][ex][r] == pytest.approx(2.5 * 0.5 * _sig(w[r] - th).sum(), rel=1e-6), r
    assert np.allclose([o["I"][ex] for o in out][0], 2.5 * 0.5 * _sig(-th).sum())    ##no input yet: the resting gate value as a current

def test_inhibitory_path_and_inh_rows_are_dc():
    """The inh rows keep D&C's ge/gi current (exc -> inh feeds Ge), and an inh spike reaches the exc rows through Gi as in D&C (negative current)."""
    run = _run(_build(psc=False, som_inhibition=False))
    ctx, net = run.ctx, run.ctx.net
    ex, ih = _rows(net, "exc"), _rows(net, "inh")
    frames = np.zeros((6, N_IN), bool)
    run._datahandler.begin_sample(frames, ctx)
    ctx.neur.V[ex.start] = 0.0                                                        ##exc row 0 fires in step 0 (threshold -52) ...
    run.step()
    assert ctx.neur.Spikes[net.n_input:][ex.start] and not ctx.neur.Spikes[net.n_input:][ih].any()
    run.step()                                                                        ##... step 1: its 10.4 lands in Ge of inh row 0 (one_to_one) and drives V from -100 ...
    assert run.synapse.Ge[ih.start] > 0 and run.synapse.Ge[ex].max() == 0.0 and run.synapse.Ge[ih][1:].max() == 0.0
    for _ in range(3):                                                                ##... until the inh neuron passes -40 (two more steps here)
        if ctx.neur.Spikes[net.n_input:][ih.start]:
            break
        run.step()
    assert ctx.neur.Spikes[net.n_input:][ih.start] and ctx.neur.Spikes[net.n_input:][ih].sum() == 1
    V0 = ctx.neur.V.astype(np.float64)[ex]
    run.step()                                                                        ##the step after: gi += 17 on every exc row but 0; the current is D&C's gi term + the resting gate
    I, Gi = ctx.upd.I2V.astype(np.float64)[ex], run.synapse.Gi[ex]
    y_rest = 0.5 * _sig(-np.array([1.0, 5.0, 10.0])).sum()
    assert Gi[0] == 0.0 and np.allclose(Gi[1:], 17.0 * (1 - 0.5 / 2.0))                ##stored AFTER the step's Euler decay (tau_gi 2 ms); the current used 17
    assert np.allclose(I, y_rest * (0.0 - V0) + (Gi > 0) * 17.0 * (-100.0 - V0))

def test_fresh_network_resets_the_dendrites_and_keeps_the_delays():
    run = _run(_build())
    syn = run.synapse
    _drive(run, (np.random.default_rng(0).random((30, N_IN)) < 0.3))
    assert syn.Dend.any() and syn.Buf.any()
    K = syn.K.copy()
    fresh_network(run)
    assert not syn.Dend.any() and not syn.Buf.any() and syn.Delay_drawn[0] and np.array_equal(syn.K, K)
    fresh_network(run, redraw_delays=True)
    assert not syn.Delay_drawn[0]


# ------------------------------------------------------------------ one spike per step (Fig. 2)
def _threshold_case(select, one_spike=True, seed=7, one_spike_losers="reset"):
    run = _run(_build(one_spike=one_spike, one_spike_select=select, seed=seed, one_spike_losers=one_spike_losers))
    ctx, net = run.ctx, run.ctx.net
    ex, ih = _rows(net, "exc"), _rows(net, "inh")
    run.threshold.freeze(False)
    ctx.neur.Since[:] = 1000
    cand = np.array([ex.start + k for k in (1, 4, 7, 9, 12)])
    ctx.neur.V[:] = -70.0
    ctx.neur.V[cand] = [-40.0, -30.0, -45.0, -35.0, -41.0]                           ##all above -52; row ex+4 highest
    ctx.neur.V[ih.start:ih.start + 3] = -20.0                                         ##three inh rows above -40
    run.threshold.step(ctx)
    return run, ctx.upd.rec_spikes.copy(), cand, ex, ih

def test_one_spike_keeps_exactly_one_exc_spike_and_resets_every_crosser():
    """One exc spike per step; with losers "reset" (BindsNET's order) every crosser adapts and is reset, with "keep" only the winner adapts."""
    run, s, cand, ex, ih = _threshold_case("max_v")
    assert s[ex].sum() == 1 and s[ex.start + 4] and s[ih].sum() == 3                 ##inh rows are not limited
    th = run.threshold.Theta[ex]
    adapted = 20.0 + 0.05 - 20.0 / 1e7 * 0.5
    assert np.allclose(th[[1, 4, 7, 9, 12]], adapted) and np.allclose(np.delete(th, [1, 4, 7, 9, 12]), 20.0 - 20.0 / 1e7 * 0.5)
    losers = cand[cand != ex.start + 4]
    assert np.all(run.ctx.neur.V[losers] == np.float32(-65.0)) and np.all(run.ctx.neur.Since[losers] == 0)
    assert run.ctx.neur.V[ex.start + 4] == np.float32(-30.0)                         ##the winner is reset by the orchestra, not here
    run, s, cand, ex, ih = _threshold_case("max_v", one_spike_losers="keep")
    th = run.threshold.Theta[ex]
    assert th[4] == pytest.approx(adapted, abs=1e-9) and np.allclose(np.delete(th, 4), 20.0 - 20.0 / 1e7 * 0.5)
    assert np.all(run.ctx.neur.V[cand] == np.float32([-40.0, -30.0, -45.0, -35.0, -41.0])) and np.all(run.ctx.neur.Since[cand] == 1000)
    run, s, cand, ex, ih = _threshold_case("random")
    assert s[ex].sum() == 1 and s[cand].sum() == 1 and s[ih].sum() == 3
    run2, s2, *_ = _threshold_case("random")                                          ##the keyed stream: same build, same choice
    assert np.array_equal(s, s2)
    run, s, cand, ex, ih = _threshold_case("random", one_spike=False)                 ##D&C: everybody above threshold fires
    assert s[cand].all() and s[ex].sum() == 5
    with pytest.raises(ValueError):
        _run(_build(one_spike_select="loudest"))                                      ##checked when the module is instantiated

def test_one_spike_in_a_run_never_shows_two_exc_spikes_in_one_step():
    run = _run(_build(theta_f=(0.05, 0.1, 0.2)))
    run.threshold.freeze(False)
    out = _drive(run, np.random.default_rng(1).random((120, N_IN)) < 0.5)             ##hard drive: many exc neurons at threshold together
    ex = _rows(run.ctx.net, "exc")
    per_step = np.array([o["S"][ex].sum() for o in out])
    assert per_step.max() == 1 and per_step.sum() >= 2


# ------------------------------------------------------------------ the readout the paper names
def test_proportion_weighting_hand_case():
    resp_assign = np.array([[10, 1], [10, 1], [0, 5], [0, 5]], dtype=float)          ##4 assign examples x 2 neurons: classes 0, 0, 1, 1
    y_assign = np.array([0, 0, 1, 1])
    assign, prop, pred = proportion_weighting(resp_assign, y_assign, np.array([[0, 6], [4, 1]]), n_classes=2)
    assert list(assign) == [0, 1] and np.allclose(prop, [[1.0, 0.0], [1 / 6, 5 / 6]])
    assert list(pred) == [1, 0]                                                       ##[0, 6]: class 1 scores 6 * 5/6 = 5 vs 0 ; [4, 1]: class 0 scores 4 vs 5/6
    ##as bindsnet.evaluation: a silent neuron is unassigned (-1), a sample with no weighted activity abstains (-1)
    assign, prop, pred = proportion_weighting(np.zeros((4, 1)), y_assign, np.array([[3.0]]), n_classes=2)
    assert list(assign) == [-1] and not prop.any() and list(pred) == [-1]
    resp_assign = np.array([[10, 1, 0], [10, 1, 0], [0, 5, 0], [0, 5, 0]], dtype=float)   ##a third, silent neuron must not enter class 0's divisor
    assign, prop, pred = proportion_weighting(resp_assign, y_assign, np.array([[4, 1, 0], [0, 0, 9]]), n_classes=2)
    assert list(assign) == [0, 1, -1] and list(pred) == [0, -1]


# ------------------------------------------------------------------ end to end: the D&C protocol runs the Zhang build
def test_protocol_runs_end_to_end_and_saves_responses(tmp_path):
    from zhang_2025 import mnist_input
    rng = np.random.default_rng(0)
    x = (rng.random((6, 28, 28)) * 255).astype(np.uint8); y = np.arange(6) % 10
    kw = dict(present_ms=20.0, rest_ms=10.0)
    data, test = mnist_input(x, y, **kw), mnist_input(x[:3], y[:3], seed=8, **kw)
    pnet = build(784, 8, grid=(2, 4), norm_target=78.0)
    assert pnet.network.neuronslices["inh"].stop - pnet.network.neuronslices["inh"].start == 3     ##2 x 4 grid, shared 2: three 2 x 2 tiles
    rc = runcfg(n_train=3, n_assign=3, n_test=3, test_passes=1, max_repeats=2, report_every=1, **kw)
    run = pnet.train(data, rc, test_data=test, out_dir=tmp_path, save_run=False, save_history=False)
    r = np.load(tmp_path / "responses.npz")
    assert r["resp_test"].shape == (1, 3, 8) and r["resp_assign"].shape == (3, 8) and (tmp_path / "results.json").exists()
    assign, prop, pred = proportion_weighting(r["resp_assign"], r["y_assign"], r["resp_test"][0])
    assert assign.shape == (8,) and prop.shape == (8, 10) and pred.shape == (3,)
    run.check_persistent_unchanged()


def test_no_where_in_the_new_files():
    for f in (_ROOT / "experiment_dynamics" / "zhang2025.py", _ROOT / "experiments" / "zhang_2025.py"):
        src = f.read_text()
        assert "where=" not in src, f
