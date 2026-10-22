#test_hlsp -- the HLSP-LSM build (Luo 2026): the current synapse with short-term depression, the discrete LIF of Eq. (1)-(2) with a refractory hold,
# E-STDP + I-STDP (Eq. 8-15) in dense and sparse form, the fixed +-W_in input weights, the self-organise -> state-vector -> readout protocol and the
# shared readout's options. Each is asserted against a HAND reference (a two-neuron liquid whose spikes are forced step by step), never against the modules.
# Run: pytest system/tests/test_hlsp.py -q     (from the repo root)
from __future__ import annotations
import re
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
##sys.path is set by system/tests/conftest.py
import json
import numpy as np
from modelinput import ModelInput
import pytest

from phasicnet import PhasicNet
from cfg import Cfg, SimulationCfg, NoiseCfg, PhaseCfg
from original_dynamics import TestNoise
from phases import DefaultPhase
from hlsp2026 import (HLSPSynapseCfg, HLSPIntegratorCfg, HLSPThresholdCfg, HLSPPlasticityCfg, HLSPCurrentSynapse, HLSPCurrentSynapseSparse,
                      HLSPLIF, HLSPThreshold, HLSPPlasticity, HLSPPlasticitySparse)
from readouts import LogisticReadout
import hlsp_lsm_2026 as D

FORMATS = ["dense", "sparse"]
BETA, ALPHA, GAMMA = np.exp(-1 / 64.0), np.exp(-1 / 1.0), np.exp(-1 / 150.0)       ##Table 2-3 at dt = 1 ms
DE, DI1, DI2 = np.exp(-1 / 10.0), np.exp(-1 / 5.0), np.exp(-1 / 10.0)                ##trace decays: tau_E 10, tau_I1 5, tau_I2 10
KI = 0.15 / (1 - 5.0 / 10.0)                                                            ##Eq. (14): A_I / (1 - r)


# --------------------------------------------------------------------- helpers

class _Toy:
    """Two classes: input channels 0..19 hot (class 0) or 20..39 hot (class 1), Bernoulli 0.2 per step; lazy, keyed by (seed, i)."""
    def __init__(self, n, T=250, seed=0):
        self.y = np.random.default_rng(seed).integers(0, 2, n); self.T, self.seed = T, seed
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        r = np.random.default_rng([self.seed, int(i)])
        p = np.zeros(40); p[self.y[i] * 20:(self.y[i] + 1) * 20] = 0.2
        return r.random((self.T, 40)) < p, int(self.y[i])


def _micro(fmt, *, w_in=0.0, w_ee=0.5, w_ei=0.5, w_ie=0.5, std=True, pre_depletion=False, estdp=True, istdp=True, A_E=0.15, A_I=0.15, w_max=3.0,
           mode="train"):
    """input(1) -> exc A, B (all_to_all, w_in); A <-> B (w_ee); exc -> inh (w_ei); inh -> exc (w_ie, stored negative). Luo's cfg otherwise.
    Rows under post_offset: [inh, A, B]; columns: [input, inh, A, B]. Returns the prepared Run in `mode`."""
    pnet = PhasicNet(1)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", 1, True), ("exc", 2), ("inh", 1, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc="all_to_all", exc_exc="all_but_self", exc_inh="all_to_all", inh_exc="all_to_all")
    pnet.builder.set_weights({"in_exc": w_in, "exc_exc": w_ee, "exc_inh": w_ei, "inh_exc": w_ie})
    pnet.builder.realize(network_name="micro", post_offset=True, save=False)
    sparse = fmt == "sparse"
    pnet.cfg = Cfg(simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on="event" if sparse else "timestep"),
                   synapse=HLSPSynapseCfg(b=None, tau_syn=1.0, std=std, std_u=0.4, std_tau=150.0, std_pre_depletion=pre_depletion),
                   integrator=HLSPIntegratorCfg(tau_mem=64.0, v_rest=0.0, r=1.0, t_refrac=2.0),
                   threshold=HLSPThresholdCfg(v_reset=0.0, v_th=20.0, tau_th=None, adpt_th=None, t_refrac=2.0),
                   plasticity=HLSPPlasticityCfg(tau_pretrace=10.0, tau_posttrace=10.0, A_ltp=A_E, A_ltd=A_E, eta_exc=1.0, eta_inh=1.0, w_min=-w_max, w_max=w_max,
                                                freeze=("in_exc",), A_I=A_I, tau_I1=5.0, tau_I2=10.0, trace_inc=0.1, estdp=estdp, istdp=istdp),
                   noise=NoiseCfg(0.0, 0.0), phases=(PhaseCfg("DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=HLSPCurrentSynapseSparse if sparse else HLSPCurrentSynapse, integrator_cls=HLSPLIF, threshold_cls=HLSPThreshold,
                             plasticity_cls=HLSPPlasticitySparse if sparse else HLSPPlasticity, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.cfg_name = pnet.run_name = "micro"
    data = ModelInput.from_tonic([(np.zeros((20, 1), bool), 0), (np.zeros((20, 1), bool), 1)])
    run = pnet._prep_run(data, D.runcfg(), test_data=data)
    run.set_mode(mode)
    return run


def _rows(run):
    """(n_input, row A, row B, row inh); the column of a liquid neuron is n_input + its row."""
    net = run.ctx.net
    return net.n_input, net.neuronrows["exc"].start, net.neuronrows["exc"].start + 1, net.neuronrows["inh"].start


def _w(run, post_row, pre_col) -> float:
    syn = run.ctx.syn
    if syn.W.ndim == 2:
        return float(syn.W[post_row, pre_col])
    k = np.flatnonzero((syn.post == post_row) & (syn.pre == pre_col))
    assert k.size == 1, (post_row, pre_col, k)
    return float(syn.W[k[0]])


def _drive(run, T, force=None, frames=None):
    """Step through ONE sample by hand. force = {step: rows}: V of those rows is set to 25 mV before the step (-> a spike in that step unless
    refractory). Returns per-step copies of I2V, V, liquid spikes (Spikes[n_input:] after the step = this step's liquid spikes) and the STD x."""
    ctx = run.ctx
    frames = np.zeros((T, ctx.net.n_input), bool) if frames is None else np.asarray(frames, bool)
    run._datahandler.begin_sample(frames, ctx)
    syn_mod = run.modules["synapse"]
    out = []
    for m in range(T):
        for r in (force or {}).get(m, ()):
            ctx.neur.V[r] = 25.0
        run.step()
        out.append(dict(I=ctx.upd.I2V.astype(np.float64).copy(), V=ctx.neur.V.astype(np.float64).copy(), S=ctx.neur.Spikes[ctx.net.n_input:].copy(),
                        X=syn_mod.X.copy()))
    return out


# --------------------------------------------------------------------- 1. the neuron and synapse, Eq. (1)-(3)

@pytest.mark.parametrize("fmt", FORMATS)
def test_membrane_and_current_follow_eq2_eq3(fmt):
    """One input spike of weight 5 at step 0: C[t] = alpha^t * 5, V[t] = beta V[t-1] + C[t] (Eq. 2 with (1 - beta) absorbed into W), no spike."""
    run = _micro(fmt, w_in=5.0, mode="eval")
    n0, rA, rB, rI = _rows(run)
    frames = np.zeros((6, 1), bool); frames[0, 0] = True
    tr = _drive(run, 6, frames=frames)
    C, V = 0.0, 0.0
    for t, row in enumerate(tr):
        C = ALPHA * C + (5.0 if t == 0 else 0.0)
        V = BETA * V + C
        assert abs(row["I"][rA] - C) < 1e-5 and abs(row["V"][rA] - V) < 1e-4, (t, row["I"][rA], C, row["V"][rA], V)
        assert row["I"][rI] == 0.0 and not row["S"].any()                      ##the input reaches exc only; nothing fires at 5 mV


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("std,pre_depletion,factor", [(False, False, 1.0), (True, False, 0.4 * 0.6), (True, True, 0.4 * 1.0)])
def test_recurrent_spike_arrives_one_step_late_scaled_by_std(fmt, std, pre_depletion, factor):
    """Eq. (3): a liquid spike at m drives its targets at m + 1 with U x_j (x after depletion 0.4 * 0.6, before it 0.4 * 1.0, STD off 1.0)."""
    run = _micro(fmt, w_ee=0.5, w_ei=0.7, std=std, pre_depletion=pre_depletion, mode="eval")
    n0, rA, rB, rI = _rows(run)
    tr = _drive(run, 4, force={0: [rA]})
    assert tr[0]["S"].tolist() == [r == rA for r in range(3)]                   ##A fired at step 0
    assert tr[0]["I"][rB] == 0.0 and tr[0]["I"][rI] == 0.0                     ##nothing yet at step 0
    assert abs(tr[1]["I"][rB] - 0.5 * factor) < 1e-6, (tr[1]["I"][rB], 0.5 * factor)
    assert abs(tr[1]["I"][rI] - 0.7 * factor) < 1e-6
    assert abs(tr[2]["I"][rB] - ALPHA * 0.5 * factor) < 1e-6                   ##then the current decays with alpha
    assert tr[1]["X"][n0 + rA] == pytest.approx(1.0 - 0.4 if std else 1.0)   ##Eq. (18) applied once; input entries untouched
    assert tr[1]["X"][:n0].tolist() == [1.0]


@pytest.mark.parametrize("fmt", FORMATS)
def test_std_depletes_and_recovers_eq18(fmt):
    """Eq. (18): x depletes to 0.6 after a spike, recovers as 1 - 0.4 gamma^k, and a second spike depletes U x of the recovered value."""
    run = _micro(fmt, mode="eval")
    n0, rA, rB, rI = _rows(run)
    tr = _drive(run, 8, force={0: [rA], 5: [rA]})
    x = [row["X"][n0 + rA] for row in tr]
    assert x[0] == 1.0 and x[1] == pytest.approx(0.6)
    for k in range(2, 6):
        assert x[k] == pytest.approx(1.0 - 0.4 * GAMMA ** (k - 1)), k
    x5 = 1.0 - 0.4 * GAMMA ** 4
    assert x[6] == pytest.approx(GAMMA * x5 + (1 - GAMMA) - 0.4 * x5)
    assert tr[5]["S"][rA] and tr[6]["I"][rB] == pytest.approx(ALPHA ** 5 * 0.5 * 0.4 * 0.6 + 0.5 * 0.4 * x[6], abs=1e-6)   ##literal: U x6 drives B at step 6


@pytest.mark.parametrize("fmt", FORMATS)
def test_refractory_hold_two_steps(fmt):
    """T_ref = 2 ms: a neuron that fired at s is held at V = 0 and fires again at s + 2 at the earliest (A: 0, 2, 4...; B: 1, 3, 5...)."""
    run = _micro(fmt, w_in=25.0, w_ee=0.0, w_ei=0.0, w_ie=0.0, mode="eval")
    n0, rA, rB, rI = _rows(run)
    frames = np.ones((8, 1), bool); frames[0, 0] = False
    tr = _drive(run, 8, force={0: [rA]}, frames=frames)
    assert [m for m, r in enumerate(tr) if r["S"][rA]] == [0, 2, 4, 6]
    assert [m for m, r in enumerate(tr) if r["S"][rB]] == [1, 3, 5, 7]
    assert tr[1]["V"][rA] == 0.0 and tr[2]["V"][rB] == 0.0                    ##held at the reset value, no integration while refractory


# --------------------------------------------------------------------- 2. plasticity, Eq. (8)-(17)

@pytest.mark.parametrize("fmt", FORMATS)
def test_estdp_pre_before_post_potentiates_reverse_edge_depresses(fmt):
    """E-STDP: A at 0, B at 2 potentiates B <- A by A_E * 0.1 d^2 and depresses A <- B by the same; nothing else moves."""
    run = _micro(fmt)
    n0, rA, rB, rI = _rows(run)
    cA, cB, cI = n0 + rA, n0 + rB, n0 + rI
    _drive(run, 5, force={0: [rA], 2: [rB]})
    dw = 0.15 * 0.1 * DE ** 2
    assert _w(run, rB, cA) == pytest.approx(0.5 + dw, abs=1e-6)
    assert _w(run, rA, cB) == pytest.approx(0.5 - dw, abs=1e-6)
    assert _w(run, rI, cA) == 0.5 and _w(run, rI, cB) == 0.5                   ##exc -> inh: inh never fired, no trace on the post side
    assert _w(run, rA, cI) == -0.5 and _w(run, rB, cI) == -0.5                 ##inh -> exc: pre never fired
    assert _w(run, rA, 0) == 0.0 and _w(run, rB, 0) == 0.0                     ##frozen input projection untouched


@pytest.mark.parametrize("fmt", FORMATS)
def test_estdp_same_step_pair_cancels(fmt):
    """Traces update with S[t] BEFORE the increments, so an E-STDP pair in the same step cancels on both edges (Algorithm 1)."""
    run = _micro(fmt)
    n0, rA, rB, rI = _rows(run)
    _drive(run, 3, force={0: [rA, rB]})
    assert _w(run, rB, n0 + rA) == 0.5 and _w(run, rA, n0 + rB) == 0.5


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("delta,sign", [(2, -1), (10, +1)])
def test_istdp_window_near_pairs_strengthen_far_pairs_weaken(fmt, delta, sign):
    """Eq. (12)-(15): inh at 0, A at delta strengthens A <- inh for near pairs (delta < 6.9 ms), weakens it for far ones; inh <- A gets E-STDP LTD."""
    run = _micro(fmt)
    n0, rA, rB, rI = _rows(run)
    _drive(run, delta + 2, force={0: [rI], delta: [rA]})
    eI = KI * 0.1 * (DI1 ** delta - 0.5 * DI2 ** delta)
    assert np.sign(eI) == -sign
    assert _w(run, rA, n0 + rI) == pytest.approx(-0.5 - eI, abs=1e-6)
    assert _w(run, rB, n0 + rI) == -0.5                                        ##B silent: no I-STDP on its edge
    assert _w(run, rI, n0 + rA) == pytest.approx(0.5 - 0.15 * 0.1 * DE ** delta, abs=1e-6)
    assert _w(run, rI, n0 + rB) == 0.5


@pytest.mark.parametrize("fmt", FORMATS)
def test_istdp_same_step_pair_includes_the_minus_ai_term(fmt):
    """Eq. (15)'s - A_I S_j S_i term: inh and A firing in the SAME step weaken A <- inh to -0.38 (-0.53 without the term)."""
    run = _micro(fmt)
    n0, rA, rB, rI = _rows(run)
    _drive(run, 2, force={0: [rI, rA]})
    eI = KI * 0.1 * (1 - 0.5)
    assert _w(run, rA, n0 + rI) == pytest.approx(-0.5 - (2 * eI - 0.15), abs=1e-6) and _w(run, rA, n0 + rI) > -0.5
    assert _w(run, rB, n0 + rI) == -0.5                                        ##B silent
    assert _w(run, rI, n0 + rA) == pytest.approx(0.5, abs=1e-6)                ##E-STDP: the same-step pair cancels (Eq. 10 has no such term)


@pytest.mark.parametrize("fmt", FORMATS)
def test_istdp_off_means_estdp_on_the_inhibitory_synapses(fmt):
    """istdp=False (Table 5's E-STDP-only ablation) puts the inhibitory synapses under E-STDP: A <- inh moves like A <- B."""
    run = _micro(fmt, istdp=False); n0, rA, rB, rI = _rows(run)
    _drive(run, 4, force={0: [rB, rI], 2: [rA]})
    dB = _w(run, rA, n0 + rB) - 0.5
    assert dB > 0 and _w(run, rA, n0 + rI) == pytest.approx(-0.5 - dB, abs=1e-7)
    assert _w(run, rI, n0 + rA) == pytest.approx(_w(run, rB, n0 + rA), abs=1e-7) and _w(run, rB, n0 + rA) < 0.5   ##the reverse edges (A is pre, exc column): the same LTD


@pytest.mark.parametrize("fmt", FORMATS)
def test_istdp_window_is_symmetric(fmt):
    """Eq. (15) is symmetric in pre and post: A at 0 then inh at 2 moves the edge A <- inh by the same amount as inh at 0 then A at 2."""
    a = _micro(fmt); n0, rA, rB, rI = _rows(a); _drive(a, 4, force={0: [rI], 2: [rA]})
    b = _micro(fmt); _drive(b, 4, force={0: [rA], 2: [rI]})
    assert _w(a, rA, n0 + rI) == pytest.approx(_w(b, rA, n0 + rI), abs=1e-7) and _w(a, rA, n0 + rI) < -0.5


@pytest.mark.parametrize("fmt", FORMATS)
def test_dale_clamp_per_sign_eq16_eq17(fmt):
    """W^E in [0, W_max], W^I in [0, W_max] (stored in [-W_max, 0]). A large A_E / A_I pushes one pair past the bound: clipped. LTD from 0 stays at 0."""
    run = _micro(fmt, w_ee=2.8, w_ie=2.8, A_E=5.0, A_I=5.0)
    n0, rA, rB, rI = _rows(run)
    _drive(run, 4, force={0: [rB, rI], 2: [rA]})                              ##B and inh at 0, A at 2: A <- B potentiates, A <- inh strengthens
    assert _w(run, rA, n0 + rB) == 3.0 and _w(run, rA, n0 + rI) == -3.0
    assert 0.0 < _w(run, rB, n0 + rA) < 2.8                                    ##the reverse edge depresses, within bounds
    run0 = _micro(fmt, w_ee=0.0)
    _drive(run0, 4, force={0: [rB], 2: [rA]})
    assert _w(run0, rA, n0 + rB) == pytest.approx(0.15 * 0.1 * DE ** 2, abs=1e-6) and _w(run0, rB, n0 + rA) == 0.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_ablation_switches(fmt):
    """Table 5's ablations: estdp=False leaves the exc columns alone, istdp=False puts the inh columns under E-STDP, std=False keeps x at 1."""
    n0, rA, rB, rI = _rows(_micro(fmt))
    run = _micro(fmt, estdp=False); _drive(run, 4, force={0: [rA, rI], 2: [rB]})
    assert _w(run, rB, n0 + rA) == 0.5 and _w(run, rA, n0 + rB) == 0.5 and _w(run, rB, n0 + rI) < -0.5
    run = _micro(fmt, istdp=False); _drive(run, 4, force={0: [rA, rI], 2: [rB]})
    assert _w(run, rB, n0 + rI) < -0.5 and _w(run, rA, n0 + rI) == -0.5 and _w(run, rB, n0 + rA) > 0.5   ##B <- inh strengthened (causal pair), A <- inh untouched (A never post)
    run = _micro(fmt, std=False); tr = _drive(run, 4, force={0: [rA, rB, rI]})
    assert all(np.all(row["X"] == 1.0) for row in tr)


@pytest.mark.parametrize("fmt", FORMATS)
def test_reset_transient_restores_c_x_and_traces_keeps_w(fmt):
    """Algorithm 1 line 6 (per sample): V = 0, C = 0, x = 1, traces = 0; W (the learned liquid) is kept."""
    run = _micro(fmt)
    n0, rA, rB, rI = _rows(run)
    _drive(run, 4, force={0: [rA], 2: [rB]})
    syn_mod, pl = run.modules["synapse"], run.modules["plasticity"]
    ##sparse keeps the traces per synapse (E_pre_e ...) plus T_e; dense per neuron
    traces = [pl.E_pre, pl.E_post, pl.I1_pre, pl.I2_pre, pl.I1_post, pl.I2_post] if fmt == "dense" else [getattr(pl, n) for n in pl._EDGE_TRACES] + [pl.T_e]
    assert syn_mod.X.min() < 1.0 and traces[0].max() > 0 and traces[5].max() > 0
    w = _w(run, rB, n0 + rA)
    run.reset_transient()
    assert np.all(syn_mod.X == 1.0) and np.all(syn_mod.C == 0.0) and np.all(run.ctx.neur.V == 0.0)
    for tr in traces:
        assert np.all(tr == 0)
    assert _w(run, rB, n0 + rA) == w


# --------------------------------------------------------------------- 3. the network as built

@pytest.mark.parametrize("fmt", FORMATS)
def test_build_fixed_pm_w_in_zero_liquid_autapses_frozen_input(fmt):
    """Sec. 3.1: input weights +-W_in only (fixed in training), liquid weights 0, no self-connections, 80/20, input projections frozen."""
    pnet = D.build(n_liquid=50, n_input=40, w_in=0.7, seed=2, weights_format=fmt)
    net = pnet.network
    assert net.neuronrows["exc"].stop - net.neuronrows["exc"].start == 40 and net.neuronrows["inh"].stop - net.neuronrows["inh"].start == 10
    is_in = net.pre_idx < net.n_input
    assert set(np.unique(net.W0[is_in]).tolist()) == {-np.float32(0.7), np.float32(0.7)}
    assert np.all(net.W0[~is_in] == 0.0)
    assert np.sum(net.pre_idx == net.post_idx + net.row_offset) == 0                  ##no autapses
    assert pnet.cfg.plasticity.freeze == ("in_exc", "in_inh")
    assert pnet.cfg.simulation.update_on == ("hybrid_event" if fmt == "sparse" else "timestep")
    run = pnet.train(ModelInput.from_tonic(_Toy(6)), D.runcfg(readout_epochs=3, rate_probe=2, report_every=3), test_data=ModelInput.from_tonic(_Toy(4, seed=1)), save_run=False, save_history=False)
    W = run.ctx.syn.W
    if fmt == "dense":
        w_in_now = W[net.post_idx[is_in], net.pre_idx[is_in]]
    else:
        w_in_now = W[run.ctx.syn.pre < net.n_input]
    assert set(np.unique(w_in_now).tolist()) == {-np.float32(0.7), np.float32(0.7)}      ##fixed through training


def test_dense_equals_sparse_after_learning(tmp_path):
    """Same seed, same samples: identical state vectors in both formats and the same weights at every edge (to float tolerance)."""
    out = {}
    for fmt in FORMATS:
        pnet = D.build(n_liquid=60, n_input=40, w_in=1.0, seed=3, weights_format=fmt)
        out[fmt] = pnet.train(ModelInput.from_tonic(_Toy(10)), D.runcfg(partitions=2, readout_epochs=3, rate_probe=5, report_every=5), test_data=ModelInput.from_tonic(_Toy(4, seed=1)), out_dir=tmp_path / fmt,
                              save_run=False, save_history=False)
    sd, ss = out["dense"].ctx.syn, out["sparse"].ctx.syn
    assert np.allclose(sd.W[ss.post, ss.pre], ss.W, rtol=1e-5, atol=1e-6)
    assert (ss.W[ss.pre >= 40] != 0).sum() > 0                                           ##the liquid did learn something
    zd, zs = np.load(tmp_path / "dense" / "responses.npz"), np.load(tmp_path / "sparse" / "responses.npz")
    assert np.array_equal(zd["resp_train"], zs["resp_train"]) and np.array_equal(zd["resp_test"], zs["resp_test"])


class _FullArray(HLSPPlasticitySparse):
    """The stock event module with its (k, dw_k) scattered back into the full dw, so Run.step_sparse takes its full-array branch for the same physics."""
    def step(self, ctx):
        out = super().step(ctx)
        if isinstance(out, tuple):
            k, dw_k = out
            dw = np.zeros(ctx.syn.W.size, dtype=dw_k.dtype); dw[k] = dw_k
            return dw
        return out


@pytest.mark.parametrize("update_on", ["event", "hybrid_event"])
def test_pair_commit_equals_the_full_array_commit_bit_for_bit(update_on, tmp_path):
    """The pair commit gives the same liquid (W to the bit) and state vectors as the full-array branch; under hybrid_event also as a "timestep" run."""
    import dataclasses
    runs = [("pair", None, update_on), ("full", _FullArray, update_on)] + ([("timestep", None, "timestep")] if update_on == "hybrid_event" else [])
    out = {}
    for name, cls, u in runs:
        pnet = D.build(n_liquid=60, n_input=40, w_in=1.0, seed=3, weights_format="sparse")
        pnet.cfg.simulation = dataclasses.replace(pnet.cfg.simulation, update_on=u)      ##override the build's update_on
        if cls is not None:
            pnet.set_runtime_classes(plasticity_cls=cls)
        out[name] = pnet.train(ModelInput.from_tonic(_Toy(10)), D.runcfg(partitions=2, readout_epochs=3, rate_probe=5, report_every=5), test_data=ModelInput.from_tonic(_Toy(4, seed=1)),
                               out_dir=tmp_path / name, save_run=False, save_history=False)
    a, b = out["pair"].ctx.syn, out["full"].ctx.syn
    assert out["pair"].plasticity.update_on == update_on and isinstance(out["full"].plasticity, _FullArray) and not isinstance(out["pair"].plasticity, _FullArray)
    assert (a.W[a.pre >= 40] != 0).sum() > 0 and np.array_equal(a.W, b.W)
    z = {name: np.load(tmp_path / name / "responses.npz") for name in out}
    assert np.array_equal(z["pair"]["resp_train"], z["full"]["resp_train"]) and np.array_equal(z["pair"]["resp_test"], z["full"]["resp_test"])
    if update_on == "hybrid_event":
        assert out["timestep"].plasticity.update_on == "timestep" and np.array_equal(a.W, out["timestep"].ctx.syn.W)
        assert np.array_equal(z["pair"]["resp_train"], z["timestep"]["resp_train"]) and np.array_equal(z["pair"]["resp_test"], z["timestep"]["resp_test"])


# --------------------------------------------------------------------- 4. protocol, stimulus, readout

@pytest.mark.parametrize("fmt", FORMATS)
def test_protocol_end_to_end(fmt, tmp_path):
    """Self-organise (train) -> saved liquid -> state vectors of P partitions (eval, W guarded) -> readout; results.json + responses.npz."""
    train, test = ModelInput.from_tonic(_Toy(12)), ModelInput.from_tonic(_Toy(6, seed=1))
    pnet = D.build(n_liquid=60, n_input=40, w_in=1.0, seed=3, weights_format=fmt)
    run = pnet.train(train, D.runcfg(partitions=2, readout_epochs=30, rate_probe=5, report_every=6), test_data=test, out_dir=tmp_path, save_run=False, save_history=False)
    r = run.report
    ##12 training samples -> 2 validation (1/6, split before the liquid learns) + 10 fit
    assert r["protocol"] == "HLSPProtocol" and r["n_train"] == 10 and r["n_val"] == 2 and r["n_test"] == 6
    assert r["test_acc"] == 100.0 and r["train_acc"] == 100.0 and r["val_acc"] == 100.0  ##separable toy
    assert r["initial_rate_hz"] > 0 and r["w_at_bound_frac"] == 0.0 and run.ctx.sim.mode == "eval"
    assert not np.array_equal(run.ctx.syn.W, run._W0)                                     ##the liquid changed in the training pass
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["test_acc"] == r["test_acc"] and res["protocol_cfg"]["partitions"] == 2 and "paper_reference" in res
    z = np.load(tmp_path / "responses.npz")
    assert z["resp_train"].shape == (10, 120) and z["resp_val"].shape == (2, 120) and z["resp_test"].shape == (6, 120) and z["readout_W"].shape == (120, 2)
    assert (z["y_train"] >= 0).all() and (z["y_val"] >= 0).all() and set(z["classes"].tolist()) == {0, 1}
    assert list(tmp_path.glob("TrainedModel_*.npz"))                                      ##the self-organised liquid, saved before the eval stages
    ##the validation samples are training-set samples the liquid never self-organised on: fit and val indices are disjoint, cover the 12, and are
    ##the same for the same seed (keyed rng), so a resumed run splits identically
    r2 = pnet._prep_run(train, D.runcfg(partitions=2), test_data=test)
    pa, pb = D.HLSPProtocol(D.HLSPProtocolCfg(partitions=2), r2), D.HLSPProtocol(D.HLSPProtocolCfg(partitions=2), r2)
    fit, val = set(pa._fit_set.index.tolist()), set(pa._val_set.index.tolist())
    assert len(fit) == 10 and len(val) == 2 and not (fit & val) and fit | val == set(range(12))
    assert np.array_equal(pa._val_set.index, pb._val_set.index)
    assert "Rate" in pa.state_dict and pa.Rate.shape == (2,)      ##the rate probe's (sum, count), snapshotted as proto__Rate
    ##a state vector is ONE presentation (one begin_sample for the P partitions), P * N_liquid long
    begins, orig = [], r2._datahandler.begin_sample
    r2._datahandler.begin_sample = lambda *a, **kw: (begins.append(1), orig(*a, **kw))[1]
    r2.set_mode("eval")
    assert pa.state_vector(*train[0]).shape == (120,) and len(begins) == 1


def test_protocol_requires_test_data():
    import dataclasses
    pnet = D.build(n_liquid=50, n_input=40, w_in=1.0, seed=3)
    rc = D.runcfg(); rc = dataclasses.replace(rc, data=dataclasses.replace(rc.data, make_test_split=True, test_fraction=0.0))
    with pytest.raises(ValueError, match="test set"):
        pnet.train(ModelInput.from_tonic(_Toy(6)), rc, save_run=False)


def test_train_window_start_and_random():
    """train_window: 'start' = the first t_train_ms steps, 'random' = a keyed contiguous window, 'subsample' = sorted random steps; all reproducible."""
    pnet = D.build(n_liquid=50, n_input=40, w_in=1.0, seed=3)
    run = pnet._prep_run(ModelInput.from_tonic(_Toy(4)), D.runcfg(t_train_ms=5.0), test_data=ModelInput.from_tonic(_Toy(2, seed=1)))
    p = D.HLSPProtocol(D.HLSPProtocolCfg(t_train_ms=5.0), run)
    frames = np.arange(20)[:, None] * np.ones((1, 40), int)
    assert np.array_equal(p._train_segment(frames, 0)[:, 0], np.arange(5))
    pr = D.HLSPProtocol(D.HLSPProtocolCfg(t_train_ms=5.0, train_window="random"), run)
    seg = pr._train_segment(frames, 3)
    assert seg.shape == (5, 40) and np.array_equal(seg[:, 0], np.arange(seg[0, 0], seg[0, 0] + 5)) and 0 <= seg[0, 0] <= 15
    assert np.array_equal(seg, pr._train_segment(frames, 3))
    ps = D.HLSPProtocol(D.HLSPProtocolCfg(t_train_ms=5.0, train_window="subsample"), run)
    sub = ps._train_segment(frames, 3)[:, 0]
    assert sub.shape == (5,) and np.all(np.diff(sub) > 0) and set(sub.tolist()) <= set(range(20)) and np.array_equal(sub, ps._train_segment(frames, 3)[:, 0])
    with pytest.raises(ValueError, match="train_window"):
        D.HLSPProtocol(D.HLSPProtocolCfg(train_window="middle"), run)


def test_poisson_dataset_keyed_and_scaled_by_dataset_max():
    """PoissonDataset is keyed per sample and scaled by the DATASET's maximum: a full pixel fires at peak_hz, a half one at half, a zero one never."""
    imgs = np.zeros((3, 4, 4)); imgs[0] = 255.0; imgs[1] = 127.5
    ##PoissonDataset is a frames-only source; ModelInput holds the labels
    ds = D.ModelInput(D.PoissonDataset(imgs, T=4000, dt_ms=1.0, peak_hz=100.0, seed=5), np.array([0, 1, 2]))   ##peak given explicitly (default 1000)
    f0, y0 = ds[0]
    assert f0.shape == (4000, 16) and f0.dtype == bool and y0 == 0 and ds.n_input == 16 and ds.labels().tolist() == [0, 1, 2]
    assert np.array_equal(f0, ds[0][0]) and not np.array_equal(f0, D.PoissonDataset(imgs, T=4000, peak_hz=100.0, seed=6)[0])
    assert abs(f0.mean() - 0.1) < 0.005                                                   ##100 Hz at dt 1 ms = 0.1 per step
    assert abs(ds[1][0].mean() - 0.05) < 0.005
    assert ds[2][0].sum() == 0


def test_readout_minibatch_with_validation_plateau():
    """The shared readout's Luo settings (mini-batches, val fraction, lr halving on a plateau, explicit val set, input dropout) learn a toy."""
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (60, 5)), rng.normal(3, 1, (60, 5))]); y = np.repeat([0, 1], 60)
    ro = LogisticReadout(lr=0.05, epochs=40, batch_size=16, val_fraction=0.25, plateau_patience=2, l2=1e-4).fit(X, y)
    assert ro.score(X, y) == 1.0 and len(ro.loss_) == 40 and len(ro.val_loss_) == 40 and ro.val_loss_[-1] < ro.val_loss_[0]
    full = LogisticReadout(lr=0.05, epochs=40).fit(X, y)
    assert full.val_loss_ == [] and full.score(X, y) == 1.0
    Xv = np.concatenate([rng.normal(0, 1, (10, 5)), rng.normal(3, 1, (10, 5))]) + 100.0; yv = np.repeat([0, 1], 10)
    ex = LogisticReadout(lr=0.05, epochs=40, batch_size=16, val_fraction=0.25, l2=1e-4, dropout=0.2).fit(X, y, Xv, yv)
    assert len(ex.val_loss_) == 40 and np.allclose(ex.mean_, X.mean(0)) and ex.score(X, y) == 1.0       ##val rows (offset by 100) never enter the mean
    with pytest.raises(ValueError, match="validation labels"):
        LogisticReadout(epochs=1).fit(X, y, Xv, yv + 5)
    assert LogisticReadout(epochs=1, dropout=0.2).fit(X, y).predict(X[:3]).shape == (3,)


def test_no_where_in_new_files():
    """Caroline's rule: no `where=` in ufuncs (np.where(cond, a, b) in one-time setup is a different thing and is not matched)."""
    for f in (_ROOT / "experiment_dynamics" / "hlsp2026.py", _ROOT / "experiments" / "hlsp_lsm_2026.py"):
        assert not re.search(r"where\s*=", f.read_text()), f
