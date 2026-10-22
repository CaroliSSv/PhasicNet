#test_nalsm -- tests for the NALSM build (Ivanov & Michmizos 2021): the last-spike synapse kernel, the LIF that integrates while refractory,
# the astrocyte STDP, the per-projection clamp, the input neurons, the protocol and the readout options. Each is checked against a hand
# reference on a forced-spike micro liquid; dense and sparse must agree.
# Run: pytest system/tests/test_nalsm.py -q     (from the repo root)
from __future__ import annotations
import re
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
##system/tests/conftest.py sets sys.path for pytest
import json
import numpy as np
from modelinput import ModelInput
import pytest

from phasicnet import PhasicNet
from cfg import Cfg, SimulationCfg, NoiseCfg, PhaseCfg
from original_dynamics import TestNoise
from phases import DefaultPhase
from nalsm2021 import (NALSMSynapseCfg, NALSMIntegratorCfg, NALSMThresholdCfg, NALSMPlasticityCfg, NALSMExpSynapse, NALSMExpSynapseSparse, NALSMLIF,
                       NALSMThreshold, NALSMAstroSTDP, NALSMAstroSTDPSparse, _t_lim)
from readouts import LogisticReadout
import nalsm_2021 as D

FORMATS = ["dense", "sparse"]
M = np.exp(-0.1)                      ##trace decay per step, tau 10 ms
E1 = np.exp(-1.0)                     ##kernel decay per step, tau_u 1 ms


# --------------------------------------------------------------------- helpers

class _Toy:
    """Two classes through the paper's input neurons: channels 0..19 (class 0) or 20..39 (class 1) get p = 0.25 per step."""
    def __init__(self, n, T=250, seed=0):
        self.y = np.random.default_rng(seed).integers(0, 2, n); self.T, self.seed = T, seed
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        p = np.zeros(40); p[self.y[i] * 20:(self.y[i] + 1) * 20] = 0.25
        return D.input_neuron_spikes(np.broadcast_to(p, (self.T, 40)), np.random.default_rng([self.seed, int(i)])), int(self.y[i])


def _micro(fmt, *, w_in=0.0, w_ee=0.5, w_ei=0.5, w_ie=0.5, astro=False, decay=0.99, A_plus=0.15, astro_w=0.015, w_max=3.0, since_init=None, mode="train",
           update_on=None):
    """input(1) -> exc A, B (all_to_all, w_in); A <-> B (w_ee); exc -> inh (w_ei); inh -> exc (w_ie, stored negative). The NALSM cfg otherwise.
    Rows under post_offset: [inh, A, B]; columns: [input, inh, A, B]. Returns the prepared Run in `mode`."""
    pnet = PhasicNet(1)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", 1, True), ("exc", 2), ("inh", 1, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc="all_to_all", exc_exc="all_but_self", exc_inh="all_to_all", inh_exc="all_to_all")
    pnet.builder.set_weights({"in_exc": w_in, "exc_exc": w_ee, "exc_inh": w_ei, "inh_exc": -abs(w_ie)})
    pnet.builder.realize(network_name="micro", post_offset=True, save=False)
    sparse = fmt == "sparse"
    pnet.cfg = Cfg(simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on=update_on or ("event" if sparse else "timestep")),
                   synapse=NALSMSynapseCfg(b=None, tau_syn=1.0, since_init=since_init),
                   integrator=NALSMIntegratorCfg(tau_mem=64.0, v_rest=0.0, r=1.0, b=0.0, t_refrac=2.0),
                   threshold=NALSMThresholdCfg(v_reset=0.0, v_th=20.0, tau_th=None, adpt_th=None, t_refrac=2.0),
                   plasticity=NALSMPlasticityCfg(tau_pretrace=10.0, tau_posttrace=10.0, A_ltp=A_plus, A_ltd=A_plus, eta_exc=1.0, eta_inh=1.0, w_min=-w_max, w_max=w_max,
                                                 freeze=(), trace_inc=0.1, astro_tau=100.0, astro_w=astro_w, astro_bias_offset=0.0, decay=decay, astro=astro,
                                                 input_bounds=(-w_max, w_max)),
                   noise=NoiseCfg(0.0, 0.0), phases=(PhaseCfg("DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=NALSMExpSynapseSparse if sparse else NALSMExpSynapse, integrator_cls=NALSMLIF, threshold_cls=NALSMThreshold,
                             plasticity_cls=NALSMAstroSTDPSparse if sparse else NALSMAstroSTDP, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.cfg_name = pnet.run_name = "micro"
    data = ModelInput.from_tonic([(np.zeros((20, 1), bool), 0), (np.zeros((20, 1), bool), 1)])
    run = pnet._prep_run(data, D.runcfg(n_init=0), test_data=data)
    run.set_mode(mode)
    return run


def _rows(run):
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
    """Step through ONE sample by hand. force = {step: rows}: V of those rows is set to 25 before the step (-> a spike in that step unless refractory).
    Per-step copies of I2V, V (after the step), the liquid spikes of the step, the plasticity scalars."""
    ctx = run.ctx
    frames = np.zeros((T, ctx.net.n_input), bool) if frames is None else np.asarray(frames, bool)
    run._datahandler.begin_sample(frames, ctx)
    pl = run.modules["plasticity"]
    out = []
    for m in range(T):
        for r in (force or {}).get(m, ()):
            ctx.neur.V[r] = 25.0
        run.step()
        out.append(dict(I=ctx.upd.I2V.astype(np.float64).copy(), V=ctx.neur.V.astype(np.float64).copy(), S=ctx.neur.Spikes[ctx.net.n_input:].copy(),
                        A=float(pl.A[0]), Lr=float(pl.Lr[0])))
    return out


# --------------------------------------------------------------------- 1. synapse, membrane, threshold

@pytest.mark.parametrize("fmt", FORMATS)
def test_kernel_of_the_last_spike_input_same_step_liquid_next_step(fmt):
    """Eq. (2): input spikes act from their own step, liquid spikes one step later, decaying e^-1 per step; silent neurons add W e^-20."""
    run = _micro(fmt, w_in=5.0, w_ee=0.5, w_ei=0.7, mode="eval")
    n0, rA, rB, rI = _rows(run)
    frames = np.zeros((6, 1), bool); frames[1, 0] = True
    tr = _drive(run, 6, frames=frames, force={1: [rA]})
    silent = (0.5 + 0.0) * np.exp(-20.0)                                        ##A's input from B (0.5) and inh (-0.5): both silent at step 0 -> e^-20 each
    assert abs(tr[0]["I"][rA] - (5.0 * np.exp(-20.0) + 0.5 * np.exp(-20.0) - 0.5 * np.exp(-20.0))) < 1e-12
    assert tr[1]["I"][rA] == pytest.approx(5.0 + 0.5 * np.exp(-20.0) - 0.5 * np.exp(-20.0), abs=1e-6)      ##the input spike: W e^0 at its own step
    assert tr[1]["S"][rA] and not tr[1]["S"][rB]                                   ##A forced at step 1
    assert tr[2]["I"][rB] == pytest.approx(5.0 * E1 + 0.5 * 1.0 - 0.5 * np.exp(-20.0), abs=1e-6)          ##A's spike reaches B one step later with W e^0
    assert tr[2]["I"][rI] == pytest.approx(0.7 * 1.0 + 0.7 * np.exp(-20.0), abs=1e-6)
    assert tr[3]["I"][rB] == pytest.approx(5.0 * E1 ** 2 + 0.5 * E1 - 0.5 * np.exp(-20.0), abs=1e-6)      ##and decays with e^-1 per step
    assert _t_lim(1.0) == 20.0 and _t_lim(10.0) == 50.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_second_spike_resets_the_kernel_not_additive(fmt):
    """Two input spikes two steps apart: at the second one the kernel restarts at W e^0, not W (e^0 + e^-2)."""
    run = _micro(fmt, w_in=5.0, w_ee=0.0, w_ei=0.0, w_ie=0.0, mode="eval")
    n0, rA, rB, rI = _rows(run)
    frames = np.zeros((5, 1), bool); frames[0, 0] = frames[2, 0] = True
    tr = _drive(run, 5, frames=frames)
    assert tr[2]["I"][rA] == pytest.approx(5.0, abs=1e-6) and tr[3]["I"][rA] == pytest.approx(5.0 * E1, abs=1e-6)


@pytest.mark.parametrize("fmt", FORMATS)
def test_membrane_integrates_during_refractory_and_counter_blocks_two_steps(fmt):
    """V integrates every step, refractory or not; the refractory counter blocks the two steps after a spike (fires at 0, 3, 6)."""
    run = _micro(fmt, w_in=25.0, w_ee=0.0, w_ei=0.0, w_ie=0.0, mode="eval")
    n0, rA, rB, rI = _rows(run)
    frames = np.ones((8, 1), bool)
    tr = _drive(run, 8, frames=frames)
    assert [m for m, r in enumerate(tr) if r["S"][rA]] == [0, 3, 6]
    ##after the spike at 0 (reset to 0 at the end of step 0): V(1) = 0 + 25 - 0 = 25 (integrating, blocked), V(2) = 25 + 25 - 25/64
    assert tr[1]["V"][rA] == pytest.approx(25.0, abs=1e-4) and tr[2]["V"][rA] == pytest.approx(25.0 + 25.0 - 25.0 / 64.0, abs=1e-4)
    assert tr[3]["V"][rA] == 0.0                                                 ##fired at 3 (v ~ 74 >= 20), reset


# --------------------------------------------------------------------- 2. plasticity

@pytest.mark.parametrize("fmt", FORMATS)
def test_stdp_pairs_this_steps_input_with_last_steps_liquid(fmt):
    """STDP pairs this step's input spikes with last step's liquid spikes: post-before-pre depresses, pre-before-post potentiates."""
    run = _micro(fmt, w_in=1.0, decay=1.0)
    n0, rA, rB, rI = _rows(run)
    frames = np.zeros((6, 1), bool); frames[3, 0] = True
    _drive(run, 6, frames=frames, force={0: [rA]})
    assert _w(run, rA, 0) == pytest.approx(1.0 - 0.15 * 0.1 * M ** 2, abs=1e-6)
    assert _w(run, rB, 0) == 1.0 and _w(run, rB, n0 + rA) == 0.5 and _w(run, rA, n0 + rB) == 0.5 and _w(run, rI, n0 + rA) == 0.5
    run = _micro(fmt, w_in=1.0, decay=1.0)
    frames = np.zeros((6, 1), bool); frames[0, 0] = True
    _drive(run, 6, frames=frames, force={2: [rA]})
    assert _w(run, rA, 0) == pytest.approx(1.0 + 0.15 * 0.1 * M ** 3, abs=1e-6)
    assert _w(run, rB, 0) == 1.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_stdp_liquid_pair_and_inhibitory_role_swap(fmt):
    """Liquid pairs: pre-before-post potentiates, post-before-pre depresses; on an inhibitory column (stored negative) the sign of dW swaps."""
    run = _micro(fmt, decay=1.0)
    n0, rA, rB, rI = _rows(run)
    _drive(run, 6, force={0: [rA], 2: [rB]})
    dw = 0.15 * 0.1 * M ** 2
    assert _w(run, rB, n0 + rA) == pytest.approx(0.5 + dw, abs=1e-6) and _w(run, rA, n0 + rB) == pytest.approx(0.5 - dw, abs=1e-6)
    assert _w(run, rI, n0 + rA) == 0.5 and _w(run, rI, n0 + rB) == 0.5 and _w(run, rA, n0 + rI) == -0.5
    run = _micro(fmt, decay=1.0)
    _drive(run, 6, force={0: [rI], 2: [rA]})
    assert _w(run, rA, n0 + rI) == pytest.approx(-0.5 - dw, abs=1e-6)                   ##-A- S_post x_pre on the inhibitory column
    assert _w(run, rI, n0 + rA) == pytest.approx(0.5 - dw, abs=1e-6)                    ##exc -> inh: post (inh) before pre (A): depression
    run = _micro(fmt, decay=1.0)
    _drive(run, 6, force={0: [rA], 2: [rI]})
    assert _w(run, rA, n0 + rI) == pytest.approx(-0.5 + dw, abs=1e-6)                   ##+A+ y_post S_pre: weaker inhibition


@pytest.mark.parametrize("fmt", FORMATS)
def test_astrocyte_tracks_liquid_minus_input_spikes_and_decays(fmt):
    """Eq. (7): the astrocyte tracks liquid-minus-input spikes with 0.99 decays per step (hand recursion); decay off keeps Lr at 0.15."""
    run = _micro(fmt, w_in=0.0, astro=True)
    n0, rA, rB, rI = _rows(run)
    frames = np.zeros((6, 1), bool); frames[1, 0] = frames[4, 0] = True
    tr = _drive(run, 6, frames=frames, force={0: [rA, rB], 3: [rI]})
    liq = {1: 2, 4: 1}                                                              ##their S_t liquid counts: forced at 0 -> t = 1; at 3 -> t = 4
    inp = {1: 1, 4: 1}
    A, w, b, lr = 0.15, 0.015, 0.15, 0.15
    for t in range(6):
        A = A + (w * (liq.get(t, 0) - inp.get(t, 0)) + b - A) / 100.0
        A *= 0.99; w *= 0.99; b *= 0.99; lr *= 0.99
        assert tr[t]["A"] == pytest.approx(A, abs=1e-6) and tr[t]["Lr"] == pytest.approx(lr, abs=1e-6), t
    run = _micro(fmt, astro=True); run.plasticity.decay_active = False
    tr = _drive(run, 4, force={0: [rA]})
    assert all(r["Lr"] == pytest.approx(0.15) for r in tr) and tr[1]["A"] == pytest.approx(0.15 + (0.015 * 1 + 0.15 - 0.15) / 100.0, abs=1e-7)


@pytest.mark.parametrize("fmt", FORMATS)
def test_clamp_input_either_sign_liquid_per_sign(fmt):
    """Sec. 2.1 bounds: IL [-3, 3] (an input weight may cross zero), E [0, 3], I [-3, 0]. A large A+ pushes pairs past the bounds."""
    run = _micro(fmt, w_in=0.005, w_ee=2.9, w_ie=2.9, A_plus=5.0, decay=1.0)
    n0, rA, rB, rI = _rows(run)
    frames = np.zeros((6, 1), bool); frames[3, 0] = True
    _drive(run, 6, frames=frames, force={0: [rA, rI], 2: [rB]})
    assert _w(run, rA, 0) == pytest.approx(0.005 - 5.0 * 0.1 * M ** 2, abs=1e-6) and _w(run, rA, 0) < 0     ##input crossed zero: allowed
    assert _w(run, rB, n0 + rA) == 3.0                                            ##E pair pre-before-post: 2.9 + 0.409 -> clipped to 3
    assert _w(run, rB, n0 + rI) == -3.0                                           ##I column, inh before B: -2.9 - 0.409 -> clipped to -3
    run = _micro(fmt, w_ee=0.0, A_plus=5.0, decay=1.0)
    _drive(run, 6, force={0: [rA], 2: [rB]})
    assert _w(run, rA, n0 + rB) == 0.0 and _w(run, rB, n0 + rA) > 0.0            ##E floor at 0
    ##the engine's plan: the input projection carries the override, the liquid the per-sign bounds
    if fmt == "dense":
        plan = {tuple(p[:2]): p[2:] for p in run._clamp_plan}
        assert any(lo == -3.0 and hi == 3.0 for lo, hi in plan.values()) and any(lo == 0.0 and hi == 3.0 for lo, hi in plan.values()) and any(lo == -3.0 and hi == 0.0 for lo, hi in plan.values())
    else:
        lo, hi = run._clamp_plan; syn = run.ctx.syn
        assert np.all(lo[syn.pre < n0] == -3.0) and np.all(hi[syn.pre < n0] == 3.0) and np.all(lo[syn.pre == n0 + rI] == -3.0) and np.all(hi[syn.pre == n0 + rI] == 0.0)


@pytest.mark.parametrize("fmt", FORMATS)
def test_reset_transient_restores_counters_traces_astrocyte_keeps_w(fmt):
    run = _micro(fmt, astro=True)
    n0, rA, rB, rI = _rows(run)
    _drive(run, 3, force={0: [rA], 2: [rB]})                                      ##ends right after B's spike: S_prev holds it
    syn_mod, pl = run.modules["synapse"], run.modules["plasticity"]
    X, Y = (pl.X_pre, pl.Y_post) if fmt == "dense" else (pl.X_e, pl.Y_e)         ##sparse keeps the traces per synapse
    assert syn_mod.Last[n0 + rA] < 20.0 and X.max() > 0 and pl.Lr[0] < 0.15 and pl.S_prev.any()
    w = _w(run, rB, n0 + rA)
    run.reset_transient()
    assert np.all(syn_mod.Last == 20.0) and np.all(X == 0) and np.all(Y == 0) and not pl.S_prev.any()
    if fmt == "sparse":
        assert not pl.T_e.any()
    assert pl.Lr[0] == np.float32(0.15) and pl.A[0] == np.float32(0.15) and pl.Aw[0] == np.float32(0.015) and pl.Ab[0] == np.float32(0.15)
    assert _w(run, rB, n0 + rA) == w


@pytest.mark.parametrize("fmt", FORMATS)
def test_since_init_zero_means_a_full_kick(fmt):
    """since_init = 0 (the other reading of the code's reset scalar): every presynaptic neuron counts as 'just spiked' at step 0."""
    run = _micro(fmt, w_in=0.0, w_ee=0.5, w_ei=0.7, w_ie=0.4, since_init=0.0, mode="eval")
    n0, rA, rB, rI = _rows(run)
    tr = _drive(run, 2)
    assert tr[0]["I"][rA] == pytest.approx(0.5 - 0.4, abs=1e-6) and tr[0]["I"][rI] == pytest.approx(1.4, abs=1e-6)
    assert tr[1]["I"][rA] == pytest.approx((0.5 - 0.4) * E1, abs=1e-6)


# --------------------------------------------------------------------- 3. the input neurons and the network

def test_input_neurons_follow_the_code_step_for_step():
    """p = 1: hit every step, current 100 -> spike at step 1 (v from step 0), refractory 2 -> 4, 7, ...; p = 0: silent; keyed = reproducible."""
    T = 12
    s = D.input_neuron_spikes(np.ones((T, 3)) * np.array([1.0, 0.0, 1.0]), np.random.default_rng(0))
    assert s.shape == (T, 3) and s.dtype == bool
    assert np.flatnonzero(s[:, 0]).tolist() == [1, 4, 7, 10] and not s[:, 1].any() and np.array_equal(s[:, 0], s[:, 2])
    p = np.full((4000, 2), 0.3)
    a = D.input_neuron_spikes(p, np.random.default_rng([5, 1])); b = D.input_neuron_spikes(p, np.random.default_rng([5, 1]))
    assert np.array_equal(a, b) and 0.2 < a.mean() < 0.3                           ##a refractory of 2 caps the rate below 1/3 per step


def test_normalise_images_and_lazy_dataset():
    imgs = np.zeros((3, 2, 2)); imgs[0] = [[2, 2], [0, 0]]; imgs[1] = [[1, 1], [1, 1]]; imgs[2] = [[0, 0], [0, 4]]
    p, scale = D.normalise_images(imgs)
    assert scale == 1.0 and p[0].tolist() == [0.5, 0.5, 0, 0] and p[1].tolist() == [0.25] * 4 and p[2].tolist() == [0, 0, 0, 1.0]   ##sum-normalised, / global max
    p2, _ = D.normalise_images(imgs[:2], scale=1.0)
    assert p2[0].tolist() == [0.5, 0.5, 0, 0]
    ds = D.ModelInput(D.NALSMInputs(imgs, T=30, seed=2), np.array([0, 1, 2]))   ##frames-only source
    f, y = ds[2]
    assert f.shape == (30, 4) and y == 2 and ds.n_input == 4 and np.array_equal(f, ds[2][0])
    assert np.flatnonzero(f[:, 3]).tolist() == [1, 4, 7, 10, 13, 16, 19, 22, 25, 28] and not f[:, :3].any()


@pytest.mark.parametrize("fmt", FORMATS)
def test_build_maximal_weights_kernel_blocks_grid_check(fmt):
    """NALSM / LSM+STDP start at +-3 (input: random sign; liquid: Dale), the LSM baseline at one value; C by block; the grid must hold the liquid."""
    pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=2, weights_format=fmt)
    net = pnet.network
    is_in = net.pre_idx < net.n_input
    assert set(np.unique(net.W0[is_in]).tolist()) == {-3.0, 3.0}
    inh_pre = net.inh_neuron_mask[net.pre_idx]
    assert np.all(net.W0[~is_in & inh_pre] == -3.0) and np.all(net.W0[~is_in & ~inh_pre] == 3.0)
    assert np.sum(net.pre_idx == net.post_idx + net.row_offset) == 0                 ##no self-connections (the code removes the diagonal)
    assert pnet.cfg.plasticity.input_bounds == (-3.0, 3.0) and pnet.cfg.plasticity.freeze == () and pnet.cfg.plasticity.astro
    c = net.synapse_counts
    assert c["n_inh_exc"] / (12 * 48) > c["n_exc_inh"] / (48 * 12)                 ##C_IE .3 > C_EI .1 at equal pair counts
    lsm = D.build(model="lsm", lsm_weight=0.8, n_liquid=60, grid=(5, 4, 3), n_input=40, seed=2, weights_format=fmt)
    assert set(np.unique(lsm.network.W0).tolist()) == {-np.float32(0.8), np.float32(0.8)} and type(lsm.cfg.plasticity).__name__ == "NoPlasticityCfg"
    stdp = D.build(model="lsm_stdp", n_liquid=60, grid=(5, 4, 3), n_input=40, seed=2, weights_format=fmt)
    assert not stdp.cfg.plasticity.astro
    with pytest.raises(ValueError, match="grid"):
        D.build(n_liquid=60, grid=(5, 4, 4), n_input=40)


# --------------------------------------------------------------------- 4. protocol and readout

@pytest.mark.parametrize("fmt", FORMATS)
def test_protocol_end_to_end(fmt, tmp_path):
    """Initialization pass (decay off, W accumulates) -> W_ini saved -> counting from W_ini per sample with plasticity on -> readout."""
    train, test = ModelInput.from_tonic(_Toy(12)), ModelInput.from_tonic(_Toy(6, seed=1))
    pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=3, weights_format=fmt)
    run = pnet.train(train, D.runcfg(n_init=10, readout_epochs=60, readout_patience=None, report_every=5), test_data=test, out_dir=tmp_path, save_run=False, save_history=False)
    r = run.report
    assert r["protocol"] == "NALSMProtocol" and r["n_train"] == 12 and r["n_test"] == 6 and r["n_val"] == 2
    assert r["test_acc"] == 100.0 and r["train_acc"] == 100.0
    assert r["init_rate_hz_first200"] > 0 and r["astro_A_end_of_sample_first200"] < 0.15 and r["readout_epochs_run"] == 60
    assert run.ctx.sim.mode == "train" and run.plasticity.decay_active
    z = np.load(tmp_path / "responses.npz")
    assert z["resp_train"].shape == (12, 60) and z["resp_test"].shape == (6, 60) and z["readout_W"].shape == (60, 2)
    W_ini = z["W_ini"]
    assert not np.array_equal(W_ini, run._W0)                                     ##the initialization pass changed W
    assert np.abs(W_ini[W_ini != 0]).min() >= 0.0 and np.abs(W_ini).max() <= 3.0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["test_acc"] == r["test_acc"] and res["protocol_cfg"]["n_init"] == 10 and "paper_reference" in res
    assert list(tmp_path.glob("TrainedModel_*.npz"))
    ##counting restores W_ini before every sample: a second count of the same sample is identical
    p = run._protocol_for_test if hasattr(run, "_protocol_for_test") else None
    proto = D.NALSMProtocol(D.NALSMProtocolCfg(n_init=10), run); proto.W_ini[...] = W_ini
    assert {"Init_rate", "Astro"} <= proto.state_dict.keys() and proto.Astro.shape == (2,)   ##the (sum, count) pairs are public arrays, snapshotted
    f0, y0 = run.train_set[0]   ##the run presents train in a seeded shuffled order
    c1 = proto.count_sample(f0, y0); c2 = proto.count_sample(f0, y0)
    assert np.array_equal(c1, c2) and np.array_equal(c1, z["resp_train"][0])


def test_dense_equals_sparse_after_initialization_and_counting(tmp_path):
    out = {}
    for fmt in FORMATS:
        pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=3, weights_format=fmt)
        out[fmt] = pnet.train(ModelInput.from_tonic(_Toy(10)), D.runcfg(n_init=8, readout_epochs=3, readout_patience=None, report_every=5), test_data=ModelInput.from_tonic(_Toy(4, seed=1)),
                              out_dir=tmp_path / fmt, save_run=False, save_history=False)
    zd, zs = np.load(tmp_path / "dense" / "responses.npz"), np.load(tmp_path / "sparse" / "responses.npz")
    ss = out["sparse"].ctx.syn
    assert np.allclose(zd["W_ini"][ss.post, ss.pre], zs["W_ini"], rtol=1e-5, atol=1e-6)   ##lazy decay -> float tolerance
    assert np.array_equal(zd["resp_train"], zs["resp_train"]) and np.array_equal(zd["resp_test"], zs["resp_test"])
    assert zd["resp_train"].sum() > 0


class _FullArray(NALSMAstroSTDPSparse):
    """The stock event module with (k, dw_k) scattered into a full (n_synapses,) dw, so Run.step_sparse takes its full-array branch."""
    def step(self, ctx):
        out = super().step(ctx)
        if isinstance(out, tuple):
            k, dw_k = out
            dw = np.zeros(ctx.syn.W.size, dtype=dw_k.dtype); dw[k] = dw_k
            return dw
        return out


@pytest.mark.parametrize("update_on", ["event", "hybrid_event"])   ##both event modes return the pair
def test_pair_commit_equals_the_full_array_commit_bit_for_bit(update_on, tmp_path):
    """The (k, dw_k) pair commit gives the same W_ini and counts to the bit as the full-array commit (and as "timestep" under hybrid_event)."""
    import dataclasses
    runs = [("pair", None, update_on), ("full", _FullArray, update_on)] + ([("timestep", None, "timestep")] if update_on == "hybrid_event" else [])
    out = {}
    for name, cls, u in runs:
        pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=3, weights_format="sparse")
        pnet.cfg.simulation = dataclasses.replace(pnet.cfg.simulation, update_on=u)      ##the build's SimulationCfg says "event" for sparse
        if cls is not None:
            pnet.set_runtime_classes(plasticity_cls=cls)
        out[name] = pnet.train(ModelInput.from_tonic(_Toy(10)), D.runcfg(n_init=8, readout_epochs=3, readout_patience=None, report_every=5), test_data=ModelInput.from_tonic(_Toy(4, seed=1)),
                               out_dir=tmp_path / name, save_run=False, save_history=False)
    z = {name: np.load(tmp_path / name / "responses.npz") for name in out}
    za, zb = z["pair"], z["full"]
    assert out["pair"].plasticity.update_on == update_on and isinstance(out["full"].plasticity, _FullArray) and not np.array_equal(za["W_ini"], out["pair"]._W0)
    assert np.array_equal(za["W_ini"], zb["W_ini"]) and np.array_equal(out["pair"].ctx.syn.W, out["full"].ctx.syn.W)
    assert np.array_equal(za["resp_train"], zb["resp_train"]) and np.array_equal(za["resp_test"], zb["resp_test"]) and za["resp_train"].sum() > 0
    if update_on == "hybrid_event":
        zt = z["timestep"]
        assert out["timestep"].plasticity.update_on == "timestep" and np.array_equal(za["W_ini"], zt["W_ini"]) and np.array_equal(out["pair"].ctx.syn.W, out["timestep"].ctx.syn.W)
        assert np.array_equal(za["resp_train"], zt["resp_train"]) and np.array_equal(za["resp_test"], zt["resp_test"])


@pytest.mark.parametrize("fmt", FORMATS)
def test_lsm_baseline_and_static_counting(fmt):
    """model 'lsm': no initialization, fixed weights (eval mode); plastic_counting=False with NALSM: W_ini fixed during counting."""
    lsm = D.build(model="lsm", n_liquid=60, grid=(5, 4, 3), n_input=40, seed=3, weights_format=fmt)
    run = lsm.train(ModelInput.from_tonic(_Toy(6)), D.runcfg(n_init=0, readout_epochs=3, readout_patience=None), test_data=ModelInput.from_tonic(_Toy(4, seed=1)), save_run=False, save_history=False)
    assert run.ctx.sim.mode == "eval" and np.array_equal(run.ctx.syn.W, run._W0) and run.report["init_rate_hz_first200"] is None
    na = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=3, weights_format=fmt)
    run = na.train(ModelInput.from_tonic(_Toy(6)), D.runcfg(n_init=4, plastic_counting=False, readout_epochs=3, readout_patience=None), test_data=ModelInput.from_tonic(_Toy(4, seed=1)), save_run=False, save_history=False)
    assert run.ctx.sim.mode == "eval" and not np.array_equal(run.ctx.syn.W, run._W0)


def test_protocol_requires_test_data_and_validates_window():
    import dataclasses
    pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=3)
    rc = D.runcfg(); rc = dataclasses.replace(rc, data=dataclasses.replace(rc.data, make_test_split=True, test_fraction=0.0))
    with pytest.raises(ValueError, match="test set"):
        pnet.train(ModelInput.from_tonic(_Toy(6)), rc, save_run=False)
    run = pnet._prep_run(ModelInput.from_tonic(_Toy(4)), D.runcfg(), test_data=ModelInput.from_tonic(_Toy(2, seed=1)))
    with pytest.raises(ValueError, match="init_window"):
        D.NALSMProtocol(D.NALSMProtocolCfg(init_window="middle"), run)
    p = D.NALSMProtocol(D.NALSMProtocolCfg(n_init=7, init_window="random", t_init_ms=5.0), run)
    order = p._init_order()
    assert len(order) == 7 and sorted(order[:4].tolist()) == [0, 1, 2, 3] and np.array_equal(order, p._init_order())
    frames = np.arange(20)[:, None] * np.ones((1, 40), int)
    seg = p._snapshot(frames, 2)
    assert seg.shape == (5, 40) and np.array_equal(seg[:, 0], np.arange(seg[0, 0], seg[0, 0] + 5)) and np.array_equal(seg, p._snapshot(frames, 2))


def test_normalise_counts_and_readout_soft_targets_val_selection():
    R = np.array([[2, 2, 0], [0, 0, 5], [0, 0, 0]], dtype=np.int16)
    X, scale = D.NALSMProtocol.normalise_counts(R)
    assert scale == 1.0 and X[0].tolist() == [0.5, 0.5, 0.0] and X[1].tolist() == [0, 0, 1.0] and X[2].tolist() == [0, 0, 0]
    X2, _ = D.NALSMProtocol.normalise_counts(R[:1], scale=2.0)
    assert X2[0].tolist() == [0.25, 0.25, 0.0]
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (60, 5)), rng.normal(3, 1, (60, 5))]); y = np.repeat([0, 1], 60)
    perm = rng.permutation(120); X, y = X[perm], y[perm]
    ro = LogisticReadout(lr=0.05, epochs=300, batch_size=16, val_fraction=0.25, val_from_end=True, select_by_val_acc=True, early_stop_patience=10,
                         soft_target_logit=1.0, standardize=False).fit(X, y)
    assert ro.score(X, y) == 1.0 and len(ro.val_acc_) == len(ro.loss_) < 300 and max(ro.val_acc_) == 1.0
    ##soft targets keep the fitted probabilities away from 1
    assert ro.predict_proba(X).max() < 0.999


def test_no_where_in_new_files():
    for f in (_ROOT / "experiment_dynamics" / "nalsm2021.py", _ROOT / "experiments" / "nalsm_2021.py"):
        assert not re.search(r"where\s*=", f.read_text()), f
