#test_deckers -- tests for the Deckers 2022 ELSM build: grids and the gaussian kernel, conductance synapse delays (dense and sparse), the
# refractory hold, the SFA threshold, no-op plasticity, the fixed-liquid protocol and the logistic readout, each against a hand reference.
# Run: pytest system/tests/test_deckers.py -q     (from the repo root)
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
from dynamicscfg import DynamicsCfg, SimulationCfg, NoiseCfg, PhaseCfg
from network import _canon_to_specs, NeuronSpec
from original_dynamics import TestNoise
from phases import DefaultPhase
from deckers2022 import (ELSMSynapseCfg, ELSMIntegratorCfg, ELSMThresholdCfg, NoPlasticityCfg, ELSMConductanceSynapse, ELSMConductanceSynapseSparse,
                         ELSMConductanceLIF, SFAThreshold, NoPlasticity, steps_of)
from readouts import LogisticReadout, fit_evaluate, readout_report
import deckers_2022 as D

FORMATS = ["dense", "sparse"]


# --------------------------------------------------------------------- helpers

def _data(n_in=40, n=24, T=60, seed=0):
    """Three classes, each with its own 10 hot input channels: separable by exc spike counts."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lab = i % 3
        p = np.full(n_in, 0.02); p[lab * 10:(lab + 1) * 10] = 0.25
        out.append((rng.random((T, n_in)) < p, lab))
    return ModelInput.from_tonic(out)


def _micro(fmt, *, delays, w_in=5.0, w_rec=0.0, elsm=True, dt=1.0, t_refrac=3.0, e_exc=40.0):
    """A 2-neuron liquid: input(1) -> exc(2) all_to_all, exc -> exc both directions (w_rec = 0 switches it off); everything else is Deckers' cfg.
    Returns the prepared Run in eval mode."""
    pnet = PhasicNet(1)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", 1, True), ("exc", 2), ("inh", 1, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"))
    pnet.builder.set_connectivity(in_exc="all_to_all", exc_exc="all_but_self", exc_inh="all_to_all")
    pnet.builder.set_weights({"in_exc": w_in, "exc_exc": w_rec, "exc_inh": 0.0})
    pnet.builder.realize(network_name="micro", post_offset=True, save=False)
    pnet.cfg = DynamicsCfg(simulation=SimulationCfg(dt=dt, weights_format=fmt, update_on="timestep"),
                   synapse=ELSMSynapseCfg(tau_ge=3.0, tau_gi=10.0, e_exc=e_exc, e_inh=0.0, delays=delays),
                   integrator=ELSMIntegratorCfg(tau_mem=30.0, v_rest=13.5, r=1.0, tau_range=None, t_refrac=t_refrac, t_refrac_inh=2.0, v_init=None),
                   threshold=ELSMThresholdCfg(v_reset=13.5, v_th=15.0, tau_th=550.0 if elsm else None, adpt_th=1.0 if elsm else None, t_refrac=t_refrac,
                                              sfa_range=None, sfa_exc_only=True, t_refrac_inh=2.0),
                   plasticity=NoPlasticityCfg(), noise=NoiseCfg(0.0, 0.0), phases=(PhaseCfg("DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=ELSMConductanceSynapseSparse if fmt == "sparse" else ELSMConductanceSynapse, integrator_cls=ELSMConductanceLIF,
                             threshold_cls=SFAThreshold, plasticity_cls=NoPlasticity, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.cfg_name = pnet.run_name = "micro"
    data = ModelInput.from_tonic([(np.zeros((20, 1), bool), 0), (np.zeros((20, 1), bool), 1)])
    run = pnet._prep_run(data, D.runcfg(), test_data=data)
    run.set_mode("eval")
    return run


def _trace(run, frames):
    """Step through ONE sample by hand; returns per-step copies of (I2V, V, V_th, rec spikes) BEFORE the step's commit is overwritten."""
    ctx = run.ctx
    run._datahandler.begin_sample(np.asarray(frames, bool), ctx)
    rows = []
    for _ in range(frames.shape[0]):
        run.step()
        rows.append((ctx.upd.I2V.copy(), ctx.neur.V.copy(), run.threshold.V_th.copy(), ctx.neur.Spikes[ctx.net.n_input:].copy()))
    return rows


# --------------------------------------------------------------------- 1. the spatial builder step

def test_gaussian_kernel_degree_matches_deckers_on_the_2000_grid():
    """Table 2: lambda = 2 on a unit 20x10x10 grid with C_EE/EI/IE/II = .3/.2/.4/.1 gives ~9.5 recurrent synapses per neuron (Maass 2002 rule)."""
    pnet = D.build(weights_format="sparse", seed=3)
    net = pnet.network
    rec = sum(v for k, v in net.synapse_counts.items() if not k.startswith("n_in_"))
    assert 9.0 < rec / 2000 < 10.1, rec / 2000
    ##the input projection: ~7 channels per liquid neuron
    assert 6.0 < (net.synapse_counts["n_in_exc"] + net.synapse_counts["n_in_inh"]) / 2000 < 8.0


def test_kernel_is_distance_dependent_and_shared_grid_cells_are_unique():
    """Near pairs are wired far more often than distant ones; exc and inh never share a cell on the shared grid; a plain probability ignores distance."""
    from builder import _grid_positions
    pnet = PhasicNet(5); pnet._instantiate_builder(); b = pnet.builder
    b.set_neurons(("input", 4, True), ("exc", 160), ("inh", 40, False, True))
    b.set_positions(exc=(10, 10, 2), inh=(10, 10, 2), share="liquid")
    b.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"))
    b.set_connectivity(in_exc=0.5, exc_exc={"kernel": "gaussian", "C": 0.9, "lambda": 1.5})
    b.set_weights({"in_exc": 1.0, "exc_exc": 1.0})
    net = b.realize(network_name="k", post_offset=True, save=False)
    pos = _grid_positions(b.neuronspecs, pnet.rng)
    cells = np.concatenate([pos["exc"], pos["inh"]])
    assert len(np.unique(cells, axis=0)) == 200
    ex = net.neuronslices["exc"]
    edges = net.syntype == net.type_key["exc_exc"]
    pre, post = net.pre_idx[edges] - ex.start, net.post_idx[edges] + net.row_offset - ex.start   ##post_idx are W rows
    d = np.linalg.norm(pos["exc"][pre] - pos["exc"][post], axis=1)
    assert d.max() <= 5.0 and np.mean(d <= 1.0) > 0.25, (d.max(), np.mean(d <= 1.0))   ##P(D=5) ~ 1e-5; a plain p would reach max D ~ 12.8
    ##expected out-degree ~ sum over the lattice of C exp(-(D/lambda)^2)
    deg = edges.sum() / 160
    diff = pos["exc"][:, None, :] - pos["exc"][None, :, :]
    P = 0.9 * np.exp(-(diff ** 2).sum(-1) / 1.5 ** 2); np.fill_diagonal(P, 0.0)
    assert abs(deg - P.sum(1).mean()) < 0.6, (deg, P.sum(1).mean())


def test_spatial_spec_round_trip_and_grid_fields_canonical():
    """A spatial spec survives JSON (grid list -> tuple, kernel dict kept), every group carries the grid keys, and the D&C micro ids are pinned."""
    pnet = PhasicNet(2); pnet._instantiate_builder(); b = pnet.builder
    b.set_neurons(("input", 2, True), ("exc", 6)); b.set_positions(exc=(3, 2)); b.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"))
    b.set_connectivity(in_exc=1.0, exc_exc={"kernel": "gaussian", "C": 0.5, "lambda": 1.0}); b.set_weights({"in_exc": 1.0, "exc_exc": 0.5})
    raw = json.loads(json.dumps({"spec_name": "s", **b._canonicalize_netspecs()}))
    _, ns, ss = _canon_to_specs(raw)
    assert ns["exc"].grid == (3, 2) and isinstance(ns["exc"].grid, tuple) and ns["exc"].grid_share is None
    assert ss["exc_exc"].p_connection == {"kernel": "gaussian", "C": 0.5, "lambda": 1.0}
    assert raw["neuronspecs"]["exc"]["grid"] == [3, 2] and raw["neuronspecs"]["input"]["grid"] is None     ##every group carries the keys
    ##the loaded specs realize the same network as the originals (positions and kernel draw from keyed streams)
    net_a = b.realize(network_name="a", save=False)
    q = PhasicNet(2); q._instantiate_builder(); q.builder.neuronspecs, q.builder.synapsespecs = ns, ss
    net_b = q.builder.realize(network_name="a", save=False)
    assert net_a.network_id == net_b.network_id
    ##drift check: the D&C micro network's ids
    p = PhasicNet(7); p._instantiate_builder(); bb = p.builder
    bb.set_neurons(("input", 32, True), ("exc", 6), ("inh", 6, False, True)); bb.set_synapses(("in_exc", "input", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    bb.set_connectivity(in_exc="all_to_all", exc_inh="one_to_one", inh_exc="all_but_self"); bb.set_weights({"in_exc": (0.003, 0.303), "exc_inh": 10.4, "inh_exc": -17.0})
    net = bb.realize(network_name="x", post_offset=True, save=False)
    assert net.spec_id == "b12d5680f9f669f0" and net.network_id == "7c7bc0fd82d5eaca9ee5fb7e4dc8fc08"
    assert NeuronSpec("a", 3).grid is None


def test_kernel_validation():
    pnet = PhasicNet(2); pnet._instantiate_builder(); b = pnet.builder
    b.set_neurons(("input", 2, True), ("exc", 6)); b.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc")); b.set_weights({"in_exc": 1.0, "exc_exc": 0.5})
    with pytest.raises(ValueError):
        b.set_connectivity(exc_exc={"kernel": "box", "C": 0.5, "lambda": 1.0})
    with pytest.raises(ValueError):
        b.set_connectivity(exc_exc={"kernel": "gaussian", "C": 1.5, "lambda": 1.0})
    with pytest.raises(ValueError):
        b.set_connectivity(exc_exc={"kernel": "gaussian", "C": 0.5})
    b.set_connectivity(in_exc=1.0, exc_exc={"kernel": "gaussian", "C": 0.5, "lambda": 1.0})
    with pytest.raises(ValueError, match="no grid"):            ##kernel without positions
        b.realize(network_name="x", save=False)
    with pytest.raises(ValueError, match="does not fit"):       ##grid too small
        b.set_positions(exc=(2, 2))
    b.set_positions(exc=(3, 2))
    b.realize(network_name="x", save=False)
    ##shared grids: same shape required, and the sum of the groups must fit
    b.set_neurons(("inh", 3, False, True))
    with pytest.raises(ValueError, match="same grid"):
        b.set_positions(exc=(3, 2), inh=(2, 2), share="s")
    with pytest.raises(ValueError, match="cells for"):
        b.set_positions(exc=(3, 2), inh=(3, 2), share="s")


# --------------------------------------------------------------------- 2. the conductance synapse: delays (Brian 2 timing)

@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("d_ms", [0.0, 1.0, 2.0])
def test_input_delay_lands_one_step_after_brian_emission(fmt, d_ms):
    """An INPUT spike of step m with delay d steps first reaches the current at step m + 1 + d (Brian 2 timing), as I2V = w (E_exc - V)."""
    run = _micro(fmt, delays={"in_exc": d_ms}, w_in=0.2, elsm=False)   ##0.2 * 26.5 mV = 5.3 mV drive: V moves, no spike
    ex = run.ctx.net.neuronrows["exc"].start                    ##rows under post_offset: inh first, then exc
    frames = np.zeros((8, 1), bool); frames[0, 0] = True
    rows = _trace(run, frames)
    I = np.array([r[0][ex] for r in rows])
    first = int(np.flatnonzero(I != 0)[0])
    assert first == 1 + steps_of(d_ms, 1.0), (first, I)
    assert I[first] == pytest.approx(0.2 * (40.0 - 13.5), rel=1e-6)
    assert rows[first][1][ex] > 13.5 and rows[first - 1][1][ex] == 13.5    ##V moves on that step, not before


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("d_ms", [0.0, 1.0, 2.0])
def test_recurrent_delay_counts_from_the_spike_step(fmt, d_ms):
    """A RECURRENT spike fired at step s with delay d first reaches the target's current at step s + 1 + d."""
    run = _micro(fmt, delays={"in_exc": 0.0, "exc_exc": d_ms}, w_in=50.0, w_rec=2.0, elsm=False)
    ctx = run.ctx; net = ctx.net
    ex = net.neuronrows["exc"].start; e0 = net.neuronslices["exc"].start - net.n_input      ##row of exc neuron 0 / its index in Spikes[n_input:]
    r1 = ex + 1
    if fmt == "dense":
        ctx.syn.W[r1, 0] = 0.0                                  ##neuron 1 hears nothing from the input
    else:
        e = np.flatnonzero((ctx.syn.post == r1) & (ctx.syn.pre == 0)); ctx.syn.W[e] = 0.0
    frames = np.zeros((10, 1), bool); frames[0, 0] = True
    rows = _trace(run, frames)
    sp0 = np.array([r[3][e0] for r in rows]); I1 = np.array([r[0][r1] for r in rows])
    s = int(np.flatnonzero(sp0)[0])                             ##neuron 0's spike step
    first = int(np.flatnonzero(I1 != 0)[0])
    assert first == s + 1 + steps_of(d_ms, 1.0), (s, first, I1)
    assert I1[first] == pytest.approx(2.0 * (40.0 - 13.5), rel=1e-6)


@pytest.mark.parametrize("fmt", FORMATS)
def test_conductances_decay_euler_and_inhibitory_edges_feed_gi(fmt):
    """After one arrival: Ge(t+1) = Ge(t) (1 - dt / tau_ge) (Euler); an inhibitory pre (stored negative) raises Gi, not Ge, by |w|."""
    run = _micro(fmt, delays={"in_exc": 0.0, "exc_exc": 1.5}, w_in=0.2, elsm=False)
    syn = run.synapse
    ex = run.ctx.net.neuronrows["exc"].start
    frames = np.zeros((6, 1), bool); frames[0, 0] = True
    rows = _trace(run, frames)
    I = np.array([r[0][ex] for r in rows]); V = np.array([r[1][ex] for r in rows])
    ##I(t) = Ge(t) (E - V(t)) with V(t) = the V recorded after step t-1; Ge decays by (1 - 1/3) per step
    ge = I[1:5] / (40.0 - V[0:4])
    assert np.allclose(ge[1:] / ge[:-1], 1 - 1.0 / 3.0, rtol=1e-5), ge
    ##the slot offsets per group: input d + 1, recurrent d (Brian 2's rounding: 1.5 ms -> 2 steps), and the Dale flag follows the PRE group
    k = {name: (kk, inh) for name, _, _, inh, kk in syn._projs}
    assert k["in_exc"] == (1, False) and k["exc_exc"] == (2, False) and k["exc_inh"] == (0, False)
    assert syn._R == 3
    ##an inhibitory PRE group feeds Gi by |w| (W stored negative)
    ctx = run.ctx; net = ctx.net
    run.reset_transient()
    ih_row, ih_id = net.neuronrows["inh"].start, net.neuronslices["inh"].start
    if fmt == "dense":
        ctx.syn.W[ex, ih_id] = -0.7                                     ##a new inh -> exc edge (dense: any entry of W is an edge)
        syn._projs.append(("inh_exc", slice(ex, ex + 2), slice(ih_id, ih_id + 1), True, 0))
        ctx.neur.Spikes[ih_id] = True
        syn.step(ctx)
        assert syn.Gi[ex] == pytest.approx(0.7 * (1 - 1.0 / 10.0)) and syn.Ge[ex] == 0.0 and ctx.upd.I2V[ex] == pytest.approx(0.7 * (0.0 - 13.5))


# --------------------------------------------------------------------- 3. the integrator: refractory hold, heterogeneous tau

@pytest.mark.parametrize("fmt", FORMATS)
def test_refractory_hold_brian2_semantics(fmt):
    """t_refrac = 3 ms at dt = 1 ms: V is held at v_reset for 2 steps after a spike and integrates again on the 3rd (Brian 2 >= 2.2)."""
    run = _micro(fmt, delays={"in_exc": 0.0}, w_in=50.0, elsm=False, t_refrac=3.0)
    frames = np.ones((12, 1), bool)
    rows = _trace(run, frames)
    net = run.ctx.net; ex = net.neuronrows["exc"].start; e0 = net.neuronslices["exc"].start - net.n_input
    V = np.array([r[1][ex] for r in rows]); sp = np.array([r[3][e0] for r in rows])
    s = int(np.flatnonzero(sp)[0])
    assert V[s] == 13.5                                         ##reset on the spike step
    assert V[s + 1] == 13.5 and V[s + 2] == 13.5                ##held (Since 1, 2): no integration, no spike
    assert not sp[s + 1] and not sp[s + 2] and sp[s + 3]        ##integrates again at Since 3 and fires at once
    assert np.flatnonzero(sp).tolist() == list(range(s, 12, 3)) ##period = refractory steps: s, s+3, s+6, ...
    ##refractory steps per row (inh, exc, exc); the inh row holds 1 step
    assert run.integrator._refrac_row.tolist() == [2, 3, 3]


def test_heterogeneous_tau_is_keyed_and_in_range():
    a = D.build(n_liquid=50, grid=(5, 5, 2), n_input=10, seed=4, weights_format="sparse", inputs_per_neurons=3)
    b = D.build(n_liquid=50, grid=(5, 5, 2), n_input=10, seed=4, weights_format="sparse", inputs_per_neurons=3)
    c = D.build(n_liquid=50, grid=(5, 5, 2), n_input=10, seed=5, weights_format="sparse", inputs_per_neurons=3)
    data = _data(n_in=10, n=6, T=10)
    runs = [p._prep_run(data, D.runcfg(), test_data=data) for p in (a, b, c)]
    for r in runs:
        r.set_mode("eval"); r.run_sample(data[0][0])
    ta, tb, tc = (r.integrator.tau_row for r in runs)
    assert np.array_equal(ta, tb) and not np.array_equal(ta, tc)
    assert ta.min() >= 5.0 and ta.max() <= 55.0 and ta.std() > 5.0
    sa = runs[0].threshold.tau_row
    assert sa.min() >= 50.0 and sa.max() <= 1050.0 and sa.std() > 50.0
    ##the LSM baseline: every tau 30, no adaptation
    lsm = D.build(n_liquid=50, grid=(5, 5, 2), n_input=10, seed=4, weights_format="sparse", ei_balance=False, sfa=False, tau_heterogeneous=False, inputs_per_neurons=3)
    r = lsm._prep_run(data, D.runcfg(), test_data=data); r.set_mode("eval"); r.run_sample(data[0][0])
    assert np.all(r.integrator.tau_row == 30.0) and np.all(r.threshold.V_th == 15.0) and np.isinf(r.threshold.tau_row).all()


def test_table4_configs_and_inhibitory_input_factor():
    """The Table 4 configs: inh input range U[0, factor * 0.4], separate switches, and 250 steps (4 ms bins) per 1 s SHD sample."""
    kw = dict(n_liquid=50, grid=(5, 5, 2), n_input=10, seed=4, weights_format="sparse", inputs_per_neurons=3)
    for name, (ei, sfa, het) in D.CONFIGS.items():
        pnet = D.build(ei_balance=ei, sfa=sfa, tau_heterogeneous=het, **kw)
        net = pnet.network
        in_inh = net.W0[net.syntype == net.type_key["in_inh"]]
        in_exc = net.W0[net.syntype == net.type_key["in_exc"]]
        assert in_exc.max() <= 0.4 and in_exc.max() > 0.3
        assert in_inh.max() <= 0.4 * (3.0 if ei else 0.5) and in_inh.max() > 0.4 * (3.0 if ei else 0.5) * 0.8, name
        assert pnet.network.network_name == f"deckers_2022_{name}_n50"
        assert (pnet.cfg.threshold.tau_th is not None) == sfa and (pnet.cfg.integrator.tau_range is not None) == het
    assert D.BIN_MS == 4.0 and int(round(D.T_MAX_MS / D.BIN_MS)) == 250 and D.DT_MS == 1.0
    with pytest.raises(ValueError):
        D.main(config="lsm")


# --------------------------------------------------------------------- 4. the SFA threshold

@pytest.mark.parametrize("fmt", FORMATS)
def test_sfa_bumps_exc_threshold_only_and_decays_back(fmt):
    """SFA: +1 mV per exc spike, Euler decay back to 15 with tau 550; inh rows keep 15 (decay first, then the increment)."""
    run = _micro(fmt, delays={"in_exc": 0.0}, w_in=50.0, elsm=True)
    frames = np.zeros((40, 1), bool); frames[0, 0] = True
    rows = _trace(run, frames)
    net = run.ctx.net; ex = net.neuronrows["exc"].start; ih = net.neuronrows["inh"].start; e0 = net.neuronslices["exc"].start - net.n_input
    sp = np.array([r[3][e0] for r in rows]); Vth = np.array([r[2] for r in rows])
    s = int(np.flatnonzero(sp)[0])
    assert np.all(Vth[:s, ex] == 15.0)
    assert Vth[s, ex] == pytest.approx(16.0, abs=1e-6)                      ##decay of a threshold AT v_th is 0, then + 1 mV
    assert Vth[s + 1, ex] == pytest.approx(16.0 - 1.0 / 550.0, rel=1e-6)    ##Euler decay towards 15
    assert np.all(Vth[:, ih] == 15.0)                                       ##inh row: never adapts
    assert np.all(np.diff(Vth[s:, ex]) <= 0) or sp[s + 1:].any()           ##monotone decay unless it spikes again
    ##a second spike must beat the raised threshold: V > V_th + decay, so with exc at 16 mV a V of 15.5 does not fire
    assert run.threshold.persistent == ()                                  ##nothing here is a persistent array (the liquid is fixed)


# --------------------------------------------------------------------- 5. dense == sparse, no learning, protocol

def test_dense_and_sparse_liquids_produce_identical_spikes():
    data = _data()
    counts = {}
    for fmt in FORMATS:
        pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=11, weights_format=fmt, inputs_per_neurons=8)
        run = pnet._prep_run(data, D.runcfg(), test_data=data)
        run.set_mode("eval")
        for i in range(4):
            run.reset_transient(); run.integrator.init_membrane(run.ctx, f"s{i}"); run.run_sample(data[i][0])
        counts[fmt] = run.history.Counts.copy()
        assert counts[fmt][run.ctx.net.n_input:].sum() > 100
    assert np.array_equal(counts["dense"], counts["sparse"])


@pytest.mark.parametrize("fmt", FORMATS)
def test_protocol_end_to_end_fixed_liquid_and_readout(fmt, tmp_path):
    """Train responses -> test responses -> logistic readout; W untouched (eval contract), results.json + responses.npz written, state arrays resumable."""
    train, test = _data(n=24), _data(n=9, seed=1)
    pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=11, weights_format=fmt, inputs_per_neurons=8)
    run = pnet.train(train, D.runcfg(readout_epochs=200, report_every=10), test_data=test, out_dir=tmp_path, save_run=False, save_history=False)
    r = run.report
    assert r["protocol"] == "ELSMProtocol" and r["n_train"] == 24 and r["n_test"] == 9
    assert r["test_acc"] >= 2 / 3 * 100 and r["train_acc"] > 90, (r["test_acc"], r["train_acc"])   ##separable toy classes
    assert np.array_equal(run.ctx.syn.W, run._W0)                       ##nothing learned
    assert run.ctx.sim.mode == "eval"
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["test_acc"] == r["test_acc"] and "paper_reference" in res
    z = np.load(tmp_path / "responses.npz")
    assert z["resp_train"].shape == (24, 48) and z["resp_test"].shape == (9, 48) and z["readout_W"].shape == (48, 3)
    assert (z["y_train"] >= 0).all() and set(z["classes"].tolist()) == {0, 1, 2}
    ##the inhibitory spike totals are a public (2,) array, reported per training sample
    proto = D.ELSMProtocol(D.ELSMProtocolCfg(), pnet._prep_run(train, D.runcfg(), test_data=test))
    assert "Inh_spikes" in proto.state_dict and proto.Inh_spikes.shape == (2,) and r["inh_spikes_per_sample_train"] >= 0.0


def test_protocol_resume_reports_what_the_uninterrupted_run_reports(tmp_path, monkeypatch):
    """A run killed after its first snapshot and resumed reports the same values and a bit-identical responses.npz as an uninterrupted run."""
    from checkpoint import CheckpointPolicy
    train, test = _data(n=24), _data(n=9, seed=1)
    kw = dict(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=11, weights_format="sparse", inputs_per_neurons=8)
    rc = D.runcfg(every=5, readout_epochs=50, report_every=10)
    full = D.build(**kw).train(train, rc, test_data=test, out_dir=tmp_path / "full", save_run=False, save_history=False)
    orig = CheckpointPolicy.on_sample_end
    def kill_after_first(policy, **k):
        path = orig(policy, **k)
        if path is not None:
            raise KeyboardInterrupt
        return path
    monkeypatch.setattr(CheckpointPolicy, "on_sample_end", kill_after_first)
    with pytest.raises(KeyboardInterrupt):
        D.build(**kw).train(train, rc, test_data=test, out_dir=tmp_path / "killed", save_run=False, save_history=False)
    monkeypatch.undo()
    st = json.loads((tmp_path / "killed" / "status.json").read_text())
    assert st["status"] == "interrupted" and st["last_snapshot"] == "snap_s00_i00000005.npz" and st["progress"] == {"stage": 0, "i": 5}
    back = D.build(**kw).train(train, rc, test_data=test, out_dir=tmp_path / "killed", resume_from=tmp_path / "killed" / st["last_snapshot"],
                               save_run=False, save_history=False)
    keys = ("inh_spikes_per_sample_train", "spikes_per_neuron_train", "exc_spikes_per_sample_train", "exc_spikes_per_sample_test", "silent_samples_train",
            "test_acc", "train_acc", "readout_final_loss", "n_train", "n_test")
    assert [back.report[k] for k in keys] == [full.report[k] for k in keys], [(k, back.report[k], full.report[k]) for k in keys]
    assert full.report["inh_spikes_per_sample_train"] > 0.0                                   ##the probe is not vacuous: the toy liquid's inh neurons fire
    za, zb = np.load(tmp_path / "full" / "responses.npz"), np.load(tmp_path / "killed" / "responses.npz")
    assert za.files == zb.files and all(np.array_equal(za[k], zb[k]) for k in za.files)


def test_protocol_requires_test_data_and_record_counts():
    from runcfg import HistoryCfg
    import dataclasses
    pnet = D.build(n_liquid=60, grid=(5, 4, 3), n_input=40, seed=11, weights_format="sparse", inputs_per_neurons=8)
    data = _data(n=12)
    with pytest.raises(ValueError, match="test set"):
        pnet.train(data, dataclasses.replace(D.runcfg(), data=dataclasses.replace(D.runcfg().data, make_test_split=True, test_fraction=0.0)), save_run=False)
    rc = D.runcfg(); rc = dataclasses.replace(rc, history=HistoryCfg(include_input=False, record_counts=False))
    with pytest.raises(RuntimeError, match="record_counts"):
        pnet.train(data, rc, test_data=data, save_run=False)


def test_no_plasticity_is_inert_in_both_formats():
    for fmt in FORMATS:
        run = _micro(fmt, delays={"in_exc": 0.0}, w_in=5.0)
        run.set_mode("train")                                   ##even in train: dW is zero / zeros returned, W does not move, no clamp planned
        W0 = run.ctx.syn.W.copy()
        run.run_sample(np.ones((10, 1), bool))
        assert np.array_equal(W0, run.ctx.syn.W)
        assert run._clamp_plan in ([], None)
        assert run.plasticity.w_min == -np.inf and run.plasticity.w_max == np.inf


# --------------------------------------------------------------------- 6. the readout, and the house rules

def test_logistic_readout_learns_a_separable_problem_and_standardises():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(m, 1.0, size=(60, 5)) * np.array([1, 10, 100, 1000, 1]) for m in (0.0, 3.0, -3.0)])
    y = np.repeat([0, 1, 2], 60)
    ro = LogisticReadout(l2=1e-3, lr=0.05, epochs=300).fit(X, y)
    assert ro.score(X, y) > 0.95 and ro.loss_[-1] < ro.loss_[0]
    assert ro.predict_proba(X[:3]).shape == (3, 3) and np.allclose(ro.predict_proba(X[:3]).sum(1), 1.0)
    assert np.allclose(ro.mean_, X.mean(0)) and ro.W_.shape == (5, 3)
    with pytest.raises(ValueError):
        LogisticReadout().fit(X, np.zeros(len(X)))              ##one class


def test_readout_report_is_the_shared_tail_of_the_readout_protocols(tmp_path):
    """readout_report turns fit_evaluate's dict into the shared report keys and responses.npz arrays; every readout protocol uses it."""
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(m, 1.0, size=(30, 4)) for m in (0.0, 4.0, -4.0)]); y = np.repeat([0, 1, 2], 30)
    Xt, yt = X[::3], y[::3]
    ro = fit_evaluate(X, y, Xt, yt, epochs=100)
    res, files = readout_report(ro, X, y, Xt, yt)
    assert res == {"test_acc": ro["test_acc"], "train_acc": ro["train_acc"], "n_train": 90, "n_test": 30, "classes": [0, 1, 2], "readout_final_loss": ro["final_loss"]}
    assert files.keys() == {"resp_train", "y_train", "resp_test", "y_test", "pred_test", "confusion_test", "classes", "readout_W", "readout_b", "readout_mean", "readout_std"}
    assert files["resp_train"] is X and files["y_test"] is yt and files["readout_W"] is ro["readout"].W_ and files["confusion_test"].shape == (3, 3)
    np.savez(tmp_path / "r.npz", **files)
    assert set(np.load(tmp_path / "r.npz").files) == files.keys() and json.dumps(res)      ##responses.npz / results.json take them as they are
    for f in ("deckers_2022", "hlsp_lsm_2026", "nalsm_2021", "krenzer_2025"):
        assert "readout_report(" in (_ROOT / "experiments" / f"{f}.py").read_text(), f


def test_new_files_use_no_where_keyword():
    """No `where=` keyword in the Deckers files or the builder's kernel code."""
    for f in ("experiment_dynamics/deckers2022.py", "experiments/deckers_2022.py", "experiments/readouts.py"):
        src = (_ROOT / f).read_text()
        assert not re.search(r"\bwhere\s*=", src), f
    builder_src = (_ROOT / "system" / "builder.py").read_text()
    added = builder_src[builder_src.index("DISTANCE_KERNELS ="):builder_src.index("def _generate_netmeta")]
    assert "where=" not in added


def test_grid_fields_are_canonical_like_every_other_field():
    """grid / grid_share are canonical even while None, no spatial-field exclusion is left, and the kernel tuple is DISTANCE_KERNELS."""
    import network as N
    import builder as B
    from network import NeuronSpec, _canonicalize_netspecs
    canon = _canonicalize_netspecs({"a": NeuronSpec("a", 3, True)}, {})
    assert canon["neuronspecs"]["a"] == {"neur_type": "a", "n": 3, "input": True, "inh": False, "grid": None, "grid_share": None}
    assert not hasattr(N, "_canon_neuronspec") and not hasattr(N, "_SPATIAL_FIELDS")
    assert B.DISTANCE_KERNELS == ("gaussian",) and not hasattr(B, "KERNELS")
    with pytest.raises(ValueError, match="Known kernels"):
        B._check_kernel("s", {"kernel": "boxcar", "C": 0.5, "lambda": 1.0})
