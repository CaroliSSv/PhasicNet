#test_reproducibility - the claims the whole framework's credibility rests on.
# These are NOT input-validation tests: every one covers something that would RUN FINE
# while being wrong. Run: pytest system/tests/test_reproducibility.py -q
from __future__ import annotations
import json
import logging
from dataclasses import asdict

import numpy as np
from modelinput import ModelInput   ##train() / _prep_run require a ModelInput
import pytest  ##only needed when run under pytest

from phasicnet import PhasicNet
from rnghandler import RNGHandler
from builder import Network
from cfg import (Cfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg,
                 PlasticityCfg, NoiseCfg, PhaseCfg)
from runcfg import RunCfg, DataCfg
from protocols import HoldoutValidation
from utils import _hash, _inf_safe
from original_dynamics import DeltaCurrentDense, LIFexact, AdaptiveThreshold, TraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase
import capture

SEED = 4242


def make_cfg(sigma_mem: float = 0.5) -> Cfg:
    return Cfg(
        simulation=SimulationCfg(dt=1.0),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=10.0),
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.005,
                                 A_ltd=0.006, eta_exc=1e-3, eta_inh=1e-3,
                                 w_min=-10.0, w_max=10.0),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=sigma_mem),
        phases=(PhaseCfg(phase_type="DEFAULT"),))


def spec_builder(pnet: PhasicNet) -> PhasicNet:
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", 6, True), ("exc", 10), ("inh", 4, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"),
                              ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc=0.6, exc_exc=0.3, exc_inh=0.4, inh_exc=0.4)
    pnet.builder.set_weights({"in_exc": 1.2, "exc_exc": 2.2, "exc_inh": 1.1, "inh_exc": 2.4})
    pnet.builder.spec_name = "repro_spec"
    return pnet


def build_model(seed: int = SEED, sigma_mem: float = 0.5) -> PhasicNet:
    ##returns the recipe; the run state is built inside pnet.train() / pnet._prep_run()
    pnet = spec_builder(PhasicNet(seed))
    pnet.builder.realize(network_name="repro_net", save=False)
    pnet.cfg = make_cfg(sigma_mem)
    pnet.set_runtime_classes(synapse_cls=DeltaCurrentDense, integrator_cls=LIFexact,
                             threshold_cls=AdaptiveThreshold, plasticity_cls=TraceSTDP,
                             noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    return pnet


##train() needs a dataset to split and a RunCfg; tiny random binary input, two classes.
RUNCFG = RunCfg(data=DataCfg(test_fraction=0.25))


def make_data(n_samples=8, T=30):
    X = (np.random.default_rng(0).random((n_samples * T, 6)) < 0.3)
    return ModelInput.from_tonic([(X[i * T:(i + 1) * T], i % 2) for i in range(n_samples)])


# --------------------------------------------------------------------------- 1. build

def test_realize_twice_is_idempotent():
    """THE call-order bug: realize() twice from one seed must give the SAME network (spawn() advances a counter)."""
    pnet = spec_builder(PhasicNet(SEED))
    first = pnet.builder.realize("a", save=False)
    second = pnet.builder.realize("b", save=False)

    assert np.array_equal(first.pre_idx, second.pre_idx)
    assert np.array_equal(first.post_idx, second.post_idx)
    assert np.array_equal(first.W0, second.W0)
    assert first.network_id == second.network_id


def test_same_seed_same_network_across_experiments():
    ##two PhasicNet instances with one seed must build one network
    a = spec_builder(PhasicNet(SEED)).builder.realize("a", save=False)
    b = spec_builder(PhasicNet(SEED)).builder.realize("a", save=False)
    assert a.network_id == b.network_id


def test_different_seed_different_network():
    """A different seed gives a different network (an unused seed would make everything 'reproducible' by accident)."""
    a = spec_builder(PhasicNet(SEED)).builder.realize("a", save=False)
    b = spec_builder(PhasicNet(SEED + 1)).builder.realize("a", save=False)
    assert a.network_id != b.network_id


def test_stream_derivation_is_order_independent():
    """Streams are keyed by NAME, so asking for them in a different order must not matter."""
    ##RNGHandler.rng(domain, key) is stateless: nothing to spawn in an order
    one = RNGHandler(SEED)
    two = RNGHandler(SEED)
    two.rng("phase", "SLEEP")          # extra spawn first
    two.rng("noise", "weights"); two.rng("noise", "synapse")

    assert one.rng("noise", "membrane").random() == two.rng("noise", "membrane").random()
    assert one.rng("noise", "membrane").random() == one.rng("noise", "membrane").random()      ##asking twice gives the same stream again: nothing is consumed
    assert one.rng("noise", "membrane").random() != one.rng("noise", "weights").random()       ##and different names are different streams
    ##the real consumer: the noise module takes its streams from the handler it is given, in whatever state that handler is in
    cfg = make_cfg().noise
    a, b = TestNoise(cfg, SimulationCfg(), RNGHandler(SEED)), TestNoise(cfg, SimulationCfg(), two)
    assert a.noise_rng["membrane"].random() == b.noise_rng["membrane"].random()


# --------------------------------------------------------------------------- 2. cfg hashing

def test_cfg_hash_survives_json_roundtrip():
    """cfg_id feeds run_id, so the cfg hash must survive a JSON round-trip (or identical runs get two ids)."""
    ##the plain cfg and its _inf_safe form (+-inf is not JSON) must both round-trip
    canon = asdict(make_cfg())
    assert _hash(canon, digest_size=8) == _hash(json.loads(json.dumps(canon)), digest_size=8)
    assert _hash(_inf_safe(canon), digest_size=8) == _hash(json.loads(json.dumps(_inf_safe(canon))), digest_size=8)


def test_cfg_hash_changes_with_content():
    a = asdict(make_cfg())
    b = asdict(make_cfg())
    b["integrator"]["tau_mem"] = 21.0
    assert _hash(a, digest_size=8) != _hash(b, digest_size=8)


# --------------------------------------------------------------------------- 3. network save/load

def test_network_save_load_roundtrip(tmp_path, caplog):
    """rehash_network() is the drift check. It has to survive an actual disk round-trip."""
    for post_offset in (False, True):      ##both layouts: a compressed net is decompressed before hashing, which is the part that can drift
        name = f"roundtrip_net_{post_offset}"
        net = spec_builder(PhasicNet(SEED)).builder.realize(
            name, post_offset=post_offset, save=True, filename=str(tmp_path / name))

        pnet = PhasicNet(SEED)
        pnet.load_net(str(tmp_path / f"{name}.json"))

        assert pnet.network.network_id == net.network_id
        assert np.array_equal(pnet.network.W0, net.W0)
        assert np.array_equal(pnet.network.pre_idx, net.pre_idx)
        assert np.array_equal(pnet.network.post_idx, net.post_idx)
        assert pnet.network.post_offset == net.post_offset == post_offset

        ##rehash_network() recomputes the id from the loaded arrays and WARNS on a mismatch (it returns nothing)
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            pnet.network.rehash_network()
        assert not [r for r in caplog.records if "inconsistent" in r.getMessage()], "the network drifted on its way through the disk"

        pnet.network.W0[0] += 1.0                  ##power check: a drifted array IS reported
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            pnet.network.rehash_network()
        assert [r for r in caplog.records if "Network_ID inconsistent" in r.getMessage()]


def test_rehash_network_logs_cleanly_when_consistent():
    ##guards the rehash_network log format (a missing %s) on a consistent network
    net = spec_builder(PhasicNet(SEED)).builder.realize("consistent_net", save=False)
    net.rehash_network()


# --------------------------------------------------------------------------- 4. run

def test_same_seed_same_run_result():
    """Identical seed + identical cfg -> identical dynamics (noise streams and plasticity, not just topology)."""
    data = make_data()      ##more than two samples, so the train split and the shuffle are non-trivial

    results = []
    for _ in range(2):
        pnet = build_model()
        run = pnet.train(data, RUNCFG, save_run=False)
        results.append((run.history.spike_i().copy(),
                        run.history.spike_ptr().copy(),
                        run.ctx.syn.W.copy(),
                        run.runmeta.run_id))

    assert results[0][0].size > 0, "no spikes at all: the comparison below proves nothing"
    assert np.array_equal(results[0][0], results[1][0]), "spike neuron indices diverged"
    assert np.array_equal(results[0][1], results[1][1]), "spike times diverged"
    assert np.array_equal(results[0][2], results[1][2]), "final weights diverged"
    assert results[0][3] == results[1][3], "the same run was filed under two ids"      ##run_id hashes derived cfg + runcfg + rng spec


def test_different_seed_different_run_result():
    data = make_data()

    weights, run_ids = [], []
    for seed in (SEED, SEED + 1):
        pnet = build_model(seed)
        run = pnet.train(data, RUNCFG, save_run=False)
        weights.append(run.ctx.syn.W.copy())
        run_ids.append(run.runmeta.run_id)

    assert not np.array_equal(weights[0], weights[1])
    assert run_ids[0] != run_ids[1]      ##the seed is part of run_id (rng_spec)


def test_fit_streams_are_reproducible_and_distinct():
    """Prerequisite for k-fold and Optuna: fit k must be reproducible IN ISOLATION."""
    ##the key is ("fold", (k, e)) -- fold k, epoch e -- so a resumed epoch re-derives its own shuffle
    assert RNGHandler(SEED).rng("fold", (3, 0)).random() == RNGHandler(SEED).rng("fold", (3, 0)).random()
    assert RNGHandler(SEED).rng("fold", (3, 0)).random() != RNGHandler(SEED).rng("fold", (4, 0)).random()
    assert RNGHandler(SEED).rng("fold", (3, 0)).random() != RNGHandler(SEED).rng("fold", (3, 1)).random()

    ##the spec (seed + derivation rule) alone rebuilds the handler, so a fit's stream can be reproduced from it
    spec = RNGHandler(SEED).rng_spec
    assert spec["seed"] == RNGHandler(SEED).seed == SEED
    assert RNGHandler.from_rng_spec(spec).rng("fold", (3, 0)).random() == RNGHandler(SEED).rng("fold", (3, 0)).random()


def _fit_fold_1(sigma_mem: float, after_fold_0: bool) -> np.ndarray:
    """Fold 1's weights on a fresh Run, with or without fold 0 fitted first (the halves of a val_split=0.5 holdout stand in for two folds)."""
    run = build_model(sigma_mem=sigma_mem)._prep_run(make_data(), RUNCFG)
    (half_a, half_b), = run.datahandler.fold_split(run.train_set, val_split=0.5, rng=run.ctx.rng.rng("data", f"folds:{run.runmeta.dataset_id}"))
    proto = HoldoutValidation(None, run)
    if after_fold_0:
        proto.ops.run_fit(half_a, None, epochs=1, fold=0)
    proto.ops.run_fit(half_b, None, epochs=1, fold=1)
    return run.ctx.syn.W.copy()


def test_fold_is_reproducible_in_isolation():
    """Without membrane noise, fold 1 comes out the same whether or not fold 0 ran first (the shuffle is keyed per (fold, epoch))."""
    alone, after = _fit_fold_1(0.0, after_fold_0=False), _fit_fold_1(0.0, after_fold_0=True)
    assert not np.array_equal(alone, build_model()._prep_run(make_data(), RUNCFG).ctx.syn.W), "fold 1 did not learn, so this proves nothing"
    assert np.array_equal(alone, after)


# --------------------------------------------------------------------------- 5. positions stay with the Network

def test_network_positions_survive_save_load(tmp_path):
    """Grid positions survive save/load (pos_<group> in the npz); a grid-less network has {} and no pos_ key."""
    p = PhasicNet(SEED); p._instantiate_builder(); b = p.builder
    b.set_neurons(("input", 4, True), ("exc", 6), ("inh", 2, False, True)); b.set_positions(exc=(2, 2, 2), inh=(2, 2, 2), share="liquid")
    b.set_synapses(("in_exc", "input", "exc"), ("exc_inh", "exc", "inh"))
    b.set_connectivity(in_exc=0.5, exc_inh={"kernel": "gaussian", "C": 0.5, "lambda": 2.0}); b.set_weights(in_exc=1.0, exc_inh=1.0)
    net = b.realize("pos_net", save=True, filename=str(tmp_path / "pos_net"))
    assert set(net.positions) == {"exc", "inh"} and net.positions["exc"].shape == (6, 3) and net.positions["inh"].shape == (2, 3)
    with np.load(tmp_path / "pos_net.npz") as z:
        assert {"pos_exc", "pos_inh"} <= set(z.files)
    assert "positions" not in json.loads((tmp_path / "pos_net.json").read_text())      ##not in the json: the npz carries them

    q = PhasicNet(SEED); q.load_net(str(tmp_path / "pos_net.json"))
    assert q.network.network_id == net.network_id
    assert set(q.network.positions) == {"exc", "inh"}
    for g in net.positions:
        assert np.array_equal(q.network.positions[g], net.positions[g])

    flat = spec_builder(PhasicNet(SEED)).builder.realize("flat_net", save=True, filename=str(tmp_path / "flat_net"))     ##no grid
    assert flat.positions == {}
    with np.load(tmp_path / "flat_net.npz") as z:
        assert not [k for k in z.files if k.startswith("pos_")]
    r = PhasicNet(SEED); r.load_net(str(tmp_path / "flat_net.json"))
    assert r.network.positions == {}
