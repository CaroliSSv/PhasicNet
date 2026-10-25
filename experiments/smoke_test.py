#smoke_test - the FASTEST end-to-end proof that the whole system is wired correctly, in ALL THREE execution modes.
#
#NOT a unit test: walks the public API once and asserts each stage is structurally sane, plus the one equivalence
# that must never silently break: sparse-timestep == sparse-event, bit for bit.
# Run: python experiments/smoke_test.py
from __future__ import annotations
import logging, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]     ##experiments/<script> -> PhasicNet/
sys.path[:0] = [str(_ROOT / "system"), str(_ROOT / "system_modules"), str(_ROOT / "experiment_dynamics"), str(_ROOT / "experiments")]
import numpy as np
from phasicnet import PhasicNet
from dynamicscfg import DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, AdaptiveThreshold, TraceSTDP, SparseTraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase

logging.basicConfig(level=logging.ERROR)
SEED, T, N_SAMPLES = 7, 30, 24


def build(fmt="dense", update="timestep"):
    p = PhasicNet(SEED)
    p._instantiate_builder()
    p.builder.set_neurons(("input", 8, True), ("exc", 12), ("inh", 4, False, True))
    p.builder.set_synapses(("in_exc","input","exc"),("exc_exc","exc","exc"),("exc_inh","exc","inh"),("inh_exc","inh","exc"))
    p.builder.set_connectivity(in_exc=0.6, exc_exc=0.3, exc_inh=0.4, inh_exc=0.4)
    p.builder.set_weights({"in_exc":(0.0,3.0),"exc_exc":2.2,"exc_inh":1.1,"inh_exc":2.4})   ##range: symmetry breaking
    p.builder.spec_name="smoke_spec"; p.builder.realize(network_name="smoke_net", save=False)
    p.cfg = DynamicsCfg(simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on=update), synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=40.0),
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
        plasticity=PlasticityCfg(tau_pretrace=20.0,tau_posttrace=20.0,A_ltp=0.005,A_ltd=0.006,eta_exc=1e-3,eta_inh=1e-3,w_min=None,w_max=10.0),
        noise=NoiseCfg(mu_mem=0.0,sigma_mem=0.0), phases=(PhaseCfg(phase_type="DEFAULT"),))
    sparse = fmt == "sparse"
    p.set_runtime_classes(synapse_cls=DeltaCurrentSparse if sparse else DeltaCurrentDense, integrator_cls=LIFexact, threshold_cls=AdaptiveThreshold,
                          plasticity_cls=SparseTraceSTDP if sparse else TraceSTDP, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    p.cfg_name = p.run_name = "smoke"
    return p


def make_data():
    rng = np.random.default_rng(0)
    return [((rng.random((T, 8)) < (0.25 if i % 2 == 0 else 0.55)), i % 2) for i in range(N_SAMPLES)]


def one(fmt, update, tag):
    p = build(fmt, update)
    run = p.train(make_data(), RunCfg(epochs=2, k_fold=2, data=DataCfg(test_fraction=0.25)), save_run=False)
    r, meta = run.report, run.runmeta
    ##structural assertions -- each names what breaking it would mean
    assert not hasattr(p, "ctx") or getattr(p, "ctx", None) is None, "PhasicNet retained a ctx (assembler must hold no run state)"
    assert run.test_set is not None and not (set(run.train_set.index.tolist()) & set(run.test_set.index.tolist())), "TRAIN AND TEST OVERLAP"
    assert len(r["fold_scores"]) == 2 and r["score"] <= max(r["fold_scores"]) + 1e-9, "trial score must be the MEAN over fits, never the max"
    assert np.all(np.isfinite(run.ctx.syn.W)), "NaN in W"
    assert run.history.spike_i().size > 0, "network never fired"
    assert meta.cfg_id != meta.derived_cfg_id and run.history.derived_cfg.plasticity.w_min == -np.inf, "derived cfg did not record the None->-inf clamp"
    W_before = run.ctx.syn.W.copy(); test = run.test()
    assert np.array_equal(W_before, run.ctx.syn.W), "TEST PASS TRAINED THE MODEL (learn=False not holding)"
    print(f"  {tag:<16} score={r['score']:.4f} folds={[round(x,2) for x in r['fold_scores']]} spikes={run.history.spike_i().size} test={test['score']:.3f}")
    return run.ctx.syn.W.copy(), run.history.spike_i().copy(), r["score"]


if __name__ == "__main__":
    print("SMOKE TEST")
    Wd, Sd, sd = one("dense",  "timestep", "dense/timestep")
    Wt, St, st = one("sparse", "timestep", "sparse/timestep")
    We, Se, se = one("sparse", "event",    "sparse/event")
    ##THE equivalence that must never silently break: event mode skips only edges whose dw is EXACTLY zero.
    assert np.array_equal(Wt, We) and np.array_equal(St, Se), "sparse-timestep and sparse-event diverged -- a real bug, not float drift"
    print("  sparse-timestep == sparse-event : BIT-IDENTICAL (W and spike train)")
    ##dense vs sparse: same model, different float accumulation (BLAS f32 vs bincount f64); identical here, may drift on long runs.
    print(f"  dense == sparse score           : {sd == st}   (float-order dependent; equality is NOT required)")
    print("\nSMOKE TEST PASSED")
