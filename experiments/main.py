#main - end-to-end demonstration of the PhasicNet API on REAL MNIST digits, Poisson-encoded (Brian2-STDP style).
#
#THE API, in the order it is meant to be used:
#   PhasicNet(seed)                      one RNGHandler, keyed streams
#   builder -> realize -> save_net       the network, saved ONCE to networks/
#   Cfg (physics) + dynamics classes     SimulationCfg carries dt AND the execution mode
#   RunCfg                               the protocol + history/scorer/checkpoint/data policies
#   ModelInput(source, y)                the data
#   pnet.train(data, runcfg) -> Run      split -> pick -> holdout fit -> epochs
#   run.test()                           held-out set, forward-only, once
#   run.save_trained_model()             best checkpoint -> runs/TrainedModel_<run_id>
#   run.history.save()                   History_<run_id>, carrying the DERIVED cfg
#
#Switch weights_format / update_on below and nothing else changes: same API, same report shape.
from __future__ import annotations
import logging
import numpy as np
from phasicnet import PhasicNet
from cfg import Cfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, AdaptiveThreshold, TraceSTDP, SparseTraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase
from paths import RAW_DIR, NETWORKS_DIR, RUNS_DIR
from preprocessing import load_mnist, poisson_prep
from modelinput import ModelInput

logger = logging.getLogger("main")

SEED, N_TRAIN, N_TEST, T_PRESENT, PEAK_HZ, DOWNSAMPLE = 7, 160, 40, 60, 120.0, 4   ##DataCfg picks N_TRAIN / N_TEST from the split
N_EXC, N_INH = 40, 10
WEIGHTS, UPDATE = "sparse", "hybrid_event"    ##<- the one switch. "dense"/"timestep" | "sparse"/"timestep" | "sparse"/"hybrid_event" | "sparse"/"event"


class PoissonDigits:
    """The demo's stimulus as a LAZY source: sample i = Poisson frames (T_PRESENT, 49) of the block-meaned 7x7 digit i, drawn from a
    stream keyed (seed, i), so a digit is the same spikes whenever it is asked for."""
    def __init__(self, x, *, seed: int = SEED):
        k = DOWNSAMPLE
        self.x, self.seed = x.reshape(len(x), 28 // k, k, 28 // k, k).mean(axis=(2, 4)), seed      ##block-mean 28x28 -> 7x7
        self.peak = float(self.x.max())        ##ONE intensity scale for every digit
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        d = self.x[int(i)]                     ##rate = d / peak * PEAK_HZ (poisson_prep scales by the batch max)
        return poisson_prep(d[None], T=T_PRESENT, peak_hz=PEAK_HZ * float(d.max()) / self.peak, rng=np.random.default_rng([self.seed, int(i)]), flatten=True)[0]


def make_data(seed=SEED) -> ModelInput:
    """The WHOLE MNIST training set as a ModelInput over PoissonDigits; the split and the pick are the run's (RunCfg.data)."""
    x, y, _, _ = load_mnist(RAW_DIR / "mnist.npz")
    return ModelInput(PoissonDigits(x, seed=seed), y, axis_labels="TX")


def build(n_input):
    pnet = PhasicNet(SEED)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", n_input, True), ("exc", N_EXC), ("inh", N_INH, False, True))
    pnet.builder.set_synapses(("in_exc","input","exc"), ("exc_exc","exc","exc"), ("exc_inh","exc","inh"), ("inh_exc","inh","exc"))
    pnet.builder.set_connectivity(in_exc=0.60, exc_exc=0.10, exc_inh=0.30, inh_exc=0.30)
    pnet.builder.set_weights({"in_exc": (0.0, 4.0), "exc_exc": 1.0, "exc_inh": 3.0, "inh_exc": 3.0})   ##range = symmetry breaking (Brian2: rand()*gmax)
    pnet.builder.spec_name = "mnist_spec"
    net = pnet.builder.realize(network_name="mnist_net", save=False)
    net.save_net(network_name="mnist_net", spec_name="mnist_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)

    sparse = WEIGHTS == "sparse"
    pnet.cfg = Cfg(
        simulation=SimulationCfg(dt=1.0, weights_format=WEIGHTS, update_on=UPDATE),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=60.0),            ##r sized so a typical digit drives V_inf past threshold
        threshold=ThresholdCfg(v_reset=-65.0, v_th=-54.0, tau_th=100.0, adpt_th=0.4),
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.010, A_ltd=0.0105,   ##Brian2 STDP example values
                                 eta_exc=1.0, eta_inh=1.0, w_min=-6.0, w_max=6.0, freeze=("inh",)),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.2),
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=DeltaCurrentSparse if sparse else DeltaCurrentDense, integrator_cls=LIFexact, threshold_cls=AdaptiveThreshold,
                             plasticity_cls=SparseTraceSTDP if sparse else TraceSTDP, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.cfg_name, pnet.run_name = "mnist_cfg", "mnist_run"
    return pnet


def main():
    data = make_data()
    n_input = data.n_input
    logger.info("data: %d MNIST digits, %d input neurons, %d ms each (the run picks %d train / %d test)", len(data), n_input, T_PRESENT, N_TRAIN, N_TEST)
    pnet = build(n_input)
    runcfg = RunCfg(protocol=ProtocolCfg(epochs=3, val_split=0.2),
                   
                    data=DataCfg(test_fraction=0.20, stratify=True, n_train=N_TRAIN, n_test=N_TEST),   ##the pick is DataHandler's
                    checkpoint=CheckpointCfg(to_disk=True), scorer=ScorerCfg(), history=HistoryCfg(include_input=False))
    run = pnet.train(data, runcfg, save_run=True, save_history=True)
    r, m = run.report, run.runmeta
    print("\n" + "=" * 72)
    print(f"TRIAL  score (the holdout fit's val score): {r['score']:.4f}   epochs {[round(e['score'], 3) for e in r['folds'][0]['epochs']]}")   ##one fold, no mean/std over folds
    print(f"       stopped: {[f['stopped'] for f in r['folds']]}   mode: {WEIGHTS}/{UPDATE}")
    print(f"       run_id={m.run_id}  cfg_id={m.cfg_id}  derived_cfg_id={m.derived_cfg_id}  network_id={m.network_id[:12]}..  dataset_id={m.dataset_id}")
    t = run.test()
    print(f"TEST   n={t['n_test']}  score={t['score']:.4f}  spikes={t['spikes']}")
    print("\nARTEFACTS"); [print(f"  {d.name}/{f.name}") for d in (NETWORKS_DIR, RUNS_DIR) for f in sorted(d.iterdir())]
    return run

if __name__ == "__main__":
    main()
