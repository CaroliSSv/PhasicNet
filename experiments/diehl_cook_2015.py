#main - Diehl & Cook (2015) assembled in PhasicNet: the network (builder), the physics (diehlcook.py cfgs + modules), the stimulus
# encoding, and ONE call that trains it. The model itself (equations, constants, deviations) is documented in diehlcook.py.
#
#THE ARCHITECTURE:
#   input --all_to_all--> exc      plastic, w0 ~ U(0.003, 0.303) (the script's (rand + 0.01) * 0.3), normalised to sum 78 per exc
#                                  neuron before every example (by the protocol, diehlcook.dc_normalise), delays U[0, 10) ms per synapse
#   exc   --one_to_one--> inh      fixed 10.4
#   inh   --all_but_self--> exc    fixed 17 (stored -17: Dale), the lateral-inhibition / winner-take-all circuit
# post_offset=True: input neurons receive nothing, so W is (n_exc + n_inh, N) rather than (N, N).
#
#STIMULUS (user-side preprocessing, by design): the script shows each MNIST digit for 350 ms as Poisson spikes at pixel/4 Hz
# (input_intensity = 2 -> rate = pixel / 8 * 2), then 150 ms of silence, at dt = 0.5 ms. DCPoissonImages is that stimulus as a dataset that can
# RE-PRESENT (variant(i, intensity=, rng=) -> poisson_frames: (T, 784) bool incl. the rest period); mnist_input() wraps it as the ModelInput every
# D&C run takes. For N-MNIST, bin the events at dt (preprocessing.tonic_prep) and append the rest frames with dc_append_rest().
#
#SIZES: the paper reports 100 / 400 / 1600 / 6400 exc neurons; the dense runtime is fine up to a few hundred on CPU. With
# n_exc=100, N = 784 + 200 and T = 1000 steps per digit.
from __future__ import annotations
import logging
import numpy as np
from phasicnet import PhasicNet
from dynamicscfg import DynamicsCfg, SimulationCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from diehlcook import (DCSynapseCfg, DCIntegratorCfg, DCThresholdCfg, DCPlasticityCfg,
                       DCConductanceSynapse, DCLIFeuler, DCAdaptiveThreshold, DCTripletSTDP)
from original_dynamics import TestNoise
from phases import DefaultPhase
from paths import RAW_DIR, NETWORKS_DIR
from preprocessing import load_mnist
from modelinput import ModelInput

logger = logging.getLogger("main")

#---------------------------------------------------------------- the script's protocol constants
DT_MS = 0.5                     #ms   b2.defaultclock.dt
T_PRESENT_MS = 350.0            #ms   single_example_time
T_REST_MS = 150.0               #ms   resting_time
INPUT_INTENSITY = 2.0           #     rate = pixel / 8 * input_intensity  -> 63.75 Hz for pixel 255
SEED = 7
V_INIT_BELOW_REST = 40.0        #mV   the script creates both groups at v = v_rest - 40 mV


   ######################################################################################
##########################   Stimulus   ###################################################
   ######################################################################################

def dc_append_rest(frames: np.ndarray, *, dt_ms: float = DT_MS, rest_ms: float = T_REST_MS) -> np.ndarray:
    """(T, N_in) sample -> (T + rest/dt, N_in): the script's silent period after every example, as frames of zeros."""
    rest = np.zeros((int(round(rest_ms / dt_ms)), frames.shape[1]), dtype=bool)
    return np.concatenate([np.asarray(frames, dtype=bool), rest], axis=0)


def poisson_frames(image: np.ndarray, intensity: float, rng: np.random.Generator, n_present: int, n_rest: int) -> np.ndarray:
    """(n_present + n_rest, 784) bool: Poisson at pixel/8*intensity Hz for n_present steps, then silence. The PoissonGroup's own
    expression in float64: rand(N) < rate * dt, rate = x / 8. * input_intensity, dt in seconds."""
    rate = image.reshape(-1).astype(np.float64) / 8. * intensity
    p = rate * (DT_MS * 0.001)
    frames = np.zeros((n_present + n_rest, p.size), dtype=bool)
    frames[:n_present] = rng.random((n_present, p.size)) < p
    return frames


class DCPoissonImages:
    """MNIST images as a dataset that can RE-PRESENT. (len, [i] -> (image, label)) is what Run reads (the dataset id, the labels);
    variant(i, intensity=, rng=) -> poisson_frames(...) is what DCProtocol.present_example asks for ("example i, again, at this intensity",
    with the keyed rng IT chose, so a resumed run gets the same spikes).
    dt_ms/present_ms/rest_ms set the frame counts only; the Poisson probability uses DT_MS."""
    def __init__(self, x, y, *, dt_ms: float = DT_MS, present_ms: float = T_PRESENT_MS, rest_ms: float = T_REST_MS):
        self.x, self.y = x, y
        self.n_present, self.n_rest = int(round(present_ms / dt_ms)), int(round(rest_ms / dt_ms))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[int(i)], int(self.y[int(i)])
    def variant(self, i, *, intensity: float, rng: np.random.Generator) -> np.ndarray:
        return poisson_frames(self.x[int(i)], intensity, rng, self.n_present, self.n_rest)


def mnist_input(x, y, *, seed: int = SEED, **kw) -> ModelInput:
    """MNIST images -> ModelInput over DCPoissonImages. DCPoissonImages[i] is the IMAGE (not spikes), so the source here is a subclass whose [i]
    is the spike frames at INPUT_INTENSITY from a stream keyed (seed, i) -- what ModelInput validates and hashes. DCProtocol's presentations
    still come from variant(i, intensity=, rng=) (its own keyed draws), through ModelInput.variant. kw = present_ms / rest_ms / dt_ms of DCPoissonImages."""
    class _Frames(DCPoissonImages):
        def __getitem__(self, i):
            return self.variant(i, intensity=INPUT_INTENSITY, rng=np.random.default_rng([seed, int(i)]))
    return ModelInput(_Frames(x, y, **kw), np.asarray(y), axis_labels="TX")


   ######################################################################################
##########################   Assemble   ###################################################
   ######################################################################################

def build(n_input: int, n_exc: int, *, seed: int = SEED, dt_ms: float = DT_MS, norm_target: float = 78.0, save_net: bool = False) -> PhasicNet:
    """The D&C network + physics, ready for pnet.train(data, runcfg). n_inh == n_exc (one-to-one).
    norm_target = 78 is the script's value for 784 inputs (mean input weight 0.1); scale it with n_input for other input sizes."""
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", n_input, True), ("exc", n_exc), ("inh", n_exc, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc="all_to_all", exc_inh="one_to_one", inh_exc="all_but_self")
    pnet.builder.set_weights({"in_exc": (0.003, 0.303), "exc_inh": 10.4, "inh_exc": -17.0})     ##-17: Dale's sign given explicitly (no warning); in_exc = the script's (rand() + 0.01) * 0.3
    pnet.builder.spec_name = "diehl_cook_2015_spec"
    net = pnet.builder.realize(network_name=f"diehl_cook_2015_e{n_exc}", post_offset=True, save=False)
    if save_net:
        net.save_net(network_name=net.network_name, spec_name="diehl_cook_2015_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)

    ##values several modules must agree on, set ONCE here
    delays = dict(delay_syn_group="in_exc", delay_min=0.0, delay_max=10.0)      ##the script: delay['ee_input'] = (0 ms, 10 ms); exc->inh has none
    t_refrac, t_refrac_inh, v_reset_inh = 5.0, 2.0, -45.0
    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=dt_ms, weights_format="dense", update_on="timestep"),
        synapse=DCSynapseCfg(b=None, tau_ge=1.0, tau_gi=2.0, e_exc=0.0, e_inh=-100.0, e_inh_inh=-85.0, **delays, v_reset_inh=v_reset_inh, t_refrac_inh=t_refrac_inh),
        integrator=DCIntegratorCfg(tau_mem=100.0, v_rest=-65.0, r=1.0, tau_mem_inh=10.0, v_rest_inh=-60.0, t_refrac=t_refrac, t_refrac_inh=t_refrac_inh,
                                   v_reset_inh=v_reset_inh),
        threshold=DCThresholdCfg(v_reset=-65.0, v_th=-52.0, tau_th=1e7, adpt_th=0.05, v_reset_inh=v_reset_inh, v_th_inh=-40.0, theta_init=20.0, offset=20.0,
                                 t_refrac=t_refrac, t_refrac_inh=t_refrac_inh),       ##the exc spike block lives in the threshold
        plasticity=DCPlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.01, A_ltd=1e-4, eta_exc=1.0, eta_inh=None,
                                   w_min=None, w_max=None, freeze=("inh",),     ##"inh" = any projection touching an inh neuron: exc_inh + inh_exc fixed, only input -> exc learns. Clamp off (fixed 10.4 / -17 must not be clipped)
                                   tau_posttrace2=40.0, w_max_ee=1.0, norm_target=norm_target, norm_syn_group="in_exc", **delays),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),      ##D&C has no noise; sigma_mem MUST be 0 (refractory hold)
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=DCConductanceSynapse, integrator_cls=DCLIFeuler, threshold_cls=DCAdaptiveThreshold,
                             plasticity_cls=DCTripletSTDP, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.cfg_name, pnet.run_name = "diehl_cook_2015_cfg", "diehl_cook_2015_run"
    return pnet


def fresh_network(run, *, redraw_delays: bool = False) -> None:
    """Put a built Run into the state of a NEWLY CREATED Brian network -- the script creates one for training and one for every
    test-mode run: V = v_rest - 40 mV per group, no spikes, ge = gi = 0, empty delay ring, no spike history, empty STDP traces. W and theta
    are KEPT (the test script loads them). redraw_delays: the new network draws its delays again, over the nonzero weights it starts with
    (diehlcook.dc_input_delays: the same stream as training -- the script's np.random.seed(0)); without it the delays are put back after
    run.reset_transient()."""
    ctx = run.ctx
    syn, pl = run.synapse, run.plasticity
    if not redraw_delays:
        kept = [(m, k, getattr(m, k).copy()) for m in (syn, pl) for k in ("K", "KLTP", "LTD", "Delay_drawn") if isinstance(getattr(m, k, None), np.ndarray)]
    run.reset_transient()
    if redraw_delays:
        syn.redraw_delays(); pl.redraw_delays()
    else:
        for m, k, v in kept:
            np.copyto(getattr(m, k), v)
    ctx.neur.V[...] = (run.integrator._v_rest_row - V_INIT_BELOW_REST)[ctx.net.rec_slice] if ctx.net.n_post > ctx.net.n_rec else run.integrator._v_rest_row - V_INIT_BELOW_REST
    if ctx.upd.dW is not None:
        pl._touched = (np.zeros(0, np.int64), np.zeros(0, np.int64))


def runcfg(*, epochs: int = 1, n_train: int | None = None, n_test: int | None = None) -> RunCfg:
    """ONE pass over the training set is the script's default (num_examples = 60000 * 3 is three passes).
    n_train / n_test = how many digits of the split's train / test set a run uses (DataCfg, stratified); None = all."""
    return RunCfg(protocol=ProtocolCfg(epochs=epochs), data=DataCfg(test_fraction=0.20, stratify=True, n_train=n_train, n_test=n_test),
                  checkpoint=CheckpointCfg(to_disk=False), scorer=ScorerCfg(rate_hz_min=0.0), history=HistoryCfg(include_input=False))


   ######################################################################################
##########################   Run   ########################################################
   ######################################################################################

def main(n_train: int = 160, n_test: int = 40, n_exc: int = 100, epochs: int = 1):
    """The data is the WHOLE MNIST training set as a ModelInput (mnist_input: lazy, keyed Poisson draws); the split, the pick of
    n_train / n_test digits (stratified) and the order are DataHandler's."""
    mnist = RAW_DIR / "mnist.npz"
    ##MNIST or fail (load_mnist raises if missing)
    x, y, _, _ = load_mnist(mnist)
    data = mnist_input(x, y, seed=SEED)
    n_input = data.n_input
    logger.info("data: %d samples, %d inputs, T=%d steps each (dt=%.1f ms)", len(data), n_input, data[0][0].shape[0], DT_MS)
    pnet = build(n_input, n_exc, norm_target=78.0 * n_input / 784)   ##78 for the real 784 inputs
    ##NOTE: pnet.train() does NOT run the per-example normalisation or the re-presentation rule (dc_reproduce.py does); this demo only exercises the assembly.
    run = pnet.train(data, runcfg(epochs=epochs, n_train=n_train, n_test=n_test), save_run=False, save_history=False)
    r, m = run.report, run.runmeta
    logger.info("=" * 72)
    logger.info(f"TRIAL  score: {r['score']}   folds {r['fold_scores']}   stopped: {[f['stopped'] for f in r['folds']]}")
    logger.info(f"       run_id={m.run_id}  cfg_id={m.cfg_id}  derived_cfg_id={m.derived_cfg_id}  network_id={m.network_id[:12]}..")
    syn, net = run.ctx.syn, run.ctx.net
    blk = syn.W[net.synapseslices["in_exc"].post, net.synapseslices["in_exc"].pre]
    logger.info(f"in_exc  W in [{blk.min():.3f}, {blk.max():.3f}]  per-exc sums {blk.sum(axis=1).min():.1f}..{blk.sum(axis=1).max():.1f} (78 before each example)")
    logger.info(f"theta   exc {run.threshold.Theta[net.rec_slice][net.n_exc - net.n_input:].min():.3f}..{run.threshold.Theta.max():.3f} mV")
    run.threshold.freeze(True)      ##the paper's test state: mode "eval" (test) raises if the persistent Theta still adapts
    t = run.test()
    logger.info(f"TEST   n={t['n_test']}  score={t['score']}  spikes={t['spikes']}")
    return run


if __name__ == "__main__":
    main()
