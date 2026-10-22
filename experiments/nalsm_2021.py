#main - Ivanov & Michmizos (2021), "Increasing liquid state machine performance with edge-of-chaos dynamics organized by astrocyte-modulated
# plasticity" (NeurIPS 2021), assembled in PhasicNet: network, stimulus, protocol and ONE call that runs it. The model itself (equations, constants,
# step order) is documented in nalsm2021.py.
#
#THE ARCHITECTURE (Sec. 2.1, generate_spatial_network.py):
#   input (784 | 2312 N-MNIST) --15% of all pairs--> liquid, weights +-w (random sign per edge), PLASTIC within [-3, 3]
#   liquid: 1000 LIF on a 3-D grid (GRID, INFERRED 10x10x10), 80/20 exc/inh, P = C exp(-(D/lambda)^2), lambda 3, C_EE .2, C_EI .1, C_IE .3, C_II .05
#           (the code's C list by block position: [post exc, pre exc], [post inh, pre exc], [post exc, pre inh], [post inh, pre inh]); no self-connections
#   weights: NALSM / LSM+STDP start MAXIMAL (3 exc, -3 inh, +-3 input; Sec. 2.2.1); the LSM baseline uses ONE value w everywhere (Fig. 2A: 0.4-1.2)
# post_offset=True: W is (N, N_in + N). Default weights_format "sparse" with update_on "hybrid_event" (per-neuron traces, only synapses with a
# spiking end are updated; fastest). "event" = per-synapse traces decayed lazily; the astrocyte stays clocked either way. "dense" = clocked.
#
#STIMULUS (user-side): the paper's input neurons (LIF, theta 20, tau_v 64, t_rfr 2) hit by a current of 100 with probability p per 1 ms step,
# p = pixel / image total / max of that over the training set (NALSM_SIM_SUPPORT.assemble_data). N-MNIST: p = 1 on an event, 0 otherwise.
# The liquid never sees a pixel.
#
#PROTOCOL (Sec. 2.2): (1) initialization: n_init random training images, 20 ms each, STDP + astrocyte on, NO decay -> W_ini. (2) counting: every
# sample 250 ms from W_ini (restored per sample), STDP + astrocyte and the 0.99 decay on -> liquid spike counts. (3) readout: normalised counts,
# linear layer, soft targets, L2 5e-10, the epoch with the best validation accuracy (val = the last n_val training samples).
# Paper: 96.15% MNIST (N = 1000), 96.13% N-MNIST, 83.54% F-MNIST; 97.61 / 97.51 / 85.84 at N = 8000.
#
#INFERRED: GRID; the per-sample reset of the synapse counters; the readout optimiser (Adam lr 1e-3; theirs Adadelta lr 0.1) and its
# stopping; input connections drawn with p = 0.15 per pair (theirs: exactly 15%); STDP traces reset per sample (theirs carry over).
#
#RUN (from the repo root):  python -c "import experiments, nalsm_2021; nalsm_2021.main(dataset='mnist', n_train=1000, n_test=500, tag='t1')"
# In parallel: parallel.Job(load=load_dataset, load_kw={"name": ...}). Runtime classes: the packs NALSM_SPARSE / NALSM_DENSE.
from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
##experiments/__init__.py puts the code folders on sys.path (run from the repo root)
import numpy as np
from phasicnet import PhasicNet, RuntimeClasses
from cfg import Cfg, SimulationCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from protocols import Protocol
from nalsm2021 import (NALSMSynapseCfg, NALSMIntegratorCfg, NALSMThresholdCfg, NALSMPlasticityCfg, NALSMExpSynapse, NALSMExpSynapseSparse, NALSMLIF,
                       NALSMThreshold, NALSMAstroSTDP, NALSMAstroSTDPSparse)
from deckers2022 import NoPlasticityCfg, NoPlasticity
from original_dynamics import TestNoise
from phases import DefaultPhase
from paths import RAW_DIR, NETWORKS_DIR, RUNS_DIR
from modelinput import ModelInput
from preprocessing import load_mnist
from readouts import fit_evaluate, readout_report
import capture

logger = logging.getLogger("nalsm_2021")

#---------------------------------------------------------------- generate_spatial_network.py, NALSM_RUN_MAIN_SIM_MNIST.py, Sec. 2
DT_MS = 1.0
SEED = 1
N_LIQUID = 1000
GRID = (10, 10, 10)             #INFERRED: "3-dimensional", 1000 neurons; the script asks for the three dimensions
F_E = 0.8
TAU_MEM, V_TH, V_RESET, T_REFRAC, TAU_SYN, BIAS = 64.0, 20.0, 0.0, 2.0, 1.0, 0.0
LAMBDA = 3.0
C_EE, C_EI, C_IE, C_II = 0.2, 0.1, 0.3, 0.05   #by block position in the code's C_EE_EI_IE_II_l = [0.2, 0.1, 0.3, 0.05] (post, pre): EE, (inh <- exc), (exc <- inh), II
P_IN = 0.15
W_MAX = 3.0
LSM_WEIGHT = {"mnist": 1.0, "nmnist": 1.0, "fmnist": 1.0}   #the LSM baseline's single weight (Fig. 2A: tuned 0.4-1.2 MNIST / 0.8-1.35 N-MNIST)
INPUT_CURRENT = 100.0           #the current one input event injects into an input neuron (NALSM_RUN_MAIN_SIM_MNIST.py: INPUT_CURRENT_IN)
A_PLUS, TAU_TRACE, TRACE_INC = 0.15, 10.0, 0.1
ASTRO_TAU, ASTRO_W, DECAY = 100.0, {1000: 0.015, 8000: 0.0075}, 0.99
T_INIT_MS, T_SAMPLE_MS, N_INIT = 20.0, 250.0, 50000
NMNIST_SENSOR = (34, 34, 2)
READOUT = dict(lr=1e-3, batch_size=250, epochs=5000, l2=5e-10, patience=50)    #Eq. (9): lambda 5e-10, m 250; lr / epochs / patience INFERRED (see header)


   ######################################################################################
##########################   Stimulus: the input neurons   ################################
   ######################################################################################

def input_neuron_spikes(p: np.ndarray, rng: np.random.Generator, *, current: float = INPUT_CURRENT, tau_v: float = TAU_MEM, v_th: float = V_TH,
                        t_refrac: int = 2) -> np.ndarray:
    """The paper's input neurons, step for step as coded (NALSM_LIF_v1: register_spikes -> reset -> update v -> refractory counter):
        p (T, n) = probability of an input event per step per neuron (a pixel's p every step, or a 0/1 event frame)
        v <- v + current * [u < p] - v / tau_v ;  spike iff v >= v_th and the refractory counter <= 0 ;  reset 0 ;  counter <- t_refrac on a spike
    Returns the (T, n) bool spike trains the liquid receives. Vectorised over neurons; the loop is over time only."""
    T, n = p.shape
    v = np.zeros(n, dtype=np.float64); rfr = np.zeros(n, dtype=np.float64)
    out = np.zeros((T, n), dtype=bool)
    for t in range(T):
        s = (v >= v_th) & (rfr <= 0.0)                        ##from v of the previous step (the code's order)
        v *= ~s                                               ##reset
        v += current * (rng.random(n) < p[t]) - v / tau_v     ##Eq. (1), dt 1 ms
        rfr = (rfr - 1.0) * (~s) + t_refrac * s
        out[t] = s
    return out


def normalise_images(images: np.ndarray, scale: float | None = None):
    """NALSM_SIM_SUPPORT.assemble_data: pixel / the image's total intensity, then / the maximum of that over the set (scale; None = this set's).
    Returns (p in [0, 1] per pixel, scale)."""
    x = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    s = x.sum(axis=1); s[s == 0] = 1.0
    x = x / s[:, None]
    scale = float(x.max()) if scale is None else float(scale)
    return x / scale, scale


class NALSMInputs:
    """images (n, H, W) | (n, D) -> sample i = input spikes (T, D) bool, DRAWN WHEN ASKED from a stream keyed (seed, i): the same index always gives
    the same spikes (resume-safe), nothing stored. The probabilities follow normalise_images (scale = the training set's).
    A ModelInput source: frames only, the labels go to ModelInput(NALSMInputs(...), labels)."""

    def __init__(self, images, *, T: int, scale: float | None = None, seed: int = 0):
        self.p, self.scale = normalise_images(images, scale)
        self.T, self.seed = int(T), int(seed)

    def __len__(self) -> int:
        return len(self.p)

    def __getitem__(self, i) -> np.ndarray:
        i = int(i)
        p = np.broadcast_to(self.p[i], (self.T, self.p.shape[1]))
        return input_neuron_spikes(p, np.random.default_rng([self.seed, i]))

    @property
    def n_input(self) -> int:
        return int(self.p.shape[1])


class NALSMEventInputs:
    """A FRAMED tonic dataset (N-MNIST with the transform chain in load_dataset) -> sample i = (input spikes (T, 2312) bool, label): a binned
    event becomes p = 1 for that step and neuron (NALSM_SIM_SUPPORT.get_nmnist_data_point), then the same input neuron. Lazy, keyed.
    Only the model's own input encoder; wrapped by ModelInput.from_tonic."""

    def __init__(self, dataset, *, seed: int = 0):
        self.ds, self.seed = dataset, int(seed)

    def __len__(self) -> int:
        return len(self.ds)

    @property
    def targets(self):   ##tonic's own label list: ModelInput.labels() needs no encoding of every sample
        return None if getattr(self.ds, "target_transform", None) is not None else getattr(self.ds, "targets", None)

    def __getitem__(self, i) -> tuple:
        frames, label = self.ds[int(i)]
        p = np.asarray(frames, dtype=np.float64).reshape(len(frames), -1)          ##(T, P, H, W) -> (T, 2312), row-major as ModelInput.to_TI
        return input_neuron_spikes(p, np.random.default_rng([self.seed, int(i)])), int(label)


def load_dataset(name: str, *, seed: int = SEED):   ##returns the WHOLE set; DataCfg.n_train / n_test pick from it
    """-> (train, test, n_input). mnist / fmnist: RAW_DIR/<name>.npz (preprocessing.load_mnist keys); nmnist: tonic (RAW_DIR). T = 250 steps."""
    T = int(round(T_SAMPLE_MS / DT_MS))
    if name in ("mnist", "fmnist"):
        x, y, xt, yt = load_mnist(RAW_DIR / f"{name}.npz")
        train = ModelInput(NALSMInputs(x, T=T, seed=seed), y, axis_labels="TX")
        test = ModelInput(NALSMInputs(xt, T=T, scale=None, seed=seed + 1), yt, axis_labels="TX")   ##the code rescales the test set by ITS OWN maximum (assemble_data)
        return train, test, train.n_input
    if name == "nmnist":
        import tonic
        from tonic_helpers import ClampBinary, FixedLength, US_TO_MS
        frame = tonic.transforms.Compose([tonic.transforms.ToFrame(sensor_size=NMNIST_SENSOR, time_window=int(DT_MS * US_TO_MS)), ClampBinary(dtype=np.bool_), FixedLength(T)])
        tr, te = (tonic.datasets.NMNIST(save_to=str(RAW_DIR), train=flag, transform=frame) for flag in (True, False))
        train = ModelInput.from_tonic(NALSMEventInputs(tr, seed=seed))
        test = ModelInput.from_tonic(NALSMEventInputs(te, seed=seed + 1))
        return train, test, train.n_input
    raise ValueError(f"Unknown dataset {name!r}: mnist | nmnist | fmnist")


   ######################################################################################
##########################   Assemble   ###################################################
   ######################################################################################

def build(*, model: str = "nalsm", n_liquid: int = N_LIQUID, grid=GRID, n_input: int = 784, seed: int = SEED, dt_ms: float = DT_MS, weights_format: str = "sparse",
          astro_w: float | None = None, lsm_weight: float = 1.0, w_max: float = W_MAX, cutoff: float | None = None, save_net: bool = False) -> PhasicNet:
    """The NALSM network + physics, ready for pnet.train(train, runcfg(...), test_data=test).
        model = "nalsm" (astrocyte-modulated STDP) | "lsm_stdp" (STDP, A- = A+ fixed) | "lsm" (fixed weights = lsm_weight, no plasticity)
    astro_w: None = 0.015 for n_liquid <= 1000, 0.0075 above (README). weights_format "sparse" (default, update_on "hybrid_event") or "dense": same model.
    cutoff: sparse only, skip presynaptic neurons silent for more than this many steps (None = exact)."""
    if model not in ("nalsm", "lsm_stdp", "lsm"):
        raise ValueError(f"model must be nalsm | lsm_stdp | lsm, got {model!r}")
    if int(np.prod(grid)) != n_liquid:
        raise ValueError(f"grid {grid} holds {int(np.prod(grid))} cells, n_liquid is {n_liquid}: the code places the liquid on an exact grid.")
    n_exc = int(round(n_liquid * F_E)); n_inh = n_liquid - n_exc
    kern = lambda C: {"kernel": "gaussian", "C": C, "lambda": LAMBDA}
    w0 = lsm_weight if model == "lsm" else w_max
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", n_input, True), ("exc", n_exc), ("inh", n_inh, False, True))
    pnet.builder.set_positions(exc=tuple(grid), inh=tuple(grid), share="liquid")
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("in_inh", "input", "inh"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"),
                              ("inh_exc", "inh", "exc"), ("inh_inh", "inh", "inh"))
    pnet.builder.set_connectivity(in_exc=P_IN, in_inh=P_IN, exc_exc=kern(C_EE), exc_inh=kern(C_EI), inh_exc=kern(C_IE), inh_inh=kern(C_II))
    pnet.builder.set_weights({"in_exc": {"values": [-w0, w0]}, "in_inh": {"values": [-w0, w0]},        ##+-w, random sign per edge (constant_randomize_+/-)
                              "exc_exc": w0, "exc_inh": w0, "inh_exc": -w0, "inh_inh": -w0})            ##inh pre groups negative (Dale)
    pnet.builder.spec_name = "nalsm_2021_spec"
    net = pnet.builder.realize(network_name=f"nalsm_2021_n{n_liquid}", post_offset=True, save=False)
    if save_net:
        net.save_net(network_name=net.network_name, spec_name="nalsm_2021_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)
    logger.info("liquid: %d exc + %d inh on %s, %d inputs; synapses %s (recurrent per neuron %.2f)", n_exc, n_inh, grid, n_input, net.synapse_counts,
                sum(v for k, v in net.synapse_counts.items() if not k.startswith("n_in_")) / n_liquid)

    sparse = weights_format == "sparse"
    aw = (ASTRO_W[1000] if n_liquid <= 1000 else ASTRO_W[8000]) if astro_w is None else float(astro_w)
    if model == "lsm":
        plasticity_cfg, plasticity_cls = NoPlasticityCfg(), NoPlasticity
    else:
        plasticity_cfg = NALSMPlasticityCfg(tau_pretrace=TAU_TRACE, tau_posttrace=TAU_TRACE, A_ltp=A_PLUS, A_ltd=A_PLUS, eta_exc=1.0, eta_inh=1.0, w_min=-w_max,
                                            w_max=w_max, freeze=(), trace_inc=TRACE_INC, astro_tau=ASTRO_TAU, astro_w=aw, astro_bias_offset=0.0, decay=DECAY,
                                            astro=model == "nalsm", input_bounds=(-w_max, w_max))
        plasticity_cls = NALSMAstroSTDPSparse if sparse else NALSMAstroSTDP
    pnet.cfg = Cfg(
        simulation=SimulationCfg(dt=dt_ms, weights_format=weights_format, update_on="hybrid_event" if sparse else "timestep"),
        synapse=NALSMSynapseCfg(b=None, tau_syn=TAU_SYN, cutoff=cutoff),
        integrator=NALSMIntegratorCfg(tau_mem=TAU_MEM, v_rest=V_RESET, r=1.0, b=BIAS, t_refrac=T_REFRAC),
        threshold=NALSMThresholdCfg(v_reset=V_RESET, v_th=V_TH, tau_th=None, adpt_th=None, t_refrac=T_REFRAC),
        plasticity=plasticity_cfg,
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(replace(NALSM_SPARSE if sparse else NALSM_DENSE, plasticity_cls=plasticity_cls))     ##the pack (below), plasticity per model
    pnet.cfg_name, pnet.run_name = f"nalsm_2021_{model}_cfg", f"nalsm_2021_{model}_run"
    return pnet


   ######################################################################################
##########################   Protocol (Sec. 2.2)   ########################################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class NALSMProtocolCfg(ProtocolCfg):
    """Sec. 2.2's settings. Hashed into run_id with the rest of RunCfg except report_every.
    Args:
        n_init = snapshots in the initialization pass (50000; 0 = none: the LSM baseline), drawn without replacement from a keyed permutation of
            the training set (repeated when n_init > n_train); t_init_ms = 20 ms each; init_window "start" | "random" (N-MNIST: a random window)
        t_sample_ms = 250: the counting presentation
        plastic_counting = STDP + astrocyte stay on during counting, W restored to W_ini before every sample (NALSM, LSM+STDP); False = fixed W
        n_val = validation block = the LAST n_val of train (None = n_train / 6). n_train / n_test are DataCfg's (runcfg() below)
        readout_* = readouts.LogisticReadout with NALSM_OUTPUT_LAYER's loss (soft +-1 targets, L2 5e-10); readout_patience = epochs
            without a better validation accuracy before stopping (the selected epoch is the best one)
    """
    protocol_type: str = "NALSM"
    n_init: int = N_INIT
    t_init_ms: float = T_INIT_MS
    init_window: str = "start"
    t_sample_ms: float = T_SAMPLE_MS
    plastic_counting: bool = True
    n_val: int | None = None
    readout_lr: float = READOUT["lr"]
    readout_batch: int = READOUT["batch_size"]
    readout_epochs: int = READOUT["epochs"]
    readout_l2: float = READOUT["l2"]
    readout_patience: int | None = READOUT["patience"]
    report_every: int = 500


class NALSMProtocol(Protocol):
    """
    Stage 0 (initialization, mode "train", decay off): n_init snapshots -> W_ini (proto__W_ini; saved as the TrainedModel). Stage 1 / 2 (counting
    train / test, decay on, W restored per sample): Resp_train / Resp_test (int16) = liquid spike counts, Y_*, Init_rate / Astro = [sum, count] of
    the two 200-sample probes. Then the readout.
    progress = {"stage": 0 | 1 | 2, "i": samples done}: resumable with CheckpointCfg.every + out_dir (a stage-0 resume continues the W pass).
    """
    protocol_type = "NALSM"
    needs_test_set = True     ##test_data=test to PhasicNet.train; Protocol.__init__ checks it

    def __init__(self, cfg: NALSMProtocolCfg | None, run, *, out_dir=None):
        super().__init__(NALSMProtocolCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c, net = self.cfg, run.ctx.net
        if c.init_window not in ("start", "random"):
            raise ValueError(f"init_window must be 'start' or 'random', got {c.init_window!r}")
        dt = float(run.history.dt)
        self._n_init_steps = int(round(c.t_init_ms / dt)); self._n_steps = int(round(c.t_sample_ms / dt))
        self._n_train, self._n_test = len(run.train_set), len(run.test_set)
        self._n_val = max(1, self._n_train // 6) if c.n_val is None else min(int(c.n_val), self._n_train - 1)
        self._n_liq = net.n_total - net.n_input
        self._plastic = hasattr(run.plasticity, "decay_active")           ##NoPlasticity (the LSM baseline) has no passes to switch
        self.W_ini = np.zeros_like(run.ctx.syn.W)
        self.Resp_train = np.zeros((self._n_train, self._n_liq), dtype=np.int16)
        self.Resp_test = np.zeros((self._n_test, self._n_liq), dtype=np.int16)
        self.Y_train = np.full(self._n_train, -1, dtype=np.int64)
        self.Y_test = np.full(self._n_test, -1, dtype=np.int64)
        self.Init_rate, self.Astro = np.zeros(2), np.zeros(2)   ##[sum, count] of the two 200-sample probes

    # -------------------------------------------------------------------------
    def _init_order(self) -> np.ndarray:
        """n_init indices: keyed permutations of the training set, concatenated (sample_random_inputs: without replacement per pass)."""
        n, rng = self._n_train, self.rng.rng("protocol", "init_order")
        reps = int(np.ceil(self.cfg.n_init / n)) if n else 0
        return np.concatenate([rng.permutation(n) for _ in range(reps)])[:self.cfg.n_init] if reps else np.zeros(0, dtype=int)

    def _snapshot(self, frames: np.ndarray, j: int) -> np.ndarray:
        n = self._n_init_steps
        if self.cfg.init_window == "random" and frames.shape[0] > n:
            s = int(self.rng.rng("protocol", f"init_window:{j}").integers(0, frames.shape[0] - n + 1))
            return frames[s:s + n]
        return frames[:n]

    def _initialize(self) -> bool:
        """Stage 0 = ops.for_samples over the init order: snapshot j -> reset, its t_init_ms window, STDP + astrocyte on, no decay.
        W_ini = W at the end. Returns False when a resumed run is already past it (W_ini came back from the snapshot)."""
        run, c, ops = self.run, self.cfg, self.ops
        snapshots = run.train_set.subset(self._init_order(), "nalsm_init")     ##the n_init samples in their keyed order (repeats included)
        run.set_mode("train")
        if self._plastic:
            run.plasticity.decay_active = False

        def step(j, frames, target):                                        ##after reset=True below: V, counters, traces, astrocyte back to their initial values; W kept
            counts = self.present(self._snapshot(frames, j), target)
            if j < 200:
                self.Init_rate += (capture.rate_hz(counts[run.ctx.net.n_input:], self._n_liq, self._n_init_steps, run.history.dt), 1)

        def tail(lo, hi):      ##capture: realized edges only, dense and sparse alike
            return (f"|W| mean {capture.mean_globalweight(run.ctx, signed=False):.3f}  at a bound "
                    f"{100 * capture.pinned_weights(run.ctx, w_min=run.plasticity.w_min, w_max=run.plasticity.w_max)['saturated_frac']:.1f}%  "
                    f"A- {float(run.plasticity.A[0]) if self._plastic else float('nan'):.4f}")
        ran = ops.for_samples(snapshots, step, stage=0, name="init", reset=True, extra=tail)
        if ran:
            np.copyto(self.W_ini, run.ctx.syn.W)
        return ran

    def count_sample(self, frames: np.ndarray, target) -> np.ndarray:
        """ONE counting presentation: W <- W_ini, state reset, t_sample_ms with plasticity on (plastic_counting) -> liquid spike counts (N_liquid,)."""
        run, n0 = self.run, self.run.ctx.net.n_input
        np.copyto(run.ctx.syn.W, self.W_ini)
        run.reset_transient()
        return self.present(frames[:self._n_steps], target)[n0:]

    def _count_sample(self, frames: np.ndarray, target, key: str) -> np.ndarray:
        """count_sample as ops.responses calls it (+ the astrocyte probe over the first 200 samples)."""
        counts = self.count_sample(frames, target)
        if self._plastic and self.Astro[1] < 200:
            self.Astro += (float(self.run.plasticity.A[0]), 1)     ##[sum, count]
        return counts

    @staticmethod
    def normalise_counts(R: np.ndarray, scale: float | None = None):
        """NALSM_OUTPUT_LAYER: each sample's counts / their sum, then / the maximum over the (training) set. Returns (features, scale)."""
        X = np.asarray(R, dtype=np.float64)
        s = X.sum(axis=1); s[s == 0] = 1.0
        X = X / s[:, None]
        scale = float(X.max()) if scale is None else float(scale)
        return X / max(scale, 1e-12), scale

    def execute(self) -> dict:
        run, c, ops = self.run, self.cfg, self.ops
        ops.header()
        ##(1) initialization pass
        if self._initialize():
            ops.save_model("initialized liquid (W_ini)")
        ##W_ini's |W| statistics through capture: W_ini goes into the live W first (count_sample does that before every sample anyway)
        np.copyto(run.ctx.syn.W, self.W_ini)
        w_ini_abs_mean = capture.mean_globalweight(run.ctx, signed=False)
        w_ini_at_bound = capture.pinned_weights(run.ctx, w_min=run.plasticity.w_min, w_max=run.plasticity.w_max)["saturated_frac"] if self._plastic else None
        ##(2) counting, from W_ini every sample
        if self._plastic:
            run.plasticity.decay_active = True
        run.set_mode("train" if (c.plastic_counting and self._plastic) else "eval")
        ops.responses(run.train_set, self._count_sample, self.Resp_train, self.Y_train, stage=1, name="count-train", n=self._n_train)
        ops.responses(run.test_set, self._count_sample, self.Resp_test, self.Y_test, stage=2, name="count-test", n=self._n_test)
        ##(3) readout: the last n_val training samples validate
        Xtr, scale = self.normalise_counts(self.Resp_train)
        Xte, _ = self.normalise_counts(self.Resp_test, scale)
        ro = fit_evaluate(Xtr, self.Y_train, Xte, self.Y_test, lr=c.readout_lr, epochs=c.readout_epochs, batch_size=c.readout_batch, l2=c.readout_l2,
                          standardize=False, soft_target_logit=1.0, val_fraction=self._n_val / self._n_train, val_from_end=True, select_by_val_acc=True,
                          early_stop_patience=c.readout_patience)
        res, files = readout_report(ro, self.Resp_train, self.Y_train, self.Resp_test, self.Y_test)     ##the RAW counts go to responses.npz (the readout saw them normalised); own keys below
        res.update({"val_acc_best": max(ro["readout"].val_acc_) if ro["readout"].val_acc_ else None,
                    "readout_epochs_run": len(ro["readout"].loss_), "n_val": int(self._n_val),
                    "init_rate_hz_first200": float(self.Init_rate[0] / self.Init_rate[1]) if self.Init_rate[1] else None,
                    "astro_A_end_of_sample_first200": float(self.Astro[0] / self.Astro[1]) if self.Astro[1] else None,
                    "liquid_spikes_per_sample_test": float(self.Resp_test.sum(1).mean()),
                    "liquid_rate_hz_test": capture.rate_hz(self.Resp_test.sum(1).mean(), self._n_liq, self._n_steps, run.history.dt),
                    "w_ini_abs_mean": w_ini_abs_mean, "w_ini_at_bound_frac": w_ini_at_bound,     ##taken above, before the counting pass
                    "paper_reference": "Sec. 3.4 / Table 1: NALSM1000 96.15% MNIST, 96.13% N-MNIST, 83.54% F-MNIST; NALSM8000 97.61 / 97.51 / 85.84; LSM 95.44, LSM+STDP 90.52 (MNIST)"})
        files.update(count_scale=np.asarray(scale), W_ini=self.W_ini)
        return ops.finish(res, files=files, lines=(
            f"  test accuracy : {res['test_acc']:.2f}%  (train {res['train_acc']:.2f}%, best val {res['val_acc_best']})   paper: {res['paper_reference']}",
            f"  liquid        : {res['liquid_rate_hz_test']:.1f} Hz in the test presentations; W_ini |W| mean {res['w_ini_abs_mean']:.3f}; initialization rate "
            f"(first 200 snapshots) {None if res['init_rate_hz_first200'] is None else round(res['init_rate_hz_first200'], 1)} Hz"))


##the runtime classes as ONE pack each (phasicnet.RuntimeClasses); build() replaces plasticity_cls per model (nalsm / lsm_stdp / lsm)
NALSM_DENSE = RuntimeClasses(synapse_cls=NALSMExpSynapse, integrator_cls=NALSMLIF, threshold_cls=NALSMThreshold, plasticity_cls=NALSMAstroSTDP,
                             noise_cls=TestNoise, phase_classes=(DefaultPhase,), protocol_classes=(NALSMProtocol,))
NALSM_SPARSE = replace(NALSM_DENSE, synapse_cls=NALSMExpSynapseSparse, plasticity_cls=NALSMAstroSTDPSparse)


def runcfg(*, every: int | None = None, n_train: int | None = None, n_test: int | None = None, **proto) -> RunCfg:
    """n_train / n_test -> DataCfg, which PICKS the samples (stratified, seeded, shuffled order); n_val stays in the protocol cfg."""
    return RunCfg(data=DataCfg(make_test_split=False, test_fraction=0.0, n_train=n_train, n_test=n_test, shuffle=True),
                  checkpoint=CheckpointCfg(to_disk=False, every=every), scorer=ScorerCfg(rate_hz_min=0.0),
                  history=HistoryCfg(include_input=False, record_spikes=False, record_counts=True), protocol=NALSMProtocolCfg(**proto))


   ######################################################################################
##########################   Run   ########################################################
   ######################################################################################

def main(*, dataset: str = "mnist", model: str = "nalsm", n_liquid: int = N_LIQUID, grid=GRID, astro_w: float | None = None, lsm_weight: float | None = None,
         n_init: int = N_INIT, n_train: int | None = None, n_val: int | None = None, n_test: int | None = None, seed: int = SEED, static_counting: bool = False,
         cutoff: float | None = None, dense: bool = False, every: int | None = None, resume: str | None = None, tag: str = "", out: str | None = None):
    """ONE run; every setting is a keyword argument (defaults = the constants).
        dataset: "mnist" | "nmnist" | "fmnist"; model: "nalsm" | "lsm_stdp" | "lsm"; astro_w None: 0.015 for N <= 1000, 0.0075 above; lsm_weight None: the
        baseline's 1.0; static_counting: fixed W_ini during counting; cutoff (sparse): skip presynaptic neurons silent longer than this many ms;
        dense: dense weights (default sparse); every: snapshot every N samples; resume: a snap_*.npz; out: the run folder (default runs/nalsm_2021_<model>_<dataset>_n<N>[_tag]).
    Returns the Run: run.report = NALSMProtocol's results (also <out>/results.json; log in <out>/run.jsonl, numbers in <out>/results.jsonl, status in <out>/status.json)."""
    train, test, n_input = load_dataset(dataset, seed=seed)
    pnet = build(model=model, n_liquid=n_liquid, grid=tuple(grid), n_input=n_input, seed=seed, weights_format="dense" if dense else "sparse",
                 astro_w=astro_w, lsm_weight=LSM_WEIGHT[dataset] if lsm_weight is None else lsm_weight, cutoff=cutoff)
    out = Path(out) if out else RUNS_DIR / f"nalsm_2021_{model}_{dataset}_n{n_liquid}{('_' + tag) if tag else ''}"
    return pnet.train(train, runcfg(every=every, n_init=0 if model == "lsm" else n_init, init_window="random" if dataset == "nmnist" else "start",
                                    plastic_counting=not static_counting, n_train=n_train, n_val=n_val, n_test=n_test),
                      test_data=test, out_dir=out, resume_from=resume, save_run=False, save_history=False, dataset_name=dataset)


if __name__ == "__main__":
    main()
