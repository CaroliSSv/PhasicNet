#main - Luo et al. (2026), "HLSP-LSM: Enhancing image recognition performance of liquid state machines via brain-inspired hybrid long short-term
# plasticity", assembled in PhasicNet: network, stimulus, protocol (Algorithm 1) and ONE call that runs it. The model itself (equations,
# constants, timing) is documented in hlsp2026.py.
#
#THE ARCHITECTURE (Sec. 3.1, Table 2):
#   input (784 pixels | 2 x 34 x 34 N-MNIST channels) --p = 0.15--> exc, inh    FIXED weights, each edge +W_in or -W_in at random (builder w0 {"values": [-w, w]})
#   liquid: N neurons, f_E = 0.8 exc / 0.2 inh, every neuron -> any OTHER neuron with p = 0.10 (Sec. 3.1 "excluding self-connections"; the authors'
#           LSM_BaseClass.py zeroes the diagonal)
#   exc/inh --p = 0.10--> exc/inh   liquid weights START AT ZERO (Algorithm 1 line 3) and are learned by E-STDP (exc pre) / I-STDP (inh pre),
#                                   clipped to [0, W_max = 3] in magnitude (inh stored negative: [-3, 0])
# post_offset=True: W is (N, N_in + N). Default weights_format "sparse" with update_on "hybrid_event" (per-neuron traces, only synapses with a
# spiking end are updated; fastest). "event" = per-synapse traces decayed lazily (synapticindex.py); "dense" = clocked.
#
#STIMULUS (user-side): pixels -> Poisson spikes, rate proportional to the normalised intensity (Sec. 4.1). PEAK_HZ is not in the paper (see the
# constant). The protocol logs the initial liquid rate (paper: ~7 Hz MNIST / F-MNIST, ~30 Hz N-MNIST, Fig. 2) to calibrate PEAK_HZ x W_in.
# N-MNIST: tonic events binned at dt. Samples are encoded lazily (60k x 250 x 784 frames would not fit in memory).
#
#PROTOCOL (Algorithm 1, Sec. 4.1): the validation block (1/6) is split off the training set BEFORE the STDP pass, as the authors' random_split.
# Phase 2 (mode "train"): every fit sample once, 50 ms, state reset before each, hybrid plasticity on. Phase 3 (mode "eval"): W frozen (STD still on),
# every sample 250 ms from reset -> liquid spike counts in P temporal partitions (P x N). Phase 4: softmax readout (readouts.py, as the authors'
# LinearModel), the validation states drive the lr schedule. Paper: 95.63% MNIST (N = 1000, P = 1); N-MNIST 98.05% (N = 1000, P = 4);
# F-MNIST 84.26% (Tables 4-5).
#
#INFERRED: PEAK_HZ; the STD reading (hlsp2026 std_pre_depletion); liquid spikes of the presentation's first step (reset -> none);
# the sample order (a seeded shuffle, like the authors' DataLoader(shuffle=True)); F-MNIST file format (same npz keys as MNIST).
#
#RUN (from the repo root):  python -c "import experiments, hlsp_lsm_2026; hlsp_lsm_2026.main(dataset='mnist', n_train=1000, n_test=500, tag='t1')"
# In parallel: parallel.Job(load=load_dataset, load_kw={"name": ...}). Runtime classes: the packs HLSP_SPARSE / HLSP_DENSE.
from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
##experiments/__init__.py puts the code folders on sys.path (run from the repo root)
import numpy as np
from phasicnet import PhasicNet, RuntimeClasses
from dynamicscfg import DynamicsCfg, SimulationCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from protocols import Protocol
from hlsp2026 import (HLSPSynapseCfg, HLSPIntegratorCfg, HLSPThresholdCfg, HLSPPlasticityCfg, HLSPCurrentSynapse, HLSPCurrentSynapseSparse,
                      HLSPLIF, HLSPThreshold, HLSPPlasticity, HLSPPlasticitySparse)
from original_dynamics import TestNoise
from phases import DefaultPhase
from paths import RAW_DIR, NETWORKS_DIR, RUNS_DIR
from modelinput import ModelInput
from preprocessing import load_mnist, poisson_prep
from readouts import fit_evaluate, readout_report
from checkpoint import snapshots
import capture

logger = logging.getLogger("hlsp_lsm_2026")

#---------------------------------------------------------------- Table 2 (neurons), Table 3 (plasticity), Sec. 3.1 / 4.1 (network, protocol)
DT_MS = 1.0
SEED = 1
N_LIQUID = 1000
F_E = 0.8
TAU_MEM = 64.0                  #ms
V_RESET, V_TH = 0.0, 20.0       #mV
T_REFRAC = 2.0                  #ms
TAU_SYN = 1.0                   #ms
P_IN, P_LIQ = 0.15, 0.10        #     input -> liquid, liquid -> liquid connection probabilities
W_MAX = 3.0
W_IN = {"mnist": 0.5, "nmnist": 4.0, "fmnist": 1.0}      #Sec. 4.3.2: the optimal input weight per dataset (Fig. 2)
STD_U, STD_TAU = 0.4, 150.0
A_E, TAU_E, A_I, TAU_I1, TAU_I2, TRACE_INC = 0.15, 10.0, 0.15, 5.0, 10.0, 0.1
T_TRAIN_MS, T_STATE_MS = 50.0, 250.0     #Sec. 4.1: 50 ms per sample in the self-organising phase, 250 ms for the state vector
PARTITIONS = {"mnist": 1, "nmnist": 4, "fmnist": 1}     #TODO Rename partitions? #Sec. 4.3.3: P = 1 best for MNIST / F-MNIST, P = 4 for N-MNIST
PEAK_HZ = 1000.0                #Hz   the rate of a pixel at FULL intensity. Not in the paper; the authors' code (LSM_BaseClass.py) uses the normalised
                                #     pixel AS the spike probability per 1 ms step, i.e. 1000 Hz at intensity 1. With the authors' image scaling (load_dataset)
                                #     the fresh liquid fires ~7 Hz at W_in 0.5 (Fig. 2: ~7). peak_hz / w_in in main() to calibrate.
NMNIST_SENSOR = (34, 34, 2)
READOUT = dict(lr=1e-3, batch_size=512, epochs=50, l2_lambda=1e-4, dropout=0.2)     #Function.py: LinearModel = Dropout(0.2) -> Linear on the RAW counts; Adam(lr 1e-3,
                                #weight_decay 1e-4), ReduceLROnPlateau(factor 0.5, patience 2), 50 epochs, batch 512
VAL_FRACTION = 1.0 / 6.0        #     Sec. 4.1: 10k of the 60k training images are validation, split off BEFORE the self-organising pass


   ######################################################################################
##########################   Stimulus (lazy datasets)   ###################################
   ######################################################################################

class PoissonDataset:
    """images (n, H, W) or (n, D) -> sample i = frames (T, D) bool, DRAWN WHEN ASKED from a stream keyed by (seed, i): the same index always gives the
    same spikes (resume-safe), nothing is stored. rate = intensity / max * peak_hz.
    A ModelInput source: frames only, the labels go to ModelInput(PoissonDataset(...), labels)."""

    def __init__(self, images, *, T: int, dt_ms: float = DT_MS, peak_hz: float = PEAK_HZ, seed: int = 0, full_scale: float | None = None):
        self.images = np.asarray(images)
        self.T, self.dt_ms, self.peak_hz, self.seed = int(T), float(dt_ms), float(peak_hz), int(seed)
        self._peak = float(self.images.max()) if full_scale is None else float(full_scale)   ##full_scale: fixed by the caller so a subset keeps the whole set's scale
        if self._peak <= 0:
            raise ValueError("All-zero images.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i) -> np.ndarray:
        i = int(i)
        img = self.images[i:i + 1].astype(np.float64)
        return poisson_prep(np.concatenate([img, np.full_like(img, self._peak)]), T=self.T, dt_ms=self.dt_ms, peak_hz=self.peak_hz,
                            rng=np.random.default_rng([self.seed, i]), flatten=True)[0]   ##the appended full-intensity image fixes the scale

    @property
    def n_input(self) -> int:
        return int(np.prod(self.images.shape[1:]))


def load_dataset(name: str, *, dt_ms: float = DT_MS, peak_hz: float = PEAK_HZ, seed: int = SEED):   ##returns the WHOLE set; DataCfg.n_train / n_test pick from it
    """-> (train, test, n_input). mnist / fmnist: RAW_DIR/<name>.npz (preprocessing.load_mnist keys) as Poisson spikes; nmnist: tonic (RAW_DIR)."""
    T = int(round(T_STATE_MS / dt_ms))
    if name in ("mnist", "fmnist"):
        x, y, xt, yt = load_mnist(RAW_DIR / f"{name}.npz")
        ##the authors' scaling (HLSPLSM.py, class MNIST): each image / its own pixel sum, then the split / its largest value (a typical digit peaks near 0.2)
        x, xt = (a / a.sum(axis=(1, 2), keepdims=True, dtype=np.float64) for a in (x, xt))
        x, xt = x / x.max(), xt / xt.max()
        train = ModelInput(PoissonDataset(x, T=T, dt_ms=dt_ms, peak_hz=peak_hz, seed=seed, full_scale=1.0), y, axis_labels="TX")
        test = ModelInput(PoissonDataset(xt, T=T, dt_ms=dt_ms, peak_hz=peak_hz, seed=seed + 1, full_scale=1.0), yt, axis_labels="TX")
        return train, test, train.n_input
    if name == "nmnist":
        import tonic
        from tonic_helpers import ClampBinary, FixedLength, US_TO_MS
        frame = tonic.transforms.Compose([tonic.transforms.ToFrame(sensor_size=NMNIST_SENSOR, time_window=int(dt_ms * US_TO_MS)), ClampBinary(dtype=np.bool_), FixedLength(T)])
        tr, te = (tonic.datasets.NMNIST(save_to=str(RAW_DIR), train=flag, transform=frame) for flag in (True, False))
        train = ModelInput.from_tonic(tr, axis_labels="TPHW")
        test = ModelInput.from_tonic(te, axis_labels="TPHW")
        return train, test, train.n_input
    raise ValueError(f"Unknown dataset {name!r}: mnist | nmnist | fmnist")


   ######################################################################################
##########################   Assemble   ###################################################
   ######################################################################################

def build(*, n_liquid: int = N_LIQUID, n_input: int = 784, w_in: float = W_IN["mnist"], seed: int = SEED, dt_ms: float = DT_MS, weights_format: str = "sparse",
          std: bool = True, estdp: bool = True, istdp: bool = True, std_pre_depletion: bool = False, save_net: bool = False) -> PhasicNet:
    """The HLSP-LSM network + physics, ready for pnet.train(train, runcfg(...), test_data=test).
    std / estdp / istdp = the paper's ablations (Table 5). weights_format "sparse" (default, update_on "hybrid_event") or "dense": same model."""
    n_exc = int(round(n_liquid * F_E)); n_inh = n_liquid - n_exc
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", n_input, True), ("exc", n_exc), ("inh", n_inh, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("in_inh", "input", "inh"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"),
                              ("inh_exc", "inh", "exc"), ("inh_inh", "inh", "inh"))
    pnet.builder.set_connectivity(in_exc=P_IN, in_inh=P_IN, exc_exc=P_LIQ, exc_inh=P_LIQ, inh_exc=P_LIQ, inh_inh=P_LIQ)
    pnet.builder.set_weights({"in_exc": {"values": [-w_in, w_in]}, "in_inh": {"values": [-w_in, w_in]},    ##fixed +-W_in, random sign per edge
                              "exc_exc": 0.0, "exc_inh": 0.0, "inh_exc": 0.0, "inh_inh": 0.0})              ##liquid weights start at ZERO (Algorithm 1 line 3)
    pnet.builder.spec_name = "hlsp_lsm_2026_spec"
    net = pnet.builder.realize(network_name=f"hlsp_lsm_2026_n{n_liquid}", post_offset=True, allow_autapse=False, save=False)   ##no self-connections (Sec. 3.1; LSM_BaseClass.py fill_diagonal(C, 0))
    if save_net:
        net.save_net(network_name=net.network_name, spec_name="hlsp_lsm_2026_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)
    logger.info("liquid: %d exc + %d inh, %d inputs; synapses %s", n_exc, n_inh, n_input, net.synapse_counts)

    sparse = weights_format == "sparse"
    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=dt_ms, weights_format=weights_format, update_on="hybrid_event" if sparse else "timestep"),
        synapse=HLSPSynapseCfg(b=None, tau_syn=TAU_SYN, std=std, std_u=STD_U, std_tau=STD_TAU, std_pre_depletion=std_pre_depletion),
        integrator=HLSPIntegratorCfg(tau_mem=TAU_MEM, v_rest=V_RESET, r=1.0, t_refrac=T_REFRAC),
        threshold=HLSPThresholdCfg(v_reset=V_RESET, v_th=V_TH, tau_th=None, adpt_th=None, t_refrac=T_REFRAC),
        plasticity=HLSPPlasticityCfg(tau_pretrace=TAU_E, tau_posttrace=TAU_E, A_ltp=A_E, A_ltd=A_E, eta_exc=1.0, eta_inh=1.0, w_min=-W_MAX, w_max=W_MAX,
                                     freeze=("in_exc", "in_inh"), A_I=A_I, tau_I1=TAU_I1, tau_I2=TAU_I2, trace_inc=TRACE_INC, estdp=estdp, istdp=istdp),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(HLSP_SPARSE if sparse else HLSP_DENSE)     ##the pack (defined after the protocol below)
    pnet.cfg_name, pnet.run_name = "hlsp_lsm_2026_cfg", "hlsp_lsm_2026_run"
    return pnet


   ######################################################################################
##########################   Protocol (Algorithm 1)   #####################################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class HLSPProtocolCfg(ProtocolCfg):
    """Algorithm 1's settings. Hashed into run_id with the rest of RunCfg except report_every (rate_probe IS hashed: leave it at its default
    for comparable run_ids).
    Args:
        t_train_ms, t_state_ms = presentation per sample in the self-organising pass (50) and for the state vector (250)
        train_window = "start": the first t_train_ms of each sample | "random": a random contiguous t_train_ms window inside t_state_ms |
            "subsample": t_train_ms worth of time steps drawn at random (sorted) from the whole sample, as the authors' HLSPLSM.py does for
            spiking data; for Poisson-encoded images "start" is equivalent
        partitions = P, the number of temporal partitions of the state vector (MNIST 1, N-MNIST 4)
        val_split = this fraction of the training samples is the VALIDATION block (ProtocolCfg.val_split; default 1/6 = Sec. 4.1's 10k of 60k), split off
            by ops.partition (DataHandler's stratified holdout) before Phase 2: the liquid self-organises on the rest ("fit" samples), the readout watches
            the validation loss for its lr schedule
        readout_* = readouts.LogisticReadout (readout_l2 = the code's Adam weight_decay 1e-4, which is LogisticReadout's l2 as is; Eq. 19 writes it
            as lambda/(2m) ||W||^2 per batch of m; readout_dropout = the authors' Dropout(0.2))
        rate_probe = the initial population firing rate (Hz) is logged over this many first training samples (Fig. 2: calibrate W_in / PEAK_HZ)
        n_train / n_test are DataCfg's (runcfg() below); test_data MUST be given to PhasicNet.train
    """
    protocol_type: str = "HLSP"
    t_train_ms: float = T_TRAIN_MS
    t_state_ms: float = T_STATE_MS
    train_window: str = "start"
    partitions: int = 1
    readout_lr: float = READOUT["lr"]
    readout_batch: int = READOUT["batch_size"]
    readout_epochs: int = READOUT["epochs"]
    readout_l2: float = READOUT["l2_lambda"]
    readout_dropout: float = READOUT["dropout"]
    val_split: float | None = VAL_FRACTION
    report_every: int = 500
    rate_probe: int = 200


class HLSPProtocol(Protocol):
    """
    Phase 2 (mode "train"): one unsupervised pass over the FIT samples (the training set minus the validation block), t_train_ms per sample, reset
    before each. Phase 3 (mode "eval", W guarded): state vectors of every fit, validation and test sample (P partitions x N liquid counts).
    Phase 4: softmax readout. STATE (proto__*): Resp_train / Resp_val / Resp_test (int16), Y_train / Y_val / Y_test, Rate = [sum, count] of the
    initial-rate probe; progress = {"stage": 0 train | 1 states-train | 2 states-val | 3 states-test, "i": samples done}: resumable
    with CheckpointCfg.every + out_dir.
    """
    protocol_type = "HLSP"
    needs_test_set = True    ##test_data=test to PhasicNet.train; Protocol.__init__ checks it

    def __init__(self, cfg: HLSPProtocolCfg | None, run, *, out_dir=None):
        super().__init__(HLSPProtocolCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c, net = self.cfg, run.ctx.net
        if c.train_window not in ("start", "random", "subsample"):
            raise ValueError(f"train_window must be 'start', 'random' or 'subsample', got {c.train_window!r}")
        dt = float(run.history.dt)
        self._n_tr_steps = int(round(c.t_train_ms / dt)); self._n_st_steps = int(round(c.t_state_ms / dt))
        self._bounds = np.linspace(0, self._n_st_steps, int(c.partitions) + 1).astype(int)
        n_avail, self._n_test = len(run.train_set), len(run.test_set)
        ##the validation block comes out of the training set HERE, before the liquid sees anything (ops.partition: keyed on the dataset -> the same split on a resume)
        (self._fit_set, val), = self.ops.partition(val_split=c.val_split or None)
        self._val_set = run.train_set.subset(np.zeros(0, dtype=np.int64), "val") if val is None else val
        self._n_train, self._n_val = len(self._fit_set), len(self._val_set)
        self._n_liq = net.n_total - net.n_input
        d = int(c.partitions) * self._n_liq
        self.Resp_train = np.zeros((self._n_train, d), dtype=np.int16)
        self.Resp_val = np.zeros((self._n_val, d), dtype=np.int16)
        self.Resp_test = np.zeros((self._n_test, d), dtype=np.int16)
        self.Y_train = np.full(self._n_train, -1, dtype=np.int64)
        self.Y_val = np.full(self._n_val, -1, dtype=np.int64)
        self.Y_test = np.full(self._n_test, -1, dtype=np.int64)
        self.Rate = np.zeros(2)     ##[sum of the per-sample rates, samples probed]

    # -------------------------------------------------------------------------
    def _train_segment(self, frames: np.ndarray, i: int) -> np.ndarray:
        n = self._n_tr_steps
        if self.cfg.train_window == "random" and frames.shape[0] > n:
            s = int(self.rng.rng("protocol", f"window:{i}").integers(0, frames.shape[0] - n + 1))
            return frames[s:s + n]
        if self.cfg.train_window == "subsample" and frames.shape[0] > n:
            return frames[np.sort(self.rng.rng("protocol", f"window:{i}").choice(frames.shape[0], size=n, replace=False))]
        return frames[:n]

    def _self_organise(self) -> bool:
        """Phase 2 = ops.for_samples over the fit set: sample i -> reset, its t_train_ms segment with plasticity on; the first rate_probe samples
        give the initial population rate (logged once). Returns False when a resumed run is already past it."""
        run, c, net, ops = self.run, self.cfg, self.run.ctx.net, self.ops
        run.set_mode("train")

        def step(i, frames, target):
            counts = self.present(self._train_segment(frames, i), target)   ##after reset=True below: Algorithm 1 line 6: V = 0, S = 0, traces = 0, x = 1
            if i < c.rate_probe:
                self.Rate += (capture.rate_hz(counts[net.n_input:], self._n_liq, self._n_tr_steps, run.history.dt), 1)
            done = i + 1
            if done == c.rate_probe or (done < c.rate_probe and done == self._n_train):
                logger.info("initial population firing rate over the first %d samples: %.1f Hz (paper: ~7 Hz MNIST / F-MNIST, ~30 Hz N-MNIST at the optimum W_in)",
                            int(self.Rate[1]), self.Rate[0] / self.Rate[1])

        def tail(lo, hi):      ##capture: mean |W| per REALIZED synapse, the fraction at a bound
            return (f"|W| mean {capture.mean_globalweight(run.ctx, signed=False):.3f}  at bound "
                    f"{100 * capture.pinned_weights(run.ctx, w_min=run.plasticity.w_min, w_max=run.plasticity.w_max)['saturated_frac']:.1f}%")
        return ops.for_samples(self._fit_set, step, stage=0, name="train", reset=True, extra=tail)

    def state_vector(self, frames: np.ndarray, target, key: str | None = None) -> np.ndarray:
        """Phase 3 for ONE sample: reset, t_state_ms from rest, liquid spike counts per partition -> (P * N_liquid,). ONE presentation
        (ops.binned_counts), the partitions are bins of it (no reset in between). group=("inh", "exc") = the liquid (there is no "liquid" group);
        the columns stay in true_neuronID order. key = the per-sample key ops.responses passes (unused here)."""
        self.run.reset_transient()
        return self.ops.binned_counts(frames[:self._n_st_steps], target, self._bounds[1:], group=("inh", "exc")).reshape(-1)

    def execute(self) -> dict:
        run, c, ops = self.run, self.cfg, self.ops
        ops.header()
        ##Phase 2
        if self._self_organise():
            ops.save_model("self-organised liquid")
        ##Phase 3 (W frozen from here on; STD keeps running inside the synapse)
        run.set_mode("eval")
        ops.responses(self._fit_set, self.state_vector, self.Resp_train, self.Y_train, stage=1, name="states-train", n=self._n_train)
        ops.responses(self._val_set, self.state_vector, self.Resp_val, self.Y_val, stage=2, name="states-val", n=self._n_val)
        ops.responses(run.test_set, self.state_vector, self.Resp_test, self.Y_test, stage=3, name="states-test", n=self._n_test)
        run.check_persistent_unchanged()
        ##Phase 4: the validation states are the readout's validation set; raw counts + input dropout as the authors' LinearModel
        val = dict(X_val=self.Resp_val, y_val=self.Y_val) if self._n_val else {}
        ro = fit_evaluate(self.Resp_train, self.Y_train, self.Resp_test, self.Y_test, lr=c.readout_lr, epochs=c.readout_epochs, batch_size=c.readout_batch,
                          l2=c.readout_l2, dropout=c.readout_dropout, standardize=False, **val)
        res, files = readout_report(ro, self.Resp_train, self.Y_train, self.Resp_test, self.Y_test)     ##the shared tail; this protocol's own keys below
        res.update({"val_acc": ro["val_acc"], "n_val": int(self._n_val),
                    "initial_rate_hz": float(self.Rate[0] / self.Rate[1]) if self.Rate[1] else None,
                    "liquid_spikes_per_sample_test": float(self.Resp_test.sum(1).mean()),
                    ##mean |W| per REALIZED synapse and the fraction at a bound, dense and sparse alike
                    "w_abs_mean": capture.mean_globalweight(run.ctx, signed=False),
                    "w_at_bound_frac": capture.pinned_weights(run.ctx, w_min=run.plasticity.w_min, w_max=run.plasticity.w_max)["saturated_frac"],
                    "paper_reference": "N = 1000 (mean of 10 runs): 95.63% MNIST (P=1, Table 5), 98.05% N-MNIST (P=4, Sec. 4.3.3; 95.42% at P=1), 84.26% F-MNIST (P=1); best (Sec. 4.2): 98.05% MNIST at N=8000, 98.49% N-MNIST at N=6000 (P=4), 86.64% F-MNIST at N=10000"})
        files.update(resp_val=self.Resp_val, y_val=self.Y_val, W=run.ctx.syn.W)
        return ops.finish(res, files=files, lines=(
            f"  test accuracy : {res['test_acc']:.2f}%  (train {res['train_acc']:.2f}%, val {res['val_acc']})   paper: {res['paper_reference']}",
            f"  liquid        : |W| mean {res['w_abs_mean']:.3f}, {100 * res['w_at_bound_frac']:.1f}% at the bound; initial rate "
            f"{None if res['initial_rate_hz'] is None else round(res['initial_rate_hz'], 1)} Hz; {self._n_train} + {self._n_val} + {self._n_test} state vectors of {c.partitions} x {self._n_liq}"))


##the runtime classes as ONE pack each (phasicnet.RuntimeClasses), reused by build() and by anyone assembling this model by hand
HLSP_DENSE = RuntimeClasses(synapse_cls=HLSPCurrentSynapse, integrator_cls=HLSPLIF, threshold_cls=HLSPThreshold, plasticity_cls=HLSPPlasticity,
                            noise_cls=TestNoise, phase_classes=(DefaultPhase,), protocol_classes=(HLSPProtocol,))
HLSP_SPARSE = replace(HLSP_DENSE, synapse_cls=HLSPCurrentSynapseSparse, plasticity_cls=HLSPPlasticitySparse)


def runcfg(*, every: int | None = None, n_train: int | None = None, n_test: int | None = None, **proto) -> RunCfg:
    """n_train / n_test -> DataCfg, which PICKS the samples (stratified, seeded, shuffled order)."""
    return RunCfg(data=DataCfg(make_test_split=False, test_fraction=0.0, n_train=n_train, n_test=n_test, shuffle=True),
                  checkpoint=CheckpointCfg(to_disk=False, every=every, keep_last=None), scorer=ScorerCfg(rate_hz_min=0.0),   ##keep_last=None: plots.over_weights reads the whole stage-0 snapshot series
                  history=HistoryCfg(include_input=False, record_spikes=False, record_counts=True), protocol=HLSPProtocolCfg(**proto))


   ######################################################################################
##########################   Run   ########################################################
   ######################################################################################

def main(*, dataset: str = "mnist", n_liquid: int = N_LIQUID, w_in: float | None = None, peak_hz: float = PEAK_HZ, partitions: int | None = None,
         n_train: int | None = None, n_test: int | None = None, seed: int = SEED, no_std: bool = False, no_istdp: bool = False, no_estdp: bool = False,
         std_pre_depletion: bool = False, dense: bool = False, every: int | None = None, resume: str | None = None, tag: str = "", out: str | None = None):
    """ONE run; every setting is a keyword argument (defaults = the constants).
        dataset: "mnist" | "nmnist" | "fmnist"; w_in / partitions None: the paper's optimum for the dataset (W_IN / PARTITIONS); peak_hz: the Poisson rate
        at full intensity; no_std / no_istdp / no_estdp: the ablations (no_istdp = the authors' E-STDP-only model: E-STDP on the inhibitory synapses too);
        std_pre_depletion: the Tsodyks-Markram reading of the STD release; dense: dense weights (default sparse); every: snapshot every N samples;
        resume: a snap_*.npz, or "auto" = the last one in out
    Returns the Run: run.report = HLSPProtocol's results (also <out>/results.json; log in <out>/run.jsonl, numbers in <out>/results.jsonl, status in <out>/status.json)."""
    train, test, n_input = load_dataset(dataset, peak_hz=peak_hz, seed=seed)
    w_in = W_IN[dataset] if w_in is None else w_in
    P = PARTITIONS[dataset] if partitions is None else partitions
    pnet = build(n_liquid=n_liquid, n_input=n_input, w_in=w_in, seed=seed, weights_format="dense" if dense else "sparse", std=not no_std,
                 estdp=not no_estdp, istdp=not no_istdp, std_pre_depletion=std_pre_depletion)
    out = Path(out) if out else RUNS_DIR / f"hlsp_lsm_2026_{dataset}_n{n_liquid}{('_' + tag) if tag else ''}" 
    if resume == "auto":
        found = snapshots(out)
        resume = str(found[-1]) if found else None
    return pnet.train(train, runcfg(every=every, partitions=P, train_window="subsample" if dataset == "nmnist" else "start", n_train=n_train, n_test=n_test),
                      test_data=test, out_dir=out, resume_from=resume, save_run=False, save_history=False, dataset_name=dataset)


if __name__ == "__main__":
    main()
