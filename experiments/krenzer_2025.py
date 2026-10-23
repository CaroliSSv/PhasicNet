#main - Krenzer & Bogdan (2025) RLSM assembled in PhasicNet: two Maass-type liquids with a winner-takes-all layer between them, the liquid 1 -> WTA
# synapses under the paper's reward-modulated STDP (krenzer2025.RSTDPEligibility), a logistic-regression readout on liquid 2, and the protocol
# (unsupervised first half, then feedback from the readout's prediction after every sample). What the paper states and what is inferred: krenzer2025.py
# header.
# NOT A REPLICATION: the paper's dynamic synapse model (El-Laithy & Bogdan 2010) and most of the network's numbers are not in the paper; every value
# marked INFERRED below is a choice made here, exposed as a build() argument so you can change it.
#
#THE ARCHITECTURE (Fig. 1-2 of the paper, with the gaps filled as marked):
#   input (70 = every 10th SHD channel) --p_in, U(0, W_IN)--> liquid 1 (108 exc + 27 inh on a 15 x 3 x 3 grid, P = C exp(-(d/2)^2))     INFERRED: p_in, weights, grid
#   liquid 1 exc --all_to_all, R-STDP--> wta (N_WTA exc LIF)        W_ex ~ U(0, 1) scaled to row sum alpha = 15 at creation (Eq. 5 "initialization step")
#   wta --all_to_all--> wta_inh (N_WTA_INH) --all_to_all, negative--> wta       the WTA's "inhibit each other via round inhibitory neurons"  INFERRED sizes/weights
#   wta --p, U(0, W)--> liquid 2 (same construction as liquid 1, its own grid)                                                           INFERRED p, weights
#   readout: logistic regression on liquid 2's exc spike counts per sample                                                              INFERRED features
# Synapses: the framework's delta-current synapse (I2V = sum of the weights of this step's presynaptic spikes), LIFeuler with R_m 30 / tau 20 / V_rest 0,
# StaticThreshold at 15 mV with reset to 0 (no refractory period: none stated). NoiseCfg zero. Sparse by default (dense gives the same model).
#
#THE PROTOCOL (RLSMProtocol, the paper's §4.2: feedback "introduced halfway through the training samples", "the weights are updated every sample"):
#   mode "train": one pass over the training set, a fresh liquid per sample (INFERRED), the local R-STDP acting online in every step; from sample
#   n_train * feedback_start on, a readout fitted on the features so far (refitted every refit_every samples) predicts this sample -> r = (yhat == y)
#   -> plasticity.apply_feedback (Eq. 7-9). Then mode "eval": the test set's features with the final W; the final readout is fitted on the training
#   features (final_fit) and scored on the test set.
#
#RUN (from the repo root):  python -c "import experiments, krenzer_2025; krenzer_2025.main(n_train=1000, n_test=500, tag='t1')"
# In parallel: parallel.Job(load=shd70_dataset). Runtime classes: the packs RLSM_SPARSE / RLSM_DENSE.
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict, replace
from pathlib import Path
##experiments/__init__.py puts the code folders on sys.path (run from the repo root)
import numpy as np
from phasicnet import PhasicNet, RuntimeClasses
from builder import _generate_netmeta
from cfg import Cfg, SimulationCfg, IntegratorCfg, ThresholdCfg, SynapseCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from protocols import Protocol
import capture
from krenzer2025 import RSTDPCfg, RSTDPEligibility, RSTDPEligibilitySparse, scale_rows_to
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFeuler, StaticThreshold
from original_dynamics import TestNoise
from phases import DefaultPhase
from paths import NETWORKS_DIR, RUNS_DIR
from readouts import LogisticReadout, fit_evaluate, readout_report
from deckers_2022 import shd_frames                 ##SHD via tonic at dt, cut / padded to t_max (Deckers build); downsampled to 70 channels below

logger = logging.getLogger("krenzer_2025")

#---------------------------------------------------------------- stated (Tables 1-2, §3-4)
N_LIQUID = 135
EXC_FRACTION = 0.8
C_EE, C_EI, C_II, C_IE = 0.3, 0.2, 0.1, 0.4     #     first index = presynaptic
LAMBDA = 2.0
TAU_MEM = 20.0                  #ms
R_M = 30.0
V_TH = 15.0                     #mV
V_REST = 0.0                    #mV
TAU_PRE = 300.0                 #ms
ETA = 1e-4
W_MAX = 1.0
X_TAR = 20.0
ALPHA = 15.0
TAU_TRACE = 500.0               #ms
N_INPUT = 70                    #     every 10th of SHD's 700 channels
CHANNEL_STEP = 10
N_CLASSES = 20
FEEDBACK_START = 0.5            #     "halfway through the training samples"
#---------------------------------------------------------------- INFERRED (not in the paper)
DT_MS = 1.0                     #ms   SHD events binned at 1 ms, as the Deckers build
T_MAX_MS = 1000.0               #ms   every SHD sample cut / zero-padded to 1 s
GRID = (15, 3, 3)               #     Maass 2002's 135-neuron liquid
SEED = 5
N_WTA = 20                      #     one WTA exc neuron per class
N_WTA_INH = 5                   #     the liquid's 4:1
P_IN = 0.1                      #     ~7 input channels per liquid neuron (Deckers' "approximately 7")
W_IN = 8.0                      #mV   U(0, W_IN): a spike of weight w moves V by R_m * w * dt / tau = 1.5 w mV. 8 chosen from a sweep on synthetic 70-channel
                                #     Poisson input (channel rates U(0, 40) Hz): liquid 1 at ~36 Hz, WTA ~7 Hz; 2 gave 0.5 Hz, 12 gave 58 Hz (CHANGES)
W_EE, W_EI, W_IE, W_II = 1.0, 1.0, 2.0, 1.0
W_WTA_INH = 3.0                 #     wta -> wta_inh
W_INH_WTA = 10.0                #     wta_inh -> wta (stored negative): one inh spike takes 15 mV off every WTA neuron
P_WTA_LIQ2 = 0.5
W_WTA_LIQ2 = 8.0                #     same sweep: liquid 2 at ~15 Hz with the WTA at ~7 Hz (2 gave 0.2 Hz, 4 gave 5 Hz)
MU = 0.1                        #     "set low"
K = 1.0                         #     Eq. 8/9 scale of e; None = the literal "same range" scaling (RSTDPCfg.k says why 1.0 is the default)
GAMMA = 1e-6                    #     "gamma << eta"


   ######################################################################################
##########################   Stimulus   ###################################################
   ######################################################################################

@dataclass(frozen=True)
class EveryNthChannel:   ##the channel subsampling as a frame-level tonic transform
    """(T, ..., 700) frames -> (T, ..., 700 / step): channels 0, step, 2 step, ... (the paper's 70 of SHD's 700, §4.1)."""
    step: int

    def __call__(self, frames):
        return frames[..., ::self.step]


def shd70_frames(train: bool, *, dt_ms: float = DT_MS, t_max_ms: float = T_MAX_MS, step: int = CHANNEL_STEP):   ##the WHOLE set; DataCfg picks
    """SHD -> ModelInput, sample i = (frames (T, 1, 70) bool, label): Deckers' shd_frames (events binned at dt, cut / padded to t_max) with every
    `step`-th channel as one more tonic transform (§4.1)."""
    return shd_frames(train, bin_ms=dt_ms, t_max_ms=t_max_ms, frame_transforms=(EveryNthChannel(step),))   ##one bin = one dt step


def shd70_dataset(*, dt_ms: float = DT_MS, t_max_ms: float = T_MAX_MS, step: int = CHANNEL_STEP):   ##returns the WHOLE set; DataCfg.n_train / n_test pick from it
    """(train, test) = the official SHD split at 70 channels, two ModelInputs -- what main() runs on and what a
    parallel.Job(load=shd70_dataset, load_kw={...}) child loads."""
    return (shd70_frames(True, dt_ms=dt_ms, t_max_ms=t_max_ms, step=step),
            shd70_frames(False, dt_ms=dt_ms, t_max_ms=t_max_ms, step=step))


   ######################################################################################
##########################   Assemble   ###################################################
   ######################################################################################

def build(*, n_liquid: int = N_LIQUID, grid=GRID, n_input: int = N_INPUT, n_wta: int = N_WTA, n_wta_inh: int = N_WTA_INH, seed: int = SEED,
          dt_ms: float = DT_MS, weights_format: str = "sparse", p_in: float = P_IN, w_in: float = W_IN, w_liquid=(W_EE, W_EI, W_IE, W_II),
          w_wta_inh: float = W_WTA_INH, w_inh_wta: float = W_INH_WTA, p_wta_liq2: float = P_WTA_LIQ2, w_wta_liq2: float = W_WTA_LIQ2,
          feedback: str = "forgiving", mu: float = MU, k: float | None = K, gamma: float = GAMMA, scale_after_feedback: bool = False, alpha: float = ALPHA, x_tar: float = X_TAR,
          eta: float = ETA, w_max: float = W_MAX, tau_pre: float = TAU_PRE, tau_trace: float = TAU_TRACE, save_net: bool = False) -> PhasicNet:
    """Two liquids + WTA + R-STDP, ready for pnet.train(data, runcfg(...), test_data=...). w_liquid = (W_EE, W_EI, W_IE, W_II) magnitudes of U(0, w)."""
    n_exc = int(round(n_liquid * EXC_FRACTION)); n_inh = n_liquid - n_exc
    w_ee, w_ei, w_ie, w_ii = (float(w) for w in w_liquid)
    kern = lambda C: {"kernel": "gaussian", "C": C, "lambda": LAMBDA}
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    b = pnet.builder
    b.set_neurons(("input", n_input, True), ("exc1", n_exc), ("inh1", n_inh, False, True), ("wta", n_wta), ("wta_inh", n_wta_inh, False, True),
                  ("exc2", n_exc), ("inh2", n_inh, False, True))
    b.set_positions(exc1=tuple(grid), inh1=tuple(grid), share="liquid1")
    b.set_positions(exc2=tuple(grid), inh2=tuple(grid), share="liquid2")
    liquid = lambda e, i: ((f"{e}_{e}", e, e), (f"{e}_{i}", e, i), (f"{i}_{e}", i, e), (f"{i}_{i}", i, i))
    b.set_synapses(("in_exc1", "input", "exc1"), ("in_inh1", "input", "inh1"), *liquid("exc1", "inh1"),
                   ("exc1_wta", "exc1", "wta"), ("wta_wtainh", "wta", "wta_inh"), ("wtainh_wta", "wta_inh", "wta"),
                   ("wta_exc2", "wta", "exc2"), ("wta_inh2", "wta", "inh2"), *liquid("exc2", "inh2"))
    conn = dict(in_exc1=p_in, in_inh1=p_in, exc1_wta="all_to_all", wta_wtainh="all_to_all", wtainh_wta="all_to_all", wta_exc2=p_wta_liq2, wta_inh2=p_wta_liq2)
    wts = {"in_exc1": (0.0, w_in), "in_inh1": (0.0, w_in), "exc1_wta": (0.0, w_max), "wta_wtainh": float(w_wta_inh), "wtainh_wta": -float(w_inh_wta),
           "wta_exc2": (0.0, w_wta_liq2), "wta_inh2": (0.0, w_wta_liq2)}
    for e, i in (("exc1", "inh1"), ("exc2", "inh2")):
        conn.update({f"{e}_{e}": kern(C_EE), f"{e}_{i}": kern(C_EI), f"{i}_{e}": kern(C_IE), f"{i}_{i}": kern(C_II)})
        wts.update({f"{e}_{e}": (0.0, w_ee), f"{e}_{i}": (0.0, w_ei), f"{i}_{e}": (0.0, w_ie), f"{i}_{i}": (0.0, w_ii)})    ##inh pre: Dale's sign by the builder
    b.set_connectivity(**conn)
    b.set_weights(wts)
    b.spec_name = "krenzer_2025_spec"
    net = b.realize(network_name=f"krenzer_2025_n{n_liquid}_wta{n_wta}", post_offset=True, save=False)
    ##Eq. 5 at initialisation: every WTA neuron's incoming R-STDP weights sum to alpha; the meta is generated again so network_id hashes the scaled W0
    m = net.syntype == net.type_key["exc1_wta"]
    rows = net.post_idx[m] - (net.synapseslices["exc1_wta"].post.start)
    W0 = net.W0[m].astype(np.float64)
    s = np.bincount(rows, weights=W0, minlength=n_wta); s[s == 0] = alpha
    net.W0[m] = (W0 * (alpha / s)[rows]).astype(np.float32)
    net.meta = _generate_netmeta(neuronspecs=net.neuronspecs, synapsespecs=net.synapsespecs, spec_name=net.meta.spec_name, network_name=net.meta.network_name,
                                 neuronslices=net.neuronslices, synapseslices=net.synapseslices, type_key=net.type_key, syntype=net.syntype,
                                 pre_idx=net.pre_idx, post_idx=net.post_idx, W0=net.W0, post_offset=net.post_offset, n_input=net.n_input, rng=b.rng)
    if save_net:
        net.save_net(network_name=net.network_name, spec_name="krenzer_2025_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)
    logger.info("liquids: 2 x (%d exc + %d inh) on %s; wta %d + %d inh; synapses %s", n_exc, n_inh, grid, n_wta, n_wta_inh, net.synapse_counts)

    sparse = weights_format == "sparse"
    frozen = tuple(name for name in net.synapsespecs if name != "exc1_wta")        ##only the R-STDP projection learns
    pnet.cfg = Cfg(
        simulation=SimulationCfg(dt=dt_ms, weights_format=weights_format, update_on="timestep"),
        synapse=SynapseCfg(b=None),
        integrator=IntegratorCfg(tau_mem=TAU_MEM, v_rest=V_REST, r=R_M),
        threshold=ThresholdCfg(v_reset=V_REST, v_th=V_TH, tau_th=None, adpt_th=None, t_refrac=None),
        plasticity=RSTDPCfg(tau_pretrace=tau_pre, tau_posttrace=1.0, A_ltp=eta, A_ltd=0.0, eta_exc=1.0, eta_inh=None, w_min=None, w_max=w_max, freeze=frozen,
                            syn_group="exc1_wta", x_tar=x_tar, mu=mu, alpha=alpha, tau_trace=tau_trace, feedback=feedback, k=k, gamma=gamma, scale_after_feedback=scale_after_feedback),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(RLSM_SPARSE if sparse else RLSM_DENSE)     ##the pack (defined after the protocol below)
    pnet.cfg_name, pnet.run_name = "krenzer_2025_cfg", "krenzer_2025_run"
    return pnet


   ######################################################################################
##########################   Protocol   ###################################################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class RLSMProtocolCfg(ProtocolCfg):
    """Train (online R-STDP, feedback from feedback_start on) -> test -> readout. Hashed into run_id with the rest of RunCfg.
    Args:
        feedback_start = fraction of the training samples after which the feedback acts (paper: halfway); the plasticity cfg says WHICH feedback (forgiving / strict / none)
        refit_every = the online readout is refitted on all features so far every this many samples (INFERRED; the paper: "rough estimator")
        refit_epochs = epochs of those online fits (cheap); readout_* = the final readout (readouts.LogisticReadout)
        features = "exc2" (default, INFERRED) | "exc1" | "exc1+exc2": whose spike counts the readout sees
        final_fit = "all": final readout on every training sample's online features | "after_feedback": only the feedback half | "final_pass": a second,
            frozen pass over the training set (costs a full pass)
        report_every = logging only (RunCfg.HASH_EXCLUDE)
        reset_between_samples (inherited) is ignored: every sample starts from a fresh liquid (reset=True on the ops passes)
        n_train / n_test are DataCfg's (runcfg() below); test_data MUST be given to PhasicNet.train
    """
    protocol_type: str = "RLSM"
    feedback_start: float = FEEDBACK_START
    refit_every: int = 100
    refit_epochs: int = 60
    readout_l2: float = 1e-3
    readout_lr: float = 0.02
    readout_epochs: int = 400
    features: str = "exc2"
    final_fit: str = "all"
    report_every: int = 200


class RLSMProtocol(Protocol):
    """
    STATE (proto__*): Resp_train (n_train, n_feat) int32, Y_train, Pred_train (int16, -1 = no prediction made), Resp_test, Y_test; progress = {"stage": 0 train |
    1 test | 2 final pass, "i": samples done}. The online readout is rebuilt from Resp_train on resume (not snapshotted).
    """
    protocol_type = "RLSM"
    needs_test_set = True    ##the official SHD test set (test_data=shd70_frames(train=False)); Protocol.__init__ checks it

    def __init__(self, cfg: RLSMProtocolCfg | None, run, *, out_dir=None):
        super().__init__(RLSMProtocolCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c, net = self.cfg, run.ctx.net
        if c.features not in ("exc2", "exc1", "exc1+exc2") or c.final_fit not in ("all", "after_feedback", "final_pass"):
            raise ValueError(f"RLSMProtocol: features={c.features!r} / final_fit={c.final_fit!r} not understood.")
        self._groups = c.features.split("+")
        self._n_train, self._n_test = len(run.train_set), len(run.test_set)
        self._n_fb = int(round(c.feedback_start * self._n_train))
        n_feat = sum(net.neuronslices[g].stop - net.neuronslices[g].start for g in self._groups)
        self.Resp_train = np.zeros((self._n_train, n_feat), dtype=np.int32)
        self.Resp_test = np.zeros((self._n_test, n_feat), dtype=np.int32)
        self.Resp_final = np.zeros((self._n_train if c.final_fit == "final_pass" else 0, n_feat), dtype=np.int32)
        self.Y_train = np.full(self._n_train, -1, dtype=np.int64)
        self.Y_test = np.full(self._n_test, -1, dtype=np.int64)
        self.Pred_train = np.full(self._n_train, -1, dtype=np.int16)
        self._readout = None

    def respond(self, frames, target, key: str | None = None) -> np.ndarray:
        """ONE sample on a fresh liquid (the pass's reset=True: reset_transient -- V, I2V, traces, eligibility; W kept) -> the feature groups' spike counts.
        key: unused here."""
        run = self.run
        if len(self._groups) == 1:
            return self.present(frames, target, group=self._groups[0])
        counts = self.present(frames, target)
        return np.concatenate([counts[run.ctx.net.neuronslices[g]] for g in self._groups])

    def _fit_online(self, i: int) -> None:
        c = self.cfg
        X, y = self.Resp_train[:i], self.Y_train[:i]
        if len(np.unique(y)) < 2:
            return
        self._readout = LogisticReadout(l2=c.readout_l2, lr=c.readout_lr, epochs=c.refit_epochs).fit(X, y)

    def _train(self) -> bool:
        """Stage 0 = ops.for_samples over the training set (reset=True = the fresh liquid per sample): sample i -> its features; from n_fb on,
        the online readout predicts it and the plasticity gets the feedback. Returns False when a resumed run is already past it."""
        run, c, pl, ops = self.run, self.cfg, self.run.plasticity, self.ops
        run.set_mode("train")
        fb_on = pl.feedback != "none"
        hits = [0, 0]                                                       ##[correct, predicted] online, for the progress line

        def step(i, frames, target):
            self.Resp_train[i] = self.respond(frames, target)
            self.Y_train[i] = int(target)
            if fb_on and i >= self._n_fb:
                if self._readout is None or (i - self._n_fb) % c.refit_every == 0:
                    self._fit_online(i)
                if self._readout is not None:
                    yhat = int(self._readout.predict(self.Resp_train[i:i + 1])[0])
                    self.Pred_train[i] = yhat
                    pl.apply_feedback(run.ctx, correct=yhat == int(target))
                    hits[1] += 1; hits[0] += int(yhat == int(target))

        def tail(lo, hi):
            return (f"feat spikes/sample {float(self.Resp_train[lo:hi].sum(1).mean()):.1f}  online acc {f'{100 * hits[0] / hits[1]:.1f}%' if hits[1] else '-'}  "
                    f"|W_ex| mean {float(np.abs(self._wex(run.ctx.syn.W)).mean()):.3f}")
        return ops.for_samples(run.train_set, step, stage=0, name="train", reset=True, extra=tail)

    def _wex(self, W):
        proj = self.run.ctx.net.synapseslices["exc1_wta"]
        if W.ndim == 2:
            return W[proj.post, proj.pre]
        return W[capture._syn_block(self.run.ctx, proj.post, proj.pre)]

    def execute(self) -> dict:
        run, c, pl, ops = self.run, self.cfg, self.run.plasticity, self.ops
        ops.header(feedback=pl.feedback, feedback_from=self._n_fb)
        self._train()
        model = ops.save_model()
        run.set_mode("eval")                                            ##W_ex is guarded from here on
        ops.responses(run.test_set, self.respond, self.Resp_test, self.Y_test, stage=1, name="test", n=self._n_test, reset=True)
        if c.final_fit == "final_pass":
            ops.responses(run.train_set, self.respond, self.Resp_final, self.Y_train, stage=2, name="final", n=self._n_train, reset=True)
        run.check_persistent_unchanged()
        ##READOUT
        if c.final_fit == "final_pass":
            X, y = self.Resp_final, self.Y_train
        elif c.final_fit == "after_feedback":
            X, y = self.Resp_train[self._n_fb:], self.Y_train[self._n_fb:]
        else:
            X, y = self.Resp_train, self.Y_train
        ro = fit_evaluate(X, y, self.Resp_test, self.Y_test, l2=c.readout_l2, lr=c.readout_lr, epochs=c.readout_epochs)
        pred = self.Pred_train[self._n_fb:]
        online = float(100 * np.mean(pred[pred >= 0] == self.Y_train[self._n_fb:][pred >= 0])) if (pred >= 0).any() else None
        res, files = readout_report(ro, self.Resp_train, self.Y_train, self.Resp_test, self.Y_test)     ##the shared tail; this protocol's own keys below
        res.update({"feedback": pl.feedback, "online_acc_after_feedback": online, "feedback_from": int(self._n_fb),
                    "feat_spikes_per_sample_train": float(self.Resp_train.sum(1).mean()), "feat_spikes_per_sample_test": float(self.Resp_test.sum(1).mean()),
                    "silent_samples_train": int((self.Resp_train.sum(1) == 0).sum()), "wex_mean": float(np.abs(self._wex(run.ctx.syn.W)).mean()),
                    "trained_model": model.name,
                    "paper_reference": "Fig. 3: forgiving feedback up to +2.7 % over the no-feedback baseline on SHD (70 channels), 100 runs; absolute accuracies are read off the violin plots only"})
        files.update(pred_train=self.Pred_train, resp_final=self.Resp_final, wex=np.asarray(self._wex(run.ctx.syn.W)))
        return ops.finish(res, files=files, lines=(
            f"  test accuracy : {res['test_acc']:.2f}%  (train {res['train_acc']:.2f}%, online after feedback {online})   feedback {pl.feedback}",))


##the runtime classes as ONE pack each (phasicnet.RuntimeClasses), reused by build() and by anyone assembling this model by hand
RLSM_DENSE = RuntimeClasses(synapse_cls=DeltaCurrentDense, integrator_cls=LIFeuler, threshold_cls=StaticThreshold, plasticity_cls=RSTDPEligibility,
                            noise_cls=TestNoise, phase_classes=(DefaultPhase,), protocol_classes=(RLSMProtocol,))
RLSM_SPARSE = replace(RLSM_DENSE, synapse_cls=DeltaCurrentSparse, plasticity_cls=RSTDPEligibilitySparse)


def runcfg(*, every: int | None = None, n_train: int | None = None, n_test: int | None = None, **proto) -> RunCfg:
    """One pass over train, one over test; no folds. `every` = snapshot every N samples (needs train(out_dir=)).
    n_train / n_test -> DataCfg, which PICKS the samples (stratified, seeded, shuffled order)."""
    return RunCfg(data=DataCfg(make_test_split=False, test_fraction=0.0, n_train=n_train, n_test=n_test, shuffle=True),
                  checkpoint=CheckpointCfg(to_disk=False, every=every), scorer=ScorerCfg(rate_hz_min=0.0),
                  history=HistoryCfg(include_input=False, record_spikes=False, record_counts=True), protocol=RLSMProtocolCfg(**proto))


   ######################################################################################
##########################   Run   ########################################################
   ######################################################################################

def main(*, n_train: int | None = None, n_test: int | None = None, feedback: str = "forgiving", feedback_start: float = FEEDBACK_START, refit_every: int = 100,
         features: str = "exc2", final_fit: str = "all", n_wta: int = N_WTA, mu: float = MU, k: float | None = K, gamma: float = GAMMA, w_in: float = W_IN,
         t_max_ms: float = T_MAX_MS, dense: bool = False, seed: int = SEED, every: int | None = None, resume: str | None = None, report_every: int = 200,
         tag: str = "", out: str | None = None):
    """ONE run; every setting is a keyword argument (defaults = the constants).
        feedback: "forgiving" | "strict" | "none"; features: "exc2" | "exc1" | "exc1+exc2"; final_fit: "all" | "after_feedback" | "final_pass"; k None = the
        literal auto scaling; dense: dense weights (default sparse); every: snapshot every N samples; resume: a snap_*.npz;
        out: the run folder (default runs/krenzer_2025_<tag or feedback>).
    Returns the Run: run.report = RLSMProtocol's results (also <out>/results.json; log in <out>/run.jsonl, numbers in <out>/results.jsonl, status in <out>/status.json)."""
    train, test = shd70_dataset(t_max_ms=t_max_ms)
    pnet = build(n_wta=n_wta, seed=seed, weights_format="dense" if dense else "sparse", feedback=feedback, mu=mu, k=k, gamma=gamma, w_in=w_in)
    out = Path(out) if out else RUNS_DIR / f"krenzer_2025_{tag or feedback}"
    rc = runcfg(every=every, n_train=n_train, n_test=n_test, feedback_start=feedback_start, refit_every=refit_every, features=features,
                final_fit=final_fit, report_every=report_every)
    return pnet.train(train, rc, test_data=test, out_dir=out, resume_from=resume, save_run=False, save_history=False, dataset_name="shd70")


if __name__ == "__main__":
    main()
