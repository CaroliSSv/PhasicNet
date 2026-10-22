#hlsp2026 - Luo, Chu, Wu, Rasheed & Wang (2026), "HLSP-LSM: Enhancing image recognition performance of liquid state machines via brain-inspired
# hybrid long short-term plasticity", Neural Networks 204:109245 -- the model as PhasicNet runtime modules (+ the cfgs that go with them). ONE
# file, so the whole model is in one place. The network (random sparse liquid, +-W_in input) and the protocol (unsupervised pass -> state
# vectors -> softmax readout) are built in experiments/hlsp_lsm_2026.py.
#
#SOURCE OF TRUTH: the paper's equations (1)-(18), Algorithm 1, Tables 2-3, Sec. 4.1 (the authors' repository ships the plasticity only as
# compiled binaries). Where the paper is silent the value is a cfg PARAMETER marked INFERRED (experiments/hlsp_lsm_2026.py).
#
#WHERE: membrane + refractory clamp, Eq. (1)-(2) -> HLSPLIF, HLSPThreshold; current Eq. (3) + STD Eq. (18) -> HLSPCurrentSynapse;
# E-STDP Eq. (8)-(10) + I-STDP Eq. (12)-(15) -> HLSPPlasticity (traces updated with S[t] FIRST, Algorithm 1). Dale, Eq. (4): inhibitory
# columns stored NEGATIVE; clip Eq. (16)-(17) -> Run._clamp (w_min = -W_max, w_max = +W_max); input weights fixed +-W_in (freeze by name);
# no noise, dt = 1 ms (Table 2).
#
#SPIKE TIMING: neur.Spikes at the start of step m = this step's INPUT spikes + the liquid spikes of step m-1. That IS Eq. (3): no delay ring.
# In the plasticity step, neur.pre_spike = Spikes AFTER this step's threshold (input m, liquid m) and neur.post_spike = liquid m: pre and post
# spikes of the SAME step, as Eq. (10) and (15) are written.
#
#UNITS: ms, mV; W dimensionless (Table 2: R = 1 MOhm with current in nA gives mV, and (1 - beta) is absorbed: W IS the jump of V per spike).
from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging

import numpy as np
from synapticindex import touched, elapsed   ##the lazy-decay helpers of update_on "event"

from cfg import SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg
from states import RunContext, NeuronalState, SynapticState
from dynamics import Synapse, Integrator, Threshold, Plasticity
from diehlcook import _row_vector
from deckers2022 import steps_of

logger = logging.getLogger(__name__)


   ######################################################################################
##########################   Synapse: exponential current + STD   #########################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class HLSPSynapseCfg(SynapseCfg):
    """SynapseCfg + Eq. (3) and (18).
    Args:
        tau_syn = ms, decay of the synaptic current (Table 2: 1 ms)
        std = short-term depression on the LIQUID columns (Table 3: U = 0.4, tau_x = 150 ms). False = x stays 1 (the paper's E/I-STDP ablation)
        std_u, std_tau = U and tau_x
        std_pre_depletion = False: the current uses x_j[t-1] AFTER the depletion of spike S_j[t-1] (Eq. 3 and 18 read literally).
            True: the Tsodyks-Markram reading, the spike releases U times the resources available BEFORE its own depletion. INFERRED either way.
    """
    tau_syn: float = 1.0
    std: bool = True
    std_u: float = 0.4
    std_tau: float = 150.0
    std_pre_depletion: bool = False
    cfg_type: str = field(default="HLSPSynapseCfg", kw_only=True)


class HLSPCurrentSynapse(Synapse):
    """
    Eq. (3):  C_i[t] = alpha C_i[t-1] + sum_j W_ij drive_j,  drive_j = S_j[t] (input) | U x_j[t-1] S_j[t-1] (liquid)      -> I2V = C
    Eq. (18): x_j[t] = gamma x_j[t-1] + (1 - gamma) - U x_j[t-1] S_j[t]   per LIQUID presynaptic neuron, x(0) = 1
    C (n_post) and X (n_total; input entries stay 1) are module state (reset_transient restores C = 0, X = 1 per sample, Algorithm 1 line 6).
    ONE matrix-vector product per step (dense) / one bincount (sparse): the STD factor multiplies the spike VECTOR, not W.
    """
    weights_format = "dense"
    writes = ("I2V",)
    requires = ()

    def __init__(self, cfg: HLSPSynapseCfg, sim_cfg: SimulationCfg = None):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("HLSPCurrentSynapse requires HLSPSynapseCfg and SimulationCfg with dt.")
        if cfg.tau_syn <= 0 or cfg.std_tau <= 0 or not (0.0 <= cfg.std_u <= 1.0):
            raise ValueError("HLSPCurrentSynapse requires tau_syn > 0, std_tau > 0, 0 <= std_u <= 1.")
        self.b = None if cfg.b is None else float(cfg.b)
        self.tau_syn = float(cfg.tau_syn)
        self.std = bool(cfg.std)
        self.std_u = float(cfg.std_u)
        self.std_tau = float(cfg.std_tau)
        self.std_pre_depletion = bool(cfg.std_pre_depletion)
        self.dt = float(sim_cfg.dt)
        self._alpha = float(np.exp(-self.dt / self.tau_syn))
        self._gamma = float(np.exp(-self.dt / self.std_tau))
        self.C = None
        self.X = None
        self._n_input = 0
        self._drive = None          ##scratch (n_total,) float64

    def configure_for_network(self, net) -> None:
        self.C = np.zeros(net.n_post, dtype=np.float64)
        self.X = np.ones(net.n_total, dtype=np.float64)
        self._n_input = net.n_input
        self._drive = np.zeros(net.n_total, dtype=np.float64)

    def _std_update_and_drive(self, S: np.ndarray) -> np.ndarray:
        """drive (n_total,) float64 = Eq. (3)'s spike vector: input spikes as 1.0; liquid spikes (last step's) as U x_j, after Eq. (18) has
        been applied to X with those same spikes (x_j[t-1] in the paper's indexing). STD off: liquid spikes as 1.0, X untouched."""
        n0 = self._n_input
        drive = self._drive
        drive[...] = S
        if not self.std:
            return drive
        x = self.X[n0:]
        s_liq = drive[n0:]                                          ##0/1 of last step's liquid spikes
        avail = x * s_liq                                           ##resources of the spiking columns BEFORE this spike's depletion
        x *= self._gamma; x += (1.0 - self._gamma); x -= self.std_u * avail      ##Eq. (18)
        if self.std_pre_depletion:
            s_liq[...] = avail; s_liq *= self.std_u                 ##Tsodyks-Markram: the spike releases U x_before
        else:
            s_liq *= x; s_liq *= self.std_u                         ##literal Eq. (3): U x_j[t-1] with x already depleted by S_j[t-1]
        return drive

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        drive = self._std_update_and_drive(neur.Spikes)
        C = self.C
        C *= self._alpha
        C += np.dot(syn.W, drive.astype(np.float32)).astype(np.float64)
        if self.b is not None:
            C += self.b
        upd.I2V[...] = C


class HLSPCurrentSynapseSparse(HLSPCurrentSynapse):
    """The same current over the EDGE LIST: per-edge W_k * drive[pre_k] summed into post rows (one bincount). RETURNS I."""
    weights_format = "sparse"

    def step(self, ctx: RunContext) -> np.ndarray:
        neur, syn = ctx.neur, ctx.syn
        drive = self._std_update_and_drive(neur.Spikes)
        C = self.C
        C *= self._alpha
        active = np.flatnonzero(drive)
        if active.size:
            e = syn.weights.index.synIDs_by_pre(active)
            if e.size:
                e.sort()
                C += np.bincount(syn.post[e], weights=syn.W[e] * drive[syn.pre[e]], minlength=syn.weights.n_post)
        if self.b is not None:
            C += self.b
        return C.astype(np.float32)


   ######################################################################################
##########################   Integrator + threshold: discrete LIF of Eq. (1)-(2)   #########
   ######################################################################################

@dataclass(frozen=True, slots=True)
class HLSPIntegratorCfg(IntegratorCfg):
    """IntegratorCfg (tau_mem 64 ms, v_rest 0, r 1.0: (1 - beta) R is absorbed into W) + the refractory clamp.
    Args: t_refrac = ms (Table 2: 2 ms): V held at 0 (no integration) while Since < steps(t_refrac)."""
    t_refrac: float = 2.0
    cfg_type: str = field(default="HLSPIntegratorCfg", kw_only=True)


class HLSPLIF(Integrator):
    """Eq. (2): V[t] = beta V[t-1] + C[t]  ->  dV = (beta - 1) V + r C;  beta = exp(-dt / tau_mem). Refractory rows: dV = 0 (V is 0 after the reset)."""
    writes = ("dV",)
    requires = ("Since",)

    def __init__(self, cfg: HLSPIntegratorCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("HLSPLIF requires HLSPIntegratorCfg and SimulationCfg with dt.")
        if cfg.tau_mem <= 0 or cfg.t_refrac < 0:
            raise ValueError("HLSPLIF requires tau_mem > 0 and t_refrac >= 0.")
        self.dt = float(sim_cfg.dt)
        self.v_rest = np.float32(cfg.v_rest)
        self.tau_mem = float(cfg.tau_mem)
        self.r = float(cfg.r)
        self.t_refrac = float(cfg.t_refrac)
        self._beta = float(np.exp(-self.dt / self.tau_mem))
        self._refrac_steps = steps_of(self.t_refrac, self.dt)
        self._gate = None

    def configure_for_network(self, net) -> None:
        self._gate = np.zeros(net.n_post, dtype=np.float64)

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        dV = neur.V.astype(np.float64)
        dV *= (self._beta - 1.0)
        dV += upd.I2V.astype(np.float64) * self.r
        np.greater_equal(neur.Since, self._refrac_steps, out=self._gate)
        dV *= self._gate
        upd.dV[...] = dV


@dataclass(frozen=True, slots=True)
class HLSPThresholdCfg(ThresholdCfg):
    """ThresholdCfg: v_reset 0, v_th 20 mV, t_refrac 2 ms (Table 2). tau_th / adpt_th unused (fixed threshold)."""
    cfg_type: str = field(default="HLSPThresholdCfg", kw_only=True)


class HLSPThreshold(Threshold):
    """Eq. (2): spike iff V >= V_th (>=, as written) and the neuron is not refractory (Since >= steps(t_refrac)). Reset to v_reset by the orchestra."""
    writes = ("rec_spikes",)
    requires = ("Since",)

    def __init__(self, cfg: HLSPThresholdCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("HLSPThreshold requires HLSPThresholdCfg and SimulationCfg with dt.")
        self.dt = float(sim_cfg.dt)
        self.v_th = float(cfg.v_th)
        self.v_reset = np.float32(cfg.v_reset)
        self.t_refrac = 0.0 if cfg.t_refrac is None else float(cfg.t_refrac)
        self._refrac_steps = steps_of(self.t_refrac, self.dt)
        self._gate = None

    def configure_for_network(self, net) -> None:
        self._gate = np.zeros(net.n_post, dtype=bool)

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        np.greater_equal(neur.Since, self._refrac_steps, out=self._gate)
        np.greater_equal(neur.V, self.v_th, out=upd.rec_spikes)
        np.logical_and(upd.rec_spikes, self._gate, out=upd.rec_spikes)


   ######################################################################################
##########################   Plasticity: E-STDP + I-STDP   ################################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class HLSPPlasticityCfg(PlasticityCfg):
    """PlasticityCfg mapped onto Table 3 + Eq. (7)-(15). Parent fields: tau_pretrace = tau_posttrace = tau_E (10 ms), A_ltp = A_ltd = A_E (0.15),
    eta_exc = 1 (the increments are applied as written), w_min = -W_max / w_max = +W_max (3; the per-sign clamp of Eq. 16), freeze = the input
    projections by name (set in build()).
    Args:
        A_I, tau_I1, tau_I2 = I-STDP intensity and time constants (0.15, 5 ms, 10 ms)
        trace_inc = a, the increment of every trace per spike (0.1)
        estdp = False: the exc columns do not learn. istdp = False: the inhibitory columns follow E-STDP instead of I-STDP (the paper's "E-STDP"
            ablation rows, Table 5 = the authors' LSMESTDP, which puts ESTDP on all four projections)
    """
    tau_pretrace: float = 10.0
    tau_posttrace: float = 10.0
    A_ltp: float = 0.15
    A_ltd: float = 0.15
    eta_exc: float = 1.0
    eta_inh: Optional[float] = 1.0
    w_min: Optional[float] = -3.0
    w_max: Optional[float] = 3.0
    A_I: float = 0.15
    tau_I1: float = 5.0
    tau_I2: float = 10.0
    trace_inc: float = 0.1
    estdp: bool = True
    istdp: bool = True
    cfg_type: str = field(default="HLSPPlasticityCfg", kw_only=True)


class HLSPPlasticity(Plasticity):
    """
    E-STDP (exc pre columns):   dW_ij = A_E (S_i e^E_j - S_j e^E_i)                          Eq. (10); e^E = exp(-dt/tau_E) e^E + a S   Eq. (8)-(9)
    I-STDP (inh pre columns):   dW^I_ij = S_j e^I_i + S_i e^I_j - A_I S_j S_i, e^I = A_I/(1-r) (e^I1 - r e^I2)   Eq. (12)-(15), r = tau_I1 / tau_I2
                                stored weight = -W^I, so dW_ij = -(S_j e^I_i + S_i e^I_j - A_I S_j S_i)
    Traces are per NEURON (one pre value per column, one post value per row), updated with THIS step's spikes before the increments
    (Algorithm 1). eta_exc / eta_inh scale the two rules (1 = the paper). Frozen projections (the input) get dW = 0; the orchestra clamps.
    dense: two outer products per rule over (n_post, n_total) -- fine to N ~ 2000; beyond that use the sparse module.
    """
    weights_format = "dense"
    writes = ("dW",)
    requires = ("pre_spike", "post_spike")

    def __init__(self, cfg: HLSPPlasticityCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("HLSPPlasticity requires HLSPPlasticityCfg and SimulationCfg with dt.")
        if cfg.tau_pretrace <= 0 or cfg.tau_I1 <= 0 or cfg.tau_I2 <= 0 or cfg.tau_I1 == cfg.tau_I2:
            raise ValueError("HLSPPlasticity requires tau_pretrace > 0, tau_I1 > 0, tau_I2 > 0, tau_I1 != tau_I2.")
        self.dt = float(sim_cfg.dt)
        self.tau_pretrace = float(cfg.tau_pretrace)
        self.tau_posttrace = float(cfg.tau_posttrace)
        self.A_ltp = float(cfg.A_ltp)
        self.A_ltd = float(cfg.A_ltd)
        self.eta_exc = float(cfg.eta_exc)
        self.eta_inh = 1.0 if cfg.eta_inh is None else float(cfg.eta_inh)
        self.w_min = None if cfg.w_min is None else float(cfg.w_min)
        self.w_max = None if cfg.w_max is None else float(cfg.w_max)
        self.freeze = tuple(cfg.freeze)
        self.A_I = float(cfg.A_I)
        self.tau_I1 = float(cfg.tau_I1)
        self.tau_I2 = float(cfg.tau_I2)
        self.trace_inc = float(cfg.trace_inc)
        self.estdp = bool(cfg.estdp)
        self.istdp = bool(cfg.istdp)
        self._dE_pre = float(np.exp(-self.dt / self.tau_pretrace))
        self._dE_post = float(np.exp(-self.dt / self.tau_posttrace))
        self._dI1 = float(np.exp(-self.dt / self.tau_I1))
        self._dI2 = float(np.exp(-self.dt / self.tau_I2))
        r = self.tau_I1 / self.tau_I2
        self._kI = self.A_I / (1.0 - r)                             ##Eq. (14)'s prefactor
        self._rI = r
        ##module-owned traces (Capitalised -> state_dict, reset per sample by reset_transient)
        self.E_pre = None; self.E_post = None
        self.I1_pre = None; self.I2_pre = None; self.I1_post = None; self.I2_post = None
        self._exc_col = None        ##(n_total,) float32: 1 on excitatory (non-input) columns
        self._inh_col = None        ##(n_total,) float32: 1 on inhibitory columns
        self._tmp = None            ##dense scratch (n_post, n_total)

    def configure_for_network(self, net) -> None:
        self.conf_traces(net)                                           ##per-neuron here, per-synapse in the event mode
        exc = np.ones(net.n_total, dtype=np.float32); exc[:net.n_input] = 0.0; exc[net.inh_slice] = 0.0
        inh = np.zeros(net.n_total, dtype=np.float32); inh[net.inh_slice] = 1.0
        self._exc_col = exc * (self.eta_exc * (1.0 if self.estdp else 0.0))
        self._inh_col = inh * self.eta_inh                              ##also used when istdp=False (step() branches)
        self.conf_scratch(net)

    def conf_traces(self, net) -> None:
        self.E_pre = np.zeros(net.n_total, dtype=np.float32); self.E_post = np.zeros(net.n_post, dtype=np.float32)
        self.I1_pre = np.zeros(net.n_total, dtype=np.float32); self.I2_pre = np.zeros(net.n_total, dtype=np.float32)
        self.I1_post = np.zeros(net.n_post, dtype=np.float32); self.I2_post = np.zeros(net.n_post, dtype=np.float32)

    def conf_scratch(self, net) -> None:
        self._tmp = np.zeros((net.n_post, net.n_total), dtype=np.float32)

    def _rule(self, S_pre: np.ndarray, S_post: np.ndarray, pre: np.ndarray, e_pre, e_post, i1_pre, i2_pre, i1_post, i2_post) -> np.ndarray:
        """Both rules per edge, given the edge's spikes and the SIX trace values at its two ends (already updated with this step's spikes,
        Algorithm 1). Same operand order as the dense step; shared by both trace layouts."""
        dw = S_post * e_pre; np.multiply(dw, self.A_ltp, out=dw)
        ltd = e_post * S_pre; np.multiply(ltd, self.A_ltd, out=ltd)
        np.subtract(dw, ltd, out=dw)
        if not self.istdp:                                              ##inhibitory edges follow E-STDP too (see the dense step)
            inh = dw * self._inh_col[pre]
            np.multiply(dw, self._exc_col[pre], out=dw)
            np.subtract(dw, inh, out=dw)
            return dw
        np.multiply(dw, self._exc_col[pre], out=dw)
        eI_pre, eI_post = self._eI(i1_pre, i2_pre), self._eI(i1_post, i2_post)
        inh = eI_post * S_pre
        inh += S_post * eI_pre
        inh -= self.A_I * (S_post * S_pre)                            ##Eq. (15)'s - A_I S_j S_i, same operand order as dense
        np.multiply(inh, self._inh_col[pre], out=inh)
        np.subtract(dw, inh, out=dw)
        return dw

    def update_traces(self, neur: NeuronalState) -> None:
        a = self.trace_inc
        for tr, dec, s in ((self.E_pre, self._dE_pre, neur.pre_spike), (self.E_post, self._dE_post, neur.post_spike),
                           (self.I1_pre, self._dI1, neur.pre_spike), (self.I2_pre, self._dI2, neur.pre_spike),
                           (self.I1_post, self._dI1, neur.post_spike), (self.I2_post, self._dI2, neur.post_spike)):
            np.multiply(tr, dec, out=tr)
            tr += a * s                                             ##one temp per trace; sizes are n_total / n_post

    def _eI(self, I1: np.ndarray, I2: np.ndarray) -> np.ndarray:
        """Eq. (14): A_I / (1 - r) (e^I1 - r e^I2)."""
        return self._kI * (I1 - self._rI * I2)

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        self.update_traces(neur)                                    ##Algorithm 1: traces with S[t] first ...
        S_pre, S_post, dW, tmp = neur.pre_spike, neur.post_spike, upd.dW, self._tmp
        ##E-STDP: A (S_post e_pre - e_post S_pre), exc columns
        np.multiply(S_post[:, None], self.E_pre[None, :], out=dW); np.multiply(dW, self.A_ltp, out=dW)
        np.multiply(self.E_post[:, None], S_pre[None, :], out=tmp); np.multiply(tmp, self.A_ltd, out=tmp)
        np.subtract(dW, tmp, out=dW)                                    ##dW = the E-STDP magnitude on EVERY column
        if not self.istdp:
            ##istdp=False: the authors' E-STDP-only model. The inhibitory magnitude follows E-STDP, stored negative, so it is SUBTRACTED like I-STDP.
            np.multiply(dW, self._inh_col[None, :], out=tmp)
            np.multiply(dW, self._exc_col[None, :], out=dW)
            np.subtract(dW, tmp, out=dW)
        else:
            np.multiply(dW, self._exc_col[None, :], out=dW)
            ##I-STDP: -(e^I_post S_pre + S_post e^I_pre - A_I S_post S_pre), inh columns (stored negative)
            eI_pre, eI_post = self._eI(self.I1_pre, self.I2_pre), self._eI(self.I1_post, self.I2_post)
            np.multiply(eI_post[:, None], S_pre[None, :], out=tmp)
            tmp += S_post[:, None] * eI_pre[None, :]
            tmp -= self.A_I * (S_post[:, None] * S_pre[None, :])      ##Eq. (15)'s - A_I S_j S_i (same-step pairs)
            np.multiply(tmp, self._inh_col[None, :], out=tmp)
            np.subtract(dW, tmp, out=dW)
        ##frozen projections (the input) and unrealized pairs
        for proj in syn.freeze_slices:
            dW[proj.post, proj.pre] = 0.0
        np.multiply(dW, syn.syn_mask, out=dW)


class HLSPPlasticitySparse(HLSPPlasticity):
    """The same two rules over the edge list (per-edge, same operand order). RETURNS dw.
    update_on: hybrid_event (per-neuron traces decayed every step, rule at spiking edges; bit-identical to timestep).
    update_on: event: the six traces live PER SYNAPSE (E_pre_e ... I2_post_e, T_e) and are decayed lazily at the synapses a spike touches
    (synapticindex.py); the per-neuron arrays are None in that mode. Float tolerance against timestep.
    In BOTH event modes step() returns the pair (k, dw_k) -- the touched plastic edges and their dw; Run.step_sparse commits W[k] += dw_k and
    clips W[k] only. "timestep" returns the full dw."""
    weights_format = "sparse"
    _EDGE_TRACES = ("E_pre_e", "E_post_e", "I1_pre_e", "I2_pre_e", "I1_post_e", "I2_post_e")

    def __init__(self, cfg: HLSPPlasticityCfg, sim_cfg: SimulationCfg):
        super().__init__(cfg, sim_cfg)
        self.update_on = sim_cfg.update_on

    def conf_traces(self, net) -> None:
        if self.update_on != "event":
            return super().conf_traces(net)
        ##per-synapse traces + the step of each edge's last event; no per-neuron traces
        self.E_pre = self.E_post = self.I1_pre = self.I2_pre = self.I1_post = self.I2_post = None
        for name in self._EDGE_TRACES:
            setattr(self, name, np.zeros(net.n_synapses, dtype=np.float32))
        self.T_e = np.zeros(net.n_synapses, dtype=np.int64)

    def conf_scratch(self, net) -> None:
        self._tmp = None

    def _dw_at(self, syn: SynapticState, neur: NeuronalState, k: np.ndarray | None) -> np.ndarray:
        """Per-neuron traces (update_on timestep / hybrid_event)."""
        pre, post = (syn.pre, syn.post) if k is None else (syn.pre[k], syn.post[k])
        return self._rule(neur.pre_spike[pre], neur.post_spike[post], pre, self.E_pre[pre], self.E_post[post],
                          self.I1_pre[pre], self.I2_pre[pre], self.I1_post[post], self.I2_post[post])

    def _step_event(self, ctx: RunContext) -> tuple:
        """Event mode: at the plastic edges a spike touches, bring the six per-synapse traces up to date lazily and apply this
        step's update (Algorithm 1: traces with S[t] first), then the two rules on them. RETURNS (k, dw_k).
        t = sim.train_step, not current_runstep: eval steps are not elapsed time for the traces."""
        neur, syn, t = ctx.neur, ctx.syn, ctx.sim.train_step
        k = touched(syn, neur.post_spike > 0, neur.pre_spike > 0)
        if not k.size:
            return k, np.zeros(0, dtype=np.float32)
        pre, post = syn.pre[k], syn.post[k]
        S_pre, S_post = neur.pre_spike[pre], neur.post_spike[post]
        n = elapsed(self.T_e, k, t) + 1                             ##owed decay steps + this step's own decay (update_traces: decay, then + a S)
        a = self.trace_inc
        pE_pre, pE_post, pI1, pI2 = (np.power(d, n) for d in (self._dE_pre, self._dE_post, self._dI1, self._dI2))     ##the decay factors ONCE per touched batch: four powers (tau_pretrace / tau_posttrace / tau_I1 / tau_I2), not six
        for name, p, s in (("E_pre_e", pE_pre, S_pre), ("E_post_e", pE_post, S_post), ("I1_pre_e", pI1, S_pre),
                           ("I2_pre_e", pI2, S_pre), ("I1_post_e", pI1, S_post), ("I2_post_e", pI2, S_post)):
            X = getattr(self, name)
            X[k] = (X[k] * p).astype(np.float32, copy=False) + a * s      ##= decayed(X, k, dec, n) with the power shared; bit-identical
        self.T_e[k] = t
        return k, self._rule(S_pre, S_post, pre, self.E_pre_e[k], self.E_post_e[k], self.I1_pre_e[k], self.I2_pre_e[k], self.I1_post_e[k], self.I2_post_e[k])     ##the pair; k is already restricted to plastic edges

    def step(self, ctx: RunContext):
        neur, syn = ctx.neur, ctx.syn
        if self.update_on == "event":
            return self._step_event(ctx)
        self.update_traces(neur)
        if self.update_on == "hybrid_event":
            k = touched(syn, neur.post_spike > 0, neur.pre_spike > 0)   ##the PLASTIC touched edges (spiking_syns + plastic_mask); the pair (k, dw_k) is returned
            return k, self._dw_at(syn, neur, k)
        dw = self._dw_at(syn, neur, None)
        if syn.plastic_mask is not None:
            np.multiply(dw, syn.plastic_mask, out=dw)
        return dw
