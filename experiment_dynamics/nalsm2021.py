#nalsm2021 - Ivanov & Michmizos (2021), "Increasing liquid state machine performance with edge-of-chaos dynamics organized by astrocyte-modulated
# plasticity", NeurIPS 2021 -- the model as PhasicNet runtime modules (+ the cfgs that go with them). ONE file, so the whole model is in one place.
# The network (3-D liquid, gaussian kernel, +-W input), the input neurons (user-side encoder) and the protocol (STDP initialization pass ->
# per-sample spike counts with astrocyte-modulated STDP -> linear readout) are built in experiments/nalsm_2021.py.
#
#SOURCE OF TRUTH: the paper's Eq. (1)-(9) and Sec. 2.2; the PDF has NO appendix, so every constant and the step order come from the authors'
# TensorFlow code (github.com/combra-lab/NALSM). Where the code could not be read the value is a cfg PARAMETER marked INFERRED
# (experiments/nalsm_2021.py).
#
#WHERE: Eq. (1) -> NALSMLIF (no refractory hold: v integrates while refractory); spike / reset / refractory (t_rfr = 2) -> NALSMThreshold;
# Eq. (2), kernel of the LAST presynaptic spike -> NALSMExpSynapse; STDP Eq. (4)-(6) + astrocyte Eq. (7) + the per-step decays (Sec. 2.2.1)
# -> NALSMAstroSTDP; bounds (IL [-3, 3], E [0, 3], I [-3, 0]) -> Run._clamp + clamp_bounds; input neurons (LIF hit by a current of 100 with
# probability = normalised pixel per step) -> USER-SIDE encoder in experiments/nalsm_2021.py.
#
#SPIKE PAIRING (exact): their STDP at step t uses S_t = (input spikes that drive v at t) + (liquid spikes computed from v(t-1)). PhasicNet's
# plasticity step m sees the liquid half one step LATER, so NALSMAstroSTDP keeps the previous step's liquid spikes (S_prev) and pairs
# input m with liquid m-1: exactly their S_t. The last liquid spikes of a sample get no update (theirs neither).
#
#UNITS: ms, 'mV' (the code's dimensionless v with theta 20), W dimensionless.
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import logging

import numpy as np
from synapticindex import touched, elapsed, decayed  

from cfg import SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg
from states import RunContext, NeuronalState, SynapticState
from dynamics import Synapse, Integrator, Threshold, Plasticity
from deckers2022 import steps_of

logger = logging.getLogger(__name__)


   ######################################################################################
##########################   Synapse: exponential kernel of the last spike   ##############
   ######################################################################################

@dataclass(frozen=True, slots=True)
class NALSMSynapseCfg(SynapseCfg):
    """SynapseCfg + Eq. (2) as the code runs it.
    Args:
        tau_syn = tau_u, ms (Table S1 / generate_spatial_network: 1 ms)
        max_since = the cap on 'steps since the last presynaptic spike' (code: max_dt_val 100)
        since_init = the counter's value before any spike and after a reset (code: t_lim, 20 for tau_u = 1; the per-sample reset value is
            INFERRED the same 20). None = that rule.
        cutoff = sparse mode only: presynaptic neurons with since > cutoff are skipped (None = max_since: exact, dense == sparse to the bit;
            20 skips contributions below exp(-20) ~ 2e-9 and is several times faster)
    """
    tau_syn: float = 1.0
    max_since: float = 100.0
    since_init: Optional[float] = None
    cutoff: Optional[float] = None
    cfg_type: str = field(default="NALSMSynapseCfg", kw_only=True)


def _t_lim(tau_steps: float) -> float:
    """The code's t_lim: from 20 in steps of 5 until exp(-t_lim / tau_u) <= 0.01 (GEN_NET_LIF.py)."""
    t = 20.0
    while np.exp(-t / tau_steps) > 0.01 and t < 5000.0:
        t += 5.0
    return t


class NALSMExpSynapse(Synapse):
    """
    I2V_i = sum_j W_ij exp(-Last_j / tau_u) / tau_u   where Last_j = steps since the last spike of presynaptic neuron j (0 at the step whose
    Spikes carry that spike), capped at max_since. Last is module state (reset per sample by reset_transient). One matrix-vector product per step.
    """
    weights_format = "dense"
    writes = ("I2V",)
    requires = ()

    def __init__(self, cfg: NALSMSynapseCfg, sim_cfg: SimulationCfg = None):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("NALSMExpSynapse requires NALSMSynapseCfg and SimulationCfg with dt.")
        if cfg.tau_syn <= 0 or cfg.max_since <= 0:
            raise ValueError("NALSMExpSynapse requires tau_syn > 0 and max_since > 0.")
        self.b = None if cfg.b is None else float(cfg.b)
        self.tau_syn = float(cfg.tau_syn)
        self.max_since = float(cfg.max_since)
        self.since_init = _t_lim(self.tau_syn / float(sim_cfg.dt)) if cfg.since_init is None else float(cfg.since_init)
        self.cutoff = self.max_since if cfg.cutoff is None else float(cfg.cutoff)
        self.dt = float(sim_cfg.dt)
        self._tau_steps = self.tau_syn / self.dt           ##exp(-Last / tau) with Last in steps
        self.Last = None
        self._drive = None
        self._keep = None

    def configure_for_network(self, net) -> None:
        self.Last = np.full(net.n_total, self.since_init, dtype=np.float64)
        self._drive = np.zeros(net.n_total, dtype=np.float64)
        self._keep = np.zeros(net.n_total, dtype=np.float64)

    def _drive_vector(self, S: np.ndarray) -> np.ndarray:
        """code order: ss reset by this step's spikes -> alpha(ss) used -> ss += 1 (capped). Returns the (n_total,) kernel values."""
        last, drive, keep = self.Last, self._drive, self._keep
        np.subtract(1.0, S, out=keep)                       ##1 - S: 0 on the neurons that spiked
        np.multiply(last, keep, out=last)                   ##their ss <- 0
        np.multiply(last, -1.0 / self._tau_steps, out=drive); np.exp(drive, out=drive); drive /= self.tau_syn
        last += 1.0; np.minimum(last, self.max_since, out=last)
        return drive

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        drive = self._drive_vector(neur.Spikes)
        I = np.dot(syn.W, drive.astype(np.float32)).astype(np.float64)
        if self.b is not None:
            I += self.b
        upd.I2V[...] = I


class NALSMExpSynapseSparse(NALSMExpSynapse):
    """The same over the EDGE LIST: W_k * drive[pre_k] summed into post rows for the presynaptic neurons with Last <= cutoff. RETURNS I."""
    weights_format = "sparse"

    def step(self, ctx: RunContext) -> np.ndarray:
        neur, syn = ctx.neur, ctx.syn
        drive = self._drive_vector(neur.Spikes)
        I = np.zeros(syn.weights.n_post, dtype=np.float64)
        active = np.flatnonzero(self.Last <= self.cutoff + 1.0)      ##Last was advanced by 1 after the kernel was read: kernel arg <= cutoff
        if active.size:
            e = syn.weights.index.synIDs_by_pre(active)
            if e.size:
                e.sort()
                I += np.bincount(syn.post[e], weights=syn.W[e] * drive[syn.pre[e]], minlength=syn.weights.n_post)
        if self.b is not None:
            I += self.b
        return I.astype(np.float32)


   ######################################################################################
##########################   Integrator + threshold   #####################################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class NALSMIntegratorCfg(IntegratorCfg):
    """IntegratorCfg: tau_mem 64 ms, v_rest 0 (the leak target), r 1.0; b = the membrane bias of Eq. (2) (code: param_b = 0).
    Args: t_refrac = ms (2): used by the THRESHOLD only -- the code integrates v while refractory (update_neuron_states_wo_R)."""
    b: float = 0.0
    t_refrac: float = 2.0
    cfg_type: str = field(default="NALSMIntegratorCfg", kw_only=True)


class NALSMLIF(Integrator):
    """Eq. (1) as coded: v <- v + I + b - v / tau_v per 1 ms step  ->  dV = r I2V + b - (V - v_rest) dt / tau_mem. No refractory gate."""
    writes = ("dV",)
    requires = ()

    def __init__(self, cfg: NALSMIntegratorCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("NALSMLIF requires NALSMIntegratorCfg and SimulationCfg with dt.")
        if cfg.tau_mem <= 0:
            raise ValueError("NALSMLIF requires tau_mem > 0.")
        self.dt = float(sim_cfg.dt)
        self.v_rest = np.float32(cfg.v_rest)
        self.tau_mem = float(cfg.tau_mem)
        self.r = float(cfg.r)
        self.b = float(cfg.b)
        self.t_refrac = float(cfg.t_refrac)

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        dV = neur.V.astype(np.float64)
        dV -= float(self.v_rest)
        dV *= -self.dt / self.tau_mem
        dV += upd.I2V.astype(np.float64) * self.r
        dV += self.b
        upd.dV[...] = dV


@dataclass(frozen=True, slots=True)
class NALSMThresholdCfg(ThresholdCfg):
    """ThresholdCfg: v_reset 0, v_th 20, t_refrac 2 ms. tau_th / adpt_th unused (fixed threshold)."""
    cfg_type: str = field(default="NALSMThresholdCfg", kw_only=True)


class NALSMThreshold(Threshold):
    """spike iff V >= v_th and Since > steps(t_refrac): the code's counter (set to t_rfr on the spike, -1 per step, blocks while > 0) blocks
    exactly t_rfr steps after the spike step. Reset to v_reset by the orchestra."""
    writes = ("rec_spikes",)
    requires = ("Since",)

    def __init__(self, cfg: NALSMThresholdCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("NALSMThreshold requires NALSMThresholdCfg and SimulationCfg with dt.")
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
        np.greater(neur.Since, self._refrac_steps, out=self._gate)
        np.greater_equal(neur.V, self.v_th, out=upd.rec_spikes)
        np.logical_and(upd.rec_spikes, self._gate, out=upd.rec_spikes)


   ######################################################################################
##########################   Plasticity: astrocyte-modulated STDP   #######################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class NALSMPlasticityCfg(PlasticityCfg):
    """PlasticityCfg mapped onto Eq. (4)-(7). Parent fields: tau_pretrace = tau_posttrace = 10 ms, A_ltp = A+ (0.15; also the initial astrocyte
    value and bias), A_ltd unused (A- IS the astrocyte output), eta_exc = eta_inh = 1, w_min = -3 / w_max = 3 (Sec. 2.1), freeze = () (the
    input projection IS plastic, within [w_min, w_max]: input_bounds).
    Args:
        trace_inc = a+ = a- (0.1)
        astro_tau = tau_astro, ms (code: 100)
        astro_w = w_astro (0.015 at N = 1000, 0.0075 at N = 8000; README: accuracy peaks near 0.01)
        astro_bias_offset = b_astro = A+ (1 + offset) (code: 0.0)
        decay = the per-step factor on A+, A, w_astro, b_astro during a counting presentation (0.99; Sec. 2.2.1)
        astro = False: plain LSM+STDP, A- = A+ fixed (the paper's LSM+STDP model)
        input_bounds = (lo, hi) of the input projections (None = the per-sign clamp as for the liquid)
    """
    tau_pretrace: float = 10.0
    tau_posttrace: float = 10.0
    A_ltp: float = 0.15
    A_ltd: float = 0.15
    eta_exc: float = 1.0
    eta_inh: Optional[float] = 1.0
    w_min: Optional[float] = -3.0
    w_max: Optional[float] = 3.0
    trace_inc: float = 0.1
    astro_tau: float = 100.0
    astro_w: float = 0.015
    astro_bias_offset: float = 0.0
    decay: float = 0.99
    astro: bool = True
    input_bounds: Optional[Tuple[float, float]] = (-3.0, 3.0)
    cfg_type: str = field(default="NALSMPlasticityCfg", kw_only=True)


class NALSMAstroSTDP(Plasticity):
    """
    dW_ij = (S_i x_j) (Lr c^EI_j - A c^I_j) + (y_i S_j) (-A c^EI_j + Lr c^I_j)      Eq. (4) with A- = A (the astrocyte), per column class
    x, y <- m (a S + x), m = exp(-dt / tau)                                           Eq. (5)-(6), AFTER the increments
    A <- A + (Aw (n liquid spikes - n input spikes) + Ab - A) / tau_astro             Eq. (7), after the traces
    Lr, A, Aw, Ab <- decay * (...)   while decay_active                              Sec. 2.2.1 (the code: every step of a counting presentation)
    S here = THEIR S_t: input spikes of this step + liquid spikes of the PREVIOUS step (S_prev, module state). Frozen projections get dW = 0;
    the orchestra clamps (per sign, the input projections within input_bounds via clamp_bounds).
    """
    weights_format = "dense"
    writes = ("dW",)
    requires = ("pre_spike", "post_spike")

    def __init__(self, cfg: NALSMPlasticityCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("NALSMAstroSTDP requires NALSMPlasticityCfg and SimulationCfg with dt.")
        if cfg.tau_pretrace <= 0 or cfg.tau_posttrace <= 0 or cfg.astro_tau <= 0 or not (0.0 < cfg.decay <= 1.0):
            raise ValueError("NALSMAstroSTDP requires tau_pretrace, tau_posttrace, astro_tau > 0 and 0 < decay <= 1.")
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
        self.trace_inc = float(cfg.trace_inc)
        self.astro_tau = float(cfg.astro_tau)
        self.astro_w = float(cfg.astro_w)
        self.astro_bias_offset = float(cfg.astro_bias_offset)
        self.decay = float(cfg.decay)
        self.astro = bool(cfg.astro)
        self.input_bounds = None if cfg.input_bounds is None else (float(cfg.input_bounds[0]), float(cfg.input_bounds[1]))
        self._m_pre = float(np.exp(-self.dt / self.tau_pretrace))
        self._m_post = float(np.exp(-self.dt / self.tau_posttrace))
        self._decay_active = True
        ##module-owned state (Capitalised -> state_dict, reset per sample by reset_transient)
        self.X_pre = None; self.Y_post = None; self.S_prev = None
        self.Lr = None; self.A = None; self.Aw = None; self.Ab = None
        self._col_ei = None; self._col_i = None; self._n_input = 0
        self._S_pre = None; self._tmp = None
        self.clamp_bounds: Dict[str, Tuple[float, float]] = {}    ##read once by Run._plan_clamp

    @property
    def decay_active(self) -> bool:
        """The per-step decay of A+, A, w_astro, b_astro: on in the counting pass, OFF in the initialization pass. Set by the protocol."""
        return self._decay_active

    @decay_active.setter
    def decay_active(self, on: bool) -> None:
        self._decay_active = bool(on)

    def configure_for_network(self, net) -> None:
        self.conf_traces(net)                                           ##(true event-based): per-neuron here, per-synapse in the event mode
        self.S_prev = np.zeros(net.n_post, dtype=np.float32)
        a0 = np.float32(self.A_ltp)
        self.Lr = np.full(1, a0, dtype=np.float32)
        self.A = np.full(1, a0 if self.astro else self.A_ltd, dtype=np.float32)
        self.Aw = np.full(1, self.astro_w, dtype=np.float32)
        self.Ab = np.full(1, a0 * (1.0 + self.astro_bias_offset), dtype=np.float32)
        col = np.ones(net.n_total, dtype=np.float32); col[net.inh_slice] = 0.0
        self._col_ei = col                                                  ##exc AND input columns
        self._col_i = 1.0 - col
        self._n_input = net.n_input
        self._S_pre = np.zeros(net.n_total, dtype=np.float32)
        if self.input_bounds is not None:
            self.clamp_bounds = {name: self.input_bounds for name, spec in net.synapsespecs.items()
                                 if net.neuronspecs[spec.pre_neuron].input and name not in self.freeze}
        self.conf_scratch(net)

    def conf_traces(self, net) -> None:
        self.X_pre = np.zeros(net.n_total, dtype=np.float32); self.Y_post = np.zeros(net.n_post, dtype=np.float32)

    def conf_scratch(self, net) -> None:
        self._tmp = np.zeros((net.n_post, net.n_total), dtype=np.float32)

    # -------------------------------------------------------------------------
    def _their_spikes(self, neur: NeuronalState) -> Tuple[np.ndarray, np.ndarray]:
        """S_t = input spikes of this step + liquid spikes of the previous step (pre, n_total) and the latter alone (post, n_post)."""
        S_pre = self._S_pre
        S_pre[:self._n_input] = neur.pre_spike[:self._n_input]
        S_pre[self._n_input:] = self.S_prev
        return S_pre, self.S_prev

    def _coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """(n_total,) learning rates of the two terms: c1 = Lr col_EI - A col_I (on S_post x_pre), c2 = -A col_EI + Lr col_I (on y_post S_pre)."""
        lr, A = float(self.Lr[0]), float(self.A[0])
        c1 = self._col_ei * (lr * self.eta_exc) - self._col_i * (A * self.eta_inh)
        c2 = self._col_i * (lr * self.eta_inh) - self._col_ei * (A * self.eta_exc)
        return c1, c2

    def _after_dw(self, neur: NeuronalState, S_pre: np.ndarray, S_post: np.ndarray, traces: bool = True) -> None:
        """traces (Eq. 5-6), astrocyte (Eq. 7), decays, then remember this step's liquid spikes for the next step's pairing.
        traces=False (event mode): the event step updated its per-synapse traces; the astrocyte and the decays stay clocked."""
        a = self.trace_inc
        if traces:
            X, Y = self.X_pre, self.Y_post
            X += a * S_pre; X *= self._m_pre
            Y += a * S_post; Y *= self._m_post
        if self.astro:
            n_liq = float(S_post.sum()); n_inp = float(S_pre[:self._n_input].sum())
            self.A += (self.Aw * (n_liq - n_inp) + self.Ab - self.A) / self.astro_tau
        if self._decay_active:
            self.Lr *= self.decay; self.Aw *= self.decay; self.Ab *= self.decay
            if self.astro:
                self.A *= self.decay
        self.S_prev[...] = neur.post_spike

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        S_pre, S_post = self._their_spikes(neur)
        c1, c2 = self._coefficients()
        dW, tmp = upd.dW, self._tmp
        np.multiply(S_post[:, None], (self.X_pre * c1)[None, :], out=dW)
        np.multiply(self.Y_post[:, None], (S_pre * c2)[None, :], out=tmp)
        dW += tmp
        for proj in syn.freeze_slices:
            dW[proj.post, proj.pre] = 0.0
        np.multiply(dW, syn.syn_mask, out=dW)
        self._after_dw(neur, S_pre, S_post)


class NALSMAstroSTDPSparse(NALSMAstroSTDP):
    """The same over the edge list, evaluated only at edges with a spiking endpoint (update_on: hybrid_event) or all edges ('timestep'). RETURNS dw.
    update_on: event : the traces live PER SYNAPSE (X_e, Y_e, T_e) and are decayed lazily at the synapses a spike
    touches ; X_pre / Y_post are None in that mode. The astrocyte (A, Lr, Aw, Ab) stays clocked: it is global, not per synapse.
    
    in BOTH event modes step() returns the pair (k, dw_k) -- the touched plastic edges and their dw -- not a zero-filled (n_synapses,)
    array; Run.step_sparse commits W[k] += dw_k and clips W[k] only. "timestep" still returns the full dw."""
    weights_format = "sparse"

    def __init__(self, cfg: NALSMPlasticityCfg, sim_cfg: SimulationCfg):
        super().__init__(cfg, sim_cfg)
        self.update_on = sim_cfg.update_on

    def conf_traces(self, net) -> None:
        if self.update_on != "event":
            return super().conf_traces(net)
        ##per-synapse traces + the step of each edge's last event; no per-neuron traces
        self.X_pre = None; self.Y_post = None
        self.X_e = np.zeros(net.n_synapses, dtype=np.float32); self.Y_e = np.zeros(net.n_synapses, dtype=np.float32)
        self.T_e = np.zeros(net.n_synapses, dtype=np.int64)

    def conf_scratch(self, net) -> None:
        self._tmp = None

    def _rule(self, S_pre: np.ndarray, S_post: np.ndarray, pre: np.ndarray, x: np.ndarray, y: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """The expression per edge, given the edge's spikes and traces. NOTE split out of _dw_at so both trace layouts share it."""
        dw = S_post * (x * c1[pre])
        dw += y * (S_pre * c2[pre])
        return dw

    def _dw_at(self, syn: SynapticState, S_pre: np.ndarray, S_post: np.ndarray, c1: np.ndarray, c2: np.ndarray, k: np.ndarray | None) -> np.ndarray:
        """Per-neuron traces (update_on timestep / hybrid_event)."""
        pre, post = (syn.pre, syn.post) if k is None else (syn.pre[k], syn.post[k])
        return self._rule(S_pre[pre], S_post[post], pre, self.X_pre[pre], self.Y_post[post], c1, c2)

    def _step_event(self, ctx: RunContext, S_pre: np.ndarray, S_post: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> tuple:
        """Event mode: the rule at the plastic edges a spike touches, on per-synapse traces brought up to date lazily. Dense order: the rule
        reads the traces as they stood at the START of the step, then Eq. 5-6's update (+ a S, then decay).
        NOTE: RETURNS (k, dw_k), the touched edges and their dw (was a zero-filled (n_synapses,) array with dw at k).
        NOTE: t = sim.train_step, not current_runstep -- eval steps are not elapsed time for the traces (synapticindex docstring)."""
        syn, t = ctx.syn, ctx.sim.train_step
        k = touched(syn, S_post > 0, S_pre > 0)
        if not k.size:
            return k, np.zeros(0, dtype=np.float32)
        pre, post = syn.pre[k], syn.post[k]
        sp, ss = S_pre[pre], S_post[post]
        n = elapsed(self.T_e, k, t)
        x, y = decayed(self.X_e, k, self._m_pre, n), decayed(self.Y_e, k, self._m_post, n)
        dw = self._rule(sp, ss, pre, x, y, c1, c2)
        self.X_e[k] = (x + self.trace_inc * sp) * self._m_pre
        self.Y_e[k] = (y + self.trace_inc * ss) * self._m_post
        self.T_e[k] = t
        return k, dw                                                    ##the pair; k is already restricted to plastic edges

    def step(self, ctx: RunContext):
        neur, syn = ctx.neur, ctx.syn
        S_pre, S_post = self._their_spikes(neur)
        c1, c2 = self._coefficients()
        if self.update_on == "event":                                   #(true event-based)
            dw = self._step_event(ctx, S_pre, S_post, c1, c2)           
            self._after_dw(neur, S_pre, S_post, traces=False)
            return dw
        if self.update_on == "hybrid_event":                            
            k = touched(syn, S_post > 0, S_pre > 0)                     # the PLASTIC touched edges (spiking_syns + plastic_mask) and the pair (k, dw_k) returned; was a zero-filled (n_synapses,) dw, masked below
            dw = self._dw_at(syn, S_pre, S_post, c1, c2, k)
            self._after_dw(neur, S_pre, S_post)
            return k, dw
        dw = self._dw_at(syn, S_pre, S_post, c1, c2, None)
        if syn.plastic_mask is not None:
            np.multiply(dw, syn.plastic_mask, out=dw)
        self._after_dw(neur, S_pre, S_post)
        return dw
