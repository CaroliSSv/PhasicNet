#deckers2022 - Deckers, Tsang, Van Leekwijck & Latre (2022), "Extended liquid state machines for speech recognition", Front. Neurosci. 16:1023470
# -- the Extended LSM (ELSM) as PhasicNet runtime modules + their cfgs. The network (3-D grid, gaussian-kernel wiring, input projection) and
# the protocol (fixed liquid -> spike counts -> logistic regression) are in experiments/deckers_2022.py.
#
#SOURCE: the paper only (no published code); Brian 2 scheduling details are Brian 2's documented defaults. Values the paper leaves open are
# cfg PARAMETERS, marked "INFERRED" in experiments/deckers_2022.py.
#
#LINE BY LINE                              Deckers 2022 (paper)                                            PhasicNet (this file)
#   neuron (both groups)                    tau dv/dt = (E_rest - v) + g_e (E_exc - v) + g_i (E_inh - v)      ELSMConductanceSynapse: I2V = Ge (E_exc - V) + Gi (E_inh - V)
#                                           Euler (Brian 2 falls back to euler for the g*v product)          ELSMConductanceLIF: dV = (E_rest - V + I2V) dt / tau_row
#   E_rest = v_reset = 13.5 mV, v_th = 15   Table 1                                                           cfg: integrator.v_rest, threshold.v_reset / v_th
#   tau_m = 30 ms (LSM) | U[5, 55] (ELSM)   Table 1 / Sec. 2.3 "heterogeneous membrane time constants"        integrator.tau_mem (all rows) | tau_range -> one draw per neuron (keyed rng)
#   tau_ge = 3 ms, tau_gi = 10 ms           Table 1                                                           synapse.tau_ge / tau_gi; dg/dt = -g/tau (Euler), g += w on arrival
#   E_exc, E_inh                            Table 1 prints 0 mV for BOTH (contradicts E_rest 13.5 < v_th 15)  synapse.e_exc / e_inh: PARAMETERS, no default claimed
#   refractory 3 ms exc / 2 ms inh          Table 1                                                           integrator.t_refrac / t_refrac_inh (v held) + threshold (spike blocked)
#   v(0) ~ U[13.5, 15]                      Sec. 2.1                                                          integrator.v_init -> ELSMConductanceLIF.init_membrane(ctx, key), per sample
#   SFA (ELSM): v_th += 1 mV per spike,     Sec. 2.3: adaptive threshold, exc neurons only, decays back to    SFAThreshold: dV_th = (v_th - V_th) dt / tau_sfa_row + inc_row * spikes, inc_row = adpt_th on exc rows,
#   tau_sfa = 550 ms | U[50, 1050] ms       15 mV with tau_sfa (fixed or heterogeneous)                       0 on inh rows (sfa_exc_only); tau_th fixed | sfa_range -> one draw per neuron (keyed rng)
#   delays: input 1.0, EE 1.5, EI/IE/II 0.8  Table 2 (ms)                                                     synapse.delays = {synapse group: ms} -> int(round(d / dt)) steps, Brian 2's rounding; per-group ring slots
#   no learning (fixed liquid)              Sec. 2.4: liquid weights fixed; readout trained on spike counts    NoPlasticity + freeze=("all",); the protocol runs every pass in mode "eval"
#   3-D grid, P = C exp(-(D/lambda)^2)      Table 2 (Maass 2002 rule), lambda = 2, C_EE/EI/IE/II = .3/.2/.4/.1 builder: set_positions(share=) + p_connection={"kernel": "gaussian", ...}
#
#SPIKE TIMING: at step m, neur.Spikes holds this step's INPUT spikes and the recurrent spikes of step m-1. Brian 2 delivers a spike of step s
# with delay d before the update of step s + 1 + d, so the ring slot offset is k = d + (1 if the pre group is an input group); R = max(k) + 1.
#
#UNITS: ms, mV; conductances dimensionless. dt is NOT in the paper (cfg: SimulationCfg.dt).
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import logging

import numpy as np

from dynamicscfg import SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg
from states import RunContext
from dynamics import Synapse, Integrator, Threshold, Plasticity
from diehlcook import _row_vector, _inh_rows          ##per-ROW constants (inh rows vs the rest), shared with D&C

logger = logging.getLogger(__name__)


def steps_of(ms: float, dt_ms: float) -> int:
    """ms -> whole steps with Brian 2's rounding (np.round: half to even)."""
    return int(np.round(float(ms) / float(dt_ms)))


   ######################################################################################
##########################   Synapse: conductances ge, gi, per-group delays   ##############
   ######################################################################################

@dataclass(frozen=True, slots=True)
class ELSMSynapseCfg(SynapseCfg):
    """SynapseCfg + the conductance model of Deckers 2022 (Table 1).
    Args:
        tau_ge, tau_gi = ms, decay of the excitatory / inhibitory conductance (3 / 10)
        e_exc, e_inh = mV, reversal potentials. REQUIRED (None raises): the paper's 0 mV for both cannot be right.
        delays = ms per SYNAPSE GROUP, e.g. {"in_exc": 1.0, "exc_exc": 1.5}; unlisted groups have delay 0. Rounded to steps.
    """
    tau_ge: float = 3.0
    tau_gi: float = 10.0
    e_exc: Optional[float] = None
    e_inh: Optional[float] = None
    delays: Optional[Dict[str, float]] = None


class ELSMConductanceSynapse(Synapse):
    """
    Deckers 2022:  dg_e/dt = -g_e/tau_ge ; dg_i/dt = -g_i/tau_gi ;  arrival: g_e += w (exc pre) | g_i += w (inh pre)
                   I = g_e (E_exc - v) + g_i (E_inh - v)                                              (Euler)

    Ge/Gi are per-neuron module state (checkpointed, reset by Run.reset_transient). Per step:
        1. ring slot due now -> Ge/Gi, slot cleared
        2. this step's spikes per group: k = 0 -> Ge/Gi now, k >= 1 -> ring slot (cur + k) % R
        3. I from V at the start of the step -> I2V
        4. Euler decay of Ge/Gi
    Inhibitory weights are stored NEGATIVE (Dale), so gi += -(W . S).
    """
    weights_format = "dense"
    writes = ("I2V",)
    requires = ()

    def __init__(self, cfg: ELSMSynapseCfg, sim_cfg: SimulationCfg = None):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("ELSMConductanceSynapse requires ELSMSynapseCfg and SimulationCfg with dt.")
        if cfg.tau_ge <= 0 or cfg.tau_gi <= 0:
            raise ValueError("ELSMConductanceSynapse requires tau_ge > 0 and tau_gi > 0.")
        if cfg.e_exc is None or cfg.e_inh is None:
            raise ValueError("ELSMConductanceSynapse requires e_exc and e_inh (the paper's Table 1 is contradictory here: choose and pass them).")
        self.b = None if cfg.b is None else float(cfg.b)
        self.tau_ge = float(cfg.tau_ge)                                     ##stored under cfg names (naming contract)
        self.tau_gi = float(cfg.tau_gi)
        self.e_exc = float(cfg.e_exc)
        self.e_inh = float(cfg.e_inh)
        self.delays = {} if cfg.delays is None else {str(k): float(v) for k, v in dict(cfg.delays).items()}
        self.dt = float(sim_cfg.dt)
        self.Ge = None
        self.Gi = None
        self.Buf_e = None           ##(R, n_post) delayed excitatory arrivals, one ring for every group (slot = step offset)
        self.Buf_i = None           ##(R, n_post) delayed inhibitory arrivals (stored POSITIVE)
        self.Cur = None             ##(1,) ring index of THIS step's slot
        self._R = 1
        self._projs = None          ##[(name, post_rows, pre_cols, inhibitory?, k)] in synapseslices order

    def configure_for_network(self, net) -> None:
        unknown = set(self.delays) - set(net.synapseslices)
        if unknown:
            raise RuntimeError(f"ELSMConductanceSynapse: delays name synapse groups {sorted(unknown)} that are not in this network {list(net.synapseslices)}.")
        self.Ge = np.zeros(net.n_post, dtype=np.float64)
        self.Gi = np.zeros(net.n_post, dtype=np.float64)
        self._projs = []
        for name, proj in net.synapseslices.items():
            pre_spec = net.neuronspecs[net.synapsespecs[name].pre_neuron]
            d = steps_of(self.delays.get(name, 0.0), self.dt)
            if d < 0:
                raise RuntimeError(f"ELSMConductanceSynapse: negative delay for '{name}'.")
            k = d + (1 if pre_spec.input else 0)                             ##see SPIKE TIMING in the header
            self._projs.append((name, proj.post, proj.pre, bool(pre_spec.inh), k))
        self._R = max((k for *_, k in self._projs), default=0) + 1
        self.Buf_e = np.zeros((self._R, net.n_post), dtype=np.float64)
        self.Buf_i = np.zeros((self._R, net.n_post), dtype=np.float64)
        self.Cur = np.zeros(1, dtype=np.int64)
        logger.info("ELSMConductanceSynapse: %d projections, ring of %d slots: %s", len(self._projs),
                    self._R, {n: k for n, _, _, _, k in self._projs})

    def _contribution(self, ctx: RunContext, S: np.ndarray, name: str, post: slice, pre: slice) -> Optional[np.ndarray]:
        """(n_post_group,) float64 sum of weights of the pre spikes of this step onto each post neuron of the group; None if no pre spike."""
        s = S[pre]
        if not s.any():
            return None
        return np.dot(ctx.syn.W[post, pre], s.astype(np.float32)).astype(np.float64)

    def _step(self, ctx: RunContext) -> np.ndarray:
        """The shared body: returns I (n_post,) float64."""
        neur = ctx.neur
        S = neur.Spikes
        Ge, Gi = self.Ge, self.Gi
        cur = int(self.Cur[0])
        ##1. arrivals due this step
        Ge += self.Buf_e[cur]; Gi += self.Buf_i[cur]
        self.Buf_e[cur] = 0.0; self.Buf_i[cur] = 0.0
        ##2. this step's spikes, per group
        for name, post, pre, inhibitory, k in self._projs:
            c = self._contribution(ctx, S, name, post, pre)
            if c is None:
                continue
            if inhibitory:
                target = Gi if k == 0 else self.Buf_i[(cur + k) % self._R]
                target[post] -= c                                           ##Dale: stored negative -> gi += |w|
            else:
                target = Ge if k == 0 else self.Buf_e[(cur + k) % self._R]
                target[post] += c
        ##3. current at the V of the start of the step
        V = neur.V.astype(np.float64)
        I = Ge * (self.e_exc - V) + Gi * (self.e_inh - V)
        if self.b is not None:
            I += self.b
        ##4. decay (Euler)
        Ge += (-Ge / self.tau_ge) * self.dt
        Gi += (-Gi / self.tau_gi) * self.dt
        self.Cur[0] = (cur + 1) % self._R
        return I

    def step(self, ctx: RunContext) -> None:
        ctx.upd.I2V[...] = self._step(ctx)


class ELSMConductanceSynapseSparse(ELSMConductanceSynapse):
    """The same model over the edge list (weights_format="sparse"): per group, the edges whose pre neuron spiked are summed into their post
    rows with np.bincount. Returns I. The mode for Deckers' 2000-neuron liquid."""
    weights_format = "sparse"

    def __init__(self, cfg: ELSMSynapseCfg, sim_cfg: SimulationCfg = None):
        super().__init__(cfg, sim_cfg)
        self._edges = None          ##{group: int64 edge ids}, resolved on the first step

    def _resolve_edges(self, syn) -> None:
        pre, post = syn.pre, syn.post
        self._edges = {}
        for name, prows, pcols, _, _ in self._projs:
            inside = (post >= prows.start) & (post < prows.stop) & (pre >= pcols.start) & (pre < pcols.stop)
            self._edges[name] = np.flatnonzero(inside)

    def _contribution(self, ctx: RunContext, S: np.ndarray, name: str, post: slice, pre: slice) -> Optional[np.ndarray]:
        syn = ctx.syn
        if self._edges is None:
            self._resolve_edges(syn)
        e = self._edges[name]
        active = e[S[syn.pre[e]]]
        if not active.size:
            return None
        return np.bincount(syn.post[active] - post.start, weights=syn.W[active], minlength=post.stop - post.start)

    def step(self, ctx: RunContext) -> np.ndarray:
        return self._step(ctx).astype(np.float32, copy=False)


   ######################################################################################
##########################   Integrator: Euler conductance LIF, heterogeneous tau   ########
   ######################################################################################

@dataclass(frozen=True, slots=True)
class ELSMIntegratorCfg(IntegratorCfg):
    """IntegratorCfg (tau_mem, v_rest = E_rest, r = 1.0: I2V is already in mV) + Deckers' heterogeneity and refractoriness.
    Args:
        tau_range = (lo, hi) ms: heterogeneous tau_mem, ONE keyed draw per neuron. None = tau_mem everywhere (LSM baseline: 30 ms).
        t_refrac, t_refrac_inh = ms, refractory periods exc / inh (3 / 2): v is HELD (dV = 0) while Since < steps_of(t_refrac).
        v_init = (lo, hi) mV, initial potential U[13.5, 15] (Sec. 2.1), drawn by init_membrane(ctx, key). None = v_rest.
    """
    tau_range: Optional[Tuple[float, float]] = None
    t_refrac: float = 3.0
    t_refrac_inh: float = 2.0
    v_init: Optional[Tuple[float, float]] = (13.5, 15.0)


class ELSMConductanceLIF(Integrator):
    """
    Deckers 2022:  tau dv/dt = (E_rest - v) + I,   I from the synapse (I2V)     Euler: dV = ((v_rest - V) + I) dt / tau_row
    Refractory rows (Since < refrac_steps) keep dV = 0. tau_row: tau_mem everywhere, or U[tau_range] per neuron (drawn on the first step).
    """
    writes = ("dV",)
    requires = ("Since",)

    def __init__(self, cfg: ELSMIntegratorCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("ELSMConductanceLIF requires ELSMIntegratorCfg and SimulationCfg with dt.")
        if cfg.tau_mem <= 0:
            raise ValueError("ELSMConductanceLIF requires tau_mem > 0.")
        if cfg.tau_range is not None and not (0 < cfg.tau_range[0] <= cfg.tau_range[1]):
            raise ValueError(f"ELSMConductanceLIF: tau_range must be (lo, hi) with 0 < lo <= hi, got {cfg.tau_range!r}.")
        if cfg.t_refrac < 0 or cfg.t_refrac_inh < 0:
            raise ValueError("ELSMConductanceLIF requires t_refrac >= 0 and t_refrac_inh >= 0.")
        self.dt = float(sim_cfg.dt)
        self.v_rest = np.float32(cfg.v_rest)            ##the orchestra's reset_neuronal_state reads this scalar
        self.tau_mem = float(cfg.tau_mem)
        self.r = float(cfg.r)
        self.tau_range = None if cfg.tau_range is None else (float(cfg.tau_range[0]), float(cfg.tau_range[1]))
        self.t_refrac = float(cfg.t_refrac)
        self.t_refrac_inh = float(cfg.t_refrac_inh)
        self.v_init = None if cfg.v_init is None else (float(cfg.v_init[0]), float(cfg.v_init[1]))
        self._tau_row = None        ##(n_post,) float64, per-neuron tau (drawn once if tau_range)
        self._dt_over_tau = None
        self._refrac_row = None     ##(n_post,) int64 steps: Since >= this -> not refractory
        self._gate = None           ##scratch (n_post,) float64: 1 integrates, 0 held
        self._drawn = False
        self._n_input_rows = 0

    def configure_for_network(self, net) -> None:
        self._tau_row = np.full(net.n_post, self.tau_mem, dtype=np.float64)
        self._dt_over_tau = self.dt / self._tau_row
        self._refrac_row = _row_vector(net, steps_of(self.t_refrac, self.dt), steps_of(self.t_refrac_inh, self.dt), dtype=np.int64)
        self._gate = np.zeros(net.n_post, dtype=np.float64)
        self._drawn = self.tau_range is None
        self._n_input_rows = net.n_post - net.n_rec     ##> 0 only without post_offset: input rows exist in V, they receive nothing

    def _draw(self, ctx: RunContext) -> None:
        lo, hi = self.tau_range
        self._tau_row[...] = ctx.rng.rng("integrator", "tau_mem").uniform(lo, hi, self._tau_row.size)
        self._dt_over_tau = self.dt / self._tau_row
        self._drawn = True
        logger.info("ELSMConductanceLIF: tau_mem drawn U[%g, %g] ms per neuron (mean %.1f).", lo, hi, self._tau_row.mean())

    @property
    def tau_row(self) -> np.ndarray:
        """The per-neuron membrane time constants in use (after the first step, or tau_mem everywhere)."""
        return self._tau_row

    def init_membrane(self, ctx: RunContext, key: str = "0") -> None:
        """Sec. 2.1: v(0) ~ U[v_init] per neuron, keyed (same key -> same draw). Input rows stay at v_rest."""
        V = ctx.neur.V
        if self.v_init is None:
            V[...] = self.v_rest
            return
        lo, hi = self.v_init
        V[...] = ctx.rng.rng("integrator", f"v_init:{key}").uniform(lo, hi, V.size).astype(np.float32)
        if self._n_input_rows:
            V[:self._n_input_rows] = self.v_rest

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        if not self._drawn:
            self._draw(ctx)
        V = neur.V.astype(np.float64)
        dV = (float(self.v_rest) - V)
        dV += upd.I2V.astype(np.float64) * self.r
        dV *= self._dt_over_tau
        np.greater_equal(neur.Since, self._refrac_row, out=self._gate)      ##refractory gate: bool -> 0/1 float
        dV *= self._gate
        upd.dV[...] = dV


   ######################################################################################
##########################   Threshold: spike-frequency adaptation (exc rows)   ###########
   ######################################################################################

@dataclass(frozen=True, slots=True)
class ELSMThresholdCfg(ThresholdCfg):
    """ThresholdCfg (v_reset = 13.5, v_th = 15, tau_th = tau_sfa, adpt_th = SFA increment, t_refrac = exc refractory) + Deckers' rest.
    Args:
        adpt_th = mV added to a neuron's threshold per spike (1 mV). None or 0 = no SFA (LSM baseline).
        tau_th = ms, decay of the threshold back to v_th (550). None = no decay.
        sfa_range = (lo, hi) ms: heterogeneous tau_sfa, ONE keyed draw per neuron (paper: U[50, 1050]). None = tau_th everywhere.
        sfa_exc_only = adaptation on excitatory neurons only (Sec. 2.3).
        t_refrac_inh = ms, inh refractory period (2): spikes blocked while Since < steps.
    """
    sfa_range: Optional[Tuple[float, float]] = None
    sfa_exc_only: bool = True
    t_refrac_inh: float = 2.0


class SFAThreshold(Threshold):
    """
    Deckers 2022 (ELSM):  spike iff v > v_th_i and not refractory ;  on spike: v_th_i += adpt_th (exc) ;  d v_th_i/dt = (v_th - v_th_i) / tau_sfa_i
    The adaptive threshold is NeuronalState.V_th: the orchestra adds dV_th and resets V_th to v_th per sample.
    Brian 2 order: the test uses V_th + decay, the increment comes after it. Both go out in ONE dV_th = decay + inc_row * spikes.
    """
    writes = ("rec_spikes", "dV_th")
    requires = ("V_th", "dV_th", "Since")

    def __init__(self, cfg: ELSMThresholdCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("SFAThreshold requires ELSMThresholdCfg and SimulationCfg with dt.")
        if cfg.tau_th is not None and cfg.tau_th <= 0:
            raise ValueError("SFAThreshold requires tau_th > 0 (or None = no decay).")
        if cfg.sfa_range is not None and not (0 < cfg.sfa_range[0] <= cfg.sfa_range[1]):
            raise ValueError(f"SFAThreshold: sfa_range must be (lo, hi) with 0 < lo <= hi, got {cfg.sfa_range!r}.")
        self.dt = float(sim_cfg.dt)
        self.v_th = float(cfg.v_th)                     ##the orchestra's reset_neuronal_state writes this scalar into V_th
        self.v_reset = np.float32(cfg.v_reset)
        self.tau_th = None if cfg.tau_th is None else float(cfg.tau_th)
        self.adpt_th = 0.0 if cfg.adpt_th is None else float(cfg.adpt_th)
        self.t_refrac = 0.0 if cfg.t_refrac is None else float(cfg.t_refrac)
        self.t_refrac_inh = float(cfg.t_refrac_inh)
        self.sfa_range = None if cfg.sfa_range is None else (float(cfg.sfa_range[0]), float(cfg.sfa_range[1]))
        self.sfa_exc_only = bool(cfg.sfa_exc_only)
        self._inc_row = None        ##(n_post,) float64: adpt_th on adapting rows, 0 elsewhere
        self._tau_row = None        ##(n_post,) float64 tau_sfa per neuron
        self._dt_over_tau = None
        self._refrac_row = None     ##(n_post,) int64
        self._gate = None           ##scratch bool
        self._drawn = False

    def configure_for_network(self, net) -> None:
        inh_value = 0.0 if self.sfa_exc_only else self.adpt_th
        self._inc_row = _row_vector(net, self.adpt_th, inh_value, dtype=np.float64)
        if net.n_post > net.n_rec:                      ##input rows (no post_offset) never adapt
            self._inc_row[:net.n_post - net.n_rec] = 0.0
        tau = np.inf if self.tau_th is None else self.tau_th
        self._tau_row = np.full(net.n_post, tau, dtype=np.float64)
        self._dt_over_tau = self.dt / self._tau_row
        self._refrac_row = _row_vector(net, steps_of(self.t_refrac, self.dt), steps_of(self.t_refrac_inh, self.dt), dtype=np.int64)
        self._gate = np.zeros(net.n_post, dtype=bool)
        self._drawn = self.sfa_range is None

    def _draw(self, ctx: RunContext) -> None:
        lo, hi = self.sfa_range
        self._tau_row[...] = ctx.rng.rng("threshold", "tau_th").uniform(lo, hi, self._tau_row.size)
        self._dt_over_tau = self.dt / self._tau_row
        self._drawn = True
        logger.info("SFAThreshold: tau_sfa drawn U[%g, %g] ms per neuron (mean %.0f).", lo, hi, self._tau_row.mean())

    @property
    def tau_row(self) -> np.ndarray:
        return self._tau_row

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        if not self._drawn:
            self._draw(ctx)
        V_th = neur.V_th.astype(np.float64)
        decay = (self.v_th - V_th) * self._dt_over_tau                     ##Euler, the state update's part (0 when tau is inf)
        thr = V_th + decay                                                  ##Brian 2 tests the UPDATED threshold
        np.greater_equal(neur.Since, self._refrac_row, out=self._gate)
        spikes = (neur.V.astype(np.float64) > thr) & self._gate
        upd.rec_spikes[...] = spikes
        decay += self._inc_row * spikes                                     ##the reset's part: v_th += adpt_th on adapting rows that spiked
        upd.dV_th[...] = decay
        #### MEMBRANE V RESET NOT APPLIED. Left to the orchestrator (scalar v_reset).


   ######################################################################################
##########################   Plasticity: none (fixed liquid)   ############################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class NoPlasticityCfg(PlasticityCfg):
    """A fixed liquid has no plasticity, but the engine requires the section: every field inert, freeze=("all",), no clamp planned."""
    tau_pretrace: float = 1.0
    tau_posttrace: float = 1.0
    A_ltp: float = 0.0
    A_ltd: float = 0.0
    eta_exc: float = 0.0
    freeze: Tuple[str, ...] = ("all",)


class NoPlasticity(Plasticity):
    """No learning: dW stays zero (dense) or zeros are returned (sparse); works in both weights formats. Deckers' protocol never calls it:
    every pass runs in mode "eval"."""
    weights_format = None
    writes = ("dW",)
    requires = ()

    def __init__(self, cfg: PlasticityCfg, sim_cfg: SimulationCfg = None):
        if cfg is None:
            raise ValueError("NoPlasticity requires a PlasticityCfg (NoPlasticityCfg()).")
        self.w_min = cfg.w_min                          ##read by Run._plan_clamp (None -> no clamp)
        self.w_max = cfg.w_max
        self.freeze = tuple(cfg.freeze)
        self.eta_exc = float(cfg.eta_exc)
        self.eta_inh = cfg.eta_inh

    def step(self, ctx: RunContext):
        upd = ctx.upd
        if upd.dW is not None:
            return None                                 ##dense: dW is all zeros and stays so
        return np.zeros(ctx.syn.W.shape, dtype=ctx.syn.W.dtype)
