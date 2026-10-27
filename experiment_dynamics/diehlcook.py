#diehlcook - Diehl & Cook (2015), "Unsupervised learning of digit recognition using spike-timing-dependent plasticity",
# Front. Comput. Neurosci. 9:99 -- the model as PhasicNet runtime modules (+ the cfgs that go with them). ONE file, so the whole
# model is in one place. The network itself is built in experiments/diehl_cook_2015.py (builder patterns: all_to_all / one_to_one / all_but_self).
#
#SOURCE OF TRUTH: the authors' Brian 1 script (github.com/peter-u-diehl/stdp-mnist, Diehl&Cook_spiking_MNIST.py) AND what the Brian 1
# simulator does with it; the paper's Methods only where the script is silent. Units: ms, mV, conductances dimensionless, dt = 0.5 ms.
# system/tests/dc_brian1_reference.py is an independent Brian 1 transcription; test_diehl_cook.py checks these modules against it.
#
#WHERE: neuron + conductances + input delays -> DCConductanceSynapse, DCLIFeuler; adaptive threshold + refractoriness -> DCAdaptiveThreshold;
# STDP (input -> exc ONLY) -> DCTripletSTDP; normalisation (sum 78 before every presentation) -> dc_normalise, called by the PROTOCOL;
# fixed weights, initial weights U(0.003, 0.303), initial V (v_rest - 40) and the stimulus -> builder / protocol / preprocessing.
#
#KNOWN DEVIATIONS:
#   1. FLOAT32 shared arrays (W, V, I2V, dV, dW). Module-owned state and module arithmetic are float64, as Brian.
#   2. RANDOM STREAMS: same distributions (delays U[0, 10) ms, Poisson input), drawn from the run's keyed RNG, not np.random.seed(0).
#   3. Brian's 'timer' variable is not modelled: it exceeds refrac_e one step after a spike, the refractory block lasts 9 steps -> never matters.
#   4. Weights pushed above wmax by NORMALISATION stay there until a spike touches their row or column, as in Brian (dW: up to one float32 ulp).
#   5. ORDER inside Brian's 'connections' slot is undefined (dict order). Here: propagate with the pre-STDP W, then LTD, then LTP.
#   6. Reproduced on purpose: Brian 1's TRANSPOSED LTP-delay index and its 100-substep Euler post-trace decay.
#
#WHY Theta IS float64: the per-step decay (5e-8 * theta) is below float32 resolution at theta ~ 20 mV, so in float32 theta would never decay.
#
#CONTRACTS: Capitalised module arrays = state_dict (checkpointed, fit-reset); leading underscore = scratch or per-row constants.
# The ONLY shared arrays touched are the declared "writes" (I2V, dV, rec_spikes, dW). W is written by nobody here: the [0, wmax] bound is
# expressed THROUGH dW (commit: W += dW), so the orchestra stays untouched. Normalisation writes W directly, but it is not part of
# any step: dc_normalise() is a protocol call between samples, exactly where the script calls normalize_weights().
# Shared without a shared array: the input->exc delays (synapse and plasticity draw them from the same keyed stream) and NeuronalState.Since.
# experiments/diehl_cook_2015.build() fills the duplicated cfg fields (delays, t_refrac_inh, v_reset_inh) from one place and checks they agree.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging

import numpy as np

from dynamicscfg import SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg
from states import RunContext
from dynamics import Synapse, Integrator, Threshold, Plasticity

logger = logging.getLogger(__name__)

_MS = 0.001         ##brian_no_units' ms in SI seconds: delay arithmetic in Brian's units, so every int() lands where Brian's does


   ######################################################################################
##########################   Row helpers (per-group constants)   ##########################
   ######################################################################################

def _rows(net, slc: slice) -> slice:
    """A band of true_neuronIDs -> the matching band of W ROWS (post_offset drops the n_input leading rows)."""
    off = net.n_total - net.n_post
    return slice(slc.start - off, slc.stop - off)


def _row_vector(net, exc_value, inh_value, dtype=np.float32) -> np.ndarray:
    """(n_post,) per-ROW constant: inh_value on inhibitory rows, exc_value elsewhere (input rows count as exc; they never spike)."""
    vec = np.full(net.n_post, exc_value, dtype=dtype)
    inh = net.inh_slice
    if inh.stop > inh.start:
        vec[_rows(net, inh)] = inh_value
    return vec


def _inh_rows(net) -> np.ndarray:
    """(n_post,) bool, True on inhibitory rows."""
    return _row_vector(net, False, True, dtype=bool)


def brian_refractory_steps(t_refrac_ms: float, dt_ms: float) -> Tuple[int, int]:
    """Brian 1's two refractory numbers, from its own formulas.
        first_allowed = smallest steps-since-spike at which a spike passes (next_allowed = t_spike + refrac - dt/2 <= t)
        clamp_steps   = int(refrac/dt) + 1: v is held at reset at the end of the spike step and the clamp_steps-1 steps after it"""
    first_allowed = int(np.ceil(t_refrac_ms / dt_ms - 0.5))
    clamp_steps = int((t_refrac_ms * _MS) / (dt_ms * _MS)) + 1
    return first_allowed, clamp_steps


def brian_numeric_decay(tau_ms: float, dt_ms: float) -> float:
    """The per-step trace factor Brian 1 uses when a group's linear system is SINGULAR: 100 Euler sub-steps of dt/100. The post group of the
    script's STDP is singular because the parameter "post2before" becomes a zero-derivative equation in it. SI units, Brian's arithmetic."""
    dt_s, tau_s = dt_ms * _MS, tau_ms * _MS
    e = 1.0 + (dt_s / 100) * (-1.0 / tau_s)
    c = 1.0
    for _ in range(100):
        c = e * c
    return c


def dc_input_delays(rng_handler, W_block: np.ndarray, delay_min_ms: float, delay_max_ms: float, dt_ms: float, key: str = "delays"):
    """The script's input -> exc delays and Brian's integer arithmetic on them. Called by both the synapse and the plasticity on the same W
    (RNGHandler.rng is idempotent -> identical matrices). One uniform stream, consumed row by row over the NONZERO weights only, so a test
    network's delays are the training stream SHIFTED after each zero weight (and identical for
    the assignment and the test run). One fixed "key" = the script's one seed.
        W_block   (n_post, n_pre) the weights at creation (PhasicNet orientation)
        d         (n_post, n_pre) float64 SECONDS, U[min, max) on nonzero weights, 0 elsewhere
        K         int(d / dt): steps from emission to the ge increment
        LTD       int((max_delay - d) / dt), max_delay = R * dt
        KLTP      the LTP lag Brian 1 ACTUALLY reads (transposed delay index, a Brian 1 bug in the published results, reproduced):
                  KLTP[j, i] = int(D.flat[i + j * n_post] / dt) with D in Brian's (n_pre, n_post) layout
        R         int(delay_max / dt) + 1: ring length of the delayed reaction (also the LTD delay in steps)"""
    dt_s = dt_ms * _MS
    dmin, dmax = delay_min_ms * _MS, delay_max_ms * _MS
    n_post, n_pre = W_block.shape
    nz = np.asarray(W_block).T != 0                                             ##Brian layout (n_pre, n_post), row-major = set_delays' order
    stream = rng_handler.rng("dc_delays", key).random(n_pre * n_post)
    D = np.zeros((n_pre, n_post), dtype=np.float64)
    D[nz] = stream[:int(nz.sum())] * (dmax - dmin) + dmin
    R = int(dmax / dt_s) + 1
    idt = 1 / dt_s
    K = np.array(idt * D, dtype=np.int64).T.copy()
    LTD = np.array(idt * (R * dt_s - D), dtype=np.int64).T.copy()
    flat_idx = np.arange(n_pre)[None, :] + np.arange(n_post)[:, None] * n_post          ##(n_post, n_pre): i + j * n_post
    if flat_idx.max() >= D.size:
        raise ValueError(f"Brian 1's LTP delay lookup reads past the delay matrix for n_post={n_post} > ~n_pre={n_pre} (undefined in the original).")
    KLTP = np.array(idt * D.reshape(-1)[flat_idx], dtype=np.int64)
    return D.T.copy(), K, LTD, KLTP, R


def dc_normalise(W: np.ndarray, proj, target: float) -> None:
    """ADDED: the script's normalize_weights(), IN PLACE on W, for one projection: each post neuron's input weights scaled to sum "target"
    (colSums over the presynaptic axis, factor = target / colSum). No clipping, as the script. Called by the protocol before every
    training presentation -- NOT a dynamics step."""
    blk = W[proj.post, proj.pre]
    colsum = np.sum(blk.astype(np.float64), axis=1)
    blk *= (target / colsum)[:, None]


   ######################################################################################
##########################   Synapse: conductances ge, gi   ###############################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class DCSynapseCfg(SynapseCfg):
    """SynapseCfg + the conductance model. Reversal potentials are per GROUP (D&C: E_inh differs between exc and inh neurons).
    Args:
        tau_ge, tau_gi = ms, decay of the excitatory / inhibitory conductance
        e_exc = mV, excitatory reversal potential (both groups)
        e_inh = mV, inhibitory reversal potential onto EXC neurons
        e_inh_inh = mV, inhibitory reversal potential onto INH neurons
        delay_syn_group = the projection with per-synapse delays (None = no delays anywhere)
        delay_min, delay_max = ms, delays U[delay_min, delay_max) (the script: 0, 10)
        v_reset_inh, t_refrac_inh = the inh clamp (MUST equal DCThresholdCfg.v_reset_inh / DCIntegratorCfg.t_refrac_inh). None = no clamp.
    """
    tau_ge: float = 1.0
    tau_gi: float = 2.0
    e_exc: float = 0.0
    e_inh: float = -100.0
    e_inh_inh: float = -85.0
    delay_syn_group: Optional[str] = "in_exc"
    delay_min: float = 0.0
    delay_max: float = 10.0
    v_reset_inh: Optional[float] = -45.0
    t_refrac_inh: float = 2.0


class DCConductanceSynapse(Synapse):
    """
    D&C:   dge/dt = -ge/tau_ge ; dgi/dt = -gi/tau_gi ;  on_pre: ge += w (excitatory pre) | gi += w (inhibitory pre)
           I = ge*(E_exc - v) + gi*(E_inh - v)                                                  (Euler, as the script)

    Ge/Gi: module-owned PER-NEURON conductances (the integrator only sees I2V).
    ORDER per step m (= Brian 1's slots; neur.Spikes holds this step's INPUT spikes and last step's REC spikes):
        1. last step's rec spikes -> Ge/Gi
        2. the delayed input arrivals due at step m-1 -> Ge
        3. current from Ge/Gi and V at the start of the step (inh rows on their first step after the clamp: V = v_reset_inh)
        4. Ge/Gi decay (Euler)
        5. this step's input spikes enter the delay ring with the CURRENT W; they reach Ge int(d/dt) steps later
    Dale's law gives inhibitory columns NEGATIVE weights, so gi += -W for projections whose pre group is inhibitory.
    """
    weights_format = "dense"
    writes = ("I2V",)
    requires = ("Since",) 

    def __init__(self, cfg: DCSynapseCfg, sim_cfg: SimulationCfg = None):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("DCConductanceSynapse requires DCSynapseCfg and SimulationCfg with dt.")
        if cfg.tau_ge <= 0 or cfg.tau_gi <= 0:
            raise ValueError("DCConductanceSynapse requires tau_ge > 0 and tau_gi > 0.")
        self.b = None if cfg.b is None else float(cfg.b)
        self.tau_ge = float(cfg.tau_ge)                                     ##stored under cfg names (naming contract -> derived cfg)
        self.tau_gi = float(cfg.tau_gi)
        self.e_exc = float(cfg.e_exc)
        self.e_inh = float(cfg.e_inh)
        self.e_inh_inh = float(cfg.e_inh_inh)
        self.delay_syn_group = cfg.delay_syn_group
        self.delay_min = float(cfg.delay_min)
        self.delay_max = float(cfg.delay_max)
        self.v_reset_inh = None if cfg.v_reset_inh is None else float(cfg.v_reset_inh)
        self.t_refrac_inh = float(cfg.t_refrac_inh)
        self.dt = float(sim_cfg.dt)                                         ## Euler as Brian writes it (x += dt * dx)
        self.Ge = None
        self.Gi = None
        self.Buf = None             ##delay ring (R, n_post of the delayed group): the script's _delayedreaction
        self.Cur = None             ##(1,) ring index of THIS step's slot (= step % R)
        self.K = None               ##(n_post, n_pre) int steps of delay, drawn on the first step
        self.Delay_drawn = None     ##(1,) 0 = draw K at the next step (fresh model, or redraw_delays())
        self._R = 1
        self._e_inh_row = None      ##per-row E_inh (exc rows -100, inh rows -85): re-derived in configure_for_network
        self._projs = None          ##[(post_rows, pre_cols, inhibitory?)] immediate projections
        self._dproj = None          ##the delayed projection (post rows, pre cols)
        self._eff_rows = None       ##scratch: inh rows whose V is the clamp value at the start of this step
        self._inh = None
        self._clamp_since = None

    def configure_for_network(self, net) -> None:
        self.Ge = np.zeros(net.n_post, dtype=np.float64)                  
        self.Gi = np.zeros(net.n_post, dtype=np.float64)
        self._e_inh_row = _row_vector(net, self.e_inh, self.e_inh_inh, dtype=np.float64)
        self._inh = _inh_rows(net)
        off = net.n_total - net.n_post
        self._projs, self._dproj = [], None
        for name, proj in net.synapseslices.items():
            if name == self.delay_syn_group:
                self._dproj = proj
                continue
            inhibitory = proj.pre.start >= net.inh_slice.start and proj.pre.stop <= net.inh_slice.stop and net.inh_slice.stop > net.inh_slice.start
            self._projs.append((proj.post, proj.pre, inhibitory))
        if self.delay_syn_group is not None and self._dproj is None:
            raise RuntimeError(f"DCConductanceSynapse: delay_syn_group={self.delay_syn_group!r} is not a synapse group of this network.")
        if self._dproj is not None:
            n_post, n_pre = self._dproj.post.stop - self._dproj.post.start, self._dproj.pre.stop - self._dproj.pre.start
            self._R = int((self.delay_max * _MS) / (self.dt * _MS)) + 1
            self.Buf = np.zeros((self._R, n_post), dtype=np.float64)
            self.K = np.zeros((n_post, n_pre), dtype=np.int64)
        else:
            self.Buf, self.K = np.zeros((1, 0)), np.zeros((0, 0), dtype=np.int64)
        self.Cur = np.zeros(1, dtype=np.int64)
        self.Delay_drawn = np.zeros(1, dtype=np.int64)
        if self.v_reset_inh is not None:
            first_allowed, clamp_steps = brian_refractory_steps(self.t_refrac_inh, self.dt)
            self._clamp_since = (first_allowed + 1, clamp_steps)          ##rows needing the clamped V at step start: since in [A+1, P]
        self._eff_rows = np.zeros(net.n_post, dtype=bool)

    def redraw_delays(self) -> None:
        """New delays at the next step, over the nonzero weights of that moment (the script redraws at every network creation)."""
        self.Delay_drawn[0] = 0

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn, net = ctx.upd, ctx.neur, ctx.syn, ctx.net
        W, S = syn.W, neur.Spikes
        ##no per-module Since update; the orchestra advances neur.Since after every Spikes commit (same values at this point)
        ##1. last step's REC spikes, immediate projections (the delayed input group is never immediate)
        for post, pre, inhibitory in self._projs:
            G = self.Gi if inhibitory else self.Ge
            for j in np.flatnonzero(S[pre]):                                ##one row per spike, in spike order (Brian's propagation loop)
                col = W[post, pre.start + j].astype(np.float64)
                if inhibitory:
                    G[post] += -col                                         ##Dale: stored negative -> gi += |w|
                else:
                    G[post] += col
        ##2. delayed arrivals of Brian's step m-1 (after_connections), then the slot is free again
        if self._dproj is not None:
            if not self.Delay_drawn[0]:
                self._draw(ctx)
            slot = (self.Cur[0] - 1) % self._R
            self.Ge[self._dproj.post] += self.Buf[slot]
            self.Buf[slot] = 0.0
        ##3. I from V at the start of the step; inh rows right after the clamp use the clamp value
        V = neur.V.astype(np.float64)
        if self._clamp_since is not None:
            lo, hi = self._clamp_since
            np.logical_and(self._inh, (neur.Since >= lo) & (neur.Since <= hi), out=self._eff_rows)   
            V[self._eff_rows] = self.v_reset_inh
        I = self.Ge * (self.e_exc - V) + self.Gi * (self._e_inh_row - V)
        if self.b is not None:
            I = I + self.b
        upd.I2V[...] = I
        ##4. decay (Euler, Brian's form: x += dt * (-x / tau))
        self.Ge += (-self.Ge / self.tau_ge) * self.dt
        self.Gi += (-self.Gi / self.tau_gi) * self.dt
        ##5. this step's input spikes enter the ring with the current W (emission)
        if self._dproj is not None:
            pre_sp = np.flatnonzero(S[self._dproj.pre])
            if pre_sp.size:
                blk = W[self._dproj.post, self._dproj.pre]
                cols = np.arange(self.Buf.shape[1])
                for i in pre_sp:                                            ##in spike order, as Brian's propagation loop
                    np.add.at(self.Buf, ((self.Cur[0] + self.K[:, i]) % self._R, cols), blk[:, i].astype(np.float64))
            self.Cur[0] = (self.Cur[0] + 1) % self._R

    def _draw(self, ctx) -> None:
        _, K, _, _, R = dc_input_delays(ctx.rng, ctx.syn.W[self._dproj.post, self._dproj.pre], self.delay_min, self.delay_max, self.dt)
        assert R == self._R
        self.K[...] = K
        self.Delay_drawn[0] = 1
        logger.info("DCConductanceSynapse: delays drawn, %d..%d steps.", int(K.min()), int(K.max()))


   ######################################################################################
##########################   Integrator: Euler LIF, two groups, inh clamp   ###############
   ######################################################################################

@dataclass(frozen=True, slots=True)
class DCIntegratorCfg(IntegratorCfg):
    """IntegratorCfg (tau_mem, v_rest, r apply to EXC neurons; r = 1.0: I2V is already in mV) + the inhibitory group + refractory periods.
    Args:
        tau_mem_inh, v_rest_inh = inhibitory-group counterparts of tau_mem, v_rest
        t_refrac, t_refrac_inh = ms, refractory periods, exc / inh. CHANGED: the EXC period no longer acts here 
        v_reset_inh = the inh clamp value (Brian's Refractoriness holds inh v at the reset value). None = no clamp.
    """
    tau_mem_inh: float = 10.0
    v_rest_inh: float = -60.0
    t_refrac: float = 5.0
    t_refrac_inh: float = 2.0
    v_reset_inh: Optional[float] = -45.0


class DCLIFeuler(Integrator):
    """
    D&C:   dv/dt = ((v_rest - v) + I) / tau_mem      Euler, Brian's form:  V_new = V_eff + dt * ((v_rest - V_eff) + I) / tau

    EXC rows integrate on every step, also while refractory (the script's string reset does not clamp).
    INH rows: Brian's Refractoriness sets v = v_reset_inh at the end of the spike step and of the 4 steps after it, while spikes stay blocked
    for 3 steps. The orchestra's reset covers the spike step; here, by steps since the last spike (Since):
        Since 1..3   dV = 0                     (v stays at the clamp value; the threshold could not pass anyway)
        Since 4      normal Euler from the clamp value (the threshold sees this value; Brian clamps AFTER the threshold)
        Since 5      Euler from the clamp value, not from V: Brian's clamp at the end of step 4 is applied here (V_eff), and the synapse formed
                     this step's current at the same V_eff
    Membrane NOISE is added by the orchestra AFTER dV -- D&C has none; keep sigma_mem 0 for this model.
    """
    writes = ("dV", "I2V")      ##I2V multiplied by r in place (as LIFeuler); with r = 1.0 that is a no-op kept for the cfg contract
    requires = ("Since",)   

    def __init__(self, cfg: DCIntegratorCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("DCLIFeuler requires DCIntegratorCfg and SimulationCfg with dt.")
        if cfg.tau_mem <= 0 or cfg.tau_mem_inh <= 0:
            raise ValueError("DCLIFeuler requires tau_mem > 0 and tau_mem_inh > 0.")
        if cfg.t_refrac < 0 or cfg.t_refrac_inh < 0:
            raise ValueError("DCLIFeuler requires t_refrac >= 0 and t_refrac_inh >= 0.")
        self.dt = float(sim_cfg.dt)
        self.v_rest = np.float32(cfg.v_rest)            ##EXC value under the cfg name: the orchestra's reset_neuronal_state reads it as the scalar
        self.v_rest_inh = float(cfg.v_rest_inh)
        self.tau_mem = float(cfg.tau_mem)
        self.tau_mem_inh = float(cfg.tau_mem_inh)
        self.r = float(cfg.r)
        self.t_refrac = float(cfg.t_refrac)
        self.t_refrac_inh = float(cfg.t_refrac_inh)
        self.v_reset_inh = None if cfg.v_reset_inh is None else float(cfg.v_reset_inh)
        self._v_rest_row = None     ##per-row constants, re-derived in configure_for_network
        self._tau_row = None
        self._hold = None           ##scratch: rows held this step
        self._eff = None            ##scratch: rows integrated from the clamp value
        self._inh = None
        self._windows = None

    def configure_for_network(self, net) -> None:
        self._v_rest_row = _row_vector(net, float(self.v_rest), self.v_rest_inh, dtype=np.float64)
        self._tau_row = _row_vector(net, self.tau_mem, self.tau_mem_inh, dtype=np.float64)
        self._inh = _inh_rows(net)
        if self.v_reset_inh is not None:
            first_allowed, clamp_steps = brian_refractory_steps(self.t_refrac_inh, self.dt)
            self._windows = (min(first_allowed - 1, clamp_steps - 1), first_allowed + 1, clamp_steps)   ##hold 1..H ; V_eff A+1..P
        self._hold = np.zeros(net.n_post, dtype=bool)
        self._eff = np.zeros(net.n_post, dtype=bool)


    def step(self, ctx: RunContext) -> None:
        upd, neur, net = ctx.upd, ctx.neur, ctx.net
        I = upd.I2V.astype(np.float64) * self.r
        upd.I2V[...] = I
        V = neur.V.astype(np.float64)
        V_eff = V.copy()
        if self._windows is not None:
            h, lo, hi = self._windows
            np.logical_and(self._inh, (neur.Since >= 1) & (neur.Since <= h), out=self._hold)   #
            np.logical_and(self._inh, (neur.Since >= lo) & (neur.Since <= hi), out=self._eff)
            V_eff[self._eff] = self.v_reset_inh
        V_new = V_eff + (((self._v_rest_row - V_eff) + I) / self._tau_row) * self.dt      ##Brian: v__tmp = (...)/tau ; v += v__tmp*dt
        dV = V_new - V
        if self._windows is not None:
            dV[self._hold] = 0.0
        upd.dV[...] = dV


   ######################################################################################
##########################   Threshold: theta - offset (exc), fixed (inh)   ###############
   ######################################################################################

@dataclass(frozen=True, slots=True)
class DCThresholdCfg(ThresholdCfg):
    """ThresholdCfg (v_reset, v_th, tau_th, adpt_th, t_refrac apply to EXC neurons: tau_th = tau_theta, adpt_th = theta_plus) + the rest of D&C.
    Args:
        v_reset_inh, v_th_inh = mV, inhibitory group (no adaptation: Theta stays 0 on inh rows)
        theta_init = mV, theta at t = 0 (training value; the script's theta = 20 mV)
        offset = mV, spike iff V > v_th + theta - offset
        t_refrac_inh = ms, inh refractory period (spikes blocked, Brian's next_allowed_spiketime)
    """
    v_reset_inh: float = -45.0
    v_th_inh: float = -40.0
    theta_init: float = 20.0
    offset: float = 20.0
    t_refrac_inh: float = 2.0


class DCAdaptiveThreshold(Threshold):
    """
    D&C (exc):   spike iff v > theta - offset + v_thresh  AND not refractory ;  theta += theta_plus on spike ;  dtheta/dt = -theta/tau_theta (Euler)
    D&C (inh):   spike iff v > v_thresh_inh                AND not refractory                                       (no theta)

    Theta is module-owned (it is read by nobody else): writes = ("rec_spikes",) only. Theta(0) = theta_init on exc rows, 0 on inh rows, and a fit-reset restores
    exactly that through state_dict. float64: see the header. Strict ">" as in the script.
    Refractoriness: a spike passes only if the steps since the row's last spike >= first_allowed (exc 10, inh 4): Brian's
    next_allowed_spiketime = t_spike + refrac - dt/2. The script's extra "timer > refrac_e" is always true by then (header, deviation 3).
    v_reset is PER ROW (-65 exc / -45 inh) and public: the orchestra's reset line "np.putmask(V[rec], spikes, threshold.v_reset)"  takes it as-is. Being an ndarray it lands in state_dict (type-based classification; constant, so harmless) and not in params,
    so the derived cfg keeps the entered v_reset.
    """
    writes = ("rec_spikes",)
    requires = ("Since",)   
    persistent = ("Theta",)   

    def __init__(self, cfg: DCThresholdCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("DCAdaptiveThreshold requires DCThresholdCfg and SimulationCfg with dt.")
        if cfg.tau_th is None or cfg.tau_th <= 0:
            raise ValueError("DCAdaptiveThreshold requires tau_th > 0 (tau_theta).")
        if cfg.adpt_th is None:
            raise ValueError("DCAdaptiveThreshold requires adpt_th (theta_plus).")
        self.v_th = float(cfg.v_th)                     ##EXC base threshold under the cfg name
        self.v_th_inh = float(cfg.v_th_inh)
        self.v_reset_inh = np.float32(cfg.v_reset_inh)
        self.tau_th = float(cfg.tau_th)
        self.adpt_th = float(cfg.adpt_th)
        self.theta_init = float(cfg.theta_init)
        self.offset = float(cfg.offset)
        self.t_refrac = 0.0 if cfg.t_refrac is None else float(cfg.t_refrac)    ##NOTE: now used -- the exc spike block
        self.t_refrac_inh = float(cfg.t_refrac_inh)
        self.dt = float(sim_cfg.dt)                     ##NOTE: Euler in Brian's form (theta += dt * (-theta / tau)), float64
        self._v_reset_exc = np.float32(cfg.v_reset)     ##scalar kept for configure_for_network; the public v_reset becomes the per-row array
        self.v_reset = None
        self.Theta = None
        self._first_allowed = None  ##per row, smallest Since at which a spike may pass
        self._exc_rows = None       ##rows with theta (exc); inh rows: fixed threshold
        self._thr = None            ##scratch (float64)
        self._frozen = False        ##ADDED: test mode 

    def freeze(self, frozen: bool = True) -> None:
        """freeze(True) fixes Theta (no decay, no theta_plus) for label assignment and testing. learn=False alone does NOT: it only skips plasticity."""
        self._frozen = bool(frozen)

    def configure_for_network(self, net) -> None:
        self.v_reset = _row_vector(net, self._v_reset_exc, self.v_reset_inh)[net.rec_slice]        ##(n_rec,): what the orchestra resets with
        self.Theta = _row_vector(net, self.theta_init, 0.0, dtype=np.float64)
        self._exc_rows = ~_inh_rows(net)
        if net.n_post > net.n_rec:                      ##no post_offset: input rows exist in V; make sure they can never pass the threshold
            self.Theta[:net.n_input] = 0.0
            self._exc_rows[:net.n_input] = False
        fa_exc, _ = brian_refractory_steps(self.t_refrac, self.dt)
        fa_inh, _ = brian_refractory_steps(self.t_refrac_inh, self.dt)
        self._first_allowed = _row_vector(net, fa_exc, fa_inh, dtype=np.int64)
        self._thr = np.full(net.n_post, np.inf, dtype=np.float64)       ##exc rows filled every step; input rows (no post_offset) stay +inf
        self._thr[_inh_rows(net)] = self.v_th_inh



    def step(self, ctx: RunContext) -> None:
        upd, neur, net = ctx.upd, ctx.neur, ctx.net
        ##Since is advanced by the orchestra, not here
        ex = self._exc_rows
        if not self._frozen:                                                ##frozen (test mode) -> Theta fixed
            self.Theta[ex] += (-self.Theta[ex] / self.tau_th) * self.dt     ##state update (decay, Euler) ...
        self._thr[ex] = (self.Theta[ex] - self.offset) + self.v_th          ##... the script's expression, in its order ...
        spikes = (neur.V.astype(np.float64) > self._thr) & (neur.Since >= self._first_allowed)     ##... strict, then the refractory filter
        upd.rec_spikes[...] = spikes
        if not self._frozen:                                                ##frozen (test mode) -> no adaptation
            self.Theta[spikes & ex] += self.adpt_th                         ##... then reset: theta += theta_plus (exc rows only)
        #### MEMBRANE V RESET NOT APPLIED. Left to the orchestrator (per-row v_reset above).


   ######################################################################################
##########################   Plasticity: triplet STDP on the delayed connection   ##########
   ######################################################################################

@dataclass(frozen=True, slots=True)
class DCPlasticityCfg(PlasticityCfg):
    """PlasticityCfg with D&C meanings: tau_pretrace = tau_pre, tau_posttrace = tau_post1, A_ltp = nu_post, A_ltd = nu_pre,
    eta_exc = global multiplier on both rates (1.0 = D&C). eta_inh, w_min, w_max: leave None -- a finite w_max would clip the FIXED
    exc->inh = 10.4 every step; this module bounds its own block.
    Args:
        tau_posttrace2 = ms, the slow post trace (post2)
        w_max_ee = upper bound of the plastic input weights (clipped to [0, w_max_ee] through dW)
        norm_target = divisive normalisation: input weights of each exc neuron are scaled to this sum before every example (None = off),
            by the PROTOCOL (dc_normalise), not inside a step.
        norm_syn_group = the synapse group that is normalised (and must be plastic)
        delay_syn_group, delay_min, delay_max = MUST equal DCSynapseCfg's (the STDP reads the traces at the delayed times)
        triplet = True: the script's triplet rule (LTP * slow post trace post2); False: the PAIR rule of Zhang 2025 Eq. 2-3 (post2 read
            as 1), everything else unchanged. D&C keeps True.
    """
    tau_posttrace2: float = 40.0
    triplet: bool = True
    w_max_ee: float = 1.0
    norm_target: Optional[float] = 78.0
    norm_syn_group: str = "in_exc"
    delay_syn_group: Optional[str] = "in_exc"
    delay_min: float = 0.0
    delay_max: float = 10.0


class DCTripletSTDP(Plasticity):
    """
    The script's triplet rule as Brian 1's C-STDP executes it on a DelayConnection. Per learning step m:

        traces decay: pre EXACTLY (linear group), post1/post2 by Brian's 100-substep Euler (their group is singular)
        pre spike at m:          pre = 1
        pre spike at m - R:      every synapse (j, i) of input i:  w -= nu_pre * post1_j recorded at the end of step m-1-LTD[j,i], then clip
                                 (= post1 at the end of the step the spike REACHED neuron j: t_pre + int(d/dt))
        post spike at m:         post2before = post2 ; post1 = post2 = 1 ; every synapse (j, i) of neuron j:
                                 w += nu_post * pre_i recorded at the end of step m-1-KLTP[j,i] * post2before, then clip
                                 (KLTP = the lag Brian 1 actually reads -- its dense column loop indexes the delays transposed; reproduced)
        end of step:             record pre, post1 (Brian's RecentStateMonitors, 23 steps deep)
    LTD runs before LTP within a step; every touched synapse is clipped to [0, w_max_ee], also with a zero change (as Brian).
    dW = (new w - old w) on the touched rows/columns, 0 elsewhere. Requires delays (delay_syn_group = the one plastic projection).
    Event-driven: only this step's post-spike rows and the columns of inputs that spiked R steps ago are touched.
    """
    weights_format = "dense"
    writes = ("dW",)
    requires = ("pre_spike", "post_spike")

    def __init__(self, cfg: DCPlasticityCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("DCTripletSTDP requires DCPlasticityCfg and SimulationCfg with dt.")
        if cfg.tau_pretrace <= 0 or cfg.tau_posttrace <= 0 or cfg.tau_posttrace2 <= 0:
            raise ValueError("DCTripletSTDP requires tau_pretrace, tau_posttrace, tau_posttrace2 > 0.")
        if cfg.w_max_ee is None or cfg.w_max_ee <= 0:
            raise ValueError("DCTripletSTDP requires w_max_ee > 0.")
        if cfg.delay_syn_group is None:
            raise ValueError("DCTripletSTDP (Brian 1 semantics) requires delay_syn_group: the plastic projection with delays.")
        self.tau_pretrace = float(cfg.tau_pretrace)
        self.tau_posttrace = float(cfg.tau_posttrace)
        self.tau_posttrace2 = float(cfg.tau_posttrace2)
        self.A_ltp = float(cfg.A_ltp)                   ##nu_post
        self.A_ltd = float(cfg.A_ltd)                   ##nu_pre
        self.eta_exc = float(cfg.eta_exc)
        self.eta_inh = cfg.eta_inh                      ##unused (no inhibitory plasticity in D&C); kept for the cfg contract
        self.w_min = None if cfg.w_min is None else np.float32(cfg.w_min)
        self.w_max = None if cfg.w_max is None else np.float32(cfg.w_max)
        self.w_max_ee = float(cfg.w_max_ee)
        self.triplet = bool(cfg.triplet)                ##False = pair rule (Zhang 2025), post2before read as 1
        self.norm_target = None if cfg.norm_target is None else float(cfg.norm_target)
        self.norm_syn_group = cfg.norm_syn_group
        self.delay_syn_group = cfg.delay_syn_group
        self.delay_min = float(cfg.delay_min)
        self.delay_max = float(cfg.delay_max)
        self.freeze = tuple(cfg.freeze)
        self.dt = float(sim_cfg.dt)
        self.decay_pretrace = float(np.exp(-sim_cfg.dt / cfg.tau_pretrace))      ##exact decay (Brian 1: linear trace groups -> exact updates)
        self.decay_posttrace = brian_numeric_decay(cfg.tau_posttrace, sim_cfg.dt)     ##NOTE (review): the post group is singular in Brian
        self.decay_posttrace2 = brian_numeric_decay(cfg.tau_posttrace2, sim_cfg.dt)   ## (post2before) -> 100-substep Euler, not exp
        self.Pre = None             ##module state: traces (float64, as Brian)
        self.Post1 = None
        self.Post2 = None
        self.PreRec = None          ##end-of-step records (RecentStateMonitor), (ND, n_pre) / (ND, n_post)
        self.Post1Rec = None
        self.PreHist = None         ##input spikes of the last R+1 steps (the input group's spike history LS)
        self.Clock = None           ##(1,) learning steps done (= Brian's step index during training)
        self.KLTP = None            ##the LTP lags Brian reads (transposed, see dc_input_delays) / LTD lags; drawn on the first step
        self.LTD = None             ## from the same stream and W as the synapse
        self.Delay_drawn = None
        self._R = None
        self._ND = None
        self._proj = None
        self._touched = None        ##rows, cols written into dW last step (zeroed before writing)

    def configure_for_network(self, net) -> None:
        if self.delay_syn_group not in net.synapseslices:
            raise RuntimeError(f"DCTripletSTDP: delay_syn_group={self.delay_syn_group!r} is not a synapse group of this network.")
        self._proj = proj = net.synapseslices[self.delay_syn_group]
        n_post, n_pre = proj.post.stop - proj.post.start, proj.pre.stop - proj.pre.start
        self._R = int((self.delay_max * _MS) / (self.dt * _MS)) + 1
        self._ND = int(((self._R + 1) * (self.dt * _MS)) / (self.dt * _MS)) + 1        ##RecentStateMonitor: duration (R+1)*dt -> int(dur/dt)+1
        self.Pre = np.zeros(n_pre, dtype=np.float64)
        self.Post1 = np.zeros(n_post, dtype=np.float64)
        self.Post2 = np.zeros(n_post, dtype=np.float64)
        self.PreRec = np.zeros((self._ND, n_pre), dtype=np.float64)
        self.Post1Rec = np.zeros((self._ND, n_post), dtype=np.float64)
        self.PreHist = np.zeros((self._R + 1, n_pre), dtype=bool)
        self.Clock = np.zeros(1, dtype=np.int64)
        self.KLTP = np.zeros((n_post, n_pre), dtype=np.int64)
        self.LTD = np.zeros((n_post, n_pre), dtype=np.int64)
        self.Delay_drawn = np.zeros(1, dtype=np.int64)
        self._touched = (np.zeros(0, np.int64), np.zeros(0, np.int64))

    def redraw_delays(self) -> None:
        """As DCConductanceSynapse.redraw_delays (same stream, same W -> same matrices)."""
        self.Delay_drawn[0] = 0

    def _draw(self, ctx) -> None:
        if not ctx.syn.syn_mask[self._proj.post, self._proj.pre].all():
            raise RuntimeError("DCTripletSTDP: the delayed projection must be fully realized (D&C: all_to_all); dW is not masked.")
        _, _, LTD, KLTP, R = dc_input_delays(ctx.rng, ctx.syn.W[self._proj.post, self._proj.pre], self.delay_min, self.delay_max, self.dt)
        assert R == self._R
        self.KLTP[...] = KLTP; self.LTD[...] = LTD; self.Delay_drawn[0] = 1

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        if not self.Delay_drawn[0]:
            self._draw(ctx)
        proj, R, ND = self._proj, self._R, self._ND
        m = int(self.Clock[0])
        dW = upd.dW[proj.post, proj.pre]
        prev_r, prev_c = self._touched                                      ##zero what was written last step (dW persists between steps)
        if prev_r.size: dW[prev_r, :] = 0.0
        if prev_c.size: dW[:, prev_c] = 0.0
        pre_sp = neur.pre_spike[proj.pre] > 0
        post_sp = np.flatnonzero(neur.post_spike[proj.post] > 0)
        ##'groups': exact trace decay
        self.Pre *= self.decay_pretrace
        self.Post1 *= self.decay_posttrace
        self.Post2 *= self.decay_posttrace2
        ##'connections', pre (immediate): pre = 1
        self.Pre[pre_sp] = 1.0
        ##'connections', pre (delayed by R steps): LTD over the whole row of each input that spiked at m - R
        cols = np.flatnonzero(self.PreHist[(m - R) % (R + 1)]) if m >= R else np.zeros(0, np.int64)
        W = syn.W[proj.post, proj.pre]
        new_cols = None
        if cols.size:
            post_idx = np.arange(W.shape[0])
            new_cols = W[:, cols].astype(np.float64)
            for n, i in enumerate(cols):
                post1 = self.Post1Rec[(ND + (m % ND) - 1 - self.LTD[:, i]) % ND, post_idx]
                v = new_cols[:, n] - (self.A_ltd * self.eta_exc) * post1
                new_cols[:, n] = np.minimum(np.maximum(v, 0.0, out=v), self.w_max_ee, out=v)      ##clip (max/min: same values, cheaper call)
        ##'connections', post: post2before = post2 ; post1 = post2 = 1 ; LTP over the whole column of each spiking post neuron
        new_rows = None
        if post_sp.size:
            ##pair rule (triplet=False, Zhang 2025 Eq. 3): post2before = 1; Post2 is still kept (same state layout)
            post2before = self.Post2[post_sp].copy() if self.triplet else np.ones(post_sp.size, dtype=np.float64)
            self.Post1[post_sp] = 1.0
            self.Post2[post_sp] = 1.0
            pre_idx = np.arange(W.shape[1])
            new_rows = W[post_sp, :].astype(np.float64)
            if new_cols is not None:                                        ##LTD of this step already applied to these synapses
                new_rows[:, cols] = new_cols[post_sp, :]
            for n, j in enumerate(post_sp):
                pre = self.PreRec[(ND + (m % ND) - 1 - self.KLTP[j, :]) % ND, pre_idx]     ##Brian's (transposed) LTP lag, see dc_input_delays
                v = new_rows[n] + (self.A_ltp * self.eta_exc) * pre * post2before[n]
                new_rows[n] = np.minimum(np.maximum(v, 0.0, out=v), self.w_max_ee, out=v)
        ##dW = new - old on the touched entries
        if new_cols is not None:
            dW[:, cols] = new_cols - W[:, cols].astype(np.float64)
        if new_rows is not None:
            dW[post_sp, :] = new_rows - W[post_sp, :].astype(np.float64)
        self._touched = (post_sp, cols)
        ##'end': record, spike history, clock
        self.PreRec[m % ND] = self.Pre
        self.Post1Rec[m % ND] = self.Post1
        self.PreHist[m % (R + 1)] = pre_sp
        self.Clock[0] = m + 1
