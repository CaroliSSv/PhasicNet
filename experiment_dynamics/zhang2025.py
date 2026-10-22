#zhang2025 - Zhang, Yang, Zhang & Chen (2025), "Biologically plausible unsupervised learning for self-organizing spiking neural networks with
# dendritic computation", Neurocomputing 631:129707 -- the paper's additions to Diehl & Cook (2015) as subclasses of the diehlcook.py modules
# (D&C itself is not changed). The network (SOM-graded inhibition, partially shared wiring) is built in experiments/zhang_2025.py. Dense only.
#
#WHAT THE PAPER STATES:
#   §3.1  Eq. 4   dI^E_j/dt = -I^E_j / tau_L + u^X_j             tau_L = 10 ms    (dendritic current, driven by the presynaptic activity u^X_j)
#         Eq. 5   y_j = sum_i gamma x_i                           gamma = 0.5     (soma input = gamma-weighted activities x_i of the dendrites)
#         Eq. 6   I^{E_r}_j = 1 / (1 + exp(-(I^E_j - theta_f)))   theta_f = [1, 5, 10]
#   §3.3  Eq. 7   lateral inhibition graded by the distance between neurons (SOM), lambda = 100, beta_s = 1.0; Fig. 2: of the neurons reaching
#                 threshold at the same moment only one spikes
#   §3.4  partially shared exc -> inh wiring;  §4.2  D&C's STDP and adaptive threshold
# Everything else is an INFERRED cfg default (ZhangSynapseCfg, ZhangThresholdCfg, experiments/zhang_2025.py). THIS IS A DENDRITIC D&C VARIANT BUILT FROM
# THE PAPER'S EQUATIONS. IT CANNOT BE CLAIMED TO BE ZHANG ET AL.'S MODEL (no code published).
#
#DEFAULTS: theta_f = [1, 5, 10] brackets the Eq. 4 current of a dendrite that sees every input (~6.5), hence "gate"; y as a current would be
# ~1 mV and the soma would never fire, hence "conductance".
#
#WHERE EACH ADDITION LIVES:
#   dendrites           ZhangSynapseCfg + ZhangDendriticSynapse (here)
#   one spike per step  ZhangThresholdCfg + ZhangOneSpikeThreshold (here)
#   SOM inhibition      experiments/zhang_2025.py som_inhibition_weights(): Eq. 7 written into the inh -> exc weights at BUILD time
#   shared wiring       experiments/zhang_2025.py psc_groups() + two connectivity patterns in builder.CONNECTIVITY_PATTERNS
# The learning rule is D&C's DCTripletSTDP with triplet=False (the pair rule the paper states).
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import logging

import numpy as np

from cfg import SimulationCfg
from states import RunContext
from diehlcook import DCSynapseCfg, DCConductanceSynapse, DCThresholdCfg, DCAdaptiveThreshold

logger = logging.getLogger(__name__)


   ######################################################################################
##########################   Synapse: D&C conductances + excitatory dendrites   ###########
   ######################################################################################

@dataclass(frozen=True, slots=True)
class ZhangSynapseCfg(DCSynapseCfg):
    """DCSynapseCfg + the excitatory dendrites (Eq. 4-6). The dendritic projection is DCSynapseCfg.delay_syn_group (D&C: in_exc).
    Args:
        theta_f = Eq. 6 thresholds, ONE DENDRITE PER ENTRY (paper: [1, 5, 10])
        gamma = Eq. 5 amplification (paper: 0.5)
        tau_dend = ms, Eq. 4 decay of the dendritic current (paper: tau_L = 10 ms)
        dendrite_assign = INFERRED "all": every input feeds every dendrite | "split": input i feeds dendrite i mod n_dend only
        dendrite_rule = INFERRED "gate": the soma receives sigmoid(I_f - theta_f), I_f keeps integrating |
            "retain": the soma receives I_f, then I_f <- sigmoid(I_f - theta_f)
        soma_input = INFERRED "conductance": y replaces ge in D&C's soma equation | "current": soma_gain * y is added to I2V in mV
        soma_gain = mV per unit of y under "current" (no value in the paper)
    """
    theta_f: Tuple[float, ...] = (1.0, 5.0, 10.0)
    gamma: float = 0.5
    tau_dend: float = 10.0
    dendrite_assign: str = "all"
    dendrite_rule: str = "gate"
    soma_input: str = "conductance"
    soma_gain: float = 1.0
    cfg_type: str = field(default="ZhangSynapseCfg", kw_only=True)


class ZhangDendriticSynapse(DCConductanceSynapse):
    """
    D&C's DCConductanceSynapse with the DELAYED projection (input -> exc) re-routed to excitatory dendrites:
        Eq. 4   Dend_f += arrivals ; Dend_f += dt * (-Dend_f / tau_dend)   (Euler)
        Eq. 6   x_f = 1 / (1 + exp(-(Dend_f - theta_f)))
        Eq. 5   y = gamma * sum_f x_f
        soma    I = y * (E_exc - V) + gi * (E_inh - V)      ("conductance")   |   I = soma_gain * y + gi * (E_inh - V)      ("current")
    Other rows keep D&C's ge/gi current, and the step order is D&C's, so input spike timing is unchanged. Dend has one row per dendrite,
    or ONE row under "all" + "gate" (the currents would be identical).
    """

    def __init__(self, cfg: ZhangSynapseCfg, sim_cfg: SimulationCfg = None):
        super().__init__(cfg, sim_cfg)
        theta = tuple(float(t) for t in cfg.theta_f)
        if not theta:
            raise ValueError("ZhangDendriticSynapse requires at least one theta_f (one dendrite).")
        if cfg.tau_dend <= 0:
            raise ValueError("ZhangDendriticSynapse requires tau_dend > 0.")
        if cfg.dendrite_assign not in ("all", "split"):
            raise ValueError(f"ZhangDendriticSynapse: dendrite_assign must be 'all' or 'split', got {cfg.dendrite_assign!r}.")
        if cfg.dendrite_rule not in ("gate", "retain"):
            raise ValueError(f"ZhangDendriticSynapse: dendrite_rule must be 'gate' or 'retain', got {cfg.dendrite_rule!r}.")
        if cfg.soma_input not in ("conductance", "current"):
            raise ValueError(f"ZhangDendriticSynapse: soma_input must be 'conductance' or 'current', got {cfg.soma_input!r}.")
        if cfg.delay_syn_group is None:
            raise ValueError("ZhangDendriticSynapse: the dendritic projection is delay_syn_group (D&C: 'in_exc'); None is not allowed.")
        self.theta_f = theta
        self.gamma = float(cfg.gamma)
        self.tau_dend = float(cfg.tau_dend)
        self.dendrite_assign = cfg.dendrite_assign
        self.dendrite_rule = cfg.dendrite_rule
        self.soma_input = cfg.soma_input
        self.soma_gain = float(cfg.soma_gain)
        self.Dend = None            ##(n_stored, n_dpost) float64: the dendritic currents I^E (Eq. 4), transient state
        self._theta_col = None      ##(n_dend, 1)
        self._dend_of_pre = None    ##(n_pre,) dendrite of each input ("split"); None = every input to every dendrite ("all")
        self._x = None              ##scratch (n_dend, n_dpost): the Eq. 6 sigmoids

    def configure_for_network(self, net) -> None:
        super().configure_for_network(net)
        n_dpost, n_pre = self.K.shape
        n_dend = len(self.theta_f)
        self._theta_col = np.asarray(self.theta_f, dtype=np.float64)[:, None]
        if self.dendrite_assign == "split":
            self._dend_of_pre = np.arange(n_pre) % n_dend
            n_stored = n_dend
        else:
            self._dend_of_pre = None
            n_stored = n_dend if self.dendrite_rule == "retain" else 1
        self.Dend = np.zeros((n_stored, n_dpost), dtype=np.float64)
        self.Buf = np.zeros((self._R, n_dend if self._dend_of_pre is not None else 1, n_dpost), dtype=np.float64)   ##D&C's ring, one slot per dendrite
        self._x = np.zeros((n_dend, n_dpost), dtype=np.float64)

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        W, S = syn.W, neur.Spikes
        ##1. last step's REC spikes -> Ge / Gi (as D&C)
        for post, pre, inhibitory in self._projs:
            G = self.Gi if inhibitory else self.Ge
            for j in np.flatnonzero(S[pre]):
                col = W[post, pre.start + j].astype(np.float64)
                if inhibitory:
                    G[post] += -col
                else:
                    G[post] += col
        ##2. delayed arrivals -> the DENDRITIC current(s) instead of Ge
        if not self.Delay_drawn[0]:
            self._draw(ctx)
        slot = (self.Cur[0] - 1) % self._R
        self.Dend += self.Buf[slot]
        self.Buf[slot] = 0.0
        ##3. the current: D&C's ge/gi expression + the dendritic drive
        V = neur.V.astype(np.float64)
        if self._clamp_since is not None:
            lo, hi = self._clamp_since
            np.logical_and(self._inh, (neur.Since >= lo) & (neur.Since <= hi), out=self._eff_rows)
            V[self._eff_rows] = self.v_reset_inh
        I = self.Ge * (self.e_exc - V) + self.Gi * (self._e_inh_row - V)
        x = self._x
        np.subtract(self.Dend, self._theta_col, out=x)                   ##Eq. 6: sigmoid(Dend_f - theta_f), in place
        np.negative(x, out=x); np.exp(x, out=x); x += 1.0; np.reciprocal(x, out=x)
        if self.dendrite_rule == "gate":
            y = self.gamma * x.sum(axis=0)                               ##Eq. 5 over the gated activities
        else:
            y = self.gamma * self.Dend.sum(axis=0)                       ##Eq. 5 over the currents themselves ...
            self.Dend[...] = x                                           ##... then Eq. 6 is what the dendrites retain
        rows = self._dproj.post
        if self.soma_input == "conductance":
            I[rows] += y * (self.e_exc - V[rows])
        else:
            I[rows] += self.soma_gain * y
        if self.b is not None:
            I = I + self.b
        upd.I2V[...] = I
        ##4. decay (Euler), incl. Eq. 4's leak for the dendrites
        self.Ge += (-self.Ge / self.tau_ge) * self.dt
        self.Gi += (-self.Gi / self.tau_gi) * self.dt
        self.Dend += (-self.Dend / self.tau_dend) * self.dt
        ##5. this step's input spikes enter the ring, in the input's dendrite slot
        pre_sp = np.flatnonzero(S[self._dproj.pre])
        if pre_sp.size:
            blk = W[self._dproj.post, self._dproj.pre]
            cols = np.arange(blk.shape[0])
            for i in pre_sp:                                            ##in spike order, as D&C
                d = 0 if self._dend_of_pre is None else int(self._dend_of_pre[i])
                np.add.at(self.Buf, ((self.Cur[0] + self.K[:, i]) % self._R, d, cols), blk[:, i].astype(np.float64))
        self.Cur[0] = (self.Cur[0] + 1) % self._R


   ######################################################################################
##########################   Threshold: D&C adaptive threshold + one spike per step   #####
   ######################################################################################

@dataclass(frozen=True, slots=True)
class ZhangThresholdCfg(DCThresholdCfg):
    """DCThresholdCfg + Fig. 2's rule: of the exc neurons that reach threshold in the same step, one spikes.
    Args:
        one_spike = True: at most one EXC spike per step (inh rows are not limited)
        one_spike_select = INFERRED "random": uniformly among the candidates, keyed rng("zhang", "one_spike") (as BindsNET's one_spike) |
            "max_v": the candidate with the largest V - threshold (ties: lowest row)
        one_spike_losers = what happens to the candidates NOT chosen: "reset" (default, BindsNET's order): reset, refractory and theta += theta_plus
            like the winner, only the spike is dropped | "keep": they keep V and theta and may fire on the next step.
    """
    one_spike: bool = True
    one_spike_select: str = "random"
    one_spike_losers: str = "reset"
    cfg_type: str = field(default="ZhangThresholdCfg", kw_only=True)


class ZhangOneSpikeThreshold(DCAdaptiveThreshold):
    """
    DCAdaptiveThreshold with one insertion: when more than one exc row passes, one is kept as THE spike and the others are cleared from
    rec_spikes. Under one_spike_losers="reset" the cleared rows are still reset here and their theta grows. Active in every stage (test included).
    The "random" generator restarts after a resume, so a resumed run is not bit-identical to an uninterrupted one.
    """

    def __init__(self, cfg: ZhangThresholdCfg, sim_cfg: SimulationCfg):
        super().__init__(cfg, sim_cfg)
        if cfg.one_spike_select not in ("random", "max_v"):
            raise ValueError(f"ZhangOneSpikeThreshold: one_spike_select must be 'random' or 'max_v', got {cfg.one_spike_select!r}.")
        if cfg.one_spike_losers not in ("reset", "keep"):
            raise ValueError(f"ZhangOneSpikeThreshold: one_spike_losers must be 'reset' or 'keep', got {cfg.one_spike_losers!r}.")
        self.one_spike = bool(cfg.one_spike)
        self.one_spike_select = cfg.one_spike_select
        self.one_spike_losers = cfg.one_spike_losers
        self._gen = None

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        ex = self._exc_rows
        if not self._frozen:
            self.Theta[ex] += (-self.Theta[ex] / self.tau_th) * self.dt
        self._thr[ex] = (self.Theta[ex] - self.offset) + self.v_th
        V = neur.V.astype(np.float64)
        spikes = (V > self._thr) & (neur.Since >= self._first_allowed)
        adapt = spikes & ex                                                 ##the rows whose theta grows (all crossers under "reset")
        if self.one_spike:
            cand = np.flatnonzero(adapt)
            if cand.size > 1:
                if self.one_spike_select == "max_v":
                    keep = cand[int(np.argmax(V[cand] - self._thr[cand]))]
                else:
                    if self._gen is None:
                        self._gen = ctx.rng.rng("zhang", "one_spike")
                    keep = cand[int(self._gen.integers(cand.size))]
                spikes[cand] = False
                spikes[keep] = True
                if self.one_spike_losers == "reset":
                    ##the orchestra resets only rec_spikes rows, so the losers are reset here (Since = 0: refractory like a spike)
                    losers = cand[cand != keep]
                    neur.V[losers] = self.v_reset[losers - (neur.V.shape[0] - self.v_reset.shape[0])]
                    neur.Since[losers] = 0
                else:
                    adapt = spikes & ex                                     ##"keep": only the winner adapts
        upd.rec_spikes[...] = spikes
        if not self._frozen:
            self.Theta[adapt] += self.adpt_th
