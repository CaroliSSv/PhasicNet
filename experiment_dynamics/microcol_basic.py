#microcol_basic - DYNAMICS of the back-to-basics microcolumn experiment: three layers of LIF neurons in columns (builder.set_columns), a fixed
# threshold per layer with a refractory period, escape noise as the one noise source, soft-bounded STDP in the lower layers and R-STDP on the
# decision layer only.
#
#THE MODEL (one step = dt ms):
#   synapse        original_dynamics.DeltaCurrentSparse: I = sum W over the spikes that arrive this step
#   integrator     original_dynamics.LIFexact, unchanged (tau_mem, v_rest, r)
#   EscapeThreshold   one fixed threshold per group (group_v_th), absolute refractory period (t_refrac, read from neur.Since), and escape noise:
#                  P(spike in a step) = 1 / (1 + exp(-(V - v_th) / width))   (Gerstner & Kistler 2002, ch. 5.3), width = beta x (v_th - v_reset):
#                  beta is the noise width as a FRACTION of the neuron's threshold gap, so one beta means the same for groups with different
#                  thresholds (at rest P = 1 / (1 + exp(1 / beta)) per step: beta 0.1 -> 0.05 Hz, 0.15 -> 1.3 Hz, 0.2 -> 6.7 Hz at dt 1 ms).
#                  beta = 0: no noise, spike iff V >= v_th.
#                  The draws come from the run's keyed streams rng("noise", "threshold:<runstep>"), a new one at the first step after every
#                  reset_transient (= every sample of the protocol), so a resumed run draws what a straight run draws.
#                  Membrane noise stays off (NoiseCfg sigma_mem 0): one noise source.
#   SoftBoundSTDP  pair STDP on per-neuron traces (as original_dynamics.SparseTraceSTDP, hybrid_event) times the soft bound:
#                      dW = (A_ltp * post_spike * Pre_trace - A_ltd * Post_trace * pre_spike) * (W - w_min) * (w_max - W)
#                  R-STDP (decision_group): the same two terms are not applied during the sample but summed per synapse (Elig_ltp, Elig_ltd). After the
#                  sample the protocol calls apply_reward on the neurons of the decided class:
#                      reward      dW = phi * (Ar_ltp * Elig_ltp - Ar_ltd * Elig_ltd) * (W - w_min) * (w_max - W)
#                      punishment  dW = phi * (Ap_ltp * Elig_ltd - Ap_ltd * Elig_ltp) * (W - w_min) * (w_max - W)
#                  (the four rates as Mozafari et al. 2019, on traces instead of first-spike order). No reward call = nothing changes.
#                  normalize (optional, normalize_weights): after a training sample every neuron's plastic incoming weights are scaled back to the sum they were built
#                  with (per-neuron normalisation, Diehl & Cook 2015). Not applied to the decision layer.
#WHICH GROUPS LEARN is the protocol's (Run.set_frozen per stage); inhibition and feedback inhibition are fixed synapse groups of the network file.
# Sparse weights + update_on "hybrid_event" only.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from dynamicscfg import SimulationCfg, ThresholdCfg, PlasticityCfg
from states import RunContext
from dynamics import Threshold, Plasticity
from synapticindex import touched
from phasicnet import RuntimeClasses
from original_dynamics import DeltaCurrentSparse, LIFexact, TestNoise
from phases import DefaultPhase


   ######################################################################################
##########################   Threshold: fixed per group, refractory, escape noise   #######
   ######################################################################################

@dataclass(frozen=True, slots=True)
class EscapeThresholdCfg(ThresholdCfg):
    """ThresholdCfg (v_th = the threshold of every group not listed, v_reset, t_refrac; tau_th / adpt_th unused: no adaptation) +
    Args:
        beta = width of the escape noise as a fraction of the threshold gap (v_th - v_reset) of each neuron. 0 = no noise
        group_v_th = {group: v_th}
    """
    beta: float = 0.0
    group_v_th: dict = field(default_factory=dict)


class EscapeThreshold(Threshold):
    """Spike with probability 1 / (1 + exp(-(V - v_th) / (beta (v_th - v_reset)))) per step (beta 0: iff V >= v_th), never within t_refrac of the
    neuron's last spike."""
    writes = ("rec_spikes",)
    requires = ("Since",)

    def __init__(self, cfg: EscapeThresholdCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("EscapeThreshold requires EscapeThresholdCfg and SimulationCfg with dt.")
        if cfg.beta < 0:
            raise ValueError(f"EscapeThreshold requires beta >= 0, got {cfg.beta}.")
        self.v_th = float(cfg.v_th)
        self.v_reset = np.float32(cfg.v_reset)
        self.beta = float(cfg.beta)
        self.group_v_th = dict(cfg.group_v_th)
        self._refrac_steps = int(np.ceil((cfg.t_refrac or 0.0) / sim_cfg.dt))
        self.t_refrac = self._refrac_steps * float(sim_cfg.dt)       ##the refractory time actually used (whole steps)
        self._V_th = None
        self._width = None                                           ##per neuron: beta x (v_th - v_reset)
        self._rng = None
        self.New_stream = np.ones(1, dtype=bool)                     ##transient state: reset_transient sets it again -> the next step takes a new stream

    def configure_for_network(self, net) -> None:
        self._V_th = np.full(net.n_post, self.v_th, dtype=np.float32)
        for g, v in self.group_v_th.items():
            self._V_th[net.neuronrows[g]] = v
        self._width = (self.beta * (self._V_th - self.v_reset)).astype(np.float32)
        if self.beta > 0 and self._width.min() <= 0:
            raise ValueError(f"EscapeThreshold with beta > 0 requires every threshold above v_reset = {self.v_reset} (v_th {self.v_th}, group_v_th {self.group_v_th}).")

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        if self.beta > 0:
            if self._rng is None or self.New_stream[0]:
                self._rng = ctx.rng.rng("noise", f"threshold:{ctx.sim.current_runstep}")
                self.New_stream[0] = False
            z = np.clip((neur.V - self._V_th) / self._width, -50.0, 50.0)
            np.less(self._rng.random(z.size, dtype=np.float32), 1.0 / (1.0 + np.exp(-z)), out=upd.rec_spikes)
        else:
            np.greater_equal(neur.V, self._V_th, out=upd.rec_spikes)
        if self._refrac_steps:
            upd.rec_spikes &= neur.Since >= self._refrac_steps


   ######################################################################################
##########################   Plasticity: soft-bounded STDP + R-STDP   #####################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class SoftBoundSTDPCfg(PlasticityCfg):
    """PlasticityCfg (tau_pretrace / tau_posttrace, A_ltp / A_ltd, w_min / w_max = the soft bounds; eta_* unused) +
    Args:
        decision_group = the R-STDP layer (None = no R-STDP); Ar_ltp, Ar_ltd = reward rates; Ap_ltp, Ap_ltd = punishment rates
        normalize = per-neuron normalisation of the plastic incoming weights after every training sample (not the decision layer)
    """
    tau_pretrace: float = 20.0
    tau_posttrace: float = 20.0
    A_ltp: float = 0.01
    A_ltd: float = 0.005
    eta_exc: float = 1.0
    w_min: Optional[float] = 0.0
    w_max: Optional[float] = 1.0
    decision_group: Optional[str] = "L3_exc"
    Ar_ltp: float = 0.004
    Ar_ltd: float = 0.003
    Ap_ltp: float = 0.0005
    Ap_ltd: float = 0.004
    normalize: bool = True


class SoftBoundSTDP(Plasticity):
    """Soft-bounded pair STDP on every plastic edge with a spiking end; on the decision group's incoming edges the two terms are summed (Elig_*)
    for apply_reward instead of applied (header). Returns (k, dw_k) as the hybrid_event modules do. Elig_* and the traces are transient."""
    weights_format = "sparse"
    writes = ("dW",)
    requires = ("pre_spike", "post_spike")

    def __init__(self, cfg: SoftBoundSTDPCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None:
            raise ValueError("SoftBoundSTDP requires SoftBoundSTDPCfg and SimulationCfg.")
        if sim_cfg.update_on != "hybrid_event":
            raise ValueError(f"SoftBoundSTDP runs with weights_format 'sparse' + update_on 'hybrid_event', got {sim_cfg.update_on!r}.")
        if cfg.w_min is None or cfg.w_max is None or cfg.tau_pretrace <= 0 or cfg.tau_posttrace <= 0:
            raise ValueError("SoftBoundSTDP requires w_min, w_max and tau_pretrace, tau_posttrace > 0.")
        for name in ("tau_pretrace", "tau_posttrace", "A_ltp", "A_ltd", "eta_exc", "w_min", "w_max", "decision_group", "Ar_ltp", "Ar_ltd",
                     "Ap_ltp", "Ap_ltd", "normalize"):
            setattr(self, name, getattr(cfg, name))
        self._decay_pre = np.float32(np.exp(-sim_cfg.dt / cfg.tau_pretrace))
        self._decay_post = np.float32(np.exp(-sim_cfg.dt / cfg.tau_posttrace))
        self.Pre_trace = self.Post_trace = self.Elig_ltp = self.Elig_ltd = None
        self._W0 = None                 ##the weights as built, in the edge order of syn.W: the sums normalize_weights restores
        self._dec = None                ##bool per edge: an incoming edge of the decision group
        self._norm = (None, None, None, None)       ##(syn.frozen it was resolved for, edges, their post rows, target sum per row)

    def configure_for_network(self, net) -> None:
        self.Pre_trace = np.zeros(net.n_total, dtype=np.float32)
        self.Post_trace = np.zeros(net.n_post, dtype=np.float32)
        self.Elig_ltp = np.zeros(net.n_synapses, dtype=np.float32)
        self.Elig_ltd = np.zeros(net.n_synapses, dtype=np.float32)
        order = np.lexsort((net.pre_idx, net.post_idx))                 ##the edge order of PhasicNet.to_sparse
        self._W0 = np.asarray(net.W0, dtype=np.float32)[order]
        post = np.asarray(net.post_idx)[order]
        rows = net.neuronrows.get(self.decision_group) if self.decision_group else None
        if self.decision_group and rows is None:
            raise ValueError(f"decision_group {self.decision_group!r} is not a neuron group of the network ({tuple(net.neuronrows)}).")
        self._dec = np.zeros(post.size, dtype=bool) if rows is None else (post >= rows.start) & (post < rows.stop)

    def step(self, ctx: RunContext):
        neur, syn = ctx.neur, ctx.syn
        k = touched(syn, neur.post_spike > 0, neur.pre_spike > 0)       ##the plastic edges with a spiking end
        dw = np.zeros(0, dtype=np.float32)
        if k.size:
            pre, post = syn.pre[k], syn.post[k]
            ltp = neur.post_spike[post] * self.Pre_trace[pre]
            ltd = self.Post_trace[post] * neur.pre_spike[pre]
            dec = self._dec[k]
            if dec.any():                                               ##decision edges: remembered for apply_reward, not applied
                self.Elig_ltp[k[dec]] += ltp[dec]
                self.Elig_ltd[k[dec]] += ltd[dec]
                ltp[dec] = 0.0; ltd[dec] = 0.0
            w = syn.W[k]
            dw = ((self.A_ltp * ltp - self.A_ltd * ltd) * (w - self.w_min) * (self.w_max - w)).astype(np.float32, copy=False)
        np.multiply(self.Pre_trace, self._decay_pre, out=self.Pre_trace)        ##the rule reads the traces, then they are updated (as TraceSTDP)
        np.add(self.Pre_trace, neur.pre_spike, out=self.Pre_trace)
        np.multiply(self.Post_trace, self._decay_post, out=self.Post_trace)
        np.add(self.Post_trace, neur.post_spike, out=self.Post_trace)
        return k, dw

    def _incoming(self, syn, rows: np.ndarray) -> np.ndarray:
        k = syn.weights.index.synIDs_by_post(rows)
        return k[syn.plastic_mask[k]] if syn.plastic_mask is not None else k

    def apply_reward(self, ctx: RunContext, rows: np.ndarray, reward: bool, phi: float) -> int:
        """R-STDP on the plastic incoming edges of the decision neurons `rows` (W row indices), after their sample. Mode "train" only.
        Returns the edge count."""
        if ctx.sim.mode != "train":
            raise RuntimeError("apply_reward writes W: mode 'train' only.")
        syn = ctx.syn
        k = self._incoming(syn, np.asarray(rows))
        ep, en, w = self.Elig_ltp[k], self.Elig_ltd[k], syn.W[k]
        dw = phi * (self.Ar_ltp * ep - self.Ar_ltd * en) if reward else phi * (self.Ap_ltp * en - self.Ap_ltd * ep)
        syn.W[k] = np.clip(w + dw * (w - self.w_min) * (self.w_max - w), self.w_min, self.w_max)
        return int(k.size)

    def bound_decision(self, ctx: RunContext) -> None:
        """The decision group's PLASTIC incoming weights into [w_min, w_max] once, before R-STDP (outside them the soft bound pushes away).
        Its fixed incoming weights (inhibition) are not touched."""
        syn = ctx.syn
        k = np.flatnonzero(self._dec if syn.plastic_mask is None else self._dec & syn.plastic_mask)
        syn.W[k] = np.clip(syn.W[k], self.w_min, self.w_max)

    def normalize_weights(self, ctx: RunContext) -> None:
        """Per-neuron normalisation (cfg normalize): every neuron's plastic incoming weights (not the decision group's) x the factor that returns
        their sum to the sum they were built with, then clipped to the bounds. Called by the protocol after a training sample. Mode "train" only."""
        if not self.normalize:
            return
        if ctx.sim.mode != "train":
            raise RuntimeError("normalize_weights writes W: mode 'train' only.")
        syn = ctx.syn
        if self._norm[0] is not syn.frozen:                             ##Run.set_frozen replaced syn.frozen: the plastic edges changed
            k = np.flatnonzero(syn.plastic_mask & ~self._dec)
            post = syn.post[k]
            self._norm = (syn.frozen, k, post, np.bincount(post, weights=self._W0[k], minlength=syn.weights.n_post))
        _, k, post, target = self._norm
        if not k.size:
            return
        total = np.bincount(post, weights=syn.W[k], minlength=target.size)
        scale = np.divide(target, total, out=np.ones_like(total), where=total > 0)
        syn.W[k] = np.clip(syn.W[k] * scale[post], self.w_min, self.w_max)


MICROCOL_BASIC = RuntimeClasses(synapse_cls=DeltaCurrentSparse, integrator_cls=LIFexact, threshold_cls=EscapeThreshold, plasticity_cls=SoftBoundSTDP,
                                noise_cls=TestNoise, phase_classes=(DefaultPhase,))
