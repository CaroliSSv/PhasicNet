#microcol_mozafari - DYNAMICS of the layered microcolumn experiment, variant B: Mozafari, Ganjtabesh, Nowzari-Dalini, Thorpe & Masquelier (2019),
# "Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks", Pattern Recognition
# 94:87-95 (arXiv:1804.00227), the layer-wise trained spiking network closest to the microcolumn design with a complete parameter set (paper
# Tables 1-3 + the first author's SpykeTorch code, MozafariDeep.py). Predecessor with the same L1/L2 values: Kheradpisheh et al. 2018, Neural
# Networks 99:56 (SVM readout instead of R-STDP).
#
#THE MODEL (one step = one latency bin; the paper's time is ordinal):
#   synapse     original_dynamics.DeltaCurrentSparse: I = sum W over the spikes that arrive this step
#   IFIntegrator   non-leaky integrate-and-fire, dV = r I (paper Eq. 1)
#   GroupThreshold one threshold per layer (Table 1: L1 15, L2 10; the decision layer None = never fires, its potential is the output);
#               at most ONE spike per neuron per sample (the authors' cumulative spike waves: a neuron's first spike is its only one)
#               + the competition (sf.pointwise_inhibition): in mode "train" each COLUMN of a wta group whose input is plastic fires once per
#               sample, the earliest neuron, ties -> the highest potential; the column is then silent for the sample. Frozen layers and mode
#               "eval": no competition. ##authors choice> mozafari NOT followed here: the authors also inhibit the frozen layers, but their spikes
#               keep counting for the rest of the sample and ours count once, so L1 starved L2 (L2 never fired in training, R-STDP 0% hits)
#   MozafariSTDP   first-spike STDP (snn.STDP): dW = (a+ if the pre spike ARRIVED by then, else a-) W (1 - W), on the k WINNERS of a sample only
#               (sf.get_k_winners, cfg kwta): the earliest spikes of the layer, ties -> highest potential, k = 5 / 8 per sample and a winner
#               blocks the columns within radius 3 / 1 (L1 / L2). ##authors choice> mozafari (MozafariDeep.py k1 r1 k2 r2)
#               a+ doubles every 500 samples up to 0.15, a- = 0.75 a+ (MozafariDeep.py; schedule()); R-STDP on the decision layer after each
#               sample (apply_reward, called by the protocol): reward = (a_r+, a_r-), punishment = (a_p-, a_p+), no stabiliser, W in [0.2, 0.8]
#
#COLUMNS INSTEAD OF CONVOLUTION (decided here, documented in the network file): no weight sharing (each column learns its own features). The
# authors' k-winners rule is kept as it is (k winners per sample, radius in columns); its "the winner's feature is excluded elsewhere" part has no
# counterpart (another column's same slot is another neuron). The per-position competition (pointwise inhibition) is the per-column competition
# above. Pooling layers are folded into the column receptive fields.
#"ARRIVED": input spikes reach L1 in the same step, a layer's spikes reach the next layer one step later (PhasicNet's delta synapse). The paper
# computes every layer in the same time bin, so "pre before or with post" there = "pre arrived by the post spike" here (inputs inclusive, layers
# exclusive of the current step). Sparse weights + update_on "hybrid_event" only (Mozafari's layers are ~10^3 x 10^4; dense is not useful).
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging

import numpy as np
from dynamicscfg import SimulationCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg
from states import RunContext, NEVER_SPIKED
from dynamics import Integrator, Threshold, Plasticity
from phasicnet import RuntimeClasses
from original_dynamics import DeltaCurrentSparse, TestNoise
from phases import DefaultPhase

logger = logging.getLogger(__name__)


   ######################################################################################
##########################   Integrator: non-leaky IF   ###################################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class IFIntegratorCfg(IntegratorCfg):
    """Non-leaky integrate-and-fire (paper Eq. 1): v_rest 0, r 1 (W is the potential jump per spike). tau_mem unused (no leak)."""
    tau_mem: Optional[float] = None
    v_rest: float = 0.0
    r: float = 1.0


class IFIntegrator(Integrator):
    """dV = r I. V is reset to v_rest by reset_transient before every sample."""
    writes = ("dV",)
    requires = ()

    def __init__(self, cfg: IFIntegratorCfg, sim_cfg: SimulationCfg = None):
        if cfg is None:
            raise ValueError("IFIntegrator requires IFIntegratorCfg.")
        self.tau_mem = None
        self.v_rest = np.float32(cfg.v_rest)
        self.r = np.float32(cfg.r)

    def step(self, ctx: RunContext) -> None:
        np.multiply(ctx.upd.I2V, self.r, out=ctx.upd.dV)


   ######################################################################################
##########################   Threshold: per layer, one spike, column competition   #######
   ######################################################################################

@dataclass(frozen=True, slots=True)
class GroupThresholdCfg(ThresholdCfg):
    """ThresholdCfg (v_th = the default for groups not listed, v_reset) +
    Args:
        group_v_th = {group: v_th}; None = the group never fires (the decision layer: its potential is read, capture.membrane)
        one_spike = a neuron fires at most once per sample (needs the reset before every sample)
        wta_groups = groups with the per-column competition; wta_dims = how many leading grid coordinates are the column (builder kernel "column")
    """
    group_v_th: dict = field(default_factory=dict)
    one_spike: bool = True
    wta_groups: Tuple[str, ...] = ()
    wta_dims: int = 2


class GroupThreshold(Threshold):
    """Spike iff V >= the group's threshold (and, one_spike, the neuron has not fired this sample). In mode "train", in a wta group whose incoming
    projections are not all frozen: per column only the earliest neuron fires (ties -> highest V) and the column stays silent for the rest of the
    sample (Won, transient). Mode "eval": no competition. The column of a neuron = its first wta_dims grid coordinates (net.positions)."""
    writes = ("rec_spikes",)
    requires = ("Since",)

    def __init__(self, cfg: GroupThresholdCfg, sim_cfg: SimulationCfg = None):
        if cfg is None:
            raise ValueError("GroupThreshold requires GroupThresholdCfg.")
        self.v_th = np.float32(cfg.v_th)
        self.v_reset = np.float32(cfg.v_reset)
        self.group_v_th = dict(cfg.group_v_th)
        self.one_spike = bool(cfg.one_spike)
        self.wta_groups = tuple(cfg.wta_groups)
        self.wta_dims = int(cfg.wta_dims)
        self.Won = None             ##(columns of every wta group,) bool: the column has fired this sample
        self._V_th = None
        self._wta = []              ##per wta group: (rows, column id per neuron, offset into Won, incoming synapse groups)

    def configure_for_network(self, net) -> None:
        self._V_th = np.full(net.n_post, self.v_th, dtype=np.float32)
        for g, v in self.group_v_th.items():
            self._V_th[net.neuronrows[g]] = np.inf if v is None else v
        offset = 0
        for g in self.wta_groups:
            pos = np.asarray(net.positions[g])[:, :self.wta_dims].astype(np.int64)
            col = np.ravel_multi_index(pos.T, tuple(pos.max(axis=0) + 1))
            self._wta.append((net.neuronrows[g], col, offset, tuple(s for s, spec in net.synapsespecs.items() if spec.post_neuron == g)))
            offset += int(col.max()) + 1
        self.Won = np.zeros(offset, dtype=bool)

    def step(self, ctx: RunContext) -> None:
        upd, neur = ctx.upd, ctx.neur
        np.greater_equal(neur.V, self._V_th, out=upd.rec_spikes)
        if self.one_spike:
            upd.rec_spikes &= neur.Since >= NEVER_SPIKED
        if ctx.sim.mode != "train" or not self._wta:
            return
        frozen = ctx.syn.frozen or {}
        for rows, col, off, incoming in self._wta:
            if "all" in frozen or all(s in frozen for s in incoming):          ##frozen layers do not compete (header)
                continue
            cand = np.flatnonzero(upd.rec_spikes[rows])
            if not cand.size:
                continue
            upd.rec_spikes[rows] = False
            c = col[cand]
            free = ~self.Won[off + c]
            cand, c = cand[free], c[free]
            if cand.size:
                order = np.lexsort((-neur.V[rows][cand], c))                    ##by column, then highest V first
                first = np.concatenate(([True], c[order][1:] != c[order][:-1]))
                win = cand[order][first]
                upd.rec_spikes[rows.start + win] = True
                self.Won[off + col[win]] = True


   ######################################################################################
##########################   Plasticity: first-spike STDP + R-STDP   ######################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class MozafariSTDPCfg(PlasticityCfg):
    """PlasticityCfg mapped onto Table 3 + MozafariDeep.py. A_ltp / A_ltd = a+ / |a-| of the STDP layers (0.004 / 0.003), w_min / w_max = the
    stabiliser bounds [0, 1]. tau_pretrace / tau_posttrace / eta_* unused (no traces: the rule reads first-spike ORDER, not timing).
    Args:
        rate_double_every, A_ltp_max, ltd_ratio = a+ x 2 every this many samples of a layer's training, up to A_ltp_max; then a- = ltd_ratio a+
        stabilizer = multiply by (W - w_min)(w_max - W)
        decision_group = the R-STDP layer; Ar_ltp, Ar_ltd = reward (a_r+, |a_r-|); Ap_ltp, Ap_ltd = punishment (a_p+, |a_p-|); decision_w_min / _max = its bounds
        kwta = {group: [k, radius]}: per sample only the k earliest-spiking neurons of the group update (ties -> highest potential), each blocking
               the columns within `radius` of its own (##authors choice> mozafari: k1 5 r1 3, k2 8 r2 1). {} = every spike of a learning neuron updates
    """
    tau_pretrace: float = 0.0
    tau_posttrace: float = 0.0
    A_ltp: float = 0.004
    A_ltd: float = 0.003
    eta_exc: float = 1.0
    w_min: Optional[float] = 0.0
    w_max: Optional[float] = 1.0
    rate_double_every: int = 500
    A_ltp_max: float = 0.15
    ltd_ratio: float = 0.75
    stabilizer: bool = True
    decision_group: Optional[str] = "L3_exc"
    Ar_ltp: float = 0.004
    Ar_ltd: float = 0.003
    Ap_ltp: float = 0.0005
    Ap_ltd: float = 0.004
    decision_w_min: float = 0.2
    decision_w_max: float = 0.8
    kwta: dict = field(default_factory=lambda: {"L1_exc": [5, 3], "L2_exc": [8, 1]})      ##authors choice> mozafari


class MozafariSTDP(Plasticity):
    """
    STDP (layers 1, 2): at each spike of a neuron whose incoming edges are plastic AND that is one of the sample's k winners (kwta; Vacc = the
    potential at the spike, Nwin / Blocked = winners so far and the blocked columns, all transient), on ALL its plastic incoming edges:
        dW = (a+ if Arrived[pre] else -a-) * (W - w_min)(w_max - W)          Arrived = the pre spike has reached the post neuron this sample
    R-STDP (decision layer, apply_reward): on the winner's plastic incoming edges, phi = the protocol's adaptive factor:
        reward  dW = phi (a_r+ if Arrived else -a_r-)        punishment  dW = phi (-a_p- if Arrived else +a_p+)        clipped to [0.2, 0.8]
    Returns (k, dw_k) as the hybrid_event modules do (Run.step_sparse commits and clips). Arrived is transient (reset every sample).
    """
    weights_format = "sparse"
    writes = ("dW",)
    requires = ("pre_spike", "post_spike")

    def __init__(self, cfg: MozafariSTDPCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None:
            raise ValueError("MozafariSTDP requires MozafariSTDPCfg and SimulationCfg.")
        if sim_cfg.update_on != "hybrid_event":
            raise ValueError(f"MozafariSTDP runs with weights_format 'sparse' + update_on 'hybrid_event', got {sim_cfg.update_on!r}.")
        for name in ("A_ltp", "A_ltd", "A_ltp_max", "ltd_ratio", "rate_double_every", "stabilizer", "decision_group", "Ar_ltp", "Ar_ltd", "Ap_ltp",
                     "Ap_ltd", "decision_w_min", "decision_w_max", "w_min", "w_max", "eta_exc"):
            setattr(self, name, getattr(cfg, name))
        self.freeze = tuple(cfg.freeze)
        self.clamp_bounds = {}          ##Run._plan_clamp hook: the decision projection's bounds
        self.Arrived = None
        self.kwta = {g: (int(k), int(r)) for g, (k, r) in dict(cfg.kwta).items()}
        self.Vacc = self.Nwin = self.Blocked = None     ##k-WTA state (##authors choice> mozafari): V at the spike, winners so far, blocked columns
        self._kwta = []                                 ##per kwta group: (rows, k, column id per row, offset into Blocked, neighbours per column)
        self._n_input = 0
        self._ap, self._an = float(cfg.A_ltp), float(cfg.A_ltd)
        self._frozen_seen, self._row_plastic = None, None     ##rows with plastic incoming edges, for the current syn.frozen

    def configure_for_network(self, net) -> None:
        self._n_input = net.n_input
        self.Arrived = np.zeros(net.n_total, dtype=np.float32)
        self.clamp_bounds = {s: (self.decision_w_min, self.decision_w_max) for s, spec in net.synapsespecs.items() if spec.post_neuron == self.decision_group}
        self.Vacc = np.zeros(net.n_post, dtype=np.float32)
        self.Nwin = np.zeros(max(len(self.kwta), 1), dtype=np.int32)
        offset = 0
        for g, (k, r) in self.kwta.items():                   ##the column grid of each kwta group and, per column, the columns within the radius
            if g not in net.neuronrows:
                raise ValueError(f"kwta group {g!r} is not a neuron group of the network ({tuple(net.neuronrows)}).")
            pos = np.asarray(net.positions[g])[:, :-1].astype(np.int64)          ##grid (x, y, slot): all but the slot name the column
            shape = tuple(pos.max(axis=0) + 1)
            col = np.ravel_multi_index(pos.T, shape)
            grid = np.indices(shape).reshape(len(shape), -1).T                   ##every column's coordinates
            neigh = [np.flatnonzero(np.all(np.abs(grid - c) <= r, axis=1)) for c in grid]
            self._kwta.append((net.neuronrows[g], k, col, offset, neigh))
            offset += int(np.prod(shape))
        self.Blocked = np.zeros(max(offset, 1), dtype=bool)

    def schedule(self, j: int) -> None:
        """The learning rates for a layer's j-th training sample (0-based, counted over its epochs): MozafariDeep.py doubles a+ when its counter reaches
        rate_double_every (a- = ltd_ratio a+ from then on), capped at A_ltp_max."""
        n = (int(j) + 1) // int(self.rate_double_every)
        self._ap = min(self.A_ltp * 2.0 ** n, self.A_ltp_max) if n else self.A_ltp
        self._an = self.ltd_ratio * self._ap if n else self.A_ltd

    def _incoming(self, syn, rows: np.ndarray) -> np.ndarray:
        k = syn.weights.index.synIDs_by_post(rows)
        return k[syn.plastic_mask[k]] if syn.plastic_mask is not None else k

    def _plastic_rows(self, syn) -> np.ndarray:
        """bool per W row: the row has plastic incoming edges. Recomputed when syn.frozen is replaced (Run.set_frozen), so the spikes of frozen
        layers cost nothing here."""
        if syn.frozen is not self._frozen_seen:
            self._row_plastic = np.zeros(syn.weights.n_post, dtype=bool)
            self._row_plastic[syn.post if syn.plastic_mask is None else syn.post[syn.plastic_mask]] = True
            self._frozen_seen = syn.frozen
        return self._row_plastic

    def winners(self, rows: np.ndarray) -> np.ndarray:
        """##authors choice> mozafari (sf.get_k_winners): of this step's learning spikes `rows`, keep those in a kwta group only if the group has
        fewer than k winners this sample and their column is not blocked; highest Vacc first; each winner blocks its radius. Earlier steps won
        earlier, so the k winners are the k earliest spikes (ties -> highest potential). Rows of groups without kwta pass unchanged."""
        keep = np.ones(rows.size, dtype=bool)
        for gi, (grp, k, col, off, neigh) in enumerate(self._kwta):
            sel = np.flatnonzero((rows >= grp.start) & (rows < grp.stop))
            if not sel.size:
                continue
            keep[sel] = False
            if self.Nwin[gi] >= k:
                continue
            for j in sel[np.argsort(-self.Vacc[rows[sel]], kind="stable")]:
                c = col[rows[j] - grp.start]
                if self.Blocked[off + c]:
                    continue
                keep[j] = True
                self.Nwin[gi] += 1
                self.Blocked[off + neigh[c]] = True
                if self.Nwin[gi] >= k:
                    break
        return rows[keep]

    def step(self, ctx: RunContext):
        neur, syn, n0, A = ctx.neur, ctx.syn, self._n_input, self.Arrived
        np.add(self.Vacc, ctx.upd.I2V, out=self.Vacc)                         ##= V before this step's reset (IFIntegrator r 1, no leak): the tie-break
        np.maximum(A[:n0], neur.pre_spike[:n0], out=A[:n0])                   ##input spikes arrive this step
        rows = np.flatnonzero(neur.post_spike)
        rows = rows[self._plastic_rows(syn)[rows]]
        if rows.size and self._kwta:
            rows = self.winners(rows)
        k = self._incoming(syn, rows) if rows.size else np.zeros(0, dtype=np.int64)
        if k.size:
            h = A[syn.pre[k]]
            dw = self._ap * h - self._an * (1.0 - h)
            if self.stabilizer:
                w = syn.W[k]
                dw *= (w - self.w_min) * (self.w_max - w)
        np.maximum(A[n0:], neur.pre_spike[n0:], out=A[n0:])                   ##layer spikes arrive next step
        return k, (dw.astype(np.float32) if k.size else np.zeros(0, dtype=np.float32))

    def apply_reward(self, ctx: RunContext, row: int, reward: bool, phi: float) -> int:
        """R-STDP on the plastic incoming edges of the decision neuron `row` (a W row index), after its sample. Mode "train" only. Returns the edge count."""
        if ctx.sim.mode != "train":
            raise RuntimeError("apply_reward writes W: mode 'train' only.")
        syn = ctx.syn
        k = self._incoming(syn, np.array([row]))
        h = self.Arrived[syn.pre[k]]
        dw = phi * (self.Ar_ltp * h - self.Ar_ltd * (1.0 - h)) if reward else phi * (self.Ap_ltp * (1.0 - h) - self.Ap_ltd * h)
        syn.W[k] = np.clip(syn.W[k] + dw.astype(np.float32), self.decision_w_min, self.decision_w_max)
        return int(k.size)

    def bound_decision(self, ctx: RunContext) -> None:
        """Clip the whole decision projection into [decision_w_min, decision_w_max] once, before R-STDP: the authors' first update clamps all of
        conv3 (snn.STDP clamps the layer after every update), which cuts the N(0.8, 0.05) start at 0.8."""
        syn = ctx.syn
        rows = np.arange(ctx.net.neuronrows[self.decision_group].start, ctx.net.neuronrows[self.decision_group].stop)
        k = syn.weights.index.synIDs_by_post(rows)
        syn.W[k] = np.clip(syn.W[k], self.decision_w_min, self.decision_w_max)


MICROCOL_MOZAFARI = RuntimeClasses(synapse_cls=DeltaCurrentSparse, integrator_cls=IFIntegrator, threshold_cls=GroupThreshold, plasticity_cls=MozafariSTDP,
                                   noise_cls=TestNoise, phase_classes=(DefaultPhase,))
