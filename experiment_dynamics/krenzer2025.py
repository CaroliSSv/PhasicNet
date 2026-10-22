#krenzer2025 - Krenzer & Bogdan (2025), "Reinforced liquid state machines: new training strategies for spiking neural networks based on
# reinforcements", Front. Comput. Neurosci. 19:1569374 -- the reward-modulated STDP of the liquid -> winner-takes-all synapses (Eq. 3-9) as a
# PhasicNet plasticity module (dense + sparse). The network and the protocol are in experiments/krenzer_2025.py.
#
#EQUATIONS (§3, Tables 1-2):
#   Eq. 3    dx_pre/dt = -x_pre / tau_pre + sum_f delta(t - t_f)                       tau_pre 300 ms (one trace per PRESYNAPTIC neuron)
#   Eq. 4    on a POST spike:  dw_local = eta (x_pre - x_tar) (w_max - w)^mu           eta 1e-4, w_max 1, x_tar 20, mu "set low" (no value)
#   Eq. 5    synaptic scaling after the update: w_ij <- w_ij / sum_j w_ij * alpha        alpha 15
#   Eq. 6    de/dt = -e / tau_trace + dw_local                                           tau_trace 500 ms (per synapse)
#   Eq. 7-9  after each sample: correct -> dw = k e;  wrong -> 0 (forgiving) | -gamma w (strict)
# Values the paper does not give are INFERRED cfg fields (RSTDPCfg). THIS IS NOT A REPLICATION OF KRENZER & BOGDAN'S MODEL: their synapse model
# and most of the network's numbers are not in the paper, and no code is published.
#
#ORDER PER STEP: X and e decay (exact exp), pre spikes enter X; for each WTA neuron that spiked: dw_local on its row, clip to [0, w_max],
# Eq. 5, w_max again as a hard bound; e += dw_local (unscaled). apply_feedback() is called by the protocol between samples.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np

from cfg import SimulationCfg, PlasticityCfg
from states import RunContext, SynapticState
from dynamics import Plasticity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RSTDPCfg(PlasticityCfg):
    """PlasticityCfg with the paper's meanings: tau_pretrace = tau_pre (300), A_ltp = eta (1e-4), w_max = w_max (1). tau_posttrace, A_ltd, eta_exc are
    unused: pass tau_posttrace=1.0, A_ltd=0.0, eta_exc=1.0. Leave w_min None: this module clips its own block.
    Args:
        syn_group = the R-STDP projection (liquid 1 exc -> WTA exc), the ONLY one that learns (freeze the others by name)
        x_tar = Eq. 4 target of the presynaptic trace (20)
        mu = Eq. 4 weight dependence (INFERRED 0.1: "set low")
        alpha = Eq. 5 sum of a WTA neuron's incoming weights (15)
        tau_trace = ms, Eq. 6 eligibility decay (500)
        feedback = "forgiving" (Eq. 8) | "strict" (Eq. 9) | "none" (local rule only, the paper's baseline)
        k = Eq. 8/9 scale of e (INFERRED 1.0). None = w_max / max |e| (the literal reading; drains W_ex unless scale_after_feedback).
        gamma = Eq. 9 decay when wrong (INFERRED 1e-6: "gamma << eta")
        scale_after_feedback = INFERRED False: no Eq. 5 after the feedback update (it would undo the strict decay exactly). True = scale there too.
    """
    syn_group: str = "exc1_wta"
    x_tar: float = 20.0
    mu: float = 0.1
    alpha: float = 15.0
    tau_trace: float = 500.0
    feedback: str = "forgiving"
    k: Optional[float] = 1.0
    gamma: float = 1e-6
    scale_after_feedback: bool = False
    cfg_type: str = field(default="RSTDPCfg", kw_only=True)


def scale_rows_to(W_block: np.ndarray, alpha: float) -> None:
    """Eq. 5 IN PLACE on a (n_post, n_pre) block: every row's sum -> alpha (all-zero rows stay zero). Also used for initialisation."""
    s = W_block.sum(axis=1, keepdims=True, dtype=np.float64)
    s[s == 0] = alpha                                                   ##no division by zero: an all-zero row stays all-zero
    W_block *= (alpha / s).astype(W_block.dtype)


class RSTDPEligibility(Plasticity):
    """
    Dense form. State: X (n_total,) presynaptic traces (Eq. 3), E (n_rows, n_cols) eligibility of the R-STDP block (Eq. 6); float64,
    transient (cleared per sample; INFERRED). Writes dW on the block only.
    """
    weights_format = "dense"
    writes = ("dW",)
    requires = ("pre_spike", "post_spike")

    def __init__(self, cfg: RSTDPCfg, sim_cfg: SimulationCfg):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("RSTDPEligibility requires RSTDPCfg and SimulationCfg with dt.")
        if cfg.tau_pretrace <= 0 or cfg.tau_trace <= 0:
            raise ValueError("RSTDPEligibility requires tau_pretrace > 0 and tau_trace > 0.")
        if cfg.w_max is None or cfg.w_max <= 0:
            raise ValueError("RSTDPEligibility requires w_max > 0 (Table 2: 1).")
        if cfg.feedback not in ("forgiving", "strict", "none"):
            raise ValueError(f"RSTDPEligibility: feedback must be 'forgiving', 'strict' or 'none', got {cfg.feedback!r}.")
        if cfg.mu < 0 or cfg.alpha <= 0:
            raise ValueError("RSTDPEligibility requires mu >= 0 and alpha > 0.")
        self.syn_group = cfg.syn_group
        self.tau_pretrace = float(cfg.tau_pretrace)
        self.tau_posttrace = float(cfg.tau_posttrace)       ##unused, kept for the cfg contract
        self.A_ltp = float(cfg.A_ltp)                       ##eta
        self.A_ltd = float(cfg.A_ltd)                       ##unused
        self.eta_exc = float(cfg.eta_exc)                   ##unused (1.0)
        self.eta_inh = cfg.eta_inh
        self.w_min = None if cfg.w_min is None else np.float32(cfg.w_min)    ##None: the engine's clamp stays off; this module clips its block
        self.w_max = float(cfg.w_max)
        self.x_tar = float(cfg.x_tar)
        self.mu = float(cfg.mu)
        self.alpha = float(cfg.alpha)
        self.tau_trace = float(cfg.tau_trace)
        self.feedback = cfg.feedback
        self.k = None if cfg.k is None else float(cfg.k)
        self.gamma = float(cfg.gamma)
        self.scale_after_feedback = bool(cfg.scale_after_feedback)
        self.freeze = tuple(cfg.freeze)
        self.dt = float(sim_cfg.dt)
        self._decay_pre = float(np.exp(-self.dt / self.tau_pretrace))
        self._decay_trace = float(np.exp(-self.dt / self.tau_trace))
        self.X = None               ##(n_total,) presynaptic traces
        self.E = None               ##(n_rows, n_cols) eligibility of the block
        self._proj = None
        self._touched = np.zeros(0, dtype=np.int64)

    def configure_for_network(self, net) -> None:
        if self.syn_group not in net.synapseslices:
            raise RuntimeError(f"RSTDPEligibility: syn_group={self.syn_group!r} is not a synapse group of this network.")
        self._proj = proj = net.synapseslices[self.syn_group]
        self.X = np.zeros(net.n_total, dtype=np.float64)
        self.E = np.zeros((proj.post.stop - proj.post.start, proj.pre.stop - proj.pre.start), dtype=np.float64)
        self._touched = np.zeros(0, dtype=np.int64)

    def _local(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Eq. 4 for the rows W (k, n_pre) given the pre traces x (n_pre,): eta (x - x_tar) (w_max - w)^mu, the soft bound floored at 0."""
        return self.A_ltp * (x - self.x_tar)[None, :] * np.maximum(self.w_max - W, 0.0) ** self.mu

    def _traces(self, neur) -> None:
        self.X *= self._decay_pre
        self.X[neur.pre_spike > 0] += 1.0
        self.E *= self._decay_trace

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        proj = self._proj
        dW = upd.dW[proj.post, proj.pre]
        if self._touched.size:
            dW[self._touched, :] = 0.0                                   ##dW persists between steps: clear last step's rows
        self._traces(neur)
        post_sp = np.flatnonzero(neur.post_spike[proj.post] > 0)
        if post_sp.size:
            W = syn.W[proj.post, proj.pre][post_sp].astype(np.float64)
            dloc = self._local(W, self.X[proj.pre])
            new = np.minimum(np.maximum(W + dloc, 0.0), self.w_max)     ##clip to [0, w_max] ...
            scale_rows_to(new, self.alpha)                               ##... then Eq. 5 on the updated rows ...
            np.minimum(new, self.w_max, out=new)                         ##... and w_max stays a hard bound
            dW[post_sp, :] = new - W
            self.E[post_sp, :] += dloc                                   ##Eq. 6: the local change enters the eligibility trace
        self._touched = post_sp

    def _k(self) -> float:
        if self.k is not None:
            return self.k
        m = float(np.abs(self.E).max()) if self.E.size else 0.0
        return self.w_max / m if m > 0 else 0.0

    def apply_feedback(self, ctx: RunContext, correct: bool) -> dict:
        """Eq. 7-9 after a sample, written into W directly: correct -> W += k e; wrong -> strict: W -= gamma W, forgiving: nothing.
        Then clip to [0, w_max] (+ Eq. 5 if scale_after_feedback). Returns {k, mean_abs_dw} for logging."""
        syn, proj = ctx.syn, self._proj
        blk = syn.W[proj.post, proj.pre]
        W = blk.astype(np.float64)
        k = 0.0
        if self.feedback == "none":
            return {"k": 0.0, "mean_abs_dw": 0.0}
        if correct:
            k = self._k()
            new = W + k * self.E
        elif self.feedback == "strict":
            new = W - self.gamma * W
        else:
            return {"k": 0.0, "mean_abs_dw": 0.0}
        new = np.minimum(np.maximum(new, 0.0), self.w_max)
        if self.scale_after_feedback:
            scale_rows_to(new, self.alpha)
            np.minimum(new, self.w_max, out=new)
        blk[...] = new.astype(blk.dtype)
        return {"k": k, "mean_abs_dw": float(np.abs(new - W).mean())}


class RSTDPEligibilitySparse(RSTDPEligibility):
    """Sparse form: E per edge ((n_syn,), zero outside the R-STDP projection); the local update runs over the edges of the WTA neurons
    that spiked, Eq. 5 by bincount. step() returns the per-edge dw."""
    weights_format = "sparse"

    def configure_for_network(self, net) -> None:
        super().configure_for_network(net)
        self.E = None               ##(n_syn,), allocated at the first step
        self._ids = None            ##the projection's edge ids (storage order)

    def _resolve(self, syn: SynapticState) -> None:
        proj = self._proj
        m = (syn.post >= proj.post.start) & (syn.post < proj.post.stop) & (syn.pre >= proj.pre.start) & (syn.pre < proj.pre.stop)
        self._ids = np.flatnonzero(m)
        if self.E is None or self.E.shape != syn.W.shape:
            self.E = np.zeros(syn.W.shape[0], dtype=np.float64)

    def _scaled(self, syn: SynapticState, ids: np.ndarray, new: np.ndarray) -> np.ndarray:
        """Eq. 5 over the rows of the edges `ids`: `new` scaled so each row sums to alpha. `new` must hold every projection edge of those rows."""
        rows = syn.post[ids] - self._proj.post.start
        s = np.bincount(rows, weights=new, minlength=self.E_rows)
        s[s == 0] = self.alpha
        return new * (self.alpha / s)[rows]

    @property
    def E_rows(self) -> int:
        return self._proj.post.stop - self._proj.post.start

    def step(self, ctx: RunContext) -> np.ndarray:
        neur, syn = ctx.neur, ctx.syn
        if self._ids is None or self.E.shape != syn.W.shape:
            self._resolve(syn)
        proj = self._proj
        self.X *= self._decay_pre
        self.X[neur.pre_spike > 0] += 1.0
        self.E *= self._decay_trace
        dw = np.zeros(syn.W.shape[0], dtype=np.float32)
        rows = np.flatnonzero(neur.post_spike > 0)
        rows = rows[(rows >= proj.post.start) & (rows < proj.post.stop)]
        if rows.size:
            ids = syn.weights.index.synIDs_by_post(rows)
            ids = ids[(syn.pre[ids] >= proj.pre.start) & (syn.pre[ids] < proj.pre.stop)]
            if ids.size:
                W = syn.W[ids].astype(np.float64)
                dloc = self.A_ltp * (self.X[syn.pre[ids]] - self.x_tar) * np.maximum(self.w_max - W, 0.0) ** self.mu
                new = np.minimum(self._scaled(syn, ids, np.minimum(np.maximum(W + dloc, 0.0), self.w_max)), self.w_max)
                dw[ids] = new - W
                self.E[ids] += dloc
        if syn.plastic_mask is not None:
            np.multiply(dw, syn.plastic_mask, out=dw)
        return dw

    def apply_feedback(self, ctx: RunContext, correct: bool) -> dict:
        syn = ctx.syn
        if self._ids is None or self.E is None or self.E.shape != syn.W.shape:
            self._resolve(syn)
        ids = self._ids
        W = syn.W[ids].astype(np.float64)
        k = 0.0
        if self.feedback == "none":
            return {"k": 0.0, "mean_abs_dw": 0.0}
        if correct:
            k = self._k()
            new = W + k * self.E[ids]
        elif self.feedback == "strict":
            new = W - self.gamma * W
        else:
            return {"k": 0.0, "mean_abs_dw": 0.0}
        new = np.minimum(np.maximum(new, 0.0), self.w_max)
        if self.scale_after_feedback:
            new = np.minimum(self._scaled(syn, ids, new), self.w_max)
        syn.W[ids] = new.astype(syn.W.dtype)
        return {"k": k, "mean_abs_dw": float(np.abs(new - W).mean())}

    def _k(self) -> float:
        if self.k is not None:
            return self.k
        m = float(np.abs(self.E[self._ids]).max()) if self._ids is not None and self._ids.size else 0.0
        return self.w_max / m if m > 0 else 0.0
