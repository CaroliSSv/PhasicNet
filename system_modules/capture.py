#capture = RUNTIME OBSERVABLES: 
# ## = capture a measurement from a RunContext.
 
from __future__ import annotations
from typing import Literal
import logging

import numpy as np

from states import RunContext

logger = logging.getLogger(__name__)

SUMMARY = Literal["mean", "sum", "max", "min"]


   #############################################################################
##########################   Coordinate helpers   ###############################
   ###############################################################################


def post_rows(ctx: RunContext, post_group: str | None) -> slice:
    """A neuron group -> its band of W ROWS. slice(None) selects every row."""
    if post_group is None:
        return slice(None)
    return ctx.net.neuronrows[post_group]


def pre_cols(ctx: RunContext, pre_group: str | None) -> slice:
    """A neuron group -> its band of W COLUMNS. Columns are neuron space and never shift."""
    if pre_group is None:
        return slice(None)
    return ctx.net.neuronslices[pre_group]


def _sparse(ctx: RunContext) -> bool:
    """check format from W itself: 1D = per-edge list, 2D = dense W[post, pre]."""
    return ctx.syn.W.ndim == 1


def _syn_block(ctx: RunContext, post_slc: slice, pre_slc: slice) -> np.ndarray: ##NOTE TODO this looks like duplicate work again.
    """Sparse: bool (n_synapses,) mask of the edges inside the (post rows, pre cols) block, i.e. the edge list's W[post_slc, pre_slc]."""
    syn = ctx.syn
    lo, hi, _ = post_slc.indices(ctx.net.n_post)
    plo, phi, _ = pre_slc.indices(ctx.net.n_total)
    return (syn.post >= lo) & (syn.post < hi) & (syn.pre >= plo) & (syn.pre < phi)


   ###########################################################################
##########################   Spikes   ##################################
   ######################################################################

def spike_indices(ctx: RunContext, *, include_input: bool = False) -> np.ndarray:
    """
    ONE timestep of committed spike state -> true neuron ids that fired.

    NOTE Reads ctx.neur.Spikes (the state that "EXISTS")

    include_input=False starts at the recurrent neurons; NetBuilder orders inputs first, so the
    non-input neurons are exactly Spikes[n_input:].
    """
    start = 0 if include_input else ctx.net.n_input
    spike_i = np.flatnonzero(ctx.neur.Spikes[start:]).astype(np.uint32, copy=False)
    if start:
        spike_i += np.uint32(start)     ##local -> true neuron ids
    return spike_i


   ############################################################################
##########################   Weight statistics   ################################
   ############################################################################

def summarize(values: np.ndarray, summary: SUMMARY | None = None):
    """
    Collapse 1dimension -> 1 value (vector-> float)
    summary=None returns the vector unreduced.
    
    """
    if summary is None:
        return values
    if summary == "mean":
        return float(np.mean(values)) if values.size else 0.0
    if summary == "sum":
        return float(np.sum(values))
    if summary == "max":
        return float(np.max(values)) if values.size else 0.0
    if summary == "min":
        return float(np.min(values)) if values.size else 0.0
    raise ValueError(f"Unknown summary {summary!r}. Expected None, 'mean', 'sum', 'max', or 'min'.")


def sum_globalweight(ctx: RunContext, *, signed: bool = True) -> float:
    """Total weight across the whole matrix.
        Args:
            signed: True preserves sign (positive/negative weights cancel each other). 
                False sums ABSOLUTE values (total global strength). """
    return float(np.sum(ctx.syn.W if signed else np.abs(ctx.syn.W)))


def mean_globalweight(ctx: RunContext, *, per: Literal["synapse", "neuron"] = "synapse", signed: bool = True) -> float:
    """
        per="synapse"  -> Mean weight per synapse (over REALIZED synapses: net.n_synapses, the COO length)
        per="neuron"   ->  Mean weight per neuron.
        signed: True preserves sign (exc/inh can cancel); False averages ABSOLUTE values.
    """
    if per == "synapse":
        n = int(ctx.net.n_synapses)     ##COO length from Network, format-agnostic -- no syn_mask needed
    elif per == "neuron":
        n = int(ctx.net.n_total)
    else:
        raise ValueError(f"Unknown per={per!r}. Expected 'synapse' or 'neuron'.")
    W = ctx.syn.W if signed else np.abs(ctx.syn.W)
    return float(np.sum(W) / n) if n else 0.0


def sum_localweight(ctx: RunContext, axis: int | None = None, *, signed: bool = True):
    """
    Local weight sums.
        axis=None: scalar total
        axis=0:    OUTGOING strength of each neuron (sum across all outgoing synapses: across all post-rows)
        axis=1:    INCOMING strength of each neuron (sum across all incoming synapses: across all pre-cols)
        signed: True preserves sign.
                False sums ABSOLUTE values. 
                NOTE: Irrelevant for axis=0 (one presynaptic neuron = one sign, (assuming still enforced))
    """
    W = ctx.syn.W if signed else np.abs(ctx.syn.W)
    if axis is None or not _sparse(ctx):
        return np.sum(W, axis=axis)
    ##per-edge list summed per pre column (axis=0, (n_total,)) or per post row (axis=1, (n_post,)) (TO ACCEPT same axis ask as dense)
    syn, net = ctx.syn, ctx.net
    if axis == 0:
        return np.bincount(syn.pre, weights=W, minlength=net.n_total).astype(W.dtype, copy=False) ##NOTE TODO is this check neccecary? astype seems excessive.
    return np.bincount(syn.post, weights=W, minlength=net.n_post).astype(W.dtype, copy=False) ##TODO is astype here actually neccecary


def sum_groupweight(ctx: RunContext, syn_group: str) -> float:
    """
    Sum runtime weights for one synapse group.
        net.synapseslices maps syn_group -> (post_slice, pre_slice)
        -> ie. group weights of a specific group contained in W[post_slice, pre_slice] of THAT group
        NOTE no signed because a group of neurons always have ONE-sign (assuming Dale/ no 0-crossing still enforcced)
    """
    try:
        post_slc, pre_slc = ctx.net.synapseslices[syn_group]
    except KeyError:
        raise KeyError(f"Unknown synapse group {syn_group!r}. "
                       f"Available synapse groups: {tuple(ctx.net.synapseslices.keys())}")
    if _sparse(ctx):  
        return float(np.sum(ctx.syn.W[_syn_block(ctx, post_slc, pre_slc)]))
    group_weights = ctx.syn.W[post_slc, pre_slc]
    return float(np.sum(group_weights))

def sum_timeinphase(ctx: RunContext, phase: str | None = None) -> int: 
    """Cumulative timesteps spent in 'phase' (default: the CURRENT phase)."""
    name = ctx.sim.current_phase if phase is None else phase
    return int(ctx.sim.time_in_phase.get(name, 0))


def time_in_currentphase(ctx: RunContext) -> int:
    """Timesteps in the CURRENT phase since the last switch
        * (sim.time_in_current_phase: ticked by Run._tick, reset by PhaseScheduler.switch_phase)."""
    return int(ctx.sim.time_in_current_phase)


   #########################################################################
##########################   Degree   ###################################
   #########################################################################

def _degree(ctx: RunContext, *, axis: int, post_group: str | None, pre_group: str | None,
            eps: float, weighted: bool, signed: bool, summary: SUMMARY | None):
    """
    Active synapse rule:  
        abs(W[post, pre]) >  eps -> active
        abs(W[post, pre]) <= eps -> dead
    Args:
        axis = in-degree or out-degree evaluated.
            axis=1 reduces across PRE  -> one value per selected POST neuron  (in-)
            axis=0 reduces across POST -> one value per selected PRE  neuron  (out-)
        weighted = calculate node-degree or node-strength
            weighted=False --> count synapses. 
            weighted=True  --> sums their weights (strength).
    """
    post_slc, pre_slc = post_rows(ctx, post_group), pre_cols(ctx, pre_group)
    if _sparse(ctx):     ##sparse: edges binned per post row (axis=1) / per pre column (axis=0); same values and shapes as dense ##TODO if edge_block gets modified this must change also
        syn = ctx.syn
        sel = _syn_block(ctx, post_slc, pre_slc)
        w = syn.W[sel]
        active = np.abs(w) > eps
        lo, hi, _ = (post_slc.indices(ctx.net.n_post) if axis == 1 else pre_slc.indices(ctx.net.n_total))
        ids = ((syn.post if axis == 1 else syn.pre)[sel] - lo)[active]
        if not weighted:
            values = np.bincount(ids, minlength=hi - lo).astype(np.int32, copy=False)
        else:
            values = np.bincount(ids, weights=(w if signed else np.abs(w))[active], minlength=hi - lo).astype(np.float32, copy=False)
        return summarize(values, summary)

    block = ctx.syn.W[post_slc, pre_slc]
    active = np.abs(block) > eps

    if not weighted:
        values = np.count_nonzero(active, axis=axis).astype(np.int32, copy=False)
    else:
        active_weights = np.where(active, block if signed else np.abs(block), 0.0)
        values = np.sum(active_weights, axis=axis).astype(np.float32, copy=False)

    return summarize(values, summary)


def in_degree(ctx: RunContext, *, post_group: str | None = None, pre_group: str | None = None,
              eps: float = 0.0, summary: SUMMARY | None = None) -> np.ndarray | float:
    """
        Find directed in-degree.
          - For each postsynaptic neuron, count the number of active, incoming synapses it receives.

        Active synapse rule:
            abs(W[post, pre]) > eps  -> active
            abs(W[post, pre]) <= eps -> dead

        Args:
            post_group:   The neuron group whose in-degree is being described.
               If None: consider all postsynaptic neurons.

            pre_group:  Optional source restriction.
               If None: consider all presynaptic neurons.

            eps: Activity threshold around zero.
                eps=0.0: only exactly zero weights are dead.
                  eps=anything else: weights with abs(weight) <= eps are treated as dead.

            summary:
                None: return one in-degree value per selected post neuron.
                "mean", "sum", "max", "min":  return one scalar summary.

        Returns:
            np.ndarray:  active in-degree per selected post neuron

            or float:  scalar summary
    """
    return _degree(ctx, axis=1, post_group=post_group, pre_group=pre_group,
                   eps=eps, weighted=False, signed=True, summary=summary)


def out_degree(ctx: RunContext, *, pre_group: str | None = None, post_group: str | None = None,
               eps: float = 0.0, summary: SUMMARY | None = None) -> np.ndarray | float:
    """
        Find directed out-degree.
         - For each selected presynaptic neuron, count the number of outgoing synapses it sends.

        Active synapse rule:
            abs(W[post, pre]) > eps  -> active
            abs(W[post, pre]) <= eps -> dead

        Args:
            pre_group:  The neuron group whose out-degree is being described.
                If None: describe all presynaptic neurons.

            post_group:  Optional target restriction.
                If None: count outgoing synapses to all postsynaptic neurons.

            eps: Activity threshold around zero.
                eps=0.0: only exactly zero weights are dead.
                eps = anything else: weights with abs(weight) <= eps are treated as dead.

            summary:
                None: return one out-degree value per selected pre neuron.

                "mean", "sum", "max", "min":  return one scalar summary.

        Returns:
            np.ndarray: active out-degree per selected pre neuron
            
            or float: scalar summary
    """
    return _degree(ctx, axis=0, post_group=post_group, pre_group=pre_group,
                   eps=eps, weighted=False, signed=True, summary=summary)


def degree_strength(ctx: RunContext, *, direction: Literal["in", "out"] = "in",
                    post_group: str | None = None, pre_group: str | None = None,
                    signed: bool = True, eps: float = 0.0,
                    summary: SUMMARY | None = None) -> np.ndarray | float:
    """
        Find weighted directed degree (strength). 
         - Strength = sum of active synaptic weights 

        Active synapse rule:
            abs(W[post, pre]) > eps  -> active
            abs(W[post, pre]) <= eps -> dead

        Args:
            direction:
                "in": In-degree: For each selected post neuron, sum incoming active weights.

                "out": Our-degree: For each selected pre neuron, sum outgoing active weights.

            post_group: Target neuron group restriction.
            pre_group:  Source neuron group restriction.

            signed:
                True: preserve signs. Inhibitory/excitatory values may cancel.

                False: use absolute weight magnitude.

            eps:
                Activity threshold around zero.
                eps=0.0:  only exactly zero weights are dead.
                eps>0.0: weights with abs(weight) <= eps are treated as dead

            summary:
                None: return one strength value per selected described neuron.
                "mean", "sum", "max", "min":  return one scalar summary.

        Returns:
            np.ndarray:  degree strength per selected neuron
            
            or float: scalar summary
            """
    if direction == "in":
        axis = 1
    elif direction == "out":
        axis = 0
    else:
        raise ValueError(f"Unknown direction {direction!r}. Expected 'in' or 'out'.")
    return _degree(ctx, axis=axis, post_group=post_group, pre_group=pre_group,
                   eps=eps, weighted=True, signed=signed, summary=summary)


def deadened_synapses(ctx: RunContext, *, syn_group: str | None = None,
                      eps: float = 0.0) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Deadened synapses as COO-style coordinates.

    A synapse is deadened when   syn_mask[post, pre] is True   AND   abs(W[post, pre]) <= eps. 
    eps = Death-threshold

    Returns:
        (dead_post_idx, dead_pre_idx) -- post is a W ROW index, pre is a true neuron id.
        The pair indexes W directly:  ctx.syn.W[dead_post_idx, dead_pre_idx]
        which is what SleepPhase.revive_dormant_synapses needs. true post neuron id is
        dead_post_idx + ctx.net.row_offset.
        
        IF sparse: Return ONE int64 array of syn indices instead 
    """
    if syn_group is None:
        post_slc, pre_slc = slice(None), slice(None)
    else:
        try:
            post_slc, pre_slc = ctx.net.synapseslices[syn_group]
        except KeyError:
            raise KeyError(f"Unknown synapse group {syn_group!r}. "
                           f"Available synapse groups: {tuple(ctx.net.synapseslices.keys())}")

    if _sparse(ctx):   
        return np.flatnonzero(_syn_block(ctx, post_slc, pre_slc) & (np.abs(ctx.syn.W) <= eps)).astype(np.int64, copy=False)

    W_block = ctx.syn.W[post_slc, pre_slc]
    mask_block = ctx.syn.syn_mask[post_slc, pre_slc]
    dead_block = mask_block & (np.abs(W_block) <= eps)

    local_post_idx, local_pre_idx = np.nonzero(dead_block)
    if local_post_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    post_start = 0 if post_slc.start is None else post_slc.start
    pre_start = 0 if pre_slc.start is None else pre_slc.start
    return (local_post_idx.astype(np.int64, copy=False) + post_start,
            local_pre_idx.astype(np.int64, copy=False) + pre_start)


   ######################################################################################
##########################   Raster metadata   #######################################
   ######################################################################################

##Reads of the network that ANALYSIS needs to interpret a stored raster. ¨spike_indices=transform-data. analysis=reconstruction. here: neither

def capture_span(ctx: RunContext, *, include_input: bool = False) -> tuple[int, int]:
    """
    [start, stop) of the TRUE neuron ids a recording with this policy stores
    """
    return (0 if include_input else ctx.net.n_input), ctx.net.n_total


def neuron_bands(ctx: RunContext) -> dict[str, dict]:
    """
    Per-group true neuron band, in declaration order, with role flags.
        {group: {"start", "stop", "n", "input", "inh"}}
    spike_i holds true neuron ids
    """
    bands = {}
    for name, slc in ctx.net.neuronslices.items():
        spec = ctx.net.neuronspecs[name]
        bands[name] = {"start": int(slc.start), "stop": int(slc.stop),
                       "n": int(slc.stop - slc.start),
                       "input": bool(spec.input), "inh": bool(spec.inh)}
    return bands


def raster_meta(ctx: RunContext, history=None) -> dict:
    """
    Derive info for representing meta. JSON-safe, RunHistory.metadata() - paired if a saved history should be fully self-describing.
    NOTE REVIEW: -> behaviour here hardcoded not ideal.  -> helpers + meta.py ?
    """
    include_input = True if history is None else bool(history.cfg.include_input)
    start, stop = capture_span(ctx, include_input=include_input)
    meta = {"n_total": int(ctx.net.n_total), "n_input": int(ctx.net.n_input),
            "capture_start": start, "capture_stop": stop,
            "include_input": include_input,
            "bands": neuron_bands(ctx)}
    if history is not None:
        meta["dt"] = float(history.dt)
        meta["t_start"] = history.t_start
        meta["n_steps"] = int(history.n_steps)
    return meta


   #####################################################################
##########################   Spike counts (responses)   ##############
   ####################################################################

def spike_counts(ctx: RunContext, history, *, since_event: int | None = None, since_mark: np.ndarray | None = None,
                 group: str | None = None) -> np.ndarray:
    """
    Spike counts per neuron (true_neuronID order, float64) over the events History recorded since EITHER:
         1. an event index 'since_event'.
         2. a bookmark (history.mark -> bookmark)
    Args:
        group: str = neuron-group (only record one group)
        group: str = None -> record everything
        """
    n_total = ctx.net.n_total
    if history is None:
        counts = np.zeros(n_total, dtype=np.float64)
    elif since_mark is not None:
        counts = history.since(since_mark).astype(np.float64)
    elif since_event is not None:
        spike_i = history.spike_i()[since_event:]
        counts = np.bincount(spike_i.astype(np.int64), minlength=n_total).astype(np.float64)
    else:
        raise ValueError("spike_counts needs since_event= <event-index> or since_mark= <history.mark()-bookmark>.")
    return counts if group is None else counts[ctx.net.neuronslices[group]]


def spike_count_responses(per_sample_counts: list[np.ndarray]) -> np.ndarray:
    """Stack per-sample spike-count vectors into (n_samples, n_neurons) for scoring."""
    return np.vstack([np.asarray(c, dtype=np.float64) for c in per_sample_counts])


   ########################################################################
##########################   Health readings   ###########################
   ########################################################################

def mean_rate_hz(ctx: RunContext, *, spike_count: int, steps: int, dt: float) -> float:
    """Mean firing rate per recurrent neuron (Hz) over 'steps' timesteps of 'dt' ms.
    
    NB! ms timesteps hardcoded here.
    TODO LATER: this SHOULD receive simstate to ENSURE ms not used if actual timesteps are NOT 1 ms """
    n_rec = max(ctx.net.n_rec, 1)
    seconds = max(steps * dt / 1000.0, 1e-12)
    return float(spike_count) / seconds / n_rec


def rate_hz(counts, n_neurons: int, steps: int, dt: float) -> float:
    """Population rate in Hz per neuron, across _ms_ dt steps.  
    NOTE: NO CTX, relates to protocol work. 
        *  "counts" spikes of "n_neurons" neurons over "steps" timesteps of "dt" ms.
    
    NB! ms hardcoded here. 
    TODO LATER: this SHOULD receive simstate to ENSURE ms not used if actual timesteps are NOT 1 ms """
    return float(np.sum(counts)) / (max(int(n_neurons), 1) * max(steps * dt / 1000.0, 1e-12))


def pinned_weights(ctx: RunContext, *, w_min: float | None, w_max: float | None, eps: float | None = None) -> dict: 
    """
    fraction of realized synapses pinned at bound, zt 0, or declared pinned (saturated) at |w| > eps

    Args:
        eps = epsilon from w_min or w_max 
        * eps = abs(W) > eps  => saturated
        NOTE no lower-bound eps (around 0) to call a near-dead synapse
    """
    syn, net = ctx.syn, ctx.net
    n_syn = max(int(net.n_synapses), 1)
    W = syn.W

    at_bound = np.zeros_like(W, dtype=bool)
    if w_max is not None and np.isfinite(w_max):
        np.logical_or(at_bound, W >= w_max, out=at_bound)
    if w_min is not None and np.isfinite(w_min):
        np.logical_or(at_bound, W <= w_min, out=at_bound)
    if eps is not None: 
        np.logical_or(at_bound, np.abs(W) > eps, out=at_bound)
    if hasattr(syn.weights, "syn_mask"):
        np.logical_and(at_bound, syn.weights.syn_mask, out=at_bound)

    saturated = float(np.count_nonzero(at_bound)) / n_syn
    zero = (W == 0.0)
    if hasattr(syn.weights, "syn_mask"):
        zero &= syn.weights.syn_mask       ##dense: only REALIZED edges
    dead = float(np.count_nonzero(zero)) / n_syn
    return {"saturated_frac": saturated, "dead_syn_frac": dead,
            "w_absmax": float(np.abs(W).max()) if W.size else 0.0}
