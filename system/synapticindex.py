from __future__ import annotations
import logging
##no "states" import: states imports THIS module for the SparseWeights.index type, so importing back is a cycle.
## The constructor takes the raw edge arrays instead.

import numpy as np


logger = logging.getLogger(__name__)


def _syn_ids(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Vectorized lookup/collection of synapse-ids from ptr arrays [starts[neur_id], stops[neur_id])."""
    lens = stops - starts
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    ##offset each range so that arange(total) lands on the right absolute indices
    offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return offsets + np.arange(total, dtype=np.int64)

class SynapticIndex:
    """
    For finding synapses with spiking pre or post-neurons in sparse runtime. (COO storage)

    CONVENTIONS followed: "pointer array" => _ptr and "permutation" => _perm,
    
        * post_ptr/pre_ptr = arrays holding cumulative sum of the number of synapses counted at synapse-idx i.
        Useful:
            * post_ptr[i]      -> synapse-idx of *first* synapse touched by post neuron I.
            * post_ptr[i+1]    -> synapse-idx of *last* synapse touched by post neuron I.
            * post_ptr[i+1] - post_ptr[i]  -> number of synapses with neuron- post_idx i
            * post_ptr[i] : post_ptr[i+1]  -> synapse-indices with neuron- post_idx == i
        NOTE:
            * Because of row-major ordering, post_ptr corresponds directly with post_idx. pre_ptr has a different permutation of the same synapses, 
                so pre_perm reorders/maps pre_ptr[i] to synapse-idx i  synapse index- order
            * pre_perm[ pre_ptr[j] : pre_ptr[j+1] ]  -> edge indices with pre_idx == j

    NOTE: per-neuron traces decay EVERY STEP even in eventmode: O(N) per step due to being applied on NEURONS. Does NOT to apply lazy decay. 
            Accumulated decay+round-off to float32 causes drift/divergence between the two modes.
            - (Brian2 applies "lazy analytic decay" on EDGE-wise updates. (Lazy-decay = Decay applied ON spike right before spike maths is applied w/ formula) -> 
                trace * exp(-k*dt/tau))  )
    ##TODO read below, filter, write
    WHAT "event" MEANS. Plasticity traces live PER SYNAPSE (Brian2's Apre / Apost) and are touched ONLY when a spike arrives at one of the
    synapse's two neurons: the decay owed since the synapse's last event is applied analytically in one go (trace * decay ** n_steps, what Brian2's
    "(event-driven)" keyword does), then the rule runs at those synapses. Nothing on the plasticity side costs work per step in proportion to the number
    of synapses or neurons; the cost is per spike x fan-out. The neurons stay clock-driven (currents, membrane, threshold every step), as in Brian2.

    "hybrid_event": per-NEURON traces decayed EVERY step (O(N) per step), the rule evaluated only at edges with a spiking endpoint.
    Bit-exact against "timestep".

    EXACTNESS. "event" agrees with "timestep" to float tolerance (~1e-6 relative on a trace), not to the bit (system/tests/test_sparse.py).
    decayed() also takes a per-edge decay array (heterogeneous tau); no module passes one yet.

    THE SCHEME every event-mode plasticity module follows (SparseTraceSTDP, HLSPPlasticitySparse, NALSMAstroSTDPSparse):
        per-edge arrays   X_e (one per trace) and T_e (int64: the step at which the edge was last brought up to date); Capitalised -> state_dict.
                          The per-neuron trace arrays are None in this mode.
        at step t         k = touched(syn, post_spike, pre_spike)        the PLASTIC edges with a spiking endpoint
                          n = elapsed(T_e, k, t)                         decay steps owed: steps T_e[k] + 1 .. t - 1
                          v = decayed(X_e, k, decay, n)                  the trace as the clocked rule reads it at the START of step t
                          ... the module's own order: rule then trace update (TraceSTDP, NALSM), or trace update then rule (HLSP, Algorithm 1) ...
                          X_e[k] = the value after step t's update;  T_e[k] = t
        t                 = ctx.sim.train_step (counts "train" steps only, so eval passes do not age the traces). After reset_transient X_e and T_e are 0.
        the module        RETURNS (k, dw_k) in both event modes; Run.step_sparse commits W[k] += dw_k and clips W[k] only.

    """
    __slots__ = ("post_ptr", "pre_ptr", "pre_perm", "n_post", "n_total")

    def __init__(self, pre: np.ndarray, post: np.ndarray, n_post: int, n_total: int):
        self.n_post, self.n_total = int(n_post), int(n_total)
        ##CSR over post - corresponds with storage
        counts = np.bincount(post, minlength=n_post)
        self.post_ptr = np.zeros(n_post + 1, dtype=np.int64)
        np.cumsum(counts, out=self.post_ptr[1:])
        ##CSC over pre - different pernutation, hence pre_perm argsort
        self.pre_perm = np.argsort(pre, stable=True).astype(np.int64)
        counts = np.bincount(pre, minlength=n_total)
        self.pre_ptr = np.zeros(n_total + 1, dtype=np.int64)
        np.cumsum(counts, out=self.pre_ptr[1:])

    def synIDs_by_post(self, post_ids: np.ndarray) -> np.ndarray:
        """Synapse-indices for every spiking POST neuron (used by ltp))."""
        return _syn_ids(self.post_ptr[post_ids], self.post_ptr[post_ids + 1])

    def synIDs_by_pre(self, pre_ids: np.ndarray) -> np.ndarray:
        """Synapse-indices for every spiking PRE neuron (used by synapse and ltd). 
         NOTE: NOT ascending by construction, so pre_perm reorders to the correct permutation 
         NOTE: use .sort() for storage order (ex. for bincount summation in DeltaCurrentSparse)"""
        return self.pre_perm[_syn_ids(self.pre_ptr[pre_ids], self.pre_ptr[pre_ids + 1])]

    
    def spiking_syns(self, post_spike_bool: np.ndarray, pre_spike_bool: np.ndarray) -> np.ndarray:
        """Every synapse spiking in pre or post during the current step. Returns IDs corresponding with storage syn-idx"""
        ltp = self.synIDs_by_post(np.flatnonzero(post_spike_bool))
        ltd = self.synIDs_by_pre(np.flatnonzero(pre_spike_bool))
        if ltp.size == 0:
            return np.sort(ltd)
        if ltd.size == 0:
            return ltp
        k = np.concatenate((ltp, ltd)); k.sort()                                    #NOTE TODO old (hybrid_event) used: return np.union1d(ltp, ltd)
        return k[np.concatenate(([True], k[1:] != k[:-1]))]                        ## sorted UNIQUE ids; faster than union1d

   ######################################################################################
##########################   Event-mode helpers (lazy decay)   ############################
   ######################################################################################
##"event" is explained in the SynapticIndex docstring above (that docstring
## already discusses Brian2's lazy decay). touched() takes "syn" as an argument, so no "states" import is needed here either.


def touched(syn, post_spike_bool: np.ndarray, pre_spike_bool: np.ndarray) -> np.ndarray:
    """The edges whose pre OR post neuron spiked this step, restricted to the plastic ones (syn.plastic_mask). Storage (row-major) order."""
    k = syn.weights.index.spiking_syns(post_spike_bool, pre_spike_bool)
    if k.size and syn.plastic_mask is not None:
        k = k[syn.plastic_mask[k]]
    return k


def elapsed(T_e: np.ndarray, k: np.ndarray, t: int) -> np.ndarray:
    """Decay steps owed to edges k at step t: the steps T_e[k] + 1 .. t - 1 (step t's own decay belongs to the module's update).
    t = sim.train_step."""
    return np.maximum(t - 1 - T_e[k], 0)


def decayed(X_e: np.ndarray, k: np.ndarray, decay, n: np.ndarray) -> np.ndarray:
    """X_e[k] after n steps of decay, float32 (a new array; X_e itself is untouched). decay: one scalar for the whole trace, or a per-edge array
    (indexed by k) when every synapse has its own time constant."""
    d = decay if np.isscalar(decay) else decay[k]
    return (X_e[k] * np.power(d, n)).astype(np.float32, copy=False)
