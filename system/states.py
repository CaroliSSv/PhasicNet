from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple
import logging

import numpy as np

from network import Network, WeightProjection
from rnghandler import RNGHandler
from synapticindex import SynapticIndex   

logger = logging.getLogger(__name__)


NEVER_SPIKED = 1 << 30    ## << bit-shift operator -> set with default sentinel value == "never spiked"

def _since_update(since: np.ndarray, last_spikes_rec: np.ndarray, rec: slice) -> None:
    """ Steps since each ROW's last spike, advanced once per step from LAST step's spikes (neur.Spikes[n_input:], a read-only view).
        * since == 1 on the step right after a spike. Brian keeps the same information as LS (spike history) + next_allowed_spiketime.
    
    NeuronalState.Since is written ONCE per step by the orchestra (right after the Spikes commit) and only READ by modules."""
    np.add(since, 1, out=since)
    np.minimum(since, NEVER_SPIKED, out=since)
    since[rec][last_spikes_rec] = 1
# ---------------- STATES ----------------

@dataclass
class SimulationState:
    """
    Args:
        current_phase: str = "DEFAULT"
        step_in_sample: int = 0  ###dataloader cursor WITHIN the bound sample. Runner's loop condition.
        sample_T: int = 0        ###length of the bound sample, set by DataLoader.begin_sample
        current_epoch: int = 0
        current_datarun: int = 0
        current_runstep: int = 0 #Absolute simulation time. NO wrap, ####datacapture: spike_t = current_runstep
        train_step: int = 0      ##train only clock to avoid aging eventmode traces in eval (where plasticity is frozen)

        time_in_current_phase: int = 0
        time_in_phase: dict[str, int] 
        mode: str = "train"     #NOTE  "train" | "eval". Set by Run.set_mode
    """
    current_phase: str = "DEFAULT"
    step_in_sample: int = 0
    sample_T: int = 0
    current_epoch: int = 0
    current_datarun: int = 0
    current_runstep: int = 0 
    train_step: int = 0     ## the train-step clock

    time_in_current_phase: int = 0
    time_in_phase: dict[str, int] = field(default_factory=dict)
    mode: str = "train"    



@dataclass
class NeuronalState:
    """Neuronal dynamical state arrays.
    NB: RELIES on Typing to collect OPTIONAL_STATE_ARRAYS, used during assembly. ALWAYS Type"""
    V: np.ndarray                               #float, (n_post):           (N,) or (n_rec,),   Membrane state (voltage)
    Spikes: np.ndarray                          #bool,  (N,),               Spike state

    V_th: Optional[np.ndarray] = None           #float, (n_post):           (N,) or (n_rec,),   Optional adaptive threshold state 
    pre_spike: Optional[np.ndarray] = None      ##float, (N,),              Optional spike-trace state
    post_spike: Optional[np.ndarray] = None     ##float, (n_post):          (N,) or (n_rec,),  Optional spike-trace state
    Since: Optional[np.ndarray] = None          ##int64, (n_post):          steps since each ROW's last 




@dataclass
class DenseWeights:
    W: np.ndarray
    syn_mask: np.ndarray
    plastic_mask: np.ndarray | None = None

@dataclass
class SparseWeights:
    W: np.ndarray                           #float32, (n_synapses,)   one weight per REALIZED edge
    pre: np.ndarray                         #int32,   (n_synapses,)   presynaptic true_neuronID (W column)  
    post: np.ndarray                        #int32,   (n_synapses,)   postsynaptic W ROW index
    n_post: int                             #row count. (n_total or n_rec)
    index: SynapticIndex                    #CSR/CSC pointers over (post, pre); see synapticindex.py
    plastic_mask: np.ndarray | None = None  #bool, (n_synapses,)  realized AND non-frozen.


@dataclass
class SynapticState:
    weights: DenseWeights | SparseWeights             
    frozen: Optional[dict[str, WeightProjection]] = None  

    @property
    def W(self) -> np.ndarray:
        """Active synaptic weights (values). Either dense  or sparse formatted."""
        return self.weights.W

    @property
    def syn_mask(self):
        """Mask of realized synapses in (dense) W. """
        if not isinstance(self.weights, DenseWeights):
            raise RuntimeError( "syn_mask is only defined for dense weights (realized connectivity inbuilt in sparse)")
        return self.weights.syn_mask

    @property
    def pre(self):
        """Presynaptic true_neuronID per realized edge (sparse only)."""
        if not isinstance(self.weights, SparseWeights):
            raise RuntimeError("pre is only defined for sparse weights (dense has no edge list)")
        return self.weights.pre
    @property
    def post(self):
        """Postsynaptic W ROW index per realized edge (sparse only)."""
        if not isinstance(self.weights, SparseWeights):
            raise RuntimeError("post is only defined for sparse weights (dense has no edge list)")
        return self.weights.post

    @property
    def freeze_slices(self) -> Tuple[WeightProjection, ...]:
        """Projection rectangles/ syn slices that are to be frozen during runtime that must NOT learn, as (post_rows, pre_cols) basic slices."""
        if self.frozen is None:
            raise RuntimeError("freeze_slices unresolved. PhasicNet._implement_freeze() must run AFTER _instantiate_states().")
        return tuple(self.frozen.values())
  
    @property
    def plastic_mask(self) -> np.ndarray:
        """Realized AND non-frozen synapses, either in dense or sparse format. (in their native representation)"""
        return self.weights.plastic_mask


@dataclass
class UpdateState:
    """Dataclass with buffer arrays for inplace runtime computation. (ONLY keeps arrays read by more than one module or by the orchestra itself).
    
        NB: RELIES on Typing to collect OPTIONAL_STATE_ARRAYS, used during assembly. ALWAYS Type"""

    #dataloader
    step_input: np.ndarray      #Bool, (n_in,): Input spikes 

    #integrator
    I2V: np.ndarray                     #float, (n_post,):         (N,) or (n_rec,),   Sum of currents, then scratch buffer for I to V calculus
    dV: np.ndarray                      #float, (n_post,):         (N,) or (n_rec,),   voltageUpdate
    rec_spikes: np.ndarray              #bool, (n_post,):          (N,) or (n_rec,),    spike-state update

    #plasticity
    dW: Optional[np.ndarray]             ##float, (n_post, N) DENSE only; None under sparse, where plasticity RETURNS its per-edge dw instead

    #For debug; Check noise directly
    neur_noise: Optional[np.ndarray ] = None    ##float, (n_post,):           (N,) or (n_rec,)
    syn_noise: Optional[np.ndarray ]  = None    ##float, (n_post, N):         (N,N) or (n_rec, N)
    
    dV_th: Optional[np.ndarray] = None       #float, (n_post,):           (N,) or (n_rec,),   Optional adaptive_threshold Update



@dataclass(slots=True)
class RunContext: 
    """All shared, mutable states, network, and rng used in runtime (for easy reference)
    
    Args:
        sim: SimulationState
        upd: UpdateState
        neur: NeuronalState
        syn: SynapticState
        net: Network
        """
    sim: SimulationState
    upd: UpdateState
    neur: NeuronalState
    syn: SynapticState
    net: Network
    rng: RNGHandler


def optional_state_arrays() -> tuple[str, ...]:
    out = []
    for cls in (NeuronalState, UpdateState):
        for f in fields(cls):
            if "Optional" in str(f.type) or "None" in str(f.type):
                out.append(f.name)
    return tuple(out)
OPTIONAL_STATE_ARRAYS = optional_state_arrays()
