#dynamics - abstract roles for the level-5 (lowest lvl) runtime modules:   Synapse, Integrator, Threshold, Plasticity, Noise, Phase 
from __future__ import annotations
 
from abc import ABC, abstractmethod 
from typing import Tuple
import logging
from states import RunContext, OPTIONAL_STATE_ARRAYS
from utils import _capitalize_strip_name
from dataclasses import is_dataclass
import numpy as np

logger = logging.getLogger(__name__)

###Dynamics- specific helpers. MOVE these to utils?
   ######################################################################################
##########################   HELPERS   #####################################################
   ######################################################################################


def collect_requirements(*modules) -> set[str]:
    """Union of every installed module's 'requires'"""
    required: set[str] = set()

    for module in modules:
        if module is None:
            continue
        for name in getattr(module, "requires", ()):
            if name not in OPTIONAL_STATE_ARRAYS:
                raise ValueError(
                    f"{type(module).__name__} requires unknown state array {name!r}. "
                    f"Known optional arrays: {tuple(OPTIONAL_STATE_ARRAYS)}")
            required.add(name)

    logger.debug("Collected required optional state arrays: %s", sorted(required))
    return required


def check_role(cls, role: type, label: str) -> None: 
    """Reject a module installed into the wrong slot, before run. Warns.
    """
    if cls is None:
        return
    if not (isinstance(cls, type) and issubclass(cls, role)):
        logger.warning("%s=%s does not subclass %s. Role and write-set cannot be checked; "
                       "declare it as a %s subclass to enable validation.",
                       label, getattr(cls, "__name__", cls), role.__name__, role.__name__)


   ######################################################################################
##########################   Dynamics abstract class   ###############################
   ######################################################################################
##NOTE split to make each class header of their own file?

class Dynamics(ABC):
    """
    Runtime module: does its own maths, writes only its declared arrays. IF inplace updates: writes to/mutates ONLY UpdateState arrays specified in "writes"
    
        
    Args:

        output: Semantic output used for module ROLE DEFINITION. For debug, NOT edited in subclasses

        writes: ALL UpdateState arrays modified by module, including scratch buffers. ANY not mentioned raises an error during debug 

        requires: Optional state arrays neccecary for this module (allocated during instantiation of state arrays).

        persistent: names of state_dict arrays that PERSIST through a transient reset, guarded by "eval" (unchanging during eval)
        
        Default (): every array is transient (traces, conductances, refractory counts...).

    """

    output: str = ""
    writes: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    persistent: Tuple[str, ...] = ()   
    weights_format: str | None = None   ##"dense" | "sparse" | None (= does not touch W). (Checked against SimulationCfg 

    @property
    def params(self) -> dict:
        """STATIC values (resolved)
            - Consumed by prep_run -> the derived-cfg source.
            - Excludes arrays and private (_) attributes."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and not isinstance(v, np.ndarray)}

    @property
    def state_dict(self) -> dict:
        """DYNAMICAL arrays (that don't cross module-boundaries). 
            - Consumed by Checkpoint -> save and restore for fit-reset
            - Excludes non-array formats (scalars)

            NOTE: accidentally catches _static_ arrays ex. eta_pre and eta_post in checkpoints. Wasteful, but minor issue"""
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and isinstance(v, np.ndarray)}

    @property
    def transient_state(self) -> dict:
        """state_dict minus 'persistent'
        RESET on fresh network (Run.reset_transient)."""
        persistent = self.persistent
        return {k: v for k, v in self.state_dict.items() if k not in persistent}


    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            np.copyto(getattr(self, k), v)

    def __repr__(self):
        #EXample output: 'BaseSynapse[Synapse, sparse](b=None)'.
        role = next((b.__name__ for b in type(self).__mro__[1:] if b.__name__ in _ROLES), "")
        w_fmt = f", {self.weights_format}" if self.weights_format else ""
        kv = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}[{role}{w_fmt}]({kv})"


    def configure_for_network(self, net) -> None:
        """Shared method for any parameter that is network-specific (ex. presynaptic-neuron specific eta; eta_inh and eta_exc). 
        During instantiation of runtime, configure_for_network gets called AFTER instantiation of both network in use and runtime modules.
        """
        return None


def _configured_for_network(module) -> bool: 
    """Check if subclass defined configure_for_network (used for debug)
    
    Returns True if base is overwritten."""
    base = Dynamics.configure_for_network
    return getattr(type(module), "configure_for_network", base) is not base


######################################################################################
##########################   Dynamicss   ###############################



_ROLES = {"Synapse", "Integrator", "Threshold", "Plasticity", "Noise", "Phase"}


class Synapse(Dynamics):
    """
    Calculate inputs received across all synaptic junctures of a neuron, 
            output = "I" - Synapse-role definition, differentiates Synapse from other Dynamicss (here in shared "I2V" scratch array)
            writes = ("I2V", ) - allowed updates/returns. must ALWAYS be included, 
    
            weights_format: "Dense" | "Sparse"  -> Optional declaration of format used for computation (checked against simulationcfg.weights_format)
    Converts presynaptic spike events into postsynaptic current via the weight matrix.
        NOTE: (I is summed from all post-synaptic-endplate of a neuron, BUT NO spatial degradation/neuronal dynamics applied).

    """
    output = "I2V"      #NOTE despite syn taking sparse or dense, neuronal states like I2V are always dense so this holds regardless
    writes = ("I2V",)
    
    @abstractmethod
    def step(self, ctx: RunContext):
        """
        Advance one timestep, calculate internally and return output defined in role-definition. 

            IF inplace-updates: VIEWS NeuronalState and SynapticState. Writes ONLY to UpdateState arrayys named in self.writes.
            IF return-updates (sparse / event orchestra): RETURNS the output array named in 'output'."""
        raise NotImplementedError


class Integrator(Dynamics):
    """
    Integrate membrane: current -> discrete voltage update.
        output = "dV" - integrator-role definition, differentiates from other modules
        writes = ("dV", ) - allowed updates/returns. dV must ALWAYS be included. Used to instantiate neccecary optional arrays in shared state-classes
    
    Calculates membrane dynamics, apply discrete voltage update

    """    
    output = "dV"
    writes = ("dV",)

    @abstractmethod
    def step(self, ctx: RunContext):
        """
        Advance one timestep, calculate internally and return output defined in role-definition. 

            IF inplace-updates: VIEWS NeuronalState and SynapticState. Writes ONLY to UpdateState arrayys named in self.writes.
            IF return-updates (sparse / event orchestra): RETURNS the output array named in 'output'."""
        raise NotImplementedError

class Threshold(Dynamics):
    """
    Emit (recurrent-network) spikes (calculations involved in thresholding) 
        output = "rec_spikes" - threshold-role definition 
        writes = ("rec_spikes", ) - allowed updates/returns. rec_spikes must ALWAYS be included. Used to instantiate neccecary optional arrays in shared state-classes
    """
    output = "rec_spikes"
    writes = ("rec_spikes",)

    @abstractmethod
    def step(self, ctx: RunContext):
        """
        Advance one timestep, calculate internally and return output defined in role-definition. 

            IF inplace-updates: VIEWS NeuronalState and SynapticState. Writes ONLY to UpdateState arrayys named in self.writes.
            IF return-updates (sparse / event orchestra): RETURNS the output array named in 'output'."""
        raise NotImplementedError

    

class Plasticity(Dynamics):
    """Synaptic learning: spike history -> discrete weight update.
   
    Apply update to weights (calculations for dW)
        output: "dW"  -  role definition used in debug
        writes: ("dW", ) -  dW must *ALWAYS* be included.  Used to instantiate neccecary optional arrays in shared state-classes

    
        weights_format: "Dense" | "Sparse"  -> Optional declaration of format used for computation (checked against simulationcfg.weights_format)
    """
    output = "dW"                   ##NOTE the shape of this array will depend on weights_format (dw=N*N OR dw=)
    writes = ("dW",)
    w_min = None
    w_max = None


    @abstractmethod
    def step(self, ctx: RunContext):
        """
        Advance one timestep, calculate internally and return output defined in role-definition. 

            IF inplace-updates: VIEWS NeuronalState and SynapticState. Writes ONLY to UpdateState arrayys named in self.writes.
            IF return-updates (sparse / event orchestra): RETURNS the output array named in 'output'.
            
            IF update_on event / hybrid_event a sparse plasticity module returns the PAIR (k, dw_k) -- the touched plastic edges, (unique, storage order) and their dw 
                -> Run.step_sparse commits W[k] += dw_k."""
        raise NotImplementedError

    def _safety_clamp(self) -> None:
        """
        If no w_min or w_max set, default to -inf or +inf respectively
        """
        if self.w_min is None:
            object.__setattr__(self, "w_min", -np.inf)   ##object.__setattr__: runtime modules hold STATIC params -> MAY freeze classes later
        if self.w_max is None:                          
            object.__setattr__(self, "w_max", np.inf)


###Phase

class Phase(Dynamics):  
    phase_type: str #   ## the transition contract. transition_writes = the learned arrays a phase writes AT ITS ENTRY (superseeds the "no mods on true neuronal/synaptic states by modules") rule
    transition_writes: tuple[str, ...] = ()
    weights_format: str | None = None
    
    def __init__(self, cfg: PhaseCfg, sim_cfg: SimulationCfg=None, rng: np.random.Generator = None):
        if _capitalize_strip_name(cfg.phase_type) != _capitalize_strip_name(self.phase_type): 
            raise ValueError(f"{type(self).__name__} expected {self.phase_type}, got {cfg.phase_type}")

        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.rng = rng

        self.triggers = cfg.triggers
        self.trigger_mode = cfg.trigger_mode
        self.exit_triggers = cfg.exit_triggers

        self.dataloader_active = cfg.dataloader_active

    def on_enter(self, ctx, capture=None) -> None:
        pass

    def on_exit(self, ctx, capture = None) -> None:
        pass


    def alternate_inputs(self, ctx) -> None:
        """Default alternate inputs: False
        step_input = False """

        ctx.upd.step_input.fill(False)

    def post_dataloader(self, ctx) -> None:
        pass
    
    def post_synapse(self, ctx) -> None:
        pass

    def post_integrator(self,ctx) ->None:
        pass

    def post_threshold(self,ctx) ->None:
        pass

    def post_plasticity(self,ctx) ->None: ##NOTE remove? N*N change is HEAVY. REGULARIZATION = 
        pass

    def on_noise(self, ctx, current_noise) -> None:
        """MODIFY noise directly INSIDE noise module"""
        pass


###Noise

class Noise(Dynamics):   ##Dynamics base (no step requirement): gets params()/state_dict() like every other module
    def __init__(self, cfg: NoiseCfg, sim_cfg: SimulationCfg, rng: RNGHandler):
        pass

    def on_post_syn(self, ctx: RunContext, phase: Phase):
        pass
    def on_membrane(self, ctx: RunContext, phase: Phase):
        pass

    def on_threshold(self, ctx: RunContext, phase: Phase):
        pass

    def on_weights(self, ctx: RunContext, phase: Phase): 
        pass
