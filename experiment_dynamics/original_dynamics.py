#original_dynamics - MERGED 

##synapse

### Integrators, aka. classes that represent different neuron-integration (temporal summation) models. ex. Leaky-integrate and fire. 
### Integrated for the delta_time -> 0 case, with steady state assumption/simplification (delta_I = 0)

### Thresholding, aka classes defining static or adaptice thresholds   


##### Weight plasticity classes aka STDP rules; how synaptic weights are updated during training 

from __future__ import annotations
import logging

import numpy as np
from synapticindex import touched, elapsed, decayed   #(true event-based): the lazy-decay helpers of update_on "event"

from dynamicscfg import SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, SimulationCfg, NoiseCfg
from states import NeuronalState, SynapticState, UpdateState, RunContext
from dynamics import Synapse, Integrator, Threshold, Plasticity, Noise, Phase
from rnghandler import RNGHandler

logger = logging.getLogger(__name__)


   ###################################################################################### 
##########################   Synapses   ######################################################
   ######################################################################################

class DeltaCurrentDense(Synapse):
    """Dense, Delta-current synapse, dense W: I = W @ Spikes (+ b). One-step current pulse per spike."""
    weights_format = "dense"
    writes = ("I2V",) # writing I to shared scratch buffer
    requires = ()

    def __init__(self, cfg: SynapseCfg, sim_cfg: SimulationCfg=None):
        if cfg is None:
            raise ValueError("DeltaCurrentDense requires SynapseCfg.")
        self.b = None if cfg.b is None else np.float32(cfg.b)

    def step(self, ctx: RunContext) -> None:
        upd = ctx.upd
        neur = ctx.neur
        syn = ctx.syn
        # upd.I2V[...] = syn.W @ neur.Spikes if self.b is None else syn.W @ neur.Spikes + self.b
        np.dot(syn.W, neur.Spikes, out=upd.I2V)
        if self.b is not None:
            np.add(upd.I2V, self.b, out=upd.I2V)

class DeltaCurrentSparse(Synapse):
    """
    Sparse, Delta-current synapse, sparse W. RETURNS I (return-style orchestra) instead of writing upd.I2V.

    NOTE bit-identical to dense float64 accumulation (due to bincount->float64) but drifts from dense float32 (np.dot) by <=1 ULP on ~6% of patterns.
    
    update_on="event": only edges whose presynaptic neuron fired contribute; every other edge ==0 (dropped sparse)
    update_on: hybrid_event (the same gating for both event mds.)

    """
    weights_format = "sparse"
    writes = ("I2V",)
    requires = ()

    def __init__(self, cfg: SynapseCfg, sim_cfg: SimulationCfg=None):
        if cfg is None:
            raise ValueError("DeltaCurrentSparse requires SynapseCfg.")
        self.b = None if cfg.b is None else np.float32(cfg.b)
        self.update_on = "timestep" if sim_cfg is None else sim_cfg.update_on

    def step(self, ctx: RunContext) -> np.ndarray:
        neur, syn = ctx.neur, ctx.syn
        n_post = syn.weights.n_post
        if self.update_on in ("event", "hybrid_event"):
            active = syn.weights.index.synIDs_by_pre(np.flatnonzero(neur.Spikes))
            if active.size:
                active.sort()    ##back to storage (post, pre) order: keeps per-row summation order identical to timestep mode
                I = np.bincount(syn.post[active], weights=syn.W[active], minlength=n_post).astype(np.float32, copy=False)
            else:
                I = np.zeros(n_post, dtype=np.float32)
        else:
            per_synapse = syn.W * neur.Spikes[syn.pre] ##NOTE -> due to indexing this allocates new anyway. scratch buffer forces a second overwrite so actively harmful
            I = np.bincount(syn.post, weights=per_synapse, minlength=n_post).astype(np.float32, copy=False)
        if self.b is not None:
            np.add(I, self.b, out=I)
        return I


   ######################################################################################
##########################   Integrators   ###################################################
   ######################################################################################

class LIFexact(Integrator):
    """Integrate membrane."""
    ##I2V shared buffer. dV = output.
    writes = ("dV", "I2V")
    requires = ()

    def __init__(self, cfg: IntegratorCfg, sim_cfg: SimulationCfg):
        if cfg is None:
            raise ValueError("LIFexact requires IntegratorCfg.")
        if sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("LIFexact requires SimulationCfg with dt.")
        if cfg.tau_mem <= 0:
            raise ValueError("LIFexact requires tau_mem > 0.")
 
        self.v_rest = np.float32(cfg.v_rest)
        self.r = np.float32(cfg.r)                  #R for resistance
        self.tau_mem = np.float32(cfg.tau_mem)      ##stored under its cfg name for the derived cfg
        self.e_decay = np.float32(1-(np.exp(-(sim_cfg.dt/cfg.tau_mem))))  ## EXACT step update (from steady state solution of ODE)

    def step(self, ctx: RunContext) -> None: 
        """
        Integration step: 
            v_inf = V_rest + RI --> steady state solution PER NEURON PER STEP (assuming constant voltage during timestep dt)
        
        Exact decay: dv = (V_inf-V)*(1-e^-(dt/τ_membrane))
        
        CODE: upd.dV[...] = ((self.v_rest+self.r*upd.I2V) - neur.V) * self.e_decay
"""
        upd = ctx.upd
        neur = ctx.neur
        # EXACT LIF update
        np.multiply(upd.I2V, self.r, out=upd.I2V)                       # upd.I2V *= self.r                     ## HERE upd.I2V = r*I
        np.add(self.v_rest, upd.I2V, out=upd.I2V)                       # upd.I2V += self.v_rest                ## HERE upd.I2V = V_inf
        np.subtract(upd.I2V, neur.V, upd.dV)                           # upd.dV[...] = upd.I2V-neur.V              ## upd.dV = (V_inf-V)= -(V-V_inf))
        np.multiply(upd.dV, self.e_decay, out=upd.dV)                   # upd.dV *= self.e_decay

##      PREVIOUS non-simplified of exact decay version (just derived:)
#       Integration step: dV = (V_inf-V) + ( (V-V_inf)*e^-(dt/τ_membrane) )
#       CODE: upd.dV[...] = (self.v_rest+self.r*upd.I2V) - neur.V + (neur.V - (self.v_rest+self.r*upd.I2V)) * self.e_dttau

        # EXACT LIF update (alternative writing: simpler)
#        upd.I2V[...] = self.v_rest + self.r * upd.I2V      # upd.I2V = V_inf AFTER calculation, upd.I2V = I BEFORE calculation
#        upd.dV[...] = (upd.I2V - neur.V) * (self.e_dttau) # UPDATE overwriting previous dV using (V_inf-V) * (1 - (e^-(dt/τ_membrane)))


##ROugher euler decay based step
class LIFeuler(Integrator):
    """Integrate membrane."""
    writes = ("dV", "I2V")
    requires = ()
    def __init__(self, cfg: IntegratorCfg, sim_cfg: SimulationCfg): 
        if cfg is None:
            raise ValueError("LIFeuler requires IntegratorCfg.")
        if sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("LIFeuler requires SimulationCfg with dt.")
        if cfg.tau_mem <= 0:
            raise ValueError("LIFeuler requires tau_mem > 0.")

        self.v_rest = np.float32(cfg.v_rest)
        self.r = np.float32(cfg.r)                  #R for resistance
        self.tau_mem = np.float32(cfg.tau_mem)      ##keep copy for resolved-cfg
        self.dt_tau = np.float32(sim_cfg.dt/cfg.tau_mem)  # Euler decay update


    def step(self, ctx: RunContext) -> None: 
        """
        Integration step: 
            Euler decay: dv = (v_rest - V  +  R*I) * dt/τ_membrane
            
            CODE: upd.dV[...] = (self.v_rest - neur.V  +  self.r*upd.I2V) * self.dt_tau
        """
        upd = ctx.upd
        neur = ctx.neur
        # SIMPLE LIF update (euler decay) 
        np.multiply(upd.I2V, self.r, out=upd.I2V)       # upd.I2V *= self.r             ## Multiply current w/ resistance, (overwrite I)
        np.subtract(self.v_rest, neur.V, out=upd.dV)   # upd.dV[...] = self.v_rest-neur.V  ## OVERWRITE previous dV
        np.add(upd.dV, upd.I2V, out=upd.dV)             # upd.dV += upd.I2V             ## Add in place (overwrite dV)
        np.multiply(upd.dV, self.dt_tau, out=upd.dV)    # upd.dV *= self.dt_tau         ## Scale in place (w/ timestep over membrane-timeconstant)


   ######################################################################################
##########################   Thresholds   ####################################################
   ######################################################################################

class StaticThreshold(Threshold):
    writes = ("rec_spikes",)
    requires = ()
    def __init__(self, cfg: ThresholdCfg, sim_cfg: SimulationCfg=None):
        if cfg is None:
            raise ValueError("StaticThreshold requires ThresholdCfg.")
        if cfg.tau_th is not None or cfg.adpt_th is not None:
            logger.warning("StaticThreshold will ignore inputs in tau_th and adpt_th in ThresholdCfg.")
        if cfg.t_refrac is not None:       
            logger.warning("StaticThreshold will ignore inputs in t_refrac in ThresholdCfg.")
        
        self.v_th = np.float32(cfg.v_th)
        self.v_reset = np.float32(cfg.v_reset)

    def step(self, ctx: RunContext) -> None:
        upd = ctx.upd
        neur = ctx.neur
        np.greater_equal(neur.V, self.v_th, out=upd.rec_spikes)

class AdaptiveThreshold(Threshold):
    """Adaptive spiking threshold mechanism.""" #βθ
    
    writes = ("rec_spikes", "dV_th")
    requires = ("V_th", "dV_th")
    def __init__(self, cfg: ThresholdCfg, sim_cfg:SimulationCfg):
        if cfg is None:
            raise ValueError("AdaptiveThreshold requires ThresholdCfg.")
        if sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("AdaptiveThreshold requires SimulationCfg with dt.")
        if cfg.tau_th is None or cfg.tau_th <= 0:
            raise ValueError("AdaptiveThreshold requires tau_th > 0.")
        if cfg.adpt_th is None:
            raise ValueError("AdaptiveThreshold requires adpt_th.")
        if cfg.t_refrac is not None and type(self) is AdaptiveThreshold:     
            logger.warning("AdaptiveThreshold will ignore inputs in t_refrac in ThresholdCfg.")
        
        self.v_reset = np.float32(cfg.v_reset)
        self.v_th = np.float32(cfg.v_th)   #Default threshold in adaptive case
        self.tau_th = np.float32(cfg.tau_th)        ##for derived_cfg (storage)
        self.dt_tau = np.float32(sim_cfg.dt/cfg.tau_th)
        self.adpt_th = np.float32(cfg.adpt_th)  ##Threshold adaption on spike


    def step(self, ctx: RunContext) -> None:
        """
        Adaptive threshold update:  
        Decay:              dV_th ← (v_th - V_th) * dt / tau_th                    
        Decay code:   upd.dV_th = (self.v_th - neur.V_th) * dt_tau
        Spikeadaption:          upd.dV_th + self.adpt_th  #adaption increment on spike
        """
        upd = ctx.upd
        neur = ctx.neur

        # Threshold decay for all neurons
        np.subtract(self.v_th, neur.V_th, out=upd.dV_th)                   # upd.dV_th = self.v_th - neur.V_th
        np.multiply(upd.dV_th, self.dt_tau, out=upd.dV_th)                  # upd.dV_th *= self.dt_tau

        #Emit spikes
        np.greater_equal(neur.V, neur.V_th, out=upd.rec_spikes)              # -> BOOL; spike=TRUE=1 per neuron

        #Spike adaption to spiking neurons
        np.add(upd.dV_th, self.adpt_th, out=upd.dV_th, where=upd.rec_spikes)   # upd.dV_th[upd.rec_spikes] += self.adpt_th 

        ###Neat pytorch version for later: upd.dV_th.add_(upd.rec_spikes.to(upd.dV_th.dtype), alpha=self.th_adaption)
        #### NOTE MEMBRANE V RESET NOT APPLIED. This is left to the orchestrator


class RefractoryAbs(AdaptiveThreshold):
    """
    Absolute refractory period + adaptive thresholding

    Refractory period, Dense, timebased: 
        t_refrac = refractory period
        refrac_steps = # of simulation-steps this is equivalent to 
        _refrac_count = COUNTDOWN of refrac_step until refrac_period ends
    """

    writes = ("rec_spikes", "dV_th")
    requires = ("V_th", "dV_th")

    def __init__(self, cfg: ThresholdCfg, sim_cfg: SimulationCfg):
        super().__init__(cfg, sim_cfg)
        if cfg.t_refrac is None or cfg.t_refrac <= 0:   #NOTE. D&C uses 5 ms exc / 2 ms inh.
            raise ValueError("RefractoryAbs requires t_refrac > 0.")   
        self.refrac_steps = int(np.ceil(cfg.t_refrac / sim_cfg.dt))  ##NOTE np.ceil forces round-UP to nearest step in simulation time (closest discretion) so that refractory becomes MINIMUM this value. Derived cfg catches the actual n_steps that was used
        self.t_refrac = np.float32(self.refrac_steps * sim_cfg.dt)   ## derived cfg catches the actual minimum refrac time (as a result of np.ceil as seen above)
        self.Refrac_count = None                                     ##module-lvl dynamical state (state_dict)

    def configure_for_network(self, net) -> None:
        self.Refrac_count = np.zeros(net.n_post, dtype=np.int32)

    def step(self, ctx: RunContext) -> None:
        super().step(ctx) #AdaptiveThreshold step
        upd = ctx.upd
        refractory = self.Refrac_count > 0
        suppressed_spikes = upd.rec_spikes & refractory
        if suppressed_spikes.any():
            upd.rec_spikes[suppressed_spikes] = False
            np.subtract(upd.dV_th, self.adpt_th, out=upd.dV_th, where=suppressed_spikes)  #upd.dV_th[suppressed_spikes] -= self.adpt_th 
            ##NOTE the subtract above UNDOES computation. Better alternative -> NEW class that AVOIDS add-then-subtract**  later
        
        np.maximum(self.Refrac_count - 1, 0, out=self.Refrac_count) ##Countdown
        self.Refrac_count[upd.rec_spikes] = self.refrac_steps #start new countdown on spikes
        ##NOTE the CURRENT refrac does a NUMBER of gather/scatter ops. CAN this be bypassed? ->probably not since syns are scattered, as are spikes


   ######################################################################################
##########################   Plasticities   ##################################################
   ######################################################################################

class TraceSTDP(Plasticity):
    """
    Pair-based additive STDP with hard bounds. (== the rule in Brian2's STDP example (after Song, Miller & Abbott 2000). 
    Excitatory presynaptic columns use +eta_exc; inhibitory presynaptic columns use -eta_inh
    (inverted STDP, homeostatic load-balancing).

    Traces are PER-NEURON, not per-synapse: for homogeneous tau this is the exact reduction of Brian2's
    per-synapse Apre/Apost (every synapse of one neuron carries the same number). Verified bit-for-bit.
    """
    weights_format = "dense"
    writes = ("dW",)
    requires = ("pre_spike", "post_spike")

    def __init__(self, cfg:PlasticityCfg, sim_cfg: SimulationCfg):
        if cfg is None:
            raise ValueError("TraceSTDP requires PlasticityCfg.")
        if sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("TraceSTDP requires SimulationCfg with dt.")
        if cfg.tau_pretrace <= 0:
            raise ValueError("TraceSTDP requires tau_pretrace > 0.")
        if cfg.tau_posttrace <= 0:
            raise ValueError("TraceSTDP requires tau_posttrace > 0.")
        if cfg.eta_inh is None:
            raise ValueError("TraceSTDP requires eta_inh.")

        self.tau_pretrace = np.float32(cfg.tau_pretrace)
        self.tau_posttrace = np.float32(cfg.tau_posttrace)
        self.A_ltp = np.float32(cfg.A_ltp)
        self.A_ltd = np.float32(cfg.A_ltd)
        self.eta_exc = np.float32(cfg.eta_exc)
        self.eta_inh = - np.abs(np.float32(cfg.eta_inh))        # Ensure NEGATIVE eta_inh sign (moved up from inside attribute_eta)
        self.w_min = None if cfg.w_min is None else np.float32(cfg.w_min)
        self.w_max = None if cfg.w_max is None else np.float32(cfg.w_max)
        ##Assuming CONSTANT dt (timebased updates). -> later: timebased=flag
        self.decay_pretrace = np.float32(np.exp(-sim_cfg.dt / self.tau_pretrace))
        self.decay_posttrace = np.float32(np.exp(-sim_cfg.dt / self.tau_posttrace)) 
        ###Instantiation: None. -> attribute_eta: network knows WHICH neurons-slices are inh / exc -> eta_pre filled in
        self.eta_pre = None ###
        ##Module-OWNED state (Capitalised = state_dict). Allocated in configure_for_network, where sizes are known.
        self.Pre_trace = None
        self.Post_trace = None
        self._ltd = None        ##scratch; underscore = neither params nor state_dict

    def configure_for_network(self, net):
        self.conf_states(net)
        self.conf_eta(net)

    def conf_states(self, net) -> None:
        """Allocate the module-owned traces and the ltd scratch (sizes from net)."""
        self.Pre_trace = np.zeros(net.n_total, dtype=np.float32)
        self.Post_trace = np.zeros(net.n_post, dtype=np.float32)
        self._ltd = np.zeros((net.n_post, net.n_total), dtype=np.float32)

    def conf_eta(self, net) -> None:
        """Attribute eta_inh to synapses with inhibitory presynaptic neurons, and eta_exc to the rest.  """
        eta_pre = np.full(shape= net.n_total, fill_value= self.eta_exc, dtype=np.float32) # presynaptic neurons excitatory -> assigned to all
        eta_pre[net.inh_slice] = self.eta_inh                       # presynaptic neurons inhibitory -> OVERWRITE 
        self.eta_pre = eta_pre

        logger.debug("TraceSTDP bound to network: n_total=%d inh_slice=%s (eta_exc=%s, eta_inh=%s)",
                     net.n_total, net.inh_slice, self.eta_exc, self.eta_inh)

    def update_traces(self, neur: NeuronalState):
        #Pre_trace[...]  = decay_pretrace  * Pre_trace  + neur.pre_spike 
        np.multiply(self.Pre_trace, self.decay_pretrace, out=self.Pre_trace) 
        np.add(self.Pre_trace, neur.pre_spike, out= self.Pre_trace) 
          
        #Post_trace[...] = decay_posttrace * Post_trace + neur.post_spike 
        np.multiply(self.Post_trace, self.decay_posttrace, out=self.Post_trace) 
        np.add(self.Post_trace, neur.post_spike, out= self.Post_trace) 

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        
        ##Calculate LTP, LTD & apply LTP-LTD
        #upd.dW[...] = self.A_ltp * np.outer(neur.post_spike, Pre_trace)
        np.multiply(neur.post_spike[:, None], self.Pre_trace[None, :], out=upd.dW)
        np.multiply(upd.dW, self.A_ltp, out=upd.dW)

        #NO SCRATCH LTD VARIANT: 
        #upd.dW[...] -= self.A_ltd * np.outer(Post_trace, neur.pre_spike) *NOTE generates NEW array on np.outer()
        np.multiply(self.Post_trace[:, None], neur.pre_spike[None, :], out=self._ltd) 
        np.multiply(self._ltd, self.A_ltd, out=self._ltd)           
        np.subtract(upd.dW, self._ltd, out=upd.dW)

        self.update_traces(neur)

        # per-column vector (eta_pre = +eta_exc | -eta_inh) 
        # upd.dW[...] = eta_pre * (ltp - ltd), 
        if self.eta_pre is None:
            raise RuntimeError("eta_pre (plasticity) is None")
        np.multiply(upd.dW, self.eta_pre[None, :], out=upd.dW)

        ##Null out frozen in dW
        for proj in syn.freeze_slices:
            upd.dW[proj.post, proj.pre] = 0.0
        np.multiply(upd.dW, syn.syn_mask, out=upd.dW)


class SparseTraceSTDP(TraceSTDP):
    """
    The SAME rule over the edge list. Evaluates the identical scalar expression at the edges that exist:

        timestepped & hybrid_evemt: Both use per-neuron trace-decays that are updated every timestep. 
            dW = (A_ltp * post_spike[:,None] * Pre_trace[None,:] - A_ltd * Post_trace[:,None] * pre_spike[None,:]) * eta_pre[None,:]

        event: Applies lazy decay updates only ON spike (right before on-spike adaption is applied). dW not bit-identical due to rounding-off drift:
        Pre_trace[pre_k] / Post_trace[post_k] -> Pre_e[k] / Post_e[k] * decay**n  (n = train steps since k was last touched)

    TraceSTDP over the edge list (same rule, same operand order -> bit-identical to dense). update_on:
      timestep      per-neuron traces, all synapses, returns full dw
      hybrid_event  per-neuron traces, only synapses with a spiking end, returns (k, dw_k); bit-identical to timestep
      event         per-SYNAPSE traces (Pre_e, Post_e, T_e) decayed lazily on touch, returns (k, dw_k); equal to float rounding
    """

    weights_format = "sparse"

    def __init__(self, cfg: PlasticityCfg, sim_cfg: SimulationCfg):
        super().__init__(cfg, sim_cfg)
        self.update_on = sim_cfg.update_on

    def conf_states(self, net) -> None:
        if self.update_on == "event":   ##per-synapse traces + last-event step; no per-neuron traces
            self.Pre_trace = None; self.Post_trace = None
            self.Pre_e = np.zeros(net.n_synapses, dtype=np.float32); self.Post_e = np.zeros(net.n_synapses, dtype=np.float32)
            self.T_e = np.zeros(net.n_synapses, dtype=np.int64)
        else:
            self.Pre_trace = np.zeros(net.n_total, dtype=np.float32)
            self.Post_trace = np.zeros(net.n_post, dtype=np.float32)
        self._ltd = None                 ##per-edge scratch is a local temp

    def _rule(self, S_pre: np.ndarray, S_post: np.ndarray, pre: np.ndarray, x_pre: np.ndarray, x_post: np.ndarray) -> np.ndarray:
        """The scalar expression per edge, given that edge's spikes and traces (shared by both trace layouts). Same operand order as dense."""
        ltp = S_post * x_pre
        np.multiply(ltp, self.A_ltp, out=ltp)
        ltd = x_post * S_pre
        np.multiply(ltd, self.A_ltd, out=ltd)
        np.subtract(ltp, ltd, out=ltp)
        np.multiply(ltp, self.eta_pre[pre], out=ltp)
        return ltp

    def _dw_at(self, syn: SynapticState, neur: NeuronalState, k: np.ndarray | None) -> np.ndarray:
        """dw for edge subset k (None = all edges), per-neuron traces (update_on timestep / hybrid_event)."""
        pre, post = (syn.pre, syn.post) if k is None else (syn.pre[k], syn.post[k])
        return self._rule(neur.pre_spike[pre], neur.post_spike[post], pre, self.Pre_trace[pre], self.Post_trace[post])

    def _step_event(self, ctx: RunContext) -> tuple:
        """
        Event mode: the rule at the plastic synapses a spike touches, on per-synapse traces brought up to date lazily.
        Same order as dense: the rule reads the traces as they were at the start of the step, then updates them (decay, + spike).
        Time base t = sim.train_step, so eval steps don't age the traces. Returns (k, dw_k)."""
        
        neur, syn, t = ctx.neur, ctx.syn, ctx.sim.train_step
        k = touched(syn, neur.post_spike > 0, neur.pre_spike > 0)
        if not k.size:
            return k, np.zeros(0, dtype=np.float32)
        pre, post = syn.pre[k], syn.post[k]
        S_pre, S_post = neur.pre_spike[pre], neur.post_spike[post]
        n = elapsed(self.T_e, k, t)
        x_pre, x_post = decayed(self.Pre_e, k, self.decay_pretrace, n), decayed(self.Post_e, k, self.decay_posttrace, n)
        dw = self._rule(S_pre, S_post, pre, x_pre, x_post)
        self.Pre_e[k] = x_pre * self.decay_pretrace + S_pre        ##update_traces' order, per edge: decay, then + spike
        self.Post_e[k] = x_post * self.decay_posttrace + S_post
        self.T_e[k] = t
        return k, dw                                                    ##the pair; k is already restricted to plastic edges

    def step(self, ctx: RunContext):
        neur, syn = ctx.neur, ctx.syn
        if self.update_on == "event":
            return self._step_event(ctx)                                ##(true event-based)
        if self.update_on == "hybrid_event":                            #HYBRID EVENT: only edges whose pre OR post spiked this step can have nonzero dw; the traces still decay every step below.
            k = touched(syn, neur.post_spike > 0, neur.pre_spike > 0)   ##NOTE: the PLASTIC touched edges (spiking_syns + plastic_mask) and the pair (k, dw_k) returned; was a zero-filled (n_synapses,) dw, masked below
            dw = self._dw_at(syn, neur, k)
            self.update_traces(neur)
            return k, dw
        dw = self._dw_at(syn, neur, None)
        self.update_traces(neur)
        ##WHERE the update may apply: plastic_mask is realized AND non-frozen, one bool per edge.
        if syn.plastic_mask is not None:
            np.multiply(dw, syn.plastic_mask, out=dw)
        return dw


   ######################################################################################
##########################   Noise   #########################################################
   ######################################################################################

class TestNoise(Noise):
    __slots__ = ("sigma_mem", "mu_mem", "noise_rng")     

    def __init__(self, cfg: NoiseCfg, sim_cfg: SimulationCfg, rng: RNGHandler):
        if cfg is None:
            raise ValueError("Noise requires NoiseCfg.")
        if rng is None:
            raise ValueError("Noise requires RNGHandler for its keyed streams.")
        if cfg.sigma_mem < 0:
            raise ValueError("Noise requires sigma_mem >= 0.")

        self.sigma_mem = np.float32(cfg.sigma_mem)
        self.mu_mem = np.float32(cfg.mu_mem)
        ##keyed streams from the RNGHandler: same name -> same stream, order-independent
        self.noise_rng = {name: rng.rng("noise", name) for name in ("synapse", "membrane", "threshold")}   
        
    def on_membrane(self, ctx: RunContext, phase: Phase):
        ## N, array
        upd = ctx.upd
        ##NOTE BEST for MINDRE arrays (ALT genereres hver gang, men innebygget C++ kode gjør det mer effektivt enn repetitiv allokering via += *=)
        noise_mem = self.noise_rng["membrane"].normal( loc=self.mu_mem, scale=self.sigma_mem, size=upd.dV.shape,).astype(upd.dV.dtype, copy=False)
        
        ###Phase HOOK to modify noise
        phase.on_noise(ctx, noise_mem)

        np.add(upd.dV, noise_mem, out=upd.dV)



    ####NOTE LATER: Check out this - Normal noise applied to weight update for generalizing effect as is L2 regularization. 
    ## THOUGH NOTE: L2  penalises the gradient norm (-->> flat minima). STDP, has no loss & adds variance with no weights shrinkage toward 0
    ##### CONFIRM: https://www.deeplearningbook.org/contents/regularization.html , P.239, discussed: https://www.reddit.com/r/MLQuestions/comments/s9158p/adding_noise_to_the_weights/
    ##
    # ##NOTE TODO stale code: the base no-op hook (Noise.on_weights) still called by orchestra **
        ##NOTE best for større arrays MEN: scratch array burde allokeres m/ NUMPY løkke    

        ###remove: Buffer use redundant * NUMPY DOCS: normal() assigns new array ANYWAY -> neur_noise becomes a redundant copy       
        # upd.neur_noise[...] = self.noise_rng["membrane"].normal(loc=self.mu_mem, scale=self.sigma_mem,size=upd.dV.shape).astype(upd.dV.dtype, copy=False)
        # np.add(upd.dV, upd.neur_noise, out=upd.dV)
