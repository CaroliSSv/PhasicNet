#### Model class + methods

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Protocol, Type
import numpy as np
## PLANNED:
import yaml #-> configs as a dict file system (YAML)
# import json -> configs frozen backup to secure reproducibility
# import hashlib -> hash ID and secure reproducibility
# import pytorch -> for port to pytorch. (and to CUDA?)
# import tqdm 

# Preprocessor: Use sklearn to import from MNIST? Or use custom?





###########################################################
########################################################### CURRENT WORK: WIP: TESTING NEURONS
################-----SETTINGS-----#########################

#############NOTE when is FLOAT actually neccecary as opposed to INT or FLOAT32 RATHER THAN 64? CHECK resource-cost

"""class Settings:
    def __init__(self):
        self.globalseed = None
        self.datatype = None
        self.device = None
        self.logger = None"""



########################################################
######################################################## CURRENT WORK: WIP: TESTING NEURONS
################-----MODEL-----#########################



################-----SETUP-----#########################

####----architecture builder plugin-----####
"""class DenseTopology:
    "Topology builder w/ setters for planning a neural network, then build to compile and freeze architecture, + JSONdump instance."
    def __init__(self, seed, ID):
        pass

    pass

"""

####NOTE frozen classes (frozen after architecture compile) must create an UNFROZEN runtime copy OR include only values that DO NOT evolve during runtime.
####NOTE correct solution? 1. NEVER FREEZE the dataclasses (JSON dump = frozen state at the beginning anyway) 2. Freeze then make copy for runtime.

####----DATACLASS-----####
"""
@dataclass(frozen=True)
class TopologyCfg:
    N_in: int
    N_out: int
    T: int
    seed: int
"""


########################################################## CURRENT WORK: WIP: TESTING NEURONS
################-----RUNTIME-----#########################
####----DATACLASSES-----####
##NOTE REWRITE for synruns
"""@dataclass(frozen=True)
class SynapseCfg:
    b: float"""

@dataclass(frozen=True)
class IntegratorCfg:
    dt: float   ####SHOULD move to SNNCfg 
    tau_mem: float
    v_rest: float
    r: float   # resistance (membrane) NOTE CURRENT convention: Capital letters = matrix (ex. V_th as opposed to v_th). 


@dataclass(frozen=True)
class ThresholdCfg:
    v_reset: float
    v_th: float         # default threshold
    dt: Optional[float] = None  ###SHOULD MOVE TO SNNCfg
    tau_th: Optional[float] = None
    adpt_tau: Optional[float] = None

"""@dataclass(frozen=True)
class PlasticityCfg: ## Should all be float? Possibly FLOAT32?
    tau_pretrace: float
    tau_posttrace: float
    LTP_amplitude: float
    LTD_amplitude: float
    eta_exc: float
    eta_inh: float

    ## alternative to a hard clamp: CHECK DYPLÆRING NOTES. NOTE: Prefer leaky threshold or soft threshold
    clamp_min: float  ### NOTE CLAMPING MUST BE PURPOSEFUL AND EXPOSED (+ Warnings added?)  edit
    clamp_max: float ### NOTE CLAMPING EDIT"""

##### RUNTIME config to contain all in one place **Excessive?




# ---------------- STATES ----------------
@dataclass
class NeuronalState:
    V: np.ndarray #float, (N,), Membrane state (voltage)
    Spikes: np.ndarray #bool, (N,), Spike state
    I: np.ndarray #float, (N,), Current state
    V_th: Optional[np.ndarray] = None #float, (N,), Optional adaptive threshold state

"""@dataclass
class SynapticState:
## EARLIER SPLIT Version (V3;;)
#    signage: np.ndarray # attribution of inhibitory/excitatory (- or + applied to conductance)
#    G: np.ndarray # Conductance (neurotransmitter state; absolute weight value) (conductance)

    W: np.ndarray # FULL, dense array of all weights w/ sign according to /source/ inh or exc neuron
    b: Optional[np.ndarray] = None # synaptic bias (sensitivity of individual synaptic receptors
    pre_trace: Optional[np.ndarray] = None
    post_trace: Optional[np.ndarray] = None"""


@dataclass
class UpdateState:
    I: np.ndarray           #float, (N,), Sum of currents
    dV: np.ndarray          #float, (N,), voltageUpdate
    Spiking: np.ndarray      #bool, (N,), spike-state update
    dV_th: Optional[np.ndarray] = None       #float, (N,), Optional adaptive_thresholdUpdate
    dt_counters: Optional[np.ndarray] = None #INT, counts time since last spike


@dataclass(frozen=True)
class RuntimeCfg:
#    SynapseCfg: SynapseCfg
    integratorcfg: IntegratorCfg 
    thresholdcfg: ThresholdCfg
#    PlasticityCfg: PlasticityCfg


########################### ----------------CLASSES------------------------------------
###NOTE WRONG EDIT must REWRITE
"""
class DataLoader:
    def __init__(self, X: np.ndarray):
        self.X = X                      ### save state** ie current X
        self.T = X.shape[0]             ### NOTE why T? Why T?
        self.cursor = 0                 ###################  ***  This is CURRENT input (ie timestep_n = ...) from the beginning
        self.last_shape = X.shape[1:]   ###################  *** This is CURRENT shape, WRONG PLACE**

    def load(self):
        x = self.X[self.cursor]
        self.cursor = (self.cursor + 1) % self.T ###NOTE cursor + 1 better than for v in a? AVOID unoptimized for loops, DO vectorization/optimized runthroughs
        return x
"""


### NOTE REWRITE for syn runs
"""class Synapse:
    def __init__(self, cfg: SynapseCfg): ###STATEFUL? NOTE check when to give state
        self.b = cfg.b #?


    def step(self, synstate: SynapticState, neurstate: NeuronalState, upd: UpdateState):
        upd.I = synstate.W @ neurstate.Spikes if self.b is None else synstate.W @ neurstate.Spikes + self.b
    """    

class LIFIntegrator:
    """Integrate membrane."""
    def __init__(self, cfg: IntegratorCfg): ###STATEFUL? NOTE check when to give state
        self.v_rest = cfg.v_rest
        self.r = cfg.r                  #R for resistance
        self.dt_tau = cfg.dt/cfg.tau_mem  # Euler decay update



    #### MODULE SURFACE
    def step(self, state: NeuronalState, upd: UpdateState) -> None: 
        """
        Integration step: dv ← (v_rest - state.V + R*I) * dt/τ_membrane
        CODE: upd.dV = (self.v_rest - state.V + self.r*state.I) * self.dt_tau
        """
        upd.I *= self.r
        np.subtract(self.v_rest, state.V, out=upd.dV)
        np.add(upd.dV, state.I, out=upd.dV)
        upd.dV *= self.dt_tau



"""        #### EXPONENTIAL DECAY VARIANT
        self.e_dttau = np.exp(-(cfg.dt/cfg.tau_mem))  # Exponential decay update
        np.add(upd.dV, state.I, out=upd.dV)
        upd.dV *= self.e_dttau"""


###############################MAJOR REWRITE NEEDED WRONG ***
########## Check why "v and theta is not accessed by pylance"
class StaticThreshold:
    def __init__(self, cfg: ThresholdCfg):
        self.v_th = cfg.v_th
        self.v_reset = cfg.v_reset

    def step(self, state: NeuronalState, upd: UpdateState) -> None:
        np.greater_equal(state.V, self.v_th, out=upd.Spiking)

class AdaptiveThreshold:
    """Adaptive spiking threshold mechanism.""" #βθ
    def __init__(self, cfg: ThresholdCfg):
        self.v_reset = cfg.v_reset
        self.v_th = cfg.v_th   #Default threshold in adaptive case
        self.dt_tau = cfg.dt/cfg.tau_th ## NOTE
        self.adpt_th = cfg.adpt_th  ##Threshold adaption


    def step(self, state: NeuronalState, upd: UpdateState) -> None:
        """
        Adaptive threshold update:  
        Euler Decay:              dV_th ← (v_th - V_th) * τ_threshold
        Euler Decay code:   upd.dV_th = (self.v_th - state.V_th) * self.τ_threshold 
        Spikeadaption:          upd.dV_th + self.adpt_th  #adaption increment on spike
        """
        # Threshold decay for all neurons
        np.subtract(self.v_th, state.V_th, out=upd.dV_th)
        upd.dV_th *= self.dt_tau

        #Emit spikes
        np.greater_equal(state.V, state.V_th, out=upd.Spiking)

        #Spike adaption to spiking neurons  ###EDIT TO MORE EFFICIENT VARIANT (more efficient than array[...])
        upd.dV_th[upd.Spiking] += self.adpt_th

        ###Neat pytorch version for later: upd.dV_th.add_(upd.Spiking.to(upd.dV_th.dtype), alpha=self.th_adaption)
        #### NOTE MEMBRANE V RESET NOT APPLIED. This is left to the orchestrator

        

    ## alternative to a hard clamp: CHECK DYPLÆRING NOTES. NOTE: Prefer leady threshold or soft threshold
    
"""
### MUST CLEAR UP: 
#       1. REMOVE G and make - and + compliant with inh or exc neurons based on IDX
class STDPPlasticity:
    #STDP CLASS
    def __init__(self, cfg: PlasticityCfg, neur_state, syn_state):   ####### NOTE WRONG?
        self.tau_pretrace = cfg.tau_pretrace
        self.tau_posttrace = cfg.tau_posttrace
        self.LTP_amplitude = cfg.LTP_amplitude
        self.LTD_amplitude = cfg.LTD_amplitude
        self.eta_exc = cfg.eta_exc
        self.eta_inh = cfg.eta_inh
#        # CLAMPING should be EXPOSED (MOVE to SNN?)
#        self.clamp_min = cfg.clamp_min
#        self.clamp_max = cfg.clamp_max

    ##NOTE 
    def update_tracedecays(self, state: SynapticState): 
        #ON SPIKE update trace decay coefficients for pre and post traces# (?)
        self.decay_pretrace  = np.exp(-state.dt_counter / self.tau_pretrace)  ####MATHSTransformation 
        self.decay_posttrace = np.exp(-state.dt_counter / self.tau_posttrace) ####MATHSTransformation

    def update_trace(self, state: SynapticState):
        self.syn.state.pre_trace  = self.decay_pretrace  * state.pre_trace  + pre_spike    ####MATHSTransformation
        self.syn.state.post_trace = self.decay_posttrace * state.post_trace + post_spike   ####MATHSTransformation

    # MATHSTransformation MATHSTransformation - MERGE G and SIGNAGE 
    def update_weights(self, state: SynapticState):
        ##Calculate LTP/LTD and apply trace update#

        #NOTE This section must be rewritten for new E/I with inh exc allocation masks
        # LTP / LTD contributions
        ltp = self.LTP_amplitude * np.outer(post_spike, state.pre_trace)  ####MATHSTransformation 
        ltd = self.LTD_amplitude * np.outer(state.post_trace, pre_spike)  ####MATHSTransformation

        # Apply E/I learning rules NOTE This must be rewritten for new E/I setup
        dG_exc = self.eta_exc * (ltp - ltd)  ####MATHSTransformation
        dG_inh = self.eta_inh * (-ltp + ltd)  ####MATHSTransformation
        ###NOTE remove the below?
        dG = gate_exc * dG_exc + gate_inh * dG_inh ####MATHSTransformation

        # Update magnitudes NOTE This must be rewritten for new E/I setup
        self.syn.state.G += dG   ####MATHSTransformation

        # *** Clamping intentionally removed (by your request) NOTE AGAIN SHOULD CLAMPING BE THERE???  WRONG ERROR

    def step(self, state: SynapticState, upd: UpdateState):
        #
        #One full STDP update step:
        #  1. update decays if needed
        #  2. update traces
        #  3. update weights
          
        # Update decays automatically (cached if dt constant) ##NOTE WRONG REWRITE ADD COUNTER
        if self.decay_pretrace is None or self.decay_posttrace is None:
            self.update_tracedecays(dt)

        st = self.syn.state  ### WHY ST? AGAIN, NOTE EDIT 

        # Update traces


        # Update weights
        self.update_weights(pre_spike, post_spike, gate_exc, gate_inh)"""


"""
############# LATER
class Noise:
    def __init__(self):
        self.placeholder = None
        pass
    pass

############# LATER
# ---------------- 
class HomeostaticSleep:
    def __init__(self):
        self.placeholder = None
        pass
    pass

############# LATER
# ----------------
class DynamicsCapture:
    def __init__(self):
        self.placeholder = None
        pass
    pass
"""

# ---------------- ORCHESTRATOR ---------------- ######MAJOR REWRITE NEEDED. THIS IS WRONG NOTE *** 
class SNN:
    def __init__(self):

        ## Class holders ##NOTE maybe remove this after all?
        self.dataloader_class = None
        self.synapse_class = None
        self.integrator_class = None
        self.threshold_class = None
        self.plasticity_class = None
        self.noise_class = None
        self.phase_class = None
        self.synapticstate_class = None
        self.neuronalstate_class = None
        self.updatestate_class = None
        #Runtime Instatiated class holders
        self.dataloader = None        
        self.synapse = None
        self.integrator = None        
        self.threshold = None
        self.plasticity = None
        self.noise = None
        self.phase = None
        # Configs,      Dataclasses for static parameters
        self.topologycfg = None
        self.runtimecfg = None
        # States,        Dataclasses for dynamic parameters
        self.syn = None
        self.neur = None
        self.upd = None
        # Data and randomseed
        self.data = None
        self.rngs = None

#    def set_integrator(self, integrator_class) -> None:
#        self.integrator_class = integrator_class

    def set_integrator(self, integrator_class) -> None:
        self.integrator_class = integrator_class

    def set_threshold(self, threshold_class) -> None:
        self.threshold_class = threshold_class

    def instantiate_rules(self) -> None:
        if self.integrator_class is None:
            raise RuntimeError("Can't instantiate an integrator object without storing an integrator class first.")
        if self.threshold_class is None:
            raise RuntimeError("Can't instantiate an threshold object without storing an threshold class first.")
        self.integrator = self.integrator_class(self.runtimecfg.integratorcfg)
        self.threshold = self.threshold_class(self.runtimecfg.thresholdcfg)

    def step(self, input):
        # Load data(time_step) -> input nodes in Spikes array
#        self.dataloader.step(input, self.neur)

        # W @ Spikes -> I summed per neuron
#        self.synapse.step(self.neur, self.syn, self.upd)

        # Integrate membrane -> dV per neuron
        self.integrator.step(self.neur, self.upd)
        self.neur.V += self.upd.dV #Update internal state array
        
        # (Update) threshold and 

        self.threshold.step(self.neur, self.upd)
        if self.neur.V_th is not None and self.upd.dV_th is not None:
            self.neur.V_th += self.upd.dV_th       ## ONLY if adaptive threshold in use. Else: V_th = None and dV_th = None
            ###NOTE possible error: what if ONE of the arrays are not NONE? THEN: quiet, static threshold -> sol1. add warning, sol2. automatic recording  
        self.neur.Spikes = self.upd.Spiking

#        self.plasticity.step(self.neur, self.syn, self.upd)
#        self.syn.W += self.upd.dW


###### INPUTS: Spikearray (T, N_ins), 
#       N_in = Electricity_sources/input neurons = W * H * C = Widt_pixel * Height_pixel * (colour) Channels, 
#          T = T_total/dt = timesteps

"""
Source for pyYAML & JSON :
https://python.land/data-processing/python-yaml#Installing_and_importing_PyYAML
# Add encoding to ensure æøå in YAML?
"""
thresholdcfg_test = ThresholdCfg(v_reset=-90, v_th=-55)
integratorcfg_test = IntegratorCfg(dt=1, tau_mem=20, v_rest=-70, r=10)
runtimecfg_test = RuntimeCfg(integratorcfg_test,thresholdcfg_test)
print("yaml data; ", yaml.dump(asdict(runtimecfg_test)))

snn_test = SNN()
snn_test.set_threshold(AdaptiveThreshold)
print(snn_test.threshold_class)

with open('runtimecfg.yaml', 'w') as file:
    yaml.dump(asdict(runtimecfg_test), file)

with open('runtimecfg.yaml', 'r') as file:
    runtimecfg_test2 = yaml.safe_load(file)

print(runtimecfg_test2)
print(type(runtimecfg_test2))  ##Dict


with open("runtimecfg.yaml", "r") as file:
    loaded_dict = yaml.safe_load(file) ##NOTE This can be a quiet object in runtime


runtimecfg_test3 = RuntimeCfg(
    integratorcfg = IntegratorCfg(**loaded_dict["integratorcfg"]),
    thresholdcfg = ThresholdCfg(**loaded_dict["thresholdcfg"]))

print(runtimecfg_test3)
print(type(runtimecfg_test3))  # <class '__main__.RuntimeCfg'>

print(runtimecfg_test)
print(type(runtimecfg_test))
print(type(loaded_dict["thresholdcfg"]["tau_th"])) # <class 'NoneType'> 


"""
Some typical parameters
v_rest = -70
v_th = -55
v_reset = -90 (hyperpolarization)

v_spike_max = +40
ltp = 50% -> 100% +








"""