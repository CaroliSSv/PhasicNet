### IO_utils; loading, saving, etc logic seperated out for easy access  
from __future__ import annotations
from dataclasses import dataclass, asdict, replace, field, fields
from typing import Any, Dict, Optional, Literal, Tuple
import logging
import sys 
import numpy as np
from utils import _capitalize_strip_name


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimulationCfg:
    """
    Cfg w/ runtime shared settings INTEGRAL to maths&code in runtime modules

    Args:

        dt = discrete timestep for simulation
        array_library = "numpy" | "torch"
        weights_format = "dense" | "sparse"
        update_on = "timestep" | "event" | "hybrid_event"
            "timestep"      every synapse every step 
            "event"         plasticity traces live per synapse + lazy decay on spike (applied before spike)
                            The neurons stay clock-driven
            "hybrid_event"  per-neuron traces decayed every step, update synapses with a spiking endpoint. 

    NOTE timestep & hybrid_event = no noise: bit-identical (rounded every dt).
    
        event = NOT bit-identical regardless
    
    """

    dt: float = 1 

    array_library: Literal["numpy", "torch"] = "numpy"
    weights_format: Literal["dense", "sparse"] = "dense"
    update_on: Literal["timestep", "event", "hybrid_event"] = "timestep"

    def __post_init__(self):

        if self.array_library != "numpy":
            raise NotImplementedError("Only NumPy is implemented.")
        ##SAFETYVALVE: event or event hybrid must have sparse weights
        if self.update_on in ("event", "hybrid_event") and self.weights_format == "dense":   
            raise ValueError(f"update_on={self.update_on!r} requires weights_format='sparse'; dense has no edge list to gate on.")

    @property
    def sparse(self) -> bool:
        return self.weights_format == "sparse"



@dataclass(frozen=True, slots=True)
class SynapseCfg:
    """
    Args:
        b = synaptic plate bias (placeholder)
    """ 
    b: Optional[float] = None       
    cfg_type: str = field(default="SynapseCfg", kw_only=True)   ##Default names itself. Overridden by subclasses


@dataclass(frozen=True, slots=True)
class IntegratorCfg:
    """
    Args:
        tau_mem = ms, Time-constant of decay of membrane potential
        v_rest = mV, Resting membrane potential
        r = mΩ, resistance of membrane
    """
    tau_mem: float
    v_rest: float
    r: float   #  NOTE CURRENT convention: Capital letters = matrix (ex. V_th as opposed to v_th). 
    cfg_type: str = field(default="IntegratorCfg", kw_only=True)   ##Default names itself. Overridden by subclasses


@dataclass(frozen=True, slots=True)
class ThresholdCfg:
    """
    Args:
        v_reset = mV, Voltage reset on spike
        v_th = mV, Voltage threshold (base)
        tau_th = ms, threshold decay timeconstant
        adpt_th = mV, Threshold adaption on spike
        t_refrac = ms, absolute refractory period
            ##NOTE RefractoryAbs(AdaptiveThreshold) is the only one currently using refrac, but the refrac extension there CAN be extended to StaticThreshold subclasses, etc)
    """
    v_reset: float
    v_th: float         # default threshold
    tau_th: Optional[float] = None
    adpt_th: Optional[float] = None
    t_refrac: Optional[float] = None ## absolute refractory period in ms. None = no refractory period applied
    cfg_type: str = field(default="ThresholdCfg", kw_only=True)   ##Default names itself. Overridden by subclasses



@dataclass(frozen=True, slots=True)
class PlasticityCfg:
    """
    Args:
        tau_pretrace = ms, Time-constant of pretrace decay
        tau_posttrace = ms, Time-constant of pretrace decay
        A_ltp = Scalar of causal potentiation (pre before post spike)
        A_ltd = Scalar of anticausal depression (post before pre spike)
        eta_exc = Scalar of weight update in synapses with excitatory effect
        eta_inh = Scalar of weight update in synapses with inhibitory effect
        w_min = Minimum synaptic weight (floor)        
        w_max = Maximum synaptic weight (ceiling)
    """
    tau_pretrace: float
    tau_posttrace: float
    A_ltp: float
    A_ltd: float
    eta_exc: float
    eta_inh: Optional[float] = None
    # HARD clamp
    w_min: Optional[float] = None 
    w_max: Optional[float] = None
    freeze: Tuple[str, ...] = ()  ##ex. "input", "inh", "exc", "all", or "<synapse-name>"
    cfg_type: str = field(default="PlasticityCfg", kw_only=True)   ##Default names itself. Overridden by subclasses


@dataclass(frozen=True, slots=True)
class NoiseCfg:
    """
    Args:
        mu_mem = Mean membrane noise
        sigma_mem = Standard deviation of membrane noise
    
    """
    mu_mem: float
    sigma_mem: float
    cfg_type: str = field(default="NoiseCfg", kw_only=True)   ##Default names itself. Overridden by subclasses




@dataclass(frozen=True)
class PhaseCfg:
    """Available modification-parameters (constants) for easy use (Custom behaviour might need a new phasecfg)
    phase_type = the Phase class this cfg is fed to (its "phase_type"); _load_phase_cfg picks the PhaseCfg subclass whose default names it. 
    triggers = ENTRY triggers {capture_name: (value, comparison)}: the phase is entered when they fire (trigger_mode "all" / "any"; PhaseScheduler.step).
    exit_triggers = checked FIRST on the current phase; they fire -> back to DEFAULT
    dataloader_active = False holds the dataloader inplace, (Phase.alternate_inputs) and the data-sample stretches for as long as the phase lasts
    """
    phase_type: str = "DEFAULT"     
    triggers: dict = field(default_factory=dict)
    trigger_mode: Literal["all", "any"] = "all"
    exit_triggers: dict = None
    dataloader_active:bool = True
    
    def __post_init__(self):
        object.__setattr__(self, "phase_type", _capitalize_strip_name(self.phase_type))  ##object.__setattr__ BYPASSES frozen-restriction




@dataclass(frozen=True)
class SleepPhaseCfg(PhaseCfg):
    phase_type: str = "SLEEP"     
    sleep_fraction: float = 0.30
    jolt_magnitude: float = 0.05
    revive_syn_group: str | None = None
    dead_eps: float = 0.0
    noise_scale: float = 3.0
    noise_shift: float = 0.0

 


def _cfg_subclasses(cls: type) -> Tuple[type, ...]:
    """Return tuple of available phasecfg subclasses for use in instantiation (keyed to cfg phase_type and phase type) during loading."""
    classes = [cls]

    for subcls in cls.__subclasses__():
        ##persistence-BUG resolved: Keep only the class its module actually holds.
        if getattr(sys.modules.get(subcls.__module__), subcls.__qualname__, subcls) is not subcls:
            continue
        classes.extend(_cfg_subclasses(subcls))

    return tuple(classes)

def _load_section_cfg(base: type, raw: dict, key: str = "cfg_type"):
    """Finds the subclass of "base" whose DEFAULT of field "key" names raw[key] 
    - (first match, base first) Missing "key" defaults -> base.
    NOTE Subclass must be defined BEFORE this is called."""
    if key not in raw:
        return base(**raw)
    name = _capitalize_strip_name(raw[key])

    for cfg_cls in _cfg_subclasses(base):
        ##NOTE: read the field DEFAULT, since fields() works on finding attributes regardless while getattr() relies on slots=false**        
        default = next((f.default for f in fields(cfg_cls) if f.name == key), "")
        if _capitalize_strip_name(default) == name:
            return cfg_cls(**raw)

    raise RuntimeError(f"No {base.__name__} subclass found for {key} {name!r}. "
                       f"Available {key} names: "
                       f"{tuple(_capitalize_strip_name(next((f.default for f in fields(c) if f.name == key), '')) for c in _cfg_subclasses(base))}. "
                       f"Import the module that defines it before loading.")

def _load_phase_cfg(raw: dict) ->PhaseCfg:
    if "phase_name" in raw:     ##TODO LATER: DELETE. backwards compatibility hook**
        raw = {**raw, "phase_type": raw["phase_name"]}; raw.pop("phase_name")
    return _load_section_cfg(PhaseCfg, raw, key="phase_type")   



@dataclass  ###NOTE EITHER ADD OPTIONALS OR DONT FREEZE.
class DynamicsCfg:
    """
    Container of cfgs with args for model runtime.
    Args:
        simulation: SimulationCfg
        synapse: SynapseCfg
        integrator: IntegratorCfg
        threshold: ThresholdCfg
        plasticity: PlasticityCfg
        noise: NoiseCfg
        phases: PhaseCfg 
        """
    simulation: Optional[SimulationCfg] =   None
    synapse: Optional[SynapseCfg] = None
    integrator: Optional[IntegratorCfg] = None
    threshold: Optional[ThresholdCfg] = None
    plasticity: Optional[PlasticityCfg] = None
    noise: Optional[NoiseCfg] = None
    phases: Tuple[PhaseCfg, ...] = ()

    @classmethod
    def load(cls, raw_data: Dict) -> DynamicsCfg:
        return cls(
            simulation=SimulationCfg(**raw_data["simulation"]) if raw_data.get("simulation") is not None else None,
            synapse=_load_section_cfg(SynapseCfg, raw_data["synapse"]) if raw_data.get("synapse") is not None else None,
            integrator=_load_section_cfg(IntegratorCfg, raw_data["integrator"]) if raw_data.get("integrator") is not None else None,
            threshold=_load_section_cfg(ThresholdCfg, raw_data["threshold"]) if raw_data.get("threshold") is not None else None,
            plasticity=_load_section_cfg(PlasticityCfg, raw_data["plasticity"]) if raw_data.get("plasticity") is not None else None,
            noise=_load_section_cfg(NoiseCfg, {k: v for k, v in raw_data["noise"].items() if k not in ("mu_weights", "sigma_weights")}) if raw_data.get("noise") is not None else None,   ##TODO LATER: DELETE. backwards compatibility hook**
            phases=tuple(_load_phase_cfg(phase_raw) for phase_raw in raw_data.get("phases", ())))
    
    def _canonicalize(self):
        return asdict(self)
    


      
