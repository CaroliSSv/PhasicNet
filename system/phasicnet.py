#### Model class + methods 
from __future__ import annotations
##VERIFIED : input neurons receive NO connections (tests/test_system.py::test_input_neurons_receive_no_synapses).

##REMAINING TODO:
#####1. REF mcoding https://www.youtube.com/watch?v=9L77QExPmI0 , https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging  ->  CLEANUP outputs and SPLIT stdout vs external logs
#######1.1. Add -- || structure around TQDM bars. Currently illegible 
#####2. OPTUNA WRAPPER 
#####3. Synchronize in TORCH variant (GPU-version becomes accessible. RELIES on nn.modules though so might need another orchestra branch)
######3. DELETE all sparse-torch impl BEFORE sync (time)


from dataclasses import fields, replace, dataclass, field, asdict
from typing import Any, Dict, Optional, Protocol, Type, Tuple, Literal
import importlib    
import numpy as np
import logging
## PLANNED:

# import pytorch #-> for port to pytorch. (and to CUDA?) 

##FOR LOGGING

from time import perf_counter, time 
import os
from datetime import datetime
from pathlib import Path
from abc import ABC

from rnghandler import RNGHandler
from builder import NetBuilder, Network
from cfg import Cfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, HistoryCfg, DataCfg
from meta import RunMeta
from synapticindex import SynapticIndex
from synapticindex import SynapticIndex
from utils import _inf_safe, _hash, _read_json_yaml, _read_net_npz, _write_json_yaml, _filepath, _read_history_npz, _write_history_npz, _COMPARISONS, _capitalize_strip_name   
from paths import EXPORT_DIR
from datahandler import DataHandler, _dataset_hash
from modelinput import ModelInput  
from operations import RunStatus   
from runlog import setup_logging, open_run_log, close_run_log   
import capture
from history import RunHistory
from phasescheduler import PhaseScheduler, validate_triggers   
from dynamics import Synapse, Integrator, Threshold, Plasticity, Noise, Phase
from states import SimulationState, NeuronalState, SynapticState, UpdateState, RunContext, DenseWeights, SparseWeights, NEVER_SPIKED  
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, TraceSTDP, TestNoise  
from phases import DefaultPhase, SleepPhase 
from checkpoint import CheckpointCfg, restore_snapshot, save_trained_model
from scorer import ScorerCfg
from run import Run
from protocols import Protocol, HoldoutValidation 
from freeze import _build_freeze_slices, _build_sparse_plastic_mask  
from dynamics import (Synapse as SynapseRole, Integrator as IntegratorRole,
                          Threshold as ThresholdRole, Plasticity as PlasticityRole,
                          collect_requirements, check_role, _configured_for_network)  

setup_logging()     ##runlog.yml via dictConfig, once per process
logger = logging.getLogger(__name__)






###NOTE from development; 
###1. PhasicNet is getting into MASSIVE godclass territory; maybe SPLIT OFF every instance?
###1_2: ALTERNATIVE: create a SEPERATE CACHE system for RUN or RUNTIME? EX. introduce RunContext & RunContext handling (for states, active runtime network, etc.) INSTEAD of having 
#### the "active" variant/ instantiated variant of SUBclasses ALSO stored on PhasicNet




###########################################################
########################################################### 
################-----Orchestrator-----#######################
def _check_weights_format(fmt: str, *modules) -> None:
    """SAFETYVALVE - WIRING CHECK: 
     
     If declared weights_format, module declaration must fit cfg"""
    for module in modules:
        declared = getattr(module, "weights_format", None)
        if declared and declared != fmt:
            raise TypeError(f"{type(module).__name__} is a {declared!r} module but SimulationCfg.weights_format={fmt!r}.")


def _derive_cfg(cfg: Cfg, modules: dict) -> Cfg:
    """
    The cfg ACTUALLY used, read back from the INSTANTIATED modules (dynamics modules)
        -   Uses original cfg to collect the field names; each cfg section is searched ONLY inside its own module (so threshold.tau_ and integrator.tau_ won't collide).
        - Catches (accidental) cfg drift due to module behaviours (ex. clamping/ceilings -> alter input values) -> saves ACTUAL cfg in use inside run 
        - Idempotent and never fed back as input: Optuna -> original (held by PhasicNet) -> derived (held by Run) (avoid repeat resolutions compounding drift)
    """
    sections = {}
    for name in ("synapse", "integrator", "threshold", "plasticity", "noise"):
        section = getattr(cfg, name, None)
        module = modules.get(name)
        if section is None or module is None:
            sections[name] = section
            continue
        used = module.params
        resolved, missing = {}, []
        for f in fields(section):
            if f.name in used:
                v = used[f.name]
                resolved[f.name] = v.item() if isinstance(v, np.generic) else v
            else:
                resolved[f.name] = getattr(section, f.name)
                if f.name != "cfg_type": missing.append(f.name)   
        if missing:
            logger.debug("derived cfg: %s fields not stored by %s (kept as entered): %s", name, type(module).__name__, missing)
        sections[name] = type(section)(**resolved)
    return replace(cfg, **sections)


def _generate_runmeta(cfg_name, run_name, derived_cfg: Cfg, original_cfg: Cfg, runcfg: RunCfg,
                      network_name, network_id, rng_spec, dataset_id=None, dataset_name=None, dataset_id_n_samples=None) -> RunMeta:
    """
    Meta to identify a run
    Args:
        cfg_id hashes the ORIGINAL cfg (Optuna's input, the reproduction key). 
        derived_cfg_id the DERIVED cfg (from instantiated dynamics-modules collection-sweep)
        run_id hashes the DERIVED cfg + RunCfg.hashable() (minus the declared exclusions) + rng_spec
    """
    cfg_id = _hash(_inf_safe(asdict(original_cfg)), digest_size=8)
    derived_cfg_id = _hash(_inf_safe(asdict(derived_cfg)), digest_size=8)
    run_id = _hash(_inf_safe(asdict(derived_cfg) | {"runcfg": runcfg.hashable()} | {"rng": rng_spec}), digest_size=16)
    ##the FINAL-REFIT protocol fixed by RunCfg.
    return RunMeta(cfg_name=cfg_name, cfg_id=cfg_id, derived_cfg_id=derived_cfg_id, run_name=run_name, run_id=run_id,
                   network_name=network_name, network_id=network_id, dataset_name=dataset_name, dataset_id=dataset_id,
                   dataset_id_n_samples=dataset_id_n_samples, globalseed=rng_spec["seed"], rng_spec=rng_spec)   
@dataclass(frozen=True)
class RuntimeClasses:
    """
    The seven class slots of a PhasicNet as ONE pack (set_runtime_classes(pack) == the keyword form). 
        * None = leave that slot as is.
    
    paths() / from_paths(): classes <-> "module.Class" names, saved by Run.save_record as runtime_classes.json.
    NB: a class in a script run as `python file.py` is saved as "__main__.X" and cannot be loaded later -- start scripts via import.
    In  other words: any class that is to be loaded with this system must be defined and imported from a DIFFERENT file
    """
    synapse_cls: type | None = None
    integrator_cls: type | None = None
    threshold_cls: type | None = None
    plasticity_cls: type | None = None
    noise_cls: type | None = None
    phase_classes: tuple | None = None
    protocol_classes: tuple | None = None

    ##registry (i): io <-> file as importable paths
    def paths(self) -> dict:
        """{field: "module.Class"} per non-None slot; the tuple slots (phase_classes, protocol_classes) -> a list of paths."""
        path = lambda c: f"{c.__module__}.{c.__qualname__}"
        return {f.name: ([path(c) for c in v] if isinstance(v, tuple) else path(v))
                for f in fields(self) if (v := getattr(self, f.name)) is not None}

    @classmethod
    def from_paths(cls, d: dict) -> "RuntimeClasses":
        """import every "module.Class" (the module must be importable, i.e. on sys.path as experiments/__init__.py puts them)"""
        def resolve(path: str):
            module, _, name = path.rpartition(".")
            return getattr(importlib.import_module(module), name)
        return cls(**{k: (tuple(resolve(x) for x in v) if isinstance(v, (list, tuple)) else resolve(v)) for k, v in d.items()})


class PhasicNet:
    def __init__(self, rng: RNGHandler | int | None = None):
        ## Class holders #
        self.rng = rng if isinstance(rng, RNGHandler) else RNGHandler(rng)  ##rng: an int seed OR a ready RNGHandler 
        self.builder_cls = NetBuilder
        self.datahandler_cls = DataHandler

        self.synapse_cls = None
        self.integrator_cls = None
        self.threshold_cls = None
        self.plasticity_cls = None
        self.noise_cls = None

        self.phase_classes: Tuple[type[Phase], ...] = ()
        self.protocol_classes: Tuple[type[Protocol], ...] = (HoldoutValidation,)  

        # Configs,      Dataclasses for static parameters #backup storage?
        self.cfg = None
        self.cfg_name = None
        self.runcfg = None
        self.run_name = None

        ############# --> default setup STOP
        #      
        #Runtime Instatiated class holders
        self.builder = None
        self.network = None  ##OUTPUT from builder, used to REALIZE states -> ctx




    ######################################################################################
    ###################################################################################### 
    #################################-----LOAD // SAVE-----###############################

    def load_cfg(self, file: str, fmt: Literal["json","yaml","yml"]|None = None) -> None:
        """Load one canonicalized cfg file into the PhasicNet."""
        file = _filepath(file)
        raw_data = _read_json_yaml(file=file, fmt=fmt)
        self.cfg = Cfg.load(raw_data)
        logger.info("Loaded cfg: %s", self.cfg)

    def load_net(self, file: str, npz_file = None, fmt: Literal["json", "yaml", "yml"] | None = None) -> None:
        """Load one canonicalized-specs-file into the builder (json or yaml). 
        And load one npz file. IF npz_file is None, uses base of the json/yaml file to look for a corresponding npz file."""
        file = _filepath(file)

        if npz_file is None:
            base, _ = os.path.splitext(file)
            npz_file = f"{base}.npz"
        npz_file = _filepath(npz_file)
        
        raw_data = _read_json_yaml(file=file, fmt=fmt)
        syntype, pre_idx, post_idx, W0, positions = _read_net_npz(npz_file)  

        self.network = Network.load(raw_data, syntype=syntype, pre_idx=pre_idx, post_idx=post_idx, W0=W0, positions=positions)
        logger.info("Loaded net: %s.", self.network.meta.network_id)
        logger.debug("Loded inputs: %s", self.cfg)   

    def load_run(self, file: str, fmt: Literal["json","yaml","yml"]|None = None) -> None:
        """Load runcfg and cfg of one canonicalized file into the PhasicNet."""
        file = _filepath(file)
        raw_data = _read_json_yaml(file=file, fmt=fmt)
        self.cfg = Cfg.load(raw_data)
        self.runcfg = RunCfg.load(raw_data)
        logger.info("Loaded runcfg: %s", asdict(self.runcfg))
        logger.info("Loaded cfg: %s", asdict(self.cfg))
        logger.debug("Loaded cfg: %s", asdict(self.cfg))  

    def load_runcfg(self, file: str, fmt: Literal["json","yaml","yml"]|None = None) -> None:
        """
        load ONE runcfg file into the PhasicNet 
        NOTE a protocol cfg subclass is found by protocol_type, so its module must be imported first (same as others)
        
        NOTE Searches in runs/ by default"""
        file = _filepath(file)
        raw_data = _read_json_yaml(file=file, fmt=fmt)
        self.runcfg = RunCfg.load(raw_data)
        logger.info("Loaded runcfg: %s", asdict(self.runcfg))

    def save_runcfg(self, filename: str = None, runcfg: RunCfg | None = None, JSON: bool = False, YAML: bool = True, sort_keys: bool = False, save_to=EXPORT_DIR):
        """
        Save as one runcfg as a json/yaml file
        Default name Runcfg_<dd-mm_HHMMSS>.yaml in runs/. (runcfg isn't hashed)
        """
        runcfg = self.runcfg if runcfg is None else runcfg
        if runcfg is None:
            raise ValueError("No runcfg set")
        if filename is None:
            filename = f"Runcfg_{datetime.now().strftime('%d-%m_%H%M%S')}"
        _write_json_yaml(file=save_to / filename, data=_inf_safe(asdict(runcfg)), JSON=JSON, YAML=YAML, sort_keys=sort_keys, log_label="RunCfg")


    def save_cfg(self, filename: str = None, cfg_name = None, JSON: bool = False, YAML: bool = True, sort_keys: bool = False, save_to=EXPORT_DIR):
    
        if self.cfg is None:
            raise ValueError("No cfg set")
        
        if cfg_name is None:
            cfg_name = self.cfg_name

        cfg_name_entry = {"cfg_name": cfg_name}
        canon_cfg = self.cfg._canonicalize()
        cfg_id = _hash(canon_cfg, digest_size=8)
        cfg_id_entry = {"cfg_id": cfg_id}
        export_cfg = cfg_name_entry|cfg_id_entry| canon_cfg

        if filename is None:
            now = datetime.now().strftime("%d-%m_%H%M%S")
            filename = f"Cfg_{now}_{cfg_name}" if cfg_name is not None else f"Cfg_{now}_{cfg_id}"
        
        file = save_to / filename
        _write_json_yaml(file=file, data=export_cfg, JSON=JSON, YAML=YAML, sort_keys=sort_keys, log_label="Cfg")





    ######################################################################################
    ###################################################################################### 
    ###############################------PRE - SETTERS-----###############################


    _ROLES = {"synapse_cls": SynapseRole, "integrator_cls": IntegratorRole, "threshold_cls": ThresholdRole, "plasticity_cls": PlasticityRole}   ##the role check per slot (noise, phases, protocols have none)

    @property
    def runtime_classes(self) -> RuntimeClasses:
        """(registry (i)): built from its seven class slots"""
        return RuntimeClasses(**{f.name: getattr(self, f.name) for f in fields(RuntimeClasses)})

    def set_runtime_classes(self, synapse_cls=None, integrator_cls=None, threshold_cls=None, plasticity_cls=None, noise_cls=None, phase_classes:tuple=None, protocol_classes: tuple = None):

        """Update one or more classes using class names.
         Empty fields are ignored, leaving corresponding classes unchanged."""

        pack = synapse_cls if isinstance(synapse_cls, RuntimeClasses) else RuntimeClasses(synapse_cls, integrator_cls, threshold_cls, plasticity_cls, noise_cls, phase_classes, protocol_classes)
        updated = False
        for f in fields(RuntimeClasses):
            v = getattr(pack, f.name)
            if f.name in self._ROLES:
                check_role(v, self._ROLES[f.name], f.name)      ##Using abstract classes as a safety system -> avoid misplacing modules
            if v is None:
                continue
            setattr(self, f.name, v)
            ##JOIN for multiple classes (phases), divided by , (tuple), ELSE simply class name retrieved
            names = ", ".join(c.__name__ for c in v) if isinstance(v, tuple) else v.__name__
            logger.info("%s saved as %s.", f.name, names)
            updated = True

        if not updated:
            logger.info("No runtime classes defined or replaced.")




    def _requirements_rules(self):
        if self.cfg is None:
            raise RuntimeError("self.cfg missing.")

        required = {
            "synapse": (self.synapse_cls, self.cfg.synapse),
            "integrator": (self.integrator_cls, self.cfg.integrator),
            "threshold": (self.threshold_cls, self.cfg.threshold),
            "plasticity": (self.plasticity_cls, self.cfg.plasticity),
            "noise": (self.noise_cls, self.cfg.noise)
            }

        missing = []
        for name, (cls_obj, config) in required.items():
            if cls_obj is None:
                missing.append(f"{name}_cls")
            if config is None:
                missing.append(f"cfg.{name}")

        if self.cfg.simulation is None:
            missing.append("cfg.simulation")

        if missing:
            raise RuntimeError(f"Missing runtime requirements: {missing}")
        
    ######################################################################################
    ###################################################################################### 
    ########################------INSTANTIATE INSTANTIATE  RULES -----###########################

    def _instantiate_builder(self):

        self.builder = self.builder_cls(self.rng, pnet=self)
        logger.debug("Instantiated builder: '%s' with %r", type(self.builder).__name__, self.rng)

    def _instantiate_datahandler(self) -> DataHandler:
        datahandler = self.datahandler_cls()
        logger.debug("Instantiated dataloader: '%s' , ", datahandler)
        return datahandler


    def _instantiate_rules(self) -> Tuple[Synapse, Integrator, Threshold, Plasticity, Noise]:
        self._requirements_rules()
        
        synapse= self.synapse_cls(self.cfg.synapse, self.cfg.simulation)
        integrator= self.integrator_cls(self.cfg.integrator, self.cfg.simulation)
        threshold = self.threshold_cls(self.cfg.threshold, self.cfg.simulation)
        plasticity = self.plasticity_cls(self.cfg.plasticity, self.cfg.simulation)
        noise = self.noise_cls(self.cfg.noise, self.cfg.simulation, self.rng)

        ##WIRING CHECK: a module's declared weights_format must match the cfg (fails at build, not in the hot loop)
        _check_weights_format(self.cfg.simulation.weights_format, synapse, plasticity)     

        if self.network is None:
            raise RuntimeError("Cannot instantiate runtime rules without a network. " \
                                "Correct order: " \
                                "1. network.  " \
                                "2.runtime rules (this method), then phases. (Some of these consume network info for network-specific configuration)" \
                                "3. states, then plastic implementation. (These consume network information and runtime-rule parameters)"
                                "(4.history) (These consume network, rules and state information to configure for a run and instantiate a history-object)")

        for module in (synapse, integrator, threshold, plasticity, noise):

            safety_clamp = getattr(module, "_safety_clamp", None)
            if safety_clamp is not None:
                safety_clamp()

            configure_for_network = getattr(module, "configure_for_network", None)
            if configure_for_network is None:
                continue   ##Noise is not a BaseModule and has no such hook
            configure_for_network(self.network)
            if _configured_for_network(module):
                logger.debug("%s configured itself for the network.", type(module).__name__)
            else:
                logger.debug("%s has no network-specific configuration.", type(module).__name__)

        logger.debug("Instantiated runtime-rules: Synapse: %s, Integrator: %s, Threshold: %s, Plasticity: %s, Noise: %s", 
                    type(synapse).__name__, type(integrator).__name__, type(threshold).__name__, 
                    type(plasticity).__name__, type(noise).__name__)
        
        return synapse, integrator, threshold, plasticity, noise


    def _instantiate_phases(self) -> None:
        if self.cfg is None:
            raise RuntimeError("Missing cfg.")
        if not self.phase_classes:
            raise RuntimeError("Missing phase_classes.")
        if DefaultPhase not in self.phase_classes:
            self.phase_classes = (DefaultPhase, *self.phase_classes)

        phase_cfgs = self.cfg.phases

        if not phase_cfgs:
            phase_cfgs = (PhaseCfg(phase_type="DEFAULT", triggers={}, exit_triggers=None, dataloader_active=True),)  

        if not any(_capitalize_strip_name(phase_cfg.phase_type) == "DEFAULT" for phase_cfg in phase_cfgs):
            default_cfg = PhaseCfg(phase_type="DEFAULT", triggers={}, exit_triggers=None, dataloader_active=True)
            phase_cfgs = (default_cfg, *phase_cfgs)

        # Mapping phase_types received from self.phase_classes 
        name_class: dict[str, type[Phase]] = {}

        for phase_cls in self.phase_classes:
            if not issubclass(phase_cls, Phase):
                raise TypeError(f"{phase_cls.__name__} must be a Phase - child-class.")

            name = _capitalize_strip_name(phase_cls.phase_type)
            name_class[name] = phase_cls
                
        phase_rng = {_capitalize_strip_name(c.phase_type): self.rng.rng("phase", _capitalize_strip_name(c.phase_type)) for c in phase_cfgs}
            
        # Instantiate phases 
        phases: dict[str, Phase] = {}

        for phase_cfg in phase_cfgs:
            name = _capitalize_strip_name(phase_cfg.phase_type)

            try:
                phase_cls = name_class[name]
            except KeyError:
                raise RuntimeError(f"No phase class registered in self.phase_classes have phase_type: {name}. "
                                f"Currently stored phase classes: {name_class.keys()}")

            phases[name] = phase_cls(phase_cfg, self.cfg.simulation, rng=phase_rng[name])
            validate_triggers(phase_cfg.triggers, f"phase {name} triggers")               
            validate_triggers(phase_cfg.exit_triggers, f"phase {name} exit_triggers")

        _check_weights_format(self.cfg.simulation.weights_format, *phases.values())
        return phases

    def _instantiate_protocol(self, runcfg: RunCfg, run: "Run", out_dir=None) -> Protocol:
        """Instantiate protocol, looked up in protocol_classes by
        protocol_type. Instantiated AFTER the Run (core) it holds."""
        cfg = runcfg.protocol
        name = _capitalize_strip_name(HoldoutValidation.protocol_type if cfg is None else cfg.protocol_type)
        classes = self.protocol_classes  

        name_class: dict[str, type[Protocol]] = {}
        for protocol_cls in classes:
            if not (isinstance(protocol_cls, type) and issubclass(protocol_cls, Protocol)):
                raise TypeError(f"{getattr(protocol_cls, '__name__', protocol_cls)} must be a Protocol - child-class.")
            name_class[_capitalize_strip_name(protocol_cls.protocol_type)] = protocol_cls

        try:
            protocol_cls = name_class[name]
        except KeyError:
            raise RuntimeError(f"No protocol class registered in self.protocol_classes has protocol_type: {name}. "
                               f"Currently stored protocol classes: {tuple(name_class)}. Pass it via set_runtime_classes(protocol_classes=(...,)).")

        protocol = protocol_cls(cfg, run, out_dir=out_dir)
        logger.info("Instantiated protocol: '%s' (%s).", type(protocol).__name__, name)
        return protocol
    

    ###################################################################################### 
    #######################------Instantiate state-arrays-----############################




    def _implement_freeze(self, ctx: RunContext) -> RunContext:
        """Resolve cfg.plasticity.freeze into the plastic projection blocks on SynapticState.
        Must be called AFTER sparseweights or denseweights are generated"""
        # NOTE     Re-runnable: call again w/ new cfg to re-freeze the SAME network w/ different freeze blocks 
        freeze = () if self.cfg.plasticity is None else getattr(self.cfg.plasticity, "freeze", ())
        slices = _build_freeze_slices(self.network, freeze)
        ##frozen as a DICT keyed by group namees
        by_slices = {tuple(v): k for k, v in self.network.synapseslices.items()}
        ctx.syn.frozen = {by_slices.get(tuple(p), "all"): p for p in slices}
        if not isinstance(ctx.syn.weights, DenseWeights):
            ctx.syn.weights.plastic_mask = _build_sparse_plastic_mask(ctx.syn, self.network.n_total)
        return ctx

    def _to_dense(self) -> DenseWeights:
        net = self.network
        n_post = net.n_post

        ##Dense matrix
        W = np.zeros((n_post, net.n_total), dtype=np.float32)
        W[net.post_idx, net.pre_idx] = net.W0

        #Mask w/ realized synapses (all to all comp computes dW EVERYWHERE, the FILTER for dW -> W)
        _syn_mask = np.zeros((n_post, net.n_total), dtype=bool)
        _syn_mask[net.post_idx, net.pre_idx] = True

        return DenseWeights(W, _syn_mask)

    def to_sparse(self) -> SparseWeights:
        """
        The sparse counterpart of _to_dense()
        """
        net=self.network
        pre = np.asarray(net.pre_idx, dtype=np.int32)      
        post = np.asarray(net.post_idx, dtype=np.int32)
        W = np.array(net.W0 , dtype=np.float32)

        order = np.lexsort((pre, post))          ##primary post, (ensure same row-major that dense maths does)
        pre, post, W = pre[order], post[order], W[order]
        index = SynapticIndex(pre, post, net.n_post, net.n_total)
        return SparseWeights(W=W, pre=pre, post=post, n_post=net.n_post, index=index)



    
    def _instantiate_states(self, *, synapse, integrator, threshold, plasticity, noise, phases = None) -> RunContext:
        def _optional(name, fill_value=0.0, offset_affected = False):
            """Allocate optional array if name in required_arrays, else return None.
             
              IF array allocated: 
              - np.full generates a 1D array w/:

                    fill_value=fill_value 
                    shape = n_post (if offset_affected), else n_total."""
            if name not in required_arrays:
                return None
            if offset_affected:
                return np.full(n_post, fill_value, dtype=np.float32)
            else:
                return np.full(n_total, fill_value, dtype=np.float32)

        if phases is None:
            raise RuntimeError("_instantiate_states requires phases: call _instantiate_phases() first due to seeding." \
                               "Correct order: " \
                                "1. network.  " \
                                "2.runtime rules, then phases. These might use network info to allocate ex. eta_inh ONCE" \
                                "3. states (this method), then plastic implementation. These use runtime rules and network info to allocate arrays and seed time_in_phase (variable length dict) "
                                "(4. history)")
        
        sim = SimulationState()
        sim.time_in_phase = {name: 0 for name in phases}

        n_total = self.network.n_total
        n_input = self.network.n_input
        n_post = self.network.n_post

        required_arrays = collect_requirements(synapse, integrator, threshold, plasticity)
        logger.info("Allocating optional state arrays required by installed modules: %s", sorted(required_arrays))


        ###INITIATE NEURONAL STATE ARRAYS
        V = np.full(n_post, self.cfg.integrator.v_rest, dtype=np.float32)
        V_th = _optional("V_th", self.cfg.threshold.v_th, True)    

        Spikes = np.zeros(n_total, dtype=bool)
        pre_spike = _optional("pre_spike", 0)        
        post_spike = _optional("post_spike", 0, True)
        Since = np.full(n_post, NEVER_SPIKED, dtype=np.int64) if "Since" in required_arrays else None  
        
        neur = NeuronalState(V=V, Spikes=Spikes, pre_spike=pre_spike, post_spike=post_spike, V_th=V_th, Since=Since)

        def _shapes(state) -> dict:
            return {f.name: (None if getattr(state, f.name) is None else getattr(state, f.name).shape) for f in fields(state)}
        logger.debug("Instantiated neuronal state arrays: %s", _shapes(neur))

        ###INITIATE SYNAPTICSTATE ARRAYS
        if self.cfg.simulation.weights_format == "dense":
            weights = self._to_dense()
        else:
            weights = self.to_sparse()

        syn = SynapticState(weights=weights)
        logger.debug("Instantiated synaptic state arrays: W: %s", syn.W.shape)

        ###INITIATE UPDATE STATE ARRAYS
        step_input = np.zeros(n_input, bool)

        I2V = np.zeros(n_post, dtype=np.float32)
        dV = np.zeros(n_post, dtype=np.float32)
        rec_spikes = np.zeros(n_post, dtype=bool)

        dV_th = _optional("dV_th", 0, True)

        dW = np.zeros((n_post, n_total), dtype=np.float32) if self.cfg.simulation.weights_format == "dense" else None

        neur_noise = np.zeros(n_post, dtype=np.float32)

        upd = UpdateState(step_input=step_input, dV=dV, I2V=I2V, dW=dW, rec_spikes=rec_spikes,
                          neur_noise=neur_noise, dV_th=dV_th)
        logger.debug("Instantiated Update-state arrays successfully: %s", _shapes(upd))
        
        ctx = RunContext(sim=sim, upd=upd, neur=neur, syn=syn, net=self.network, rng=self.rng)
        logger.info("Instantiated neuronal-, synaptic- and update-state arrays successfully")
        return ctx


    def _instantiate_history( self, run_name: str | None = None,
                            *, history_cfg: HistoryCfg | None = None, runmeta: RunMeta | None = None, cfg: Cfg = None) -> RunHistory:
        """Build the RunHistory for this run. history_cfg is BOTH the recording request and, stored
        with the finished history, the description of what it contains.
        'cfg' is the DERIVED cfg (what the run actually used); history is the only place it is persisted."""
        if self.cfg is None or self.cfg.simulation is None:
            raise RuntimeError("Cannot instantiate history without cfg.simulation.")
        if history_cfg is None:
            raise RuntimeError("Cannot instantiate history without a HistoryCfg.")

        history = RunHistory(hist_name=run_name, runmeta=runmeta, cfg=history_cfg, derived_cfg=cfg,
            n_total=self.network.n_total, n_input=self.network.n_input,
            dt=float(self.cfg.simulation.dt))

        logger.debug("Instantiated RunHistory '%s'. record_spikes=%s include_input=%s",
                     run_name, history_cfg.record_spikes, history_cfg.include_input)
        return history


    
    ######################################################################################
    ###################################################################################### 
    ##################################------Save/load history -----###################################

    def save_history(self, filename: str | Path | None = None, 
                     *, JSON: bool = True, YAML: bool = False, sort_keys: bool = False, export_dir: Path = EXPORT_DIR) -> None:
        """
        Save RunHistory

        Saves:
            metadata -> JSON/YAML
            arrays   -> NPZ
        """

        if self.history is None:
            raise RuntimeError("No active history. Call start_history() first.")

        self.history.save(filename=filename, JSON=JSON,YAML=YAML, sort_keys=sort_keys, export_dir=export_dir)


    def spikeraster_arrays(self) -> dict[str, np.ndarray]:
        """
        All arrays from history.
        Array keys:
            spike_ptr           events at t = spike_i[spike_ptr[t]:spike_ptr[t+1]]
            spike_i             true_neuronID per event
            phase_starts/_values, epoch_*, datarun_*   segment timelines (not per-spike columns)
        """
        if self.history is None:
            raise RuntimeError("No active history")
        return self.history.arrays()


    def spike_events(self) -> tuple[np.ndarray, np.ndarray]:
        """(spike_t, spike_i) expanded to COO, for consumers that want per-event timestamps.
        Derived from the pointer form; never stored that way."""
        if self.history is None:
            raise RuntimeError("No active history")
        return self.history.events()


    def sparse2dense(self, T: int | None = None) -> np.ndarray:
        """
        Reconstruct T x n_total dense raster from sparse spike events.
        """
        if self.history is None:
            raise RuntimeError("No active history")

        return self.history.sparse2dense(T=T)





    ######################################################################################
    ###################################################################################### 
    ##################################------TRAIN -----###################################


    def _prep_run(self, data, runcfg: RunCfg = None, *, cfg_name=None, run_name=None, dataset_name=None, test_data=None) -> "Run":
        """
        ASSEMBLE one Run. Instantiate locals (clean) -> Run-attributes. NO instantiated states retained

        NOTE:
        PhasicNet = assembler (classes + cfgs) -> Run(instantiated classes etc.)

        ORDER: rules -> phases -> states -> freeze -> derived cfg -> runmeta -> split -> history -> Run.
        data = a ModelInput
        test_data =  a test set already held out (a ModelInput also)
        """
        runcfg = (self.runcfg or RunCfg()) if runcfg is None else runcfg
        for d, what in ((data, "data"), (test_data, "test_data")):    
            if not isinstance(d, ModelInput) and not (d is None and what == "test_data"):
                raise TypeError(f"train() needs a ModelInput as {what}, got {type(d).__name__}: wrap the data first -- ModelInput(X, y), "
                                f"ModelInput.from_tonic(dataset) (any source whose [i] is (frames, target)).")

        datahandler = self._instantiate_datahandler()
        synapse, integrator, threshold, plasticity, noise = self._instantiate_rules()
        phases = self._instantiate_phases()
        ctx = self._instantiate_states(synapse=synapse, integrator=integrator, threshold=threshold, plasticity=plasticity, phases=phases, noise=noise)
        ctx = self._implement_freeze(ctx)

        ##DERIVED CFG from the instantiated modules; ORIGINAL stays on self.cfg (Optuna's input) and is never replaced.
        derived_cfg = _derive_cfg(self.cfg, {"synapse": synapse, "integrator": integrator, "threshold": threshold,
                                             "plasticity": plasticity, "noise": noise})
        ##record the RESOLVED freeze in derived_cfg
        if derived_cfg.plasticity is not None:
            derived_cfg = replace(derived_cfg, plasticity=replace(derived_cfg.plasticity, freeze=tuple(ctx.syn.frozen)))

        for d in (data, test_data):   ##safetyvalve via ModelInput: check data vs network n_input before run
            if d is not None:   
                d.validate_network(self.network)

        ##THE ONE-TIME TEST/TRAIN SPLIT --  Held out identically across every run, --> avoids dataleakage. 
        dataset_id, dataset_id_n_samples = _dataset_hash(data)
        if test_data is None:
            train, test = datahandler.split_test_train(data, dataset_id=dataset_id, data_cfg=runcfg.data, rng=self.rng)
        else:
            train, test = data.subset(np.arange(len(data)), "train"), test_data.subset(np.arange(len(test_data)), "test")  

        test_id = dataset_id if test_data is None else _dataset_hash(test_data)[0]
        train = datahandler.pick(train, runcfg.data.n_train, data_cfg=runcfg.data, rng=self.rng.rng("data", f"pick_train:{dataset_id}"))
        if test is not None:
            test = datahandler.pick(test, runcfg.data.n_test, data_cfg=runcfg.data, rng=self.rng.rng("data", f"pick_test:{test_id}"))

        runmeta = _generate_runmeta(cfg_name=cfg_name, run_name=run_name, derived_cfg=derived_cfg, original_cfg=self.cfg,
                                    runcfg=runcfg, network_name=self.network.network_name, network_id=self.network.network_id,
                                    rng_spec=self.rng.rng_spec, dataset_id=dataset_id, dataset_name=dataset_name, dataset_id_n_samples=dataset_id_n_samples)
        history = self._instantiate_history(run_name, runmeta=runmeta, history_cfg=runcfg.history, cfg=derived_cfg)

        run = Run(synapse=synapse, integrator=integrator, threshold=threshold, plasticity=plasticity, noise=noise, phases=phases,
                  ctx=ctx, datahandler=datahandler, history=history, runcfg=runcfg, train=train, test=test,   ##runmeta reaches Run via history
                  phasescheduler=PhaseScheduler(phases=phases))
        logger.info("_prep_run complete: '%s' ready (run_id=%s, weights=%s, update_on=%s).", run_name, runmeta.run_id,
                    self.cfg.simulation.weights_format, self.cfg.simulation.update_on)
        return run

    def train(self, data, runcfg: RunCfg = None, *, cfg_name=None, run_name=None, dataset_name=None, test_data=None, out_dir=None, resume_from=None,
              save_run=True, save_history=False) -> "Run":
        """
        Assemble a Run and execute it end to end on the TRAIN portion of 'data'.

        Returns the finished Run: its .report is the trial/run result

        calling .test() on it -> defaults to run on held-out set (one hold-out split & save generated during prep_run, based on datacfg instructions)

            data        = a ModelInput data
            test_data   = the user's own held-out set (a ModelInput)
            
            out_dir     = a run FOLDER (receives exports by default)
            resume_from = a snap_*.npz from such a folder (Protocol.resume continues run from it) before execute().
        """
        run = self._prep_run(data, runcfg, cfg_name=cfg_name, run_name=run_name, dataset_name=dataset_name, test_data=test_data)
        protocol = self._instantiate_protocol(run.runcfg, run, out_dir=out_dir)
        run_log = None
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            run_log = open_run_log(out_dir, run.runmeta)     
            self.save_cfg(filename="cfg", cfg_name=self.cfg_name if cfg_name is None else cfg_name, JSON=True, YAML=False, save_to=out_dir)
            run.save_record(out_dir, runtime_classes=self.runtime_classes)
            run.checkpoints.out_dir = out_dir
        elif run.runcfg.checkpoint.every:
            logger.warning("CheckpointCfg.every=%r but no out_dir was given to train(): no resumable snapshots will be written.", run.runcfg.checkpoint.every)   
        ##the experiment boundary:  try/except around a run
        status = RunStatus(out_dir, run, protocol)
        run.status = status.record
        try:
            if resume_from is not None:
                protocol.resume(resume_from)
            run.report = protocol.execute()
            if save_run:
                run.save_trained_model(save_dir=out_dir)
            if save_history:
                run.history.save(export_dir=EXPORT_DIR if out_dir is None else out_dir)     
            status.end("completed")
        except KeyboardInterrupt:
            status.end("interrupted")
            raise
        except BaseException as error:
            status.end("failed", error)
            raise
        finally:
            if run_log is not None:
                close_run_log(run_log)  
        return run
