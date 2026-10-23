
from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict, asdict, replace   
from typing import Dict, Literal, Optional
import numpy as np
from numpy.random import SeedSequence, Generator, PCG64
from hashlib import blake2b
import logging
from utils import _capitalize_strip_name     
from cfg import _load_section_cfg      

logger = logging.getLogger(__name__)



@dataclass(frozen=True, slots=True)
class DataCfg:
    """
    Which samples a run sees: the one-time test/train split (owned by DataHandler) and the per-run pick from each set.
    Train/val splitting is NOT here: see ProtocolCfg.val_split.

    Args:
        make_test_split: create the split if none is saved (datasets/train_test/split_<dataset_id>.json) False = raise instead.
        test_fraction:   share of the dataset held out as TEST; saved once and reused by every run on that dataset.
        stratify:        keep class proportions in every split and pick (test/train, val, n_train / n_test).
        n_train/n_test:  use only this many samples from train / test (None = all). Seeded and stratified.
        shuffle:         present samples in a seeded shuffled order; False = the source's own order (D&C).

    NB: with train(test_data=...) no split is made: the two ModelInputs ARE the sets.
    """
    
    make_test_split: bool = True
    test_fraction: float = 0.15
    stratify: bool = True
    n_train: Optional[int] = None            # N-MNIST one-class crash
    n_test: Optional[int] = None             # N-MNIST one-class crash
    shuffle: bool = False                    # N-MNIST one-class crash
    cfg_type: str = field(default="DataCfg", kw_only=True)   # cfg types: names the class to rebuilds a SUBCLASS


@dataclass(frozen=True, slots=True)
class HistoryCfg:
    """
    WHAT a RunHistory retains. Stored WITH the history as its own description.

    Args:
        record_spikes = capture the spike raster at all (captured in timelines)
        include_input = include input nodes (no membrane; their spikes are the DRIVE, not a response)
        record_phase / record_epoch / record_datarun = keep the matching segment timeline
        chunk_events = event-buffer flush size (purely mechanical memory-preservative. Config to computer+runtime (ideally))
    """
    record_spikes: bool = True
    include_input: bool = False

    record_phase: bool = True
    record_epoch: bool = True
    record_datarun: bool = True
    record_counts: bool = True      ##running spike COUNT per neuron

    cfg_type: str = field(default="HistoryCfg", kw_only=True)   # cfg types: names the class
    chunk_events: int = 60000   ##Number of events before buffer flushes.60000 uint32 events = 234 KB per flush (NOTE very small**) 



@dataclass(frozen=True, slots=True)
class ScorerCfg:
    """
    Thresholds for the health probes, and which score drives model selection.

    Args:
        rate_hz_max = mean recurrent firing rate treated as runaway
        rate_hz_min = mean recurrent firing rate treated as collapse-to-silence
        scorer = Scorer Trial selects on. "vrc" is Calinski-Harabasz.
    """

    rate_hz_max: float = 500.0
    rate_hz_min: float = 0.01
    scorer: str = "vrc"
    cfg_type: str = field(default="ScorerCfg", kw_only=True)   # cfg types: names the class, so RunCfg.load rebuilds a subcls


@dataclass(frozen=True, slots=True)
class CheckpointCfg:
    """
    POLICY for checkpointing. Frozen and archived with the run: it changes which weights are reported.

    Args:
        greater_is_better = direction of "score". True for separability scores, False for losses.
        min_improvement: smallest score gain that counts as a new best (and resets patience).
        to_disk:         also write the best model to TrainedModel_<run_id>.npz in the run folder on each new best.
        every:           write a resume snapshot (snap_s<stage>_i<position>.npz) every N samples. None or 0 = never.
        keep_last:       how many resume snapshots stay on disk. older ones are deleted. None = keep all. Not in run_id.

        select =  WHICH state becomes the fit's model at the end of run_fit 
                * "final" = the state the fit ends in
                * "best" = the best-validation epoch is restored

        content = WHAT a trained model is.
            "full" (default) -> W + neuronal + module state: exact resumption of the simulation.
            "weights"        -> W only: the learned synaptic configuration. test() starts every other state from rest.
    """

    select: Literal["final", "best"] = "final"
    content: Literal["full", "weights"] = "full"
    greater_is_better: bool = True
    min_improvement: float = 0.0
    to_disk: bool = True         ##IO only; excluded from run_id ( RunCfg.HASH_EXCLUDE)
    every: Optional[int] = None  # IO only, excluded from run_id
    keep_last: Optional[int] = 3   ##IO only, excluded from run_id
    cfg_type: str = field(default="CheckpointCfg", kw_only=True)  


@dataclass(frozen=True, slots=True)
class ProtocolCfg:
    """
    Experimental protocol parameters. 
     Same system as phase and dynamics cfgs: a subclass is recognized by it's name (type) which MUST be filled in, and filled in at runstart with the corresponding cfg.  
     
     * Default: protocol_type's by default becomes the class's protocol_type, if no name is explicitly entered

    Args:
        protocol_type = the Protocol class (its "protocol_type") that executes the run. 
        
        Protocol: default HOLDOUTVAL =
            protocols.HoldoutValidation, the straightforward holdout run: ONE fit on the training set for "epochs". 
                * val_split None (default) = no validation (the plain holdout) 
                * val_split given = ONE partition of the training set held out and scored after every epoch
        
        epochs = num of runs through trainingset of data 
        patience = stop a fit after this many epochs without a better validation score (None = never)
        val_split = portion of the TRAINING set held out for validation (0-1); None = none.
        
        reset_between_samples = Applied by Protocol.run_pass / present through Run.reset_transient 
            True  -> V, V_th, traces and Spikes are returned to baseline before each sample. Samples are then INDEPENDENT presentations.
            False -> state carries over. Continual/streaming regime. Legitimate, but then sample ORDER is physically meaningful, so
                     shuffling changes results. EXPLICIT choice in protocol. W always left out
        report_every =  one progress line (elapsed, rate, ETA, spikes/sample) per this many samples in every pass
        
    """
    protocol_type: str = "HOLDOUTVAL" 
    reset_between_samples: bool = False
    report_every: int = 500       
    epochs: int = 1                       
    patience: Optional[int] = None          
    val_split: Optional[float] = None  
           

    def __post_init__(self):
        object.__setattr__(self, "protocol_type", _capitalize_strip_name(self.protocol_type))   ##as PhaseCfg.phase_type (object.__setattr__ bypasses frozen)


##### RUNTIME config to contain all in one place **Excessive?
@dataclass(frozen=True, slots=True)
class RunCfg:

    history: HistoryCfg = field(default_factory=HistoryCfg)
    scorer: ScorerCfg = field(default_factory=ScorerCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    data: DataCfg = field(default_factory=DataCfg)        ##NOTE DataCfg here,  not in DataHandler **
    protocol: ProtocolCfg = field(default_factory=ProtocolCfg)   

    HASH_EXCLUDE = ("history", "checkpoint.to_disk", "checkpoint.every", "checkpoint.keep_last", "protocol.report_every", "scorer.cfg_type", "checkpoint.cfg_type", "data.cfg_type") 

    def hashable(self) -> Dict:
        """asdict() minus HASH_EXCLUDE. Hashable section = what run_id is computedd from."""
        d = asdict(self)
        # the exclusions applied BEFORE the default-protocol comparison, --> make protocols hash-neutral 
        default_protocol = asdict(ProtocolCfg())
        for path in self.HASH_EXCLUDE:
            node, *rest = path.split(".")
            if not rest: d.pop(node, None)
            elif node in d and d[node] is not None: d[node].pop(rest[0], None)
            if node == "protocol" and rest: default_protocol.pop(rest[0], None)
        for key, default in (("n_train", None), ("n_test", None), ("shuffle", False)):   
            if d["data"].get(key) == default: d["data"].pop(key)                         
        if d["protocol"] is None or d["protocol"] == default_protocol:
            d.pop("protocol")     
        return d



    @classmethod
    def load(cls, raw_data: Dict) -> "RunCfg":

        nested = {"history": HistoryCfg, "scorer": ScorerCfg, "checkpoint": CheckpointCfg, "data": DataCfg, "protocol": ProtocolCfg}  
        if "protocol_name" in (raw_data.get("protocol") or {}):     ##BACKWARDS compatibility (TODO later: delete)
            proto = dict(raw_data["protocol"]); proto["protocol_type"] = proto.pop("protocol_name")
            raw_data = {**raw_data, "protocol": proto}
        old = {k: raw_data[k] for k in ("epochs", "val_split", "k_fold", "patience") if k in raw_data}  
        if old:
            proto = dict(raw_data.get("protocol") or {})
            if _capitalize_strip_name(proto.get("protocol_type", "CROSSVAL")) == "CROSSVAL":   
                proto["protocol_type"] = "CROSSVAL" if (old.get("k_fold") or 0) > 1 else "HOLDOUTVAL"   #NOTE TODO capitalize strip name still extended to crossval despite never using it **remove?
            own = {"CROSSVAL": "k_fold", "HOLDOUTVAL": "val_split"}.get(_capitalize_strip_name(proto["protocol_type"]))
            raw_data = {**raw_data, "protocol": {**proto, **{k: v for k, v in old.items() if k in ("epochs", "patience", own)}}}
        proto = raw_data.get("protocol")
        if proto and _capitalize_strip_name(proto.get("protocol_type", "")) == "HOLDOUT":     #
            raw_data = {**raw_data, "protocol": {**proto, "protocol_type": "HOLDOUTVAL"}}
        proto = raw_data.get("protocol")
        if proto and ("val_fraction" in proto or "n_val" in proto):     ##BACKWARDS compatibility (TODO later: delete): HLSP val_fraction / NALSM n_val -> val_split
            raw_data = {**raw_data, "protocol": {**{k: v for k, v in proto.items() if k not in ("val_fraction", "n_val")},
                                                 **({"val_split": proto["val_fraction"]} if "val_fraction" in proto else {})}}
        kwargs = {}
        for f in fields(cls):
            if f.name not in raw_data:
                continue                                    ##missing key -> dataclass default
            value = raw_data[f.name]
            if f.name in nested and value is not None:
                if f.name == "data":
                    value = {k: v for k, v in value.items() if k not in ("warn_seconds", "dataset", "dataset_kw")}   ##BACKWARD compatibility .old files still have data.warn_seconds (skip) NOTE TODO later: delete
                value = _load_section_cfg(nested[f.name], value, key="protocol_type" if f.name == "protocol" else "cfg_type")   ##EVERY section picks its OWN subclas
            kwargs[f.name] = value
        return cls(**kwargs)
    
    def _canonicalize(self):
        return asdict(self)