#protocols - experimental protocol

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any     
import logging

import numpy as np

from operations import Operations    
from runcfg import ProtocolCfg
from utils import _capitalize_strip_name

logger = logging.getLogger(__name__)


   ######################################################################################
##########################   Protocol (L1)   ##############################################
   ######################################################################################

class Protocol(ABC):
    """
    ONE protocol run over ONE Run. Subclass: set 'protocol_type' (what RunCfg.protocol names), write execute() from the blocks below.

    Args:
        cfg = the ProtocolCfg (or subclass) from RunCfg.protocol; None = ProtocolCfg() defaults (HoldoutValidation needs nothing else).
        run = the assembled Run (PhasicNet._prep_run). Held strongly; the Run does not know the protocol.
        out_dir = the run FOLDER (PhasicNet.train(out_dir=)): where a protocol writes its own files (results, responses). None = no folder.

    STATE: public ndarray attributes are the protocol's resumable state (state_dict, same rule as Dynamics), saved as proto__* by
    run.checkpoints.on_sample_end and read back by resume(). 'progress' = the dict a resumed run continues from: {"stage": s, "i": samples done}
    from self.ops passes, {"fold": k, "epoch": e, "i": samples done} from the fit ops, anything a hand-written loop puts there otherwise.
    ops = this protocol's Operations (operations.py): the loops with logging, timing and snapshots; 'progress' lives there.
    """
    protocol_type: str = ""
    needs_test_set: bool = False     ##True if execute() reads run.test_set (checked once below)

    def __init__(self, cfg: ProtocolCfg | None, run, *, out_dir: str | Path | None = None):
        self.cfg = ProtocolCfg() if cfg is None else cfg
        if self.protocol_type and _capitalize_strip_name(self.cfg.protocol_type) != _capitalize_strip_name(self.protocol_type):
            raise ValueError(f"{type(self).__name__} expected protocol_type {self.protocol_type!r}, got {self.cfg.protocol_type!r}")
        if run.history is None or not run.history.cfg.record_counts:
            raise RuntimeError("A Protocol reads its responses from history.Counts: HistoryCfg.record_counts must be True.")
        if self.needs_test_set and (run.test_set is None or len(run.test_set) == 0):
            raise ValueError(f"{type(self).__name__} needs the test set: PhasicNet.train(data, runcfg, test_data=<ModelInput>), or DataCfg.test_fraction > 0.")
        self.run = run
        self.out_dir = None if out_dir is None else Path(out_dir)
        self._restored_by_subclass: tuple = ()   ##arrays a subclass's load_state_dict restores itself (by rows) before calling the base
        self.ops = Operations(run, state=self, out_dir=self.out_dir, report_every=getattr(self.cfg, "report_every", 500),
                              reset_between_samples=self.cfg.reset_between_samples)

    @property
    def progress(self) -> dict:
        return self.ops.progress

    @progress.setter
    def progress(self, value: dict) -> None:
        self.ops.progress = dict(value)

    @property
    def rng(self):
        """The run's RNGHandler: keyed streams, e.g. self.rng.rng("input", key)."""
        return self.run.ctx.rng

    @abstractmethod
    def execute(self) -> dict:
        """The whole protocol, start to finish. Returns the report (PhasicNet.train stores it as run.report)."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared blocks
    # -------------------------------------------------------------------------

    def present(self, frames, target: Any = None, *, group: str | None = None, until: int | None = None) -> np.ndarray:
        """
        ONE sample -> its RESPONSE: spike count per neuron (float64, true_neuronID order; group= one group's band) over the sample, or over its
        first 'until' steps (the rest then runs uncounted). Delegates to Operations.present; per TIME WINDOW: self.ops.binned_counts.
        """
        return self.ops.present(frames, target, group=group, until=until)

    # -------------------------------------------------------------------------
    # State (snapshots / resume) + report
    # -------------------------------------------------------------------------

    @property
    def state_dict(self) -> dict:
        """Public ndarray attributes = the protocol's own resumable state (the rule of Dynamics.state_dict). Saved as proto__* in a snapshot."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and isinstance(v, np.ndarray)}

    def load_state_dict(self, state: dict) -> None:
        """Inverse of state_dict, STRICT: every array of this protocol must be in the snapshot (else it was written by another protocol/version).
        Writes the protocol's OWN arrays (the base rule above) and nothing else: a subclass that adds non-array state to state_dict (DCProtocol's
        json list) reads that itself before calling this."""
        arrays = Protocol.state_dict.fget(self)
        missing = [k for k in arrays if k not in state and k not in self._restored_by_subclass]
        if missing:
            raise RuntimeError(f"{type(self).__name__}: the snapshot has no protocol state for {missing}: it cannot be resumed exactly.")
        for k, a in arrays.items():
            if k in state:
                np.copyto(a, state[k])

    def resume(self, path: str | Path) -> dict:
        """Continue from a snapshot written by run.checkpoints.on_sample_end: W, neuronal + module state and this protocol's state are restored,
        'progress' is set (and returned). PhasicNet.train(resume_from=) calls this before execute()."""
        self.progress = self.run.checkpoints.resume(Path(path), protocol=self)
        logger.info("%s resumed from %s: progress=%s", type(self).__name__, path, self.progress)
        return self.progress

    def report(self) -> dict:
        """The common tail of every report: which protocol, its cfg, the scorer, the run's identity and settings."""
        run = self.run
        return {"protocol": type(self).__name__, "protocol_cfg": asdict(self.cfg), "scorer": run.scorer.metadata(),
                "runmeta": asdict(run.runmeta), "runcfg": asdict(run.runcfg)}


   ######################################################################################
##########################   HoldoutValidation (L2)   ####################################
   ######################################################################################

class HoldoutValidation(Protocol):
    """
    The DEFAULT protocol (ProtocolCfg(), protocol_type "HOLDOUTVAL"): ONE fresh fit on the training set for cfg.epochs.
    ProtocolCfg.val_split = ONE holdout partition of the training set, scored after every epoch (patience, select="best"); val_split None
    = the plain holdout: one train-only fold, no score, no patience.
        PhasicNet(seed).train(data, RunCfg(protocol=ProtocolCfg(epochs=3)))                  the straightforward holdout run
        PhasicNet(seed).train(data, RunCfg(protocol=ProtocolCfg(epochs=3, val_split=0.2)))   with validation
    execute() does NOT run test(): the caller runs run.test() on the held-out TEST set (DataCfg.test_fraction / test_data), CLEARLY separate.
    val_split = the VALIDATION split (inside train); DataCfg.test_fraction / test_data = the TEST split (touched once, by test()).
    Resumable (CheckpointCfg.every + train(out_dir=), train(resume_from=)); exact under select="final", since best / patience restart at a resume.
    """
    protocol_type = "HOLDOUTVAL"

    def partitions(self) -> list:
        return self.ops.partition(val_split=self.cfg.val_split or None)     ##None / 0 = one train-only fold

    def execute(self) -> dict:
        cfg = self.cfg
        self.run.report = {**self.ops.run_folds(self.partitions(), epochs=cfg.epochs, patience=cfg.patience), **self.report()}
        return self.run.report
