#checkpoint - in-memory snapshots of a running model + the POLICY that decides when to take them. 

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Literal, Optional
from pathlib import Path
import logging
import re      
from runcfg import CheckpointCfg

import numpy as np

from paths import RUNS_DIR
from utils import _write_checkpoint_npz, _write_snapshot_npz, _is_better     ##NOTE  + the resumable-snapshot writer "_is_better"

logger = logging.getLogger(__name__)



   ######################################################################################
##########################   Snapshot   ###################################################
   ######################################################################################

##IN MEMORY: A Snapshot = everything a run needs to CONTINUE from this point. 
# Dynamic states = nonderivable model. 
# network topology, cfgs and RNG spec => recoverable & referenced by id

@dataclass(slots=True)
class Checkpoint:
    """One in-memory snapshot of the mutable model state. """
    epoch: int
    runstep: int
    score: Optional[float]
    reason: str                                   ##"best" | "scheduled" | "collapse" | "final"
    W: np.ndarray 
    neur: Dict[str, np.ndarray] = field(default_factory=dict) #STATES
    upd: Dict[str, np.ndarray] = field(default_factory=dict) #STATES
    modules: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)   ##module-owned state_dicts, keyed by role
    health: Dict[str, float] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)                     ##NOTE for resumable snapshots; {"stage": s, "i": samples done} (fit ops: {"fold", "epoch", "i"}) -> resume continues from here
    protocol_state: Dict[str, np.ndarray] = field(default_factory=dict)        ##protocol's own state_dict**
    train_step: int = 0                                                        ##sim.train_step, the event-mode time base; restored with runstep by CheckpointPolicy.resume

    def nbytes(self) -> int:
        total = self.W.nbytes
        for store in (self.neur, self.upd):
            total += sum(a.nbytes for a in store.values())
        return int(total)


## _NEUR_FIELDS = the State arrays a snapshot copies: the state one timestep HANDS to the next
## _UPD_FIELDS = () : UpdateState = per-step SCRATCH buffer: every module rewrites its buffer so a resumed run recomputes them
_NEUR_FIELDS = ("V", "V_th", "Spikes", "pre_spike", "post_spike", "Since")  
_UPD_FIELDS = ()      


def take_snapshot(ctx, *, epoch: int, score: float | None, reason: str,
                  health: Dict[str, float] | None = None, modules: dict = None,
                  progress: dict | None = None, protocol_state: dict | None = None) -> Checkpoint:
    """Copy the mutable model state out of a live RunContext. Copies, never views."""
    neur = {name: getattr(ctx.neur, name).copy()
            for name in _NEUR_FIELDS if getattr(ctx.neur, name, None) is not None}
    upd = {name: getattr(ctx.upd, name).copy()
           for name in _UPD_FIELDS if getattr(ctx.upd, name, None) is not None}
    mods = {name: {k: v.copy() for k, v in m.state_dict.items()} for name, m in (modules or {}).items()}
    proto = {k: np.array(v, copy=True) for k, v in (protocol_state or {}).items()}
    return Checkpoint(epoch=int(epoch), runstep=int(ctx.sim.current_runstep),
                      score=None if score is None else float(score), reason=reason,
                      W=ctx.syn.W.copy(), neur=neur, upd=upd, modules=mods, health=dict(health or {}),
                      progress=dict(progress or {}), protocol_state=proto, train_step=int(ctx.sim.train_step))


def restore_snapshot(ctx, snapshot: Checkpoint, modules: dict = None) -> None:
    """
    Write a snapshot back into a live RunContext, in place (shapes must already match). Leaves sim.current_runstep / current_epoch / train_step alone
    
    Set by CheckpointPolicy.resume. finalize() restores the best epoch INSIDE a running fit."""
    np.copyto(ctx.syn.W, snapshot.W)
    for name, array in snapshot.neur.items():
        np.copyto(getattr(ctx.neur, name), array)
    for name, array in snapshot.upd.items():
        np.copyto(getattr(ctx.upd, name), array)
    for name, state in snapshot.modules.items():
        if modules and name in modules:
            modules[name].load_state_dict(state)
    logger.info("Restored checkpoint from epoch %d (runstep %d, score=%s, reason=%s).",
                snapshot.epoch, snapshot.runstep, snapshot.score, snapshot.reason)


   ######################################################################################
##########################   Checkpoint policy (behaviour)   ##############################
   ######################################################################################

class CheckpointPolicy:
    """
    Decides WHEN to snapshot and keeps the BEST one. Holds the cfg that describes that choice (NOTE duplicate with RunCfg.checkpoint-).
    Run= iteration, checkpoint= retention.
    ##NOTE: best-tracking + restoring ONLY W/ cfg.select="best". W/ "final" (default) validation is observational, finalize() restores nothing.
    Run binds ctx + modules ONCE (explicit arguments still win). new_fit() clears the per-fit bookkeeping.
    on_sample_end()/resume() = resumable snapshots (cfg.every, out_dir).
    """

    def __init__(self, cfg: CheckpointCfg = None, *, runmeta=None, dataset_id: str = None, ctx=None, modules: dict = None, out_dir: Path = None):
        self.cfg = CheckpointCfg() if cfg is None else cfg
        self._runmeta = runmeta          ##provenance for the disk write. None -> in-memory only
        self._dataset_id = dataset_id
        self._ctx = ctx                  ##bound by Run (no reference back to the Run itself)
        self._modules = modules
        self.out_dir = None if out_dir is None else Path(out_dir)   ##where on_sample_end writes; None = no resumable snapshots
        self.best: Optional[Checkpoint] = None
        self.collapse: Optional[Checkpoint] = None      ##the state as the probe flagged collapse; inspectable after the fact
        self.history: list[dict] = []    ##book-keeping only: (epoch, score, kept)

    def bind(self, ctx, modules: dict, *, out_dir: Path = None) -> None:
        """(re)bind the live state this policy snapshots."""
        self._ctx, self._modules = ctx, modules
        if out_dir is not None:
            self.out_dir = Path(out_dir)

    def new_fit(self) -> None:
        """A NEW fit (next fold / the refit): best, collapse and the epoch book are cleared; cfg and binding stay."""
        self.best, self.collapse, self.history = None, None, []

    def _is_better(self, score: float) -> bool:
        """Strictly better by MORE than min_improvement. Equal scores never displace an earlier best, so
        the EARLIEST epoch reaching a plateau wins -- fewer epochs is the better model at equal score."""
        best = None if self.best is None else self.best.score
        return _is_better(score, best, greater_is_better=self.cfg.greater_is_better, min_improvement=self.cfg.min_improvement)

    def on_epoch_end(self, ctx=None, *, epoch: int, score: float | None, modules: dict = None,
                     health: Dict[str, float] | None = None) -> Optional[Checkpoint]:
        """Called once per epoch. Snapshots on IMPROVEMENT only. Returns the snapshot taken, or None.
        ctx/modules default to the bound ones."""
        ctx = self._ctx if ctx is None else ctx
        modules = self._modules if modules is None else modules
        improved = self.cfg.select == "best" and score is not None and self._is_better(score)
        snapshot = None
        if improved:
            snapshot = take_snapshot(ctx, epoch=epoch, score=score, reason="best", health=health, modules=modules)
            self.best = snapshot
            ##Write on IMPROVEMENT: a crash then keeps the best weights so far.
            if self.cfg.to_disk and self._runmeta is not None:
                save_trained_model(snapshot, runmeta=self._runmeta, dataset_id=self._dataset_id,
                                   save_dir=RUNS_DIR if self.out_dir is None else self.out_dir)     ##the run folder when bound
        self.history.append({"epoch": int(epoch), "score": None if score is None else float(score), "kept": improved})
        return snapshot

    def on_collapse(self, ctx=None, *, epoch: int, health: Dict[str, float], modules: dict = None) -> Checkpoint:
        """Snapshot a collapsing run so the state is inspectable AFTER the fact. Always on. ctx/modules default to the bound ones."""
        ctx = self._ctx if ctx is None else ctx
        modules = self._modules if modules is None else modules
        self.collapse = take_snapshot(ctx, epoch=epoch, score=None, reason="collapse", health=health, modules=modules)
        logger.warning("Collapse checkpoint taken at epoch %d: %s", epoch, health)
        return self.collapse

    def finalize(self, ctx=None, modules: dict = None) -> Optional[Checkpoint]:
        """select="best": restore the best snapshot. select="final": no-op (the final state IS the model). Returns the best, if any.
        ctx/modules default to the bound ones."""
        ctx = self._ctx if ctx is None else ctx
        modules = self._modules if modules is None else modules
        if self.cfg.select == "best" and self.best is not None:
            restore_snapshot(ctx, self.best, modules=modules)
        return self.best

    def on_sample_end(self, *, stage: int, position: int, progress: dict | None = None, protocol=None, force: bool = False) -> Optional[Path]:
        """Called by a protocol after every sample it counts.
        Writes out_dir/snap_s<stage:02d>_i<position:08d>.npz when cfg.every divides `position` (or force=True). Needs cfg.every and out_dir.
        protocol = the Protocol whose state_dict is saved as proto__*.
        
        
        NOTE: "<stage:02d>_i" = the pass index:
                * stage = Operations._loop's stage (ex. fold number in kfold)
                * i = position: samples trained so far in that fit
        
        CAROUSEL memory system: only the newest cfg.keep_last files stay, by the (stage, position) parsed from the names. Deletes on disk
            * None = keep all."""
        every = self.cfg.every
        if not every or self.out_dir is None or not (force or position % int(every) == 0):     ##`not every`: None or 0 = never
            return None
        snapshot = take_snapshot(self._ctx, epoch=self._ctx.sim.current_epoch, score=None, reason="scheduled", modules=self._modules,
                                 progress=progress, protocol_state=None if protocol is None else protocol.state_dict)
        path = save_snapshot(snapshot, self.out_dir / f"snap_s{int(stage):02d}_i{int(position):08d}.npz", runmeta=self._runmeta)
        if self.cfg.keep_last:                                              ##CAROUSEL W/ deletion
            mine = [p for p in snapshots(self.out_dir) if _snap_key(p) <= (int(stage), int(position))]   
            for old in mine[:-int(self.cfg.keep_last)]:
                old.unlink()
        return path

    def resume(self, path: Path, *, protocol=None) -> dict:
        """Load a snapshot written by on_sample_end into the bound ctx + modules, and its proto__* into `protocol`.
        Returns the snapshot's progress (the protocol continues from it)."""
        snapshot = load_snapshot(path)
        restore_snapshot(self._ctx, snapshot, modules=self._modules)
        
        self._ctx.sim.current_runstep, self._ctx.sim.current_epoch = snapshot.runstep, snapshot.epoch
        self._ctx.sim.train_step = snapshot.train_step  
        if protocol is not None:
            protocol.load_state_dict(snapshot.protocol_state)
        return snapshot.progress

    def metadata(self) -> dict:
        """JSON-safe. The cfg IS the description of how these choices were made."""
        return {"cfg": asdict(self.cfg),
                "best": None if self.best is None else {"epoch": self.best.epoch, "score": self.best.score, "runstep": self.best.runstep},
                "collapsed": self.collapse is not None, "epochs": self.history}


   ######################################################################################
##########################   TrainedModel (disk)   ########################################
   ######################################################################################

# Two file types:
#   TrainedModel_<run_id>.npz + .json = the MODEL (+ provenance json). Written on improvement (select="best", to_disk; overwritten)
#                                       and once at run end (save_run=True). Load: restore_snapshot(load_trained_model()).
#   snap_s<stage>_i<position>.npz     = a RESUME point: model + protocol state + progress + clock. Written every cfg.every samples,
#                                       keep_last kept. Read by PhasicNet.train(resume_from=).

def save_trained_model(snapshot: "Checkpoint", *, runmeta, dataset_id: str = None,
                       save_dir: Path = RUNS_DIR) -> Path:
    """Write a snapshot + its provenance to runs/TrainedModel_<run_id>.npz (+ .json)."""
    import json
    from dataclasses import asdict as _asdict
    save_dir.mkdir(parents=True, exist_ok=True)
    run_id = getattr(runmeta, "run_id", None) or "unknown"
    stem = save_dir / f"TrainedModel_{run_id}"

    arrays = {"W": snapshot.W}
    arrays.update({f"neur__{k}": v for k, v in snapshot.neur.items()})
    arrays.update({f"upd__{k}": v for k, v in snapshot.upd.items()})
    arrays.update({f"mod__{m}__{k}": v for m, d in snapshot.modules.items() for k, v in d.items()})
    _write_checkpoint_npz(stem, arrays)   

    meta = {"run_id": run_id, "run_name": getattr(runmeta, "run_name", None),
            "network_id": getattr(runmeta, "network_id", None),
            "network_name": getattr(runmeta, "network_name", None),
            "cfg_id": getattr(runmeta, "cfg_id", None), "dataset_id": dataset_id,
            "epoch": snapshot.epoch, "runstep": snapshot.runstep, "train_step": snapshot.train_step, "score": snapshot.score,
            "reason": snapshot.reason, "health": snapshot.health,
            "arrays": sorted(arrays)}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("TrainedModel saved: %s (epoch %d, score=%s)", stem.name, snapshot.epoch, snapshot.score)
    return stem.with_suffix(".npz")


def save_snapshot(snapshot: "Checkpoint", path: Path, *, runmeta=None) -> Path:
    """Write a resumable snapshot (write-then-rename)."""
    import json
    path = Path(path)
    arrays = {"W": snapshot.W}
    arrays.update({f"neur__{k}": v for k, v in snapshot.neur.items()})
    arrays.update({f"upd__{k}": v for k, v in snapshot.upd.items()})
    arrays.update({f"mod__{m}__{k}": v for m, d in snapshot.modules.items() for k, v in d.items()})
    arrays.update({f"proto__{k}": np.asarray(v) for k, v in snapshot.protocol_state.items()})
    arrays.update({f"progress__{k}": np.asarray(v) for k, v in snapshot.progress.items()})
    meta = {"run_id": getattr(runmeta, "run_id", None), "epoch": snapshot.epoch, "runstep": snapshot.runstep, "train_step": snapshot.train_step, 
            "score": snapshot.score, "reason": snapshot.reason, "health": snapshot.health}
    arrays["meta"] = np.asarray(json.dumps(meta, default=str))
    _write_snapshot_npz(path, arrays)
    return path


_SNAP_NAME = re.compile(r"snap_s(\d+)_i(\d+)\.npz") 


def _snap_key(path: Path) -> tuple[int, int] | None:
    """ (stage, position) parsed from a snapshot's name; None for any other file. 
        * snapshots() orders by it, the carousel compares by it."""
    m = _SNAP_NAME.fullmatch(path.name)
    return None if m is None else (int(m.group(1)), int(m.group(2)))


def snapshots(out_dir: str | Path) -> list[Path]:
    """
    the snap_s*_i*.npz of a run folder in (stage, position) order PARSED from the names -> to int numbers 
    Later stages = lowest positions. The last snap = where resume starts."""
    found = [(k, p) for p in Path(out_dir).glob("snap_*.npz") if (k := _snap_key(p)) is not None]
    return [p for _, p in sorted(found)]


def load_snapshot(path: Path) -> "Checkpoint":
    """Inverse of save_snapshot -> a Checkpoint for restore_snapshot (+ protocol_state / progress).
    Also reads a TrainedModel .npz (no proto__/progress__ keys -> empty dicts)."""
    import json
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        W = z["W"]
        neur = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("neur__")}
        upd = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("upd__")}
        proto = {k[len("proto__"):]: z[k] for k in z.files if k.startswith("proto__")}
        progress = {k[len("progress__"):]: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files if k.startswith("progress__")}
        modules = {}
        for k in z.files:
            if k.startswith("mod__"):
                role, name = k[len("mod__"):].split("__", 1)
                modules.setdefault(role, {})[name] = z[k]
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return Checkpoint(epoch=int(meta.get("epoch", 0)), runstep=int(meta.get("runstep", 0)), score=meta.get("score"), reason=meta.get("reason", "scheduled"),
                      W=W, neur=neur, upd=upd, modules=modules, health=meta.get("health", {}), progress=progress, protocol_state=proto,
                      train_step=int(meta.get("train_step", meta.get("runstep", 0))))    


def load_trained_model(path: Path) -> "Checkpoint":
    """Read a TrainedModel back into a Checkpoint, ready for restore_snapshot()."""
    import json
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as z:
        W = z["W"]
        neur = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("neur__")}
        upd = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("upd__")}
        modules = {}
        for k in z.files:
            if k.startswith("mod__"):
                role, name = k[len("mod__"):].split("__", 1)
                modules.setdefault(role, {})[name] = z[k]
    return Checkpoint(epoch=meta["epoch"], runstep=meta["runstep"], score=meta["score"],
                      reason=meta["reason"], W=W, neur=neur, upd=upd, modules=modules, health=meta.get("health", {}),
                      train_step=int(meta.get("train_step", meta["runstep"])))     