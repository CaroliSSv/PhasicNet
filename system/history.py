from __future__ import annotations
from dataclasses import asdict
from typing import Optional
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
from runcfg import HistoryCfg
from meta import RunMeta
from paths import EXPORT_DIR
from utils import _filepath, _write_history_npz, _write_json_yaml, _capitalize_strip_name, _inf_safe, _read_history_npz, _read_json_yaml 

logger = logging.getLogger(__name__)

class _Timeline:
    """
    Storage structure: timestep based saves.

    Append start_step-, value- segments. Records a value ONLY when it changes.

     ++- timeline based saves follows actual simtime
    """
    __slots__ = ("name", "starts", "values", "_last")

    def __init__(self, name: str):
        self.name = name
        self.starts: list[int] = []
        self.values: list = []
        self._last = object()          ##sentinel: when nothing recorded **

    def mark(self, step: int, value) -> None:
        if value != self._last:
            self.starts.append(int(step))
            self.values.append(value)
            self._last = value

    def at(self, step: int):
        """Value in force at `step` (after first segment/chunk)."""
        if not self.starts:
            return None
        idx = int(np.searchsorted(np.asarray(self.starts), step, side="right")) - 1
        return self.values[idx] if idx >= 0 else None

    def segments(self) -> list[dict]:
        return [{"start_step": s, "value": v} for s, v in zip(self.starts, self.values)]


MODE_KEY = {"train": 0, "eval": 1}      ##MODE->KEY-int history.mode_timeline / mode_segments / mode_values record


class RunHistory:
    """
    Storage of one run; APPEND ONLY, stored as timestep-pointer (CSR-without-values).

    Spike raster:
        spike_ptr[t]   = index into spike_i where timestep t begins
        spike_i[k]     = true_neuronID of the kth event
        events at t    = spike_i[spike_ptr[t] : spike_ptr[t+1]]

    A stored event means True; there are no values to store. Reconstruct dense with sparse2dense().
    """

    def __init__(self, hist_name: str, n_total: int, n_input: int, dt: float,
                 cfg: HistoryCfg = None, runmeta: RunMeta = None, derived_cfg=None):
        self.hist_name: str = hist_name
        self.derived_cfg = derived_cfg
        self.n_total: int = n_total
        self.n_input: int = n_input
        self.dt: float = dt

        ##MAJOR-EDIT-1: cfg=policy + statement
        self.cfg: HistoryCfg = HistoryCfg() if cfg is None else cfg
        self.runmeta: Optional[RunMeta] = runmeta

        ##Temporal (timelines based)
        self.t_start: Optional[int] = None
        self.n_steps: int = 0

        ##Per-timestep event counts -> spike_ptr = [0] + cumsum(counts). 
        # uint32 array one entry per STEO (timeline)
        self._counts = np.zeros(4096, dtype=np.uint32)

        ## cfg.record_counts. Protocols mark & read diff ("since" last mark) -> response per sample
        self.Counts = np.zeros(n_total, dtype=np.int64) #spike count per true_neuronID ((ALL neurons,) 

        ##MAJOR-EDIT-2: chunks/segmnebts. Events accumulate into one buffer and flush at cfg.chunk_events
        self._event_buf = np.empty(max(int(self.cfg.chunk_events), 1), dtype=np.uint32)
        self._buf_used = 0
        self._event_chunks: list[np.ndarray] = []
        self._n_events = 0

        ##Segment timelines (cfg controlleD).
        self.phase_timeline = _Timeline("phase")
        self.epoch_timeline = _Timeline("epoch")
        self.datarun_timeline = _Timeline("datarun")

        #NOTE: 1 timeline per fit (fold)
        self.fold_timeline = _Timeline("fold")   #DATA-PARTITION boundaries; -1 = the final refit
        self.mode_timeline = _Timeline("mode") # 0=train, 1=eval (MODE_KEY). marked by Run.set_mode (per switch).

    ##################################################################################################
    ################################################################################################
            ##### RUNTIME RECORDING #####
    #############################################################################################33
    ################################################################################################
    def _grow_counts(self) -> None:
        if self.n_steps < self._counts.size:
            return
        bigger = np.zeros(self._counts.size * 2, dtype=np.uint32)
        bigger[:self._counts.size] = self._counts
        self._counts = bigger

    def _flush(self) -> None:
        if self._buf_used:
            self._event_chunks.append(self._event_buf[:self._buf_used].copy())
            self._buf_used = 0

    def record_step(self, step: int, spike_i: np.ndarray) -> None:
        """
        Record ONE timestep. MUST be called every recorded step, including SILENT ones (to keep timelines stored == simtime)

        Args:
            step:    absolute runstep (ctx.sim.current_runstep)
            spike_i: true_neuronIDs that fired this step. (incl empty)
        """
        if self.t_start is None:
            self.t_start = int(step)

        self._grow_counts()
        n = int(spike_i.size)
        self._counts[self.n_steps] = n
        self.n_steps += 1
        self._n_events += n

        if n:
            ##Split across the flush boundary
            src = np.asarray(spike_i, dtype=np.uint32)
            while src.size:
                room = self._event_buf.size - self._buf_used
                take = min(room, src.size)
                self._event_buf[self._buf_used:self._buf_used + take] = src[:take]
                self._buf_used += take
                src = src[take:]
                if self._buf_used == self._event_buf.size:
                    self._flush()

    def mark_labels(self, step: int, *, phase=None, epoch=None, datarun=None) -> None:
        """Record the segment timelines for this step. Each marks only when its value CHANGES."""
        if self.cfg.record_phase and phase is not None:
            self.phase_timeline.mark(step, _capitalize_strip_name(phase))
        if self.cfg.record_epoch and epoch is not None:
            self.epoch_timeline.mark(step, int(epoch))
        if self.cfg.record_datarun and datarun is not None:
            self.datarun_timeline.mark(step, int(datarun))

    ## bookmarks on Counts
    def mark(self) -> np.ndarray:
        """A BOOKMARK = Counts (current cumulative per-neuron-spike-counts). 
        
        then: since(mark) -> Counts (current) - mark (previous) = spike count since last mark."""
        return self.Counts.copy()

    def since(self, mark: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Spikes per true_neuronID (int64, (n_total,)) recorded since `mark` (a mark()). out= writes into a caller buffer."""
        return np.subtract(self.Counts, mark, out=out)


    ##################################################
            ##### arrays #####
    #############################################33

    def spike_i(self) -> np.ndarray:
        """All event neuron ids, concatenated. (n_events,) uint32."""
        self._flush()
        if not self._event_chunks:
            return np.zeros(0, dtype=np.uint32)
        if len(self._event_chunks) > 1:
            ##Collapse to one block so repeated reads do not re-concatenate.
            self._event_chunks = [np.concatenate(self._event_chunks)]
        return self._event_chunks[0]

    def spike_ptr(self) -> np.ndarray:
        """(n_steps + 1,) uint64. events at t = spike_i()[ptr[t]:ptr[t+1]]."""
        ptr = np.zeros(self.n_steps + 1, dtype=np.uint64)
        if self.n_steps:
            np.cumsum(self._counts[:self.n_steps], dtype=np.uint64, out=ptr[1:])
        return ptr

    def spikes_at(self, step: int) -> np.ndarray:
        """true_neuronIDs that fired at absolute runstep 'step'"""
        if self.t_start is None:
            return np.zeros(0, dtype=np.uint32)
        t = int(step) - self.t_start
        if not (0 <= t < self.n_steps):
            return np.zeros(0, dtype=np.uint32)
        ptr = self.spike_ptr()
        return self.spike_i()[ptr[t]:ptr[t + 1]]

    def arrays(self) -> dict[str, np.ndarray]:
        """arrays for .npz."""
        arrays = {"spike_ptr": self.spike_ptr(), "spike_i": self.spike_i()}
        if self.cfg.record_counts:
            arrays["counts"] = self.Counts.copy()    
        for timeline in (self.phase_timeline, self.epoch_timeline, self.datarun_timeline,
                         self.fold_timeline, self.mode_timeline):    
            if timeline.starts and timeline.name != "phase":
                arrays[f"{timeline.name}_starts"] = np.asarray(timeline.starts, dtype=np.uint64)
                arrays[f"{timeline.name}_values"] = np.asarray(timeline.values, dtype=np.int64)
        if self.phase_timeline.starts:
            arrays["phase_starts"] = np.asarray(self.phase_timeline.starts, dtype=np.uint64)
            arrays["phase_values"] = np.asarray(
                [self.phase_key()[v] for v in self.phase_timeline.values], dtype=np.uint16)
        return arrays

    def phase_key(self) -> dict[str, int]:
        """name -> int, derived from the phases this history actually saw (declaration order)."""
        seen = []
        for value in self.phase_timeline.values:
            if value not in seen:
                seen.append(value)
        return {name: idx for idx, name in enumerate(seen)}

    ##################################################################################################
             ############### Reconstruction ############################
    #############################################################################################33
    def events(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Expand to COO (spike_t, spike_i) for per-event timestamps.
        spike_t is ABSOLUTE (t_start included). Derived.
        """
        counts = self._counts[:self.n_steps]
        base = 0 if self.t_start is None else self.t_start
        spike_t = np.repeat(np.arange(base, base + self.n_steps, dtype=np.uint64), counts)
        return spike_t, self.spike_i()

    def sparse2dense(self, T: int | None = None) -> np.ndarray:
        """
        Dense (T, n_total) bool raster. T defaults to the RECORDED .
        """
        T = self.n_steps if T is None else int(T)
        raster = np.zeros((T, self.n_total), dtype=bool)
        spike_t, spike_i = self.events()
        if spike_t.size:
            base = 0 if self.t_start is None else self.t_start
            local = (spike_t - base).astype(np.int64)
            keep = local < T
            raster[local[keep], spike_i[keep]] = True
        return raster

    def phase_at(self, step: int):
        return self.phase_timeline.at(step)

    def epoch_at(self, step: int):
        return self.epoch_timeline.at(step)

    def datarun_at(self, step: int):
        return self.datarun_timeline.at(step)

    def fold_at(self, step: int):
        return self.fold_timeline.at(step)

    def mode_at(self, step: int):
        """current mode (0= train, 1= eval: MODE_KEY) at 'step' (Run marks 0 at its first step, set_mode every switch)."""
        return self.mode_timeline.at(step)

################################################ METADATA

    def metadata(self) -> dict:
        """JSON/YAML-safe. Heavy arrays excluded.
        ## HistoryCfg (record_spikes, include_input, ...) = WHICH recordings in this file
        ## REST= structural facts needed to read them (n_total, n_input, dt, ids, timelines)."""
        return {
            "n_total": int(self.n_total),
            "n_input": int(self.n_input),
            "dt": float(self.dt),
            "t_start": None if self.t_start is None else int(self.t_start),
            "n_steps": int(self.n_steps),
            "n_events": int(self._n_events),
            "representation": "timestep_pointer (spike_ptr[t]:spike_ptr[t+1] -> spike_i)",
            "cfg": None if self.cfg is None else asdict(self.cfg),
            "phase_key": self.phase_key(),
            "phase_segments": self.phase_timeline.segments(),
            "epoch_segments": self.epoch_timeline.segments(),
            "datarun_segments": self.datarun_timeline.segments(),
            "fold_segments": self.fold_timeline.segments(),
            "mode_segments": self.mode_timeline.segments(),     ##0=train, 1=eval
            "runmeta": None if self.runmeta is None else asdict(self.runmeta),
            "derived_cfg": None if self.derived_cfg is None else _inf_safe(asdict(self.derived_cfg)),   ##the cfg ACTUALLY used
        }

    def save(self, filename: str | Path | None = None, *,
             JSON: bool = True, YAML: bool = False,
             sort_keys: bool = False, export_dir: Path = EXPORT_DIR) -> None:
        """metadata -> .json/.yaml, arrays -> .npz. Same split as Network saving."""
        if filename is None:
            now = datetime.now().strftime("%d-%m_%H%M%S")
            filename = f"History_{now}_{self.hist_name}"
        file = _filepath(filename, default_dir=export_dir)
        _write_json_yaml(data=self.metadata(), file=file, JSON=JSON, YAML=YAML,
                         sort_keys=sort_keys, log_label="RunHistory metadata")
        _write_history_npz(file, self.arrays())

    @classmethod
    def load(cls, file: str | Path) -> "RunHistory":
        """The inverse of save(): <file>.json (metadata) + <file>.npz (arrays) -> a RunHistory for analysis.py. For reading only."""
        file = Path(file)
        meta = _read_json_yaml(file.with_suffix(".json"))
        arrays = _read_history_npz(file)
        hist = cls(file.stem, int(meta["n_total"]), int(meta["n_input"]), float(meta["dt"]), cfg=None if meta["cfg"] is None else HistoryCfg(**meta["cfg"]))
        hist.t_start, hist.n_steps, hist._n_events = meta["t_start"], int(meta["n_steps"]), int(meta["n_events"])
        hist._counts = np.diff(arrays["spike_ptr"]).astype(np.uint32)
        hist._event_chunks = [arrays["spike_i"]]
        if "counts" in arrays:
            hist.Counts = arrays["counts"]
        for timeline in (hist.phase_timeline, hist.epoch_timeline, hist.datarun_timeline, hist.fold_timeline, hist.mode_timeline):
            for seg in meta.get(f"{timeline.name}_segments", []):
                timeline.mark(seg["start_step"], seg["value"])
        return hist
