#operations - the COMPOSABLE BLOCKS between a Protocol and the Run, with the logging, progress reporting, timing and snapshots written INTO them.
#
#   L0  Run (run.py)            the ENGINE: step() / run_sample(until=) / continue_sample(until=) / finish_sample() / set_mode("train"|"eval") / resets.
#   L1  Operations (this file)  the blocks a protocol is written FROM (self.ops): one sample (binned_counts, present), one resumable pass (responses,
#                               for_samples, repeat), the fit levels (run_folds -> run_fold -> run_fit -> run_epoch), partition(), test(), the common tail.
#   L2  Protocol subclasses     WHAT the experiment is (HoldoutValidation, DCProtocol, ELSM / HLSP / NALSM / RLSM), picked by RunCfg.protocol.
#   boundary                    PhasicNet.train() around protocol.execute(): status.json (RunStatus) + run.jsonl / results.jsonl.
#
#RESUME: progress = {"stage": s, "i": done} for a pass, {"fold": k, "epoch": e, "i": done} for a fit; saved in every snapshot (snap_s<stage>_i<position>.npz).
# NOT restored: the fit's best / patience bookkeeping and the phase state (a run resumes in DEFAULT).
#RULE: nothing here runs INSIDE a step; reporting, timing and snapshots happen at sample boundaries.
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, List
import logging
import traceback
import weakref

import numpy as np
from tqdm.auto import tqdm

import capture
from utils import _write_json_yaml, _write_responses_npz, _inf_safe, _is_better
from checkpoint import snapshots
from runlog import fields          ##extra= of a results line (numpy -> Python numbers)

logger = logging.getLogger(__name__)
results_log = logging.getLogger("results")      ##-> results.jsonl. Not `results`: finish() has a parameter of that name


   ######################################################################################
##########################   Operations (L1)   ############################################
   ######################################################################################

class Operations:
    """
    The blocks between a Protocol and its Run. Built by Protocol.__init__ (self.ops), or directly: Operations(run, out_dir=...).

    Args:
        run = the assembled Run.
        state = the Protocol: its state_dict goes into every snapshot (proto__*), finish() reads its cfg / report(). None = run arrays + progress only.
        out_dir = the run FOLDER; None = nothing is written.
        report_every = one progress line per this many samples.
        reset_between_samples = run.reset_transient() before every sample in present() / run_epoch().
    STATE: progress (the resumable position), seconds (wall time per named pass), epoch_log (the last fit's per-epoch score + health probes).
    """

    def __init__(self, run, *, state=None, out_dir: str | Path | None = None, report_every: int = 500, reset_between_samples: bool = False):
        self.run = run
        self._state = None if state is None else weakref.ref(state)    ##WEAK: a strong reference back would be a cycle keeping the Run alive
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.report_every = max(1, int(report_every or 1))      ##0 / None would make `done % every` fail
        self.reset_between_samples = bool(reset_between_samples)
        self.progress: dict = {}
        self.seconds: dict = {}
        self.epoch_log: List[dict] = []     ##cleared by run_fit: ONE fit's epochs
        self._t0 = perf_counter()

    @property
    def state(self):
        """The Protocol (or None): whose state_dict the snapshots carry, whose cfg / report() finish() reads."""
        return None if self._state is None else self._state()

    # -------------------------------------------------------------------------
    # one sample
    # -------------------------------------------------------------------------

    def binned_counts(self, frames, target: Any, bounds, *, group: str | tuple[str, ...] | slice | np.ndarray | None = None) -> np.ndarray:
        """
        ONE presentation of a sample -> its spike counts per TIME WINDOW: (n_bins, n_selected) float64, true_neuronID order within the selection.
            bounds = window boundaries IN STEPS, ascending: bin k = steps [bounds[k-1], bounds[k]); steps after the last bound run UNCOUNTED.
            group = None (every neuron) | a group name | a tuple of names (bands concatenated in that order; HLSP's liquid is ("inh", "exc"))
                | a slice / index array of the count vector.
        One begin_sample; reset_between_samples once at the start, never between bins. present() is the one-bin case.
        """
        run, history = self.run, self.run.history
        T = int(np.shape(frames)[0])
        bounds = [int(b) for b in bounds]
        if not bounds or bounds[0] < 0 or bounds[-1] > T or any(b < a for a, b in zip(bounds, bounds[1:])):
            raise ValueError(f"binned_counts: bounds must be non-decreasing step boundaries in [0, T={T}], got {bounds}")
        sel = self._band(group)
        if self.reset_between_samples:
            run.reset_transient()
        rows, mark = [], history.mark()
        for k, b in enumerate(bounds):
            if k == 0:
                run.run_sample(frames, target, until=b)
            else:
                run.continue_sample(b)
            rows.append(capture.spike_counts(run.ctx, history, since_mark=mark)[sel])
            mark = history.mark()
        run.finish_sample()         ##the remainder past the last bound, uncounted
        return np.vstack(rows)

    def _band(self, group):
        """group -> what indexes a count vector: None = everything; a name = its band; a tuple = the bands concatenated; a slice as is."""
        net = self.run.ctx.net
        if group is None:
            return slice(None)
        if isinstance(group, (slice, np.ndarray)):
            return group
        if isinstance(group, str):
            return net.neuronslices[group]
        return np.r_[tuple(net.neuronslices[g] for g in group)]

    def present(self, frames, target: Any = None, *, group: str | None = None, until: int | None = None) -> np.ndarray:
        """
        ONE sample -> its RESPONSE: spike count per neuron (float64, true_neuronID order; group= one group's band) over the sample, or over its
        first `until` steps (the rest runs uncounted). The one-bin case of binned_counts.
        """
        T = int(np.shape(frames)[0])
        return self.binned_counts(frames, target, [T if until is None else min(int(until), T)], group=group)[0]

    # -------------------------------------------------------------------------
    # one pass (resumable, logged, checkpointed)
    # -------------------------------------------------------------------------

    def responses(self, data, respond: Callable, Resp: np.ndarray, Y: np.ndarray, *, stage: int, name: str, n: int | None = None,
                  extra: Callable[[int, int], str] | None = None, reset: bool = False) -> bool:
        """
        ONE PASS over `data`: Resp[i] = respond(frames, target, key), Y[i] = target for i in [resumed position, n).
            respond(frames, target, key) -> the response row. key = f"{name}:{i}", unique per sample and pass (for a per-presentation rng).
            n = how many samples (None = len(data)); Resp / Y need at least n rows.
            extra(lo, hi) -> str: tail of the progress line over samples [lo, hi). Default: spikes/sample of Resp.
            reset = run.reset_transient() before every sample (False when respond resets itself).
        Returns True if the pass ran, False if it was skipped (completed before the snapshot).
        """
        n = len(data) if n is None else min(int(n), len(data))
        run = self.run

        def body(i):
            frames, target = data[i]
            if reset:
                run.reset_transient()
            Resp[i] = respond(frames, target, f"{name}:{i}")
            Y[i] = int(target)

        if extra is None:
            def extra(lo, hi):
                return f"spikes/sample {float(Resp[lo:hi].sum(1).mean()):.1f}"
        return self._loop(n, body, stage=stage, name=name, extra=extra)

    def for_samples(self, data, handle: Callable[[int, Any, Any], Any], *, stage: int, name: str, reset: bool = False,
                    extra: Callable[[int, int], str] | None = None) -> bool:
        """The same resumable, logged, checkpointed loop over a dataset WITHOUT response arrays: handle(i, frames, target) per sample,
        run.reset_transient() first when `reset`. For training / initialization passes. Returns True if the pass ran, False if it was skipped."""
        run = self.run

        def body(i):
            frames, target = data[i]
            if reset:
                run.reset_transient()
            handle(i, frames, target)
        return self._loop(len(data), body, stage=stage, name=name, extra=extra)

    def repeat(self, n: int, body: Callable[[int], Any], *, stage: int, name: str, extra: Callable[[int, int], str] | None = None) -> bool:
        """The same loop over an INDEX only: body(i) for i in [resumed position, n), for a body that picks its own data (DC's test passes).
        Returns True if the pass ran, False if it was skipped."""
        return self._loop(int(n), body, stage=stage, name=name, extra=extra)

    def _loop(self, n: int, body: Callable[[int], Any], *, stage: int, name: str, extra) -> bool:
        """The one loop: resume at progress, body per sample, a progress line every report_every samples, run.checkpoints.on_sample_end after
        every sample, seconds[name] accumulated; progress then moves to the next stage."""
        run, every = self.run, self.report_every
        cur = int(self.progress.get("stage", 0))
        if cur > stage:
            logger.info("  %s: stage %d already complete in the resumed snapshot, skipped", name, stage)
            return False
        i0 = min(int(self.progress.get("i", 0)), n) if cur == stage else 0
        t0 = perf_counter()
        for i in range(i0, n):
            body(i)
            done = i + 1
            if done % every == 0 or done == n:
                el = perf_counter() - t0
                rate = (done - i0) / el if el > 0 else float("inf")
                tail = "" if extra is None else "  " + str(extra(max(i0, done - every), done))
                logger.info("  %-6s %6d/%d  %6.1f min  %.2f samples/s  ETA %5.1f min%s", name, done, n, el / 60, rate,
                            (n - done) / rate / 60 if rate else float("nan"), tail)
            self.progress = {"stage": int(stage), "i": done}
            run.checkpoints.on_sample_end(stage=stage, position=done, progress=self.progress, protocol=self.state, force=done == n)
        self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - t0
        self.progress = {"stage": int(stage) + 1, "i": 0}
        return True

    def starting(self, stage: int) -> bool:
        """True when `stage` has NOT begun: nothing of it is in the resumed position. A protocol's stage-entry act (a fresh network, a counter reset,
        a save after the previous stage) runs under this, so a run resumed INSIDE the stage does not repeat it."""
        cur, i = int(self.progress.get("stage", 0)), int(self.progress.get("i", 0))
        return cur < stage or (cur == stage and i == 0)

    # -------------------------------------------------------------------------
    # the fit levels: _run -> _fold/_fit -> _epoch -> _steps
    # -------------------------------------------------------------------------

    def run_epoch(self, data, *, shuffle: bool = False, rng: np.random.Generator | None = None,
                  max_samples: int | None = None, collect_responses: bool = False, fold: int | None = None, epoch: int = 0) -> dict:
        """
        Run every sample in `data` once.

        Args:
            data: ModelInput (or anything with len() and [i] -> (frames, target)).
            shuffle: reorder samples for this epoch with `rng` (reproducible: the ORDER is a run parameter).
            collect_responses: accumulate a per-sample spike-count vector for scoring (a VAL pass needs it, a TRAIN pass does not).
            fold, epoch: given by run_fit for a TRAIN pass -> resumable (progress {"fold", "epoch", "i"}). fold None = no snapshots, no resume.
        The caller sets run.set_mode("train" | "eval") BEFORE the pass. reset_between_samples is applied per sample.
        summary["spikes"] = every recorded spike of the pass; summary["recurrent_spikes"] = the non-input spikes (for the health probe).
        """
        run = self.run
        ctx, history = run.ctx, run.history
        sim = ctx.sim

        ##DATA PREP FOR THIS EPOCH happens HERE, _epoch function IS the epoch. Datahandler owns shuffling of data
        if shuffle:
            if rng is None:
                raise ValueError("shuffle=True requires an rng, so the order stays reproducible.")
            data = run.datahandler.shuffled(data, rng)

        n = len(data) if max_samples is None else min(int(max_samples), len(data))
        order = np.arange(n)
        ##the resumed position of THIS epoch (0 unless resumed)
        p = self.progress
        i0 = min(int(p.get("i", 0)), n) if fold is not None and (p.get("fold"), p.get("epoch")) == (fold, epoch) else 0

        pass_mark = history.mark()
        reset = self.reset_between_samples
        steps = 0
        targets = []
        responses: List[np.ndarray] = []

        bar = tqdm(total=n, initial=i0, desc=f"  samples (epoch {sim.current_epoch})", leave=False)
        for position, index in enumerate(order[i0:], start=i0):
            frames, target = data[int(index)]

            sim.current_datarun = int(index)

            if reset:
                run.reset_transient()
            if collect_responses:
                before = history.mark()
            steps += run.run_sample(frames, target)
            targets.append(target)

            if collect_responses:
                ##Per-sample response = spike COUNT per neuron over the sample
                responses.append(capture.spike_counts(ctx, history, since_mark=before))
            bar.update(1)
            if fold is not None and position + 1 < n:       ##resumable position + scheduled snapshot; the epoch's last sample is run_fit's, after the val pass
                self.progress = {"fold": int(fold), "epoch": int(epoch), "i": position + 1}
                run.checkpoints.on_sample_end(stage=fold, position=epoch * n + position + 1, progress=self.progress, protocol=self.state)
        bar.close()

        counts = history.since(pass_mark)
        recurrent = int(counts[ctx.net.n_input:].sum())
        summary = {"epoch": sim.current_epoch, "samples": n, "steps": steps,
                   "spikes": int(counts.sum()) if history.cfg.include_input else recurrent, "recurrent_spikes": recurrent, "targets": targets}
        if collect_responses and responses:
            summary["responses"] = capture.spike_count_responses(responses)
        logger.info("epoch %d: samples=%d steps=%d spikes=%d",
                    sim.current_epoch, n, steps, summary["spikes"])
        return summary

    def run_fit(self, train, val=None, *, epochs: int = 1, patience: int | None = None,
                shuffle: bool = True, max_samples: int | None = None, fold: int | None = None) -> dict:
        """
        ONE fit: epochs over `train`, scoring `val` at the end of each.
        patience = epochs without improvement before stopping (ProtocolCfg.patience); None = no patience stop.
        fold     = labels the per-epoch results lines and keys the resume position and the shuffle streams; None = no snapshots, no resume.

        TRAIN is reshuffled every epoch from the keyed stream rng("fold", (fold, e)); VAL never is.

        Returns the fit summary
            "final" (default): validation is OBSERVATIONAL. The fit's model is its final state; its score is the LAST epoch's val score.
            "best":            the best-val epoch is restored and scores the fit ("better" = utils._is_better, as for patience).
        The fit ends in mode "train" (finalize() may restore W, a training-side write).
        """
        run = self.run
        ctx = run.ctx
        sim = ctx.sim
        best_score, best_epoch, last_score, collapse_epoch = None, None, None, None
        ##resumed = continues from a snapshot of ITS fold: no model reset; best / patience / epoch_log restart here
        p = self.progress
        resumed = fold is not None and p.get("fold") == fold and (int(p.get("epoch", 0)) or int(p.get("i", 0))) > 0   ##{"fold": k, "epoch": 0, "i": 0} = fold k's FRESH start
        e0 = int(p.get("epoch", 0)) if resumed else 0
        if not resumed:
            run.reset_model()
        stale_epochs = 0
        stop_reason = None
        self.epoch_log = []

        bar = tqdm(total=epochs, initial=e0, desc=" epochs")
        for e in range(e0, epochs):     ##e = the epoch within THIS fit (sim.current_epoch counts across fits)
            run.set_mode("train")
            summary = self.run_epoch(train, shuffle=shuffle, rng=run.ctx.rng.rng("fold", (fold, e)), max_samples=max_samples, fold=fold, epoch=e)

            # HEALTH PROBES
            ##NOTE The health probe asks about NETWORK activity, so it counts recurrent spikes only (NOOT input drive, ie data hz)
            health = run.scorer.health(ctx, steps=summary["steps"], spike_count=summary["recurrent_spikes"], dt=float(run.history.dt))
            health.update(run.scorer.weight_health(ctx, w_min=run.plasticity.w_min, w_max=run.plasticity.w_max))
            collapsed = run.scorer.is_collapsed(health)

            ##SEPERABILITY: uses labelled forward pass. END-of-epoch ONLY.
            score = None
            if val is not None:
                ##mode "eval": measurement (no training)
                run.set_mode("eval")
                val_summary = self.run_epoch(val, shuffle=False, collect_responses=True)
                if "responses" in val_summary:
                    score = run.scorer.separability(val_summary["responses"], np.asarray(val_summary["targets"]))

            if collapsed:
                run.checkpoints.on_collapse(epoch=sim.current_epoch, health=health)
            run.checkpoints.on_epoch_end(epoch=sim.current_epoch, score=score, health=health)

            last_score = score
            c = run.runcfg.checkpoint     ##the checkpoint policy's definition of "better"
            improved = score is not None and _is_better(score, best_score, greater_is_better=c.greater_is_better, min_improvement=c.min_improvement)
            if improved:
                best_score, best_epoch = score, sim.current_epoch
                stale_epochs = 0
            elif score is not None:
                stale_epochs += 1

            self.epoch_log.append({"epoch": sim.current_epoch, "score": score,
                                   "collapsed": collapsed, **health})
            results_log.info("epoch", extra=fields({"fold": fold, **self.epoch_log[-1]}))
            bar.set_postfix(score=None if score is None else round(score, 2),
                            rate=round(health["rate_hz"], 1))
            bar.update(1)

            sim.current_epoch += 1

            ##On collapse
            if collapsed:
                stop_reason = "collapse"
                collapse_epoch = sim.current_epoch - 1
                logger.warning("fit stopped early at epoch %d: network collapsed (%s).",
                               sim.current_epoch - 1,
                               {k: round(v, 4) for k, v in health.items() if k != "steps"})
                break
            if patience is not None and stale_epochs >= patience:
                stop_reason = "patience"
                logger.info("fit stopped early at epoch %d: %d epochs without improvement.",
                            sim.current_epoch - 1, stale_epochs)
                break
            ##the epoch's END is a snapshot position: after the val pass, as the next epoch's start
            if fold is not None:
                self.progress = {"fold": int(fold), "epoch": e + 1, "i": 0}
                run.checkpoints.on_sample_end(stage=fold, position=(e + 1) * summary["samples"], progress=self.progress, protocol=self.state)
        bar.close()

        run.set_mode("train")              ##checks the eval contract; the restore below is a training-side write
        run.checkpoints.finalize()          ##select="best": restores best W + every module's state_dict. "final": no-op
        if fold is not None:
            self.progress = {"fold": int(fold) + 1, "epoch": 0, "i": 0}     ##this fold is complete
        selected = best_score if run.runcfg.checkpoint.select == "best" else last_score
        return {"score": selected, "best_score": best_score, "best_epoch": best_epoch, "last_score": last_score,
                "stopped": stop_reason, "collapsed": stop_reason == "collapse", "collapse_epoch": collapse_epoch,
                "epochs_run": len(self.epoch_log), "epochs": list(self.epoch_log),
                "checkpoints": run.checkpoints.metadata()}

    def run_fold(self, train, val=None, *, fold: int, epochs: int = 1, patience: int | None = None, shuffle: bool = True) -> dict:
        """
        ONE fold: ONE fresh fit on ONE data partition (train, val). fold = k for a cross-validation fold, 0 for the holdout fit, -1 for a refit.
        # NOTE a FOLD is a partition of the data. a FIT is training one fresh model on it (run_fit).
        """
        run = self.run
        run.checkpoints.new_fit()       ##cleared, not re-instantiated (bound to the Run)
        ##one RunHistory spans every fold: mark the fold boundary in the record
        if run.history is not None:
            run.history.fold_timeline.mark(run.ctx.sim.current_runstep, fold)     ##-1 = the final refit, not a fold
        report = self.run_fit(train, val, epochs=epochs, patience=patience, shuffle=shuffle, fold=fold)   ##run_fit() resets the model itself
        report["fold"] = fold
        results_log.info("fold", extra=fields({k: report[k] for k in ("fold", "score", "best_score", "best_epoch", "last_score", "epochs_run", "stopped")}))
        return report

    def run_folds(self, partitions, *, epochs: int = 1, patience: int | None = None, shuffle: bool = True) -> dict:
        """
        The FULL Run: every fold, then aggregate. `partitions` = the (train, val) pairs of ONE data partition (partition()), drawn once per run.
        Epoch order of fold k, epoch e: rng("fold", (k, e)), reproducible in isolation. The noise streams are NOT re-keyed per fold (one stream
        per run, as Brian2's seed()): one fold of a NOISY run reproduces only together with the folds before it.
        On resume the folds below progress["fold"] are skipped.
        Returns:
            score = MEAN over folds. NOT the max. std is returned alongside; fold_scores, folds (each fold's run_fit summary)
        """
        epochs = int(epochs or 1)
        fold_scores, fold_reports = [], []
        k0 = int(self.progress.get("fold", 0))

        bar = tqdm(total=len(partitions), initial=min(k0, len(partitions)), desc="folds")
        for k, (train, val) in enumerate(partitions):
            if k < k0:
                logger.info("  fold %d already complete in the resumed snapshot, skipped", k)
                continue
            report = self.run_fold(train, val, fold=k, epochs=epochs, patience=patience, shuffle=shuffle)
            fold_reports.append(report)
            if report["score"] is not None:
                fold_scores.append(report["score"])
            bar.update(1)
        bar.close()

        scores = np.asarray(fold_scores, dtype=np.float64)
        out = {"score": float(scores.mean()) if scores.size else None,
               "score_std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
               "fold_scores": fold_scores, "folds": fold_reports}
        results_log.info("folds", extra=fields({"score": out["score"], "score_std": out["score_std"], "fold_scores": fold_scores, "n_folds": len(partitions)}))
        return out

    def partition(self, *, val_split: float | None = None) -> list:
        """
        The data partition of this run: DataHandler.fold_split on the TRAIN set (val_split -> ONE holdout; None -> one train-only fold).
        """
        run = self.run
        ##NOTE: rng keyed on the DATASET: same data + same seed -> same folds, SAME partitions.
        return run.datahandler.fold_split(run.train_set, val_split=val_split,
                                          rng=run.ctx.rng.rng("data", f"folds:{run.runmeta.dataset_id}"), stratify=run.runcfg.data.stratify)

    def test(self, data=None) -> dict:
        """
        ONE forward pass over the HELD-OUT test set with the model currently loaded --
        select="best" => its restored best epoch.
        select="final" => whatever it ended with
        learn=False
        Mode "eval" (the Run stays in eval afterwards; check_persistent_unchanged at the end is the contract).
        content="weights" -> run.reset_transient(): W and the persistent arrays are kept, only transient state starts from rest.
        """
        run = self.run
        data = run.test_set if data is None else data
        if data is None:
            raise ValueError("No test set: pass data= or give train() a dataset so the split can be made.")
        run.set_mode("eval")
        if run.runcfg.checkpoint.content == "weights":
            run.reset_transient()
        summary = self.run_epoch(data, shuffle=False, collect_responses=True)
        score = None
        if "responses" in summary:
            score = run.scorer.separability(summary["responses"], np.asarray(summary["targets"]))
        run.check_persistent_unchanged()
        logger.info("test: %d samples, %d spikes, score=%s", summary["samples"], summary["spikes"], score)
        results_log.info("test", extra=fields({"score": score, "samples": summary["samples"], "spikes": summary["spikes"], "n_test": len(data)}))
        return {"score": score, "samples": summary["samples"], "steps": summary["steps"], "spikes": summary["spikes"], "n_test": len(data)}

    # -------------------------------------------------------------------------
    # header / model / the common tail
    # -------------------------------------------------------------------------

    def header(self, **extra) -> None:
        """The first log line of every protocol: the run's identity (+ extra key=value pairs the protocol wants on it)."""
        m = self.run.runmeta
        tail = "".join(f"  {k}={v}" for k, v in extra.items())
        logger.info("run_id=%s network_id=%s cfg_id=%s%s", m.run_id, m.network_id, m.cfg_id, tail)

    def save_model(self, label: str = "trained model") -> Path:
        """The run's TrainedModel into the run folder (runs/ when there is none), logged. Returns the .npz path."""
        model = self.run.save_trained_model(save_dir=self.out_dir)
        logger.info("%s saved: %s", label, model.name)
        return model

    def finish(self, results: dict, *, files: dict | None = None, lines=()) -> dict:
        """
        The common tail of every protocol's execute(): identity keys + `results` + "seconds" -> results.json, `files` -> responses.npz (both only
        with an out_dir); the RUN COMPLETE block (`lines` = the protocol's own summary lines). Returns {**res, **state.report()} = run.report.
        """
        m = self.run.runmeta
        cfg = getattr(self.state, "cfg", None)
        res = {"run_id": m.run_id, "network_id": m.network_id, "cfg_id": m.cfg_id, "protocol_cfg": None if cfg is None else asdict(cfg),
               **results, "seconds": {**{k: float(v) for k, v in self.seconds.items()}, "total": perf_counter() - self._t0}}
        if self.out_dir is not None:
            _write_json_yaml(self.out_dir / "results", _inf_safe(res), JSON=True, YAML=False, log_label="Results")
            if files:
                _write_responses_npz(self.out_dir / "responses.npz", files)
        logger.info("=" * 72); logger.info("RUN COMPLETE")
        for line in lines:
            logger.info(line)
        logger.info("  seconds       : %s", {k: round(v, 1) for k, v in res["seconds"].items()})
        logger.info("  outputs       : %s", self.out_dir)
        logger.info("=" * 72)
        ##the protocol's NUMBERS as one results line (text and nested dicts stay in results.json)
        nums = {k: v for k, v in res.items() if isinstance(v, (int, float, np.number)) or
                (isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float, np.number)) for x in v))}
        results_log.info("results", extra=fields({**nums, "seconds_total": res["seconds"]["total"]}))
        report = getattr(self.state, "report", None)
        return {**res, **(report() if callable(report) else {})}


   ######################################################################################
##########################   The experiment boundary   ####################################
   ######################################################################################

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")      ##24-hour (local time; the run.jsonl timestamps are UTC)


class RunStatus:
    """
    status.json of a run folder: "running" from the start, then "completed" | "failed" | "interrupted" (Ctrl-C), with times, the progress at the
    end, the newest snap_*.npz (where a resume starts) and, on failure, the error and traceback. Written by PhasicNet.train() only; also run.status.
    """

    def __init__(self, out_dir: str | Path | None, run, protocol):
        self.out_dir = None if out_dir is None else Path(out_dir)
        self._protocol = protocol
        self._t0 = perf_counter()
        self.record = {"status": "running", "run_id": run.runmeta.run_id, "run_name": run.runmeta.run_name, "protocol": type(protocol).__name__,
                       "started": _now(), "ended": None, "seconds": None, "progress": None, "last_snapshot": None, "error": None}
        self.write()

    def last_snapshot(self) -> str | None:
        if self.out_dir is None:
            return None
        snaps = snapshots(self.out_dir)      ##ordered by the (stage, position) parsed from the names
        return snaps[-1].name if snaps else None

    def end(self, status: str, error: BaseException | None = None) -> dict:
        r = self.record
        r.update(status=status, ended=_now(), seconds=perf_counter() - self._t0, progress=dict(self._protocol.progress), last_snapshot=self.last_snapshot(),
                 error=None if error is None else {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        self.write()
        if status == "completed":
            logger.info("RUN %s in %.1f min", status.upper(), r["seconds"] / 60)
        else:
            logger.error("RUN %s after %.1f min at progress %s (last snapshot: %s)\n%s", status.upper(), r["seconds"] / 60, r["progress"], r["last_snapshot"],
                         "" if error is None else r["error"]["traceback"])
        results_log.info("status", extra=fields({"status": status, "seconds": r["seconds"], "progress": r["progress"], "last_snapshot": r["last_snapshot"],
                                             "error": None if error is None else r["error"]["type"]}))      ##the last results line
        return r

    def write(self) -> None:
        if self.out_dir is not None:
            _write_json_yaml(self.out_dir / "status", _inf_safe(self.record), JSON=True, YAML=False, log_label="Status")
