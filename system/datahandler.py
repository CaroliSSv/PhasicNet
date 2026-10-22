#datahandler - owns ALL data handling: the one-time test/train split, per-fold and per-epoch
# subsets, and the per-timestep handover into the runtime.

#WHO COMMANDS WHAT:
#   PhasicNet   ONCE, before any Run: split_test_train(). Same TEST set for every Run.
#   Operations.partition   fold_split()   -> the run's (train, val) partition, drawn ONCE (run_folds loops over it; val_split None = one train-only fold)
#   Operations.run_epoch   shuffled()     -> reshuffled train order for this epoch
#   Run.step               step()         -> one timestep of the bound sample
#   
# NOTE Every set handed around here is a ModelInput VIEW (ModelInput.subset: an index into the ORIGINAL source)).
#
#STRATIFIED BY DEFAULT (RunCfg.data.stratify). 

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence
import json
import logging
import time

import numpy as np

from paths import TRAIN_TEST_DIR
from states import RunContext
from modelinput import _dataset_hash   
from modelinput import ModelInput      

logger = logging.getLogger(__name__)




   ######################################################################################
##########################   Stratification helpers   #####################################
   ######################################################################################




def _stratified_holdout(labels: np.ndarray, fraction: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(keep, holdout) index arrays, class proportions preserved in both."""
    keep, holdout = [], []
    for cls in np.unique(labels):
        members = np.flatnonzero(labels == cls)
        rng.shuffle(members)
        n_hold = int(round(len(members) * fraction))
        ##At least one per class, else a rare class vanishes from the holdout
        if fraction > 0 and n_hold == 0 and len(members) > 1:
            n_hold = 1
        holdout.extend(members[:n_hold].tolist())
        keep.extend(members[n_hold:].tolist())
    return np.sort(np.asarray(keep, dtype=np.int64)), np.sort(np.asarray(holdout, dtype=np.int64))


def _stratified_pick(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:   ##NOTE resolved for one-class crash; used by DataHandler.pick
    """n indices with the class proportions of `labels`, each class's members drawn at random, returned in a SHUFFLED order (classes mixed).
    Quotas: n * class share, rounded down; the leftover places go to the classes with the largest remainders (ties: lower class first)."""
    if labels.size == 0:       
        return np.zeros(0, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    exact = n * counts / counts.sum()
    quota = np.floor(exact).astype(np.int64)
    quota[np.argsort(-(exact - quota), kind="stable")[:n - int(quota.sum())]] += 1
    picked = [rng.permutation(np.flatnonzero(labels == cls))[:q] for cls, q in zip(classes, quota)]
    return rng.permutation(np.concatenate(picked)).astype(np.int64)


   ######################################################################################
##########################   DataHandler   ################################################
   ######################################################################################

class DataHandler:
    """
    Dense data handling: one (T, N_in) sample -> one row of upd.step_input per step(), PLUS the
    split/shuffle operations the Run layers ask for.

    _frames = current sample. (format stays here -> return 1 timestep vector at a time on step())
    """

    def __init__(self):
        self._frames = None ##All 
        ###NOTE datacfg lives inside runcfg in Run for ease-of access in other parts of the system (runcfg keeps all technicals), and is sent to datahandler on need-to-know basis rather than copied into state in datahandler


   ######################################################################
##########################  TEST/TRAIN split ONCE & reuse  ###################

    def split_test_train(self, data: ModelInput, *, dataset_id: str, data_cfg, rng,
                         save_dir: Path | None = None) -> tuple[ModelInput, ModelInput]:   
        """
        Hold out a TEST set ONCE per raw dataset, and reuse it for every Run afterwards.
            -> written to datasets/train_test/, and reused verbatim thereafter.

            -> stores INDICES and spec --> relies on same original raw data + spec to recreate identical test-train split
            ->  stored split is reused ONLY under the policy it was made with (test_fraction / stratified). a changed DataCfg raises

        Returns:
            (train, test) ModelInput views. (train -> train/val sets, test -> final testing of PRE-TRAINED model generalization
        """

        stratify, fraction, make = data_cfg.stratify, data_cfg.test_fraction, data_cfg.make_test_split

        save_dir = TRAIN_TEST_DIR if save_dir is None else save_dir   
        save_dir.mkdir(parents=True, exist_ok=True)
        spec_file = save_dir / f"split_{dataset_id}.json"

        if spec_file.exists():
            spec = json.loads(spec_file.read_text())
            
            for key, now in (("test_fraction", float(fraction)), ("stratified", bool(stratify))):
                if spec.get(key) != now:
                    raise ValueError(f"Stored test/train split {spec_file} was made with {key}={spec.get(key)!r}, but DataCfg asks for {key}={now!r}. "
                                     f"Keep the stored setting, or delete that file to split dataset {dataset_id} again.")
            logger.info("Reusing EXISTING test/train split for dataset %s (%s).", dataset_id, spec_file.name)
            return (data.subset(spec["train_idx"], "train"), data.subset(spec["test_idx"], "test"))

        if not make:
            raise FileNotFoundError(
                f"No test/train split found for dataset {dataset_id!r} and DataCfg."     
                f"make_test_split=False. Either preprocess a test set first, or set it True.")

        labels = data.labels()      
        gen = rng.rng("data", f"test_split:{dataset_id}")     ##keyed on the dataset: same data -> same split, always
        if stratify:
            train_idx, test_idx = _stratified_holdout(labels, fraction, gen)
        else:
            order = gen.permutation(len(data))
            cut = int(round(len(data) * (1.0 - fraction)))
            train_idx, test_idx = np.sort(order[:cut]), np.sort(order[cut:])

        spec = {"dataset_id": dataset_id, "n_total": int(len(data)),
                "test_fraction": float(fraction), "stratified": bool(stratify),
                "rng_spec": rng.rng_spec,
                "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}
        spec_file.write_text(json.dumps(spec, indent=2))
        logger.info("Created test/train split for dataset %s: %d train / %d test (stratified=%s) -> %s",
                    dataset_id, train_idx.size, test_idx.size, stratify, spec_file.name)
        return (data.subset(train_idx, "train"), data.subset(test_idx, "test"))


     ######################################################  
#############   Per-FOLD and per-EPOCH datawrangling  #############

    # -------------------------------------------------------------------------

    def fold_split(self, train: ModelInput, *, val_split: float = None,
                   rng: np.random.Generator = None, stratify: bool = True) -> list[tuple[ModelInput, ModelInput]]:
        """
        Turn the TRAIN set into a list of (train, val) pairs. TEST is not involved and never is.

            val_split  -> ONE stratified holdout
            None       -> one fold, val=None (train-only run; no tier-2 score)

        Drawn ONCE per Run: every fold of a Run/run works with the same partition
        """
        rng = rng if rng is not None else np.random.default_rng(0)
        labels = train.labels()
        n = len(train)

        if val_split:
            if stratify:
                tr_idx, va_idx = _stratified_holdout(labels, float(val_split), rng)
            else:
                order = rng.permutation(n)
                cut = int(round(n * (1.0 - float(val_split))))
                tr_idx, va_idx = np.sort(order[:cut]), np.sort(order[cut:])
            logger.info("Holdout split: %d train / %d val (stratified=%s).", tr_idx.size, va_idx.size, stratify)
            return [(train.subset(tr_idx, "train"), train.subset(va_idx, "val"))]

        return [(train.subset(np.arange(n), "train"), None)]

    def shuffled(self, data: ModelInput, rng: np.random.Generator) -> ModelInput:
        """
        Reshuffled view for ONE epoch. (Called at the top of Operations.run_epoch; as Run._epoch.

        Returns a view  rather than shuffling in place: the underlying order is data, not state,
        and a view keeps the original index available for the history's datarun labels.
        """
        order = np.arange(len(data))
        rng.shuffle(order)
        return data.subset(order, f"{data.name or 'data'}_shuffled")

    def pick(self, data: ModelInput, n: int | None, *, data_cfg, rng: np.random.Generator) -> ModelInput:   ##to stratify

        """
        WHICH samples of the train or test set a run uses, and in what order (DataCfg.n_train / n_test / shuffle).

            n given    -> n samples, class proportions kept (data_cfg.stratify; else a plain random draw), in a shuffled order
            n None     -> every sample: shuffled when data_cfg.shuffle, else returned unchanged (D&C keeps the source's order)

        """
        if n is None:
            return self.shuffled(data, rng) if data_cfg.shuffle else data
        n = min(int(n), len(data))
        idx = _stratified_pick(data.labels(), n, rng) if data_cfg.stratify else rng.permutation(len(data))[:n]
        logger.info("Picked %d of %d samples from %r (stratified=%s, shuffled order).", n, len(data), data.name or "data", data_cfg.stratify)
        return data.subset(idx, f"{data.name or 'data'}_n{n}")

    # -------------------------------------------------------------------------
    # Cost estimate
    # -------------------------------------------------------------------------

    def estimate_cost(self, data, *, steps_per_sample: int = None, us_per_step: float = None,
                      epochs: int = 1, folds: int = 1, warn_seconds: float = 60.0) -> dict:
        """
        Rough wall-clock estimate BEFORE a long run or a split of a massive dataset.
        Times ONE sample read, then multiplies out.
        """
        warn_at = warn_seconds
        n = len(data)

        t0 = time.perf_counter()
        frames, _ = data[0]
        read_s = time.perf_counter() - t0
        T = int(np.asarray(frames).shape[0]) if steps_per_sample is None else steps_per_sample

        ##default 200 us/step (measured at N~1000 dense)
        us = 200.0 if us_per_step is None else us_per_step
        run_s = n * T * us * 1e-6 * epochs * folds
        total = run_s + n * read_s * epochs * folds

        report = {"n_samples": n, "steps_per_sample": T, "epochs": epochs, "folds": folds,
                  "sample_read_s": read_s, "est_run_s": run_s, "est_total_s": total,
                  "est_total_h": total / 3600.0}
        if warn_at and total > warn_at:
            logger.warning("Estimated runtime %.1f s (%.2f h) for %d samples x %d steps x %d epochs "
                           "x %d folds. Reduce the dataset or the step count if that is unintended.",
                           total, total / 3600.0, n, T, epochs, folds)
        return report

    #####################################################  
        #############   Per-timestep   #############

    def begin_sample(self, sample: np.ndarray, ctx: RunContext) -> int:
        """
        Bind one sample and RETURN its length T to the runtime.
        """
        sample = np.asarray(sample)

        ##SAFETY VALVES against erroneous formats
        if sample.ndim != 2:
            raise ValueError(f"Dense sample must be (T, N_in). Got shape {sample.shape}.")
        if sample.shape[1] != ctx.net.n_input:
            raise ValueError(f"Sample has N_in={sample.shape[1]} but the network declares "
                             f"n_input={ctx.net.n_input}.")

        self._frames = sample
        ctx.sim.step_in_sample = 0
        ctx.sim.sample_T = int(sample.shape[0])

        logger.debug("begin_sample: T=%d N_in=%d", ctx.sim.sample_T, sample.shape[1])
        return ctx.sim.sample_T

    def step(self, ctx: RunContext) -> None:
        """Load ONE timestep into upd.step_input and advance the cursor."""

        if self._frames is None:
            raise RuntimeError("DataHandler.step called before begin_sample.")
        ##SAFETYVALVE against silent wrapping around data (restart at T=0)
        if ctx.sim.step_in_sample >= ctx.sim.sample_T:
            raise IndexError(f"step_in_sample={ctx.sim.step_in_sample} is past the end of the sample "
                             f"(T={ctx.sim.sample_T}). The Run's loop should have ended this sample.")

        ctx.upd.step_input[...] = self._frames[ctx.sim.step_in_sample]
        ctx.sim.step_in_sample += 1