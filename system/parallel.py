#parallel - INDEPENDENT runs in SEPARATE PROCESSES, one CPU each (e.g. several Optuna trials at the same time).
#
#WHY PROCESSES: the hot loop is NumPy on small arrays under the GIL, so two THREADS share one core; two processes get two cores.
# Each process pins its BLAS/OpenMP threads to 1 (THREAD_VARS), set in THIS process before the pool starts (the children inherit it).
#
#A JOB IS DATA, not live objects (it crosses the process boundary by pickle):
#   Job(build, load, runcfg, build_kw={}, load_kw={}, train_kw={})
#       build    = an IMPORTABLE module-level function returning a PhasicNet (hlsp_lsm_2026.build, deckers_2022.build, ...). No lambdas.
#       build_kw = its keyword arguments (n_liquid=..., seed=..., weights_format="sparse", ...)
#       load     = an IMPORTABLE function returning a ModelInput or (train, test, ...) ModelInputs (hlsp_lsm_2026.load_dataset, ...).
#                  The CHILD calls load(**load_kw) itself, so nothing large is pickled.
#       load_kw  = its keyword arguments (name="nmnist", seed=..., ...)
#       runcfg   = the RunCfg (DataCfg.n_train / n_test pick from what load returns)
#       train_kw = PhasicNet.train keyword arguments (out_dir, run_name, cfg_name, save_run, ...). Give every job its OWN out_dir.
#   -> a Result(report, status, run_id, out_dir, error)
#
#TWO ENTRY POINTS:
#   run_parallel(jobs, workers=2)   all jobs, `workers` at a time, returns the Results in job order. A failed job is a Result with .error.
#   run_in_process(job)             ONE job in a fresh process, blocking. For Optuna: objective(trial) builds the RunCfg, calls
#                                   run_in_process(Job(...)) and returns the metric from .report; study.optimize(objective, n_jobs=2).
#
#WINDOWS: processes start with "spawn" (on every OS), which re-imports the launching script in each child -- the launcher must sit under
# `if __name__ == "__main__":`, and `build` / `load` must be importable from the child (experiments/__init__.py sets sys.path).
# The child's logging goes to its run folder's run.jsonl / results.jsonl; the console shows the parent's lines only.
#
#NOT BUILT: a pruning callback (Optuna's trial.report / should_prune needs a metric DURING the run).
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import logging
import multiprocessing as mp
import os
import traceback

from runcfg import RunCfg

logger = logging.getLogger(__name__)

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")   ##the libraries NumPy may sit on


@dataclass(frozen=True)
class Job:
    """One independent run: build(**build_kw).train(train, runcfg, test_data=test, **train_kw) with (train, test) = load(**load_kw). See the header."""
    build: Callable[..., Any]
    load: Callable[..., Any]
    runcfg: RunCfg
    build_kw: dict = field(default_factory=dict)
    load_kw: dict = field(default_factory=dict)
    train_kw: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, fn in (("build", self.build), ("load", self.load)):
            if getattr(fn, "__name__", "<lambda>") == "<lambda>" or "<locals>" in getattr(fn, "__qualname__", ""):
                raise ValueError(f"Job: `{name}` must be an importable module-level function (a lambda / local function cannot reach another process).")


@dataclass(frozen=True)
class Result:
    """What comes back from a child: the run's report and status record; error = the traceback text when the run failed (report None then)."""
    report: dict | None
    status: dict | None
    run_id: str | None
    out_dir: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _execute(job: Job) -> Result:
    """Runs IN THE CHILD. Never raises: a failure comes back as Result.error so the other jobs finish and the caller sees every outcome."""
    out_dir = job.train_kw.get("out_dir")
    try:
        pnet = job.build(**job.build_kw)
        out = job.load(**job.load_kw)      ##the child loads its own copy
        train, test = (out if isinstance(out, tuple) else (out, None))[:2]
        run = pnet.train(train, job.runcfg, test_data=test, **job.train_kw)
        return Result(run.report, run.status, run.runmeta.run_id, None if out_dir is None else str(out_dir))
    except BaseException:
        return Result(None, None, None, None if out_dir is None else str(out_dir), traceback.format_exc())


def _pinned(threads: int):
    """Set the thread variables for the children (they inherit the environment at spawn) and return a restore() for this process."""
    threads, old = str(int(threads)), {k: os.environ.get(k) for k in THREAD_VARS}
    for k in THREAD_VARS:
        os.environ[k] = threads

    def restore():
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return restore


def run_parallel(jobs, *, workers: int = 2, threads: int = 1) -> list[Result]:
    """
    Every job in `jobs`, `workers` of them at a time, each in its own process (start method "spawn"), BLAS threads per process = `threads` (1: one
    core per run). Blocks until all are done; returns the Results in job order. workers > the number of jobs is cut down to it.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    for j in jobs:
        if not isinstance(j, Job):
            raise TypeError(f"run_parallel: every job must be a parallel.Job, got {type(j).__name__}")
    workers = max(1, min(int(workers), len(jobs)))
    restore = _pinned(threads)
    try:
        ctx = mp.get_context("spawn")
        logger.info("run_parallel: %d job(s), %d at a time, %s thread(s) per process", len(jobs), workers, threads)
        with ctx.Pool(workers) as pool:
            results = pool.map(_execute, jobs, chunksize=1)
    finally:
        restore()
    for k, r in enumerate(results):
        if r.ok:
            logger.info("  job %d: %s run_id=%s -> %s", k, r.status["status"] if r.status else "completed", r.run_id, r.out_dir)
        else:
            logger.error("  job %d FAILED -> %s\n%s", k, r.out_dir, r.error)
    return results


def run_in_process(job: Job, *, threads: int = 1) -> Result:
    """ONE job in a fresh process, blocking; the Result as run_parallel gives it. The Optuna objective's one call (header)."""
    return run_parallel([job], workers=1, threads=threads)[0]
