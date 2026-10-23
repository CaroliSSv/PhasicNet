"""HLSP speed / results / RAM comparison: mode "hybrid_event" | "event" (sparse weights) | "dense" (dense weights, timestep), with rasters and degrees saved.

One call = one run = one Python process, so the peak RAM in compare.json belongs to that run alone:
    python -c "import experiments, compare_runs as c; c.run('event', seed=1, n_train=2000, n_test=500, every=500)"
    python -c "import experiments, compare_runs as c; c.summary()"
n_train / n_test None = the full MNIST set (60k train, of which 10k become validation, + 10k test). raster=False for long runs: the spike
record grows with every step (about 0.7 GB of RAM on full MNIST).
Each run folder runs/compare_hlsp_<mode>_s<seed>[_tag] holds HLSP's own results.json (its readout score + seconds per stage), compare.json
(time, peak RAM, accuracy), degrees.npz (hlsp_plots.degree_table) and hlsp_plots' plots (rasters only with raster=True)."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import RUNS_DIR
from utils import _read_json_yaml, _write_json_yaml
import hlsp_lsm_2026 as hlsp
import hlsp_plots

MODES = ("hybrid_event", "event", "dense")


def peak_ram_mb() -> float | None:
    """Peak resident memory of THIS process in MB: psutil if installed, else the Windows API call, else the unix one."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 2**20
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):   ##PROCESS_MEMORY_COUNTERS, read-only numbers about our own process
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = Counters(); c.cb = ctypes.sizeof(c)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        return None


def run(mode: str = "hybrid_event", *, seed: int = 1, n_train: int | None = None, n_test: int | None = None, every: int | None = None,
        raster: bool = True, dataset: str = "mnist", tag: str = ""):
    """ONE HLSP run with the paper's defaults; only the update mode, the spike record and the saving differ from hlsp_lsm_2026.main(). Returns the Run."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    dense = mode == "dense"
    out = RUNS_DIR / f"compare_hlsp_{mode}_s{seed}{('_' + tag) if tag else ''}"
    train, test, n_input = hlsp.load_dataset(dataset, peak_hz=hlsp.PEAK_HZ, seed=seed)
    pnet = hlsp.build(n_input=n_input, w_in=hlsp.W_IN[dataset], seed=seed, weights_format="dense" if dense else "sparse")
    if not dense:
        pnet.cfg = replace(pnet.cfg, simulation=replace(pnet.cfg.simulation, update_on=mode))
    rc = hlsp.runcfg(every=every, partitions=hlsp.PARTITIONS[dataset], train_window="subsample" if dataset == "nmnist" else "start", n_train=n_train, n_test=n_test)
    rc = replace(rc, history=replace(rc.history, record_spikes=raster))
    r = pnet.train(train, rc, test_data=test, out_dir=out, save_run=True, save_history=True, dataset_name=dataset)
    rep = r.report
    _write_json_yaml(out / "compare", {"mode": mode, "weights_format": pnet.cfg.simulation.weights_format, "update_on": pnet.cfg.simulation.update_on,
                                       "seed": seed, "n_train": n_train, "n_test": n_test, "seconds": rep["seconds"], "peak_ram_mb": peak_ram_mb(),
                                       "test_acc": rep.get("test_acc"), "val_acc": rep.get("val_acc"), "train_acc": rep.get("train_acc")},
                     JSON=True, YAML=False, log_label="Compare")
    plots(out, raster=raster)
    return r


def plots(run_dir, *, raster: bool = True):
    """hlsp_plots' set for one run (rasters and the rate need raster=True) + degrees.npz = in/out/total degree per saved weight state."""
    r = hlsp_plots.load_run(run_dir)
    labels, deg = hlsp_plots.degree_table(r)
    np.savez(r["dir"] / "degrees.npz", labels=np.array(labels), **deg)     ##(checkpoints, n_liquid) each, neurons in global order
    if raster:
        hlsp_plots.all_plots(run_dir)
    else:
        hlsp_plots.degrees(run_dir, show=False); hlsp_plots.degrees(run_dir, weighted=True, show=False); hlsp_plots.label_responses(run_dir, show=False)
    plt.close("all")


def summary(pattern: str = "compare_hlsp_*"):
    """One line per finished compare run: seconds (total and self-organising pass), peak RAM, accuracy."""
    rows = []
    for f in sorted(RUNS_DIR.glob(f"{pattern}/compare.json")):
        c = _read_json_yaml(f)
        rows.append((f.parent.name, c["seconds"]["total"], c["seconds"].get("train"), c["peak_ram_mb"], c["test_acc"]))
    print(f"{'run':40s} {'total s':>9s} {'train s':>9s} {'peak MB':>9s} {'test %':>7s}")
    for name, s, st, mb, acc in rows:
        print(f"{name:40s} {s:9.0f} {st if st is None else round(st):>9} {mb if mb is None else round(mb):>9} {acc if acc is None else round(acc, 2):>7}")
    return rows
