#hlsp_plots - plots from ONE finished HLSP run folder (e.g. runs/hlsp_lsm_2026_*, runs/hlsp_mnist_raster, runs/hlsp_nmnist_raster).
# Reads only the files the run wrote; nothing is re-simulated. Every plot is saved as a PNG in <run folder>/plots and shown.
#   raster(run_dir, ...)          spike raster of a window of samples        <- History_*.npz (spike_ptr / spike_i: who spiked at which step)
#   degrees(run_dir, ...)         in / out / total degree over the run       <- network.npz (wiring + W0) + snap_*.npz + TrainedModel_*.npz (W)
#                                 needs CheckpointCfg.keep_last=None (hlsp_lsm_2026.runcfg() sets it), else only the newest 3 snapshots exist
#   label_responses(run_dir, ...) mean spikes per neuron per digit           <- responses.npz (the state vectors + labels the readout used)
# From PowerShell, repo root, env PhasicNet:
#   python -c "import experiments, hlsp_plots as hp; hp.all_plots('runs/hlsp_mnist_raster')"
#   python -c "import experiments, hlsp_plots as hp; hp.raster('runs/hlsp_mnist_raster', part='test', first=0, n=10)"
#SAMPLE TIMING: every sample runs a fixed number of steps: mode "train" (self-organising) = n_fit samples x t_train_ms; mode "eval" = the fit,
# then the validation, then the test samples, each t_state_ms. Labels come from responses.npz (y_train is the fit set, used for both passes).
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from capture import rate_hz

PARTS = ("self-organise", "fit", "val", "test")     #the four consecutive blocks of samples in a run, in the order they were simulated


   ######################################################################################
##########################   Loading   ####################################################
   ######################################################################################

def _one(run_dir: Path, pattern: str) -> Path:
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {run_dir} (was the run saved with save_history=True / save_run=True and did it complete?)")
    return files[-1]


def load_run(run_dir) -> dict:
    """Everything the plots need from a run folder, as plain arrays. Neuron ids are GLOBAL (input first), as in the History file."""
    run_dir = Path(run_dir)
    net = json.loads((run_dir / "network.json").read_text())
    proto = json.loads((run_dir / "runcfg.json").read_text())["protocol"]
    resp = np.load(run_dir / "responses.npz")
    hist_file = _one(run_dir, "History_*.npz")
    hmeta = json.loads(hist_file.with_suffix(".json").read_text())
    dt = float(hmeta["dt"])
    n_tr, n_st = int(round(proto["t_train_ms"] / dt)), int(round(proto["t_state_ms"] / dt))
    n_fit, n_val, n_test = len(resp["y_train"]), len(resp["y_val"]), len(resp["y_test"])
    ##first step and steps per sample of each block (see SAMPLE TIMING)
    starts = np.cumsum([0, n_fit * n_tr, n_fit * n_st, n_val * n_st])
    blocks = {"self-organise": (starts[0], n_tr, resp["y_train"]), "fit": (starts[1], n_st, resp["y_train"]),
              "val": (starts[2], n_st, resp["y_val"]), "test": (starts[3], n_st, resp["y_test"])}
    sl = {k: slice(v[0], v[1]) for k, v in net["neuronslices"].items()}
    return dict(dir=run_dir, net=net, proto=proto, resp=resp, hist=np.load(hist_file), dt=dt, blocks=blocks, slices=sl, n_input=int(net["n_input"]),
                n_total=int(net["n_total"]), partitions=int(proto["partitions"]))


def _save(fig, r: dict, name: str):
    out = r["dir"] / "plots"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / name}.png")


   ######################################################################################
##########################   1. Raster   ##################################################
   ######################################################################################

def raster(run_dir, *, part: str = "test", first: int = 0, n: int = 10, show: bool = True):
    """Spike raster of samples [first, first + n) of one block (part = "self-organise" | "fit" | "val" | "test"). One dot per spike; y = neuron
    (inhibitory at the bottom, excitatory above); grey lines = sample boundaries, the digit above each sample. Keep n small or the dots merge."""
    r = load_run(run_dir)
    s0, per, y = r["blocks"][part]
    first, n = int(first), min(int(n), len(y) - int(first))
    a, b = s0 + first * per, s0 + (first + n) * per                       ##step window
    ptr, idx = r["hist"]["spike_ptr"], r["hist"]["spike_i"]
    ev0, ev1 = int(ptr[a]), int(ptr[b])
    steps = np.searchsorted(ptr, np.arange(ev0, ev1), side="right") - 1    ##event k happened at step t where ptr[t] <= k < ptr[t+1]
    neur = idx[ev0:ev1].astype(np.int64)
    inh = (neur >= r["slices"]["inh"].start) & (neur < r["slices"]["inh"].stop)
    fig, ax = plt.subplots(figsize=(12, 6))
    t_ms = (steps - a) * r["dt"]
    ax.scatter(t_ms[~inh], neur[~inh], s=1.5, c="tab:blue", label="excitatory", rasterized=True, linewidths=0)
    ax.scatter(t_ms[inh], neur[inh], s=1.5, c="tab:red", label="inhibitory", rasterized=True, linewidths=0)
    for k in range(n + 1):
        ax.axvline(k * per * r["dt"], color="0.8", lw=0.6, zorder=0)
    for k in range(n):
        ax.text((k + 0.5) * per * r["dt"], r["n_total"] + 2, str(int(y[first + k])), ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0, n * per * r["dt"]); ax.set_ylim(r["n_input"] - 1, r["n_total"] + 15)
    ax.set_xlabel(f"time in window (ms)   [{part} samples {first}-{first + n - 1}, {per * r['dt']:.0f} ms each]"); ax.set_ylabel("neuron (global id)")
    ax.set_title(f"{r['dir'].name}: raster, {len(neur)} spikes ({len(neur) / max(n, 1):.0f} per sample)")
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0), markerscale=6, frameon=False, fontsize=8)
    _save(fig, r, f"raster_{part}_{first}-{first + n - 1}")
    if show:
        plt.show()
    return fig


def rate_over_run(run_dir, *, bin_samples: int = 50, show: bool = True):
    """Population rate (Hz per neuron, exc and inh) per bin of samples across the WHOLE run, one panel per block. A cheap overview before
    choosing raster windows."""
    r = load_run(run_dir)
    ptr, idx = r["hist"]["spike_ptr"], r["hist"]["spike_i"]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for ax, part in zip(axs, PARTS):
        s0, per, y = r["blocks"][part]
        edges = np.arange(0, len(y) + bin_samples, bin_samples).clip(max=len(y))
        edges = np.unique(edges)
        for kind, col in (("exc", "tab:blue"), ("inh", "tab:red")):
            lo, hi = r["slices"][kind].start, r["slices"][kind].stop
            rate = []
            for e0, e1 in zip(edges[:-1], edges[1:]):
                ev = idx[int(ptr[s0 + e0 * per]):int(ptr[s0 + e1 * per])]
                rate.append(rate_hz(((ev >= lo) & (ev < hi)).sum(), hi - lo, (e1 - e0) * per, r["dt"]))
            ax.plot(edges[1:], rate, color=col, label=kind)
        ax.set_title(part); ax.set_xlabel("samples")
    axs[0].set_ylabel("rate (Hz per neuron)"); axs[0].legend(frameon=False)
    fig.suptitle(f"{r['dir'].name}: population rate over the run")
    _save(fig, r, "rate_over_run")
    if show:
        plt.show()
    return fig


   ######################################################################################
##########################   2. Degrees over time   #######################################
   ######################################################################################

def _weight_files(r: dict) -> list[tuple[str, np.ndarray]]:
    """(label, W) in run order: network.npz W0 (start), every snap_*.npz by its saved progress (stage, sample), TrainedModel (end).
    A snapshot whose W equals the previous one is dropped (learning is off after stage 0)."""
    d = r["dir"]
    out = [("start", np.load(d / "network.npz")["W0"])]
    snaps = []
    for f in d.glob("snap_*.npz"):
        z = np.load(f)
        snaps.append((int(z["progress__stage"]), int(z["progress__i"]), z["W"]))
    for st, i, W in sorted(snaps, key=lambda s: (s[0], s[1])):
        out.append((f"stage {st}, {i}", W))
    tm = sorted(d.glob("TrainedModel_*.npz"))
    if tm:
        out.append(("end", np.load(tm[-1])["W"]))
    keep = [out[0]]
    for lab, W in out[1:]:
        if not np.array_equal(W, keep[-1][1]):
            keep.append((lab, W))
    return keep


def degree_table(run_dir, *, threshold: float = 0.0, weighted: bool = False, liquid_only: bool = True) -> tuple[list[str], dict]:
    """Per checkpoint and neuron: in-, out- and total degree. An edge COUNTS when |w| > threshold (the liquid starts at w = 0); weighted=True
    sums |w| instead. liquid_only: only liquid -> liquid edges (input edges are fixed).
    Returns (labels, {"in"|"out"|"total": array (checkpoints, n_liquid)}), neurons in global order."""
    r = run_dir if isinstance(run_dir, dict) else load_run(run_dir)
    net = np.load(r["dir"] / "network.npz")
    pre, post = net["pre_idx"].astype(np.int64), net["post_idx"].astype(np.int64) + r["n_input"]     ##post_idx is stored without the input offset
    m = pre >= r["n_input"] if liquid_only else np.ones(len(pre), bool)
    n_liq = r["n_total"] - r["n_input"]
    labels, deg = [], {"in": [], "out": [], "total": []}
    for lab, W in _weight_files(r):
        w = np.abs(W[m]).astype(np.float64)
        v = w if weighted else (w > threshold).astype(np.float64)
        d_in = np.bincount(post[m] - r["n_input"], weights=v, minlength=n_liq)
        d_out = np.bincount(pre[m] - r["n_input"], weights=v, minlength=n_liq)
        labels.append(lab); deg["in"].append(d_in); deg["out"].append(d_out); deg["total"].append(d_in + d_out)
    return labels, {k: np.array(v) for k, v in deg.items()}


def degrees(run_dir, *, threshold: float = 0.0, weighted: bool = False, show: bool = True):
    """Top row: in / out / total degree over the checkpoints (median and 10-90 % band, excitatory and inhibitory separately).
    Bottom row: the distribution at the first and the last checkpoint. Works with any number of checkpoints."""
    r = load_run(run_dir)
    labels, deg = degree_table(r, threshold=threshold, weighted=weighted)
    lo = r["n_input"]
    kinds = {"exc": slice(r["slices"]["exc"].start - lo, r["slices"]["exc"].stop - lo), "inh": slice(r["slices"]["inh"].start - lo, r["slices"]["inh"].stop - lo)}
    cols = {"exc": "tab:blue", "inh": "tab:red"}
    what = "strength (sum |w|)" if weighted else f"degree (|w| > {threshold:g})"
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    x = np.arange(len(labels))
    for c, key in enumerate(("in", "out", "total")):
        ax = axs[0, c]
        for kind, s in kinds.items():
            D = deg[key][:, s]
            ax.plot(x, np.median(D, 1), color=cols[kind], marker="o", label=kind)
            ax.fill_between(x, np.percentile(D, 10, 1), np.percentile(D, 90, 1), color=cols[kind], alpha=0.2, lw=0)
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7); ax.set_title(f"{key}-{what}")
        ax = axs[1, c]
        for kind, s in kinds.items():
            for row, ls in ((0, ":"), (-1, "-")):
                ax.hist(deg[key][row, s], bins=40, histtype="step", color=cols[kind], ls=ls, label=f"{kind}, {labels[row]}")
        ax.set_xlabel(f"{key}-{what}"); ax.set_ylabel("neurons")
    axs[0, 0].legend(frameon=False, fontsize=8); axs[1, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{r['dir'].name}: liquid {what} over {len(labels)} checkpoints")
    fig.tight_layout()
    _save(fig, r, "degrees_weighted" if weighted else f"degrees_thr{threshold:g}")
    if show:
        plt.show()
    return fig


   ######################################################################################
##########################   3. Spikes per neuron by label   ##############################
   ######################################################################################

def label_responses(run_dir, *, part: str = "test", show: bool = True):
    """Mean spike count of every liquid neuron for each digit, over one block's state pass (part = "train" (fit) | "val" | "test"); partitions
    are summed. Left: neurons x digit heatmap, sorted by preferred digit. Right: neurons per preferred digit and their selectivity (preferred / mean)."""
    r = load_run(run_dir)
    R, y = r["resp"][f"resp_{part}"].astype(np.float64), r["resp"][f"y_{part}"]
    n_liq = r["n_total"] - r["n_input"]
    R = R.reshape(len(R), r["partitions"], n_liq).sum(1)                  ##state vector = P blocks of N counts -> counts per neuron
    classes = np.unique(y)
    M = np.stack([R[y == c].mean(0) for c in classes], 1)                  ##(neurons, digits)
    pref = M.argmax(1)
    sel = M.max(1) / np.maximum(M.mean(1), 1e-12)
    order = np.lexsort((-sel, pref))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw=dict(width_ratios=(2, 1)))
    im = a1.imshow(M[order], aspect="auto", interpolation="nearest", cmap="viridis")
    a1.set_xticks(range(len(classes)), classes); a1.set_xlabel("digit"); a1.set_ylabel("liquid neuron (sorted by preferred digit)")
    fig.colorbar(im, ax=a1, label=f"mean spikes per sample ({r['proto']['t_state_ms']:.0f} ms)")
    a2.bar(classes, np.bincount(pref, minlength=len(classes)), color="0.6")
    a2.set_xlabel("preferred digit"); a2.set_ylabel("neurons")
    a3 = a2.twinx()
    a3.plot(classes, [np.median(sel[pref == k]) if (pref == k).any() else np.nan for k in range(len(classes))], "o-", color="tab:orange")
    a3.set_ylabel("median selectivity (preferred / mean)", color="tab:orange")
    fig.suptitle(f"{r['dir'].name}: spikes per neuron by label ({part}, {len(y)} samples); silent neurons: {(M.sum(1) == 0).sum()} of {n_liq}")
    fig.tight_layout()
    _save(fig, r, f"label_responses_{part}")
    if show:
        plt.show()
    return fig


def all_plots(run_dir, *, show: bool = False):
    """The standard set for one run, saved to <run folder>/plots: rate over the run, rasters (first 10 self-organising and 10 test samples),
    degrees (count and strength), spikes by label (test)."""
    rate_over_run(run_dir, show=show)
    raster(run_dir, part="self-organise", first=0, n=10, show=show)
    raster(run_dir, part="test", first=0, n=10, show=show)
    degrees(run_dir, show=show)
    degrees(run_dir, weighted=True, show=show)
    label_responses(run_dir, part="test", show=show)
    if not show:
        plt.close("all")
