#analysis - reading, summarizing and PLOTTING a stored RunHistory.
#  
# DIVISION OF LABOUR (deliberate, mirrors Trial vs DataLoader):
#   capture      module of stateless observables (HOW to capture ...)
#   Run,RunCfg           WHEN+what to capture/measure/record
#   RunHistory   append-only timestep-pointer storage + meta to read it
#   analysis      EVERY reconstruction, aggregation and figure. Nothing during simulation
#                   works from RunHistory + meta dict from capture.raster_meta()
from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import _read_responses_npz

logger = logging.getLogger(__name__)

##per neuron-role
ROLE_COLORS = {"input": "#8c8c8c", "exc": "#1f5fa9", "inh": "#c1443c"}


def _role(band: dict) -> str:
    """Role of one neuron band. input wins over inh: an inhibitory INPUT node is still input."""
    if band["input"]:
        return "input"
    return "inh" if band["inh"] else "exc"


# ##################################
#               Reading
# #################################

##history = timestep-pointer (CSR, timelines) w/ events stored on demand
def events(history) -> dict[str, np.ndarray]:
    """COO view of the stored raster. spike_t = ABSOLUTE runstep, spike_i = true_neuronID."""
    spike_t, spike_i = history.events()
    return {"spike_t": spike_t, "spike_i": spike_i}


def n_steps(history) -> int:
    """
    Number of RECORDED timesteps.
    """
    return int(history.n_steps)


def dense_raster(history, T: Optional[int] = None) -> np.ndarray:
    """(T, n_total) bool. -- offline."""
    return history.sparse2dense(T)


def spikes_at(history, step: int) -> np.ndarray:
    """true_neuronIDs that fired at absolute runstep "step". O(1) under the pointer form."""
    return history.spikes_at(step)


def spike_counts(history, n_total: Optional[int] = None) -> np.ndarray:
    """Total spikes per neuron, (n_total,)."""
    spike_i = events(history)["spike_i"]
    length = history.n_total if n_total is None else n_total
    return np.bincount(spike_i.astype(np.int64), minlength=length)


def firing_rates(history, meta: dict, T: Optional[int] = None) -> np.ndarray:
    """
    Per-neuron rate in Hz, (n_total,).

    dt is in ms (the framework's convention), so duration_s = T * dt / 1000.
    NOTE: started at meta["capture_start"]. Neurons OUTSIDE the capture span => 0.0.
    """
    T = n_steps(history) if T is None else T
    if T == 0:
        return np.zeros(meta["n_total"], dtype=np.float64)
    seconds = T * meta.get("dt", history.dt) / 1000.0
    return spike_counts(history, meta["n_total"]) / seconds


def population_rate(history, meta: dict, *, bin_ms: float = 10.0, T: Optional[int] = None,
                    band: Optional[tuple[int, int]] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Binned population rate.

    Args:
        band: (start, stop) true_neuronIDs to restrict to. None = every captured neuron.

    Returns:
        (bin_centres_in_ms, rate_in_Hz_per_neuron)
    """
    arrays = events(history)
    ##spike_t is ABSOLUTE: bin against the recorded window so bin 0 is the first RECORDED step.
    t0 = int(meta.get("t_start") or 0)
    spike_t, spike_i = arrays["spike_t"].astype(np.int64) - t0, arrays["spike_i"].astype(np.int64)
    T = n_steps(history) if T is None else T
    dt = meta.get("dt", history.dt)

    if band is None:
        start, stop = meta["capture_start"], meta["capture_stop"]
    else:
        start, stop = band
    keep = (spike_i >= start) & (spike_i < stop)
    spike_t = spike_t[keep]

    n_neurons = max(stop - start, 1)
    steps_per_bin = max(int(round(bin_ms / dt)), 1)
    n_bins = max(int(np.ceil(T / steps_per_bin)), 1)

    counts = np.bincount(spike_t // steps_per_bin, minlength=n_bins)[:n_bins]
    seconds = steps_per_bin * dt / 1000.0
    centres = (np.arange(n_bins) + 0.5) * steps_per_bin * dt
    return centres, counts / (seconds * n_neurons)


def describe(history, meta: dict, T: Optional[int] = None) -> str:
    """Text summary"""
    T = n_steps(history) if T is None else T
    arrays = events(history)
    rates = firing_rates(history, meta, T)
    start, stop = meta["capture_start"], meta["capture_stop"]

    lines = [f"RunHistory '{history.hist_name}': {arrays['spike_i'].size} spikes over {T} steps "
             f"(dt={meta.get('dt', history.dt)} ms)",
             f"  captured neurons {start}..{stop - 1} of {meta['n_total']} "
             f"(include_input={meta['include_input']})"]
    
    for name, band in meta["bands"].items():
        sub = rates[band["start"]:band["stop"]]
        recorded = band["start"] >= start
        silent = int((sub == 0).sum())
        lines.append(f"  {name:<8} n={band['n']:<4} role={_role(band):<5} "
                     + (f"rate {sub.mean():6.2f} Hz (max {sub.max():6.2f})  silent {silent}/{band['n']}"
                        if recorded else "NOT CAPTURED"))
    
    ##phase/epoch/datarun = SEGMENT TIMELINES
    for timeline in (history.phase_timeline, history.epoch_timeline, history.datarun_timeline):
        segs = timeline.segments()
        if segs:
            lines.append(f"  {timeline.name:<8} {len(segs)} segment(s): "
                         + ", ".join(f"{s['value']}@{s['start_step']}" for s in segs[:6])
                         + (" ..." if len(segs) > 6 else ""))
    return "\n".join(lines)


def segment_spans(history, timeline_name: str = "phase") -> list[dict]:
    """
    Segments as (value, start_step, stop_step), stop taken from the RECORDED extent.
    Useful for shading a raster by phase.
    """
    timeline = getattr(history, f"{timeline_name}_timeline")
    segs = timeline.segments()
    end = (history.t_start or 0) + history.n_steps
    spans = []
    for i, seg in enumerate(segs):
        stop = segs[i + 1]["start_step"] if i + 1 < len(segs) else end
        spans.append({"value": seg["value"], "start_step": seg["start_step"], "stop_step": stop})
    return spans


############################################# PLOTS #########################################3

def plot_raster(history, meta: dict, *, path: str | Path = None, T: Optional[int] = None,
                bin_ms: float = 10.0, title: str = None, show_population: bool = True,
                marker_size: float = 2.0, window_ms: Optional[tuple[float, float]] = None,
                rate_band: Optional[tuple[int, int]] = None):
    """
    Spike raster (neuron id vs time), coloured by neuron ROLE, with an optional population-rate panel below.

        - window_ms=(t_from, t_to) in ms: plot only that stretch of time
        - rate_band=(start, stop) true_neuronIDs the population-rate panel averages over. None = every captured neuron. 
    """
    arrays = events(history)
    t0 = int(meta.get("t_start") or 0)
    spike_t, spike_i = arrays["spike_t"].astype(np.int64) - t0, arrays["spike_i"].astype(np.int64)
    T = n_steps(history) if T is None else T
    dt = meta.get("dt", history.dt)
    start, stop = meta["capture_start"], meta["capture_stop"]

    w_a, w_b = (0.0, T * dt) if window_ms is None else (float(window_ms[0]), float(window_ms[1]))
    if window_ms is not None:
        in_window = (spike_t * dt >= w_a) & (spike_t * dt < w_b)
        spike_t, spike_i = spike_t[in_window], spike_i[in_window]

    n_panels = 2 if show_population else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 6 if show_population else 4.2),
                             sharex=True, height_ratios=[3, 1] if show_population else None)
    ax = axes[0] if show_population else axes

    ##One scatter per BAND. legend by role 
    seen_roles = []
    for name, band in meta["bands"].items():
        if band["stop"] <= start:
            continue                    ##no band captured _> continue
        in_band = (spike_i >= band["start"]) & (spike_i < band["stop"])
        role = _role(band)
        n_band = band["stop"] - band["start"]
        ##small bands get larger ticks so they stay visible
        ax.scatter(spike_t[in_band] * dt, spike_i[in_band], s=marker_size if n_band > 20 else marker_size * 30,
                   c=ROLE_COLORS[role], marker="|", linewidths=0.8,
                   label=name if role not in seen_roles else None)
        seen_roles.append(role)
        ##Band boundary + label in the margin.
        ax.axhline(band["stop"] - 0.5, color="0.85", lw=0.6, zorder=0)
        ax.text(w_b + (w_b - w_a) * 0.005, (band["start"] + band["stop"]) / 2 - 0.5, name,
                va="center", ha="left", fontsize=8, color=ROLE_COLORS[role])

    if not meta["include_input"] and start > 0:
        ## WHY the input band is blank
        ax.axhspan(-0.5, start - 0.5, color="0.96", zorder=0)
        ax.text((w_a + w_b) / 2, (start - 1) / 2, "input not captured (HistoryCfg.include_input=False)",
                ha="center", va="center", fontsize=8, color="0.55", style="italic")

    ax.set_ylim(-0.5, meta["n_total"] - 0.5)
    ax.set_xlim(w_a, w_b)
    ax.set_ylabel("true neuron id")  
    ax.set_title(title or f"Spike raster -- {history.hist_name}", loc="left")
    handles = [Line2D([], [], color=c, marker="|", ls="", ms=8, label=r)
               for r, c in ROLE_COLORS.items() if r in seen_roles]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    if show_population:
        centres, rate = population_rate(history, meta, bin_ms=bin_ms, T=T, band=rate_band)
        axes[1].fill_between(centres, rate, step="mid", color="#4a4a4a", alpha=0.7)
        axes[1].set_ylabel(f"pop. rate\n(Hz/neuron, {bin_ms:g} ms bins)", fontsize=8)
        axes[1].set_xlabel("time (ms)")
        axes[1].margins(x=0)
    else:
        ax.set_xlabel("time (ms)")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("raster written to %s", path)
        plt.close(fig)
        return path
    return fig


def plot_rate_distribution(history, meta: dict, *, path: str | Path = None, T: Optional[int] = None):
    """Per-neuron rate as a bar per neuron."""
    rates = firing_rates(history, meta, T)
    start = meta["capture_start"]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    colors = ["0.9"] * meta["n_total"]
    for name, band in meta["bands"].items():
        for idx in range(band["start"], band["stop"]):
            colors[idx] = ROLE_COLORS[_role(band)] if idx >= start else "0.9"
    ax.bar(np.arange(meta["n_total"]), rates, color=colors, width=0.85)
    ax.set_xlabel("true neuron id")  
    ax.set_ylabel("rate (Hz)")
    ax.set_title("Per-neuron firing rate", loc="left")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig


def plot_separability(responses, labels, *, score: Optional[float] = None, title: str = None,
                      class_names: Optional[dict] = None, axes=None, path: str | Path = None):
    """
    The picture behind a separability score: do per-sample response vectors fall apart by label?

    Args:
        responses = (n_samples, n_features) spike counts per sample
        labels = (n_samples,) class per sample
        score = Calinski-Harabasz value to PRINT in the title (computed by Scorer.separability, not here)
        class_names = {label: text} for the legend
        axes = optional pair (ax_left, ax_right) to draw into. None -> a new figure.

    n_features == 1: per-class histograms (left), class mean +- sd (right).
    n_features >= 2: PCA scatter (left), class-mean response of the 30 most class-dependent neurons (right).
    Returns the figure, or the path if one was given.
    """
    responses = np.asarray(responses, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    names = {c: (class_names or {}).get(c, str(c)) for c in classes}
    colors = plt.cm.tab10(np.arange(len(classes)) % 10)

    own_figure = axes is None
    if own_figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = axes[0].figure
    ax_left, ax_right = axes

    if responses.shape[1] == 1:
        x = responses[:, 0]
        bins = np.histogram_bin_edges(x, bins=15)
        for c, col in zip(classes, colors):
            ax_left.hist(x[labels == c], bins=bins, alpha=0.6, color=col, label=names[c])
        ax_left.set(xlabel="response (spikes per sample)", ylabel="samples")
        ax_left.legend(fontsize=8)
        ax_right.bar([names[c] for c in classes], [x[labels == c].mean() for c in classes],
                     yerr=[x[labels == c].std() for c in classes], color=colors, capsize=4)
        ax_right.set(ylabel="mean response +- sd")
        ax_right.tick_params(axis="x", labelsize=7)
    else:
        centred = responses - responses.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        projected = centred @ vt[:2].T
        for c, col in zip(classes, colors):
            in_class = labels == c
            ax_left.scatter(projected[in_class, 0], projected[in_class, 1], s=12, color=col, label=names[c])
        ax_left.set(xlabel="PC1", ylabel="PC2")
        ax_left.legend(fontsize=8)
        class_means = np.vstack([responses[labels == c].mean(axis=0) for c in classes])
        top = np.argsort(class_means.max(axis=0) - class_means.min(axis=0))[::-1][:30]
        image = ax_right.imshow(class_means[:, top], aspect="auto", cmap="viridis")
        ax_right.set_yticks(range(len(classes)), [names[c] for c in classes])
        ax_right.set(xlabel="neuron (the 30 most class-dependent)")
        fig.colorbar(image, ax=ax_right, label="mean spike count")

    heading = title or "separability"
    ax_left.set_title(heading if score is None else f"{heading}   (Calinski-Harabasz = {score:.2f})", loc="left", fontsize=10)

    if own_figure:
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return path
    return fig


   ######################################################################################
##########################   Stored protocol runs (experiments/dc_reproduce.py)   ################
   ######################################################################################
##reads a finished runs/DC_<tag>/ folder. Works from the saved files only (responses.npz + results.json): no simulation

def protocol_scores(run_dir: str | Path, *, n_classes: int = 10) -> dict:  ##NOTE TODO protocol_scores is misleading. DC2015_score is better. 
    """
    DC score of a finished protocol run, recomputed from its stored responses (exc spike counts per example).


    Returns:
        accuracy_per_pass   Diehl & Cook vote accuracy (%), labels from the assignment pass:(-1 for silent neurons, ties -> higher class)
        confusion           (passes, n_classes, n_classes), row = true, column = predicted
        neurons_per_class   how many exc neurons were assigned to each digit (-1 = never fired: excluded)
    """
    from scorer import dc_assign_labels, dc_rank_predict, confusion_matrix, accuracy
    run_dir = Path(run_dir)
    z = _read_responses_npz(run_dir / "responses.npz")
    r_as, y_as, r_te, y_te = z["resp_assign"], z["y_assign"], z["resp_test"], z["y_test"]
    assign = dc_assign_labels(r_as, y_as, n_classes)
    acc, cms = [], []
    for r in r_te:                                       ##one entry per test pass
        yhat = dc_rank_predict(r, assign, n_classes)
        acc.append(accuracy(y_te, yhat)); cms.append(confusion_matrix(y_te, yhat, n_classes))
    return {"accuracy_per_pass": acc, "accuracy_mean": float(np.mean(acc)), "confusion": np.stack(cms),
            "neurons_per_class": np.bincount(assign[assign >= 0], minlength=n_classes).tolist(), "unassigned": int((assign < 0).sum())}


def plot_protocol_separability(run_dir: str | Path, *, test_pass: int = 0, path: str | Path = None): ###NOTE TODO again this is a CH<year>__score NOT "protocol" which is vague
    """plot_separability for a stored protocol run's test responses"""
    from scorer import calinski_harabasz
    run_dir = Path(run_dir)
    z = _read_responses_npz(run_dir / "responses.npz")
    r, y = z["resp_test"][test_pass], z["y_test"]
    return plot_separability(r, y, score=calinski_harabasz(r, y), title=f"{run_dir.name} test pass {test_pass + 1}", path=path)
 