#plots - figures from finished run folders, for ANY model. Nothing is re-simulated.
# Reads only through the framework's loaders (RuntimeClasses.from_paths, PhasicNet.load_net / load_cfg / load_runcfg, RunHistory.load,
# checkpoint.snapshots / load_snapshot / load_trained_model, utils._read_responses_npz / _read_json_yaml) and draws with analysis.py where it has the plot.
#
#   run = load(run_dir)                                      one finished run; the heavy parts are read on first use
#   raster(run, window_ms=(0, 1000))                         analysis.plot_raster (needs HistoryCfg.record_spikes=True)
#   rate_over_time(run, bin_ms=1000)                         analysis.population_rate per neuron group, train / eval stretches shaded
#   rates(run)                                               analysis.plot_rate_distribution
#   separability(run, part="test")                           analysis.plot_separability, scored by scorer.calinski_harabasz
#   over_weights(run, y=capture.in_degree, series=...)       ANY capture function of the weights, at every saved weight state (start, snapshots, end)
#   across_runs(run_dirs, y="test_acc", by="simulation.update_on")   ANY saved value or scorer, one point per run, grouped
#   all_plots(run_dir)                                       the standard set, saved into <run folder>/plots
#
# The y-axis is an input:
#   over_weights: any f(ctx, **kw) from capture (in_degree, out_degree, degree_strength, mean_globalweight, deadened_synapses, ...)
#   across_runs:  a saved value by name (run.value: results.json key, "<file>.<key>" for another json in the folder, or a cfg / runcfg field)
#                 or any f(run) -> number, e.g. lambda r: scorer.calinski_harabasz(r.responses["resp_test"], r.responses["y_test"])
# From PowerShell, repo root, env PhasicNet:
#   python -c "import experiments, plots; plots.all_plots('runs/compare_hlsp_event_s1')"
from __future__ import annotations
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import analysis
import capture
import scorer
from phasicnet import PhasicNet, RuntimeClasses
from history import RunHistory
from states import RunContext, SynapticState
from checkpoint import snapshots, load_snapshot, load_trained_model
from utils import _read_json_yaml, _read_responses_npz


   ######################################################################################
##########################   Loading   ####################################################
   ######################################################################################

class SavedRun:
    """One finished run folder (what PhasicNet.train(out_dir=...) wrote). Network, cfg and runcfg load at once; history, responses, results and
    the weight states on first use."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.name = self.dir.name
        RuntimeClasses.from_paths(_read_json_yaml(self.dir / "runtime_classes.json"))     ##imports the run's modules: their cfg classes register
        self.pnet = PhasicNet()
        self.pnet.load_net(self.dir / "network.json")
        self.pnet.load_cfg(self.dir / "cfg.json")
        self.pnet.load_runcfg(self.dir / "runcfg.json")

    @property
    def net(self):
        return self.pnet.network

    @property
    def cfg(self):
        return self.pnet.cfg

    @property
    def runcfg(self):
        return self.pnet.runcfg

    @cached_property
    def history(self) -> RunHistory:
        files = sorted(self.dir.glob("History_*.json"))
        if not files:
            raise FileNotFoundError(f"no History_*.json in {self.dir} (train(..., save_history=True))")
        return RunHistory.load(files[-1])

    @cached_property
    def ctx(self) -> RunContext:
        """The network and its weights in the run's own layout (to_sparse / _to_dense), W0 in place. Only what capture's weight functions read."""
        weights = self.pnet._to_dense() if self.cfg.simulation.weights_format == "dense" else self.pnet.to_sparse()
        return RunContext(sim=None, upd=None, neur=None, syn=SynapticState(weights), net=self.net, rng=None)

    @cached_property
    def meta(self) -> dict:
        return capture.raster_meta(self.ctx, self.history)

    @cached_property
    def responses(self) -> dict:
        f = self.dir / "responses.npz"
        return _read_responses_npz(f) if f.exists() else {}

    @cached_property
    def results(self) -> dict:
        f = self.dir / "results.json"
        return _read_json_yaml(f) if f.exists() else {}

    def weight_states(self) -> list[tuple[str, np.ndarray]]:
        """(label, W) in run order: start (W0), every snapshot by its (stage, sample), end (TrainedModel). A W equal to the previous one is dropped."""
        states = [("start", self.ctx.syn.weights.W.copy())]
        for f in snapshots(self.dir):
            snap = load_snapshot(f)
            states.append((f"stage {int(snap.progress.get('stage', 0))}, {int(snap.progress.get('i', 0))}", snap.W))
        trained = sorted(self.dir.glob("TrainedModel_*.json"))
        if trained:
            states.append(("end", load_trained_model(trained[-1]).W))
        keep = states[:1]
        for label, W in states[1:]:
            if not np.array_equal(W, keep[-1][1]):
                keep.append((label, W))
        return keep

    def value(self, key: str):
        """A saved value by dotted name: results.json first, then "<file>.<key>" for another json in the folder, then the cfg, then the runcfg."""
        head, _, rest = key.partition(".")
        sources = [self.results]
        if rest and (self.dir / f"{head}.json").exists():
            sources, key = [_read_json_yaml(self.dir / f"{head}.json")], rest
        sources += [asdict(self.cfg), asdict(self.runcfg)]
        for source in sources:
            found = _lookup(source, key)
            if found is not None:
                return found
        raise KeyError(f"{key!r} is not in {self.name}'s results, json files, cfg or runcfg")


def _lookup(d: dict, key: str):
    for part in key.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def load(run_dir) -> SavedRun:
    return run_dir if isinstance(run_dir, SavedRun) else SavedRun(run_dir)


def _groups(run: SavedRun) -> list[str]:
    """The non-input neuron groups, in true-neuron order."""
    return [g for g in sorted(run.net.neuronslices, key=lambda g: run.net.neuronslices[g].start) if not run.net.neuronspecs[g].input]


def _out(fig, path):
    """Save to `path` (and close) when one is given; else return the figure."""
    if path is None:
        return fig
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


   ######################################################################################
##########################   Spikes (the History)   #######################################
   ######################################################################################

def raster(run, *, window_ms: tuple[float, float] | None = None, path=None):
    """analysis.plot_raster over `window_ms` (None = the whole run: slow for long runs)."""
    run = load(run)
    if not run.history.cfg.record_spikes:
        raise ValueError(f"{run.name} recorded no spikes (HistoryCfg.record_spikes=False)")
    return analysis.plot_raster(run.history, run.meta, window_ms=window_ms, title=f"{run.name}", path=path)


def rate_over_time(run, *, bin_ms: float = 1000.0, path=None):
    """analysis.population_rate of every neuron group over the whole run; stretches in eval mode shaded."""
    run = load(run)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for g in _groups(run):
        s = run.net.neuronslices[g]
        t, rate = analysis.population_rate(run.history, run.meta, bin_ms=bin_ms, band=(s.start, s.stop))
        ax.plot(t / 1000.0, rate, label=g, color=analysis.ROLE_COLORS["inh" if run.net.neuronspecs[g].inh else "exc"])
    t0, dt = run.history.t_start or 0, run.history.dt
    for span in analysis.segment_spans(run.history, "mode"):
        if span["value"] == 1:
            ax.axvspan((span["start_step"] - t0) * dt / 1000.0, (span["stop_step"] - t0) * dt / 1000.0, color="0.92", zorder=0)
    ax.set(xlabel="run time (s)", ylabel=f"rate (Hz per neuron, {bin_ms:g} ms bins)", title=f"{run.name}: population rate (shaded = eval)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _out(fig, path)


def rates(run, *, path=None):
    """analysis.plot_rate_distribution: every neuron's rate over the whole run."""
    run = load(run)
    return analysis.plot_rate_distribution(run.history, run.meta, path=path)


   ######################################################################################
##########################   Responses (what the readout saw)   ###########################
   ######################################################################################

def separability(run, *, part: str = "test", path=None):
    """analysis.plot_separability of the saved responses of `part` ("train", "val", "test", ...), titled with scorer.calinski_harabasz."""
    run = load(run)
    R, y = run.responses[f"resp_{part}"], run.responses[f"y_{part}"]
    if R.ndim == 3:                                     ##protocols with several test passes: the first
        R = R[0]
    return analysis.plot_separability(R, y, score=scorer.calinski_harabasz(R, y), title=f"{run.name} {part}", path=path)


   ######################################################################################
##########################   The weights over the run   ###################################
   ######################################################################################

def recurrent_in_degree(ctx, *, post_group: str | None = None, eps: float = 0.0):
    """capture.in_degree counting only edges from non-input groups (input edges stay fixed in most models)."""
    pre = [g for g in ctx.net.neuronslices if not ctx.net.neuronspecs[g].input]
    return sum(capture.in_degree(ctx, post_group=post_group, pre_group=g, eps=eps) for g in pre)


def weight_series(run, y: Callable, **kw) -> tuple[list[str], list]:
    """y(ctx, **kw) at every saved weight state -> (labels, values). The live W is restored afterwards."""
    run = load(run)
    weights = run.ctx.syn.weights
    W0, labels, values = weights.W, [], []
    for label, W in run.weight_states():
        weights.W = W
        labels.append(label); values.append(y(run.ctx, **kw))
    weights.W = W0
    return labels, values


def over_weights(run, y: Callable = capture.in_degree, *, series: dict[str, dict] | None = None, ylabel: str | None = None, path=None):
    """y(ctx, **kw) at every saved weight state, one line per series {label: kw}. Arrays (one value per neuron): median and 10-90 % band on top,
    the distribution at the first and the last state below. Scalars: one line.
        over_weights(run, capture.in_degree, series={"exc": dict(post_group="exc"), "inh": dict(post_group="inh")})"""
    run = load(run)
    series = {"all": {}} if series is None else series
    data = {name: weight_series(run, y, **kw) for name, kw in series.items()}
    labels = next(iter(data.values()))[0]
    arrays = np.ndim(next(iter(data.values()))[1][0]) > 0
    ylabel = ylabel or getattr(y, "__name__", "y")
    fig, axs = plt.subplots(2 if arrays else 1, 1, figsize=(10, 7 if arrays else 3.5), squeeze=False)
    x = np.arange(len(labels))
    for name, (_, vals) in data.items():
        V = np.asarray(vals, dtype=np.float64)
        if arrays:
            line, = axs[0, 0].plot(x, np.median(V, axis=1), marker="o", label=name, color=analysis.ROLE_COLORS.get(name))
            axs[0, 0].fill_between(x, np.percentile(V, 10, axis=1), np.percentile(V, 90, axis=1), color=line.get_color(), alpha=0.2, lw=0)
            for row, ls in ((0, ":"), (-1, "-")):
                axs[1, 0].hist(V[row], bins=40, histtype="step", ls=ls, color=line.get_color(), label=f"{name}, {labels[row]}")
        else:
            axs[0, 0].plot(x, V, marker="o", label=name, color=analysis.ROLE_COLORS.get(name))
    axs[0, 0].set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    axs[0, 0].set(ylabel=ylabel, title=f"{run.name}: {ylabel} at {len(labels)} weight states" + (" (median, 10-90 %)" if arrays else ""))
    axs[0, 0].legend(frameon=False, fontsize=8)
    if arrays:
        axs[1, 0].set(xlabel=ylabel, ylabel="neurons"); axs[1, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _out(fig, path)


   ######################################################################################
##########################   Several runs   ###############################################
   ######################################################################################

def across_runs(run_dirs, y: str | Callable = "test_acc", *, by: str | Callable | None = None, ylabel: str | None = None, path=None):
    """One point per run, grouped on the x-axis by `by` (a run.value name or f(run) -> label; None = one group per run); the group mean as a bar.
    y: a run.value name or f(run) -> number."""
    runs = [load(d) for d in run_dirs]
    get = (lambda r, k: r.value(k) if isinstance(k, str) else k(r))
    groups = {}
    for r in runs:
        groups.setdefault(str(r.name if by is None else get(r, by)), []).append(float(get(r, y)))
    fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(groups)), 3.8))
    for i, (g, vals) in enumerate(groups.items()):
        ax.bar(i, np.mean(vals), color="0.85", width=0.6)
        ax.scatter(np.full(len(vals), i), vals, color="k", s=14, zorder=3)
    ax.set_xticks(range(len(groups)), list(groups), rotation=30, ha="right")
    ax.set(ylabel=ylabel or (y if isinstance(y, str) else getattr(y, "__name__", "y")), title=f"{len(runs)} runs")
    fig.tight_layout()
    return _out(fig, path)


   ######################################################################################
##########################   The standard set   ###########################################
   ######################################################################################

def all_plots(run_dir, *, window_ms: float = 1000.0):
    """Into <run folder>/plots: raster of the first `window_ms` and population rate (if spikes were recorded), per-neuron rates, separability of every
    saved response part, recurrent in-degree and out-degree per group over the saved weight states."""
    run = load(run_dir)
    out = run.dir / "plots"
    out.mkdir(exist_ok=True)
    if run.history.cfg.record_spikes:
        raster(run, window_ms=(0.0, min(window_ms, run.history.n_steps * run.history.dt)), path=out / "raster.png")
        rate_over_time(run, path=out / "rate_over_time.png")
    rates(run, path=out / "rates.png")
    for part in ("train", "val", "test"):
        if f"resp_{part}" in run.responses and len(run.responses[f"y_{part}"]):
            separability(run, part=part, path=out / f"separability_{part}.png")
    groups = _groups(run)
    over_weights(run, recurrent_in_degree, series={g: dict(post_group=g) for g in groups}, ylabel="recurrent in-degree", path=out / "in_degree.png")
    over_weights(run, capture.out_degree, series={g: dict(pre_group=g) for g in groups}, ylabel="out-degree", path=out / "out_degree.png")
    return out
