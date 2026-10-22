#test_round3_capture_clamps_since -- clamps, finite values, Since allocation and capture readings:
#   clamps are explicit: no bound in the cfg -> no per-step clamp; a bound -> the per-sign clip
#   weights and voltages stay finite (a test, not a runtime mode)
#   Since is only allocated when a module requires it; the D&C modules hold no copy
#   Scorer's readings come from capture (same numbers)
from __future__ import annotations
import dataclasses
import numpy as np
import pytest

from runcfg import RunCfg
from scorer import Scorer
import capture
from test_lookup_and_loading import _pnet, _data, N_IN


def _run(fmt, **plasticity):
    p = _pnet(False, fmt)
    p.cfg = dataclasses.replace(p.cfg, plasticity=dataclasses.replace(p.cfg.plasticity, **plasticity))
    return p._prep_run(_data(), RunCfg())


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
def test_no_bounds_means_no_clamp(fmt):
    run = _run(fmt, w_min=None, w_max=None)
    assert run._clamp_plan in ([], None)
    run = _run(fmt, w_min=None, w_max=6.0)                 ##ONE bound set -> the per-sign clip is planned
    assert run._clamp_plan not in ([], None)


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
def test_weights_and_voltages_stay_finite(fmt):
    run = _run(fmt, w_min=None, w_max=None)                ##no clamp at all: the honest case for a finite check
    for frames, _ in _data():
        run.run_sample(frames)
    assert np.all(np.isfinite(run.ctx.syn.W)) and np.all(np.isfinite(run.ctx.neur.V))


def test_since_is_allocated_only_when_required():
    from diehlcook import DCConductanceSynapse, DCLIFeuler, DCAdaptiveThreshold
    run = _run("dense")
    assert run.ctx.neur.Since is None                       ##the default modules do not require it: no per-step cost
    for cls in (DCConductanceSynapse, DCLIFeuler, DCAdaptiveThreshold):
        assert "Since" in cls.requires and "Since" not in vars(cls)      ##declared, not copied


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
def test_scorer_readings_come_from_capture(fmt):
    run = _run(fmt)
    for frames, _ in _data():
        run.run_sample(frames)
    ctx = run.ctx
    assert Scorer().weight_health(ctx, w_min=-6.0, w_max=6.0) == capture.pinned_weights(ctx, w_min=-6.0, w_max=6.0)
    ##no bounds + no eps -> nothing pinned; eps=0 -> every realized synapse is either above 0 or exactly 0
    inf = float("inf")
    assert capture.pinned_weights(ctx, w_min=-inf, w_max=inf)["saturated_frac"] == 0.0
    assert capture.pinned_weights(ctx, w_min=None, w_max=None) == capture.pinned_weights(ctx, w_min=-inf, w_max=inf)   ##None = no clamp (what Run passes)
    p0 = capture.pinned_weights(ctx, w_min=-inf, w_max=inf, eps=0.0)
    assert p0["saturated_frac"] + p0["dead_syn_frac"] == pytest.approx(1.0)
    assert capture.pinned_weights(ctx, w_min=-inf, w_max=inf, eps=p0["w_absmax"])["saturated_frac"] == 0.0
    probes = Scorer().health(ctx, steps=100, spike_count=50, dt=1.0)
    assert probes["rate_hz"] == capture.mean_rate_hz(ctx, spike_count=50, steps=100, dt=1.0) == 50 / 0.1 / ctx.net.n_rec
    counts = capture.spike_counts(ctx, run.history, since_event=0)
    assert counts.shape == (ctx.net.n_total,) and counts.sum() == run.history._n_events
    assert np.array_equal(capture.spike_counts(ctx, run.history, since_event=0, group="exc"), counts[ctx.net.neuronslices["exc"]])


def test_rate_hz_is_spikes_per_neuron_per_second():
    """capture.rate_hz = sum(counts) / n_neurons / (steps * dt / 1000 s), for scalars, any array shape and zero denominators."""
    counts = np.array([10.0, 20.0, 30.0])
    assert capture.rate_hz(counts, 3, 1000, 1.0) == pytest.approx(20.0)            ##60 spikes / 3 neurons / 1 s
    assert capture.rate_hz(counts, 3, 1000, 0.5) == pytest.approx(40.0)            ##0.5 s
    assert capture.rate_hz(60.0, 3, 1000, 1.0) == capture.rate_hz(counts, 3, 1000, 1.0) == capture.rate_hz(counts.reshape(3, 1), 3, 1000, 1.0)
    assert capture.rate_hz(np.zeros(3), 3, 10, 1.0) == 0.0 and np.isfinite(capture.rate_hz(counts, 0, 0, 1.0))
    assert type(capture.rate_hz(counts, 3, 1000, 1.0)) is float
    ##on a run: the rate of one presentation from its count vector, its T steps and history's dt
    from operations import Operations
    run = _run("dense"); frames = _data()[0][0]; T = frames.shape[0]
    c = Operations(run).present(frames)
    assert capture.rate_hz(c, run.ctx.net.n_total, T, run.history.dt) == pytest.approx(c.sum() / run.ctx.net.n_total / (T * run.history.dt / 1000.0))
