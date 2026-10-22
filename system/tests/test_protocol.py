#test_protocol -- the training loop lives in the protocol (protocols.HoldoutValidation) and a Run has a mode ("train" / "eval") instead of a `learn` flag.
# Silent failure modes asserted here: a pass that trains in "eval" (inflates every test score), a bookmark that counts the wrong neurons, a snapshot
# that misses the protocol's state. The per-step primitives (Run.set_mode, run_sample(until=) + finish_sample, history.Counts) are tested in every mode.
# Run: pytest system/tests/test_protocol.py -q     (from the repo root)
from __future__ import annotations
##sys.path is set by system/tests/conftest.py
import dataclasses
import numpy as np
from modelinput import ModelInput
import pytest

from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from protocols import Protocol, HoldoutValidation
from checkpoint import save_snapshot, load_snapshot, take_snapshot
import test_sparse as S          ##its assemble(fmt=, update=) builds the same network in every execution mode

MODES = [("dense", "timestep"), ("sparse", "timestep"), ("sparse", "event"), ("sparse", "hybrid_event")]


def _traces(run):
    """The plasticity's traces in whichever layout the mode keeps (per neuron, or per synapse under "event")."""
    pl = run.plasticity
    return (pl.Pre_trace, pl.Post_trace) if pl.Pre_trace is not None else (pl.Pre_e, pl.Post_e)


def _run(fmt="dense", update="timestep", **rc):
    """A Run in one execution mode. rc = RunCfg overrides (S.assemble fixes epochs; the rest is replaced on the Run's cfg)."""
    pnet, run = S.assemble(fmt=fmt, update=update)
    if rc:
        run.runcfg = dataclasses.replace(run.runcfg, **rc)
    return run


def _dc_run(**rc):
    """The D&C model (DCAdaptiveThreshold.persistent = ("Theta",)): the one stock module with a persistent array besides W."""
    from diehl_cook_2015 import build, dc_append_rest
    n_in, n_exc = 32, 6
    rng = np.random.default_rng(0)
    data = ModelInput.from_tonic([(dc_append_rest(rng.random((40, n_in)) < 0.3, rest_ms=10.0), i % 2) for i in range(12)])
    cfg = dict(data=DataCfg(test_fraction=0.25), checkpoint=CheckpointCfg(to_disk=False), scorer=ScorerCfg(rate_hz_min=0.0),
               history=HistoryCfg(include_input=False)); cfg.update(rc)
    return build(n_in, n_exc, norm_target=78.0 * n_in / 784)._prep_run(data, RunCfg(**cfg))


# --------------------------------------------------------------------- 1. the mode ("train" / "eval")

@pytest.mark.parametrize("fmt,update", MODES)
def test_mode_train_learns_and_mode_eval_does_not(fmt, update):
    """A Run starts in "train" (W moves); "eval" is forward-only (W and the raster still move, W does not); back to "train" learns again."""
    run = _run(fmt, update)
    frames = S._data()[0][0]
    assert run.ctx.sim.mode == "train" and run._learning
    W0 = run.ctx.syn.W.copy()
    run.run_sample(frames)
    assert not np.array_equal(W0, run.ctx.syn.W), "training did not change W, so the eval check below proves nothing"

    run.set_mode("eval")
    assert run.ctx.sim.mode == "eval" and not run._learning
    W1, c1 = run.ctx.syn.W.copy(), run.history.Counts.copy()
    run.run_sample(frames)
    assert np.array_equal(W1, run.ctx.syn.W), "mode 'eval' trained"
    assert (run.history.Counts - c1)[run.ctx.net.n_input:].sum() > 0, "nothing spiked in eval, so the forward pass is not shown to run"
    run.check_persistent_unchanged()                  ##passes: nothing persistent moved
    run.set_mode("train")                            ##the guard is checked here too, and dropped
    assert run._eval_guard is None
    run.run_sample(frames)
    assert not np.array_equal(W1, run.ctx.syn.W), "back in 'train' the Run did not learn"
    ##the switches are on the timeline (0 train / 1 eval = MODE_KEY), and only the switches: the same mode twice marks nothing
    segs = run.history.mode_timeline.segments()
    run.set_mode("train")
    assert run.history.mode_timeline.segments() == segs and [s["value"] for s in segs][-2:] == [1, 0]
    assert segs[0] == {"start_step": 0, "value": 0} and run.history.mode_at(0) == 0      ##the initial "train" segment is recorded
    ##the 0 / 1 are MODE_KEY; on disk the timeline is "mode" (mode_starts / mode_values, mode_segments)
    from history import MODE_KEY
    assert MODE_KEY == {"train": 0, "eval": 1} and [s["value"] for s in segs] == [MODE_KEY[m] for m in ("train", "eval", "train")]
    arrays, meta = run.history.arrays(), run.history.metadata()
    assert "mode_starts" in arrays and "mode_values" in arrays and "stage_starts" not in arrays and "mode_segments" in meta and "stage_segments" not in meta
    assert run.ctx.sim.mode == "train" and not hasattr(run.ctx.sim, "stage") and not hasattr(run, "set_stage")


def test_history_npz_loader_accepts_the_old_stage_keys(tmp_path):
    """_read_history_npz reads old stage_* keys as mode_* (COMPAT hook; delete with it); a new file round-trips under the new names."""
    from utils import _write_history_npz, _read_history_npz
    _write_history_npz(tmp_path / "old", {"spike_ptr": np.zeros(1, np.uint64), "spike_i": np.zeros(0, np.uint32),
                                           "stage_starts": np.array([0, 7], np.uint64), "stage_values": np.array([0, 1], np.int64)})
    back = _read_history_npz(tmp_path / "old")
    assert "stage_starts" not in back and back["mode_starts"].tolist() == [0, 7] and back["mode_values"].tolist() == [0, 1]
    run = _run(); run.run_sample(S._data()[0][0]); run.set_mode("eval")
    run.history.save(filename="new", export_dir=tmp_path)
    new = _read_history_npz(tmp_path / "new")
    assert new["mode_values"].tolist() == [0, 1] and new["mode_starts"][1] == run.ctx.sim.current_runstep


def test_set_mode_rejects_unknown_modes_and_released_training():
    run = _run()
    with pytest.raises(ValueError, match="mode must be"):
        run.set_mode("test")
    run.release_training_state()
    assert run.ctx.sim.mode == "eval" and run._eval_guard is None       ##released = eval-only, unguarded (nothing to compare against)
    assert run._eval_only and not hasattr(run, "_released")
    with pytest.raises(RuntimeError):
        run.set_mode("train")
    run.set_mode("eval")                                                 ##a no-op, allowed


# --------------------------------------------------------------------- 2. the eval contract (learned arrays are guarded)

def test_eval_guard_names_a_learned_array_that_moved():
    """D&C's Theta adapts unless frozen: an unfrozen threshold in "eval" is caught when the mode ends, by name. Frozen, the same pass is clean."""
    run = _dc_run()
    frames = run.train_set[0][0]
    run.run_sample(frames)                            ##train once so there is activity to adapt on
    run.set_mode("eval")
    run.run_sample(frames)
    with pytest.raises(RuntimeError, match="threshold.Theta"):
        run.set_mode("train")
    ##the guard is a check at the boundary, not a rollback: the mode is still eval, and freezing + re-entering eval gives a clean pass
    assert run.ctx.sim.mode == "eval"
    run._eval_guard = None; run.ctx.sim.mode = "train"; run._learning = True     ##leave the failed eval by hand (the test's own recovery)
    run.threshold.freeze(True)
    run.set_mode("eval"); run.run_sample(frames); run.check_persistent_unchanged(); run.set_mode("train")


@pytest.mark.parametrize("fmt,update", MODES)
def test_eval_guard_catches_a_write_to_W(fmt, update):
    run = _run(fmt, update)
    run.set_mode("eval")
    run.ctx.syn.W.flat[0] += 1.0
    with pytest.raises(RuntimeError, match="mode 'eval' changed a persistent array: W"):
        run.check_persistent_unchanged()


# --------------------------------------------------------------------- 2b. phase types

class _Prune(S.DefaultPhase):
    """A phase declared under a NON-normalised type."""
    phase_type = " prune "


def test_phase_types_are_normalised_on_both_sides_and_the_strict_check_stays():
    """Phase.__init__ normalises the type on both sides (class and cfg), the scheduler keys by it, and a cfg of another type is still refused."""
    from cfg import PhaseCfg
    from dynamics import Phase
    assert isinstance(_Prune(PhaseCfg(phase_type="Prune")), Phase)                  ##both sides normalised -> PRUNE == PRUNE
    with pytest.raises(ValueError, match="expected  prune , got DEFAULT"):
        _Prune(PhaseCfg(phase_type="DEFAULT"))
    pnet, run = S.assemble(fmt="dense")
    pnet.cfg = dataclasses.replace(pnet.cfg, phases=(PhaseCfg(phase_type="DEFAULT"), PhaseCfg(phase_type="prune")))
    pnet.set_runtime_classes(phase_classes=(S.DefaultPhase, _Prune))
    run = pnet._prep_run(S._data(), run.runcfg)
    assert set(run.phases) == {"DEFAULT", "PRUNE"} and set(run.ctx.sim.time_in_phase) == {"DEFAULT", "PRUNE"}
    sched = run._phasescheduler
    sched.switch_phase(run.phases["DEFAULT"], run.phases["PRUNE"], run.ctx)
    assert run.ctx.sim.current_phase == "PRUNE" and run.ctx.sim.time_in_current_phase == 0    ##normalised
    frames = S._data()[0][0]
    run.run_sample(frames)                                                            ##the step looks the phase up by sim.current_phase; the scheduler compares by the dict key
    assert run.ctx.sim.time_in_phase["PRUNE"] == frames.shape[0] and run.history.phase_at(run.ctx.sim.current_runstep - 1) == "PRUNE"
    assert not sched.step(run.ctx)                                                    ##no triggers anywhere: no switch, and no exception from the raw name


# --------------------------------------------------------------------- 2c. phases + triggers + the run loop
import logging
import capture
from cfg import PhaseCfg, SleepPhaseCfg
from phases import SleepPhase


class _Pause(S.DefaultPhase):
    """A second phase with nothing of its own: the trigger / exit / dataloader_active behaviour under test is the cfg's."""
    phase_type = "PAUSE"


def _phased(fmt="dense", phases=(), classes=(), **rc):
    """The S.assemble Run with extra phase cfgs / classes (DEFAULT stays first); rc = RunCfg overrides."""
    pnet, run = S.assemble(fmt=fmt)
    pnet.cfg = dataclasses.replace(pnet.cfg, phases=(PhaseCfg(phase_type="DEFAULT"), *phases))
    pnet.set_runtime_classes(phase_classes=(S.DefaultPhase, *classes))
    return pnet, pnet._prep_run(S._data(), dataclasses.replace(run.runcfg, **rc) if rc else run.runcfg)


def _group_w(run, group):
    """One projection's realized weights, in either format."""
    W, p = run.ctx.syn.W, run.ctx.net.synapseslices[group]
    return W[p.post, p.pre][run.ctx.syn.syn_mask[p.post, p.pre]] if W.ndim == 2 else W[capture._syn_block(run.ctx, p.post, p.pre)]


def test_time_in_currentphase_is_the_time_since_the_switch():
    """capture.time_in_currentphase is the time since the last switch; sum_timeinphase stays cumulative."""
    pnet, run = _phased(phases=(PhaseCfg(phase_type="PAUSE"),), classes=(_Pause,))
    frames = S._data()[0][0]; T = frames.shape[0]
    run.run_sample(frames)
    assert capture.time_in_currentphase(run.ctx) == T == capture.sum_timeinphase(run.ctx)
    assert run._phasescheduler.switch_phase(run.phases["DEFAULT"], run.phases["PAUSE"], run.ctx) == "PAUSE"     ##switch_phase returns the normalised name
    assert capture.time_in_currentphase(run.ctx) == 0 and capture.sum_timeinphase(run.ctx, "DEFAULT") == T
    run.run_sample(frames)
    assert capture.time_in_currentphase(run.ctx) == T == capture.sum_timeinphase(run.ctx) and capture.sum_timeinphase(run.ctx, "DEFAULT") == T


def test_exit_triggers_send_the_phase_back_to_default():
    """The current phase's exit_triggers fire first and send it back to DEFAULT; step() returns the name switched into, else None."""
    pause = PhaseCfg(phase_type="PAUSE", triggers={"time_in_currentphase": (5, ">=")}, exit_triggers={"time_in_currentphase": (3, ">=")})
    pnet, run = _phased(phases=(pause,), classes=(_Pause,))
    frames = S._data()[0][0]; T = frames.shape[0]
    sched, switches = run._phasescheduler, []
    orig = sched.step
    def spy(ctx):
        r = orig(ctx); switches.append(r); return r
    sched.step = spy
    assert run.run_sample(frames) == T
    assert switches[:13] == [None] * 4 + ["PAUSE"] + [None] * 2 + ["DEFAULT"] + [None] * 4 + ["PAUSE"]
    assert run.ctx.sim.time_in_phase["PAUSE"] + run.ctx.sim.time_in_phase["DEFAULT"] == T
    assert run.history.phase_at(4) == "DEFAULT" and run.history.phase_at(5) == "PAUSE" and run.history.phase_at(8) == "DEFAULT" and run.history.phase_at(13) == "PAUSE"
    ##the exit trigger is the phase's own: DEFAULT has none (None), and a phase without one stays until another phase's entry trigger fires
    assert run.phases["DEFAULT"].exit_triggers is None and sched.step(run.ctx) is None


def test_a_long_dataloader_off_phase_is_not_a_stuck_run(caplog):
    """A phase holding the dataloader longer than 10 x T just stretches the sample (no cap); the scheduler logs the switch with dataloader_active."""
    frames = S._data()[0][0]; T = frames.shape[0]
    pause = PhaseCfg(phase_type="PAUSE", triggers={"time_in_currentphase": (5, ">=")}, exit_triggers={"time_in_currentphase": (10 * T + 5, ">=")},
                     dataloader_active=False)
    pnet, run = _phased(phases=(pause,), classes=(_Pause,))
    with caplog.at_level(logging.INFO, logger="phasescheduler"):
        executed = run.run_sample(frames)
    assert executed > 10 * T and run.ctx.sim.step_in_sample == T               ##the sample finished
    assert run.ctx.sim.time_in_phase["DEFAULT"] == T                            ##the dataloader advanced in DEFAULT only: T data steps, the rest was the pause
    assert run.ctx.sim.time_in_phase["PAUSE"] == executed - T
    lines = [r.getMessage() for r in caplog.records if r.name == "phasescheduler"]
    assert any("from DEFAULT to PAUSE" in m and "dataloader_active=False" in m for m in lines)
    assert any("from PAUSE to DEFAULT" in m and "dataloader_active=True" in m for m in lines)
    import inspect
    from run import Run
    assert "cap" not in inspect.signature(Run._run_to).parameters


def test_phase_timeline_is_recorded_without_the_raster():
    """The phase and epoch timelines are marked even with record_spikes=False."""
    pause = PhaseCfg(phase_type="PAUSE", triggers={"time_in_currentphase": (5, ">=")}, exit_triggers={"time_in_currentphase": (3, ">=")})
    pnet, run = _phased(phases=(pause,), classes=(_Pause,), history=HistoryCfg(record_spikes=False))
    frames = S._data()[0][0]
    run.run_sample(frames)
    h = run.history
    assert not h.cfg.record_spikes and h.n_steps == 0                                             ##no raster
    assert h.phase_at(0) == "DEFAULT" and h.phase_at(5) == "PAUSE" and h.phase_at(8) == "DEFAULT"   ##the phase timeline is there
    assert [s["value"] for s in h.phase_timeline.segments()][:4] == ["DEFAULT", "PAUSE", "DEFAULT", "PAUSE"]
    assert h.epoch_timeline.segments() == [{"start_step": 0, "value": 0}]                          ##and the epoch's


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
def test_transition_write_is_signed_and_clamped_after_the_switch(fmt):
    """SleepPhase writes a SIGNED revival jolt at the switch (both formats) and Run._run_to clamps right after it: a jolt past w_max lands on w_max."""
    sleep = SleepPhaseCfg(triggers={"time_in_currentphase": (3, ">=")}, sleep_fraction=1.0, jolt_magnitude=100.0, revive_syn_group="in_exc", dead_eps=2.0)
    pnet, run = _phased(fmt=fmt, phases=(sleep,), classes=(SleepPhase,))
    frames = S._data()[0][0]
    assert SleepPhase.transition_writes == ("W",) and SleepPhase.weights_format is None and S.DefaultPhase.transition_writes == ()
    assert np.all(_group_w(run, "in_exc") < 2.0)                                     ##every in_exc synapse counts as deadened at eps=2.0
    assert run.run_sample(frames, until=3) == 3 and run.ctx.sim.current_phase == "SLEEP"     ##the switch at the end of step 2; nothing ran after it
    w = _group_w(run, "in_exc")
    assert np.all(w == run.plasticity.w_max), "the jolt was not clamped after the switch"       ##1.2 + 100 -> w_max = 10.0
    ##the sign: an inhibitory pre group's weights move AWAY from zero
    sleep = SleepPhaseCfg(triggers={"time_in_currentphase": (3, ">=")}, sleep_fraction=1.0, jolt_magnitude=1.0, revive_syn_group="inh_exc", dead_eps=3.0)
    pnet, run = _phased(fmt=fmt, phases=(sleep,), classes=(SleepPhase,))
    before = _group_w(run, "inh_exc").copy()
    assert np.all(before < 0)
    run.run_sample(frames, until=3)
    assert np.allclose(_group_w(run, "inh_exc"), before - 1.0, atol=1e-6)


def test_sparse_revival_matches_dense():
    """Under sparse the revival runs on the edge list: same phase rng and dead set -> the same synapses revived -> the same W as dense."""
    sleep = SleepPhaseCfg(sleep_fraction=0.5, jolt_magnitude=0.7, revive_syn_group="in_exc", dead_eps=1.3)
    runs = {fmt: _phased(fmt=fmt, phases=(sleep,), classes=(SleepPhase,))[1] for fmt in ("dense", "sparse")}
    for run in runs.values():
        run._phasescheduler.switch_phase(run.phases["DEFAULT"], run.phases["SLEEP"], run.ctx)
    d, s = runs["dense"], runs["sparse"]
    assert np.array_equal(S.scatter(s.ctx.syn, d.ctx.syn.W.shape), d.ctx.syn.W)
    changed = int(np.count_nonzero(d.ctx.syn.W != d._W0))
    assert changed == int(np.ceil(0.5 * d.ctx.net.synapse_counts["n_in_exc"])) > 0           ##a fraction of the group, nothing else
    assert int(np.count_nonzero(s.ctx.syn.W != s._W0)) == changed


def test_eval_refuses_a_switch_into_a_phase_with_transition_writes():
    """In "eval" a switch into a phase with transition_writes is refused before on_enter; other phases switch as usual, and "train" allows it."""
    sleep = SleepPhaseCfg(revive_syn_group="in_exc", dead_eps=1.3, sleep_fraction=1.0)
    pnet, run = _phased(phases=(sleep, PhaseCfg(phase_type="PAUSE")), classes=(SleepPhase, _Pause))
    sched = run._phasescheduler
    run.set_mode("eval")
    with pytest.raises(RuntimeError, match="'SLEEP' in mode 'eval'"):
        sched.switch_phase(run.phases["DEFAULT"], run.phases["SLEEP"], run.ctx)
    assert run.ctx.sim.current_phase == "DEFAULT" and np.array_equal(run.ctx.syn.W, run._W0)    ##refused before on_enter: nothing written, no switch
    assert sched.switch_phase(run.phases["DEFAULT"], run.phases["PAUSE"], run.ctx) == "PAUSE"
    run.set_mode("train")                                                                      ##the eval contract holds (nothing changed)
    assert sched.switch_phase(run.phases["PAUSE"], run.phases["SLEEP"], run.ctx) == "SLEEP" and not np.array_equal(run.ctx.syn.W, run._W0)
    ##and through the trigger path inside a sample
    sleep = SleepPhaseCfg(triggers={"time_in_currentphase": (3, ">=")}, revive_syn_group="in_exc", dead_eps=1.3)
    pnet, run = _phased(phases=(sleep,), classes=(SleepPhase,))
    run.set_mode("eval")
    with pytest.raises(RuntimeError, match="transition_writes=\\('W',\\) \\(a write to a persistent array"):
        run.run_sample(S._data()[0][0])


def test_trigger_typos_fail_at_build():
    """_instantiate_phases rejects bad triggers at build: unknown capture, comparison or summary, a bad tuple, a capture that is not f(ctx)."""
    pnet, _ = S.assemble(fmt="dense")
    pnet.set_runtime_classes(phase_classes=(S.DefaultPhase, _Pause))
    bad = [({"sum_globalweigth": (1.0, ">")}, "not a capture function"),
           ({"sum_globalweight": (1.0, "=>")}, "unknown comparison '=>'"),
           ({"in_degree": (0, ">", "count", 1, ">")}, "unknown summary 'count'"),
           ({"in_degree": (0, ">", "sum")}, "must be \\(value, comparison\\)"),
           ({"in_degree": (0, ">", "sum", 1, "=>")}, "unknown comparison '=>'"),
           ({"sum_groupweight": (1.0, ">")}, "not trigger-usable: .*needs \\('syn_group',\\) too"),     ##f(ctx) only: a capture needing more
           ({"pinned_weights": (1.0, ">")}, "needs \\('w_min', 'w_max'\\) too"),
           ({"RunContext": (1.0, ">")}, "not trigger-usable: .*not a capture function of ctx")]          ##a callable in capture's namespace that is no capture
    for triggers, msg in bad:
        for key in ("triggers", "exit_triggers"):
            pnet.cfg = dataclasses.replace(pnet.cfg, phases=(PhaseCfg(phase_type="DEFAULT"), PhaseCfg(phase_type="PAUSE", **{key: triggers})))
            with pytest.raises(ValueError, match=f"phase PAUSE {key}: .*{msg}"):
                pnet._instantiate_phases()
    pnet.cfg = dataclasses.replace(pnet.cfg, phases=(PhaseCfg(phase_type="DEFAULT"),
                                                     PhaseCfg(phase_type="PAUSE", triggers={"sum_globalweight": (1.0, ">")}, exit_triggers={"in_degree": (0, ">", "sum", 1, ">=")})))
    assert set(pnet._instantiate_phases()) == {"DEFAULT", "PAUSE"}


def test_triggers_compare_by_output_type_and_may_be_two_step():
    """Triggers compare by the capture's output type (number, ndarray, tuple); a sequence needs the two-step form (summary vs value2), else a named error."""
    pnet, run = S.assemble(fmt="dense")
    sched, ctx = run._phasescheduler, run.ctx
    check = lambda triggers, mode="all": sched.check_triggers(ctx, triggers, mode=mode)
    ##one step on a number
    assert check({"sum_globalweight": (-1e9, ">")}) is True and check({"time_in_currentphase": (0, ">")}) is False
    assert check({"sum_globalweight": (-1e9, ">"), "time_in_currentphase": (0, ">")}, mode="all") is False
    assert check({"sum_globalweight": (-1e9, ">"), "time_in_currentphase": (0, ">")}, mode="any") is True
    ##one step on an array: not one bool -> the named error
    with pytest.raises(ValueError, match="trigger 'in_degree' compares \\d+ values, not one: add a summary stage"):
        check({"in_degree": (0, ">")})
    ##two steps: elementwise, then the summary against value2
    deg = capture.in_degree(ctx)
    assert check({"in_degree": (0, ">", "sum", int((deg > 0).sum()), "==")}) is True and check({"in_degree": (0, ">", "sum", deg.size + 1, ">=")}) is False
    assert check({"in_degree": (0, ">", "any", True, "==")}) is True and check({"in_degree": (0, ">", "all", True, "==")}) is False      ##the input rows receive nothing
    assert check({"in_degree": (0, ">", "mean", float((deg > 0).mean()), "==")}) is True
    ##a tuple output (deadened_synapses under dense: (post, pre)): a tuple value of the same length, one sequence, then the summary
    post, pre = capture.deadened_synapses(ctx, syn_group="in_exc", eps=1.3)
    ctx.syn.W[post[:3], pre[:3]] = 0.0                                                         ##three dead synapses at the default eps=0.0
    assert check({"deadened_synapses": ((-1, -1), ">", "sum", 6, "==")}) is True                 ##3 posts > -1 and 3 pres > -1
    assert check({"deadened_synapses": ((-1, -1), ">", "sum", 0, "==")}) is False
    with pytest.raises(ValueError, match="returns a tuple of 2, so its value must be a tuple of 2"):
        check({"deadened_synapses": (0, ">", "sum", 0, "==")})
    ##an output that is none of the three
    with pytest.raises(TypeError, match="returned a dict"):
        check({"raster_meta": (0, ">")})


def test_a_phase_declaring_weights_format_joins_the_wiring_check():
    """A phase declaring weights_format joins the build-time format check: a dense-only phase under a sparse cfg is refused by name."""
    class _DenseOnly(S.DefaultPhase):
        phase_type = "DENSEONLY"
        weights_format = "dense"
    for fmt in ("sparse", "dense"):
        pnet, _ = S.assemble(fmt=fmt)
        pnet.cfg = dataclasses.replace(pnet.cfg, phases=(PhaseCfg(phase_type="DEFAULT"), PhaseCfg(phase_type="DENSEONLY")))
        pnet.set_runtime_classes(phase_classes=(S.DefaultPhase, _DenseOnly))
        if fmt == "sparse":
            with pytest.raises(TypeError, match="_DenseOnly is a 'dense' module but SimulationCfg.weights_format='sparse'"):
                pnet._instantiate_phases()
        else:
            assert "DENSEONLY" in pnet._instantiate_phases()


# --------------------------------------------------------------------- 3. reset_transient: a new network with the same model

def test_reset_transient_keeps_W_and_theta_and_zeroes_the_rest():
    run = _dc_run()
    for frames, _ in list(run.train_set)[:3]:
        run.run_sample(frames)
    th, pl = run.threshold, run.plasticity
    assert not np.array_equal(th.Theta, run._init_state["threshold"]["Theta"]), "theta did not adapt, so keeping it proves nothing"
    assert pl.Pre.any() and not np.allclose(run.ctx.neur.V, run.integrator.v_rest), "no transient state to reset"
    W, Theta = run.ctx.syn.W.copy(), th.Theta.copy()
    run.reset_transient()
    assert np.array_equal(W, run.ctx.syn.W) and np.array_equal(Theta, th.Theta)         ##the MODEL stays
    assert not pl.Pre.any() and not pl.Post1.any() and not run.ctx.neur.Spikes.any()     ##the transient goes
    assert np.allclose(run.ctx.neur.V, run.integrator.v_rest)
    assert np.array_equal(run.synapse.Ge, run._init_state["synapse"]["Ge"])
    ##and reset_model is the superset: W back to as-built as well
    run.reset_model()
    assert np.array_equal(run.ctx.syn.W, run._W0)


@pytest.mark.parametrize("fmt,update", MODES)
def test_reset_transient_in_every_mode(fmt, update):
    run = _run(fmt, update)
    run.run_sample(S._data()[0][0])
    W = run.ctx.syn.W.copy()
    assert _traces(run)[0].any()
    run.reset_transient()
    assert np.array_equal(W, run.ctx.syn.W) and not _traces(run)[0].any() and not _traces(run)[1].any()
    if run.ctx.upd.dW is not None:
        assert not run.ctx.upd.dW.any()


# --------------------------------------------------------------------- 4. history.Counts: the per-neuron count is the raster, summed

@pytest.mark.parametrize("fmt,update", MODES)
def test_counts_equal_the_raster_bincount(fmt, update):
    run = _run(fmt, update)
    for frames, _ in S._data().subset(np.arange(4)):
        run.run_sample(frames)
    h, n_in, n_total = run.history, run.ctx.net.n_input, run.ctx.net.n_total
    rec = np.bincount(h.spike_i(), minlength=n_total)
    assert h.Counts[n_in:].sum() > 0 and np.array_equal(h.Counts[n_in:], rec[n_in:])      ##the raster excludes input nodes (include_input=False); Counts holds every neuron
    assert h.Counts[:n_in].sum() > 0, "input spikes are counted too (a protocol may read them)"
    ##mark/since are the bookmark pair a protocol uses: since(mark) == the spikes recorded after it, per neuron
    m = h.mark()
    run.run_sample(S._data()[4][0])
    assert np.array_equal(h.since(m), h.Counts - m) and h.since(m)[n_in:].sum() > 0
    out = np.empty_like(m); assert h.since(m, out=out) is out


def test_record_counts_false_disables_the_add_and_forbids_protocols():
    run = S.assemble()[0]._prep_run(S._data(), RunCfg(history=HistoryCfg(record_counts=False), data=DataCfg(test_fraction=0.25)))
    assert not run._count_spikes
    run.run_sample(S._data()[0][0])
    assert not run.history.Counts.any()
    with pytest.raises(RuntimeError):
        HoldoutValidation(None, run)


# --------------------------------------------------------------------- 5. until + finish_sample == one run_sample

@pytest.mark.parametrize("fmt,update", MODES)
def test_until_and_finish_sample_reproduce_one_run_sample(fmt, update):
    """The boundary a protocol bookmarks (D&C: present/rest) must not change what runs: same W, V, counts, and T steps in total."""
    a, b = _run(fmt, update), _run(fmt, update)
    frames = S._data()[0][0]; T = frames.shape[0]
    na = a.run_sample(frames)
    nb1 = b.run_sample(frames, until=T // 3); nb2 = b.finish_sample()
    assert na == T and nb1 == T // 3 and nb2 == T - T // 3
    assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W) and np.array_equal(a.ctx.neur.V, b.ctx.neur.V)
    assert np.array_equal(a.history.Counts, b.history.Counts) and a.ctx.sim.current_runstep == b.ctx.sim.current_runstep
    ##until past the end clips to the sample; finish_sample on a finished sample runs nothing
    assert b.run_sample(frames, until=10 * T) == T and b.finish_sample() == 0


@pytest.mark.parametrize("fmt,update", MODES)
def test_continue_sample_walks_a_sample_in_segments(fmt, update):
    """run_sample(until=) + continue_sample(until=)... + finish_sample() == one run_sample; until is clipped, a bound already passed runs nothing."""
    a, b = _run(fmt, update), _run(fmt, update)
    frames = S._data()[0][0]; T = frames.shape[0]
    assert a.run_sample(frames) == T
    n = [b.run_sample(frames, until=T // 4), b.continue_sample(T // 4), b.continue_sample(T // 2), b.continue_sample(T // 3), b.continue_sample(10 * T), b.finish_sample()]
    assert n == [T // 4, 0, T // 2 - T // 4, 0, T - T // 2, 0] and b.ctx.sim.step_in_sample == T
    assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W) and np.array_equal(a.ctx.neur.V, b.ctx.neur.V)
    assert np.array_equal(a.history.Counts, b.history.Counts) and a.ctx.sim.current_runstep == b.ctx.sim.current_runstep


# --------------------------------------------------------------------- 6. Protocol: selection by name, cfg loading, hashing, the shared blocks

def test_protocol_is_picked_by_name_and_unknown_names_fail():
    from phasicnet import PhasicNet
    from dc_protocol import DCProtocol, DCProtocolCfg
    pnet, run = S.assemble()
    assert type(pnet._instantiate_protocol(RunCfg(), run)) is HoldoutValidation                                ##the default
    assert type(pnet._instantiate_protocol(RunCfg(protocol=ProtocolCfg(protocol_type="holdoutval")), run)) is HoldoutValidation  ##by name, case-insensitive
    with pytest.raises(RuntimeError, match="DC2015"):
        pnet._instantiate_protocol(RunCfg(protocol=DCProtocolCfg(n_train=4)), run)                             ##not registered on this PhasicNet
    with pytest.raises(RuntimeError):
        pnet._instantiate_protocol(RunCfg(protocol=ProtocolCfg("nosuch")), run)
    ##a protocol cfg subclass round-trips through RunCfg.load by its protocol_type (same rule as PhaseCfg)
    rc = RunCfg(protocol=DCProtocolCfg(n_train=4, n_assign=3, test_passes=2))
    back = RunCfg.load(dataclasses.asdict(rc))
    assert isinstance(back.protocol, DCProtocolCfg) and back.protocol == rc.protocol and back.hashable() == rc.hashable()
    assert RunCfg.load(dataclasses.asdict(RunCfg())).protocol == ProtocolCfg()      ##the default is a ProtocolCfg() (default_factory), not None


def test_run_id_ignores_io_fields_and_treats_the_default_protocol_as_none():
    base = RunCfg().hashable()
    assert "protocol" not in base, "the default protocol must be hash-neutral: run_ids from before the protocol round stay the same"
    assert RunCfg(protocol=None).hashable() == base                                            ##None == the default protocol, spelled out or not
    assert RunCfg(checkpoint=CheckpointCfg(every=5)).hashable() == base                        ##every = IO only
    assert RunCfg(checkpoint=CheckpointCfg(to_disk=False)).hashable() == base
    assert RunCfg(protocol=ProtocolCfg(reset_between_samples=True)).hashable() != base         ##a protocol setting that changes the run changes the id
    from dc_protocol import DCProtocolCfg
    assert RunCfg(protocol=DCProtocolCfg(report_every=5)).hashable() == RunCfg(protocol=DCProtocolCfg(report_every=9)).hashable()   ##report_every = logging only
    assert RunCfg(protocol=DCProtocolCfg(n_train=5)).hashable() != RunCfg(protocol=DCProtocolCfg(n_train=9)).hashable()


@pytest.mark.parametrize("fmt,update", MODES)
def test_reset_between_samples_is_applied_before_every_sample(fmt, update):
    run = _run(fmt, update)
    data = S._data().subset(np.arange(5))
    calls, orig = [], run.reset_transient
    run.reset_transient = lambda: (calls.append(1), orig())[1]
    HoldoutValidation(ProtocolCfg(reset_between_samples=True), run).ops.run_epoch(data)
    assert len(calls) == len(data)
    calls.clear()
    HoldoutValidation(None, run).ops.run_epoch(data)
    assert not calls
    ##run.test() uses the run's own protocol setting
    run.runcfg = dataclasses.replace(run.runcfg, protocol=ProtocolCfg(reset_between_samples=True))
    run.test()
    assert len(calls) == len(run.test_set)
    run.reset_transient = orig


def test_run_pass_spike_counts_keep_their_meaning():
    """summary["spikes"] counts every recorded spike (input only with include_input), summary["recurrent_spikes"] the non-input ones."""
    run = _run()
    n_in = run.ctx.net.n_input
    proto = HoldoutValidation(None, run)
    m = run.history.mark()
    s = proto.ops.run_epoch(S._data().subset(np.arange(3)))
    rec = run.history.since(m)
    assert s["samples"] == 3 and s["recurrent_spikes"] == int(rec[n_in:].sum()) and s["recurrent_spikes"] > 0
    assert s["spikes"] == s["recurrent_spikes"]                        ##include_input=False here
    assert rec[:n_in].sum() > 0, "no input spikes: the include_input case below would be vacuous"
    run2 = S.assemble()[0]._prep_run(S._data(), RunCfg(history=HistoryCfg(include_input=True), data=DataCfg(test_fraction=0.25)))
    m2 = run2.history.mark(); s2 = HoldoutValidation(None, run2).ops.run_epoch(S._data().subset(np.arange(3))); rec2 = run2.history.since(m2)
    assert s2["spikes"] == int(rec2.sum()) and s2["recurrent_spikes"] == int(rec2[n_in:].sum()) and s2["spikes"] > s2["recurrent_spikes"]


@pytest.mark.parametrize("fmt,update", MODES)
def test_holdoutval_execute_is_the_whole_run(fmt, update):
    """PhasicNet.train with ProtocolCfg(val_split=) runs HoldoutValidation.execute: one fold score, a report naming the protocol and cfg; test() is separate."""
    pnet, _ = S.assemble(fmt=fmt, update=update)
    cv = ProtocolCfg(val_split=0.25)
    run = pnet.train(S._data(), RunCfg(protocol=cv, scorer=ScorerCfg(rate_hz_min=0.0), data=DataCfg(test_fraction=0.25)), save_run=False)
    r = run.report
    assert len(r["fold_scores"]) == 1 and "refit" not in r and r["protocol"] == "HoldoutValidation"
    assert r["protocol_cfg"] == dataclasses.asdict(cv) and r["runcfg"]["protocol"] == dataclasses.asdict(cv)
    assert run.ctx.sim.mode == "train"                       ##a finished fit ends on the training side (finalize writes W there); test() was NOT run
    t = run.test()
    assert t["n_test"] > 0 and run.ctx.sim.mode == "eval"    ##test() leaves the Run in eval


# --------------------------------------------------------------------- 7. snapshots with protocol state, resume

class _Tally(Protocol):
    """A protocol with state of its own: how many samples it saw, as an array (-> proto__Seen in a snapshot)."""
    protocol_type = "TALLY"
    def __init__(self, cfg, run, *, out_dir=None):
        super().__init__(cfg, run, out_dir=out_dir)
        self.Seen = np.zeros(1, dtype=np.int64)
    def execute(self):
        return {}


def test_sample_snapshots_round_trip_run_and_protocol_state(tmp_path):
    run = _run(checkpoint=CheckpointCfg(to_disk=False, every=2))
    run.checkpoints.cfg = run.runcfg.checkpoint; run.checkpoints.out_dir = tmp_path
    proto = _Tally(ProtocolCfg("tally"), run)
    data = S._data()
    run.ctx.sim.current_epoch = 4       ##a clock worth restoring (epoch and runstep)
    run.set_mode("eval"); run.run_sample(data[5][0]); run.set_mode("train")     ##an eval sample first, so train_step differs from runstep
    for i in range(2):
        run.run_sample(data[i][0]); proto.Seen += 1
    assert run.checkpoints.on_sample_end(stage=0, position=1, progress={"j": 1}, protocol=proto) is None      ##1 % 2 != 0: nothing written
    path = run.checkpoints.on_sample_end(stage=0, position=2, progress={"j": 2}, protocol=proto)
    assert path == tmp_path / "snap_s00_i00000002.npz" and path.exists()      ##the stage is in the name
    W2, V2, tr2 = run.ctx.syn.W.copy(), run.ctx.neur.V.copy(), run.plasticity.Pre_trace.copy()
    step2, train2 = run.ctx.sim.current_runstep, run.ctx.sim.train_step
    for i in range(2, 5):
        run.run_sample(data[i][0]); proto.Seen += 1
    assert not np.array_equal(W2, run.ctx.syn.W) and proto.Seen[0] == 5
    ##a fresh Run + protocol continue from the file: W, neuronal + module state, the protocol's array and the progress
    run2 = _run(); proto2 = _Tally(ProtocolCfg("tally"), run2)
    progress = proto2.resume(path)
    assert progress == {"j": 2} and proto2.progress == {"j": 2} and proto2.Seen[0] == 2
    assert np.array_equal(run2.ctx.syn.W, W2) and np.array_equal(run2.ctx.neur.V, V2) and np.array_equal(run2.plasticity.Pre_trace, tr2)
    ##... and the CLOCK, so a resumed run does not restart at 0
    assert run2.ctx.sim.current_runstep == step2 > 0 and run2.ctx.sim.current_epoch == 4
    ##... and the train-step clock (the event-mode time base), in the snapshot's meta
    assert run2.ctx.sim.train_step == train2 and 0 < train2 < step2 and load_snapshot(path).train_step == train2
    ##finalize() (select="best", inside a running fit) leaves the clock alone: timelines never run backwards
    before = run.ctx.sim.current_runstep
    run.checkpoints.cfg = dataclasses.replace(run.checkpoints.cfg, select="best"); run.checkpoints.best = load_snapshot(path); run.checkpoints.finalize()
    assert np.array_equal(run.ctx.syn.W, W2) and run.ctx.sim.current_runstep == before > step2
    ##the file layout: W, neur__, mod__ + proto__ / progress__ / meta
    keys = set(np.load(path).files)
    assert {"W", "proto__Seen", "progress__j", "meta"} <= keys and any(k.startswith("mod__plasticity__") for k in keys)
    ##force= writes off the interval (the last sample); a protocol whose array is missing from a snapshot refuses to resume
    assert run.checkpoints.on_sample_end(stage=0, position=5, progress={"j": 5}, protocol=proto, force=True) == tmp_path / "snap_s00_i00000005.npz"
    plain = save_snapshot(take_snapshot(run.ctx, epoch=0, score=None, reason="scheduled", modules=run.modules), tmp_path / "plain.npz")
    with pytest.raises(RuntimeError, match="Seen"):
        _Tally(ProtocolCfg("tally"), _run()).resume(plain)
    ##a snapshot without protocol state still loads as a plain Checkpoint (TrainedModel-like)
    assert load_snapshot(plain).protocol_state == {} and load_snapshot(plain).progress == {}


# --------------------------------------------------------------------- 8. test() under content="weights" keeps the learned arrays

def test_run_test_keeps_learned_theta_under_content_weights_and_needs_it_frozen():
    """test() under content="weights" resets only the transient state (keeps the learned Theta); an unfrozen Theta in the test pass is caught."""
    run = _dc_run(checkpoint=CheckpointCfg(to_disk=False, content="weights"))
    HoldoutValidation(None, run).ops.run_fit(run.train_set, None, epochs=1, shuffle=False)
    th = run.threshold
    assert not np.array_equal(th.Theta, run._init_state["threshold"]["Theta"]), "theta did not adapt in training"
    with pytest.raises(RuntimeError, match="threshold.Theta"):
        run.test()                                            ##theta adapts (or decays) during the test pass: the contract fails, by name
    run._eval_guard = None; run.ctx.sim.mode = "train"; run._learning = True
    th.freeze(True)
    Theta = th.Theta.copy()
    t = run.test()
    assert t["n_test"] > 0 and np.array_equal(Theta, th.Theta), "test() reset the learned threshold"
    ##and the TrainedModel under content="weights" keeps the learned Theta beside W
    import tempfile
    p = run.save_trained_model(save_dir=tempfile.mkdtemp())
    keys = set(np.load(p.with_suffix(".npz")).files)
    assert keys == {"W", "mod__threshold__Theta"}, keys


# --------------------------------------------------------------------- 9. modules declare `persistent`, not `learned`

def test_module_declaring_the_old_learned_name_is_refused():
    """A module still declaring `learned` is refused by name at build; the same module declaring `persistent` builds and is guarded."""
    from original_dynamics import AdaptiveThreshold
    class OldName(AdaptiveThreshold):
        learned = ("V_th",)
    class NewName(AdaptiveThreshold):
        persistent = ()
    pnet, run = S.assemble()
    pnet.set_runtime_classes(threshold_cls=OldName)
    with pytest.raises(AttributeError, match="renamed 'persistent'"):
        pnet._prep_run(S._data(), run.runcfg)
    pnet.set_runtime_classes(threshold_cls=NewName)
    run2 = pnet._prep_run(S._data(), run.runcfg)
    assert isinstance(run2.threshold, NewName) and [label for label, _ in run2._persistent()] == ["W"]
