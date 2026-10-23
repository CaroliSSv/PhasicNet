#test_operations -- the blocks between a Protocol and its Run (operations.Operations): the resumable pass (stages finished before a snapshot are
# skipped, a resumed run equals a straight one), the common tail (results.json, responses.npz, report keys), the experiment boundary in
# PhasicNet.train (status.json, run.jsonl / results.jsonl, logging restored), RunCfg hashing, the RuntimeClasses pack, parallel.run_parallel,
# the fit ops (HoldoutValidation, resumable bit-identically) and binned_counts / for_samples / needs_test_set.
# Run: pytest system/tests/test_operations.py -q     (from the repo root)
from __future__ import annotations
from pathlib import Path
##sys.path is set by system/tests/conftest.py
import dataclasses
import json
import logging
import numpy as np
import pytest

from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from protocols import Protocol, HoldoutValidation
from operations import Operations, RunStatus
from checkpoint import CheckpointPolicy, snapshots, load_snapshot
from modelinput import ModelInput
from phasicnet import PhasicNet, RuntimeClasses
import test_sparse as S          ##assemble(fmt=, update=) builds the same small network in every execution mode


# --------------------------------------------------------------------- helpers (module level: parallel's children import them by name)
def toy_build():
    """A PhasicNet with network, cfg and runtime classes set and NO run: what parallel.Job.build must return."""
    return S.assemble(fmt="dense")[0]


def toy_dataset(n_samples: int = 12):
    """The data for parallel.Job(load=toy_dataset, load_kw=): one ModelInput (train() makes its own test split)."""
    return S._data(n_samples=n_samples)


RC = dict(scorer=ScorerCfg(rate_hz_min=0.0), checkpoint=CheckpointCfg(to_disk=False), data=DataCfg(test_fraction=0.25))


def _run(**rc):
    pnet, run = S.assemble(fmt="dense")
    if rc:
        run.runcfg = dataclasses.replace(run.runcfg, **rc)
        run.checkpoints.cfg = run.runcfg.checkpoint
    return run


class _TwoStage(Protocol):
    """Stage 0: responses of the first 4 train samples; stage 1: of 3 more. respond() counts its calls (the resume test counts what is NOT re-run)."""
    protocol_type = "TWOSTAGE"
    calls: list = []

    def __init__(self, cfg, run, *, out_dir=None):
        super().__init__(cfg, run, out_dir=out_dir)
        self.Resp_a = np.zeros((4, 1), dtype=np.int32); self.Y_a = np.full(4, -1, dtype=np.int64)
        self.Resp_b = np.zeros((3, 1), dtype=np.int32); self.Y_b = np.full(3, -1, dtype=np.int64)

    def respond(self, frames, target, key):
        type(self).calls.append(key)
        return self.present(frames, target).sum()

    def execute(self):
        run, ops = self.run, self.ops
        run.set_mode("eval")
        ops.header(toy=1)
        ran0 = ops.responses(run.train_set, self.respond, self.Resp_a, self.Y_a, stage=0, name="a", n=4)
        ran1 = ops.responses(run.train_set.subset([4, 5, 6]), self.respond, self.Resp_b, self.Y_b, stage=1, name="b")
        return ops.finish({"ran": [ran0, ran1], "total": int(self.Resp_a.sum() + self.Resp_b.sum())}, files=dict(a=self.Resp_a, b=self.Resp_b),
                          lines=("  toy line",))


class _Boom(Protocol):
    protocol_type = "BOOM"
    def execute(self):
        self.ops.header()
        raise RuntimeError("the protocol broke on purpose")


class _Ctrl(Protocol):
    protocol_type = "CTRL"
    def execute(self):
        raise KeyboardInterrupt


# --------------------------------------------------------------------- 1. the resumable pass
def test_responses_resume_skips_finished_stages_and_matches_a_straight_run(tmp_path):
    _TwoStage.calls = []
    run = _run(checkpoint=CheckpointCfg(to_disk=False, every=2)); run.checkpoints.out_dir = tmp_path
    proto = _TwoStage(ProtocolCfg("twostage", report_every=2), run, out_dir=tmp_path)
    full = proto.execute()
    assert full["ran"] == [True, True] and len(_TwoStage.calls) == 7 and proto.progress == {"stage": 2, "i": 0}
    ##the stage is in the name; the default keep_last=3 carousel dropped the oldest (stage 0 at 2)
    assert [p.name for p in snapshots(tmp_path)] == ["snap_s00_i00000004.npz", "snap_s01_i00000002.npz", "snap_s01_i00000003.npz"]
    assert full["seconds"].keys() >= {"a", "b", "total"} and full["protocol"] == "_TwoStage" and full["protocol_cfg"]["protocol_type"] == "TWOSTAGE"
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["total"] == full["total"] and res["run_id"] == run.runmeta.run_id and "runmeta" not in res     ##report() keys go to run.report only
    z = np.load(tmp_path / "responses.npz"); assert np.array_equal(z["a"], proto.Resp_a) and np.array_equal(z["b"], proto.Resp_b)
    ##snap_s01_i00000002 was written by stage 1 at i = 2
    snap = np.load(tmp_path / "snap_s01_i00000002.npz")
    assert int(snap["progress__stage"]) == 1 and int(snap["progress__i"]) == 2
    ##resume there: stage 0 is skipped (its arrays came back), stage 1 runs its ONE remaining sample, the result is the straight run's
    _TwoStage.calls = []
    run2 = _run(checkpoint=CheckpointCfg(to_disk=False, every=2))
    proto2 = _TwoStage(ProtocolCfg("twostage", report_every=2), run2)
    assert proto2.ops.starting(0) and proto2.ops.starting(1)
    proto2.resume(tmp_path / "snap_s01_i00000002.npz")
    assert proto2.progress == {"stage": 1, "i": 2} and not proto2.ops.starting(0) and not proto2.ops.starting(1) and proto2.ops.starting(2)
    again = proto2.execute()
    assert again["ran"] == [False, True] and _TwoStage.calls == ["b:2"] and again["total"] == full["total"]
    assert np.array_equal(proto2.Resp_a, proto.Resp_a) and np.array_equal(proto2.Resp_b, proto.Resp_b)


def test_snapshot_names_carry_the_stage_and_the_carousel_keeps_the_newest(tmp_path, monkeypatch):
    """Snapshots are snap_s<stage>_i<position>.npz; keep_last=N keeps the newest N by the PARSED (stage, position), not by the name's spelling."""
    for keep, expected in ((None, ["snap_s00_i00000002.npz", "snap_s00_i00000004.npz", "snap_s01_i00000002.npz", "snap_s01_i00000003.npz"]),
                           (2, ["snap_s01_i00000002.npz", "snap_s01_i00000003.npz"])):
        d = tmp_path / f"keep{keep}"; d.mkdir()
        run = _run(checkpoint=CheckpointCfg(to_disk=False, every=2, keep_last=keep)); run.checkpoints.out_dir = d
        _TwoStage(ProtocolCfg("twostage"), run, out_dir=d).execute()
        assert [p.name for p in snapshots(d)] == expected, keep
    ##parsed, not sorted: a 9-digit position is NEWER than an 8-digit one although it sorts first; stage 1 at 2 is newer than stage 0 at anything;
    ##a half-written .tmp.npz (utils._write_snapshot_npz's write-then-rename) is not a snapshot
    d = tmp_path / "order"; d.mkdir()
    for name in ("snap_s01_i00000002.npz", "snap_s00_i100000000.npz", "snap_s00_i99999999.npz", "snap_s00_i00000003.tmp.npz", "snap_s00_i00000003.npz"):
        (d / name).write_bytes(b"")
    assert [p.name for p in snapshots(d)] == ["snap_s00_i00000003.npz", "snap_s00_i99999999.npz", "snap_s00_i100000000.npz", "snap_s01_i00000002.npz"]
    assert sorted(p.name for p in snapshots(d)) != [p.name for p in snapshots(d)], "a name sort agrees here, so the parsing proves nothing"
    run = _run()
    assert RunStatus(d, run, HoldoutValidation(None, run)).last_snapshot() == "snap_s01_i00000002.npz"
    assert RunStatus(None, run, HoldoutValidation(None, run)).last_snapshot() is None
    ##a REUSED run folder with an earlier run's file at a HIGHER (stage, position): the stale file neither evicts the new one nor is touched
    d = tmp_path / "stale"; d.mkdir(); (d / "snap_s03_i00000099.npz").write_bytes(b"")
    run = _run(checkpoint=CheckpointCfg(to_disk=False, every=2, keep_last=1)); run.checkpoints.out_dir = d
    written = []
    orig = CheckpointPolicy.on_sample_end
    def record(policy, **kw):
        path = orig(policy, **kw)
        if path is not None:
            assert path.exists(), path
            written.append(path.name)
        return path
    monkeypatch.setattr(CheckpointPolicy, "on_sample_end", record)
    _TwoStage(ProtocolCfg("twostage"), run, out_dir=d).execute()
    assert written == ["snap_s00_i00000002.npz", "snap_s00_i00000004.npz", "snap_s01_i00000002.npz", "snap_s01_i00000003.npz"]
    assert [p.name for p in snapshots(d)] == ["snap_s01_i00000003.npz", "snap_s03_i00000099.npz"]


def _kill_after_first_snapshot(monkeypatch):
    """The crash of a long run, reproduced: the policy writes its first snapshot and the process dies (KeyboardInterrupt -> status "interrupted")."""
    orig = CheckpointPolicy.on_sample_end
    def kill(policy, **kw):
        path = orig(policy, **kw)
        if path is not None:
            raise KeyboardInterrupt
        return path
    monkeypatch.setattr(CheckpointPolicy, "on_sample_end", kill)


def test_holdoutval_resume_is_bit_identical(tmp_path, monkeypatch):
    """The default protocol's fit is resumable: killed after its first snapshot, or resumed inside epoch 1, it gives the straight run's W, clock and val scores."""
    rc = RunCfg(protocol=ProtocolCfg(epochs=2, val_split=0.25), scorer=ScorerCfg(rate_hz_min=0.0), checkpoint=CheckpointCfg(to_disk=False, every=3, keep_last=None),
                data=DataCfg(test_fraction=0.25))
    full = S.assemble(fmt="dense")[0].train(S._data(), rc, out_dir=tmp_path / "full", save_run=False)
    n = len(HoldoutValidation(rc.protocol, full).partitions()[0][0])      ##6 train samples after the test (4) and val (2) holdouts: every=3 -> epoch 0 at i=3, its end (6), epoch 1 at 6+3, its end (12)
    assert n == 6
    names = [p.name for p in snapshots(tmp_path / "full")]
    assert names == ["snap_s00_i00000003.npz", "snap_s00_i00000006.npz", "snap_s00_i00000009.npz", "snap_s00_i00000012.npz"]
    assert full.status["progress"] == {"fold": 1, "epoch": 0, "i": 0}
    z = np.load(tmp_path / "full" / names[2])
    assert (int(z["progress__fold"]), int(z["progress__epoch"]), int(z["progress__i"])) == (0, 1, 3) and int(z["progress__i"]) < n
    ##the epoch's end is the NEXT epoch's start, written after the val pass, so nothing is re-run from it
    ends = [load_snapshot(tmp_path / "full" / names[k]) for k in (1, 3)]
    assert [(e.progress, e.epoch) for e in ends] == [({"fold": 0, "epoch": 1, "i": 0}, 1), ({"fold": 0, "epoch": 2, "i": 0}, 2)]
    assert ends[1].runstep == full.ctx.sim.current_runstep > ends[0].runstep
    ##killed after the first snapshot, resumed from it (the status record says where)
    _kill_after_first_snapshot(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        S.assemble(fmt="dense")[0].train(S._data(), rc, out_dir=tmp_path / "killed", save_run=False)
    monkeypatch.undo()
    st = json.loads((tmp_path / "killed" / "status.json").read_text())
    assert st["status"] == "interrupted" and st["last_snapshot"] == names[0] and st["progress"] == {"fold": 0, "epoch": 0, "i": 3}
    back = S.assemble(fmt="dense")[0].train(S._data(), rc, out_dir=tmp_path / "killed", resume_from=tmp_path / "killed" / st["last_snapshot"], save_run=False)
    assert np.array_equal(back.ctx.syn.W, full.ctx.syn.W) and back.ctx.sim.current_runstep == full.ctx.sim.current_runstep > 0
    assert back.report["fold_scores"] == full.report["fold_scores"] and back.status["progress"] == {"fold": 1, "epoch": 0, "i": 0}
    assert [p.name for p in snapshots(tmp_path / "killed")] == names          ##the resumed run wrote the rest of the carousel
    ##resumed inside epoch 1 (from the full run's third snapshot, into a new folder): the same W again; this process ran epoch 1 only
    later = S.assemble(fmt="dense")[0].train(S._data(), rc, out_dir=tmp_path / "later", resume_from=tmp_path / "full" / names[2], save_run=False)
    assert np.array_equal(later.ctx.syn.W, full.ctx.syn.W) and later.ctx.sim.current_runstep == full.ctx.sim.current_runstep
    assert later.report["folds"][0]["epochs_run"] == 1 and later.report["fold_scores"] == full.report["fold_scores"]
    ##and without a resume the same recipe does NOT reproduce a half-run: a fresh run from the as-built W differs from the restored one after 3 samples
    assert not np.array_equal(z["W"], full._W0), "the snapshot holds the as-built weights, so the restore proves nothing"


def test_holdoutval_epoch_aligned_every_snapshots_every_epoch_end(tmp_path):
    """every = n_train / 2 * n_train writes one snapshot per epoch end / per two; a resume from an epoch end is bit-identical, from the last one it finalizes."""
    rc = RunCfg(protocol=ProtocolCfg(epochs=2, val_split=0.25), scorer=ScorerCfg(rate_hz_min=0.0), checkpoint=CheckpointCfg(to_disk=False, every=6),
                data=DataCfg(test_fraction=0.25))
    full = S.assemble(fmt="dense")[0].train(S._data(), rc, out_dir=tmp_path / "full", save_run=False)
    assert len(HoldoutValidation(rc.protocol, full).partitions()[0][0]) == 6 and full.report["folds"][0]["epochs_run"] == 2
    names = [p.name for p in snapshots(tmp_path / "full")]
    assert names == ["snap_s00_i00000006.npz", "snap_s00_i00000012.npz"] and full.status["last_snapshot"] == names[1]
    assert [load_snapshot(tmp_path / "full" / nm).progress for nm in names] == [{"fold": 0, "epoch": 1, "i": 0}, {"fold": 0, "epoch": 2, "i": 0}]
    ##from epoch 0's end: epoch 1 alone, the same W and clock
    back = S.assemble(fmt="dense")[0].train(S._data(), rc, out_dir=tmp_path / "back", resume_from=tmp_path / "full" / names[0], save_run=False)
    assert np.array_equal(back.ctx.syn.W, full.ctx.syn.W) and back.ctx.sim.current_runstep == full.ctx.sim.current_runstep > 0
    assert back.report["folds"][0]["epochs_run"] == 1 and back.report["folds"][0]["epochs"] == full.report["folds"][0]["epochs"][1:]
    assert [p.name for p in snapshots(tmp_path / "back")] == names[1:]
    ##from epoch 1's end (the fit's last): no epoch runs, the model is the restored one, the run completes
    done = S.assemble(fmt="dense")[0].train(S._data(), rc, out_dir=tmp_path / "done", resume_from=tmp_path / "full" / names[1], save_run=False)
    assert np.array_equal(done.ctx.syn.W, full.ctx.syn.W) and done.ctx.sim.current_runstep == full.ctx.sim.current_runstep
    assert done.report["folds"][0]["epochs_run"] == 0 and done.status["status"] == "completed" and done.status["progress"] == {"fold": 1, "epoch": 0, "i": 0}
    assert snapshots(tmp_path / "done") == []
    ##every two epochs
    rc2 = dataclasses.replace(rc, checkpoint=CheckpointCfg(to_disk=False, every=12))
    two = S.assemble(fmt="dense")[0].train(S._data(), rc2, out_dir=tmp_path / "two", save_run=False)
    assert [p.name for p in snapshots(tmp_path / "two")] == ["snap_s00_i00000012.npz"] and np.array_equal(two.ctx.syn.W, full.ctx.syn.W)


def test_repeat_runs_a_body_and_report_every_zero_still_logs_once(caplog):
    run = _run()
    ops = Operations(run, report_every=0)
    seen = []
    with caplog.at_level(logging.INFO, logger="operations"):
        assert ops.repeat(3, seen.append, stage=0, name="x") and ops.progress == {"stage": 1, "i": 0}
    assert seen == [0, 1, 2] and ops.report_every == 1 and sum("x " in r.message for r in caplog.records) == 3
    assert ops.repeat(2, seen.append, stage=0, name="x") is False and seen == [0, 1, 2]       ##stage 0 is behind the position: skipped
    assert "seconds" not in ops.seconds and ops.seconds.keys() == {"x"}


def test_protocol_present_still_delegates_and_progress_is_the_ops():
    run = _run()
    proto = HoldoutValidation(None, run)
    assert proto.ops.run is run and proto.ops.state is proto and proto.progress is proto.ops.progress
    proto.progress = {"j": 3}
    assert proto.ops.progress == {"j": 3}
    counts = proto.present(*S._data()[0]); assert counts.shape[0] == run.ctx.net.n_total
    assert proto.ops.run_epoch(S._data().subset(np.arange(2)))["samples"] == 2 and not hasattr(proto, "run_pass")


# --------------------------------------------------------------------- 2. the experiment boundary (PhasicNet.train)
def _pnet_with(protocol_cls):
    pnet = S.assemble(fmt="dense")[0]
    pnet.set_runtime_classes(protocol_classes=(protocol_cls,))
    return pnet


def test_train_writes_status_and_log_and_restores_logging(tmp_path):
    root = logging.getLogger(); n_handlers, level = len(root.handlers), root.level
    pnet = _pnet_with(_TwoStage)
    run = pnet.train(S._data(), RunCfg(protocol=ProtocolCfg("twostage", report_every=3), **RC), out_dir=tmp_path, save_run=False)
    assert len(root.handlers) == n_handlers and root.level == level                     ##the file handler is gone again, the level is back
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["status"] == "completed" and st["run_id"] == run.runmeta.run_id and st["error"] is None and st["progress"] == {"stage": 2, "i": 0}
    assert st["started"] <= st["ended"] and st["seconds"] > 0 and run.status is not None and run.status["status"] == "completed"
    log = (tmp_path / "run.jsonl").read_text()
    assert "run_id=" in log and "RUN COMPLETE" in log and "toy line" in log and "RUN COMPLETED" in log and "toy=1" in log
    assert run.report["ran"] == [True, True]
    lines = [json.loads(l) for l in log.splitlines()]                                                   ##every line is JSON
    assert lines[0]["message"] == "provenance" and lines[0]["provenance"]["run_id"] == run.runmeta.run_id
    assert all(l["run_id"] == run.runmeta.run_id for l in lines)
    res = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [r["message"] for r in res] == ["provenance", "results", "status"] and res[-1]["status"] == "completed"


def test_train_records_a_failure_and_reraises(tmp_path):
    root = logging.getLogger(); n_handlers = len(root.handlers)
    pnet = _pnet_with(_Boom)
    with pytest.raises(RuntimeError, match="on purpose"):
        pnet.train(S._data(), RunCfg(protocol=ProtocolCfg("boom"), **RC), out_dir=tmp_path, save_run=False)
    assert len(root.handlers) == n_handlers
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["status"] == "failed" and st["error"]["type"] == "RuntimeError" and "on purpose" in st["error"]["message"]
    assert "Traceback" in st["error"]["traceback"] and st["last_snapshot"] is None and st["progress"] == {}
    assert "RUN FAILED" in (tmp_path / "run.jsonl").read_text()
    last = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[-1])
    assert last["message"] == "status" and last["status"] == "failed" and last["error"] == "RuntimeError"


def test_train_records_an_interruption_and_reraises(tmp_path):
    pnet = _pnet_with(_Ctrl)
    with pytest.raises(KeyboardInterrupt):
        pnet.train(S._data(), RunCfg(protocol=ProtocolCfg("ctrl"), **RC), out_dir=tmp_path, save_run=False)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "interrupted"


def test_train_without_out_dir_keeps_the_status_on_the_run_only(tmp_path):
    pnet = _pnet_with(_TwoStage)
    run = pnet.train(S._data(), RunCfg(protocol=ProtocolCfg("twostage"), **RC), save_run=False)
    assert run.status["status"] == "completed" and not list(tmp_path.iterdir())


# --------------------------------------------------------------------- 3. RunCfg: report_every, train() requires a ModelInput, load/save_runcfg
def test_report_every_is_hash_neutral_and_the_default_protocol_stays_neutral():
    a, b, c = RunCfg(), RunCfg(protocol=ProtocolCfg()), RunCfg(protocol=ProtocolCfg(report_every=7))
    assert a.hashable() == b.hashable() == c.hashable() and "protocol" not in a.hashable()
    d = RunCfg(protocol=ProtocolCfg(reset_between_samples=True, report_every=7))
    assert d.hashable()["protocol"] == {"protocol_type": "HOLDOUTVAL", "reset_between_samples": True, "epochs": 1, "patience": None, "val_split": None}
    assert not {f.name for f in dataclasses.fields(DataCfg)} & {"dataset", "dataset_kw"} and "data.dataset" not in RunCfg.HASH_EXCLUDE


def test_train_requires_a_modelinput():
    """train() refuses non-ModelInput data / test_data with a TypeError that says how to wrap them; data is required."""
    pnet = S.assemble(fmt="dense")[0]
    raw = [(np.zeros((5, S.N_INPUT), bool), k % 2) for k in range(8)]
    with pytest.raises(TypeError, match="ModelInput as data.*from_tonic"):
        pnet.train(raw, RunCfg(**RC), save_run=False)
    with pytest.raises(TypeError, match="ModelInput as test_data"):
        pnet.train(ModelInput.from_tonic(raw), RunCfg(**RC), test_data=raw, save_run=False)
    with pytest.raises(TypeError, match="ModelInput as data"):
        pnet.train(None, RunCfg(**RC), save_run=False)
    with pytest.raises(TypeError):
        pnet.train(runcfg=RunCfg(**RC), save_run=False)           ##data is positional and required
    run = pnet.train(ModelInput.from_tonic(raw), RunCfg(**RC), test_data=ModelInput.from_tonic(raw[:3]), save_run=False)
    assert len(run.train_set) == 8 and len(run.test_set) == 3 and isinstance(run.train_set, ModelInput) and run.test_set.name == "test"


def test_load_and_save_runcfg_round_trip(tmp_path):
    from dc_protocol import DCProtocolCfg
    rc = RunCfg(data=DataCfg(test_fraction=0.1, n_train=8),
                checkpoint=CheckpointCfg(select="best", every=5), protocol=DCProtocolCfg(n_train=3, report_every=9, epochs=2, patience=1))
    pnet = PhasicNet(1)
    for fmt in ("yaml", "json"):
        pnet.save_runcfg("rc_test", runcfg=rc, JSON=fmt == "json", YAML=fmt == "yaml", save_to=tmp_path)
        pnet.runcfg = None
        pnet.load_runcfg(tmp_path / f"rc_test.{fmt}")
        assert pnet.runcfg == rc and isinstance(pnet.runcfg.protocol, DCProtocolCfg) and pnet.runcfg.data.n_train == 8
    with pytest.raises(ValueError, match="No runcfg"):
        PhasicNet(1).save_runcfg("x", save_to=tmp_path)
    ##COMPAT: an old runcfg.json with data.dataset / dataset_kw is read and the fields dropped
    old = RunCfg.load({"data": {"test_fraction": 0.1, "dataset": "ops_toy", "dataset_kw": {"n_samples": 8}, "cfg_type": "DataCfg"}})
    assert old.data.test_fraction == 0.1 and old == RunCfg(data=DataCfg(test_fraction=0.1))


# --------------------------------------------------------------------- 4. the RuntimeClasses pack
def test_runtime_classes_pack_equals_the_keyword_form_and_may_be_partial():
    from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, AdaptiveThreshold, TraceSTDP
    from original_dynamics import TestNoise
    from phases import DefaultPhase
    pack = RuntimeClasses(synapse_cls=DeltaCurrentDense, integrator_cls=LIFexact, threshold_cls=AdaptiveThreshold, plasticity_cls=TraceSTDP,
                          noise_cls=TestNoise, phase_classes=(DefaultPhase,), protocol_classes=(_TwoStage,))
    a, b = PhasicNet(1), PhasicNet(1)
    a.set_runtime_classes(pack)
    b.set_runtime_classes(synapse_cls=DeltaCurrentDense, integrator_cls=LIFexact, threshold_cls=AdaptiveThreshold, plasticity_cls=TraceSTDP,
                          noise_cls=TestNoise, phase_classes=(DefaultPhase,), protocol_classes=(_TwoStage,))
    keys = ("synapse_cls", "integrator_cls", "threshold_cls", "plasticity_cls", "noise_cls", "phase_classes", "protocol_classes")
    assert all(getattr(a, k) == getattr(b, k) for k in keys)
    a.set_runtime_classes(dataclasses.replace(pack, synapse_cls=DeltaCurrentSparse, integrator_cls=None))      ##partial: None leaves a class as it is
    assert a.synapse_cls is DeltaCurrentSparse and a.integrator_cls is LIFexact


def test_runtime_classes_paths_round_trip_and_train_records_them(tmp_path):
    """RuntimeClasses.paths() / from_paths() round-trip every pack as importable "module.Class"; train(out_dir=) writes runtime_classes.json."""
    import deckers_2022 as dk, hlsp_lsm_2026 as hl, nalsm_2021 as na, krenzer_2025 as kr, zhang_2025 as zh
    for pack in (dk.ELSM_SPARSE, hl.HLSP_SPARSE, na.NALSM_SPARSE, kr.RLSM_SPARSE, zh.ZHANG_CLASSES):
        paths = pack.paths()
        assert paths["synapse_cls"].count(".") == 1 and isinstance(paths["protocol_classes"], list) and "." in paths["protocol_classes"][0]
        assert RuntimeClasses.from_paths(json.loads(json.dumps(paths))) == pack                  ##json-safe, and the classes come back by identity
    assert hl.HLSP_SPARSE.paths()["plasticity_cls"] == "hlsp2026.HLSPPlasticitySparse" and hl.HLSP_SPARSE.paths()["noise_cls"] == "original_dynamics.TestNoise"
    partial = RuntimeClasses(synapse_cls=hl.HLSP_DENSE.synapse_cls)
    assert partial.paths() == {"synapse_cls": "hlsp2026.HLSPCurrentSynapse"} and RuntimeClasses.from_paths(partial.paths()) == partial   ##None slots are left out
    pnet = hl.build(n_liquid=50, n_input=40, w_in=1.0, seed=3, weights_format="dense")
    assert pnet.runtime_classes == hl.HLSP_DENSE                                                   ##the pack the seven slots hold
    pnet = S.assemble(fmt="dense")[0]
    pnet.train(S._data(), RunCfg(**RC), out_dir=tmp_path, save_run=False)
    rec = json.loads((tmp_path / "runtime_classes.json").read_text())
    assert rec == pnet.runtime_classes.paths() and RuntimeClasses.from_paths(rec) == pnet.runtime_classes
    assert rec["threshold_cls"] == "original_dynamics.AdaptiveThreshold" and rec["protocol_classes"] == ["protocols.HoldoutValidation"]


def test_protocol_cfgs_carry_no_n_train_and_runcfg_puts_it_in_datacfg():
    """The protocol cfgs have no n_train / n_test: the builds' runcfg() puts them in DataCfg, and the protocols run len(run.train_set)."""
    import deckers_2022 as dk, hlsp_lsm_2026 as hl, nalsm_2021 as na, krenzer_2025 as kr
    for mod, cfg_cls in ((dk, dk.ELSMProtocolCfg), (hl, hl.HLSPProtocolCfg), (na, na.NALSMProtocolCfg), (kr, kr.RLSMProtocolCfg)):
        names = {f.name for f in dataclasses.fields(cfg_cls)}
        assert "n_train" not in names and "n_test" not in names, cfg_cls.__name__
        rc = mod.runcfg(n_train=4, n_test=3)
        assert rc.data.n_train == 4 and rc.data.n_test == 3 and mod.runcfg().data.n_train is None
        with pytest.raises(TypeError):
            cfg_cls(n_train=4)
    assert na.NALSMProtocolCfg().val_split == hl.HLSPProtocolCfg().val_split == 1.0 / 6.0     ##the validation blocks: ProtocolCfg.val_split -> ops.partition
    pnet = kr.build(n_input=10, n_liquid=30, n_wta=4, n_wta_inh=2)
    data = ModelInput([np.zeros((5, 10), bool)] * 8, np.arange(8) % 2)
    run = pnet._prep_run(data, kr.runcfg(n_train=6, n_test=3), test_data=data)
    proto = kr.RLSMProtocol(run.runcfg.protocol, run)
    assert len(run.train_set) == 6 and len(run.test_set) == 3 and proto._n_train == 6 and proto._n_test == 3 and proto.Resp_train.shape[0] == 6


def test_build_files_expose_their_packs():
    import deckers_2022 as dk, hlsp_lsm_2026 as hl, nalsm_2021 as na, krenzer_2025 as kr, zhang_2025 as zh
    for pack, proto in ((dk.ELSM_SPARSE, dk.ELSMProtocol), (hl.HLSP_SPARSE, hl.HLSPProtocol), (na.NALSM_SPARSE, na.NALSMProtocol), (kr.RLSM_SPARSE, kr.RLSMProtocol)):
        assert isinstance(pack, RuntimeClasses) and pack.protocol_classes == (proto,) and pack.synapse_cls.__name__.endswith("Sparse")
    assert zh.ZHANG_CLASSES.synapse_cls.__name__ == "ZhangDendriticSynapse"
    pnet = hl.build(n_liquid=50, n_input=40, w_in=1.0, seed=3, weights_format="dense")
    assert pnet.synapse_cls is hl.HLSP_DENSE.synapse_cls and pnet.plasticity_cls is hl.HLSP_DENSE.plasticity_cls
    ##every build exposes its loader as an importable module-level function (what parallel.Job(load=) takes)
    import dc_reproduce as dc, datahandler
    from parallel import Job
    for load in (dk.shd_dataset, hl.load_dataset, na.load_dataset, kr.shd70_dataset, dc.mnist_dataset):
        assert Job(toy_build, load, RunCfg()).load is load and getattr(__import__(load.__module__), load.__name__) is load
    assert not any(hasattr(datahandler, k) for k in ("register_dataset", "dataset_by_name", "_DATASETS", "Subset", "_labels_of"))


# --------------------------------------------------------------------- 5. parallel: two independent runs in two processes
def test_run_parallel_two_processes_return_both_reports(tmp_path):
    from parallel import Job, Result, run_parallel, run_in_process
    rc = RunCfg(scorer=ScorerCfg(rate_hz_min=0.0), checkpoint=CheckpointCfg(to_disk=False), data=DataCfg(test_fraction=0.25))
    jobs = [Job(toy_build, toy_dataset, rc, load_kw={"n_samples": 8}, train_kw=dict(out_dir=tmp_path / f"job{k}", save_run=False, run_name=f"job{k}")) for k in range(2)]
    results = run_parallel(jobs, workers=2)
    assert len(results) == 2 and all(isinstance(r, Result) and r.ok for r in results), [r.error for r in results]
    assert results[0].report["score"] == results[1].report["score"]                   ##same seed, same data, same settings: independent and identical
    assert all(r.status["status"] == "completed" and (Path(r.out_dir) / "status.json").exists() for r in results)
    rep = results[0].report      ##the child loaded the 8 samples itself (load(**load_kw): the run's dataset_id is THAT data's hash) and split them under the Job's RunCfg
    from modelinput import _dataset_hash
    assert rep["runcfg"]["data"]["test_fraction"] == 0.25 and rep["runmeta"]["dataset_id"] == _dataset_hash(toy_dataset(n_samples=8))[0] and rep["folds"][0]["epochs_run"] == 1
    ##a job whose run fails comes back as an error, the pool survives
    bad = Job(toy_build, toy_dataset, dataclasses.replace(rc, protocol=ProtocolCfg("boom")), load_kw={"n_samples": 8}, train_kw=dict(save_run=False))
    r = run_in_process(bad)
    assert not r.ok and "No protocol class registered" in r.error
    with pytest.raises(ValueError, match="`load` must be an importable"):
        Job(toy_build, lambda: None, rc)
    with pytest.raises(ValueError, match="`build` must be an importable"):
        Job(lambda: None, toy_dataset, rc)


# --------------------------------------------------------------------- 6. the fit levels in the operations
def _trained(protocol=None):
    pnet = S.assemble(fmt="dense")[0]
    return pnet.train(S._data(), RunCfg(protocol=protocol, **RC), save_run=False)


def test_holdoutval_without_val_split_is_the_plain_holdout():
    """HoldoutValidation with val_split None (or 0) is the plain holdout (one train-only fold, no score); with val_split one scored fold; no "refit"."""
    a, b = _trained(ProtocolCfg(epochs=2)), _trained(ProtocolCfg(epochs=2, val_split=0.0))
    assert a.report["protocol"] == b.report["protocol"] == "HoldoutValidation" and np.array_equal(a.ctx.syn.W, b.ctx.syn.W)
    assert a.report["fold_scores"] == [] and a.report["score"] is None and "refit" not in a.report
    assert a.report["folds"][0]["score"] is None and a.report["folds"][0]["fold"] == 0 and a.report["folds"][0]["epochs_run"] == 2
    rb = _trained(ProtocolCfg(epochs=2, val_split=0.25)).report
    assert rb["protocol"] == "HoldoutValidation" and len(rb["fold_scores"]) == 1 and rb["fold_scores"][0] is not None
    assert "refit" not in rb and rb["folds"][0]["fold"] == 0 and rb["folds"][0]["epochs_run"] == 2


def test_both_holdouts_go_through_run_fold(monkeypatch):
    """One fold = one run_fold: fold 0 without val for the plain holdout, fold 0 with val under val_split."""
    seen, orig = [], Operations.run_fold
    def spy(self, train, val=None, **kw):
        seen.append((kw["fold"], val is None)); return orig(self, train, val, **kw)
    monkeypatch.setattr(Operations, "run_fold", spy)
    _trained(ProtocolCfg(epochs=2))
    assert seen == [(0, True)]
    seen.clear()
    r = _trained(ProtocolCfg(epochs=2, val_split=0.25))
    assert seen == [(0, False)] and len(r.report["fold_scores"]) == 1


def test_holdout_validation_is_picked_by_name_and_is_the_only_built_in():
    """HoldoutValidation is picked by name, is the default and the only built-in; the old name HOLDOUT is refused."""
    pnet, run = S.assemble(fmt="dense")
    assert type(pnet._instantiate_protocol(RunCfg(protocol=ProtocolCfg(protocol_type="holdoutval")), run)) is HoldoutValidation
    assert type(pnet._instantiate_protocol(RunCfg(), run)) is HoldoutValidation and pnet.protocol_classes == (HoldoutValidation,)
    assert _trained(ProtocolCfg(val_split=0.0)).report["fold_scores"] == []
    with pytest.raises(RuntimeError, match="HOLDOUT"):
        pnet._instantiate_protocol(RunCfg(protocol=ProtocolCfg("holdout")), run)      ##RunCfg.load maps an old file's HOLDOUT to HOLDOUTVAL; a live cfg must say HOLDOUTVAL
    pnet.set_runtime_classes(protocol_classes=(_TwoStage,))
    with pytest.raises(RuntimeError, match="HOLDOUTVAL"):
        pnet._instantiate_protocol(RunCfg(), run)       ##a pack without HoldoutValidation does not have it


def test_results_jsonl_has_one_line_per_epoch_fold_and_the_folds(tmp_path):
    """results.jsonl: one line per epoch, one per fold, one aggregate, matching the report; a second run into the folder APPENDS behind its own provenance."""
    pnet = S.assemble(fmt="dense")[0]
    run = pnet.train(S._data(), RunCfg(protocol=ProtocolCfg(epochs=2, val_split=0.25), **RC), out_dir=tmp_path, save_run=False)
    res = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    epochs = [r for r in res if r["message"] == "epoch"]
    assert [r["fold"] for r in epochs] == [0, 0] and all("rate_hz" in r and "collapsed" in r for r in epochs)
    assert all(r["score"] is not None for r in epochs)
    folds = [r for r in res if r["message"] == "fold"]
    assert [r["fold"] for r in folds] == [0] and [r["score"] for r in folds] == run.report["fold_scores"]
    agg = next(r for r in res if r["message"] == "folds")
    assert agg["score"] == run.report["score"] and agg["n_folds"] == 1 and "refit_score" not in agg
    assert all(r["run_id"] == run.runmeta.run_id for r in res)
    pnet.train(S._data(), RunCfg(protocol=ProtocolCfg(epochs=2, val_split=0.25), **RC), out_dir=tmp_path, save_run=False)
    again = [json.loads(l)["message"] for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert again.count("provenance") == 2 and len(again) == 2 * len(res)


def test_run_fit_counts_its_own_epochs_when_called_twice():
    """run_fit clears epoch_log itself: a second call reports only its own epochs."""
    run = _run()
    ops = HoldoutValidation(None, run).ops
    assert ops.run_fit(run.train_set, None, epochs=2, shuffle=False)["epochs_run"] == 2
    assert ops.run_fit(run.train_set, None, epochs=1, shuffle=False)["epochs_run"] == 1 and len(ops.epoch_log) == 1


# --------------------------------------------------------------------- 7. binned_counts / for_samples / needs_test_set
def _count_calls(monkeypatch, obj, name):
    """Wrap obj.<name> so the returned list records one entry per call (the original still runs)."""
    calls, orig = [], getattr(obj, name)

    def wrapped(*a, **kw):
        calls.append(a)
        return orig(*a, **kw)
    monkeypatch.setattr(obj, name, wrapped)
    return calls


def test_binned_counts_bins_sum_to_present_and_are_one_presentation(monkeypatch):
    """binned_counts over ONE presentation: the bins sum to present()'s counts, one begin_sample, the remainder runs uncounted, bin 0 is present(until=)."""
    a, b = _run(), _run()
    frames, target = S._data()[0]; T = frames.shape[0]
    begins_a, begins_b = _count_calls(monkeypatch, a._datahandler, "begin_sample"), _count_calls(monkeypatch, b._datahandler, "begin_sample")
    full = Operations(a).present(frames, target)
    bins = Operations(b).binned_counts(frames, target, [T // 3, 2 * T // 3, T])
    assert bins.shape == (3, a.ctx.net.n_total) and bins.dtype == np.float64 and np.array_equal(bins.sum(0), full) and full.sum() > 0
    assert len(begins_a) == 1 and len(begins_b) == 1
    assert np.array_equal(a.ctx.syn.W, b.ctx.syn.W) and np.array_equal(a.ctx.neur.V, b.ctx.neur.V) and a.ctx.sim.current_runstep == b.ctx.sim.current_runstep
    assert np.array_equal(a.history.Counts, b.history.Counts)
    ##the last bound < T: one row, the rest of the sample runs UNCOUNTED (the network still ends at T, history still has every spike)
    c = _run()
    rows = Operations(c).binned_counts(frames, target, [T // 2])
    assert rows.shape == (1, c.ctx.net.n_total) and c.ctx.sim.step_in_sample == T and np.array_equal(c.history.Counts, a.history.Counts)
    assert np.array_equal(rows[0], Operations(_run()).present(frames, target, until=T // 2))
    assert np.array_equal(rows[0], Operations(_run()).binned_counts(frames, target, [T // 2, T])[0])


def test_binned_counts_resets_once_and_checks_its_bounds_and_groups(monkeypatch):
    run = _run(); run.set_mode("eval")
    frames, target = S._data()[0]; T = frames.shape[0]
    net = run.ctx.net
    resets = _count_calls(monkeypatch, run, "reset_transient")
    ops = Operations(run, reset_between_samples=True)
    full = ops.binned_counts(frames, target, [T // 2, T])
    assert len(resets) == 1                                                        ##ONCE at the start, never between bins
    Operations(run, reset_between_samples=False).binned_counts(frames, target, [T // 2, T])
    assert len(resets) == 1
    ##group: a name = its band, a tuple = the bands concatenated in the given order, a slice as is, None = everything (eval + reset: same counts every time)
    assert np.array_equal(ops.binned_counts(frames, target, [T // 2, T]), full)
    assert np.array_equal(ops.binned_counts(frames, target, [T // 2, T], group="exc"), full[:, net.neuronslices["exc"]])
    both = ops.binned_counts(frames, target, [T // 2, T], group=("inh", "exc"))
    assert np.array_equal(both, np.hstack([full[:, net.neuronslices["inh"]], full[:, net.neuronslices["exc"]]])) and both.shape[1] == net.n_total - net.n_input
    assert np.array_equal(ops.binned_counts(frames, target, [T // 2, T], group=slice(0, 3)), full[:, :3])
    assert np.array_equal(ops.present(frames, target, group="exc"), full[:, net.neuronslices["exc"]].sum(0))
    for bad in ([], [T + 1], [T // 2, T // 4], [-1, T]):
        with pytest.raises(ValueError, match="bounds"):
            ops.binned_counts(frames, target, bad)


def test_for_samples_walks_a_dataset_resets_on_request_and_resumes(monkeypatch):
    run = _run()
    data = S._data(n_samples=6)
    ops = Operations(run, report_every=0)
    resets = _count_calls(monkeypatch, run, "reset_transient")
    seen = []

    def handle(i, frames, target):
        seen.append((i, int(target), frames.shape[0]))
    assert ops.for_samples(data, handle, stage=0, name="x", reset=True) is True
    assert seen == [(i, int(data[i][1]), data[i][0].shape[0]) for i in range(6)] and len(resets) == 6 and ops.progress == {"stage": 1, "i": 0}
    ##reset=False (the default): no reset at all; a stage behind the position is skipped
    seen.clear()
    assert ops.for_samples(data, handle, stage=1, name="y") is True and len(seen) == 6 and len(resets) == 6
    assert ops.for_samples(data, handle, stage=0, name="x") is False and len(seen) == 6
    ##resumed at a position: the samples before it are NOT re-run; a view in another order walks that order (NALSM's init order)
    ops.progress = {"stage": 3, "i": 4}; seen.clear()
    assert ops.for_samples(data, handle, stage=3, name="z") and [s[0] for s in seen] == [4, 5]
    order = np.array([5, 2, 0]); seen.clear()
    assert ops.for_samples(data.subset(order), handle, stage=4, name="w") and [s[1] for s in seen] == [int(data[j][1]) for j in order]
    assert ops.seconds.keys() == {"x", "y", "z", "w"}


def test_responses_reset_flag_resets_before_every_sample(monkeypatch):
    run = _run()
    data = S._data(n_samples=4)
    ops = Operations(run, report_every=0)
    resets = _count_calls(monkeypatch, run, "reset_transient")
    Resp, Y = np.zeros((4, 1)), np.full(4, -1)

    def respond(frames, target, key):
        return ops.present(frames, target).sum()
    assert ops.responses(data, respond, Resp, Y, stage=0, name="a") and len(resets) == 0
    assert ops.responses(data, respond, Resp, Y, stage=1, name="b", reset=True) and len(resets) == 4
    assert Y.tolist() == [int(data[i][1]) for i in range(4)]


class _Needs(Protocol):
    protocol_type = "NEEDS"
    needs_test_set = True

    def execute(self):
        return {}


def test_needs_test_set_is_checked_once_in_protocol_init():
    """needs_test_set is checked once in Protocol.__init__: False on the base and HoldoutValidation, True on the five experiment protocols."""
    run = _run()
    cfg = ProtocolCfg(protocol_type="NEEDS")
    assert Protocol.needs_test_set is False and HoldoutValidation.needs_test_set is False and len(run.test_set) > 0
    assert isinstance(_Needs(cfg, run), _Needs)
    had = run.test_set
    for empty in (None, had.subset([])):
        run.test_set = empty
        with pytest.raises(ValueError, match="test set"):
            _Needs(cfg, run)
        HoldoutValidation(None, run)        ##unaffected
    run.test_set = had
    from deckers_2022 import ELSMProtocol
    from hlsp_lsm_2026 import HLSPProtocol
    from nalsm_2021 import NALSMProtocol
    from krenzer_2025 import RLSMProtocol
    from dc_protocol import DCProtocol
    assert all(p.needs_test_set is True for p in (ELSMProtocol, HLSPProtocol, NALSMProtocol, RLSMProtocol, DCProtocol))
