#test_trial -- the run layer decides WHICH WEIGHTS YOU REPORT; every failure mode here is silent and gives a number that looks fine:
#   1. AGGREGATION. The score is the mean over folds, never the max (the max reports the luckiest split).
#   2. FOLD INDEPENDENCE. Every fit starts from the as-built weights and a reset neuronal state.
#   3. CHECKPOINT RETENTION. "best" vs "final" decides which weights survive, so the policy cfg is archived with the run.
#   4. VAL MUST NOT BE SHUFFLED. It changes no score but consumes RNG and breaks reproducibility.
#   5. SCORER TIERS. CH must match its definition; collapse is decided on rate (silence, runaway); saturation is probed and logged.
# Run: pytest system/tests/test_trial.py -q
from __future__ import annotations
import dataclasses
import numpy as np
from modelinput import ModelInput   ##train() / _prep_run require a ModelInput
import pytest

from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, ProtocolCfg
from checkpoint import CheckpointPolicy, take_snapshot, restore_snapshot
from scorer import Scorer, calinski_harabasz
from run import Run
from protocols import HoldoutValidation      ##the default protocol
from test_sparse import protocol_for
import test_freeze as F          ##reuse its build(); the network under test is not the point here


##F.build() returns the Run that _prep_run assembles; F.assemble() also returns the PhasicNet that .train() is called on.
def build(**kw):
    return F.build(**kw)


def make_data(n_input=F.N_INPUT, n_samples=16, T=30, n_classes=2, seed=0):
    ##16 samples so the train/val parts are big enough for CH (needs n > k) to be defined
    X = (np.random.default_rng(seed).random((n_samples * T, n_input)) < 0.3)
    return ModelInput.from_tonic([(X[i * T:(i + 1) * T], i % n_classes) for i in range(n_samples)])


def trained(*, epochs=1, val_split=None, **rc):
    ##one real trial on make_data(); test_fraction=0.25 like every other test file (the stored split is keyed on the dataset, not the fraction)
    cfg = dict(protocol=protocol_for(epochs=epochs, val_split=val_split), data=DataCfg(test_fraction=0.25)); cfg.update(rc)
    pnet, _ = F.assemble()
    return pnet.train(make_data(), RunCfg(**cfg), save_run=False)


# --------------------------------------------------------------------- 1. aggregation

def test_single_split_produces_one_fold():
    report = trained(epochs=1, val_split=0.25).report
    assert len(report["folds"]) == 1
    assert report["score"] == report["fold_scores"][0] and report["score_std"] == 0.0 and "refit" not in report


# --------------------------------------------------------------------- 2. fold independence

def test_each_fold_starts_from_the_as_built_weights():
    """Every fit starts from the as-built weights: a fit is a NEW MODEL."""
    pnet, run = F.assemble()
    W0 = run.ctx.syn.W.copy()

    ##spies on the real loop: two HoldoutValidation fits on the same Run must each see the as-built W in their first training epoch
    seen, fresh = [], [False]
    reset = run.reset_model
    def spy_reset():
        reset(); fresh[0] = True
    run.reset_model = spy_reset
    for _ in range(2):      ##a fresh protocol per fit: a second execute() of the SAME protocol skips its completed fold
        proto = HoldoutValidation(ProtocolCfg(epochs=1, val_split=0.25), run)
        run_epoch = proto.ops.run_epoch
        def spy_epoch(*a, run_epoch=run_epoch, **kw):        ##not `epoch=`: run_fit passes epoch= to run_epoch
            if fresh[0] and run.ctx.sim.mode == "train":
                seen.append(run.ctx.syn.W.copy()); fresh[0] = False
            return run_epoch(*a, **kw)
        proto.ops.run_epoch = spy_epoch
        proto.execute()
    assert len(seen) == 2, "expected one fresh model per fit"
    assert all(np.array_equal(W0, w) for w in seen), "a fit did not start from the as-built weights"
    assert not np.array_equal(W0, run.ctx.syn.W), "the run did not learn, so this proves nothing"
    run.reset_model = reset
    run.reset_model()
    assert np.array_equal(W0, run.ctx.syn.W), "reset_model must restore the as-built weights"


def test_fold_reset_also_clears_neuronal_state():
    run = trained(epochs=1)
    ##non-vacuity guard: the state is really dirty before the reset (V off rest, traces non-zero)
    assert not np.allclose(run.ctx.neur.V, run.integrator.v_rest) and run.plasticity.Pre_trace.any()
    run.reset_model()
    assert np.allclose(run.ctx.neur.V, run.integrator.v_rest)
    assert not run.ctx.neur.Spikes.any()
    assert not run.plasticity.Pre_trace.any() and not run.plasticity.Post_trace.any()      ##traces are module-owned state and reset with the model too


# --------------------------------------------------------------------- 3. checkpointing

def test_policy_keeps_the_best_and_min_delta_guards_churn():
    ctx = build().ctx
    policy = CheckpointPolicy(CheckpointCfg(select="best", min_improvement=0.5))

    policy.on_epoch_end(ctx, epoch=0, score=1.0)
    assert policy.best.score == 1.0
    policy.on_epoch_end(ctx, epoch=1, score=1.2)          # +0.2, inside min_delta
    assert policy.best.epoch == 0, "min_delta must suppress churn on noise"
    policy.on_epoch_end(ctx, epoch=2, score=2.0)          # clears min_delta
    assert policy.best.epoch == 2


def test_equal_score_does_not_displace_an_earlier_best():
    """At equal score the EARLIER epoch is the better model -- fewer epochs for the same result."""
    ctx = build().ctx
    policy = CheckpointPolicy(CheckpointCfg(select="best"))
    policy.on_epoch_end(ctx, epoch=0, score=1.0)
    policy.on_epoch_end(ctx, epoch=1, score=1.0)
    assert policy.best.epoch == 0


def test_greater_is_better_false_inverts_the_comparison():
    """So a LOSS can drive selection without a second code path."""
    ctx = build().ctx
    policy = CheckpointPolicy(CheckpointCfg(select="best", greater_is_better=False))
    policy.on_epoch_end(ctx, epoch=0, score=1.0)
    policy.on_epoch_end(ctx, epoch=1, score=0.5)
    assert policy.best.epoch == 1


def test_is_better_is_the_one_definition_shared_by_run_fit_and_the_policy(monkeypatch):
    """run_fit (patience, best_epoch) and CheckpointPolicy use the one utils._is_better, so a loss-type scorer and min_improvement agree."""
    from utils import _is_better
    assert _is_better(1.0, None, greater_is_better=False, min_improvement=0.0)
    assert _is_better(0.5, 1.0, greater_is_better=False, min_improvement=0.0) and not _is_better(1.5, 1.0, greater_is_better=False, min_improvement=0.0)
    assert not _is_better(1.05, 1.0, greater_is_better=True, min_improvement=0.1) and _is_better(1.2, 1.0, greater_is_better=True, min_improvement=0.1)
    assert not _is_better(1.0, 1.0, greater_is_better=True, min_improvement=0.0) and not _is_better(1.0, 1.0, greater_is_better=False, min_improvement=0.0)

    def scripted(scores):
        it = iter(scores)
        monkeypatch.setattr(Scorer, "separability", lambda self, responses, labels: next(it))

    ##a LOSS: epoch 1 (0.5) is the best for the fit AND the policy; 0.8 is one stale epoch -> patience=1 stops there
    scripted([1.0, 0.5, 0.8, 0.9])
    fold = trained(checkpoint=CheckpointCfg(select="best", greater_is_better=False, to_disk=False),
                   protocol=ProtocolCfg(epochs=4, val_split=0.25, patience=1)).report["folds"][0]
    assert (fold["best_epoch"], fold["best_score"], fold["score"]) == (1, 0.5, 0.5) and fold["checkpoints"]["best"]["epoch"] == 1
    assert fold["stopped"] == "patience" and fold["epochs_run"] == 3
    ##min_improvement: +0.05 is NOT an improvement for either (stale -> the patience stop), +0.2 is
    for patience, expect in ((None, (2, 1.2, 3)), (1, (0, 1.0, 2))):
        scripted([1.0, 1.05, 1.2])
        fold = trained(checkpoint=CheckpointCfg(select="best", min_improvement=0.1, to_disk=False),
                       protocol=ProtocolCfg(epochs=3, val_split=0.25, patience=patience)).report["folds"][0]
        assert (fold["best_epoch"], fold["best_score"], fold["epochs_run"]) == expect and fold["checkpoints"]["best"]["epoch"] == expect[0]




def test_snapshot_round_trips_the_model_state():
    """A checkpoint is a COPY, not a view: it restores W and the module traces after the run moved on."""
    run = build()
    ctx = run.ctx
    snap = take_snapshot(ctx, epoch=0, score=1.0, reason="best", modules=run.modules)      ##modules= so module-owned state (traces) is in the snapshot too
    W_at_snapshot = ctx.syn.W.copy()
    traces_at_snapshot = run.plasticity.Pre_trace.copy()

    HoldoutValidation(None, run).execute()
    assert not np.array_equal(W_at_snapshot, ctx.syn.W)
    assert run.plasticity.Pre_trace.any()                ##non-vacuity guard
    assert np.array_equal(snap.W, W_at_snapshot), "snapshot tracked the live array instead of copying"
    assert np.array_equal(snap.modules["plasticity"]["Pre_trace"], traces_at_snapshot), "snapshot tracked a live module array instead of copying"

    restore_snapshot(ctx, snap, modules=run.modules)
    assert np.array_equal(ctx.syn.W, W_at_snapshot)
    assert np.array_equal(run.plasticity.Pre_trace, traces_at_snapshot)


def test_snapshot_excludes_scratch_buffers():
    """dW/ltd/I2V/dV are fully overwritten every timestep, so copying them is pure waste."""
    run = build()
    snap = take_snapshot(run.ctx, epoch=0, score=None, reason="scheduled", modules=run.modules)
    ##scratch is looked for everywhere a snapshot stores arrays (upd, neur, each module's state_dict)
    stored = set(snap.upd) | set(snap.neur) | {k for state in snap.modules.values() for k in state}
    assert not stored & {"dW", "_ltd", "ltd", "I2V", "dV", "dV_th"}
    assert {"Pre_trace", "Post_trace"} <= set(snap.modules["plasticity"]), "traces ARE model state and must be kept"


def test_policy_cfg_is_archived_with_the_run():
    """The checkpoint policy cfg is part of the RESULT, so it is archived in the policy metadata and the run's report."""
    cfg = CheckpointCfg(select="best", content="weights", min_improvement=0.01, greater_is_better=False, to_disk=False)
    meta = CheckpointPolicy(cfg).metadata()
    assert meta["cfg"] == dataclasses.asdict(cfg)
    assert dataclasses.is_dataclass(cfg) and cfg.__dataclass_params__.frozen
    ##and it reaches the finished run's report (every fit's, and the run's own runcfg)
    report = trained(epochs=1, val_split=0.25, checkpoint=cfg).report
    assert report["folds"][0]["checkpoints"]["cfg"] == dataclasses.asdict(cfg) and report["runcfg"]["checkpoint"] == dataclasses.asdict(cfg)


def test_select_final_disables_best_tracking_entirely():
    ##select="final" (the default): nothing is snapshotted on improvement (the final state IS the model), but the epoch is still booked
    policy = CheckpointPolicy(CheckpointCfg(select="final"))
    assert policy.on_epoch_end(build().ctx, epoch=0, score=1.0) is None
    assert policy.best is None and policy.history == [{"epoch": 0, "score": 1.0, "kept": False}]


def test_select_best_finalize_loads_the_best_weights_back():
    run = build()
    pnet_ctx = run.ctx
    policy = CheckpointPolicy(CheckpointCfg(select="best"))
    policy.on_epoch_end(pnet_ctx, epoch=0, score=5.0, modules=run.modules)
    best_W = policy.best.W.copy()
    HoldoutValidation(None, run).execute()
    assert not np.array_equal(pnet_ctx.syn.W, best_W), "the run did not learn, so this proves nothing"
    policy.finalize(pnet_ctx, modules=run.modules)
    assert np.array_equal(pnet_ctx.syn.W, best_W)


def test_select_final_finalize_restores_nothing():
    ##under select="final" the policy holds no snapshot to restore, so the trained W stays
    run = build()
    policy = CheckpointPolicy(CheckpointCfg(select="final"))
    policy.on_epoch_end(run.ctx, epoch=0, score=5.0, modules=run.modules)
    HoldoutValidation(None, run).execute()
    trained_W = run.ctx.syn.W.copy()
    assert policy.finalize(run.ctx, modules=run.modules) is None
    assert np.array_equal(run.ctx.syn.W, trained_W)


# --------------------------------------------------------------------- 4. val handling

def test_val_pass_is_never_shuffled():
    """The val pass is never shuffled: shuffling it consumes RNG and breaks reproducibility."""
    ##each epoch shuffles the TRAIN set exactly once (never val), in the order of the keyed (fold, epoch) stream
    run = build()
    (train, val), = run._datahandler.fold_split(run.train_set, val_split=0.25, rng=np.random.default_rng(0), stratify=True)
    assert val is not None and len(val) > 0
    epochs, calls, orig = 2, [], run._datahandler.shuffled
    run._datahandler.shuffled = lambda data, rng: calls.append((data, orig(data, rng))) or calls[-1][1]
    HoldoutValidation(None, run).ops.run_fit(train, val, epochs=epochs, shuffle=True, fold=0)
    assert len(calls) == epochs and all(c[0] is train for c in calls), "the val pass is being shuffled"
    for e, (_, shuffled) in enumerate(calls):
        assert np.array_equal(shuffled.index, orig(train, run.ctx.rng.rng("fold", (0, e))).index)      ##keyed per (fold, epoch): the same order again
    assert not np.array_equal(calls[0][1].index, calls[1][1].index), "epochs 0 and 1 drew the same order, so the keying proves nothing"
    run._datahandler.shuffled = orig


def test_val_pass_collects_responses_for_scoring():
    run = build()
    summary = HoldoutValidation(None, run).ops.run_epoch(make_data(), shuffle=False, collect_responses=True)
    assert "responses" in summary
    assert summary["responses"].shape == (summary["samples"], run.ctx.net.n_total)
    assert summary["responses"].any(), "no neuron spiked, so the responses prove nothing"


def test_shuffle_without_rng_still_raises():
    """run_epoch with shuffle=True and no rng raises (reproducibility guard)."""
    run = build()
    with pytest.raises(ValueError):
        HoldoutValidation(None, run).ops.run_epoch(make_data(), shuffle=True, rng=None)


# --------------------------------------------------------------------- 5. scorer

def test_calinski_harabasz_matches_its_definition():
    """Separated data must score far above unstructured data of the same shape."""
    rng = np.random.default_rng(0)
    separated = np.vstack([rng.normal(0, 0.3, (20, 5)), rng.normal(5, 0.3, (20, 5))])
    labels = np.array([0] * 20 + [1] * 20)
    unstructured = rng.normal(0, 1, (40, 5))
    assert calinski_harabasz(separated, labels) > 100 * calinski_harabasz(unstructured, labels)


def test_calinski_harabasz_degenerate_cases_return_zero():
    """Degenerate CH cases (k<2, n<=k, zero within-dispersion) return 0.0, not infinity."""
    x = np.random.default_rng(0).normal(size=(10, 3))
    assert calinski_harabasz(x, np.zeros(10)) == 0.0                     # one class
    assert calinski_harabasz(np.ones((6, 3)), np.array([0, 0, 0, 1, 1, 1])) == 0.0   # no dispersion


def test_health_probes_flag_the_three_death_modes():
    ##collapse is decided on RATE alone: silence and runaway must flip the verdict; saturation is measured (weight_health) but does not vote
    scorer = Scorer(ScorerCfg(rate_hz_max=100.0, rate_hz_min=0.5))
    ctx = build().ctx
    n_rec, steps = ctx.net.n_rec, 1000                    ##1000 ms: rate_hz = spike_count / n_rec
    silent = scorer.health(ctx, steps=steps, spike_count=0, dt=1.0)
    runaway = scorer.health(ctx, steps=steps, spike_count=int(500 * n_rec), dt=1.0)      ##500 Hz > rate_hz_max
    healthy = scorer.health(ctx, steps=steps, spike_count=int(20 * n_rec), dt=1.0)       ##20 Hz
    assert scorer.is_collapsed(silent) and silent["rate_hz"] == 0.0                      # silence
    assert scorer.is_collapsed(runaway) and runaway["rate_hz"] > 100.0                   # runaway
    assert not scorer.is_collapsed(healthy)
    assert not scorer.is_collapsed({"collapsed": 0.0, "silent_row_frac": 0.9, "saturated_frac": 0.9})      # rate alone decides; the fractions are logged, not acted on
    ##saturation: every realized weight sits at a bound (|w| >= 1.0 for these weights, bounds +-1.0) -> the probe reads 1.0, a free-floating network reads 0.0
    pinned = scorer.weight_health(ctx, w_min=-1.0, w_max=1.0)
    free = scorer.weight_health(ctx, w_min=-100.0, w_max=100.0)
    assert pinned["saturated_frac"] == pytest.approx(1.0) and free["saturated_frac"] == 0.0


def test_health_probes_are_data_free_and_cheap():
    """Tier 1 must be runnable mid-epoch: no forward pass, no labels, no dataset."""
    ctx = build().ctx
    step_before, W_before = ctx.sim.current_runstep, ctx.syn.W.copy()
    probes = Scorer().health(ctx, steps=100, spike_count=50, dt=1.0)
    probes.update(Scorer().weight_health(ctx, w_min=-10.0, w_max=10.0))
    for key in ("rate_hz", "collapsed", "saturated_frac", "dead_syn_frac", "w_absmax"):
        assert key in probes and np.isfinite(probes[key])
    assert ctx.sim.current_runstep == step_before and np.array_equal(ctx.syn.W, W_before)


def test_dead_synapse_probe_counts_realized_zero_weights_only():
    ##non-edges of a dense W are 0.0 too and must NOT count as dead
    ctx = build().ctx
    syn = ctx.syn
    assert Scorer().weight_health(ctx, w_min=-10.0, w_max=10.0)["dead_syn_frac"] == 0.0
    edges = np.flatnonzero(syn.weights.syn_mask)
    k = edges.size // 4
    assert k > 0
    syn.W.flat[edges[:k]] = 0.0
    assert Scorer().weight_health(ctx, w_min=-10.0, w_max=10.0)["dead_syn_frac"] == pytest.approx(k / ctx.net.n_synapses)


def test_epoch_log_records_probes_beside_the_score():
    """The epoch log records the health probes beside the score (0.0 from a dead network and from a uniform one differ)."""
    report = trained(epochs=2, val_split=0.25).report
    entry = report["folds"][0]["epochs"][0]
    for key in ("score", "collapsed", "rate_hz", "saturated_frac"):
        assert key in entry
    assert entry["score"] is not None                    ##non-vacuity guard
