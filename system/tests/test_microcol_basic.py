#test_microcol_basic -- the back-to-basics microcolumn experiment (microcol_basic*) and the ModelInput column blocks.
# WHAT IS TESTED:
#   1. ModelInput column_blocks  -- every column's inputs are one image block, the round trip, the dataset id, a wrong column count
#   2. EscapeThreshold           -- fixed threshold per group, refractory period, the escape-noise probability, a new stream per sample
#   3. SoftBoundSTDP             -- the rule and the eligibility sums against a plain loop over the edges; reward / punishment; the decision
#                                   bound and the normalisation leave the fixed (inhibitory) weights alone
#   4. the protocol              -- runs end to end on a toy network, each stage changes only its own group, killed and resumed it reports
#                                   what a straight run reports (with noise on), the sweep records are there
#   5. the two experiment files  -- build the network of the header; search values reach the cfg and the weights
# Run: pytest system/tests/test_microcol_basic.py -q
from __future__ import annotations
import json
from dataclasses import replace

import numpy as np
import pytest

from phasicnet import PhasicNet
from modelinput import ModelInput
from dynamicscfg import DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg
from checkpoint import CheckpointPolicy, snapshots
from microcol_basic import EscapeThresholdCfg, SoftBoundSTDPCfg, MICROCOL_BASIC
from microcol_basic_protocol import BasicCfg, BasicProtocol

T = 20
FIXED = ("L1_ei", "L1_ie", "L2_ei", "L2_ie", "L3_ei", "L3_ie", "fb_L2_L1", "fb_L3_L2")


def tiny(seed: int = 3, *, beta: float = 0.0, normalize: bool = True) -> PhasicNet:
    """The experiment in miniature: input 2 x 4 x 4 -> L1 16 + 4 -> L2 12 + 4 -> L3 8 + 4 (2 classes x 4), 4 columns, the same synapse groups."""
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    b = pnet.builder
    b.set_neurons(("input", 32, True), ("L1_exc", 16), ("L1_inh", 4, False, True), ("L2_exc", 12), ("L2_inh", 4, False, True),
                  ("L3_exc", 8), ("L3_inh", 4, False, True))
    b.set_columns(4)
    b.set_synapses(("in_L1", "input", "L1_exc"), ("L1_ei", "L1_exc", "L1_inh"), ("L1_ie", "L1_inh", "L1_exc"),
                   ("L1_L2", "L1_exc", "L2_exc"), ("L2_ei", "L2_exc", "L2_inh"), ("L2_ie", "L2_inh", "L2_exc"), ("fb_L2_L1", "L2_exc", "L1_inh"),
                   ("L2_L3", "L2_exc", "L3_exc", True), ("L3_ei", "L3_exc", "L3_inh", True), ("L3_ie", "L3_inh", "L3_exc", True),
                   ("fb_L3_L2", "L3_exc", "L2_inh", True))
    b.set_connectivity({s: 1.0 for s in ("in_L1", "L1_L2", "L2_L3") + FIXED})
    b.set_weights(in_L1=(0.2, 0.8), L1_L2=(0.2, 0.8), L2_L3=(0.2, 0.8), L1_ei=0.5, L2_ei=0.5, L3_ei=0.5, L1_ie=-0.2, L2_ie=-0.2, L3_ie=-0.2,
                  fb_L2_L1=0.3, fb_L3_L2=0.3)
    b.realize(network_name="tiny_basic", post_offset=True, allow_autapse=False, save=False)
    pnet.cfg = DynamicsCfg(simulation=SimulationCfg(dt=1.0, weights_format="sparse", update_on="hybrid_event"), synapse=SynapseCfg(b=None),
                           integrator=IntegratorCfg(tau_mem=20.0, v_rest=0.0, r=20.0),
                           threshold=EscapeThresholdCfg(v_reset=0.0, v_th=1.0, tau_th=None, adpt_th=None, t_refrac=2.0, beta=beta,
                                                        group_v_th={"L1_exc": 4.0, "L2_exc": 1.5, "L3_exc": 2.0}),
                           plasticity=SoftBoundSTDPCfg(normalize=normalize, freeze=FIXED),
                           noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0), phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(replace(MICROCOL_BASIC, protocol_classes=(BasicProtocol,)))
    pnet.cfg_name = "tiny_basic"
    return pnet


def event_data(n: int, seed: int) -> ModelInput:
    """n toy samples (T, 2, 4, 4) bool: class 0 has its events in the upper half of the image, class 1 in the lower half."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = np.full((2, 4, 4), 0.03)
        p[:, :2] += 0.15 * (i % 2 == 0)
        p[:, 2:] += 0.15 * (i % 2 == 1)
        out.append((rng.random((T, 2, 4, 4)) < p, i % 2))
    return ModelInput.from_tonic(out, axis_labels="TPHW", column_blocks=(2, 2))


def runcfg(**protocol) -> RunCfg:
    p = dict(t_train_ms=float(T), t_state_ms=float(T), stage_epochs=(1, 1), rstdp_epochs=2, rstdp_batch=5, n_classes=2, n_per_class=4, n_init=2,
             val_split=0.25, rate_probe=3, report_every=100, health_every=5, trajectory_every=6)
    return RunCfg(data=DataCfg(make_test_split=False, test_fraction=0.0, shuffle=False), scorer=ScorerCfg(rate_hz_min=0.0),
                  checkpoint=CheckpointCfg(to_disk=False, every=protocol.pop("every", None), keep_last=None),
                  history=HistoryCfg(include_input=False, record_spikes=False, record_counts=True), protocol=BasicCfg(**{**p, **protocol}))


def prepped(**kw):
    """(Run, protocol) on 16 toy training samples, nothing run yet."""
    run = tiny(**kw)._prep_run(event_data(16, 0), runcfg(), test_data=event_data(6, 1))
    return run, BasicProtocol(run.runcfg.protocol, run)


def group_edges(run, name: str) -> np.ndarray:
    """The edges of one synapse group in syn.W."""
    syn, p = run.ctx.syn, run.ctx.net.synapseslices[name]
    return np.flatnonzero((syn.post >= p.post.start) & (syn.post < p.post.stop) & (syn.pre >= p.pre.start) & (syn.pre < p.pre.stop))


# --------------------------------------------------------------------- 1. ModelInput column_blocks

def test_column_blocks_give_each_column_one_image_block():
    """(2, 2) blocks on a (2, 6, 4) frame: neurons 0..11 = the upper left 3 x 2 block of both polarities, then upper right, lower left, lower right."""
    frames = np.random.default_rng(0).random((5, 2, 6, 4)) < 0.3
    plain = ModelInput.from_tonic([(frames, 1)], axis_labels="TPHW")
    block = ModelInput.from_tonic([(frames, 1)], axis_labels="TPHW", column_blocks=(2, 2))
    got, _ = block[0]
    assert got.shape == (5, 48) and got.sum() == frames.sum()
    for k, (rows, cols) in enumerate(((slice(0, 3), slice(0, 2)), (slice(0, 3), slice(2, 4)), (slice(3, 6), slice(0, 2)), (slice(3, 6), slice(2, 4)))):
        assert np.array_equal(got[:, 12 * k:12 * (k + 1)], frames[:, :, rows, cols].reshape(5, -1))
        assert all(block.neuron_coordinates(j)[1:] in [(h, w) for h in range(rows.start, rows.stop) for w in range(cols.start, cols.stop)]
                   for j in range(12 * k, 12 * (k + 1)))
    assert all(block.neuron_index(*block.neuron_coordinates(j)) == j for j in range(48))
    assert np.array_equal(block.reconstruct_step(got[2]), frames[2])
    assert block.dataset_hash != plain.dataset_hash                         ##another neuron order = another dataset to a run
    with pytest.raises(ValueError):
        ModelInput.from_tonic([(frames, 1)], axis_labels="TPHW", column_blocks=(4, 2))        ##6 rows do not split into 4 blocks
    pnet = PhasicNet(1)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", 48, True), ("exc", 12)).set_columns(3)
    pnet.builder.set_synapses(("in_exc", "input", "exc")).set_connectivity(in_exc=1.0).set_weights(in_exc=0.5)
    net = pnet.builder.realize(network_name="blocks", post_offset=True, allow_autapse=False, save=False)
    with pytest.raises(ValueError):
        block.validate_network(net)                                     ##4 blocks, 3 columns


# --------------------------------------------------------------------- 2. EscapeThreshold

def test_fixed_threshold_per_group_and_refractory_period():
    run, _ = prepped()
    ctx, th = run.ctx, run.threshold
    net, neur, upd = ctx.net, ctx.neur, ctx.upd
    want_th = np.full(net.n_post, 1.0, dtype=np.float32)                ##v_th = the groups not listed (inh)
    for g, v in (("L1_exc", 4.0), ("L2_exc", 1.5), ("L3_exc", 2.0)):
        want_th[net.neuronrows[g]] = v
    neur.V[:] = 1.7
    neur.Since[:] = 100
    th.step(ctx)
    assert np.array_equal(upd.rec_spikes, 1.7 >= want_th) and 0 < upd.rec_spikes.sum() < net.n_post
    neur.V[:] = 5.0
    neur.Since[:] = np.arange(net.n_post) % 4                           ##0 / 1 = inside the 2 ms refractory period
    th.step(ctx)
    assert np.array_equal(upd.rec_spikes, neur.Since >= 2)


def test_escape_noise_probability_and_one_stream_per_sample():
    run, _ = prepped(beta=0.125)                                        ##L1_exc: width 0.125 x (4 - 0) = 0.5
    ctx, th = run.ctx, run.threshold
    neur, upd, rows = ctx.neur, ctx.upd, ctx.net.neuronrows["L1_exc"]
    neur.Since[:] = 100
    hits = np.zeros(3)
    for j, v in enumerate((3.5, 4.0, 4.5)):                             ##(V - 4) / 0.5 = -1, 0, 1
        neur.V[:] = v
        for _ in range(400):
            th.step(ctx)
            hits[j] += upd.rec_spikes[rows].mean()
    assert np.allclose(hits / 400, 1 / (1 + np.exp(-np.array([-1.0, 0.0, 1.0]))), atol=0.02)
    draws = []
    for _ in range(2):                                                  ##reset_transient -> the same runstep's stream again: the same draws
        run.reset_transient()
        neur.V[:] = 4.0
        neur.Since[:] = 100
        th.step(ctx)
        draws.append(upd.rec_spikes.copy())
    assert np.array_equal(*draws) and 0 < draws[0].sum() < draws[0].size
    with pytest.raises(ValueError):
        tiny(beta=-1.0)._prep_run(event_data(16, 0), runcfg(), test_data=event_data(6, 1))


# --------------------------------------------------------------------- 3. SoftBoundSTDP

def test_soft_bound_rule_and_eligibility_against_a_plain_loop():
    """Random traces and spikes, in_L1 + L2_L3 plastic: step() returns, for every plastic edge with a spiking end,
    (A_ltp post_spike Pre_trace - A_ltd Post_trace pre_spike) w (1 - w); on the decision edges 0, the two terms go into Elig_*."""
    run, _ = prepped()
    ctx, p = run.ctx, run.plasticity
    syn, neur, rng = ctx.syn, ctx.neur, np.random.default_rng(4)
    run.set_mode("train")
    run.set_frozen(tuple(s for s in ctx.net.synapsespecs if s not in ("in_L1", "L2_L3")))
    p.Pre_trace[:] = rng.random(p.Pre_trace.size)
    p.Post_trace[:] = rng.random(p.Post_trace.size)
    neur.pre_spike[:] = rng.random(neur.pre_spike.size) < 0.4
    neur.post_spike[:] = rng.random(neur.post_spike.size) < 0.4
    pre_tr, post_tr = p.Pre_trace.copy(), p.Post_trace.copy()
    k, dw = p.step(ctx)
    dec, plain = set(group_edges(run, "L2_L3").tolist()), set(group_edges(run, "in_L1").tolist())
    want, elig_p, elig_n = {}, np.zeros(syn.W.size), np.zeros(syn.W.size)
    for e in sorted(dec | plain):
        i, j = int(syn.pre[e]), int(syn.post[e])
        if not (neur.pre_spike[i] or neur.post_spike[j]):
            continue
        ltp, ltd = float(neur.post_spike[j] * pre_tr[i]), float(post_tr[j] * neur.pre_spike[i])
        if e in dec:
            want[e], elig_p[e], elig_n[e] = 0.0, ltp, ltd
        else:
            want[e] = (0.01 * ltp - 0.005 * ltd) * float(syn.W[e]) * (1.0 - float(syn.W[e]))
    assert sorted(k.tolist()) == sorted(want) and len(want) > 20
    assert np.allclose(dw, [want[e] for e in k.tolist()], atol=1e-7)
    assert np.allclose(p.Elig_ltp, elig_p, atol=1e-7) and np.allclose(p.Elig_ltd, elig_n, atol=1e-7) and elig_p.sum() > 0
    decay = np.exp(-1.0 / 20.0)                                         ##the traces after the rule: decayed, + this step's spikes
    assert np.allclose(p.Pre_trace, pre_tr * decay + neur.pre_spike, atol=1e-6) and np.allclose(p.Post_trace, post_tr * decay + neur.post_spike, atol=1e-6)


def test_reward_punishment_bound_and_normalisation_leave_the_fixed_weights():
    run, _ = prepped()
    ctx, p = run.ctx, run.plasticity
    syn, net, rng = ctx.syn, ctx.net, np.random.default_rng(5)
    fixed = np.concatenate([group_edges(run, s) for s in FIXED])
    W_built = syn.W.copy()
    assert (W_built[group_edges(run, "L3_ie")] < 0).all()
    run.set_mode("train")
    run.set_frozen(tuple(s for s in net.synapsespecs if s != "L2_L3"))
    p.bound_decision(ctx)
    assert np.array_equal(syn.W, W_built)                               ##plastic decision weights are inside [0, 1]; the inhibition onto L3 stays
    rows = net.neuronrows["L3_exc"].start + np.array([0, 2])
    k = np.flatnonzero(np.isin(syn.post, rows) & np.isin(np.arange(syn.W.size), group_edges(run, "L2_L3")))
    p.Elig_ltp[:] = rng.random(syn.W.size)
    p.Elig_ltd[:] = rng.random(syn.W.size)
    ep, en = p.Elig_ltp[k].astype(np.float64), p.Elig_ltd[k].astype(np.float64)
    for reward, change in ((True, 0.5 * (0.004 * ep - 0.003 * en)), (False, 0.5 * (0.0005 * en - 0.004 * ep))):
        w = syn.W[k].astype(np.float64)
        rest = np.setdiff1d(np.arange(syn.W.size), k)
        before = syn.W[rest].copy()
        assert p.apply_reward(ctx, rows, reward, 0.5) == k.size == 2 * 12
        assert np.allclose(syn.W[k], w + change * w * (1 - w), atol=1e-7) and np.array_equal(syn.W[rest], before)
    ##normalisation: the plastic non-decision edges of every neuron back to their built sum
    run.set_frozen(tuple(s for s in net.synapsespecs if s != "in_L1"))
    k1 = group_edges(run, "in_L1")
    built = np.bincount(syn.post[k1], weights=W_built[k1], minlength=net.n_post)
    syn.W[k1] *= rng.uniform(0.5, 1.2, k1.size).astype(np.float32)
    dec_before = syn.W[group_edges(run, "L2_L3")].copy()
    p.normalize_weights(ctx)
    assert np.allclose(np.bincount(syn.post[k1], weights=syn.W[k1], minlength=net.n_post), built, rtol=1e-5)
    assert np.array_equal(syn.W[fixed], W_built[fixed]) and np.array_equal(syn.W[group_edges(run, "L2_L3")], dec_before)
    run.set_mode("eval")
    for call in (lambda: p.apply_reward(ctx, rows, True, 1.0), lambda: p.normalize_weights(ctx)):
        with pytest.raises(RuntimeError):
            call()


# --------------------------------------------------------------------- 4. the protocol

def _train(out, every=None, resume_from=None, beta=0.15, **kw):
    return tiny(beta=beta).train(event_data(16, 0), runcfg(every=every, **kw), test_data=event_data(6, 1), out_dir=out, resume_from=resume_from, save_run=False)


KEYS = ("val_decision_acc", "test_decision_acc", "rstdp_train_acc", "rstdp_val_acc", "rstdp_best_pass", "val_ami_L1_exc", "test_ami_L3_exc",
        "val_ami_L3_exc_untrained", "val_ami_gain_L3_exc", "rate_hz_L1_inh_untrained", "rate_hz_L2_inh", "rate_hz_L3_exc_cols", "dead_frac_L2_exc",
        "decision_top_share", "decision_tie_frac", "decision_confusion", "w_moved_in_L1", "w_hist_L2_L3", "flag_silent", "flag_runaway", "flag_runaway_inh",
        "flag_no_learning", "health", "initial_rate_hz")


def test_protocol_runs_records_and_resumes_bit_identically(tmp_path, monkeypatch):
    """Straight run (escape noise on) vs killed after its first snapshot and resumed, and vs resumed inside the first R-STDP pass: the same
    report, the same responses.npz. The stages change only their own group; the sweep records are complete."""
    full = _train(tmp_path / "full", every=5)
    r = full.report
    assert r["n_fit"] == 12 and r["n_val"] == 4 and r["n_test"] == 6 and len(r["rstdp_val_acc"]) == 2
    h = r["health"]
    assert h["at"] == [[k, i] for k in range(4) for i in (5, 10, 12)]                        ##4 passes, a record every 5 samples and at the end
    assert set(h["neurons"]) == set(full.ctx.net.neuronslices) and set(h["weights"]) == {"in_L1", "L1_L2", "L2_L3"}
    change = {s: np.array(h["weights"][s]["change"]) for s in h["weights"]}
    assert change["in_L1"][:3].min() > 0 and change["in_L1"][3:].max() == 0                 ##in_L1 moves in pass 0 only
    assert change["L1_L2"][3:6].min() > 0 and change["L1_L2"][:3].max() == 0 and change["L1_L2"][6:].max() == 0
    assert change["L2_L3"][:6].max() == 0 and change["L2_L3"][6:].max() > 0
    assert max(h["neurons"]["L1_exc"]["rate_hz"]) > 0 and r["rate_hz_L1_exc_untrained"] > 0 and r["rate_hz_input"] > 0
    built = np.load(tmp_path / "full" / "network.npz")
    order = np.lexsort((built["pre_idx"], built["post_idx"]))
    for s in FIXED:                                                                           ##inhibition and feedback inhibition: as built
        k = group_edges(full, s)
        assert np.array_equal(full.ctx.syn.W[k], built["W0"][order][k])
    z = np.load(tmp_path / "full" / "responses.npz")
    assert z["resp_untr"].shape == z["resp_val"].shape and z["health"].shape == (12, 7, 3) and z["wstat"].shape == (12, 3, 5)
    names = [p.name for p in snapshots(tmp_path / "full")]
    assert names[0] == "snap_s00_i00000005.npz" and "snap_s02_i00000005.npz" in names      ##stage 2 = the first R-STDP pass
    orig = CheckpointPolicy.on_sample_end

    def kill_after_first(policy, **k):
        path = orig(policy, **k)
        if path is not None:
            raise KeyboardInterrupt
        return path
    monkeypatch.setattr(CheckpointPolicy, "on_sample_end", kill_after_first)
    with pytest.raises(KeyboardInterrupt):
        _train(tmp_path / "killed", every=5)
    monkeypatch.undo()
    assert json.loads((tmp_path / "killed" / "status.json").read_text())["status"] == "interrupted"
    runs = (_train(tmp_path / "killed", every=5, resume_from=tmp_path / "killed" / names[0]),
            _train(tmp_path / "inside", every=5, resume_from=tmp_path / "full" / "snap_s02_i00000005.npz"))
    for back, folder in zip(runs, ("killed", "inside")):
        assert [back.report[k] for k in KEYS] == [r[k] for k in KEYS], [k for k in KEYS if back.report[k] != r[k]]
        assert np.array_equal(back.ctx.syn.W, full.ctx.syn.W)
        zb = np.load(tmp_path / folder / "responses.npz")
        assert z.files == zb.files and all(np.array_equal(z[k], zb[k]) for k in z.files)


def test_untrained_run_and_wrong_decision_group(tmp_path):
    """No training pass at all (the rate search), no noise: the weights stay as built, trained = untrained on the validation set."""
    run = _train(tmp_path / "untrained", beta=0.0, stage_epochs=(0, 0), rstdp_epochs=0)
    r = run.report
    assert r["rstdp_val_acc"] == [] and r["flag_no_learning"] == 1 and r["health"] == {} and r["w_moved_in_L1"] == 0.0
    assert r["val_ami_gain_L3_exc"] == 0.0 and r["rate_hz_L2_inh_val"] == r["rate_hz_L2_inh_untrained"]
    z = np.load(tmp_path / "untrained" / "responses.npz")
    assert np.array_equal(z["resp_untr"], z["resp_val"])
    run, _ = prepped()
    for bad in (dict(n_classes=3), dict(decision_group="L2_exc", cluster_groups=("L1_exc", "L3_exc"))):
        with pytest.raises(ValueError):
            BasicProtocol(replace(run.runcfg.protocol, **bad), run)


# --------------------------------------------------------------------- 5. the two experiment files

def test_experiment_files_build_the_network_and_take_search_values():
    import microcol_basic_main as m
    net = m.build().network
    sizes = {g: s.stop - s.start for g, s in net.neuronslices.items()}
    assert sizes == {"input": 2312, "L1_exc": 640, "L1_inh": 160, "L2_exc": 400, "L2_inh": 100, "L3_exc": 240, "L3_inh": 60} and net.n_columns == 4
    order = np.lexsort((net.pre_idx, net.post_idx))
    pre, post, kind = net.pre_idx[order], net.post_idx[order] + net.row_offset, net.syntype[order]
    for s in net.synapsespecs:
        crossing = (net.column_ids[pre[kind == net.type_key[s]]] != net.column_ids[post[kind == net.type_key[s]]]).any()
        assert crossing == (s in ("L2_L3", "L3_ei", "L3_ie", "fb_L3_L2")), s
    p = {"th_L1": 11.0, "th_L3": 4.0, "th_inh": 7.0, "beta": 0.2, "w_ie": 0.5, "w_fb": 0.0, "A_ltp": 0.02, "ltd_ratio": 0.5, "reward": 10.0, "normalize": False}
    job = m.params_job(p, out="unused", n_train=50, n_test=20)
    v, w = job.build_kw["values"], job.build_kw["weights"]
    assert v["dynamics.threshold.group_v_th"] == {"L1_exc": 11.0, "L2_exc": 5.0, "L3_exc": 4.0} and v["dynamics.threshold.beta"] == 0.2
    assert v["dynamics.threshold.v_th"] == 7.0
    assert v["dynamics.plasticity.A_ltd"] == pytest.approx(0.01) and v["dynamics.plasticity.Ar_ltp"] == pytest.approx(0.04)
    assert v["dynamics.plasticity.normalize"] is False and v["run.data.n_train"] == 50
    assert w == {"L1_ie": -0.5, "L2_ie": -0.5, "L3_ie": -0.5, "fb_L2_L1": 0.0, "fb_L3_L2": 0.0}
    assert job.runcfg.protocol.rstdp_epochs == 3 and job.runcfg.protocol.trajectory_every is None
    off = m.params_job({}, out="unused", train=False)
    assert tuple(off.runcfg.protocol.stage_epochs) == (0, 0) and off.runcfg.protocol.rstdp_epochs == 0 and off.build_kw["weights"] is None
