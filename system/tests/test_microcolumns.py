#test_microcolumns -- the layered microcolumn experiments and the system parts they added.
# WHAT IS TESTED:
#   1. builder kernel "column"   -- each column reads its own patch (rf / stride), the within-column patterns, the network file round trip
#   2. ExperimentCfg             -- one file holds dynamics + run, its cfg_id is the run's RunMeta.cfg_id, override by dotted path
#   3. Run.set_frozen            -- learning moves from one layer to the next mid-run (plastic mask + clamp plan)
#   4. clustering scores         -- ARI / AMI against reference values, k-means on separable data
#   5. variant B dynamics        -- one winner per column while a layer learns, the STDP / R-STDP rules, the decision bound
#   6. the protocols             -- B runs end to end on a toy network; killed and resumed (also inside R-STDP) it reports what a straight run
#                                   reports; A's stages each change only their own groups
#   7. the DoG latency encoder   -- against a plain per-pixel loop
# Run: pytest system/tests/test_microcolumns.py -q
from __future__ import annotations
import json
from dataclasses import replace

import numpy as np
import pytest

from phasicnet import PhasicNet
from modelinput import ModelInput
from dynamicscfg import DynamicsCfg, SimulationCfg, SynapseCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ExperimentCfg
from checkpoint import CheckpointPolicy, snapshots
from scorer import kmeans, adjusted_rand_index, adjusted_mutual_info, cluster_scores
from microcol_mozafari import IFIntegratorCfg, GroupThresholdCfg, MozafariSTDPCfg, MICROCOL_MOZAFARI
from microcol_mozafari_protocol import MozafariCfg, MozafariProtocol
from preprocessing import dog_kernel, dog_latency_prep, DoGLatencyDataset, MOZAFARI_DOG

T, BINS = 6, 4      ##toy samples: 4 latency bins + 2 flush steps


def tiny_b(seed: int = 5, *, w=(0.7, 0.9), freeze=(), kwta=None) -> PhasicNet:
    """Variant B in miniature: input (8, 8, 2) -> L1 (4, 4, 3) -> L2 (2, 2, 4) -> L3 (1, 1, 4) = 2 classes x 2, every projection rf 2 stride 2.
    kwta None = the authors' default (k 5 radius 3: ONE winner on a 4 x 4 grid); {} = every column winner updates."""
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    b = pnet.builder
    b.set_neurons(("input", 8 * 8 * 2, True), ("L1_exc", 4 * 4 * 3), ("L2_exc", 2 * 2 * 4), ("L3_exc", 4))
    b.set_positions(input=(8, 8, 2), L1_exc=(4, 4, 3), L2_exc=(2, 2, 4), L3_exc=(1, 1, 4))
    b.set_synapses(("in_L1", "input", "L1_exc"), ("L1_L2", "L1_exc", "L2_exc"), ("L2_L3", "L2_exc", "L3_exc"))
    col = {"kernel": "column", "p": 1.0, "rf": 2, "stride": 2}
    b.set_connectivity(in_L1=col, L1_L2=col, L2_L3=col)
    b.set_weights(in_L1=w, L1_L2=w, L2_L3=w)
    b.realize(network_name="tiny_b", post_offset=True, allow_autapse=False, save=False)
    pnet.cfg = DynamicsCfg(simulation=SimulationCfg(dt=1.0, weights_format="sparse", update_on="hybrid_event"), synapse=SynapseCfg(b=None),
                           integrator=IFIntegratorCfg(), plasticity=MozafariSTDPCfg(rate_double_every=4, freeze=freeze, **({} if kwta is None else {"kwta": kwta})),
                           threshold=GroupThresholdCfg(v_reset=0.0, v_th=2.0, group_v_th={"L3_exc": None}, wta_groups=("L1_exc", "L2_exc")),
                           noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0), phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(replace(MICROCOL_MOZAFARI, protocol_classes=(MozafariProtocol,)))
    pnet.cfg_name = "tiny_b"
    return pnet


def latency_data(n: int, seed: int) -> ModelInput:
    """n toy samples (T, 128) bool: half the inputs spike once, at a random bin; class 1 shifts its spikes to the second half of the grid."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = np.zeros((T, 128), dtype=bool)
        on = np.flatnonzero(rng.random(128) < (0.35 + 0.3 * ((np.arange(128) >= 64) == i % 2)))
        f[rng.integers(0, BINS, on.size), on] = True
        out.append((f, i % 2))
    return ModelInput.from_tonic(out)


def runcfg(**protocol) -> RunCfg:
    p = dict(t_train_ms=float(T), t_state_ms=float(T), stage_epochs=(1, 1), rstdp_epochs=2, rstdp_batch=5, n_per_class=2, n_init=2, val_split=0.25,
             rate_probe=3, report_every=100)
    return RunCfg(data=DataCfg(make_test_split=False, test_fraction=0.0, shuffle=False), scorer=ScorerCfg(rate_hz_min=0.0),
                  checkpoint=CheckpointCfg(to_disk=False, every=protocol.pop("every", None), keep_last=None),
                  history=HistoryCfg(include_input=False, record_spikes=False, record_counts=True), protocol=MozafariCfg(**{**p, **protocol}))


def prepped(**kw):
    """(pnet, Run, protocol) on 16 toy training samples, nothing run yet."""
    pnet = tiny_b(**kw)
    run = pnet._prep_run(latency_data(16, 0), runcfg(), test_data=latency_data(6, 1))
    return pnet, run, MozafariProtocol(run.runcfg.protocol, run)


def edges(net, syn: str):
    """(pre neuron, post neuron) of one synapse group, both in neuron space (true_neuronID)."""
    m = net.syntype == net.type_key[syn]
    return net.pre_idx[m], net.post_idx[m] + net.row_offset


# --------------------------------------------------------------------- 1. builder kernel "column"

def test_column_kernel_reads_each_columns_own_patch():
    """L1 column (x, y) reads input rows 2x..2x+1, columns 2y..2y+1, every channel; L2 column reads its 2 x 2 block of L1 columns."""
    pnet = tiny_b()
    net = pnet.network
    pos = {g: np.asarray(net.positions[g]) for g in ("input", "L1_exc", "L2_exc")}
    for syn, pre_g, post_g in (("in_L1", "input", "L1_exc"), ("L1_L2", "L1_exc", "L2_exc")):
        pre, post = edges(net, syn)
        pre_xy = pos[pre_g][pre - net.neuronslices[pre_g].start][:, :2]
        post_xy = pos[post_g][post - net.neuronslices[post_g].start][:, :2]
        assert np.all((pre_xy >= 2 * post_xy) & (pre_xy < 2 * post_xy + 2))
        per_post = np.bincount(post - net.neuronslices[post_g].start)
        fan_in = 2 * 2 * pos[pre_g].shape[0] // np.prod(pos[pre_g].max(axis=0)[:2] + 1)
        assert per_post.min() == per_post.max() == fan_in                        ##p = 1: the whole patch, every neuron of it
    assert net.synapse_counts == {"n_in_L1": 48 * 8, "n_L1_L2": 16 * 12, "n_L2_L3": 4 * 16}


def test_column_patterns_and_the_network_file_round_trip(tmp_path):
    """Within a group: one_to_one = the same slot of the same column (minus the autapse), all_but_self = the column's other neurons;
    save_specs -> load_specs rebuilds the same network."""
    pnet = PhasicNet(3)
    pnet._instantiate_builder()
    b = pnet.builder
    b.set_neurons(("input", 4, True), ("A", 2 * 2 * 3), ("B", 2 * 2 * 3))
    b.set_positions(input=(2, 2), A=(2, 2, 3), B=(2, 2, 3))
    b.set_synapses(("in_A", "input", "A"), ("A_A", "A", "A"), ("A_B", "A", "B"))
    b.set_connectivity(in_A={"kernel": "column", "p": 1.0}, A_A={"kernel": "column", "p": 1.0, "pattern": "all_but_self"},
                       A_B={"kernel": "column", "p": 1.0, "pattern": "one_to_one"})
    b.set_weights(in_A=1.0, A_A=0.5, A_B=0.5)
    net = b.realize(network_name="pat", post_offset=True, allow_autapse=False, save=False)
    assert net.synapse_counts == {"n_in_A": 12, "n_A_A": 4 * 3 * 2, "n_A_B": 12}
    pre, post = edges(net, "A_B")
    assert np.array_equal(pre - net.neuronslices["A"].start, post - net.neuronslices["B"].start)
    b.save_specs(filename="pat_net", spec_name="pat", YAML=True, save_to=tmp_path)
    again = PhasicNet(3)
    again._instantiate_builder()
    again.builder.load_specs(tmp_path / "pat_net.yaml")
    net2 = again.builder.realize(network_name="pat", post_offset=True, allow_autapse=False, save=False)
    assert net2.network_id == net.network_id


def test_column_kernel_rejects_half_a_receptive_field():
    pnet = tiny_b()
    with pytest.raises(ValueError):
        pnet.builder.set_connectivity(in_L1={"kernel": "column", "p": 1.0, "rf": 2})


# --------------------------------------------------------------------- 2. ExperimentCfg

def test_experiment_file_round_trip_and_cfg_id(tmp_path):
    """save_experiment -> ExperimentCfg.load_file gives the same dynamics + run; the file's cfg_id is the run's RunMeta.cfg_id."""
    pnet = tiny_b()
    pnet.runcfg = runcfg()
    pnet.save_experiment(filename="exp", cfg_name="tiny", save_to=tmp_path)
    exp = ExperimentCfg.load_file(tmp_path / "exp.yaml")
    assert exp.cfg_name == "tiny" and isinstance(exp.run.protocol, MozafariCfg)
    as_saved = lambda e: json.loads(json.dumps(e._canonicalize()))                    ##YAML gives lists where the dataclasses hold tuples
    assert as_saved(exp) == as_saved(ExperimentCfg(dynamics=pnet.cfg, run=pnet.runcfg, cfg_name="tiny"))
    import yaml
    cfg_id = yaml.safe_load((tmp_path / "exp.yaml").read_text())["cfg_id"]
    run = pnet._prep_run(latency_data(16, 0), exp.run, test_data=latency_data(6, 1))
    assert cfg_id == run.runmeta.cfg_id


def test_experiment_override_by_dotted_path():
    exp = ExperimentCfg(dynamics=tiny_b().cfg, run=runcfg(), cfg_name="tiny")
    new = exp.override({"dynamics.plasticity.A_ltp": 0.01, "run.protocol.rstdp_epochs": 7, "run.data.n_train": 10})
    assert (new.dynamics.plasticity.A_ltp, new.run.protocol.rstdp_epochs, new.run.data.n_train) == (0.01, 7, 10)
    assert new.dynamics.threshold == exp.dynamics.threshold and exp.dynamics.plasticity.A_ltp == 0.004      ##the rest untouched, the original too
    assert exp.override(None) is exp
    with pytest.raises(TypeError):
        exp.override({"dynamics.plasticity.no_such_field": 1})


# --------------------------------------------------------------------- 3. Run.set_frozen

def test_set_frozen_moves_learning_from_layer_to_layer():
    _, run, _ = prepped()
    net, syn = run.ctx.net, run.ctx.syn
    for learn, bounds in ((("in_L1",), (0.0, 1.0)), (("L2_L3",), (0.2, 0.8))):
        run.set_frozen(tuple(s for s in net.synapsespecs if s not in learn))
        proj = net.synapseslices[learn[0]]
        want = (syn.pre >= proj.pre.start) & (syn.pre < proj.pre.stop) & (syn.post >= proj.post.start) & (syn.post < proj.post.stop)
        assert np.array_equal(syn.plastic_mask, want)
        lo, hi = run._clamp_plan
        assert np.all(np.isinf(lo[~want])) and np.all(np.isinf(hi[~want]))
        assert np.all(lo[want] == bounds[0]) and np.all(hi[want] == np.float32(bounds[1]))


# --------------------------------------------------------------------- 4. clustering scores

def test_ari_ami_reference_values():
    """sklearn.metrics.adjusted_rand_score / adjusted_mutual_info_score (arithmetic) on the same labels: 0.0909090909..., 0.1715242354..."""
    t = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    p = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 0])
    assert adjusted_rand_index(t, p) == pytest.approx(0.09090909090909091, abs=1e-12)
    assert adjusted_mutual_info(t, p) == pytest.approx(0.17152423540072848, abs=1e-12)
    perm = (t + 1) % 3
    assert adjusted_rand_index(t, perm) == pytest.approx(1.0) and adjusted_mutual_info(t, perm) == pytest.approx(1.0)


def test_kmeans_and_cluster_scores_find_separable_classes():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 30)
    X = np.eye(3)[y] * 10 + rng.normal(0, 0.3, (90, 3))
    assert adjusted_rand_index(y, kmeans(X, 3, np.random.default_rng(1))) == pytest.approx(1.0)
    sc = cluster_scores(np.vstack([X, np.zeros((1, 3))]), np.append(y, 0), rng=np.random.default_rng(1), k=3)
    assert sc["k"] == 3 and sc["silent_frac"] == pytest.approx(1 / 91) and sc["ari"] > 0.95


# --------------------------------------------------------------------- 5. variant B dynamics

def _l1_learning(run):
    run.set_mode("train")
    run.set_frozen(("L1_L2", "L2_L3"))


def test_one_winner_per_column_while_the_layer_learns_and_none_otherwise():
    """All inputs on for 3 steps: learning L1 -> exactly one spike per L1 column, the neuron with the highest potential;
    eval -> every L1 neuron fires, once (one_spike)."""
    _, run, proto = prepped()
    net, syn = run.ctx.net, run.ctx.syn
    frames = np.zeros((T, 128), dtype=bool)
    frames[:3] = True
    rows = net.neuronrows["L1_exc"]
    drive = np.bincount(syn.post, weights=syn.W, minlength=syn.weights.n_post)[rows] * 1.0       ##every input arrives at step 0
    col = np.asarray(net.positions["L1_exc"])[:, :2] @ np.array([4, 1])
    _l1_learning(run)
    run.reset_transient()
    counts = proto.ops.binned_counts(frames, 0, [T], group="L1_exc")[0]
    assert counts.sum() == 16 and np.all(np.bincount(col, weights=counts) == 1)
    assert all(counts[np.flatnonzero(col == c)[np.argmax(drive[col == c])]] == 1 for c in range(16))
    run.set_mode("eval")
    run.reset_transient()
    assert np.all(proto.ops.binned_counts(frames, 0, [T], group="L1_exc")[0] == 1)


def test_stdp_rule_on_the_winners_only():
    """Half the inputs spike at step 0 (kwta off: every column winner updates): each winner's edges from them grow by a+ w (1 - w), the others
    shrink by a- w (1 - w); losers and the frozen layers keep their weights."""
    _, run, proto = prepped(kwta={})
    syn, net = run.ctx.syn, run.ctx.net
    frames = np.zeros((T, 128), dtype=bool)
    frames[0, ::2] = True
    _l1_learning(run)
    run.reset_transient()
    run.plasticity.schedule(0)
    W0 = syn.W.copy()
    counts = proto.ops.binned_counts(frames, 0, [T], group="L1_exc")[0]
    win = net.neuronrows["L1_exc"].start + np.flatnonzero(counts)
    assert win.size
    k = np.flatnonzero(np.isin(syn.post, win))
    h = (syn.pre[k] < net.n_input) & (syn.pre[k] % 2 == 0)
    w = W0[k].astype(np.float64)
    want = np.where(h, 0.004, -0.003) * w * (1 - w)
    assert np.allclose(syn.W[k] - W0[k], want, atol=1e-7)
    rest = np.setdiff1d(np.arange(syn.W.size), k)
    assert np.array_equal(syn.W[rest], W0[rest])


def test_k_winners_with_radius_update_only():
    """##authors choice> mozafari (sf.get_k_winners): with kwta L1 (3, 1) and every input at step 0, the column winners all spike at step 0;
    only the 3 with the highest potential update, each blocking its 3 x 3 neighbourhood (so no two winners are neighbours); the default (5, 3)
    leaves ONE winner on the 4 x 4 grid."""
    for kwta, k_want in (({"L1_exc": [3, 1], "L2_exc": [1, 0]}, 3), (None, 1)):
        _, run, proto = prepped(kwta=kwta)
        syn, net = run.ctx.syn, run.ctx.net
        frames = np.zeros((T, 128), dtype=bool)
        frames[0] = True
        _l1_learning(run)
        run.reset_transient()
        run.plasticity.schedule(0)
        W0 = syn.W.copy()
        rows = net.neuronrows["L1_exc"]
        drive = np.bincount(syn.post, weights=syn.W, minlength=syn.weights.n_post)[rows]           ##= V after step 0 (IF, r 1)
        pos = np.asarray(net.positions["L1_exc"])[:, :2]
        col, cpos = pos @ np.array([4, 1]), np.indices((4, 4)).reshape(2, -1).T                     ##column of each neuron; coordinates of each column
        spikers = [np.flatnonzero(col == c)[np.argmax(drive[col == c])] for c in range(16)]        ##the column competition: highest V per column
        spikers = [n for n in spikers if drive[n] >= 2.0]
        k, r = (kwta or {"L1_exc": [5, 3]})["L1_exc"]
        want, blocked = [], np.zeros(16, dtype=bool)
        for n in sorted(spikers, key=lambda n: -drive[n]):                                         ##reference: greedy by potential, radius blocking
            if blocked[col[n]] or len(want) == k:
                continue
            want.append(n)
            blocked[np.all(np.abs(cpos - pos[n]) <= r, axis=1)] = True
        proto.ops.binned_counts(frames, 0, [T], group="L1_exc")
        changed = np.unique(syn.post[np.flatnonzero(syn.W != W0)]) - rows.start
        assert len(want) == k_want and set(changed.tolist()) == set(want)
        assert np.array_equal(run.plasticity.Nwin[:1], [len(want)])
        run.reset_transient()                                                                       ##transient: the next sample starts afresh
        assert run.plasticity.Nwin.sum() == 0 and not run.plasticity.Blocked.any() and not run.plasticity.Vacc.any()


def test_learning_rate_schedule():
    _, run, _ = prepped()
    p = run.plasticity
    got = []
    for j in (0, 2, 3, 7, 11, 100):
        p.schedule(j)
        got.append((round(p._ap, 6), round(p._an, 6)))
    assert got == [(0.004, 0.003), (0.004, 0.003), (0.008, 0.006), (0.016, 0.012), (0.032, 0.024), (0.15, 0.1125)]


def test_reward_punishment_and_the_decision_bound():
    _, run, proto = prepped(w=(0.7, 0.95))
    syn, net = run.ctx.syn, run.ctx.net
    run.set_mode("train")
    run.set_frozen(("in_L1", "L1_L2"))
    p = run.plasticity
    p.bound_decision(run.ctx)
    dec = np.isin(syn.post, np.arange(net.neuronrows["L3_exc"].start, net.neuronrows["L3_exc"].stop))
    assert syn.W[dec].max() <= np.float32(0.8) and syn.W[~dec].max() > 0.9
    frames = np.zeros((T, 128), dtype=bool)
    xy = np.indices((8, 8, 2)).reshape(3, -1)
    frames[0, (xy[0] < 4) & (xy[1] < 4)] = True                 ##one quadrant: only L2 column (0, 0) fires, all 4 of its neurons
    run.reset_transient()                                       ##(frozen L2: no competition): 4 of L3's 16 inputs arrive
    proto.present(frames)
    row = net.neuronrows["L3_exc"].start
    k = np.flatnonzero(syn.post == row)
    h = p.Arrived[syn.pre[k]]
    assert h.sum() == 4 and k.size == 16
    for reward, want in ((True, 0.5 * np.where(h > 0, 0.004, -0.003)), (False, 0.5 * np.where(h > 0, -0.004, 0.0005))):
        W0 = syn.W[k].astype(np.float64)
        assert p.apply_reward(run.ctx, row, reward, 0.5) == k.size
        assert np.allclose(syn.W[k], np.clip(W0 + want, 0.2, 0.8), atol=1e-7)
    run.set_mode("eval")
    with pytest.raises(RuntimeError):
        p.apply_reward(run.ctx, row, True, 1.0)


# --------------------------------------------------------------------- 6. variant B protocol

def _train(out, every=None, resume_from=None):
    return tiny_b().train(latency_data(16, 0), runcfg(every=every), test_data=latency_data(6, 1), out_dir=out, resume_from=resume_from, save_run=False)


KEYS = ("val_decision_acc", "test_decision_acc", "rstdp_train_acc", "rstdp_val_acc", "rstdp_best_pass", "val_ami_L1_exc", "test_ami_L2_exc",
        "test_ami_L3_pot", "test_ami_input", "w_mean_in_L1", "w_mean_L2_L3", "rate_hz_L1_exc", "initial_rate_hz")


def test_protocol_runs_and_resumes_bit_identically(tmp_path, monkeypatch):
    """Straight run vs killed after its first snapshot (STDP stage) and resumed, and vs resumed inside the first R-STDP pass: same report, same
    responses.npz."""
    full = _train(tmp_path / "full", every=5)
    r = full.report
    assert len(r["rstdp_val_acc"]) == 2 and r["rstdp_best_pass"] in (0, 1) and 0.0 <= r["test_decision_acc"] <= 100.0
    assert r["n_fit"] == 12 and r["n_val"] == 4 and r["n_test"] == 6
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
    st = json.loads((tmp_path / "killed" / "status.json").read_text())
    assert st["status"] == "interrupted" and st["last_snapshot"] == names[0]
    runs = (_train(tmp_path / "killed", every=5, resume_from=tmp_path / "killed" / names[0]),
            _train(tmp_path / "inside", every=5, resume_from=tmp_path / "full" / "snap_s02_i00000005.npz"))
    za = np.load(tmp_path / "full" / "responses.npz")
    for back, folder in zip(runs, ("killed", "inside")):
        assert [back.report[k] for k in KEYS] == [r[k] for k in KEYS], [(k, back.report[k], r[k]) for k in KEYS if back.report[k] != r[k]]
        assert np.array_equal(back.ctx.syn.W, full.ctx.syn.W)
        zb = np.load(tmp_path / folder / "responses.npz")
        assert za.files == zb.files and all(np.array_equal(za[k], zb[k]) for k in za.files)


def test_protocol_rejects_unknown_groups():
    _, run, _ = prepped()
    for bad in (dict(decision_group="nope"), dict(rstdp_learn=("nope",)), dict(n_per_class=3)):
        with pytest.raises(ValueError):
            MozafariProtocol(replace(run.runcfg.protocol, **bad), run)


def test_variant_a_protocol_trains_only_the_stage_groups(tmp_path):
    """Variant A's protocol (HLSP modules) on a toy two-layer network: every stage changes only its own groups, the input projection never
    learns, the report has the per-layer scores and the input control."""
    import hlsp_lsm_2026 as H
    from hlsp2026 import HLSPSynapseCfg, HLSPIntegratorCfg, HLSPThresholdCfg, HLSPPlasticityCfg
    from microcol_hlsp import MICROCOL_HLSP_SPARSE
    from microcol_hlsp_protocol import LayerwiseCfg, LayerwiseProtocol
    pnet = PhasicNet(4)
    pnet._instantiate_builder()
    b = pnet.builder
    b.set_neurons(("input", 64, True), ("L1_exc", 2 * 2 * 6), ("L1_inh", 2 * 2 * 2, False, True), ("L2_exc", 5))
    b.set_positions(input=(8, 8), L1_exc=(2, 2, 6), L1_inh=(2, 2, 2), L2_exc=(1, 1, 5))
    b.set_synapses(("in_L1e", "input", "L1_exc"), ("in_L1i", "input", "L1_inh"), ("L1_ee", "L1_exc", "L1_exc"), ("L1_ei", "L1_exc", "L1_inh"),
                   ("L1_ie", "L1_inh", "L1_exc"), ("L1_L2", "L1_exc", "L2_exc"))
    col = lambda p, **kw: {"kernel": "column", "p": p, **kw}
    b.set_connectivity(in_L1e=col(1.0, rf=4, stride=4), in_L1i=col(1.0, rf=4, stride=4), L1_ee=col(0.5), L1_ei=col(0.5), L1_ie=col(0.5),
                       L1_L2=col(1.0, rf=2, stride=2))
    b.set_weights(in_L1e={"values": [-8.0, 8.0]}, in_L1i={"values": [-8.0, 8.0]}, L1_ee=0.0, L1_ei=0.0, L1_ie=0.0, L1_L2=(2.0, 3.0))
    b.realize(network_name="tiny_a", post_offset=True, allow_autapse=False, save=False)
    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=1.0, weights_format="sparse", update_on="hybrid_event"),
        synapse=HLSPSynapseCfg(b=None, tau_syn=H.TAU_SYN, std=True, std_u=H.STD_U, std_tau=H.STD_TAU, std_pre_depletion=False),
        integrator=HLSPIntegratorCfg(tau_mem=H.TAU_MEM, v_rest=H.V_RESET, r=1.0, t_refrac=H.T_REFRAC),
        threshold=HLSPThresholdCfg(v_reset=H.V_RESET, v_th=H.V_TH, tau_th=None, adpt_th=None, t_refrac=H.T_REFRAC),
        plasticity=HLSPPlasticityCfg(tau_pretrace=H.TAU_E, tau_posttrace=H.TAU_E, A_ltp=H.A_E, A_ltd=H.A_E, eta_exc=1.0, eta_inh=1.0, w_min=-H.W_MAX,
                                     w_max=H.W_MAX, freeze=("in_L1e", "in_L1i"), A_I=H.A_I, tau_I1=H.TAU_I1, tau_I2=H.TAU_I2, trace_inc=H.TRACE_INC),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0), phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(replace(MICROCOL_HLSP_SPARSE, protocol_classes=(LayerwiseProtocol,)))
    pnet.cfg_name = "tiny_a"
    rng = np.random.default_rng(0)
    data = lambda n: ModelInput.from_tonic([(rng.random((30, 64)) < 0.2 + 0.3 * ((np.arange(64) < 32) == i % 2), i % 2) for i in range(n)])
    rc = replace(runcfg(), protocol=LayerwiseCfg(t_train_ms=15.0, t_state_ms=30.0, stage_learn=(("L1_ee", "L1_ei", "L1_ie"), ("L1_L2",)),
                                                 stage_epochs=(1, 1), cluster_groups=("L1_exc", "L2_exc"), n_init=2, val_split=0.25, rate_probe=3))
    run = pnet._prep_run(data(16), rc, test_data=data(6))
    W0 = run.ctx.syn.W.copy()
    proto = LayerwiseProtocol(rc.protocol, run)
    changed = {}
    for s, learn in enumerate(rc.protocol.stage_learn):
        before = run.ctx.syn.W.copy()
        run.set_mode("train")
        run.set_frozen(proto.frozen_for(learn))
        proto.ops.for_samples(proto._fit_set, proto.learn_step(s, 0), stage=s, name=f"L{s + 1}", reset=True)
        moved = run.ctx.syn.W != before
        changed[s] = {g for g in run.ctx.net.synapsespecs if moved[_mask(run, g)].any()}
    assert changed[0] and changed[0] <= {"L1_ee", "L1_ei", "L1_ie"} and changed[1] == {"L1_L2"}
    assert np.array_equal(run.ctx.syn.W[_mask(run, "in_L1e")], W0[_mask(run, "in_L1e")])
    full = pnet.train(data(16), rc, test_data=data(6), out_dir=tmp_path / "a", save_run=False).report
    assert {"val_ami_L1_exc", "test_ami_L2_exc", "test_vote_acc_L2_exc", "test_ami_input", "w_mean_L1_L2", "val_ami"} <= set(full)
    assert full["rate_hz_L1_exc"] > 0 and full["rate_hz_L2_exc"] > 0


def _mask(run, g):
    """Sparse edges of synapse group g (storage order)."""
    syn, proj = run.ctx.syn, run.ctx.net.synapseslices[g]
    return (syn.pre >= proj.pre.start) & (syn.pre < proj.pre.stop) & (syn.post >= proj.post.start) & (syn.post < proj.post.stop)


# --------------------------------------------------------------------- 7. the DoG latency encoder

def test_dog_latency_encoder_against_a_plain_loop():
    img = np.zeros((12, 12))
    img[3:9, 5:7] = 255.0
    img[6, 2:10] = 200.0
    kernels = [dog_kernel(*k) for k in MOZAFARI_DOG[:4]]
    got = dog_latency_prep(img, kernels, threshold=50.0, norm_radius=3, n_bins=5, pad=1)
    ##the same recipe written out pixel by pixel
    maps = np.zeros((len(kernels), 12, 12))
    for c, k in enumerate(kernels):
        h = k.shape[0] // 2
        p = np.pad(img, h)
        for i in range(12):
            for j in range(12):
                maps[c, i, j] = (p[i:i + k.shape[0], j:j + k.shape[1]] * k).sum()
    maps[maps < 50.0] = 0.0
    norm = np.zeros_like(maps)
    for c in range(len(kernels)):
        p = np.pad(maps[c], 3)
        for i in range(12):
            for j in range(12):
                norm[c, i, j] = maps[c, i, j] / (p[i:i + 7, j:j + 7].mean() + 1e-12)
    flat = norm.ravel()
    size = np.count_nonzero(flat) // 5
    want = np.full(flat.size, -1)
    want[np.argsort(-flat, kind="stable")[:5 * size]] = np.arange(5 * size) // size
    assert size > 0 and got.shape == (14, 14, 4)
    assert np.array_equal(got[1:-1, 1:-1], want.reshape(4, 12, 12).transpose(1, 2, 0)) and np.all(got[0] == -1)
    ds = DoGLatencyDataset(img[None], dog=MOZAFARI_DOG[:4], norm_radius=3, n_bins=5, pad=1)
    frames = ds[0]
    assert frames.shape == (7, 14 * 14 * 4) and frames.dtype == bool and not frames[5:].any()
    assert np.array_equal(np.flatnonzero(frames.any(axis=0)), np.flatnonzero(got.ravel() >= 0))
    assert np.array_equal(frames.argmax(axis=0)[got.ravel() >= 0], got.ravel()[got.ravel() >= 0])
