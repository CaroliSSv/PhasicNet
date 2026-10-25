#test_lookup_and_loading -- lookups and loaders that change no numbers but that other code leans on:
#   1. Network.neuronrows / row_offset: the one-time group -> row_idx lookup agrees with the builder's own synapseslices,
#      with and without post_offset, and post_offset changes NO result (dense, sparse/timestep, sparse/event).
#   2. Network.synapse_counts: realized n_<synapse_group_name> per group, saved with the network.
#   3. DynamicsCfg.load finds cfg SUBCLASSES by cfg_type: a D&C cfg survives a JSON round trip.
#   4. RunCfg.load reads every field incl. the nested policy sections.
#   5. The derived cfg records the RESOLVED freeze (synapse-group names), not the requested keywords.
# Run: pytest system/tests/test_lookup_and_loading.py -q
from __future__ import annotations
import dataclasses, json
import logging, re     ##the duplicate-edge WARNING test
import numpy as np
from modelinput import ModelInput   ##train() / _prep_run require a ModelInput
import pytest

from phasicnet import PhasicNet
from system.dynamicscfg import DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, HistoryCfg, ScorerCfg, ProtocolCfg
from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, AdaptiveThreshold, TraceSTDP, SparseTraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase

SEED, N_IN, T = 7, 12, 30


def _pnet(post_offset: bool, fmt: str = "dense", update_on: str = "timestep") -> PhasicNet:
    p = PhasicNet(SEED); p._instantiate_builder()
    p.builder.set_neurons(("input", N_IN, True), ("exc", 16), ("inh", 5, False, True))
    p.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    p.builder.set_connectivity(in_exc=0.6, exc_exc=0.1, exc_inh=0.3, inh_exc=0.3)
    p.builder.set_weights({"in_exc": (0.0, 4.0), "exc_exc": 1.0, "exc_inh": 3.0, "inh_exc": 3.0})
    p.builder.realize(network_name="t", post_offset=post_offset)
    sp = fmt == "sparse"
    p.cfg = DynamicsCfg(simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on=update_on), synapse=SynapseCfg(b=None),
                integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=60.0), threshold=ThresholdCfg(v_reset=-65.0, v_th=-54.0, tau_th=100.0, adpt_th=0.4),
                plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.01, A_ltd=0.0105, eta_exc=1.0, eta_inh=1.0,
                                         w_min=-6.0, w_max=6.0, freeze=("inh",)),
                noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0), phases=(PhaseCfg(phase_type="DEFAULT"),))
    p.set_runtime_classes(synapse_cls=DeltaCurrentSparse if sp else DeltaCurrentDense, integrator_cls=LIFexact, threshold_cls=AdaptiveThreshold,
                          plasticity_cls=SparseTraceSTDP if sp else TraceSTDP, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    return p


def _data():
    rng = np.random.default_rng(1)
    return ModelInput.from_tonic([(rng.random((T, N_IN)) < 0.25, i % 2) for i in range(4)])


@pytest.mark.parametrize("post_offset", [False, True])
def test_neuronrows_agree_with_synapseslices(post_offset):
    net = _pnet(post_offset).network
    assert net.row_offset == (N_IN if post_offset else 0)
    for syn, spec in net.synapsespecs.items():
        assert net.synapseslices[syn].post == net.neuronrows[spec.post_neuron]
    for group, slc in net.neuronslices.items():
        if post_offset and net.neuronspecs[group].input:
            assert group not in net.neuronrows                      ##inputs have no rows under post_offset
        else:
            assert net.neuronrows[group] == slice(slc.start - net.row_offset, slc.stop - net.row_offset)
    assert net.rec_slice == slice(net.neuronrows["inh"].start, net.neuronrows["exc"].stop)


@pytest.mark.parametrize("fmt,update_on", [("dense", "timestep"), ("sparse", "timestep"), ("sparse", "event"), ("sparse", "hybrid_event")])
def test_post_offset_changes_no_result(fmt, update_on):
    """Row-space post ids (post_offset) vs true-id post ids: identical spikes and identical final W, in every execution mode."""
    out = {}
    for off in (False, True):
        run = _pnet(off, fmt, update_on)._prep_run(_data(), RunCfg())
        rec, orig = [], run.step
        def step(*a, _o=orig, _r=rec, _run=run, **k):
            r = _o(*a, **k); _r.append(_run.ctx.neur.Spikes[N_IN:].copy()); return r
        run.step = step
        for frames, _ in _data():
            run.run_sample(frames)
        net, syn = run.ctx.net, run.ctx.syn
        W = np.zeros((net.n_total, net.n_total), np.float32)
        if fmt == "sparse":
            W[syn.post + net.row_offset, syn.pre] = syn.W
        else:
            W[net.row_offset:] = syn.W
        out[off] = (np.array(rec), W)
    assert np.array_equal(out[False][0], out[True][0]) and out[False][0].any()
    assert np.array_equal(out[False][1], out[True][1])


def test_synapse_counts_are_realized_per_group(tmp_path):
    net = _pnet(True).network
    assert sum(net.synapse_counts.values()) == net.n_synapses
    for syn, key in net.type_key.items():
        assert net.synapse_counts[f"n_{syn}"] == int(np.count_nonzero(net.syntype == key))
    net.save_net(filename="net", save_to=tmp_path)
    assert json.loads((tmp_path / "net.json").read_text())["synapse_counts"] == net.synapse_counts


def test_duplicate_edges_across_groups_are_dropped_with_a_warning(caplog):
    """Duplicate edges across two groups are dropped with ONE WARNING carrying the real counts; no duplicates -> nothing logged."""
    _MSG = r"Removed duplicate synapses during COO ndarray generation\. Length before: (\d+), length after: (\d+)"
    def build(dup: bool):
        p = PhasicNet(SEED); p._instantiate_builder()
        p.builder.set_neurons(("input", N_IN, True), ("exc", 16))
        groups = (("in_exc", "input", "exc"),) + ((("in_exc_dup", "input", "exc"),) if dup else ())     ##the second group duplicates EVERY edge of the first
        p.builder.set_synapses(*groups)
        p.builder.set_connectivity({g[0]: "all_to_all" for g in groups})
        p.builder.set_weights({g[0]: 1.0 for g in groups})
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="builder"):
            net = p.builder.realize(network_name="dup")
        return net, [r for r in caplog.records if r.name == "builder" and r.getMessage().startswith("Removed duplicate synapses")]
    net, records = build(dup=True)
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    before, after = (int(x) for x in re.fullmatch(_MSG, records[0].getMessage()).groups())
    assert (before, after) == (2 * N_IN * 16, N_IN * 16) and net.n_synapses == after
    net, records = build(dup=False)
    assert records == [] and net.n_synapses == N_IN * 16


def test_dc_cfg_round_trips_through_json():
    from diehl_cook_2015 import build
    p = build(32, 6, norm_target=78.0 * 32 / 784)
    raw = json.loads(json.dumps(dataclasses.asdict(p.cfg)))
    loaded = DynamicsCfg.load(raw)
    assert type(loaded.synapse).__name__ == "DCSynapseCfg" and type(loaded.plasticity).__name__ == "DCPlasticityCfg"
    assert dataclasses.replace(loaded, plasticity=dataclasses.replace(loaded.plasticity, freeze=tuple(loaded.plasticity.freeze))) == p.cfg
    raw["synapse"]["cfg_type"] = "NoSuchCfg"
    with pytest.raises(RuntimeError, match="No SynapseCfg subclass"):
        DynamicsCfg.load(raw)


def test_runcfg_load_reads_every_field():
    rc = RunCfg(protocol=ProtocolCfg(epochs=3, val_split=0.2, patience=2), checkpoint=CheckpointCfg(select="best", to_disk=False, keep_last=None),
                data=DataCfg(test_fraction=0.2), history=HistoryCfg(record_spikes=False))
    raw = json.loads(json.dumps(dataclasses.asdict(rc)))
    assert RunCfg.load(raw) == rc
    raw["data"]["warn_seconds"] = 60.0                                  ##old files carry warn_seconds
    assert RunCfg.load(raw) == rc
    assert RunCfg.load({}) == RunCfg()                                  ##missing keys keep their defaults


def test_runcfg_load_reads_files_saved_before_the_run_settings_moved():
    """An old file's top-level epochs / k_fold / val_split / patience go into the protocol the values ran."""
    assert RunCfg.load({"epochs": 2}) == RunCfg(protocol=ProtocolCfg(epochs=2))
    assert RunCfg.load({"epochs": 1, "k_fold": None, "val_split": 0.25}) == RunCfg(protocol=ProtocolCfg(val_split=0.25))
    named = {"epochs": 1, "k_fold": 1, "val_split": 0.3, "protocol": {"protocol_name": "holdoutval", "reset_between_samples": False, "report_every": 500}}   ##an OLD file: protocol_name (read by a COMPAT hook)
    assert RunCfg.load(named).protocol == ProtocolCfg(val_split=0.3)       ##a named protocol keeps its name and takes only its own fields
    ##a k-fold file names CROSSVAL, whose cfg class is deferred: not silently a holdout
    with pytest.raises(RuntimeError, match="CROSSVAL"):
        RunCfg.load({"epochs": 3, "k_fold": 2, "val_split": None, "patience": 2, "protocol": None})
    ##an old HOLDOUT section loads as HOLDOUTVAL with val_split None -- the same run
    assert RunCfg.load({"protocol": {"protocol_name": "HOLDOUT", "epochs": 2, "reset_between_samples": True}}) == RunCfg(protocol=ProtocolCfg(epochs=2, reset_between_samples=True))   ##old files carry protocol_name (COMPAT)
    assert RunCfg.load({"epochs": 2, "protocol": {"protocol_name": "holdout"}}) == RunCfg(protocol=ProtocolCfg(epochs=2))


def test_phase_type_and_protocol_type_are_the_keys_and_old_files_still_load():
    """phase_type / protocol_type are the saved keys the loaders pick subclasses by; old files (phase_name / protocol_name) still load (COMPAT)."""
    from system.dynamicscfg import SleepPhaseCfg, _load_phase_cfg
    from dc_protocol import DCProtocolCfg
    raw = dataclasses.asdict(_pnet(True).cfg)
    assert [k for k in raw["phases"][0] if k.endswith("_name") or k.endswith("_type")] == ["phase_type"]
    assert dataclasses.asdict(RunCfg(protocol=DCProtocolCfg(n_train=4)))["protocol"]["protocol_type"] == "DC2015"
    ##the new keys: the subclass is picked by phase_type / protocol_type
    assert type(_load_phase_cfg({"phase_type": "sleep", "sleep_fraction": 0.5})) is SleepPhaseCfg
    assert type(RunCfg.load({"protocol": {"protocol_type": "dc2015", "n_train": 4}}).protocol) is DCProtocolCfg
    ##the old keys (COMPAT, delete later): the same result, and the old key does not survive into the cfg
    old = {**raw, "phases": [{"phase_name": "DEFAULT", "triggers": {}, "trigger_mode": "all", "exit_triggers": None, "dataloader_active": True},
                             {"phase_name": "Sleep", "triggers": {}, "trigger_mode": "all", "exit_triggers": None, "dataloader_active": True, "sleep_fraction": 0.5}]}
    loaded = DynamicsCfg.load(old)
    assert [type(c) for c in loaded.phases] == [PhaseCfg, SleepPhaseCfg] and loaded.phases[1].phase_type == "SLEEP" and loaded.phases[1].sleep_fraction == 0.5
    assert RunCfg.load({"protocol": {"protocol_name": "dc2015", "n_train": 4}}).protocol == DCProtocolCfg(n_train=4)
    assert RunCfg.load({"protocol": {"protocol_name": "holdoutval", "val_split": 0.3}}) == RunCfg(protocol=ProtocolCfg(val_split=0.3))
    ##the strict check stays: a phase cfg under a name no PhaseCfg subclass defaults to does not load
    with pytest.raises(RuntimeError, match="No PhaseCfg subclass"):
        _load_phase_cfg({"phase_type": "PRUNE"})


def test_noise_cfg_lost_its_weight_fields_and_old_files_still_load():
    """NoiseCfg has no mu_weights / sigma_weights; a cfg file saved with them still loads (COMPAT: the two keys are dropped)."""
    assert {f.name for f in dataclasses.fields(NoiseCfg)} == {"mu_mem", "sigma_mem", "cfg_type"}
    with pytest.raises(TypeError):
        NoiseCfg(mu_mem=0.0, sigma_mem=0.0, mu_weights=0.0, sigma_weights=0.0)
    raw = json.loads(json.dumps(dataclasses.asdict(_pnet(True).cfg)))
    raw["noise"] = {**raw["noise"], "mu_weights": 0.0, "sigma_weights": 0.0}      ##an old file
    assert DynamicsCfg.load(raw).noise == NoiseCfg(mu_mem=0.0, sigma_mem=0.0)


@dataclasses.dataclass(frozen=True, slots=True)
class WideScorerCfg(ScorerCfg):      ##module level: cfg._cfg_subclasses needs the module attribute (a slots dataclass in a function leaves a pre-slots twin)
    cfg_type: str = dataclasses.field(default="WideScorerCfg", kw_only=True)
    window: int = 3


def test_runcfg_load_rebuilds_policy_subclasses_and_their_names_stay_out_of_run_id():
    """History/Scorer/Checkpoint/DataCfg subclasses load back as themselves (cfg_type), and cfg_type is not hashed."""
    rc = RunCfg(scorer=WideScorerCfg(window=5))
    assert RunCfg.load(json.loads(json.dumps(dataclasses.asdict(rc)))) == rc
    assert RunCfg(scorer=WideScorerCfg()).hashable()["scorer"] == {**RunCfg().hashable()["scorer"], "window": 3}   ##only the new FIELD enters the hash
    assert "cfg_type" not in RunCfg().hashable()["data"] and "cfg_type" not in RunCfg().hashable()["checkpoint"]


def test_derived_cfg_records_resolved_freeze():
    p = _pnet(True)
    run = p._prep_run(_data(), RunCfg())
    assert p.cfg.plasticity.freeze == ("inh",)
    assert run.history.derived_cfg.plasticity.freeze == tuple(run.ctx.syn.frozen) == ("exc_inh", "inh_exc")
