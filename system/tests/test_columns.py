#test_columns - builder columns: set_columns(n) splits every group (input included) at realize, synapses stay inside a column unless cross_columns=True
import numpy as np
import pytest

from phasicnet import PhasicNet
import capture


def edges(net, syn):
    k = net.syntype == net.type_key[syn]
    post = net.post_idx[k] + (net.n_input if net.post_offset else 0)
    return net.pre_idx[k], post


def build(columns=7, cross=False, save_to=None):
    pnet = PhasicNet(3)
    pnet._instantiate_builder()
    b = pnet.builder
    b.set_neurons(("input", 20, True), ("L1_exc", 56), ("L1_inh", 14), ("L2_exc", 56), ("L3_exc", 57))
    b.set_columns(columns)
    b.set_synapses(("in_L1", "input", "L1_exc"), ("L1_ei", "L1_exc", "L1_inh"), ("L1_ie", "L1_inh", "L1_exc"), ("L1_L2", "L1_exc", "L2_exc"),
                   ("L2_L3", "L2_exc", "L3_exc"), ("L3_L3", "L3_exc", "L3_exc", cross))
    b.set_connectivity(in_L1=0.3, L1_ei=1.0, L1_ie="all_to_all", L1_L2=0.5, L2_L3=0.5, L3_L3=0.3)
    b.set_weights(in_L1=1.0, L1_ei=0.5, L1_ie=0.5, L1_L2=0.5, L2_L3=0.5, L3_L3=0.5)
    if save_to is not None:
        b.save_specs(filename="cols", spec_name="cols", YAML=True, save_to=save_to)
    return b.realize(network_name="cols", post_offset=True, allow_autapse=False, save=False)


def cross_count(net, syn):
    pre, post = edges(net, syn)
    return int((net.column_ids[pre] != net.column_ids[post]).sum()), len(pre)


def test_column_ids_split_each_group_roughly_equally():
    net = build()
    assert net.columns == 7
    assert set(np.bincount(net.column_ids[net.neuronslices["input"]])) == {2, 3}         ##input is split too
    for g, size in (("L1_exc", 8), ("L1_inh", 2), ("L2_exc", 8)):
        assert np.array_equal(np.bincount(net.column_ids[net.neuronslices[g]]), [size] * 7)
    assert set(np.bincount(net.column_ids[net.neuronslices["L3_exc"]])) == {8, 9}        ##57 neurons: sizes differ by at most one


def test_synapses_stay_in_their_column_unless_cross_columns():
    net = build()
    for syn in ("in_L1", "L1_ei", "L1_ie", "L1_L2", "L2_L3", "L3_L3"):
        crossing, n = cross_count(net, syn)
        assert crossing == 0 and n > 0
    pre, post = edges(net, "L1_ie")                                   ##all_to_all inside the column: 2 x 8 per column
    assert len(pre) == 7 * 2 * 8
    crossing, _ = cross_count(build(cross=True), "L3_L3")
    assert crossing > 0


def test_no_columns_changes_nothing():
    a, b = build(columns=None), build(columns=1)
    assert a.columns is None and (a.column_ids == -1).all()
    assert a.network_id == b.network_id
    assert a.synapse_counts["n_L1_L2"] > 3 * build().synapse_counts["n_L1_L2"]           ##without columns: the whole group, about 7 x the edges


def test_columns_round_trip_through_the_network_file(tmp_path):
    net = build(save_to=tmp_path)
    again = PhasicNet(3)
    again._instantiate_builder()
    again.builder.load_specs(tmp_path / "cols.yaml")
    net2 = again.builder.realize(network_name="cols", post_offset=True, allow_autapse=False, save=False)
    assert net2.network_id == net.network_id and np.array_equal(net2.column_ids, net.column_ids)


def small(*calls):
    pnet = PhasicNet(3)
    pnet._instantiate_builder()
    b = pnet.builder
    for c in calls:
        c(b)
    return b


NEURONS = lambda b: b.set_neurons(("input", 7, True), ("A", 14), ("pool", 7))
SYNS = lambda b: b.set_synapses(("in_A", "input", "A"), ("A_A", "A", "A")).set_connectivity(in_A=0.5, A_A=0.5).set_weights(in_A=1.0, A_A=0.5)
real = lambda b: b.realize(network_name="s", allow_autapse=False, save=False)


def test_set_columns_checks_at_realize():
    b = small(lambda b: b.set_neurons(("input", 4, True), ("A", 3)), lambda b: b.set_columns(4))
    with pytest.raises(ValueError):                                   ##more columns than neurons
        b.realize(save=False)


def test_set_columns_order_does_not_matter():
    first = real(small(lambda b: b.set_columns(7), NEURONS, SYNS))
    after = real(small(NEURONS, SYNS, lambda b: b.set_columns(7)))
    assert first.network_id == after.network_id and first.neuronspecs["A"].columns == 7


def test_set_columns_overwrite_and_reset():
    plain = real(small(NEURONS, SYNS))
    net = real(small(NEURONS, SYNS, lambda b: b.set_columns(7), lambda b: b.set_columns(2)))
    assert net.columns == 2 and all(sp.columns == 2 for sp in net.neuronspecs.values())   ##later call overwrites, input included
    net = real(small(NEURONS, SYNS, lambda b: b.set_columns(7), lambda b: b.set_columns(None)))
    assert net.columns is None and net.network_id == plain.network_id
    b = small(NEURONS, SYNS, lambda b: b.set_columns(7))
    real(b)
    b.set_columns(None)                                               ##same builder, realize again without columns
    assert real(b).network_id == plain.network_id


def test_neurons_per_column():
    net = build()
    table = capture.neurons_per_column(net)
    assert list(table["L1_exc"]) == [8] * 7 and table["input"].sum() == 20
    assert capture.neurons_per_column(net, neuron_type="L1_inh").tolist() == [2] * 7
    assert capture.neurons_per_column(net, column_id=0)["L2_exc"] == 8
    assert capture.neurons_per_column(net, neuron_type="L3_exc", column_id=6) in (8, 9)
