#test_class_sorted_subsets -- tonic stores N-MNIST sorted by digit, so a short run that took the FIRST n samples saw one class and the readout
# crashed. DataHandler.pick chooses n_train / n_test stratified, seeded, in a shuffled order, on both branches of PhasicNet._prep_run.
# These tests use a fake CLASS-SORTED dataset, the shape of the bug.
# Run: pytest system/tests/test_class_sorted_subsets.py -q     (from the repo root)
from __future__ import annotations
import numpy as np
import pytest

from datahandler import DataHandler, _stratified_pick
from modelinput import ModelInput
from runcfg import RunCfg, DataCfg
from rnghandler import RNGHandler
from test_hlsp import _Toy
import hlsp_lsm_2026 as H


class _SortedToy(_Toy):
    """_Toy with its labels sorted: every class-0 sample first, as tonic N-MNIST stores digit 0 first."""
    def __init__(self, n, seed=0):
        super().__init__(n, seed=seed); self.y = np.sort(self.y)


class _TonicLike:
    """A paired source that keeps its own label list (.targets) like tonic N-MNIST; counts how many samples were built."""
    target_transform = None
    def __init__(self, y): self.targets, self.built = list(y), 0
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        self.built += 1
        return np.zeros((5, 3), dtype=bool), self.targets[int(i)]


def test_hlsp_short_run_on_class_sorted_data_sees_both_classes():
    """n_train / n_test smaller than the class-0 block still give the readout both classes (the original crash)."""
    train, test = _SortedToy(12), _SortedToy(8, seed=1)
    assert (train.y[:4] == 0).all() and (test.y[:3] == 0).all()                          ##the first n are one class
    pnet = H.build(n_liquid=60, n_input=40, w_in=1.0, seed=3)
    run = pnet.train(ModelInput.from_tonic(train), H.runcfg(partitions=2, readout_epochs=30, rate_probe=2, report_every=4, n_train=4, n_test=4),
                     test_data=ModelInput.from_tonic(test), save_run=False, save_history=False)
    ##1/6 of the 4 picked training samples (rounded: 1) becomes the validation block, so n_train reports the 3 fit samples
    assert run.report["n_train"] == 3 and run.report["n_val"] == 1 and run.report["n_test"] == 4 and run.report["classes"] == [0, 1]
    assert sorted(train.y[run.train_set.index].tolist()) == [0, 0, 1, 1] and sorted(test.y[run.test_set.index].tolist()) == [0, 0, 1, 1]


def test_train_refuses_a_raw_source_and_takes_the_wrapped_one():
    """train() REQUIRES ModelInputs (data AND test_data): a raw source is a TypeError that says how to wrap it."""
    pnet = H.build(n_liquid=60, n_input=40, w_in=1.0, seed=3)
    rc = H.runcfg(partitions=2, readout_epochs=30, rate_probe=2, report_every=4, n_train=4, n_test=4)
    with pytest.raises(TypeError, match="ModelInput as data"):
        pnet.train(_SortedToy(12), rc, test_data=ModelInput.from_tonic(_SortedToy(8, seed=1)), save_run=False, save_history=False)
    with pytest.raises(TypeError, match="ModelInput as test_data"):
        pnet.train(ModelInput.from_tonic(_SortedToy(12)), rc, test_data=_SortedToy(8, seed=1), save_run=False, save_history=False)


def test_pick_is_stratified_seeded_and_mixed():
    y = np.repeat(np.arange(10), [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])   ##N-MNIST train class sizes, sorted by class
    a = _stratified_pick(y, 1000, np.random.default_rng(7))
    assert a.size == 1000 and np.unique(a).size == 1000
    counts = np.bincount(y[a], minlength=10)
    assert np.abs(counts - 1000 * np.bincount(y) / y.size).max() < 1                        ##class proportions kept (rounding only)
    assert np.array_equal(a, _stratified_pick(y, 1000, np.random.default_rng(7)))           ##same seed -> same pick and order
    assert not np.array_equal(a, _stratified_pick(y, 1000, np.random.default_rng(8)))
    assert len(set(y[a[:50]].tolist())) > 3                                                  ##order shuffled: classes mixed, not in blocks
    assert sorted(y[_stratified_pick(y, 3, np.random.default_rng(0))].tolist()) == [1, 3, 7] ##n < classes: largest remainders win


def test_pick_n_none_keeps_source_order_unless_shuffle():
    """n None and shuffle False return the very same ModelInput view; shuffle=True -> every sample, in a seeded order."""
    data = ModelInput.from_tonic(_SortedToy(12)).subset(np.arange(12), "train")
    dh = DataHandler()
    assert dh.pick(data, None, data_cfg=DataCfg(), rng=np.random.default_rng(0)) is data
    s = dh.pick(data, None, data_cfg=DataCfg(shuffle=True), rng=np.random.default_rng(0))
    assert sorted(s.index.tolist()) == list(range(12)) and not np.array_equal(s.index, np.arange(12))
    u = dh.pick(data, 5, data_cfg=DataCfg(stratify=False), rng=np.random.default_rng(0))   ##unstratified: a plain random draw, still not the head
    assert len(u) == 5 and np.unique(u.index).size == 5
    e = dh.pick(data.subset([], "empty"), 2, data_cfg=DataCfg(), rng=np.random.default_rng(0))   ##n_test on an EMPTY test set (test_fraction=0, no test_data) -> an empty view, no crash
    assert len(e) == 0 and e.index.dtype == np.int64 and e.labels().size == 0


def test_same_seed_same_pick_across_prep_runs():
    """Resume and reproducibility: the pick is keyed on (seed, dataset hash), so a second _prep_run sees the same samples in the same order."""
    idx = []
    for _ in range(2):
        pnet = H.build(n_liquid=30, n_input=40, w_in=1.0, seed=5)
        run = pnet._prep_run(ModelInput.from_tonic(_SortedToy(20)), H.runcfg(n_train=6, n_test=3), test_data=ModelInput.from_tonic(_SortedToy(9, seed=1)))
        idx.append((run.train_set.index.copy(), run.test_set.index.copy()))
    assert all(np.array_equal(a, b) for a, b in zip(*idx))


def test_new_datacfg_fields_are_hash_neutral_at_defaults():
    d = RunCfg().hashable()["data"]
    assert "n_train" not in d and "n_test" not in d and "shuffle" not in d                  ##older run_ids unchanged
    assert RunCfg(data=DataCfg(n_train=10)).hashable()["data"]["n_train"] == 10             ##a pick changes WHICH samples -> part of run_id


def test_labels_read_from_a_source_label_list_without_building_samples():
    src = _TonicLike([0, 0, 1, 2]); mi = ModelInput.from_tonic(src); before = src.built       ##ModelInput's own contract check may build sample 0
    assert mi.labels().tolist() == [0, 0, 1, 2] and src.built == before


@pytest.mark.parametrize("module", ["deckers_2022", "hlsp_lsm_2026", "krenzer_2025", "nalsm_2021"])
def test_every_lsm_build_hands_n_train_n_test_to_datacfg(module):
    rc = __import__(module).runcfg(n_train=7, n_test=3)
    assert (rc.data.n_train, rc.data.n_test, rc.data.stratify, rc.data.shuffle) == (7, 3, True, True)
