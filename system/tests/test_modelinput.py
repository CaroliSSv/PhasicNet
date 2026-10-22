#test_modelinput - fast contract tests. No tonic, no downloads. Run: pytest -q
from __future__ import annotations
import numpy as np
import pytest
from modelinput import ModelInput


# ------TESTERS (heavy AI assist)

def make_arrays(n=5, T=4, frame_shape=(2, 3, 3), seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, T, *frame_shape)) < 0.3).astype(np.uint8)
    y = np.arange(n, dtype=np.int64)
    return X, y


class FakePairedDataset:
    """Stand-in for a tonic dataset: source[i] -> (frames, label). Keeps tests offline."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]





def test_flatten_roundtrip_is_lossless():
    X, y = make_arrays()
    mi = ModelInput(X, y)
    frames, target = mi[0]
    assert frames.shape == (X.shape[1], mi.n_input)
    assert target == y[0]
    for t in range(frames.shape[0]):
        assert np.array_equal(mi.reconstruct_step(frames[t]), X[0, t])


def test_coordinate_roundtrip():
    X, y = make_arrays()
    mi = ModelInput(X, y)
    for index in range(mi.n_input):
        assert mi.neuron_index(*mi.neuron_coordinates(index)) == index



def test_array_with_T_equal_two_is_still_rejected():
    """The old silent-unpack hazard: T==2 used to unpack into (data, target)."""
    X = np.zeros((5, 2, 3, 3), np.uint8)
    with pytest.raises(TypeError):
        ModelInput(X)



def test_trailing_shape_drift_is_rejected():
    X = [np.zeros((4, 2, 3), np.uint8), np.zeros((4, 2, 4), np.uint8)]
    y = [0, 1]
    mi = ModelInput(X, y)
    with pytest.raises(ValueError):
        mi[1]


def test_hash_is_deterministic_and_content_sensitive():
    X, y = make_arrays()
    first = ModelInput(X, y).dataset_hash()
    second = ModelInput(X.copy(), y.copy()).dataset_hash()
    assert first == second

    changed = X.copy()
    changed[0, 0, 0, 0, 0] ^= 1
    assert ModelInput(changed, y).dataset_hash() != first


###########BEST TEST, FROM NEURON-IDX RECONSTRUCTION:

def test_coordinate_reconstruction_matches_reshape():
    X, y = make_arrays()
    mi = ModelInput(X, y)
    frames, _ = mi[0]
    coords = np.array([mi.neuron_coordinates(i) for i in range(mi.n_input)])
    rebuilt = np.zeros((frames.shape[0], *mi.frame_shape), dtype=frames.dtype)
    rebuilt[(slice(None), *coords.T)] = frames
    assert np.array_equal(rebuilt, X[0])                        # mapping matches source

# ------ to_TI re-checks the binary contract periodically
def test_binary_contract_is_rechecked_every_kth_non_bool_sample():
    import modelinput
    k = modelinput.BINARY_RECHECK_EVERY

    class Drift:
        """sample 0 is binary (the construction probe passes); every later sample carries a 2."""
        def __len__(self): return 3 * k
        def __getitem__(self, i):
            a = np.zeros((2, 3), np.uint8); a[0, 0] = 1 if i == 0 else 2
            return a

    mi = ModelInput(Drift(), np.zeros(3 * k))
    seen = 0
    with pytest.raises(ValueError, match="not spike-valued"):             ##the same error _record_contract raises
        for seen in range(1, 3 * k):
            mi[seen]
    assert seen == k                                                       ##the k-th converted sample, not the first bad one and not never
    for _ in range(3):                                                     ##leave the parent at k + 3 before making the view ...
        mi[1]
    sub = mi.subset(np.arange(1, 3 * k))                                   ##... a view CONTINUES the parent's count (one counter per dataset): the next check is k - 3 samples away
    with pytest.raises(ValueError):
        for seen in range(3 * k - 1):
            sub[seen]
    assert seen == k - 4 and sub._n_seen is mi._n_seen
    ##Operations.run_epoch makes a NEW shuffled view every epoch: views shorter than k must still reach the check
    short = ModelInput(Drift(), np.zeros(3 * k))
    with pytest.raises(ValueError):
        for epoch in range(4):
            for frames, _ in short.subset(np.arange(1, k // 2 + 1), f"epoch{epoch}"):
                pass
    assert epoch == 1                                                      ##the k-th sample falls in the second view
    assert ModelInput(Drift(), np.zeros(3 * k), binary_check=False)[k - 1][0].max() == 2     ##binary_check=False: never
    b = ModelInput(np.zeros((2, 2, 2), bool), np.zeros(2))
    for _ in range(2 * k + 1):
        b[1]
    assert b._n_seen[0] == 0                                               ##bool frames are not counted: nothing to check
    assert not any(isinstance(getattr(ModelInput, n, None), property) and "recheck" in n for n in dir(ModelInput))   ##no user access: a module constant only
