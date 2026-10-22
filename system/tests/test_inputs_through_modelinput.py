#test_inputs_through_modelinput -- the five LSM-family builds hand PhasicNet a ModelInput: ModelInput.labels / variant, ModelInput.subset views
# reading labels without building frames, train() checking data vs network n_input, the OPEN tonic chain the builds write against the old
# hand-written loop, and every build loader returning a ModelInput (fake tonic datasets, no download).
# Run: pytest system/tests/test_inputs_through_modelinput.py -q     (from the repo root)
from __future__ import annotations
import numpy as np
import pytest

from modelinput import ModelInput, _dataset_hash


class _Counting:
    """Frames-only source that counts how often a sample is BUILT."""
    def __init__(self, n=5, T=6, D=4):
        self.n, self.T, self.D, self.built = n, T, D, 0
    def __len__(self): return self.n
    def __getitem__(self, i):
        self.built += 1
        return np.random.default_rng([1, int(i)]).random((self.T, self.D)) < 0.3
    def variant(self, i, *, k):
        return np.full((self.T + k, self.D), bool(k % 2))


class _FakeSHD:
    sensor_size = (700, 1, 1); dtype = np.dtype([("t", "<i8"), ("x", "<i8"), ("p", "<i8")])
    def __init__(self, save_to=None, train=True, transform=None):
        self.transform, self.n = transform, (5 if train else 3)
    def __len__(self): return self.n
    def __getitem__(self, i):
        g = np.random.default_rng([2, i]); m = 2000
        ev = np.zeros(m, self.dtype); ev["t"] = np.sort(g.integers(0, [80_000, 330_000, 150_000][i % 3], m)); ev["x"] = g.integers(0, 700, m)
        return (self.transform(ev) if self.transform else ev), int(i % 3)


class _FakeNMNIST(_FakeSHD):
    sensor_size = (34, 34, 2); dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("t", "<i8"), ("p", "<i8")])
    def __getitem__(self, i):
        g = np.random.default_rng([3, i]); m = 1500
        ev = np.zeros(m, self.dtype); ev["t"] = np.sort(g.integers(0, [90_000, 400_000][i % 2], m))
        ev["x"], ev["y"], ev["p"] = g.integers(0, 34, m), g.integers(0, 34, m), g.integers(0, 2, m)
        return (self.transform(ev) if self.transform else ev), int(i % 2)


@pytest.fixture
def fake_tonic(monkeypatch):
    tonic = pytest.importorskip("tonic")
    monkeypatch.setattr(tonic.datasets, "SHD", _FakeSHD)
    monkeypatch.setattr(tonic.datasets, "NMNIST", _FakeNMNIST)
    return tonic


def _old_framing(events, sensor_size, T, dt_ms=1.0):
    """The hand-written framing loop the builds replaced (deckers shd_frames / hlsp TonicFrames), kept as the reference."""
    import tonic
    n = int(np.prod(sensor_size))
    f = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=int(dt_ms * 1000))(events).reshape(-1, n) > 0
    return f[:T] if f.shape[0] >= T else np.concatenate([f, np.zeros((T - f.shape[0], n), dtype=bool)], axis=0)


# ------------------------------------------------------------------ ModelInput.labels / variant, subset views
def test_labels_held_targets_build_no_frames():
    src = _Counting(); data = ModelInput(src, np.array([4, 3, 2, 1, 0]))
    built = src.built                                                    ##the construction probe reads sample 0 once
    assert data.labels().tolist() == [4, 3, 2, 1, 0] and src.built == built
    sub = data.subset([4, 2, 0, 1]).subset([0, 2])                        ##composed: original indices 4, 0
    assert sub.labels().tolist() == [0, 4] and sub.index.tolist() == [4, 0] and data.labels().tolist() == [4, 3, 2, 1, 0] and src.built == built


def test_subset_is_a_view_that_shares_the_source_and_composes():
    """ModelInput.subset is a view: shares source / targets / meta / frame_shape, composes indices to the ORIGINAL source; _dataset_hash hashes its first samples."""
    src = _Counting(); data = ModelInput(src, np.array([4, 3, 2, 1, 0])); built = src.built
    assert data.index is None and data.name == "" and len(data) == 5
    a = data.subset([4, 2, 0, 1], "a"); b = a.subset([0, 2], "b")
    assert src.built == built                                                       ##no re-validation: nothing built
    assert a.source is data.source and a.targets is data.targets and a.meta is data.meta and a.frame_shape == data.frame_shape and a.n_input == data.n_input
    assert (len(a), len(b), a.name, b.name) == (4, 2, "a", "b") and b.index.dtype == np.int64 and b.index.tolist() == [4, 0]
    assert b[1][1] == 4 and np.array_equal(b[1][0], data[0][0]) and [t for _, t in b] == [0, 4] and b[0][0].shape == (6, 4)
    assert np.array_equal(b.labels(), data.labels()[[4, 0]]) and "'b' view of 5" in repr(b) and "view" not in repr(data)
    h_full, n_full = _dataset_hash(data); h_b, n_b = _dataset_hash(b)
    assert (n_full, n_b) == (3, 2) and h_full != h_b and _dataset_hash(data.subset([0, 1, 2, 3, 4]))[0] == h_full   ##len is mixed in; a view of ALL 5 has the same len and samples: same id
    assert _dataset_hash(data.subset([4, 0, 1, 2, 3]))[0] != h_full   ##same len, other first samples: another id
    with pytest.raises(IndexError):
        b[2]


def test_labels_of_paired_source_reads_raw():
    pairs = [(np.zeros((3, 2), bool), k) for k in (7, 8, 9)]
    data = ModelInput.from_tonic(pairs)
    assert data.labels().tolist() == [7, 8, 9] and data.subset([2, 0]).labels().tolist() == [9, 7]
    src = type("T", (), {"targets": [7, 8, 9], "target_transform": None, "__len__": lambda s: 3, "__getitem__": lambda s, i: pairs[int(i)]})()
    assert ModelInput.from_tonic(src).subset([2, 0]).labels().tolist() == [9, 7]      ##the source's own label list, indexed (not len(view) == len(list))


def test_variant_passes_through_to_TI_and_subset():
    data = ModelInput(_Counting(), np.zeros(5))
    v = data.subset([3, 1]).variant(1, k=3)                                ##member 1 = original 1
    assert v.shape == (9, 4) and v.all() and not v.flags.writeable
    with pytest.raises(ValueError):                                       ##a variant with another frame shape is refused like any sample
        ModelInput(type("S", (_Counting,), {"variant": lambda self, i, k: np.zeros((2, 5), bool)})(), np.zeros(5)).variant(0, k=0)
    with pytest.raises(AttributeError):
        ModelInput([np.zeros((2, 2), bool)] * 3, np.zeros(3)).variant(0)


def test_train_checks_data_against_network_n_input():
    from krenzer_2025 import build, runcfg
    pnet = build(n_input=10, n_liquid=30, n_wta=4, n_wta_inh=2)
    wrong = ModelInput([np.zeros((5, 12), bool)] * 4, np.arange(4) % 2)
    with pytest.raises(ValueError, match="Input size mismatch"):
        pnet.train(wrong, runcfg(n_train=2, n_test=2), test_data=wrong, save_run=False, save_history=False)


# ------------------------------------------------------------------ the builds' OPEN tonic chain == the old hand-written loop
def test_open_tonic_chain_equals_old_loop(fake_tonic):
    import tonic_helpers
    from tonic_helpers import ClampBinary, FixedLength, US_TO_MS
    tonic = fake_tonic
    assert not hasattr(tonic_helpers, "frame_transform") and not hasattr(tonic_helpers, "Head")
    for cls, T in ((_FakeSHD, 200), (_FakeNMNIST, 250)):
        raw = cls()
        ##the chain as deckers_2022.shd_frames / hlsp_lsm_2026 / nalsm_2021.load_dataset write it
        chain = tonic.transforms.Compose([tonic.transforms.ToFrame(sensor_size=cls.sensor_size, time_window=int(1.0 * US_TO_MS)), ClampBinary(dtype=np.bool_), FixedLength(T)])
        framed = ModelInput.from_tonic(cls(transform=chain))
        for i in range(len(raw)):
            f, y = framed[i]
            assert f.dtype == bool and np.array_equal(f, _old_framing(raw[i][0], cls.sensor_size, T)) and y == raw[i][1]
        head = framed.subset(np.arange(3), "head")                   ##the first n samples: a view
        assert len(head) == 3 and head[2][1] == raw[2][1] and np.array_equal(head[2][0], framed[2][0])
    assert FixedLength(4)(np.ones((2, 3), np.uint8)).tolist() == [[1] * 3] * 2 + [[0] * 3] * 2 and FixedLength(1)(np.ones((2, 3))).shape == (1, 3)


# ------------------------------------------------------------------ every build loader returns a ModelInput
def test_build_loaders_return_modelinput(fake_tonic, monkeypatch):
    import deckers_2022 as dk, krenzer_2025 as kr, hlsp_lsm_2026 as hl, nalsm_2021 as na, zhang_2025 as zh
    shd = dk.shd_frames(True, t_max_ms=200.0, bin_ms=1.0)   ##1 ms bins here so the frames match Krenzer's 1 ms loader below
    assert isinstance(shd, ModelInput) and shd.n_input == 700 and len(shd) == 5 and shd[1][0].shape == (200, 700)
    assert dk.shd_frames(True, t_max_ms=200.0)[1][0].shape == (50, 700)                  ##the Deckers default: 4 ms bins, 200 ms real time -> 50 frames
    k = kr.shd70_frames(True, t_max_ms=200.0)   ##the loaders take no n_samples (DataCfg picks)
    assert isinstance(k, ModelInput) and k.n_input == 70 and np.array_equal(k[1][0], shd[1][0][:, ::10])
    imgs = np.random.default_rng(4).integers(0, 256, (5, 28, 28)).astype(np.uint8); ys = np.arange(5) % 3
    for m in (hl, na):
        monkeypatch.setattr(m, "load_mnist", lambda path: (imgs, ys, imgs[:3], ys[:3]))
        for name, n_in in (("mnist", 784), ("nmnist", 2312)):
            tr, te, n = m.load_dataset(name)   ##the loaders take no n_train / n_test and return the WHOLE set
            assert isinstance(tr, ModelInput) and isinstance(te, ModelInput) and n == tr.n_input == n_in and len(tr) > len(te) > 0
            assert tr[0][0].dtype == bool and tr[0][0].shape[1] == n_in
    z = zh.mnist_input(imgs, ys, present_ms=20.0, rest_ms=10.0)
    assert isinstance(z, ModelInput) and z.n_input == 784 and z.labels().tolist() == ys.tolist() and z[0][0].dtype == bool
