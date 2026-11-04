#preprocessing - raw data -> (T, ...) binary spike frames a ModelInput can wrap.
#
#FOUR PIPELINES, deliberately separate because they answer different questions:
#   1. poisson_prep()  MNIST intensity -> Poisson spike trains
#   2. tonic_prep()    NMNIST event streams -> dense frames via tonic.ToFrame, microseconds -> ms.
#                      
#   3. chemo_prep()    Neighbouring-pixel data augmentation (Jamme & Duponchel 2017, J Chemometrics
#                      31:e2882). Fuses each pixel with its 8 neighbours**
#   4. r_prep()        The R script's data preparation (2_1_1_SNN_network_simulation_140225_mnist.R): a rate window
#                      with a 20 Hz FLOOR, pause/sleep blocks, optional label "response" neurons.
#
#VERSION 1: Brian2's STDP example
# (brian2.readthedocs.io/en/stable/examples/synapses.STDP.html, after Song, Miller & Abbott 2000)

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

US_TO_MS = 1000        ##tonic timestamps are microseconds; PhasicNet maths is in ms


   ######################################################################################
##########################   1. Poisson rate coding (Brian2 STDP style)   #################
   ######################################################################################

def load_mnist(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the mnist.npz bundle -> (x_train, y_train, x_test, y_test), uint8 0..255."""
    with np.load(Path(path), allow_pickle=False) as z:
        keys = set(z.files)
        if {"x_train", "y_train"} <= keys:
            return z["x_train"], z["y_train"], z.get("x_test"), z.get("y_test")
        ##some bundles nest under different names; fall back to positional
        arrays = [z[k] for k in z.files]
        return arrays[0], arrays[1], (arrays[2] if len(arrays) > 2 else None), (arrays[3] if len(arrays) > 3 else None)


def poisson_prep(images: np.ndarray, *, T: int = 350, dt_ms: float = 1.0,
                 peak_hz: float = 63.75, rng: np.random.Generator = None,
                 flatten: bool = True) -> np.ndarray:
    """
    Intensity -> Poisson spike trains. The Brian2-STDP / rate-coding encoding.

    Each pixel becomes an independent Poisson process whose rate is proportional to its intensity:
        rate_i = intensity_i / max_intensity * peak_hz          [Hz]
        P(spike in one dt) = rate_i * dt / 1000                 [dt in ms]

    Bernoulli-per-step = discrete-time approximation to a Poisson process**

    Args:
        images = (n, H, W) or (n, D) intensities, any positive scale
        T = timesteps per sample (350 ms presentation is the usual choice)
        peak_hz = firing rate of the brightest pixel
        flatten = return (n, T, H*W). False keeps (n, T, H, W) for ModelInput axis_labels="THW".

    Returns:
        (n, T, ...) bool spike frames -- exactly the (T, ...) contract ModelInput wraps.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    images = np.asarray(images, dtype=np.float64)
    n = images.shape[0]
    frame_shape = images.shape[1:]

    flat = images.reshape(n, -1)
    peak = flat.max()
    if peak <= 0:
        raise ValueError("All-zero images: nothing to encode.")
    p_spike = (flat / peak) * peak_hz * dt_ms / 1000.0          ##(n, D) per-step probability

    out_shape = (n, T, flat.shape[1]) if flatten else (n, T, *frame_shape)
    spikes = np.empty(out_shape, dtype=bool)
    for i in range(n):
        s = rng.random((T, flat.shape[1])) < p_spike[i]
        spikes[i] = s if flatten else s.reshape(T, *frame_shape)

    logger.info("poisson_prep: %d samples -> (T=%d, %s), peak %.1f Hz, mean %.2f Hz/input",
                n, T, out_shape[2:], peak_hz, spikes.mean() * 1000.0 / dt_ms)
    return spikes


   ######################################################################################
##########################   2. Tonic / NMNIST event frames   #############################
   ######################################################################################

def tonic_prep(dataset, *, dt_ms: float = 1.0, sensor_size: Tuple[int, int, int] = (34, 34, 2),
               n_samples: int = None, binary: bool = True):
    """
    NMNIST (or any tonic event dataset) -> dense (T, P, H, W) frames binned at dt_ms.

    NMNIST is ALREADY event-based, so there is no rate encoding to do: the only jobs are binning
    microsecond timestamps into dt_ms frames and clamping multi-event bins to a single spike.

        tonic time_window is in MICROSECONDS -> time_window = dt_ms * 1000
        ClampBinary: 1+ events in a bin -> one spike, because PhasicNet's contract is BINARY and a
        count of 2 would silently become "2.0" in a float weight update.

    Args:
        dataset = a tonic dataset yielding (events, label), ex. tonic.datasets.NMNIST(...)
        n_samples = take only the first n (NMNIST is 60k samples; you rarely want all of them here)

    Returns:
        list of (frames, label). NOT a numpy stack -- NMNIST samples have different T.

    """
    import tonic
    to_frame = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=int(dt_ms * US_TO_MS))

    n = len(dataset) if n_samples is None else min(int(n_samples), len(dataset))
    out = []
    for i in range(n):
        events, label = dataset[i]
        frames = to_frame(events)                       ##(T, P, H, W) counts
        if binary:
            frames = (frames > 0)
        out.append((frames, label))
    logger.info("tonic_prep: %d NMNIST samples binned at %.1f ms -> %s frames",
                n, dt_ms, out[0][0].shape if out else "none")
    return out


def nmnist_dataset(save_to, *, train: bool = True):
    """tonic.datasets.NMNIST handle. Downloads on first use; separated so tonic_prep stays pure."""
    import tonic
    return tonic.datasets.NMNIST(save_to=str(save_to), train=train)


   ######################################################################################
##########################   3. Neighbouring-pixel augmentation (chemometrics)   ##########
   ######################################################################################

def chemo_prep(images: np.ndarray, *, order: str = "anticlockwise",
               collapse: str = "stack") -> np.ndarray:
    """
    Neighbouring pixel data augmentation, after Jamme & Duponchel (2017), J Chemometrics31:e2882.

    Args:
        images = (n, H, W) intensities
        order = "anticlockwise" follows the paper (centre, right, upper-right, up, ...). "raster"
            is plain row-major. Order matters only for interpretability -- it fixes WHICH input
            neuron carries which neighbour -- not for the values.

    Returns:
        (n, H-2, W-2, 9) if collapse="stack", else (n, H-2, W-2).
    """
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError(f"chemo_prep expects (n, H, W) images, got {images.shape}.")
    n, H, W = images.shape
    if H < 3 or W < 3:
        raise ValueError(f"Images must be at least 3x3 to have a neighbourhood; got {H}x{W}.")

    ##Offsets as (row, col).  paper  = anticlockwise: 
    if order == "anticlockwise":
        offsets = [(0, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    elif order == "raster":
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    else:
        raise ValueError(f"Unknown order {order!r}. Expected 'anticlockwise' or 'raster'.")

    ##Border pixels are DROPPED (the paper's n-2, m-2): they have no complete neighbourhood.
    stack = np.empty((n, H - 2, W - 2, len(offsets)), dtype=np.float64)
    for k, (dr, dc) in enumerate(offsets):
        stack[..., k] = images[:, 1 + dr: H - 1 + dr, 1 + dc: W - 1 + dc]

    if collapse == "stack":
        logger.info("chemo_prep: (%d, %d, %d) -> %s (9 neighbours per pixel, %s order)",
                    n, H, W, stack.shape, order)
        return stack
    if collapse == "mean":
        out = stack.mean(axis=-1)
        logger.info("chemo_prep: (%d, %d, %d) -> %s (neighbourhood mean)", n, H, W, out.shape)
        return out
    raise ValueError(f"Unknown collapse {collapse!r}. Expected 'stack' or 'mean'.")


   ######################################################################################
##########################   4. R-script pipeline (rate-window + pause/sleep)   ###########
   ######################################################################################

##from 2_1_1_SNN_network_simulation_140225_mnist.R 
##  1. RATE WINDOW WITH A FLOOR. The R script maps intensity linearly onto [min_rate_x, max_rate_x]
##     = [20, 60] Hz across [data_min, data_max] = [0, 256]. A BLACK pixel therefore fires at
##     20 Hz, not 0 Hz. poisson_prep is proportional (black = silent). So here every input neuron
##     is always active and the digit is carried by the CONTRAST between 20 and 60 Hz, whereas in
##     poisson_prep it is carried by which neurons fire at all. That is a different coding scheme,
##     not a different constant.
##  2. PAUSE AND SLEEP PERIODS. Each observation is followed by `pause` ms at pause_rate (0 Hz),
##     and the whole sequence by `sleep` ms at sleep_rate (5 Hz). The pause lets membrane and traces
##     decay between digits; the sleep block is what the R model's sleep phase consumes.
##  3. RESPONSE NEURONS. Optional label neurons appended to the input vector: the true class fires
##     at response_high_rate (20 Hz), the others at response_baseline_rate (0 Hz). This makes the
##     stream SUPERVISED at the input layer -- the label is presented alongside the digit.
##

def r_prep(images: np.ndarray, labels: np.ndarray = None, *,
           duration: int = 1000, pause: int = 1000, sleep: int = 0, dt_ms: float = 1.0,
           min_rate: float = 20.0, max_rate: float = 60.0,
           data_min: float = 0.0, data_max: float = 256.0,
           pause_rate: float = 0.0, sleep_rate: float = 5.0,
           response_baseline_rate: float = 0.0, response_high_rate: float = 20.0,
           n_classes: int = None, downsample: int = 4,
           rng: np.random.Generator = None, concatenate: bool = False):
    """
    The R script's data preparation, ported.

    Args:
        images = (n, H, W) or (n, D) intensities on the [data_min, data_max] scale
        labels = (n,) class labels. Given AND n_classes set -> response neurons are appended.
        duration / pause / sleep = ms of stimulus, of gap after each stimulus, and of sleep after
            the whole sequence (R defaults 1000 / 1000 / 10)
        min_rate / max_rate = Hz at data_min and data_max (R defaults 20 / 60)
        downsample = block-mean factor. The R script loads a pre-downsampled 7x7 set (p_in=49);
            4 reproduces that from raw 28x28 MNIST. 1 = no downsampling.
        concatenate = True returns ONE continuous (T_total, D) stream, matching the R script's
            single spike_data matrix. False returns a per-sample list, which is what PhasicNet's
            DataHandler consumes.

    Returns:
        concatenate=False -> list of ((duration+pause, D) bool, label)
        concatenate=True  -> ((n*(duration+pause)+sleep, D) bool, labels array)

    NOTE ORIENTATION: the R script's spike_data is (neurons, time). PhasicNet is (time, neurons),
    so this returns the TRANSPOSE of the R matrix. Same data, opposite axis order.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    images = np.asarray(images, dtype=np.float64)
    n = images.shape[0]

    ##Block-mean downsample to the R script's 7x7 (p_in=49). Mean, not stride: a mean of
    ## intensities is still an intensity, whereas striding discards 15 of every 16 pixels.
    if downsample and downsample > 1 and images.ndim == 3:
        H, W = images.shape[1:]
        k = downsample
        images = images.reshape(n, H // k, k, W // k, k).mean(axis=(2, 4))
    flat = images.reshape(n, -1)
    p_in = flat.shape[1]

    ##LINEAR rate window with a FLOOR -- the defining feature of this pipeline.
    scaled = np.clip((flat - data_min) / (data_max - data_min), 0.0, 1.0)
    rates_hz = min_rate + scaled * (max_rate - min_rate)                ##(n, p_in), >= min_rate

    n_resp = 0
    if labels is not None and n_classes:
        n_resp = int(n_classes)
        labels = np.asarray(labels)

    D = p_in + n_resp
    steps_stim = int(round(duration / dt_ms))
    steps_pause = int(round(pause / dt_ms))

    def _draw(p_row: np.ndarray, steps: int) -> np.ndarray:
        """Bernoulli-per-step, p = rate*dt/1000 -- the R script's convention."""
        return rng.random((steps, p_row.size)) < p_row

    samples = []
    for i in range(n):
        p_stim = np.empty(D)
        p_stim[:p_in] = rates_hz[i] * dt_ms / 1000.0
        if n_resp:
            ##Response neurons: the TRUE class high, all others at baseline.
            p_stim[p_in:] = response_baseline_rate * dt_ms / 1000.0
            p_stim[p_in + int(labels[i])] = response_high_rate * dt_ms / 1000.0

        block = _draw(p_stim, steps_stim)
        if steps_pause:
            ##Pause: inputs AND response neurons drop to pause_rate (0 Hz by default), which is what
            ## lets membrane potential and STDP traces decay between observations.
            p_pause = np.full(D, pause_rate * dt_ms / 1000.0)
            block = np.concatenate([block, _draw(p_pause, steps_pause)], axis=0)
        samples.append((block, int(labels[i]) if labels is not None else None))

    logger.info("r_prep: %d samples -> (T=%d, D=%d) [%d input + %d response], rates %.0f-%.0f Hz, "
                "pause %d ms @ %.0f Hz", n, steps_stim + steps_pause, D, p_in, n_resp,
                min_rate, max_rate, pause, pause_rate)

    if not concatenate:
        return samples

    stream = [b for b, _ in samples]
    if sleep:
        ##Sleep block appended ONCE after every observation, at sleep_rate.
        steps_sleep = int(round(sleep / dt_ms))
        stream.append(_draw(np.full(D, sleep_rate * dt_ms / 1000.0), steps_sleep))
    return np.concatenate(stream, axis=0), (None if labels is None else np.asarray(labels))


   ######################################################################################
##########################   5. DoG + intensity-to-latency (Mozafari et al. 2019)   ######
   ######################################################################################

def dog_kernel(size: int, sigma1: float, sigma2: float) -> np.ndarray:
    """(size, size) difference of Gaussians, zero mean, max 1 (the authors' SpykeTorch utils.DoGKernel)."""
    w = size // 2
    x, y = np.mgrid[-w:w + 1, -w:w + 1]
    r2 = x * x + y * y
    dog = (np.exp(-0.5 * r2 / sigma1 ** 2) / sigma1 ** 2 - np.exp(-0.5 * r2 / sigma2 ** 2) / sigma2 ** 2) / (2 * np.pi)
    dog -= dog.mean()
    return dog / dog.max()


def dog_latency_prep(image: np.ndarray, kernels, *, threshold: float = 50.0, norm_radius: int = 8, n_bins: int = 15, pad: int = 2) -> np.ndarray:
    """
    ONE image (H, W), 0..255 -> the spike BIN of every input, (H + 2 pad, W + 2 pad, n_kernels) int16, -1 = never spikes.
    Mozafari et al. 2019 Sec. 2, as the authors' SpykeTorch S1C1Transform: each DoG kernel correlated with the image ('same' size, zero padding),
    values below `threshold` -> 0, local normalisation (each value / the mean of its (2 norm_radius + 1)^2 window, zero padded, per kernel),
    intensity -> latency (utils.Intensity2Latency): all values sorted descending fill n_bins bins of nonzero // n_bins values each, strongest
    first; the remainder (fewer than n_bins values) never spikes. pad = the silent border the first layer sees (the authors' sf.pad(input, 2)).
    """
    img = np.asarray(image, dtype=np.float64)
    maps = []
    for k in kernels:
        h = k.shape[0] // 2
        win = np.lib.stride_tricks.sliding_window_view(np.pad(img, h), k.shape)
        maps.append(np.einsum("ijkl,kl->ij", win, k))
    out = np.stack(maps)                                                   ##(C, H, W)
    out[out < threshold] = 0.0
    w = 2 * int(norm_radius) + 1
    rows = np.lib.stride_tricks.sliding_window_view(np.pad(out, ((0, 0), (w // 2, w // 2), (w // 2, w // 2))), w, axis=1).sum(-1)
    means = np.lib.stride_tricks.sliding_window_view(rows, w, axis=2).sum(-1) / (w * w)          ##the window mean in two 1-D passes (same values, 2x faster)
    out = out / (means + 1e-12)
    flat = out.ravel()
    nonzero = int(np.count_nonzero(flat))
    size = max(nonzero // int(n_bins), 1)
    top = np.argsort(-flat, kind="stable")[:min(int(n_bins) * size, nonzero)]
    bins = np.full(flat.size, -1, dtype=np.int16)
    bins[top] = np.arange(top.size) // size
    bins = bins.reshape(out.shape).transpose(1, 2, 0)                     ##(H, W, C): channels innermost, as a grid (H, W, C) in C order
    return np.pad(bins, ((pad, pad), (pad, pad), (0, 0)), constant_values=-1)


MOZAFARI_DOG = ((3, 3 / 9, 6 / 9), (3, 6 / 9, 3 / 9), (7, 7 / 9, 14 / 9), (7, 14 / 9, 7 / 9), (13, 13 / 9, 26 / 9), (13, 26 / 9, 13 / 9))
##(size, sigma1, sigma2) of the six DoG kernels, MozafariDeep.py: on- and off-centre at three scales


class DoGLatencyDataset:
    """images (n, H, W), 0..255 -> sample i = frames (n_bins + flush, (H + 2 pad) (W + 2 pad) n_kernels) bool: each input spikes ONCE, at the step
    of its latency bin (dog_latency_prep); the flush steps are silent (the last bin's spikes need one step per layer to reach the top).
    Encoded WHEN ASKED, nothing stored, the same index always gives the same frames. A ModelInput source: ModelInput(DoGLatencyDataset(x), y).
    Inputs are ordered as the grid (H + 2 pad, W + 2 pad, n_kernels) in C order: the network's input group needs that grid."""

    def __init__(self, images, *, dog=MOZAFARI_DOG, threshold: float = 50.0, norm_radius: int = 8, n_bins: int = 15, pad: int = 2, flush: int = 2):
        self.images = np.asarray(images)
        self.kernels = [dog_kernel(*k) for k in dog]
        self.kw = dict(threshold=threshold, norm_radius=norm_radius, n_bins=n_bins, pad=pad)
        self.T = int(n_bins) + int(flush)
        self.grid = (self.images.shape[1] + 2 * pad, self.images.shape[2] + 2 * pad, len(self.kernels))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i) -> np.ndarray:
        bins = dog_latency_prep(self.images[int(i)], self.kernels, **self.kw).ravel()
        on = np.flatnonzero(bins >= 0)
        frames = np.zeros((self.T, bins.size), dtype=np.bool_)
        frames[bins[on], on] = True
        return frames


   ######################################################################################
##########################   6. DVS Gesture event frames (Amir et al. 2017)   #############
   ######################################################################################

DVSGESTURE_SENSOR = (128, 128, 2)        ##tonic.datasets.DVSGesture.sensor_size (x, y, polarity); 11 classes


def dvsgesture_frames(train: bool, *, bin_ms: float = 10.0, t_max_ms: float = 1500.0, spatial_factor: float = 0.25, save_to=None):
    """DVS Gesture via tonic -> ModelInput, sample i = (frames (T, 2, H, W) bool, label int).   ##MISSING-RESOLUTIONS 6
    spatial_factor 0.25: 128 x 128 -> 32 x 32 pixels (2048 inputs with both polarities; tonic.Downsample floors x and y). Events are binned at bin_ms of
    REAL time (1+ events -> one spike) and cut or zero-padded to T = t_max_ms / bin_ms frames; the network runs one frame per step of SimulationCfg.dt.
    A recording is about 6 s long, so t_max_ms keeps its first part. bin_ms, t_max_ms and spatial_factor are choices, not from a paper.
    Lazy: tonic reads and frames a sample when it is asked for. Files in save_to/DVSGesture/ (default RAW_DIR; download_datasets.py fetches them)."""
    import tonic
    from tonic_helpers import ClampBinary, FixedLength
    from modelinput import ModelInput
    from paths import RAW_DIR
    T = int(round(t_max_ms / bin_ms))
    size = (int(DVSGESTURE_SENSOR[0] * spatial_factor), int(DVSGESTURE_SENSOR[1] * spatial_factor), DVSGESTURE_SENSOR[2])
    frame = tonic.transforms.Compose([tonic.transforms.Downsample(spatial_factor=spatial_factor),
                                      tonic.transforms.ToFrame(sensor_size=size, time_window=int(bin_ms * US_TO_MS)),
                                      ClampBinary(dtype=np.bool_), FixedLength(T)])      ##events -> smaller grid -> frames at bin_ms -> 1+ events = one spike -> cut / pad to T
    ds = tonic.datasets.DVSGesture(save_to=str(save_to or RAW_DIR), train=train, transform=frame)
    data = ModelInput.from_tonic(ds, axis_labels="TPHW")
    logger.info("dvsgesture_frames: %s %d samples, T=%d frames of %.1f ms real time, %s pixels, n_input=%d", "train" if train else "test", len(data), T, bin_ms,
                size[:2], data.n_input)
    return data


def dvsgesture_dataset(*, bin_ms: float = 10.0, t_max_ms: float = 1500.0, spatial_factor: float = 0.25):
    """(train, test) = the official DVS Gesture split as two ModelInputs (what a main() or a parallel.Job(load=dvsgesture_dataset, load_kw={...}) loads)."""
    kw = dict(bin_ms=bin_ms, t_max_ms=t_max_ms, spatial_factor=spatial_factor)
    return dvsgesture_frames(True, **kw), dvsgesture_frames(False, **kw)
