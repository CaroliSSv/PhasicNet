#prep_helpers
from dataclasses import dataclass
import numpy as np
import tonic, matplotlib.pyplot as plt
###MODELInput import here would create circular import?


US_TO_MS = 1000   # tonic timestamps are microseconds, PhasicNet maths is ms. So default conversion -> 1000



############################### -  for Tonic Compose - ################################################## 

@dataclass(frozen=True)
class ClampBinary:
    """ANY non-zero frame-pixel (1+ events) -> 1 (spike)"""
    dtype: type = np.uint8

    def __call__(self, frames):
        return (frames > 0).astype(self.dtype)


@dataclass(frozen=True)
class FixedLength:   
    """(T_i, ...) frames -> exactly (T, ...): cut after T steps, or zero-padding at the end. 

    * For models that read a fixed window per sample
    """
    T: int

    def __call__(self, frames):
        frames = np.asarray(frames)
        if frames.shape[0] >= self.T:
            return frames[:self.T]
        return np.concatenate([frames, np.zeros((self.T - frames.shape[0], *frames.shape[1:]), dtype=frames.dtype)], axis=0)


############################### -  Custom - ##################################################
###DESCRIBE & stop simple printing (slow). logger
def describe(frames, label=""):
    frames = np.asarray(frames)
    print(f"[{label}] shape={frames.shape} dtype={frames.dtype} "
          f"min={frames.min()} max={frames.max()} "
          f"active={np.count_nonzero(frames) / frames.size:.3%}")

def reconstruct_frames(model_input, index=0):
    """
    Rebuild (T, P, H, W) from the FLATTENED (T, N_in) using neuron_coordinates ONLY.
    Deliberately does NOT use reshape (reshape is faster)
    """
    frames_TI, label = model_input[index]
    coords = np.array([model_input.neuron_coordinates(i) for i in range(model_input.n_input)])
    rebuilt = np.zeros((frames_TI.shape[0], *model_input.frame_shape), dtype=frames_TI.dtype)
    rebuilt[(slice(None), *coords.T)] = frames_TI      # rebuilt[:, p, h, w] = frames_TI[:, i]
    return rebuilt, label

def show(model_input, index=0, *, source_frames=None, split_polarity=True):
    """
    Visualize spikes AFTER flattening and neuron-index attribution, mapped back to original (P, H, W) coordinates. 

    Args:
        model_input: ModelInput class instance
        index = 0 - Choose SINGLE sample to visualize
        source_frames=None - Optionally provides ROUNDTRIP ASSERTION against input data (prior to ModelInput-transform)
        split_polarity=True - Optionally animates each polarity seperately, to visualize which neuron IDs received which polarity.

    """


    rebuilt, label = reconstruct_frames(model_input, index)
    describe(rebuilt, f"rebuilt from neuron idx (label={label})")

    # Pairing check: the two neurons for one pixel are H*W apart under row-major.
    P, H, W = model_input.frame_shape
    stride = H * W
    ok = all(model_input.neuron_index(1, h, w) - model_input.neuron_index(0, h, w) == stride
             for h in (0, H // 2, H - 1) for w in (0, W // 2, W - 1))
    print(f"  polarity pair stride == H*W ({stride}): {ok}")

    if source_frames is not None:
        print(f"  round-trip matches pre-flatten frames: {np.array_equal(rebuilt, source_frames)}")

    keep = [tonic.utils.plot_animation(rebuilt)]
    if split_polarity and P == 2:
        for p in range(P):
            single = np.zeros_like(rebuilt)
            single[:, p] = rebuilt[:, p]        # keep (T,P,H,W) so colours stay consistent
            keep.append(tonic.utils.plot_animation(single))

    plt.show()
    return keep


########REMOVE? 
def show_pair(events, sensor_size, dt_ms=1.0):
    """Frame one sample with and without ClampBinary, print & animate both. (TESTER func)"""
    to_frame = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=int(dt_ms * US_TO_MS))
    raw = to_frame(events)
    clamped = ClampBinary()(raw)
    describe(raw, "ToFrame")
    describe(clamped, "ToFrame + ClampBinary")
    keep = [tonic.utils.plot_animation(raw), tonic.utils.plot_animation(clamped)]
    plt.show()
    return keep