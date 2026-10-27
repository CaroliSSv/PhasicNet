#### utils/helper functions, diverse. Save, load, canonicalize

from __future__ import annotations
from typing import Optional, Dict, Tuple, Literal
import json
from hashlib import blake2b
from pathlib import Path
import yaml
import os
from functools import singledispatch
from datetime import datetime  
import numpy as np
import logging
from paths import EXPORT_DIR, MNIST_PATH, MNIST_URL, FMNIST_PATH, FMNIST_URL
import gzip
from urllib.request import urlretrieve
import operator


logger = logging.getLogger(__name__)


###########Calculus, and listings############## 




_COMPARISONS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt}


def _is_better(score: float, best: float | None, *, greater_is_better: bool, min_improvement: float) -> bool:
    """he ONE definition of "better"
    Consumed by CheckpointPolicy._is_better AND Operations.run_fit."""
    if best is None:
        return True
    if greater_is_better:
        return score > best + min_improvement
    return score < best - min_improvement


def np_sigmoid(array, out=None) -> None: 
    """
    Custom Sigmoid transform (numpy), implements in inplace on array using ufuncs:

        array = 1/(1 + np.exp(-array))
    """
    if out is None:
        out = array
    np.negative(array, out=out)
    np.exp(array, out=out)
    np.add(array, 1.0, out=out)
    np.reciprocal(array, out=out)

_AVAILABLE_NP_TRANSFORMS = {
    "tanh": np.tanh,
    "exp": np.exp,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "sigmoid": np_sigmoid}


def _choice(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()

        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False

        print("Please enter yes or no.")





def _capitalize_strip_name(name: str) -> str:
    return str(name).strip().upper()


def _filepath(file:str|Path, default_dir:Path = EXPORT_DIR) -> Path:
    """Resolve filepath and prepend default_dir if none given"""
    file = Path(file)
    if file.parent == Path("."):
        return default_dir/file
    return file

def import_mnist(kind: Literal["digits", "fashion"] = "digits", save_to: str | Path | None = None) -> Path:
    """Download MNIST digits or Fashion-MNIST if needed and return local file path (mnist.npz / fmnist.npz, same keys: x_train, y_train, x_test, y_test).
    overwrite global MNIST_PATH / FMNIST_PATH if save_to provided
    """
    default, name = {"digits": (MNIST_PATH, "MNIST"), "fashion": (FMNIST_PATH, "Fashion-MNIST")}[kind]
    path = default if save_to is None else Path(save_to) / default.name
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info("%s from: %s", name, path)
        return path
    logger.info("Downloading %s to %s", name, path)
    if kind == "digits":
        urlretrieve(MNIST_URL, path)
    else:
        _download_fmnist(path)
    return path

def _download_fmnist(path: Path) -> None:
    """Fashion-MNIST exists online only as 4 idx files -> packed into one npz with mnist.npz's keys."""
    arrays = []
    for name in ("train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"):
        gz = path.parent / f"fmnist_{name}"
        urlretrieve(FMNIST_URL + name, gz)
        with gzip.open(gz, "rb") as f:
            raw = f.read()
        arrays.append(np.frombuffer(raw, np.uint8, offset=8) if "labels" in name else np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28))
        gz.unlink()
    y, x, yt, xt = arrays
    np.savez_compressed(path, x_train=x, y_train=y, x_test=xt, y_test=yt)




def _read_json_yaml(file: str, fmt: Literal["json", "yaml", "yml"] | None = None) -> Dict:
    """ Read ONE json/yaml file -> return data as dict. 
    Infer extension from file OR use explicit extension (fmt) if provided."""
    file = Path(file)
    #### INFER from EXTENSION
    if fmt is None:
        ext = file.suffix.lower()
        if ext == ".json":
            fmt = "json"
        elif ext in (".yaml", ".yml"):
            fmt = "yaml"
        else:
            raise ValueError("Could not infer format from file extension.")
        
    #### use explicit extension
    fmt = fmt.lower()
    if fmt == "json":
        with open(file, "r", encoding="utf-8") as f:
            logger.debug("Read %s-file %s, returning data.", fmt, file)
            return json.load(f)
    if fmt in ("yaml", "yml"):
        with open(file, "r", encoding="utf-8") as f:
            logger.debug("Read %s-file %s, returning data",fmt, file)
            return yaml.safe_load(f)

    raise ValueError("Only 'json' and 'yaml'/'yml' are supported.")

def _read_net_npz(file: str|Path): 
    """ Read realized Network from .npz file:

    returns tuple of np.ndarrays:
        ( syntype, pre_idx, post_idx, W0, positions ) """
    
    file = Path(file)
    if file.suffix.lower() != ".npz":
        file = file.with_suffix(".npz")
    with np.load(file) as arrays:
        syntype = arrays["syntype"]
        pre_idx = arrays["pre_idx"]
        post_idx = arrays["post_idx"]
        W0 = arrays["W0"]
        positions = {key[len("pos_"):]: arrays[key] for key in arrays.files if key.startswith("pos_")}    
        
        logger.debug("Read netfile: '%s' successfully. Returning ndarrays with shape: syntype=%s pre_idx=%s post_idx=%s W0=%s", 
                     file, syntype.shape, pre_idx.shape, post_idx.shape, W0.shape)
        
        return syntype, pre_idx, post_idx, W0, positions


def _read_history_npz(file: str | Path) -> Dict[str, np.ndarray]:
    """
    Read RunHistory arrays from .npz file.

    Returns dict of np.ndarrays; dict[str, np.ndarray]
    """
    file = Path(file)

    if file.suffix.lower() != ".npz":
        file = file.with_suffix(".npz")

    with np.load(file, allow_pickle=False) as arrays:
        out = {key: arrays[key] for key in arrays.files}
    for key in ("stage_starts", "stage_values"):        ##BACKWARDS COMPATIBILITY TODO later: delete
        if key in out:
            out["mode" + key[5:]] = out.pop(key)

    logger.debug("Read RunHistory arrays from: %s", file)

    return out





def _write_net_npz(file: str|Path, syntype: np.ndarray, pre_idx: np.ndarray, post_idx: np.ndarray, W0: np.ndarray,
                   positions: Dict[str, np.ndarray] | None = None) -> None:   
    """Write realized Network arrays to one .npz file."""
    file = Path(file)

    np.savez(file.with_suffix(".npz"), syntype=syntype, pre_idx=pre_idx, post_idx=post_idx, W0=W0, **{f"pos_{g}": p for g, p in (positions or {}).items()})
    logger.info("Network saved as: %s", file.with_suffix(".npz"))


def _write_json_yaml(file: str|Path, data: Dict, JSON: bool = False, YAML= True, sort_keys: bool = False, log_label:str = "Data") -> None: 
    exported = False
    file = Path(file)
    if JSON:
        with open(file.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, sort_keys=sort_keys)
        logger.info("%s saved as: %s", log_label, file.with_suffix(".json"))
        exported = True

    if YAML:
        with open(file.with_suffix(".yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, indent=4, sort_keys=sort_keys)
        logger.info("%s saved as: %s", log_label, file.with_suffix(".yaml"))
        exported = True

    if not exported:
        logger.warning("No file exported/saved. Check if both JSON and YAML are false.")


def _write_modelinput_npz(file: str | Path, X: np.ndarray,  y: np.ndarray | None = None) -> None:
    """Write ModelInput (spikedata) arrays to a single .npz file."""
    file = Path(file)
    if y is None:
        np.savez_compressed(file.with_suffix(".npz"), X=X)
    else:
        np.savez_compressed(file.with_suffix(".npz"), X=X, y=y)

    logger.info("ModelInput arrays saved as: %s", file.with_suffix(".npz"))




def _write_history_npz(file: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    """
    Write RunHistory arrays to a single npz file. (Variable number of arrays)
    
    Expected keys:
        spike_t              # sparse timestep coordinate
        spike_i              # sparse neuron coordinate
        spike_phase          # optional phase id per spike
        spike_epoch          # optional epoch per spike
        spike_datarun        # optional datarun per spike
        dense_spikeraster    # optional dense captured raster
    """
    file = Path(file)

    np.savez_compressed(file.with_suffix(".npz"), **arrays)

    logger.info("RunHistory arrays saved as: %s", file.with_suffix(".npz"))


def _write_checkpoint_npz(file: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write Checkpoint (TrainedModel) arrays to a single .npz file. """
    file = Path(file)

    np.savez_compressed(file.with_suffix(".npz"), **arrays)

    logger.info("Checkpoint arrays saved as: %s", file.with_suffix(".npz"))


def _write_snapshot_npz(file: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write protocol snapshot (resume) arrays to a single .npz file."""
    file = Path(file)
    tmp = file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(file)                       ##SAFETYVALVE: write-then-rename: a crash mid-write never leaves a half snapshot under the real name

    logger.info("Snapshot arrays saved as: %s", file)


def _write_responses_npz(file: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write protocol responses (spike counts, labels, assignments) to a single .npz file."""
    file = Path(file)

    np.savez_compressed(file.with_suffix(".npz"), **arrays)

    logger.info("Responses arrays saved as: %s", file.with_suffix(".npz"))


def _write_degrees_npz(file: str | Path, labels, degrees: Dict[str, np.ndarray]) -> None:
    """Write in- / out-degree per neuron at every saved weight state (labels = the states, degrees = {in_<group> | out_<group>: (states, neurons)})."""
    file = Path(file)
    np.savez(file.with_suffix(".npz"), labels=np.array(labels), **{k: np.array(v) for k, v in degrees.items()})
    logger.info("Degree arrays saved as: %s", file.with_suffix(".npz"))


def _read_responses_npz(file: str | Path) -> Dict[str, np.ndarray]:
    """Read protocol responses (the inverse of _write_responses_npz)."""
    with np.load(Path(file).with_suffix(".npz"), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}



####################################################################################
############################## HASHING ##############################

def _hash(canon, digest_size=None):
    """Hash using json.dumps to dump to str bytes, then blake2b"""
    if digest_size is None:
        raise ValueError(" Missing digest-size (length of hash)")
    canon_bytes = json.dumps(canon, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True).encode("ascii")
    return blake2b(canon_bytes, digest_size=digest_size).hexdigest()

def _hash_array(arr: np.ndarray, digest_size: int = 16) -> Dict:
    """Hash a NumPy array by dtype, shape, and raw bytes
    """
    array = np.ascontiguousarray(arr)

    hash = blake2b(digest_size=digest_size)
    hash.update(str(array.dtype).encode("ascii"))
    hash.update(str(tuple(array.shape)).encode("ascii"))
    hash.update(array.tobytes())

    return hash.hexdigest()


def _inf_safe(obj):
    """+-inf -> "inf"/"-inf" strings (deterministic), (since json with allow_nan=False rejects inf) strings are deterministic)"""
    import numpy as _np
    if isinstance(obj, dict): return {k: _inf_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_inf_safe(v) for v in obj]
    if isinstance(obj, float) and _np.isinf(obj): return "inf" if obj > 0 else "-inf"
    return obj



###ALGORITHM to handle np.arrays, classes, pytensors, ETC. in a STANDARDIZED way 
   ####-> generate JSON from pure pythonic objects 
    # JSON - https://elshad-karimov.medium.com/unleashing-the-power-of-functools-singledispatch-in-python-853a4a60b364
    # JSON - singledispatch (functools) https://docs.python.org/3/library/functools.html
    # YAML - https://medium.com/@reorx/tips-that-may-save-you-from-the-hell-of-pyyaml-572cde7e1d6f
    # YAML . https://pyyaml.org/wiki/PyYAMLDocumentation#dictionaries-without-nested-collections-are-not-dumped-correctly
