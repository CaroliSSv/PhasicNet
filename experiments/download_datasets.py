#download_datasets - ONE-TIME download of the raw datasets into datasets/raw/, on a new machine before any run starts.
#
#   python -c "import experiments, download_datasets as d; d.main()"                    every dataset below
#   python -c "import experiments, download_datasets as d; d.main(('mnist', 'shd'))"     only some
#
#Already downloaded = skipped (MNIST: the file exists; tonic sets: tonic checks its own files). Each line ends with the sample count.
#Sizes (download): mnist 11 MB, fmnist 30 MB, shd ~0.5 GB, nmnist ~1 GB, dvsgesture ~2.5 GB (+ the same again unpacked).
from __future__ import annotations
from datetime import datetime
import numpy as np
from paths import RAW_DIR
from utils import import_mnist


ALL = ("mnist", "fmnist", "shd", "nmnist", "dvsgesture")


def tonic_set(name: str):
    """Constructing a tonic dataset downloads and unpacks it into RAW_DIR (where the builds look); train and test."""
    import tonic
    cls = {"shd": tonic.datasets.SHD, "nmnist": tonic.datasets.NMNIST, "dvsgesture": tonic.datasets.DVSGesture}[name]
    return {split: len(cls(save_to=str(RAW_DIR), train=split == "train")) for split in ("train", "test")}


def say(msg):
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)    

def main(names=ALL):
    for name in names:
        say(f"{name} ...")
        try:
            if name == "mnist":
                with np.load(import_mnist("digits")) as z:
                    out = {"train": len(z["y_train"]), "test": len(z["y_test"])}
            elif name == "fmnist":
                with np.load(import_mnist("fashion")) as z:
                    out = {"train": len(z["y_train"]), "test": len(z["y_test"])}
            else:
                out = tonic_set(name)
            say(f"{name} OK: {out}")
        except Exception as e:          ##catch exception to avoid one download fail stopping others
            say(f"{name} FAILED: {e!r}")
