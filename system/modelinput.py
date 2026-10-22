#ModelInput - minimal translator wrapper between user preprocessing and PhasicNet.

from __future__ import annotations
from typing import Any, Iterator, Literal, Tuple
import copy          ##subset() = a shallow copy sharing source / targets / meta
import numpy as np
import logging

from utils import _hash, _hash_array


logger = logging.getLogger(__name__)

# Row-major flattening, recorded in meta.
FLATTEN_ORDER: Literal["C"] = "C"

## The count is the DATASET's: every subset() view (train / val / test / the per-epoch shuffles) ticks the same counter. 
BINARY_RECHECK_EVERY = 500

#####QUICKLINKS -> tonic source code:  
#######Datasets class: https://github.com/neuromorphs/tonic/blob/develop/tonic/dataset.py, 
####### ALL datasets have 1micro-s resolution

def _dataset_hash(data, *, digest_size: int = 16, max_samples: int | None = 3) -> tuple[str, int]:
    """
    Fingerprints length + the first [max_samples] samples. Two datasets that differ only past sample 3 collide

    Returns (digest, n): the id AND how many samples it was hashed over. 
    RunMeta records n as dataset_id_n_samples (n < max_samples when dataset shorter than window).
    """
    n = len(data) if max_samples is None else min(int(max_samples), len(data))
    frame_hashes = [_hash_array(data[i][0], digest_size=digest_size) for i in range(n)]
    ##len(data) is mixed in: a truncated or extended dataset must not share an id with the original.
    return _hash({"n": len(data), "samples": frame_hashes}, digest_size=digest_size), n


###############################################################################################################
############################################################################################################### 
############################### -  ModelInput (SpikeData)  - #####################################################################

class ModelInput:
    """
        Wrapper to translate dense spiking data for loading into PhasicNet.
    
    CONTRACT (dense):    one sample -> (T, N_in) array of spike values

        - axis 0 is time
        - all trailing axes are flattened using row-major into N_in (input neurons)
        - T (= #dt steps per sample) of may very per sample. 
        - trailing (non-time) axes must be FIXED across the dataset (ex. no variance in image-size or number of polarities recorded)
        - values must be binary spikes (dtype: bool or 0/1)

    Example:
            - sample shape (T, C, H, W) = (100, 2, 10, 10)
                - N_in = 2 * 10 * 10 = 200
                - Network used to run data must have 200 input neurons
    
                
    _Construction:_
    - default:  REQUIRED: (X: np.ndarray, y: np.ndarray, ...)
    - from tonic:  REQUIRED: (dataset: tonic.Dataset, ...)
    - OPTIONAL inputs:
        - meta: dict - Left untouched except the addition of "model_input" key. (If no dict provided, ModelInput generates one)
        - axis_labels: str - labels describing order using UNIQUE, SINGLE-letter labels. Gets recorded in attached meta.
            - (Ex. axis_labels = "THWC")
        - binary_check:bool = True - Ensure data is binary. Prevent silent transforms on runtime

    INTERNAL storage pattern:
        - targets is None:
            - source[i] returns (data, target)     # ex. a Tonic dataset
        - targets is not None:
            - source[i] returns data, targets[i] returns target    # ex. X / y arrays

    index = None is the whole source; subset(index, name) returns a VIEW 
    """

    __slots__ = ("source", "targets", "meta", "axis_labels", "frame_shape",
                 "sample_dtype", "binary_check", "index", "name", "_n_seen")

    def __init__(self, source, targets=None, *, axis_labels: str = None,
                 meta: dict | None = None, binary_check:bool = True ):

        if len(source) == 0:
            raise ValueError("No source entered.")
        if targets is not None and len(source) != len(targets):
            raise ValueError(f"Source and targets must have equal length. Got {len(source)} and {len(targets)}.")

        self.source = source
        self.targets = targets
        self.meta = {} if meta is None else dict(meta)
        self.binary_check = binary_check
        self.axis_labels = self._validate_axis_labels(axis_labels)
        self.frame_shape: Tuple[int, ...] = ()
        self.sample_dtype: np.dtype | None = None
        self.index: np.ndarray | None = None     ##None = everything; subset() sets it
        self.name: str = ""
        self._n_seen = [0]                       ## one-element list, shared by subset()

        # EAGER probe: frame_shape/n_input must be known BEFORE the network is built.
        self._validate_source()
        self._record_contract()
        self._update_meta()

        logger.info("ModelInput ready: n_samples=%d frame_shape=%s n_input=%d dtype=%s",
                    len(self), self.frame_shape, self.n_input, self.sample_dtype)

    @classmethod
    def from_tonic(cls, dataset, **kwargs) -> "ModelInput":
        """Tonic-style dataset where dataset[i] returns (frames, label), frames already dense AND binary."""
        return cls(source=dataset, targets=None, **kwargs)

    def subset(self, index, name: str = "") -> "ModelInput":
        """
        A VIEW of this ModelInput by index -- the (len, [i] -> (frames, target)) contract (avoid dublicating data)
        
        Shares source / targets / meta / frame_shape; composes indices, so "index" refers to the
        ORIGINAL source whatever the nesting (what datahandler.Subset's flattening did). "name" labels the view (train / val / test / ..._shuffled).
        ##NOTE TODO does this not duplicate getitem which already looks at the original source?
        """
        index = np.asarray(index, dtype=np.int64)
        if self.index is not None:
            index = self.index[index]
        view = copy.copy(self)          ##shallow: every slot shared, _n_seen included (a view continues the parent's count)
        view.index, view.name = index, name
        return view

############################################################################################################
############################### -  Validate & gather info  - ###############################################

    def _validate_axis_labels(self, axis_labels: str | None) -> str | None:
        """Descriptive only, for data.meta records, and for (later) reconstruction. Does NOT change how data is flattened."""
        if axis_labels is None:
            return None
        axis_labels = axis_labels.strip().upper()
        if not axis_labels.startswith("T"):
            raise ValueError(f"axis_labels must start with T. Got {axis_labels!r}.")
        if len(set(axis_labels)) != len(axis_labels):
            raise ValueError(f"axis_labels labels must be unique. Found repetition in the labels. {axis_labels!r}.")
        return axis_labels

    def _validate_source(self) -> None:
        """One-time check of source formatting & safety checks.
            - Ex. avoid unpacking of X array unpacking into (source, target) along the time dimension.""" 
        if self.targets is not None:
            return
        if isinstance(self.source, np.ndarray):
            raise TypeError("An array source requires targets: ModelInput(X, y). "
                            "Paired sources returning (data, target) go through ModelInput.from_tonic(dataset).")
        item = self.source[0]
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(f"With targets=None the source must return (data, target) pairs. "
                            f"Got {type(item).__name__} from source[0].")

    def _src(self, index) -> int:
        """a member "index" of this view -> its index in the ORIGINAL source (the identity when index is None).""" ##NOTE TODO does this not duplicate getitem which already looks at the original source?
        return int(index) if self.index is None else int(self.index[int(index)])

    def _raw(self, index) -> Tuple[Any, Any]:
        """Return (data, target) from the source, before any conversion through _src (a subset's members)."""
        index = self._src(index)
        if self.targets is None:
            data, target = self.source[index]
        else:
            data, target = self.source[index], self.targets[index]
        return data, target

    def _record_contract(self) -> None:
        """Read first sample at construction time to record frame_shape and dtype, and ensure binary (if chosen)"""
        data, _ = self._raw(0)
        data = np.asarray(data)

        if data.dtype.names is not None:
            raise TypeError("Structured event arrays are not accepted. Convert to dense frames first, ex. using tonic.transforms.ToFrame"
                             f"Got fields: {data.dtype.names}.")
        if data.ndim < 2:
            raise ValueError(f"Sample must have shape (T, ...). "
                             f"Got {data.shape}.")

        # T is deliberately NOT recorded: only the trailing (per-timestep) axes are fixed.
        frame_shape = tuple(int(n) for n in data.shape[1:])

        if self.axis_labels is not None and len(self.axis_labels) != data.ndim:
            raise ValueError(f"axis_labels {self.axis_labels!r} describes {len(self.axis_labels)} axes, but samples have shape {data.shape}.")
        if 0 in frame_shape:
            raise ValueError(f"Sample has a zero-length axis: {data.shape}.")

        self.frame_shape = frame_shape
        self.sample_dtype = data.dtype

        if self.binary_check:
            self._check_binary(data)

    @staticmethod
    def _check_binary(values: np.ndarray) -> None:
        """Reject non-spike-valued data (values outside {0, 1})."""
        if values.dtype == np.bool_:
            return
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"Non-numeric input data: {values.dtype}.")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"Input data is not spike-valued (found values outside {{0, 1}}: "
                             f"min={values.min()}, max={values.max()}). Encode to binary before wrapping.")

    def validate_network(self, network) -> None:
        """Fail loudly at bind time instead of as a broadcasting error inside the step loop."""
        n_input = getattr(network, "n_input", None)
        if n_input is None:
            raise TypeError(f"Expected a Network with n_input, got {type(network).__name__}.")
        if int(n_input) != self.n_input:
            raise ValueError(f"Input size mismatch: data flattens to N_in={self.n_input} "
                              f"(frame_shape={self.frame_shape}), but the network declares n_input={int(n_input)}.")
        logger.debug("ModelInput validated against network: n_input=%d", self.n_input)


#######################################################################################################
############################### - basics - ############################################################
    @property
    def n_input(self) -> int:
        """Number of input neurons the paired Network must declare."""
        return int(np.prod(self.frame_shape))

    @property
    def n_samples(self) -> int:
        return len(self)

    
    def __len__(self) -> int:
        if self.index is not None:      ## a subset's size ##TODO this assumed the original source has size as a func does it not? I have developed this for tonic and for personal code which I want to roughly replicate tonic. There seems to be a lot of extra work done in this file that assumes user and source behaviour I havent seen (unless you behaved this way when developing the experiment files, in which case we should update it to fit the actual system)
            return int(self.index.size)
        return len(self.source) ##TONIC returns number of samples here (by default), same as arrays will (assuming contract adherance)

    def __getitem__(self, index) -> Tuple[np.ndarray, Any]:
        data, target = self._raw(index)
        return self.to_TI(data), target

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Any]]:
        for index in range(len(self)):
            yield self[index]

    def labels(self) -> np.ndarray:
        """
        Retrieve labels (targets) in source order.
        
        * Searches for targets fastest ways first:
            1. search self.targets
            2. search source for targets attr
            3. iter through src_dataset for targets
        NOTE a subset's labels are the source's at its index (1. and 2. index the full vector; 3. iterates the members only).
        """
        if self.targets is not None:
            y = np.asarray(self.targets)
        else:
            src_targets = getattr(self.source, "targets", None)
            if src_targets is not None and getattr(self.source, "target_transform", None) is None and len(src_targets) == len(self.source):
                y = np.asarray(src_targets)
            else:
                return np.asarray([self._raw(index)[1] for index in range(len(self))])
        return y if self.index is None else y[self.index]

    def variant(self, index, **kw) -> np.ndarray:  
        """
        Return another rendering of sample 'index' via request for:  variant(index, **kw)  (DC protocol: re-presentation)
         * A source without variant() -> AttributeError
         NOTE: "index" is a member of this view (a subset asks for its original)"""
        return self.to_TI(self.source.variant(self._src(index), **kw))
            
    def __repr__(self) -> str:
        view = "" if self.index is None else f"{self.name!r} view of {len(self.source)}, "
        return (f"ModelInput({view}n_samples={len(self)}, frame_shape={self.frame_shape}, "
                f"n_input={self.n_input}, dtype={self.sample_dtype}, axis_labels={self.axis_labels!r})")

    def _update_meta(self) -> None:
        model_input_meta = dict(self.meta.get("model_input", {}))
        model_input_meta.update({
            "dense_contract": "(T, ...) -> (T, N_in)",
            "n_input": self.n_input,
            "n_samples": len(self),
            "axis_labels": self.axis_labels,
            "frame_shape": list(self.frame_shape),
            "flatten_order": FLATTEN_ORDER,
            "sample_dtype": str(self.sample_dtype)
            })
        self.meta["model_input"] = model_input_meta


#################################################################################################################################################
############################### - Reconstruction, coordinates & indexing, & hashing - ############################################################

    def neuron_index(self, *coordinates) -> int:
        """Map original (non-time) coordinates -> flat input-neuron index."""
        return int(np.ravel_multi_index(coordinates, self.frame_shape, order=FLATTEN_ORDER))

    def neuron_coordinates(self, index: int) -> Tuple[int, ...]:
        """Map flat input-neuron index -> original (non-time) coordinates."""
        return tuple(int(value) for value in np.unravel_index(index, self.frame_shape, order=FLATTEN_ORDER))

    def reconstruct_step(self, flat_step) -> np.ndarray:
        """Reshape one flattened timestep back to the original sample geometry."""
        flat_step = np.asarray(flat_step)
        if flat_step.size != self.n_input:
            raise ValueError(f"Expected {self.n_input} input values, got {flat_step.size}.")
        return flat_step.reshape(self.frame_shape, order=FLATTEN_ORDER)


    def dataset_hash(self, *, digest_size: int = 16, max_samples: int | None = 3) -> str:
        """
        Hash the FLATTENED arrays only & save hash + number of samples hashed in meta.
        - NOTE: two datasets that flatten to identical arrays get identical hashes regardless of meta

        Args:
            max_samples: hash only the first N samples
           
        """
        digest, n = _dataset_hash(self, digest_size=digest_size, max_samples=max_samples)
 
        self.meta["model_input"]["dataset_id"] = digest
        self.meta["model_input"]["dataset_id_n_samples"] = n
 
        logger.info("ModelInput dataset_id=%s, hashed over %d/%d samples", digest, n, len(self))
        return digest    

#######################################################################################################
############################### -  Transforms - flatten array - ###############################################


    def to_TI(self, data) -> np.ndarray:
        """Convert one (T, ...) sample to (T, N_in). Pure, view returned -- does not mutate ModelInput state. Include security checks of data."""
        data = np.asarray(data)

        # double check for structured arrays (not accepted in dataloader)
        if data.dtype.names is not None:
            raise TypeError("Data must be in unstructured arrays. Got structured array. Convert to dense frames first, ex. using tonic.transforms.ToFrame.")

        #double check that trailing axis lenghts do not drift; N_in must stay CONSTANT & receive from the SAME coordinates THROUGHOUT a run
        if tuple(data.shape[1:]) != self.frame_shape:
            raise ValueError(f"Expected a sample of shape (T, {self.frame_shape}), got {data.shape}. "
                              "PhasicNet requires a fixed N_in across the dataset.")

        ##cheap periodic binary re-check 
        if self.binary_check and data.dtype != np.bool_:
            self._n_seen[0] += 1
            if self._n_seen[0] % BINARY_RECHECK_EVERY == 0:
                self._check_binary(data)

        frames = data.reshape(data.shape[0], -1, order=FLATTEN_ORDER)
        # NUMPY flag: Read-only VIEW to prevent silent data drift
        frames.flags.writeable = False
        return frames