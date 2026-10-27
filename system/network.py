from __future__ import annotations 
from dataclasses import dataclass, asdict, replace
from typing import Optional, Dict, Tuple, NamedTuple
from datetime import datetime
import logging
from meta import NetMeta
import numpy as np

from paths import EXPORT_DIR
from utils import _write_json_yaml, _write_net_npz, _hash, _read_net_npz, _read_json_yaml

logger = logging.getLogger(__name__)


##############################################################################
############################## Specs  ##############################   
class WeightProjection(NamedTuple):
    """ONE projection rectangle of W: (post_rows, pre_cols) pair (TUPLE of slices).

    post = post neurons (band of W ROWS). Shrinks under post_offset compression (variable length = n_post)
    pre  = pre-neurons (band of W COLUMNS). ALWAYS n_total, unaffected by post_offset. ALWAYS carries true neuron ids.""" 
    post: slice
    pre: slice


@dataclass(slots=True)
class SynapseSpec: 
    """
    Specifications for a group of edges/synapses with shared characteristics to be realized. 
    syn_type = name of group 
    pre_neuron = type of neuron (presynaptic)
    post_neuron = type of neuron (postsynaptic)
    w0 = Initial weights
    p_connection = probability of connection between pre and post neurons defined in this synapse group
                => If distance kernel used -> dict: {kernel: str, C: float in range: (0-1], lambda: float} SRC: Deckers2022, Maass2002
                => kernel: gaussian = "gaussian distance-decay function" used to calculate probability of connection from distance between nodes (gaussian distribution)
                => C = maximum probability of connection at 0 distance
                => lambda = shapes how fast probability of connection falls with distance

    ####NOTE POSSIBLE ERROR: Definition of two groups W/ SAME pre and post neurons, but ex. different W0 might ADD TO or OVERWRITE connections

    """
    ##FILLED ON INSTANTIATION of SynapseSpec using set_synapse:
    syn_type: str
    pre_neuron: str
    post_neuron: str

    ## FILLED LATER w/ setters:
    w0: Optional[float | tuple] = None   ##scalar = every edge; (lo, hi) = uniform per edge
    p_connection: Optional[float | str | dict] = None   ##float in [0,1], or a named pattern (builder.CONNECTIVITY_PATTERNS) or a distance (kernel + decay descriptors)


@dataclass(slots=True)
class NeuronSpec:
    """
    Specifications for nodes/neurons to be realized
    neur_type = name of neuron-group with shared characteristics
    n = Number of neurons of this type in the network
    input = Define input nodes (NOT true neurons. These receive data from dataloader during runtime)
    """
    neur_type: str
    n: int
    input: bool = False
    inh: bool = False          

    grid: Optional[Tuple[int, ...]] = None ##spatial -- shape of grid group is in
    grid_share: Optional[str] = None #spatial - grid name (for shared grids)





#######################################################################################
 ############################## CANONICALIZATION ####################################

def _counts(neuronspecs) -> Dict:
    """Count summaries for saving. (network property during runtime)."""
    return {
        "n_total": sum(spec.n for spec in neuronspecs.values()),
        "n_input": sum(spec.n for spec in neuronspecs.values() if spec.input),
        "n_inh": sum(spec.n for spec in neuronspecs.values() if spec.inh) }


def _decompress(n_input, post_idx: np.ndarray | None = None, synapseslices: Dict | None = None) -> np.ndarray | Dict:
    if post_idx is not None:
        return post_idx + n_input

    if synapseslices is not None:
        decompressed_synapseslices = {}

        for syn, (post_slc, pre_slc) in synapseslices.items():
            decompressed_post_slc = slice(post_slc.start + n_input, post_slc.stop + n_input, post_slc.step)

            decompressed_synapseslices[syn] = WeightProjection(post=decompressed_post_slc, pre=pre_slc)

        return decompressed_synapseslices

    raise ValueError("Provide post_idx or synapseslices.")



def _canonicalize_netspecs(neuronspecs, synapsespecs):
    """Network section 1; neuron and synapse specifications"""
    canon_specs = {
        "neuronspecs": {k: asdict(v) for k, v in neuronspecs.items()},
        "synapsespecs": {k: asdict(v) for k, v in synapsespecs.items()}
        }
    return canon_specs


def _canonicalize_netdiv(neuronslices, synapseslices, type_key):
    """Network section 2: realized layout (neuron + synapse slices) and COO neurontype integer key"""
    canon_div= {
        "neuronslices": { key: [slc.start, slc.stop, slc.step]
            for key, slc in neuronslices.items()},
        "synapseslices": {key: [[post_slc.start, post_slc.stop, post_slc.step],[pre_slc.start, pre_slc.stop, pre_slc.step]]
            for key, (post_slc, pre_slc) in synapseslices.items()},
        "type_key": type_key
        }
    return canon_div


def _canonicalize_netlists(syntype, pre_idx, post_idx, W0, positions=None):
    """Network section 3: Full arrays to lists for hashing (only). Grid positions included only when a group has a grid."""
    canon_lists = {
        "syntype": syntype.tolist(),
        "pre_idx": pre_idx.tolist(),
        "post_idx": post_idx.tolist(),
        "W0": W0.tolist()
    }
    if positions:
        canon_lists["positions"] = {g: np.asarray(positions[g]).tolist() for g in sorted(positions)}
    return canon_lists


def _canonicalize_netmeta(meta):
    """Network section 3: NetMeta object on RNG settings, specID and network ID (realized) for """
    canon_netmeta = {"netmeta": asdict(meta)}
    return canon_netmeta



############## INSTANTIATE SPECS #######################


def _canon_to_specs(raw_data: Dict) -> Tuple[str, Dict[str, NeuronSpec], Dict[str, SynapseSpec]]:
    spec_name = raw_data.get("spec_name")
    neuronspecs = { key: NeuronSpec(**specs)
        for key, specs in raw_data["neuronspecs"].items()  
    }
    for spec in neuronspecs.values():               ##spatial: grid->list
        if isinstance(spec.grid, list):
            spec.grid = tuple(int(g) for g in spec.grid)
    synapsespecs = { key: SynapseSpec(**specs)
        for key, specs in raw_data["synapsespecs"].items()
    }
    return spec_name, neuronspecs, synapsespecs



#######################################################################################
  ################################ Network #########################################


class Network:
    """
    Realized network w/ metadata: 

        netspec = Network spec
        N_total = Total number of neurons
        N_input = Number of input neurons
        neuronslices = slices w/ ID ranges of neurontypes
        synapseslices = slices w/ ID ranges of synapse types
        syntype = COO; list of types per synapse
        pre_idx = COO; list of presynaptic neuronID (true_neuronID = W column)
        post_idx = COO; list of postsynaptic neuronID (row_idx = W row: true_neuronID - row_offset)
        W0 = COO; list of initial weights per synapse
        meta = NetMeta instance: seed and rng_handling copy
        positions = {group: (n, ndim) grid positions} of the groups placed with set_positions. Default: {} without a grid. saved as pos_<group>
        
    """

    def __init__(self,  neuronspecs: Dict[str, NeuronSpec], synapsespecs: Dict[str, SynapseSpec],
                 neuronslices: Dict[str, slice], synapseslices: Dict[str, WeightProjection], type_key: Dict[str, int], 
                 syntype: np.ndarray, pre_idx: np.ndarray, post_idx: np.ndarray, W0: np.ndarray, meta: NetMeta,
                 positions: Dict[str, np.ndarray] | None = None):   

        self.neuronspecs: Dict[str, NeuronSpec] = neuronspecs
        self.synapsespecs: Dict[str, SynapseSpec] = synapsespecs
        ##SINGE compute on network creation (import/realize)
        self.n_total: int = sum(spec.n for spec in neuronspecs.values())
        self.n_input: int = sum(spec.n for spec in neuronspecs.values() if spec.input)
        self.n_inh: int = sum(spec.n for spec in neuronspecs.values() if spec.inh)

        #### Slices 
        self.inh_slice:int = self._calc_inh_slice(neuronspecs, neuronslices, self.n_inh) ##Computed ONCE
        self.neuronslices: Dict[str, slice] = neuronslices
        self.synapseslices: Dict[str, WeightProjection] = synapseslices

        self.type_key: Dict[str, int] = type_key

        self.syntype: np.ndarray = syntype
        self.pre_idx: np.ndarray = pre_idx
        self.post_idx: np.ndarray = post_idx
        self.W0: np.ndarray = W0
        ## meta (class)
        self.meta: NetMeta = meta
        self.positions: Dict[str, np.ndarray] = {} if positions is None else dict(positions)     # {} when no group has a grid

        
        self.row_offset: int = self.n_total - self.n_post       ##n_input under post_offset, else 0
        self.neuronrows: Dict[str, slice] = self._calc_neuronrows(neuronspecs, neuronslices, synapsespecs, synapseslices, self.row_offset, self.post_offset)
        
        self.synapse_counts: Dict[str, int] = self._calc_synapse_counts(syntype, type_key)



    @staticmethod
    def _calc_inh_slice(neuronspecs:dict, neuronslices:dict, n_inh) -> slice:
        """ONE-time call rather than recalc every step. 
        Return contiguous range of inhibitory neurons. slice(0, 0) when there are none."""
        start, stop = None, 0
        for neur_type, spec in neuronspecs.items():
            if not spec.inh:
                continue
            slc = neuronslices[neur_type]
            if start is None or slc.start < start:
                start = slc.start
            stop = max(stop, slc.stop)

        if start is None:
            return slice(0, 0)

        ##SAFETYVALVE: verify contiguity (_order_neuronspecs should have done this)
        expected = n_inh
        if (stop - start) != expected:
            raise RuntimeError(f"Inhibitory neurons are not contiguous. {expected} inh neurons fouond should cover"
                               f"slice(start, stop): slice({start}, {stop}) but doesn't.")
        return slice(start, stop)

    @staticmethod
    def _calc_neuronrows(neuronspecs: dict, neuronslices: dict, synapsespecs: dict, synapseslices: dict, row_offset: int, post_offset: bool) -> Dict[str, slice]:
        """
        Lookup: Neuron group -> its W row-band (neuronrows) (row_idx),
        """
        rows = {neur_type: slice(slc.start - row_offset, slc.stop - row_offset)
                for neur_type, slc in neuronslices.items() if not (post_offset and neuronspecs[neur_type].input)}
        ##SAFETYVALVE: every synapse group's post band (builder's own mapping) must be the row band of its post group
        for syn, spec in synapsespecs.items():
            post = synapseslices[syn].post
            if rows.get(spec.post_neuron) != post:
                raise RuntimeError(f"Synapse group '{syn}': post rows {post} disagree with the row band of '{spec.post_neuron}' "
                                   f"({rows.get(spec.post_neuron)}). Check builder._assign_synapseslices.")
        return rows

    @staticmethod
    def _calc_synapse_counts(syntype: np.ndarray, type_key: dict) -> Dict[str, int]:
        """Realized (post-deduplication) synapse count per group: {"n_<synapse_group_name>": count}."""
        counts = np.bincount(np.asarray(syntype, dtype=np.int64), minlength=len(type_key))
        return {f"n_{syn}": int(counts[key]) for syn, key in type_key.items()}


    #################### is-compressed check ##################
    
    @property
    def post_offset(self):
        """Whether dense weights: (W = [post, pre] ) has had rows with an input-group as the post-synaptic neuron deleted. (Deletion assumes input neurons are NEVER receiving synapses.)
        
        Compresses network, saves computational burden. 
        
        Downstream consequences: input neurons have no NeuronalState and no post_trace, post_spike.
        (Modifies lenght of certain arrays during instantiation. See notes.)"""
        return self.meta.post_offset
    
    #####################################################
    ################  --  Counts --  ####################

    @property
    def n_synapses(self) -> int:
        """REALIZED synapse count, derived from COO"""
        return int(self.pre_idx.size)

    @property
    def n_rec(self) -> int:
        return self.n_total - self.n_input

    @property
    def n_exc(self) -> int:
        """Excitatory neurons (input nodes delivering excitatory drive included)."""
        return self.n_total - self.n_inh

    @property
    def n_post(self) -> int:
        """Number of rows (post-synaptic neurons) in dense W array (depends on offset)"""
        if self.meta.post_offset:
            return self.n_rec
        else:
            return self.n_total
        

    @property
    def n_inh_input(self) -> int:    ##REMOVE?
        """Input nodes delivering INHIBITORY drive"""
        return sum(spec.n for spec in self.neuronspecs.values() if spec.inh and spec.input)
    @property
    def n_exc_input(self) -> int:    ##REMOVE?
        """Input nodes delivering EXCITATORY drive (default)"""
        return self.n_input - self.n_inh_input
    


    #############################################################
    ################  --  Slices & masks --  ####################

    @property
    def rec_slice(self) -> slice:
        """The recurrent neurons' band of W ROWS -- correct regardless of offset.

        offset    -> slice(0, n_rec)          rows ARE the recurrent set
        no offset -> slice(n_input, n_total)  inputs occupy the first n_input rows.
        """
        return slice(self.n_post - self.n_rec, self.n_post)
    
     
    @property
    def inh_neuron_mask(self) -> np.ndarray:      ##REMOVE? (redundant when slice exists)
        """Bool (n_total,) -- True where the neuron is inhibitory. 
        Easy to use, but heavy computationally if used in hot-loop. Don't. Use inh_slice
        """
        mask = np.zeros(self.n_total, dtype=bool)
        mask[self.inh_slice] = True
        return mask


    #####################################################
    ################  -- Identity --  ####################

    @property
    def network_id(self):
        return self.meta.network_id
    
    @property
    def network_name(self):
        return self.meta.network_name
    
    @network_name.setter
    def network_name(self, new):
        old=self.network_name
        self.meta = replace(self.meta, network_name=new)
        logger.info("Renamed network. Old: %s, New: %s", old, self.network_name)
    
    @property
    def spec_id(self):
        return self.meta.spec_id
    
    @property
    def spec_name(self):
        return self.meta.spec_name
    
    @spec_name.setter
    def spec_name(self, new):
        old=self.spec_name
        self.meta = replace(self.meta, spec_name=new)
        logger.info("Renamed spec. Old: %s, New: %s", old, self.spec_name)

    #################### canonicalize self: ########################

    def _canonicalize_netspecs(self):
        return _canonicalize_netspecs(neuronspecs=self.neuronspecs, synapsespecs=self.synapsespecs)

    def _canonicalize_netdiv(self):
        return _canonicalize_netdiv(neuronslices=self.neuronslices, synapseslices=self.synapseslices, type_key=self.type_key)

    def _canonicalize_netlists(self):
        return _canonicalize_netlists(syntype=self.syntype, pre_idx=self.pre_idx, post_idx=self.post_idx, W0=self.W0, positions=self.positions)
    
    def _canonicalize_netmeta(self):
        return _canonicalize_netmeta(meta=self.meta)


    #################### rehash check: ########################

    def rehash_network(self):
        """Mirrors _generate_netmeta function. Exists for checking hash (accidental network drift/change => new hash).
        Decompresses a compressed dense network (ie post-n_inputs removed)"""
        if self.post_offset:
            normalized_synapseslices = _decompress(self.n_input, synapseslices=self.synapseslices)
            normalized_post_idx = _decompress(self.n_input, post_idx=self.post_idx)

        else:
            normalized_synapseslices = self.synapseslices
            normalized_post_idx = self.post_idx

        canon_specs = _canonicalize_netspecs(self.neuronspecs, self.synapsespecs)
        canon_div = _canonicalize_netdiv(self.neuronslices, normalized_synapseslices, self.type_key)
        canon_lists = _canonicalize_netlists(self.syntype, self.pre_idx, normalized_post_idx, self.W0, self.positions)

        spec_id = _hash(canon_specs, digest_size=8)

        canon_ID = canon_specs | canon_div | canon_lists
        network_id = _hash(canon_ID, digest_size=16)

        if self.meta.spec_id == spec_id:
            logger.info("Spec ID consistent. Spec_ID: %s", self.meta.spec_id)  
            
        if self.meta.network_id == network_id:
            logger.info("Network ID consistent. Network_ID: %s", self.meta.network_id) 

        if self.meta.network_id != network_id:
            logger.warning("Network_ID inconsistent: Previous: '%s' , new: '%s'.", self.meta.network_id, network_id)

        if self.meta.spec_id != spec_id:
            logger.warning("Spec_ID inconsistent: Previous: '%s' , new: '%s'.", self.meta.spec_id, spec_id)
    


    ####################################################################
    ############################## load: ##############################

    @classmethod
    def load(cls, raw_data: Dict, syntype: np.ndarray, pre_idx: np.ndarray, post_idx: np.ndarray, W0: np.ndarray,
             positions: Dict[str, np.ndarray] | None = None) -> Network:

        _, neuronspecs, synapsespecs = _canon_to_specs(raw_data) 
        
        neuronslices = { key: slice(*value)
            for key, value in raw_data["neuronslices"].items() }
        
        ##Repacked -> WeightProjections from loaded tuple w/ slices
        synapseslices = { key: WeightProjection(post=slice(*value[0]), pre=slice(*value[1]))
            for key, value in raw_data["synapseslices"].items() }
        
        type_key = { key: value 
                    for key, value in raw_data['type_key'].items() }

        meta = NetMeta(**raw_data["netmeta"])
        
        network = cls(neuronspecs= neuronspecs, synapsespecs=synapsespecs, 
                          neuronslices= neuronslices, synapseslices= synapseslices, type_key=type_key, 
                          syntype=syntype, pre_idx=pre_idx, post_idx = post_idx, W0=W0, positions=positions, meta=meta)

        ## SAFETYCHECK: make sure totals concurr between saved truth and realized network
        stored = {key: raw_data[key] for key in ("n_total", "n_input", "n_inh") if key in raw_data}
        derived = _counts(neuronspecs)
        mismatched = {key: (value, derived[key]) for key, value in stored.items() if value != derived[key]}
        if mismatched:
            raise ValueError(f"Saved network summary does not agree with its own specifications (specs)."
                             f"Disagreements: {mismatched}..")

        return network

    @classmethod
    def load_file(cls, file, fmt=None) -> Network:
        syntype, pre_idx, post_idx, W0, positions = _read_net_npz(file)   # already swaps .json -> .npz
        return cls.load(_read_json_yaml(file, fmt), syntype, pre_idx, post_idx, W0, positions)



    #########################################################################################################
    ###################################### save() - SAVE TO FILES: #######################################


    def save_specs(self, spec_name = None, filename: str = None, JSON: bool = False, YAML: bool = True, sort_keys: bool = False, save_to=EXPORT_DIR):
        """
        Save specs to JSON or YAML with optional OVERWRITE of spec-name.
        Arg:
            filename=None auto-generates based on spec_name if present or spec_id
            spec_name=None
            JSON: False
            YAML: True
            sort_keys: False
            save_to: Output folder"""

        if spec_name is None:
            spec_name = self.spec_name

        spec_name_entry = {"spec_name": spec_name}
        canon_specs = self._canonicalize_netspecs()
        export_specs = spec_name_entry | canon_specs

        if filename is None:
            now = datetime.now().strftime("%d-%m_%H%M%S")
            filename = f"NetSpec_{now}_{self.meta.spec_name}" if self.meta.spec_name is not None else f"NetSpec_{now}_{self.meta.spec_id}"
        
        file = save_to / filename
        _write_json_yaml( data=export_specs, file=file, JSON=JSON, YAML=YAML, sort_keys=sort_keys, log_label="Netspec")


    def save_net(self, filename: str = None, network_name=None, spec_name=None, JSON=True, YAML=False, sort_keys=False, save_to= EXPORT_DIR):
        """Save instantiated network object. Optionally overwrite network name or spec name with new name."""
        if network_name is None:
            network_name = self.meta.network_name
        if spec_name is None:
            spec_name = self.meta.spec_name

        canon_specs = self._canonicalize_netspecs()
        canon_div = self._canonicalize_netdiv()
        canon_netmeta = self._canonicalize_netmeta()
        canon_network = canon_netmeta | _counts(self.neuronspecs) | {"synapse_counts": self.synapse_counts}| canon_specs | canon_div   

        if self.meta.spec_id is None or self.meta.network_id is None:
            raise ValueError("Error with NetMeta object. Hashes missing.")

        if filename is None:
            now = datetime.now().strftime("%d-%m_%H%M%S")
            filename = (f"Network_{now}_{self.meta.network_name}") if self.meta.network_name is not None else (f"Network_{now}_{self.meta.network_id}")

        file = save_to / filename
        _write_json_yaml( data=canon_network, file=file, JSON=JSON, YAML=YAML, sort_keys=sort_keys, log_label="Network")

        if JSON or YAML:
            _write_net_npz(file, self.syntype, self.pre_idx, self.post_idx, self.W0, positions=self.positions)  
