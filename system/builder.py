from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Optional, Dict, Tuple, Literal, Callable
from datetime import datetime
from pathlib import Path
import weakref
import logging

import numpy as np

from paths import EXPORT_DIR
from utils import _read_json_yaml, _write_json_yaml, _hash, _filepath
from freeze import reserved_name, FREEZE_KEYWORDS   ##name-collision guard (WeightProjection comes from network)
from network import WeightProjection, NetMeta, Network, NeuronSpec, SynapseSpec, _canon_to_specs, _canonicalize_netspecs, _canonicalize_netdiv, _canonicalize_netlists, _decompress

logger = logging.getLogger(__name__)

CONNECTIVITY_PATTERNS: Dict[str, Callable[[int, int], np.ndarray]] = {
    "all_to_all":   lambda pre_n, post_n: np.ones((pre_n, post_n), dtype=bool),
    "one_to_one":   lambda pre_n, post_n: np.eye(pre_n, post_n, dtype=bool),
    "all_but_self": lambda pre_n, post_n: ~np.eye(pre_n, post_n, dtype=bool)}
_SQUARE_PATTERNS = ("one_to_one", "all_but_self")  
DISTANCE_KERNELS = ("gaussian",)  ## (spatial): distance kernels  

#(spatial): safetyvalve 
def _check_kernel(syn_type: str, prob: dict) -> dict:
    
    kernel = prob.get("kernel")
    if kernel not in DISTANCE_KERNELS:     
        raise ValueError(f"Connectivity for '{syn_type}': unknown kernel {kernel!r}. Known kernels: {DISTANCE_KERNELS}.")
    missing = [k for k in ("C", "lambda") if k not in prob]
    if missing or set(prob) - {"kernel", "C", "lambda"}:
        raise ValueError(f"Connectivity for '{syn_type}': a kernel needs exactly the keys kernel, C, lambda. Got {sorted(prob)}.")
    C, lam = float(prob["C"]), float(prob["lambda"])
    if not (0.0 <= C <= 1.0):
        raise ValueError(f"Connectivity for '{syn_type}': C must be in [0, 1], got {C!r}.")
    if not lam > 0.0:
        raise ValueError(f"Connectivity for '{syn_type}': lambda must be > 0, got {lam!r}.")
    return {"kernel": kernel, "C": C, "lambda": lam}


def _grid_cells(grid: Tuple[int, ...]) -> np.ndarray:
    """C_order: (prod(grid), ndim) """
    return np.stack(np.unravel_index(np.arange(int(np.prod(grid))), grid), axis=1).astype(np.int16)


def _grid_positions(neuronspecs: Dict[str, NeuronSpec], rng) -> Dict[str, np.ndarray]:
    """Group -> (n, ndim) positions for every group with NeuronSpec.grid. 
    Alone: the first n cells in C order. 
    Shared (grid_share): one keyed permutation rng("network", "positions:<share>") of grid's cells, handed to groups in NAME order (neurongroup-order agnostic). Raises if a grid is too small."""
    positions: Dict[str, np.ndarray] = {}
    shares: Dict[str, list] = {}
    for name in sorted(neuronspecs):
        spec = neuronspecs[name]
        if spec.grid is None:
            continue
        if spec.grid_share is None:
            if spec.n > int(np.prod(spec.grid)):
                raise ValueError(f"Neuron group '{name}': n={spec.n} does not fit on grid {spec.grid} ({int(np.prod(spec.grid))} cells).")
            positions[name] = _grid_cells(spec.grid)[:spec.n]
        else:
            shares.setdefault(spec.grid_share, []).append(name)
    for share, names in shares.items():
        grids = {tuple(neuronspecs[n].grid) for n in names}
        if len(grids) != 1:
            raise ValueError(f"Shared grid '{share}': groups {names} must have the same grid, got {sorted(grids)}.")
        grid = grids.pop()
        n_cells, n_needed = int(np.prod(grid)), sum(neuronspecs[n].n for n in names)
        if n_needed > n_cells:
            raise ValueError(f"Shared grid '{share}' {grid} has {n_cells} cells for {n_needed} neurons ({names}).")
        cells = _grid_cells(grid)[rng.rng("network", f"positions:{share}").permutation(n_cells)]
        start = 0
        for n in names:
            positions[n] = cells[start:start + neuronspecs[n].n]
            start += neuronspecs[n].n
    return positions


def _kernel_mask(syn: str, kernel: dict, pos_pre: np.ndarray, pos_post: np.ndarray, rng) -> np.ndarray:
    """(src_Maass2002, Deckers2022)"""
    diff = pos_pre[:, None, :].astype(np.float64) - pos_post[None, :, :]
    D2 = np.einsum("ijk,ijk->ij", diff, diff)                              ##squared distance, no sqrt needed: (D / lambda)^2 = D2 / lambda^2
    P = kernel["C"] * np.exp(-D2 / (kernel["lambda"] ** 2))
    return rng.random(P.shape, dtype=np.float32) < P

############## Generate META #######################

def _generate_netmeta(neuronspecs, synapsespecs, spec_name, network_name, 
                      neuronslices, synapseslices, type_key,
                      syntype, pre_idx, post_idx, W0, 
                      post_offset, n_input, rng, positions=None):
    """Generate meta object BEFORE Network instantiation"""
    ## HASH canon objects
    if post_offset:
        normalized_synapseslices = _decompress(n_input, synapseslices=synapseslices)
        normalized_post_idx = _decompress(n_input, post_idx=post_idx)

    else:
        normalized_synapseslices = synapseslices
        normalized_post_idx = post_idx


    canon_specs = _canonicalize_netspecs(neuronspecs, synapsespecs)
    canon_div = _canonicalize_netdiv(neuronslices, normalized_synapseslices, type_key)
    canon_lists = _canonicalize_netlists(syntype, pre_idx, normalized_post_idx, W0, positions)

    spec_id = _hash(canon_specs, digest_size=8)

    canon_ID = canon_specs | canon_div | canon_lists
    network_id = _hash(canon_ID, digest_size=16)

    globalseed = rng.seed
    rng_spec = rng.rng_spec

    meta = NetMeta(spec_id=spec_id, spec_name=spec_name, network_id=network_id, network_name=network_name, 
                   post_offset=post_offset, globalseed=globalseed, rng_spec=rng_spec)
    return meta

##############################################################################
############################## NETWORK BUILDER  ##############################

class NetBuilder:
    
    def __init__(self, rng, pnet = None):
        self._pnet_ref = (weakref.ref(pnet) if pnet is not None else None) ##CIRCULAR reference, but used for returning realized network to pnet ABOVE
        self.rng = rng
        self.neuronspecs: Dict[str, NeuronSpec] = {}
        self.synapsespecs: Dict[str, SynapseSpec] = {}  ##NOTE not mutable-default silent gen issue because it is INSIDE __init__, so new net gets empty dict
        self.spec_name: str = None

    def _canonicalize_netspecs(self):
        return _canonicalize_netspecs(neuronspecs=self.neuronspecs, synapsespecs=self.synapsespecs)

    ##############################################################################
    ############################## DEFINE NEURONS  ##############################

    def load_specs(self, file: str, fmt: Literal["json", "yaml", "yml"] | None = None) -> None:
        """Load specs dictionaries from file to builder. 
            Looks in default_dir = root/exports by default."""
        file = _filepath(file)

        raw_data = _read_json_yaml(file=file, fmt=fmt)
        spec_name, neuronspecs, synapsespecs = _canon_to_specs(raw_data)

        self.neuronspecs = neuronspecs
        self.synapsespecs = synapsespecs
        self.spec_name = spec_name
        logger.info("Loaded specs from file='%s' fmt='%s': neurons=%s synapses=%s", file, fmt, 
                    self.neuronspecs.keys(), 
                    self.synapsespecs.keys())


    def set_neurons(self, *type_number: Tuple[str, int] | Tuple[str, int, bool], clear: bool = False) -> "NetBuilder":
        """
        Define and declare neuron groups using name/type of synapse, number of neurons of this type

        Ex.:   (neur_type, number, optional(input = TRUE/False), optional(inh = True/False)). when optional, False is the default  

        network.set_neurons(("input", 20, True), ("exc", 80), ("inh", 10))
            NOTE: number must be positive int.
        
        clear=True clears existing node_specs first
        """
        if clear:
            self.neuronspecs.clear()
            logger.debug("Previous neurons cleared.")

        for arg in type_number:
            if len(arg) == 2:
                neur_type, number = arg
                input = False
                inh = False
            elif len(arg) == 3:
                neur_type, number, input = arg
                inh = False
            elif len(arg) == 4:
                neur_type, number, input, inh = arg
            else:
                raise ValueError("Each neuron spec must be (neur_type, n), (neur_type, n, input) "
                                 "or (neur_type, n, input, inh)")

            if not isinstance(number, int) or number <= 0:
                raise ValueError(
                    f"Neuron count of '{neur_type}' must be a positive integer, got {number!r}.")
            
            if neur_type in self.neuronspecs:
                logger.warning("Overwriting neuron spec '%s': old= %s new= %s", 
                               neur_type,
                               asdict(self.neuronspecs[neur_type]),
                               {"neur_type": neur_type, "n": number, "input": bool(input), "inh": bool(inh)})

            self.neuronspecs[neur_type] = NeuronSpec(neur_type=neur_type, n=number, input=bool(input), inh=bool(inh),)
            logger.debug("Set neuron spec: '%s': %s", neur_type, asdict(self.neuronspecs[neur_type]))
           
        return self

    def set_positions(self, mapping: Optional[Dict[str, Tuple[int, ...]]] = None, *, share: Optional[str] = None, **neur_type) -> "NetBuilder":
        """
        Place PREDEFINED neuron groups (set_neurons) on an integer grid, 
        (later, set_connectivity: synapses -> use distance kernel) 
        
       NOTE Positions drawn on realize, Network.positions saves them as: ({group: (n, ndim) array});


        Ex.:   (neurongroup, x_dim, y_dim, z_dim), (share = None|str)
            * network.set_positions(exc = (20, 10, 10), inh= (20, 10, 10), share =liquid)  
            * network.set_positions({"exc": (20, 10, 10), "inh": (20, 10, 10), "share": liquid})                
                - = 20x10x10 grid with exc and inh neurons (Deckers 2022: 2000 cells, 80/20)
        
        Args:
            neurongroup : str  =
            x_dim, y_dim, z_dim: int  = Dimensions of grid for neurongroup to be placed in 
            share : str | None,  
                -  <grid_name> = specific grid (must have same dimensions every call).
                -  None = seperate grids (each group fills its own grid)
        note - Groups not listed keep grid=None.
        """
        grids: Dict[str, Tuple[int, ...]] = {}
        if mapping is not None:
            grids.update(mapping)
        grids.update(neur_type)
        for name, grid in grids.items():
            if name not in self.neuronspecs:
                raise KeyError(f"Unknown neuron group '{name}'. Define it first with set_neurons(...). Current groups: {list(self.neuronspecs)}")
            try:
                grid = tuple(int(g) for g in grid)
            except TypeError:
                raise ValueError(f"Grid of '{name}' must be a tuple of positive ints, got {grid!r}.")
            if not grid or any(g <= 0 for g in grid):
                raise ValueError(f"Grid of '{name}' must be a tuple of positive ints, got {grid!r}.")
            spec = self.neuronspecs[name]
            if spec.grid is not None:
                logger.warning("Overwriting previous grid: '%s' : %r -> %r", name, spec.grid, grid)
            spec.grid, spec.grid_share = grid, (None if share is None else str(share))
            logger.debug("            Set grid: '%s' : %r share=%r", name, grid, share)
        _grid_positions(self.neuronspecs, self.rng)     ## SAFETYVALVE - validate sizes
        return self
    
    ##############################################################################
    ############################## DEFINE SYNAPSES  ##############################

    def set_synapses(self, *type_pre_post: Tuple[str, str, str], clear: bool = False) -> "NetBuilder":
        """
        Define and declare synaptic groups using name/type of synapse, and type/name of pre and post neurons

        Ex.:   (syn_type, pre_neuron, post_neuron)
        network.set_synapses(("input_exc","input","exc"), ("exc_exc","exc","exc"), ("inh_exc","inh","exc"))
        
        clear=True clears existing synapsespecs first
        NOTE: pre_neuron/post_neuron existence IS checked below -- an unknown neuron group raises KeyError.
        """
        ####TO ADD LATER: Typo/existence check in pre_node and post_node: IF nodes DO NOT exist -> Warning, define. If nodes DO exist: info message? ++ Node added
        if clear:
            self.synapsespecs.clear()
            logger.debug(f"Previous synapses cleared.")
        
        for syn_type, pre_neuron, post_neuron in type_pre_post:
            if pre_neuron not in self.neuronspecs or post_neuron not in self.neuronspecs:
                raise KeyError(f"Unknown neuron type in synapse '{syn_type}' w/ pre_neuron='{pre_neuron}', post_neuron='{post_neuron}'. \n\nKnown neurons: {self.neuronspecs.keys()}")
            
            if reserved_name(syn_type):
                raise ValueError(f"Synapse group cannot be named {syn_type!r} as that specific name overlaps with keywords used in freeze-funtion."
                                 f"RESERVED namespace: {FREEZE_KEYWORDS}.")

            if syn_type in self.synapsespecs:
                logger.warning("Overwriting synapse spec '%s': old= %s new= %s", 
                               syn_type,
                               asdict(self.synapsespecs[syn_type]),
                               {"syn_type": syn_type, "pre_neuron": pre_neuron, "post_neuron": post_neuron})
            
            
            self.synapsespecs[syn_type] = SynapseSpec(syn_type=syn_type, pre_neuron=pre_neuron, post_neuron=post_neuron)
            logger.debug("Set synapse spec '%s': %s", syn_type, asdict(self.synapsespecs[syn_type]))

            
        return self


    def set_connectivity(self, mapping: Optional[Dict[str, float | str]] = None, **syn_type) -> "NetBuilder":
        """
        Set probability=[0, 1] of connection associated with each PREDEFINED synapse (set using set_synapse()),
        OR a named pattern from CONNECTIVITY_PATTERNS ("all_to_all" | "one_to_one" | "all_but_self").

        Input style:
            network.set_connectivity({"exc_exc":0.1, "exc_inh"=0.2})
        OR:
            network.set_connectivity(exc_exc=0.1, exc_inh="one_to_one")
        """
        synapses: Dict[str, float | str] = {}
        if mapping is not None:
            synapses.update(mapping)
        synapses.update(syn_type)

        for syn_type, prob in synapses.items():
            ###SAFETY CHECKS
            if syn_type not in self.synapsespecs:
                raise KeyError(f"Unknown synapse '{syn_type}'. Define it first with set_synapses(...). Current synapses: {self.synapsespecs.keys()}")
            if isinstance(prob, str):   ##NOTE validated by shape in realize()
                if prob not in CONNECTIVITY_PATTERNS:
                    raise ValueError(f"Unknown connectivity pattern {prob!r} for '{syn_type}'. Known patterns: {tuple(CONNECTIVITY_PATTERNS)}, or a probability in [0, 1].")
            elif isinstance(prob, dict):   ## spatial: DISTANCE KERNEL
                prob = _check_kernel(syn_type, prob)
            else:
                try:
                    prob = float(prob)
                except (TypeError, ValueError):
                    raise ValueError(f"probability of connection for '{syn_type}' must be a number in: [0, 1] or a pattern name {tuple(CONNECTIVITY_PATTERNS)}. Got {prob!r}.")
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(f"Connectivity for '{syn_type}' must be in [0, 1], got {prob!r}.")
            if self.synapsespecs[syn_type].p_connection is not None:
                logger.warning("Overwriting previous p_connection: '%s' : '%r'", syn_type, self.synapsespecs[syn_type].p_connection) 
            
            self.synapsespecs[syn_type].p_connection = prob
            logger.debug("            Set p_connection: '%s' : '%r'", syn_type, prob)  
        return self



    def set_weights(self, mapping: Optional[Dict[str, float]] = None, **syn_type) -> "NetBuilder":
        """
        Set weights=float of connection associated with each PREDEFINED synapse (set using set_synapse()).

        Input style:
            network.set_weights({"exc_exc":0.1, "exc_inh"=0.2})
        OR:
            network.set_weights(exc_exc=0.1, exc_inh=0.2)
        """
        synapses: Dict[str, float] = {}
        if mapping is not None:
            synapses.update(mapping)
        synapses.update(syn_type) 

        for syn_type, w0 in synapses.items():
            ###SAFETY CHECKS
            if syn_type not in self.synapsespecs:
                raise KeyError(f"Unknown synapse '{syn_type}'. Define it first with set_synapses(...). Current synapses: {self.synapsespecs.keys()}")
        
            if isinstance(w0, (tuple, list)) and len(w0) == 2:
                lo, hi = float(w0[0]), float(w0[1])
                if not hi > lo:
                    raise ValueError(f"Weight range for '{syn_type}' must be (lo, hi) with hi > lo, got {w0!r}.")
                w0 = (lo, hi)
            elif isinstance(w0, dict):   ##(HLSP): {"values": [a, b, ...]} = one value per syn
                vals = w0.get("values")
                if set(w0) != {"values"} or not isinstance(vals, (tuple, list)) or not vals:
                    raise ValueError(f"Weight for '{syn_type}': a dict must be {{'values': [a, b, ...]}} (one value per synapse (uniform draw) ), got {w0!r}.")
                w0 = {"values": [float(v) for v in vals]}
            else:
                try:
                    w0 = float(w0)
                except (TypeError, ValueError):
                    raise ValueError(f"Weight for '{syn_type}' must be numeric or a (lo, hi) range, got {w0!r}.")
            if self.synapsespecs[syn_type].w0 is not None:
                logger.warning("Overwriting previous synaptic weight: Synapse: '%s' : '%r'", syn_type, self.synapsespecs[syn_type].w0) 
            
            self.synapsespecs[syn_type].w0 = w0
            logger.debug("            Set synaptic weight: '%s' : '%r'", syn_type, w0)
        return self
    


    def save_specs(self, filename: str = None, spec_name = None, JSON: bool = False, YAML: bool = True, sort_keys: bool = False, save_to = EXPORT_DIR):
        """
        Save specs to JSON or YAML with optional spec-name or spec_ID.
            Args:
                filename - Auto-generates based on spec_name if present, else spec_ID
                spec_name=None
                JSON: False
                YAML: True
                sort_keys: False"""

        if spec_name is None:
            spec_name = self.spec_name

        spec_name_entry = {"spec_name": spec_name}
        canon_specs = self._canonicalize_netspecs()
        spec_id = _hash(canon_specs, digest_size=8)
        spec_id_entry = {"spec_ID": spec_id}
        export_specs = spec_name_entry | spec_id_entry | canon_specs

        if filename is None:
            now = datetime.now().strftime("%d-%m_%H%M%S")
            filename = f"NetSpec_{now}_{spec_name}" if spec_name is not None else f"NetSpec_{now}_{spec_id}"
        
        file = save_to / filename
        _write_json_yaml( data=export_specs, file=file, JSON=JSON, YAML=YAML, sort_keys=sort_keys, log_label="Netspec")

    ##############################################################################################################
    ##########################################################################################################
    ###################################### DEFINE SYNAPSES  #############################################

    def _validate_specs(self) -> None:
        ###SAFETY CHECKS
        for syn_type, syn_spec in self.synapsespecs.items():
            if syn_spec.pre_neuron not in self.neuronspecs:
                raise KeyError(f"\nSynapse '{syn_type}' refers to unknown pre_neuron '{syn_spec.pre_neuron}'.")
            if syn_spec.post_neuron not in self.neuronspecs:
                raise KeyError(f"\nSynapse '{syn_type}' refers to unknown post_neuron '{syn_spec.post_neuron}'.")

    def _order_synspecs(self):
        "Order deterministically so that if logical structure is identical -> synspecs identical regardless of input order"
        #return dict(sorted(self.synapsespecs.items(), key=lambda spec: (spec[0], spec[1]))) 
        return dict(sorted(self.synapsespecs.items(), key=lambda spec: spec[0])) ## ORDER BY KEY ("syn_type")

    def _order_neurspecs(self):
        """Order neuron groups deterministically:
        1. input neurons
            1.excitatory input-nodes
            2.(if inhibitory) input-nodes        
        2. inhibitory neurons
        3. excitatory neurons
    
        """
        
        def _order_key(item):
            name, spec = item
            if spec.input:
                return (0, 1 if spec.inh else 0, name)   # inputs first, inhibitory LAST within them
            return (1, 0 if spec.inh else 1, name)       # non-inputs, inhibitory FIRST within them

        return dict(sorted(self.neuronspecs.items(), key=_order_key))

    def _assign_neuronslices(self):
        """Generate dictionary keyed to neur_type and w/ realized neuronIDs sorted in slices"""
        neuronslices = {}
        n_total = 0
        n_input = 0

        ordered_neuronspecs = self._order_neurspecs()

        inh_start = None
        inh_stop = 0

        for neur_type, spec in ordered_neuronspecs.items():
            neuronslices[neur_type] = slice(n_total, n_total + spec.n)

            if spec.inh:
                if inh_start is None:
                    inh_start = n_total
                inh_stop = n_total + spec.n

            n_total += spec.n

            if spec.input:
                n_input += spec.n

        ##Safety check: inh must be CCONTIGUOUS
        expected = sum(spec.n for spec in ordered_neuronspecs.values() if spec.inh)
        covered = 0 if inh_start is None else (inh_stop - inh_start)
        if covered != expected:
            raise RuntimeError(f"Inhibitory neurons are not contiguous: covered {covered} of {expected}. FIND ERROR in builder.py")

        return neuronslices, n_total, n_input, ordered_neuronspecs
    
    def _assign_synapseslices(self, neuronslices: Dict[str, slice], n_input: int, post_offset:bool): 
        """
        Generate dictionary keyed to syn_type, with tuple slice-pairs of network W sections. (POST, PRE)
        
        IF post_offset:
            POST slices (W rows) are offset due to n_input row-deletion: 
                compressed_post = post_neuron - n_input
        """
        synapseslices = {}

        ordered_synapsespecs = self._order_synspecs()

        for syn, spec in ordered_synapsespecs.items():

            pre_neur = spec.pre_neuron
            post_neur = spec.post_neuron
            if post_offset:            
                #Digging into slc structure to implement post_neuron_slice - n_input for all post-neurons (all IDs)
                post_slc = neuronslices[post_neur]
                pre_slc = neuronslices[pre_neur]
                compressed_post_slc = slice(post_slc.start - n_input, post_slc.stop - n_input, post_slc.step)
                synapseslices[syn] = WeightProjection(post=compressed_post_slc, pre=pre_slc)
            else:
                synapseslices[syn] = WeightProjection(post=neuronslices[post_neur], pre=neuronslices[pre_neur])

        return synapseslices, ordered_synapsespecs
    
    
    def realize(self, network_name = None, spec_name = None, post_offset: bool = False,
                enforce_dale: bool = True, offset_fallback: bool = True, allow_autapse: bool = False,
                filename:str|None=None, save=False, JSON: bool = True, YAML: bool = False, sort_keys=True) -> Network:
        """
        Realize Network -> generates COO and save copy of network in json and npz.
        
        Args:
                spec_name = overwrite spec_name in netbuilder. Saved to meta
                network_name = Give network name, saved to meta.
                filename = Choose filename. Else automatic based on network_name or network_id if no name
                post_offset = remove n_input-post rows from DENSE weight arrays, reducing computational load 
                enforce_dale = True  - SAFETYVALVE; True -> enforces synaptic-sign based on pre-synaptic neuron identity: pre-syn INHIBITORY neurons -> NEGATIVE weights, etc. False -> ignore
                offset_fallback = True - SAFETYVALVE; post_offset compression assumes NO post-synaptic input neurons. True -> drop offsetting and continue. False -> raise Error
                allow_autapse = False - permit self-connections (neuron i -> itself). Only REACHABLE when pre and post are the SAME neuron group
                save = False 
                sort_keys = True
                """
        logger.debug("Realize network with: filename=%r, save=%r, sort_keys=%r with neurons=%s synapses=%s", filename, save, sort_keys,
            self.neuronspecs.keys(), self.synapsespecs.keys()  )
        
        if spec_name is None:
            spec_name = self.spec_name

        if filename is not None and not isinstance(filename, str):
            raise ValueError(f"Name of network must be string, got {filename!r}")
        
        self._validate_specs()
        logger.debug("Spec validation passed.")
        
        neuronslices, n_total, n_input, ordered_neuronspecs = self._assign_neuronslices()
        if n_input == 0:
            raise ValueError("No input neurons defined. At least one neuron group must have input=True.")
        logger.debug("Assigned neuron slices: n_total=%d n_input=%d neuronslices=%s", n_total, n_input, neuronslices)


        ##SAFETY VALVE: ensure post_offset does not quietly distort post-synaptic input neurons (raise warning & drop compression)
        ## or except out of realize 
        if post_offset:
            receiving_inputs = [syn for syn, spec in self._order_synspecs().items() if self.neuronspecs[spec.post_neuron].input]
            
            if receiving_inputs:
                logger.warning("post_offset=True requested but synapse group(s) %s have a postsynaptic input-group. ",
                               receiving_inputs)
                if offset_fallback:
                    post_offset = False
                else:
                    raise RuntimeError(
                        "Network realization cancelled: post_offset=True is incompatible "
                        "with postsynaptic input neurons.")

        synapseslices, ordered_synapsespecs = self._assign_synapseslices(neuronslices, n_input, post_offset)
        logger.debug("Assigned synapse slices: %s", synapseslices)

        ##keyed streams: one per synapse group, order-independent (rng('network', group))
        syn_rng = {syn: self.rng.rng("network", syn) for syn in ordered_synapsespecs}
        logger.debug("Spawned synapse RNG streams for synapses=%s", ordered_synapsespecs.keys())


        pre_idx = []
        post_idx = []
        syntype = []
        W0 = []

        ### Assign integer key to each synapse type 
        type_key = {syn: i for i, syn in enumerate(ordered_synapsespecs)}
        logger.debug("Assigned synapse type_key=%s", type_key)

        positions = _grid_positions(self.neuronspecs, self.rng)   ## spatial: {} when no group has a grid
        if positions:
            logger.debug("Grid positions drawn for groups=%s", list(positions))

        ### COO BUILD; (Gilbert style network generation) 
        for syn, syn_spec in ordered_synapsespecs.items():
            
            ###SECURITY CHECKS
            if syn_spec.p_connection is None:
                raise ValueError(f"Synapse '{syn}' does not have p_connection. Set it with set_connectivity().")
            if isinstance(syn_spec.p_connection, str):   ##NOTE load_specs, re-validates the name
                if syn_spec.p_connection not in CONNECTIVITY_PATTERNS:
                    raise ValueError(f"Synapse '{syn}' has unknown connectivity pattern {syn_spec.p_connection!r}. Known: {tuple(CONNECTIVITY_PATTERNS)}.")
            elif isinstance(syn_spec.p_connection, dict):   ## spatial
                _check_kernel(syn, syn_spec.p_connection)
                for grp in (syn_spec.pre_neuron, syn_spec.post_neuron):
                    if grp not in positions:
                        raise ValueError(f"Synapse '{syn}' uses a distance kernel but neuron group '{grp}' has no grid. Call set_positions() first.")
            elif not (0.0 <= syn_spec.p_connection <= 1.0):
                raise ValueError(f"Synapse '{syn}' has p_connection: {syn_spec.p_connection}. Must be in [0, 1].")
            if syn_spec.w0 is None:
                raise ValueError(f"Synapse '{syn}' does not have w0. Set it with set_weights().")
            
            syn_key = type_key[syn]
            pre_neuron = syn_spec.pre_neuron
            post_neuron = syn_spec.post_neuron

            pre_ids = neuronslices[pre_neuron].start
            post_ids = neuronslices[post_neuron].start
            pre_n = self.neuronspecs[pre_neuron].n
            post_n = self.neuronspecs[post_neuron].n

            if isinstance(syn_spec.p_connection, str):   #NOTE deterministic pattern (no RNG usage)
                if syn_spec.p_connection in _SQUARE_PATTERNS and pre_n != post_n:
                    raise ValueError(f"Synapse '{syn}': pattern {syn_spec.p_connection!r} pairs local index i -> i, so pre and post groups must be "
                                     f"the same size. Got pre '{pre_neuron}' n={pre_n}, post '{post_neuron}' n={post_n}.")
                mask = CONNECTIVITY_PATTERNS[syn_spec.p_connection](pre_n, post_n)
            elif isinstance(syn_spec.p_connection, dict):   ## spatial p(syn) =from distance** 
                mask = _kernel_mask(syn, syn_spec.p_connection, positions[pre_neuron], positions[post_neuron], syn_rng[syn])
            else:
                mask = syn_rng[syn].random((pre_n, post_n), dtype=np.float32) < syn_spec.p_connection

            if pre_neuron == post_neuron and not allow_autapse:
                np.fill_diagonal(mask, False)

            local_pre, local_post = np.nonzero(mask) ##IDX WITHIN group

            ##IDX assignment in GLOBAL group 
            pre = pre_ids + local_pre  
            post = post_ids + local_post

            if len(pre) == 0:
                logger.warning("Synapse group '%s' realized with 0 edges.", syn)
            else:
                logger.debug("Synapse group '%s' realized with %d edges.", syn, len(pre))


            pre_idx.append(pre)
            post_idx.append(post)
            syntype.extend([syn_key] *len(pre))

            # Resolve a (lo, hi) range to ONE magnitude per edge
            if isinstance(syn_spec.w0, (tuple, list)):   ##list after a JSON round-trip
                lo, hi = syn_spec.w0
                w0_edges = syn_rng[syn].uniform(lo, hi, size=len(pre)).astype(np.float32)
                w0_scalar = None
            elif isinstance(syn_spec.w0, dict):   ## *HLSP,- one listed value per syn
                vals = np.asarray(syn_spec.w0["values"], dtype=np.float32)
                w0_edges = vals[syn_rng[syn].integers(0, len(vals), size=len(pre))]
                w0_scalar = None
                if not self.neuronspecs[pre_neuron].inh:      ##NOTE an exc. (or input) pre-group CAN list NEGATIVE values*
                    W0.extend(w0_edges.tolist())
                    if (w0_edges < 0).any():
                        logger.info("Synapse group '%s': listed values include negatives on a non-inhibitory pre group; kept as listed (no Dale sign).", syn) ##Safetyvalve
                    continue
            else:
                w0_edges = None
                w0_scalar = syn_spec.w0

            # SAFETY VALVE: Expected sign according to presynaptic neuron (inh/exc, Dale's law)
            sign = -1.0 if self.neuronspecs[pre_neuron].inh else 1.0
            if w0_edges is not None:
                W0.extend((sign * np.abs(w0_edges)).tolist()) 
                continue
            w0_dale = sign * abs(w0_scalar)
            w0_used = w0_scalar

            if w0_dale != w0_scalar:
                logger.warning("Synapse group '%s': w0=%s does not comply with Dale's law, given "
                               "presynaptic group '%s' (inh=%s). Realizing with w0=%s.",
                               syn, syn_spec.w0, pre_neuron, self.neuronspecs[pre_neuron].inh, w0_dale)
                
                if enforce_dale:
                    w0_used = w0_dale
                else:
                    logger.warning("Synapse group '%s' realized VIOLATING Dale's law (enforce_dale=False). "
                                   "Runtime per-sign weight clamping will fight this every timestep.", syn)

            W0.extend([w0_used] * len(pre))


        ## CONCATENATE list of np.arrays into ONE list
        if pre_idx:
            pre_idx = np.concatenate( pre_idx ).astype(np.int32, copy=False)
            post_idx = np.concatenate(post_idx).astype(np.int32, copy=False)
        else:
            logger.warning("Network realized with 0 total synapses.")
            pre_idx = np.array([], dtype=np.int32)
            post_idx = np.array([], dtype=np.int32)
         
        syntype = np.array(syntype, dtype= np.int16) #stay in NUMPY
        W0 = np.array(W0, dtype= np.float32) #stay in NUMPY

        pre_dd_synt = len(syntype)
        pre_dd_pre = len(pre_idx)
        pre_dd_post = len(post_idx)
        pre_dd_W0 = len(W0)

        #Safety check
        if not pre_dd_post == pre_dd_pre == pre_dd_synt == pre_dd_W0:
            logger.warning(("Generated mismatched length of COO ndarrays (pre-deduplication): pre_idx: %r, post_idx: %r, W0: %r, syntype: %r .", 
                             pre_dd_pre, pre_dd_post, pre_dd_W0, pre_dd_synt))
            raise ValueError(f"Generated mismatched length of COO ndarrays (pre-deduplication):"
                             f"pre_idx={pre_dd_pre!r}, post_idx={pre_dd_post!r}, W0={pre_dd_W0!r}, syntype={pre_dd_synt!r}.")
        ###### ENSURE NO DUPLICATE edges
        ###ENSURE NO DUPLICATES of edges/synapses by testing for unique (and keeping the FIRST instance, dropping the rest)
        edge_key = pre_idx.astype(np.int64) * np.int64(n_total) + post_idx 
        ## pre->post is NOT same as pre<-post. Assign numerical key to give each edge a unique number (unless duplicate) 

        _, idx = np.unique(edge_key, return_index=True) #Keep only UNIQUE numbers (edge_keys) ###CONVENTION; _ means "ignore"
        idx = np.sort(idx)
        pre_idx = pre_idx[idx]
        post_idx = post_idx[idx]
        W0 = W0[idx]
        syntype = syntype[idx]

        if post_offset:
            post_idx -= n_input

        post_dd_synt = len(syntype)
        post_dd_pre = len(pre_idx)
        post_dd_post = len(post_idx)
        post_dd_W0 = len(W0)

        #Safety check:
        if not post_dd_post == post_dd_pre == post_dd_synt == post_dd_W0:
            logger.warning("Generated mismatched length of COO ndarrays (post-deduplication): pre_idx: %r, post_idx: %r, W0: %r, syntype: %r .", 
                             post_dd_pre, post_dd_post, post_dd_W0, post_dd_synt)
            raise ValueError(f"Generated mismatched length of COO ndarrays (post-deduplication):"
                             f"pre_idx={post_dd_pre!r}, post_idx={post_dd_post!r}, W0={post_dd_W0!r}, syntype={post_dd_synt!r}.")
        
        if pre_dd_synt != post_dd_synt:
            logger.warning("Removed duplicate synapses during COO ndarray generation. Length before: %r, length after: %r", pre_dd_synt, post_dd_synt)     ##warning: a dropped edge across two groups is a possible spec error
        logger.debug("Generated COO successfully with: %r synapses.", post_dd_pre)

        
        meta = _generate_netmeta( neuronspecs=ordered_neuronspecs, synapsespecs=ordered_synapsespecs, spec_name=spec_name, network_name=network_name, 
                            neuronslices=neuronslices, synapseslices=synapseslices, type_key=type_key, 
                            syntype=syntype, pre_idx=pre_idx, post_idx=post_idx, W0=W0,
                            post_offset=post_offset, n_input=n_input, rng=self.rng, positions=positions)
        logger.debug("Generated NetMeta: spec_name=%s, network_name=%s, spec_id=%s, network_id=%s, globalseed=%s", 
                     meta.spec_name, meta.network_name, meta.spec_id, meta.network_id, meta.globalseed)

        network = Network(neuronspecs=ordered_neuronspecs, synapsespecs=ordered_synapsespecs, 
                          neuronslices=neuronslices, synapseslices=synapseslices, 
                          syntype=syntype, pre_idx=pre_idx, post_idx=post_idx, W0=W0, type_key=type_key, meta=meta,
                          positions=positions)
        logger.info("Created Network. Total neurons= %d, Total synapses= %d, Network_ID: %s", n_total, post_dd_post, meta.network_id)

        if self._pnet_ref is not None:
            pnet = self._pnet_ref() # Use reference to collect PhasicNet-object (if it (still) exists)
            if pnet is not None:
                pnet.network = network 
        if save:
            network.save_net(filename=filename, JSON=JSON, YAML=YAML, sort_keys=sort_keys)

        return network
