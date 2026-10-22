#freeze - packs functions needed to resolve cfg.plasticity.freeze into tuple of FROZEN WeightProjections (These will null out any and all learning)

from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Tuple
import logging
from network import WeightProjection
import numpy as np

logger = logging.getLogger(__name__)

   ######################################################################################
##########################   Weight-space annotations   ###################################
   ######################################################################################




   ######################################################################################
##########################   Freeze vocabulary   ##########################################
   ######################################################################################

##Matcher = conventional name in python for a dict that groups to a property that can be collected by a function (so a matcher returns the function)
_ROLE_MATCHERS: dict[str, Callable] = {"input": lambda spec: bool(spec.input),
                                       "inh": lambda spec: bool(spec.inh),
                                       "exc": lambda spec: not bool(spec.inh)}

_NEURON_ROLES = tuple(_ROLE_MATCHERS)
_SYNAPTIC_POSITIONS = ("pre", "post")


FREEZE_KEYWORDS = ("all",) + _NEURON_ROLES + tuple(f"{position}_{role}"
                                                   for position in _SYNAPTIC_POSITIONS for role in _NEURON_ROLES)


def reserved_name(name: str) -> bool:
    """Is `name` part of the internal-namespace (ex. freeze-keywords)? 
    Blocks ALL combinations used by freeze to parse role&position from being used as synapse-names."""
    return name in FREEZE_KEYWORDS


   ######################################################################################
##########################   Freeze parsing   #############################################
   ######################################################################################

def _parse_freeze(net, freeze: Iterable[str] | None) -> tuple[bool, set[str], list[tuple[str | None, str]]]:
    """
    Parse strings in freeze iterable; -> returns (freeze_all, synapse-names, position-&-role-pairs)

    Supported targets:
        "all"                   --- freeze every projection/synapsegroup
        "role"               ---  "input" | "inh" | "exc"  (freeze a synapse with a specific role in either position (pre/post))
        "position_role"  ---  "pre_role" | "post_role"  (freeze a synapse w/ a neuron w/ a specific role in a specific position)
        "synapse_group"     ---   one named projection
    """
    freeze_all = False
    names: set[str] = set()
    roles: list[tuple[str | None, str]] = []
    unknown: list[str] = []

    for string in freeze or ():
        if string == "all":
            freeze_all = True
            continue
        if string in _ROLE_MATCHERS: ##Collect "inh", "exc", "input" directly
            roles.append((None, string)) #collect in the form of a tuple-pair (with position unspecified (handled internally as "any"))
            continue

        position, sep, role = string.partition("_") ## seperate position and role by _ (tuple-unpacking)
        if sep and position in _SYNAPTIC_POSITIONS and role in _ROLE_MATCHERS: ##if _ found and position is "pre"|"post" and role is "inh"|"exc"|"input"
            roles.append((position, role)) ## collect in the form of a tuple (pair)
            continue

        if string in net.synapsespecs: ##collect synapses by name from specs 
            names.add(string)   ##no need for further parsing -> added to local names set.
            continue

        #Safety valve: catches typos etc. -> keyerror
        unknown.append(string)
 
    if unknown:
        raise KeyError(f"Unknown freeze target(s) {sorted(unknown)}. Expected a synapse-group name "
                       f"{tuple(net.synapsespecs)} or keyword {FREEZE_KEYWORDS}.")
    return freeze_all, names, roles


def _frozen(net, syn: str, spec, names: set[str], roles: list[tuple[str | None, str]]) -> bool:
    """bool-checker: Is this ONE synapse/projection group frozen? 
    
        - yes -> True, no -> False
    Based on parsing by ROLE (inh, input, exc), and (if present) synaptic position (pre/post)"""
    if syn in names:
        return True

    pre, post = net.neuronspecs[spec.pre_neuron], net.neuronspecs[spec.post_neuron]
    for position, role in roles:
        matches = _ROLE_MATCHERS[role] ###matcher = conventional term for this kind of func
        if position in (None, "pre") and matches(pre):
            return True
        if position in (None, "post") and matches(post):
            return True
    return False


   ######################################################################################
##########################   Freeze realization   #########################################
   ######################################################################################

def _build_freeze_slices(net, freeze: Iterable[str] | None = None) -> Tuple[WeightProjection, ...]:
    """
    Resolve a freeze spec into the projection rectangles that WILL NOT learn

    **Args:**
        **net** = Network
        **freeze** = tuple of strings w/ names and/or keywords to freeze.  
        
        Supported strings:
            "all"                   --- freeze every projection/synapsegroup
            "role"               ---  "input" | "inh" | "exc"  (freeze a synapse with a specific role in either position (pre/post))
            "position_role"  ---  "pre_role" | "post_role"  (freeze a synapse w/ a neuron w/ a specific role in a specific position)
            "synapse_group"     ---   one named projection
    Returns:
        Tuple of (post_rows, pre_cols) slice pairs, in net.synapsespecs order. (The slices belonging to all synapsegroups that met specifications to be frozen)

        NOTE: freezing is **idempotent**. Freezing twice makes no difference (since freeze does NULLS out dW twice). A neuron group may satisfy MORE THAN ONE role, and attached synapses can get frozen by EITHER role mentioned

        freeze = () -> returns NOTHING frozen.
        freeze = "all" -> returns ENTIRE matrix in ONE rectangle.
    """

    freeze = tuple(freeze or ()) 
    freeze_all, names, roles = _parse_freeze(net, freeze)

    if freeze_all:
        logger.warning("Freeze ('all'): NO projection can learn in this run.")  ##Defaults to a single slice covering the entire thing ("no slice is made" when slice(None))
        return (WeightProjection(post=slice(None), pre=slice(None)),)

    frozen = [syn for syn, spec in net.synapsespecs.items() if _frozen(net, syn, spec, names, roles)]
    frozen_slices = tuple(net.synapseslices[syn] for syn in frozen)

    if frozen:
        logger.info("Freeze %s -> %d of %d projections frozen: %s.",
                    sorted(freeze), len(frozen), len(net.synapsespecs), frozen)
        if len(frozen) == len(net.synapsespecs):
            logger.warning("Freeze %s leaves NO plastic projections -- this run cannot learn anything.",
                           sorted(freeze))
    return frozen_slices

def _build_sparse_plastic_mask(syn, n_total: int) -> np.ndarray:
    """
    Generate the sparse PLASTIC mask: one bool per realized edge, True = may learn (realized AND not frozen).
    """
    w = syn.weights
    frozen = np.zeros(w.W.size, dtype=bool)
    for proj in syn.freeze_slices:
        post_lo, post_hi, post_step = proj.post.indices(w.n_post)
        pre_lo, pre_hi, pre_step = proj.pre.indices(n_total)
        #Safetyvalve
        if post_step != 1 or pre_step != 1:
            raise ValueError("Freeze projections must be contiguous slices.")
        ##__ior__ -> in-place union of elements  (element=True ANYWHERE -> true.)
        frozen |= ((w.post >= post_lo) & (w.post < post_hi) & (w.pre >= pre_lo) & (w.pre < pre_hi))
    logger.info("Sparse freeze: %d of %d edges frozen.", int(frozen.sum()), w.W.size)
    return ~frozen ##inverse of frozen mask


def _build_dense_plastic_mask(syn) -> np.ndarray:
    """
    Builds DEBUG mask: mask of ONLY realized AND unfrozen synapses (NO OTHER W should drift or "learn" during runtime)

    dense mask of plastic (not-frozen) neurons (DEBUG; heavy computationally; forces gather/scatter and N*N mask) 
    """
    syn_mask = syn.syn_mask ##SHOULD ACCESS THE PROPERTY successfully
    plastic_mask = syn_mask.copy()
    for proj in syn.freeze_slices:
        plastic_mask[proj.post, proj.pre] = False
    return plastic_mask