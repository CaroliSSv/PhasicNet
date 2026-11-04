#microcol_hlsp - DYNAMICS of the layered microcolumn experiment, variant A: the HLSP-LSM physics of Luo, Chu, Wu, Rasheed & Wang (2026),
# Neural Networks 204:109245, applied to every layer. The runtime modules are hlsp2026.py's, UNCHANGED (equations, constants and sources are
# documented there); this file is the pack of them and how they read in a layered network.
#
#WHAT EACH MODULE DOES IN LAYERS (one exc + one inh group per layer, columns = grid coordinates, builder kernel "column"):
#   HLSPCurrentSynapse(Sparse)  Eq. (3) current, Eq. (18) STD on EVERY non-input presynaptic neuron: the feed-forward drive L1 -> L2 -> L3 is
#                               depressed like the liquid's recurrent drive; the input projection is not (as in HLSP)
#   HLSPLIF, HLSPThreshold      Eq. (1)-(2): one v_th (20 mV), tau_mem 64 ms, t_refrac 2 ms for every layer
#   HLSPPlasticity(Sparse)      E-STDP (Eq. 8-10) on every plastic edge with an excitatory pre (within-column exc -> exc/inh AND the feed-forward
#                               L(k) -> L(k+1)); I-STDP (Eq. 12-15) on every plastic edge with an inhibitory pre; input columns never learn
#   Run._clamp                  Eq. (16)-(17): [0, W_max] for exc pre groups, [-W_max, 0] for inh pre groups (Dale)
#
#LAYER-WISE TRAINING is not a dynamics setting: the protocol (experiments/microcol_hlsp_protocol.py) re-freezes per stage (Run.set_frozen), and
# the modules read syn.frozen / the plastic mask every step, so nothing in them changes.
#WEIGHTS: sparse + update_on "hybrid_event" (MICROCOL_HLSP_SPARSE, the cfg file's default) or dense + "timestep" (MICROCOL_HLSP_DENSE, same model).
# The protocol class is added by the main file (protocol_classes), so this pack stays protocol-free.
from dataclasses import replace

from phasicnet import RuntimeClasses
from hlsp2026 import HLSPCurrentSynapse, HLSPCurrentSynapseSparse, HLSPLIF, HLSPThreshold, HLSPPlasticity, HLSPPlasticitySparse
from original_dynamics import TestNoise
from phases import DefaultPhase

MICROCOL_HLSP_DENSE = RuntimeClasses(synapse_cls=HLSPCurrentSynapse, integrator_cls=HLSPLIF, threshold_cls=HLSPThreshold, plasticity_cls=HLSPPlasticity,
                                     noise_cls=TestNoise, phase_classes=(DefaultPhase,))
MICROCOL_HLSP_SPARSE = replace(MICROCOL_HLSP_DENSE, synapse_cls=HLSPCurrentSynapseSparse, plasticity_cls=HLSPPlasticitySparse)
