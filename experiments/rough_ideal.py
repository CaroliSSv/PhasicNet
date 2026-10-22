from __future__ import annotations
import argparse, json, logging, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from phasicnet import PhasicNet
from cfg import Cfg, SimulationCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from protocols import Protocol
from example_modules import SynapseExample, IntegratorExample, ThresholdExample, PlasticityExample, ProtocolExample
from noise import NoiseExample
from phases import DefaultPhase, PhaseExample1, PhaseExample2
from paths import RAW_DIR, NETWORKS_DIR, RUNS_DIR
from preprocessing import load_mnist, poisson_prep
from utils import _write_responses_npz
from readouts import fit_evaluate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

GlOBALSEED = 7744

##IN a different file, MADE ONCE:

def build_network():
    pn = PhasicNet(GlOBALSEED)
    pn.builder.set_neurons("input", n_input, True), ("exc", n_exc), ("inh", n_inh, False, True)
    pn.builder.set_synapses(("in_exc", "input", "exc"), ("in_inh", "input", "inh"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"))
    ...#the rest to generate a network.
    pn.builder.realize("NetExample", post_offset=True, save=True)

def build_experimental_protocol():
    ##Here there are MULTIPLE solutions. We will discuss what is best before implementing. I like your current setup with tiny units being put together by the user. But I will need more prebuilt "packages" of units in larger groupings so that those funcs can be called. In this example I am simply pretending protocol is the one that holds such groupings of the primitive runtime elements that we already have. I am also using a scikit learn inspired (or pytorch inspired) allowed pipeline/sequence to represent an experimentalprotocol. (Since run owns the elementals as far as I understand)
    pr = Protocol
    pr.sequence([
        datahandler.split_test_train(split=0.8)
        pr.holdoutfit(fits=10, shuffle=True, epochs=10, batch=4) #This is the simplest test on TRAIN -> train and eval. Batch means 4 PARALLELL RUNS that should be COMPLETELY INDEPENDENT. 
        pr.test(5) ##Perform the test on the actual test_train split data (the one NOT used to learn)
    ])
exampleprotocol = build_experimental_protocol()
exampleprotocol.save("exampleprotocol")

pn = PhasicNet(GlOBALSEED)
pn.set_runtime_classes(SynapseExample, IntegratorExample, ThresholdExample, PlasticityExample, NoiseExample, (PhaseExample1, PhaseExample2), ProtocolExample)
##TODO: build the alternative pre-pack posibility for setting runtime classes using some saved (perhaps dict)
pn.set_runtime_classes(ExamplePacked) ##Here examplePacked should do exactly the same as set_runtime_classes above, but allow it to get packed once into a set of classes I might run together so that they do not have to live in main as a block of code. Same as network, the setup might be built ONCE then reused.
pn.load_net("previously_built_NetExample.yaml")
pn.load_cfg("cfg_file_extended_from_a_default_base.yaml") ##in cfg_dir
pn.load_runcfg("runcfg_file_extended_from_a_default_base.yaml") ##in cfg_dir ##TODO missing function. Run loads both runcfg and run. Nothig loads ONLY a runcfg instance.
pn.load_protocol("exampleprotocol")

##Because of how different protocol is from the current protocol structure, which I LIKE (I like letting users set up their own protocol instance which allows more customizing) maybe protocol could be renamed experiment or experimentalprotocol, leaving protocol for a set of smaller packed functions that can be used to construct runs more easily? I miss the opportunity to basically write "run 10 runs, when scorer (ex. ctx.scorer._metric < runcfg.scorer_cfg_maxvalue) put current image in a queue to be rerun.). That would require a Mode/sys_phase control that looked something like PhaseScheduler except for technical elements to allow for some more experimental control to implement code like you see in the selected articles." 
# Basically what I am asking is if I can make pre-optimized loop structures that can be called directly (maybe avoiding python looping in the process by prewriting the code) so that the run structure is just a little more like brian2/tensorflow/scikitlearn atleast in the api, but KEEPING the current structure with a run engine and primitives that can be put together? That way I will also have pre-written loggers inside the functions (which compose primitives together in a FAST way) so that user does not have to compose their own from scratch using only primitives. Again I still like the extend-protocol-cfg system, and custom protocol, but I suspect I need a second class to have sets of precomposed primitives in groupings that can be called and make the api of the CURRENT articles WAY smaller  (while staying reusable for later. The same kind of structures return across experiments so blocks of primitives precomposed makes for perfect composability while keeping the opportunity of using primitives directly. I do NOT need something overengineered at this point but I think you understand what I would like.)