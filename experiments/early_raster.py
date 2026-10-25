#raster_demo - build a network, run it, ARCHIVE everything, and write a spike raster.
# Run: python raster_demo.py            (writes into ./runs/<run_name>/)
#
# The archive makes the raster reproducible: NetSpec (spec_id), Network (network_id), DynamicsCfg (cfg_id), History (run_id),
# plus manifest.yaml for the harness parameters none of those hashes cover.
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import logging
import time
import numpy as np
import yaml

from phasicnet import PhasicNet, _generate_runmeta
from settings import Experiment
from dynamicscfg import (DynamicsCfg, RunCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg,
                 PlasticityCfg, NoiseCfg, HistoryCfg, PhaseCfg)
from synapses import BaseSynapse
from integrators import LIFexact
from thresholds import AdaptiveThreshold
from plasticities import TraceSTDP
from original_dynamics import TestNoise
from phases import DefaultPhase
from dataloader import DataLoader
from runner import Runner
import analysis

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- harness parameters
# NOT covered by any cfg or spec hash -- recorded into manifest.yaml explicitly.
SEED = 7                    ##Experiment globalseed: network realization + noise + phases
DATA_SEED = 0               ##separate stream for the synthetic input, so data and model decouple
N_INPUT, N_EXC, N_INH = 24, 60, 16
T_SAMPLE, N_SAMPLES, EPOCHS = 150, 4, 1
INPUT_RATE = 0.25           ##P(spike) per input neuron per timestep (Bernoulli, i.i.d.)
POST_OFFSET = False         ##drop structurally-dead input rows from W

SPEC_NAME = "raster_spec"
NETWORK_NAME = "raster_net"
CFG_NAME = "raster_cfg"
HIST_NAME = "raster_hist"
RUN_NAME = "raster_demo"


def make_cfg() -> DynamicsCfg:
    """The runtime physics. Everything here is hashed into cfg_id."""
    return DynamicsCfg(
        simulation=SimulationCfg(dt=1.0),                                  ##ms per timestep
        synapse=SynapseCfg(b=None),                                        ##no synaptic bias
        integrator=IntegratorCfg(tau_mem=15.0, v_rest=-65.0, r=8.0),       ##ms, mV, MOhm
        threshold=ThresholdCfg(v_reset=-70.0, v_th=-54.0,                  ##mV, mV
                               tau_th=40.0, adpt_th=1.5),                  ##ms, mV per spike
        plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0,    ##ms, ms
                                 A_ltp=0.010, A_ltd=0.011,                 ##A_ltd > A_ltp -> net depression bias
                                 eta_exc=5e-3, eta_inh=5e-3,               ##eta_inh sign-flipped inside TraceSTDP
                                 w_min=-8.0, w_max=8.0,                    ##hard clamp, per-sign at runtime
                                 freeze=()),                               ##nothing frozen
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.35),                        ##mV per step, membrane
        phases=(PhaseCfg(phase_type="DEFAULT"),))   


def build(seed: int = SEED, post_offset: bool = POST_OFFSET) -> PhasicNet:
    pnet = PhasicNet(Experiment(seed))
    pnet._instantiate_builder()
    ##(name, n, is_input, is_inh)
    pnet.builder.set_neurons(("input", N_INPUT, True), ("exc", N_EXC), ("inh", N_INH, False, True))
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"),
                              ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    pnet.builder.set_connectivity(in_exc=0.25, exc_exc=0.10, exc_inh=0.25, inh_exc=0.30)
    ##w0 is a MAGNITUDE; realize() forces the sign (Dale's law), so the inh_exc warning at build time is expected.
    pnet.builder.set_weights({"in_exc": 2.0, "exc_exc": 1.0, "exc_inh": 1.5, "inh_exc": 1.5})
    pnet.builder.spec_name = SPEC_NAME
    pnet.builder.realize(network_name=NETWORK_NAME, post_offset=post_offset, save=False)

    pnet.cfg = make_cfg()
    pnet.set_runtime_classes(synapse_cls=BaseSynapse, integrator_cls=LIFexact,
                             threshold_cls=AdaptiveThreshold, plasticity_cls=TraceSTDP,
                             noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    pnet.dataloader_cls = DataLoader

    ##ORDER IS LOAD-BEARING: phases seed sim.time_in_phase, which states reads;
    ## _implement_plastic needs states to exist.
    pnet._instantiate_rules()
    pnet._instantiate_phases()
    pnet._instantiate_states(debug=False)
    pnet._implement_plastic()
    pnet._instantiate_dataloader()

    ##RunMeta must exist BEFORE history, so the saved history carries the ids that identify it.
    runcfg = RunCfg(epochs=EPOCHS, val_split=None, k_fold=None, batch=None)
    pnet.runmeta = _generate_runmeta(
        cfg_name=CFG_NAME, run_name=RUN_NAME, hist_name=HIST_NAME,
        cfg=pnet.cfg, runcfg=runcfg,
        globalseed=pnet.experiment.globalseed,
        network_name=pnet.network.network_name, network_id=pnet.network.network_id,
        rng_spec=pnet.experiment.rng_spec)

    ##include_input=True so the raster shows the DRIVE as well as the response.
    pnet.instantiate_history(HIST_NAME, runmeta=pnet.runmeta,
                             history_cfg=HistoryCfg(record_spikes=True, include_input=True))
    return pnet, runcfg


def make_data(n_input: int):
    """Synthetic Bernoulli drive. NOT a ModelInput -- Runner only needs len() and [i]->(frames, target)."""
    rng = np.random.default_rng(DATA_SEED)
    X = rng.random((T_SAMPLE * N_SAMPLES, n_input)) < INPUT_RATE
    return [(X[i * T_SAMPLE:(i + 1) * T_SAMPLE], i) for i in range(N_SAMPLES)]


def main(out_root: str | Path = "runs"):
    pnet, runcfg = build()
    net = pnet.network
    data = make_data(net.n_input)

    out = Path(out_root) / f"{RUN_NAME}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    summaries = Runner(pnet, reset_between_samples=True).run(data, epochs=EPOCHS)
    wall = time.perf_counter() - t0
    steps = sum(s["steps"] for s in summaries)

    # ---------------------------------------------------------------- archive
    ##Each of these already owns its own hashing; nothing is recomputed here.
    pnet.builder.save_specs(spec_name=SPEC_NAME, JSON=False, YAML=True, save_to=out)
    net.save_net(network_name=NETWORK_NAME, spec_name=SPEC_NAME, JSON=True, YAML=False, save_to=out)
    pnet.save_cfg(cfg_name=CFG_NAME, JSON=False, YAML=True, save_to=out)
    pnet.save_history(JSON=True, YAML=False, export_dir=out)

    meta = pnet.capture.raster_meta(pnet.ctx, pnet.history)
    analysis.plot_raster(pnet.history, meta, path=out / "raster.png",
                         title=f"{RUN_NAME} -- {N_SAMPLES} samples x {T_SAMPLE} steps "
                               f"(N={net.n_total}, seed={SEED})")
    analysis.plot_rate_distribution(pnet.history, meta, path=out / "rates.png")

    rates = analysis.firing_rates(pnet.history, meta)
    manifest = {
        "run_name": RUN_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        ##--- identity: every hash the framework produces, in one place
        "ids": {"spec_id": net.spec_id, "network_id": net.network_id,
                "cfg_id": pnet.runmeta.cfg_id, "run_id": pnet.runmeta.run_id,
                "globalseed": int(pnet.experiment.globalseed)},
        ##--- harness parameters: NOT covered by any of the hashes above
        "harness": {"seed": SEED, "data_seed": DATA_SEED,
                    "n_input": N_INPUT, "n_exc": N_EXC, "n_inh": N_INH,
                    "T_sample": T_SAMPLE, "n_samples": N_SAMPLES, "epochs": EPOCHS,
                    "input_rate": INPUT_RATE, "post_offset": POST_OFFSET,
                    "input_kind": "bernoulli iid, no temporal structure",
                    "reset_between_samples": True},
        ##--- modules actually installed (a cfg does not record WHICH class consumed it)
        "modules": {k: type(v).__name__ for k, v in
                    (("synapse", pnet.synapse), ("integrator", pnet.integrator),
                     ("threshold", pnet.threshold), ("plasticity", pnet.plasticity),
                     ("noise", pnet.noise))},
        "phases": sorted(pnet.phases),
        ##--- realized size
        "network": {"n_total": int(net.n_total), "n_input": int(net.n_input),
                    "n_post_rows": int(pnet.network.n_post), "W_shape": list(pnet.ctx.syn.W.shape),
                    "realized_synapses": int(pnet.ctx.syn.syn_mask.sum()),
                    "frozen_projections": len(pnet.ctx.syn.freeze_slices)},
        ##--- what the run cost and what it produced
        "result": {"steps": int(steps), "spikes": int(pnet.history.spike_i().size),
                   "recorded_steps": int(pnet.history.n_steps),
                   "wall_seconds": round(wall, 3),
                   "us_per_step": round(wall / max(steps, 1) * 1e6, 1),
                   "simulated_ms": float(steps * pnet.cfg.simulation.dt),
                   "realtime_factor": round(steps * pnet.cfg.simulation.dt / 1000.0 / wall, 3),
                   "mean_rate_hz": {name: round(float(rates[b["start"]:b["stop"]].mean()), 2)
                                    for name, b in meta["bands"].items()}},
    }
    with open(out / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, indent=4)

    print(analysis.describe(pnet.history, meta))
    print(f"\nwall {wall:.3f}s  |  {wall/max(steps,1)*1e6:.0f} us/step  |  "
          f"{steps * pnet.cfg.simulation.dt/1000.0/wall:.2f}x realtime")
    print(f"\narchived to {out}/")
    for path in sorted(out.iterdir()):
        print(f"    {path.name}")
    return pnet, out


if __name__ == "__main__":
    main()
