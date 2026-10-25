#main - the Brian2 STDP example (brian2.readthedocs.io/en/stable/examples/synapses.STDP.html) reproduced in PhasicNet as literally
# as the framework allows. This is the validation run named in the project plan: same equations, same parameters, same stimulus,
# same 100 s -- so any difference in the RESULT is about the framework, not about the setup.
#
#THE EXAMPLE: ONE LIF neuron driven by N=1000 independent 15 Hz Poisson inputs through CONDUCTANCE synapses, under pair-based ADDITIVE STDP
# with hard bounds [0, gmax] and a slight bias toward depression. No data, no classes: the result is the weight distribution (BIMODAL: weights
# pile up at 0 or at gmax) and the output firing rate.
#
#LINE BY LINE                  Brian2                                       PhasicNet
#   Poisson inputs             PoissonGroup(N, rates=F)                     PoissonStream: p = F*dt per input per step, keyed rng per chunk
#   neuron                     dv/dt = (ge*(Ee-vr) + El - v)/taum           LIFeuler(tau_mem=taum, v_rest=El, r=Ee-vr), I = ge   (method='euler')
#   spike / reset              threshold v>vt, reset v=vr                   StaticThreshold(v_th=vt, v_reset=vr)
#   synaptic conductance       dge/dt = -ge/taue ; on_pre: ge += w          ConductanceSynapse (below): ge -> ge*(1-dt/taue) + sum(w of spiking inputs)
#   connectivity / weights     S.connect() ; S.w = rand()*gmax              in_exc p=1.0, weights (0, gmax) uniform
#   STDP                       dApre/dpost event-driven traces, clip(0,gmax) Brian2OrderSTDP (a TraceSTDP subclass): A_ltp=dApre, A_ltd=-dApost, w in [0, gmax]
#   StateMonitor(S,'w',[0,1])  records w of synapses 0 and 1                WeightMonitor (a Phase hook -- the framework's own extension point)
#   run(100*second)            dt = 0.1 ms (Brian2 default)                 SimulationCfg(dt=0.1), 1,000,000 steps, ONE training pass
#
#KNOWN DEVIATIONS:
#   1. Initial v: Brian2 starts at v=0 mV (above vt: one spike at step 1); PhasicNet at v_rest=El. A transient of a few tens of ms.
#   2. Weights are clamped ONCE per step; Brian2 clips after on_pre and again after on_post (differs only when a pre and a post spike coincide at a bound).
#   3. The Poisson stream is fed as 100 chunks of 1 s; state carries across chunks (no reset), so the seams are invisible.
#   4. Weights are recorded every 100 ms for the plot, not every dt.
#
#STDP ORDER: Brian2 decays the traces to "now", then on_pre, then on_post; the framework's TraceSTDP reads both traces as the previous step left
# them. Tiny per event, but it shifts the net drift (~13% lower output rate over 100 s). Brian2OrderSTDP (below) uses Brian2's order; swap
# STDP_CLASS for TraceSTDP to see the framework's own convention.
#
#VALIDATION (against plain-NumPy transcriptions of the equations on the SAME input; Brian2 itself was not run): 1 s: all output spikes identical,
# 999 of 1000 weights equal (the other is deviation 2). 100 s: 33.3 Hz, mean w/gmax 0.436 vs the reference 31.9 +- 1.0 Hz, 0.436 +- 0.001 (4 seeds).
#
#BEYOND THE EXAMPLE (not Brian2 results; the printed output keeps them apart):
#   RASTER      the 1000 inputs are recorded too (include_input=True); two one-second windows, early and late, with the output rate underneath.
#   SEPARABILITY The example has no classes, so a separability score needs a labelled test that I made up: with learning OFF, class 0 boosts the
#               200 LOWEST-weight inputs, class 1 the 200 HIGHEST; the spike count is scored (Calinski-Harabasz) with the trained and with the
#               UNTRAINED (as-built) weights. The comparison with the untrained control is the number that means something.
from __future__ import annotations
import logging, sys, time
from dataclasses import dataclass
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]     ##experiments/<script> -> PhasicNet/
sys.path[:0] = [str(_ROOT / "system"), str(_ROOT / "system_modules"), str(_ROOT / "experiment_dynamics"), str(_ROOT / "experiments")]
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import analysis, capture
from phasicnet import PhasicNet
from dynamicscfg import DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, HistoryCfg
from dynamics import Synapse
from states import RunContext
from original_dynamics import LIFeuler, StaticThreshold, TraceSTDP
from dynamics import Noise
from phases import DefaultPhase
from paths import NETWORKS_DIR, RUNS_DIR

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("main")

#---------------------------------------------------------------- the example's parameters, verbatim (ms, mV, Hz)
N = 1000
TAUM = 10.0                           #ms   membrane time constant
TAUPRE = 20.0                         #ms
TAUPOST = TAUPRE
EE = 0.0                              #mV   excitatory reversal potential
VT = -54.0                            #mV   threshold
VR = -60.0                            #mV   reset
EL = -74.0                            #mV   leak / rest
TAUE = 5.0                            #ms   conductance decay
F = 15.0                              #Hz   input rate, every input
GMAX = 0.01
DAPRE = 0.01
DAPOST = -DAPRE * TAUPRE / TAUPOST * 1.05
DAPOST *= GMAX
DAPRE *= GMAX
DT_MS = 0.1                           #ms   Brian2's default clock
DURATION_S = 100                      #s    run(100*second)
CHUNK_S = 1.0                         #s    one PhasicNet "sample" = one chunk of the stream
SEED = 7
RECORD_EVERY_MS = 100.0               #ms   weight-monitor resolution (plot only)
RASTER_S = 1.0                        #s    length of each raster window (early: from t=0, late: the last second)
PROBE_SET = 200                       #inputs boosted per probe class
PROBE_HZ = 30.0                       #Hz   rate of the boosted inputs (the rest stay at F)
PROBE_S = 0.5                         #s    one probe sample
PROBE_PER_CLASS = 30                  #samples per class


   ######################################################################################
##########################   Stimulus: PoissonGroup(N, rates=F)   #########################
   ######################################################################################

class PoissonStream:
    """
    N independent Poisson inputs at rate_hz, as a LAZY sequence of (frames, target) chunks (100 s held at once would be 1 GB).
    A KEYED stream per chunk: chunk i is the same every time. Bernoulli per step with p = rate*dt, as Brian2's PoissonGroup.
    """
    def __init__(self, rng_handler, n_input: int, rate_hz: float, dt_ms: float, chunk_s: float, n_chunks: int):
        self.rng_handler, self.n_input, self.n_chunks = rng_handler, n_input, n_chunks
        self.p = rate_hz * dt_ms / 1000.0
        self.steps = int(round(chunk_s * 1000.0 / dt_ms))

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, i: int):
        gen = self.rng_handler.rng("input", int(i))
        return gen.random((self.steps, self.n_input), dtype=np.float32) < self.p, 0     ##one class: the stream has no labels


class ProbeStream:
    """
    Labelled probe stimuli, lazily generated like PoissonStream: every input fires at base_hz EXCEPT one class's boost set, which fires at boost_hz.
    Sample i has label i % 2 (0 -> boost_sets[0], 1 -> boost_sets[1]); a keyed stream per sample, so the SAME stimuli can be replayed through
    different weights, which is what the trained-vs-untrained comparison needs.
    """
    def __init__(self, rng_handler, n_input: int, dt_ms: float, sample_s: float, n_samples: int, boost_sets, base_hz: float, boost_hz: float):
        self.rng_handler, self.n_input, self.n_samples, self.boost_sets = rng_handler, n_input, n_samples, boost_sets
        self.steps = int(round(sample_s * 1000.0 / dt_ms))
        self.p = [np.full(n_input, base_hz * dt_ms / 1000.0, dtype=np.float32) for _ in boost_sets]
        for p, boosted in zip(self.p, boost_sets):
            p[boosted] = boost_hz * dt_ms / 1000.0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int):
        label = int(i) % 2
        gen = self.rng_handler.rng("probe", int(i))
        return gen.random((self.steps, self.n_input), dtype=np.float32) < self.p[label], label


   ######################################################################################
##########################   Conductance synapse (not in the framework; user-defined)   ###
   ######################################################################################

@dataclass(frozen=True, slots=True)
class ConductanceSynapseCfg(SynapseCfg):
    """SynapseCfg + the conductance decay. A cfg SUBCLASS rather than a new field on SynapseCfg: only this module needs tau_e."""
    tau_e: float = 5.0                #ms


class ConductanceSynapse(Synapse):
    """
    Brian2:   dge/dt = -ge/taue   ;   on_pre: ge += w        (Euler)

    ge is PER-NEURON state (Capitalised -> state_dict, so it is checkpointed and reset with the fit). The integrator reads ge as the current;
    spikes arriving THIS step enter ge for the NEXT step, which is the order Brian2 runs them in (state update, then on_pre).
    """
    weights_format = "dense"
    writes = ("I2V",)
    requires = ()

    def __init__(self, cfg: ConductanceSynapseCfg, sim_cfg: SimulationCfg = None):
        if cfg is None or sim_cfg is None or sim_cfg.dt is None:
            raise ValueError("ConductanceSynapse requires ConductanceSynapseCfg and SimulationCfg with dt.")
        if cfg.tau_e <= 0:
            raise ValueError("ConductanceSynapse requires tau_e > 0.")
        self.b = None if cfg.b is None else np.float32(cfg.b)
        self.tau_e = np.float32(cfg.tau_e)                                 ##stored under its cfg name (naming contract -> derived cfg)
        self.decay_e = np.float32(1.0 - sim_cfg.dt / cfg.tau_e)            ##Euler, as Brian2's method='euler'
        self.Ge = None
        self._arrivals = None

    def configure_for_network(self, net) -> None:
        self.Ge = np.zeros(net.n_post, dtype=np.float32)
        self._arrivals = np.zeros(net.n_post, dtype=np.float32)

    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        np.copyto(upd.I2V, self.Ge)                                        ##a COPY: the integrator multiplies I2V in place and must not touch Ge
        if self.b is not None:
            np.add(upd.I2V, self.b, out=upd.I2V)
        np.dot(syn.W, neur.Spikes, out=self._arrivals)                     ##sum of w over the inputs that spiked this step
        np.multiply(self.Ge, self.decay_e, out=self.Ge)
        np.add(self.Ge, self._arrivals, out=self.Ge)


   ######################################################################################
##########################   STDP in Brian2's within-step order   #########################
   ######################################################################################

class Brian2OrderSTDP(TraceSTDP):
    """
    TraceSTDP with Brian2's WITHIN-STEP ORDER (see the header). Per step, Brian2: traces decay to now; on_pre (Apre += dApre; w += Apost); then
    on_post (Apost += dApost; w += Apre). So LTD at a pre spike reads Apost decayed to now but WITHOUT this step's post spike, and LTP at a post
    spike reads Apre decayed to now AND INCLUDING this step's pre spike. Everything else (cfg, traces, eta, freeze, masks) is inherited.
    """
    def step(self, ctx: RunContext) -> None:
        upd, neur, syn = ctx.upd, ctx.neur, ctx.syn
        np.multiply(self.Pre_trace, self.decay_pretrace, out=self.Pre_trace)                    ##both traces decay to NOW
        np.multiply(self.Post_trace, self.decay_posttrace, out=self.Post_trace)
        np.multiply(self.Post_trace[:, None], neur.pre_spike[None, :], out=self._ltd)           ##LTD: Apost BEFORE this step's post spike
        np.multiply(self._ltd, self.A_ltd, out=self._ltd)
        np.add(self.Pre_trace, neur.pre_spike, out=self.Pre_trace)                              ##on_pre:  Apre += dApre
        np.multiply(neur.post_spike[:, None], self.Pre_trace[None, :], out=upd.dW)              ##LTP: Apre INCLUDING this step's pre spike
        np.multiply(upd.dW, self.A_ltp, out=upd.dW)
        np.subtract(upd.dW, self._ltd, out=upd.dW)
        np.add(self.Post_trace, neur.post_spike, out=self.Post_trace)                           ##on_post: Apost += dApost
        np.multiply(upd.dW, self.eta_pre[None, :], out=upd.dW)
        for proj in syn.freeze_slices:                                                          ##same tail as TraceSTDP: where the update may apply
            upd.dW[proj.post, proj.pre] = 0.0
        np.multiply(upd.dW, syn.syn_mask, out=upd.dW)


STDP_CLASS = Brian2OrderSTDP          ##TraceSTDP = the framework's own ordering (about 13% lower output rate here; see the header)


   ######################################################################################
##########################   StateMonitor(S, 'w', record=[0, 1])   ########################
   ######################################################################################

class WeightMonitor(DefaultPhase):
    """The DEFAULT phase with one hook overridden. post_plasticity runs every learning step, after dW is computed and before it is committed."""
    phase_type = "DEFAULT"

    def __init__(self, cfg, sim_cfg=None, rng=None):
        super().__init__(cfg, sim_cfg, rng=rng)
        self.every = max(int(round(RECORD_EVERY_MS / DT_MS)), 1)
        self.t_ms, self.w = [], []

    def post_plasticity(self, ctx) -> None:
        step = ctx.sim.current_runstep
        if step % self.every == 0:
            self.t_ms.append(step * DT_MS)
            self.w.append(ctx.syn.W[0, :2].copy())                         ##synapses 0 and 1, as record=[0, 1]


   ######################################################################################
##########################   Assemble   ###################################################
   ######################################################################################

def build() -> PhasicNet:
    pnet = PhasicNet(SEED)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", N, True), ("exc", 1))
    pnet.builder.set_synapses(("in_exc", "input", "exc"))
    pnet.builder.set_connectivity(in_exc=1.0)                              ##S.connect(): every input to the one neuron
    pnet.builder.set_weights({"in_exc": (0.0, GMAX)})                      ##S.w = 'rand() * gmax'
    pnet.builder.spec_name = "brian2_stdp_spec"
    net = pnet.builder.realize(network_name="brian2_stdp_net", post_offset=True, save=False)   ##post_offset: W is (1, N+1), not (N+1, N+1)
    net.save_net(network_name="brian2_stdp_net", spec_name="brian2_stdp_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)

    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=DT_MS, weights_format="dense", update_on="timestep"),
        synapse=ConductanceSynapseCfg(b=None, tau_e=TAUE),
        integrator=IntegratorCfg(tau_mem=TAUM, v_rest=EL, r=EE - VR),      ##ge*(Ee-vr): a constant 60 mV driving force per unit ge
        threshold=ThresholdCfg(v_reset=VR, v_th=VT),
        plasticity=PlasticityCfg(tau_pretrace=TAUPRE, tau_posttrace=TAUPOST, A_ltp=DAPRE, A_ltd=-DAPOST,
                                 eta_exc=1.0, eta_inh=1.0, w_min=0.0, w_max=GMAX),   ##eta_inh unused: no inhibitory neurons
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),   ##the example has NO noise: the framework requires the entry, the base Noise class ignores it
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=ConductanceSynapse, integrator_cls=LIFeuler, threshold_cls=StaticThreshold,
                             plasticity_cls=STDP_CLASS, noise_cls=Noise, phase_classes=(WeightMonitor,))
    pnet.cfg_name, pnet.run_name = "brian2_stdp_cfg", "brian2_stdp_run"
    return pnet


   ######################################################################################
##########################   Results   ####################################################
   ######################################################################################

def plot(run, path: Path) -> Path | None:
    """The example's three panels: final weights by synapse index, their histogram (20 bins), and the weight traces of synapses 0 and 1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping the figure.")
        return None
    w = run.ctx.syn.W[0, :N] / GMAX
    mon = run.phases["DEFAULT"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    ax[0].plot(w, ".k"); ax[0].set(ylabel="Weight / gmax", xlabel="Synapse index")
    ax[1].hist(w, 20); ax[1].set(xlabel="Weight / gmax")
    ax[2].plot(np.asarray(mon.t_ms) / 1000.0, np.asarray(mon.w) / GMAX); ax[2].set(xlabel="Time (s)", ylabel="Weight / gmax")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def rasters(run, duration_s: float):
    """Two one-second raster windows (early / late) of the TRAINING run, each with the output neuron's own rate underneath. Call BEFORE the probe,
    which appends its own steps to the same history."""
    meta = capture.raster_meta(run.ctx, run.history)
    n_in, n_total = run.ctx.net.n_input, run.ctx.net.n_total
    paths = []
    for tag, start_s in (("early", 0.0), ("late", duration_s - RASTER_S)):
        paths.append(analysis.plot_raster(run.history, meta, path=RUNS_DIR / f"Raster_{tag}_{run.runmeta.run_id}.png",
                                          window_ms=(start_s * 1000.0, (start_s + RASTER_S) * 1000.0), bin_ms=20.0, rate_band=(n_in, n_total),
                                          title=f"Brian2 STDP, {tag}: t = {start_s:g}-{start_s + RASTER_S:g} s"))
    return meta, paths


def separability_probe(run, rng_handler, n_per_class: int):
    """The probe described in the header. Returns {"trained"|"untrained": (responses, labels, Calinski-Harabasz), "weight_corr": r}. Leaves the run RESET to its as-built weights."""
    n_in = run.ctx.net.n_input
    order = np.argsort(run.ctx.syn.W[0, :n_in])
    boost_sets = (order[:PROBE_SET], order[-PROBE_SET:])               ##class 0: the LOWEST learned weights; class 1: the HIGHEST
    stream = ProbeStream(rng_handler, n_in, DT_MS, PROBE_S, 2 * n_per_class, boost_sets, F, PROBE_HZ)
    out = {}
    w_trained = run.ctx.syn.W[0, :n_in].copy()
    for tag in ("trained", "untrained"):
        if tag == "untrained":
            run.reset_model()                                          ##back to the as-built random weights; same stimuli
            ##the classes come from the TRAINED weights: the correlation says how much of that pattern the untrained weights already had
            out["weight_corr"] = float(np.corrcoef(run.ctx.syn.W[0, :n_in], w_trained)[0, 1])
        summary = run._epoch(stream, shuffle=False, collect_responses=True, learn=False)       ##the call run.test() makes; test() returns only the score
        ##the first n_in columns are the recorded STIMULUS (include_input=True), not the response: keep only the output neuron
        responses = np.asarray(summary["responses"])[:, n_in:]
        labels = np.asarray(summary["targets"])
        out[tag] = (responses, labels, run.scorer.separability(responses, labels))
    return out


def plot_probe(out, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    names = {0: f"boost the {PROBE_SET} LOWEST-weight inputs", 1: f"boost the {PROBE_SET} HIGHEST-weight inputs"}
    for row, tag in zip(axes, ("trained", "untrained")):
        responses, labels, ch = out[tag]
        analysis.plot_separability(responses, labels, score=ch, title=f"{tag} weights", class_names=names, axes=row)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main(duration_s: float = DURATION_S, probe_per_class: int = PROBE_PER_CLASS):
    n_chunks = int(round(duration_s / CHUNK_S))
    pnet = build()
    data = PoissonStream(pnet.rng, N, F, DT_MS, CHUNK_S, n_chunks)
    runcfg = RunCfg(epochs=1, data=DataCfg(test_fraction=0.0), history=HistoryCfg(include_input=True))   ##no test set, no folds, no validation: ONE training pass. Inputs recorded for the raster
    t0 = time.perf_counter()
    run = pnet.train(data, runcfg, cfg_name=pnet.cfg_name, run_name=pnet.run_name, dataset_name="poisson_15Hz", save_run=True, save_history=True)   ##train() takes the names from ITS arguments; pnet.run_name alone is ignored
    wall = time.perf_counter() - t0

    w = run.ctx.syn.W[0, :N] / GMAX
    ##the history now holds the INPUT spikes too (include_input=True), so count only ids at or above n_input: the output neuron's own spikes
    n_spikes = int(np.count_nonzero(run.history.spike_i() >= run.ctx.net.n_input))
    steps = int(run.history.n_steps)
    print("\n" + "=" * 72)
    print(f"BRIAN2 STDP EXAMPLE   {duration_s:g} s at dt={DT_MS} ms = {steps:,} steps   ({wall:.1f} s wall, {wall / max(steps, 1) * 1e6:.1f} us/step)")
    print(f"  output neuron: {n_spikes} spikes -> {n_spikes / duration_s:.2f} Hz mean rate")
    print(f"  weights / gmax: mean {w.mean():.3f}   at the lower bound (<0.05): {np.mean(w < 0.05):.1%}   at the upper bound (>0.95): {np.mean(w > 0.95):.1%}")
    print(f"  run_id={run.runmeta.run_id}  cfg_id={run.runmeta.cfg_id}  derived_cfg_id={run.runmeta.derived_cfg_id}")
    fig = plot(run, RUNS_DIR / f"Brian2STDP_{run.runmeta.run_id}.png")
    print(f"  figure: {fig.name if fig else 'not written'}")

    meta, raster_paths = rasters(run, duration_s)
    print(f"  rasters: {', '.join(r.name for r in raster_paths)}")
    print("\n" + analysis.describe(run.history, meta, T=steps))

    out = separability_probe(run, pnet.rng, probe_per_class)
    print("\n" + "-" * 72)
    print(f"SEPARABILITY PROBE (not part of Brian2's example): {2 * probe_per_class} samples of {PROBE_S:g} s, learning OFF")
    for tag in ("trained", "untrained"):
        responses, labels, ch = out[tag]
        by_class = {c: responses[labels == c, 0] for c in (0, 1)}
        print(f"  {tag:9s} weights: Calinski-Harabasz = {ch:8.2f}   output spikes per sample: class 0 (low-weight inputs boosted) "
              f"{by_class[0].mean():5.1f} +- {by_class[0].std():4.1f}   class 1 (high-weight) {by_class[1].mean():5.1f} +- {by_class[1].std():4.1f}")
    print(f"  correlation between the initial and the final weights: r = {out['weight_corr']:.3f}  (the untrained control separates the classes only through this)")
    print(f"  figure: {plot_probe(out, RUNS_DIR / f'Separability_{run.runmeta.run_id}.png').name}")
    return run


if __name__ == "__main__":
    main()
