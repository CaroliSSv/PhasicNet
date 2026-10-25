#main - Deckers et al. (2022), "Extended liquid state machines for speech recognition", assembled in PhasicNet: the network (3-D grid +
# gaussian kernel), the physics (deckers2022.py), the SHD input, the protocol (fixed liquid -> spike counts -> logistic readout) and ONE call that runs it.
#
#THE ARCHITECTURE (Sec. 2.1-2.2, Table 2):
#   input (700 SHD channels) --random--> exc, inh    ~7 input channels per liquid neuron, w ~ U[0, 0.4] onto exc, U[0, INH_INPUT_FACTOR * 0.4] onto inh
#   liquid: 2000 neurons, 80% exc / 20% inh, on ONE 20x10x10 grid (unit spacing), cells dealt at random (builder.set_positions share=)
#   exc/inh --gaussian kernel--> exc/inh              P = C exp(-(D/lambda)^2), lambda = 2 -> ~9.5 synapses per neuron; w ~ U[0, w_max]
#   delays (ms): input 1.0, EE 1.5, EI/IE/II 0.8      Table 2 -> deckers2022.ELSMSynapseCfg.delays
#
#STIMULUS: SHD via tonic, events binned at BIN_MS (4 ms real time) into (T, 700) binary frames, cut/padded to T_MAX_MS. ONE 4 ms BIN IS ONE 1 ms STEP
# of the liquid (250 steps per sample, p. 6 / p. 9). Preprocessing stays outside the framework (shd_frames).
#
#PROTOCOL (Sec. 2.4): nothing in the liquid learns. Every sample: fresh liquid + v(0) ~ U[13.5, 15], run the sample, read the EXC spike counts.
# Then logistic regression (readouts.py) on train counts, accuracy on the official SHD test set. Paper (Table 4, 2000 neurons, 5 runs each):
# B 80.9%, B+E/I 82.1%, B+E/I+SFA 85.3%, ELSM 86.1%; spikes per neuron 8.56 / 7.58 / 6.03 / 6.83 (the calibration target for E_EXC).
#
#WHAT IS INFERRED (not in the paper): E_EXC / E_INH (Table 1 contradiction), INH_INPUT_FACTOR = 3 (read off Fig. 3), dt = 1 ms per bin, the
# liquid reset between samples, the refractory semantics (Brian 2 >= 2.2), Euler, the readout solver, T_MAX_MS = 1000 ms.
#
#CONFIGURATIONS (Table 4): main(config=) = "b" (Maass baseline), "b_ei" (+ E/I balance), "b_ei_sfa" (+ SFA), "elsm" (+ heterogeneous tau_mem, default).
#
# from PowerShell:  python -c "import experiments, deckers_2022; deckers_2022.main(n_train=1000, n_test=500, tag='t1')"
from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import numpy as np
from phasicnet import PhasicNet, RuntimeClasses
from dynamicscfg import DynamicsCfg, SimulationCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg, ProtocolCfg
from protocols import Protocol
from deckers2022 import (ELSMSynapseCfg, ELSMIntegratorCfg, ELSMThresholdCfg, NoPlasticityCfg,
                         ELSMConductanceSynapse, ELSMConductanceSynapseSparse, ELSMConductanceLIF, SFAThreshold, NoPlasticity)
from original_dynamics import TestNoise
from phases import DefaultPhase
from paths import RAW_DIR, NETWORKS_DIR, RUNS_DIR
from modelinput import ModelInput
from readouts import fit_evaluate, readout_report

logger = logging.getLogger("deckers_2022")

#---------------------------------------------------------------- Table 1 (neuron), Table 2 (network), Sec. 2.3 (ELSM additions)
DT_MS = 1.0                     #ms   liquid time per input frame. NOT in the paper (INFERRED; Brian 2's default clock is 0.1 ms)
BIN_MS = 4.0                    #ms   REAL time per input frame: "binned in 4 ms bins to create 250 ms long samples" (p. 6)
T_MAX_MS = 1000.0               #ms   REAL time: every SHD sample cut / zero-padded to this length -> 250 frames = 250 ms of liquid time
SEED = 3
N_INPUT = 700                   #     SHD channels
N_LIQUID = 2000
GRID = (20, 10, 10)             #     2000 cells, unit spacing
EXC_FRACTION = 0.8
V_REST = 13.5                   #mV   E_rest = v_reset
V_TH = 15.0                     #mV
V_INIT = (13.5, 15.0)           #mV   v(0) ~ U
TAU_MEM = 30.0                  #ms   LSM baseline
TAU_RANGE = (5.0, 55.0)         #ms   ELSM heterogeneous membrane time constants
TAU_GE, TAU_GI = 3.0, 10.0      #ms
T_REFRAC, T_REFRAC_INH = 3.0, 2.0   #ms
SFA_INC = 1.0                   #mV   threshold increment per spike (exc only)
TAU_SFA = 550.0                 #ms   fixed tau_sfa
SFA_RANGE = (50.0, 1050.0)      #ms   heterogeneous tau_sfa
E_EXC = 40.0                    #mV   INFERRED: Table 1 prints 0 mV for E_exc AND E_inh, impossible with E_rest 13.5 < v_th 15. Calibrate against Fig. 4B/C.
E_INH = 0.0                     #mV   INFERRED: the printed 0 mV is plausible for E_inh (13.5 mV below rest), so it is kept
INH_INPUT_FACTOR = 3.0          #     Fig. 3: inh input weights ~ U[0, factor * 0.4], factor relative to the EXC range (p. 8); accuracy peaks around 3-4
INH_INPUT_FACTOR_BASELINE = 0.5 #     the Maass baseline: "the relative input-inhibitory/excitatory weight distribution ratio is equal to 1/2" (p. 7) = U[0, 0.2]
INPUTS_PER_NEURON = 7.0         #     "approximately 7 input neurons per liquid neuron"
W_IN = 0.4                      #     U[0, W_IN] input weights onto exc; onto inh U[0, factor * W_IN]
W_EE, W_EI, W_IE, W_II = 0.6, 1.2, 0.4, 0.4    #     U[0, w] recurrent weights. p. 3 prints IE twice (a typo); EI is the strong one as in Maass et al. 2002
C_EE, C_EI, C_IE, C_II = 0.3, 0.2, 0.4, 0.1    #     kernel amplitudes
LAMBDA = 2.0
DELAYS_MS = {"in_exc": 1.0, "in_inh": 1.0, "exc_exc": 1.5, "exc_inh": 0.8, "inh_exc": 0.8, "inh_inh": 0.8}


   ######################################################################################
##########################   Stimulus   ###################################################
   ######################################################################################

def shd_frames(train: bool, *, bin_ms: float = BIN_MS, t_max_ms: float = T_MAX_MS, save_to=RAW_DIR,
               frame_transforms: tuple = ()) -> ModelInput:
    """SHD via tonic -> ModelInput, sample i = (frames (T, 1, 700) bool, label int), T = t_max_ms / bin_ms: events binned at bin_ms of REAL time (1+ events
    -> one spike), cut or zero-padded; the liquid then runs one frame per step of SimulationCfg.dt. Lazy: tonic reads and frames a sample when it is asked for.
    tonic puts the h5 files in save_to/hsd/ (downloads them there if missing). frame_transforms = extra frame-level tonic transforms after the cut/pad
    (Krenzer: every 10th channel)."""
    import tonic
    from tonic_helpers import ClampBinary, FixedLength, US_TO_MS
    T = int(round(t_max_ms / bin_ms))
    frame = tonic.transforms.Compose([tonic.transforms.ToFrame(sensor_size=tonic.datasets.SHD.sensor_size, time_window=int(bin_ms * US_TO_MS)),
                                      ClampBinary(dtype=np.bool_), FixedLength(T), *frame_transforms])      ##events -> frames at bin_ms -> 1+ events = one spike -> cut / pad to T
    ds = tonic.datasets.SHD(save_to=str(save_to), train=train, transform=frame)
    data = ModelInput.from_tonic(ds, axis_labels="TPX")
    logger.info("shd_frames: %s %d samples, T=%d frames of %.1f ms real time, n_input=%d", "train" if train else "test", len(data), T, bin_ms, data.n_input)
    return data


def shd_dataset(*, bin_ms: float = BIN_MS, t_max_ms: float = T_MAX_MS) -> tuple[ModelInput, ModelInput]:   ##the WHOLE set; DataCfg.n_train / n_test pick from it
    """(train, test) = the official SHD split as two ModelInputs -- what main() runs on and what a parallel.Job(load=shd_dataset,
    load_kw={...}) child loads."""
    return shd_frames(True, bin_ms=bin_ms, t_max_ms=t_max_ms), shd_frames(False, bin_ms=bin_ms, t_max_ms=t_max_ms)


   ######################################################################################
##########################   Assemble   ###################################################
   ######################################################################################

CONFIGS = {   ##Table 4's four columns as (ei_balance, sfa, tau_heterogeneous)
    "b": (False, False, False), "b_ei": (True, False, False), "b_ei_sfa": (True, True, False), "elsm": (True, True, True)}


def build(*, n_liquid: int = N_LIQUID, grid=GRID, n_input: int = N_INPUT, seed: int = SEED, dt_ms: float = DT_MS, weights_format: str = "sparse",
          ei_balance: bool = True, sfa: bool = True, tau_heterogeneous: bool = True, sfa_heterogeneous: bool = True,
          inh_input_factor: float = INH_INPUT_FACTOR, e_exc: float = E_EXC, e_inh: float = E_INH,
          inputs_per_neurons: float = INPUTS_PER_NEURON, save_net: bool = False) -> PhasicNet:
    """The Deckers network + physics, ready for pnet.train(data, runcfg(), test_data=...). The three extensions of Table 4, each on its own
    (all True = ELSM, all False = the baseline B; main(config=) names the four columns):
        ei_balance        inh input weights ~ U[0, inh_input_factor * 0.4] (3 = the E/I optimum of Fig. 3) | False: the baseline's factor 1/2 (U[0, 0.2])
        sfa               threshold + 1 mV per exc spike, decaying with tau_sfa ~ U[50, 1050] ms (sfa_heterogeneous) or fixed 550 ms | False: fixed threshold
        tau_heterogeneous tau_mem ~ U[5, 55] ms per neuron | False: 30 ms everywhere
    n_liquid neurons, 80/20 exc/inh on `grid` (needs prod(grid) >= n_liquid). weights_format "sparse" (default) or "dense": same model."""
    n_exc = int(round(n_liquid * EXC_FRACTION)); n_inh = n_liquid - n_exc
    p_in = min(1.0, inputs_per_neurons / n_input)
    kern = lambda C: {"kernel": "gaussian", "C": C, "lambda": LAMBDA}
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    pnet.builder.set_neurons(("input", n_input, True), ("exc", n_exc), ("inh", n_inh, False, True))
    pnet.builder.set_positions(exc=tuple(grid), inh=tuple(grid), share="liquid")
    pnet.builder.set_synapses(("in_exc", "input", "exc"), ("in_inh", "input", "inh"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"),
                              ("inh_exc", "inh", "exc"), ("inh_inh", "inh", "inh"))
    pnet.builder.set_connectivity(in_exc=p_in, in_inh=p_in, exc_exc=kern(C_EE), exc_inh=kern(C_EI), inh_exc=kern(C_IE), inh_inh=kern(C_II))
    factor = inh_input_factor if ei_balance else INH_INPUT_FACTOR_BASELINE
    ##the factor multiplies the EXC range 0.4 (the paper's definition)
    pnet.builder.set_weights({"in_exc": (0.0, W_IN), "in_inh": (0.0, W_IN * factor), "exc_exc": (0.0, W_EE), "exc_inh": (0.0, W_EI),
                              "inh_exc": (0.0, W_IE), "inh_inh": (0.0, W_II)})      ##inh pre groups: Dale's sign applied by the builder (U ranges are magnitudes)
    pnet.builder.spec_name = "deckers_2022_spec"
    config = next((k for k, v in CONFIGS.items() if v == (ei_balance, sfa, tau_heterogeneous)), "custom")
    net = pnet.builder.realize(network_name=f"deckers_2022_{config}_n{n_liquid}", post_offset=True, save=False)
    if save_net:
        net.save_net(network_name=net.network_name, spec_name="deckers_2022_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)
    logger.info("liquid: %d exc + %d inh on %s; synapses %s (recurrent per neuron %.2f)", n_exc, n_inh, grid, net.synapse_counts,
                sum(v for k, v in net.synapse_counts.items() if not k.startswith("n_in_")) / n_liquid)

    sparse = weights_format == "sparse"
    pnet.cfg = DynamicsCfg(
        simulation=SimulationCfg(dt=dt_ms, weights_format=weights_format, update_on="timestep"),
        synapse=ELSMSynapseCfg(b=None, tau_ge=TAU_GE, tau_gi=TAU_GI, e_exc=e_exc, e_inh=e_inh, delays=dict(DELAYS_MS)),
        integrator=ELSMIntegratorCfg(tau_mem=TAU_MEM, v_rest=V_REST, r=1.0, tau_range=TAU_RANGE if tau_heterogeneous else None, t_refrac=T_REFRAC,
                                     t_refrac_inh=T_REFRAC_INH, v_init=V_INIT),
        threshold=ELSMThresholdCfg(v_reset=V_REST, v_th=V_TH, tau_th=TAU_SFA if sfa else None, adpt_th=SFA_INC if sfa else None, t_refrac=T_REFRAC,
                                   sfa_range=SFA_RANGE if (sfa and sfa_heterogeneous) else None, sfa_exc_only=True, t_refrac_inh=T_REFRAC_INH),
        plasticity=NoPlasticityCfg(),
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),      ##no noise in the paper; sigma_mem MUST stay 0 (refractory hold)
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(ELSM_SPARSE if sparse else ELSM_DENSE)     ##the pack (defined after the protocol below)
    pnet.cfg_name, pnet.run_name = "deckers_2022_cfg", "deckers_2022_run"
    return pnet


   ######################################################################################
##########################   Protocol   ###################################################
   ######################################################################################

@dataclass(frozen=True, slots=True)
class ELSMProtocolCfg(ProtocolCfg):
    """Fixed liquid -> responses -> readout. Hashed into run_id with the rest of RunCfg.
    Args:
        readout_l2, readout_lr, readout_epochs = readouts.LogisticReadout settings
        report_every = logging only (RunCfg.HASH_EXCLUDE lists protocol.report_every)
        reset_between_samples (inherited) must stay False: respond() resets the liquid itself and THEN draws v(0) ~ U[13.5, 15]
    DataCfg.n_train / n_test pick the samples; test_data MUST be given to PhasicNet.train (the official SHD test set).
    """
    protocol_type: str = "ELSM"
    readout_l2: float = 1e-3
    readout_lr: float = 0.02
    readout_epochs: int = 400
    report_every: int = 200


class ELSMProtocol(Protocol):
    """
    RESPONSES(train) -> RESPONSES(test) -> READOUT -> REPORT. Everything in mode "eval" (the liquid never learns; run.check_persistent_unchanged()
    proves it at the end). Response of a sample = EXC spike counts over the whole sample.
    STATE (proto__* in snapshots): Resp_train, Resp_test (int32), Y_train, Y_test (int64), Inh_spikes (2,: the train / test pass totals);
    progress = {"stage": 0 train | 1 test, "i": samples done} so a run with CheckpointCfg.every + train(out_dir=) can be resumed (train(resume_from=)).
    The two passes are self.ops.responses, the tail is self.ops.finish.
    """
    protocol_type = "ELSM"
    needs_test_set = True    ##the official SHD test set; Protocol.__init__ checks it

    def __init__(self, cfg: ELSMProtocolCfg | None, run, *, out_dir=None):
        super().__init__(ELSMProtocolCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c = self.cfg
        if c.reset_between_samples:   ##the flag would overwrite the U[13.5, 15] start with 13.5
            raise ValueError("ELSMProtocol resets the liquid itself (respond); leave reset_between_samples False.")
        self.Inh_spikes = np.zeros(2); self._phase = 0   ##inhibitory spike totals of the train / test pass (snapshotted)
        self._n_train, self._n_test = len(run.train_set), len(run.test_set)
        n_exc = run.ctx.net.neuronslices["exc"].stop - run.ctx.net.neuronslices["exc"].start
        self.Resp_train = np.zeros((self._n_train, n_exc), dtype=np.int32)
        self.Resp_test = np.zeros((self._n_test, n_exc), dtype=np.int32)
        self.Y_train = np.full(self._n_train, -1, dtype=np.int64)
        self.Y_test = np.full(self._n_test, -1, dtype=np.int64)

    def respond(self, frames, target, key: str) -> np.ndarray:
        """ONE sample on a FRESH liquid: reset_transient (Ge, Gi, rings, V_th, Since, V), v(0) ~ U[13.5, 15] keyed by `key`, run, exc spike counts."""
        run = self.run
        run.reset_transient()
        run.integrator.init_membrane(run.ctx, key)
        net = run.ctx.net
        counts = self.present(frames, target)                            ##every neuron, so the inhibitory spikes can be counted ...
        self.Inh_spikes[self._phase] += float(counts[net.inh_slice].sum())
        return counts[net.neuronslices["exc"]]                            ##... and the exc band is the response

    def execute(self) -> dict:
        run, c, ops = self.run, self.cfg, self.ops
        run.set_mode("eval")                       ##the liquid is fixed: W (and nothing else learned) is guarded from here on
        ops.header()
        self._phase = 0; ops.responses(run.train_set, self.respond, self.Resp_train, self.Y_train, stage=0, name="train", n=self._n_train)
        self._phase = 1; ops.responses(run.test_set, self.respond, self.Resp_test, self.Y_test, stage=1, name="test", n=self._n_test)
        run.check_persistent_unchanged()            ##eval contract: nothing in the model moved

        ##READOUT (readouts.py): logistic regression on the exc counts
        ro = fit_evaluate(self.Resp_train, self.Y_train, self.Resp_test, self.Y_test, l2=c.readout_l2, lr=c.readout_lr, epochs=c.readout_epochs)
        silent_train = int((self.Resp_train.sum(1) == 0).sum())
        res, files = readout_report(ro, self.Resp_train, self.Y_train, self.Resp_test, self.Y_test)     ##the shared tail; this protocol's own keys below
        res.update({"exc_spikes_per_sample_train": float(self.Resp_train.sum(1).mean()), "exc_spikes_per_sample_test": float(self.Resp_test.sum(1).mean()),
                    "active_exc_fraction_train": float((self.Resp_train > 0).mean()), "silent_samples_train": silent_train,
                    "exc_spikes_per_neuron_train": float(self.Resp_train.sum(1).mean() / self.Resp_train.shape[1]),   ##exc spikes per exc neuron per sample
                    ##Table 4's "spikes/neuron" counts ALL liquid neurons (exc + inh over n_liquid): the number to calibrate E_EXC against
                    "spikes_per_neuron_train": float((self.Resp_train.sum() + self.Inh_spikes[0]) / max(self._n_train, 1) / (self.Resp_train.shape[1] + run.ctx.net.n_inh)),
                    "inh_spikes_per_sample_train": float(self.Inh_spikes[0] / max(self._n_train, 1)),
                    "paper_reference": "Table 4 (SHD, 2000 neurons, mean of 5 runs): B 80.9 / B+E/I 82.1 / B+E/I+SFA 85.3 / ELSM 86.1 % test accuracy; spikes per neuron 8.56 / 7.58 / 6.03 / 6.83"})
        return ops.finish(res, files=files, lines=(
            f"  test accuracy : {res['test_acc']:.2f}%  (train {res['train_acc']:.2f}%)   paper: {res['paper_reference']}",
            f"  liquid        : {self._n_train} train + {self._n_test} test samples; exc spikes/sample {res['exc_spikes_per_sample_train']:.1f} "
            f"+ inh {res['inh_spikes_per_sample_train']:.1f} = {res['spikes_per_neuron_train']:.2f} per liquid neuron (paper Table 4: 8.56 / 7.58 / 6.03 / 6.83), "
            f"{silent_train} silent training samples"))


##the runtime classes as ONE pack each, reused by build() and by anyone assembling this model by hand
ELSM_DENSE = RuntimeClasses(synapse_cls=ELSMConductanceSynapse, integrator_cls=ELSMConductanceLIF, threshold_cls=SFAThreshold, plasticity_cls=NoPlasticity,
                            noise_cls=TestNoise, phase_classes=(DefaultPhase,), protocol_classes=(ELSMProtocol,))
ELSM_SPARSE = replace(ELSM_DENSE, synapse_cls=ELSMConductanceSynapseSparse)


def runcfg(*, every: int | None = None, n_train: int | None = None, n_test: int | None = None, **proto) -> RunCfg:
    """No folds, no epochs: the protocol is one pass over train and one over test. `every` = snapshot every N samples (needs train(out_dir=)).
    n_train / n_test -> DataCfg."""
    return RunCfg(data=DataCfg(make_test_split=False, test_fraction=0.0, n_train=n_train, n_test=n_test, shuffle=True),   ##DataCfg PICKS the samples (stratified, seeded, shuffled order)
                  checkpoint=CheckpointCfg(to_disk=False, every=every), scorer=ScorerCfg(rate_hz_min=0.0),
                  history=HistoryCfg(include_input=False, record_spikes=False, record_counts=True),      ##counts only: no raster for 10k x 1000 steps
                  protocol=ELSMProtocolCfg(**proto))


   ######################################################################################
##########################   Run   ########################################################
   ######################################################################################

def main(*, n_liquid: int = N_LIQUID, grid=GRID, dt: float = DT_MS, bin_ms: float = BIN_MS, t_max: float = T_MAX_MS, n_train: int | None = None,
         n_test: int | None = None, seed: int = SEED, config: str = "elsm", fixed_sfa: bool = False, inh_factor: float = INH_INPUT_FACTOR, e_exc: float = E_EXC,
         e_inh: float = E_INH, dense: bool = False, every: int | None = None, resume: str | None = None, tag: str = "", out: str | None = None):
    """ONE run; every setting is a keyword argument.
        config: "b" | "b_ei" | "b_ei_sfa" | "elsm" = Table 4's columns;
        bin_ms: real ms of SHD per liquid step (4 ms -> 250 steps per sample); fixed_sfa=True: tau_sfa fixed 550 ms;
        dense=True: dense weights (default sparse); every: snapshot every N samples; resume: a snap_*.npz to continue from; out: the run folder (default
        runs/deckers_2022_<config>_n<N>[_tag]).
    Returns the Run: run.report = ELSMProtocol's results (also <out>/results.json; log in <out>/run.jsonl, numbers in <out>/results.jsonl, status in <out>/status.json)."""
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {sorted(CONFIGS)}, got {config!r}")
    ei_balance, sfa, tau_heterogeneous = CONFIGS[config]
    train, test = shd_dataset(bin_ms=bin_ms, t_max_ms=t_max)
    pnet = build(n_liquid=n_liquid, grid=tuple(grid), seed=seed, dt_ms=dt, weights_format="dense" if dense else "sparse", ei_balance=ei_balance, sfa=sfa,
                 tau_heterogeneous=tau_heterogeneous, sfa_heterogeneous=not fixed_sfa, inh_input_factor=inh_factor, e_exc=e_exc, e_inh=e_inh)
    out = Path(out) if out else RUNS_DIR / f"deckers_2022_{config}_n{n_liquid}{('_' + tag) if tag else ''}"
    return pnet.train(train, runcfg(every=every, n_train=n_train, n_test=n_test), test_data=test, out_dir=out, resume_from=resume,
                      save_run=False, save_history=False, dataset_name="SHD")


if __name__ == "__main__":
    main()
