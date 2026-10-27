#Run - ONE hyperparameter setting, end to end. The runtime that RunHistory is a history OF. 
#
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence
from dataclasses import asdict
from pathlib import Path 
import logging
import numpy as np
from meta import RunMeta
from runcfg import RunCfg
from checkpoint import CheckpointPolicy, save_trained_model, take_snapshot
from scorer import Scorer
from states import RunContext, NEVER_SPIKED, _since_update
import capture
from dynamics import Synapse, Integrator, Threshold, Plasticity, Noise, Phase
from datahandler import DataHandler
from modelinput import ModelInput    
from phasescheduler import PhaseScheduler
from history import RunHistory, MODE_KEY    
from utils import _write_json_yaml, _inf_safe     
from paths import RUNS_DIR                 

logger = logging.getLogger(__name__)


   ######################################################################################
##########################   Run   ######################################################
   ######################################################################################

class Run:
    """
    Drives ONE hyperparameter setting: folds -> epochs -> samples -> timesteps (+ one final refit for k-fold).
    Run = an ENGINE with run-primitives. 
    
    Protocol is written from primitives (or ops=prepackaged primitives): 
    
    Primitives:
    
        step() / run_sample(until=) / continue_sample(until=) / finish_sample(),
        set_mode("train" | "eval")    
        reset_neuronal_state < reset_transient < reset_model
    
    test() = convenience delegator

    OWNS everything instantiated: dynamics modules, ctx (all state arrays), phases, history, checkpoints.
    Args:
        synapse, integrator, threshold, plasticity, noise, phases = INSTANTIATED dynamics modules
        ctx = RunContext (all shared, mutable state arrays + network + rng)
        datahandler = per-timestep handover + fold/epoch data operations
        train, test = the one-time split (Subsets). test is held and never touched until .test()
        history, runmeta, runcfg = the record, the identity, the technical settings of THIS run
        phasescheduler = PhaseScheduler, stepped ONCE per timestep AFTER the physics step
    """

    def __init__(self, synapse: Synapse, integrator: Integrator, threshold: Threshold, plasticity: Plasticity,
                 phases: Dict[str, Phase], noise: Noise, ctx: RunContext, phasescheduler: PhaseScheduler,
                 datahandler: DataHandler, history: RunHistory, runcfg: RunCfg,
                 train: ModelInput, test: ModelInput | None = None):


        ##INSTANTIATED classes
        self._datahandler = datahandler 
        self._phasescheduler = phasescheduler

        # dynamics
        self._synapse = synapse
        self._integrator= integrator        
        self._threshold = threshold
        self._plasticity = plasticity
        self._noise = noise
        self._phases=phases

        # states, cfg, meta
        self.ctx = ctx                  #shared States + net
        self.history= history           #includes history.derived_cfg (the cfg ACTUALLY used during run)
        self.runcfg = runcfg          # runcfg = technical settings: epochs/folds/patience + history/scorer/checkpoint/data policies

        #runtime use
        self.train_set, self.test_set = train, test

        ##THE ONE MODE DECISION-> dense vs sparse
        dense = ctx.upd.dW is not None
        self._step_impl = type(self).step_dense if dense else type(self).step_sparse  ##NOTE NOT BOUND (allow deletion without gc)
        self._clamp_impl = type(self)._clamp_dense if dense else type(self)._clamp_sparse
        self._clamp_plan = self._plan_clamp(dense)  #resolve once**

        self._eval_only = False        
        self._learning = True
        if history is not None:
            history.mode_timeline.mark(ctx.sim.current_runstep, MODE_KEY["train"])  

        self._eval_guard = None
        self._count_spikes = history is not None and bool(history.cfg.record_counts)

        self.checkpoints = CheckpointPolicy(runcfg.checkpoint, runmeta=self.runmeta, dataset_id=self.runmeta.dataset_id,
                                            ctx=ctx, modules=self.modules)
        self.scorer = Scorer(runcfg.scorer)
        ##As-built state_dict of every moduleE: what a fit-reset restores
        self._init_state = {name: {k: v.copy() for k, v in m.state_dict.items()}
                            for name, m in self.modules.items()}
        ##TRANSIENT = state_dict minus 'persistent', resolved ONCE
        self._transient_init = {name: {k: v for k, v in init.items() if k not in getattr(m, "persistent", ())}
                                for (name, init), m in zip(self._init_state.items(), self.modules.values())}
        for name, m in self.modules.items():     ##BACKWARD COMPATIBILITY (TODO later: remove)
            if getattr(m, "learned", None):
                raise AttributeError(f"{type(m).__name__}.learned = {m.learned!r}: 'learned' was renamed 'persistent' (22.10). Declare persistent = {m.learned!r}.")
        self._W0 = ctx.syn.W.copy()     ##reset_model needs the realized copy, release_training_state frees it
        self.report: dict | None = None
        self.status: dict | None = None     #running / completed / failed / interrupted

    ##safety properties; avoid overwriting/editing/setting classes
    @property
    def datahandler(self):
        return self._datahandler
    @property
    def synapse(self):
        return self._synapse
    @property
    def integrator(self):
        return self._integrator
    @property
    def threshold(self):
        return self._threshold
    @property
    def plasticity(self):
        return self._plasticity
    @property
    def noise(self):
        return self._noise
    @property
    def phases(self):
        return self._phases
    @property
    def runmeta(self) -> RunMeta:
        """The run's identity. Delegates to history.runmeta: ONE object, nothing to keep in sync."""
        return self.history.runmeta

    def step(self):
        """ONE runtime step. Dense or sparse orchestra, chosen ONCE in __init_."""
        return self._step_impl(self)

   ######################################################################################
##########################   MODE    ######################################################
   ######################################################################################

    def set_mode(self, mode: str) -> None:
        """
        "train" -> plasticity runs (the commit W += dW happens). 
        "eval" -> forward only, persistent arrays frozen & guarded
        
        'Marks history.mode_timeline (0=train, 1=eval: MODE_KEY). The same mode again is a no-op."""
        mode = str(mode).lower()
        if mode not in MODE_KEY:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        if mode == "train" and self._eval_only:   
            raise RuntimeError("release_training_state() was called: this Run can only be evaluated (mode 'eval'), not trained.")
        sim = self.ctx.sim
        if mode == sim.mode:
            return
        if mode == "eval":
            self._eval_guard = [kept.copy() for _, kept in self._persistent()]
        else:
            self.check_persistent_unchanged()
            self._eval_guard = None
        sim.mode = mode
        self._learning = mode == "train"
        if self.history is not None:
            self.history.mode_timeline.mark(sim.current_runstep, MODE_KEY[mode])

    def _persistent(self) -> list:
        """Non-transient arrays."""
        out = [("W", self.ctx.syn.W)]
        for name, m in self.modules.items():
            for k in getattr(m, "persistent", ()):
                out.append((f"{name}.{k}", getattr(m, k)))
        return out

    def check_persistent_unchanged(self) -> None: 
        """Confirm no persistent-array drift during validation (eval) """
        if self._eval_guard is None:
            return
        for (label, live), kept in zip(self._persistent(), self._eval_guard):
            if not np.array_equal(live, kept, equal_nan=True):      ##equal_nan: a NaN already in W/Theta before eval is not "eval changed it"
                raise RuntimeError(f"mode 'eval' changed a persistent array: {label}. Evaluation must not train "
                                   f"(an unfrozen adaptive threshold? a phase or noise writing W? a protocol writing in the wrong mode?).")

    def _clamp(self) -> None:
        return self._clamp_impl(self)

    @property
    def modules(self) -> dict:
        return {"synapse": self._synapse, "integrator": self._integrator, "threshold": self._threshold,
                "plasticity": self._plasticity, "noise": self._noise}

   ######################################################################################
##########################   RESET    ######################################################
   ######################################################################################

    def reset_neuronal_state(self) -> None:
        """Return neuronal state to baseline. NOT W or synaptic states, NOR local, specific states."""
        ctx = self.ctx
        neur, upd = ctx.neur, ctx.upd
        neur.V[...] = self._integrator.v_rest
        neur.Spikes[...] = False
        upd.step_input[...] = False
        for array in (neur.pre_spike, neur.post_spike):
            if array is not None:
                array[...] = 0.0
        if neur.Since is not None:
            neur.Since[...] = NEVER_SPIKED      

    def reset_transient(self) -> None:
        """
        a NEW NETWORK with the SAME MODEL. 

            - Neuronal state + every module's transient arrays reset to "start" 
            - W and the persistent arrays (e.g. Theta) retained
        
        NOTE effectively between reset_neuronal_state (neuronal only) and reset_model (reset everything)."""
        for name, module in self.modules.items():
            module.load_state_dict(self._transient_init[name])      ##resolved once in __init__
        if self.ctx.upd.dW is not None:
            self.ctx.upd.dW[...] = 0.0
        self.reset_neuronal_state()

    def reset_model(self) -> None:
        """fit-level reset: state AND weights back to as-built. 
        A fit = a NEW model on a new split, (carrying weights across fits make them dependent ->         variance estimate averaging them => biased."""
        if self._eval_only:  
            raise RuntimeError("release_training_state() was called: the as-built weights needed for a reset are gone.")
        self.set_mode("train") 
        np.copyto(self.ctx.syn.W, self._W0)
        self._rest_state()
        logger.debug("Model reset to as-built weights for a new fit.")

    def _rest_state(self) -> None:
        """Everything EXCEPT W back to as-built: neuronal state + every module's state_dict (traces, refractory counts...)."""
        for name, module in self.modules.items():
            module.load_state_dict(self._init_state[name])      ##module-owned state (traces, refractory counts, ...)
        self.reset_neuronal_state()

    def release_training_state(self) -> None:
        """
        END-OF-LIFE: Keep ONLY everything test() and the artifacts need:
        (ctx incl. W, modules, history, report, datahandler, test set). 
    
        """
        self._W0 = None
        self.checkpoints.best = None
        self.checkpoints.collapse = None
        self.ctx.upd.dW = None          ##dense plasticity buffer; never read on a learn=False step 
        self._eval_only = True  
        self.ctx.sim.mode, self._learning, self._eval_guard = "eval", False, None      
        if self.history is not None:
            self.history.mode_timeline.mark(self.ctx.sim.current_runstep, MODE_KEY["eval"])



    def _plan_clamp(self, dense: bool):
        """
         Resolve once:
            - Only PLASTIC projections clamped; 
          
            - default bounds per projection follow Dale's law from the presynaptic group.
        """
        syn, net = self.ctx.syn, self.ctx.net
        w_min, w_max = self._plasticity.w_min, self._plasticity.w_max
        frozen = set(syn.frozen) if syn.frozen is not None else set()
        if "all" in frozen or not (np.isfinite(w_min) or np.isfinite(w_max)):
            return [] if dense else None
        ##NOTE 
        ## (NALSM hook: the input projection is plastic within [-W_max, W_max], either sign). 
        bounds = getattr(self._plasticity, "clamp_bounds", None) or {}
        if dense:
            plan = []
            for name, proj in net.synapseslices.items():
                if name in frozen:
                    continue
                inh = net.neuronspecs[net.synapsespecs[name].pre_neuron].inh
                lo, hi = bounds.get(name, (w_min, 0.0) if inh else (0.0, w_max))      ##(NALSM): per-projection override
                plan.append((proj.post, proj.pre, lo, hi))
            return plan
        is_inh = net.inh_neuron_mask[syn.pre]
        plastic = syn.plastic_mask if syn.plastic_mask is not None else np.ones(syn.W.size, bool)
        lo = np.where(plastic, np.where(is_inh, w_min, 0.0), -np.inf).astype(syn.W.dtype)    ##one-time**

        hi = np.where(plastic, np.where(is_inh, 0.0, w_max), np.inf).astype(syn.W.dtype)
        for name, (blo, bhi) in bounds.items():                                             
            if name in frozen:
                continue
            proj = net.synapseslices[name]
            m = plastic & (syn.post >= proj.post.start) & (syn.post < proj.post.stop) & (syn.pre >= proj.pre.start) & (syn.pre < proj.pre.stop)
            lo[m] = blo; hi[m] = bhi
        return (lo, hi)

    def _clamp_dense(self) -> None:
        """Hard clamp W, PER SIGN, using the contiguous presynaptic column ranges."""
        W = self.ctx.syn.W
        for post, pre, lo, hi in self._clamp_plan:
            np.clip(W[post, pre], lo, hi, out=W[post, pre])

    def _clamp_sparse(self) -> None:
        """Clamp synapses: per-syn sign lookup on the presynaptic neuron.
        """
        if self._clamp_plan is None:
            return
        lo, hi = self._clamp_plan
        np.clip(self.ctx.syn.W, lo, hi, out=self.ctx.syn.W)

    ##############################################################################################
    ##################################------The Orchestra -----###################################

    ############# Bookkeeping helpers: ###################################
    def _afferent(self, phase) -> None:
        """DATALOADER STEP. Afferent input for THIS timestep. A phase may suppress it, in which case the cursor
        does NOT advance and the sample stretches rather than ending early."""
        ctx, upd, neur, net = self.ctx, self.ctx.upd, self.ctx.neur, self.ctx.net
        if phase.dataloader_active:
            self._datahandler.step(ctx)
        else:
            phase.alternate_inputs(ctx)
        phase.post_dataloader(ctx)
        # Load input nodes w/ spike data
        neur.Spikes[:net.n_input] = upd.step_input

    def _record(self) -> None:
        """history: record EVERY step:  --> index == timestep."""
        sim, history = self.ctx.sim, self.history
        if history is None:
            return
        if self._count_spikes:         
            np.add(history.Counts, self.ctx.neur.Spikes, out=history.Counts)
        if history.cfg.record_spikes:
            history.record_step(sim.current_runstep, capture.spike_indices(self.ctx, include_input=history.cfg.include_input))
        history.mark_labels(sim.current_runstep, phase=sim.current_phase, epoch=sim.current_epoch, datarun=sim.current_datarun)

    def _tick(self) -> None:
        sim = self.ctx.sim
        sim.current_runstep += 1
        if self._learning:
            sim.train_step += 1     ##event-mode time base ticks in mode "train" only (because traces must not learn during eval)
        sim.time_in_current_phase += 1
        sim.time_in_phase[sim.current_phase] += 1

    
    def step_dense(self): ##NOTE here the ONLY RunContext NOT available is that given by experiment (ie backend settings, rng etc. Should that be included?)
        """ONE runtime step, DENSE: modules write into shared UpdateState buffers and return nothing.

        SNN time-stepping convention: afferent signals occurred during the previous timestep,
        ## but ENTER this system at the start of "this" step, together with the recurrent spikes carried over from the previous step.
        # (IE dataloader step placed at start of this step by convention, but "happened" end of last step)
        
        """
        ##RunContext for shared args (step functions)
        ctx = self.ctx
        ##RunContext unwrapped for orchestrator use
        upd = ctx.upd
        sim = ctx.sim
        neur = ctx.neur
        syn = ctx.syn
        net = ctx.net
        phase = self.phases[sim.current_phase]

        self._afferent(phase)

    ###### SYNAPSE STEP: 
        # W @ Spikes -> I summed per neuron (upd.I2V)
        self.synapse.step(ctx)
        phase.post_synapse(ctx)

    ###### INTEGRATOR STEP:
        # Integrate membrane -> dV per neuron (upd.dV)
        self.integrator.step(ctx)
        phase.post_integrator(ctx)
        self.noise.on_membrane(ctx, phase)
        np.add(neur.V, upd.dV, out=neur.V)        # neur.V += upd.dV #Update internal state array

    ###### THRESHOLD STEP:
        # (Update) threshold -> emit spikes (upd.rec_spikes)
        self.threshold.step(ctx)
        phase.post_threshold(ctx)

        neur.Spikes[net.n_input:] = upd.rec_spikes[net.rec_slice]
        if neur.Since is not None:
            _since_update(neur.Since, neur.Spikes[net.n_input:], net.rec_slice)  

        ## RESET recurrent neurons 
        np.copyto(neur.V[net.rec_slice], self.threshold.v_reset, where=neur.Spikes[net.n_input:])

        self._record() ##RECORD SPIKES

    ###### PLASTICITY STEP:
        ##Skip on a forward-only pass
        if not self._learning:         
            self._tick()
            return

        # Update Weights -> masked weight update (upd.dW)
        if neur.pre_spike is not None:
            np.copyto(neur.pre_spike, neur.Spikes)          ##all N (input + rec)
        if neur.post_spike is not None:
            np.copyto(neur.post_spike[net.rec_slice], neur.Spikes[net.n_input:]) #rec only

        self.plasticity.step(ctx)                  ##dW computed, frozen rectangles zeroed, masked to realized synapses
        self.noise.on_weights(ctx, phase)          ##Weight noise hook (inactive in TestNoise).
        phase.post_plasticity(ctx)
        np.add(syn.W, upd.dW, out=syn.W)            ##COMMIT:gated by 'learn'- 
        self._clamp()

        #####ALTERNATIVE NAMING: phase.pre_threshold_mod, post_threshold, pre_threshold, threshold_mod, spiking_mod 
        self._tick()

    def step_sparse(self):
        """ONE runtime step, SPARSE: modules RETURN their outputs.
        Return-style when touching sparse weights. Direct to upd in dense neuron-only states..
       NOTE the plasticity step returns EITHER the (n_synapses,) dw (update_on timestep) OR (k, dw_k), the touched edges and their dw
        (update_on event / hybrid_event); the commit + clip then run over k only.
        """
        ctx = self.ctx
        upd = ctx.upd
        sim = ctx.sim
        neur = ctx.neur
        syn = ctx.syn
        net = ctx.net
        phase = self._phases[sim.current_phase]

        self._afferent(phase)

    ###### SYNAPSE STEP:
        upd.I2V[...] = self.synapse.step(ctx)      
        phase.post_synapse(ctx)

    ###### INTEGRATOR STEP:
        self.integrator.step(ctx) ##NOTE even with sparse weights, neur stays dense here and is vectorized
        phase.post_integrator(ctx)
        self.noise.on_membrane(ctx, phase)
        np.add(neur.V, upd.dV, out=neur.V)

    ###### THRESHOLD STEP:
        self.threshold.step(ctx)
        phase.post_threshold(ctx)
        neur.Spikes[net.n_input:] = upd.rec_spikes[net.rec_slice]
        if neur.Since is not None:
            _since_update(neur.Since, neur.Spikes[net.n_input:], net.rec_slice)   

        np.copyto(neur.V[net.rec_slice], self._threshold.v_reset, where=neur.Spikes[net.n_input:])

        self._record()   

    ###### PLASTICITY STEP:
        if not self._learning:
            self._tick()
            return
        if neur.pre_spike is not None:
            np.copyto(neur.pre_spike, neur.Spikes)
        if neur.post_spike is not None:
            np.copyto(neur.post_spike[net.rec_slice], neur.Spikes[net.n_input:])

        dw = self.plasticity.step(ctx)             ##(n_synapses,) pre-plastic-masked per-edge update, OR the pair (k, dw_k) = the touched edges + their dw
        self.noise.on_weights(ctx, phase)
        phase.post_plasticity(ctx)
        if isinstance(dw, tuple):                   # commit + clip over the TOUCHED edges only 
            k, dw_k = dw                            ##NOTE k= unique by construction (SynapticIndex.spiking_syns: disjoint CSR / CSC ranges, de-duplicated union), so no np.add.at
            if self._clamp_plan is None:
                syn.W[k] += dw_k
            else:
                lo, hi = self._clamp_plan
                syn.W[k] = np.clip(syn.W[k] + dw_k, lo[k], hi[k])     ##NOTE the clip LAST; the same per-element ops as the full-array branch -> bit-identical W (test_sparse) 
        else:
            np.add(syn.W, dw, out=syn.W)                ##COMMIT
            self._clamp()
        self._tick()

    # ####################################
    ###############################################
    # timesteps of ONE sample)
    # #################################


    def run_sample(self, sample, target: Any = None, *, until: int | None = None) -> int:
        """
        Run ONE sample to its end. Returns the number of timesteps executed.
        
          until= stop (temporarily) after 'until' steps of the sample 
        """
        ctx = self.ctx
        sim = ctx.sim


        T = self._datahandler.begin_sample(sample, ctx)
        stop = T if until is None else min(int(until), T)
        return self._run_to(stop)   

    def continue_sample(self, until: int) -> int: 
        """
        Run the bound sample from where run_sample(until=) / an earlier continue_sample stopped to step "until" (clipped to sim.sample_T).
        Returns the steps executed (0 when the sample is already there).
        
        called by Operations.binned_counts(one presentation, counts per time window)"""
        T = self.ctx.sim.sample_T
        return self._run_to(min(int(until), T)) 

    def finish_sample(self) -> int:
        """ run the bound sample from where run_sample(until=) stopped to its end. Returns the steps executed."""
        return self.continue_sample(self.ctx.sim.sample_T)  

    def _run_to(self, stop: int) -> int:
        """ 
        per-step loop:
            step, until step_in_sample reaches 'stop'. 
        
         NOTE cursor in data does not advance when dataloader_active=False, so stretches the sample
         """
        ctx = self.ctx
        sim = ctx.sim
        scheduler = self._phasescheduler
        executed = 0
        while sim.step_in_sample < stop:
            self.step()
            executed += 1

            ##SPhaseScheduler
            if scheduler is not None:
                switched = scheduler.step(ctx)
                if switched and "W" in self._phases[switched].transition_writes:
                    self._clamp()

        return executed


    # -------------------------------------------------------------------------
    # Convenience + records
    # -------------------------------------------------------------------------

    def test(self, data=None) -> dict:
        """
        ONE forward pass over the HELD-OUT test set with the model currently loaded 
        select="best" => its restored best epoch.
        select="final" => whatever it ended with
        learn=False
        
        """
        from operations import Operations            
        cfg = self.runcfg.protocol                   
        if self.history is None or not self.history.cfg.record_counts:     
            raise RuntimeError("A Protocol reads its responses from history.Counts: HistoryCfg.record_counts must be True.")
        return Operations(self, reset_between_samples=cfg is not None and cfg.reset_between_samples).test(data)

    def save_record(self, out_dir: str | Path, *, runtime_classes=None) -> None:
        """Self-identification before run begins:
         
          -  network.json/.npz, cfg_derived.json, runcfg.json, runmeta.json, runtime_classes.json."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.net.save_net(filename="network", JSON=True, YAML=False, save_to=out_dir)
        _write_json_yaml(out_dir / "cfg_derived", _inf_safe(asdict(self.history.derived_cfg)), JSON=True, YAML=False, log_label="Derived cfg")
        _write_json_yaml(out_dir / "runcfg", _inf_safe(asdict(self.runcfg)), JSON=True, YAML=False, log_label="RunCfg")
        _write_json_yaml(out_dir / "runmeta", _inf_safe(asdict(self.runmeta)), JSON=True, YAML=False, log_label="RunMeta")
        if runtime_classes is not None:
            _write_json_yaml(out_dir / "runtime_classes", runtime_classes.paths(), JSON=True, YAML=False, log_label="RuntimeClasses")

    def save_trained_model(self, *, save_dir: str | Path | None = None):
        """The run's model -> runs/TrainedModel_<run_id>. 
        save_dir= (a run folder); None = runs/."""
        c = self.runcfg.checkpoint
        best = self.checkpoints.best
        score = best.score if (c.select == "best" and best is not None) else None
        snap = take_snapshot(self.ctx, epoch=self.ctx.sim.current_epoch, score=score, reason=c.select, modules=self.modules)
        if c.content == "weights":
            snap.neur, snap.upd = {}, {}
            snap.modules = {n: {k: v for k, v in d.items() if k in getattr(self.modules[n], "persistent", ())} for n, d in snap.modules.items()}
            snap.modules = {n: d for n, d in snap.modules.items() if d}
        return save_trained_model(snap, runmeta=self.runmeta, dataset_id=self.runmeta.dataset_id,
                                  save_dir=RUNS_DIR if save_dir is None else Path(save_dir))

    # #####################

