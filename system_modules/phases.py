from __future__ import annotations
from dynamicscfg import SimulationCfg, PhaseCfg, SleepPhaseCfg
import capture   ##observables are module functions now
import numpy as np
from dynamics import Phase
from states import RunContext


import logging


logger = logging.getLogger(__name__)



class DefaultPhase(Phase):
    """ALL phase-modifications are INACTIVE. 
    
    Default runtime defined by other runtime modules"""
    phase_type = "DEFAULT"


class SleepPhase(Phase):
    phase_type = "SLEEP"       
    transition_writes = ("W",) 
    weights_format = None       

    def __init__(self, cfg: SleepPhaseCfg, sim_cfg: SimulationCfg=None, rng: np.random.Generator = None):
        super().__init__(cfg, sim_cfg, rng=rng)

        self.sleep_fraction = np.float32(cfg.sleep_fraction)
        self.jolt_magnitude = np.float32(cfg.jolt_magnitude)
        self.revive_syn_group = cfg.revive_syn_group
        self.dead_eps = np.float32(cfg.dead_eps)
        self.noise_scale = np.float32(cfg.noise_scale)
        self.noise_shift = np.float32(cfg.noise_shift)


    def revive_dormant_synapses(self, ctx, 
                                *, dead, 
                                fraction: float, jolt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Revive a fraction of deadened synapses by adding a weight jolt.

        Args:
            dead = what capture.deadened_synapses() returned: 
                *If dense: COO-style coordinates (dead_post_idx, dead_pre_idx)
                *if sparse: synapse-indices 
            fraction:
                Fraction of deadened synapses to revive.

            jolt:
                Signed Weight boost (increases magnitude, + or -)

        Returns:
            revived_post_idx, revived_pre_idx
        """

        if self.rng is None:
            raise RuntimeError("SleepPhase requires rng.")

        edges = dead if ctx.syn.W.ndim == 1 else None
        dead_post_idx, dead_pre_idx = (ctx.syn.post[edges], ctx.syn.pre[edges]) if edges is not None else dead

        if dead_post_idx.size != dead_pre_idx.size:
            raise ValueError("dead_post_idx and dead_pre_idx must have the same length.")

        n_dead = dead_post_idx.size

        if n_dead == 0:
            return (np.array([], dtype=np.int64),  np.array([], dtype=np.int64))

        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction!r}.")

        if fraction == 0.0:
            return ( np.array([], dtype=np.int64),  np.array([], dtype=np.int64))

        n_revive = int(np.ceil(n_dead * fraction))

        selected = self.rng.choice( n_dead, size=n_revive, replace=False,)

        revived_post_idx = dead_post_idx[selected]
        revived_pre_idx = dead_pre_idx[selected]

        jolt_signed = np.where(ctx.net.inh_neuron_mask[revived_pre_idx], -np.float32(jolt), np.float32(jolt))     
        if edges is None:
            #     W[post, pre]
            ctx.syn.W[revived_post_idx, revived_pre_idx] += jolt_signed
        else:
            ctx.syn.W[edges[selected]] += jolt_signed     

        return revived_post_idx, revived_pre_idx
    
    def on_enter(self, ctx: RunContext, capture=None) -> None:
            """
            Sleep entry hook.

            Finds deadened synapses and revives a fraction of them.
            """

            if capture is None:
                logger.warning("No datacapture class. Returned with no change")
                return

            if self.revive_syn_group is None:
                return

            if self.sleep_fraction <= 0.0:
                return

            if self.jolt_magnitude <= 0.0:
                return

            dead = capture.deadened_synapses(ctx,syn_group=self.revive_syn_group,eps=float(self.dead_eps),)

            self.revive_dormant_synapses( ctx, dead=dead, 
                                         fraction=float(self.sleep_fraction),  jolt=float(self.jolt_magnitude),)
    
    def on_noise(self, ctx: RunContext, current_noise) -> None:
        """Sleep modifies noise directly INSIDE noise module"""

        np.multiply(current_noise, self.noise_scale, out=current_noise)
        np.add(current_noise, self.noise_shift, out=current_noise)
