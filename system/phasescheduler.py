#PhaseScheduler 

from __future__ import annotations
import logging
import inspect     
import numpy as np    

import capture  ##Stateless class w/ algorithms for collecting observables -> module functions (capture remains namespace)
from dynamics import Phase
from utils import _COMPARISONS, _capitalize_strip_name    

logger = logging.getLogger(__name__)

## a trigger is  {capture_name: (value, comparison)}  or, TWO-STEP,  {capture_name: (value, comparison, summary, value2, comparison2)}.
## Step 1 compares the capture's OUTPUT with 'value' by isinstance: ndarray -> elementwise -> a bool array of the same shape; a number (int / float / bool /
## numpy scalar) -> one bool; a tuple -> "compare two": 'value' is a tuple of the same length, element i compared with value[i] by the same rule, the results
## concatenated into ONE bool sequence. 
# 
# Step 2 (optional) reduces that sequence with 'summary' ("sum" = how many satisfy, "mean" = the fraction, "any", "all")
## and compares the summary with value2 -> the bool a trigger needs. A one-step trigger whose step-1 result is not ONE bool is an error (add the summary stage).
## Nothing more ("For now I DONT need more than that"): a trigger carries no kwargs, so a capture that needs one (sum_groupweight) is not trigger-usable. ##NOTE TODO: ADD MINIMAL kwargs system that alows kward capture functions.

_SUMMARIES = {"sum": np.count_nonzero, "mean": np.mean, "any": np.any, "all": np.all}


def validate_triggers(triggers: dict | None, label: str) -> None:
    """
    SAFETYVALVE build-time check of one phase's triggers / exit_triggers (PhasicNet._instantiate_phases): 
        - the capture name exists, 
        - comparison is in _COMPARISONS, 
        - the summary of the two-step form in _SUMMARIES
        """
    
    for name, spec in (triggers or {}).items():
        fn = getattr(capture, name, None)
        if not callable(fn):
            raise ValueError(f"{label}: trigger {name!r} is not a capture function (capture.{name} does not exist).")
        
        params = list(inspect.signature(fn).parameters.values()) if inspect.isfunction(fn) else []
        needs = tuple(p.name for p in params[1:] if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
        if not params or params[0].name != "ctx" or needs:
            raise ValueError(f"{label}: trigger {name!r} is not trigger-usable: a trigger calls capture.{name}(ctx) and nothing else"
                             + (f", which needs {needs} too." if needs else " (not a capture function of ctx)."))
        if len(spec) not in (2, 5):
            raise ValueError(f"{label}: trigger {name!r} must be (value, comparison) or (value, comparison, summary, value2, comparison2), got {spec!r}.")
        for comparison in ([spec[1], spec[4]] if len(spec) == 5 else [spec[1]]):
            if comparison not in _COMPARISONS:
                raise ValueError(f"{label}: trigger {name!r}: unknown comparison {comparison!r}. Expected one of {tuple(_COMPARISONS)}.")
        if len(spec) == 5 and spec[2] not in _SUMMARIES:
            raise ValueError(f"{label}: trigger {name!r}: unknown summary {spec[2]!r}. Expected one of {tuple(_SUMMARIES)}.")


def _compare(out, value, compare, name: str):
    """step 1 -- the capture output against 'value', by the output's type (the header note)."""
    if isinstance(out, tuple):
        if not isinstance(value, (tuple, list)) or len(value) != len(out):
            raise ValueError(f"trigger {name!r}: capture.{name} returns a tuple of {len(out)}, so its value must be a tuple of {len(out)}, got {value!r}.")
        return np.concatenate([np.atleast_1d(_compare(o, v, compare, name)) for o, v in zip(out, value)])
    if isinstance(out, np.ndarray):
        result = compare(out, value)
        return bool(result) if np.ndim(result) == 0 else result
    if isinstance(out, (int, float, bool, np.number, np.bool_)):
        return bool(compare(out, value))
    raise TypeError(f"trigger {name!r}: capture.{name} returned a {type(out).__name__}; a trigger compares an array, a number or a tuple of those.")


class PhaseScheduler():
    """Integrated phase scheduler, w/ hooks in Orchestra.
    
    Evaluates phaserules - triggers to decide on phase-switching between registered phases."""
    def __init__(self, capture=capture, *, phases: dict[str, Phase]):
        self.capture = capture
        self.phases = phases

    @property
    def triggers_all(self):
        _triggers: dict[Phase: dict] = {}
        for phase in self.phases.values():  
            _triggers[phase] = phase.triggers  
        return _triggers                        
    
    def check_triggers(self, ctx, triggers: dict, mode: str = "any") -> bool:
        capture = self.capture
        if not triggers:
            return False ##Safety valve

        results = []
        for capture_func_name, spec in triggers.items():     ##(value, comparison) or the two-step (value, comparison, summary, value2, comparison2)
            trigger_value, comparison = spec[0], spec[1]
            if comparison not in _COMPARISONS:
                raise ValueError(f"Unknown comparison {comparison!r}")

            cap_func = getattr(capture, capture_func_name)
            current_value = cap_func(ctx)

            compare = _COMPARISONS[comparison]
            result = _compare(current_value, trigger_value, compare, capture_func_name)  
            if len(spec) == 5: 
                summary, value2, comparison2 = spec[2:]
                result = bool(_COMPARISONS[comparison2](_SUMMARIES[summary](result), value2))
            elif not isinstance(result, bool):
                raise ValueError(f"trigger {capture_func_name!r} compares {np.size(result)} values, not one: add a summary stage -- "
                                 f"(value, comparison, summary, value2, comparison2) with summary in {tuple(_SUMMARIES)}.")
            results.append(result)

        if mode == "all":
            return all(results)

        if mode == "any":
            return any(results)

        raise ValueError(f"Failure in evaluate_trigger method (phasescheduler)")

    def switch_phase(self, current_phase, new_phase, ctx):
        if new_phase.transition_writes and ctx.sim.mode == "eval":     ##NOTE IMPORTANT: a phase writing a persistent array at its entry (SleepPhase's revival) is refused in eval

            raise RuntimeError(f"Cannot switch into phase '{new_phase.phase_type}' in mode 'eval': it declares transition_writes={new_phase.transition_writes} "
                               f"(a write to a persistent array; evaluation must not train).")

        logger.info("Switching phase from %s to %s at runstep %s (dataloader_active=%s).", current_phase.phase_type, new_phase.phase_type,     
                    ctx.sim.current_runstep, new_phase.dataloader_active)

        
        current_phase.on_exit(ctx, self.capture)
        new_phase.on_enter(ctx, self.capture)
        ctx.sim.current_phase = _capitalize_strip_name(new_phase.phase_type)    
        ctx.sim.time_in_current_phase = 0  ## NOTE time_in_phase[name] is CUMULATIVE per phase and NOT reset.
        return ctx.sim.current_phase  



    def step(self, ctx):
        """Evaluate all non-active phases and switch if one's triggers fire
        
        NOTE the CURRENT phase's exit_triggers are checked FIRST. If they fire -> back to DEFAULT. Entry triggers checked second.
        
        Returns the (normalised) name of the phase switched into, None when nothing fired
        """
        curr_phase = self.phases[ctx.sim.current_phase]
        default = self.phases["DEFAULT"]
        if curr_phase is not default and self.check_triggers(ctx=ctx, triggers=curr_phase.exit_triggers, mode=curr_phase.trigger_mode): 
            return self.switch_phase(curr_phase, default, ctx)

        candidates = [phase for name, phase in self.phases.items()
                      if name != ctx.sim.current_phase 
                      and self.check_triggers(ctx=ctx, triggers=phase.triggers, mode=phase.trigger_mode)]

        if not candidates:
            return None

        #SAFETYVALVE: incase of 2 phases' triggerconditions met at once, raise warning
        if len(candidates) > 1:
            logger.warning("Multiple phases triggered at runstep %s: %s. Taking '%s' (declaration order).",
                           ctx.sim.current_runstep, [p.phase_type for p in candidates], candidates[0].phase_type) 

        return self.switch_phase(curr_phase, candidates[0], ctx)