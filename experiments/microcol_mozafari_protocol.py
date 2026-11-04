#microcol_mozafari_protocol - PROTOCOL of the layered microcolumn experiment, variant B: Mozafari, Ganjtabesh, Nowzari-Dalini, Thorpe & Masquelier
# (2019), Pattern Recognition 94:87-95, as in the first author's SpykeTorch code (MozafariDeep.py). It is variant A's protocol (LayerwiseProtocol,
# microcol_hlsp_protocol.py) plus the paper's third stage and its decision readout.
#
#STAGES:
#   1, 2   STDP, one layer at a time (inherited train_stages): L1 2 epochs, L2 4 epochs (MozafariDeep.py). Before every sample the layer's learning
#          rate is set by its sample count (MozafariSTDP.schedule: a+ doubles every 500 samples up to 0.15)
#   3      R-STDP on the decision projection, rstdp_epochs passes over the fit samples, same order every pass (the authors' loader: shuffle False).
#          Per sample: the decision (below) -> reward if right, punishment if wrong, nothing if silent (MozafariSTDP.apply_reward). After every
#          rstdp_batch samples (counted across passes): reward x the batch's MISS fraction, punishment x its HIT fraction (the paper's adaptive
#          rates). After every pass the validation decisions are scored; select_best_val keeps the weights of the best pass (the authors keep
#          the best epoch on the TEST set; here the validation block takes that role, the test set is used once, at the end)
#DECISION (paper Sec. 2; sf.fire without threshold + sf.get_k_winners(k=1)): the decision group never fires; at the end of the sample its neuron
# with the highest potential wins (capture.membrane), class = neuron index // n_per_class (20 per class); no potential above 0 = silent.
#EVALUATION: variant A's readout-free scores on cluster_groups (L1, L2) and the input control, the same scores on the decision group's
# potentials ("L3_pot"), and the decision accuracy on val and test (val_decision_acc = the search metric).
#STAGE NUMBERS (resume): STDP passes 0 .. P-1 (P = sum(stage_epochs)), R-STDP pass r = P + 2r, its validation P + 2r + 1, final val / test responses
# P + 2 rstdp_epochs (+1). STATE (proto__*): variant A's + Pot_val / Pot_test, Phi, Batch, Train_perf, Val_perf, Best, Best_W (the weights of
# the decision projection only: the other layers are frozen during R-STDP, so nothing else can differ between passes).
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import capture
from runlog import fields
from microcol_hlsp_protocol import LayerwiseCfg, LayerwiseProtocol

logger = logging.getLogger("microcol")
results_log = logging.getLogger("results")      ##-> results.jsonl (one "rstdp" line per pass: the number optuna_sweep's pruner reads)


@dataclass(frozen=True, slots=True)
class MozafariCfg(LayerwiseCfg):
    """LayerwiseCfg with Mozafari et al.'s defaults (one step = one latency bin, 15 bins + 2 flush steps) +
    Args:
        rstdp_learn = the synapse groups trained by R-STDP (the decision projection); rstdp_epochs = passes (the authors ran 680)
        rstdp_batch = samples per adaptive-rate update (the authors' batch: 1000)
        decision_group, n_per_class = the decision layer and its neurons per class (200 = 20 x 10)
        phi_init = (reward, punishment) factors before the first batch. MozafariDeep.py computes 0.9 / 0.1 but never applies them, so
                   (1, 1) is what ran
        select_best_val = keep the weights of the R-STDP pass with the best validation decisions
    """
    protocol_type: str = "MOZAFARI"
    t_train_ms: float = 17.0
    t_state_ms: float = 17.0
    stage_learn: Tuple[Tuple[str, ...], ...] = (("in_L1",), ("L1_L2",))
    stage_epochs: Tuple[int, ...] = (2, 4)
    cluster_groups: Tuple[str, ...] = ("L1_exc", "L2_exc")
    rstdp_learn: Tuple[str, ...] = ("L2_L3",)
    rstdp_epochs: int = 20
    rstdp_batch: int = 1000
    decision_group: str = "L3_exc"
    n_per_class: int = 20
    phi_init: Tuple[float, float] = (1.0, 1.0)
    select_best_val: bool = True


class MozafariProtocol(LayerwiseProtocol):
    """STDP stages, the R-STDP stage, then variant A's scores + the decision accuracy (header). Needs test_data in PhasicNet.train."""
    protocol_type = "MOZAFARI"
    count_dtype = np.uint8          ##one spike per neuron per sample
    paper_reference = ("Mozafari et al. 2019 (convolutional, shared weights, 680 R-STDP epochs, best epoch chosen on the test set): 97.2% MNIST. "
                       "The microcolumn variant (no weight sharing) has no published counterpart.")

    def __init__(self, cfg: MozafariCfg | None, run, *, out_dir=None):
        super().__init__(MozafariCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c, net = self.cfg, run.ctx.net
        if c.decision_group not in net.neuronrows or not set(c.rstdp_learn) <= set(net.synapsespecs):
            raise ValueError(f"Unknown decision_group {c.decision_group!r} or rstdp_learn {c.rstdp_learn}.")
        self._rows = net.neuronrows[c.decision_group]
        n_dec = self._rows.stop - self._rows.start
        if n_dec % c.n_per_class:
            raise ValueError(f"{c.decision_group} has {n_dec} neurons, not a multiple of n_per_class = {c.n_per_class}.")
        self._P = int(sum(c.stage_epochs))
        self._first_eval_stage = self._P + 2 * int(c.rstdp_epochs)
        self.Pot_val = np.zeros((len(self._val_set), n_dec), dtype=np.float32)
        self.Pot_test = np.zeros((len(run.test_set), n_dec), dtype=np.float32)
        self.Phi = np.array(c.phi_init, dtype=np.float64)               ##(reward, punishment) factors
        self.Batch = np.zeros(3, dtype=np.int64)                         ##(hit, miss, silent) in the current batch
        self.Train_perf = np.zeros((int(c.rstdp_epochs), 3))             ##per R-STDP pass: (hit, miss, silent) fractions
        self.Val_perf = np.zeros((int(c.rstdp_epochs), 3))
        self.Best = np.array([-1.0, -1.0])                               ##(best validation hit fraction, its pass)
        self._best_k = run.ctx.syn.weights.index.synIDs_by_post(np.arange(self._rows.start, self._rows.stop))      ##the decision projection's edges
        self.Best_W = run.ctx.syn.W[self._best_k].copy() if c.select_best_val else np.zeros(0, dtype=np.float32)

    def _n_passes(self) -> int:
        return int(sum(self.cfg.stage_epochs)) + int(self.cfg.rstdp_epochs)

    # ------------------------------------------------------------------------- STDP stages
    def learn_step(self, s: int, e: int):
        """Variant A's step with the layer's learning rate set first (its j-th sample over all its epochs)."""
        base, n, plasticity = super().learn_step(s, e), len(self._fit_set), self.run.plasticity

        def step(i, frames, target):
            plasticity.schedule(e * n + i)
            base(i, frames, target)
        return step

    # ------------------------------------------------------------------------- decision
    def decide(self, pot: np.ndarray) -> tuple[int, int]:
        """ONE sample's decision group potentials -> (winner index in the group, class); (-1, -1) = silent."""
        w = int(np.argmax(pot))
        return (w, w // self.cfg.n_per_class) if pot[w] > 0 else (-1, -1)

    def decision_perf(self, Pot: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """(hit, miss, silent) fractions of the decisions of a response set."""
        if not len(Y):
            return np.zeros(3)
        d = np.where(Pot.max(axis=1) > 0, Pot.argmax(axis=1) // self.cfg.n_per_class, -1)
        return np.array([np.mean(d == Y), np.mean((d >= 0) & (d != Y)), np.mean(d < 0)])

    # ------------------------------------------------------------------------- R-STDP stage
    def rstdp_step(self, r: int):
        """ONE fit sample of R-STDP pass r: present, decide, reward / punish the winner; the adaptive factors after every rstdp_batch samples."""
        run, c, n = self.run, self.cfg, len(self._fit_set)
        plasticity, every, pass_k = run.plasticity, int(c.trajectory_every or 0), self._P + r

        def step(i, frames, target):
            if every and i % every == 0:
                self.record_trajectory(pass_k, i)
            self.present(frames[:self._n_tr], target)
            w, d = self.decide(capture.membrane(run.ctx, c.decision_group))
            j = 2 if d < 0 else (0 if d == int(target) else 1)
            if j < 2:
                plasticity.apply_reward(run.ctx, self._rows.start + w, j == 0, self.Phi[j])
            self.Batch[j] += 1
            self.Train_perf[r, j] += 1.0 / n
            if self.Batch.sum() == c.rstdp_batch:                       ##batches run on across passes: no short batch at a pass's end
                self.Phi[:] = self.Batch[1] / c.rstdp_batch, self.Batch[0] / c.rstdp_batch     ##reward x miss fraction, punishment x hit fraction
                self.Batch[:] = 0
        return step

    def train_rstdp(self) -> bool:
        """The R-STDP passes, each followed by its validation pass (stage numbers in the header). Returns True if any pass ran."""
        run, c, ops = self.run, self.cfg, self.ops
        ran = False
        for r in range(int(c.rstdp_epochs)):
            k = self._P + 2 * r
            run.set_mode("train")
            run.set_frozen(self.frozen_for(c.rstdp_learn))
            if r == 0 and ops.starting(k):
                run.plasticity.bound_decision(run.ctx)
            ran |= ops.for_samples(self._fit_set, self.rstdp_step(r), stage=k, name=f"R-e{r}", reset=True,
                                   extra=lambda lo, hi, r=r: f"hit {100 * self.Train_perf[r, 0] * len(self._fit_set) / hi:.1f}% so far, "
                                                             f"factors (reward, punishment) {np.round(self.Phi, 3).tolist()}")
            run.set_mode("eval")
            self.record(self._val_set, self.Resp_val, self.Y_val, self.Pot_val, stage=k + 1, name=f"val-e{r}", end=lambda r=r: self._validated(r))
        if c.select_best_val and self.Best[1] >= 0 and ops.starting(self._first_eval_stage):
            run.set_mode("train")
            run.ctx.syn.W[self._best_k] = self.Best_W
            logger.info("R-STDP: weights of pass %d restored (validation hit %.2f%%)", int(self.Best[1]), 100 * self.Best[0])
        return ran

    def _validated(self, r: int) -> None:
        """The end of validation pass r (inside the pass, so a snapshot holds it): its scores, the best pass's weights."""
        self.Val_perf[r] = self.decision_perf(self.Pot_val, self.Y_val)
        if self.cfg.select_best_val and self.Val_perf[r, 0] >= self.Best[0]:       ##>= as the authors' best_test[0] <= perf_test[0]
            self.Best[:] = self.Val_perf[r, 0], r
            np.copyto(self.Best_W, self.run.ctx.syn.W[self._best_k])
        logger.info("R-STDP pass %d: train hit %.2f%%, validation hit %.2f%% (silent %.2f%%)", r, 100 * self.Train_perf[r, 0],
                    100 * self.Val_perf[r, 0], 100 * self.Val_perf[r, 2])
        results_log.info("rstdp", extra=fields({"rstdp_pass": r, "train_acc": 100 * self.Train_perf[r, 0], "val_decision_acc": 100 * self.Val_perf[r, 0],
                                                "val_decision_silent": 100 * self.Val_perf[r, 2]}))      ##optuna pruning: report=("rstdp", "val_decision_acc")

    def train_all(self) -> None:
        super().train_all()
        if self.train_rstdp():
            self.ops.save_model("decision layer trained (R-STDP)")

    # ------------------------------------------------------------------------- responses
    def record(self, data, Resp, Y, Pot, *, stage: int, name: str, end=None) -> bool:
        """ONE pass (after a reset per sample): Resp[i] = the cluster groups' counts (state_vector), Pot[i] = the decision group's potentials at the
        end, Y[i] = the label; end() after the last sample."""
        run, dec, n = self.run, self.cfg.decision_group, len(data)

        def handle(i, frames, target):
            Resp[i] = self.state_vector(frames, target)
            Pot[i] = capture.membrane(run.ctx, dec)
            Y[i] = int(target)
            if end is not None and i + 1 == n:
                end()
        ran = self.ops.for_samples(data, handle, stage=stage, name=name, reset=True)
        run.check_persistent_unchanged()
        return ran

    def record_responses(self, stage: int) -> None:
        run = self.run
        run.set_mode("eval")
        self.record(self._val_set, self.Resp_val, self.Y_val, self.Pot_val, stage=stage, name="states-val")
        self.record(run.test_set, self.Resp_test, self.Y_test, self.Pot_test, stage=stage + 1, name="states-test")

    # ------------------------------------------------------------------------- scores
    def scores(self) -> dict:
        res = super().scores()
        n_classes = int(max(self.Y_val.max(initial=0), self.Y_test.max(initial=0)) + 1)
        res.update(self.group_scores("L3_pot", self.Pot_val, self.Pot_test, n_classes))
        for split, Pot, Y in (("val", self.Pot_val, self.Y_val), ("test", self.Pot_test, self.Y_test)):
            hit, _, silent = self.decision_perf(Pot, Y)
            res[f"{split}_decision_acc"], res[f"{split}_decision_silent"] = float(100 * hit), float(100 * silent)
        res["rstdp_train_acc"] = (100 * self.Train_perf[:, 0]).tolist()
        res["rstdp_val_acc"] = (100 * self.Val_perf[:, 0]).tolist()
        res["rstdp_best_pass"] = int(self.Best[1])
        net = self.run.ctx.net
        for syn in self.cfg.rstdp_learn:
            res[f"w_mean_{syn}"] = capture.sum_groupweight(self.run.ctx, syn) / net.synapse_counts[f"n_{syn}"]
        return res

    def result_files(self) -> dict:
        return dict(super().result_files(), pot_val=self.Pot_val, pot_test=self.Pot_test)

    def summary(self, res: dict) -> tuple:
        return (f"  decision    : val {res['val_decision_acc']:.2f}%  test {res['test_decision_acc']:.2f}%  (silent {res['test_decision_silent']:.2f}%), "
                f"best R-STDP pass {res['rstdp_best_pass']}",) + super().summary(res)
