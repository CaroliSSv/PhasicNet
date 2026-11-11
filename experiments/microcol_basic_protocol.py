#microcol_basic_protocol - PROTOCOL of the back-to-basics microcolumn experiment: variant B's protocol (MozafariProtocol: STDP one layer at a time,
# then R-STDP on the decision layer with a validation pass after every R-STDP pass) on spiking LIF layers, plus the records of a development sweep.
#
#DIFFERENT FROM MozafariProtocol:
#   decision     the decision group spikes. Class of a decision neuron = its index % n_classes (so every column holds every class). The decided
#                class = the class whose neurons fired most over the sample (no spike = silent; a tie = the lowest class, counted in
#                decision_tie_frac). Reward / punishment goes to ALL neurons of the decided class (SoftBoundSTDP.apply_reward).
#   STDP stages  no learning-rate schedule; SoftBoundSTDP.normalize_weights after every sample (cfg normalize).
#RECORDS (results.json; every plain number also becomes an Optuna user attribute, optuna_sweep):
#   untrained    BEFORE any training the validation set is presented once (Resp_untr, responses.npz resp_untr): val_ami_<group>_untrained and
#                val_ami_gain_<group> = trained - untrained (the same samples, the same clustering)
#   rates        rate_hz_<group>_untrained (validation set, as built), rate_hz_<group>_val, rate_hz_<group> (test set) for EVERY neuron group,
#                the inhibitory groups and the input too (Rate_all)
#   health       every health_every fit samples of every training pass ("health" in results.json; one "health" line each in results.jsonl):
#                per neuron group the mean rate (Hz), the share of its neurons that did not fire and the highest single-neuron rate in the window;
#                per learned synapse group the mean, sd, the shares within 2 % of w_min / of w_max, and the mean |change| since the last record
#   end          per cluster group: dead_frac_<g> (neurons silent on every test sample), max_rate_hz_<g>, active_frac_<g> (mean share of the
#                group's neurons that fire in a test sample), count_cv_<g> (sd / mean of the group's spike count over the test samples),
#                rate_hz_<g>_cols (per column) and input_rate_hz_cols; rstdp_val_acc_p<r> per R-STDP pass;
#                per learned group: w_sd, w_at_min, w_at_max, w_moved (mean |W - W as built| / the range), w_hist (20 bins);
#                the decision confusion matrix, decision_top_share (largest share of one decided class on test: 0.1 = balanced, 1 = one class
#                always), decision_tie_frac; sec_per_sample_train / _eval (this process only: a resumed run counts its last part);
#                flag_silent / flag_runaway (a cluster group under rate_low_hz / over rate_high_hz on test), flag_runaway_inh (another group, not
#                the input, over rate_high_hz on test), flag_no_learning (every w_moved < 0.01)
#STATE (proto__*): MozafariProtocol's + Resp_untr, Rate_all, Health / Wstat / Health_at, Acc (spike counts of the running health window), W_prev.
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from runlog import fields
from scorer import cluster_scores, confusion_matrix
from microcol_hlsp_protocol import LayerwiseProtocol
from microcol_mozafari_protocol import MozafariCfg, MozafariProtocol

logger = logging.getLogger("microcol")
results_log = logging.getLogger("results")


@dataclass(frozen=True, slots=True)
class BasicCfg(MozafariCfg):
    """MozafariCfg with the back-to-basics defaults (n_per_class = decision neurons / n_classes) +
    Args:
        n_classes = classes of the decision group (header: class = index % n_classes)
        health_every = fit samples between two health records (header); None = off
        rate_low_hz, rate_high_hz = the limits of flag_silent / flag_runaway
    """
    protocol_type: str = "BASIC"
    t_train_ms: float = 100.0
    t_state_ms: float = 100.0
    stage_learn: Tuple[Tuple[str, ...], ...] = (("in_L1",), ("L1_L2",))
    stage_epochs: Tuple[int, ...] = (1, 1)
    cluster_groups: Tuple[str, ...] = ("L1_exc", "L2_exc", "L3_exc")
    rstdp_epochs: int = 3
    rstdp_batch: int = 500
    n_per_class: int = 24
    trajectory_every: Optional[int] = 1000
    n_classes: int = 10
    health_every: Optional[int] = 250
    rate_low_hz: float = 0.5
    rate_high_hz: float = 100.0


class BasicProtocol(MozafariProtocol):
    """Untrained validation responses, STDP stages, the R-STDP stage, then the scores and the sweep records (header). Needs test_data."""
    protocol_type = "BASIC"
    count_dtype = np.int16
    paper_reference = "Own design (back to basics, Nov 2026): no published counterpart."
    HEALTH_NEURON = ("rate_hz", "silent_frac", "max_rate_hz")
    HEALTH_WEIGHT = ("mean", "sd", "at_min", "at_max", "change")

    def __init__(self, cfg: BasicCfg | None, run, *, out_dir=None):
        super().__init__(BasicCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c, ctx = self.cfg, run.ctx
        net, syn = ctx.net, ctx.syn
        n_dec = self._rows.stop - self._rows.start
        if c.decision_group not in c.cluster_groups or n_dec % c.n_classes:
            raise ValueError(f"decision_group {c.decision_group!r} must be one of cluster_groups {c.cluster_groups} and have a multiple of "
                             f"n_classes = {c.n_classes} neurons (it has {n_dec}).")
        self._class = np.arange(n_dec) % c.n_classes                                      ##class of each decision neuron
        self._learned = tuple(dict.fromkeys([g for learn in c.stage_learn for g in learn] + list(c.rstdp_learn)))
        self._edges = {}                                                                  ##per learned synapse group: its edges in syn.W
        for name in self._learned:
            p = net.synapseslices[name]
            self._edges[name] = np.flatnonzero((syn.post >= p.post.start) & (syn.post < p.post.stop) & (syn.pre >= p.pre.start) & (syn.pre < p.pre.stop))
        self._w_built = syn.W.copy()
        every = int(c.health_every or 0)
        self._h_per_pass = -(-len(self._fit_set) // every) if every else 0
        n_points = self._n_passes() * self._h_per_pass
        self._sel = np.r_[tuple(net.neuronslices[g] for g in c.cluster_groups)]          ##the cluster groups' neurons in a count vector
        self.Resp_untr = np.zeros_like(self.Resp_val)
        self.Rate_all = np.zeros((3, len(net.neuronslices)))                              ##(untrained val, val, test) x neuron group: spikes per neuron, summed over the samples
        self.Health = np.zeros((n_points, len(net.neuronslices), len(self.HEALTH_NEURON)), dtype=np.float32)
        self.Wstat = np.zeros((n_points, len(self._learned), len(self.HEALTH_WEIGHT)), dtype=np.float32)
        self.Health_at = np.full((n_points, 2), -1, dtype=np.int64)                       ##(pass, fit samples done in it) of each record
        self.Acc = np.zeros(net.n_total, dtype=np.float64)
        self.W_prev = syn.W.copy()

    # ------------------------------------------------------------------------- untrained responses
    def record_untrained(self) -> None:
        """The validation set once through the network AS BUILT (mode "eval"), before the first training sample. Not a numbered stage: a run
        interrupted in here starts it again; afterwards Resp_untr is in every snapshot."""
        run = self.run
        run.set_mode("eval")
        for i in range(len(self._val_set)):
            frames, target = self._val_set[i]
            run.reset_transient()
            self.Resp_untr[i] = self.all_counts(frames, target, 0, i)
        run.check_persistent_unchanged()
        run.set_mode("train")
        logger.info("untrained validation responses recorded (%d samples)", len(self._val_set))

    def all_counts(self, frames, target, row: int | None, i: int) -> np.ndarray:
        """state_vector (the cluster groups' counts of ONE sample) that also adds every neuron group's mean count to Rate_all[row] (None = not)."""
        counts = self.present(frames[:self._n_st], target)
        if row is not None:
            if i == 0:
                self.Rate_all[row] = 0.0
            self.Rate_all[row] += [counts[s].mean() for s in self.run.ctx.net.neuronslices.values()]
        return counts[self._sel]

    def train_all(self) -> None:
        if self.ops.starting(0):
            self.record_untrained()
        super().train_all()

    # ------------------------------------------------------------------------- health records
    def present(self, frames, target=None, *, group=None, until=None) -> np.ndarray:
        """Protocol.present; a training sample's counts are also added to the running health window (Acc)."""
        counts = super().present(frames, target, group=group, until=until)
        if group is None and self.run.ctx.sim.mode == "train":
            np.add(self.Acc, counts, out=self.Acc)
        return counts

    def health(self, pass_k: int, i: int) -> None:
        """After fit sample i of training pass pass_k: one health record when a window of health_every samples (or the pass) ends."""
        c, n = self.cfg, len(self._fit_set)
        every = int(c.health_every or 0)
        if not every or ((i + 1) % every and i + 1 != n):
            return
        ctx, plasticity = self.run.ctx, self.run.plasticity
        net, W, idx = ctx.net, ctx.syn.W, pass_k * self._h_per_pass + i // every
        seconds = (i % every + 1) * self._n_tr * float(self.run.history.dt) / 1000.0     ##simulated time of the window
        for j, s in enumerate(net.neuronslices.values()):
            a = self.Acc[s]
            self.Health[idx, j] = a.mean() / seconds, np.mean(a == 0), a.max() / seconds
        lo, hi = float(plasticity.w_min), float(plasticity.w_max)
        for j, k in enumerate(self._edges.values()):
            w = W[k]
            if w.size:
                self.Wstat[idx, j] = w.mean(), w.std(), np.mean(w <= lo + 0.02 * (hi - lo)), np.mean(w >= hi - 0.02 * (hi - lo)), np.abs(w - self.W_prev[k]).mean()
        np.copyto(self.W_prev, W)
        self.Acc[:] = 0.0
        self.Health_at[idx] = pass_k, i + 1
        results_log.info("health", extra=fields({"pass": pass_k, "sample": i + 1,
                                                 **{f"rate_hz_{g}": float(self.Health[idx, j, 0]) for j, g in enumerate(net.neuronslices)},
                                                 **{f"w_mean_{s}": float(self.Wstat[idx, j, 0]) for j, s in enumerate(self._learned)},
                                                 **{f"w_change_{s}": float(self.Wstat[idx, j, 4]) for j, s in enumerate(self._learned)}}))

    def health_summary(self) -> dict:
        """results.json "health": the records in time order, per neuron group and per learned synapse group."""
        done = self.Health_at[:, 0] >= 0
        if not done.any():
            return {}
        H, Wst = self.Health[done], self.Wstat[done]
        return {"every": self.cfg.health_every, "at": self.Health_at[done].tolist(),
                "neurons": {g: {m: H[:, j, q].tolist() for q, m in enumerate(self.HEALTH_NEURON)} for j, g in enumerate(self.run.ctx.net.neuronslices)},
                "weights": {s: {m: Wst[:, j, q].tolist() for q, m in enumerate(self.HEALTH_WEIGHT)} for j, s in enumerate(self._learned)}}

    # ------------------------------------------------------------------------- STDP stages
    def learn_step(self, s: int, e: int):
        """Variant A's step (no learning-rate schedule), then the normalisation and the health record."""
        base, run, pass_k = LayerwiseProtocol.learn_step(self, s, e), self.run, int(sum(self.cfg.stage_epochs[:s])) + e

        def step(i, frames, target):
            base(i, frames, target)
            run.plasticity.normalize_weights(run.ctx)
            self.health(pass_k, i)
        return step

    # ------------------------------------------------------------------------- decision
    def class_counts(self, counts: np.ndarray) -> np.ndarray:
        """(samples, decision neurons) spike counts -> (samples, n_classes): the spikes of each class's neurons."""
        counts = np.atleast_2d(counts)
        out = np.zeros((counts.shape[0], self.cfg.n_classes))
        np.add.at(out.T, self._class, counts.T.astype(np.float64))
        return out

    def decisions(self, counts: np.ndarray) -> np.ndarray:
        """Decided class per sample; -1 = silent."""
        S = self.class_counts(counts)
        return np.where(S.max(axis=1) > 0, S.argmax(axis=1), -1)

    def decision_perf(self, Pot: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """(hit, miss, silent) fractions of the decisions of a response set (Pot = the decision group's spike counts)."""
        if not len(Y):
            return np.zeros(3)
        d = self.decisions(Pot)
        return np.array([np.mean(d == Y), np.mean((d >= 0) & (d != Y)), np.mean(d < 0)])

    # ------------------------------------------------------------------------- R-STDP stage
    def rstdp_step(self, r: int):
        """ONE fit sample of R-STDP pass r: present, decide by spike count, reward / punish the decided class's neurons; the adaptive factors after
        every rstdp_batch samples (as MozafariProtocol)."""
        run, c, n = self.run, self.cfg, len(self._fit_set)
        plasticity, every, pass_k = run.plasticity, int(c.trajectory_every or 0), self._P + r
        band = run.ctx.net.neuronslices[c.decision_group]
        members = [self._rows.start + np.flatnonzero(self._class == k) for k in range(c.n_classes)]     ##W rows of each class's neurons

        def step(i, frames, target):
            if every and i % every == 0:
                self.record_trajectory(pass_k, i)
            d = int(self.decisions(self.present(frames[:self._n_tr], target)[band])[0])
            j = 2 if d < 0 else (0 if d == int(target) else 1)
            if j < 2:
                plasticity.apply_reward(run.ctx, members[d], j == 0, self.Phi[j])
            self.Batch[j] += 1
            self.Train_perf[r, j] += 1.0 / n
            if self.Batch.sum() == c.rstdp_batch:
                self.Phi[:] = self.Batch[1] / c.rstdp_batch, self.Batch[0] / c.rstdp_batch     ##reward x miss fraction, punishment x hit fraction
                self.Batch[:] = 0
            self.health(pass_k, i)
        return step

    # ------------------------------------------------------------------------- responses
    def record(self, data, Resp, Y, Pot, *, stage: int, name: str, end=None) -> bool:
        """ONE pass (after a reset per sample): Resp[i] = the cluster groups' counts, Pot[i] = the decision group's counts, Y[i] = the label."""
        band, n = self._bands[self.cfg.decision_group], len(data)
        row = {"states-val": 1, "states-test": 2}.get(name)                              ##the two final passes also give every group's rate

        def handle(i, frames, target):
            Resp[i] = self.all_counts(frames, target, row, i)
            Pot[i] = Resp[i, band]
            Y[i] = int(target)
            if end is not None and i + 1 == n:
                end()
        ran = self.ops.for_samples(data, handle, stage=stage, name=name, reset=True)
        self.run.check_persistent_unchanged()
        return ran

    # ------------------------------------------------------------------------- scores
    def scores(self) -> dict:
        run, c = self.run, self.cfg
        ctx, net, plasticity = run.ctx, run.ctx.net, run.plasticity
        res = LayerwiseProtocol.scores(self)
        dt, n_test = float(run.history.dt), max(len(self.Y_test), 1)
        seconds = self._n_st * dt / 1000.0                                               ##simulated time of one response
        for g, b in self._bands.items():
            if len(self.Y_val):                                                           ##the untrained network on the same validation samples
                sc = cluster_scores(self.Resp_untr[:, b], self.Y_val, rng=self.rng.rng("protocol", f"kmeans:val:{g}"), k=c.n_classes, n_init=c.n_init)
                res[f"val_ami_{g}_untrained"] = sc["ami"]
                res[f"val_ami_gain_{g}"] = res[f"val_ami_{g}"] - sc["ami"]
            per_neuron = self.Resp_test[:, b].sum(axis=0) / (n_test * seconds)            ##Hz per neuron over the test set
            res[f"dead_frac_{g}"] = float(np.mean(per_neuron == 0))
            res[f"max_rate_hz_{g}"] = float(per_neuron.max())
            if len(self.Y_test):
                total = self.Resp_test[:, b].sum(axis=1, dtype=np.float64)
                res[f"active_frac_{g}"] = float(np.mean(self.Resp_test[:, b] > 0))
                res[f"count_cv_{g}"] = float(total.std() / total.mean()) if total.mean() > 0 else None
            if net.n_columns:
                col = net.column_ids[net.neuronslices[g]]
                res[f"rate_hz_{g}_cols"] = [float(per_neuron[col == k].mean()) for k in range(net.n_columns)]
        for row, tag, n in ((0, "_untrained", len(self.Y_val)), (1, "_val", len(self.Y_val)), (2, "", len(self.Y_test))):
            if n:
                res.update({f"rate_hz_{g}{tag}": float(v / (n * seconds)) for g, v in zip(net.neuronslices, self.Rate_all[row])})
        if c.input_control and net.n_columns:
            per_input = self.input_counts(run.test_set).sum(axis=0) / (n_test * seconds)
            res["input_rate_hz_cols"] = [float(per_input[net.column_ids[:net.n_input] == k].mean()) for k in range(net.n_columns)]
        ##the decision (as MozafariProtocol.scores, on spike counts)
        for split, Pot, Y in (("val", self.Pot_val, self.Y_val), ("test", self.Pot_test, self.Y_test)):
            hit, _, silent = self.decision_perf(Pot, Y)
            res[f"{split}_decision_acc"], res[f"{split}_decision_silent"] = float(100 * hit), float(100 * silent)
        res["rstdp_train_acc"] = (100 * self.Train_perf[:, 0]).tolist()
        res["rstdp_val_acc"] = (100 * self.Val_perf[:, 0]).tolist()
        res["rstdp_best_pass"] = int(self.Best[1])
        res.update({f"rstdp_val_acc_p{r}": float(100 * v) for r, v in enumerate(self.Val_perf[:, 0])})
        if len(self.Y_test):
            d, S = self.decisions(self.Pot_test), np.sort(self.class_counts(self.Pot_test), axis=1)
            res["decision_top_share"] = float(np.bincount(d[d >= 0], minlength=c.n_classes).max() / len(d))
            res["decision_tie_frac"] = float(np.mean((S[:, -1] > 0) & (S[:, -1] == S[:, -2])))
            res["decision_confusion"] = confusion_matrix(self.Y_test[d >= 0], d[d >= 0], c.n_classes).tolist()
        ##the learned weights
        lo, hi = float(plasticity.w_min), float(plasticity.w_max)
        moved = []
        for name, k in self._edges.items():
            w = ctx.syn.W[k]
            if not w.size:
                continue
            res[f"w_mean_{name}"], res[f"w_sd_{name}"] = float(w.mean()), float(w.std())
            res[f"w_at_min_{name}"] = float(np.mean(w <= lo + 0.02 * (hi - lo)))
            res[f"w_at_max_{name}"] = float(np.mean(w >= hi - 0.02 * (hi - lo)))
            res[f"w_moved_{name}"] = float(np.abs(w - self._w_built[k]).mean() / (hi - lo))
            res[f"w_hist_{name}"] = np.histogram(w, bins=20, range=(lo, hi))[0].tolist()
            moved.append(res[f"w_moved_{name}"])
        ##timing and flags
        sec = self.ops.seconds
        train = [v for k, v in sec.items() if k.startswith(("L", "R-"))]
        evals = [v for k, v in sec.items() if k.startswith(("val-", "states-val"))]
        if train:
            res["sec_per_sample_train"] = float(sum(train) / (len(train) * max(len(self._fit_set), 1)))
        if evals:
            res["sec_per_sample_eval"] = float(sum(evals) / (len(evals) * max(len(self._val_set), 1)))
        rates = [res[f"rate_hz_{g}"] for g in self._bands]
        others = [res[f"rate_hz_{g}"] for g, s in net.neuronslices.items() if g not in self._bands and s.start >= net.n_input]
        res["flag_silent"] = int(min(rates) < c.rate_low_hz)
        res["flag_runaway"] = int(max(rates) > c.rate_high_hz)
        res["flag_runaway_inh"] = int(max(others, default=0.0) > c.rate_high_hz)
        res["flag_no_learning"] = int(bool(moved) and max(moved) < 0.01)
        res["health"] = self.health_summary()
        return res

    def result_files(self) -> dict:
        return dict(super().result_files(), resp_untr=self.Resp_untr, health=self.Health, wstat=self.Wstat, health_at=self.Health_at)

    def summary(self, res: dict) -> tuple:
        g = self._vote
        gain = res.get(f"val_ami_gain_{g}")
        return super().summary(res) + (
            f"  untrained   : val AMI {g} {res.get(f'val_ami_{g}_untrained', float('nan')):.3f}  gain {float('nan') if gain is None else gain:+.3f}",
            f"  flags       : silent {res['flag_silent']}  runaway {res['flag_runaway']} (inh {res['flag_runaway_inh']})  no learning {res['flag_no_learning']}  "
            f"top class share {res.get('decision_top_share', float('nan')):.2f}")
