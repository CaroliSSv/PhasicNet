#microcol_hlsp_protocol - PROTOCOL of the layered microcolumn experiments: train one layer after the other, then score every layer without a
# trained readout. Variant A (HLSP values) runs it as is; variant B (Mozafari et al. 2019) subclasses it in microcol_mozafari_protocol.py.
#
#STAGES (layer-wise training as in Kheradpisheh et al. 2018, Neural Networks 99:56, and Mozafari et al. 2019, Pattern Recognition 94:87):
#   stage s: ONLY the synapse groups in stage_learn[s] are plastic (Run.set_frozen on all others), stage_epochs[s] passes over the FIT samples
#   (the training set minus the validation block), t_train_ms per sample from a reset (HLSP Algorithm 1: 50 ms, state reset before each sample).
#   Lower layers stay frozen while a higher one learns; groups in no stage never learn (the input projection, a fixed WTA).
#EVALUATION (mode "eval", W guarded): every validation and test sample t_state_ms from a reset (HLSP: 250 ms). Response = the spike counts of each
# group in cluster_groups (ops.responses + ops.binned_counts). Per group, on val and on test:
#   readout-free   k-means on the SAMPLES (k = number of classes, L2-normalised vectors), clusters vs labels: ARI, AMI (scorer.cluster_scores);
#                  Calinski-Harabasz on the labels (scorer.calinski_harabasz)
#   label vote     neuron labels assigned on VAL, test predicted by the vote (scorer.vote_accuracy). Uses labels, so it is a readout (D&C's)
#   input control  the same scores on the INPUT spike counts of the same samples: what the network adds is the difference
#   val_ami        the deepest group's validation AMI: the search metric (no labels shape the clusters, the test set is never touched)
#TRAJECTORY (trajectory_every): every this many fit samples of every training pass, and once after training, the in-degree, out-degree, in-strength
# and out-strength of EVERY neuron (capture.in_degree / out_degree / degree_strength, |W| > 0) go into Traj (responses.npz: traj, traj_at); per group
# their mean and 10 / 50 / 90 percentiles go into results.json ("trajectory") - how the wiring evolves over the run.
#STATE (proto__*): Resp_val / Resp_test (counts per cluster group, concatenated), Y_val / Y_test, Rate (initial Hz per group per stage), Traj / Traj_at;
# progress {"stage", "i"}: resumable with CheckpointCfg.every + out_dir.
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from runcfg import ProtocolCfg
from protocols import Protocol
from scorer import cluster_scores, calinski_harabasz, vote_accuracy
import capture

logger = logging.getLogger("microcol")


@dataclass(frozen=True, slots=True)
class LayerwiseCfg(ProtocolCfg):
    """Hashed into run_id with the rest of RunCfg (report_every excluded).
    Args:
        t_train_ms, t_state_ms = presentation per sample in a training stage / for the responses (HLSP: 50 / 250 ms)
        stage_learn = per stage, the synapse groups that learn (everything else is frozen in that stage)
        stage_epochs = per stage, the passes over the fit samples
        cluster_groups = the groups whose responses are recorded and scored (the deepest last); vote_group = the group voted on (None = the last)
        n_init = k-means restarts; input_control = also score the input spike counts
        val_split = the validation block of the training set (HLSP Sec. 4.1: 1/6), split off before any training (ops.partition)
        rate_probe = the first this-many samples of each stage log the mean rate per cluster group (calibration)
        trajectory_every = fit samples between two records of every neuron's degrees and strengths (header); None = off
    """
    protocol_type: str = "LAYERWISE"
    t_train_ms: float = 50.0
    t_state_ms: float = 250.0
    stage_learn: Tuple[Tuple[str, ...], ...] = ()
    stage_epochs: Tuple[int, ...] = ()
    cluster_groups: Tuple[str, ...] = ()
    vote_group: Optional[str] = None
    n_init: int = 10
    input_control: bool = True
    val_split: float | None = 1.0 / 6.0
    report_every: int = 500
    rate_probe: int = 200
    trajectory_every: Optional[int] = 5000


class LayerwiseProtocol(Protocol):
    """Layer-wise training stages, then val / test responses, then readout-free scores per group (header). Needs test_data in PhasicNet.train."""
    protocol_type = "LAYERWISE"
    needs_test_set = True
    count_dtype = np.int16

    def __init__(self, cfg: LayerwiseCfg | None, run, *, out_dir=None):
        super().__init__(LayerwiseCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c, net = self.cfg, run.ctx.net
        if len(c.stage_learn) != len(c.stage_epochs):
            raise ValueError(f"stage_learn ({len(c.stage_learn)} stages) and stage_epochs ({len(c.stage_epochs)}) must have the same length.")
        unknown = sorted({g for learn in c.stage_learn for g in learn} - set(net.synapsespecs)) + sorted(set(c.cluster_groups) - set(net.neuronslices))
        if unknown or not c.cluster_groups:
            raise ValueError(f"Unknown groups {unknown} in stage_learn / cluster_groups, or no cluster_groups. Synapses {tuple(net.synapsespecs)}, neurons {tuple(net.neuronslices)}.")
        dt = float(run.history.dt)
        self._n_tr = int(round(c.t_train_ms / dt)); self._n_st = int(round(c.t_state_ms / dt))
        (self._fit_set, val), = self.ops.partition(val_split=c.val_split or None)
        self._val_set = run.train_set.subset(np.zeros(0, dtype=np.int64), "val") if val is None else val
        sizes = [net.neuronslices[g].stop - net.neuronslices[g].start for g in c.cluster_groups]
        edges = np.concatenate(([0], np.cumsum(sizes)))
        self._bands = {g: slice(int(a), int(b)) for g, a, b in zip(c.cluster_groups, edges[:-1], edges[1:])}     ##each group's columns in a Resp row
        self._vote = c.vote_group or c.cluster_groups[-1]
        self._first_eval_stage = int(sum(c.stage_epochs))
        n_val, n_test = len(self._val_set), len(run.test_set)
        self.Resp_val = np.zeros((n_val, int(edges[-1])), dtype=self.count_dtype)
        self.Resp_test = np.zeros((n_test, int(edges[-1])), dtype=self.count_dtype)
        self.Y_val = np.full(n_val, -1, dtype=np.int64)
        self.Y_test = np.full(n_test, -1, dtype=np.int64)
        self.Rate = np.zeros((len(c.stage_learn), len(c.cluster_groups)))
        every = int(c.trajectory_every or 0)
        self._per_pass = -(-len(self._fit_set) // every) if every else 0                          ##records per training pass
        n_points = self._n_passes() * self._per_pass + 1 if every else 0
        self.Traj = np.zeros((n_points, 4, net.n_total), dtype=np.float32)                       ##(point, measure, neuron); header
        self.Traj_at = np.full((n_points, 3), -1, dtype=np.int64)                                 ##(pass, sample, runstep) of each point

    TRAJ_MEASURES = ("in_degree", "out_degree", "in_strength", "out_strength")

    def _n_passes(self) -> int:
        """Training passes over the fit set (variant B adds its R-STDP passes)."""
        return int(sum(self.cfg.stage_epochs))

    def record_trajectory(self, pass_k: int, i: int, *, final: bool = False) -> None:
        """One Traj point: the four measures of every neuron now (in-measures of the inputs are 0). final = the point after all training."""
        if not self.Traj.shape[0]:
            return
        ctx, n0 = self.run.ctx, self.run.ctx.net.n_input
        idx = self.Traj.shape[0] - 1 if final else pass_k * self._per_pass + i // int(self.cfg.trajectory_every)
        self.Traj[idx, 0, n0:] = capture.in_degree(ctx)
        self.Traj[idx, 1] = capture.out_degree(ctx)
        self.Traj[idx, 2, n0:] = capture.degree_strength(ctx, direction="in", signed=False)
        self.Traj[idx, 3] = capture.degree_strength(ctx, direction="out", signed=False)
        self.Traj_at[idx] = pass_k, i, ctx.sim.current_runstep

    def trajectory_summary(self) -> dict:
        """results.json "trajectory": per neuron group and measure, the mean and the 10 / 50 / 90 percentiles over the group at every point."""
        if not self.Traj.shape[0]:
            return {}
        net, done = self.run.ctx.net, self.Traj_at[:, 0] >= 0
        out = {"every": self.cfg.trajectory_every, "measures": list(self.TRAJ_MEASURES), "at": self.Traj_at[done].tolist()}
        for g, s in net.neuronslices.items():
            block = self.Traj[done][:, :, s]                                                      ##(points, measure, neurons of g)
            out[g] = {m: {"mean": block[:, j].mean(axis=1).tolist(), **{f"p{q}": np.percentile(block[:, j], q, axis=1).tolist() for q in (10, 50, 90)}}
                      for j, m in enumerate(self.TRAJ_MEASURES)}
        return out

    # ------------------------------------------------------------------------- training
    def frozen_for(self, learn) -> tuple:
        """Everything but `learn` (the freeze list Run.set_frozen takes)."""
        return tuple(n for n in self.run.ctx.net.synapsespecs if n not in learn)

    def train_stages(self) -> bool:
        """Every stage: Run.set_frozen(all but stage_learn[s]), then stage_epochs[s] resumable passes over the fit set (stage numbers 0, 1, ... in
        order). Returns True if any pass ran (False: a resumed run is already past them)."""
        run, c = self.run, self.cfg
        run.set_mode("train")
        ran, k = False, 0
        for s, learn in enumerate(c.stage_learn):
            run.set_frozen(self.frozen_for(learn))
            for e in range(int(c.stage_epochs[s])):
                ran |= self.ops.for_samples(self._fit_set, self.learn_step(s, e), stage=k, name=f"L{s + 1}-e{e}", reset=True, extra=self._tail)
                k += 1
        return ran

    def learn_step(self, s: int, e: int):
        """ONE fit sample of stage s, epoch e: its first t_train_ms with plasticity on (after the reset of for_samples). The first rate_probe samples
        of the stage's first epoch give the initial rate per cluster group (logged once)."""
        c, dt, net = self.cfg, float(self.run.history.dt), self.run.ctx.net
        n_probe, every, pass_k = min(c.rate_probe, len(self._fit_set)), int(c.trajectory_every or 0), int(sum(c.stage_epochs[:s])) + e

        def step(i, frames, target):
            if every and i % every == 0:
                self.record_trajectory(pass_k, i)
            counts = self.present(frames[:self._n_tr], target)
            if e == 0 and i < n_probe:
                self.Rate[s] += [capture.rate_hz(counts[net.neuronslices[g]], b.stop - b.start, self._n_tr, dt) / n_probe for g, b in self._bands.items()]
                if i + 1 == n_probe:
                    logger.info("stage %d: initial rate (Hz) over %d samples: %s", s + 1, i + 1, {g: round(float(r), 2) for g, r in zip(self._bands, self.Rate[s])})
        return step

    def _tail(self, lo, hi):
        """capture: mean |W| per realized synapse and the fraction at a bound (the progress line of a training pass)."""
        run = self.run
        return (f"|W| mean {capture.mean_globalweight(run.ctx, signed=False):.3f}  at bound "
                f"{100 * capture.pinned_weights(run.ctx, w_min=run.plasticity.w_min, w_max=run.plasticity.w_max)['saturated_frac']:.1f}%")

    # ------------------------------------------------------------------------- responses
    def state_vector(self, frames: np.ndarray, target, key: str | None = None) -> np.ndarray:
        """ONE sample (after the reset of ops.responses): t_state_ms from rest -> the counts of the cluster groups, concatenated in their order."""
        return self.ops.binned_counts(frames[:self._n_st], target, [self._n_st], group=tuple(self.cfg.cluster_groups))[0]

    def record_responses(self, stage: int) -> None:
        """The val and test passes (mode "eval", stages `stage` and `stage` + 1)."""
        run, ops = self.run, self.ops
        run.set_mode("eval")
        ops.responses(self._val_set, self.state_vector, self.Resp_val, self.Y_val, stage=stage, name="states-val", reset=True)
        ops.responses(run.test_set, self.state_vector, self.Resp_test, self.Y_test, stage=stage + 1, name="states-test", reset=True)
        run.check_persistent_unchanged()

    def input_counts(self, data) -> np.ndarray:
        """The input control: each sample's INPUT spike counts over the same t_state_ms (the frames themselves; nothing is simulated)."""
        return np.stack([np.asarray(data[i][0][:self._n_st]).sum(axis=0) for i in range(len(data))]).astype(np.float32)

    # ------------------------------------------------------------------------- scores
    def group_scores(self, name: str, val: np.ndarray, test: np.ndarray, n_classes: int) -> dict:
        """The readout-free scores of ONE response set (val and test), the test CH and the vote (assigned on val, scored on test)."""
        c, out = self.cfg, {}
        for split, X, Y in (("val", val, self.Y_val), ("test", test, self.Y_test)):
            if len(Y):
                sc = cluster_scores(X, Y, rng=self.rng.rng("protocol", f"kmeans:{split}:{name}"), k=n_classes, n_init=c.n_init)
                out.update({f"{split}_ari_{name}": sc["ari"], f"{split}_ami_{name}": sc["ami"], f"{split}_silent_{name}": sc["silent_frac"]})
        out[f"test_ch_{name}"] = calinski_harabasz(test, self.Y_test)
        if len(self.Y_val):
            out[f"test_vote_acc_{name}"] = vote_accuracy(val, self.Y_val, test, self.Y_test, n_classes)
        return out

    def scores(self) -> dict:
        run, c = self.run, self.cfg
        n_classes = int(max(self.Y_val.max(initial=0), self.Y_test.max(initial=0)) + 1)
        res = {}
        for g, b in self._bands.items():
            res.update(self.group_scores(g, self.Resp_val[:, b], self.Resp_test[:, b], n_classes))
            res[f"rate_hz_{g}"] = capture.rate_hz(self.Resp_test[:, b], len(self.Y_test) * (b.stop - b.start), self._n_st, float(run.history.dt))
        if c.input_control:
            res.update(self.group_scores("input", self.input_counts(self._val_set), self.input_counts(run.test_set), n_classes))
        res["val_ami"] = res.get(f"val_ami_{self._vote}")
        res["test_ami"] = res.get(f"test_ami_{self._vote}")
        res["test_vote_acc"] = res.get(f"test_vote_acc_{self._vote}")
        net = run.ctx.net
        for learn in c.stage_learn:     ##mean weight per realized synapse of every group that learned
            for syn in learn:
                n = net.synapse_counts[f"n_{syn}"]
                res[f"w_mean_{syn}"] = capture.sum_groupweight(run.ctx, syn) / n if n else None
        return res

    # ------------------------------------------------------------------------- the whole protocol
    paper_reference = ("HLSP-LSM (one liquid, N = 1000, softmax readout): 95.63% MNIST (Luo et al. 2026, Table 5). "
                       "The layered microcolumn variant has no published counterpart.")

    def train_all(self) -> None:
        """Every training pass, then the trained model saved (variant B adds its R-STDP stage here)."""
        if self.train_stages():
            self.ops.save_model("layer-wise trained model")

    def result_files(self) -> dict:
        return dict(resp_val=self.Resp_val, y_val=self.Y_val, resp_test=self.Resp_test, y_test=self.Y_test, W=self.run.ctx.syn.W,
                    traj=self.Traj, traj_at=self.Traj_at)

    def summary(self, res: dict) -> tuple:
        g = self._vote
        return (f"  {g:<12}: val AMI {res['val_ami']:.3f}  test AMI {res['test_ami']:.3f}  ARI {res.get(f'test_ari_{g}', float('nan')):.3f}  "
                f"vote {res['test_vote_acc'] if res['test_vote_acc'] is None else round(res['test_vote_acc'], 2)}%",
                f"  input       : test AMI {res.get('test_ami_input', float('nan')):.3f}  vote {res.get('test_vote_acc_input')}",
                "  rates (Hz)  : " + ", ".join(f"{k} {res[f'rate_hz_{k}']:.2f}" for k in self._bands))

    def execute(self) -> dict:
        run, c, ops = self.run, self.cfg, self.ops
        ops.header(stages=len(c.stage_learn), fit=len(self._fit_set), val=len(self._val_set), test=len(run.test_set))
        self.train_all()
        if ops.starting(self._first_eval_stage):
            self.record_trajectory(0, 0, final=True)                                          ##the trained wiring (after variant B's restore)
        self.record_responses(self._first_eval_stage)
        res = self.scores()
        res.update({"n_fit": len(self._fit_set), "n_val": len(self._val_set), "n_test": len(run.test_set),
                    "bands": {g: [b.start, b.stop] for g, b in self._bands.items()}, "initial_rate_hz": self.Rate.tolist(),
                    "trajectory": self.trajectory_summary(), "paper_reference": self.paper_reference})
        return ops.finish(res, files=self.result_files(), lines=self.summary(res))
