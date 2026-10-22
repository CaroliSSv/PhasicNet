#dc_protocol - Diehl & Cook (2015) as a PhasicNet Protocol: the steps AROUND the network, as the named protocol class RunCfg(protocol=DCProtocolCfg(...))
# picks. The CLI (arguments, MNIST, protocol.json) is dc_reproduce.py; its header documents the protocol's steps (TRAIN / ASSIGN / TEST / REPORT /
# TRAIN ACC) and their sources.
#
#   spike counts     ops.binned_counts(frames, bounds=[n_present, T], group="exc"): the present and the rest window of ONE presentation
#   re-presentation  DCProtocol.present_example(): keyed Poisson draw rng("input", f"{key}:{rep}"), dc_normalise before every TRAINING presentation
#   the input        diehl_cook_2015.DCPoissonImages: the DATASET encodes (variant(i, intensity=, rng=)); the protocol only asks for "example i, again"
#   the examples     ModelInput views: TRAIN / ASSIGN = run.train_set.subset(np.arange(n) % len(train_set)), TEST = run.test_set
#   eval             run.set_mode("eval") + run.check_persistent_unchanged() (Theta is DCAdaptiveThreshold.persistent)
#   new network      diehl_cook_2015.fresh_network (run.reset_transient + V = v_rest - 40), per stage
from __future__ import annotations
import json, logging
from dataclasses import dataclass, field, asdict
import numpy as np
from runcfg import ProtocolCfg
from protocols import Protocol
from scorer import dc_assign_labels, dc_rank_predict, confusion_matrix     ##the authors' exact readout (scorer)
from diehl_cook_2015 import fresh_network, DT_MS, T_PRESENT_MS, T_REST_MS, INPUT_INTENSITY
from diehlcook import dc_normalise                                                             ##the script's normalize_weights()

logger = logging.getLogger("dc_reproduce")       ##same logger name as the CLI: one log


@dataclass(frozen=True, slots=True)
class DCProtocolCfg(ProtocolCfg):
    """The protocol's settings = the CLI's arguments + the script's constants (dc_reproduce.py header). Hashed into run_id with the rest of RunCfg.
    Args:
        n_train, n_assign, test_passes, max_repeats, report_every = the CLI arguments of the same names (defaults = the paper's 100-neuron run); n_train /
            n_assign are PRESENTATION counts over the training set the run holds (example j = image j % len(run.train_set): 40,000 of 60,000, then all
            60,000 once). DataCfg.n_test picks the test examples (stratified); a pass runs the whole picked test set
        min_spikes = the script: re-present if np.sum(current_spike_count) < 5
        block = Fig. 2C block size for the running training accuracy
        present_ms, rest_ms, dt_ms, intensity = the script's protocol constants (diehl_cook_2015); the dataset must encode with the same dt/present/rest
        intensity_step = +1 per re-presentation: +255/8 = 31.9 Hz on the brightest pixel, the paper's "+32 Hz"
        reset_between_samples (inherited) must stay False: the script's resets are its own (fresh_network per stage)
        report_every (inherited, default 500) is logging only and excluded from run_id (RunCfg.HASH_EXCLUDE).
    """
    protocol_type: str = "DC2015"
    n_train: int = 40_000
    n_assign: int = 60_000
    test_passes: int = 10
    max_repeats: int = 100
    min_spikes: int = 5
    block: int = 10_000
    present_ms: float = T_PRESENT_MS
    rest_ms: float = T_REST_MS
    dt_ms: float = DT_MS
    intensity: float = INPUT_INTENSITY
    intensity_step: float = 1.0


class DCProtocol(Protocol):
    """
    TRAIN -> (trained model saved) -> ASSIGN -> TEST -> REPORT; see the dc_reproduce.py header for each step's source.
    Needs: diehl_cook_2015.build() modules (DCAdaptiveThreshold.freeze, dc_normalise's norm target on the plasticity), a ModelInput over a dataset with
    variant(i, intensity=, rng=) (diehl_cook_2015.mnist_input over DCPoissonImages) as `data` AND the official test set as PhasicNet.train(test_data=).
    STATE (proto__* in snapshots): Carry, Assign_train, Resp_train, Reps_train, Pred_train, Resp_assign, Reps_assign, Resp_test, Reps_test (+ block_acc as json).
    The three loops are self.ops.repeat (stage 0 train, 1 assign, 2 test), all resumable; progress = {"stage", "i"}. The tail is self.ops.finish.
    """
    protocol_type = "DC2015"
    needs_test_set = True      ##the official test set (test_data=mnist_input(x_test, y_test)); Protocol.__init__ checks it

    def __init__(self, cfg: DCProtocolCfg | None, run, *, out_dir=None):
        super().__init__(DCProtocolCfg() if cfg is None else cfg, run, out_dir=out_dir)
        c, net = self.cfg, run.ctx.net
        if c.reset_between_samples:   ##the script never resets between presentations
            raise ValueError("DCProtocol resets per stage itself (fresh_network); leave reset_between_samples False.")
        self._n_present, self._n_rest = int(round(c.present_ms / c.dt_ms)), int(round(c.rest_ms / c.dt_ms))
        self._exc = net.neuronslices["exc"]                                                   ##true_neuronIDs of the exc group (history.Counts is over ALL neurons)
        self._exc_rows = slice(self._exc.start - net.row_offset, self._exc.stop - net.row_offset)   ##the same neurons as W rows (Theta)
        n_exc = self._exc.stop - self._exc.start
        self._norm = (net.synapseslices[run.plasticity.norm_syn_group], run.plasticity.norm_target)   ##what dc_normalise scales
        ##protocol STATE (public ndarrays -> state_dict -> proto__*)
        self.Carry = np.zeros(n_exc, dtype=np.int64)              ##exc spikes of the last rest period: the script counts them towards the NEXT presentation
        self.Assign_train = np.zeros(n_exc, dtype=np.int64)       ##the training script starts from assignments = zeros
        self.Resp_train = np.zeros((c.n_train, n_exc), dtype=np.int32)
        self.Reps_train = np.zeros(c.n_train, dtype=np.int16)
        self.Pred_train = np.zeros(c.n_train, dtype=np.int16)
        ##ASSIGN / TEST responses as state (resumable stages)
        self.Resp_assign = np.zeros((c.n_assign, n_exc), dtype=np.int32)
        self.Reps_assign = np.zeros(c.n_assign, dtype=np.int16)
        n_te = len(run.test_set)      ##the picked test set's size
        self.Resp_test = np.zeros((c.test_passes, n_te, n_exc), dtype=np.int32)
        self.Reps_test = np.zeros((c.test_passes, n_te), dtype=np.int16)
        self._block_acc: list = []
        self._restored_by_subclass = ("Resp_train", "Reps_train", "Pred_train")   ##restored by rows in load_state_dict (resume across --n-train)

    ##block_acc is a list of dicts (json), not an array: carried beside the arrays
    @property
    def state_dict(self) -> dict:
        return {**super().state_dict, "block_acc": np.asarray(json.dumps(self._block_acc))}

    def load_state_dict(self, state: dict) -> None:
        self._block_acc = json.loads(str(state["block_acc"])) if "block_acc" in state else []
        ##the per-example arrays are restored by ROWS, so a snapshot can be continued to a larger --n-train or
        ## re-run with a smaller one; the base copies the rest (Carry, Assign_train) whole
        state = dict(state)
        for k in ("Resp_train", "Reps_train", "Pred_train"):
            if k in state:
                src, dst = state.pop(k), getattr(self, k)
                j = min(len(src), len(dst))
                dst[:j] = src[:j]
        super().load_state_dict(state)

    # -------------------------------------------------------------------------
    # the presentation rule
    # -------------------------------------------------------------------------

    def present_example(self, data, i: int, key: str) -> tuple[np.ndarray, int]:      ##not Protocol.present (other arguments, other return): its own name
        """ONE example under the script's rule (dc_reproduce header): shown again at intensity + intensity_step until the window (= exc spikes since the
        END of the previous presentation: the carried rest count + this presentation) reaches min_spikes. Returns (that window, number of presentations).
        Normalises before every presentation in mode "train" only. Counts: ops.binned_counts with the bounds [n_present, T] -> the present window
        and the rest window of ONE presentation."""
        run, c = self.run, self.cfg
        training = run.ctx.sim.mode == "train"
        intensity = c.intensity
        window = self.Carry.copy()
        for rep in range(c.max_repeats):
            if training:
                dc_normalise(run.ctx.syn.W, *self._norm)
            frames = data.variant(i, intensity=intensity, rng=self.rng.rng("input", f"{key}:{rep}"))   ##keyed: same example -> same spikes
            present, rest = self.ops.binned_counts(frames, None, [self._n_present, frames.shape[0]], group="exc")
            window = present.astype(np.int64) + self.Carry                 ##spikes since the end of the previous presentation
            self.Carry[:] = rest                                           ##this rest period counts towards the NEXT presentation
            if window.sum() >= c.min_spikes:
                return window, rep + 1
            intensity += c.intensity_step
        return window, c.max_repeats            ##capped (the script has no cap): caller counts these

    # -------------------------------------------------------------------------
    # the protocol
    # -------------------------------------------------------------------------

    def execute(self) -> dict:
        run, c, ops, log = self.run, self.cfg, self.ops, logger.info
        train_set, test_set = run.train_set, run.test_set
        ops.header(derived_cfg_id=run.runmeta.derived_cfg_id)

        ##the example lists are ModelInput views of the run's sets; their labels() are the label vectors
        n_src = len(train_set)
        train = train_set.subset(np.arange(c.n_train) % n_src, "dc_train")      ##the script: example j = image j % 60000
        assign_set = train_set.subset(np.arange(c.n_assign) % n_src, "dc_assign")
        y_train, y_assign, y_test = train.labels(), assign_set.labels(), test_set.labels()
        n_te = len(test_set)
        j0 = int(self.progress.get("i", 0)) if int(self.progress.get("stage", 0)) == 0 else c.n_train   ##examples trained before this process (the report's "train" block)
        if ops.starting(0):
            fresh_network(run, redraw_delays=True)               ##the script's newly created network (V = v_rest - 40 mV, ...)
        else:
            log(f"resumed: {j0} examples already trained")

        ##-------------------------------------------------------------- TRAIN (stage 0)
        run.set_mode("train")

        def train_example(j):
            self.Resp_train[j], self.Reps_train[j] = self.present_example(train, j, f"train:{j}")
            if j % c.block == 0 and j > 0:                      ##the training script: new labels from the previous 10,000 responses (unassigned -> 0)
                self.Assign_train[:] = dc_assign_labels(self.Resp_train[j - c.block:j], y_train[j - c.block:j], unassigned=0)
            self.Pred_train[j] = dc_rank_predict(self.Resp_train[j:j + 1], self.Assign_train)[0]
            done = j + 1
            if done % c.block == 0:                             ##Fig. 2C: block accuracy with the labels of the previous block (the first block: all 0)
                cur = slice(done - c.block, done)
                acc = float((self.Pred_train[cur] == y_train[cur]).mean() * 100)
                self._block_acc.append({"examples": done, "train_acc": acc})
                log(f"  [Fig2C] training accuracy, examples {cur.start + 1}-{done}: {acc:.2f}%" + ("  (labels: all 0, as the script starts)" if done == c.block else ""))

        def train_tail(lo, hi):
            th = run.threshold.Theta[self._exc_rows]
            return (f"exc spikes/ex {self.Resp_train[lo:hi].sum(1).mean():5.1f}  re-presented {(self.Reps_train[lo:hi] > 1).mean() * 100:4.1f}%  "
                    f"theta {th.min():.2f}..{th.max():.2f}")
        ops.repeat(c.n_train, train_example, stage=0, name="train", extra=train_tail)
        capped_tr = int((self.Reps_train[j0:] >= c.max_repeats).sum())

        ##the trained model in the framework's own format (same keys as the snapshots; dc_reproduce.load_trained() reads it back)
        model = ops.save_model()

        ##-------------------------------------------------------------- ASSIGN (stage 1; the script's test mode: new network, learning off, theta fixed)
        run.threshold.freeze(True)
        run.set_mode("eval")                               ##from here on W and Theta are guarded (check_persistent_unchanged below)
        if ops.starting(1):
            fresh_network(run, redraw_delays=True); self.Carry[:] = 0

        def assign_example(k):
            self.Resp_assign[k], self.Reps_assign[k] = self.present_example(assign_set, k, f"assign:{k}")
        ops.repeat(c.n_assign, assign_example, stage=1, name="assign", extra=lambda lo, hi: f"re-presented {(self.Reps_assign[lo:hi] > 1).mean() * 100:.1f}%")
        assign = dc_assign_labels(self.Resp_assign, y_assign)

        ##-------------------------------------------------------------- TEST (stage 2; a new network again: same delays; passes = new Poisson draws)
        if ops.starting(2):
            fresh_network(run, redraw_delays=True); self.Carry[:] = 0

        def test_example(t):                                    ##consecutive passes in ONE run (no new network in between): t = pass * n_te + k
            q, k = divmod(t, n_te)
            self.Resp_test[q, k], self.Reps_test[q, k] = self.present_example(test_set, k, f"test{q}:{k}")
        ops.repeat(c.test_passes * n_te, test_example, stage=2, name="test",
                   extra=lambda lo, hi: f"re-presented {(self.Reps_test.reshape(-1)[lo:hi] > 1).mean() * 100:.1f}%")
        accs, cms = [], []
        for p, resp_te in enumerate(self.Resp_test):
            yhat = dc_rank_predict(resp_te, assign)
            accs.append(float((yhat == y_test).mean() * 100)); cms.append(confusion_matrix(y_test, yhat))
            log(f"  test pass {p + 1}: accuracy {accs[-1]:.2f}%  re-presented {(self.Reps_test[p] > 1).mean() * 100:.1f}%")
        run.check_persistent_unchanged()

        ##-------------------------------------------------------------- REPORT
        res = {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),     ##np.std as the evaluation script (ddof=0)
               "accuracy_per_pass": accs,
               "train_block_accuracy": self._block_acc, "unassigned_neurons": int((assign < 0).sum()),
               "neurons_per_class": np.bincount(assign[assign >= 0], minlength=10).tolist(),
               "train": {"examples": c.n_train - j0, "re_presented_frac": float((self.Reps_train[j0:] > 1).mean()) if c.n_train > j0 else None,
                         "capped": capped_tr, "exc_spikes_per_example": float(self.Resp_train[j0:].sum(1).mean()) if c.n_train > j0 else None},
               "assign": {"examples": c.n_assign}, "test": {"examples": n_te, "passes": c.test_passes}, "trained_model": model.name,
               "paper_reference": "82.9% (100 exc, 40,000 examples: Results text); Fig. 2B shows the triplet rule at about the same value at 100 neurons"}
        files = dict(assign=assign, resp_assign=self.Resp_assign, y_assign=y_assign, resp_test=self.Resp_test, y_test=y_test, confusion=np.stack(cms),
                     resp_train=self.Resp_train, y_train=y_train, reps_train=self.Reps_train, pred_train=self.Pred_train, reps_assign=self.Reps_assign,
                     reps_test=self.Reps_test)
        return ops.finish(res, files=files, lines=(
            f"  accuracy        : {res['accuracy_mean']:.2f}% (+- {res['accuracy_std']:.2f}, {c.test_passes} test pass(es) x {n_te}, labels from {c.n_assign})   paper: {res['paper_reference']}",
            f"  neurons/class   : {res['neurons_per_class']}   unassigned: {res['unassigned_neurons']}",
            f"  train           : {c.n_train - j0} examples   re-presented {res['train']['re_presented_frac']}   capped {capped_tr}"))
