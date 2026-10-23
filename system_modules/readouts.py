#readouts - the LINEAR readout of a liquid state machine: multinomial logistic regression on per-sample spike counts, numpy only.
# Shared by the LSM experiments (Deckers 2022, Luo 2026, NALSM, Krenzer 2025). USER-side scoring, not a Dynamics module: it never sees ctx.
#
#Softmax regression on features standardised with the TRAINING mean/std, cross-entropy + L2, Adam. The solver is NOT Deckers' (they name none);
# the loss is convex, so any solver reaches the same optimum (sklearn's C = 1 / (l2 * n_samples)).
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
from scorer import confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class LogisticReadout:
    """
    Args:
        l2 = L2 penalty on the weights (not the biases); loss = mean cross-entropy + l2/2 * ||W||^2
        lr, epochs = Adam step size and number of full-batch epochs
        standardize = z-score the features with the training set's mean and std (zero-variance features -> 0)
        seed = initial weights ~ N(0, 0.01) from this seed (the problem is convex: the seed only changes the path, not the optimum)
    After fit(): classes_ (label values in column order), W_ (n_features, n_classes), b_ (n_classes,), mean_/std_, loss_ (per epoch;
    the mean training loss of the epoch's batches), val_loss_ (per epoch, when val_fraction > 0).
    The penalty is l2/2 ||W||^2 on the MEAN cross-entropy: Luo 2026's Eq. (19) writes lambda/(2m) ||W||^2 per batch of m, i.e. l2 = lambda / m here.
    """
    l2: float = 1e-3
    lr: float = 0.02
    epochs: int = 400
    standardize: bool = True
    seed: int = 0
    dropout: float = 0.0                    ##input dropout during training only, as torch's nn.Dropout(p) (HLSP: 0.2)
    ##HLSP-LSM readout (Luo 2026 Sec. 4.1): mini-batches, a validation split and a plateau schedule. Defaults keep the full-batch behaviour.
    batch_size: Optional[int] = None        ##None = full batch; else Adam on shuffled mini-batches of this size (Luo: 512)
    val_fraction: float = 0.0               ##> 0: this fraction of the training set is held out; val loss decides the lr schedule (Luo: 10k of 60k)
    plateau_patience: int = 2               ##torch ReduceLROnPlateau(patience): lr *= plateau_factor once MORE than this many epochs failed to improve val loss by 1e-4 rel.
    plateau_factor: float = 0.5
    ##NALSM readout (Ivanov & Michmizos 2021): soft targets, a validation block from the END, selection by validation ACCURACY. Defaults keep the above.
    soft_target_logit: Optional[float] = None   ##not None: the target of a sample is softmax(+-logit one-hot) instead of one-hot (NALSM: +-1 -> 0.45 / 0.061)
    val_from_end: bool = False                  ##True: the validation block is the LAST round(val_fraction n) samples, not a random draw (NALSM: train[50000:60000])
    select_by_val_acc: bool = False             ##True: W_, b_ are those of the epoch with the best validation accuracy (val_acc_ per epoch)
    early_stop_patience: Optional[int] = None   ##with select_by_val_acc: stop after this many epochs without a new best (None = run all epochs)
    val_acc_: list = field(default_factory=list, repr=False)
    val_loss_: list = field(default_factory=list, repr=False)
    classes_: np.ndarray = field(default=None, repr=False)
    W_: np.ndarray = field(default=None, repr=False)
    b_: np.ndarray = field(default=None, repr=False)
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)
    loss_: list = field(default_factory=list, repr=False)

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.standardize:
            return X
        return (X - self.mean_) / self.std_

    def fit(self, X, y, X_val=None, y_val=None) -> "LogisticReadout":
        """X_val / y_val = an EXPLICIT validation set: when given, val_fraction is ignored and nothing is held out of X.
        Its labels must be among y's classes."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"X must be (n_samples, n_features) with one label per sample, got {X.shape} and {len(y)} labels.")
        self.classes_, yi = np.unique(y, return_inverse=True)
        n, d = X.shape
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64); y_val = np.asarray(y_val)
            if not np.isin(y_val, self.classes_).all():
                raise ValueError("fit: the validation labels contain a class absent from the training labels.")
        k = len(self.classes_)
        if k < 2:
            raise ValueError("fit needs at least two classes.")
        self.mean_ = X.mean(axis=0) if self.standardize else np.zeros(d)           ##from the TRAINING rows only (X_val is appended below)
        std = X.std(axis=0) if self.standardize else np.ones(d)
        self.std_ = np.where(std > 0, std, 1.0)                       ##one-time setup, not a hot loop
        if X_val is not None:
            X = np.concatenate([X, X_val]); yi = np.concatenate([yi, np.searchsorted(self.classes_, y_val)])     ##appended, then taken as the validation block below
        n_all = len(X)
        Z = self._features(X)
        Y = np.zeros((n_all, k), dtype=np.float64)
        Y[np.arange(n_all), yi] = 1.0
        if self.soft_target_logit is not None:                      ##(NALSM): softmax of a +-logit vector as the target distribution
            L = np.full((n_all, k), -float(self.soft_target_logit)); L[np.arange(n_all), yi] = float(self.soft_target_logit)
            Y = np.exp(L); Y /= Y.sum(axis=1, keepdims=True)

        rng = np.random.default_rng(self.seed)
        ##validation split (val loss drives the plateau schedule), mini-batches
        if X_val is not None:                                                             #the explicit block, after the n training rows
            n_val = len(X_val); val_idx, tr_idx = np.arange(n, n + n_val), np.arange(n)
        else:
            n_val = int(round(self.val_fraction * n)) if self.val_fraction > 0 else 0
            perm = np.arange(n) if self.val_from_end else rng.permutation(n)                   ##(NALSM): val_from_end
            val_idx, tr_idx = (perm[n - n_val:], perm[:n - n_val]) if self.val_from_end else (perm[:n_val], perm[n_val:])
        Zt, Yt, yit = Z[tr_idx], Y[tr_idx], yi[tr_idx]
        n_tr = len(tr_idx)
        bs = n_tr if self.batch_size is None else max(1, min(int(self.batch_size), n_tr))
        W = rng.normal(0.0, 0.01, size=(d, k))
        b = np.zeros(k)
        mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr, t, stale, best_val = float(self.lr), 0, 0, None
        self.loss_, self.val_loss_, self.val_acc_ = [], [], []
        best_acc, best_Wb, acc_stale = -1.0, None, 0                                       ##(NALSM): selection by validation accuracy

        def _probs(Zb):
            logits = Zb @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            return P

        for epoch in range(int(self.epochs)):
            order = rng.permutation(n_tr) if bs < n_tr else np.arange(n_tr)
            losses = []
            for start in range(0, n_tr, bs):
                sl = order[start:start + bs]
                Zb, Yb, yb = Zt[sl], Yt[sl], yit[sl]
                m = len(sl)
                if self.dropout > 0:                                                      ##inverted dropout on the inputs, training only
                    Zb = Zb * (rng.random(Zb.shape) >= self.dropout) / (1.0 - self.dropout)
                P = _probs(Zb)
                losses.append(-np.log(P[np.arange(m), yb] + 1e-300).mean() + 0.5 * self.l2 * np.sum(W * W))
                G = (P - Yb) / m
                gW = Zb.T @ G + self.l2 * W
                gb = G.sum(axis=0)
                t += 1
                ##Adam
                mW = beta1 * mW + (1 - beta1) * gW; vW = beta2 * vW + (1 - beta2) * gW * gW
                mb = beta1 * mb + (1 - beta1) * gb; vb = beta2 * vb + (1 - beta2) * gb * gb
                c1, c2 = 1 - beta1 ** t, 1 - beta2 ** t
                W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
                b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
            self.loss_.append(float(np.mean(losses)))
            if n_val:
                Pv = _probs(Z[val_idx])
                vl = float(-np.log(Pv[np.arange(n_val), yi[val_idx]] + 1e-300).mean())
                self.val_loss_.append(vl)
                va = float(np.mean(np.argmax(Pv, axis=1) == yi[val_idx])); self.val_acc_.append(va)   ##(NALSM)
                if self.select_by_val_acc:
                    if va > best_acc:
                        best_acc, best_Wb, acc_stale = va, (W.copy(), b.copy()), 0
                    else:
                        acc_stale += 1
                        if self.early_stop_patience is not None and acc_stale >= self.early_stop_patience:
                            logger.info("LogisticReadout: validation accuracy peaked at %.4f, stopping after epoch %d", best_acc, epoch + 1)
                            break
                ##(second pass, HLSP): torch's ReduceLROnPlateau(mode="min", threshold=1e-4 "rel"), which the authors call: an epoch
                ## improves only if vl < best * (1 - 1e-4), and lr drops when MORE than `patience` consecutive epochs failed to (the 3rd with patience 2)
                if best_val is None or vl < best_val * (1.0 - 1e-4):
                    best_val, stale = vl, 0
                else:
                    stale += 1
                    if stale > self.plateau_patience:
                        lr *= self.plateau_factor; stale = 0
                        logger.info("LogisticReadout: epoch %d, val loss %.4f not improving -> lr %.2e", epoch + 1, vl, lr)
        self.W_, self.b_ = (W, b) if best_Wb is None else best_Wb                        ##(NALSM): the selected epoch when select_by_val_acc
        logger.info("LogisticReadout: %d samples x %d features -> %d classes, loss %.4f -> %.4f over %d epochs",
                    n, d, k, self.loss_[0], self.loss_[-1], len(self.loss_))
        return self

    def predict_proba(self, X) -> np.ndarray:
        logits = self._features(X) @ self.W_ + self.b_
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        """Accuracy in [0, 1]."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


def fit_evaluate(X_train, y_train, X_test, y_test, *, X_val=None, y_val=None, **readout_kw) -> dict:
    """Fit on the training responses, score both sets. Returns accuracies (percent), the confusion matrix on the test set and the readout.
    X_val / y_val = an explicit validation set (passed to fit); "val_acc" is then reported too."""
    ro = LogisticReadout(**readout_kw).fit(X_train, y_train, X_val, y_val)
    pred_test = ro.predict(X_test)
    return {"train_acc": 100.0 * ro.score(X_train, y_train), "test_acc": 100.0 * float(np.mean(pred_test == np.asarray(y_test))),
            "val_acc": None if X_val is None else 100.0 * ro.score(X_val, y_val),
            "confusion_test": confusion_matrix(y_test, pred_test, int(ro.classes_.max()) + 1), "classes": ro.classes_, "pred_test": pred_test,
            "final_loss": ro.loss_[-1] if ro.loss_ else None, "readout": ro}


def readout_report(ro: dict, Resp_train, Y_train, Resp_test, Y_test) -> tuple[dict, dict]:
    """
    The tail ELSM, HLSP, NALSM and RLSM each wrote after fit_evaluate (`ro` = its dict). Returns (res, files):
        res   = test_acc, train_acc, n_train, n_test, classes, readout_final_loss
        files = the arrays of responses.npz: resp_train, y_train, resp_test, y_test (the RAW responses), pred_test, confusion_test, classes,
                readout_W, readout_b, readout_mean, readout_std
    Each protocol adds its own keys on top.
    """
    r = ro["readout"]
    res = {"test_acc": ro["test_acc"], "train_acc": ro["train_acc"], "n_train": int(len(Y_train)), "n_test": int(len(Y_test)),
           "classes": ro["classes"].tolist(), "readout_final_loss": ro["final_loss"]}
    files = dict(resp_train=Resp_train, y_train=Y_train, resp_test=Resp_test, y_test=Y_test, pred_test=ro["pred_test"], confusion_test=ro["confusion_test"],
                 classes=ro["classes"], readout_W=r.W_, readout_b=r.b_, readout_mean=r.mean_, readout_std=r.std_)
    return res, files
