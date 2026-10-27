#scorer - scoring for a run. 
# 
#HEALTH probes. O(N) reads of live state, cheap for mid-epoch evals.
#                 -> look for silence, runaway or saturation.   (model collapse eval-> early stop)
#SEPARABILITY. Calinski-Harabasz / Variance Ratio Criterion over per-sample spike-count vectors, grouped by label. 
#               -> perform labelled val pass at END-of-EPOCH. Expensive. (model quality eval.->model selection)
#               -> MODEL-SELECTION score ** (NOTE partially less relevant for neuromorphic networks; most states are meant to evolve and cannot be reconstituted to "best model", ie. restoring V to the best combination makes no sense in neurom. nets. Restoring an adaptive threshold config  (and other long-term parameters) DOES make sense)

#NOTE: CH/VRC =  NOT comparable across different number of classes or different samples.. 
#               -> Comparing CH across epochs of one fit is valid (same k, same n). NOT across datasets (unless with statistical dists to represent sampling effects)*** *SRC*)
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from runcfg import ScorerCfg
import logging

import numpy as np
import capture   

logger = logging.getLogger(__name__)



   ######################################################################################
##########################  Scoring/Seperability  #######################################
   ######################################################################################

def calinski_harabasz(responses: np.ndarray, labels: np.ndarray) -> float:
    """
    Calinski-Harabasz score (Variance Ratio Criterion).

        CH = (B / (k - 1)) / (W / (n - k))

    where B is between-class dispersion and W is within-class dispersion.

    Args:
        responses = (n_samples, n_features) per-sample response vectors (spike counts or rates)
        labels = (n_samples,) class label per sample

    Returns:
        Higher = better separated. 0.0 when undefined (k < 2, n <= k, or zero within-dispersion).

    NOTE  Verified against sklearn's definition
    """
    responses = np.asarray(responses, dtype=np.float64)
    labels = np.asarray(labels)
    n_samples = responses.shape[0]
    classes = np.unique(labels)
    k = classes.size

    if k < 2 or n_samples <= k:
        return 0.0

    overall_mean = responses.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls in classes:
        members = responses[labels == cls]
        class_mean = members.mean(axis=0)
        between += members.shape[0] * float(np.sum((class_mean - overall_mean) ** 2))
        within += float(np.sum((members - class_mean) ** 2))

    if within <= 0.0:
        ##Every sample in a class is identical: perfectly separated OR dead. 0.0, not +inf: the health probes tell them apart.
        return 0.0
    return (between / (k - 1)) / (within / (n_samples - k))


   ##########################################################
        ###########  Accuracy (any readout) ##########

##  accuracy / confusion_matrix score the PREDICTIONS of any readout (logistic regression, the neuron-label votes below, ...)

def accuracy(labels: np.ndarray, predicted: np.ndarray) -> float:
    """Percent of predictions equal to the labels (the one accuracy every protocol and readout reports)."""
    return float((np.asarray(predicted) == np.asarray(labels)).mean() * 100.0)


def confusion_matrix(labels: np.ndarray, predicted: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """(n_classes, n_classes) counts; row = true class, column = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels), np.asarray(predicted)), 1)
    return cm


   ##########################################################
        ###########  Label-assignment vote ##########

##      Accuracy - DiehlCook2015
##  assign_neuron_labels: label assigned by highest mean-spike-count response to said label by node/neuron
##    vote_predict: example's label assigned by labeled neurons: class with most exuberant (ascribed) neurons
##   vote_accuracy: % correct predictions
##  NOTE the neuron-label family (a class per neuron, then a vote) is D&C's own, unusual method, not best practice (the authors say so).
##       Each readout below returns PREDICTIONS; accuracy() above scores them. Exact D&C copy: dc_*; Zhang / BindsNET: proportion_weighting.

def assign_neuron_labels(responses: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """(n_examples, n_neurons) counts -> (n_neurons,) class per neuron = argmax over classes of its MEAN response. A neuron that never fired
    gets -1 (unassigned), as in the authors' evaluation code, instead of argmax's silent class 0."""
    responses = np.asarray(responses, dtype=np.float64)
    labels = np.asarray(labels)
    rates = np.zeros((n_classes, responses.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            rates[c] = responses[m].mean(axis=0)
    assign = rates.argmax(axis=0)
    assign[rates.max(axis=0) <= 0.0] = -1
    return assign



def vote_predict(responses: np.ndarray, assign: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """(n_examples, n_neurons) counts -> predicted class = argmax over classes of the MEAN count of that class's neurons. A class with no
    assigned neuron scores -inf, so it can never win by default; an all-silent example predicts the first class that has a neuron."""
    responses = np.asarray(responses, dtype=np.float64)
    scores = np.full((responses.shape[0], n_classes), -np.inf)
    for c in range(n_classes):
        m = assign == c
        if m.any():
            scores[:, c] = responses[:, m].mean(axis=1)
    return scores.argmax(axis=1)


def vote_accuracy(assign_responses: np.ndarray, assign_labels: np.ndarray, test_responses: np.ndarray, test_labels: np.ndarray,
                  n_classes: int = 10) -> float:
    """Percent correct on the test responses, with neuron labels taken from the assignment responses."""
    assign = assign_neuron_labels(assign_responses, assign_labels, n_classes)
    return accuracy(test_labels, vote_predict(test_responses, assign, n_classes))


def dc_assign_labels(responses: np.ndarray, labels: np.ndarray, n_classes: int = 10, unassigned: int = -1) -> np.ndarray:
    """
    Copy of the authors' get_new_assignments(): rate[c] = mean response to class c; a STRICT `>` against a running maximum from 0 -> ties go
    to the LOWER class; a neuron that never fired keeps `unassigned` (-1 in the paper's evaluation, 0 in the training script's Fig. 2C)."""
    responses = np.asarray(responses, dtype=np.float64)
    labels = np.asarray(labels)
    assign = np.full(responses.shape[1], unassigned, dtype=np.int64)
    best = np.zeros(responses.shape[1], dtype=np.float64)
    for c in range(n_classes):
        m = labels == c
        n = int(m.sum())
        if n == 0:
            continue
        rate = np.sum(responses[m], axis=0) / n
        win = rate > best
        best[win] = rate[win]; assign[win] = c
    return assign


def dc_rank_predict(responses: np.ndarray, assign: np.ndarray, n_classes: int = 10) -> np.ndarray: ##NOTE TODO: this detail is not very important. It does not need perfect replication**
    """
    Copy of the authors' get_recognized_number_ranking(assignments, rates)[0]: score[c] = mean response of class-c neurons (0 without neurons).
    TIES go to the HIGHER class index (2015 numpy's stable argsort, reversed), done explicitly since today's argsort is not stable for floats."""
    responses = np.asarray(responses, dtype=np.float64)
    scores = np.zeros((responses.shape[0], n_classes), dtype=np.float64)
    for c in range(n_classes):
        m = assign == c
        k = int(m.sum())
        if k > 0:
            scores[:, c] = np.sum(responses[:, m], axis=1) / k
    return (n_classes - 1) - np.argmax(scores[:, ::-1], axis=1)


def proportion_weighting(resp_assign: np.ndarray, y_assign: np.ndarray, resp_test: np.ndarray, n_classes: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zhang et al. 2025's readout (§4.2: "records the proportion of spikes each neuron emits for each input class and uses these proportions in combination with a
    weighted sum of network activity"), as BindsNET's evaluation.assign_labels (rates[n, c] = mean response of neuron n to class c; proportions = rates /
    row sum; assignment = argmax) + evaluation.proportion_weighting (score[c] = sum over neurons assigned c of proportion[n, c] * response[n] / count of
    such neurons; prediction = argmax). A neuron that never fired during assignment is UNASSIGNED (-1); a test sample with no weighted activity is
    predicted -1 (abstain, scored as wrong). Ties -> the lower class.
    Returns (assignments (n_exc,), proportions (n_exc, n_classes), predictions (n_test,))."""
    resp_assign = np.asarray(resp_assign, dtype=np.float64); y_assign = np.asarray(y_assign); resp_test = np.asarray(resp_test, dtype=np.float64)
    n = resp_assign.shape[1]
    rates = np.zeros((n, n_classes), dtype=np.float64)
    for c in range(n_classes):
        m = y_assign == c
        if m.any():
            rates[:, c] = resp_assign[m].mean(axis=0)
    tot = rates.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0                                                                ##a neuron that never fired: proportions 0 (no np.where keyword)
    proportions = rates / tot
    assign = np.argmax(rates, axis=1)
    assign[rates.sum(axis=1) == 0] = -1                                                 ##silent neuron -> unassigned
    scores = np.zeros((resp_test.shape[0], n_classes), dtype=np.float64)
    for c in range(n_classes):
        m = assign == c
        k = int(m.sum())
        if k > 0:
            scores[:, c] = (resp_test[:, m] * proportions[m, c][None, :]).sum(axis=1) / k
    pred = np.argmax(scores, axis=1)
    pred[scores.sum(axis=1) == 0] = -1                                                  ##silent sample -> abstain
    return assign, proportions, pred


   ######################################################################################
#########################health probes   ######################################
   ######################################################################################

class Scorer:
    """
    Two-tier scoring. 
    Health probes = cheap safety. -> detect collapse/fatal cascade -> abort/snapshot
    Score / separability = labelled metrics for "best" model found
    """

    def __init__(self, cfg: ScorerCfg = None):
        self.cfg = ScorerCfg() if cfg is None else cfg

    def health(self, ctx, *, steps: int, spike_count: int, dt: float) -> Dict[str, float]:
        """
        TIER 1. O(N) over state that already exists. No data, no forward pass, safe mid-epoch.

        Returns a dict of probes plus "collapsed" (1.0/0.0). The three failure modes:
            silence     --  No spikes: learning depressed everything, killing the network
            saturation  -- Weights stuck/pinned at bounds: cannot learn  
            runaway     -- All spikes: excitation escaped inhibition triggering fatal cascades.
        """
        rate_hz = capture.mean_rate_hz(ctx, spike_count=spike_count, steps=steps, dt=dt)   
        probes = {"rate_hz": rate_hz, "spike_count": float(spike_count), "steps": float(steps)}
        probes["collapsed"] = float(rate_hz > self.cfg.rate_hz_max or rate_hz < self.cfg.rate_hz_min)
        return probes

    def weight_health(self, ctx, *, w_min: float | None, w_max: float | None, eps: float | None = None) -> Dict[str, float]:
        """
        Weight-side probes, (need the clamp bounds from plasticity).
        """
        return capture.pinned_weights(ctx, w_min=w_min, w_max=w_max, eps=eps)

    def is_collapsed(self, probes: Dict[str, float]) -> bool:
        """Collapse verdict from a merged probe dict. Kept separate so Run can log the probes even when it does not act on them. 
        Hz catches cascades"""
        return probes.get("collapsed", 0.0) >= 1.0

    def separability(self, responses: np.ndarray, labels: np.ndarray) -> float:
        """The model-selection score. (greater_is_better=True)."""
        return calinski_harabasz(responses, labels)

    def accuracy(self, assign_responses: np.ndarray, assign_labels: np.ndarray, test_responses: np.ndarray, test_labels: np.ndarray) -> float:
        """Diehl & Cook's label-assignment vote accuracy (%), see vote_accuracy. Comparable to published SNN results."""
        return vote_accuracy(assign_responses, assign_labels, test_responses, test_labels)

    def metadata(self) -> dict:
        return {"cfg": asdict(self.cfg), "scorer": self.cfg.scorer,
                "greater_is_better": True}