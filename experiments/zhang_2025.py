#main - Zhang, Yang, Zhang & Chen (2025) assembled in PhasicNet: Diehl & Cook (2015) (diehl_cook_2015.py, diehlcook.py) plus the paper's three
# additions, each behind a build() switch so the ablation of Table 2 / Fig. 6 can be run: dendrites (zhang2025.ZhangDendriticSynapse), adaptive
# self-organising inhibition (Eq. 7 SOM-graded inh -> exc weights here + zhang2025.ZhangOneSpikeThreshold), partially shared exc -> inh wiring (here).
# The equations, what the paper states and what it does not are documented in zhang2025.py.
# THIS IS A DENDRITIC D&C VARIANT BUILT FROM THE PAPER'S EQUATIONS, NOT A REPLICATION: the paper gives no code and leaves the items listed in
# zhang2025.py open; every such item is an argument below with its default marked INFERRED.
#
#THE ARCHITECTURE (all switches on), with D&C's builder patterns where the paper keeps D&C's wiring:
#   input --all_to_all--> exc        plastic as D&C (w0 ~ U(0.003, 0.303), normalised to sum 78 before every example, delays U[0, 10) ms), PAIR STDP
#                                    nu_pre 1e-4 / nu_post 1e-2 (Eq. 2-3; DCTripletSTDP with triplet=False) -- ends on the excitatory DENDRITES (Eq. 4-6)
#   exc   --psc_g4_s2_grid--> inh    fixed 10.4 (D&C's). Partially shared: inh k receives a 2 x 2 TILE of the exc grid; adjacent tiles share `shared`
#                                    exc neurons (see psc_groups; INFERRED: the paper gives no construction). psc=False: D&C's one_to_one, n_inh = n_exc.
#   inh   --psc_g4_s2_back--> exc    all exc neurons EXCEPT the inh neuron's own sources (Eq. 7 speaks of "other neurons"; D&C: all_but_self), weight
#                                    -R with R = lambda * d / max d + beta_s (Eq. 7, lambda 100, beta_s 1), d = Euclidean distance on the exc grid between
#                                    the exc neuron and the inh neuron's position = the MEAN position of its sources (INFERRED; one_to_one: its exc neuron).
#                                    som_inhibition=False: D&C's fixed -17.
# post_offset=True as D&C. The exc grid (square, side sqrt(n_exc), C order) is recorded in the spec (set_positions); a non-square n_exc needs grid=.
#
#STIMULUS: user-side, as D&C (dc_reproduce.DCPoissonImages): Poisson at pixel/8*intensity Hz for PRESENT_MS_PAPER = 100 ms (§4.2) + D&C's 150 ms
# rest (the paper gives no rest period). runcfg(present_ms=, rest_ms=) sets the protocol's window; the dataset must be built with the same values.
#
#READOUT: DCProtocol's (assignment + rank vote) is what the protocol scores; the paper's proportion-weighted scheme (BindsNET's) is computed
# afterwards from the saved responses by proportion_weighting() below and reported beside it.
#
#RUN (from the repo root):  python -c "import experiments, zhang_2025; zhang_2025.main(n_train=1000, n_assign=1000, n_test=1000, tag='t1')"
# In parallel: load=dc_reproduce.mnist_dataset, load_kw present_ms / rest_ms. Runtime classes: the pack ZHANG_CLASSES.
from __future__ import annotations
import json, logging
from dataclasses import replace
from pathlib import Path
##experiments/__init__.py puts the code folders on sys.path (run from the repo root)
import numpy as np
from phasicnet import PhasicNet, RuntimeClasses
from builder import CONNECTIVITY_PATTERNS, _generate_netmeta
from cfg import Cfg, SimulationCfg, NoiseCfg, PhaseCfg
from runcfg import RunCfg, DataCfg, CheckpointCfg, ScorerCfg, HistoryCfg
from diehlcook import (DCSynapseCfg, DCIntegratorCfg, DCThresholdCfg, DCPlasticityCfg,
                       DCConductanceSynapse, DCLIFeuler, DCAdaptiveThreshold, DCTripletSTDP)
from zhang2025 import ZhangSynapseCfg, ZhangDendriticSynapse, ZhangThresholdCfg, ZhangOneSpikeThreshold
from original_dynamics import TestNoise
from phases import DefaultPhase
from paths import RAW_DIR, NETWORKS_DIR, RUNS_DIR
from modelinput import ModelInput
from preprocessing import load_mnist
from diehl_cook_2015 import DT_MS, T_PRESENT_MS, T_REST_MS, INPUT_INTENSITY, SEED, fresh_network, mnist_input     ##D&C's constants; fresh_network works unchanged (Dend is transient state)
from dc_protocol import DCProtocol, DCProtocolCfg
from dc_reproduce import mnist_dataset   ##(train, test) loader

logger = logging.getLogger("zhang_2025")

#---------------------------------------------------------------- the paper's stated values (§4.2, §3.1, §3.3, §3.4)
GAMMA = 0.5                     #     Eq. 5 amplification
THETA_F = (1.0, 5.0, 10.0)      #     Eq. 6 thresholds, one dendrite each (the count 3 is INFERRED from the list)
TAU_L_MS = 10.0                 #ms   Eq. 4 decay of the dendritic current
LAMBDA = 100.0                  #     Eq. 7
BETA_S = 1.0                    #     Eq. 7
PSC_GROUP = 4                   #     exc neurons per inh neuron (§3.4)
PSC_SHARED = 2                  #     shared exc neurons between adjacent groups (§3.4, Fig. 6 "Shared 2")
PRESENT_MS_PAPER = 100.0        #ms   Poisson encoding duration (§4.2); the rest period is not stated
ETA_PRE, ETA_POST = 1e-4, 1e-2  #     "eta+ = 0.0001, eta+ = 0.01" (sic) = D&C's nu_pre, nu_post


   ######################################################################################
##########################   Partially shared connection (§3.4)   #########################
   ######################################################################################

def psc_groups(n_exc: int, group_size: int = PSC_GROUP, shared: int = PSC_SHARED, grid=None) -> np.ndarray:
    """(n_inh, group_size) exc indices feeding each inh neuron.
    grid=None ("index" layout): group k = {k * stride, ..., k * stride + group_size - 1} modulo n_exc, stride = group_size - shared, n_inh = ceil(n_exc / stride).
        shared 2 of 4 -> every exc neuron in exactly two groups; shared 0 -> disjoint groups, n_inh = n_exc / 4. Groups cross row ends of the exc grid
        and wrap around.
    grid=(rows, cols) ("grid" layout): groups are 2 x 2 TILES of the exc grid (group_size must be 4, rows even), so the inh neuron's position (the
        mean of its sources) IS where its sources are. shared 2: tile origins step one column; shared 1: origins on a checkerboard (diagonal
        neighbours share one corner); shared 0: disjoint tiles. n_inh for 10 x 10: 45 / 41 / 25. INFERRED: the paper gives no construction."""
    n_exc, group_size, shared = int(n_exc), int(group_size), int(shared)
    if group_size < 1 or not 0 <= shared < group_size:
        raise ValueError(f"psc_groups: need group_size >= 1 and 0 <= shared < group_size, got {group_size}, {shared}.")
    if n_exc < group_size:
        raise ValueError(f"psc_groups: n_exc={n_exc} is smaller than one group of {group_size}.")
    if grid is None:
        stride = group_size - shared
        n_inh = -(-n_exc // stride)
        return (np.arange(n_inh)[:, None] * stride + np.arange(group_size)[None, :]) % n_exc
    rows, cols = (int(g) for g in grid)
    if group_size != 4 or rows * cols != n_exc or rows % 2 or cols < 2:
        raise ValueError(f"psc_groups: the grid layout needs group_size 4 and an even number of rows with rows * cols == n_exc, got grid {grid}, n_exc {n_exc}, group_size {group_size}.")
    if shared == 2:
        origins = [(r, c) for r in range(0, rows - 1, 2) for c in range(cols - 1)]
    elif shared == 1:
        origins = [(r, c) for r in range(rows - 1) for c in range(cols - 1) if (r + c) % 2 == 0]
    else:
        origins = [(r, c) for r in range(0, rows - 1, 2) for c in range(0, cols - 1, 2)]
    return np.array([[r * cols + c, r * cols + c + 1, (r + 1) * cols + c, (r + 1) * cols + c + 1] for r, c in origins], dtype=np.int64)


def register_psc_patterns(group_size: int = PSC_GROUP, shared: int = PSC_SHARED, grid=None) -> tuple[str, str]:
    """Registers two deterministic connectivity patterns in builder.CONNECTIVITY_PATTERNS and returns their names: 'psc_g<group>_s<shared>[_grid<rows>x<cols>]'
    (exc -> inh: the groups) and '..._back' (inh -> exc: every exc EXCEPT the inh neuron's sources). Registered at import for the paper's (4, 0..2) in
    both layouts (grid 10 x 10); call this for other sizes BEFORE set_connectivity, and before loading a saved spec that uses them."""
    fwd = f"psc_g{int(group_size)}_s{int(shared)}" + ("" if grid is None else f"_grid{int(grid[0])}x{int(grid[1])}")
    back = fwd + "_back"

    def _forward(pre_n: int, post_n: int) -> np.ndarray:         ##pre = exc, post = inh
        src = psc_groups(pre_n, group_size, shared, grid)
        if post_n != len(src):
            raise ValueError(f"{fwd}: {pre_n} exc neurons form {len(src)} groups, but the inh group has {post_n} neurons.")
        mask = np.zeros((pre_n, post_n), dtype=bool)
        mask[src, np.arange(post_n)[:, None]] = True
        return mask

    def _backward(pre_n: int, post_n: int) -> np.ndarray:        ##pre = inh, post = exc
        return ~_forward(post_n, pre_n).T

    CONNECTIVITY_PATTERNS[fwd], CONNECTIVITY_PATTERNS[back] = _forward, _backward
    return fwd, back


for _s in (0, 1, 2):
    register_psc_patterns(PSC_GROUP, _s)
    register_psc_patterns(PSC_GROUP, _s, (10, 10))     ##the paper's 100-neuron grid


   ######################################################################################
##########################   Adaptive self-organising inhibition (§3.3, Eq. 7)   ##########
   ######################################################################################

def exc_grid(n_exc: int, grid=None) -> tuple[tuple[int, ...], np.ndarray]:
    """The exc grid and the (n_exc, ndim) float64 position of each exc neuron, C order (the builder's convention for a group alone on its grid).
    grid None: a square of side sqrt(n_exc) (the paper's 100 -> 10 x 10, Fig. 7); other sizes must pass grid= with prod(grid) >= n_exc."""
    n_exc = int(n_exc)
    if grid is None:
        side = int(round(np.sqrt(n_exc)))
        if side * side != n_exc:
            raise ValueError(f"exc_grid: n_exc={n_exc} is not a square; pass grid=(rows, cols) with rows * cols >= n_exc.")
        grid = (side, side)
    grid = tuple(int(g) for g in grid)
    if int(np.prod(grid)) < n_exc:
        raise ValueError(f"exc_grid: grid {grid} holds {int(np.prod(grid))} cells < n_exc={n_exc}.")
    pos = np.stack(np.unravel_index(np.arange(n_exc), grid), axis=1).astype(np.float64)
    return grid, pos


def som_inhibition_weights(pos_exc: np.ndarray, pos_inh: np.ndarray, pre_inh: np.ndarray, post_exc: np.ndarray, *,
                           lam: float = LAMBDA, beta_s: float = BETA_S, sqrt_distance: bool = False) -> np.ndarray:
    """Eq. 7 for the realized inh -> exc edges (pre_inh[e], post_exc[e]: local indices): R_e = lam * d_e / max_e d_e + beta_s, d_e = ||pos_exc[post] - pos_inh[pre]||_2.
    The max is over the realized edges (the paper: "max ||S_i - A_j||", over the other neurons). Returns the POSITIVE magnitudes (n_edges,), float64; the
    builder's Dale sign (negative) is applied by the caller. R in [beta_s, lam + beta_s]; the nearest other neuron gets the least inhibition."""
    d = np.linalg.norm(pos_exc[np.asarray(post_exc)] - pos_inh[np.asarray(pre_inh)], axis=1)
    if sqrt_distance:
        ##BindsNET's IncreasingInhibitionNetwork (the "SOM algorithm [18]" the paper cites) grades by sqrt(d); Eq. 7 as printed has no sqrt (the default)
        d = np.sqrt(d)
    dmax = float(d.max()) if d.size else 1.0
    if dmax <= 0:
        raise ValueError("som_inhibition_weights: every realized edge has distance 0 (one exc neuron inhibiting itself only?).")
    return float(lam) * d / dmax + float(beta_s)


def proportion_weighting(resp_assign: np.ndarray, y_assign: np.ndarray, resp_test: np.ndarray, n_classes: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The paper's readout (§4.2: "records the proportion of spikes each neuron emits for each input class and uses these proportions in combination with a
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
##########################   Assemble   ###################################################
   ######################################################################################

##the runtime classes as ONE pack (phasicnet.RuntimeClasses): all switches on; build() replaces synapse_cls / threshold_cls for the ablations
ZHANG_CLASSES = RuntimeClasses(synapse_cls=ZhangDendriticSynapse, integrator_cls=DCLIFeuler, threshold_cls=ZhangOneSpikeThreshold, plasticity_cls=DCTripletSTDP,
                               noise_cls=TestNoise, phase_classes=(DefaultPhase,), protocol_classes=(DCProtocol,))


def build(n_input: int, n_exc: int, *, dendrites: bool = True, som_inhibition: bool = True, psc: bool = True, one_spike: bool = True,
          group_size: int = PSC_GROUP, shared: int = PSC_SHARED, psc_layout: str = "grid", grid=None, lam: float = LAMBDA, beta_s: float = BETA_S, som_sqrt: bool = False,
          theta_f=THETA_F, gamma: float = GAMMA, tau_dend: float = TAU_L_MS, dendrite_assign: str = "all", dendrite_rule: str = "gate",
          soma_input: str = "conductance", soma_gain: float = 1.0, one_spike_select: str = "random", one_spike_losers: str = "reset",
          triplet: bool = False, inh_weight: float = 17.0, exc_inh_weight: float = 10.4, seed: int = SEED, dt_ms: float = DT_MS, norm_target: float = 78.0,
          save_net: bool = False) -> PhasicNet:
    """D&C + the paper's additions, ready for pnet.train(data, runcfg(...), test_data=...). All four switches False AND triplet=True =
    diehl_cook_2015.build() (same wiring, weights, cfg and classes; only the exc grid in the spec and the names differ).
        dendrites       ZhangSynapseCfg / ZhangDendriticSynapse instead of D&C's synapse (theta_f, gamma, tau_dend, dendrite_assign, dendrite_rule, soma_input, soma_gain)
        som_inhibition  inh -> exc weights = -(lam * d / max d + beta_s) on the exc grid (Eq. 7) instead of -inh_weight
        one_spike       ZhangOneSpikeThreshold instead of D&C's threshold (one_spike_select, one_spike_losers) -- the paper lists it under adaptive self-organising inhibition
        psc             exc -> inh in groups of group_size with `shared` shared neurons instead of one_to_one; inh -> exc = all exc but the sources instead
                        of all_but_self. psc_layout "grid" (2 x 2 tiles of the exc grid) | "index" (consecutive indices): see psc_groups
        triplet         False (default) = the paper's pair STDP (Eq. 2-3); True = D&C's triplet rule
    norm_target = 78 is D&C's value for 784 inputs; scale it with n_input otherwise."""
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    b = pnet.builder
    grid, pos_exc = exc_grid(n_exc, grid)
    if psc_layout not in ("grid", "index"):
        raise ValueError(f"build: psc_layout must be 'grid' or 'index', got {psc_layout!r}.")
    if psc:
        psc_grid = grid if psc_layout == "grid" else None
        sources = psc_groups(n_exc, group_size, shared, psc_grid)
        fwd, back = register_psc_patterns(group_size, shared, psc_grid)
        n_inh, conn = len(sources), dict(in_exc="all_to_all", exc_inh=fwd, inh_exc=back)
    else:
        sources = np.arange(n_exc)[:, None]
        n_inh, conn = n_exc, dict(in_exc="all_to_all", exc_inh="one_to_one", inh_exc="all_but_self")
    b.set_neurons(("input", n_input, True), ("exc", n_exc), ("inh", n_inh, False, True))
    b.set_positions(exc=grid)                                                  ##recorded in the spec; the positions used below are the same C-order cells
    b.set_synapses(("in_exc", "input", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    b.set_connectivity(**conn)
    b.set_weights({"in_exc": (0.003, 0.303), "exc_inh": float(exc_inh_weight), "inh_exc": -float(inh_weight)})     ##D&C's values; -: Dale's sign given explicitly
    b.spec_name = "zhang_2025_spec"
    tag = ("".join(k for k, on in (("d", dendrites), ("s", som_inhibition), ("o", one_spike), ("p", psc)) if on) or "dc") + ("" if triplet else "_pair")
    net = b.realize(network_name=f"zhang_2025_e{n_exc}_{tag}", post_offset=True, save=False)
    if som_inhibition:
        ##Eq. 7 into the realized inh -> exc edges, then the network meta (ids hash W0) is generated again so network_id reflects these weights
        pos_inh = pos_exc[sources].mean(axis=1)
        m = net.syntype == net.type_key["inh_exc"]
        pre_local = net.pre_idx[m] - net.neuronslices["inh"].start
        post_local = net.post_idx[m] + net.row_offset - net.neuronslices["exc"].start
        R = som_inhibition_weights(pos_exc, pos_inh, pre_local, post_local, lam=lam, beta_s=beta_s, sqrt_distance=som_sqrt)   ##som_sqrt: BindsNET's sqrt grading
        net.W0[m] = (-R).astype(np.float32)
        net.meta = _generate_netmeta(neuronspecs=net.neuronspecs, synapsespecs=net.synapsespecs, spec_name=net.meta.spec_name, network_name=net.meta.network_name,
                                     neuronslices=net.neuronslices, synapseslices=net.synapseslices, type_key=net.type_key, syntype=net.syntype,
                                     pre_idx=net.pre_idx, post_idx=net.post_idx, W0=net.W0, post_offset=net.post_offset, n_input=net.n_input, rng=b.rng)
        logger.info("SOM inhibition: %d inh -> exc edges, |w| in [%.2f, %.2f] (lam %.1f, beta_s %.1f); network_id %s", int(m.sum()), R.min(), R.max(), lam, beta_s, net.meta.network_id)
    if save_net:
        net.save_net(network_name=net.network_name, spec_name="zhang_2025_spec", JSON=True, YAML=False, save_to=NETWORKS_DIR)

    ##D&C's cfg (diehl_cook_2015.build, values unchanged) with the Zhang sections where switched on
    delays = dict(delay_syn_group="in_exc", delay_min=0.0, delay_max=10.0)
    t_refrac, t_refrac_inh, v_reset_inh = 5.0, 2.0, -45.0
    syn_kw = dict(b=None, tau_ge=1.0, tau_gi=2.0, e_exc=0.0, e_inh=-100.0, e_inh_inh=-85.0, **delays, v_reset_inh=v_reset_inh, t_refrac_inh=t_refrac_inh)
    thr_kw = dict(v_reset=-65.0, v_th=-52.0, tau_th=1e7, adpt_th=0.05, v_reset_inh=v_reset_inh, v_th_inh=-40.0, theta_init=20.0, offset=20.0,
                  t_refrac=t_refrac, t_refrac_inh=t_refrac_inh)
    if dendrites:
        synapse = ZhangSynapseCfg(**syn_kw, theta_f=tuple(float(t) for t in theta_f), gamma=gamma, tau_dend=tau_dend, dendrite_assign=dendrite_assign,
                                  dendrite_rule=dendrite_rule, soma_input=soma_input, soma_gain=soma_gain)
    else:
        synapse = DCSynapseCfg(**syn_kw)
    threshold = ZhangThresholdCfg(**thr_kw, one_spike=True, one_spike_select=one_spike_select, one_spike_losers=one_spike_losers) if one_spike else DCThresholdCfg(**thr_kw)
    pnet.cfg = Cfg(
        simulation=SimulationCfg(dt=dt_ms, weights_format="dense", update_on="timestep"),
        synapse=synapse,
        integrator=DCIntegratorCfg(tau_mem=100.0, v_rest=-65.0, r=1.0, tau_mem_inh=10.0, v_rest_inh=-60.0, t_refrac=t_refrac, t_refrac_inh=t_refrac_inh,
                                   v_reset_inh=v_reset_inh),
        threshold=threshold,
        plasticity=DCPlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=ETA_POST, A_ltd=ETA_PRE, eta_exc=1.0, eta_inh=None,
                                   w_min=None, w_max=None, freeze=("inh",), tau_posttrace2=40.0, w_max_ee=1.0, norm_target=norm_target, norm_syn_group="in_exc", **delays,
                                   triplet=triplet),       ##the paper's pair rule by default
        noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0),
        phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(replace(ZHANG_CLASSES, synapse_cls=ZhangDendriticSynapse if dendrites else DCConductanceSynapse,     ##the pack, switched modules swapped
                                     threshold_cls=ZhangOneSpikeThreshold if one_spike else DCAdaptiveThreshold))
    pnet.cfg_name, pnet.run_name = "zhang_2025_cfg", "zhang_2025_run"
    return pnet


def runcfg(*, every=None, present_ms: float = PRESENT_MS_PAPER, rest_ms: float = T_REST_MS, n_test: int | None = None, **proto) -> RunCfg:
    """dc_reproduce._runcfg with the protocol's presentation window exposed: the paper's 100 ms + D&C's 150 ms rest. The dataset (DCPoissonImages)
    must be built with the same present_ms / rest_ms. proto = DCProtocolCfg fields (n_train, n_assign, test_passes, max_repeats, report_every, ...).
    n_test -> DataCfg.n_test (DataHandler.pick chooses the test examples; the protocol runs the whole picked set per pass)."""
    return RunCfg(data=DataCfg(test_fraction=0.0, n_test=n_test), checkpoint=CheckpointCfg(to_disk=False, every=every),
                  scorer=ScorerCfg(rate_hz_min=0.0), history=HistoryCfg(record_spikes=False, record_counts=True),
                  protocol=DCProtocolCfg(present_ms=present_ms, rest_ms=rest_ms, dt_ms=DT_MS, intensity=INPUT_INTENSITY, **proto))


   ######################################################################################
##########################   Run   ########################################################
   ######################################################################################

def main(*, npz=RAW_DIR / "mnist.npz", n_exc: int = 100, grid=None, no_dendrites: bool = False, no_som: bool = False, no_psc: bool = False, no_one_spike: bool = False,
         shared: int = PSC_SHARED, group_size: int = PSC_GROUP, psc_layout: str = "grid", lam: float = LAMBDA, beta_s: float = BETA_S, som_sqrt: bool = False, theta_f=THETA_F, gamma: float = GAMMA,
         tau_dend: float = TAU_L_MS, assign: str = "all", rule: str = "gate", soma: str = "conductance", soma_gain: float = 1.0, select: str = "random",
         losers: str = "reset", triplet: bool = False, present_ms: float = PRESENT_MS_PAPER, rest_ms: float = T_REST_MS, n_train: int = 40_000,
         n_assign: int = 60_000, n_test: int | None = None, test_passes: int = 1, max_repeats: int = 100,
         report_every: int = 500, seed: int = SEED, every: int | None = None, resume: str | None = None, tag: str = "", out: str | None = None):
    """ONE run; every setting is a keyword argument (defaults = the constants).
        no_dendrites / no_som / no_psc / no_one_spike: the ablation switches; grid: exc grid (rows, cols), None = square; psc_layout "grid" | "index";
        assign "all" | "split"; rule "gate" | "retain"; soma "conductance" | "current"; select "random" | "max_v"; losers "reset" | "keep" (what the
        one-spike rule does to the other crossers); triplet: D&C's triplet rule instead of the paper's pair rule; present_ms / rest_ms: the protocol's
        window (the paper's 100 ms + D&C's 150 ms rest; D&C: 350 + 150); every: snapshot every N examples; resume: a snapshot .npz; out: the run folder
        (default runs/zhang_2025_<tag or network name>); n_test: None = the whole official test set, a smaller number is a stratified pick (DataCfg.n_test).
    Returns the Run; results.json also carries the paper's proportion-weighted accuracy, computed from the saved responses."""
    kw = dict(present_ms=present_ms, rest_ms=rest_ms)
    data, test = mnist_dataset(npz=npz, seed=seed, **kw)      ##dc_reproduce's (train, test) ModelInputs
    n_input = data.n_input
    pnet = build(n_input, n_exc, dendrites=not no_dendrites, som_inhibition=not no_som, psc=not no_psc, one_spike=not no_one_spike, group_size=group_size,
                 shared=shared, psc_layout=psc_layout, grid=grid, lam=lam, beta_s=beta_s, som_sqrt=som_sqrt, theta_f=theta_f, gamma=gamma, tau_dend=tau_dend, dendrite_assign=assign,
                 dendrite_rule=rule, soma_input=soma, soma_gain=soma_gain, one_spike_select=select, one_spike_losers=losers, triplet=triplet, seed=seed,
                 norm_target=78.0 * n_input / 784)
    out = Path(out) if out else RUNS_DIR / f"zhang_2025_{tag or pnet.network.network_name}"
    out.mkdir(parents=True, exist_ok=True)
    rc = runcfg(every=every, present_ms=present_ms, rest_ms=rest_ms, n_train=n_train, n_assign=n_assign, n_test=n_test, test_passes=test_passes,
                max_repeats=max_repeats, report_every=report_every)
    run = pnet.train(data, rc, test_data=test, out_dir=out, resume_from=resume, save_run=False, save_history=False, dataset_name="mnist")
    ##the paper's proportion-weighted readout, from the responses the protocol saved
    r = np.load(out / "responses.npz")
    accs = []
    for resp_te in r["resp_test"]:
        _, _, pred = proportion_weighting(r["resp_assign"], r["y_assign"], resp_te)
        accs.append(float((pred == r["y_test"]).mean() * 100))
    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res["proportion_weighting_accuracy_per_pass"] = accs
    res["proportion_weighting_accuracy_mean"] = float(np.mean(accs))
    res["paper_reference_zhang"] = "Table 2 (MNIST ablation, neuron count not stated; 96.7 = the 6400-exc headline of §4.3): baseline 95.0, + dendrites 95.4, + SOM inhibition 96.7, + shared 96.7 (BindsNET, 100 ms encoding)"
    res_path.write_text(json.dumps(res, indent=2))
    logger.info("proportion-weighted accuracy: %s  (D&C rank vote: %s)", accs, res.get("accuracy_per_pass"))
    return run


if __name__ == "__main__":
    main()
