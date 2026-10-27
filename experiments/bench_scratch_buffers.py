"""Do shared scratch buffers (UpdateState, out=) pay off, or is a freshly returned array just as fast?

Two parts, both written to runs/bench_scratch_buffers/<time>/:
    micro.csv   the step's numpy operations, each written TWO ways (buffer = out= into a preallocated array, fresh = return a new array),
                over network size N, connection probability p and spike rate. Pure numpy: no PhasicNet objects.
    system.csv  the real Run.step(), timed per module (synapse / integrator / threshold / plasticity / rest of the step),
                for dense timestep, sparse timestep, hybrid_event and event, in mode "train" (plasticity on) and "eval" (off).
    summary.md  the two tables in readable form.

    python -c "import experiments, bench_scratch_buffers as b; b.main()"                        (all, ~10 min)
    python -c "import experiments, bench_scratch_buffers as b; b.main(part='ops')"              (part 1 only; part='step' = part 2 only)
    python -c "import experiments, bench_scratch_buffers as b; b.main(sizes=(500, 2000), probs=(0.05,))"

ratio = fresh / buffer: > 1 means the buffer is faster, < 1 means returning a fresh array is faster.
Only the timestep itself is measured: the protocol decides HOW MANY steps run and how many are "train" (plasticity) vs "eval" steps."""
from __future__ import annotations
import csv
import time
from datetime import datetime
import numpy as np
from paths import RUNS_DIR

SIZES = (500, 1000, 2000, 4000, 8000)
PROBS = (0.01, 0.05, 0.2, 1.0)
RATES = (0.02, 0.07)       ##spike rate per step; the real step (part 2) runs at drive 0.1 (~2%) and 0.3 (~7%)
MAX_DENSE_N = 4000          ##dense W is N x N float32 (+ dW + the ltd scratch): 4000 -> 3 x 64 MB
MAX_SYNAPSES = 16_000_000   ##sparse: W + pre + post + scratch, about 16 bytes per edge


def _time(fn, min_seconds=0.05, repeats=5) -> float:
    """Microseconds per call: the best of `repeats` timing rounds, each at least min_seconds long."""
    fn()
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = time.perf_counter() - t0
        if dt >= min_seconds:
            break
        n *= 2
    best = dt / n
    for _ in range(repeats - 1):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


   ######################################################################################
##########################   PART 1: the step's operations, buffer vs fresh   ##############
   ######################################################################################

def _neuron_ops(N, rate, rng):
    """Neuronal state (size N): the LIFexact update, the threshold test, and the sparse orchestra's copy of the returned I into upd.I2V."""
    V = rng.uniform(-70, -50, N).astype(np.float32); I = rng.random(N).astype(np.float32)
    v_rest, r, e = np.float32(-65), np.float32(10), np.float32(0.05); v_th = np.float32(-52)
    I2V, dV, spk = np.empty_like(I), np.empty_like(V), np.empty(N, bool)

    def lif_buffer():
        np.multiply(I, r, out=I2V); np.add(v_rest, I2V, out=I2V); np.subtract(I2V, V, dV); np.multiply(dV, e, out=dV)
    def lif_fresh():
        return ((v_rest + r * I) - V) * e
    def thr_buffer():
        np.greater_equal(V, v_th, out=spk)
    def thr_fresh():
        return V >= v_th
    def copy_into_buffer():
        I2V[...] = I            ##what step_sparse does with the synapse's returned I
    def keep_returned():
        return I
    return [("neuron", "integrator (LIFexact)", lif_buffer, lif_fresh),
            ("neuron", "threshold test", thr_buffer, thr_fresh),
            ("neuron", "returned I -> upd.I2V", copy_into_buffer, keep_returned)]


def _dense_ops(N, p, rate, rng):
    """Dense synaptic state (N x N): the synapse product and TraceSTDP's dW (two outer products, eta, the syn_mask)."""
    W = (rng.random((N, N)) < p).astype(np.float32) * rng.random((N, N), dtype=np.float32)
    mask = W != 0
    s = rng.random(N) < rate; sf = s.astype(np.float32)
    pre_tr, post_tr, eta = rng.random(N, dtype=np.float32), rng.random(N, dtype=np.float32), rng.random(N, dtype=np.float32)
    I, dW, ltd = np.empty(N, np.float32), np.empty((N, N), np.float32), np.empty((N, N), np.float32)
    A_ltp, A_ltd = np.float32(0.005), np.float32(0.006)

    def syn_buffer():
        np.dot(W, s, out=I)         ##as DeltaCurrentDense (bool spikes)
    def syn_fresh():
        return W @ s
    def stdp_buffer():
        np.multiply(sf[:, None], pre_tr[None, :], out=dW); np.multiply(dW, A_ltp, out=dW)
        np.multiply(post_tr[:, None], sf[None, :], out=ltd); np.multiply(ltd, A_ltd, out=ltd); np.subtract(dW, ltd, out=dW)
        np.multiply(dW, eta[None, :], out=dW); np.multiply(dW, mask, out=dW)
    def stdp_fresh():
        return (A_ltp * np.outer(sf, pre_tr) - A_ltd * np.outer(post_tr, sf)) * eta[None, :] * mask
    return [("synapse dense", "synapse W @ spikes", syn_buffer, syn_fresh),
            ("synapse dense", "plasticity dW (TraceSTDP)", stdp_buffer, stdp_fresh)]


def _sparse_ops(N, p, rate, rng):
    """Sparse synaptic state (edge list, sorted by post): the synapse sum over all edges (timestep) or over the edges of spiking pre
    neurons (event / hybrid_event), and the STDP rule over all edges (timestep) or over the touched edges (hybrid_event / event)."""
    n_syn = int(round(p * N * N))
    flat = np.sort(rng.choice(N * N, size=n_syn, replace=False))
    post, pre = (flat // N).astype(np.int32), (flat % N).astype(np.int32)
    W = rng.random(n_syn, dtype=np.float32)
    s = rng.random(N) < rate; sf = s.astype(np.float32)
    pre_tr, post_tr, eta = rng.random(N, dtype=np.float32), rng.random(N, dtype=np.float32), rng.random(N, dtype=np.float32)
    A_ltp, A_ltd = np.float32(0.005), np.float32(0.006)
    ##the edges a spike touches (pre OR post spiked), as SynapticIndex returns them
    k = np.flatnonzero(s[pre] | s[post])
    order = np.argsort(pre, kind="stable"); ptr = np.searchsorted(pre[order], np.arange(N + 1))
    def active_edges():
        return np.sort(np.concatenate([order[ptr[i]:ptr[i + 1]] for i in np.flatnonzero(s)])) if s.any() else np.zeros(0, np.int64)
    act = active_edges()
    gather, prod, Ibuf = np.empty(n_syn, np.float32), np.empty(n_syn, np.float32), np.empty(N, np.float32)
    kb = {name: np.empty(n_syn, np.float32) for name in ("a", "b", "c")}
    kp, kq = np.empty(n_syn, np.int32), np.empty(n_syn, np.int32)

    def syn_ts_buffer():
        np.take(sf, pre, out=gather); np.multiply(W, gather, out=prod)
        Ibuf[...] = np.bincount(post, weights=prod, minlength=N)       ##bincount always returns a new (float64) array
    def syn_ts_fresh():
        return np.bincount(post, weights=W * sf[pre], minlength=N).astype(np.float32, copy=False)
    def syn_ev_buffer():
        m = act.size; np.take(W, act, out=prod[:m]); np.take(post, act, out=kp[:m])
        Ibuf[...] = np.bincount(kp[:m], weights=prod[:m], minlength=N)
    def syn_ev_fresh():
        return np.bincount(post[act], weights=W[act], minlength=N).astype(np.float32, copy=False)

    def _rule_buffer(pp, qq, m, a, b, c):
        np.take(sf, qq, out=a[:m]); np.take(pre_tr, pp, out=b[:m]); np.multiply(a[:m], b[:m], out=a[:m]); np.multiply(a[:m], A_ltp, out=a[:m])
        np.take(post_tr, qq, out=b[:m]); np.take(sf, pp, out=c[:m]); np.multiply(b[:m], c[:m], out=b[:m]); np.multiply(b[:m], A_ltd, out=b[:m])
        np.subtract(a[:m], b[:m], out=a[:m]); np.take(eta, pp, out=b[:m]); np.multiply(a[:m], b[:m], out=a[:m])
    def _rule_fresh(pp, qq):
        ltp = sf[qq] * pre_tr[pp]; np.multiply(ltp, A_ltp, out=ltp)
        ltd = post_tr[qq] * sf[pp]; np.multiply(ltd, A_ltd, out=ltd)
        np.subtract(ltp, ltd, out=ltp); np.multiply(ltp, eta[pp], out=ltp)
        return ltp
    def stdp_ts_buffer():
        _rule_buffer(pre, post, n_syn, kb["a"], kb["b"], kb["c"])
    def stdp_ts_fresh():
        return _rule_fresh(pre, post)
    def stdp_k_buffer():
        m = k.size; np.take(pre, k, out=kp[:m]); np.take(post, k, out=kq[:m]); _rule_buffer(kp[:m], kq[:m], m, kb["a"], kb["b"], kb["c"])
    def stdp_k_fresh():
        return _rule_fresh(pre[k], post[k])
    return [("synapse sparse", "synapse sum, timestep (all edges)", syn_ts_buffer, syn_ts_fresh),
            ("synapse sparse", "synapse sum, event/hybrid (edges of spiking pre)", syn_ev_buffer, syn_ev_fresh),
            ("synapse sparse", "plasticity dw, timestep (all edges)", stdp_ts_buffer, stdp_ts_fresh),
            ("synapse sparse", "plasticity dw, hybrid/event (touched edges)", stdp_k_buffer, stdp_k_fresh)], n_syn, int(k.size), int(act.size)


def micro(sizes=SIZES, probs=PROBS, rates=RATES, seed=0, log=print) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    for N in sizes:
        for rate in rates:
            for section, op, buf, fresh in _neuron_ops(N, rate, rng):
                rows.append(_row(section, op, N, None, rate, N, None, buf, fresh, log))
            for p in probs:
                if N <= MAX_DENSE_N:
                    for section, op, buf, fresh in _dense_ops(N, p, rate, rng):
                        rows.append(_row(section, op, N, p, rate, N * N, None, buf, fresh, log))
                if p * N * N <= MAX_SYNAPSES:
                    ops, n_syn, n_touched, n_active = _sparse_ops(N, p, rate, rng)
                    for section, op, buf, fresh in ops:
                        k = n_touched if "touched" in op else n_active if "spiking pre" in op else n_syn
                        rows.append(_row(section, op, N, p, rate, n_syn, k, buf, fresh, log))
    return rows


def _row(section, op, N, p, rate, n_elements, k, buf, fresh, log) -> dict:
    tb, tf = _time(buf), _time(fresh)
    row = {"section": section, "op": op, "N": N, "p": p, "rate": rate, "elements": n_elements, "k": k,
           "buffer_us": round(tb, 2), "fresh_us": round(tf, 2), "ratio_fresh_over_buffer": round(tf / tb, 3)}
    log(f"{section:15s} {op:50s} N={N:5d} p={p} rate={rate}: buffer {tb:9.1f} us, fresh {tf:9.1f} us, ratio {tf / tb:5.2f}")
    return row


   ######################################################################################
##########################   PART 2: the real step, per module   ##########################
   ######################################################################################

def _build(N, p, fmt, update_on, seed, drive=1.0):
    """A main.py-style network (10% input, 80/20 exc/inh) with LIFexact + AdaptiveThreshold + (Sparse)TraceSTDP, connection probability p."""
    from phasicnet import PhasicNet
    from dynamicscfg import DynamicsCfg, SimulationCfg, SynapseCfg, IntegratorCfg, ThresholdCfg, PlasticityCfg, NoiseCfg, PhaseCfg
    from original_dynamics import DeltaCurrentDense, DeltaCurrentSparse, LIFexact, AdaptiveThreshold, TraceSTDP, SparseTraceSTDP, TestNoise
    from phases import DefaultPhase
    n_in = N // 10; n_exc = int(0.8 * (N - n_in)); n_inh = N - n_in - n_exc
    pnet = PhasicNet(seed)
    pnet._instantiate_builder()
    b = pnet.builder
    b.set_neurons(("input", n_in, True), ("exc", n_exc), ("inh", n_inh, False, True))
    b.set_synapses(("in_exc", "input", "exc"), ("exc_exc", "exc", "exc"), ("exc_inh", "exc", "inh"), ("inh_exc", "inh", "exc"))
    b.set_connectivity(in_exc=p, exc_exc=p, exc_inh=p, inh_exc=p)
    w = drive * 30.0 / max(1.0, p * n_exc)          ##keeps the recurrent drive roughly constant across N and p (rate printed per run)
    b.set_weights({"in_exc": (0.0, drive * 600.0 / max(1.0, p * n_in)), "exc_exc": w, "exc_inh": 3 * w, "inh_exc": 3 * w})
    b.spec_name = "bench_spec"
    b.realize(network_name=f"bench_n{N}_p{p}", save=False)
    sparse = fmt == "sparse"
    pnet.cfg = DynamicsCfg(simulation=SimulationCfg(dt=1.0, weights_format=fmt, update_on=update_on), synapse=SynapseCfg(b=None),
                           integrator=IntegratorCfg(tau_mem=20.0, v_rest=-65.0, r=10.0),
                           threshold=ThresholdCfg(v_reset=-70.0, v_th=-52.0, tau_th=30.0, adpt_th=2.0),
                           plasticity=PlasticityCfg(tau_pretrace=20.0, tau_posttrace=20.0, A_ltp=0.005, A_ltd=0.006, eta_exc=1e-3, eta_inh=1e-3),
                           noise=NoiseCfg(mu_mem=0.0, sigma_mem=0.0), phases=(PhaseCfg(phase_type="DEFAULT"),))
    pnet.set_runtime_classes(synapse_cls=DeltaCurrentSparse if sparse else DeltaCurrentDense, integrator_cls=LIFexact, threshold_cls=AdaptiveThreshold,
                             plasticity_cls=SparseTraceSTDP if sparse else TraceSTDP, noise_cls=TestNoise, phase_classes=(DefaultPhase,))
    return pnet, n_in


def _timed(fn, acc, key):
    def wrapped(ctx):
        t0 = time.perf_counter(); out = fn(ctx); acc[key] += time.perf_counter() - t0
        return out
    return wrapped


def system(sizes=(500, 1000, 2000, 4000), probs=(0.05, 0.2), steps=300, seed=0, input_rate=0.05, drives=(0.1, 0.3), log=print) -> list[dict]:
    from modelinput import ModelInput
    from runcfg import RunCfg
    rows = []
    for N in sizes:
        for p in probs:
          for drive in drives:
            for fmt, update_on in (("dense", "timestep"), ("sparse", "timestep"), ("sparse", "hybrid_event"), ("sparse", "event")):
                if fmt == "dense" and N > MAX_DENSE_N:
                    continue
                pnet, n_in = _build(N, p, fmt, update_on, seed, drive=drive)
                rng = np.random.default_rng(seed)
                frames = rng.random((steps * 2, n_in)) < input_rate
                data = ModelInput.from_tonic([(frames[:steps], 0), (frames[steps:], 1)] * 2)
                run = pnet._prep_run(data, RunCfg(), test_data=data)
                acc = dict.fromkeys(("synapse", "integrator", "threshold", "plasticity"), 0.0)
                for name in acc:
                    module = getattr(run, name)
                    module.step = _timed(module.step, acc, name)
                for mode in ("train", "eval"):
                    run.set_mode(mode)
                    run.reset_transient()
                    run._datahandler.begin_sample(frames[:20], run.ctx)            ##warm-up
                    for _ in range(20):
                        run.step()
                    acc.update(dict.fromkeys(acc, 0.0))
                    run._datahandler.begin_sample(frames[:steps], run.ctx)
                    rec, n_spikes = run.ctx.neur.Spikes[n_in:], 0
                    t0 = time.perf_counter()
                    for _ in range(steps):
                        run.step()
                        n_spikes += np.count_nonzero(rec)       ##~1 us, inside the timing
                    total = time.perf_counter() - t0
                    rate = n_spikes / steps / (N - n_in)
                    row = {"N": N, "p": p, "drive": drive, "weights_format": fmt, "update_on": update_on, "mode": mode, "n_synapses": int(run.ctx.net.n_synapses),
                           "rec_rate_per_step": round(rate, 4), "step_ms": round(1e3 * total / steps, 4)}
                    for name, t in acc.items():
                        row[f"{name}_ms"] = round(1e3 * t / steps, 4)
                    row["orchestra_ms"] = round(row["step_ms"] - sum(row[f"{n}_ms"] for n in acc), 4)
                    rows.append(row)
                    log(f"N={N:5d} p={p} drive={drive} {fmt:6s} {update_on:12s} {mode:5s}: step {row['step_ms']:8.3f} ms | syn {row['synapse_ms']:.3f} "
                        f"int {row['integrator_ms']:.3f} thr {row['threshold_ms']:.3f} plast {row['plasticity_ms']:.3f} rest {row['orchestra_ms']:.3f} | rate {rate:.3f}")
    return rows


   ######################################################################################
##########################   entry   ####################################################
   ######################################################################################

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _summary(path, micro_rows, system_rows):
    lines = ["# Scratch buffers: buffer (out=) vs fresh array", "", "ratio = fresh / buffer (> 1: buffer faster, < 1: fresh faster)", ""]
    if micro_rows:
        lines += ["## Operations", "", "| section | op | N | p | rate | k | buffer us | fresh us | ratio |", "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['section']} | {r['op']} | {r['N']} | {r['p']} | {r['rate']} | {r['k']} | {r['buffer_us']} | {r['fresh_us']} | {r['ratio_fresh_over_buffer']} |" for r in micro_rows]
        lines.append("")
    if system_rows:
        lines += ["## Real step per module (ms per step)", "", "| N | p | drive | format | update_on | mode | synapses | rate | step | synapse | integrator | threshold | plasticity | rest |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['N']} | {r['p']} | {r['drive']} | {r['weights_format']} | {r['update_on']} | {r['mode']} | {r['n_synapses']} | {r['rec_rate_per_step']} | {r['step_ms']} | "
                  f"{r['synapse_ms']} | {r['integrator_ms']} | {r['threshold_ms']} | {r['plasticity_ms']} | {r['orchestra_ms']} |" for r in system_rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(*, part="both", sizes=None, probs=None, rates=RATES, steps=300, seed=0, tag=None):
    """part "both" | "ops" | "step". Writes micro.csv, system.csv and summary.md into runs/bench_scratch_buffers/<time>[_tag]/."""
    out = RUNS_DIR / "bench_scratch_buffers" / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    import logging; logging.disable(logging.WARNING)
    m = micro(sizes or SIZES, probs or PROBS, rates, seed) if part in ("both", "ops") else []
    _write_csv(out / "micro.csv", m)
    s = system(sizes or (500, 1000, 2000, 4000), probs or (0.05, 0.2), steps, seed) if part in ("both", "step") else []
    _write_csv(out / "system.csv", s)
    _summary(out / "summary.md", m, s)
    print(f"written: {out}")
    return out
