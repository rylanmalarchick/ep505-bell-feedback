"""Phase 6A.5: rounding noise of the fixed-point filter.

Closed loop as in sim.phase6a (offset law, k = 2, from |00>, Ts = 0.01,
tau = 0, command 12 bits), with a shadow float filter. The shadow filter
gets the same command codes and the same quantized output words as the
fixed-point filter, so the difference between the two is the rounding of
the filter arithmetic and nothing else.

Per sample step and run, the script records
    e    = max over entries |rho_fixed - rho_shadow|, in state LSB 2^-(w-2),
    lmin = smallest eigenvalue of rho_fixed, in state LSB,
    F    = fidelity of the shadow filter.
It reports, per word length, the quantiles of e over all runs and steps
after t = 5, the Hill estimate of the tail index of e (top 1%), the median
of e in bins of F, and the quantiles of lmin. Writes <out>/phase6a_noise.npz.

    python -m sim.phase6a_noise --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import _whole_steps
from .coords import fidelity
from .fixed import FixedFilter
from .observer import observer_step
from .params import Plant
from .plant import plant_step

K, T, DT, TS = 2.0, 40.0, 1e-3, 0.01
W_GRID = (12, 16, 20, 24)
N = 1000
T_BURN = 5.0
SEED = 11001
F_EDGES = np.array([0.0, 0.5, 0.9, 0.99, 0.999, 1.0 + 1e-9])
Q = (0.5, 0.9, 0.99, 0.999, 1.0)


def hill(x: np.ndarray, frac: float = 0.01) -> float:
    """Hill estimate of the tail index from the top frac of x (x > 0)."""
    x = np.sort(x[x > 0])
    k = max(int(frac * x.size), 10)
    top, xk = x[-k:], x[-k - 1]
    return float(1.0 / np.mean(np.log(top / xk)))


def job(args):
    w, seed = args
    prm = Plant()
    flt = FixedFilter(w, TS, prm, K)
    n_steps, n_s = _whole_steps(T, DT, "T"), _whole_steps(TS, DT, "Ts")
    rng = np.random.default_rng(seed)
    rho = np.zeros((N, 4, 4), dtype=complex)
    rho[:, 0, 0] = 1
    R, I = flt.encode(rho)
    shadow = flt.decode(R, I)
    lsb = 2.0 ** -(w - 2)
    n = np.zeros((N, 4), dtype=np.int64)
    dy_acc = np.zeros(N)
    E, LM, FS = [], [], []
    for k in range(n_steps):
        if k % n_s == 0:
            n = flt.codes(R, I)
        rho, dy = plant_step(rho, flt.command(n), rng.normal(0.0, np.sqrt(DT), N), DT, prm)
        dy_acc += dy
        if (k + 1) % n_s == 0:
            dy_w = flt.adc(dy_acc)
            dy_acc = np.zeros(N)
            R, I = flt.step(R, I, n, dy_w)
            shadow = observer_step(shadow, flt.command(n), dy_w / 2.0**flt.FD, TS, prm)
            if (k + 1) * DT > T_BURN:
                est = flt.decode(R, I)
                E.append(np.abs(est - shadow).max(axis=(1, 2)) / lsb)
                LM.append(np.linalg.eigvalsh(est).min(axis=1) / lsb)
                FS.append(fidelity(shadow))
    run_max_e = np.max(np.stack(E), axis=0) * lsb  # per run, absolute units
    run_min_eig = np.min(np.stack(LM), axis=0) * lsb
    E, LM, FS = np.concatenate(E), np.concatenate(LM), np.concatenate(FS)
    qe = np.quantile(E, Q)
    ql = np.quantile(LM, (0.0, 0.001, 0.01, 0.5))
    bins = np.digitize(FS, F_EDGES) - 1
    by_F = np.array([np.median(E[bins == b]) if np.any(bins == b) else np.nan
                     for b in range(len(F_EDGES) - 1)])
    frac_F = np.array([np.mean(bins == b) for b in range(len(F_EDGES) - 1)])
    return dict(qe=qe, ql=ql, hill=hill(E), by_F=by_F, frac_F=frac_F,
                run_max_e=run_max_e, run_min_eig=run_min_eig,
                hist=np.histogram(np.log10(np.maximum(E, 1e-3)), bins=60, range=(-3, 6))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    jobs = [(w, SEED + i) for i, w in enumerate(W_GRID)]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  rounding noise, K = {K}, T = {T}, Ts = {TS}, N = {N}, after t = {T_BURN}")
    print("e = max entry |fixed - shadow float| in LSB")
    print(" w  | e quantiles " + " ".join(f"q{q:g}" for q in Q) + " | Hill index (top 1%)"
          " | min eig in LSB: min, q0.001, q0.01, median")
    for (w, _), o in zip(jobs, outs):
        print(f" {w:2d} | " + " ".join(f"{v:9.2f}" for v in o["qe"]) + f" | {o['hill']:.2f} | "
              + " ".join(f"{v:9.2f}" for v in o["ql"]))
    print("median e by shadow F bin, edges " + " ".join(f"{x:g}" for x in F_EDGES[:-1]) + " 1")
    for (w, _), o in zip(jobs, outs):
        print(f" {w:2d} | " + " ".join(f"{v:8.2f}" for v in o["by_F"])
              + " | fraction of samples: " + " ".join(f"{v:.3f}" for v in o["frac_F"]))
    print("runs (of N) whose largest error, in absolute units, exceeds a level; runs with min eig < -0.01")
    for (w, _), o in zip(jobs, outs):
        print(f" {w:2d} | " + " ".join(f">{lv:g}: {int(np.sum(o['run_max_e'] > lv)):4d}"
                                        for lv in (1e-3, 1e-2, 1e-1, 1.0))
              + f" | {int(np.sum(o['run_min_eig'] < -0.01))}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "phase6a_noise.npz",
             run_max_e=np.stack([o["run_max_e"] for o in outs]),
             run_min_eig=np.stack([o["run_min_eig"] for o in outs]), w=np.array(W_GRID), q=np.array(Q),
             qe=np.stack([o["qe"] for o in outs]), ql=np.stack([o["ql"] for o in outs]),
             hill=np.array([o["hill"] for o in outs]), by_F=np.stack([o["by_F"] for o in outs]),
             frac_F=np.stack([o["frac_F"] for o in outs]), F_edges=F_EDGES,
             hist=np.stack([o["hist"] for o in outs]),
             K=K, T=T, TS=TS, N=N, T_BURN=T_BURN, SEED=SEED, commit=sha)
    print(f"wrote {out / 'phase6a_noise.npz'}")


if __name__ == "__main__":
    main()
