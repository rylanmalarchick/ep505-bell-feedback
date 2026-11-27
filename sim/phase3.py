"""Phase 3: observer mismatch and the hardware sweep.

Offset law, k = 2, from |00>, T = 40, dt = 1e-3 (Phase 2 settings).

--part observer  (Figure 3)
    Filter started from rho_hat0 = (1 - s)|00><00| + s|11><11|, so the
    trace distance D(rho_hat0, rho0) = s. |00> and |11> have the same
    parity, so the output cannot tell them apart at t = 0.
    s in S_GRID (s = 0 is the exact start), N = N_OBS. Ts = dt.
    Writes <out>/phase3_observer.npz.

--part sweep  (Figure 4)
    Filter from the exact start, run at the sample period Ts with the
    summed output. Grid TS_GRID x TAU_GRID x W_GRID, full scale UMAX,
    N = N_SWEEP per cell. Metric: P(F > 0.99) at T.
    Writes <out>/phase3_sweep.npz and the wall time.

--part wline  (Figure 5)
    Same as the sweep, w in WLINE_W at the (Ts, tau) points WLINE_POINTS.
    Writes <out>/phase3_wline.npz.

    python -m sim.phase3 --part observer --out results
    python -m sim.phase3 --part sweep --out results
    python -m sim.phase3 --part wline --out results
"""
import argparse
import itertools
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import Hooks, simulate
from .laws import Offset
from .params import Plant
from .phase2 import wilson

K, T, DT = 2.0, 40.0, 1e-3

S_GRID = (0.0, 0.1, 0.5, 0.9)
N_OBS, REC_OBS = 500, 50
SEED_OBS = 6001

# Grid from a pilot run: the loop holds up to Ts = 0.05 and fails at
# Ts = 0.1; delay degrades it slowly up to tau = 1.
TS_GRID = (0.001, 0.01, 0.03, 0.05, 0.07, 0.1)
TAU_GRID = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0)
W_GRID = (6, 8, 10, 12)
UMAX = 4.0
N_SWEEP = 200
SEED_SWEEP = 7001

# w line for Figure 5: word length at two fixed (Ts, tau) points.
WLINE_W = (2, 3, 4, 5, 6, 8, 10, 12)
WLINE_POINTS = ((0.001, 0.0), (0.05, 0.5))
SEED_WLINE = 8001


def ket(i):
    v = np.zeros(4, dtype=complex)
    v[i] = 1
    return np.outer(v, v)


def observer_job(job):
    s, seed = job
    rho0 = np.repeat(ket(0)[None], N_OBS, axis=0)
    rho_hat0 = np.repeat(((1 - s) * ket(0) + s * ket(3))[None], N_OBS, axis=0)
    res = simulate(rho0, Offset(K), Plant(), T=T, dt=DT, seed=seed, rho_hat0=rho_hat0,
                   record_every=REC_OBS)
    return dict(t=res.t, F_mean=res.F.mean(1), p99=(res.F > 0.99).mean(1),
                err_mean=res.est_err.mean(1), err_q90=np.quantile(res.est_err, 0.9, axis=1),
                F_final=res.F[-1])


def sweep_job(job):
    Ts, tau, w, seed = job
    rho0 = np.repeat(ket(0)[None], N_SWEEP, axis=0)
    res = simulate(rho0, Offset(K), Plant(), T=T, dt=DT, seed=seed, rho_hat0=rho0,
                   hooks=Hooks(Ts=Ts, tau=tau, w=w, umax=UMAX), record_every=round(T / DT))
    return res.F[-1]


def commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def run_observer(out, workers):
    jobs = [(s, SEED_OBS + i) for i, s in enumerate(S_GRID)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(observer_job, jobs))
    print(f"commit {commit()}  observer mismatch, K = {K}, T = {T}, N = {N_OBS}, dt = {DT}")
    times = [x for x in (0, 5, 10, 20, 40) if x <= T]
    print(" s   | P(F>0.99) at T [95% CI] | mean F at T | mean D at t = " + ", ".join(f"{x:g}" for x in times))
    t = res[0]["t"]
    marks = [int(round(x / (DT * REC_OBS))) for x in times]
    for (s, _), r in zip(jobs, res):
        k99 = int((r["F_final"] > 0.99).sum())
        lo, hi = wilson(k99, N_OBS)
        print(f" {s:.1f} | {k99 / N_OBS:.3f} [{lo:.3f}, {hi:.3f}] | {r['F_final'].mean():.4f} | "
              + " ".join(f"{r['err_mean'][m]:.2e}" for m in marks))
    np.savez(out / "phase3_observer.npz", s=np.array(S_GRID), t=t,
             F_mean=np.stack([r["F_mean"] for r in res]), p99=np.stack([r["p99"] for r in res]),
             err_mean=np.stack([r["err_mean"] for r in res]),
             err_q90=np.stack([r["err_q90"] for r in res]),
             F_final=np.stack([r["F_final"] for r in res]),
             K=K, T=T, N=N_OBS, DT=DT, SEED=SEED_OBS, commit=commit())
    print(f"wrote {out / 'phase3_observer.npz'}")


def run_sweep(out, workers):
    if TS_GRID is None or TAU_GRID is None:
        raise SystemExit("TS_GRID and TAU_GRID are not set.")
    grid = list(itertools.product(TS_GRID, TAU_GRID, W_GRID))
    jobs = [(Ts, tau, w, SEED_SWEEP + i) for i, (Ts, tau, w) in enumerate(grid)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        F_final = list(ex.map(sweep_job, jobs))
    wall = time.time() - t0
    p99 = np.array([np.mean(f > 0.99) for f in F_final]).reshape(len(TS_GRID), len(TAU_GRID), len(W_GRID))
    ci = np.array([wilson(int(np.sum(f > 0.99)), N_SWEEP) for f in F_final])
    ci = ci.reshape(len(TS_GRID), len(TAU_GRID), len(W_GRID), 2)
    print(f"commit {commit()}  sweep, K = {K}, T = {T}, N = {N_SWEEP} per cell, dt = {DT}, "
          f"UMAX = {UMAX}, {len(jobs)} cells, {workers} workers, wall time {wall:.0f} s")
    for iw, w in enumerate(W_GRID):
        print(f"\nw = {w}: P(F > 0.99) at T. Rows Ts, columns tau = {TAU_GRID}")
        for its, Ts in enumerate(TS_GRID):
            print(f"  Ts = {Ts:6g} | " + " ".join(f"{p99[its, it, iw]:.3f}" for it in range(len(TAU_GRID))))
    np.savez(out / "phase3_sweep.npz", Ts=np.array(TS_GRID), tau=np.array(TAU_GRID),
             w=np.array(W_GRID), p99=p99, ci=ci, F_final=np.stack(F_final),
             K=K, T=T, N=N_SWEEP, DT=DT, UMAX=UMAX, SEED=SEED_SWEEP, wall_s=wall, commit=commit())
    print(f"wrote {out / 'phase3_sweep.npz'}")


def run_wline(out, workers):
    grid = [(Ts, tau, w) for Ts, tau in WLINE_POINTS for w in WLINE_W]
    jobs = [(Ts, tau, w, SEED_WLINE + i) for i, (Ts, tau, w) in enumerate(grid)]
    with ProcessPoolExecutor(workers) as ex:
        F_final = list(ex.map(sweep_job, jobs))
    k99 = np.array([int(np.sum(f > 0.99)) for f in F_final])
    ci = np.array([wilson(k, N_SWEEP) for k in k99])
    shape = (len(WLINE_POINTS), len(WLINE_W))
    print(f"commit {commit()}  w line, K = {K}, T = {T}, N = {N_SWEEP}, dt = {DT}, UMAX = {UMAX}")
    for ip, (Ts, tau) in enumerate(WLINE_POINTS):
        print(f"Ts = {Ts:g}, tau = {tau:g}:")
        for iw, w in enumerate(WLINE_W):
            i = ip * len(WLINE_W) + iw
            f = F_final[i]
            print(f"  w = {w:2d} | P(F>0.99) {k99[i] / N_SWEEP:.3f} [{ci[i, 0]:.3f}, {ci[i, 1]:.3f}]"
                  f" | P(F<0.01) {np.mean(f < 0.01):.3f} | mean F {f.mean():.4f}")
    np.savez(out / "phase3_wline.npz", w=np.array(WLINE_W), points=np.array(WLINE_POINTS),
             p99=(k99 / N_SWEEP).reshape(shape), ci=ci.reshape(*shape, 2),
             F_final=np.stack(F_final).reshape(*shape, N_SWEEP),
             K=K, T=T, N=N_SWEEP, DT=DT, UMAX=UMAX, SEED=SEED_WLINE, commit=commit())
    print(f"wrote {out / 'phase3_wline.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("observer", "sweep", "wline"), required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    {"observer": run_observer, "sweep": run_sweep, "wline": run_wline}[args.part](out, args.workers)


if __name__ == "__main__":
    main()
