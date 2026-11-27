"""Phase 6C: sample-period and delay limits of the offset law.

--part ts (6C.1b): which part of the loop carries the sample-period limit.

Offset law, k = 2, from |00>, T = 40, dt = 1e-3, no quantizer, tau = 0.
Two loops at each sample period Ts:
    hold    perfect observer (x_hat = x at each sample), command held for Ts.
            Only the controller is sampled.
    filter  the filter of Phase 3: one Rouchon-Ralph step of length Ts on
            the summed output, exact start, command held for Ts.
If "hold" keeps P(F > 0.99) high where "filter" fails, the filter step
carries the limit. Result (p6c-ts): both fail at Ts = 0.07, so the hold does.

--part flip (6C.2): the hold limit against the prediction Ts* = 1 / (8 k)
    (verification/check_sampled.py). Perfect observer, no quantizer, tau = 0,
    k in FLIP_K, Ts on a grid around the prediction.
--part delay (6C.2): the delay limit. Ts = dt, no quantizer, k = 2, tau on
    DELAY_TAU. Two loops: "perfect" (x_hat = x, so the controller sees the
    delayed plant) and "filter" (the Phase 3 filter, fed the undelayed
    command). The linear delayed loop turns unstable at tau = 0.05 to 0.055.

    python -m sim.phase6c --part ts --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import Hooks, simulate
from .laws import Offset
from .params import Plant
from .phase2 import wilson

K, T, DT, N = 2.0, 40.0, 1e-3, 200
TS_GRID = (0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5)
LOOPS = ("hold", "filter")
SEED = 14001
FLIP_K = {1.0: (0.1, 0.115, 0.122, 0.128, 0.135, 0.15),
          2.0: (0.05, 0.058, 0.061, 0.064, 0.067, 0.075),
          4.0: (0.025, 0.029, 0.031, 0.032, 0.034, 0.038)}
SEED_FLIP = 16001
DELAY_TAU = (0.02, 0.04, 0.05, 0.055, 0.06, 0.08, 0.1, 0.2)
SEED_DELAY = 17001


def rho00():
    rho0 = np.zeros((N, 4, 4), dtype=complex)
    rho0[:, 0, 0] = 1
    return rho0


def flip_job(args):
    k, Ts, seed = args
    res = simulate(rho00(), Offset(k), Plant(), T=T, dt=DT, seed=seed, hooks=Hooks(Ts=Ts),
                   record_every=round(T / DT))
    return res.F[-1]


def delay_job(args):
    loop, tau, seed = args
    rho0 = rho00()
    res = simulate(rho0, Offset(K), Plant(), T=T, dt=DT, seed=seed,
                   rho_hat0=None if loop == "perfect" else rho0.copy(), hooks=Hooks(tau=tau),
                   record_every=round(T / DT))
    return res.F[-1]


def run_flip(out, workers, sha):
    jobs = [(k, Ts, SEED_FLIP + i) for i, (k, Ts) in
            enumerate((k, Ts) for k, grid in FLIP_K.items() for Ts in grid)]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(flip_job, jobs))
    print(f"commit {sha}  hold limit, perfect observer, T = {T}, N = {N}, dt = {DT}")
    print(" k   | Ts     | Ts / (1/(8k)) | P(F>0.99) [95% CI] | P(F<0.01)")
    rows = []
    for (k, Ts, _), F in zip(jobs, outs):
        k99 = int(np.sum(F > 0.99))
        lo, hi = wilson(k99, N)
        rows.append((k, Ts, k99 / N, lo, hi))
        print(f" {k:3g} | {Ts:6g} | {Ts * 8 * k:13.3f} | {k99 / N:.3f} [{lo:.3f}, {hi:.3f}] | {np.mean(F < 0.01):.3f}")
    np.savez(out / "phase6c_flip.npz", rows=np.array(rows), F_final=np.stack(outs), T=T, N=N, DT=DT,
             SEED=SEED_FLIP, commit=sha)
    print(f"wrote {out / 'phase6c_flip.npz'}")


def run_delay(out, workers, sha):
    jobs = [(lp, tau, SEED_DELAY + i) for i, tau in enumerate(DELAY_TAU) for lp in ("perfect", "filter")]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(delay_job, jobs))
    print(f"commit {sha}  delay limit, k = {K}, Ts = dt, T = {T}, N = {N}, dt = {DT}")
    print(" tau   | loop    | P(F>0.99) [95% CI] | P(F<0.01)")
    rows = []
    for (lp, tau, _), F in zip(jobs, outs):
        k99 = int(np.sum(F > 0.99))
        lo, hi = wilson(k99, N)
        rows.append((tau, lp == "filter", k99 / N, lo, hi))
        print(f" {tau:5g} | {lp:7s} | {k99 / N:.3f} [{lo:.3f}, {hi:.3f}] | {np.mean(F < 0.01):.3f}")
    np.savez(out / "phase6c_delay.npz", rows=np.array(rows), F_final=np.stack(outs), T=T, N=N, DT=DT,
             SEED=SEED_DELAY, commit=sha)
    print(f"wrote {out / 'phase6c_delay.npz'}")


def job(args):
    loop, Ts, seed = args
    rho0 = np.zeros((N, 4, 4), dtype=complex)
    rho0[:, 0, 0] = 1
    res = simulate(rho0, Offset(K), Plant(), T=T, dt=DT, seed=seed,
                   rho_hat0=None if loop == "hold" else rho0.copy(), hooks=Hooks(Ts=Ts),
                   record_every=round(T / DT))
    return res.F[-1], res.est_err[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("ts", "flip", "delay"), default="ts")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if args.part != "ts":
        {"flip": run_flip, "delay": run_delay}[args.part](out, args.workers, sha)
        return
    # The two loops at one Ts share a seed, so they see the same noise.
    jobs = [(lp, Ts, SEED + i) for i, Ts in enumerate(TS_GRID) for lp in LOOPS]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  sample-period diagnosis, K = {K}, T = {T}, N = {N}, dt = {DT}, no quantizer, tau = 0")
    print(" Ts    | loop   | P(F>0.99) [95% CI] | P(F<0.01) | median D(rho_hat, rho) at T")
    p = np.zeros((len(TS_GRID), len(LOOPS)))
    ci = np.zeros((len(TS_GRID), len(LOOPS), 2))
    for (lp, Ts, _), (F, err) in zip(jobs, outs):
        i, j = TS_GRID.index(Ts), LOOPS.index(lp)
        k99 = int(np.sum(F > 0.99))
        p[i, j], ci[i, j] = k99 / N, wilson(k99, N)
        print(f" {Ts:5g} | {lp:6s} | {k99 / N:.3f} [{ci[i, j, 0]:.3f}, {ci[i, j, 1]:.3f}] | "
              f"{np.mean(F < 0.01):.3f} | {np.median(err):.1e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "phase6c_ts.npz", Ts=np.array(TS_GRID), loops=np.array(LOOPS), p99=p, ci=ci,
             F_final=np.stack([o[0] for o in outs]).reshape(len(TS_GRID), len(LOOPS), N),
             K=K, T=T, N=N, DT=DT, SEED=SEED, commit=sha)
    print(f"wrote {out / 'phase6c_ts.npz'}")


if __name__ == "__main__":
    main()
