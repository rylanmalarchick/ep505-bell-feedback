"""Gain sweep, plant-parameter sweep, and delay ablation at k = 1 and 4.

--part ksweep   offset law, exact observer, no hooks, k in K_GRID, the 21
                starts of Phase 2 other than Phi-, N = 200, T = 40. Records
                P(F > 0.99) at T and the median first time with F > 0.99.
--part params   offset law, k = 2, (omega, eta) on PARAM_GRID (M = 1), the
                same 21 starts, N = 200, T = 80. Records P(F > 0.99) at
                t = 40 and t = 80.
--part delay2   the delay ablation of Phase 6C at the gains 1 and 4: Ts = dt,
                no quantizer, from |00>, N = 200, T = 40. Two loops per tau:
                "perfect" (x_hat = x, sees the delayed plant) and "filter"
                (the Phase 3 filter, fed the undelayed command). The linear
                delayed loop turns unstable between tau = 0.100 and 0.105 at
                k = 1 and between 0.025 and 0.0275 at k = 4 (grid point 0.027)
                (verification/check_sampled.py, growth()).

    python -m sim.phase7r --part ksweep --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import Hooks, _whole_steps, simulate
from .laws import Offset
from .params import Plant
from .phase2 import initial_states, wilson

DT, N = 1e-3, 200
K_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
PARAM_GRID = tuple((om, eta) for om in (0.0, 0.3, 1.0) for eta in (0.1, 0.3, 1.0))
DELAY_TAU = {1.0: (0.04, 0.08, 0.1, 0.11, 0.12, 0.16, 0.2, 0.4),
             4.0: (0.01, 0.02, 0.025, 0.027, 0.03, 0.04, 0.05, 0.1)}
SEED_K, SEED_P, SEED_D = 31001, 32001, 33001
REC = 0.1  # record interval of F, in units of 1/M


def starts():
    """The 21 Phase 2 starts other than Phi-, in a fixed order."""
    return [(name, rho) for name, rho in initial_states().items() if name != "phi_minus"]


def batch(rho):
    return np.repeat(rho[None], N, axis=0)


def first_cross(F, rec):
    """Median over runs of the first recorded time with F > 0.99 (inf if never)."""
    hit = F > 0.99
    t = np.where(hit.any(axis=0), hit.argmax(axis=0) * rec, np.inf)
    return float(np.median(t))


def ksweep_job(args):
    k, rho, seed = args
    res = simulate(batch(rho), Offset(k), Plant(), T=40.0, dt=DT, seed=seed, record_every=round(REC / DT))
    return res.F[-1], first_cross(res.F, REC)


def params_job(args):
    om, eta, rho, seed = args
    res = simulate(batch(rho), Offset(2.0), Plant(omega=om, eta=eta), T=80.0, dt=DT, seed=seed,
                   record_every=round(40.0 / DT))
    return res.F[1], res.F[-1]


def delay_job(args):
    k, loop, tau, seed = args
    rho0 = np.zeros((N, 4, 4), dtype=complex)
    rho0[:, 0, 0] = 1
    res = simulate(rho0, Offset(k), Plant(), T=40.0, dt=DT, seed=seed,
                   rho_hat0=None if loop == "perfect" else rho0.copy(), hooks=Hooks(tau=tau),
                   record_every=round(40.0 / DT))
    return res.F[-1]


def frac(F):
    k99 = int(np.sum(F > 0.99))
    return k99 / N, *wilson(k99, N)


def run_ksweep(out, workers, sha):
    st = starts()
    jobs = [(k, rho, SEED_K + 100 * i + j) for i, k in enumerate(K_GRID) for j, (_, rho) in enumerate(st)]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(ksweep_job, jobs))
    p = np.array([frac(F)[0] for F, _ in outs]).reshape(len(K_GRID), len(st))
    tc = np.array([t for _, t in outs]).reshape(len(K_GRID), len(st))
    print(f"commit {sha}  k sweep, offset law, exact observer, N = {N}, T = 40, 21 starts")
    print(" k    | P(F>0.99) min  mean   max | median first t with F > 0.99: min  median  max")
    for i, k in enumerate(K_GRID):
        print(f" {k:4g} | {p[i].min():.3f} {p[i].mean():.3f} {p[i].max():.3f} | "
              f"{tc[i].min():6.2f} {np.median(tc[i]):6.2f} {tc[i].max():6.2f}")
    np.savez(out / "phase7r_ksweep.npz", k=np.array(K_GRID), p99=p, t_cross=tc,
             starts=np.array([n for n, _ in st]), F_final=np.stack([F for F, _ in outs]), N=N, DT=DT,
             SEED=SEED_K, commit=sha)
    print(f"wrote {out / 'phase7r_ksweep.npz'}")


def run_params(out, workers, sha):
    st = starts()
    jobs = [(om, eta, rho, SEED_P + 100 * i + j) for i, (om, eta) in enumerate(PARAM_GRID)
            for j, (_, rho) in enumerate(st)]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(params_job, jobs))
    p40 = np.array([frac(a)[0] for a, _ in outs]).reshape(len(PARAM_GRID), len(st))
    p80 = np.array([frac(b)[0] for _, b in outs]).reshape(len(PARAM_GRID), len(st))
    print(f"commit {sha}  plant sweep, offset law k = 2, M = 1, N = {N}, 21 starts")
    print(" omega eta  | P(F>0.99) at t = 40: min mean | at t = 80: min mean")
    for i, (om, eta) in enumerate(PARAM_GRID):
        print(f" {om:4g} {eta:4g} | {p40[i].min():.3f} {p40[i].mean():.3f} | {p80[i].min():.3f} {p80[i].mean():.3f}")
    np.savez(out / "phase7r_params.npz", grid=np.array(PARAM_GRID), p40=p40, p80=p80,
             starts=np.array([n for n, _ in st]), N=N, DT=DT, SEED=SEED_P, commit=sha)
    print(f"wrote {out / 'phase7r_params.npz'}")


def run_delay2(out, workers, sha):
    for taus in DELAY_TAU.values():
        for tau in taus:
            _whole_steps(tau, DT, "tau")  # fail before any job starts
    jobs = [(k, lp, tau, SEED_D + 100 * ik + it) for ik, (k, taus) in enumerate(DELAY_TAU.items())
            for it, tau in enumerate(taus) for lp in ("perfect", "filter")]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(delay_job, jobs))
    print(f"commit {sha}  delay ablation, Ts = dt, from |00>, N = {N}, T = 40")
    print(" k  | tau    | loop    | P(F>0.99) [95% CI]")
    rows = []
    for (k, lp, tau, _), F in zip(jobs, outs):
        f, lo, hi = frac(F)
        rows.append((k, tau, lp == "filter", f, lo, hi))
        print(f" {k:2g} | {tau:6g} | {lp:7s} | {f:.3f} [{lo:.3f}, {hi:.3f}]")
    np.savez(out / "phase7r_delay2.npz", rows=np.array(rows), F_final=np.stack(outs), N=N, DT=DT,
             SEED=SEED_D, commit=sha)
    print(f"wrote {out / 'phase7r_delay2.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("ksweep", "params", "delay2"), required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    {"ksweep": run_ksweep, "params": run_params, "delay2": run_delay2}[args.part](out, args.workers, sha)


if __name__ == "__main__":
    main()
