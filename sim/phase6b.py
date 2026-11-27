"""Phase 6B: the offset law fed by reduced filters.

Offset law, k = 2, T = 40, dt = 1e-3, filter step = dt, no hooks. The plant
is the Rouchon-Ralph SME step (sim.plant). Filters:
    exact   the Phase 2 loop (perfect observer), the reference
    full    15 coordinates, Euler step in x coordinates (sim.reduced)
    base9   9 coordinates: the closure for the baseline law
    lam2    2 coordinates: (Lambda, Re c). h = 0 on it, so u = u0.
Starts: the 22 starts of sim.phase2, N = 500 each. The filter starts at the
projection of the true start.

Metrics per (filter, start): P(F > 0.99) at T with a Wilson interval,
P(F < 0.01) at T, and the tracking error |x_hat - x| (median over runs) at
t = 1, 10, and T. Writes <out>/phase6b.npz.

    python -m sim.phase6b --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import _whole_steps, simulate
from .coords import fidelity, to_x
from .laws import Offset
from .params import Plant
from .phase2 import initial_states, wilson
from .plant import plant_step
from .reduced import ReducedFilter, subspace

K, T, DT, N = 2.0, 40.0, 1e-3, 500
FILTERS = ("exact", "full", "base9", "lam2")
SEED = 15001
MARKS = (1.0, 10.0, T)


def job(args):
    fname, sname, rho0, seed = args
    prm = Plant()
    rho = np.repeat(rho0[None], N, axis=0)
    if fname == "exact":
        res = simulate(rho, Offset(K), prm, T=T, dt=DT, seed=seed, record_every=round(T / DT))
        return res.F[-1], np.zeros((len(MARKS), )), 0
    flt = ReducedFilter(subspace(fname, prm), prm)
    law = Offset(K)
    y = flt.encode(to_x(rho))
    rng = np.random.default_rng(seed)
    n_steps = _whole_steps(T, DT, "T")
    marks = {_whole_steps(m, DT, "mark"): i for i, m in enumerate(MARKS)}
    err = np.zeros(len(MARKS))
    for kstep in range(n_steps):
        u = law(flt.decode(y))
        rho, dy = plant_step(rho, u, rng.normal(0.0, np.sqrt(DT), N), DT, prm)
        y = flt.step(y, u, dy, DT)
        if kstep + 1 in marks:
            err[marks[kstep + 1]] = np.median(np.linalg.norm(flt.decode(y) - to_x(rho), axis=1))
    return fidelity(rho), err, flt.mults_per_step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    st = initial_states()
    names = list(st)
    jobs = [(f, s, st[s], SEED + i) for i, s in enumerate(names) for f in FILTERS]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  reduced filters, offset law, K = {K}, T = {T}, N = {N}, dt = {DT}")
    print("filter | dim | mults/step | start      | P(F>0.99) [95% CI] | P(F<0.01) | median |x_hat - x| at t = "
          + ", ".join(f"{m:g}" for m in MARKS))
    shape = (len(names), len(FILTERS))
    p99, ci, low = np.zeros(shape), np.zeros(shape + (2,)), np.zeros(shape)
    errs, mults = np.zeros(shape + (len(MARKS),)), np.zeros(len(FILTERS), dtype=int)
    dims = {f: (15 if f == "exact" else subspace(f, Plant()).shape[0]) for f in FILTERS}
    for (f, s, _, _), (F, err, mu) in zip(jobs, outs):
        i, j = names.index(s), FILTERS.index(f)
        k99 = int(np.sum(F > 0.99))
        p99[i, j], ci[i, j], low[i, j], errs[i, j] = k99 / N, wilson(k99, N), np.mean(F < 0.01), err
        mults[j] = mu
        print(f"{f:6s} | {dims[f]:3d} | {mu:10d} | {s:10s} | {k99 / N:.3f} [{ci[i, j, 0]:.3f}, {ci[i, j, 1]:.3f}] | "
              f"{low[i, j]:.3f} | " + " ".join(f"{e:.1e}" for e in err))
    print("summary over the 21 starts other than phi_minus: P(F>0.99) min to max, mean")
    keep = [i for i, s in enumerate(names) if s != "phi_minus"]
    for j, f in enumerate(FILTERS):
        print(f"  {f:6s} {p99[keep, j].min():.3f} to {p99[keep, j].max():.3f}, mean {p99[keep, j].mean():.3f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "phase6b.npz", filters=np.array(FILTERS), states=np.array(names),
             dims=np.array([dims[f] for f in FILTERS]), mults=mults, p99=p99, ci=ci, low=low,
             err=errs, marks=np.array(MARKS), K=K, T=T, N=N, DT=DT, SEED=SEED, commit=sha)
    print(f"wrote {out / 'phase6b.npz'}")


if __name__ == "__main__":
    main()
