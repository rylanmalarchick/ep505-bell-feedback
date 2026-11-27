"""Escape from the invariant face span{Phi-, Psi-} under the offset law.

The offset law keeps every state supported on span{Phi-, Psi-} on that face,
where F = 0 (verification/check_control_model.py, section 10b). This script
starts next to the face and measures how often the loop leaves it:

    rho0 = (1 - eps) |Phi-><Phi-| + eps I / 4,   F(0) = eps / 4.

eps = 0 is the exact start Phi-. There, only round-off can seed the escape.
Runs K_GRID x EPS_GRID, N trajectories each, perfect observer, T, dt as in
Phase 2. Prints P(F > 0.99) at T with a 95% Wilson interval, P(F < 0.01) at
T, and the median first time with F > 0.5. Writes <out>/escape.npz.

    python -m sim.escape --out results
"""
import argparse
import itertools
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import simulate
from .coords import I4, PHI_MINUS
from .laws import Offset
from .params import Plant
from .phase2 import wilson

T, DT, N, REC = 40.0, 1e-3, 500, 50
K_GRID = (0.25, 0.5, 1.0, 2.0)
EPS_GRID = (0.0, 1e-12, 1e-8, 1e-4, 1e-2)
SEED = 9001


def job(args):
    k, eps, seed = args
    rho0 = (1 - eps) * np.outer(PHI_MINUS, PHI_MINUS.conj()) + eps * I4 / 4
    res = simulate(np.repeat(rho0[None], N, axis=0), Offset(k), Plant(), T=T, dt=DT,
                   seed=seed, record_every=REC)
    above = res.F > 0.5
    # First record time with F > 0.5. inf for a run that never gets there.
    t_half = np.where(above.any(axis=0), res.t[above.argmax(axis=0)], np.inf)
    return res.F[-1], t_half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    grid = list(itertools.product(K_GRID, EPS_GRID))
    jobs = [(k, eps, SEED + i) for i, (k, eps) in enumerate(grid)]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  escape, offset law, T = {T}, N = {N}, dt = {DT}")
    print("  k    |  eps   | P(F>0.99) at T [95% CI] | P(F<0.01) at T | median first t with F > 0.5")
    for (k, eps, _), (F_final, t_half) in zip(jobs, outs):
        k99 = int((F_final > 0.99).sum())
        lo, hi = wilson(k99, N)
        print(f" {k:5.2f} | {eps:6.0e} | {k99 / N:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"{np.mean(F_final < 0.01):.3f} | {np.median(t_half):.2f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = (len(K_GRID), len(EPS_GRID), N)
    np.savez(out / "escape.npz", k=np.array(K_GRID), eps=np.array(EPS_GRID),
             F_final=np.stack([o[0] for o in outs]).reshape(shape),
             t_half=np.stack([o[1] for o in outs]).reshape(shape),
             T=T, N=N, DT=DT, SEED=SEED, commit=sha)
    print(f"wrote {out / 'escape.npz'}")


if __name__ == "__main__":
    main()
