"""Phase 2: closed loop with a perfect observer.

Runs the baseline and offset laws from 22 initial states:
  "00"          |00>
  "phi_minus"   Phi-, the point xbar' of E+ with F = 0
  "haar_0..9"   10 Haar-random pure states
  "hs_0..9"     10 Hilbert-Schmidt-random mixed states (Ginibre 4 x 4)
N trajectories per (law, state). Writes <out>/phase2.npz and prints, per
run, P(F > 0.99) at T with a 95% Wilson interval, P(on E+ with F < 0.99)
(p1 > 0.99 and F < 0.99), and P(F < 0.01).

Settings (K, T, N, DT, seeds) are module constants and are saved in the
npz. Run:
    python -m sim.phase2 --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import simulate
from .coords import PHI_MINUS
from .laws import Baseline, Offset
from .params import Plant

K = 2.0  # feedback gain, both laws (pilot p2-pilot-gain: best P(F>0.99) at t=40)
T = 40.0  # horizon (pilot: offset law, k = 2, reaches P(F>0.99) = 1.000 by t = 40)
N = 500  # trajectories per (law, state)
DT = 1e-3  # step (test 1.6d)
REC = 50  # record every REC steps
N_PATHS = 20  # sample paths kept per run
SEED_STATES = 2002  # random initial states
SEED_RUNS = 4001  # run i uses SEED_RUNS + i
LAWS = {"baseline": Baseline, "offset": Offset}


def initial_states():
    """Name -> 4 x 4 density matrix. Fixed by SEED_STATES."""
    rng = np.random.default_rng(SEED_STATES)
    out = {"00": np.diag([1, 0, 0, 0]).astype(complex),
           "phi_minus": np.outer(PHI_MINUS, PHI_MINUS.conj())}
    for i in range(10):
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        out[f"haar_{i}"] = np.outer(psi, psi.conj())
    for i in range(10):
        G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        rho = G @ G.conj().T
        out[f"hs_{i}"] = rho / np.trace(rho).real
    return out


def wilson(k, n, z=1.959964):
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    # At k = 0 or k = n one bound equals p exactly. Round-off (~1e-16) can put
    # it on the wrong side of p or outside [0, 1], so clamp to both.
    return max(min(mid - half, p), 0.0), min(max(mid + half, p), 1.0)


def one(job):
    law, name, rho, seed = job
    res = simulate(np.repeat(rho[None], N, axis=0), LAWS[law](K), Plant(), T=T, dt=DT,
                   seed=seed, record_every=REC)
    return dict(t=res.t, F_mean=res.F.mean(1), p99=(res.F > 0.99).mean(1),
                F_paths=res.F[:, :N_PATHS], F_final=res.F[-1], p1_final=res.p1[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    states = initial_states()
    jobs = [(law, name, rho, SEED_RUNS + i) for i, (law, (name, rho)) in
            enumerate((law, item) for law in LAWS for item in states.items())]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(one, jobs))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  K = {K}, T = {T}, N = {N}, dt = {DT}, M = 1, omega = 0.3, eta = 0.3")
    print(f"{'law':9s} {'state':10s} {'F0':>6s} | P(F>0.99) [95% CI] | P(E+, F<0.99) | P(F<0.01) | mean F")
    rows = []
    for (law, name, rho, _), o in zip(jobs, outs):
        k99 = int((o["F_final"] > 0.99).sum())
        lo, hi = wilson(k99, N)
        stuck = np.mean((o["p1_final"] > 0.99) & (o["F_final"] < 0.99))
        low = np.mean(o["F_final"] < 0.01)
        F0 = o["F_mean"][0]
        rows.append((k99 / N, lo, hi, stuck, low))
        print(f"{law:9s} {name:10s} {F0:6.3f} | {k99 / N:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"{stuck:.3f} | {low:.3f} | {o['F_final'].mean():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.out) / "phase2.npz",
             law=np.array([j[0] for j in jobs]), state=np.array([j[1] for j in jobs]),
             seed=np.array([j[3] for j in jobs]), t=outs[0]["t"],
             F_mean=np.stack([o["F_mean"] for o in outs]), p99=np.stack([o["p99"] for o in outs]),
             F_paths=np.stack([o["F_paths"] for o in outs]),
             F_final=np.stack([o["F_final"] for o in outs]),
             p1_final=np.stack([o["p1_final"] for o in outs]),
             summary=np.array(rows),  # columns: p99, lo, hi, P(E+, F<0.99), P(F<0.01)
             K=K, T=T, N=N, DT=DT, SEED_STATES=SEED_STATES, SEED_RUNS=SEED_RUNS, commit=sha)
    print(f"wrote {Path(args.out) / 'phase2.npz'}")


if __name__ == "__main__":
    main()
