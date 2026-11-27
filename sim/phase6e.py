"""Phase 6E: half-parity measurement, target Psi+.

Model: L = sqrt(M) (ZI + IZ) / 2, H0 = omega ZZ, four local drives, the
Phase 2 plant parameters. Target Psi+ = (|01> + |10>) / sqrt 2. Laws:
    baseline   u = k h(x)
    offset_x   u = (1, 0, -1, 0) + k h(x)    (X1 - X2)
    offset_y   u = (0, 1, 0, -1) + k h(x)    (Y1 - Y2, the offset of MvH 2007 Sec. 5)
Perfect observer, dt = 1e-3.

--part pilot  k in K_PILOT, T = 40, N = 200, from |01> and haar_0.
              Result: mean F still rising at T = 40, about the
              same for every k. So the main run uses T = 200.
--part main   k = K, T = 200, N = 500, from 23 starts: |01> (F = 1/2),
              |00> and Psi- (both F = 0), the 10 Haar and 10
              Hilbert-Schmidt states of sim.phase2 (same seed).
Writes <out>/phase6e_<part>.npz.

    python -m sim.phase6e --part pilot --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import simulate
from .coords import PSI_MINUS, PSI_PLUS, xbar_of
from .laws import Baseline, Offset
from .params import Plant
from .phase2 import initial_states, wilson

K = 2.0  # pilot: no k in 0.5 to 8 is clearly better; 2 matches Phase 2
T_PART = {"pilot": 40.0, "main": 200.0}
DT, REC = 1e-3, 50
N_PILOT, N_MAIN, N_PATHS = 200, 500, 20
K_PILOT = (0.5, 1.0, 2.0, 4.0, 8.0)
SEED_PILOT, SEED_MAIN = 12001, 13001
XBAR = xbar_of(PSI_PLUS)
U0 = {"offset_x": np.array([1.0, 0.0, -1.0, 0.0]), "offset_y": np.array([0.0, 1.0, 0.0, -1.0])}
PRM = Plant(meas="half")


def law(name, k):
    if name == "baseline":
        return Baseline(k, XBAR)
    return Offset(k, U0[name], XBAR)


def starts():
    out = {"01": np.diag([0, 1, 0, 0]).astype(complex),
           "00": np.diag([1, 0, 0, 0]).astype(complex),
           "psi_minus": np.outer(PSI_MINUS, PSI_MINUS.conj())}
    for name, rho in initial_states().items():
        if name.startswith(("haar", "hs")):
            out[name] = rho
    return out


def job(args):
    lname, k, sname, rho, n, seed, T = args
    res = simulate(np.repeat(rho[None], n, axis=0), law(lname, k), PRM, T=T, dt=DT, seed=seed,
                   record_every=REC, target=PSI_PLUS)
    return dict(t=res.t, F_mean=res.F.mean(1), F_paths=res.F[:, :N_PATHS], F_final=res.F[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("pilot", "main"), required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    st = starts()
    if args.part == "pilot":
        n, seed0 = N_PILOT, SEED_PILOT
        combos = [(ln, k, s) for ln in U0 for k in K_PILOT for s in ("01", "haar_0")]
    else:
        n, seed0 = N_MAIN, SEED_MAIN
        combos = [(ln, K, s) for ln in ("baseline", "offset_x", "offset_y") for s in st]
    T = T_PART[args.part]
    jobs = [(ln, k, s, st[s], n, seed0 + i, T) for i, (ln, k, s) in enumerate(combos)]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  half parity, target Psi+, part {args.part}, T = {T}, N = {n}, dt = {DT}, "
          f"M = {PRM.M}, omega = {PRM.omega}, eta = {PRM.eta}")
    print(f"{'law':9s} {'k':>5s} {'start':10s} {'F0':>6s} | P(F>0.99) [95% CI] | P(F<0.01) | mean F")
    rows = []
    for (ln, k, s, rho, _, _, _), o in zip(jobs, outs):
        kk = int((o["F_final"] > 0.99).sum())
        lo, hi = wilson(kk, n)
        rows.append((kk / n, lo, hi, np.mean(o["F_final"] < 0.01)))
        print(f"{ln:9s} {k:5.2f} {s:10s} {o['F_mean'][0]:6.3f} | {kk / n:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"{np.mean(o['F_final'] < 0.01):.3f} | {o['F_final'].mean():.4f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"phase6e_{args.part}.npz", law=np.array([j[0] for j in jobs]),
             k=np.array([j[1] for j in jobs]), state=np.array([j[2] for j in jobs]),
             seed=np.array([j[5] for j in jobs]), t=outs[0]["t"],
             F_mean=np.stack([o["F_mean"] for o in outs]), F_paths=np.stack([o["F_paths"] for o in outs]),
             F_final=np.stack([o["F_final"] for o in outs]),
             summary=np.array(rows),  # columns: p99, lo, hi, P(F<0.01)
             T=T, N=n, DT=DT, commit=sha)
    print(f"wrote {out / f'phase6e_{args.part}.npz'}")


if __name__ == "__main__":
    main()
