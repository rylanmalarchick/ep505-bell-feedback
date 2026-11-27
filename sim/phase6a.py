"""Phase 6A: closed loop with the fixed-point filter.

Offset law, k = 2, from |00>, T = 40, dt = 1e-3 (Phase 2 settings). The
filter and the law run in w-bit fixed point (sim/fixed.py) at the sample
period Ts. The command is a WC-bit code over the full scale UMAX. The
plant is in floating point. Each (Ts, tau) point also runs the float
filter with the same command quantizer and the same seed, as the
reference.

Grid: W_GRID at POINTS, N trajectories each. Metric: P(F > 0.99) at T
with a 95% Wilson interval. Writes <out>/phase6a.npz.

    python -m sim.phase6a --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import Hooks, simulate
from .fixed import FixedFilter, simulate_fixed
from .laws import Offset
from .params import Plant
from .phase2 import wilson

K, T, DT = 2.0, 40.0, 1e-3
W_GRID = (8, 10, 12, 14, 16, 20, 24)
POINTS = ((0.001, 0.0), (0.01, 0.1), (0.05, 0.5))
WC, UMAX = 12, 4.0
N = 200
SEED = 10001  # point i uses SEED + i for every word length and for the float run


def rho00():
    rho = np.zeros((N, 4, 4), dtype=complex)
    rho[:, 0, 0] = 1
    return rho


def job(args):
    Ts, tau, w, seed = args
    prm = Plant()
    if w is None:  # float filter, same command quantizer
        res = simulate(rho00(), Offset(K), prm, T=T, dt=DT, seed=seed, rho_hat0=rho00(),
                       hooks=Hooks(Ts=Ts, tau=tau, w=WC, umax=UMAX), record_every=round(T / DT))
        return res.F[-1], res.est_err[-1]
    out = simulate_fixed(rho00(), FixedFilter(w, Ts, prm, K, wc=WC, umax=UMAX), prm,
                         T=T, dt=DT, seed=seed, tau=tau)
    return out["F"], out["est_err"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    words = (None,) + W_GRID
    jobs = [(Ts, tau, w, SEED + ip) for ip, (Ts, tau) in enumerate(POINTS) for w in words]
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(job, jobs))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  fixed-point filter, K = {K}, T = {T}, N = {N}, dt = {DT}, "
          f"command {WC} bits over {UMAX}")
    shape = (len(POINTS), len(words))
    k99 = np.array([int(np.sum(F > 0.99)) for F, _ in outs]).reshape(shape)
    ci = np.array([wilson(int(k), N) for k in k99.ravel()]).reshape(*shape, 2)
    err = np.array([np.median(e) for _, e in outs]).reshape(shape)
    for ip, (Ts, tau) in enumerate(POINTS):
        print(f"Ts = {Ts:g}, tau = {tau:g}:")
        for iw, w in enumerate(words):
            name = "float" if w is None else f"w = {w:2d}"
            print(f"  {name:6s} | P(F>0.99) {k99[ip, iw] / N:.3f} "
                  f"[{ci[ip, iw, 0]:.3f}, {ci[ip, iw, 1]:.3f}] | median D(rho_hat, rho) at T {err[ip, iw]:.1e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "phase6a.npz", w=np.array(W_GRID), points=np.array(POINTS),
             p99=k99[:, 1:] / N, ci=ci[:, 1:], p99_float=k99[:, 0] / N, ci_float=ci[:, 0],
             err_median=err[:, 1:], err_median_float=err[:, 0],
             F_final=np.stack([F for F, _ in outs]).reshape(*shape, N),
             K=K, T=T, N=N, DT=DT, WC=WC, UMAX=UMAX, SEED=SEED, commit=sha)
    print(f"wrote {out / 'phase6a.npz'}")


if __name__ == "__main__":
    main()
