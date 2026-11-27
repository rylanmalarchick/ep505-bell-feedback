"""The k sweep of sim.phase7r on a GPU, to measure the speedup over the CPU run.

Same experiment as `python -m sim.phase7r --part ksweep`: offset law, exact
observer (state feedback every dt), no hooks, k in phase7r.K_GRID, the 21
Phase 2 starts other than Phi-, N = 200 per (k, start), T = 40, dt = 1e-3.
All 8 x 21 x 200 = 33 600 trajectories advance as one batch, each with its
own k. The plant step is sim.torch_loop.TorchLoop.kraus_step, the same
Rouchon-Ralph step as sim.plant. The noise comes from torch, so the runs
differ from the CPU runs path by path; the success fractions agree only
within their binomial intervals. tests/test_gpu_ksweep.py checks the step
against sim.closed_loop.simulate on shared noise.

    python -m sim.gpu_ksweep --out results
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .params import Plant
from .phase7r import DT, K_GRID, N, REC, first_cross, starts
from .torch_loop import TorchLoop


def run_batch(rho0, k, T, dt, prm, device, seed=0, dW=None, rec=REC):
    """Exact-observer offset loop for a batch. rho0: (B, 4, 4) complex, k: (B,).
    dW, if given, is (n_steps, B). Returns F at every rec interval (n_rec + 1, B)
    and the final states."""
    B = rho0.shape[0]
    loop = TorchLoop(B, Ts=dt, tau=0.0, prm=prm, dt=dt, device=device, seed=seed)
    rho = torch.as_tensor(rho0, device=device).to(torch.complex128)
    kt = torch.as_tensor(k, device=device, dtype=torch.float64)[:, None]
    n_steps, every = round(T / dt), round(rec / dt)
    sq = np.sqrt(dt)
    F = [loop.fidelity(rho)]
    for i in range(n_steps):
        x = loop.to_x(rho)
        u = loop.U0 + kt * (x @ loop.H_ROWS.T)
        w = (torch.randn(B, generator=loop.gen, dtype=torch.float64, device=device) * sq if dW is None
             else torch.as_tensor(dW[i], device=device))
        dy = prm.c * loop.p1(rho) * dt + w
        rho = loop.kraus_step(rho, loop.hamiltonian(u), dy, dt)
        if (i + 1) % every == 0:
            F.append(loop.fidelity(rho))
    return torch.stack(F).cpu().numpy(), rho.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = starts()
    rho0 = np.concatenate([np.repeat(rho[None], N, axis=0) for _ in K_GRID for _, rho in st])
    k = np.repeat([kk for kk in K_GRID for _ in st], N).astype(float)
    t0 = time.perf_counter()
    F, _ = run_batch(rho0, k, 40.0, DT, Plant(), args.device, seed=41001)
    wall = time.perf_counter() - t0
    Fr = F.reshape(F.shape[0], len(K_GRID), len(st), N)
    p = np.mean(Fr[-1] > 0.99, axis=-1)
    tc = np.array([[first_cross(Fr[:, i, j], REC) for j in range(len(st))] for i in range(len(K_GRID))])
    dev_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"commit {sha}  GPU k sweep on {dev_name}: {rho0.shape[0]} trajectories, "
          f"{round(40.0 / DT)} steps, wall time {wall:.0f} s")
    print(" k    | P(F>0.99) min  mean   max | median first t with F > 0.99: min  median  max")
    for i, kk in enumerate(K_GRID):
        print(f" {kk:4g} | {p[i].min():.3f} {p[i].mean():.3f} {p[i].max():.3f} | "
              f"{tc[i].min():6.2f} {np.median(tc[i]):6.2f} {tc[i].max():6.2f}")
    np.savez(out / "gpu_ksweep.npz", k=np.array(K_GRID), p99=p, t_cross=tc, wall=wall, device=dev_name,
             starts=np.array([n for n, _ in st]), N=N, DT=DT, commit=sha)
    print(f"wrote {out / 'gpu_ksweep.npz'}")


if __name__ == "__main__":
    main()
