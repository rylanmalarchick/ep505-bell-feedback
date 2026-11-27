"""Offset law at omega = 0 from |00>: split of the fidelity at T = 40.

k = 2, eta = 0.3, N = 200, seed 5. Prints the fraction of runs with
F > 0.99 and with F < 0.01 at T, and for the runs with F < 0.01 the
smallest population of span{Phi-, Psi-} (the face F2).
Report: Section IV-C, the plant sweep.

    python tools/omega0_split.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.closed_loop import simulate  # noqa: E402
from sim.coords import PHI_MINUS  # noqa: E402
from sim.laws import Offset  # noqa: E402
from sim.params import Plant  # noqa: E402

N = 200
PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
rho0 = np.zeros((N, 4, 4), complex)
rho0[:, 0, 0] = 1
r = simulate(rho0, Offset(2.0), Plant(omega=0.0), T=40.0, dt=1e-3, seed=5, record_every=5000)
F = r.F[-1]
face = sum(np.einsum("i,nij,j->n", v.conj(), r.rho_final, v).real for v in (PHI_MINUS, PSI_MINUS))
low = F < 0.01
print(f"F > 0.99: {np.mean(F > 0.99):.3f}  F < 0.01: {np.mean(low):.3f}  between: {np.mean(~low & (F <= 0.99)):.3f}")
print(f"runs with F < 0.01: smallest population of span(Phi-, Psi-) = {face[low].min():.6f}")
