"""Median fidelity of the offset loop from the incoherent start next to |Phi->.

Start (1 - eps)|Phi-><Phi-| + eps I/4 with eps = 1e-4, offset law k = 0.25,
N = 40, T = 10, seed 9001. Prints the median of F at t = 0, 1, ..., 10.
Report: Section IV-C, the decay of the population without coherence.

    python tools/escape_population.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.closed_loop import simulate  # noqa: E402
from sim.coords import I4, PHI_MINUS  # noqa: E402
from sim.laws import Offset  # noqa: E402
from sim.params import Plant  # noqa: E402

EPS, N = 1e-4, 40
rho0 = np.repeat(((1 - EPS) * np.outer(PHI_MINUS, PHI_MINUS.conj()) + EPS * I4 / 4)[None], N, 0)
r = simulate(rho0, Offset(0.25), Plant(), T=10.0, dt=1e-3, seed=9001, record_every=1000)
print("median F at t = 0..10:", " ".join(f"{x:.1e}" for x in np.median(r.F, 1)))
