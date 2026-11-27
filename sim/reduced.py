"""Reduced filters in x coordinates.

A reduced filter keeps d coordinates y = Pi x, where the rows of Pi (d x 15)
are an orthonormal basis of a subspace V of functionals. It estimates
x_hat = Pi^T y, so the coordinates outside V are set to zero. Projecting
the observer equation of the report (eq. "observer") onto V gives

    dy = (Ar + sum_j u_j Br_j) y dt + c (Pi e1 + Sr y - p1_hat y) (dy_out - c p1_hat dt),
    Ar = Pi A0 Pi^T,  Br_j = Pi B_j Pi^T,  Sr = Pi S~ Pi^T,  p1_hat = e1^T Pi^T y,

with S~ = diag(S, 0). For V = R^15 this is the full observer. The step is
Euler-Maruyama, so positivity is not kept. With V = R^15 it shows the
error of the Euler step alone.

Subspaces (verification/check_reduction.py computes their dimension exactly):
    full       R^15
    base9      the closure for the baseline law (dimension 9). It holds p1,
               F, and h. Exact for the baseline law, not for the offset law.
    lam2       span{p1, F}: the (Lambda, Re c) closure for u = 0.
               h is 0 on it, so the offset law fed by it applies u = u0.
"""
import numpy as np

from verification.coords_spec import P_LAB

from .coords import B_MATS, H_ROWS, X_BAR, a0_matrix
from .params import Plant

NP = len(P_LAB)
E1 = np.eye(15)[0]


def s_tilde() -> np.ndarray:
    S = np.zeros((15, 15))
    for a, b, s in [("IZ", "ZI", 1), ("XX", "YY", -1), ("XY", "YX", 1)]:
        i, j = P_LAB.index(a), P_LAB.index(b)
        S[i, j] = S[j, i] = s
    return S


S_T = s_tilde()


def closure_basis(start: np.ndarray, mats, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal rows spanning the smallest subspace that contains the rows of
    start and is closed under v -> A^T v for A in mats and under v -> S~ v."""
    basis = np.zeros((0, 15))

    def add(v):
        nonlocal basis
        r = v - basis.T @ (basis @ v)
        n = np.linalg.norm(r)
        if n > tol * max(1.0, np.linalg.norm(v)):
            basis = np.vstack([basis, r / n])

    for v in start:
        add(v)
    i = 0
    while i < basis.shape[0]:  # at most 15 rows
        v = basis[i]
        for A in mats:
            add(A.T @ v)
        add(S_T @ v)
        i += 1
    return basis


def subspace(name: str, prm: Plant) -> np.ndarray:
    if name == "full":
        return np.eye(15)
    if name == "lam2":
        return closure_basis(np.array([E1, X_BAR]), [])
    if name == "base9":
        A0 = a0_matrix(prm)
        return closure_basis(np.vstack([E1, X_BAR, H_ROWS]),
                             [A0, B_MATS[0] + B_MATS[2], B_MATS[1] - B_MATS[3]])
    raise ValueError(name)


class ReducedFilter:
    def __init__(self, Pi: np.ndarray, prm: Plant):
        self.Pi, self.prm = Pi, prm
        self.d = Pi.shape[0]
        self.Ar = Pi @ a0_matrix(prm) @ Pi.T
        self.Br = np.einsum("ia,jab,kb->jik", Pi, B_MATS, Pi)
        self.Sr = Pi @ S_T @ Pi.T
        self.e1r = Pi @ E1
        self.p1row = Pi @ E1  # p1_hat = e1^T Pi^T y = (Pi e1) . y

    def encode(self, x: np.ndarray) -> np.ndarray:
        return x @ self.Pi.T

    def decode(self, y: np.ndarray) -> np.ndarray:
        return y @ self.Pi

    def step(self, y: np.ndarray, u: np.ndarray, dy: np.ndarray, dt: float) -> np.ndarray:
        c = self.prm.c
        p1 = y @ self.p1row
        M = self.Ar + np.einsum("nj,jik->nik", u, self.Br)
        drift = np.einsum("nik,nk->ni", M, y)
        gain = c * (self.e1r + y @ self.Sr.T - p1[:, None] * y)
        return y + drift * dt + gain * (dy - c * p1 * dt)[:, None]

    def mults_per_step(self) -> int:
        """Real multiplies per step, dense count: the drift matrix (4 d^2),
        its product with y (d^2), the gain (d^2 + d), and the update (2 d)."""
        d = self.d
        return 6 * d * d + 3 * d
