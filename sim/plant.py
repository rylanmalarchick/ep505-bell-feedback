"""Plant: the two-qubit SME in rho form, integrated with the Rouchon-Ralph map.

    d rho = -i [H, rho] dt + D[L] rho dt + sqrt(eta) G[L] rho dW
    dy    = 2 sqrt(eta) Tr(L rho) dt + dW
    H     = omega ZZ + sum_j u_j H_j
    L     = sqrt(M) ZZ (full parity, dy = c p1 dt + dW) or
            sqrt(M) (ZI + IZ) / 2 (half parity), set by Plant.meas

One step (Rouchon and Ralph, PRA 91, 012118, 2015):

    K   = I + (-i H - L^2 / 2) dt + sqrt(eta) L dy + (eta / 2) L^2 (dy^2 - dt)
    rho <- (K rho K^dag + (1 - eta) L rho L dt) / Tr(...)

The map is completely positive before the normalization, so rho stays
positive semidefinite for any dt. Every function takes a batch of N
trajectories: rho (N, 4, 4), u (N, 4), dy and dW (N,).
"""
import numpy as np

from .coords import H_J, HALF, I4, ZZ, p1
from .params import Plant


def hamiltonian(u: np.ndarray, prm: Plant) -> np.ndarray:
    """H = omega ZZ + sum_j u_j H_j. Shape (N, 4) -> (N, 4, 4)."""
    return prm.omega * ZZ + np.einsum("nj,jab->nab", u.astype(complex), H_J)


def kraus_step(rho: np.ndarray, H: np.ndarray, dy: np.ndarray, dt: float,
               prm: Plant) -> np.ndarray:
    """One Rouchon-Ralph step driven by the output increment dy."""
    if prm.meas == "ZZ":
        L = np.sqrt(prm.M) * ZZ
        L2 = prm.M * I4  # L^2, since ZZ^2 = I
    else:
        L = np.sqrt(prm.M) * HALF
        L2 = prm.M * (HALF @ HALF)
    eta = prm.eta
    dyc = dy[:, None, None]
    K = (I4 + (-1j * H - 0.5 * L2) * dt
         + np.sqrt(eta) * L * dyc
         + 0.5 * eta * L2 * (dyc**2 - dt))
    out = K @ rho @ K.conj().transpose(0, 2, 1) + (1 - eta) * dt * (L @ rho @ L)
    out = 0.5 * (out + out.conj().transpose(0, 2, 1))
    return out / np.einsum("nii->n", out).real[:, None, None]


def plant_step(rho: np.ndarray, u: np.ndarray, dW: np.ndarray, dt: float,
               prm: Plant) -> tuple[np.ndarray, np.ndarray]:
    """Advance the true state one step. Returns (rho_next, dy)."""
    if prm.meas == "ZZ":
        dy = prm.c * p1(rho) * dt + dW
    else:
        d = np.einsum("nii->ni", rho).real
        dy = prm.c * (d[:, 0] - d[:, 3]) * dt + dW  # 2 sqrt(eta M) Tr(HALF rho)
    return kraus_step(rho, hamiltonian(u, prm), dy, dt, prm), dy
