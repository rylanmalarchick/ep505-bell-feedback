"""Shared test fixtures: initial states and random states."""
import numpy as np


def pure(psi, n):
    """n copies of |psi><psi|, normalized."""
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.repeat(np.outer(psi, psi.conj())[None], n, axis=0)


def random_states(n, seed, rank=None):
    """n random density matrices (Ginibre). rank=1 gives pure states."""
    rng = np.random.default_rng(seed)
    r = 4 if rank is None else rank
    G = rng.normal(size=(n, 4, r)) + 1j * rng.normal(size=(n, 4, r))
    rho = G @ G.conj().transpose(0, 2, 1)
    return rho / np.einsum("nii->n", rho).real[:, None, None]
