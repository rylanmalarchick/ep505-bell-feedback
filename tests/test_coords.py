"""sim/coords.py against the control model in the report."""
import numpy as np

from sim import coords as C
from sim.params import Plant

from .helpers import random_states

PRM = Plant()


def lindblad_drift(rho, u, prm):
    """SME drift -i[H, rho] + D[L] rho, written out directly."""
    H = prm.omega * C.ZZ + np.einsum("j,jab->ab", u, C.H_J)
    return -1j * (H @ rho - rho @ H) + prm.M * (C.ZZ @ rho @ C.ZZ - rho)


def test_round_trip():
    rho = random_states(50, seed=1)
    assert np.allclose(C.to_rho(C.to_x(rho)), rho, atol=1e-14, rtol=0)


def test_targets_are_bell_states():
    for x, psi in [(C.X_BAR, C.PHI_PLUS), (C.X_BAD, C.PHI_MINUS)]:
        assert np.allclose(C.to_rho(x[None])[0], np.outer(psi, psi.conj()), atol=1e-15)


def test_fidelity_is_affine_in_x():
    rho = random_states(50, seed=2)
    assert np.allclose(C.fidelity(rho), 0.25 * (1 + C.to_x(rho) @ C.X_BAR), atol=1e-14)


def test_p1_is_first_coordinate():
    rho = random_states(50, seed=3)
    assert np.allclose(C.p1(rho), C.to_x(rho)[:, 0], atol=1e-14)


def test_a0_block_structure():
    A0 = C.a0_matrix(PRM)
    R = np.array([[-2 * PRM.M, -2 * PRM.omega], [2 * PRM.omega, -2 * PRM.M]])
    expect = np.zeros((15, 15))
    expect[7:, 7:] = np.kron(np.eye(4), R)
    assert np.allclose(A0, expect, atol=1e-14)


def test_input_matrices_couple_p_and_q_only():
    for B in C.B_MATS:
        assert np.allclose(B, -B.T, atol=1e-15)
        assert np.allclose(B[:7, :7], 0, atol=1e-15)
        assert np.allclose(B[7:, 7:], 0, atol=1e-15)


def test_drift_is_bilinear_in_x_and_u():
    """to_x(SME drift) = (A0 + sum_j u_j B_j) x on random states and inputs."""
    rng = np.random.default_rng(4)
    rho = random_states(20, seed=4)
    A0 = C.a0_matrix(PRM)
    for r in rho:
        u = rng.normal(size=4)
        lhs = C.to_x(lindblad_drift(r, u, PRM)[None])[0]
        rhs = (A0 + np.einsum("j,jab->ab", u, C.B_MATS)) @ C.to_x(r[None])[0]
        assert np.allclose(lhs, rhs, atol=1e-13)


def test_h_matches_proposal_v2_form():
    """h_j(x) / 4 = -i Tr(H_j [rho, rho_bar]) (check_control_model.py, part 8)."""
    rho = random_states(30, seed=5)
    rho_bar = np.outer(C.PHI_PLUS, C.PHI_PLUS.conj())
    comm = rho @ rho_bar - rho_bar @ rho
    g = (-1j * np.einsum("jab,nba->nj", C.H_J, comm)).real
    assert np.allclose(C.h(C.to_x(rho)) / 4, g, atol=1e-14)


def test_offset_does_not_move_target():
    """sum_j u0_j B_j xbar = 0 (part 10a of the sympy check)."""
    assert np.allclose(np.einsum("j,jab,b->a", C.U_0, C.B_MATS, C.X_BAR), 0, atol=1e-15)
