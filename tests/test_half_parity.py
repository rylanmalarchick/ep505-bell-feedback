"""Half-parity model, L = sqrt(M) (ZI + IZ) / 2."""
import numpy as np
import pytest

from sim import coords as C
from sim.closed_loop import simulate
from sim.laws import Offset
from sim.params import Plant

from .helpers import pure, random_states

HALF = Plant(meas="half")
XBAR_PSI = C.xbar_of(C.PSI_PLUS)
U0_X = np.array([1.0, 0.0, -1.0, 0.0])  # X1 - X2
U0_Y = np.array([0.0, 1.0, 0.0, -1.0])  # Y1 - Y2


def test_bad_measurement_name_is_rejected():
    with pytest.raises(ValueError):
        Plant(meas="xx")


def test_trace_and_positivity():
    rho0 = np.concatenate([random_states(32, seed=40, rank=1), random_states(32, seed=41)])
    for u0 in (U0_X, U0_Y):
        res = simulate(rho0, Offset(2.0, u0, XBAR_PSI), HALF, T=3.0, dt=1e-3, seed=42,
                       check_physical=True, target=C.PSI_PLUS)
        assert res.max_trace_err < 1e-12
        assert res.min_eig > -1e-12


@pytest.mark.parametrize("u0", [U0_X, U0_Y])
def test_target_is_a_fixed_point(u0):
    """Psi+ is an eigenvector of L, of ZZ, and of the offset Hamiltonian, and
    h = 0 there. The offset loop started at Psi+ stays at F = 1."""
    res = simulate(pure(C.PSI_PLUS, 8), Offset(2.0, u0, XBAR_PSI), HALF, T=3.0, dt=1e-3,
                   seed=43, target=C.PSI_PLUS)
    assert np.abs(res.F - 1.0).max() < 1e-12


def _complement_states(psi, n, seed):
    """n random density matrices supported on the complement of psi (F = 0)."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(np.column_stack([psi, rng.normal(size=(4, 3))]))[0][:, 1:]
    G = rng.normal(size=(n, 3, 3)) + 1j * rng.normal(size=(n, 3, 3))
    R = G @ G.conj().transpose(0, 2, 1)
    return Q @ (R / np.einsum("nii->n", R).real[:, None, None]) @ Q.conj().T


@pytest.mark.parametrize("prm, psi, u0", [
    (HALF, C.PSI_PLUS, U0_X), (HALF, C.PSI_PLUS, U0_Y), (Plant(), C.PHI_PLUS, C.U_0)])
def test_zero_fidelity_set_is_invariant_one_step(prm, psi, u0):
    """One step of the offset loop maps a state with rho psi = 0 to a state
    with |rho psi| at round-off level, for random noise increments.

    The set {F = 0} is invariant in exact arithmetic but unstable: over many
    steps round-off grows (step 6H). So the test is one step. A wrong offset
    (H_c psi != 0) gives |rho psi| of order dt = 1e-3.
    """
    from sim.plant import plant_step
    rho0 = _complement_states(psi, 64, seed=44)
    u = Offset(2.0, u0, C.xbar_of(psi))(C.to_x(rho0))
    rng = np.random.default_rng(45)
    rho1, _ = plant_step(rho0, u, rng.normal(0, np.sqrt(1e-3), 64), 1e-3, prm)
    assert np.abs(rho1 @ psi).max() < 1e-14


def test_output_mean():
    """E[dy] = 2 sqrt(eta M) (rho_00 - rho_33) dt for the half-parity output.
    One step from fixed states, N = 200000 noise draws, 4 standard errors."""
    from sim.plant import plant_step
    rho = random_states(1, seed=46)
    n, dt = 200000, 1e-2
    rng = np.random.default_rng(47)
    _, dy = plant_step(np.repeat(rho, n, axis=0), np.zeros((n, 4)), rng.normal(0, np.sqrt(dt), n),
                       dt, HALF)
    expect = HALF.c * (rho[0, 0, 0].real - rho[0, 3, 3].real) * dt
    assert abs(dy.mean() - expect) < 4 * np.sqrt(dt / n)


def test_mean_follows_lindblad():
    """E[rho] obeys the Lindblad equation of L = sqrt(M) diag(1, 0, 0, -1).

    With u = 0, the coherence rho_{00,11} decays as exp(-2 M t), because
    (l_00 - l_11)^2 / 2 = 2, and the diagonal is constant. Under the ZZ
    measurement that coherence would not decay. N = 4000, T = 0.5,
    4 standard errors; the step error is O(dt) = 1e-3, below that.
    """
    n, T = 4000, 0.5
    rho0 = np.repeat(random_states(1, seed=49), n, axis=0)
    res = simulate(rho0, Offset(0.0, np.zeros(4)), HALF, T=T, dt=1e-3, seed=50, record_every=500)
    r = res.rho_final
    c = r[:, 0, 3]
    want = rho0[0, 0, 3] * np.exp(-2 * HALF.M * T)
    se = np.abs(c - c.mean()).std() / np.sqrt(n) + 1e-12
    assert abs(c.mean() - want) < 4 * np.sqrt(2) * se + 2e-3 * abs(want), (c.mean(), want)
    d = np.einsum("nii->ni", r).real
    se_d = d.std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(d.mean(axis=0) - np.diag(rho0[0]).real) < 4 * se_d + 1e-12)


@pytest.mark.slow
def test_collapse_statistics():
    """With u = 0 and eta = 1 the state collapses onto |00>, the odd sector,
    or |11>, with probabilities equal to the initial populations."""
    prm = Plant(meas="half", eta=1.0, omega=0.0)
    psi = np.array([0.5, 0.6, 0.4, np.sqrt(1 - 0.25 - 0.36 - 0.16)], dtype=complex)
    n = 4000
    res = simulate(pure(psi, n), Offset(0.0, np.zeros(4)), prm, T=20.0, dt=1e-3, seed=48,
                   record_every=20000)
    d = np.einsum("nii->ni", res.rho_final).real
    p00, podd, p11 = np.mean(d[:, 0] > 0.99), np.mean(d[:, 1] + d[:, 2] > 0.99), np.mean(d[:, 3] > 0.99)
    for got, want in ((p00, 0.25), (podd, 0.36 + 0.16), (p11, 0.23)):
        assert abs(got - want) < 3 * np.sqrt(want * (1 - want) / n), (got, want)
    assert p00 + podd + p11 > 0.99
