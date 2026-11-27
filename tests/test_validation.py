"""Statistical validation runs (slow).

These take minutes. Run:
    python -m pytest -q -s -m slow
The tolerances on Monte Carlo quantities are 3 standard errors. They are
stochastic quantities, and the seeds are fixed, so each result is
deterministic for a given commit.
"""
import numpy as np
import pytest

from sim import coords as C
from sim.closed_loop import simulate
from sim.laws import Offset
from sim.params import Plant
from sim.plant import kraus_step

from .helpers import pure

pytestmark = pytest.mark.slow


class Zero:
    def reset(self, n):
        pass

    def __call__(self, x):
        return np.zeros((x.shape[0], 4))


def test_1_6a_qnd_collapse_statistics():
    """1.6a. With u = 0, P(even parity at large t) = Lambda_0.

    ZZ commutes with H = omega ZZ, so the parity is a QND observable: the
    state collapses to the even or the odd sector, and the probability of
    the even sector is Lambda_0 = Tr(Pi_even rho_0). Lambda_0 = 0.3 here.
    eta = 1 for a fast collapse.
    """
    lam0, N = 0.3, 4000
    rho0 = pure([np.sqrt(lam0), np.sqrt(1 - lam0), 0, 0], N)
    res = simulate(rho0, Zero(), Plant(eta=1.0), T=6.0, dt=1e-3, seed=101,
                   record_every=6000)
    lam_T = 0.5 * (1 + res.p1[-1])
    collapsed = np.mean((lam_T > 0.99) | (lam_T < 0.01))
    p_even = np.mean(lam_T > 0.99)
    se = np.sqrt(lam0 * (1 - lam0) / N)
    print(f"\n1.6a collapsed = {collapsed:.4f}, P(even) = {p_even:.4f}, "
          f"Lambda_0 = {lam0}, 3 se = {3 * se:.4f}")
    assert collapsed > 0.99
    assert abs(p_even - lam0) < 3 * se


DT_RUN = 1e-3  # step size for the Phase 2 and 3 runs
DT_BUDGET = 0.005  # allowed |mean F(dt) - mean F(dt/2)| at T


def test_1_6d_dt_halving():
    """1.6d. Halving dt changes the mean of F at T by less than DT_BUDGET.

    DT_BUDGET = 0.005 is half of the 0.01 margin in the Phase 2 success
    metric (F > 0.99). All three runs use the same Brownian paths: the
    increments at dt are sums of the increments at dt/2 and dt/4, so the
    difference is integrator error, not sampling noise.
    Offset law, k = 1, from |00>, N = 500, T = 10.
    """
    N, T = 500, 10.0
    dt_f = DT_RUN / 4
    rng = np.random.default_rng(202)
    dW_f = rng.normal(0.0, np.sqrt(dt_f), (round(T / dt_f), N))
    rho0 = pure([1, 0, 0, 0], N)
    meanF = {}
    for m in (4, 2, 1):  # dt = DT_RUN, DT_RUN/2, DT_RUN/4
        dt = dt_f * m
        dW = dW_f.reshape(-1, m, N).sum(axis=1)
        res = simulate(rho0, Offset(1.0), Plant(), T=T, dt=dt, seed=0, dW=dW,
                       record_every=round(T / dt))
        meanF[dt] = res.F[-1].mean()
    d1 = abs(meanF[DT_RUN] - meanF[DT_RUN / 2])
    d2 = abs(meanF[DT_RUN / 2] - meanF[DT_RUN / 4])
    print(f"\n1.6d mean F(T): {meanF}")
    print(f"1.6d |dt - dt/2| = {d1:.2e}, |dt/2 - dt/4| = {d2:.2e}, budget {DT_BUDGET}")
    assert d1 < DT_BUDGET


def test_1_6e_liang_amini_mason_reproduction():
    """1.6e. sim.plant.kraus_step reproduces a reference run of the one-channel law of Liang, Amini, and Mason (2019).

    Same setup, seed, and noise draws as the reference: target Psi+, start
    Phi-, eta = 0.3, omega = 0.3, gamma1 = 4, gamma2 = -4, eps = 0.15,
    N = 256, dt = 5e-4. The reference
    run gave P(X > 0.99) = 0.656 at t = 10 and 0.805 at t = 30.
    The new step adds the Ito term (eta/2) L^2 (dy^2 - dt), so the paths
    differ by O(dt). Pass: |P(t=30) - 0.805| < 3 se (binomial, N = 256),
    and P(X > 0.99) grows from t = 10 to t = 30 (Prop. 5.2 of the paper:
    convergence to the target).
    """
    prm = Plant(omega=0.3, M=1.0, eta=0.3)
    sy, sz, i2 = C._PAULI["Y"], C._PAULI["Z"], C._PAULI["I"]
    H1 = np.kron(sz, sy) - 3 * np.kron(i2, sy)
    H2 = np.kron(sy, sz) - 3 * np.kron(sy, i2)
    psi_p = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    target = np.outer(psi_p, psi_p.conj())
    eps, g1, g2 = 0.15, 4.0, -4.0

    def f_gate(x):
        mid = 0.5 * np.sin(np.pi * (x - 0.5) / (1 - 2 * eps)) + 0.5
        return np.where(x < eps, 0.0, np.where(x > 1 - eps, 1.0, mid))

    def tr_t(a):
        return np.einsum("nij,ji->n", a, target).real

    N, dt, T = 256, 5e-4, 30.0
    rng = np.random.default_rng(101)
    rho = pure(C.PHI_MINUS, N)
    p99 = {}
    for k in range(1, round(T / dt) + 1):
        X = tr_t(rho)
        u1 = g1 - tr_t(1j * (H1 @ rho - rho @ H1))
        u2 = f_gate(X) * (g2 - tr_t(1j * (H2 @ rho - rho @ H2)))
        H = prm.omega * C.ZZ + u1[:, None, None] * H1 + u2[:, None, None] * H2
        dW = rng.normal(0.0, np.sqrt(dt), N)
        dy = prm.c * C.p1(rho) * dt + dW
        rho = kraus_step(rho, H, dy, dt, prm)
        if k in (round(10 / dt), round(30 / dt)):
            p99[round(k * dt)] = np.mean(tr_t(rho) > 0.99)
    ref, se = 0.805, np.sqrt(0.805 * 0.195 / N)
    print(f"\n1.6e P(X > 0.99): t=10 {p99[10]:.3f} (ref 0.656), "
          f"t=30 {p99[30]:.3f} (ref 0.805), 3 se = {3 * se:.3f}")
    assert abs(p99[30] - ref) < 3 * se
    assert p99[30] > p99[10]
