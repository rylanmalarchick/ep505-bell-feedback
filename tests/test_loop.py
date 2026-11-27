"""Closed-loop tests that run in seconds, and observer checks."""
import numpy as np

from sim import coords as C
from sim.closed_loop import Hooks, simulate
from sim.laws import Baseline, Offset
from sim.params import Plant

from .helpers import pure, random_states

PRM = Plant()


def test_1_6b_baseline_holds_bad_equilibrium():
    """1.6b. The baseline law started at xbar' (Phi-) stays at xbar'.

    xbar' is an exact equilibrium (h = 0, drift = 0, diffusion = 0), so the
    only error is floating-point round-off.
    """
    rho0 = pure(C.PHI_MINUS, 8)
    res = simulate(rho0, Baseline(k=2.0), PRM, T=5.0, dt=1e-3, seed=3)
    assert np.abs(C.to_x(res.rho_final) - C.X_BAD).max() < 1e-12
    assert np.abs(res.F).max() < 1e-12


def test_1_6c_trace_and_positivity():
    """1.6c. Tr rho = 1 and rho >= 0 to 1e-12 at every step.

    Random pure and mixed initial states, offset law, two step sizes. The
    larger step and gain stress the map: dt = 1e-2 with |u| up to about 20.
    """
    rho0 = np.concatenate([random_states(32, seed=10, rank=1), random_states(32, seed=11)])
    for dt, k, T in [(1e-3, 1.0, 2.0), (1e-2, 5.0, 5.0)]:
        res = simulate(rho0, Offset(k), PRM, T=T, dt=dt, seed=12, check_physical=True)
        assert res.max_trace_err < 1e-12, (dt, res.max_trace_err)
        assert res.min_eig > -1e-12, (dt, res.min_eig)


def test_observer_with_exact_start_equals_plant():
    """With rho_hat0 = rho0, no hooks, and Ts = dt, the filter is the plant."""
    rho0 = random_states(16, seed=20)
    res = simulate(rho0, Offset(1.0), PRM, T=1.0, dt=1e-3, seed=21, rho_hat0=rho0)
    assert np.abs(res.F - res.F_hat).max() < 1e-12


def test_observer_mismatch_is_seen():
    """A wrong initial estimate gives F_hat != F at the start."""
    rho0 = pure([1, 0, 0, 0], 4)
    res = simulate(rho0, Offset(1.0), PRM, T=0.1, dt=1e-3, seed=22,
                   rho_hat0=pure(C.PHI_PLUS, 4))
    assert np.abs(res.F_hat[0] - 1).max() < 1e-15 and np.abs(res.F[0] - 0.5).max() < 1e-15


def test_hooks_run_and_quantized_input_is_on_grid():
    """Hooks on: the applied input takes only quantizer levels (no actuator filter)."""
    rho0 = pure([1, 0, 0, 0], 8)
    hk = Hooks(Ts=5e-3, tau=2e-3, w=6, umax=4.0)
    res = simulate(rho0, Offset(1.0), PRM, T=0.5, dt=1e-3, seed=23, hooks=hk)
    delta = 4.0 / 2**5
    assert np.allclose(res.u / delta, np.round(res.u / delta), atol=1e-12)


def test_rejects_ts_not_multiple_of_dt():
    rho0 = pure([1, 0, 0, 0], 1)
    try:
        simulate(rho0, Offset(1.0), PRM, T=0.1, dt=1e-3, seed=0, hooks=Hooks(Ts=1.5e-3))
    except ValueError:
        return
    raise AssertionError("Ts = 1.5 dt was accepted")


def test_estimation_error_is_trace_distance():
    """est_err at t = 0 is the trace distance of the two initial states.

    rho0 = |00><00|, rho_hat0 = (1 - s)|00><00| + s|11><11|: D = s.
    """
    s = 0.37
    rho0 = pure([1, 0, 0, 0], 3)
    rho_hat0 = (1 - s) * rho0 + s * pure([0, 0, 0, 1], 3)
    res = simulate(rho0, Offset(1.0), PRM, T=0.01, dt=1e-3, seed=30, rho_hat0=rho_hat0)
    assert np.abs(res.est_err[0] - s).max() < 1e-14
    exact = simulate(rho0, Offset(1.0), PRM, T=0.01, dt=1e-3, seed=30, rho_hat0=rho0)
    assert np.abs(exact.est_err).max() < 1e-14
