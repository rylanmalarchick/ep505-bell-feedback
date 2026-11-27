"""sim/hooks.py and sim/laws.py."""
import numpy as np
import pytest

from sim import coords as C
from sim.hooks import DelayLine, actuator_alpha, actuator_step, quantize
from sim.laws import Baseline, Hysteresis, Offset


# Quantizer
def test_quantize_none_is_identity():
    u = np.linspace(-3, 3, 11)
    assert quantize(u, None, None) is u


@pytest.mark.parametrize("w", [4, 8, 12])
def test_quantize_error_within_half_step(w):
    umax = 2.0
    delta = umax / 2 ** (w - 1)
    u = np.linspace(-umax, umax - delta, 10001)
    err = np.abs(quantize(u, w, umax) - u)
    assert err.max() <= delta / 2 + 1e-15


def test_quantize_levels_and_clipping():
    w, umax = 3, 1.0  # levels -1, -0.75, ..., 0.75
    q = quantize(np.linspace(-5, 5, 2001), w, umax)
    assert np.array_equal(np.unique(q), np.arange(-4, 4) * 0.25)
    assert quantize(np.array([0.0]), w, umax)[0] == 0.0


# Delay
def test_delay_shifts_by_whole_steps():
    d = 3
    line = DelayLine(d, np.zeros((1, 4)))
    ins = [np.full((1, 4), k + 1.0) for k in range(10)]
    outs = [line.push(x) for x in ins]
    for k in range(10):
        expect = 0.0 if k < d else ins[k - d][0, 0]
        assert outs[k][0, 0] == expect


def test_delay_zero_is_passthrough():
    line = DelayLine(0, np.zeros((2, 4)))
    x = np.arange(8.0).reshape(2, 4)
    assert line.push(x) is x


# Actuator
def test_actuator_step_response_is_exact():
    dt, tau_a = 1e-3, 0.05
    a = actuator_alpha(dt, tau_a)
    u = np.zeros(1)
    for _ in range(200):
        u = actuator_step(u, np.ones(1), a)
    assert abs(u[0] - (1 - np.exp(-200 * dt / tau_a))) < 1e-12


# Laws
def e_plus_point(a, b, d):
    return np.array([[1, a, a, b, -b, d, d] + [0] * 8], dtype=float)


def test_baseline_is_zero_on_e_plus():
    """On E+, q = 0, so h = 0 and the baseline law applies no input."""
    for a, b, d in [(0.2, -0.5, 0.1), (0, -1, 0), (0, 1, 0)]:
        assert np.allclose(Baseline(3.0)(e_plus_point(a, b, d)), 0, atol=1e-15)


def test_baseline_symmetry():
    """The baseline law gives u1 = u3 and u2 = -u4 (control model, Sec. 7)."""
    x = C.to_x(np.array([np.eye(4) / 4 + 0.05 * (C.pauli("XI") + C.pauli("YZ"))]))
    x = x + np.random.default_rng(0).normal(scale=0.1, size=(5, 15))
    u = Baseline(1.0)(x)
    assert np.allclose(u[:, 0], u[:, 2]) and np.allclose(u[:, 1], -u[:, 3])


def test_offset_law_adds_u0():
    x = np.random.default_rng(1).normal(size=(4, 15))
    assert np.allclose(Offset(2.0)(x) - Baseline(2.0)(x), C.U_0)


def test_hysteresis_switching():
    law = Hysteresis(k=1.0, u_c=[0, 0, 7, 0], f_low=0.05, f_high=0.10)
    law.reset(1)

    def x_with_F(F):  # point on the xbar line with fidelity F
        return ((4 * F - 1) / 3 * C.X_BAR)[None]

    assert law(x_with_F(0.5))[0, 2] != 7  # feedback
    assert law(x_with_F(0.04))[0, 2] == 7  # below f_low: constant
    assert law(x_with_F(0.08))[0, 2] == 7  # between: keep constant
    assert law(x_with_F(0.10))[0, 2] != 7  # at f_high: feedback
    assert law(x_with_F(0.08))[0, 2] != 7  # between: keep feedback


# Wilson interval (sim/phase2.py)
def test_wilson_interval():
    from sim.phase2 import wilson
    lo, hi = wilson(0, 500)
    assert lo == 0.0 and abs(hi - 3.8415 / 503.8415) < 1e-6  # z^2 / (n + z^2) at k = 0
    lo, hi = wilson(500, 500)
    assert hi == 1.0 and lo < 1.0
    lo, hi = wilson(250, 500)
    assert lo < 0.5 < hi and abs((hi - 0.5) - (0.5 - lo)) < 1e-12


def test_switching_law_leaves_the_zero_fidelity_set():
    """From Phi- exactly (F = 0), the offset law keeps F at round-off level over
    a short run, and the switching law with u_c = (1, 1, 1, 0) raises F above
    1e-3 (verification/check_switch.py: u_c leaves no face of {F = 0} invariant)."""
    from sim import coords as C
    from sim.closed_loop import simulate
    from sim.laws import Hysteresis, Offset
    from sim.params import Plant

    from .helpers import pure
    rho0 = pure(C.PHI_MINUS, 16)
    off = simulate(rho0, Offset(2.0), Plant(), T=0.5, dt=1e-3, seed=70)
    sw = simulate(rho0, Hysteresis(2.0, (1, 1, 1, 0), 0.05, 0.1), Plant(), T=0.5, dt=1e-3, seed=70)
    assert off.F.max() < 1e-12
    assert sw.F[-1].min() > 1e-3
