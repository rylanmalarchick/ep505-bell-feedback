"""Hardware hooks between the controller and the plant.

Order in the loop (report, Sec. "Hardware hooks"):
    controller -> quantizer -> hold (Ts) -> delay (tau) -> actuator -> plant

quantize and actuator_step are pure functions. DelayLine holds a ring
buffer, because a copy of the whole buffer at every step costs O(delay).
The hold is the sample schedule in closed_loop.simulate.
"""
import numpy as np


def quantize(u: np.ndarray, w: int | None, umax: float | None) -> np.ndarray:
    """Uniform mid-tread quantizer with w bits over [-umax, umax).

    Levels are n * Delta for n in [-2^(w-1), 2^(w-1) - 1], Delta = umax / 2^(w-1)
    (a two's-complement DAC code). Inputs outside the range clip to the end
    levels. Inside the range, |quantize(u) - u| <= Delta / 2.
    w = None returns u unchanged.
    """
    if w is None:
        return u
    if umax is None or umax <= 0:
        raise ValueError("quantize needs umax > 0")
    half = 2 ** (w - 1)
    delta = umax / half
    return np.clip(np.round(u / delta), -half, half - 1) * delta


def actuator_step(u_act: np.ndarray, u_cmd: np.ndarray, alpha: float) -> np.ndarray:
    """First-order actuator, exact for a command held over one step:
    u_act <- u_act + alpha (u_cmd - u_act), alpha = 1 - exp(-dt / tau_a)."""
    return u_act + alpha * (u_cmd - u_act)


def actuator_alpha(dt: float, tau_a: float) -> float:
    return 1.0 - float(np.exp(-dt / tau_a))


class DelayLine:
    """Delay of d whole steps. The output at step k is the input at step k - d.

    Before step d, the output is u_init. d = 0 passes the input through.
    """

    def __init__(self, d: int, u_init: np.ndarray):
        if d < 0:
            raise ValueError("delay must be >= 0 steps")
        self.d = d
        self.buf = np.repeat(np.asarray(u_init, dtype=float)[None], max(d, 1), axis=0)
        self.head = 0

    def push(self, u: np.ndarray) -> np.ndarray:
        if self.d == 0:
            return u
        out = self.buf[self.head].copy()
        self.buf[self.head] = u
        self.head = (self.head + 1) % self.d
        return out
