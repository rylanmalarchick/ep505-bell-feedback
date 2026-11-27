"""Feedback laws u = kappa(x_hat). Each law maps x_hat (N, 15) to u (N, 4).

Baseline:   u = k h(x_hat)                 (Jurdjevic-Quinn damping)
Offset:     u = u0 + k h(x_hat),  u0 = (1, 0, -1, 0)
Hysteresis: the offset law, or a constant input u_c near {F = 0}, in the
            style of Mirrahimi and van Handel (2007), Sec. 5.

Every law has reset(n), which sets its internal state for n trajectories.
xbar is the coordinate vector of the target (Phi+ by default).
"""
import numpy as np

from .coords import H_ROWS, U_0, X_BAR, h, h_rows


def fidelity_x(x: np.ndarray, xbar: np.ndarray = X_BAR) -> np.ndarray:
    """F = (1 + xbar^T x) / 4."""
    return 0.25 * (1.0 + x @ xbar)


def _rows(xbar):
    return H_ROWS if xbar is X_BAR else h_rows(np.asarray(xbar, dtype=float))


class Baseline:
    def __init__(self, k: float, xbar: np.ndarray = X_BAR):
        self.k = k
        self.rows = _rows(xbar)

    def reset(self, n: int) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.k * h(x, self.rows)


class Offset:
    def __init__(self, k: float, u0: np.ndarray = U_0, xbar: np.ndarray = X_BAR):
        self.k = k
        self.u0 = np.asarray(u0, dtype=float)
        self.rows = _rows(xbar)

    def reset(self, n: int) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.u0 + self.k * h(x, self.rows)


class Hysteresis:
    """Offset law with a switch to a constant input u_c near {F = 0}.

    A trajectory in feedback mode switches to u_c when F < f_low. A
    trajectory in constant mode switches back to feedback when F >= f_high.
    For f_low <= F < f_high, the mode does not change.
    """

    def __init__(self, k: float, u_c, f_low: float, f_high: float, u0: np.ndarray = U_0,
                 xbar: np.ndarray = X_BAR):
        if not 0.0 < f_low < f_high < 1.0:
            raise ValueError("need 0 < f_low < f_high < 1")
        self.feedback = Offset(k, u0, xbar)
        self.xbar = xbar
        self.u_c = np.asarray(u_c, dtype=float)
        self.f_low = f_low
        self.f_high = f_high
        self.constant_mode = np.zeros(0, dtype=bool)

    def reset(self, n: int) -> None:
        self.constant_mode = np.zeros(n, dtype=bool)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        F = fidelity_x(x, self.xbar)
        self.constant_mode = np.where(self.constant_mode, F < self.f_high, F < self.f_low)
        return np.where(self.constant_mode[:, None], self.u_c, self.feedback(x))
