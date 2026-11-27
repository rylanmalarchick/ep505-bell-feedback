"""Plant parameters.

The defaults match the one-channel simulation of Liang, Amini, and Mason
(CDC 2019, arXiv:1903.07470): M = 1, omega = 0.3, eta = 0.3. Rates are in
units of M.

meas selects the measured operator:
  "ZZ"    full parity,  L = sqrt(M) ZZ (the default)
  "half"  half parity,  L = sqrt(M) (ZI + IZ) / 2 = sqrt(M) diag(1, 0, 0, -1)
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Plant:
    omega: float = 0.3  # drift rotation rate (H0 = omega ZZ)
    M: float = 1.0  # measurement strength, L = sqrt(M) ZZ
    eta: float = 0.3  # detector efficiency, in (0, 1]
    meas: str = "ZZ"  # measured operator: "ZZ" or "half"

    def __post_init__(self):
        if self.meas not in ("ZZ", "half"):
            raise ValueError(f"meas must be 'ZZ' or 'half', not {self.meas!r}")

    @property
    def c(self) -> float:
        """Output gain c = 2 sqrt(eta M). The output is dy = c p1 dt + dW."""
        return 2.0 * math.sqrt(self.eta * self.M)
