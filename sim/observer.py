"""Observer: the quantum filter run from its own initial estimate.

The observer uses the same Rouchon-Ralph map as the plant, driven by the
recorded output dy and by the input the controller believes it applied.
With rho_hat_0 = rho_0, the same dy, and the same input, the observer state
equals the plant state at every step.

With a sample period Ts > dt, the observer takes one step of length Ts with
the summed output Delta y_k = sum of dy over the period.
"""
import numpy as np

from .params import Plant
from .plant import hamiltonian, kraus_step


def observer_step(rho_hat: np.ndarray, u: np.ndarray, dy: np.ndarray, dt: float,
                  prm: Plant) -> np.ndarray:
    """Advance the estimate one step of length dt with output increment dy."""
    return kraus_step(rho_hat, hamiltonian(u, prm), dy, dt, prm)
