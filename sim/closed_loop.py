"""Closed-loop simulation: plant, observer, law, and hardware hooks.

Per plant step k (length dt):
  1. At a sample instant (k a multiple of n_s = Ts / dt):
     v <- quantize(law(x_hat)). The hold keeps v until the next sample.
  2. u_cmd <- delay(v).  3. u <- actuator(u_cmd).  4. Step the plant with u.
  5. At the end of a sample period, step the separate observer (if any)
     over the period with the held command v and the summed output Delta y.

The observer gets the command v, not the delayed and filtered input u.
So delay and actuator dynamics are a model mismatch for the observer, the
same as on hardware that does not model them.

With rho_hat0 = None the observer is perfect: x_hat = x at each sample.
"""
from dataclasses import dataclass, field

import numpy as np

from .coords import PHI_PLUS, fidelity, p1, to_x
from .hooks import DelayLine, actuator_alpha, actuator_step, quantize
from .observer import observer_step
from .params import Plant
from .plant import plant_step


@dataclass(frozen=True)
class Hooks:
    Ts: float | None = None  # sample period; None means Ts = dt
    tau: float = 0.0  # loop delay
    w: int | None = None  # quantizer bits; None means no quantizer
    umax: float | None = None  # quantizer full scale
    tau_a: float | None = None  # actuator time constant; None means no filter


@dataclass
class Result:
    t: np.ndarray  # (R,) record times
    F: np.ndarray  # (R, N) fidelity of the true state
    F_hat: np.ndarray  # (R, N) fidelity of the estimate
    p1: np.ndarray  # (R, N) parity of the true state
    u: np.ndarray  # (R, N, 4) input applied to the plant
    rho_final: np.ndarray  # (N, 4, 4)
    est_err: np.ndarray  # (R, N) trace distance D(rho_hat, rho); 0 for a perfect observer
    max_trace_err: float = float("nan")  # max |Tr rho - 1| over all steps
    min_eig: float = float("nan")  # min eigenvalue of rho over all steps
    meta: dict = field(default_factory=dict)


def _whole_steps(value: float, dt: float, name: str) -> int:
    n = round(value / dt)
    if abs(n * dt - value) > 1e-9 * max(1.0, value):
        raise ValueError(f"{name} = {value} is not a whole number of steps dt = {dt}")
    return n


def simulate(rho0: np.ndarray, law, prm: Plant, T: float, dt: float, seed: int,
             rho_hat0: np.ndarray | None = None, hooks: Hooks = Hooks(),
             record_every: int = 1, dW: np.ndarray | None = None,
             check_physical: bool = False, target: np.ndarray = PHI_PLUS) -> Result:
    """Run N trajectories from rho0 (N, 4, 4) for time T.

    target: the state whose fidelity F the result records (Phi+ by default).

    dW: optional Wiener increments, shape (T/dt, N). If None, they are drawn
    from numpy default_rng(seed) one step at a time.
    check_physical: compute the trace error and the smallest eigenvalue of
    rho at every step (costs one eigvalsh per step).
    """
    n_steps = _whole_steps(T, dt, "T")
    n_s = 1 if hooks.Ts is None else _whole_steps(hooks.Ts, dt, "Ts")
    Ts = n_s * dt
    d = _whole_steps(hooks.tau, dt, "tau")
    N = rho0.shape[0]
    if dW is not None and dW.shape != (n_steps, N):
        raise ValueError(f"dW has shape {dW.shape}, need {(n_steps, N)}")
    rng = np.random.default_rng(seed)
    sqdt = np.sqrt(dt)

    rho = rho0.copy()
    rho_hat = None if rho_hat0 is None else rho_hat0.copy()
    law.reset(N)
    v = np.zeros((N, 4))
    delay = DelayLine(d, np.zeros((N, 4)))
    alpha = None if hooks.tau_a is None else actuator_alpha(dt, hooks.tau_a)
    u = np.zeros((N, 4))
    dy_acc = np.zeros(N)

    n_rec = n_steps // record_every + 1
    t_rec = np.zeros(n_rec)
    F_rec = np.zeros((n_rec, N))
    Fh_rec = np.zeros((n_rec, N))
    p1_rec = np.zeros((n_rec, N))
    u_rec = np.zeros((n_rec, N, 4))
    err_rec = np.zeros((n_rec, N))
    max_tr, min_eig = 0.0, np.inf

    def record(i, t):
        t_rec[i] = t
        F_rec[i] = fidelity(rho, target)
        Fh_rec[i] = fidelity(rho if rho_hat is None else rho_hat, target)
        p1_rec[i] = p1(rho)
        u_rec[i] = u
        if rho_hat is not None:
            err_rec[i] = 0.5 * np.abs(np.linalg.eigvalsh(rho_hat - rho)).sum(axis=1)

    record(0, 0.0)
    for k in range(n_steps):
        if k % n_s == 0:
            x_hat = to_x(rho if rho_hat is None else rho_hat)
            v = quantize(law(x_hat), hooks.w, hooks.umax)
        u_cmd = delay.push(v)
        u = u_cmd if alpha is None else actuator_step(u, u_cmd, alpha)
        dWk = rng.normal(0.0, sqdt, N) if dW is None else dW[k]
        rho, dy = plant_step(rho, u, dWk, dt, prm)
        dy_acc += dy
        if rho_hat is not None and (k + 1) % n_s == 0:
            rho_hat = observer_step(rho_hat, v, dy_acc, Ts, prm)
            dy_acc = np.zeros(N)
        if check_physical:
            max_tr = max(max_tr, float(np.abs(np.einsum("nii->n", rho) - 1).max()))
            min_eig = min(min_eig, float(np.linalg.eigvalsh(rho).min()))
        if (k + 1) % record_every == 0:
            record((k + 1) // record_every, (k + 1) * dt)

    return Result(t_rec, F_rec, Fh_rec, p1_rec, u_rec, rho, err_rec,
                  max_tr if check_physical else float("nan"),
                  min_eig if check_physical else float("nan"),
                  meta=dict(T=T, dt=dt, seed=seed, N=N, hooks=hooks, prm=prm,
                            law=type(law).__name__))
