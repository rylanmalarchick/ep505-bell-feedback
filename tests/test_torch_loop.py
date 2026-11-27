"""The torch closed loop equals the numpy closed loop."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim import coords as C  # noqa: E402
from sim.closed_loop import Hooks, simulate  # noqa: E402
from sim.laws import Offset  # noqa: E402
from sim.params import Plant  # noqa: E402
from sim.plant import kraus_step  # noqa: E402
from sim.torch_loop import TorchLoop  # noqa: E402

from .helpers import random_states  # noqa: E402


def test_kraus_step_equals_numpy():
    rho = random_states(32, seed=80)
    rng = np.random.default_rng(81)
    u = rng.normal(size=(32, 4))
    dy = rng.normal(0, 0.05, 32)
    from sim.plant import hamiltonian
    want = kraus_step(rho, hamiltonian(u, Plant()), dy, 1e-2, Plant())
    env = TorchLoop(32, 0.01, 0.0)
    got = env.kraus_step(torch.as_tensor(rho), env.hamiltonian(torch.as_tensor(u)), torch.as_tensor(dy), 1e-2)
    assert np.abs(got.numpy() - want).max() < 1e-13


@pytest.mark.parametrize("Ts, tau", [(0.01, 0.0), (0.01, 0.03), (0.05, 0.1)])
def test_closed_loop_equals_numpy(Ts, tau):
    """Offset law, separate filter, quantizer, hold, delay: same noise, same
    fidelity path to 1e-9 over T = 1."""
    N, T, dt = 16, 1.0, 1e-3
    rho0 = random_states(N, seed=82)
    n_steps = round(T / dt)
    dW = np.random.default_rng(83).normal(0, np.sqrt(dt), size=(n_steps, N))
    ref = simulate(rho0, Offset(2.0), Plant(), T=T, dt=dt, seed=0, rho_hat0=rho0.copy(),
                   hooks=Hooks(Ts=Ts, tau=tau, w=12, umax=4.0), dW=dW, record_every=round(Ts / dt))
    env = TorchLoop(N, Ts, tau)
    x = env.reset(torch.as_tensor(rho0))
    n_s = round(Ts / dt)
    F = []
    for j in range(round(T / Ts)):
        x, f = env.step(env.offset_law(x), torch.as_tensor(dW[j * n_s:(j + 1) * n_s]))
        F.append(f.numpy())
    assert np.abs(np.array(F) - ref.F[1:]).max() < 1e-9
