"""sim.gpu_ksweep.run_batch against sim.closed_loop.simulate on shared noise."""
import numpy as np

from sim.closed_loop import simulate
from sim.gpu_ksweep import run_batch
from sim.laws import Offset
from sim.params import Plant

from .helpers import random_states


def test_gpu_batch_equals_cpu_simulator_on_shared_noise():
    """Two gains in one torch batch (on the CPU device) against two CPU runs
    with the same Wiener increments. Deterministic, so the tolerance is the
    round-off of 500 steps in float64."""
    n, T, dt = 3, 0.5, 1e-3
    rho0 = random_states(n, seed=4)
    steps = round(T / dt)
    rng = np.random.default_rng(9)
    dW = rng.normal(0.0, np.sqrt(dt), (steps, 2 * n))
    F_gpu, rho_gpu = run_batch(np.concatenate([rho0, rho0]), np.array([1.0] * n + [4.0] * n), T, dt, Plant(),
                               "cpu", dW=dW, rec=T)
    for j, k in enumerate((1.0, 4.0)):
        sl = slice(j * n, (j + 1) * n)
        res = simulate(rho0.copy(), Offset(k), Plant(), T=T, dt=dt, seed=0, dW=dW[:, sl], record_every=steps)
        np.testing.assert_allclose(rho_gpu[sl], res.rho_final, rtol=0, atol=1e-12)
        np.testing.assert_allclose(F_gpu[-1, sl], res.F[-1], rtol=0, atol=1e-12)
