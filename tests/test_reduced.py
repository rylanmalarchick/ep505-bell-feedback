"""Reduced filters."""
import numpy as np

from sim import coords as C
from sim.params import Plant
from sim.reduced import ReducedFilter, S_T, subspace

from .helpers import random_states

PRM = Plant()


def test_subspace_dimensions_match_exact_closure():
    """verification/check_reduction.py gives 15, 9, and 2 in exact arithmetic."""
    assert [subspace(n, PRM).shape[0] for n in ("full", "base9", "lam2")] == [15, 9, 2]
    for n in ("full", "base9", "lam2"):
        Pi = subspace(n, PRM)
        assert np.allclose(Pi @ Pi.T, np.eye(Pi.shape[0]), atol=1e-12)


def test_full_reduced_filter_is_the_euler_observer():
    """With Pi = I one step equals x + f dt + b(x) (dy - c p1 dt), with f and b
    built here from the model matrices (report eqs. plant and observer)."""
    rng = np.random.default_rng(60)
    x = C.to_x(random_states(16, seed=61))
    u = rng.normal(size=(16, 4))
    dy = rng.normal(0, 0.03, 16)
    dt, c = 1e-3, PRM.c
    A = C.a0_matrix(PRM) + np.einsum("nj,jab->nab", u, C.B_MATS)
    f = np.einsum("nab,nb->na", A, x)
    b = c * (np.eye(15)[0] + x @ S_T.T - x[:, :1] * x)
    want = x + f * dt + b * (dy - c * x[:, 0] * dt)[:, None]
    flt = ReducedFilter(subspace("full", PRM), PRM)
    assert np.allclose(flt.decode(flt.step(flt.encode(x), u, dy, dt)), want, atol=1e-14)


def test_reduced_offset_law_sees_no_h_on_lam2():
    """h is zero on span{p1, F}, so the law fed by lam2 applies u0 exactly."""
    Pi = subspace("lam2", PRM)
    x = C.to_x(random_states(8, seed=62)) @ Pi.T @ Pi
    assert np.abs(C.h(x)).max() < 1e-12
