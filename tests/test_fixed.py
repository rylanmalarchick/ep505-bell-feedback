"""Fixed-point filter and law."""
import numpy as np
import pytest

from sim import coords as C
from sim.fixed import FixedFilter, default_shift, rnd, sat
from sim.hooks import quantize
from sim.laws import Offset
from sim.observer import observer_step
from sim.params import Plant

from .helpers import random_states

PRM = Plant()
K = 2.0


def _one_step_error(w, Ts, seed):
    """Max entry error of one fixed-point step against the float step."""
    rng = np.random.default_rng(seed)
    N = 256
    rho = random_states(N, seed=seed)
    flt = FixedFilter(w, Ts, PRM, K)
    n = rng.integers(-2 ** (flt.wc - 1), 2 ** (flt.wc - 1), size=(N, 4))
    dy = PRM.c * C.p1(rho) * Ts + rng.normal(0.0, np.sqrt(Ts), N)
    R, I = flt.encode(rho)
    dy_w = flt.adc(dy)
    # The float reference starts from the same quantized state, input, and output.
    ref = observer_step(flt.decode(R, I), flt.command(n), dy_w / 2.0**flt.FD, Ts, PRM)
    out = flt.decode(*flt.step(R, I, n, dy_w))
    return np.abs(out - ref).max()


@pytest.mark.parametrize("Ts", [0.001, 0.01, 0.05])
def test_one_step_matches_float(Ts):
    """One step differs from the float step by a few state LSB, for every w.

    A state LSB is 2^-(w-2). The bound of 4 LSB covers the rounding of the
    increment, of the reciprocal, and of the normalization product. An
    error in a coefficient or a sign is O(Ts) or O(dy), far above 4 LSB at
    w = 24.
    """
    for w in (12, 16, 20, 24):
        err = _one_step_error(w, Ts, seed=5)
        assert err < 4 * 2.0 ** -(w - 2), (w, Ts, err * 2.0 ** (w - 2))


def test_state_stays_physical_and_tracks_float():
    """2000 steps in closed loop, against the float filter on the same
    codes and the same quantized output.

    Exact: the state stays Hermitian, and the trace stays within 3 LSB of 1.
    Stochastic: the filter amplifies rounding noise, with rare large
    excursions, so no tight bound on the error exists. The bounds below are
    loose on purpose. The median error over the runs is tens of LSB at both
    word lengths (below 128 LSB), which shows that the error scales with the
    LSB. At w = 24 the largest error and the largest negative eigenvalue
    stay below 2^-10. A wrong coefficient gives an O(1) error instead.
    """
    Ts, N = 0.01, 64
    for w in (16, 24):
        flt = FixedFilter(w, Ts, PRM, K)
        rng = np.random.default_rng(8)
        rho = np.concatenate([random_states(N // 2, seed=20, rank=1), random_states(N // 2, seed=21)])
        R, I = flt.encode(rho)
        ref = flt.decode(R, I)
        lsb = 2.0 ** -(w - 2)
        worst_tr, worst_eig, worst_err, med_err = 0, 0.0, 0.0, 0.0
        for _ in range(2000):
            n = flt.codes(R, I)
            est = flt.decode(R, I)
            dy_w = flt.adc(PRM.c * C.p1(est) * Ts + rng.normal(0.0, np.sqrt(Ts), N))
            ref = observer_step(ref, flt.command(n), dy_w / 2.0**flt.FD, Ts, PRM)
            R, I = flt.step(R, I, n, dy_w)
            assert np.array_equal(R, R.transpose(0, 2, 1))
            assert np.array_equal(I, -I.transpose(0, 2, 1))
            err = np.abs(flt.decode(R, I) - ref).max(axis=(1, 2))
            worst_tr = max(worst_tr, int(np.abs(np.einsum("nii->n", R) - 2 ** (w - 2)).max()))
            worst_eig = min(worst_eig, float(np.linalg.eigvalsh(flt.decode(R, I)).min()))
            worst_err, med_err = max(worst_err, float(err.max())), max(med_err, float(np.median(err)))
        assert worst_tr <= 3, (w, worst_tr)
        assert med_err < 128 * lsb, (w, med_err / lsb)
        if w == 24:
            assert worst_err < 2.0**-10, worst_err
            assert worst_eig > -(2.0**-10), worst_eig


def test_saturation_clamps():
    """sat clamps to the word range and does not wrap. The ADC clips a
    large input to full scale with the correct sign."""
    x = np.array([-300, -129, -128, 0, 127, 128, 300])
    assert sat(x, 8).tolist() == [-128, -128, -128, 0, 127, 127, 127]
    flt = FixedFilter(8, 0.01, PRM, K)
    assert flt.adc(np.array([1e3, -1e3])).tolist() == [127, -128]


def test_rnd_rounds_ties_to_even():
    x = np.array([-3, -2, -1, 0, 1, 2, 3])
    assert rnd(x, 1).tolist() == [-2, -1, 0, 0, 0, 1, 2]
    assert rnd(np.array([5, 6, 7, 9, 10, 11]), 2).tolist() == [1, 2, 2, 2, 2, 3]
    assert rnd(x, 0).tolist() == x.tolist()
    assert rnd(x, -2).tolist() == (4 * x).tolist()


def test_law_codes_match_float_law():
    """The integer law gives exactly the code of the float law on the same
    state. The float sum is exact here (dyadic values), and numpy rounds
    ties to even, as rnd does."""
    flt = FixedFilter(16, 0.01, PRM, K)
    rho = random_states(512, seed=30)
    R, I = flt.encode(rho)
    u = Offset(K)(C.to_x(flt.decode(R, I)))
    ref = np.rint(quantize(u, flt.wc, 4.0) / flt.du).astype(np.int64)
    assert np.array_equal(flt.codes(R, I), ref)


def test_shift_keeps_dy_in_range():
    """With the default shift, |dy| <= c Ts + 6 sqrt(Ts) does not clip."""
    for Ts in (0.001, 0.01, 0.03, 0.05):
        flt = FixedFilter(16, Ts, PRM, K)
        assert flt.s == default_shift(Ts, PRM)
        bound = PRM.c * Ts + 6 * np.sqrt(Ts)
        assert abs(int(flt.adc(np.array([bound]))[0])) < 2 ** (flt.w - 1) - 1
