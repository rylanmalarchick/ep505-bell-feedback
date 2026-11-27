"""Fixed-point model of the filter and the offset law.

This is the bit-level reference for the FPGA (step 6F). Every operation is
an integer add, multiply, shift, or clamp, plus one integer division per
filter step. The plant stays in floating point.

Word formats (w bits, two's complement, saturating):
    F  = w - 2          fraction bits of a state word, range [-2, 2)
    s  = shift          extra fraction bits of the small terms
    FD = F + s          fraction bits of dy and of delta = K - 1
The shift s comes from the sample period: the output increment dy and the
entries of delta shrink with Ts, so their words get s more fraction bits.

One filter step of length Ts (Rouchon-Ralph map in increment form):
    delta = alpha 1 + beta ZZ - i Ts H
    alpha = -M Ts / 2 + (eta M / 2)(dy^2 - Ts),  beta = sqrt(eta M) dy
    S     = delta rho + (delta rho)^dag + delta rho delta^dag
            + (1 - eta) M Ts ZZ rho ZZ
    rho  <- (rho + S) / Tr(rho + S)
The map equals K rho K^dag + (1 - eta) M Ts ZZ rho ZZ with K = 1 + delta,
which is sim.plant.kraus_step. The increment form keeps the terms of
order Ts apart from the O(1) state, so they do not round away first.

Rounding: rnd(x, n) is x / 2^n to nearest, ties to even. A matrix
product rounds once per entry, after the sum (one multiply-accumulate).
Normalization: r = round(2^(2F) / t) with t = Tr(rho + S) clamped to
t >= 1/4, then one multiply per entry. The lower triangle is the conjugate
of the upper triangle, and the diagonal is real.

Law: h_j is an exact integer sum of state words. The command code is
    n_j = clamp(round((u0_j + k h_j) / du)),  du = umax / 2^(wc-1),
a wc-bit code. The plant gets u_j = n_j du.
"""
import math

import numpy as np

from .closed_loop import _whole_steps
from .coords import BASIS, H_J, H_ROWS, U_0, fidelity
from .hooks import DelayLine
from .params import Plant
from .plant import plant_step

GUARD = 8  # extra fraction bits of the constants k and du * Ts
ZZ_DIAG = np.array([1, -1, -1, 1], dtype=np.int64)
ZZ_SIGN = np.outer(ZZ_DIAG, ZZ_DIAG)  # (ZZ rho ZZ)_ij = ZZ_SIGN_ij rho_ij


def rnd(x, n: int):
    """x / 2^n, rounded to nearest, ties to even. n <= 0 is a left shift by -n.

    Ties to even has zero mean error. Round half up has a mean error of
    2^-(n+1) LSB per operation, and the filter step integrates that bias.
    """
    if n <= 0:
        return x << (-n)
    q = x >> n
    rem = x & ((1 << n) - 1)
    half = 1 << (n - 1)
    return q + (rem > half) + ((rem == half) & (q & 1))


def sat(x, w: int):
    """Clamp to the range of a w-bit two's-complement word."""
    lim = 1 << (w - 1)
    return np.clip(x, -lim, lim - 1)


def _exact_int(a: np.ndarray, name: str) -> np.ndarray:
    r = np.rint(a)
    if not np.array_equal(r, a):
        raise ValueError(f"{name} must have integer entries")
    return r.astype(np.int64)


def _cmul(ar, ai, br, bi):
    """Complex matrix product on integer (real, imaginary) parts."""
    return ar @ br - ai @ bi, ar @ bi + ai @ br


def default_shift(Ts: float, prm: Plant) -> int:
    """Largest s with |dy| 2^s < 2 for |dy| <= c Ts + 6 sqrt(Ts)."""
    bound = prm.c * Ts + 6.0 * math.sqrt(Ts)
    return max(0, math.floor(math.log2(2.0 / bound)))


class FixedFilter:
    """Filter and offset law in w-bit fixed point at the sample period Ts."""

    def __init__(self, w: int, Ts: float, prm: Plant, k: float, wc: int = 12,
                 umax: float = 4.0, u0: np.ndarray = U_0, shift: int | None = None):
        if w < 4:
            raise ValueError("need w >= 4")
        if prm.meas != "ZZ":
            raise ValueError("the fixed-point filter implements the full-parity model only")
        self.w, self.wc, self.Ts = w, wc, Ts
        self.F = w - 2
        self.s = default_shift(Ts, prm) if shift is None else shift
        self.FD = self.F + self.s
        F, FD = self.F, self.FD
        e = math.log2(umax)
        if e != int(e):
            raise ValueError("umax must be a power of two")
        self.du_exp = int(e) - wc + 1  # du = 2^du_exp
        self.du = 2.0 ** self.du_exp
        etaM = prm.eta * prm.M
        self.C_b = round(math.sqrt(etaM) * 2**F)
        self.C_e = round(etaM / 2 * 2**F)
        self.C_a0 = round(prm.M * Ts / 2 * 2**FD)
        self.C_dt = round(Ts * 2**FD)
        self.C_w = round(prm.omega * Ts * 2**FD)
        self.C_g = round((1 - prm.eta) * prm.M * Ts * 2**FD)
        self.C_u = round(self.du * Ts * 2 ** (FD + GUARD))
        self.C_k = round(k * 2**GUARD)
        if self.C_k != k * 2**GUARD:
            raise ValueError(f"k = {k} needs more than {GUARD} fraction bits")
        self.u0 = _exact_int(np.asarray(u0, dtype=float), "u0")
        self.h_rows = _exact_int(H_ROWS, "H_ROWS")
        self.basis_r = _exact_int(BASIS.real, "Re BASIS")
        self.basis_i = _exact_int(BASIS.imag, "Im BASIS")
        self.hj_r = _exact_int(H_J.real, "Re H_J")
        self.hj_i = _exact_int(H_J.imag, "Im H_J")

    def encode(self, rho: np.ndarray):
        """Density matrices (N, 4, 4) -> state words (real, imaginary)."""
        scale = float(2**self.F)
        return (sat(np.rint(rho.real * scale).astype(np.int64), self.w),
                sat(np.rint(rho.imag * scale).astype(np.int64), self.w))

    def decode(self, R: np.ndarray, I: np.ndarray) -> np.ndarray:
        return (R + 1j * I) / float(2**self.F)

    def adc(self, dy: np.ndarray) -> np.ndarray:
        """Output increment over one sample period -> a w-bit word (FD fraction bits)."""
        return sat(np.rint(dy * float(2**self.FD)).astype(np.int64), self.w)

    def codes(self, R: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Command codes n (N, 4) of the offset law from the state words."""
        x = np.einsum("aij,nji->na", self.basis_r, R) - np.einsum("aij,nji->na", self.basis_i, I)
        h = x @ self.h_rows.T  # F fraction bits
        acc = (self.u0 << (self.F + GUARD)) + self.C_k * h  # F + GUARD fraction bits
        return sat(rnd(acc, self.F + GUARD + self.du_exp), self.wc)

    def command(self, n: np.ndarray) -> np.ndarray:
        """Codes -> the input the plant gets."""
        return n * self.du

    def delta(self, n: np.ndarray, dy_w: np.ndarray):
        """delta = K - 1 as words with FD fraction bits. n (N, 4), dy_w (N,)."""
        F, FD = self.F, self.FD
        dy2 = rnd(dy_w * dy_w, FD)
        beta = rnd(self.C_b * dy_w, F)
        alpha = -self.C_a0 + rnd(self.C_e * (dy2 - self.C_dt), F)
        ud = rnd(n * self.C_u, GUARD)  # Ts u_j
        hd_r = np.einsum("nj,jab->nab", ud, self.hj_r)
        hd_i = np.einsum("nj,jab->nab", ud, self.hj_i)
        idx = np.arange(4)
        hd_r[:, idx, idx] += self.C_w * ZZ_DIAG
        # delta = (alpha + beta ZZ) - i (hd_r + i hd_i) = (alpha + beta ZZ + hd_i) - i hd_r
        d_r = hd_i
        d_r[:, idx, idx] += alpha[:, None] + beta[:, None] * ZZ_DIAG
        return sat(d_r, self.w), sat(-hd_r, self.w)

    def step(self, R: np.ndarray, I: np.ndarray, n: np.ndarray, dy_w: np.ndarray):
        """One filter step. State words in, state words out."""
        F, FD, s = self.F, self.FD, self.s
        d_r, d_i = self.delta(n, dy_w)
        a_r, a_i = _cmul(d_r, d_i, R, I)
        a_r, a_i = rnd(a_r, F), rnd(a_i, F)  # A = delta rho, FD fraction bits
        dT_r, dT_i = d_r.transpose(0, 2, 1), -d_i.transpose(0, 2, 1)
        b_r, b_i = _cmul(a_r, a_i, dT_r, dT_i)
        b_r, b_i = rnd(b_r, FD), rnd(b_i, FD)  # B = A delta^dag
        g_r, g_i = rnd(self.C_g * ZZ_SIGN * R, F), rnd(self.C_g * ZZ_SIGN * I, F)
        s_r = a_r + a_r.transpose(0, 2, 1) + b_r + g_r
        s_i = a_i - a_i.transpose(0, 2, 1) + b_i + g_i
        R1, I1 = R + rnd(s_r, s), I + rnd(s_i, s)
        t = np.maximum(np.einsum("nii->n", R1), 1 << (F - 2))
        r = ((1 << (2 * F)) + t // 2) // t
        R2, I2 = rnd(R1 * r[:, None, None], F), rnd(I1 * r[:, None, None], F)
        # The core stores the upper triangle in w-bit words and reads the lower
        # one as its mirror, so saturate first and mirror after: a lower
        # imaginary entry can then be +2^(w-1), one past the w-bit range.
        up_r, up_i = np.triu(sat(R2, self.w)), np.triu(sat(I2, self.w), 1)
        R3 = up_r + np.triu(up_r, 1).transpose(0, 2, 1)
        I3 = up_i - up_i.transpose(0, 2, 1)
        return R3, I3


def simulate_fixed(rho0: np.ndarray, flt: FixedFilter, prm: Plant, T: float, dt: float,
                   seed: int, tau: float = 0.0, rho_hat0: np.ndarray | None = None) -> dict:
    """Closed loop: float plant, fixed-point filter and law.

    The loop order matches sim.closed_loop.simulate with hooks (Ts, tau,
    w = wc, umax) and a separate observer: sample, command, delay, plant
    step, filter step at the end of each sample period. The noise comes
    from numpy default_rng(seed) one step at a time, as in simulate, so a
    run with the same seed sees the same Wiener increments.
    Returns the final F of the true state, the final F of the estimate,
    and the final trace distance between them.
    """
    n_steps = _whole_steps(T, dt, "T")
    n_s = _whole_steps(flt.Ts, dt, "Ts")
    d = _whole_steps(tau, dt, "tau")
    N = rho0.shape[0]
    rng = np.random.default_rng(seed)
    sqdt = np.sqrt(dt)
    rho = rho0.copy()
    R, I = flt.encode(rho0 if rho_hat0 is None else rho_hat0)
    delay = DelayLine(d, np.zeros((N, 4)))
    n = np.zeros((N, 4), dtype=np.int64)
    dy_acc = np.zeros(N)
    for k in range(n_steps):
        if k % n_s == 0:
            n = flt.codes(R, I)
        u = delay.push(flt.command(n))
        rho, dy = plant_step(rho, u, rng.normal(0.0, sqdt, N), dt, prm)
        dy_acc += dy
        if (k + 1) % n_s == 0:
            R, I = flt.step(R, I, n, flt.adc(dy_acc))
            dy_acc = np.zeros(N)
    rho_hat = flt.decode(R, I)
    err = 0.5 * np.abs(np.linalg.eigvalsh(rho_hat - rho)).sum(axis=1)
    return dict(F=fidelity(rho), F_hat=fidelity(rho_hat), est_err=err)
