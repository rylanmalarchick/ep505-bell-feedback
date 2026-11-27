"""Sample-period limit of the held offset law from the linearization at x_bar.

Exact part (sympy):
    Near x_bar the law u = u0 + k h(x) adds to the drift Jacobian the term
        J_fb = k sum_j (B_j x_bar) (grad h_j)^T = -k sum_j b_j b_j^T,  b_j = B_j x_bar,
    because grad h_j = B_j^T x_bar = -B_j x_bar (B_j antisymmetric). The
    script finds the eigenvalues of G = sum_j b_j b_j^T.
Numeric part (numpy, scipy expm):
    With the input held over a period Ts, the linearized sampled loop is
        z_{n+1} = Phi(Ts) z_n,  Phi = e^{A Ts} + (int_0^Ts e^{A s} ds) J_fb,
    with A = A0 + sum_j u0_j B_j (the drift with the input frozen). The script
    finds, for each k, the smallest Ts at which an eigenvalue of Phi leaves
    the unit disk through -1 (a flip), and compares it with 2 / (16 k).
    This is a necessary condition for local stability of the sampled loop
    without noise. It does not include the measurement noise, the filter,
    the delay, or the quantizer.

    python check_sampled.py
"""
import numpy as np
import sympy as sp
from scipy.linalg import expm
from sympy.physics.quantum import TensorProduct as T

from coords_spec import H_LAB, LAB, U0, XBAR

PAULI = {
    "I": sp.eye(2),
    "X": sp.Matrix([[0, 1], [1, 0]]),
    "Y": sp.Matrix([[0, -sp.I], [sp.I, 0]]),
    "Z": sp.diag(1, -1),
}


def P(label):
    return T(PAULI[label[0]], PAULI[label[1]])


def comm(a, b):
    return a * b - b * a


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("PASS", name)


BASIS = [P(lab) for lab in LAB]


def coord_matrix(op):
    return sp.Matrix(15, 15, lambda a, b: sp.nsimplify(sp.expand((BASIS[a] * op(BASIS[b])).trace() / 4)))


B = [coord_matrix(lambda r, H=P(lab): -sp.I * comm(H, r)) for lab in H_LAB]
xbar = sp.Matrix(XBAR)
b = [Bj * xbar for Bj in B]
check("grad h_j = B_j^T xbar = -B_j xbar", all((Bj.T * xbar + Bj * xbar).is_zero_matrix for Bj in B))
check("every b_j = B_j xbar lies in the q block (coherences of Phi+ with the odd sector)",
      all(bj[:7, 0].is_zero_matrix for bj in b))
G = sum((bj * bj.T for bj in b), sp.zeros(15))
ev = G.eigenvals()
print("eigenvalues of G = sum_j b_j b_j^T:", ev)
check("G has eigenvalue 16 with multiplicity 2 and 0 with multiplicity 13", ev == {16: 2, 0: 13})

# Numeric sampled-data map.
M, omega = 1.0, 0.3
ZZ = P("ZZ")
A0 = np.array(coord_matrix(lambda r: -sp.I * omega * comm(ZZ, r) + M * (ZZ * r * ZZ - r)), dtype=float)
Bn = [np.array(Bj, dtype=float) for Bj in B]
A = A0 + sum(U0[j] * Bn[j] for j in range(4))
Gn = np.array(G, dtype=float)


def phi(Ts, k):
    n = 15
    # Integral of e^{A s} over [0, Ts] from the block exponential.
    big = np.zeros((2 * n, 2 * n))
    big[:n, :n] = A
    big[:n, n:] = np.eye(n)
    E = expm(big * Ts)
    return E[:n, :n] + E[:n, n:] @ (-k * Gn)


def flip_ts(k, lo=1e-4, hi=1.0):
    """Smallest Ts with an eigenvalue of Phi below -1, by bisection on a grid start."""
    grid = np.linspace(lo, hi, 2000)
    bad = [Ts for Ts in grid if np.linalg.eigvals(phi(Ts, k)).real.min() < -1 - 1e-12]
    if not bad:
        return np.nan
    a, c = grid[np.searchsorted(grid, bad[0]) - 1], bad[0]
    for _ in range(60):  # bisection to machine precision
        mid = 0.5 * (a + c)
        if np.linalg.eigvals(phi(mid, k)).real.min() < -1 - 1e-12:
            c = mid
        else:
            a = mid
    return c


print(" k    | flip Ts (numeric) | 2 / (16 k) | ratio")
for k in (0.5, 1.0, 2.0, 4.0, 8.0):
    ts = flip_ts(k)
    print(f" {k:4g} | {ts:.5f}           | {2 / (16 * k):.5f}    | {ts / (2 / (16 * k)):.4f}")
    check(f"k = {k}: flip within 5% of 2 / (16 k)", abs(ts / (2 / (16 * k)) - 1) < 0.05)

# Delay (numeric): the linear delay system z' = A z + J_fb z(t - tau), input not
# held (Ts -> 0). Growth rate of |z| from a fixed start over t in [10, 20],
# Euler steps of 1e-4. Positive growth means the delayed linear loop is unstable.
def growth(k, tau, h=1e-4, T=20.0):
    n = round(T / h)
    d = round(tau / h)
    J = -k * Gn
    z = np.zeros((n + 1, 15))
    z[0] = np.linalg.eigh(Gn)[1][:, -1]  # start on the fast direction
    for i in range(n):
        z[i + 1] = z[i] + h * (A @ z[i] + J @ z[max(i - d, 0)])
    nrm = np.linalg.norm(z, axis=1)
    i10, i20 = round(10.0 / h), n
    return (np.log(nrm[i20] + 1e-300) - np.log(nrm[i10] + 1e-300)) / 10.0


print(" k = 2, delay tau | growth rate of the linear delayed loop")
taus = (0.02, 0.04, 0.045, 0.05, 0.055, 0.06, 0.08, 0.1)
g = {tau: growth(2.0, tau) for tau in taus}
for tau in taus:
    print(f"  {tau:5g}          | {g[tau]:+.3f}")
stable = [t for t in taus if g[t] < 0]
unstable = [t for t in taus if g[t] > 0]
print(f"  stable up to tau = {max(stable):g}, unstable from tau = {min(unstable):g}")
check("k = 2: the delayed linear loop turns unstable between tau = 0.045 and 0.055",
      max(stable) >= 0.045 and min(unstable) <= 0.055 and max(stable) < min(unstable))

# Mean-square stability of the linearization on all of R^15 (numeric). The
# second moment of z obeys dP/dt = J P + P J^T + N P N^T, with J the drift
# Jacobian and N the Jacobian of the noise gain at xbar. Result: J has an
# eigenvalue 0 and the operator an eigenvalue 4 c^2 > 0, so the linearization
# on R^15 is not mean-square exponentially stable. xbar is an extreme point of
# K, so this does not decide the stability of the loop on K.
from coords_spec import P_LAB  # noqa: E402
c_ = 2 * np.sqrt(0.3 * M)
Sn = np.zeros((7, 7))
for a_, b2, s_ in [("IZ", "ZI", 1), ("XX", "YY", -1), ("XY", "YX", 1)]:
    i_, j_ = P_LAB.index(a_), P_LAB.index(b2)
    Sn[i_, j_] = Sn[j_, i_] = s_
xb = np.array(XBAR, dtype=float)
Nb = np.zeros((15, 15))
Nb[:7, :7] = c_ * (Sn - np.eye(7) - np.outer(xb[:7], np.eye(7)[0]))
Nb[7:, 7:] = -c_ * np.eye(8)
for k in (0.5, 2.0):
    J = A - k * Gn
    op = np.kron(np.eye(15), J) + np.kron(J, np.eye(15)) + np.kron(Nb, Nb)
    top_op, top_j = np.linalg.eigvals(op).real.max(), np.linalg.eigvals(J).real.max()
    print(f" k = {k}: largest real part, drift Jacobian {top_j:+.2e}, second-moment operator {top_op:+.4f}")
    check(f"k = {k}: the linearization on R^15 is not mean-square exponentially stable (4 c^2 = {4 * c_**2:.2f})",
          abs(top_j) < 1e-9 and abs(top_op - 4 * c_**2) < 1e-9)
print("\nall checks passed")
