"""Exact checks for the half-parity model and for the zero-fidelity set.

Part A: the zero-fidelity set {F = 0} is invariant under the offset law,
        for the full-parity model (target Phi+) and the half-parity model
        (target Psi+). A state with F = 0 satisfies rho psi = 0. The check
        shows that the drift and the noise gain map such a state to a
        matrix that also annihilates psi.
Part B: the half-parity model, L = sqrt(M) diag(1, 0, 0, -1):
        offsets, the generator of 1 - F^2, and the largest invariant set
        of its zero set off {F = 0}.

    python check_half_parity.py
"""
import sympy as sp
from sympy.physics.quantum import TensorProduct as T

from coords_spec import H_LAB, LAB, U0

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


omega, M, eta, k = sp.symbols("omega M eta k", positive=True)
u = sp.symbols("u1:5", real=True)
H_J = [P(lab) for lab in H_LAB]
ZZ = P("ZZ")
HALF = (P("ZI") + P("IZ")) / 2
phi_p = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)
psi_p = sp.Matrix([0, 1, 1, 0]) / sp.sqrt(2)
psi_m = sp.Matrix([0, 1, -1, 0]) / sp.sqrt(2)
U0_X, U0_Y = (1, 0, -1, 0), (0, 1, 0, -1)
check("coords_spec U0 is X1 - X2", tuple(U0) == U0_X)


def sme(L, uvec):
    """Drift and noise-gain maps of the SME for measurement operator L."""
    H = omega * ZZ + sum((uvec[j] * H_J[j] for j in range(4)), sp.zeros(4))

    def drift(r):
        return -sp.I * comm(H, r) + L * r * L - (L * L * r + r * L * L) / 2

    def diff(r):
        return sp.sqrt(eta) * (L * r + r * L - 2 * (L * r).trace() * r)
    return drift, diff


def h_of(r, target):
    """h_j = xbar^T B_j x = 4 Tr(rho_bar (-i [H_j, rho])), the damping functions."""
    rb = target * target.T
    return [sp.expand(4 * (rb * (-sp.I) * comm(Hj, r)).trace()) for Hj in H_J]


def hermitian(prefix, n):
    """Generic n x n Hermitian matrix of real symbols."""
    m = sp.zeros(n)
    for i in range(n):
        m[i, i] = sp.Symbol(f"{prefix}{i}{i}", real=True)
        for j in range(i + 1, n):
            a, b = sp.symbols(f"{prefix}r{i}{j} {prefix}i{i}{j}", real=True)
            m[i, j] = a + sp.I * b
            m[j, i] = a - sp.I * b
    return m


def complement_basis(v):
    """Orthonormal basis (columns) of the complement of the unit vector v."""
    cols = [v]
    for e in sp.eye(4).columnspace():
        w = e - sum(((c.T * e)[0] * c for c in cols), sp.zeros(4, 1))
        if sp.simplify((w.T * w)[0]) != 0:
            cols.append(sp.simplify(w / sp.sqrt((w.T * w)[0])))
        if len(cols) == 4:
            break
    return sp.Matrix.hstack(*cols[1:])


# ---------------------------------------------------------------- Part A
for name, L, target, offsets in [
    ("full parity, Phi+", sp.sqrt(M) * ZZ, phi_p, [U0_X]),
    ("half parity, Psi+", sp.sqrt(M) * HALF, psi_p, [U0_X, U0_Y]),
]:
    Q = complement_basis(target)
    check(f"{name}: complement basis is orthonormal and orthogonal to the target",
          sp.simplify(Q.T * Q - sp.eye(3)).is_zero_matrix and sp.simplify(Q.T * target).is_zero_matrix)
    R = hermitian("R", 3)
    rho0 = Q * R * Q.T  # every state with F = 0 has this form
    check(f"{name}: rho0 psi = 0", sp.simplify(rho0 * target).is_zero_matrix)
    for u0 in offsets:
        h0 = h_of(rho0, target)
        check(f"{name}, u0 = {u0}: h = 0 on the zero-fidelity set", all(sp.simplify(x) == 0 for x in h0))
        drift, diff = sme(L, [u0[j] + k * h0[j] for j in range(4)])
        check(f"{name}, u0 = {u0}: drift(rho0) psi = 0",
              sp.simplify(drift(rho0) * target).is_zero_matrix)
        check(f"{name}, u0 = {u0}: noise gain(rho0) psi = 0",
              sp.simplify(diff(rho0) * target).is_zero_matrix)

# Full parity: Psi+ is a closed-loop equilibrium of the offset law with F = 0.
rho_pp = psi_p * psi_p.T
drift, diff = sme(sp.sqrt(M) * ZZ, [U0_X[j] + k * h_of(rho_pp, phi_p)[j] for j in range(4)])
check("full parity: Psi+ is an equilibrium of the offset law (drift and noise gain 0)",
      sp.simplify(drift(rho_pp)).is_zero_matrix and sp.simplify(diff(rho_pp)).is_zero_matrix)

# ---------------------------------------------------------------- Part B
L = sp.sqrt(M) * HALF
rho = hermitian("r", 4)
rho = rho.subs(rho[3, 3], 1 - rho[0, 0] - rho[1, 1] - rho[2, 2])
rb = psi_p * psi_p.T
F = sp.expand((rho * rb).trace())
m = sp.expand(rho[0, 0] - rho[3, 3])
c = 2 * sp.sqrt(eta * M)

check("L psi+ = 0, ZZ psi+ = -psi+", (L * psi_p).is_zero_matrix and (ZZ * psi_p + psi_p).is_zero_matrix)
for u0 in (U0_X, U0_Y):
    Hc = sum((u0[j] * H_J[j] for j in range(4)), sp.zeros(4))
    check(f"half, u0 = {u0}: H_c psi+ = 0", (Hc * psi_p).is_zero_matrix)
    check(f"half, u0 = {u0}: H_c psi- != 0", not (Hc * psi_m).is_zero_matrix)

h = h_of(rho, psi_p)
for u0 in (U0_X, U0_Y):
    check(f"half, u0 = {u0}: sum_j u0_j h_j = 0", sp.expand(sum(u0[j] * h[j] for j in range(4))) == 0)
    drift, diff = sme(L, [u0[j] + k * h[j] for j in range(4)])
    dF = sp.expand((drift(rho) * rb).trace())
    gF = sp.expand((diff(rho) * rb).trace())
    check(f"half, u0 = {u0}: drift of F = (k/4) sum h^2", sp.expand(dF - k / 4 * sum(x**2 for x in h)) == 0)
    check(f"half, u0 = {u0}: noise gain of F = -c m F", sp.simplify(gF + c * m * F) == 0)
    AV2 = sp.expand(-2 * F * dF - gF**2)
    check(f"half, u0 = {u0}: A(1 - F^2) = -(k/2) F sum h^2 - 4 eta M m^2 F^2",
          sp.simplify(AV2 - (-(k / 2) * F * sum(x**2 for x in h) - 4 * eta * M * m**2 * F**2)) == 0)
    gm = sp.expand(sp.Matrix([1, 0, 0, -1]).T * sp.diag(*[1] * 4) * sp.Matrix([diff(rho)[i, i] for i in range(4)]))[0]
    check(f"half, u0 = {u0}: noise gain of m = c (rho_00 + rho_33 - m^2)",
          sp.simplify(gm - c * (rho[0, 0] + rho[3, 3] - m**2)) == 0)

# Zero set off {F = 0}: m = 0 and h = 0. An invariant subset keeps m = 0, so
# the noise gain of m is 0: rho_00 + rho_33 = 0, and positivity puts rho in
# the odd sector. On the odd sector, write rho by (a, br, bi).
a, br, bi = sp.symbols("a b_r b_i", real=True)
rho_odd = sp.zeros(4)
rho_odd[1, 1], rho_odd[2, 2] = a, 1 - a
rho_odd[1, 2], rho_odd[2, 1] = br + sp.I * bi, br - sp.I * bi
odd_idx = (1, 2)
for u0 in (U0_X, U0_Y):
    hodd = [sp.expand(x) for x in h_of(rho_odd, psi_p)]
    drift, _ = sme(L, [u0[j] + k * hodd[j] for j in range(4)])
    D = sp.expand(drift(rho_odd))
    leak = [D[i, j] for i in range(4) for j in range(4) if not (i in odd_idx and j in odd_idx)]
    eqs = [sp.expand(e) for e in hodd] + [sp.re(e) for e in leak] + [sp.im(e) for e in leak]
    sol = sp.solve([e for e in eqs if e != 0], [a, br, bi], dict=True)
    check(f"half, u0 = {u0}: the only invariant odd-sector state with h = 0 is Psi+ (a = b_r = 1/2, b_i = 0)",
          sol == [{a: sp.Rational(1, 2), br: sp.Rational(1, 2), bi: 0}])

print("\nall checks passed")
