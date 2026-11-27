"""Smallest exact linear reduction of the filter for a given feedback law.

The filter in x coordinates is
    dx = (A0 + sum_j u_j B_j) x dt + b(x) dW,  b(x) = c (e1 + S p - p1 p, -p1 q).
For a linear functional l(x) = v^T x,
    d l = v^T (A0 + sum_j u_j B_j) x dt + c (v_1 + v_p^T S p - p1 l) dW.
Let V be a subspace of functionals that contains p1 = e1^T x. If V is
invariant under A^T for every matrix A that the law puts in the drift, and
under the map v -> (S v_p, 0), then the coordinates of x in V obey a closed
SDE: the drift is linear in V, the law reads only V (if h_j and F are in
V), and the noise term is c (constant + linear in V - p1 l). So dim V is the
dimension of an exact reduced filter.

This script computes the smallest such V that contains p1, F, and the
damping functions h_j the law needs, in exact rational arithmetic, for:
    generic   four independent inputs: A0, B1, B2, B3, B4
    offset    u = u0 + k h: A0 + B1 - B3, B1 + B3, B2 - B4
    baseline  u = k h:      A0, B1 + B3, B2 - B4
    unforced  u = 0:        A0                      (the (Lambda, c) closure of the proposal)
for two (M, omega) pairs. The unforced case starts from p1 and F only.

    python check_reduction.py
"""
import sympy as sp
from sympy.physics.quantum import TensorProduct as T

from coords_spec import H_LAB, LAB, P_LAB, XBAR

PAULI = {
    "I": sp.eye(2),
    "X": sp.Matrix([[0, 1], [1, 0]]),
    "Y": sp.Matrix([[0, -sp.I], [sp.I, 0]]),
    "Z": sp.diag(1, -1),
}
NP = len(P_LAB)


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
    """Matrix of rho -> op(rho) in x coordinates: [a, b] = Tr(P_a op(P_b)) / 4."""
    return sp.Matrix(15, 15, lambda a, b: sp.nsimplify(sp.expand((BASIS[a] * op(BASIS[b])).trace() / 4)))


B = [coord_matrix(lambda r, H=P(lab): -sp.I * comm(H, r)) for lab in H_LAB]
ZZ = P("ZZ")


def a0(M, omega):
    return coord_matrix(lambda r: -sp.I * omega * comm(ZZ, r) + M * (ZZ * r * ZZ - r))


S = sp.zeros(NP)
for a, b_, s in [("IZ", "ZI", 1), ("XX", "YY", -1), ("XY", "YX", 1)]:
    i, j = P_LAB.index(a), P_LAB.index(b_)
    S[i, j] = S[j, i] = s


def s_map(v):
    out = sp.zeros(15, 1)
    out[:NP, 0] = S.T * v[:NP, 0]
    return out


def closure(start, mats):
    basis = []

    def add(v):
        trial = sp.Matrix.hstack(*basis, v) if basis else v
        if trial.rank() > len(basis):
            basis.append(v)
            return True
        return False

    for v in start:
        add(v)
    i = 0
    while i < len(basis):  # each vector is processed once; the basis has at most 15
        v = basis[i]
        for A in mats:
            add(A.T * v)
        add(s_map(v))
        i += 1
    return len(basis)


e1 = sp.Matrix([1] + [0] * 14)
xbar = sp.Matrix(XBAR)
hvecs = [Bj.T * xbar for Bj in B]  # h_j(x) = xbar^T B_j x = (B_j^T xbar)^T x
check("h_1 = h_3 and h_2 = -h_4 as functionals", hvecs[0] == hvecs[2] and hvecs[1] == -hvecs[3])
start = [e1, xbar] + hvecs

for M, omega in [(1, sp.Rational(3, 10)), (sp.Rational(7, 3), sp.Rational(5, 11))]:
    A0 = a0(M, omega)
    cases = {
        "generic": [A0] + B,
        "offset": [A0 + B[0] - B[2], B[0] + B[2], B[1] - B[3]],
        "baseline": [A0, B[0] + B[2], B[1] - B[3]],
    }
    dims = {name: closure(start, mats) for name, mats in cases.items()}
    dims["unforced"] = closure([e1, xbar], [A0])
    print(f"M = {M}, omega = {omega}: " + ", ".join(f"{k} {v}" for k, v in dims.items()))
    check(f"M = {M}, omega = {omega}: generic inputs need all 15 coordinates", dims["generic"] == 15)
    check(f"M = {M}, omega = {omega}: the offset law needs all 15 coordinates", dims["offset"] == 15)
    # Lambda = (1 + p1) / 2 and Re c = F - Lambda / 2. Im c does not enter:
    # H0 = omega ZZ does not rotate the |00><11| coherence (ZZ = +1 on both).
    check(f"M = {M}, omega = {omega}: u = 0 closes on 2 functionals (Lambda and Re c)",
          dims["unforced"] == 2)
    check(f"M = {M}, omega = {omega}: the baseline law closes on 9 functionals", dims["baseline"] == 9)

print("\nall checks passed")
