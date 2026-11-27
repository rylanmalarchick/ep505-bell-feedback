"""Constant inputs that leave the zero-fidelity set.

Under a constant input u_c the Kraus operators of the filter are polynomials
in L = sqrt(M) ZZ and H = omega ZZ + sum_j u_c,j H_j. A subspace W orthogonal
to the target is invariant for the filter if and only if it is invariant
under ZZ and H. Such a W exists if and only if the smallest subspace that
contains the target and is invariant under ZZ and H is not all of C^4
(take W as its orthogonal complement; ZZ and H are Hermitian).

The script computes that dimension for several inputs. Dimension 4 means
no state with F = 0 stays at F = 0 under u_c: the switching law can leave
the set. Dimension < 4 means a face of {F = 0} is invariant under u_c too.

    python check_switch.py
"""
import sympy as sp
from sympy.physics.quantum import TensorProduct as T

from coords_spec import H_LAB

PAULI = {
    "I": sp.eye(2),
    "X": sp.Matrix([[0, 1], [1, 0]]),
    "Y": sp.Matrix([[0, -sp.I], [sp.I, 0]]),
    "Z": sp.diag(1, -1),
}


def P(label):
    return T(PAULI[label[0]], PAULI[label[1]])


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("PASS", name)


ZZ = P("ZZ")
H_J = [P(lab) for lab in H_LAB]
omega = sp.Rational(3, 10)
phi_p = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)


def krylov_dim(v, mats):
    basis = []

    def add(w):
        trial = sp.Matrix.hstack(*basis, w) if basis else w
        if trial.rank(simplify=True) > len(basis):
            basis.append(w)

    add(v)
    i = 0
    while i < len(basis):  # at most 4 vectors
        for A in mats:
            add(sp.expand(A * basis[i]))
        i += 1
    return len(basis)


results = {}
for u in [(1, 0, 0, 0), (0, 1, 0, 0), (1, 0, -1, 0), (0, 1, 0, 1), (1, 1, 0, 0), (1, 0, 0, 1),
          (1, 1, 1, 0)]:
    H = omega * ZZ + sum((u[j] * H_J[j] for j in range(4)), sp.zeros(4))
    results[u] = krylov_dim(phi_p, [ZZ, H])
    print(f"u_c = {u}: smallest invariant subspace containing Phi+ has dimension {results[u]}")
check("X1 alone keeps a face of {F = 0} invariant (dimension 2)", results[(1, 0, 0, 0)] == 2)
check("Y1 alone keeps a face of {F = 0} invariant (dimension 2)", results[(0, 1, 0, 0)] == 2)
check("the offset u0 = X1 - X2 keeps a face of {F = 0} invariant", results[(1, 0, -1, 0)] < 4)
check("the offset u0 has Phi+ as an eigenvector (dimension 1)", results[(1, 0, -1, 0)] == 1)
# Drives on qubit 1 alone commute with Z on qubit 2, and so does ZZ.
check("X1 + Y1 keeps a face invariant (dimension 2: Z on qubit 2 is conserved)", results[(1, 1, 0, 0)] == 2)
check("X1 + Y2 keeps a face invariant (dimension 3)", results[(1, 0, 0, 1)] == 3)
check("u_c = X1 + Y1 + X2 leaves no face of {F = 0} invariant", results[(1, 1, 1, 0)] == 4)
print("\nall checks passed")
