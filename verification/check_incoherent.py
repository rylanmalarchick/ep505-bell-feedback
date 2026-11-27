"""The set B of states with no coherence to the target (exact checks).

B = { rho : rho |Phi+> = F |Phi+> },  i.e. rho = F |Phi+><Phi+| + R with R |Phi+> = 0.
B contains {F = 0} (F = 0) and states arbitrarily close to the target
(F = 1 - delta). Claims checked here, for the full-parity model, the offset
law u = u0 + k h(x), and symbolic F, R:
  1. h = 0 on B, so the input is u0.
  2. The drift and the noise gain map B into matrices whose |Phi+> column is
     a multiple of |Phi+>: B is invariant.
  3. On B the drift of F is 0, so F is a martingale.
  4. The Kraus operator of one filter step, K = 1 + (-i H(u0) - L^2/2) dt +
     sqrt(eta) L dy + (eta/2) L^2 (dy^2 - dt), has |Phi+> as an eigenvector for
     every output increment dy, so a filter started in B stays in B whatever
     the measured record is.
  5. The filter start (|00><00| + |11><11|)/2 of the wrong-estimate run with
     s = 0.5 lies in B (it equals (Phi+ + Phi-)/2 in projector form).

    python check_incoherent.py
"""
import sympy as sp
from sympy.physics.quantum import TensorProduct as T

from coords_spec import H_LAB, U0

PAULI = {"I": sp.eye(2), "X": sp.Matrix([[0, 1], [1, 0]]), "Y": sp.Matrix([[0, -sp.I], [sp.I, 0]]),
         "Z": sp.diag(1, -1)}


def P(label):
    return T(PAULI[label[0]], PAULI[label[1]])


def comm(a, b):
    return a * b - b * a


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("PASS", name)


omega, M, eta, k, dt, dy = sp.symbols("omega M eta k dt dy", real=True)
ZZ = P("ZZ")
H_J = [P(lab) for lab in H_LAB]
phi_p = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)
L = sp.sqrt(M) * ZZ

# A basis of the complement of Phi+, and a generic Hermitian R on it.
Q = sp.Matrix.hstack(sp.Matrix([1, 0, 0, -1]) / sp.sqrt(2), sp.Matrix([0, 1, 1, 0]) / sp.sqrt(2),
                     sp.Matrix([0, 1, -1, 0]) / sp.sqrt(2))
check("complement basis orthonormal, orthogonal to Phi+",
      sp.simplify(Q.T * Q - sp.eye(3)).is_zero_matrix and sp.simplify(Q.T * phi_p).is_zero_matrix)
Fs = sp.Symbol("F", real=True)
R3 = sp.zeros(3)
for i in range(3):
    R3[i, i] = sp.Symbol(f"r{i}{i}", real=True)
    for j in range(i + 1, 3):
        a_, b_ = sp.symbols(f"a{i}{j} b{i}{j}", real=True)
        R3[i, j], R3[j, i] = a_ + sp.I * b_, a_ - sp.I * b_
rho = Fs * phi_p * phi_p.T + Q * R3 * Q.T
rb = phi_p * phi_p.T

h = [sp.expand(4 * (rb * (-sp.I) * comm(Hj, rho)).trace()) for Hj in H_J]
check("1. h = 0 on B", all(sp.simplify(x) == 0 for x in h))
u = [U0[j] + k * h[j] for j in range(4)]
H = omega * ZZ + sum((u[j] * H_J[j] for j in range(4)), sp.zeros(4))
drift = -sp.I * comm(H, rho) + L * rho * L - (L * L * rho + rho * L * L) / 2
diff = sp.sqrt(eta) * (L * rho + rho * L - 2 * (L * rho).trace() * rho)
perp = sp.eye(4) - rb  # projector on the complement


def keeps_B(Dm):
    return sp.simplify(perp * Dm * phi_p).is_zero_matrix


check("2. drift keeps B (no coherence to Phi+ created)", keeps_B(drift))
check("2. noise gain keeps B", keeps_B(diff))
dF = sp.simplify((rb * drift).trace())
check("3. drift of F = 0 on B (F is a martingale on B)", dF == 0)
gF = sp.simplify((rb * diff).trace())
print("   noise gain of F on B:", gF)

H0 = omega * ZZ + sum((U0[j] * H_J[j] for j in range(4)), sp.zeros(4))
K = sp.eye(4) + (-sp.I * H0 - L * L / 2) * dt + sp.sqrt(eta) * L * dy + eta / 2 * L * L * (dy**2 - dt)
check("4. every filter Kraus operator with input u0 has Phi+ as an eigenvector",
      sp.simplify(perp * K * phi_p).is_zero_matrix)

r00 = sp.zeros(4)
r00[0, 0] = r00[3, 3] = sp.Rational(1, 2)
check("5. (|00><00| + |11><11|)/2 lies in B", sp.simplify(perp * r00 * phi_p).is_zero_matrix)
r_eps = (1 - sp.Symbol("e")) * sp.Matrix([1, 0, 0, -1]) * sp.Matrix([1, 0, 0, -1]).T / 2 + sp.Symbol("e") * sp.eye(4) / 4
check("5. the Table IV start (1 - eps) Phi- + eps I/4 lies in B", sp.simplify(perp * r_eps * phi_p).is_zero_matrix)
print("\nall checks passed")

# 6. Everywhere on the state set (not only on B): H(u0) commutes with the
#    target projector, so sum_j u0_j h_j = 0 and the drift of F under the
#    offset law is (k/4) sum_j h_j^2 >= 0: F is a bounded submartingale.
check("6. [H(u0), |Phi+><Phi+|] = 0", sp.simplify(comm(H0, rb)).is_zero_matrix)
Gm = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"g{i}{j}"))
rho_g = Gm  # any matrix: the identities below are linear in rho
h_g = [4 * (rb * (-sp.I) * comm(Hj, rho_g)).trace() for Hj in H_J]
u_g = [U0[j] + k * h_g[j] for j in range(4)]
Hg = omega * ZZ + sum((u_g[j] * H_J[j] for j in range(4)), sp.zeros(4))
drift_g = -sp.I * comm(Hg, rho_g) + L * rho_g * L - (L * L * rho_g + rho_g * L * L) / 2
check("6. drift of F = (k/4) sum_j h_j^2 for every rho",
      sp.expand((rb * drift_g).trace() - k / 4 * sum(x**2 for x in h_g)) == 0)
print("all checks including 6 passed")
