"""Exact symbolic checks for the control model of the report.

Builds the control-form model from the stochastic master equation and
asserts every equation stated in the document. Writes the matrices used
in the document to ../generated_matrices.tex (build output, not tracked).

    python check_control_model.py
"""
import itertools
from pathlib import Path

import sympy as sp
from sympy.physics.quantum import TensorProduct as T

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


# Coordinate order used in the document (shared with sim/).
from coords_spec import H_LAB, LAB, P_LAB, Q_LAB, U0, XBAD, XBAR
NP, NQ = len(P_LAB), len(Q_LAB)

omega, M, eta, k = sp.symbols("omega M eta k", positive=True)
u = sp.symbols("u1:5", real=True)
x = sp.Matrix(sp.symbols("x1:16", real=True))
p, q = x[:NP, 0], x[NP:, 0]
c = 2 * sp.sqrt(eta * M)

rho = (sp.eye(4) + sum((x[i] * P(l) for i, l in enumerate(LAB)), sp.zeros(4))) / 4
H_J = [P(l) for l in H_LAB]
H = omega * P("ZZ") + sum((u[j] * H_J[j] for j in range(4)), sp.zeros(4))
L = sp.sqrt(M) * P("ZZ")

drift_rho = -sp.I * comm(H, rho) + L * rho * L - (L * L * rho + rho * L * L) / 2
diff_rho = sp.sqrt(eta) * (L * rho + rho * L - 2 * (L * rho).trace() * rho)
coord = lambda m: sp.Matrix([sp.expand((P(l) * m).trace()) for l in LAB])
f = coord(drift_rho)
g = coord(diff_rho)

# 1. Drift is bilinear: f = (A0 + sum u_j B_j) x, no constant term.
Afull = f.jacobian(x)
check("drift has no constant term", f.subs({xi: 0 for xi in x}).is_zero_matrix)
check("drift linear in x", sp.expand(f - Afull * x).is_zero_matrix)
A0 = Afull.subs({ui: 0 for ui in u})
B = [sp.expand(Afull.diff(uj)) for uj in u]
check("drift = (A0 + sum u_j B_j) x",
      sp.expand(Afull - A0 - sum((u[j] * B[j] for j in range(4)), sp.zeros(15))).is_zero_matrix)

# 2. A0 block structure.
R = sp.Matrix([[-2 * M, -2 * omega], [2 * omega, -2 * M]])
Aq = sp.diag(*([R] * 4))
check("A0 = diag(0_7, I4 (x) R)", (A0 - sp.diag(sp.zeros(NP), Aq)).is_zero_matrix)
check("eigenvalues of A_q are -2M +- 2i omega",
      sp.Matrix(Aq).eigenvals() == {-2 * M - 2 * sp.I * omega: 4, -2 * M + 2 * sp.I * omega: 4})

# 3. Input matrices: B(u) = [[0, -G^T], [G, 0]].
Bu = sp.expand(sum((u[j] * B[j] for j in range(4)), sp.zeros(15)))
G = Bu[NP:, :NP]
check("B(u) has zero diagonal blocks", Bu[:NP, :NP].is_zero_matrix and Bu[NP:, NP:].is_zero_matrix)
check("B(u) antisymmetric", (Bu + Bu.T).is_zero_matrix)
check("G(u) linear in u", sp.expand(G - sum((G.diff(uj) * uj for uj in u), sp.zeros(NQ, NP))).is_zero_matrix)

# 4. Diffusion gain.
e1 = sp.Matrix([1] + [0] * (NP - 1))
S = sp.zeros(NP)
for a, b_, s in [("IZ", "ZI", 1), ("XX", "YY", -1), ("XY", "YX", 1)]:
    i, j = P_LAB.index(a), P_LAB.index(b_)
    S[i, j] = s
    S[j, i] = s
bp = c * (e1 + S * p - p[0] * p)
bq = -c * p[0] * q
check("b_p(x) = c (e1 + S p - p1 p)", sp.simplify(g[:NP, 0] - bp).is_zero_matrix)
check("b_q(x) = -c p1 q", sp.simplify(g[NP:, 0] - bq).is_zero_matrix)

# 5. Output.
check("2 sqrt(eta) Tr(L rho) = c p1", sp.simplify(2 * sp.sqrt(eta) * (L * rho).trace() - c * p[0]) == 0)

# 6. Target, performance index.
xbar = sp.Matrix(XBAR)
xbad = sp.Matrix(XBAD)
phi_p = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)
phi_m = sp.Matrix([1, 0, 0, -1]) / sp.sqrt(2)
rho_of = lambda v: (sp.eye(4) + sum((v[i] * P(l) for i, l in enumerate(LAB)), sp.zeros(4))) / 4
check("xbar is the coordinate vector of Phi+", (rho_of(xbar) - phi_p * phi_p.T).is_zero_matrix)
check("xbad is the coordinate vector of Phi-", (rho_of(xbad) - phi_m * phi_m.T).is_zero_matrix)
F = sp.expand((rho * phi_p * phi_p.T).trace())
check("F = (1 + xbar^T x)/4", sp.expand(F - (1 + (xbar.T * x)[0]) / 4) == 0)

# 7. Unforced equilibria.
at = lambda expr, v: expr.subs(dict(zip(list(x), list(v))))
f0 = A0 * x
for name, v in [("xbar", xbar), ("xbad", xbad)]:
    check(f"{name}: unforced drift = 0", at(f0, v).is_zero_matrix)
    check(f"{name}: diffusion = 0", sp.simplify(at(g, v)).is_zero_matrix)
check("xbar^T A0 = 0", (xbar.T * A0).is_zero_matrix)

# Set where unforced drift and diffusion both vanish, p1 = +1 branch.
a, bb, d = sp.symbols("a b d", real=True)
xe = sp.Matrix([1, a, a, bb, -bb, d, d] + [0] * NQ)
check("E+: drift = 0", at(f0, xe).is_zero_matrix)
check("E+: diffusion = 0", sp.simplify(at(g, xe)).is_zero_matrix)
# Necessity: diffusion p1-component is c (1 - p1^2).
check("diffusion p1-component = c (1 - p1^2)", sp.simplify(g[0] - c * (1 - p[0] ** 2)) == 0)
# With p1 = 1 and drift = 0 (q = 0 needed since A_q invertible), diffusion = 0 forces E+ form.
sub = {x[0]: 1, **{qi: 0 for qi in q}}
sol = sp.solve(list(sp.simplify(g.subs(sub))), [x[1], x[3], x[5]], dict=True)
check("p1=1, q=0, b=0 forces p2=p3, p4=-p5, p6=p7",
      sol == [{x[1]: x[2], x[3]: -x[4], x[5]: x[6]}])
sub_m = {x[0]: -1, **{qi: 0 for qi in q}}
sol_m = sp.solve(list(sp.simplify(g.subs(sub_m))), [x[1], x[3], x[5]], dict=True)
check("p1=-1, q=0, b=0 forces p2=-p3, p4=p5, p6=-p7",
      sol_m == [{x[1]: -x[2], x[3]: x[4], x[5]: -x[6]}])
lam = sp.symbols("lambda")
r2 = a**2 + bb**2 + d**2
check("E+ charpoly = lambda^2 (lambda^2 - lambda + (1 - a^2 - b^2 - d^2)/4)",
      sp.expand(rho_of(xe).charpoly(lam).as_expr() - lam**2 * (lam**2 - lam + (1 - r2) / 4)) == 0)

# Companion note: coordinate split and meaning of (a, b, d).
check("p coordinates commute with ZZ", all(comm(P(l), P("ZZ")).is_zero_matrix for l in P_LAB))
check("q coordinates anticommute with ZZ", all((P(l) * P("ZZ") + P("ZZ") * P(l)).is_zero_matrix for l in Q_LAB))
re_ = rho_of(xe)
check("E+ state: rho_00,00=(1+a)/2, rho_11,11=(1-a)/2, rho_00,11=(b - i d)/2, rest 0",
      (re_ - sp.Matrix([[(1 + a) / 2, 0, 0, (bb - sp.I * d) / 2], [0, 0, 0, 0], [0, 0, 0, 0],
                        [(bb + sp.I * d) / 2, 0, 0, (1 - a) / 2]])).is_zero_matrix)

# 8. Baseline law and Lyapunov identity.
h = [sp.expand((xbar.T * B[j] * x)[0]) for j in range(4)]
rho_bar = phi_p * phi_p.T
for j in range(4):
    gj_v2 = sp.expand(-sp.I * (H_J[j] * comm(rho, rho_bar)).trace())
    check(f"h_{j+1}(x)/4 = g_{j+1}(rho) from proposal v2", sp.expand(h[j] / 4 - gj_v2) == 0)
u_law = {u[j]: k * h[j] for j in range(4)}
Vdot = -sp.Rational(1, 4) * (xbar.T * f)[0]
check("E[dV]/dt = -(k/4) sum h_j^2 under u_j = k h_j",
      sp.expand(Vdot.subs(u_law) + k / 4 * sum(hj**2 for hj in h)) == 0)
check("F on E+ = (1 + b)/2", sp.simplify(at(F, xe) - (1 + bb) / 2) == 0)
cl_drift = f.subs(u_law)
check("closed loop: every point of E+ is an equilibrium (drift)", sp.expand(at(cl_drift, xe)).is_zero_matrix)
check("Tr(rho^2) = (1 + |x|^2)/4", sp.expand((rho * rho).trace() - (1 + (x.T * x)[0]) / 4) == 0)
check("h_j(xbar) = 0 for all j", all(at(hj, xbar) == 0 for hj in h))
check("h_j(xbad) = 0 for all j", all(at(hj, xbad) == 0 for hj in h))

# 8b. Baseline law: the Phi- population F- is a martingale, p1 is a submartingale.
Fm = sp.expand((rho * phi_m * phi_m.T).trace())
check("F- = (1 + xbad^T x)/4", sp.expand(Fm - (1 + (xbad.T * x)[0]) / 4) == 0)
check("H_1 + H_3 and H_2 - H_4 annihilate Phi-",
      ((H_J[0] + H_J[2]) * phi_m).is_zero_matrix and ((H_J[1] - H_J[3]) * phi_m).is_zero_matrix)
check("baseline law gives u1 = u3 and u2 = -u4", sp.expand(h[0] - h[2]) == 0 and sp.expand(h[1] + h[3]) == 0)
check("baseline: drift of F- = 0 for every x",
      sp.expand((xbad.T * f)[0].subs(u_law)) == 0)
check("diffusion of F = c (1 - p1) F", sp.simplify((xbar.T * g)[0] / 4 - c * (1 - p[0]) * F) == 0)
check("diffusion of F- = c (1 - p1) F-", sp.simplify((xbad.T * g)[0] / 4 - c * (1 - p[0]) * Fm) == 0)
check("baseline: drift of p1 = (k/2) sum h_j^2",
      sp.expand(f[0].subs(u_law) - k / 2 * sum(hj**2 for hj in h)) == 0)
check("F + F- = 1 on E+", sp.simplify(at(F + Fm, xe) - 1) == 0)
xem = sp.Matrix([-1, a, -a, bb, bb, d, -d] + [0] * NQ)
check("F = F- = 0 on E-", at(F, xem) == 0 and at(Fm, xem) == 0)
law_off = {u[j]: U0[j] + k * h[j] for j in range(4)}
check("offset law: drift of F- is not identically 0",
      sp.expand((xbad.T * f)[0].subs(law_off)) != 0)

# 9. Lie algebra dimensions in x-coordinates (Hamiltonian part, M = 0).
def lie_dim(gens):
    basis, vecs = [], []

    def add(m):
        v = sp.Matrix(list(m))
        trial = sp.Matrix.hstack(*vecs, v) if vecs else v
        if trial.rank() > len(vecs):
            basis.append(m)
            vecs.append(v)
            return True
        return False

    for gen in gens:
        add(gen)
    grew = True
    while grew:
        grew = False
        for m1, m2 in itertools.combinations(list(basis), 2):
            if add(sp.expand(comm(m1, m2))):
                grew = True
    return len(basis)


J0 = A0.subs({M: 0, omega: 1})
check("Lie dim {J0, B2, B4} = 6", lie_dim([J0, B[1], B[3]]) == 6)
check("Lie dim {J0, B1..B4} = 15", lie_dim([J0] + B) == 15)

# 10a. Offset law (Mirrahimi-van Handel 2007, Sec. 5 template): u = c + k h.
Pi_even = (sp.eye(4) + P("ZZ")) / 2
check("p1 = 1 on K forces q = 0: Pi_even P Pi_even = 0 for every q coordinate",
      all((Pi_even * P(l) * Pi_even).is_zero_matrix for l in Q_LAB))
V2 = 1 - F**2
for cname, cvec in [("X1-X2", U0), ("Y1+Y2", (0, 1, 0, 1))]:
    Hc = sum((cvec[j] * H_J[j] for j in range(4)), sp.zeros(4))
    check(f"offset {cname}: Hc Phi+ = 0", (Hc * phi_p).is_zero_matrix)
    check(f"offset {cname}: sum_j u0_j B_j xbar = 0",
          (sum((cvec[j] * B[j] for j in range(4)), sp.zeros(15)) * xbar).is_zero_matrix)
    check(f"offset {cname}: sum c_j h_j = 0", sp.expand(sum(cvec[j] * h[j] for j in range(4))) == 0)
    law_c = {u[j]: cvec[j] + k * h[j] for j in range(4)}
    dF_drift = sp.expand((xbar.T * f)[0] / 4).subs(law_c)
    dF_diff = sp.expand((xbar.T * g)[0] / 4)
    AV2 = sp.expand(-2 * F * dF_drift - dF_diff**2)
    check(f"offset {cname}: A(1-F^2) = -(k/2) F sum h^2 - 4 eta M (1-p1)^2 F^2",
          sp.simplify(AV2 - (-(k / 2) * F * sum(hj**2 for hj in h) - 4 * eta * M * (1 - p[0])**2 * F**2)) == 0)
    Gc = G.subs({u[j]: cvec[j] for j in range(4)})
    check(f"offset {cname}: on E+, G(c) p = 0 only at xbar (a=d=0, b=1)",
          sp.solve(list(Gc * xe[:NP, 0]), [a, bb, d], dict=True) == [{a: 0, bb: 1, d: 0}])

# 10b. Offset law u0 = (1, 0, -1, 0): the face of states supported on
# span{Phi-, Psi-} is invariant, and F = 0 on it.
psi_m = sp.Matrix([0, 1, -1, 0]) / sp.sqrt(2)
P2 = phi_m * phi_m.T + psi_m * psi_m.T
Q2 = sp.eye(4) - P2
Hc = sum((U0[j] * H_J[j] for j in range(4)), sp.zeros(4))
check("P2 commutes with ZZ and with sum_j u0_j H_j",
      comm(P2, P("ZZ")).is_zero_matrix and comm(P2, Hc).is_zero_matrix)
fx, fy, fz = sp.symbols("f_x f_y f_z", real=True)
rho_face = ((1 + fz) * phi_m * phi_m.T + (1 - fz) * psi_m * psi_m.T
            + (fx - sp.I * fy) * phi_m * psi_m.T + (fx + sp.I * fy) * psi_m * phi_m.T) / 2
on_face = dict(zip(list(x), list(coord(rho_face))))
check("face: F = 0", sp.expand(F.subs(on_face)) == 0)
check("face: h_j = 0 for all j", all(sp.expand(hj.subs(on_face)) == 0 for hj in h))
face_u = {u[j]: U0[j] for j in range(4)}
check("face: offset-law drift stays on the face",
      sp.expand(Q2 * drift_rho.subs(face_u).subs(on_face)).is_zero_matrix)
check("face: diffusion stays on the face",
      sp.simplify(Q2 * diff_rho.subs(on_face)).is_zero_matrix)

# 10. Write LaTeX in p, q names.
pn = sp.symbols("p1:8", real=True)
qn = sp.symbols("q1:9", real=True)
ren = dict(zip(list(x), list(pn) + list(qn)))
L_ = sp.latex
out = [
    "% generated by verification/check_control_model.py, do not edit",
    r"\newcommand{\GenG}{" + L_(sp.Matrix(G), mat_delim="(") + "}",
    r"\newcommand{\GenS}{" + L_(S, mat_delim="(") + "}",
    r"\newcommand{\GenR}{" + L_(R, mat_delim="(") + "}",
]
for j in range(4):
    out.append(r"\newcommand{\Genh" + "abcd"[j] + "}{" + L_(sp.factor(h[j].subs(ren))) + "}")
Path(__file__).resolve().parent.parent.joinpath("generated_matrices.tex").write_text("\n".join(out) + "\n")
print("wrote generated_matrices.tex")
