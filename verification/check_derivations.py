"""Symbolic verification of the derivations behind the Lie-algebra and closure claims.

Checks, exactly and symbolically:
  1. Fidelity identity        Tr(rho P_phi+) = Lambda/2 + Re rho_03
  2. Measurement+drift closure of (Lambda, rho_03): D[L], G[L], -i[H0, .]
     restricted to entries (0,0)+(3,3) and (0,3)
  3. Local drives break the closure (entry (0,3) of -i[Y x I, rho])
  4. Bracket closure of span{Y1, 1Y, ZZ, XZ, ZX, XX} and its dimension (6),
     YY orthogonal to the span, generated algebra of {Y1, 1Y, ZZ} = that span
  5. Universality with both quadratures: {X1, Y1, 1X, 1Y, ZZ} generates su(4)
  6. Half-parity operator commutes with ZZ

Run: python check_derivations.py
"""

import sympy as sp
from sympy import I, Matrix, Rational, eye, zeros, conjugate, symbols, simplify

# Pauli matrices and tensor products, basis order |00>, |01>, |10>, |11>
sx = Matrix([[0, 1], [1, 0]])
sy = Matrix([[0, -I], [I, 0]])
sz = Matrix([[1, 0], [0, -1]])
id2 = eye(2)


def kron(a, b):
    out = zeros(a.rows * b.rows, a.cols * b.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            out[i * b.rows:(i + 1) * b.rows, j * b.cols:(j + 1) * b.cols] = a[i, j] * b
    return out


Y1, _1Y = kron(sy, id2), kron(id2, sy)
X1, _1X = kron(sx, id2), kron(id2, sx)
ZZ, XZ, ZX, XX, YY = (kron(sz, sz), kron(sx, sz), kron(sz, sx),
                      kron(sx, sx), kron(sy, sy))

# Hermitian rho with symbolic entries, trace constraint applied via r2
r0, r1, r2, r3 = symbols("r0 r1 r2 r3", real=True)
re = {(i, j): symbols(f"a{i}{j}", real=True) for i in range(4) for j in range(i + 1, 4)}
im = {(i, j): symbols(f"b{i}{j}", real=True) for i in range(4) for j in range(i + 1, 4)}
rho = zeros(4, 4)
for i in range(4):
    rho[i, i] = [r0, r1, r2, r3][i]
for (i, j), a in re.items():
    rho[i, j] = a + I * im[(i, j)]
    rho[j, i] = a - I * im[(i, j)]
trace1 = {r2: 1 - r0 - r1 - r3}
Lam = r0 + r3


def is_zero(expr):
    return simplify(sp.expand(expr)) == 0


# 1. fidelity
Pphi = Rational(1, 2) * Matrix([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
assert is_zero((rho * Pphi).trace() - (Lam / 2 + re[(0, 3)])), "fidelity identity FAILED"
print("PASS  1. Tr(rho P_phi+) = Lambda/2 + Re rho_03")

# 2. measurement + drift closure; L = m ZZ with m = sqrt(M) > 0, H0 = w ZZ
m, w = symbols("m omega", positive=True)
L = m * ZZ
G = L * rho + rho * L - 2 * (L * rho).trace() * rho
D = L * rho * L - Rational(1, 2) * ((L * L) * rho + rho * (L * L))
drift = -I * (w * ZZ * rho - rho * (w * ZZ))

assert is_zero((G[0, 0] + G[3, 3] - 4 * m * Lam * (1 - Lam)).subs(trace1)), "G population FAILED"
assert is_zero((G[0, 3] - 4 * m * (1 - Lam) * rho[0, 3]).subs(trace1)), "G coherence FAILED"
for E, name in [(D, "D[L]"), (drift, "-i[H0,.]")]:
    for entry in [E[0, 0], E[3, 3], E[0, 3]]:
        assert is_zero(entry), f"{name} contributes to (Lambda, c): FAILED"
print("PASS  2. (Lambda, rho_03) closed under measurement and drift:")
print("         G_pop = 4 sqrt(M) Lam(1-Lam),  G_coh = 4 sqrt(M)(1-Lam) rho_03,")
print("         D[L] and -i[H0,.] contribute 0 to both")

# 2b. the drift does rotate cross-sector coherences (entry (0,1)): -2i w rho_01
assert is_zero(drift[0, 1] - (-2 * I * w * rho[0, 1])), "drift cross-sector FAILED"
assert is_zero(drift[1, 1]) and is_zero(drift[0, 3])
print("PASS  2b. drift rotates cross-sector coherences: (-i[H0,rho])_01 = -2i omega rho_01")

# 3. local drive breaks closure
ctrl = sp.expand(-I * (Y1 * rho - rho * Y1))
extra = ctrl[0, 3].free_symbols - {r0, r3, re[(0, 3)], im[(0, 3)]}
assert extra, "expected the local drive to couple outside (Lambda, c)"
print(f"PASS  3. -i[Y x 1, rho] entry (0,3) involves cross-sector terms: {sorted(map(str, extra))}")

# 4. bracket closure of the six-element span
basis = [("Y1", Y1), ("1Y", _1Y), ("ZZ", ZZ), ("XZ", XZ), ("ZX", ZX), ("XX", XX)]


def vec(Mx):
    return Matrix([f(Mx[i, j]) for i in range(4) for j in range(4)
                   for f in (sp.re, sp.im)])


# the real Lie algebra is the real span of the skew-Hermitian elements i*B_k
skew = [(name, I * B) for name, B in basis]
span = Matrix.hstack(*[vec(S) for _, S in skew])
assert span.rank() == 6, "six basis elements not independent: FAILED"

print("PASS  4. brackets of i*{Y1, 1Y, ZZ, XZ, ZX, XX} close on their real span (dim 6):")
for a in range(6):
    for b in range(a + 1, 6):
        br = skew[a][1] * skew[b][1] - skew[b][1] * skew[a][1]
        sol = span.solve_least_squares(vec(br))
        assert is_zero((span * sol - vec(br)).norm()), \
            f"[i{basis[a][0]},i{basis[b][0]}] leaves the span: FAILED"
        # [S_a, S_b] = gamma S_k  <=>  [B_a, B_b] = -gamma*i B_k
        terms = [f"{sp.nsimplify(-c)}i*{basis[k][0]}" for k, c in enumerate(sol) if c != 0]
        print(f"         [{basis[a][0]},{basis[b][0]}] = {' + '.join(terms) if terms else '0'}")

for name, B in basis:
    assert (YY * B).trace() == 0, f"YY not orthogonal to {name}: FAILED"
assert (YY * YY).trace() == 4
print("         YY orthogonal to all six, |YY|^2 = 4: YY is outside the span")


def lie_closure(gens, cap=40):
    mats = list(gens)
    spanm = Matrix.hstack(*[vec(M_) for M_ in mats])
    for _ in range(cap):
        added = False
        for A in list(mats):
            for B in list(mats):
                br = A * B - B * A
                v = vec(br)
                cand = Matrix.hstack(spanm, v)
                if cand.rank() > spanm.rank():
                    mats.append(br)
                    spanm = cand
                    added = True
        if not added:
            return spanm.rank()
    raise RuntimeError(f"Lie closure did not stabilize within {cap} sweeps")


dim_y_only = lie_closure([I * Y1, I * _1Y, I * ZZ])
assert dim_y_only == 6, f"y-only closure dim {dim_y_only} != 6: FAILED"
print("PASS  4b. algebra generated by {iY1, i1Y, iZZ} has dimension exactly 6")

# 5. both quadratures: full su(4)
dim_full = lie_closure([I * X1, I * Y1, I * _1X, I * _1Y, I * ZZ])
assert dim_full == 15, f"full closure dim {dim_full} != 15: FAILED"
print("PASS  5. algebra generated by {iX1, iY1, i1X, i1Y, iZZ} has dimension 15 = dim su(4)")

# 6b. structure of the 6-dim obstruction algebra: two commuting su(2) ideals
S = [I * B for _, B in basis]  # S0..S5 = i*{Y1, 1Y, ZZ, XZ, ZX, XX}
plus = [(S[0] + S[1]) / 2, (S[3] + S[4]) / 2, (S[2] - S[5]) / 2]
minus = [(S[0] - S[1]) / 2, (S[3] - S[4]) / 2, (S[2] + S[5]) / 2]


def comm(A, B):
    return A * B - B * A


for P in plus:
    for Q in minus:
        assert comm(P, Q) == zeros(4, 4), "ideals do not commute: FAILED"
# each triple closes as su(2): [f1,f2] = 2 f3 cyclically (Pauli normalization)
for tri, sgn in [(plus, 1), (minus, 1)]:
    f1, f2, f3 = tri
    assert comm(f1, f2) == 2 * f3 and comm(f2, f3) == 2 * f1 \
        and comm(f3, f1) == 2 * f2, "su(2) structure constants: FAILED"
print("PASS  6b. y-only algebra = su(2) (+) su(2), commuting ideals")
print("          I+ = span i{Y1+1Y, XZ+ZX, ZZ-XX}/2,  I- = span i{Y1-1Y, XZ-ZX, ZZ+XX}/2")

# invariant subspaces: Casimir of each ideal, exact eigenspaces
for label, tri in [("I+", plus), ("I-", minus)]:
    C = -(tri[0] ** 2 + tri[1] ** 2 + tri[2] ** 2)  # positive semidefinite
    eig = C.eigenvects()
    desc = ", ".join(f"eval {sp.nsimplify(v)} x{m}" for v, m, _ in eig)
    print(f"          Casimir({label}): {desc}")
    for v, m, vecs in eig:
        for vct in vecs:
            print(f"            {label} eval {sp.nsimplify(v)}: "
                  f"{[sp.nsimplify(x) for x in vct]}")

# 6c. consequence: block populations are feedback-invariant martingales.
# P1 projects onto span{Phi+, Psi-}; y-only drives and L = ZZ commute with
# it (block population drift is control-free), while the Liang-Amini-Mason
# two-body control Hamiltonian does not (it transfers between blocks).
phi_p = Matrix([1, 0, 0, 1]) / sp.sqrt(2)
psi_m = Matrix([0, 1, -1, 0]) / sp.sqrt(2)
P1 = phi_p * phi_p.T + psi_m * psi_m.T
assert P1 * P1 == P1 and P1.trace() == 2
for H, name in [(Y1, "Y1"), (_1Y, "1Y"), (ZZ, "ZZ (= L and H0)")]:
    assert comm(H, P1) == zeros(4, 4), f"[{name}, P1] != 0: FAILED"
H_lam = kron(sz, sy) - 3 * kron(id2, sy)
assert comm(H_lam, P1) != zeros(4, 4)
assert comm(X1, P1) != zeros(4, 4) and comm(_1X, P1) != zeros(4, 4)
print("PASS  6c. P_block commutes with Y1, 1Y, ZZ (y-only feedback cannot move")
print("          block population) but not with LAM's H1 or with X1, 1X")

# 6d. the full drift of B_t = Tr(P1 rho_t) vanishes under y-only feedback:
# unitary part (arbitrary u1, u2), Lindblad part, and Hamiltonian drift all
# contribute zero, so B_t is a bounded martingale for every feedback law.
u1s, u2s = symbols("u1 u2", real=True)
Hfb = w * ZZ + u1s * Y1 + u2s * _1Y
drift_B = (P1 * (-I * (Hfb * rho - rho * Hfb) + D)).trace()
assert is_zero(drift_B), "drift of block population nonzero: FAILED"
diff_B = (P1 * G).trace()
assert not is_zero(diff_B)  # diffusion remains: B_t is a true martingale
print("PASS  6d. d Tr(P1 rho) has zero drift for arbitrary feedback (u1, u2):")
print("          block population is a bounded martingale, E[B_inf] = B_0,")
print("          so P(reach Phi+) <= Tr(P1 rho_0) for any y-only feedback")

# 6. half-parity commutes with ZZ; ZZ acts as +/- identity on parity sectors
half = Rational(1, 2) * (kron(sz, id2) + kron(id2, sz))
assert (half * ZZ - ZZ * half) == zeros(4, 4)
assert ZZ[0, 0] == ZZ[3, 3] == 1 and ZZ[1, 1] == ZZ[2, 2] == -1
print("PASS  6. half-parity commutes with ZZ; ZZ = +1 on even sector, -1 on odd")

print("\nall checks passed")
