"""Microcode for the fixed-point filter and offset law.

The FPGA core runs a straight-line program of five operations on a register
file and one wide accumulator ACC:

    CLR              ACC = 0
    MAC  a b neg     ACC += (-1)^neg R[a] R[b]
    ADD  a sh neg    ACC += (-1)^neg (R[a] << sh)
    ST   d n sat     R[d] = sat(rnd(ACC, n))     sat: 0 none, 1 to w bits, 2 to wc bits
    DIV  d a         t = max(R[a], 2^(F-2)),  R[d] = (2^(2F) + floor(t / 2)) div t
    NOP              (pipeline spacing, no effect)

rnd(x, n) is x / 2^n rounded to nearest, ties to even (sim.fixed.rnd).

compile_program(flt) writes the step and the law of sim.fixed.FixedFilter as
such a program, line for line. run(program, regs, ...) executes it with
Python integers. tests/test_isa.py checks that run() equals FixedFilter.step
and FixedFilter.codes word for word, so the program is a second, independent
statement of the same arithmetic. The RTL (fpga/rtl/core.v) executes the
same program, and fpga/run_tb.py checks the RTL against run().

Register map:
    0 to 31     per-context registers (one context is one trajectory): state
                (upper triangle, 16 words), input dy (16), codes (17 to 20)
    32 to 127   temporaries, shared by all contexts. A run writes every
                temporary before it reads it, so no value lives across runs.
    128 to 143  constants shared by all contexts
The program has two entry points: STEP (address 0) runs the filter step and
then the law; CODES runs the law only. Both end with END.
"""
from dataclasses import dataclass, field

import numpy as np

from sim.fixed import GUARD, ZZ_DIAG, ZZ_SIGN, FixedFilter

OPS = {"NOP": 0, "CLR": 1, "MAC": 2, "ADD": 3, "ST": 4, "DIV": 5, "END": 7}
N_CTX_REGS = 32
TEMP_BASE = 32
CONST_BASE = 128
SAT_NONE, SAT_W, SAT_WC = 0, 1, 2
CONST_NAMES = ("C_b", "C_e", "C_a0", "C_dt", "C_w", "C_g", "C_u", "C_k", "U0_0", "U0_1", "U0_2", "U0_3")
PIPE_GAP = 3  # an instruction that reads R[x] issues at least this many slots after the write of R[x]


def rnd_int(x: int, n: int) -> int:
    if n <= 0:
        return x << (-n)
    q = x >> n
    rem = x & ((1 << n) - 1)
    half = 1 << (n - 1)
    return q + (rem > half) + ((rem == half) and (q & 1))


def sat_int(x: int, w: int) -> int:
    lim = 1 << (w - 1)
    return min(max(x, -lim), lim - 1)


@dataclass
class Program:
    w: int
    wc: int
    F: int
    instrs: list = field(default_factory=list)  # tuples (op, x, y, z)
    regs: dict = field(default_factory=dict)  # name -> register index
    codes_pc: int = 0
    consts: dict = field(default_factory=dict)  # register index -> value

    def reg(self, name):
        return self.regs[name]


class _Builder:
    def __init__(self, flt: FixedFilter):
        self.flt = flt
        self.p = Program(w=flt.w, wc=flt.wc, F=flt.F)
        self.next_reg = 0
        self.free = []
        for i, name in enumerate(CONST_NAMES):
            self.p.regs[name] = CONST_BASE + i
        f = flt
        vals = [f.C_b, f.C_e, f.C_a0, f.C_dt, f.C_w, f.C_g, f.C_u, f.C_k] + [int(v) for v in f.u0]
        self.p.consts = {CONST_BASE + i: int(v) for i, v in enumerate(vals)}
        # Fixed per-context registers: state (upper triangle), dy, codes.
        for i in range(4):
            for j in range(i, 4):
                self.fixed(("R", i, j))
            for j in range(i + 1, 4):
                self.fixed(("I", i, j))
        self.fixed("dy")
        for j in range(4):
            self.fixed(("n", j))
        if self.next_reg > N_CTX_REGS:
            raise RuntimeError("per-context registers do not fit")
        self.next_reg = TEMP_BASE

    def fixed(self, name):
        self.p.regs[name] = self.next_reg
        self.next_reg += 1

    def temp(self):
        if self.free:
            return self.free.pop()
        r = self.next_reg
        self.next_reg += 1
        if r >= CONST_BASE:
            raise RuntimeError("out of temporary registers")
        return r

    def release(self, *regs):
        for r in regs:
            if r is not None and r < CONST_BASE and r not in self.p.regs.values():
                self.free.append(r)

    def emit(self, op, x=0, y=0, z=0):
        self.p.instrs.append((OPS[op], x, y, z))

    # Access to the full state matrices through the stored upper triangle.
    def R(self, i, j):
        return (self.p.reg(("R", min(i, j), max(i, j))), 1)

    def I(self, i, j):  # noqa: E743
        if i == j:
            return None
        return (self.p.reg(("I", min(i, j), max(i, j))), 1 if i < j else -1)

    def store_sum(self, terms, n, sat, dest=None):
        """ACC = sum of terms; dest = sat(rnd(ACC, n)). A term is
        ("mac", ra, rb, sign) or ("add", r, sh, sign). Returns dest, or None
        if there are no terms (the value is exactly 0)."""
        if not terms:
            return None
        d = self.temp() if dest is None else dest
        self.emit("CLR")
        for t in terms:
            if t[0] == "mac":
                _, a, b, sg = t
                self.emit("MAC", a, b, int(sg < 0))
            else:
                _, a, sh, sg = t
                self.emit("ADD", a, sh, int(sg < 0))
        self.emit("ST", d, n, sat)
        return d


def compile_program(flt: FixedFilter) -> Program:
    """The microcode of FixedFilter.step followed by FixedFilter.codes."""
    b = _Builder(flt)
    p = b.p
    F, FD, s = flt.F, flt.FD, flt.s
    c = p.reg
    dy = c("dy")

    # ---- delta = K - 1 (FixedFilter.delta) ----
    dy2 = b.store_sum([("mac", dy, dy, 1)], FD, SAT_NONE)
    beta = b.store_sum([("mac", c("C_b"), dy, 1)], F, SAT_NONE)
    tmp = b.store_sum([("add", dy2, 0, 1), ("add", c("C_dt"), 0, -1)], 0, SAT_NONE)
    t2 = b.store_sum([("mac", c("C_e"), tmp, 1)], F, SAT_NONE)
    alpha = b.store_sum([("add", t2, 0, 1), ("add", c("C_a0"), 0, -1)], 0, SAT_NONE)
    b.release(dy2, tmp, t2)
    ud = [b.store_sum([("mac", c(("n", j)), c("C_u"), 1)], GUARD, SAT_NONE) for j in range(4)]
    hj_r, hj_i = flt.hj_r, flt.hj_i
    d_r = [[None] * 4 for _ in range(4)]
    d_i = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            tr = [("add", ud[q], 0, int(hj_i[q, i, j])) for q in range(4) if hj_i[q, i, j] != 0]
            ti = [("add", ud[q], 0, -int(hj_r[q, i, j])) for q in range(4) if hj_r[q, i, j] != 0]
            if i == j:
                tr += [("add", alpha, 0, 1), ("add", beta, 0, int(ZZ_DIAG[i]))]
                ti += [("add", c("C_w"), 0, -int(ZZ_DIAG[i]))]
            d_r[i][j] = b.store_sum(tr, 0, SAT_W)
            d_i[i][j] = b.store_sum(ti, 0, SAT_W)
    b.release(alpha, beta, *ud)

    # ---- A = delta rho, rounded by F ----
    a_r = [[None] * 4 for _ in range(4)]
    a_i = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            tr, ti = [], []
            for k in range(4):
                Rkj, Ikj = b.R(k, j), b.I(k, j)
                if d_r[i][k] is not None:
                    tr.append(("mac", d_r[i][k], Rkj[0], Rkj[1]))
                    if Ikj is not None:
                        ti.append(("mac", d_r[i][k], Ikj[0], Ikj[1]))
                if d_i[i][k] is not None:
                    if Ikj is not None:
                        tr.append(("mac", d_i[i][k], Ikj[0], -Ikj[1]))
                    ti.append(("mac", d_i[i][k], Rkj[0], Rkj[1]))
            a_r[i][j] = b.store_sum(tr, F, SAT_NONE)
            a_i[i][j] = b.store_sum(ti, F, SAT_NONE)

    # ---- B = A delta^dag, rounded by FD; only the entries the update uses ----
    b_r, b_i = {}, {}
    for i in range(4):
        for j in range(i, 4):
            tr, ti = [], []
            for k in range(4):
                # b_r = sum_k a_r[i,k] d_r[j,k] + a_i[i,k] d_i[j,k]
                # b_i = sum_k a_i[i,k] d_r[j,k] - a_r[i,k] d_i[j,k]
                if a_r[i][k] is not None and d_r[j][k] is not None:
                    tr.append(("mac", a_r[i][k], d_r[j][k], 1))
                if a_i[i][k] is not None and d_i[j][k] is not None:
                    tr.append(("mac", a_i[i][k], d_i[j][k], 1))
                if j > i:
                    if a_i[i][k] is not None and d_r[j][k] is not None:
                        ti.append(("mac", a_i[i][k], d_r[j][k], 1))
                    if a_r[i][k] is not None and d_i[j][k] is not None:
                        ti.append(("mac", a_r[i][k], d_i[j][k], -1))
            b_r[i, j] = b.store_sum(tr, FD, SAT_NONE)
            if j > i:
                b_i[i, j] = b.store_sum(ti, FD, SAT_NONE)
    b.release(*[x for row in d_r for x in row], *[x for row in d_i for x in row])

    # ---- g = (1 - eta) M Ts ZZ rho ZZ, rounded by F ----
    g_r, g_i = {}, {}
    for i in range(4):
        for j in range(i, 4):
            g_r[i, j] = b.store_sum([("mac", c("C_g"), b.R(i, j)[0], int(ZZ_SIGN[i, j]))], F, SAT_NONE)
            if j > i:
                g_i[i, j] = b.store_sum([("mac", c("C_g"), b.I(i, j)[0], int(ZZ_SIGN[i, j]))], F, SAT_NONE)

    # ---- S, then rho + rnd(S, s) ----
    R1, I1 = {}, {}
    for i in range(4):
        for j in range(i, 4):
            terms = [("add", x, 0, sg) for x, sg in ((a_r[i][j], 1), (a_r[j][i], 1), (b_r[i, j], 1), (g_r[i, j], 1))
                     if x is not None]
            sr = b.store_sum(terms, s, SAT_NONE)
            R1[i, j] = b.store_sum([("add", b.R(i, j)[0], 0, 1)] + ([("add", sr, 0, 1)] if sr is not None else []),
                                   0, SAT_NONE)
            b.release(sr)
            if j > i:
                terms = [("add", x, 0, sg) for x, sg in ((a_i[i][j], 1), (a_i[j][i], -1), (b_i[i, j], 1), (g_i[i, j], 1))
                         if x is not None]
                si = b.store_sum(terms, s, SAT_NONE)
                I1[i, j] = b.store_sum([("add", b.I(i, j)[0], 0, 1)] + ([("add", si, 0, 1)] if si is not None else []),
                                       0, SAT_NONE)
                b.release(si)
    b.release(*[x for row in a_r for x in row], *[x for row in a_i for x in row], *b_r.values(), *b_i.values(),
              *g_r.values(), *g_i.values())

    # ---- normalization: one division, one multiply per entry, saturate to w ----
    tr_ = b.store_sum([("add", R1[i, i], 0, 1) for i in range(4)], 0, SAT_NONE)
    r = b.temp()
    b.emit("DIV", r, tr_, 0)
    for i in range(4):
        for j in range(i, 4):
            b.store_sum([("mac", R1[i, j], r, 1)], F, SAT_W, dest=c(("R", i, j)))
            if j > i:
                b.store_sum([("mac", I1[i, j], r, 1)], F, SAT_W, dest=c(("I", i, j)))
    b.release(tr_, r, *R1.values(), *I1.values())

    # ---- law: n = sat(rnd((u0 << (F + G)) + C_k h, F + G + du_exp), wc) ----
    p.codes_pc = len(p.instrs)
    coef = np.zeros((4, 2, 4, 4), dtype=np.int64)  # h_j coefficient on R[i, j] (0) and I[i, j] (1), i <= j
    for jj in range(4):
        for a in range(15):
            ha = int(flt.h_rows[jj, a])
            if ha == 0:
                continue
            for i in range(4):
                for j in range(4):
                    # x_a = sum_ij basis_r[a,i,j] R[j,i] - basis_i[a,i,j] I[j,i]
                    br, bi = int(flt.basis_r[a, i, j]), int(flt.basis_i[a, i, j])
                    if br:
                        coef[jj, 0, min(i, j), max(i, j)] += ha * br
                    if bi and i != j:
                        # I[j, i] = I[min, max] if j < i else -I[min, max]
                        coef[jj, 1, min(i, j), max(i, j)] -= ha * bi * (1 if j < i else -1)
    h = []
    for jj in range(4):
        terms = []
        for kind in (0, 1):
            for i in range(4):
                for j in range(4):
                    v = int(coef[jj, kind, i, j])
                    if v == 0:
                        continue
                    reg = c(("R", i, j)) if kind == 0 else c(("I", i, j))
                    sg, m = (1 if v > 0 else -1), abs(v)
                    for bit in range(m.bit_length()):  # m = sum of powers of two
                        if (m >> bit) & 1:
                            terms.append(("add", reg, bit, sg))
        h.append(b.store_sum(terms, 0, SAT_NONE))
    shift = F + GUARD + flt.du_exp
    for jj in range(4):
        terms = [("add", c(f"U0_{jj}"), F + GUARD, 1)]
        if h[jj] is not None:
            terms.append(("mac", c("C_k"), h[jj], 1))
        b.store_sum(terms, shift, SAT_WC, dest=c(("n", jj)))
    b.emit("END")
    p.n_regs = b.next_reg
    return p


def run(p: Program, regs: dict, entry: str = "step", track: dict | None = None) -> None:
    """Execute p on regs (index -> int, per-context and constants), in place.
    track, if given, collects the largest bit length of any stored value and of ACC."""
    acc = 0
    pc = 0 if entry == "step" else p.codes_pc
    lim_w, lim_wc = p.w, p.wc
    for op, x, y, z in p.instrs[pc:]:
        if op == OPS["END"]:
            return
        if op == OPS["NOP"]:
            continue
        if op == OPS["CLR"]:
            acc = 0
        elif op == OPS["MAC"]:
            v = regs[x] * regs[y]
            acc += -v if z else v
        elif op == OPS["ADD"]:
            v = regs[x] << y
            acc += -v if z else v
        elif op == OPS["ST"]:
            v = rnd_int(acc, y)
            if z == SAT_W:
                v = sat_int(v, lim_w)
            elif z == SAT_WC:
                v = sat_int(v, lim_wc)
            regs[x] = v
            if track is not None:
                track["reg"] = max(track.get("reg", 0), v.bit_length() + 1)
        elif op == OPS["DIV"]:
            t = max(regs[y], 1 << (p.F - 2))
            regs[x] = ((1 << (2 * p.F)) + t // 2) // t
        else:
            raise ValueError(f"bad op {op}")
        if track is not None:
            track["acc"] = max(track.get("acc", 0), acc.bit_length() + 1)
    raise RuntimeError("program has no END")


def load_state(p: Program, R: np.ndarray, I: np.ndarray, n: np.ndarray | None = None) -> dict:
    """Register dict for one trajectory: constants, state words (upper triangle), codes."""
    regs = dict(p.consts)
    for i in range(4):
        for j in range(i, 4):
            regs[p.reg(("R", i, j))] = int(R[i, j])
            if j > i:
                regs[p.reg(("I", i, j))] = int(I[i, j])
    for j in range(4):
        regs[p.reg(("n", j))] = 0 if n is None else int(n[j])
    regs[p.reg("dy")] = 0
    return regs


def read_state(p: Program, regs: dict):
    R = np.zeros((4, 4), dtype=np.int64)
    I = np.zeros((4, 4), dtype=np.int64)
    for i in range(4):
        for j in range(i, 4):
            R[i, j] = R[j, i] = regs[p.reg(("R", i, j))]
            if j > i:
                I[i, j] = regs[p.reg(("I", i, j))]
                I[j, i] = -I[i, j]
    n = np.array([regs[p.reg(("n", j))] for j in range(4)], dtype=np.int64)
    return R, I, n


def schedule(p: Program) -> list:
    """Insert NOPs so that every read of R[x] issues at least PIPE_GAP slots after
    the instruction that writes R[x] (ST or DIV). Returns the instruction list
    for the ROM. The entry address of CODES moves too: see codes_pc_scheduled."""
    out, last_write = [], {}
    p.codes_pc_scheduled = None
    for idx, ins in enumerate(p.instrs):
        if idx == p.codes_pc:
            p.codes_pc_scheduled = len(out)
            last_write = {}  # the CODES entry starts after a frame boundary
        op, x, y, z = ins
        reads = []
        if op == OPS["MAC"]:
            reads = [x, y]
        elif op in (OPS["ADD"], OPS["DIV"]):
            reads = [x] if op == OPS["ADD"] else [y]
        need = max((last_write.get(r, -10) + PIPE_GAP for r in reads), default=0)
        while len(out) < need:
            out.append((OPS["NOP"], 0, 0, 0))
        out.append(ins)
        if op in (OPS["ST"], OPS["DIV"]):
            last_write[x] = len(out) - 1
    return out


def encode_word(ins) -> int:
    """40-bit ROM word: op[39:36] x[35:28] y[27:20] z[19:18]; ADD/ST shift in y."""
    op, x, y, z = ins
    if not (0 <= x < 256 and 0 <= y < 256 and 0 <= z < 4):
        raise ValueError(f"field out of range: {ins}")
    return (op << 36) | (x << 28) | (y << 20) | (z << 18)
