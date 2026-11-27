"""Microcode of the FPGA core against sim/fixed.py."""
import numpy as np
import pytest

from fpga import isa
from sim import coords as C
from sim.fixed import FixedFilter
from sim.params import Plant

from .helpers import random_states

PRM = Plant()


def _random_inputs(flt, n, seed):
    rng = np.random.default_rng(seed)
    rho = random_states(n, seed=seed)
    R, I = flt.encode(rho)
    codes = rng.integers(-2 ** (flt.wc - 1), 2 ** (flt.wc - 1), size=(n, 4))
    dy = PRM.c * C.p1(rho) * flt.Ts + rng.normal(0.0, np.sqrt(flt.Ts), n)
    return R, I, codes, flt.adc(dy)


@pytest.mark.parametrize("w, Ts", [(12, 0.01), (16, 0.01), (16, 0.05), (24, 0.001), (24, 0.01)])
def test_one_step_and_law_equal_fixed_filter(w, Ts):
    flt = FixedFilter(w, Ts, PRM, 2.0)
    p = isa.compile_program(flt)
    R, I, codes, dyw = _random_inputs(flt, 64, seed=w)
    R1, I1 = flt.step(R, I, codes, dyw)
    n1 = flt.codes(R1, I1)
    for k in range(64):
        regs = isa.load_state(p, R[k], I[k], codes[k])
        regs[p.reg("dy")] = int(dyw[k])
        isa.run(p, regs)
        Rg, Ig, ng = isa.read_state(p, regs)
        assert np.array_equal(Rg, R1[k]) and np.array_equal(Ig, I1[k]), k
        assert np.array_equal(ng, n1[k]), k


def test_codes_entry_equals_law():
    flt = FixedFilter(16, 0.01, PRM, 2.0)
    p = isa.compile_program(flt)
    R, I, _, _ = _random_inputs(flt, 32, seed=3)
    n = flt.codes(R, I)
    for k in range(32):
        regs = isa.load_state(p, R[k], I[k])
        isa.run(p, regs, entry="codes")
        assert np.array_equal(isa.read_state(p, regs)[2], n[k])


def test_closed_loop_trajectory_word_for_word():
    """500 steps of the closed loop: microcode and FixedFilter see the same
    outputs and produce the same state words and codes at every step."""
    from sim.plant import plant_step
    w, Ts, dt = 16, 0.01, 1e-3
    flt = FixedFilter(w, Ts, PRM, 2.0)
    p = isa.compile_program(flt)
    rng = np.random.default_rng(9)
    N = 4
    rho = np.zeros((N, 4, 4), dtype=complex)
    rho[:, 0, 0] = 1
    R, I = flt.encode(rho)
    regs = [isa.load_state(p, R[k], I[k]) for k in range(N)]
    for r in regs:
        isa.run(p, r, entry="codes")
    n = flt.codes(R, I)
    track = {}
    for _ in range(500):
        assert all(np.array_equal(isa.read_state(p, regs[k])[2], n[k]) for k in range(N))
        dy_acc = np.zeros(N)
        for _ in range(round(Ts / dt)):
            rho, dy = plant_step(rho, flt.command(n), rng.normal(0, np.sqrt(dt), N), dt, PRM)
            dy_acc += dy
        dyw = flt.adc(dy_acc)
        R, I = flt.step(R, I, n, dyw)
        n = flt.codes(R, I)
        for k in range(N):
            regs[k][p.reg("dy")] = int(dyw[k])
            isa.run(p, regs[k], track=track)
            Rg, Ig, _ = isa.read_state(p, regs[k])
            assert np.array_equal(Rg, R[k]) and np.array_equal(Ig, I[k])


def test_schedule_keeps_results_and_spacing():
    flt = FixedFilter(16, 0.01, PRM, 2.0)
    p = isa.compile_program(flt)
    rom = isa.schedule(p)
    last = {}
    for pos, (op, x, y, z) in enumerate(rom):
        reads = [x, y] if op == isa.OPS["MAC"] else [x] if op == isa.OPS["ADD"] else [y] if op == isa.OPS["DIV"] else []
        if pos == p.codes_pc_scheduled:
            last = {}
        for r in reads:
            assert pos - last.get(r, -10) >= isa.PIPE_GAP
        if op in (isa.OPS["ST"], isa.OPS["DIV"]):
            last[x] = pos
    # Running the scheduled list gives the same result (NOPs do nothing).
    q = isa.Program(w=p.w, wc=p.wc, F=p.F, instrs=rom, regs=p.regs, codes_pc=p.codes_pc_scheduled,
                    consts=p.consts)
    R, I, codes, dyw = _random_inputs(flt, 8, seed=5)
    for k in range(8):
        a = isa.load_state(p, R[k], I[k], codes[k])
        a[p.reg("dy")] = int(dyw[k])
        b = dict(a)
        isa.run(p, a)
        isa.run(q, b)
        assert a == b


@pytest.mark.parametrize("w", [12, 16, 24])
def test_register_and_accumulator_widths(w):
    """With every input word drawn over its full range (saturated states,
    codes, and outputs), no stored value needs more than w + 10 bits and the
    accumulator no more than 2 (w + 10) + 6 bits. The RTL uses these widths."""
    for Ts in (0.001, 0.01, 0.05):
        flt = FixedFilter(w, Ts, PRM, 2.0)
        p = isa.compile_program(flt)
        rng = np.random.default_rng(w)
        track = {}
        lim = 2 ** (w - 1)
        for _ in range(300):
            R = rng.integers(-lim, lim, size=(4, 4))
            I = rng.integers(-lim, lim, size=(4, 4))
            regs = isa.load_state(p, R, I, rng.integers(-2 ** (flt.wc - 1), 2 ** (flt.wc - 1), size=4))
            regs[p.reg("dy")] = int(rng.integers(-lim, lim))
            isa.run(p, regs, track=track)
        assert track["reg"] <= w + 10, (Ts, track)
        assert track["acc"] <= 2 * (w + 10) + 6, (Ts, track)


def test_small_trace_uses_the_divider_floor():
    """A state with trace far below 1 (all words near 0) reaches the floor
    t = 2^(F-2) of the normalization. The microcode matches FixedFilter there too."""
    flt = FixedFilter(16, 0.01, PRM, 2.0)
    p = isa.compile_program(flt)
    rng = np.random.default_rng(12)
    for _ in range(20):
        R = rng.integers(-40, 40, size=(1, 4, 4))
        R = (R + R.transpose(0, 2, 1)) // 2
        I = np.zeros_like(R)
        codes = rng.integers(-2048, 2048, size=(1, 4))
        dyw = rng.integers(-200, 200, size=1)
        R1, I1 = flt.step(R, I, codes, dyw)
        regs = isa.load_state(p, R[0], I[0], codes[0])
        regs[p.reg("dy")] = int(dyw[0])
        isa.run(p, regs)
        Rg, Ig, _ = isa.read_state(p, regs)
        assert np.array_equal(Rg, R1[0]) and np.array_equal(Ig, I1[0])


def test_temporaries_are_written_before_read_in_every_run():
    """The temporaries (32 to 127) are shared by all contexts. That is safe only
    if each run (STEP from address 0, CODES from its entry) writes a temporary
    before it reads it."""
    flt = FixedFilter(16, 0.01, PRM, 2.0)
    p = isa.compile_program(flt)
    for start in (0, p.codes_pc):
        written = set()
        for op, x, y, z in p.instrs[start:]:
            if op == isa.OPS["END"]:
                break
            reads = [x, y] if op == isa.OPS["MAC"] else [x] if op == isa.OPS["ADD"] else \
                [y] if op == isa.OPS["DIV"] else []
            for r in reads:
                if isa.TEMP_BASE <= r < isa.CONST_BASE:
                    assert r in written, (start, r)
            if op in (isa.OPS["ST"], isa.OPS["DIV"]):
                written.add(x)


def test_saturated_state_equals_fixed_filter():
    """A diverged w = 12 state whose upper imaginary words sit at -2^(w-1).
    The core stores only the upper triangle, so the lower entry is the exact
    negative 2^(w-1). FixedFilter must use the same value, not re-saturate it
    to 2^(w-1) - 1 (board diagnostic of 2026-11-13, context 12, step 453)."""
    flt = FixedFilter(12, 0.05, PRM, 2.0)
    p = isa.compile_program(flt)
    lo, hi = -(1 << 11), (1 << 11) - 1
    R = np.diag([hi, lo, lo, hi]).astype(np.int64)
    I = np.zeros((4, 4), dtype=np.int64)
    I[0, 1], I[0, 2], I[0, 3], I[1, 3], I[2, 3] = -1704, 1704, -1900, lo, hi
    I = I - I.T
    codes = np.array([[100, -200, 300, -400]])
    for dyw in (-300, 0, 300):
        R1, I1 = flt.step(R[None], I[None], codes, np.array([dyw]))
        regs = isa.load_state(p, R, I, codes[0])
        regs[p.reg("dy")] = dyw
        isa.run(p, regs)
        Rg, Ig, ng = isa.read_state(p, regs)
        assert np.array_equal(Rg, R1[0]) and np.array_equal(Ig, I1[0]), dyw
        assert np.array_equal(ng, flt.codes(R1, I1)[0]), dyw
