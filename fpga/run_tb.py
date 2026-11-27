"""Bit-exact check of the RTL core against the microcode interpreter.

    python -m fpga.run_tb --w 16 --Ts 0.01 --steps 2000 [--ctx 6] [--keep DIR]

Writes the ROM (fpga.isa.schedule) and a stimulus file, runs fpga/tb/tb_core.v
in iverilog, and compares every state word and code after every run with
fpga.isa.run on the same inputs. Contexts:
    0 .. n-3   closed loop: a float plant from |00> or a random state drives
               the outputs dy; the codes come from the interpreter
    n-2, n-1   stress: random initial words and random dy over the full
               w-bit range at every step
Exit status 0 if every word matches and the overflow flag stays 0.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from fpga import isa
from sim.fixed import FixedFilter
from sim.params import Plant
from sim.plant import plant_step

ROOT = Path(__file__).resolve().parent
PCB, CTXB, WC = 10, 8, 12
OSS = Path.home() / ".local/oss-cad-suite/bin"


def widths(w):
    rw = w + 10
    return dict(W=w, WC=WC, F=w - 2, RW=rw, AW=2 * rw + 6, CTXB=CTXB, PCB=PCB)


def write_rom(p, path):
    rom = isa.schedule(p)
    if len(rom) > 2**PCB:
        raise RuntimeError("program does not fit the ROM")
    end = (isa.OPS["END"], 0, 0, 0)
    words = [isa.encode_word(x) for x in rom] + [isa.encode_word(end)] * (2**PCB - len(rom))
    path.write_text("\n".join(f"{x:010x}" for x in words) + "\n")
    return p.codes_pc_scheduled


def random_rho(rng):
    G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    r = G @ G.conj().T
    return r / np.trace(r).real


def build_case(w, Ts, steps, n_ctx, seed):
    prm = Plant()
    flt = FixedFilter(w, Ts, prm, 2.0)
    p = isa.compile_program(flt)
    rng = np.random.default_rng(seed)
    dt = 1e-3
    lim = 2 ** (w - 1)
    init, regs, rho = [], [], []
    for c in range(n_ctx):
        if c >= n_ctx - 2:
            R = rng.integers(-lim, lim, size=(4, 4))
            I = rng.integers(-lim, lim, size=(4, 4))
            r0 = None
        else:
            r0 = np.diag([1, 0, 0, 0]).astype(complex) if c == 0 else random_rho(rng)
            R, I = flt.encode(r0[None])
            R, I = R[0], I[0]
        rg = isa.load_state(p, R, I)
        regs.append(rg)
        rho.append(r0)
        for a, v in rg.items():
            if a < isa.CONST_BASE:
                init.append((c, a, v))
    init += [(0, a, v) for a, v in p.consts.items()]
    lines, expect = [], []
    # Run 0: the CODES entry gives the first commands.
    for c in range(n_ctx):
        isa.run(p, regs[c], entry="codes")
        lines.append((1, 0))
        expect.append(dict(regs[c]))
    for s in range(steps):
        for c in range(n_ctx):
            if rho[c] is None:
                dyw = int(rng.integers(-lim, lim))
            else:
                n = np.array([regs[c][p.reg(("n", j))] for j in range(4)])
                r = rho[c][None]
                acc = 0.0
                for _ in range(round(Ts / dt)):
                    r, dy = plant_step(r, flt.command(n)[None], rng.normal(0, np.sqrt(dt), 1), dt, prm)
                    acc += dy[0]
                rho[c] = r[0]
                dyw = int(flt.adc(np.array([acc]))[0])
            regs[c][p.reg("dy")] = dyw
            isa.run(p, regs[c])
            lines.append((0, dyw))
            expect.append(dict(regs[c]))
    return p, init, lines, expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--Ts", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep", default=None)
    ap.add_argument("--cediv", type=int, default=1, help="core clock enable: one clock in CEDIV")
    args = ap.parse_args()
    p, init, lines, expect = build_case(args.w, args.Ts, args.steps, args.ctx, args.seed)
    wd = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="tb_core_"))
    wd.mkdir(parents=True, exist_ok=True)
    codes_pc = write_rom(p, wd / "rom.hex")
    with open(wd / "stim.txt", "w") as f:
        f.write(f"{args.ctx} {args.steps + 1} {len(init)} {codes_pc}\n")
        for c, a, v in init:
            f.write(f"{c} {a} {v}\n")
        for e, d in lines:
            f.write(f"{e} {d}\n")
    prm = widths(args.w)
    defs = [f"-Ptb_core.{k}={v}" for k, v in prm.items()] + [f"-Ptb_core.CEDIV={args.cediv}"]
    env = dict(os.environ, PATH=f"{OSS}:{os.environ['PATH']}")
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", *defs, str(ROOT / "tb/tb_core.v"),
                    str(ROOT / "rtl/core.v")], cwd=wd, check=True, env=env)
    run = subprocess.run(["vvp", "-n", "sim.vvp"], cwd=wd, capture_output=True, text=True, env=env)
    print(run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-2000:])
    got = [list(map(int, ln.split())) for ln in (wd / "out.txt").read_text().splitlines()]
    order = [p.reg(("R", i, j)) for i in range(4) for j in range(i, 4)]  # not used: tb reads 0..15
    del order
    bad, cycles = 0, []
    if len(got) != len(expect):
        print(f"FAIL: {len(got)} lines from the testbench, {len(expect)} expected")
        return 1
    for row, exp in zip(got, expect):
        vals = row[2:18] + row[18:22]
        want = [exp[a] for a in range(16)] + [exp[a] for a in range(17, 21)]
        if vals != want or row[22] != 0:
            bad += 1
            if bad <= 3:
                print(f"MISMATCH ctx {row[0]} run {row[1]}: got {vals} overflow {row[22]}\n   want {want}")
        cycles.append(row[23])
    n_words = len(got) * 20
    print(f"w = {args.w}, Ts = {args.Ts}: {len(got)} runs, {n_words} words compared, {bad} runs differ; "
          f"cycles per STEP run: {int(np.median(cycles[args.ctx:]))}")
    if not args.keep:
        shutil.rmtree(wd)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
