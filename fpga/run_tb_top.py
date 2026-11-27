"""UART-level check of fpga/rtl/top.v.

    python -m fpga.run_tb_top [--w 16] [--Ts 0.01] [--steps 4] [--ctx 3]

Sends constants and states over the simulated UART, then CODES and STEP
commands for several contexts, one read-back of every state register, and
compares every reply byte with fpga.isa.run. Exit 0 on a full match.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from fpga import isa, protocol
from fpga.run_tb import OSS, ROOT, widths, write_rom
from sim.fixed import FixedFilter
from sim.params import Plant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--Ts", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=3)
    args = ap.parse_args()
    flt = FixedFilter(args.w, args.Ts, Plant(), 2.0)
    p = isa.compile_program(flt)
    rng = np.random.default_rng(3)
    lim = 2 ** (args.w - 1)
    regs = {}
    for c in range(args.ctx):
        R = rng.integers(-lim // 2, lim // 2, size=(4, 4))
        regs[c] = isa.load_state(p, (R + R.T) // 2, rng.integers(-lim // 2, lim // 2, size=(4, 4)))
    host, n_reply = protocol.init_bytes(p, regs)
    expect = bytes([0x5A]) * n_reply
    for c in range(args.ctx):
        host += protocol.codes(c)
        isa.run(p, regs[c], entry="codes")
        expect += b"".join(int(regs[c][p.reg(("n", j))]).to_bytes(2, "big", signed=True) for j in range(4))
    for _ in range(args.steps):
        for c in range(args.ctx):
            dy = int(rng.integers(-lim, lim))
            host += protocol.step(c, dy)
            regs[c][p.reg("dy")] = dy
            isa.run(p, regs[c])
            expect += b"".join(int(regs[c][p.reg(("n", j))]).to_bytes(2, "big", signed=True) for j in range(4))
    for c in range(args.ctx):
        for a in range(16):
            host += protocol.read(c, a)
            expect += int(regs[c][a]).to_bytes(4, "big", signed=True)

    wd = Path(tempfile.mkdtemp(prefix="tb_top_"))
    write_rom(p, wd / "rom.hex")
    (wd / "host.txt").write_text(f"{len(host)}\n" + "\n".join(str(b) for b in host) + "\n")
    prm = widths(args.w)
    prm.pop("F")
    defs = [f"-Ptb_top.{k}={v}" for k, v in prm.items()] + [f"-Ptb_top.CODES_PC={p.codes_pc_scheduled}",
                                                            f"-Ptb_top.N_EXPECT={len(expect)}"]
    env = dict(os.environ, PATH=f"{OSS}:{os.environ['PATH']}")
    subprocess.run(["iverilog", "-g2012", "-o", "sim.vvp", *defs, str(ROOT / "tb/tb_top.v"), str(ROOT / "rtl/top.v"),
                    str(ROOT / "rtl/core.v"), str(ROOT / "rtl/uart.v")], cwd=wd, check=True, env=env)
    run = subprocess.run(["vvp", "-n", "sim.vvp"], cwd=wd, capture_output=True, text=True, env=env)
    print("\n".join(run.stdout.strip().splitlines()[-3:]) if run.stdout.strip() else run.stderr[-2000:])
    got = bytes(int(x) for x in (wd / "board.txt").read_text().split())
    shutil.rmtree(wd)
    ok = got == expect
    first = next((i for i, (a, b) in enumerate(zip(got, expect)) if a != b), None)
    print(f"w = {args.w}: {len(host)} bytes sent, {len(got)} received, {len(expect)} expected, "
          f"{'MATCH' if ok else f'MISMATCH (first difference at byte {first})'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
