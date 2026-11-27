"""Build the Tang Nano 20K bitstream for one word length.

    python -m fpga.build --w 16 --Ts 0.01 [--out fpga/build/w16]

Steps: compile the microcode (fpga.isa) and write rom.hex; yosys synth_gowin;
nextpnr-himbaechel for GW2AR-LV18QN88C8/I7 (family GW2A-18C) at 27 MHz;
gowin_pack. Writes top.fs, the logs, and report.json (resource use, the
maximum clock frequency from nextpnr, cycles per filter step, and the step
latency at 27 MHz).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from fpga import isa
from fpga.run_tb import OSS, PCB, ROOT, widths, write_rom
from sim.fixed import FixedFilter
from sim.params import Plant

DEVICE, FAMILY = "GW2AR-LV18QN88C8/I7", "GW2A-18C"
CLK_MHZ = 27.0


def run(cmd, cwd, log):
    env = dict(os.environ, PATH=f"{OSS}:{os.environ['PATH']}")
    with open(cwd / log, "w") as f:
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        tail = (cwd / log).read_text().splitlines()[-30:]
        raise SystemExit(f"{cmd[0]} failed, see {cwd / log}:\n" + "\n".join(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--Ts", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--core-div", type=int, default=1, help="the core advances one clock in CORE_DIV")
    ap.add_argument("--nodsp", action="store_true", help="synthesize the multiplier from LUTs, not DSP blocks")
    args = ap.parse_args()
    out = Path(args.out or ROOT / "build" / f"w{args.w}")
    out.mkdir(parents=True, exist_ok=True)
    p = isa.compile_program(FixedFilter(args.w, args.Ts, Plant(), 2.0))
    codes_pc = write_rom(p, out / "rom.hex")
    prm = widths(args.w)
    prm.pop("F")
    prm["CODES_PC"] = codes_pc
    prm["CORE_DIV"] = args.core_div
    chparam = " ".join(f"-set {k} {v}" for k, v in prm.items())
    srcs = " ".join(str(ROOT / "rtl" / f) for f in ("top.v", "core.v", "uart.v"))
    ys = (f"read_verilog {srcs}; chparam {chparam} top; "
          f"synth_gowin -top top{' -nodsp' if args.nodsp else ''} -json top.json")
    run(["yosys", "-p", ys], out, "yosys.log")
    run(["nextpnr-himbaechel", "--json", "top.json", "--write", "pnr.json", "--device", DEVICE,
         "--vopt", f"family={FAMILY}", "--vopt", f"cst={ROOT / 'tangnano20k.cst'}",
         "--freq", str(CLK_MHZ), "--seed", "1"], out, "nextpnr.log")
    run(["gowin_pack", "-d", FAMILY, "-o", "top.fs", "pnr.json"], out, "pack.log")

    log = (out / "nextpnr.log").read_text()
    fmax = [float(x) for x in re.findall(r"Max frequency for clock[^:]*:\s*([0-9.]+) MHz", log)]
    util = {}
    for m in re.finditer(r"^Info:\s+(\w+):\s+(\d+)/\s*(\d+)\s+(\d+)%", log, re.M):
        util[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    rom = isa.schedule(p)
    div_cycles = 2 * (args.w - 2) + 2 + 2
    cycles = len(rom) + 3 + div_cycles  # fetch to END through the pipeline, plus the divider stall
    # nextpnr prints one estimate after placement and one after routing. The
    # routed critical path of the w = 16 build (2026-11-10) runs from the reset
    # register to a divider enable and contains no multiplier or block RAM, so
    # the open-source GW2A timing model likely does not time those cells. The
    # design runs at the 27 MHz board clock; latency is reported at 27 MHz only.
    rep = dict(w=args.w, Ts=args.Ts, instructions=len(p.instrs), rom_words=len(rom), codes_pc=codes_pc,
               fmax_placed_mhz=fmax[0] if fmax else None, fmax_routed_mhz=fmax[-1] if fmax else None,
               timing_at_27mhz="PASS" if "PASS at 27.00 MHz" in log else "FAIL",
               utilization=util, nodsp=args.nodsp, core_div=args.core_div, cycles_per_step_estimate=cycles * args.core_div,
               latency_us_at_27mhz=cycles * args.core_div / CLK_MHZ)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
