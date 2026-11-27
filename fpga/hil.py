"""Hardware in the loop on the Tang Nano 20K.

    python -m fpga.hil --dev /dev/ttyUSB1 [--points 1,2] [--w 12,16,20,24] [--smoke]

For each word length w and each (Ts, tau) point of sim.phase6a:
  1. build the bitstream for (w, Ts) (fpga.build) and load it into the
     FPGA SRAM with openFPGALoader;
  2. run the loop of sim.fixed.simulate_fixed with the board in place of
     the fixed-point filter: the PC runs the float plant for N = 200
     trajectories (one board context each) and sends the output words, the
     board returns the command codes;
  3. run sim.fixed.FixedFilter in lockstep on the same words and count the
     codes that differ (they must all agree);
  4. record P(F > 0.99) at T and the final F of every run. With the seed of
     sim.phase6a for that point, the final F must equal the 6A model run
     exactly, run for run.
--smoke: load w = 16, Ts = 0.01, run 200 sample steps on 8 contexts, read
back the state words, compare, and measure the round trip of one frame.
Writes <out>/hil.npz and prints a table.
"""
import argparse
import faulthandler
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fpga import isa
from fpga.host import Board
from fpga.run_tb import OSS, ROOT
from sim.closed_loop import _whole_steps
from sim.coords import fidelity
from sim.fixed import FixedFilter
from sim.hooks import DelayLine
from sim.params import Plant
from sim.phase2 import wilson
from sim.phase6a import DT, K, N, POINTS, SEED, T, UMAX, WC
from sim.plant import plant_step


def build_and_load(w, Ts, out, core_div=1, nodsp=False):
    subprocess.run([sys.executable, "-m", "fpga.build", "--w", str(w), "--Ts", str(Ts), "--out", str(out),
                    "--core-div", str(core_div)] + (["--nodsp"] if nodsp else []),
                   check=True, stdout=subprocess.DEVNULL)
    env = dict(os.environ, PATH=f"{OSS}:{os.environ['PATH']}")
    r = subprocess.run(["openFPGALoader", "-b", "tangnano20k", str(out / "top.fs")], env=env,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit("openFPGALoader failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    time.sleep(0.5)


def run_point(board, w, Ts, tau, seed, n_traj, T_run):
    prm = Plant()
    flt = FixedFilter(w, Ts, prm, K, wc=WC, umax=UMAX)
    p = isa.compile_program(flt)
    rho = np.zeros((n_traj, 4, 4), dtype=complex)
    rho[:, 0, 0] = 1
    R, I = flt.encode(rho)
    board.load(p, {c: isa.load_state(p, R[c], I[c]) for c in range(n_traj)})
    ctxs = range(n_traj)
    n = board.codes(ctxs)
    n_sw = flt.codes(R, I)
    mism = int(np.sum(n != n_sw))
    n_steps, n_s, d = _whole_steps(T_run, DT, "T"), _whole_steps(Ts, DT, "Ts"), _whole_steps(tau, DT, "tau")
    rng = np.random.default_rng(seed)
    delay = DelayLine(d, np.zeros((n_traj, 4)))
    dy_acc = np.zeros(n_traj)
    t_board, frames = 0.0, 0
    for k in range(n_steps):
        u = delay.push(flt.command(n))
        rho, dy = plant_step(rho, u, rng.normal(0.0, np.sqrt(DT), n_traj), DT, prm)
        dy_acc += dy
        if (k + 1) % n_s == 0:
            dyw = flt.adc(dy_acc)
            dy_acc = np.zeros(n_traj)
            t0 = time.perf_counter()
            n = board.step(ctxs, dyw)
            t_board += time.perf_counter() - t0
            frames += n_traj
            R, I = flt.step(R, I, n_sw, dyw)
            n_sw = flt.codes(R, I)
            mism += int(np.sum(n != n_sw))  # the plant always gets the board's codes n
            faulthandler.dump_traceback_later(600, exit=True)  # re-arm the watchdog
    # Final state words read back from the board for a few contexts.
    state_mism = 0
    for c in range(min(4, n_traj)):
        for i in range(4):
            for j in range(i, 4):
                state_mism += board.read_reg(c, p.reg(("R", i, j))) != int(R[c, i, j])
                if j > i:
                    state_mism += board.read_reg(c, p.reg(("I", i, j))) != int(I[c, i, j])
    return fidelity(rho), mism, state_mism, t_board / max(frames, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/ttyUSB1")
    ap.add_argument("--w", default="12,16,20,24")
    ap.add_argument("--points", default="0,1,2", help="indices into sim.phase6a.POINTS")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=None, help="phase6a.npz of the 6A model run, to compare final F")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if args.smoke:
        build_and_load(16, 0.01, out / "build_w16_ts0.01")
        b = Board(args.dev)
        rt = b.round_trip(200)
        F, mism, smism, per = run_point(b, 16, 0.01, 0.1, SEED + 1, 8, 2.0)
        b.close()
        print(f"commit {sha}  smoke: w = 16, Ts = 0.01, 8 contexts, 200 sample steps: "
              f"{mism} codes differ, {smism} read-back state words differ; "
              f"single-frame round trip {rt * 1e3:.2f} ms; {per * 1e6:.1f} us per frame in batches")
        return 0 if mism == 0 and smism == 0 else 1
    ws = [int(x) for x in args.w.split(",")]
    pts = [int(x) for x in args.points.split(",")]
    model = np.load(args.model) if args.model else None
    rows, finals = [], []
    # Dump every thread's stack to the log if the run makes no progress: the
    # watchdog is re-armed after each sample step in run_point.
    faulthandler.dump_traceback_later(600, exit=True)
    print(f"commit {sha}  hardware in the loop, N = {N}, T = {T}, command {WC} bits over {UMAX}")
    print(" w  | Ts     tau  | P(F>0.99) [95% CI]   | codes differ | state words differ | us/frame | recoveries | equals 6A model")
    for w in ws:
        for ip in pts:
            Ts, tau = POINTS[ip]
            build_and_load(w, Ts, out / f"build_w{w}_ts{Ts}")
            b = Board(args.dev)
            t0 = time.time()
            try:
                F, mism, smism, per = run_point(b, w, Ts, tau, SEED + ip, N, T)
            except TimeoutError as e:
                print(f" {w:2d} | {Ts:<6g} {tau:<4g} | FAILED: {e}", flush=True)
                b.close()
                continue
            b.close()
            k99 = int(np.sum(F > 0.99))
            lo, hi = wilson(k99, N)
            same = "n/a"
            if model is not None:
                iw = list(model["w"]).index(w) + 1 if w in model["w"] else None
                if iw is not None:
                    same = "yes" if np.array_equal(F, model["F_final"][ip, iw]) else "NO"
            rows.append((w, Ts, tau, k99 / N, lo, hi, mism, smism, per, time.time() - t0, b.recoveries))
            finals.append(F)
            print(f" {w:2d} | {Ts:<6g} {tau:<4g} | {k99 / N:.3f} [{lo:.3f}, {hi:.3f}] | {mism:12d} | {smism:18d} | "
                  f"{per * 1e6:8.1f} | {b.recoveries:10d} | {same}", flush=True)
            # Save after every point, so a later failure loses nothing measured.
            np.savez(out / "hil.npz", rows=np.array(rows), F_final=np.stack(finals), N=N, T=T, commit=sha)
    print(f"wrote {out / 'hil.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
