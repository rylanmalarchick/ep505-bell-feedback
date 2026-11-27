"""Print the numbers that the report quotes, from the saved run data.

    python tools/report_numbers.py --data <dir with phase2.npz, phase3_*.npz> [--escape <escape.npz>]
        [--p7r <dir>] [--gpu <gpu_ksweep.npz>] [--ppo <phase6d_eval_p1.npz>] [--hil <hil.npz>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.coords import PHI_MINUS  # noqa: E402
from sim.phase2 import initial_states  # noqa: E402


def phase2(d):
    states = initial_states()
    law, name = d["law"], d["state"]
    print("== Phase 2 (Figures 1, 2). N =", int(d["N"]), "T =", float(d["T"]), "k =", float(d["K"]))
    rows = []
    for i in np.flatnonzero(law == "baseline"):
        fm0 = float(np.real(PHI_MINUS.conj() @ states[name[i]] @ PHI_MINUS))
        F = d["F_final"][i]
        se = F.std() / np.sqrt(F.size)
        rows.append((name[i], fm0, F.mean(), se, np.mean(d["p1_final"][i] > 0.99), np.mean(F > 0.99)))
    gap = [(abs(m - (1 - fm0)), se, n) for n, fm0, m, se, _, _ in rows]
    worst = max(gap)
    print(f"baseline: min over starts of P(p1 > 0.99 at T) = {min(r[4] for r in rows):.3f}")
    print(f"baseline: largest |mean F(T) - (1 - F-(0))| = {worst[0]:.4f} at {worst[2]} (se {worst[1]:.4f})")
    print(f"baseline: largest gap in standard errors (starts with se > 1e-3) = "
          f"{max(g / se for g, se, _ in gap if se > 1e-3):.2f}")
    print(f"baseline: largest gap over the starts with se <= 1e-3 = "
          f"{max(g for g, se, _ in gap if se <= 1e-3):.1e}")
    for n, fm0, m, se, _, p99 in rows:
        if p99 > 0.5:
            print(f"baseline: {n} has P(F>0.99) = {p99:.3f}, F-(0) = {fm0:.4f}")
    rnd = [r[5] for r in rows if r[0] not in ("00", "phi_minus") and r[5] <= 0.5]
    print(f"baseline: P(F>0.99) over the other {len(rnd)} random starts: {min(rnd):.3f} to {max(rnd):.3f}")
    off = [(np.mean(d["F_final"][i] > 0.99), name[i]) for i in np.flatnonzero(law == "offset")]
    rest = [p for p, n in off if n != "phi_minus"]
    print(f"offset: P(F>0.99) over the {len(rest)} starts other than Phi-: {min(rest):.3f} to {max(rest):.3f}")
    print(f"offset: Phi- start: {dict((n, p) for p, n in off)['phi_minus']:.3f}")


def observer(d):
    print("== Phase 3 observer (Figure 3). N =", int(d["N"]))
    for i, s in enumerate(d["s"]):
        print(f"s = {s:g}: P(F>0.99 at T) = {np.mean(d['F_final'][i] > 0.99):.3f}, "
              f"mean D at T = {d['err_mean'][i][-1]:.2e}")


def sweep(d):
    Ts, tau, w, p = d["Ts"], d["tau"], d["w"], d["p99"]
    print("== Phase 3 sweep (Figure 4). N =", int(d["N"]), "wall time", f"{float(d['wall_s']):.0f} s")

    def rng(mt, mtau, mw):
        sub = p[np.ix_(mt, mtau, mw)]
        return f"{sub.min():.3f} to {sub.max():.3f}"

    lo, hi = Ts <= 0.05, Ts >= 0.07
    print("Ts <= 0.05, tau <= 0.2, w >= 8:", rng(lo, tau <= 0.2, w >= 8))
    print("Ts <= 0.05, tau <= 0.2, w = 6: ", rng(lo, tau <= 0.2, w == 6))
    for t in (0.5, 1.0, 2.0):
        print(f"Ts <= 0.05, tau = {t:g}, all w:    ", rng(lo, tau == t, w > 0))
    print("Ts >= 0.07, all tau, all w:    ", rng(hi, tau >= 0, w > 0))
    i = int(np.flatnonzero(Ts == 0.07)[0])
    print("Ts = 0.07, tau = 0, by w:", {int(x): float(p[i, 0, j]) for j, x in enumerate(w)})


def wline(d):
    print("== Phase 3 w line (Figure 5). N =", int(d["N"]))
    for ip, (Ts, tau) in enumerate(d["points"]):
        F = d["F_final"][ip]
        print(f"Ts = {Ts:g}, tau = {tau:g}: " + ", ".join(
            f"w={int(x)}: {d['p99'][ip, j]:.3f} (F<0.01: {np.mean(F[j] < 0.01):.3f})"
            for j, x in enumerate(d["w"])))


def escape(d):
    print("== Escape (Table). N =", int(d["N"]))
    for i, k in enumerate(d["k"]):
        for j, eps in enumerate(d["eps"]):
            F = d["F_final"][i, j]
            print(f"k = {k:g}, eps = {eps:g}: P(F>0.99) = {np.mean(F > 0.99):.3f}, "
                  f"P(F<0.01) = {np.mean(F < 0.01):.3f}, median t_half = {np.median(d['t_half'][i, j]):.2f}")


def fixed6a(d):
    print("== 6A fixed-point filter (Figure 6). N =", int(d["N"]))
    for ip, (Ts, tau) in enumerate(d["points"]):
        lo, hi = d["ci_float"][ip]
        inside = [int(w) for iw, w in enumerate(d["w"]) if lo <= d["p99"][ip, iw] <= hi]
        print(f"Ts = {Ts:g}, tau = {tau:g}: float {d['p99_float'][ip]:.3f} [{lo:.3f}, {hi:.3f}]; "
              + ", ".join(f"w={int(w)}: {d['p99'][ip, iw]:.3f}" for iw, w in enumerate(d["w"]))
              + f"; smallest w inside the float interval: {min(inside) if inside else None}")


def noise6a(d):
    print("== 6A.5 rounding noise (Table). N =", int(d["N"]))
    for i, w in enumerate(d["w"]):
        print(f"w = {int(w)}: median {d['qe'][i, 0]:.1f} LSB, q0.99 {d['qe'][i, 2]:.1f} LSB, Hill {d['hill'][i]:.2f}, "
              f"runs error > 0.1: {int(np.sum(d['run_max_e'][i] > 0.1))}, runs min eig < -0.01: "
              f"{int(np.sum(d['run_min_eig'][i] < -0.01))}")


def reduced6b(d):
    print("== 6B reduced filters (Table). N =", int(d["N"]))
    keep = [i for i, s in enumerate(d["states"]) if s != "phi_minus"]
    for j, f in enumerate(d["filters"]):
        p = d["p99"][keep, j]
        print(f"{f}: dim {int(d['dims'][j])}, mults {int(d['mults'][j])}, P(F>0.99) {p.min():.3f} to {p.max():.3f}, "
              f"mean {p.mean():.3f}, median |x_hat - x| at t = {list(d['marks'])}: {np.median(d['err'][keep, j], 0).round(3)}")


def limits6c(ts, flip, delay):
    print("== 6C limits (Figure 7)")
    for i, Ts in enumerate(ts["Ts"]):
        print(f"Ts = {Ts:g}: hold {ts['p99'][i, 0]:.3f}, filter {ts['p99'][i, 1]:.3f}")
    for k, Ts, p, lo, hi in flip["rows"]:
        print(f"flip k = {k:g}, Ts = {Ts:g} (Ts 8k = {Ts * 8 * k:.3f}): {p:.3f}")
    for tau, flt, p, lo, hi in delay["rows"]:
        print(f"delay tau = {tau:g}, {'filter' if flt else 'perfect'}: {p:.3f}")


def half6e(d):
    print("== 6E half parity (Figure 9). N =", int(d["N"]), "T =", float(d["T"]))
    for law in ("baseline", "offset_x", "offset_y"):
        p = d["summary"][d["law"] == law, 0]
        print(f"{law}: P(F>0.99) {p.min():.3f} to {p.max():.3f}")


def zero6h(lyap, lyap2, coh, switch, quant):
    print("== 6H zero set (Figure 8, Tables)")
    for k, l in zip(lyap["k"], lyap["lam"]):
        print(f"face k = {k:g}: mean lambda {l.mean():+.3f}")
    for st, k, l in zip(lyap2["start"], lyap2["k"], lyap2["lam"]):
        print(f"{st} k = {k:g}: mean {l.mean():+.3f}, median {np.median(l):+.3f}, fraction > 0 {np.mean(l > 0):.3f}")
    for k, c, p, lo, hi, th, esc in coh["rows"]:
        print(f"coherent test k = {k:g}, {'coherent' if c else 'incoherent'}: P(F>0.99) {p:.3f}, median t_half {th:.2f}, "
              f"escaped {esc:.3f}")
    for job, row in zip(switch["jobs"], switch["rows"]):
        print(f"switch {' '.join(job)}: {row[0]:.3f}")
    for job, row in zip(quant["jobs"], quant["rows"]):
        print(f"quant {' '.join(job)}: {row[0]:.3f}")


def ksweep7r(d, gpu=None):
    print("== 7R k sweep (Table ksweep). N =", int(d["N"]))
    for i, k in enumerate(d["k"]):
        p, tc = d["p99"][i], d["t_cross"][i]
        line = f"k = {k:g}: min {p.min():.3f} mean {p.mean():.3f}, median first time {np.median(tc):.2f}"
        if gpu is not None:
            line += f" | gpu mean {gpu['p99'][i].mean():.3f} (diff {abs(gpu['p99'][i].mean() - p.mean()):.3f})"
        print(line)
    if gpu is not None:
        print(f"gpu wall {float(gpu['wall']):.0f} s on {gpu['device']}")


def params7r(d):
    print("== 7R plant sweep. N =", int(d["N"]))
    st = list(d["starts"])
    i00 = st.index("00")
    rest = [j for j in range(len(st)) if j != i00]
    for (om, eta), a, b in zip(d["grid"], d["p40"], d["p80"]):
        print(f"omega {om:g} eta {eta:g}: t40 min {a.min():.3f} mean {a.mean():.3f}; t80 min {b.min():.3f} "
              f"mean {b.mean():.3f}; |00> {a[i00]:.3f}/{b[i00]:.3f}; others t80 min {b[rest].min():.3f}")


def delay7r(d):
    print("== 7R delay ablation. N =", int(d["N"]))
    for k, tau, flt, p, lo, hi in d["rows"]:
        print(f"k = {k:g} tau = {tau:g} {'filter ' if flt else 'perfect'}: {p:.3f}")


def ppo_eval(d):
    print("== 6D eval at Ts =", float(d["Ts"]), "tau =", float(d["tau"]), "N =", int(d["N"]))
    st = list(d["starts"])
    ipm = st.index("phi_minus")
    rest = [j for j in range(len(st)) if j != ipm]
    for name, row in zip(d["names"], d["table"]):
        p = row[:, 0]
        print(f"{name}: others {p[rest].min():.3f} to {p[rest].max():.3f}, Phi- {p[ipm]:.3f}")


def hil(d):
    print("== 6F board (Table hil). N =", int(d["N"]))
    for w, Ts, tau, p, lo, hi, mism, smism, per, wall, rec in d["rows"]:
        print(f"w = {w:g} Ts = {Ts:g} tau = {tau:g}: {p:.3f}, codes differ {mism:g}, words differ {smism:g}, "
              f"recoveries {rec:g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with every .npz of make_figures --run")
    ap.add_argument("--escape", help="escape.npz, if it is not in --data")
    ap.add_argument("--p7r", help="directory with phase7r_*.npz")
    ap.add_argument("--gpu", help="gpu_ksweep.npz")
    ap.add_argument("--ppo", help="phase6d_eval_p1.npz")
    ap.add_argument("--hil", help="hil.npz of the board sweep")
    args = ap.parse_args()
    data = Path(args.data)
    phase2(np.load(data / "phase2.npz"))
    observer(np.load(data / "phase3_observer.npz"))
    sweep(np.load(data / "phase3_sweep.npz"))
    wline(np.load(data / "phase3_wline.npz"))
    escape(np.load(args.escape or data / "escape.npz"))
    load = {p.stem: p for p in data.glob("*.npz")}

    def get(name):
        return np.load(load[name]) if name in load else None

    if get("phase6a") is not None:
        fixed6a(get("phase6a"))
    if get("phase6a_noise") is not None:
        noise6a(get("phase6a_noise"))
    if get("phase6b") is not None:
        reduced6b(get("phase6b"))
    if all(get(n) is not None for n in ("phase6c_ts", "phase6c_flip", "phase6c_delay")):
        limits6c(get("phase6c_ts"), get("phase6c_flip"), get("phase6c_delay"))
    if get("phase6e_main") is not None:
        half6e(get("phase6e_main"))
    names = ("phase6h_lyap", "phase6h_lyap2", "phase6h_coherent", "phase6h_switch", "phase6h_quant")
    if all(get(n) is not None for n in names):
        zero6h(*(get(n) for n in names))
    if args.p7r:
        p7 = Path(args.p7r)
        ksweep7r(np.load(p7 / "phase7r_ksweep.npz"), np.load(args.gpu) if args.gpu else None)
        params7r(np.load(p7 / "phase7r_params.npz"))
        delay7r(np.load(p7 / "phase7r_delay2.npz"))
    if args.ppo:
        ppo_eval(np.load(args.ppo))
    if args.hil:
        hil(np.load(args.hil))


if __name__ == "__main__":
    main()
