"""Build the report figures from saved simulation data.

    python -m sim.make_figures [--data results] [--out report/figs] [--run]

--run first regenerates all simulation data (RUNS below). That takes about
two hours. Without --run, the script reads the .npz files in
<data>. Some runs make no figure: their data feed tables in the report, and
tools/report_numbers.py prints them. Not in --run: the reinforcement-learning
runs (sim.phase6d, GPU) and the board runs (fpga.hil, needs the board).

Figure 1 (fig1_paths): F(t), 20 sample paths and the mean of N, baseline
    law and offset law, from |00> and from the random pure state haar_0.
Figure 2 (fig2_success): P(F > 0.99) at T per initial state with a 95%
    Wilson interval, both laws.
Figure 3 (fig3_observer): offset law with a wrong initial estimate, mean F
    and mean trace distance D(rho_hat, rho) over time, three mismatch sizes.
Figure 4 (fig4_sweep): P(F > 0.99) at T over the Ts x tau grid, one panel
    per word length w.
Figure 5 (fig5_wline): P(F > 0.99) at T against w at two (Ts, tau) points.
Figure 6 (fig6_fixed): P(F > 0.99) at T against the word length of the
    fixed-point filter at three (Ts, tau) points, with the float filter as
    the reference.
Figure 7 (fig7_limits): (a) the hold limit, P(F > 0.99) against Ts / (1/(8k))
    for k = 1, 2, 4; (b) the delay limit, perfect observer against the filter
    fed the undelayed command.
Figure 8 (fig8_exponent): transverse exponent of F next to {F = 0} against k.
Figure 9 (fig9_half): half-parity model, P(F > 0.99) at T = 200 per start,
    baseline law and the two offsets.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1 and 2 of the dataviz reference palette (light mode).
COLOR = {"offset": "#2a78d6", "baseline": "#eb6834"}
MARKER = {"offset": "o", "baseline": "s"}
LABEL = {"offset": "offset law", "baseline": "baseline law"}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
# Ordinal steps 300, 500, 700 of the reference blue ramp (light to dark).
BLUE_ORDINAL = ("#6da7ec", "#256abf", "#0d366b")
# Sequential blue ramp 100 to 700 of the reference palette.
BLUE_SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def fig1_paths(d, out):
    t = d["t"]
    starts = (("00", "|00>"), ("haar_0", "random state haar_0"))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex=True, sharey=True)
    for r, law in enumerate(("baseline", "offset")):
        for c, (state, name) in enumerate(starts):
            ax = axes[r, c]
            i = int(np.flatnonzero((d["law"] == law) & (d["state"] == state))[0])
            ax.plot(t, d["F_paths"][i], color=COLOR[law], lw=0.6, alpha=0.35)
            ax.plot(t, d["F_mean"][i], color=COLOR[law], lw=2.0, label=f"mean of N = {int(d['N'])}")
            ax.axhline(0.99, color=MUTED, lw=0.6, ls="--")
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_title(f"{LABEL[law]}, from {name}", fontsize=8, color=INK, loc="left")
            ax.legend(loc="lower right", frameon=False)
        axes[r, 0].set_ylabel("F")
    for ax in axes[-1]:
        ax.set_xlabel("t (units of 1/M)")
    fig.tight_layout()
    save(fig, out, "fig1_paths")


def fig2_success(d, out):
    states = list(dict.fromkeys(d["state"].tolist()))
    x = np.arange(len(states))
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for dx, law in ((-0.15, "baseline"), (0.15, "offset")):
        idx = [int(np.flatnonzero((d["law"] == law) & (d["state"] == s))[0]) for s in states]
        p, lo, hi = (d["summary"][idx, c] for c in (0, 1, 2))
        # At p = 0 or 1 the Wilson bound equals p up to round-off (|err| ~ 1e-19).
        yerr = np.clip([p - lo, hi - p], 0.0, None)
        ax.errorbar(x + dx, p, yerr=yerr, fmt=MARKER[law], ms=4, color=COLOR[law],
                    ecolor=COLOR[law], elinewidth=1.0, capsize=0, label=LABEL[law])
    ax.set_xticks(x, [s.replace("phi_minus", "Phi-").replace("_", " ") for s in states],
                  rotation=60, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f"P(F > 0.99) at T = {float(d['T']):g}")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    ax.set_title(f"N = {int(d['N'])} per point, 95% Wilson intervals", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig2_success")


def fig3_observer(d, out):
    t, S = d["t"], d["s"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for i, s in enumerate(S):
        col, ls = (MUTED, "--") if s == 0 else (BLUE_ORDINAL[i - 1], "-")
        lab = "s = 0 (exact start)" if s == 0 else f"s = {s:g}"
        a1.plot(t, d["F_mean"][i], color=col, lw=1.6, ls=ls, label=lab)
        if s > 0:
            a2.semilogy(t, d["err_mean"][i], color=col, lw=1.6, label=lab)
    a1.set_ylim(0.4, 1.01)
    a1.set_ylabel("mean F (true state)")
    a1.axhline(0.99, color=MUTED, lw=0.6, ls=":")
    a1.legend(loc="lower right", frameon=False)
    a2.set_ylabel(r"mean $D(\hat\rho, \rho)$")
    a2.legend(loc="upper right", frameon=False)
    for ax, title in ((a1, "(a) closed loop with a wrong initial estimate"),
                      (a2, "(b) estimation error, trace distance")):
        ax.set_xlabel("t (units of 1/M)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig3_observer")


def fig4_sweep(d, out):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
    Ts, tau, W, p99 = d["Ts"], d["tau"], d["w"], d["p99"]
    fig, axes = plt.subplots(1, len(W), figsize=(7.0, 2.6), sharey=True, layout="constrained")
    for iw, ax in enumerate(axes):
        im = ax.imshow(p99[:, :, iw], cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(len(Ts)):
            for j in range(len(tau)):
                v = p99[i, j, iw]
                ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1", ha="center", va="center",
                        fontsize=5.5, color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(tau)), [f"{x:g}" for x in tau], rotation=60)
        ax.set_yticks(range(len(Ts)), [f"{x:g}" for x in Ts])
        ax.set_xlabel(r"delay $\tau$")
        ax.set_title(f"w = {int(W[iw])} bits", fontsize=8, color=INK, loc="left")
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel(r"sample period $T_s$")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(f"P(F > 0.99) at T = {float(d['T']):g}")
    cb.outline.set_visible(False)
    fig.suptitle(f"N = {int(d['N'])} per cell, full scale {float(d['UMAX']):g}", fontsize=8,
                 color=INK, x=0.01, ha="left")
    save(fig, out, "fig4_sweep")


def fig5_wline(d, out):
    W, pts, p99, ci = d["w"], d["points"], d["p99"], d["ci"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    styles = ((COLOR["offset"], "o", "-"), (COLOR["baseline"], "s", "--"))
    for ip, (Ts, tau) in enumerate(pts):
        col, mk, ls = styles[ip]
        yerr = np.clip([p99[ip] - ci[ip, :, 0], ci[ip, :, 1] - p99[ip]], 0.0, None)
        ax.errorbar(W, p99[ip], yerr=yerr, color=col, marker=mk, ms=4, ls=ls, lw=1.2,
                    elinewidth=1.0, capsize=0, label=rf"$T_s$ = {Ts:g}, $\tau$ = {tau:g}")
    ax.set_xticks(W)
    ax.set_xlabel("quantizer word length w (bits)")
    ax.set_ylabel(f"P(F > 0.99) at T = {float(d['T']):g}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"N = {int(d['N'])} per point, full scale {float(d['UMAX']):g}", fontsize=8,
                 color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig5_wline")


def fig6_fixed(d, out):
    W, pts = d["w"], d["points"]
    fig, axes = plt.subplots(1, len(pts), figsize=(7.0, 2.3), sharey=True)
    xs = np.arange(len(W))
    for ip, ax in enumerate(axes):
        Ts, tau = pts[ip]
        lo, hi = d["ci_float"][ip]
        ax.axhspan(lo, hi, color=GRID, lw=0)
        ax.axhline(d["p99_float"][ip], color=MUTED, lw=1.0, ls="--", label="float filter, 95% interval")
        yerr = np.clip([d["p99"][ip] - d["ci"][ip, :, 0], d["ci"][ip, :, 1] - d["p99"][ip]], 0.0, None)
        ax.errorbar(xs, d["p99"][ip], yerr=yerr, color=COLOR["offset"], marker="o", ms=4, lw=1.2,
                    elinewidth=1.0, capsize=0, label="fixed-point filter")
        ax.set_xticks(xs, [str(int(w)) for w in W])
        ax.set_xlabel("filter word length w (bits)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_title(rf"$T_s$ = {Ts:g}, $\tau$ = {tau:g}", fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel(f"P(F > 0.99) at T = {float(d['T']):g}")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.suptitle(f"N = {int(d['N'])} per point, command {int(d['WC'])} bits", fontsize=8,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "fig6_fixed")


def fig7_limits(flip, delay, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    rows = flip["rows"]  # k, Ts, p99, lo, hi
    marks = {1.0: "o", 2.0: "s", 4.0: "^"}
    cols = {1.0: BLUE_ORDINAL[0], 2.0: BLUE_ORDINAL[1], 4.0: BLUE_ORDINAL[2]}
    for k in (1.0, 2.0, 4.0):
        r = rows[rows[:, 0] == k]
        x = r[:, 1] * 8 * k
        a1.errorbar(x, r[:, 2], yerr=np.clip([r[:, 2] - r[:, 3], r[:, 4] - r[:, 2]], 0, None),
                    color=cols[k], marker=marks[k], ms=4, lw=1.2, elinewidth=1.0, capsize=0, label=f"k = {k:g}")
    a1.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    a1.set_xlabel(r"$T_s \,/\, (1/(8k))$")
    a1.set_ylabel(f"P(F > 0.99) at T = {float(flip['T']):g}")
    a1.legend(loc="lower left", frameon=False)
    a1.set_title("(a) hold, perfect observer", fontsize=8, color=INK, loc="left")
    rows = delay["rows"]  # tau, is_filter, p99, lo, hi
    for flt, lab, col, mk in ((0, "perfect observer", COLOR["baseline"], "s"),
                              (1, "filter, undelayed command", COLOR["offset"], "o")):
        r = rows[rows[:, 1] == flt]
        a2.errorbar(r[:, 0], r[:, 2], yerr=np.clip([r[:, 2] - r[:, 3], r[:, 4] - r[:, 2]], 0, None), color=col,
                    marker=mk, ms=4, lw=1.2, elinewidth=1.0, capsize=0, label=lab)
    a2.axvline(0.051, color=MUTED, lw=0.8, ls="--")
    a2.set_xscale("log")
    a2.set_xticks([0.02, 0.04, 0.06, 0.1, 0.2], ["0.02", "0.04", "0.06", "0.1", "0.2"])
    a2.minorticks_off()
    a2.set_xlabel(r"delay $\tau$ (k = 2, $T_s$ = dt)")
    a2.legend(loc="lower left", frameon=False)
    a2.set_title("(b) delay", fontsize=8, color=INK, loc="left")
    for ax in (a1, a2):
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout()
    save(fig, out, "fig7_limits")


def fig8_exponent(lyap, lyap2, out):
    k, lam = lyap["k"], lyap["lam"]
    kk = list(k)
    mean = list(lam.mean(1))
    se = list(lam.std(1) / np.sqrt(lam.shape[1]))
    for st, k2, l2 in zip(lyap2["start"], lyap2["k"], lyap2["lam"]):
        if st == "face":
            kk.append(k2)
            mean.append(l2.mean())
            se.append(l2.std() / np.sqrt(l2.size))
    o = np.argsort(kk)
    kk, mean, se = np.array(kk)[o], np.array(mean)[o], np.array(se)[o]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.errorbar(kk, mean, yerr=se, color=COLOR["offset"], marker="o", ms=3.5, lw=1.2, capsize=0,
                label=r"next to the face, from $\Phi_-$ (mean)")
    # Next to Psi+ a minority of runs escape (fraction lambda > 0: 0.005, 0.185,
    # 0.43 at k = 0.25, 2, 4), which pulls the mean up. Plot the median.
    for st, k2, l2 in zip(lyap2["start"], lyap2["k"], lyap2["lam"]):
        if st == "psi_plus":
            ax.plot(k2, np.median(l2), marker="s", ms=4, color=COLOR["baseline"], ls="none",
                    label=r"next to $\Psi_+$ (median)" if k2 == min(lyap2["k"][lyap2["start"] == "psi_plus"]) else None)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("gain k")
    ax.set_ylabel(r"exponent $\lambda$ of F (1/time)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(f"N = {int(lyap['N'])} runs per point", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig8_exponent")


def fig9_half(d, out):
    states = list(dict.fromkeys(d["state"].tolist()))
    x = np.arange(len(states))
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    style = {"baseline": (COLOR["baseline"], "s", -0.2), "offset_x": (COLOR["offset"], "o", 0.0),
             "offset_y": (BLUE_ORDINAL[2], "^", 0.2)}
    lab = {"baseline": "baseline law", "offset_x": r"offset $X_1 - X_2$", "offset_y": r"offset $Y_1 - Y_2$"}
    for law, (col, mk, dx) in style.items():
        idx = [int(np.flatnonzero((d["law"] == law) & (d["state"] == s))[0]) for s in states]
        p, lo, hi = (d["summary"][idx, c] for c in (0, 1, 2))
        ax.errorbar(x + dx, p, yerr=np.clip([p - lo, hi - p], 0, None), fmt=mk, ms=4, color=col, elinewidth=1.0,
                    capsize=0, label=lab[law])
    names = [s.replace("psi_minus", "Psi-").replace("_", " ") for s in states]
    ax.set_xticks(x, names, rotation=60, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f"P(F > 0.99) at T = {float(d['T']):g}")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False)
    ax.set_title(f"half parity, target Psi+, N = {int(d['N'])} per point", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig9_half")


def fig10_rl(rl_dir, out):
    """PPO learning curves at Ts = 0.05, tau = 0.5 (sim.phase6d), from curve_*.npy
    of the training runs: columns update, seconds, mean F(T), P(F > 0.99)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3), sharex=True)
    for f in sorted(Path(rl_dir).glob("curve_p0_*.npy")):
        c = np.load(f)
        resid = "residual" in f.name
        col = COLOR["offset"] if resid else COLOR["baseline"]
        lab = ("residual policy" if resid else "plain policy") if f.name.endswith("_s1.npy") else None
        a1.plot(c[:, 0], c[:, 2], color=col, lw=1.0, label=lab)
        a2.plot(c[:, 0], c[:, 3], color=col, lw=1.0)
    a1.set_ylabel("mean F(T) in training")
    a2.set_ylabel("P(F > 0.99) in training")
    for ax in (a1, a2):
        ax.set_xlabel("PPO update (512 episodes each)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color=GRID, lw=0.5)
    a1.legend(loc="center right", frameon=False)
    a1.set_title("three seeds per policy", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig10_rl")


# Every simulation run that --run regenerates: (module, extra arguments).
RUNS = (
    ("phase2", []),
    ("phase3", ["--part", "observer"]), ("phase3", ["--part", "sweep"]), ("phase3", ["--part", "wline"]),
    ("escape", []), ("phase6a", []), ("phase6a_noise", []),
    ("phase6e", ["--part", "pilot"]), ("phase6e", ["--part", "main"]), ("phase6b", []),
    ("phase6c", ["--part", "ts"]), ("phase6c", ["--part", "flip"]), ("phase6c", ["--part", "delay"]),
    ("phase6h", ["--part", "lyap"]), ("phase6h", ["--part", "lyap2"]), ("phase6h", ["--part", "switch"]),
    ("phase6h", ["--part", "coherent"]), ("phase6h", ["--part", "quant"]),
)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results")
    ap.add_argument("--out", default="report/figs")
    ap.add_argument("--run", action="store_true", help="regenerate the data first (slow)")
    ap.add_argument("--rl", default=None, help="directory with the curve_*.npy files of sim.phase6d (Figure 10)")
    ap.add_argument("--workers", type=int, default=None, help="worker processes for each run (default: each module decides)")
    args = ap.parse_args()
    if args.run:
        import importlib
        import sys

        for name, extra in RUNS:
            mod = importlib.import_module(f"sim.{name}")
            sys.argv = [name, *extra, "--out", args.data] + ([] if args.workers is None else ["--workers", str(args.workers)])
            mod.main()
    out = Path(args.out)
    data = Path(args.data)
    d = np.load(data / "phase2.npz")
    fig1_paths(d, out)
    fig2_success(d, out)
    fig3_observer(np.load(data / "phase3_observer.npz"), out)
    fig4_sweep(np.load(data / "phase3_sweep.npz"), out)
    fig5_wline(np.load(data / "phase3_wline.npz"), out)
    fig6_fixed(np.load(data / "phase6a.npz"), out)
    fig7_limits(np.load(data / "phase6c_flip.npz"), np.load(data / "phase6c_delay.npz"), out)
    fig8_exponent(np.load(data / "phase6h_lyap.npz"), np.load(data / "phase6h_lyap2.npz"), out)
    fig9_half(np.load(data / "phase6e_main.npz"), out)
    if args.rl:
        fig10_rl(args.rl, out)


if __name__ == "__main__":
    main()
