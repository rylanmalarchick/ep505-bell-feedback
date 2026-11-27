"""Phase 6H: the zero-fidelity set {F = 0}, its transverse exponent, and a switch.

The set {F = 0} is invariant under the offset law
(verification/check_half_parity.py, Part A). These runs measure how the loop
behaves next to it. Full-parity model, Phase 2 parameters, dt = 1e-3,
perfect observer unless stated.

--part lyap   Transverse exponent lambda(k) of F next to {F = 0}. Start
              rho0 = (1 - eps) Phi- + eps I / 4 (eps = 1e-2), "face", or
              rho0 = (1 - eps) Psi+ + eps I / 4, "psi_plus". The second
              set (LYAP2) adds k = 0.25 and the Psi+ start. Every R_LYAP
              steps, scale the Phi+ amplitude by s (rho <- D rho D / Tr,
              D = 1 - (1 - s) |Phi+><Phi+|) so that F = F_REF again, and add
              log(F / F_REF) to a sum. lambda = sum / (T - T_BURN), counting
              only after T_BURN. lambda > 0: F grows next to the set (it
              repels). lambda < 0: F decays (it attracts). Numeric estimate
              of the top Lyapunov exponent of F along the set.
--part switch Offset law against the switching law (sim.laws.Hysteresis,
              u_c = (1, 1, 1, 0), F_LOW, F_HIGH) from Phi-, Psi+ (both F = 0)
              and the eps = 1e-4 start of Table IV, k in (0.25, 2), T = 80.
              verification/check_switch.py: u_c leaves no face of {F = 0}
              invariant.
--part coherent  Test of the exponent: the Table IV start (incoherent: it
              adds Phi+ population but no coherence) against a coherent start
              psi = sqrt(1 - delta) Phi- + sqrt(delta) Phi+, delta = 1e-6.
              Offset law, k in K_COH, T = 200, N = 200. Records the first
              time with F > 0.5 and P(F > 0.99) at T.
--part quant  Quantized command at Ts = dt, tau = 0, full scale 4, w in
              (3, 4, 5, 6): offset law against the switching law, k = 2,
              from |00>, T = 40. Tests the dead-zone explanation of the
              Figure 5 failures.

    python -m sim.phase6h --part lyap --out results
"""
import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .closed_loop import Hooks, _whole_steps, simulate
from .coords import I4, PHI_MINUS, PHI_PLUS, PSI_PLUS, fidelity, to_x
from .laws import Hysteresis, Offset
from .params import Plant
from .phase2 import wilson
from .plant import plant_step

DT = 1e-3
K_LYAP = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 4.0)
LYAP2 = (("face", 0.22), ("face", 0.25), ("face", 0.27), ("psi_plus", 0.25), ("psi_plus", 2.0),
         ("psi_plus", 4.0))
T_LYAP, T_BURN, R_LYAP, F_REF, EPS_LYAP, N_LYAP = 200.0, 20.0, 10, 1e-12, 1e-2, 200
SEED_LYAP = 18001
U_C = (1.0, 1.0, 1.0, 0.0)
F_LOW, F_HIGH = 0.05, 0.1
K_SWITCH, T_SWITCH, N_SWITCH = (0.25, 2.0), 80.0, 500
SEED_SWITCH = 19001
K_COH, T_COH, N_COH, DELTA_COH, EPS_COH = (0.15, 0.25, 0.5), 200.0, 200, 1e-6, 1e-4
SEED_COH = 21001
W_QUANT, T_QUANT, N_QUANT, UMAX = (3, 4, 5, 6), 40.0, 200, 4.0
SEED_QUANT = 20001
PP = np.outer(PHI_PLUS, PHI_PLUS.conj())


def lyap_job(args):
    k, seed = args[:2]
    start = args[2] if len(args) > 2 else "face"
    psi = PHI_MINUS if start == "face" else PSI_PLUS
    prm = Plant()
    N = N_LYAP
    rho = np.repeat(((1 - EPS_LYAP) * np.outer(psi, psi.conj()) + EPS_LYAP * I4 / 4)[None], N, axis=0)
    law = Offset(k)
    rng = np.random.default_rng(seed)
    n_steps = _whole_steps(T_LYAP, DT, "T")
    n_burn = _whole_steps(T_BURN, DT, "T_BURN")
    acc = np.zeros(N)

    def rescale(r):
        F = fidelity(r)
        s = np.sqrt(F_REF / F)
        D = I4[None] - (1 - s)[:, None, None] * PP[None]
        r = D @ r @ D.conj().transpose(0, 2, 1)
        return r / np.einsum("nii->n", r).real[:, None, None], F

    rho, _ = rescale(rho)
    for i in range(n_steps):
        rho, _ = plant_step(rho, law(to_x(rho)), rng.normal(0.0, np.sqrt(DT), N), DT, prm)
        if (i + 1) % R_LYAP == 0:
            rho, F = rescale(rho)
            if i + 1 > n_burn:
                acc += np.log(F / F_REF)
    return acc / (T_LYAP - T_BURN)


def start_states():
    eps = 1e-4
    return {"phi_minus": np.outer(PHI_MINUS, PHI_MINUS.conj()),
            "psi_plus": np.outer(PSI_PLUS, PSI_PLUS.conj()),
            "eps_1e-4": (1 - eps) * np.outer(PHI_MINUS, PHI_MINUS.conj()) + eps * I4 / 4}


def make_law(name, k):
    if name == "offset":
        return Offset(k)
    return Hysteresis(k, U_C, F_LOW, F_HIGH)


def switch_job(args):
    lname, k, sname, seed = args
    rho = np.repeat(start_states()[sname][None], N_SWITCH, axis=0)
    res = simulate(rho, make_law(lname, k), Plant(), T=T_SWITCH, dt=DT, seed=seed,
                   record_every=round(T_SWITCH / DT))
    return res.F[-1]


def coherent_job(args):
    k, start, seed = args
    if start == "coherent":
        psi = np.sqrt(1 - DELTA_COH) * PHI_MINUS + np.sqrt(DELTA_COH) * PHI_PLUS
        rho0 = np.outer(psi, psi.conj())
    else:
        rho0 = (1 - EPS_COH) * np.outer(PHI_MINUS, PHI_MINUS.conj()) + EPS_COH * I4 / 4
    rho = np.repeat(rho0[None], N_COH, axis=0)
    res = simulate(rho, Offset(k), Plant(), T=T_COH, dt=DT, seed=seed, record_every=100)
    above = res.F > 0.5
    t_half = np.where(above.any(axis=0), res.t[above.argmax(axis=0)], np.inf)
    return res.F[-1], t_half


def quant_job(args):
    lname, w, seed = args
    rho = np.zeros((N_QUANT, 4, 4), dtype=complex)
    rho[:, 0, 0] = 1
    res = simulate(rho, make_law(lname, 2.0), Plant(), T=T_QUANT, dt=DT, seed=seed,
                   rho_hat0=rho.copy(), hooks=Hooks(w=w, umax=UMAX), record_every=round(T_QUANT / DT))
    return res.F[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("lyap", "lyap2", "coherent", "switch", "quant"), required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if args.part == "coherent":
        jobs = [(k, st, SEED_COH + i) for i, (k, st) in
                enumerate((k, st) for k in K_COH for st in ("incoherent", "coherent"))]
        with ProcessPoolExecutor(args.workers) as ex:
            outs = list(ex.map(coherent_job, jobs))
        print(f"commit {sha}  coherent against incoherent start, offset law, T = {T_COH}, N = {N_COH}, "
              f"delta = {DELTA_COH:g}, eps = {EPS_COH:g}")
        print("  k    | start      | P(F>0.99) at T [95% CI] | median first t with F > 0.5 | P(escaped by T)")
        rows = []
        for (k, st, _), (F, th) in zip(jobs, outs):
            k99 = int(np.sum(F > 0.99))
            lo, hi = wilson(k99, N_COH)
            rows.append((k, st == "coherent", k99 / N_COH, lo, hi, float(np.median(th)), float(np.mean(np.isfinite(th)))))
            print(f" {k:5g} | {st:10s} | {k99 / N_COH:.3f} [{lo:.3f}, {hi:.3f}] | {np.median(th):8.2f} | "
                  f"{np.mean(np.isfinite(th)):.3f}")
        np.savez(out / "phase6h_coherent.npz", rows=np.array(rows), t_half=np.stack([o[1] for o in outs]),
                 F_final=np.stack([o[0] for o in outs]), T=T_COH, N=N_COH, commit=sha)
        print(f"wrote {out / 'phase6h_coherent.npz'}")
        return
    if args.part == "lyap2":
        jobs = [(k, SEED_LYAP + 100 + i, st) for i, (st, k) in enumerate(LYAP2)]
        with ProcessPoolExecutor(args.workers) as ex:
            lam = list(ex.map(lyap_job, jobs))
        print(f"commit {sha}  transverse exponent, second set, T = {T_LYAP}, burn-in {T_BURN}, N = {N_LYAP}")
        print(" start    |  k    | mean lambda | std error | median | fraction lambda > 0")
        for (k, _, st), l in zip(jobs, lam):
            print(f" {st:8s} | {k:5g} | {l.mean():+.4f}     | {l.std() / np.sqrt(l.size):.4f}    | "
                  f"{np.median(l):+.3f} | {np.mean(l > 0):.3f}")
        np.savez(out / "phase6h_lyap2.npz", start=np.array([j[2] for j in jobs]), k=np.array([j[0] for j in jobs]),
                 lam=np.stack(lam), T=T_LYAP, T_BURN=T_BURN, N=N_LYAP, F_REF=F_REF, commit=sha)
        print(f"wrote {out / 'phase6h_lyap2.npz'}")
        return
    if args.part == "lyap":
        jobs = [(k, SEED_LYAP + i) for i, k in enumerate(K_LYAP)]
        with ProcessPoolExecutor(args.workers) as ex:
            lam = list(ex.map(lyap_job, jobs))
        print(f"commit {sha}  transverse exponent, T = {T_LYAP}, burn-in {T_BURN}, N = {N_LYAP}, "
              f"rescale every {R_LYAP} steps to F = {F_REF:g}")
        print("  k    | mean lambda | std error | q0.1 | median | q0.9 | fraction of runs with lambda > 0")
        for (k, _), l in zip(jobs, lam):
            print(f" {k:5g} | {l.mean():+.4f}     | {l.std() / np.sqrt(l.size):.4f}    | {np.quantile(l, 0.1):+.3f} | "
                  f"{np.median(l):+.3f} | {np.quantile(l, 0.9):+.3f} | {np.mean(l > 0):.3f}")
        np.savez(out / "phase6h_lyap.npz", k=np.array(K_LYAP), lam=np.stack(lam), T=T_LYAP, T_BURN=T_BURN,
                 N=N_LYAP, F_REF=F_REF, R=R_LYAP, SEED=SEED_LYAP, commit=sha)
        print(f"wrote {out / 'phase6h_lyap.npz'}")
        return
    if args.part == "switch":
        combos = [(ln, k, s) for k in K_SWITCH for s in start_states() for ln in ("offset", "switch")]
        jobs = [(ln, k, s, SEED_SWITCH + i // 2) for i, (ln, k, s) in enumerate(combos)]
        fn, n, title = switch_job, N_SWITCH, f"switching law u_c = {U_C}, F_low = {F_LOW}, F_high = {F_HIGH}, T = {T_SWITCH}"
    else:
        combos = [(ln, w) for w in W_QUANT for ln in ("offset", "switch")]
        jobs = [(ln, w, SEED_QUANT + i // 2) for i, (ln, w) in enumerate(combos)]
        fn, n, title = quant_job, N_QUANT, f"quantized command, Ts = dt, tau = 0, full scale {UMAX}, k = 2, T = {T_QUANT}"
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(fn, jobs))
    print(f"commit {sha}  {title}, N = {n}, dt = {DT}. Pairs of laws share a seed.")
    rows = []
    for job, F in zip(jobs, outs):
        k99 = int(np.sum(F > 0.99))
        lo, hi = wilson(k99, n)
        rows.append((k99 / n, lo, hi, np.mean(F < 0.01)))
        print(" " + " | ".join(str(x) for x in job[:-1]) + f" | P(F>0.99) {k99 / n:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"P(F<0.01) {np.mean(F < 0.01):.3f}")
    np.savez(out / f"phase6h_{args.part}.npz", jobs=np.array([[str(x) for x in j[:-1]] for j in jobs]),
             rows=np.array(rows), F_final=np.stack(outs), N=n, U_C=np.array(U_C), F_LOW=F_LOW,
             F_HIGH=F_HIGH, commit=sha)
    print(f"wrote {out / f'phase6h_{args.part}.npz'}")


if __name__ == "__main__":
    main()
