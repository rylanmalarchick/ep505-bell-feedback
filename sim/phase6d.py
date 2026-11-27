"""Phase 6D: reinforcement-learning baseline with PPO.

Environment: sim.torch_loop.TorchLoop, full-parity model, the Phase 2 plant,
k-free. The policy sees what the offset law sees: the filter estimate x_hat
(15 numbers) at each sample, plus the time t / T. Its command goes through
the same 12-bit quantizer over [-4, 4), hold Ts, and delay tau.

Policies (Gaussian, mean from an MLP 16 -> 64 -> 64 -> 4, tanh hidden units):
    plain     u = 4 tanh(a)
    residual  u = offset law (k = 2) + tanh(a)
Reward per decision: F(t + Ts) - F(t) of the true state, so the return is
F(T) - F(0). The true state is used for the reward in training only.
PPO: GAE with gamma = 1 and lambda = 0.95, clip 0.2, 4 epochs, minibatch
4096, Adam 3e-4, value weight 0.5, gradient norm 0.5, B episodes per update.
Training starts: |00>, Phi-, Haar-random pure states, and Hilbert-Schmidt
random mixed states, a quarter each.

--part train   one (point, variant, seed), stops after ITERS updates or
               --hours of wall time; saves the policy and the learning curve.
--part eval    N = 500 per start, from |00>, Phi-, and haar_0 .. haar_4 of
               sim.phase2, deterministic policy (the mean), against the
               offset law on the same noise.

    python -m sim.phase6d --part train --point 0 --variant plain --seed 1 --out results
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .coords import PHI_MINUS
from .phase2 import initial_states, wilson
from .torch_loop import TorchLoop

POINTS = ((0.05, 0.5), (0.01, 0.1))
T = 40.0
B = 512
ITERS = 300
UMAX = 4.0


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
        self.v = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((4,), -1.0))
        with torch.no_grad():
            self.mu[-1].weight.mul_(0.01)
            self.mu[-1].bias.zero_()

    def dist(self, obs):
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())


def command(variant, env, x, a):
    if variant == "plain":
        return UMAX * torch.tanh(a)
    return env.offset_law(x) + torch.tanh(a)


def random_starts(n, rng):
    """n start states: |00>, Phi-, Haar pure, Hilbert-Schmidt mixed, a quarter each."""
    out = np.zeros((n, 4, 4), dtype=complex)
    kind = np.arange(n) % 4
    out[kind == 0, 0, 0] = 1
    out[kind == 1] = np.outer(PHI_MINUS, PHI_MINUS.conj())
    for i in np.flatnonzero(kind >= 2):
        if kind[i] == 2:
            psi = rng.normal(size=4) + 1j * rng.normal(size=4)
            out[i] = np.outer(psi, psi.conj()) / np.vdot(psi, psi).real
        else:
            G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            r = G @ G.conj().T
            out[i] = r / np.trace(r).real
    return out


def obs_of(x, t):
    return torch.cat([x.to(torch.float32), torch.full((x.shape[0], 1), t / T, device=x.device)], dim=1)


def rollout(env, pol, variant, rho0, deterministic=False):
    L = round(T / env.Ts)
    x = env.reset(rho0)
    F0 = env.fidelity(env.rho)
    O, A, LP, R, V = [], [], [], [], []
    f_prev = F0
    for j in range(L):
        o = obs_of(x, j * env.Ts)
        with torch.no_grad():
            d = pol.dist(o)
            a = d.mean if deterministic else d.sample()
            lp = d.log_prob(a).sum(1)
            v = pol.v(o).squeeze(1)
        x, f = env.step(command(variant, env, x, a.to(torch.float64)))
        O.append(o); A.append(a); LP.append(lp); V.append(v); R.append((f - f_prev).to(torch.float32))
        f_prev = f
    return dict(obs=torch.stack(O), act=torch.stack(A), logp=torch.stack(LP), rew=torch.stack(R),
                val=torch.stack(V), F=f_prev)


def ppo_update(pol, opt, ro, lam=0.95, clip=0.2, epochs=4, mb=4096):
    rew, val = ro["rew"], ro["val"]
    L, Bn = rew.shape
    adv = torch.zeros_like(rew)
    last = torch.zeros(Bn, device=rew.device)
    for t in reversed(range(L)):
        nxt = val[t + 1] if t + 1 < L else torch.zeros(Bn, device=rew.device)
        delta = rew[t] + nxt - val[t]
        last = delta + lam * last
        adv[t] = last
    ret = adv + val
    obs, act, logp = ro["obs"].reshape(L * Bn, -1), ro["act"].reshape(L * Bn, -1), ro["logp"].reshape(-1)
    adv, ret = adv.reshape(-1), ret.reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=obs.device)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            d = pol.dist(obs[idx])
            ratio = (d.log_prob(act[idx]).sum(1) - logp[idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
            vl = (pol.v(obs[idx]).squeeze(1) - ret[idx]).pow(2).mean()
            opt.zero_grad()
            (pg + 0.5 * vl).backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
            opt.step()


def train(args, dev):
    Ts, tau = POINTS[args.point]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = TorchLoop(B, Ts, tau, device=dev, seed=args.seed)
    pol = Policy().to(dev)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    curve, t0 = [], time.time()
    for it in range(ITERS):
        rho0 = torch.as_tensor(random_starts(B, rng), device=dev)
        ro = rollout(env, pol, args.variant, rho0)
        ppo_update(pol, opt, ro)
        F = ro["F"].cpu().numpy()
        curve.append((it, time.time() - t0, float(F.mean()), float(np.mean(F > 0.99))))
        print(f"iter {it:4d}  {time.time() - t0:7.0f} s  mean F(T) {F.mean():.4f}  P(F>0.99) {np.mean(F > 0.99):.3f}",
              flush=True)
        if time.time() - t0 > args.hours * 3600:
            break
    tag = f"p{args.point}_{args.variant}_s{args.seed}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), out / f"policy_{tag}.pt")
    np.save(out / f"curve_{tag}.npy", np.array(curve))
    gpu_h = (time.time() - t0) / 3600
    (out / f"train_{tag}.json").write_text(json.dumps(dict(point=args.point, variant=args.variant, seed=args.seed,
                                                           iters=len(curve), gpu_hours=gpu_h)))
    print(f"saved {tag}, {len(curve)} updates, {gpu_h:.2f} GPU hours")


def evaluate(args, dev):
    Ts, tau = POINTS[args.point]
    N = 500
    st = initial_states()
    starts = {"00": st["00"], "phi_minus": st["phi_minus"], **{f"haar_{i}": st[f"haar_{i}"] for i in range(5)}}
    policies = {"offset": None}
    for f in sorted(Path(args.policies).glob(f"policy_p{args.point}_*.pt")):
        pol = Policy().to(dev)
        pol.load_state_dict(torch.load(f, map_location=dev))
        policies[f.stem.replace("policy_", "")] = pol
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"commit {sha}  evaluation, Ts = {Ts}, tau = {tau}, N = {N}, T = {T}; same noise seed per start")
    print(f"{'policy':28s} | " + " | ".join(f"{s:>21s}" for s in starts))
    table = {}
    for name, pol in policies.items():
        cells = []
        for i, (s, rho) in enumerate(starts.items()):
            env = TorchLoop(N, Ts, tau, device=dev, seed=50000 + i)
            rho0 = torch.as_tensor(np.repeat(rho[None], N, axis=0), device=dev)
            if pol is None:
                x = env.reset(rho0)
                for _ in range(round(T / Ts)):
                    x, f = env.step(env.offset_law(x))
                F = f.cpu().numpy()
            else:
                variant = "residual" if "residual" in name else "plain"
                F = rollout(env, pol, variant, rho0, deterministic=True)["F"].cpu().numpy()
            k = int(np.sum(F > 0.99))
            lo, hi = wilson(k, N)
            cells.append((k / N, lo, hi))
        table[name] = cells
        print(f"{name:28s} | " + " | ".join(f"{p:.3f} [{lo:.3f},{hi:.3f}]" for p, lo, hi in cells), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"phase6d_eval_p{args.point}.npz", names=np.array(list(table)), starts=np.array(list(starts)),
             table=np.array(list(table.values())), N=N, T=T, Ts=Ts, tau=tau, commit=sha)
    print(f"wrote {out / f'phase6d_eval_p{args.point}.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("train", "eval"), required=True)
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--variant", choices=("plain", "residual"), default="plain")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--policies", default="results")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (train if args.part == "train" else evaluate)(args, dev)


if __name__ == "__main__":
    main()
