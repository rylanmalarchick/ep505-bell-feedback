"""Batched torch version of the closed loop, for reinforcement learning.

Same model and order as sim.closed_loop.simulate with a separate observer:
the plant takes Rouchon-Ralph steps of dt; the controller decides every Ts
from the filter estimate; the decision is quantized (wc bits over umax),
held for Ts, and delayed by tau before it reaches the plant; the filter
takes one step of length Ts with the summed output and the held, undelayed
command. tau must be a whole number of sample periods, so the delay works
in decisions.

Full-parity model only. Complex128 throughout.
"""
import numpy as np
import torch

from .coords import BASIS, H_J, H_ROWS, I4, PHI_PLUS, U_0, ZZ
from .params import Plant


def _t(a, dev):
    return torch.as_tensor(np.asarray(a), device=dev)


class TorchLoop:
    def __init__(self, B: int, Ts: float, tau: float, prm: Plant = Plant(), dt: float = 1e-3,
                 wc: int = 12, umax: float = 4.0, device: str = "cpu", seed: int = 0):
        if prm.meas != "ZZ":
            raise ValueError("TorchLoop implements the full-parity model only")
        self.B, self.Ts, self.dt, self.prm = B, Ts, dt, prm
        self.n_s = round(Ts / dt)
        self.d = round(tau / Ts)
        if abs(self.n_s * dt - Ts) > 1e-12 or abs(self.d * Ts - tau) > 1e-12:
            raise ValueError("Ts must be a multiple of dt and tau a multiple of Ts")
        self.wc, self.umax = wc, umax
        self.delta = umax / 2 ** (wc - 1)
        self.dev = device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.ZZ = _t(ZZ, device)
        self.I4 = _t(I4, device)
        self.HJ = _t(H_J, device)
        self.BASIS = _t(BASIS, device)
        self.H_ROWS = _t(H_ROWS, device).to(torch.float64)
        self.U0 = _t(U_0, device).to(torch.float64)
        self.phi = _t(PHI_PLUS, device)

    # ---- model pieces (match sim.plant) ----
    def hamiltonian(self, u):
        return self.prm.omega * self.ZZ + torch.einsum("nj,jab->nab", u.to(torch.complex128), self.HJ)

    def kraus_step(self, rho, H, dy, dt):
        M, eta = self.prm.M, self.prm.eta
        L = np.sqrt(M) * self.ZZ
        L2 = M * self.I4
        dyc = dy[:, None, None].to(torch.complex128)
        K = (self.I4 + (-1j * H - 0.5 * L2) * dt + np.sqrt(eta) * L * dyc + 0.5 * eta * L2 * (dyc ** 2 - dt))
        out = K @ rho @ K.conj().transpose(1, 2) + (1 - eta) * dt * (L @ rho @ L)
        out = 0.5 * (out + out.conj().transpose(1, 2))
        tr = torch.einsum("nii->n", out).real
        return out / tr[:, None, None]

    def p1(self, rho):
        d = torch.einsum("nii->ni", rho).real
        return d[:, 0] - d[:, 1] - d[:, 2] + d[:, 3]

    def to_x(self, rho):
        return torch.einsum("aij,nji->na", self.BASIS, rho).real

    def fidelity(self, rho):
        return torch.einsum("i,nij,j->n", self.phi.conj(), rho, self.phi).real

    def offset_law(self, x, k=2.0):
        return self.U0 + k * x @ self.H_ROWS.T

    def quantize(self, u):
        half = 2 ** (self.wc - 1)
        return torch.clamp(torch.round(u / self.delta), -half, half - 1) * self.delta

    # ---- episode ----
    def reset(self, rho0: torch.Tensor):
        self.rho = rho0.clone()
        self.rho_hat = rho0.clone()
        self.queue = [torch.zeros(self.B, 4, dtype=torch.float64, device=self.dev) for _ in range(self.d)]
        return self.to_x(self.rho_hat)

    def step(self, u_cmd: torch.Tensor, dW: torch.Tensor | None = None):
        """Apply one decision (B, 4), before quantization. dW, if given, has shape
        (n_s, B). Returns the new estimate x_hat and the true fidelity."""
        v = self.quantize(u_cmd)
        if self.d:
            self.queue.append(v)
            u = self.queue.pop(0)
        else:
            u = v
        H = self.hamiltonian(u)
        sq = np.sqrt(self.dt)
        dy_acc = torch.zeros(self.B, dtype=torch.float64, device=self.dev)
        for i in range(self.n_s):
            w = (torch.randn(self.B, generator=self.gen, dtype=torch.float64, device=self.dev) * sq
                 if dW is None else dW[i])
            dy = self.prm.c * self.p1(self.rho) * self.dt + w
            self.rho = self.kraus_step(self.rho, H, dy, self.dt)
            dy_acc = dy_acc + dy
        self.rho_hat = self.kraus_step(self.rho_hat, self.hamiltonian(v), dy_acc, self.Ts)
        return self.to_x(self.rho_hat), self.fidelity(self.rho)
