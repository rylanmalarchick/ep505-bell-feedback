"""Byte protocol of fpga/rtl/top.v, shared by the testbench driver and the host.

    A1 ctx dy[23:0]      STEP with output word dy, reply 4 codes x 2 bytes
    A4 ctx               CODES, reply 4 codes x 2 bytes
    A2 ctx reg v[31:0]   write register, reply 5A
    A3 ctx reg           read register, reply v[31:0]
"""
import numpy as np


def _s(v: int, nbytes: int) -> bytes:
    return int(v).to_bytes(nbytes, "big", signed=True)


def step(ctx: int, dy: int) -> bytes:
    return bytes([0xA1, ctx]) + _s(dy, 3)


def codes(ctx: int) -> bytes:
    return bytes([0xA4, ctx])


def write(ctx: int, reg: int, v: int) -> bytes:
    return bytes([0xA2, ctx, reg]) + _s(v, 4)


def read(ctx: int, reg: int) -> bytes:
    return bytes([0xA3, ctx, reg])


def parse_codes(b: bytes) -> np.ndarray:
    """8 reply bytes -> 4 signed codes."""
    return np.array([int.from_bytes(b[2 * j:2 * j + 2], "big", signed=True) for j in range(4)], dtype=np.int64)


def init_bytes(p, ctx_regs: dict) -> tuple[bytes, int]:
    """Write commands for the constants and for the per-context registers
    {ctx: regs}. Returns the bytes and the number of reply bytes (one ack each)."""
    out = b"".join(write(0, a, v) for a, v in p.consts.items())
    n = len(p.consts)
    for c, regs in ctx_regs.items():
        for a, v in regs.items():
            if a < 128:
                out += write(c, a, v)
                n += 1
    return out, n
