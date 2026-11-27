"""Host side of the board protocol over a serial port (standard library only).

    b = Board("/dev/ttyUSB1")
    b.load(program, {ctx: regs, ...})       # constants and per-context registers
    n = b.codes(range(N))                  # (N, 4) codes from the CODES entry
    n = b.step(range(N), dy_words)         # STEP on each context, (N, 4) new codes

Frames go out in chunks of CHUNK contexts. The board FIFOs hold 1024 bytes
each way, but the USB bridge of the board drops bytes in bursts above 256
bytes from the FPGA (measured: 32 frames of 8 bytes lossless, 48 frames
92.7%, 64 frames 88.5%). So CHUNK = 32. A read that gets no byte for
TIMEOUT seconds raises TimeoutError.
"""
import os
import select
import termios
import time

import numpy as np

from fpga import protocol

CHUNK = 32
# STEP frames are 5 bytes toward the board. The bridge loses reply bytes when
# a host write exceeds about one USB packet (64 bytes) while replies flow
# back (measured: 13 frames lossless, 16 frames
# 95.3%, 24 frames 76.0%). 12 frames = 60 bytes per write.
STEP_CHUNK = 12
TIMEOUT = 0.5


class Board:
    def __init__(self, dev: str, baud: int = 1_000_000):
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        speed = getattr(termios, f"B{baud}")
        attr = termios.tcgetattr(self.fd)
        attr[0] = 0                                     # iflag: raw
        attr[1] = 0                                     # oflag
        attr[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attr[3] = 0                                     # lflag: raw, no echo
        attr[4] = attr[5] = speed
        attr[6][termios.VMIN] = 0
        attr[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attr)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def close(self):
        os.close(self.fd)

    def _write(self, data: bytes):
        """Write all of data. The port is non-blocking: if the bridge accepts no
        byte for TIMEOUT seconds, raise TimeoutError instead of blocking forever."""
        view = memoryview(data)
        while view:
            _, w, _ = select.select([], [self.fd], [], TIMEOUT)
            if not w:
                raise TimeoutError(f"port accepted no byte for {TIMEOUT} s, {len(view)} bytes left")
            try:
                n = os.write(self.fd, view)
            except BlockingIOError:
                n = 0
            view = view[n:]

    def _read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            r, _, _ = select.select([self.fd], [], [], TIMEOUT)
            if not r:
                raise TimeoutError(f"board sent {len(out)} of {n} bytes, then nothing for {TIMEOUT} s")
            out += os.read(self.fd, n - len(out))
        return bytes(out)

    # The USB bridge of the board can drop bytes on the way to the host
    # Every command except STEP is idempotent (a write
    # of the same value, a read, CODES), so a short reply is handled by
    # draining the line and sending the same commands again. A STEP batch
    # whose reply comes back short has already run on the board; its codes
    # are recovered with CODES, which recomputes them from the stepped state.
    # This assumes that no byte is lost on the way to the board; fpga.hil
    # checks every code against a software filter, so a violation shows up
    # as a code mismatch. self.recoveries counts the recovered batches.
    recoveries = 0

    def _drain(self):
        time.sleep(0.1)
        while select.select([self.fd], [], [], 0.05)[0]:
            os.read(self.fd, 4096)

    def _exchange(self, data: bytes, n: int, attempts: int = 8) -> bytes:
        """Send idempotent commands and read n reply bytes, retrying on loss."""
        for _ in range(attempts):
            self._write(data)
            try:
                return self._read(n)
            except TimeoutError:
                self.recoveries += 1
                self._drain()
        raise TimeoutError(f"no complete reply after {attempts} attempts")

    def write_reg(self, ctx: int, reg: int, v: int):
        if self._exchange(protocol.write(ctx, reg, v), 1) != b"\x5a":
            raise RuntimeError("bad acknowledge")

    def read_reg(self, ctx: int, reg: int) -> int:
        return int.from_bytes(self._exchange(protocol.read(ctx, reg), 4), "big", signed=True)

    def load(self, p, ctx_regs: dict):
        data, n_ack = protocol.init_bytes(p, ctx_regs)
        cmds = [data[i:i + 7] for i in range(0, len(data), 7)]  # each write command is 7 bytes
        for i in range(0, len(cmds), CHUNK):
            part = cmds[i:i + CHUNK]
            if self._exchange(b"".join(part), len(part)) != b"\x5a" * len(part):
                raise RuntimeError("bad acknowledge during load")

    @staticmethod
    def _parse(rep: bytes, n: int) -> np.ndarray:
        return np.array([protocol.parse_codes(rep[8 * k:8 * k + 8]) for k in range(n)], dtype=np.int64)

    def codes(self, ctxs) -> np.ndarray:
        ctxs = list(ctxs)
        out = np.zeros((len(ctxs), 4), dtype=np.int64)
        for i in range(0, len(ctxs), CHUNK):
            part = ctxs[i:i + CHUNK]
            rep = self._exchange(b"".join(protocol.codes(c) for c in part), 8 * len(part))
            out[i:i + len(part)] = self._parse(rep, len(part))
        return out

    def step(self, ctxs, dy_words) -> np.ndarray:
        ctxs = list(ctxs)
        out = np.zeros((len(ctxs), 4), dtype=np.int64)
        for i in range(0, len(ctxs), STEP_CHUNK):
            part = ctxs[i:i + STEP_CHUNK]
            self._write(b"".join(protocol.step(c, int(d)) for c, d in zip(part, dy_words[i:i + STEP_CHUNK])))
            try:
                rep = self._read(8 * len(part))
                out[i:i + len(part)] = self._parse(rep, len(part))
            except TimeoutError:
                self.recoveries += 1
                self._drain()
                out[i:i + len(part)] = self.codes(part)
        return out

    def round_trip(self, n: int = 1000) -> float:
        """Mean seconds for one CODES frame sent and answered alone."""
        t0 = time.perf_counter()
        for _ in range(n):
            self.codes([0])
        return (time.perf_counter() - t0) / n
