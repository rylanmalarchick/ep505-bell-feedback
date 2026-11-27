"""Recovery of fpga.host.Board from bytes lost on the way to the host."""
import numpy as np

from fpga import host


class LossyBoard(host.Board):
    """Board double: each context holds a step counter; codes = (counter, ctx, 0, 0).
    The reply to the first STEP batch loses its last 37 bytes."""

    def __init__(self):  # no serial port
        self.count = {}
        self.pending = b""
        self.dropped = False
        self.recoveries = 0

    def _write(self, data):
        i = 0
        while i < len(data):
            cmd, ctx = data[i], data[i + 1]
            if cmd == 0xA1:
                self.count[ctx] = self.count.get(ctx, 0) + 1
                i += 5
            else:
                i += 2
            c = self.count.get(ctx, 0)
            self.pending += b"".join(int(v).to_bytes(2, "big", signed=True) for v in (c, ctx, 0, 0))
        if not self.dropped and data[0] == 0xA1:
            self.pending = self.pending[:-37]
            self.dropped = True

    def _read(self, n):
        if len(self.pending) < n:
            self.pending = b""
            raise TimeoutError("short")
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def _drain(self):
        self.pending = b""


def test_step_recovers_without_stepping_twice():
    b = LossyBoard()
    ctxs = list(range(70))
    n = b.step(ctxs, np.zeros(70, dtype=int))
    assert b.recoveries == 1
    assert all(b.count[c] == 1 for c in ctxs)          # each context stepped exactly once
    assert np.array_equal(n[:, 0], np.ones(70)) and np.array_equal(n[:, 1], ctxs)
