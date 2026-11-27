"""RTL core against the microcode interpreter in iverilog.
The long runs (10000 steps per word length) are fpga/run_tb.py."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

OSS = Path.home() / ".local/oss-cad-suite/bin"
pytestmark = pytest.mark.skipif(not (OSS / "iverilog").exists() and shutil.which("iverilog") is None,
                                reason="iverilog not installed")


@pytest.mark.parametrize("w, Ts", [(12, 0.05), (16, 0.01), (24, 0.001)])
def test_rtl_matches_microcode(w, Ts):
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "fpga.run_tb", "--w", str(w), "--Ts", str(Ts), "--steps", "15"],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 runs differ" in r.stdout


@pytest.mark.parametrize("w, Ts", [(12, 0.05), (16, 0.01), (24, 0.001)])
def test_uart_top_matches_microcode(w, Ts):
    """Constants, states, CODES, STEP, and register reads over the simulated
    UART: every reply byte equals the interpreter's."""
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "fpga.run_tb_top", "--w", str(w), "--Ts", str(Ts), "--steps", "2",
                        "--ctx", "2"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MATCH" in r.stdout and "MISMATCH" not in r.stdout


def test_rtl_with_core_clock_enable_matches_microcode():
    """With the core advancing one clock in 3, the results are unchanged."""
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "fpga.run_tb", "--w", "12", "--Ts", "0.05", "--steps", "10",
                        "--cediv", "3"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 runs differ" in r.stdout
