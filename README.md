# ep505-bell-feedback

Code for an EP505 (Embry-Riddle, Fall 2026) final project: output-feedback
stabilization of the Bell state |Φ+⟩ for two always-coupled qubits under a
continuous Z⊗Z parity measurement, with four local drives.

## Layout

| Path | Contents |
|------|----------|
| `sim/` | Stochastic master equation simulator (Rouchon-Ralph step), feedback laws, observer, hardware hooks (quantizer, hold, delay), fixed-point filter, experiments, figure script |
| `verification/` | sympy checks of the model and the derivations, Lean 4 proofs of the Lie-algebra dimensions |
| `fpga/` | Microcode compiler, Verilog core for the Tang Nano 20K, testbenches, host and hardware-in-the-loop scripts |
| `tests/` | pytest suite (fast tests by default, `-m slow` for the statistical ones) |
| `tools/` | Two short runs the report quotes, and a script that prints the report numbers from saved data |

## Run

Python 3.12 with the packages in `requirements.txt`.

```
pip install -r requirements.txt
python -m pytest -q -m "not slow"          # fast tests
python -m pytest -q -m slow                # statistical tests (minutes)
python verification/check_control_model.py # sympy model check
python -m sim.make_figures --run --data results --out figs   # all data and figures (~2 h)
```

Lean proofs: `cd verification && lean EP505Brackets.lean && lean EP505Su4.lean` (core Lean 4 v4.30.0, no mathlib).

FPGA (needs yosys, nextpnr-himbaechel, gowin_pack, iverilog):

```
python -m fpga.run_tb --w 16 --Ts 0.01 --steps 2000   # RTL against the interpreter
python -m fpga.build --w 16 --Ts 0.01                 # bitstream
```
