Verification of the analytical claims
=====================================

sympy scripts (exact arithmetic). Each asserts the identities tagged [S]
in the report and exits nonzero on a failure. Run from this folder:

    python check_control_model.py   model equations from the SME
    python check_derivations.py     fidelity identity, closures, Lie-algebra dimensions
    python check_incoherent.py      invariant set B, drift of F under the offset law
    python check_half_parity.py     zero-fidelity set, half-parity model
    python check_switch.py          escape test for the switching input u_c
    python check_sampled.py         sample-period limit Ts* = 1/(8k), delay limit
    python check_reduction.py       smallest exact linear reductions of the filter

coords_spec.py holds the shared coordinate definitions.

Lean 4 (core Lean, no mathlib, toolchain pinned in lean-toolchain),
Lie-algebra dimensions 6 and 15:

    lean EP505Brackets.lean         dimension 6 with the Y drives only
    python gen_su4_certificate.py   writes the rank certificate in EP505Su4.lean
    lean EP505Su4.lean              dimension 15 (su(4)) with all four drives
