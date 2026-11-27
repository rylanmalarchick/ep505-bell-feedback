/-
Kernel-checked verification of the Lie-algebra claims in
the project report. Core Lean 4 only, no mathlib;
every theorem closes by `decide`, so the Lean kernel evaluates the
arithmetic itself.

Claims verified:
  * the brackets of B = {Y1, 1Y, ZZ, XZ, ZX, XX} are the stated
    Gaussian-integer combinations of B (structure-constant table below),
    hence the real span of i*B is closed under the bracket;
  * the six elements are trace-orthogonal (Tr(B_a B_b) = 4 delta_ab),
    hence linearly independent, so the algebra has dimension exactly 6;
  * Y (x) Y is trace-orthogonal to all six but has norm 4, so it lies
    outside the span (in particular the closure is proper in su(4)).

Basis order: |00>, |01>, |10>, |11>.
-/

structure GInt where
  re : Int
  im : Int
deriving DecidableEq, Repr

instance : Add GInt := ⟨fun a b => ⟨a.re + b.re, a.im + b.im⟩⟩
instance : Sub GInt := ⟨fun a b => ⟨a.re - b.re, a.im - b.im⟩⟩
instance : Neg GInt := ⟨fun a => ⟨-a.re, -a.im⟩⟩
instance : Mul GInt :=
  ⟨fun a b => ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩⟩
instance (n : Nat) : OfNat GInt n := ⟨⟨(n : Int), 0⟩⟩

/-- the imaginary unit -/
def gi : GInt := ⟨0, 1⟩

abbrev Mat := Fin 4 → Fin 4 → GInt

def mmul (A B : Mat) : Mat := fun i j =>
  A i 0 * B 0 j + A i 1 * B 1 j + A i 2 * B 2 j + A i 3 * B 3 j

def bracket (A B : Mat) : Mat := fun i j =>
  mmul A B i j - mmul B A i j

def trace (A : Mat) : GInt := A 0 0 + A 1 1 + A 2 2 + A 3 3

-- 2x2 Paulis as Nat-indexed tables
def px : Nat → Nat → GInt
  | 0, 1 => 1 | 1, 0 => 1 | _, _ => 0
def py : Nat → Nat → GInt
  | 0, 1 => -gi | 1, 0 => gi | _, _ => 0
def pz : Nat → Nat → GInt
  | 0, 0 => 1 | 1, 1 => -1 | _, _ => 0
def pid : Nat → Nat → GInt
  | 0, 0 => 1 | 1, 1 => 1 | _, _ => 0

def kron (A B : Nat → Nat → GInt) : Mat := fun i j =>
  A (i.val / 2) (j.val / 2) * B (i.val % 2) (j.val % 2)

def YY : Mat := kron py py

/-- B0 = Y1, B1 = 1Y, B2 = ZZ, B3 = XZ, B4 = ZX, B5 = XX -/
def basis : Fin 6 → Mat
  | 0 => kron py pid
  | 1 => kron pid py
  | 2 => kron pz pz
  | 3 => kron px pz
  | 4 => kron pz px
  | 5 => kron px px

/-- upper-triangle structure constants: [B_a, B_b] = sum_k (t a b k) B_k
    for a < b; the full table is antisymmetrized below. -/
def t : Nat → Nat → Nat → GInt
  | 0, 2, 3 => ⟨0, 2⟩   -- [Y1,ZZ] =  2i XZ
  | 0, 3, 2 => ⟨0, -2⟩  -- [Y1,XZ] = -2i ZZ
  | 0, 4, 5 => ⟨0, 2⟩   -- [Y1,ZX] =  2i XX
  | 0, 5, 4 => ⟨0, -2⟩  -- [Y1,XX] = -2i ZX
  | 1, 2, 4 => ⟨0, 2⟩   -- [1Y,ZZ] =  2i ZX
  | 1, 3, 5 => ⟨0, 2⟩   -- [1Y,XZ] =  2i XX
  | 1, 4, 2 => ⟨0, -2⟩  -- [1Y,ZX] = -2i ZZ
  | 1, 5, 3 => ⟨0, -2⟩  -- [1Y,XX] = -2i XZ
  | 2, 3, 0 => ⟨0, 2⟩   -- [ZZ,XZ] =  2i Y1
  | 2, 4, 1 => ⟨0, 2⟩   -- [ZZ,ZX] =  2i 1Y
  | 3, 5, 1 => ⟨0, 2⟩   -- [XZ,XX] =  2i 1Y
  | 4, 5, 0 => ⟨0, 2⟩   -- [ZX,XX] =  2i Y1
  | _, _, _ => 0        -- [Y1,1Y] = [ZZ,XX] = [XZ,ZX] = 0

def c (a b k : Fin 6) : GInt := t a.val b.val k.val - t b.val a.val k.val

def lincomb (a b : Fin 6) : Mat := fun i j =>
  c a b 0 * basis 0 i j + c a b 1 * basis 1 i j + c a b 2 * basis 2 i j +
  c a b 3 * basis 3 i j + c a b 4 * basis 4 i j + c a b 5 * basis 5 i j

/-- every pairwise bracket is the stated combination of basis elements,
    so the real span of i*B is a Lie subalgebra -/
theorem bracket_closure :
    ∀ a b : Fin 6, ∀ i j : Fin 4,
      bracket (basis a) (basis b) i j = lincomb a b i j := by decide

/-- Tr(B_a B_b) = 4 delta_ab: the six elements are orthogonal, hence
    independent, hence the subalgebra has dimension exactly 6 -/
theorem basis_orthogonal :
    ∀ a b : Fin 6,
      trace (mmul (basis a) (basis b)) = if a = b then 4 else 0 := by decide

/-- YY is orthogonal to every basis element ... -/
theorem yy_orthogonal :
    ∀ k : Fin 6, trace (mmul YY (basis k)) = 0 := by decide

/-- ... but not to itself, so YY is outside the span: were
    YY = sum_k c_k B_k, then 4 = Tr(YY*YY) = sum_k c_k Tr(YY*B_k) = 0. -/
theorem yy_not_null : trace (mmul YY YY) = 4 := by decide
