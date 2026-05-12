-- HDIM Core — Clifford algebra and invariant layer.
-- Uses Lean4 core only; Float equality denotes ideal mathematical equality.

-- ============================================================
--  0. Helper Lemma
-- ============================================================

theorem pow_pos (b n : Nat) (hb : b > 0) : b ^ n > 0 := by
  induction n with
  | zero => rw [Nat.pow_zero]; exact Nat.zero_lt_one
  | succ n ih => rw [Nat.pow_succ]; exact Nat.mul_pos ih hb

-- ============================================================
--  1. Clifford Algebra Signature
-- ============================================================

structure CliffordSignature where
  p : Nat
  q : Nat
  r : Nat
  deriving Repr, BEq

def cliffordDim (sig : CliffordSignature) : Nat :=
  2 ^ (sig.p + sig.q + sig.r)

theorem cliffordDim_pos (sig : CliffordSignature) : cliffordDim sig > 0 :=
  pow_pos 2 (sig.p + sig.q + sig.r) (Nat.zero_lt_two)

-- ============================================================
--  2. Multivector and arithmetic
-- ============================================================

def Multivector (sig : CliffordSignature) : Type :=
  Fin (cliffordDim sig) → Float

def scalarPart {sig : CliffordSignature} (x : Multivector sig) : Float :=
  x ⟨0, cliffordDim_pos sig⟩

def mvAdd {sig : CliffordSignature} (a b : Multivector sig) : Multivector sig :=
  fun i => a i + b i

def mvScale {sig : CliffordSignature} (α : Float) (a : Multivector sig) : Multivector sig :=
  fun i => α * a i

def mvSub {sig : CliffordSignature} (a b : Multivector sig) : Multivector sig :=
  fun i => a i - b i

-- ============================================================
--  3. Core Clifford algebra operations (axiomatic)
-- ============================================================

class CliffordAlgebra (sig : CliffordSignature) where
  dim : Nat := cliffordDim sig
  signature : CliffordSignature := sig
  geometricProduct : Multivector sig → Multivector sig → Multivector sig
  reverse : Multivector sig → Multivector sig
  norm : Multivector sig → Float

export CliffordAlgebra (geometricProduct reverse norm)

def geom_prod {sig : CliffordSignature} [CliffordAlgebra sig]
    (a b : Multivector sig) : Multivector sig :=
  geometricProduct a b

def scalarOne {sig : CliffordSignature} : Multivector sig :=
  fun i => if i.val == 0 then 1.0 else 0.0

axiom geom_prod_assoc {sig : CliffordSignature} [CliffordAlgebra sig] :
  ∀ (a b c : Multivector sig),
    geometricProduct a (geometricProduct b c) = geometricProduct (geometricProduct a b) c

axiom geom_prod_one_left {sig : CliffordSignature} [CliffordAlgebra sig] :
  ∀ (a : Multivector sig), geometricProduct scalarOne a = a

axiom geom_prod_one_right {sig : CliffordSignature} [CliffordAlgebra sig] :
  ∀ (a : Multivector sig), geometricProduct a scalarOne = a

axiom reverse_involutive {sig : CliffordSignature} [CliffordAlgebra sig] :
  ∀ (a : Multivector sig), reverse (reverse a) = a

axiom reverse_mul {sig : CliffordSignature} [CliffordAlgebra sig] :
  ∀ (a b : Multivector sig),
    reverse (geometricProduct a b) = geometricProduct (reverse b) (reverse a)

axiom reverse_scalarOne {sig : CliffordSignature} [CliffordAlgebra sig] :
  reverse (scalarOne : Multivector sig) = scalarOne

axiom norm_scalarOne {sig : CliffordSignature} [CliffordAlgebra sig] :
  norm (scalarOne : Multivector sig) = 1.0

-- Compatibility aliases retained for existing formal statements.
class HasReverse (α : Type) where
  reverse : α → α

def cliffordNorm {sig : CliffordSignature} [CliffordAlgebra sig]
    (x : Multivector sig) : Float :=
  norm x

-- ============================================================
--  4. Sandwich, domain rotors, and invariants
-- ============================================================

def sandwich {sig : CliffordSignature} [CliffordAlgebra sig]
    (R G : Multivector sig) : Multivector sig :=
  geometricProduct (geometricProduct R G) (reverse R)

structure DomainRotor (sig : CliffordSignature) [CliffordAlgebra sig] where
  R : Multivector sig
  h_unit : norm R = 1.0

def identity {sig : CliffordSignature} [CliffordAlgebra sig] : DomainRotor sig where
  R := scalarOne
  h_unit := norm_scalarOne

def rotorInverse {sig : CliffordSignature} [CliffordAlgebra sig]
    (R : DomainRotor sig) : Multivector sig :=
  reverse R.R

def extractInvariant {sig : CliffordSignature} [CliffordAlgebra sig]
    (R : DomainRotor sig) (G : Multivector sig) : Multivector sig :=
  sandwich (rotorInverse R) G

def domainTransfer {sig : CliffordSignature} [CliffordAlgebra sig]
    (R : DomainRotor sig) (U : Multivector sig) : Multivector sig :=
  sandwich R.R U

def sameStructure {sig : CliffordSignature} [CliffordAlgebra sig]
    (G1 : Multivector sig) (R1 : DomainRotor sig)
    (G2 : Multivector sig) (R2 : DomainRotor sig) : Prop :=
  extractInvariant R1 G1 = extractInvariant R2 G2

def analogyMatch {sig : CliffordSignature} (U V : Multivector sig) : Prop :=
  U = V

axiom extract_transfer_roundtrip_axiom {sig : CliffordSignature} [CliffordAlgebra sig] :
  ∀ (R : DomainRotor sig) (U : Multivector sig),
    extractInvariant R (domainTransfer R U) = U

axiom norm_extract_preservation_axiom {sig : CliffordSignature} [CliffordAlgebra sig] :
  ∀ (R : DomainRotor sig) (G : Multivector sig),
    norm (extractInvariant R G) = norm G

-- ============================================================
--  5. Core theorems
-- ============================================================

theorem identity_extraction {sig : CliffordSignature} [CliffordAlgebra sig]
    (G : Multivector sig) :
    extractInvariant (identity : DomainRotor sig) G = G := by
  unfold extractInvariant rotorInverse identity sandwich
  rw [reverse_scalarOne]
  rw [geom_prod_one_left]
  rw [reverse_scalarOne]
  exact geom_prod_one_right G

theorem transfer_roundtrip {sig : CliffordSignature} [CliffordAlgebra sig]
    (R : DomainRotor sig) (U : Multivector sig) :
    extractInvariant R (domainTransfer R U) = U := by
  exact extract_transfer_roundtrip_axiom R U

theorem cross_domain_invariance {sig : CliffordSignature} [CliffordAlgebra sig]
    (G1 : Multivector sig) (R1 : DomainRotor sig)
    (G2 : Multivector sig) (R2 : DomainRotor sig)
    (U : Multivector sig)
    (h1 : G1 = domainTransfer R1 U)
    (h2 : G2 = domainTransfer R2 U) :
    extractInvariant R1 G1 = extractInvariant R2 G2 := by
  rw [h1, h2, transfer_roundtrip R1 U, transfer_roundtrip R2 U]

theorem norm_preservation_unit_rotor {sig : CliffordSignature} [CliffordAlgebra sig]
    (R : DomainRotor sig) (G : Multivector sig) :
    norm (extractInvariant R G) = norm G := by
  exact norm_extract_preservation_axiom R G

theorem analogy_equivalence_refl {sig : CliffordSignature}
    (U : Multivector sig) : analogyMatch U U := by
  rfl

theorem analogy_equivalence_symm {sig : CliffordSignature}
    (U V : Multivector sig) : analogyMatch U V → analogyMatch V U := by
  intro h
  exact h.symm

theorem analogy_equivalence_trans {sig : CliffordSignature}
    (U V W : Multivector sig) :
    analogyMatch U V → analogyMatch V W → analogyMatch U W := by
  intro huv hvw
  exact huv.trans hvw
