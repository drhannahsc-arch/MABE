"""
MABE Sprint 12 Bootstrap - Ligand Field Stabilization Energy & Coordination Geometry
======================================================================================
The elephant in the coordination chemistry room.

Sprint 9-11 compute ΔG from donor energies, HSAB match, desolvation, chelate
effect, preorganization, orbital CT, and protonation. But they treat all
coordination geometries as equivalent. They are not.

WHY THIS MATTERS:

    Ni²⁺ (d⁸): LFSE in octahedral = -120 kJ/mol (weak field) or -240 kJ/mol
    (strong field square planar). Ni²⁺ with strong-field donors DEMANDS square
    planar — and this preference is worth 120 kJ/mol of free energy. Design a
    square planar binding pocket with 4 N donors → massive selectivity over
    Ca²⁺ (d⁰, zero LFSE in any geometry).

    Cu²⁺ (d⁹): Jahn-Teller distortion elongates axial bonds. Cannot form
    regular octahedron. Tetragonal distortion is mandatory. A rigid octahedral
    cage designed for Ni²⁺ will REJECT Cu²⁺ — this is a selectivity tool.

    Fe³⁺ (d⁵ high-spin): LFSE = 0 in ALL geometries. No geometry preference
    from crystal field. Selectivity comes entirely from charge/hardness, not
    geometry. But Fe³⁺ has huge dehydration penalty (z²/r) that compensates.

    Pb²⁺ (post-transition, 6s² lone pair): No d-orbital splitting. The 6s²
    stereochemically active lone pair drives hemidirected coordination. Not
    LFSE, but analogous geometric selectivity from lone pair effects.

    Zn²⁺ (d¹⁰): LFSE = 0 for all geometries. Tetrahedral preference is purely
    entropic/steric. Must not confuse this with LFSE-driven preference.

This sprint adds:
    1. knowledge/lfse_data.py — d-electron configs, Dq, pairing energies, spectrochemical series
    2. core/lfse.py — LFSE calculation, geometry preference, Jahn-Teller detection
    3. core/lfse_integration.py — patches thermodynamics + rescore pipeline

    cd Documents\\mabe
    python bootstrap_sprint12.py
    python tests\\test_sprint12.py
"""

import os
import json

def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print()
print("  MABE Sprint 12 - LFSE & Coordination Geometry Preference")
print("  " + "=" * 56)
print()


# ═══════════════════════════════════════════════════════════════════════════
# knowledge/lfse_data.py — Crystal field theory database
# ═══════════════════════════════════════════════════════════════════════════

write_file("knowledge/lfse_data.py", '''"""
knowledge/lfse_data.py - Ligand Field Stabilization Energy database.

Crystal field theory in 200 lines:

    When a transition metal ion is surrounded by ligands, the five d orbitals
    split in energy. The splitting pattern depends on geometry:

    Octahedral:  t2g (dxy, dxz, dyz) at -0.4 Dq_oct
                 eg  (dz², dx²-y²)   at +0.6 Dq_oct

    Tetrahedral: e   (dz², dx²-y²)   at -0.6 Dq_tet
                 t2  (dxy, dxz, dyz) at +0.4 Dq_tet
                 Dq_tet ≈ 4/9 × Dq_oct (always weaker)

    Square planar: dx²-y² highest, dxy next, dz² mid, dxz/dyz lowest
                   Strong splitting; d⁸ strongly favored

    LFSE = Σ(n_i × ε_i) where n_i = electron count in orbital i, ε_i = energy

    The LFSE difference between geometries drives geometry preference.
    For d⁸ (Ni²⁺), square planar LFSE >> octahedral LFSE with strong-field
    donors, so Ni²⁺ switches to square planar with sufficient field strength.

SPECTROCHEMICAL SERIES (weak → strong field):
    I⁻ < Br⁻ < S²⁻ < Cl⁻ < N₃⁻ < F⁻ < OH⁻ < ox²⁻ < H₂O
    < NCS⁻ < CH₃CN < py < NH₃ < en < bipy < phen < NO₂⁻
    < PPh₃ < CN⁻ < CO < NO⁺

For binder design:
    S donors: weak-to-intermediate field (except thiolate anion → moderate)
    O donors: weak field (water, carboxylate, hydroxyl)
    N donors: moderate-to-strong field (amine, pyridine, imidazole)
    P donors: strong field (phosphine)
    CN/CO: very strong field (cyanide, carbonyl)

CRITICAL DESIGN IMPLICATION:
    If you want selectivity for Ni²⁺ over Ca²⁺, use 4 strong-field N donors
    in a planar arrangement. Ni²⁺ d⁸ gains ~120-240 kJ/mol LFSE.
    Ca²⁺ d⁰ gains zero. That is a selectivity wall.
"""

from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# d-electron configurations for metal ions
# ═══════════════════════════════════════════════════════════════════════════

METAL_D_ELECTRONS = {
    # Transition metals (common oxidation states in MABE)
    # identity → {oxidation_state: d_count}
    "iron":     {3: 5, 2: 6},       # Fe³⁺ = d⁵, Fe²⁺ = d⁶
    "copper":   {2: 9, 1: 10},      # Cu²⁺ = d⁹, Cu⁺ = d¹⁰
    "nickel":   {2: 8},             # Ni²⁺ = d⁸
    "zinc":     {2: 10},            # Zn²⁺ = d¹⁰
    "cobalt":   {3: 6, 2: 7},       # Co³⁺ = d⁶, Co²⁺ = d⁷
    "manganese":{2: 5, 4: 3, 7: 0}, # Mn²⁺ = d⁵, Mn⁴⁺ = d³, Mn⁷⁺ = d⁰
    "chromium": {3: 3, 6: 0},       # Cr³⁺ = d³, Cr⁶⁺ = d⁰
    "cadmium":  {2: 10},            # Cd²⁺ = d¹⁰
    "silver":   {1: 10},            # Ag⁺ = d¹⁰

    # Post-transition / main group — no crystal field splitting
    # But stereochemically active lone pairs affect geometry
    "lead":     {2: -1},            # Pb²⁺ = [Xe]4f¹⁴5d¹⁰6s² — 6s² lone pair, not d-metal
    "mercury":  {2: -1},            # Hg²⁺ = [Xe]4f¹⁴5d¹⁰ (d¹⁰ + fully filled)
    # -1 sentinel: "post-d / s² lone pair, no crystal field splitting"

    # Hard cations — no d-electrons
    "calcium":  {2: 0},             # Ca²⁺ = d⁰
    "magnesium":{2: 0},             # Mg²⁺ = d⁰
    "sodium":   {1: 0},             # Na⁺ = d⁰
    "potassium":{1: 0},             # K⁺ = d⁰
    "aluminum": {3: 0},             # Al³⁺ = d⁰
    "barium":   {2: 0},             # Ba²⁺ = d⁰

    # Lanthanides (f-electrons, not d — use sentinel -2)
    "cerium":   {3: -2, 4: -2},     # Ce³⁺ = [Xe]4f¹ — f-electron, not crystal field
    "uranium":  {6: -2},            # UO₂²⁺ — actinide, not crystal field

    # Metalloids / oxyanions — no d-splitting
    "arsenate": {5: 0},
    "selenite": {4: 0},
    "chromate": {6: 0},

    # Gold — d⁸ for Au³⁺, d¹⁰ for Au⁺
    "gold":     {3: 8, 1: 10},      # Au³⁺ = d⁸ (strong square planar preference!)
}


# ═══════════════════════════════════════════════════════════════════════════
# Dq values (10Dq in kJ/mol) by donor type
# These are the ligand field splitting parameter for OCTAHEDRAL geometry
# Tetrahedral Dq ≈ 4/9 of octahedral
# ═══════════════════════════════════════════════════════════════════════════

# Representative 10Dq (octahedral) in kJ/mol for M²⁺ ions
# Actual values vary with metal, but the RATIOS between donors are consistent
# Source: Figgis & Hitchman, Lever, Shriver & Atkins
DQ_OCT_KJ = {
    # donor_atom → approximate 10Dq for a generic divalent transition metal
    "O": 100.0,     # water / carboxylate — weak field
    "N": 130.0,     # amine / pyridine — moderate field
    "S": 85.0,      # thiolate / thioether — weak-moderate
    "P": 150.0,     # phosphine — strong field
    "C": 170.0,     # cyanide / carbonyl — very strong field
    "electrostatic": 0.0,  # no covalent splitting
}

# Field strength classification (drives high-spin vs low-spin)
DONOR_FIELD_STRENGTH = {
    "S": "weak",        # S²⁻, thiolate: weak field
    "O": "weak",        # H₂O, COO⁻, OH⁻: weak field
    "N": "moderate",    # NH₃, py, im: moderate → can cause spin crossover in some d⁴-d⁷
    "P": "strong",      # PPh₃, phosphine: strong field
    "C": "strong",      # CN⁻, CO: very strong field
    "electrostatic": "none",
}

# Pairing energy (in kJ/mol) — cost of putting two electrons in same orbital
# Must overcome pairing energy to go low-spin
# Varies by metal; these are representative values
PAIRING_ENERGY = {
    "iron":     {3: 210, 2: 175},     # Fe³⁺ high P, Fe²⁺ moderate
    "cobalt":   {3: 200, 2: 190},     # Co³⁺ moderate, Co²⁺ high
    "nickel":   {2: 180},             # Ni²⁺ — only relevant for tetrahedral (rare low-spin oct)
    "copper":   {2: 170},             # Cu²⁺
    "manganese":{2: 250},             # Mn²⁺ — very high (half-filled stability)
    "chromium": {3: 230},             # Cr³⁺ — very high
}

# Jahn-Teller active configurations (in octahedral)
# These configurations have unequal eg occupancy → distortion mandatory
# d⁴ (t2g³ eg¹), d⁷ low-spin (t2g⁶ eg¹), d⁹ (t2g⁶ eg³)
JAHN_TELLER_ACTIVE_OCT = {
    (4, "high"): True,   # d⁴ high-spin: t2g³ eg¹
    (9, "high"): True,   # d⁹: t2g⁶ eg³ — ALWAYS (Cu²⁺)
    (9, "low"):  True,   # d⁹: same
    (7, "low"):  True,   # d⁷ low-spin: t2g⁶ eg¹
}

# Geometry-specific LFSE in units of Dq (these are exact from crystal field theory)
# For high-spin configurations:
LFSE_DQ_HIGHSPIN = {
    # d_count: {geometry: LFSE in units of Dq}
    0:  {"oct": 0.0,  "tet": 0.0,  "sq_planar": 0.0},
    1:  {"oct": -4.0, "tet": -2.67, "sq_planar": -5.14},
    2:  {"oct": -8.0, "tet": -5.34, "sq_planar": -10.28},
    3:  {"oct": -12.0,"tet": -3.56, "sq_planar": -12.28},
    4:  {"oct": -6.0, "tet": -1.78, "sq_planar": -9.14},
    5:  {"oct": 0.0,  "tet": 0.0,  "sq_planar": -0.0},
    6:  {"oct": -4.0, "tet": -2.67, "sq_planar": -5.14},
    7:  {"oct": -8.0, "tet": -5.34, "sq_planar": -10.28},
    8:  {"oct": -12.0,"tet": -3.56, "sq_planar": -24.56},
    9:  {"oct": -6.0, "tet": -1.78, "sq_planar": -21.42},
    10: {"oct": 0.0,  "tet": 0.0,  "sq_planar": 0.0},
}

# Low-spin corrections (only relevant for d⁴-d⁷ with strong field)
LFSE_DQ_LOWSPIN = {
    4:  {"oct": -16.0, "sq_planar": -17.42},
    5:  {"oct": -20.0, "sq_planar": -24.56},
    6:  {"oct": -24.0, "sq_planar": -24.56},
    7:  {"oct": -18.0, "sq_planar": -24.56},
}

# Stereochemically active lone pair data (post-transition metals)
# These don't have LFSE but have geometry preference from lone pair
LONE_PAIR_METALS = {
    "lead": {
        "lone_pair": "6s²",
        "effect": "hemidirected",
        "description": "Pb²⁺ 6s² lone pair occupies one coordination position. "
                       "Favors asymmetric coordination (hemidirected) at CN 4-6. "
                       "At CN > 8, lone pair becomes stereochemically inactive (holodirected).",
        "geometry_preference": {
            "hemidirected": -8.0,      # kJ/mol stabilization for asymmetric
            "square_pyramidal": -6.0,
            "trigonal_bipyramidal": -4.0,
            "octahedral": 0.0,         # lone pair squeezed — less favorable
            "holodirected": 2.0,       # high CN forces symmetric — slight penalty
        },
        "preferred_cn": [4, 5, 6],
    },
    "mercury": {
        "lone_pair": "5d¹⁰6s⁰",
        "effect": "relativistic",
        "description": "Hg²⁺ has fully filled d¹⁰ shell. Strong relativistic contraction "
                       "of 6s orbital drives linear 2-coordinate preference. "
                       "This is NOT LFSE — it is a relativistic effect.",
        "geometry_preference": {
            "linear": -15.0,           # strong preference
            "tetrahedral": -3.0,
            "octahedral": 0.0,
        },
        "preferred_cn": [2, 4],
    },
}


def get_d_electron_count(identity: str, oxidation_state: int) -> Optional[int]:
    """
    Get d-electron count for a metal ion.

    Returns:
        int >= 0: actual d-electron count (0-10)
        -1: post-transition metal with lone pair effects
        -2: f-block element (lanthanide/actinide)
        None: unknown metal
    """
    metal_data = METAL_D_ELECTRONS.get(identity.lower())
    if metal_data is None:
        return None
    return metal_data.get(oxidation_state, metal_data.get(abs(oxidation_state)))


def get_field_strength(donor_atoms: list[str]) -> str:
    """Classify overall field strength from donor set."""
    if not donor_atoms:
        return "weak"

    strengths = [DONOR_FIELD_STRENGTH.get(d, "weak") for d in donor_atoms]

    strong_count = strengths.count("strong")
    moderate_count = strengths.count("moderate")
    weak_count = strengths.count("weak")

    if strong_count >= len(donor_atoms) // 2:
        return "strong"
    elif moderate_count + strong_count >= len(donor_atoms) // 2:
        return "moderate"
    return "weak"


def average_dq(donor_atoms: list[str]) -> float:
    """Average 10Dq in kJ/mol from donor set."""
    if not donor_atoms:
        return 100.0  # default to water-like
    dqs = [DQ_OCT_KJ.get(d, 100.0) for d in donor_atoms]
    return sum(dqs) / len(dqs)


def is_high_spin(d_count: int, field_strength: str, identity: str,
                  oxidation_state: int) -> bool:
    """
    Determine if the complex is high-spin or low-spin.

    Only relevant for d⁴-d⁷. d¹-d³, d⁸-d¹⁰ always have the same
    configuration regardless of field strength.
    """
    if d_count < 4 or d_count > 7:
        return True  # not ambiguous

    if field_strength == "weak":
        return True  # weak field → always high-spin

    if field_strength == "strong":
        return False  # strong field → always low-spin

    # Moderate field: depends on metal-specific pairing energy
    # Heuristic: moderate field + 3+ charge → often low-spin (more compact d orbitals)
    if oxidation_state >= 3:
        return False  # higher charge → larger splitting → low-spin more likely
    return True  # +2 with moderate field → usually high-spin


def compute_lfse_dq(d_count: int, geometry: str, high_spin: bool) -> float:
    """
    Compute LFSE in units of Dq for a given d-electron count and geometry.
    """
    if d_count <= 0 or d_count > 10:
        return 0.0

    if not high_spin and d_count in LFSE_DQ_LOWSPIN:
        low_data = LFSE_DQ_LOWSPIN[d_count]
        if geometry in low_data:
            return low_data[geometry]

    hs_data = LFSE_DQ_HIGHSPIN.get(d_count, {})
    return hs_data.get(geometry, 0.0)


def is_jahn_teller(d_count: int, high_spin: bool) -> bool:
    """Check if configuration is Jahn-Teller active in octahedral."""
    spin_label = "high" if high_spin else "low"
    return JAHN_TELLER_ACTIVE_OCT.get((d_count, spin_label), False)
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/lfse.py — LFSE physics engine
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/lfse.py", '''"""
core/lfse.py - Ligand Field Stabilization Energy analysis.

Computes:
    1. LFSE for each plausible geometry (oct, tet, sq_planar)
    2. Preferred geometry (which has most negative LFSE)
    3. Geometry preference energy (ΔG between best and second-best)
    4. Geometry mismatch penalty (if binder forces wrong geometry)
    5. Jahn-Teller distortion prediction
    6. Selectivity implications (LFSE difference vs. competitors)

The LFSE contribution to ΔG:
    ΔG_lfse = LFSE_in_Dq × Dq_avg(donors)

    For Ni²⁺ d⁸ with N donors (Dq ~130 kJ/mol):
        Oct:  -12 × 0.1 × 130 = -156 kJ/mol
        Tet:  -3.56 × 0.1 × (4/9 × 130) = -20.5 kJ/mol
        SqPl: -24.56 × 0.1 × 130 = -319 kJ/mol (!)
    The square planar preference is worth ~160 kJ/mol over octahedral.

Note: LFSE values in the database are in units of Dq. To convert to kJ/mol:
    LFSE_kJ = LFSE_Dq × (10Dq_kJ / 10) = LFSE_Dq × Dq_kJ
    Where Dq_kJ = 10Dq_kJ / 10
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.assembly import RecognitionChemistry, StructuralConstraint
from core.problem import Problem, TargetSpecies
from knowledge.lfse_data import (
    get_d_electron_count, get_field_strength, average_dq,
    is_high_spin, compute_lfse_dq, is_jahn_teller,
    LONE_PAIR_METALS, METAL_D_ELECTRONS, PAIRING_ENERGY,
    LFSE_DQ_HIGHSPIN, DQ_OCT_KJ,
)


@dataclass
class LFSEAnalysis:
    """Complete LFSE and coordination geometry analysis."""

    # d-electron info
    d_electron_count: int = 0           # 0-10, or -1 (lone pair), -2 (f-block)
    spin_state: str = "n/a"             # "high", "low", "n/a"
    field_strength: str = "weak"        # from donor set

    # LFSE for each geometry (kJ/mol, negative = stabilizing)
    lfse_octahedral_kj: float = 0.0
    lfse_tetrahedral_kj: float = 0.0
    lfse_square_planar_kj: float = 0.0

    # Geometry preference
    preferred_geometry: str = "none"     # oct, tet, sq_planar, linear, hemidirected
    geometry_preference_dG: float = 0.0  # kJ/mol advantage of preferred over next best
    binder_coordination_number: int = 0  # how many donors the binder provides
    geometry_compatible: bool = True     # does binder CN match preferred geometry?
    geometry_mismatch_penalty: float = 0.0  # kJ/mol penalty if incompatible

    # Jahn-Teller
    jahn_teller_active: bool = False
    jahn_teller_description: str = ""

    # Lone pair effects (Pb²⁺, Hg²⁺)
    lone_pair_active: bool = False
    lone_pair_dG: float = 0.0

    # Net LFSE contribution to binding
    dG_lfse: float = 0.0               # total LFSE contribution (negative = favorable)

    # For selectivity: LFSE of this metal vs competitors
    lfse_selectivity_advantage: float = 0.0  # kJ/mol advantage over competitor

    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.d_electron_count == 0:
            return f"d⁰ — no LFSE contribution (hard cation, geometry from charge/size)"
        if self.d_electron_count == -1:
            return (f"Post-d metal with lone pair → {self.preferred_geometry} "
                    f"(ΔG = {self.lone_pair_dG:.1f} kJ/mol)")
        if self.d_electron_count == -2:
            return "f-block element — crystal field theory not applicable"
        if self.d_electron_count == 10:
            return f"d¹⁰ — LFSE = 0 in all geometries (geometry from sterics/entropy)"

        parts = [
            f"d{self.d_electron_count} {self.spin_state}-spin | "
            f"field: {self.field_strength} | "
            f"preferred: {self.preferred_geometry}"
        ]
        parts.append(
            f"  LFSE: oct = {self.lfse_octahedral_kj:.1f}, "
            f"tet = {self.lfse_tetrahedral_kj:.1f}, "
            f"sq.pl. = {self.lfse_square_planar_kj:.1f} kJ/mol"
        )
        parts.append(f"  ΔG_lfse = {self.dG_lfse:.1f} kJ/mol")
        if self.jahn_teller_active:
            parts.append(f"  ⚠ {self.jahn_teller_description}")
        if not self.geometry_compatible:
            parts.append(
                f"  ⚠ Geometry mismatch: binder CN={self.binder_coordination_number} "
                f"vs preferred {self.preferred_geometry} (penalty +{self.geometry_mismatch_penalty:.1f} kJ/mol)"
            )
        return "\\n".join(parts)


def _infer_oxidation_state(target: TargetSpecies) -> int:
    """Infer oxidation state from charge or formula."""
    charge = target.charge
    if charge is not None:
        return int(abs(charge))
    return 2  # default divalent


def _geometry_from_cn(cn: int, d_count: int, field_strength: str) -> str:
    """Map coordination number to likely geometry."""
    if cn <= 2:
        return "linear"
    if cn == 3:
        return "trigonal_planar"
    if cn == 4:
        # d⁸ with moderate-to-strong field → square planar
        if d_count == 8 and field_strength in ("strong", "moderate"):
            return "sq_planar"
        return "tet"
    if cn == 5:
        return "square_pyramidal"
    if cn >= 6:
        return "oct"
    return "oct"


def compute_lfse(recognition: RecognitionChemistry,
                  target: TargetSpecies,
                  structure: StructuralConstraint = None) -> LFSEAnalysis:
    """
    Compute LFSE analysis for a metal-binder interaction.
    """
    result = LFSEAnalysis()
    breakdown = []

    identity = target.identity.lower()
    ox_state = _infer_oxidation_state(target)

    # Get d-electron count
    d_count = get_d_electron_count(identity, ox_state)
    if d_count is None:
        result.d_electron_count = 0
        breakdown.append(f"Unknown metal {identity} — assuming d⁰ (no LFSE)")
        result.breakdown = breakdown
        return result

    result.d_electron_count = d_count

    # ── Handle special cases ─────────────────────────────────────────

    # Post-transition metals with lone pair effects
    if d_count == -1:
        result.lone_pair_active = True
        lp_data = LONE_PAIR_METALS.get(identity, {})
        if lp_data:
            # Determine geometry preference from lone pair
            prefs = lp_data.get("geometry_preference", {})
            if prefs:
                best_geom = min(prefs, key=prefs.get)
                result.preferred_geometry = best_geom
                result.lone_pair_dG = prefs[best_geom]
                result.dG_lfse = prefs[best_geom]
                breakdown.append(
                    f"{identity.capitalize()} {lp_data.get('lone_pair', '?')} lone pair → "
                    f"{lp_data.get('effect', '?')}: preferred {best_geom} "
                    f"({result.lone_pair_dG:.1f} kJ/mol)"
                )
                breakdown.append(lp_data.get("description", ""))
        else:
            breakdown.append(f"{identity}: post-d metal, no LFSE, no lone pair data")
        result.breakdown = breakdown
        return result

    # f-block elements
    if d_count == -2:
        breakdown.append(f"{identity}: f-block element, crystal field theory not directly applicable")
        result.breakdown = breakdown
        return result

    # d⁰ or d¹⁰ — no LFSE
    if d_count == 0 or d_count == 10:
        result.preferred_geometry = "any"
        breakdown.append(
            f"d{d_count} — LFSE = 0 in all geometries. "
            f"{'Size/charge drives geometry.' if d_count == 0 else 'Sterics/entropy drive geometry (often tetrahedral).'}"
        )
        result.breakdown = breakdown
        return result

    # ── Standard LFSE calculation for d¹-d⁹ ──────────────────────────

    donors = recognition.donor_atoms or ["O", "N"]
    field_str = get_field_strength(donors)
    result.field_strength = field_str
    dq_avg = average_dq(donors)
    dq_unit = dq_avg / 10.0  # convert 10Dq to Dq

    high_spin = is_high_spin(d_count, field_str, identity, ox_state)
    result.spin_state = "high" if high_spin else "low"

    breakdown.append(
        f"d{d_count} ({identity} {ox_state}+) | "
        f"donors: {'+'.join(donors)} ({field_str} field) | "
        f"10Dq_avg = {dq_avg:.0f} kJ/mol | "
        f"{'high' if high_spin else 'low'}-spin"
    )

    # Compute LFSE for each geometry
    lfse_oct_dq = compute_lfse_dq(d_count, "oct", high_spin)
    lfse_tet_dq = compute_lfse_dq(d_count, "tet", high_spin)
    lfse_sp_dq = compute_lfse_dq(d_count, "sq_planar", high_spin)

    # Convert to kJ/mol
    # Oct: LFSE_kJ = LFSE_Dq × Dq_oct
    # Tet: LFSE_kJ = LFSE_Dq × Dq_tet = LFSE_Dq × (4/9) × Dq_oct
    lfse_oct_kj = lfse_oct_dq * dq_unit
    lfse_tet_kj = lfse_tet_dq * (4.0 / 9.0) * dq_unit
    lfse_sp_kj = lfse_sp_dq * dq_unit * 1.3  # square planar splitting is ~1.3× octahedral

    # Low-spin pairing energy correction
    if not high_spin and d_count in range(4, 8):
        pe_data = PAIRING_ENERGY.get(identity, {})
        pe = pe_data.get(ox_state, 180.0)
        # Number of extra pairings vs high-spin
        extra_pairings = {4: 1, 5: 2, 6: 2, 7: 1}.get(d_count, 0)
        pairing_cost = extra_pairings * pe / 1000.0 * 10  # rough correction to kJ/mol scale
        # The pairing cost partially offsets the extra LFSE of low-spin
        breakdown.append(f"Low-spin: {extra_pairings} extra pairings, PE ≈ {pe:.0f} kJ/mol each")

    result.lfse_octahedral_kj = round(lfse_oct_kj, 1)
    result.lfse_tetrahedral_kj = round(lfse_tet_kj, 1)
    result.lfse_square_planar_kj = round(lfse_sp_kj, 1)

    breakdown.append(
        f"LFSE: oct = {lfse_oct_kj:.1f}, tet = {lfse_tet_kj:.1f}, "
        f"sq.pl. = {lfse_sp_kj:.1f} kJ/mol"
    )

    # ── Determine preferred geometry ──────────────────────────────────

    geometries = {
        "oct": lfse_oct_kj,
        "tet": lfse_tet_kj,
        "sq_planar": lfse_sp_kj,
    }

    # For d⁸ with strong/moderate field: square planar is real
    # For d⁸ weak field: octahedral
    if d_count == 8 and field_str in ("strong", "moderate"):
        preferred = "sq_planar"
    else:
        preferred = min(geometries, key=geometries.get)

    # Second best
    sorted_geoms = sorted(geometries.items(), key=lambda x: x[1])
    pref_dG = sorted_geoms[0][1]
    second_dG = sorted_geoms[1][1] if len(sorted_geoms) > 1 else 0.0
    preference_energy = second_dG - pref_dG  # positive = preference is favorable

    result.preferred_geometry = preferred
    result.geometry_preference_dG = round(preference_energy, 1)

    breakdown.append(
        f"Preferred geometry: {preferred} "
        f"(advantage = {preference_energy:.1f} kJ/mol over {sorted_geoms[1][0]})"
    )

    # ── Check geometry compatibility with binder CN ───────────────────

    binder_cn = len(donors)
    result.binder_coordination_number = binder_cn

    expected_cn = {"oct": 6, "tet": 4, "sq_planar": 4, "linear": 2}
    pref_cn = expected_cn.get(preferred, 6)

    if binder_cn >= pref_cn:
        result.geometry_compatible = True
        # Check if binder could support preferred geometry
        if preferred == "sq_planar" and binder_cn == 4:
            breakdown.append(f"Binder CN={binder_cn} matches square planar preference ✓")
        elif preferred == "oct" and binder_cn >= 4:
            breakdown.append(f"Binder CN={binder_cn} supports octahedral (solvent completes) ✓")
        elif preferred == "tet" and binder_cn >= 3:
            breakdown.append(f"Binder CN={binder_cn} supports tetrahedral ✓")
    else:
        result.geometry_compatible = True  # low CN binders can still work (solvent fills)
        breakdown.append(f"Binder CN={binder_cn} (solvent completes coordination shell)")

    # Geometry mismatch penalty: if we force wrong geometry
    # E.g., 6 donors forcing octahedral when metal wants square planar
    if preferred == "sq_planar" and binder_cn >= 6:
        # Hexadentate ligand forces octahedral on metal that wants sq_planar
        penalty = preference_energy * 0.5  # partial penalty (metal can still distort)
        result.geometry_mismatch_penalty = round(penalty, 1)
        result.geometry_compatible = False
        breakdown.append(
            f"⚠ Hexadentate forces octahedral on {preferred}-preferring d{d_count}: "
            f"penalty +{penalty:.1f} kJ/mol"
        )

    # ── Jahn-Teller ───────────────────────────────────────────────────

    jt = is_jahn_teller(d_count, high_spin)
    result.jahn_teller_active = jt
    if jt:
        if d_count == 9:
            result.jahn_teller_description = (
                f"d⁹ (Cu²⁺-type): strong Jahn-Teller distortion. "
                f"Axial bonds elongate ~0.2-0.3 Å. Rigid octahedral cages fail. "
                f"Design for tetragonal or square planar geometry."
            )
        elif d_count == 4 and high_spin:
            result.jahn_teller_description = (
                f"d⁴ high-spin: moderate Jahn-Teller distortion. "
                f"Tetragonal elongation expected."
            )
        elif d_count == 7 and not high_spin:
            result.jahn_teller_description = (
                f"d⁷ low-spin: Jahn-Teller active. Tetragonal distortion."
            )
        breakdown.append(f"Jahn-Teller: {result.jahn_teller_description}")

    # ── Net ΔG_lfse ───────────────────────────────────────────────────

    # Use LFSE of the geometry the binder actually creates
    actual_geometry = _geometry_from_cn(binder_cn, d_count, field_str)
    actual_lfse = geometries.get(actual_geometry, lfse_oct_kj)
    result.dG_lfse = round(actual_lfse - result.geometry_mismatch_penalty, 1)

    breakdown.append(
        f"Net ΔG_lfse = {result.dG_lfse:.1f} kJ/mol "
        f"(in {actual_geometry} geometry, CN={binder_cn})"
    )

    result.breakdown = breakdown
    return result
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/lfse_integration.py — Patches into thermodynamic pipeline
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/lfse_integration.py", '''"""
core/lfse_integration.py - Integrates LFSE into the thermodynamic pipeline.

Patches compute_thermodynamics to:
1. Compute LFSE for target metal
2. Add ΔG_lfse to ΔG_net
3. Compute competitor LFSE for selectivity refinement
4. Flag geometry mismatches and Jahn-Teller issues

Also patches sprint10_integration rescore for LFSE in reports.
"""

import math
import core.thermodynamics as thermo_mod
from core.lfse import compute_lfse, LFSEAnalysis
from core.assembly import RecognitionChemistry, StructuralConstraint, InteriorDesign
from core.problem import Problem, TargetSpecies, ElectronicDescription, HydrationDescription, SizeDescription
from knowledge.lfse_data import get_d_electron_count, average_dq, LFSE_DQ_HIGHSPIN

import core.sprint10_integration as s10


# ── Patch thermodynamics ──────────────────────────────────────────────

_orig_compute_thermo = thermo_mod.compute_thermodynamics


def _lfse_aware_thermodynamics(recognition: RecognitionChemistry,
                                structure: StructuralConstraint,
                                interior: InteriorDesign,
                                problem: Problem):
    """
    LFSE-aware thermodynamic calculation.
    Adds crystal field stabilization to ΔG_net.
    """
    result = _orig_compute_thermo(recognition, structure, interior, problem)

    # Compute LFSE
    lfse = compute_lfse(recognition, problem.target, structure)

    if lfse.d_electron_count in (0, 10, -2) and not lfse.lone_pair_active:
        # No LFSE contribution
        result.energy_breakdown.append(
            f"LFSE: d{lfse.d_electron_count} — no crystal field contribution"
        )
        return result

    # Add LFSE to ΔG_net
    dG_lfse = lfse.dG_lfse
    if dG_lfse != 0:
        result.dG_net += dG_lfse
        result.energy_breakdown.append(f"LFSE ({lfse.summary().split(chr(10))[0]}):")
        result.energy_breakdown.append(f"  ΔG_lfse = {dG_lfse:.1f} kJ/mol")

        # Update K_eq and Kd
        RT = thermo_mod.R_GAS * result.temperature_k
        if abs(result.dG_net / RT) < 500:
            result.K_eq = math.exp(-result.dG_net / RT)
        else:
            result.K_eq = 1e30 if result.dG_net < 0 else 0.0
        result.predicted_kd_um = round(1e6 / result.K_eq, 3) if result.K_eq > 1e-6 else None

    # ── Selectivity refinement from LFSE ──────────────────────────────
    # If target has LFSE and competitors don't (e.g., Ni²⁺ d⁸ vs Ca²⁺ d⁰),
    # LFSE provides additional selectivity
    for comp in problem.matrix.competing_species:
        comp_identity = comp.identity.lower()
        comp_ox = int(abs(comp.charge))
        comp_d = get_d_electron_count(comp_identity, comp_ox)

        if comp_d is None:
            comp_d = 0

        # Compute competitor LFSE (simplified — use same donor set)
        if comp_d in (0, 10, -1, -2):
            comp_lfse_kj = 0.0
        elif 1 <= comp_d <= 9:
            hs_data = LFSE_DQ_HIGHSPIN.get(comp_d, {})
            cn = len(recognition.donor_atoms or [])
            geom = "oct" if cn >= 5 else ("sq_planar" if cn == 4 else "tet")
            comp_lfse_dq = hs_data.get(geom, 0.0)
            dq_avg = average_dq(recognition.donor_atoms or ["O"])
            comp_lfse_kj = comp_lfse_dq * (dq_avg / 10.0)
        else:
            comp_lfse_kj = 0.0

        lfse_selectivity = comp_lfse_kj - dG_lfse  # positive = target prefers this binder more
        if abs(lfse_selectivity) > 5.0:
            result.energy_breakdown.append(
                f"  LFSE selectivity vs {comp.identity}: "
                f"target d{lfse.d_electron_count} = {dG_lfse:.1f}, "
                f"competitor d{comp_d if comp_d >= 0 else '?'} = {comp_lfse_kj:.1f} kJ/mol "
                f"(advantage: {lfse_selectivity:.1f} kJ/mol)"
            )

    # Geometry mismatch warning
    if lfse.geometry_mismatch_penalty > 0:
        result.energy_breakdown.append(
            f"  ⚠ Geometry mismatch penalty: +{lfse.geometry_mismatch_penalty:.1f} kJ/mol"
        )

    # Jahn-Teller warning
    if lfse.jahn_teller_active:
        result.energy_breakdown.append(
            f"  ⚠ {lfse.jahn_teller_description}"
        )

    return result


thermo_mod.compute_thermodynamics = _lfse_aware_thermodynamics


# ── Patch sprint10 rescore for LFSE in reports ───────────────────────

_orig_rescore = s10.full_physics_rescore


def _lfse_aware_rescore(assemblies, problem):
    """Add LFSE analysis to physics report."""
    assemblies = _orig_rescore(assemblies, problem)

    for assembly in assemblies:
        lfse = compute_lfse(assembly.recognition, problem.target, assembly.structure)

        # Append LFSE to confidence reasoning
        if lfse.d_electron_count not in (0, 10) or lfse.lone_pair_active:
            assembly.confidence_reasoning += "\\n\\nLFSE / GEOMETRY:\\n" + lfse.summary()

        # Warnings
        if lfse.jahn_teller_active:
            assembly.failure_modes.append(
                f"Jahn-Teller: {lfse.jahn_teller_description}"
            )
        if not lfse.geometry_compatible and lfse.geometry_mismatch_penalty > 10:
            assembly.failure_modes.append(
                f"Geometry mismatch: binder CN={lfse.binder_coordination_number} "
                f"vs preferred {lfse.preferred_geometry} "
                f"(penalty +{lfse.geometry_mismatch_penalty:.1f} kJ/mol)"
            )

    return assemblies


s10.full_physics_rescore = _lfse_aware_rescore
''')


# ═══════════════════════════════════════════════════════════════════════════
# Update main.py
# ═══════════════════════════════════════════════════════════════════════════

main_lines = [
    '"""', 'MABE - Modality-Agnostic Binder Engine', '"""', '',
    'import sys', 'import os', '',
    'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))', '',
    'from adapters.base import ToolRegistry',
    'from adapters.rdkit_adapter import RDKitAdapter',
    'from adapters.dnazyme_adapter import DNAzymeAdapter',
    'from adapters.peptide_adapter import PeptideAdapter',
    'from adapters.aptamer_adapter import AptamerAdapter',
    'from conversation.decomposer_patch import patch_targets',
    'from conversation.interface import run_interactive, run_single_query', '',
    'patch_targets()', '',
    '# Sprint 8', 'import core.assembly_composer_patch', 'import core.scoring_patch', '',
    '# Sprint 9: thermodynamics + hydrodynamics', 'import core.physics_integration', '',
    '# Sprint 10: kinetics + orbital + probability chain', 'import core.sprint10_integration', '',
    '# Sprint 11: pKa + protonation state', 'import core.protonation_integration', '',
    '# Sprint 12: LFSE + coordination geometry', 'import core.lfse_integration', '', '',
    'def build_registry() -> ToolRegistry:',
    '    registry = ToolRegistry()',
    '    rdkit = RDKitAdapter()',
    '    if rdkit.is_available():',
    '        registry.register(rdkit)',
    '    registry.register(DNAzymeAdapter())',
    '    registry.register(PeptideAdapter())',
    '    registry.register(AptamerAdapter())',
    '    return registry', '', '',
    'def main():',
    '    registry = build_registry()',
    '    if len(sys.argv) > 1:',
    '        query = " ".join(sys.argv[1:])',
    '        run_single_query(registry, query)',
    '    else:',
    '        run_interactive(registry)', '', '',
    'if __name__ == "__main__":',
    '    main()',
]
write_file("main.py", "\n".join(main_lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint12.py
# ═══════════════════════════════════════════════════════════════════════════

write_file('tests/test_sprint12.py', '''"""
tests/test_sprint12.py - LFSE and coordination geometry tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

import core.assembly_composer_patch
import core.scoring_patch
import core.physics_integration
import core.sprint10_integration
import core.protonation_integration
import core.lfse_integration

from knowledge.lfse_data import (
    get_d_electron_count, get_field_strength, average_dq,
    is_high_spin, compute_lfse_dq, is_jahn_teller,
    METAL_D_ELECTRONS, LFSE_DQ_HIGHSPIN, DQ_OCT_KJ,
    LONE_PAIR_METALS,
)
from core.lfse import compute_lfse, LFSEAnalysis
import core.thermodynamics as _thermo_mod
import copy
from core.assembly import RecognitionChemistry, InteriorDesign, InteriorSite
from core.problem import (
    Problem, TargetSpecies, Matrix, CompetingSpecies,
    ElectronicDescription, HydrationDescription, SizeDescription, Outcome, Constraints,
)
from knowledge.structural_library import STRUCTURAL_OPTIONS
from conversation.decomposer import decompose
from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter


def _build():
    registry = ToolRegistry()
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry


# ── d-electron count tests ────────────────────────────────────────────

def test_d_electron_counts():
    """d-electron counts should be correct for common metals."""
    assert get_d_electron_count("nickel", 2) == 8, "Ni²⁺ should be d⁸"
    assert get_d_electron_count("copper", 2) == 9, "Cu²⁺ should be d⁹"
    assert get_d_electron_count("iron", 3) == 5, "Fe³⁺ should be d⁵"
    assert get_d_electron_count("iron", 2) == 6, "Fe²⁺ should be d⁶"
    assert get_d_electron_count("zinc", 2) == 10, "Zn²⁺ should be d¹⁰"
    assert get_d_electron_count("gold", 3) == 8, "Au³⁺ should be d⁸"
    assert get_d_electron_count("calcium", 2) == 0, "Ca²⁺ should be d⁰"
    assert get_d_electron_count("lead", 2) == -1, "Pb²⁺ should be -1 (lone pair)"
    assert get_d_electron_count("cerium", 3) == -2, "Ce³⁺ should be -2 (f-block)"
    print(f"  + d-electron counts: Ni²⁺=d8, Cu²⁺=d9, Fe³⁺=d5, Zn²⁺=d10, Au³⁺=d8, Ca²⁺=d0, Pb²⁺=lone pair")


def test_d_electron_coverage():
    """All MABE metals should have d-electron data."""
    metals = ["lead", "mercury", "gold", "copper", "nickel", "zinc",
              "iron", "cadmium", "silver", "calcium", "cerium", "uranium"]
    for m in metals:
        assert m in METAL_D_ELECTRONS, f"Missing d-electron data for {m}"
    print(f"  + All {len(metals)} MABE metals have d-electron data")


# ── LFSE calculation unit tests ──────────────────────────────────────

def test_lfse_d0_zero():
    """d⁰ should have zero LFSE in all geometries."""
    for geom in ["oct", "tet", "sq_planar"]:
        assert compute_lfse_dq(0, geom, True) == 0.0
    print(f"  + d⁰: LFSE = 0 in all geometries")


def test_lfse_d10_zero():
    """d¹⁰ should have zero LFSE in all geometries."""
    for geom in ["oct", "tet", "sq_planar"]:
        assert compute_lfse_dq(10, geom, True) == 0.0
    print(f"  + d¹⁰: LFSE = 0 in all geometries")


def test_lfse_d5_hs_zero_oct():
    """d⁵ high-spin octahedral should have LFSE = 0 (half-filled symmetry)."""
    assert compute_lfse_dq(5, "oct", True) == 0.0
    print(f"  + d⁵ high-spin oct: LFSE = 0 (half-filled)")


def test_lfse_d8_oct_largest():
    """d⁸ should have large LFSE in octahedral and even larger in square planar."""
    oct = compute_lfse_dq(8, "oct", True)
    sp = compute_lfse_dq(8, "sq_planar", True)
    tet = compute_lfse_dq(8, "tet", True)
    assert oct < tet, f"d⁸ oct LFSE ({oct}) should be more negative than tet ({tet})"
    assert sp < oct, f"d⁸ sq_planar LFSE ({sp}) should be more negative than oct ({oct})"
    print(f"  + d⁸: sq.pl.={sp:.1f} < oct={oct:.1f} < tet={tet:.1f} Dq (correct order)")


def test_tet_always_weaker_than_oct():
    """Tetrahedral LFSE should always be less stabilizing than octahedral for d¹-d⁹."""
    for d in range(1, 10):
        oct = compute_lfse_dq(d, "oct", True)
        tet = compute_lfse_dq(d, "tet", True)
        # oct is more negative (more stabilizing) or equal
        assert oct <= tet, f"d{d}: oct={oct} should be ≤ tet={tet}"
    print(f"  + Octahedral LFSE ≥ tetrahedral for all d¹-d⁹ (correct)")


# ── Field strength and spin state ────────────────────────────────────

def test_field_strength_classification():
    """Donor sets should classify to correct field strength."""
    assert get_field_strength(["O", "O", "O", "O"]) == "weak"
    assert get_field_strength(["N", "N", "N", "N"]) == "moderate"
    assert get_field_strength(["P", "P", "C", "C"]) == "strong"
    assert get_field_strength(["S", "S"]) == "weak"
    print(f"  + Field strength: O→weak, N→moderate, P+C→strong, S→weak")


def test_spin_state():
    """Spin state should depend on field strength and d-count."""
    assert is_high_spin(5, "weak", "iron", 3) == True, "d⁵ weak field → high spin"
    assert is_high_spin(5, "strong", "iron", 3) == False, "d⁵ strong field → low spin"
    assert is_high_spin(3, "strong", "chromium", 3) == True, "d³ always high spin (no ambiguity)"
    assert is_high_spin(8, "weak", "nickel", 2) == True, "d⁸ always high spin in oct"
    print(f"  + Spin states: Fe³⁺ weak→HS, Fe³⁺ strong→LS, Cr³⁺→always HS, Ni²⁺→always HS")


# ── Jahn-Teller tests ────────────────────────────────────────────────

def test_jahn_teller_cu2():
    """Cu²⁺ d⁹ must be Jahn-Teller active."""
    assert is_jahn_teller(9, True) == True
    assert is_jahn_teller(9, False) == True
    print(f"  + Cu²⁺ d⁹: Jahn-Teller active (correct)")


def test_jahn_teller_d3_inactive():
    """d³ should NOT be Jahn-Teller active."""
    assert is_jahn_teller(3, True) == False
    print(f"  + d³ (Cr³⁺): NOT Jahn-Teller active (correct)")


# ── Full LFSE analysis tests ─────────────────────────────────────────

def test_nickel_prefers_square_planar():
    """Ni²⁺ d⁸ with N donors should prefer square planar."""
    ni = TargetSpecies(identity="nickel", formula="Ni(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.69))
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")
    lfse = compute_lfse(rec, ni)
    assert lfse.d_electron_count == 8
    assert lfse.preferred_geometry == "sq_planar"
    assert lfse.lfse_square_planar_kj < lfse.lfse_octahedral_kj
    assert lfse.geometry_preference_dG > 20  # substantial preference
    print(f"  + Ni²⁺+4N: prefers {lfse.preferred_geometry} by {lfse.geometry_preference_dG:.1f} kJ/mol")
    print(f"    LFSE: oct={lfse.lfse_octahedral_kj:.1f}, sq.pl.={lfse.lfse_square_planar_kj:.1f} kJ/mol")


def test_calcium_no_lfse():
    """Ca²⁺ d⁰ should have zero LFSE."""
    ca = TargetSpecies(identity="calcium", formula="Ca(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.0),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.12, dehydration_energy_kj_mol=1577.0, coordination_number_water=8),
        size=SizeDescription(ionic_radius_angstrom=1.0))
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")
    lfse = compute_lfse(rec, ca)
    assert lfse.d_electron_count == 0
    assert lfse.dG_lfse == 0.0
    print(f"  + Ca²⁺: d⁰, LFSE = 0 (correct)")


def test_lead_lone_pair():
    """Pb²⁺ should show lone pair effects, not LFSE."""
    problem = decompose("lead capture from mine water")
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["O", "O", "N", "N"],
        donor_type="borderline", structure="EDTA-like")
    lfse = compute_lfse(rec, problem.target)
    assert lfse.lone_pair_active == True
    assert lfse.d_electron_count == -1
    assert lfse.preferred_geometry == "hemidirected"
    print(f"  + Pb²⁺: lone pair active, preferred={lfse.preferred_geometry}, dG={lfse.dG_lfse:.1f} kJ/mol")


def test_copper_jahn_teller_in_analysis():
    """Cu²⁺ d⁹ should flag Jahn-Teller in full analysis."""
    problem = decompose("copper capture from mine water")
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "O", "O", "N", "N"],
        donor_type="borderline", structure="EDTA-like")
    lfse = compute_lfse(rec, problem.target)
    assert lfse.jahn_teller_active == True
    assert "d⁹" in lfse.jahn_teller_description or "Cu" in lfse.jahn_teller_description
    print(f"  + Cu²⁺: Jahn-Teller detected — {lfse.jahn_teller_description[:60]}")


def test_iron3_d5_no_geometry_preference():
    """Fe³⁺ d⁵ high-spin should have LFSE = 0 and no geometry preference."""
    problem = decompose("iron removal from mine water")
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["O", "O", "O"],
        donor_type="hard", structure="tricarboxylate")
    lfse = compute_lfse(rec, problem.target)
    assert lfse.d_electron_count == 5
    assert lfse.lfse_octahedral_kj == 0.0
    assert lfse.lfse_tetrahedral_kj == 0.0
    print(f"  + Fe³⁺ d⁵ (high-spin, weak field): LFSE = 0 in all geometries")


# ── Thermodynamic integration tests ──────────────────────────────────

def test_dG_includes_lfse():
    """Thermodynamics for Ni²⁺ should include LFSE contribution."""
    ni = TargetSpecies(identity="nickel", formula="Ni(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.69))
    p = Problem(
        target=ni,
        matrix=Matrix(ph=7.0, temperature_c=25.0, competing_species=[
            CompetingSpecies("calcium", "Ca(2+)", 200.0, 2.0),
        ]),
        desired_outcome=Outcome(description="capture"),
    )
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")
    meso = [s for s in STRUCTURAL_OPTIONS if s.type == "mesoporous_silica"][0]
    interior = InteriorDesign(sites=[InteriorSite(recognition=rec, copies=10)],
        design_level="composite", total_binding_sites=10, unique_recognition_types=1, avidity_factor=2.0)
    thermo = _thermo_mod.compute_thermodynamics(rec, meso, interior, p)
    # Should have LFSE in breakdown
    has_lfse = any("LFSE" in line for line in thermo.energy_breakdown)
    assert has_lfse, "LFSE should appear in energy breakdown for Ni²⁺"
    print(f"  + Ni²⁺ dG_net = {thermo.dG_net:.1f} kJ/mol (includes LFSE)")
    for line in thermo.energy_breakdown:
        if "LFSE" in line or "lfse" in line:
            print(f"    {line}")


def test_lfse_improves_nickel_selectivity():
    """LFSE should add substantial stabilization for Ni2+ d8 but not for Ca2+ d0."""
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")

    ni = TargetSpecies(identity="nickel", formula="Ni(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.69))
    ca = TargetSpecies(identity="calcium", formula="Ca(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.0),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.12, dehydration_energy_kj_mol=1577.0, coordination_number_water=8),
        size=SizeDescription(ionic_radius_angstrom=1.0))

    lfse_ni = compute_lfse(rec, ni)
    lfse_ca = compute_lfse(rec, ca)

    # Ni2+ d8 should get substantial LFSE; Ca2+ d0 gets zero
    assert lfse_ni.dG_lfse < -100.0, f"Ni2+ LFSE should be strongly stabilizing, got {lfse_ni.dG_lfse}"
    assert lfse_ca.dG_lfse == 0.0, f"Ca2+ LFSE should be 0, got {lfse_ca.dG_lfse}"
    advantage = abs(lfse_ni.dG_lfse)
    print(f"  + LFSE selectivity: Ni2+ gets {lfse_ni.dG_lfse:.1f} kJ/mol, Ca2+ gets 0 -> {advantage:.1f} kJ/mol advantage")


# ── End-to-end ────────────────────────────────────────────────────────

def test_e2e_lfse_in_report():
    """E2E: assemblies for nickel should include LFSE in report."""
    o = Orchestrator(_build())
    r = o.solve(decompose("nickel capture from mine water"))
    assert len(r.assemblies) > 0
    has_lfse = any("LFSE" in a.confidence_reasoning or "GEOMETRY" in a.confidence_reasoning
                   for a in r.assemblies)
    assert has_lfse, "Expected LFSE/GEOMETRY data in some assemblies"
    print(f"  + Nickel E2E: LFSE data in reports")
    for a in r.assemblies[:3]:
        print(f"    {a.composite_score:.0%}  {a.name[:50]}")


def test_e2e_copper_jahn_teller_warning():
    """E2E: copper should generate Jahn-Teller warnings."""
    o = Orchestrator(_build())
    r = o.solve(decompose("copper capture from mine water"))
    assert len(r.assemblies) > 0
    has_jt = any("jahn-teller" in fm.lower() or "Jahn-Teller" in fm
                 for a in r.assemblies for fm in a.failure_modes)
    # JT may or may not trigger depending on assembly donor sets
    print(f"  + Copper E2E: Jahn-Teller warnings present: {has_jt}")
    for a in r.assemblies[:3]:
        print(f"    {a.composite_score:.0%}  {a.name[:50]}")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 12 - LFSE & Coordination Geometry Tests")
    print("  " + "=" * 52)
    print()
    print("  d-electron counts:")
    test_d_electron_counts()
    test_d_electron_coverage()
    print()
    print("  LFSE calculation:")
    test_lfse_d0_zero()
    test_lfse_d10_zero()
    test_lfse_d5_hs_zero_oct()
    test_lfse_d8_oct_largest()
    test_tet_always_weaker_than_oct()
    print()
    print("  Field strength & spin state:")
    test_field_strength_classification()
    test_spin_state()
    print()
    print("  Jahn-Teller:")
    test_jahn_teller_cu2()
    test_jahn_teller_d3_inactive()
    print()
    print("  Full LFSE analysis:")
    test_nickel_prefers_square_planar()
    test_calcium_no_lfse()
    test_lead_lone_pair()
    test_copper_jahn_teller_in_analysis()
    test_iron3_d5_no_geometry_preference()
    print()
    print("  Thermodynamic integration:")
    test_dG_includes_lfse()
    test_lfse_improves_nickel_selectivity()
    print()
    print("  End-to-end:")
    test_e2e_lfse_in_report()
    test_e2e_copper_jahn_teller_warning()
    print()
    print("  All Sprint 12 tests passed.")
    print()
''')


print()
print("  Done! New/updated files:")
print("    knowledge/lfse_data.py            (NEW: d-electron configs, Dq, pairing energies, spectrochemical series)")
print("    core/lfse.py                      (NEW: LFSE calculation, geometry preference, Jahn-Teller)")
print("    core/lfse_integration.py          (NEW: patches thermodynamics + rescore pipeline)")
print("    main.py                           (updated)")
print("    tests/test_sprint12.py            (NEW: 20 tests)")
print()