"""
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
        return "\n".join(parts)


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
