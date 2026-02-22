"""
core/repulsion.py - Repulsion force calculations.

Three repulsion mechanisms:
1. Steric exclusion: hydrated ion vs pore size
2. Pauli/Born repulsion: electron cloud overlap at short range
3. Charge-charge repulsion: like-charge framework rejection

Each returns a ΔG penalty (positive = unfavorable = repulsive).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.problem import Problem, TargetSpecies
from core.assembly import RecognitionChemistry, StructuralConstraint
from knowledge.repulsion_data import (
    get_hydrated_radius, get_ionic_radius, VDW_RADII,
    FRAMEWORK_CHARGE, BORN_MAYER_RHO, BORN_MAYER_B,
)
from knowledge.ionic_data import debye_length_nm


@dataclass
class RepulsionAnalysis:
    """Complete repulsion analysis for a binding interaction."""

    # Steric
    r_hydrated_target_A: float = 0.0      # Å
    pore_diameter_A: float = 0.0          # Å (0 = no pore constraint)
    steric_ratio: float = 0.0            # r_hyd / r_pore (>1 = excluded)
    dG_steric: float = 0.0              # kJ/mol penalty
    steric_excluded: bool = False        # True = cannot enter at all

    # Born / Pauli
    r_contact_A: float = 0.0            # sum of ionic + donor vdW radii
    r_pocket_A: float = 0.0             # estimated pocket size
    pocket_strain: float = 0.0          # compression ratio (>1 = strain)
    dG_born: float = 0.0                # kJ/mol penalty

    # Charge repulsion
    target_charge_sign: int = 0
    framework_charge_sign: int = 0
    charge_repelled: bool = False        # True = wrong charge sign
    dG_charge_repulsion: float = 0.0    # kJ/mol penalty

    # Total
    dG_repulsion_total: float = 0.0

    # Competitor comparison
    competitor_excluded: dict = field(default_factory=dict)
    # {identity: excluded_bool} — competitors blocked by size

    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.steric_excluded:
            parts.append(
                f"STERIC EXCLUSION: r_hyd={self.r_hydrated_target_A:.1f} Å > "
                f"pore={self.pore_diameter_A:.1f} Å — target cannot enter"
            )
        elif self.dG_steric > 0.1:
            parts.append(
                f"Steric penalty: {self.dG_steric:+.1f} kJ/mol "
                f"(r_hyd/r_pore = {self.steric_ratio:.2f})"
            )
        if self.dG_born > 0.1:
            parts.append(f"Born repulsion: {self.dG_born:+.1f} kJ/mol (pocket strain)")
        if self.charge_repelled:
            parts.append(
                f"CHARGE EXCLUSION: target {self.target_charge_sign:+d} vs "
                f"framework {self.framework_charge_sign:+d} — repelled"
            )
        elif self.dG_charge_repulsion > 0.1:
            parts.append(f"Charge repulsion: {self.dG_charge_repulsion:+.1f} kJ/mol")
        if self.competitor_excluded:
            excluded = [k for k, v in self.competitor_excluded.items() if v]
            if excluded:
                parts.append(f"Competitors excluded by size: {', '.join(excluded)}")
        if not parts:
            parts.append("No significant repulsion barriers")
        parts.append(f"ΔG_repulsion_total = {self.dG_repulsion_total:+.1f} kJ/mol")
        return "\n".join(parts)


def compute_repulsion(
    recognition: RecognitionChemistry,
    structure: StructuralConstraint,
    target: TargetSpecies,
    problem: Problem,
) -> RepulsionAnalysis:
    """Compute all repulsion forces for a binding interaction."""
    result = RepulsionAnalysis()
    breakdown = []

    target_charge = target.charge if target.charge else 0.0
    target_identity = target.identity.lower()
    I_mm = problem.matrix.ionic_strength_mm or 10.0
    temp_c = problem.matrix.temperature_c or 25.0

    # ── 1. Steric exclusion ───────────────────────────────────────────

    r_hyd = _get_target_hydrated_radius(target)
    result.r_hydrated_target_A = r_hyd

    pore_nm = structure.pore_size_nm
    if pore_nm is not None and pore_nm > 0:
        pore_A = pore_nm * 10.0  # nm → Å
        result.pore_diameter_A = pore_A

        # Ratio: hydrated diameter vs pore diameter
        hyd_diameter = 2.0 * r_hyd
        ratio = hyd_diameter / pore_A
        result.steric_ratio = round(ratio, 3)

        if ratio > 1.0:
            # Target too large — but can it partially dehydrate?
            # Dehydration cost already in ΔG_desolv. The question is
            # whether the BARE ion fits. Check ionic radius.
            r_ionic = _get_target_ionic_radius(target)
            bare_diameter = 2.0 * r_ionic
            bare_ratio = bare_diameter / pore_A

            if bare_ratio > 0.95:
                # Even bare ion doesn't fit
                result.steric_excluded = True
                result.dG_steric = 200.0  # effectively infinite
                breakdown.append(
                    f"STERIC EXCLUSION: even bare ion (d={bare_diameter:.1f} Å) "
                    f"cannot fit in pore ({pore_A:.1f} Å)"
                )
            else:
                # Hydrated doesn't fit but bare does — extra dehydration cost
                # Penalty scales with how much of the hydration shell must be stripped
                excess_fraction = (ratio - 1.0) / ratio
                dG_steric = excess_fraction * 30.0  # ~30 kJ/mol for full extra shell stripping
                result.dG_steric = round(dG_steric, 2)
                breakdown.append(
                    f"Steric squeeze: hydrated d={hyd_diameter:.1f} Å > pore {pore_A:.1f} Å "
                    f"→ extra dehydration penalty {dG_steric:+.1f} kJ/mol"
                )
        elif ratio > 0.8:
            # Tight fit — friction/confinement penalty
            confinement = (ratio - 0.8) / 0.2 * 5.0  # 0-5 kJ/mol
            result.dG_steric = round(confinement, 2)
            if confinement > 0.5:
                breakdown.append(
                    f"Confinement penalty: ratio={ratio:.2f} → {confinement:+.1f} kJ/mol"
                )
        else:
            breakdown.append(f"Pore large enough: ratio={ratio:.2f} (no steric barrier)")

        # Check competitors for size exclusion selectivity
        for comp in problem.matrix.competing_species:
            comp_r_hyd = get_hydrated_radius(comp.identity, abs(comp.charge))
            comp_diameter = 2.0 * comp_r_hyd
            comp_ratio = comp_diameter / pore_A
            excluded = comp_ratio > 1.0
            result.competitor_excluded[comp.identity] = excluded
            if excluded:
                breakdown.append(
                    f"  Size excludes {comp.identity}: "
                    f"d_hyd={comp_diameter:.1f} Å > pore {pore_A:.1f} Å"
                )
    else:
        breakdown.append("No pore constraint — steric exclusion not applicable")

    # ── 2. Born / Pauli repulsion ─────────────────────────────────────

    r_ionic = _get_target_ionic_radius(target)
    # Average donor vdW radius
    donors = recognition.donor_atoms or ["O", "N"]
    donor_vdw_avg = sum(VDW_RADII.get(d, 1.55) for d in donors) / len(donors)

    # Contact distance = sum of radii
    r_contact = r_ionic + donor_vdw_avg
    result.r_contact_A = round(r_contact, 2)

    # If structure constrains pocket size, check for compression
    if pore_nm is not None and pore_nm > 0:
        # Effective pocket radius ≈ pore radius / 2 for interior binding
        pocket_r = (pore_nm * 10.0) / 2.0
        result.r_pocket_A = round(pocket_r, 2)

        if pocket_r < r_contact:
            # Pocket too small — Born repulsion
            compression = r_contact / pocket_r
            result.pocket_strain = round(compression, 3)

            # Born-Mayer: exponential rise as atoms compress
            overlap = r_contact - pocket_r  # Å of compression
            dG_born = BORN_MAYER_B * math.exp(-overlap / BORN_MAYER_RHO)
            # Cap at reasonable values (this is semi-empirical)
            dG_born = min(dG_born, 100.0)
            # Only count if significant
            if overlap > 0.1:
                result.dG_born = round(dG_born * (overlap / r_contact), 2)
                breakdown.append(
                    f"Born repulsion: pocket={pocket_r:.1f} Å < contact={r_contact:.1f} Å "
                    f"→ {result.dG_born:+.1f} kJ/mol"
                )
        else:
            breakdown.append(
                f"Pocket accommodates target: pocket={pocket_r:.1f} Å ≥ contact={r_contact:.1f} Å"
            )

    # ── 3. Charge-charge repulsion ────────────────────────────────────

    framework_sign = FRAMEWORK_CHARGE.get(structure.type, 0)
    target_sign = 1 if target_charge > 0 else (-1 if target_charge < 0 else 0)
    result.target_charge_sign = target_sign
    result.framework_charge_sign = framework_sign

    if framework_sign != 0 and target_sign != 0:
        if framework_sign * target_sign > 0:
            # Same sign → REPULSION
            # Magnitude depends on charge and screening
            kappa_inv_nm = debye_length_nm(I_mm, temp_c)
            # At typical binding distance (~0.5 nm), screened Coulomb
            r_bind_nm = 0.5
            screening = math.exp(-r_bind_nm / kappa_inv_nm) if kappa_inv_nm > 0.01 else 0.0

            # Unscreened repulsion energy (Coulomb)
            # ΔG ≈ z_t × z_f × 14.4 / (ε_r × r) eV → kJ/mol
            # For unit charges at 0.5 nm in water (ε=78): ~3.5 kJ/mol
            eps_r = 78.0
            dG_coulomb = abs(target_charge) * abs(framework_sign) * 138.9 / (eps_r * 5.0)
            dG_screened = dG_coulomb * screening
            result.dG_charge_repulsion = round(dG_screened, 2)

            if dG_screened > 5.0:
                result.charge_repelled = True
                breakdown.append(
                    f"CHARGE REPULSION: target {target_sign:+d} vs framework {framework_sign:+d} "
                    f"→ {dG_screened:+.1f} kJ/mol (screened by κ⁻¹={kappa_inv_nm:.1f} nm)"
                )
            elif dG_screened > 0.5:
                breakdown.append(
                    f"Mild charge repulsion: {dG_screened:+.1f} kJ/mol "
                    f"(screened at I={I_mm:.0f} mM)"
                )
        else:
            # Opposite sign → attraction (already handled by ΔG_electrostatic)
            breakdown.append(
                f"Charge attraction: target {target_sign:+d} × framework {framework_sign:+d} "
                f"(favorable, in ΔG_electrostatic)"
            )
    else:
        breakdown.append("No framework charge → no charge-based exclusion")

    # ── Total ─────────────────────────────────────────────────────────

    dG_total = result.dG_steric + result.dG_born + result.dG_charge_repulsion
    result.dG_repulsion_total = round(dG_total, 2)
    result.breakdown = breakdown

    return result


def _get_target_hydrated_radius(target: TargetSpecies) -> float:
    """Get hydrated radius from target, with fallback."""
    if target.hydration and target.hydration.hydrated_radius_angstrom:
        return target.hydration.hydrated_radius_angstrom
    if target.size and target.size.hydrated_radius_angstrom:
        return target.size.hydrated_radius_angstrom
    return get_hydrated_radius(target.identity, abs(target.charge) if target.charge else 2.0)


def _get_target_ionic_radius(target: TargetSpecies) -> float:
    """Get ionic radius from target, with fallback."""
    if target.size and target.size.ionic_radius_angstrom:
        return target.size.ionic_radius_angstrom
    return get_ionic_radius(target.identity, abs(target.charge) if target.charge else 2.0)
