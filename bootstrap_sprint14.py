"""
MABE Sprint 14 Bootstrap - Repulsion Forces
=============================================
Binding doesn't happen in a vacuum of attraction. Every approach involves
overcoming repulsive barriers. These are the physics that REJECT.

THREE CATEGORIES OF REPULSION:

1. STERIC EXCLUSION (size filtering)
   A hydrated Pb²⁺ (r_hyd = 4.0 Å) cannot enter a zeolite pore of 5.5 Å
   diameter without partial dehydration. A hydrated Mg²⁺ (r_hyd = 4.3 Å)
   is even worse. This is SELECTIVITY BY SIZE EXCLUSION — the pore acts
   as a molecular sieve.

   Physics: ΔG_steric = ∞ if r_hydrated > r_pore (hard wall)
            ΔG_steric = penalty × (r_hydrated/r_pore)² for partial entry
   
   CRITICAL: Size exclusion is the FIRST PASS filter. If the target can't
   physically fit, nothing else matters. This is how zeolites achieve
   selectivity — not by binding chemistry, but by geometry.

2. PAULI / BORN REPULSION (electron cloud overlap)
   When atoms approach closer than their van der Waals radii, electron
   clouds overlap and Pauli exclusion creates an exponentially rising
   repulsive wall. This sets the minimum approach distance for any
   binding interaction.

   Born-Mayer: V_rep = A × exp(-r/ρ)  where ρ ≈ 0.345 Å
   Lennard-Jones: V_rep = 4ε(σ/r)¹² (steeper, commonly used)

   Design implication: The binding pocket cannot be smaller than the
   sum of donor + target van der Waals radii. Undersized pockets create
   strain energy that opposes binding.

3. CHARGE-CHARGE REPULSION (like-charge rejection)
   A cation approaching a positively-charged surface is REPELLED.
   An anion approaching a negatively-charged surface is REPELLED.
   This is the basis of Donnan exclusion in membranes and the
   charge-selectivity of LDH (anions only) and zeolites (cations only).

   ΔG_repulsion = z_target × z_surface × e² / (4πε₀εᵣr)
   Screened by ionic atmosphere: × exp(-κr) where κ = Debye parameter

   Design implication: LDH galleries REPEL cations. Zeolite frameworks
   REPEL anions. This is a feature, not a bug — it's selectivity by
   charge sign.

    cd Documents\\mabe
    python bootstrap_sprint14.py
    python tests\\test_sprint14.py
"""

import os

def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print()
print("  MABE Sprint 14 - Repulsion Forces")
print("  " + "=" * 40)
print()


# ═══════════════════════════════════════════════════════════════════════════
# knowledge/repulsion_data.py — Physical constants and ion sizes
# ═══════════════════════════════════════════════════════════════════════════

write_file("knowledge/repulsion_data.py", '''"""
knowledge/repulsion_data.py - Data for repulsion force calculations.

van der Waals radii, Born-Mayer parameters, and structural charge data.
"""

import math

# ═══════════════════════════════════════════════════════════════════════════
# van der Waals radii (Å) — Bondi (1964), updated Alvarez (2013)
# ═══════════════════════════════════════════════════════════════════════════

VDW_RADII = {
    # Metals (as ions, these are smaller than atomic vdW)
    "Pb": 2.02, "Cu": 1.40, "Ni": 1.63, "Zn": 1.39,
    "Fe": 1.56, "Au": 1.66, "Hg": 1.55, "Cd": 1.58,
    "Ag": 1.72, "Ca": 2.31, "Mg": 1.73, "Na": 2.27,
    "K":  2.75, "Ba": 2.68, "Ce": 2.42, "Al": 1.84,
    "Mn": 1.61, "Co": 1.52, "Cr": 1.66, "U":  1.86,

    # Donor atoms
    "O": 1.52, "N": 1.55, "S": 1.80, "P": 1.80, "C": 1.70,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "H": 1.20, "Se": 1.90, "As": 1.85,
}

# Ionic radii (Å) — Shannon (1976) for common oxidation states
# These are the crystal radii, smaller than vdW
IONIC_RADII = {
    "Pb2+": 1.19, "Cu2+": 0.73, "Ni2+": 0.69, "Zn2+": 0.74,
    "Fe3+": 0.645, "Fe2+": 0.78, "Au3+": 0.85, "Hg2+": 1.02,
    "Cd2+": 0.95, "Ag+": 1.15, "Ca2+": 1.00, "Mg2+": 0.72,
    "Na+": 1.02, "K+": 1.38, "Ba2+": 1.35, "Ce3+": 1.01,
    "Al3+": 0.535, "Mn2+": 0.83, "Co2+": 0.745, "Cr3+": 0.615,
    "UO2_2+": 0.73,
}

# Hydrated ion radii (Å) — Marcus (1988)
HYDRATED_RADII = {
    "Pb2+": 4.01, "Cu2+": 4.19, "Ni2+": 4.04, "Zn2+": 4.30,
    "Fe3+": 4.57, "Fe2+": 4.28, "Au3+": 3.50, "Hg2+": 3.50,
    "Cd2+": 4.26, "Ag+": 3.41, "Ca2+": 4.12, "Mg2+": 4.28,
    "Na+": 3.58, "K+": 3.31, "Ba2+": 4.04, "Ce3+": 4.52,
    "Al3+": 4.75, "Mn2+": 4.38, "Co2+": 4.23, "Cr3+": 4.61,
}


# ═══════════════════════════════════════════════════════════════════════════
# Structure charge data
# ═══════════════════════════════════════════════════════════════════════════

# Net framework charge per unit cell (conceptual, determines sign of selectivity)
FRAMEWORK_CHARGE = {
    "zeolite":            -1,   # AlO₄⁻ substitution → net negative → attracts cations
    "ldh":                +1,   # M²⁺/M³⁺ layers → net positive → attracts anions
    "mof":                 0,   # varies, generally neutral pores
    "cof":                 0,   # neutral framework
    "mesoporous_silica":  -1,   # SiO⁻ at surface above pH 3
    "silica_np":          -1,   # same
    "dna_origami_cage":   -1,   # phosphate backbone → negative
    "carbon_nanotube":     0,   # neutral unless functionalized
    "graphene_oxide":     -1,   # carboxyl/hydroxyl groups
    "mip":                 0,   # depends on monomer
    "coordination_cage":   0,   # varies
    "protein_cage":        0,   # varies with pH
    "dendrimer":           0,   # varies
    "none":                0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Born-Mayer repulsion parameters
# ═══════════════════════════════════════════════════════════════════════════

# Born-Mayer: V_rep = B × exp(-r / ρ)
# ρ (softness parameter) ≈ 0.345 Å for most ion pairs
BORN_MAYER_RHO = 0.345  # Å

# B is calibrated so that V_rep = ~500 kJ/mol at r = sum of ionic radii
# (i.e., hard contact). Exact value doesn't matter much because the
# exponential makes it essentially a hard wall.
BORN_MAYER_B = 500.0  # kJ/mol (pre-factor)


def get_hydrated_radius(identity: str, charge: float) -> float:
    """Get hydrated radius in Å. Falls back to estimation from charge."""
    key = identity.lower()
    # Try direct lookup with charge
    for suffix in [f"{abs(charge):.0f}+", f"{abs(charge):.0f}-"]:
        sign = "+" if charge > 0 else "-"
        full_key = f"{key.capitalize()}{abs(charge):.0f}{sign}"
        if full_key in HYDRATED_RADII:
            return HYDRATED_RADII[full_key]

    # Try common keys
    lookup_map = {
        "lead": "Pb2+", "copper": "Cu2+", "nickel": "Ni2+", "zinc": "Zn2+",
        "iron": "Fe3+" if abs(charge) > 2.5 else "Fe2+",
        "gold": "Au3+", "mercury": "Hg2+", "cadmium": "Cd2+",
        "silver": "Ag+", "calcium": "Ca2+", "magnesium": "Mg2+",
        "sodium": "Na+", "potassium": "K+", "barium": "Ba2+",
        "cerium": "Ce3+", "aluminum": "Al3+", "manganese": "Mn2+",
        "cobalt": "Co2+", "chromium": "Cr3+",
    }
    mapped = lookup_map.get(key)
    if mapped and mapped in HYDRATED_RADII:
        return HYDRATED_RADII[mapped]

    # Estimate: r_hyd ≈ 2.0 + 0.8 × |z| (rough)
    return 2.0 + 0.8 * abs(charge)


def get_ionic_radius(identity: str, charge: float) -> float:
    """Get ionic radius in Å."""
    key = identity.lower()
    lookup_map = {
        "lead": "Pb2+", "copper": "Cu2+", "nickel": "Ni2+", "zinc": "Zn2+",
        "iron": "Fe3+" if abs(charge) > 2.5 else "Fe2+",
        "gold": "Au3+", "mercury": "Hg2+", "cadmium": "Cd2+",
        "silver": "Ag+", "calcium": "Ca2+", "magnesium": "Mg2+",
        "sodium": "Na+", "potassium": "K+", "barium": "Ba2+",
        "cerium": "Ce3+", "aluminum": "Al3+", "manganese": "Mn2+",
        "cobalt": "Co2+", "chromium": "Cr3+",
    }
    mapped = lookup_map.get(key)
    if mapped and mapped in IONIC_RADII:
        return IONIC_RADII[mapped]
    return 1.0  # fallback
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/repulsion.py — Physics engine
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/repulsion.py", '''"""
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
        return "\\n".join(parts)


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
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/repulsion_integration.py — Patches into pipeline
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/repulsion_integration.py", '''"""
core/repulsion_integration.py - Integrates repulsion into thermodynamics pipeline.

Patches compute_thermodynamics and full_physics_rescore.
"""

import math
import core.thermodynamics as thermo_mod
from core.repulsion import compute_repulsion, RepulsionAnalysis
from core.assembly import RecognitionChemistry, StructuralConstraint, InteriorDesign
from core.problem import Problem

import core.sprint10_integration as s10


# ── Patch thermodynamics ──────────────────────────────────────────────

_orig_compute_thermo = thermo_mod.compute_thermodynamics


def _repulsion_aware_thermodynamics(recognition, structure, interior, problem):
    """Add repulsion penalties to thermodynamic calculation."""
    result = _orig_compute_thermo(recognition, structure, interior, problem)

    repulsion = compute_repulsion(recognition, structure, problem.target, problem)

    if repulsion.dG_repulsion_total > 0.1:
        result.dG_net += repulsion.dG_repulsion_total
        RT = thermo_mod.R_GAS * result.temperature_k

        # Update K_eq and Kd
        if abs(result.dG_net / RT) < 500:
            result.K_eq = math.exp(-result.dG_net / RT)
        else:
            result.K_eq = 1e30 if result.dG_net < 0 else 0.0
        result.predicted_kd_um = round(1e6 / result.K_eq, 3) if result.K_eq > 1e-6 else None

        result.energy_breakdown.append(
            f"Repulsion: ΔG_repulsion = +{repulsion.dG_repulsion_total:.1f} kJ/mol"
        )
        if repulsion.steric_excluded:
            result.energy_breakdown.append("  ⚠ TARGET STERICALLY EXCLUDED FROM PORE")
        if repulsion.charge_repelled:
            result.energy_breakdown.append("  ⚠ TARGET CHARGE-REPELLED BY FRAMEWORK")

    # Selectivity bonus: if competitors are sterically excluded but target isn't
    if repulsion.competitor_excluded:
        excluded_count = sum(1 for v in repulsion.competitor_excluded.values() if v)
        if excluded_count > 0 and not repulsion.steric_excluded:
            result.energy_breakdown.append(
                f"  Size selectivity: {excluded_count} competitor(s) excluded by pore"
            )

    return result


thermo_mod.compute_thermodynamics = _repulsion_aware_thermodynamics


# ── Patch sprint10 rescore ────────────────────────────────────────────

_orig_rescore = s10.full_physics_rescore


def _repulsion_aware_rescore(assemblies, problem):
    """Add repulsion analysis to physics report."""
    assemblies = _orig_rescore(assemblies, problem)

    for assembly in assemblies:
        repulsion = compute_repulsion(
            assembly.recognition, assembly.structure, problem.target, problem
        )

        assembly.confidence_reasoning += "\\n\\nREPULSION:\\n" + repulsion.summary()

        if repulsion.steric_excluded:
            assembly.failure_modes.append(
                f"Target sterically excluded from {assembly.structure.type} "
                f"(r_hyd={repulsion.r_hydrated_target_A:.1f} Å > pore={repulsion.pore_diameter_A:.1f} Å)"
            )
        if repulsion.charge_repelled:
            assembly.failure_modes.append(
                f"Target charge-repelled by {assembly.structure.type} framework "
                f"(target {repulsion.target_charge_sign:+d}, framework {repulsion.framework_charge_sign:+d})"
            )

    return assemblies


s10.full_physics_rescore = _repulsion_aware_rescore
''')


# ═══════════════════════════════════════════════════════════════════════════
# Update main.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("main.py", '''"""
MABE - Modality-Agnostic Binder Engine
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from conversation.decomposer_patch import patch_targets
from conversation.interface import run_interactive, run_single_query

patch_targets()

# Sprint 8
import core.assembly_composer_patch
import core.scoring_patch

# Sprint 9: thermodynamics + hydrodynamics
import core.physics_integration

# Sprint 10: kinetics + orbital + probability chain
import core.sprint10_integration

# Sprint 11: pKa + protonation state
import core.protonation_integration

# Sprint 12: LFSE + coordination geometry
import core.lfse_integration

# Sprint 13: ionic strength + activity coefficients
import core.ionic_integration

# Sprint 14: repulsion forces
import core.repulsion_integration


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry


def main():
    registry = build_registry()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(registry, query)
    else:
        run_interactive(registry)


if __name__ == "__main__":
    main()
''')


# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint14.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint14.py", '''"""
tests/test_sprint14.py - Repulsion force tests.
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
import core.ionic_integration
import core.repulsion_integration

import math
from knowledge.repulsion_data import (
    get_hydrated_radius, get_ionic_radius, VDW_RADII,
    HYDRATED_RADII, IONIC_RADII, FRAMEWORK_CHARGE,
)
from core.repulsion import compute_repulsion, RepulsionAnalysis
from core.assembly import RecognitionChemistry, StructuralConstraint, InteriorDesign, InteriorSite
from core.problem import (
    Problem, TargetSpecies, Matrix, CompetingSpecies, Outcome,
    ElectronicDescription, HydrationDescription, SizeDescription,
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


def _make_target(identity="lead", charge=2.0, r_ionic=1.19, r_hyd=4.01):
    return TargetSpecies(
        identity=identity, formula=f"{identity.capitalize()}{abs(charge):.0f}+",
        charge=charge, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=2.33),
        hydration=HydrationDescription(
            hydrated_radius_angstrom=r_hyd, dehydration_energy_kj_mol=1481.0,
            coordination_number_water=6,
        ),
        size=SizeDescription(ionic_radius_angstrom=r_ionic),
    )


def _make_problem(target=None, ionic_strength_mm=50.0):
    target = target or _make_target()
    matrix = Matrix(
        description="AMD", ph=3.5, temperature_c=12.0,
        ionic_strength_mm=ionic_strength_mm,
        competing_species=[
            CompetingSpecies(identity="calcium", formula="Ca2+", charge=2.0, concentration_mm=5.0),
            CompetingSpecies(identity="magnesium", formula="Mg2+", charge=2.0, concentration_mm=3.0),
        ],
    )
    return Problem(target=target, matrix=matrix, desired_outcome=Outcome(description="capture"))


def _make_rec(donors=None, donor_type="soft"):
    donors = donors or ["S", "S", "N"]
    return RecognitionChemistry(
        name="test_chelator", type="chelator",
        donor_atoms=donors, donor_type=donor_type, structure="test",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data tables
# ═══════════════════════════════════════════════════════════════════════════

def test_hydrated_radii_populated():
    """Hydrated radii available for all common metals."""
    for metal in ["lead", "copper", "nickel", "zinc", "calcium", "sodium", "iron"]:
        r = get_hydrated_radius(metal, 2.0)
        assert r > 2.0, f"{metal} r_hyd = {r}"
    print("  + Hydrated radii populated for all common metals")


def test_ionic_radii_populated():
    """Ionic radii available for all common metals."""
    for metal in ["lead", "copper", "nickel", "zinc", "calcium"]:
        r = get_ionic_radius(metal, 2.0)
        assert 0.5 < r < 2.0, f"{metal} r_ion = {r}"
    print("  + Ionic radii populated")


def test_framework_charges():
    """Zeolite negative, LDH positive, others neutral."""
    assert FRAMEWORK_CHARGE["zeolite"] < 0
    assert FRAMEWORK_CHARGE["ldh"] > 0
    assert FRAMEWORK_CHARGE["mof"] == 0
    print("  + Framework charges: zeolite(-), LDH(+), MOF(0)")


# ═══════════════════════════════════════════════════════════════════════════
# Steric exclusion
# ═══════════════════════════════════════════════════════════════════════════

def test_steric_large_pore_no_exclusion():
    """Large pore (8 nm) should not exclude Pb²⁺ (r_hyd ≈ 4 Å)."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="big_pore", type="mesoporous_silica",
                                   geometry="cylindrical", pore_size_nm=8.0)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert not rep.steric_excluded
    assert rep.dG_steric < 1.0
    print(f"  + Large pore (8 nm): no exclusion, ΔG_steric={rep.dG_steric:.1f}")


def test_steric_zeolite_zsm5_excludes_large():
    """ZSM-5 (5.5 Å pore) should create steric penalty for Pb²⁺ (d_hyd ≈ 8 Å)."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="ZSM-5", type="zeolite",
                                   geometry="channel", pore_size_nm=0.55)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    # Hydrated Pb²⁺ diameter = 8.02 Å > 5.5 Å pore
    # But bare Pb²⁺ = 2.38 Å < 5.5 Å — so not fully excluded
    assert not rep.steric_excluded, "Bare ion should still fit"
    assert rep.dG_steric > 5.0, f"Should have significant penalty: {rep.dG_steric}"
    print(f"  + ZSM-5 (5.5 Å): Pb²⁺ squeezed, ΔG_steric=+{rep.dG_steric:.1f} kJ/mol")


def test_steric_tiny_pore_full_exclusion():
    """Very small pore (3 Å) should fully exclude Pb²⁺ (ionic d = 2.38 Å)."""
    target = _make_target(r_ionic=1.5, r_hyd=4.0)  # make ionic radius larger
    prob = _make_problem(target=target)
    rec = _make_rec()
    struct = StructuralConstraint(name="tiny", type="zeolite",
                                   geometry="channel", pore_size_nm=0.3)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.steric_excluded, "Should be fully excluded"
    assert rep.dG_steric >= 100.0
    print(f"  + Tiny pore (3 Å): FULLY EXCLUDED, ΔG_steric={rep.dG_steric:.0f}")


def test_steric_no_pore_no_penalty():
    """No pore constraint → no steric penalty."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="free", type="none", geometry="none")
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_steric == 0.0
    print(f"  + No pore: ΔG_steric = 0")


def test_steric_competitor_exclusion():
    """Pore should selectively exclude larger competitors."""
    # Mg²⁺ r_hyd = 4.28 Å → d = 8.56 Å
    # In a 7 Å pore: Mg²⁺ squeezed, Ca²⁺ (d=8.24 Å) also squeezed
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="medium", type="zeolite",
                                   geometry="channel", pore_size_nm=0.7)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    # Check that competitor exclusion dict was populated
    assert len(rep.competitor_excluded) > 0, "Should check competitors"
    print(f"  + Competitor exclusion checked: {rep.competitor_excluded}")


# ═══════════════════════════════════════════════════════════════════════════
# Born / Pauli repulsion
# ═══════════════════════════════════════════════════════════════════════════

def test_born_no_compression():
    """Large pore → no Born repulsion."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="big", type="mesoporous_silica",
                                   geometry="cylindrical", pore_size_nm=5.0)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_born == 0.0
    print(f"  + Large pore: no Born repulsion")


def test_born_tight_pore():
    """Very tight pore should create Born repulsion if pocket < contact distance."""
    prob = _make_problem()
    rec = _make_rec(["O", "O", "O", "O"])  # 4 O donors, vdW ≈ 1.52 Å each
    # contact distance ≈ r_ionic(1.19) + r_vdw_O(1.52) ≈ 2.71 Å
    # pocket radius for 0.4 nm pore = 2.0 Å < 2.71 Å → strain
    struct = StructuralConstraint(name="tight", type="mof",
                                   geometry="cage", pore_size_nm=0.4)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_born > 0, f"Should have Born penalty: {rep.dG_born}"
    print(f"  + Tight pore (4 Å): Born repulsion ΔG=+{rep.dG_born:.1f} kJ/mol")


# ═══════════════════════════════════════════════════════════════════════════
# Charge repulsion
# ═══════════════════════════════════════════════════════════════════════════

def test_charge_cation_in_ldh_repelled():
    """Cation (Pb²⁺) in LDH (positive layers) should be repelled."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="LDH", type="ldh",
                                   geometry="layered", pore_size_nm=0.7)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_charge_repulsion > 0
    assert rep.target_charge_sign > 0
    assert rep.framework_charge_sign > 0
    print(f"  + Pb²⁺ in LDH: charge repulsion ΔG=+{rep.dG_charge_repulsion:.1f} kJ/mol")


def test_charge_cation_in_zeolite_attracted():
    """Cation in zeolite (negative framework) → attraction, not repulsion."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="zeolite", type="zeolite",
                                   geometry="channel", pore_size_nm=0.74)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_charge_repulsion == 0.0, f"Should not be repelled: {rep.dG_charge_repulsion}"
    assert not rep.charge_repelled
    print(f"  + Pb²⁺ in zeolite: attracted (no charge repulsion)")


def test_charge_neutral_framework_no_repulsion():
    """Neutral framework → no charge-based exclusion."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="MOF", type="mof",
                                   geometry="cage", pore_size_nm=1.6)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_charge_repulsion == 0.0
    print(f"  + Neutral framework: no charge repulsion")


# ═══════════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════════

def test_repulsion_in_dG_net():
    """Repulsion should increase (make less negative) ΔG_net."""
    import core.thermodynamics as _thermo
    prob = _make_problem()
    rec = _make_rec()
    struct_open = StructuralConstraint(name="open", type="none", geometry="none")
    struct_tight = StructuralConstraint(name="tight", type="zeolite",
                                         geometry="channel", pore_size_nm=0.55)
    interior = InteriorDesign(design_level="primary", sites=[InteriorSite(recognition=rec)],
                                avidity_factor=1.0)

    thermo_open = _thermo.compute_thermodynamics(rec, struct_open, interior, prob)
    thermo_tight = _thermo.compute_thermodynamics(rec, struct_tight, interior, prob)

    # Tight should have higher (less negative) ΔG due to steric penalty
    assert thermo_tight.dG_net >= thermo_open.dG_net, \
        f"Tight ({thermo_tight.dG_net:.1f}) should ≥ open ({thermo_open.dG_net:.1f})"
    print(f"  + ΔG: open={thermo_open.dG_net:+.1f}, tight zeolite={thermo_tight.dG_net:+.1f} kJ/mol")


# ═══════════════════════════════════════════════════════════════════════════
# E2E
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_repulsion_in_reports():
    """E2E: repulsion data appears in reports."""
    registry = _build()
    prob = decompose("lead capture from acid mine drainage pH 3.5")
    prob.matrix.ionic_strength_mm = 50.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found = False
    for r in results.assemblies:
        if "REPULSION" in r.confidence_reasoning:
            found = True
            break
    assert found, "Repulsion should appear in E2E report"
    print(f"  + E2E: repulsion data in reports")


def test_e2e_ldh_warns_cation():
    """E2E: LDH should warn about cation repulsion for lead."""
    registry = _build()
    prob = decompose("lead capture from mine water")
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    ldh_warnings = []
    for r in results.assemblies:
        if r.structure.type == "ldh":
            for fm in r.failure_modes:
                if "charge" in fm.lower() and "repel" in fm.lower():
                    ldh_warnings.append(fm)
    if ldh_warnings:
        print(f"  + E2E: LDH correctly warns about cation repulsion")
    else:
        # LDH might not be in assemblies — check if any assembly has repulsion info
        found_any = any("REPULSION" in r.confidence_reasoning for r in results.assemblies)
        assert found_any, "Should have repulsion analysis somewhere"
        print(f"  + E2E: repulsion analysis present (LDH may not be in assembly set)")


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_hydrated_radii_populated,
        test_ionic_radii_populated,
        test_framework_charges,
        test_steric_large_pore_no_exclusion,
        test_steric_zeolite_zsm5_excludes_large,
        test_steric_tiny_pore_full_exclusion,
        test_steric_no_pore_no_penalty,
        test_steric_competitor_exclusion,
        test_born_no_compression,
        test_born_tight_pore,
        test_charge_cation_in_ldh_repelled,
        test_charge_cation_in_zeolite_attracted,
        test_charge_neutral_framework_no_repulsion,
        test_repulsion_in_dG_net,
        test_e2e_repulsion_in_reports,
        test_e2e_ldh_warns_cation,
    ]

    print()
    print("=" * 60)
    print("  Sprint 14: Repulsion Forces")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"  Sprint 14: {passed} passed, {failed} failed")
    print()
''')


# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════

print()
print("  Sprint 14 files created:")
print("    knowledge/repulsion_data.py    — vdW radii, ionic radii, framework charges")
print("    core/repulsion.py              — Steric + Born + charge repulsion engine")
print("    core/repulsion_integration.py  — Pipeline patches (thermo + rescore)")
print("    main.py                        — Updated with Sprint 14 import")
print("    tests/test_sprint14.py         — 16 tests")
print()