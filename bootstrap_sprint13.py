"""
MABE Sprint 13 Bootstrap - Ionic Strength & Activity Coefficients
==================================================================
Real solutions are not ideal. AMD at Sudbury: I ~ 50-200 mM.
Seawater: I ~ 700 mM. Clean river: I ~ 1-5 mM.

At I = 100 mM, γ(Pb²⁺) ≈ 0.37 — effective concentration is 37% of
analytical. The ionic atmosphere screens charges and shifts equilibria.

Effects on binding:
  1. K_eq shifts via activity coefficients
  2. Electrostatic screening (Debye-Hückel shielding)
  3. pKa shifts (0.1-0.5 units)
  4. Differential selectivity (different z → different γ)

Models:
  Extended Debye-Hückel:  log γ = -Az²√I / (1 + Ba√I)  (I < 100 mM)
  Davies equation:        log γ = -Az²(√I/(1+√I) - 0.3I) (I < 500 mM)

    cd Documents\\mabe
    python bootstrap_sprint13.py
    python tests\\test_sprint13.py
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
print("  MABE Sprint 13 - Ionic Strength & Activity Coefficients")
print("  " + "=" * 56)
print()


# ═══════════════════════════════════════════════════════════════════════════
# knowledge/ionic_data.py — Ion-specific parameters
# ═══════════════════════════════════════════════════════════════════════════

write_file("knowledge/ionic_data.py", '''"""
knowledge/ionic_data.py - Ion-specific parameters for activity coefficient calculation.

Extended Debye-Hückel:
    log γ_i = -A z_i² √I / (1 + B a_i √I)

Where a_i = effective hydrated ion diameter (Å) from Kielland (1937).

Temperature dependence:
    A(T) = 1.825e6 × (ε_r × T)^(-3/2)
    B(T) = 50.3 × (ε_r × T)^(-1/2)
"""

import math
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Debye-Hückel constants
# ═══════════════════════════════════════════════════════════════════════════

DH_A_25C = 0.5085
DH_B_25C = 0.3281


def _water_permittivity(temp_c: float) -> float:
    """Relative permittivity of water vs temperature (°C).
    Fit: eps = 87.74 - 0.4001T + 9.398e-4 T² - 1.410e-6 T³"""
    T = temp_c
    return 87.74 - 0.4001 * T + 9.398e-4 * T**2 - 1.41e-6 * T**3


def dh_A(temp_c: float = 25.0) -> float:
    """Debye-Hückel A parameter at given temperature."""
    T_K = temp_c + 273.15
    eps = _water_permittivity(temp_c)
    return 1.825e6 * (eps * T_K) ** (-1.5)


def dh_B(temp_c: float = 25.0) -> float:
    """Debye-Hückel B parameter at given temperature."""
    T_K = temp_c + 273.15
    eps = _water_permittivity(temp_c)
    return 50.29 * (eps * T_K) ** (-0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Ion-specific effective diameters (Å) — Kielland table
# ═══════════════════════════════════════════════════════════════════════════

ION_DIAMETER_ANGSTROM = {
    "lead":      4.5,  "copper":    6.0,  "nickel":    6.0,
    "zinc":      6.0,  "iron_3":    9.0,  "iron_2":    6.0,
    "gold":      4.0,  "mercury":   4.0,  "cadmium":   5.0,
    "silver":    2.5,  "calcium":   6.0,  "magnesium": 8.0,
    "sodium":    4.0,  "potassium": 3.0,  "barium":    5.0,
    "cerium":    9.0,  "uranium":   9.0,  "aluminum":  9.0,
    "manganese": 6.0,  "cobalt":    6.0,  "chromium":  9.0,
    "hydrogen":  9.0,
    "chloride":  3.0,  "sulfate":   4.0,  "nitrate":   3.0,
    "carbonate": 4.5,  "phosphate": 4.0,  "hydroxide": 3.5,
    "fluoride":  3.5,  "arsenate":  4.0,  "selenite":  4.0,
}

DEFAULT_ION_DIAMETER = 5.0


def get_ion_diameter(identity: str, oxidation_state: int = 2) -> float:
    """Get effective hydrated ion diameter in Å."""
    key = identity.lower().strip()
    if key == "iron" and oxidation_state == 3:
        key = "iron_3"
    elif key == "iron" and oxidation_state == 2:
        key = "iron_2"
    return ION_DIAMETER_ANGSTROM.get(key, DEFAULT_ION_DIAMETER)


# ═══════════════════════════════════════════════════════════════════════════
# Activity coefficient calculations
# ═══════════════════════════════════════════════════════════════════════════

def ionic_strength_from_mm(I_mm: float) -> float:
    """Convert ionic strength from mM to mol/L (M)."""
    return I_mm / 1000.0


def debye_huckel_extended(charge: float, I_M: float, a_angstrom: float,
                           temp_c: float = 25.0) -> float:
    """Extended Debye-Hückel. Valid I < ~100 mM."""
    if I_M <= 0 or charge == 0:
        return 1.0
    A = dh_A(temp_c)
    B = dh_B(temp_c)
    z2 = charge ** 2
    sqrt_I = math.sqrt(I_M)
    log_gamma = -A * z2 * sqrt_I / (1.0 + B * a_angstrom * sqrt_I)
    return 10.0 ** log_gamma


def davies_equation(charge: float, I_M: float, temp_c: float = 25.0) -> float:
    """Davies equation. Valid I up to ~500 mM."""
    if I_M <= 0 or charge == 0:
        return 1.0
    A = dh_A(temp_c)
    z2 = charge ** 2
    sqrt_I = math.sqrt(I_M)
    log_gamma = -A * z2 * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I_M)
    return 10.0 ** log_gamma


def compute_activity_coefficient(identity: str, charge: float,
                                   I_mm: float, temp_c: float = 25.0,
                                   oxidation_state: int = 2) -> float:
    """
    Activity coefficient for an ion at given ionic strength.
    Extended DH for I < 100 mM, Davies above.
    """
    I_M = ionic_strength_from_mm(I_mm)
    if I_M <= 0 or charge == 0:
        return 1.0
    a = get_ion_diameter(identity, oxidation_state)
    if I_M <= 0.1:
        return debye_huckel_extended(charge, I_M, a, temp_c)
    else:
        return davies_equation(charge, I_M, temp_c)


def pka_ionic_strength_correction(pka_zero: float, charge_acid: float,
                                    charge_base: float, I_mm: float,
                                    temp_c: float = 25.0) -> float:
    """
    Correct pKa for ionic strength.
    pKa(I) = pKa(0) + log(γ_acid) - log(γ_base) - log(γ_H+)
    """
    I_M = ionic_strength_from_mm(I_mm)
    if I_M <= 0:
        return pka_zero
    A = dh_A(temp_c)
    sqrt_I = math.sqrt(I_M)

    def log_gamma(z):
        if z == 0:
            return 0.0
        return -A * z**2 * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I_M)

    correction = log_gamma(charge_acid) - log_gamma(charge_base) - log_gamma(1)
    return pka_zero + correction


def debye_length_nm(I_mm: float, temp_c: float = 25.0) -> float:
    """Debye screening length in nm. κ⁻¹ ≈ 0.304/√I(M) at 25°C."""
    I_M = ionic_strength_from_mm(I_mm)
    if I_M <= 0:
        return 1000.0
    eps = _water_permittivity(temp_c)
    eps_25 = _water_permittivity(25.0)
    T_K = temp_c + 273.15
    temp_factor = math.sqrt(eps * T_K / (eps_25 * 298.15))
    return 0.304 * temp_factor / math.sqrt(I_M)
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/ionic_strength.py — Physics engine
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/ionic_strength.py", '''"""
core/ionic_strength.py - Ionic strength corrections for binding thermodynamics.

Corrections:
1. ΔG_ionic: activity coefficient correction to K_eq
2. Electrostatic screening via Debye length
3. pKa correction for ionic strength
4. Selectivity adjustment (different z → different γ)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.problem import Problem, TargetSpecies
from core.assembly import RecognitionChemistry, StructuralConstraint
from knowledge.ionic_data import (
    compute_activity_coefficient, debye_length_nm,
    pka_ionic_strength_correction, ionic_strength_from_mm,
    get_ion_diameter, dh_A,
)


@dataclass
class IonicStrengthAnalysis:
    """Complete ionic strength correction analysis."""
    ionic_strength_mm: float = 0.0
    ionic_strength_M: float = 0.0
    temperature_c: float = 25.0

    gamma_target: float = 1.0
    gamma_product: float = 1.0
    gamma_competitor_worst: float = 1.0

    debye_length_nm: float = 100.0
    electrostatic_screening_factor: float = 1.0

    dG_activity: float = 0.0
    dG_screening: float = 0.0

    pka_shift_typical: float = 0.0
    selectivity_correction_factor: float = 1.0

    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ionic_strength_mm < 0.1:
            return "I ≈ 0 — ideal dilute solution, no correction"
        parts = [
            f"I = {self.ionic_strength_mm:.0f} mM ({self.ionic_strength_M:.4f} M)",
            f"  γ_target = {self.gamma_target:.3f} | Debye length = {self.debye_length_nm:.1f} nm",
            f"  ΔG_activity = {self.dG_activity:+.1f} kJ/mol | "
            f"ΔG_screening = {self.dG_screening:+.1f} kJ/mol",
        ]
        if abs(self.pka_shift_typical) > 0.05:
            parts.append(f"  pKa shift ≈ {self.pka_shift_typical:+.2f} units")
        return "\\n".join(parts)


def _estimate_product_charge(target_charge: float, donor_atoms: list[str],
                               donor_type: str) -> float:
    """Estimate charge of metal-binder complex.
    Anionic donors (O, S) partially neutralize the metal charge."""
    anionic_donors = {"O": 0.5, "S": 0.7, "N": 0.0, "P": 0.0, "electrostatic": 0.0}
    negative = sum(anionic_donors.get(d, 0.0) for d in donor_atoms)
    return max(-2.0, min(target_charge - negative, target_charge))


def compute_ionic_strength_correction(
    recognition: RecognitionChemistry,
    target: TargetSpecies,
    problem: Problem,
    structure: StructuralConstraint = None,
) -> IonicStrengthAnalysis:
    """Compute ionic strength corrections for binding thermodynamics."""
    result = IonicStrengthAnalysis()
    breakdown = []

    I_mm = problem.matrix.ionic_strength_mm
    if I_mm is None or I_mm <= 0:
        I_mm = _estimate_ionic_strength(problem)
        breakdown.append(f"Estimated I = {I_mm:.0f} mM from competing species")

    temp_c = problem.matrix.temperature_c or 25.0
    I_M = ionic_strength_from_mm(I_mm)

    result.ionic_strength_mm = I_mm
    result.ionic_strength_M = I_M
    result.temperature_c = temp_c

    if I_M < 1e-6:
        breakdown.append("Near-zero ionic strength — ideal solution")
        result.breakdown = breakdown
        return result

    target_charge = abs(target.charge) if target.charge else 2.0
    target_identity = target.identity.lower()
    donors = recognition.donor_atoms or ["O", "N"]
    donor_type = recognition.donor_type or "borderline"
    RT = 8.314e-3 * (temp_c + 273.15)

    # ── Activity coefficients ─────────────────────────────────────────

    gamma_target = compute_activity_coefficient(
        target_identity, target_charge, I_mm, temp_c
    )
    result.gamma_target = round(gamma_target, 4)

    product_charge = _estimate_product_charge(target_charge, donors, donor_type)
    gamma_product = compute_activity_coefficient(
        "unknown_complex", product_charge, I_mm, temp_c
    )
    result.gamma_product = round(gamma_product, 4)

    breakdown.append(
        f"γ_target({target_identity} z={target_charge:+.0f}) = {gamma_target:.3f}"
    )
    breakdown.append(
        f"γ_complex(z={product_charge:+.1f}) = {gamma_product:.3f}"
    )

    # ΔG correction: K_real = K_ideal × γ_reactants / γ_products
    # For M + L(immobilized) → ML: K_real = K_ideal × γ_M / γ_ML
    if gamma_target > 0 and gamma_product > 0:
        ratio = gamma_target / gamma_product
        dG_activity = -RT * math.log(ratio) if ratio > 0 else 0.0
    else:
        dG_activity = 0.0

    result.dG_activity = round(dG_activity, 2)
    breakdown.append(f"ΔG_activity = -RT ln(γ_target/γ_product) = {dG_activity:+.2f} kJ/mol")

    # ── Debye screening ───────────────────────────────────────────────

    kappa_inv = debye_length_nm(I_mm, temp_c)
    result.debye_length_nm = round(kappa_inv, 2)

    r_typical = 1.0  # nm approach distance
    screening = math.exp(-r_typical / kappa_inv) if kappa_inv > 0.01 else 0.0
    result.electrostatic_screening_factor = round(screening, 3)

    breakdown.append(
        f"Debye length κ⁻¹ = {kappa_inv:.1f} nm | "
        f"screening at {r_typical:.0f} nm = {screening:.2f}"
    )

    # ── pKa shift ─────────────────────────────────────────────────────

    pka_shift = pka_ionic_strength_correction(4.5, 0.0, -1.0, I_mm, temp_c) - 4.5
    result.pka_shift_typical = round(pka_shift, 3)
    breakdown.append(f"pKa shift (carboxylate): {pka_shift:+.2f} units at I = {I_mm:.0f} mM")

    # ── Selectivity effect ────────────────────────────────────────────

    worst_gamma = 1.0
    for comp in problem.matrix.competing_species:
        comp_charge = abs(comp.charge)
        comp_gamma = compute_activity_coefficient(
            comp.identity, comp_charge, I_mm, temp_c
        )
        if comp_gamma < worst_gamma:
            worst_gamma = comp_gamma
    result.gamma_competitor_worst = round(worst_gamma, 4)

    if worst_gamma > 0 and gamma_target > 0:
        sel_ratio = gamma_target / worst_gamma
        result.selectivity_correction_factor = round(sel_ratio, 3)
        if abs(sel_ratio - 1.0) > 0.01:
            breakdown.append(
                f"γ differential: target {gamma_target:.3f} vs competitor {worst_gamma:.3f} "
                f"(ratio {sel_ratio:.3f})"
            )

    result.breakdown = breakdown
    return result


def _estimate_ionic_strength(problem: Problem) -> float:
    """Estimate I from competing species. I = 0.5 × Σ(c_i × z_i²)."""
    I = 0.0
    for comp in problem.matrix.competing_species:
        c_mM = comp.concentration_mm  # already in mM
        z = abs(comp.charge)
        I += c_mM * z ** 2
    target_z = abs(problem.target.charge) if problem.target.charge else 2.0
    I += 0.01 * target_z ** 2
    I *= 0.5
    return max(1.0, I)
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/ionic_integration.py — Patches into pipeline
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/ionic_integration.py", '''"""
core/ionic_integration.py - Integrates ionic strength corrections into pipeline.

Patches compute_thermodynamics and full_physics_rescore.
"""

import math
import core.thermodynamics as thermo_mod
from core.ionic_strength import compute_ionic_strength_correction, IonicStrengthAnalysis
from core.assembly import RecognitionChemistry, StructuralConstraint, InteriorDesign
from core.problem import Problem

import core.sprint10_integration as s10


# ── Patch thermodynamics ──────────────────────────────────────────────

_orig_compute_thermo = thermo_mod.compute_thermodynamics


def _ionic_aware_thermodynamics(recognition: RecognitionChemistry,
                                  structure: StructuralConstraint,
                                  interior: InteriorDesign,
                                  problem: Problem):
    """Ionic-strength-aware thermodynamic calculation."""
    result = _orig_compute_thermo(recognition, structure, interior, problem)

    ionic = compute_ionic_strength_correction(recognition, problem.target, problem, structure)

    if ionic.ionic_strength_M < 1e-6:
        result.energy_breakdown.append("Ionic strength: ideal dilute solution (no correction)")
        return result

    RT = thermo_mod.R_GAS * result.temperature_k

    # 1. Activity coefficient correction
    dG_activity = ionic.dG_activity
    if abs(dG_activity) > 0.01:
        result.dG_net += dG_activity
        result.energy_breakdown.append(
            f"Ionic strength ({ionic.ionic_strength_mm:.0f} mM): "
            f"ΔG_activity = {dG_activity:+.1f} kJ/mol (γ = {ionic.gamma_target:.3f})"
        )

    # 2. Electrostatic screening
    screening = ionic.electrostatic_screening_factor
    if screening < 0.99:
        dG_elec_original = result.dG_electrostatic
        dG_screening = dG_elec_original * (screening - 1.0)
        result.dG_net += dG_screening
        if abs(dG_screening) > 0.1:
            result.energy_breakdown.append(
                f"  ΔG_screening = {dG_screening:+.1f} kJ/mol "
                f"(Debye κ⁻¹ = {ionic.debye_length_nm:.1f} nm)"
            )

    # Update K_eq and Kd
    if abs(result.dG_net / RT) < 500:
        result.K_eq = math.exp(-result.dG_net / RT)
    else:
        result.K_eq = 1e30 if result.dG_net < 0 else 0.0
    result.predicted_kd_um = round(1e6 / result.K_eq, 3) if result.K_eq > 1e-6 else None

    return result


thermo_mod.compute_thermodynamics = _ionic_aware_thermodynamics


# ── Patch sprint10 rescore ────────────────────────────────────────────

_orig_rescore = s10.full_physics_rescore


def _ionic_aware_rescore(assemblies, problem):
    """Add ionic strength analysis to physics report."""
    assemblies = _orig_rescore(assemblies, problem)

    for assembly in assemblies:
        ionic = compute_ionic_strength_correction(
            assembly.recognition, problem.target, problem, assembly.structure
        )
        if ionic.ionic_strength_M > 1e-6:
            assembly.confidence_reasoning += "\\n\\nIONIC STRENGTH:\\n" + ionic.summary()

            if ionic.ionic_strength_mm > 500:
                assembly.failure_modes.append(
                    f"Very high ionic strength ({ionic.ionic_strength_mm:.0f} mM): "
                    f"Davies equation at limit of validity"
                )
            if ionic.gamma_target < 0.2:
                assembly.failure_modes.append(
                    f"Low activity coefficient γ = {ionic.gamma_target:.2f}: "
                    f"effective [target] significantly reduced"
                )

    return assemblies


s10.full_physics_rescore = _ionic_aware_rescore
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
# tests/test_sprint13.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint13.py", '''"""
tests/test_sprint13.py - Ionic strength and activity coefficient tests.
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

import math
from knowledge.ionic_data import (
    compute_activity_coefficient, debye_length_nm, davies_equation,
    debye_huckel_extended, pka_ionic_strength_correction,
    ionic_strength_from_mm, get_ion_diameter, dh_A, dh_B,
    _water_permittivity,
)
from core.ionic_strength import (
    compute_ionic_strength_correction, IonicStrengthAnalysis,
)
import core.thermodynamics as _thermo_mod
from core.assembly import RecognitionChemistry, InteriorDesign, InteriorSite
from core.problem import (
    Problem, TargetSpecies, Matrix, CompetingSpecies,
    ElectronicDescription, HydrationDescription, SizeDescription, Outcome,
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


# ═══════════════════════════════════════════════════════════════════════════
# Debye-Hückel constants
# ═══════════════════════════════════════════════════════════════════════════

def test_dh_constants_25c():
    """DH constants at 25°C match textbook values."""
    A = dh_A(25.0)
    B = dh_B(25.0)
    assert 0.50 < A < 0.52, f"A = {A}"
    assert 0.32 < B < 0.34, f"B = {B}"
    print(f"  + DH constants: A = {A:.4f}, B = {B:.4f}")


def test_dh_temperature_dependence():
    """A increases with temperature (more screening at higher T)."""
    A_10 = dh_A(10.0)
    A_25 = dh_A(25.0)
    A_50 = dh_A(50.0)
    assert A_10 < A_25 < A_50
    print(f"  + A(10°C)={A_10:.4f}, A(25°C)={A_25:.4f}, A(50°C)={A_50:.4f}")


def test_water_permittivity_range():
    """ε(H₂O) decreases with T: ~87 at 0°C, ~78 at 25°C, ~70 at 50°C."""
    eps_0 = _water_permittivity(0.0)
    eps_25 = _water_permittivity(25.0)
    eps_50 = _water_permittivity(50.0)
    assert 86 < eps_0 < 89
    assert 77 < eps_25 < 80
    assert eps_50 < eps_25
    print(f"  + ε(0°C)={eps_0:.1f}, ε(25°C)={eps_25:.1f}, ε(50°C)={eps_50:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# Activity coefficients
# ═══════════════════════════════════════════════════════════════════════════

def test_gamma_ideal_dilute():
    """γ = 1.0 at zero ionic strength."""
    g = compute_activity_coefficient("lead", 2.0, 0.0)
    assert g == 1.0
    print(f"  + γ(I=0) = {g:.3f}")


def test_gamma_decreases_with_I():
    """γ decreases as I increases."""
    g1 = compute_activity_coefficient("lead", 2.0, 1.0)
    g10 = compute_activity_coefficient("lead", 2.0, 10.0)
    g100 = compute_activity_coefficient("lead", 2.0, 100.0)
    assert g1 > g10 > g100
    assert g100 < 0.6, f"Pb²⁺ at 100 mM: γ = {g100:.3f}"
    print(f"  + Pb²⁺ γ: I=1→{g1:.3f}, I=10→{g10:.3f}, I=100→{g100:.3f}")


def test_gamma_charge_dependence():
    """Higher charge → lower γ (z² dependence)."""
    g_1 = compute_activity_coefficient("sodium", 1.0, 100.0)
    g_2 = compute_activity_coefficient("calcium", 2.0, 100.0)
    g_3 = compute_activity_coefficient("iron", 3.0, 100.0, oxidation_state=3)
    assert g_1 > g_2 > g_3
    print(f"  + γ(100 mM): Na⁺={g_1:.3f}, Ca²⁺={g_2:.3f}, Fe³⁺={g_3:.3f}")


def test_gamma_neutral():
    """Neutral species: γ = 1."""
    assert compute_activity_coefficient("complex", 0.0, 500.0) == 1.0
    print(f"  + γ(neutral) = 1.0")


def test_edh_vs_davies_low_I():
    """EDH and Davies agree at low I."""
    g_edh = debye_huckel_extended(2.0, 0.01, 5.0)
    g_dav = davies_equation(2.0, 0.01)
    assert abs(g_edh - g_dav) < 0.05
    print(f"  + I=10 mM: EDH={g_edh:.3f}, Davies={g_dav:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Debye length
# ═══════════════════════════════════════════════════════════════════════════

def test_debye_length():
    """Debye length: ~9.6 nm at 1 mM, ~0.96 nm at 100 mM."""
    d1 = debye_length_nm(1.0)
    d100 = debye_length_nm(100.0)
    d700 = debye_length_nm(700.0)
    assert 8.0 < d1 < 11.0, f"κ⁻¹(1 mM)={d1:.1f}"
    assert 0.7 < d100 < 1.2, f"κ⁻¹(100 mM)={d100:.2f}"
    assert d700 < 0.5
    print(f"  + κ⁻¹: 1 mM→{d1:.1f} nm, 100 mM→{d100:.2f} nm, 700 mM→{d700:.2f} nm")


# ═══════════════════════════════════════════════════════════════════════════
# pKa ionic strength correction
# ═══════════════════════════════════════════════════════════════════════════

def test_pka_correction_neutral_acid():
    """Neutral acid (HA → H⁺ + A⁻): pKa increases with I."""
    pka_0 = 4.50
    pka_100 = pka_ionic_strength_correction(pka_0, 0.0, -1.0, 100.0)
    # At 100 mM, shift should be ~ +0.1 to +0.4 units for neutral acid
    assert pka_100 > pka_0, f"pKa should increase: {pka_0} → {pka_100}"
    shift = pka_100 - pka_0
    assert 0.05 < shift < 0.5, f"shift = {shift:.3f}"
    print(f"  + Carboxylate pKa: {pka_0:.2f} → {pka_100:.2f} (Δ = {shift:+.2f})")


def test_pka_correction_diprotic():
    """Diprotic acid H₂PO₄⁻(z=-1) → HPO₄²⁻(z=-2) + H⁺: pKa shifts with I."""
    # acid charge = -1, base charge = -2
    # log_γ(-1) - log_γ(-2) - log_γ(+1)
    # = -A(1)(term) - (-A(4)(term)) - (-A(1)(term))
    # = -A(term) + 4A(term) + A(term) = +4A(term) > 0 → pKa increases
    pka_0 = 7.20
    pka_100 = pka_ionic_strength_correction(pka_0, -1.0, -2.0, 100.0)
    shift = pka_100 - pka_0
    assert abs(shift) > 0.01, f"Should see shift: {pka_0} → {pka_100}"
    assert shift > 0, f"pKa should increase for this case: shift = {shift}"
    print(f"  + H₂PO₄⁻ pKa: {pka_0:.2f} → {pka_100:.2f} (Δ = {shift:+.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# Composite ionic strength analysis
# ═══════════════════════════════════════════════════════════════════════════

def _make_pb_problem(ionic_strength_mm=50.0):
    target = TargetSpecies(
        identity="lead", formula="Pb2+", charge=2.0,
        geometry="hemidirected",
        electronic=ElectronicDescription(
            hardness_softness="borderline", electronegativity=2.33,
        ),
        hydration=HydrationDescription(
            dehydration_energy_kj_mol=1481.0, coordination_number_water=6,
        ),
        size=SizeDescription(ionic_radius_angstrom=1.19),
    )
    matrix = Matrix(
        description="AMD", ph=3.5, temperature_c=12.0,
        ionic_strength_mm=ionic_strength_mm,
        competing_species=[
            CompetingSpecies(identity="calcium", formula="Ca2+", charge=2.0, concentration_mm=5.0),
            CompetingSpecies(identity="sodium", formula="Na+", charge=1.0, concentration_mm=2.0),
            CompetingSpecies(identity="iron", formula="Fe3+", charge=3.0, concentration_mm=1.0),
        ],
    )
    outcome = Outcome(description="capture_release")
    return Problem(target=target, matrix=matrix, desired_outcome=outcome)


def _make_recognition(donors=None, donor_type=None):
    donors = donors or ["S", "S", "N"]
    donor_type = donor_type or "soft"
    return RecognitionChemistry(
        name="dithiol_amine", type="chelator",
        donor_atoms=donors, donor_type=donor_type,
        structure="dithiol_amine",
    )


def test_ionic_correction_amd():
    """AMD at 50 mM: should see significant γ correction."""
    prob = _make_pb_problem(50.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    assert ionic.gamma_target < 0.7, f"γ = {ionic.gamma_target}"
    assert ionic.debye_length_nm < 2.0
    assert abs(ionic.dG_activity) > 0.5, f"ΔG_act = {ionic.dG_activity}"
    print(f"  + AMD (50 mM): γ={ionic.gamma_target:.3f}, "
          f"κ⁻¹={ionic.debye_length_nm:.1f} nm, "
          f"ΔG_act={ionic.dG_activity:+.1f} kJ/mol")


def test_ionic_correction_seawater():
    """Seawater at 700 mM: extreme screening."""
    prob = _make_pb_problem(700.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    assert ionic.gamma_target < 0.4
    assert ionic.debye_length_nm < 0.5
    print(f"  + Seawater (700 mM): γ={ionic.gamma_target:.3f}, "
          f"κ⁻¹={ionic.debye_length_nm:.2f} nm")


def test_ionic_correction_dilute():
    """Clean river at 1 mM: minimal correction."""
    prob = _make_pb_problem(1.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    assert ionic.gamma_target > 0.8
    assert ionic.debye_length_nm > 5.0
    print(f"  + Dilute (1 mM): γ={ionic.gamma_target:.3f}, "
          f"κ⁻¹={ionic.debye_length_nm:.1f} nm")


def test_selectivity_differential():
    """Na⁺ (z=1) should have higher γ than Pb²⁺ (z=2) at same I."""
    prob = _make_pb_problem(100.0)
    rec = _make_recognition()
    ionic = compute_ionic_strength_correction(rec, prob.target, prob)
    g_pb = ionic.gamma_target
    g_na = compute_activity_coefficient("sodium", 1.0, 100.0)
    assert g_na > g_pb, f"Na⁺ γ={g_na:.3f} should > Pb²⁺ γ={g_pb:.3f}"
    print(f"  + Selectivity differential: γ(Pb²⁺)={g_pb:.3f} vs γ(Na⁺)={g_na:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Thermodynamics integration
# ═══════════════════════════════════════════════════════════════════════════

def test_thermo_with_ionic_strength():
    """ΔG_net should change when ionic strength is present."""
    prob_dilute = _make_pb_problem(0.001)
    prob_amd = _make_pb_problem(50.0)
    rec = _make_recognition()
    struct = [s for s in STRUCTURAL_OPTIONS if s.type == "mesoporous_silica"][0]
    interior = InteriorDesign(
        design_level="primary",
        sites=[InteriorSite(recognition=rec)],
        avidity_factor=1.0,
    )
    thermo_dilute = _thermo_mod.compute_thermodynamics(rec, struct, interior, prob_dilute)
    thermo_amd = _thermo_mod.compute_thermodynamics(rec, struct, interior, prob_amd)
    diff = abs(thermo_amd.dG_net - thermo_dilute.dG_net)
    assert diff > 0.1, f"Ionic strength should change ΔG: dilute={thermo_dilute.dG_net:.1f}, AMD={thermo_amd.dG_net:.1f}"
    print(f"  + ΔG: dilute={thermo_dilute.dG_net:+.1f}, AMD(50 mM)={thermo_amd.dG_net:+.1f} kJ/mol (Δ={diff:.1f})")


# ═══════════════════════════════════════════════════════════════════════════
# E2E
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_ionic_in_reports():
    """E2E: ionic strength data appears in confidence_reasoning."""
    registry = _build()
    prob = decompose("lead capture from acid mine drainage pH 3.5 ionic strength 50 mM")
    prob.matrix.ionic_strength_mm = 50.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found_ionic = False
    for r in results.assemblies:
        if "IONIC STRENGTH" in r.confidence_reasoning or "ionic" in r.confidence_reasoning.lower():
            found_ionic = True
            break
    assert found_ionic, "Ionic strength should appear in E2E report"
    print(f"  + E2E: ionic strength data in reports ({len(results.assemblies)} assemblies)")


def test_e2e_seawater_warning():
    """E2E: high ionic strength should trigger warning."""
    registry = _build()
    prob = decompose("lead capture from seawater")
    prob.matrix.ionic_strength_mm = 700.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found_warning = False
    for r in results.assemblies:
        for fm in r.failure_modes:
            if "ionic strength" in fm.lower() or "activity coefficient" in fm.lower() or "davies" in fm.lower():
                found_warning = True
    assert found_warning, "Seawater should trigger high I warning"
    print(f"  + E2E: seawater warning found")


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_dh_constants_25c,
        test_dh_temperature_dependence,
        test_water_permittivity_range,
        test_gamma_ideal_dilute,
        test_gamma_decreases_with_I,
        test_gamma_charge_dependence,
        test_gamma_neutral,
        test_edh_vs_davies_low_I,
        test_debye_length,
        test_pka_correction_neutral_acid,
        test_pka_correction_diprotic,
        test_ionic_correction_amd,
        test_ionic_correction_seawater,
        test_ionic_correction_dilute,
        test_selectivity_differential,
        test_thermo_with_ionic_strength,
        test_e2e_ionic_in_reports,
        test_e2e_seawater_warning,
    ]

    print()
    print("=" * 60)
    print("  Sprint 13: Ionic Strength & Activity Coefficients")
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
    print(f"  Sprint 13: {passed} passed, {failed} failed")
    print()
''')


# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════

print()
print("  Sprint 13 files created:")
print("    knowledge/ionic_data.py     — DH constants, Kielland table, γ calculations")
print("    core/ionic_strength.py      — Composite ionic strength analysis engine")
print("    core/ionic_integration.py   — Pipeline patches (thermo + rescore)")
print("    main.py                     — Updated with Sprint 13 import")
print("    tests/test_sprint13.py      — 18 tests")
print()
print("  Run:")
print("    python bootstrap_sprint13.py")
print("    python tests\\test_sprint13.py")
print()