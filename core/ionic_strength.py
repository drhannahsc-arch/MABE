"""
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
        return "\n".join(parts)


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
