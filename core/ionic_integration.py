"""
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
            assembly.confidence_reasoning += "\n\nIONIC STRENGTH:\n" + ionic.summary()

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
