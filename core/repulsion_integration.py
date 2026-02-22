"""
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

        assembly.confidence_reasoning += "\n\nREPULSION:\n" + repulsion.summary()

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
