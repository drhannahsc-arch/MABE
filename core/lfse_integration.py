"""
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
            assembly.confidence_reasoning += "\n\nLFSE / GEOMETRY:\n" + lfse.summary()

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
