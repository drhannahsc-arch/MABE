"""
core/protonation_integration.py - Integrates protonation into thermodynamic pipeline.

Patches compute_thermodynamics to:
1. Compute protonation profile
2. Adjust ΔG_bind by weighting each donor by its availability
3. Recalculate chelate effect from effective denticity
4. Add protonation penalty to ΔG_net

Also patches sprint10_integration to include protonation in the physics report.
"""

import core.thermodynamics as thermo_mod
from core.protonation import compute_protonation, ProtonationProfile
from core.assembly import RecognitionChemistry, StructuralConstraint, InteriorDesign
from core.problem import Problem
from knowledge.pka_data import fraction_deprotonated, effective_pka, get_donor_pka, DONOR_PKA

import core.sprint10_integration as s10

# ── Patch thermodynamics ──────────────────────────────────────────────

_orig_compute_thermo = thermo_mod.compute_thermodynamics


def _pka_aware_thermodynamics(recognition: RecognitionChemistry,
                                structure: StructuralConstraint,
                                interior: InteriorDesign,
                                problem: Problem):
    """
    pH-aware thermodynamic calculation.
    Replaces original compute_thermodynamics.
    """
    # Get base thermodynamics
    result = _orig_compute_thermo(recognition, structure, interior, problem)
    
    # Compute protonation
    prot = compute_protonation(recognition, problem)
    
    if prot.nominal_denticity == 0 or prot.fraction_total_available >= 0.95:
        # No significant protonation effect — return base result
        result.energy_breakdown.append(f"pH {prot.ph:.1f}: all donors available (no protonation penalty)")
        return result
    
    # ── Adjust ΔG_bind ─────────────────────────────────────────────
    # Original ΔG_bind assumed all donors fully available.
    # Scale by fraction available.
    old_bind = result.dG_bind
    result.dG_bind = old_bind * prot.fraction_total_available
    bind_loss = result.dG_bind - old_bind  # positive (less favorable)
    
    # ── Adjust chelate effect ──────────────────────────────────────
    # Recalculate from effective denticity
    eff_dent = max(1, round(prot.effective_denticity))
    old_chelate = result.dG_chelate
    if eff_dent > 1:
        result.dG_chelate = -6.0 * (eff_dent - 1)
    else:
        result.dG_chelate = 0.0
    chelate_loss = result.dG_chelate - old_chelate  # positive (less favorable)
    
    # ── Add protonation penalty ────────────────────────────────────
    prot_penalty = prot.dG_protonation_penalty
    
    # ── Recalculate net ────────────────────────────────────────────
    result.dG_net = (result.dG_bind + result.dG_desolv + result.dG_preorg +
                     result.dG_chelate + result.dG_electrostatic + prot_penalty)
    
    # Update breakdown
    result.energy_breakdown.append(
        f"PROTONATION (pH {prot.ph:.1f}):"
    )
    result.energy_breakdown.append(
        f"  Effective denticity: {prot.effective_denticity:.1f} / {prot.nominal_denticity}"
    )
    result.energy_breakdown.append(
        f"  ΔG_bind adjusted: {old_bind:.1f} → {result.dG_bind:.1f} kJ/mol "
        f"({prot.fraction_total_available:.0%} donors available)"
    )
    if chelate_loss != 0:
        result.energy_breakdown.append(
            f"  Chelate effect: {old_chelate:.1f} → {result.dG_chelate:.1f} kJ/mol "
            f"(effective {eff_dent}-dentate)"
        )
    if prot_penalty > 0:
        result.energy_breakdown.append(
            f"  Protonation competition penalty: +{prot_penalty:.1f} kJ/mol"
        )
    result.energy_breakdown.append(
        f"  pH-corrected ΔG_net: {result.dG_net:.1f} kJ/mol"
    )
    
    # Recalculate K_eq and Kd
    import math
    RT = thermo_mod.R_GAS * result.temperature_k
    if abs(result.dG_net / RT) < 500:
        result.K_eq = math.exp(-result.dG_net / RT)
    else:
        result.K_eq = 1e30 if result.dG_net < 0 else 0.0
    result.predicted_kd_um = round(1e6 / result.K_eq, 3) if result.K_eq > 1e-6 else None
    
    return result


thermo_mod.compute_thermodynamics = _pka_aware_thermodynamics


# ── Patch sprint10 to include protonation in report ───────────────────

_orig_rescore = s10.full_physics_rescore


def _pka_aware_rescore(assemblies, problem):
    """Add protonation profile to physics report."""
    assemblies = _orig_rescore(assemblies, problem)
    
    for assembly in assemblies:
        prot = compute_protonation(assembly.recognition, problem)
        
        # Append protonation to confidence reasoning
        if prot.fraction_total_available < 0.95:
            assembly.confidence_reasoning += "\n\nPROTONATION:\n" + prot.summary()
        
        # Add warnings
        if prot.fraction_total_available < 0.3:
            assembly.failure_modes.append(
                f"Severe protonation loss at pH {prot.ph:.1f}: only "
                f"{prot.fraction_total_available:.0%} of donors available. "
                f"Consider acid-stable donors (thioether, thiourea, dithiocarbamate, xanthate)."
            )
        elif prot.denticity_loss > 2:
            assembly.failure_modes.append(
                f"Denticity loss: {prot.nominal_denticity} → {prot.effective_denticity:.1f} "
                f"at pH {prot.ph:.1f}. Chelate effect weakened."
            )
    
    return assemblies


s10.full_physics_rescore = _pka_aware_rescore
