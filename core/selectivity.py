"""
core/selectivity.py — Sprint 33: Selectivity Scoring

For each binder design, computes binding ΔG and Kd for common
interferent ions. Reports selectivity ratios (Kd_interferent / Kd_target).

Interferent panels:
  drinking_water: Ca2+, Mg2+, Na+, K+, Fe3+, Zn2+, Cu2+, Mn2+
  seawater:       Na+, Mg2+, Ca2+, K+, Sr2+, Ba2+
  acid_mine:      Fe3+, Fe2+, Cu2+, Zn2+, Mn2+, Al3+, Cd2+
  nuclear_waste:  Cs+, Sr2+, Ba2+, Ca2+, Na+, K+, La3+
  soil:           Ca2+, Mg2+, Fe3+, Al3+, Mn2+, Zn2+
"""
from dataclasses import dataclass, field
import math

from core.physics_integration import compute_enhanced_thermodynamics
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)
from core.coordination_generator import (
    METAL_D_ELECTRONS, METAL_HSAB_SOFTNESS, _IONIC_RADII,
)


@dataclass
class InterferentResult:
    """Binding result for a single interferent."""
    formula: str
    charge: int
    dg_net_kj: float
    predicted_kd_uM: float
    selectivity_ratio: float    # Kd(interferent) / Kd(target). >1 = selective
    selectivity_class: str      # "excellent", "good", "moderate", "poor", "none"
    binding_note: str = ""


@dataclass
class SelectivityProfile:
    """Full selectivity analysis for a binder design."""
    target_formula: str
    target_kd_uM: float
    interferents: list            # List of InterferentResult
    worst_interferent: str        # Formula of most competitive interferent
    worst_selectivity_ratio: float
    overall_selectivity_class: str  # Based on worst case
    selectivity_score: float      # 0-100
    deployment_matrix: str        # Which panel was used
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# INTERFERENT PANELS
# ═══════════════════════════════════════════════════════════════════════════

_PANELS = {
    "drinking_water": [
        ("Ca2+", 2, 0), ("Mg2+", 2, 0), ("Na+", 1, 0), ("K+", 1, 0),
        ("Fe3+", 3, 5), ("Zn2+", 2, 10), ("Cu2+", 2, 9), ("Mn2+", 2, 5),
    ],
    "seawater": [
        ("Na+", 1, 0), ("Mg2+", 2, 0), ("Ca2+", 2, 0), ("K+", 1, 0),
        ("Sr2+", 2, 0), ("Ba2+", 2, 0),
    ],
    "acid_mine": [
        ("Fe3+", 3, 5), ("Fe2+", 2, 6), ("Cu2+", 2, 9), ("Zn2+", 2, 10),
        ("Mn2+", 2, 5), ("Al3+", 3, 0), ("Cd2+", 2, 10),
    ],
    "nuclear_waste": [
        ("Na+", 1, 0), ("K+", 1, 0), ("Ca2+", 2, 0),
        ("Ba2+", 2, 0), ("La3+", 3, 0),
    ],
    "soil": [
        ("Ca2+", 2, 0), ("Mg2+", 2, 0), ("Fe3+", 3, 5),
        ("Al3+", 3, 0), ("Mn2+", 2, 5), ("Zn2+", 2, 10),
    ],
}

# Typical concentrations in environmental water (µM) for context
_TYPICAL_CONC = {
    "Ca2+": 1000, "Mg2+": 500, "Na+": 2000, "K+": 100,
    "Fe3+": 10, "Zn2+": 5, "Cu2+": 2, "Mn2+": 10,
    "Al3+": 5, "Cd2+": 0.05, "Ba2+": 1, "Sr2+": 5,
    "La3+": 0.01, "Fe2+": 50, "Pb2+": 0.5, "Hg2+": 0.01,
}


def _classify_selectivity(ratio):
    """Classify selectivity ratio."""
    if ratio >= 1000:
        return "excellent"
    elif ratio >= 100:
        return "good"
    elif ratio >= 10:
        return "moderate"
    elif ratio >= 2:
        return "poor"
    else:
        return "none"


def compute_selectivity(target_formula, target_kd_uM, target_charge,
                          recognition, structure, interior, matrix,
                          panel="drinking_water", target_dg_kj=None):
    """Compute selectivity against an interferent panel.

    Runs compute_enhanced_thermodynamics for each interferent using
    the SAME binder (recognition + structure + interior) but different
    target metal. This is the key insight: the binder is fixed,
    the metal changes.
    """
    panel_ions = _PANELS.get(panel, _PANELS["drinking_water"])

    # Filter out the target itself from interferents
    panel_ions = [(f, c, d) for f, c, d in panel_ions if f != target_formula]

    results = []

    for int_formula, int_charge, int_d_electrons in panel_ions:
        int_softness = METAL_HSAB_SOFTNESS.get(int_formula, 0.3)
        int_radius = _IONIC_RADII.get(int_formula, 80)
        int_hr = (int_radius + 140) / 1000.0

        # Build problem for interferent
        int_prob = Problem(
            target=TargetSpecies(
                identity=int_formula, formula=int_formula,
                charge=int_charge, d_electrons=int_d_electrons,
                hsab_softness=int_softness, ionic_radius_pm=int_radius,
                hydrated_radius_nm=int_hr,
                coordination_number=6),
            matrix=matrix)

        try:
            int_thermo = compute_enhanced_thermodynamics(
                recognition, structure, interior, int_prob)
            int_kd = int_thermo.predicted_kd_um
            int_dg = int_thermo.dg_net_kj
        except Exception:
            # If calculation fails, assume no binding
            int_kd = 1e12
            int_dg = 100.0

        # Selectivity ratio based on ΔG difference
        # When both Kd underflow to 0, use ΔG directly:
        # ΔΔG = ΔG(interferent) - ΔG(target_from_thermo)
        # Selectivity ratio = exp(ΔΔG / RT)
        R = 8.314e-3  # kJ/(mol·K)
        T = 298.15
        RT = R * T  # 2.479 kJ/mol

        # We need target ΔG. Use passed value if available.
        if target_dg_kj is not None:
            target_dg = target_dg_kj
        elif target_kd_uM > 1e-10:
            target_dg = RT * math.log(target_kd_uM * 1e-6)  # Convert µM to M
        else:
            target_dg = -100.0  # Very strong binding placeholder

        # ΔΔG: positive means interferent binds WEAKER (good selectivity)
        ddg = int_dg - target_dg  # Both negative; if int less negative → ddg > 0

        if ddg > 0:
            # Interferent binds weaker → selective
            ratio = min(1e6, math.exp(ddg / RT))
        elif ddg > -RT:
            # Near-equal binding
            ratio = math.exp(ddg / RT)
        else:
            # Interferent binds stronger → no selectivity
            ratio = max(0.01, math.exp(ddg / RT))

        sel_class = _classify_selectivity(ratio)

        # Note about concentration-weighted selectivity
        conc_target = _TYPICAL_CONC.get(target_formula, 1.0)
        conc_int = _TYPICAL_CONC.get(int_formula, 100.0)
        note = ""
        if conc_int / max(0.001, conc_target) > 100 and ratio < 100:
            note = (f"Interferent [{int_formula}] is {conc_int/max(0.001,conc_target):.0f}× "
                    f"more concentrated than target — effective selectivity worse than ratio suggests")

        results.append(InterferentResult(
            formula=int_formula, charge=int_charge,
            dg_net_kj=round(int_dg, 1),
            predicted_kd_uM=round(int_kd, 2) if int_kd < 1e6 else int_kd,
            selectivity_ratio=round(ratio, 1) if ratio < 1e6 else ratio,
            selectivity_class=sel_class,
            binding_note=note,
        ))

    # Overall assessment
    if results:
        worst = min(results, key=lambda r: r.selectivity_ratio)
        worst_formula = worst.formula
        worst_ratio = worst.selectivity_ratio
        overall = _classify_selectivity(worst_ratio)
    else:
        worst_formula = "none"
        worst_ratio = 1e6
        overall = "excellent"

    # Score: 0-100 based on geometric mean of all ratios
    if results:
        log_ratios = [math.log10(max(0.01, min(1e6, r.selectivity_ratio))) for r in results]
        avg_log = sum(log_ratios) / len(log_ratios)
        # avg_log: -2 (ratio 0.01) → 0, 0 (ratio 1) → 0, 3 (ratio 1000) → 100
        score = max(0, min(100, (avg_log + 1) * 25))  # -1→0, 0→25, 3→100
    else:
        score = 100

    notes_parts = []
    poor = [r for r in results if r.selectivity_class in ("poor", "none")]
    if poor:
        names = ", ".join(r.formula for r in poor)
        notes_parts.append(f"Poor selectivity vs: {names}")
    conc_warnings = [r for r in results if r.binding_note]
    if conc_warnings:
        notes_parts.append(f"{len(conc_warnings)} interferent(s) at much higher concentration")

    return SelectivityProfile(
        target_formula=target_formula,
        target_kd_uM=target_kd_uM,
        interferents=results,
        worst_interferent=worst_formula,
        worst_selectivity_ratio=worst_ratio,
        overall_selectivity_class=overall,
        selectivity_score=round(score, 1),
        deployment_matrix=panel,
        notes="; ".join(notes_parts),
    )


def print_selectivity(profile):
    """Pretty-print selectivity analysis."""
    print(f"\n  SELECTIVITY ({profile.overall_selectivity_class}, score={profile.selectivity_score:.0f}/100)")
    print(f"  {'─'*60}")
    print(f"  Target: {profile.target_formula} Kd={profile.target_kd_uM:.2f} µM")
    print(f"  Panel:  {profile.deployment_matrix}")
    print(f"  {'Interferent':12s} {'Kd(µM)':>12s} {'Ratio':>10s} {'Class':>12s}")
    print(f"  {'─'*48}")
    for r in sorted(profile.interferents, key=lambda x: x.selectivity_ratio):
        kd_str = f"{r.predicted_kd_uM:.1f}" if r.predicted_kd_uM < 1e6 else ">10⁶"
        ratio_str = f"{r.selectivity_ratio:.0f}×" if r.selectivity_ratio < 1e6 else ">10⁶×"
        flag = " ⚠" if r.selectivity_class in ("poor", "none") else ""
        print(f"  {r.formula:12s} {kd_str:>12s} {ratio_str:>10s} {r.selectivity_class:>12s}{flag}")
        if r.binding_note:
            print(f"  {'':12s}  └ {r.binding_note}")
    if profile.notes:
        print(f"\n  ⚠ {profile.notes}")
    print()

