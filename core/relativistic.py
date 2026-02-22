"""
core/relativistic.py — Sprint 27: Relativistic Effects on Heavy Elements

Why Au-S bonds are disproportionately strong: 6s orbital contraction
from scalar relativistic effects makes Au+ a much better Lewis acid
than Ag+ despite being in the same group. Why Pb2+ has a hemidirected
lone pair: relativistic stabilization of 6s2 (inert pair effect).

Physics:
  Scalar relativistic contraction: r_rel = r_nr × (1 - (Z/c)² × f_orbital)
  6s contraction in Au: ~20% orbital shrinkage → enhanced Lewis acidity
  Inert pair effect: 6s² stabilized in Tl+, Pb2+, Bi3+
  Spin-orbit coupling: splits d-orbital manifold, affects LFSE for 5d metals
  Gold anomaly: explains why Au behaves more like a halogen than Ag
"""
from dataclasses import dataclass
import math


@dataclass
class RelativisticProfile:
    """Relativistic corrections for a heavy element."""
    formula: str
    atomic_number: int
    period: int                     # 4=3d, 5=4d, 6=5d/6p
    s_contraction_pct: float        # % contraction of valence s orbital
    d_expansion_pct: float          # % expansion of d orbitals (indirect)
    inert_pair_stabilization_kj: float  # Energy stabilization of ns² pair
    spin_orbit_coupling_cm1: float  # ζ(d) spin-orbit parameter
    lewis_acidity_correction: float # Multiplier on binding energy (>1 for contracted s)
    lone_pair_stereochemistry: str  # "none", "hemidirected", "holodirected"
    relativistic_significance: str  # "negligible", "moderate", "large", "dominant"
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# RELATIVISTIC PARAMETERS DATABASE
# Sources: Pyykkö (1988, 2012), Autschbach (2012), Schwerdtfeger (2002)
# ═══════════════════════════════════════════════════════════════════════════

_ATOMIC_NUMBERS = {
    "Na+": 11, "K+": 19, "Ca2+": 20, "Ba2+": 56,
    "Cr3+": 24, "Mn2+": 25, "Fe2+": 26, "Fe3+": 26,
    "Co2+": 27, "Co3+": 27, "Ni2+": 28, "Cu2+": 29, "Cu+": 29,
    "Zn2+": 30, "Ag+": 47, "Cd2+": 48, "Pd2+": 46, "Ru3+": 44,
    "Rh3+": 45, "Mo3+": 42,
    "La3+": 57, "Ce3+": 58,
    "Au+": 79, "Au3+": 79, "Hg2+": 80, "Tl+": 81,
    "Pb2+": 82, "Bi3+": 83, "Pt2+": 78, "Pt4+": 78,
    "Ir3+": 77, "Os3+": 76, "Re3+": 75,
    "UO2_2+": 92, "Th4+": 90,
}

_RELATIVISTIC_DATA = {
    # formula: (s_contract%, d_expand%, inert_pair_kj, ζ_d cm⁻¹,
    #           lewis_correction, lone_pair, significance)

    # Period 4 (3d): negligible relativistic effects
    "Cr3+":  (0.5, 0.0,   0, 230,   1.00, "none", "negligible"),
    "Mn2+":  (0.5, 0.0,   0, 255,   1.00, "none", "negligible"),
    "Fe2+":  (0.5, 0.0,   0, 400,   1.00, "none", "negligible"),
    "Fe3+":  (0.5, 0.0,   0, 460,   1.00, "none", "negligible"),
    "Co2+":  (0.5, 0.0,   0, 515,   1.00, "none", "negligible"),
    "Ni2+":  (0.5, 0.0,   0, 600,   1.00, "none", "negligible"),
    "Cu2+":  (0.6, 0.1,   0, 830,   1.00, "none", "negligible"),
    "Cu+":   (0.6, 0.1,   0, 830,   1.00, "none", "negligible"),
    "Zn2+":  (0.6, 0.1,   0,   0,   1.00, "none", "negligible"),

    # Period 5 (4d): moderate effects
    "Mo3+":  (3.0, 1.0,   0, 750,   1.05, "none", "moderate"),
    "Ru3+":  (3.5, 1.2,   0, 880,   1.05, "none", "moderate"),
    "Rh3+":  (3.5, 1.2,   0, 1000,  1.05, "none", "moderate"),
    "Pd2+":  (4.0, 1.5,   0, 1200,  1.10, "none", "moderate"),
    "Ag+":   (5.0, 2.0,  30, 1350,  1.10, "none", "moderate"),
    "Cd2+":  (5.0, 2.0,  25,    0,  1.05, "none", "moderate"),

    # Period 6 (5d/6s/6p): LARGE relativistic effects
    "Re3+":  (10, 5.0,    0, 2200,  1.15, "none", "large"),
    "Os3+":  (10, 5.0,    0, 2500,  1.15, "none", "large"),
    "Ir3+":  (11, 5.5,    0, 2800,  1.20, "none", "large"),
    "Pt2+":  (12, 6.0,    0, 3500,  1.25, "none", "large"),
    "Pt4+":  (12, 6.0,    0, 3500,  1.25, "none", "large"),
    "Au+":   (15, 8.0,   60, 4000,  1.40, "none", "dominant"),  # THE gold anomaly
    "Au3+":  (15, 8.0,   60, 4000,  1.35, "none", "dominant"),
    "Hg2+":  (14, 7.0,   80,    0,  1.15, "none", "large"),
    "Tl+":   (13, 6.0,  120,    0,  1.00, "hemidirected", "dominant"),
    "Pb2+":  (12, 5.5,  100,    0,  1.00, "hemidirected", "dominant"),
    "Bi3+":  (11, 5.0,   80,    0,  1.00, "hemidirected", "large"),

    # Actinides: very large effects
    "UO2_2+": (18, 9.0,   0, 2000,  1.20, "none", "dominant"),
    "Th4+":   (17, 8.5,   0, 1800,  1.15, "none", "large"),

    # Lanthanides: moderate (4f shielded)
    "La3+":  (3.0, 1.0,   0,  650,  1.00, "none", "moderate"),
    "Ce3+":  (3.0, 1.0,   0,  700,  1.00, "none", "moderate"),
}


def get_relativistic_profile(formula):
    """Get relativistic correction profile for a metal ion."""
    Z = _ATOMIC_NUMBERS.get(formula, 30)  # Default to Zn
    period = 4
    if Z > 56: period = 6
    elif Z > 36: period = 5
    elif Z > 86: period = 7  # Actinides

    data = _RELATIVISTIC_DATA.get(formula)
    if data is None:
        return _estimate_relativistic(formula, Z, period)

    s_con, d_exp, inert, soc, lewis, lone, sig = data

    notes_parts = []
    if s_con > 10:
        notes_parts.append(f"Strong 6s contraction ({s_con}%): enhanced Lewis acidity")
    if inert > 50:
        notes_parts.append(f"Inert pair effect ({inert} kJ/mol): 6s² resists oxidation")
    if lone != "none":
        notes_parts.append(f"Stereochemically active lone pair → {lone} geometry")
    if formula in ("Au+", "Au3+"):
        notes_parts.append("Gold anomaly: relativistic 6s contraction makes Au "
                           "uniquely strong Lewis acid vs Ag")

    return RelativisticProfile(
        formula=formula, atomic_number=Z, period=period,
        s_contraction_pct=s_con, d_expansion_pct=d_exp,
        inert_pair_stabilization_kj=inert,
        spin_orbit_coupling_cm1=soc,
        lewis_acidity_correction=lewis,
        lone_pair_stereochemistry=lone,
        relativistic_significance=sig,
        notes="; ".join(notes_parts),
    )


def _estimate_relativistic(formula, Z, period):
    """Estimate relativistic effects from atomic number."""
    # Approximate: s contraction ~ (Z/137)² × 100%
    alpha = Z / 137.036
    s_con = round(alpha**2 * 100 * 2.5, 1)  # Empirical scaling
    d_exp = round(s_con * 0.5, 1)

    lewis = 1.0 + s_con / 100.0
    inert = max(0, (s_con - 5) * 10)  # Only significant if >5%

    soc = round(Z**2 * 0.5)  # Very rough

    sig = "negligible"
    if s_con > 10: sig = "dominant"
    elif s_con > 5: sig = "large"
    elif s_con > 2: sig = "moderate"

    lone = "none"
    if inert > 50 and period >= 6:
        lone = "hemidirected"

    return RelativisticProfile(
        formula=formula, atomic_number=Z, period=period,
        s_contraction_pct=s_con, d_expansion_pct=d_exp,
        inert_pair_stabilization_kj=inert,
        spin_orbit_coupling_cm1=soc,
        lewis_acidity_correction=round(lewis, 2),
        lone_pair_stereochemistry=lone,
        relativistic_significance=sig,
        notes=f"Estimated from Z={Z}. Verify with DFT for quantitative use.",
    )


def correct_binding_energy(dg_binding_kj, metal_formula):
    """Apply relativistic Lewis acidity correction to binding energy.

    For 5d/6s metals, the contracted 6s orbital makes the ion a stronger
    Lewis acid than non-relativistic calculations would predict.

    Returns corrected ΔG and the correction factor.
    """
    profile = get_relativistic_profile(metal_formula)
    corrected = dg_binding_kj * profile.lewis_acidity_correction
    return round(corrected, 2), profile.lewis_acidity_correction


def predict_geometry_from_lone_pair(metal_formula, coordination_number):
    """Predict whether lone pair is stereochemically active.

    For Pb2+, Tl+, Bi3+ with 6s² inert pair:
    - Low CN (≤4): hemidirected (lone pair occupies one face)
    - High CN (≥6): holodirected (lone pair becomes spherical)
    """
    profile = get_relativistic_profile(metal_formula)

    if profile.lone_pair_stereochemistry == "none":
        return "holodirected", 0.0, "No stereochemically active lone pair"

    if coordination_number <= 4:
        return "hemidirected", profile.inert_pair_stabilization_kj, \
            f"CN={coordination_number}: lone pair stereoactive, creates void in coordination sphere"
    elif coordination_number <= 6:
        # Transition zone
        frac = (coordination_number - 4) / 2.0
        stabilization = profile.inert_pair_stabilization_kj * (1 - frac)
        return "hemidirected", round(stabilization, 1), \
            f"CN={coordination_number}: partially hemidirected"
    else:
        return "holodirected", 0.0, \
            f"CN={coordination_number}: lone pair becomes spherically distributed"


def compute_spin_orbit_splitting(metal_formula, d_electrons, lfse_kj):
    """Compute spin-orbit coupling correction to LFSE.

    For 5d metals, SOC is large enough to significantly mix states
    and modify the effective LFSE. For 3d metals, it's perturbative.

    Correction: δ(LFSE) ≈ ζ²/(10Dq) for first-order, but simplified
    to a percentage correction based on ζ/10Dq ratio.
    """
    profile = get_relativistic_profile(metal_formula)
    soc = profile.spin_orbit_coupling_cm1

    if soc == 0 or abs(lfse_kj) < 1.0:
        return 0.0, "Spin-orbit coupling negligible"

    # Convert ζ to kJ/mol: 1 cm⁻¹ = 0.01196 kJ/mol
    soc_kj = soc * 0.01196

    # SOC correction ≈ ζ² / (10Dq) as fraction of LFSE
    # For 3d: ζ ~ 200-800 cm⁻¹, correction < 5%
    # For 5d: ζ ~ 2000-4000 cm⁻¹, correction 10-30%
    correction_fraction = (soc_kj / max(1.0, abs(lfse_kj)))**0.5 * 0.15

    correction_kj = lfse_kj * correction_fraction
    sign = "reduces" if correction_kj > 0 else "enhances"

    return round(correction_kj, 2), \
        f"SOC ζ={soc} cm⁻¹: {sign} LFSE by {abs(correction_fraction)*100:.1f}%"

