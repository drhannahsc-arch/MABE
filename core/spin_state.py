"""
core/spin_state.py — Sprint 19: Spin State Predictor + Strong/Weak Field LFSE

Determines high-spin vs low-spin from d-electron count + ligand field
strength. Replaces the single LFSE table with spin-state-aware calculation.
Adds magnetic moment prediction.

Physics:
  If 10Dq > pairing energy P → low-spin
  If 10Dq < P → high-spin
  d1-d3, d8-d10: no spin-state choice (same either way)
  d4-d7: spin-state depends on field strength
"""
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class SpinStateResult:
    d_electrons: int
    spin_state: str             # "high_spin", "low_spin", "no_choice"
    unpaired_electrons: int
    magnetic_moment_bm: float   # Bohr magnetons, spin-only μ = √(n(n+2))
    lfse_oct_kj: float          # LFSE in octahedral field
    lfse_tet_kj: float          # LFSE in tetrahedral field
    lfse_sq_planar_kj: float    # LFSE in square planar field
    pairing_energy_kj: float    # Estimated P for this ion
    field_strength_10dq_kj: float  # Estimated 10Dq from ligand field
    spin_crossover_possible: bool  # 10Dq ≈ P
    rationale: str

@dataclass
class LFSEResult:
    """LFSE for a specific geometry, incorporating spin state."""
    geometry: str
    lfse_kj: float
    spin_state: str
    unpaired_electrons: int
    magnetic_moment_bm: float
    jahn_teller: str            # "none", "weak", "strong"
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# PAIRING ENERGY DATABASE (kJ/mol)
# From Figgis & Hitchman, Lever; values for aqua ions
# ═══════════════════════════════════════════════════════════════════════════

_PAIRING_ENERGY = {
    # First row d-block (kJ/mol)
    "Ti3+": 168, "V3+": 188, "V2+": 172, "Cr3+": 208, "Cr2+": 176,
    "Mn3+": 230, "Mn2+": 213, "Fe3+": 252, "Fe2+": 210,
    "Co3+": 264, "Co2+": 226, "Ni2+": 248, "Cu2+": 230,
    # Second row
    "Mo3+": 160, "Ru3+": 180, "Ru2+": 170, "Rh3+": 190, "Pd2+": 195,
    # Third row
    "Re3+": 150, "Os3+": 170, "Os2+": 160, "Ir3+": 180, "Pt2+": 200,
    "Pt4+": 220, "Au3+": 210,
}

# Default pairing energies by row and charge
_DEFAULT_P = {
    (3, 2): 210,  # 1st row, 2+
    (3, 3): 240,  # 1st row, 3+
    (4, 2): 170,  # 2nd row, 2+
    (4, 3): 180,  # 2nd row, 3+
    (5, 2): 160,  # 3rd row, 2+
    (5, 3): 170,  # 3rd row, 3+
}


# ═══════════════════════════════════════════════════════════════════════════
# 10Dq ESTIMATION FROM LIGAND FIELD STRENGTHS
# Uses average Dq from the donor ligand templates
# ═══════════════════════════════════════════════════════════════════════════

# Reference 10Dq(oct) for aqua complexes in kJ/mol
_10DQ_AQUA = {
    "Ti3+": 243, "V3+": 216, "V2+": 147, "Cr3+": 208, "Cr2+": 166,
    "Mn3+": 252, "Mn2+": 90, "Fe3+": 164, "Fe2+": 124,
    "Co3+": 220, "Co2+": 111, "Ni2+": 102, "Cu2+": 151,
    "Ru2+": 240, "Rh3+": 340, "Pd2+": 310, "Ir3+": 340,
    "Pt2+": 370, "Pt4+": 420, "Au3+": 380,
}

# Spectrochemical series multipliers relative to water (f factor)
_FIELD_MULTIPLIER = {
    "water": 1.00,
    # Weak field
    "hydroxide": 0.72, "fluoride": 0.90, "chloride": 0.78,
    "bromide": 0.72, "iodide": 0.60, "sulfide": 0.65,
    # Moderate field
    "carboxylate": 0.85, "phosphonate": 0.80,
    "phenolate": 0.88, "catechol": 0.90,
    # Borderline
    "primary_amine": 1.25, "tertiary_amine": 1.20,
    "imidazole": 1.30, "pyridine": 1.35, "bipyridyl": 1.90,
    "hydroxamate": 0.95,
    # Strong field
    "iminodiacetate": 1.15, "salicylaldehyde_imine": 1.35,
    "cyanide": 2.50, "carbonyl": 2.60, "nitrosyl": 2.70,
    "phosphine": 1.60,
    # Soft donors
    "thiolate": 0.75, "thioether": 0.80, "dithiocarbamate": 0.78,
    "thiourea": 0.82, "crown_ether_O": 0.70,
}


def _get_10dq(metal_formula, ligand_names):
    """Estimate 10Dq for a metal with given ligands.

    Uses spectrochemical series: 10Dq(complex) = 10Dq(aqua) × f_avg
    where f_avg is the average field multiplier of the ligands.
    """
    base_10dq = _10DQ_AQUA.get(metal_formula, 120.0)

    if not ligand_names:
        return base_10dq

    multipliers = []
    for name in ligand_names:
        m = _FIELD_MULTIPLIER.get(name, 1.0)
        multipliers.append(m)

    f_avg = sum(multipliers) / len(multipliers) if multipliers else 1.0

    # 2nd/3rd row metals have inherently larger 10Dq (~1.5x, ~2.0x)
    row_factor = 1.0
    if metal_formula in ("Ru2+", "Rh3+", "Mo3+", "Pd2+"):
        row_factor = 1.45
    elif metal_formula in ("Ir3+", "Pt2+", "Pt4+", "Os2+", "Os3+", "Re3+", "Au3+"):
        row_factor = 1.75

    return base_10dq * f_avg * row_factor


def _get_pairing_energy(metal_formula):
    """Get pairing energy for a metal ion."""
    if metal_formula in _PAIRING_ENERGY:
        return _PAIRING_ENERGY[metal_formula]
    # Estimate from defaults
    charge = 2
    if "3+" in metal_formula or "3+$" in metal_formula:
        charge = 3
    row = 3  # Default to first row
    return _DEFAULT_P.get((row, charge), 210)


# ═══════════════════════════════════════════════════════════════════════════
# LFSE TABLES — HIGH SPIN AND LOW SPIN
# Values in units of Dq (multiply by 10Dq/10 = Dq to get kJ/mol)
# Format: (LFSE_oct_Dq, unpaired_e_oct, LFSE_tet_Dq, unpaired_e_tet)
# ═══════════════════════════════════════════════════════════════════════════

# Octahedral LFSE in Dq units
_HS_OCT = {
    0: (0, 0), 1: (-4, 1), 2: (-8, 2), 3: (-12, 3),
    4: (-6, 4), 5: (0, 5), 6: (-4, 4), 7: (-8, 3),
    8: (-12, 2), 9: (-6, 1), 10: (0, 0),
}

_LS_OCT = {
    0: (0, 0), 1: (-4, 1), 2: (-8, 2), 3: (-12, 3),
    4: (-16, 2), 5: (-20, 1), 6: (-24, 0), 7: (-18, 1),
    8: (-12, 2), 9: (-6, 1), 10: (0, 0),
}

_HS_TET = {
    0: (0, 0), 1: (-2.67, 1), 2: (-5.34, 2), 3: (-3.56, 3),
    4: (-1.78, 4), 5: (0, 5), 6: (-2.67, 4), 7: (-5.34, 3),
    8: (-3.56, 2), 9: (-1.78, 1), 10: (0, 0),
}

# Square planar LFSE — only relevant for d8 (and some d9, d7)
_SQ_PLANAR = {
    7: (-24.56, 1), 8: (-24.56, 0), 9: (-21.89, 1), 10: (0, 0),
}

# Jahn-Teller distortion strength
_JT_STRENGTH = {
    # (d_electrons, spin_state): "none" / "weak" / "strong"
    (4, "high_spin"): "strong",   # eg: 1 electron in d(z2) or d(x2-y2)
    (7, "low_spin"): "strong",
    (9, "high_spin"): "strong",   # Cu2+ classic
    (9, "low_spin"): "strong",
    (1, "high_spin"): "weak",
    (2, "high_spin"): "weak",
    (6, "high_spin"): "weak",
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

def predict_spin_state(metal_formula, d_electrons, ligand_names=None):
    """Predict spin state from metal d-count and ligand field strengths.

    Args:
        metal_formula: e.g. "Ni2+", "Fe3+", "Au3+"
        d_electrons: number of d electrons
        ligand_names: list of ligand names from donor_chemistry.py

    Returns:
        SpinStateResult with spin state, magnetic moment, and LFSE values
    """
    ligand_names = ligand_names or []

    # d0-d3 and d8-d10: no spin-state ambiguity
    if d_electrons <= 3 or d_electrons >= 8:
        return _no_choice_result(metal_formula, d_electrons, ligand_names)

    # d4-d7: compare 10Dq vs pairing energy
    ten_dq = _get_10dq(metal_formula, ligand_names)
    P = _get_pairing_energy(metal_formula)
    dq = ten_dq / 10.0

    crossover = abs(ten_dq - P) < P * 0.15  # Within 15% = spin crossover region

    if ten_dq > P:
        # Low spin
        lfse_oct_dq, unpaired_oct = _LS_OCT[d_electrons]
        spin = "low_spin"
    else:
        # High spin
        lfse_oct_dq, unpaired_oct = _HS_OCT[d_electrons]
        spin = "high_spin"

    lfse_tet_dq, unpaired_tet = _HS_TET[d_electrons]  # Tet is almost always HS

    lfse_oct = lfse_oct_dq * dq
    lfse_tet = lfse_tet_dq * (ten_dq * 4 / 90)  # 10Dq(tet) ≈ 4/9 × 10Dq(oct)

    # Square planar (if applicable)
    lfse_sq = 0.0
    if d_electrons in _SQ_PLANAR:
        sq_dq, _ = _SQ_PLANAR[d_electrons]
        lfse_sq = sq_dq * dq

    mu = math.sqrt(unpaired_oct * (unpaired_oct + 2))

    rationale = (f"d{d_electrons} {metal_formula}: 10Dq={ten_dq:.0f} kJ/mol, "
                 f"P={P:.0f} kJ/mol → {'10Dq > P → low-spin' if spin == 'low_spin' else '10Dq < P → high-spin'}. "
                 f"Ligands: {', '.join(ligand_names[:4]) if ligand_names else 'aqua'}.")
    if crossover:
        rationale += " Near spin-crossover boundary."

    return SpinStateResult(
        d_electrons=d_electrons, spin_state=spin,
        unpaired_electrons=unpaired_oct,
        magnetic_moment_bm=round(mu, 2),
        lfse_oct_kj=round(lfse_oct, 1),
        lfse_tet_kj=round(lfse_tet, 1),
        lfse_sq_planar_kj=round(lfse_sq, 1),
        pairing_energy_kj=P, field_strength_10dq_kj=round(ten_dq, 1),
        spin_crossover_possible=crossover,
        rationale=rationale,
    )


def _no_choice_result(metal_formula, d_electrons, ligand_names):
    """For d0-d3, d8-d10 where spin state is unambiguous."""
    ten_dq = _get_10dq(metal_formula, ligand_names)
    dq = ten_dq / 10.0

    lfse_oct_dq, unpaired_oct = _HS_OCT[d_electrons]
    lfse_tet_dq, unpaired_tet = _HS_TET[d_electrons]

    lfse_oct = lfse_oct_dq * dq
    lfse_tet = lfse_tet_dq * (ten_dq * 4 / 90)
    lfse_sq = 0.0
    if d_electrons in _SQ_PLANAR:
        sq_dq, unpaired_sq = _SQ_PLANAR[d_electrons]
        lfse_sq = sq_dq * dq
        unpaired_oct = unpaired_sq if d_electrons == 8 else unpaired_oct

    # d8 square planar is always low-spin (0 unpaired)
    if d_electrons == 8:
        unpaired_oct = 2  # Octahedral d8 has 2 unpaired
        spin = "no_choice"
    else:
        spin = "no_choice"

    mu = math.sqrt(unpaired_oct * (unpaired_oct + 2))

    return SpinStateResult(
        d_electrons=d_electrons, spin_state=spin,
        unpaired_electrons=unpaired_oct,
        magnetic_moment_bm=round(mu, 2),
        lfse_oct_kj=round(lfse_oct, 1),
        lfse_tet_kj=round(lfse_tet, 1),
        lfse_sq_planar_kj=round(lfse_sq, 1),
        pairing_energy_kj=_get_pairing_energy(metal_formula),
        field_strength_10dq_kj=round(ten_dq, 1),
        spin_crossover_possible=False,
        rationale=(f"d{d_electrons} {metal_formula}: no spin-state choice "
                   f"(same ground state). 10Dq={ten_dq:.0f} kJ/mol. "
                   f"LFSE(oct)={lfse_oct:.1f} kJ/mol."),
    )


def compute_lfse_for_geometry(metal_formula, d_electrons, geometry,
                               ligand_names=None):
    """Get LFSE for a specific geometry, with spin state awareness.

    This replaces the Sprint 12 fixed LFSE lookup with a field-strength-
    dependent calculation.
    """
    ss = predict_spin_state(metal_formula, d_electrons, ligand_names)

    if "square_planar" in geometry:
        if d_electrons in _SQ_PLANAR:
            sq_dq, unpaired = _SQ_PLANAR[d_electrons]
            dq = ss.field_strength_10dq_kj / 10.0
            lfse = sq_dq * dq
            jt = "none"  # Square planar is already distorted
            return LFSEResult(geometry, round(lfse, 1), "low_spin" if d_electrons == 8 else ss.spin_state,
                              unpaired if d_electrons == 8 else ss.unpaired_electrons,
                              round(math.sqrt(unpaired * (unpaired + 2)), 2),
                              jt, f"Square planar d{d_electrons}")
        return LFSEResult(geometry, 0.0, ss.spin_state, ss.unpaired_electrons,
                          ss.magnetic_moment_bm, "none")

    if "tetrahedral" in geometry:
        jt_key = (d_electrons, ss.spin_state)
        jt = _JT_STRENGTH.get(jt_key, "none")
        return LFSEResult(geometry, ss.lfse_tet_kj, "high_spin",  # Tet is always HS
                          _HS_TET[d_electrons][1],
                          round(math.sqrt(_HS_TET[d_electrons][1] * (_HS_TET[d_electrons][1] + 2)), 2),
                          jt)

    if "linear" in geometry:
        # Linear: LFSE ≈ 0 for most cases, dominated by orbital preference
        return LFSEResult(geometry, 0.0, ss.spin_state, ss.unpaired_electrons,
                          ss.magnetic_moment_bm, "none", "Linear geometry — LFSE minimal")

    # Default to octahedral (includes tetragonal_elongated, hemidirected, etc.)
    jt_key = (d_electrons, ss.spin_state)
    jt = _JT_STRENGTH.get(jt_key, "none")
    if jt == "none" and ss.spin_state == "no_choice":
        # Try both HS and LS keys for no_choice metals
        jt = _JT_STRENGTH.get((d_electrons, "high_spin"),
             _JT_STRENGTH.get((d_electrons, "low_spin"), "none"))
    return LFSEResult(geometry, ss.lfse_oct_kj, ss.spin_state,
                      ss.unpaired_electrons, ss.magnetic_moment_bm, jt)

