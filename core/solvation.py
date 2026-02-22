"""
core/solvation.py — Sprint 20: Solvation Structure + Desolvation Thermodynamics

Replaces flat +8 kJ/mol per water with ion-specific hydration energies
from Born equation + experimental corrections. Adds water exchange
kinetics (labile vs inert classification).

Physics:
  ΔG_hydr = -(z²e²/8πε₀)(1/r_ion - 1/ε_r) × correction
  ΔG_desolv = (waters_displaced / CN) × ΔG_hydr_total
  k_ex = water exchange rate (classifies kinetic lability)
"""
from dataclasses import dataclass
import math


@dataclass
class HydrationProfile:
    """Complete hydration characterization of an ion."""
    formula: str
    hydration_energy_kj: float      # Total ΔG_hydration (negative = favorable)
    hydrated_radius_nm: float
    first_shell_waters: int
    first_shell_distance_A: float
    second_shell_waters: int
    water_exchange_rate_s: float    # k_ex in s⁻¹
    lability_class: str             # "labile", "intermediate", "inert"
    desolv_per_water_kj: float     # Cost to remove one first-shell water


# ═══════════════════════════════════════════════════════════════════════════
# HYDRATION DATABASE — experimental + Born-corrected values
# Sources: Marcus (1991), Burgess (1999), Helm & Merbach (2005)
# ═══════════════════════════════════════════════════════════════════════════

_HYDRATION_DATA = {
    # formula: (ΔG_hydr kJ/mol, r_hydr nm, CN_1st, d_M-O Å, CN_2nd, k_ex s⁻¹)
    # Alkali / alkaline earth
    "Na+":   (-405,  0.236, 6, 2.36, 12, 8.0e9),
    "K+":    (-321,  0.275, 6, 2.80, 12, 1.5e10),
    "Ba2+":  (-1305, 0.290, 8, 2.90, 16, 8.0e8),
    "Ca2+":  (-1592, 0.243, 6, 2.43, 12, 3.0e8),
    "Mg2+":  (-1920, 0.208, 6, 2.08, 12, 6.7e5),
    # First-row transition metals
    "Cr3+":  (-4560, 0.204, 6, 2.00, 12, 2.4e-6),  # INERT
    "Mn2+":  (-1845, 0.219, 6, 2.19, 12, 2.1e7),
    "Fe3+":  (-4430, 0.204, 6, 2.03, 12, 1.6e2),
    "Fe2+":  (-1946, 0.212, 6, 2.12, 12, 4.4e6),
    "Co2+":  (-2010, 0.210, 6, 2.09, 12, 3.2e6),
    "Ni2+":  (-2105, 0.206, 6, 2.06, 12, 3.2e4),
    "Cu2+":  (-2100, 0.213, 6, 2.11, 12, 5.0e9),   # Jahn-Teller labilizes
    "Zn2+":  (-2044, 0.210, 6, 2.10, 12, 2.0e7),
    # Heavier metals
    "Ag+":   (-475,  0.252, 4, 2.41, 8,  5.0e8),
    "Cd2+":  (-1807, 0.226, 6, 2.30, 12, 3.0e8),
    "Pb2+":  (-1480, 0.240, 6, 2.54, 12, 7.0e9),
    "Hg2+":  (-1824, 0.230, 6, 2.33, 12, 2.0e9),
    "Au3+":  (-4600, 0.200, 4, 2.00, 8,  1.0e6),
    "Au+":   (-610,  0.240, 2, 2.10, 4,  1.0e8),
    "Pt2+":  (-2150, 0.210, 4, 2.05, 8,  3.9e-4),  # Very inert
    "Pd2+":  (-2100, 0.210, 4, 2.05, 8,  5.6e2),
    # Oxo-cations and lanthanides
    "UO2_2+": (-1650, 0.250, 5, 2.42, 10, 1.3e6),
    "Ce3+":  (-3370, 0.253, 9, 2.54, 18, 5.0e8),
    "La3+":  (-3296, 0.258, 9, 2.58, 18, 6.0e8),
    "Al3+":  (-4660, 0.190, 6, 1.90, 12, 1.3e0),   # Very slow exchange
    "Cr2+":  (-1850, 0.215, 6, 2.15, 12, 7.0e8),   # JT labilized
}


def get_hydration_profile(formula):
    """Get full hydration profile for a metal ion."""
    if formula not in _HYDRATION_DATA:
        return _estimate_hydration(formula)

    dg, r_h, cn1, d_mo, cn2, k_ex = _HYDRATION_DATA[formula]

    if k_ex > 1e8:
        lability = "labile"
    elif k_ex > 1e2:
        lability = "intermediate"
    else:
        lability = "inert"

    desolv_per = abs(dg) / cn1

    return HydrationProfile(
        formula=formula, hydration_energy_kj=dg,
        hydrated_radius_nm=r_h, first_shell_waters=cn1,
        first_shell_distance_A=d_mo, second_shell_waters=cn2,
        water_exchange_rate_s=k_ex, lability_class=lability,
        desolv_per_water_kj=round(desolv_per, 1),
    )


def _estimate_hydration(formula):
    """Estimate hydration from Born equation for unknown ions."""
    # Parse charge from formula
    charge = 2  # default
    if "3+" in formula: charge = 3
    elif "4+" in formula: charge = 4
    elif "+" in formula: charge = 1
    elif "2-" in formula: charge = -2
    elif "-" in formula: charge = -1

    # Born equation: ΔG = -69.5 * z² / r_pm (kJ/mol, r in pm)
    r_pm = 80  # default
    dg = -69.5 * charge**2 / (r_pm / 100.0)
    cn = 6

    return HydrationProfile(
        formula=formula, hydration_energy_kj=round(dg, 0),
        hydrated_radius_nm=0.22, first_shell_waters=cn,
        first_shell_distance_A=2.1, second_shell_waters=12,
        water_exchange_rate_s=1e6, lability_class="intermediate",
        desolv_per_water_kj=round(abs(dg) / cn, 1),
    )


def compute_desolvation_energy(formula, waters_displaced, total_cn=6):
    """Compute ΔG_desolvation for displacing N waters from first shell.

    This is the key correction: replacing flat +8 kJ/mol with
    ion-specific values that range from +67 (Na+) to +777 (Al3+) per water.

    The actual penalty is scaled by what fraction of the shell is being
    replaced — full displacement is much harder than partial.
    """
    profile = get_hydration_profile(formula)

    if waters_displaced <= 0:
        return 0.0, profile

    frac_displaced = min(1.0, waters_displaced / max(1, profile.first_shell_waters))

    # Non-linear scaling: first waters are easier to displace than last
    # Cooperative disruption: disrupting >50% of shell is disproportionately costly
    if frac_displaced <= 0.5:
        scale = frac_displaced * 0.85   # Slightly easier than linear
    else:
        excess = frac_displaced - 0.5
        scale = 0.425 + excess * 1.15   # Harder above 50%

    dg_desolv = abs(profile.hydration_energy_kj) * scale

    return round(dg_desolv, 1), profile

