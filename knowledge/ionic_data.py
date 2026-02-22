"""
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
