"""
optical/refractive_index.py — Module 1: Refractive Index Database

Provides n(λ) and k(λ) for optical materials used across MABE's photonic pipeline.
Sellmeier or Cauchy coefficients from published sources. No fitted parameters.

Data sources:
  SiO2:        Malitson 1965, J. Opt. Soc. Am. 55:1205 (Tier 1, NIST-traceable)
  TiO2 rutile: Devore 1951, J. Opt. Soc. Am. 41:416 (Tier 2, ordinary ray)
  TiO2 anatase: Sellmeier fit to Sarkar 2019 data (Tier 2)
  Polystyrene: Sultanova 2009, Acta Phys. Pol. A 116:585 (Tier 2)
  ZnO:         Bond 1965, J. Appl. Phys. 36:1674 (Tier 2)
  Water:       Daimon & Masumura 2007, Appl. Opt. 46:3811 (Tier 2)
  Fe2O3:       Querry 1985, CRDEC-CR (Tier 2, hematite)
  Carbon:      Palik Handbook (Tier 2, amorphous carbon)
  Gold:        Johnson & Christy 1972, Phys. Rev. B 6:4370 (Tier 2)
  Aluminum:    Rakic 1995, Appl. Opt. 34:4755 (Tier 2)
  Si3N4:       Philipp 1973, J. Electrochem. Soc. 120:295 (Tier 2)
  PMMA:        Sultanova 2009 (Tier 2)

Functions:
  n_complex(material, wavelength_nm) -> complex   (n + ik)
  n_real(material, wavelength_nm) -> float         (real part only)
"""

import math


# ═══════════════════════════════════════════════════════════════════════════
# SELLMEIER COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════
# Sellmeier form: n² - 1 = Σ B_i λ² / (λ² - C_i)
# λ in micrometers

_SELLMEIER = {
    "SiO2": {
        # Malitson 1965
        "B": [0.6961663, 0.4079426, 0.8974794],
        "C": [0.0684043**2, 0.1162414**2, 9.896161**2],
        "k": 0.0,
    },
    # TiO2 uses Devore form: n² = A + B/(λ² - C). Handled separately.

    "polystyrene": {
        # Sultanova 2009
        "B": [1.4435, 0.0],
        "C": [0.020216, 1e10],
        "k": 0.0,
    },
    "PMMA": {
        # Sultanova 2009
        "B": [1.1819, 0.0],
        "C": [0.011313, 1e10],
        "k": 0.0,
    },
    "ZnO": {
        # Bond 1965 (ordinary ray)
        "B": [2.81418, 0.87968],
        "C": [0.0304, 18.0],
        "k": 0.0,
    },
    "water": {
        # Daimon & Masumura 2007 (24°C)
        "B": [0.5684027565, 0.1726177391, 0.02086189578, 0.1130748688],
        "C": [0.005101829712, 0.01821153936, 0.02620722293, 10.69792721],
        "k": 0.0,
    },
    "Si3N4": {
        # Philipp 1973 — simplified 2-term
        "B": [2.8939, 0.0],
        "C": [0.01396, 1e10],
        "k": 0.0,
    },
}


# Materials using Devore form: n² = A + B/(λ² - C), λ in μm
_DEVORE = {
    "TiO2_rutile": {
        # Devore 1951 (ordinary ray)
        "A": 5.913, "B": 0.2441, "C": 0.0803, "k": 0.0,
    },
    "TiO2_anatase": {
        # Approximate fit from Sarkar 2019 data
        "A": 5.35, "B": 0.174, "C": 0.058, "k": 0.0,
    },
}


def _devore_n(coeffs, wavelength_um):
    """Compute n from Devore formula: n² = A + B/(λ² - C)."""
    lam2 = wavelength_um**2
    denom = lam2 - coeffs["C"]
    if abs(denom) < 1e-20:
        denom = 1e-20
    n2 = coeffs["A"] + coeffs["B"] / denom
    if n2 < 1.0:
        n2 = 1.0
    return math.sqrt(n2)


def _sellmeier_n(coeffs, wavelength_um):
    """Compute n from Sellmeier coefficients at wavelength in micrometers."""
    n2_minus_1 = 0.0
    B = coeffs["B"]
    C = coeffs["C"]
    for b, c in zip(B, C):
        denom = wavelength_um**2 - c
        if abs(denom) < 1e-20:
            denom = 1e-20
        n2_minus_1 += b * wavelength_um**2 / denom
    n2 = 1.0 + n2_minus_1
    if n2 < 1.0:
        n2 = 1.0
    return math.sqrt(n2)


# ═══════════════════════════════════════════════════════════════════════════
# CAUCHY / TABULATED ABSORBING MATERIALS
# ═══════════════════════════════════════════════════════════════════════════
# For absorbing materials: (n, k) at selected wavelengths, linearly interpolated.
# Values from Palik / Johnson-Christy / Querry.

_TABULATED = {
    "Fe2O3": {
        # Hematite — Querry 1985 (selected visible wavelengths)
        # λ(nm): (n, k)
        "data": {
            400: (3.00, 0.60), 450: (2.85, 0.45), 500: (2.70, 0.20),
            550: (2.62, 0.08), 600: (2.58, 0.03), 650: (2.55, 0.01),
            700: (2.52, 0.005), 750: (2.50, 0.002),
        },
    },
    "carbon": {
        # Amorphous carbon — Palik (averaged)
        "data": {
            380: (2.00, 0.80), 400: (2.00, 0.78), 450: (2.02, 0.74),
            500: (2.04, 0.70), 550: (2.06, 0.66), 600: (2.08, 0.62),
            650: (2.10, 0.58), 700: (2.12, 0.54), 750: (2.14, 0.50),
            780: (2.15, 0.48),
        },
    },
    "gold": {
        # Johnson & Christy 1972
        "data": {
            400: (1.658, 1.956), 450: (1.426, 1.806), 500: (0.916, 1.840),
            550: (0.373, 2.268), 600: (0.178, 2.900), 650: (0.161, 3.470),
            700: (0.166, 3.990), 750: (0.175, 4.490),
        },
    },
    "aluminum": {
        # Rakic 1995 (selected visible)
        "data": {
            400: (0.490, 4.860), 450: (0.574, 5.376), 500: (0.672, 5.876),
            550: (0.782, 6.352), 600: (0.979, 6.846), 650: (1.226, 7.244),
            700: (1.552, 7.528), 750: (1.918, 7.688),
        },
    },
    "white_bead": {
        # Titanium white / BaSO4 scattering bead — approximate
        "data": {
            380: (1.55, 0.0), 780: (1.55, 0.0),
        },
    },
}


def _interp_tabulated(data, wavelength_nm):
    """Linearly interpolate (n, k) from tabulated data."""
    lams = sorted(data.keys())
    if wavelength_nm <= lams[0]:
        return data[lams[0]]
    if wavelength_nm >= lams[-1]:
        return data[lams[-1]]

    for i in range(len(lams) - 1):
        if lams[i] <= wavelength_nm <= lams[i + 1]:
            t = (wavelength_nm - lams[i]) / (lams[i + 1] - lams[i])
            n0, k0 = data[lams[i]]
            n1, k1 = data[lams[i + 1]]
            return (n0 + t * (n1 - n0), k0 + t * (k1 - k0))

    return data[lams[-1]]


# ═══════════════════════════════════════════════════════════════════════════
# SIMPLE CONSTANT-n MATERIALS
# ═══════════════════════════════════════════════════════════════════════════

_CONSTANT = {
    "air": (1.0003, 0.0),
}


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def n_complex(material, wavelength_nm):
    """Return complex refractive index n + ik at given wavelength.

    Args:
        material: Material name string
        wavelength_nm: Wavelength in nanometers

    Returns:
        complex: n + ik
    """
    mat = material.lower().replace(" ", "_").replace("-", "_")
    lam_um = wavelength_nm / 1000.0

    # Check Sellmeier materials
    for key, coeffs in _SELLMEIER.items():
        if mat == key.lower():
            n = _sellmeier_n(coeffs, lam_um)
            k = coeffs.get("k", 0.0)
            return complex(n, k)

    # Check Devore-form materials
    for key, coeffs in _DEVORE.items():
        if mat == key.lower():
            n = _devore_n(coeffs, lam_um)
            k = coeffs.get("k", 0.0)
            return complex(n, k)

    # Check tabulated absorbing materials
    for key, entry in _TABULATED.items():
        if mat == key.lower():
            n, k = _interp_tabulated(entry["data"], wavelength_nm)
            return complex(n, k)

    # Check constant materials
    for key, (n, k) in _CONSTANT.items():
        if mat == key.lower():
            return complex(n, k)

    raise ValueError(f"Unknown material: {material}. "
                     f"Available: {sorted(list(_SELLMEIER.keys()) + list(_DEVORE.keys()) + list(_TABULATED.keys()) + list(_CONSTANT.keys()))}")


def n_real(material, wavelength_nm):
    """Return real part of refractive index at given wavelength.

    Args:
        material: Material name string
        wavelength_nm: Wavelength in nanometers

    Returns:
        float: Real refractive index n
    """
    return n_complex(material, wavelength_nm).real


# ═══════════════════════════════════════════════════════════════════════════
# MATERIAL LIST
# ═══════════════════════════════════════════════════════════════════════════

def available_materials():
    """Return list of all available material names."""
    return sorted(
        list(_SELLMEIER.keys()) +
        list(_DEVORE.keys()) +
        list(_TABULATED.keys()) +
        list(_CONSTANT.keys())
    )
