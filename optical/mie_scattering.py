"""
optical/mie_scattering.py — Module 3: Lorenz-Mie Scattering for Single Spheres

Wraps miepython (Wiscombe's stable recurrence) for exact EM scattering
by a homogeneous sphere.

Outputs: Q_sca, Q_ext, Q_abs, Q_back, g (asymmetry parameter)

Validation:
  Rayleigh limit (D << λ): Q_sca ∝ λ⁻⁴
  Geometric limit (D >> λ): Q_ext → 2
"""

import math
import miepython


def mie_efficiencies(diameter_nm, n_sphere, n_medium, wavelength_nm):
    """Compute Mie scattering efficiencies for a homogeneous sphere.

    Args:
        diameter_nm: Sphere diameter in nm
        n_sphere: Complex refractive index of sphere (complex or float)
        n_medium: Refractive index of surrounding medium (real, float)
        wavelength_nm: Wavelength in nm

    Returns:
        dict with keys: Q_ext, Q_sca, Q_abs, Q_back, g
    """
    if isinstance(n_sphere, complex):
        m = n_sphere
    else:
        m = complex(n_sphere, 0.0)

    n_med = float(n_medium.real) if isinstance(n_medium, complex) else float(n_medium)

    # miepython v3.2 API: efficiencies(m, d, lambda0, n_env)
    # m = complex refractive index of sphere
    # d = diameter (same units as lambda0)
    # lambda0 = vacuum wavelength
    # n_env = real index of surrounding medium
    qext, qsca, qback, g = miepython.efficiencies(m, diameter_nm, wavelength_nm,
                                                    n_env=n_med)
    qabs = qext - qsca

    return {
        "Q_ext": float(qext),
        "Q_sca": float(qsca),
        "Q_abs": max(0.0, float(qabs)),
        "Q_back": float(qback),
        "g": float(g),
    }


def scattering_cross_section(diameter_nm, n_sphere, n_medium, wavelength_nm):
    """Compute scattering cross-section C_sca in nm².

    C_sca = Q_sca × π × r²
    """
    eff = mie_efficiencies(diameter_nm, n_sphere, n_medium, wavelength_nm)
    r = diameter_nm / 2.0
    return eff["Q_sca"] * math.pi * r**2


def backscatter_cross_section(diameter_nm, n_sphere, n_medium, wavelength_nm):
    """Compute backscattering cross-section C_back in nm².

    C_back = Q_back × π × r²
    """
    eff = mie_efficiencies(diameter_nm, n_sphere, n_medium, wavelength_nm)
    r = diameter_nm / 2.0
    return eff["Q_back"] * math.pi * r**2
