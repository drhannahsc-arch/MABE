"""
optical/photonic_glass.py — Module 6: Disordered Colloidal Reflectance

Combines Module 3 (Mie) × Module 5 (structure factor) to predict the
reflectance spectrum of a disordered colloidal film (photonic glass).

R(λ) ∝ S(q) × |F(q)|²

where F(q) is the single-particle form factor from Mie backscattering
and S(q) is the Percus-Yevick hard-sphere structure factor.

This model exhibits the red problem: individual Mie backscattering biases
blue, competing with the structural S(q) peak at red wavelengths.

Absorber modeled via Maxwell-Garnett effective medium theory.

Validation:
  Reproduce Magkiriadou 2014 photonic glass data for 3+ particle sizes
  Red problem visible for D > 270 nm (desaturated purple, not clean red)
"""

import math
import numpy as np

from optical.refractive_index import n_complex, n_real
from optical.mie_scattering import mie_efficiencies
from optical.structure_factor import structure_factor_PY


def _maxwell_garnett_eff(n_host, n_inclusion, f_inclusion, wavelength_nm=None):
    """Maxwell-Garnett effective medium for n_eff with absorber inclusions.

    Args:
        n_host: Complex refractive index of host medium
        n_inclusion: Complex refractive index of inclusion
        f_inclusion: Volume fraction of inclusion (0 to ~0.3)

    Returns:
        complex: Effective refractive index
    """
    if f_inclusion <= 0:
        return n_host

    eh = n_host**2
    ei = n_inclusion**2

    # MG mixing: ε_eff = ε_h × [1 + 3f(εi-εh)/(εi+2εh - f(εi-εh))]
    num = 3 * f_inclusion * (ei - eh)
    den = ei + 2 * eh - f_inclusion * (ei - eh)
    if abs(den) < 1e-30:
        return n_host
    e_eff = eh * (1 + num / den)

    # sqrt of complex
    import cmath
    return cmath.sqrt(e_eff)


def photonic_glass_reflectance(diameter_nm, sphere_material, n_medium_base,
                                packing_fraction, wavelengths_nm,
                                absorber_material="carbon",
                                absorber_fraction=0.0):
    """Predict reflectance spectrum of a disordered photonic glass.

    R(λ) ∝ S(q_back) × Q_back(λ) × attenuation(k_eff)

    Args:
        diameter_nm: Particle diameter in nm
        sphere_material: Material name for refractive index lookup
        n_medium_base: Base refractive index of interstitial medium
        packing_fraction: Volume fraction φ (typically 0.50-0.64)
        wavelengths_nm: Array of wavelengths in nm
        absorber_material: Absorber material for Maxwell-Garnett mixing
        absorber_fraction: Volume fraction of absorber (0-0.10)

    Returns:
        np.ndarray: Normalized reflectance R(λ) (peak = 1.0)
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    R = np.zeros(len(wavelengths_nm))
    D = diameter_nm
    phi = packing_fraction

    for i, lam in enumerate(wavelengths_nm):
        # Sphere refractive index at this wavelength
        n_sph = n_complex(sphere_material, lam)

        # Effective medium index (with absorber)
        n_med = complex(n_medium_base, 0)
        if absorber_fraction > 0:
            n_abs = n_complex(absorber_material, lam)
            n_med = _maxwell_garnett_eff(n_med, n_abs, absorber_fraction)

        n_eff_real = math.sqrt(phi * n_real(sphere_material, lam)**2 +
                               (1 - phi) * abs(n_med.real)**2)

        # Backscattering wavevector
        q_back = 4 * math.pi * n_eff_real / lam

        # Structure factor
        Sq = structure_factor_PY(q_back, D, phi)

        # Mie backscattering efficiency
        eff = mie_efficiencies(D, n_sph, abs(n_med.real), lam)
        Q_back = eff["Q_back"]
        C_back = (math.pi / 4) * D**2 * Q_back

        # Absorber attenuation (Beer-Lambert through effective medium)
        k_eff = abs(n_med.imag)
        if k_eff > 0:
            # Attenuation over ~10 particle diameters (typical film thickness)
            L = 10 * D
            atten = math.exp(-4 * math.pi * k_eff * L / lam)
        else:
            atten = 1.0

        R[i] = C_back * Sq * atten

    # Normalize to peak = 1
    Rmax = R.max()
    if Rmax > 0:
        R = R / Rmax
    return R


def photonic_glass_peak_wavelength(diameter_nm, sphere_material,
                                    n_medium, packing_fraction):
    """Estimate the peak reflectance wavelength for a photonic glass.

    Uses the simple analytical estimate:
    λ_peak ≈ 2 × n_eff × d_avg
    where d_avg ≈ D × (π/(6φ))^(1/3) for random packing

    More precisely, from the S(q) peak position.

    Args:
        diameter_nm: Particle diameter
        sphere_material: Material name
        n_medium: Interstitial medium index
        packing_fraction: Volume fraction φ

    Returns:
        float: Estimated peak wavelength in nm
    """
    n_sph = n_real(sphere_material, 550)  # Use mid-visible for estimate
    n_eff = math.sqrt(packing_fraction * n_sph**2 +
                      (1 - packing_fraction) * n_medium**2)

    # S(q) peak for hard spheres at this φ
    from optical.structure_factor import structure_factor_peak
    pk = structure_factor_peak(diameter_nm, packing_fraction)
    q_peak = pk["q_peak"]

    if q_peak > 0:
        # Backscattering geometry: q_back = 4π n_eff / λ
        # At peak: λ_peak = 4π n_eff / q_peak
        lam_peak = 4 * math.pi * n_eff / q_peak
    else:
        # Fallback: simple scaling
        lam_peak = 2.0 * n_eff * diameter_nm

    return lam_peak
