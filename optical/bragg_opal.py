"""
optical/bragg_opal.py — Module 2: Bragg Diffraction for Ordered Opals

λ_peak = 2 × d_111 × n_eff
       = 2 × D × √(2/3) × n_eff
       = 1.633 × D × n_eff    (for FCC close-packed)

One equation. Zero free parameters.

Data source for validation:
  Waterhouse & Waterland 2007, Polyhedron 26:356-368 (Tier 2)
"""

import math

from optical.refractive_index import n_real


def n_eff_volume_average(n_sphere, n_medium, fill_fraction=0.7405):
    """Volume-average effective refractive index.

    n_eff = sqrt(f * n_sphere^2 + (1-f) * n_medium^2)

    Args:
        n_sphere: Refractive index of spheres
        n_medium: Refractive index of interstitial medium
        fill_fraction: Volume fraction (0.7405 for FCC close-packed)

    Returns:
        float: Effective refractive index
    """
    n2 = fill_fraction * n_sphere**2 + (1 - fill_fraction) * n_medium**2
    return math.sqrt(n2)


def bragg_opal(diameter_nm, n_sphere=None, n_medium=1.0, fill_fraction=0.7405,
               material=None, wavelength_ref=None):
    """Predict peak reflectance wavelength for an ordered FCC opal.

    λ_peak = 1.633 × D × n_eff

    Args:
        diameter_nm: Sphere diameter in nm
        n_sphere: Refractive index of sphere (or use material)
        n_medium: Refractive index of interstitial medium
        fill_fraction: Volume fraction (default 0.7405 for FCC)
        material: Material name for n lookup (overrides n_sphere)
        wavelength_ref: Reference wavelength for n lookup (default: estimated peak)

    Returns:
        float: Peak wavelength in nm
    """
    if material is not None and n_sphere is None:
        # Estimate peak first for n lookup
        n_est = n_real(material, 550)
        n_eff_est = n_eff_volume_average(n_est, n_medium, fill_fraction)
        lam_est = 1.633 * diameter_nm * n_eff_est
        # Refine with n at estimated peak
        ref = wavelength_ref or max(380, min(780, lam_est))
        n_sphere = n_real(material, ref)
    elif n_sphere is None:
        raise ValueError("Either n_sphere or material must be provided")

    n_eff = n_eff_volume_average(n_sphere, n_medium, fill_fraction)
    return 1.633 * diameter_nm * n_eff
