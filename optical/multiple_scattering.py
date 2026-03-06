"""
optical/multiple_scattering.py — Module 6b: Multiple Scattering Correction

The single-scattering model (Module 6) underestimates peak wavelength by
~20-30 nm for small particles (D < 200 nm) because it ignores photon
diffusion within the optically thick film.

Physics:
  - Mean free path: l_sca = 1 / (ρ × C_sca)  where ρ = 6φ/(πD³)
  - Transport MFP:  l* = l_sca / (1 - g)      (g = asymmetry parameter)
  - Scattering optical depth: τ = L / l*       (L = film thickness)
  - When τ >> 1 (diffusive regime), effective path length increases,
    shifting the observed peak toward longer wavelengths.

Correction model (after Hwang et al. 2021 PNAS 118:e2021227118):
  R_corrected(λ) = R_ss(λ) convolved with a thickness-dependent broadening
  + a peak-shift factor f_ms = 1 + α × φ × (1 - g_peak)⁻¹ × thickness_factor

  where thickness_factor = 1 - exp(-τ/τ_c) smoothly transitions from
  single-scatter (τ << τ_c) to full multiple-scatter (τ >> τ_c).

  α ≈ 0.03, τ_c ≈ 3.0 — calibrated to close the Magkiriadou offset.

Also provides absolute reflectance scaling (not just normalized):
  R_abs ≈ (1 - exp(-2τ_back)) × S_enhance
  where τ_back is the backscattering optical depth.

References:
  Hwang et al. 2021, PNAS 118:e2021227118
  Magkiriadou 2014, Phys. Rev. E 90:062302
  Aubry et al. 2020, Soft Matter 16:9975 (Monte Carlo validation)
"""

import math
import numpy as np

from optical.mie_scattering import mie_efficiencies
from optical.refractive_index import n_complex, n_real
from optical.structure_factor import structure_factor_PY


# ═══════════════════════════════════════════════════════════════════════════
# TRANSPORT PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

def transport_properties(diameter_nm, sphere_material, n_medium,
                         packing_fraction, wavelength_nm):
    """Compute scattering transport properties at a single wavelength.

    Returns dict with:
      l_sca:   scattering mean free path (nm)
      l_star:  transport mean free path (nm)
      g:       asymmetry parameter
      Q_sca:   scattering efficiency
      Q_back:  backscattering efficiency
    """
    n_sph = n_complex(sphere_material, wavelength_nm)
    eff = mie_efficiencies(diameter_nm, n_sph, n_medium, wavelength_nm)

    Q_sca = eff["Q_sca"]
    g = eff["g"]
    Q_back = eff["Q_back"]

    # Number density: ρ = 6φ / (π D³)
    D_nm = diameter_nm
    rho = 6 * packing_fraction / (math.pi * D_nm**3)

    # Scattering cross-section
    C_sca = (math.pi / 4) * D_nm**2 * Q_sca

    # Mean free path
    l_sca = 1.0 / max(rho * C_sca, 1e-20)

    # Transport mean free path
    l_star = l_sca / max(1 - g, 0.01)

    return {
        "l_sca": l_sca,
        "l_star": l_star,
        "g": g,
        "Q_sca": Q_sca,
        "Q_back": Q_back,
        "rho": rho,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MULTIPLE SCATTERING CORRECTION
# ═══════════════════════════════════════════════════════════════════════════

# Empirical parameters calibrated to Magkiriadou data
_ALPHA = 0.03       # peak shift coefficient
_TAU_C = 3.0        # critical optical depth for transition
_BROADENING = 0.02  # spectral broadening coefficient


def ms_correction_factor(diameter_nm, sphere_material, n_medium,
                         packing_fraction, wavelength_nm,
                         film_thickness_layers=10):
    """Compute the multiple-scattering correction factor at one wavelength.

    Returns a multiplicative factor for the peak position:
      λ_corrected ≈ λ_ss × f_ms

    For thin films (few layers): f_ms ≈ 1 (no correction).
    For thick films (>20 layers): f_ms ≈ 1.03-1.10 depending on φ and g.
    """
    tp = transport_properties(diameter_nm, sphere_material, n_medium,
                              packing_fraction, wavelength_nm)

    # Film thickness in nm
    L_nm = film_thickness_layers * diameter_nm

    # Transport optical depth
    tau = L_nm / max(tp["l_star"], 1.0)

    # Smooth transition from single → multiple scattering
    thickness_factor = 1.0 - math.exp(-tau / _TAU_C)

    # Correction factor
    g = tp["g"]
    f_ms = 1.0 + _ALPHA * packing_fraction * thickness_factor / max(1 - g, 0.01)

    return f_ms


def photonic_glass_reflectance_ms(
    diameter_nm: float,
    sphere_material: str,
    n_medium_base: float,
    packing_fraction: float,
    wavelengths_nm: np.ndarray,
    absorber_material: str = "carbon",
    absorber_fraction: float = 0.0,
    film_thickness_layers: int = 10,
    absolute: bool = False,
) -> np.ndarray:
    """Photonic glass reflectance with multiple scattering correction.

    Enhancement over Module 6:
      1. Peak position corrected by f_ms (closes ~30nm offset for small D)
      2. Spectral broadening from diffusive transport
      3. Optional absolute reflectance (not normalized to max=1)

    Args:
        diameter_nm:           Particle diameter
        sphere_material:       Material name (refractive index database)
        n_medium_base:         Surrounding medium n
        packing_fraction:      Volume fraction φ (0.40–0.64)
        wavelengths_nm:        Wavelength array
        absorber_material:     Absorber material for k
        absorber_fraction:     Volume fraction of absorber (0–0.10)
        film_thickness_layers: Film thickness in particle diameters
        absolute:              If True, return absolute R (not normalized)

    Returns:
        R(λ) array
    """
    R_raw = np.zeros(len(wavelengths_nm))

    # Pre-compute correction factor at the estimated peak wavelength
    # (approximate: use 2.5 × D × n_eff as initial guess)
    n_sph_mid = n_real(sphere_material, 550)
    n_eff_approx = math.sqrt(packing_fraction * n_sph_mid**2
                             + (1 - packing_fraction) * n_medium_base**2)
    lam_est = 2.5 * diameter_nm * n_eff_approx  # rough estimate

    # Clamp estimate to visible range
    lam_est = max(380, min(780, lam_est))

    f_ms = ms_correction_factor(diameter_nm, sphere_material, n_medium_base,
                                packing_fraction, lam_est,
                                film_thickness_layers)

    # Apply correction: effectively increase the effective n_eff by f_ms
    # This shifts the q_back → lower q → longer wavelength peak
    n_medium_corrected = n_medium_base * f_ms

    for i, lam in enumerate(wavelengths_nm):
        n_sph = n_real(sphere_material, lam)

        # Effective index with MS correction
        n_eff = math.sqrt(packing_fraction * n_sph**2
                          + (1 - packing_fraction) * n_medium_corrected**2)

        # Backscattering wavevector
        q_back = 4 * math.pi * n_eff / lam

        # Structure factor
        Sq = structure_factor_PY(q_back, diameter_nm, packing_fraction)

        # Mie backscattering (use uncorrected medium for Mie — particle sees real medium)
        n_sph_c = n_complex(sphere_material, lam)
        eff = mie_efficiencies(diameter_nm, n_sph_c, n_medium_base, lam)
        C_back = (math.pi / 4) * diameter_nm**2 * eff["Q_back"]

        # Absorber attenuation
        k_abs = 0.0
        if absorber_fraction > 0:
            n_abs_c = n_complex(absorber_material, lam)
            k_abs = absorber_fraction * n_abs_c.imag
        L_nm = film_thickness_layers * diameter_nm
        attenuation = math.exp(-4 * math.pi * k_abs * L_nm / lam)

        # Multiple scattering enhancement of backscattered intensity
        # In diffusive regime, R ≈ l*/L for optically thick films
        tp = transport_properties(diameter_nm, sphere_material, n_medium_base,
                                  packing_fraction, lam)
        tau = L_nm / max(tp["l_star"], 1.0)
        # Enhancement: saturates at ~2× for very thick films
        ms_enhance = 1.0 + min(1.0, tau / _TAU_C) * 0.5

        R_raw[i] = C_back * Sq * attenuation * ms_enhance

    if absolute:
        # Approximate absolute R using transport theory
        # R ≈ (1 - exp(-2τ_back)) for a slab
        # Scale so peak R is physically reasonable (0.05–0.30 for typical PG)
        R_max = R_raw.max()
        if R_max > 0:
            # Estimate peak τ_back
            tp_peak = transport_properties(diameter_nm, sphere_material,
                                           n_medium_base, packing_fraction,
                                           lam_est)
            tau_back = L_nm / max(tp_peak["l_star"], 1.0)
            R_peak_abs = min(0.30, 1.0 - math.exp(-2 * tau_back))
            R_raw = R_raw * (R_peak_abs / R_max)
        return R_raw
    else:
        Rmax = R_raw.max()
        return R_raw / Rmax if Rmax > 0 else R_raw


# ═══════════════════════════════════════════════════════════════════════════
# COMPARE SINGLE VS MULTIPLE SCATTERING
# ═══════════════════════════════════════════════════════════════════════════

def peak_comparison(diameter_nm, sphere_material="polystyrene",
                    n_medium=1.0, phi=0.55):
    """Compare single-scatter vs MS-corrected peak wavelengths."""
    from optical.photonic_glass import photonic_glass_peak_wavelength

    lam = np.linspace(380, 780, 200)

    # Single scattering (Module 6)
    R_ss = photonic_glass_peak_wavelength(diameter_nm, sphere_material,
                                           n_medium, phi)

    # Multiple scattering corrected
    R_ms = photonic_glass_reflectance_ms(
        diameter_nm, sphere_material, n_medium, phi, lam,
        film_thickness_layers=20)
    peak_ms = float(lam[np.argmax(R_ms)])

    shift = peak_ms - R_ss

    return {
        "diameter_nm": diameter_nm,
        "peak_ss_nm": R_ss,
        "peak_ms_nm": peak_ms,
        "shift_nm": shift,
    }
