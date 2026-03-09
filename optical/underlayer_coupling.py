"""
optical/underlayer_coupling.py — Module 8: Substrate Spectral Coupling

Models the optical coupling between a photonic glass film and an opaque
or semi-transparent underlayer/substrate. Light transmitted through the
film is reflected by the substrate and passes back through the film,
adding a spectrally shaped component to the total reflectance.

Physics:
  R_total(λ) = R_film(λ) + T_film(λ)² × R_under(λ) / (1 - R_film(λ)×R_under(λ))

  For strongly scattering films (photonic glasses), simplify to:
  R_total ≈ R_film + (1 - R_film)² × R_under × attenuation

Validation:
  Black underlayer (carbon): maximum saturation (Iwata 2017)
  White underlayer: washed-out color
  Fe₂O₃: reddened color (novel MABE prediction)
"""

import math
import numpy as np

from optical.refractive_index import n_complex, n_real


# ═══════════════════════════════════════════════════════════════════════════
# UNDERLAYER REFLECTANCE
# ═══════════════════════════════════════════════════════════════════════════

def underlayer_reflectance(substrate, wavelengths_nm, film_n=1.35):
    """Compute Fresnel reflectance of the film/substrate interface.

    For opaque substrates, this is the specular reflectance at the
    interface between the effective film medium and the substrate.

    Args:
        substrate: Substrate material name
        wavelengths_nm: Wavelength array in nm
        film_n: Effective refractive index of the photonic glass film

    Returns:
        np.ndarray: R_under(λ) array
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    R_under = np.zeros(len(wavelengths_nm))

    for i, lam in enumerate(wavelengths_nm):
        n_sub = n_complex(substrate, lam)
        n_film = complex(film_n, 0)

        # Fresnel at normal incidence
        r = (n_film - n_sub) / (n_film + n_sub)
        R_under[i] = abs(r)**2

    return np.clip(R_under, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# COUPLED REFLECTANCE
# ═══════════════════════════════════════════════════════════════════════════

def coupled_reflectance(R_film, R_under, film_absorption_fraction=0.0):
    """Combine film and underlayer reflectances with Fabry-Perot coupling.

    Simple two-pass model:
    R_total = R_film + (1-R_film)² × R_under × atten² / (1 - R_film×R_under×atten²)

    Args:
        R_film: Film reflectance spectrum (array)
        R_under: Underlayer reflectance spectrum (array)
        film_absorption_fraction: Fraction of light absorbed per pass through film

    Returns:
        np.ndarray: Total reflectance R_total(λ)
    """
    R_film = np.asarray(R_film)
    R_under = np.asarray(R_under)

    # Transmission through film (per pass)
    T_film = (1.0 - R_film) * (1.0 - film_absorption_fraction)
    T_film = np.clip(T_film, 0.0, 1.0)

    # Two-pass coupling
    denom = 1.0 - R_film * R_under * (1.0 - film_absorption_fraction)**2
    denom = np.maximum(denom, 1e-10)

    R_total = R_film + T_film**2 * R_under / denom
    return np.clip(R_total, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════

def photonic_glass_on_substrate(R_film, wavelengths_nm, substrate="carbon",
                                 film_n=1.35, film_absorption=0.0):
    """Apply substrate coupling to a photonic glass reflectance spectrum.

    Args:
        R_film: Photonic glass reflectance from Module 6
        wavelengths_nm: Wavelength array
        substrate: Substrate material name
        film_n: Effective film refractive index
        film_absorption: Absorption fraction per pass

    Returns:
        np.ndarray: R_total(λ) including substrate coupling
    """
    R_under = underlayer_reflectance(substrate, wavelengths_nm, film_n=film_n)
    return coupled_reflectance(R_film, R_under, film_absorption)
