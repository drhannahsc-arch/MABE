"""
optical/base_layer.py — Functionalization Base Layer Model

Models a thin primer/functionalization layer between the substrate and
the photonic glass film. This layer serves dual purpose:
  1. Chemical: anchors particles to substrate (silane, PDA, polymer brush)
  2. Optical: modifies the effective substrate reflectance via thin-film
     interference before light reaches the photonic glass.

Physics:
  The base layer is a thin dielectric film on the substrate. Light
  transmitted through the photonic glass encounters:
    air/base_layer interface → base_layer → base_layer/substrate interface

  This is a 3-layer TMM problem (ambient=film effective medium, base layer,
  substrate). The result is an effective R_under(λ) that replaces the simple
  Fresnel R_under in the underlayer coupling model (Module 8).

  For very thin layers (<10 nm, e.g. silane SAMs), the optical effect is
  negligible. For thicker layers (50-200 nm PDA, sol-gel TiO₂), thin-film
  interference fringes modulate the substrate reflectance spectrally.

Base layer materials:
  polydopamine   — n ≈ 1.7, k ≈ 0.02-0.05 (UV-absorbing), 10-100 nm typical
  silane_SAM     — n ≈ 1.45, k ≈ 0, 1-3 nm (optically negligible)
  sol_gel_TiO2   — n ≈ 2.3, k ≈ 0 (visible), 20-200 nm
  sol_gel_SiO2   — n ≈ 1.45, k ≈ 0, 20-200 nm
  polymer_brush  — n ≈ 1.50, k ≈ 0, 5-50 nm
  Al2O3_ALD      — n ≈ 1.77, k ≈ 0, 5-100 nm (ALD-deposited)

References:
  Kang et al. 2012, Langmuir 28:12199 (polydopamine optical constants)
  Liu et al. 2014, ACS Nano 8:5559 (PDA-mediated particle attachment)
"""

import math
import numpy as np

from optical.refractive_index import n_complex
from optical.tmm import tmm_reflectance


# ═══════════════════════════════════════════════════════════════════════════
# BASE LAYER OPTICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# (n, k) at 550 nm. For full spectral: use Cauchy or constant approximation.
BASE_LAYER_MATERIALS = {
    "polydopamine":  {"n": 1.70, "k": 0.03, "notes": "Melanin-like, UV-absorbing"},
    "silane_SAM":    {"n": 1.45, "k": 0.0,  "notes": "APTES/MPTMS, optically negligible"},
    "sol_gel_TiO2":  {"n": 2.30, "k": 0.0,  "notes": "High-n, strong interference"},
    "sol_gel_SiO2":  {"n": 1.45, "k": 0.0,  "notes": "Index-matched to SiO2 spheres"},
    "polymer_brush": {"n": 1.50, "k": 0.0,  "notes": "PGMA, PS-b-P2VP, etc."},
    "Al2O3_ALD":     {"n": 1.77, "k": 0.0,  "notes": "Atomic layer deposited"},
    "gold_thin":     {"n": 0.37, "k": 2.82, "notes": "Au thin film, ~10-50 nm"},
    "none":          {"n": 1.00, "k": 0.0,  "notes": "No base layer (air gap)"},
}


def base_layer_n(material, wavelength_nm):
    """Complex refractive index of base layer material.

    Uses constant approximation (adequate for thin layers where
    dispersion effects are small relative to thickness effects).
    """
    if material in BASE_LAYER_MATERIALS:
        props = BASE_LAYER_MATERIALS[material]
        return complex(props["n"], props["k"])
    # Fall back to main refractive index database
    return n_complex(material, wavelength_nm)


# ═══════════════════════════════════════════════════════════════════════════
# EFFECTIVE SUBSTRATE REFLECTANCE WITH BASE LAYER
# ═══════════════════════════════════════════════════════════════════════════

def effective_substrate_reflectance(
    base_material: str,
    base_thickness_nm: float,
    substrate_material: str,
    wavelengths_nm: np.ndarray,
    film_n: float = 1.35,
) -> np.ndarray:
    """Compute effective R_under(λ) for substrate + base layer stack.

    The photonic glass film (effective n ≈ film_n) sits on top of:
      base_layer (thickness t, n_base) → substrate (semi-infinite, n_sub)

    This replaces the simple Fresnel R_under in Module 8.

    Args:
        base_material:      Base layer material name
        base_thickness_nm:  Base layer thickness in nm
        substrate_material: Substrate material (from refractive index db)
        wavelengths_nm:     Wavelength array
        film_n:             Effective n of the photonic glass film above

    Returns:
        R_effective(λ) array — effective reflectance seen from the film side
    """
    if base_material == "none" or base_thickness_nm < 0.5:
        # No base layer — direct film/substrate interface
        from optical.underlayer_coupling import underlayer_reflectance
        return underlayer_reflectance(substrate_material, wavelengths_nm,
                                      film_n=film_n)

    R_eff = np.zeros(len(wavelengths_nm))

    for i, lam in enumerate(wavelengths_nm):
        # Build 3-layer stack: [film_medium, base_layer, substrate]
        # TMM wants: [(material, thickness), ...] with first/last semi-infinite
        # We create an effective "film_medium" entry
        stack = [
            ("_film_medium", 0),           # semi-infinite ambient (film side)
            (base_material, base_thickness_nm),
            (substrate_material, 0),       # semi-infinite substrate
        ]

        # TMM needs to resolve material names to n.
        # Our TMM uses the refractive_index module.
        # For base layer materials not in the main database, we need to
        # handle them specially.
        try:
            # Use custom n lookup
            n_film = complex(film_n, 0.0)
            n_base = base_layer_n(base_material, lam)
            n_sub = n_complex(substrate_material, lam)

            # 2×2 transfer matrix for the base layer
            # Phase acquired in base layer
            delta = 2 * math.pi * n_base * base_thickness_nm / lam

            # Fresnel coefficients at each interface (normal incidence, s-pol)
            r12 = (n_film - n_base) / (n_film + n_base)
            r23 = (n_base - n_sub) / (n_base + n_sub)

            # Transfer matrix for the layer
            # M = [[exp(iδ), r23*exp(iδ)], [r23*exp(-iδ), exp(-iδ)]]
            # Total r = (r12 + r23*exp(2iδ)) / (1 + r12*r23*exp(2iδ))
            import cmath
            phase = cmath.exp(2j * delta)
            r_total = (r12 + r23 * phase) / (1 + r12 * r23 * phase)
            R_eff[i] = abs(r_total)**2

        except Exception:
            # Fallback: just use Fresnel at film/substrate
            n_sub = n_complex(substrate_material, lam)
            r = (complex(film_n) - n_sub) / (complex(film_n) + n_sub)
            R_eff[i] = abs(r)**2

    return np.clip(R_eff, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# FULL STACK: PHOTONIC GLASS + BASE LAYER + SUBSTRATE
# ═══════════════════════════════════════════════════════════════════════════

def photonic_glass_on_base_layer(
    R_film: np.ndarray,
    wavelengths_nm: np.ndarray,
    base_material: str = "polydopamine",
    base_thickness_nm: float = 50.0,
    substrate_material: str = "carbon",
    film_n: float = 1.35,
    film_absorption_fraction: float = 0.0,
) -> np.ndarray:
    """Apply base layer + substrate coupling to photonic glass reflectance.

    Replaces photonic_glass_on_substrate() when a base layer is present.

    Stack (top to bottom):
      air → photonic glass film → base layer → substrate

    Args:
        R_film:                   Photonic glass reflectance from Module 6/6b
        wavelengths_nm:           Wavelength array
        base_material:            Base layer material
        base_thickness_nm:        Base layer thickness (nm)
        substrate_material:       Substrate material
        film_n:                   Effective n of photonic glass
        film_absorption_fraction: Absorption per pass through film

    Returns:
        R_total(λ) including base layer interference effects
    """
    from optical.underlayer_coupling import coupled_reflectance

    R_under = effective_substrate_reflectance(
        base_material, base_thickness_nm,
        substrate_material, wavelengths_nm,
        film_n=film_n)

    return coupled_reflectance(R_film, R_under, film_absorption_fraction)


# ═══════════════════════════════════════════════════════════════════════════
# MULTILEVEL ATTACHMENT STACK
# ═══════════════════════════════════════════════════════════════════════════

def multilevel_stack_reflectance(
    R_film: np.ndarray,
    wavelengths_nm: np.ndarray,
    layers: list,
    substrate_material: str = "carbon",
    film_n: float = 1.35,
    film_absorption_fraction: float = 0.0,
) -> np.ndarray:
    """Handle arbitrary multilevel attachment stacks.

    For complex architectures: primer + adhesion layer + spacer + etc.

    Args:
        R_film:              Photonic glass reflectance
        wavelengths_nm:      Wavelength array
        layers:              List of (material, thickness_nm) tuples,
                             ordered top-to-bottom (closest to film first)
        substrate_material:  Bottom substrate
        film_n:              Film effective n
        film_absorption_fraction: Film absorption per pass

    Returns:
        R_total(λ) array
    """
    from optical.underlayer_coupling import coupled_reflectance

    # Compute effective R_under for the entire sub-film stack
    R_under = np.zeros(len(wavelengths_nm))

    for i, lam in enumerate(wavelengths_nm):
        import cmath

        # Build transfer matrix for all layers
        n_top = complex(film_n, 0.0)

        # Start from substrate side and work up
        n_sub = n_complex(substrate_material, lam)

        # Initialize with substrate
        # Use iterative Fresnel for each layer
        r_cumulative = complex(0, 0)

        # Process layers bottom-to-top
        n_below = n_sub
        for layer_mat, layer_t in reversed(layers):
            n_layer = base_layer_n(layer_mat, lam)

            # Fresnel at layer/below interface
            r_lb = (n_layer - n_below) / (n_layer + n_below)

            # Phase in this layer
            delta = 2 * math.pi * n_layer * layer_t / lam
            phase = cmath.exp(2j * delta)

            # Combine with what's below using Airy formula
            r_cumulative = (r_lb + r_cumulative * phase) / (
                1 + r_lb * r_cumulative * phase)

            n_below = n_layer

        # Final interface: film/top_layer
        r_ft = (n_top - n_below) / (n_top + n_below)
        # No additional phase (top of stack)
        r_total = (r_ft + r_cumulative) / (1 + r_ft * r_cumulative)
        R_under[i] = abs(r_total)**2

    R_under = np.clip(R_under, 0.0, 1.0)
    return coupled_reflectance(R_film, R_under, film_absorption_fraction)
