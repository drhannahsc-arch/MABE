"""
optical/janus_sphere.py — Janus Sphere Two-Way Color Model

Janus particles have two hemispheres of different optical properties.
In a photonic glass, this creates orientation-dependent structural color:
  - Side A (cap material faces viewer): one scattering cross-section
  - Side B (base material faces viewer): different scattering cross-section
  - Random orientation: weighted average of both sides

Two-way color applications:
  - Security/anti-counterfeiting: different color from each side of a film
  - Switchable displays: external field (magnetic, electric) rotates particles
  - Decorative: angle-dependent color effects

Physics model:
  A Janus sphere is approximated as two hemisphere-weighted Mie contributions.
  For each hemisphere orientation:
    Q_back_eff(λ) = f_cap × Q_back_coreshell(λ) + f_base × Q_back_core(λ)
  where f_cap, f_base are hemisphere solid angle fractions (0.5 each for
  symmetric Janus; adjustable for patchy particles).

  The core-shell model (Module 4) handles the cap hemisphere — the cap
  coating modifies the outer shell n. The base hemisphere uses the bare
  core Mie (Module 3).

  For a film with aligned particles (all caps up): use Q_back_cap.
  For a film with all caps down: use Q_back_base.
  For random orientation (standard): average Q_back = 0.5 × (cap + base).

  The resulting Q_back_eff feeds into the photonic glass model (M6/M6b)
  to produce the full reflectance spectrum.

Cap materials:
  gold (Au)        — plasmonic, strong wavelength-dependent scattering
  TiO2_rutile      — high-n dielectric, broadband scattering enhancement
  Fe2O3            — absorbing + magnetic (enables external alignment)
  carbon           — absorbing cap (one side dark, one side colored)
  polydopamine     — bio-compatible absorbing cap

References:
  Jiang et al. 2010, JACS 132:14862 (Janus particle photonic crystals)
  Hu & Bhatt 2020, Soft Matter 16:4069 (Janus structural color)
  Nie et al. 2019, Angew. Chem. 58:15556 (magnetically switchable Janus)
"""

import sys
import os
import math
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from optical.mie_scattering import mie_efficiencies
from optical.core_shell_mie import mie_coated_efficiencies
from optical.refractive_index import n_complex, n_real
from optical.structure_factor import structure_factor_PY
from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, XYZ_to_sRGB


# ═══════════════════════════════════════════════════════════════════════════
# JANUS SPHERE EFFECTIVE MACK-SCATTERING
# ═══════════════════════════════════════════════════════════════════════════

def janus_Q_back(
    diameter_nm: float,
    core_material: str,
    cap_material: str,
    cap_thickness_nm: float,
    n_medium: float,
    wavelength_nm: float,
    cap_fraction: float = 0.5,
    orientation: str = "random",
) -> float:
    """Effective backscattering efficiency for a Janus sphere.

    Args:
        diameter_nm:      Total sphere diameter (core only, before cap)
        core_material:    Core material
        cap_material:     Cap hemisphere coating material
        cap_thickness_nm: Cap coating thickness
        n_medium:         Medium refractive index
        wavelength_nm:    Wavelength
        cap_fraction:     Fraction of surface covered by cap (0.5 = symmetric Janus)
        orientation:      "cap_up" (cap faces viewer), "cap_down", or "random"

    Returns:
        Q_back_eff — effective backscattering efficiency
    """
    n_core = n_complex(core_material, wavelength_nm)
    r_core = diameter_nm / 2

    # Cap side: core-shell Mie (core + cap coating)
    d_total_cap = diameter_nm + 2 * cap_thickness_nm
    try:
        eff_cap = mie_coated_efficiencies(
            diameter_nm, d_total_cap,
            n_core, n_complex(cap_material, wavelength_nm),
            n_medium, wavelength_nm)
        Q_back_cap = eff_cap["Q_back"]
    except Exception:
        Q_back_cap = 0.0

    # Base side: bare core Mie (no coating)
    eff_base = mie_efficiencies(diameter_nm, n_core, n_medium, wavelength_nm)
    Q_back_base = eff_base["Q_back"]

    if orientation == "cap_up":
        return Q_back_cap
    elif orientation == "cap_down":
        return Q_back_base
    else:  # random
        return cap_fraction * Q_back_cap + (1 - cap_fraction) * Q_back_base


# ═══════════════════════════════════════════════════════════════════════════
# JANUS PHOTONIC GLASS REFLECTANCE
# ═══════════════════════════════════════════════════════════════════════════

def janus_photonic_glass_reflectance(
    diameter_nm: float,
    core_material: str,
    cap_material: str,
    cap_thickness_nm: float,
    n_medium: float,
    packing_fraction: float,
    wavelengths_nm: np.ndarray,
    cap_fraction: float = 0.5,
    orientation: str = "random",
    absorber_fraction: float = 0.0,
    absorber_material: str = "carbon",
) -> np.ndarray:
    """Full photonic glass reflectance using Janus spheres.

    Same physics as Module 6 but with orientation-dependent Q_back.

    Args:
        diameter_nm:       Core diameter (nm)
        core_material:     Core material
        cap_material:      Cap material
        cap_thickness_nm:  Cap thickness (nm)
        n_medium:          Medium n
        packing_fraction:  φ
        wavelengths_nm:    Wavelength array
        cap_fraction:      Surface coverage of cap (0–1)
        orientation:       "cap_up", "cap_down", or "random"
        absorber_fraction: Volume fraction of absorber
        absorber_material: Absorber material

    Returns:
        R(λ) array (normalized to max=1)
    """
    R = np.zeros(len(wavelengths_nm))
    D_eff = diameter_nm + 2 * cap_thickness_nm  # effective diameter with cap

    for i, lam in enumerate(wavelengths_nm):
        n_core_r = n_real(core_material, lam)

        # Effective index uses the effective diameter
        n_eff = math.sqrt(packing_fraction * n_core_r**2
                          + (1 - packing_fraction) * n_medium**2)

        # Backscattering wavevector
        q_back = 4 * math.pi * n_eff / lam

        # Structure factor (use effective diameter for packing)
        Sq = structure_factor_PY(q_back, D_eff, packing_fraction)

        # Janus backscattering cross-section
        Qback = janus_Q_back(diameter_nm, core_material, cap_material,
                              cap_thickness_nm, n_medium, lam,
                              cap_fraction, orientation)
        C_back = (math.pi / 4) * D_eff**2 * Qback

        # Absorber attenuation
        k_abs = 0.0
        if absorber_fraction > 0:
            n_abs_c = n_complex(absorber_material, lam)
            k_abs = absorber_fraction * n_abs_c.imag
        L_nm = 10 * D_eff
        attenuation = math.exp(-4 * math.pi * k_abs * L_nm / lam)

        R[i] = C_back * Sq * attenuation

    Rmax = R.max()
    return R / Rmax if Rmax > 0 else R


# ═══════════════════════════════════════════════════════════════════════════
# TWO-WAY COLOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def two_way_color(
    diameter_nm: float,
    core_material: str,
    cap_material: str,
    cap_thickness_nm: float,
    n_medium: float = 1.0,
    packing_fraction: float = 0.55,
    wavelengths_nm: np.ndarray = None,
) -> dict:
    """Compute colors for all three orientations of a Janus photonic glass.

    Returns CIE coordinates and perceived color differences for:
      - cap_up: all caps face viewer
      - cap_down: all bases face viewer
      - random: randomly oriented

    The ΔE between cap_up and cap_down quantifies the two-way color effect.

    Returns:
        dict with keys: cap_up, cap_down, random — each containing:
          peak_nm, cie_xy, Lab, sRGB, R_spectrum
        plus:
          delta_E_two_way (CIE76 distance between cap_up and cap_down)
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 81)

    result = {}

    for orient in ["cap_up", "cap_down", "random"]:
        R = janus_photonic_glass_reflectance(
            diameter_nm, core_material, cap_material, cap_thickness_nm,
            n_medium, packing_fraction, wavelengths_nm,
            orientation=orient)

        peak_nm = float(wavelengths_nm[np.argmax(R)])
        X, Y, Z = spectrum_to_XYZ(R, wavelengths_nm)
        x, y, _ = XYZ_to_xyY(X, Y, Z)
        L, a, b = XYZ_to_Lab(X, Y, Z)
        srgb = XYZ_to_sRGB(X, Y, Z)
        srgb_int = tuple(max(0, min(255, int(round(c * 255)))) for c in srgb)

        result[orient] = {
            "peak_nm": peak_nm,
            "cie_xy": (round(x, 4), round(y, 4)),
            "Lab": (round(L, 1), round(a, 1), round(b, 1)),
            "sRGB": srgb_int,
            "R_spectrum": R,
        }

    # ΔE between the two oriented states
    from optical.cie_color import cie_delta_E
    dE = cie_delta_E(result["cap_up"]["Lab"], result["cap_down"]["Lab"])
    result["delta_E_two_way"] = round(dE, 1)

    return result


def print_two_way(result):
    """Pretty-print two-way color analysis."""
    print()
    print("  MABE Janus Two-Way Color Analysis")
    print(f"  ΔE (cap_up vs cap_down): {result['delta_E_two_way']:.1f}")
    print()
    for orient in ["cap_up", "cap_down", "random"]:
        d = result[orient]
        print(f"  {orient:10s}: peak={d['peak_nm']:.0f}nm  "
              f"xy=({d['cie_xy'][0]:.3f},{d['cie_xy'][1]:.3f})  "
              f"Lab=({d['Lab'][0]:.0f},{d['Lab'][1]:.0f},{d['Lab'][2]:.0f})  "
              f"sRGB={d['sRGB']}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 60)
    print("MABE Janus Sphere Two-Way Color — Self-Test")
    print("=" * 60)

    lam = np.linspace(380, 780, 81)

    # Test 1: SiO2 core + Au cap
    print("\n--- SiO2 core (200nm) + Au cap (20nm) ---")
    r1 = two_way_color(200, "SiO2", "Au", 20.0, wavelengths_nm=lam)
    print_two_way(r1)

    # Test 2: SiO2 core + TiO2 cap
    print("--- SiO2 core (200nm) + TiO2 cap (30nm) ---")
    r2 = two_way_color(200, "SiO2", "TiO2_rutile", 30.0, wavelengths_nm=lam)
    print_two_way(r2)

    # Test 3: SiO2 core + carbon cap (absorber on one side)
    print("--- SiO2 core (200nm) + carbon cap (15nm) ---")
    r3 = two_way_color(200, "SiO2", "carbon", 15.0, wavelengths_nm=lam)
    print_two_way(r3)

    # Test 4: PS core + Fe2O3 cap (magnetic Janus)
    print("--- PS core (250nm) + Fe2O3 cap (25nm) — magnetic switchable ---")
    r4 = two_way_color(250, "polystyrene", "Fe2O3", 25.0, wavelengths_nm=lam)
    print_two_way(r4)
