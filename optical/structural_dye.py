"""
optical/structural_dye.py — Structural Dye Forward Model

Wires the coordination-optics bridge (M4a) through the full photonic pipeline:

  Shell chemistry → n_shell(λ), k_shell(λ)    [M4a]
       ↓
  Core-shell Mie → Q_back(λ)                   [M4]
       ↓
  Structure factor → S(q_back)                  [M5]
       ↓
  Photonic glass reflectance R(λ)               [M6-style]
       ↓
  CIE colorimetry → (x, y), Lab, sRGB          [M9]
       ↓
  Chromaticity shift ΔE vs bare core

This is the central claim of the META 2026 abstract:
"Same core, different shell, different color."

Zero parameters fitted to structural color data.
"""

import math
import numpy as np

from optical.refractive_index import n_complex, n_real
from optical.core_shell_mie import mie_coated_efficiencies
from optical.mie_scattering import mie_efficiencies
from optical.structure_factor import structure_factor_PY
from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, XYZ_to_sRGB, cie_delta_E,
)
from optical.coordination_optics_bridge import coordination_optics_bridge


# ═══════════════════════════════════════════════════════════════════════════
# CORE FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════════════

def structural_dye_reflectance(
    core_diameter_nm: float,
    core_material: str,
    shell_thickness_nm: float,
    donor_subtypes: list,
    metal: str,
    packing_fraction: float = 0.55,
    n_medium: float = 1.0,
    surface_coverage_nm2: float = 3.0,
    wavelengths_nm: np.ndarray = None,
    absorber_material: str = "carbon",
    absorber_fraction: float = 0.0,
):
    """Predict reflectance spectrum of a structural dye photonic glass.

    Full pipeline: shell chemistry → optical properties → Mie → S(q) → R(λ) → CIE.

    Args:
        core_diameter_nm: Core particle diameter (nm)
        core_material: Core material name (e.g. "SiO2")
        shell_thickness_nm: Functionalized shell thickness (nm)
        donor_subtypes: Donor subtype list (from binding engine classification)
        metal: Metal formula (e.g. "Cu2+", "Pb2+", "" for no metal)
        packing_fraction: Volume fraction φ
        n_medium: Interstitial medium refractive index
        surface_coverage_nm2: Metal sites per nm²
        wavelengths_nm: Wavelength array (default: 380-780nm, 2nm steps)
        absorber_material: Broadband absorber material
        absorber_fraction: Absorber volume fraction

    Returns:
        dict with:
          R: reflectance spectrum (normalized)
          wavelengths_nm: wavelength array
          peak_nm: peak wavelength
          CIE_xy: (x, y) chromaticity
          Lab: (L*, a*, b*)
          sRGB: (R, G, B) in [0,1]
          bridge: full M4a bridge output dict
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 201)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    d_total = core_diameter_nm + 2 * shell_thickness_nm

    # ── Get shell optical properties from M4a bridge ─────────────────────
    bridge = coordination_optics_bridge(
        donor_subtypes=donor_subtypes,
        metal=metal,
        shell_thickness_nm=shell_thickness_nm,
        surface_coverage_nm2=surface_coverage_nm2,
        particle_diameter_nm=core_diameter_nm,
        base_shell_n=1.46,  # APTES baseline
        wavelength_array_nm=wavelengths_nm,
    )
    n_shell_arr = bridge["n_shell"]
    k_shell_arr = bridge["k_shell"]

    # ── Compute R(λ) = S(q) × Q_back(λ) with absorber attenuation ───────
    R = np.zeros(len(wavelengths_nm))

    for i, lam in enumerate(wavelengths_nm):
        n_core = n_complex(core_material, lam)
        n_shell = complex(n_shell_arr[i], k_shell_arr[i])

        # Effective medium index for S(q) calculation
        n_core_real = n_real(core_material, lam)
        n_eff = math.sqrt(
            packing_fraction * n_core_real**2 +
            (1 - packing_fraction) * n_medium**2
        )

        # Backscattering wavevector
        q_back = 4 * math.pi * n_eff / lam

        # Structure factor (use total diameter for particle spacing)
        Sq = structure_factor_PY(q_back, d_total, packing_fraction)

        # Core-shell Mie backscattering
        eff = mie_coated_efficiencies(
            core_diameter_nm, d_total,
            n_core, n_shell, n_medium, lam
        )
        Q_back = eff["Q_back"]
        C_back = (math.pi / 4) * d_total**2 * Q_back

        # Absorber attenuation
        atten = 1.0
        if absorber_fraction > 0:
            n_abs = n_complex(absorber_material, lam)
            k_abs = absorber_fraction * abs(n_abs.imag)
            L = 10 * d_total
            atten = math.exp(-4 * math.pi * k_abs * L / lam)

        R[i] = C_back * Sq * atten

    # Normalize
    Rmax = R.max()
    if Rmax > 0:
        R = R / Rmax

    # ── CIE colorimetry ──────────────────────────────────────────────────
    X, Y, Z = spectrum_to_XYZ(R, wavelengths_nm)
    x, y, Y_lum = XYZ_to_xyY(X, Y, Z)
    Lab = XYZ_to_Lab(X, Y, Z)
    srgb = XYZ_to_sRGB(X, Y, Z)

    peak_nm = float(wavelengths_nm[np.argmax(R)])

    return {
        "R": R,
        "wavelengths_nm": wavelengths_nm,
        "peak_nm": peak_nm,
        "CIE_xy": (x, y),
        "Lab": Lab,
        "sRGB": srgb,
        "bridge": bridge,
    }


def bare_core_reflectance(
    core_diameter_nm: float,
    core_material: str,
    packing_fraction: float = 0.55,
    n_medium: float = 1.0,
    wavelengths_nm: np.ndarray = None,
    absorber_fraction: float = 0.0,
):
    """Reflectance of bare (unfunctionalized) photonic glass — reference."""
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 201)

    from optical.photonic_glass import photonic_glass_reflectance
    R = photonic_glass_reflectance(
        core_diameter_nm, core_material, n_medium,
        packing_fraction, wavelengths_nm,
        absorber_fraction=absorber_fraction,
    )

    X, Y, Z = spectrum_to_XYZ(R, wavelengths_nm)
    x, y, Y_lum = XYZ_to_xyY(X, Y, Z)
    Lab = XYZ_to_Lab(X, Y, Z)
    srgb = XYZ_to_sRGB(X, Y, Z)
    peak_nm = float(wavelengths_nm[np.argmax(R)])

    return {
        "R": R,
        "wavelengths_nm": wavelengths_nm,
        "peak_nm": peak_nm,
        "CIE_xy": (x, y),
        "Lab": Lab,
        "sRGB": srgb,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CHROMATICITY SHIFT CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def chromaticity_shift(bare_result, dye_result):
    """Compute chromaticity shift between bare core and structural dye.

    Returns:
        dict with:
          delta_peak_nm: Peak wavelength shift (positive = redshift)
          delta_xy: Euclidean distance in CIE xy
          delta_E: CIE76 ΔE in Lab space
          exceeds_perceptual_threshold: bool (ΔExy > 0.03)
    """
    dx = dye_result["CIE_xy"][0] - bare_result["CIE_xy"][0]
    dy = dye_result["CIE_xy"][1] - bare_result["CIE_xy"][1]
    delta_xy = math.sqrt(dx**2 + dy**2)
    delta_E = cie_delta_E(bare_result["Lab"], dye_result["Lab"])
    delta_peak = dye_result["peak_nm"] - bare_result["peak_nm"]

    return {
        "delta_peak_nm": delta_peak,
        "delta_x": dx,
        "delta_y": dy,
        "delta_xy": delta_xy,
        "delta_E_Lab": delta_E,
        "exceeds_perceptual_threshold": delta_xy > 0.03,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT FIGURE 2: THREE SHELLS ON SAME CORE
# ═══════════════════════════════════════════════════════════════════════════

def figure_2_panel(core_diameter_nm=225, core_material="SiO2",
                   shell_thickness_nm=1.5, packing_fraction=0.55):
    """Reproduce META 2026 abstract Figure 2.

    Three molecular shells on the same SiO₂ core produce distinct
    chromaticity shifts via Δn + Δk or pure Δn.

    Returns:
        dict with bare + 3 shell results + shifts
    """
    lam = np.linspace(380, 780, 201)

    bare = bare_core_reflectance(
        core_diameter_nm, core_material,
        packing_fraction=packing_fraction,
        wavelengths_nm=lam,
    )

    shells = {
        "BPMEN+Cu2+": {
            "donors": ["N_pyridine", "N_pyridine", "N_amine", "N_amine"],
            "metal": "Cu2+",
        },
        "DTC+Pb2+": {
            "donors": ["S_dithiocarbamate", "S_dithiocarbamate", "N_amine"],
            "metal": "Pb2+",
        },
        "Bipy+Cu2+": {
            "donors": ["N_pyridine", "N_pyridine"],
            "metal": "Cu2+",
        },
    }

    results = {"bare": bare, "shells": {}}

    for name, spec in shells.items():
        dye = structural_dye_reflectance(
            core_diameter_nm=core_diameter_nm,
            core_material=core_material,
            shell_thickness_nm=shell_thickness_nm,
            donor_subtypes=spec["donors"],
            metal=spec["metal"],
            packing_fraction=packing_fraction,
            wavelengths_nm=lam,
        )
        shift = chromaticity_shift(bare, dye)
        results["shells"][name] = {
            "dye": dye,
            "shift": shift,
        }

    return results


def print_figure_2(results=None):
    """Pretty-print Figure 2 results."""
    if results is None:
        results = figure_2_panel()

    bare = results["bare"]
    print(f"  Bare SiO₂: λ_peak={bare['peak_nm']:.0f}nm, "
          f"CIE xy=({bare['CIE_xy'][0]:.4f}, {bare['CIE_xy'][1]:.4f})")
    print()

    for name, data in results["shells"].items():
        dye = data["dye"]
        s = data["shift"]
        bridge = dye["bridge"]

        print(f"  {name}:")
        print(f"    λ_peak = {dye['peak_nm']:.0f}nm "
              f"(Δλ = {s['delta_peak_nm']:+.0f}nm)")
        print(f"    CIE xy = ({dye['CIE_xy'][0]:.4f}, {dye['CIE_xy'][1]:.4f}) "
              f"  Δxy = {s['delta_xy']:.4f}")
        print(f"    ΔE_Lab = {s['delta_E_Lab']:.1f}")
        print(f"    Δn = {bridge['delta_n_total']:+.5f} "
              f"(func={bridge['delta_n_functionalization']:+.5f}, "
              f"coord={bridge['delta_n_coordination']:+.5f})")
        if bridge["lambda_dd_nm"] > 0:
            print(f"    d-d band: λ={bridge['lambda_dd_nm']:.0f}nm, "
                  f"k_max={bridge['k_shell'].max():.5f}")
        else:
            print(f"    No d-d absorption (Δk = 0)")
        print(f"    Perceptual: {'YES' if s['exceeds_perceptual_threshold'] else 'NO'} "
              f"(threshold: Δxy > 0.03)")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("  MABE Structural Dye — Abstract Figure 2")
    print("  " + "=" * 44)
    print()
    results = figure_2_panel()
    print_figure_2(results)
