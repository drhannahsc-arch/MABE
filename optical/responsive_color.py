"""
optical/responsive_color.py — Stimulus-Responsive Structural Color

Models five mechanisms for dynamically shifting structural color:

1. MAGNETIC: External field rotates Fe2O3-capped Janus particles.
   Langevin function maps field strength → orientation distribution →
   weighted Q_back → continuous color tuning between cap-up and cap-down.

2. THERMAL (PNIPAM): Poly(N-isopropylacrylamide) spacer undergoes
   coil-globule transition at LCST ≈ 32°C. Below LCST: swollen hydrogel
   (long spacer). Above LCST: collapsed (short spacer). Changes
   interparticle spacing → peak wavelength shift.

3. UV/IR PHOTOCHROMIC: Azobenzene (trans↔cis) or spiropyran (closed↔open)
   in the spacer changes molecular length and/or refractive index under
   UV irradiation. Reversible under visible light or heat.

4. ELECTRIC: Dielectrophoretic torque on Janus particles in AC electric
   field. Analogous to magnetic but driven by dielectric contrast
   between cap and base hemispheres.

5. ASSEMBLY-DIRECTED: Same functionalized particles assembled into
   ordered opal (Bragg M2) vs disordered glass (M6). Linker chemistry
   controls assembly: flexible linkers → disorder, rigid/directional → order.

All mechanisms produce a StimulusResponse containing before/after spectra,
Δλ_peak, ΔE (CIE76), and the physical parameters driving the shift.

References:
  Ge & Yin 2011, Angew. Chem. 50:1492 (magnetic responsive photonic crystals)
  Takeoka 2012, J. Mater. Chem. 22:23299 (PNIPAM-particle responsive PCs)
  Shang et al. 2015, Acc. Chem. Res. 48:2803 (responsive photonic crystals)
  Saito et al. 2013, J. Mater. Chem. C 1:999 (azobenzene photonic gels)
"""

import sys
import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, cie_delta_E, XYZ_to_sRGB,
)

_LAM = np.linspace(380, 780, 81)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColorState:
    """Optical state at one stimulus condition."""
    label: str
    stimulus_value: float        # field strength, temperature, UV dose, etc.
    R_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peak_nm: float = 0.0
    cie_xy: tuple = (0.0, 0.0)
    Lab: tuple = (0.0, 0.0, 0.0)
    sRGB: tuple = (0, 0, 0)


@dataclass
class StimulusResponse:
    """Full stimulus-response characterization."""
    mechanism: str               # "magnetic", "thermal", "uv", "electric", "assembly"
    core_material: str = ""
    core_diameter_nm: float = 0.0
    states: list = field(default_factory=list)  # list[ColorState]
    delta_E_range: float = 0.0   # ΔE between most extreme states
    delta_lambda_nm: float = 0.0 # Peak shift between extremes
    notes: str = ""


def _color_from_R(R, lam):
    """Extract color metrics from a reflectance spectrum."""
    X, Y, Z = spectrum_to_XYZ(R, lam)
    x, y, _ = XYZ_to_xyY(X, Y, Z)
    Lab = XYZ_to_Lab(X, Y, Z)
    srgb = XYZ_to_sRGB(X, Y, Z)
    srgb_int = tuple(max(0, min(255, int(round(c * 255)))) for c in srgb)
    peak = float(lam[np.argmax(R)])
    return peak, (round(x, 4), round(y, 4)), tuple(round(v, 1) for v in Lab), srgb_int


# ═══════════════════════════════════════════════════════════════════════════
# 1. MAGNETIC FIELD — JANUS PARTICLE ROTATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Physics: Fe2O3 cap has permanent magnetic moment m.
# In external field B, torque aligns cap along field.
# Orientation parameter: <cos θ> = L(ξ) = coth(ξ) - 1/ξ
#   where ξ = m·B / (k_B·T)
# ξ = 0: random (thermal), ξ → ∞: fully aligned
#
# Effective Q_back = f_aligned × Q_cap + (1-f_aligned) × Q_base
#   where f_aligned = (1 + L(ξ)) / 2  (maps L∈[-1,1] to f∈[0,1])
# ═══════════════════════════════════════════════════════════════════════════

def _langevin(xi):
    """Langevin function L(ξ) = coth(ξ) - 1/ξ."""
    if abs(xi) < 1e-8:
        return xi / 3.0  # Taylor expansion for small ξ
    return 1.0 / math.tanh(xi) - 1.0 / xi


# Magnetic moment of Fe2O3 cap (rough estimate)
# Magnetite ~ 480 kA/m saturation, hematite ~ 2 kA/m (weakly ferromagnetic)
# For a 25nm hemispherical cap on a 250nm sphere:
# V_cap ≈ (2/3)π r_cap² × t_cap ≈ 1.6e-23 m³ for t=25nm, D=250nm
# m = M_s × V_cap ≈ 480e3 × 1.6e-23 ≈ 7.7e-18 A·m² for magnetite
_M_MAGNETITE = 480e3    # A/m saturation magnetization
_M_HEMATITE = 2.5e3     # A/m (α-Fe2O3, weakly ferromagnetic)
_KB = 1.381e-23         # J/K


def magnetic_cap_moment(diameter_nm, cap_thickness_nm, cap_fraction=0.5,
                        cap_magnetization="magnetite"):
    """Estimate magnetic moment of a hemispherical cap.

    Args:
        diameter_nm: core particle diameter
        cap_thickness_nm: cap coating thickness
        cap_fraction: fraction of surface covered
        cap_magnetization: "magnetite" (Fe3O4) or "hematite" (α-Fe2O3)

    Returns:
        magnetic moment in A·m²
    """
    Ms = _M_MAGNETITE if cap_magnetization == "magnetite" else _M_HEMATITE
    # Cap volume: hemispherical shell
    r_core = diameter_nm * 1e-9 / 2
    r_outer = r_core + cap_thickness_nm * 1e-9
    V_shell = (2 / 3) * math.pi * (r_outer**3 - r_core**3) * cap_fraction
    return Ms * V_shell


def magnetic_response(
    diameter_nm: float,
    core_material: str,
    cap_material: str = "Fe2O3",
    cap_thickness_nm: float = 25.0,
    n_medium: float = 1.0,
    packing_fraction: float = 0.55,
    B_fields_mT: list = None,
    temperature_K: float = 298.0,
    cap_magnetization: str = "magnetite",
    wavelengths_nm: np.ndarray = None,
) -> StimulusResponse:
    """Compute color vs magnetic field strength for Janus particles.

    Args:
        diameter_nm:       Core particle diameter
        core_material:     Core material
        cap_material:      Magnetic cap material
        cap_thickness_nm:  Cap thickness
        n_medium:          Medium n
        packing_fraction:  φ
        B_fields_mT:       List of field strengths in mT (default: 0→500)
        temperature_K:     Temperature
        cap_magnetization: "magnetite" or "hematite"
        wavelengths_nm:    Wavelength array

    Returns:
        StimulusResponse with ColorState per field strength
    """
    if B_fields_mT is None:
        B_fields_mT = [0, 5, 20, 50, 100, 200, 500]
    if wavelengths_nm is None:
        wavelengths_nm = _LAM

    from optical.janus_sphere import janus_photonic_glass_reflectance, janus_Q_back

    m = magnetic_cap_moment(diameter_nm, cap_thickness_nm,
                            cap_magnetization=cap_magnetization)

    states = []
    for B_mT in B_fields_mT:
        B = B_mT * 1e-3  # convert mT to T
        xi = m * B / (_KB * temperature_K) if B > 0 else 0.0
        L = _langevin(xi)
        f_cap_up = (1 + L) / 2  # fraction oriented cap-up

        # Compute spectrum as weighted mix of cap-up and cap-down
        R = np.zeros(len(wavelengths_nm))
        for i, lam in enumerate(wavelengths_nm):
            Q_up = janus_Q_back(diameter_nm, core_material, cap_material,
                                 cap_thickness_nm, n_medium, lam,
                                 orientation="cap_up")
            Q_down = janus_Q_back(diameter_nm, core_material, cap_material,
                                   cap_thickness_nm, n_medium, lam,
                                   orientation="cap_down")
            Q_eff = f_cap_up * Q_up + (1 - f_cap_up) * Q_down

            # Simplified PG model inline (avoid full recompute)
            from optical.refractive_index import n_real
            from optical.structure_factor import structure_factor_PY
            n_sph = n_real(core_material, lam)
            n_eff = math.sqrt(packing_fraction * n_sph**2
                              + (1 - packing_fraction) * n_medium**2)
            q_back = 4 * math.pi * n_eff / lam
            D_eff = diameter_nm + 2 * cap_thickness_nm
            Sq = structure_factor_PY(q_back, D_eff, packing_fraction)
            C_back = (math.pi / 4) * D_eff**2 * Q_eff
            R[i] = C_back * Sq

        Rmax = R.max()
        R = R / Rmax if Rmax > 0 else R

        peak, xy, Lab, srgb = _color_from_R(R, wavelengths_nm)
        states.append(ColorState(
            label=f"B={B_mT}mT (ξ={xi:.1f}, f_cap={f_cap_up:.2f})",
            stimulus_value=B_mT,
            R_spectrum=R, peak_nm=peak, cie_xy=xy, Lab=Lab, sRGB=srgb,
        ))

    # Compute range
    peaks = [s.peak_nm for s in states]
    dE = cie_delta_E(states[0].Lab, states[-1].Lab) if len(states) > 1 else 0.0

    return StimulusResponse(
        mechanism="magnetic",
        core_material=core_material,
        core_diameter_nm=diameter_nm,
        states=states,
        delta_E_range=round(dE, 1),
        delta_lambda_nm=round(max(peaks) - min(peaks), 1),
        notes=f"m={m:.2e} A·m², {cap_magnetization} cap",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. THERMAL — PNIPAM SPACER COLLAPSE/SWELL
# ═══════════════════════════════════════════════════════════════════════════
#
# PNIPAM LCST ≈ 32°C in water.
# Below LCST: swollen coil, R_h ∝ N^0.59 (good solvent)
# Above LCST: collapsed globule, R_h ∝ N^0.33 (poor solvent)
#
# Spacer length model:
#   L_swollen = L_max × (T_LCST - T) / (T_LCST - T_low)  for T < LCST
#   L_collapsed = L_min                                     for T > LCST
#   Smooth transition via sigmoid around LCST.
#
# L_max from PEG-equivalent extended length of PNIPAM chain.
# L_min ≈ L_max × (0.33/0.59)^(1/3) ≈ L_max × 0.42
# ═══════════════════════════════════════════════════════════════════════════

_LCST_PNIPAM = 32.0  # °C
_TRANSITION_WIDTH = 2.0  # °C (sharpness of the coil-globule transition)


@dataclass
class PNIPAMSpacer:
    """PNIPAM thermoresponsive spacer properties."""
    name: str
    n_repeat_units: int
    L_swollen_nm: float      # length below LCST (good solvent)
    L_collapsed_nm: float    # length above LCST (poor solvent)
    n_swollen: float = 1.36  # n of hydrated PNIPAM
    n_collapsed: float = 1.50 # n of dehydrated PNIPAM


PNIPAM_SPACERS = {
    "PNIPAM_10":  PNIPAMSpacer("PNIPAM_10", 10, 2.5, 1.0, 1.36, 1.50),
    "PNIPAM_25":  PNIPAMSpacer("PNIPAM_25", 25, 5.0, 2.1, 1.36, 1.50),
    "PNIPAM_50":  PNIPAMSpacer("PNIPAM_50", 50, 8.0, 3.4, 1.36, 1.50),
    "PNIPAM_100": PNIPAMSpacer("PNIPAM_100", 100, 13.0, 5.5, 1.36, 1.50),
}


def pnipam_length_at_T(spacer_name, T_celsius):
    """Compute PNIPAM spacer length at a given temperature.

    Sigmoid transition around LCST.

    Returns (length_nm, n_effective)
    """
    sp = PNIPAM_SPACERS.get(spacer_name)
    if sp is None:
        raise ValueError(f"Unknown PNIPAM spacer: {spacer_name}. "
                         f"Options: {list(PNIPAM_SPACERS)}")

    # Sigmoid: fraction collapsed
    f_collapsed = 1.0 / (1.0 + math.exp(-(T_celsius - _LCST_PNIPAM) / _TRANSITION_WIDTH))

    L = sp.L_swollen_nm * (1 - f_collapsed) + sp.L_collapsed_nm * f_collapsed
    n = sp.n_swollen * (1 - f_collapsed) + sp.n_collapsed * f_collapsed

    return L, n


def thermal_response(
    diameter_nm: float,
    core_material: str,
    anchor: str,
    pnipam_spacer: str,
    click: str,
    n_medium: float = 1.33,  # water
    packing_fraction: float = 0.55,
    temperatures_C: list = None,
    wavelengths_nm: np.ndarray = None,
) -> StimulusResponse:
    """Compute color vs temperature for PNIPAM-functionalized particles.

    Args:
        diameter_nm:     Core diameter
        core_material:   Core material
        anchor:          Anchor chemistry
        pnipam_spacer:   PNIPAM spacer name (e.g. "PNIPAM_25")
        click:           Click chemistry
        n_medium:        Medium n (1.33 for water)
        packing_fraction: φ
        temperatures_C:  List of temperatures (default: 20→50)
        wavelengths_nm:  Wavelength array

    Returns:
        StimulusResponse with ColorState per temperature
    """
    if temperatures_C is None:
        temperatures_C = [20, 25, 28, 30, 31, 32, 33, 34, 36, 40, 50]
    if wavelengths_nm is None:
        wavelengths_nm = _LAM

    from optical.click_linker import (
        compute_chain, ANCHORS, CLICK_PAIRS,
        effective_diameter, effective_packing,
    )
    from optical.photonic_glass import photonic_glass_reflectance

    # Get anchor and click lengths (constant with T)
    anc = ANCHORS.get(anchor)
    clk = CLICK_PAIRS.get(click)
    L_anchor = anc.length_nm if anc else 0.0
    L_click = clk.half_length_nm if clk else 0.0

    states = []
    for T in temperatures_C:
        L_pnipam, n_pnipam = pnipam_length_at_T(pnipam_spacer, T)

        # Total shell per side
        L_total = L_anchor + L_pnipam + L_click
        D_eff = diameter_nm + 2 * L_total
        phi_eff = packing_fraction * (diameter_nm / D_eff)**3

        # Forward model with effective diameter
        # Use bare photonic glass model but with modified D and φ
        R = photonic_glass_reflectance(
            D_eff, core_material, n_medium, phi_eff, wavelengths_nm)

        peak, xy, Lab, srgb = _color_from_R(R, wavelengths_nm)
        states.append(ColorState(
            label=f"T={T}°C (L_spacer={L_pnipam:.1f}nm)",
            stimulus_value=T,
            R_spectrum=R, peak_nm=peak, cie_xy=xy, Lab=Lab, sRGB=srgb,
        ))

    peaks = [s.peak_nm for s in states]
    dE = cie_delta_E(states[0].Lab, states[-1].Lab) if len(states) > 1 else 0.0

    return StimulusResponse(
        mechanism="thermal_PNIPAM",
        core_material=core_material,
        core_diameter_nm=diameter_nm,
        states=states,
        delta_E_range=round(dE, 1),
        delta_lambda_nm=round(max(peaks) - min(peaks), 1),
        notes=f"LCST={_LCST_PNIPAM}°C, spacer={pnipam_spacer}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. UV/IR PHOTOCHROMIC — AZOBENZENE / SPIROPYRAN
# ═══════════════════════════════════════════════════════════════════════════
#
# Azobenzene: trans (9.0 Å, n≈1.6) ↔ cis (5.5 Å, n≈1.7) under UV (365nm)
#   Back-conversion: visible (450nm) or thermal (kT)
#
# Spiropyran: closed (no absorption, L≈0.6nm) ↔ open merocyanine
#   (absorbs ~550nm, L≈1.0nm, n≈1.7) under UV (365nm)
#   Back-conversion: visible light or heat
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Photochrome:
    name: str
    L_state_A_nm: float      # length in state A (dark/ground)
    L_state_B_nm: float      # length in state B (irradiated)
    n_state_A: float
    n_state_B: float
    trigger_A_to_B: str      # "UV_365nm", etc.
    trigger_B_to_A: str
    k_absorption_B: float    # extinction coefficient in state B (0 if transparent)


PHOTOCHROMES = {
    "azobenzene": Photochrome(
        "azobenzene",
        L_state_A_nm=0.9,   # trans
        L_state_B_nm=0.55,  # cis (shorter!)
        n_state_A=1.60, n_state_B=1.70,
        trigger_A_to_B="UV_365nm",
        trigger_B_to_A="visible_450nm_or_heat",
        k_absorption_B=0.0,
    ),
    "spiropyran": Photochrome(
        "spiropyran",
        L_state_A_nm=0.6,   # closed (transparent)
        L_state_B_nm=1.0,   # open merocyanine (absorbs ~550nm)
        n_state_A=1.55, n_state_B=1.70,
        trigger_A_to_B="UV_365nm",
        trigger_B_to_A="visible_or_heat",
        k_absorption_B=0.15,  # absorbs green/yellow
    ),
    "diarylethene": Photochrome(
        "diarylethene",
        L_state_A_nm=0.7,   # open (transparent)
        L_state_B_nm=0.7,   # closed (absorbs red, same length)
        n_state_A=1.55, n_state_B=1.65,
        trigger_A_to_B="UV_300nm",
        trigger_B_to_A="visible_500nm",
        k_absorption_B=0.10,  # absorbs red/orange
    ),
}


def photochromic_response(
    diameter_nm: float,
    core_material: str,
    anchor: str,
    photochrome: str,
    click: str,
    n_medium: float = 1.0,
    packing_fraction: float = 0.55,
    conversion_fractions: list = None,
    wavelengths_nm: np.ndarray = None,
) -> StimulusResponse:
    """Compute color vs photochromic conversion for UV/vis-responsive particles.

    Args:
        diameter_nm:          Core diameter
        core_material:        Core material
        anchor:               Anchor chemistry
        photochrome:          Photochrome name ("azobenzene", "spiropyran", etc.)
        click:                Click chemistry
        n_medium:             Medium n
        packing_fraction:     φ
        conversion_fractions: List of A→B conversion fractions (0=all A, 1=all B)
        wavelengths_nm:       Wavelength array

    Returns:
        StimulusResponse with ColorState per conversion fraction
    """
    if conversion_fractions is None:
        conversion_fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
    if wavelengths_nm is None:
        wavelengths_nm = _LAM

    pc = PHOTOCHROMES.get(photochrome)
    if pc is None:
        raise ValueError(f"Unknown photochrome: {photochrome}. "
                         f"Options: {list(PHOTOCHROMES)}")

    from optical.click_linker import ANCHORS, CLICK_PAIRS
    from optical.photonic_glass import photonic_glass_reflectance

    anc = ANCHORS.get(anchor)
    clk = CLICK_PAIRS.get(click)
    L_anchor = anc.length_nm if anc else 0.0
    L_click = clk.half_length_nm if clk else 0.0

    states = []
    for f_B in conversion_fractions:
        # Interpolate length and n between states
        L_pc = pc.L_state_A_nm * (1 - f_B) + pc.L_state_B_nm * f_B
        n_pc = pc.n_state_A * (1 - f_B) + pc.n_state_B * f_B

        L_total = L_anchor + L_pc + L_click
        D_eff = diameter_nm + 2 * L_total
        phi_eff = packing_fraction * (diameter_nm / D_eff)**3

        # Absorber from photochrome (if state B absorbs)
        abs_frac = f_B * pc.k_absorption_B * 0.01  # scale to volume fraction

        R = photonic_glass_reflectance(
            D_eff, core_material, n_medium, phi_eff, wavelengths_nm,
            absorber_fraction=abs_frac)

        peak, xy, Lab, srgb = _color_from_R(R, wavelengths_nm)
        state_label = "dark" if f_B == 0 else f"UV {f_B*100:.0f}%"
        states.append(ColorState(
            label=f"{state_label} (L_pc={L_pc:.2f}nm, n={n_pc:.2f})",
            stimulus_value=f_B,
            R_spectrum=R, peak_nm=peak, cie_xy=xy, Lab=Lab, sRGB=srgb,
        ))

    peaks = [s.peak_nm for s in states]
    dE = cie_delta_E(states[0].Lab, states[-1].Lab) if len(states) > 1 else 0.0

    return StimulusResponse(
        mechanism=f"photochromic_{photochrome}",
        core_material=core_material,
        core_diameter_nm=diameter_nm,
        states=states,
        delta_E_range=round(dE, 1),
        delta_lambda_nm=round(max(peaks) - min(peaks), 1),
        notes=f"{pc.trigger_A_to_B} → {pc.trigger_B_to_A}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. ELECTRIC FIELD — DIELECTROPHORETIC JANUS ROTATION
# ═══════════════════════════════════════════════════════════════════════════
#
# AC electric field induces dipole in Janus particle.
# Dielectric contrast between cap and base creates torque.
# Orientation follows Boltzmann: P(θ) ∝ exp(-U(θ)/kT)
# For linear dipole: U ∝ -E² (induced, not permanent)
# Effective alignment parameter: ξ_e = ε₀ Δε V E² / (2kT)
# ═══════════════════════════════════════════════════════════════════════════

_EPS0 = 8.854e-12  # F/m

# Approximate dielectric constants at ~1 MHz
_DIELECTRIC = {
    "SiO2": 3.9,
    "polystyrene": 2.6,
    "TiO2_rutile": 86.0,
    "Fe2O3": 25.0,
    "carbon": 12.0,
    "Au": 1e6,  # metallic
    "water": 80.0,
    "air": 1.0,
}


def electric_response(
    diameter_nm: float,
    core_material: str,
    cap_material: str,
    cap_thickness_nm: float = 25.0,
    n_medium: float = 1.33,
    packing_fraction: float = 0.55,
    E_fields_Vmm: list = None,
    temperature_K: float = 298.0,
    medium_dielectric: float = 80.0,
    wavelengths_nm: np.ndarray = None,
) -> StimulusResponse:
    """Compute color vs AC electric field for dielectrophoretic Janus rotation.

    Args:
        diameter_nm:       Core diameter
        core_material:     Core material
        cap_material:      Cap material (dielectric contrast drives torque)
        cap_thickness_nm:  Cap thickness
        n_medium:          Medium n
        packing_fraction:  φ
        E_fields_Vmm:      List of field strengths in V/mm
        temperature_K:     Temperature
        medium_dielectric: Relative permittivity of medium
        wavelengths_nm:    Wavelength array

    Returns:
        StimulusResponse
    """
    if E_fields_Vmm is None:
        E_fields_Vmm = [0, 10, 50, 100, 500, 1000]
    if wavelengths_nm is None:
        wavelengths_nm = _LAM

    from optical.janus_sphere import janus_Q_back
    from optical.refractive_index import n_real
    from optical.structure_factor import structure_factor_PY

    eps_cap = _DIELECTRIC.get(cap_material, 5.0)
    eps_base = _DIELECTRIC.get(core_material, 3.0)
    delta_eps = abs(eps_cap - eps_base)

    # Effective volume of asymmetric dielectric
    r = diameter_nm * 1e-9 / 2
    V_eff = (4 / 3) * math.pi * r**3 * 0.5  # half-sphere

    states = []
    for E_Vmm in E_fields_Vmm:
        E = E_Vmm * 1e3  # V/mm → V/m
        # Alignment energy
        U = _EPS0 * delta_eps * V_eff * E**2 / 2
        xi_e = U / (_KB * temperature_K) if E > 0 else 0.0
        L = _langevin(xi_e)
        f_aligned = (1 + L) / 2

        # Compute spectrum
        D_eff = diameter_nm + 2 * cap_thickness_nm
        R = np.zeros(len(wavelengths_nm))
        for i, lam in enumerate(wavelengths_nm):
            Q_up = janus_Q_back(diameter_nm, core_material, cap_material,
                                 cap_thickness_nm, n_medium, lam,
                                 orientation="cap_up")
            Q_down = janus_Q_back(diameter_nm, core_material, cap_material,
                                   cap_thickness_nm, n_medium, lam,
                                   orientation="cap_down")
            Q_eff = f_aligned * Q_up + (1 - f_aligned) * Q_down

            n_sph = n_real(core_material, lam)
            n_eff = math.sqrt(packing_fraction * n_sph**2
                              + (1 - packing_fraction) * n_medium**2)
            q_back = 4 * math.pi * n_eff / lam
            Sq = structure_factor_PY(q_back, D_eff, packing_fraction)
            R[i] = (math.pi / 4) * D_eff**2 * Q_eff * Sq

        Rmax = R.max()
        R = R / Rmax if Rmax > 0 else R

        peak, xy, Lab, srgb = _color_from_R(R, wavelengths_nm)
        states.append(ColorState(
            label=f"E={E_Vmm}V/mm (ξ={xi_e:.1f}, f={f_aligned:.2f})",
            stimulus_value=E_Vmm,
            R_spectrum=R, peak_nm=peak, cie_xy=xy, Lab=Lab, sRGB=srgb,
        ))

    peaks = [s.peak_nm for s in states]
    dE = cie_delta_E(states[0].Lab, states[-1].Lab) if len(states) > 1 else 0.0

    return StimulusResponse(
        mechanism="electric_DEP",
        core_material=core_material,
        core_diameter_nm=diameter_nm,
        states=states,
        delta_E_range=round(dE, 1),
        delta_lambda_nm=round(max(peaks) - min(peaks), 1),
        notes=f"Δε={delta_eps:.0f}, cap={cap_material}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. ASSEMBLY-DIRECTED: OPAL vs GLASS FROM SAME PARTICLES
# ═══════════════════════════════════════════════════════════════════════════

def assembly_comparison(
    diameter_nm: float,
    core_material: str,
    anchor: str = "APTES",
    spacer: str = "PEG4",
    click: str = "SPAAC",
    n_medium: float = 1.0,
    packing_fraction: float = 0.55,
    wavelengths_nm: np.ndarray = None,
) -> StimulusResponse:
    """Compare same functionalized particles in ordered vs disordered assembly.

    Ordered: Bragg opal (M2) — iridescent, angle-dependent
    Disordered: photonic glass (M6) — non-iridescent, angle-independent

    Args:
        diameter_nm:     Core diameter
        core_material:   Core material
        anchor, spacer, click: attachment chain
        n_medium:        Medium n
        packing_fraction: φ
        wavelengths_nm:  Wavelength array

    Returns:
        StimulusResponse with two states: "ordered_opal" and "disordered_glass"
    """
    if wavelengths_nm is None:
        wavelengths_nm = _LAM

    from optical.click_linker import (
        compute_chain, effective_diameter, effective_packing,
    )
    from optical.photonic_glass import photonic_glass_reflectance
    from optical.bragg_opal import bragg_opal
    from optical.refractive_index import n_real

    chain = compute_chain(core_material, anchor, spacer, click)
    D_eff = effective_diameter(diameter_nm, chain) if chain.compatible else diameter_nm
    phi_eff = effective_packing(packing_fraction, diameter_nm, chain) if chain.compatible else packing_fraction

    # Bragg opal: peak from M2
    n_sph_550 = n_real(core_material, 550)
    n_eff_approx = math.sqrt(phi_eff * n_sph_550**2
                             + (1 - phi_eff) * n_medium**2)
    bragg_peak = bragg_opal(D_eff, n_sph_550, n_medium, phi_eff)

    # Make a simple Gaussian Bragg reflectance for CIE
    sigma_bragg = D_eff * 0.05  # ~5% of diameter as bandwidth
    R_bragg = np.exp(-0.5 * ((wavelengths_nm - bragg_peak) / max(sigma_bragg, 5))**2)

    # Photonic glass from M6
    R_glass = photonic_glass_reflectance(D_eff, core_material, n_medium,
                                          phi_eff, wavelengths_nm)

    states = []
    for label, R in [("ordered_opal", R_bragg), ("disordered_glass", R_glass)]:
        peak, xy, Lab, srgb = _color_from_R(R, wavelengths_nm)
        states.append(ColorState(
            label=label,
            stimulus_value=1.0 if "opal" in label else 0.0,
            R_spectrum=R, peak_nm=peak, cie_xy=xy, Lab=Lab, sRGB=srgb,
        ))

    dE = cie_delta_E(states[0].Lab, states[1].Lab)

    return StimulusResponse(
        mechanism="assembly_directed",
        core_material=core_material,
        core_diameter_nm=diameter_nm,
        states=states,
        delta_E_range=round(dE, 1),
        delta_lambda_nm=round(abs(states[0].peak_nm - states[1].peak_nm), 1),
        notes=f"D_eff={D_eff:.0f}nm, Bragg={bragg_peak:.0f}nm, same particles",
    )


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_response(resp):
    """Pretty-print a StimulusResponse."""
    print()
    print(f"  MABE Responsive Color — {resp.mechanism}")
    print(f"  Core: {resp.core_material} {resp.core_diameter_nm:.0f}nm")
    print(f"  Δλ: {resp.delta_lambda_nm:.0f}nm   ΔE: {resp.delta_E_range:.1f}")
    if resp.notes:
        print(f"  {resp.notes}")
    print()
    print(f"  {'State':40s}  {'Peak':>5s}  {'CIE xy':14s}  sRGB")
    print(f"  {'─'*80}")
    for s in resp.states:
        print(f"  {s.label:40s}  {s.peak_nm:5.0f}  "
              f"({s.cie_xy[0]:.3f},{s.cie_xy[1]:.3f})  {s.sRGB}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MABE Responsive Color — Self-Test")
    print("=" * 70)

    # 1. Magnetic
    print("\n--- 1. Magnetic (PS 250nm + magnetite cap) ---")
    r1 = magnetic_response(250, "polystyrene", "Fe2O3", 25,
                            B_fields_mT=[0, 50, 200, 500])
    print_response(r1)

    # 2. Thermal PNIPAM
    print("--- 2. Thermal PNIPAM (SiO2 200nm) ---")
    r2 = thermal_response(200, "SiO2", "APTES", "PNIPAM_25", "SPAAC",
                           n_medium=1.33,
                           temperatures_C=[20, 28, 32, 36, 50])
    print_response(r2)

    # 3. UV photochromic
    print("--- 3. UV azobenzene (SiO2 200nm) ---")
    r3 = photochromic_response(200, "SiO2", "APTES", "azobenzene", "SPAAC")
    print_response(r3)

    # 4. Electric
    print("--- 4. Electric DEP (SiO2 200nm + TiO2 cap) ---")
    r4 = electric_response(200, "SiO2", "TiO2_rutile", 30,
                            E_fields_Vmm=[0, 100, 500, 1000])
    print_response(r4)

    # 5. Assembly comparison
    print("--- 5. Assembly: opal vs glass (SiO2 200nm) ---")
    r5 = assembly_comparison(200, "SiO2")
    print_response(r5)
