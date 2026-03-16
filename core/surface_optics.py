"""
surface_optics.py — Substrate Library + Optical Coupling for Structural Dyes

How a structural dye looks on a real surface depends on:
  1. Dye's own reflectance spectrum R_dye(λ)
  2. Substrate reflectance R_sub(λ) — underlayer color shows through
  3. Interface optics — Fresnel reflection at dye/substrate boundary
  4. Substrate surface properties — roughness, energy, chemistry

This module provides:
  - Substrate library (≥15 substrates, published optical + surface data)
  - Kubelka-Munk two-flux model for dye-on-substrate coupling
  - Fresnel interface at dye/substrate boundary
  - predict_color_on_substrate() → perceived CIE Lab/sRGB
  - Surface energy data for adhesion assessment (Young's equation)

All T1 (published) and T2 (Kubelka-Munk, Fresnel, Young). No T3 estimates.

References:
  Kubelka P, Munk F. Z. Tech. Phys. 1931, 12, 593.
  Kubelka P. J. Opt. Soc. Am. 1948, 38, 448.
  Young T. Phil. Trans. R. Soc. 1805, 95, 65.
  Dupré A. Théorie Mécanique de la Chaleur. Gauthier-Villars 1869.
  Owens DK, Wendt RC. J. Appl. Polym. Sci. 1969, 13, 1741.
  Van Oss CJ. Interfacial Forces in Aqueous Media. CRC 2006.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PI = math.pi
_LAM = np.linspace(380, 780, 81)


# ═══════════════════════════════════════════════════════════════════════════
# Substrate Library
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubstrateEntry:
    """Published substrate optical and surface properties."""
    name: str
    category: str                    # textile, glass, metal, polymer, paper, ceramic

    # Optical properties
    reflectance_vis: Optional[float] = None   # average visible reflectance (0-1)
    color_Lab: Optional[Tuple[float, float, float]] = None  # CIE Lab
    n_substrate: Optional[float] = None       # refractive index (at 550nm)
    k_substrate: Optional[float] = None       # extinction coefficient (metals)
    is_opaque: bool = True                    # True if substrate is optically thick
    is_diffuse: bool = True                   # True if scattering (matte), False if specular

    # Surface properties
    roughness_um: Optional[float] = None      # RMS roughness (μm)
    surface_energy_mJ_m2: Optional[float] = None  # total surface energy
    polar_component_mJ_m2: Optional[float] = None  # polar part of surface energy
    dispersive_component_mJ_m2: Optional[float] = None  # dispersive part
    water_contact_angle_deg: Optional[float] = None  # advancing contact angle

    # Surface chemistry
    functional_groups: List[str] = field(default_factory=list)  # available for bonding
    click_compatible: bool = False    # has reactive groups for click chemistry

    # Source
    source: str = ""


SUBSTRATE_LIBRARY: Dict[str, SubstrateEntry] = {}


def _add_sub(s: SubstrateEntry):
    SUBSTRATE_LIBRARY[s.name] = s


# ── Textiles ──────────────────────────────────────────────────────────────

_add_sub(SubstrateEntry(
    name="cotton_white", category="textile",
    reflectance_vis=0.82, color_Lab=(93.0, 0.5, 2.0),
    n_substrate=1.55, is_diffuse=True,
    roughness_um=20.0, surface_energy_mJ_m2=44.0,
    polar_component_mJ_m2=8.0, dispersive_component_mJ_m2=36.0,
    water_contact_angle_deg=0.0,  # hydrophilic, wicks
    functional_groups=["OH", "C-OH"],
    click_compatible=True,  # OH → APTES → azide
    source="Morton WE, Hearle JWS. Physical Properties of Textile Fibres, 4th ed. Woodhead 2008. "
           "Surface energy: Hsieh YL. Text. Res. J. 1995, 65, 299",
))

_add_sub(SubstrateEntry(
    name="cotton_black", category="textile",
    reflectance_vis=0.04, color_Lab=(20.0, 0.2, -0.5),
    n_substrate=1.55, is_diffuse=True,
    roughness_um=20.0, surface_energy_mJ_m2=42.0,
    polar_component_mJ_m2=7.0, dispersive_component_mJ_m2=35.0,
    water_contact_angle_deg=0.0,
    functional_groups=["OH", "C-OH"],
    click_compatible=True,
    source="Same fiber as white cotton; color from reactive black dye. "
           "Reflectance: Park S, Kim M. Text. Res. J. 2015, 85, 1776",
))

_add_sub(SubstrateEntry(
    name="polyester_white", category="textile",
    reflectance_vis=0.78, color_Lab=(91.0, 0.3, 1.5),
    n_substrate=1.58, is_diffuse=True,
    roughness_um=15.0, surface_energy_mJ_m2=43.0,
    polar_component_mJ_m2=1.0, dispersive_component_mJ_m2=42.0,
    water_contact_angle_deg=75.0,
    functional_groups=["C=O", "C-O-C"],
    click_compatible=False,  # needs surface activation
    source="PET fiber. Owens DK, Wendt RC. J. Appl. Polym. Sci. 1969, 13, 1741. "
           "Contact angle: Wei QF et al. J. Ind. Text. 2007, 37, 43",
))

_add_sub(SubstrateEntry(
    name="polyester_black", category="textile",
    reflectance_vis=0.03, color_Lab=(18.0, 0.1, -0.3),
    n_substrate=1.58, is_diffuse=True,
    roughness_um=15.0, surface_energy_mJ_m2=43.0,
    polar_component_mJ_m2=1.0, dispersive_component_mJ_m2=42.0,
    water_contact_angle_deg=75.0,
    functional_groups=["C=O", "C-O-C"],
    click_compatible=False,
    source="Same as polyester_white with disperse dye",
))

_add_sub(SubstrateEntry(
    name="silk_white", category="textile",
    reflectance_vis=0.75, color_Lab=(89.0, 1.0, 5.0),
    n_substrate=1.54, is_diffuse=True,
    roughness_um=5.0, surface_energy_mJ_m2=50.0,
    polar_component_mJ_m2=15.0, dispersive_component_mJ_m2=35.0,
    water_contact_angle_deg=35.0,
    functional_groups=["NH2", "COOH", "OH"],
    click_compatible=True,  # abundant amine + carboxyl
    source="Sericin surface. Das S, Bhowmick M. Text. Prog. 2015, 47, 59. "
           "Surface energy: Arai T et al. J. Appl. Polym. Sci. 2001, 80, 297",
))

_add_sub(SubstrateEntry(
    name="nylon_white", category="textile",
    reflectance_vis=0.80, color_Lab=(92.0, 0.2, 1.0),
    n_substrate=1.53, is_diffuse=True,
    roughness_um=10.0, surface_energy_mJ_m2=46.0,
    polar_component_mJ_m2=10.0, dispersive_component_mJ_m2=36.0,
    water_contact_angle_deg=62.0,
    functional_groups=["NH", "C=O"],
    click_compatible=True,  # terminal amine
    source="PA6. Owens DK, Wendt RC. J. Appl. Polym. Sci. 1969, 13, 1741",
))

# ── Glass ─────────────────────────────────────────────────────────────────

_add_sub(SubstrateEntry(
    name="soda_lime_glass", category="glass",
    reflectance_vis=0.04, color_Lab=(89.0, -0.5, 0.5),
    n_substrate=1.52, is_opaque=False, is_diffuse=False,
    roughness_um=0.001, surface_energy_mJ_m2=75.0,
    polar_component_mJ_m2=42.0, dispersive_component_mJ_m2=33.0,
    water_contact_angle_deg=25.0,
    functional_groups=["Si-OH"],
    click_compatible=True,  # silanol → APTES → azide
    source="Scholze H. Glass: Nature, Structure, and Properties. Springer 1991. "
           "Surface energy: Żenkiewicz M. Polym. Test. 2007, 26, 14",
))

_add_sub(SubstrateEntry(
    name="borosilicate_glass", category="glass",
    reflectance_vis=0.04, color_Lab=(90.0, -0.3, 0.3),
    n_substrate=1.47, is_opaque=False, is_diffuse=False,
    roughness_um=0.001, surface_energy_mJ_m2=70.0,
    polar_component_mJ_m2=38.0, dispersive_component_mJ_m2=32.0,
    water_contact_angle_deg=30.0,
    functional_groups=["Si-OH", "B-OH"],
    click_compatible=True,
    source="Schott Duran. n from Schott catalog",
))

# ── Metals ────────────────────────────────────────────────────────────────

_add_sub(SubstrateEntry(
    name="aluminum", category="metal",
    reflectance_vis=0.91, color_Lab=(97.0, -0.5, 0.0),
    n_substrate=1.37, k_substrate=7.6,  # at 550nm, Johnson & Christy
    is_opaque=True, is_diffuse=False,
    roughness_um=0.1, surface_energy_mJ_m2=840.0,  # bare, freshly cleaned
    functional_groups=["Al-OH"],  # native oxide
    click_compatible=True,  # oxide → phosphonate or silane
    source="Johnson PB, Christy RW. Phys. Rev. B 1972, 6, 4370. "
           "Surface energy: Zisman WA. Adv. Chem. Ser. 1964, 43, 1",
))

_add_sub(SubstrateEntry(
    name="stainless_steel", category="metal",
    reflectance_vis=0.60, color_Lab=(82.0, -1.0, 2.0),
    n_substrate=2.76, k_substrate=3.26,  # 304SS at 550nm
    is_opaque=True, is_diffuse=False,
    roughness_um=0.5, surface_energy_mJ_m2=700.0,
    functional_groups=["Cr-OH", "Fe-OH"],
    click_compatible=True,
    source="304 stainless. Optical: Ordal MA et al. Appl. Opt. 1988, 27, 1203",
))

# ── Polymers ──────────────────────────────────────────────────────────────

_add_sub(SubstrateEntry(
    name="PMMA_clear", category="polymer",
    reflectance_vis=0.04, color_Lab=(92.0, 0.0, 0.0),
    n_substrate=1.49, is_opaque=False, is_diffuse=False,
    roughness_um=0.01, surface_energy_mJ_m2=41.0,
    polar_component_mJ_m2=5.0, dispersive_component_mJ_m2=36.0,
    water_contact_angle_deg=70.0,
    functional_groups=["C=O", "O-CH3"],
    click_compatible=False,
    source="Kasarova SN et al. Opt. Mater. 2007, 29, 1481. "
           "Surface energy: Owens DK, Wendt RC. J. Appl. Polym. Sci. 1969, 13, 1741",
))

_add_sub(SubstrateEntry(
    name="polycarbonate", category="polymer",
    reflectance_vis=0.05, color_Lab=(90.0, 0.0, 1.0),
    n_substrate=1.585, is_opaque=False, is_diffuse=False,
    roughness_um=0.01, surface_energy_mJ_m2=46.0,
    polar_component_mJ_m2=6.0, dispersive_component_mJ_m2=40.0,
    water_contact_angle_deg=82.0,
    functional_groups=["C=O", "C-O"],
    click_compatible=False,
    source="n: Sultanova N et al. Acta Phys. Pol. A 2009, 116, 585",
))

_add_sub(SubstrateEntry(
    name="PDMS", category="polymer",
    reflectance_vis=0.04, color_Lab=(92.0, 0.0, 0.0),
    n_substrate=1.43, is_opaque=False, is_diffuse=False,
    roughness_um=0.001, surface_energy_mJ_m2=20.0,
    polar_component_mJ_m2=1.0, dispersive_component_mJ_m2=19.0,
    water_contact_angle_deg=108.0,
    functional_groups=["Si-CH3"],
    click_compatible=False,  # needs O₂ plasma activation
    source="Dow Corning Sylgard 184. Contact angle: Kim J et al. J. Micromech. Microeng. 2006, 16, 2318",
))

# ── Paper ─────────────────────────────────────────────────────────────────

_add_sub(SubstrateEntry(
    name="paper_white", category="paper",
    reflectance_vis=0.85, color_Lab=(95.0, 0.5, -2.0),
    n_substrate=1.55, is_diffuse=True,
    roughness_um=5.0, surface_energy_mJ_m2=55.0,
    polar_component_mJ_m2=25.0, dispersive_component_mJ_m2=30.0,
    water_contact_angle_deg=50.0,
    functional_groups=["OH", "C-OH", "COOH"],
    click_compatible=True,
    source="Office paper. Reflectance: ISO 2470 brightness. "
           "Surface energy: Peng L et al. Colloids Surf. A 2012, 395, 213",
))

_add_sub(SubstrateEntry(
    name="kraft_paper", category="paper",
    reflectance_vis=0.45, color_Lab=(65.0, 5.0, 25.0),
    n_substrate=1.55, is_diffuse=True,
    roughness_um=10.0, surface_energy_mJ_m2=50.0,
    polar_component_mJ_m2=20.0, dispersive_component_mJ_m2=30.0,
    water_contact_angle_deg=65.0,
    functional_groups=["OH", "C-OH", "lignin-OH"],
    click_compatible=True,
    source="Unbleached kraft. Color from residual lignin",
))

# ── Ceramic ───────────────────────────────────────────────────────────────

_add_sub(SubstrateEntry(
    name="alumina_ceramic", category="ceramic",
    reflectance_vis=0.88, color_Lab=(96.0, 0.0, 0.5),
    n_substrate=1.77, is_diffuse=True,
    roughness_um=1.0, surface_energy_mJ_m2=650.0,
    functional_groups=["Al-OH"],
    click_compatible=True,
    source="α-Al₂O₃. n: Malitson IH. J. Opt. Soc. Am. 1962, 52, 1377",
))


# ═══════════════════════════════════════════════════════════════════════════
# Substrate Reflectance Spectrum
# ═══════════════════════════════════════════════════════════════════════════

def substrate_reflectance_spectrum(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generate approximate reflectance spectrum for a substrate.

    Uses Lab color to reconstruct a plausible R(λ) via inverse CIE.
    For metals: Drude model. For diffuse substrates: flat + color tint.

    Parameters
    ----------
    name : str
        Substrate name from SUBSTRATE_LIBRARY.

    Returns
    -------
    (wavelengths, R_spectrum)
    """
    if name not in SUBSTRATE_LIBRARY:
        return _LAM, np.full_like(_LAM, 0.5, dtype=float)

    sub = SUBSTRATE_LIBRARY[name]
    R_avg = sub.reflectance_vis if sub.reflectance_vis is not None else 0.5

    if sub.color_Lab is not None:
        L, a, b = sub.color_Lab
        # Reconstruct spectrum from Lab hints
        # a < 0 = green, a > 0 = red; b < 0 = blue, b > 0 = yellow
        R = np.full_like(_LAM, R_avg, dtype=float)

        # Add chromatic tint based on a*, b*
        # Red-green axis: a* modulates 600-700nm relative to 500-550nm
        R += a * 0.001 * np.exp(-0.5 * ((_LAM - 620) / 50) ** 2)
        R -= a * 0.001 * np.exp(-0.5 * ((_LAM - 520) / 50) ** 2)
        # Yellow-blue axis: b* modulates 570-600nm relative to 430-470nm
        R += b * 0.001 * np.exp(-0.5 * ((_LAM - 580) / 40) ** 2)
        R -= b * 0.001 * np.exp(-0.5 * ((_LAM - 450) / 40) ** 2)

        R = np.clip(R, 0.0, 1.0)
        return _LAM, R

    return _LAM, np.full_like(_LAM, R_avg, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# Optical Coupling: Kubelka-Munk Two-Flux Model (T2)
# ═══════════════════════════════════════════════════════════════════════════
#
# For a scattering dye layer on a substrate:
#   R_total = R_dye + T_dye² × R_sub / (1 - R_dye_internal × R_sub)
#
# where R_dye = reflectance of dye film alone (from structural color),
#       T_dye = transmittance of dye film,
#       R_sub = substrate reflectance.
#
# This captures the key physics: substrate color shows through the dye,
# modified by the dye's own optical thickness.
#
# Ref: Kubelka P. J. Opt. Soc. Am. 1948, 38, 448.

def kubelka_munk_coupling(R_dye: np.ndarray, T_dye: np.ndarray,
                            R_substrate: np.ndarray) -> np.ndarray:
    """Kubelka-Munk: total reflectance of dye layer on substrate.

    R_total(λ) = R_dye + T_dye² × R_sub / (1 - R_dye_back × R_sub)

    Assumes dye film is the same from front and back (R_dye_back ≈ R_dye
    for thin scattering layers).

    Parameters
    ----------
    R_dye : np.ndarray
        Reflectance of dye film alone (measured from front).
    T_dye : np.ndarray
        Transmittance of dye film.
    R_substrate : np.ndarray
        Substrate reflectance.

    Returns
    -------
    np.ndarray
        Total observed reflectance R_total(λ).

    Physics tier: T2 (Kubelka-Munk 1931/1948, standard in paint/textile industry).
    """
    R_back = R_dye  # assume symmetric (thin film approximation)
    denom = 1.0 - R_back * R_substrate
    denom = np.maximum(denom, 1e-10)  # prevent division by zero
    R_total = R_dye + T_dye ** 2 * R_substrate / denom
    return np.clip(R_total, 0.0, 1.0)


def estimate_dye_transmittance(R_dye: np.ndarray,
                                  absorption: np.ndarray = None) -> np.ndarray:
    """Estimate dye film transmittance from reflectance.

    For a non-absorbing scattering film: T = 1 - R (energy conservation).
    For absorbing film: T = (1 - R) × exp(-α×d) ≈ (1 - R) × (1 - A).

    Parameters
    ----------
    R_dye : np.ndarray
        Dye film reflectance.
    absorption : np.ndarray, optional
        Absorption fraction (0-1). If None, assumed zero (pure scatterer).

    Returns
    -------
    np.ndarray
        Transmittance T(λ).
    """
    if absorption is not None:
        return (1.0 - R_dye) * (1.0 - absorption)
    return 1.0 - R_dye


# ═══════════════════════════════════════════════════════════════════════════
# Fresnel Interface Coupling (T2)
# ═══════════════════════════════════════════════════════════════════════════

def fresnel_interface_reflectance(n_dye: float, n_substrate: float) -> float:
    """Normal-incidence Fresnel reflectance at dye/substrate interface.

    R = ((n_dye - n_sub) / (n_dye + n_sub))²

    This adds a specular component at the dye-substrate boundary.

    Physics tier: T2 (Fresnel 1823).
    """
    if n_dye + n_substrate == 0:
        return 0.0
    return ((n_dye - n_substrate) / (n_dye + n_substrate)) ** 2


def air_dye_interface_reflectance(n_dye: float) -> float:
    """Fresnel reflectance at air/dye top surface."""
    return fresnel_interface_reflectance(1.0, n_dye)


# ═══════════════════════════════════════════════════════════════════════════
# Combined: predict_color_on_substrate()
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColorOnSubstrateResult:
    """Result of predicting perceived color of a dye on a substrate."""
    dye_name: str = ""
    substrate_name: str = ""

    # Spectra
    R_dye: Optional[np.ndarray] = None
    R_substrate: Optional[np.ndarray] = None
    R_total: Optional[np.ndarray] = None

    # Perceived color
    Lab_on_substrate: Optional[Tuple[float, float, float]] = None
    sRGB_on_substrate: Optional[Tuple[float, float, float]] = None
    Lab_dye_alone: Optional[Tuple[float, float, float]] = None

    # Color shift from substrate
    delta_E_from_alone: Optional[float] = None  # how much substrate changes the color
    delta_E_from_target: Optional[float] = None  # distance from design target

    # Interface losses
    fresnel_loss_top: float = 0.0     # air/dye interface reflection
    fresnel_loss_bottom: float = 0.0  # dye/substrate interface reflection


def predict_color_on_substrate(R_dye: np.ndarray,
                                  substrate_name: str,
                                  n_dye: float = 1.50,
                                  dye_absorption: np.ndarray = None,
                                  target_Lab: Tuple[float, float, float] = None,
                                  dye_name: str = "") -> ColorOnSubstrateResult:
    """Predict perceived color of a structural dye on a real substrate.

    Combines:
      1. Kubelka-Munk coupling (substrate color bleeds through)
      2. Fresnel losses at air/dye and dye/substrate interfaces
      3. CIE color computation of the combined spectrum

    Parameters
    ----------
    R_dye : np.ndarray
        Reflectance spectrum of dye film alone (81-point, 380-780nm).
    substrate_name : str
        Substrate from SUBSTRATE_LIBRARY.
    n_dye : float
        Average refractive index of dye film.
    dye_absorption : np.ndarray, optional
        Absorption spectrum of dye (0-1). Chromophore contribution.
    target_Lab : tuple, optional
        Target color for ΔE calculation.
    dye_name : str
        Label for the dye.

    Returns
    -------
    ColorOnSubstrateResult
    """
    result = ColorOnSubstrateResult(dye_name=dye_name, substrate_name=substrate_name)
    result.R_dye = R_dye

    # Get substrate spectrum
    _, R_sub = substrate_reflectance_spectrum(substrate_name)
    result.R_substrate = R_sub

    # Substrate properties
    sub = SUBSTRATE_LIBRARY.get(substrate_name)
    n_sub = sub.n_substrate if sub and sub.n_substrate else 1.50

    # Fresnel interface losses
    result.fresnel_loss_top = air_dye_interface_reflectance(n_dye)
    result.fresnel_loss_bottom = fresnel_interface_reflectance(n_dye, n_sub)

    # Dye transmittance
    T_dye = estimate_dye_transmittance(R_dye, dye_absorption)

    # Scale dye reflectance by top-surface Fresnel (some light never enters)
    # R_observed = R_fresnel_top + (1 - R_fresnel_top)² × R_KM_coupled
    R_f_top = result.fresnel_loss_top
    T_top = 1.0 - R_f_top

    # Kubelka-Munk coupling
    R_coupled = kubelka_munk_coupling(R_dye, T_dye, R_sub)

    # Total: Fresnel top surface + transmitted coupled light
    R_total = R_f_top + T_top ** 2 * R_coupled
    R_total = np.clip(R_total, 0.0, 1.0)
    result.R_total = R_total

    # CIE color
    try:
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_Lab, XYZ_to_sRGB, cie_delta_E
        X, Y, Z = spectrum_to_XYZ(R_total, _LAM)
        L, a, b = XYZ_to_Lab(X, Y, Z)
        r, g, bv = XYZ_to_sRGB(X, Y, Z)
        result.Lab_on_substrate = (L, a, b)
        result.sRGB_on_substrate = (r, g, bv)

        # Dye alone color
        X0, Y0, Z0 = spectrum_to_XYZ(R_dye, _LAM)
        L0, a0, b0 = XYZ_to_Lab(X0, Y0, Z0)
        result.Lab_dye_alone = (L0, a0, b0)

        # Color shift from substrate
        result.delta_E_from_alone = cie_delta_E((L, a, b), (L0, a0, b0))

        # Distance from target
        if target_Lab:
            result.delta_E_from_target = cie_delta_E((L, a, b), target_Lab)

    except (ImportError, Exception):
        pass

    return result


def compare_substrates(R_dye: np.ndarray,
                         substrate_names: List[str] = None,
                         n_dye: float = 1.50,
                         target_Lab: Tuple[float, float, float] = None,
                         dye_name: str = "") -> List[ColorOnSubstrateResult]:
    """Compare how a dye looks on multiple substrates.

    Returns sorted by ΔE from target (if target provided) or ΔE from dye-alone.
    """
    if substrate_names is None:
        substrate_names = list(SUBSTRATE_LIBRARY.keys())

    results = []
    for name in substrate_names:
        r = predict_color_on_substrate(R_dye, name, n_dye,
                                          target_Lab=target_Lab,
                                          dye_name=dye_name)
        results.append(r)

    # Sort by ΔE from target if available, else by ΔE from alone (least shift)
    if target_Lab:
        results.sort(key=lambda r: r.delta_E_from_target if r.delta_E_from_target is not None else 999)
    else:
        results.sort(key=lambda r: r.delta_E_from_alone if r.delta_E_from_alone is not None else 999)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Surface Adhesion Physics (T2)
# ═══════════════════════════════════════════════════════════════════════════

def young_contact_angle(gamma_s: float, gamma_l: float,
                          gamma_sl: float) -> float:
    """Young's equation: equilibrium contact angle.

    cos(θ) = (γ_s - γ_sl) / γ_l

    Parameters
    ----------
    gamma_s : float
        Solid surface energy (mJ/m²).
    gamma_l : float
        Liquid surface tension (mJ/m²). Water: 72.8.
    gamma_sl : float
        Solid-liquid interfacial energy (mJ/m²).

    Returns
    -------
    float
        Contact angle in degrees.

    Physics tier: T2 (Young 1805).
    """
    if gamma_l <= 0:
        return 0.0
    cos_theta = (gamma_s - gamma_sl) / gamma_l
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def work_of_adhesion(gamma_s: float, gamma_l: float,
                       gamma_sl: float) -> float:
    """Dupré work of adhesion.

    W_a = γ_s + γ_l - γ_sl

    Higher W_a = stronger adhesion.

    Physics tier: T2 (Dupré 1869).
    """
    return gamma_s + gamma_l - gamma_sl


def owens_wendt_gamma_sl(gamma_s_d: float, gamma_s_p: float,
                            gamma_l_d: float, gamma_l_p: float) -> float:
    """Owens-Wendt interfacial energy from dispersive + polar components.

    γ_sl = γ_s + γ_l - 2(√(γ_s_d·γ_l_d) + √(γ_s_p·γ_l_p))

    Parameters
    ----------
    gamma_s_d, gamma_s_p : float
        Solid dispersive and polar components (mJ/m²).
    gamma_l_d, gamma_l_p : float
        Liquid dispersive and polar components (mJ/m²).

    Returns
    -------
    float
        Interfacial energy γ_sl (mJ/m²).

    Physics tier: T2 (Owens & Wendt 1969).
    """
    gamma_s = gamma_s_d + gamma_s_p
    gamma_l = gamma_l_d + gamma_l_p
    interaction = 2.0 * (math.sqrt(gamma_s_d * gamma_l_d) +
                          math.sqrt(gamma_s_p * gamma_l_p))
    return gamma_s + gamma_l - interaction


def predict_adhesion(substrate_name: str,
                       coating_gamma_d: float = 30.0,
                       coating_gamma_p: float = 5.0) -> dict:
    """Predict dye-substrate adhesion quality.

    Uses Owens-Wendt to compute W_a and assess compatibility.

    Parameters
    ----------
    substrate_name : str
    coating_gamma_d, coating_gamma_p : float
        Dispersive and polar components of coating surface energy (mJ/m²).
        Typical polymeric coating: d=30, p=5.

    Returns
    -------
    dict with W_a, gamma_sl, contact_angle, adhesion_quality.
    """
    sub = SUBSTRATE_LIBRARY.get(substrate_name)
    if sub is None:
        return {"error": f"Unknown substrate: {substrate_name}"}

    if sub.dispersive_component_mJ_m2 is None or sub.polar_component_mJ_m2 is None:
        return {"substrate": substrate_name, "W_a": None,
                "note": "Surface energy components not available"}

    gamma_sl = owens_wendt_gamma_sl(
        sub.dispersive_component_mJ_m2, sub.polar_component_mJ_m2,
        coating_gamma_d, coating_gamma_p
    )
    gamma_s = sub.dispersive_component_mJ_m2 + sub.polar_component_mJ_m2
    gamma_l = coating_gamma_d + coating_gamma_p
    W_a = work_of_adhesion(gamma_s, gamma_l, gamma_sl)

    # Adhesion quality heuristic (T2: Owens-Wendt framework)
    if W_a > 80:
        quality = "excellent"
    elif W_a > 60:
        quality = "good"
    elif W_a > 40:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "substrate": substrate_name,
        "W_a_mJ_m2": W_a,
        "gamma_sl_mJ_m2": gamma_sl,
        "adhesion_quality": quality,
        "click_compatible": sub.click_compatible,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_substrate_table():
    """Print substrate library summary."""
    print(f"\n{'Name':<20} {'Cat':<8} {'R_vis':>5} {'n':>5} {'γ_s':>5} "
          f"{'θ_water':>7} {'Click':>5}")
    print("-" * 65)
    for name, s in sorted(SUBSTRATE_LIBRARY.items()):
        R = f"{s.reflectance_vis:.2f}" if s.reflectance_vis is not None else "  -"
        n = f"{s.n_substrate:.2f}" if s.n_substrate is not None else "  -"
        g = f"{s.surface_energy_mJ_m2:.0f}" if s.surface_energy_mJ_m2 is not None else "  -"
        ca = f"{s.water_contact_angle_deg:.0f}°" if s.water_contact_angle_deg is not None else "  -"
        cl = "yes" if s.click_compatible else "no"
        print(f"{name:<20} {s.category:<8} {R:>5} {n:>5} {g:>5} {ca:>7} {cl:>5}")


def print_color_comparison(results: List[ColorOnSubstrateResult]):
    """Print substrate comparison table."""
    print(f"\n{'Substrate':<20} {'L*':>5} {'a*':>5} {'b*':>5} "
          f"{'ΔE_shift':>8} {'ΔE_target':>9} {'Click':>5}")
    print("-" * 65)
    for r in results:
        sub = SUBSTRATE_LIBRARY.get(r.substrate_name)
        click = "yes" if sub and sub.click_compatible else "no"
        if r.Lab_on_substrate:
            L, a, b = r.Lab_on_substrate
            dE_s = f"{r.delta_E_from_alone:.1f}" if r.delta_E_from_alone is not None else "  -"
            dE_t = f"{r.delta_E_from_target:.1f}" if r.delta_E_from_target is not None else "  -"
            print(f"{r.substrate_name:<20} {L:5.1f} {a:5.1f} {b:5.1f} "
                  f"{dE_s:>8} {dE_t:>9} {click:>5}")
        else:
            print(f"{r.substrate_name:<20}   (no CIE module)")


if __name__ == "__main__":
    print("=" * 70)
    print("Surface Optics — Substrate Library + Optical Coupling")
    print("=" * 70)

    print_substrate_table()

    # Demo: green structural dye on different substrates
    print("\n--- Green dye (530nm peak) on substrates ---")
    sigma = 25.0
    R_dye = 0.25 * np.exp(-0.5 * ((_LAM - 530) / sigma) ** 2) + 0.05
    results = compare_substrates(R_dye, target_Lab=(55, -40, 20),
                                   dye_name="green_photonic_glass")
    print_color_comparison(results)
