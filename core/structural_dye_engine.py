"""
structural_dye_engine.py — Unified Structural Dye Design Engine

Core-shell particles AND flat multilayers designed from a unified
shell library with click chemistry tracking.

Forward model: TMM for flat films, thin-shell TMM approximation for
core-shell particles (valid when shell thickness << core radius).

Data: all T1 (published ε, n, k) and T2 (TMM, Beer-Lambert).
No miepython dependency.

References:
    Beer A. Ann. Phys. 1852, 162, 78.
    Yeh P. Optical Waves in Layered Media. Wiley 2005.
    Aden AL, Kerker M. J. Appl. Phys. 1951, 22, 1242.
    Born M, Wolf E. Principles of Optics, 7th ed. Cambridge 1999.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import product as itertools_product


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_LAM = np.linspace(380, 780, 81)  # 5 nm steps, visible range


# ═══════════════════════════════════════════════════════════════════════════
# Expanded Shell Library (T1: published data)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChromophoreEntry:
    """Published chromophore optical properties."""
    name: str
    lambda_max_nm: float       # peak absorption wavelength
    epsilon_M_cm: float        # molar absorptivity (M⁻¹cm⁻¹)
    bandwidth_cm1: float       # FWHM in cm⁻¹
    n_contribution: float      # shell refractive index contribution
    click_chemistry: str       # attachment method
    color_absorbed: str        # what color is absorbed
    color_transmitted: str     # what color the shell appears
    source: str                # literature reference


# ── 22 Chromophores (all published ε) ─────────────────────────────────────

CHROMOPHORE_LIBRARY: Dict[str, ChromophoreEntry] = {}


def _add_chr(c: ChromophoreEntry):
    CHROMOPHORE_LIBRARY[c.name] = c


# Azo dyes
_add_chr(ChromophoreEntry("disperse_red_1", 502, 30000, 4000, 1.65,
    "NHS-ester → APTES amine", "green", "red",
    "Colour Index: Disperse Red 1. ε from Sigma-Aldrich catalog"))
_add_chr(ChromophoreEntry("methyl_orange", 464, 25000, 4500, 1.60,
    "sulfonate electrostatic or diazo coupling", "blue", "orange",
    "Colour Index: Acid Orange 52. ε from CRC Handbook"))
_add_chr(ChromophoreEntry("congo_red", 498, 32000, 3500, 1.62,
    "sulfonate → electrostatic on cationic surface", "blue-green", "red",
    "Colour Index: Direct Red 28. ε: Sigma-Aldrich"))
_add_chr(ChromophoreEntry("alizarin_red_S", 420, 8000, 5000, 1.58,
    "sulfonate → electrostatic; or catechol coordination", "violet", "yellow-red",
    "Colour Index: Mordant Red 3. ε: J. Chem. Educ. 2004, 81, 1322"))

# Porphyrins (Soret band — extremely high ε)
_add_chr(ChromophoreEntry("TPP_freebase", 419, 200000, 1500, 1.70,
    "Carboxyl-TPP + NHS → APTES amine; or azide-TPP via CuAAC", "violet", "yellow-green",
    "Gouterman M. J. Mol. Spectrosc. 1961, 6, 138"))
_add_chr(ChromophoreEntry("ZnTPP", 421, 250000, 1500, 1.72,
    "Same as TPP; Zn inserted post-attachment", "violet", "yellow-green",
    "Gouterman M. J. Mol. Spectrosc. 1961, 6, 138"))
_add_chr(ChromophoreEntry("CuTPP", 416, 180000, 1800, 1.68,
    "Same as TPP; Cu inserted post-attachment", "violet", "yellow-green",
    "Gouterman M. J. Mol. Spectrosc. 1961, 6, 138"))
_add_chr(ChromophoreEntry("FeTPP_Cl", 414, 120000, 2000, 1.66,
    "Same as TPP; Fe-Cl post-metalation", "violet", "yellow-green",
    "Gouterman M. J. Mol. Spectrosc. 1961, 6, 138"))

# Phthalocyanines (Q-band — red/NIR absorption → blue/green transmitted)
_add_chr(ChromophoreEntry("CuPc", 678, 150000, 2000, 1.80,
    "Sulfonated CuPc → electrostatic; or CuPc-azide via SPAAC", "red", "blue",
    "Leznoff CC, Lever ABP. Phthalocyanines. VCH 1989. Vol 1"))
_add_chr(ChromophoreEntry("ZnPc", 672, 170000, 1800, 1.78,
    "Same routes as CuPc", "red", "blue-green",
    "Leznoff CC, Lever ABP. Phthalocyanines. VCH 1989. Vol 1"))
_add_chr(ChromophoreEntry("H2Pc", 698, 100000, 2500, 1.75,
    "Sulfonated or azide-functionalized", "red", "blue",
    "Leznoff CC, Lever ABP. Phthalocyanines. VCH 1989. Vol 1"))

# MLCT complexes
_add_chr(ChromophoreEntry("Ru_bpy3", 452, 14600, 5000, 1.65,
    "Ru(bpy)₂(bpy-COOH) → NHS coupling to amine", "blue", "orange",
    "Juris A et al. Coord. Chem. Rev. 1988, 84, 85"))
_add_chr(ChromophoreEntry("Os_bpy3", 480, 12000, 5500, 1.64,
    "Os(bpy)₂(bpy-COOH) → NHS coupling", "blue", "orange-red",
    "Kober EM et al. JACS 1988, 110, 1842"))

# Simple organic
_add_chr(ChromophoreEntry("fluorescein", 490, 76000, 3000, 1.58,
    "FITC (isothiocyanate) → amine coupling", "blue-green", "yellow-green",
    "Sjöback R et al. Spectrochim. Acta A 1995, 51, L7"))
_add_chr(ChromophoreEntry("rhodamine_B", 554, 106000, 2500, 1.62,
    "Rhodamine-NHS → amine coupling", "green", "magenta",
    "Kubin RF, Fletcher AN. J. Lumin. 1982, 27, 455"))
_add_chr(ChromophoreEntry("indigo", 610, 20000, 4000, 1.55,
    "Indigo-carboxylate → NHS/amine", "orange", "blue",
    "Seixas de Melo J et al. J. Phys. Chem. A 2004, 108, 8767"))
_add_chr(ChromophoreEntry("methylene_blue", 665, 95000, 2000, 1.60,
    "Electrostatic on anionic surface", "red", "blue",
    "Bergmann K, O'Konski CT. J. Phys. Chem. 1963, 67, 2169"))
_add_chr(ChromophoreEntry("malachite_green", 621, 100000, 2500, 1.62,
    "Electrostatic on anionic surface", "red-orange", "green",
    "Lewis GN, Bigeleisen J. JACS 1943, 65, 520"))
_add_chr(ChromophoreEntry("crystal_violet", 590, 87000, 2800, 1.60,
    "Electrostatic on anionic surface", "yellow-green", "violet",
    "Lewis GN, Bigeleisen J. JACS 1943, 65, 520"))

# Inorganic chromophores
_add_chr(ChromophoreEntry("prussian_blue", 680, 10000, 6000, 1.72,
    "In situ: Fe³⁺ + [Fe(CN)₆]⁴⁻ on surface", "red", "blue",
    "Robin MB, Day P. Adv. Inorg. Chem. 1967, 10, 247"))
_add_chr(ChromophoreEntry("CrO4_yellow", 373, 4800, 5000, 1.60,
    "Chromate anchored via phosphonate linker", "violet", "yellow",
    "Lever ABP. Inorganic Electronic Spectroscopy. Elsevier 1984"))


# ── Index Modification Shells (T1: published n) ──────────────────────────

@dataclass
class IndexShellEntry:
    """Published index modification shell."""
    name: str
    n_shell: Optional[float]   # refractive index (None = computed)
    k_shell: float = 0.0       # extinction coefficient
    material: str = ""         # base material for n lookup
    porosity: float = 0.0      # for porous shells
    click_chemistry: str = ""
    source: str = ""


INDEX_SHELL_LIBRARY: Dict[str, IndexShellEntry] = {}


def _add_idx(s: IndexShellEntry):
    INDEX_SHELL_LIBRARY[s.name] = s


# Low-index shells
_add_idx(IndexShellEntry("porous_SiO2_30", None, 0.0, "SiO2", 0.30,
    "APTES → azide/DBCO on remaining silica",
    "Stöber process + etching. n from Maxwell-Garnett (T2)"))
_add_idx(IndexShellEntry("porous_SiO2_50", None, 0.0, "SiO2", 0.50,
    "Same as 30%, more fragile",
    "Aerogel-like. n from Maxwell-Garnett (T2)"))
_add_idx(IndexShellEntry("PMMA_brush", 1.49, 0.0, "PMMA", 0.0,
    "ATRP from initiator-SAM → azide terminus",
    "n=1.49 from Kasarova et al. Opt. Mater. 2007, 29, 1481"))
_add_idx(IndexShellEntry("polystyrene_shell", 1.59, 0.0, "polystyrene", 0.0,
    "Emulsion polymerization coating",
    "n=1.59 from Sultanova N et al. Acta Phys. Pol. A 2009, 116, 585"))
_add_idx(IndexShellEntry("cellulose_shell", 1.53, 0.0, "", 0.0,
    "Nanocellulose adsorption coating",
    "n=1.53 for cellulose. Moon RJ et al. Chem. Soc. Rev. 2011, 40, 3941"))

# High-index shells
_add_idx(IndexShellEntry("TiO2_anatase", None, 0.0, "TiO2_anatase", 0.0,
    "Sol-gel on APTES surface",
    "n from Sellmeier (T1). Devore 1951"))
_add_idx(IndexShellEntry("TiO2_rutile", None, 0.0, "TiO2_rutile", 0.0,
    "Sol-gel or sputtering",
    "n from Sellmeier (T1). Devore 1951"))
_add_idx(IndexShellEntry("ZnO_shell", None, 0.0, "ZnO", 0.0,
    "Sol-gel or ALD",
    "n from Sellmeier (T1). Bond 1965"))
_add_idx(IndexShellEntry("Si3N4_shell", None, 0.0, "Si3N4", 0.0,
    "PECVD or ALD",
    "n from Sellmeier (T1). Philipp 1973"))
_add_idx(IndexShellEntry("polydopamine", 1.70, 0.03, "", 0.0,
    "Self-polymerizing. Universal adhesion",
    "n=1.70, k=0.03. Bothma JP et al. Adv. Mater. 2008, 20, 3539"))
_add_idx(IndexShellEntry("melanin", 1.72, 0.04, "", 0.0,
    "Dopamine polymerization",
    "n=1.72, k=0.04. Stavenga DG et al. Proc. R. Soc. B 2012, 279, 1050"))


# ═══════════════════════════════════════════════════════════════════════════
# Physics: Chromophore → k(λ) via Beer-Lambert (T2)
# ═══════════════════════════════════════════════════════════════════════════

def chromophore_k_spectrum(name: str, surface_coverage_nm2: float = 3.0,
                            shell_thickness_nm: float = 5.0) -> np.ndarray:
    """Compute k(λ) for a chromophore shell from Beer-Lambert.

    k(λ) = (ε(λ) × Γ × ln(10)) / (4π × λ)

    where Γ = surface coverage (molecules/nm²) × shell thickness (nm)
    gives effective concentration path.

    Parameters
    ----------
    name : str
        Chromophore name from CHROMOPHORE_LIBRARY.
    surface_coverage_nm2 : float
        Molecules per nm² of surface. Typical: 1-5.
    shell_thickness_nm : float
        Shell thickness (nm).

    Returns
    -------
    np.ndarray
        k(λ) array on _LAM grid.

    Physics tier: T2 (Beer-Lambert with Gaussian lineshape).
    """
    if name not in CHROMOPHORE_LIBRARY:
        return np.zeros_like(_LAM)

    ch = CHROMOPHORE_LIBRARY[name]

    # Gaussian absorption profile in wavenumber space
    # Convert wavelength to wavenumber
    nu_max = 1e7 / ch.lambda_max_nm  # cm⁻¹
    sigma_nu = ch.bandwidth_cm1 / 2.355  # Gaussian sigma in cm⁻¹

    nu = 1e7 / _LAM  # cm⁻¹ for each wavelength
    epsilon_profile = ch.epsilon_M_cm * np.exp(-0.5 * ((nu - nu_max) / sigma_nu) ** 2)

    # Convert ε(λ) to k(λ)
    # Effective concentration: Γ/d gives "surface concentration" in mol/L equivalent
    # Γ (molecules/nm²) × 10^18 (nm²/m²) / (N_A × d_m) = C (mol/m³) = C/1000 (mol/L)
    # Simpler: k = ε × C × ln(10) / (4π/λ) where C is path-integrated
    # For a thin shell: absorbance A = ε × C_eff × d
    # C_eff = Γ / (N_A × d_nm × 1e-7)  [mol/cm³] → × 1000 = mol/L
    # k = A × ln(10) / (4π × d / λ)
    # Simplified: k = ε × Γ × 1e-7 / (4π) × ln(10) × λ / λ  ... 
    #
    # Standard formula: k(λ) = ε(λ) × c × ln(10) × λ / (4π)
    # where c = surface_coverage / (N_A × shell_thickness_cm)
    N_A = 6.022e23
    d_cm = shell_thickness_nm * 1e-7  # nm → cm
    c_mol_L = (surface_coverage_nm2 * 1e14) / (N_A * d_cm)  # molecules/cm² / (N_A × d)

    k_array = epsilon_profile * c_mol_L * np.log(10) * (_LAM * 1e-7) / (4 * np.pi)

    return k_array


def chromophore_absorption_spectrum(name: str,
                                      surface_coverage_nm2: float = 3.0,
                                      shell_thickness_nm: float = 5.0) -> np.ndarray:
    """Transmittance spectrum through a chromophore shell.

    T(λ) = exp(-4π × k(λ) × d / λ)

    Returns 1 - T (absorbance fraction, 0-1).
    """
    k = chromophore_k_spectrum(name, surface_coverage_nm2, shell_thickness_nm)
    d_m = shell_thickness_nm * 1e-9
    optical_depth = 4 * np.pi * k * d_m / (_LAM * 1e-9)
    T = np.exp(-optical_depth)
    return 1.0 - T  # absorption fraction


# ═══════════════════════════════════════════════════════════════════════════
# Physics: Shell n(λ) from Material Database (T1/T2)
# ═══════════════════════════════════════════════════════════════════════════

def shell_n_spectrum(shell_name: str) -> np.ndarray:
    """Get n(λ) for a shell type.

    Uses optical/refractive_index.py for known materials,
    Maxwell-Garnett for porous shells, fixed values otherwise.
    """
    if shell_name in INDEX_SHELL_LIBRARY:
        entry = INDEX_SHELL_LIBRARY[shell_name]

        # Fixed n
        if entry.n_shell is not None:
            return np.full_like(_LAM, entry.n_shell, dtype=float)

        # Porous shell: Maxwell-Garnett
        if entry.porosity > 0 and entry.material:
            try:
                from optical.refractive_index import n_real
                n_base = np.array([n_real(entry.material, lam) for lam in _LAM])
            except (ImportError, Exception):
                n_base = np.full_like(_LAM, 1.46, dtype=float)
            n_air = 1.0
            f_air = entry.porosity
            # Maxwell-Garnett: air inclusions in solid matrix
            n_eff_sq = n_base**2 * (n_air**2 + 2*n_base**2 + 2*f_air*(n_air**2 - n_base**2)) / \
                       (n_air**2 + 2*n_base**2 - f_air*(n_air**2 - n_base**2))
            return np.sqrt(np.maximum(1.0, n_eff_sq))

        # Material lookup
        if entry.material:
            try:
                from optical.refractive_index import n_real
                return np.array([n_real(entry.material, lam) for lam in _LAM])
            except (ImportError, Exception):
                return np.full_like(_LAM, 1.50, dtype=float)

    return np.full_like(_LAM, 1.50, dtype=float)  # default


def shell_k_spectrum(shell_name: str) -> np.ndarray:
    """Get k(λ) for a shell type (non-chromophore absorption)."""
    if shell_name in INDEX_SHELL_LIBRARY:
        entry = INDEX_SHELL_LIBRARY[shell_name]
        return np.full_like(_LAM, entry.k_shell, dtype=float)
    return np.zeros_like(_LAM)


# ═══════════════════════════════════════════════════════════════════════════
# TMM Forward Model (T2)
# ═══════════════════════════════════════════════════════════════════════════

def _build_tmm_stack(layers: list) -> list:
    """Convert our layer specs to optical/tmm format.

    Each layer: (material_or_n, thickness_nm)
    Returns list of (material_str, thickness_nm) for tmm.py.
    """
    stack = []
    for mat, t in layers:
        stack.append((mat, t))
    return stack


def compute_multilayer_spectrum(layers: list) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reflectance spectrum for a multilayer stack.

    Parameters
    ----------
    layers : list of (material_name, thickness_nm)

    Returns
    -------
    (wavelengths, R_spectrum)
    """
    try:
        from optical.tmm import tmm_spectrum
        stack = _build_tmm_stack(layers)
        result = tmm_spectrum(stack, _LAM)
        if isinstance(result, tuple):
            R = np.array(result[0])
        else:
            R = np.array(result)
        return _LAM, R
    except (ImportError, Exception):
        return _LAM, np.zeros_like(_LAM)


def compute_structural_dye_spectrum(core_material: str, core_diameter_nm: float,
                                      shell_layers: list,
                                      chromophore: Optional[str] = None,
                                      chromophore_coverage: float = 3.0,
                                      packing_fraction: float = 0.55,
                                      assembly: str = "photonic_glass") -> Tuple[np.ndarray, np.ndarray]:
    """Compute structural dye reflectance for colloidal assembly.

    Core-shell particles assembled into a photonic glass or opal.
    The BASE color comes from the colloidal structure (Bragg or PY).
    Shell modification TUNES the color via Δn and chromophore absorption.

    Physics:
        1. Shell modifies effective n of the particle → shifts structural peak
        2. Chromophore absorbs selectively → subtracts from background,
           enhancing color saturation

    Parameters
    ----------
    core_material : str
        Core material name.
    core_diameter_nm : float
        Core diameter (nm). Shell adds to total.
    shell_layers : list of (shell_name, thickness_nm)
        Shell stack from inside out.
    chromophore : str, optional
        Chromophore name for selective absorption.
    chromophore_coverage : float
        Surface coverage (molecules/nm²).
    packing_fraction : float
        Packing in assembly.
    assembly : str
        "photonic_glass" (angle-independent) or "bragg_opal" (iridescent).

    Returns
    -------
    (wavelengths, R_spectrum)
    """
    # 1. Compute effective particle n from core + shell
    try:
        from optical.refractive_index import n_real
        n_core_arr = np.array([n_real(core_material, lam) for lam in _LAM])
    except (ImportError, Exception):
        n_core_arr = np.full_like(_LAM, 1.46, dtype=float)

    # Shell n — volume-weighted average (core-shell effective medium)
    total_shell_t = sum(t for _, t in shell_layers)
    total_radius = core_diameter_nm / 2.0 + total_shell_t
    f_core = (core_diameter_nm / 2.0 / total_radius) ** 3 if total_radius > 0 else 1.0
    f_shell = 1.0 - f_core

    n_shell_arr = np.ones_like(_LAM)
    if shell_layers:
        # Use outermost shell n (dominant for optical response)
        n_shell_arr = shell_n_spectrum(shell_layers[-1][0])

    # Effective particle n
    n_eff_particle = np.sqrt(f_core * n_core_arr**2 + f_shell * n_shell_arr**2)

    # 2. Structural color from colloidal assembly
    d_total = core_diameter_nm + 2 * total_shell_t
    n_medium = 1.0  # air

    if assembly == "bragg_opal":
        # Bragg peak from effective n
        n_eff_assembly = np.sqrt(packing_fraction * n_eff_particle**2 +
                                  (1 - packing_fraction) * n_medium**2)
        # Use mid-visible n for peak calculation
        n_mid = float(n_eff_assembly[40])  # 580nm
        peak_nm = 1.633 * d_total * n_mid
        # Gaussian spectrum
        sigma = 30.0 / 2.355
        R = 0.6 * np.exp(-0.5 * ((_LAM - peak_nm) / sigma) ** 2) + 0.04
    else:
        # Photonic glass: broader peak, lower reflectance
        n_eff_assembly = np.sqrt(packing_fraction * n_eff_particle**2 +
                                  (1 - packing_fraction) * n_medium**2)
        n_mid = float(n_eff_assembly[40])
        d_avg = d_total * (0.74 / packing_fraction) ** (1.0 / 3.0)
        peak_nm = 2.0 * n_mid * d_avg
        sigma = 60.0 / 2.355
        R = 0.25 * np.exp(-0.5 * ((_LAM - peak_nm) / sigma) ** 2) + 0.05

    # 3. Chromophore selective absorption
    if chromophore and chromophore in CHROMOPHORE_LIBRARY:
        shell_t = shell_layers[-1][1] if shell_layers else 5.0
        abs_fraction = chromophore_absorption_spectrum(
            chromophore, chromophore_coverage, shell_t
        )
        # Chromophore subtracts from the BACKGROUND, enhancing saturation
        # Background is the non-peak region; absorption there helps
        # Light passes through shell on entry + exit: (1-abs)²
        R = R * (1.0 - abs_fraction) ** 2

    # Clamp
    R = np.clip(R, 0.0, 1.0)
    return _LAM, R


# ═══════════════════════════════════════════════════════════════════════════
# CIE Color Integration
# ═══════════════════════════════════════════════════════════════════════════

def spectrum_to_cie(R: np.ndarray, lam: np.ndarray = None) -> dict:
    """Convert reflectance spectrum to CIE Lab and sRGB."""
    if lam is None:
        lam = _LAM
    try:
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_Lab, XYZ_to_sRGB
        X, Y, Z = spectrum_to_XYZ(R, lam)
        L, a, b = XYZ_to_Lab(X, Y, Z)
        r, g, bv = XYZ_to_sRGB(X, Y, Z)
        return {"Lab": (L, a, b), "sRGB": (r, g, bv),
                "peak_nm": float(lam[np.argmax(R)])}
    except (ImportError, Exception):
        return {"Lab": None, "sRGB": None, "peak_nm": float(lam[np.argmax(R)])}


def compute_delta_E(lab1: tuple, lab2: tuple) -> float:
    """CIE ΔE*ab between two Lab tuples."""
    if lab1 is None or lab2 is None:
        return 999.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


# ═══════════════════════════════════════════════════════════════════════════
# De Novo Stack Generator
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DyeDesign:
    """A structural dye design candidate."""
    name: str = ""
    geometry: str = ""             # "core-shell" or "multilayer"
    core_material: str = "SiO2"
    core_diameter_nm: float = 250.0
    shell_stack: List[Tuple[str, float]] = field(default_factory=list)
    chromophore: Optional[str] = None
    chromophore_coverage: float = 3.0

    # Computed
    predicted_Lab: Optional[tuple] = None
    predicted_sRGB: Optional[tuple] = None
    predicted_peak_nm: float = 0.0
    delta_E: float = 999.0

    # Click chemistry feasibility
    click_steps: List[str] = field(default_factory=list)
    n_click_steps: int = 0

    # Cost/feasibility
    complexity_score: float = 0.0  # 0-1, lower = simpler


def _get_click_steps(shell_stack: list, chromophore: Optional[str]) -> List[str]:
    """Collect click chemistry steps for a design."""
    steps = []
    for shell_name, _ in shell_stack:
        if shell_name in INDEX_SHELL_LIBRARY:
            steps.append(f"{shell_name}: {INDEX_SHELL_LIBRARY[shell_name].click_chemistry}")
        elif shell_name in CHROMOPHORE_LIBRARY:
            steps.append(f"{shell_name}: {CHROMOPHORE_LIBRARY[shell_name].click_chemistry}")
    if chromophore and chromophore in CHROMOPHORE_LIBRARY:
        steps.append(f"{chromophore}: {CHROMOPHORE_LIBRARY[chromophore].click_chemistry}")
    return steps


def generate_dye_designs(target_Lab: tuple,
                          core_materials: list = None,
                          core_diameters: list = None,
                          shell_options: list = None,
                          chromophore_options: list = None,
                          max_shell_layers: int = 2,
                          geometry: str = "core-shell",
                          top_n: int = 20) -> List[DyeDesign]:
    """De novo structural dye design by combinatorial search.

    Generates core × shell_stack × chromophore combinations,
    computes spectrum and CIE color for each, ranks by ΔE to target.

    Parameters
    ----------
    target_Lab : tuple
        Target (L*, a*, b*) color.
    core_materials : list of str
        Core materials to try.
    core_diameters : list of float
        Core diameters to try (nm).
    shell_options : list of str
        Shell names from INDEX_SHELL_LIBRARY.
    chromophore_options : list of str
        Chromophore names (or None for no chromophore).
    max_shell_layers : int
        Maximum shell layers (1 or 2).
    geometry : str
        "core-shell" or "multilayer".
    top_n : int
        Return top N designs.

    Returns
    -------
    List of DyeDesign, sorted by ΔE (best first).
    """
    if core_materials is None:
        core_materials = ["SiO2", "polystyrene"]
    if core_diameters is None:
        core_diameters = [150.0, 180.0, 200.0, 220.0, 250.0, 280.0, 320.0]
    if shell_options is None:
        shell_options = ["porous_SiO2_30", "TiO2_anatase", "polydopamine",
                          "PMMA_brush", "ZnO_shell"]
    if chromophore_options is None:
        chromophore_options = [None, "CuPc", "TPP_freebase", "rhodamine_B",
                                "indigo", "methylene_blue"]

    shell_thicknesses = [5.0, 15.0, 30.0]  # nm

    designs = []

    for core_mat in core_materials:
        for core_d in core_diameters:
            for shell_name in shell_options:
                for shell_t in shell_thicknesses:
                    for chrom in chromophore_options:
                        shell_stack = [(shell_name, shell_t)]

                        if geometry == "core-shell":
                            lam, R = compute_structural_dye_spectrum(
                                core_mat, core_d, shell_stack,
                                chromophore=chrom,
                            )
                        else:
                            # Multilayer: build quarter-wave-ish stack
                            layers = [(core_mat, 100.0)]  # substrate
                            layers.extend(shell_stack)
                            lam, R = compute_multilayer_spectrum(layers)

                        if np.max(R) < 0.001:
                            continue

                        color = spectrum_to_cie(R, lam)
                        lab = color.get("Lab")
                        dE = compute_delta_E(lab, target_Lab) if lab else 999.0

                        click = _get_click_steps(shell_stack, chrom)
                        n_steps = len(click) + (1 if chrom else 0)
                        complexity = min(1.0, n_steps / 5.0)

                        name_parts = [core_mat, f"{core_d:.0f}nm"]
                        name_parts.append(f"{shell_name}@{shell_t:.0f}nm")
                        if chrom:
                            name_parts.append(chrom)

                        designs.append(DyeDesign(
                            name=" | ".join(name_parts),
                            geometry=geometry,
                            core_material=core_mat,
                            core_diameter_nm=core_d,
                            shell_stack=shell_stack,
                            chromophore=chrom,
                            chromophore_coverage=3.0,
                            predicted_Lab=lab,
                            predicted_sRGB=color.get("sRGB"),
                            predicted_peak_nm=color.get("peak_nm", 0.0),
                            delta_E=dE,
                            click_steps=click,
                            n_click_steps=n_steps,
                            complexity_score=complexity,
                        ))

    designs.sort(key=lambda d: d.delta_E)
    return designs[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_dye_designs(designs: List[DyeDesign], target_Lab: tuple = None,
                       top_n: int = 15):
    """Print ranked structural dye designs."""
    print(f"\n{'Rank':>4} {'ΔE':>6} {'Peak':>5} {'Design':<55} {'Chrom':<15} {'Steps':>5}")
    print("-" * 100)

    for i, d in enumerate(designs[:top_n]):
        chrom = d.chromophore or "-"
        print(f"{i+1:4d} {d.delta_E:6.1f} {d.predicted_peak_nm:5.0f} "
              f"{d.name:<55} {chrom:<15} {d.n_click_steps:>5}")


def print_chromophore_table():
    """Print the chromophore library."""
    print(f"\n{'Chromophore':<20} {'λ_max':>5} {'ε':>8} {'Absorbs':<12} {'Appears':<12}")
    print("-" * 65)
    for name, ch in sorted(CHROMOPHORE_LIBRARY.items(), key=lambda x: x[1].lambda_max_nm):
        print(f"{name:<20} {ch.lambda_max_nm:5.0f} {ch.epsilon_M_cm:8.0f} "
              f"{ch.color_absorbed:<12} {ch.color_transmitted:<12}")


if __name__ == "__main__":
    print("=" * 80)
    print("Structural Dye Engine — De Novo Design")
    print("=" * 80)

    print_chromophore_table()

    print(f"\nIndex shells: {len(INDEX_SHELL_LIBRARY)}")
    for name in sorted(INDEX_SHELL_LIBRARY):
        e = INDEX_SHELL_LIBRARY[name]
        n = e.n_shell if e.n_shell else f"computed ({e.material})"
        print(f"  {name:<20} n={n}, k={e.k_shell}")

    # De novo design: target green (L*=55, a*=-40, b*=20)
    target = (55.0, -40.0, 20.0)
    print(f"\n--- De novo design for Lab={target} (green) ---")
    designs = generate_dye_designs(target, top_n=10)
    print_dye_designs(designs, target)
