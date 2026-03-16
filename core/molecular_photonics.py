"""
molecular_photonics.py — 6 Light-Redirection Mechanisms

L1. Fluorescence/Phosphorescence (Stokes shift, QY, lifetime)
L2. Rayleigh/Raman Scattering (elastic + inelastic)
L3. Optical Rotation / Circular Dichroism (chiral light manipulation)
L4. Photonic Bandgap Engineering (forbidden bands from periodicity)
L5. Plasmon Resonance (metal NP LSPR)
L6. Nonlinear Optics (SHG, Kerr, two-photon, optical limiting)

All equations T2 (established EM/quantum theory).
Published data T1 (fluorophore QY, Au/Ag dielectric, Raman frequencies).

References:
  Lakowicz JR. Principles of Fluorescence Spectroscopy, 3rd ed. Springer 2006.
  Bohren CF, Huffman DR. Absorption and Scattering of Light. Wiley 1983.
  Jackson JD. Classical Electrodynamics, 3rd ed. Wiley 1999.
  Boyd RW. Nonlinear Optics, 4th ed. Academic 2020.
  Barron LD. Molecular Light Scattering and Optical Activity. Cambridge 2004.
  Joannopoulos JD et al. Photonic Crystals. Princeton 2008.
  Maier SA. Plasmonics: Fundamentals and Applications. Springer 2007.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PI = math.pi
C_LIGHT = 2.998e8        # m/s
H_PLANCK = 6.626e-34     # J·s
HBAR = 1.055e-34          # J·s
K_BOLTZ = 1.381e-23       # J/K
E_CHARGE = 1.602e-19      # C
EPS_0 = 8.854e-12         # F/m
M_ELECTRON = 9.109e-31    # kg
N_AVO = 6.022e23
_LAM = np.linspace(380, 780, 81)


# ═══════════════════════════════════════════════════════════════════════════
# L1. Fluorescence / Phosphorescence
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FluorophoreEntry:
    """Published fluorophore properties."""
    name: str
    abs_max_nm: float          # absorption maximum
    em_max_nm: float           # emission maximum
    stokes_shift_nm: float     # em_max - abs_max
    quantum_yield: float       # Φ (0-1)
    lifetime_ns: float         # fluorescence lifetime
    epsilon_M_cm: float        # molar absorptivity at abs_max
    solvent: str               # measurement solvent
    source: str


FLUOROPHORE_LIBRARY: Dict[str, FluorophoreEntry] = {}


def _add_fl(f: FluorophoreEntry):
    FLUOROPHORE_LIBRARY[f.name] = f


# Published fluorophore data (T1)
# Source: Lakowicz 2006 Table 1.4; Molecular Probes Handbook
_add_fl(FluorophoreEntry("fluorescein", 490, 514, 24, 0.93, 4.0, 76000,
    "water pH 9", "Sjöback R et al. Spectrochim. Acta A 1995, 51, L7"))
_add_fl(FluorophoreEntry("rhodamine_6G", 530, 556, 26, 0.95, 4.1, 116000,
    "ethanol", "Kubin RF, Fletcher AN. J. Lumin. 1982, 27, 455"))
_add_fl(FluorophoreEntry("rhodamine_B", 554, 580, 26, 0.49, 1.7, 106000,
    "ethanol", "Kubin RF, Fletcher AN. J. Lumin. 1982, 27, 455"))
_add_fl(FluorophoreEntry("coumarin_6", 460, 505, 45, 0.78, 2.5, 54000,
    "ethanol", "Jones G et al. J. Phys. Chem. 1985, 89, 294"))
_add_fl(FluorophoreEntry("DAPI", 358, 461, 103, 0.04, 2.8, 27000,
    "water", "Kubista M et al. Biochemistry 1987, 26, 4545"))
_add_fl(FluorophoreEntry("Cy3", 550, 570, 20, 0.15, 0.3, 150000,
    "PBS", "Mujumdar RB et al. Bioconjugate Chem. 1993, 4, 105"))
_add_fl(FluorophoreEntry("Cy5", 649, 670, 21, 0.27, 1.0, 250000,
    "PBS", "Mujumdar RB et al. Bioconjugate Chem. 1993, 4, 105"))
_add_fl(FluorophoreEntry("FITC", 494, 521, 27, 0.93, 4.1, 73000,
    "water pH 9", "Lakowicz 2006 Table 1.4"))
_add_fl(FluorophoreEntry("TRITC", 544, 572, 28, 0.28, 2.0, 85000,
    "methanol", "Lakowicz 2006 Table 1.4"))
_add_fl(FluorophoreEntry("Ru_bpy3", 452, 620, 168, 0.042, 600.0, 14600,
    "water", "Juris A et al. Coord. Chem. Rev. 1988, 84, 85"))
_add_fl(FluorophoreEntry("Eu_DOTA", 394, 614, 220, 0.15, 1.1e6, 50,
    "water", "Bünzli JCG. Chem. Rev. 2010, 110, 2729"))
_add_fl(FluorophoreEntry("perylene", 436, 470, 34, 0.94, 4.3, 38500,
    "cyclohexane", "Berlman IB. Handbook of Fluorescence Spectra. Academic 1971"))


def stokes_shift_energy(abs_nm: float, em_nm: float) -> float:
    """Stokes shift in eV.

    ΔE = hc × (1/λ_abs - 1/λ_em)

    Physics tier: T2 (energy conservation).
    """
    if abs_nm <= 0 or em_nm <= 0:
        return 0.0
    return 1239.84 * (1.0/abs_nm - 1.0/em_nm)


def lippert_mataga_shift(delta_mu_D: float, a_A: float,
                           Delta_f: float) -> float:
    """Lippert-Mataga equation: predict Stokes shift from solvent polarity.

    Δν̃ = (2Δμ² / hca³) × Δf

    where Δμ = change in dipole moment (D), a = cavity radius (Å),
    Δf = orientation polarizability = f(ε) - f(n²).

    Parameters
    ----------
    delta_mu_D : float
        Change in dipole moment (Debye). Typical: 5-20 D.
    a_A : float
        Onsager cavity radius (Å). Typical: 3-6 Å.
    Delta_f : float
        Orientation polarizability. Δf = (ε-1)/(2ε+1) - (n²-1)/(2n²+1).
        Water: 0.32, ethanol: 0.29, hexane: 0.001.

    Returns
    -------
    float
        Stokes shift in cm⁻¹.

    Physics tier: T2 (Lippert 1955, Mataga 1956).
    """
    # Convert: D → C·m (1 D = 3.336e-30 C·m), Å → m
    delta_mu_Cm = delta_mu_D * 3.336e-30
    a_m = a_A * 1e-10
    h = H_PLANCK
    c = C_LIGHT

    if a_m <= 0:
        return 0.0

    # Δν̃ in m⁻¹, convert to cm⁻¹
    shift_m1 = 2.0 * delta_mu_Cm**2 * Delta_f / (h * c * a_m**3)
    return shift_m1 * 1e-2  # m⁻¹ → cm⁻¹


def strickler_berg_lifetime(epsilon_max: float, abs_fwhm_cm1: float,
                              n_solvent: float = 1.33,
                              em_max_nm: float = 500.0) -> float:
    """Strickler-Berg equation: radiative lifetime from absorption spectrum.

    1/τ_rad = 2.88e-9 × n² × ν̃_em² × ∫ε dν̃

    Simplified for Gaussian band: ∫ε dν̃ ≈ ε_max × Δν̃_FWHM × √(π/ln2)/2

    Parameters
    ----------
    epsilon_max : float
        Peak molar absorptivity (M⁻¹cm⁻¹).
    abs_fwhm_cm1 : float
        Absorption band FWHM (cm⁻¹).
    n_solvent : float
        Solvent refractive index.
    em_max_nm : float
        Emission maximum (nm) for ν̃_em.

    Returns
    -------
    float
        Radiative lifetime τ_rad in ns.

    Physics tier: T2 (Strickler & Berg, J. Chem. Phys. 1962, 37, 814).
    """
    if epsilon_max <= 0 or abs_fwhm_cm1 <= 0:
        return 999.0

    nu_em_cm1 = 1e7 / em_max_nm  # cm⁻¹
    integral_epsilon = epsilon_max * abs_fwhm_cm1 * math.sqrt(PI / math.log(2)) / 2.0
    k_rad = 2.88e-9 * n_solvent**2 * nu_em_cm1**2 * integral_epsilon

    if k_rad <= 0:
        return 999.0
    return 1e9 / k_rad  # s → ns


def emission_spectrum(abs_max_nm: float, stokes_shift_nm: float,
                       fwhm_nm: float = 40.0,
                       quantum_yield: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian emission spectrum.

    Parameters
    ----------
    abs_max_nm : float
        Absorption peak (nm).
    stokes_shift_nm : float
        Stokes shift (nm).
    fwhm_nm : float
        Emission FWHM (nm).
    quantum_yield : float
        Fraction of absorbed photons re-emitted.

    Returns
    -------
    (wavelengths, emission_intensity)
    """
    em_max = abs_max_nm + stokes_shift_nm
    sigma = fwhm_nm / 2.355
    emission = quantum_yield * np.exp(-0.5 * ((_LAM - em_max) / sigma) ** 2)
    return _LAM, emission


# ═══════════════════════════════════════════════════════════════════════════
# L2. Rayleigh / Raman Scattering
# ═══════════════════════════════════════════════════════════════════════════

def rayleigh_cross_section(diameter_nm: float, n_particle: float,
                             n_medium: float, wavelength_nm: float) -> float:
    """Rayleigh scattering cross-section for a small sphere.

    σ_sca = (2π⁵/3) × (d⁶/λ⁴) × ((m²-1)/(m²+2))²

    where m = n_particle/n_medium. Valid when d << λ (Rayleigh regime).

    Parameters
    ----------
    diameter_nm : float
    n_particle, n_medium : float
    wavelength_nm : float

    Returns
    -------
    float
        Scattering cross-section in nm².

    Physics tier: T2 (Rayleigh 1871, exact in small-particle limit).
    """
    d = diameter_nm
    lam = wavelength_nm
    m = n_particle / n_medium
    m2 = m ** 2

    prefactor = (2.0 * PI**5 / 3.0)
    sigma = prefactor * d**6 / lam**4 * ((m2 - 1) / (m2 + 2)) ** 2
    return sigma


def rayleigh_wavelength_dependence(wavelength_nm: float,
                                     ref_wavelength: float = 550.0) -> float:
    """Relative Rayleigh scattering intensity: I ∝ λ⁻⁴.

    Parameters
    ----------
    wavelength_nm : float
    ref_wavelength : float

    Returns
    -------
    float
        Relative intensity (1.0 at reference wavelength).

    Physics tier: T2.
    """
    return (ref_wavelength / wavelength_nm) ** 4


# Published Raman shifts (cm⁻¹) — T1
# Source: Schrader B. Infrared and Raman Spectroscopy. VCH 1995.
RAMAN_SHIFTS_CM1: Dict[str, float] = {
    "C-H_stretch": 2900,
    "C=C_stretch": 1600,
    "C=O_stretch": 1700,
    "C-C_stretch": 1060,
    "O-H_stretch": 3400,
    "N-H_stretch": 3350,
    "S-H_stretch": 2570,
    "C-N_stretch": 1090,
    "C≡N_stretch": 2220,
    "C≡C_stretch": 2100,
    "aromatic_ring_breathing": 1000,
    "Si-O_stretch": 1050,
    "P=O_stretch": 1250,
    "S=O_stretch": 1050,
    "water_bend": 1640,
}


def raman_shifted_wavelength(excitation_nm: float,
                               shift_cm1: float) -> float:
    """Compute Raman-shifted wavelength.

    ν̃_scattered = ν̃_excitation - Δν̃_Raman (Stokes)

    Parameters
    ----------
    excitation_nm : float
        Excitation laser wavelength (nm).
    shift_cm1 : float
        Raman shift (cm⁻¹, positive for Stokes).

    Returns
    -------
    float
        Scattered wavelength (nm).

    Physics tier: T2 (Raman 1928).
    """
    nu_exc = 1e7 / excitation_nm  # cm⁻¹
    nu_scat = nu_exc - shift_cm1
    if nu_scat <= 0:
        return 0.0
    return 1e7 / nu_scat


def raman_cross_section_relative(shift_cm1: float,
                                    wavelength_nm: float = 532.0) -> float:
    """Relative Raman cross-section (arbitrary units).

    σ_Raman ∝ ν_exc⁴ × |α'|² (α' = polarizability derivative).
    Since |α'| varies by bond, this returns the ν⁴ factor only.
    Multiply by bond-specific Raman activity for absolute value.

    Physics tier: T2 (Placzek 1934 polarizability theory).
    """
    nu = 1e7 / wavelength_nm  # cm⁻¹
    return (nu / 1e4) ** 4  # normalized to ~1 for green excitation


# ═══════════════════════════════════════════════════════════════════════════
# L3. Optical Rotation / Circular Dichroism
# ═══════════════════════════════════════════════════════════════════════════

def drude_specific_rotation(rotational_strengths: List[Tuple[float, float]],
                              wavelength_nm: float) -> float:
    """Drude equation: optical rotation from electronic transitions.

    [α] = Σ_i  A_i / (λ² - λ_i²)

    where A_i = rotational strength of transition i, λ_i = transition wavelength.

    Parameters
    ----------
    rotational_strengths : list of (λ_i nm, R_i deg·cm²/dmol)
        Each electronic transition contributing to rotation.
    wavelength_nm : float
        Measurement wavelength.

    Returns
    -------
    float
        Specific rotation [α] in degrees.

    Physics tier: T2 (Drude 1900; Condon 1937 for quantum form).
    """
    alpha = 0.0
    lam2 = wavelength_nm ** 2
    for lam_i, R_i in rotational_strengths:
        denom = lam2 - lam_i ** 2
        if abs(denom) < 1e-6:
            continue  # skip resonance
        alpha += R_i / denom
    return alpha


def cd_spectrum_gaussian(transitions: List[Tuple[float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate CD spectrum from Gaussian bands.

    Each transition: (λ_max nm, Δε_max M⁻¹cm⁻¹, bandwidth nm).
    Δε = ε_L - ε_R (differential absorption of left/right circular polarization).

    Returns (wavelengths, Δε(λ)).

    Physics tier: T2 (Rosenfeld 1929, Moffitt-Yang 1956).
    """
    cd = np.zeros_like(_LAM, dtype=float)
    for lam_max, de_max, bw in transitions:
        sigma = bw / 2.355
        cd += de_max * np.exp(-0.5 * ((_LAM - lam_max) / sigma) ** 2)
    return _LAM, cd


# Published specific rotations (T1)
# Source: CRC Handbook; Eliel & Wilen, Stereochemistry of Organic Compounds
SPECIFIC_ROTATIONS: Dict[str, float] = {
    "D-glucose": 52.7,        # [α]_D²⁰, water
    "L-glucose": -52.7,
    "sucrose": 66.5,          # water
    "D-fructose": -92.0,      # water
    "cholesterol": -31.5,     # chloroform
    "L-alanine": 1.8,         # water
    "D-tartaric_acid": 12.0,  # water
    "quinine": -117.0,        # ethanol
    "(R)-limonene": 98.0,     # neat
    "(S)-limonene": -98.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# L4. Photonic Bandgap Engineering
# ═══════════════════════════════════════════════════════════════════════════

def photonic_bandgap_1D(n_high: float, n_low: float,
                          d_high_nm: float, d_low_nm: float) -> dict:
    """1D photonic bandgap from a periodic dielectric stack.

    Center wavelength: λ₀ = 2(n_H·d_H + n_L·d_L)  (quarter-wave condition)
    Gap width: Δλ/λ₀ = (4/π)·arcsin((n_H - n_L)/(n_H + n_L))

    Parameters
    ----------
    n_high, n_low : float
        Refractive indices of alternating layers.
    d_high_nm, d_low_nm : float
        Layer thicknesses (nm).

    Returns
    -------
    dict with center_nm, gap_width_nm, gap_ratio, n_contrast.

    Physics tier: T2 (Yeh 2005; Joannopoulos 2008, exact for 1D).
    """
    lam_center = 2.0 * (n_high * d_high_nm + n_low * d_low_nm)
    n_contrast = (n_high - n_low) / (n_high + n_low)

    if abs(n_contrast) > 0:
        gap_ratio = (4.0 / PI) * math.asin(abs(n_contrast))
    else:
        gap_ratio = 0.0

    gap_width = lam_center * gap_ratio

    return {
        "center_nm": lam_center,
        "gap_width_nm": gap_width,
        "gap_ratio": gap_ratio,  # Δλ/λ₀
        "n_contrast": n_contrast,
        "n_high": n_high,
        "n_low": n_low,
    }


def photonic_dos_1D(wavelength_nm: float, center_nm: float,
                      gap_width_nm: float) -> float:
    """Photonic density of states near a 1D bandgap.

    Inside gap: DOS = 0 (forbidden).
    At band edge: DOS diverges (Van Hove singularity).
    Far from gap: DOS → 1 (bulk-like).

    Simplified model: DOS = 0 inside gap, enhanced at edges.

    Physics tier: T2 (Joannopoulos 2008).
    """
    half_gap = gap_width_nm / 2.0
    dist_from_center = abs(wavelength_nm - center_nm)

    if dist_from_center < half_gap:
        return 0.0  # inside bandgap
    elif dist_from_center < half_gap * 1.2:
        # Band edge enhancement (Van Hove)
        return 3.0
    else:
        return 1.0  # bulk-like


def quarter_wave_design(target_nm: float, n_high: float,
                          n_low: float) -> Tuple[float, float]:
    """Quarter-wave layer thicknesses for target center wavelength.

    d = λ₀ / (4n)

    Returns (d_high_nm, d_low_nm).
    Physics tier: T2 (exact).
    """
    return target_nm / (4.0 * n_high), target_nm / (4.0 * n_low)


def bandgap_reflectance_spectrum(n_high: float, n_low: float,
                                   n_bilayers: int,
                                   target_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate bandgap reflectance spectrum.

    Peak reflectance: R_max = ((n_H/n_L)^(2N) - 1)² / ((n_H/n_L)^(2N) + 1)²

    Gaussian-like stopband centered at target_nm.

    Physics tier: T2.
    """
    ratio = n_high / n_low
    R_max = ((ratio ** (2 * n_bilayers) - 1) / (ratio ** (2 * n_bilayers) + 1)) ** 2

    gap_info = photonic_bandgap_1D(n_high, n_low,
                                     target_nm / (4*n_high),
                                     target_nm / (4*n_low))
    sigma = gap_info["gap_width_nm"] / 2.355

    R = R_max * np.exp(-0.5 * ((_LAM - target_nm) / max(sigma, 1.0)) ** 2)
    return _LAM, R


# ═══════════════════════════════════════════════════════════════════════════
# L5. Plasmon Resonance
# ═══════════════════════════════════════════════════════════════════════════

def drude_dielectric(wavelength_nm: float, omega_p_eV: float,
                       gamma_eV: float,
                       eps_inf: float = 1.0) -> complex:
    """Drude model for metal dielectric function.

    ε(ω) = ε_∞ - ω_p² / (ω² + iγω)

    Parameters
    ----------
    wavelength_nm : float
    omega_p_eV : float
        Plasma frequency (eV). Au: 9.0, Ag: 9.2.
    gamma_eV : float
        Damping rate (eV). Au: 0.07, Ag: 0.02.
    eps_inf : float
        High-frequency dielectric constant. Au: 9.5, Ag: 3.7.

    Returns
    -------
    complex
        ε(ω) = ε₁ + iε₂.

    Physics tier: T2 (Drude 1900).
    """
    E_eV = 1239.84 / wavelength_nm
    omega = E_eV
    eps = eps_inf - omega_p_eV**2 / (omega**2 + 1j * gamma_eV * omega)
    return eps


# Published Drude parameters (T1)
# Source: Johnson PB, Christy RW. Phys. Rev. B 1972, 6, 4370.
METAL_DRUDE_PARAMS = {
    # (ω_p eV, γ eV, ε_∞)
    "Au": (9.0, 0.07, 9.5),
    "Ag": (9.2, 0.02, 3.7),
    "Cu": (8.8, 0.09, 10.0),
    "Al": (15.0, 0.60, 1.0),
}


def lspr_wavelength(n_medium: float, metal: str = "Au") -> float:
    """Localized surface plasmon resonance wavelength (Fröhlich condition).

    LSPR occurs when Re(ε_metal) = -2ε_medium = -2n²_medium.

    Solved from Drude model: ω_LSPR = ω_p / √(ε_∞ + 2n²)

    Parameters
    ----------
    n_medium : float
        Surrounding medium refractive index.
    metal : str
        Metal type (Au, Ag, Cu, Al).

    Returns
    -------
    float
        LSPR wavelength in nm.

    Physics tier: T2 (Fröhlich 1949; Mie 1908 for spheres).
    """
    if metal not in METAL_DRUDE_PARAMS:
        return 0.0
    omega_p, gamma, eps_inf = METAL_DRUDE_PARAMS[metal]

    eps_m = n_medium ** 2
    denom = eps_inf + 2 * eps_m
    if denom <= 0:
        return 0.0
    omega_lspr = omega_p / math.sqrt(denom)
    return 1239.84 / omega_lspr  # eV → nm


def lspr_shift_per_RIU(metal: str = "Au") -> float:
    """LSPR sensitivity: wavelength shift per refractive index unit.

    dλ/dn ≈ 2n × λ² × ω_p / (1239.84 × (ε_∞ + 2n²)^(3/2))

    Evaluated at n=1.33 (water). Typical: Au ~50-100 nm/RIU for spheres.

    Physics tier: T2 (differentiation of Fröhlich condition).
    """
    if metal not in METAL_DRUDE_PARAMS:
        return 0.0
    n = 1.33
    omega_p, gamma, eps_inf = METAL_DRUDE_PARAMS[metal]
    lam1 = lspr_wavelength(n, metal)
    lam2 = lspr_wavelength(n + 0.01, metal)
    return (lam2 - lam1) / 0.01  # nm/RIU


def plasmon_absorption_spectrum(metal: str, diameter_nm: float,
                                  n_medium: float = 1.33) -> Tuple[np.ndarray, np.ndarray]:
    """Absorption spectrum for a metal nanosphere (quasi-static limit).

    C_abs ∝ Im(ε - ε_m) / |ε + 2ε_m|²

    Valid for d << λ (quasi-static, < ~50nm for Au).

    Physics tier: T2 (Mie 1908, quasi-static limit).
    """
    if metal not in METAL_DRUDE_PARAMS:
        return _LAM, np.zeros_like(_LAM)

    omega_p, gamma, eps_inf = METAL_DRUDE_PARAMS[metal]
    eps_m = n_medium ** 2
    volume = (PI / 6.0) * (diameter_nm * 1e-9) ** 3

    C_abs = np.zeros_like(_LAM, dtype=float)
    for i, lam in enumerate(_LAM):
        eps = drude_dielectric(float(lam), omega_p, gamma, eps_inf)
        num = eps.imag
        denom = abs(eps + 2 * eps_m) ** 2
        if denom > 0:
            C_abs[i] = num / denom

    # Normalize to peak = 1
    if np.max(C_abs) > 0:
        C_abs /= np.max(C_abs)

    return _LAM, C_abs


# ═══════════════════════════════════════════════════════════════════════════
# L6. Nonlinear Optics
# ═══════════════════════════════════════════════════════════════════════════

def shg_wavelength(fundamental_nm: float) -> float:
    """Second harmonic generation: output at half the input wavelength.

    λ_SHG = λ_fundamental / 2

    Physics tier: T2 (exact, energy conservation).
    """
    return fundamental_nm / 2.0


def millers_rule_chi2(n_omega: float, n_2omega: float,
                        chi1_omega: float = None) -> float:
    """Miller's rule: estimate χ⁽²⁾ from linear susceptibility.

    χ⁽²⁾(2ω) ≈ Δ × χ⁽¹⁾(ω)² × χ⁽¹⁾(2ω)

    where χ⁽¹⁾ = n² - 1, and Δ ≈ constant (Miller's delta).

    This predicts that high-n materials tend to have high χ⁽²⁾.

    Parameters
    ----------
    n_omega : float
        Refractive index at fundamental frequency.
    n_2omega : float
        Refractive index at second harmonic.

    Returns
    -------
    float
        Relative χ⁽²⁾ (arbitrary units, for ranking).

    Physics tier: T2 (Miller 1964; Boyd Ch. 1).
    """
    chi1_w = n_omega ** 2 - 1.0
    chi1_2w = n_2omega ** 2 - 1.0
    return chi1_w ** 2 * chi1_2w


# Published χ⁽²⁾ values (pm/V) — T1
# Source: Boyd Table 1.5.3; Dmitriev VG et al. Handbook of NLO Crystals
CHI2_VALUES_PM_V: Dict[str, float] = {
    "KDP": 0.39,           # KH₂PO₄
    "BBO": 2.2,            # β-BaB₂O₄
    "LiNbO3": 25.0,        # lithium niobate (d₃₃)
    "KTP": 16.9,           # KTiOPO₄
    "LiTaO3": 13.8,
    "GaAs": 170.0,         # gallium arsenide
    "ZnSe": 78.0,
    "AgGaS2": 33.0,
    "quartz": 0.30,        # α-SiO₂
}


def kerr_coefficient(n: float, n2_cm2_W: float) -> float:
    """Kerr effect: intensity-dependent refractive index.

    n(I) = n₀ + n₂ × I

    Parameters
    ----------
    n : float
        Linear refractive index.
    n2_cm2_W : float
        Nonlinear refractive index (cm²/W).
        Typical: 3e-16 for glass, 1e-13 for CS₂.

    Returns
    -------
    float
        n₂ in m²/W (SI).

    Physics tier: T2 (Kerr 1875; Boyd Ch. 4).
    """
    return n2_cm2_W * 1e-4  # cm²/W → m²/W


# Published n₂ values (cm²/W) — T1
# Source: Boyd Table 4.1.2; Weber MJ. Handbook of Optical Materials
KERR_N2_CM2_W: Dict[str, float] = {
    "fused_silica": 2.5e-16,
    "BK7_glass": 3.2e-16,
    "sapphire": 3.0e-16,
    "CS2": 3.2e-14,
    "water": 4.1e-16,
    "polystyrene": 1.2e-14,
    "ZnSe": 1.7e-13,
    "GaAs": 1.5e-13,
    "Si": 4.5e-14,
}


def two_photon_absorption_wavelength(bandgap_eV: float) -> float:
    """Wavelength threshold for two-photon absorption.

    TPA occurs when 2×photon_energy > bandgap:
    λ_threshold = 2 × hc / E_gap

    Parameters
    ----------
    bandgap_eV : float
        Material bandgap (eV).

    Returns
    -------
    float
        Maximum wavelength for TPA (nm).

    Physics tier: T2 (energy conservation).
    """
    if bandgap_eV <= 0:
        return 0.0
    return 2.0 * 1239.84 / bandgap_eV


def optical_limiting_threshold(n2_cm2_W: float, I_sat_W_cm2: float = 1e9) -> float:
    """Optical limiting: intensity at which nonlinear absorption dominates.

    I_lim ≈ λ / (2π × n₂ × L_eff)

    Simplified: for a given n₂, higher n₂ → lower threshold.

    Returns intensity threshold in W/cm².

    Physics tier: T2 (Boyd Ch. 4).
    """
    if n2_cm2_W <= 0:
        return float('inf')
    return 1.0 / (abs(n2_cm2_W) * 1e6)  # rough scaling


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_fluorophore_table():
    """Print fluorophore library."""
    print(f"\n{'Name':<16} {'Abs':>5} {'Em':>5} {'ΔS':>4} {'QY':>5} {'τ(ns)':>8} {'ε':>8}")
    print("-" * 65)
    for name, f in sorted(FLUOROPHORE_LIBRARY.items(), key=lambda x: x[1].abs_max_nm):
        print(f"{name:<16} {f.abs_max_nm:5.0f} {f.em_max_nm:5.0f} "
              f"{f.stokes_shift_nm:4.0f} {f.quantum_yield:5.2f} "
              f"{f.lifetime_ns:8.1f} {f.epsilon_M_cm:8.0f}")


def print_plasmon_table():
    """Print LSPR for metals in different media."""
    print(f"\n{'Metal':<6}", end="")
    for n in [1.0, 1.33, 1.5]:
        print(f"  n={n:.2f}", end="")
    print(f"  {'Sensitivity':>12}")
    print("-" * 55)
    for metal in ["Au", "Ag", "Cu", "Al"]:
        print(f"{metal:<6}", end="")
        for n in [1.0, 1.33, 1.5]:
            lam = lspr_wavelength(n, metal)
            print(f"  {lam:6.0f}", end="")
        sens = lspr_shift_per_RIU(metal)
        print(f"  {sens:8.1f} nm/RIU")


if __name__ == "__main__":
    print("=" * 70)
    print("Molecular Photonics — 6 Light-Redirection Mechanisms")
    print("=" * 70)

    print("\n--- L1: Fluorescence ---")
    print_fluorophore_table()

    print("\n--- L5: Plasmon Resonance ---")
    print_plasmon_table()

    print("\n--- L4: Photonic Bandgap ---")
    gap = photonic_bandgap_1D(2.5, 1.46, 50.0, 90.0)
    print(f"  TiO₂/SiO₂ stack: center={gap['center_nm']:.0f}nm, "
          f"gap={gap['gap_width_nm']:.0f}nm, Δλ/λ={gap['gap_ratio']:.3f}")

    print("\n--- L6: Nonlinear Optics ---")
    for mat, chi2 in sorted(CHI2_VALUES_PM_V.items(), key=lambda x: -x[1]):
        print(f"  {mat:<12} χ⁽²⁾ = {chi2:.1f} pm/V")
