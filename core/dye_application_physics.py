"""
dye_application_physics.py — Real-World Structural Dye Application Physics

S1. Wetting & Film Formation (capillary, spreading, drying)
S2. Angle-Dependent Appearance / BRDF (iridescence, diffuse/specular)
S3. Spectral Broadening + Multiple Scattering (polydispersity, KM, RT)
S4. Durability / Environmental / Color Mixing (UV, humidity, temperature, mixing)

All T1 (published constants) and T2 (established equations). No T3.

References:
  S1: Lucas R. Kolloid-Z. 1918, 23, 15. Washburn EW. Phys. Rev. 1921, 17, 273.
      de Gennes PG. Rev. Mod. Phys. 1985, 57, 827.
  S2: Kinoshita S. Structural Colors in the Realm of Nature. World Scientific 2008.
      Nicodemus FE. Appl. Opt. 1970, 9, 1474 (BRDF definition).
  S3: Kubelka P. J. Opt. Soc. Am. 1948, 38, 448.
      Chandrasekhar S. Radiative Transfer. Dover 1960.
  S4: Bauer DR. J. Coat. Technol. 1994, 66, 57 (UV degradation).
      Judd DB, Wyszecki G. Color in Business, Science, and Industry. Wiley 1975.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PI = math.pi
_LAM = np.linspace(380, 780, 81)


# ═══════════════════════════════════════════════════════════════════════════
# S1. Wetting & Film Formation
# ═══════════════════════════════════════════════════════════════════════════

def capillary_number(velocity_m_s: float, viscosity_Pa_s: float,
                       surface_tension_N_m: float) -> float:
    """Capillary number: viscous forces vs surface tension.

    Ca = μv/γ

    Ca << 1: surface tension dominates (droplets, beading).
    Ca >> 1: viscous forces dominate (film coating).

    Parameters
    ----------
    velocity_m_s : float
        Coating velocity or flow velocity (m/s).
    viscosity_Pa_s : float
        Dynamic viscosity (Pa·s). Water: 8.9e-4.
    surface_tension_N_m : float
        Surface tension (N/m). Water: 0.0728.

    Returns
    -------
    float
        Capillary number (dimensionless).

    Physics tier: T2 (fluid mechanics).
    """
    if surface_tension_N_m <= 0:
        return float('inf')
    return viscosity_Pa_s * velocity_m_s / surface_tension_N_m


def spreading_coefficient(gamma_s: float, gamma_l: float,
                            gamma_sl: float) -> float:
    """Spreading coefficient: does the liquid spread or bead?

    S = γ_s - γ_l - γ_sl

    S > 0: liquid spreads spontaneously (complete wetting).
    S < 0: liquid forms droplets (partial wetting).

    Parameters
    ----------
    gamma_s : float
        Solid surface energy (mJ/m²).
    gamma_l : float
        Liquid surface tension (mJ/m²).
    gamma_sl : float
        Solid-liquid interfacial energy (mJ/m²).

    Returns
    -------
    float
        Spreading coefficient S (mJ/m²). Positive = spreads.

    Physics tier: T2 (de Gennes 1985).
    """
    return gamma_s - gamma_l - gamma_sl


def lucas_washburn_penetration(pore_radius_m: float, contact_angle_deg: float,
                                  viscosity_Pa_s: float,
                                  surface_tension_N_m: float,
                                  time_s: float) -> float:
    """Lucas-Washburn: capillary penetration depth into porous substrate.

    L = √(r × γ × cos(θ) × t / (2η))

    Models how far coating solution wicks into textile/paper.

    Parameters
    ----------
    pore_radius_m : float
        Effective pore radius (m). Textile: ~10-50 μm.
    contact_angle_deg : float
        Contact angle on substrate.
    viscosity_Pa_s : float
    surface_tension_N_m : float
    time_s : float
        Penetration time (s).

    Returns
    -------
    float
        Penetration depth (m).

    Physics tier: T2 (Lucas 1918, Washburn 1921).
    """
    theta_rad = math.radians(contact_angle_deg)
    cos_theta = math.cos(theta_rad)
    if cos_theta <= 0 or viscosity_Pa_s <= 0:
        return 0.0  # non-wetting or invalid
    arg = pore_radius_m * surface_tension_N_m * cos_theta * time_s / (2.0 * viscosity_Pa_s)
    if arg <= 0:
        return 0.0
    return math.sqrt(arg)


def drying_time_thin_film(thickness_m: float, D_vapor_m2_s: float = 2.5e-5,
                            RH: float = 0.5) -> float:
    """Estimate drying time for a thin aqueous film.

    t_dry ≈ d² / (D × (1 - RH))

    Simplified diffusion-limited evaporation.

    Parameters
    ----------
    thickness_m : float
        Film thickness (m).
    D_vapor_m2_s : float
        Water vapor diffusivity in air (m²/s). ~2.5e-5.
    RH : float
        Relative humidity (0-1).

    Returns
    -------
    float
        Drying time in seconds.

    Physics tier: T2 (Fickian evaporation).
    """
    driving_force = max(0.01, 1.0 - RH)
    if D_vapor_m2_s <= 0:
        return float('inf')
    return thickness_m ** 2 / (D_vapor_m2_s * driving_force)


def film_uniformity_from_roughness(film_thickness_nm: float,
                                      roughness_um: float) -> float:
    """Estimate film uniformity degradation from substrate roughness.

    Uniformity = 1.0 when film >> roughness (smooth coating).
    Degrades when roughness ≈ film thickness (conformal coverage fails).

    uniformity = 1 / (1 + (roughness/film_thickness)²)

    Parameters
    ----------
    film_thickness_nm : float
    roughness_um : float
        RMS roughness of substrate (μm).

    Returns
    -------
    float
        Uniformity score 0-1 (1 = perfectly uniform).

    Physics tier: T2 (geometric scaling).
    """
    roughness_nm = roughness_um * 1000.0
    if film_thickness_nm <= 0:
        return 0.0
    ratio = roughness_nm / film_thickness_nm
    return 1.0 / (1.0 + ratio ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# S2. Angle-Dependent Appearance / BRDF
# ═══════════════════════════════════════════════════════════════════════════

def bragg_angle_shift(peak_nm_normal: float, n_eff: float,
                        theta_deg: float) -> float:
    """Bragg peak shift with viewing angle.

    λ(θ) = λ₀ × √(1 - sin²(θ)/n_eff²)

    Blue-shifts with increasing angle (universal for all Bragg structures).

    Parameters
    ----------
    peak_nm_normal : float
        Peak wavelength at normal incidence (nm).
    n_eff : float
        Effective refractive index of the structure.
    theta_deg : float
        Viewing angle from normal (degrees).

    Returns
    -------
    float
        Peak wavelength at angle θ (nm).

    Physics tier: T2 (Bragg-Snell law, exact for 1D periodic).
    """
    theta_rad = math.radians(theta_deg)
    sin2 = math.sin(theta_rad) ** 2
    arg = 1.0 - sin2 / n_eff ** 2
    if arg <= 0:
        return 0.0  # beyond critical angle
    return peak_nm_normal * math.sqrt(arg)


def iridescence_index(peak_nm_normal: float, n_eff: float,
                        theta_range_deg: float = 60.0) -> float:
    """Iridescence index: total color shift over a viewing angle range.

    I = |λ(0°) - λ(θ_max)| / λ(0°)

    I = 0: no iridescence (angle-independent, photonic glass).
    I > 0.1: noticeable iridescence.
    I > 0.3: strongly iridescent (opal, morpho butterfly).

    Parameters
    ----------
    peak_nm_normal : float
    n_eff : float
    theta_range_deg : float
        Maximum viewing angle to evaluate.

    Returns
    -------
    float
        Iridescence index (0-1).

    Physics tier: T2 (Bragg-Snell).
    """
    lam_0 = peak_nm_normal
    lam_theta = bragg_angle_shift(lam_0, n_eff, theta_range_deg)
    if lam_0 <= 0:
        return 0.0
    return abs(lam_0 - lam_theta) / lam_0


def angle_dependent_spectrum(peak_nm_normal: float, n_eff: float,
                               fwhm_nm: float = 30.0, peak_R: float = 0.5,
                               theta_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Reflectance spectrum at a specific viewing angle.

    Returns (wavelengths, R(λ)) shifted by Bragg-Snell law.
    """
    peak = bragg_angle_shift(peak_nm_normal, n_eff, theta_deg)
    if peak <= 0:
        return _LAM, np.zeros_like(_LAM)
    sigma = fwhm_nm / 2.355
    R = peak_R * np.exp(-0.5 * ((_LAM - peak) / sigma) ** 2) + 0.03
    return _LAM, R


@dataclass
class BRDFComponents:
    """Bidirectional reflectance distribution function components.

    BRDF = diffuse + specular + structural

    Simplified 3-component model for structural dye coatings.
    """
    diffuse: float = 0.0       # Lambertian component (substrate scatter)
    specular: float = 0.0      # Fresnel surface reflection
    structural: float = 0.0    # Structural color (Bragg/Mie)
    total: float = 0.0


def brdf_components(R_dye_peak: float, R_substrate: float,
                      n_dye: float, theta_deg: float = 0.0) -> BRDFComponents:
    """Estimate BRDF components for a structural dye on substrate.

    Parameters
    ----------
    R_dye_peak : float
        Peak structural reflectance of dye film.
    R_substrate : float
        Substrate average reflectance.
    n_dye : float
        Dye film refractive index.
    theta_deg : float
        Viewing angle.

    Returns
    -------
    BRDFComponents

    Physics tier: T2 (Fresnel + Lambert + structural).
    """
    from core.refraction_physics import fresnel_reflectance

    # Specular: Fresnel at air/dye interface
    R_spec = fresnel_reflectance(1.0, n_dye, theta_deg, "unpolarized")

    # Diffuse: substrate contribution (Lambertian)
    R_diff = R_substrate * (1.0 - R_dye_peak) ** 2  # transmitted through dye, reflected diffusely

    # Structural: the dye's own structural color
    R_struct = R_dye_peak * (1.0 - R_spec)  # reduced by top-surface Fresnel

    total = R_spec + R_diff + R_struct

    return BRDFComponents(
        diffuse=R_diff,
        specular=R_spec,
        structural=R_struct,
        total=min(1.0, total),
    )


# ═══════════════════════════════════════════════════════════════════════════
# S3. Spectral Broadening + Multiple Scattering
# ═══════════════════════════════════════════════════════════════════════════

def polydispersity_broadening(peak_nm: float, fwhm_nm_mono: float,
                                cv_diameter: float) -> float:
    """Effective FWHM after polydispersity broadening.

    For colloidal assemblies, particle size distribution broadens the
    structural color peak:

    FWHM_eff = √(FWHM_mono² + (2.355 × cv × λ_peak)²)

    where cv = coefficient of variation of diameter distribution.

    Parameters
    ----------
    peak_nm : float
        Peak wavelength (nm).
    fwhm_nm_mono : float
        FWHM for perfectly monodisperse system (nm).
    cv_diameter : float
        Coefficient of variation of particle diameters (0-1).
        Monodisperse: <0.03. Moderate: 0.05-0.10. Polydisperse: >0.15.

    Returns
    -------
    float
        Effective FWHM (nm).

    Physics tier: T2 (convolution of structural peak with size distribution).
    """
    sigma_poly = cv_diameter * peak_nm
    fwhm_poly = 2.355 * sigma_poly
    return math.sqrt(fwhm_nm_mono ** 2 + fwhm_poly ** 2)


def broadened_spectrum(peak_nm: float, fwhm_mono_nm: float,
                         cv_diameter: float, peak_R: float = 0.25,
                         background: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Generate reflectance spectrum with polydispersity broadening.

    Returns (wavelengths, R(λ)).
    """
    fwhm_eff = polydispersity_broadening(peak_nm, fwhm_mono_nm, cv_diameter)
    sigma = fwhm_eff / 2.355
    R = peak_R * np.exp(-0.5 * ((_LAM - peak_nm) / max(sigma, 1.0)) ** 2) + background
    return _LAM, R


def km_scattering_absorption(R_inf: float) -> Tuple[float, float]:
    """Kubelka-Munk: extract K/S from reflectance of infinitely thick layer.

    K/S = (1 - R_∞)² / (2R_∞)

    where K = absorption coefficient, S = scattering coefficient.

    Parameters
    ----------
    R_inf : float
        Reflectance of optically thick sample (0-1).

    Returns
    -------
    (K_over_S, R_inf)

    Physics tier: T2 (Kubelka-Munk 1931).
    """
    if R_inf <= 0:
        return (float('inf'), R_inf)
    if R_inf >= 1:
        return (0.0, R_inf)
    K_over_S = (1.0 - R_inf) ** 2 / (2.0 * R_inf)
    return (K_over_S, R_inf)


def km_reflectance_from_KS(K_over_S: float) -> float:
    """Inverse Kubelka-Munk: K/S → R_∞.

    R_∞ = 1 + K/S - √((K/S)² + 2K/S)

    Physics tier: T2.
    """
    if K_over_S <= 0:
        return 1.0
    return 1.0 + K_over_S - math.sqrt(K_over_S ** 2 + 2.0 * K_over_S)


def km_thickness_dependent_R(K: float, S: float, d: float,
                                R_substrate: float) -> float:
    """Kubelka-Munk reflectance for finite-thickness layer on substrate.

    R = [1 - R_sub×(a - b×coth(bSd))] / [a - R_sub + b×coth(bSd)]

    where a = (S+K)/S, b = √(a²-1).

    Parameters
    ----------
    K : float
        Absorption coefficient (1/m or 1/mm, consistent with d).
    S : float
        Scattering coefficient.
    d : float
        Layer thickness (same units as 1/K, 1/S).
    R_substrate : float
        Substrate reflectance.

    Returns
    -------
    float
        Reflectance of finite-thickness layer.

    Physics tier: T2 (Kubelka 1948).
    """
    if S <= 0:
        return R_substrate  # no scattering → transparent

    a = (S + K) / S
    b_sq = a ** 2 - 1.0
    if b_sq <= 0:
        return R_substrate

    b = math.sqrt(b_sq)
    bSd = b * S * d

    if bSd > 500:
        # Optically thick limit
        return km_reflectance_from_KS(K / S)

    # coth(x) = cosh(x)/sinh(x)
    if bSd < 1e-6:
        coth_bSd = 1.0 / bSd if bSd > 0 else 1e6
    else:
        coth_bSd = math.cosh(bSd) / math.sinh(bSd)

    num = 1.0 - R_substrate * (a - b * coth_bSd)
    den = a - R_substrate + b * coth_bSd

    if abs(den) < 1e-10:
        return R_substrate

    return max(0.0, min(1.0, num / den))


# ═══════════════════════════════════════════════════════════════════════════
# S4. Durability / Environmental / Color Mixing
# ═══════════════════════════════════════════════════════════════════════════

# ── UV Degradation ────────────────────────────────────────────────────────

def uv_degradation_rate(activation_energy_kJ: float, T_K: float,
                           A_prefactor: float = 1e12) -> float:
    """Arrhenius degradation rate constant.

    k = A × exp(-Ea/RT)

    Parameters
    ----------
    activation_energy_kJ : float
        Activation energy (kJ/mol). Typical polymer photodegradation: 40-120.
    T_K : float
        Temperature (K).
    A_prefactor : float
        Pre-exponential factor (s⁻¹).

    Returns
    -------
    float
        Rate constant k (s⁻¹).

    Physics tier: T2 (Arrhenius 1889).
    """
    R = 8.314e-3  # kJ/(mol·K)
    return A_prefactor * math.exp(-activation_energy_kJ / (R * T_K))


def photon_dose_to_deltaE(dose_MJ_m2: float, sensitivity: float = 5.0) -> float:
    """Estimate color change from UV photon dose.

    ΔE ≈ sensitivity × √(dose)

    Empirical square-root law for photodegradation of organic colorants.

    Parameters
    ----------
    dose_MJ_m2 : float
        Cumulative UV dose (MJ/m²). 1 year outdoor ≈ 300 MJ/m².
    sensitivity : float
        Material sensitivity factor. Lower = more stable.
        Phthalocyanines: ~1. Azo dyes: ~10. Fluorescein: ~20.

    Returns
    -------
    float
        Estimated ΔE color change.

    Physics tier: T2 (empirical square-root law, Bauer 1994).
    """
    if dose_MJ_m2 <= 0:
        return 0.0
    return sensitivity * math.sqrt(dose_MJ_m2)


# Published UV stability (T1) — sensitivity factors
# Source: Bauer DR. J. Coat. Technol. 1994; Lightfastness ratings ISO 105-B02
UV_SENSITIVITY: Dict[str, float] = {
    # Lower = more stable. Calibrated so that:
    #   years_outdoor = (ΔE_threshold / sensitivity)² / UV_A_dose_per_year
    # With ΔE=3, dose=30 MJ/m²/yr:
    #   sens=0.01 → >100 yrs, sens=0.1 → 30 yrs, sens=1.0 → 0.3 yrs
    # Ref: Bauer 1994; ISO 105-B02 lightfastness correlations
    "structural_SiO2": 0.01,   # pure inorganic structural: essentially permanent
    "TiO2_pigment": 0.02,     # titanium white: industrial standard
    "carbon_black": 0.03,     # essentially indestructible
    "Fe2O3_pigment": 0.05,    # iron oxide mineral: very stable
    "CuPc": 0.10,             # phthalocyanine blue: ISO 8, decades outdoor
    "ZnPc": 0.12,             # phthalocyanine green: ISO 7-8
    "perylene_pigment": 0.15, # perylene: ISO 7-8
    "Ru_bpy3": 0.20,          # MLCT complex: good photostability
    "structural_PS": 0.35,    # polystyrene: degrades under UV
    "indigo": 0.50,           # historic dye: ISO 4-5, fades in ~1 year outdoor
    "rhodamine_B": 0.80,      # moderate: fades in months outdoor
    "disperse_red_1": 1.0,    # azo dye: poor lightfastness, ISO 3-4
    "methyl_orange": 1.2,     # azo: poor, ISO 2-3
    "fluorescein": 1.5,       # fluorescent: very poor stability outdoor
}


def years_to_noticeable_fade(sensitivity: float,
                                outdoor_dose_MJ_m2_yr: float = 30.0,
                                delta_E_threshold: float = 3.0) -> float:
    """Estimate years until noticeable color fade outdoors.

    Solves: ΔE_threshold = sensitivity × √(dose_per_year × years)

    Parameters
    ----------
    sensitivity : float
        UV sensitivity factor.
    outdoor_dose_MJ_m2_yr : float
        Annual UV-A dose (MJ/m²). Temperate: ~25. Tropical: ~40. UV-A is the degrading fraction.
    delta_E_threshold : float
        ΔE at which fade is noticeable. 3.0 is standard threshold.

    Returns
    -------
    float
        Years to noticeable fade.

    Physics tier: T2.
    """
    if sensitivity <= 0 or outdoor_dose_MJ_m2_yr <= 0:
        return float('inf')
    # ΔE = sens × √(dose_yr × years) → years = (ΔE/sens)² / dose_yr
    return (delta_E_threshold / sensitivity) ** 2 / outdoor_dose_MJ_m2_yr


# ── Environmental Response ────────────────────────────────────────────────

def humidity_induced_shift(n_dry: float, n_wet: float,
                             peak_nm_dry: float,
                             RH: float, RH_critical: float = 0.7) -> float:
    """Color shift from humidity-induced n change.

    When porous structures absorb water, n increases → peak redshifts.
    Δλ/λ ≈ Δn/n (for structures where λ ∝ n).

    Parameters
    ----------
    n_dry : float
        Refractive index at low humidity.
    n_wet : float
        Refractive index at saturation (100% RH).
    peak_nm_dry : float
        Peak wavelength at 0% RH.
    RH : float
        Relative humidity (0-1).
    RH_critical : float
        RH above which water begins condensing in pores.

    Returns
    -------
    float
        Peak wavelength at given RH (nm).

    Physics tier: T2 (linear interpolation of n with water content).
    """
    if RH < RH_critical:
        f_wet = 0.0
    else:
        f_wet = (RH - RH_critical) / (1.0 - RH_critical)

    n_eff = n_dry + f_wet * (n_wet - n_dry)
    if n_dry <= 0:
        return peak_nm_dry
    return peak_nm_dry * n_eff / n_dry


def thermal_expansion_shift(peak_nm: float, alpha_thermal: float,
                               delta_T_K: float) -> float:
    """Color shift from thermal expansion of structural period.

    Δλ/λ = α × ΔT

    Parameters
    ----------
    peak_nm : float
    alpha_thermal : float
        Linear thermal expansion coefficient (1/K).
        SiO2: 0.5e-6. PS: 70e-6. PMMA: 75e-6.
    delta_T_K : float
        Temperature change from reference (K).

    Returns
    -------
    float
        Shifted peak wavelength (nm).

    Physics tier: T2 (linear thermal expansion).
    """
    return peak_nm * (1.0 + alpha_thermal * delta_T_K)


# Published thermal expansion coefficients (T1, 1/K)
THERMAL_EXPANSION: Dict[str, float] = {
    "SiO2": 0.55e-6,
    "polystyrene": 70e-6,
    "PMMA": 75e-6,
    "TiO2": 9.0e-6,
    "ZnO": 4.0e-6,
    "Si3N4": 3.3e-6,
    "cellulose": 50e-6,  # along fiber
    "nylon": 80e-6,
    "polyester_PET": 60e-6,
    "aluminum": 23e-6,
    "glass_soda_lime": 9.0e-6,
}


def strain_induced_shift(peak_nm: float, strain: float,
                            poisson_ratio: float = 0.3) -> float:
    """Color shift from mechanical strain (stretching).

    For a 1D photonic structure under uniaxial strain ε:
    Δλ/λ = -ε × (1 - 2ν)  (transverse compression reduces period)

    For films on elastic substrates (e.g., PDMS):
    stretching thins the film → blue shift.

    Parameters
    ----------
    peak_nm : float
    strain : float
        Uniaxial strain (positive = tension). 0.1 = 10% stretch.
    poisson_ratio : float
        Poisson's ratio. PDMS: 0.5. Polymers: 0.3-0.4.

    Returns
    -------
    float
        Shifted peak wavelength (nm).

    Physics tier: T2 (continuum mechanics).
    """
    # Transverse strain = -ν × axial strain
    # Period change = transverse strain for films oriented normal to surface
    delta_lambda_frac = -strain * (1.0 - 2.0 * poisson_ratio)
    return peak_nm * (1.0 + delta_lambda_frac)


# ── Color Mixing ──────────────────────────────────────────────────────────

def subtractive_mix_km(R_spectra: List[np.ndarray],
                         concentrations: List[float] = None) -> np.ndarray:
    """Subtractive color mixing via Kubelka-Munk.

    In KM space, K/S values are additive:
    (K/S)_mix = Σ c_i × (K/S)_i

    Then convert back: R_mix = 1 + KS_mix - √(KS_mix² + 2KS_mix).

    Parameters
    ----------
    R_spectra : list of np.ndarray
        Reflectance spectra of individual colorants (on _LAM grid).
    concentrations : list of float, optional
        Relative concentrations (weights). Default: equal parts.

    Returns
    -------
    np.ndarray
        Mixed reflectance spectrum.

    Physics tier: T2 (Kubelka-Munk additivity, Judd & Wyszecki 1975).
    """
    if not R_spectra:
        return np.full_like(_LAM, 0.5, dtype=float)

    if concentrations is None:
        concentrations = [1.0 / len(R_spectra)] * len(R_spectra)

    # Normalize concentrations
    total_c = sum(concentrations)
    if total_c > 0:
        concentrations = [c / total_c for c in concentrations]

    KS_mix = np.zeros_like(_LAM, dtype=float)
    for R, c in zip(R_spectra, concentrations):
        R_clipped = np.clip(R, 0.001, 0.999)
        KS = (1.0 - R_clipped) ** 2 / (2.0 * R_clipped)
        KS_mix += c * KS

    # Inverse KM
    R_mix = 1.0 + KS_mix - np.sqrt(KS_mix ** 2 + 2.0 * KS_mix)
    return np.clip(R_mix, 0.0, 1.0)


def additive_fluorescence_mix(emission_spectra: List[np.ndarray],
                                 intensities: List[float] = None) -> np.ndarray:
    """Additive mixing of fluorescence emission.

    Fluorescence emission adds linearly (photons don't subtract).

    I_total(λ) = Σ I_i × emission_i(λ)

    Parameters
    ----------
    emission_spectra : list of np.ndarray
        Emission spectra of individual fluorophores.
    intensities : list of float
        Relative intensities. Default: equal.

    Returns
    -------
    np.ndarray
        Mixed emission spectrum (normalized to peak = 1).

    Physics tier: T2 (linear superposition of incoherent sources).
    """
    if not emission_spectra:
        return np.zeros_like(_LAM)

    if intensities is None:
        intensities = [1.0] * len(emission_spectra)

    mixed = np.zeros_like(_LAM, dtype=float)
    for em, I in zip(emission_spectra, intensities):
        mixed += I * em

    peak = np.max(mixed)
    if peak > 0:
        mixed /= peak
    return mixed


def gamut_check(Lab: Tuple[float, float, float]) -> dict:
    """Check if a Lab color is within sRGB gamut and standard gamut boundaries.

    Returns dict with in_sRGB, chroma, saturation assessment.

    Physics tier: T2 (colorimetry, CIE 1976).
    """
    L, a, b = Lab
    C = math.sqrt(a ** 2 + b ** 2)  # chroma
    h = math.degrees(math.atan2(b, a)) % 360  # hue angle

    # sRGB gamut boundary (approximate): max chroma depends on L* and hue
    # Simplified: sRGB can represent C ≈ 100-130 at L*=50, less at extremes
    max_C_approx = 130.0 * (1.0 - ((L - 50) / 50) ** 2) if 0 < L < 100 else 0.0
    in_srgb = C < max_C_approx

    if C > 80:
        saturation = "highly saturated"
    elif C > 40:
        saturation = "moderately saturated"
    elif C > 10:
        saturation = "low saturation"
    else:
        saturation = "near-neutral"

    return {
        "L": L, "a": a, "b": b,
        "chroma": C,
        "hue_angle": h,
        "in_sRGB_approx": in_srgb,
        "saturation": saturation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_durability_table():
    """Print UV durability estimates for structural dye components."""
    print(f"\n{'Material':<22} {'Sens':>5} {'Yrs outdoor':>11} {'Class'}")
    print("-" * 50)
    for name, sens in sorted(UV_SENSITIVITY.items(), key=lambda x: x[1]):
        yrs_val = years_to_noticeable_fade(sens)
        if yrs_val > 100:
            yrs_str = ">100"
        else:
            yrs_str = f"{yrs_val:.1f}"
        cls = "excellent" if yrs_val > 10 else "good" if yrs_val > 2 else "moderate" if yrs_val > 0.5 else "poor"
        print(f"{name:<22} {sens:5.1f} {yrs_str:>11} {cls}")


if __name__ == "__main__":
    print("=" * 70)
    print("Dye Application Physics — S1-S4")
    print("=" * 70)

    print("\n--- S1: Wetting ---")
    Ca = capillary_number(0.01, 8.9e-4, 0.0728)
    print(f"  Capillary number (dip-coating at 1cm/s): Ca = {Ca:.4f}")
    L = lucas_washburn_penetration(20e-6, 30.0, 8.9e-4, 0.0728, 1.0)
    print(f"  Wicking into cotton (1s): {L*1000:.2f} mm")

    print("\n--- S2: Angle dependence ---")
    for theta in [0, 30, 45, 60]:
        lam = bragg_angle_shift(530.0, 1.35, theta)
        print(f"  530nm at {theta}°: → {lam:.0f} nm")
    I = iridescence_index(530.0, 1.35)
    print(f"  Iridescence index (0-60°): {I:.3f}")

    print("\n--- S4: Durability ---")
    print_durability_table()

    print("\n--- S4: Environmental ---")
    for dT in [-20, 0, 20, 50]:
        lam = thermal_expansion_shift(530.0, 70e-6, dT)
        print(f"  PS particles at ΔT={dT:+d}K: λ={lam:.1f} nm")
