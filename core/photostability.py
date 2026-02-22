"""
core/photostability.py — Sprint 29a: Photodegradation + Photothermal Capture

UV/visible light stability of binder materials. Photothermal heating
via plasmonic nanoparticles for capture and release.

Physics:
  Photodegradation: quantum yield × absorption cross-section × flux
  Plasmonic heating: ΔT = σ_abs × I / (4π × r × κ)
  LSPR wavelength: depends on particle size, shape, dielectric
"""
from dataclasses import dataclass
import math


@dataclass
class PhotostabilityProfile:
    """UV/visible light stability assessment."""
    material_type: str
    uv_tolerance_dose_j_cm2: float  # Dose at 50% degradation (254 nm)
    visible_tolerance_dose_j_cm2: float  # Dose at 50% degradation (vis)
    degradation_mechanism: str
    stability_class: str            # "excellent", "good", "moderate", "poor", "UV_sensitive"
    operational_lifetime_outdoor_days: float  # In direct sunlight
    protection_strategy: str
    notes: str = ""

@dataclass
class PlasmonicProfile:
    """Plasmonic nanoparticle heating for capture/release."""
    particle_type: str
    lspr_wavelength_nm: float       # Localized surface plasmon resonance
    absorption_cross_section_nm2: float
    heating_efficiency: float       # Fraction of absorbed light → heat
    delta_T_per_W_cm2: float        # Temperature rise per irradiance
    optimal_laser_nm: float         # Matched to LSPR
    photothermal_release: bool      # Can drive thermal release
    capture_mechanism: str          # How photothermal aids capture
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# PHOTOSTABILITY DATABASE
# ═══════════════════════════════════════════════════════════════════════════

# UV dose (J/cm²) at 254 nm for 50% activity loss
_PHOTOSTABILITY = {
    # Inorganic — generally UV-stable
    "zeolite":          (1e6, 1e8,  "Framework inert to UV", "excellent"),
    "mesoporous_silica": (1e6, 1e8, "SiO2 transparent in UV-Vis", "excellent"),
    "ldh":              (5e5, 1e7,  "Hydroxide layers stable; interlayer dyes may bleach", "good"),
    "mof":              (1e3, 1e5,  "Organic linker photolysis; carboxylate more stable than azole", "moderate"),
    "carbon_nanotube":  (1e5, 1e7,  "sp2 carbon absorbs UV but stable", "good"),
    # Organic — moderate
    "mip":              (5e2, 1e4,  "Polymer chain scission under UV; add UV stabilizer", "moderate"),
    "cof":              (1e2, 1e3,  "Imine/azine bonds UV-labile", "poor"),
    "coordination_cage": (1e3, 5e4, "Metal nodes stable; organic struts vulnerable", "moderate"),
    "dendrimer":        (2e2, 5e3,  "Branch point photolysis", "poor"),
    # Biological — UV sensitive
    "dna_origami":      (10,  1e3,  "Pyrimidine dimers, strand breaks, base oxidation", "UV_sensitive"),
    "aptamer":          (10,  1e3,  "Same as DNA; G-quadruplex slightly more resistant", "UV_sensitive"),
    "peptide":          (50,  5e3,  "Trp/Tyr/Cys photooxidation; disulfide scrambling", "poor"),
    # Photoswitches (special)
    "azobenzene":       (1e4, 1e5,  "Designed for photocycling; fatigue after ~10⁵ cycles", "good"),
    "spiropyran":       (1e2, 1e3,  "Merocyanine form degrades; limited fatigue resistance", "moderate"),
    "diarylethene":     (1e4, 1e5,  "Thermally stable both forms; good fatigue", "good"),
}

# Solar UV irradiance: ~4 W/m² UVB (280-315) + ~30 W/m² UVA (315-400)
_SOLAR_UV_W_M2 = 34  # Total UV at ground level
_SOLAR_UV_J_CM2_PER_DAY = _SOLAR_UV_W_M2 * 3600 * 8 / 10000  # 8 hrs, convert to J/cm²


def assess_photostability(material_type, outdoor_exposure=False):
    """Assess UV/visible light stability of binder material."""
    key = material_type.lower().replace(" ", "_")

    data = None
    for k in _PHOTOSTABILITY:
        if k in key or key in k:
            data = _PHOTOSTABILITY[k]
            break

    if data is None:
        return PhotostabilityProfile(material_type, 1e4, 1e6, "Unknown", "unknown",
                                      1e4 / _SOLAR_UV_J_CM2_PER_DAY, "Unknown material")

    uv_dose, vis_dose, mechanism, rating = data

    lifetime_days = uv_dose / _SOLAR_UV_J_CM2_PER_DAY if _SOLAR_UV_J_CM2_PER_DAY > 0 else 1e6

    protection = "None needed"
    if rating in ("poor", "UV_sensitive"):
        protection = "Encapsulate in UV-opaque housing or add UV-absorbing coating"
    elif rating == "moderate":
        protection = "UV stabilizer additive (benzotriazole/HALS) or indoor use only"

    notes = ""
    if outdoor_exposure and lifetime_days < 30:
        notes = f"WARNING: {lifetime_days:.0f} day outdoor lifetime. {protection}"

    return PhotostabilityProfile(
        material_type=material_type,
        uv_tolerance_dose_j_cm2=uv_dose,
        visible_tolerance_dose_j_cm2=vis_dose,
        degradation_mechanism=mechanism,
        stability_class=rating,
        operational_lifetime_outdoor_days=round(lifetime_days, 1),
        protection_strategy=protection,
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PLASMONIC NANOPARTICLE HEATING
# ═══════════════════════════════════════════════════════════════════════════

_PLASMONIC_PARTICLES = {
    # type: (LSPR nm, σ_abs nm², heating_eff, mechanism)
    "Au_sphere_20nm":   (520,  320,   0.95, "Photothermal: local heating denatures binder → release"),
    "Au_sphere_50nm":   (530,  2000,  0.85, "Photothermal: efficient visible-light heating"),
    "Au_nanorod_AR3":   (780,  5000,  0.90, "NIR photothermal: tissue-penetrating wavelength"),
    "Au_nanorod_AR4":   (850,  6000,  0.90, "NIR photothermal: deep tissue penetration"),
    "Au_nanoshell":     (800,  8000,  0.85, "NIR tunable; silica core + Au shell"),
    "Au_nanocage":      (750,  4500,  0.88, "Hollow interior for cargo loading + photothermal release"),
    "Ag_sphere_40nm":   (410,  3000,  0.70, "UV-Vis plasmonic; less biocompatible than Au"),
    "Ag_nanoprism":     (700,  5000,  0.75, "Tunable NIR; sharp resonance"),
    "Fe3O4_Au_core_shell": (550, 1500, 0.80, "Dual magnetic + photothermal"),
}


def predict_photothermal(particle_type, irradiance_W_cm2=1.0,
                          medium_thermal_conductivity=0.6):
    """Predict photothermal heating from plasmonic nanoparticle.

    ΔT = σ_abs × I / (4π × r × κ)
    where r = particle radius, κ = thermal conductivity of medium
    """
    data = _PLASMONIC_PARTICLES.get(particle_type)
    if data is None:
        return PlasmonicProfile(particle_type, 0, 0, 0, 0, 0, False, "Unknown particle type")

    lspr, sigma_nm2, eff, mechanism = data

    sigma_m2 = sigma_nm2 * 1e-18  # nm² → m²

    # Collective heating at typical NP concentration (100 nM)
    # ΔT/dt = σ_abs × I × [NP] × ε / (ρ × Cp)
    I = irradiance_W_cm2 * 1e4  # W/cm² → W/m²
    NP_conc = 100e-9 * 6.022e23  # 100 nM in particles/m³
    rho_cp = 4.18e6  # J/(m³·K) water

    P_density = sigma_m2 * I * NP_conc * eff  # W/m³
    dT_per_s = P_density / rho_cp  # °C/s
    delta_T_60s = dT_per_s * 60  # °C in 1 minute

    photothermal_release = delta_T_60s > 5.0  # >5°C in 1 min = useful

    return PlasmonicProfile(
        particle_type=particle_type,
        lspr_wavelength_nm=lspr,
        absorption_cross_section_nm2=sigma_nm2,
        heating_efficiency=eff,
        delta_T_per_W_cm2=round(delta_T_60s, 2),
        optimal_laser_nm=lspr,
        photothermal_release=photothermal_release,
        capture_mechanism=mechanism,
        notes=f"Collective ΔT={delta_T_60s:.1f}°C in 60s at 100 nM NP, {irradiance_W_cm2} W/cm²; "
              f"dT/dt={dT_per_s:.2f} °C/s",
    )

