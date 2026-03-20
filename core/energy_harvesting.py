"""
core/energy_harvesting.py -- Energy Harvesting Layer Forward Models

Predicts power output from four harvesting modalities integrated into
MABE building envelope elements:

  1. Photovoltaic (PV) -- solar cells below the structural color layer
  2. Thermoelectric (TEG) -- Seebeck generators across Kapitza barriers
  3. Piezoelectric (vibration) -- mechanical vibration harvesters
  4. Acoustic-to-piezo -- sound pressure harvesting (negligible power; sensing value)

Each sublayer has its own material database and forward model.
The combined model sums all contributions into a PowerBudget.

Physics:
  - PV: Shockley-Queisser framework; spectral integration above E_g;
    temperature derating from module-level coefficient.
  - TEG: Carnot-bounded ZT-dependent conversion efficiency.
  - Piezo: Matched-load power from d33 coupling under harmonic stress.
  - Acoustic: Sound pressure -> absorbed intensity -> piezo conversion;
    always negligible for power, valuable for sensing.

Soft dependencies (graceful fallback if absent):
  - optical/photonic_glass.py (transmitted spectrum)
  - core/colloidal_thermal.py (delta_T for TEG)
  - core/colloidal_acoustic.py (alpha spectrum for acoustic harvest)

Data tier: Tier 2 (DOI per entry; values from peer-reviewed ranges).

References:
  - Shockley & Queisser, J. Appl. Phys. 1961, 32, 510
  - Snyder & Toberer, Nat. Mater. 2008, 7, 105 (thermoelectrics)
  - Roundy et al., Comput. Commun. 2003, 26, 1131 (piezo vibration)
  - Li et al., Appl. Phys. Rev. 2014, 1, 041301 (acoustic energy harvesting)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

AM15_TOTAL_W_m2 = 1000.0
VANCOUVER_VERTICAL_SOUTH_AVG_W_m2 = 100.0
PLANCK_h = 6.626e-34           # J*s
SPEED_OF_LIGHT = 3.0e8         # m/s
ELECTRON_CHARGE = 1.602e-19    # C
BOLTZMANN_k = 1.381e-23        # J/K
EPSILON_0 = 8.854e-12          # F/m
REF_INTENSITY_W_m2 = 1.0e-12   # 0 dB SPL reference


# ---------------------------------------------------------------------------
# AM1.5G spectral approximation
# ---------------------------------------------------------------------------

def _am15_spectral_irradiance(wavelength_nm: float) -> float:
    """
    Approximate AM1.5G spectral irradiance (W/m2/nm).

    Uses a simplified blackbody-envelope model scaled to integrate to
    ~1000 W/m2 over 280-4000 nm, with atmospheric absorption dips
    approximated as Gaussian notches at 940, 1130, 1380, 1870 nm.

    Good enough for relative spectral utilization estimates.
    Not a substitute for ASTM G173 tabulated data.
    """
    # AM1.5 is zero below ~280 nm (ozone cutoff)
    if wavelength_nm < 280.0 or wavelength_nm > 4000.0:
        return 0.0

    wl_m = wavelength_nm * 1e-9
    # Planck function at T_sun ~ 5778 K, scaled
    T = 5778.0
    # Spectral radiance (relative)
    x = PLANCK_h * SPEED_OF_LIGHT / (wl_m * BOLTZMANN_k * T)
    if x > 500:
        return 0.0
    bb = 1.0 / (wl_m ** 5 * (math.exp(x) - 1.0))

    # Atmospheric absorption notches (approximate)
    notches = [(940, 30), (1130, 40), (1380, 60), (1870, 80)]
    atm = 1.0
    for center, width in notches:
        atm *= 1.0 - 0.7 * math.exp(-0.5 * ((wavelength_nm - center) / width) ** 2)

    # Scale factor calibrated so integral over 280-4000 nm (10nm steps) ~ 1000 W/m2
    SCALE = 6.793e-30
    return max(0.0, SCALE * bb * atm)


def am15_photon_flux(wavelength_nm: float) -> float:
    """Photon flux (photons/m2/s/nm) from AM1.5G at given wavelength."""
    E_photon = PLANCK_h * SPEED_OF_LIGHT / (wavelength_nm * 1e-9)
    return _am15_spectral_irradiance(wavelength_nm) / E_photon


# ---------------------------------------------------------------------------
# PV materials database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PVMaterial:
    """Photovoltaic absorber material properties."""
    name: str
    bandgap_eV: float
    pce_lab: float           # best lab PCE (fraction, not percent)
    pce_module: float        # realistic module PCE (fraction)
    temp_coeff_per_C: float  # relative PCE change per degree C (negative)
    pb_free: bool
    source: str


_PV_MATERIALS = {
    "organic_PM6Y6": PVMaterial(
        "organic_PM6Y6", 1.4, 0.182, 0.14, -0.003, True,
        "Yuan et al. Joule 2019, 3, 1140",
    ),
    "organic_PBDBT_ITIC": PVMaterial(
        "organic_PBDBT_ITIC", 1.5, 0.13, 0.10, -0.003, True,
        "Zhao et al. Adv. Mater. 2017, 29, 1604059",
    ),
    "perovskite_MAPbI3": PVMaterial(
        "perovskite_MAPbI3", 1.55, 0.257, 0.20, -0.0045, False,
        "Min et al. Nature 2021, 598, 444",
    ),
    "perovskite_CsAgBiBr": PVMaterial(
        "perovskite_CsAgBiBr", 1.95, 0.06, 0.04, -0.003, True,
        "Slavney et al. J. Am. Chem. Soc. 2017, 139, 5015",
    ),
    "amorphous_Si": PVMaterial(
        "amorphous_Si", 1.7, 0.14, 0.10, -0.002, True,
        "Staebler & Wronski, Appl. Phys. Lett. 1977, 31, 292",
    ),
    "CZTS": PVMaterial(
        "CZTS", 1.5, 0.13, 0.09, -0.003, True,
        "Wang et al. Adv. Energy Mater. 2014, 4, 1301465",
    ),
}


# ---------------------------------------------------------------------------
# TEG materials database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TEMaterial:
    """Thermoelectric material properties."""
    name: str
    seebeck_uV_K: float       # Seebeck coefficient (uV/K)
    zt_300K: float             # figure of merit at 300 K
    sigma_elec_S_m: float
    kappa_te_W_mK: float
    source: str


_TE_MATERIALS = {
    "PEDOT_PSS_doped": TEMaterial(
        "PEDOT_PSS_doped", 30.0, 0.25, 1000.0, 0.3,
        "Bubnova et al. Nat. Mater. 2011, 10, 429",
    ),
    "Bi2Te3": TEMaterial(
        "Bi2Te3", 200.0, 1.0, 1.1e5, 1.5,
        "Poudel et al. Science 2008, 320, 634",
    ),
    "SnSe": TEMaterial(
        "SnSe", 500.0, 2.6, 1.0e4, 0.5,
        "Zhao et al. Nature 2014, 508, 373",
    ),
    "Cu2Se": TEMaterial(
        "Cu2Se", 150.0, 1.5, 5.0e4, 1.0,
        "Liu et al. Nat. Mater. 2012, 11, 422",
    ),
    "Ca3Co4O9": TEMaterial(
        "Ca3Co4O9", 125.0, 0.3, 1.0e4, 2.0,
        "Shikano & Funahashi, Appl. Phys. Lett. 2003, 82, 1851",
    ),
}


# ---------------------------------------------------------------------------
# Piezoelectric materials database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PiezoMaterial:
    """Piezoelectric material properties."""
    name: str
    d33_pC_N: float        # piezoelectric charge coefficient
    epsilon_r: float       # relative permittivity
    density_kg_m3: float
    flexible: bool
    pb_free: bool
    source: str


_PIEZO_MATERIALS = {
    "PVDF": PiezoMaterial(
        "PVDF", 25.0, 10.0, 1780.0, True, True,
        "Kawai, Jpn. J. Appl. Phys. 1969, 8, 975",
    ),
    "P_VDF_TrFE": PiezoMaterial(
        "P_VDF_TrFE", 35.0, 12.0, 1880.0, True, True,
        "Furukawa, Adv. Colloid Interface Sci. 1997, 71, 183",
    ),
    "BaTiO3": PiezoMaterial(
        "BaTiO3", 150.0, 1200.0, 6020.0, False, True,
        "Jaffe et al. Piezoelectric Ceramics, Academic Press 1971",
    ),
    "KNN": PiezoMaterial(
        "KNN", 120.0, 500.0, 4510.0, False, True,
        "Saito et al. Nature 2004, 432, 84",
    ),
    "ZnO": PiezoMaterial(
        "ZnO", 12.0, 10.0, 5610.0, False, True,
        "Wang & Song, Science 2006, 312, 242",
    ),
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PVResult:
    """Photovoltaic prediction output."""
    power_W_m2: float
    jsc_mA_cm2: float
    voc_V: float
    efficiency_actual: float
    spectral_utilization: float
    material: str
    temperature_C: float


@dataclass
class TEGResult:
    """Thermoelectric generator prediction output."""
    power_W_m2: float
    efficiency_percent: float
    voltage_V: float
    delta_T_used_K: float
    material: str


@dataclass
class PiezoResult:
    """Piezoelectric vibration harvester prediction output."""
    power_W_m2: float
    voltage_peak_V: float
    optimal_load_ohm: float
    material: str


@dataclass
class AcousticHarvestResult:
    """Acoustic-to-piezo harvester prediction output."""
    power_W_m2: float
    dominant_frequency_Hz: float
    note: str


@dataclass
class PowerBudget:
    """Combined power output from all harvesting modalities."""
    pv_W_m2: float
    teg_W_m2: float
    piezo_W_m2: float
    acoustic_W_m2: float
    total_W_m2: float
    daily_kWh_m2: float
    per_modality_breakdown: Dict[str, float]


@dataclass
class EnvironmentSpec:
    """Environmental conditions for harvesting prediction."""
    solar_irradiance_W_m2: float = VANCOUVER_VERTICAL_SOUTH_AVG_W_m2
    delta_T_K: float = 5.0
    vibration_acceleration_m_s2: float = 0.1
    vibration_freq_Hz: float = 60.0
    sound_level_dBA: float = 70.0
    temperature_C: float = 20.0


@dataclass
class HarvestingSpec:
    """Material selections for the harvesting layer."""
    pv_material: str = "organic_PM6Y6"
    pv_thickness_nm: float = 300.0
    te_material: str = "Bi2Te3"
    n_te_couples_per_m2: float = 10000.0
    piezo_material: str = "PVDF"
    piezo_thickness_um: float = 100.0
    area_m2: float = 1.0


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------

def get_pv_material(name: str) -> PVMaterial:
    if name not in _PV_MATERIALS:
        raise KeyError(f"Unknown PV material '{name}'. Available: {sorted(_PV_MATERIALS.keys())}")
    return _PV_MATERIALS[name]

def get_te_material(name: str) -> TEMaterial:
    if name not in _TE_MATERIALS:
        raise KeyError(f"Unknown TE material '{name}'. Available: {sorted(_TE_MATERIALS.keys())}")
    return _TE_MATERIALS[name]

def get_piezo_material(name: str) -> PiezoMaterial:
    if name not in _PIEZO_MATERIALS:
        raise KeyError(f"Unknown piezo material '{name}'. Available: {sorted(_PIEZO_MATERIALS.keys())}")
    return _PIEZO_MATERIALS[name]

def list_pv_materials() -> List[str]:
    return sorted(_PV_MATERIALS.keys())

def list_te_materials() -> List[str]:
    return sorted(_TE_MATERIALS.keys())

def list_piezo_materials() -> List[str]:
    return sorted(_PIEZO_MATERIALS.keys())


# ---------------------------------------------------------------------------
# 2a. Photovoltaic model
# ---------------------------------------------------------------------------

def _shockley_queisser_limit(bandgap_eV: float) -> float:
    """
    Approximate Shockley-Queisser PCE limit for a given bandgap.

    Uses a polynomial fit to the detailed-balance curve.
    Maximum 33.7% at 1.34 eV under AM1.5.
    """
    # Fit: quadratic around peak, clamped to [0, 0.337]
    x = bandgap_eV
    # Approximate: peak at 1.34 eV, drops off on both sides
    pce = 0.337 * math.exp(-2.5 * (x - 1.34) ** 2)
    return max(0.0, min(0.337, pce))


def predict_pv_power(
    solar_irradiance_W_m2: float,
    pv_material: str,
    temperature_C: float = 25.0,
    transmitted_fraction: float = 1.0,
    area_m2: float = 1.0,
) -> PVResult:
    """
    Predict PV power output.

    Simplified model:
      P = irradiance * transmitted_fraction * PCE_module * temp_derating * area

    Spectral utilization estimated from bandgap vs AM1.5 spectrum.

    Parameters
    ----------
    solar_irradiance_W_m2 : float
        Incident solar irradiance (W/m2). Use AM15_TOTAL_W_m2 for standard.
    pv_material : str
        Key into PV materials database.
    temperature_C : float
        Cell temperature.
    transmitted_fraction : float
        Fraction of solar spectrum reaching PV layer (0-1). Accounts for
        absorption/reflection by upstream color layer.
    area_m2 : float
        Panel area.

    Returns
    -------
    PVResult
    """
    if solar_irradiance_W_m2 < 0:
        raise ValueError("solar_irradiance_W_m2 must be >= 0")
    if not 0.0 <= transmitted_fraction <= 1.0:
        raise ValueError("transmitted_fraction must be in [0, 1]")

    mat = get_pv_material(pv_material)

    # Temperature derating
    temp_derating = 1.0 + mat.temp_coeff_per_C * (temperature_C - 25.0)
    temp_derating = max(0.0, temp_derating)

    # Spectral utilization: fraction of photons with E > E_g
    # Approximate by integrating AM1.5 above bandgap wavelength
    lambda_cutoff_nm = 1240.0 / mat.bandgap_eV  # E = hc/lambda in eV*nm
    # Integrate usable fraction numerically (coarse)
    total_power = 0.0
    usable_power = 0.0
    for wl in range(280, 4001, 10):
        irr = _am15_spectral_irradiance(float(wl))
        total_power += irr * 10.0
        if wl <= lambda_cutoff_nm:
            usable_power += irr * 10.0
    spectral_util = usable_power / total_power if total_power > 0 else 0.0

    # Effective PCE
    pce_eff = mat.pce_module * temp_derating

    # Power
    incident = solar_irradiance_W_m2 * transmitted_fraction
    power = incident * pce_eff * area_m2

    # Estimate Jsc and Voc for reporting
    # Jsc ~ spectral_util * irradiance * q / (avg_photon_energy)
    avg_photon_eV = 1.8  # rough average for AM1.5
    jsc = (spectral_util * incident * ELECTRON_CHARGE /
           (avg_photon_eV * ELECTRON_CHARGE)) * 0.001 * pce_eff / mat.pce_module
    jsc_mA_cm2 = jsc * 0.1  # A/m2 to mA/cm2 conversion

    # Voc estimate from bandgap (empirical: Voc ~ 0.7 * Eg for organics, 0.85 for perovskites)
    voc = mat.bandgap_eV * 0.75  # rough average

    return PVResult(
        power_W_m2=power / area_m2 if area_m2 > 0 else 0.0,
        jsc_mA_cm2=max(0.0, jsc_mA_cm2),
        voc_V=voc,
        efficiency_actual=pce_eff,
        spectral_utilization=spectral_util,
        material=pv_material,
        temperature_C=temperature_C,
    )


# ---------------------------------------------------------------------------
# 2b. Thermoelectric model
# ---------------------------------------------------------------------------

def predict_teg_power(
    delta_T_K: float,
    te_material: str,
    n_couples_per_m2: float = 10000.0,
    area_m2: float = 1.0,
    t_cold_K: float = 293.15,
) -> TEGResult:
    """
    Predict thermoelectric generator power output.

    Uses ZT-dependent efficiency bounded by Carnot:
      eta = (dT/T_hot) * (sqrt(1+ZT) - 1) / (sqrt(1+ZT) + T_cold/T_hot)

    Power per couple estimated from Seebeck voltage and internal resistance.

    Parameters
    ----------
    delta_T_K : float
        Temperature difference across TEG (K).
    te_material : str
        Key into TE materials database.
    n_couples_per_m2 : float
        Number of p-n thermocouple pairs per m2.
    area_m2 : float
        Panel area.
    t_cold_K : float
        Cold-side temperature (K).

    Returns
    -------
    TEGResult
    """
    if delta_T_K < 0:
        raise ValueError("delta_T_K must be >= 0")

    mat = get_te_material(te_material)

    if delta_T_K == 0:
        return TEGResult(
            power_W_m2=0.0, efficiency_percent=0.0,
            voltage_V=0.0, delta_T_used_K=0.0, material=te_material,
        )

    t_hot_K = t_cold_K + delta_T_K
    t_avg_K = (t_hot_K + t_cold_K) / 2.0

    # Scale ZT with temperature (linear approximation from 300K reference)
    zt_avg = mat.zt_300K * (t_avg_K / 300.0)

    # Carnot-bounded ZT efficiency
    carnot = delta_T_K / t_hot_K
    sqrt_zt = math.sqrt(1.0 + zt_avg)
    eta = carnot * (sqrt_zt - 1.0) / (sqrt_zt + t_cold_K / t_hot_K)
    eta = max(0.0, min(eta, carnot))  # can't exceed Carnot

    # Heat flux through TEG (approximate)
    # Q_in ~ kappa * A * dT / L, but we work per-couple
    # Seebeck voltage per couple
    seebeck_V_K = mat.seebeck_uV_K * 1e-6
    v_per_couple = seebeck_V_K * delta_T_K
    v_total = v_per_couple * math.sqrt(n_couples_per_m2)  # rough: series connection

    # Power density: use efficiency * heat flux approach
    # Q_in per m2 ~ kappa_te * dT / L_leg, with L_leg ~ 1mm typical
    L_leg_m = 1.0e-3
    q_in = mat.kappa_te_W_mK * delta_T_K / L_leg_m  # W/m2 of TEG area
    # Scale by fill factor (thermocouples don't cover 100%)
    fill_factor = min(1.0, n_couples_per_m2 * 1e-6)  # each couple ~ 1mm2 = 1e-6 m2
    power_density = q_in * eta * fill_factor

    return TEGResult(
        power_W_m2=power_density,
        efficiency_percent=eta * 100.0,
        voltage_V=v_per_couple * n_couples_per_m2 * fill_factor,
        delta_T_used_K=delta_T_K,
        material=te_material,
    )


# ---------------------------------------------------------------------------
# 2c. Piezoelectric vibration model
# ---------------------------------------------------------------------------

def predict_piezo_power(
    acceleration_m_s2: float,
    frequency_Hz: float,
    piezo_material: str,
    area_m2: float = 1.0,
    thickness_um: float = 100.0,
) -> PiezoResult:
    """
    Predict piezoelectric power from mechanical vibration.

    Stress from vibration: sigma = rho * thickness * acceleration
    Power at matched load: P = (d33^2 / (4 * eps0 * eps_r)) * sigma^2 * V * f

    Parameters
    ----------
    acceleration_m_s2 : float
        Peak vibration acceleration (m/s2).
    frequency_Hz : float
        Vibration frequency (Hz).
    piezo_material : str
        Key into piezo materials database.
    area_m2 : float
        Active piezo area.
    thickness_um : float
        Piezo layer thickness in micrometers.

    Returns
    -------
    PiezoResult
    """
    if acceleration_m_s2 < 0:
        raise ValueError("acceleration_m_s2 must be >= 0")
    if frequency_Hz < 0:
        raise ValueError("frequency_Hz must be >= 0")

    mat = get_piezo_material(piezo_material)
    t_m = thickness_um * 1e-6

    if acceleration_m_s2 == 0 or frequency_Hz == 0:
        return PiezoResult(
            power_W_m2=0.0, voltage_peak_V=0.0,
            optimal_load_ohm=float("inf"), material=piezo_material,
        )

    # Stress from vibration
    sigma = mat.density_kg_m3 * t_m * acceleration_m_s2  # Pa

    # Volume per m2
    volume_per_m2 = t_m  # m3 per m2 of area

    # d33 in C/N
    d33 = mat.d33_pC_N * 1e-12

    # Power density at matched load
    eps = EPSILON_0 * mat.epsilon_r
    p_density = (d33 ** 2 / (4.0 * eps)) * sigma ** 2 * volume_per_m2 * frequency_Hz

    # Peak voltage
    v_peak = d33 * sigma * t_m / eps

    # Optimal load impedance (at resonance)
    capacitance = eps * area_m2 / t_m
    if capacitance > 0 and frequency_Hz > 0:
        optimal_load = 1.0 / (2.0 * math.pi * frequency_Hz * capacitance)
    else:
        optimal_load = float("inf")

    return PiezoResult(
        power_W_m2=p_density,
        voltage_peak_V=abs(v_peak),
        optimal_load_ohm=optimal_load,
        material=piezo_material,
    )


# ---------------------------------------------------------------------------
# 2d. Acoustic-to-piezo model
# ---------------------------------------------------------------------------

def _dba_to_intensity(level_dBA: float) -> float:
    """Convert A-weighted sound level (dBA) to approximate intensity (W/m2)."""
    return 10.0 ** ((level_dBA - 120.0) / 10.0)


# Default indoor spectral weights (fraction of energy per octave band)
_ACOUSTIC_SPECTRAL_WEIGHTS = {
    250.0: 0.15,
    500.0: 0.25,
    1000.0: 0.30,
    2000.0: 0.20,
    4000.0: 0.10,
}


def predict_acoustic_harvest(
    sound_level_dBA: float,
    piezo_material: str,
    area_m2: float = 1.0,
    alpha_spectrum: Optional[Dict[float, float]] = None,
) -> AcousticHarvestResult:
    """
    Predict power harvested from ambient sound.

    Always negligible for power generation; value is in sensing data.

    Parameters
    ----------
    sound_level_dBA : float
        A-weighted sound pressure level.
    piezo_material : str
        Piezo material for the acoustic transducer.
    area_m2 : float
        Transducer area.
    alpha_spectrum : dict or None
        Absorption coefficient spectrum {freq_Hz: alpha}. If None,
        assumes alpha = 0.5 across all bands.

    Returns
    -------
    AcousticHarvestResult
    """
    if sound_level_dBA < 0:
        raise ValueError("sound_level_dBA must be >= 0")

    mat = get_piezo_material(piezo_material)
    d33 = mat.d33_pC_N * 1e-12
    eps = EPSILON_0 * mat.epsilon_r

    # Electromechanical coupling squared (approximate)
    # k^2 ~ d33^2 / (eps * s33), assume s33 ~ 1/(rho * v^2) ~ 1e-11 m2/N
    s33_approx = 1e-11  # typical compliance for ceramics/polymers
    k2 = d33 ** 2 / (eps * s33_approx)
    k2 = min(k2, 0.5)  # physical bound

    # Quality factor (typical for MEMS/film piezo)
    Q = 50.0

    i_total = _dba_to_intensity(sound_level_dBA)

    total_power = 0.0
    max_freq = 0.0
    max_power_at_freq = 0.0

    for freq, weight in _ACOUSTIC_SPECTRAL_WEIGHTS.items():
        alpha = 0.5  # default
        if alpha_spectrum is not None:
            # Find nearest frequency
            nearest = min(alpha_spectrum.keys(), key=lambda f: abs(f - freq))
            if abs(nearest - freq) < freq * 0.5:
                alpha = alpha_spectrum[nearest]

        i_absorbed = i_total * weight * alpha
        # Piezo conversion
        p_elec = i_absorbed * k2 * Q / (1.0 + k2 * Q)
        total_power += p_elec

        if p_elec > max_power_at_freq:
            max_power_at_freq = p_elec
            max_freq = freq

    return AcousticHarvestResult(
        power_W_m2=total_power,
        dominant_frequency_Hz=max_freq,
        note="negligible power; value is in sensing data",
    )


# ---------------------------------------------------------------------------
# 2e. Combined harvest
# ---------------------------------------------------------------------------

def predict_total_harvest(
    harvesting_spec: Optional[HarvestingSpec] = None,
    environment_spec: Optional[EnvironmentSpec] = None,
    transmitted_fraction: float = 0.5,
    alpha_spectrum: Optional[Dict[float, float]] = None,
) -> PowerBudget:
    """
    Predict total power from all harvesting modalities.

    Parameters
    ----------
    harvesting_spec : HarvestingSpec or None
        Material and geometry selections. Defaults if None.
    environment_spec : EnvironmentSpec or None
        Environmental conditions. Defaults if None.
    transmitted_fraction : float
        Fraction of solar spectrum reaching PV (from upstream color layer).
    alpha_spectrum : dict or None
        Acoustic absorption spectrum from upstream acoustic module.

    Returns
    -------
    PowerBudget
    """
    hs = harvesting_spec or HarvestingSpec()
    env = environment_spec or EnvironmentSpec()

    # PV
    pv = predict_pv_power(
        solar_irradiance_W_m2=env.solar_irradiance_W_m2,
        pv_material=hs.pv_material,
        temperature_C=env.temperature_C,
        transmitted_fraction=transmitted_fraction,
        area_m2=hs.area_m2,
    )

    # TEG
    teg = predict_teg_power(
        delta_T_K=env.delta_T_K,
        te_material=hs.te_material,
        n_couples_per_m2=hs.n_te_couples_per_m2,
        area_m2=hs.area_m2,
        t_cold_K=273.15 + env.temperature_C,
    )

    # Piezo
    piezo = predict_piezo_power(
        acceleration_m_s2=env.vibration_acceleration_m_s2,
        frequency_Hz=env.vibration_freq_Hz,
        piezo_material=hs.piezo_material,
        area_m2=hs.area_m2,
        thickness_um=hs.piezo_thickness_um,
    )

    # Acoustic
    acoustic = predict_acoustic_harvest(
        sound_level_dBA=env.sound_level_dBA,
        piezo_material=hs.piezo_material,
        area_m2=hs.area_m2,
        alpha_spectrum=alpha_spectrum,
    )

    total = pv.power_W_m2 + teg.power_W_m2 + piezo.power_W_m2 + acoustic.power_W_m2

    # Daily energy: assume solar for 5 peak-sun-hours equivalent,
    # TEG/piezo/acoustic for 24h
    daily_kwh = (
        pv.power_W_m2 * 5.0 +
        (teg.power_W_m2 + piezo.power_W_m2 + acoustic.power_W_m2) * 24.0
    ) / 1000.0

    return PowerBudget(
        pv_W_m2=pv.power_W_m2,
        teg_W_m2=teg.power_W_m2,
        piezo_W_m2=piezo.power_W_m2,
        acoustic_W_m2=acoustic.power_W_m2,
        total_W_m2=total,
        daily_kWh_m2=daily_kwh,
        per_modality_breakdown={
            "pv": pv.power_W_m2,
            "teg": teg.power_W_m2,
            "piezo": piezo.power_W_m2,
            "acoustic": acoustic.power_W_m2,
        },
    )
