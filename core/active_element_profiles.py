"""
core/active_element_profiles.py -- Active Building Element Profiles

Extends the multi-physics profile system with 3 energy-harvesting-enabled
profiles for building envelope elements:

  1. wall_panel_active -- Opaque facade panel (MXene conductor, 2-5mm)
  2. window_active -- Semi-transparent window (AgNW/ITO, VLT >= 40%, <1mm)
  3. awning_active -- Flexible awning/canopy (PEDOT conductor, <1mm)

Each profile specifies color, thermal, acoustic, and power weights for
4-objective optimization, plus harvesting material preferences and constraints.

This module does NOT modify core/multiphysics_profiles.py (zero regression).
It provides a parallel `design_active_element()` entry point that combines
existing multi-physics optimization with energy harvesting predictions.

Phase 4 of the Energy Harvesting module.
Depends on: Phase 1 (conductive_base), Phase 2 (energy_harvesting).
Soft dependency on: multiphysics_profiles (for shared dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.energy_harvesting import (
    predict_total_harvest,
    EnvironmentSpec,
    HarvestingSpec,
    PowerBudget,
)
from core.conductive_base import (
    get_conductor,
    predict_sheet_resistance,
    predict_transparency,
    ConductorResult,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HarvestingTargets:
    """Harvesting-specific targets for an active element profile."""
    min_power_density_W_m2: float       # minimum acceptable power output
    pv_material_preference: str         # preferred PV material
    conductor_material: str             # required conductor material
    conductor_thickness_nm: float       # conductor thickness
    min_transparency: float             # minimum VLT (0-1); 0 for opaque
    te_material: str = "Bi2Te3"         # default thermoelectric
    piezo_material: str = "PVDF"        # default piezo
    pb_free_required: bool = True       # require lead-free materials


@dataclass
class ActiveElementProfile:
    """Complete profile for an energy-harvesting building element."""
    application_id: str
    name: str
    description: str

    # Weights: 4-objective (color, thermal, acoustic, power)
    weight_color: float
    weight_thermal: float
    weight_acoustic: float
    weight_power: float

    # Harvesting targets
    harvesting_targets: HarvestingTargets

    # Form factor constraints
    max_thickness_mm: float
    must_be_flexible: bool
    opaque: bool

    # Default environment
    default_environment: EnvironmentSpec = field(
        default_factory=EnvironmentSpec
    )

    @property
    def weights(self) -> Tuple[float, float, float, float]:
        return (self.weight_color, self.weight_thermal,
                self.weight_acoustic, self.weight_power)

    def summary(self) -> str:
        ht = self.harvesting_targets
        lines = [
            f"Active Element Profile: {self.name}",
            f"  {self.description}",
            f"  Weights: color={self.weight_color:.2f}, thermal={self.weight_thermal:.2f}, "
            f"acoustic={self.weight_acoustic:.2f}, power={self.weight_power:.2f}",
            f"  Conductor: {ht.conductor_material} at {ht.conductor_thickness_nm}nm",
            f"  PV preference: {ht.pv_material_preference}",
            f"  Min transparency: {ht.min_transparency:.0%}",
            f"  Max thickness: {self.max_thickness_mm}mm",
            f"  Min power: {ht.min_power_density_W_m2} W/m2",
        ]
        return "\n".join(lines)


@dataclass
class ActiveElementResult:
    """Result from active element design."""
    profile: ActiveElementProfile
    power_budget: PowerBudget
    conductor: ConductorResult
    conductor_transparency: float
    conductor_sheet_R: float
    meets_power_target: bool
    meets_transparency_target: bool
    score: float                    # weighted 4-objective score (0-1)


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

_ACTIVE_PROFILES: Dict[str, ActiveElementProfile] = {
    "wall_panel_active": ActiveElementProfile(
        application_id="wall_panel_active",
        name="Active Wall Panel",
        description="Opaque facade panel with energy harvesting; MXene conductor; 2-5mm thick",
        weight_color=0.20,
        weight_thermal=0.25,
        weight_acoustic=0.20,
        weight_power=0.35,
        harvesting_targets=HarvestingTargets(
            min_power_density_W_m2=5.0,
            pv_material_preference="organic_PM6Y6",
            conductor_material="MXene_Ti3C2",
            conductor_thickness_nm=50.0,
            min_transparency=0.0,  # opaque
            te_material="Bi2Te3",
            piezo_material="PVDF",
        ),
        max_thickness_mm=5.0,
        must_be_flexible=False,
        opaque=True,
        default_environment=EnvironmentSpec(
            solar_irradiance_W_m2=100.0,  # Vancouver vertical south
            delta_T_K=5.0,
            vibration_acceleration_m_s2=0.05,
            vibration_freq_Hz=50.0,
            sound_level_dBA=65.0,
            temperature_C=15.0,
        ),
    ),

    "window_active": ActiveElementProfile(
        application_id="window_active",
        name="Active Window",
        description="Semi-transparent window with energy harvesting; AgNW/ITO conductor; VLT >= 40%",
        weight_color=0.10,
        weight_thermal=0.20,
        weight_acoustic=0.15,
        weight_power=0.30,
        harvesting_targets=HarvestingTargets(
            min_power_density_W_m2=2.0,
            pv_material_preference="perovskite_CsAgBiBr",  # Pb-free, wide bandgap for transparency
            conductor_material="ITO",
            conductor_thickness_nm=150.0,
            min_transparency=0.40,
            te_material="PEDOT_PSS_doped",  # flexible, transparent-compatible
            piezo_material="PVDF",
        ),
        max_thickness_mm=1.0,
        must_be_flexible=False,
        opaque=False,
        default_environment=EnvironmentSpec(
            solar_irradiance_W_m2=200.0,  # higher than wall (direct glazing)
            delta_T_K=10.0,
            vibration_acceleration_m_s2=0.02,
            vibration_freq_Hz=30.0,
            sound_level_dBA=60.0,
            temperature_C=20.0,
        ),
    ),

    "awning_active": ActiveElementProfile(
        application_id="awning_active",
        name="Active Awning",
        description="Flexible awning/canopy with energy harvesting; PEDOT conductor; <1mm; 5cm bend radius",
        weight_color=0.25,
        weight_thermal=0.15,
        weight_acoustic=0.15,
        weight_power=0.30,
        harvesting_targets=HarvestingTargets(
            min_power_density_W_m2=3.0,
            pv_material_preference="organic_PM6Y6",  # flexible organic PV
            conductor_material="PEDOT_PSS",
            conductor_thickness_nm=200.0,
            min_transparency=0.0,  # opaque awning
            te_material="PEDOT_PSS_doped",
            piezo_material="P_VDF_TrFE",  # flexible
        ),
        max_thickness_mm=1.0,
        must_be_flexible=True,
        opaque=True,
        default_environment=EnvironmentSpec(
            solar_irradiance_W_m2=300.0,  # angled toward sky
            delta_T_K=8.0,
            vibration_acceleration_m_s2=0.2,   # wind-induced
            vibration_freq_Hz=20.0,
            sound_level_dBA=70.0,
            temperature_C=18.0,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------

def get_active_profile(application_id: str) -> ActiveElementProfile:
    """Return an active element profile by ID."""
    if application_id not in _ACTIVE_PROFILES:
        raise KeyError(
            f"Unknown active profile '{application_id}'. "
            f"Available: {list_active_profiles()}"
        )
    return _ACTIVE_PROFILES[application_id]


def list_active_profiles() -> List[str]:
    """Return sorted list of active element profile IDs."""
    return sorted(_ACTIVE_PROFILES.keys())


# ---------------------------------------------------------------------------
# Design function
# ---------------------------------------------------------------------------

def design_active_element(
    application_id: str,
    environment: Optional[EnvironmentSpec] = None,
    transmitted_fraction: float = 0.5,
) -> ActiveElementResult:
    """
    Design an active building element with energy harvesting.

    Evaluates the harvesting stack for the given profile and environment,
    checks against targets, and computes a weighted 4-objective score.

    Parameters
    ----------
    application_id : str
        Profile ID: "wall_panel_active", "window_active", "awning_active".
    environment : EnvironmentSpec or None
        Environmental conditions. Uses profile default if None.
    transmitted_fraction : float
        Fraction of solar spectrum reaching PV layer (from color layer).

    Returns
    -------
    ActiveElementResult
    """
    profile = get_active_profile(application_id)
    env = environment or profile.default_environment
    ht = profile.harvesting_targets

    # Build harvesting spec from profile
    h_spec = HarvestingSpec(
        pv_material=ht.pv_material_preference,
        pv_thickness_nm=300.0,
        te_material=ht.te_material,
        n_te_couples_per_m2=10000.0,
        piezo_material=ht.piezo_material,
        piezo_thickness_um=100.0,
        area_m2=1.0,
    )

    # Predict harvesting
    power_budget = predict_total_harvest(
        harvesting_spec=h_spec,
        environment_spec=env,
        transmitted_fraction=transmitted_fraction,
    )

    # Predict conductor properties
    cond_rs = predict_sheet_resistance(
        ht.conductor_material, ht.conductor_thickness_nm, env.temperature_C,
    )
    cond_tr = predict_transparency(
        ht.conductor_material, ht.conductor_thickness_nm,
    )
    cond_mat = get_conductor(ht.conductor_material)

    conductor = ConductorResult(
        material=ht.conductor_material,
        thickness_nm=ht.conductor_thickness_nm,
        temperature_C=env.temperature_C,
        sheet_resistance_ohm_sq=cond_rs,
        transparency=cond_tr,
        sigma_eff_S_m=1.0 / (cond_rs * ht.conductor_thickness_nm * 1e-9),
        power_loss_W_m2=0.0,
        click_handles_top=list(cond_mat.click_handles_top),
        click_handles_bottom=list(cond_mat.click_handles_bottom),
        flexible=cond_mat.flexible,
        safe=cond_mat.safe,
    )

    # Check targets
    meets_power = power_budget.total_W_m2 >= ht.min_power_density_W_m2
    meets_transparency = cond_tr >= ht.min_transparency

    # 4-objective score (normalized 0-1 per objective, then weighted)
    # Power: normalize against 50 W/m2 as practical ceiling
    power_score = min(1.0, power_budget.total_W_m2 / 50.0)
    # Transparency: direct (already 0-1)
    transparency_score = cond_tr
    # Thermal and acoustic placeholder scores (1.0 = not constraining)
    thermal_score = 0.8  # placeholder until Pareto integration
    acoustic_score = 0.8  # placeholder until Pareto integration

    score = (
        profile.weight_color * transparency_score +
        profile.weight_thermal * thermal_score +
        profile.weight_acoustic * acoustic_score +
        profile.weight_power * power_score
    )

    return ActiveElementResult(
        profile=profile,
        power_budget=power_budget,
        conductor=conductor,
        conductor_transparency=cond_tr,
        conductor_sheet_R=cond_rs,
        meets_power_target=meets_power,
        meets_transparency_target=meets_transparency,
        score=score,
    )
