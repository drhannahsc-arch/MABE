"""
optical/optical_realization.py -- Layer 3/4 Optical Realization Adapters

Converts FieldInteractionSpec + inverse design output into ranked
fabrication specifications.  Parallel structure to the molecular
adapters (CD, crown ether, porphyrin, lignin).

Three adapters:
  PhotonicGlassAdapter  -- disordered colloidal film (non-iridescent)
  BraggOpalAdapter      -- ordered FCC opal film (iridescent)
  TMMMultilayerAdapter  -- vacuum-deposited thin-film stack

Each implements:
  estimate_fidelity(spec) -> OpticalRealizationScore
  design(spec)            -> <Material>FabSpec

The OpticalRanker scores all three against a FieldInteractionSpec and
recommends one, using application-aware weight profiles.

Knowledge sources (zero fitted parameters):
  Stoeber synthesis: Bogush, Tracy & Zukoski, J Colloid Interface Sci 124:688 (1988)
  Vertical deposition: Jiang & Bertone, J Appl Phys 86:4 (1999)
  TMM fabrication: standard optical thin-film engineering references
  Supplier pricing: Sigma-Aldrich 2025 catalog (approximate)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from models import (
    AngularBehavior, ApplicationContext, FieldInteractionSpec,
    FieldType, InteractionParadigm, InteractionSpec, Solvent,
)


# =====================================================================
# FABRICATION SPECS
# =====================================================================

@dataclass
class OpticalFabSpec:
    """Base for all optical fabrication outputs."""
    system: str
    target_wavelength_nm: Optional[float] = None
    target_xy: Optional[tuple] = None
    predicted_xy: Optional[tuple] = None
    predicted_delta_E: float = 999.0
    synthesis_steps: list = field(default_factory=list)
    materials_list: list = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    estimated_time: str = ""
    validation_plan: list = field(default_factory=list)
    notes: str = ""


@dataclass
class PhotonicGlassFabSpec(OpticalFabSpec):
    """Fabrication spec for a disordered photonic glass coating."""
    system: str = "photonic_glass"

    # Particle synthesis
    particle_material: str = "SiO2"
    target_diameter_nm: float = 0.0
    stober_TEOS_mL: float = 0.0
    stober_NH3_mL: float = 0.0
    stober_EtOH_mL: float = 0.0
    stober_H2O_mL: float = 0.0
    stober_temperature_C: float = 25.0
    expected_PDI: float = 0.05

    # Absorber
    absorber_type: str = "carbon_black"
    absorber_fraction: float = 0.0
    absorber_rationale: str = ""

    # Coating
    substrate: str = "glass"
    coating_method: str = "drop_cast"
    suspension_wt_pct: float = 5.0
    film_thickness_um: float = 10.0
    drying_conditions: str = ""

    # Underlayer
    underlayer: str = "none"
    underlayer_thickness_nm: float = 0.0
    underlayer_rationale: str = ""

    # Click handles (if shell-functionalized)
    click_chemistry: str = "none"
    shell_description: str = ""


@dataclass
class BraggOpalFabSpec(OpticalFabSpec):
    """Fabrication spec for an ordered FCC opal film."""
    system: str = "bragg_opal"

    # Same Stoeber particles
    particle_material: str = "SiO2"
    target_diameter_nm: float = 0.0
    required_PDI: float = 0.05   # must be <5% for ordering

    # Vertical deposition
    suspension_wt_pct: float = 0.5
    solvent: str = "ethanol"
    deposition_temperature_C: float = 60.0
    deposition_time_h: float = 24.0
    substrate: str = "glass"
    substrate_orientation: str = "vertical"

    # Properties
    expected_peak_nm: float = 0.0
    angular_shift_formula: str = ""

    # Stoeber params (inherited from same synthesis)
    stober_TEOS_mL: float = 0.0
    stober_NH3_mL: float = 0.0
    stober_EtOH_mL: float = 0.0


@dataclass
class TMMMultilayerFabSpec(OpticalFabSpec):
    """Fabrication spec for a vacuum-deposited multilayer stack."""
    system: str = "tmm_multilayer"

    # Layer stack
    layer_materials: list = field(default_factory=list)
    layer_thicknesses_nm: list = field(default_factory=list)
    n_layers: int = 0
    total_thickness_nm: float = 0.0

    # Deposition
    deposition_method: str = "electron_beam_evaporation"
    substrate: str = "glass"
    chamber_pressure_torr: float = 1e-6
    deposition_rate_nm_s: float = 0.1

    # Design
    design_type: str = ""  # "quarter_wave", "rugate", "custom"
    stopband_center_nm: float = 0.0
    stopband_width_nm: float = 0.0


# =====================================================================
# STOEBER SYNTHESIS DATABASE (Bogush, Tracy & Zukoski 1988)
# =====================================================================

# Lookup table: target diameter -> approximate reagent ratios
# Conditions: 25 deg C, 200 mL total batch
# Format: (D_nm, TEOS_mL, NH3_mL, EtOH_mL, H2O_mL)
STOBER_CONDITIONS = [
    (50,   2.0, 2.0, 190.0, 6.0),
    (100,  3.0, 4.0, 185.0, 8.0),
    (150,  4.5, 6.0, 180.0, 9.5),
    (200,  6.0, 8.0, 175.0, 11.0),
    (250,  8.0, 10.0, 170.0, 12.0),
    (300,  10.0, 13.0, 163.0, 14.0),
    (400,  14.0, 18.0, 150.0, 18.0),
    (500,  18.0, 22.0, 140.0, 20.0),
    (600,  22.0, 27.0, 128.0, 23.0),
    (800,  30.0, 35.0, 108.0, 27.0),
]


def _interpolate_stober(target_d_nm):
    """Interpolate Stoeber conditions for a target diameter."""
    diameters = [s[0] for s in STOBER_CONDITIONS]
    if target_d_nm <= diameters[0]:
        return STOBER_CONDITIONS[0][1:]
    if target_d_nm >= diameters[-1]:
        return STOBER_CONDITIONS[-1][1:]
    for i in range(len(diameters) - 1):
        if diameters[i] <= target_d_nm <= diameters[i + 1]:
            f = (target_d_nm - diameters[i]) / (diameters[i + 1] - diameters[i])
            row_a = STOBER_CONDITIONS[i]
            row_b = STOBER_CONDITIONS[i + 1]
            return tuple(
                row_a[j + 1] + f * (row_b[j + 1] - row_a[j + 1])
                for j in range(4)
            )
    return STOBER_CONDITIONS[4][1:]  # fallback: 250 nm


# =====================================================================
# SUPPLIER & COST DATABASE
# =====================================================================

SUPPLIERS = {
    "TEOS": {"supplier": "Sigma-Aldrich", "catalog": "86578", "price_per_L": 45.0,
             "description": "Tetraethyl orthosilicate, 98%"},
    "NH4OH": {"supplier": "Fisher Scientific", "catalog": "A669-500",
              "price_per_L": 25.0, "description": "Ammonium hydroxide, 28-30%"},
    "EtOH": {"supplier": "Fisher Scientific", "catalog": "A962-4",
             "price_per_L": 35.0, "description": "Ethanol, 200 proof, anhydrous"},
    "carbon_black": {"supplier": "Sigma-Aldrich", "catalog": "633100",
                     "price_per_100g": 30.0, "description": "Carbon black, <100 nm"},
    "glass_slides": {"supplier": "Fisher Scientific", "catalog": "12-550-A3",
                     "price_per_box": 15.0, "description": "Glass microscope slides, 75x25 mm"},
    "glass_tiles": {"supplier": "local hardware", "catalog": "n/a",
                    "price_each": 2.0, "description": "5x5 cm glass tiles"},
    "black_fabric": {"supplier": "fabric store", "catalog": "n/a",
                     "price_per_m": 10.0, "description": "Black polyester satin"},
    "TiO2_target": {"supplier": "Kurt J. Lesker", "catalog": "EJTIO2",
                    "price_each": 120.0, "description": "TiO2 e-beam evaporation source"},
    "SiO2_target": {"supplier": "Kurt J. Lesker", "catalog": "EJSIO2",
                    "price_each": 80.0, "description": "SiO2 e-beam evaporation source"},
}


def _estimate_cost_photonic_glass(d_nm, substrate, area_cm2=25.0):
    """Estimate materials cost for one batch of photonic glass coating."""
    teos, nh3, etoh, h2o = _interpolate_stober(d_nm)
    cost = (teos / 1000 * SUPPLIERS["TEOS"]["price_per_L"]
            + nh3 / 1000 * SUPPLIERS["NH4OH"]["price_per_L"]
            + etoh / 1000 * SUPPLIERS["EtOH"]["price_per_L"])
    cost += 5.0  # carbon black (small amount)
    if "glass" in substrate:
        cost += SUPPLIERS["glass_tiles"]["price_each"]
    elif "fabric" in substrate or "textile" in substrate:
        cost += SUPPLIERS["black_fabric"]["price_per_m"] * 0.1
    return round(cost, 2)


def _estimate_cost_tmm(n_layers):
    """Estimate cost for a TMM multilayer deposition run."""
    source_cost = 200.0  # 1-2 evaporation sources
    facility_time = 100.0 * (n_layers / 4)  # ~$100/hour facility
    substrate_cost = 5.0
    return round(source_cost + facility_time + substrate_cost, 2)


# =====================================================================
# SCORING
# =====================================================================

@dataclass
class OpticalRealizationScore:
    """Score for one optical material system against a FieldInteractionSpec."""
    system: str
    physics_fidelity: float         # 0-1: how close to target color
    angular_match: float            # 0-1: iridescent vs non-iridescent match
    fabrication_accessibility: float # 0-1: lab complexity
    cost_score: float               # 0-1: inverted cost
    scalability: float              # 0-1: m^2 production potential
    durability: float               # 0-1: mechanical/environmental stability
    gamut_coverage: float           # 0-1: can this system reach the target color
    composite: float = 0.0
    feasible: bool = True
    infeasibility_reason: str = ""
    advantages: list = field(default_factory=list)
    limitations: list = field(default_factory=list)


def _composite_score(s: OpticalRealizationScore,
                     application: ApplicationContext) -> float:
    """Application-weighted composite."""
    if application == ApplicationContext.RESEARCH:
        w = (0.30, 0.20, 0.15, 0.10, 0.05, 0.05, 0.15)
    elif application == ApplicationContext.REMEDIATION:
        # "remediation" = industrial/textile application
        w = (0.15, 0.15, 0.10, 0.25, 0.20, 0.10, 0.05)
    else:
        w = (0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.10)
    axes = (s.physics_fidelity, s.angular_match, s.fabrication_accessibility,
            s.cost_score, s.scalability, s.durability, s.gamut_coverage)
    return sum(a * b for a, b in zip(w, axes))


# =====================================================================
# ADAPTERS
# =====================================================================

class PhotonicGlassAdapter:
    """
    Disordered colloidal photonic glass on substrate.

    Non-iridescent, angle-independent structural color.
    Stoeber SiO2 particles + carbon black absorber + substrate.
    Coating: drop-cast, spray, or dip-coat.
    """

    system_id = "photonic_glass"

    def estimate_fidelity(self, spec: FieldInteractionSpec) -> OpticalRealizationScore:
        # Angular match: non-iridescent is this system's strength
        if spec.angular_behavior == AngularBehavior.NON_IRIDESCENT:
            angular = 1.0
        elif spec.angular_behavior == AngularBehavior.ISOTROPIC:
            angular = 0.8
        else:
            angular = 0.2  # wrong system for iridescent

        # Gamut: blue-green easy, red is the problem
        gamut = 1.0
        if spec.target_wavelength_nm is not None:
            wl = spec.target_wavelength_nm
            if wl < 500:
                gamut = 0.95   # blue: excellent
            elif wl < 570:
                gamut = 0.80   # green: good but broader
            elif wl < 620:
                gamut = 0.40   # yellow-orange: Mie bias starts
            else:
                gamut = 0.15   # red: the red problem

        # Physics: heuristic based on gamut position
        # (actual forward model runs in design(), not here — must be fast)
        if gamut > 0.8:
            physics = 0.80
        elif gamut > 0.5:
            physics = 0.55
        else:
            physics = 0.25

        feasible = gamut > 0.1
        advantages = ["Non-iridescent — angle-independent color",
                      "Stoeber SiO2 is commodity chemistry",
                      "Drop-cast or spray — no vacuum equipment"]
        limitations = []
        if gamut < 0.5:
            limitations.append("Red problem: Mie resonance suppresses long-wavelength color")
        if angular < 0.5:
            limitations.append("Cannot produce iridescent/directional color")

        score = OpticalRealizationScore(
            system="photonic_glass",
            physics_fidelity=physics,
            angular_match=angular,
            fabrication_accessibility=0.90,
            cost_score=0.85,
            scalability=0.70,
            durability=0.50,  # fragile without binder
            gamut_coverage=gamut,
            feasible=feasible,
            infeasibility_reason="" if feasible else "Target color outside accessible gamut",
            advantages=advantages,
            limitations=limitations,
        )
        score.composite = _composite_score(score, spec.target_application)
        return score

    def design(self, spec: FieldInteractionSpec) -> PhotonicGlassFabSpec:
        """Full fabrication design from FieldInteractionSpec."""
        # Run inverse design to get particle spec
        from optical.inverse_design import inverse_design_photonic_glass

        target_x = spec.target_x or 0.15
        target_y = spec.target_y or 0.10
        result = inverse_design_photonic_glass(
            target_x, target_y, sphere_material="SiO2",
        )
        design = result.design
        d_nm = design.diameter_nm
        absorber_frac = design.absorber_fraction

        # Stoeber conditions
        teos, nh3, etoh, h2o = _interpolate_stober(d_nm)

        # Substrate
        substrate = spec.substrate or "glass"

        # Underlayer
        underlayer = design.underlayer
        under_thick = 0.0
        under_rationale = "None"
        if underlayer and underlayer != "none":
            under_thick = 50.0  # nm, typical primer
            under_rationale = (f"{underlayer} underlayer: selective spectral recycling "
                               "of transmitted light (Module 8 prediction)")

        # Absorber rationale
        if absorber_frac > 0:
            abs_rationale = (f"{absorber_frac:.1%} carbon black: suppresses incoherent "
                             "backscattering for color saturation")
        else:
            abs_rationale = "No absorber needed"

        # Cost
        cost = _estimate_cost_photonic_glass(d_nm, substrate)

        # Synthesis steps
        steps = [
            f"1. Stoeber synthesis: {teos:.1f} mL TEOS + {nh3:.1f} mL NH4OH "
            f"+ {etoh:.1f} mL EtOH + {h2o:.1f} mL H2O, stir 25 deg C, 12h",
            "2. Centrifuge 3x at 5000 rpm, redisperse in EtOH (wash cycle)",
            "3. DLS measurement: confirm diameter and PDI < 10%",
        ]
        if absorber_frac > 0:
            steps.append(
                f"4. Add carbon black ({absorber_frac:.1%} v/v) to suspension, sonicate 30 min"
            )
        if underlayer and underlayer != "none":
            steps.append(
                f"5. Prepare underlayer: spin-coat {underlayer} at {under_thick:.0f} nm "
                f"on {substrate}, cure"
            )
        step_n = len(steps) + 1
        steps.append(
            f"{step_n}. Coat substrate: drop-cast 50 uL/cm2 of 5 wt% suspension "
            f"onto {substrate}, dry covered at RT 12h"
        )
        steps.append(f"{step_n + 1}. Photograph under D65 illumination with color card")
        steps.append(
            f"{step_n + 2}. UV-Vis reflectance: compare measured spectrum to "
            f"MABE prediction (target CIE x={target_x:.3f}, y={target_y:.3f})"
        )

        # Validation
        validation = [
            f"DLS: diameter = {d_nm:.0f} +/- {d_nm * 0.05:.0f} nm, PDI < 0.10",
            f"UV-Vis reflectance: peak within +/-20 nm of prediction",
            f"CIE chromaticity: delta_E < 10 from target ({target_x:.3f}, {target_y:.3f})",
            "Photograph: visually distinct color under D65 illumination",
        ]

        # Materials list
        materials = [
            f"TEOS {teos:.1f} mL — {SUPPLIERS['TEOS']['supplier']} #{SUPPLIERS['TEOS']['catalog']}",
            f"NH4OH {nh3:.1f} mL — {SUPPLIERS['NH4OH']['supplier']}",
            f"EtOH {etoh:.1f} mL — {SUPPLIERS['EtOH']['supplier']}",
        ]
        if absorber_frac > 0:
            materials.append(f"Carbon black — {SUPPLIERS['carbon_black']['supplier']}")
        materials.append(f"Substrate: {substrate}")

        return PhotonicGlassFabSpec(
            target_wavelength_nm=spec.target_wavelength_nm,
            target_xy=(target_x, target_y),
            predicted_xy=(design.cie_x, design.cie_y),
            predicted_delta_E=result.delta_E,
            synthesis_steps=steps,
            materials_list=materials,
            estimated_cost_usd=cost,
            estimated_time="2-3 days (synthesis + coating + drying)",
            validation_plan=validation,
            particle_material="SiO2",
            target_diameter_nm=d_nm,
            stober_TEOS_mL=teos,
            stober_NH3_mL=nh3,
            stober_EtOH_mL=etoh,
            stober_H2O_mL=h2o,
            absorber_type="carbon_black" if absorber_frac > 0 else "none",
            absorber_fraction=absorber_frac,
            absorber_rationale=abs_rationale,
            substrate=substrate,
            coating_method="drop_cast",
            suspension_wt_pct=5.0,
            drying_conditions="Covered, room temperature, 12 hours",
            underlayer=underlayer,
            underlayer_thickness_nm=under_thick,
            underlayer_rationale=under_rationale,
        )


class BraggOpalAdapter:
    """
    Ordered FCC colloidal crystal (opal) on substrate.

    Iridescent, angle-dependent structural color.
    Same Stoeber particles as photonic glass, different deposition.
    Vertical evaporative deposition (Jiang & Bertone 1999).
    """

    system_id = "bragg_opal"

    def estimate_fidelity(self, spec: FieldInteractionSpec) -> OpticalRealizationScore:
        # Angular match: iridescent is this system's strength
        if spec.angular_behavior == AngularBehavior.IRIDESCENT:
            angular = 1.0
        elif spec.angular_behavior == AngularBehavior.DIRECTIONAL:
            angular = 0.7
        else:
            angular = 0.15  # wrong system for non-iridescent

        # Gamut: Bragg covers full visible (no red problem!)
        gamut = 0.90
        if spec.target_wavelength_nm is not None:
            wl = spec.target_wavelength_nm
            # Need D such that 1.633*D*n_eff = wl
            # n_eff ~ 1.35 for SiO2 in air at phi=0.74
            d_needed = wl / (1.633 * 1.35)
            if 30 < d_needed < 500:  # Stoeber range
                gamut = 0.95
            else:
                gamut = 0.3

        physics = 0.85  # Bragg prediction is very accurate (zero free params)

        feasible = angular > 0.1 and gamut > 0.2
        advantages = ["Full visible spectrum — no red problem",
                      "Sharp spectral peak — vivid color",
                      "Zero-parameter Bragg prediction (1.633*D*n_eff)"]
        limitations = ["Iridescent — color changes with viewing angle",
                       "Requires PDI < 5% for ordering",
                       "Slow deposition (24-48h)",
                       "Fragile — cracks easily"]
        if angular < 0.3:
            limitations.insert(0, "Cannot produce angle-independent color")

        score = OpticalRealizationScore(
            system="bragg_opal",
            physics_fidelity=physics,
            angular_match=angular,
            fabrication_accessibility=0.70,  # needs controlled evaporation
            cost_score=0.80,
            scalability=0.30,  # cm-scale, slow
            durability=0.30,   # fragile
            gamut_coverage=gamut,
            feasible=feasible,
            infeasibility_reason="" if feasible else "Angular behavior mismatch or diameter out of range",
            advantages=advantages,
            limitations=limitations,
        )
        score.composite = _composite_score(score, spec.target_application)
        return score

    def design(self, spec: FieldInteractionSpec) -> BraggOpalFabSpec:
        """Full fabrication design for Bragg opal."""
        from optical.bragg_opal import bragg_opal

        # Target wavelength
        target_wl = spec.target_wavelength_nm
        if target_wl is None and spec.target_x is not None:
            # Rough: map CIE xy to dominant wavelength
            target_wl = 470.0  # default blue

        # Solve for diameter: wl = 1.633 * D * n_eff
        n_eff_approx = 1.35
        d_nm = target_wl / (1.633 * n_eff_approx)

        # Verify with forward model
        predicted_peak = bragg_opal(d_nm, material="SiO2")

        # Stoeber conditions
        teos, nh3, etoh, h2o = _interpolate_stober(d_nm)

        substrate = spec.substrate or "glass"
        cost = _estimate_cost_photonic_glass(d_nm, substrate) + 5.0  # extra for slow deposition

        steps = [
            f"1. Stoeber synthesis: {teos:.1f} mL TEOS + {nh3:.1f} mL NH4OH "
            f"+ {etoh:.1f} mL EtOH + {h2o:.1f} mL H2O, stir 25 deg C, 12h",
            "2. Centrifuge 3x at 5000 rpm, redisperse in EtOH",
            "3. DLS: confirm diameter and PDI < 5% (CRITICAL for opal ordering)",
            f"4. Dilute to 0.5 wt% in EtOH",
            f"5. Vertical deposition: stand clean {substrate} slide upright "
            "in beaker of suspension",
            "6. Place in 60 deg C oven, evaporate 24-48h undisturbed",
            "7. Remove slide — opal film on lower portion",
            "8. Photograph at 0, 15, 30, 45 deg tilt to show iridescence",
            f"9. UV-Vis reflectance at normal incidence: "
            f"expect peak at {predicted_peak:.0f} nm",
        ]

        validation = [
            f"DLS: diameter = {d_nm:.0f} +/- {d_nm * 0.03:.0f} nm, PDI < 0.05",
            f"Reflectance peak at {predicted_peak:.0f} +/- 10 nm",
            "Peak blue-shifts with increasing angle (Bragg law cos(theta))",
            "Iridescent shimmer visible by eye when tilted",
        ]

        return BraggOpalFabSpec(
            target_wavelength_nm=target_wl,
            target_xy=(spec.target_x, spec.target_y) if spec.target_x else None,
            predicted_delta_E=abs(predicted_peak - target_wl) if target_wl else 999,
            synthesis_steps=steps,
            materials_list=[
                f"TEOS {teos:.1f} mL", f"NH4OH {nh3:.1f} mL",
                f"EtOH {etoh:.1f} mL", f"Substrate: {substrate}",
            ],
            estimated_cost_usd=cost,
            estimated_time="3-4 days (synthesis 12h + deposition 24-48h)",
            validation_plan=validation,
            particle_material="SiO2",
            target_diameter_nm=d_nm,
            required_PDI=0.05,
            suspension_wt_pct=0.5,
            deposition_temperature_C=60.0,
            deposition_time_h=36.0,
            substrate=substrate,
            expected_peak_nm=predicted_peak,
            angular_shift_formula="lambda(theta) = lambda_peak * cos(theta)",
            stober_TEOS_mL=teos,
            stober_NH3_mL=nh3,
            stober_EtOH_mL=etoh,
        )


class TMMMultilayerAdapter:
    """
    Vacuum-deposited thin-film multilayer (1D photonic crystal).

    Precise spectral engineering. Requires vacuum deposition equipment.
    """

    system_id = "tmm_multilayer"

    def estimate_fidelity(self, spec: FieldInteractionSpec) -> OpticalRealizationScore:
        # Angular: directional is this system's sweet spot
        if spec.angular_behavior == AngularBehavior.DIRECTIONAL:
            angular = 1.0
        elif spec.angular_behavior == AngularBehavior.IRIDESCENT:
            angular = 0.7   # TMM is iridescent too but less dramatically
        elif spec.angular_behavior == AngularBehavior.NON_IRIDESCENT:
            angular = 0.3   # can design wide-angle but not fully isotropic
        else:
            angular = 0.5

        # Gamut: full spectrum, no red problem
        gamut = 0.95

        # Physics: TMM is exact for ideal films
        physics = 0.90

        # But expensive and needs vacuum
        feasible = True
        advantages = ["Exact spectral control — any wavelength, any bandwidth",
                      "Full gamut including red",
                      "Industrial-scale process (established in optics)"]
        limitations = ["Requires vacuum deposition equipment",
                       "Higher cost per unit area",
                       "Iridescent at non-normal incidence"]

        score = OpticalRealizationScore(
            system="tmm_multilayer",
            physics_fidelity=physics,
            angular_match=angular,
            fabrication_accessibility=0.30,  # needs vacuum chamber
            cost_score=0.30,
            scalability=0.85,  # standard industrial process
            durability=0.90,   # oxide films are robust
            gamut_coverage=gamut,
            feasible=feasible,
            advantages=advantages,
            limitations=limitations,
        )
        score.composite = _composite_score(score, spec.target_application)
        return score

    def design(self, spec: FieldInteractionSpec) -> TMMMultilayerFabSpec:
        """Design a quarter-wave multilayer for the target wavelength."""
        from optical.tmm import quarter_wave_thickness

        target_wl = spec.target_wavelength_nm or 550.0
        substrate = spec.substrate or "glass"

        # Quarter-wave stack: [H L]^N where H=TiO2_rutile, L=SiO2
        n_pairs = 5
        mat_H = "TiO2_rutile"
        mat_L = "SiO2"
        d_H = quarter_wave_thickness(mat_H, target_wl)
        d_L = quarter_wave_thickness(mat_L, target_wl)

        layer_mats = []
        layer_thick = []
        for _ in range(n_pairs):
            layer_mats.extend([mat_H, mat_L])
            layer_thick.extend([d_H, d_L])

        total = sum(layer_thick)

        cost = _estimate_cost_tmm(2 * n_pairs)

        steps = [
            f"1. Clean {substrate} substrate: piranha or UV-ozone, 15 min",
            f"2. Load {substrate} into e-beam evaporator, pump to <1e-6 Torr",
        ]
        for i in range(2 * n_pairs):
            mat = layer_mats[i]
            d = layer_thick[i]
            steps.append(
                f"3.{i + 1}. Deposit {mat} at {d:.1f} nm "
                f"(rate: 0.1 nm/s, QCM monitored)"
            )
        steps.append(f"4. Vent chamber, remove sample")
        steps.append(f"5. UV-Vis reflectance: expect stopband at {target_wl:.0f} nm")

        validation = [
            f"Reflectance peak centered at {target_wl:.0f} +/- 5 nm",
            "Peak reflectance > 90% for 5-pair stack",
            "R + T = 1.0 (energy conservation, lossless films)",
            f"Total thickness {total:.0f} nm measurable by profilometry",
        ]

        return TMMMultilayerFabSpec(
            target_wavelength_nm=target_wl,
            target_xy=(spec.target_x, spec.target_y) if spec.target_x else None,
            synthesis_steps=steps,
            materials_list=[
                f"TiO2 evaporation source — {SUPPLIERS['TiO2_target']['supplier']}",
                f"SiO2 evaporation source — {SUPPLIERS['SiO2_target']['supplier']}",
                f"Substrate: {substrate}",
            ],
            estimated_cost_usd=cost,
            estimated_time="4-6 hours (deposition) + 1h prep/characterization",
            validation_plan=validation,
            layer_materials=layer_mats,
            layer_thicknesses_nm=layer_thick,
            n_layers=2 * n_pairs,
            total_thickness_nm=total,
            deposition_method="electron_beam_evaporation",
            substrate=substrate,
            design_type="quarter_wave",
            stopband_center_nm=target_wl,
            stopband_width_nm=target_wl * 0.1,  # ~10% for TiO2/SiO2
        )


# =====================================================================
# OPTICAL RANKER
# =====================================================================

ALL_ADAPTERS = [
    PhotonicGlassAdapter(),
    BraggOpalAdapter(),
    TMMMultilayerAdapter(),
]


@dataclass
class OpticalRankerOutput:
    """Ranked optical realization options."""
    spec: FieldInteractionSpec
    scores: list                    # List[OpticalRealizationScore], sorted
    recommended: str
    recommendation_rationale: str
    fab_spec: Optional[OpticalFabSpec] = None  # designed from recommended


def rank_optical_realizations(
    spec: FieldInteractionSpec,
    run_design: bool = False,
) -> OpticalRankerOutput:
    """
    Score all optical adapters against a FieldInteractionSpec.

    If run_design=True, also generates FabSpec from the top-ranked adapter.
    """
    scores = []
    adapters_by_system = {}
    for adapter in ALL_ADAPTERS:
        score = adapter.estimate_fidelity(spec)
        scores.append(score)
        adapters_by_system[score.system] = adapter

    scores.sort(key=lambda s: s.composite, reverse=True)

    feasible = [s for s in scores if s.feasible]
    if feasible:
        best = feasible[0]
        rationale = (f"Recommended: {best.system} "
                     f"(composite={best.composite:.3f}). "
                     f"{'; '.join(best.advantages[:2])}")
    else:
        best = scores[0] if scores else None
        rationale = "No feasible system found for this target."

    fab = None
    if run_design and best and best.feasible:
        adapter = adapters_by_system[best.system]
        try:
            fab = adapter.design(spec)
        except Exception as e:
            rationale += f" Design failed: {e}"

    return OpticalRankerOutput(
        spec=spec,
        scores=scores,
        recommended=best.system if best else "none",
        recommendation_rationale=rationale,
        fab_spec=fab,
    )
