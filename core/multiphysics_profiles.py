"""
core/multiphysics_profiles.py — Application Profiles for Multi-Physics Design

Pre-configured target specifications for real-world applications:
  1. Building: exterior facade panel
  2. Building: roof coating (cool roof)
  3. Building: interior wall / ceiling tile
  4. Wearable: textile coating
  5. Wearable: protective garment
  6. Wearable: smart textile (thermoresponsive)

Each profile sets:
  - Color, thermal, acoustic targets appropriate for the application
  - Form factor constraints (thickness, flexibility, mass)
  - Backing structure (if applicable)
  - Material constraints (fire rating, washability, UV stability)
  - Optimizer parameters (search ranges, weights)

Usage:
  from core.multiphysics_profiles import design_for_application
  result = design_for_application("wall_tile", color="forest_green")

Phase 4 of the Multi-Physics Structural Element module.
Depends on: multiphysics_pareto (Phase 3), colloidal_thermal (Phase 1),
            colloidal_acoustic (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.multiphysics_pareto import (
    optimize_multiphysics,
    MultiPhysicsTarget,
    ColorTarget,
    ThermalTarget,
    AcousticTarget,
    ParetoResult,
    NAMED_COLORS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Application profile dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ApplicationProfile:
    """Complete application profile with targets, constraints, and optimizer config."""

    # Identity
    application_id: str
    name: str
    description: str
    category: str                  # "building" or "wearable"

    # Physics targets
    color_target: Optional[ColorTarget] = None
    thermal_target: Optional[ThermalTarget] = None
    acoustic_target: Optional[AcousticTarget] = None

    # Backing structure
    backing_thickness_mm: float = 0.0
    backing_flow_resistivity: float = 10000.0
    air_gap_mm: float = 0.0

    # Form factor
    max_thickness_mm: float = 100.0
    must_be_flexible: bool = False
    max_mass_kg_m2: Optional[float] = None

    # Constraints
    fire_rating: Optional[str] = None        # EN 13501: "A1", "A2", "B"
    uv_stable: bool = True
    washable: bool = False
    water_resistant: bool = False
    biodegradable: bool = False

    # Optimizer configuration
    diameter_range: tuple[float, float] = (180.0, 380.0)
    diameter_step: float = 20.0
    phi_range: tuple[float, float] = (0.35, 0.60)
    phi_step: float = 0.05
    absorber_fractions: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])
    film_thicknesses_um: list[float] = field(default_factory=lambda: [50.0, 200.0, 1000.0])
    materials: list[str] = field(default_factory=lambda: ["SiO2"])
    matrix_materials: list[str] = field(default_factory=lambda: ["air"])
    weights: tuple[float, float, float] = (0.40, 0.35, 0.25)  # color, thermal, acoustic

    # Recommended materials note
    material_notes: str = ""

    def to_target(self) -> MultiPhysicsTarget:
        """Convert to MultiPhysicsTarget for the optimizer."""
        return MultiPhysicsTarget(
            color=self.color_target,
            thermal=self.thermal_target,
            acoustic=self.acoustic_target,
            max_thickness_mm=self.max_thickness_mm,
            application=self.application_id,
            backing_thickness_mm=self.backing_thickness_mm,
            backing_flow_resistivity=self.backing_flow_resistivity,
            air_gap_mm=self.air_gap_mm,
        )

    def summary(self) -> str:
        lines = [
            f"Application: {self.name}",
            f"  {self.description}",
            f"  Category: {self.category}",
        ]
        if self.color_target:
            lines.append(f"  Color: ({self.color_target.cie_x:.2f}, {self.color_target.cie_y:.2f}) "
                         f"{'non-iridescent' if self.color_target.angle_independent else 'iridescent OK'}")
        if self.thermal_target:
            if self.thermal_target.min_R_value_m2KW:
                lines.append(f"  Thermal: R ≥ {self.thermal_target.min_R_value_m2KW} m²K/W")
            elif self.thermal_target.max_kappa_W_mK:
                lines.append(f"  Thermal: κ ≤ {self.thermal_target.max_kappa_W_mK} W/mK")
        if self.acoustic_target:
            if self.acoustic_target.min_nrc:
                lines.append(f"  Acoustic: NRC ≥ {self.acoustic_target.min_nrc}")
        lines.append(f"  Max thickness: {self.max_thickness_mm} mm")
        if self.must_be_flexible:
            lines.append(f"  Flexible: required")
        if self.fire_rating:
            lines.append(f"  Fire rating: {self.fire_rating}")
        if self.washable:
            lines.append(f"  Washable: required")
        if self.material_notes:
            lines.append(f"  Materials: {self.material_notes}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Profile definitions
# ═══════════════════════════════════════════════════════════════════════════

def _facade_panel(color: Optional[ColorTarget] = None) -> ApplicationProfile:
    """Exterior facade panel: color + high thermal resistance + sound blocking."""
    return ApplicationProfile(
        application_id="facade_panel",
        name="Exterior Facade Panel",
        description=(
            "Structural color facade for commercial/residential buildings. "
            "Replaces painted cladding. Non-iridescent color, high R-value for "
            "energy efficiency, STC for external noise reduction."
        ),
        category="building",
        color_target=color or ColorTarget(0.31, 0.33, "white", 10.0, True),
        thermal_target=ThermalTarget(min_R_value_m2KW=0.01),
        acoustic_target=AcousticTarget(mode="block"),
        backing_thickness_mm=50.0,
        backing_flow_resistivity=12000.0,
        air_gap_mm=25.0,
        max_thickness_mm=150.0,
        fire_rating="A2",
        uv_stable=True,
        water_resistant=True,
        diameter_range=(200.0, 380.0),
        phi_range=(0.40, 0.55),
        absorber_fractions=[0.002, 0.005, 0.01],
        film_thicknesses_um=[500.0, 2000.0, 5000.0],
        materials=["SiO2", "TiO2_rutile"],
        matrix_materials=["silicone"],
        weights=(0.30, 0.35, 0.35),
        material_notes=(
            "SiO₂: fire-resistant, UV-stable, zero VOC. "
            "TiO₂: higher Δn for vivid color + photocatalytic self-cleaning. "
            "Silicone binder: weatherproof, flexible, fire-retardant."
        ),
    )


def _roof_coating(color: Optional[ColorTarget] = None) -> ApplicationProfile:
    """Cool roof coating: high solar reflectance + thermal insulation + rain noise."""
    return ApplicationProfile(
        application_id="roof_coating",
        name="Cool Roof Coating",
        description=(
            "High solar reflectance structural color coating for roofs. "
            "Reduces cooling load by reflecting NIR. Incidental rain noise absorption. "
            "White or light color for maximum reflectance."
        ),
        category="building",
        color_target=color or ColorTarget(0.31, 0.33, "white", 15.0, True),
        thermal_target=ThermalTarget(mode="insulate"),
        acoustic_target=AcousticTarget(mode="absorb"),
        max_thickness_mm=5.0,
        fire_rating="B",
        uv_stable=True,
        water_resistant=True,
        diameter_range=(280.0, 380.0),
        phi_range=(0.30, 0.50),
        absorber_fractions=[0.001, 0.003],
        film_thicknesses_um=[200.0, 500.0, 1000.0],
        materials=["SiO2", "TiO2_rutile"],
        matrix_materials=["silicone", "polyurethane"],
        weights=(0.15, 0.55, 0.30),
        material_notes=(
            "Large particles (>300nm) for blue-shifted peak → broadband visible reflectance. "
            "TiO₂ preferred: high reflectance + photocatalytic self-cleaning. "
            "Low φ for porosity → insulation + rain absorption."
        ),
    )


def _wall_tile(color: Optional[ColorTarget] = None) -> ApplicationProfile:
    """Interior wall/ceiling tile: designer color + moderate thermal + high NRC."""
    return ApplicationProfile(
        application_id="wall_tile",
        name="Interior Wall / Ceiling Tile",
        description=(
            "Structural color replacement for painted drywall or acoustic ceiling tiles. "
            "High NRC for room acoustics. Moderate thermal insulation. "
            "Non-iridescent designer colors. Matte finish."
        ),
        category="building",
        color_target=color or ColorTarget(0.30, 0.52, "green", 10.0, True),
        thermal_target=ThermalTarget(mode="insulate"),
        acoustic_target=AcousticTarget(min_nrc=0.5, mode="absorb"),
        backing_thickness_mm=25.0,
        backing_flow_resistivity=8000.0,
        air_gap_mm=50.0,
        max_thickness_mm=100.0,
        fire_rating="A2",
        diameter_range=(180.0, 360.0),
        phi_range=(0.35, 0.55),
        absorber_fractions=[0.003, 0.005, 0.01],
        film_thicknesses_um=[100.0, 500.0, 2000.0],
        materials=["SiO2"],
        matrix_materials=["air", "PVA"],
        weights=(0.40, 0.20, 0.40),
        material_notes=(
            "Colloidal surface provides color (replaces paint). "
            "Open-cell porous substrate behind colloidal layer for acoustic absorption. "
            "Air-filled voids in colloidal layer contribute to both thermal and acoustic."
        ),
    )


def _textile_coating(color: Optional[ColorTarget] = None) -> ApplicationProfile:
    """Textile coating: vivid color + breathability + wash durability."""
    return ApplicationProfile(
        application_id="textile_coating",
        name="Textile Coating",
        description=(
            "Structural color coating for fabrics. Replaces chemical dyes. "
            "Must survive washing, bending, stretching. Breathable for comfort. "
            "Non-iridescent, vivid, and fast."
        ),
        category="wearable",
        color_target=color or ColorTarget(0.15, 0.06, "blue", 10.0, True),
        thermal_target=ThermalTarget(mode="insulate"),
        acoustic_target=None,
        max_thickness_mm=0.5,
        must_be_flexible=True,
        max_mass_kg_m2=0.05,
        washable=True,
        diameter_range=(180.0, 360.0),
        phi_range=(0.40, 0.55),
        absorber_fractions=[0.003, 0.005, 0.01],
        film_thicknesses_um=[10.0, 30.0, 50.0],
        materials=["SiO2", "polystyrene"],
        matrix_materials=["polyurethane"],
        weights=(0.60, 0.15, 0.05),
        material_notes=(
            "SiO₂ in polyurethane binder: wash-resistant (Manchester group, 2017). "
            "PS spheres: lighter, cheaper, but less UV-stable. "
            "Polyurethane binder gives flexibility + abrasion resistance. "
            "Thin films (<50µm) for drape and breathability."
        ),
    )


def _protective_garment(color: Optional[ColorTarget] = None) -> ApplicationProfile:
    """Protective garment: thermal barrier + high-vis color + optional acoustic."""
    return ApplicationProfile(
        application_id="protective_garment",
        name="Protective Garment",
        description=(
            "Structural color for firefighter/industrial protective clothing. "
            "High-visibility color (yellow/orange) or camo. Thermal barrier to "
            "protect wearer from external heat. Acoustic damping if integrated "
            "into helmet/headgear."
        ),
        category="wearable",
        color_target=color or ColorTarget(0.50, 0.44, "orange", 12.0, True),
        thermal_target=ThermalTarget(max_kappa_W_mK=0.1),
        acoustic_target=AcousticTarget(mode="damp"),
        max_thickness_mm=5.0,
        must_be_flexible=True,
        max_mass_kg_m2=0.5,
        diameter_range=(200.0, 380.0),
        phi_range=(0.40, 0.55),
        absorber_fractions=[0.001, 0.005],
        film_thicknesses_um=[100.0, 500.0, 1000.0],
        materials=["SiO2", "TiO2_rutile"],
        matrix_materials=["silicone", "polyurethane"],
        weights=(0.35, 0.45, 0.20),
        material_notes=(
            "Core-shell particles (SiO₂ core + silicone shell) for double Kapitza "
            "barriers → thermal insulation. TiO₂ for high-vis orange/yellow. "
            "Graded colloidal film with aerogel-like porosity for extreme insulation."
        ),
    )


def _smart_textile(color: Optional[ColorTarget] = None) -> ApplicationProfile:
    """Thermoresponsive smart textile: color changes with temperature."""
    return ApplicationProfile(
        application_id="smart_textile",
        name="Smart Textile (Thermoresponsive)",
        description=(
            "PNIPAM hydrogel matrix: temperature changes inter-particle spacing, "
            "simultaneously shifting color, porosity, and thermal conductivity. "
            "Below 32°C: swollen → red-shifted → insulating. "
            "Above 32°C: collapsed → blue-shifted → conducting (releases heat). "
            "One stimulus, three coupled responses."
        ),
        category="wearable",
        color_target=color or ColorTarget(0.30, 0.52, "green", 15.0, True),
        thermal_target=ThermalTarget(mode="insulate"),
        acoustic_target=None,
        max_thickness_mm=2.0,
        must_be_flexible=True,
        max_mass_kg_m2=0.1,
        water_resistant=False,
        diameter_range=(200.0, 340.0),
        phi_range=(0.30, 0.50),
        phi_step=0.05,
        absorber_fractions=[0.003, 0.005],
        film_thicknesses_um=[30.0, 100.0, 300.0],
        materials=["SiO2", "polystyrene"],
        matrix_materials=["PNIPAM_swollen", "PNIPAM_collapsed"],
        weights=(0.45, 0.35, 0.05),
        material_notes=(
            "PNIPAM LCST = 32°C. Below: hydrogel swollen, D_eff increases → "
            "red shift + high porosity (insulating). Above: collapsed, D_eff "
            "decreases → blue shift + low porosity (conducting). "
            "Both states evaluated for Pareto optimization."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Profile registry
# ═══════════════════════════════════════════════════════════════════════════

_PROFILES = {
    "facade_panel": _facade_panel,
    "roof_coating": _roof_coating,
    "wall_tile": _wall_tile,
    "textile_coating": _textile_coating,
    "protective_garment": _protective_garment,
    "smart_textile": _smart_textile,
}


def list_applications() -> list[str]:
    """Return available application IDs."""
    return list(_PROFILES.keys())


def get_profile(
    application_id: str,
    color_name: Optional[str] = None,
    color_xy: Optional[tuple[float, float]] = None,
) -> ApplicationProfile:
    """Get an application profile, optionally overriding the color target.

    Args:
        application_id: key from list_applications()
        color_name: named color ("green", "blue", "red", etc.)
        color_xy: explicit CIE xy coordinates (overrides color_name)
    """
    if application_id not in _PROFILES:
        raise ValueError(f"Unknown application: {application_id}. "
                         f"Available: {list_applications()}")

    # Build color target
    color_target = None
    if color_xy:
        color_target = ColorTarget(cie_x=color_xy[0], cie_y=color_xy[1],
                                   name="custom", max_delta_e=10.0, angle_independent=True)
    elif color_name:
        xy = NAMED_COLORS.get(color_name.lower())
        if xy:
            color_target = ColorTarget(cie_x=xy[0], cie_y=xy[1],
                                       name=color_name, max_delta_e=10.0, angle_independent=True)

    return _PROFILES[application_id](color_target)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def design_for_application(
    application_id: str,
    color: Optional[str] = None,
    color_xy: Optional[tuple[float, float]] = None,
    verbose: bool = False,
) -> ParetoResult:
    """Design a multi-physics structural element for a specific application.

    Args:
        application_id: "facade_panel", "roof_coating", "wall_tile",
                        "textile_coating", "protective_garment", "smart_textile"
        color: named color ("green", "blue", "red", "forest_green", etc.)
        color_xy: explicit CIE (x, y) chromaticity (overrides color)
        verbose: if True, print profile summary before optimizing

    Returns:
        ParetoResult with application-optimized designs.

    Example:
        result = design_for_application("wall_tile", color="forest_green")
        print(result.summary())
    """
    profile = get_profile(application_id, color_name=color, color_xy=color_xy)

    if verbose:
        print(profile.summary())
        print()

    target = profile.to_target()

    return optimize_multiphysics(
        target,
        diameter_range=profile.diameter_range,
        diameter_step=profile.diameter_step,
        phi_range=profile.phi_range,
        phi_step=profile.phi_step,
        absorber_fractions=profile.absorber_fractions,
        film_thicknesses_um=profile.film_thicknesses_um,
        materials=profile.materials,
        matrix_materials=profile.matrix_materials,
        weights=profile.weights,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Compare across applications
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ApplicationComparison:
    """Side-by-side comparison of same color across different applications."""
    color_name: str
    results: dict[str, ParetoResult]

    def summary(self) -> str:
        lines = [f"Multi-Physics Design Comparison: {self.color_name}", ""]
        for app_id, result in self.results.items():
            profile = get_profile(app_id, self.color_name)
            lines.append(f"  {profile.name}:")
            if result.best_balanced:
                dp = result.best_balanced
                lines.append(f"    Best: D={dp.particle_diameter_nm:.0f}nm "
                             f"φ={dp.volume_fraction:.2f} t={dp.film_thickness_um:.0f}µm")
                if dp.color.predicted:
                    lines.append(f"    Color ΔE*={dp.obj_color_delta_e:.1f}")
                if dp.thermal:
                    lines.append(f"    Thermal κ={dp.thermal.kappa_eff_W_mK:.4f} "
                                 f"R={dp.thermal.R_value_m2KW:.4f}")
                if dp.acoustic:
                    lines.append(f"    Acoustic NRC={dp.acoustic.nrc:.2f}")
            else:
                lines.append(f"    No viable design found")
            lines.append(f"    Pareto front: {len(result.pareto_front)} designs")
            lines.append("")
        return "\n".join(lines)


def compare_applications(
    color: str = "green",
    applications: Optional[list[str]] = None,
) -> ApplicationComparison:
    """Compare the same color target across multiple applications.

    Shows how the optimal design changes depending on whether you're
    making a facade panel vs a textile vs a wall tile.
    """
    if applications is None:
        applications = list_applications()

    results = {}
    for app_id in applications:
        results[app_id] = design_for_application(app_id, color=color)

    return ApplicationComparison(color_name=color, results=results)
