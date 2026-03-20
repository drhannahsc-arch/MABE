"""
core/multiphysics_pareto.py — Multi-Objective Pareto Optimizer

Wraps three forward models with shared design parameters:
  1. Optical: photonic glass reflectance → CIE color → ΔE from target
  2. Thermal: Hasselman-Johnson with Kapitza → κ_eff → R-value
  3. Acoustic: JCA porous media → α(f) → NRC

Shared parameters varied: D (nm), φ, absorber fraction, film thickness,
material, matrix, air gap.

Returns Pareto-optimal designs: no design is dominated on all three
objectives simultaneously.

Phase 3 of the Multi-Physics Structural Element module.

Dependencies:
  - core/colloidal_thermal.py (Phase 1)
  - core/colloidal_acoustic.py (Phase 2)
  - optical/* (photonic_glass, cie_color) — soft dependency, graceful fallback
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.colloidal_thermal import (
    predict_from_optical as thermal_predict,
    predict_core_shell as thermal_predict_cs,
    ThermalResult,
)
from core.colloidal_acoustic import (
    predict_from_optical as acoustic_predict,
    predict_with_backing as acoustic_predict_backing,
    AcousticResult,
)

# Optical pipeline — soft dependency
try:
    import numpy as np
    from optical.photonic_glass import (
        photonic_glass_reflectance,
        photonic_glass_peak_wavelength,
    )
    from optical.cie_color import (
        spectrum_to_XYZ,
        XYZ_to_Lab,
        XYZ_to_xyY,
        cie_delta_E,
    )
    _OPTICAL_AVAILABLE = True
except ImportError:
    _OPTICAL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColorTarget:
    """Target color specification."""
    cie_x: float                       # CIE 1931 x chromaticity
    cie_y: float                       # CIE 1931 y chromaticity
    name: str = ""                     # "forest green", "sky blue", etc.
    max_delta_e: float = 10.0          # acceptable ΔE* from target
    angle_independent: bool = True     # non-iridescent required


@dataclass
class ThermalTarget:
    """Target thermal performance."""
    min_R_value_m2KW: Optional[float] = None    # minimum R-value
    max_kappa_W_mK: Optional[float] = None      # maximum thermal conductivity
    mode: str = "insulate"                       # "insulate" or "conduct"


@dataclass
class AcousticTarget:
    """Target acoustic performance."""
    min_nrc: Optional[float] = None              # minimum NRC
    min_alpha_500: Optional[float] = None        # minimum α at 500 Hz
    mode: str = "absorb"                         # "absorb" or "block"


@dataclass
class MultiPhysicsTarget:
    """Combined multi-physics target specification."""
    color: Optional[ColorTarget] = None
    thermal: Optional[ThermalTarget] = None
    acoustic: Optional[AcousticTarget] = None

    # Form factor constraints
    max_thickness_mm: float = 100.0
    application: str = "building_panel"

    # Backing (for acoustic)
    backing_thickness_mm: float = 0.0
    backing_flow_resistivity: float = 10000.0
    air_gap_mm: float = 0.0


@dataclass
class ColorResult:
    """Color prediction for one design point."""
    peak_wavelength_nm: float = 0.0
    cie_x: float = 0.0
    cie_y: float = 0.0
    Lab: tuple = (0.0, 0.0, 0.0)
    delta_e: float = 999.0              # ΔE* from target
    predicted: bool = False             # True if optical model ran


@dataclass
class DesignPoint:
    """One evaluated design in the multi-objective space."""
    # Design parameters
    particle_diameter_nm: float
    particle_material: str
    volume_fraction: float
    absorber_fraction: float
    film_thickness_um: float
    matrix_material: str

    # Predictions
    color: ColorResult = field(default_factory=ColorResult)
    thermal: Optional[ThermalResult] = None
    acoustic: Optional[AcousticResult] = None

    # Objective values (lower is better for all — minimization)
    obj_color_delta_e: float = 999.0     # ΔE* from target color
    obj_thermal_neg_R: float = 0.0       # negative R-value (minimize = maximize R)
    obj_acoustic_neg_nrc: float = 0.0    # negative NRC (minimize = maximize NRC)

    # Pareto status
    is_pareto: bool = False
    dominated_by: int = 0                # count of designs that dominate this one

    def summary(self) -> str:
        lines = [
            f"D={self.particle_diameter_nm:.0f}nm {self.particle_material} "
            f"φ={self.volume_fraction:.2f} abs={self.absorber_fraction:.3f} "
            f"t={self.film_thickness_um:.0f}µm",
        ]
        if self.color.predicted:
            lines.append(f"  Color: ΔE*={self.obj_color_delta_e:.1f} "
                         f"(λ_peak={self.color.peak_wavelength_nm:.0f}nm, "
                         f"xy=({self.color.cie_x:.3f},{self.color.cie_y:.3f}))")
        if self.thermal:
            lines.append(f"  Thermal: κ={self.thermal.kappa_eff_W_mK:.4f} W/mK, "
                         f"R={self.thermal.R_value_m2KW:.4f} m²K/W")
        if self.acoustic:
            lines.append(f"  Acoustic: NRC={self.acoustic.nrc:.2f}")
        if self.is_pareto:
            lines.append(f"  ★ PARETO OPTIMAL")
        return "\n".join(lines)


@dataclass
class ParetoResult:
    """Complete Pareto optimization output."""
    all_designs: list[DesignPoint]
    pareto_front: list[DesignPoint]
    n_evaluated: int
    target: MultiPhysicsTarget

    # Best on each axis
    best_color: Optional[DesignPoint] = None
    best_thermal: Optional[DesignPoint] = None
    best_acoustic: Optional[DesignPoint] = None
    best_balanced: Optional[DesignPoint] = None   # best weighted composite

    def summary(self) -> str:
        lines = [
            f"Multi-Physics Pareto Optimization",
            f"  Evaluated: {self.n_evaluated} design points",
            f"  Pareto front: {len(self.pareto_front)} designs",
            "",
        ]
        if self.best_balanced:
            lines.append(f"RECOMMENDED (balanced):")
            lines.append(self.best_balanced.summary())
            lines.append("")
        if self.best_color and self.best_color != self.best_balanced:
            lines.append(f"Best color:")
            lines.append(self.best_color.summary())
            lines.append("")
        if self.best_thermal and self.best_thermal != self.best_balanced:
            lines.append(f"Best thermal:")
            lines.append(self.best_thermal.summary())
            lines.append("")
        if self.best_acoustic and self.best_acoustic != self.best_balanced:
            lines.append(f"Best acoustic:")
            lines.append(self.best_acoustic.summary())
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Color prediction
# ═══════════════════════════════════════════════════════════════════════════

# Pre-defined target colors in CIE 1931 xy
NAMED_COLORS: dict[str, tuple[float, float]] = {
    "red":           (0.64, 0.33),
    "green":         (0.30, 0.60),
    "forest_green":  (0.30, 0.52),
    "blue":          (0.15, 0.06),
    "sky_blue":      (0.24, 0.28),
    "yellow":        (0.42, 0.51),
    "orange":        (0.50, 0.44),
    "purple":        (0.27, 0.12),
    "white":         (0.31, 0.33),
    "cyan":          (0.22, 0.33),
}


def _target_lab(cie_x: float, cie_y: float) -> tuple:
    """Convert CIE xy to approximate Lab for ΔE computation.

    Assumes Y=50 (mid-luminance) for the target since we only have chromaticity.
    """
    if cie_y <= 0:
        return (50.0, 0.0, 0.0)
    Y = 50.0
    X = (cie_x / cie_y) * Y
    Z = ((1 - cie_x - cie_y) / cie_y) * Y
    return XYZ_to_Lab(X, Y, Z) if _OPTICAL_AVAILABLE else (50.0, 0.0, 0.0)


def _predict_color(
    diameter_nm: float,
    material: str,
    volume_fraction: float,
    absorber_fraction: float,
    target: ColorTarget,
) -> ColorResult:
    """Predict structural color and compute ΔE from target."""
    if not _OPTICAL_AVAILABLE:
        # Fallback: estimate peak wavelength analytically
        lam_peak = photonic_glass_peak_analytical(diameter_nm, material, volume_fraction)
        return ColorResult(
            peak_wavelength_nm=lam_peak,
            predicted=False,
            delta_e=999.0,
        )

    wl = np.arange(380, 781, 5.0)

    # Material name mapping for optical pipeline
    optical_material_map = {
        "SiO2": "SiO2",
        "TiO2_rutile": "TiO2",
        "TiO2_anatase": "TiO2",
        "polystyrene": "polystyrene",
        "PMMA": "PMMA",
        "ZnS": "ZnS",
    }
    opt_mat = optical_material_map.get(material, "SiO2")

    try:
        R = photonic_glass_reflectance(
            diameter_nm, opt_mat, 1.0, volume_fraction, wl,
            absorber_material="carbon",
            absorber_fraction=absorber_fraction,
        )

        X, Y, Z = spectrum_to_XYZ(R, wl)
        x, y, Y_lum = XYZ_to_xyY(X, Y, Z)
        Lab_pred = XYZ_to_Lab(X, Y, Z)
        Lab_target = _target_lab(target.cie_x, target.cie_y)
        dE = cie_delta_E(Lab_pred, Lab_target)

        peak_lam = float(wl[np.argmax(R)])

        return ColorResult(
            peak_wavelength_nm=peak_lam,
            cie_x=round(float(x), 4),
            cie_y=round(float(y), 4),
            Lab=tuple(round(v, 2) for v in Lab_pred),
            delta_e=round(float(dE), 2),
            predicted=True,
        )
    except Exception:
        lam_peak = photonic_glass_peak_analytical(diameter_nm, material, volume_fraction)
        return ColorResult(peak_wavelength_nm=lam_peak, predicted=False, delta_e=999.0)


def photonic_glass_peak_analytical(
    diameter_nm: float,
    material: str = "SiO2",
    volume_fraction: float = 0.50,
) -> float:
    """Analytical estimate of photonic glass peak wavelength.

    λ_peak ≈ 2 × n_eff × d_avg
    where d_avg ≈ D (nearest-neighbor distance ≈ particle diameter)
    and n_eff ≈ √(φ × n_p² + (1-φ) × n_m²)
    """
    n_map = {"SiO2": 1.46, "TiO2_rutile": 2.61, "TiO2_anatase": 2.49,
             "polystyrene": 1.59, "PMMA": 1.49, "ZnS": 2.36, "BaTiO3": 2.40}
    n_p = n_map.get(material, 1.46)
    n_m = 1.0  # air
    n_eff = math.sqrt(volume_fraction * n_p**2 + (1 - volume_fraction) * n_m**2)
    return 2.0 * n_eff * diameter_nm


# ═══════════════════════════════════════════════════════════════════════════
# Design space grid
# ═══════════════════════════════════════════════════════════════════════════

def _build_design_grid(
    target: MultiPhysicsTarget,
    diameter_range: tuple[float, float] = (180.0, 380.0),
    diameter_step: float = 20.0,
    phi_range: tuple[float, float] = (0.35, 0.60),
    phi_step: float = 0.05,
    absorber_fractions: list[float] = None,
    film_thicknesses_um: list[float] = None,
    materials: list[str] = None,
    matrix_materials: list[str] = None,
) -> list[dict]:
    """Build a grid of design parameter combinations to evaluate."""

    if absorber_fractions is None:
        absorber_fractions = [0.001, 0.005, 0.01]
    if film_thicknesses_um is None:
        film_thicknesses_um = [50.0, 200.0, 1000.0]
    if materials is None:
        materials = ["SiO2"]
    if matrix_materials is None:
        matrix_materials = ["air"]

    grid = []
    d = diameter_range[0]
    while d <= diameter_range[1] + 0.1:
        phi = phi_range[0]
        while phi <= phi_range[1] + 0.001:
            for absorber in absorber_fractions:
                for thickness in film_thicknesses_um:
                    for mat in materials:
                        for mx in matrix_materials:
                            grid.append({
                                "diameter_nm": d,
                                "material": mat,
                                "volume_fraction": round(phi, 3),
                                "absorber_fraction": absorber,
                                "film_thickness_um": thickness,
                                "matrix_material": mx,
                            })
            phi += phi_step
        d += diameter_step

    return grid


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_design(params: dict, target: MultiPhysicsTarget) -> DesignPoint:
    """Evaluate one design point across all three physics domains."""
    D = params["diameter_nm"]
    mat = params["material"]
    phi = params["volume_fraction"]
    absorber = params["absorber_fraction"]
    t_um = params["film_thickness_um"]
    mx = params["matrix_material"]

    dp = DesignPoint(
        particle_diameter_nm=D,
        particle_material=mat,
        volume_fraction=phi,
        absorber_fraction=absorber,
        film_thickness_um=t_um,
        matrix_material=mx,
    )

    # ── Color ──
    if target.color:
        dp.color = _predict_color(D, mat, phi, absorber, target.color)
        dp.obj_color_delta_e = dp.color.delta_e
    else:
        dp.obj_color_delta_e = 0.0  # no color target → all equally good

    # ── Thermal ──
    try:
        dp.thermal = thermal_predict(D, mat, phi, mx, t_um)
        dp.obj_thermal_neg_R = -dp.thermal.R_value_m2KW  # negate: minimize = maximize R
    except Exception:
        dp.obj_thermal_neg_R = 0.0

    # ── Acoustic ──
    try:
        if target.backing_thickness_mm > 0:
            dp.acoustic = acoustic_predict_backing(
                D, phi, t_um,
                backing_thickness_mm=target.backing_thickness_mm,
                backing_flow_resistivity=target.backing_flow_resistivity,
                air_gap_mm=target.air_gap_mm,
            )
        else:
            dp.acoustic = acoustic_predict(D, phi, t_um, target.air_gap_mm)
        dp.obj_acoustic_neg_nrc = -dp.acoustic.nrc  # negate: minimize = maximize NRC
    except Exception:
        dp.obj_acoustic_neg_nrc = 0.0

    return dp


# ═══════════════════════════════════════════════════════════════════════════
# Pareto front extraction
# ═══════════════════════════════════════════════════════════════════════════

def _dominates(a: DesignPoint, b: DesignPoint) -> bool:
    """Does design `a` dominate design `b`? (all objectives ≤, at least one <)"""
    objs_a = (a.obj_color_delta_e, a.obj_thermal_neg_R, a.obj_acoustic_neg_nrc)
    objs_b = (b.obj_color_delta_e, b.obj_thermal_neg_R, b.obj_acoustic_neg_nrc)

    all_leq = all(oa <= ob for oa, ob in zip(objs_a, objs_b))
    any_lt = any(oa < ob for oa, ob in zip(objs_a, objs_b))
    return all_leq and any_lt


def extract_pareto_front(designs: list[DesignPoint]) -> list[DesignPoint]:
    """Extract non-dominated designs (Pareto front)."""
    n = len(designs)
    for i in range(n):
        designs[i].dominated_by = 0
        designs[i].is_pareto = True

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(designs[j], designs[i]):
                designs[i].dominated_by += 1
                designs[i].is_pareto = False

    return [d for d in designs if d.is_pareto]


# ═══════════════════════════════════════════════════════════════════════════
# Main optimizer
# ═══════════════════════════════════════════════════════════════════════════

def optimize_multiphysics(
    target: MultiPhysicsTarget,
    diameter_range: tuple[float, float] = (180.0, 380.0),
    diameter_step: float = 20.0,
    phi_range: tuple[float, float] = (0.35, 0.60),
    phi_step: float = 0.05,
    absorber_fractions: list[float] = None,
    film_thicknesses_um: list[float] = None,
    materials: list[str] = None,
    matrix_materials: list[str] = None,
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> ParetoResult:
    """Run multi-objective optimization across color + thermal + acoustic.

    Args:
        target: MultiPhysicsTarget specification
        diameter_range: (min, max) particle diameter in nm
        diameter_step: grid step for diameter
        phi_range: (min, max) volume fraction
        phi_step: grid step for φ
        absorber_fractions: list of absorber fractions to try
        film_thicknesses_um: list of film thicknesses to try
        materials: particle materials to try
        matrix_materials: matrix materials to try
        weights: (w_color, w_thermal, w_acoustic) for balanced score

    Returns:
        ParetoResult with all designs, Pareto front, and recommendations.
    """
    grid = _build_design_grid(
        target,
        diameter_range=diameter_range,
        diameter_step=diameter_step,
        phi_range=phi_range,
        phi_step=phi_step,
        absorber_fractions=absorber_fractions,
        film_thicknesses_um=film_thicknesses_um,
        materials=materials,
        matrix_materials=matrix_materials,
    )

    # Evaluate all design points
    designs = []
    for params in grid:
        dp = evaluate_design(params, target)
        designs.append(dp)

    if not designs:
        return ParetoResult([], [], 0, target)

    # Extract Pareto front
    pareto = extract_pareto_front(designs)

    # Find best on each axis
    best_color = min(designs, key=lambda d: d.obj_color_delta_e)
    best_thermal = min(designs, key=lambda d: d.obj_thermal_neg_R)
    best_acoustic = min(designs, key=lambda d: d.obj_acoustic_neg_nrc)

    # Normalize objectives to [0, 1] for weighted scoring
    de_vals = [d.obj_color_delta_e for d in designs]
    tr_vals = [d.obj_thermal_neg_R for d in designs]
    ac_vals = [d.obj_acoustic_neg_nrc for d in designs]

    de_min, de_max = min(de_vals), max(de_vals)
    tr_min, tr_max = min(tr_vals), max(tr_vals)
    ac_min, ac_max = min(ac_vals), max(ac_vals)

    def _norm(v, vmin, vmax):
        if vmax - vmin < 1e-12:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    w_c, w_t, w_a = weights

    best_balanced = None
    best_score = float('inf')
    for d in designs:
        score = (w_c * _norm(d.obj_color_delta_e, de_min, de_max) +
                 w_t * _norm(d.obj_thermal_neg_R, tr_min, tr_max) +
                 w_a * _norm(d.obj_acoustic_neg_nrc, ac_min, ac_max))
        if score < best_score:
            best_score = score
            best_balanced = d

    return ParetoResult(
        all_designs=designs,
        pareto_front=pareto,
        n_evaluated=len(designs),
        target=target,
        best_color=best_color,
        best_thermal=best_thermal,
        best_acoustic=best_acoustic,
        best_balanced=best_balanced,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: quick design for named color
# ═══════════════════════════════════════════════════════════════════════════

def quick_design(
    color_name: str = "green",
    thermal_R_min: float = 0.0,
    acoustic_nrc_min: float = 0.0,
    application: str = "building_panel",
    backing_mm: float = 0.0,
    air_gap_mm: float = 0.0,
) -> ParetoResult:
    """Quick multi-physics design from a color name.

    Args:
        color_name: key in NAMED_COLORS or "red", "green", "blue", etc.
        thermal_R_min: minimum R-value target (0 = no constraint)
        acoustic_nrc_min: minimum NRC target (0 = no constraint)
        application: "building_panel", "textile", etc.
        backing_mm: fibrous backing thickness in mm (0 = none)
        air_gap_mm: air gap behind structure (0 = none)
    """
    xy = NAMED_COLORS.get(color_name.lower(), (0.31, 0.33))

    target = MultiPhysicsTarget(
        color=ColorTarget(cie_x=xy[0], cie_y=xy[1], name=color_name, max_delta_e=10.0),
        thermal=ThermalTarget(min_R_value_m2KW=thermal_R_min) if thermal_R_min > 0 else None,
        acoustic=AcousticTarget(min_nrc=acoustic_nrc_min) if acoustic_nrc_min > 0 else None,
        application=application,
        backing_thickness_mm=backing_mm,
        backing_flow_resistivity=10000.0,
        air_gap_mm=air_gap_mm,
    )

    return optimize_multiphysics(target)
