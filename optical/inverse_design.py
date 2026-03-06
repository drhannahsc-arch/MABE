"""
optical/inverse_design.py — Module 11: Inverse Design Wrapper

Given a target color (CIE xy or Lab), find the particle/film parameters
that produce it. Uses scipy.optimize.differential_evolution over the
forward model chain (Modules 1→3→5→6→8→9).

Entry points:
  inverse_design_photonic_glass(target_xy, ...) → DesignResult
  inverse_design_multilayer(target_spectrum, ...) → DesignResult

No new physics. Pure optimization wrapper around validated forward models.
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import differential_evolution, minimize

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from optical.photonic_glass import photonic_glass_reflectance
from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, cie_delta_E, XYZ_to_sRGB,
)
from optical.underlayer_coupling import photonic_glass_on_substrate
from optical.tmm import tmm_reflectance


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhotonicGlassDesign:
    """Output of photonic glass inverse design."""
    diameter_nm: float
    sphere_material: str
    n_medium: float
    packing_fraction: float
    absorber_fraction: float
    underlayer: str

    # Predicted optical properties
    peak_wavelength_nm: float = 0.0
    cie_x: float = 0.0
    cie_y: float = 0.0
    Lab: tuple = (0.0, 0.0, 0.0)
    sRGB: tuple = (0, 0, 0)
    delta_E: float = 0.0            # ΔE from target

    def __repr__(self):
        return (f"PhotonicGlassDesign(D={self.diameter_nm:.0f}nm, "
                f"{self.sphere_material}, φ={self.packing_fraction:.2f}, "
                f"abs={self.absorber_fraction:.3f}, under={self.underlayer}, "
                f"ΔE={self.delta_E:.1f})")


@dataclass
class MultilayerDesign:
    """Output of multilayer inverse design."""
    materials: list                  # [str, ...] per layer
    thicknesses_nm: list             # [float, ...] per layer
    n_layers: int = 0

    # Predicted
    stopband_centre_nm: float = 0.0
    cie_x: float = 0.0
    cie_y: float = 0.0
    Lab: tuple = (0.0, 0.0, 0.0)
    delta_E: float = 0.0


@dataclass
class DesignResult:
    """Full inverse design output."""
    target_xy: tuple = (0.0, 0.0)
    target_Lab: tuple = (0.0, 0.0, 0.0)
    design: object = None            # PhotonicGlassDesign or MultilayerDesign
    converged: bool = False
    delta_E: float = 999.0
    n_evaluations: int = 0
    elapsed_s: float = 0.0
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# WAVELENGTH GRID
# ═══════════════════════════════════════════════════════════════════════════

_LAM = np.linspace(380, 780, 81)   # 5nm steps across visible range


# ═══════════════════════════════════════════════════════════════════════════
# PHOTONIC GLASS INVERSE DESIGN
# ═══════════════════════════════════════════════════════════════════════════

# Design variable bounds
_PG_BOUNDS = [
    (80, 500),      # diameter_nm
    (0.40, 0.64),   # packing_fraction
    (0.0, 0.10),    # absorber_fraction
]

# Material options
_SPHERE_MATERIALS = ["SiO2", "polystyrene", "TiO2_rutile"]
_UNDERLAYER_OPTIONS = ["air", "carbon", "Fe2O3", "TiO2_rutile", "white_bead"]


def _pg_objective(params, target_x, target_y, sphere_material, n_medium,
                  underlayer, lam):
    """Objective function: chromaticity distance + lightness penalty."""
    diameter, phi, abs_frac = params

    try:
        R = photonic_glass_reflectance(
            diameter_nm=diameter,
            sphere_material=sphere_material,
            n_medium_base=n_medium,
            packing_fraction=phi,
            wavelengths_nm=lam,
            absorber_fraction=abs_frac,
        )

        # Apply underlayer if specified
        if underlayer and underlayer != "air":
            R = photonic_glass_on_substrate(
                R, lam, substrate=underlayer)

        X, Y, Z = spectrum_to_XYZ(R, lam)
        x, y, Y_lum = XYZ_to_xyY(X, Y, Z)

        # Chromaticity distance (weighted by 100 to match Lab scale)
        dx = (x - target_x) * 100
        dy = (y - target_y) * 100
        chroma_dist = np.sqrt(dx**2 + dy**2)

        return chroma_dist
    except Exception:
        return 999.0


def inverse_design_photonic_glass(
    target_x, target_y,
    sphere_material="SiO2",
    n_medium=1.0,
    underlayer="carbon",
    diameter_bounds=(80, 500),
    phi_bounds=(0.40, 0.64),
    absorber_bounds=(0.0, 0.10),
    max_iter=200,
    tol=0.5,
    seed=42,
):
    """Design a photonic glass to match a target CIE chromaticity.

    Args:
        target_x, target_y: CIE 1931 chromaticity coordinates
        sphere_material: particle material (from refractive_index database)
        n_medium: surrounding medium refractive index
        underlayer: substrate material ("air", "carbon", "Fe2O3", etc.)
        diameter_bounds: (min_nm, max_nm) for particle diameter
        phi_bounds: (min, max) for packing fraction
        absorber_bounds: (min, max) for absorber volume fraction
        max_iter: max optimizer iterations
        tol: ΔE tolerance for convergence
        seed: random seed for reproducibility

    Returns:
        DesignResult with PhotonicGlassDesign
    """
    import time
    t0 = time.time()

    bounds = [diameter_bounds, phi_bounds, absorber_bounds]
    n_evals = [0]

    def _obj(params):
        n_evals[0] += 1
        return _pg_objective(params, target_x, target_y, sphere_material,
                             n_medium, underlayer, _LAM)

    result = differential_evolution(
        _obj, bounds,
        maxiter=max_iter,
        tol=tol / 100,   # relative tolerance, scale down
        seed=seed,
        polish=True,
        init='sobol',
    )

    # Extract best design
    diameter, phi, abs_frac = result.x
    best_dE = result.fun

    # Compute full predicted properties at optimum
    R = photonic_glass_reflectance(
        diameter_nm=diameter,
        sphere_material=sphere_material,
        n_medium_base=n_medium,
        packing_fraction=phi,
        wavelengths_nm=_LAM,
        absorber_fraction=abs_frac,
    )
    if underlayer and underlayer != "air":
        R = photonic_glass_on_substrate(R, _LAM, substrate=underlayer)

    X, Y, Z = spectrum_to_XYZ(R, _LAM)
    x_pred, y_pred, _ = XYZ_to_xyY(X, Y, Z)
    Lab_pred = XYZ_to_Lab(X, Y, Z)
    srgb = XYZ_to_sRGB(X, Y, Z)
    srgb_int = tuple(max(0, min(255, int(round(c * 255)))) for c in srgb)
    peak_nm = _LAM[np.argmax(R)]

    design = PhotonicGlassDesign(
        diameter_nm=round(diameter, 1),
        sphere_material=sphere_material,
        n_medium=n_medium,
        packing_fraction=round(phi, 3),
        absorber_fraction=round(abs_frac, 4),
        underlayer=underlayer,
        peak_wavelength_nm=peak_nm,
        cie_x=round(x_pred, 4),
        cie_y=round(y_pred, 4),
        Lab=tuple(round(v, 1) for v in Lab_pred),
        sRGB=srgb_int,
        delta_E=round(best_dE, 2),
    )

    converged = best_dE < tol * 5  # generous convergence criterion

    notes = ""
    if best_dE > 20:
        notes = "WARNING: Large ΔE — target may be unreachable with this material/geometry"
    elif best_dE > 10:
        notes = "Moderate ΔE — approximate match only"

    return DesignResult(
        target_xy=(target_x, target_y),
        target_Lab=Lab_pred,
        design=design,
        converged=converged,
        delta_E=round(best_dE, 2),
        n_evaluations=n_evals[0],
        elapsed_s=round(time.time() - t0, 2),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MULTILAYER (TMM) INVERSE DESIGN
# ═══════════════════════════════════════════════════════════════════════════

def inverse_design_multilayer(
    target_centre_nm,
    material_H="TiO2_rutile",
    material_L="SiO2",
    n_periods=5,
    thickness_bounds_nm=(20, 300),
    max_iter=100,
    seed=42,
):
    """Design a dielectric multilayer (Bragg mirror) targeting a stopband centre.

    Simple alternating H/L stack. Optimizes layer thicknesses to place
    the reflection peak at target_centre_nm.

    Args:
        target_centre_nm: desired reflection peak wavelength
        material_H: high-n material
        material_L: low-n material
        n_periods: number of H/L pairs
        thickness_bounds_nm: bounds for each layer thickness
        max_iter: optimizer iterations
        seed: random seed

    Returns:
        DesignResult with MultilayerDesign
    """
    import time
    t0 = time.time()

    # 2 thicknesses to optimize: d_H and d_L
    bounds = [thickness_bounds_nm, thickness_bounds_nm]
    n_evals = [0]

    def _obj(params):
        n_evals[0] += 1
        d_H, d_L = params

        # Build alternating stack: [ambient, H, L, H, L, ..., substrate]
        stack = [("air", 0)]
        for _ in range(n_periods):
            stack.append((material_H, d_H))
            stack.append((material_L, d_L))
        stack.append(("SiO2", 0))

        try:
            R_spectrum = np.array([
                tmm_reflectance(stack, lam_i)[0]
                for lam_i in _LAM
            ])
            peak_nm = _LAM[np.argmax(R_spectrum)]
            return abs(peak_nm - target_centre_nm)
        except Exception:
            return 999.0

    result = differential_evolution(
        _obj, bounds, maxiter=max_iter, seed=seed, polish=True,
    )

    d_H, d_L = result.x
    materials = []
    thicknesses = []
    for _ in range(n_periods):
        materials.extend([material_H, material_L])
        thicknesses.extend([round(d_H, 1), round(d_L, 1)])

    # Compute predicted spectrum at optimum
    stack = [("air", 0)]
    for m, t in zip(materials, thicknesses):
        stack.append((m, t))
    stack.append(("SiO2", 0))

    R = np.array([tmm_reflectance(stack, lam_i)[0] for lam_i in _LAM])
    peak_nm = _LAM[np.argmax(R)]
    X, Y, Z = spectrum_to_XYZ(R, _LAM)
    x_pred, y_pred, _ = XYZ_to_xyY(X, Y, Z)
    Lab_pred = XYZ_to_Lab(X, Y, Z)

    design = MultilayerDesign(
        materials=materials,
        thicknesses_nm=thicknesses,
        n_layers=len(materials),
        stopband_centre_nm=peak_nm,
        cie_x=round(x_pred, 4),
        cie_y=round(y_pred, 4),
        Lab=tuple(round(v, 1) for v in Lab_pred),
        delta_E=round(result.fun, 2),
    )

    return DesignResult(
        target_xy=(0, 0),
        target_Lab=(0, 0, 0),
        design=design,
        converged=result.fun < 10,
        delta_E=round(result.fun, 2),
        n_evaluations=n_evals[0],
        elapsed_s=round(time.time() - t0, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-MATERIAL SCAN
# ═══════════════════════════════════════════════════════════════════════════

def scan_materials(target_x, target_y, underlayer="carbon", max_iter=100):
    """Try all sphere material × underlayer combinations.

    Returns list of DesignResult, sorted by ΔE (best first).
    """
    results = []
    for mat in _SPHERE_MATERIALS:
        for under in _UNDERLAYER_OPTIONS:
            try:
                r = inverse_design_photonic_glass(
                    target_x, target_y,
                    sphere_material=mat,
                    underlayer=under,
                    max_iter=max_iter,
                )
                results.append(r)
            except Exception:
                continue

    results.sort(key=lambda r: r.delta_E)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_design(result):
    """Pretty-print a design result."""
    print()
    d = result.design
    print(f"  MABE Optical Inverse Design")
    print(f"  Target: CIE xy = ({result.target_xy[0]:.3f}, {result.target_xy[1]:.3f})")
    print(f"  Target Lab: {result.target_Lab}")
    print(f"  Converged: {result.converged}  ΔE = {result.delta_E:.1f}")
    print(f"  Evaluations: {result.n_evaluations}  Time: {result.elapsed_s:.1f}s")
    if result.notes:
        print(f"  {result.notes}")
    print()

    if isinstance(d, PhotonicGlassDesign):
        print(f"  ── Photonic Glass Design ──")
        print(f"  Particle diameter:   {d.diameter_nm:.0f} nm")
        print(f"  Material:            {d.sphere_material}")
        print(f"  Packing fraction:    {d.packing_fraction:.3f}")
        print(f"  Absorber fraction:   {d.absorber_fraction:.4f}")
        print(f"  Underlayer:          {d.underlayer}")
        print(f"  Peak wavelength:     {d.peak_wavelength_nm:.0f} nm")
        print(f"  Predicted CIE xy:    ({d.cie_x:.4f}, {d.cie_y:.4f})")
        print(f"  Predicted Lab:       {d.Lab}")
        print(f"  sRGB:                {d.sRGB}")

    elif isinstance(d, MultilayerDesign):
        print(f"  ── Multilayer Design ──")
        print(f"  Layers: {d.n_layers}")
        print(f"  Stopband centre: {d.stopband_centre_nm:.0f} nm")
        for i in range(min(6, d.n_layers)):
            print(f"    Layer {i+1}: {d.materials[i]:15s} {d.thicknesses_nm[i]:.1f} nm")
        if d.n_layers > 6:
            print(f"    ... ({d.n_layers - 6} more layers)")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION-LEVEL INVERSE DESIGN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ApplicationDesign:
    """Output of application-level inverse design."""
    # Particle parameters
    diameter_nm: float = 0.0
    sphere_material: str = "SiO2"
    packing_fraction: float = 0.55
    absorber_fraction: float = 0.0
    absorber_material: str = "carbon"

    # Application parameters
    scenario: str = ""            # "glass_film", "textile_dipcoat", etc.
    substrate: str = ""
    n_layers: int = 10
    coverage_fraction: float = 1.0
    viewing: str = "lambertian"

    # Optional linker
    anchor: str = ""
    spacer: str = ""
    click: str = ""
    shell_thickness_nm: float = 0.0

    # Predicted observed color
    peak_nm: float = 0.0
    cie_x: float = 0.0
    cie_y: float = 0.0
    Lab: tuple = (0.0, 0.0, 0.0)
    sRGB: tuple = (0, 0, 0)
    brightness: float = 0.0
    delta_E: float = 0.0

    # Transmission (glass only)
    cie_x_T: float = 0.0
    cie_y_T: float = 0.0
    sRGB_T: tuple = (0, 0, 0)


# Scenario-specific defaults
_SCENARIO_DEFAULTS = {
    "glass_film": {
        "geometry": "bulk_film",
        "n_layers": 10,
        "coverage": 1.0,
        "viewing": "specular",
        "substrate": "carbon",
        "phi_bounds": (0.45, 0.64),
        "abs_bounds": (0.0, 0.08),
        "transmission": True,
    },
    "textile_dipcoat": {
        "geometry": "thin_film",
        "n_layers": 3,
        "coverage": 0.7,
        "viewing": "lambertian",
        "substrate": "black_polyester",
        "phi_bounds": (0.40, 0.60),
        "abs_bounds": (0.0, 0.05),
        "transmission": False,
    },
    "beads_on_glass": {
        "geometry": "monolayer",
        "n_layers": 1,
        "coverage": 0.6,
        "viewing": "specular",
        "substrate": "carbon",
        "phi_bounds": (0.50, 0.60),
        "abs_bounds": (0.0, 0.0),
        "transmission": True,
    },
    "beads_on_textile": {
        "geometry": "sparse",
        "n_layers": 1,
        "coverage": 0.3,
        "viewing": "lambertian",
        "substrate": "black_polyester",
        "phi_bounds": (0.50, 0.60),
        "abs_bounds": (0.0, 0.0),
        "transmission": False,
    },
}


def _application_objective(params, target_x, target_y, scenario_cfg,
                           sphere_material, substrate, lam):
    """Objective: chromaticity distance through full application stack."""
    diameter, phi, abs_frac = params

    try:
        from optical.application_optics import (
            CoatingSpec, predict_application,
        )

        spec = CoatingSpec(
            geometry=scenario_cfg["geometry"],
            n_layers=scenario_cfg["n_layers"],
            coverage_fraction=scenario_cfg["coverage"],
            core_diameter_nm=float(diameter),
            core_material=sphere_material,
            n_medium=1.0,
            packing_fraction=float(phi),
            absorber_fraction=float(abs_frac),
            absorber_material="carbon",
        )

        result = predict_application(
            spec, substrate=substrate,
            viewing=scenario_cfg["viewing"],
            include_transmission=False,
            wavelengths_nm=lam,
        )

        x, y = result.cie_xy
        # Chromaticity distance (scaled to ~Lab range)
        dx = (x - target_x) * 100
        dy = (y - target_y) * 100
        return float(np.sqrt(dx**2 + dy**2))

    except Exception as e:
        return 999.0


def inverse_design_application(
    target_x: float,
    target_y: float,
    scenario: str = "textile_dipcoat",
    substrate: str = None,
    sphere_material: str = "SiO2",
    diameter_bounds: tuple = (100, 400),
    n_layers: int = None,
    coverage: float = None,
    max_iter: int = 150,
    tol: float = 1.0,
    seed: int = 42,
) -> DesignResult:
    """Inverse design optimizing through the full application stack.

    The optimizer sees what the observer sees: coating geometry →
    substrate coupling → viewing angle integration → CIE color.

    Args:
        target_x, target_y:  Target CIE chromaticity
        scenario:            "glass_film", "textile_dipcoat",
                             "beads_on_glass", "beads_on_textile"
        substrate:           Override substrate (default: scenario-dependent)
        sphere_material:     Particle material
        diameter_bounds:     (min_nm, max_nm)
        n_layers:            Override layer count
        coverage:            Override coverage fraction
        max_iter:            Optimizer iterations
        tol:                 ΔE tolerance
        seed:                Random seed

    Returns:
        DesignResult with ApplicationDesign
    """
    import time
    t0 = time.time()

    if scenario not in _SCENARIO_DEFAULTS:
        raise ValueError(f"Unknown scenario: {scenario}. "
                         f"Options: {list(_SCENARIO_DEFAULTS)}")

    cfg = dict(_SCENARIO_DEFAULTS[scenario])
    if substrate is not None:
        cfg["substrate"] = substrate
    else:
        substrate = cfg["substrate"]
    if n_layers is not None:
        cfg["n_layers"] = n_layers
    if coverage is not None:
        cfg["coverage"] = coverage

    bounds = [diameter_bounds, cfg["phi_bounds"], cfg["abs_bounds"]]
    n_evals = [0]

    def _obj(params):
        n_evals[0] += 1
        return _application_objective(params, target_x, target_y, cfg,
                                       sphere_material, substrate, _LAM)

    result = differential_evolution(
        _obj, bounds,
        maxiter=max_iter,
        tol=tol / 100,
        seed=seed,
        polish=True,
        init='sobol',
    )

    diameter, phi, abs_frac = result.x
    best_dE = result.fun

    # Compute full application result at optimum
    from optical.application_optics import (
        CoatingSpec, predict_application,
    )

    spec = CoatingSpec(
        geometry=cfg["geometry"],
        n_layers=cfg["n_layers"],
        coverage_fraction=cfg["coverage"],
        core_diameter_nm=diameter,
        core_material=sphere_material,
        n_medium=1.0,
        packing_fraction=phi,
        absorber_fraction=abs_frac,
        absorber_material="carbon",
    )

    app_result = predict_application(
        spec, substrate=substrate,
        viewing=cfg["viewing"],
        include_transmission=cfg.get("transmission", False),
        wavelengths_nm=_LAM,
    )

    design = ApplicationDesign(
        diameter_nm=round(diameter, 1),
        sphere_material=sphere_material,
        packing_fraction=round(phi, 3),
        absorber_fraction=round(abs_frac, 4),
        scenario=scenario,
        substrate=substrate,
        n_layers=cfg["n_layers"],
        coverage_fraction=cfg["coverage"],
        viewing=cfg["viewing"],
        peak_nm=app_result.peak_nm,
        cie_x=app_result.cie_xy[0],
        cie_y=app_result.cie_xy[1],
        Lab=app_result.Lab,
        sRGB=app_result.sRGB,
        brightness=app_result.brightness,
        delta_E=round(best_dE, 2),
    )

    if cfg.get("transmission", False) and len(app_result.T_observed) > 0:
        design.cie_x_T = app_result.cie_xy_T[0]
        design.cie_y_T = app_result.cie_xy_T[1]
        design.sRGB_T = app_result.sRGB_T

    converged = best_dE < tol * 5
    notes = ""
    if best_dE > 15:
        notes = f"WARNING: Large ΔE={best_dE:.1f} — target may be unreachable for {scenario}"
    elif best_dE > 5:
        notes = f"Moderate ΔE={best_dE:.1f} — approximate match"

    # Scenario-specific advice
    if scenario == "beads_on_textile" and best_dE > 10:
        notes += ". Sparse beads have very low R — consider textile_dipcoat for stronger color."
    if scenario == "textile_dipcoat" and abs_frac < 0.001:
        notes += ". Consider adding absorber (carbon black) to improve saturation."

    return DesignResult(
        target_xy=(target_x, target_y),
        target_Lab=(0, 0, 0),
        design=design,
        converged=converged,
        delta_E=round(best_dE, 2),
        n_evaluations=n_evals[0],
        elapsed_s=round(time.time() - t0, 2),
        notes=notes,
    )


def print_application_design(result):
    """Pretty-print an application-level design result."""
    print()
    d = result.design
    print(f"  MABE Application-Level Inverse Design")
    print(f"  Target: CIE xy = ({result.target_xy[0]:.3f}, {result.target_xy[1]:.3f})")
    print(f"  Scenario: {d.scenario}")
    print(f"  Converged: {result.converged}  ΔE = {result.delta_E:.1f}")
    print(f"  Evaluations: {result.n_evaluations}  Time: {result.elapsed_s:.1f}s")
    if result.notes:
        print(f"  {result.notes}")
    print()
    print(f"  ── Particle Spec ──")
    print(f"  Diameter:          {d.diameter_nm:.0f} nm")
    print(f"  Material:          {d.sphere_material}")
    print(f"  Packing fraction:  {d.packing_fraction:.3f}")
    print(f"  Absorber:          {d.absorber_fraction:.4f} ({d.absorber_material})")
    print()
    print(f"  ── Application Spec ──")
    print(f"  Geometry:          {d.scenario}")
    print(f"  Substrate:         {d.substrate}")
    print(f"  Layers:            {d.n_layers}")
    print(f"  Coverage:          {d.coverage_fraction:.0%}")
    print(f"  Viewing:           {d.viewing}")
    print()
    print(f"  ── Observed Color ──")
    print(f"  Peak wavelength:   {d.peak_nm:.0f} nm")
    print(f"  CIE xy:            ({d.cie_x:.4f}, {d.cie_y:.4f})")
    print(f"  Lab:               {d.Lab}")
    print(f"  sRGB:              {d.sRGB}")
    print(f"  Brightness (Y):    {d.brightness:.1f}")
    if d.cie_x_T > 0:
        print(f"  Transmitted CIE:   ({d.cie_x_T:.4f}, {d.cie_y_T:.4f})")
        print(f"  Transmitted sRGB:  {d.sRGB_T}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MABE Optical Inverse Design — Self-Test")
    print("=" * 60)

    # Test 1: Blue target
    print("\n--- Test 1: Design blue photonic glass ---")
    r = inverse_design_photonic_glass(0.15, 0.10, max_iter=50)
    print_design(r)

    # Test 2: Green target
    print("--- Test 2: Design green photonic glass ---")
    r2 = inverse_design_photonic_glass(0.27, 0.40, max_iter=50)
    print_design(r2)

    # Test 3: Red target (should show difficulty)
    print("--- Test 3: Design red photonic glass (hard) ---")
    r3 = inverse_design_photonic_glass(0.50, 0.30, max_iter=50)
    print_design(r3)

    # Test 4: Multilayer
    print("--- Test 4: Design green Bragg mirror ---")
    r4 = inverse_design_multilayer(550, max_iter=30)
    print_design(r4)

    # Test 5: APPLICATION — blue on black polyester textile
    print("--- Test 5: Blue on black polyester textile ---")
    r5 = inverse_design_application(0.15, 0.10,
                                     scenario="textile_dipcoat",
                                     substrate="black_polyester",
                                     max_iter=50)
    print_application_design(r5)

    # Test 6: APPLICATION — green on glass
    print("--- Test 6: Green on glass film ---")
    r6 = inverse_design_application(0.27, 0.40,
                                     scenario="glass_film",
                                     max_iter=50)
    print_application_design(r6)

    # Test 7: APPLICATION — blue on denim
    print("--- Test 7: Blue on denim textile ---")
    r7 = inverse_design_application(0.15, 0.15,
                                     scenario="textile_dipcoat",
                                     substrate="denim",
                                     max_iter=50)
    print_application_design(r7)
