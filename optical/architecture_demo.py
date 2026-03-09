"""
optical/architecture_demo.py — Module 12: Architecture Isomorphism

Demonstrates that the MABE four-layer architecture handles molecular
recognition (DiscretePocketSpec) and wavelength capture
(FieldInteractionSpec) identically.

The key architectural claim: Layers 1-2 never branch on domain type.
The same dispatch mechanism routes both spec types through the engine.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from models import (
    AngularBehavior, ApplicationContext, CavityDimensions, CavityShape,
    DiscretePocketSpec, DonorPosition, FieldInteractionSpec, FieldResponse,
    FieldType, InteractionParadigm, InteractionSpec, ScaleClass, Solvent,
)


@dataclass
class PhysicsResult:
    """Generic result from a Layer 1 physics solver."""
    solver_name: str
    spec_type: str
    primary_metric: float
    primary_metric_name: str
    secondary_metrics: dict = field(default_factory=dict)
    feasible: bool = True
    notes: list[str] = field(default_factory=list)


def solve_molecular_physics(spec: DiscretePocketSpec) -> PhysicsResult:
    """Layer 1 solver for molecular recognition (simplified demo)."""
    n_donors = len(spec.donor_positions)
    cavity_vol = spec.cavity_dimensions.volume_A3
    donor_types = spec.required_donor_types
    dG_estimate = -5.0 * n_donors - 0.1 * abs(cavity_vol - 30.0)

    return PhysicsResult(
        solver_name="unified_metal_scorer",
        spec_type=spec.spec_type,
        primary_metric=dG_estimate,
        primary_metric_name="ΔG_bind (kJ/mol)",
        secondary_metrics={
            "n_donors": n_donors,
            "donor_types": sorted(donor_types),
            "cavity_volume_A3": cavity_vol,
        },
        feasible=dG_estimate < 0,
        notes=[
            f"Dispatched via spec_type='{spec.spec_type}'",
            f"Solver: thermodynamic scoring ({n_donors} donors)",
        ],
    )


def solve_field_physics(spec: FieldInteractionSpec) -> PhysicsResult:
    """Layer 1 solver for field interactions — runs actual optical pipeline."""
    try:
        from optical.photonic_glass import photonic_glass_reflectance
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_sRGB

        wl = np.linspace(380, 780, 81)
        if spec.target_wavelength_nm is not None:
            d_est = spec.target_wavelength_nm / (1.22 * 1.33)
        else:
            d_est = 220.0

        R = photonic_glass_reflectance(
            diameter_nm=d_est, sphere_material="SiO2",
            n_medium_base=1.0, packing_fraction=0.50, wavelengths_nm=wl,
        )
        X, Y, Z = spectrum_to_XYZ(R, wl)
        x, y, _ = XYZ_to_xyY(X, Y, Z)
        r, g, b = XYZ_to_sRGB(X, Y, Z)

        dE = 0.0
        if spec.target_x is not None and spec.target_y is not None:
            dE = ((x - spec.target_x)**2 + (y - spec.target_y)**2)**0.5

        return PhysicsResult(
            solver_name="optical_forward_model",
            spec_type=spec.spec_type,
            primary_metric=dE,
            primary_metric_name="ΔE_CIE (chromaticity distance)",
            secondary_metrics={
                "particle_diameter_nm": d_est,
                "CIE_x": round(x, 4),
                "CIE_y": round(y, 4),
                "sRGB": (r, g, b),
                "peak_reflectance": float(np.max(R)),
                "angular_behavior": spec.angular_behavior.value,
            },
            feasible=True,
            notes=[
                f"Dispatched via spec_type='{spec.spec_type}'",
                "Solver: Mie + PY S(q) + CIE (81-point spectrum)",
            ],
        )
    except ImportError as e:
        return PhysicsResult(
            solver_name="optical_forward_model",
            spec_type=spec.spec_type,
            primary_metric=0.0,
            primary_metric_name="ΔE_CIE (chromaticity distance)",
            feasible=True,
            notes=[
                f"Dispatched via spec_type='{spec.spec_type}'",
                f"Optical modules not importable — stub ({e})",
            ],
        )


_SOLVERS = {
    InteractionParadigm.POCKET.value: solve_molecular_physics,
    InteractionParadigm.FIELD.value: solve_field_physics,
}


def dispatch_physics(spec: InteractionSpec) -> PhysicsResult:
    """Universal dispatcher: routes any InteractionSpec to its solver."""
    solver = _SOLVERS.get(spec.spec_type)
    if solver is None:
        raise NotImplementedError(
            f"No solver registered for spec_type='{spec.spec_type}'. "
            f"Known types: {sorted(_SOLVERS.keys())}"
        )
    return solver(spec)


@dataclass
class RealizationOption:
    name: str
    physics_class: str
    estimated_fidelity: float
    estimated_cost: str
    key_advantage: str
    key_limitation: str


def get_realization_options(spec: InteractionSpec) -> list[RealizationOption]:
    """Layer 3: Suggest realization options based on spec_type."""
    if spec.spec_type == InteractionParadigm.POCKET.value:
        return [
            RealizationOption("crown_ether", "covalent_cavity", 0.85, "$10-80/g",
                              "Size-match selectivity (Izatt data)", "Cation-only"),
            RealizationOption("cyclodextrin", "covalent_cavity", 0.75, "$0.15-15/g",
                              "Commodity, BackSolve-calibrated", "Hydrophobic guests only"),
            RealizationOption("porphyrin", "covalent_cavity", 0.80, "$5-50/g",
                              "0.01 A positioning precision", "4N planar only"),
            RealizationOption("functionalized_lignin", "bulk_sorbent", 0.35, "$0.50/kg",
                              "Scalable to tonnes, renewable", "Distributed sites"),
        ]
    elif spec.spec_type == InteractionParadigm.FIELD.value:
        return [
            RealizationOption("photonic_glass", "disordered_colloidal", 0.80, "$5-50/m2",
                              "Non-iridescent, angle-independent", "Red problem"),
            RealizationOption("bragg_opal", "ordered_colloidal", 0.90, "$10-100/m2",
                              "Sharp spectral peak, full visible range", "Iridescent"),
            RealizationOption("tmm_multilayer", "thin_film_stack", 0.85, "$50-500/m2",
                              "Precise filter design", "Requires vacuum deposition"),
        ]
    return []


def make_cu2_pocket_spec() -> DiscretePocketSpec:
    return DiscretePocketSpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(33.0, 3.96, 3.4, 3.96),
        symmetry="D4h",
        donor_positions=[
            DonorPosition("N", "equatorial", (1.98, 0, 0), 0.05, "sp2"),
            DonorPosition("N", "equatorial", (0, 1.98, 0), 0.05, "sp2"),
            DonorPosition("N", "equatorial", (-1.98, 0, 0), 0.05, "sp2"),
            DonorPosition("N", "equatorial", (0, -1.98, 0), 0.05, "sp2"),
        ],
        pocket_scale_nm=0.40,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
    )


def make_blue_structural_color_spec() -> FieldInteractionSpec:
    return FieldInteractionSpec(
        field_type=FieldType.ELECTROMAGNETIC,
        target_wavelength_nm=470.0,
        target_bandwidth_nm=80.0,
        target_response=FieldResponse.REFLECT,
        target_x=0.15, target_y=0.10,
        angular_behavior=AngularBehavior.NON_IRIDESCENT,
        dimensionality="3D",
        allowed_materials=["SiO2", "TiO2_rutile", "polystyrene"],
        substrate="SiO2",
        target_application=ApplicationContext.RESEARCH,
    )


def run_isomorphism_demo() -> dict:
    mol_spec = make_cu2_pocket_spec()
    opt_spec = make_blue_structural_color_spec()
    mol_result = dispatch_physics(mol_spec)
    opt_result = dispatch_physics(opt_spec)
    return {
        "molecular": {"spec_type": mol_spec.spec_type, "physics_result": mol_result,
                       "realization_options": get_realization_options(mol_spec)},
        "optical": {"spec_type": opt_spec.spec_type, "physics_result": opt_result,
                     "realization_options": get_realization_options(opt_spec)},
        "isomorphism_check": {
            "same_dispatch_function": True,
            "same_base_class": isinstance(mol_spec, InteractionSpec) and isinstance(opt_spec, InteractionSpec),
            "no_domain_branching_in_dispatch": (
                "dispatch_physics uses spec.spec_type for routing — "
                "no isinstance() checks on DiscretePocketSpec or FieldInteractionSpec"
            ),
        },
    }
