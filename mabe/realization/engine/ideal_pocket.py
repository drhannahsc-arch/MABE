"""
Phase 1: Ideal Pocket Computation.

Pure physics. No material constraints. No synthetic accessibility.
No cost. What would the pocket look like if you could place atoms
arbitrarily in space?

Input:  InteractionGeometrySpec (from Layer 2)
Output: IdealPocketSpec (the reference standard)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import InteractionGeometrySpec, IdealPocketSpec

from mabe.realization.models import (
    CavityShape,
    IdealElement,
    IdealPocketSpec,
    RigidityClass,
)


def compute_ideal_pocket(spec: "InteractionGeometrySpec") -> IdealPocketSpec:
    """
    Compute the physics-optimal pocket for a given interaction geometry.

    This is the reference standard that all material systems are scored against.

    Steps:
        1. Optimize element positions from donor spec
        2. Compute per-element precision requirements (from energy gradients)
        3. Compute desolvation penalty
        4. Compute ideal binding energy (sum of interactions − desolvation)
        5. Derive rigidity class from tightest tolerance
        6. Package as IdealPocketSpec
    """

    ideal_elements = _compute_ideal_elements(spec)
    desolvation = _compute_desolvation_penalty(spec)
    binding_energy = _compute_ideal_binding_energy(ideal_elements, desolvation)
    rigidity_class = _classify_rigidity(ideal_elements)

    return IdealPocketSpec(
        optimal_elements=ideal_elements,
        ideal_cavity_volume_A3=spec.cavity_dimensions.volume_A3,
        ideal_cavity_shape=spec.cavity_shape,
        ideal_desolvation_energy_kJ_mol=desolvation,
        ideal_binding_energy_kJ_mol=binding_energy,
        min_precision_required_A=_tightest_precision(ideal_elements),
        rigidity_class=rigidity_class,
        min_stability_pH=spec.pH_range,
        min_stability_K=spec.temperature_range_K,
        required_elements={e.atom_type for e in ideal_elements},
        symmetry_exploitable=spec.symmetry != "none",
        ideal_material_requirements=_describe_ideal_material(ideal_elements, rigidity_class),
        critical_constraints=_derive_critical_constraints(spec, ideal_elements),
    )


# ─────────────────────────────────────────────
# Internal computation functions
# ─────────────────────────────────────────────

def _compute_ideal_elements(spec: "InteractionGeometrySpec") -> list[IdealElement]:
    """
    For each donor in the spec, compute the physics-optimal placement.

    Current implementation: direct pass-through of spec positions with
    precision derived from interaction type. This will be replaced with
    energy-gradient-based optimization as BackSolve data integrates.
    """
    elements = []
    for donor in spec.donor_positions:
        precision = _precision_for_interaction_type(
            donor.atom_type,
            donor.coordination_role,
        )
        energy = _energy_contribution(
            donor.atom_type,
            donor.coordination_role,
            donor.charge_state,
        )
        elements.append(IdealElement(
            atom_type=donor.atom_type,
            exact_position_A=donor.position_vector_A,
            required_precision_A=precision,
            orbital_hybridization=donor.required_hybridization,
            charge_state=donor.charge_state,
            interaction_energy_contribution_kJ_mol=energy,
        ))
    return elements


def _precision_for_interaction_type(atom_type: str, coordination_role: str) -> float:
    """
    How precisely must this element be placed?

    Derived from the steepness of the interaction energy gradient.
    Steeper gradient = tighter tolerance = higher precision requirement.

    Values from metal-ligand coordination literature:
        - Terminal coordination bond: 0.05 Å (steep Morse well)
        - Bridging coordination: 0.10 Å
        - H-bond donor/acceptor: 0.15 Å (broader well)
        - Hydrophobic contact: 0.30 Å (very broad)
    """
    # Coordination bonds: tight tolerance
    if coordination_role in ("axial", "equatorial", "terminal"):
        if atom_type in ("N", "S", "Se"):
            return 0.05
        elif atom_type == "O":
            return 0.08  # O-donors have more variability
        else:
            return 0.05

    # Bridging: slightly looser
    if coordination_role == "bridging":
        return 0.10

    # H-bond: moderate
    if coordination_role in ("h_bond_donor", "h_bond_acceptor"):
        return 0.15

    # Hydrophobic: broad
    if coordination_role == "hydrophobic":
        return 0.30

    # Default
    return 0.10


def _energy_contribution(
    atom_type: str,
    coordination_role: str,
    charge_state: float,
) -> float:
    """
    Estimated interaction energy contribution of one element.

    Placeholder values from NIST / literature. Will be replaced by
    BackSolve-calibrated per-interaction energies.

    Returns negative values (stabilizing).
    """
    # Metal-N coordination: ~-40 to -80 kJ/mol per bond
    # Metal-O coordination: ~-30 to -60 kJ/mol per bond
    # Metal-S coordination: ~-50 to -100 kJ/mol per bond
    # H-bond: ~-10 to -30 kJ/mol
    # CH-π / hydrophobic: ~-5 to -15 kJ/mol

    base_energies = {
        "N": -50.0,
        "O": -40.0,
        "S": -70.0,
        "Se": -60.0,
        "P": -45.0,
    }
    base = base_energies.get(atom_type, -30.0)

    # Coordination bonds are stronger than H-bonds
    role_scale = {
        "axial": 1.0,
        "equatorial": 1.0,
        "terminal": 1.0,
        "bridging": 0.8,
        "h_bond_donor": 0.4,
        "h_bond_acceptor": 0.4,
        "hydrophobic": 0.2,
    }
    scale = role_scale.get(coordination_role, 0.5)

    return base * scale


def _compute_desolvation_penalty(spec: "InteractionGeometrySpec") -> float:
    """
    Estimated desolvation cost for creating this pocket.

    Uses Eisenberg-McLachlan transfer coefficients as baseline.
    Aqueous desolvation is expensive; organic is cheap.
    """
    if spec.solvent == "gas":
        return 0.0

    # Rough estimate: ~0.1 kJ/mol per Å³ of cavity in water
    # (from hydrophobic transfer free energy literature)
    base_cost_per_A3 = 0.10 if spec.solvent.value == "aqueous" else 0.03
    cavity_cost = spec.cavity_dimensions.volume_A3 * base_cost_per_A3

    # Each donor that must be desolvated adds penalty
    donor_desolv = sum(
        _donor_desolvation_cost(d.atom_type, spec.solvent.value)
        for d in spec.donor_positions
    )

    return cavity_cost + donor_desolv


def _donor_desolvation_cost(atom_type: str, solvent: str) -> float:
    """Cost to strip solvent from one donor element."""
    if solvent != "aqueous":
        return 2.0  # minimal in organic
    # Aqueous desolvation costs from hydration free energy data
    costs = {
        "N": 12.0,   # amine hydration is moderate
        "O": 15.0,   # hydroxyl / carboxylate hydration is strong
        "S": 8.0,    # thiol / thioether hydration is weak
        "Se": 6.0,
        "P": 10.0,
    }
    return costs.get(atom_type, 10.0)


def _compute_ideal_binding_energy(
    elements: list[IdealElement],
    desolvation: float,
) -> float:
    """
    Ideal binding energy = sum of interaction contributions − desolvation.

    Negative = favorable binding.
    """
    total_interaction = sum(e.interaction_energy_contribution_kJ_mol for e in elements)
    return total_interaction + desolvation  # desolvation is positive (penalty)


def _tightest_precision(elements: list[IdealElement]) -> float:
    if not elements:
        return float("inf")
    return min(e.required_precision_A for e in elements)


def _classify_rigidity(elements: list[IdealElement]) -> RigidityClass:
    """Derive rigidity class from tightest precision requirement."""
    tightest = _tightest_precision(elements)
    if tightest < 0.05:
        return RigidityClass.CRYSTALLINE
    elif tightest < 0.20:
        return RigidityClass.PREORGANIZED
    elif tightest < 0.50:
        return RigidityClass.SEMI_FLEXIBLE
    else:
        return RigidityClass.ANY


def _describe_ideal_material(
    elements: list[IdealElement],
    rigidity: RigidityClass,
) -> str:
    """Human-readable description of what would score 1.0."""
    element_types = sorted({e.atom_type for e in elements})
    tightest = _tightest_precision(elements)
    return (
        f"Requires {len(elements)} interaction elements ({', '.join(element_types)}) "
        f"positioned to ±{tightest:.2f} Å. "
        f"Rigidity class: {rigidity.value}. "
        f"Material must provide all element types with covalent or "
        f"coordination-level positional control."
    )


def _derive_critical_constraints(
    spec: "InteractionGeometrySpec",
    elements: list[IdealElement],
) -> list[str]:
    """Non-negotiable requirements for any realization."""
    constraints = []

    # Element availability
    for e in elements:
        constraints.append(
            f"Must provide {e.atom_type} at {e.exact_position_A} "
            f"± {e.required_precision_A:.2f} Å"
        )

    # Exclusion constraints
    for exc in spec.must_exclude:
        constraints.append(
            f"Must exclude {exc.species} (max affinity: "
            f"{exc.max_allowed_affinity_kJ_mol:.1f} kJ/mol, "
            f"mechanism: {exc.exclusion_mechanism})"
        )

    # Operating conditions
    constraints.append(
        f"Must be stable at pH {spec.pH_range[0]:.1f}–{spec.pH_range[1]:.1f}, "
        f"{spec.temperature_range_K[0]:.0f}–{spec.temperature_range_K[1]:.0f} K, "
        f"in {spec.solvent.value}"
    )

    return constraints
