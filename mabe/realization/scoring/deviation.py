"""
Deviation Scoring — PRIMARY axis.

Measures how far a material system deviates from the IdealPocketSpec.
This is the most important scoring function. Everything else is secondary.

Physics fidelity = exp(-mean_deviation / decay_constant)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.registry.material_registry import MaterialCapability

from mabe.realization.models import DeviationReport, IdealPocketSpec, RigidityClass


# Exponential decay constant for fidelity conversion.
# 0.1 Å mean deviation → 0.82 fidelity
# 0.5 Å mean deviation → 0.37 fidelity
# 1.0 Å mean deviation → 0.14 fidelity
FIDELITY_DECAY_A = 0.5


def compute_deviation(
    ideal: IdealPocketSpec,
    cap: "MaterialCapability",
) -> DeviationReport:
    """
    For each element in the ideal spec, compute how far this material
    system can get from the ideal position.

    The deviation at each element is bounded below by the material's
    positioning precision — it cannot do better than its intrinsic limit.
    """

    element_deviations: list[float] = []
    missing: list[str] = []

    for element in ideal.optimal_elements:
        # Can this material provide this atom type at all?
        if element.atom_type not in cap.donor_types_available:
            element_deviations.append(float("inf"))
            missing.append(
                f"{element.atom_type} at {element.exact_position_A}"
            )
            continue

        # Best achievable deviation = material's positioning precision
        # The material cannot place an element more precisely than this
        achievable = cap.positioning_precision_A

        # If the material's precision is sufficient, it can meet the spec
        # The deviation is the precision limit itself (best case)
        element_deviations.append(achievable)

    # Rigidity deviation
    rigidity_dev = _rigidity_deviation(ideal.rigidity_class, cap.rigidity_range)

    finite_devs = [d for d in element_deviations if d < float("inf")]

    return DeviationReport(
        material_system=cap.system_id,
        element_deviations_A=element_deviations,
        max_deviation_A=max(element_deviations) if element_deviations else float("inf"),
        mean_deviation_A=(
            sum(finite_devs) / len(finite_devs) if finite_devs else float("inf")
        ),
        rigidity_deviation=rigidity_dev,
        missing_interactions=missing,
    )


def deviation_to_fidelity(dev: DeviationReport) -> float:
    """
    Convert deviation report to 0–1 fidelity score.

    Missing interactions → 0.0
    Otherwise: exp(-mean_deviation / decay_constant) * rigidity_factor
    """
    if dev.missing_interactions:
        return 0.0

    if dev.mean_deviation_A == float("inf") or dev.mean_deviation_A < 0:
        return 0.0

    position_fidelity = math.exp(-dev.mean_deviation_A / FIDELITY_DECAY_A)

    # Rigidity mismatch penalizes further
    rigidity_factor = 1.0 - 0.3 * dev.rigidity_deviation  # max 30% penalty

    return max(0.0, min(1.0, position_fidelity * rigidity_factor))


def _rigidity_deviation(
    required: RigidityClass,
    material_range: tuple[str, str],
) -> float:
    """
    How far is the material's rigidity from the requirement?
    0.0 = perfect match or MORE rigid than required, 1.0 = too flexible.

    Asymmetric: too rigid is fine (or slightly beneficial).
    Too flexible is penalized — a flexible material can't reproduce
    a pocket that requires preorganized geometry.
    """
    rigidity_order = {
        "flexible": 0,
        "semi-rigid": 1,
        "semi-flexible": 1,
        "preorganized": 2,
        "rigid": 3,
        "crystalline": 3,
    }

    required_level = rigidity_order.get(required.value, 1)

    mat_min = rigidity_order.get(material_range[0], 1)
    mat_max = rigidity_order.get(material_range[1], 1)

    # Material CAN reach the required rigidity or higher → no penalty
    if mat_max >= required_level:
        return 0.0

    # Material is too flexible — can't achieve required rigidity
    shortfall = required_level - mat_max
    return min(1.0, shortfall / 3.0)
