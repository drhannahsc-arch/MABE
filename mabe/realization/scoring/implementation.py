"""
Implementation Scoring — SECONDARY axes.

These only matter after physics fidelity. They break ties between
material systems that achieve similar physics fidelity.

All return 0.0–1.0, higher = better.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import InteractionGeometrySpec
    from mabe.realization.registry.material_registry import MaterialCapability


def score_synthetic_accessibility(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    """
    How hard is it to make?

    Proxy: inverse of typical synthesis complexity.
    Will be replaced by per-adapter SA estimates in Sprint R2+.
    """
    # Simple heuristic from registry data
    # More design tools = easier to design
    tool_bonus = min(1.0, len(cap.design_tools_available) * 0.2)

    # Higher validation rate = more likely to succeed on first try
    success_bonus = cap.literature_validation_rate

    # More literature = better understood synthesis
    lit_bonus = min(1.0, math.log10(max(1, cap.literature_examples)) / 5.0)

    return (tool_bonus + success_bonus + lit_bonus) / 3.0


def score_cost(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    """
    Cost score at required scale.

    If cost ceiling specified: hard fail below threshold.
    Otherwise: soft logarithmic penalty.
    """
    # Use midpoint of cost range as estimate
    estimated_cost = (cap.cost_per_unit_range[0] + cap.cost_per_unit_range[1]) / 2.0

    if spec.cost_ceiling_per_unit is not None:
        if estimated_cost > spec.cost_ceiling_per_unit:
            return 0.0
        return 1.0 - (estimated_cost / spec.cost_ceiling_per_unit)

    # Soft penalty: cheaper is better
    return 1.0 / (1.0 + math.log10(max(1.0, estimated_cost)))


def score_scalability(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    """Can it be produced at the required scale?"""
    from mabe.realization.models import ScaleClass

    try:
        required = spec.required_scale.rank
        max_cap = ScaleClass(cap.max_practical_scale).rank
    except (ValueError, AttributeError):
        return 0.5  # can't evaluate, neutral

    if required > max_cap:
        return 0.0  # should have been caught by feasibility gate

    headroom = max_cap - required
    # More headroom = better (easier to scale)
    return min(1.0, 0.5 + 0.1 * headroom)


def score_operating_conditions(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    """Does the material survive the operating environment?"""
    score = 1.0

    # pH headroom
    ph_margin_low = spec.pH_range[0] - cap.pH_stability[0]
    ph_margin_high = cap.pH_stability[1] - spec.pH_range[1]
    ph_margin = min(ph_margin_low, ph_margin_high)
    if ph_margin < 0:
        return 0.0  # should have been caught by gate
    if ph_margin < 1.0:
        score *= 0.5 + 0.5 * ph_margin  # penalize tight margins

    # Temperature headroom
    temp_margin = cap.thermal_stability_K[1] - spec.temperature_range_K[1]
    if temp_margin < 0:
        return 0.0
    if temp_margin < 50:
        score *= 0.5 + 0.01 * temp_margin

    return score


def score_reusability(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    """
    Can the pocket be regenerated and reused?

    Rigid, chemically stable systems score higher.
    """
    # Rigid materials are more reusable
    rigidity_map = {"rigid": 1.0, "semi-rigid": 0.7, "semi-flexible": 0.4, "flexible": 0.2}
    rigidity_score = rigidity_map.get(cap.rigidity_range[1], 0.5)

    # Wider operating envelope = more robust to regeneration conditions
    ph_range = cap.pH_stability[1] - cap.pH_stability[0]
    robustness = min(1.0, ph_range / 10.0)

    return (rigidity_score + robustness) / 2.0
