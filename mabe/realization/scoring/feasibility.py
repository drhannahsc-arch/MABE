"""
Feasibility Gate.

Binary pass/fail. If this fails, the material system is not scored.
Prevents wasting compute on impossible realizations.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import InteractionGeometrySpec
    from mabe.realization.registry.material_registry import MaterialCapability

from mabe.realization.models import ScaleClass


def feasibility_gate(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> tuple[bool, Optional[str]]:
    """
    Binary: can this material system even attempt this geometry?

    Returns (True, None) if feasible, (False, reason) if not.
    """

    # ── Pocket too small? ──
    if spec.pocket_scale_nm < cap.min_pocket_size_nm:
        return False, (
            f"Pocket {spec.pocket_scale_nm:.2f} nm below minimum "
            f"{cap.min_pocket_size_nm:.2f} nm for {cap.system_id}"
        )

    # ── Pocket too large? ──
    if spec.pocket_scale_nm > cap.max_pocket_size_nm:
        return False, (
            f"Pocket {spec.pocket_scale_nm:.2f} nm above maximum "
            f"{cap.max_pocket_size_nm:.2f} nm for {cap.system_id}"
        )

    # ── Required donors not available? ──
    required = spec.required_donor_types
    available = set(cap.donor_types_available)
    missing = required - available
    if missing:
        return False, (
            f"Required donors {missing} not available in {cap.system_id} "
            f"(available: {available})"
        )

    # ── Too many donors? ──
    if len(spec.donor_positions) > cap.max_donor_count:
        return False, (
            f"Spec requires {len(spec.donor_positions)} donors, "
            f"{cap.system_id} supports max {cap.max_donor_count}"
        )

    # ── pH incompatible? ──
    if spec.pH_range[0] < cap.pH_stability[0] or spec.pH_range[1] > cap.pH_stability[1]:
        return False, (
            f"Required pH {spec.pH_range} outside "
            f"{cap.system_id} stability {cap.pH_stability}"
        )

    # ── Temperature incompatible? ──
    if spec.temperature_range_K[1] > cap.thermal_stability_K[1]:
        return False, (
            f"Required temp {spec.temperature_range_K[1]:.0f} K exceeds "
            f"{cap.system_id} max {cap.thermal_stability_K[1]:.0f} K"
        )

    # ── Solvent incompatible? ──
    if spec.solvent.value not in cap.solvent_compatibility:
        return False, (
            f"Solvent {spec.solvent.value} not compatible with "
            f"{cap.system_id} (supports: {cap.solvent_compatibility})"
        )

    # ── Scale impossible? ──
    spec_rank = ScaleClass(spec.required_scale).rank if isinstance(spec.required_scale, str) else spec.required_scale.rank
    max_rank = ScaleClass(cap.max_practical_scale).rank if isinstance(cap.max_practical_scale, str) else -1
    # Graceful handling: skip scale check if we can't parse
    try:
        max_cap = ScaleClass(cap.max_practical_scale)
        if spec.required_scale.rank > max_cap.rank:
            return False, (
                f"Required scale {spec.required_scale.value} exceeds "
                f"{cap.system_id} max {cap.max_practical_scale}"
            )
    except (ValueError, AttributeError):
        pass  # skip scale gate if values don't parse cleanly

    return True, None
