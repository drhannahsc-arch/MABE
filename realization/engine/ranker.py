"""
Phase 2: Material System Ranking.

Scores every registered material system against the IdealPocketSpec.
Physics fidelity is the primary axis. Implementation concerns are secondary.

Input:  IdealPocketSpec + InteractionGeometrySpec
Output: RankedRealizations (sorted, with gap analysis)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from mabe.realization.models import (
    ApplicationContext,
    DeviationReport,
    InteractionGeometrySpec,
    IdealPocketSpec,
    RankedRealizations,
    RealizationScore,
)
from mabe.realization.registry.material_registry import MATERIAL_REGISTRY
from mabe.realization.scoring.feasibility import feasibility_gate
from mabe.realization.scoring.deviation import compute_deviation, deviation_to_fidelity
from mabe.realization.scoring.implementation import (
    score_synthetic_accessibility,
    score_cost,
    score_scalability,
    score_operating_conditions,
    score_reusability,
)
from mabe.realization.scoring.composite import compute_composite
from mabe.realization.scoring.gap_report import generate_gap_report, generate_novel_material_spec


def rank_realizations(
    ideal: IdealPocketSpec,
    spec: InteractionGeometrySpec,
) -> RankedRealizations:
    """
    Score all registered material systems against the ideal pocket.

    Physics fidelity always dominates (60% of composite).
    Implementation factors split the remaining 40% based on application context.
    """

    scores: list[RealizationScore] = []

    for cap in MATERIAL_REGISTRY.all():
        # ── Hard gate ──
        feasible, reason = feasibility_gate(spec, cap)
        if not feasible:
            scores.append(RealizationScore(
                material_system=cap.system_id,
                adapter_id=cap.adapter_class,
                deviation_from_ideal=DeviationReport(
                    material_system=cap.system_id,
                    element_deviations_A=[],
                    max_deviation_A=float("inf"),
                    mean_deviation_A=float("inf"),
                ),
                physics_fidelity=0.0,
                feasible=False,
                infeasibility_reason=reason,
            ))
            continue

        # ── Deviation from ideal ──
        deviation = compute_deviation(ideal, cap)
        physics_fidelity = deviation_to_fidelity(deviation)

        # ── Implementation scores ──
        sa = score_synthetic_accessibility(spec, cap)
        cost = score_cost(spec, cap)
        scale = score_scalability(spec, cap)
        conditions = score_operating_conditions(spec, cap)
        reuse = score_reusability(spec, cap) if spec.reusability_required else 1.0

        # ── Confidence (calibrated, not precedent-biased) ──
        confidence = _calibrate_confidence(cap, deviation)

        # ── Composite ──
        composite = compute_composite(
            physics_fidelity=physics_fidelity,
            synthetic_accessibility=sa,
            cost_score=cost,
            scalability=scale,
            operating_condition_compatibility=conditions,
            reusability_score=reuse,
            application=spec.target_application,
        )

        scores.append(RealizationScore(
            material_system=cap.system_id,
            adapter_id=cap.adapter_class,
            deviation_from_ideal=deviation,
            physics_fidelity=physics_fidelity,
            synthetic_accessibility=sa,
            cost_score=cost,
            scalability=scale,
            operating_condition_compatibility=conditions,
            reusability_score=reuse,
            composite_score=composite,
            confidence=confidence,
            advantages=cap.known_strengths,
            limitations=cap.known_limitations,
            feasible=True,
        ))

    # ── Sort by composite ──
    scores.sort(key=lambda s: s.composite_score, reverse=True)

    # ── Gap analysis ──
    feasible_scores = [s for s in scores if s.feasible]
    best_fidelity = max((s.physics_fidelity for s in feasible_scores), default=0.0)
    gap = 1.0 - best_fidelity

    gap_report = None
    novel_suggestion = None
    if gap > 0.3:
        gap_report = generate_gap_report(ideal, feasible_scores)
    if gap > 0.5:
        novel_suggestion = generate_novel_material_spec(ideal, feasible_scores)

    recommended = feasible_scores[0].material_system if feasible_scores else "none"
    rationale = _build_rationale(feasible_scores, gap) if feasible_scores else "No feasible material system found."

    return RankedRealizations(
        geometry_spec=spec,
        ideal_pocket=ideal,
        rankings=scores,
        recommended=recommended,
        recommendation_rationale=rationale,
        best_physics_fidelity=best_fidelity,
        gap_to_ideal=gap,
        gap_report=gap_report,
        novel_material_suggestion=novel_suggestion,
    )


def _calibrate_confidence(cap, deviation: DeviationReport) -> float:
    """
    How likely is it that a design in this material system will actually work?
    Based on published design-to-validation success rates.
    Not a scoring bonus for familiarity — a calibrated risk estimate.
    """
    base_rate = cap.literature_validation_rate
    # Operating near precision limit = higher risk of failure
    if deviation.max_deviation_A < float("inf") and cap.positioning_precision_A > 0:
        utilization = deviation.max_deviation_A / cap.positioning_precision_A
        if utilization > 0.8:
            return base_rate * 0.5
    return base_rate


def _build_rationale(scores: list[RealizationScore], gap: float) -> str:
    """Build human-readable recommendation rationale."""
    if not scores:
        return "No feasible material systems."
    top = scores[0]
    parts = [
        f"{top.material_system} recommended (physics fidelity: {top.physics_fidelity:.2f}, "
        f"composite: {top.composite_score:.2f})."
    ]
    if gap > 0.3:
        parts.append(f"Gap to ideal: {gap:.2f}. See gap report for improvement opportunities.")
    if len(scores) > 1:
        runner = scores[1]
        parts.append(
            f"Runner-up: {runner.material_system} "
            f"(fidelity: {runner.physics_fidelity:.2f})."
        )
    return " ".join(parts)