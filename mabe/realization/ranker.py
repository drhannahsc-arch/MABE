"""
Application-Aware Ranker — Layer 3 Integration.

Scores all registered adapters against an InteractionGeometrySpec,
using application-appropriate weight profiles. Handles both precision
binders (Class A: CD, crown, porphyrin) and bulk sorbents (Class D:
lignin, SAP, MIP, resin) on the same composite scale.

Key design decision: the ranker doesn't privilege precision over capacity.
A remediation spec weights cost_per_kg_processed and capacity heavily,
causing a bulk sorbent to outscore a precision binder even though the
binder has higher physics_fidelity.

Output groups results by physics class so the user sees:
    "Best precision binder: β-CD (composite=0.72)"
    "Best bulk sorbent: dithiocarbamate-lignin (composite=0.85)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.models import (
    ApplicationContext,
    InteractionGeometrySpec,
    RealizationScore,
    RankedRealizations,
)


# ─────────────────────────────────────────────
# Weight profiles
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class WeightProfile:
    """Application-specific scoring weights."""

    name: str

    # Precision axes
    w_physics_fidelity: float = 0.0
    w_synthetic_accessibility: float = 0.0
    w_cost_score: float = 0.0
    w_scalability: float = 0.0
    w_operating_conditions: float = 0.0
    w_reusability: float = 0.0

    # Bulk sorbent axes
    w_capacity: float = 0.0
    w_selectivity: float = 0.0
    w_throughput: float = 0.0
    w_regenerability: float = 0.0
    w_cost_per_kg: float = 0.0

    def total(self) -> float:
        return (
            self.w_physics_fidelity + self.w_synthetic_accessibility +
            self.w_cost_score + self.w_scalability +
            self.w_operating_conditions + self.w_reusability +
            self.w_capacity + self.w_selectivity +
            self.w_throughput + self.w_regenerability +
            self.w_cost_per_kg
        )


WEIGHT_PROFILES: dict[str, WeightProfile] = {
    "research": WeightProfile(
        name="research",
        w_physics_fidelity=0.40,
        w_synthetic_accessibility=0.15,
        w_cost_score=0.10,
        w_scalability=0.05,
        w_operating_conditions=0.05,
        w_reusability=0.05,
        w_selectivity=0.15,
        w_capacity=0.05,
    ),
    "diagnostic": WeightProfile(
        name="diagnostic",
        w_physics_fidelity=0.30,
        w_selectivity=0.30,
        w_synthetic_accessibility=0.10,
        w_cost_score=0.05,
        w_scalability=0.05,
        w_operating_conditions=0.10,
        w_reusability=0.10,
    ),
    "separation": WeightProfile(
        name="separation",
        w_selectivity=0.25,
        w_capacity=0.20,
        w_throughput=0.15,
        w_regenerability=0.15,
        w_cost_per_kg=0.10,
        w_physics_fidelity=0.10,
        w_scalability=0.05,
    ),
    "remediation": WeightProfile(
        name="remediation",
        w_cost_per_kg=0.25,
        w_capacity=0.25,
        w_throughput=0.15,
        w_regenerability=0.10,
        w_selectivity=0.10,
        w_scalability=0.10,
        w_physics_fidelity=0.05,
    ),
}


def get_weight_profile(application: ApplicationContext) -> WeightProfile:
    """Map ApplicationContext enum to weight profile."""
    mapping = {
        ApplicationContext.RESEARCH: "research",
        ApplicationContext.DIAGNOSTIC: "diagnostic",
        ApplicationContext.SEPARATION: "separation",
        ApplicationContext.REMEDIATION: "remediation",
    }
    profile_name = mapping.get(application, "research")
    return WEIGHT_PROFILES[profile_name]


# ─────────────────────────────────────────────
# Composite scoring
# ─────────────────────────────────────────────

def compute_composite(score: RealizationScore, weights: WeightProfile) -> float:
    """
    Compute application-weighted composite score.

    All axes are 0.0–1.0 normalized. Bulk sorbent fields are
    normalized against reference values before weighting.
    """
    # Precision axes (already 0–1)
    composite = 0.0
    composite += weights.w_physics_fidelity * score.physics_fidelity
    composite += weights.w_synthetic_accessibility * score.synthetic_accessibility
    composite += weights.w_cost_score * score.cost_score
    composite += weights.w_scalability * score.scalability
    composite += weights.w_operating_conditions * score.operating_condition_compatibility
    composite += weights.w_reusability * score.reusability_score

    # Bulk sorbent axes — normalize against reference values
    # Capacity: 5 mmol/g is excellent for most sorbents
    cap_norm = min(1.0, score.capacity_mmol_per_g / 5.0) if score.capacity_mmol_per_g > 0 else 0.0
    composite += weights.w_capacity * cap_norm

    # Selectivity: log scale, 100× is excellent
    import math
    sel_norm = min(1.0, math.log10(max(1.0, score.selectivity_factor)) / 2.0)
    composite += weights.w_selectivity * sel_norm

    # Throughput: 100 L/h/kg is excellent for column operation
    tp_norm = min(1.0, score.throughput_L_per_h_per_kg / 100.0) if score.throughput_L_per_h_per_kg > 0 else 0.0
    composite += weights.w_throughput * tp_norm

    # Regenerability: 50 cycles is excellent
    regen_norm = min(1.0, score.regenerability_cycles / 50.0) if score.regenerability_cycles > 0 else 0.0
    composite += weights.w_regenerability * regen_norm

    # Cost per kg processed: inverted (lower is better). $10/kg is excellent.
    if score.cost_per_kg_processed > 0:
        cost_norm = min(1.0, 10.0 / score.cost_per_kg_processed)
    else:
        cost_norm = 0.0
    composite += weights.w_cost_per_kg * cost_norm

    return composite


# ─────────────────────────────────────────────
# Ranked output with physics class grouping
# ─────────────────────────────────────────────

@dataclass
class GroupedRanking:
    """Rankings grouped by physics class."""
    physics_class: str
    class_label: str             # human-readable: "Precision Binder", "Bulk Sorbent"
    rankings: list[RealizationScore]
    best: Optional[RealizationScore] = None


@dataclass
class RankerOutput:
    """Full ranker output with application context."""

    spec: InteractionGeometrySpec
    application: ApplicationContext
    weight_profile: WeightProfile

    # ── All scores, sorted by composite ──
    all_rankings: list[RealizationScore]

    # ── Grouped by physics class ──
    groups: list[GroupedRanking]

    # ── Overall recommendation ──
    recommended_system: str
    recommended_adapter: str
    recommendation_rationale: str


# ─────────────────────────────────────────────
# Ranker
# ─────────────────────────────────────────────

PHYSICS_CLASS_LABELS = {
    "covalent_cavity": "Precision Binder",
    "bulk_sorbent": "Bulk Sorbent",
    "periodic_lattice": "Periodic Framework",
    "foldable_polymer": "Foldable Polymer",
    "emergent_cavity": "Self-Assembled Cage",
}


class AdapterRanker:
    """
    Application-aware ranker. Scores all registered adapters and
    groups results by physics class.
    """

    def __init__(self, adapters: list = None):
        """
        Args:
            adapters: list of RealizationAdapter instances to rank.
                      If None, auto-discover from AdapterRegistry.
        """
        self._adapters = adapters or []

    def add_adapter(self, adapter) -> None:
        self._adapters.append(adapter)

    def rank(
        self,
        spec: InteractionGeometrySpec,
        application: Optional[ApplicationContext] = None,
        custom_weights: Optional[WeightProfile] = None,
    ) -> RankerOutput:
        """
        Score all adapters against spec, return grouped rankings.

        Args:
            spec: the interaction geometry to realize
            application: if None, uses spec.target_application
            custom_weights: override default weight profile
        """
        app = application or spec.target_application or ApplicationContext.RESEARCH
        weights = custom_weights or get_weight_profile(app)

        # ── Score all adapters ──
        scores: list[RealizationScore] = []
        for adapter in self._adapters:
            try:
                score = adapter.estimate_fidelity(spec)
                if not score.physics_class:
                    score.physics_class = adapter.capability.physics_class
                score.composite_score = compute_composite(score, weights)
                scores.append(score)
            except Exception:
                # Adapter failed — skip, don't crash the ranker
                continue

        # ── Sort by composite ──
        scores.sort(key=lambda s: s.composite_score, reverse=True)

        # ── Group by physics class ──
        class_groups: dict[str, list[RealizationScore]] = {}
        for s in scores:
            cls = s.physics_class or "unknown"
            class_groups.setdefault(cls, []).append(s)

        groups = []
        for cls, cls_scores in class_groups.items():
            cls_scores.sort(key=lambda s: s.composite_score, reverse=True)
            feasible = [s for s in cls_scores if s.feasible]
            best = feasible[0] if feasible else None
            groups.append(GroupedRanking(
                physics_class=cls,
                class_label=PHYSICS_CLASS_LABELS.get(cls, cls),
                rankings=cls_scores,
                best=best,
            ))

        # Sort groups so highest-scoring class comes first
        groups.sort(
            key=lambda g: g.best.composite_score if g.best else 0.0,
            reverse=True,
        )

        # ── Recommendation ──
        feasible_scores = [s for s in scores if s.feasible]
        if feasible_scores:
            best = feasible_scores[0]
            rec_system = best.material_system
            rec_adapter = best.adapter_id
            rec_rationale = _build_rationale(best, weights, groups)
        else:
            rec_system = "none"
            rec_adapter = "none"
            rec_rationale = "No feasible adapter found."

        return RankerOutput(
            spec=spec,
            application=app,
            weight_profile=weights,
            all_rankings=scores,
            groups=groups,
            recommended_system=rec_system,
            recommended_adapter=rec_adapter,
            recommendation_rationale=rec_rationale,
        )


def _build_rationale(
    best: RealizationScore,
    weights: WeightProfile,
    groups: list[GroupedRanking],
) -> str:
    """Build human-readable recommendation rationale."""
    parts = [
        f"Recommended: {best.material_system} "
        f"(composite={best.composite_score:.3f}, "
        f"profile={weights.name})"
    ]

    if best.advantages:
        parts.append(f"Strengths: {'; '.join(best.advantages[:3])}")

    # Note if another physics class is competitive
    for g in groups:
        if g.best and g.best.material_system != best.material_system:
            delta = best.composite_score - g.best.composite_score
            if delta < 0.15:
                parts.append(
                    f"Note: {g.class_label} option "
                    f"({g.best.material_system}, "
                    f"composite={g.best.composite_score:.3f}) "
                    f"is competitive"
                )

    return " | ".join(parts)
