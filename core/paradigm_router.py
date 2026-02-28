"""
Paradigm Router + Cross-Paradigm Ranker — Polymorphic Integration Layer.

Routes InteractionSpec objects to the correct adapter pipeline(s) and
enables cross-paradigm comparison of results.

Design principles (from MABE_Polymorphic_InteractionSpec_DataIntegrity.md):
    - Router is PERMISSIVE: generate specs for all non-infeasible paradigms
    - Let scoring + ranking determine the winner, not premature filtering
    - Each paradigm scores independently with its own physics
    - Cross-paradigm comparison normalizes on APPLICATION-RELEVANT axes
    - Confidence calibration penalizes paradigms with less data

Architecture:
    InteractionSpec (any subtype)
        → ParadigmRouter.route()
            → [adapter1.score(), adapter2.score(), ...]
        → CrossParadigmRanker.rank()
            → ParadigmComparisonResult (normalized, grouped, ranked)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from mabe.realization.models import (
    InteractionSpec,
    InteractionParadigm,
    DiscretePocketSpec,
    NetworkInteractionSpec,
    NetworkMechanism,
    ResinType,
    SurfaceInteractionSpec,
    SurfaceMechanism,
    BaseMaterial,
    IsothermModel,
    ApplicationContext,
    ScaleClass,
    ExclusionSpec,
    FabricationSpec,
)


# ─────────────────────────────────────────────
# Adapter protocol
# ─────────────────────────────────────────────

@runtime_checkable
class ParadigmAdapter(Protocol):
    """Protocol that all paradigm adapters must satisfy."""
    system_id: str
    supported_spec_types: list[str]

    def score(self, spec: InteractionSpec) -> dict: ...
    def design(self, spec: InteractionSpec) -> FabricationSpec: ...


# ─────────────────────────────────────────────
# Adapter result (normalized)
# ─────────────────────────────────────────────

@dataclass
class AdapterResult:
    """Normalized result from a single adapter evaluation."""

    paradigm: str                   # "pocket", "network", "surface"
    adapter_id: str                 # "ion_exchange_resin", "activated_carbon", etc.
    feasible: bool
    reason: str = ""                # if not feasible

    # ── Normalized scores (0-1) ──
    composite_score: float = 0.0
    selectivity_score: float = 0.0
    capacity_score: float = 0.0
    confidence: float = 0.0

    # ── Calibration ──
    calibration_status: str = "uncalibrated"

    # ── Raw adapter output (paradigm-specific details) ──
    raw_scores: dict = field(default_factory=dict)

    # ── Confidence-weighted composite ──
    @property
    def weighted_score(self) -> float:
        """Composite × confidence = decision score."""
        return self.composite_score * self.confidence


# ─────────────────────────────────────────────
# Paradigm comparison result
# ─────────────────────────────────────────────

@dataclass
class ParadigmGroup:
    """All results from one paradigm."""
    paradigm: str
    results: list[AdapterResult]
    best: Optional[AdapterResult] = None

    def __post_init__(self):
        feasible = [r for r in self.results if r.feasible]
        if feasible:
            self.best = max(feasible, key=lambda r: r.weighted_score)


@dataclass
class ParadigmComparisonResult:
    """Cross-paradigm ranking output."""

    # ── Input ──
    spec: InteractionSpec
    application: ApplicationContext

    # ── Grouped by paradigm ──
    groups: list[ParadigmGroup]

    # ── All results, ranked by weighted_score ──
    all_results: list[AdapterResult]

    # ── Overall recommendation ──
    recommended_paradigm: str = ""
    recommended_adapter: str = ""
    recommendation_rationale: str = ""

    # ── Metadata ──
    paradigms_evaluated: int = 0
    adapters_evaluated: int = 0
    feasible_count: int = 0


# ─────────────────────────────────────────────
# Paradigm Router
# ─────────────────────────────────────────────

class ParadigmRouter:
    """
    Routes an InteractionSpec to the correct adapter pipeline(s).

    Two modes:
        1. TYPED routing: spec is already a specific subtype
           (DiscretePocketSpec, NetworkInteractionSpec, etc.)
           → route to adapters for that paradigm only.

        2. EXPLORATORY routing: spec is a base InteractionSpec
           or user wants cross-paradigm comparison
           → route to ALL registered adapters, let scoring decide.

    The router is intentionally permissive. Premature filtering
    is a silent data integrity failure.
    """

    def __init__(self):
        self._adapters: dict[str, list[ParadigmAdapter]] = {
            "pocket": [],
            "network": [],
            "surface": [],
            "bulk": [],
            "field": [],
            "composite": [],
        }

    def register(self, adapter: ParadigmAdapter) -> None:
        """Register an adapter for its supported paradigm(s)."""
        for spec_type in adapter.supported_spec_types:
            if spec_type in self._adapters:
                self._adapters[spec_type].append(adapter)

    def registered_paradigms(self) -> list[str]:
        """Which paradigms have at least one adapter registered."""
        return [p for p, adapters in self._adapters.items() if adapters]

    def adapters_for(self, paradigm: str) -> list[ParadigmAdapter]:
        return self._adapters.get(paradigm, [])

    def route(self, spec: InteractionSpec) -> list[tuple[str, ParadigmAdapter]]:
        """
        Determine which adapters can handle this spec.

        Returns list of (paradigm, adapter) tuples.
        For typed specs: only matching paradigm.
        """
        paradigm = spec.spec_type
        matches = []

        if paradigm in self._adapters:
            for adapter in self._adapters[paradigm]:
                matches.append((paradigm, adapter))

        return matches

    def route_all(self, spec: InteractionSpec) -> list[tuple[str, ParadigmAdapter]]:
        """
        Return ALL registered adapters (for cross-paradigm comparison).

        The caller is responsible for generating paradigm-appropriate
        specs if the original spec doesn't match the adapter's paradigm.
        """
        matches = []
        for paradigm, adapters in self._adapters.items():
            for adapter in adapters:
                matches.append((paradigm, adapter))
        return matches


# ─────────────────────────────────────────────
# Cross-Paradigm Ranker
# ─────────────────────────────────────────────

class CrossParadigmRanker:
    """
    Scores and ranks results across paradigms.

    The key insight: different paradigms use different physics,
    so their raw scores aren't directly comparable. We normalize
    on APPLICATION-RELEVANT axes that are paradigm-agnostic:

        - selectivity (does it grab the target?)
        - capacity (how much can it hold?)
        - cost (can we afford it?)
        - confidence (how much do we trust the prediction?)
        - calibration status (is the model validated?)

    The confidence-weighted composite (score × confidence) is the
    decision metric. This automatically penalizes paradigms with
    less calibration data.
    """

    def __init__(self, router: ParadigmRouter):
        self._router = router

    def evaluate(
        self,
        spec: InteractionSpec,
        application: Optional[ApplicationContext] = None,
    ) -> ParadigmComparisonResult:
        """
        Score spec against all matching adapters and produce comparison.

        For typed specs (DiscretePocketSpec, NetworkInteractionSpec, etc.),
        only evaluates adapters for that paradigm.
        For cross-paradigm comparison, caller should use evaluate_cross().
        """
        app = application or self._infer_application(spec)
        matches = self._router.route(spec)

        results = []
        for paradigm, adapter in matches:
            result = self._evaluate_one(spec, paradigm, adapter)
            results.append(result)

        return self._build_comparison(spec, app, results)

    def evaluate_cross(
        self,
        specs: dict[str, InteractionSpec],
        application: Optional[ApplicationContext] = None,
    ) -> ParadigmComparisonResult:
        """
        Evaluate multiple paradigm-specific specs for the same problem.

        Args:
            specs: {"network": NetworkInteractionSpec(...),
                    "surface": SurfaceInteractionSpec(...), ...}
            application: override application context

        This is the main entry point for cross-paradigm comparison.
        The caller generates paradigm-specific specs for the same target,
        and this method scores each against its matching adapters.
        """
        app = application
        results = []

        for paradigm_key, spec in specs.items():
            if app is None:
                app = self._infer_application(spec)
            matches = self._router.route(spec)
            for paradigm, adapter in matches:
                result = self._evaluate_one(spec, paradigm, adapter)
                results.append(result)

        return self._build_comparison(
            list(specs.values())[0] if specs else InteractionSpec(),
            app or ApplicationContext.RESEARCH,
            results,
        )

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _evaluate_one(
        self,
        spec: InteractionSpec,
        paradigm: str,
        adapter: ParadigmAdapter,
    ) -> AdapterResult:
        """Score one spec against one adapter."""
        try:
            raw = adapter.score(spec)
        except Exception as e:
            return AdapterResult(
                paradigm=paradigm,
                adapter_id=adapter.system_id,
                feasible=False,
                reason=f"Adapter error: {e}",
            )

        feasible = raw.get("feasible", False)
        if not feasible:
            return AdapterResult(
                paradigm=paradigm,
                adapter_id=adapter.system_id,
                feasible=False,
                reason=raw.get("reason", "Infeasible"),
                raw_scores=raw,
            )

        return AdapterResult(
            paradigm=paradigm,
            adapter_id=adapter.system_id,
            feasible=True,
            composite_score=raw.get("composite_score", 0.0),
            selectivity_score=raw.get("selectivity_score",
                                     raw.get("competition_score", 0.0)),
            capacity_score=raw.get("capacity_score", 0.0),
            confidence=raw.get("confidence", 0.0),
            calibration_status=raw.get("calibration_status", "unknown"),
            raw_scores=raw,
        )

    def _build_comparison(
        self,
        spec: InteractionSpec,
        application: ApplicationContext,
        results: list[AdapterResult],
    ) -> ParadigmComparisonResult:
        """Build the grouped + ranked comparison result."""

        # Group by paradigm
        paradigm_map: dict[str, list[AdapterResult]] = {}
        for r in results:
            paradigm_map.setdefault(r.paradigm, []).append(r)

        groups = []
        for paradigm, group_results in paradigm_map.items():
            groups.append(ParadigmGroup(paradigm=paradigm, results=group_results))

        # Sort all results by weighted_score (composite × confidence)
        all_sorted = sorted(
            results,
            key=lambda r: r.weighted_score if r.feasible else -1.0,
            reverse=True,
        )

        # Recommendation
        feasible = [r for r in all_sorted if r.feasible]
        if feasible:
            best = feasible[0]
            rec_paradigm = best.paradigm
            rec_adapter = best.adapter_id
            rec_rationale = (
                f"{best.adapter_id} ({best.paradigm} paradigm) scored highest "
                f"with weighted_score={best.weighted_score:.3f} "
                f"(composite={best.composite_score:.3f} × "
                f"confidence={best.confidence:.2f}). "
                f"Calibration: {best.calibration_status}."
            )
        else:
            rec_paradigm = ""
            rec_adapter = ""
            rec_rationale = "No feasible solutions found across any paradigm."

        return ParadigmComparisonResult(
            spec=spec,
            application=application,
            groups=groups,
            all_results=all_sorted,
            recommended_paradigm=rec_paradigm,
            recommended_adapter=rec_adapter,
            recommendation_rationale=rec_rationale,
            paradigms_evaluated=len(paradigm_map),
            adapters_evaluated=len(results),
            feasible_count=len(feasible),
        )

    def _infer_application(self, spec: InteractionSpec) -> ApplicationContext:
        """Try to get application from spec if it has the field."""
        return getattr(spec, "target_application", ApplicationContext.RESEARCH)