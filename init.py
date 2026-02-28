"""
MABE Realization Engine — Layer 3 + Layer 4

Physics first. Compute the ideal interaction specification. Then measure deviation.

Layer 2 Output:   InteractionSpec (polymorphic: pocket, network, surface, bulk, field, composite)
Layer 3 Phase 1:  InteractionSpec → IdealPocketSpec (pocket paradigm)
Layer 3 Phase 2:  IdealSpec × MaterialRegistry → RankedRealizations
Layer 4:          RankedRealizations → FabricationSpec (per adapter)
"""

from mabe.realization.models import (
    # ── Polymorphic base + paradigm enum ──
    InteractionSpec,
    InteractionParadigm,
    DiscretePocketSpec,
    # ── Backward-compat alias ──
    InteractionGeometrySpec,
    # ── Downstream types ──
    IdealPocketSpec,
    DeviationReport,
    RealizationScore,
    RankedRealizations,
    FabricationSpec,
)
from mabe.realization.engine.ideal_pocket import compute_ideal_pocket
from mabe.realization.engine.ranker import rank_realizations

__all__ = [
    # New
    "InteractionSpec",
    "InteractionParadigm",
    "DiscretePocketSpec",
    # Backward compat
    "InteractionGeometrySpec",
    # Existing
    "IdealPocketSpec",
    "DeviationReport",
    "RealizationScore",
    "RankedRealizations",
    "FabricationSpec",
    "compute_ideal_pocket",
    "rank_realizations",
]