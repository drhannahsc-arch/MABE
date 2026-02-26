"""
MABE Realization Engine — Layer 3 + Layer 4

Physics first. Compute the ideal pocket. Then measure deviation.

Layer 3 Phase 1: InteractionGeometrySpec → IdealPocketSpec
Layer 3 Phase 2: IdealPocketSpec × MaterialRegistry → RankedRealizations
Layer 4:         RankedRealizations → FabricationSpec (per adapter)
"""

from mabe.realization.models import (
    InteractionGeometrySpec,
    IdealPocketSpec,
    DeviationReport,
    RealizationScore,
    RankedRealizations,
    FabricationSpec,
)
from mabe.realization.engine.ideal_pocket import compute_ideal_pocket
from mabe.realization.engine.ranker import rank_realizations

__all__ = [
    "InteractionGeometrySpec",
    "IdealPocketSpec",
    "DeviationReport",
    "RealizationScore",
    "RankedRealizations",
    "FabricationSpec",
    "compute_ideal_pocket",
    "rank_realizations",
]
