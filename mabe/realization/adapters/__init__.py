"""
Layer 4 Implementation Adapters.

Each adapter takes an InteractionGeometrySpec and produces a FabricationSpec.
Organized by physics class, not material origin.

Sprint R1: base class only. Concrete adapters in Sprint R2+.
"""

from mabe.realization.adapters.base import RealizationAdapter

__all__ = ["RealizationAdapter"]
