"""
RealizationAdapter — base class for all Layer 4 adapters.

Each adapter handles one material system. It does two things:
    1. estimate_fidelity() — quick score for the ranker (Phase 2)
    2. design()            — full pocket design → FabricationSpec (Phase 4)

Concrete adapters implement these for specific material systems.
Sprint R2+ will add: CyclodextrinAdapter, CrownEtherAdapter, PorphyrinAdapter, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import (
        InteractionGeometrySpec,
        RealizationScore,
        FabricationSpec,
    )
    from mabe.realization.registry.material_registry import MaterialCapability


class RealizationAdapter(ABC):
    """Base class for all Layer 4 material-system adapters."""

    system_id: str
    capability: "MaterialCapability"

    def __init__(self, capability: "MaterialCapability"):
        self.system_id = capability.system_id
        self.capability = capability

    @abstractmethod
    def estimate_fidelity(
        self,
        spec: "InteractionGeometrySpec",
    ) -> "RealizationScore":
        """
        Quick score without full design. Used by Layer 3 ranker.

        Must be fast (<1 second). Uses capability envelope + heuristics.
        Full design is expensive and only runs on selected systems.
        """
        ...

    @abstractmethod
    def design(
        self,
        spec: "InteractionGeometrySpec",
    ) -> "FabricationSpec":
        """
        Full pocket design. Produces fabrication-ready output.

        This is the expensive call. Only runs when the ranker selects
        this material system (or the user requests it explicitly).
        """
        ...

    @abstractmethod
    def validate_design(
        self,
        fab: "FabricationSpec",
    ) -> "ValidationReport":
        """
        Check the design for internal consistency, strain, clashes.

        Catches designs that look good on paper but would fail in practice.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system_id={self.system_id!r})"


@dataclass
class ValidationReport:
    """Result of adapter self-validation on a design."""
    valid: bool
    issues: list[str]
    warnings: list[str]
    strain_energy_kJ_mol: float = 0.0
    steric_clashes: int = 0
    confidence: float = 0.0
