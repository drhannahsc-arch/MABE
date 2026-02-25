"""
realization_ranker/epistemic.py — Epistemic scoring infrastructure.

Every numerical score in Layer 3 carries its provenance:
  - What equation produced it (if physics-derived)
  - What API/database was queried (if empirical)
  - What uncertainty bounds apply
  - Whether it's a heuristic guess requiring more data

This prevents the foundry from ever presenting a guess as a fact.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EpistemicBasis(Enum):
    """How was this score derived?"""
    PHYSICS_DERIVED = "physics_derived"          # From equations with known parameters
    API_EMPIRICAL = "api_empirical"              # Queried from experimental database
    HEURISTIC_ESTIMATE = "heuristic_estimate"    # Best guess, more data required


@dataclass
class EpistemicScore:
    """A score that knows how confident it is and why."""
    value: float                                  # 0.0–1.0 normalized
    basis: EpistemicBasis
    equation: Optional[str] = None                # LaTeX or plain-text equation used
    data_source: Optional[str] = None             # API/database queried
    uncertainty: float = 0.0                       # ± in same units as value
    note: Optional[str] = None                     # Human-readable caveat

    def __post_init__(self):
        if self.basis == EpistemicBasis.HEURISTIC_ESTIMATE and self.note is None:
            self.note = "Best guess, more data required"

    @property
    def is_heuristic(self) -> bool:
        return self.basis == EpistemicBasis.HEURISTIC_ESTIMATE

    @property
    def is_physics(self) -> bool:
        return self.basis == EpistemicBasis.PHYSICS_DERIVED

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 4),
            "basis": self.basis.value,
            "equation": self.equation,
            "data_source": self.data_source,
            "uncertainty": round(self.uncertainty, 4),
            "note": self.note,
        }


@dataclass
class RealizationScore:
    """Complete scored evaluation of one material system realization."""
    realization_type: str                          # e.g. "small_molecule", "MOF_lattice"
    geometric_fidelity: EpistemicScore
    synthetic_accessibility: EpistemicScore
    operating_conditions: EpistemicScore
    scale_feasibility: EpistemicScore
    composite: float = 0.0                         # Weighted combination
    disqualified: bool = False
    disqualification_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)    # Realization-specific details

    def compute_composite(
        self,
        w_geo: float = 0.40,
        w_sa: float = 0.25,
        w_oc: float = 0.25,
        w_scale: float = 0.10,
    ) -> float:
        """Weighted composite. Returns 0 if disqualified."""
        if self.disqualified:
            self.composite = 0.0
            return 0.0
        self.composite = (
            w_geo * self.geometric_fidelity.value
            + w_sa * self.synthetic_accessibility.value
            + w_oc * self.operating_conditions.value
            + w_scale * self.scale_feasibility.value
        )
        return self.composite

    @property
    def heuristic_count(self) -> int:
        """How many axes are heuristic guesses?"""
        return sum(
            1 for s in [
                self.geometric_fidelity,
                self.synthetic_accessibility,
                self.operating_conditions,
                self.scale_feasibility,
            ]
            if s.is_heuristic
        )

    @property
    def epistemic_summary(self) -> str:
        """One-line summary of confidence basis."""
        labels = {
            EpistemicBasis.PHYSICS_DERIVED: "P",
            EpistemicBasis.API_EMPIRICAL: "E",
            EpistemicBasis.HEURISTIC_ESTIMATE: "H",
        }
        axes = [
            ("geo", self.geometric_fidelity),
            ("SA", self.synthetic_accessibility),
            ("OC", self.operating_conditions),
            ("scale", self.scale_feasibility),
        ]
        parts = [f"{name}={labels[s.basis]}" for name, s in axes]
        return f"[{', '.join(parts)}]"

    def to_dict(self) -> dict:
        return {
            "realization_type": self.realization_type,
            "composite": round(self.composite, 4),
            "disqualified": self.disqualified,
            "disqualification_reason": self.disqualification_reason,
            "epistemic_summary": self.epistemic_summary,
            "heuristic_count": self.heuristic_count,
            "axes": {
                "geometric_fidelity": self.geometric_fidelity.to_dict(),
                "synthetic_accessibility": self.synthetic_accessibility.to_dict(),
                "operating_conditions": self.operating_conditions.to_dict(),
                "scale_feasibility": self.scale_feasibility.to_dict(),
            },
            "metadata": self.metadata,
        }


@dataclass
class RankedRealizations:
    """Output of Layer 3: ranked list with full epistemic transparency."""
    target_summary: str
    geometry_spec_hash: str                        # Trace back to Layer 2 output
    realizations: list[RealizationScore] = field(default_factory=list)
    weights_used: dict = field(default_factory=dict)

    def rank(
        self,
        w_geo: float = 0.40,
        w_sa: float = 0.25,
        w_oc: float = 0.25,
        w_scale: float = 0.10,
    ) -> list[RealizationScore]:
        """Compute composites and sort descending. Disqualified entries sort last."""
        self.weights_used = {
            "geometric_fidelity": w_geo,
            "synthetic_accessibility": w_sa,
            "operating_conditions": w_oc,
            "scale_feasibility": w_scale,
        }
        for r in self.realizations:
            r.compute_composite(w_geo, w_sa, w_oc, w_scale)
        self.realizations.sort(
            key=lambda r: (not r.disqualified, r.composite), reverse=True
        )
        return self.realizations

    def qualified_only(self) -> list[RealizationScore]:
        return [r for r in self.realizations if not r.disqualified]

    def to_dict(self) -> dict:
        return {
            "target_summary": self.target_summary,
            "geometry_spec_hash": self.geometry_spec_hash,
            "weights_used": self.weights_used,
            "n_qualified": len(self.qualified_only()),
            "n_total": len(self.realizations),
            "realizations": [r.to_dict() for r in self.realizations],
        }