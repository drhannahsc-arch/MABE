"""
realization_ranker/ranker.py — Layer 3 main entry point.

Input: InteractionGeometrySpec (from Layer 2) + deployment conditions
Output: RankedRealizations with per-axis epistemic scores

Pipeline:
  1. Run disqualification gates for all material systems
  2. Score geometric fidelity for qualified systems
  3. Score synthetic accessibility
  4. Score operating conditions compatibility
  5. Score scale feasibility
  6. Compute weighted composite and rank

Every score carries its epistemic provenance. No silent heuristics.
"""

from dataclasses import dataclass, field
from typing import Optional
import hashlib
import json

from .epistemic import (
    EpistemicScore,
    EpistemicBasis,
    RealizationScore,
    RankedRealizations,
)
from .disqualification import run_all_gates, ALL_REALIZATION_TYPES
from .geometric_fidelity import score_geometric_fidelity
from .synthetic_accessibility import score_synthetic_accessibility
from .operating_conditions import score_operating_conditions
from .scale_feasibility import score_scale_feasibility


@dataclass
class InteractionGeometrySpec:
    """
    Layer 2 output — realization-agnostic geometry specification.

    This is a simplified version for Layer 3 consumption. The full
    InteractionGeometrySpec in Layer 2 has additional fields for
    field/charge distribution, flexibility constraints, etc.
    """
    cavity_diameter_nm: float                      # Required pocket/cavity size
    donor_count: int                                # Number of interaction elements
    donor_types: list[str] = field(default_factory=list)  # Donor subtypes needed
    donor_distances_nm: list[float] = field(default_factory=list)  # Pairwise distances
    coordination_geometry: Optional[str] = None     # e.g. "octahedral", "tetrahedral"
    target_summary: str = ""                        # Human-readable target description

    @property
    def hash(self) -> str:
        """Deterministic hash for tracing back to Layer 2."""
        data = json.dumps({
            "cavity": self.cavity_diameter_nm,
            "donors": self.donor_count,
            "types": sorted(self.donor_types),
            "distances": sorted(self.donor_distances_nm),
            "geometry": self.coordination_geometry,
        }, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:12]


@dataclass
class DeploymentConditions:
    """Environmental conditions the realization must survive."""
    temperature_C: float = 25.0
    pH: float = 7.0
    oxidants: list[str] = field(default_factory=list)
    environment: str = "environmental_water"    # "serum", "environmental_water", "industrial"
    operational_hours: float = 24.0
    target_scale: str = "pilot"                 # "research", "diagnostic", etc.
    strong_oxidant: bool = False                # Simplified flag for disqualification gates


def rank_realizations(
    geometry: InteractionGeometrySpec,
    conditions: DeploymentConditions,
    realization_types: list[str] = None,
    weights: dict = None,
) -> RankedRealizations:
    """
    Main Layer 3 entry point.

    Parameters
    ----------
    geometry : InteractionGeometrySpec
        From Layer 2 — what geometry do we need?
    conditions : DeploymentConditions
        Where will this be deployed?
    realization_types : list[str], optional
        Which material systems to consider. Default: all known types.
    weights : dict, optional
        Custom axis weights. Keys: "geometric_fidelity", "synthetic_accessibility",
        "operating_conditions", "scale_feasibility". Default: 0.40/0.25/0.25/0.10.

    Returns
    -------
    RankedRealizations with full epistemic transparency.
    """

    if realization_types is None:
        realization_types = ALL_REALIZATION_TYPES

    if weights is None:
        weights = {
            "geometric_fidelity": 0.40,
            "synthetic_accessibility": 0.25,
            "operating_conditions": 0.25,
            "scale_feasibility": 0.10,
        }

    results = RankedRealizations(
        target_summary=geometry.target_summary,
        geometry_spec_hash=geometry.hash,
    )

    for rt in realization_types:

        # ─── Step 1: Disqualification gates ───
        dq = run_all_gates(
            realization_type=rt,
            required_cavity_nm=geometry.cavity_diameter_nm,
            required_donors=set(geometry.donor_types),
            pH=conditions.pH,
            temperature_C=conditions.temperature_C,
            strong_oxidant=conditions.strong_oxidant,
        )

        if not dq.passed:
            results.realizations.append(
                RealizationScore(
                    realization_type=rt,
                    geometric_fidelity=EpistemicScore(0.0, EpistemicBasis.PHYSICS_DERIVED),
                    synthetic_accessibility=EpistemicScore(0.0, EpistemicBasis.PHYSICS_DERIVED),
                    operating_conditions=EpistemicScore(0.0, EpistemicBasis.PHYSICS_DERIVED),
                    scale_feasibility=EpistemicScore(0.0, EpistemicBasis.PHYSICS_DERIVED),
                    disqualified=True,
                    disqualification_reason=dq.reason,
                )
            )
            continue

        # ─── Step 2: Geometric fidelity ───
        geo = score_geometric_fidelity(
            realization_type=rt,
            required_cavity_nm=geometry.cavity_diameter_nm,
            required_donor_count=geometry.donor_count,
            required_donor_types=geometry.donor_types,
            required_donor_distances_nm=geometry.donor_distances_nm,
            required_coordination_geometry=geometry.coordination_geometry,
        )

        # ─── Step 3: Synthetic accessibility ───
        sa = score_synthetic_accessibility(
            realization_type=rt,
        )

        # ─── Step 4: Operating conditions ───
        oc = score_operating_conditions(
            realization_type=rt,
            operating_temp_C=conditions.temperature_C,
            operating_pH=conditions.pH,
            oxidants_present=conditions.oxidants,
            required_donor_types=geometry.donor_types,
            environment=conditions.environment,
            required_operational_hours=conditions.operational_hours,
        )

        # ─── Step 5: Scale feasibility ───
        scale = score_scale_feasibility(
            realization_type=rt,
            target_scale=conditions.target_scale,
        )

        # ─── Assemble RealizationScore ───
        rs = RealizationScore(
            realization_type=rt,
            geometric_fidelity=geo,
            synthetic_accessibility=sa,
            operating_conditions=oc,
            scale_feasibility=scale,
        )
        results.realizations.append(rs)

    # ─── Step 6: Rank ───
    results.rank(
        w_geo=weights["geometric_fidelity"],
        w_sa=weights["synthetic_accessibility"],
        w_oc=weights["operating_conditions"],
        w_scale=weights["scale_feasibility"],
    )

    return results


def print_rankings(ranked: RankedRealizations, show_disqualified: bool = True):
    """Pretty-print rankings to console."""
    print(f"\n{'='*72}")
    print(f"LAYER 3 REALIZATION RANKING")
    print(f"Target: {ranked.target_summary}")
    print(f"Geometry hash: {ranked.geometry_spec_hash}")
    print(f"Weights: {ranked.weights_used}")
    print(f"{'='*72}")

    for i, r in enumerate(ranked.realizations):
        if r.disqualified and not show_disqualified:
            continue

        status = "DISQUALIFIED" if r.disqualified else f"#{i+1}"
        print(f"\n{status}: {r.realization_type}")

        if r.disqualified:
            print(f"  Reason: {r.disqualification_reason}")
            continue

        print(f"  Composite: {r.composite:.3f}  {r.epistemic_summary}")
        print(f"  ├─ Geometric fidelity:      {r.geometric_fidelity.value:.3f} "
              f"[{r.geometric_fidelity.basis.value}]")
        print(f"  ├─ Synthetic accessibility: {r.synthetic_accessibility.value:.3f} "
              f"[{r.synthetic_accessibility.basis.value}]")
        print(f"  ├─ Operating conditions:    {r.operating_conditions.value:.3f} "
              f"[{r.operating_conditions.basis.value}]")
        print(f"  └─ Scale feasibility:       {r.scale_feasibility.value:.3f} "
              f"[{r.scale_feasibility.basis.value}]")

        if r.heuristic_count > 0:
            print(f"  ⚠ {r.heuristic_count} axis/axes flagged as heuristic estimate")

    n_qual = len(ranked.qualified_only())
    n_total = len(ranked.realizations)
    print(f"\n{'─'*72}")
    print(f"Qualified: {n_qual}/{n_total} material systems")
    print(f"{'='*72}\n")