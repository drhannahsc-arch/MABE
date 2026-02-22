"""
core/assembly_composer.py - Composes binder assemblies using interior designer.

The interior designer creates the binding pocket.
The assembly composer wraps that in structure, selectivity, and release.
"""

from __future__ import annotations

from core.problem import Problem
from core.assembly import (
    BinderAssembly, InteriorDesign, StructuralConstraint,
    SelectivityFilter, ReleaseMechanism,
)
from core.candidate import CandidateResult
from core.interior_designer import design_interior
from knowledge.structural_library import (
    STRUCTURAL_OPTIONS, generate_selectivity_filter, get_compatible_releases,
)


def _score_structural_match(interior: InteriorDesign,
                             structure: StructuralConstraint,
                             problem: Problem) -> float:
    """Score how well structure matches interior design + problem."""
    score = 0.5

    if structure.type == "none":
        return 0.5

    matrix_ph = problem.matrix.ph or 7.0
    if structure.ph_stable_range[0] <= matrix_ph <= structure.ph_stable_range[1]:
        score += 0.1
    else:
        score -= 0.3

    matrix_temp = problem.matrix.temperature_c or 25.0
    if structure.temp_stable_c[0] <= matrix_temp <= structure.temp_stable_c[1]:
        score += 0.05
    else:
        score -= 0.2

    if structure.type == "dna_origami_cage" and problem.matrix.competing_species:
        score += 0.15

    if structure.type == "mof":
        score += 0.1

    # Bonus for tertiary designs — structure matters more
    if interior.design_level == "tertiary":
        score += 0.1

    # Avidity bonus
    if interior.avidity_factor > 3.0:
        score += 0.1

    if structure.synthesis_complexity == "complex":
        score -= 0.1
    elif structure.synthesis_complexity == "expert":
        score -= 0.2

    return max(0.0, min(1.0, score))


def compose_assemblies(candidates: list[CandidateResult],
                        problem: Problem,
                        max_assemblies: int = 8) -> list[BinderAssembly]:
    """Compose assemblies with designed interiors."""
    assemblies = []
    wants_release = "release" in problem.desired_outcome.description.lower()

    real_candidates = [c for c in candidates if c.source_tool != "dummy"]
    top_recognition = sorted(real_candidates,
                              key=lambda c: c.performance.probability_of_success,
                              reverse=True)[:4]

    if not top_recognition:
        top_recognition = candidates[:3]

    none_struct = [s for s in STRUCTURAL_OPTIONS if s.type == "none"][0]

    for candidate in top_recognition:
        # Get compatible structures
        struct_scores = []
        for structure in STRUCTURAL_OPTIONS:
            if structure.type == "none":
                continue
            # Quick pH pre-filter
            matrix_ph = problem.matrix.ph or 7.0
            if not (structure.ph_stable_range[0] - 0.5 <= matrix_ph <= structure.ph_stable_range[1] + 0.5):
                continue
            struct_scores.append(structure)

        # Design interiors for: free + best structures (up to 4 for diversity)
        selected = [none_struct]
        if struct_scores:
            # Take up to 2 compatible structures
            selected.extend(struct_scores[:4])

        for structure in selected:
            # DESIGN THE INTERIOR
            interior = design_interior(candidate, structure, problem, real_candidates)

            # Selectivity filter
            target_radius = 0.3
            if problem.target.size and problem.target.size.hydrated_radius_angstrom:
                target_radius = problem.target.size.hydrated_radius_angstrom / 10.0
            selectivity = generate_selectivity_filter(structure, target_radius)

            # Release
            primary_type = interior.sites[0].recognition.type if interior.sites else "chelator"
            releases = get_compatible_releases(primary_type, structure.type, wants_release)
            release = releases[0] if releases else ReleaseMechanism(
                name="No active release", trigger="none", description="Permanent capture")

            # Score
            base_prob = candidate.performance.probability_of_success
            struct_bonus = _score_structural_match(interior, structure, problem) - 0.5

            # Avidity improves effective Kd
            if interior.avidity_factor > 1.0:
                avidity_boost = min(0.15, (interior.avidity_factor - 1.0) * 0.03)
            else:
                avidity_boost = 0.0

            composite = max(0.05, min(0.95, base_prob + struct_bonus * 0.3 + avidity_boost))

            # Confidence
            if structure.type == "none":
                confidence = candidate.performance.confidence
                conf_reason = candidate.performance.confidence_reasoning
            elif interior.design_level == "tertiary":
                confidence = "speculative"
                conf_reason = (
                    f"Tertiary pocket design: {interior.design_rationale[:100]}... "
                    f"Mixed donor cooperativity predicted but requires experimental validation."
                )
            else:
                confidence = "low" if candidate.performance.confidence == "low" else "speculative"
                conf_reason = (
                    f"Recognition: {candidate.performance.confidence_reasoning} "
                    f"Structural integration predicted but not validated."
                )

            # Cost
            if structure.type == "none":
                cost = candidate.accessibility.estimated_cost
            else:
                cost = f"{candidate.accessibility.estimated_cost} (binder) + {structure.cost_per_unit} (structure)"

            # Failure modes
            failure_modes = list(candidate.performance.failure_modes)
            if structure.type == "dna_origami_cage":
                failure_modes.append("Cage assembly yield ~30-70% — optimize Mg2+ and anneal protocol")
                failure_modes.append("Metal ions pass through DNA walls — interior binding mandatory")
            if structure.type == "mof":
                failure_modes.append("Post-synthetic modification efficiency varies — characterize loading")
            if interior.design_level == "tertiary":
                failure_modes.append("Mixed pocket cooperativity is predicted, not measured — validate by ITC")

            # Improvements
            improvements = list(candidate.performance.what_improves_odds)
            if interior.avidity_factor > 2.0:
                improvements.append(
                    f"Multivalent interior ({interior.total_binding_sites} sites) "
                    f"provides {interior.avidity_factor:.0f}x avidity enhancement"
                )
            if interior.design_level == "tertiary":
                improvements.append("Mixed donor pocket may show emergent selectivity — test against panel")

            # Name
            if structure.type == "none":
                name = f"{candidate.name} (free)"
                desc = (
                    f"Recognition: {candidate.name}. Free in solution. "
                    f"Release by {release.name}."
                )
            else:
                level_label = {"simple": "", "composite": "composite ", "tertiary": "TERTIARY "}
                name = f"{level_label.get(interior.design_level, '')}{candidate.name} in {structure.name}"
                desc = (
                    f"{interior.design_level.title()} binder: {interior.summary()}. "
                    f"Structure: {structure.name}. "
                    f"Release by {release.name}."
                )

            assemblies.append(BinderAssembly(
                name=name.strip(),
                description=desc,
                design_level=interior.design_level,
                interior=interior,
                structure=structure,
                selectivity=selectivity,
                release=release,
                composite_score=round(composite, 2),
                confidence=confidence,
                confidence_reasoning=conf_reason,
                estimated_cost=cost,
                community_lab_feasible=(
                    candidate.accessibility.community_lab_feasible
                    and structure.synthesis_complexity in ("trivial", "standard")
                ),
                failure_modes=failure_modes,
                what_improves_odds=improvements,
            ))

    assemblies.sort(key=lambda a: a.composite_score, reverse=True)
    return assemblies[:max_assemblies]
