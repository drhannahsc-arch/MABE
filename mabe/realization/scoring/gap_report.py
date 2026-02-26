"""
Gap Report Generator.

When the best available material system only achieves partial physics
fidelity, the gap report answers: "Why can't anything build this pocket?"
and "What would need to exist?"

This is the output that lets you invent a new material.
"""

from __future__ import annotations

from typing import Optional

from mabe.realization.models import IdealPocketSpec, RealizationScore


def generate_gap_report(
    ideal: IdealPocketSpec,
    feasible_scores: list[RealizationScore],
) -> str:
    """
    Identify which ideal requirements are not met by any material system.

    Called when gap_to_ideal > 0.3.
    """
    if not feasible_scores:
        return "No feasible material systems registered for this geometry."

    lines = ["## Gap Analysis", ""]

    # Best achiever
    best = max(feasible_scores, key=lambda s: s.physics_fidelity)
    lines.append(
        f"Best physics fidelity: {best.physics_fidelity:.2f} "
        f"({best.material_system})"
    )
    lines.append("")

    # Per-element analysis: which elements are hardest to place?
    if best.deviation_from_ideal.element_deviations_A:
        lines.append("### Element-level deviations (best system):")
        for i, (element, dev) in enumerate(zip(
            ideal.optimal_elements,
            best.deviation_from_ideal.element_deviations_A,
        )):
            status = "✓" if dev <= element.required_precision_A else "✗"
            lines.append(
                f"  {status} Element {i}: {element.atom_type} — "
                f"required ±{element.required_precision_A:.2f} Å, "
                f"best achievable: ±{dev:.2f} Å"
            )
        lines.append("")

    # Missing interactions across ALL systems
    all_missing = set()
    for s in feasible_scores:
        all_missing.update(s.deviation_from_ideal.missing_interactions)
    if all_missing:
        lines.append("### Interactions no registered material can provide:")
        for m in sorted(all_missing):
            lines.append(f"  - {m}")
        lines.append("")

    # Precision bottleneck
    precision_gap = _identify_precision_bottleneck(ideal, feasible_scores)
    if precision_gap:
        lines.append(f"### Precision bottleneck: {precision_gap}")
        lines.append("")

    return "\n".join(lines)


def generate_novel_material_spec(
    ideal: IdealPocketSpec,
    feasible_scores: list[RealizationScore],
) -> str:
    """
    When gap > 0.5, describe what a novel material would need.

    This is the spec for something that doesn't exist yet but
    physics says should work.
    """
    lines = ["## Novel Material Specification", ""]
    lines.append("No registered material system achieves >0.5 physics fidelity.")
    lines.append("A material with the following properties would score 1.0:")
    lines.append("")

    # Required elements
    elements = sorted(ideal.required_elements)
    lines.append(f"**Required elements:** {', '.join(elements)}")

    # Required precision
    lines.append(
        f"**Positioning precision:** ≤ {ideal.min_precision_required_A:.2f} Å"
    )

    # Rigidity
    lines.append(f"**Rigidity class:** {ideal.rigidity_class.value}")

    # Stability
    lines.append(
        f"**Stability:** pH {ideal.min_stability_pH[0]:.1f}–"
        f"{ideal.min_stability_pH[1]:.1f}, "
        f"{ideal.min_stability_K[0]:.0f}–{ideal.min_stability_K[1]:.0f} K"
    )

    # Per-element requirements
    lines.append("")
    lines.append("**Per-element placement:**")
    for e in ideal.optimal_elements:
        lines.append(
            f"  - {e.atom_type} at {e.exact_position_A} ± {e.required_precision_A:.2f} Å "
            f"(energy contribution: {e.interaction_energy_contribution_kJ_mol:.1f} kJ/mol)"
        )

    # What's blocking existing systems
    lines.append("")
    lines.append("**Blocking constraints from existing systems:**")
    blockers = _identify_blockers(ideal, feasible_scores)
    for b in blockers:
        lines.append(f"  - {b}")

    return "\n".join(lines)


def _identify_precision_bottleneck(
    ideal: IdealPocketSpec,
    scores: list[RealizationScore],
) -> Optional[str]:
    """Find the element whose precision requirement eliminates the most systems."""
    if not ideal.optimal_elements:
        return None

    element_failures = {}
    for element in ideal.optimal_elements:
        failures = 0
        for s in scores:
            idx = ideal.optimal_elements.index(element)
            if idx < len(s.deviation_from_ideal.element_deviations_A):
                dev = s.deviation_from_ideal.element_deviations_A[idx]
                if dev > element.required_precision_A:
                    failures += 1
        element_failures[element.atom_type] = failures

    if not element_failures:
        return None

    worst = max(element_failures, key=element_failures.get)
    count = element_failures[worst]
    if count == 0:
        return None
    return (
        f"{worst} placement (required ±{ideal.optimal_elements[0].required_precision_A:.2f} Å) "
        f"fails for {count}/{len(scores)} feasible systems"
    )


def _identify_blockers(
    ideal: IdealPocketSpec,
    scores: list[RealizationScore],
) -> list[str]:
    """Identify what prevents existing systems from scoring well."""
    blockers = []

    if not scores:
        blockers.append("No registered material systems pass feasibility gate")
        return blockers

    best = max(scores, key=lambda s: s.physics_fidelity)

    # Precision blocker
    if best.deviation_from_ideal.mean_deviation_A > ideal.min_precision_required_A * 2:
        blockers.append(
            f"Best system ({best.material_system}) achieves "
            f"±{best.deviation_from_ideal.mean_deviation_A:.2f} Å mean, "
            f"but spec requires ±{ideal.min_precision_required_A:.2f} Å"
        )

    # Rigidity blocker
    if best.deviation_from_ideal.rigidity_deviation > 0.3:
        blockers.append(
            f"Required rigidity ({ideal.rigidity_class.value}) not achievable "
            f"by best system ({best.material_system})"
        )

    # Missing interaction blocker
    if best.deviation_from_ideal.missing_interactions:
        blockers.append(
            f"Best system cannot provide: "
            f"{', '.join(best.deviation_from_ideal.missing_interactions)}"
        )

    if not blockers:
        blockers.append("No single blocker identified — cumulative deviation across elements")

    return blockers
