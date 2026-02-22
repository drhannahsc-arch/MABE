"""
core/sprint10_integration.py - Integrates kinetics, orbital, and probability chain
into the assembly pipeline.

Runs after physics_integration (sprint 9). Adds:
- Orbital analysis
- Kinetic profile
- Probability chain
- Updates composite score with probability chain

The composite score is now:
    score = f(P_cycle, selectivity, practical_factors)
"""

import core.assembly_composer as composer
import core.thermodynamics as _thermo_mod
from core.hydrodynamics import compute_hydrodynamics
from core.orbital_binding import compute_orbital_binding
from core.kinetics import compute_kinetics
from core.probability_chain import compute_probability_chain
from core.assembly import BinderAssembly
from core.problem import Problem
from knowledge.electronic_data import enrich_target_electronic


def full_physics_rescore(assemblies: list[BinderAssembly],
                          problem: Problem) -> list[BinderAssembly]:
    """
    Full physics pipeline: thermo → hydro → orbital → kinetics → probability chain.
    """
    # Enrich target with DFT data if available
    enrich_target_electronic(problem.target)

    for assembly in assemblies:
        recognition = assembly.recognition
        structure = assembly.structure
        interior = assembly.interior

        # Sprint 9: thermodynamics + hydrodynamics
        thermo = _thermo_mod.compute_thermodynamics(recognition, structure, interior, problem)
        hydro = compute_hydrodynamics(structure, interior, problem)

        # Sprint 10: orbital + kinetics + probability chain
        orbital = compute_orbital_binding(recognition, problem.target)
        kinetics = compute_kinetics(thermo, hydro, orbital, structure, interior, problem)
        chain = compute_probability_chain(thermo, hydro, kinetics, orbital, assembly, problem)

        # ── Update composite score from probability chain ─────────
        # Primary: P(capture) — most important for function
        # Secondary: selectivity factor, practical concerns
        selectivity_score = 0.5
        if thermo.selectivity_factor > 100:
            selectivity_score = 1.0
        elif thermo.selectivity_factor > 10:
            selectivity_score = 0.8
        elif thermo.selectivity_factor > 3:
            selectivity_score = 0.6
        else:
            selectivity_score = 0.3

        practical = 0.5
        if structure.synthesis_complexity == "trivial":
            practical = 0.7
        elif structure.synthesis_complexity == "standard":
            practical = 0.5
        elif structure.synthesis_complexity == "complex":
            practical = 0.3

        composite = (
            0.45 * chain.p_capture +
            0.25 * selectivity_score +
            0.15 * chain.p_retain +
            0.10 * chain.p_release +
            0.05 * practical
        )
        assembly.composite_score = max(0.01, min(0.99, round(composite, 3)))

        # ── Build comprehensive confidence reasoning ──────────────
        lines = [
            "PHYSICS:",
            thermo.summary(),
            "",
            "ORBITAL:",
            orbital.summary(),
            "",
            "KINETICS:",
            kinetics.summary(),
            "",
            "PROBABILITY CHAIN:",
            chain.summary(),
            "",
            "TRANSPORT:",
            hydro.summary(),
        ]
        assembly.confidence_reasoning = "\n".join(lines)

        # Confidence from ΔG + kinetics
        if thermo.dG_net < -25 and kinetics.fractional_occupancy > 0.8:
            assembly.confidence = "high"
        elif thermo.dG_net < -15 and kinetics.fractional_occupancy > 0.5:
            assembly.confidence = "moderate"
        elif thermo.dG_net < -5:
            assembly.confidence = "low"
        else:
            assembly.confidence = "speculative"

        # Add kinetic warnings
        if kinetics.rate_limiting_step.startswith("transport"):
            assembly.failure_modes.append(
                f"Transport-limited: k_on effective only {kinetics.k_on_M_s:.0e} M⁻¹s⁻¹"
            )
        if kinetics.time_to_equilibrium_s > 3600:
            assembly.failure_modes.append(
                f"Slow equilibrium: {kinetics.time_to_equilibrium_s/60:.0f} min to reach 98%"
            )
        if chain.p_capture < 0.1:
            assembly.failure_modes.append(
                f"Low capture probability: {chain.p_capture:.1%} per cycle"
            )

    assemblies.sort(key=lambda a: a.composite_score, reverse=True)
    return assemblies


# ── Patch into pipeline ───────────────────────────────────────────────

_prev_compose = composer.compose_assemblies


def _sprint10_compose(candidates, problem, max_assemblies=8):
    assemblies = _prev_compose(candidates, problem, max_assemblies)
    return full_physics_rescore(assemblies, problem)


composer.compose_assemblies = _sprint10_compose
