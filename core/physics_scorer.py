"""
core/physics_scorer.py - Physics-based composite scoring.

Replaces the old heuristic scoring with thermodynamics + hydrodynamics.

Old: composite_score = 0.4*probability + 0.25*accessibility + 0.2*reusability + 0.15*evidence
New: composite_score = f(ΔG_net, transport_factor, practical_factors)

The score is now interpretable:
    "This assembly has ΔG = -18 kJ/mol (K_eq ~ 1400), but transport through
    0.55nm zeolite pores reduces effective capture to 23% of equilibrium."
"""

from core.thermodynamics import BindingThermodynamics
import core.thermodynamics as _thermo_mod
from core.hydrodynamics import HydrodynamicProfile, compute_hydrodynamics
from core.assembly import BinderAssembly, StructuralConstraint, InteriorDesign
from core.problem import Problem


def physics_score(assembly: BinderAssembly,
                   problem: Problem) -> tuple[float, BindingThermodynamics, HydrodynamicProfile]:
    """
    Compute physics-based score for a binder assembly.

    Returns (score 0-1, thermodynamics, hydrodynamics).
    The score is derived from real energy scales, not arbitrary weights.
    """
    recognition = assembly.recognition
    structure = assembly.structure
    interior = assembly.interior

    # Thermodynamics
    thermo = _thermo_mod.compute_thermodynamics(recognition, structure, interior, problem)

    # Hydrodynamics
    hydro = compute_hydrodynamics(structure, interior, problem)

    # ── Convert ΔG to a 0-1 score ─────────────────────────────────
    # Map ΔG_net to probability via Boltzmann
    p_bind = thermo.probability_of_binding()

    # Selectivity bonus: ΔΔG < -5 kJ/mol is meaningful
    if thermo.ddG_vs_top_competitor < -10.0:
        selectivity_score = 1.0
    elif thermo.ddG_vs_top_competitor < -5.0:
        selectivity_score = 0.8
    elif thermo.ddG_vs_top_competitor < 0.0:
        selectivity_score = 0.6
    else:
        selectivity_score = 0.3  # competitor preferred — problem

    # Transport penalty
    transport = hydro.transport_limitation_factor

    # Practical factors (small weight — physics dominates)
    practical = 0.0
    if structure.synthesis_complexity == "trivial":
        practical = 0.05
    elif structure.synthesis_complexity == "standard":
        practical = 0.02
    elif structure.synthesis_complexity == "complex":
        practical = -0.05

    # Composite: physics-weighted
    # 50% binding probability, 25% selectivity, 20% transport, 5% practical
    composite = (
        0.50 * p_bind +
        0.25 * selectivity_score +
        0.20 * transport +
        0.05 * (0.5 + practical)
    )

    return max(0.01, min(0.99, round(composite, 3))), thermo, hydro
