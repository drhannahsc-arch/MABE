"""
core/scoring_patch.py - Extends structural match scoring for sprint 8 structures.

Adds scoring bonuses/penalties for:
- mesoporous_silica: huge bonus for harsh pH, cheap
- zeolite: huge bonus for cation targets, very cheap
- mip: bonus for any target, extreme pH stability
- ldh: bonus for anion targets
- cof: bonus for stability
- carbon_nanotube, graphene_oxide: bonus if electrochemical desired
"""

import core.assembly_composer as composer
from core.assembly import InteriorDesign, StructuralConstraint
from core.problem import Problem


_original_score = composer._score_structural_match


def _extended_score(interior: InteriorDesign,
                     structure: StructuralConstraint,
                     problem: Problem) -> float:
    """Extended scoring that handles sprint 8 structure types."""
    # Start with original scoring for backward compatibility
    score = _original_score(interior, structure, problem)

    stype = structure.type

    # Silicates: bonus for harsh conditions
    if stype in ("mesoporous_silica", "zeolite", "silica_np"):
        matrix_ph = problem.matrix.ph or 7.0
        if matrix_ph < 3.0 or matrix_ph > 11.0:
            score += 0.15  # survives extreme pH
        score += 0.05  # cheap and scalable

    # Zeolites: inherent cation exchange
    if stype == "zeolite":
        charge = problem.target.charge if problem.target.charge else 0
        if charge > 0:
            score += 0.15  # natural cation exchanger
        else:
            score -= 0.3  # not for anions

    # LDH: inherent anion exchange
    if stype == "ldh":
        charge = problem.target.charge if problem.target.charge else 0
        if charge < 0:
            score += 0.15  # natural anion exchanger
        else:
            score -= 0.3  # not for cations

    # MIP: works for anything, extreme stability
    if stype == "mip":
        score += 0.1  # always applicable, very stable

    # COF: ultra-stable
    if stype == "cof":
        score += 0.05

    # Electrochemical structures: bonus if conductive substrate useful
    if stype in ("carbon_nanotube", "graphene_oxide"):
        if "monitor" in problem.desired_outcome.description.lower():
            score += 0.1

    return max(0.0, min(1.0, score))


# Apply the patch
composer._score_structural_match = _extended_score
