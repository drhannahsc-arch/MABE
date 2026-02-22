"""
core/interior_designer_hook.py - Hooks self-binding structures into interior designer.

Wraps the original design_interior to check for self-binding structures first.
"""

from core.assembly import InteriorDesign, StructuralConstraint
from core.problem import Problem
from core.candidate import CandidateResult
from core.interior_designer import design_interior as _original_design_interior
from core.interior_designer_patch import design_self_binding_interior


def design_interior(candidate: CandidateResult,
                     structure: StructuralConstraint,
                     problem: Problem,
                     all_candidates: list[CandidateResult]) -> InteriorDesign:
    """
    Design interior — check self-binding structures first, then fall back
    to original interior designer for structures that need external recognition.
    """
    # Self-binding structures don't need external recognition
    self_design = design_self_binding_interior(structure, problem)
    if self_design is not None:
        return self_design

    # Fall back to original designer for DNA origami, MOFs, protein cages, etc.
    return _original_design_interior(candidate, structure, problem, all_candidates)
