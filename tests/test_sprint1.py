"""
tests/test_sprint1.py - End-to-end test for Sprint 1 skeleton.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.problem import Problem, TargetSpecies, Matrix, Outcome, Constraints
from core.candidate import CandidateResult
from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose, KNOWN_TARGETS


def test_target_species():
    se = KNOWN_TARGETS["selenite"]
    assert se.charge == -2.0
    assert se.geometry == "trigonal pyramidal"
    assert se.electronic.hardness_softness == "borderline"
    assert len(se.redox_states) == 4

    ni = KNOWN_TARGETS["nickel"]
    assert ni.magnetic.type == "paramagnetic"
    assert ni.magnetic.unpaired_electrons == 2
    print("  + TargetSpecies - physics descriptions correct")


def test_decompose_selenite_mine():
    problem = decompose("I need selenite capture from a mine in BC")
    assert problem.target.identity == "selenite"
    assert problem.target.charge == -2.0
    assert problem.matrix.ph == 3.5
    assert problem.constraints.no_environmental_release is True
    print(f"  + Decompose - selenite from mine detected")


def test_decompose_lead_release():
    problem = decompose("capture lead from mine water and release as feedstock")
    assert problem.target.identity == "lead"
    assert "release" in problem.desired_outcome.description.lower()
    assert problem.constraints.required_reusability_cycles >= 20
    print(f"  + Decompose - capture/release outcome detected")


def test_decompose_unknown():
    problem = decompose("I need to capture unobtanium from the atmosphere")
    assert problem.target.identity == "unknown target"
    assert len(problem.assumptions_made) > 0
    print(f"  + Decompose - unknown target handled with assumptions")


def test_registry():
    registry = ToolRegistry()
    registry.register(DummyAdapter())
    assert len(registry.available_adapters()) == 1
    problem = decompose("lead capture from mine")
    contributors = registry.find_contributors(problem)
    assert len(contributors) > 0
    print(f"  + Registry - {len(contributors)} tool(s) contribute")


def test_orchestrator():
    registry = ToolRegistry()
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)
    problem = decompose("selenite capture and release from mine water in BC")
    result = orchestrator.solve(problem)

    assert len(result.candidates) > 0
    for i, c in enumerate(result.candidates):
        assert c.rank == i + 1
    for c in result.candidates:
        assert c.performance.confidence in ("high", "moderate", "low", "speculative")
        assert len(c.performance.failure_modes) > 0
        assert c.evidence.source_type
        assert c.accessibility.estimated_cost
    has_connections = any(len(c.other_applications) > 0 for c in result.candidates)
    assert has_connections
    has_immob = any(len(c.immobilization_options) > 0 for c in result.candidates)
    assert has_immob
    print(f"  + Orchestrator - {len(result.candidates)} candidates, ranked, with uncertainty")


def test_values_ranking():
    registry = ToolRegistry()
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    chelator = None
    nanocage = None
    for c in result.candidates:
        if "chelator" in c.modality.lower():
            chelator = c
        if "nanocage" in c.modality.lower():
            nanocage = c

    if chelator and nanocage:
        assert chelator.rank < nanocage.rank
        print(f"  + Values - accessible chelator ranks above expensive nanocage")
    else:
        print(f"  + Values - ranking test skipped")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 1 - Skeleton Tests")
    print("  " + "=" * 40)
    print()

    test_target_species()
    test_decompose_selenite_mine()
    test_decompose_lead_release()
    test_decompose_unknown()
    test_registry()
    test_orchestrator()
    test_values_ranking()

    print()
    print("  All tests passed.")
    print()
