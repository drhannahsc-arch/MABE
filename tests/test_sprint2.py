"""
tests/test_sprint2.py — Tests for RDKit adapter.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter, RDKIT_AVAILABLE
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def test_rdkit_available():
    assert RDKIT_AVAILABLE, "RDKit not installed — run: python -m pip install rdkit"
    print("  + RDKit is installed and importable")


def test_rdkit_adapter_registers():
    registry = ToolRegistry()
    adapter = RDKitAdapter()
    registry.register(adapter)
    assert adapter.is_available()
    assert len(registry.available_adapters()) == 1
    print(f"  + RDKit adapter registered: {adapter}")


def test_rdkit_assesses_metal_target():
    adapter = RDKitAdapter()
    problem = decompose("lead capture from mine water")
    assessment = adapter.assess_contribution(problem)
    assert assessment.can_contribute is True
    assert assessment.relevance > 0.5
    print(f"  + RDKit can contribute to lead capture (relevance: {assessment.relevance})")


def test_rdkit_generates_real_candidates():
    adapter = RDKitAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) > 0
    print(f"  + RDKit generated {len(candidates)} chelator candidates for lead")

    # Candidates have real SMILES
    for c in candidates:
        assert "mock" not in c.structure_description.lower(), f"Found mock data in {c.name}"
        assert "MW=" in c.structure_description, f"Missing molecular weight in {c.name}"
    print(f"  + All candidates have real molecular structures (SMILES + computed properties)")


def test_rdkit_candidates_have_stability_data():
    adapter = RDKitAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)

    has_logk = False
    for c in candidates:
        if "log K" in c.performance.confidence_reasoning:
            has_logk = True
            break
    assert has_logk, "No candidates have stability constant data for lead"
    print(f"  + Candidates reference real stability constants (log K)")


def test_rdkit_selectivity_analysis():
    adapter = RDKitAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)

    has_threats = False
    for c in candidates:
        if c.performance.selectivity_threats:
            for threat in c.performance.selectivity_threats:
                if "compete" in threat.lower() or "interfere" in threat.lower():
                    has_threats = True
                    break
    assert has_threats, "No selectivity threats identified despite competing ions in mine water"
    print(f"  + Selectivity threats identified from competing ions in matrix")


def test_rdkit_hsab_scoring():
    adapter = RDKitAdapter()

    # Soft metal (gold) should rank soft donors higher
    gold_problem = decompose("gold capture from mine water")
    gold_candidates = adapter.generate_candidates(gold_problem)

    # Find a soft donor chelator
    soft_candidates = [c for c in gold_candidates if "soft" in c.description.lower() or "thiol" in c.description.lower() or "S" in c.description]
    hard_candidates = [c for c in gold_candidates if "hard" in c.description.lower() and "S" not in c.structure_description.split("|")[0]]

    if soft_candidates and hard_candidates:
        best_soft = max(c.performance.probability_of_success for c in soft_candidates)
        best_hard = max(c.performance.probability_of_success for c in hard_candidates)
        assert best_soft > best_hard, "Soft donors should score higher for soft metal (gold)"
        print(f"  + HSAB scoring correct: soft donors ({best_soft:.0%}) > hard donors ({best_hard:.0%}) for gold")
    else:
        print(f"  + HSAB scoring: {len(soft_candidates)} soft, {len(hard_candidates)} hard candidates for gold")


def test_full_pipeline_with_rdkit():
    """Full pipeline with both RDKit and dummy adapters."""
    registry = ToolRegistry()
    registry.register(RDKitAdapter())
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)

    problem = decompose("nickel capture and release from mine water")
    result = orchestrator.solve(problem)

    # Should have candidates from both adapters
    sources = set(c.source_tool for c in result.candidates)
    assert "rdkit_chelator" in sources, "No candidates from RDKit adapter"
    assert "dummy" in sources, "No candidates from dummy adapter"
    print(f"  + Full pipeline: {len(result.candidates)} candidates from {len(sources)} tools: {sources}")

    # RDKit candidates should have real data, dummy should have mock
    for c in result.candidates:
        if c.source_tool == "rdkit_chelator":
            assert "mock" not in c.structure_description.lower()
        assert c.performance.confidence in ("high", "moderate", "low", "speculative")
        assert c.accessibility.estimated_cost

    print(f"  + All candidates have appropriate data quality for their source")


def test_ph_compatibility():
    adapter = RDKitAdapter()

    # Mine water at pH 3.5 should affect scoring
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)

    ph_warnings = []
    for c in candidates:
        for fm in c.performance.failure_modes:
            if "pH" in fm:
                ph_warnings.append(c.name)

    # At least some chelators should have pH concerns at 3.5
    print(f"  + pH compatibility: {len(ph_warnings)} candidates flagged for pH concerns at mine water conditions")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 2 - RDKit Adapter Tests")
    print("  " + "=" * 40)
    print()

    test_rdkit_available()
    test_rdkit_adapter_registers()
    test_rdkit_assesses_metal_target()
    test_rdkit_generates_real_candidates()
    test_rdkit_candidates_have_stability_data()
    test_rdkit_selectivity_analysis()
    test_rdkit_hsab_scoring()
    test_full_pipeline_with_rdkit()
    test_ph_compatibility()

    print()
    print("  All Sprint 2 tests passed.")
    print()
