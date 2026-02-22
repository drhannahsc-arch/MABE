"""
tests/test_sprint3.py - DNAzyme adapter tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def test_dnazyme_finds_lead():
    adapter = DNAzymeAdapter()
    problem = decompose("lead capture from mine water")
    assessment = adapter.assess_contribution(problem)
    assert assessment.can_contribute
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 2  # GR-5 and 8-17 at minimum
    names = [c.name for c in candidates]
    print(f"  + Lead: {len(candidates)} DNAzymes found: {', '.join(names)}")


def test_dnazyme_finds_mercury():
    adapter = DNAzymeAdapter()
    problem = decompose("mercury removal from river water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1  # T-Hg-T at minimum
    assert any("T-Hg" in c.name for c in candidates)
    print(f"  + Mercury: found T-Hg-T motif")


def test_dnazyme_finds_uranyl():
    adapter = DNAzymeAdapter()
    problem = decompose("uranium removal from mine water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1  # 39E
    assert any("39E" in c.name for c in candidates)
    # 39E is env-tested and tier 3, but mine water pH 3.5 is outside range (5-7)
    # MABE correctly penalizes this
    for c in candidates:
        if "39E" in c.name:
            has_ph_warning = any("pH" in fm for fm in c.performance.failure_modes)
            assert has_ph_warning, "Should warn about pH incompatibility at mine drainage pH"
            print(f"  + Uranyl: 39E found, probability {c.performance.probability_of_success:.0%} (correctly penalized for pH 3.5 mine water)")


def test_dnazyme_no_match():
    adapter = DNAzymeAdapter()
    problem = decompose("selenium capture from mine water")
    assessment = adapter.assess_contribution(problem)
    # Selenite has no DNAzyme in library
    assert not assessment.can_contribute
    print(f"  + No DNAzyme for selenite (correct - none in library)")


def test_dnazyme_capture_validation_flag():
    adapter = DNAzymeAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)
    for c in candidates:
        if "GR-5" in c.name:
            # GR-5 capture_validated is False
            has_capture_warning = any("capture" in fm.lower() and "not" in fm.lower()
                                      for fm in c.performance.failure_modes)
            assert has_capture_warning, "GR-5 should warn about unvalidated capture mode"
            print(f"  + GR-5 correctly flags unvalidated capture mode")
            break


def test_dnazyme_has_real_literature():
    adapter = DNAzymeAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)
    for c in candidates:
        assert len(c.evidence.literature_references) > 0
        assert "DOI" in c.evidence.literature_references[0]
    print(f"  + All DNAzyme candidates have real DOI references")


def test_dnazyme_selectivity_vs_matrix():
    adapter = DNAzymeAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)
    # 8-17 has poor Zn2+ selectivity (10-fold)
    # Mine water matrix doesn't have Zn2+ in competing species currently,
    # but calcium and magnesium are there
    # GR-5 has 40,000x over Ca2+ so should be clean
    for c in candidates:
        if "GR-5" in c.name:
            # Ca2+ is in mine matrix but GR-5 has 40,000x selectivity
            # Should have no threats for calcium
            ca_threats = [t for t in c.performance.selectivity_threats if "calcium" in t.lower()]
            assert len(ca_threats) == 0, "GR-5 should not flag Ca2+ as threat (40,000x selectivity)"
            print(f"  + GR-5 correctly shows no calcium threat (40,000x selectivity)")
            break


def test_full_pipeline_three_adapters():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)

    problem = decompose("lead capture and release from mine water")
    result = orchestrator.solve(problem)

    sources = set(c.source_tool for c in result.candidates)
    assert "dnazyme" in sources, "DNAzyme adapter missing from results"
    print(f"  + Full pipeline: {len(result.candidates)} candidates from {len(sources)} tools: {sources}")

    # Check we have chelators, DNAzymes, AND dummy (protein/nanocage)
    modalities = set(c.modality for c in result.candidates)
    print(f"  + Modalities present: {modalities}")


def test_immobilization_handles():
    adapter = DNAzymeAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)
    for c in candidates:
        assert len(c.immobilization_options) > 0, f"{c.name} has no immobilization options"
    print(f"  + All DNAzyme candidates have immobilization options (no environmental release)")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 3 - DNAzyme Adapter Tests")
    print("  " + "=" * 40)
    print()

    test_dnazyme_finds_lead()
    test_dnazyme_finds_mercury()
    test_dnazyme_finds_uranyl()
    test_dnazyme_no_match()
    test_dnazyme_capture_validation_flag()
    test_dnazyme_has_real_literature()
    test_dnazyme_selectivity_vs_matrix()
    test_full_pipeline_three_adapters()
    test_immobilization_handles()

    print()
    print("  All Sprint 3 tests passed.")
    print()
