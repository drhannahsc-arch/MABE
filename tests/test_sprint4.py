"""
tests/test_sprint4.py - Peptide + Aptamer adapter tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def test_peptide_finds_lead():
    adapter = PeptideAdapter()
    problem = decompose("lead capture from mine water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    names = [c.name for c in candidates]
    print(f"  + Lead peptides: {len(candidates)} found: {', '.join(names)}")


def test_peptide_finds_nickel():
    adapter = PeptideAdapter()
    problem = decompose("nickel capture from mine water")
    candidates = adapter.generate_candidates(problem)
    his_tags = [c for c in candidates if "His" in c.name]
    assert len(his_tags) >= 1, "His-tag should match nickel"
    print(f"  + Nickel peptides: {len(candidates)} found, including His-tags")


def test_peptide_finds_mercury():
    adapter = PeptideAdapter()
    problem = decompose("mercury capture from river water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    # Should find Cys-rich peptides for soft Hg2+
    soft = [c for c in candidates if "soft" in c.description.lower() or "Cys" in c.description or "thiol" in c.description.lower()]
    assert len(soft) >= 1, "Should find Cys/thiol peptides for mercury"
    print(f"  + Mercury peptides: {len(candidates)} found, {len(soft)} with soft donors")


def test_peptide_biodegradable():
    adapter = PeptideAdapter()
    problem = decompose("cadmium removal from mine water")
    candidates = adapter.generate_candidates(problem)
    for c in candidates:
        assert "biodegradable" in c.accessibility.end_of_life.lower()
    print(f"  + All peptide candidates are biodegradable")


def test_aptamer_finds_arsenic():
    adapter = AptamerAdapter()
    problem = decompose("arsenic removal from mine water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    assert any("Ars" in c.name for c in candidates)
    print(f"  + Arsenic aptamer found: {candidates[0].name}")


def test_aptamer_finds_mercury():
    adapter = AptamerAdapter()
    problem = decompose("mercury removal from river water")
    candidates = adapter.generate_candidates(problem)
    assert len(candidates) >= 1
    print(f"  + Mercury aptamer found: {candidates[0].name}")


def test_aptamer_capture_ready():
    adapter = AptamerAdapter()
    problem = decompose("mercury removal from river water")
    candidates = adapter.generate_candidates(problem)
    for c in candidates:
        # Aptamers should NOT have "capture not validated" warnings
        capture_warnings = [fm for fm in c.performance.failure_modes if "capture" in fm.lower() and "not" in fm.lower()]
        assert len(capture_warnings) == 0, f"Aptamer should be capture-ready, but got: {capture_warnings}"
    print(f"  + Aptamers are capture-ready (no modification needed)")


def test_full_pipeline_five_adapters():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)

    problem = decompose("mercury removal from river water")
    result = orchestrator.solve(problem)
    sources = set(c.source_tool for c in result.candidates)
    modalities = set(c.modality for c in result.candidates)
    print(f"  + Full pipeline: {len(result.candidates)} candidates from {len(sources)} tools")
    print(f"    Tools: {sources}")
    print(f"    Modalities: {modalities}")


def test_lead_all_modalities():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)

    problem = decompose("lead capture and release from mine water in BC")
    result = orchestrator.solve(problem)
    modalities = set(c.modality for c in result.candidates)
    print(f"  + Lead (all modalities): {len(result.candidates)} candidates")
    print(f"    Modalities: {modalities}")
    # Should have at least: chelator, dnazyme, peptide_chelator, designed_protein, nanocage
    assert len(modalities) >= 4, f"Expected 4+ modalities, got {modalities}"


if __name__ == "__main__":
    print()
    print("  MABE Sprint 4 - Peptide + Aptamer Tests")
    print("  " + "=" * 40)
    print()

    test_peptide_finds_lead()
    test_peptide_finds_nickel()
    test_peptide_finds_mercury()
    test_peptide_biodegradable()
    test_aptamer_finds_arsenic()
    test_aptamer_finds_mercury()
    test_aptamer_capture_ready()
    test_full_pipeline_five_adapters()
    test_lead_all_modalities()

    print()
    print("  All Sprint 4 tests passed.")
    print()
