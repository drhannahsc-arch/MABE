"""
tests/test_sprint5.py - Cross-domain connection engine tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from core.connections import discover_connections
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def _build_full_registry():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    return registry


def test_dnazyme_gets_diagnostic_connections():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    # Find the GR-5 DNAzyme candidate
    gr5 = None
    for c in result.candidates:
        if "GR-5" in c.name:
            gr5 = c
            break

    assert gr5 is not None, "GR-5 should be in results"

    domains = [a.domain for a in gr5.other_applications]
    assert "field_diagnostic" in domains, f"GR-5 should have field diagnostic connection, got: {domains}"
    assert "electrochemical_sensor" in domains, f"GR-5 should have electrochemical connection"
    assert "fluorescent_sensor" in domains, f"GR-5 DNAzyme should have fluorescent sensor (native function)"
    print(f"  + GR-5 DNAzyme: {len(gr5.other_applications)} connections including diagnostics")


def test_chelator_gets_indicator_displacement():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    edta = None
    for c in result.candidates:
        if "EDTA" in c.name:
            edta = c
            break

    assert edta is not None
    domains = [a.domain for a in edta.other_applications]
    assert "field_diagnostic" in domains, f"EDTA should have indicator displacement diagnostic"
    print(f"  + EDTA: {len(edta.other_applications)} connections including indicator displacement")


def test_peptide_gets_diagnostic():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    peptides = [c for c in result.candidates if c.modality == "peptide_chelator"]
    if peptides:
        domains = [a.domain for a in peptides[0].other_applications]
        assert "field_diagnostic" in domains or "electrochemical_sensor" in domains
        print(f"  + Peptide {peptides[0].name}: {len(peptides[0].other_applications)} connections")


def test_research_tool_connection():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_research_tool = False
    for c in result.candidates:
        for a in c.other_applications:
            if a.domain == "research_tool":
                has_research_tool = True
                assert "ICP-MS" in a.description or "mass spec" in a.description.lower()
                break

    assert has_research_tool, "At least one candidate should have research tool connection"
    print(f"  + Research tool connections found (ICP-MS replacement)")


def test_dna_barcode_multiplexing():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_multiplex = False
    for c in result.candidates:
        for a in c.other_applications:
            if a.domain == "multiplexed_diagnostics":
                has_multiplex = True
                assert "barcode" in a.description.lower()
                break

    assert has_multiplex, "DNA-based candidates should have multiplexed diagnostic connection"
    print(f"  + DNA barcode multiplexing connections found")


def test_monitoring_network():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_monitoring = False
    for c in result.candidates:
        for a in c.other_applications:
            if a.domain == "monitoring_network":
                has_monitoring = True
                break

    assert has_monitoring, "Should have monitoring network connections"
    print(f"  + Monitoring network connections found")


def test_total_connections_reported():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    total = sum(len(c.other_applications) for c in result.candidates)
    assert total > 10, f"Expected 10+ total connections, got {total}"

    # The note should mention connections
    has_connection_note = any("cross-domain" in n.lower() for n in result.notes)
    assert has_connection_note, "Orchestrator should report total connections in notes"
    print(f"  + Total cross-domain connections: {total}")


def test_mercury_gets_lateral_flow():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("mercury removal from river water")
    result = orchestrator.solve(problem)

    has_lateral_flow = False
    for c in result.candidates:
        for a in c.other_applications:
            if "lateral flow" in a.description.lower():
                has_lateral_flow = True
                break

    assert has_lateral_flow, "Mercury binders should suggest lateral flow strip diagnostic"
    print(f"  + Mercury: lateral flow diagnostic discovered")


def test_connections_mention_cost_savings():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_cost_comparison = False
    for c in result.candidates:
        for a in c.other_applications:
            if "ICP-MS" in a.description or "$" in a.description:
                has_cost_comparison = True
                break

    assert has_cost_comparison, "Connections should mention cost savings vs traditional methods"
    print(f"  + Cost comparison to mass spec included in connections")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 5 - Cross-Domain Connection Tests")
    print("  " + "=" * 40)
    print()

    test_dnazyme_gets_diagnostic_connections()
    test_chelator_gets_indicator_displacement()
    test_peptide_gets_diagnostic()
    test_research_tool_connection()
    test_dna_barcode_multiplexing()
    test_monitoring_network()
    test_total_connections_reported()
    test_mercury_gets_lateral_flow()
    test_connections_mention_cost_savings()

    print()
    print("  All Sprint 5 tests passed.")
    print()
