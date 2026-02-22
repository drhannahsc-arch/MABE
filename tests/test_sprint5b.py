"""
tests/test_sprint5b.py - Universal lab tool connection tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def _build():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available(): registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    return registry


def test_pulldown_connection():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_pulldown = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "affinity_pulldown":
                has_pulldown = True
                assert "Fc" in a.description or "batch variation" in a.description
                break
    assert has_pulldown, "Should discover affinity pulldown application"
    print("  + Affinity pull-down connection found (replaces antibody IP)")


def test_column_capture_connection():
    o = Orchestrator(_build())
    r = o.solve(decompose("nickel capture from mine water"))
    has_column = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "affinity_column":
                has_column = True
                break
    assert has_column, "Should discover affinity column application"
    print("  + Affinity chromatography column connection found")


def test_cell_sorting_no_fc():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_cell = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "cell_capture":
                has_cell = True
                assert "Fc" in a.description
                assert "activation" in a.description.lower()
                break
    assert has_cell, "Should discover cell capture without Fc activation"
    print("  + Cell capture connection found (no Fc activation)")


def test_sample_prep_connection():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_prep = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "sample_preparation":
                has_prep = True
                assert "pre-concentration" in a.description.lower() or "magnetic" in a.description.lower()
                break
    assert has_prep, "Should discover sample prep application"
    print("  + Sample preparation connection found (replaces SPE)")


def test_process_analytical():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_pac = False
    for c in r.candidates:
        for a in c.other_applications:
            if a.domain == "process_analytical":
                has_pac = True
                break
    assert has_pac
    print("  + Process analytical (inline QC) connection found")


def test_no_fc_in_dna_peptide():
    """DNA and peptide binders should explicitly note no Fc artifacts."""
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    fc_mentions = 0
    for c in r.candidates:
        if c.modality in ("dnazyme", "dna_aptamer", "peptide_chelator", "dna_motif"):
            for a in c.other_applications:
                if "Fc" in a.description or "no Fc" in a.description:
                    fc_mentions += 1
    assert fc_mentions >= 1, "DNA/peptide candidates should mention Fc-free advantage"
    print(f"  + {fc_mentions} connections explicitly mention Fc-free advantage")


def test_total_connection_domains():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    all_domains = set()
    for c in r.candidates:
        for a in c.other_applications:
            all_domains.add(a.domain)
    print(f"  + All connection domains discovered: {sorted(all_domains)}")
    expected = {"field_diagnostic", "affinity_pulldown", "research_tool", "sample_preparation"}
    missing = expected - all_domains
    assert not missing, f"Missing expected domains: {missing}"


def test_mass_spec_replacement_mentioned():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_mass_spec = False
    for c in r.candidates:
        for a in c.other_applications:
            if "ICP-MS" in a.description or "mass spec" in a.description.lower():
                has_mass_spec = True
                break
    assert has_mass_spec
    print("  + ICP-MS/mass spec replacement explicitly mentioned")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 5b - Universal Lab Tool Tests")
    print("  " + "=" * 40)
    print()

    test_pulldown_connection()
    test_column_capture_connection()
    test_cell_sorting_no_fc()
    test_sample_prep_connection()
    test_process_analytical()
    test_no_fc_in_dna_peptide()
    test_total_connection_domains()
    test_mass_spec_replacement_mentioned()

    print()
    print("  All Sprint 5b tests passed.")
    print()
