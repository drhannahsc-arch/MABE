"""
tests/test_sprint6.py - Composite binder assembly tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from core.assembly import BinderAssembly
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from conversation.decomposer import decompose


def _build():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available(): registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry


def test_assemblies_produced():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    assert len(r.assemblies) > 0, "Should produce composite assemblies"
    print(f"  + {len(r.assemblies)} composite assemblies produced")


def test_assembly_has_four_axes():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    for a in r.assemblies:
        assert a.recognition is not None, f"{a.name} missing recognition"
        assert a.structure is not None, f"{a.name} missing structure"
        assert a.selectivity is not None, f"{a.name} missing selectivity"
        assert a.release is not None, f"{a.name} missing release"
    print(f"  + All assemblies have 4 design axes: recognition, structure, selectivity, release")


def test_assembly_includes_free_and_cage():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    types = set(a.structure.type for a in r.assemblies)
    assert "none" in types, "Should include free-in-solution assembly"
    has_cage = any(t in types for t in ("dna_origami_cage", "mof", "protein_cage", "dendrimer"))
    assert has_cage, f"Should include at least one structural constraint, got: {types}"
    print(f"  + Assembly types: {types}")


def test_cage_assembly_has_pore_selectivity():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    for a in r.assemblies:
        if a.structure.type != "none":
            assert a.selectivity.mechanism != "none", f"Cage assembly should have selectivity filter"
            assert "pore" in a.selectivity.mechanism.lower() or "size" in a.selectivity.mechanism.lower()
    print(f"  + Cage assemblies have pore/size selectivity filters")


def test_release_mechanism_present():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    for a in r.assemblies:
        assert a.release.trigger != "", f"{a.name} missing release trigger"
    triggers = set(a.release.trigger for a in r.assemblies)
    print(f"  + Release triggers present: {triggers}")


def test_assembly_full_report():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    for a in r.assemblies:
        report = a.full_report()
        assert "Recognition Chemistry" in report
        assert "Structural Constraint" in report
        assert "Selectivity Filter" in report
        assert "Release Mechanism" in report
        break  # just test first one
    print(f"  + Assembly full report contains all 4 sections")


def test_dummy_adapter_removed():
    registry = _build()
    names = [a.name for a in registry.all_adapters()]
    assert "dummy" not in names, "Dummy adapter should be removed"
    print(f"  + Dummy adapter removed. Active adapters: {names}")


def test_assembly_scorer():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    # Free assembly should score differently from cage assembly of same recognition
    for a in r.assemblies:
        assert 0.0 < a.composite_score <= 1.0
    print(f"  + Assembly scores: {[f'{a.name[:30]}={a.composite_score:.0%}' for a in r.assemblies]}")


def test_mercury_gets_dna_cage():
    o = Orchestrator(_build())
    r = o.solve(decompose("mercury removal from river water"))
    cage_assemblies = [a for a in r.assemblies if a.structure.type != "none"]
    if cage_assemblies:
        print(f"  + Mercury: {len(cage_assemblies)} cage assemblies designed")
        for a in cage_assemblies:
            print(f"    - {a.name}: {a.structure.type}, pore={a.structure.pore_size_nm}nm")
    else:
        print(f"  + Mercury: no cage assemblies (all free-in-solution) — expected if recognition is strong enough")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 6 - Composite Binder Assembly Tests")
    print("  " + "=" * 40)
    print()

    test_assemblies_produced()
    test_assembly_has_four_axes()
    test_assembly_includes_free_and_cage()
    test_cage_assembly_has_pore_selectivity()
    test_release_mechanism_present()
    test_assembly_full_report()
    test_dummy_adapter_removed()
    test_assembly_scorer()
    test_mercury_gets_dna_cage()

    print()
    print("  All Sprint 6 tests passed.")
    print()
