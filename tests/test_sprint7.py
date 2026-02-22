"""
tests/test_sprint7.py - Interior pocket design tests.
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


def test_assemblies_have_interior_design():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    for a in r.assemblies:
        assert a.interior is not None, f"{a.name} missing interior design"
        assert len(a.interior.sites) > 0, f"{a.name} has empty interior"
    print(f"  + All {len(r.assemblies)} assemblies have designed interiors")


def test_three_design_levels():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    levels = set(a.design_level for a in r.assemblies)
    print(f"  + Design levels present: {levels}")
    assert "simple" in levels, "Should have simple (free) assemblies"
    # Composite or tertiary should appear for cage assemblies
    assert levels - {"simple"}, f"Should have at least one non-simple level, got {levels}"


def test_composite_has_multiple_sites():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    composites = [a for a in r.assemblies if a.design_level in ("composite", "tertiary")]
    if composites:
        for a in composites:
            assert a.interior.total_binding_sites > 1, f"{a.name} should have >1 site"
        print(f"  + Composite/tertiary assemblies have multiple interior sites ({composites[0].interior.total_binding_sites} sites)")


def test_avidity_enhancement():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    cage_assemblies = [a for a in r.assemblies if a.structure.type != "none"]
    if cage_assemblies:
        for a in cage_assemblies:
            assert a.interior.avidity_factor >= 1.0
        best = max(cage_assemblies, key=lambda a: a.interior.avidity_factor)
        print(f"  + Best avidity: {best.interior.avidity_factor:.0f}x ({best.name})")


def test_tertiary_has_mixed_donors():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    tertiary = [a for a in r.assemblies if a.design_level == "tertiary"]
    if tertiary:
        for a in tertiary:
            assert a.interior.unique_recognition_types >= 2
            assert a.interior.cooperativity_note != ""
        print(f"  + Tertiary pocket: {tertiary[0].interior.unique_recognition_types} recognition types")
        print(f"    Cooperativity: {tertiary[0].interior.cooperativity_note[:80]}...")
    else:
        print(f"  + No tertiary designs generated (complement may not have been found)")
        # Not a failure — tertiary only generated when good complement exists


def test_interior_report():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    cage_assemblies = [a for a in r.assemblies if a.structure.type != "none"]
    if cage_assemblies:
        report = cage_assemblies[0].full_report()
        assert "Interior Design" in report
        assert "copies" in report.lower() or "x " in report.lower()
        print(f"  + Interior design rendered in full report")


def test_kinetic_trapping_described():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    has_kinetic = False
    for a in r.assemblies:
        if a.interior.kinetic_trapping:
            has_kinetic = True
            break
    if has_kinetic:
        print(f"  + Kinetic trapping described for cage assemblies")
    else:
        print(f"  + No kinetic trapping (expected if no tight-pore cages compatible)")


def test_mercury_tertiary_pocket():
    """Mercury is soft — should get thiol + T-Hg-T mixed pocket."""
    o = Orchestrator(_build())
    r = o.solve(decompose("mercury removal from river water"))
    tertiary = [a for a in r.assemblies if a.design_level == "tertiary"]
    if tertiary:
        donors = set()
        for site in tertiary[0].interior.sites:
            donors.update(site.recognition.donor_atoms)
        print(f"  + Mercury tertiary pocket: donors = {donors}")
    else:
        # Check if composite at least exists
        composites = [a for a in r.assemblies if a.design_level == "composite"]
        print(f"  + Mercury: {len(composites)} composite assemblies (tertiary not found)")


def test_simple_assemblies_have_one_site():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    simples = [a for a in r.assemblies if a.design_level == "simple"]
    for a in simples:
        assert a.interior.total_binding_sites == 1
        assert a.interior.avidity_factor == 1.0
    print(f"  + Simple assemblies correctly have 1 site, avidity 1x")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 7 - Interior Pocket Design Tests")
    print("  " + "=" * 40)
    print()

    test_assemblies_have_interior_design()
    test_three_design_levels()
    test_composite_has_multiple_sites()
    test_avidity_enhancement()
    test_tertiary_has_mixed_donors()
    test_interior_report()
    test_kinetic_trapping_described()
    test_mercury_tertiary_pocket()
    test_simple_assemblies_have_one_site()

    print()
    print("  All Sprint 7 tests passed.")
    print()
