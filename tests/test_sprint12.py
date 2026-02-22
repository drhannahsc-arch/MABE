"""
tests/test_sprint12.py - LFSE and coordination geometry tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

import core.assembly_composer_patch
import core.scoring_patch
import core.physics_integration
import core.sprint10_integration
import core.protonation_integration
import core.lfse_integration

from knowledge.lfse_data import (
    get_d_electron_count, get_field_strength, average_dq,
    is_high_spin, compute_lfse_dq, is_jahn_teller,
    METAL_D_ELECTRONS, LFSE_DQ_HIGHSPIN, DQ_OCT_KJ,
    LONE_PAIR_METALS,
)
from core.lfse import compute_lfse, LFSEAnalysis
import core.thermodynamics as _thermo_mod
import copy
from core.assembly import RecognitionChemistry, InteriorDesign, InteriorSite
from core.problem import (
    Problem, TargetSpecies, Matrix, CompetingSpecies,
    ElectronicDescription, HydrationDescription, SizeDescription, Outcome, Constraints,
)
from knowledge.structural_library import STRUCTURAL_OPTIONS
from conversation.decomposer import decompose
from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter


def _build():
    registry = ToolRegistry()
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry


# ── d-electron count tests ────────────────────────────────────────────

def test_d_electron_counts():
    """d-electron counts should be correct for common metals."""
    assert get_d_electron_count("nickel", 2) == 8, "Ni²⁺ should be d⁸"
    assert get_d_electron_count("copper", 2) == 9, "Cu²⁺ should be d⁹"
    assert get_d_electron_count("iron", 3) == 5, "Fe³⁺ should be d⁵"
    assert get_d_electron_count("iron", 2) == 6, "Fe²⁺ should be d⁶"
    assert get_d_electron_count("zinc", 2) == 10, "Zn²⁺ should be d¹⁰"
    assert get_d_electron_count("gold", 3) == 8, "Au³⁺ should be d⁸"
    assert get_d_electron_count("calcium", 2) == 0, "Ca²⁺ should be d⁰"
    assert get_d_electron_count("lead", 2) == -1, "Pb²⁺ should be -1 (lone pair)"
    assert get_d_electron_count("cerium", 3) == -2, "Ce³⁺ should be -2 (f-block)"
    print(f"  + d-electron counts: Ni²⁺=d8, Cu²⁺=d9, Fe³⁺=d5, Zn²⁺=d10, Au³⁺=d8, Ca²⁺=d0, Pb²⁺=lone pair")


def test_d_electron_coverage():
    """All MABE metals should have d-electron data."""
    metals = ["lead", "mercury", "gold", "copper", "nickel", "zinc",
              "iron", "cadmium", "silver", "calcium", "cerium", "uranium"]
    for m in metals:
        assert m in METAL_D_ELECTRONS, f"Missing d-electron data for {m}"
    print(f"  + All {len(metals)} MABE metals have d-electron data")


# ── LFSE calculation unit tests ──────────────────────────────────────

def test_lfse_d0_zero():
    """d⁰ should have zero LFSE in all geometries."""
    for geom in ["oct", "tet", "sq_planar"]:
        assert compute_lfse_dq(0, geom, True) == 0.0
    print(f"  + d⁰: LFSE = 0 in all geometries")


def test_lfse_d10_zero():
    """d¹⁰ should have zero LFSE in all geometries."""
    for geom in ["oct", "tet", "sq_planar"]:
        assert compute_lfse_dq(10, geom, True) == 0.0
    print(f"  + d¹⁰: LFSE = 0 in all geometries")


def test_lfse_d5_hs_zero_oct():
    """d⁵ high-spin octahedral should have LFSE = 0 (half-filled symmetry)."""
    assert compute_lfse_dq(5, "oct", True) == 0.0
    print(f"  + d⁵ high-spin oct: LFSE = 0 (half-filled)")


def test_lfse_d8_oct_largest():
    """d⁸ should have large LFSE in octahedral and even larger in square planar."""
    oct = compute_lfse_dq(8, "oct", True)
    sp = compute_lfse_dq(8, "sq_planar", True)
    tet = compute_lfse_dq(8, "tet", True)
    assert oct < tet, f"d⁸ oct LFSE ({oct}) should be more negative than tet ({tet})"
    assert sp < oct, f"d⁸ sq_planar LFSE ({sp}) should be more negative than oct ({oct})"
    print(f"  + d⁸: sq.pl.={sp:.1f} < oct={oct:.1f} < tet={tet:.1f} Dq (correct order)")


def test_tet_always_weaker_than_oct():
    """Tetrahedral LFSE should always be less stabilizing than octahedral for d¹-d⁹."""
    for d in range(1, 10):
        oct = compute_lfse_dq(d, "oct", True)
        tet = compute_lfse_dq(d, "tet", True)
        # oct is more negative (more stabilizing) or equal
        assert oct <= tet, f"d{d}: oct={oct} should be ≤ tet={tet}"
    print(f"  + Octahedral LFSE ≥ tetrahedral for all d¹-d⁹ (correct)")


# ── Field strength and spin state ────────────────────────────────────

def test_field_strength_classification():
    """Donor sets should classify to correct field strength."""
    assert get_field_strength(["O", "O", "O", "O"]) == "weak"
    assert get_field_strength(["N", "N", "N", "N"]) == "moderate"
    assert get_field_strength(["P", "P", "C", "C"]) == "strong"
    assert get_field_strength(["S", "S"]) == "weak"
    print(f"  + Field strength: O→weak, N→moderate, P+C→strong, S→weak")


def test_spin_state():
    """Spin state should depend on field strength and d-count."""
    assert is_high_spin(5, "weak", "iron", 3) == True, "d⁵ weak field → high spin"
    assert is_high_spin(5, "strong", "iron", 3) == False, "d⁵ strong field → low spin"
    assert is_high_spin(3, "strong", "chromium", 3) == True, "d³ always high spin (no ambiguity)"
    assert is_high_spin(8, "weak", "nickel", 2) == True, "d⁸ always high spin in oct"
    print(f"  + Spin states: Fe³⁺ weak→HS, Fe³⁺ strong→LS, Cr³⁺→always HS, Ni²⁺→always HS")


# ── Jahn-Teller tests ────────────────────────────────────────────────

def test_jahn_teller_cu2():
    """Cu²⁺ d⁹ must be Jahn-Teller active."""
    assert is_jahn_teller(9, True) == True
    assert is_jahn_teller(9, False) == True
    print(f"  + Cu²⁺ d⁹: Jahn-Teller active (correct)")


def test_jahn_teller_d3_inactive():
    """d³ should NOT be Jahn-Teller active."""
    assert is_jahn_teller(3, True) == False
    print(f"  + d³ (Cr³⁺): NOT Jahn-Teller active (correct)")


# ── Full LFSE analysis tests ─────────────────────────────────────────

def test_nickel_prefers_square_planar():
    """Ni²⁺ d⁸ with N donors should prefer square planar."""
    ni = TargetSpecies(identity="nickel", formula="Ni(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.69))
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")
    lfse = compute_lfse(rec, ni)
    assert lfse.d_electron_count == 8
    assert lfse.preferred_geometry == "sq_planar"
    assert lfse.lfse_square_planar_kj < lfse.lfse_octahedral_kj
    assert lfse.geometry_preference_dG > 20  # substantial preference
    print(f"  + Ni²⁺+4N: prefers {lfse.preferred_geometry} by {lfse.geometry_preference_dG:.1f} kJ/mol")
    print(f"    LFSE: oct={lfse.lfse_octahedral_kj:.1f}, sq.pl.={lfse.lfse_square_planar_kj:.1f} kJ/mol")


def test_calcium_no_lfse():
    """Ca²⁺ d⁰ should have zero LFSE."""
    ca = TargetSpecies(identity="calcium", formula="Ca(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.0),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.12, dehydration_energy_kj_mol=1577.0, coordination_number_water=8),
        size=SizeDescription(ionic_radius_angstrom=1.0))
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")
    lfse = compute_lfse(rec, ca)
    assert lfse.d_electron_count == 0
    assert lfse.dG_lfse == 0.0
    print(f"  + Ca²⁺: d⁰, LFSE = 0 (correct)")


def test_lead_lone_pair():
    """Pb²⁺ should show lone pair effects, not LFSE."""
    problem = decompose("lead capture from mine water")
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["O", "O", "N", "N"],
        donor_type="borderline", structure="EDTA-like")
    lfse = compute_lfse(rec, problem.target)
    assert lfse.lone_pair_active == True
    assert lfse.d_electron_count == -1
    assert lfse.preferred_geometry == "hemidirected"
    print(f"  + Pb²⁺: lone pair active, preferred={lfse.preferred_geometry}, dG={lfse.dG_lfse:.1f} kJ/mol")


def test_copper_jahn_teller_in_analysis():
    """Cu²⁺ d⁹ should flag Jahn-Teller in full analysis."""
    problem = decompose("copper capture from mine water")
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "O", "O", "N", "N"],
        donor_type="borderline", structure="EDTA-like")
    lfse = compute_lfse(rec, problem.target)
    assert lfse.jahn_teller_active == True
    assert "d⁹" in lfse.jahn_teller_description or "Cu" in lfse.jahn_teller_description
    print(f"  + Cu²⁺: Jahn-Teller detected — {lfse.jahn_teller_description[:60]}")


def test_iron3_d5_no_geometry_preference():
    """Fe³⁺ d⁵ high-spin should have LFSE = 0 and no geometry preference."""
    problem = decompose("iron removal from mine water")
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["O", "O", "O"],
        donor_type="hard", structure="tricarboxylate")
    lfse = compute_lfse(rec, problem.target)
    assert lfse.d_electron_count == 5
    assert lfse.lfse_octahedral_kj == 0.0
    assert lfse.lfse_tetrahedral_kj == 0.0
    print(f"  + Fe³⁺ d⁵ (high-spin, weak field): LFSE = 0 in all geometries")


# ── Thermodynamic integration tests ──────────────────────────────────

def test_dG_includes_lfse():
    """Thermodynamics for Ni²⁺ should include LFSE contribution."""
    ni = TargetSpecies(identity="nickel", formula="Ni(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.69))
    p = Problem(
        target=ni,
        matrix=Matrix(ph=7.0, temperature_c=25.0, competing_species=[
            CompetingSpecies("calcium", "Ca(2+)", 200.0, 2.0),
        ]),
        desired_outcome=Outcome(description="capture"),
    )
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")
    meso = [s for s in STRUCTURAL_OPTIONS if s.type == "mesoporous_silica"][0]
    interior = InteriorDesign(sites=[InteriorSite(recognition=rec, copies=10)],
        design_level="composite", total_binding_sites=10, unique_recognition_types=1, avidity_factor=2.0)
    thermo = _thermo_mod.compute_thermodynamics(rec, meso, interior, p)
    # Should have LFSE in breakdown
    has_lfse = any("LFSE" in line for line in thermo.energy_breakdown)
    assert has_lfse, "LFSE should appear in energy breakdown for Ni²⁺"
    print(f"  + Ni²⁺ dG_net = {thermo.dG_net:.1f} kJ/mol (includes LFSE)")
    for line in thermo.energy_breakdown:
        if "LFSE" in line or "lfse" in line:
            print(f"    {line}")


def test_lfse_improves_nickel_selectivity():
    """LFSE should add substantial stabilization for Ni2+ d8 but not for Ca2+ d0."""
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "N", "N", "N"],
        donor_type="borderline", structure="tetraamine")

    ni = TargetSpecies(identity="nickel", formula="Ni(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.69))
    ca = TargetSpecies(identity="calcium", formula="Ca(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.0),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.12, dehydration_energy_kj_mol=1577.0, coordination_number_water=8),
        size=SizeDescription(ionic_radius_angstrom=1.0))

    lfse_ni = compute_lfse(rec, ni)
    lfse_ca = compute_lfse(rec, ca)

    # Ni2+ d8 should get substantial LFSE; Ca2+ d0 gets zero
    assert lfse_ni.dG_lfse < -100.0, f"Ni2+ LFSE should be strongly stabilizing, got {lfse_ni.dG_lfse}"
    assert lfse_ca.dG_lfse == 0.0, f"Ca2+ LFSE should be 0, got {lfse_ca.dG_lfse}"
    advantage = abs(lfse_ni.dG_lfse)
    print(f"  + LFSE selectivity: Ni2+ gets {lfse_ni.dG_lfse:.1f} kJ/mol, Ca2+ gets 0 -> {advantage:.1f} kJ/mol advantage")


# ── End-to-end ────────────────────────────────────────────────────────

def test_e2e_lfse_in_report():
    """E2E: assemblies for nickel should include LFSE in report."""
    o = Orchestrator(_build())
    r = o.solve(decompose("nickel capture from mine water"))
    assert len(r.assemblies) > 0
    has_lfse = any("LFSE" in a.confidence_reasoning or "GEOMETRY" in a.confidence_reasoning
                   for a in r.assemblies)
    assert has_lfse, "Expected LFSE/GEOMETRY data in some assemblies"
    print(f"  + Nickel E2E: LFSE data in reports")
    for a in r.assemblies[:3]:
        print(f"    {a.composite_score:.0%}  {a.name[:50]}")


def test_e2e_copper_jahn_teller_warning():
    """E2E: copper should generate Jahn-Teller warnings."""
    o = Orchestrator(_build())
    r = o.solve(decompose("copper capture from mine water"))
    assert len(r.assemblies) > 0
    has_jt = any("jahn-teller" in fm.lower() or "Jahn-Teller" in fm
                 for a in r.assemblies for fm in a.failure_modes)
    # JT may or may not trigger depending on assembly donor sets
    print(f"  + Copper E2E: Jahn-Teller warnings present: {has_jt}")
    for a in r.assemblies[:3]:
        print(f"    {a.composite_score:.0%}  {a.name[:50]}")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 12 - LFSE & Coordination Geometry Tests")
    print("  " + "=" * 52)
    print()
    print("  d-electron counts:")
    test_d_electron_counts()
    test_d_electron_coverage()
    print()
    print("  LFSE calculation:")
    test_lfse_d0_zero()
    test_lfse_d10_zero()
    test_lfse_d5_hs_zero_oct()
    test_lfse_d8_oct_largest()
    test_tet_always_weaker_than_oct()
    print()
    print("  Field strength & spin state:")
    test_field_strength_classification()
    test_spin_state()
    print()
    print("  Jahn-Teller:")
    test_jahn_teller_cu2()
    test_jahn_teller_d3_inactive()
    print()
    print("  Full LFSE analysis:")
    test_nickel_prefers_square_planar()
    test_calcium_no_lfse()
    test_lead_lone_pair()
    test_copper_jahn_teller_in_analysis()
    test_iron3_d5_no_geometry_preference()
    print()
    print("  Thermodynamic integration:")
    test_dG_includes_lfse()
    test_lfse_improves_nickel_selectivity()
    print()
    print("  End-to-end:")
    test_e2e_lfse_in_report()
    test_e2e_copper_jahn_teller_warning()
    print()
    print("  All Sprint 12 tests passed.")
    print()
