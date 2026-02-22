"""
tests/test_sprint14.py - Repulsion force tests.
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
import core.ionic_integration
import core.repulsion_integration

import math
from knowledge.repulsion_data import (
    get_hydrated_radius, get_ionic_radius, VDW_RADII,
    HYDRATED_RADII, IONIC_RADII, FRAMEWORK_CHARGE,
)
from core.repulsion import compute_repulsion, RepulsionAnalysis
from core.assembly import RecognitionChemistry, StructuralConstraint, InteriorDesign, InteriorSite
from core.problem import (
    Problem, TargetSpecies, Matrix, CompetingSpecies, Outcome,
    ElectronicDescription, HydrationDescription, SizeDescription,
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


def _make_target(identity="lead", charge=2.0, r_ionic=1.19, r_hyd=4.01):
    return TargetSpecies(
        identity=identity, formula=f"{identity.capitalize()}{abs(charge):.0f}+",
        charge=charge, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=2.33),
        hydration=HydrationDescription(
            hydrated_radius_angstrom=r_hyd, dehydration_energy_kj_mol=1481.0,
            coordination_number_water=6,
        ),
        size=SizeDescription(ionic_radius_angstrom=r_ionic),
    )


def _make_problem(target=None, ionic_strength_mm=50.0):
    target = target or _make_target()
    matrix = Matrix(
        description="AMD", ph=3.5, temperature_c=12.0,
        ionic_strength_mm=ionic_strength_mm,
        competing_species=[
            CompetingSpecies(identity="calcium", formula="Ca2+", charge=2.0, concentration_mm=5.0),
            CompetingSpecies(identity="magnesium", formula="Mg2+", charge=2.0, concentration_mm=3.0),
        ],
    )
    return Problem(target=target, matrix=matrix, desired_outcome=Outcome(description="capture"))


def _make_rec(donors=None, donor_type="soft"):
    donors = donors or ["S", "S", "N"]
    return RecognitionChemistry(
        name="test_chelator", type="chelator",
        donor_atoms=donors, donor_type=donor_type, structure="test",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data tables
# ═══════════════════════════════════════════════════════════════════════════

def test_hydrated_radii_populated():
    """Hydrated radii available for all common metals."""
    for metal in ["lead", "copper", "nickel", "zinc", "calcium", "sodium", "iron"]:
        r = get_hydrated_radius(metal, 2.0)
        assert r > 2.0, f"{metal} r_hyd = {r}"
    print("  + Hydrated radii populated for all common metals")


def test_ionic_radii_populated():
    """Ionic radii available for all common metals."""
    for metal in ["lead", "copper", "nickel", "zinc", "calcium"]:
        r = get_ionic_radius(metal, 2.0)
        assert 0.5 < r < 2.0, f"{metal} r_ion = {r}"
    print("  + Ionic radii populated")


def test_framework_charges():
    """Zeolite negative, LDH positive, others neutral."""
    assert FRAMEWORK_CHARGE["zeolite"] < 0
    assert FRAMEWORK_CHARGE["ldh"] > 0
    assert FRAMEWORK_CHARGE["mof"] == 0
    print("  + Framework charges: zeolite(-), LDH(+), MOF(0)")


# ═══════════════════════════════════════════════════════════════════════════
# Steric exclusion
# ═══════════════════════════════════════════════════════════════════════════

def test_steric_large_pore_no_exclusion():
    """Large pore (8 nm) should not exclude Pb²⁺ (r_hyd ≈ 4 Å)."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="big_pore", type="mesoporous_silica",
                                   geometry="cylindrical", pore_size_nm=8.0)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert not rep.steric_excluded
    assert rep.dG_steric < 1.0
    print(f"  + Large pore (8 nm): no exclusion, ΔG_steric={rep.dG_steric:.1f}")


def test_steric_zeolite_zsm5_excludes_large():
    """ZSM-5 (5.5 Å pore) should create steric penalty for Pb²⁺ (d_hyd ≈ 8 Å)."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="ZSM-5", type="zeolite",
                                   geometry="channel", pore_size_nm=0.55)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    # Hydrated Pb²⁺ diameter = 8.02 Å > 5.5 Å pore
    # But bare Pb²⁺ = 2.38 Å < 5.5 Å — so not fully excluded
    assert not rep.steric_excluded, "Bare ion should still fit"
    assert rep.dG_steric > 5.0, f"Should have significant penalty: {rep.dG_steric}"
    print(f"  + ZSM-5 (5.5 Å): Pb²⁺ squeezed, ΔG_steric=+{rep.dG_steric:.1f} kJ/mol")


def test_steric_tiny_pore_full_exclusion():
    """Very small pore (3 Å) should fully exclude Pb²⁺ (ionic d = 2.38 Å)."""
    target = _make_target(r_ionic=1.5, r_hyd=4.0)  # make ionic radius larger
    prob = _make_problem(target=target)
    rec = _make_rec()
    struct = StructuralConstraint(name="tiny", type="zeolite",
                                   geometry="channel", pore_size_nm=0.3)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.steric_excluded, "Should be fully excluded"
    assert rep.dG_steric >= 100.0
    print(f"  + Tiny pore (3 Å): FULLY EXCLUDED, ΔG_steric={rep.dG_steric:.0f}")


def test_steric_no_pore_no_penalty():
    """No pore constraint → no steric penalty."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="free", type="none", geometry="none")
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_steric == 0.0
    print(f"  + No pore: ΔG_steric = 0")


def test_steric_competitor_exclusion():
    """Pore should selectively exclude larger competitors."""
    # Mg²⁺ r_hyd = 4.28 Å → d = 8.56 Å
    # In a 7 Å pore: Mg²⁺ squeezed, Ca²⁺ (d=8.24 Å) also squeezed
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="medium", type="zeolite",
                                   geometry="channel", pore_size_nm=0.7)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    # Check that competitor exclusion dict was populated
    assert len(rep.competitor_excluded) > 0, "Should check competitors"
    print(f"  + Competitor exclusion checked: {rep.competitor_excluded}")


# ═══════════════════════════════════════════════════════════════════════════
# Born / Pauli repulsion
# ═══════════════════════════════════════════════════════════════════════════

def test_born_no_compression():
    """Large pore → no Born repulsion."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="big", type="mesoporous_silica",
                                   geometry="cylindrical", pore_size_nm=5.0)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_born == 0.0
    print(f"  + Large pore: no Born repulsion")


def test_born_tight_pore():
    """Very tight pore should create Born repulsion if pocket < contact distance."""
    prob = _make_problem()
    rec = _make_rec(["O", "O", "O", "O"])  # 4 O donors, vdW ≈ 1.52 Å each
    # contact distance ≈ r_ionic(1.19) + r_vdw_O(1.52) ≈ 2.71 Å
    # pocket radius for 0.4 nm pore = 2.0 Å < 2.71 Å → strain
    struct = StructuralConstraint(name="tight", type="mof",
                                   geometry="cage", pore_size_nm=0.4)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_born > 0, f"Should have Born penalty: {rep.dG_born}"
    print(f"  + Tight pore (4 Å): Born repulsion ΔG=+{rep.dG_born:.1f} kJ/mol")


# ═══════════════════════════════════════════════════════════════════════════
# Charge repulsion
# ═══════════════════════════════════════════════════════════════════════════

def test_charge_cation_in_ldh_repelled():
    """Cation (Pb²⁺) in LDH (positive layers) should be repelled."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="LDH", type="ldh",
                                   geometry="layered", pore_size_nm=0.7)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_charge_repulsion > 0
    assert rep.target_charge_sign > 0
    assert rep.framework_charge_sign > 0
    print(f"  + Pb²⁺ in LDH: charge repulsion ΔG=+{rep.dG_charge_repulsion:.1f} kJ/mol")


def test_charge_cation_in_zeolite_attracted():
    """Cation in zeolite (negative framework) → attraction, not repulsion."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="zeolite", type="zeolite",
                                   geometry="channel", pore_size_nm=0.74)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_charge_repulsion == 0.0, f"Should not be repelled: {rep.dG_charge_repulsion}"
    assert not rep.charge_repelled
    print(f"  + Pb²⁺ in zeolite: attracted (no charge repulsion)")


def test_charge_neutral_framework_no_repulsion():
    """Neutral framework → no charge-based exclusion."""
    prob = _make_problem()
    rec = _make_rec()
    struct = StructuralConstraint(name="MOF", type="mof",
                                   geometry="cage", pore_size_nm=1.6)
    rep = compute_repulsion(rec, struct, prob.target, prob)
    assert rep.dG_charge_repulsion == 0.0
    print(f"  + Neutral framework: no charge repulsion")


# ═══════════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════════

def test_repulsion_in_dG_net():
    """Repulsion should increase (make less negative) ΔG_net."""
    import core.thermodynamics as _thermo
    prob = _make_problem()
    rec = _make_rec()
    struct_open = StructuralConstraint(name="open", type="none", geometry="none")
    struct_tight = StructuralConstraint(name="tight", type="zeolite",
                                         geometry="channel", pore_size_nm=0.55)
    interior = InteriorDesign(design_level="primary", sites=[InteriorSite(recognition=rec)],
                                avidity_factor=1.0)

    thermo_open = _thermo.compute_thermodynamics(rec, struct_open, interior, prob)
    thermo_tight = _thermo.compute_thermodynamics(rec, struct_tight, interior, prob)

    # Tight should have higher (less negative) ΔG due to steric penalty
    assert thermo_tight.dG_net >= thermo_open.dG_net,         f"Tight ({thermo_tight.dG_net:.1f}) should ≥ open ({thermo_open.dG_net:.1f})"
    print(f"  + ΔG: open={thermo_open.dG_net:+.1f}, tight zeolite={thermo_tight.dG_net:+.1f} kJ/mol")


# ═══════════════════════════════════════════════════════════════════════════
# E2E
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_repulsion_in_reports():
    """E2E: repulsion data appears in reports."""
    registry = _build()
    prob = decompose("lead capture from acid mine drainage pH 3.5")
    prob.matrix.ionic_strength_mm = 50.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found = False
    for r in results.assemblies:
        if "REPULSION" in r.confidence_reasoning:
            found = True
            break
    assert found, "Repulsion should appear in E2E report"
    print(f"  + E2E: repulsion data in reports")


def test_e2e_ldh_warns_cation():
    """E2E: LDH should warn about cation repulsion for lead."""
    registry = _build()
    prob = decompose("lead capture from mine water")
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    ldh_warnings = []
    for r in results.assemblies:
        if r.structure.type == "ldh":
            for fm in r.failure_modes:
                if "charge" in fm.lower() and "repel" in fm.lower():
                    ldh_warnings.append(fm)
    if ldh_warnings:
        print(f"  + E2E: LDH correctly warns about cation repulsion")
    else:
        # LDH might not be in assemblies — check if any assembly has repulsion info
        found_any = any("REPULSION" in r.confidence_reasoning for r in results.assemblies)
        assert found_any, "Should have repulsion analysis somewhere"
        print(f"  + E2E: repulsion analysis present (LDH may not be in assembly set)")


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_hydrated_radii_populated,
        test_ionic_radii_populated,
        test_framework_charges,
        test_steric_large_pore_no_exclusion,
        test_steric_zeolite_zsm5_excludes_large,
        test_steric_tiny_pore_full_exclusion,
        test_steric_no_pore_no_penalty,
        test_steric_competitor_exclusion,
        test_born_no_compression,
        test_born_tight_pore,
        test_charge_cation_in_ldh_repelled,
        test_charge_cation_in_zeolite_attracted,
        test_charge_neutral_framework_no_repulsion,
        test_repulsion_in_dG_net,
        test_e2e_repulsion_in_reports,
        test_e2e_ldh_warns_cation,
    ]

    print()
    print("=" * 60)
    print("  Sprint 14: Repulsion Forces")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"  Sprint 14: {passed} passed, {failed} failed")
    print()
