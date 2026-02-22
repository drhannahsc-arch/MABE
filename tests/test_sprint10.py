"""
tests/test_sprint10.py - Kinetic response dynamics tests.
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

from knowledge.electronic_data import get_electronic_data, enrich_target_electronic, DONOR_HOMO
from core.orbital_binding import compute_orbital_binding, OrbitalAnalysis
from core.kinetics import compute_kinetics, KineticProfile
from core.probability_chain import compute_probability_chain, ProbabilityChain
from core.thermodynamics import compute_thermodynamics
from core.hydrodynamics import compute_hydrodynamics
from core.assembly import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign, InteriorSite,
    BinderAssembly, SelectivityFilter, ReleaseMechanism,
)
from core.problem import TargetSpecies, ElectronicDescription, HydrationDescription, SizeDescription
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


# ── Electronic data tests ─────────────────────────────────────────

def test_dft_data_exists():
    """DFT data should exist for common metals."""
    for metal in ["lead", "gold", "mercury", "copper", "zinc", "calcium"]:
        data = get_electronic_data(metal)
        assert data, f"No DFT data for {metal}"
        assert "homo_ev" in data
        assert "lumo_ev" in data
        assert "polarizability" in data
    print(f"  + DFT data present for 6 common metals")


def test_enrich_populates_target():
    """enrich_target_electronic should fill None fields."""
    problem = decompose("lead capture from mine water")
    assert problem.target.electronic.homo_ev is None
    enrich_target_electronic(problem.target)
    assert problem.target.electronic.homo_ev is not None
    assert problem.target.electronic.lumo_ev is not None
    assert problem.target.electronic.polarizability is not None
    print(f"  + Lead enriched: HOMO={problem.target.electronic.homo_ev}, LUMO={problem.target.electronic.lumo_ev}, alpha={problem.target.electronic.polarizability}")


def test_gold_lower_lumo_than_calcium():
    """Soft metals (gold) should have lower (more negative) LUMO than hard metals (calcium)."""
    au = get_electronic_data("gold")
    ca = get_electronic_data("calcium")
    assert au["lumo_ev"] < ca["lumo_ev"], "Gold LUMO should be lower (better acceptor)"
    print(f"  + Gold LUMO={au['lumo_ev']:.1f} < Calcium LUMO={ca['lumo_ev']:.1f} (soft accepts better)")


# ── Orbital binding tests ─────────────────────────────────────────

def test_orbital_soft_match():
    """S donors + gold (soft-soft) should show covalent character."""
    problem = decompose("gold recovery from mine tailings")
    enrich_target_electronic(problem.target)
    rec = RecognitionChemistry(name="thiol", type="chelator", donor_atoms=["S", "S"],
        donor_type="soft", structure="dithiol")
    orbital = compute_orbital_binding(rec, problem.target)
    assert orbital.dft_data_available
    assert orbital.bond_character in ("covalent", "mixed")
    assert orbital.covalent_fraction > 0.5
    print(f"  + Gold+S: {orbital.bond_character} ({orbital.covalent_fraction:.0%} covalent), CT dG={orbital.charge_transfer_dG_kj:.1f} kJ/mol")


def test_orbital_hard_match():
    """O donors + calcium (hard-hard) should show ionic character."""
    ca = TargetSpecies(identity="calcium", formula="Ca(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.0),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.12, dehydration_energy_kj_mol=1577.0, coordination_number_water=8),
        size=SizeDescription(ionic_radius_angstrom=1.0))
    enrich_target_electronic(ca)
    rec = RecognitionChemistry(name="carboxylate", type="chelator", donor_atoms=["O", "O", "O"],
        donor_type="hard", structure="tricarboxylate")
    orbital = compute_orbital_binding(rec, ca)
    assert orbital.bond_character == "ionic"
    assert orbital.covalent_fraction < 0.5
    print(f"  + Ca+O: {orbital.bond_character} ({orbital.covalent_fraction:.0%} covalent)")


def test_charge_transfer_favorable_for_good_match():
    """Good donor-metal HOMO-LUMO gap should give negative CT energy."""
    problem = decompose("lead capture from mine water")
    enrich_target_electronic(problem.target)
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "S"],
        donor_type="borderline", structure="NS-chelator")
    orbital = compute_orbital_binding(rec, problem.target)
    assert orbital.charge_transfer_favorable
    assert orbital.charge_transfer_dG_kj < 0
    print(f"  + Lead+NS: CT favorable, dG_CT={orbital.charge_transfer_dG_kj:.1f} kJ/mol")


# ── Kinetics tests ────────────────────────────────────────────────

def _make_lead_thermo_hydro():
    """Helper: compute thermo+hydro for lead in mesoporous silica."""
    problem = decompose("lead capture from mine water")
    enrich_target_electronic(problem.target)
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "O", "O", "N"],
        donor_type="borderline", structure="EDTA-like")
    from knowledge.structural_library import STRUCTURAL_OPTIONS
    meso = [s for s in STRUCTURAL_OPTIONS if s.type == "mesoporous_silica"][0]
    interior = InteriorDesign(sites=[InteriorSite(recognition=rec, copies=10)],
        design_level="composite", total_binding_sites=10, unique_recognition_types=1, avidity_factor=3.0)
    thermo = compute_thermodynamics(rec, meso, interior, problem)
    hydro = compute_hydrodynamics(meso, interior, problem)
    orbital = compute_orbital_binding(rec, problem.target)
    return thermo, hydro, orbital, meso, interior, problem


def test_k_on_physical_range():
    """k_on should be in physically meaningful range (1 to 10^9 M-1s-1)."""
    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()
    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)
    assert 1.0 <= kinetics.k_on_M_s <= 1e10, f"k_on = {kinetics.k_on_M_s:.1e} out of range"
    print(f"  + k_on = {kinetics.k_on_M_s:.1e} M-1s-1 (physical)")


def test_k_off_from_keq():
    """k_off = k_on / K_eq should give reasonable residence time."""
    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()
    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)
    assert kinetics.k_off_s > 0
    assert kinetics.residence_time_s > 0
    assert kinetics.residence_time_s < 1e10  # not infinite
    print(f"  + k_off = {kinetics.k_off_s:.1e} s-1, tau = {kinetics.residence_time_s:.1e} s")


def test_occupancy_reasonable():
    """Fractional occupancy should be between 0 and 1."""
    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()
    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)
    assert 0.0 <= kinetics.fractional_occupancy <= 1.0
    print(f"  + Occupancy: {kinetics.fractional_occupancy:.0%} at working [Pb2+]")


def test_rate_limiting_identified():
    """Rate-limiting step should be identified."""
    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()
    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)
    assert kinetics.rate_limiting_step != ""
    print(f"  + Rate-limiting: {kinetics.rate_limiting_step[:50]}")


# ── Probability chain tests ───────────────────────────────────────

def test_chain_all_probabilities_valid():
    """All probabilities in chain should be 0-1."""
    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()
    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)
    assembly = BinderAssembly(
        name="test_assembly", description="test", design_level="composite",
        interior=interior, structure=struct,
        selectivity=SelectivityFilter(name="none", mechanism="none", description="none"),
        release=ReleaseMechanism(name="pH_shift", trigger="pH_change", description="pH shift release"),
    )
    chain = compute_probability_chain(thermo, hydro, kinetics, orbital, assembly, problem)
    for name, val in [("enter", chain.p_enter), ("encounter", chain.p_encounter),
                      ("bind", chain.p_bind), ("retain", chain.p_retain),
                      ("release", chain.p_release), ("capture", chain.p_capture),
                      ("cycle", chain.p_cycle)]:
        assert 0.0 <= val <= 1.0, f"P({name}) = {val} out of range"
    print(f"  + All probabilities valid: P(cycle) = {chain.p_cycle:.4f}")


def test_chain_product_correct():
    """P(capture) should equal P(enter) * P(encounter) * P(bind)."""
    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()
    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)
    assembly = BinderAssembly(
        name="test_assembly", description="test", design_level="composite",
        interior=interior, structure=struct,
        selectivity=SelectivityFilter(name="none", mechanism="none", description="none"),
        release=ReleaseMechanism(name="pH_shift", trigger="pH_change", description="pH shift release"),
    )
    chain = compute_probability_chain(thermo, hydro, kinetics, orbital, assembly, problem)
    expected = chain.p_enter * chain.p_encounter * chain.p_bind
    assert abs(chain.p_capture - expected) < 0.001, f"P(capture)={chain.p_capture} != product={expected}"
    print(f"  + P(capture) = {chain.p_enter:.3f} x {chain.p_encounter:.3f} x {chain.p_bind:.3f} = {chain.p_capture:.4f}")


# ── End-to-end ────────────────────────────────────────────────────

def test_e2e_lead():
    """Full pipeline should produce physics + kinetics + probability data."""
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    assert len(r.assemblies) > 0
    top = r.assemblies[0]
    cr = top.confidence_reasoning
    assert "KINETICS:" in cr, "Missing kinetics"
    assert "PROBABILITY CHAIN:" in cr, "Missing probability chain"
    assert "ORBITAL:" in cr, "Missing orbital analysis"
    assert "k_on" in cr, "Missing k_on"
    assert "P(capture)" in cr, "Missing P(capture)"
    print(f"  + Lead E2E: full physics pipeline present")
    for a in r.assemblies[:3]:
        print(f"    {a.composite_score:.0%}  {a.name[:50]}")


def test_e2e_gold():
    """Gold should produce orbital analysis with DFT data."""
    o = Orchestrator(_build())
    r = o.solve(decompose("gold recovery from mine tailings"))
    assert len(r.assemblies) > 0
    cr = r.assemblies[0].confidence_reasoning
    assert "covalent" in cr.lower() or "mixed" in cr.lower(), "Gold should show covalent/mixed character"
    print(f"  + Gold E2E: orbital analysis confirms covalent/mixed character")
    print(f"    Top: {r.assemblies[0].name[:50]} ({r.assemblies[0].composite_score:.0%})")


def test_e2e_mercury():
    """Mercury should work through full pipeline."""
    o = Orchestrator(_build())
    r = o.solve(decompose("mercury removal from river water"))
    assert len(r.assemblies) > 0
    cr = r.assemblies[0].confidence_reasoning
    assert "kJ/mol" in cr
    assert "M" in cr  # rate constant units
    print(f"  + Mercury E2E: {r.assemblies[0].name[:50]} ({r.assemblies[0].composite_score:.0%})")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 10 - Kinetic Response Dynamics Tests")
    print("  " + "=" * 50)
    print()
    print("  Electronic data:")
    test_dft_data_exists()
    test_enrich_populates_target()
    test_gold_lower_lumo_than_calcium()
    print()
    print("  Orbital binding:")
    test_orbital_soft_match()
    test_orbital_hard_match()
    test_charge_transfer_favorable_for_good_match()
    print()
    print("  Kinetics:")
    test_k_on_physical_range()
    test_k_off_from_keq()
    test_occupancy_reasonable()
    test_rate_limiting_identified()
    print()
    print("  Probability chain:")
    test_chain_all_probabilities_valid()
    test_chain_product_correct()
    print()
    print("  End-to-end:")
    test_e2e_lead()
    test_e2e_gold()
    test_e2e_mercury()
    print()
    print("  All Sprint 10 tests passed.")
    print()
