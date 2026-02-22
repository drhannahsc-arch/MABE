"""
tests/test_sprint9.py - Physics-based scoring tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

import core.assembly_composer_patch
import core.scoring_patch
import core.physics_integration

from core.thermodynamics import (
    estimate_binding_energy, estimate_desolvation_penalty,
    estimate_chelate_effect, estimate_preorganization,
    compute_thermodynamics,
)
from core.hydrodynamics import compute_hydrodynamics
from core.assembly import RecognitionChemistry, InteriorDesign, InteriorSite
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


def test_soft_metal_prefers_soft_donors():
    soft_rec = RecognitionChemistry(name="thiol", type="chelator", donor_atoms=["S", "S"], donor_type="soft", structure="dithiol")
    hard_rec = RecognitionChemistry(name="carboxylate", type="chelator", donor_atoms=["O", "O"], donor_type="hard", structure="dicarboxylate")
    gold = TargetSpecies(identity="gold", formula="Au(3+)", charge=3.0, geometry="square_planar",
        electronic=ElectronicDescription(hardness_softness="soft", electronegativity=2.54),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.5, dehydration_energy_kj_mol=4690.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.85))
    dG_soft, _ = estimate_binding_energy(soft_rec, gold)
    dG_hard, _ = estimate_binding_energy(hard_rec, gold)
    assert dG_soft < dG_hard
    print(f"  + Gold: S donors = {dG_soft:.1f}, O donors = {dG_hard:.1f} kJ/mol (soft prefers soft)")


def test_desolvation_scales_with_charge():
    pb = TargetSpecies(identity="lead", formula="Pb(2+)", charge=2.0, geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline"),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.01, dehydration_energy_kj_mol=1481.0, coordination_number_water=9),
        size=SizeDescription(ionic_radius_angstrom=1.19))
    au = TargetSpecies(identity="gold", formula="Au(3+)", charge=3.0, geometry="square_planar",
        electronic=ElectronicDescription(hardness_softness="soft"),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.5, dehydration_energy_kj_mol=4690.0, coordination_number_water=6),
        size=SizeDescription(ionic_radius_angstrom=0.85))
    desolv_pb, _ = estimate_desolvation_penalty(pb, 2)
    desolv_au, _ = estimate_desolvation_penalty(au, 2)
    assert desolv_au > desolv_pb
    print(f"  + Desolvation: Pb2+ = +{desolv_pb:.0f}, Au3+ = +{desolv_au:.0f} kJ/mol")


def test_chelate_effect():
    dG_2, _ = estimate_chelate_effect(2)
    dG_4, _ = estimate_chelate_effect(4)
    dG_6, _ = estimate_chelate_effect(6)
    assert dG_4 < dG_2 < 0
    assert dG_6 < dG_4
    print(f"  + Chelate: 2d={dG_2:.1f}, 4d={dG_4:.1f}, 6d={dG_6:.1f} kJ/mol")


def test_macrocyclic():
    dG_open, _ = estimate_chelate_effect(4, False)
    dG_macro, _ = estimate_chelate_effect(4, True)
    assert dG_macro < dG_open
    print(f"  + Macrocyclic: open={dG_open:.1f}, macro={dG_macro:.1f} kJ/mol")


def test_preorg_mip_best():
    from knowledge.structural_library import STRUCTURAL_OPTIONS
    mip = [s for s in STRUCTURAL_OPTIONS if s.type == "mip"][0]
    none_s = [s for s in STRUCTURAL_OPTIONS if s.type == "none"][0]
    origami = [s for s in STRUCTURAL_OPTIONS if s.type == "dna_origami_cage"][0]
    interior = InteriorDesign(design_level="simple")
    dG_mip, _ = estimate_preorganization(mip, interior)
    dG_none, _ = estimate_preorganization(none_s, interior)
    dG_origami, _ = estimate_preorganization(origami, interior)
    assert dG_mip < dG_origami < dG_none
    print(f"  + Preorg: MIP={dG_mip:.1f}, origami={dG_origami:.1f}, free={dG_none:.1f} kJ/mol")


def test_net_dG_negative():
    problem = decompose("lead capture from mine water")
    rec = RecognitionChemistry(name="test", type="chelator", donor_atoms=["N", "O", "O", "N"],
        donor_type="borderline", structure="EDTA-like")
    from knowledge.structural_library import STRUCTURAL_OPTIONS
    meso = [s for s in STRUCTURAL_OPTIONS if s.type == "mesoporous_silica"][0]
    interior = InteriorDesign(sites=[InteriorSite(recognition=rec, copies=10)],
        design_level="composite", total_binding_sites=10, unique_recognition_types=1, avidity_factor=3.0)
    thermo = compute_thermodynamics(rec, meso, interior, problem)
    assert thermo.dG_net < 0
    print(f"  + Lead + NONO in silica: dG_net = {thermo.dG_net:.1f} kJ/mol")


def test_stokes_einstein():
    problem = decompose("lead capture from mine water")
    from knowledge.structural_library import STRUCTURAL_OPTIONS
    none_s = [s for s in STRUCTURAL_OPTIONS if s.type == "none"][0]
    hydro = compute_hydrodynamics(none_s, InteriorDesign(design_level="simple"), problem)
    assert 1e-10 < hydro.diffusion_coeff_m2s < 1e-8
    print(f"  + Pb2+ diffusion: D = {hydro.diffusion_coeff_m2s:.2e} m2/s")


def test_zeolite_excludes_lead():
    problem = decompose("lead capture from mine water")
    from knowledge.structural_library import STRUCTURAL_OPTIONS
    zsm5 = [s for s in STRUCTURAL_OPTIONS if "ZSM-5" in s.name][0]
    hydro = compute_hydrodynamics(zsm5, InteriorDesign(design_level="simple", total_binding_sites=2), problem)
    assert hydro.pore_restriction_factor < 0.01
    print(f"  + ZSM-5 vs Pb2+: restriction = {hydro.pore_restriction_factor:.4f} (excluded)")


def test_mesoporous_open():
    problem = decompose("lead capture from mine water")
    from knowledge.structural_library import STRUCTURAL_OPTIONS
    mcm = [s for s in STRUCTURAL_OPTIONS if "MCM-41" in s.name][0]
    hydro = compute_hydrodynamics(mcm, InteriorDesign(design_level="composite", total_binding_sites=50), problem)
    # MCM-41 2.5nm pores with Pb2+ 4.01A hydrated radius: lambda=0.32, real restriction
    # but NOT excluded like zeolite — restriction factor should be meaningful (>0.1)
    assert hydro.pore_restriction_factor > 0.1, f"Should pass through, got {hydro.pore_restriction_factor}"
    assert hydro.pore_restriction_factor < 0.99, "Should have some restriction"
    print(f"  + MCM-41 vs Pb2+: restriction = {hydro.pore_restriction_factor:.3f} (hindered but passable)")


def test_free_no_transport_limit():
    problem = decompose("lead capture from mine water")
    from knowledge.structural_library import STRUCTURAL_OPTIONS
    none_s = [s for s in STRUCTURAL_OPTIONS if s.type == "none"][0]
    hydro = compute_hydrodynamics(none_s, InteriorDesign(design_level="simple"), problem)
    assert hydro.transport_limitation_factor == 1.0
    print(f"  + Free: transport factor = 1.0 (correct)")


def test_e2e_physics():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture and release from mine water"))
    for a in r.assemblies[:3]:
        assert "PHYSICS" in a.confidence_reasoning
    print(f"  + All assemblies have physics scoring")
    for a in r.assemblies[:3]:
        print(f"    {a.composite_score:.0%}  {a.name[:50]}")


def test_gold_e2e():
    o = Orchestrator(_build())
    r = o.solve(decompose("gold recovery from mine tailings"))
    if r.assemblies:
        print(f"  + Gold top: {r.assemblies[0].name[:60]}")
        print(f"    Score: {r.assemblies[0].composite_score:.0%}")


def test_real_units():
    o = Orchestrator(_build())
    r = o.solve(decompose("lead capture from mine water"))
    if r.assemblies:
        assert "kJ/mol" in r.assemblies[0].confidence_reasoning
        print(f"  + Scores in real units (kJ/mol, m2/s)")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 9 - Energy Landscape + Hydrodynamics Tests")
    print("  " + "=" * 55)
    print()
    print("  Thermodynamics:")
    test_soft_metal_prefers_soft_donors()
    test_desolvation_scales_with_charge()
    test_chelate_effect()
    test_macrocyclic()
    test_preorg_mip_best()
    test_net_dG_negative()
    print()
    print("  Hydrodynamics:")
    test_stokes_einstein()
    test_zeolite_excludes_lead()
    test_mesoporous_open()
    test_free_no_transport_limit()
    print()
    print("  End-to-end:")
    test_e2e_physics()
    test_gold_e2e()
    test_real_units()
    print()
    print("  All Sprint 9 tests passed.")
    print()
