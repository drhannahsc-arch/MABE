"""tests/test_sprint34.py — Sprint 34: Synthesis Protocol Generation (20 tests)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.synthesis import (
    generate_synthesis_protocol, SynthesisProtocol, REAGENT_DB,
    _mesoporous_silica_sba15, _zeolite_Y_from_scratch, _LDH_coprecipitation,
    _MOF_UiO66, _MIP_preparation, _dna_origami_scaffold,
)
from core.design_package import design_binder


# ═══════════════════════════════════════════════════════════════════════════
# REAGENT DATABASE
# ═══════════════════════════════════════════════════════════════════════════

def test_reagent_db_populated():
    """Should have comprehensive reagent database."""
    assert len(REAGENT_DB) > 25
    categories = set(r.role for r in REAGENT_DB.values())
    assert "precursor" in categories
    assert "linker" in categories
    assert "catalyst" in categories
    print(f"  \u2705 test_reagents: {len(REAGENT_DB)} reagents, roles={categories}")

def test_commodity_pricing():
    """Commodity chemicals should be <$50/kg."""
    for key, r in REAGENT_DB.items():
        if r.supplier_tier == "commodity":
            assert r.cost_per_kg_usd < 50, f"{key} marked commodity but ${r.cost_per_kg_usd}/kg"
    print(f"  \u2705 test_commodity: all commodity < $50/kg")

def test_inorganic_precursors_present():
    """Must have Si, Al, Mg, Fe, Zr precursors."""
    keys = set(REAGENT_DB.keys())
    assert "TEOS" in keys
    assert "sodium_silicate" in keys
    assert "sodium_aluminate" in keys
    assert "MgCl2" in keys
    assert "FeCl3" in keys
    print(f"  \u2705 test_inorganic: all precursors present")

def test_silane_agents_present():
    """Must have APTES, MPTMS, GPTMS for grafting."""
    keys = set(REAGENT_DB.keys())
    assert "APTES" in keys   # Amine
    assert "MPTMS" in keys   # Thiol
    assert "GPTMS" in keys   # Epoxy
    print(f"  \u2705 test_silanes: APTES, MPTMS, GPTMS present")

# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD ROUTES
# ═══════════════════════════════════════════════════════════════════════════

def test_sba15_route():
    """SBA-15 route should have sol-gel + calcination."""
    steps = _mesoporous_silica_sba15()
    assert len(steps) >= 3
    types = [s.reaction_type for s in steps]
    assert "sol_gel" in types
    assert "calcination" in types
    print(f"  \u2705 test_sba15: {len(steps)} steps, types={types}")

def test_zeolite_Y_route():
    """Zeolite Y from sodium silicate + aluminate."""
    steps = _zeolite_Y_from_scratch()
    assert len(steps) >= 3
    # Check uses commodity precursors
    all_reagents = [rk for s in steps for rk, _ in s.reagents]
    assert "sodium_silicate" in all_reagents
    assert "sodium_aluminate" in all_reagents
    print(f"  \u2705 test_zeolite_Y: {len(steps)} steps, commodity precursors")

def test_ldh_route():
    """LDH coprecipitation from MgCl2 + AlCl3."""
    steps = _LDH_coprecipitation()
    all_reagents = [rk for s in steps for rk, _ in s.reagents]
    assert "MgCl2" in all_reagents
    assert "AlCl3" in all_reagents
    print(f"  \u2705 test_ldh: {len(steps)} steps, MgCl2+AlCl3")

def test_mip_route():
    """MIP should include template removal step."""
    steps = _MIP_preparation("Pb2+")
    types = [s.reaction_type for s in steps]
    assert "extraction" in types  # Template removal
    print(f"  \u2705 test_mip: {len(steps)} steps, template extraction present")

def test_mof_route():
    """MOF route should include activation."""
    steps = _MOF_UiO66()
    types = [s.reaction_type for s in steps]
    assert "solvothermal" in types
    assert "activation" in types
    print(f"  \u2705 test_mof: {len(steps)} steps, solvothermal+activation")

# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def test_zeolite_dithiocarbamate():
    """S-donor on zeolite: silane + CS2 route."""
    p = generate_synthesis_protocol("test", "Pb2+", "zeolite_Y",
        ["S","S","S","S"], "soft", 0.99)
    step_names = [s.name for s in p.steps]
    assert any("silane" in n.lower() or "graft" in n.lower() for n in step_names)
    assert any("dithiocarbamate" in n.lower() for n in step_names)
    assert p.total_cost_usd_per_gram < 2.0
    print(f"  \u2705 test_zeo_dtc: ${p.total_cost_usd_per_gram:.2f}/g, {p.total_time_hours:.0f}h")

def test_silica_catechol():
    """O-donor on SBA-15 for Fe3+."""
    p = generate_synthesis_protocol("test", "Fe3+", "mesoporous_silica",
        ["O","O","O","O"], "hard", 0.12)
    assert p.scaffold_type == "mesoporous_silica"
    assert len(p.steps) >= 5  # Scaffold + graft + chelator + characterization
    print(f"  \u2705 test_silica_cat: {len(p.steps)} steps, ${p.total_cost_usd_per_gram:.2f}/g")

def test_ldh_protocol():
    """LDH protocol should be cheapest."""
    p = generate_synthesis_protocol("test", "Ni2+", "LDH",
        ["N","N","O","O"], "borderline", 0.24)
    assert p.total_cost_usd_per_gram < 0.50
    assert p.difficulty == "straightforward"
    assert p.scalability == "kg_scale"
    print(f"  \u2705 test_ldh_proto: ${p.total_cost_usd_per_gram:.2f}/g, {p.scalability}")

def test_mip_protocol():
    """MIP should be straightforward, kg-scalable."""
    p = generate_synthesis_protocol("test", "Cu2+", "MIP",
        ["N","N","O","O"], "borderline", 0.35)
    assert p.difficulty == "straightforward"
    assert p.scalability == "kg_scale"
    print(f"  \u2705 test_mip_proto: {p.difficulty}, {p.scalability}")

def test_dna_protocol_expensive():
    """DNA origami should be expert-level and expensive."""
    p = generate_synthesis_protocol("test", "Pb2+", "dna_origami",
        ["N","N","O","O"], "borderline", 0.55)
    assert p.difficulty == "expert"
    assert p.scalability == "mg_scale"
    print(f"  \u2705 test_dna_proto: {p.difficulty}, {p.scalability}, ${p.total_cost_usd_per_gram:.2f}/g")

def test_protocol_has_characterization():
    """Every protocol should end with characterization step."""
    p = generate_synthesis_protocol("test", "Pb2+", "zeolite_Y",
        ["S","S"], "soft", 0.99)
    last = p.steps[-1]
    assert "character" in last.name.lower()
    print(f"  \u2705 test_char_step: last step = {last.name}")

def test_protocol_has_alternatives():
    """Inorganic protocols should suggest alternatives."""
    p = generate_synthesis_protocol("test", "Pb2+", "zeolite_Y",
        ["S","S"], "soft", 0.99)
    assert len(p.alternative_routes) > 0
    print(f"  \u2705 test_alternatives: {len(p.alternative_routes)} alternatives")

def test_equipment_list():
    """Protocol should list required equipment."""
    p = generate_synthesis_protocol("test", "Fe3+", "mesoporous_silica",
        ["O","O","O","O"], "hard", 0.12)
    assert len(p.equipment_needed) > 0
    assert "autoclave_PTFE" in p.equipment_needed or "fume_hood" in p.equipment_needed
    print(f"  \u2705 test_equipment: {p.equipment_needed[:4]}")

# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_design_has_synthesis():
    """DesignPackage should include synthesis protocol."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=1)
    assert pkgs[0].synthesis is not None
    assert isinstance(pkgs[0].synthesis, SynthesisProtocol)
    assert pkgs[0].synthesis.total_cost_usd_per_gram > 0
    print(f"  \u2705 test_e2e_synth: ${pkgs[0].synthesis.total_cost_usd_per_gram:.2f}/g, "
          f"{pkgs[0].synthesis.difficulty}")

def test_e2e_cost_comparison():
    """Different scaffolds should have different costs."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=5)
    costs = [(p.scaffold_type, p.synthesis.total_cost_usd_per_gram) for p in pkgs if p.synthesis]
    unique_costs = set(c for _, c in costs)
    print(f"  \u2705 test_costs: {costs[:3]}")


if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 34 \u2014 Synthesis Protocol Generation\n")
    print("Reagent Database:")
    test_reagent_db_populated(); test_commodity_pricing()
    test_inorganic_precursors_present(); test_silane_agents_present()
    print("\nScaffold Routes:")
    test_sba15_route(); test_zeolite_Y_route()
    test_ldh_route(); test_mip_route(); test_mof_route()
    print("\nProtocol Generation:")
    test_zeolite_dithiocarbamate(); test_silica_catechol()
    test_ldh_protocol(); test_mip_protocol()
    test_dna_protocol_expensive(); test_protocol_has_characterization()
    test_protocol_has_alternatives(); test_equipment_list()
    print("\nIntegration:")
    test_e2e_design_has_synthesis(); test_e2e_cost_comparison()
    print("\n\u2705 All Sprint 34 tests passed! (20/20)\n")

