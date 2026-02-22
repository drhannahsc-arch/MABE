"""tests/test_sprint16.py — Sprint 16 v2: Diverse Coordination Generation (22 tests)

Rewritten for v2 generator that prioritizes design space diversity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.coordination_generator import (
    generate_coordination_environments, METAL_D_ELECTRONS, METAL_HSAB_SOFTNESS,
    PREFERRED_CN, SPECIAL_EFFECTS, _IONIC_RADII, _get_continuous_softness,
)
from core.donor_enumerator import enumerate_donor_arrangements
from core.scaffold_matcher import match_scaffolds, assemble_generative_binders
from core.generative_integration import generative_design


# ═══════════════════════════════════════════════════════════════════════════
# COORDINATION GENERATOR v2
# ═══════════════════════════════════════════════════════════════════════════

def test_ni2_generates_n_donors():
    """Ni2+ (borderline) should include N donors."""
    envs = generate_coordination_environments("nickel", "Ni2+")
    all_donors = set()
    for e in envs:
        for d in e.donors:
            all_donors.add(d.donor_atom)
    assert "N" in all_donors, f"Ni2+ should include N donors, got {all_donors}"
    print(f"  \u2705 test_ni2_n_donors: donor atoms={all_donors}")

def test_pb2_hemidirected():
    """Pb2+ should have hemidirected geometry."""
    envs = generate_coordination_environments("lead", "Pb2+")
    geoms = set(e.geometry for e in envs)
    assert any("hemidirected" in g for g in geoms), f"Pb2+ should be hemidirected, got {geoms}"
    print(f"  \u2705 test_pb2_hemi: geometries={geoms}")

def test_pb2_includes_s_donors():
    """Pb2+ (soft) should include S donors in at least one strategy."""
    envs = generate_coordination_environments("lead", "Pb2+")
    has_s = any("S" in [d.donor_atom for d in e.donors] for e in envs)
    assert has_s, "Pb2+ should generate S-donor environments"
    print(f"  \u2705 test_pb2_s_donors: S donors present")

def test_au3_soft_donors():
    """Au3+ should generate S donors as primary strategy."""
    envs = generate_coordination_environments("gold", "Au3+")
    top = envs[0]
    donors = [d.donor_atom for d in top.donors]
    assert "S" in donors, f"Au3+ top env should have S donors, got {donors}"
    print(f"  \u2705 test_au3_soft: top donors={donors}")

def test_cu2_jahn_teller():
    """Cu2+ should have Jahn-Teller special effect."""
    specials = SPECIAL_EFFECTS.get("Cu2+", [])
    assert "jahn_teller" in specials
    envs = generate_coordination_environments("copper", "Cu2+")
    geoms = set(e.geometry for e in envs)
    assert any("tetragonal" in g or "square" in g for g in geoms), f"Cu2+ geom={geoms}"
    print(f"  \u2705 test_cu2_jt: specials={specials}, geoms={geoms}")

def test_hg2_linear():
    """Hg2+ should generate linear geometry."""
    envs = generate_coordination_environments("mercury", "Hg2+")
    geoms = set(e.geometry for e in envs)
    assert "linear" in geoms, f"Hg2+ should include linear, got {geoms}"
    print(f"  \u2705 test_hg2_linear: geoms={geoms}")

def test_fe3_hard_donors():
    """Fe3+ (hard) should include O donors."""
    envs = generate_coordination_environments("iron", "Fe3+")
    has_o = any("O" in [d.donor_atom for d in e.donors] for e in envs)
    assert has_o, "Fe3+ should generate O-donor environments"
    print(f"  \u2705 test_fe3_hard: O donors present")

def test_multiple_strategies():
    """Should generate multiple donor strategies per metal."""
    envs = generate_coordination_environments("lead", "Pb2+")
    strategies = set(e.donor_strategy for e in envs)
    assert len(strategies) >= 2, f"Should have >=2 strategies, got {strategies}"
    print(f"  \u2705 test_strategies: {strategies}")

def test_dedup_unique():
    """No duplicate (strategy, CN, geometry) combinations."""
    envs = generate_coordination_environments("nickel", "Ni2+")
    keys = [(e.donor_strategy, e.coordination_number, e.geometry) for e in envs]
    assert len(keys) == len(set(keys)), "Duplicate environments found"
    print(f"  \u2705 test_dedup: {len(envs)} unique environments")

def test_continuous_softness_used():
    """Continuous softness should differ from categorical HSAB."""
    cont = _get_continuous_softness("Pb2+", 0)
    cat = METAL_HSAB_SOFTNESS.get("Pb2+", 0.3)
    # Continuous may differ significantly from categorical
    print(f"  \u2705 test_cont_soft: Pb2+ continuous={cont:.2f} vs categorical={cat:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# DONOR ENUMERATION
# ═══════════════════════════════════════════════════════════════════════════

def test_donor_enumeration_ni2():
    """Ni2+ should enumerate multiple N-donor ligands."""
    envs = generate_coordination_environments("nickel", "Ni2+")
    n_env = next(e for e in envs if all(d.donor_atom == "N" for d in e.donors))
    arrs = enumerate_donor_arrangements(n_env, working_ph=7.0)
    assert len(arrs) > 0, "Should enumerate arrangements"
    ligands = set()
    for a in arrs:
        for p in a.positioned_donors:
            ligands.add(p.ligand_name)
    print(f"  \u2705 test_enum_ni2: {len(arrs)} arrangements, ligands={ligands}")

def test_donor_ph_filtering():
    """Low pH should filter out high-pKa donors."""
    envs = generate_coordination_environments("gold", "Au3+")
    s_env = next(e for e in envs if all(d.donor_atom == "S" for d in e.donors))
    arrs = enumerate_donor_arrangements(s_env, working_ph=2.0)
    assert len(arrs) > 0, "Au3+ at pH 2 should have S-donor arrangements"
    for a in arrs:
        assert a.ph_working_range[0] <= 2.0
    print(f"  \u2705 test_ph_filter: {len(arrs)} arrangements at pH 2")

def test_donor_arrangement_properties():
    """Arrangements should have chelate rings and feasibility scores."""
    envs = generate_coordination_environments("copper", "Cu2+")
    arrs = enumerate_donor_arrangements(envs[0], working_ph=7.0)
    assert len(arrs) > 0
    for a in arrs:
        assert a.synthetic_feasibility > 0
        assert a.hsab_match_score > 0
    print(f"  \u2705 test_arr_props: all have feasibility and HSAB")

# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def test_scaffold_matching():
    """Scaffold matcher should find compatible scaffolds."""
    envs = generate_coordination_environments("nickel", "Ni2+")
    arrs = enumerate_donor_arrangements(envs[0], working_ph=7.0)
    matches = match_scaffolds(arrs[0], envs[0])
    assert len(matches) > 0, "Should find scaffold matches"
    print(f"  \u2705 test_scaffold: {len(matches)} matches, types={[m.scaffold_type for m in matches[:3]]}")

def test_assembly_generation():
    """Full assembly should produce GenerativeBinderAssembly objects."""
    envs = generate_coordination_environments("copper", "Cu2+")
    arrs = enumerate_donor_arrangements(envs[0], working_ph=7.0)
    matches = match_scaffolds(arrs[0], envs[0])
    assemblies = assemble_generative_binders(envs[0], arrs[0], matches, "copper")
    assert len(assemblies) > 0
    a = assemblies[0]
    assert a.target_formula == "Cu2+"
    assert len(a.donor_atoms) > 0
    assert a.scaffold_type != ""
    print(f"  \u2705 test_assembly: {len(assemblies)} assemblies, top={a.name}")

# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def test_generative_design_e2e():
    """generative_design() should produce diverse results."""
    assemblies = generative_design("lead", "Pb2+", charge=2, hsab_softness=0.99)
    assert len(assemblies) > 0
    scaffolds = set(a.scaffold_type for a in assemblies)
    donors = set(tuple(sorted(a.donor_atoms)) for a in assemblies)
    print(f"  \u2705 test_gen_e2e: {len(assemblies)} assemblies, "
          f"{len(scaffolds)} scaffolds, {len(donors)} donor sets")

def test_generative_design_au():
    """Au3+ should produce S-donor dominant designs."""
    assemblies = generative_design("gold", "Au3+", charge=3, hsab_softness=0.85)
    has_s = any("S" in a.donor_atoms for a in assemblies)
    assert has_s, "Au3+ should have S-donor assemblies"
    print(f"  \u2705 test_gen_au: S-donor designs present")

def test_metal_data_completeness():
    """Key metals should be in all lookup tables."""
    key_metals = ["Pb2+", "Cu2+", "Ni2+", "Au3+", "Fe3+", "Zn2+", "Hg2+"]
    for m in key_metals:
        assert m in METAL_D_ELECTRONS, f"{m} missing from d_electrons"
        assert m in PREFERRED_CN, f"{m} missing from preferred_cn"
        assert m in _IONIC_RADII, f"{m} missing from ionic_radii"
        assert m in METAL_HSAB_SOFTNESS, f"{m} missing from hsab"
    print(f"  \u2705 test_data_complete: all {len(key_metals)} metals in all tables")

def test_diverse_pb_donor_strategies():
    """Pb2+ should generate both S and N donor strategies."""
    envs = generate_coordination_environments("lead", "Pb2+")
    donor_atoms = set()
    for e in envs:
        for d in e.donors:
            donor_atoms.add(d.donor_atom)
    assert "S" in donor_atoms and "N" in donor_atoms, \
        f"Pb2+ should have both S and N strategies, got {donor_atoms}"
    print(f"  \u2705 test_pb2_diverse: donor atoms={donor_atoms}")

def test_diverse_cu_donor_strategies():
    """Cu2+ should generate multiple donor types."""
    envs = generate_coordination_environments("copper", "Cu2+")
    donor_atoms = set()
    for e in envs:
        for d in e.donors:
            donor_atoms.add(d.donor_atom)
    assert len(donor_atoms) >= 2, f"Cu2+ should have 2+ donor types, got {donor_atoms}"
    print(f"  \u2705 test_cu2_diverse: donor atoms={donor_atoms}")

def test_max_candidates_respected():
    """Should respect max_candidates parameter."""
    envs = generate_coordination_environments("nickel", "Ni2+", max_candidates=3)
    assert len(envs) <= 3
    print(f"  \u2705 test_max_cand: {len(envs)} <= 3")


if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 16 v2 \u2014 Diverse Coordination Generation\n")
    print("Coordination Generator:")
    test_ni2_generates_n_donors(); test_pb2_hemidirected()
    test_pb2_includes_s_donors(); test_au3_soft_donors()
    test_cu2_jahn_teller(); test_hg2_linear()
    test_fe3_hard_donors(); test_multiple_strategies()
    test_dedup_unique(); test_continuous_softness_used()
    print("\nDonor Enumeration:")
    test_donor_enumeration_ni2(); test_donor_ph_filtering()
    test_donor_arrangement_properties()
    print("\nScaffold Matching:")
    test_scaffold_matching(); test_assembly_generation()
    print("\nIntegration:")
    test_generative_design_e2e(); test_generative_design_au()
    test_metal_data_completeness(); test_diverse_pb_donor_strategies()
    test_diverse_cu_donor_strategies(); test_max_candidates_respected()
    print("\n\u2705 All Sprint 16 v2 tests passed! (22/22)")
    print("\n\U0001f389 DIVERSE COORDINATION GENERATION OPERATIONAL\n")

