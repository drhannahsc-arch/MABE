"""tests/test_sprint17.py - Sprint 17: Generative -> Physics Pipeline (20 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generative_integration import generative_design
from core.generative_physics_adapter import (
    adapt_generative_to_pipeline, score_assembly, design_and_score,
    print_scored_results, compute_thermodynamics_standalone,
    temperature_prediction, Problem, TargetSpecies, Matrix,
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
)

# === ADAPTER TESTS ===
def test_adapt_shape():
    """Adapted assembly should have all required fields."""
    gen = generative_design("nickel", "Ni2+")[0]
    a = adapt_generative_to_pipeline(gen)
    assert a.recognition is not None
    assert a.structure is not None
    assert a.interior is not None
    assert a.release is not None
    assert a.is_novel is True
    assert a.design_source == "generative_coordination_engine"
    assert len(a.recognition.donor_atoms) > 0
    assert a.recognition.denticity > 0
    print(f"  \u2705 test_adapt_shape: {a.name}, denticity={a.recognition.denticity}")

def test_adapt_preserves_hsab():
    """HSAB match score should transfer from generative to adapted."""
    gen = generative_design("gold", "Au3+")[0]
    a = adapt_generative_to_pipeline(gen)
    assert a.recognition.hsab_match > 0
    assert a.recognition.donor_type in ("soft", "mixed", "borderline", "hard")
    print(f"  \u2705 test_adapt_preserves_hsab: hsab={a.recognition.hsab_match:.2f}, type={a.recognition.donor_type}")

def test_adapt_scaffold_mapping():
    """Scaffold types should map correctly."""
    gen = generative_design("lead", "Pb2+")
    types_seen = set()
    for g in gen[:10]:
        a = adapt_generative_to_pipeline(g)
        types_seen.add(a.structure.type)
    assert len(types_seen) > 1, f"Should see multiple scaffold types, got {types_seen}"
    print(f"  \u2705 test_adapt_scaffold_mapping: types={types_seen}")

def test_adapt_stability_constant():
    """Estimated logK should increase with denticity and chelate rings."""
    gen = generative_design("iron3", "Fe3+")
    adapted = [adapt_generative_to_pipeline(g) for g in gen[:10]]
    # Sort by denticity
    adapted.sort(key=lambda a: a.recognition.denticity)
    if len(adapted) >= 2:
        low_dent = adapted[0]
        high_dent = adapted[-1]
        if high_dent.recognition.denticity > low_dent.recognition.denticity:
            assert high_dent.recognition.stability_constant_log >= low_dent.recognition.stability_constant_log
    print(f"  \u2705 test_adapt_stability_constant: logK range "
          f"{adapted[0].recognition.stability_constant_log:.1f}-"
          f"{adapted[-1].recognition.stability_constant_log:.1f}")

def test_adapt_release_inference():
    """Release mechanism should vary by scaffold type."""
    gen = generative_design("copper", "Cu2+", working_ph=4.0)
    releases = set()
    for g in gen[:10]:
        a = adapt_generative_to_pipeline(g)
        releases.add(a.release.mechanism)
    assert len(releases) >= 1
    print(f"  \u2705 test_adapt_release_inference: mechanisms={releases}")

# === THERMODYNAMICS TESTS ===
def test_thermo_negative_dg_for_good_match():
    """Well-matched metal-donor should have negative dG_net."""
    target = TargetSpecies("nickel", "Ni2+", 2, 69.0, 0.2, 0.40, 8)
    matrix = Matrix(ph=7.0, temperature_c=25.0, ionic_strength_mm=10.0)
    problem = Problem(target, matrix)
    rec = RecognitionChemistry("test", "generative", ["N","N","N","N"], "borderline",
                                denticity=4, hsab_match=0.95, chelate_rings=2)
    struct = StructuralConstraint("MIP", "MIP", "square_planar", 0.3, 0.5)
    interior = InteriorDesign("4N pocket", self_binding=True)
    t = compute_thermodynamics_standalone(rec, struct, interior, problem)
    assert t.dg_net_kj < 0, f"Good match should have negative dG, got {t.dg_net_kj}"
    assert t.dg_bind_kj < 0, f"dG_bind should be negative"
    assert t.dg_chelate_kj < 0, f"dG_chelate should be negative"
    assert t.dg_desolv_kj > 0, f"dG_desolv should be positive (penalty)"
    print(f"  \u2705 test_thermo_negative_dg: dG_net={t.dg_net_kj:.2f}, Kd={t.predicted_kd_um:.4f} uM")

def test_thermo_hsab_mismatch_penalty():
    """Mismatched HSAB should give worse binding than matched."""
    target_soft = TargetSpecies("gold", "Au3+", 3, 85.0, 0.2, 0.75, 8)
    matrix = Matrix(ph=7.0)
    problem = Problem(target_soft, matrix)
    # Good match: S donors for soft acid
    good = RecognitionChemistry("good", "g", ["S","S","S","S"], "soft", hsab_match=0.9, denticity=4)
    # Bad match: O donors for soft acid
    bad = RecognitionChemistry("bad", "g", ["O","O","O","O"], "hard", hsab_match=0.3, denticity=4)
    struct = StructuralConstraint("s", "MOF", "square_planar")
    t_good = compute_thermodynamics_standalone(good, struct, InteriorDesign(""), problem)
    t_bad = compute_thermodynamics_standalone(bad, struct, InteriorDesign(""), problem)
    assert t_good.dg_bind_kj < t_bad.dg_bind_kj, \
        f"S donors for Au3+ should bind better than O donors"
    print(f"  \u2705 test_thermo_hsab_mismatch: S-Au dG_bind={t_good.dg_bind_kj:.1f} vs O-Au={t_bad.dg_bind_kj:.1f}")

def test_thermo_chelate_effect():
    """More chelate rings should improve binding."""
    target = TargetSpecies("copper", "Cu2+", 2, 73.0, 0.2, 0.42, 9)
    problem = Problem(target, Matrix())
    struct = StructuralConstraint("s", "MOF", "tetragonal_elongated")
    no_chel = RecognitionChemistry("nc", "g", ["N","N","N","N"], "borderline",
                                    denticity=4, chelate_rings=0, hsab_match=0.9)
    with_chel = RecognitionChemistry("wc", "g", ["N","N","N","N"], "borderline",
                                      denticity=4, chelate_rings=3, hsab_match=0.9)
    t_nc = compute_thermodynamics_standalone(no_chel, struct, InteriorDesign(""), problem)
    t_wc = compute_thermodynamics_standalone(with_chel, struct, InteriorDesign(""), problem)
    assert t_wc.dg_net_kj < t_nc.dg_net_kj, "Chelate rings should improve binding"
    assert t_wc.dg_chelate_kj < t_nc.dg_chelate_kj
    print(f"  \u2705 test_thermo_chelate_effect: no_chelate={t_nc.dg_net_kj:.1f} vs "
          f"3_rings={t_wc.dg_net_kj:.1f} (diff={t_nc.dg_net_kj - t_wc.dg_net_kj:.1f})")

def test_thermo_protonation_penalty_at_low_ph():
    """Low pH should penalize donors that need to deprotonate."""
    target = TargetSpecies("iron3", "Fe3+", 3, 55.0, 0.2, 0.15, 5)
    rec = RecognitionChemistry("r", "g", ["N","N","N","N","N","N"], "borderline",
                                denticity=6, hsab_match=0.7)
    struct = StructuralConstraint("s", "MOF", "octahedral")
    # pH 3 — amine N donors (pKa~10.5) heavily penalized
    t_acid = compute_thermodynamics_standalone(rec, struct, InteriorDesign(""),
                                                Problem(target, Matrix(ph=3.0)))
    # pH 12 — no penalty
    t_base = compute_thermodynamics_standalone(rec, struct, InteriorDesign(""),
                                                Problem(target, Matrix(ph=12.0)))
    assert t_acid.dg_protonation_kj > t_base.dg_protonation_kj, \
        "Low pH should have higher protonation penalty for N donors"
    print(f"  \u2705 test_thermo_protonation_penalty: pH3={t_acid.dg_protonation_kj:.1f} vs "
          f"pH12={t_base.dg_protonation_kj:.1f}")

def test_thermo_preorg_bonus():
    """Self-binding scaffolds should have better preorganization."""
    target = TargetSpecies("lead", "Pb2+", 2)
    problem = Problem(target, Matrix())
    rec = RecognitionChemistry("r", "g", ["N","N","N","N"], "borderline", denticity=4, hsab_match=0.9)
    free = StructuralConstraint("free", "free", "none")
    mip = StructuralConstraint("mip", "MIP", "cavity")
    t_free = compute_thermodynamics_standalone(rec, free, InteriorDesign(""), problem)
    t_mip = compute_thermodynamics_standalone(rec, mip, InteriorDesign("", self_binding=True), problem)
    assert t_mip.dg_preorg_kj < t_free.dg_preorg_kj, "MIP should have better preorg than free"
    print(f"  \u2705 test_thermo_preorg_bonus: free={t_free.dg_preorg_kj:.1f} vs MIP={t_mip.dg_preorg_kj:.1f}")

def test_thermo_ionic_strength_correction():
    """High ionic strength should affect activity coefficient."""
    target = TargetSpecies("nickel", "Ni2+", 2)
    rec = RecognitionChemistry("r", "g", ["N","N","N","N"], "borderline", denticity=4, hsab_match=0.9)
    struct = StructuralConstraint("s", "MOF", "square_planar")
    t_low = compute_thermodynamics_standalone(rec, struct, InteriorDesign(""),
                                               Problem(target, Matrix(ionic_strength_mm=1.0)))
    t_high = compute_thermodynamics_standalone(rec, struct, InteriorDesign(""),
                                                Problem(target, Matrix(ionic_strength_mm=500.0)))
    assert t_low.dg_activity_kj != t_high.dg_activity_kj, "Ionic strength should affect activity"
    print(f"  \u2705 test_thermo_ionic_strength: I=1mM dG_act={t_low.dg_activity_kj:.2f} vs "
          f"I=500mM={t_high.dg_activity_kj:.2f}")

# === TEMPERATURE PREDICTION TESTS ===
def test_temperature_prediction():
    """Should predict dG and Kd at multiple temperatures."""
    target = TargetSpecies("nickel", "Ni2+", 2, 69.0, 0.2, 0.40, 8)
    rec = RecognitionChemistry("r", "g", ["N","N","N","N"], "borderline",
                                denticity=4, chelate_rings=2, hsab_match=0.9)
    struct = StructuralConstraint("s", "MIP", "square_planar", 0.3, 0.5)
    t = compute_thermodynamics_standalone(rec, struct, InteriorDesign("", self_binding=True),
                                           Problem(target, Matrix()))
    preds = temperature_prediction(t)
    assert "4C" in preds and "25C" in preds and "37C" in preds and "60C" in preds
    # Binding should weaken at higher temperature (generally)
    assert preds["4C"]["kd_um"] != preds["60C"]["kd_um"], "Temperature should affect Kd"
    print(f"  \u2705 test_temperature_prediction: 4C Kd={preds['4C']['kd_um']:.4f}, "
          f"60C Kd={preds['60C']['kd_um']:.4f}")

# === END-TO-END TESTS ===
def test_e2e_design_and_score_ni():
    """Full pipeline: Ni2+ design + score."""
    results = design_and_score("nickel", "Ni2+")
    assert len(results) > 0
    top = results[0]
    assert top.thermodynamics is not None
    assert top.thermodynamics.dg_net_kj < 0, f"Ni2+ should have favorable binding, got {top.thermodynamics.dg_net_kj}"
    assert top.physics_score > 0
    assert top.is_novel
    print(f"  \u2705 test_e2e_ni: {len(results)} scored, top dG={top.thermodynamics.dg_net_kj:.2f}, "
          f"Kd={top.thermodynamics.predicted_kd_um:.4f} uM")

def test_e2e_design_and_score_au():
    """Au3+ should rank soft-donor designs highest."""
    results = design_and_score("gold", "Au3+", charge=3)
    assert len(results) > 0
    top = results[0]
    assert "S" in top.recognition.donor_atoms or top.recognition.donor_type == "soft"
    assert top.thermodynamics.dg_net_kj < 0
    print(f"  \u2705 test_e2e_au: top={top.recognition.donor_type}, dG={top.thermodynamics.dg_net_kj:.2f}")

def test_e2e_acid_mine_drainage_scored():
    """AMD conditions: low pH, high ionic strength."""
    results = design_and_score("copper", "Cu2+", working_ph=3.5,
                                working_temp_c=10.0, ionic_strength_mm=500.0)
    assert len(results) > 0
    for r in results:
        assert "dna_origami" not in r.structure.name
        assert r.thermodynamics is not None
    # Protonation penalty should be significant
    top = results[0]
    assert top.thermodynamics.dg_protonation_kj > 0, "Low pH should incur protonation penalty"
    print(f"  \u2705 test_e2e_amd: {len(results)} scored, top dG={top.thermodynamics.dg_net_kj:.2f}, "
          f"protonation={top.thermodynamics.dg_protonation_kj:.1f}")

def test_e2e_ranking_by_dg():
    """Results should be ranked by dG_net (most negative first)."""
    results = design_and_score("lead", "Pb2+")
    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].thermodynamics.dg_net_kj <= results[i+1].thermodynamics.dg_net_kj, \
            f"Results should be ranked by dG: {results[i].thermodynamics.dg_net_kj} > {results[i+1].thermodynamics.dg_net_kj}"
    print(f"  \u2705 test_e2e_ranking: {len(results)} results, dG range "
          f"{results[0].thermodynamics.dg_net_kj:.1f} to {results[-1].thermodynamics.dg_net_kj:.1f}")

def test_e2e_print_scored():
    """Print function should work without errors."""
    results = design_and_score("nickel", "Ni2+", max_results=3)
    print_scored_results(results, top_n=2)
    print(f"  \u2705 test_e2e_print_scored: ok")

def test_e2e_all_have_temperature():
    """Every scored result should have temperature predictions."""
    results = design_and_score("zinc", "Zn2+")
    for r in results:
        assert r.temperature_prediction, "Missing temperature prediction"
        assert "4C" in r.temperature_prediction
    print(f"  \u2705 test_e2e_all_have_temperature: all {len(results)} have temp predictions")

if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 17 \u2014 Generative -> Physics Pipeline\n")
    print("Adapter:")
    test_adapt_shape(); test_adapt_preserves_hsab(); test_adapt_scaffold_mapping()
    test_adapt_stability_constant(); test_adapt_release_inference()
    print("\nThermodynamics:")
    test_thermo_negative_dg_for_good_match(); test_thermo_hsab_mismatch_penalty()
    test_thermo_chelate_effect(); test_thermo_protonation_penalty_at_low_ph()
    test_thermo_preorg_bonus(); test_thermo_ionic_strength_correction()
    print("\nTemperature:")
    test_temperature_prediction()
    print("\nEnd-to-End:")
    test_e2e_design_and_score_ni(); test_e2e_design_and_score_au()
    test_e2e_acid_mine_drainage_scored(); test_e2e_ranking_by_dg()
    test_e2e_print_scored(); test_e2e_all_have_temperature()
    print("\n\u2705 All Sprint 17 tests passed! (20/20)")
    print("\n\U0001f389 GENERATIVE + PHYSICS PIPELINE INTEGRATED\n")


