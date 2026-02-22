"""tests/test_sprint18.py — Sprint 18: Speciation Gate + Redox Routing (15 tests)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.speciation_gate import (
    predict_speciation, speciation_gated_design,
    _nernst_dg, _REDOX_PATHWAYS,
)

# === SPECIATION PREDICTION ===
def test_pb_free_ion_at_low_ph():
    """Pb2+ at pH 4 should be >90% free ion."""
    s = predict_speciation("Pb2+", 4.0)
    assert s.free_ion_fraction >= 0.90
    assert s.design_strategy == "free_ion_binding"
    assert s.effective_formula == "Pb2+"
    print(f"  \u2705 test_pb_free_ion_at_low_ph: {s.free_ion_fraction*100:.0f}% free ion, strategy={s.design_strategy}")

def test_pb_precipitate_at_high_ph():
    """Pb2+ at pH 10 should precipitate — no binder needed."""
    s = predict_speciation("Pb2+", 10.0)
    assert s.free_ion_fraction < 0.1
    assert s.design_strategy in ("precipitation_sufficient", "anion_capture")
    print(f"  \u2705 test_pb_precipitate_at_high_ph: strategy={s.design_strategy}, free_ion={s.free_ion_fraction*100:.0f}%")

def test_pb_mixed_speciation():
    """Pb2+ at pH 7 should show mixed speciation."""
    s = predict_speciation("Pb2+", 7.0)
    assert s.free_ion_fraction < 0.90
    assert s.bindable_fraction > 0.3
    assert len(s.species) > 1
    print(f"  \u2705 test_pb_mixed_speciation: {len(s.species)} species, free_ion={s.free_ion_fraction*100:.0f}%")

def test_fe3_precipitates_above_ph4():
    """Fe3+ at pH 6 should be precipitated rust."""
    s = predict_speciation("Fe3+", 6.0)
    assert s.free_ion_fraction < 0.05
    assert s.design_strategy == "precipitation_sufficient"
    print(f"  \u2705 test_fe3_precipitates: strategy={s.design_strategy}")

def test_fe3_free_ion_in_acid():
    """Fe3+ at pH 2 should be free ion."""
    s = predict_speciation("Fe3+", 2.0)
    assert s.free_ion_fraction >= 0.85
    assert s.design_strategy == "free_ion_binding"
    print(f"  \u2705 test_fe3_free_ion_in_acid: {s.free_ion_fraction*100:.0f}%")

def test_au3_chloro_complex():
    """Au3+ at pH 1 should be AuCl4- complex."""
    s = predict_speciation("Au3+", 1.0)
    assert any("AuCl4" in sp.formula for sp in s.species)
    print(f"  \u2705 test_au3_chloro: dominant={s.dominant_species}")

def test_uo2_carbonate_at_high_ph():
    """UO2 2+ at pH 8 should be uranyl carbonate anion."""
    s = predict_speciation("UO2_2+", 8.0)
    assert any("CO3" in sp.formula for sp in s.species)
    assert s.design_strategy in ("anion_capture", "mixed_species_binding")
    print(f"  \u2705 test_uo2_carbonate: dominant={s.dominant_species}, strategy={s.design_strategy}")

def test_hg2_thiol_bindable_at_all_ph():
    """Hg2+ species should be thiol-bindable even as Hg(OH)2."""
    for ph in [2.0, 5.0, 8.0]:
        s = predict_speciation("Hg2+", ph)
        assert s.bindable_fraction >= 0.5, f"Hg should be bindable at pH {ph}"
    print(f"  \u2705 test_hg2_thiol_bindable: bindable at all pH tested")

def test_na_always_free():
    """Na+ should be free ion at any pH."""
    s = predict_speciation("Na+", 12.0)
    assert s.free_ion_fraction >= 0.95
    assert s.design_strategy == "free_ion_binding"
    print(f"  \u2705 test_na_always_free: {s.free_ion_fraction*100:.0f}%")

def test_unknown_metal():
    """Unknown metal should return reasonable fallback."""
    s = predict_speciation("Tl+", 5.0)
    assert s.free_ion_fraction > 0
    assert "No speciation data" in s.design_notes
    print(f"  \u2705 test_unknown_metal: {s.design_notes[:60]}")

# === REDOX ROUTING ===
def test_redox_nernst_calculation():
    """Nernst equation should give reasonable dG values."""
    # Au3+ reduction at Eh=1600mV (strong reductant), pH 2: should be favorable
    dg_fav = _nernst_dg(1.50, 3, 1600, 2.0)
    assert dg_fav < 0, f"Au3+ reduction at high Eh should be favorable, got dG={dg_fav}"
    # At low Eh (800mV), Au3+ reduction is NOT favorable
    dg_unfav = _nernst_dg(1.50, 3, 800, 2.0)
    assert dg_unfav > 0, f"Au3+ reduction at low Eh should be unfavorable, got dG={dg_unfav}"
    print(f"  \u2705 test_redox_nernst: Eh=1600 dG={dg_fav:.1f}, Eh=800 dG={dg_unfav:.1f} kJ/mol")

def test_redox_au_pathway():
    """Au3+ with strongly reducing conditions should flag reductive deposition."""
    s = predict_speciation("Au3+", 2.0, redox_mv=1600)
    assert s.design_strategy == "redox_capture"
    assert "reduc" in s.design_notes.lower()
    print(f"  \u2705 test_redox_au: strategy={s.design_strategy}")

# === GATED DESIGN ===
def test_gated_design_normal():
    """Normal conditions should produce scored assemblies."""
    spec, assemblies = speciation_gated_design("copper", "Cu2+", working_ph=4.0)
    assert spec.design_strategy == "free_ion_binding"
    assert len(assemblies) > 0
    assert assemblies[0].thermodynamics is not None
    # Should have speciation annotation
    assert any("Speciation" in c for c in assemblies[0].confidence_reasoning)
    print(f"  \u2705 test_gated_design_normal: {len(assemblies)} assemblies, "
          f"top dG={assemblies[0].thermodynamics.dg_net_kj:.1f}")

def test_gated_design_precipitation():
    """Fe3+ at pH 8 should return empty — precipitation sufficient."""
    spec, assemblies = speciation_gated_design("iron3", "Fe3+", working_ph=8.0)
    assert spec.design_strategy == "precipitation_sufficient"
    assert len(assemblies) == 0
    print(f"  \u2705 test_gated_design_precipitation: no binder needed (correct)")

def test_gated_design_low_free_ion():
    """Pb2+ at pH 7.5 should annotate reduced free ion fraction."""
    spec, assemblies = speciation_gated_design("lead", "Pb2+", working_ph=7.5)
    assert spec.free_ion_fraction < 0.5
    if assemblies:
        assert any("free ion" in f.lower() for f in assemblies[0].failure_modes)
    print(f"  \u2705 test_gated_design_low_free_ion: free_ion={spec.free_ion_fraction*100:.0f}%, "
          f"assemblies={len(assemblies)}")

if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 18 \u2014 Speciation Gate + Redox Routing\n")
    print("Speciation Prediction:")
    test_pb_free_ion_at_low_ph(); test_pb_precipitate_at_high_ph()
    test_pb_mixed_speciation(); test_fe3_precipitates_above_ph4()
    test_fe3_free_ion_in_acid(); test_au3_chloro_complex()
    test_uo2_carbonate_at_high_ph(); test_hg2_thiol_bindable_at_all_ph()
    test_na_always_free(); test_unknown_metal()
    print("\nRedox Routing:")
    test_redox_nernst_calculation(); test_redox_au_pathway()
    print("\nGated Design:")
    test_gated_design_normal(); test_gated_design_precipitation()
    test_gated_design_low_free_ion()
    print("\n\u2705 All Sprint 18 tests passed! (15/15)")
    print("\n\U0001f389 SPECIATION GATE OPERATIONAL\n")

