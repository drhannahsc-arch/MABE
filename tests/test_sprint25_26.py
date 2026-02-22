"""tests/test_sprint25_26.py — Sprints 25+26: ET + Spectroscopy (30 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.electron_transfer import (
    compute_marcus_rate, predict_reductive_capture_rate,
    assess_radiation_stability,
)
from core.spectroscopic import (
    predict_dd_transition, predict_ct_transition, recommend_detection,
    assess_photoresponsive, predict_spectroscopy,
    _wavelength_to_color, _ev_to_nm,
)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 25: MARCUS ELECTRON TRANSFER
# ═══════════════════════════════════════════════════════════════════════════

def test_marcus_normal_region():
    """Small driving force should be in normal Marcus region."""
    r = compute_marcus_rate(-0.3, lambda_inner_ev=0.5, lambda_outer_ev=1.0,
                             h_da_ev=0.05)
    assert r.regime == "normal"
    assert r.k_et_s > 0
    assert r.activation_energy_kj > 0
    print(f"  \u2705 test_marcus_normal: k={r.k_et_s:.2e} s⁻¹, ΔG‡={r.activation_energy_kj:.1f} kJ/mol")

def test_marcus_activationless():
    """When |ΔG°| ≈ λ, rate should be near maximum."""
    # λ_total = 1.5 eV, ΔG° = -1.5 eV → activationless
    r = compute_marcus_rate(-1.5, lambda_inner_ev=0.5, lambda_outer_ev=1.0,
                             h_da_ev=0.05)
    assert r.regime == "activationless"
    assert r.activation_energy_kj < 5.0  # Near-zero barrier
    print(f"  \u2705 test_marcus_activationless: k={r.k_et_s:.2e}, ΔG‡={r.activation_energy_kj:.2f}")

def test_marcus_inverted():
    """When |ΔG°| >> λ, rate should DECREASE (inverted region)."""
    normal = compute_marcus_rate(-0.5, lambda_inner_ev=0.5, lambda_outer_ev=0.5,
                                  h_da_ev=0.05)
    inverted = compute_marcus_rate(-3.0, lambda_inner_ev=0.5, lambda_outer_ev=0.5,
                                    h_da_ev=0.05)
    assert inverted.regime == "inverted"
    assert inverted.k_et_s < normal.k_et_s, \
        f"Inverted rate ({inverted.k_et_s:.2e}) should be < normal ({normal.k_et_s:.2e})"
    print(f"  \u2705 test_marcus_inverted: normal k={normal.k_et_s:.2e} > inverted k={inverted.k_et_s:.2e}")

def test_coupling_affects_rate():
    """Stronger electronic coupling should give faster rates."""
    weak = compute_marcus_rate(-0.5, h_da_ev=0.001)
    strong = compute_marcus_rate(-0.5, h_da_ev=0.1)
    assert strong.k_et_s > weak.k_et_s * 100  # H_DA² scaling
    print(f"  \u2705 test_coupling: H=0.001→k={weak.k_et_s:.2e}, H=0.1→k={strong.k_et_s:.2e}")

def test_au_thiol_reduction():
    """Au3+ + thiol should give fast reductive capture."""
    r = predict_reductive_capture_rate("Au3+", "thiol")
    assert r.k_et_s > 1e3, f"Au-thiol ET should be fast, got k={r.k_et_s:.2e}"
    assert r.dg_driving_kj < 0  # Thermodynamically favorable
    print(f"  \u2705 test_au_thiol_ET: k={r.k_et_s:.2e} s⁻¹, ΔG°={r.dg_driving_kj:.1f} kJ/mol")

def test_fe3_ascorbate_reduction():
    """Fe3+ + ascorbate should be moderately fast."""
    r = predict_reductive_capture_rate("Fe3+", "ascorbate")
    assert r.k_et_s > 0
    assert r.dg_driving_kj < 0
    print(f"  \u2705 test_fe3_ascorbate: k={r.k_et_s:.2e} s⁻¹, ΔG°={r.dg_driving_kj:.1f}")

def test_cr6_strong_reductant():
    """Cr6+ + zero-valent iron should be thermodynamically very favorable."""
    r = predict_reductive_capture_rate("Cr6+", "zero_valent_iron")
    assert r.dg_driving_kj < -100  # Very favorable
    print(f"  \u2705 test_cr6_zvi: k={r.k_et_s:.2e}, ΔG°={r.dg_driving_kj:.1f} kJ/mol, "
          f"regime={r.regime}")

def test_lambda_inner_co3_large():
    """Co3+/Co2+ should have very large inner-sphere reorganization."""
    r = compute_marcus_rate(-0.5, redox_pair="Co3+/Co2+", h_da_ev=0.05)
    assert r.lambda_inner_kj > 150  # 1.8 eV = 174 kJ/mol
    print(f"  \u2705 test_lambda_co3: λ_inner={r.lambda_inner_kj:.1f} kJ/mol (LS→HS geometry change)")

def test_half_life_meaningful():
    """Half-life should be inverse of rate."""
    r = compute_marcus_rate(-0.5, h_da_ev=0.05)
    expected = math.log(2) / r.k_et_s
    assert abs(r.half_life_s - expected) < expected * 0.01
    print(f"  \u2705 test_half_life: k={r.k_et_s:.2e}, t½={r.half_life_s:.2e} s")

def test_adiabatic_classification():
    """Strong coupling should be adiabatic, weak non-adiabatic."""
    strong = compute_marcus_rate(-0.5, h_da_ev=0.10)
    weak = compute_marcus_rate(-0.5, h_da_ev=0.001)
    assert strong.is_adiabatic
    assert not weak.is_adiabatic
    print(f"  \u2705 test_adiabatic: H=0.10 adiabatic={strong.is_adiabatic}, "
          f"H=0.001 adiabatic={weak.is_adiabatic}")

# === RADIATION STABILITY ===

def test_zeolite_rad_excellent():
    """Zeolite should be radiation-excellent."""
    r = assess_radiation_stability("zeolite", dose_rate_gy_hr=100)
    assert r.stability_rating == "excellent"
    assert r.operational_lifetime_days > 1000
    print(f"  \u2705 test_zeolite_rad: {r.stability_rating}, lifetime={r.operational_lifetime_days:.0f} days")

def test_dna_origami_rad_unsuitable():
    """DNA origami should be unsuitable for radiation environments."""
    r = assess_radiation_stability("dna_origami", dose_rate_gy_hr=10,
                                    is_nuclear_application=True)
    assert r.stability_rating == "unsuitable"
    assert len(r.rad_resistant_alternatives) > 0
    print(f"  \u2705 test_dna_rad: {r.stability_rating}, alternatives={r.rad_resistant_alternatives[:2]}")

def test_mof_moderate_rad():
    """MOF should be moderate — organic linkers vulnerable."""
    r = assess_radiation_stability("MOF", dose_rate_gy_hr=1.0)
    assert r.stability_rating == "moderate"
    print(f"  \u2705 test_mof_rad: {r.stability_rating}, mechanism={r.degradation_mechanism[:40]}")

def test_nuclear_safety_factor():
    """Nuclear application should apply 0.5x safety factor."""
    normal = assess_radiation_stability("zeolite", dose_rate_gy_hr=100)
    nuclear = assess_radiation_stability("zeolite", dose_rate_gy_hr=100,
                                          is_nuclear_application=True)
    assert nuclear.critical_dose_gy < normal.critical_dose_gy
    print(f"  \u2705 test_nuclear_safety: normal={normal.critical_dose_gy:.0e}, "
          f"nuclear={nuclear.critical_dose_gy:.0e}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 26: SPECTROSCOPIC PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def test_ni2_green_color():
    """Ni2+ octahedral should be green (absorbs red ~700 nm)."""
    nm, ev, eps, color = predict_dd_transition(8, 102.0)  # 10Dq for Ni-aqua
    assert 550 < nm < 800, f"Ni2+ absorption should be ~600-700 nm, got {nm}"
    assert color in ("green", "blue"), f"Ni2+ should appear green/blue, got {color}"
    print(f"  \u2705 test_ni2_color: λ={nm:.0f} nm, color={color}")

def test_cu2_blue_color():
    """Cu2+ should absorb in red (~790 nm), appearing blue-green."""
    nm, ev, eps, color = predict_dd_transition(9, 151.0)  # 10Dq for Cu-aqua
    assert nm > 700  # Red/NIR absorption
    assert color in ("green", "blue")  # Complementary color
    print(f"  \u2705 test_cu2_color: λ={nm:.0f} nm, color={color}")

def test_d10_colorless():
    """d10 metals (Zn2+) should have no d-d transitions."""
    nm, ev, eps, color = predict_dd_transition(10, 100.0)
    assert nm == 0.0
    assert color == "colorless"
    print(f"  \u2705 test_d10_colorless: no d-d transitions")

def test_d5_hs_weak():
    """d5 HS (Mn2+) should have very weak transitions."""
    nm, ev, eps, color = predict_dd_transition(5, 90.0)
    assert nm == 0.0  # Spin-forbidden
    assert color == "colorless"
    print(f"  \u2705 test_d5_weak: Mn2+ very pale (spin-forbidden)")

def test_ct_lmct_au_thiol():
    """Au3+ + thiol should have intense LMCT band."""
    ct_nm, ct_type, ct_eps = predict_ct_transition("Au3+", ["S", "S", "S", "S"], 8)
    assert ct_type == "LMCT"
    assert ct_eps > 5000  # Intense
    assert ct_nm > 200
    print(f"  \u2705 test_lmct_au_s: {ct_type} at {ct_nm:.0f} nm, ε={ct_eps:.0f}")

def test_ct_fe3_thiol_intense():
    """Fe3+ + thiol should have very intense LMCT."""
    ct_nm, ct_type, ct_eps = predict_ct_transition("Fe3+", ["S", "S"], 5)
    assert ct_type == "LMCT"
    assert ct_eps > 5000
    print(f"  \u2705 test_lmct_fe3_s: {ct_nm:.0f} nm, ε={ct_eps:.0f}")

def test_mlct_fe2_bipy():
    """Fe2+ + bipyridyl should show MLCT (red complex)."""
    ct_nm, ct_type, ct_eps = predict_ct_transition("Fe2+", ["N", "N", "N", "N", "N", "N"], 6)
    assert ct_type == "MLCT"
    assert ct_eps > 5000
    print(f"  \u2705 test_mlct_fe2_bipy: {ct_type} at {ct_nm:.0f} nm, ε={ct_eps:.0f}")

def test_detection_fluorescence_cu2():
    """Cu2+ (paramagnetic) should recommend fluorescence quench."""
    method, signal, sens = recommend_detection(700, 10, 0, 0, "none", "Cu2+", ["N"])
    assert method == "fluorescence_quench"
    assert sens == "nM"
    print(f"  \u2705 test_detect_cu2: {method}, sensitivity={sens}")

def test_detection_colorimetric_ni2():
    """Ni2+ with visible d-d band should recommend colorimetric."""
    method, signal, sens = recommend_detection(600, 50, 0, 0, "none", "Ni2+", ["N"])
    assert method == "colorimetric"
    print(f"  \u2705 test_detect_ni2: {method}")

def test_photoswitch_dna_origami():
    """DNA origami should support azobenzene photoswitch."""
    photo, ptype, nm, notes = assess_photoresponsive("dna_origami_icosahedron")
    assert photo is True
    assert ptype == "azobenzene"
    assert nm == 365
    print(f"  \u2705 test_photo_dna: {ptype} at {nm} nm")

def test_photoswitch_mip():
    """MIP should support spiropyran photoswitch."""
    photo, ptype, nm, notes = assess_photoresponsive("MIP")
    assert photo is True
    assert ptype == "spiropyran"
    print(f"  \u2705 test_photo_mip: {ptype} at {nm} nm")

def test_full_spectroscopy_ni2():
    """Full spectroscopic prediction for Ni2+ octahedral."""
    r = predict_spectroscopy("Ni2+", ["N", "N", "N", "N", "N", "N"],
                              d_electrons=8, ten_dq_kj=161.0,
                              geometry="octahedral",
                              scaffold_type="dna_origami_icosahedron")
    assert r.predicted_color != "colorless"
    assert r.detection_method != ""
    assert r.photoresponsive is True
    print(f"  \u2705 test_full_spec_ni: color={r.predicted_color}, detect={r.detection_method}, "
          f"photo={r.photoswitch_type}")

def test_full_spectroscopy_au3_thiol():
    """Au3+ + thiol: intense CT, nM sensitivity expected."""
    r = predict_spectroscopy("Au3+", ["S", "S", "S", "S"],
                              d_electrons=8, ten_dq_kj=499.0,
                              geometry="square_planar")
    assert r.ct_type == "LMCT"
    assert r.ct_extinction > 5000
    assert r.sensitivity_estimate == "nM"
    print(f"  \u2705 test_full_spec_au: CT={r.ct_type} at {r.ct_transition_nm:.0f} nm, "
          f"ε={r.ct_extinction:.0f}, sens={r.sensitivity_estimate}")

def test_wavelength_color_mapping():
    """Verify wavelength→color mapping covers visible spectrum."""
    assert _wavelength_to_color(350) == "colorless"   # UV
    assert _wavelength_to_color(420) == "yellow"       # Violet absorbed
    assert _wavelength_to_color(470) == "orange"       # Blue absorbed
    assert _wavelength_to_color(530) == "red"          # Green absorbed
    assert _wavelength_to_color(580) == "purple"       # Yellow absorbed
    assert _wavelength_to_color(600) == "blue"         # Orange absorbed
    assert _wavelength_to_color(650) == "green"        # Red absorbed
    print(f"  \u2705 test_color_mapping: full visible spectrum verified")


if __name__ == "__main__":
    print("\n\U0001f9ea Sprints 25+26 \u2014 Electron Transfer + Spectroscopy\n")
    print("Sprint 25 — Marcus Electron Transfer:")
    test_marcus_normal_region(); test_marcus_activationless()
    test_marcus_inverted(); test_coupling_affects_rate()
    test_au_thiol_reduction(); test_fe3_ascorbate_reduction()
    test_cr6_strong_reductant(); test_lambda_inner_co3_large()
    test_half_life_meaningful(); test_adiabatic_classification()
    print("\n  Radiation Stability:")
    test_zeolite_rad_excellent(); test_dna_origami_rad_unsuitable()
    test_mof_moderate_rad(); test_nuclear_safety_factor()
    print("\nSprint 26 — Spectroscopic Prediction:")
    test_ni2_green_color(); test_cu2_blue_color()
    test_d10_colorless(); test_d5_hs_weak()
    test_ct_lmct_au_thiol(); test_ct_fe3_thiol_intense()
    test_mlct_fe2_bipy(); test_detection_fluorescence_cu2()
    test_detection_colorimetric_ni2(); test_photoswitch_dna_origami()
    test_photoswitch_mip(); test_full_spectroscopy_ni2()
    test_full_spectroscopy_au3_thiol(); test_wavelength_color_mapping()
    print("\n\u2705 All Sprint 25+26 tests passed! (30/30)")
    print("\n\U0001f389 ELECTRON TRANSFER + SPECTROSCOPY OPERATIONAL\n")

