"""tests/test_sprint27_28.py — Sprints 27+28: Relativistic + Surface/Magnetic (25 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.relativistic import (
    get_relativistic_profile, correct_binding_energy,
    predict_geometry_from_lone_pair, compute_spin_orbit_splitting,
)
from core.surface_magnetic import (
    get_surface_profile, compute_magnetic_properties,
    recommend_magnetic_strategy,
)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 27: RELATIVISTIC EFFECTS
# ═══════════════════════════════════════════════════════════════════════════

def test_au_dominant_relativistic():
    """Au should show dominant relativistic effects (6s contraction)."""
    p = get_relativistic_profile("Au+")
    assert p.relativistic_significance == "dominant"
    assert p.s_contraction_pct > 10
    assert p.lewis_acidity_correction > 1.3
    assert "Gold anomaly" in p.notes
    print(f"  \u2705 test_au_relativistic: s_contract={p.s_contraction_pct}%, "
          f"lewis_corr={p.lewis_acidity_correction}, {p.relativistic_significance}")

def test_ni_negligible_relativistic():
    """Ni2+ (3d) should have negligible relativistic effects."""
    p = get_relativistic_profile("Ni2+")
    assert p.relativistic_significance == "negligible"
    assert p.lewis_acidity_correction == 1.0
    assert p.s_contraction_pct < 1
    print(f"  \u2705 test_ni_negligible: s_contract={p.s_contraction_pct}%, {p.relativistic_significance}")

def test_ag_moderate_relativistic():
    """Ag+ (4d) should have moderate relativistic effects."""
    p = get_relativistic_profile("Ag+")
    assert p.relativistic_significance == "moderate"
    assert p.s_contraction_pct > 3
    print(f"  \u2705 test_ag_moderate: s_contract={p.s_contraction_pct}%, {p.relativistic_significance}")

def test_au_stronger_than_ag():
    """Au Lewis acidity correction should exceed Ag (same group!)."""
    au = get_relativistic_profile("Au+")
    ag = get_relativistic_profile("Ag+")
    assert au.lewis_acidity_correction > ag.lewis_acidity_correction
    assert au.s_contraction_pct > ag.s_contraction_pct * 2
    print(f"  \u2705 test_au_vs_ag: Au correction={au.lewis_acidity_correction} > "
          f"Ag={ag.lewis_acidity_correction}")

def test_pb_inert_pair():
    """Pb2+ should have inert pair effect with hemidirected geometry."""
    p = get_relativistic_profile("Pb2+")
    assert p.inert_pair_stabilization_kj > 50
    assert p.lone_pair_stereochemistry == "hemidirected"
    print(f"  \u2705 test_pb_inert_pair: stabilization={p.inert_pair_stabilization_kj} kJ/mol, "
          f"lone_pair={p.lone_pair_stereochemistry}")

def test_binding_energy_correction():
    """Au binding energy should be enhanced by relativistic correction."""
    dg_orig = -100.0
    dg_corr, factor = correct_binding_energy(dg_orig, "Au+")
    assert abs(dg_corr) > abs(dg_orig)
    assert factor > 1.3
    print(f"  \u2705 test_binding_correction: Au+ {dg_orig}→{dg_corr} kJ/mol (×{factor})")

def test_no_correction_for_3d():
    """3d metals should get factor ≈ 1.0."""
    _, factor = correct_binding_energy(-100.0, "Ni2+")
    assert factor == 1.0
    print(f"  \u2705 test_no_correction_3d: Ni2+ factor={factor}")

def test_lone_pair_low_cn():
    """Pb2+ at low CN should be hemidirected."""
    geom, stab, note = predict_geometry_from_lone_pair("Pb2+", 3)
    assert geom == "hemidirected"
    assert stab > 50
    print(f"  \u2705 test_lone_pair_low_cn: Pb2+ CN=3 → {geom}, stab={stab}")

def test_lone_pair_high_cn():
    """Pb2+ at high CN should be holodirected."""
    geom, stab, note = predict_geometry_from_lone_pair("Pb2+", 8)
    assert geom == "holodirected"
    assert stab == 0.0
    print(f"  \u2705 test_lone_pair_high_cn: Pb2+ CN=8 → {geom}")

def test_spin_orbit_5d_large():
    """5d metal (Pt2+) should have significant SOC correction."""
    corr, note = compute_spin_orbit_splitting("Pt2+", 8, -200.0)
    assert abs(corr) > 1.0  # Non-trivial correction
    print(f"  \u2705 test_soc_5d: Pt2+ correction={corr:.2f} kJ/mol, {note}")

def test_spin_orbit_3d_small():
    """3d metal should have small SOC correction."""
    corr_3d, _ = compute_spin_orbit_splitting("Ni2+", 8, -200.0)
    corr_5d, _ = compute_spin_orbit_splitting("Pt2+", 8, -200.0)
    assert abs(corr_3d) < abs(corr_5d)
    print(f"  \u2705 test_soc_3d_vs_5d: Ni={corr_3d:.2f} vs Pt={corr_5d:.2f} kJ/mol")

def test_unknown_heavy_element():
    """Unknown heavy element should get estimated profile."""
    p = get_relativistic_profile("Fl2+")  # Flerovium (Z=114)
    assert p.s_contraction_pct > 0
    assert "Estimated" in p.notes
    print(f"  \u2705 test_unknown_heavy: s_contract={p.s_contraction_pct}%, {p.relativistic_significance}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 28: SURFACE ENERGY + MAGNETIC
# ═══════════════════════════════════════════════════════════════════════════

def test_zeolite_hydrophilic():
    """Zeolite should be superhydrophilic."""
    s = get_surface_profile("zeolite_Y")
    assert s.spontaneous_wetting
    assert s.contact_angle_water_deg < 30
    assert "hydrophilic" in s.wettability
    print(f"  \u2705 test_zeolite_surface: θ={s.contact_angle_water_deg}°, {s.wettability}")

def test_cnt_hydrophobic():
    """Carbon nanotubes should be hydrophobic."""
    s = get_surface_profile("carbon_nanotube")
    assert not s.spontaneous_wetting
    assert s.contact_angle_water_deg > 90
    assert s.wettability == "hydrophobic"
    print(f"  \u2705 test_cnt_hydrophobic: θ={s.contact_angle_water_deg}°, treatment: {s.surface_treatment[:40]}")

def test_dna_superhydrophilic():
    """DNA origami should be superhydrophilic."""
    s = get_surface_profile("dna_origami_icosahedron")
    assert s.wettability == "superhydrophilic"
    assert s.contact_angle_water_deg < 10
    print(f"  \u2705 test_dna_surface: θ={s.contact_angle_water_deg}°, {s.wettability}")

def test_capillary_pressure():
    """Mesoporous silica with small pores should have high capillary pressure."""
    s = get_surface_profile("mesoporous_silica_MCM41", pore_diameter_nm=3.5)
    assert s.capillary_pressure_kpa > 1000  # Nanopore = enormous capillary pressure
    print(f"  \u2705 test_capillary: MCM-41 P_cap={s.capillary_pressure_kpa:.0f} kPa ({s.capillary_pressure_kpa/1000:.0f} MPa)")

def test_mip_marginal_wetting():
    """MIP polymer should be marginally hydrophilic (θ near 90°)."""
    s = get_surface_profile("MIP")
    assert 60 < s.contact_angle_water_deg < 95
    assert s.spontaneous_wetting  # Just barely
    print(f"  \u2705 test_mip_wetting: θ={s.contact_angle_water_deg}°, wet={s.spontaneous_wetting}")

def test_paramagnetic_fe3():
    """Fe3+ d5 HS (5 unpaired) should be strongly paramagnetic."""
    m = compute_magnetic_properties(5, particle_diameter_um=5.0)
    assert m.paramagnetic
    assert m.complex_magnetic_moment_bm > 5.0  # √(5×7) = 5.92 BM
    print(f"  \u2705 test_paramagnetic_fe3: μ={m.complex_magnetic_moment_bm:.2f} BM, "
          f"v={m.separation_velocity_um_s:.4f} µm/s")

def test_diamagnetic_zn():
    """Zn2+ d10 (0 unpaired) should be diamagnetic."""
    m = compute_magnetic_properties(0)
    assert not m.paramagnetic
    assert m.complex_magnetic_moment_bm == 0.0
    assert not m.separation_feasible
    print(f"  \u2705 test_diamagnetic_zn: paramagnetic={m.paramagnetic}, bead={m.bead_recommendation[:30]}")

def test_magnetic_bead_recommendation():
    """Diamagnetic complex should recommend Fe3O4 bead."""
    rec = recommend_magnetic_strategy(0, "zeolite", "Zn2+")
    assert rec["strategy"] == "magnetic_bead_conjugation"
    assert "Fe₃O₄" in rec["bead_type"]
    print(f"  \u2705 test_bead_rec: {rec['strategy']}")

def test_paramagnetic_strategy():
    """Paramagnetic complex should get appropriate strategy."""
    rec = recommend_magnetic_strategy(5, "chelator", "Fe3+")
    assert "magnetic" in rec["strategy"]
    print(f"  \u2705 test_para_strategy: {rec['strategy']}")

def test_magnetic_moment_formula():
    """Spin-only magnetic moment should follow μ = √(n(n+2))."""
    m = compute_magnetic_properties(3)  # 3 unpaired
    expected = math.sqrt(3 * 5)  # √15 = 3.87
    assert abs(m.complex_magnetic_moment_bm - expected) < 0.01
    print(f"  \u2705 test_mu_formula: n=3, μ={m.complex_magnetic_moment_bm:.2f} BM "
          f"(expected {expected:.2f})")

if __name__ == "__main__":
    print("\n\U0001f9ea Sprints 27+28 \u2014 Relativistic + Surface/Magnetic\n")
    print("Sprint 27 — Relativistic Effects:")
    test_au_dominant_relativistic(); test_ni_negligible_relativistic()
    test_ag_moderate_relativistic(); test_au_stronger_than_ag()
    test_pb_inert_pair(); test_binding_energy_correction()
    test_no_correction_for_3d(); test_lone_pair_low_cn()
    test_lone_pair_high_cn(); test_spin_orbit_5d_large()
    test_spin_orbit_3d_small(); test_unknown_heavy_element()
    print("\nSprint 28 — Surface Energy + Magnetic:")
    test_zeolite_hydrophilic(); test_cnt_hydrophobic()
    test_dna_superhydrophilic(); test_capillary_pressure()
    test_mip_marginal_wetting(); test_paramagnetic_fe3()
    test_diamagnetic_zn(); test_magnetic_bead_recommendation()
    test_paramagnetic_strategy(); test_magnetic_moment_formula()
    print("\n\u2705 All Sprint 27+28 tests passed! (23/23)")
    print("\n\U0001f389 RELATIVISTIC + SURFACE/MAGNETIC OPERATIONAL\n")

