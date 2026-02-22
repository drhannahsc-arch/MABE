"""tests/test_sprint23_24.py — Sprints 23+24: System-Level Prediction (25 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cooperativity import (
    compute_site_repulsion, compute_hill_cooperativity,
    compute_loading_curve, compute_capacity, analyze_cooperativity,
)
from core.mass_transport import (
    stokes_einstein, hindered_diffusion, compute_thiele_modulus,
    analyze_transport, predict_capture_time,
)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 23: COOPERATIVITY
# ═══════════════════════════════════════════════════════════════════════════

def test_repulsion_close_sites():
    """Close sites (0.3 nm) with divalent ions should have large repulsion."""
    dg = compute_site_repulsion(2, 24, 0.30)
    assert dg > 5, f"Close divalent sites should repel strongly, got {dg}"
    print(f"  \u2705 test_repulsion_close: LDH 0.3nm spacing, dG={dg:.1f} kJ/mol")

def test_repulsion_far_sites():
    """Far sites (6 nm, DNA origami) should have negligible repulsion."""
    dg = compute_site_repulsion(2, 12, 6.0)
    assert dg < 2, f"6nm spacing should have minimal repulsion, got {dg}"
    print(f"  \u2705 test_repulsion_far: DNA origami 6nm, dG={dg:.2f} kJ/mol")

def test_repulsion_monovalent_less():
    """Monovalent ions should repel less than divalent at same spacing."""
    dg1 = compute_site_repulsion(1, 12, 1.0)
    dg2 = compute_site_repulsion(2, 12, 1.0)
    assert dg1 < dg2
    assert dg2 / dg1 > 3  # z² scaling: 4/1
    print(f"  \u2705 test_repulsion_charge: z=1 dG={dg1:.2f} vs z=2 dG={dg2:.2f} (ratio={dg2/dg1:.1f}x)")

def test_hill_negative_close_sites():
    """Close, highly-charged sites should give negative cooperativity."""
    hill, ctype = compute_hill_cooperativity(48, 0.30, 2)
    assert hill < 1.0, f"Close sites should be negative, got n_Hill={hill}"
    assert "negative" in ctype
    print(f"  \u2705 test_hill_negative: n_Hill={hill:.2f}, type={ctype}")

def test_hill_independent_mip():
    """MIP single-site should be independent (n_Hill=1)."""
    hill, ctype = compute_hill_cooperativity(1, 0.0, 2)
    assert hill == 1.0
    assert ctype == "independent"
    print(f"  \u2705 test_hill_mip: n_Hill={hill}, type={ctype}")

def test_hill_positive_dna_origami():
    """DNA origami with wide spacing can show positive cooperativity."""
    hill, ctype = compute_hill_cooperativity(12, 6.0, 2, "dna_origami_icosahedron")
    assert hill >= 1.0, f"DNA origami should have positive/neutral cooperativity"
    print(f"  \u2705 test_hill_dna_origami: n_Hill={hill:.2f}, type={ctype}")

def test_loading_curve_decreases():
    """K_effective should decrease with loading for repulsive systems."""
    curve = compute_loading_curve(2, 48, 0.30)
    assert curve[0.0] > curve[1.0], \
        f"K should decrease: empty={curve[0.0]:.4f}, full={curve[1.0]:.4f}"
    print(f"  \u2705 test_loading_curve: K at 0%={curve[0.0]:.4f}, 50%={curve[0.4]:.4f}, 100%={curve[1.0]:.4f}")

def test_capacity_zeolite():
    """Zeolite should have measurable capacity in mmol/g."""
    mmol, mg, max_load = compute_capacity("zeolite_Y", 58.7)  # Ni2+ MW=58.7
    assert mmol > 0.1
    assert mg > 5
    assert 0 < max_load <= 1.0
    print(f"  \u2705 test_capacity_zeolite: {mmol:.3f} mmol/g, {mg:.1f} mg Ni/g, max_load={max_load:.0%}")

def test_capacity_mip_single():
    """MIP single-site capacity should be lower than multi-site scaffolds."""
    mmol_mip, _, _ = compute_capacity("MIP", 207.2)  # Pb2+ MW
    mmol_zeo, _, _ = compute_capacity("zeolite_Y", 207.2)
    assert mmol_mip < mmol_zeo, "MIP should have less capacity than zeolite"
    print(f"  \u2705 test_capacity_mip: MIP={mmol_mip:.3f} vs zeolite={mmol_zeo:.3f} mmol/g")

def test_full_cooperativity_analysis():
    """Full analysis should return all fields."""
    r = analyze_cooperativity("zeolite_Y", target_charge=2, target_mw_g_mol=63.5)
    assert r.n_sites == 48
    assert r.site_spacing_nm == 0.74
    assert r.capacity_mmol_per_g > 0
    assert r.hill_coefficient < 1.0  # Should be negative for zeolite
    assert r.loading_curve is not None
    assert len(r.loading_curve) > 3
    print(f"  \u2705 test_full_coop: n_Hill={r.hill_coefficient:.2f}, "
          f"cap={r.capacity_mg_per_g:.1f} mg/g, max_load={r.max_practical_loading:.0%}")

def test_dendrimer_many_sites():
    """Dendrimer G4 (64 sites at 0.5 nm) should show strong negative cooperativity."""
    r = analyze_cooperativity("dendrimer_PAMAM_G4", target_charge=3)
    assert r.hill_coefficient < 0.95, f"64 sites at 0.5nm with 3+ should be negative, got {r.hill_coefficient}"
    assert r.cooperativity_type == "negative"
    print(f"  \u2705 test_dendrimer: n_Hill={r.hill_coefficient:.2f}, "
          f"repulsion={r.dg_site_repulsion_kj:.1f} kJ/mol")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 24: MASS TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════

def test_stokes_einstein_reasonable():
    """D for typical metal ion should be ~0.5-2 × 10⁻⁹ m²/s."""
    D = stokes_einstein(0.21)  # Ni2+ hydrated radius
    assert 1e-10 < D < 5e-9, f"D={D:.2e} outside expected range"
    print(f"  \u2705 test_stokes_einstein: D(Ni2+)={D:.2e} m²/s")

def test_hindered_diffusion_unhindered():
    """Large pore (3.5 nm) should barely hinder small ion (0.2 nm)."""
    _, hindrance, lam, regime = hindered_diffusion(1e-9, 0.2, 1.75)
    assert hindrance > 0.4
    assert regime == "unhindered"
    print(f"  \u2705 test_hind_unhindered: λ={lam:.3f}, H={hindrance:.3f}, regime={regime}")

def test_hindered_diffusion_severe():
    """ZSM-5 (0.28 nm radius) should severely hinder Pb2+ (0.24 nm hydrated)."""
    _, hindrance, lam, regime = hindered_diffusion(1e-9, 0.24, 0.28)
    assert hindrance < 0.1
    assert regime in ("severely_hindered", "excluded")
    print(f"  \u2705 test_hind_severe: λ={lam:.3f}, H={hindrance:.4f}, regime={regime}")

def test_hindered_diffusion_excluded():
    """Ion larger than pore should be excluded."""
    _, hindrance, lam, regime = hindered_diffusion(1e-9, 0.30, 0.25)
    assert hindrance < 0.001
    assert regime == "excluded"
    print(f"  \u2705 test_hind_excluded: λ={lam:.3f}, regime={regime}")

def test_thiele_reaction_limited():
    """Small particle + slow reaction → reaction-limited."""
    phi, eta = compute_thiele_modulus(1e-9, 1.0, 1e-6)
    assert phi < 0.3, f"Should be reaction-limited, φ={phi}"
    assert eta > 0.9
    print(f"  \u2705 test_thiele_rxn: φ={phi:.3f}, η={eta:.4f}")

def test_thiele_diffusion_limited():
    """Large particle + fast reaction → diffusion-limited."""
    phi, eta = compute_thiele_modulus(1e-12, 100.0, 1e-3)
    assert phi > 3, f"Should be diffusion-limited, φ={phi}"
    assert eta < 0.5
    print(f"  \u2705 test_thiele_diff: φ={phi:.1f}, η={eta:.4f}")

def test_transport_analysis_zeolite():
    """Full transport analysis for Pb2+ in zeolite Y."""
    r = analyze_transport(hydrated_radius_nm=0.24, pore_diameter_nm=0.74,
                           particle_diameter_um=5.0, k_on_M_s=1e6)
    assert r.d_bulk_m2_s > 0
    assert r.hindrance_factor < 1.0
    assert r.lambda_ratio > 0
    print(f"  \u2705 test_transport_zeolite: D_bulk={r.d_bulk_m2_s:.2e}, "
          f"D_pore={r.d_pore_m2_s:.2e}, λ={r.lambda_ratio:.3f}, "
          f"regime={r.transport_regime}")

def test_transport_dna_origami():
    """DNA origami (8 nm pore) should be unhindered for any ion."""
    r = analyze_transport(hydrated_radius_nm=0.30, pore_diameter_nm=8.0)
    assert r.transport_regime == "unhindered"
    assert r.hindrance_factor > 0.5
    print(f"  \u2705 test_transport_dna: regime={r.transport_regime}, H={r.hindrance_factor:.4f}")

def test_capture_time_fast():
    """High k_on + low concentration should give short capture time."""
    ct = predict_capture_time(target_conc_uM=10.0, capacity_mmol_g=1.0,
                               material_g_per_L=1.0, k_on_M_s=1e6)
    assert ct.time_to_90pct_s > 0
    assert ct.time_to_90pct_s < ct.time_to_99pct_s
    print(f"  \u2705 test_capture_fast: t50={ct.time_to_50pct_s:.0f}s, "
          f"t90={ct.time_to_90pct_s:.0f}s, t99={ct.time_to_99pct_s:.0f}s")

def test_capture_time_column():
    """Column mode should predict breakthrough time."""
    ct = predict_capture_time(target_conc_uM=1.0, capacity_mmol_g=2.0,
                               material_g_per_L=10.0,
                               flow_rate_mL_min=1.0, column_volume_mL=5.0)
    assert ct.breakthrough_time_s > 0
    print(f"  \u2705 test_capture_column: breakthrough={ct.breakthrough_time_s:.0f}s "
          f"({ct.breakthrough_time_s/60:.1f} min)")

def test_effectiveness_reduces_capture():
    """Low effectiveness factor should slow capture."""
    ct_full = predict_capture_time(1.0, 1.0, 1.0, 1e6, effectiveness=1.0)
    ct_low = predict_capture_time(1.0, 1.0, 1.0, 1e6, effectiveness=0.1)
    assert ct_low.time_to_90pct_s > ct_full.time_to_90pct_s
    print(f"  \u2705 test_effectiveness: η=1.0 t90={ct_full.time_to_90pct_s:.0f}s, "
          f"η=0.1 t90={ct_low.time_to_90pct_s:.0f}s")

if __name__ == "__main__":
    print("\n\U0001f9ea Sprints 23+24 \u2014 System-Level Prediction\n")
    print("Sprint 23 — Cooperativity:")
    test_repulsion_close_sites(); test_repulsion_far_sites()
    test_repulsion_monovalent_less(); test_hill_negative_close_sites()
    test_hill_independent_mip(); test_hill_positive_dna_origami()
    test_loading_curve_decreases(); test_capacity_zeolite()
    test_capacity_mip_single(); test_full_cooperativity_analysis()
    test_dendrimer_many_sites()
    print("\nSprint 24 — Mass Transport:")
    test_stokes_einstein_reasonable(); test_hindered_diffusion_unhindered()
    test_hindered_diffusion_severe(); test_hindered_diffusion_excluded()
    test_thiele_reaction_limited(); test_thiele_diffusion_limited()
    test_transport_analysis_zeolite(); test_transport_dna_origami()
    test_capture_time_fast(); test_capture_time_column()
    test_effectiveness_reduces_capture()
    print("\n\u2705 All Sprint 23+24 tests passed! (22/22)")
    print("\n\U0001f389 SYSTEM-LEVEL PREDICTION OPERATIONAL\n")

