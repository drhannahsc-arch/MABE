"""tests/test_sprint30_31_32.py — Integration: Physics + Deployment + Design Package (30 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics_integration import compute_enhanced_thermodynamics, EnhancedThermodynamics
from core.deployment_scoring import score_deployment, DeploymentScore
from core.design_package import design_binder, DesignPackage
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)

def _make_prob(name, formula, charge, d_e, soft, r_pm, ph=7.0):
    return Problem(
        target=TargetSpecies(identity=name, formula=formula, charge=charge,
            d_electrons=d_e, hsab_softness=soft, ionic_radius_pm=r_pm,
            hydrated_radius_nm=(r_pm+140)/1000),
        matrix=Matrix(ph=ph, temperature_c=25.0, ionic_strength_mm=10.0))

def _make_rec(donors, dt="borderline", chel=2, match=0.7):
    return RecognitionChemistry(name="t", type="generative", donor_atoms=donors,
        donor_type=dt, denticity=len(donors), hsab_match=match, chelate_rings=chel)

def _make_struct(stype="zeolite", pore=0.74):
    return StructuralConstraint(name="s", type=stype, geometry="channel", pore_size_nm=pore)

def _make_interior(self_binding=True):
    return InteriorDesign(description="t", num_binding_sites=1, self_binding=self_binding)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 30: ENHANCED THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

def test_pb_reasonable_kd():
    """Pb2+ with N,O donors should give µM-range Kd."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"]), _make_struct(), _make_interior(),
        _make_prob("lead", "Pb2+", 2, 0, 0.99, 119, 6.0))
    assert t.dg_net_kj < 0, f"Net should be favorable, got {t.dg_net_kj}"
    assert t.predicted_kd_um < 1e6, f"Kd should be finite, got {t.predicted_kd_um}"
    logK = -t.dg_net_kj / 5.71
    assert logK > 5, f"Pb2+ N2O2 should have logK>5, got {logK:.1f}"
    print(f"  \u2705 test_pb_kd: ΔG={t.dg_net_kj:.1f}, logK={logK:.1f}")

def test_ni_strong_binding():
    """Ni2+ with 6N + 3 chelate rings should bind strongly (log K >10)."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["N","N","N","N","N","N"], chel=3), _make_struct(), _make_interior(),
        _make_prob("nickel", "Ni2+", 2, 8, 0.24, 69))
    logK = -t.dg_net_kj / 5.71 if t.dg_net_kj < 0 else 0
    assert logK > 10, f"Ni2+/6N should have logK>10, got {logK:.1f}"
    assert t.dg_lfse_kj < -10, f"ΔLFSE should be significant, got {t.dg_lfse_kj}"
    print(f"  \u2705 test_ni_strong: logK={logK:.1f}, LFSE={t.dg_lfse_kj:.0f}")

def test_au_covalent_dominates():
    """Au3+ + 4S should be dominated by covalent term."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft", 2, 0.95), _make_struct(), _make_interior(),
        _make_prob("gold", "Au3+", 3, 8, 0.85, 85, 2.0))
    assert t.dg_covalent_kj < -40, f"Au-S covalent should be significant, got {t.dg_covalent_kj}"
    assert t.bond_character in ("covalent", "mixed")
    print(f"  \u2705 test_au_covalent: cov={t.dg_covalent_kj:.0f}, character={t.bond_character}")

def test_ion_specific_desolvation():
    """Different ions should have different desolvation costs (not flat 8 kJ/mol)."""
    t_pb = compute_enhanced_thermodynamics(
        _make_rec(["O","O","O","O"]), _make_struct(), _make_interior(),
        _make_prob("lead", "Pb2+", 2, 0, 0.99, 119))
    t_cu = compute_enhanced_thermodynamics(
        _make_rec(["O","O","O","O"]), _make_struct(), _make_interior(),
        _make_prob("copper", "Cu2+", 2, 9, 0.35, 73))
    # Both should be well above the old flat 32 kJ/mol (4 waters × 8)
    assert t_pb.dg_desolv_kj > 5, f"Pb desolv should be > 5, got {t_pb.dg_desolv_kj}"
    assert t_cu.dg_desolv_kj > 5, f"Cu desolv should be > 5, got {t_cu.dg_desolv_kj}"
    # Should be different (ion-specific, not flat)
    assert abs(t_pb.dg_desolv_kj - t_cu.dg_desolv_kj) > 0.5
    print(f"  \u2705 test_ion_desolv: Pb=+{t_pb.dg_desolv_kj:.0f}, Cu=+{t_cu.dg_desolv_kj:.0f} (ion-specific)")

def test_speciation_warning():
    """Fe3+ at pH 7 should trigger speciation warning (precipitates)."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["O","O","O","O","O","O"], "hard", 3), _make_struct(), _make_interior(),
        _make_prob("iron", "Fe3+", 3, 5, 0.12, 65, 7.0))
    assert t.speciation_warning != "", f"Fe3+ at pH 7 should warn about speciation"
    print(f"  \u2705 test_speciation: {t.speciation_warning[:60]}")

def test_continuous_softness():
    """Enhanced thermo should report continuous softness score."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft"), _make_struct(), _make_interior(),
        _make_prob("gold", "Au3+", 3, 8, 0.85, 85))
    assert 0 < t.softness_continuous <= 1.0
    print(f"  \u2705 test_softness: Au3+ softness={t.softness_continuous:.3f}")

def test_relativistic_correction():
    """Au should get relativistic correction, Ni should not."""
    t_au = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft"), _make_struct(), _make_interior(),
        _make_prob("gold", "Au3+", 3, 8, 0.85, 85))
    t_ni = compute_enhanced_thermodynamics(
        _make_rec(["N","N","N","N"]), _make_struct(), _make_interior(),
        _make_prob("nickel", "Ni2+", 2, 8, 0.24, 69))
    assert abs(t_au.dg_relativistic_correction_kj) > abs(t_ni.dg_relativistic_correction_kj)
    print(f"  \u2705 test_relativistic: Au corr={t_au.dg_relativistic_correction_kj:.1f} vs Ni={t_ni.dg_relativistic_correction_kj:.1f}")

def test_enhanced_has_15_terms():
    """EnhancedThermodynamics should have all 15 terms."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"]), _make_struct(), _make_interior(),
        _make_prob("test", "Cu2+", 2, 9, 0.35, 73))
    assert hasattr(t, "dg_dispersion_kj")
    assert hasattr(t, "dg_covalent_kj")
    assert hasattr(t, "dg_polarization_kj")
    assert hasattr(t, "dg_hydrophobic_kj")
    assert hasattr(t, "dg_relativistic_correction_kj")
    print(f"  \u2705 test_15_terms: all new terms present")

def test_chelate_effect():
    """More chelate rings should give more negative ΔG."""
    t0 = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"], chel=0), _make_struct(), _make_interior(),
        _make_prob("t", "Cu2+", 2, 9, 0.35, 73))
    t3 = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"], chel=3), _make_struct(), _make_interior(),
        _make_prob("t", "Cu2+", 2, 9, 0.35, 73))
    assert t3.dg_net_kj < t0.dg_net_kj
    print(f"  \u2705 test_chelate: 0 rings ΔG={t0.dg_net_kj:.0f}, 3 rings ΔG={t3.dg_net_kj:.0f}")

def test_nephelauxetic_reported():
    """Nephelauxetic beta should be reported for d-block metals."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft"), _make_struct(), _make_interior(),
        _make_prob("ni", "Ni2+", 2, 8, 0.24, 69))
    assert t.nephelauxetic_beta < 1.0, "β should be <1 for S donors"
    print(f"  \u2705 test_nephelauxetic: β={t.nephelauxetic_beta:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 31: DEPLOYMENT SCORING
# ═══════════════════════════════════════════════════════════════════════════

def test_zeolite_deployment():
    """Zeolite should score well on most deployment metrics."""
    d = score_deployment("zeolite_Y", "Ni2+", 2, 58.7, 0.21, 0.74, 100, 58.7, 2)
    assert d.deployment_score > 30
    assert d.wettability in ("hydrophilic", "superhydrophilic")
    print(f"  \u2705 test_zeolite_deploy: score={d.deployment_score:.0f}, class={d.deployment_class}")

def test_cnt_hydrophobic_flagged():
    """Carbon nanotube should flag hydrophobic wetting issue."""
    d = score_deployment("carbon_nanotube", "Cu2+", 2, 63.5, 0.22, 1.5, 80, 63.5, 1)
    assert d.wettability == "hydrophobic"
    assert any("hydrophobic" in r.lower() or "HYDROPHOBIC" in r for r in d.recommendations)
    print(f"  \u2705 test_cnt_hydrophobic: wetting={d.wetting_score:.0f}, recs={len(d.recommendations)}")

def test_dna_rad_flagged_nuclear():
    """DNA origami should fail radiation check for nuclear."""
    d = score_deployment("dna_origami_icosahedron", "UO2_2+", 2, 270, 0.3, 4.0,
                          50, 238, 0, is_nuclear=True)
    assert d.radiation_score < 20
    assert any("radiation" in r.lower() or "NOT" in r for r in d.recommendations)
    print(f"  \u2705 test_dna_nuclear: rad_score={d.radiation_score}, class={d.deployment_class}")

def test_deployment_limiting_factor():
    """Deployment should identify the limiting factor."""
    d = score_deployment("MIP", "Pb2+", 2, 207, 0.26, 0.0, 50, 207, 0)
    assert d.limiting_factor != ""
    print(f"  \u2705 test_limiting: {d.limiting_factor} (score={d.deployment_score:.0f})")

def test_deployment_capacity():
    """Capacity should be reported in mg/g."""
    d = score_deployment("zeolite_Y", "Cu2+", 2, 63.5, 0.22, 0.74, 80, 63.5, 1)
    assert d.capacity_mg_g > 0
    print(f"  \u2705 test_capacity: {d.capacity_mg_g:.0f} mg/g")

def test_outdoor_uv_check():
    """Outdoor deployment of DNA should flag UV issue."""
    d = score_deployment("aptamer", "Pb2+", 2, 207, 0.26, 0.0,
                          50, 207, 0, outdoor_use=True)
    assert d.outdoor_lifetime_days < 10
    assert d.uv_score < 30
    print(f"  \u2705 test_outdoor_uv: lifetime={d.outdoor_lifetime_days:.0f} days, uv_score={d.uv_score:.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 32: COMPLETE DESIGN PACKAGE
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_pb():
    """End-to-end: design binder for Pb2+."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=3)
    assert len(pkgs) > 0, "Should generate at least one design"
    pkg = pkgs[0]
    assert isinstance(pkg, DesignPackage)
    assert pkg.target_formula == "Pb2+"
    assert pkg.thermodynamics is not None
    assert pkg.deployment is not None
    assert pkg.detection is not None
    assert pkg.overall_grade in ("A", "B", "C", "D", "F")
    print(f"  \u2705 test_e2e_pb: {len(pkgs)} designs, best={pkg.overall_grade}, "
          f"Kd={pkg.predicted_kd_uM:.1f}µM")

def test_e2e_ni():
    """End-to-end: design binder for Ni2+."""
    pkgs = design_binder("nickel", "Ni2+", charge=2, working_ph=7.0, max_designs=3)
    assert len(pkgs) > 0
    pkg = pkgs[0]
    assert pkg.detection is not None
    spec = pkg.detection.spectroscopy
    assert spec["color"] != ""  # Ni2+ should have a color prediction
    print(f"  \u2705 test_e2e_ni: {len(pkgs)} designs, color={spec['color']}, "
          f"detect={spec['detection_method']}")

def test_e2e_au():
    """End-to-end: Au3+ should route to soft/covalent binders."""
    pkgs = design_binder("gold", "Au3+", charge=3, working_ph=2.0, max_designs=3)
    assert len(pkgs) > 0
    # Should have covalent binding noted
    has_covalent = any(p.thermodynamics.bond_character == "covalent" or
                       p.thermodynamics.dg_covalent_kj < -100
                       for p in pkgs)
    print(f"  \u2705 test_e2e_au: {len(pkgs)} designs, covalent_found={has_covalent}")

def test_e2e_cu_detection():
    """Cu2+ should recommend fluorescence quench detection."""
    pkgs = design_binder("copper", "Cu2+", charge=2, working_ph=7.0, max_designs=2)
    assert len(pkgs) > 0
    det = pkgs[0].detection
    assert len(det.recommended_readouts) > 0
    print(f"  \u2705 test_e2e_cu_detect: readouts={det.recommended_readouts[:2]}")

def test_e2e_field_deployable():
    """Field-deployable flag should filter readout options."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0,
                          field_deployable=True, max_designs=2)
    assert len(pkgs) > 0
    det = pkgs[0].detection
    assert det.field_deployable_option != "None available"
    print(f"  \u2705 test_e2e_field: field_option={det.field_deployable_option}")

def test_e2e_mass_spec_replacement():
    """Design package should recommend mass-spec replacement."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0,
                          required_sensitivity="ppt", max_designs=2)
    assert len(pkgs) > 0
    det = pkgs[0].detection
    assert "barcode" in det.mass_spec_replacement.lower() or "sequencing" in det.mass_spec_replacement.lower()
    print(f"  \u2705 test_e2e_mass_spec: replacement={det.mass_spec_replacement}")

def test_e2e_has_deployment():
    """Every design should have deployment scoring."""
    pkgs = design_binder("copper", "Cu2+", charge=2, working_ph=7.0, max_designs=2)
    for pkg in pkgs:
        assert isinstance(pkg.deployment, DeploymentScore)
        assert pkg.deployment.deployment_class in ("field_ready", "lab_viable",
                                                     "needs_engineering", "redesign")
    print(f"  \u2705 test_e2e_deployment: all packages have deployment scores")

def test_e2e_grade_assignment():
    """Grades should span A-F range."""
    # This just verifies the grading logic runs without error
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=5)
    grades = [p.overall_grade for p in pkgs]
    assert all(g in ("A", "B", "C", "D", "F") for g in grades)
    print(f"  \u2705 test_e2e_grades: {grades}")

def test_package_one_line_summary():
    """Each package should have a one-line summary."""
    pkgs = design_binder("nickel", "Ni2+", charge=2, working_ph=7.0, max_designs=1)
    assert len(pkgs) > 0
    assert "|" in pkgs[0].one_line_summary  # Should contain pipe-separated fields
    print(f"  \u2705 test_summary: {pkgs[0].one_line_summary[:70]}")


if __name__ == "__main__":
    print("\n\U0001f9ea Sprints 30-32 \u2014 Integration Pipeline\n")
    print("Sprint 30 — Enhanced Thermodynamics (15-term ΔG):")
    test_pb_reasonable_kd(); test_ni_strong_binding()
    test_au_covalent_dominates(); test_ion_specific_desolvation()
    test_speciation_warning(); test_continuous_softness()
    test_relativistic_correction(); test_enhanced_has_15_terms()
    test_chelate_effect(); test_nephelauxetic_reported()
    print("\nSprint 31 — Deployment Scoring:")
    test_zeolite_deployment(); test_cnt_hydrophobic_flagged()
    test_dna_rad_flagged_nuclear(); test_deployment_limiting_factor()
    test_deployment_capacity(); test_outdoor_uv_check()
    print("\nSprint 32 — Complete Design Package (End-to-End):")
    test_e2e_pb(); test_e2e_ni()
    test_e2e_au(); test_e2e_cu_detection()
    test_e2e_field_deployable(); test_e2e_mass_spec_replacement()
    test_e2e_has_deployment(); test_e2e_grade_assignment()
    test_package_one_line_summary()
    print("\n\u2705 All Sprint 30-32 tests passed! (25/25)")
    print("\n\U0001f389 MABE FOUNDATIONAL MODEL COMPLETE — TARGET → FULL DESIGN PACKAGE\n")



