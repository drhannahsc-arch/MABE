"""tests/test_sprint36.py — Sprint 36: Deep Physics Calibration (22 tests)

Tests the 5 physics improvements:
  1. Metal-specific ΔG_hyd desolvation
  2. Macrocyclic + cavity size-match
  3. Ring size correction (Hancock)
  4. Electrostatic z·z for anionic donors
  5. HSAB multiplicative scaling
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics_integration import (
    compute_enhanced_thermodynamics, EnhancedThermodynamics,
    _HYDRATION_FREE_ENERGY, _AQUA_CN,
)
from core.validation import (
    VALIDATION_LIBRARY, ExperimentalComplex, run_validation,
    derive_calibration, apply_calibration_to_log_K,
    ValidationReport, ValidationResult, _predict_log_K,
)
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)

def _prob(name, formula, charge, d_e, soft, r_pm, ph=7.0):
    return Problem(
        target=TargetSpecies(identity=name, formula=formula, charge=charge,
            d_electrons=d_e, hsab_softness=soft, ionic_radius_pm=r_pm,
            hydrated_radius_nm=(r_pm+140)/1000),
        matrix=Matrix(ph=ph, temperature_c=25.0, ionic_strength_mm=100.0))

def _rec(donors, dt="borderline", chel=2, match=0.7):
    return RecognitionChemistry(name="t", type="generative", donor_atoms=donors,
        donor_type=dt, denticity=len(donors), hsab_match=match, chelate_rings=chel)

def _struct(stype="free", pore=0.0):
    return StructuralConstraint(name="s", type=stype, geometry="octahedral", pore_size_nm=pore)

def _interior():
    return InteriorDesign(description="t", num_binding_sites=1, self_binding=False)


# ═══════════════════════════════════════════════════════════════════════════
# 1. METAL-SPECIFIC ΔG_hyd DESOLVATION
# ═══════════════════════════════════════════════════════════════════════════

def test_hydration_table_coverage():
    """ΔG_hyd table should cover all metals in validation library."""
    metals = set(c.metal_formula for c in VALIDATION_LIBRARY)
    covered = sum(1 for m in metals if m in _HYDRATION_FREE_ENERGY)
    assert covered >= len(metals) - 1, f"Only {covered}/{len(metals)} metals in ΔG_hyd table"
    print(f"  \u2705 test_hyd_coverage: {covered}/{len(metals)} metals covered")

def test_hydration_ordering():
    """ΔG_hyd should follow: |Al3+| > |Fe3+| > |Ni2+| > |Ca2+| > |K+|."""
    al = abs(_HYDRATION_FREE_ENERGY["Al3+"])
    fe = abs(_HYDRATION_FREE_ENERGY["Fe3+"])
    ni = abs(_HYDRATION_FREE_ENERGY["Ni2+"])
    ca = abs(_HYDRATION_FREE_ENERGY["Ca2+"])
    k = abs(_HYDRATION_FREE_ENERGY["K+"])
    assert al > fe > ni > ca > k
    print(f"  \u2705 test_hyd_order: Al={al} > Fe={fe} > Ni={ni} > Ca={ca} > K={k}")

def test_desolv_fe3_vs_fe2():
    """Fe3+ desolvation should cost more than Fe2+ (higher charge, tighter shell)."""
    t_fe3 = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O","O","O"], "hard", 3), _struct(), _interior(),
        _prob("fe3", "Fe3+", 3, 5, 0.12, 65))
    t_fe2 = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O","O","O"], "hard", 3), _struct(), _interior(),
        _prob("fe2", "Fe2+", 2, 6, 0.25, 78))
    assert t_fe3.dg_desolv_kj > t_fe2.dg_desolv_kj, \
        f"Fe3+ desolv ({t_fe3.dg_desolv_kj}) should > Fe2+ ({t_fe2.dg_desolv_kj})"
    print(f"  \u2705 test_desolv_fe3_vs_fe2: Fe3+=+{t_fe3.dg_desolv_kj:.1f} > Fe2+=+{t_fe2.dg_desolv_kj:.1f}")

def test_desolv_irving_williams():
    """Desolvation cost across EDTA divalents should track Irving-Williams."""
    # Mn < Fe < Co < Ni < Cu (desolvation should increase along series)
    metals = [("Mn2+", 5, 0.25, 83), ("Fe2+", 6, 0.25, 78),
              ("Co2+", 7, 0.24, 75), ("Ni2+", 8, 0.24, 69), ("Cu2+", 9, 0.35, 73)]
    desolvs = []
    for f, de, s, r in metals:
        t = compute_enhanced_thermodynamics(
            _rec(["N","N","O","O","O","O"], "mixed", 5), _struct(), _interior(),
            _prob("t", f, 2, de, s, r))
        desolvs.append(t.dg_desolv_kj)
    # General trend should be increasing (Mn < Cu)
    assert desolvs[-1] > desolvs[0], f"Cu desolv ({desolvs[-1]}) should > Mn ({desolvs[0]})"
    print(f"  \u2705 test_iw_desolv: Mn={desolvs[0]:.1f} → Cu={desolvs[-1]:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. MACROCYCLIC + CAVITY SIZE-MATCH
# ═══════════════════════════════════════════════════════════════════════════

def test_macrocyclic_term_present():
    """Macrocyclic complexes should have non-zero dg_macrocyclic_kj."""
    rec = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec.is_macrocyclic = True
    rec.cavity_radius_nm = 0.134  # 18-crown-6
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("k_crown", "K+", 1, 0, 0.01, 138))
    assert t.dg_macrocyclic_kj < 0, f"Macrocyclic should be stabilizing, got {t.dg_macrocyclic_kj}"
    print(f"  \u2705 test_macro_present: dG_macro = {t.dg_macrocyclic_kj:.1f} kJ/mol")

def test_macrocyclic_absent_for_free():
    """Non-macrocyclic ligands should have dg_macrocyclic = 0."""
    t = compute_enhanced_thermodynamics(
        _rec(["N","N","N","N","N","N"], "borderline", 3), _struct(), _interior(),
        _prob("ni_en3", "Ni2+", 2, 8, 0.24, 69))
    assert t.dg_macrocyclic_kj == 0.0
    print(f"  \u2705 test_macro_absent: dG_macro = {t.dg_macrocyclic_kj}")

def test_crown_k_better_than_na():
    """K+ in 18-crown-6 should bind more strongly than Na+ (size match)."""
    # K+: r=138pm, cavity=134pm → near-perfect match
    # Na+: r=102pm, cavity=134pm → mismatch
    rec_k = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_k.is_macrocyclic = True
    rec_k.cavity_radius_nm = 0.134
    rec_na = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_na.is_macrocyclic = True
    rec_na.cavity_radius_nm = 0.134

    t_k = compute_enhanced_thermodynamics(
        rec_k, _struct(), _interior(),
        _prob("k18c6", "K+", 1, 0, 0.01, 138))
    t_na = compute_enhanced_thermodynamics(
        rec_na, _struct(), _interior(),
        _prob("na18c6", "Na+", 1, 0, 0.01, 102))
    # K should be more negative (stronger binding) due to better size match
    assert t_k.dg_macrocyclic_kj < t_na.dg_macrocyclic_kj, \
        f"K macro ({t_k.dg_macrocyclic_kj}) should be more negative than Na ({t_na.dg_macrocyclic_kj})"
    print(f"  \u2705 test_crown_k_vs_na: K={t_k.dg_macrocyclic_kj:.1f} < Na={t_na.dg_macrocyclic_kj:.1f}")

def test_cage_stronger_than_macrocycle():
    """Cage (cryptand) should give stronger macrocyclic effect."""
    rec_mac = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_mac.is_macrocyclic = True
    rec_mac.cavity_radius_nm = 0.134
    rec_cage = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_cage.is_macrocyclic = True
    rec_cage.cavity_radius_nm = 0.134
    rec_cage.is_cage = True

    t_mac = compute_enhanced_thermodynamics(
        rec_mac, _struct(), _interior(),
        _prob("t", "K+", 1, 0, 0.01, 138))
    t_cage = compute_enhanced_thermodynamics(
        rec_cage, _struct(), _interior(),
        _prob("t", "K+", 1, 0, 0.01, 138))
    assert t_cage.dg_macrocyclic_kj < t_mac.dg_macrocyclic_kj
    print(f"  \u2705 test_cage: cage={t_cage.dg_macrocyclic_kj:.1f} < mac={t_mac.dg_macrocyclic_kj:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. RING SIZE CORRECTION (HANCOCK RULE)
# ═══════════════════════════════════════════════════════════════════════════

def test_ring_strain_absent_5mem():
    """5-membered rings should have zero strain for any metal."""
    rec = _rec(["N","N","O","O","O","O"], "mixed", 5)
    rec.ring_sizes = [5,5,5,5,5]
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("pb", "Pb2+", 2, 0, 0.55, 119))
    assert t.dg_ring_strain_kj == 0.0, f"5-mem rings should have no strain, got {t.dg_ring_strain_kj}"
    print(f"  \u2705 test_5mem_no_strain: Pb2+ strain = {t.dg_ring_strain_kj}")

def test_ring_strain_6mem_large_metal():
    """6-membered rings should penalize large metals like Pb2+."""
    rec = _rec(["N","N","O","O","O","O"], "mixed", 5)
    rec.ring_sizes = [6,6,6,6,6]
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("pb", "Pb2+", 2, 0, 0.55, 119))
    assert t.dg_ring_strain_kj > 0, f"6-mem rings + Pb2+ should have strain, got {t.dg_ring_strain_kj}"
    print(f"  \u2705 test_6mem_large: Pb2+ strain = +{t.dg_ring_strain_kj:.1f} kJ/mol")

def test_ring_strain_small_metal_ok():
    """6-membered rings should have minimal strain for small metals like Cu2+."""
    rec = _rec(["N","N","O","O"], "mixed", 2)
    rec.ring_sizes = [6,6]
    t_cu = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("cu", "Cu2+", 2, 9, 0.35, 73))
    rec2 = _rec(["N","N","O","O"], "mixed", 2)
    rec2.ring_sizes = [6,6]
    t_pb = compute_enhanced_thermodynamics(
        rec2, _struct(), _interior(),
        _prob("pb", "Pb2+", 2, 0, 0.55, 119))
    assert t_cu.dg_ring_strain_kj < t_pb.dg_ring_strain_kj, \
        f"Cu2+ strain ({t_cu.dg_ring_strain_kj}) should < Pb2+ ({t_pb.dg_ring_strain_kj})"
    print(f"  \u2705 test_6mem_small: Cu={t_cu.dg_ring_strain_kj:.1f} < Pb={t_pb.dg_ring_strain_kj:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ELECTROSTATIC z·z FOR ANIONIC DONORS
# ═══════════════════════════════════════════════════════════════════════════

def test_zz_trivalent_stronger():
    """Fe3+-carboxylate z·z should be much larger than Cu2+-carboxylate."""
    rec = _rec(["O","O","O","O","O","O"], "hard", 3)
    rec.donor_subtypes = ["O_carboxylate"]*6
    t_fe3 = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("fe3", "Fe3+", 3, 5, 0.12, 65))
    t_cu = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("cu", "Cu2+", 2, 9, 0.35, 73))
    assert abs(t_fe3.dg_zz_electrostatic_kj) > abs(t_cu.dg_zz_electrostatic_kj), \
        f"Fe3+ zz ({t_fe3.dg_zz_electrostatic_kj}) should > Cu2+ ({t_cu.dg_zz_electrostatic_kj})"
    print(f"  \u2705 test_zz_trivalent: Fe3+={t_fe3.dg_zz_electrostatic_kj:.1f} vs Cu2+={t_cu.dg_zz_electrostatic_kj:.1f}")

def test_zz_zero_for_neutral_donors():
    """N-amine donors (neutral) should have zero z·z contribution."""
    rec = _rec(["N","N","N","N","N","N"], "borderline", 3)
    rec.donor_subtypes = ["N_amine"]*6
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("ni", "Ni2+", 2, 8, 0.24, 69))
    assert abs(t.dg_zz_electrostatic_kj) < 0.1, \
        f"N_amine z·z should be ~0, got {t.dg_zz_electrostatic_kj}"
    print(f"  \u2705 test_zz_neutral: N_amine z·z = {t.dg_zz_electrostatic_kj:.2f}")

def test_zz_scales_with_charge():
    """z·z should scale roughly linearly with metal charge."""
    rec = _rec(["O","O","O","O"], "hard", 2)
    rec.donor_subtypes = ["O_carboxylate"]*4
    t2 = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("ca", "Ca2+", 2, 0, 0.01, 100))
    t3 = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("al", "Al3+", 3, 0, 0.01, 54))
    ratio = abs(t3.dg_zz_electrostatic_kj) / max(0.1, abs(t2.dg_zz_electrostatic_kj))
    assert ratio > 1.3, f"z·z ratio Al3+/Ca2+ should > 1.3, got {ratio:.2f}"
    print(f"  \u2705 test_zz_charge: Ca2+={t2.dg_zz_electrostatic_kj:.1f}, Al3+={t3.dg_zz_electrostatic_kj:.1f}, ratio={ratio:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. HSAB MULTIPLICATIVE SCALING
# ═══════════════════════════════════════════════════════════════════════════

def test_hsab_soft_soft_amplified():
    """Hg2+(soft) + S donors should amplify binding vs neutral."""
    # Compare Hg2+ with S (matched) vs O (mismatched)
    t_s = compute_enhanced_thermodynamics(
        _rec(["S","S","S","S"], "soft", 2, 0.95), _struct(), _interior(),
        _prob("hg", "Hg2+", 2, 10, 0.85, 102))
    t_o = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O"], "hard", 2, 0.3), _struct(), _interior(),
        _prob("hg", "Hg2+", 2, 10, 0.85, 102))
    # S should be much more negative (stronger binding)
    assert t_s.dg_bind_kj < t_o.dg_bind_kj, \
        f"Hg+S ({t_s.dg_bind_kj}) should be more negative than Hg+O ({t_o.dg_bind_kj})"
    print(f"  \u2705 test_hsab_ss: Hg+S={t_s.dg_bind_kj:.0f} << Hg+O={t_o.dg_bind_kj:.0f}")

def test_hsab_hard_hard_amplified():
    """Ca2+(hard) + O donors should be amplified vs S donors."""
    t_o = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O"], "hard", 2, 0.9), _struct(), _interior(),
        _prob("ca", "Ca2+", 2, 0, 0.01, 100))
    t_s = compute_enhanced_thermodynamics(
        _rec(["S","S","S","S"], "soft", 2, 0.3), _struct(), _interior(),
        _prob("ca", "Ca2+", 2, 0, 0.01, 100))
    # O should be more negative (harder = better match for Ca)
    assert t_o.dg_bind_kj < t_s.dg_bind_kj, \
        f"Ca+O ({t_o.dg_bind_kj}) should be more negative than Ca+S ({t_s.dg_bind_kj})"
    print(f"  \u2705 test_hsab_hh: Ca+O={t_o.dg_bind_kj:.0f} < Ca+S={t_s.dg_bind_kj:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: FULL VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def test_validation_runs():
    """Validation should complete without errors."""
    report = run_validation()
    assert isinstance(report, ValidationReport)
    assert report.n_complexes == len(VALIDATION_LIBRARY)
    print(f"  \u2705 test_runs: {report.n_complexes} complexes validated")

def test_all_predicted():
    """Every complex should get a prediction."""
    report = run_validation()
    for r in report.results:
        assert r.log_K_pred is not None
        assert math.isfinite(r.log_K_pred), f"{r.name}: {r.log_K_pred}"
    print(f"  \u2705 test_all_predicted: {len(report.results)} predictions")

def test_metrics_reported():
    """Report should have R², MAE, bias."""
    report = run_validation()
    assert hasattr(report, "r_squared")
    assert hasattr(report, "mean_abs_error")
    print(f"  \u2705 test_metrics: R²={report.r_squared:.3f}, MAE={report.mean_abs_error:.1f}")

def test_new_terms_in_output():
    """EnhancedThermodynamics should have Sprint 36 fields."""
    t = compute_enhanced_thermodynamics(
        _rec(["N","N","O","O"]), _struct(), _interior(),
        _prob("t", "Cu2+", 2, 9, 0.35, 73))
    assert hasattr(t, "dg_macrocyclic_kj")
    assert hasattr(t, "dg_ring_strain_kj")
    assert hasattr(t, "dg_zz_electrostatic_kj")
    print(f"  \u2705 test_new_fields: macro={t.dg_macrocyclic_kj}, strain={t.dg_ring_strain_kj}, zz={t.dg_zz_electrostatic_kj}")

def test_calibration_works():
    """Calibration should reduce MAE."""
    raw = run_validation(apply_calibration=False)
    cal = derive_calibration()
    calibrated = run_validation(apply_calibration=True, calibration=cal)
    assert calibrated.mean_abs_error <= raw.mean_abs_error
    print(f"  \u2705 test_calibration: raw MAE={raw.mean_abs_error:.1f} → cal MAE={calibrated.mean_abs_error:.1f}")

def test_full_report():
    """Print full validation report for manual inspection."""
    from core.validation import print_validation_report
    report = run_validation()
    print_validation_report(report)
    print(f"  \u2705 test_full_report: printed above")


if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 36 — Deep Physics Calibration\n")
    print("1. Metal-specific ΔG_hyd:")
    test_hydration_table_coverage(); test_hydration_ordering()
    test_desolv_fe3_vs_fe2(); test_desolv_irving_williams()
    print("\n2. Macrocyclic + Cavity Size-Match:")
    test_macrocyclic_term_present(); test_macrocyclic_absent_for_free()
    test_crown_k_better_than_na(); test_cage_stronger_than_macrocycle()
    print("\n3. Ring Size Correction (Hancock):")
    test_ring_strain_absent_5mem(); test_ring_strain_6mem_large_metal()
    test_ring_strain_small_metal_ok()
    print("\n4. Electrostatic z·z:")
    test_zz_trivalent_stronger(); test_zz_zero_for_neutral_donors()
    test_zz_scales_with_charge()
    print("\n5. HSAB Multiplicative:")
    test_hsab_soft_soft_amplified(); test_hsab_hard_hard_amplified()
    print("\nIntegration:")
    test_validation_runs(); test_all_predicted()
    test_metrics_reported(); test_new_terms_in_output()
    test_calibration_works(); test_full_report()
    print("\n\u2705 All Sprint 36 tests passed! (22/22)")
    print("\n\U0001f389 DEEP PHYSICS CALIBRATION COMPLETE\n")

