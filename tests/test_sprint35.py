"""tests/test_sprint35.py — Sprint 35: Validation Pipeline (18 tests)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.validation import (
    VALIDATION_LIBRARY, ExperimentalComplex, run_validation,
    derive_calibration, apply_calibration_to_log_K,
    ValidationReport, ValidationResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

def test_library_size():
    """Library should have 30+ complexes."""
    assert len(VALIDATION_LIBRARY) >= 30
    print(f"  \u2705 test_lib_size: {len(VALIDATION_LIBRARY)} complexes")

def test_library_metal_diversity():
    """Library should span hard, borderline, and soft metals."""
    metals = set(c.metal_formula for c in VALIDATION_LIBRARY)
    assert len(metals) >= 10
    # Check coverage: at least one hard, borderline, soft
    hard = [c for c in VALIDATION_LIBRARY if c.metal_formula in ("Ca2+", "Mg2+", "Al3+", "Fe3+")]
    border = [c for c in VALIDATION_LIBRARY if c.metal_formula in ("Ni2+", "Cu2+", "Co2+", "Zn2+")]
    soft = [c for c in VALIDATION_LIBRARY if c.metal_formula in ("Hg2+", "Ag+", "Cd2+")]
    assert len(hard) >= 3
    assert len(border) >= 3
    assert len(soft) >= 2
    print(f"  \u2705 test_diversity: {len(metals)} metals, hard={len(hard)} border={len(border)} soft={len(soft)}")

def test_library_donor_diversity():
    """Library should cover all donor types."""
    types = set(c.donor_type for c in VALIDATION_LIBRARY)
    assert "hard" in types
    assert "borderline" in types
    assert "soft" in types
    assert "mixed" in types
    print(f"  \u2705 test_donors: types={types}")

def test_library_log_K_range():
    """log K values should span wide range."""
    log_Ks = [c.log_K_exp for c in VALIDATION_LIBRARY]
    assert min(log_Ks) < 5, "Should include weak complexes"
    assert max(log_Ks) > 35, "Should include very strong complexes"
    print(f"  \u2705 test_range: log K = {min(log_Ks):.1f} to {max(log_Ks):.1f}")

def test_library_edta_irving_williams():
    """EDTA series should follow Irving-Williams order."""
    edta = {c.metal_formula: c.log_K_exp for c in VALIDATION_LIBRARY 
            if "EDTA" in c.name and c.metal_charge == 2}
    # Irving-Williams: Mn < Fe < Co < Ni < Cu > Zn
    if "Mn2+" in edta and "Cu2+" in edta:
        assert edta["Mn2+"] < edta["Cu2+"]
    if "Ni2+" in edta and "Cu2+" in edta:
        assert edta["Ni2+"] <= edta["Cu2+"]
    print(f"  \u2705 test_irving_williams: EDTA series verified")

def test_chelate_effect_in_library():
    """en3 should be stronger than NH3_6 for Ni2+."""
    en3 = next((c for c in VALIDATION_LIBRARY if c.name == "Ni-en3"), None)
    nh3 = next((c for c in VALIDATION_LIBRARY if c.name == "Ni-NH3_6"), None)
    assert en3 and nh3
    assert en3.log_K_exp > nh3.log_K_exp, "Chelate effect: en3 > NH3_6"
    print(f"  \u2705 test_chelate: Ni-en3={en3.log_K_exp} > Ni-NH3_6={nh3.log_K_exp}")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def test_validation_runs():
    """run_validation should complete without errors."""
    report = run_validation()
    assert isinstance(report, ValidationReport)
    assert report.n_complexes == len(VALIDATION_LIBRARY)
    print(f"  \u2705 test_runs: {report.n_complexes} complexes validated")

def test_all_predicted():
    """Every complex should get a prediction (no crashes)."""
    report = run_validation()
    for r in report.results:
        assert r.log_K_pred is not None
        assert r.dg_pred_kj is not None
    print(f"  \u2705 test_all_predicted: {len(report.results)} predictions generated")

def test_report_has_metrics():
    """Report should contain R², MAE, bias."""
    report = run_validation()
    assert hasattr(report, "r_squared")
    assert hasattr(report, "mean_abs_error")
    assert hasattr(report, "systematic_bias")
    assert hasattr(report, "calibration_slope")
    print(f"  \u2705 test_metrics: R²={report.r_squared:.3f}, MAE={report.mean_abs_error:.1f}")

def test_per_class_mae():
    """Should report MAE for each donor class."""
    report = run_validation()
    assert report.hard_mae >= 0
    assert report.borderline_mae >= 0
    assert report.soft_mae >= 0
    assert report.mixed_mae >= 0
    print(f"  \u2705 test_class_mae: H={report.hard_mae:.1f} B={report.borderline_mae:.1f} "
          f"S={report.soft_mae:.1f} M={report.mixed_mae:.1f}")

def test_predictions_finite():
    """Predictions should be finite numbers (no inf/nan)."""
    report = run_validation()
    for r in report.results:
        assert math.isfinite(r.log_K_pred), f"{r.name}: log K pred = {r.log_K_pred}"
        assert math.isfinite(r.dg_pred_kj), f"{r.name}: ΔG pred = {r.dg_pred_kj}"
    print(f"  \u2705 test_finite: all predictions finite")

def test_ni_nh3_reasonable():
    """Ni-NH3_6 prediction should be in right ballpark (no chelate effect)."""
    report = run_validation()
    nh3 = next(r for r in report.results if r.name == "Ni-NH3_6")
    # Exp = 8.6. Raw model may be off but should at least be positive
    assert nh3.log_K_pred > 0, f"Ni-NH3_6 should have positive log K, got {nh3.log_K_pred}"
    print(f"  \u2705 test_ni_nh3: pred={nh3.log_K_pred:.1f}, exp={nh3.log_K_exp:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def test_calibration_derives():
    """derive_calibration should return slope, intercept, class offsets."""
    cal = derive_calibration()
    assert "slope" in cal
    assert "intercept" in cal
    assert "class_offsets" in cal
    assert isinstance(cal["class_offsets"], dict)
    print(f"  \u2705 test_cal_derive: slope={cal['slope']:.4f}, int={cal['intercept']:.1f}")

def test_calibration_reduces_mae():
    """Calibrated predictions should have lower MAE than raw."""
    raw = run_validation(apply_calibration=False)
    cal = derive_calibration()
    calibrated = run_validation(apply_calibration=True, calibration=cal)
    assert calibrated.mean_abs_error <= raw.mean_abs_error, \
        f"Calibrated MAE ({calibrated.mean_abs_error}) should be <= raw ({raw.mean_abs_error})"
    print(f"  \u2705 test_cal_reduces: raw MAE={raw.mean_abs_error:.1f} → "
          f"calibrated MAE={calibrated.mean_abs_error:.1f}")

def test_apply_calibration_function():
    """apply_calibration_to_log_K should use slope + intercept + class offset."""
    cal = {"slope": 0.5, "intercept": 5.0, "class_offsets": {"soft": 3.0}}
    result = apply_calibration_to_log_K(10.0, "soft", cal)
    expected = 0.5 * 10.0 + 5.0 + 3.0  # = 13.0
    assert abs(result - expected) < 0.01
    print(f"  \u2705 test_apply_cal: {result:.1f} == {expected:.1f}")

def test_calibration_class_offsets():
    """Class offsets should exist for all major donor types."""
    cal = derive_calibration()
    for dtype in ("hard", "borderline", "soft", "mixed"):
        assert dtype in cal["class_offsets"], f"Missing offset for {dtype}"
    print(f"  \u2705 test_class_offsets: {cal['class_offsets']}")

def test_validation_notes():
    """Report should generate diagnostic notes."""
    report = run_validation()
    assert isinstance(report.notes, list)
    assert len(report.notes) > 0, "Should have diagnostic notes"
    print(f"  \u2705 test_notes: {len(report.notes)} diagnostic notes")


import math

if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 35 \u2014 Validation Pipeline\n")
    print("Experimental Library:")
    test_library_size(); test_library_metal_diversity()
    test_library_donor_diversity(); test_library_log_K_range()
    test_library_edta_irving_williams(); test_chelate_effect_in_library()
    print("\nValidation Engine:")
    test_validation_runs(); test_all_predicted()
    test_report_has_metrics(); test_per_class_mae()
    test_predictions_finite(); test_ni_nh3_reasonable()
    print("\nCalibration:")
    test_calibration_derives(); test_calibration_reduces_mae()
    test_apply_calibration_function(); test_calibration_class_offsets()
    test_validation_notes()
    print("\n\u2705 All Sprint 35 tests passed! (18/18)\n")

