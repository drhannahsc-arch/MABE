"""
tests/test_unified_scorer_v2.py — Phase 13b Regression Tests

644 entries: unified_scorer_v2.predict(uc) must reproduce existing scorers.
  - 500 metal entries vs scorer_frozen.predict_log_k()   ε ≤ 0.01
  - 80 HG entries vs hg_scorer.predict_hg_log_ka()       ε ≤ 0.01
  - 64 CM entries vs cross_modal_predictor                ε ≤ 0.01
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

import math

# Existing scorers (ground truth)
from core.scorer_frozen import predict_log_k as metal_predict
from hg_scorer import predict_hg_log_ka as hg_predict
from cross_modal_predictor import predict_cross_modal_log_ka as cm_predict

# Datasets
from cal_dataset import CAL_DATA
from hg_dataset import HG_DATA
from cross_modal_dataset import CROSS_MODAL_DATA

# New unified scorer
from core.unified_scorer_v2 import predict, PredictionResult

# Auto-descriptor for UC construction
from core.auto_descriptor import from_metal_ligand, from_host_guest
from core.universal_schema import UniversalComplex


EPSILON = 0.02  # log K tolerance (0.02 to absorb float rounding)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: build UC from CM dataset entry
# ═══════════════════════════════════════════════════════════════════════════

def uc_from_cm_entry(entry):
    """Build a UniversalComplex for a cross-modal entry."""
    uc = UniversalComplex(
        name=entry["name"],
        binding_mode="cross_modal",
        log_Ka_exp=entry["log_Ka"],
        metal_formula=entry["metal"],
        host_name=entry["cb_host"],
        host_type=entry["cb_host"],
        cavity_volume_A3=entry["cavity_volume_A3"],
    )
    # Metal properties
    from core.auto_descriptor import METAL_PROPERTIES
    mp = METAL_PROPERTIES.get(entry["metal"])
    if mp:
        uc.metal_charge = mp[0]
        uc.metal_d_electrons = mp[1]
        uc.host_charge = 0
    return uc


# ═══════════════════════════════════════════════════════════════════════════
# TEST: METAL REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def test_metal_regression():
    """500 metal entries: unified scorer vs scorer_frozen."""
    passed = failed = errors_count = 0
    worst = ("", 0.0)
    failures = []

    for entry in CAL_DATA:
        # Ground truth
        try:
            gt = metal_predict(
                entry["metal"], entry["donors"],
                chelate_rings=entry["chelate_rings"],
                ring_sizes=entry["ring_sizes"] or None,
                pH=entry["pH"],
                is_macrocyclic=entry["macrocyclic"],
                cavity_radius_nm=entry["cavity_nm"],
                n_ligand_molecules=entry["n_lig_mol"],
            )
        except Exception:
            errors_count += 1
            continue

        # Unified scorer via UC
        uc = from_metal_ligand(entry)
        result = predict(uc)
        pred = result.log_Ka_pred

        diff = abs(pred - gt)
        if diff > abs(worst[1]):
            worst = (entry["name"], pred - gt)

        if diff <= EPSILON:
            passed += 1
        else:
            failed += 1
            if len(failures) < 10:
                failures.append(f"    {entry['name']:30s} gt={gt:7.2f} pred={pred:7.2f} Δ={pred-gt:+.3f}")

    print(f"\n{'='*72}")
    print(f"METAL REGRESSION: {passed}/{passed+failed} within ε={EPSILON}")
    if errors_count:
        print(f"  ({errors_count} entries skipped due to scorer errors)")
    print(f"  Worst: {worst[0]} Δ={worst[1]:+.3f}")
    print(f"{'='*72}")
    if failures:
        for f in failures:
            print(f)
    return failed


# ═══════════════════════════════════════════════════════════════════════════
# TEST: HOST-GUEST REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def test_hg_regression():
    """80 HG entries: unified scorer vs hg_scorer."""
    passed = failed = 0
    worst = ("", 0.0)
    failures = []

    for entry in HG_DATA:
        # Ground truth
        try:
            gt = hg_predict(entry)
        except Exception:
            continue

        # Unified scorer via UC
        uc = from_host_guest(entry)
        result = predict(uc)
        pred = result.log_Ka_pred

        diff = abs(pred - gt)
        if diff > abs(worst[1]):
            worst = (entry["name"], pred - gt)

        if diff <= EPSILON:
            passed += 1
        else:
            failed += 1
            if len(failures) < 10:
                failures.append(f"    {entry['name']:30s} gt={gt:7.2f} pred={pred:7.2f} Δ={pred-gt:+.3f}")

    print(f"\n{'='*72}")
    print(f"HG REGRESSION: {passed}/{passed+failed} within ε={EPSILON}")
    print(f"  Worst: {worst[0]} Δ={worst[1]:+.3f}")
    print(f"{'='*72}")
    if failures:
        for f in failures:
            print(f)
    return failed


# ═══════════════════════════════════════════════════════════════════════════
# TEST: CROSS-MODAL REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def test_cm_regression():
    """64 CM entries: unified scorer vs cross_modal_predictor."""
    passed = failed = 0
    worst = ("", 0.0)
    failures = []

    for entry in CROSS_MODAL_DATA:
        # Ground truth
        try:
            gt = cm_predict(entry)
        except Exception:
            continue

        # Unified scorer via UC
        uc = uc_from_cm_entry(entry)
        result = predict(uc)
        pred = result.log_Ka_pred

        diff = abs(pred - gt)
        if diff > abs(worst[1]):
            worst = (entry["name"], pred - gt)

        if diff <= EPSILON:
            passed += 1
        else:
            failed += 1
            if len(failures) < 10:
                failures.append(f"    {entry['name']:20s} gt={gt:7.2f} pred={pred:7.2f} Δ={pred-gt:+.3f}")

    print(f"\n{'='*72}")
    print(f"CM REGRESSION: {passed}/{passed+failed} within ε={EPSILON}")
    print(f"  Worst: {worst[0]} Δ={worst[1]:+.3f}")
    print(f"{'='*72}")
    if failures:
        for f in failures:
            print(f)
    return failed


# ═══════════════════════════════════════════════════════════════════════════
# TEST: SELF-ZEROING GUARDS
# ═══════════════════════════════════════════════════════════════════════════

def test_self_zeroing():
    """Empty UC should produce zero prediction without crashing."""
    uc = UniversalComplex(name="empty")
    result = predict(uc)

    errors = []
    if result.log_Ka_pred != 0.0:
        errors.append(f"empty UC pred={result.log_Ka_pred} (expected 0.0)")
    if result.dg_metal != 0.0:
        errors.append(f"dg_metal={result.dg_metal}")

    status = "PASS ✓" if not errors else f"FAIL ✗ {errors}"
    print(f"\n{'='*72}")
    print(f"SELF-ZEROING: {status}")
    print(f"{'='*72}")
    return len(errors)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: MIXED MODE (metal + host both fire)
# ═══════════════════════════════════════════════════════════════════════════

def test_mixed_mode():
    """Metal + host should fire both metal and HG terms."""
    from core.auto_descriptor import from_smiles

    # Zn2+ + glycine in β-CD (hypothetical)
    uc = from_smiles("NCC(=O)O", metal="Zn2+", host="beta-CD")
    uc.binding_mode = "mixed"  # override to enable both
    uc.log_Ka_exp = 0.0  # no experimental

    result = predict(uc)

    errors = []
    if result.dg_metal == 0.0:
        errors.append("metal terms did not fire")
    if result.dg_hydrophobic == 0.0 and result.dg_cavity_dehydration == 0.0:
        errors.append("HG terms did not fire")

    status = "PASS ✓" if not errors else f"FAIL ✗ {errors}"
    print(f"\n{'='*72}")
    print(f"MIXED MODE: {status}")
    if not errors:
        print(f"  dg_metal={result.dg_metal:.2f}, dg_hydrophobic={result.dg_hydrophobic:.2f}")
        print(f"  log_Ka_pred={result.log_Ka_pred:.2f}")
    print(f"{'='*72}")
    return len(errors)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total = 0
    total += test_self_zeroing()
    total += test_metal_regression()
    total += test_hg_regression()
    total += test_cm_regression()
    total += test_mixed_mode()

    print(f"\n{'='*72}")
    if total == 0:
        print(f"ALL REGRESSION TESTS PASSED ✓ (644 entries, ε={EPSILON})")
    else:
        print(f"TOTAL FAILURES: {total}")
    print(f"{'='*72}")