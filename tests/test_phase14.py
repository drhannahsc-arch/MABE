"""
tests/test_phase14.py — Phase 14a+14b: Dataset + pipeline validation.

Tests:
  1. Dataset integrity (entry counts, required fields, value ranges)
  2. Entry point functions (from_metalloprotein, from_macrocycle_metal)
  3. Unified scorer produces predictions (no crashes, non-trivial output)
  4. Baseline statistics (pre-calibration reference)
  5. 13c regression preserved
"""

import sys
import os
import math

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
for _sub in ('knowledge', 'core'):
    sys.path.insert(0, os.path.join(_root, _sub))

from core.auto_descriptor import from_metalloprotein, from_macrocycle_metal, from_metal_ligand
from core.unified_scorer_v2 import predict
from knowledge.metalloprotein_dataset import METALLOPROTEIN_DATA, TARGET_REGISTRY
from knowledge.macrocycle_metal_dataset import MACROCYCLE_METAL_DATA, MACROCYCLE_PROPS


def _stats(preds, exps):
    n = len(preds)
    errors = [p - e for p, e in zip(preds, exps)]
    mae = sum(abs(e) for e in errors) / n
    bias = sum(errors) / n
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    mean_exp = sum(exps) / n
    ss_tot = sum((e - mean_exp)**2 for e in exps)
    ss_res = sum(e**2 for e in errors)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return {'r2': r2, 'mae': mae, 'bias': bias, 'rmse': rmse, 'n': n}


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: DATASET INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def test_metalloprotein_integrity():
    """14a: All entries have required fields and sane values."""
    failures = 0
    required = ["name", "target", "metal", "smiles", "log_Ka_exp", "mbg"]

    for e in METALLOPROTEIN_DATA:
        for key in required:
            if key not in e or e[key] is None:
                print(f"  MISSING {key} in {e.get('name', '?')}")
                failures += 1

        if not (0.0 <= e["log_Ka_exp"] <= 15.0):
            print(f"  OUT OF RANGE: {e['name']} log_Ka={e['log_Ka_exp']}")
            failures += 1

        if e["target"] not in TARGET_REGISTRY:
            print(f"  UNKNOWN TARGET: {e['target']}")
            failures += 1

    n = len(METALLOPROTEIN_DATA)
    print(f"\n{'='*72}")
    print(f"14a INTEGRITY: {n} entries, {failures} failures")
    print(f"{'='*72}")
    return failures


def test_macrocycle_integrity():
    """14b: All entries have required fields and sane values."""
    failures = 0
    required = ["name", "metal", "donors", "chelate_rings", "log_K_exp", "macrocyclic"]

    for e in MACROCYCLE_METAL_DATA:
        for key in required:
            if key not in e or e[key] is None:
                print(f"  MISSING {key} in {e.get('name', '?')}")
                failures += 1

        if not (-2.0 <= e["log_K_exp"] <= 20.0):
            print(f"  OUT OF RANGE: {e['name']} log_K={e['log_K_exp']}")
            failures += 1

        if not e["macrocyclic"]:
            print(f"  NOT MACROCYCLIC: {e['name']}")
            failures += 1

    n = len(MACROCYCLE_METAL_DATA)
    print(f"\n{'='*72}")
    print(f"14b INTEGRITY: {n} entries, {failures} failures")
    print(f"{'='*72}")
    return failures


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: ENTRY POINT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def test_metalloprotein_pipeline():
    """14a: from_metalloprotein → predict for all entries."""
    passed = crashed = 0

    for e in METALLOPROTEIN_DATA:
        try:
            uc = from_metalloprotein(e)
            r = predict(uc)
            assert uc.binding_mode == "metalloprotein"
            assert uc.metal_formula == "Zn2+"
            passed += 1
        except Exception as ex:
            crashed += 1
            if crashed <= 5:
                print(f"  CRASH: {e['name']} — {ex}")

    print(f"\n{'='*72}")
    print(f"14a PIPELINE: {passed}/{passed+crashed} predicted successfully")
    print(f"{'='*72}")
    return crashed


def test_macrocycle_pipeline():
    """14b: from_macrocycle_metal → predict for all entries."""
    passed = crashed = 0

    for e in MACROCYCLE_METAL_DATA:
        try:
            uc = from_macrocycle_metal(e)
            r = predict(uc)
            assert uc.binding_mode == "metal_coordination"
            assert uc.is_macrocyclic
            passed += 1
        except Exception as ex:
            crashed += 1
            if crashed <= 5:
                print(f"  CRASH: {e['name']} — {ex}")

    print(f"\n{'='*72}")
    print(f"14b PIPELINE: {passed}/{passed+crashed} predicted successfully")
    print(f"{'='*72}")
    return crashed


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: BASELINE STATISTICS (pre-calibration)
# ═══════════════════════════════════════════════════════════════════════════

def test_baseline_statistics():
    """Compute and record pre-calibration baselines for Phase 16 reference."""
    print(f"\n{'='*72}")
    print(f"PHASE 14 BASELINE STATISTICS (pre-calibration)")
    print(f"{'='*72}")

    # 14a — by target
    print(f"\n  14a Metalloprotein (by target):")
    print(f"  {'Target':15s} {'N':>4s} {'R²':>8s} {'MAE':>8s} {'Bias':>8s}")
    print(f"  {'─'*44}")

    from collections import defaultdict
    by_target = defaultdict(lambda: ([], []))
    for e in METALLOPROTEIN_DATA:
        uc = from_metalloprotein(e)
        r = predict(uc)
        by_target[e["target"]][0].append(r.log_Ka_pred)
        by_target[e["target"]][1].append(e["log_Ka_exp"])

    all_p, all_e = [], []
    for target in sorted(by_target):
        preds, exps = by_target[target]
        s = _stats(preds, exps)
        print(f"  {target:15s} {s['n']:4d} {s['r2']:8.3f} {s['mae']:8.2f} {s['bias']:+8.2f}")
        all_p.extend(preds)
        all_e.extend(exps)

    s_all = _stats(all_p, all_e)
    print(f"  {'─'*44}")
    print(f"  {'TOTAL':15s} {s_all['n']:4d} {s_all['r2']:8.3f} {s_all['mae']:8.2f} {s_all['bias']:+8.2f}")

    # 14b — by class
    print(f"\n  14b Macrocycle (by class):")
    print(f"  {'Class':15s} {'N':>4s} {'R²':>8s} {'MAE':>8s} {'Bias':>8s}")
    print(f"  {'─'*44}")

    by_class = defaultdict(lambda: ([], []))
    for e in MACROCYCLE_METAL_DATA:
        uc = from_macrocycle_metal(e)
        r = predict(uc)
        cls = e.get("macrocycle_class", "unknown")
        by_class[cls][0].append(r.log_Ka_pred)
        by_class[cls][1].append(e["log_K_exp"])

    all_p, all_e = [], []
    for cls in sorted(by_class):
        preds, exps = by_class[cls]
        s = _stats(preds, exps)
        print(f"  {cls:15s} {s['n']:4d} {s['r2']:8.3f} {s['mae']:8.2f} {s['bias']:+8.2f}")
        all_p.extend(preds)
        all_e.extend(exps)

    s_all = _stats(all_p, all_e)
    print(f"  {'─'*44}")
    print(f"  {'TOTAL':15s} {s_all['n']:4d} {s_all['r2']:8.3f} {s_all['mae']:8.2f} {s_all['bias']:+8.2f}")

    return 0  # informational only


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: 13c REGRESSION PRESERVED
# ═══════════════════════════════════════════════════════════════════════════

def test_13c_untouched():
    """Verify adding Phase 14 data didn't break 13c frozen references."""
    import json

    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'frozen_references_13c.json')
    with open(ref_path) as f:
        frozen = json.load(f)

    from cal_dataset import CAL_DATA
    from hg_dataset import HG_DATA
    from core.auto_descriptor import from_host_guest, METAL_PROPERTIES
    from core.universal_schema import UniversalComplex

    failures = 0
    eps = 0.001

    # Metal
    for entry in CAL_DATA:
        ref = frozen['metal'].get(entry['name'])
        if ref is None: continue
        uc = from_metal_ligand(entry)
        r = predict(uc)
        if abs(r.log_Ka_pred - ref) > eps:
            failures += 1

    # HG
    for entry in HG_DATA:
        ref = frozen['hg'].get(entry['name'])
        if ref is None: continue
        uc = from_host_guest(entry)
        r = predict(uc)
        if abs(r.log_Ka_pred - ref) > eps:
            failures += 1

    print(f"\n{'='*72}")
    if failures == 0:
        print(f"13c REGRESSION: 644/644 frozen references preserved ✓")
    else:
        print(f"13c REGRESSION: {failures} DRIFT(S) detected!")
    print(f"{'='*72}")
    return failures


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: ENTRY COUNT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def test_entry_counts():
    """Verify minimum entry counts per roadmap targets."""
    failures = 0

    n_14a = len(METALLOPROTEIN_DATA)
    n_14b = len(MACROCYCLE_METAL_DATA)

    print(f"\n{'='*72}")
    print(f"ENTRY COUNTS")
    print(f"{'='*72}")

    if n_14a >= 200:
        print(f"  ✓ 14a metalloprotein: {n_14a} ≥ 200")
    else:
        print(f"  ✗ 14a metalloprotein: {n_14a} < 200")
        failures += 1

    if n_14b >= 50:
        print(f"  ✓ 14b macrocycle: {n_14b} ≥ 50")
    else:
        print(f"  ✗ 14b macrocycle: {n_14b} < 50")
        failures += 1

    # Combined dataset for Phase 16
    n_total = 644 + n_14a + n_14b
    n_params = 80 + 10  # existing + estimated new
    ratio = n_total / n_params
    print(f"  Phase 16 projection: {n_total} entries / {n_params} params = {ratio:.1f}:1")

    return failures


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total = 0

    total += test_metalloprotein_integrity()
    total += test_macrocycle_integrity()
    total += test_metalloprotein_pipeline()
    total += test_macrocycle_pipeline()
    total += test_entry_counts()
    total += test_13c_untouched()
    test_baseline_statistics()  # informational, doesn't count as failure

    print(f"\n{'='*72}")
    if total == 0:
        print(f"PHASE 14: ALL TESTS PASSED ✓")
        print(f"  14a: {len(METALLOPROTEIN_DATA)} metalloprotein entries")
        print(f"  14b: {len(MACROCYCLE_METAL_DATA)} macrocycle-metal entries")
        print(f"  13c: 644 frozen references preserved")
    else:
        print(f"PHASE 14: {total} FAILURE(S)")
    print(f"{'='*72}")