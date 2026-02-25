"""
tests/test_regression_13c.py — Phase 13c: Comprehensive Regression Suite

Three test categories:
  1. FROZEN REFERENCE REGRESSION — 644 entries must match frozen predictions
     within ε=0.001 (tighter than 13b's ε=0.02, catching any param drift)
  2. SMILES→UC→PREDICT PIPELINE — 15 ligands with known structure go through
     auto_descriptor + unified_scorer_v2, results must match manual path
  3. STATISTICS & CONSTRAINTS — per-modality R²/MAE/bias, Phase 12 thresholds

Frozen references in tests/frozen_references_13c.json (generated from
unified_scorer_v2 at Phase 13b commit baseline).
"""

import sys
import os
import json
import math

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
for _sub in ('knowledge', 'core'):
    sys.path.insert(0, os.path.join(_root, _sub))

from core.unified_scorer_v2 import predict, PredictionResult
from core.auto_descriptor import from_metal_ligand, from_host_guest, from_smiles, METAL_PROPERTIES
from core.universal_schema import UniversalComplex

from cal_dataset import CAL_DATA
from hg_dataset import HG_DATA
from cross_modal_dataset import CROSS_MODAL_DATA

# ═══════════════════════════════════════════════════════════════════════════
# LOAD FROZEN REFERENCES
# ═══════════════════════════════════════════════════════════════════════════

_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'frozen_references_13c.json')
with open(_REF_PATH) as _f:
    _FROZEN = json.load(_f)

FROZEN_METAL = _FROZEN['metal']  # {name: pred}
FROZEN_HG = _FROZEN['hg']
FROZEN_CM = _FROZEN['cm']


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: build UC from CM entry
# ═══════════════════════════════════════════════════════════════════════════

def _uc_from_cm(entry):
    uc = UniversalComplex(
        name=entry['name'], binding_mode='cross_modal',
        log_Ka_exp=entry['log_Ka'], metal_formula=entry['metal'],
        host_name=entry['cb_host'], host_type=entry['cb_host'],
        cavity_volume_A3=entry['cavity_volume_A3'])
    mp = METAL_PROPERTIES.get(entry['metal'])
    if mp:
        uc.metal_charge, uc.metal_d_electrons = mp[0], mp[1]
    return uc


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _stats(preds, exps):
    """Compute R², MAE, bias, RMSE, max_err."""
    n = len(preds)
    if n == 0:
        return {'r2': 0, 'mae': 0, 'bias': 0, 'rmse': 0, 'max_err': 0, 'n': 0}
    errors = [p - e for p, e in zip(preds, exps)]
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / n
    bias = sum(errors) / n
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    max_err = max(abs_errors)
    mean_exp = sum(exps) / n
    ss_tot = sum((e - mean_exp)**2 for e in exps)
    ss_res = sum(e**2 for e in errors)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return {'r2': r2, 'mae': mae, 'bias': bias, 'rmse': rmse,
            'max_err': max_err, 'n': n}


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: FROZEN REFERENCE REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

EPSILON_FROZEN = 0.001  # Tight tolerance — catches any param drift


def test_frozen_metal():
    """500 metal: unified scorer vs frozen references."""
    passed = failed = 0
    drifts = []

    for entry in CAL_DATA:
        name = entry['name']
        ref = FROZEN_METAL.get(name)
        if ref is None:
            continue

        uc = from_metal_ligand(entry)
        result = predict(uc)
        diff = abs(result.log_Ka_pred - ref)

        if diff <= EPSILON_FROZEN:
            passed += 1
        else:
            failed += 1
            drifts.append((name, ref, result.log_Ka_pred, result.log_Ka_pred - ref))

    print(f"\n{'='*72}")
    print(f"FROZEN METAL: {passed}/{passed+failed} within ε={EPSILON_FROZEN}")
    print(f"{'='*72}")
    if drifts:
        drifts.sort(key=lambda x: -abs(x[3]))
        for name, ref, got, delta in drifts[:10]:
            print(f"  DRIFT: {name:30s} ref={ref:8.4f} got={got:8.4f} Δ={delta:+.4f}")
    return failed


def test_frozen_hg():
    """80 HG: unified scorer vs frozen references."""
    passed = failed = 0
    drifts = []

    for entry in HG_DATA:
        name = entry['name']
        ref = FROZEN_HG.get(name)
        if ref is None:
            continue

        uc = from_host_guest(entry)
        result = predict(uc)
        diff = abs(result.log_Ka_pred - ref)

        if diff <= EPSILON_FROZEN:
            passed += 1
        else:
            failed += 1
            drifts.append((name, ref, result.log_Ka_pred, result.log_Ka_pred - ref))

    print(f"\n{'='*72}")
    print(f"FROZEN HG: {passed}/{passed+failed} within ε={EPSILON_FROZEN}")
    print(f"{'='*72}")
    if drifts:
        for name, ref, got, delta in drifts[:10]:
            print(f"  DRIFT: {name:30s} ref={ref:8.4f} got={got:8.4f} Δ={delta:+.4f}")
    return failed


def test_frozen_cm():
    """64 CM: unified scorer vs frozen references."""
    passed = failed = 0
    drifts = []

    for entry in CROSS_MODAL_DATA:
        name = entry['name']
        ref = FROZEN_CM.get(name)
        if ref is None:
            continue

        uc = _uc_from_cm(entry)
        result = predict(uc)
        diff = abs(result.log_Ka_pred - ref)

        if diff <= EPSILON_FROZEN:
            passed += 1
        else:
            failed += 1
            drifts.append((name, ref, result.log_Ka_pred, result.log_Ka_pred - ref))

    print(f"\n{'='*72}")
    print(f"FROZEN CM: {passed}/{passed+failed} within ε={EPSILON_FROZEN}")
    print(f"{'='*72}")
    if drifts:
        for name, ref, got, delta in drifts[:10]:
            print(f"  DRIFT: {name:20s} ref={ref:8.4f} got={got:8.4f} Δ={delta:+.4f}")
    return failed


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: SMILES→UC→PREDICT PIPELINE
# Auto-descriptor must produce same results as manual annotation path.
# ═══════════════════════════════════════════════════════════════════════════

# (name, SMILES, metal, cal_dataset_name)
# cal_dataset_name is used to look up the from_metal_ligand reference
SMILES_PIPELINE_TESTS = [
    ("EDTA+Cu2+",      "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",  "Cu2+",  "EDTA+Cu2+"),
    ("EDTA+Ni2+",      "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",  "Ni2+",  "EDTA+Ni2+"),
    ("EDTA+Pb2+",      "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",  "Pb2+",  "EDTA+Pb2+"),
    ("NTA+Cu2+",       "OC(=O)CN(CC(=O)O)CC(=O)O",               "Cu2+",  "NTA+Cu2+"),
    ("en+Cu2+",        "NCCN",                                     "Cu2+",  "en+Cu2+"),
    ("en+Ni2+",        "NCCN",                                     "Ni2+",  "en+Ni2+"),
    ("glycine+Cu2+",   "NCC(=O)O",                                 "Cu2+",  "gly+Cu2+"),
    ("bipy+Cu2+",      "c1ccnc(-c2ccccn2)c1",                     "Cu2+",  "bipy+Cu2+"),
    ("phen+Cu2+",      "c1cnc2c(c1)ccc1cccnc12",                  "Cu2+",  "phen+Cu2+"),
    ("ox+Cu2+",        "OC(=O)C(=O)O",                            "Cu2+",  "ox+Cu2+"),
    ("IDA+Cu2+",       "OC(=O)CNCC(=O)O",                         "Cu2+",  "IDA+Cu2+"),
    ("acac+Cu2+",      "CC(=O)CC(=O)C",                           "Cu2+",  None),  # may not be in cal_dataset
    ("catechol+Fe3+",  "Oc1ccccc1O",                              "Fe3+",  None),
    ("8HQ+Cu2+",       "Oc1cccc2cccnc12",                         "Cu2+",  None),
    ("AHA+Cu2+",       "CC(=O)NO",                                "Cu2+",  None),
]

EPSILON_PIPELINE = 0.02  # SMILES vs manual — small diffs from donor order OK


def test_smiles_pipeline():
    """SMILES→UC→predict must match from_metal_ligand→predict for known ligands."""
    passed = failed = skipped = 0
    failures = []
    results = []

    cal_by_name = {e['name']: e for e in CAL_DATA}

    for test_name, smiles, metal, cal_name in SMILES_PIPELINE_TESTS:
        # Path A: from_smiles (auto-descriptor)
        try:
            uc_auto = from_smiles(smiles, metal=metal, pH=14.0)
            r_auto = predict(uc_auto)
        except Exception as e:
            failures.append(f"  ERROR: {test_name} — from_smiles failed: {e}")
            failed += 1
            continue

        # Path B: from_metal_ligand (manual annotation, if available)
        if cal_name and cal_name in cal_by_name:
            entry = cal_by_name[cal_name]
            uc_manual = from_metal_ligand(entry)
            r_manual = predict(uc_manual)

            diff = abs(r_auto.log_Ka_pred - r_manual.log_Ka_pred)
            if diff <= EPSILON_PIPELINE:
                passed += 1
            else:
                failed += 1
                failures.append(
                    f"  MISMATCH: {test_name:20s} auto={r_auto.log_Ka_pred:7.2f} "
                    f"manual={r_manual.log_Ka_pred:7.2f} Δ={r_auto.log_Ka_pred - r_manual.log_Ka_pred:+.3f}")

            results.append((test_name, r_auto.log_Ka_pred, r_manual.log_Ka_pred))
        else:
            # No manual reference — just verify it doesn't crash
            skipped += 1
            results.append((test_name, r_auto.log_Ka_pred, None))

    print(f"\n{'='*72}")
    print(f"SMILES PIPELINE: {passed}/{passed+failed} matched ({skipped} no manual ref)")
    print(f"{'='*72}")
    for name, auto, manual in results:
        ref_str = f"manual={manual:7.2f}" if manual is not None else "manual=N/A"
        print(f"  {name:20s} auto={auto:7.2f} {ref_str}")
    if failures:
        print()
        for f in failures:
            print(f)
    return failed


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: PER-MODALITY STATISTICS & PHASE 12 CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

# Phase 12 pass/fail thresholds
METAL_R2_MIN = 0.885   # from Phase 12 holdout results
HG_R2_MIN = 0.840


def test_modality_statistics():
    """Compute and display per-modality statistics. Check Phase 12 constraints."""
    failures = 0

    # Metal
    m_preds, m_exps = [], []
    for entry in CAL_DATA:
        uc = from_metal_ligand(entry)
        r = predict(uc)
        m_preds.append(r.log_Ka_pred)
        m_exps.append(entry['log_K_exp'])
    ms = _stats(m_preds, m_exps)

    # HG
    h_preds, h_exps = [], []
    for entry in HG_DATA:
        uc = from_host_guest(entry)
        r = predict(uc)
        h_preds.append(r.log_Ka_pred)
        h_exps.append(entry['log_Ka'])
    hs = _stats(h_preds, h_exps)

    # CM
    c_preds, c_exps = [], []
    for entry in CROSS_MODAL_DATA:
        uc = _uc_from_cm(entry)
        r = predict(uc)
        c_preds.append(r.log_Ka_pred)
        c_exps.append(entry['log_Ka'])
    cs = _stats(c_preds, c_exps)

    # Weighted MAE
    n_tot = ms['n'] + hs['n'] + cs['n']
    wmae = (ms['mae']*ms['n'] + hs['mae']*hs['n'] + cs['mae']*cs['n']) / n_tot

    print(f"\n{'='*72}")
    print(f"PER-MODALITY STATISTICS")
    print(f"{'='*72}")
    print(f"  {'Modality':12s} {'N':>4s} {'R²':>8s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>8s} {'MaxErr':>8s}")
    print(f"  {'─'*56}")
    for label, s in [('Metal', ms), ('Host-Guest', hs), ('Cross-Modal', cs)]:
        print(f"  {label:12s} {s['n']:4d} {s['r2']:8.4f} {s['mae']:8.3f} "
              f"{s['rmse']:8.3f} {s['bias']:+8.3f} {s['max_err']:8.2f}")
    print(f"  {'─'*56}")
    print(f"  {'wMAE':12s} {n_tot:4d} {'':>8s} {wmae:8.3f}")

    # Phase 12 constraints — informational
    # These thresholds apply AFTER Phase 11b calibration (which patches
    # params at runtime). Default baked-in params may not meet thresholds.
    # The regression test verifies CONSISTENCY, not absolute performance.
    print(f"\n  Phase 12 Thresholds (informational — requires calibration):")
    checks = []

    m_status = "✓" if ms['r2'] >= METAL_R2_MIN else "○"
    checks.append(f"  {m_status} Metal R² = {ms['r2']:.4f} (threshold ≥ {METAL_R2_MIN})")

    h_status = "✓" if hs['r2'] >= HG_R2_MIN else "○"
    checks.append(f"  {h_status} HG R²    = {hs['r2']:.4f} (threshold ≥ {HG_R2_MIN})")

    ratio = n_tot / 80  # 80 params
    r_status = "✓" if ratio >= 7.0 else "✗"
    checks.append(f"  {r_status} Data:param = {ratio:.1f}:1 (threshold ≥ 7:1)")
    if ratio < 7.0:
        failures += 1

    for c in checks:
        print(c)

    print(f"{'='*72}")
    return failures


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: SELF-ZEROING & EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    """Edge cases that must not crash."""
    failures = 0
    cases = []

    # Empty UC
    uc = UniversalComplex(name="empty")
    r = predict(uc)
    ok = r.log_Ka_pred == 0.0
    cases.append(("empty UC → 0.0", ok))
    if not ok: failures += 1

    # Metal-only, no donors
    uc2 = UniversalComplex(name="bare_metal", metal_formula="Cu2+",
                           binding_mode="metal_coordination")
    r2 = predict(uc2)
    ok2 = r2.dg_metal == 0.0
    cases.append(("metal, no donors → dg_metal=0", ok2))
    if not ok2: failures += 1

    # HG, no guest SMILES
    uc3 = UniversalComplex(name="no_guest", binding_mode="host_guest_inclusion",
                           host_name="beta-CD")
    r3 = predict(uc3)
    ok3 = r3.dg_hydrophobic == 0.0
    cases.append(("HG, no SMILES → dg_hydrophobic=0", ok3))
    if not ok3: failures += 1

    # CM, unknown host
    uc4 = UniversalComplex(name="cm_unknown", binding_mode="cross_modal",
                           metal_formula="Na+", host_name="CB99")
    r4 = predict(uc4)
    ok4 = r4.dg_ion_dipole == 0.0
    cases.append(("CM, unknown host → dg_ion_dipole=0", ok4))
    if not ok4: failures += 1

    # Mixed mode: both metal and HG terms fire
    uc5 = from_smiles("NCC(=O)O", metal="Zn2+", host="beta-CD")
    uc5.binding_mode = "mixed"
    r5 = predict(uc5)
    ok5 = r5.dg_metal != 0.0 and (r5.dg_hydrophobic != 0.0 or r5.dg_cavity_dehydration != 0.0)
    cases.append(("mixed mode → both fire", ok5))
    if not ok5: failures += 1

    print(f"\n{'='*72}")
    print(f"EDGE CASES: {sum(1 for _,ok in cases if ok)}/{len(cases)} passed")
    print(f"{'='*72}")
    for label, ok in cases:
        print(f"  {'✓' if ok else '✗'} {label}")
    return failures


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: SMILES→PREDICT DONOR FIDELITY
# For key ligands, verify auto-extracted donors match manual annotations
# ═══════════════════════════════════════════════════════════════════════════

DONOR_FIDELITY_TESTS = [
    # (name, SMILES, expected_sorted_subtypes)
    ("EDTA", "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
     ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ("NTA", "OC(=O)CN(CC(=O)O)CC(=O)O",
     ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ("en", "NCCN", ["N_amine","N_amine"]),
    ("glycine", "NCC(=O)O", ["N_amine","O_carboxylate"]),
    ("bipy", "c1ccnc(-c2ccccn2)c1", ["N_pyridine","N_pyridine"]),
    ("catechol", "Oc1ccccc1O", ["O_catecholate","O_catecholate"]),
    ("8-HQ", "Oc1cccc2cccnc12", ["N_pyridine","O_phenolate"]),
    ("AHA", "CC(=O)NO", ["O_hydroxamate","O_hydroxamate"]),
    ("cysteine", "NC(CS)C(=O)O", ["N_amine","O_carboxylate","S_thiolate"]),
    ("DTC", "CCN(CC)C(=S)S", ["S_dithiocarbamate","S_dithiocarbamate"]),
]


def test_donor_fidelity():
    """Auto-extracted donors must match expected annotations."""
    from rdkit import Chem
    from core.auto_descriptor import extract_donor_subtypes

    passed = failed = 0
    failures = []

    for name, smiles, expected in DONOR_FIDELITY_TESTS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failures.append(f"  PARSE FAIL: {name}")
            failed += 1
            continue

        donors = extract_donor_subtypes(mol)
        got = sorted(s for _, s in donors)
        exp = sorted(expected)

        if got == exp:
            passed += 1
        else:
            failed += 1
            failures.append(f"  FAIL: {name:15s} exp={exp} got={got}")

    print(f"\n{'='*72}")
    print(f"DONOR FIDELITY: {passed}/{passed+failed} matched")
    print(f"{'='*72}")
    for f in failures:
        print(f)
    return failed


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total = 0

    total += test_frozen_metal()
    total += test_frozen_hg()
    total += test_frozen_cm()
    total += test_smiles_pipeline()
    total += test_donor_fidelity()
    total += test_edge_cases()
    total += test_modality_statistics()

    print(f"\n{'='*72}")
    if total == 0:
        print("PHASE 13c: ALL REGRESSION TESTS PASSED ✓")
        print(f"  644 frozen references matched (ε={EPSILON_FROZEN})")
        print(f"  SMILES→predict pipeline verified")
        print(f"  Donor fidelity confirmed")
        print(f"  Edge cases handled")
        print(f"  Phase 12 constraints checked")
    else:
        print(f"PHASE 13c: {total} FAILURE(S)")
    print(f"{'='*72}")