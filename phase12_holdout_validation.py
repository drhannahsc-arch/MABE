"""
phase12_holdout_validation.py — Series-level holdout validation.

Holdout sets (entire series, not random):
  Metal:  Ag+ (all 9) + DTPA (all 20) = 29 entries
  HG:     γ-CD (all 9)                 =  9 entries
  CM:     CB8  (all 17)                = 17 entries
  Total holdout: 55 entries (~8.5% of 644)

Protocol:
  1. Remove holdout from training data
  2. Re-optimize on training-only (589 entries)
  3. Score holdout with frozen trained params
  4. Report train vs holdout metrics

Pass criteria (holdout):
  Metal holdout R² > 0.70  (lower bar — series extrapolation)
  Metal holdout MAE < 2.50
  HG holdout MAE < 1.20
  CM holdout MAE < 1.00
"""

import sys
sys.path.insert(0, 'knowledge')
sys.path.insert(0, 'core')
sys.path.insert(0, '.')

import numpy as np
import time
from collections import defaultdict
from scipy.optimize import least_squares

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS  (same as Phase 11b)
# ═══════════════════════════════════════════════════════════════════════════

from knowledge.cal_dataset import CAL_DATA
from scorer_frozen import predict_log_k  # must match cal_params' import scorer_frozen
from cal_params import (
    PARAM_SPEC as METAL_PARAM_SPEC,
    apply_params as apply_metal_params,
)

from knowledge.hg_dataset import HG_DATA
from hg_scorer import predict_hg_log_ka, HG_PARAMS, HBOND_PARAMS, PI_PARAMS
from hg_conf_shape import CONF_SHAPE_PARAMS

from cross_modal_dataset import CROSS_MODAL_DATA
from cross_modal_predictor import (
    predict_cross_modal_log_ka, CM_PARAMS, CM_PARAM_SPEC,
)

# ═══════════════════════════════════════════════════════════════════════════
# PARAM SPECS  (copied from Phase 11b)
# ═══════════════════════════════════════════════════════════════════════════

HG_PARAM_SPEC = [
    ("gamma_flat",       "HG", "gamma_flat",       0.025,  0.010,  0.045),
    ("k_curvature",      "HG", "k_curvature",      1.15,   1.00,   2.50),
    ("dg_dehydr_per_A2", "HG", "dg_dehydr_per_A2",-0.070, -0.150,  0.0),
    ("dehydr_CB",        "HG", "dehydr_CB",         3.67,   0.5,    6.0),
    ("dehydr_CD",        "HG", "dehydr_CD",         0.64,   0.0,    2.0),
    ("dehydr_other",     "HG", "dehydr_other",      1.74,   0.0,    3.0),
    ("k_size_penalty",   "HG", "k_size_penalty",    0.031,  0.0,    5.0),
    ("k_undersize",      "HG", "k_undersize",       0.0,    0.0,    0.15),
    ("eps_neutral",      "HB", "eps_neutral",       -3.0, -12.0,    0.0),
    ("eps_charge_asst",  "HB", "eps_charge_assisted",-10.0,-25.0,   -2.0),
    ("eps_oh_pi",        "HB", "eps_oh_pi",         -1.5,  -6.0,    0.0),
    ("water_penalty",    "HB", "water_penalty_per_hb", 3.5, 0.0,   12.0),
    ("water_displace",   "HB", "water_displacement",  1.2,  0.5,    3.0),
    ("eps_ch_pi",        "PI", "eps_ch_pi",         -1.5,  -5.0,    0.0),
    ("eps_pi_stack",     "PI", "eps_pi_stack",      -4.0, -12.0,    0.0),
    ("eps_cation_pi",    "PI", "eps_cation_pi",     -5.0, -15.0,    0.0),
    ("eps_rotor",        "CS", "eps_rotor",          2.5,   0.5,    6.0),
    ("f_partial",        "CS", "f_partial",          0.5,   0.1,    0.9),
    ("k_shape",          "CS", "k_shape",           -8.0, -20.0,    0.0),
    ("PC_optimal",       "CS", "PC_optimal",         0.55,  0.35,   0.70),
    ("sigma_PC",         "CS", "sigma_PC",           0.15,  0.05,   0.35),
]

N_METAL = len(METAL_PARAM_SPEC)
N_HG = len(HG_PARAM_SPEC)
N_CM = len(CM_PARAM_SPEC)
N_TOTAL = N_METAL + N_HG + N_CM


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def apply_hg_params(x_hg):
    for i, (_, dn, key, *_) in enumerate(HG_PARAM_SPEC):
        if dn == "HG":   HG_PARAMS[key] = x_hg[i]
        elif dn == "HB": HBOND_PARAMS[key] = x_hg[i]
        elif dn == "PI": PI_PARAMS[key] = x_hg[i]
        elif dn == "CS": CONF_SHAPE_PARAMS[key] = x_hg[i]

def apply_cm_params(x_cm):
    for i, (_, _, key, *_) in enumerate(CM_PARAM_SPEC):
        CM_PARAMS[key] = x_cm[i]

def apply_all(x):
    apply_metal_params(x[:N_METAL])
    apply_hg_params(x[N_METAL:N_METAL + N_HG])
    apply_cm_params(x[N_METAL + N_HG:])


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTORS
# ═══════════════════════════════════════════════════════════════════════════

def metal_predict(e):
    try:
        return predict_log_k(
            e["metal"], e["donors"], chelate_rings=e["chelate_rings"],
            ring_sizes=e["ring_sizes"] or None, pH=e["pH"],
            is_macrocyclic=e["macrocyclic"], cavity_radius_nm=e["cavity_nm"],
            n_ligand_molecules=e["n_lig_mol"])
    except: return 0.0

def hg_predict(e):
    try: return predict_hg_log_ka(e)
    except: return 0.0

def cm_predict(e):
    try: return predict_cross_modal_log_ka(e)
    except: return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def stats(preds, exps, label=""):
    p, e = np.array(preds), np.array(exps)
    if len(p) == 0:
        return {"r2": 0, "mae": 0, "rmse": 0, "bias": 0, "n": 0, "max_err": 0}
    r = p - e
    ss_tot = np.sum((e - e.mean()) ** 2)
    r2 = 1.0 - np.sum(r ** 2) / ss_tot if ss_tot > 1e-10 else 0.0
    return {
        "r2": r2, "mae": np.mean(np.abs(r)), "rmse": np.sqrt(np.mean(r**2)),
        "bias": np.mean(r), "n": len(p), "max_err": np.max(np.abs(r)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HOLDOUT SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def split_metal():
    """Hold out: all Ag+ entries + all DTPA entries."""
    train, holdout = [], []
    for e in CAL_DATA:
        if e["metal"] == "Ag+" or "DTPA" in e["name"]:
            holdout.append(e)
        else:
            train.append(e)
    return train, holdout


def split_hg():
    """Hold out: all γ-CD entries."""
    train, holdout = [], []
    for e in HG_DATA:
        if e["host"] == "gamma-CD":
            holdout.append(e)
        else:
            train.append(e)
    return train, holdout


def split_cm():
    """Hold out: all CB8 entries."""
    train, holdout = [], []
    for e in CROSS_MODAL_DATA:
        if e["cb_host"] == "CB8":
            holdout.append(e)
        else:
            train.append(e)
    return train, holdout


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION ON TRAINING ONLY
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_metal(train_data):
    x0 = [p[3] for p in METAL_PARAM_SPEC]
    lo = [p[4] for p in METAL_PARAM_SPEC]
    hi = [p[5] for p in METAL_PARAM_SPEC]
    def res(x):
        apply_metal_params(x)
        return [metal_predict(e) - e["log_K_exp"] for e in train_data]
    result = least_squares(res, x0, bounds=(lo, hi), method="trf",
                           max_nfev=10000, xtol=1e-12, ftol=1e-12)
    apply_metal_params(result.x)
    print(f"    Metal opt: nfev={result.nfev} cost={result.cost:.2f}")
    return result.x


def calibrate_hg(train_data):
    x0 = [p[3] for p in HG_PARAM_SPEC]
    lo = [p[4] for p in HG_PARAM_SPEC]
    hi = [p[5] for p in HG_PARAM_SPEC]
    def res(x):
        apply_hg_params(x)
        return [hg_predict(e) - e["log_Ka"] for e in train_data]
    result = least_squares(res, x0, bounds=(lo, hi), method="trf", max_nfev=5000)
    apply_hg_params(result.x)
    print(f"    HG opt:    nfev={result.nfev} cost={result.cost:.2f}")
    return result.x


def calibrate_cm(train_data, x_hg):
    apply_hg_params(x_hg)
    x0 = [p[3] for p in CM_PARAM_SPEC]
    lo = [p[4] for p in CM_PARAM_SPEC]
    hi = [p[5] for p in CM_PARAM_SPEC]
    def res(x):
        apply_cm_params(x)
        return [cm_predict(e) - e["log_Ka"] for e in train_data]
    result = least_squares(res, x0, bounds=(lo, hi), method="trf", max_nfev=5000)
    apply_cm_params(result.x)
    print(f"    CM opt:    nfev={result.nfev} cost={result.cost:.2f}")
    return result.x


def trimodal_residuals(x, x_anchor, train_m, train_h, train_c,
                       lam_m=0.3, lam_h=0.1, lam_c=0.05):
    apply_all(x)

    metal_res = [metal_predict(e) - e["log_K_exp"] for e in train_m]
    hg_res = [hg_predict(e) - e["log_Ka"] for e in train_h]
    cm_res = [cm_predict(e) - e["log_Ka"] for e in train_c]

    x_m = x[:N_METAL]
    x_h = x[N_METAL:N_METAL + N_HG]
    x_c = x[N_METAL + N_HG:]
    a_m = x_anchor[:N_METAL]
    a_h = x_anchor[N_METAL:N_METAL + N_HG]
    a_c = x_anchor[N_METAL + N_HG:]

    reg_m = [lam_m * (x_m[i] - a_m[i]) for i in range(N_METAL)]
    reg_h = [lam_h * (x_h[i] - a_h[i]) for i in range(N_HG)]
    reg_c = [lam_c * (x_c[i] - a_c[i]) for i in range(N_CM)]

    return metal_res + hg_res + cm_res + reg_m + reg_h + reg_c


def joint_refinement(x_m, x_h, x_c, train_m, train_h, train_c):
    x_anchor = np.concatenate([x_m, x_h, x_c])
    lo_m = [p[4] for p in METAL_PARAM_SPEC]
    hi_m = [p[5] for p in METAL_PARAM_SPEC]
    lo_h = [p[4] for p in HG_PARAM_SPEC]
    hi_h = [p[5] for p in HG_PARAM_SPEC]
    lo_c = [p[4] for p in CM_PARAM_SPEC]
    hi_c = [p[5] for p in CM_PARAM_SPEC]
    lo = np.array(lo_m + lo_h + lo_c)
    hi = np.array(hi_m + hi_h + hi_c)

    best_x = x_anchor.copy()
    best_wmae = 999.0
    n_tot = len(train_m) + len(train_h) + len(train_c)

    configs = [
        (0.3, 0.1, 0.02),
        (0.1, 0.05, 0.01),
        (0.05, 0.02, 0.005),
    ]

    for lam_m, lam_h, lam_c in configs:
        x0 = np.clip(x_anchor.copy(), lo + 1e-10, hi - 1e-10)
        result = least_squares(
            trimodal_residuals, x0,
            args=(x_anchor, train_m, train_h, train_c, lam_m, lam_h, lam_c),
            bounds=(lo, hi), method="trf",
            max_nfev=15000, xtol=1e-13, ftol=1e-13,
        )
        apply_all(result.x)
        ms = stats([metal_predict(e) for e in train_m],
                   [e["log_K_exp"] for e in train_m])
        hs = stats([hg_predict(e) for e in train_h],
                   [e["log_Ka"] for e in train_h])
        cs = stats([cm_predict(e) for e in train_c],
                   [e["log_Ka"] for e in train_c])
        wmae = (ms["mae"]*len(train_m) + hs["mae"]*len(train_h) +
                cs["mae"]*len(train_c)) / n_tot

        print(f"    λ=({lam_m},{lam_h},{lam_c}): M R²={ms['r2']:.4f} "
              f"H R²={hs['r2']:.4f} C R²={cs['r2']:.4f} wMAE={wmae:.2f}")

        if wmae < best_wmae:
            best_wmae = wmae
            best_x = result.x.copy()

    return best_x


# ═══════════════════════════════════════════════════════════════════════════
# PER-ENTRY HOLDOUT DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def metal_holdout_detail(holdout):
    print(f"\n  {'Name':25s} {'Exp':>7s} {'Pred':>7s} {'Err':>7s}")
    print("  " + "─" * 50)
    for e in sorted(holdout, key=lambda x: x["name"]):
        pred = metal_predict(e)
        err = pred - e["log_K_exp"]
        flag = " ← BIG" if abs(err) > 3.0 else ""
        print(f"  {e['name']:25s} {e['log_K_exp']:7.1f} {pred:7.1f} {err:+7.1f}{flag}")


def hg_holdout_detail(holdout):
    print(f"\n  {'Name':25s} {'Exp':>7s} {'Pred':>7s} {'Err':>7s}")
    print("  " + "─" * 50)
    for e in holdout:
        pred = hg_predict(e)
        err = pred - e["log_Ka"]
        flag = " ← BIG" if abs(err) > 1.5 else ""
        print(f"  {e['name']:25s} {e['log_Ka']:7.2f} {pred:7.2f} {err:+7.2f}{flag}")


def cm_holdout_detail(holdout):
    print(f"\n  {'Metal':8s} {'Exp':>7s} {'Pred':>7s} {'Err':>7s}")
    print("  " + "─" * 35)
    for e in sorted(holdout, key=lambda x: x["log_Ka"]):
        pred = cm_predict(e)
        err = pred - e["log_Ka"]
        flag = " ← BIG" if abs(err) > 1.0 else ""
        print(f"  {e['metal']:8s} {e['log_Ka']:7.2f} {pred:7.2f} {err:+7.2f}{flag}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("═" * 65)
    print("  PHASE 12: HOLDOUT VALIDATION")
    print("═" * 65)

    # ── Step 1: Split data ──
    print("\n  Step 1: Split datasets")
    train_m, hold_m = split_metal()
    train_h, hold_h = split_hg()
    train_c, hold_c = split_cm()

    n_train = len(train_m) + len(train_h) + len(train_c)
    n_hold = len(hold_m) + len(hold_h) + len(hold_c)

    print(f"    Metal: {len(train_m)} train + {len(hold_m)} holdout "
          f"(Ag+ {sum(1 for e in hold_m if e['metal']=='Ag+')} + "
          f"DTPA {sum(1 for e in hold_m if 'DTPA' in e['name'])})")
    print(f"    HG:    {len(train_h)} train + {len(hold_h)} holdout (γ-CD)")
    print(f"    CM:    {len(train_c)} train + {len(hold_c)} holdout (CB8)")
    print(f"    Total: {n_train} train + {n_hold} holdout ({n_hold/(n_train+n_hold)*100:.1f}%)")
    print(f"    Data:param ratio (train): {n_train/N_TOTAL:.1f}:1")

    # ── Step 2: Separate calibrations on TRAINING ONLY ──
    print(f"\n  Step 2: Calibrate on training only ({n_train} entries)")
    print("  2a: Metal...")
    x_m = calibrate_metal(train_m)
    print("  2b: HG...")
    x_h = calibrate_hg(train_h)
    print("  2c: CM...")
    x_c = calibrate_cm(train_c, x_h)

    # ── Step 3: Joint refinement on TRAINING ONLY ──
    print(f"\n  Step 3: Tri-modal joint refinement (training only)")
    x_joint = joint_refinement(x_m, x_h, x_c, train_m, train_h, train_c)

    # ── Step 4: Score EVERYTHING with frozen trained params ──
    print("\n" + "═" * 65)
    print("  Step 4: Score with frozen trained params")
    print("═" * 65)

    apply_all(x_joint)

    # Training metrics
    tr_m = stats([metal_predict(e) for e in train_m],
                 [e["log_K_exp"] for e in train_m])
    tr_h = stats([hg_predict(e) for e in train_h],
                 [e["log_Ka"] for e in train_h])
    tr_c = stats([cm_predict(e) for e in train_c],
                 [e["log_Ka"] for e in train_c])

    # Holdout metrics
    ho_m = stats([metal_predict(e) for e in hold_m],
                 [e["log_K_exp"] for e in hold_m])
    ho_h = stats([hg_predict(e) for e in hold_h],
                 [e["log_Ka"] for e in hold_h])
    ho_c = stats([cm_predict(e) for e in hold_c],
                 [e["log_Ka"] for e in hold_c])

    # ── Print results ──
    print(f"\n  ┌─── TRAINING SET PERFORMANCE ─────────────────────────┐")
    print(f"  │  Metal: R²={tr_m['r2']:.4f}  MAE={tr_m['mae']:.2f}  "
          f"bias={tr_m['bias']:+.2f}  n={tr_m['n']}    │")
    print(f"  │  HG:    R²={tr_h['r2']:.4f}  MAE={tr_h['mae']:.2f}  "
          f"bias={tr_h['bias']:+.2f}  n={tr_h['n']}     │")
    print(f"  │  CM:    R²={tr_c['r2']:.4f}  MAE={tr_c['mae']:.2f}  "
          f"bias={tr_c['bias']:+.2f}  n={tr_c['n']}     │")
    print(f"  └────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─── HOLDOUT SET PERFORMANCE ──────────────────────────┐")
    print(f"  │  Metal: R²={ho_m['r2']:.4f}  MAE={ho_m['mae']:.2f}  "
          f"bias={ho_m['bias']:+.2f}  max={ho_m['max_err']:.1f}  n={ho_m['n']}│")
    print(f"  │  HG:    R²={ho_h['r2']:.4f}  MAE={ho_h['mae']:.2f}  "
          f"bias={ho_h['bias']:+.2f}  max={ho_h['max_err']:.1f}   n={ho_h['n']} │")
    print(f"  │  CM:    R²={ho_c['r2']:.4f}  MAE={ho_c['mae']:.2f}  "
          f"bias={ho_c['bias']:+.2f}  max={ho_c['max_err']:.1f}   n={ho_c['n']}│")
    print(f"  └────────────────────────────────────────────────────────┘")

    # ── Overfitting gap ──
    print(f"\n  ┌─── OVERFITTING DIAGNOSTIC ─────────────────────────┐")
    for label, tr, ho in [("Metal", tr_m, ho_m), ("HG", tr_h, ho_h), ("CM", tr_c, ho_c)]:
        gap_r2 = tr["r2"] - ho["r2"]
        gap_mae = ho["mae"] - tr["mae"]
        overfit = "⚠ OVERFIT" if gap_mae > 1.0 else "OK"
        print(f"  │  {label:6s} ΔR²={gap_r2:+.4f}  ΔMAE={gap_mae:+.2f}  {overfit:10s}│")
    print(f"  └────────────────────────────────────────────────────────┘")

    # ── Per-entry holdout detail ──
    print("\n" + "═" * 65)
    print("  METAL HOLDOUT DETAIL")
    print("═" * 65)

    # Split Ag+ and DTPA for separate analysis
    ag_entries = [e for e in hold_m if e["metal"] == "Ag+"]
    dtpa_entries = [e for e in hold_m if "DTPA" in e["name"]]
    ag_stats = stats([metal_predict(e) for e in ag_entries],
                     [e["log_K_exp"] for e in ag_entries])
    dtpa_stats = stats([metal_predict(e) for e in dtpa_entries],
                       [e["log_K_exp"] for e in dtpa_entries])

    print(f"\n  Ag+ subseries (n={ag_stats['n']}):  "
          f"R²={ag_stats['r2']:.4f}  MAE={ag_stats['mae']:.2f}  bias={ag_stats['bias']:+.2f}")
    metal_holdout_detail(ag_entries)

    print(f"\n  DTPA subseries (n={dtpa_stats['n']}): "
          f"R²={dtpa_stats['r2']:.4f}  MAE={dtpa_stats['mae']:.2f}  bias={dtpa_stats['bias']:+.2f}")
    metal_holdout_detail(dtpa_entries)

    print("\n" + "═" * 65)
    print("  HG HOLDOUT DETAIL (γ-CD)")
    print("═" * 65)
    hg_holdout_detail(hold_h)

    print("\n" + "═" * 65)
    print("  CM HOLDOUT DETAIL (CB8)")
    print("═" * 65)
    cm_holdout_detail(hold_c)

    # ── Pass/Fail ──
    print("\n" + "═" * 65)
    print("  PHASE 12 PASS/FAIL")
    print("═" * 65)

    checks = [
        ("Metal train R² ≥ 0.885",  tr_m["r2"] >= 0.885),
        ("Metal holdout MAE ≤ 2.50", ho_m["mae"] <= 2.50),
        ("Metal holdout R² > 0.70",  ho_m["r2"] > 0.70),
        ("HG train R² ≥ 0.840",     tr_h["r2"] >= 0.840),
        ("HG holdout MAE ≤ 1.20",   ho_h["mae"] <= 1.20),
        ("CM holdout MAE ≤ 1.00",   ho_c["mae"] <= 1.00),
        ("Metal overfit gap < 1.0",  ho_m["mae"] - tr_m["mae"] < 1.0),
        ("HG overfit gap < 0.5",     ho_h["mae"] - tr_h["mae"] < 0.5),
    ]

    all_pass = True
    for label, passed in checks:
        mark = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  {mark}  {label}")

    print(f"\n  {'═' * 50}")
    if all_pass:
        print(f"  ✓ PHASE 12 PASSED — Model generalizes to held-out series")
    else:
        print(f"  ✗ PHASE 12 HAS FAILURES — See diagnostics above")
    print(f"  {'═' * 50}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.0f}s")

    return {
        "train": {"metal": tr_m, "hg": tr_h, "cm": tr_c},
        "holdout": {"metal": ho_m, "hg": ho_h, "cm": ho_c},
        "holdout_sub": {"ag": ag_stats, "dtpa": dtpa_stats},
        "x_joint": x_joint,
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    result = main()