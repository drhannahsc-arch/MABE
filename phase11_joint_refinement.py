"""
phase11_joint_refinement.py — MABE Phase 11: Universal Joint Refinement

Merges metal coordination (500 entries, 55 params) and host-guest inclusion
(80 entries, 21 params) into a single bounded least-squares optimization with
dual regularization.

Strategy:
  1. Run metal-only calibration → x_metal_anchor (Phase 5 result)
  2. Run HG-only calibration    → x_hg_anchor    (Phase 10 result)
  3. Joint optimization with regularization toward anchors
  4. Hard constraint: metal R² ≥ 0.885 (within 0.01 of Phase 5)
  5. Report per-modality + combined statistics

Usage:
    python phase11_joint_refinement.py
"""

import sys
import math
import time
import numpy as np
from scipy.optimize import least_squares
from collections import defaultdict

sys.path.insert(0, 'knowledge')
sys.path.insert(0, 'core')
sys.path.insert(0, '.')

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS — Metal scorer
# ═══════════════════════════════════════════════════════════════════════════
from cal_params import PARAM_SPEC as METAL_PARAM_SPEC, apply_params as apply_metal_params
from cal_dataset import CAL_DATA
from scorer_frozen import predict_log_k

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS — Host-guest scorer
# ═══════════════════════════════════════════════════════════════════════════
from hg_dataset import HG_DATA, HOST_DB
from hg_scorer import predict_hg_log_ka, HG_PARAMS, compute_guest_sasa
from hg_hbond import HBOND_PARAMS
from hg_pi import PI_PARAMS
from hg_conf_shape import CONF_SHAPE_PARAMS

# ═══════════════════════════════════════════════════════════════════════════
# HG PARAMETER SPECIFICATION (from hg_calibrate)
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
N_TOTAL = N_METAL + N_HG


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def apply_hg_params(x_hg):
    """Patch HG module dicts from flat vector."""
    for i, (_, dn, key, *_) in enumerate(HG_PARAM_SPEC):
        if dn == "HG":
            HG_PARAMS[key] = x_hg[i]
        elif dn == "HB":
            HBOND_PARAMS[key] = x_hg[i]
        elif dn == "PI":
            PI_PARAMS[key] = x_hg[i]
        elif dn == "CS":
            CONF_SHAPE_PARAMS[key] = x_hg[i]


def apply_all(x):
    """Apply combined parameter vector: [metal | hg]."""
    apply_metal_params(x[:N_METAL])
    apply_hg_params(x[N_METAL:])


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def metal_predict(e):
    try:
        return predict_log_k(
            e["metal"], e["donors"],
            chelate_rings=e["chelate_rings"],
            ring_sizes=e["ring_sizes"] or None,
            pH=e["pH"],
            is_macrocyclic=e["macrocyclic"],
            cavity_radius_nm=e["cavity_nm"],
            n_ligand_molecules=e["n_lig_mol"],
        )
    except Exception:
        return 0.0


def hg_predict(e):
    try:
        return predict_hg_log_ka(e)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(preds, exps):
    """R², MAE, RMSE, bias from arrays."""
    p, e = np.array(preds), np.array(exps)
    r = p - e
    ss_res = np.sum(r ** 2)
    ss_tot = np.sum((e - e.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(r))
    rmse = np.sqrt(np.mean(r ** 2))
    bias = np.mean(r)
    return {"r2": r2, "mae": mae, "rmse": rmse, "bias": bias, "n": len(p)}


def eval_metal():
    preds = [metal_predict(e) for e in CAL_DATA]
    exps = [e["log_K_exp"] for e in CAL_DATA]
    return compute_stats(preds, exps)


def eval_hg():
    preds = [hg_predict(e) for e in HG_DATA]
    exps = [e["log_Ka"] for e in HG_DATA]
    return compute_stats(preds, exps)


def eval_per_host():
    """Per-host-family MAE for HG."""
    host_errors = defaultdict(list)
    for e in HG_DATA:
        pred = hg_predict(e)
        host_errors[e["host"]].append(pred - e["log_Ka"])
    result = {}
    for host, errs in sorted(host_errors.items()):
        errs = np.array(errs)
        result[host] = {
            "n": len(errs),
            "mae": np.mean(np.abs(errs)),
            "bias": np.mean(errs),
        }
    return result


def print_stats(label, metal_s, hg_s):
    """Print formatted statistics block."""
    combined_r2 = 1.0 - (
        (metal_s["rmse"] ** 2 * metal_s["n"] + hg_s["rmse"] ** 2 * hg_s["n"])
        / (metal_s["n"] + hg_s["n"])
    )  # approximate
    print(f"\n  ┌─── {label} ─────────────────────────────────────┐")
    print(f"  │  Metal:  R²={metal_s['r2']:.4f}  MAE={metal_s['mae']:.2f}  "
          f"Bias={metal_s['bias']:+.2f}  n={metal_s['n']}  │")
    print(f"  │  HG:     R²={hg_s['r2']:.4f}  MAE={hg_s['mae']:.2f}  "
          f"Bias={hg_s['bias']:+.2f}  n={hg_s['n']}   │")
    print(f"  └──────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: SEPARATE CALIBRATIONS (establish anchors)
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_metal_only():
    """Phase 5 metal-only calibration. Returns optimized x vector."""
    x0 = [p[3] for p in METAL_PARAM_SPEC]
    lo = [p[4] for p in METAL_PARAM_SPEC]
    hi = [p[5] for p in METAL_PARAM_SPEC]

    def residuals(x):
        apply_metal_params(x)
        return [metal_predict(e) - e["log_K_exp"] for e in CAL_DATA]

    result = least_squares(residuals, x0, bounds=(lo, hi), method="trf",
                           max_nfev=10000, xtol=1e-12, ftol=1e-12)
    apply_metal_params(result.x)
    return result.x


def calibrate_hg_only():
    """Phase 10 HG-only calibration. Returns optimized x vector."""
    x0 = [p[3] for p in HG_PARAM_SPEC]
    lo = [p[4] for p in HG_PARAM_SPEC]
    hi = [p[5] for p in HG_PARAM_SPEC]

    def residuals(x):
        apply_hg_params(x)
        return [hg_predict(e) - e["log_Ka"] for e in HG_DATA]

    result = least_squares(residuals, x0, bounds=(lo, hi), method="trf",
                           max_nfev=5000)
    apply_hg_params(result.x)
    return result.x


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 11: JOINT REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════

def joint_residuals(x, x_anchor, lambda_metal, lambda_hg):
    """Combined residual vector: data fit + regularization.

    Returns:
        [metal_residuals | hg_residuals | metal_reg | hg_reg]

    The regularization terms penalize deviation from anchor values,
    preventing the optimizer from destroying modality-specific fits.
    """
    apply_all(x)

    # Data residuals
    metal_res = [metal_predict(e) - e["log_K_exp"] for e in CAL_DATA]
    hg_res = [hg_predict(e) - e["log_Ka"] for e in HG_DATA]

    # Regularization toward anchor (Tikhonov)
    x_m = x[:N_METAL]
    x_h = x[N_METAL:]
    anchor_m = x_anchor[:N_METAL]
    anchor_h = x_anchor[N_METAL:]

    metal_reg = [lambda_metal * (x_m[i] - anchor_m[i]) for i in range(N_METAL)]
    hg_reg = [lambda_hg * (x_h[i] - anchor_h[i]) for i in range(N_HG)]

    return metal_res + hg_res + metal_reg + hg_reg


def run_joint(x_metal_anchor, x_hg_anchor, lambda_metal=0.5, lambda_hg=0.3,
              max_nfev=15000):
    """Run joint optimization from anchor starting point."""
    x0 = np.concatenate([x_metal_anchor, x_hg_anchor])
    x_anchor = x0.copy()

    lo_m = [p[4] for p in METAL_PARAM_SPEC]
    hi_m = [p[5] for p in METAL_PARAM_SPEC]
    lo_h = [p[4] for p in HG_PARAM_SPEC]
    hi_h = [p[5] for p in HG_PARAM_SPEC]
    lo = np.array(lo_m + lo_h)
    hi = np.array(hi_m + hi_h)

    result = least_squares(
        joint_residuals, x0,
        args=(x_anchor, lambda_metal, lambda_hg),
        bounds=(lo, hi),
        method="trf",
        max_nfev=max_nfev,
        xtol=1e-13,
        ftol=1e-13,
    )

    apply_all(result.x)
    return result


def adaptive_joint(x_metal_anchor, x_hg_anchor, metal_r2_floor=0.885):
    """Run joint refinement with adaptive λ_metal to protect metal R².

    If metal R² drops below floor after joint optimization, increase
    λ_metal and re-run until the constraint is met.
    """
    lambda_metal = 0.3
    lambda_hg = 0.2
    best_x = None
    best_combined_mae = 999.0

    for attempt in range(6):
        print(f"\n  ── Joint attempt {attempt + 1}: "
              f"λ_metal={lambda_metal:.2f}, λ_hg={lambda_hg:.2f} ──")

        result = run_joint(x_metal_anchor, x_hg_anchor,
                           lambda_metal, lambda_hg)
        apply_all(result.x)

        ms = eval_metal()
        hs = eval_hg()
        combined_mae = (ms["mae"] * ms["n"] + hs["mae"] * hs["n"]) / (ms["n"] + hs["n"])

        print(f"    Metal: R²={ms['r2']:.4f} MAE={ms['mae']:.2f}")
        print(f"    HG:    R²={hs['r2']:.4f} MAE={hs['mae']:.2f}")
        print(f"    Combined MAE: {combined_mae:.2f}")
        print(f"    Converged: {result.success}, nfev={result.nfev}")

        if ms["r2"] >= metal_r2_floor:
            if combined_mae < best_combined_mae:
                best_combined_mae = combined_mae
                best_x = result.x.copy()
            # Try reducing lambda to see if we can improve HG further
            if attempt < 3:
                lambda_metal *= 0.7
                lambda_hg *= 0.7
            else:
                break
        else:
            print(f"    ⚠ Metal R² {ms['r2']:.4f} < {metal_r2_floor} — "
                  f"increasing λ_metal")
            lambda_metal *= 2.0
            # Restore anchor and retry
            if best_x is not None:
                # Start from best known good
                x_metal_anchor = best_x[:N_METAL].copy()
                x_hg_anchor = best_x[N_METAL:].copy()

    if best_x is None:
        print("  ⚠ Could not meet metal R² constraint. Using metal anchor + HG anchor.")
        best_x = np.concatenate([x_metal_anchor, x_hg_anchor])

    return best_x


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER DRIFT REPORT
# ═══════════════════════════════════════════════════════════════════════════

def drift_report(x_before, x_after, param_specs, label):
    """Show which parameters moved most from anchor."""
    drifts = []
    for i, (name, *_) in enumerate(param_specs):
        d = x_after[i] - x_before[i]
        pct = abs(d / x_before[i]) * 100 if abs(x_before[i]) > 1e-6 else 0.0
        drifts.append((name, x_before[i], x_after[i], d, pct))

    # Sort by absolute drift percentage
    drifts.sort(key=lambda t: -t[4])

    print(f"\n  {label} — Top parameter drifts from anchor:")
    print(f"  {'Param':25s} {'Anchor':>8s} {'Joint':>8s} {'Δ':>8s} {'%':>6s}")
    print("  " + "─" * 60)
    for name, before, after, delta, pct in drifts[:15]:
        flag = " ←" if pct > 20 else ""
        print(f"  {name:25s} {before:8.3f} {after:8.3f} {delta:+8.3f} {pct:5.1f}%{flag}")


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_unified_params(x):
    """Print all unified parameters as pasteable Python."""
    x_m = x[:N_METAL]
    x_h = x[N_METAL:]

    print("\n# ═══════════════════════════════════════════════════════════")
    print("# PHASE 11 UNIFIED PARAMETERS")
    print("# Metal: scorer_frozen.py   HG: hg_scorer.py + modules")
    print("# ═══════════════════════════════════════════════════════════\n")

    print("# ── Metal: SUBTYPE_EXCHANGE ──")
    print("SUBTYPE_EXCHANGE = {")
    for i, (name, target, key, *_) in enumerate(METAL_PARAM_SPEC):
        if target == "EXCHANGE":
            print(f'    "{key}": {x_m[i]:.3f},')
    print("}\n")

    print("# ── Metal: IRVING_WILLIAMS_BONUS ──")
    print("IRVING_WILLIAMS_BONUS = {")
    for i, (name, target, key, *_) in enumerate(METAL_PARAM_SPEC):
        if target == "IW":
            print(f'    "{key}": {x_m[i]:.2f},')
    print("}\n")

    print("# ── Metal: PARAMS ──")
    print("METAL_PARAMS = {")
    for i, (name, target, key, *_) in enumerate(METAL_PARAM_SPEC):
        if target == "PARAMS":
            print(f'    "{key}": {x_m[i]:.4f},')
    print("}\n")

    print("# ── Host-Guest: HG_PARAMS ──")
    print("HG_PARAMS = {")
    for i, (name, dn, key, *_) in enumerate(HG_PARAM_SPEC):
        if dn == "HG":
            print(f'    "{key}": {x_h[i]:.4f},')
    print("}\n")

    print("# ── Host-Guest: HBOND_PARAMS ──")
    print("HBOND_PARAMS = {")
    for i, (name, dn, key, *_) in enumerate(HG_PARAM_SPEC):
        if dn == "HB":
            print(f'    "{key}": {x_h[i]:.4f},')
    print("}\n")

    print("# ── Host-Guest: PI_PARAMS ──")
    print("PI_PARAMS = {")
    for i, (name, dn, key, *_) in enumerate(HG_PARAM_SPEC):
        if dn == "PI":
            print(f'    "{key}": {x_h[i]:.4f},')
    print("}\n")

    print("# ── Host-Guest: CONF_SHAPE_PARAMS ──")
    print("CONF_SHAPE_PARAMS = {")
    for i, (name, dn, key, *_) in enumerate(HG_PARAM_SPEC):
        if dn == "CS":
            print(f'    "{key}": {x_h[i]:.4f},')
    print("}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("═" * 65)
    print("  MABE Phase 11: Universal Joint Refinement")
    print("═" * 65)
    print(f"  Metal entries:  {len(CAL_DATA)}")
    print(f"  HG entries:     {len(HG_DATA)}")
    print(f"  Total entries:  {len(CAL_DATA) + len(HG_DATA)}")
    print(f"  Metal params:   {N_METAL}")
    print(f"  HG params:      {N_HG}")
    print(f"  Total params:   {N_TOTAL}")
    print(f"  Data:param ratio: {(len(CAL_DATA) + len(HG_DATA)) / N_TOTAL:.1f}:1")

    # ── Pre-warm SASA cache ──
    print("\n── Pre-computing guest SASA (one-time) ──")
    n_ok = 0
    for e in HG_DATA:
        try:
            compute_guest_sasa(e["guest_smiles"])
            n_ok += 1
        except Exception as ex:
            print(f"  ⚠ SASA failed: {e['name']}: {ex}")
    print(f"  {n_ok}/{len(HG_DATA)} guests computed")

    # ── Step 1: Metal-only calibration (Phase 5 anchor) ──
    print("\n── Step 1: Metal-only calibration (Phase 5 anchor) ──")
    x_metal_anchor = calibrate_metal_only()
    ms_anchor = eval_metal()
    print(f"  Metal anchor: R²={ms_anchor['r2']:.4f} MAE={ms_anchor['mae']:.2f}")

    # ── Step 2: HG-only calibration (Phase 10 anchor) ──
    print("\n── Step 2: HG-only calibration (Phase 10 anchor) ──")
    x_hg_anchor = calibrate_hg_only()
    hs_anchor = eval_hg()
    print(f"  HG anchor:    R²={hs_anchor['r2']:.4f} MAE={hs_anchor['mae']:.2f}")

    # ── Anchor baseline ──
    apply_metal_params(x_metal_anchor)
    apply_hg_params(x_hg_anchor)
    print_stats("ANCHOR (separate calibrations)", ms_anchor, hs_anchor)

    per_host_anchor = eval_per_host()
    print("\n  Per-host MAE (anchor):")
    for host, s in per_host_anchor.items():
        print(f"    {host:15s}  n={s['n']:2d}  MAE={s['mae']:.2f}  "
              f"Bias={s['bias']:+.2f}")

    # ── Step 3: Adaptive joint refinement ──
    print("\n" + "═" * 65)
    print("  Step 3: Adaptive Joint Refinement")
    print("═" * 65)

    x_joint = adaptive_joint(x_metal_anchor, x_hg_anchor,
                             metal_r2_floor=0.885)

    # ── Final evaluation ──
    apply_all(x_joint)
    ms_joint = eval_metal()
    hs_joint = eval_hg()
    print_stats("AFTER JOINT REFINEMENT", ms_joint, hs_joint)

    per_host_joint = eval_per_host()
    print("\n  Per-host MAE (joint):")
    for host, s in per_host_joint.items():
        a = per_host_anchor.get(host, {"mae": 0})
        delta = s["mae"] - a["mae"]
        arrow = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "="
        print(f"    {host:15s}  n={s['n']:2d}  MAE={s['mae']:.2f}  "
              f"Bias={s['bias']:+.2f}  Δ={delta:+.2f} {arrow}")

    # ── Drift analysis ──
    x_anchor_combined = np.concatenate([x_metal_anchor, x_hg_anchor])
    drift_report(x_metal_anchor, x_joint[:N_METAL],
                 METAL_PARAM_SPEC, "Metal params")
    drift_report(x_hg_anchor, x_joint[N_METAL:],
                 HG_PARAM_SPEC, "HG params")

    # ── Summary comparison ──
    print("\n" + "═" * 65)
    print("  PHASE 11 SUMMARY")
    print("═" * 65)
    print(f"\n  {'Metric':20s} {'Anchor':>10s} {'Joint':>10s} {'Δ':>8s}")
    print("  " + "─" * 52)
    print(f"  {'Metal R²':20s} {ms_anchor['r2']:10.4f} {ms_joint['r2']:10.4f} "
          f"{ms_joint['r2'] - ms_anchor['r2']:+8.4f}")
    print(f"  {'Metal MAE':20s} {ms_anchor['mae']:10.2f} {ms_joint['mae']:10.2f} "
          f"{ms_joint['mae'] - ms_anchor['mae']:+8.2f}")
    print(f"  {'HG R²':20s} {hs_anchor['r2']:10.4f} {hs_joint['r2']:10.4f} "
          f"{hs_joint['r2'] - hs_anchor['r2']:+8.4f}")
    print(f"  {'HG MAE':20s} {hs_anchor['mae']:10.2f} {hs_joint['mae']:10.2f} "
          f"{hs_joint['mae'] - hs_anchor['mae']:+8.2f}")

    # Combined weighted MAE
    anchor_cmae = (ms_anchor["mae"] * len(CAL_DATA) + hs_anchor["mae"] * len(HG_DATA)) / (len(CAL_DATA) + len(HG_DATA))
    joint_cmae = (ms_joint["mae"] * len(CAL_DATA) + hs_joint["mae"] * len(HG_DATA)) / (len(CAL_DATA) + len(HG_DATA))
    print(f"  {'Combined wMAE':20s} {anchor_cmae:10.2f} {joint_cmae:10.2f} "
          f"{joint_cmae - anchor_cmae:+8.2f}")

    # ── Constraint check ──
    print(f"\n  Metal R² constraint (≥0.885): "
          f"{'✓ PASS' if ms_joint['r2'] >= 0.885 else '✗ FAIL'}")
    print(f"  HG R² constraint (≥0.840):    "
          f"{'✓ PASS' if hs_joint['r2'] >= 0.840 else '✗ FAIL'}")

    # ── Export ──
    export_unified_params(x_joint)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.0f}s")

    return x_joint, ms_joint, hs_joint


if __name__ == "__main__":
    x_opt, ms, hs = main()