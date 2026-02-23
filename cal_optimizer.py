"""
cal_optimizer.py — Bounded least-squares calibration for MABE frozen scorer.

Fits scorer parameters against the calibration dataset using scipy.optimize.
Computes residuals, R², MAE, RMSE, and exports fitted parameters.

Usage:
    from cal_optimizer import run_calibration, evaluate
    x_opt, r2, mae = run_calibration()
    evaluate(x_opt, verbose=True)
"""

import math
from scipy.optimize import least_squares

from scorer_frozen import predict_log_k
from cal_dataset import CAL_DATA
from cal_params import (
    PARAM_SPEC, N_PARAMS, PARAM_NAMES,
    get_x0, get_bounds, apply_params, extract_params, print_params,
)


def _predict_one(entry: dict) -> float:
    """Predict log K for one calibration entry. Returns 0.0 on error."""
    try:
        return predict_log_k(
            entry["metal"],
            entry["donors"],
            chelate_rings=entry["chelate_rings"],
            ring_sizes=entry["ring_sizes"] or None,
            pH=entry["pH"],
            is_macrocyclic=entry["macrocyclic"],
            cavity_radius_nm=entry["cavity_nm"],
            n_ligand_molecules=entry["n_lig_mol"],
        )
    except Exception:
        return 0.0


def residuals(x):
    """Residual vector (pred - exp) for least-squares minimization."""
    apply_params(x)
    return [_predict_one(e) - e["log_K_exp"] for e in CAL_DATA]


def evaluate(x, verbose=False):
    """Compute R², MAE, RMSE and optionally print per-complex breakdown.

    Returns (r2, mae, rmse).
    """
    apply_params(x)
    preds, exps = [], []
    for e in CAL_DATA:
        preds.append(_predict_one(e))
        exps.append(e["log_K_exp"])

    n = len(preds)
    errors = [p - e for p, e in zip(preds, exps)]
    mae = sum(abs(e) for e in errors) / n
    mean_exp = sum(exps) / n
    ss_res = sum(e ** 2 for e in errors)
    ss_tot = sum((e - mean_exp) ** 2 for e in exps)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = (ss_res / n) ** 0.5

    if verbose:
        print(f"\n  {'Complex':35s} {'Exp':>6s} {'Pred':>6s} {'Err':>6s}  Src")
        print("  " + "─" * 70)
        for i, e in enumerate(CAL_DATA):
            flag = " ***" if abs(errors[i]) > 5.0 else ""
            print(f"  {e['name']:33s} {exps[i]:6.1f} {preds[i]:6.1f} "
                  f"{errors[i]:+6.1f}  {e['source']}{flag}")
        print("  " + "─" * 70)

    # Per-category breakdown
    categories = {}
    for i, e in enumerate(CAL_DATA):
        cat = e["name"].split("+")[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(errors[i])

    if verbose:
        print(f"\n  Per-ligand family:")
        print(f"  {'Family':15s} {'N':>3s} {'MAE':>6s} {'Bias':>6s}")
        for cat in sorted(categories, key=lambda c: -len(categories[c])):
            errs = categories[cat]
            cat_mae = sum(abs(e) for e in errs) / len(errs)
            cat_bias = sum(errs) / len(errs)
            print(f"  {cat:15s} {len(errs):3d} {cat_mae:6.2f} {cat_bias:+6.2f}")

    print(f"\n  N = {n} complexes, {N_PARAMS} parameters "
          f"({n / N_PARAMS:.1f}:1 ratio)")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.2f} log K")
    print(f"  RMSE = {rmse:.2f} log K")
    max_err_idx = max(range(n), key=lambda i: abs(errors[i]))
    print(f"  Max  = {abs(errors[max_err_idx]):.1f} "
          f"({CAL_DATA[max_err_idx]['name']})")

    return r2, mae, rmse


def export_params(x):
    """Print fitted parameters as pasteable Python code."""
    print("\n# ════════════════════════════════════════════════════")
    print("# FITTED PARAMETERS — paste into scorer_frozen.py")
    print("# ════════════════════════════════════════════════════\n")

    print("SUBTYPE_EXCHANGE = {")
    for i, (name, target, key, *_) in enumerate(PARAM_SPEC):
        if target == "EXCHANGE":
            print(f'    "{key}": {x[i]:.3f},')
    print("}\n")

    print("IRVING_WILLIAMS_BONUS = {")
    print('    "Ca2+": 0.0, "Mg2+": 0.0, "Ba2+": 0.0, "Sr2+": 0.0,')
    for i, (name, target, key, *_) in enumerate(PARAM_SPEC):
        if target == "IW":
            print(f'    "{key}": {x[i]:.2f},')
    # Extended entries not fitted
    print('    "Co3+": -22.0, "Ga3+": -5.0,')
    print("}\n")

    print("PARAMS = {")
    for i, (name, target, key, *_) in enumerate(PARAM_SPEC):
        if target == "PARAMS":
            print(f'    "{key}": {x[i]:.4f},')
    # Non-fitted PARAMS entries
    print('    "macro_sigma": 0.015,')
    print('    "freeze_mono": 0.25,')
    print('    "freeze_macro": 0.10,')
    print("}")


def run_calibration(verbose=False, export=True):
    """Full calibration pipeline: baseline → optimize → evaluate → export.

    Returns (x_opt, r2, mae).
    """
    print("═" * 65)
    print("  MABE Frozen Scorer Calibration")
    print("═" * 65)
    print(f"  Dataset:    {len(CAL_DATA)} complexes")
    print(f"  Parameters: {N_PARAMS}")
    print(f"  Ratio:      {len(CAL_DATA) / N_PARAMS:.1f}:1\n")

    x0 = get_x0()
    lo, hi = get_bounds()

    # ── Baseline ──
    print("── BEFORE calibration ──")
    r2_before, mae_before, _ = evaluate(x0)

    # ── Phase 1: broad optimization ──
    print("\n── Phase 1: bounded least-squares (trf) ──")
    result = least_squares(
        residuals, x0,
        bounds=(lo, hi),
        method='trf',
        max_nfev=10000,
        xtol=1e-12,
        ftol=1e-12,
    )
    x1 = result.x
    print(f"  Converged: {result.success}")
    print(f"  Evaluations: {result.nfev}")
    r2_1, mae_1, _ = evaluate(x1)

    # ── Phase 2: refine from Phase 1 solution ──
    print("\n── Phase 2: refinement pass ──")
    result2 = least_squares(
        residuals, x1,
        bounds=(lo, hi),
        method='trf',
        max_nfev=10000,
        xtol=1e-14,
        ftol=1e-14,
    )
    x_opt = result2.x
    print(f"  Converged: {result2.success}")
    print(f"  Evaluations: {result2.nfev}")

    # ── Final evaluation ──
    print("\n── AFTER calibration ──")
    r2_after, mae_after, rmse_after = evaluate(x_opt, verbose=verbose)

    print(f"\n  ┌──────────────────────────────────┐")
    print(f"  │  R²:  {r2_before:.4f} → {r2_after:.4f}          │")
    print(f"  │  MAE: {mae_before:.2f} → {mae_after:.2f} log K     │")
    print(f"  └──────────────────────────────────┘")

    # Show parameters at bounds
    at_bound = []
    for i, (name, target, key, init, lb, ub) in enumerate(PARAM_SPEC):
        if abs(x_opt[i] - lb) < 1e-6 or abs(x_opt[i] - ub) < 1e-6:
            at_bound.append(name)
    if at_bound:
        print(f"\n  ⚠ Parameters at bounds: {', '.join(at_bound)}")
        print("    (Consider widening bounds for these)")

    if export:
        export_params(x_opt)
        print_params(x_opt, "Fitted values")

    return x_opt, r2_after, mae_after