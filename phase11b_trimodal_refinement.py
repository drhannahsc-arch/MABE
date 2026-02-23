"""
phase11b_trimodal_refinement.py — MABE Phase 11b: Tri-Modal Joint Refinement

Three modalities optimized simultaneously:
  1. Metal coordination (500 entries, 55 params)
  2. Host-guest inclusion (80 entries, 21 params)  
  3. Cross-modal metal@CB[n] (64 entries, 4 NEW params + SHARED)

Parameter coupling:
  - Cross-modal predictions use HG params: dehydr_CB, dg_dehydr_per_A2,
    k_shape, PC_optimal, sigma_PC → changing these affects BOTH HG and CM
  - 4 new CM params: cm_portal_kq, cm_desolv_scale, cm_portal_size_k,
    cm_portal_sigma → only affect CM entries
  - Metal params remain decoupled (CM uses its own ion-dipole physics)

Total: 55 metal + 21 HG + 4 CM = 80 params, 644 entries (8.1:1 ratio)
"""

import sys
import time
import numpy as np
from scipy.optimize import least_squares
from collections import defaultdict

sys.path.insert(0, 'knowledge')
sys.path.insert(0, 'core')
sys.path.insert(0, '.')

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
from cal_params import PARAM_SPEC as METAL_PARAM_SPEC, apply_params as apply_metal_params, get_x0 as metal_x0
from cal_dataset import CAL_DATA
from scorer_frozen import predict_log_k

from hg_dataset import HG_DATA
from hg_scorer import predict_hg_log_ka, HG_PARAMS, compute_guest_sasa
from hg_hbond import HBOND_PARAMS
from hg_pi import PI_PARAMS
from hg_conf_shape import CONF_SHAPE_PARAMS

from cross_modal_dataset import CROSS_MODAL_DATA
from cross_modal_predictor import (
    predict_cross_modal_log_ka, CM_PARAMS, CM_PARAM_SPEC,
    evaluate_crossmodal,
)

# ═══════════════════════════════════════════════════════════════════════════
# HG PARAMETER SPEC (from Phase 10)
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

# Indices of SHARED HG params that also affect CM predictions
# These are the params that create actual coupling
SHARED_HG_INDICES = {
    "dg_dehydr_per_A2": 2,   # index in HG_PARAM_SPEC
    "dehydr_CB":        3,
    "k_shape":          18,
    "PC_optimal":       19,
    "sigma_PC":         20,
}


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def apply_hg_params(x_hg):
    """Patch HG module dicts."""
    for i, (_, dn, key, *_) in enumerate(HG_PARAM_SPEC):
        if dn == "HG":   HG_PARAMS[key] = x_hg[i]
        elif dn == "HB": HBOND_PARAMS[key] = x_hg[i]
        elif dn == "PI": PI_PARAMS[key] = x_hg[i]
        elif dn == "CS": CONF_SHAPE_PARAMS[key] = x_hg[i]


def apply_cm_params(x_cm):
    """Patch cross-modal params."""
    for i, (_, _, key, *_) in enumerate(CM_PARAM_SPEC):
        CM_PARAMS[key] = x_cm[i]


def apply_all(x):
    """Apply [metal | hg | cm] parameter vector."""
    apply_metal_params(x[:N_METAL])
    apply_hg_params(x[N_METAL:N_METAL + N_HG])
    apply_cm_params(x[N_METAL + N_HG:])


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
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

def stats(preds, exps):
    p, e = np.array(preds), np.array(exps)
    r = p - e
    ss_tot = np.sum((e - e.mean()) ** 2)
    r2 = 1.0 - np.sum(r ** 2) / ss_tot if ss_tot > 0 else 0.0
    return {"r2": r2, "mae": np.mean(np.abs(r)), "rmse": np.sqrt(np.mean(r**2)),
            "bias": np.mean(r), "n": len(p), "max_err": np.max(np.abs(r))}


def eval_metal():
    return stats([metal_predict(e) for e in CAL_DATA],
                 [e["log_K_exp"] for e in CAL_DATA])

def eval_hg():
    return stats([hg_predict(e) for e in HG_DATA],
                 [e["log_Ka"] for e in HG_DATA])

def eval_cm():
    return stats([cm_predict(e) for e in CROSS_MODAL_DATA],
                 [e["log_Ka"] for e in CROSS_MODAL_DATA])


def print_tri_stats(label, ms, hs, cs):
    print(f"\n  ┌─── {label} {'─' * max(0, 48 - len(label))}┐")
    print(f"  │  Metal:  R²={ms['r2']:.4f}  MAE={ms['mae']:.2f}  n={ms['n']}       │")
    print(f"  │  HG:     R²={hs['r2']:.4f}  MAE={hs['mae']:.2f}  n={hs['n']}        │")
    print(f"  │  CM:     R²={cs['r2']:.4f}  MAE={cs['mae']:.2f}  n={cs['n']}        │")
    n_tot = ms['n'] + hs['n'] + cs['n']
    wmae = (ms['mae']*ms['n'] + hs['mae']*hs['n'] + cs['mae']*cs['n']) / n_tot
    print(f"  │  wMAE:   {wmae:.2f}  ({n_tot} entries, {N_TOTAL} params)  │")
    print(f"  └──────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════
# SEPARATE CALIBRATIONS (anchors)
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_metal_only():
    x0 = [p[3] for p in METAL_PARAM_SPEC]
    lo = [p[4] for p in METAL_PARAM_SPEC]
    hi = [p[5] for p in METAL_PARAM_SPEC]
    def res(x):
        apply_metal_params(x)
        return [metal_predict(e) - e["log_K_exp"] for e in CAL_DATA]
    result = least_squares(res, x0, bounds=(lo, hi), method="trf",
                           max_nfev=10000, xtol=1e-12, ftol=1e-12)
    apply_metal_params(result.x)
    return result.x


def calibrate_hg_only():
    x0 = [p[3] for p in HG_PARAM_SPEC]
    lo = [p[4] for p in HG_PARAM_SPEC]
    hi = [p[5] for p in HG_PARAM_SPEC]
    def res(x):
        apply_hg_params(x)
        return [hg_predict(e) - e["log_Ka"] for e in HG_DATA]
    result = least_squares(res, x0, bounds=(lo, hi), method="trf", max_nfev=5000)
    apply_hg_params(result.x)
    return result.x


def calibrate_cm_only(x_hg_anchor):
    """CM-only calibration with HG params frozen at anchor."""
    apply_hg_params(x_hg_anchor)  # fix shared params
    x0 = [p[3] for p in CM_PARAM_SPEC]
    lo = [p[4] for p in CM_PARAM_SPEC]
    hi = [p[5] for p in CM_PARAM_SPEC]
    def res(x):
        apply_cm_params(x)
        return [cm_predict(e) - e["log_Ka"] for e in CROSS_MODAL_DATA]
    result = least_squares(res, x0, bounds=(lo, hi), method="trf", max_nfev=5000)
    apply_cm_params(result.x)
    return result.x


# ═══════════════════════════════════════════════════════════════════════════
# TRI-MODAL JOINT REFINEMENT
# ═══════════════════════════════════════════════════════════════════════════

def trimodal_residuals(x, x_anchor, lam_m, lam_h, lam_c):
    """Combined residuals: [metal | hg | cm | reg_metal | reg_hg | reg_cm]."""
    apply_all(x)

    metal_res = [metal_predict(e) - e["log_K_exp"] for e in CAL_DATA]
    hg_res = [hg_predict(e) - e["log_Ka"] for e in HG_DATA]
    cm_res = [cm_predict(e) - e["log_Ka"] for e in CROSS_MODAL_DATA]

    # Regularization
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


def run_trimodal(x_anchor, lam_m=0.3, lam_h=0.1, lam_c=0.05, max_nfev=20000):
    """Single run of tri-modal optimization."""
    lo_m = [p[4] for p in METAL_PARAM_SPEC]
    hi_m = [p[5] for p in METAL_PARAM_SPEC]
    lo_h = [p[4] for p in HG_PARAM_SPEC]
    hi_h = [p[5] for p in HG_PARAM_SPEC]
    lo_c = [p[4] for p in CM_PARAM_SPEC]
    hi_c = [p[5] for p in CM_PARAM_SPEC]

    lo = np.array(lo_m + lo_h + lo_c)
    hi = np.array(hi_m + hi_h + hi_c)
    x0 = x_anchor.copy()
    # Clip to bounds
    x0 = np.clip(x0, lo + 1e-10, hi - 1e-10)

    result = least_squares(
        trimodal_residuals, x0,
        args=(x_anchor, lam_m, lam_h, lam_c),
        bounds=(lo, hi), method="trf",
        max_nfev=max_nfev, xtol=1e-13, ftol=1e-13,
    )
    apply_all(result.x)
    return result


def adaptive_trimodal(x_metal_anchor, x_hg_anchor, x_cm_anchor):
    """Run tri-modal refinement with multiple λ sweeps."""
    x_anchor = np.concatenate([x_metal_anchor, x_hg_anchor, x_cm_anchor])

    best_x = x_anchor.copy()
    best_wmae = 999.0

    # Grid: try different regularization strengths
    configs = [
        (0.5, 0.2, 0.05, "Strong metal reg"),
        (0.3, 0.1, 0.02, "Moderate reg"),
        (0.1, 0.05, 0.01, "Light reg"),
        (0.05, 0.02, 0.005, "Very light reg"),
        (0.01, 0.01, 0.001, "Minimal reg"),
    ]

    for lam_m, lam_h, lam_c, label in configs:
        print(f"\n  ── {label}: λ_m={lam_m}, λ_h={lam_h}, λ_c={lam_c} ──")

        result = run_trimodal(x_anchor, lam_m, lam_h, lam_c)
        apply_all(result.x)

        ms = eval_metal()
        hs = eval_hg()
        cs = eval_cm()
        n_tot = ms["n"] + hs["n"] + cs["n"]
        wmae = (ms["mae"]*ms["n"] + hs["mae"]*hs["n"] + cs["mae"]*cs["n"]) / n_tot

        print(f"    Metal: R²={ms['r2']:.4f} MAE={ms['mae']:.2f}")
        print(f"    HG:    R²={hs['r2']:.4f} MAE={hs['mae']:.2f}")
        print(f"    CM:    R²={cs['r2']:.4f} MAE={cs['mae']:.2f}")
        print(f"    wMAE={wmae:.2f}, nfev={result.nfev}")

        # Check metal constraint
        if ms["r2"] < 0.885:
            print(f"    ⚠ Metal R² {ms['r2']:.4f} < 0.885 — rejected")
            continue
        if hs["r2"] < 0.840:
            print(f"    ⚠ HG R² {hs['r2']:.4f} < 0.840 — rejected")
            continue

        if wmae < best_wmae:
            best_wmae = wmae
            best_x = result.x.copy()
            print(f"    ✓ New best wMAE={wmae:.2f}")

    return best_x


# ═══════════════════════════════════════════════════════════════════════════
# DRIFT REPORT
# ═══════════════════════════════════════════════════════════════════════════

def drift_report(x_before, x_after, spec, label, top_n=10):
    drifts = []
    for i, (name, *_) in enumerate(spec):
        d = x_after[i] - x_before[i]
        pct = abs(d / x_before[i]) * 100 if abs(x_before[i]) > 1e-6 else 0.0
        drifts.append((name, x_before[i], x_after[i], d, pct))
    drifts.sort(key=lambda t: -t[4])

    print(f"\n  {label} — Top drifts:")
    print(f"  {'Param':25s} {'Anchor':>8s} {'Joint':>8s} {'Δ':>8s} {'%':>6s}")
    print("  " + "─" * 56)
    for name, bef, aft, delta, pct in drifts[:top_n]:
        flag = " ←" if pct > 10 else ""
        print(f"  {name:25s} {bef:8.4f} {aft:8.4f} {delta:+8.4f} {pct:5.1f}%{flag}")


# ═══════════════════════════════════════════════════════════════════════════
# PER-HOST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def per_host_cm():
    host_errors = defaultdict(list)
    for e in CROSS_MODAL_DATA:
        pred = cm_predict(e)
        host_errors[e["cb_host"]].append(pred - e["log_Ka"])
    result = {}
    for host in sorted(host_errors):
        errs = np.array(host_errors[host])
        result[host] = {"n": len(errs), "mae": np.mean(np.abs(errs)),
                        "bias": np.mean(errs)}
    return result


def per_host_hg():
    host_errors = defaultdict(list)
    for e in HG_DATA:
        pred = hg_predict(e)
        host_errors[e["host"]].append(pred - e["log_Ka"])
    result = {}
    for host in sorted(host_errors):
        errs = np.array(host_errors[host])
        result[host] = {"n": len(errs), "mae": np.mean(np.abs(errs)),
                        "bias": np.mean(errs)}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("═" * 65)
    print("  MABE Phase 11b: Tri-Modal Joint Refinement")
    print("═" * 65)
    print(f"  Metal:  {len(CAL_DATA)} entries, {N_METAL} params")
    print(f"  HG:     {len(HG_DATA)} entries, {N_HG} params")
    print(f"  CM:     {len(CROSS_MODAL_DATA)} entries, {N_CM} NEW params")
    total = len(CAL_DATA) + len(HG_DATA) + len(CROSS_MODAL_DATA)
    print(f"  Total:  {total} entries, {N_TOTAL} params ({total/N_TOTAL:.1f}:1)")
    print(f"\n  Shared HG→CM params: {list(SHARED_HG_INDICES.keys())}")

    # Pre-warm SASA
    print("\n── Pre-computing SASA ──")
    for e in HG_DATA:
        try: compute_guest_sasa(e["guest_smiles"])
        except: pass

    # ── Step 1: Metal anchor ──
    print("\n── Step 1: Metal-only calibration ──")
    x_metal = calibrate_metal_only()
    ms0 = eval_metal()
    print(f"  R²={ms0['r2']:.4f} MAE={ms0['mae']:.2f}")

    # ── Step 2: HG anchor ──
    print("\n── Step 2: HG-only calibration ──")
    x_hg = calibrate_hg_only()
    hs0 = eval_hg()
    print(f"  R²={hs0['r2']:.4f} MAE={hs0['mae']:.2f}")

    # ── Step 3: CM anchor (with HG frozen) ──
    print("\n── Step 3: CM-only calibration (HG frozen) ──")
    x_cm = calibrate_cm_only(x_hg)
    cs0 = eval_cm()
    print(f"  R²={cs0['r2']:.4f} MAE={cs0['mae']:.2f}")

    # Print CM params
    for i, (name, *_) in enumerate(CM_PARAM_SPEC):
        print(f"    {name:20s} = {x_cm[i]:.4f}")

    print_tri_stats("ANCHOR (separate)", ms0, hs0, cs0)

    # Per-host
    hg_hosts = per_host_hg()
    cm_hosts = per_host_cm()
    print("\n  HG per-host (anchor):")
    for h, s in hg_hosts.items():
        print(f"    {h:15s} n={s['n']:2d} MAE={s['mae']:.2f} Bias={s['bias']:+.2f}")
    print("  CM per-host (anchor):")
    for h, s in cm_hosts.items():
        print(f"    {h:15s} n={s['n']:2d} MAE={s['mae']:.2f} Bias={s['bias']:+.2f}")

    # ── Step 4: Tri-modal joint refinement ──
    print("\n" + "═" * 65)
    print("  Step 4: Adaptive Tri-Modal Refinement")
    print("═" * 65)

    x_joint = adaptive_trimodal(x_metal, x_hg, x_cm)

    # ── Final evaluation ──
    apply_all(x_joint)
    ms1 = eval_metal()
    hs1 = eval_hg()
    cs1 = eval_cm()
    print_tri_stats("AFTER TRI-MODAL REFINEMENT", ms1, hs1, cs1)

    # Per-host after
    hg_hosts1 = per_host_hg()
    cm_hosts1 = per_host_cm()
    print("\n  HG per-host (joint):")
    for h, s in hg_hosts1.items():
        a = hg_hosts.get(h, {"mae": 0})
        d = s["mae"] - a["mae"]
        arrow = "↑" if d > 0.05 else "↓" if d < -0.05 else "="
        print(f"    {h:15s} n={s['n']:2d} MAE={s['mae']:.2f} Δ={d:+.2f} {arrow}")
    print("  CM per-host (joint):")
    for h, s in cm_hosts1.items():
        a = cm_hosts.get(h, {"mae": 0})
        d = s["mae"] - a["mae"]
        arrow = "↑" if d > 0.05 else "↓" if d < -0.05 else "="
        print(f"    {h:15s} n={s['n']:2d} MAE={s['mae']:.2f} Δ={d:+.2f} {arrow}")

    # ── Drift analysis ──
    x_anchor = np.concatenate([x_metal, x_hg, x_cm])
    drift_report(x_metal, x_joint[:N_METAL], METAL_PARAM_SPEC, "Metal")
    drift_report(x_hg, x_joint[N_METAL:N_METAL+N_HG], HG_PARAM_SPEC, "HG")
    drift_report(x_cm, x_joint[N_METAL+N_HG:], CM_PARAM_SPEC, "CM")

    # ── Summary ──
    print("\n" + "═" * 65)
    print("  PHASE 11b SUMMARY")
    print("═" * 65)
    print(f"\n  {'Metric':20s} {'Anchor':>10s} {'Joint':>10s} {'Δ':>8s}")
    print("  " + "─" * 52)
    for label, s0, s1 in [("Metal R²", ms0, ms1), ("HG R²", hs0, hs1), ("CM R²", cs0, cs1)]:
        print(f"  {label:20s} {s0['r2']:10.4f} {s1['r2']:10.4f} {s1['r2']-s0['r2']:+8.4f}")
    for label, s0, s1 in [("Metal MAE", ms0, ms1), ("HG MAE", hs0, hs1), ("CM MAE", cs0, cs1)]:
        print(f"  {label:20s} {s0['mae']:10.2f} {s1['mae']:10.2f} {s1['mae']-s0['mae']:+8.2f}")

    anchor_wmae = (ms0["mae"]*500 + hs0["mae"]*80 + cs0["mae"]*64) / 644
    joint_wmae = (ms1["mae"]*500 + hs1["mae"]*80 + cs1["mae"]*64) / 644
    print(f"  {'wMAE':20s} {anchor_wmae:10.2f} {joint_wmae:10.2f} {joint_wmae-anchor_wmae:+8.2f}")

    print(f"\n  Metal R² ≥ 0.885: {'✓ PASS' if ms1['r2'] >= 0.885 else '✗ FAIL'}")
    print(f"  HG R² ≥ 0.840:    {'✓ PASS' if hs1['r2'] >= 0.840 else '✗ FAIL'}")
    print(f"  CM R² > 0:         {'✓ PASS' if cs1['r2'] > 0 else '✗ FAIL'}")

    # Print shared param values
    print(f"\n  SHARED PARAMS (HG→CM coupling):")
    x_h_joint = x_joint[N_METAL:N_METAL+N_HG]
    for name, idx in SHARED_HG_INDICES.items():
        print(f"    {name:20s}  anchor={x_hg[idx]:.4f}  joint={x_h_joint[idx]:.4f}  "
              f"Δ={x_h_joint[idx]-x_hg[idx]:+.4f}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.0f}s")
    return x_joint


if __name__ == "__main__":
    x_opt = main()