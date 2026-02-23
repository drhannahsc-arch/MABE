"""
cross_modal_predictor.py — Unified predictor for cross-modal entries.

For metal@CB[n], the physics is ion-dipole binding at the portal, NOT
inner-sphere coordination. The scoring function combines:

  NEW cross-modal params (ion-dipole physics):
    1. cm_portal_kq: ion-dipole attraction per charge per portal C=O
    2. cm_desolv_scale: desolvation penalty scaling (charge²/r proxy)
    3. cm_portal_size_k: size-match penalty at portal

  SHARED from HG scorer (coupling with pure organic@CB entries):
    4. dg_dehydr_per_A2 × dehydr_CB: CB cavity dehydration
    5. k_shape, PC_optimal, sigma_PC: shape complementarity

  SHARED from metal scorer (coupling with pure metal-ligand entries):
    6. macro_cavity_k: macrocyclic cavity size-match (also used for crowns)

Parameter coupling:
  - Adjusting dehydr_CB to improve organic@CB also changes metal@CB predictions
  - Adjusting macro_cavity_k for crown-ether metals also changes metal@CB
  - The optimizer MUST balance these shared parameters across all three modalities
"""

import math
import sys

sys.path.insert(0, 'knowledge')
sys.path.insert(0, 'core')
sys.path.insert(0, '.')

from scorer_frozen import METAL_DB
from hg_scorer import HG_PARAMS, LN10_RT
from hg_conf_shape import CONF_SHAPE_PARAMS
from cross_modal_dataset import CB_PORTAL_INFO

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-MODAL PARAMETERS (new, fitted in Phase 11b)
# ═══════════════════════════════════════════════════════════════════════════
CM_PARAMS = {
    # Ion-dipole attraction: ΔG = -cm_portal_kq × charge × min(n_carb, 6) / (r_ion + d_portal)
    # Units: kJ·Å/mol per unit charge per carbonyl
    "cm_portal_kq":     -3.0,     # attractive; typical ion-dipole ~2-8 kJ/mol

    # Desolvation cost: ΔG = +cm_desolv_scale × charge² / r_ion
    # Larger ions with lower hydration energy pay less
    "cm_desolv_scale":   1.5,     # kJ·Å/mol

    # Portal size mismatch: Gaussian penalty when ion doesn't fit portal
    "cm_portal_size_k":  5.0,     # kJ/mol max penalty
    "cm_portal_sigma":   0.6,     # Å, width of Gaussian
}

# Parameter specification for optimizer
CM_PARAM_SPEC = [
    ("cm_portal_kq",    "CM", "cm_portal_kq",     -3.0, -15.0,  -0.1),
    ("cm_desolv_scale", "CM", "cm_desolv_scale",    1.5,   0.0,  10.0),
    ("cm_portal_size_k","CM", "cm_portal_size_k",   5.0,   0.0,  20.0),
    ("cm_portal_sigma", "CM", "cm_portal_sigma",    0.6,   0.1,   2.0),
]


# ═══════════════════════════════════════════════════════════════════════════
# ION PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

def ion_volume_A3(metal_formula: str) -> float:
    """Compute ion van der Waals volume from ionic radius."""
    md = METAL_DB.get(metal_formula)
    if md is None:
        return 0.0
    r_A = md.ionic_radius_pm / 100.0
    return (4.0 / 3.0) * math.pi * r_A ** 3


def ion_effective_sasa(metal_formula: str) -> float:
    """Effective surface for cavity water displacement.

    Uses ionic radius + 0.5 Å hydration shell offset.
    """
    md = METAL_DB.get(metal_formula)
    if md is None:
        return 0.0
    r_eff = md.ionic_radius_pm / 100.0 + 0.5
    return 4.0 * math.pi * r_eff ** 2


# ═══════════════════════════════════════════════════════════════════════════
# ENERGY TERMS
# ═══════════════════════════════════════════════════════════════════════════

def dg_ion_dipole(entry: dict) -> float:
    """Ion-dipole attraction between cation and portal C=O groups.

    Scales as: charge × n_effective_carbonyls / (r_ion + d_portal)
    where d_portal accounts for the distance from ion center to C=O.

    Returns negative (favorable) kJ/mol.
    """
    md = METAL_DB[entry["metal"]]
    charge = md.charge
    r_A = md.ionic_radius_pm / 100.0
    n_carb = min(entry["n_portal_carbonyls"], 8)  # cap effective contacts
    portal_r = entry["portal_radius_nm"] * 10.0   # nm → Å

    # Effective distance: ion radius + half portal radius (geometric mean)
    d_eff = r_A + portal_r * 0.5

    return CM_PARAMS["cm_portal_kq"] * charge * n_carb / d_eff


def dg_desolvation(entry: dict) -> float:
    """Desolvation penalty for removing hydration shell.

    Scales as charge² / r (Born model proxy).
    Larger, less-charged ions pay less.

    Returns positive (unfavorable) kJ/mol.
    """
    md = METAL_DB[entry["metal"]]
    charge = md.charge
    r_A = md.ionic_radius_pm / 100.0

    return CM_PARAMS["cm_desolv_scale"] * charge ** 2 / r_A


def dg_portal_size_match(entry: dict) -> float:
    """Penalty for ion-portal size mismatch.

    Gaussian centered on optimal ion radius for portal.
    Ions much smaller or larger than portal radius bind poorly.

    Returns positive (unfavorable) kJ/mol for poor match.
    """
    md = METAL_DB[entry["metal"]]
    r_ion = md.ionic_radius_pm / 100.0
    portal_r = entry["portal_radius_nm"] * 10.0  # nm → Å

    delta = r_ion - portal_r
    sigma = CM_PARAMS["cm_portal_sigma"]
    mismatch = 1.0 - math.exp(-delta ** 2 / (2 * sigma ** 2))

    return CM_PARAMS["cm_portal_size_k"] * mismatch


def dg_cb_dehydration(entry: dict) -> float:
    """CB cavity dehydration (SHARED with HG scorer).

    Same parameters as used for organic guests in CB cavities.
    Ion entering cavity displaces high-energy water.

    Returns negative (favorable) kJ/mol.
    """
    metal = entry["metal"]
    cavity_sasa = entry["cavity_sasa"]

    ion_sasa = ion_effective_sasa(metal)
    buried = min(ion_sasa, cavity_sasa)

    base = HG_PARAMS["dg_dehydr_per_A2"]
    mult = HG_PARAMS["dehydr_CB"]  # CB-specific multiplier

    return base * mult * buried


def dg_shape_complementarity(entry: dict) -> float:
    """Shape complementarity in CB cavity (SHARED with HG scorer).

    Packing coefficient model: PC = V_ion / V_cavity.
    Uses same PC_optimal, sigma_PC, k_shape as organic guests.

    Returns kJ/mol.
    """
    metal = entry["metal"]
    v_ion = ion_volume_A3(metal)
    v_cav = entry["cavity_volume_A3"]

    if v_cav <= 0 or v_ion <= 0:
        return 0.0

    PC = v_ion / v_cav
    PC_opt = CONF_SHAPE_PARAMS["PC_optimal"]
    sigma = CONF_SHAPE_PARAMS["sigma_PC"]
    f_pack = math.exp(-(PC - PC_opt) ** 2 / (2 * sigma ** 2))

    if PC > 0.90:
        f_pack = -abs(f_pack) * (PC - 0.90) / 0.10

    return CONF_SHAPE_PARAMS["k_shape"] * f_pack


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

def predict_cross_modal_log_ka(entry: dict, verbose: bool = False) -> float:
    """Predict log Ka for metal@CB[n].

    Energy terms:
      1. Ion-dipole at portal     (NEW cm_portal_kq)
      2. Desolvation cost         (NEW cm_desolv_scale)
      3. Portal size match        (NEW cm_portal_size_k)
      4. CB dehydration           (SHARED HG: dehydr_CB)
      5. Shape complementarity    (SHARED HG: k_shape, PC_opt)
    """
    dg1 = dg_ion_dipole(entry)
    dg2 = dg_desolvation(entry)
    dg3 = dg_portal_size_match(entry)
    dg5 = dg_cb_dehydration(entry)
    dg6 = dg_shape_complementarity(entry)

    dg_total = dg1 + dg2 + dg3 + dg5 + dg6
    log_ka = -dg_total / LN10_RT

    if verbose:
        md = METAL_DB[entry["metal"]]
        r_A = md.ionic_radius_pm / 100.0
        v_ion = ion_volume_A3(entry["metal"])
        PC = v_ion / entry["cavity_volume_A3"] if entry["cavity_volume_A3"] > 0 else 0
        print(f"  {entry['name']}:")
        print(f"    Metal: {entry['metal']} (r={r_A:.2f}Å, z={md.charge})")
        print(f"    Host:  {entry['cb_host']} (portal Ø"
              f"{CB_PORTAL_INFO[entry['cb_host']]['portal_diameter_A']}Å)")
        print(f"    V_ion={v_ion:.1f}ų, PC={PC:.3f}")
        print(f"    ── Energy terms (kJ/mol) ──")
        print(f"    Ion-dipole:     {dg1:+8.2f}  [NEW: cm_portal_kq]")
        print(f"    Desolvation:    {dg2:+8.2f}  [NEW: cm_desolv_scale]")
        print(f"    Portal size:    {dg3:+8.2f}  [NEW: cm_portal_size_k]")
        print(f"    CB dehydr:      {dg5:+8.2f}  [SHARED: dehydr_CB]")
        print(f"    Shape:          {dg6:+8.2f}  [SHARED: k_shape]")
        print(f"    ΔG total:       {dg_total:+8.2f} kJ/mol")
        print(f"    log Ka:         {log_ka:+8.2f} "
              f"(exp={entry['log_Ka']:.2f}, err={log_ka - entry['log_Ka']:+.2f})")

    return log_ka


# ═══════════════════════════════════════════════════════════════════════════
# BATCH EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_crossmodal(data: list) -> dict:
    """Evaluate cross-modal entries and return stats."""
    import numpy as np

    preds, exps = [], []
    for e in data:
        try:
            pred = predict_cross_modal_log_ka(e)
        except Exception:
            pred = 0.0
        preds.append(pred)
        exps.append(e["log_Ka"])

    p, e = np.array(preds), np.array(exps)
    r = p - e
    ss_res = np.sum(r ** 2)
    ss_tot = np.sum((e - e.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(r))
    rmse = np.sqrt(np.mean(r ** 2))
    bias = np.mean(r)

    return {
        "n": len(data), "r2": r2, "mae": mae, "rmse": rmse,
        "bias": bias, "preds": p, "exps": e,
    }


if __name__ == "__main__":
    from cross_modal_dataset import CROSS_MODAL_DATA
    import numpy as np
    from collections import defaultdict

    print("═" * 60)
    print("  Cross-Modal Predictor: Metal@CB[n] baseline")
    print("═" * 60)
    print(f"  {len(CROSS_MODAL_DATA)} entries\n")

    stats = evaluate_crossmodal(CROSS_MODAL_DATA)
    print(f"  R²   = {stats['r2']:.4f}")
    print(f"  MAE  = {stats['mae']:.2f} log Ka")
    print(f"  RMSE = {stats['rmse']:.2f}")
    print(f"  Bias = {stats['bias']:+.2f}")

    # Per-host
    host_errors = defaultdict(list)
    for e in CROSS_MODAL_DATA:
        pred = predict_cross_modal_log_ka(e)
        host_errors[e["cb_host"]].append(pred - e["log_Ka"])

    print(f"\n  Per-host:")
    for host in sorted(host_errors):
        errs = np.array(host_errors[host])
        print(f"    {host}: n={len(errs):2d} MAE={np.mean(np.abs(errs)):.2f} "
              f"Bias={np.mean(errs):+.2f}")

    # Verbose for a few
    print("\n── Sample predictions ──")
    for name in ["CB5+Ba2+", "CB6+Cs+", "CB7+La3+", "CB7+Cu2+", "CB8+Na+"]:
        entry = next((e for e in CROSS_MODAL_DATA if e["name"] == name), None)
        if entry:
            predict_cross_modal_log_ka(entry, verbose=True)