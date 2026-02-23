#!/usr/bin/env python3
"""
run_nist_calibration.py — Feed NIST SRD 46 entries through MABE physics engine.
v2: Smart donor selection fixes overcounting from RDKit.

USAGE:
    cd /path/to/MABE
    python run_nist_calibration.py

REQUIRES:
    - All sprint bootstraps run (through Sprint 36a, or Sprint 36 + bootstrap_nist_physics.py)
    - knowledge/nist_calibration_entries.json (from nist_mabe_pipeline.py)
"""

import json
import math
import os
import sys
import csv
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from core.physics_integration import compute_enhanced_thermodynamics
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    Problem, TargetSpecies, Matrix,
)


# ═══════════════════════════════════════════════════════════════════════════
# METAL PROPERTIES FOR BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

PREFERRED_CN = {
    "Li+": 4, "Na+": 6, "K+": 6, "Rb+": 8, "Cs+": 8,
    "Ag+": 2, "Cu+": 4, "Tl+": 6, "Au+": 2,
    "Mg2+": 6, "Ca2+": 6, "Sr2+": 8, "Ba2+": 8,
    "Mn2+": 6, "Fe2+": 6, "Co2+": 6, "Ni2+": 6,
    "Cu2+": 6, "Zn2+": 6, "Cd2+": 6, "Pb2+": 6,
    "Hg2+": 4, "Pd2+": 4, "Pt2+": 4, "Sn2+": 6,
    "Ru2+": 6, "Os2+": 6, "VO2+": 5,
    "Fe3+": 6, "Al3+": 6, "Cr3+": 6, "Ga3+": 6, "In3+": 6,
    "Bi3+": 6, "Sc3+": 6, "Au3+": 4, "Rh3+": 6,
    "Ir3+": 6, "Ti3+": 6, "V3+": 6, "Ru3+": 6,
    "La3+": 9, "Ce3+": 9, "Nd3+": 9, "Sm3+": 9,
    "Eu3+": 8, "Gd3+": 8, "Tb3+": 8,
    "Dy3+": 8, "Ho3+": 8, "Er3+": 8,
    "Yb3+": 8, "Lu3+": 8,
    "Th4+": 8, "U4+": 8, "UO2_2+": 5,
}

HYDRATED_RADIUS_NM = {
    "Li+": 0.38, "Na+": 0.36, "K+": 0.33, "Rb+": 0.33, "Cs+": 0.33,
    "Ag+": 0.34, "Cu+": 0.30, "Tl+": 0.33, "Au+": 0.30,
    "Mg2+": 0.43, "Ca2+": 0.41, "Sr2+": 0.41, "Ba2+": 0.40,
    "Mn2+": 0.44, "Fe2+": 0.43, "Co2+": 0.42, "Ni2+": 0.40,
    "Cu2+": 0.42, "Zn2+": 0.43, "Cd2+": 0.43, "Pb2+": 0.40,
    "Hg2+": 0.40, "Pd2+": 0.40, "Pt2+": 0.40, "Sn2+": 0.40,
    "Fe3+": 0.46, "Al3+": 0.48, "Cr3+": 0.46, "Ga3+": 0.47,
    "In3+": 0.44, "Au3+": 0.40, "Bi3+": 0.40,
    "La3+": 0.45, "Ce3+": 0.45, "Nd3+": 0.44, "Sm3+": 0.44,
    "Eu3+": 0.43, "Gd3+": 0.43, "Dy3+": 0.43,
    "Yb3+": 0.42, "Lu3+": 0.42,
    "Sc3+": 0.44, "Rh3+": 0.42, "Ir3+": 0.42,
    "Ti3+": 0.44, "V3+": 0.44, "Ru2+": 0.42, "Ru3+": 0.42,
    "Os2+": 0.42, "VO2+": 0.40, "Th4+": 0.45, "U4+": 0.45,
}

DONOR_PRIORITY = {
    "O_catecholate": 10, "O_hydroxamate": 9, "O_phenolate": 8,
    "S_thiolate": 8, "N_pyridine": 7, "N_imine": 7,
    "N_amine": 7, "N_imidazole": 7, "S_thioether": 6,
    "O_carboxylate": 6, "O_hydroxyl": 5, "P_phosphine": 5,
    "O_phosphoryl": 4, "O_carbonyl": 3, "O_ether": 2,
    "O_sulfonate": 2, "S_thiosulfate": 3, "S_dithiocarbamate": 5,
}

# Donors where RDKit double-counts: multiple atoms per functional group
_PAIRED_DONORS = {
    "O_carboxylate": 2,   # COO- has 2 O, 1 coordinates
    "O_phosphoryl": 2,    # PO3 has 2-3 terminal O, ~1 coordinates
    "O_sulfonate": 3,     # SO3- has 3 O, ~1 coordinates
}


def deduplicate_donors(donor_subtypes):
    """Fix RDKit overcounting: reduce paired donors to coordination count.
    EDTA: 2N + 8O_carb -> 2N + 4O_carb (one O per COO- group)
    Glycine: 1N + 2O_carb -> 1N + 1O_carb
    """
    counts = Counter(donor_subtypes)
    deduped = []
    for subtype in donor_subtypes:
        if subtype in _PAIRED_DONORS:
            n_already = sum(1 for d in deduped if d == subtype)
            n_real = max(1, counts[subtype] // _PAIRED_DONORS[subtype])
            if n_already < n_real:
                deduped.append(subtype)
        else:
            deduped.append(subtype)
    return deduped


def select_coordination_donors(donor_subtypes, cn):
    """Select up to CN donors, strongest first."""
    if len(donor_subtypes) <= cn:
        return donor_subtypes
    ranked = sorted(donor_subtypes,
                    key=lambda d: DONOR_PRIORITY.get(d, 1), reverse=True)
    return ranked[:cn]


def nist_to_engine_formula(nist_formula):
    import re
    roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    m = re.match(r'^(\w+)\((\w+)\)$', nist_formula)
    if not m:
        return nist_formula
    elem, roman = m.groups()
    charge = roman_to_int.get(roman, 0)
    return f"{elem}+" if charge == 1 else f"{elem}{charge}+"


def subtypes_to_elements(ds):
    return [d.split("_")[0] for d in ds]


def compute_hsab_match(metal_softness, donor_subtypes):
    _ds = {"O_carboxylate": 0.15, "O_hydroxyl": 0.10, "O_phenolate": 0.20,
           "O_ether": 0.10, "O_phosphoryl": 0.15, "O_carbonyl": 0.15,
           "O_sulfonate": 0.15, "N_amine": 0.40, "N_pyridine": 0.45,
           "N_imine": 0.50, "S_thiolate": 0.80, "S_thioether": 0.75,
           "P_phosphine": 0.75}
    if not donor_subtypes:
        return 0.5
    avg = sum(_ds.get(d, 0.3) for d in donor_subtypes) / len(donor_subtypes)
    return max(0.0, 1.0 - abs(metal_softness - avg) * 2.0)


def calc_stats(actual, predicted):
    n = len(actual)
    if n == 0:
        return {"n": 0, "R2": 0, "MAE": 0, "RMSE": 0, "bias": 0}
    mean_a = sum(actual) / n
    ss_res = sum((a - p)**2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean_a)**2 for a in actual)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
    rmse = math.sqrt(ss_res / n)
    bias = sum(predicted) / n - mean_a
    return {"n": n, "R2": round(r2, 4), "MAE": round(mae, 2),
            "RMSE": round(rmse, 2), "bias": round(bias, 2)}


def run_calibration(json_path, max_entries=None, k1_only=True):
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        entries = json.load(f)

    if k1_only:
        entries = [e for e in entries if e["log_K_type"] == "K1"]
        print(f"  K1 only: {len(entries)} entries")
    if max_entries:
        entries = entries[:max_entries]
        print(f"  Capped at {max_entries}")

    actual_all, predicted_all = [], []
    results, errors = [], []
    by_metal = defaultdict(lambda: {"actual": [], "predicted": []})
    by_hsab = defaultdict(lambda: {"actual": [], "predicted": []})
    by_donor = defaultdict(lambda: {"actual": [], "predicted": []})
    dedup_stats = {"before": 0, "after": 0}

    for i, entry in enumerate(entries):
        try:
            engine_formula = nist_to_engine_formula(entry["metal_formula"])
            charge = entry["metal_charge"]
            d_elec = entry["metal_d_electrons"]
            softness = entry["metal_softness"]
            radius_pm = entry["metal_ionic_radius_pm"]
            hsab = entry["metal_hsab"]

            raw_subtypes = entry["donor_subtypes"]
            dedup_stats["before"] += len(raw_subtypes)

            deduped = deduplicate_donors(raw_subtypes)
            cn = PREFERRED_CN.get(engine_formula, 6)
            effective_subtypes = select_coordination_donors(deduped, cn)
            dedup_stats["after"] += len(effective_subtypes)

            effective_donors = subtypes_to_elements(effective_subtypes)
            denticity = len(effective_donors)
            if denticity == 0:
                errors.append({"entry_id": entry.get("entry_id"), "error": "No donors"})
                continue

            chelate_rings = max(0, denticity - 1) if denticity > 1 else 0
            hsab_match = compute_hsab_match(softness, effective_subtypes)
            hydrated_r = HYDRATED_RADIUS_NM.get(engine_formula, 0.40)

            target = TargetSpecies(
                identity=entry["metal_formula"], formula=engine_formula,
                charge=charge, d_electrons=d_elec, hsab_softness=softness,
                coordination_number=denticity, ionic_radius_pm=radius_pm,
                hydrated_radius_nm=hydrated_r)
            matrix = Matrix(ph=7.0, temperature_c=25.0, ionic_strength_mm=100.0)
            problem = Problem(target=target, matrix=matrix)

            recognition = RecognitionChemistry(
                name=entry["entry_id"], type="chelator",
                donor_atoms=effective_donors, donor_type="mixed",
                denticity=denticity, hsab_match=hsab_match,
                chelate_rings=chelate_rings)
            recognition.donor_subtypes = effective_subtypes
            recognition.is_macrocyclic = entry.get("is_macrocyclic", False)
            recognition.cavity_radius_nm = 0.0
            recognition.ring_sizes = None
            recognition.is_cage = False

            structure = StructuralConstraint(
                name="free", type="free", geometry="octahedral", pore_size_nm=0.0)
            interior = InteriorDesign(
                description="free ligand", num_binding_sites=1, self_binding=False)

            thermo = compute_enhanced_thermodynamics(recognition, structure, interior, problem)
            # NIST log K = thermodynamic (fully deprotonated ligand).
            # Subtract protonation penalty which only applies to conditional K.
            dg_thermo = thermo.dg_net_kj - thermo.dg_protonation_kj
            log_k_pred_raw = -dg_thermo / 5.71
            from core.metal_offsets import apply_offset
            log_k_pred = apply_offset(entry["metal_formula"], log_k_pred_raw)
            log_k_exp = entry["log_K_exp"]
            residual = log_k_pred - log_k_exp

            actual_all.append(log_k_exp)
            predicted_all.append(log_k_pred)
            by_metal[entry["metal_formula"]]["actual"].append(log_k_exp)
            by_metal[entry["metal_formula"]]["predicted"].append(log_k_pred)
            by_hsab[hsab]["actual"].append(log_k_exp)
            by_hsab[hsab]["predicted"].append(log_k_pred)
            for ds in set(effective_subtypes):
                by_donor[ds]["actual"].append(log_k_exp)
                by_donor[ds]["predicted"].append(log_k_pred)

            results.append({
                "entry_id": entry["entry_id"], "metal": entry["metal_formula"],
                "ligand": entry["ligand_name"][:40],
                "log_K_exp": log_k_exp, "log_K_pred": round(log_k_pred, 2),
                "residual": round(residual, 2), "dg_net": round(thermo.dg_net_kj, 2),
                "donors_raw": len(raw_subtypes), "donors_dedup": len(deduped),
                "donors_used": denticity})

            if (i + 1) % 1000 == 0:
                s = calc_stats(actual_all, predicted_all)
                print(f"  [{i+1}/{len(entries)}] R2={s['R2']:.3f} MAE={s['MAE']:.2f} bias={s['bias']:+.2f}")

        except Exception as ex:
            errors.append({"entry_id": entry.get("entry_id", "?"), "error": str(ex)})
            if len(errors) <= 10:
                import traceback
                print(f"  ERROR {entry.get('entry_id','?')}: {ex}")
                traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  NIST SRD 46 CALIBRATION RESULTS")
    print("=" * 70)
    print(f"  Entries processed: {len(actual_all)}")
    print(f"  Errors: {len(errors)}")
    avg_b = dedup_stats["before"] / max(1, len(actual_all))
    avg_a = dedup_stats["after"] / max(1, len(actual_all))
    print(f"  Donor dedup: avg {avg_b:.1f} raw -> {avg_a:.1f} used")

    overall = calc_stats(actual_all, predicted_all)
    print(f"\n  OVERALL: R2={overall['R2']:.4f}  MAE={overall['MAE']:.2f}  "
          f"RMSE={overall['RMSE']:.2f}  bias={overall['bias']:+.2f}")

    print(f"\n  {'Metal':12s} {'n':>5s} {'R2':>7s} {'MAE':>6s} {'bias':>7s}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*7}")
    for metal in sorted(by_metal, key=lambda m: -len(by_metal[m]["actual"])):
        d = by_metal[metal]
        if len(d["actual"]) >= 10:
            s = calc_stats(d["actual"], d["predicted"])
            print(f"  {metal:12s} {s['n']:5d} {s['R2']:7.3f} {s['MAE']:6.2f} {s['bias']:+7.2f}")

    print(f"\n  {'HSAB':12s} {'n':>5s} {'R2':>7s} {'MAE':>6s} {'bias':>7s}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*6} {'-'*7}")
    for h in ["hard", "borderline", "soft"]:
        if h in by_hsab:
            d = by_hsab[h]
            s = calc_stats(d["actual"], d["predicted"])
            print(f"  {h:12s} {s['n']:5d} {s['R2']:7.3f} {s['MAE']:6.2f} {s['bias']:+7.2f}")

    print(f"\n  {'Donor':20s} {'n':>5s} {'MAE':>6s} {'bias':>7s}")
    print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*7}")
    for ds in sorted(by_donor, key=lambda d: -len(by_donor[d]["actual"])):
        d = by_donor[ds]
        if len(d["actual"]) >= 20:
            s = calc_stats(d["actual"], d["predicted"])
            print(f"  {ds:20s} {s['n']:5d} {s['MAE']:6.2f} {s['bias']:+7.2f}")

    results_sorted = sorted(results, key=lambda r: abs(r["residual"]), reverse=True)
    print(f"\n  WORST 20:")
    print(f"  {'Metal':10s} {'Ligand':40s} {'exp':>6s} {'pred':>6s} {'resid':>7s} {'d_raw':>5s} {'d_use':>5s}")
    print(f"  {'-'*10} {'-'*40} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*5}")
    for r in results_sorted[:20]:
        print(f"  {r['metal']:10s} {r['ligand']:40s} {r['log_K_exp']:6.1f} "
              f"{r['log_K_pred']:6.1f} {r['residual']:+7.1f} {r['donors_raw']:5d} {r['donors_used']:5d}")

    out_csv = os.path.join(BASE, "nist_calibration_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Detailed: {out_csv}")
    return overall, results


if __name__ == "__main__":
    json_path = os.path.join(BASE, "knowledge", "nist_calibration_entries.json")
    if not os.path.exists(json_path):
        for alt in ["nist_calibration_entries.json",
                     os.path.join(BASE, "data", "nist_calibration_entries.json")]:
            if os.path.exists(alt):
                json_path = alt
                break
    if not os.path.exists(json_path):
        print("ERROR: Cannot find nist_calibration_entries.json")
        sys.exit(1)
    run_calibration(json_path, k1_only=True)