"""
core/validation_pipeline.py -- End-to-end validation across all MABE modalities.

Scores known compounds against experimental data, computes statistics,
flags outliers, and generates poster-ready output.

Usage:
    python -m core.validation_pipeline
    python -m core.validation_pipeline --json output.json
    python -m core.validation_pipeline --metal-only
    python -m core.validation_pipeline --glycan-only

Modalities:
  1. Glycan: 35 lectin-sugar predictions (R2 target: 0.96+)
  2. Metal: 19 representative complexes from NIST (Irving-Williams, EDTA series)
  3. Host-guest: 10 representative CD inclusions (Rekharsky & Inoue)
  4. De novo: known binder ranking validation

No new physics. Uses existing scorers only.
"""

from __future__ import annotations

import json
import math
import sys
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _r2(obs, pred):
    n = len(obs)
    if n < 3:
        return 0.0
    mean_o = sum(obs) / n
    ss_tot = sum((o - mean_o) ** 2 for o in obs)
    ss_res = sum((p - o) ** 2 for p, o in zip(pred, obs))
    return 1 - ss_res / max(ss_tot, 1e-10)


def _mae(obs, pred):
    return sum(abs(p - o) for p, o in zip(pred, obs)) / max(len(obs), 1)


def _rmse(obs, pred):
    mse = sum((p - o) ** 2 for p, o in zip(pred, obs)) / max(len(obs), 1)
    return math.sqrt(mse)


def _spearman(obs, pred):
    """Spearman rank correlation."""
    n = len(obs)
    if n < 3:
        return 0.0
    rank_o = _ranks(obs)
    rank_p = _ranks(pred)
    d2 = sum((ro - rp) ** 2 for ro, rp in zip(rank_o, rank_p))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def _ranks(values):
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(indexed):
        ranks[idx] = rank + 1
    return ranks


@dataclass
class ValidationResult:
    """Results for one modality."""
    modality: str
    n_total: int = 0
    n_scored: int = 0
    n_failed: int = 0
    r2: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    spearman: float = 0.0
    entries: List[dict] = field(default_factory=list)
    outliers: List[dict] = field(default_factory=list)
    selectivity_checks: List[dict] = field(default_factory=list)
    per_scaffold: Dict[str, dict] = field(default_factory=dict)
    elapsed_s: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# GLYCAN VALIDATION
# ---------------------------------------------------------------------------

def validate_glycan() -> ValidationResult:
    """Run glycan scorer on all predictions, compute stats."""
    t0 = time.time()
    from glycan.scorer import GlycanScorer

    s = GlycanScorer()
    preds = s.score_all()

    entries = []
    obs_list = []
    pred_list = []

    for p in preds:
        entry = {
            "scaffold": p.scaffold, "ligand": p.ligand,
            "dG_pred": round(p.dG_pred, 2),
            "dG_obs": round(p.dG_obs, 2) if p.dG_obs is not None else None,
            "residual": round(p.residual, 2) if p.residual is not None else None,
            "dG_HB": round(p.dG_HB, 2), "dG_CHP": round(p.dG_CHP, 2),
            "dG_desolv": round(p.dG_desolv, 2), "dG_conf": round(p.dG_conf, 2),
            "confidence": p.confidence,
        }
        entries.append(entry)
        if p.dG_obs is not None and p.residual is not None:
            obs_list.append(p.dG_obs)
            pred_list.append(p.dG_pred)

    # Per-scaffold breakdown
    scaffolds = sorted(set(p.scaffold for p in preds))
    per_scaffold = {}
    for sc in scaffolds:
        sc_obs = [o for o, p in zip(obs_list, pred_list)
                  if any(e["scaffold"] == sc and e["dG_obs"] == o for e in entries)]
        sc_pred = [p for o, p in zip(obs_list, pred_list)
                   if any(e["scaffold"] == sc and e["dG_obs"] == o for e in entries)]
        sc_entries = [e for e in entries if e["scaffold"] == sc and e["dG_obs"] is not None]
        sc_obs_v = [e["dG_obs"] for e in sc_entries]
        sc_pred_v = [e["dG_pred"] for e in sc_entries]
        per_scaffold[sc] = {
            "n": len(sc_entries),
            "mae": round(_mae(sc_obs_v, sc_pred_v), 2) if sc_obs_v else 0,
            "r2": round(_r2(sc_obs_v, sc_pred_v), 4) if len(sc_obs_v) >= 3 else None,
        }

    # Outliers (|residual| > 2.0 kJ/mol)
    outliers = [e for e in entries if e["residual"] is not None and abs(e["residual"]) > 2.0]

    # Selectivity checks
    sel_checks = []
    for sc, a, b, expected in [
        ("ConA", "Man", "Glc", True),
        ("Davis", "Glc", "Gal", True),
        ("Davis", "Glc", "Man", True),
        ("DGL", "Man", "Glc", True),
    ]:
        pa = s.score(sc, a)
        pb = s.score(sc, b)
        correct = pa.dG_pred < pb.dG_pred
        sel_checks.append({
            "scaffold": sc, "preferred": a, "over": b,
            "ddG": round(pa.dG_pred - pb.dG_pred, 2),
            "correct": correct,
        })

    # HIGH confidence subset
    high_obs = [e["dG_obs"] for e in entries if e["confidence"] == "HIGH" and e["dG_obs"] is not None]
    high_pred = [e["dG_pred"] for e in entries if e["confidence"] == "HIGH" and e["dG_obs"] is not None]

    result = ValidationResult(
        modality="glycan",
        n_total=len(preds),
        n_scored=len(obs_list),
        r2=round(_r2(obs_list, pred_list), 4),
        mae=round(_mae(obs_list, pred_list), 2),
        rmse=round(_rmse(obs_list, pred_list), 2),
        spearman=round(_spearman(obs_list, pred_list), 4),
        entries=entries,
        outliers=outliers,
        selectivity_checks=sel_checks,
        per_scaffold=per_scaffold,
        elapsed_s=round(time.time() - t0, 2),
        notes=f"HIGH subset: R2={_r2(high_obs, high_pred):.4f} MAE={_mae(high_obs, high_pred):.2f} n={len(high_obs)}",
    )
    return result


# ---------------------------------------------------------------------------
# METAL VALIDATION (representative subset)
# ---------------------------------------------------------------------------

# 19 representative metal-ligand complexes
_METAL_VALIDATION_SET = [
    # EDTA series (Irving-Williams + alkaline earth)
    "EDTA+Ca2+", "EDTA+Mg2+", "EDTA+Mn2+", "EDTA+Fe2+", "EDTA+Co2+",
    "EDTA+Ni2+", "EDTA+Cu2+", "EDTA+Zn2+", "EDTA+Pb2+", "EDTA+Cd2+",
    # Diverse ligands
    "NTA+Cu2+", "NTA+Ni2+", "DTPA+Cu2+",
    "bipy+Fe2+", "bipy+Cu2+",
    "phen+Fe2+", "phen+Cu2+",
    "en+Cu2+", "en+Ni2+",
]


def validate_metal() -> ValidationResult:
    """Score representative metal complexes through unified scorer."""
    t0 = time.time()
    from knowledge.cal_dataset import CAL_DATA
    from core.auto_descriptor import from_metal_ligand
    from core.unified_scorer_v2 import predict

    by_name = {e["name"]: e for e in CAL_DATA}

    entries = []
    obs_list = []
    pred_list = []
    errors = []

    for name in _METAL_VALIDATION_SET:
        if name not in by_name:
            continue
        cal = by_name[name]

        try:
            uc = from_metal_ligand(cal)
            result = predict(uc)
            residual = round(result.log_Ka_pred - cal["log_K_exp"], 2)

            entries.append({
                "name": name, "metal": cal["metal"],
                "logKa_obs": cal["log_K_exp"],
                "logKa_pred": round(result.log_Ka_pred, 2),
                "residual": residual,
                "dG_pred": round(result.dg_total_kj, 1),
            })
            obs_list.append(cal["log_K_exp"])
            pred_list.append(result.log_Ka_pred)
        except Exception as e:
            errors.append({"name": name, "error": str(e)})

    outliers = [e for e in entries if abs(e["residual"]) > 2.0]

    # Irving-Williams order check (for EDTA series)
    iw_order = ["Mn2+", "Fe2+", "Co2+", "Ni2+", "Cu2+", "Zn2+"]
    iw_pred = {}
    for e in entries:
        if e["name"].startswith("EDTA+") and e["metal"] in iw_order:
            iw_pred[e["metal"]] = e["logKa_pred"]

    iw_checks = []
    for i in range(len(iw_order) - 1):
        m1, m2 = iw_order[i], iw_order[i + 1]
        if m1 in iw_pred and m2 in iw_pred:
            # Cu2+ should be max, Zn2+ drops
            if m2 == "Zn2+":
                correct = iw_pred[m2] < iw_pred[m1]
            else:
                correct = iw_pred[m2] >= iw_pred[m1]
            iw_checks.append({
                "scaffold": "EDTA", "preferred": m2, "over": m1,
                "ddG": round(iw_pred[m2] - iw_pred[m1], 2),
                "correct": correct,
            })

    return ValidationResult(
        modality="metal",
        n_total=len(_METAL_VALIDATION_SET),
        n_scored=len(entries),
        n_failed=len(errors),
        r2=round(_r2(obs_list, pred_list), 4) if len(obs_list) >= 3 else 0.0,
        mae=round(_mae(obs_list, pred_list), 2) if obs_list else 0.0,
        rmse=round(_rmse(obs_list, pred_list), 2) if obs_list else 0.0,
        spearman=round(_spearman(obs_list, pred_list), 4) if len(obs_list) >= 3 else 0.0,
        entries=entries,
        outliers=outliers,
        selectivity_checks=iw_checks,
        elapsed_s=round(time.time() - t0, 2),
        notes=f"Irving-Williams checks: {sum(1 for c in iw_checks if c['correct'])}/{len(iw_checks)}",
    )


# ---------------------------------------------------------------------------
# HOST-GUEST VALIDATION (representative CD subset)
# ---------------------------------------------------------------------------

_HG_VALIDATION_NAMES = [
    # alpha-CD alkanol homologous series (test hydrophobic driving force)
    "aCD+1-butanol", "aCD+1-pentanol", "aCD+1-hexanol",
    # beta-CD aromatics (test aromatic inclusion)
    "bCD+benzene", "bCD+toluene", "bCD+naphthalene",
    "bCD+phenol", "bCD+benzoic-acid", "bCD+cyclohexanol",
    "bCD+adamantane-COOH",
]


def validate_host_guest() -> ValidationResult:
    """Score representative host-guest pairs through unified scorer."""
    t0 = time.time()
    from knowledge.hg_dataset import HG_DATA
    from core.auto_descriptor import from_host_guest
    from core.unified_scorer_v2 import predict

    by_name = {e["name"]: e for e in HG_DATA}

    entries = []
    obs_list = []
    pred_list = []
    errors = []

    for name in _HG_VALIDATION_NAMES:
        if name not in by_name:
            # Try fuzzy match
            match = [e for e in HG_DATA if name.split("+")[1] in e["name"]]
            if match:
                hg = match[0]
                name = hg["name"]
            else:
                errors.append({"name": name, "error": "not found"})
                continue
        else:
            hg = by_name[name]

        try:
            uc = from_host_guest(hg)
            result = predict(uc)
            log_ka_obs = hg.get("log_Ka_exp", hg.get("log_Ka", 0.0))
            residual = round(result.log_Ka_pred - log_ka_obs, 2)

            entries.append({
                "name": hg["name"], "host": hg["host"],
                "logKa_obs": log_ka_obs,
                "logKa_pred": round(result.log_Ka_pred, 2),
                "residual": residual,
            })
            obs_list.append(log_ka_obs)
            pred_list.append(result.log_Ka_pred)
        except Exception as e:
            errors.append({"name": name, "error": str(e)})

    outliers = [e for e in entries if abs(e["residual"]) > 1.0]

    return ValidationResult(
        modality="host_guest",
        n_total=len(_HG_VALIDATION_NAMES),
        n_scored=len(entries),
        n_failed=len(errors),
        r2=round(_r2(obs_list, pred_list), 4) if len(obs_list) >= 3 else 0.0,
        mae=round(_mae(obs_list, pred_list), 2) if obs_list else 0.0,
        rmse=round(_rmse(obs_list, pred_list), 2) if obs_list else 0.0,
        spearman=round(_spearman(obs_list, pred_list), 4) if len(obs_list) >= 3 else 0.0,
        entries=entries,
        outliers=outliers,
        elapsed_s=round(time.time() - t0, 2),
        notes=f"{len(errors)} entries not found/failed",
    )


# ---------------------------------------------------------------------------
# DE NOVO VALIDATION (known binder ranking)
# ---------------------------------------------------------------------------

def validate_denovo() -> ValidationResult:
    """Validate that known binders rank correctly in de novo output."""
    t0 = time.time()

    try:
        from glycan.de_novo_binder import (
            score_glycan_binder, compute_receptor_descriptor, list_sugar_targets,
        )
        from glycan.demand_grammar import generate_from_demand
    except ImportError:
        return ValidationResult(modality="denovo", notes="glycan module not available")

    KNOWN = {
        "Davis anth.+diurea": ("NC(=O)Nc1ccc2cc3ccc(NC(N)=O)cc3cc2c1", "Glc"),
        "Ke anthracene": ("c1ccc2cc3ccccc3cc2c1", "Glc"),
        "Phenylboronic acid": ("OB(O)c1ccccc1", "Fru"),
    }

    entries = []
    checks = []

    for name, (smi, expected_best) in KNOWN.items():
        desc = compute_receptor_descriptor(smi)
        scores = {}
        for t in list_sugar_targets():
            bs = score_glycan_binder(smi, t, desc)
            scores[t] = bs.dG_total
        actual_best = min(scores, key=scores.get)
        correct = actual_best == expected_best

        entries.append({
            "name": name, "expected_best": expected_best,
            "actual_best": actual_best, "correct": correct,
            "best_dG": round(scores[actual_best], 2),
        })
        checks.append({
            "scaffold": name, "preferred": expected_best, "over": "others",
            "ddG": round(scores[expected_best] - min(v for k, v in scores.items() if k != expected_best), 2),
            "correct": correct,
        })

    # Grammar vs fixed comparison for Glc
    from glycan.demand_grammar import compare_approaches
    comp = compare_approaches("Glc", max_candidates=100)
    grammar_better = comp["grammar"]["best_dG"] < comp["fixed"]["best_dG"]
    checks.append({
        "scaffold": "Glc", "preferred": "grammar", "over": "fixed",
        "ddG": round(comp["fixed"]["best_dG"] - comp["grammar"]["best_dG"], 2),
        "correct": grammar_better,
    })

    return ValidationResult(
        modality="denovo",
        n_total=len(KNOWN),
        n_scored=len(entries),
        entries=entries,
        selectivity_checks=checks,
        elapsed_s=round(time.time() - t0, 2),
        notes=f"Grammar vs fixed: {'PASS' if grammar_better else 'FAIL'} "
              f"(improvement: {comp['fixed']['best_dG'] - comp['grammar']['best_dG']:+.1f} kJ/mol)",
    )


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

@dataclass
class FullValidation:
    """Complete validation across all modalities."""
    glycan: Optional[ValidationResult] = None
    metal: Optional[ValidationResult] = None
    host_guest: Optional[ValidationResult] = None
    denovo: Optional[ValidationResult] = None
    total_elapsed_s: float = 0.0
    summary: str = ""


def run_full_validation(
    glycan: bool = True,
    metal: bool = True,
    host_guest: bool = True,
    denovo: bool = True,
) -> FullValidation:
    """Run all validation modalities."""
    t0 = time.time()
    result = FullValidation()

    if glycan:
        result.glycan = validate_glycan()
    if metal:
        result.metal = validate_metal()
    if host_guest:
        result.host_guest = validate_host_guest()
    if denovo:
        result.denovo = validate_denovo()

    result.total_elapsed_s = round(time.time() - t0, 2)

    # Build summary
    lines = ["MABE Validation Summary", "=" * 50]
    for name, vr in [("Glycan", result.glycan), ("Metal", result.metal),
                      ("Host-Guest", result.host_guest), ("De Novo", result.denovo)]:
        if vr is None:
            continue
        lines.append(f"{name}: R2={vr.r2:.4f} MAE={vr.mae:.2f} "
                     f"n={vr.n_scored}/{vr.n_total} "
                     f"rho={vr.spearman:.3f} "
                     f"outliers={len(vr.outliers)} "
                     f"[{vr.elapsed_s:.1f}s]")
        if vr.selectivity_checks:
            n_pass = sum(1 for c in vr.selectivity_checks if c["correct"])
            lines.append(f"  Selectivity: {n_pass}/{len(vr.selectivity_checks)} correct")
        if vr.notes:
            lines.append(f"  {vr.notes}")

    lines.append(f"\nTotal time: {result.total_elapsed_s:.1f}s")
    result.summary = "\n".join(lines)
    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_json(validation: FullValidation, path: str):
    """Export validation results to JSON for poster figures."""
    data = {}
    for name, vr in [("glycan", validation.glycan), ("metal", validation.metal),
                      ("host_guest", validation.host_guest), ("denovo", validation.denovo)]:
        if vr is None:
            continue
        data[name] = {
            "r2": vr.r2, "mae": vr.mae, "rmse": vr.rmse,
            "spearman": vr.spearman,
            "n_scored": vr.n_scored, "n_total": vr.n_total,
            "entries": vr.entries,
            "outliers": vr.outliers,
            "selectivity_checks": vr.selectivity_checks,
            "per_scaffold": vr.per_scaffold,
            "notes": vr.notes,
        }
    data["summary"] = validation.summary

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Formatted print
# ---------------------------------------------------------------------------

def print_validation(validation: FullValidation):
    """Print formatted validation report."""
    print()
    print(validation.summary)
    print()

    # Glycan detail
    if validation.glycan:
        vr = validation.glycan
        print("=" * 70)
        print("GLYCAN VALIDATION")
        print("=" * 70)
        print()
        print(f"  {'Scaffold':<10} {'Ligand':<16} {'Pred':>7} {'Obs':>7} "
              f"{'Resid':>7}  {'HB':>6} {'CH-pi':>6} {'Desol':>6} {'Conf':>6}  Conf.")
        print("  " + "-" * 95)
        for e in vr.entries:
            obs = f"{e['dG_obs']:7.2f}" if e["dG_obs"] is not None else "   NB  "
            res = f"{e['residual']:+6.2f}" if e["residual"] is not None else "   -- "
            print(f"  {e['scaffold']:<10} {e['ligand']:<16} {e['dG_pred']:7.2f} {obs} "
                  f"{res}  {e['dG_HB']:6.2f} {e['dG_CHP']:6.2f} {e['dG_desolv']:6.2f} "
                  f"{e['dG_conf']:6.2f}  [{e['confidence']}]")
        print()

        print("  Per-scaffold:")
        for sc, stats in sorted(vr.per_scaffold.items()):
            r2s = f"R2={stats['r2']:.4f}" if stats["r2"] is not None else "R2=n/a"
            print(f"    {sc:<10} n={stats['n']}  MAE={stats['mae']:.2f}  {r2s}")

        if vr.outliers:
            print(f"\n  Outliers (|residual| > 2.0):")
            for o in vr.outliers:
                print(f"    {o['scaffold']} {o['ligand']}: "
                      f"residual={o['residual']:+.2f} [{o['confidence']}]")

        print(f"\n  Selectivity:")
        for c in vr.selectivity_checks:
            tag = "PASS" if c["correct"] else "FAIL"
            print(f"    {c['scaffold']} {c['preferred']} > {c['over']}: "
                  f"ddG={c['ddG']:+.2f} [{tag}]")
        print()

    # Metal detail
    if validation.metal and validation.metal.entries:
        vr = validation.metal
        print("=" * 70)
        print("METAL VALIDATION (representative subset)")
        print("=" * 70)
        print()
        print(f"  {'Complex':<24} {'logKa obs':>10} {'logKa pred':>10} {'Resid':>7}")
        print("  " + "-" * 55)
        for e in vr.entries:
            print(f"  {e['name']:<24} {e['logKa_obs']:>10.1f} {e['logKa_pred']:>10.1f} "
                  f"{e['residual']:>+7.2f}")
        print()
        if vr.selectivity_checks:
            print("  Irving-Williams order (EDTA):")
            for c in vr.selectivity_checks:
                tag = "PASS" if c["correct"] else "FAIL"
                print(f"    {c['preferred']} > {c['over']}: "
                      f"dlogKa={c['ddG']:+.1f} [{tag}]")
        print()

    # Host-guest detail
    if validation.host_guest and validation.host_guest.entries:
        vr = validation.host_guest
        print("=" * 70)
        print("HOST-GUEST VALIDATION (representative subset)")
        print("=" * 70)
        print()
        print(f"  {'Complex':<30} {'logKa obs':>10} {'logKa pred':>10} {'Resid':>7}")
        print("  " + "-" * 60)
        for e in vr.entries:
            print(f"  {e['name']:<30} {e['logKa_obs']:>10.2f} {e['logKa_pred']:>10.2f} "
                  f"{e['residual']:>+7.2f}")
        print()

    # De novo detail
    if validation.denovo and validation.denovo.entries:
        vr = validation.denovo
        print("=" * 70)
        print("DE NOVO VALIDATION")
        print("=" * 70)
        print()
        for e in vr.entries:
            tag = "PASS" if e["correct"] else "FAIL"
            print(f"  {e['name']:<24} expected={e['expected_best']:<8} "
                  f"actual={e['actual_best']:<8} dG={e['best_dG']:+.1f} [{tag}]")
        print()
        for c in vr.selectivity_checks:
            tag = "PASS" if c["correct"] else "FAIL"
            print(f"  {c['scaffold']:<24} {c['preferred']} > {c['over']}: "
                  f"ddG={c['ddG']:+.1f} [{tag}]")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MABE Validation Pipeline")
    parser.add_argument("--json", type=str, default=None, help="Export JSON path")
    parser.add_argument("--glycan-only", action="store_true")
    parser.add_argument("--metal-only", action="store_true")
    parser.add_argument("--hg-only", action="store_true")
    parser.add_argument("--denovo-only", action="store_true")
    args = parser.parse_args()

    if args.glycan_only:
        v = run_full_validation(glycan=True, metal=False, host_guest=False, denovo=False)
    elif args.metal_only:
        v = run_full_validation(glycan=False, metal=True, host_guest=False, denovo=False)
    elif args.hg_only:
        v = run_full_validation(glycan=False, metal=False, host_guest=True, denovo=False)
    elif args.denovo_only:
        v = run_full_validation(glycan=False, metal=False, host_guest=False, denovo=True)
    else:
        v = run_full_validation()

    print_validation(v)

    if args.json:
        export_json(v, args.json)
        print(f"Exported to {args.json}")


if __name__ == "__main__":
    main()
