"""
core/design_api.py — MABE Unified Design API

Single entry point for all binding design and scoring:
  - Metal chelator design (enumerate + score)
  - Cage/pocket design (topology + donor placement)
  - Drug-target ranking (SMILES → ranked by affinity)
  - Host-guest ranking (guests for synthetic hosts)
  - Selectivity screening (any modality)

Auto-detects mode from target specification when possible.

Usage:
    from core.design_api import design, score, list_targets

    # Score one molecule
    r = score("CC(=O)NO", target="Cu2+")
    r = score("CC(C)Cc1ccc(cc1)C(C)C(=O)O", target="COX-2")
    r = score("C1C2CC3CC1CC(C2)C3", target="beta-CD")

    # Rank a library
    results = design(target="Pb2+", interferents=["Ca2+", "Mg2+"],
                     smiles=["NCCN", "OC(=O)CN(CC(=O)O)CC(=O)O"])

    # Design a cage
    results = design(target="Hg2+", mode="cage", interferents=["Zn2+"])

    # List all available targets
    list_targets()
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.scorer_frozen import METAL_DB
from core.host_registry import HOST_REGISTRY
from core.unified_scorer_v2 import (
    PER_TARGET_PL_MODELS, PL_TARGET_OFFSETS, predict
)


# ═══════════════════════════════════════════════════════════════════════════
# TARGET CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

_METAL_NAMES = set(METAL_DB.keys())
_HOST_NAMES = set(HOST_REGISTRY.keys())
_PL_TARGETS = set(PER_TARGET_PL_MODELS.keys())
_METALLO_TARGETS = set(PL_TARGET_OFFSETS.keys())


def _classify_target(target):
    """Classify a target string into a modality.

    Returns: 'metal', 'host', 'metalloprotein', 'protein', or 'unknown'
    """
    if target in _METAL_NAMES:
        return "metal"
    if target in _HOST_NAMES:
        return "host"
    if target in _METALLO_TARGETS:
        return "metalloprotein"
    if target in _PL_TARGETS:
        return "protein"
    # Fuzzy match: try common aliases
    t_lower = target.lower().strip()
    for m in _METAL_NAMES:
        if t_lower == m.lower():
            return "metal"
    for h in _HOST_NAMES:
        if t_lower == h.lower():
            return "host"
    for p in _PL_TARGETS:
        if t_lower == p.lower():
            return "protein"
    return "unknown"


def _resolve_target(target):
    """Resolve target string to exact key in the appropriate database."""
    if target in _METAL_NAMES or target in _HOST_NAMES or target in _PL_TARGETS or target in _METALLO_TARGETS:
        return target
    t_lower = target.lower().strip()
    for db in [_METAL_NAMES, _HOST_NAMES, _PL_TARGETS, _METALLO_TARGETS]:
        for key in db:
            if t_lower == key.lower():
                return key
    return target


# ═══════════════════════════════════════════════════════════════════════════
# SCORE: Single molecule against any target
# ═══════════════════════════════════════════════════════════════════════════

def score(smiles, target, pH=7.4, n_ligand_molecules=1, name=None):
    """Score a single SMILES against any target.

    Auto-detects modality from target:
      - Metal ion → metal chelation scoring
      - Host name → host-guest inclusion scoring
      - Protein name → general PL scoring

    Returns:
        dict with log_Ka_pred, dg_total_kj, modality, decomposition
    """
    target = _resolve_target(target)
    modality = _classify_target(target)

    if modality == "metal":
        from core.auto_descriptor import from_smiles
        uc = from_smiles(smiles, metal=target, pH=pH,
                         n_ligand_molecules=n_ligand_molecules)
        r = predict(uc)
        return {
            "smiles": smiles, "target": target, "modality": "metal",
            "log_Ka_pred": r.log_Ka_pred, "dg_total_kj": r.dg_total_kj,
            "dg_metal": r.dg_metal, "name": name or smiles[:30],
        }

    elif modality == "host":
        from core.auto_descriptor import from_smiles
        uc = from_smiles(smiles, host=target, pH=pH)
        r = predict(uc)
        return {
            "smiles": smiles, "target": target, "modality": "host_guest",
            "log_Ka_pred": r.log_Ka_pred, "dg_total_kj": r.dg_total_kj,
            "dg_hydrophobic": r.dg_hydrophobic,
            "dg_cavity_dehydration": r.dg_cavity_dehydration,
            "dg_hbond": r.dg_hbond, "dg_shape": r.dg_shape,
            "name": name or smiles[:30],
        }

    elif modality in ("protein", "metalloprotein"):
        from core.auto_descriptor import from_protein_ligand
        uc = from_protein_ligand(smiles, target, name=name)
        r = predict(uc)
        return {
            "smiles": smiles, "target": target, "modality": modality,
            "log_Ka_pred": r.log_Ka_pred, "dg_total_kj": r.dg_total_kj,
            "dg_hydrophobic": r.dg_hydrophobic,
            "dg_hbond": r.dg_hbond, "dg_conf_entropy": r.dg_conf_entropy,
            "name": name or smiles[:30],
        }

    else:
        # Unknown target — try as protein with fallback model
        from core.auto_descriptor import from_protein_ligand
        uc = from_protein_ligand(smiles, target, name=name)
        r = predict(uc)
        return {
            "smiles": smiles, "target": target, "modality": "unknown_protein",
            "log_Ka_pred": r.log_Ka_pred, "dg_total_kj": r.dg_total_kj,
            "note": "Unknown target — using fallback global model",
            "name": name or smiles[:30],
        }


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN: Multi-molecule ranking or de novo design
# ═══════════════════════════════════════════════════════════════════════════

def design(target, smiles=None, interferents=None, mode=None,
           pH=7.4, n_ligand_molecules=None, top_n=20, **kwargs):
    """Universal design entry point.

    Auto-selects mode:
      - smiles provided + metal target → rank chelators
      - smiles provided + protein target → rank drugs
      - smiles provided + host target → rank guests
      - no smiles + metal target → enumerate donor sets (design_engine_v1)
      - mode="cage" → pocket designer
      - mode="library" → screen built-in ligand library
      - mode="selectivity" → forced selectivity screening

    Args:
        target: metal ion, host key, or protein target name
        smiles: list of SMILES to rank (or None for de novo)
        interferents: list of competing targets for selectivity
        mode: override auto-detection ("rank", "cage", "library", "selectivity")
        pH: working pH
        n_ligand_molecules: int or list (for metal chelation)
        top_n: max results to return
        **kwargs: passed to underlying engine

    Returns:
        dict with ranked results
    """
    target = _resolve_target(target)
    modality = _classify_target(target)

    # Force cage mode
    if mode == "cage":
        from core.pocket_designer import design_pocket, print_pocket_designs
        designs, elapsed = design_pocket(
            target, interferents=interferents, pH=pH,
            max_results=top_n, **kwargs)
        return {
            "mode": "cage_design", "target": target,
            "n_designs": len(designs), "elapsed_s": elapsed,
            "designs": designs,
            "print": lambda: print_pocket_designs(designs, elapsed, verbose=True),
        }

    # Force library screen
    if mode == "library":
        from core.design_engine_v2 import screen_ligand_library
        result = screen_ligand_library(
            target, interferents=interferents, pH=pH, top_n=top_n)
        return {
            "mode": "library_screen", "target": target,
            "result": result,
            "print": lambda: _print_de_result(result),
        }

    # SMILES provided → rank them
    if smiles is not None and len(smiles) > 0:
        names = kwargs.get("names", None)

        if modality == "metal":
            from core.design_engine_v2 import rank_binders, selectivity_screen, print_ranking
            if interferents:
                result = selectivity_screen(
                    target, interferents, smiles, names=names, pH=pH,
                    n_ligand_molecules=n_ligand_molecules)
            else:
                result = rank_binders(
                    target, smiles, names=names, pH=pH,
                    n_ligand_molecules=n_ligand_molecules)
            return {
                "mode": "metal_ranking", "target": target,
                "result": result,
                "print": lambda: print_ranking(result, verbose=True),
            }

        elif modality == "host":
            from core.design_engine_v2 import rank_hosts, print_ranking
            result = rank_hosts(target, smiles, names=names)
            return {
                "mode": "host_guest_ranking", "target": target,
                "result": result,
                "print": lambda: print_ranking(result, verbose=True),
            }

        else:
            # Protein target — rank by affinity
            from core.auto_descriptor import from_protein_ligand
            t0 = time.time()
            candidates = []
            errors = []
            name_list = names or [None] * len(smiles)
            for smi, nm in zip(smiles, name_list):
                try:
                    uc = from_protein_ligand(smi, target, name=nm)
                    r = predict(uc)
                    candidates.append({
                        "smiles": smi, "name": nm or smi[:30],
                        "log_Ka_pred": r.log_Ka_pred,
                        "dg_total_kj": r.dg_total_kj,
                    })
                except Exception as e:
                    errors.append((smi, str(e)))

            candidates.sort(key=lambda c: c["log_Ka_pred"], reverse=True)
            for i, c in enumerate(candidates):
                c["rank"] = i + 1

            return {
                "mode": "protein_ranking", "target": target,
                "n_scored": len(candidates), "n_failed": len(errors),
                "elapsed_s": time.time() - t0,
                "candidates": candidates[:top_n],
                "errors": errors,
                "print": lambda: _print_protein_ranking(target, candidates[:top_n]),
            }

    # No SMILES, metal target → de novo cage design
    if modality == "metal":
        from core.pocket_designer import design_pocket, print_pocket_designs
        designs, elapsed = design_pocket(
            target, interferents=interferents, pH=pH,
            max_results=top_n, **kwargs)
        return {
            "mode": "cage_design", "target": target,
            "n_designs": len(designs), "elapsed_s": elapsed,
            "designs": designs,
            "print": lambda: print_pocket_designs(designs, elapsed, verbose=True),
        }

    return {"mode": "error", "message": f"Cannot auto-detect design mode for target '{target}' without SMILES."}


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _print_de_result(result):
    from core.design_engine_v2 import print_ranking
    print_ranking(result, verbose=True)


def _print_protein_ranking(target, candidates):
    print(f"\n  MABE Protein-Ligand Ranking: {target}")
    print(f"  {'#':>3}  {'logKa':>7}  {'dG kJ':>7}  Name")
    print(f"  {'─'*50}")
    for c in candidates:
        print(f"  {c['rank']:3d}  {c['log_Ka_pred']:+7.2f}  "
              f"{c['dg_total_kj']:+7.0f}  {c['name'][:35]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def list_targets():
    """Print all available scoring targets grouped by modality."""
    print("\n  MABE Available Targets")
    print("  " + "=" * 60)

    print(f"\n  Metal ions ({len(_METAL_NAMES)}):")
    metals_sorted = sorted(_METAL_NAMES)
    for i in range(0, len(metals_sorted), 8):
        chunk = metals_sorted[i:i+8]
        print(f"    {', '.join(chunk)}")

    print(f"\n  Synthetic hosts ({len(_HOST_NAMES)}):")
    for h in sorted(_HOST_NAMES):
        print(f"    {h}")

    print(f"\n  Zn metalloprotein targets ({len(_METALLO_TARGETS)}):")
    for t in sorted(_METALLO_TARGETS):
        print(f"    {t}")

    print(f"\n  Non-metal protein targets ({len(_PL_TARGETS)}):")
    for t in sorted(_PL_TARGETS):
        print(f"    {t}")

    print(f"\n  Total scoring targets: {len(_METAL_NAMES) + len(_HOST_NAMES) + len(_METALLO_TARGETS) + len(_PL_TARGETS)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MABE Unified Design API — Self-Test")
    print("=" * 60)

    # 1. Score across modalities
    print("\n── Score: one molecule, four modalities ──")
    tests = [
        ("OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O", "Cu2+",    "EDTA→Cu²⁺"),
        ("C1C2CC3CC1CC(C2)C3",                     "beta-CD", "Adamantane→β-CD"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",             "COX-2",   "Ibuprofen→COX-2"),
        ("Fc1ccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)cc1Cl", "EGFR", "Gefitinib→EGFR"),
    ]
    for smi, tgt, label in tests:
        r = score(smi, tgt, name=label)
        print(f"  {label:25s}  logKa={r['log_Ka_pred']:+7.2f}  mode={r['modality']}")

    # 2. Rank chelators
    print("\n── Design: rank chelators for Pb²⁺ ──")
    r = design("Pb2+",
               smiles=["NCCN", "OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O", "CC(=O)NO"],
               names=["en", "EDTA", "AcHA"],
               interferents=["Ca2+"], pH=5.0)
    r["print"]()

    # 3. Rank drugs for EGFR
    print("── Design: rank drugs for EGFR ──")
    r = design("EGFR",
               smiles=[
                   "Fc1ccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)cc1Cl",
                   "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
                   "c1ccc2c(c1)c1cccnc1nc2",
               ],
               names=["Gefitinib", "Erlotinib", "Phenanthroline"])
    r["print"]()

    # 4. Cage design
    print("── Design: cage for Hg²⁺ ──")
    r = design("Hg2+", mode="cage", interferents=["Zn2+", "Pb2+"],
               pH=5.0, top_n=5)
    r["print"]()

    # 5. Host-guest
    print("── Design: β-CD guests ──")
    r = design("beta-CD",
               smiles=["C1C2CC3CC1CC(C2)C3", "c1ccccc1", "CCCCCCCC"],
               names=["adamantane", "benzene", "octane"])
    r["print"]()

    # 6. List targets
    list_targets()
