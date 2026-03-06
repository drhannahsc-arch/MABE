"""
core/design_engine_v2.py — Generalized Design Engine (Phase 17)

Scores and ranks any set of SMILES-based candidates against any target.
Consumer of from_smiles → predict pipeline. No new physics.

Entry points:
  rank_binders(metal, smiles_list, ...) → ranked by predicted log Ka
  selectivity_screen(metal, interferents, smiles_list, ...) → ranked by selectivity
  rank_hosts(host_key, smiles_list) → ranked by host-guest log Ka
  score_one(smiles, metal=None, host=None, ...) → full decomposition

Works across modalities: metal chelation, host-guest, mixed-mode.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.auto_descriptor import from_smiles
from core.unified_scorer_v2 import predict, PredictionResult, LN10_RT


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScoredCandidate:
    """Full scoring result for one SMILES candidate."""
    smiles: str
    name: str
    log_Ka_pred: float
    dg_total_kj: float
    prediction: PredictionResult
    # Selectivity fields (populated by selectivity_screen)
    interferent_scores: dict = field(default_factory=dict)   # metal → log Ka
    selectivity_gaps: dict = field(default_factory=dict)     # metal → gap
    min_gap: float = 0.0
    worst_interferent: str = ""
    grade: str = ""
    rank: int = 0

    @property
    def donors(self):
        return self.prediction.binding_mode


@dataclass
class RankingResult:
    """Output of a ranking or screening run."""
    target: str
    mode: str                       # "metal", "host_guest", "mixed", "selectivity"
    interferents: list = field(default_factory=list)
    candidates: list = field(default_factory=list)   # list[ScoredCandidate]
    n_input: int = 0
    n_scored: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def best(self) -> Optional[ScoredCandidate]:
        return self.candidates[0] if self.candidates else None


# ═══════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_one(smiles, metal=None, host=None, pH=7.4, n_ligand_molecules=1,
              name=None):
    """Score a single SMILES and return full decomposition.

    Args:
        smiles: SMILES string
        metal: e.g. "Cu2+", "Pb2+" (activates metal coordination)
        host: e.g. "beta-CD", "CB7" (activates host-guest)
        pH: working pH (default 7.4)
        n_ligand_molecules: for bis/tris complexes
        name: optional label

    Returns:
        ScoredCandidate with full prediction decomposition
    """
    uc = from_smiles(smiles, metal=metal, host=host, pH=pH,
                     n_ligand_molecules=n_ligand_molecules)
    r = predict(uc)
    return ScoredCandidate(
        smiles=smiles,
        name=name or smiles[:40],
        log_Ka_pred=r.log_Ka_pred,
        dg_total_kj=r.dg_total_kj,
        prediction=r,
    )


# ═══════════════════════════════════════════════════════════════════════════
# RANKING: METAL CHELATORS
# ═══════════════════════════════════════════════════════════════════════════

def rank_binders(metal, smiles_list, names=None, pH=7.4,
                 n_ligand_molecules=None, descending=True):
    """Rank SMILES candidates by predicted log Ka for a target metal.

    Args:
        metal: target metal ion, e.g. "Pb2+"
        smiles_list: list of SMILES strings
        names: optional parallel list of names
        pH: working pH
        n_ligand_molecules: int or list[int] (per-candidate)
        descending: True = strongest binder first

    Returns:
        RankingResult with candidates sorted by log_Ka_pred
    """
    t0 = time.time()
    if names is None:
        names = [None] * len(smiles_list)
    if n_ligand_molecules is None:
        n_ligs = [1] * len(smiles_list)
    elif isinstance(n_ligand_molecules, int):
        n_ligs = [n_ligand_molecules] * len(smiles_list)
    else:
        n_ligs = list(n_ligand_molecules)

    candidates = []
    errors = []
    for smi, nm, nl in zip(smiles_list, names, n_ligs):
        try:
            sc = score_one(smi, metal=metal, pH=pH,
                           n_ligand_molecules=nl, name=nm)
            candidates.append(sc)
        except Exception as e:
            errors.append((smi, str(e)))

    candidates.sort(key=lambda c: c.log_Ka_pred, reverse=descending)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return RankingResult(
        target=metal,
        mode="metal",
        candidates=candidates,
        n_input=len(smiles_list),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY SCREENING
# ═══════════════════════════════════════════════════════════════════════════

def _grade(min_gap):
    """Selectivity grade from minimum gap."""
    if min_gap >= 5.0:
        return "A"
    elif min_gap >= 3.0:
        return "B"
    elif min_gap >= 1.0:
        return "C"
    elif min_gap >= 0.0:
        return "D"
    return "F"


def selectivity_screen(target_metal, interferents, smiles_list,
                        names=None, pH=7.4, n_ligand_molecules=None):
    """Rank candidates by selectivity: target log Ka minus worst interferent.

    Args:
        target_metal: e.g. "Pb2+"
        interferents: list of competing metals, e.g. ["Ca2+", "Mg2+", "Fe3+"]
        smiles_list: candidate SMILES
        names: optional labels
        pH: working pH
        n_ligand_molecules: int or list[int]

    Returns:
        RankingResult with candidates sorted by min_gap (best selectivity first)
    """
    t0 = time.time()
    if names is None:
        names = [None] * len(smiles_list)
    if n_ligand_molecules is None:
        n_ligs = [1] * len(smiles_list)
    elif isinstance(n_ligand_molecules, int):
        n_ligs = [n_ligand_molecules] * len(smiles_list)
    else:
        n_ligs = list(n_ligand_molecules)

    candidates = []
    errors = []
    for smi, nm, nl in zip(smiles_list, names, n_ligs):
        try:
            # Score target
            sc = score_one(smi, metal=target_metal, pH=pH,
                           n_ligand_molecules=nl, name=nm)

            # Score each interferent
            for intf in interferents:
                try:
                    intf_sc = score_one(smi, metal=intf, pH=pH,
                                        n_ligand_molecules=nl)
                    sc.interferent_scores[intf] = intf_sc.log_Ka_pred
                    sc.selectivity_gaps[intf] = (sc.log_Ka_pred
                                                  - intf_sc.log_Ka_pred)
                except Exception:
                    sc.interferent_scores[intf] = 0.0
                    sc.selectivity_gaps[intf] = sc.log_Ka_pred

            if sc.selectivity_gaps:
                sc.worst_interferent = min(sc.selectivity_gaps,
                                           key=sc.selectivity_gaps.get)
                sc.min_gap = sc.selectivity_gaps[sc.worst_interferent]
            else:
                sc.min_gap = sc.log_Ka_pred
            sc.grade = _grade(sc.min_gap)

            candidates.append(sc)
        except Exception as e:
            errors.append((smi, str(e)))

    candidates.sort(key=lambda c: c.min_gap, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return RankingResult(
        target=target_metal,
        mode="selectivity",
        interferents=list(interferents),
        candidates=candidates,
        n_input=len(smiles_list),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HOST-GUEST RANKING
# ═══════════════════════════════════════════════════════════════════════════

def rank_hosts(host_key, smiles_list, names=None):
    """Rank guest SMILES by predicted log Ka for a synthetic host.

    Args:
        host_key: HOST_REGISTRY key, e.g. "beta-CD", "CB7"
        smiles_list: guest SMILES
        names: optional labels

    Returns:
        RankingResult sorted by log_Ka_pred (best guest first)
    """
    t0 = time.time()
    if names is None:
        names = [None] * len(smiles_list)

    candidates = []
    errors = []
    for smi, nm in zip(smiles_list, names):
        try:
            sc = score_one(smi, host=host_key, name=nm)
            candidates.append(sc)
        except Exception as e:
            errors.append((smi, str(e)))

    candidates.sort(key=lambda c: c.log_Ka_pred, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return RankingResult(
        target=host_key,
        mode="host_guest",
        candidates=candidates,
        n_input=len(smiles_list),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LIBRARY SCREENING (from LIGAND_DB)
# ═══════════════════════════════════════════════════════════════════════════

def screen_ligand_library(target_metal, interferents=None, pH=7.4,
                           categories=None, top_n=20):
    """Screen the built-in ligand library for a target metal.

    Args:
        target_metal: e.g. "Pb2+"
        interferents: optional list of competing metals
        pH: working pH
        categories: optional list to filter, e.g. ["aminocarboxylate", "crown_ether"]
        top_n: how many to return

    Returns:
        RankingResult (selectivity mode if interferents provided)
    """
    from knowledge.ligand_library import LIGAND_DB

    smiles_list = []
    names_list = []
    n_lig_list = []

    for lig in LIGAND_DB:
        if categories and lig.category not in categories:
            continue
        smiles_list.append(lig.smiles)
        names_list.append(lig.name)
        n_lig_list.append(1)

    if interferents:
        result = selectivity_screen(
            target_metal, interferents, smiles_list,
            names=names_list, pH=pH, n_ligand_molecules=n_lig_list)
    else:
        result = rank_binders(
            target_metal, smiles_list,
            names=names_list, pH=pH, n_ligand_molecules=n_lig_list)

    result.candidates = result.candidates[:top_n]
    for i, c in enumerate(result.candidates):
        c.rank = i + 1
    return result


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_ranking(result, verbose=False):
    """Pretty-print a RankingResult."""
    print()
    print(f"  MABE Design Engine v2 — {result.mode} ranking")
    print(f"  Target: {result.target}")
    if result.interferents:
        print(f"  Interferents: {', '.join(result.interferents)}")
    print(f"  Scored: {result.n_scored}/{result.n_input} "
          f"({result.n_failed} failed) in {result.elapsed_s:.1f}s")
    print()

    if not result.candidates:
        print("  No candidates scored.")
        return

    if result.mode == "selectivity":
        print(f"  {'#':>3s}  {'Grade':5s}  {'logKa':>6s}  {'MinGap':>7s}  "
              f"{'Worst':>6s}  Name")
        print(f"  {'─'*65}")
        for c in result.candidates:
            print(f"  {c.rank:3d}  {c.grade:^5s}  {c.log_Ka_pred:+6.1f}  "
                  f"{c.min_gap:+7.1f}  {c.worst_interferent:>6s}  "
                  f"{c.name[:35]}")
    else:
        print(f"  {'#':>3s}  {'logKa':>7s}  {'dG kJ/mol':>10s}  Name")
        print(f"  {'─'*55}")
        for c in result.candidates:
            print(f"  {c.rank:3d}  {c.log_Ka_pred:+7.2f}  "
                  f"{c.dg_total_kj:+10.1f}  {c.name[:35]}")

    if verbose and result.best:
        b = result.best
        print()
        print(f"  ── Best: {b.name} ──")
        r = b.prediction
        for attr in ['dg_metal', 'dg_hydrophobic', 'dg_hbond',
                     'dg_conf_entropy', 'dg_shape', 'dg_cavity_dehydration',
                     'dg_ion_dipole']:
            val = getattr(r, attr, 0.0)
            if abs(val) > 0.01:
                print(f"    {attr:25s} {val:+8.2f} kJ/mol")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MABE Design Engine v2 — Self-Test")
    print("=" * 60)

    # Test 1: Rank chelators for Cu2+
    chelators = [
        ("OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O", "EDTA"),
        ("OC(=O)CN(CC(=O)O)CC(=O)O", "NTA"),
        ("NCCN", "en"),
        ("CC(=O)CC(C)=O", "acac"),
        ("CC(=O)NO", "acetohydroxamic acid"),
        ("c1ccc2c(c1)c1cccnc1nc2", "1,10-phen"),
        ("OC(=O)CNCC(=O)O", "IDA"),
    ]
    smiles = [s for s, n in chelators]
    names = [n for s, n in chelators]

    r = rank_binders("Cu2+", smiles, names=names, pH=7.4)
    print_ranking(r, verbose=True)

    # Test 2: Selectivity screen for Pb2+ over Ca/Mg
    r2 = selectivity_screen(
        "Pb2+", ["Ca2+", "Mg2+", "Fe3+"],
        smiles, names=names, pH=5.0)
    print_ranking(r2)

    # Test 3: Host-guest ranking for beta-CD
    guests = [
        ("C1C2CC3CC1CC(C2)C3", "adamantane"),
        ("c1ccccc1", "benzene"),
        ("CCCCCCCC", "octane"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ibuprofen"),
        ("OC1CCCCC1", "cyclohexanol"),
    ]
    r3 = rank_hosts("beta-CD", [s for s,n in guests], names=[n for s,n in guests])
    print_ranking(r3, verbose=True)

    # Test 4: Library screen
    print("Screening ligand library for Pb2+ selectivity over Ca2+...")
    r4 = screen_ligand_library("Pb2+", interferents=["Ca2+", "Mg2+"],
                                pH=5.0, top_n=10)
    print_ranking(r4)
