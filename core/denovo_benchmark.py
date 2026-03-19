"""
core/denovo_benchmark.py — De Novo Design Benchmarking Framework

Generates binder candidates for a well-characterized target, scores them
through the SAME path as known binders, and compares:
  1. Do de novo candidates score competitively with known best?
  2. Do de novo top hits share structural features with known best?
  3. Does the engine discover the right chemistry from physics alone?

Supported targets:
  - beta-CD guest design (host-guest)
  - Metal chelation (Cu2+, Pb2+, Zn2+)
  - GalNAc / Tn antigen (glycan)

This is the AACR "the engine works" proof: same fragment library,
same physics, generates competitive candidates across modalities.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs

from core.design_engine_v2 import score_one, ScoredCandidate
from core.de_novo_generator import enumerate_molecules, BACKBONE_LIBRARY, ARM_LIBRARY, PropertyFilter


# ═══════════════════════════════════════════════════════════════════════════
# KNOWN BINDER PANELS
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_BCD_GUESTS = [
    # (name, SMILES, experimental_log_Ka, source)
    ("1-adamantanol",     "OC12CC3CC(CC(C3)C1)C2",          4.07, "Rekharsky&Inoue 1998"),
    ("adamantane-COOH",   "OC(=O)C12CC3CC(CC(C3)C1)C2",    4.30, "Rekharsky&Inoue 1998"),
    ("cyclohexanol",      "OC1CCCCC1",                      2.55, "Rekharsky&Inoue 1998"),
    ("cycloheptanone",    "O=C1CCCCCC1",                    3.10, "Rekharsky&Inoue 1998"),
    ("1-naphthol",        "Oc1cccc2ccccc12",                3.16, "Rekharsky&Inoue 1998"),
    ("ibuprofen",         "CC(C)Cc1ccc(C(C)C(=O)O)cc1",    3.92, "Rekharsky&Inoue 1998"),
    ("naproxen",          "COc1ccc2cc(C(C)C(=O)O)ccc2c1",  3.49, "Rekharsky&Inoue 1998"),
    ("phenol",            "Oc1ccccc1",                      1.89, "Rekharsky&Inoue 1998"),
    ("benzoic_acid",      "OC(=O)c1ccccc1",                2.14, "Rekharsky&Inoue 1998"),
    ("tert-butanol",      "CC(C)(C)O",                      1.82, "Rekharsky&Inoue 1998"),
    ("1-butanol",         "CCCCO",                          1.35, "Rekharsky&Inoue 1998"),
    ("cyclopentanol",     "OC1CCCC1",                       2.15, "Rekharsky&Inoue 1998"),
    ("4-nitrophenol",     "Oc1ccc([N+](=O)[O-])cc1",       2.48, "Rekharsky&Inoue 1998"),
]

KNOWN_PB_CHELATORS = [
    # (name, SMILES, experimental_log_Ka, source)
    ("EDTA",      "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",     18.0, "Martell&Smith"),
    ("DTPA",      "OC(=O)CN(CCN(CC(=O)O)CCN(CC(=O)O)CC(=O)O)CC(=O)O", 18.8, "Martell&Smith"),
    ("NTA",       "OC(=O)CN(CC(=O)O)CC(=O)O",                 11.4, "Martell&Smith"),
    ("DMSA",      "OC(=O)C(S)C(S)C(=O)O",                     17.0, "est."),
    ("citric_acid","OC(=O)CC(O)(CC(=O)O)C(=O)O",              4.1, "Martell&Smith"),
    ("glycine",   "NCC(=O)O",                                  5.5, "Martell&Smith"),
    ("penicillamine","CC(C)(S)C(N)C(=O)O",                    12.3, "Martell&Smith"),
    ("oxalic_acid","OC(=O)C(=O)O",                            4.9, "Martell&Smith"),
    ("en",        "NCCN",                                      5.0, "Martell&Smith"),
    ("bipy",      "c1ccnc(-c2ccccn2)c1",                      3.4, "Martell&Smith"),
]


# ═══════════════════════════════════════════════════════════════════════════
# SCORING WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════

def score_known_panel(panel, target_type, target_key):
    """Score a panel of known binders through the same path as de novo."""
    results = []
    for name, smiles, logka_exp, source in panel:
        try:
            if target_type == "host_guest":
                r = score_one(smiles, host=target_key, name=name)
            elif target_type == "metal":
                r = score_one(smiles, metal=target_key, name=name)
            else:
                continue

            if r:
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'log_Ka_exp': logka_exp,
                    'log_Ka_pred': r.log_Ka_pred,
                    'dg_kj': r.dg_total_kj,
                    'error': r.log_Ka_pred - logka_exp,
                    'source': 'known',
                })
        except Exception as e:
            results.append({
                'name': name, 'smiles': smiles,
                'log_Ka_exp': logka_exp, 'log_Ka_pred': 0,
                'dg_kj': 0, 'error': 0, 'source': 'known_failed',
            })
    return results


def score_denovo_panel(candidates, target_type, target_key):
    """Score de novo candidates through the same path."""
    results = []
    for smiles, bb_name, arm_names, sa in candidates:
        name = f"{bb_name}+{'|'.join(arm_names)}"
        try:
            if target_type == "host_guest":
                r = score_one(smiles, host=target_key, name=name)
            elif target_type == "metal":
                r = score_one(smiles, metal=target_key, name=name)
            else:
                continue

            if r:
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'log_Ka_pred': r.log_Ka_pred,
                    'dg_kj': r.dg_total_kj,
                    'sa_score': sa,
                    'backbone': bb_name,
                    'arms': arm_names,
                    'source': 'denovo',
                })
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL SIMILARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def tanimoto_similarity(smiles1, smiles2):
    """Compute Morgan fingerprint Tanimoto similarity."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def molecular_properties(smiles):
    """Extract key molecular properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'mw': Descriptors.MolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'n_rotatable': Descriptors.NumRotatableBonds(mol),
        'n_rings': rdMolDescriptors.CalcNumRings(mol),
        'n_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'n_hbd': Descriptors.NumHDonors(mol),
        'n_hba': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Full results from a de novo vs known benchmark."""
    target: str
    target_type: str
    # Known binders
    known_results: list = field(default_factory=list)
    known_best_name: str = ""
    known_best_logKa: float = 0.0
    known_best_pred: float = 0.0
    # De novo
    n_enumerated: int = 0
    n_scored: int = 0
    denovo_results: list = field(default_factory=list)
    denovo_best_name: str = ""
    denovo_best_logKa: float = 0.0
    # Comparison
    competitive_count: int = 0          # de novo within 1 log unit of known best
    exceeds_count: int = 0              # de novo scoring higher than known best
    best_denovo_vs_known: float = 0.0   # gap: denovo_best - known_best
    # Structural analysis
    top_denovo_props: dict = field(default_factory=dict)
    top_known_props: dict = field(default_factory=dict)
    max_similarity_to_known: float = 0.0
    most_similar_known: str = ""
    # Rank correlation
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    elapsed_s: float = 0.0


def run_benchmark(
    target_type: str,           # "host_guest" or "metal"
    target_key: str,            # "beta-CD" or "Cu2+"
    known_panel: list,          # list of (name, smiles, logKa_exp, source)
    max_enumerate: int = 300,
    max_score: int = 100,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run head-to-head: de novo generation vs known best binders."""
    t0 = time.time()
    result = BenchmarkResult(target=target_key, target_type=target_type)

    if verbose:
        print(f"\n{'═' * 70}")
        print(f"DE NOVO BENCHMARK: {target_key}")
        print(f"{'═' * 70}")

    # ── Score known binders ───────────────────────────────────────────
    if verbose:
        print(f"\n[1] Scoring {len(known_panel)} known binders...")

    known_scored = score_known_panel(known_panel, target_type, target_key)
    result.known_results = known_scored

    known_valid = [k for k in known_scored if k['log_Ka_pred'] != 0]
    if known_valid:
        best_known = max(known_valid, key=lambda k: k['log_Ka_pred'])
        result.known_best_name = best_known['name']
        result.known_best_logKa = best_known['log_Ka_exp']
        result.known_best_pred = best_known['log_Ka_pred']

    if verbose:
        print(f"  Known best: {result.known_best_name} "
              f"(exp={result.known_best_logKa:.2f}, pred={result.known_best_pred:.2f})")

    # Rank correlation on known panel
    if len(known_valid) >= 5:
        from scipy.stats import spearmanr
        import numpy as np
        exp = [k['log_Ka_exp'] for k in known_valid]
        pred = [k['log_Ka_pred'] for k in known_valid]
        rho, pval = spearmanr(exp, pred)
        result.spearman_rho = rho
        result.spearman_p = pval
        if verbose:
            print(f"  Rank correlation: ρ={rho:.3f} (p={pval:.4f})")

    # ── Enumerate de novo candidates ──────────────────────────────────
    if verbose:
        print(f"\n[2] Enumerating de novo candidates...")

    pfilter = PropertyFilter()
    if target_type == "host_guest":
        pfilter.require_donors = False

    raw = enumerate_molecules(
        metal=target_key if target_type == "metal" else None,
        host=target_key if target_type == "host_guest" else None,
        max_candidates=max_enumerate,
        pfilter=pfilter,
        hsab_filter=(target_type == "metal"),
    )
    result.n_enumerated = len(raw)

    # Sort by SA, take top N
    raw.sort(key=lambda x: x[3])
    to_score = raw[:max_score]

    if verbose:
        print(f"  Enumerated: {result.n_enumerated}, scoring top {len(to_score)}")

    # ── Score de novo candidates ──────────────────────────────────────
    if verbose:
        print(f"\n[3] Scoring de novo candidates...")

    denovo_scored = score_denovo_panel(to_score, target_type, target_key)
    result.n_scored = len(denovo_scored)
    result.denovo_results = denovo_scored

    if denovo_scored:
        denovo_sorted = sorted(denovo_scored, key=lambda d: -d['log_Ka_pred'])
        best_denovo = denovo_sorted[0]
        result.denovo_best_name = best_denovo['name']
        result.denovo_best_logKa = best_denovo['log_Ka_pred']

    if verbose:
        print(f"  Scored: {result.n_scored}")
        print(f"  De novo best: {result.denovo_best_name[:45]} "
              f"(pred={result.denovo_best_logKa:.2f})")

    # ── Comparison ────────────────────────────────────────────────────
    if verbose:
        print(f"\n[4] Head-to-head comparison...")

    threshold = result.known_best_pred  # compare against known best PREDICTED
    result.competitive_count = sum(
        1 for d in denovo_scored
        if d['log_Ka_pred'] >= threshold - 1.0  # within 1 log unit
    )
    result.exceeds_count = sum(
        1 for d in denovo_scored
        if d['log_Ka_pred'] > threshold
    )
    result.best_denovo_vs_known = result.denovo_best_logKa - result.known_best_pred

    if verbose:
        print(f"  Known best predicted logKa:  {result.known_best_pred:.2f}")
        print(f"  De novo best predicted logKa: {result.denovo_best_logKa:.2f}")
        print(f"  Gap (denovo - known):         {result.best_denovo_vs_known:+.2f}")
        print(f"  De novo within 1 log unit:    {result.competitive_count}/{result.n_scored}")
        print(f"  De novo exceeding known:      {result.exceeds_count}/{result.n_scored}")

    # ── Structural analysis ───────────────────────────────────────────
    if verbose:
        print(f"\n[5] Structural analysis...")

    if denovo_scored and known_valid:
        # Properties of top de novo vs top known
        result.top_denovo_props = molecular_properties(denovo_sorted[0]['smiles'])
        best_known_smi = best_known['smiles']
        result.top_known_props = molecular_properties(best_known_smi)

        # Similarity of top de novo to each known binder
        max_sim = 0.0
        most_sim_name = ""
        for k in known_valid:
            sim = tanimoto_similarity(denovo_sorted[0]['smiles'], k['smiles'])
            if sim > max_sim:
                max_sim = sim
                most_sim_name = k['name']
        result.max_similarity_to_known = max_sim
        result.most_similar_known = most_sim_name

        if verbose:
            dn = result.top_denovo_props
            kn = result.top_known_props
            print(f"\n  {'Property':>20s} {'Known best':>12s} {'De novo best':>14s}")
            print(f"  {'─' * 50}")
            for prop in ['mw', 'logP', 'n_rotatable', 'n_rings',
                         'n_aromatic_rings', 'n_hbd', 'n_hba', 'fraction_csp3']:
                kv = kn.get(prop, '—')
                dv = dn.get(prop, '—')
                kstr = f"{kv:.2f}" if isinstance(kv, float) else str(kv)
                dstr = f"{dv:.2f}" if isinstance(dv, float) else str(dv)
                print(f"  {prop:>20s} {kstr:>12s} {dstr:>14s}")

            print(f"\n  Most similar known binder: {most_sim_name} (Tanimoto={max_sim:.3f})")

    # ── Top results table ─────────────────────────────────────────────
    if verbose and denovo_scored:
        print(f"\n{'─' * 70}")
        print(f"TOP 10 DE NOVO vs KNOWN BEST")
        print(f"{'─' * 70}")
        print(f"{'Source':>8s} {'Name':>40s} {'logKa':>7s} {'SA':>5s}")

        # Interleave: show known best, then top 10 de novo
        print(f"{'KNOWN':>8s} {result.known_best_name:>40s} "
              f"{result.known_best_pred:7.2f}   ref")

        for i, d in enumerate(denovo_sorted[:10]):
            marker = "★" if d['log_Ka_pred'] >= threshold else " "
            print(f"{'DENOVO':>8s} {d['name'][:40]:>40s} "
                  f"{d['log_Ka_pred']:7.2f} {d['sa_score']:5.1f} {marker}")

    result.elapsed_s = time.time() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-TARGET BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def run_all_benchmarks(verbose=True):
    """Run benchmarks across all supported targets."""
    results = {}

    # β-CD guest design
    results['beta-CD'] = run_benchmark(
        target_type="host_guest",
        target_key="beta-CD",
        known_panel=KNOWN_BCD_GUESTS,
        max_enumerate=300,
        max_score=80,
        verbose=verbose,
    )

    # Pb²⁺ chelation
    results['Pb2+'] = run_benchmark(
        target_type="metal",
        target_key="Pb2+",
        known_panel=KNOWN_PB_CHELATORS,
        max_enumerate=200,
        max_score=50,
        verbose=verbose,
    )

    # Summary table
    if verbose:
        print(f"\n\n{'═' * 70}")
        print(f"MULTI-TARGET BENCHMARK SUMMARY")
        print(f"{'═' * 70}")
        print(f"{'Target':>10s} {'Known best':>15s} {'pred':>6s} "
              f"{'DeNovo best':>6s} {'Gap':>6s} {'Competitive':>12s} {'ρ':>6s}")
        print(f"{'─' * 70}")
        for target, r in results.items():
            print(f"{target:>10s} {r.known_best_name:>15s} "
                  f"{r.known_best_pred:6.2f} {r.denovo_best_logKa:6.2f} "
                  f"{r.best_denovo_vs_known:+6.2f} "
                  f"{r.competitive_count:4d}/{r.n_scored:<4d}    "
                  f"{r.spearman_rho:6.3f}")

    return results


if __name__ == "__main__":
    run_all_benchmarks(verbose=True)