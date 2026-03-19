"""
core/aacr_pipeline.py — AACR Rare Cancer: End-to-End GalNAc Binder Pipeline

Enumerates de novo candidates from the fragment library,
scores against GalNAc (Tn antigen), runs Pareto optimization,
and filters for scaffold-attachment compatibility.

Entry point: run_aacr_pipeline() → full results with figures data.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.de_novo_generator import (
    enumerate_molecules, BACKBONE_LIBRARY, ARM_LIBRARY,
    PropertyFilter, sa_score,
)
from core.galnac_binder_scorer import (
    score_galnac_binder, GalNAcBinderScore,
    generate_decoy_panel, KNOWN_GALNAC_BINDERS,
    score_candidate_panel, extract_receptor_features,
)
from core.pareto import Objective, pareto_rank


# ═══════════════════════════════════════════════════════════════════════════
# SUGAR-BINDING PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════

def sugar_binding_prefilter(smiles: str) -> bool:
    """Fast pre-filter: does this molecule have features for sugar binding?

    Requires:
    - ≥2 H-bond donors (for OH recognition)
    - MW 150-800 (pyranose-compatible cavity)
    - ≤15 rotatable bonds (not too floppy)
    """
    from rdkit import Chem
    from rdkit.Chem import Lipinski, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    n_hbd = Lipinski.NumHDonors(mol)
    if n_hbd < 2:
        return False

    mw = Descriptors.MolWt(mol)
    if mw < 150 or mw > 800:
        return False

    n_rot = Descriptors.NumRotatableBonds(mol)
    if n_rot > 15:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# PARETO OBJECTIVES FOR GALNAC
# ═══════════════════════════════════════════════════════════════════════════

GALNAC_OBJECTIVES = [
    Objective(
        name="galnac_affinity",
        extract=lambda c: -c.dg_galnac,  # more negative dG = stronger; negate for maximization
        maximize=True,
    ),
    Objective(
        name="selectivity",
        extract=lambda c: c.min_selectivity,  # positive = selective over competitors
        maximize=True,
    ),
    Objective(
        name="synthesizability",
        extract=lambda c: c.features.sa_score,
        maximize=False,  # lower SA = easier to synthesize
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# PARETO-COMPATIBLE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GalNAcParetoCandidate:
    """Wraps GalNAcBinderScore with fields the Pareto ranker needs."""
    score: GalNAcBinderScore
    # Mirror fields for Pareto objective extraction
    dg_galnac: float = 0.0
    min_selectivity: float = 0.0
    sa_score_val: float = 10.0
    log_Ka_pred: float = 0.0
    min_gap: float = 0.0
    features: object = None
    # Pareto results
    pareto_front_idx: int = -1
    pareto_rank: int = 0
    pareto_crowding: float = 0.0
    # Metadata
    smiles: str = ""
    name: str = ""
    backbone_name: str = ""
    arm_names: list = field(default_factory=list)
    attachability: float = 0.0

    @classmethod
    def from_score(cls, score, bb_name="", arm_names=None):
        return cls(
            score=score,
            dg_galnac=score.dg_galnac,
            min_selectivity=score.min_selectivity,
            sa_score_val=score.features.sa_score,
            log_Ka_pred=score.log_Ka_est,
            min_gap=score.min_selectivity,
            features=score.features,
            smiles=score.smiles,
            name=score.name,
            backbone_name=bb_name,
            arm_names=arm_names or [],
            attachability=score.attachability,
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AACRPipelineResult:
    """Full output of the AACR GalNAc pipeline."""
    # Enumeration
    n_enumerated: int = 0
    n_prefiltered: int = 0
    n_scored: int = 0
    n_failed: int = 0
    # Pareto
    n_pareto_front: int = 0
    pareto_candidates: list = field(default_factory=list)
    # Filters
    n_selective: int = 0         # min_selectivity > 0
    n_attachable: int = 0        # attachability > 0.5
    n_nominated: int = 0         # pass both filters
    nominated: list = field(default_factory=list)
    # Enrichment (vs decoys)
    enrichment_5pct: float = 0.0
    enrichment_10pct: float = 0.0
    mean_dg_known: float = 0.0
    mean_dg_decoy: float = 0.0
    mean_dg_generated: float = 0.0
    # Timing
    elapsed_s: float = 0.0
    errors: list = field(default_factory=list)


def run_aacr_pipeline(
    max_enumerate: int = 500,
    max_score: int = 200,
    selectivity_threshold: float = 0.0,  # kJ/mol, positive = selective
    attachability_threshold: float = 0.4,
    verbose: bool = True,
) -> AACRPipelineResult:
    """Run the complete AACR GalNAc binder discovery pipeline.

    Steps:
    1. Enumerate candidates from fragment library (42 backbones × 74 arms)
    2. Pre-filter for sugar-binding compatibility
    3. Score top candidates against GalNAc + selectivity panel
    4. Pareto rank (affinity × selectivity × synthesizability)
    5. Filter for attachability (scaffold-display compatibility)
    6. Compute enrichment vs decoy panel
    7. Nominate top candidates

    Returns AACRPipelineResult with all numbers for the abstract.
    """
    t0 = time.time()
    result = AACRPipelineResult()

    if verbose:
        print("=" * 70)
        print("AACR GalNAc Binder Pipeline — De Novo Generation")
        print("=" * 70)

    # ── Step 1: Enumerate ──────────────────────────────────────────────
    if verbose:
        print(f"\n[1] Enumerating from {len(BACKBONE_LIBRARY)} backbones × "
              f"{len(ARM_LIBRARY)} arms...")

    pfilter = PropertyFilter()
    pfilter.require_donors = False  # We'll apply sugar-specific filter separately

    raw = enumerate_molecules(
        metal=None, host=None,
        max_candidates=max_enumerate,
        pfilter=pfilter,
        hsab_filter=False,
    )
    result.n_enumerated = len(raw)
    if verbose:
        print(f"    Enumerated: {result.n_enumerated}")

    # ── Step 2: Sugar-binding pre-filter ───────────────────────────────
    if verbose:
        print(f"\n[2] Pre-filtering for sugar-binding compatibility...")

    filtered = [(smi, bb, arms, sa) for smi, bb, arms, sa in raw
                if sugar_binding_prefilter(smi)]
    result.n_prefiltered = len(filtered)
    if verbose:
        print(f"    Passed pre-filter: {result.n_prefiltered} / {result.n_enumerated}")

    # Sort by SA score (easiest first), take top N
    filtered.sort(key=lambda x: x[3])
    to_score = filtered[:max_score]

    # ── Step 3: Score against GalNAc + selectivity panel ──────────────
    if verbose:
        print(f"\n[3] Scoring {len(to_score)} candidates against GalNAc + "
              f"selectivity panel...")

    scored = []
    errors = []
    for i, (smi, bb, arm_names, sa) in enumerate(to_score):
        try:
            name = f"{bb}+{'|'.join(arm_names)}"
            gs = score_galnac_binder(smi, name=name, include_selectivity=True)
            if gs.valid:
                pc = GalNAcParetoCandidate.from_score(gs, bb_name=bb,
                                                       arm_names=arm_names)
                scored.append(pc)
        except Exception as e:
            errors.append((smi, str(e)))

        if verbose and (i + 1) % 50 == 0:
            print(f"    Scored {i+1}/{len(to_score)}...")

    result.n_scored = len(scored)
    result.n_failed = len(errors)
    result.errors = errors
    if verbose:
        print(f"    Scored: {result.n_scored}, Failed: {result.n_failed}")

    if not scored:
        result.elapsed_s = time.time() - t0
        return result

    # ── Step 4: Pareto optimization ───────────────────────────────────
    if verbose:
        print(f"\n[4] Pareto optimization (3 objectives)...")

    pareto_result = pareto_rank(scored, objectives=GALNAC_OBJECTIVES)

    # Apply Pareto results back
    for pc in pareto_result.candidates:
        c = scored[pc.index]
        c.pareto_front_idx = pc.front
        c.pareto_rank = pc.pareto_rank
        c.pareto_crowding = pc.crowding

    # Sort by Pareto rank
    scored.sort(key=lambda c: c.pareto_rank if c.pareto_rank > 0 else 9999)

    n_front0 = sum(1 for c in scored if c.pareto_front_idx == 0)
    result.n_pareto_front = n_front0
    result.pareto_candidates = scored

    if verbose:
        print(f"    Pareto front (rank 0): {n_front0} candidates")
        print(f"    Total ranked: {len(scored)}")

    # ── Step 5: Filter for selectivity + attachability ────────────────
    if verbose:
        print(f"\n[5] Filtering: selectivity > {selectivity_threshold:.1f}, "
              f"attachability > {attachability_threshold:.2f}...")

    selective = [c for c in scored if c.min_selectivity > selectivity_threshold]
    result.n_selective = len(selective)

    attachable = [c for c in scored if c.attachability > attachability_threshold]
    result.n_attachable = len(attachable)

    nominated = [c for c in scored
                 if c.min_selectivity > selectivity_threshold
                 and c.attachability > attachability_threshold]
    nominated.sort(key=lambda c: c.score.composite_score)
    result.n_nominated = len(nominated)
    result.nominated = nominated[:20]  # top 20

    if verbose:
        print(f"    Selective: {result.n_selective}")
        print(f"    Attachable: {result.n_attachable}")
        print(f"    Nominated (both): {result.n_nominated}")

    # ── Step 6: Enrichment vs known + decoys ──────────────────────────
    if verbose:
        print(f"\n[6] Computing enrichment...")

    known_results = score_candidate_panel(KNOWN_GALNAC_BINDERS)
    decoy_panel = generate_decoy_panel(40)
    decoy_results = score_candidate_panel(decoy_panel)

    known_dgs = [r.dg_galnac for r in known_results if r.valid]
    decoy_dgs = [r.dg_galnac for r in decoy_results if r.valid]
    generated_dgs = [c.dg_galnac for c in scored]

    if known_dgs:
        result.mean_dg_known = sum(known_dgs) / len(known_dgs)
    if decoy_dgs:
        result.mean_dg_decoy = sum(decoy_dgs) / len(decoy_dgs)
    if generated_dgs:
        result.mean_dg_generated = sum(generated_dgs) / len(generated_dgs)

    # Combined enrichment: known binders + generated + decoys
    all_items = []
    for r in known_results:
        if r.valid:
            all_items.append((r.dg_galnac, 'known', r.name))
    for r in decoy_results:
        if r.valid:
            all_items.append((r.dg_galnac, 'decoy', r.name))

    all_items.sort(key=lambda x: x[0])
    n_known = sum(1 for _, t, _ in all_items if t == 'known')

    for pct, attr in [(0.05, 'enrichment_5pct'), (0.10, 'enrichment_10pct')]:
        n_top = max(1, int(len(all_items) * pct))
        top = all_items[:n_top]
        n_known_in_top = sum(1 for _, t, _ in top if t == 'known')
        expected = n_known * pct
        ef = n_known_in_top / max(expected, 0.001)
        setattr(result, attr, ef)

    if verbose:
        print(f"    Mean dG (known binders): {result.mean_dg_known:+.2f} kJ/mol")
        print(f"    Mean dG (decoys):        {result.mean_dg_decoy:+.2f} kJ/mol")
        print(f"    Mean dG (generated):     {result.mean_dg_generated:+.2f} kJ/mol")
        print(f"    Enrichment at 5%:  {result.enrichment_5pct:.1f}x")
        print(f"    Enrichment at 10%: {result.enrichment_10pct:.1f}x")

    # ── Results summary ───────────────────────────────────────────────
    result.elapsed_s = time.time() - t0

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PIPELINE SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Enumerated:           {result.n_enumerated}")
        print(f"  Pre-filtered:         {result.n_prefiltered}")
        print(f"  Scored:               {result.n_scored}")
        print(f"  Pareto front:         {result.n_pareto_front}")
        print(f"  Selective:            {result.n_selective}")
        print(f"  Attachable:           {result.n_attachable}")
        print(f"  Nominated:            {result.n_nominated}")
        print(f"  Elapsed:              {result.elapsed_s:.1f}s")

        if result.nominated:
            print(f"\n{'─' * 70}")
            print(f"TOP NOMINATED CANDIDATES")
            print(f"{'─' * 70}")
            print(f"{'Rank':>4s} {'Name':40s} {'dG':>7s} {'logKa':>6s} "
                  f"{'Sel':>5s} {'Att':>4s} {'SA':>5s} {'Front':>5s}")
            for i, c in enumerate(result.nominated[:15]):
                print(f"{i+1:4d} {c.name[:40]:40s} {c.dg_galnac:+7.1f} "
                      f"{c.log_Ka_pred:6.1f} {c.min_selectivity:+5.1f} "
                      f"{c.attachability:4.2f} {c.sa_score_val:5.1f} "
                      f"{c.pareto_front_idx:5d}")

        # Backbone distribution in nominees
        if result.nominated:
            bb_counts = {}
            for c in result.nominated:
                bb = c.backbone_name
                bb_counts[bb] = bb_counts.get(bb, 0) + 1
            print(f"\n  Backbone distribution in nominees:")
            for bb, n in sorted(bb_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"    {bb:35s}: {n}")

        # Arm distribution
        if result.nominated:
            arm_counts = {}
            for c in result.nominated:
                for a in c.arm_names:
                    arm_counts[a] = arm_counts.get(a, 0) + 1
            print(f"\n  Arm distribution in nominees:")
            for arm, n in sorted(arm_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"    {arm:35s}: {n}")

        # ── Step 7: Scaffold ranking + construct spec for top nominee ──
        if result.nominated:
            try:
                from core.scaffold_realization_ranker import rank_scaffolds, scaffold_comparison_table
                from core.spacing_valency_optimizer import full_construct_spec

                top = result.nominated[0]
                feat = top.features

                print(f"\n{'═' * 70}")
                print(f"SCAFFOLD RANKING — {top.name[:50]}")
                print(f"{'═' * 70}")

                scaffolds = rank_scaffolds(
                    binder_mw=feat.mw,
                    binder_n_hbd=feat.n_hbd,
                    binder_n_aromatic=feat.n_aromatic_rings,
                    target_valency=10,
                )
                print(scaffold_comparison_table(scaffolds[:5]))

                constructs = full_construct_spec(
                    binder_name=top.name,
                    binder_mw=feat.mw,
                    binder_n_rotatable=feat.n_rotatable,
                    scaffold_results=scaffolds[:5],
                )

                print(f"\n{'─' * 70}")
                print(f"CONSTRUCT SPECIFICATIONS (top 5 scaffolds)")
                print(f"{'─' * 70}")
                print(f"{'Scaffold':30s} {'Val':>4s} {'Space':>6s} "
                      f"{'BCR':>5s} {'DC':>8s} {'Adj':>4s} {'$/mg':>6s}")
                for c in constructs:
                    print(f"{c['scaffold_name'][:30]:30s} "
                          f"{c['binder_valency']:4d} {c['binder_spacing_nm']:5.1f}nm "
                          f"{c['predicted_bcr_activation']:5.3f} "
                          f"{c['predicted_dc_activation']:>8s} "
                          f"{c['adjuvant_valency']:4d} {c['est_cost_per_mg']:6.1f}")
            except Exception as e:
                print(f"\n  [Scaffold ranking skipped: {e}]")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY HEATMAP DATA
# ═══════════════════════════════════════════════════════════════════════════

def selectivity_heatmap_data(candidates: List[GalNAcParetoCandidate],
                              max_rows: int = 20) -> dict:
    """Extract data for selectivity heatmap figure.

    Returns dict with:
        names: list of candidate names
        sugars: list of sugar panel names
        matrix: 2D list of dG values (rows=candidates, cols=sugars)
        galnac_dgs: list of GalNAc dG values
    """
    top = candidates[:max_rows]
    sugars = ['GalNAc', 'Glc', 'Man', 'Gal', 'GlcNAc', 'Fuc']
    names = [c.name[:35] for c in top]
    matrix = []
    galnac_dgs = []

    for c in top:
        row = [c.dg_galnac]
        galnac_dgs.append(c.dg_galnac)
        for sugar in sugars[1:]:
            row.append(c.score.dg_panel.get(sugar, 0.0))
        matrix.append(row)

    return {
        'names': names,
        'sugars': sugars,
        'matrix': matrix,
        'galnac_dgs': galnac_dgs,
    }


if __name__ == "__main__":
    result = run_aacr_pipeline(
        max_enumerate=500,
        max_score=200,
        selectivity_threshold=0.0,
        attachability_threshold=0.4,
        verbose=True,
    )