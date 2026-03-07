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
import numpy as np
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
    source: str = ""                     # "library", "generated", or ""

    # 3D geometry fields (populated by rerank_3d)
    fidelity_3d: float = 0.0            # 0-1, combined 3D match quality
    rmsd_3d: float = 999.0              # RMSD of donors to ideal pocket (Å)
    strain_kJ: float = 0.0              # strain energy to adopt binding geometry
    preorganization: float = 0.0        # 0-1, preorganization score
    composite_3d: float = 0.0           # blended 2D + 3D score

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
# 3D RERANKING
# ═══════════════════════════════════════════════════════════════════════════

def rerank_3d(candidates, target_metal, donor_subtypes=None,
              geometry="auto", n_conformers=30,
              weight_2d=0.5, weight_3d=0.5):
    """Rerank candidates by 3D binding geometry quality.

    Takes existing 2D-scored candidates and adds 3D assessment:
    conformer generation → donor locator → Kabsch alignment →
    strain energy → preorganization → blended composite score.

    Args:
        candidates:      list of ScoredCandidate (from any source)
        target_metal:    e.g. "Cu2+"
        donor_subtypes:  list of expected donors. If None, inferred per candidate.
        geometry:        coordination geometry
        n_conformers:    conformers per molecule
        weight_2d:       weight for 2D log Ka in composite (0-1)
        weight_3d:       weight for 3D fidelity in composite (0-1)

    Returns:
        Same candidates with 3D fields populated, sorted by composite_3d.
    """
    from core.conformer_3d import score_3d
    from core.ideal_pocket import compute_ideal_pocket

    # Compute ideal pocket once (if donor_subtypes provided)
    ideal_pos = None
    if donor_subtypes:
        pocket = compute_ideal_pocket(target_metal, donor_subtypes, geometry)
        ideal_pos = np.array([d.position_A for d in pocket.donors])

    # Normalize 2D scores for blending
    log_kas = [c.log_Ka_pred for c in candidates]
    max_ka = max(log_kas) if log_kas else 1.0
    min_ka = min(log_kas) if log_kas else 0.0
    ka_range = max(max_ka - min_ka, 0.01)

    for c in candidates:
        try:
            # Infer donors per candidate if not provided globally
            if donor_subtypes:
                dsubs = donor_subtypes
                ip = ideal_pos
            else:
                # Use auto_descriptor to get donor subtypes
                from core.auto_descriptor import from_smiles
                uc = from_smiles(c.smiles, metal=target_metal)
                dsubs = uc.donor_subtypes or ["N_amine"] * 2
                pocket = compute_ideal_pocket(target_metal, dsubs, geometry)
                ip = np.array([d.position_A for d in pocket.donors])

            r = score_3d(c.smiles, target_metal, dsubs,
                         ideal_positions=ip,
                         geometry=geometry,
                         n_conformers=n_conformers)

            c.fidelity_3d = r.fidelity_score
            c.rmsd_3d = r.best_rmsd_A
            c.strain_kJ = r.strain_energy_kJ
            c.preorganization = r.preorganization_score

        except Exception:
            c.fidelity_3d = 0.0
            c.rmsd_3d = 999.0
            c.strain_kJ = 0.0
            c.preorganization = 0.0

        # Composite: blend normalized 2D score + 3D fidelity
        norm_2d = (c.log_Ka_pred - min_ka) / ka_range if ka_range > 0 else 0.5
        c.composite_3d = weight_2d * norm_2d + weight_3d * c.fidelity_3d

    # Sort by composite_3d
    candidates.sort(key=lambda c: c.composite_3d, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED DISCOVERY: LIBRARY SCREEN + DE NOVO GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def discover_binders(target_metal, interferents=None, pH=7.4,
                     top_n=20, library_categories=None,
                     generate=True, max_generate=200, max_score_generated=50,
                     seed_smiles=None, hop=True,
                     sa_penalty_weight=0.3, hsab_filter=True,
                     score_3d_flag=False, donor_subtypes_3d=None,
                     geometry_3d="auto", n_conformers_3d=20):
    """Screen existing library AND generate novel candidates in one call.

    Combines screen_ligand_library() + de_novo_generator + scaffold
    hopping/bioisosteric replacement in a single ranked output.

    Args:
        target_metal: e.g. "Pb2+", "Cu2+", "Fe3+"
        interferents: optional list of competing metals for selectivity
        pH: working pH
        top_n: how many final candidates to return
        library_categories: filter library by category (None = all)
        generate: if True, also run de novo generation
        max_generate: max molecules to enumerate in generator
        max_score_generated: max generated molecules to score
        seed_smiles: optional SMILES to scaffold-hop / bioisostere from
        hop: if True and seed_smiles provided, run scaffold hop + bioisostere
        sa_penalty_weight: SA penalty weight for generated candidates
        hsab_filter: apply HSAB pre-filter to generator

    Returns:
        RankingResult with mixed library + generated + hopped candidates,
        each tagged with source="library", "generated", or "hopped"
    """
    t0 = time.time()
    all_candidates = []
    all_errors = []
    n_library = 0
    n_generated = 0

    # ── Phase 1: Library screen ──────────────────────────────────────
    try:
        if interferents:
            lib_result = screen_ligand_library(
                target_metal, interferents=interferents,
                pH=pH, categories=library_categories, top_n=9999)
        else:
            lib_result = screen_ligand_library(
                target_metal, pH=pH,
                categories=library_categories, top_n=9999)

        for c in lib_result.candidates:
            c.source = "library"
        all_candidates.extend(lib_result.candidates)
        all_errors.extend(lib_result.errors)
        n_library = lib_result.n_scored
    except Exception as e:
        all_errors.append(("library_screen", str(e)))

    # ── Phase 2: De novo generation ──────────────────────────────────
    if generate:
        try:
            from core.de_novo_generator import (
                generate_candidates as _gen_metal,
                generate_and_screen as _gen_screen,
            )

            if interferents:
                gen_result = _gen_screen(
                    target_metal, interferents, pH=pH,
                    max_candidates=max_generate,
                    max_scored=max_score_generated,
                    sa_penalty_weight=sa_penalty_weight,
                    hsab_filter=hsab_filter)
            else:
                gen_result = _gen_metal(
                    target_metal, pH=pH,
                    max_candidates=max_generate,
                    max_scored=max_score_generated,
                    sa_penalty_weight=sa_penalty_weight,
                    hsab_filter=hsab_filter)

            for c in gen_result.candidates:
                c.source = "generated"
            all_candidates.extend(gen_result.candidates)
            all_errors.extend(gen_result.errors)
            n_generated = gen_result.n_scored
        except Exception as e:
            all_errors.append(("de_novo_generation", str(e)))

    # ── Phase 2b: Scaffold hopping / bioisosteric replacement ──────
    if seed_smiles and hop:
        try:
            from core.de_novo_generator import hop_and_score
            hop_result = hop_and_score(
                seed_smiles, metal=target_metal, pH=pH,
                mode="both", max_candidates=max_generate // 2,
                max_scored=max_score_generated // 2,
                interferents=interferents,
                sa_penalty_weight=sa_penalty_weight)
            for c in hop_result.candidates:
                c.source = "hopped"
            all_candidates.extend(hop_result.candidates)
            all_errors.extend(hop_result.errors)
        except Exception as e:
            all_errors.append(("scaffold_hop", str(e)))

    # ── Phase 3: Deduplicate by canonical SMILES ─────────────────────
    seen = set()
    unique = []
    for c in all_candidates:
        key = c.smiles
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    # ── Phase 4: Unified ranking ─────────────────────────────────────
    mode = "selectivity" if interferents else "metal"
    if mode == "selectivity":
        unique.sort(key=lambda c: c.min_gap, reverse=True)
    else:
        unique.sort(key=lambda c: c.log_Ka_pred, reverse=True)

    unique = unique[:top_n]
    for i, c in enumerate(unique):
        c.rank = i + 1
        if interferents and not c.grade:
            c.grade = _grade(c.min_gap)

    # ── Phase 5: Optional 3D reranking ────────────────────────────────
    if score_3d_flag:
        unique = rerank_3d(
            unique, target_metal,
            donor_subtypes=donor_subtypes_3d,
            geometry=geometry_3d,
            n_conformers=n_conformers_3d,
        )
        mode = f"discover_{mode}_3d"
    else:
        mode = f"discover_{mode}"

    return RankingResult(
        target=target_metal,
        mode=mode,
        interferents=list(interferents) if interferents else [],
        candidates=unique,
        n_input=n_library + n_generated,
        n_scored=len(unique),
        n_failed=len(all_errors),
        elapsed_s=time.time() - t0,
        errors=all_errors,
    )


def discover_guests(host_key, top_n=20, generate=True,
                    max_generate=200, max_score_generated=50,
                    sa_penalty_weight=0.3):
    """Screen existing library AND generate novel guests for a host.

    Args:
        host_key: HOST_REGISTRY key, e.g. "beta-CD", "CB7"
        top_n: how many final candidates to return
        generate: if True, also run de novo generation
        max_generate, max_score_generated, sa_penalty_weight: generator params

    Returns:
        RankingResult with mixed library + generated guests
    """
    t0 = time.time()
    all_candidates = []
    all_errors = []
    n_library = 0
    n_generated = 0

    # ── Phase 1: Library screen ──────────────────────────────────────
    try:
        from knowledge.ligand_library import LIGAND_DB
        smiles_list = [lig.smiles for lig in LIGAND_DB]
        names_list = [lig.name for lig in LIGAND_DB]
        lib_result = rank_hosts(host_key, smiles_list, names=names_list)
        for c in lib_result.candidates:
            c.source = "library"
        all_candidates.extend(lib_result.candidates)
        all_errors.extend(lib_result.errors)
        n_library = lib_result.n_scored
    except Exception as e:
        all_errors.append(("library_screen", str(e)))

    # ── Phase 2: De novo generation ──────────────────────────────────
    if generate:
        try:
            from core.de_novo_generator import generate_for_host as _gen_host
            gen_result = _gen_host(
                host_key, max_candidates=max_generate,
                max_scored=max_score_generated,
                sa_penalty_weight=sa_penalty_weight)
            for c in gen_result.candidates:
                c.source = "generated"
            all_candidates.extend(gen_result.candidates)
            all_errors.extend(gen_result.errors)
            n_generated = gen_result.n_scored
        except Exception as e:
            all_errors.append(("de_novo_generation", str(e)))

    # ── Phase 3: Deduplicate + rank ──────────────────────────────────
    seen = set()
    unique = []
    for c in all_candidates:
        if c.smiles in seen:
            continue
        seen.add(c.smiles)
        unique.append(c)

    unique.sort(key=lambda c: c.log_Ka_pred, reverse=True)
    unique = unique[:top_n]
    for i, c in enumerate(unique):
        c.rank = i + 1

    return RankingResult(
        target=host_key,
        mode="discover_host_guest",
        candidates=unique,
        n_input=n_library + n_generated,
        n_scored=len(unique),
        n_failed=len(all_errors),
        elapsed_s=time.time() - t0,
        errors=all_errors,
    )


def print_discovery(result, top_n=20, verbose=False):
    """Pretty-print a discovery result showing source tags."""
    print()
    print(f"  MABE Discovery — {result.mode}")
    print(f"  Target: {result.target}")
    if result.interferents:
        print(f"  Interferents: {', '.join(result.interferents)}")
    print(f"  Scored: {result.n_scored} ({result.n_failed} failed) "
          f"in {result.elapsed_s:.1f}s")
    print()

    if not result.candidates:
        print("  No candidates found.")
        return

    shown = result.candidates[:top_n]

    if "selectivity" in result.mode:
        print(f"  {'#':>3s}  {'Src':5s}  {'Grade':5s}  {'logKa':>6s}  "
              f"{'MinGap':>7s}  Name")
        print(f"  {'─'*65}")
        for c in shown:
            src = c.source[:3].upper() if c.source else "???"
            print(f"  {c.rank:3d}  {src:5s}  {c.grade:^5s}  "
                  f"{c.log_Ka_pred:+6.1f}  {c.min_gap:+7.1f}  "
                  f"{c.name[:35]}")
    else:
        print(f"  {'#':>3s}  {'Src':5s}  {'logKa':>7s}  Name")
        print(f"  {'─'*55}")
        for c in shown:
            src = c.source[:3].upper() if c.source else "???"
            print(f"  {c.rank:3d}  {src:5s}  {c.log_Ka_pred:+7.2f}  "
                  f"{c.name[:40]}")
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

    # Test 5: Unified discovery — metal
    print("Discovery: Cu2+ (library + generated)...")
    r5 = discover_binders("Cu2+", max_generate=50, max_score_generated=10,
                           top_n=10)
    print_discovery(r5)

    # Test 6: Unified discovery — selectivity
    print("Discovery: Pb2+ selectivity over Ca2+/Mg2+ (library + generated)...")
    r6 = discover_binders("Pb2+", interferents=["Ca2+", "Mg2+"], pH=5.0,
                           max_generate=50, max_score_generated=10, top_n=10)
    print_discovery(r6)

    # Test 7: Unified discovery — host-guest
    print("Discovery: beta-CD guests (library + generated)...")
    r7 = discover_guests("beta-CD", max_generate=50, max_score_generated=10,
                          top_n=10)
    print_discovery(r7)
