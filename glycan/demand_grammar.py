"""
glycan/demand_grammar.py -- Physics demand vector → molecular construction.

The sugar target defines what the receptor MUST have (physics demand).
The grammar constructs molecules to satisfy that demand.

Pipeline:
  1. Sugar target → DemandVector (min aromatic area, HBD count, boronic need, etc.)
  2. DemandVector → scaffold filter (ring enumerator) + decorator filter
  3. Enumerate scaffold × decorator combinations
  4. Score each through glycan binder scorer
  5. Pareto-rank: affinity × selectivity × synthesizability

The demand vector is NOT a target profile to fit against -- it's a physics-derived
constraint that narrows the search space. The scorer does the actual ranking.

Cross-validation: every molecule produced by the grammar must be reproducible
by the ring enumerator (it's a subset of enumerator space, not a separate space).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from glycan.de_novo_binder import (
    get_sugar_target, SugarTarget,
    score_glycan_binder, selectivity_score,
    compute_receptor_descriptor,
    GlycanBinderCandidate, GlycanBinderResult,
)
from glycan.ring_enumerator import (
    list_ring_systems, enumerate_physics_filtered, PhysicsFilter,
    get_decorators, Decorator, DecoratedScaffold,
    DECORATOR_LIBRARY,
)


# ---------------------------------------------------------------------------
# Demand vector
# ---------------------------------------------------------------------------

@dataclass
class DemandVector:
    """Physics-derived demands for a receptor targeting a specific sugar."""
    target: str
    # Aromatic surface demand
    min_aromatic_atoms: int    # minimum aromatic atom count for CH-pi
    prefer_large_aromatic: bool  # >= 10 atoms (fused polycyclic)
    # H-bond demand
    optimal_hbd: int           # HBD count that balances HB gain vs desolvation
    hbd_strategy: str          # "none", "minimal", "moderate", "saturate"
    # Boronic acid demand
    boronic_useful: bool       # target has cis-1,2-diol
    boronic_preferred: bool    # cis-1,2-diol is best discrimination axis
    # Cavity demand
    target_volume_A3: float
    # Categories to prefer
    preferred_categories: List[str] = field(default_factory=list)
    # Selectivity notes
    selectivity_axis: str = ""  # what physical feature drives selectivity
    notes: str = ""


def compute_demand(target: str) -> DemandVector:
    """
    Derive the physics demand vector from sugar target descriptors.

    This is the core insight: the sugar's structure TELLS you what
    the receptor needs. You don't search blindly.
    """
    sugar = get_sugar_target(target)

    # -- Aromatic demand: driven by number of axial CH bonds --
    # More axial CH = more CH-pi contacts possible = bigger aromatic surface needed
    if sugar.n_axial_CH >= 3:
        min_arom = 10    # need large aromatic (anthracene-class)
        prefer_large = True
    elif sugar.n_axial_CH >= 2:
        min_arom = 6     # naphthalene-class sufficient
        prefer_large = False
    else:
        min_arom = 0     # CH-pi not the primary axis
        prefer_large = False

    # -- H-bond demand: the desolvation trap analysis --
    # Each sugar OH that gets buried costs +2.4 to +11.2 kJ/mol in desolvation.
    # Each H-bond recovers -2.25 kJ/mol.
    # Strategy depends on how many OHs the sugar exposes:
    total_OH = sugar.n_eq_OH + sugar.n_ax_OH + sugar.n_primary_OH

    if total_OH <= 2:
        # Few OHs: don't bother with H-bonds, go pure hydrophobic
        optimal_hbd = 0
        hbd_strategy = "none"
    elif total_OH <= 3 and sugar.ring_type == "furanose":
        # Furanose with few OHs: minimal HBD
        optimal_hbd = 1
        hbd_strategy = "minimal"
    else:
        # Standard pyranose (3-6 OHs): desolvation trap applies.
        # Either 0 HBD (Ke approach: pure CH-pi/boronic) or 4+ (Davis approach).
        # The valley between 1-3 HBD is universally bad for sugars.
        optimal_hbd = 0
        hbd_strategy = "bimodal"

    # -- Boronic acid demand --
    boronic_useful = sugar.has_cis_12_diol
    # Preferred when cis-1,2-diol is the strongest selectivity lever
    boronic_preferred = sugar.has_cis_12_diol and sugar.n_axial_CH < 2

    # -- Cavity demand --
    target_vol = sugar.molecular_volume_A3

    # -- Category preferences --
    cats = ["carbocyclic"]
    if sugar.n_NAc > 0:
        cats.append("N-hetero")  # N-heterocycles can interact with NAc
    if sugar.ring_type == "furanose":
        cats = ["carbocyclic", "N-hetero", "O-hetero"]

    # -- Selectivity axis --
    if sugar.n_axial_CH >= 3 and not sugar.has_cis_12_diol:
        sel_axis = "CH-pi (axial CH count)"
    elif sugar.has_cis_12_diol and sugar.n_axial_CH < 2:
        sel_axis = "boronic acid (cis-1,2-diol)"
    elif sugar.n_NAc > 0:
        sel_axis = "NAc accommodation (cavity size + HB)"
    else:
        sel_axis = "mixed (CH-pi + HB balance)"

    notes_parts = []
    if sugar.n_axial_CH >= 3:
        notes_parts.append("Best CH-pi target (all-eq or 3 axial CH)")
    if sugar.has_cis_12_diol:
        notes_parts.append("Boronic acid accessible")
    if sugar.n_NAc > 0:
        notes_parts.append("NAc group increases volume, desolvation")
    if sugar.ring_type == "furanose":
        notes_parts.append("Furanose: no alpha-face, CH-pi not available")

    return DemandVector(
        target=target,
        min_aromatic_atoms=min_arom,
        prefer_large_aromatic=prefer_large,
        optimal_hbd=optimal_hbd,
        hbd_strategy=hbd_strategy,
        boronic_useful=boronic_useful,
        boronic_preferred=boronic_preferred,
        target_volume_A3=target_vol,
        preferred_categories=cats,
        selectivity_axis=sel_axis,
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Grammar-driven generation
# ---------------------------------------------------------------------------

def _select_decorators(demand: DemandVector) -> List[List[Decorator]]:
    """
    Select decorator combinations based on demand vector.

    Returns list of (dec_for_site1, dec_for_site2) pairs.
    For bimodal strategy, returns two pools: hydrophobic-only and HBD-heavy.
    """
    all_decs = DECORATOR_LIBRARY

    if demand.hbd_strategy == "none":
        # Pure hydrophobic + acceptors only
        pool = [d for d in all_decs if d.n_hbd == 0 and not d.has_boronic]
        if demand.boronic_useful:
            pool += [d for d in all_decs if d.has_boronic]
        return [pool]

    elif demand.hbd_strategy == "minimal":
        # 0-1 HBD per site
        pool = [d for d in all_decs if d.n_hbd <= 1]
        if demand.boronic_useful:
            pool += [d for d in all_decs if d.has_boronic]
        return [pool]

    elif demand.hbd_strategy == "bimodal":
        # Two separate pools: Ke-style (0 HBD) and Davis-style (heavy HBD)
        ke_pool = [d for d in all_decs if d.n_hbd == 0 and not d.has_boronic]
        davis_pool = [d for d in all_decs if d.n_hbd >= 2 or d.n_urea >= 1]
        pools = [ke_pool, davis_pool]
        if demand.boronic_useful:
            boronic_pool = [d for d in all_decs if d.has_boronic]
            pools.append(boronic_pool)
        return pools

    else:  # saturate
        pool = [d for d in all_decs if d.n_hbd >= 1 or d.has_boronic]
        return [pool]


def generate_from_demand(
    target: str,
    max_candidates: int = 500,
    n_sites: int = 2,
    max_per_system: int = 10,
) -> GlycanBinderResult:
    """
    Physics-demand-driven glycan binder generation.

    1. Compute demand vector from sugar target
    2. Filter ring systems + decorators to match demand
    3. Enumerate scaffold × decorator combinations
    4. Score and Pareto-rank

    Returns same GlycanBinderResult as generate_glycan_binders().
    """
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required")

    from glycan.ring_enumerator import enumerate_physics_filtered
    from core.pareto import fast_non_dominated_sort
    from itertools import product as cartesian_product
    import random

    t0 = time.time()

    demand = compute_demand(target)
    sugar = get_sugar_target(target)

    # Build physics filter from demand
    pf = PhysicsFilter(
        min_aromatic_atoms=demand.min_aromatic_atoms,
        require_large_aromatic=demand.prefer_large_aromatic,
        min_rings=1 if demand.min_aromatic_atoms < 10 else 2,
        max_rings=4,
    )

    # Get scaffolds
    scaffolds = enumerate_physics_filtered(
        n_sites=n_sites, pfilter=pf,
        max_per_system=max_per_system, max_total=200,
    )

    if not scaffolds:
        # Relax filter
        pf.min_aromatic_atoms = 0
        pf.require_large_aromatic = False
        pf.min_rings = 1
        scaffolds = enumerate_physics_filtered(
            n_sites=n_sites, pfilter=pf,
            max_per_system=max_per_system, max_total=200,
        )

    # Get decorator pools
    dec_pools = _select_decorators(demand)

    competitors = [s for s in [
        "Glc", "Gal", "Man", "GalNAc", "GlcNAc", "Fuc", "Fru", "Neu5Ac"
    ] if s != target]

    candidates = []
    seen = set()
    n_enum = 0

    for scaffold in scaffolds:
        for pool in dec_pools:
            combos = list(cartesian_product(pool, repeat=n_sites))
            if len(combos) > 200:
                random.seed(42)
                combos = random.sample(combos, 200)

            for combo in combos:
                n_enum += 1
                if n_enum > max_candidates * 15:
                    break

                # Assemble: replace [*] in scaffold with decorator SMILES
                smiles = _assemble_grammar(scaffold.smiles, [d.smiles for d in combo])
                if smiles is None:
                    continue

                if smiles in seen:
                    continue
                seen.add(smiles)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                if mol.GetNumHeavyAtoms() > 60:
                    continue
                if Descriptors.MolWt(mol) > 800:
                    continue

                desc = compute_receptor_descriptor(smiles)
                if not desc.valid:
                    continue

                bscore = score_glycan_binder(smiles, target, desc)
                sel_ddG, sel_map = selectivity_score(smiles, target, competitors, desc)

                arm_names = [d.name for d in combo]
                candidates.append(GlycanBinderCandidate(
                    smiles=smiles,
                    backbone_name=scaffold.ring_system,
                    arm_names=arm_names,
                    score=bscore,
                    selectivity_ddG=sel_ddG,
                    selectivity_map=sel_map,
                    sa_score=desc.sa_score,
                ))

                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    n_scored = len(candidates)

    # Pareto ranking
    if candidates:
        obj_vectors = [
            (c.score.dG_total, c.selectivity_ddG, -c.sa_score)
            for c in candidates
        ]
        fronts = fast_non_dominated_sort(obj_vectors)
        front_set = set(fronts[0]) if fronts else set()
        for i, c in enumerate(candidates):
            c.pareto_front = 0 if i in front_set else -1
            c.pareto_rank = -1
            for fi, front in enumerate(fronts):
                if i in front:
                    c.pareto_rank = fi
                    break
        candidates.sort(key=lambda c: (c.pareto_rank, c.score.dG_total))
        n_pareto = len(front_set)
    else:
        n_pareto = 0

    elapsed = time.time() - t0

    return GlycanBinderResult(
        target=target,
        competitors=competitors,
        candidates=candidates,
        n_enumerated=n_enum,
        n_scored=n_scored,
        n_pareto_front=n_pareto,
        elapsed_s=elapsed,
    )


def _assemble_grammar(scaffold_smi: str, decorator_smiles: List[str]) -> Optional[str]:
    """
    Assemble a molecule by replacing [*] dummies in scaffold with decorators.

    Each decorator has exactly one [*]. The scaffold has n_sites [*] atoms.
    We join them pairwise.
    """
    mol = Chem.MolFromSmiles(scaffold_smi)
    if mol is None:
        return None

    # Find dummy atoms in scaffold
    dummy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_indices) != len(decorator_smiles):
        return None

    # Strategy: combine all fragments, then join dummies
    # Build combined SMILES: scaffold.dec1.dec2...
    combined = scaffold_smi
    for dec_smi in decorator_smiles:
        combined += "." + dec_smi

    combo_mol = Chem.MolFromSmiles(combined)
    if combo_mol is None:
        return None

    # Find all dummy atoms
    dummies = [a.GetIdx() for a in combo_mol.GetAtoms() if a.GetAtomicNum() == 0]
    # Scaffold dummies are first, decorator dummies follow
    n_sites = len(decorator_smiles)
    if len(dummies) != 2 * n_sites:
        # Unexpected dummy count
        return None

    from rdkit.Chem import RWMol, AllChem

    rw = RWMol(combo_mol)

    # Pair them: scaffold dummy i with decorator dummy i
    # scaffold dummies: dummies[0..n_sites-1]
    # decorator dummies: dummies[n_sites..2*n_sites-1]
    pairs = []
    for i in range(n_sites):
        sd = dummies[i]
        dd = dummies[n_sites + i]
        # Get neighbors of each dummy
        sd_nbrs = [n.GetIdx() for n in rw.GetAtomWithIdx(sd).GetNeighbors()]
        dd_nbrs = [n.GetIdx() for n in rw.GetAtomWithIdx(dd).GetNeighbors()]
        if not sd_nbrs or not dd_nbrs:
            return None
        pairs.append((sd, dd, sd_nbrs[0], dd_nbrs[0]))

    # Add bonds between the real neighbors of paired dummies
    for sd, dd, sn, dn in pairs:
        rw.AddBond(sn, dn, Chem.BondType.SINGLE)

    # Remove dummy atoms (highest index first)
    to_remove = sorted(set(d for p in pairs for d in [p[0], p[1]]), reverse=True)
    for idx in to_remove:
        rw.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(rw)
        smi = Chem.MolToSmiles(rw)
        mol2 = Chem.MolFromSmiles(smi)
        if mol2 is None:
            return None
        return Chem.MolToSmiles(mol2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_approaches(target: str = "Glc", max_candidates: int = 200):
    """
    Run both generation approaches and compare results.

    Returns dict with side-by-side comparison.
    """
    from glycan.de_novo_binder import generate_glycan_binders, GlycanBinderSpec

    # Fixed library approach
    spec = GlycanBinderSpec(target=target, max_candidates=max_candidates)
    fixed = generate_glycan_binders(spec=spec)

    # Demand-driven approach
    demand_result = generate_from_demand(target=target, max_candidates=max_candidates)

    demand = compute_demand(target)

    return {
        "target": target,
        "demand": demand,
        "fixed": {
            "n_scored": fixed.n_scored,
            "n_pareto": fixed.n_pareto_front,
            "best_dG": fixed.best.score.dG_total if fixed.best else None,
            "elapsed": fixed.elapsed_s,
        },
        "grammar": {
            "n_scored": demand_result.n_scored,
            "n_pareto": demand_result.n_pareto_front,
            "best_dG": demand_result.best.score.dG_total if demand_result.best else None,
            "elapsed": demand_result.elapsed_s,
        },
        "fixed_result": fixed,
        "grammar_result": demand_result,
    }
