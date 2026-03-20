"""
glycan/de_novo_binder.py -- De Novo Glycan Binder Design

Designs synthetic receptors for monosaccharide targets using
pharmacophore-level scoring and NSGA-II Pareto optimization.

Strategy:
  1. Define target sugar descriptors (OH pattern, CH-pi face, diol geometry)
  2. Enumerate candidate receptors from MABE fragment library
  3. Score each candidate on: aromatic CH-pi capacity, H-bond complementarity,
     cavity size match, boronic acid diol compatibility
  4. Rank by 3-objective Pareto: affinity x selectivity x synthesizability
  5. Return ranked candidates with energy decomposition

Scoring is pharmacophore-level (no docking, no MD). The terms are:
  - dG_chpi: n_aromatic_rings * eps_CHP * sugar.n_axial_CH
  - dG_hb: min(receptor_HB_capacity, sugar.n_exposed_OH) * eps_HB
  - dG_desolv: -sum(k_desolv[OH_type]) for each buried sugar OH
  - dG_shape: penalty for cavity size mismatch
  - dG_boronic: bonus for boronic acid + cis-1,2-diol match
  - dG_conf: conformational penalty from receptor flexibility

Uses existing:
  - core/de_novo_generator.py (assemble, sa_score, fragment libraries)
  - core/pareto.py (NSGA-II)
  - glycan/parameters_v23.py (locked physics parameters)

Data tier: Tier 2 (sugar descriptors from crystallography + literature).

References:
  - Davis AP, Nat. Chem. 2012, 4, 548 (design principles for synthetic lectins)
  - Asensio et al., Acc. Chem. Res. 2013, 46, 946 (CH-pi in carbohydrate recognition)
  - Nishio M, Phys. Chem. Chem. Phys. 2011, 13, 13873 (CH-pi geometry)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Import locked glycan parameters
from glycan.parameters_v23 import (
    EPS_HB_EFF,
    K_DESOLV_EQ, K_DESOLV_AX, K_DESOLV_C6, K_DESOLV_NAC,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RT = 2.479  # kJ/mol at 298 K

# Scoring parameters (from locked glycan params + literature)
EPS_CHP_AROMATIC = -1.9      # kJ/mol per aromatic ring facing sugar CH (Tyr-like)
EPS_CHP_LARGE_AROMATIC = -2.9  # kJ/mol per large aromatic (anthracene/indole, Trp-like)
EPS_HB = EPS_HB_EFF          # -2.25 kJ/mol per H-bond (context-weighted)
EPS_BORONIC_DIOL = -8.0      # kJ/mol per boronic acid + cis-1,2-diol match (covalent-ish)
CAVITY_OPTIMAL_VOL_A3 = 180.0  # Optimal cavity for pyranose (~160-200 A3)
CAVITY_PENALTY_PER_A3 = 0.02  # kJ/mol per A3 deviation from optimal
EPS_FLEXIBILITY_PENALTY = 0.3  # kJ/mol per rotatable bond beyond 6
EPS_AXIAL_CLASH = 1.5          # kJ/mol per axial OH disrupting CH-pi contacts
                               # Asensio 2013: axial OH on alpha-face sterically
                               # clashes with aromatic surface, reducing CH-pi quality


# ---------------------------------------------------------------------------
# Sugar target descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SugarTarget:
    """Pharmacophore-level descriptor of a monosaccharide target."""
    name: str
    n_eq_OH: int          # equatorial hydroxyls available for binding
    n_ax_OH: int          # axial hydroxyls
    n_primary_OH: int     # C6 primary OH
    n_NAc: int            # N-acetyl groups
    n_axial_CH: int       # axial C-H bonds on alpha-face (CH-pi donors)
    has_cis_12_diol: bool  # cis-1,2-diol for boronic acid binding
    has_cis_13_diol: bool  # 1,3-diol
    ring_type: str        # "pyranose" or "furanose"
    molecular_volume_A3: float
    notes: str = ""


_SUGAR_TARGETS = {
    "Glc": SugarTarget(
        "Glc", n_eq_OH=4, n_ax_OH=0, n_primary_OH=1, n_NAc=0,
        n_axial_CH=3, has_cis_12_diol=False, has_cis_13_diol=True,
        ring_type="pyranose", molecular_volume_A3=160.0,
        notes="All-equatorial glucose. Ideal Davis-type target. 3 axial CH on alpha-face."
    ),
    "Gal": SugarTarget(
        "Gal", n_eq_OH=3, n_ax_OH=1, n_primary_OH=1, n_NAc=0,
        n_axial_CH=2, has_cis_12_diol=False, has_cis_13_diol=True,
        ring_type="pyranose", molecular_volume_A3=160.0,
        notes="C4-axial OH disrupts alpha-face. 2 axial CH. Galectin target."
    ),
    "Man": SugarTarget(
        "Man", n_eq_OH=3, n_ax_OH=1, n_primary_OH=1, n_NAc=0,
        n_axial_CH=2, has_cis_12_diol=True, has_cis_13_diol=True,
        ring_type="pyranose", molecular_volume_A3=160.0,
        notes="C2-axial OH. ConA target. cis-1,2-diol at C1-C2."
    ),
    "GlcNAc": SugarTarget(
        "GlcNAc", n_eq_OH=3, n_ax_OH=0, n_primary_OH=1, n_NAc=1,
        n_axial_CH=3, has_cis_12_diol=False, has_cis_13_diol=True,
        ring_type="pyranose", molecular_volume_A3=195.0,
        notes="Glucose + NAc at C2. WGA target. Bulky NAc limits cavity entry."
    ),
    "GalNAc": SugarTarget(
        "GalNAc", n_eq_OH=2, n_ax_OH=1, n_primary_OH=1, n_NAc=1,
        n_axial_CH=2, has_cis_12_diol=False, has_cis_13_diol=True,
        ring_type="pyranose", molecular_volume_A3=195.0,
        notes="Tn antigen. C4-axial. PNA/SBA target. TACA on tumor cells."
    ),
    "Fuc": SugarTarget(
        "Fuc", n_eq_OH=3, n_ax_OH=1, n_primary_OH=0, n_NAc=0,
        n_axial_CH=2, has_cis_12_diol=False, has_cis_13_diol=True,
        ring_type="pyranose", molecular_volume_A3=145.0,
        notes="6-deoxy-galactose. No primary OH. Lewis antigen component."
    ),
    "Fru": SugarTarget(
        "Fru", n_eq_OH=2, n_ax_OH=0, n_primary_OH=1, n_NAc=0,
        n_axial_CH=0, has_cis_12_diol=True, has_cis_13_diol=True,
        ring_type="furanose", molecular_volume_A3=150.0,
        notes="Furanose ring. Best boronic acid target (cis-1,2-diol). No alpha-face."
    ),
    "Neu5Ac": SugarTarget(
        "Neu5Ac", n_eq_OH=3, n_ax_OH=1, n_primary_OH=0, n_NAc=1,
        n_axial_CH=1, has_cis_12_diol=True, has_cis_13_diol=True,
        ring_type="pyranose", molecular_volume_A3=250.0,
        notes="Sialic acid. Large (9-carbon). Carboxylate + NAc. Siglec target."
    ),
}


def get_sugar_target(name: str) -> SugarTarget:
    if name not in _SUGAR_TARGETS:
        raise KeyError(f"Unknown sugar '{name}'. Available: {list_sugar_targets()}")
    return _SUGAR_TARGETS[name]


def list_sugar_targets() -> List[str]:
    return sorted(_SUGAR_TARGETS.keys())


# ---------------------------------------------------------------------------
# Receptor descriptor (from SMILES)
# ---------------------------------------------------------------------------

@dataclass
class ReceptorDescriptor:
    """Computed pharmacophore properties of a candidate receptor."""
    smiles: str
    n_aromatic_rings: int = 0
    n_large_aromatic: int = 0    # fused >= 3 rings (anthracene, pyrene)
    n_hb_donors: int = 0
    n_hb_acceptors: int = 0
    n_urea_groups: int = 0
    n_boronic_acid: int = 0
    n_rotatable_bonds: int = 0
    molecular_weight: float = 0.0
    estimated_cavity_A3: float = 0.0
    sa_score: float = 10.0
    valid: bool = False


def compute_receptor_descriptor(smiles: str) -> ReceptorDescriptor:
    """Compute pharmacophore descriptor from SMILES. Requires RDKit."""
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required for receptor descriptor computation")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ReceptorDescriptor(smiles=smiles, valid=False)

    ri = mol.GetRingInfo()
    aromatic_rings = sum(1 for r in ri.BondRings()
                         if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in r))

    # Large aromatics: count fused aromatic systems with >= 10 aromatic atoms
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_large = 0
    if aromatic_atoms >= 10:
        n_large = aromatic_atoms // 10  # rough: 1 per 10 aromatic atoms

    n_hbd = Descriptors.NumHDonors(mol)
    n_hba = Descriptors.NumHAcceptors(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)

    # Count urea groups: N-C(=O)-N pattern
    urea_pat = Chem.MolFromSmarts("[NX3]C(=O)[NX3]")
    n_urea = len(mol.GetSubstructMatches(urea_pat)) if urea_pat else 0

    # Count boronic acid: B(O)(O) pattern
    boronic_pat = Chem.MolFromSmarts("[B]([OH])([OH])")
    n_boronic = len(mol.GetSubstructMatches(boronic_pat)) if boronic_pat else 0
    if n_boronic == 0:
        # Try alternate pattern B(O)O
        boronic_pat2 = Chem.MolFromSmarts("[BX3](O)(O)")
        n_boronic = len(mol.GetSubstructMatches(boronic_pat2)) if boronic_pat2 else 0

    # Cavity volume estimate: MW-based heuristic (rough)
    # ~1 A3 per dalton for organic molecules, cavity ~ 30-40% of total
    est_cavity = mw * 0.35

    # SA score (from de_novo_generator if available)
    try:
        from core.de_novo_generator import sa_score_smiles
        sa = sa_score_smiles(smiles)
    except (ImportError, Exception):
        sa = 5.0  # default mid-range

    return ReceptorDescriptor(
        smiles=smiles,
        n_aromatic_rings=aromatic_rings,
        n_large_aromatic=n_large,
        n_hb_donors=n_hbd,
        n_hb_acceptors=n_hba,
        n_urea_groups=n_urea,
        n_boronic_acid=n_boronic,
        n_rotatable_bonds=n_rot,
        molecular_weight=mw,
        estimated_cavity_A3=est_cavity,
        sa_score=sa,
        valid=True,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class BinderScore:
    """Score of a candidate receptor against a sugar target."""
    smiles: str
    target: str
    dG_total: float
    dG_chpi: float
    dG_hb: float
    dG_desolv: float
    dG_shape: float
    dG_boronic: float
    dG_flexibility: float
    dG_axial_clash: float = 0.0   # penalty for axial OH disrupting CH-pi
    sa_score: float = 10.0
    descriptor: Optional[ReceptorDescriptor] = None


def score_glycan_binder(
    smiles: str,
    target: str,
    descriptor: Optional[ReceptorDescriptor] = None,
) -> BinderScore:
    """
    Score a candidate receptor SMILES against a sugar target.

    Parameters
    ----------
    smiles : str
        Candidate receptor SMILES.
    target : str
        Sugar target name (key into _SUGAR_TARGETS).
    descriptor : ReceptorDescriptor or None
        Pre-computed descriptor. Computed if None.

    Returns
    -------
    BinderScore
    """
    sugar = get_sugar_target(target)

    if descriptor is None:
        descriptor = compute_receptor_descriptor(smiles)

    if not descriptor.valid:
        return BinderScore(
            smiles=smiles, target=target,
            dG_total=0.0, dG_chpi=0.0, dG_hb=0.0, dG_desolv=0.0,
            dG_shape=0.0, dG_boronic=0.0, dG_flexibility=0.0,
            sa_score=10.0, descriptor=descriptor,
        )

    # CH-pi: aromatic surfaces interact with sugar axial CH
    chpi_contacts = min(descriptor.n_aromatic_rings, sugar.n_axial_CH)
    dG_chpi = (
        min(descriptor.n_large_aromatic, chpi_contacts) * EPS_CHP_LARGE_AROMATIC
        + max(0, chpi_contacts - descriptor.n_large_aromatic) * EPS_CHP_AROMATIC
    )

    # H-bonds: min of receptor capacity and sugar OH count
    sugar_hb_sites = sugar.n_eq_OH + sugar.n_ax_OH + sugar.n_primary_OH
    # Urea groups are particularly effective for sugar H-bonding (2 NH per urea)
    receptor_hb_capacity = descriptor.n_hb_donors + descriptor.n_urea_groups
    n_hb = min(receptor_hb_capacity, sugar_hb_sites)
    dG_hb = n_hb * EPS_HB

    # Desolvation: buried OHs pay desolvation penalty (partially offset by H-bonds)
    # Assume receptor buries ~60% of sugar OHs that make contacts
    n_buried_eq = min(int(sugar.n_eq_OH * 0.6 + 0.5), n_hb)
    n_buried_ax = min(sugar.n_ax_OH, max(0, n_hb - n_buried_eq))
    n_buried_c6 = min(sugar.n_primary_OH, 1) if n_hb >= 2 else 0
    dG_desolv = (
        n_buried_eq * K_DESOLV_EQ
        + n_buried_ax * K_DESOLV_AX
        + n_buried_c6 * K_DESOLV_C6
    )
    if sugar.n_NAc > 0 and receptor_hb_capacity >= sugar_hb_sites:
        dG_desolv += K_DESOLV_NAC

    # Shape: cavity volume match to pyranose
    vol_diff = abs(descriptor.estimated_cavity_A3 - sugar.molecular_volume_A3)
    dG_shape = vol_diff * CAVITY_PENALTY_PER_A3

    # Boronic acid: bonus for diol-compatible sugars
    dG_boronic = 0.0
    if descriptor.n_boronic_acid > 0:
        if sugar.has_cis_12_diol:
            dG_boronic = descriptor.n_boronic_acid * EPS_BORONIC_DIOL
        elif sugar.has_cis_13_diol:
            dG_boronic = descriptor.n_boronic_acid * EPS_BORONIC_DIOL * 0.3  # weaker

    # Flexibility penalty: very flexible receptors pay entropic cost
    excess_rot = max(0, descriptor.n_rotatable_bonds - 6)
    dG_flexibility = excess_rot * EPS_FLEXIBILITY_PENALTY

    # Alpha-face disruption: axial OHs clash with aromatic CH-pi surface
    # Only applies when receptor makes CH-pi contacts (has aromatic surface)
    # Asensio 2013: axial substituents on sugar alpha-face sterically
    # interfere with face-on aromatic stacking geometry
    dG_axial_clash = 0.0
    if chpi_contacts > 0 and sugar.n_ax_OH > 0:
        dG_axial_clash = sugar.n_ax_OH * EPS_AXIAL_CLASH

    # Total: all terms use sign convention where negative = favorable
    # dG_chpi:   negative (favorable)
    # dG_hb:     negative (favorable)
    # dG_desolv: positive (penalty)
    # dG_shape:  positive (penalty)
    # dG_boronic: negative (favorable)
    # dG_flexibility: positive (penalty)
    # dG_axial_clash: positive (penalty)
    dG_total = (dG_chpi + dG_hb + dG_desolv + dG_shape
                + dG_boronic + dG_flexibility + dG_axial_clash)

    return BinderScore(
        smiles=smiles, target=target,
        dG_total=round(dG_total, 2),
        dG_chpi=round(dG_chpi, 2),
        dG_hb=round(dG_hb, 2),
        dG_desolv=round(dG_desolv, 2),
        dG_shape=round(dG_shape, 2),
        dG_boronic=round(dG_boronic, 2),
        dG_flexibility=round(dG_flexibility, 2),
        dG_axial_clash=round(dG_axial_clash, 2),
        sa_score=round(descriptor.sa_score, 2),
        descriptor=descriptor,
    )


def selectivity_score(
    smiles: str,
    target: str,
    competitors: Optional[List[str]] = None,
    descriptor: Optional[ReceptorDescriptor] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute selectivity of a receptor for target vs competitors.

    Returns (min_ddG, {competitor: ddG}) where ddG = dG_target - dG_competitor.
    More negative ddG = more selective for target.
    """
    if competitors is None:
        # Default competitor panel: all sugars except target
        competitors = [s for s in _SUGAR_TARGETS if s != target]

    if descriptor is None:
        descriptor = compute_receptor_descriptor(smiles)

    target_score = score_glycan_binder(smiles, target, descriptor)
    ddG_map = {}
    for comp in competitors:
        comp_score = score_glycan_binder(smiles, comp, descriptor)
        ddG_map[comp] = round(target_score.dG_total - comp_score.dG_total, 2)

    min_ddG = min(ddG_map.values()) if ddG_map else 0.0
    return min_ddG, ddG_map


# ---------------------------------------------------------------------------
# Binder spec and result
# ---------------------------------------------------------------------------

@dataclass
class GlycanBinderSpec:
    """Specification for glycan binder design."""
    target: str                                # sugar name
    competitors: Optional[List[str]] = None    # selectivity panel
    max_candidates: int = 200
    max_heavy_atoms: int = 60
    max_mw: float = 800.0
    require_aromatic: bool = True              # must have aromatic for CH-pi
    allow_boronic: bool = True


@dataclass
class GlycanBinderCandidate:
    """One candidate from the design pipeline."""
    smiles: str
    backbone_name: str
    arm_names: List[str]
    score: BinderScore
    selectivity_ddG: float                     # min ddG vs competitors
    selectivity_map: Dict[str, float]
    sa_score: float
    pareto_rank: int = -1
    pareto_front: int = -1


@dataclass
class GlycanBinderResult:
    """Output of glycan binder design run."""
    target: str
    competitors: List[str]
    candidates: List[GlycanBinderCandidate]
    n_enumerated: int = 0
    n_scored: int = 0
    n_pareto_front: int = 0
    elapsed_s: float = 0.0

    @property
    def best(self) -> Optional[GlycanBinderCandidate]:
        return self.candidates[0] if self.candidates else None

    def summary(self) -> str:
        lines = [
            f"Glycan Binder Design: target={self.target}",
            f"  Competitors: {', '.join(self.competitors)}",
            f"  Enumerated: {self.n_enumerated}, Scored: {self.n_scored}, "
            f"Pareto front: {self.n_pareto_front}",
            f"  Time: {self.elapsed_s:.1f}s",
        ]
        if self.best:
            b = self.best
            lines.append(f"  Best: dG={b.score.dG_total:.1f} sel={b.selectivity_ddG:.1f} "
                         f"SA={b.sa_score:.1f}")
            lines.append(f"    SMILES: {b.smiles[:80]}")
            lines.append(f"    Backbone: {b.backbone_name}, Arms: {b.arm_names}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _get_glycan_relevant_arms():
    """Filter arm library for glycan-relevant chemistry."""
    from core.de_novo_generator import ARM_LIBRARY
    relevant_categories = {
        "boronic_acid", "urea", "amide", "hydroxyl", "amine",
        "phenol", "carboxylate", "sulfonamide",
    }
    # Also include aromatic arms for CH-pi
    arms = []
    for arm in ARM_LIBRARY:
        cat = getattr(arm, "category", "")
        # Include if category matches or if arm provides H-bond capable donors
        if cat in relevant_categories:
            arms.append(arm)
        elif any(d.startswith("O_") or d.startswith("N_") for d in arm.donor_subtypes):
            arms.append(arm)
    return arms


def _get_glycan_relevant_backbones(require_aromatic: bool = True):
    """Filter backbone library for glycan-relevant scaffolds."""
    from core.de_novo_generator import BACKBONE_LIBRARY
    backbones = []
    for bb in BACKBONE_LIBRARY:
        if require_aromatic:
            # Need aromatic surface for CH-pi
            if not any(c in bb.smiles for c in ["c1ccc", "c1cc", "C1=CC"]):
                continue
        # Prefer 2-3 attachment sites (receptor-like)
        if 2 <= bb.n_sites <= 4:
            backbones.append(bb)
    return backbones


def generate_glycan_binders(
    spec: Optional[GlycanBinderSpec] = None,
    target: str = "Glc",
) -> GlycanBinderResult:
    """
    Generate de novo glycan binder candidates.

    Uses combinatorial enumeration from MABE fragment library,
    scored against target sugar, ranked by 3-objective Pareto
    (affinity x selectivity x synthesizability).

    Parameters
    ----------
    spec : GlycanBinderSpec or None
        Design specification. Defaults used if None.
    target : str
        Sugar target (overridden by spec.target if spec provided).

    Returns
    -------
    GlycanBinderResult
    """
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required for glycan binder generation")

    from core.de_novo_generator import assemble, sa_score_smiles
    from core.pareto import fast_non_dominated_sort

    t0 = time.time()

    if spec is None:
        spec = GlycanBinderSpec(target=target)
    else:
        target = spec.target

    sugar = get_sugar_target(target)
    competitors = spec.competitors or [s for s in _SUGAR_TARGETS if s != target]

    backbones = _get_glycan_relevant_backbones(spec.require_aromatic)
    arms = _get_glycan_relevant_arms()

    if not backbones or not arms:
        return GlycanBinderResult(
            target=target, competitors=competitors,
            candidates=[], elapsed_s=time.time() - t0,
        )

    # Enumerate
    from itertools import product as cartesian_product
    candidates = []
    seen_smiles = set()
    n_enumerated = 0

    for bb in backbones:
        # For each backbone, try arm combinations (limit combinatorial explosion)
        arm_combos = list(cartesian_product(arms, repeat=bb.n_sites))
        # Cap per backbone
        if len(arm_combos) > 500:
            import random
            random.seed(42)
            arm_combos = random.sample(arm_combos, 500)

        for arm_combo in arm_combos:
            n_enumerated += 1
            if n_enumerated > spec.max_candidates * 10:
                break

            smiles, mol = assemble(bb, list(arm_combo))
            if smiles is None or mol is None:
                continue

            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)

            # Property filter
            if mol.GetNumHeavyAtoms() > spec.max_heavy_atoms:
                continue

            mw = Descriptors.MolWt(mol)
            if mw > spec.max_mw:
                continue

            # Score
            desc = compute_receptor_descriptor(smiles)
            if not desc.valid:
                continue

            if spec.require_aromatic and desc.n_aromatic_rings == 0:
                continue

            bscore = score_glycan_binder(smiles, target, desc)
            sel_ddG, sel_map = selectivity_score(smiles, target, competitors, desc)

            candidates.append(GlycanBinderCandidate(
                smiles=smiles,
                backbone_name=bb.name,
                arm_names=[a.name for a in arm_combo],
                score=bscore,
                selectivity_ddG=sel_ddG,
                selectivity_map=sel_map,
                sa_score=desc.sa_score,
            ))

            if len(candidates) >= spec.max_candidates:
                break

        if len(candidates) >= spec.max_candidates:
            break

    n_scored = len(candidates)

    # Pareto ranking: minimize (-affinity, -selectivity, +SA)
    if candidates:
        # Build objective vectors (all to be MAXIMIZED for fast_non_dominated_sort)
        # affinity: more negative dG = better -> pass dG_total directly (maximize)
        # selectivity: more negative ddG = better -> pass ddG directly (maximize)
        # SA: lower = better -> negate (maximize -SA)
        obj_vectors = []
        for c in candidates:
            obj_vectors.append((
                c.score.dG_total,        # more negative = better (maximize)
                c.selectivity_ddG,       # more negative = better (maximize)
                -c.sa_score,             # lower SA = better, negate to maximize
            ))

        fronts = fast_non_dominated_sort(obj_vectors)

        front_set = set(fronts[0]) if fronts else set()
        for i, c in enumerate(candidates):
            c.pareto_front = 0 if i in front_set else -1
            c.pareto_rank = -1
            for fi, front in enumerate(fronts):
                if i in front:
                    c.pareto_rank = fi
                    break

        # Sort: Pareto front first, then by affinity within front
        candidates.sort(key=lambda c: (c.pareto_rank, c.score.dG_total))
        n_pareto = len(front_set)
    else:
        n_pareto = 0

    elapsed = time.time() - t0

    return GlycanBinderResult(
        target=target,
        competitors=competitors,
        candidates=candidates,
        n_enumerated=n_enumerated,
        n_scored=n_scored,
        n_pareto_front=n_pareto,
        elapsed_s=elapsed,
    )
