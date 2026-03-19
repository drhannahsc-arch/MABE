"""
core/galnac_binder_scorer.py — AACR Pipeline: GalNAc Binder Scoring

Scores synthetic receptor candidates for GalNAc (Tn antigen) binding
and selectivity over normal-tissue glycans.

Strategy:
  1. Analyze candidate molecular features (H-bond donors, aromatics, size)
  2. Estimate contact quality with GalNAc vs other sugars
  3. Score using glycan physics parameters (same params as lectin scoring)
  4. Compute selectivity panel against Glc, Man, Gal, GlcNAc, Fuc

This is a PREDICTIVE model — contact maps are estimated from molecular
features, not extracted from crystal structures. That's the point:
we're predicting which candidates SHOULD bind before synthesis.

Physics: all parameters from non-biological calibration (Schwarz 1996,
Jasra 1982, Laughrey 2008, GLYCAM06). Zero biological fitting.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

from mabe.glycan.sugar_properties import (
    SugarPropertyCard, ALPHA_D_GALNAC, ALPHA_D_GLUCOSE,
    ALPHA_D_MANNOSE, ALPHA_D_GALACTOSE, ALPHA_D_GLCNAC,
    ALPHA_L_FUCOSE, SUGAR_LIBRARY,
)
from mabe.glycan.contact_map import GlycanContactMap, OHContact, CHPiContact
from mabe.glycan.scorer import compute_glycan_terms, GlycanScoreDecomposition
from mabe.glycan.params import GLYCAN_PARAMS


# ═══════════════════════════════════════════════════════════════════════════
# MOLECULAR FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReceptorFeatures:
    """Molecular features relevant to sugar binding."""
    smiles: str
    name: str = ""
    n_hbd: int = 0              # H-bond donors (NH, OH, etc.)
    n_hba: int = 0              # H-bond acceptors (C=O, N, O, etc.)
    n_aromatic_rings: int = 0   # aromatic rings for CH-π
    aromatic_sa: float = 0.0    # aromatic surface area (Å²)
    mw: float = 0.0
    n_rotatable: int = 0
    cavity_compatible: bool = False  # size compatible with pyranose
    has_boronic_acid: bool = False   # reversible covalent diol binding
    sa_score: float = 10.0      # synthetic accessibility (1=easy, 10=hard)
    # Chelator geometry penalty fields
    n_aminocarboxylate: int = 0      # N-CH2-COOH motifs (EDTA/NTA/DTPA-like)
    n_ida_motif: int = 0             # iminodiacetic acid N(CH2COOH)2
    n_polycarboxylate: int = 0       # total carboxylate groups
    is_chelator_like: bool = False   # flagged as metal chelator geometry
    chelator_penalty: float = 0.0    # kJ/mol penalty applied to sugar scoring
    valid: bool = True


def extract_receptor_features(smiles: str, name: str = "") -> ReceptorFeatures:
    """Extract sugar-binding-relevant features from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ReceptorFeatures(smiles=smiles, name=name, valid=False)

    feat = ReceptorFeatures(smiles=smiles, name=name)
    feat.n_hbd = Lipinski.NumHDonors(mol)
    feat.n_hba = Lipinski.NumHAcceptors(mol)
    feat.n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    feat.mw = Descriptors.MolWt(mol)
    feat.n_rotatable = Descriptors.NumRotatableBonds(mol)

    # Aromatic surface area proxy: count aromatic atoms × ~10 Å² each
    n_arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    feat.aromatic_sa = n_arom_atoms * 10.0

    # Cavity compatibility: pyranose ring is ~5.5 Å diameter
    # Candidates with MW 200-800 and ≥2 H-bond donors are compatible
    feat.cavity_compatible = (200 <= feat.mw <= 800 and feat.n_hbd >= 2)

    # Boronic acid detection (B(O)(O) pattern)
    boronic_pattern = Chem.MolFromSmarts('[#5]([OH])([OH])')
    if boronic_pattern:
        feat.has_boronic_acid = mol.HasSubstructMatch(boronic_pattern)

    # ── Chelator geometry detection ────────────────────────────────────
    # Aminocarboxylate: N connected to CH2-COOH (EDTA/NTA/DTPA signature)
    amino_carboxy = Chem.MolFromSmarts('[NX3][CH2][CX3](=O)[OX1,OX2]')
    if amino_carboxy:
        feat.n_aminocarboxylate = len(mol.GetSubstructMatches(amino_carboxy))

    # Iminodiacetic acid motif: N(CH2COOH)2 — strong chelator signature
    ida_pattern = Chem.MolFromSmarts('[NX3]([CH2][CX3](=O)[OH,O-])([CH2][CX3](=O)[OH,O-])')
    if ida_pattern:
        feat.n_ida_motif = len(mol.GetSubstructMatches(ida_pattern))

    # Count total carboxylate groups
    carboxylate = Chem.MolFromSmarts('[CX3](=O)[OX1H,OX1-]')
    if carboxylate:
        feat.n_polycarboxylate = len(mol.GetSubstructMatches(carboxylate))

    # Chelator classification:
    # ≥2 aminocarboxylate arms OR ≥1 IDA motif OR ≥3 carboxylates with amine
    has_amine = Chem.MolFromSmarts('[NX3;!$([NX3][CX3]=[OX1])]')  # non-amide N
    n_amine = len(mol.GetSubstructMatches(has_amine)) if has_amine else 0
    feat.is_chelator_like = (
        feat.n_aminocarboxylate >= 2 or
        feat.n_ida_motif >= 1 or
        (feat.n_polycarboxylate >= 3 and n_amine >= 1)
    )

    # Chelator penalty: proportional to chelation character
    # Rationale: aminocarboxylate donors bind metals, not sugars.
    # Their H-bond donors are geometrically pre-organized for octahedral
    # metal coordination, not for pyranose OH recognition.
    if feat.is_chelator_like:
        feat.chelator_penalty = 5.0 * feat.n_aminocarboxylate + 8.0 * feat.n_ida_motif
    else:
        feat.chelator_penalty = 0.0

    # SA score
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        feat.sa_score = sascorer.calculateScore(mol)
    except Exception:
        feat.sa_score = 5.0  # default mid-range

    return feat


# ═══════════════════════════════════════════════════════════════════════════
# CONTACT MAP ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

def estimate_contact_map(
    feat: ReceptorFeatures,
    sugar: SugarPropertyCard,
) -> GlycanContactMap:
    """
    Estimate a glycan contact map from receptor molecular features.

    This is the predictive model: given a synthetic receptor's features,
    estimate how many H-bonds and CH-π contacts it would make with a sugar.

    Rules (physics-based, not fitted):
    - Each H-bond donor on the receptor can potentially H-bond one sugar OH
    - Maximum useful H-bonds = min(receptor HBD, sugar accessible OHs)
    - Aromatic rings provide CH-π contacts with axial C-H on pyranose
    - Maximum CH-π = min(receptor aromatic rings, sugar n_CH_pi_faces)
    - Boronic acid adds 2 covalent B-O bonds to cis-diol (if available)
    """
    n_sugar_oh = len(sugar.hydroxyls)

    # H-bond allocation: distribute receptor HBDs across sugar OHs
    # Prioritize C3, C4 (most commonly engaged in lectins), then C6, then C1
    priority_order = ['C3', 'C4', 'C6', 'C1', 'C2']
    oh_position_map = {oh.position: oh for oh in sugar.hydroxyls}

    available_hbd = feat.n_hbd
    oh_contacts = []

    for pos in priority_order:
        if pos not in oh_position_map:
            continue
        oh = oh_position_map[pos]
        if available_hbd >= 2:
            # Strong engagement: 2 H-bonds (like ConA C3, C4)
            oh_contacts.append(OHContact(
                pos, n_hbonds=2, is_buried=True, is_solvent_exposed=False
            ))
            available_hbd -= 2
        elif available_hbd >= 1:
            # Moderate engagement: 1 H-bond
            oh_contacts.append(OHContact(
                pos, n_hbonds=1, is_buried=False, is_solvent_exposed=True
            ))
            available_hbd -= 1
        else:
            # No engagement: OH exposed to solvent (desolvation cost only)
            oh_contacts.append(OHContact(
                pos, n_hbonds=0, is_solvent_exposed=True
            ))

    # CH-π contacts
    max_ch_pi = min(feat.n_aromatic_rings, sugar.n_CH_pi_faces)
    ch_pi_contacts = []
    if max_ch_pi > 0:
        ch_pi_contacts.append(CHPiContact(
            n_CH_contacts=max_ch_pi,
            receptor_residue=f"aromatic_ring_x{feat.n_aromatic_rings}",
        ))

    # Conserved waters: estimate 1 per 3 H-bond donors beyond sugar contacts
    n_waters = max(0, (feat.n_hba - n_sugar_oh) // 3)
    n_waters = min(n_waters, 2)  # cap at 2

    return GlycanContactMap(
        receptor_name=feat.name or feat.smiles[:30],
        sugar_key=sugar.three_letter,
        oh_contacts=oh_contacts,
        ch_pi_contacts=ch_pi_contacts,
        n_conserved_waters=n_waters,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GalNAcBinderScore:
    """Complete scoring result for a GalNAc binder candidate."""
    smiles: str
    name: str
    features: ReceptorFeatures
    # Primary target
    dg_galnac: float = 0.0
    decomp_galnac: Optional[GlycanScoreDecomposition] = None
    # Selectivity panel
    dg_panel: Dict[str, float] = field(default_factory=dict)
    selectivity_gaps: Dict[str, float] = field(default_factory=dict)
    min_selectivity: float = 0.0
    worst_competitor: str = ""
    # Composite
    attachability: float = 0.0  # 0-1, click-handle compatibility
    composite_score: float = 0.0
    rank: int = 0
    valid: bool = True

    @property
    def log_Ka_est(self) -> float:
        """Estimated log Ka from ΔG."""
        if self.dg_galnac == 0:
            return 0.0
        RT_ln10 = 5.708  # kJ/mol at 298K
        return -self.dg_galnac / RT_ln10


SELECTIVITY_PANEL = {
    'Glc': ALPHA_D_GLUCOSE,
    'Man': ALPHA_D_MANNOSE,
    'Gal': ALPHA_D_GALACTOSE,
    'GlcNAc': ALPHA_D_GLCNAC,
    'Fuc': ALPHA_L_FUCOSE,
}


def score_galnac_binder(
    smiles: str,
    name: str = "",
    params: Optional[object] = None,
    include_selectivity: bool = True,
) -> GalNAcBinderScore:
    """
    Score a synthetic receptor candidate for GalNAc binding.

    Args:
        smiles: SMILES string of receptor candidate
        name: optional label
        params: GlycanParams override (default: GLYCAN_PARAMS)
        include_selectivity: if True, score against normal glycan panel

    Returns:
        GalNAcBinderScore with full decomposition and selectivity
    """
    if params is None:
        params = GLYCAN_PARAMS

    result = GalNAcBinderScore(smiles=smiles, name=name,
                                features=ReceptorFeatures(smiles=smiles))

    # Extract features
    feat = extract_receptor_features(smiles, name)
    result.features = feat
    if not feat.valid:
        result.valid = False
        return result

    # Score against GalNAc
    contacts = estimate_contact_map(feat, ALPHA_D_GALNAC)
    decomp = compute_glycan_terms(
        sugar=ALPHA_D_GALNAC, contacts=contacts, params=params
    )
    result.dg_galnac = decomp.dG_total
    result.decomp_galnac = decomp

    # Boronic acid bonus: reversible covalent B-O with cis-diol
    # GalNAc has cis-diol at C3-C4 (equatorial C3, axial C4)
    if feat.has_boronic_acid:
        # Boronic acid-diol binding: ~-8 to -15 kJ/mol
        # Use conservative estimate
        result.dg_galnac -= 10.0

    # Chelator-geometry penalty: aminocarboxylate donors are pre-organized
    # for octahedral metal coordination, not pyranose OH recognition.
    # Their H-bonds overcount because the geometry is wrong for sugar binding.
    if feat.is_chelator_like:
        result.dg_galnac += feat.chelator_penalty

    # Selectivity panel
    if include_selectivity:
        for sugar_name, sugar_card in SELECTIVITY_PANEL.items():
            contacts_comp = estimate_contact_map(feat, sugar_card)
            decomp_comp = compute_glycan_terms(
                sugar=sugar_card, contacts=contacts_comp, params=params
            )
            dg_comp = decomp_comp.dG_total

            # Boronic acid with competitor sugars
            if feat.has_boronic_acid:
                # Glc has cis-diol at C1-C2 (if alpha) or C2-C3
                # Fructose is the best boronic acid target (furanose cis-diol)
                # GalNAc cis-diol is accessible; Glc less so
                if sugar_name in ('Glc', 'Man', 'Gal'):
                    dg_comp -= 6.0  # weaker boronic acid binding
                elif sugar_name == 'GlcNAc':
                    dg_comp -= 4.0  # even weaker (NAc blocks C2)

            # Chelator penalty applies to all sugar targets equally
            # (chelator geometry is wrong for any sugar, not just GalNAc)
            if feat.is_chelator_like:
                dg_comp += feat.chelator_penalty

            result.dg_panel[sugar_name] = dg_comp
            result.selectivity_gaps[sugar_name] = dg_comp - result.dg_galnac

        if result.selectivity_gaps:
            # Worst competitor = smallest positive gap (most competitive)
            worst = min(result.selectivity_gaps.items(), key=lambda x: x[1])
            result.worst_competitor = worst[0]
            result.min_selectivity = worst[1]

    # Attachability score (0-1)
    # Higher = more suitable for scaffold conjugation
    result.attachability = _compute_attachability(feat)

    # Composite score (for ranking)
    # Affinity (negative = stronger) + selectivity bonus + attachability bonus
    result.composite_score = (
        result.dg_galnac
        - 2.0 * max(0, result.min_selectivity)  # reward selectivity
        - 5.0 * result.attachability              # reward attachability
    )

    return result


def _compute_attachability(feat: ReceptorFeatures) -> float:
    """
    Estimate click-handle compatibility (0-1).

    Factors:
    - Must have ≥1 site distant from binding face for handle attachment
    - Larger molecules have more attachment options
    - Low rotatable bonds = more rigid = better display geometry
    """
    score = 0.0

    # Size bonus: larger molecules have more attachment sites
    if feat.mw >= 300:
        score += 0.3
    elif feat.mw >= 200:
        score += 0.15

    # Rigidity bonus: fewer rotatable bonds = better multivalent display
    if feat.n_rotatable <= 3:
        score += 0.3
    elif feat.n_rotatable <= 6:
        score += 0.15

    # Aromatic rings: can attach handles para to binding face
    if feat.n_aromatic_rings >= 2:
        score += 0.2
    elif feat.n_aromatic_rings >= 1:
        score += 0.1

    # Cavity compatible: appropriate size for pyranose
    if feat.cavity_compatible:
        score += 0.2

    return min(1.0, score)


# ═══════════════════════════════════════════════════════════════════════════
# BATCH SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_candidate_panel(
    candidates: List[Tuple[str, str]],  # (name, smiles) pairs
    include_selectivity: bool = True,
) -> List[GalNAcBinderScore]:
    """Score a panel of candidates and rank by composite score."""
    results = []
    for name, smiles in candidates:
        r = score_galnac_binder(smiles, name=name,
                                 include_selectivity=include_selectivity)
        results.append(r)

    # Sort by composite (lower = better)
    valid = [r for r in results if r.valid]
    valid.sort(key=lambda r: r.composite_score)
    for i, r in enumerate(valid):
        r.rank = i + 1

    return results


def generate_decoy_panel(n: int = 50, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Generate random decoy molecules that should NOT bind GalNAc.

    Decoys: hydrophobic molecules, metal chelators, molecules without
    H-bond donors — things the scorer should rank low.
    """
    import random
    random.seed(seed)

    decoys = [
        ("decoy_hexane", "CCCCCC"),
        ("decoy_cyclohexane", "C1CCCCC1"),
        ("decoy_toluene", "Cc1ccccc1"),
        ("decoy_naphthalene", "c1ccc2ccccc2c1"),
        ("decoy_biphenyl", "c1ccc(-c2ccccc2)cc1"),
        ("decoy_anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("decoy_fluorene", "c1ccc2c(c1)Cc1ccccc1-2"),
        ("decoy_adamantane", "C1C2CC3CC1CC(C2)C3"),
        ("decoy_decalin", "C1CCC2CCCCC2C1"),
        ("decoy_cholesterol_frag", "CC(C)CCCC(C)C1CCC2C1CCC1C3CCC(O)CC3CCC12C"),
        ("decoy_dodecane", "CCCCCCCCCCCC"),
        ("decoy_stearic", "CCCCCCCCCCCCCCCCCC(=O)O"),
        ("decoy_palmitic", "CCCCCCCCCCCCCCCC(=O)O"),
        ("decoy_ferrocene_analog", "c1cccc1"),  # cp ring only
        ("decoy_thiophene", "c1ccsc1"),
        ("decoy_pyrrole", "c1cc[nH]c1"),
        ("decoy_furan", "c1ccoc1"),
        ("decoy_imidazole", "c1cnc[nH]1"),
        ("decoy_triethylamine", "CCN(CC)CC"),
        ("decoy_EDTA_like", "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O"),
        ("decoy_dithiocarbamate", "S=C(S)N(CC)CC"),
        ("decoy_crown_12c4", "C1COCCOCCOCC1"),  # wrong cavity size
        ("decoy_chloroform", "ClC(Cl)Cl"),
        ("decoy_nitrobenzene", "[O-][N+](=O)c1ccccc1"),
        ("decoy_acetophenone", "CC(=O)c1ccccc1"),
        ("decoy_benzaldehyde", "O=Cc1ccccc1"),
        ("decoy_anisole", "COc1ccccc1"),
        ("decoy_diphenylether", "c1ccc(Oc2ccccc2)cc1"),
        ("decoy_dibutylether", "CCCCOCCCC"),
        ("decoy_pentanol", "CCCCCO"),
        ("decoy_octanol", "CCCCCCCCO"),
        ("decoy_diethylamine", "CCNCC"),
        ("decoy_pyridine", "c1ccncc1"),
        ("decoy_quinoline", "c1ccc2ncccc2c1"),
        ("decoy_acridine", "c1ccc2nc3ccccc3cc2c1"),
        ("decoy_caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("decoy_camphor", "CC1(C)C2CCC1(C)C(=O)C2"),
        ("decoy_menthol", "CC(C)C1CCC(C)CC1O"),
        ("decoy_thymol", "Cc1ccc(C(C)C)c(O)c1"),
        ("decoy_eugenol", "C=CCc1ccc(O)c(OC)c1"),
        ("decoy_vanillin", "COc1cc(C=O)ccc1O"),
        ("decoy_coumarin", "O=c1ccc2ccccc2o1"),
        ("decoy_indole", "c1ccc2[nH]ccc2c1"),
        ("decoy_carbazole", "c1ccc2c(c1)[nH]c1ccccc12"),
        ("decoy_acetic_acid", "CC(=O)O"),
        ("decoy_benzoic_acid", "OC(=O)c1ccccc1"),
        ("decoy_succinic_acid", "OC(=O)CCC(=O)O"),
        ("decoy_phthalic_acid", "OC(=O)c1ccccc1C(=O)O"),
        ("decoy_ethanol", "CCO"),
        ("decoy_DMSO", "CS(=O)C"),
    ]

    return decoys[:n]


# ═══════════════════════════════════════════════════════════════════════════
# KNOWN GALNAC BINDER REFERENCE PANEL
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_GALNAC_BINDERS = [
    # Boronic acid receptors (reversible covalent diol binding)
    ("phenylboronic_acid", "OB(O)c1ccccc1"),
    ("naphthalene_boronic", "OB(O)c1ccc2ccccc2c1"),
    ("bis_boronic_ortho", "OB(O)c1ccccc1B(O)O"),

    # Urea/thiourea receptors (H-bond donors for OH recognition)
    ("bis_urea_xylylene", "O=C(N)NCc1cccc(CNC(N)=O)c1"),
    ("isophthalamide", "O=C(N)c1cccc(C(N)=O)c1"),

    # Davis-type synthetic lectin motifs (anthracene + amide)
    ("anthracene_diamide", "O=C(N)c1ccc2cc3ccc(C(N)=O)cc3cc2c1"),

    # Pyrrole-based anion/sugar receptors (NH donors)
    ("calix_pyrrole_simple", "c1cc(c2cc[nH]c2)c(c3cc[nH]c3)[nH]1"),

    # Squaramide (strong H-bond donor pair)
    ("squaramide_phenyl", "O=c1c(Nc2ccccc2)c(=O)c1Nc1ccccc1"),

    # Known lectin-mimics with aromatic + H-bond features
    ("indole_carboxamide", "O=C(N)c1c[nH]c2ccccc12"),
    ("tryptophan_deriv", "NC(Cc1c[nH]c2ccccc12)C(=O)O"),
]


if __name__ == "__main__":
    print("=" * 70)
    print("AACR GalNAc Binder Scoring — Benchmarking Run")
    print("=" * 70)

    # Score known binders
    print("\n── KNOWN GalNAc BINDERS ──")
    known_results = score_candidate_panel(KNOWN_GALNAC_BINDERS)
    for r in sorted(known_results, key=lambda x: x.composite_score):
        if not r.valid:
            continue
        sel = f"sel={r.min_selectivity:+.1f}" if r.selectivity_gaps else ""
        print(f"  #{r.rank:2d} {r.name:30s} dG={r.dg_galnac:+6.1f} "
              f"logKa~{r.log_Ka_est:.1f} att={r.attachability:.2f} {sel}")

    # Score decoys
    print("\n── DECOY PANEL ──")
    decoys = generate_decoy_panel(30)
    decoy_results = score_candidate_panel(decoys)
    for r in sorted(decoy_results, key=lambda x: x.composite_score)[:10]:
        if not r.valid:
            continue
        print(f"  #{r.rank:2d} {r.name:30s} dG={r.dg_galnac:+6.1f} "
              f"logKa~{r.log_Ka_est:.1f} att={r.attachability:.2f}")

    # Enrichment calculation
    print("\n── ENRICHMENT ──")
    known_scores = [r.dg_galnac for r in known_results if r.valid]
    decoy_scores = [r.dg_galnac for r in decoy_results if r.valid]

    if known_scores and decoy_scores:
        avg_known = sum(known_scores) / len(known_scores)
        avg_decoy = sum(decoy_scores) / len(decoy_scores)
        print(f"  Mean dG (known binders): {avg_known:+.2f} kJ/mol")
        print(f"  Mean dG (decoys):        {avg_decoy:+.2f} kJ/mol")
        print(f"  Separation:              {avg_decoy - avg_known:+.2f} kJ/mol")

        # Enrichment at top N
        all_results = known_results + decoy_results
        all_valid = [(r.dg_galnac, r.name.startswith("decoy_") == False, r.name)
                     for r in all_results if r.valid]
        all_valid.sort(key=lambda x: x[0])  # sort by dG (lower = stronger)

        n_known = sum(1 for _, is_known, _ in all_valid if is_known)
        for pct in [0.05, 0.10, 0.20]:
            n_top = max(1, int(len(all_valid) * pct))
            top = all_valid[:n_top]
            n_known_in_top = sum(1 for _, is_known, _ in top if is_known)
            expected = n_known * pct
            enrichment = n_known_in_top / max(expected, 0.001)
            print(f"  Enrichment at top {int(pct*100):2d}%: "
                  f"{n_known_in_top}/{n_top} known "
                  f"(EF={enrichment:.1f}x)")