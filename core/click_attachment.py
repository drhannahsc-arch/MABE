"""
core/click_attachment.py — Click-Chemistry Attachment Site Analysis

Analyzes molecular graphs to identify where click handles can be
installed on binder candidates without disrupting the binding pharmacophore.

For each candidate, determines:
  1. Which atoms are part of the binding pharmacophore (H-bond donors/acceptors, aromatics)
  2. Which atoms are topologically distant from the pharmacophore (handle candidates)
  3. Which click chemistries are compatible at each candidate site
  4. What the predicted steric penalty is for handle installation
  5. What the linker geometry looks like (protrusion direction)

Click chemistries modeled:
  - SPAAC (DBCO-azide): strain-promoted, no catalyst, bioorthogonal
  - CuAAC (alkyne-azide): Cu(I)-catalyzed, fast, but Cu toxicity concern
  - NHS-amine: standard protein conjugation, fast, non-specific
  - Maleimide-thiol: selective, fast, but thiol oxidation concern
  - Tetrazine-TCO: fastest click, bioorthogonal, expensive reagents

Handle installation routes:
  - Primary amine → azide (diazotransfer), NHS-ester conjugation
  - Alcohol/phenol → propargyl ether, azido-PEG-OTs
  - Carboxylic acid → amide coupling → azide-linker
  - Aromatic C-H → directed C-H borylation → Suzuki to linker
  - Aromatic halide → Suzuki/Sonogashira → alkyne/azide linker
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdmolops


# ═══════════════════════════════════════════════════════════════════════════
# PHARMACOPHORE IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

# SMARTS for binding-relevant features
_HBD_SMARTS = Chem.MolFromSmarts('[#7H,#8H,#16H]')          # NH, OH, SH
_BORONIC_SMARTS = Chem.MolFromSmarts('[#5]([OH])([OH])')      # B(OH)2
_CARBOXYLATE_SMARTS = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')  # C(=O)OH
_AMIDE_NH_SMARTS = Chem.MolFromSmarts('[NX3H][CX3](=O)')     # amide NH
_UREA_SMARTS = Chem.MolFromSmarts('[NX3H][CX3](=O)[NX3H]')   # urea
_PRIMARY_AMINE = Chem.MolFromSmarts('[NX3H2;!$([NX3H2][CX3]=[OX1])]')
_SECONDARY_AMINE = Chem.MolFromSmarts('[NX3H1;!$([NX3H1][CX3]=[OX1])]([#6])[#6]')
_HYDROXYL = Chem.MolFromSmarts('[OX2H][#6;!$([#6]=[OX1])]')  # alcohol/phenol OH, not COOH
_THIOL = Chem.MolFromSmarts('[SX2H]')


def identify_pharmacophore(mol) -> Set[int]:
    """Identify atoms that are CORE binding pharmacophore.

    Conservative: only the atoms directly making contacts with target.
    NOT their entire neighborhoods. This leaves peripheral atoms
    available for handle installation.

    Binding-critical atoms:
    - H-bond donor heteroatoms (N-H, O-H) themselves
    - Aromatic carbons bearing C-H that participate in CH-π
    - Boronic acid B and its two oxygens
    - Urea/amide N and C=O atoms (the H-bond pair)
    """
    if mol is None:
        return set()

    pharm = set()

    # H-bond donor heteroatoms only (not their carbon neighbors)
    for match in mol.GetSubstructMatches(_HBD_SMARTS):
        pharm.add(match[0])

    # Aromatic ring atoms with C-H (the CH-π contact face)
    # Only include aromatic Cs that have ≥1 H — these are the contact atoms
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
            pharm.add(atom.GetIdx())

    # Boronic acid: B + directly bonded O atoms
    if _BORONIC_SMARTS:
        for match in mol.GetSubstructMatches(_BORONIC_SMARTS):
            for idx in match:
                pharm.add(idx)

    # Urea NH and C=O: the donor atoms only
    if _UREA_SMARTS:
        for match in mol.GetSubstructMatches(_UREA_SMARTS):
            # match is (N, C, O, N) — protect N and C=O
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() in ('N', 'O'):
                    pharm.add(idx)

    # Amide NH: protect N only
    if _AMIDE_NH_SMARTS:
        for match in mol.GetSubstructMatches(_AMIDE_NH_SMARTS):
            pharm.add(match[0])  # just the N

    return pharm


# ═══════════════════════════════════════════════════════════════════════════
# HANDLE-INSTALLABLE SITES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttachmentSite:
    """A candidate position for click-handle installation."""
    atom_idx: int
    atom_symbol: str
    functional_group: str           # 'primary_amine', 'alcohol', 'COOH', 'aromatic_CH', etc.
    min_dist_to_pharmacophore: int  # topological distance (bonds) to nearest pharmacophore atom
    compatible_chemistries: List[str] = field(default_factory=list)
    installation_route: str = ""    # how to install the handle
    synthetic_penalty: float = 0.0  # 0=trivial, 1=hard
    steric_score: float = 0.0      # 0=occluded, 1=fully accessible
    # Geometry
    n_bonds_to_periphery: int = 0   # bonds to nearest molecular terminus
    is_terminal: bool = False       # degree-1 heavy atom


def find_attachment_sites(mol, pharmacophore: Set[int]) -> List[AttachmentSite]:
    """Find all candidate sites for click-handle installation.

    Strategy: find functional groups that are NOT part of the pharmacophore
    and that can be chemically modified to bear a click handle.
    """
    if mol is None:
        return []

    sites = []
    dm = rdmolops.GetDistanceMatrix(mol)
    n_atoms = mol.GetNumAtoms()

    # Compute distance from each atom to nearest pharmacophore atom
    def min_dist_to_pharm(idx):
        if not pharmacophore:
            return n_atoms  # no pharmacophore = everything is far
        return min(int(dm[idx][p]) for p in pharmacophore if p < n_atoms)

    # Compute distance from each atom to nearest terminal (degree-1) atom
    terminal_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1]

    def dist_to_periphery(idx):
        if not terminal_atoms:
            return 0
        return min(int(dm[idx][t]) for t in terminal_atoms)

    # ── Search for installable functional groups ──────────────────────

    # Primary amines (not in pharmacophore): can install azide, NHS-ester
    if _PRIMARY_AMINE:
        for match in mol.GetSubstructMatches(_PRIMARY_AMINE):
            idx = match[0]
            if idx in pharmacophore:
                continue
            d = min_dist_to_pharm(idx)
            sites.append(AttachmentSite(
                atom_idx=idx,
                atom_symbol='N',
                functional_group='primary_amine',
                min_dist_to_pharmacophore=d,
                compatible_chemistries=['SPAAC', 'NHS-amine', 'CuAAC'],
                installation_route='diazotransfer → azide; or direct NHS-ester conjugation',
                synthetic_penalty=0.1,
                steric_score=_steric_accessibility(mol, idx),
                n_bonds_to_periphery=dist_to_periphery(idx),
                is_terminal=(mol.GetAtomWithIdx(idx).GetDegree() == 1),
            ))

    # Alcohols/phenols (not in pharmacophore)
    if _HYDROXYL:
        for match in mol.GetSubstructMatches(_HYDROXYL):
            idx = match[0]
            if idx in pharmacophore:
                continue
            d = min_dist_to_pharm(idx)
            is_phenol = mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetIsAromatic() if mol.GetAtomWithIdx(idx).GetDegree() > 0 else False
            sites.append(AttachmentSite(
                atom_idx=idx,
                atom_symbol='O',
                functional_group='phenol' if is_phenol else 'alcohol',
                min_dist_to_pharmacophore=d,
                compatible_chemistries=['CuAAC', 'SPAAC'],
                installation_route='propargyl ether (Williamson); or azido-PEG-OTs',
                synthetic_penalty=0.2 if not is_phenol else 0.3,
                steric_score=_steric_accessibility(mol, idx),
                n_bonds_to_periphery=dist_to_periphery(idx),
                is_terminal=(mol.GetAtomWithIdx(idx).GetDegree() == 1),
            ))

    # Carboxylic acids (not in pharmacophore)
    if _CARBOXYLATE_SMARTS:
        for match in mol.GetSubstructMatches(_CARBOXYLATE_SMARTS):
            c_idx = match[0]
            if c_idx in pharmacophore:
                continue
            d = min_dist_to_pharm(c_idx)
            sites.append(AttachmentSite(
                atom_idx=c_idx,
                atom_symbol='C(=O)OH',
                functional_group='carboxylic_acid',
                min_dist_to_pharmacophore=d,
                compatible_chemistries=['SPAAC', 'CuAAC', 'NHS-amine'],
                installation_route='EDC/NHS activation → azide-amine; or amide coupling to propargylamine',
                synthetic_penalty=0.15,
                steric_score=_steric_accessibility(mol, c_idx),
                n_bonds_to_periphery=dist_to_periphery(c_idx),
            ))

    # Thiols (not in pharmacophore)
    if _THIOL:
        for match in mol.GetSubstructMatches(_THIOL):
            idx = match[0]
            if idx in pharmacophore:
                continue
            d = min_dist_to_pharm(idx)
            sites.append(AttachmentSite(
                atom_idx=idx,
                atom_symbol='S',
                functional_group='thiol',
                min_dist_to_pharmacophore=d,
                compatible_chemistries=['maleimide-thiol', 'SPAAC'],
                installation_route='direct maleimide conjugation; or thiol-ene to azide-alkene',
                synthetic_penalty=0.1,
                steric_score=_steric_accessibility(mol, idx),
                n_bonds_to_periphery=dist_to_periphery(idx),
                is_terminal=(mol.GetAtomWithIdx(idx).GetDegree() == 1),
            ))

    # Aromatic C-H positions not in pharmacophore: can do C-H functionalization
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            continue
        if atom.GetIdx() in pharmacophore:
            continue
        if atom.GetTotalNumHs() == 0:
            continue  # no available C-H
        d = min_dist_to_pharm(atom.GetIdx())
        if d < 2:
            continue  # immediately adjacent to pharmacophore atom
        sites.append(AttachmentSite(
            atom_idx=atom.GetIdx(),
            atom_symbol='Ar-H',
            functional_group='aromatic_CH',
            min_dist_to_pharmacophore=d,
            compatible_chemistries=['CuAAC', 'SPAAC'],
            installation_route='directed C-H borylation → Suzuki to azide/alkyne linker; or halogenation → cross-coupling',
            synthetic_penalty=0.5,  # C-H functionalization is harder
            steric_score=_steric_accessibility(mol, atom.GetIdx()),
            n_bonds_to_periphery=dist_to_periphery(atom.GetIdx()),
        ))

    # Aromatic C without H but not in pharmacophore (bridgehead, substituted)
    # These are existing substitution points for Suzuki/Sonogashira
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            continue
        if atom.GetIdx() in pharmacophore:
            continue
        if atom.GetTotalNumHs() > 0:
            continue  # handled above
        if atom.GetSymbol() != 'C':
            continue
        # Check if this is bonded to a non-aromatic carbon (= existing substituent position)
        d = min_dist_to_pharm(atom.GetIdx())
        if d < 2:
            continue
        for neighbor in atom.GetNeighbors():
            if not neighbor.GetIsAromatic() and neighbor.GetSymbol() == 'C':
                sites.append(AttachmentSite(
                    atom_idx=atom.GetIdx(),
                    atom_symbol='Ar-C',
                    functional_group='aromatic_substituted',
                    min_dist_to_pharmacophore=d,
                    compatible_chemistries=['CuAAC', 'SPAAC'],
                    installation_route='extend existing substituent with PEG-azide linker',
                    synthetic_penalty=0.3,
                    steric_score=_steric_accessibility(mol, atom.GetIdx()) * 0.8,
                    n_bonds_to_periphery=dist_to_periphery(atom.GetIdx()),
                ))
                break  # one site per aromatic position

    # Secondary amine (non-pharmacophore): can install handle via alkylation
    if _SECONDARY_AMINE:
        for match in mol.GetSubstructMatches(_SECONDARY_AMINE):
            idx = match[0]
            if idx in pharmacophore:
                continue
            d = min_dist_to_pharm(idx)
            sites.append(AttachmentSite(
                atom_idx=idx,
                atom_symbol='NH',
                functional_group='secondary_amine',
                min_dist_to_pharmacophore=d,
                compatible_chemistries=['SPAAC', 'CuAAC'],
                installation_route='N-alkylation with azide-bromide or propargyl bromide',
                synthetic_penalty=0.25,
                steric_score=_steric_accessibility(mol, idx),
                n_bonds_to_periphery=dist_to_periphery(idx),
            ))

    # Benzylic CH2 (sp3 carbon bonded to aromatic ring, not in pharmacophore)
    # These are excellent handle installation points: radical bromination → SN2 with azide
    benzylic = Chem.MolFromSmarts('[CH2]([c])')
    if benzylic:
        for match in mol.GetSubstructMatches(benzylic):
            idx = match[0]
            if idx in pharmacophore:
                continue
            d = min_dist_to_pharm(idx)
            sites.append(AttachmentSite(
                atom_idx=idx,
                atom_symbol='CH2',
                functional_group='benzylic_CH2',
                min_dist_to_pharmacophore=d,
                compatible_chemistries=['SPAAC', 'CuAAC'],
                installation_route='radical bromination (NBS) → SN2 with NaN3; or alkylation',
                synthetic_penalty=0.3,
                steric_score=_steric_accessibility(mol, idx),
                n_bonds_to_periphery=dist_to_periphery(idx),
            ))

    # Terminal aliphatic carbons (methyl groups not in pharmacophore)
    # Can be extended via homologation or replaced with azide-bearing chain
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            continue
        if atom.GetIdx() in pharmacophore:
            continue
        if atom.GetSymbol() != 'C':
            continue
        if atom.GetDegree() != 1:
            continue  # must be terminal
        d = min_dist_to_pharm(atom.GetIdx())
        if d < 2:
            continue
        sites.append(AttachmentSite(
            atom_idx=atom.GetIdx(),
            atom_symbol='CH3',
            functional_group='terminal_methyl',
            min_dist_to_pharmacophore=d,
            compatible_chemistries=['CuAAC', 'SPAAC'],
            installation_route='replace with azide-alkyl chain; or extend via cross-metathesis',
            synthetic_penalty=0.45,
            steric_score=1.0,  # terminal = maximally accessible
            n_bonds_to_periphery=0,
            is_terminal=True,
        ))

    return sites


def _steric_accessibility(mol, atom_idx: int) -> float:
    """Estimate steric accessibility of an atom (0=buried, 1=exposed).

    Heuristic: based on neighbor count and ring membership.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    degree = atom.GetDegree()
    in_ring = atom.IsInRing()

    # Fewer neighbors = more accessible
    if degree <= 1:
        access = 1.0
    elif degree == 2:
        access = 0.8
    elif degree == 3:
        access = 0.5
    else:
        access = 0.3

    # Ring atoms are slightly less accessible
    if in_ring:
        access *= 0.8

    return access


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE ATTACHMENT SCORING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttachmentAnalysis:
    """Complete attachment analysis for a molecule."""
    smiles: str
    name: str = ""
    n_pharmacophore_atoms: int = 0
    n_total_heavy_atoms: int = 0
    pharmacophore_fraction: float = 0.0
    molecular_diameter_bonds: int = 0
    # Sites
    n_sites_found: int = 0
    sites: List[AttachmentSite] = field(default_factory=list)
    best_site: Optional[AttachmentSite] = None
    # Composite scores
    best_site_score: float = 0.0     # 0-1, quality of best attachment point
    chemistry_diversity: int = 0      # number of distinct compatible click chemistries
    max_pharmacophore_distance: int = 0  # bonds from best site to pharmacophore
    # Per-chemistry best sites
    best_per_chemistry: Dict[str, AttachmentSite] = field(default_factory=dict)
    # Composite
    composite_attachability: float = 0.0  # 0-1, overall attachment readiness


def analyze_attachment(smiles: str, name: str = "") -> AttachmentAnalysis:
    """Full attachment analysis for a candidate molecule."""
    result = AttachmentAnalysis(smiles=smiles, name=name)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    # Add Hs for accurate donor/acceptor counting, then remove for graph analysis
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=10)
    mol = Chem.RemoveHs(mol_h) if mol_h.GetNumConformers() > 0 else mol

    result.n_total_heavy_atoms = mol.GetNumHeavyAtoms()

    # Pharmacophore identification
    pharmacophore = identify_pharmacophore(mol)
    result.n_pharmacophore_atoms = len(pharmacophore)
    result.pharmacophore_fraction = len(pharmacophore) / max(1, result.n_total_heavy_atoms)

    # Molecular diameter
    dm = rdmolops.GetDistanceMatrix(mol)
    result.molecular_diameter_bonds = int(dm.max()) if dm.size > 0 else 0

    # Find attachment sites
    sites = find_attachment_sites(mol, pharmacophore)
    result.n_sites_found = len(sites)
    result.sites = sites

    if not sites:
        return result

    # Score each site: combination of distance, accessibility, synthetic ease
    for site in sites:
        site_score = 0.0
        # Distance bonus: farther from pharmacophore = better
        max_possible_dist = max(result.molecular_diameter_bonds, 1)
        dist_score = min(1.0, site.min_dist_to_pharmacophore / max(3.0, max_possible_dist * 0.5))
        site_score += 0.4 * dist_score

        # Steric accessibility
        site_score += 0.25 * site.steric_score

        # Synthetic ease (1 - penalty)
        site_score += 0.2 * (1.0 - site.synthetic_penalty)

        # Terminal bonus
        if site.is_terminal:
            site_score += 0.1

        # Chemistry diversity bonus
        site_score += 0.05 * min(1.0, len(site.compatible_chemistries) / 3.0)

        site._score = site_score

    # Sort by score, best first
    sites.sort(key=lambda s: -s._score)
    result.best_site = sites[0]
    result.best_site_score = sites[0]._score
    result.max_pharmacophore_distance = sites[0].min_dist_to_pharmacophore

    # Chemistry diversity: how many distinct click chemistries are reachable?
    all_chems = set()
    for site in sites:
        all_chems.update(site.compatible_chemistries)
    result.chemistry_diversity = len(all_chems)

    # Best site per chemistry
    for chem in all_chems:
        chem_sites = [s for s in sites if chem in s.compatible_chemistries]
        if chem_sites:
            result.best_per_chemistry[chem] = chem_sites[0]

    # Composite attachability
    result.composite_attachability = _composite_score(result)

    return result


def _composite_score(analysis: AttachmentAnalysis) -> float:
    """Compute composite attachability score (0-1)."""
    if analysis.n_sites_found == 0:
        return 0.0

    score = 0.0

    # Best site quality (0.4 weight)
    score += 0.4 * analysis.best_site_score

    # Has site ≥3 bonds from pharmacophore (0.2 weight)
    if analysis.max_pharmacophore_distance >= 3:
        score += 0.2
    elif analysis.max_pharmacophore_distance >= 2:
        score += 0.1

    # Chemistry diversity (0.15 weight)
    score += 0.15 * min(1.0, analysis.chemistry_diversity / 4.0)

    # Multiple sites available (0.1 weight) — backup options
    score += 0.1 * min(1.0, analysis.n_sites_found / 3.0)

    # Pharmacophore fraction < 0.7 (0.15 weight) — molecule not ALL pharmacophore
    if analysis.pharmacophore_fraction < 0.5:
        score += 0.15
    elif analysis.pharmacophore_fraction < 0.7:
        score += 0.08

    return min(1.0, score)


# ═══════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_panel(candidates: List[Tuple[str, str]]) -> List[AttachmentAnalysis]:
    """Analyze attachment sites for a panel of candidates."""
    results = []
    for name, smiles in candidates:
        r = analyze_attachment(smiles, name)
        results.append(r)
    return results


def attachment_report(analysis: AttachmentAnalysis) -> str:
    """Human-readable attachment analysis report."""
    lines = []
    a = analysis
    lines.append(f"Attachment Analysis: {a.name}")
    lines.append(f"  SMILES: {a.smiles}")
    lines.append(f"  Heavy atoms: {a.n_total_heavy_atoms} "
                 f"(pharmacophore: {a.n_pharmacophore_atoms}, "
                 f"{a.pharmacophore_fraction:.0%})")
    lines.append(f"  Molecular diameter: {a.molecular_diameter_bonds} bonds")
    lines.append(f"  Attachment sites found: {a.n_sites_found}")
    lines.append(f"  Compatible click chemistries: {a.chemistry_diversity}")
    lines.append(f"  Composite attachability: {a.composite_attachability:.3f}")

    if a.best_site:
        bs = a.best_site
        lines.append(f"  Best site:")
        lines.append(f"    Position: atom {bs.atom_idx} ({bs.atom_symbol})")
        lines.append(f"    Functional group: {bs.functional_group}")
        lines.append(f"    Distance to pharmacophore: {bs.min_dist_to_pharmacophore} bonds")
        lines.append(f"    Compatible: {', '.join(bs.compatible_chemistries)}")
        lines.append(f"    Installation: {bs.installation_route}")
        lines.append(f"    Synthetic penalty: {bs.synthetic_penalty:.2f}")

    if a.best_per_chemistry:
        lines.append(f"  Per-chemistry best sites:")
        for chem, site in sorted(a.best_per_chemistry.items()):
            lines.append(f"    {chem:20s}: atom {site.atom_idx} ({site.functional_group}), "
                        f"dist={site.min_dist_to_pharmacophore}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: REPLACE CRUDE ATTACHABILITY IN GALNAC SCORER
# ═══════════════════════════════════════════════════════════════════════════

def attachability_score(smiles: str) -> float:
    """Drop-in replacement for _compute_attachability.

    Returns 0-1 composite attachability score.
    """
    analysis = analyze_attachment(smiles)
    return analysis.composite_attachability


def attachability_with_details(smiles: str, name: str = "") -> Tuple[float, AttachmentAnalysis]:
    """Returns both the score and the full analysis."""
    analysis = analyze_attachment(smiles, name)
    return analysis.composite_attachability, analysis


if __name__ == "__main__":
    print("=" * 70)
    print("CLICK ATTACHMENT SITE ANALYSIS")
    print("=" * 70)

    test_molecules = [
        ("bis_urea_xylylene", "O=C(N)NCc1cccc(CNC(N)=O)c1"),
        ("phenylboronic_acid", "OB(O)c1ccccc1"),
        ("anthracene_diamide", "O=C(N)c1ccc2cc3ccc(C(N)=O)cc3cc2c1"),
        ("tryptophan_deriv", "NC(Cc1c[nH]c2ccccc12)C(=O)O"),
        ("EDTA", "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O"),
        ("triglycine", "NCC(=O)NCC(=O)NCC(=O)O"),
        ("naphthalene_boronic", "OB(O)c1ccc2ccccc2c1"),
        ("squaramide_phenyl", "O=c1c(Nc2ccccc2)c(=O)c1Nc1ccccc1"),
        ("indole_carboxamide", "O=C(N)c1c[nH]c2ccccc12"),
        ("adamantane_COOH", "OC(=O)C12CC3CC(CC(C3)C1)C2"),
    ]

    for name, smiles in test_molecules:
        analysis = analyze_attachment(smiles, name)
        print(f"\n{attachment_report(analysis)}")

    # Summary table
    print(f"\n{'═' * 70}")
    print(f"SUMMARY TABLE")
    print(f"{'═' * 70}")
    print(f"{'Name':>25s} {'Sites':>5s} {'BestDist':>8s} {'Chems':>5s} "
          f"{'PharmFrac':>9s} {'Score':>6s} {'BestGroup':>15s}")
    for name, smiles in test_molecules:
        a = analyze_attachment(smiles, name)
        bg = a.best_site.functional_group if a.best_site else "none"
        bd = a.max_pharmacophore_distance
        print(f"{name:>25s} {a.n_sites_found:5d} {bd:8d} "
              f"{a.chemistry_diversity:5d} {a.pharmacophore_fraction:9.2f} "
              f"{a.composite_attachability:6.3f} {bg:>15s}")