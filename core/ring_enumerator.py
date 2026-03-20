"""
core/ring_enumerator.py -- Physics-driven ring system enumeration for all MABE modalities.

Provides:
  1. Curated ring systems (1-4 fused rings, all common fusion patterns)
  2. Systematic positional decoration (enumerate all substitutable H positions)
  3. Physics-driven filtering (aromatic area, heteroatom content, HSAB)
  4. Unified decorator library with both glycan and metal coordination properties
  5. Conversion to Backbone/Arm format for existing de_novo_generator pipeline

This module is modality-agnostic. Modality-specific demand logic lives in
core/demand_generator.py (metal, host-guest) and glycan/demand_grammar.py (glycan).

No ML. No database lookups. Pure combinatorial chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, RWMol
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Ring system catalog (identical to glycan version + validated)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RingSystem:
    """A fused aromatic ring system."""
    name: str
    smiles: str
    n_rings: int
    n_aromatic_atoms: int
    n_substitutable: int
    has_N: bool
    has_O: bool
    has_S: bool
    has_NH: bool
    category: str  # "carbocyclic", "N-hetero", "O-hetero", "S-hetero", "mixed"
    sub_positions: Tuple[int, ...] = ()


def _classify(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "unknown"
    has_n = any(a.GetSymbol() == "N" for a in mol.GetAtoms() if a.GetIsAromatic())
    has_o = any(a.GetSymbol() == "O" for a in mol.GetAtoms() if a.GetIsAromatic())
    has_s = any(a.GetSymbol() == "S" for a in mol.GetAtoms() if a.GetIsAromatic())
    if has_n and has_o:
        return "mixed"
    if has_n:
        return "N-hetero"
    if has_o:
        return "O-hetero"
    if has_s:
        return "S-hetero"
    return "carbocyclic"


def _build_ring_system(name: str, smiles: str) -> Optional[RingSystem]:
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    positions = tuple(
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetTotalNumHs() > 0
    )
    has_N = any(a.GetSymbol() == "N" for a in mol.GetAtoms() if a.GetIsAromatic())
    has_O = any(a.GetSymbol() == "O" for a in mol.GetAtoms() if a.GetIsAromatic())
    has_S = any(a.GetSymbol() == "S" for a in mol.GetAtoms() if a.GetIsAromatic())
    has_NH = any(
        a.GetSymbol() == "N" and a.GetTotalNumHs() > 0
        for a in mol.GetAtoms() if a.GetIsAromatic()
    )
    return RingSystem(
        name=name, smiles=Chem.MolToSmiles(mol),
        n_rings=n_rings, n_aromatic_atoms=n_arom,
        n_substitutable=len(positions),
        has_N=has_N, has_O=has_O, has_S=has_S, has_NH=has_NH,
        category=_classify(smiles), sub_positions=positions,
    )


_RING_DEFS: Dict[str, str] = {
    # 1-ring (6-membered)
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1ncncn1",
    "pyrazine": "c1cnccn1",
    "pyridazine": "c1ccnnc1",
    "triazine-1,3,5": "c1ncncn1",
    # 1-ring (5-membered)
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1c[nH]cn1",
    "oxazole": "c1cocn1",
    "thiazole": "c1cscn1",
    "pyrazole": "c1cn[nH]c1",
    # 2-ring fused
    "naphthalene": "c1ccc2ccccc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    "isoquinoline": "c1ccc2ccncc2c1",
    "quinoxaline": "c1ccc2nccnc2c1",
    "quinazoline": "c1ccc2ncncc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "isoindole": "c1ccc2c[nH]cc2c1",
    "benzofuran": "c1ccc2occc2c1",
    "benzothiophene": "c1ccc2sccc2c1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "benzoxazole": "c1ccc2ocnc2c1",
    "benzothiazole": "c1ccc2scnc2c1",
    "purine": "c1nc2[nH]cnc2cn1",
    # 3-ring fused (linear)
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc1ccccc12",
    "acridine": "c1ccc2nc3ccccc3cc2c1",
    "phenanthridine": "c1ccc2c(c1)ccc1ncccc12",
    "carbazole": "c1ccc2c(c1)[nH]c1ccccc12",
    "dibenzofuran": "c1ccc2c(c1)oc1ccccc12",
    "dibenzothiophene": "c1ccc2c(c1)sc1ccccc12",
    "fluorene": "c1ccc2c(c1)Cc1ccccc1-2",
    "xanthene": "c1ccc2c(c1)oc1ccccc12",
    "phenazine": "c1ccc2nc3ccccc3nc2c1",
    "phenothiazine": "c1ccc2c(c1)Sc1ccccc1N2",
    "fluoranthene": "c1ccc2-c3cccc3Cc3ccccc3-2c1",
    # 4-ring fused
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "triphenylene": "c1ccc2c(c1)c1ccccc1c1ccccc21",
    "chrysene": "c1ccc2ccc3ccc4ccccc4c3c2c1",
}


_CATALOG: Optional[Dict[str, RingSystem]] = None


def _ensure_catalog() -> Dict[str, RingSystem]:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required for ring enumeration")
    _CATALOG = {}
    for name, smi in _RING_DEFS.items():
        rs = _build_ring_system(name, smi)
        if rs is not None:
            _CATALOG[name] = rs
    return _CATALOG


def get_catalog() -> Dict[str, RingSystem]:
    return dict(_ensure_catalog())


def get_ring_system(name: str) -> RingSystem:
    cat = _ensure_catalog()
    if name not in cat:
        raise KeyError(f"Unknown ring system '{name}'")
    return cat[name]


def list_ring_systems(
    min_rings: int = 1, max_rings: int = 99,
    category: Optional[str] = None,
    min_aromatic_atoms: int = 0, max_aromatic_atoms: int = 999,
) -> List[RingSystem]:
    cat = _ensure_catalog()
    results = []
    for rs in cat.values():
        if rs.n_rings < min_rings or rs.n_rings > max_rings:
            continue
        if category and rs.category != category:
            continue
        if rs.n_aromatic_atoms < min_aromatic_atoms:
            continue
        if rs.n_aromatic_atoms > max_aromatic_atoms:
            continue
        results.append(rs)
    return sorted(results, key=lambda r: (r.n_rings, r.n_aromatic_atoms, r.name))


# ---------------------------------------------------------------------------
# Positional decoration
# ---------------------------------------------------------------------------

@dataclass
class DecoratedScaffold:
    ring_system: str
    smiles: str
    n_sites: int
    positions: Tuple[int, ...]
    ring_smiles: str


def _add_dummy_at_positions(smiles: str, positions: List[int]) -> Optional[str]:
    """Add isotope-labeled [1*], [2*], ... dummy atoms at specified positions."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = RWMol(mol)
    # Add dummies in reverse index order, label with isotope 1, 2, ...
    for label, pos in enumerate(sorted(positions, reverse=True), 1):
        dummy = Chem.Atom(0)
        dummy.SetIsotope(len(positions) - label + 1)  # first pos gets [1*], etc.
        dummy_idx = rw.AddAtom(dummy)
        rw.AddBond(pos, dummy_idx, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None


def enumerate_decorated(
    ring_name: str, n_sites: int = 2, max_scaffolds: int = 50,
) -> List[DecoratedScaffold]:
    from itertools import combinations
    rs = get_ring_system(ring_name)
    if n_sites > rs.n_substitutable:
        return []
    seen = set()
    results = []
    for combo in combinations(rs.sub_positions, n_sites):
        smi = _add_dummy_at_positions(rs.smiles, list(combo))
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        results.append(DecoratedScaffold(
            ring_system=ring_name, smiles=can, n_sites=n_sites,
            positions=combo, ring_smiles=rs.smiles,
        ))
        if len(results) >= max_scaffolds:
            break
    return results


def enumerate_all_decorated(
    n_sites: int = 2, min_rings: int = 1, max_rings: int = 4,
    category: Optional[str] = None, max_per_system: int = 20,
) -> List[DecoratedScaffold]:
    systems = list_ring_systems(min_rings=min_rings, max_rings=max_rings, category=category)
    all_scaffolds = []
    for rs in systems:
        all_scaffolds.extend(enumerate_decorated(rs.name, n_sites, max_per_system))
    return all_scaffolds


# ---------------------------------------------------------------------------
# Physics filter
# ---------------------------------------------------------------------------

@dataclass
class PhysicsFilter:
    min_aromatic_atoms: int = 0
    max_aromatic_atoms: int = 999
    min_rings: int = 1
    max_rings: int = 4
    require_large_aromatic: bool = False
    allow_heteroatoms: bool = True
    require_NH_donor: bool = False
    categories: Optional[List[str]] = None


def enumerate_physics_filtered(
    n_sites: int = 2, pfilter: Optional[PhysicsFilter] = None,
    max_per_system: int = 20, max_total: int = 500,
) -> List[DecoratedScaffold]:
    if pfilter is None:
        pfilter = PhysicsFilter()
    systems = list_ring_systems(
        min_rings=pfilter.min_rings, max_rings=pfilter.max_rings,
        min_aromatic_atoms=pfilter.min_aromatic_atoms,
        max_aromatic_atoms=pfilter.max_aromatic_atoms,
    )
    if pfilter.require_large_aromatic:
        systems = [s for s in systems if s.n_aromatic_atoms >= 10]
    if not pfilter.allow_heteroatoms:
        systems = [s for s in systems if s.category == "carbocyclic"]
    if pfilter.require_NH_donor:
        systems = [s for s in systems if s.has_NH]
    if pfilter.categories:
        systems = [s for s in systems if s.category in pfilter.categories]
    all_scaffolds = []
    for rs in systems:
        all_scaffolds.extend(enumerate_decorated(rs.name, n_sites, max_per_system))
        if len(all_scaffolds) >= max_total:
            break
    return all_scaffolds[:max_total]


# ---------------------------------------------------------------------------
# Unified decorator library (glycan + metal properties)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decorator:
    """Arm fragment with properties for all modalities."""
    name: str
    smiles: str
    category: str       # "HBD", "HBA", "aromatic", "boronic", "hydrophobic", "linker", "donor"
    # Glycan-relevant
    n_hbd: int = 0
    n_hba: int = 0
    n_urea: int = 0
    has_boronic: bool = False
    # Metal-relevant
    donor_subtypes: Tuple[str, ...] = ()
    donor_element: str = ""
    hardness: str = ""   # "hard", "borderline", "soft", ""


DECORATOR_LIBRARY: List[Decorator] = [
    # ── H-bond donors (glycan) / hard O donors (metal) ──
    Decorator("urea", "*NC(=O)N", "HBD", n_hbd=2, n_hba=1, n_urea=1,
              donor_subtypes=("O_carbonyl",), donor_element="O", hardness="hard"),
    Decorator("amide-NH", "*NC(=O)C", "HBD", n_hbd=1, n_hba=1,
              donor_subtypes=("O_carbonyl",), donor_element="O", hardness="hard"),
    Decorator("hydroxyl", "*O", "HBD", n_hbd=1, n_hba=1,
              donor_subtypes=("O_hydroxyl",), donor_element="O", hardness="hard"),
    Decorator("amino", "*N", "HBD", n_hbd=1, n_hba=1,
              donor_subtypes=("N_amine",), donor_element="N", hardness="borderline"),
    Decorator("aminomethyl", "*CN", "HBD", n_hbd=1, n_hba=1,
              donor_subtypes=("N_amine",), donor_element="N", hardness="borderline"),
    Decorator("carbamate", "*OC(=O)N", "HBD", n_hbd=1, n_hba=2,
              donor_subtypes=("O_carbonyl",), donor_element="O", hardness="hard"),
    Decorator("sulfonamide", "*S(=O)(=O)N", "HBD", n_hbd=1, n_hba=2,
              donor_subtypes=("O_sulfonyl",), donor_element="O", hardness="hard"),
    Decorator("guanidinium", "*NC(=N)N", "HBD", n_hbd=3, n_hba=1,
              donor_subtypes=("N_amine",), donor_element="N", hardness="borderline"),
    Decorator("hydrazide", "*NNC(=O)C", "HBD", n_hbd=2, n_hba=1,
              donor_subtypes=("N_amine", "O_carbonyl"), donor_element="N", hardness="borderline"),
    # ── H-bond acceptors / metal donors ──
    Decorator("methoxy", "*OC", "HBA", n_hba=1,
              donor_subtypes=("O_ether",), donor_element="O", hardness="hard"),
    Decorator("acetyl", "*C(=O)C", "HBA", n_hba=1,
              donor_subtypes=("O_carbonyl",), donor_element="O", hardness="hard"),
    Decorator("nitrile", "*C#N", "HBA", n_hba=1,
              donor_subtypes=("N_nitrile",), donor_element="N", hardness="borderline"),
    Decorator("carboxylate", "*C(=O)O", "HBA", n_hbd=1, n_hba=2,
              donor_subtypes=("O_carboxylate",), donor_element="O", hardness="hard"),
    # ── Metal-specific donors ──
    Decorator("2-pyridylmethyl", "*Cc1ccccn1", "donor", n_hba=1,
              donor_subtypes=("N_pyridine",), donor_element="N", hardness="borderline"),
    Decorator("imidazolylmethyl", "*Cc1cnc[nH]1", "donor", n_hbd=1, n_hba=1,
              donor_subtypes=("N_imidazole",), donor_element="N", hardness="borderline"),
    Decorator("thioether", "*CSC", "donor",
              donor_subtypes=("S_thioether",), donor_element="S", hardness="soft"),
    Decorator("thiol", "*CS", "donor", n_hbd=1,
              donor_subtypes=("S_thiol",), donor_element="S", hardness="soft"),
    Decorator("phosphonate", "*CP(=O)(O)O", "donor", n_hbd=2, n_hba=3,
              donor_subtypes=("O_phosphate",), donor_element="O", hardness="hard"),
    Decorator("hydroxamate", "*CC(=O)NO", "donor", n_hbd=1, n_hba=2,
              donor_subtypes=("O_hydroxamate", "O_hydroxamate"), donor_element="O", hardness="hard"),
    Decorator("catechol", "*c1ccc(O)c(O)c1", "donor", n_hbd=2, n_hba=2,
              donor_subtypes=("O_catecholate", "O_catecholate"), donor_element="O", hardness="hard"),
    Decorator("8HQ", "*c1ccc2cccc(O)c2n1", "donor", n_hbd=1, n_hba=2,
              donor_subtypes=("N_pyridine", "O_phenolate"), donor_element="N", hardness="borderline"),
    # ── Boronic acid ──
    Decorator("boronic-acid", "*B(O)O", "boronic", n_hbd=2, n_hba=2, has_boronic=True),
    Decorator("boronate-ester", "*B1OC(C)(C)CO1", "boronic", n_hba=2, has_boronic=True),
    # ── Hydrophobic / spacers ──
    Decorator("methyl", "*C", "hydrophobic"),
    Decorator("ethyl", "*CC", "hydrophobic"),
    Decorator("phenyl", "*c1ccccc1", "aromatic", n_hba=0),
    # ── Click handles / linkers ──
    Decorator("azide-PEG", "*CCOCCOCCCN=[N+]=[N-]", "linker", n_hba=5),
    Decorator("alkyne", "*C#C", "linker"),
    # ── Dual-function ──
    Decorator("aminoethanol", "*NCCO", "HBD", n_hbd=2, n_hba=2,
              donor_subtypes=("N_amine", "O_hydroxyl"), donor_element="N", hardness="borderline"),
    Decorator("diurea", "*NC(=O)NNC(=O)N", "HBD", n_hbd=4, n_hba=2, n_urea=2,
              donor_subtypes=("O_carbonyl", "O_carbonyl"), donor_element="O", hardness="hard"),
    Decorator("picolylamine", "*NCc1ccccn1", "donor", n_hbd=1, n_hba=2,
              donor_subtypes=("N_amine", "N_pyridine"), donor_element="N", hardness="borderline"),
]


def get_decorators(
    categories: Optional[List[str]] = None,
    require_hbd: bool = False,
    require_boronic: bool = False,
    hardness: Optional[str] = None,
    donor_element: Optional[str] = None,
) -> List[Decorator]:
    """Filter decorator library by properties."""
    results = list(DECORATOR_LIBRARY)
    if categories:
        results = [d for d in results if d.category in categories]
    if require_hbd:
        results = [d for d in results if d.n_hbd > 0]
    if require_boronic:
        results = [d for d in results if d.has_boronic]
    if hardness:
        results = [d for d in results if d.hardness == hardness or d.hardness == ""]
    if donor_element:
        results = [d for d in results if d.donor_element == donor_element]
    return results


# ---------------------------------------------------------------------------
# Conversion to existing Backbone/Arm format
# ---------------------------------------------------------------------------

def scaffold_to_backbone(ds: DecoratedScaffold) -> "Backbone":
    """Convert DecoratedScaffold to de_novo_generator.Backbone."""
    from core.de_novo_generator import Backbone
    return Backbone(
        name=f"ring:{ds.ring_system}",
        smiles=ds.smiles,
        n_sites=ds.n_sites,
        category="cyclic",
        notes=f"Ring enumerator: {ds.ring_system} pos={ds.positions}",
    )


def decorator_to_arm(dec: Decorator) -> "Arm":
    """Convert Decorator to de_novo_generator.Arm."""
    from core.de_novo_generator import Arm
    return Arm(
        name=f"dec:{dec.name}",
        smiles=dec.smiles,
        donor_subtypes=list(dec.donor_subtypes) if dec.donor_subtypes else [],
        donor_element=dec.donor_element or "C",
        hardness=dec.hardness or "borderline",
        category=dec.category,
    )


def grammar_backbones(
    n_sites: int = 2,
    pfilter: Optional[PhysicsFilter] = None,
    max_per_system: int = 10,
    max_total: int = 200,
) -> list:
    """Generate Backbone objects from ring enumerator for use in enumerate_molecules."""
    scaffolds = enumerate_physics_filtered(
        n_sites=n_sites, pfilter=pfilter,
        max_per_system=max_per_system, max_total=max_total,
    )
    return [scaffold_to_backbone(s) for s in scaffolds]


def grammar_arms(
    categories: Optional[List[str]] = None,
    hardness: Optional[str] = None,
    donor_element: Optional[str] = None,
) -> list:
    """Generate Arm objects from decorator library for use in enumerate_molecules."""
    decs = get_decorators(
        categories=categories, hardness=hardness, donor_element=donor_element,
    )
    return [decorator_to_arm(d) for d in decs]
