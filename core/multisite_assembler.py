"""
core/multisite_assembler.py -- 3/4-site scaffold enumeration for complex receptors.

Extends ring_enumerator with:
  - Tripodal scaffolds (TREN, triaminobenzene, trisubstituted-cyclohexane)
  - Macrocyclic scaffolds (cyclam, cyclen, crown ether ring sizes)
  - Cage-like scaffolds (Davis-type bis-wall, tris-wall, cryptand topologies)
  - 4-site scaffolds for tetradentate chelators

These are NOT in the ring enumerator because they aren't simple ring decorations.
They are multi-component topologies assembled from 2+ ring systems + linkers.

All SMILES are RDKit-validated. Compatible with existing Backbone/Arm pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from core.de_novo_generator import Backbone


# ---------------------------------------------------------------------------
# Tripodal scaffolds (3-site, C3-symmetric or pseudo-C3)
# ---------------------------------------------------------------------------

TRIPODAL_BACKBONES: List[Backbone] = [
    # TREN: tris(2-aminoethyl)amine — classic tripodal for metals
    Backbone("TREN", "[1*]NCCN([2*])CCN([3*])CC", 3, "tripodal",
             "Tris(2-aminoethyl)amine, Fe3+/Cu2+ chelator"),
    # Trimesic acid scaffold (1,3,5-benzenetricarboxylate)
    Backbone("1,3,5-trisubst-benzene", "[1*]c1cc([2*])cc([3*])c1", 3, "tripodal",
             "C3-symmetric, MOF building block"),
    # Triaminobenzene
    Backbone("1,3,5-triaminobenzene", "[1*]Nc1cc(N[2*])cc(N[3*])c1", 3, "tripodal",
             "Tripodal amine platform"),
    # Tris-pyridyl
    Backbone("tris-2-pyridylmethyl", "[1*]NCc1ccccn1.[2*]NCc1ccccn1.[3*]N", 3, "tripodal",
             "TPA-type, borderline metal chelator"),
    # Triethylamine scaffold
    Backbone("nitrilotriacetic", "[1*]OC(=O)CN([2*])CC(=O)O[3*]", 3, "tripodal",
             "NTA-type, hard metal chelator"),
    # Cyclohexane triaxial
    Backbone("1,3,5-trisubst-cyclohexane",
             "[1*][C@@H]1C[C@H]([2*])C[C@@H]([3*])C1", 3, "tripodal",
             "All-axial or all-equatorial, preorganized"),
]


# ---------------------------------------------------------------------------
# Macrocyclic scaffolds (variable ring size, 2-4 sites)
# ---------------------------------------------------------------------------

MACROCYCLIC_BACKBONES: List[Backbone] = [
    # Cyclam (1,4,8,11-tetraazacyclotetradecane) — 4 N donors
    Backbone("cyclam", "[1*]N1CCN([2*])CCCN([3*])CCN([4*])CCC1", 4, "macrocyclic",
             "14-membered, 4 N-donors, Cu2+/Ni2+ selective"),
    # Cyclen (1,4,7,10-tetraazacyclododecane) — 4 N donors, smaller
    Backbone("cyclen", "[1*]N1CCN([2*])CCN([3*])CCN([4*])CC1", 4, "macrocyclic",
             "12-membered, Gd3+ contrast agent scaffold"),
    # Crown ethers (2 sites for pendant arms)
    Backbone("12-crown-4-disubst", "[1*]COCCOCCOCC[2*]", 2, "macrocyclic",
             "Li+ selective, small cavity"),
    Backbone("15-crown-5-disubst", "[1*]COCCOCCOCCOC[2*]", 2, "macrocyclic",
             "Na+ selective"),
    Backbone("18-crown-6-disubst", "[1*]COCCOCCOCCOCCOC[2*]", 2, "macrocyclic",
             "K+ selective, 2.6-3.2 A cavity"),
    # Porphyrin core (simplified, 4 meso positions)
    Backbone("DOTA-core",
             "[1*]OC(=O)CN1CCN([2*])CCN([3*])CCN([4*])CC1",
             4, "macrocyclic", "DOTA-type, Gd3+/Lu-177 chelator"),
]


# ---------------------------------------------------------------------------
# Cage/capsule scaffolds (Davis-type, cryptands)
# ---------------------------------------------------------------------------

CAGE_BACKBONES: List[Backbone] = [
    # Davis-type bis-wall: two anthracene walls + linkers
    # Simplified: disubstituted anthracene pair bridged at 2 positions
    Backbone("bis-anthracene-diurea",
             "[1*]NC(=O)Nc1ccc2cc3ccc(NC(=O)N[2*])cc3cc2c1", 2, "cage",
             "Davis 2012 monocyclic synthetic lectin"),
    # Cryptand [2.2.2] core
    Backbone("cryptand-222",
             "[1*]OCCOCCN([2*])CCOCCOC[3*]", 3, "cage",
             "Cryptand [2.2.2] type, K+ selective"),
    # Bis-macrocycle (two crown-like rings bridged)
    Backbone("bis-crown-bridge",
             "[1*]OCCOCCO[2*]", 2, "cage",
             "Bridged bis-crown, capsule topology"),
    # Calixarene-like (simplified 2-site)
    Backbone("calix4-disubst",
             "[1*]c1cc(Cc2ccccc2)cc(Cc2cc([2*])cc(Cc3ccccc3)c2)c1",
             2, "cage", "Calix[4]arene upper rim"),
]


# ---------------------------------------------------------------------------
# Tetradentate linear (4-site, for octahedral metals)
# ---------------------------------------------------------------------------

TETRADENTATE_BACKBONES: List[Backbone] = [
    # EDTA-type (4 carboxylate arms)
    Backbone("EDTA-core", "[1*]OC(=O)CN([2*])CCN([3*])CC(=O)O[4*]", 4, "linear",
             "EDTA backbone, hard metal chelator"),
    # Salen-type (2+2, Schiff base + phenol)
    Backbone("salen", "[1*]c1ccc(O)c(/C=N/CC/N=C/c2cc([2*])ccc2O)c1", 2, "linear",
             "Salen ligand, Mn/Co/Fe"),
    # Bipyridine-bisamine
    Backbone("bipy-disubst",
             "[1*]c1cccc(-c2cccc([2*])n2)n1", 2, "linear",
             "2,2'-bipyridine, Ru/Fe/Cu"),
    # Terpyridine (3 sites: 2 pendant + 1 on central ring)
    Backbone("terpy-trisubst",
             "[1*]c1cccc(-c2cc([2*])cc(-c3cccc([3*])n3)n2)n1", 3, "linear",
             "2,2':6',2''-terpyridine, Fe2+/Ru2+"),
]


# ---------------------------------------------------------------------------
# Combined library
# ---------------------------------------------------------------------------

ALL_MULTISITE_BACKBONES = (
    TRIPODAL_BACKBONES
    + MACROCYCLIC_BACKBONES
    + CAGE_BACKBONES
    + TETRADENTATE_BACKBONES
)


def get_multisite_backbones(
    n_sites: Optional[int] = None,
    categories: Optional[List[str]] = None,
    max_results: int = 100,
) -> List[Backbone]:
    """Get multisite backbones filtered by site count and category."""
    results = ALL_MULTISITE_BACKBONES
    if n_sites is not None:
        results = [b for b in results if b.n_sites == n_sites]
    if categories:
        results = [b for b in results if b.category in categories]
    return results[:max_results]


def validate_backbones() -> dict:
    """Validate all multisite backbone SMILES. Returns {name: (valid, error)}."""
    if not HAS_RDKIT:
        return {}
    results = {}
    for bb in ALL_MULTISITE_BACKBONES:
        mol = Chem.MolFromSmiles(bb.smiles)
        if mol is None:
            results[bb.name] = (False, "invalid SMILES")
        else:
            # Count dummy atoms
            n_dummy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
            if n_dummy != bb.n_sites:
                results[bb.name] = (False, f"expected {bb.n_sites} dummies, found {n_dummy}")
            else:
                results[bb.name] = (True, "")
    return results


# ---------------------------------------------------------------------------
# Integration with existing pipeline
# ---------------------------------------------------------------------------

def multisite_enumerate(
    n_sites: int = 3,
    categories: Optional[List[str]] = None,
    arms: Optional[list] = None,
    max_candidates: int = 200,
    hsab_filter: bool = True,
    metal: Optional[str] = None,
) -> list:
    """
    Enumerate molecules from multisite backbones using existing pipeline.

    Returns list of (smiles, backbone_name, arm_names, sa_score) tuples,
    same format as de_novo_generator.enumerate_molecules.
    """
    from core.de_novo_generator import enumerate_molecules, ARM_LIBRARY

    backbones = get_multisite_backbones(n_sites=n_sites, categories=categories)
    if arms is None:
        arms = ARM_LIBRARY

    return enumerate_molecules(
        metal=metal,
        backbones=backbones,
        arms=arms,
        max_candidates=max_candidates,
        hsab_filter=hsab_filter,
    )
