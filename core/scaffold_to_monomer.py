"""
core/scaffold_to_monomer.py -- Bridge multisite backbones to assembly engine.

Analyzes backbone SMILES to detect face chemistry, then generates
MonomerSpecs with typed faces for self-assembly prediction.

Chemistry → Face mapping:
  urea [NX3]C(=O)[NX3]      → H-bond network (structural)
  carboxylate C(=O)[OH]      → coordination (structural)
  amine N (in ring or free)  → coordination (structural)
  pyridine c1ccncc1          → coordination (structural)
  imine C=N                  → coordination or covalent (structural)
  phenol [OH]c               → coordination (structural, with metal)
  ether [OX2]([C])[C]        → coordination (weak, crown ether)
  aromatic (large)           → pi-pi or CH-pi (capture or structural)
  thiol [SH]                 → coordination soft (structural)

Entry point:
  backbone_to_monomer(backbone) -> MonomerSpec
  all_monomer_specs() -> list of (MonomerSpec, MaterialDesign)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from core.assembly_engine import (
    Face, FaceRole, InteractionMode, MonomerSpec,
    design_material, MaterialDesign,
)
from core.multisite_assembler import (
    Backbone, ALL_MULTISITE_BACKBONES, get_multisite_backbones,
)


# ---------------------------------------------------------------------------
# SMARTS patterns for face chemistry detection
# ---------------------------------------------------------------------------

_FACE_PATTERNS: List[Tuple[str, str, InteractionMode, FaceRole, int, str]] = [
    # (name, SMARTS, interaction, default_role, contacts_per_match, notes)
    ("urea",        "[NX3]C(=O)[NX3]",     InteractionMode.HBOND_NETWORK,
     FaceRole.STRUCTURAL, 4, "cooperative H-bond array"),
    ("carboxylate",  "C(=O)[OH]",           InteractionMode.COORDINATION,
     FaceRole.STRUCTURAL, 2, "metal-binding carboxylate"),
    ("pyridine_N",   "c1ccncc1",            InteractionMode.COORDINATION,
     FaceRole.STRUCTURAL, 1, "pyridine N-donor"),
    ("imine",        "[CX3]=[NX2]",         InteractionMode.COORDINATION,
     FaceRole.STRUCTURAL, 1, "Schiff base N-donor"),
    ("phenol",       "[OH]c",               InteractionMode.COORDINATION,
     FaceRole.STRUCTURAL, 1, "phenolate O-donor"),
    ("crown_ether",  "[OX2]([CX4])[CX4]",  InteractionMode.COORDINATION,
     FaceRole.STRUCTURAL, 1, "crown ether O-donor"),
    ("thiol",        "[SX2H]",              InteractionMode.COORDINATION,
     FaceRole.STRUCTURAL, 1, "soft S-donor"),
]

# Category → symmetry mapping
_CATEGORY_SYMMETRY: Dict[str, str] = {
    "tripodal": "C3",
    "macrocyclic": "C4",   # default for macrocycles; overridden by site count
    "cage": "C2",
    "linear": "C2",
}

# Category → rigidity
_CATEGORY_RIGIDITY: Dict[str, float] = {
    "tripodal": 0.7,
    "macrocyclic": 0.9,
    "cage": 0.8,
    "linear": 0.5,
}


# ---------------------------------------------------------------------------
# Chemistry analysis
# ---------------------------------------------------------------------------

def _analyze_backbone_chemistry(smiles: str) -> Dict[str, int]:
    """Detect functional groups in a backbone SMILES (dummies replaced with H)."""
    if not HAS_RDKIT:
        return {}

    # Replace all dummy atoms with H
    clean = smiles
    for i in range(1, 5):
        clean = clean.replace(f"[{i}*]", "[H]")
    clean = clean.replace("[*]", "[H]")

    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return {}

    results = {}
    for name, smarts, _, _, _, _ in _FACE_PATTERNS:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            n = len(mol.GetSubstructMatches(pat))
            if n > 0:
                results[name] = n

    # Aromatic atom count (separate from patterns)
    n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    if n_arom > 0:
        results["aromatic_atoms"] = n_arom

    # Ring info
    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    if ring_sizes:
        results["max_ring"] = max(ring_sizes)
        results["n_rings"] = len(ring_sizes)

    # MW
    results["mw"] = round(Descriptors.MolWt(mol), 1)
    results["n_heavy"] = mol.GetNumHeavyAtoms()

    return results


# ---------------------------------------------------------------------------
# Face generation from chemistry
# ---------------------------------------------------------------------------

def _generate_faces(
    backbone: Backbone,
    chemistry: Dict[str, int],
) -> List[Face]:
    """Generate typed faces from detected chemistry + backbone metadata."""
    faces = []
    face_idx = 0

    # Priority: strongest interaction type detected becomes structural faces
    # One face per dummy site on the backbone

    # Detect primary interaction mode from chemistry
    primary_mode = None
    primary_contacts = 1

    for name, smarts, interaction, role, contacts, notes in _FACE_PATTERNS:
        if name in chemistry and chemistry[name] > 0:
            if primary_mode is None:
                primary_mode = interaction
                primary_contacts = contacts
            # Generate faces for each match
            n_matches = chemistry[name]
            for i in range(min(n_matches, backbone.n_sites)):
                # Coordination faces connect to external metal nodes
                comp = "metal_node" if interaction == InteractionMode.COORDINATION else ""
                faces.append(Face(
                    name=f"{name}_{i+1}",
                    role=role,
                    interaction=interaction,
                    n_contacts=contacts,
                    area_A2=_estimate_face_area(interaction),
                    complementary_to=comp,
                    notes=notes,
                ))
                face_idx += 1

    # If we haven't generated enough faces for all sites, fill with
    # the primary mode or coordination (default for unfilled sites)
    while len(faces) < backbone.n_sites:
        mode = primary_mode or InteractionMode.COORDINATION
        comp = "metal_node" if mode == InteractionMode.COORDINATION else ""
        faces.append(Face(
            name=f"site_{len(faces)+1}",
            role=FaceRole.STRUCTURAL,
            interaction=mode,
            n_contacts=primary_contacts,
            area_A2=_estimate_face_area(mode),
            complementary_to=comp,
            notes="auto-filled",
        ))

    # Trim to site count (structural faces)
    structural_faces = faces[:backbone.n_sites]

    # Add capture face if aromatic surface is large enough
    n_arom = chemistry.get("aromatic_atoms", 0)
    if n_arom >= 10:
        structural_faces.append(Face(
            name="aromatic_capture",
            role=FaceRole.CAPTURE,
            interaction=InteractionMode.CH_PI,
            n_contacts=min(3, n_arom // 5),
            area_A2=n_arom * 8.0,
            notes=f"{n_arom} aromatic atoms, CH-pi surface",
        ))
    elif n_arom >= 6:
        structural_faces.append(Face(
            name="aromatic_face",
            role=FaceRole.CAPTURE,
            interaction=InteractionMode.PI_PI,
            n_contacts=1,
            area_A2=n_arom * 8.0,
            notes=f"{n_arom} aromatic atoms",
        ))

    return structural_faces


def _estimate_face_area(mode: InteractionMode) -> float:
    """Rough face area estimate by interaction type."""
    return {
        InteractionMode.PI_PI: 80.0,
        InteractionMode.CH_PI: 80.0,
        InteractionMode.HBOND_NETWORK: 40.0,
        InteractionMode.COORDINATION: 20.0,
        InteractionMode.HYDROPHOBIC: 60.0,
        InteractionMode.COVALENT: 15.0,
        InteractionMode.VAN_DER_WAALS: 50.0,
    }.get(mode, 30.0)


# ---------------------------------------------------------------------------
# Symmetry inference
# ---------------------------------------------------------------------------

def _infer_symmetry(backbone: Backbone) -> str:
    """Infer point group from backbone category and site count."""
    cat = backbone.category
    n = backbone.n_sites

    # Special cases
    if backbone.name in ("cyclam", "cyclen", "DOTA-core"):
        return "C4" if n == 4 else "C2"
    if backbone.name in ("EDTA-core",):
        return "C2"  # pseudo-C2

    if n == 2:
        return "C2"
    elif n == 3:
        if cat == "tripodal":
            return "C3"
        return "C3"
    elif n == 4:
        if cat == "macrocyclic":
            return "D4h" if "cyclam" in backbone.name else "C4"
        return "Td"
    elif n == 6:
        return "Oh"

    return _CATEGORY_SYMMETRY.get(cat, "C1")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def backbone_to_monomer(backbone: Backbone) -> MonomerSpec:
    """
    Convert a multisite Backbone to a face-typed MonomerSpec.

    Analyzes SMILES chemistry to detect interaction types,
    assigns structural/capture roles, infers symmetry.
    """
    chemistry = _analyze_backbone_chemistry(backbone.smiles)
    faces = _generate_faces(backbone, chemistry)
    symmetry = _infer_symmetry(backbone)
    rigidity = _CATEGORY_RIGIDITY.get(backbone.category, 0.5)

    mw = chemistry.get("mw", 200.0)
    vol = mw * 0.85  # rough: ~0.85 A3/Da for organic

    return MonomerSpec(
        name=backbone.name,
        smiles=backbone.smiles,
        faces=faces,
        molecular_weight=mw,
        monomer_volume_A3=round(vol, 0),
        symmetry=symmetry,
        rigidity=rigidity,
    )


def all_monomer_specs() -> List[Tuple[MonomerSpec, MaterialDesign]]:
    """
    Convert all multisite backbones to MonomerSpecs and predict materials.

    Returns list of (monomer, material_design) tuples.
    """
    results = []
    for bb in ALL_MULTISITE_BACKBONES:
        mono = backbone_to_monomer(bb)
        design = design_material(mono)
        results.append((mono, design))
    return results


def print_material_catalog():
    """Print a formatted catalog of all backbone → material predictions."""
    results = all_monomer_specs()
    print(f"{'Backbone':<30} {'Sites':<6} {'Cat':<12} {'Sym':<5} "
          f"{'Topology':<15} {'Dim':<4} {'Porosity':<9} "
          f"{'BET':>6} {'dG/mono':>8} {'Stab':<6}")
    print("-" * 115)
    for mono, design in results:
        t = design.topology
        p = design.properties
        print(f"{mono.name:<30} {mono.valence:<6} "
              f"{[f.interaction.value for f in mono.structural_faces()]!s:<12} "
              f"{mono.symmetry:<5} {t.topology.value:<15} {t.dimensionality:<4} "
              f"{p.porosity_fraction:<9.1%} {p.bet_surface_area_m2g:>6.0f} "
              f"{p.dG_per_monomer:>8.1f} {p.thermal_stability:<6}")
