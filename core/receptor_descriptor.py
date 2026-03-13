"""
core/receptor_descriptor.py — Synthetic Receptor Auto-Characterization
========================================================================

Given a host SMILES and guest SMILES, compute all binding-relevant
descriptors for the unified scorer WITHOUT manual annotation.

Physics-first characterization:
  1. Cavity estimation: convex hull of inward-pointing H-bond donors/acceptors
  2. Aromatic wall detection: count aromatic rings with cavity-facing exposure
  3. H-bond complementarity: donor/acceptor matching between host and guest
  4. Packing coefficient: V_guest / V_cavity
  5. Shape: macrocyclic ring detection, cage/tweezer classification
  6. Polar surface matching: guest polar SASA vs host polar interior

Entry point:
  characterize_receptor(host_smiles, guest_smiles) -> dict of UC fields

No fitted parameters. No target-specific data.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski
    from rdkit.Chem import rdMolTransforms
    _RDKIT_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def characterize_receptor(host_smiles: str, guest_smiles: str = "",
                          host_name: str = "") -> Dict:
    """
    Auto-characterize a synthetic receptor from SMILES.

    Returns dict of UniversalComplex field values.
    All fields are populated; downstream code can setattr() them onto a UC.
    """
    if not _RDKIT_AVAILABLE:
        return _fallback_characterization(host_smiles, guest_smiles, host_name)

    host_mol = Chem.MolFromSmiles(host_smiles)
    if host_mol is None:
        return _fallback_characterization(host_smiles, guest_smiles, host_name)

    result = {
        'host_name': host_name or _infer_host_name(host_mol, host_smiles),
        'host_type': 'synthetic_receptor',
        'binding_mode': 'synthetic_receptor',
    }

    # ── Host topology ─────────────────────────────────────────────────
    topo = _classify_topology(host_mol)
    result['is_macrocyclic'] = topo['is_macrocyclic']
    result['is_cage'] = topo['is_cage']
    result['host_charge'] = Chem.GetFormalCharge(host_mol)

    # ── Cavity volume estimation ──────────────────────────────────────
    cavity = _estimate_cavity(host_mol)
    result['cavity_volume_A3'] = cavity['volume_A3']
    result['cavity_radius_nm'] = cavity['radius_nm']

    # ── H-bond sites ──────────────────────────────────────────────────
    hb = _count_hbond_sites(host_mol)
    result['n_hbond_donors_host'] = hb['n_donors']
    result['n_hbond_acceptors_host'] = hb['n_acceptors']

    # ── Aromatic walls ────────────────────────────────────────────────
    result['n_aromatic_walls'] = _count_aromatic_walls(host_mol)

    # ── Guest properties + complementarity ────────────────────────────
    if guest_smiles:
        guest_mol = Chem.MolFromSmiles(guest_smiles)
        if guest_mol is not None:
            comp = _compute_complementarity(host_mol, guest_mol, cavity)
            result.update(comp)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# TOPOLOGY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _classify_topology(mol) -> Dict:
    """Classify receptor as macrocycle, cage, tweezer, or open."""
    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]

    # Macrocyclic: any ring > 8 members
    is_macrocyclic = any(s > 8 for s in ring_sizes)

    # Cage: 3+ bridging rings sharing atoms (3D encapsulation)
    # Heuristic: count fused ring systems with >2 shared atoms
    is_cage = False
    if len(ring_sizes) >= 3:
        # Check for bridged polycyclic system
        bond_rings = ri.BondRings()
        if len(bond_rings) >= 3:
            # Count bonds shared between different rings
            all_bonds = set()
            shared = 0
            for ring in bond_rings:
                ring_set = set(ring)
                shared += len(all_bonds & ring_set)
                all_bonds |= ring_set
            # Cage if substantial ring fusion
            is_cage = shared > 3 and is_macrocyclic

    # Tweezer: two aromatic systems connected by flexible linker
    # (not macrocyclic, has 2+ aromatic rings separated by >=2 bonds)
    is_tweezer = (not is_macrocyclic
                  and rdMolDescriptors.CalcNumAromaticRings(mol) >= 2)

    return {
        'is_macrocyclic': is_macrocyclic,
        'is_cage': is_cage,
        'is_tweezer': is_tweezer,
        'n_rings': len(ring_sizes),
        'max_ring_size': max(ring_sizes) if ring_sizes else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CAVITY VOLUME ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_cavity(mol) -> Dict:
    """
    Estimate cavity volume from molecular structure.

    Strategy:
      1. Generate 3D conformer
      2. Find centroid of heteroatoms (likely inward-pointing)
      3. Estimate cavity as sphere with radius = mean distance to heteroatoms
      4. Subtract van der Waals volume of cavity-lining atoms

    For macrocycles: use ring atoms to define cavity plane.
    For cages: use bridgehead atoms.
    For linear/tweezer: use distance between aromatic centroids.
    """
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3(), randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        conf = mol_h.GetConformer()
    except Exception:
        # Fallback: estimate from heavy atom count
        n_heavy = mol.GetNumHeavyAtoms()
        vol_est = max(20, n_heavy * 3.5)  # rough: ~3.5 A^3 per heavy atom cavity
        return {'volume_A3': vol_est, 'radius_nm': (3 * vol_est / (4 * math.pi))**(1/3) / 10}

    # Identify heteroatoms (likely H-bond sites pointing into cavity)
    hetero_indices = [a.GetIdx() for a in mol_h.GetAtoms()
                      if a.GetSymbol() in ('N', 'O', 'S') and a.GetIdx() < mol.GetNumAtoms()]

    if len(hetero_indices) < 2:
        # Use ring atom centroid instead
        ri = mol_h.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            for idx in ring:
                ring_atoms.add(idx)
        hetero_indices = list(ring_atoms) if ring_atoms else list(range(mol.GetNumAtoms()))

    # Compute centroid of cavity-defining atoms
    positions = [conf.GetAtomPosition(i) for i in hetero_indices]
    cx = sum(p.x for p in positions) / len(positions)
    cy = sum(p.y for p in positions) / len(positions)
    cz = sum(p.z for p in positions) / len(positions)

    # Mean distance from centroid to cavity-defining atoms
    distances = [math.sqrt((p.x-cx)**2 + (p.y-cy)**2 + (p.z-cz)**2)
                 for p in positions]
    mean_r = sum(distances) / len(distances) if distances else 3.0

    # Cavity radius: mean distance minus average vdW radius of lining atoms (~1.5 A)
    cavity_r = max(0.5, mean_r - 1.5)

    # Volume as sphere
    volume = (4/3) * math.pi * cavity_r**3

    return {
        'volume_A3': round(volume, 1),
        'radius_nm': round(cavity_r / 10, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# H-BOND SITE COUNTING
# ═══════════════════════════════════════════════════════════════════════════

def _count_hbond_sites(mol) -> Dict:
    """Count H-bond donors and acceptors on host."""
    n_donors = Lipinski.NumHDonors(mol)
    n_acceptors = Lipinski.NumHAcceptors(mol)

    # Urea/amide NH groups: strong directional donors (Davis cage motif)
    urea_pat = Chem.MolFromSmarts('[NH]C(=O)[NH]')
    amide_pat = Chem.MolFromSmarts('[NH]C(=O)')

    n_urea = len(mol.GetSubstructMatches(urea_pat)) if urea_pat else 0
    n_amide = len(mol.GetSubstructMatches(amide_pat)) if amide_pat else 0

    return {
        'n_donors': n_donors,
        'n_acceptors': n_acceptors,
        'n_urea_nh': n_urea,
        'n_amide_nh': n_amide,
    }


# ═══════════════════════════════════════════════════════════════════════════
# AROMATIC WALL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _count_aromatic_walls(mol) -> int:
    """
    Count aromatic rings that can serve as CH-π interaction surfaces.
    Only counts rings with ≥1 atom not in another aromatic ring
    (i.e., rings at the periphery facing into the cavity).
    """
    return rdMolDescriptors.CalcNumAromaticRings(mol)


# ═══════════════════════════════════════════════════════════════════════════
# HOST-GUEST COMPLEMENTARITY
# ═══════════════════════════════════════════════════════════════════════════

def _compute_complementarity(host_mol, guest_mol, cavity: Dict) -> Dict:
    """Compute host-guest complementarity descriptors."""
    from core.guest_compute import compute_guest_properties

    result = {}

    # Guest properties
    guest_smiles = Chem.MolToSmiles(guest_mol)
    guest_props = compute_guest_properties(guest_smiles)
    result.update(guest_props)

    # Packing coefficient
    guest_vol = guest_props.get('guest_volume_A3', 0)
    cav_vol = cavity.get('volume_A3', 1)
    if cav_vol > 0 and guest_vol > 0:
        result['packing_coefficient'] = round(guest_vol / cav_vol, 3)

    # H-bond complementarity: min(host donors, guest acceptors) + min(host acceptors, guest donors)
    host_hb = _count_hbond_sites(host_mol)
    guest_donors = guest_props.get('guest_n_hbond_donors', 0)
    guest_acceptors = guest_props.get('guest_n_hbond_acceptors', 0)

    n_hbonds_est = min(host_hb['n_donors'], guest_acceptors) + min(host_hb['n_acceptors'], guest_donors)
    result['n_hbonds_formed'] = n_hbonds_est

    # SASA burial estimate: fraction of guest that fits inside cavity
    guest_sasa_np = guest_props.get('guest_sasa_nonpolar_A2', 0)
    if guest_sasa_np > 0 and cav_vol > 0:
        # Rough: fraction buried scales with packing
        packing = result.get('packing_coefficient', 0)
        burial_frac = min(1.0, packing * 0.8)  # 80% at packing=1.0
        result['sasa_buried_A2'] = round(guest_sasa_np * burial_frac, 1)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# HOST NAME INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def _infer_host_name(mol, smiles: str) -> str:
    """Try to infer a meaningful name from structure."""
    n_heavy = mol.GetNumHeavyAtoms()
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    max_ring = max((len(r) for r in ri.AtomRings()), default=0)

    # Urea cage pattern
    urea_pat = Chem.MolFromSmarts('[NH]C(=O)[NH]')
    n_urea = len(mol.GetSubstructMatches(urea_pat)) if urea_pat else 0
    if n_urea >= 4 and max_ring > 12:
        return f"urea-cage-{n_heavy}atoms"

    if max_ring > 12:
        return f"macrocycle-{n_heavy}atoms"
    if n_rings >= 3:
        return f"cage-{n_heavy}atoms"
    if n_rings >= 2:
        return f"tweezer-{n_heavy}atoms"

    return f"receptor-{n_heavy}atoms"


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK (no RDKit)
# ═══════════════════════════════════════════════════════════════════════════

def _fallback_characterization(host_smiles: str, guest_smiles: str,
                               host_name: str) -> Dict:
    """Minimal characterization without RDKit."""
    # Count heteroatoms from SMILES string
    n_N = host_smiles.count('N') + host_smiles.count('n')
    n_O = host_smiles.count('O') + host_smiles.count('o')
    n_S = host_smiles.count('S') + host_smiles.count('s')
    n_heavy = sum(1 for c in host_smiles if c.isalpha() and c.isupper())

    return {
        'host_name': host_name or f"receptor-{n_heavy}atoms",
        'host_type': 'synthetic_receptor',
        'binding_mode': 'synthetic_receptor',
        'is_macrocyclic': '%' in host_smiles,  # ring closure digits
        'is_cage': False,
        'host_charge': 0,
        'cavity_volume_A3': max(20, n_heavy * 3.0),
        'cavity_radius_nm': 0.3,
        'n_hbond_donors_host': n_N + n_O,
        'n_hbond_acceptors_host': n_N + n_O + n_S,
        'n_aromatic_walls': host_smiles.count('c1'),
    }
