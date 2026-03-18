"""
core/inclusion_classifier.py — 3D geometry-based inclusion depth classifier.

Determines how deeply a guest is enclosed by a CB[n] host cavity
using 3D molecular dimensions (SVD) vs host portal/cavity geometry.

Returns inclusion_depth ∈ [0, 1]:
  0 = guest sits entirely outside (portal-associated)
  1 = guest fully enclosed within cavity
  
Used to gate the high-energy water (HEW) term:
  HEW_effective = HEW_full × inclusion_depth

Physics:
  A guest threading through a CB cavity has exposed ends that DON'T
  displace cavity water. Only the enclosed portion contributes to
  frustrated water release. inclusion_depth = min(1, cavity_height / guest_length).
"""

import math

# Host geometry database
# Portal diameters from Barrow, Scherman et al.
# Cavity heights from Lagona 2005, Assaf & Nau 2015
# Cavity equatorial diameters from Kim 2003, Lagona 2005
_CB_GEOMETRY = {
    #          portal_d  cavity_h  cavity_d
    "CB5":    (2.4,      5.5,      4.4),
    "CB6":    (3.9,      7.3,      5.8),
    "CB7":    (5.4,      9.1,      7.3),
    "CB8":    (6.9,      9.1,      8.8),
}

# Cache: smiles → (min_d, mid_d, max_d)
_DIM_CACHE = {}


def _get_guest_dims(smiles):
    """Get 3D molecular dimensions via SVD. Returns (min_d, mid_d, max_d) in Å."""
    if smiles in _DIM_CACHE:
        return _DIM_CACHE[smiles]
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np
        
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol is None:
            return None
        
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer(0)
        
        coords = []
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
        
        if len(coords) < 3:
            return None
        
        coords = np.array(coords)
        coords -= coords.mean(axis=0)
        _, _, Vt = np.linalg.svd(coords, full_matrices=False)
        extents = []
        for i in range(min(3, Vt.shape[0])):
            proj = coords @ Vt[i]
            extents.append(proj.max() - proj.min() + 3.4)  # +2×VdW radius
        extents.sort()
        
        result = tuple(extents)
        _DIM_CACHE[smiles] = result
        return result
    except Exception:
        return None


def classify_inclusion(guest_smiles, host_key, packing_coefficient=0.0):
    """Classify guest inclusion geometry.
    
    Args:
        guest_smiles: canonical SMILES of guest
        host_key: CB5/CB6/CB7/CB8
        packing_coefficient: V_guest / V_cavity
    
    Returns dict:
        inclusion_depth: float [0, 1] — fraction of guest enclosed
        fits_portal: bool — can guest pass through portal
        fully_enclosed: bool — depth >= 0.9
        effective_pc: float — PC × depth (actual water displacement fraction)
        min_cross_section: float — min_d of guest (Å)
        guest_length: float — max_d of guest (Å)
    
    Returns None if dimensions cannot be computed.
    """
    geom = _CB_GEOMETRY.get(host_key)
    if geom is None:
        return None
    
    portal_d, cavity_h, cavity_d = geom
    
    dims = _get_guest_dims(guest_smiles)
    if dims is None:
        return None
    
    min_d, mid_d, max_d = dims
    
    # Portal fit: can guest's minimum cross-section pass through?
    sphericity = min_d / mid_d if mid_d > 0.1 else 1.0
    flex_portal = portal_d + 1.0 * sphericity  # 1 Å flex for spherical guests
    fits_portal = min_d <= flex_portal + 0.5  # 0.5 Å margin for thermal fluctuation
    
    # Inclusion depth: combines axial (length) and radial (width) enclosure
    
    # Axial: fraction of guest length enclosed by cavity barrel
    if max_d > 0:
        axial_depth = min(1.0, cavity_h / max_d)
    else:
        axial_depth = 0.0
    
    # Radial: fraction of cavity cross-section filled by guest
    # A narrow guest in a wide cavity doesn't displace equatorial water
    if cavity_d > 0:
        radial_fill = min(1.0, mid_d / cavity_d)
    else:
        radial_fill = 0.0
    
    # Combined depth: geometric mean of axial and radial
    depth = (axial_depth * radial_fill) ** 0.5
    
    # If guest can't fit through portal at all, depth is limited
    if not fits_portal:
        depth = min(depth, 0.3)
    
    # Effective PC: how much water is actually displaced
    effective_pc = packing_coefficient * depth
    
    return {
        "inclusion_depth": round(depth, 3),
        "fits_portal": bool(fits_portal),
        "fully_enclosed": depth >= 0.9,
        "effective_pc": round(effective_pc, 3),
        "min_cross_section": round(min_d, 1),
        "guest_length": round(max_d, 1),
    }
