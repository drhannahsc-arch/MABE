"""
core/guest_compute.py — Sprint 37: Auto-compute guest properties from SMILES

Uses RDKit to compute all guest properties needed for energy terms:
  - Volume (→ packing coefficient, shape term)
  - SASA total/nonpolar/polar (→ hydrophobic term)
  - Rotatable bonds (→ conformational entropy term)
  - HBD/HBA counts (→ H-bond term)
  - Aromatic ring count (→ π-interaction term)
  - LogP (→ hydrophobicity cross-check)
  - MW (→ translational entropy)

Falls back gracefully if RDKit not installed.
"""

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski
    _RDKIT_AVAILABLE = True
except ImportError:
    pass

import math


def rdkit_available():
    return _RDKIT_AVAILABLE


def compute_guest_properties(smiles):
    """Compute all guest properties from SMILES string.

    Returns dict of properties, or empty dict if RDKit unavailable or SMILES invalid.
    """
    if not _RDKIT_AVAILABLE:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # Add hydrogens for accurate property calculation
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, randomSeed=42)

    props = {}

    # Molecular weight
    props["guest_mw"] = Descriptors.ExactMolWt(mol)

    # Volume (Å³) — requires 3D coordinates
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        props["guest_volume_A3"] = AllChem.ComputeMolVolume(mol_h)
    except Exception:
        # Estimate from MW: V ≈ MW * 1.0 (rough, ~1 Å³ per Da for organic)
        props["guest_volume_A3"] = props["guest_mw"] * 1.0

    # SASA — Labute approximation (fast but ~4-5x lower than Freesasa/NACCESS)
    # Scale to match real SASA for back-solve purposes
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
    LABUTE_SCALE = 4.5  # Empirical: Labute ~4.5x lower than van der Waals SASA
    props["guest_sasa_total_A2"] = round(labute_asa * LABUTE_SCALE, 1)

    # Decompose SASA into polar/nonpolar by atom contribution
    # Polar atoms: N, O, S with H attached, or charged
    total_sasa = props["guest_sasa_total_A2"]
    n_heavy = mol.GetNumHeavyAtoms()
    n_polar = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("N", "O", "S"))
    polar_frac = n_polar / max(1, n_heavy)
    props["guest_sasa_polar_A2"] = round(total_sasa * polar_frac, 1)
    props["guest_sasa_nonpolar_A2"] = round(total_sasa * (1 - polar_frac), 1)

    # Rotatable bonds
    props["guest_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # H-bond donors and acceptors (Lipinski)
    props["guest_n_hbond_donors"] = Lipinski.NumHDonors(mol)
    props["guest_n_hbond_acceptors"] = Lipinski.NumHAcceptors(mol)

    # Aromatic rings
    props["guest_n_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)

    # LogP (Wildman-Crippen)
    props["guest_logP"] = round(Descriptors.MolLogP(mol), 2)

    # Formal charge
    props["guest_charge"] = Chem.GetFormalCharge(mol)

    return props


def enrich_complex(uc):
    """Auto-fill guest properties on a UniversalComplex from its SMILES.

    Modifies uc in-place. Only fills fields that are still at default (0).
    """
    if not uc.guest_smiles:
        return uc

    props = compute_guest_properties(uc.guest_smiles)
    if not props:
        return uc

    for key, val in props.items():
        current = getattr(uc, key, None)
        if current is not None and (current == 0 or current == 0.0 or current == ""):
            setattr(uc, key, val)

    # Recompute packing coefficient now that guest volume is known
    if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0:
        uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3

    return uc


def estimate_sasa_burial(guest_sasa_nonpolar, cavity_radius_nm, binding_mode=""):
    """Estimate SASA buried when guest enters a cavity host or protein pocket.

    For cavity hosts (CD, CB): geometric model based on cavity enclosure.
    For proteins: empirical ~60-80% of guest nonpolar SASA buries.
    """
    if guest_sasa_nonpolar <= 0:
        return 0.0

    if binding_mode == "protein_ligand":
        # Protein active sites bury ~65% of ligand nonpolar SASA on average
        # (Ref: Connolly surface calculations, Hubbard & Argos 1994)
        return round(guest_sasa_nonpolar * 0.65, 1)

    if cavity_radius_nm <= 0:
        return 0.0

    # Cavity host: fraction enclosed depends on size match
    r_A = cavity_radius_nm * 10.0  # nm → Å
    cavity_sasa = 4 * math.pi * r_A**2

    # Burial = guest SASA that contacts cavity wall
    # Well-fit guest: ~80% burial. Oversized: less. Undersized: less.
    burial = min(guest_sasa_nonpolar, cavity_sasa * 0.8) * 0.80
    return round(burial, 1)
