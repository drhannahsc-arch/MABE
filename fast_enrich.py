"""
fast_enrich.py -- Protein-ligand optimized guest property computation.

Skips 3D embedding/MMFF (needed only for volume -> packing coefficient).
Volume estimated from MW for protein-ligand entries. ~20x faster.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski


def fast_compute_guest(smiles, need_3d=False):
    """Compute guest properties. Skips 3D if need_3d=False."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {}
    props["guest_mw"] = Descriptors.ExactMolWt(mol)

    if need_3d:
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            props["guest_volume_A3"] = AllChem.ComputeMolVolume(mol_h)
        except Exception:
            props["guest_volume_A3"] = props["guest_mw"] * 1.0
    else:
        # Estimate: ~1 A3 per Da for organic molecules
        props["guest_volume_A3"] = props["guest_mw"] * 1.0

    # Labute ASA -- no 3D needed
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
    LABUTE_SCALE = 4.5
    props["guest_sasa_total_A2"] = round(labute_asa * LABUTE_SCALE, 1)

    n_heavy = mol.GetNumHeavyAtoms()
    n_polar = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("N", "O", "S"))
    polar_frac = n_polar / max(1, n_heavy)
    props["guest_sasa_polar_A2"] = round(props["guest_sasa_total_A2"] * polar_frac, 1)
    props["guest_sasa_nonpolar_A2"] = round(props["guest_sasa_total_A2"] * (1 - polar_frac), 1)

    props["guest_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    props["guest_n_hbond_donors"] = Lipinski.NumHDonors(mol)
    props["guest_n_hbond_acceptors"] = Lipinski.NumHAcceptors(mol)
    props["guest_n_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    props["guest_logP"] = round(Descriptors.MolLogP(mol), 2)
    props["guest_charge"] = Chem.GetFormalCharge(mol)

    return props


def fast_enrich_complex(uc):
    """Fast enrichment -- skips 3D for protein-ligand."""
    if not uc.guest_smiles:
        return uc

    need_3d = uc.binding_mode != "protein_ligand"
    props = fast_compute_guest(uc.guest_smiles, need_3d=need_3d)
    if not props:
        return None

    for key, val in props.items():
        current = getattr(uc, key, None)
        if current is not None and (current == 0 or current == 0.0 or current == ""):
            setattr(uc, key, val)

    if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0:
        uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3

    return uc
