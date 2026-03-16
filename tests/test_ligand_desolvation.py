"""
knowledge/ligand_desolvation.py — Physics-based ligand desolvation scoring

Predicts ΔG_hydr (hydration free energy) for small molecules from
atom-type-decomposed solvent-accessible surface area (SASA) plus
hydrogen bond donor/acceptor counts.

Equation:
    ΔG_hydr = Σ_i (γ_i × SASA_i) + δ_HBD × n_HBD + δ_HBA × n_HBA + intercept

where γ_i is the per-atom-type solvation coefficient (kJ/mol/Å²),
SASA_i is the accessible surface area of atom type i, and δ_HBD/δ_HBA
are per-count H-bond donor/acceptor stabilization terms.

Calibration: Ridge regression on FreeSolv v0.52 (642 molecules,
Mobley group, DOI: 10.1007/s10822-014-9747-x).
5-fold CV: R² = 0.80 ± 0.06, MAE = 4.9 ± 0.4 kJ/mol.

Known limitations:
  - Sulfoxides/sulfones systematically underpredicted (S=O dipole
    strength exceeds surface-area model capacity). 5 compounds affected.
  - No explicit treatment of intramolecular H-bonds.
  - No ionic/charged species (FreeSolv is neutral molecules only).
  - Conformer-dependent: single 3D conformer from MMFF94.

For MABE binding predictions:
  ΔG_desolv(binding) ≈ -f_burial × ΔG_hydr
  where f_burial is the fraction of ligand surface buried upon binding.
  This captures the desolvation COST: burying hydrophilic groups is
  unfavorable, burying hydrophobic groups is favorable.
"""

import math

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS — FreeSolv v0.52, Ridge (α=1.0), 642 molecules
# Source: back-solved from experimental ΔG_hydr, DOI: 10.1007/s10822-014-9747-x
# All γ in kJ/mol/Å². δ in kJ/mol per count.
# ═══════════════════════════════════════════════════════════════════════════

SOLVATION_PARAMS = {
    # SASA-type coefficients (kJ/mol per Å²)
    "C_ali":      +0.0662,   # aliphatic carbon: hydrophobic (+) → desolvation favorable
    "C_aro":      -0.0749,   # aromatic carbon: π-water interaction makes it mildly hydrophilic
    "N_amine":    -0.1621,   # amine nitrogen
    "N_amide":    -1.1229,   # amide nitrogen (strong H-bond donor)
    "N_aro":      -0.4025,   # aromatic/heterocyclic nitrogen
    "O_hydroxyl": -0.3025,   # hydroxyl oxygen
    "O_carbonyl": -0.1992,   # carbonyl oxygen (C=O)
    "O_ether":    +0.0591,   # ether oxygen (weakly polar, near-neutral)
    "S_thio":     -0.0517,   # thioether sulfur
    "S_oxide":    -0.4071,   # sulfoxide/sulfone S (strong acceptor, still underpredicted)
    "F":          +0.0554,   # fluorine (surprisingly hydrophobic)
    "Cl":         -0.0173,   # chlorine (near-neutral)
    "Br":         -0.0341,   # bromine
    "I":          -0.0398,   # iodine

    # Count-based terms (kJ/mol per count)
    "delta_HBD":  -5.5767,   # per H-bond donor (N-H, O-H)
    "delta_HBA":  -4.7205,   # per H-bond acceptor (N, O, S=O)

    # Intercept
    "intercept":  -1.154,    # kJ/mol
}

# Atom types used in SASA decomposition
SASA_TYPES = [
    "C_ali", "C_aro", "N_amine", "N_amide", "N_aro",
    "O_hydroxyl", "O_carbonyl", "O_ether",
    "S_thio", "S_oxide", "F", "Cl", "Br", "I",
]

# SASA calculation parameters
SASA_PROBE_RADIUS = 1.4  # Å, water probe
SASA_N_POINTS = 252      # Fibonacci sphere points

# VdW radii (Bondi, Å)
VDW_RADII = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}

# Pre-computed Fibonacci sphere
_golden = (1 + math.sqrt(5)) / 2
_SPHERE_PTS = []
for _i in range(SASA_N_POINTS):
    _theta = math.acos(1 - 2 * (_i + 0.5) / SASA_N_POINTS)
    _phi = 2 * math.pi * _i / _golden
    _SPHERE_PTS.append((
        math.sin(_theta) * math.cos(_phi),
        math.sin(_theta) * math.sin(_phi),
        math.cos(_theta),
    ))


# ═══════════════════════════════════════════════════════════════════════════
# ATOM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _classify_atom_ob(atom, obmol):
    """Classify an OpenBabel OBAtom into solvation-relevant type.

    Returns None for hydrogen (accounted for by heavy-atom SASA).
    """
    from openbabel import openbabel

    an = atom.GetAtomicNum()
    if an == 1:
        return None

    if an == 6:
        return "C_aro" if atom.IsAromatic() else "C_ali"

    if an == 7:
        if atom.IsAromatic():
            return "N_aro"
        # Check amide: N bonded to C(=O)
        for bond in openbabel.OBAtomBondIter(atom):
            other = (bond.GetBeginAtom() if bond.GetEndAtom().GetIdx() == atom.GetIdx()
                     else bond.GetEndAtom())
            if other.GetAtomicNum() == 6:
                for bond2 in openbabel.OBAtomBondIter(other):
                    o2 = (bond2.GetBeginAtom() if bond2.GetEndAtom().GetIdx() == other.GetIdx()
                          else bond2.GetEndAtom())
                    if o2.GetAtomicNum() == 8 and bond2.GetBondOrder() == 2:
                        return "N_amide"
        return "N_amine"

    if an == 8:
        # Carbonyl: O=C
        for bond in openbabel.OBAtomBondIter(atom):
            if bond.GetBondOrder() == 2:
                other = (bond.GetBeginAtom() if bond.GetEndAtom().GetIdx() == atom.GetIdx()
                         else bond.GetEndAtom())
                if other.GetAtomicNum() == 6:
                    return "O_carbonyl"
        # Hydroxyl vs ether
        n_h = sum(
            1 for bond in openbabel.OBAtomBondIter(atom)
            if (bond.GetBeginAtom() if bond.GetEndAtom().GetIdx() == atom.GetIdx()
                else bond.GetEndAtom()).GetAtomicNum() == 1
        )
        return "O_hydroxyl" if n_h > 0 else "O_ether"

    if an == 16:
        from openbabel import openbabel as ob
        # Sulfoxide/sulfone: S bonded to O via double bond or charge
        n_so = sum(
            1 for bond in ob.OBAtomBondIter(atom)
            if bond.GetBondOrder() >= 2
            and (bond.GetBeginAtom() if bond.GetEndAtom().GetIdx() == atom.GetIdx()
                 else bond.GetEndAtom()).GetAtomicNum() == 8
        )
        if n_so == 0:
            for bond in ob.OBAtomBondIter(atom):
                other = (bond.GetBeginAtom() if bond.GetEndAtom().GetIdx() == atom.GetIdx()
                         else bond.GetEndAtom())
                if other.GetAtomicNum() == 8 and other.GetFormalCharge() == -1:
                    n_so += 1
        return "S_oxide" if n_so > 0 else "S_thio"

    return {9: "F", 17: "Cl", 35: "Br", 53: "I", 15: "P"}.get(an, None)


# ═══════════════════════════════════════════════════════════════════════════
# SASA DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

def compute_atom_type_sasa(smiles):
    """Compute per-atom-type SASA and H-bond counts for a SMILES string.

    Returns dict with keys from SASA_TYPES + "n_HBD", "n_HBA",
    or None if 3D embedding fails.

    Requires openbabel (pybel).
    """
    from openbabel import openbabel, pybel

    try:
        mol = pybel.readstring("smi", smiles)
        mol.addh()
        mol.make3D(forcefield='mmff94', steps=150)
    except Exception:
        return None

    obmol = mol.OBMol
    n_atoms = obmol.NumAtoms()
    coords = [
        (obmol.GetAtom(i+1).GetX(), obmol.GetAtom(i+1).GetY(), obmol.GetAtom(i+1).GetZ())
        for i in range(n_atoms)
    ]
    atomic_nums = [obmol.GetAtom(i+1).GetAtomicNum() for i in range(n_atoms)]
    radii = [VDW_RADII.get(an, 1.70) + SASA_PROBE_RADIUS for an in atomic_nums]

    # SASA per atom type
    type_sasa = {}
    for i in range(n_atoms):
        atype = _classify_atom_ob(obmol.GetAtom(i+1), obmol)
        if atype is None:
            continue

        r_i = radii[i]
        xi, yi, zi = coords[i]
        exposed = 0

        for px, py, pz in _SPHERE_PTS:
            sx, sy, sz = xi + r_i * px, yi + r_i * py, zi + r_i * pz
            buried = False
            for j in range(n_atoms):
                if j == i:
                    continue
                dx = sx - coords[j][0]
                dy = sy - coords[j][1]
                dz = sz - coords[j][2]
                if dx*dx + dy*dy + dz*dz < radii[j]**2:
                    buried = True
                    break
            if not buried:
                exposed += 1

        atom_sasa = 4 * math.pi * r_i**2 * exposed / SASA_N_POINTS
        type_sasa[atype] = type_sasa.get(atype, 0.0) + atom_sasa

    # H-bond donor/acceptor counts
    n_hbd = 0
    n_hba = 0
    for i in range(n_atoms):
        atom = obmol.GetAtom(i+1)
        an = atom.GetAtomicNum()
        if an in (7, 8):
            n_hba += 1
            for bond in openbabel.OBAtomBondIter(atom):
                other = (bond.GetBeginAtom() if bond.GetEndAtom().GetIdx() == atom.GetIdx()
                         else bond.GetEndAtom())
                if other.GetAtomicNum() == 1:
                    n_hbd += 1
        elif an == 16:
            # S=O is an H-bond acceptor
            for bond in openbabel.OBAtomBondIter(atom):
                other = (bond.GetBeginAtom() if bond.GetEndAtom().GetIdx() == atom.GetIdx()
                         else bond.GetEndAtom())
                if other.GetAtomicNum() == 8:
                    n_hba += 1

    type_sasa["n_HBD"] = n_hbd
    type_sasa["n_HBA"] = n_hba

    return type_sasa


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_dG_hydration(smiles):
    """Predict ΔG_hydr (kJ/mol) for a SMILES string.

    Returns float or None if 3D embedding fails.
    Negative = hydrophilic (favorable solvation).
    Positive = hydrophobic (unfavorable solvation).
    """
    features = compute_atom_type_sasa(smiles)
    if features is None:
        return None

    return score_from_features(features)


def score_from_features(features):
    """Score ΔG_hydr from pre-computed features dict.

    features: dict with SASA_TYPES keys + "n_HBD", "n_HBA"
    Returns: ΔG_hydr in kJ/mol
    """
    p = SOLVATION_PARAMS
    dG = p["intercept"]

    for atype in SASA_TYPES:
        sasa = features.get(atype, 0.0)
        # Fold P into O_ether (both lone-pair donors, rare)
        if atype == "O_ether" and "P" in features:
            sasa += features["P"]
        dG += p[atype] * sasa

    dG += p["delta_HBD"] * features.get("n_HBD", 0)
    dG += p["delta_HBA"] * features.get("n_HBA", 0)

    return dG


def desolvation_cost(smiles, f_burial=1.0):
    """Compute the desolvation COST of burying a ligand.

    In binding: removing a molecule from water costs -ΔG_hydr.
    Hydrophilic molecules (ΔG_hydr << 0) have HIGH desolvation cost.
    Hydrophobic molecules (ΔG_hydr > 0) have NEGATIVE cost (favorable).

    Args:
        smiles: SMILES string
        f_burial: fraction of ligand surface buried (0-1)

    Returns:
        ΔG_desolv in kJ/mol (positive = unfavorable desolvation cost)
    """
    dG_hydr = predict_dG_hydration(smiles)
    if dG_hydr is None:
        return None

    # Desolvation cost = -f_burial × ΔG_hydr
    # If ΔG_hydr is negative (hydrophilic), desolvation cost is positive (unfavorable)
    return -f_burial * dG_hydr