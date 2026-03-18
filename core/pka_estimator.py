"""
core/pka_estimator.py — pH-aware protonation state estimation.

SMARTS-based group-additive pKa lookup + Henderson-Hasselbalch
to estimate net charge and H-bond donor count at a given pH.

Used by unified_scorer_v2 to correct guest_charge and n_hbonds_formed
for host-guest entries where SMILES are stored as free bases.

Source: Perrin (Dissociation Constants of Organic Bases), Lide (CRC),
        Ertl & Schuffenhauer functional group pKa compilations.
"""

import math

_RDKIT = False
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    _RDKIT = True
except ImportError:
    Chem = None

# ═══════════════════════════════════════════════════════════════
# pKa GROUP DATABASE
# (SMARTS for neutral/protonated form, pKa, 'base'|'acid', name)
# ═══════════════════════════════════════════════════════════════

_PKA_GROUPS_RAW = [
    # Bases (SMARTS matches neutral form)
    ("[NX3;H2][CX3](=[NX3;H1])[NX3;H2]", 12.5, "base", "guanidinium"),
    ("[NX3;H2][CX3](=[NX2])[NX3;H2]", 12.5, "base", "guanidinium_v2"),
    ("[CX4][NX3;H2;!$(NC=O);!$(NC=N);!$(Nc)]", 10.5, "base", "primary_alkyl_amine"),
    ("[CX4][NX3;H1;!$(NC=O)]([CX4])", 10.0, "base", "secondary_alkyl_amine"),
    ("[CX4][NX3;H0;!$(NC=O)]([CX4])[CX4]", 9.5, "base", "tertiary_alkyl_amine"),
    ("[NX3;H1;R;!$(NC=O)]", 10.0, "base", "cyclic_sec_amine"),
    ("[NX3;H0;R;!$(NC=O);!$(Nc)]", 8.5, "base", "cyclic_tert_amine"),
    ("[nR1]1[cR1][nR1;H1][cR1][cR1]1", 7.0, "base", "imidazole"),
    ("[nX2;H0;R1]", 5.2, "base", "pyridine"),
    ("[NX3;H2;!$(NC=O)]c", 4.6, "base", "aniline"),
    # Acids (SMARTS matches protonated form)
    ("[CX3](=O)[OX2H1]", 4.5, "acid", "carboxylic_acid"),
    ("[OX2H1]c", 10.0, "acid", "phenol"),
    ("[SX2H1]", 8.3, "acid", "thiol"),
    ("[NX3;H1]S(=O)(=O)", 10.0, "acid", "sulfonamide"),
]

_COMPILED = None

def _compile_patterns():
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED
    if not _RDKIT:
        _COMPILED = []
        return _COMPILED
    _COMPILED = []
    for smarts, pka, gtype, name in _PKA_GROUPS_RAW:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            _COMPILED.append((pat, pka, gtype, name))
    return _COMPILED


def estimate_charge_at_ph(smiles, ph=7.0):
    """Estimate net molecular charge at given pH from SMILES.
    
    Returns dict with charge_int, n_protonated_bases, n_deprotonated_acids,
    n_hbd_from_protonation, groups.
    
    Returns None if RDKit unavailable or SMILES invalid.
    """
    if not _RDKIT or not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    patterns = _compile_patterns()
    charge = 0.0
    n_prot = 0
    n_deprot = 0
    claimed = set()
    groups = []
    
    for pat, pka, gtype, name in patterns:
        matches = mol.GetSubstructMatches(pat)
        if not matches:
            continue
        unique = []
        for match in matches:
            key = match[0]
            if key not in claimed:
                unique.append(match)
                claimed.add(key)
        if not unique:
            continue
        n = len(unique)
        
        if gtype == "base":
            frac = 1.0 / (1.0 + 10**(ph - pka))
            charge += n * frac
            if frac > 0.5:
                n_prot += n
            groups.append({"name": name, "pka": pka, "type": "base",
                          "n": n, "frac_protonated": round(frac, 3)})
        else:
            frac = 1.0 / (1.0 + 10**(pka - ph))
            charge -= n * frac
            if frac > 0.5:
                n_deprot += n
            groups.append({"name": name, "pka": pka, "type": "acid",
                          "n": n, "frac_deprotonated": round(frac, 3)})
    
    return {
        "charge": round(charge, 3),
        "charge_int": round(charge),
        "n_protonated_bases": n_prot,
        "n_deprotonated_acids": n_deprot,
        "n_hbd_from_protonation": n_prot,
        "groups": groups,
    }


def enrich_uc_protonation(uc, ph=7.0):
    """Update a UniversalComplex's charge and H-bond fields based on pKa estimation.
    
    Only fires if:
      - uc has guest_smiles
      - uc.guest_charge == 0 (not already assigned)
      - binding_mode is host_guest_inclusion
      
    Modifies uc in-place. Returns True if enrichment happened.
    """
    if not uc.guest_smiles:
        return False
    if uc.binding_mode != "host_guest_inclusion":
        return False
    
    # If charge already assigned, skip (avoid double-enrichment)
    if uc.guest_charge != 0:
        return False
    
    result = estimate_charge_at_ph(uc.guest_smiles, ph)
    if result is None:
        return False
    
    charge = result["charge_int"]
    if charge == 0:
        return False  # no protonation change needed
    
    uc.guest_charge = charge
    
    # Update H-bond count: protonated amines add donors
    # R-NH3+ has 3 N-H donors, R2-NH2+ has 2, R3-NH+ has 1
    if result["n_protonated_bases"] > 0 and uc.host_name.startswith("CB"):
        cap = {"CB5": 2, "CB6": 2, "CB7": 3, "CB8": 3}.get(uc.host_name, 2)
        # Count total HBD from protonated groups
        hbd_donors = 0
        for g in result["groups"]:
            if g["type"] != "base" or g.get("frac_protonated", 0) <= 0.5:
                continue
            n = g["n"]
            name = g["name"]
            if "primary" in name:
                hbd_donors += n * 3  # R-NH3+
            elif "secondary" in name or "cyclic_sec" in name:
                hbd_donors += n * 2  # R2-NH2+
            elif "guanidinium" in name:
                hbd_donors += n * 4  # (NH2)2C=NH2+
            else:
                hbd_donors += n * 1  # tertiary, pyridinium, etc.
        uc.n_hbonds_formed = min(hbd_donors, cap)
    
    return True
