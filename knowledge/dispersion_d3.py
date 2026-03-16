"""
knowledge/dispersion_d3.py — Element-specific dispersion correction

The hydrophobic SASA term (Eisenberg-McLachlan gamma) already captures
AVERAGE London dispersion for carbon-dominated surfaces. This module
computes the CORRECTION for element-specific C6 deviations:

  - Cl, Br, I, S: larger C6 than C → stronger dispersion → favorable bonus
  - F, O: smaller C6 than C → weaker dispersion → slight penalty
  - N: slightly below C → near-neutral

C6 coefficients from Grimme D3 (J Chem Phys 132:154104, 2010).
Base dispersion gamma calibrated to Grimme JCTC 7:291 (2011) benchmarks:
typical protein-ligand complex has -2 to -4 kJ/mol total dispersion
over ~200-300 Å² buried surface.

This is a CORRECTION term, not the total dispersion. The total is:
  dG_disp = dG_hydrophobic(SASA) + dG_correction(element-specific)
"""

import math


# ═══════════════════════════════════════════════════════════════════════════
# GRIMME D3 C6 COEFFICIENTS (Hartree·Bohr^6, CN-averaged)
# Source: Grimme et al. 2010, J Chem Phys 132:154104, Table III
# ═══════════════════════════════════════════════════════════════════════════

C6_GRIMME = {
    "H":    3.09,
    "C":   25.78,
    "N":   16.07,
    "O":   12.35,
    "F":    5.18,
    "S":   99.48,
    "Cl":  54.02,
    "Br":  81.40,
    "I":  130.84,
    "P":   81.94,
}


# ═══════════════════════════════════════════════════════════════════════════
# DISPERSION SURFACE TENSION (γ_disp per element)
# ═══════════════════════════════════════════════════════════════════════════

# Base: γ_disp = -0.010 kJ/mol/Å² for carbon
# Calibrated so 200 Å² buried all-C surface gives -2 kJ/mol dispersion.
# Source: Grimme 2011 JCTC 7:291 S66 protein-ligand benchmarks.
GAMMA_DISP_BASE = -0.010  # kJ/mol/Å² (carbon reference)

# Element-specific γ scales as sqrt(C6_el / C6_C) (geometric mean with protein)
C6_C = C6_GRIMME["C"]
GAMMA_DISP = {
    el: GAMMA_DISP_BASE * math.sqrt(c6 / C6_C)
    for el, c6 in C6_GRIMME.items()
}

# Correction relative to carbon (what this module adds on top of hydrophobic)
GAMMA_CORRECTION = {
    el: GAMMA_DISP[el] - GAMMA_DISP_BASE
    for el in GAMMA_DISP
}


# ═══════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════

def compute_dispersion_correction(smiles, sasa_buried_A2, sasa_total_A2=None):
    """Element-specific dispersion correction relative to all-carbon.

    Halogenated/sulfur-containing ligands get a favorable bonus.
    Fluorinated/oxygenated ligands get a small penalty.

    Args:
        smiles: ligand SMILES
        sasa_buried_A2: buried SASA upon binding (Å²)
        sasa_total_A2: total ligand SASA (for burial fraction; optional)

    Returns:
        ΔG_disp_correction in kJ/mol (negative = more favorable than average C)
    """
    atom_counts = _count_heavy_atoms(smiles)
    if not atom_counts:
        return 0.0

    total_heavy = sum(atom_counts.values())
    if total_heavy == 0:
        return 0.0

    # Burial fraction
    f_burial = 0.6
    if sasa_total_A2 and sasa_total_A2 > 0:
        f_burial = min(sasa_buried_A2 / sasa_total_A2, 1.0)

    buried_sasa = sasa_buried_A2

    # Each element's share of buried SASA ≈ proportional to atom count
    correction = 0.0
    for el, count in atom_counts.items():
        gc = GAMMA_CORRECTION.get(el, 0.0)
        frac = count / total_heavy
        correction += gc * frac * buried_sasa

    return correction


def _count_heavy_atoms(smiles):
    """Count heavy atoms by element from SMILES."""
    try:
        from openbabel import pybel
        mol = pybel.readstring("smi", smiles)
        obmol = mol.OBMol
        counts = {}
        EMAP = {6: "C", 7: "N", 8: "O", 9: "F", 15: "P",
                16: "S", 17: "Cl", 35: "Br", 53: "I"}
        for i in range(obmol.NumAtoms()):
            an = obmol.GetAtom(i+1).GetAtomicNum()
            el = EMAP.get(an)
            if el:
                counts[el] = counts.get(el, 0) + 1
        return counts
    except ImportError:
        pass

    # Fallback: crude SMILES parse
    counts = {}
    s = smiles
    for token, el in [("Cl", "Cl"), ("Br", "Br")]:
        n = s.count(token)
        if n > 0:
            counts[el] = n
            s = s.replace(token, "XX")  # mask
    for char, el in [("C", "C"), ("c", "C"), ("N", "N"), ("n", "N"),
                     ("O", "O"), ("o", "O"), ("S", "S"), ("s", "S"),
                     ("F", "F"), ("I", "I"), ("P", "P")]:
        counts[el] = counts.get(el, 0) + s.count(char)
    return {k: v for k, v in counts.items() if v > 0}


def dispersion_decomposition(smiles):
    """Diagnostic: per-element dispersion coefficients for a molecule."""
    counts = _count_heavy_atoms(smiles)
    result = {}
    for el, count in sorted(counts.items()):
        result[el] = {
            "count": count,
            "C6_grimme": C6_GRIMME.get(el, 0),
            "gamma_disp": GAMMA_DISP.get(el, GAMMA_DISP_BASE),
            "gamma_correction": GAMMA_CORRECTION.get(el, 0.0),
        }
    return result