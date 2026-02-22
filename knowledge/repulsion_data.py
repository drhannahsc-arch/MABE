"""
knowledge/repulsion_data.py - Data for repulsion force calculations.

van der Waals radii, Born-Mayer parameters, and structural charge data.
"""

import math

# ═══════════════════════════════════════════════════════════════════════════
# van der Waals radii (Å) — Bondi (1964), updated Alvarez (2013)
# ═══════════════════════════════════════════════════════════════════════════

VDW_RADII = {
    # Metals (as ions, these are smaller than atomic vdW)
    "Pb": 2.02, "Cu": 1.40, "Ni": 1.63, "Zn": 1.39,
    "Fe": 1.56, "Au": 1.66, "Hg": 1.55, "Cd": 1.58,
    "Ag": 1.72, "Ca": 2.31, "Mg": 1.73, "Na": 2.27,
    "K":  2.75, "Ba": 2.68, "Ce": 2.42, "Al": 1.84,
    "Mn": 1.61, "Co": 1.52, "Cr": 1.66, "U":  1.86,

    # Donor atoms
    "O": 1.52, "N": 1.55, "S": 1.80, "P": 1.80, "C": 1.70,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "H": 1.20, "Se": 1.90, "As": 1.85,
}

# Ionic radii (Å) — Shannon (1976) for common oxidation states
# These are the crystal radii, smaller than vdW
IONIC_RADII = {
    "Pb2+": 1.19, "Cu2+": 0.73, "Ni2+": 0.69, "Zn2+": 0.74,
    "Fe3+": 0.645, "Fe2+": 0.78, "Au3+": 0.85, "Hg2+": 1.02,
    "Cd2+": 0.95, "Ag+": 1.15, "Ca2+": 1.00, "Mg2+": 0.72,
    "Na+": 1.02, "K+": 1.38, "Ba2+": 1.35, "Ce3+": 1.01,
    "Al3+": 0.535, "Mn2+": 0.83, "Co2+": 0.745, "Cr3+": 0.615,
    "UO2_2+": 0.73,
}

# Hydrated ion radii (Å) — Marcus (1988)
HYDRATED_RADII = {
    "Pb2+": 4.01, "Cu2+": 4.19, "Ni2+": 4.04, "Zn2+": 4.30,
    "Fe3+": 4.57, "Fe2+": 4.28, "Au3+": 3.50, "Hg2+": 3.50,
    "Cd2+": 4.26, "Ag+": 3.41, "Ca2+": 4.12, "Mg2+": 4.28,
    "Na+": 3.58, "K+": 3.31, "Ba2+": 4.04, "Ce3+": 4.52,
    "Al3+": 4.75, "Mn2+": 4.38, "Co2+": 4.23, "Cr3+": 4.61,
}


# ═══════════════════════════════════════════════════════════════════════════
# Structure charge data
# ═══════════════════════════════════════════════════════════════════════════

# Net framework charge per unit cell (conceptual, determines sign of selectivity)
FRAMEWORK_CHARGE = {
    "zeolite":            -1,   # AlO₄⁻ substitution → net negative → attracts cations
    "ldh":                +1,   # M²⁺/M³⁺ layers → net positive → attracts anions
    "mof":                 0,   # varies, generally neutral pores
    "cof":                 0,   # neutral framework
    "mesoporous_silica":  -1,   # SiO⁻ at surface above pH 3
    "silica_np":          -1,   # same
    "dna_origami_cage":   -1,   # phosphate backbone → negative
    "carbon_nanotube":     0,   # neutral unless functionalized
    "graphene_oxide":     -1,   # carboxyl/hydroxyl groups
    "mip":                 0,   # depends on monomer
    "coordination_cage":   0,   # varies
    "protein_cage":        0,   # varies with pH
    "dendrimer":           0,   # varies
    "none":                0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Born-Mayer repulsion parameters
# ═══════════════════════════════════════════════════════════════════════════

# Born-Mayer: V_rep = B × exp(-r / ρ)
# ρ (softness parameter) ≈ 0.345 Å for most ion pairs
BORN_MAYER_RHO = 0.345  # Å

# B is calibrated so that V_rep = ~500 kJ/mol at r = sum of ionic radii
# (i.e., hard contact). Exact value doesn't matter much because the
# exponential makes it essentially a hard wall.
BORN_MAYER_B = 500.0  # kJ/mol (pre-factor)


def get_hydrated_radius(identity: str, charge: float) -> float:
    """Get hydrated radius in Å. Falls back to estimation from charge."""
    key = identity.lower()
    # Try direct lookup with charge
    for suffix in [f"{abs(charge):.0f}+", f"{abs(charge):.0f}-"]:
        sign = "+" if charge > 0 else "-"
        full_key = f"{key.capitalize()}{abs(charge):.0f}{sign}"
        if full_key in HYDRATED_RADII:
            return HYDRATED_RADII[full_key]

    # Try common keys
    lookup_map = {
        "lead": "Pb2+", "copper": "Cu2+", "nickel": "Ni2+", "zinc": "Zn2+",
        "iron": "Fe3+" if abs(charge) > 2.5 else "Fe2+",
        "gold": "Au3+", "mercury": "Hg2+", "cadmium": "Cd2+",
        "silver": "Ag+", "calcium": "Ca2+", "magnesium": "Mg2+",
        "sodium": "Na+", "potassium": "K+", "barium": "Ba2+",
        "cerium": "Ce3+", "aluminum": "Al3+", "manganese": "Mn2+",
        "cobalt": "Co2+", "chromium": "Cr3+",
    }
    mapped = lookup_map.get(key)
    if mapped and mapped in HYDRATED_RADII:
        return HYDRATED_RADII[mapped]

    # Estimate: r_hyd ≈ 2.0 + 0.8 × |z| (rough)
    return 2.0 + 0.8 * abs(charge)


def get_ionic_radius(identity: str, charge: float) -> float:
    """Get ionic radius in Å."""
    key = identity.lower()
    lookup_map = {
        "lead": "Pb2+", "copper": "Cu2+", "nickel": "Ni2+", "zinc": "Zn2+",
        "iron": "Fe3+" if abs(charge) > 2.5 else "Fe2+",
        "gold": "Au3+", "mercury": "Hg2+", "cadmium": "Cd2+",
        "silver": "Ag+", "calcium": "Ca2+", "magnesium": "Mg2+",
        "sodium": "Na+", "potassium": "K+", "barium": "Ba2+",
        "cerium": "Ce3+", "aluminum": "Al3+", "manganese": "Mn2+",
        "cobalt": "Co2+", "chromium": "Cr3+",
    }
    mapped = lookup_map.get(key)
    if mapped and mapped in IONIC_RADII:
        return IONIC_RADII[mapped]
    return 1.0  # fallback
