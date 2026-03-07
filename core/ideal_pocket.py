"""
core/ideal_pocket.py — Physics-Optimal Binding Pocket Computation

Given a target (metal ion or guest molecule) and desired donor set,
compute the ideal 3D arrangement of interaction elements with zero
material constraints.

The output is the reference standard that every material system
(chelator, MOF, protein, cage, MIP, etc.) is scored against.

Physics sources:
  - Shannon ionic radii (Acta Cryst. A32:751, 1976)
  - Ideal M-L bond lengths = r_ion + r_donor (donor covalent radii from CSD averages)
  - Coordination geometries: exact 3D positions from symmetry operations
  - Desolvation: Marcus 1991 hydration free energies
  - Per-bond energies: consistent with unified scorer calibration

No fitted parameters. No material constraints. Pure physics.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# SHANNON IONIC RADII (Å) — CN-dependent
# Shannon 1976, Acta Cryst. A32:751
# Format: {metal: {CN: radius}}
# ═══════════════════════════════════════════════════════════════════════════

SHANNON_RADII = {
    # Divalent first-row transition metals (most common)
    "Ca2+": {6: 1.00, 8: 1.12},
    "Mg2+": {4: 0.57, 6: 0.72},
    "Mn2+": {4: 0.66, 6: 0.83},
    "Fe2+": {4: 0.63, 6: 0.78},
    "Co2+": {4: 0.58, 6: 0.75},
    "Ni2+": {4: 0.55, 6: 0.69},
    "Cu2+": {4: 0.57, 5: 0.65, 6: 0.73},
    "Zn2+": {4: 0.60, 5: 0.68, 6: 0.74},
    "Cd2+": {4: 0.78, 6: 0.95, 8: 1.10},
    "Hg2+": {2: 0.69, 4: 0.96, 6: 1.02},
    "Pb2+": {4: 0.98, 6: 1.19, 8: 1.29},

    # Trivalent
    "Fe3+": {4: 0.49, 6: 0.65},
    "Al3+": {4: 0.39, 6: 0.54},
    "Cr3+": {6: 0.62},
    "Co3+": {6: 0.55},
    "Ga3+": {4: 0.47, 6: 0.62},
    "In3+": {6: 0.80},
    "La3+": {6: 1.03, 8: 1.16, 9: 1.22},
    "Lu3+": {6: 0.86, 8: 0.98},
    "Y3+":  {6: 0.90, 8: 1.02},
    "Eu3+": {6: 0.95, 8: 1.07},
    "Gd3+": {6: 0.94, 8: 1.05},

    # Other oxidation states
    "Cu+":  {2: 0.46, 4: 0.60},
    "Ag+":  {2: 0.67, 4: 1.00, 6: 1.15},
    "Au+":  {2: 0.77},
    "Au3+": {4: 0.64},

    # Actinides
    "UO2_2+": {5: 0.76, 6: 0.73},  # uranyl equatorial

    # Common anion targets (for host-guest)
    "Li+":  {4: 0.59, 6: 0.76},
    "Na+":  {4: 0.99, 6: 1.02, 8: 1.18},
    "K+":   {6: 1.38, 8: 1.51},
    "Rb+":  {6: 1.52, 8: 1.61},
    "Cs+":  {6: 1.67, 8: 1.74},
    "Ba2+": {6: 1.35, 8: 1.42},
    "Sr2+": {6: 1.18, 8: 1.26},
}


# ═══════════════════════════════════════════════════════════════════════════
# DONOR COVALENT RADII (Å) — added to ionic radius for bond length
# CSD averages for M-L bond length ≈ r_ion(M) + r_covalent(donor)
# ═══════════════════════════════════════════════════════════════════════════

DONOR_RADII = {
    # Atom type → effective donor radius (covalent + lone pair extension)
    "N_amine":           1.36,
    "N_pyridine":        1.34,
    "N_imidazole":       1.32,
    "N_imine":           1.32,
    "N_amide":           1.35,
    "O_carboxylate":     1.26,
    "O_hydroxyl":        1.28,
    "O_carbonyl":        1.24,
    "O_phenolate":       1.30,
    "O_phosphonate":     1.28,
    "O_ether":           1.30,
    "O_water":           1.34,
    "S_thiolate":        1.70,
    "S_thioether":       1.68,
    "S_dithiocarbamate": 1.72,
    "P_phosphine":       1.80,
    "Se_selenolate":     1.78,
}

# Shorthand: just the element
_ELEMENT_DONOR_RADIUS = {
    "N": 1.34,
    "O": 1.28,
    "S": 1.70,
    "P": 1.80,
    "Se": 1.78,
}


# ═══════════════════════════════════════════════════════════════════════════
# COORDINATION GEOMETRIES — 3D unit vectors
# Positions on a unit sphere; scale by bond length to get actual coords
# ═══════════════════════════════════════════════════════════════════════════

def _geometry_vectors(geometry, n_donors=None):
    """Return unit vectors for ideal coordination geometry.

    Each vector points from the metal center to the donor position.
    Vectors are on the unit sphere; multiply by bond length to get
    actual Cartesian coordinates.

    Returns list of np.array([x, y, z]).
    """
    if geometry == "linear" or (geometry == "auto" and n_donors == 2):
        return [
            np.array([0, 0, 1.0]),
            np.array([0, 0, -1.0]),
        ]

    elif geometry == "trigonal_planar" or (geometry == "auto" and n_donors == 3):
        return [
            np.array([1.0, 0, 0]),
            np.array([-0.5, math.sqrt(3)/2, 0]),
            np.array([-0.5, -math.sqrt(3)/2, 0]),
        ]

    elif geometry == "tetrahedral" or (geometry == "auto" and n_donors == 4):
        # Tetrahedral: vertices of a tetrahedron
        s = 1 / math.sqrt(3)
        return [
            np.array([s, s, s]),
            np.array([s, -s, -s]),
            np.array([-s, s, -s]),
            np.array([-s, -s, s]),
        ]

    elif geometry == "square_planar":
        return [
            np.array([1.0, 0, 0]),
            np.array([-1.0, 0, 0]),
            np.array([0, 1.0, 0]),
            np.array([0, -1.0, 0]),
        ]

    elif geometry == "trigonal_bipyramidal" or (geometry == "auto" and n_donors == 5):
        # 3 equatorial (120° apart in xy) + 2 axial (±z)
        return [
            np.array([1.0, 0, 0]),
            np.array([-0.5, math.sqrt(3)/2, 0]),
            np.array([-0.5, -math.sqrt(3)/2, 0]),
            np.array([0, 0, 1.0]),
            np.array([0, 0, -1.0]),
        ]

    elif geometry == "square_pyramidal":
        # 4 basal (square in xy) + 1 apical (+z)
        s = math.sqrt(2) / 2
        return [
            np.array([s, s, 0]),
            np.array([s, -s, 0]),
            np.array([-s, s, 0]),
            np.array([-s, -s, 0]),
            np.array([0, 0, 1.0]),
        ]

    elif geometry == "octahedral" or (geometry == "auto" and n_donors == 6):
        return [
            np.array([1.0, 0, 0]),
            np.array([-1.0, 0, 0]),
            np.array([0, 1.0, 0]),
            np.array([0, -1.0, 0]),
            np.array([0, 0, 1.0]),
            np.array([0, 0, -1.0]),
        ]

    elif geometry == "pentagonal_bipyramidal" or (geometry == "auto" and n_donors == 7):
        # 5 equatorial (72° apart) + 2 axial
        vecs = []
        for i in range(5):
            angle = 2 * math.pi * i / 5
            vecs.append(np.array([math.cos(angle), math.sin(angle), 0]))
        vecs.append(np.array([0, 0, 1.0]))
        vecs.append(np.array([0, 0, -1.0]))
        return vecs

    elif geometry == "cubic" or (geometry == "auto" and n_donors == 8):
        s = 1 / math.sqrt(3)
        return [
            np.array([s, s, s]),
            np.array([s, s, -s]),
            np.array([s, -s, s]),
            np.array([s, -s, -s]),
            np.array([-s, s, s]),
            np.array([-s, s, -s]),
            np.array([-s, -s, s]),
            np.array([-s, -s, -s]),
        ]

    elif geometry == "square_antiprism":
        # 8-coord: two squares rotated 45°
        vecs = []
        for i in range(4):
            angle = math.pi / 4 * (2 * i)
            vecs.append(np.array([math.cos(angle), math.sin(angle), 0.5]))
        for i in range(4):
            angle = math.pi / 4 * (2 * i + 1)
            vecs.append(np.array([math.cos(angle), math.sin(angle), -0.5]))
        # Normalize
        return [v / np.linalg.norm(v) for v in vecs]

    else:
        raise ValueError(f"Unknown geometry: {geometry}. Options: linear, "
                         f"trigonal_planar, tetrahedral, square_planar, "
                         f"trigonal_bipyramidal, square_pyramidal, octahedral, "
                         f"pentagonal_bipyramidal, cubic, square_antiprism, auto")


# ═══════════════════════════════════════════════════════════════════════════
# PREFERRED GEOMETRIES BY METAL
# ═══════════════════════════════════════════════════════════════════════════

PREFERRED_GEOMETRY = {
    # d0
    "Ca2+": ["octahedral", "cubic"],
    "La3+": ["square_antiprism", "cubic"],

    # d5 high spin
    "Mn2+": ["octahedral"],
    "Fe3+": ["octahedral", "tetrahedral"],

    # d6
    "Fe2+": ["octahedral"],
    "Co3+": ["octahedral"],

    # d7
    "Co2+": ["octahedral", "tetrahedral"],

    # d8
    "Ni2+": ["octahedral", "square_planar"],
    "Pd2+": ["square_planar"],
    "Pt2+": ["square_planar"],

    # d9
    "Cu2+": ["square_planar", "trigonal_bipyramidal", "octahedral"],

    # d10
    "Zn2+": ["tetrahedral", "octahedral"],
    "Cu+":  ["linear", "tetrahedral"],
    "Ag+":  ["linear", "tetrahedral"],
    "Au+":  ["linear"],
    "Au3+": ["square_planar"],
    "Cd2+": ["octahedral", "tetrahedral"],
    "Hg2+": ["linear", "tetrahedral"],

    # p-block
    "Pb2+": ["octahedral", "cubic"],
    "Al3+": ["octahedral", "tetrahedral"],
    "Ga3+": ["octahedral", "tetrahedral"],

    # Alkali/alkaline earth (host-guest)
    "Li+":  ["tetrahedral"],
    "Na+":  ["octahedral"],
    "K+":   ["cubic", "octahedral"],
    "Mg2+": ["octahedral"],
    "Ba2+": ["cubic"],
    "Sr2+": ["cubic", "octahedral"],
}


# ═══════════════════════════════════════════════════════════════════════════
# HYDRATION FREE ENERGIES (kJ/mol) — Marcus 1991
# Used for desolvation penalty estimation
# ═══════════════════════════════════════════════════════════════════════════

HYDRATION_DG = {
    # Divalent (Marcus 1991, J. Chem. Soc. Faraday Trans.)
    "Ca2+": -1505, "Mg2+": -1830, "Mn2+": -1760, "Fe2+": -1840,
    "Co2+": -1915, "Ni2+": -1980, "Cu2+": -2010, "Zn2+": -1955,
    "Cd2+": -1755, "Hg2+": -1760, "Pb2+": -1425, "Ba2+": -1305,
    "Sr2+": -1380,

    # Monovalent
    "Li+": -475, "Na+": -365, "K+": -295, "Rb+": -275, "Cs+": -250,
    "Cu+": -525, "Ag+": -430, "Au+": -615,

    # Trivalent
    "Fe3+": -4265, "Al3+": -4525, "Cr3+": -4340, "Co3+": -4495,
    "La3+": -3145, "Lu3+": -3425, "Gd3+": -3375, "Eu3+": -3360,
    "Y3+": -3450,
}


# ═══════════════════════════════════════════════════════════════════════════
# PER-BOND ENERGY ESTIMATES (kJ/mol)
# Consistent with unified scorer calibration parameters
# ═══════════════════════════════════════════════════════════════════════════

BOND_ENERGY = {
    # Donor subtype → typical M-L bond energy (negative = favorable)
    "N_amine":           -45.0,
    "N_pyridine":        -50.0,
    "N_imidazole":       -55.0,
    "N_imine":           -48.0,
    "O_carboxylate":     -35.0,
    "O_hydroxyl":        -30.0,
    "O_carbonyl":        -25.0,
    "O_phenolate":       -40.0,
    "O_phosphonate":     -38.0,
    "O_ether":           -20.0,
    "S_thiolate":        -65.0,
    "S_thioether":       -50.0,
    "S_dithiocarbamate": -60.0,
    "P_phosphine":       -70.0,
}

# Chelate ring stabilization (per ring)
CHELATE_RING_ENERGY = -8.0  # kJ/mol, consistent with scorer

# Macrocyclic effect
MACROCYCLIC_ENERGY = -12.0  # kJ/mol additional stabilization


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DonorPosition:
    """One donor element in the ideal pocket."""
    index: int                       # 0-indexed position
    donor_subtype: str               # e.g. "N_amine", "O_carboxylate"
    element: str                     # "N", "O", "S", "P"
    position_A: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bond_length_A: float = 0.0      # M-L distance
    tolerance_A: float = 0.05       # positional precision required
    bond_energy_kJ: float = 0.0     # per-bond contribution
    role: str = "equatorial"        # "axial", "equatorial", "apical"


@dataclass
class IdealPocket:
    """Physics-optimal binding pocket — the reference standard."""
    target: str                      # e.g. "Cu2+", "Pb2+"
    target_charge: int = 0
    geometry: str = ""               # coordination geometry name
    n_donors: int = 0
    donors: list = field(default_factory=list)  # list[DonorPosition]

    # Cavity metrics
    cavity_radius_A: float = 0.0     # average M-L distance
    cavity_volume_A3: float = 0.0    # 4/3 π r³
    ionic_radius_A: float = 0.0      # Shannon radius at this CN

    # Energetics
    total_bond_energy_kJ: float = 0.0     # sum of per-bond
    chelate_stabilization_kJ: float = 0.0
    macrocyclic_bonus_kJ: float = 0.0
    desolvation_penalty_kJ: float = 0.0
    ideal_dG_kJ: float = 0.0             # net binding free energy
    ideal_log_Ka: float = 0.0            # = -dG / (RT ln10)

    # Rigidity
    tightest_tolerance_A: float = 0.05
    rigidity_class: str = "preorganized"  # crystalline / preorganized / semi_flexible / any

    # Selectivity
    selectivity_notes: list = field(default_factory=list)

    # Description
    description: str = ""
    donor_signature: str = ""  # e.g. "4N_amine(tet)"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def compute_ideal_pocket(
    target: str,
    donor_subtypes: list,
    geometry: str = "auto",
    chelate_rings: int = 0,
    macrocyclic: bool = False,
    pH: float = 7.0,
) -> IdealPocket:
    """Compute the physics-optimal binding pocket for a target.

    Args:
        target:          Metal ion (e.g. "Cu2+") or guest name
        donor_subtypes:  List of donor subtypes, e.g. ["N_amine", "N_amine",
                         "O_carboxylate", "O_carboxylate"]
        geometry:        Coordination geometry ("auto" = infer from n_donors + metal)
        chelate_rings:   Number of chelate rings (from connected donors)
        macrocyclic:     Is the ligand macrocyclic?
        pH:              Working pH (affects protonation penalties)

    Returns:
        IdealPocket with 3D donor positions and energetics
    """
    n_donors = len(donor_subtypes)
    target_charge = _parse_charge(target)

    # Resolve geometry
    if geometry == "auto":
        geometry = _auto_geometry(target, n_donors)

    # Get ionic radius
    r_ion = _get_ionic_radius(target, n_donors)

    # Compute 3D positions
    vectors = _geometry_vectors(geometry, n_donors)
    if len(vectors) < n_donors:
        # Pad with additional positions if geometry has fewer sites
        vectors = vectors + [vectors[i % len(vectors)] for i in range(n_donors - len(vectors))]
    vectors = vectors[:n_donors]

    donors = []
    total_bond_E = 0.0

    for i, (dsub, vec) in enumerate(zip(donor_subtypes, vectors)):
        element = dsub.split("_")[0] if "_" in dsub else dsub
        r_donor = DONOR_RADII.get(dsub, _ELEMENT_DONOR_RADIUS.get(element, 1.30))
        bond_length = r_ion + r_donor

        pos_A = vec * bond_length  # 3D position in Ångströms

        # Tolerance: tighter for coordination bonds, looser for H-bonds
        tol = _tolerance_for_donor(dsub, geometry)

        # Per-bond energy
        E_bond = BOND_ENERGY.get(dsub, -30.0)
        total_bond_E += E_bond

        # Role assignment (geometry-dependent)
        if geometry == "octahedral" and i >= 4:
            role = "axial"
        elif geometry in ("trigonal_bipyramidal",) and i >= 3:
            role = "axial"
        elif geometry == "square_pyramidal" and i >= 4:
            role = "apical"
        else:
            role = "equatorial"

        donors.append(DonorPosition(
            index=i,
            donor_subtype=dsub,
            element=element,
            position_A=pos_A,
            bond_length_A=round(bond_length, 3),
            tolerance_A=tol,
            bond_energy_kJ=E_bond,
            role=role,
        ))

    # Chelate ring stabilization
    chelate_E = chelate_rings * CHELATE_RING_ENERGY
    macro_E = MACROCYCLIC_ENERGY if macrocyclic else 0.0

    # Desolvation penalty
    desolv = _desolvation_penalty(target, donor_subtypes)

    # Net binding energy
    ideal_dG = total_bond_E + chelate_E + macro_E + desolv
    ideal_log_Ka = -ideal_dG / 5.71  # RT ln(10) at 25°C ≈ 5.71 kJ/mol

    # Cavity metrics
    avg_bond = np.mean([d.bond_length_A for d in donors]) if donors else 0.0
    cavity_vol = (4/3) * math.pi * avg_bond**3

    # Rigidity classification
    tightest = min(d.tolerance_A for d in donors) if donors else 0.50
    if tightest < 0.03:
        rig = "crystalline"
    elif tightest < 0.10:
        rig = "preorganized"
    elif tightest < 0.25:
        rig = "semi_flexible"
    else:
        rig = "any"

    # Donor signature
    from collections import Counter
    counts = Counter(donor_subtypes)
    sig_parts = [f"{v}{k}" for k, v in sorted(counts.items())]
    sig = "+".join(sig_parts) + f"({geometry[:3]})"

    # Selectivity notes
    sel_notes = _selectivity_notes(target, donor_subtypes, geometry)

    return IdealPocket(
        target=target,
        target_charge=target_charge,
        geometry=geometry,
        n_donors=n_donors,
        donors=donors,
        cavity_radius_A=round(avg_bond, 3),
        cavity_volume_A3=round(cavity_vol, 1),
        ionic_radius_A=r_ion,
        total_bond_energy_kJ=round(total_bond_E, 1),
        chelate_stabilization_kJ=round(chelate_E, 1),
        macrocyclic_bonus_kJ=round(macro_E, 1),
        desolvation_penalty_kJ=round(desolv, 1),
        ideal_dG_kJ=round(ideal_dG, 1),
        ideal_log_Ka=round(ideal_log_Ka, 1),
        tightest_tolerance_A=tightest,
        rigidity_class=rig,
        selectivity_notes=sel_notes,
        description=_describe_pocket(target, donors, geometry, rig),
        donor_signature=sig,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: COMMON POCKET TYPES
# ═══════════════════════════════════════════════════════════════════════════

def ideal_pocket_for_metal(metal, n_donors=None, donor_element="N",
                           geometry="auto", chelate_rings=0,
                           macrocyclic=False):
    """Quick ideal pocket from metal + donor element + CN.

    If n_donors is None, uses the metal's preferred coordination number.
    """
    if n_donors is None:
        prefs = PREFERRED_GEOMETRY.get(metal, ["octahedral"])
        geometry = prefs[0] if geometry == "auto" else geometry
        vecs = _geometry_vectors(geometry)
        n_donors = len(vecs)

    # Build donor subtypes from element
    subtype_map = {
        "N": "N_amine", "O": "O_carboxylate", "S": "S_thiolate",
        "P": "P_phosphine",
    }
    dsub = subtype_map.get(donor_element, f"{donor_element}_generic")
    donor_subtypes = [dsub] * n_donors

    return compute_ideal_pocket(metal, donor_subtypes, geometry,
                                chelate_rings, macrocyclic)


def ideal_pocket_mixed(metal, donor_subtypes_str,
                       geometry="auto", chelate_rings=0,
                       macrocyclic=False):
    """Quick ideal pocket from a donor string.

    Example: ideal_pocket_mixed("Cu2+", "2N_amine+2O_carboxylate")
    """
    donor_subtypes = _parse_donor_string(donor_subtypes_str)
    return compute_ideal_pocket(metal, donor_subtypes, geometry,
                                chelate_rings, macrocyclic)


# ═══════════════════════════════════════════════════════════════════════════
# HOST-GUEST CAVITY
# ═══════════════════════════════════════════════════════════════════════════

def ideal_host_cavity(guest_diameter_A, n_contacts=6,
                      contact_type="O_ether", cavity_shape="sphere"):
    """Ideal host cavity for a spherical guest.

    Used for crown ethers, cyclodextrins, cryptands, cages.

    Args:
        guest_diameter_A: diameter of guest in Å
        n_contacts: number of contact points with guest
        contact_type: donor subtype at each contact
        cavity_shape: "sphere", "cylinder", "cone"

    Returns:
        IdealPocket (with guest as "target")
    """
    r_guest = guest_diameter_A / 2
    # Contact distance = guest radius + van der Waals of host atom
    vdw = {"O": 1.52, "N": 1.55, "S": 1.80, "C": 1.70}
    element = contact_type.split("_")[0] if "_" in contact_type else "O"
    r_vdw = vdw.get(element, 1.52)
    contact_distance = r_guest + r_vdw

    # Distribute contacts evenly
    if n_contacts <= 6:
        geometry = {2: "linear", 3: "trigonal_planar", 4: "tetrahedral",
                    5: "trigonal_bipyramidal", 6: "octahedral"}.get(n_contacts, "octahedral")
    else:
        geometry = "cubic"

    vectors = _geometry_vectors(geometry, n_contacts)[:n_contacts]

    donors = []
    for i, vec in enumerate(vectors):
        pos = vec * contact_distance
        E = BOND_ENERGY.get(contact_type, -20.0) * 0.5  # weaker for HG
        donors.append(DonorPosition(
            index=i, donor_subtype=contact_type, element=element,
            position_A=pos, bond_length_A=round(contact_distance, 3),
            tolerance_A=0.20, bond_energy_kJ=E, role="equatorial",
        ))

    total_E = sum(d.bond_energy_kJ for d in donors)
    cavity_vol = (4/3) * math.pi * (r_guest + 0.5)**3  # accessible volume
    desolv = sum(10.0 for _ in donors)  # rough HG desolvation

    ideal_dG = total_E + desolv
    ideal_log_Ka = -ideal_dG / 5.71

    return IdealPocket(
        target=f"guest_D={guest_diameter_A:.1f}A",
        target_charge=0,
        geometry=geometry,
        n_donors=n_contacts,
        donors=donors,
        cavity_radius_A=round(contact_distance, 3),
        cavity_volume_A3=round(cavity_vol, 1),
        ionic_radius_A=r_guest,
        total_bond_energy_kJ=round(total_E, 1),
        desolvation_penalty_kJ=round(desolv, 1),
        ideal_dG_kJ=round(ideal_dG, 1),
        ideal_log_Ka=round(ideal_log_Ka, 1),
        tightest_tolerance_A=0.20,
        rigidity_class="semi_flexible",
        donor_signature=f"{n_contacts}{contact_type}({cavity_shape})",
        description=f"Host cavity for {guest_diameter_A:.1f}Å guest, "
                    f"{n_contacts}×{contact_type}, {cavity_shape}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# DEVIATION SCORING — measure how far a real structure deviates from ideal
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeviationReport:
    """How far a real pocket deviates from the ideal."""
    ideal: IdealPocket = None
    material_name: str = ""
    per_donor_deviation_A: list = field(default_factory=list)
    mean_deviation_A: float = 0.0
    max_deviation_A: float = 0.0
    missing_donors: int = 0
    extra_donors: int = 0
    geometry_match: float = 0.0      # 0-1, angular deviation metric
    fidelity_score: float = 0.0      # 0-1, overall score
    notes: str = ""


def score_deviation(ideal: IdealPocket,
                    actual_positions_A: list,
                    actual_donor_subtypes: list = None) -> DeviationReport:
    """Score a real pocket against the ideal.

    Args:
        ideal: the IdealPocket reference
        actual_positions_A: list of np.array([x,y,z]) for actual donors
        actual_donor_subtypes: list of donor subtypes (for chemistry match)

    Returns:
        DeviationReport with fidelity_score (0-1)
    """
    n_ideal = len(ideal.donors)
    n_actual = len(actual_positions_A)

    # Match ideal to actual donors (closest-distance assignment)
    deviations = []
    used = set()

    for ideal_donor in ideal.donors:
        best_dist = float("inf")
        best_idx = -1
        for j, actual_pos in enumerate(actual_positions_A):
            if j in used:
                continue
            dist = np.linalg.norm(ideal_donor.position_A - actual_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx >= 0:
            used.add(best_idx)
            deviations.append(best_dist)
        else:
            deviations.append(float("inf"))

    # Compute metrics
    finite_devs = [d for d in deviations if d < 100]
    mean_dev = np.mean(finite_devs) if finite_devs else float("inf")
    max_dev = max(finite_devs) if finite_devs else float("inf")
    missing = sum(1 for d in deviations if d >= 100)
    extra = max(0, n_actual - n_ideal)

    # Fidelity score: exponential decay with deviation
    # Score = 1.0 at 0 deviation, ~0.5 at tolerance, ~0.1 at 3× tolerance
    if mean_dev < 100:
        avg_tol = np.mean([d.tolerance_A for d in ideal.donors])
        fidelity = math.exp(-mean_dev / avg_tol) if avg_tol > 0 else 0.0
    else:
        fidelity = 0.0

    # Penalty for missing/extra donors
    fidelity *= (1 - 0.15 * missing) * (1 - 0.05 * extra)
    fidelity = max(0.0, min(1.0, fidelity))

    return DeviationReport(
        ideal=ideal,
        per_donor_deviation_A=deviations,
        mean_deviation_A=round(mean_dev, 3),
        max_deviation_A=round(max_dev, 3),
        missing_donors=missing,
        extra_donors=extra,
        fidelity_score=round(fidelity, 3),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_ideal_pocket(p):
    """Pretty-print an ideal pocket."""
    print()
    print(f"  MABE Ideal Pocket — {p.target}")
    print(f"  Geometry: {p.geometry} ({p.n_donors} donors)")
    print(f"  Signature: {p.donor_signature}")
    print(f"  Ionic radius: {p.ionic_radius_A:.3f} Å")
    print(f"  Cavity radius: {p.cavity_radius_A:.3f} Å, volume: {p.cavity_volume_A3:.0f} ų")
    print()
    print(f"  ── Donors ──")
    for d in p.donors:
        print(f"    [{d.index}] {d.donor_subtype:20s} r={d.bond_length_A:.3f}Å "
              f"±{d.tolerance_A:.2f}Å  E={d.bond_energy_kJ:+.0f} kJ  "
              f"pos=({d.position_A[0]:+.3f},{d.position_A[1]:+.3f},{d.position_A[2]:+.3f})")
    print()
    print(f"  ── Energetics ──")
    print(f"  Bond energy:     {p.total_bond_energy_kJ:+.1f} kJ/mol")
    print(f"  Chelate rings:   {p.chelate_stabilization_kJ:+.1f} kJ/mol")
    print(f"  Macrocyclic:     {p.macrocyclic_bonus_kJ:+.1f} kJ/mol")
    print(f"  Desolvation:     {p.desolvation_penalty_kJ:+.1f} kJ/mol")
    print(f"  Ideal ΔG:        {p.ideal_dG_kJ:+.1f} kJ/mol")
    print(f"  Ideal log Ka:    {p.ideal_log_Ka:+.1f}")
    print(f"  Rigidity class:  {p.rigidity_class} (±{p.tightest_tolerance_A:.2f} Å)")
    if p.selectivity_notes:
        print(f"  Selectivity: {'; '.join(p.selectivity_notes)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _parse_charge(target):
    if "3+" in target: return 3
    if "2+" in target: return 2
    if "+" in target: return 1
    if "3-" in target: return -3
    if "2-" in target: return -2
    if "-" in target: return -1
    return 0


def _get_ionic_radius(target, cn):
    radii = SHANNON_RADII.get(target, {})
    if cn in radii:
        return radii[cn]
    # Closest available CN
    if radii:
        closest_cn = min(radii.keys(), key=lambda k: abs(k - cn))
        return radii[closest_cn]
    return 0.80  # fallback


def _auto_geometry(target, n_donors):
    prefs = PREFERRED_GEOMETRY.get(target, [])
    for geom in prefs:
        n_sites = len(_geometry_vectors(geom))
        if n_sites == n_donors:
            return geom
    # Fallback by n_donors
    return {2: "linear", 3: "trigonal_planar", 4: "tetrahedral",
            5: "trigonal_bipyramidal", 6: "octahedral",
            7: "pentagonal_bipyramidal", 8: "cubic"}.get(n_donors, "octahedral")


def _tolerance_for_donor(dsub, geometry):
    element = dsub.split("_")[0] if "_" in dsub else dsub
    # Base tolerance from bond type
    base = {"N": 0.05, "O": 0.08, "S": 0.06, "P": 0.07, "Se": 0.07}.get(element, 0.10)
    # Tighter for rigid geometries
    if geometry in ("square_planar", "linear"):
        base *= 0.8
    elif geometry in ("octahedral", "tetrahedral"):
        base *= 1.0
    else:
        base *= 1.2
    return round(base, 3)


def _desolvation_penalty(target, donor_subtypes):
    # Metal desolvation: fraction of hydration energy proportional to donors displaced
    dG_hyd = HYDRATION_DG.get(target, -1500)
    n = len(donor_subtypes)
    # Each donor displaces ~1 water from first coordination shell
    # Partial desolvation: ~(n/CN_full) × fraction of hydration energy
    cn_full = 6  # typical aqueous CN
    metal_desolv = abs(dG_hyd) * (n / cn_full) * 0.10  # ~10% of full hydration per donor

    # Donor desolvation
    donor_desolv = 0.0
    for dsub in donor_subtypes:
        element = dsub.split("_")[0]
        costs = {"N": 12, "O": 15, "S": 8, "P": 10, "Se": 6}
        donor_desolv += costs.get(element, 10)

    return metal_desolv + donor_desolv


def _parse_donor_string(s):
    """Parse "2N_amine+2O_carboxylate" → ["N_amine","N_amine","O_carboxylate","O_carboxylate"]"""
    result = []
    for part in s.split("+"):
        part = part.strip()
        # Extract count prefix
        i = 0
        while i < len(part) and part[i].isdigit():
            i += 1
        count = int(part[:i]) if i > 0 else 1
        donor = part[i:]
        result.extend([donor] * count)
    return result


def _selectivity_notes(target, donors, geometry):
    notes = []
    charge = _parse_charge(target)

    # HSAB notes
    hard_donors = sum(1 for d in donors if d.split("_")[0] in ("O",))
    soft_donors = sum(1 for d in donors if d.split("_")[0] in ("S", "P", "Se"))
    if charge >= 3 and soft_donors > 0:
        notes.append("Hard metal with soft donors — may lose selectivity vs soft competitors")
    if target in ("Hg2+", "Cd2+", "Pb2+") and hard_donors > soft_donors:
        notes.append("Soft metal with hard donors — consider S/P donors for selectivity")

    # Size selectivity
    if geometry in ("tetrahedral", "square_planar"):
        notes.append(f"{geometry} cavity is size-selective — excludes larger ions")
    if geometry == "octahedral":
        notes.append("Octahedral is accommodating — may need secondary selectivity (HSAB, chelate)")

    return notes


def _describe_pocket(target, donors, geometry, rigidity):
    elements = sorted(set(d.element for d in donors))
    return (f"Ideal {geometry} pocket for {target}: "
            f"{len(donors)} donors ({','.join(elements)}), "
            f"rigidity={rigidity}")
