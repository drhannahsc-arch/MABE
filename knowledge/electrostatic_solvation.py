"""
knowledge/electrostatic_solvation.py — Electrostatic solvation for protein-ligand binding

Two tiers:
  Tier 1 (default): Group-additive electrostatic solvation from Cabani 1981 +
    Wolfenden 1981. Pre-tabulated ΔG_elec per functional group in water.
    Burial fraction determines how much is lost upon binding.

  Tier 2 (when 3D available): Born model with Marcus-derived effective radii
    for charged groups. More accurate for ionic interactions.

Physics of electrostatic desolvation in binding:
  In water (ε=78.4): charges and dipoles are stabilized by high dielectric.
  In protein pocket (ε=4-20): stabilization reduced.
  Cost of burying a charged group = ΔG_Born(ε_pocket) - ΔG_Born(ε_water)
  This is ALWAYS positive (unfavorable) for charges.
  The cost is partially compensated by salt bridges / charged H-bonds.

Parameter sources:
  - Marcus 1994, Chem. Rev. 94:1927 — ionic hydration energies
  - Cabani 1981, J. Solution Chem. 10:563 — group solvation
  - Wolfenden 1981 — amino acid sidechain transfer
  - Warshel 1984 — effective dielectric in proteins
"""

import math

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BORN_FACTOR = 1389.4    # kJ·Å/mol = e²·N_A / (4πε₀) in Å units
EPSILON_WATER = 78.4    # dielectric constant of water at 298 K


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: GROUP-ADDITIVE ELECTROSTATIC SOLVATION
# ═══════════════════════════════════════════════════════════════════════════

# Electrostatic component of group solvation free energy in water (kJ/mol).
# These capture the POLAR/ELECTROSTATIC part of solvation only.
# The nonpolar (cavity + vdW) part is handled by ligand_desolvation.py.
#
# Source: Cabani 1981 total group solvation decomposed into electrostatic
# and nonpolar components (Pierotti 1976 cavity model subtraction).
# Cross-validated against Wolfenden 1981 amino acid transfer values.
#
# Negative = stabilized by water (favorable solvation).
# Burying this group costs +|value| per unit burial fraction.

GROUP_ELEC_SOLVATION = {
    # ── Charged groups (very large electrostatic solvation) ──────
    "NH3+":          -315.0,   # protonated amine (Marcus NH4+ scaled)
    "COO-":          -375.0,   # carboxylate (Marcus acetate)
    "guanidinium+":  -280.0,   # Arg sidechain (distributed charge)
    "imidazolium+":  -260.0,   # protonated His
    "phosphate2-":   -500.0,   # phosphate diester (estimated)
    "sulfate-":      -340.0,   # sulfate/sulfonate

    # ── Neutral polar groups ─────────────────────────────────────
    # From Cabani 1981 electrostatic decomposition
    "OH":            -25.0,    # hydroxyl (alcohol)
    "phenol_OH":     -28.0,    # phenol (more acidic, stronger dipole)
    "C=O_amide":     -30.0,    # amide carbonyl
    "C=O_ketone":    -20.0,    # ketone carbonyl
    "C=O_ester":     -15.0,    # ester carbonyl (less polar)
    "NH_amide":      -22.0,    # amide N-H
    "NH2_amine":     -18.0,    # primary amine
    "NH_amine":      -12.0,    # secondary amine
    "N_aromatic":    -16.0,    # pyridine N
    "S=O":           -35.0,    # sulfoxide
    "SO2":           -45.0,    # sulfone
    "SH":            -8.0,     # thiol (weakly polar)
    "S_thioether":   -5.0,     # thioether (very weakly polar)
    "F":             -3.0,     # fluorine (weak dipole)
    "Cl":            -5.0,     # chlorine
    "NO2":           -20.0,    # nitro group

    # ── Effectively nonpolar (minimal electrostatic solvation) ───
    "CH_aliphatic":  -0.5,     # C-H (tiny induced dipole)
    "CH_aromatic":   -2.0,     # aromatic C-H (quadrupole)
}


def compute_group_elec_desolvation(buried_groups, epsilon_pocket=8.0):
    """Tier 1: Group-additive electrostatic desolvation cost.

    Args:
        buried_groups: list of dicts with "type" and "burial_fraction" keys,
            OR list of strings (type names, burial_fraction=1.0 assumed).
        epsilon_pocket: effective dielectric of protein pocket (4-20).

    Returns:
        ΔG_elec_desolv in kJ/mol (positive = unfavorable cost).
    """
    if not buried_groups:
        return 0.0

    # Scaling: fraction of electrostatic solvation lost upon burial
    # In water: full solvation. In pocket with ε_pocket: partial.
    # Fraction lost ≈ 1 - ε_water/ε_pocket... no, that's wrong.
    # Correct: ΔG_transfer = ΔG_elec(ε_pocket) - ΔG_elec(ε_water)
    # For Born: ratio = (1/ε_pocket - 1/ε_water) / (1/1 - 1/ε_water)
    # For groups: approximate as f_lost = (1 - ε_pocket/ε_water) is also wrong
    # 
    # Simple correct model: electrostatic solvation scales as (1 - 1/ε).
    # In water: ΔG_w = ΔG_vac × (1 - 1/78.4) ≈ 0.987 × ΔG_vac
    # In pocket: ΔG_p = ΔG_vac × (1 - 1/ε_pocket)
    # Cost = ΔG_p - ΔG_w = ΔG_vac × (1/ε_water - 1/ε_pocket)
    # Since ΔG_w ≈ ΔG_vac × 0.987, we have:
    # Cost ≈ ΔG_w × (1/ε_water - 1/ε_pocket) / (1 - 1/ε_water)
    # ≈ ΔG_w × (1/ε_water - 1/ε_pocket) / 0.987
    # For ε_pocket=8: cost ≈ ΔG_w × (0.0128 - 0.125) / 0.987 ≈ ΔG_w × (-0.114)
    # Wait — this gives a NEGATIVE cost, meaning burial is favorable?
    # No: ΔG_w is negative (favorable). Losing it means cost is POSITIVE.
    #
    # Let me think again:
    # ΔG_in_water = large negative (well solvated)
    # ΔG_in_pocket = smaller negative (less solvated)
    # Cost of transfer = ΔG_pocket - ΔG_water > 0 (less favorable = positive cost)
    #
    # Fraction of electrostatic solvation RETAINED in pocket:
    # f_retain = (1 - 1/ε_pocket) / (1 - 1/ε_water)
    # Fraction LOST = 1 - f_retain
    # Cost = -ΔG_water × f_lost × burial_fraction

    f_retain = (1.0 - 1.0/epsilon_pocket) / (1.0 - 1.0/EPSILON_WATER)
    f_lost = 1.0 - f_retain  # fraction of electrostatic solvation lost

    total_cost = 0.0
    for group in buried_groups:
        if isinstance(group, dict):
            gtype = group.get("type", "OH")
            bf = group.get("burial_fraction", 1.0)
        elif isinstance(group, str):
            gtype = group
            bf = 1.0
        else:
            continue

        dG_water = GROUP_ELEC_SOLVATION.get(gtype, GROUP_ELEC_SOLVATION.get("OH", -25.0))
        # Cost = how much electrostatic solvation is lost
        # dG_water is negative → -dG_water is positive → cost is positive
        cost = -dG_water * f_lost * bf
        total_cost += cost

    return total_cost


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: BORN MODEL WITH MARCUS EFFECTIVE RADII
# ═══════════════════════════════════════════════════════════════════════════

# Effective Born radii back-solved from Marcus 1994 experimental hydration
# energies. These are NOT crystal radii — they include the solvation shell.
# Using these reproduces experimental ΔG_hydr exactly by construction.

MARCUS_BORN_RADII = {
    # Monovalent cations
    "Li+":    2.59,
    "Na+":    3.24,
    "K+":     3.90,
    "Rb+":    4.17,
    "Cs+":    4.48,
    "NH4+":   4.03,
    # Divalent cations
    "Mg2+":   2.84,
    "Ca2+":   3.41,
    "Zn2+":   2.68,
    "Cu2+":   2.61,
    "Fe2+":   2.82,
    "Mn2+":   2.90,
    "Co2+":   2.72,
    "Ni2+":   2.63,
    # Trivalent
    "Fe3+":   2.79,
    "Al3+":   2.40,
    # Monovalent anions
    "F-":     2.95,
    "Cl-":    4.03,
    "Br-":    4.35,
    "I-":     4.99,
    # Organic ion groups (back-solved from Marcus-like experimental data)
    "COO-":   3.66,   # carboxylate (Marcus acetate ΔG=-375)
    "RNH3+":  4.03,   # protonated amine (scaled from NH4+ ΔG=-340)
    "imidazolium+": 5.34,  # protonated histidine (ΔG=-260, estimated)
    "guanidinium+": 4.90,  # arginine (ΔG=-280, estimated)
    "phenolate-":   3.66,  # deprotonated tyrosine (≈COO-)
    "phosphate-":   2.74,  # phosphate monoester (ΔG=-500, estimated)
    "sulfonate-":   4.03,  # sulfonate (≈Cl-)
}

# Experimental hydration energies from Marcus (kJ/mol)
MARCUS_DG_HYDR = {
    "Li+": -529, "Na+": -424, "K+": -352, "NH4+": -340,
    "Mg2+": -1931, "Ca2+": -1608, "Zn2+": -2046, "Fe3+": -4430,
    "F-": -465, "Cl-": -340, "Br-": -315, "I-": -275,
    "COO-": -375, "RNH3+": -340,
}


def born_solvation_energy(charge, radius_eff_A, epsilon_in=1.0, epsilon_out=78.4):
    """Born solvation energy in kJ/mol.

    ΔG = -(1389.4 × q²) × (1/ε_in - 1/ε_out) / R_eff

    Negative = favorable (stabilization by higher ε medium).
    """
    if radius_eff_A <= 0 or charge == 0:
        return 0.0
    return -BORN_FACTOR * charge**2 * (1.0/epsilon_in - 1.0/epsilon_out) / radius_eff_A


def born_desolvation_cost(ion_type, epsilon_pocket=8.0):
    """Cost of moving an ion from water into a protein pocket.

    ΔG_cost = ΔG_Born(ε_pocket) - ΔG_Born(ε_water)
    Always positive (it costs energy to bury a charge).

    Args:
        ion_type: key in MARCUS_BORN_RADII
        epsilon_pocket: effective dielectric of binding site

    Returns:
        ΔG_cost in kJ/mol (positive = unfavorable)
    """
    r_eff = MARCUS_BORN_RADII.get(ion_type)
    if r_eff is None:
        return 0.0

    # Determine charge from ion type
    _ION_CHARGES = {
        "Li+": 1, "Na+": 1, "K+": 1, "Rb+": 1, "Cs+": 1, "NH4+": 1,
        "Mg2+": 2, "Ca2+": 2, "Zn2+": 2, "Cu2+": 2, "Fe2+": 2, "Mn2+": 2,
        "Co2+": 2, "Ni2+": 2, "Fe3+": 3, "Al3+": 3,
        "F-": -1, "Cl-": -1, "Br-": -1, "I-": -1,
        "COO-": -1, "RNH3+": 1, "imidazolium+": 1, "guanidinium+": 1,
        "phenolate-": -1, "phosphate-": -1, "sulfonate-": -1,
        "phosphate2-": -2,
    }
    charge = _ION_CHARGES.get(ion_type, 0)

    if charge == 0:
        return 0.0

    # ΔG in water (favorable, negative)
    dG_water = born_solvation_energy(charge, r_eff, 1.0, EPSILON_WATER)
    # ΔG in pocket (less favorable, less negative)
    dG_pocket = born_solvation_energy(charge, r_eff, 1.0, epsilon_pocket)

    # Cost = ΔG_pocket - ΔG_water (positive because pocket is less stabilizing)
    return dG_pocket - dG_water


# ═══════════════════════════════════════════════════════════════════════════
# EFFECTIVE DIELECTRIC MODEL FOR PROTEIN POCKETS
# ═══════════════════════════════════════════════════════════════════════════

# The effective dielectric constant depends on burial depth.
# Surface-exposed: ε ≈ 40-60 (partially solvated)
# Shallow pocket: ε ≈ 10-20
# Deep burial: ε ≈ 4-8
# Source: Warshel 1984, Gilson & Honig 1986, Schutz & Warshel 2001

def effective_dielectric(burial_fraction):
    """Estimate effective dielectric from burial fraction.

    burial_fraction: 0 (surface) to 1 (fully buried)
    Returns: effective dielectric constant

    Model: ε = ε_deep + (ε_surface - ε_deep) × (1 - burial_fraction)²
    Quadratic because dielectric drops steeply once fully enclosed.
    """
    EPSILON_SURFACE = 40.0   # partially solvated surface
    EPSILON_DEEP = 4.0       # fully buried, low dielectric

    f = min(max(burial_fraction, 0.0), 1.0)
    return EPSILON_DEEP + (EPSILON_SURFACE - EPSILON_DEEP) * (1.0 - f)**2


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED SCORING (Tier 1 + Tier 2)
# ═══════════════════════════════════════════════════════════════════════════

def compute_electrostatic_desolvation(uc_or_groups, burial_fraction=0.6):
    """Compute total electrostatic desolvation cost.

    If uc_or_groups is a list: use Tier 1 (group-additive).
    If uc_or_groups has guest_charge: add Tier 2 Born for the charged group,
    MINUS Coulomb correction for counter-charges / salt bridges.

    Returns: ΔG_elec_desolv in kJ/mol (positive = unfavorable)
    """
    epsilon_pocket = effective_dielectric(burial_fraction)

    # Tier 1: group-additive
    if isinstance(uc_or_groups, list):
        return compute_group_elec_desolvation(uc_or_groups, epsilon_pocket)

    # Object with attributes (UniversalComplex-like)
    uc = uc_or_groups
    total = 0.0

    # Tier 1: polar groups on ligand (from buried_groups if available)
    buried_groups = getattr(uc, 'buried_groups', [])
    if buried_groups:
        total += compute_group_elec_desolvation(buried_groups, epsilon_pocket)

    # Tier 2: Born for formal charges
    guest_charge = getattr(uc, 'guest_charge', 0)
    if abs(guest_charge) >= 1:
        # Determine ion type from charge
        if guest_charge > 0:
            ion = "RNH3+"  # generic cation
        else:
            ion = "COO-"   # generic anion
        # Scale by burial
        born_cost = born_desolvation_cost(ion, epsilon_pocket)
        raw_born = born_cost * abs(guest_charge) * burial_fraction

        # Tier 3: Coulomb correction for counter-charges / salt bridges
        # When a charged ligand forms a salt bridge with a receptor
        # counter-charge, the Born desolvation penalty is partially
        # cancelled by the Coulomb attraction.
        #
        # IMPORTANT: The charge-assisted H-bond energy (Pace -17 kJ/mol)
        # already includes the NET electrostatic effect (desolvation +
        # Coulomb + H-bond). So we must NOT add Coulomb on top of
        # the H-bond scorer — instead we REDUCE the Born penalty to
        # avoid double-counting with the Pace value.
        #
        # Physics: each salt bridge "neutralizes" one unit of charge,
        # removing its Born penalty. Remaining uncompensated charges
        # still pay full Born cost.
        reduction = _compute_born_reduction(uc, epsilon_pocket)

        total += max(raw_born - reduction, 0.0)  # floor at 0

    return total


def _compute_born_reduction(uc, epsilon_pocket):
    """Compute Born penalty reduction from receptor-side counter-charges.

    Three mechanisms (checked in order):

    1. Explicit salt bridges: n_salt_bridges on UC → each neutralizes
       one unit of Born penalty for one formal charge
    2. Host charge: if host_charge has opposite sign to guest_charge,
       the receptor provides electrostatic compensation
    3. Charged H-bonds: if hbond_types includes charge-assisted entries,
       those charges are already accounted for in Pace energies

    Returns: reduction in kJ/mol (positive = amount to subtract from Born)
    """
    guest_q = abs(getattr(uc, 'guest_charge', 0))
    if guest_q == 0:
        return 0.0

    # ── Method 1: explicit salt bridges ──
    n_sb = getattr(uc, 'n_salt_bridges', 0)
    if n_sb > 0:
        # Each salt bridge neutralizes one charge unit's Born penalty
        # Fraction of charges compensated:
        f_compensated = min(n_sb / guest_q, 1.0)
        # Full Born cost for one charge unit
        ion = "RNH3+" if getattr(uc, 'guest_charge', 0) > 0 else "COO-"
        born_per_charge = born_desolvation_cost(ion, epsilon_pocket)
        return f_compensated * born_per_charge * guest_q

    # ── Method 2: host_charge with opposite sign ──
    host_q = getattr(uc, 'host_charge', 0)
    guest_sign = 1 if getattr(uc, 'guest_charge', 0) > 0 else -1
    if host_q != 0 and (host_q * guest_sign < 0):
        # Opposite charges → compensation
        compensated = min(abs(host_q), guest_q)
        f_compensated = compensated / guest_q
        ion = "RNH3+" if guest_sign > 0 else "COO-"
        born_per_charge = born_desolvation_cost(ion, epsilon_pocket)
        return f_compensated * born_per_charge * guest_q

    # ── Method 3: charged H-bonds in hbond_types ──
    hb_types = getattr(uc, 'hbond_types', [])
    if hb_types:
        n_charged_hb = sum(1 for t in hb_types
                           if "+" in t or "carboxylate" in t
                           or "charge" in t.lower())
        if n_charged_hb > 0:
            # Each charge-assisted HB implies a counter-charge is present
            # Pace value already accounts for desolvation → reduce Born
            f_compensated = min(n_charged_hb / (guest_q * 2), 1.0)
            # × 2 because each charge-assisted HB partially (not fully) compensates
            ion = "RNH3+" if guest_sign > 0 else "COO-"
            born_per_charge = born_desolvation_cost(ion, epsilon_pocket)
            return f_compensated * born_per_charge * guest_q

    return 0.0