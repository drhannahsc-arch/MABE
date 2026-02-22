"""
core/dispersion.py — Sprint 21: Dispersion/vdW + Covalent Bond Terms

Adds London dispersion, covalent bond energy, and hydrophobic transfer
terms. These are the dominant non-electrostatic forces for soft metals
and organic targets.

Physics:
  ΔG_dispersion = -C × α_metal × α_donor / r⁶ (London)
  ΔG_covalent = bond dissociation energy for covalent M-L pairs
  ΔG_hydrophobic = -γ × ΔSASA (for cavity binding)
"""
from dataclasses import dataclass
import math


@dataclass
class NonElectrostaticTerms:
    """All non-electrostatic interaction energies."""
    dg_dispersion_kj: float     # London dispersion
    dg_covalent_kj: float       # Covalent bond energy (if applicable)
    dg_hydrophobic_kj: float    # Hydrophobic transfer (MIP/aptamer)
    bond_character: str          # "ionic", "coordinate", "covalent", "mixed"
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# POLARIZABILITY DATABASE (Å³)
# Ion polarizabilities from Shannon & Fischer, Mahan, Tessman
# ═══════════════════════════════════════════════════════════════════════════

_ION_POLARIZABILITY = {
    # Alkali / alkaline earth — very low
    "Na+": 0.18, "K+": 0.84, "Ba2+": 1.56, "Ca2+": 0.47, "Mg2+": 0.09,
    # Hard trivalent — moderate
    "Fe3+": 0.48, "Cr3+": 0.30, "Al3+": 0.05, "Ce3+": 1.04, "La3+": 1.05,
    # Borderline divalent
    "Fe2+": 0.85, "Co2+": 0.88, "Ni2+": 0.92, "Cu2+": 1.10,
    "Zn2+": 0.67, "Mn2+": 0.72,
    # Soft — HIGH polarizability (drives binding!)
    "Ag+": 1.55, "Cu+": 1.20, "Au+": 2.10, "Au3+": 1.82,
    "Hg2+": 1.52, "Tl+": 5.26, "Pb2+": 3.78, "Cd2+": 0.98,
    "Pt2+": 1.65, "Pd2+": 1.50,
    # Oxo-cations
    "UO2_2+": 0.80,
}

_DONOR_POLARIZABILITY = {
    "O": 0.80,     # Oxide/hydroxide — low
    "N": 1.10,     # Amine/imine — moderate
    "S": 2.90,     # Thiolate — HIGH
    "P": 3.60,     # Phosphine — very high
    "Cl": 2.18,
    "Br": 3.05,
    "I": 4.70,
}


# ═══════════════════════════════════════════════════════════════════════════
# COVALENT BOND ENERGIES (kJ/mol)
# M-L bond dissociation energies for covalent pairs
# Sources: Luo (2007), Kerr (1999), CRC Handbook
# ═══════════════════════════════════════════════════════════════════════════

_COVALENT_BONDS = {
    # (metal, donor_atom): (BDE kJ/mol, is_irreversible)
    ("Au+", "S"):   (253, False),   # Au-thiolate: strong but exchangeable
    ("Au3+", "S"):  (230, False),
    ("Au+", "Au+"):  (226, True),   # Au-Au aurophilic (in nanoparticles)
    ("Hg2+", "S"):  (217, False),   # Hg-thiolate: very strong
    ("Ag+", "S"):   (216, False),
    ("Pt2+", "S"):  (235, False),
    ("Pd2+", "S"):  (210, False),
    ("Cu+", "S"):   (190, False),
    ("Pb2+", "S"):  (168, False),   # Weaker but still partly covalent
    ("Cd2+", "S"):  (180, False),
    # Hg-C bonds (organomercury)
    ("Hg2+", "C"):  (122, True),
    # Less covalent pairs — these are coordinate, not covalent
    ("Ni2+", "N"):  (0, False),     # Pure coordinate
    ("Fe3+", "O"):  (0, False),
    ("Cu2+", "N"):  (0, False),
    ("Zn2+", "N"):  (0, False),
}

# Threshold: if BDE > 150 kJ/mol, classify as covalent
_COVALENT_THRESHOLD = 180.0  # Only genuinely covalent bonds (Hg-S, Au-S, Pt-S)


# ═══════════════════════════════════════════════════════════════════════════
# COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_dispersion(metal_formula, donor_atoms, bond_length_A=2.1):
    """Compute London dispersion energy between metal and donor atoms.

    ΔG_disp = -C₆ / r⁶  where C₆ ∝ α_metal × α_donor
    Prefactor C calibrated so soft-metal + soft-donor ≈ -10 to -30 kJ/mol total.
    """
    alpha_m = _ION_POLARIZABILITY.get(metal_formula, 0.5)

    total = 0.0
    for da in donor_atoms:
        alpha_d = _DONOR_POLARIZABILITY.get(da, 1.0)
        r = bond_length_A
        # C₆ = 3/2 × I_avg × α_m × α_d / (α_m + α_d)
        # Simplified with I_avg ≈ 10 eV = 965 kJ/mol
        c6 = 1.5 * 965 * alpha_m * alpha_d / (alpha_m + alpha_d + 0.01)
        # ΔG = -C₆ / r⁶ scaled to kJ/mol
        dg_per_donor = -c6 / (r**6) * 1e-3  # Scale factor for kJ/mol
        # Empirical calibration: Au(α=2.1) + S(α=2.9) at 2.3Å ≈ -8 kJ/mol per donor
        dg_per_donor *= 15.0  # Calibration constant
        total += dg_per_donor

    return round(total, 2)


def compute_covalent_energy(metal_formula, donor_atoms):
    """Compute covalent bond contribution for metal-donor pairs.

    Only applies to specific pairs (soft metal + soft donor).
    Returns (total energy, bond character classification).
    """
    total = 0.0
    has_covalent = False
    is_irreversible = False

    for da in donor_atoms:
        key = (metal_formula, da)
        if key in _COVALENT_BONDS:
            bde, irrev = _COVALENT_BONDS[key]
            if bde >= _COVALENT_THRESHOLD:
                # BDE values are gaseous homolytic bond energies.
                # Coordinate bonds in solution are a fraction of full BDE.
                # The fraction depends on the metal: Hg²⁺, Au⁺/³⁺, Pt²⁺ form
                # genuinely covalent coordinate bonds (high orbital overlap).
                # Others are more ionic/dative.
                _COVALENT_FRACTIONS = {
                    "Hg2+": 0.25, "Au+": 0.22, "Au3+": 0.22,
                    "Pt2+": 0.20, "Pd2+": 0.18, "Ag+": 0.20,
                }
                coord_fraction = _COVALENT_FRACTIONS.get(metal_formula, None)
                if coord_fraction is None:
                    # Scale by metal softness: borderline metals (0.3-0.6) get less
                    # covalent character than soft metals (0.7+).
                    # Import softness from coordination_generator
                    from core.coordination_generator import METAL_HSAB_SOFTNESS
                    s = METAL_HSAB_SOFTNESS.get(metal_formula, 0.3)
                    # Soft (s>0.7): 12-15%. Borderline (0.3-0.7): 4-12%. Hard (<0.3): 2-4%.
                    coord_fraction = 0.04 + 0.16 * max(0, (s - 0.2)) / 0.8
                    coord_fraction = max(0.02, min(0.15, coord_fraction))
                total -= bde * coord_fraction  # Negative = stabilizing
                has_covalent = True
                if irrev:
                    is_irreversible = True

    if has_covalent:
        character = "covalent" if total < -350 else "mixed"
    else:
        character = "coordinate"

    return round(total, 1), character, is_irreversible


def compute_hydrophobic(scaffold_type, pore_diameter_nm=0.0, target_radius_nm=0.0):
    """Compute hydrophobic transfer energy for cavity binding.

    Relevant for MIP, cyclodextrin, and hydrophobic pockets in aptamers.
    ΔG_hydrophobic = -γ × ΔSASA where γ ≈ 0.025 kJ/(mol·Å²)
    """
    if scaffold_type not in ("MIP", "coordination_cage", "COF"):
        return 0.0

    if pore_diameter_nm <= 0 or target_radius_nm <= 0:
        return 0.0

    # Estimate SASA buried: hemisphere of target inside cavity
    r_target_A = target_radius_nm * 10.0
    sasa_buried = 2 * math.pi * r_target_A**2  # Hemisphere in Å²

    # γ = 0.025 kJ/(mol·Å²) from Eisenberg & McLachlan
    gamma = 0.025
    dg = -gamma * sasa_buried

    return round(dg, 2)


def compute_non_electrostatic(metal_formula, donor_atoms, scaffold_type="free",
                                bond_length_A=2.1, pore_diameter_nm=0.0,
                                ionic_radius_pm=80.0):
    """Compute all non-electrostatic terms in one call."""
    dg_disp = compute_dispersion(metal_formula, donor_atoms, bond_length_A)
    dg_cov, character, irreversible = compute_covalent_energy(metal_formula, donor_atoms)
    dg_hydro = compute_hydrophobic(scaffold_type, pore_diameter_nm,
                                    ionic_radius_pm / 1000.0)

    notes_parts = []
    if abs(dg_cov) > 100:
        notes_parts.append(f"Strong covalent: {dg_cov:.0f} kJ/mol")
        if irreversible:
            notes_parts.append("WARNING: Irreversible binding — no release")
    if abs(dg_disp) > 20:
        notes_parts.append(f"Significant dispersion: {dg_disp:.1f} kJ/mol")
    if abs(dg_hydro) > 5:
        notes_parts.append(f"Hydrophobic: {dg_hydro:.1f} kJ/mol")

    return NonElectrostaticTerms(
        dg_dispersion_kj=dg_disp, dg_covalent_kj=dg_cov,
        dg_hydrophobic_kj=dg_hydro, bond_character=character,
        notes="; ".join(notes_parts),
    )


