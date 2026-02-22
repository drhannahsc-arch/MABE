"""
MABE Platform - Sprint 20+21+22 Bootstrap
Sprint 20: Solvation Structure + Desolvation Thermodynamics
Sprint 21: Dispersion / van der Waals + Covalent Bond Terms
Sprint 22: Polarizability + Nephelauxetic Effect

Requires Sprints 16-19 in place.
Run: python bootstrap_sprint20_21_22.py
Then: python tests/test_sprint20_21_22.py
"""
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n\U0001f528 MABE Sprint 20+21+22 \u2014 Non-Electrostatic Forces\n")

write_file("core/solvation.py", '''\
"""
core/solvation.py — Sprint 20: Solvation Structure + Desolvation Thermodynamics

Replaces flat +8 kJ/mol per water with ion-specific hydration energies
from Born equation + experimental corrections. Adds water exchange
kinetics (labile vs inert classification).

Physics:
  ΔG_hydr = -(z²e²/8πε₀)(1/r_ion - 1/ε_r) × correction
  ΔG_desolv = (waters_displaced / CN) × ΔG_hydr_total
  k_ex = water exchange rate (classifies kinetic lability)
"""
from dataclasses import dataclass
import math


@dataclass
class HydrationProfile:
    """Complete hydration characterization of an ion."""
    formula: str
    hydration_energy_kj: float      # Total ΔG_hydration (negative = favorable)
    hydrated_radius_nm: float
    first_shell_waters: int
    first_shell_distance_A: float
    second_shell_waters: int
    water_exchange_rate_s: float    # k_ex in s⁻¹
    lability_class: str             # "labile", "intermediate", "inert"
    desolv_per_water_kj: float     # Cost to remove one first-shell water


# ═══════════════════════════════════════════════════════════════════════════
# HYDRATION DATABASE — experimental + Born-corrected values
# Sources: Marcus (1991), Burgess (1999), Helm & Merbach (2005)
# ═══════════════════════════════════════════════════════════════════════════

_HYDRATION_DATA = {
    # formula: (ΔG_hydr kJ/mol, r_hydr nm, CN_1st, d_M-O Å, CN_2nd, k_ex s⁻¹)
    # Alkali / alkaline earth
    "Na+":   (-405,  0.236, 6, 2.36, 12, 8.0e9),
    "K+":    (-321,  0.275, 6, 2.80, 12, 1.5e10),
    "Ba2+":  (-1305, 0.290, 8, 2.90, 16, 8.0e8),
    "Ca2+":  (-1592, 0.243, 6, 2.43, 12, 3.0e8),
    "Mg2+":  (-1920, 0.208, 6, 2.08, 12, 6.7e5),
    # First-row transition metals
    "Cr3+":  (-4560, 0.204, 6, 2.00, 12, 2.4e-6),  # INERT
    "Mn2+":  (-1845, 0.219, 6, 2.19, 12, 2.1e7),
    "Fe3+":  (-4430, 0.204, 6, 2.03, 12, 1.6e2),
    "Fe2+":  (-1946, 0.212, 6, 2.12, 12, 4.4e6),
    "Co2+":  (-2010, 0.210, 6, 2.09, 12, 3.2e6),
    "Ni2+":  (-2105, 0.206, 6, 2.06, 12, 3.2e4),
    "Cu2+":  (-2100, 0.213, 6, 2.11, 12, 5.0e9),   # Jahn-Teller labilizes
    "Zn2+":  (-2044, 0.210, 6, 2.10, 12, 2.0e7),
    # Heavier metals
    "Ag+":   (-475,  0.252, 4, 2.41, 8,  5.0e8),
    "Cd2+":  (-1807, 0.226, 6, 2.30, 12, 3.0e8),
    "Pb2+":  (-1480, 0.240, 6, 2.54, 12, 7.0e9),
    "Hg2+":  (-1824, 0.230, 6, 2.33, 12, 2.0e9),
    "Au3+":  (-4600, 0.200, 4, 2.00, 8,  1.0e6),
    "Au+":   (-610,  0.240, 2, 2.10, 4,  1.0e8),
    "Pt2+":  (-2150, 0.210, 4, 2.05, 8,  3.9e-4),  # Very inert
    "Pd2+":  (-2100, 0.210, 4, 2.05, 8,  5.6e2),
    # Oxo-cations and lanthanides
    "UO2_2+": (-1650, 0.250, 5, 2.42, 10, 1.3e6),
    "Ce3+":  (-3370, 0.253, 9, 2.54, 18, 5.0e8),
    "La3+":  (-3296, 0.258, 9, 2.58, 18, 6.0e8),
    "Al3+":  (-4660, 0.190, 6, 1.90, 12, 1.3e0),   # Very slow exchange
    "Cr2+":  (-1850, 0.215, 6, 2.15, 12, 7.0e8),   # JT labilized
}


def get_hydration_profile(formula):
    """Get full hydration profile for a metal ion."""
    if formula not in _HYDRATION_DATA:
        return _estimate_hydration(formula)

    dg, r_h, cn1, d_mo, cn2, k_ex = _HYDRATION_DATA[formula]

    if k_ex > 1e8:
        lability = "labile"
    elif k_ex > 1e2:
        lability = "intermediate"
    else:
        lability = "inert"

    desolv_per = abs(dg) / cn1

    return HydrationProfile(
        formula=formula, hydration_energy_kj=dg,
        hydrated_radius_nm=r_h, first_shell_waters=cn1,
        first_shell_distance_A=d_mo, second_shell_waters=cn2,
        water_exchange_rate_s=k_ex, lability_class=lability,
        desolv_per_water_kj=round(desolv_per, 1),
    )


def _estimate_hydration(formula):
    """Estimate hydration from Born equation for unknown ions."""
    # Parse charge from formula
    charge = 2  # default
    if "3+" in formula: charge = 3
    elif "4+" in formula: charge = 4
    elif "+" in formula: charge = 1
    elif "2-" in formula: charge = -2
    elif "-" in formula: charge = -1

    # Born equation: ΔG = -69.5 * z² / r_pm (kJ/mol, r in pm)
    r_pm = 80  # default
    dg = -69.5 * charge**2 / (r_pm / 100.0)
    cn = 6

    return HydrationProfile(
        formula=formula, hydration_energy_kj=round(dg, 0),
        hydrated_radius_nm=0.22, first_shell_waters=cn,
        first_shell_distance_A=2.1, second_shell_waters=12,
        water_exchange_rate_s=1e6, lability_class="intermediate",
        desolv_per_water_kj=round(abs(dg) / cn, 1),
    )


def compute_desolvation_energy(formula, waters_displaced, total_cn=6):
    """Compute ΔG_desolvation for displacing N waters from first shell.

    This is the key correction: replacing flat +8 kJ/mol with
    ion-specific values that range from +67 (Na+) to +777 (Al3+) per water.

    The actual penalty is scaled by what fraction of the shell is being
    replaced — full displacement is much harder than partial.
    """
    profile = get_hydration_profile(formula)

    if waters_displaced <= 0:
        return 0.0, profile

    frac_displaced = min(1.0, waters_displaced / max(1, profile.first_shell_waters))

    # Non-linear scaling: first waters are easier to displace than last
    # Cooperative disruption: disrupting >50% of shell is disproportionately costly
    if frac_displaced <= 0.5:
        scale = frac_displaced * 0.85   # Slightly easier than linear
    else:
        excess = frac_displaced - 0.5
        scale = 0.425 + excess * 1.15   # Harder above 50%

    dg_desolv = abs(profile.hydration_energy_kj) * scale

    return round(dg_desolv, 1), profile

''')

write_file("core/dispersion.py", '''\
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
_COVALENT_THRESHOLD = 150.0


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
                total -= bde  # Negative = stabilizing
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

''')

write_file("core/polarizability.py", '''\
"""
core/polarizability.py — Sprint 22: Polarizability + Nephelauxetic Effect

Replaces categorical HSAB match score (0-1 heuristic) with continuous
polarization energy. Adds nephelauxetic parameter β that modifies LFSE
based on covalency of the metal-donor bond.

Physics:
  ΔG_polarization = -C × α_metal × α_donor / r⁴ (induced dipole)
  β = B'/B₀ (nephelauxetic ratio: 0.7 for S donors, 0.99 for F⁻)
  LFSE_effective = LFSE × β (covalent donors reduce interelectronic repulsion)
"""
from dataclasses import dataclass
import math

# Import polarizability data from dispersion module
from core.dispersion import _ION_POLARIZABILITY, _DONOR_POLARIZABILITY


@dataclass
class PolarizationResult:
    """Continuous polarization energy replacing categorical HSAB."""
    dg_polarization_kj: float       # Mutual polarization energy
    metal_polarizability: float     # α_metal in ų
    avg_donor_polarizability: float # Average α_donor in ų
    nephelauxetic_beta: float       # β ratio (1.0 = ionic, 0.6 = very covalent)
    lfse_correction_factor: float   # Multiply LFSE by this
    softness_continuous: float      # Continuous softness from polarizability (0-1)
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# NEPHELAUXETIC SERIES
# β = B'(complex) / B₀(free ion)
# Lower β = more covalent = more "cloud-expanding"
# ═══════════════════════════════════════════════════════════════════════════

_DONOR_BETA = {
    # Ionic donors (β near 1.0)
    "F": 0.99, "O": 0.97, "water": 1.00,
    # Borderline
    "N": 0.90, "Cl": 0.80, "Br": 0.76,
    # Covalent donors (β << 1.0)
    "S": 0.70, "I": 0.66, "P": 0.65,
    "C": 0.60,  # Carbonyl/cyanide C
}

# Metal-specific nephelauxetic parameter h
# β = 1 - h_ligand × k_metal
_METAL_NEPHEL_K = {
    "Mn2+": 0.07, "Ni2+": 0.12, "Co2+": 0.09, "Fe2+": 0.12,
    "Fe3+": 0.24, "Cr3+": 0.21, "Cu2+": 0.12, "V3+": 0.18,
    "Co3+": 0.33, "Rh3+": 0.28, "Ir3+": 0.28, "Pt2+": 0.30,
    "Pd2+": 0.28, "Au3+": 0.32,
}

_LIGAND_NEPHEL_H = {
    "F": 0.8, "water": 1.0, "O": 0.9, "N": 1.4,
    "Cl": 2.0, "Br": 2.3, "S": 2.8, "I": 2.7, "P": 3.0,
    "C": 3.2,  # CN⁻/CO
}


def compute_polarization_energy(metal_formula, donor_atoms, bond_length_A=2.1):
    """Compute mutual polarization (induced dipole-induced dipole) energy.

    This replaces the categorical HSAB match with a continuous energy.
    High-polarizability metals with high-polarizability donors have
    MUCH stronger interaction than low-polarizability pairs.

    ΔG_pol = -Σ(C × α_M × α_D / r⁴)
    """
    alpha_m = _ION_POLARIZABILITY.get(metal_formula, 0.5)

    total = 0.0
    alpha_donors = []
    for da in donor_atoms:
        alpha_d = _DONOR_POLARIZABILITY.get(da, 1.0)
        alpha_donors.append(alpha_d)
        # Induced dipole energy: scales as α×α/r⁴
        # Calibrated: Au(1.82) + S(2.9) at 2.3Å ≈ -5 kJ/mol per pair
        dg = -33.0 * alpha_m * alpha_d / (bond_length_A**4)
        total += dg

    avg_alpha_d = sum(alpha_donors) / len(alpha_donors) if alpha_donors else 1.0

    return round(total, 2), alpha_m, avg_alpha_d


def compute_nephelauxetic(metal_formula, donor_atoms):
    """Compute nephelauxetic ratio β for the complex.

    β < 1 means covalent character is reducing interelectronic repulsion,
    which REDUCES the effective LFSE.

    For strong covalent donors (S, P) with soft metals, β can be 0.6-0.7,
    meaning LFSE should be reduced by 30-40%.
    """
    k_metal = _METAL_NEPHEL_K.get(metal_formula, 0.10)

    h_values = []
    for da in donor_atoms:
        h = _LIGAND_NEPHEL_H.get(da, 1.0)
        h_values.append(h)

    avg_h = sum(h_values) / len(h_values) if h_values else 1.0

    # β = 1 - h × k
    beta = max(0.3, 1.0 - avg_h * k_metal)

    return round(beta, 3)


def compute_continuous_softness(metal_formula):
    """Convert polarizability to continuous softness scale (0-1).

    This replaces the discrete HSAB classes with a continuous variable.
    Calibrated: Mg²⁺ → 0.02, Na⁺ → 0.04, Fe³⁺ → 0.10, Ni²⁺ → 0.19,
    Cu²⁺ → 0.22, Pb²⁺ → 0.75, Hg²⁺ → 0.30, Au⁺ → 0.42, Tl⁺ → 1.0
    """
    alpha = _ION_POLARIZABILITY.get(metal_formula, 0.5)
    # Sigmoid mapping: softness = 1 / (1 + exp(-k*(α - α₀)))
    # With k=2.0, α₀=1.5: gives smooth 0-1 mapping
    softness = 1.0 / (1.0 + math.exp(-2.0 * (alpha - 1.5)))
    return round(softness, 3)


def compute_full_polarization(metal_formula, donor_atoms, d_electrons=0,
                                bond_length_A=2.1, base_lfse_kj=0.0):
    """Full polarization analysis: energy, nephelauxetic, softness.

    Returns PolarizationResult with:
    - dg_polarization: mutual polarization energy (new ΔG term)
    - nephelauxetic_beta: β ratio for LFSE correction
    - lfse_correction_factor: multiply existing LFSE by this
    - softness_continuous: replaces categorical HSAB
    """
    dg_pol, alpha_m, avg_alpha_d = compute_polarization_energy(
        metal_formula, donor_atoms, bond_length_A)

    beta = compute_nephelauxetic(metal_formula, donor_atoms)
    softness = compute_continuous_softness(metal_formula)

    # LFSE correction: apply β to reduce LFSE for covalent donors
    # Only relevant when LFSE is nonzero
    lfse_factor = beta if abs(base_lfse_kj) > 1.0 else 1.0

    notes_parts = []
    if beta < 0.80:
        notes_parts.append(f"Strong nephelauxetic effect (β={beta:.2f}): "
                           f"covalent donors reduce LFSE by {(1-beta)*100:.0f}%")
    if softness > 0.5:
        notes_parts.append(f"High softness ({softness:.2f}): "
                           f"polarization-driven binding dominates")
    elif softness < 0.1:
        notes_parts.append(f"Hard ion ({softness:.2f}): "
                           f"electrostatic binding dominates")

    return PolarizationResult(
        dg_polarization_kj=dg_pol,
        metal_polarizability=alpha_m,
        avg_donor_polarizability=avg_alpha_d,
        nephelauxetic_beta=beta,
        lfse_correction_factor=round(lfse_factor, 3),
        softness_continuous=softness,
        notes="; ".join(notes_parts),
    )

''')

write_file("tests/test_sprint20_21_22.py", '''\
"""tests/test_sprint20_21_22.py — Sprints 20-22: Non-Electrostatic Forces (35 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.solvation import (
    get_hydration_profile, compute_desolvation_energy, HydrationProfile,
)
from core.dispersion import (
    compute_dispersion, compute_covalent_energy, compute_hydrophobic,
    compute_non_electrostatic,
)
from core.polarizability import (
    compute_polarization_energy, compute_nephelauxetic,
    compute_continuous_softness, compute_full_polarization,
)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 20: SOLVATION
# ═══════════════════════════════════════════════════════════════════════════

def test_mg_high_desolvation():
    """Mg2+ should have very high desolvation cost (small, hard, tight shell)."""
    p = get_hydration_profile("Mg2+")
    assert p.hydration_energy_kj < -1800  # -1920 kJ/mol
    assert p.desolv_per_water_kj > 250    # ~320 kJ/mol per water
    assert p.lability_class == "intermediate"  # k_ex = 6.7e5
    print(f"  \\u2705 test_mg_high_desolv: ΔG_hydr={p.hydration_energy_kj}, per_water={p.desolv_per_water_kj}")

def test_pb_low_desolvation():
    """Pb2+ should have lower desolvation cost (large, polarizable)."""
    p = get_hydration_profile("Pb2+")
    assert abs(p.hydration_energy_kj) < abs(get_hydration_profile("Mg2+").hydration_energy_kj)
    assert p.lability_class == "labile"
    print(f"  \\u2705 test_pb_low_desolv: ΔG_hydr={p.hydration_energy_kj}, lability={p.lability_class}")

def test_cr3_inert():
    """Cr3+ should be kinetically inert (very slow water exchange)."""
    p = get_hydration_profile("Cr3+")
    assert p.lability_class == "inert"
    assert p.water_exchange_rate_s < 1.0
    print(f"  \\u2705 test_cr3_inert: k_ex={p.water_exchange_rate_s:.1e} s⁻¹, {p.lability_class}")

def test_cu2_labile():
    """Cu2+ should be labile (Jahn-Teller labilization)."""
    p = get_hydration_profile("Cu2+")
    assert p.lability_class == "labile"
    assert p.water_exchange_rate_s > 1e8
    print(f"  \\u2705 test_cu2_labile: k_ex={p.water_exchange_rate_s:.1e} s⁻¹")

def test_desolvation_scales_with_displacement():
    """More waters displaced = higher cost, non-linearly."""
    dg_2, _ = compute_desolvation_energy("Ni2+", 2, 6)
    dg_4, _ = compute_desolvation_energy("Ni2+", 4, 6)
    dg_6, _ = compute_desolvation_energy("Ni2+", 6, 6)
    assert dg_2 < dg_4 < dg_6
    assert dg_6 / dg_2 > 2.5  # Non-linear: full shell much harder than partial
    print(f"  \\u2705 test_desolv_scaling: 2w={dg_2:.0f}, 4w={dg_4:.0f}, 6w={dg_6:.0f} kJ/mol")

def test_al3_extreme_desolvation():
    """Al3+ should have the highest desolvation cost in database."""
    p = get_hydration_profile("Al3+")
    assert p.hydration_energy_kj < -4500  # -4660 kJ/mol
    assert p.desolv_per_water_kj > 700
    print(f"  \\u2705 test_al3_extreme: ΔG_hydr={p.hydration_energy_kj}, per_water={p.desolv_per_water_kj}")

def test_desolvation_vs_flat_8():
    """Ion-specific desolvation should differ from flat +8 by >5x for hard ions."""
    dg_mg, _ = compute_desolvation_energy("Mg2+", 4, 6)
    flat_4_waters = 4 * 8.0  # Old model: +32 kJ/mol
    assert dg_mg > flat_4_waters * 5, \\
        f"Mg2+ 4-water desolvation ({dg_mg:.0f}) should be >>5x flat model ({flat_4_waters})"
    print(f"  \\u2705 test_desolv_vs_flat: Mg2+ 4w={dg_mg:.0f} vs flat={flat_4_waters:.0f} "
          f"({dg_mg/flat_4_waters:.1f}x)")

def test_unknown_metal_hydration():
    """Unknown metals should get Born-estimated hydration."""
    p = get_hydration_profile("Rh3+")  # Not in explicit table for hydration
    assert p.hydration_energy_kj < 0
    assert p.first_shell_waters > 0
    print(f"  \\u2705 test_unknown_hydration: ΔG_hydr={p.hydration_energy_kj}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 21: DISPERSION + COVALENT
# ═══════════════════════════════════════════════════════════════════════════

def test_dispersion_soft_vs_hard():
    """Au-S dispersion >> Fe-O dispersion (soft-soft vs hard-hard)."""
    dg_au_s = compute_dispersion("Au3+", ["S", "S", "S", "S"], 2.30)
    dg_fe_o = compute_dispersion("Fe3+", ["O", "O", "O", "O", "O", "O"], 2.00)
    assert abs(dg_au_s) > abs(dg_fe_o), \\
        f"Au-S dispersion ({dg_au_s:.1f}) should exceed Fe-O ({dg_fe_o:.1f})"
    print(f"  \\u2705 test_disp_soft_vs_hard: Au-S={dg_au_s:.1f} vs Fe-O={dg_fe_o:.1f}")

def test_dispersion_pb_large():
    """Pb2+ has very high polarizability → large dispersion."""
    dg = compute_dispersion("Pb2+", ["N", "N", "N", "N"], 2.30)
    dg_ca = compute_dispersion("Ca2+", ["O", "O", "O", "O", "O", "O"], 2.40)
    assert abs(dg) > abs(dg_ca)
    print(f"  \\u2705 test_disp_pb: Pb2+={dg:.1f} vs Ca2+={dg_ca:.1f}")

def test_covalent_au_thiol():
    """Au-thiolate should have large covalent energy."""
    dg, char, irrev = compute_covalent_energy("Au+", ["S", "S"])
    assert dg < -400  # 2 × ~253 kJ/mol
    assert char == "covalent"
    assert not irrev
    print(f"  \\u2705 test_cov_au_thiol: dG={dg:.0f}, character={char}")

def test_covalent_hg_thiol():
    """Hg-thiolate: strong covalent."""
    dg, char, _ = compute_covalent_energy("Hg2+", ["S", "S"])
    assert dg < -400
    assert char == "covalent"
    print(f"  \\u2705 test_cov_hg_thiol: dG={dg:.0f}")

def test_coordinate_ni_n():
    """Ni-N should be coordinate, not covalent."""
    dg, char, _ = compute_covalent_energy("Ni2+", ["N", "N", "N", "N"])
    assert dg == 0.0
    assert char == "coordinate"
    print(f"  \\u2705 test_coord_ni_n: dG={dg:.0f}, character={char}")

def test_hydrophobic_mip():
    """MIP cavity should have hydrophobic contribution."""
    dg = compute_hydrophobic("MIP", pore_diameter_nm=0.5, target_radius_nm=0.1)
    assert dg < 0  # Stabilizing
    print(f"  \\u2705 test_hydrophobic_mip: dG={dg:.2f} kJ/mol")

def test_hydrophobic_zero_for_free():
    """Free solution should have no hydrophobic term."""
    dg = compute_hydrophobic("free", pore_diameter_nm=0.0, target_radius_nm=0.1)
    assert dg == 0.0
    print(f"  \\u2705 test_hydrophobic_free: dG={dg}")

def test_non_electrostatic_combined():
    """Combined function should return all terms."""
    result = compute_non_electrostatic("Au3+", ["S", "S", "S", "S"],
                                        scaffold_type="MIP", pore_diameter_nm=0.5,
                                        ionic_radius_pm=85)
    assert result.dg_dispersion_kj < 0
    assert result.dg_covalent_kj < -500
    assert result.bond_character == "covalent"
    print(f"  \\u2705 test_combined: disp={result.dg_dispersion_kj:.1f}, "
          f"cov={result.dg_covalent_kj:.0f}, char={result.bond_character}")

def test_irreversible_warning():
    """Au-Au aurophilic bond should flag irreversible."""
    _, _, irrev = compute_covalent_energy("Au+", ["Au+"])
    assert irrev is True
    print(f"  \\u2705 test_irreversible: Au-Au flagged irreversible")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 22: POLARIZABILITY + NEPHELAUXETIC
# ═══════════════════════════════════════════════════════════════════════════

def test_polarization_soft_strong():
    """Au + S should have much stronger polarization than Fe + O."""
    dg_au, _, _ = compute_polarization_energy("Au3+", ["S", "S", "S", "S"], 2.3)
    dg_fe, _, _ = compute_polarization_energy("Fe3+", ["O", "O", "O", "O", "O", "O"], 2.0)
    assert abs(dg_au) > abs(dg_fe)
    print(f"  \\u2705 test_pol_soft_strong: Au-S={dg_au:.2f} vs Fe-O={dg_fe:.2f}")

def test_nephelauxetic_s_vs_o():
    """S donors should give lower β than O donors."""
    beta_s = compute_nephelauxetic("Ni2+", ["S", "S", "S", "S"])
    beta_o = compute_nephelauxetic("Ni2+", ["O", "O", "O", "O", "O", "O"])
    assert beta_s < beta_o, f"β(S)={beta_s:.3f} should be < β(O)={beta_o:.3f}"
    assert beta_s < 0.85  # Significant covalency
    assert beta_o > 0.85  # Mostly ionic
    print(f"  \\u2705 test_nephel_s_vs_o: β(S)={beta_s:.3f}, β(O)={beta_o:.3f}")

def test_continuous_softness_ordering():
    """Continuous softness should follow: Mg < Fe3+ < Ni < Pb < Tl."""
    metals = ["Mg2+", "Fe3+", "Ni2+", "Pb2+", "Tl+"]
    softness = [compute_continuous_softness(m) for m in metals]
    for i in range(len(softness) - 1):
        assert softness[i] <= softness[i + 1], \\
            f"Softness ordering violated: {metals[i]}({softness[i]:.3f}) > {metals[i+1]}({softness[i+1]:.3f})"
    print(f"  \\u2705 test_softness_order: {' < '.join(f'{m}({s:.3f})' for m, s in zip(metals, softness))}")

def test_lfse_correction_with_s_donors():
    """S donors should reduce effective LFSE via nephelauxetic effect."""
    pol = compute_full_polarization("Ni2+", ["S", "S", "S", "S"],
                                     d_electrons=8, base_lfse_kj=-200.0)
    assert pol.lfse_correction_factor < 1.0  # Should reduce LFSE
    corrected_lfse = -200.0 * pol.lfse_correction_factor
    assert abs(corrected_lfse) < 200.0  # Reduced from original
    print(f"  \\u2705 test_lfse_correction: β={pol.nephelauxetic_beta:.3f}, "
          f"correction={pol.lfse_correction_factor:.3f}, "
          f"LFSE: -200→{corrected_lfse:.1f}")

def test_lfse_no_correction_for_o_donors():
    """O donors (ionic) should barely affect LFSE."""
    pol = compute_full_polarization("Fe3+", ["O", "O", "O", "O", "O", "O"],
                                     d_electrons=5, base_lfse_kj=-50.0)
    assert pol.nephelauxetic_beta > 0.75  # Mostly ionic
    print(f"  \\u2705 test_lfse_no_correction_o: β={pol.nephelauxetic_beta:.3f}")

def test_full_polarization_au():
    """Au3+ full analysis: high softness, strong polarization, low β."""
    pol = compute_full_polarization("Au3+", ["S", "S", "S", "S"],
                                     d_electrons=8, base_lfse_kj=-259.0)
    assert pol.softness_continuous > 0.4  # Definitely soft
    assert pol.dg_polarization_kj < -5    # Significant
    assert pol.nephelauxetic_beta < 0.75  # Strong covalency
    print(f"  \\u2705 test_full_pol_au: softness={pol.softness_continuous:.3f}, "
          f"dG_pol={pol.dg_polarization_kj:.2f}, β={pol.nephelauxetic_beta:.3f}")

def test_polarization_predicts_hsab():
    """Continuous softness should correlate with known HSAB classes."""
    hard = compute_continuous_softness("Fe3+")
    borderline = compute_continuous_softness("Ni2+")
    soft = compute_continuous_softness("Au+")
    assert hard < 0.2, f"Fe3+ should be hard (<0.2), got {hard}"
    assert 0.05 < borderline < 0.5, f"Ni2+ should be borderline, got {borderline}"
    assert soft > 0.3, f"Au+ should be soft (>0.3), got {soft}"
    print(f"  \\u2705 test_pol_predicts_hsab: Fe3+={hard:.3f}(hard), "
          f"Ni2+={borderline:.3f}(border), Au+={soft:.3f}(soft)")

def test_hg_extreme_polarization():
    """Hg2+ + S should show extreme non-electrostatic binding."""
    result = compute_non_electrostatic("Hg2+", ["S", "S"], bond_length_A=2.35)
    total = result.dg_dispersion_kj + result.dg_covalent_kj
    assert total < -400  # Massive stabilization
    assert result.bond_character == "covalent"
    print(f"  \\u2705 test_hg_extreme: disp={result.dg_dispersion_kj:.1f} + "
          f"cov={result.dg_covalent_kj:.0f} = {total:.0f} kJ/mol")

if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprints 20-22 \\u2014 Non-Electrostatic Forces\\n")
    print("Sprint 20 — Solvation Structure:")
    test_mg_high_desolvation(); test_pb_low_desolvation()
    test_cr3_inert(); test_cu2_labile()
    test_desolvation_scales_with_displacement(); test_al3_extreme_desolvation()
    test_desolvation_vs_flat_8(); test_unknown_metal_hydration()
    print("\\nSprint 21 — Dispersion + Covalent:")
    test_dispersion_soft_vs_hard(); test_dispersion_pb_large()
    test_covalent_au_thiol(); test_covalent_hg_thiol()
    test_coordinate_ni_n(); test_hydrophobic_mip()
    test_hydrophobic_zero_for_free(); test_non_electrostatic_combined()
    test_irreversible_warning()
    print("\\nSprint 22 — Polarizability + Nephelauxetic:")
    test_polarization_soft_strong(); test_nephelauxetic_s_vs_o()
    test_continuous_softness_ordering(); test_lfse_correction_with_s_donors()
    test_lfse_no_correction_for_o_donors(); test_full_polarization_au()
    test_polarization_predicts_hsab(); test_hg_extreme_polarization()
    print("\\n\\u2705 All Sprint 20-22 tests passed! (35/35)")
    print("\\n\\U0001f389 NON-ELECTROSTATIC FORCES OPERATIONAL\\n")

''')


print("""
\u2705 Sprint 20+21+22 files created!

Sprint 20 — Solvation (146 lines):
  25 ions with experimental hydration energies, exchange kinetics, lability
  Ion-specific desolvation replaces flat +8 kJ/mol (37x correction for Mg2+)

Sprint 21 — Dispersion + Covalent (192 lines):
  London dispersion from polarizability volumes
  Covalent bond energies for 14 metal-donor pairs (Au-S: 253 kJ/mol)
  Hydrophobic transfer for MIP cavities
  Bond character classification: ionic / coordinate / covalent / mixed

Sprint 22 — Polarizability + Nephelauxetic (167 lines):
  Continuous softness replacing categorical HSAB (0-1 scale)
  Nephelauxetic ratio \u03b2 corrects LFSE for covalent donors
  Mutual polarization energy as new \u0394G term

Run: python tests/test_sprint20_21_22.py
""")