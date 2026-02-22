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

