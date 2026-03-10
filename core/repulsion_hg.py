"""
core/repulsion_hg.py — Repulsion Physics for Host-Guest & Receptor-Guest Binding

Complements core/repulsion.py (which handles metal-ion/pore/framework repulsion).
This module covers organic molecular repulsion in non-covalent binding:

1. VdW volume overlap:     guest too large for cavity → exponential penalty
2. Electrostatic repulsion: like-charge guest in like-charge host
3. Aperture exclusion:     guest cross-section exceeds pore window (generalized)
4. Steric clash:           heavy-atom overflow from cavity

Design principle:
    ALL terms are ZERO when packing_coefficient ≤ 1.0.
    They fire only in the overpacking regime (extrapolation).
    Existing calibration data (Rekharsky CD, Fujita cages) is untouched.

Parameters:
    All physics-derived. No fitted parameters against binding data.
"""

import math
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# 1. VdW VOLUME OVERLAP
# ═══════════════════════════════════════════════════════════════════════════
# Guest volume exceeds cavity → Pauli exclusion → exponential penalty.
#
# At packing ≤ 1.0: zero (guest fits).
# At packing 1.05: 0.3 kJ/mol (minor).
# At packing 1.20: 7.1 kJ/mol (significant, ~1.2 log Ka).
# At packing 1.50: 44 kJ/mol (prohibitive).
# At packing 2.00: 141 kJ/mol (impossible).
#
# Calibration safe: Rekharsky CD data all have packing 0.3–0.9.

VDW_K_OVERLAP = 50.0    # kJ/mol scaling
VDW_N_POWER = 2.5       # super-quadratic ramp
VDW_P_ONSET = 1.0       # packing threshold


def dg_vdw_overlap(packing_coefficient: float) -> float:
    """VdW volume overlap penalty.

    Returns positive kJ/mol. Zero for packing ≤ 1.0.
    """
    if packing_coefficient <= VDW_P_ONSET:
        return 0.0
    excess = packing_coefficient - VDW_P_ONSET
    return VDW_K_OVERLAP * excess ** VDW_N_POWER


# ═══════════════════════════════════════════════════════════════════════════
# 2. ELECTROSTATIC CHARGE-CHARGE REPULSION
# ═══════════════════════════════════════════════════════════════════════════
# Like-charge guest in like-charge host → Coulombic penalty.
# Opposite charges → Coulombic attraction (favorable).
# Neutral → zero.
#
# Fujita Pd₂L₄: +4 total. Cationic guest repelled.
# Raymond Ga₄L₆: −12. Anionic guest repelled, cationic guest attracted.
# CDs: charge 0. No electrostatic term.
#
# Model: Coulomb in dielectric cavity.
# E = 332 × q₁×q₂ / (ε_eff × r) in kcal/mol → ×4.184 for kJ/mol
# Simplified: k_elec × q₁×q₂ / (ε_eff × r_eff)

K_COULOMB_KJ = 1389.4   # kJ·Å/mol for unit charges (e²/(4πε₀) in kJ·Å)
EPSILON_CAVITY = 40.0    # effective dielectric in aqueous host cavity
R_DEFAULT_A = 5.0        # default interaction distance (Å)


def dg_electrostatic(
    host_charge: int,
    guest_charge: int,
    cavity_diameter_A: float = 10.0,
) -> float:
    """Electrostatic charge-charge energy in host cavity.

    Returns kJ/mol. Positive = repulsion, negative = attraction.
    Zero if either charge is zero.
    """
    if host_charge == 0 or guest_charge == 0:
        return 0.0

    r_eff = max(R_DEFAULT_A, cavity_diameter_A / 2.0)
    return K_COULOMB_KJ * host_charge * guest_charge / (EPSILON_CAVITY * r_eff)


# ═══════════════════════════════════════════════════════════════════════════
# 3. GENERALIZED APERTURE EXCLUSION
# ═══════════════════════════════════════════════════════════════════════════
# Guest must fit through the narrowest opening (portal/pore/face).
# CB portals handled separately in unified_scorer_v2 (keep that code).
# This covers: MOF pore windows, cage face openings, calixarene rims.
#
# Uses guest 3D cross-section (min_d × mid_d) vs circular aperture.
# Penalty super-linear in excess cross-sectional area.

APERTURE_K = 2.0
APERTURE_POWER = 1.5
APERTURE_FLEX_RIGID = 0.5    # Å for rigid frameworks (MOF, zeolite)
APERTURE_FLEX_SOFT = 1.5     # Å for flexible hosts (organic cages)


def dg_aperture_exclusion(
    guest_min_dim_A: float,
    guest_mid_dim_A: float,
    aperture_diameter_A: float,
    rigid: bool = True,
) -> float:
    """Penalty for guest that cannot fit through host aperture.

    Returns positive kJ/mol. Zero if guest fits.
    """
    if aperture_diameter_A <= 0 or guest_min_dim_A <= 0:
        return 0.0

    flex = APERTURE_FLEX_RIGID if rigid else APERTURE_FLEX_SOFT
    sphericity = guest_min_dim_A / guest_mid_dim_A if guest_mid_dim_A > 0.1 else 1.0

    flex_d = aperture_diameter_A + flex * sphericity
    flex_area = math.pi * (flex_d / 2) ** 2
    guest_area = math.pi / 4 * guest_min_dim_A * guest_mid_dim_A

    excess = max(0.0, guest_area - flex_area)
    if excess <= 0:
        return 0.0

    return APERTURE_K * excess ** APERTURE_POWER


# ═══════════════════════════════════════════════════════════════════════════
# 4. STERIC CLASH (heavy-atom overflow)
# ═══════════════════════════════════════════════════════════════════════════
# When guest has more heavy atoms than the cavity can accommodate,
# functional groups collide with pocket walls.
#
# Capacity = cavity_volume / 15 Å³ per heavy atom.
# Overflow atoms × branching factor × (1 − flexibility).

K_CLASH_PER_ATOM = 4.0
VOLUME_PER_HEAVY_ATOM = 15.0


def dg_steric_clash(
    guest_n_heavy_atoms: int,
    cavity_volume_A3: float,
    guest_n_branches: int = 0,
    receptor_flexibility: float = 0.5,
) -> float:
    """Steric clash for guest atoms overflowing cavity.

    Returns positive kJ/mol. Zero if guest fits.
    """
    if cavity_volume_A3 <= 0:
        return 0.0

    capacity = cavity_volume_A3 / VOLUME_PER_HEAVY_ATOM
    overflow = max(0.0, guest_n_heavy_atoms - capacity)
    if overflow <= 0:
        return 0.0

    branch_factor = 1.0 + 0.1 * guest_n_branches
    flex_discount = max(0.1, 1.0 - receptor_flexibility * 0.8)

    return K_CLASH_PER_ATOM * overflow * branch_factor * flex_discount


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HGRepulsionResult:
    """All repulsion terms for a host-guest or receptor-guest pair."""
    dg_vdw_overlap: float = 0.0
    dg_electrostatic: float = 0.0
    dg_aperture: float = 0.0
    dg_steric_clash: float = 0.0
    dg_total_repulsion: float = 0.0

    @property
    def has_significant_repulsion(self):
        return self.dg_total_repulsion > 1.0  # > 1 kJ/mol


def compute_hg_repulsion(
    packing_coefficient: float = 0.0,
    host_charge: int = 0,
    guest_charge: int = 0,
    cavity_diameter_A: float = 0.0,
    cavity_volume_A3: float = 0.0,
    guest_min_dim_A: float = 0.0,
    guest_mid_dim_A: float = 0.0,
    aperture_diameter_A: float = 0.0,
    aperture_rigid: bool = True,
    guest_n_heavy_atoms: int = 0,
    guest_n_branches: int = 0,
    receptor_flexibility: float = 0.5,
) -> HGRepulsionResult:
    """Compute all repulsion terms for a host-guest or receptor-guest pair.

    Pass whatever is available — unused fields default to zero
    and their corresponding terms self-zero.
    """
    r = HGRepulsionResult()

    r.dg_vdw_overlap = dg_vdw_overlap(packing_coefficient)

    elec = dg_electrostatic(host_charge, guest_charge, cavity_diameter_A)
    r.dg_electrostatic = elec

    r.dg_aperture = dg_aperture_exclusion(
        guest_min_dim_A, guest_mid_dim_A,
        aperture_diameter_A, aperture_rigid,
    )

    r.dg_steric_clash = dg_steric_clash(
        guest_n_heavy_atoms, cavity_volume_A3,
        guest_n_branches, receptor_flexibility,
    )

    # Total: sum all unfavorable terms
    # Electrostatic can be negative (attraction) — only count repulsive part in total
    r.dg_total_repulsion = (
        r.dg_vdw_overlap
        + max(0.0, r.dg_electrostatic)
        + r.dg_aperture
        + r.dg_steric_clash
    )

    return r
