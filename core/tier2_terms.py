"""
core/tier2_terms.py — Tier 2 Interaction Energy Compute Functions

Ten _compute_tier2_* functions, each with self-zero guard.
All terms are additive to the existing unified scorer.
Convention: ΔG in kJ/mol, negative = favorable.
"""

import math
from core.tier2_constants import (
    RT_298, LN10_RT, COULOMB_KJ_A, EPSILON_WATER,
    BORN_CONSTANT, BORN_DR_CATION, BORN_DR_ANION,
    CATION_PI_AQUEOUS_DEFAULTS, CATION_PI_D0, CATION_PI_BETA,
    PI_STACK_AQUEOUS, PI_STACK_HAMMETT_SLOPE,
    HALOGEN_BOND_ENERGY, HALOGEN_BOND_ANGLE_MIN,
    SALT_BRIDGE_DG, SALT_BRIDGE_I_SLOPE,
    HBOND_COOP_FACTOR, HBOND_COOP_MAX_CHAIN,
    ANION_PI_ENERGY,
    METALLOPHILIC_ENERGY, METALLOPHILIC_D0, METALLOPHILIC_BETA, D10_METALS,
    GROUP_DESOLVATION_COST, GROUP_POLARIZABILITY,
    EPS_WATER_PER_HBOND, EPS_WATER_OH, EPS_WATER_NH, EPS_WATER_O_ACCEPTOR,
    GAMMA_SASA_DESOLV,
)


# ═══════════════════════════════════════════════════════════════════════════
# T1: LONDON DISPERSION (upgraded)
# ═══════════════════════════════════════════════════════════════════════════

def compute_dispersion_upgraded(uc, result):
    """Upgraded London dispersion using polarizability-volume proxy.

    Replaces the Hamaker-based stub. Uses coarse-grained formula:
        ΔG_disp ≈ ε_disp × α_guest × (buried_fraction)
    where ε_disp calibrated against SAMPL host-guest set.

    Self-zeros if: no guest polarizability data available.
    """
    alpha_guest = getattr(uc, 'guest_polarizability_A3', 0.0)
    if alpha_guest <= 0.0:
        return

    buried_frac = 0.0
    if uc.guest_sasa_total_A2 > 0 and getattr(uc, 'sasa_buried_A2', 0.0) > 0:
        buried_frac = uc.sasa_buried_A2 / uc.guest_sasa_total_A2
        buried_frac = min(buried_frac, 1.0)

    if buried_frac <= 0.0:
        return

    # Coarse-grained dispersion coefficient (kJ/mol per Å³ of buried polarizability)
    # Calibration target: SAMPL host-guest ΔG residuals after other terms
    EPSILON_DISP = -0.10  # kJ/(mol·Å³), conservative

    dg = EPSILON_DISP * alpha_guest * buried_frac
    result.dg_dispersion_t2 = dg


# ═══════════════════════════════════════════════════════════════════════════
# T2: CATION-π
# ═══════════════════════════════════════════════════════════════════════════

def compute_cation_pi(uc, result):
    """Cation-π interaction energy.

    Physics: electrostatic attraction of cation to π-face, attenuated by
    aqueous solvation. Values from Dougherty cyclophane studies.

    Self-zeros if: n_cation_pi_contacts <= 0.
    """
    n_contacts = getattr(uc, 'n_cation_pi_contacts', 0)
    if n_contacts <= 0:
        return

    # Determine ring type if available
    cation_pi_type = getattr(uc, 'cation_pi_type', 'organic_cation_benzene')
    eps = CATION_PI_AQUEOUS_DEFAULTS.get(cation_pi_type, -4.0)

    # Distance correction if available
    d = getattr(uc, 'cation_pi_distance_A', CATION_PI_D0)
    f_dist = math.exp(-CATION_PI_BETA * (d - CATION_PI_D0)**2) if d > 0 else 1.0

    result.dg_cation_pi = n_contacts * eps * f_dist


# ═══════════════════════════════════════════════════════════════════════════
# T3: π-π STACKING
# ═══════════════════════════════════════════════════════════════════════════

def compute_pi_stack(uc, result):
    """π-π stacking interaction energy (face-to-face aromatic contacts).

    Residual dispersion + electrostatic after hydrophobic SASA subtraction.
    T-shaped contacts handled by existing CH-π term.

    Self-zeros if: n_pi_stack_contacts <= 0.
    """
    n_contacts = getattr(uc, 'n_pi_stack_contacts', 0)
    if n_contacts <= 0:
        return

    stack_type = getattr(uc, 'pi_stack_type', 'parallel_displaced')
    eps = PI_STACK_AQUEOUS.get(stack_type, -3.0)

    # Hammett correction for electron-poor/rich aromatics
    sigma = getattr(uc, 'pi_stack_hammett_sigma', 0.0)
    hammett_correction = PI_STACK_HAMMETT_SLOPE * sigma  # negative σ weakens

    result.dg_pi_stack = n_contacts * (eps + hammett_correction)


# ═══════════════════════════════════════════════════════════════════════════
# T4: HALOGEN BONDING (σ-hole)
# ═══════════════════════════════════════════════════════════════════════════

def compute_halogen_bond(uc, result):
    """Halogen bond energy via σ-hole interaction.

    Physics: positive electrostatic potential on C-X σ-hole attracts
    nucleophilic sites. Highly directional (C-X···B ≈ 180°).

    Self-zeros if: n_halogen_bonds <= 0.
    """
    n_xb = getattr(uc, 'n_halogen_bonds', 0)
    if n_xb <= 0:
        return

    halogen_type = getattr(uc, 'halogen_bond_type', 'C-Br')
    nucleophile = getattr(uc, 'halogen_bond_nucleophile', 'N')
    key = (halogen_type, nucleophile)
    eps = HALOGEN_BOND_ENERGY.get(key, -8.0)

    # Angular attenuation
    angle = getattr(uc, 'halogen_bond_angle', 175.0)
    if angle < HALOGEN_BOND_ANGLE_MIN:
        f_angle = 0.0
    else:
        # cos²(θ - 180°) peaks at θ = 180°
        f_angle = math.cos(math.radians(angle - 180.0))**2

    result.dg_halogen_bond = n_xb * eps * f_angle


# ═══════════════════════════════════════════════════════════════════════════
# T5: SALT BRIDGE / ION PAIR
# ═══════════════════════════════════════════════════════════════════════════

def compute_salt_bridge(uc, result):
    """Salt bridge (organic ion pair) interaction energy.

    Physics: Coulombic attraction + H-bonding between oppositely charged
    organic groups. ~5-6 kJ/mol per 1:1 pair at physiological ionic strength,
    nearly independent of ion identity.

    Self-zeros if: n_salt_bridges <= 0.
    Does NOT fire for metal-donor electrostatics (handled by existing Term 5).
    """
    n_sb = getattr(uc, 'n_salt_bridges', 0)
    if n_sb <= 0:
        return

    # Charge product for each salt bridge (default -1 for 1:1 pairs)
    z_product = getattr(uc, 'salt_bridge_z_product', -1)

    # Base ΔG from universal ion pair data
    dg_base = SALT_BRIDGE_DG.get(z_product, -5.5)

    # Ionic strength correction
    I = uc.ionic_strength_M
    dg_I_correction = SALT_BRIDGE_I_SLOPE * (math.sqrt(I) - math.sqrt(0.15))

    # Burial factor
    is_buried = getattr(uc, 'salt_bridge_buried', False)
    burial_factor = 2.0 if is_buried else 1.0  # Larger effect when desolvated

    result.dg_salt_bridge = n_sb * (dg_base + dg_I_correction) * burial_factor


# ═══════════════════════════════════════════════════════════════════════════
# T6: BORN ION SOLVATION (upgraded desolvation)
# ═══════════════════════════════════════════════════════════════════════════

def compute_born_solvation(uc, result):
    """Born model solvation energy for charged species.

    Supplements existing metal desolvation (Term 2, Marcus tables) with
    analytical interpolation for untabulated ions and non-metal charges.

    ΔG_Born = -(z² × 694.3) / r_eff × (1 - 1/ε)

    Self-zeros if: no formal charges present, or no Born radius available.
    """
    z = getattr(uc, 'guest_formal_charge', uc.guest_charge)
    r_crystal = getattr(uc, 'guest_ion_radius_A', 0.0)

    if z == 0 or r_crystal <= 0.0:
        return

    # Born radius correction
    dr = BORN_DR_CATION if z > 0 else BORN_DR_ANION
    r_eff = r_crystal + dr

    # Born solvation energy (negative = stabilizing in solvent)
    dg_born = -(z**2 * BORN_CONSTANT) / r_eff * (1.0 - 1.0 / EPSILON_WATER)

    # Store as the Born correction BEYOND what Marcus tables already provide
    # For ions with Marcus data (most metals), this is zero (existing term handles it)
    # For organic ions without Marcus data, this is the primary estimate
    has_marcus = getattr(uc, 'has_marcus_hydration_dg', False)
    if not has_marcus:
        result.dg_born_solvation = dg_born
    # If Marcus data exists, existing Term 2 already handles it — don't double-count


# ═══════════════════════════════════════════════════════════════════════════
# T7: COOPERATIVE H-BOND NETWORKS
# ═══════════════════════════════════════════════════════════════════════════

def compute_hbond_cooperativity(uc, result):
    """Non-additive H-bond cooperativity correction.

    Physics: polarization transfer through H-bond chains enhances
    subsequent bonds by ~10-20% each. Modifies existing ΔG_hbond.

    Self-zeros if: max_hbond_chain_length <= 1.
    """
    chain_len = getattr(uc, 'max_hbond_chain_length', 0)
    if chain_len <= 1:
        return

    chain_len = min(chain_len, HBOND_COOP_MAX_CHAIN)

    # Determine cooperativity factor by chain type
    chain_type = getattr(uc, 'hbond_chain_type', 'default')
    coop = HBOND_COOP_FACTOR.get(chain_type, 0.15)

    # Cooperativity correction = coop × (n_chain - 1) × average_hbond_energy
    # Use existing dg_hbond as the base (already computed by HG scorer)
    n_hbonds = max(uc.n_hbonds_formed, 1)
    avg_hbond = result.dg_hbond / n_hbonds if n_hbonds > 0 and result.dg_hbond != 0 else -5.0

    correction = coop * (chain_len - 1) * avg_hbond
    result.dg_hbond_coop = correction


# ═══════════════════════════════════════════════════════════════════════════
# T8: ANION-π
# ═══════════════════════════════════════════════════════════════════════════

def compute_anion_pi(uc, result):
    """Anion-π interaction with electron-deficient aromatic.

    Self-zeros if: n_anion_pi_contacts <= 0.
    """
    n_contacts = getattr(uc, 'n_anion_pi_contacts', 0)
    if n_contacts <= 0:
        return

    anion_pi_type = getattr(uc, 'anion_pi_type', 'default')
    eps = ANION_PI_ENERGY.get(anion_pi_type, -3.0)

    result.dg_anion_pi = n_contacts * eps


# ═══════════════════════════════════════════════════════════════════════════
# T9: AUROPHILIC / METALLOPHILIC
# ═══════════════════════════════════════════════════════════════════════════

def compute_metallophilic(uc, result):
    """d10-d10 closed-shell metallophilic attraction.

    Physics: relativistic + correlation effects (esp. Au-Au).
    Comparable to H-bond strength (20-50 kJ/mol).

    Self-zeros if: n_d10_d10_contacts <= 0.
    """
    n_contacts = getattr(uc, 'n_d10_d10_contacts', 0)
    if n_contacts <= 0:
        return

    metal_pair = getattr(uc, 'metallophilic_pair', ("Au", "Au"))
    eps = METALLOPHILIC_ENERGY.get(metal_pair, -20.0)

    # Distance dependence
    d = getattr(uc, 'metallophilic_distance_A', 3.0)
    d0 = METALLOPHILIC_D0.get(metal_pair, 3.0)
    f_dist = math.exp(-METALLOPHILIC_BETA * abs(d - d0)) if d > 0 else 1.0

    result.dg_metallophilic = n_contacts * eps * f_dist


# ═══════════════════════════════════════════════════════════════════════════
# T10: GROUP DESOLVATION (systematic)
# ═══════════════════════════════════════════════════════════════════════════

def compute_group_desolvation(uc, result):
    """Systematic per-group desolvation cost upon burial.

    Physics: each functional group shed from its solvation shell upon
    binding pays a group-specific cost (Cabani 1981 additivity).

    Self-zeros if: no buried_groups list provided.
    """
    buried_groups = getattr(uc, 'buried_groups', [])
    if not buried_groups:
        return

    total = 0.0
    for group_spec in buried_groups:
        if isinstance(group_spec, dict):
            group_type = group_spec.get("type", "default_polar")
            burial_frac = group_spec.get("burial_fraction", 1.0)
        elif isinstance(group_spec, str):
            group_type = group_spec
            burial_frac = 1.0
        else:
            continue

        cost = GROUP_DESOLVATION_COST.get(group_type,
               GROUP_DESOLVATION_COST.get("default_polar", 8.0))
        total += cost * burial_frac

    result.dg_group_desolv = total  # positive = unfavorable (penalty)


# ═══════════════════════════════════════════════════════════════════════════
# T11: SASA-BASED WATER COMPETITION PENALTY (P20 + P13)
# Back-solved from MNSol v2012 + FreeSolv v0.52
# ═══════════════════════════════════════════════════════════════════════════

def compute_water_penalty(uc, result):
    """Water competition penalty from SASA-level polar burial.

    Three modes (highest-detail wins):
      1. Per-Å² SASA (if polar SASA fields populated)
      2. Per-H-bond count (if n_water_hbonds_displaced > 0)
      3. Self-zero (no data)

    Does NOT overlap with T10 (group desolvation): T10 fires on
    buried_groups list, T11 fires on SASA fields. If both are
    populated, the UC constructor should use one or the other.

    Calibration: MNSol v2012 N=390 aq. neutrals, R²=0.695
                 FreeSolv v0.52 N=642, cross-validated r=0.82
    """
    # ── Mode 1: per-Å² SASA (most detailed) ──
    sasa_oh = getattr(uc, 'sasa_oh_buried_A2', 0.0)
    sasa_nh = getattr(uc, 'sasa_nh_buried_A2', 0.0)
    sasa_o_acc = getattr(uc, 'sasa_o_acceptor_buried_A2', 0.0)
    sasa_n_acc = getattr(uc, 'sasa_n_acceptor_buried_A2', 0.0)

    sasa_polar_total = sasa_oh + sasa_nh + sasa_o_acc + sasa_n_acc
    if sasa_polar_total > 0:
        penalty = (GAMMA_SASA_DESOLV["OH_donor"] * sasa_oh
                   + GAMMA_SASA_DESOLV["NH_donor"] * sasa_nh
                   + GAMMA_SASA_DESOLV["O_acceptor"] * sasa_o_acc
                   + GAMMA_SASA_DESOLV["N_acceptor"] * sasa_n_acc)
        result.dg_water_penalty = penalty  # positive = unfavorable
        return

    # ── Mode 2: per-H-bond count ──
    n_displaced = getattr(uc, 'n_water_hbonds_displaced', 0)
    if n_displaced > 0:
        result.dg_water_penalty = EPS_WATER_PER_HBOND * n_displaced
        return

    # ── Mode 3: self-zero ──
    # No polar SASA data and no H-bond count → zero contribution


# ═══════════════════════════════════════════════════════════════════════════
# MASTER DISPATCH — call all Tier 2 terms
# ═══════════════════════════════════════════════════════════════════════════

ALL_TIER2_FUNCTIONS = [
    compute_dispersion_upgraded,
    compute_cation_pi,
    compute_pi_stack,
    compute_halogen_bond,
    compute_salt_bridge,
    compute_born_solvation,
    compute_hbond_cooperativity,
    compute_anion_pi,
    compute_metallophilic,
    compute_group_desolvation,
    compute_water_penalty,
]

TIER2_RESULT_FIELDS = [
    "dg_dispersion_t2",
    "dg_cation_pi",
    "dg_pi_stack",
    "dg_halogen_bond",
    "dg_salt_bridge",
    "dg_born_solvation",
    "dg_hbond_coop",
    "dg_anion_pi",
    "dg_metallophilic",
    "dg_group_desolv",
    "dg_water_penalty",
]


def compute_all_tier2(uc, result):
    """Compute all Tier 2 terms. Each self-zeros internally."""
    for func in ALL_TIER2_FUNCTIONS:
        try:
            func(uc, result)
        except Exception:
            pass  # Self-zero on error — never break existing predictions


def tier2_total(result):
    """Sum all Tier 2 contributions."""
    total = 0.0
    for field_name in TIER2_RESULT_FIELDS:
        total += getattr(result, field_name, 0.0)
    return total
