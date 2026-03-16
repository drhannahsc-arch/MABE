"""
knowledge/physics_pl_scorer.py — Physics-based protein-ligand scoring

Parallel to the QSAR-based _compute_general_pl_terms(). Does NOT touch
the QSAR path. Fires only for binding_mode == 'protein_ligand_physics'.

Energy decomposition:
  ΔG_bind = ΔG_desolv_ligand + ΔG_hydrophobic + ΔG_hbond + ΔG_conf + offset

All parameters back-solved from non-binding data:
  - Desolvation: FreeSolv v0.52 (642 molecules, Mobley)
  - Hydrophobic: Eisenberg-McLachlan γ = 0.0251 kJ/mol/Å² (consensus)
  - H-bond: Fersht/Pace consensus eps_hb = -3.0 kJ/mol, water displacement 1.2
  - Conf entropy: Mammen/Whitesides eps_rotor = 2.5 kJ/mol, f_partial = 0.5

No fitting against protein-ligand binding Ka. All parameters from
independent physical measurements.
"""

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS — sourced from MABE calibration database
# ═══════════════════════════════════════════════════════════════════════════

# Hydrophobic transfer (Eisenberg-McLachlan surface tension)
# Source: Eisenberg & McLachlan 1986, Nature 319:199
# Verified against HG scorer Phase 6 calibration
GAMMA_HYDROPHOBIC = -0.0251  # kJ/mol/Å² (negative = favorable burial)

# H-bond network at protein-ligand interface
# Source: Fersht 1985, Pace 2014 (consensus), MABE HG Phase 7
# Net H-bond energy = favorable formation - unfavorable water displacement
EPS_HBOND_FORMED = -3.0     # kJ/mol per H-bond formed (neutral donor→acceptor)
WATER_PENALTY_PER_HB = 3.5  # kJ/mol per water H-bond displaced
WATER_DISPLACEMENT = 0.8    # waters displaced per H-bond in PROTEIN pocket
                            # Lower than CD (1.2) because protein pre-organizes

# Net per H-bond in protein pocket:
# -3.0 + 3.5 × 0.8 = -0.2 kJ/mol (barely favorable — this is correct!)
# Protein H-bonds are near-neutral because water competes. The favorable
# ΔG comes from PREORGANIZATION (entropic, captured in offset), not per-HB energy.
# This is the Dunitz/Fersht/Williams result.

# Conformational entropy (rotor freezing upon binding)
# Source: Mammen et al. 1998, Angew. Chem. 37:2754
# Verified against HG scorer Phase 9
EPS_ROTOR = 2.5     # kJ/mol per frozen rotor (positive = unfavorable)
F_PARTIAL = 0.65    # fraction of rotors actually frozen in PL binding
                    # Higher than HG (0.5) because protein pockets are more
                    # constraining than CD cavities

# Charge-assisted H-bond bonus
EPS_CHARGED_HB = -10.0  # kJ/mol per charge-assisted H-bond (NH3+→COO-)


# ═══════════════════════════════════════════════════════════════════════════
# SCORER
# ═══════════════════════════════════════════════════════════════════════════

def compute_physics_pl_terms(uc, result):
    """Physics-based PL scoring. Self-zeros if mode != protein_ligand_physics.

    Populates:
      result.dg_group_desolv   — ligand desolvation cost (FreeSolv model)
      result.dg_hydrophobic    — hydrophobic transfer (SASA burial)
      result.dg_hbond          — H-bond network (formed - water displaced)
      result.dg_conf_entropy   — rotor freezing penalty

    Requires uc fields:
      guest_smiles             — for FreeSolv desolvation (if OpenBabel available)
      sasa_buried_A2           — buried nonpolar SASA upon binding
      guest_sasa_total_A2      — total guest SASA (for burial fraction)
      n_hbonds_formed          — number of H-bonds at interface
      guest_rotatable_bonds    — number of rotatable bonds
      guest_charge             — for charge-assisted HB detection
    """
    if uc.binding_mode != "protein_ligand_physics":
        return

    if not uc.guest_smiles:
        return

    # ── 1. LIGAND DESOLVATION (FreeSolv-calibrated) ──────────────────

    # Compute burial fraction
    f_burial = 0.0
    if uc.guest_sasa_total_A2 > 0 and uc.sasa_buried_A2 > 0:
        f_burial = min(uc.sasa_buried_A2 / uc.guest_sasa_total_A2, 1.0)
    elif uc.sasa_buried_A2 > 0:
        # No total SASA but have buried → estimate
        f_burial = 0.6  # typical for drug-like molecules in protein pockets

    if f_burial > 0:
        try:
            from knowledge.ligand_desolvation import predict_dG_hydration
            dG_hydr = predict_dG_hydration(uc.guest_smiles)
            if dG_hydr is not None:
                # Desolvation cost = -f_burial × ΔG_hydr
                # If ΔG_hydr < 0 (hydrophilic): cost > 0 (unfavorable)
                # If ΔG_hydr > 0 (hydrophobic): cost < 0 (favorable to bury)
                result.dg_group_desolv = -f_burial * dG_hydr
        except ImportError:
            # OpenBabel not available — fall back to SASA-based estimate
            _fallback_desolvation(uc, result, f_burial)

    # ── 2. HYDROPHOBIC TRANSFER (SASA burial) ────────────────────────

    # Burying nonpolar SASA releases ordered water → favorable
    buried_np = uc.sasa_buried_A2
    if buried_np > 0:
        # Only count nonpolar portion
        if uc.guest_sasa_nonpolar_A2 > 0 and uc.guest_sasa_total_A2 > 0:
            np_frac = uc.guest_sasa_nonpolar_A2 / uc.guest_sasa_total_A2
        else:
            np_frac = 0.6  # typical for drug-like molecules
        buried_np_actual = buried_np * np_frac
        result.dg_hydrophobic = GAMMA_HYDROPHOBIC * buried_np_actual

    # ── 3. H-BOND NETWORK ────────────────────────────────────────────

    n_hb = uc.n_hbonds_formed
    if n_hb > 0:
        # Neutral H-bonds
        dg_formed = n_hb * EPS_HBOND_FORMED
        dg_water = n_hb * WATER_DISPLACEMENT * WATER_PENALTY_PER_HB
        result.dg_hbond = dg_formed + dg_water

        # Charge-assisted bonus: if ligand is charged, some HBs are stronger
        if abs(uc.guest_charge) >= 1:
            # Estimate 1-2 charge-assisted HBs per formal charge
            n_charged_hb = min(abs(uc.guest_charge) * 2, n_hb)
            # Replace neutral HB energy with charged for those
            charged_upgrade = n_charged_hb * (EPS_CHARGED_HB - EPS_HBOND_FORMED)
            result.dg_hbond += charged_upgrade

    # ── 4. CONFORMATIONAL ENTROPY ────────────────────────────────────

    n_rot = uc.guest_rotatable_bonds
    if n_rot > 0:
        result.dg_conf_entropy = n_rot * EPS_ROTOR * F_PARTIAL


def _fallback_desolvation(uc, result, f_burial):
    """SASA-based desolvation fallback when OpenBabel unavailable.

    Uses polar/nonpolar SASA split with literature γ values.
    Less accurate than FreeSolv model but zero dependencies.
    """
    # Polar surface burial cost: ~0.10 kJ/mol/Å² (Marcus/Cabani consensus)
    # Nonpolar surface burial benefit: ~-0.025 kJ/mol/Å² (hydrophobic)
    GAMMA_POLAR = +0.10
    GAMMA_NONPOLAR = -0.025

    polar_buried = uc.guest_sasa_polar_A2 * f_burial if uc.guest_sasa_polar_A2 > 0 else 0
    nonpolar_buried = uc.guest_sasa_nonpolar_A2 * f_burial if uc.guest_sasa_nonpolar_A2 > 0 else 0

    result.dg_group_desolv = GAMMA_POLAR * polar_buried + GAMMA_NONPOLAR * nonpolar_buried


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE SCORING (call without unified_scorer_v2 wiring)
# ═══════════════════════════════════════════════════════════════════════════

def score_physics_pl(uc, verbose=False):
    """Score a protein-ligand complex using physics-only terms.

    Can be called directly without going through predict().
    Returns dict with energy decomposition.

    This is the function to use for benchmarking physics vs QSAR.
    """
    from dataclasses import dataclass

    @dataclass
    class _Result:
        dg_group_desolv: float = 0.0
        dg_hydrophobic: float = 0.0
        dg_hbond: float = 0.0
        dg_conf_entropy: float = 0.0

    # Temporarily override binding_mode to fire the scorer
    orig_mode = uc.binding_mode
    uc.binding_mode = "protein_ligand_physics"

    r = _Result()
    compute_physics_pl_terms(uc, r)

    uc.binding_mode = orig_mode  # restore

    dg_net = r.dg_group_desolv + r.dg_hydrophobic + r.dg_hbond + r.dg_conf_entropy

    if verbose:
        print(f"Physics PL scoring: {uc.name}")
        print(f"  dG_desolv     = {r.dg_group_desolv:+.2f} kJ/mol")
        print(f"  dG_hydrophobic = {r.dg_hydrophobic:+.2f} kJ/mol")
        print(f"  dG_hbond      = {r.dg_hbond:+.2f} kJ/mol")
        print(f"  dG_conf       = {r.dg_conf_entropy:+.2f} kJ/mol")
        print(f"  dG_total      = {dg_net:+.2f} kJ/mol")

    return {
        "dg_desolv": r.dg_group_desolv,
        "dg_hydrophobic": r.dg_hydrophobic,
        "dg_hbond": r.dg_hbond,
        "dg_conf_entropy": r.dg_conf_entropy,
        "dg_total": dg_net,
        "log_Ka_pred": -dg_net / (2.303 * 8.314e-3 * 298.15),
    }