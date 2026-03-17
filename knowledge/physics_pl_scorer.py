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

# Fix F: Aromatic vs aliphatic hydrophobic split
# Aromatic surfaces have π-electron cloud → stronger dispersion with protein
# aromatics (Trp, Tyr, Phe) and better vdW contact than aliphatic.
# Source: Horton & Lewis 1992, Protein Sci 1:169 (aromatic contribution to binding)
GAMMA_HYDROPHOBIC_ARO = -0.035  # kJ/mol/Å² for aromatic nonpolar SASA
GAMMA_HYDROPHOBIC_ALI = -0.020  # kJ/mol/Å² for aliphatic nonpolar SASA

# Fix A: vdW contact energy (London dispersion at interface)
# The hydrophobic γ_SASA captures CAVITY FORMATION (water release).
# The vdW contact captures DIRECT LONDON DISPERSION at the PL interface.
# These are additive: total = cavity + contact.
# Source: Horton & Lewis 1992; Kollman 2000 (MM-PBSA decomposition)
# Value: -0.08 to -0.12 kJ/mol/Å² over ALL buried SASA (polar + nonpolar)
GAMMA_VDW_CONTACT = -0.080  # kJ/mol/Å² (conservative, lower bound)

# Fix C: Desolvation attenuation for H-bonded groups
# When a polar group forms a compensating H-bond at the interface,
# its desolvation penalty is partially offset. The Pace/Fersht
# H-bond energy already includes the NET effect, but the FreeSolv
# desolvation model charges the FULL solvation cost.
# Correction: reduce desolvation by fraction of HB compensation.
# Each interface H-bond compensates ~60% of one polar group's desolvation.
DESOLV_HB_ATTENUATION = 0.60  # fraction of per-group desolvation offset per HB
DESOLV_PER_POLAR_GROUP = 6.0  # kJ/mol average desolvation per polar group (Cabani)

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

    # ── 0. PDB POCKET ANALYSIS (if structure available) ──────────────
    # Extracts protein-side descriptors → populates UC fields → enables
    # Born charge cancellation, water displacement, preorganization.
    # If no PDB text, scorer works in SMILES-only mode (unchanged).

    _pocket_desc = None
    pdb_text = getattr(uc, 'host_pdb_text', '')
    if pdb_text:
        try:
            from knowledge.pocket_analyzer import PocketAnalyzer, populate_uc_from_pocket
            analyzer = PocketAnalyzer.from_text(pdb_text)
            # Auto-detect ligand (first non-water, non-metal HETATM)
            _pocket_desc = analyzer.analyze_pocket(cutoff_A=6.0)
            if _pocket_desc and _pocket_desc.n_pocket_residues > 0:
                populate_uc_from_pocket(uc, _pocket_desc)
        except (ImportError, Exception):
            pass  # graceful fallback to SMILES-only

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

    # ── 2. HYDROPHOBIC TRANSFER + vdW CONTACT (SASA burial) ────────

    buried_total = uc.sasa_buried_A2
    if buried_total > 0:
        # Fix F: split aromatic vs aliphatic nonpolar
        if uc.guest_sasa_nonpolar_A2 > 0 and uc.guest_sasa_total_A2 > 0:
            np_frac = uc.guest_sasa_nonpolar_A2 / uc.guest_sasa_total_A2
        else:
            np_frac = 0.6

        buried_np = buried_total * np_frac

        # Estimate aromatic fraction of nonpolar SASA from SMILES
        aro_frac_of_np = _estimate_aromatic_fraction(uc.guest_smiles)
        buried_np_aro = buried_np * aro_frac_of_np
        buried_np_ali = buried_np * (1.0 - aro_frac_of_np)

        # Hydrophobic: aromatic + aliphatic with different γ
        dg_hydro_aro = GAMMA_HYDROPHOBIC_ARO * buried_np_aro
        dg_hydro_ali = GAMMA_HYDROPHOBIC_ALI * buried_np_ali
        result.dg_hydrophobic = dg_hydro_aro + dg_hydro_ali

        # Fix A: vdW contact energy over ALL buried surface (polar + nonpolar)
        # This is direct London dispersion at the protein-ligand interface,
        # additive with cavity formation (hydrophobic transfer).
        result.dg_dispersion_t2 = GAMMA_VDW_CONTACT * buried_total

    # ── 3. H-BOND NETWORK (per-type physics) ────────────────────────

    n_hb = uc.n_hbonds_formed
    if n_hb > 0:
        hb_types = getattr(uc, 'hbond_types', [])

        if hb_types and len(hb_types) > 0:
            # Detailed per-type scoring
            try:
                from knowledge.hbond_pl_physics import score_hbond_network
                hb_list = [{"type": t} for t in hb_types]
                hb_result = score_hbond_network(hb_list)
                result.dg_hbond = hb_result["total_kJ"]
            except ImportError:
                # Fallback to count-based
                _hbond_from_counts(uc, result, n_hb)
        else:
            # Count-based: estimate n_neutral vs n_charged from guest_charge
            _hbond_from_counts(uc, result, n_hb)

    # ── 3b. DESOLVATION ATTENUATION (Fix C) ──────────────────────────
    # When H-bonds form at the interface, each one partially compensates
    # the desolvation cost of one polar group. The Pace H-bond energy
    # captures the NET effect, but FreeSolv charges FULL desolvation.
    # Reduce desolvation by the H-bond compensation fraction.
    if n_hb > 0 and result.dg_group_desolv > 0:
        compensation = n_hb * DESOLV_HB_ATTENUATION * DESOLV_PER_POLAR_GROUP
        result.dg_group_desolv = max(result.dg_group_desolv - compensation, 0.0)

    # ── 4. CONFORMATIONAL ENTROPY ────────────────────────────────────

    n_rot = uc.guest_rotatable_bonds
    if n_rot > 0:
        # Try bond-type-specific entropy (requires OpenBabel for classification)
        try:
            from knowledge.conf_entropy_druglike import compute_conf_entropy
            if uc.guest_smiles:
                conf_result = compute_conf_entropy(uc.guest_smiles)
                if conf_result is not None:
                    result.dg_conf_entropy = conf_result["total_kJ"]
                    n_rot = 0  # skip fallback
        except ImportError:
            pass
        # Fallback: flat rate (only if bond-type didn't fire)
        if n_rot > 0:
            result.dg_conf_entropy = n_rot * EPS_ROTOR * F_PARTIAL

    # ── 5. ELECTROSTATIC SOLVATION (Born + group-additive) ───────

    try:
        from knowledge.electrostatic_solvation import compute_electrostatic_desolvation
        bf = 0.0
        if uc.guest_sasa_total_A2 > 0 and uc.sasa_buried_A2 > 0:
            bf = min(uc.sasa_buried_A2 / uc.guest_sasa_total_A2, 1.0)
        elif uc.sasa_buried_A2 > 0:
            bf = 0.6

        if bf > 0:
            result.dg_born_solvation = compute_electrostatic_desolvation(uc, bf)
    except ImportError:
        pass

    # ── 6. ENTROPY OF MIXING (translational + rotational loss) ───

    try:
        from knowledge.entropy_of_mixing import mixing_entropy_quick
        mw = getattr(uc, 'guest_mw', 0.0)
        if mw > 0:
            result.dg_mixing_entropy = mixing_entropy_quick(
                mw_Da=mw, include_vib_recovery=True)
        else:
            result.dg_mixing_entropy = mixing_entropy_quick(
                category="drug_like", include_vib_recovery=True)
    except ImportError:
        pass

    # ── 7. DISPERSION element-specific correction (Grimme D3) ─────
    # Adds element-specific correction ON TOP of the vdW contact from section 2.
    # Halogens/S get a bonus; F/O get a small penalty relative to carbon.

    try:
        from knowledge.dispersion_d3 import compute_dispersion_correction
        if uc.sasa_buried_A2 > 0 and uc.guest_smiles:
            result.dg_dispersion_t2 += compute_dispersion_correction(
                uc.guest_smiles, uc.sasa_buried_A2,
                uc.guest_sasa_total_A2 if uc.guest_sasa_total_A2 > 0 else None)
    except ImportError:
        pass

    # ── 8-10. PROTEIN-SIDE TERMS (from PDB pocket analysis) ─────────
    # Only fire when pocket_analyzer ran successfully in Stage 0.
    # Map to existing PredictionResult fields (self-zero otherwise):
    #   pocket desolvation  → dg_cavity_dehydration
    #   water displacement  → dg_size_mismatch
    #   preorganization     → dg_shape

    if _pocket_desc is not None and _pocket_desc.n_pocket_residues > 0:
        # 8. Pocket desolvation: protein surface that gets buried
        result.dg_cavity_dehydration = _pocket_desc.pocket_desolv_kJ

        # 9. Water displacement: favorable for unhappy waters, costly for happy
        result.dg_size_mismatch = _pocket_desc.water_displacement_kJ

        # 10. Preorganization: rigid pockets pay less entropy cost
        result.dg_shape = _pocket_desc.preorganization_kJ


def _estimate_aromatic_fraction(smiles):
    """Estimate fraction of nonpolar SASA that is aromatic.

    Uses OpenBabel atom typing if available, falls back to SMILES character counting.
    Returns float 0-1 (fraction of nonpolar surface that is aromatic C).
    """
    if not smiles:
        return 0.3  # default for drug-like

    try:
        from openbabel import pybel
        mol = pybel.readstring("smi", smiles)
        obmol = mol.OBMol
        n_aro = sum(1 for i in range(obmol.NumAtoms())
                    if obmol.GetAtom(i+1).IsAromatic()
                    and obmol.GetAtom(i+1).GetAtomicNum() == 6)
        n_c = sum(1 for i in range(obmol.NumAtoms())
                  if obmol.GetAtom(i+1).GetAtomicNum() == 6)
        if n_c == 0:
            return 0.0
        return n_aro / n_c
    except ImportError:
        pass

    # Fallback: count lowercase c (aromatic) vs uppercase C (aliphatic) in SMILES
    n_aro = smiles.count('c')
    n_ali = smiles.count('C')
    total = n_aro + n_ali
    if total == 0:
        return 0.0
    return n_aro / total


def _hbond_from_counts(uc, result, n_hb):
    """Estimate H-bond energy from counts + guest charge.

    When detailed hbond_types aren't available, estimate the breakdown:
      - If ligand is charged: ~2 HBs per formal charge are charge-assisted
      - Remainder are neutral
    Uses per-category energies from hbond_pl_physics.
    """
    try:
        from knowledge.hbond_pl_physics import score_hbond_simple
        n_charged = 0
        if abs(uc.guest_charge) >= 1:
            n_charged = min(abs(uc.guest_charge) * 2, n_hb)
        n_neutral = n_hb - n_charged
        result.dg_hbond = score_hbond_simple(
            n_neutral=n_neutral, n_charged=n_charged)
    except ImportError:
        # Ultimate fallback: flat rate
        result.dg_hbond = n_hb * EPS_HBOND_FORMED + n_hb * WATER_DISPLACEMENT * WATER_PENALTY_PER_HB
        if abs(uc.guest_charge) >= 1:
            n_ch = min(abs(uc.guest_charge) * 2, n_hb)
            result.dg_hbond += n_ch * (EPS_CHARGED_HB - EPS_HBOND_FORMED)


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
        dg_born_solvation: float = 0.0
        dg_mixing_entropy: float = 0.0
        dg_dispersion_t2: float = 0.0
        # Protein-side (from PDB pocket analysis)
        dg_cavity_dehydration: float = 0.0  # pocket desolvation
        dg_size_mismatch: float = 0.0       # water displacement
        dg_shape: float = 0.0               # preorganization

    # Temporarily override binding_mode to fire the scorer
    orig_mode = uc.binding_mode
    uc.binding_mode = "protein_ligand_physics"

    r = _Result()
    compute_physics_pl_terms(uc, r)

    uc.binding_mode = orig_mode  # restore

    dg_net = (r.dg_group_desolv + r.dg_hydrophobic + r.dg_hbond
              + r.dg_conf_entropy + r.dg_born_solvation
              + r.dg_mixing_entropy + r.dg_dispersion_t2
              + r.dg_cavity_dehydration + r.dg_size_mismatch + r.dg_shape)

    if verbose:
        print(f"Physics PL scoring: {uc.name}")
        print(f"  dG_desolv      = {r.dg_group_desolv:+.2f} kJ/mol")
        print(f"  dG_hydrophobic = {r.dg_hydrophobic:+.2f} kJ/mol")
        print(f"  dG_hbond       = {r.dg_hbond:+.2f} kJ/mol")
        print(f"  dG_conf        = {r.dg_conf_entropy:+.2f} kJ/mol")
        print(f"  dG_elec_solv   = {r.dg_born_solvation:+.2f} kJ/mol")
        print(f"  dG_mixing      = {r.dg_mixing_entropy:+.2f} kJ/mol")
        print(f"  dG_dispersion  = {r.dg_dispersion_t2:+.2f} kJ/mol")
        print(f"  dG_pocket_desolv = {r.dg_cavity_dehydration:+.2f} kJ/mol")
        print(f"  dG_water_displ = {r.dg_size_mismatch:+.2f} kJ/mol")
        print(f"  dG_preorg      = {r.dg_shape:+.2f} kJ/mol")
        print(f"  dG_total       = {dg_net:+.2f} kJ/mol")

    return {
        "dg_desolv": r.dg_group_desolv,
        "dg_hydrophobic": r.dg_hydrophobic,
        "dg_hbond": r.dg_hbond,
        "dg_conf_entropy": r.dg_conf_entropy,
        "dg_born_solvation": r.dg_born_solvation,
        "dg_mixing_entropy": r.dg_mixing_entropy,
        "dg_dispersion": r.dg_dispersion_t2,
        "dg_pocket_desolv": r.dg_cavity_dehydration,
        "dg_water_displacement": r.dg_size_mismatch,
        "dg_preorganization": r.dg_shape,
        "dg_total": dg_net,
        "log_Ka_pred": -dg_net / (2.303 * 8.314e-3 * 298.15),
    }