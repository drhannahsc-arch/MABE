"""
core/universal_predictor.py — Sprint 37: Universal ΔG Predictor

Routes any UniversalComplex through the appropriate energy terms.
Metal coordination → existing 18-term physics engine.
Host-guest / protein-ligand → existing terms + 5 new non-covalent terms.

The key: ONE set of physics parameters across ALL modalities.
Desolvation is desolvation whether it is Cu²⁺ shedding water or
adamantane shedding water. The γ for hydrophobic burial is the same
whether the cavity is a cyclodextrin or an enzyme active site.

Returns per-term decomposition for residual analysis.
"""
import math
from dataclasses import dataclass, field

from core.universal_schema import UniversalComplex


# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — these are what the back-solve engine optimizes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsParameters:
    """All tunable parameters in one object.

    Phases 1-5: metal-specific (from original back-solve protocol).
    Phases 6-10: universal non-covalent (from expanded protocol).

    Back-solve engine optimizes these against the full dataset.
    """
    # ── Phase 1-5: Metal coordination (existing, preserved) ───────────
    # Exchange energies per donor subtype (kJ/mol)
    exchange_O_ether: float = -1.0
    exchange_O_hydroxyl: float = -5.0
    exchange_O_carboxylate: float = -4.0
    exchange_O_phenolate: float = -10.0
    exchange_O_hydroxamate: float = -18.0
    exchange_O_catecholate: float = -30.0
    exchange_N_amine: float = -6.0
    exchange_N_imine: float = -8.0
    exchange_N_pyridine: float = -7.0
    exchange_N_imidazole: float = -8.0
    exchange_S_thioether: float = -10.0
    exchange_S_thiosulfate: float = -6.0
    exchange_S_thiolate: float = -18.0
    exchange_S_dithiocarbamate: float = -10.0
    exchange_O_element: float = -4.0
    exchange_N_element: float = -6.0
    exchange_S_element: float = -18.0
    exchange_P_element: float = -15.0
    exchange_Cl_element: float = -3.0
    exchange_Br_element: float = -6.0
    exchange_I_element: float = -12.0

    # Charge scaling
    alpha_charge: float = -5.0          # kJ/mol per (z²-1)/n_donors

    # Desolvation
    base_f_monovalent: float = 0.04
    base_f_divalent: float = 0.025
    base_f_trivalent: float = 0.015

    # z·z dielectric
    epsilon_eff: float = 10.0

    # Chelate effect
    chelate_base_d: float = -12.0       # d-block metals
    chelate_base_s: float = -8.0        # s-block / d0/d10

    # Ring strain (Hancock)
    k_strain: float = 80.0             # kJ/mol/nm

    # Macrocyclic
    k_macro_O: float = -1.2            # per O donor in macrocycle
    k_macro_N: float = -2.5            # per N donor
    k_macro_other: float = -1.5
    sigma_cavity: float = 0.03         # nm, Gaussian width for size-match

    # LFSE scaling
    lfse_scale: float = 1.0

    # Jahn-Teller
    jt_4coord: float = -25.0
    jt_6coord: float = -10.0

    # Translational entropy
    dg_translational_per_mol: float = 5.5  # kJ/mol per ligand molecule

    # ── Phase 6: Hydrophobic transfer ─────────────────────────────────
    gamma_hydrophobic: float = 0.025    # kJ/(mol·Å²), water release per SASA buried
    k_curvature: float = 0.5           # Amplification for concave cavities (Å)
    polar_discount: float = 0.3        # Fraction: polar SASA contributes this × γ

    # ── Phase 7: H-bond network ───────────────────────────────────────
    epsilon_hbond_neutral: float = -3.0     # kJ/mol per neutral H-bond
    epsilon_hbond_charged: float = -10.0    # kJ/mol per charge-assisted H-bond
    epsilon_hbond_OH_pi: float = -2.0       # kJ/mol per OH-π
    epsilon_water_hbond: float = 4.0        # kJ/mol penalty per broken water HB
    k_cooperativity: float = 1.1            # Multiplier for networks ≥ 3 HBs
    theta_half: float = 35.0               # degrees from linear at half-strength
    # Tiered H-bond strengths (Sprint 40)
    epsilon_hbond_strong: float = -8.0      # kJ/mol sulfonamide/amide/guanidine donors, carboxylate acceptors
    epsilon_hbond_moderate: float = -4.0    # kJ/mol hydroxyl, amine donors, carbonyl acceptors
    epsilon_hbond_weak: float = -1.5        # kJ/mol aromatic NH, thiol donors, aromatic N acceptors

    # ── Phase 8: π-interactions ───────────────────────────────────────
    epsilon_pi_parallel: float = -4.0       # kJ/mol per parallel π-π contact
    epsilon_pi_T_shaped: float = -3.0       # kJ/mol per T-shaped
    epsilon_CH_pi: float = -1.5             # kJ/mol per CH-π contact
    epsilon_cation_pi: float = -5.0         # kJ/mol per cation-π

    # ── Phase 9: Conformational entropy ───────────────────────────────
    TdS_per_rotor: float = 2.9             # kJ/mol per frozen rotatable bond (298K)
    ring_correction: float = 0.4            # Fraction: cyclic guests pre-restricted

    # ── Phase 10: Shape complementarity ───────────────────────────────
    PC_optimal: float = 0.55               # Rebek packing coefficient
    sigma_PC: float = 0.10                 # Width of Gaussian
    k_shape: float = -10.0                 # kJ/mol maximum shape stabilization
    k_clash: float = 100.0                 # kJ/mol per nm³ steric overlap

    # ── Phase 11: Lipophilic transfer (Sprint 40) ────────────────────
    epsilon_logP: float = -0.5             # kJ/mol per logP unit (hydrophobic contribution)
    k_mw_penalty: float = 0.003            # kJ/mol per Da above 300 (size penalty)

    def to_vector(self):
        """Flatten all parameters to a numpy-compatible list for optimization."""
        return [getattr(self, f.name) for f in self.__dataclass_fields__.values()
                if isinstance(getattr(self, f.name), (int, float))]

    @classmethod
    def from_vector(cls, vec):
        """Reconstruct from a flat parameter vector."""
        p = cls()
        idx = 0
        for f in cls.__dataclass_fields__.values():
            if isinstance(getattr(p, f.name), (int, float)):
                setattr(p, f.name, vec[idx])
                idx += 1
        return p

    @classmethod
    def param_names(cls):
        """Return ordered list of parameter names matching to_vector order."""
        p = cls()
        return [f.name for f in cls.__dataclass_fields__.values()
                if isinstance(getattr(p, f.name), (int, float))]

    @classmethod
    def param_count(cls):
        return len(cls.param_names())


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """Per-term energy decomposition for one UniversalComplex."""
    name: str
    binding_mode: str
    log_Ka_exp: float
    log_Ka_pred: float
    dg_pred_kj: float
    error: float               # pred - exp in log Ka

    # Per-term breakdown (kJ/mol, negative = favorable)
    dg_bind: float = 0.0             # Donor exchange (metal)
    dg_desolv: float = 0.0           # Desolvation (universal)
    dg_chelate: float = 0.0          # Chelate effect (metal)
    dg_ring_strain: float = 0.0      # Hancock rule (metal)
    dg_electrostatic: float = 0.0    # z·z Coulombic (charged species)
    dg_macrocyclic: float = 0.0      # Preorganization (macrocyclic hosts)
    dg_lfse: float = 0.0             # d-electron stabilization (metal)
    dg_jahn_teller: float = 0.0      # d9/d4 (metal)
    dg_protonation: float = 0.0      # pH competition (universal)
    dg_translational: float = 0.0    # Association entropy (universal)
    dg_activity: float = 0.0         # Ionic strength (universal)
    # New universal terms
    dg_hydrophobic: float = 0.0      # SASA burial (host-guest, protein)
    dg_hbond: float = 0.0            # H-bond network (host-guest, protein)
    dg_pi: float = 0.0               # π-interactions (aromatic hosts)
    dg_conf_entropy: float = 0.0     # Rotor freezing (flexible guests)
    dg_shape: float = 0.0            # Packing complementarity (cavity hosts)
    dg_lipophilic: float = 0.0        # LogP-based lipophilic transfer (Sprint 40)
    # Other existing
    dg_dispersion: float = 0.0
    dg_covalent: float = 0.0
    dg_polarization: float = 0.0
    dg_relativistic: float = 0.0
    dg_preorg: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def predict(uc, params=None):
    """Predict ΔG and log Ka for any UniversalComplex.

    Routes through metal terms (for coordination) and/or universal terms
    (for all modalities) based on binding_mode.

    Args:
        uc: UniversalComplex with all available annotations
        params: PhysicsParameters (uses defaults if None)

    Returns:
        PredictionResult with full per-term decomposition
    """
    if params is None:
        params = PhysicsParameters()

    # Detect routing
    is_metal = uc.is_metal()
    is_hg = uc.is_host_guest()
    is_pl = uc.is_protein_ligand()
    has_cavity = uc.cavity_volume_A3 > 0

    result = PredictionResult(
        name=uc.name,
        binding_mode=uc.binding_mode,
        log_Ka_exp=uc.log_Ka_exp,
        log_Ka_pred=0.0,
        dg_pred_kj=0.0,
        error=0.0,
    )

    # ── METAL COORDINATION TERMS (active when metal_formula set) ──────
    if is_metal:
        result = _compute_metal_terms(uc, params, result)

    # ── UNIVERSAL NON-COVALENT TERMS (active for all with guest data) ─
    if not is_metal or is_hg:
        result = _compute_hydrophobic(uc, params, result)
        result = _compute_hbond(uc, params, result)
        result = _compute_pi(uc, params, result)
        result = _compute_conf_entropy(uc, params, result)
        result = _compute_shape(uc, params, result)
        result = _compute_lipophilic(uc, params, result)

    # ── UNIVERSAL TERMS (always active) ───────────────────────────────
    result = _compute_desolvation_guest(uc, params, result)
    result = _compute_electrostatic_general(uc, params, result)
    result = _compute_translational(uc, params, result)
    result = _compute_activity(uc, params, result)

    # ── SUM AND CONVERT ───────────────────────────────────────────────
    dg_net = (result.dg_bind + result.dg_desolv + result.dg_chelate
              + result.dg_ring_strain + result.dg_electrostatic
              + result.dg_macrocyclic + result.dg_lfse + result.dg_jahn_teller
              + result.dg_protonation + result.dg_translational
              + result.dg_activity + result.dg_preorg
              + result.dg_hydrophobic + result.dg_hbond
              + result.dg_pi + result.dg_conf_entropy + result.dg_shape
              + result.dg_lipophilic
              + result.dg_dispersion + result.dg_covalent
              + result.dg_polarization + result.dg_relativistic)

    result.dg_pred_kj = dg_net
    result.log_Ka_pred = -dg_net / 5.71
    result.error = result.log_Ka_pred - uc.log_Ka_exp

    return result


# ═══════════════════════════════════════════════════════════════════════════
# METAL COORDINATION TERMS
# (Extracted from compute_enhanced_thermodynamics, parameterized)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metal_terms(uc, params, result):
    """Compute all metal-specific energy terms.

    Sprint 40: Self-contained implementation using PhysicsParameters.
    All terms are back-solvable through the universal optimizer.
    """
    from metal_physics import compute_metal_physics

    terms = compute_metal_physics(uc, params)

    result.dg_bind = terms["dg_bind"]
    result.dg_desolv = terms["dg_desolv"]
    result.dg_chelate = terms["dg_chelate"]
    result.dg_ring_strain = terms["dg_ring_strain"]
    result.dg_electrostatic = terms["dg_electrostatic"]
    result.dg_lfse = terms["dg_lfse"]
    result.dg_jahn_teller = terms["dg_jahn_teller"]
    result.dg_macrocyclic = terms["dg_macrocyclic"]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL NON-COVALENT TERMS (Phases 6–10)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_hydrophobic(uc, params, result):
    """Phase 6: Hydrophobic transfer from SASA burial."""
    if uc.sasa_buried_A2 > 0:
        sasa = uc.sasa_buried_A2
    elif uc.guest_sasa_nonpolar_A2 > 0:
        from core.guest_compute import estimate_sasa_burial
        sasa = estimate_sasa_burial(
            uc.guest_sasa_nonpolar_A2,
            uc.cavity_radius_nm,
            binding_mode=uc.binding_mode)
    else:
        sasa = 0.0

    if sasa <= 0:
        return result

    # Curvature correction: concave cavities amplify γ
    gamma = params.gamma_hydrophobic
    if uc.cavity_radius_nm > 0:
        r_A = uc.cavity_radius_nm * 10.0
        gamma *= (1.0 + params.k_curvature / max(1.0, r_A))

    # Polar SASA discount
    if uc.guest_sasa_polar_A2 > 0 and uc.guest_sasa_total_A2 > 0:
        polar_frac = uc.guest_sasa_polar_A2 / uc.guest_sasa_total_A2
        effective_sasa = sasa * (1.0 - polar_frac * (1.0 - params.polar_discount))
    else:
        effective_sasa = sasa

    result.dg_hydrophobic = -gamma * effective_sasa
    return result


def _compute_hbond(uc, params, result):
    """Phase 7: H-bond network energy."""
    if uc.n_hbonds_formed <= 0:
        return result

    dg = 0.0
    for hbt in uc.hbond_types:
        if hbt == "charge_assisted":
            dg += params.epsilon_hbond_charged
        elif hbt == "OH_pi":
            dg += params.epsilon_hbond_OH_pi
        elif hbt == "strong":
            dg += params.epsilon_hbond_strong
        elif hbt == "moderate":
            dg += params.epsilon_hbond_moderate
        elif hbt == "weak":
            dg += params.epsilon_hbond_weak
        else:
            dg += params.epsilon_hbond_neutral

    # If types not annotated, use count with neutral default
    if not uc.hbond_types:
        dg = uc.n_hbonds_formed * params.epsilon_hbond_neutral

    # Cooperativity for large networks
    if uc.n_hbonds_formed >= 3:
        dg *= params.k_cooperativity

    # Water H-bond penalty: each host/guest H-bond site was H-bonding water
    # Net effect already partially captured by γ × polar_discount
    # This is the ADDITIONAL penalty for H-bond sites that now point into cavity
    n_broken = min(uc.n_hbonds_formed, uc.guest_n_hbond_donors + uc.guest_n_hbond_acceptors)
    dg += n_broken * params.epsilon_water_hbond * 0.3  # 30% not recovered

    result.dg_hbond = dg
    return result


def _compute_pi(uc, params, result):
    """Phase 8: π-interactions."""
    if uc.n_pi_contacts <= 0 and uc.n_aromatic_walls == 0:
        return result

    dg = 0.0
    for pit in uc.pi_contact_types:
        if pit == "parallel_pp":
            dg += params.epsilon_pi_parallel
        elif pit == "t_shaped":
            dg += params.epsilon_pi_T_shaped
        elif pit == "cation_pi":
            dg += params.epsilon_cation_pi
        elif pit == "CH_pi":
            dg += params.epsilon_CH_pi
        else:
            dg += params.epsilon_pi_parallel  # default

    # If types not annotated but aromatic walls known, estimate
    if not uc.pi_contact_types and uc.n_aromatic_walls > 0:
        # Aromatic guest in aromatic host → π-π
        if uc.guest_n_aromatic_rings > 0:
            n_contacts = min(uc.guest_n_aromatic_rings, uc.n_aromatic_walls)
            dg += n_contacts * params.epsilon_pi_parallel
        # Charged guest near aromatic walls → cation-π
        if uc.guest_charge > 0:
            dg += params.epsilon_cation_pi
        # Aliphatic guest in aromatic host → CH-π
        if uc.guest_n_aromatic_rings == 0 and uc.n_aromatic_walls > 0:
            dg += uc.n_aromatic_walls * params.epsilon_CH_pi * 0.5

    result.dg_pi = dg
    return result


def _compute_conf_entropy(uc, params, result):
    """Phase 9: Conformational entropy penalty for flexible guests."""
    n_rotors = uc.guest_rotatable_bonds
    if n_rotors <= 0:
        return result

    # Assume all rotors freeze upon binding (conservative)
    penalty = params.TdS_per_rotor * n_rotors

    # Ring correction: cyclic portions already restricted
    if uc.guest_n_aromatic_rings > 0 or uc.is_macrocyclic:
        penalty *= params.ring_correction

    result.dg_conf_entropy = penalty
    return result


def _compute_shape(uc, params, result):
    """Phase 10: Shape complementarity (packing coefficient)."""
    pc = uc.packing_coefficient
    if pc <= 0 or uc.cavity_volume_A3 <= 0:
        return result

    # Gaussian shape stabilization centered on PC_optimal
    shape_match = math.exp(-((pc - params.PC_optimal)**2) / (2 * params.sigma_PC**2))
    dg_shape = params.k_shape * shape_match

    # Steric clash for overpacked guests
    if pc > 0.80:
        overlap_frac = pc - 0.80
        overlap_vol_nm3 = overlap_frac * uc.cavity_volume_A3 / 1000.0  # Å³ → nm³
        dg_shape += params.k_clash * overlap_vol_nm3

    result.dg_shape = dg_shape
    return result


def _compute_lipophilic(uc, params, result):
    """Phase 11: Lipophilic transfer free energy (Sprint 40).

    logP captures the free energy of transfer from water to octanol.
    For protein binding, this transfers into the pocket:
      dG_lipo = epsilon_logP * logP * burial_fraction

    Also applies a molecular weight penalty: large compounds pay
    entropic costs that scale beyond what TdS_per_rotor captures
    (restricted translational/rotational modes, not just torsions).
    """
    if uc.guest_logP == 0.0 and uc.guest_mw == 0.0:
        return result

    dg = 0.0

    # LogP-based lipophilic contribution
    if uc.guest_logP != 0.0 and uc.cavity_volume_A3 > 0:
        # Modulate by how much of the compound is actually buried
        if uc.sasa_buried_A2 > 0 and uc.guest_sasa_total_A2 > 0:
            burial_frac = min(1.0, uc.sasa_buried_A2 / max(1.0, uc.guest_sasa_total_A2))
        else:
            burial_frac = 0.5  # default

        dg += params.epsilon_logP * uc.guest_logP * burial_frac

    # MW penalty for large compounds (above "drug-like" threshold of 300 Da)
    if uc.guest_mw > 300:
        excess_mw = uc.guest_mw - 300.0
        dg += params.k_mw_penalty * excess_mw

    result.dg_lipophilic = dg
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ALWAYS-ACTIVE UNIVERSAL TERMS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_desolvation_guest(uc, params, result):
    """Guest desolvation: non-metal guests also pay to shed water."""
    if uc.is_metal():
        return result  # Metal desolvation handled in _compute_metal_terms

    # For host-guest: guest desolvation ≈ fraction of SASA buried × water release
    # This is partially captured by hydrophobic term. For POLAR desolvation:
    if uc.guest_sasa_polar_A2 > 0 and uc.cavity_volume_A3 > 0:
        # Polar desolvation penalty: stripping H-bonded water from polar guest surface
        polar_burial = uc.guest_sasa_polar_A2 * 0.3  # ~30% polar surface buried
        dg = polar_burial * 0.015  # ~0.015 kJ/mol/Å² polar desolvation cost
        result.dg_desolv += dg

    return result


def _compute_electrostatic_general(uc, params, result):
    """General electrostatic for non-metal charged species."""
    if uc.is_metal():
        return result  # Metal electrostatics handled in _compute_metal_terms

    if uc.guest_charge != 0 and uc.host_charge != 0:
        # Coulombic attraction/repulsion between charged host and guest
        # Simplified: point charges at cavity center
        if uc.cavity_radius_nm > 0:
            r_pm = uc.cavity_radius_nm * 1000  # nm → pm
            k_elec = 1389.4 / params.epsilon_eff
            dg = k_elec * uc.guest_charge * uc.host_charge / max(100, r_pm)
            result.dg_electrostatic += dg

    # Ion-dipole for charged guest in neutral polar host (CD, CB portals)
    if uc.guest_charge != 0 and uc.n_hbond_acceptors_host > 0 and uc.host_charge == 0:
        # Partial negative charges on acceptors attract cationic guests
        n_acc = min(uc.n_hbond_acceptors_host, 8)  # Cap contribution
        dg = -0.5 * abs(uc.guest_charge) * n_acc  # ~-0.5 kJ/mol per acceptor-charge
        result.dg_electrostatic += dg

    return result


def _compute_translational(uc, params, result):
    """Translational entropy of association: always applies."""
    # 1:1 binding → one translational degree of freedom lost
    result.dg_translational = params.dg_translational_per_mol
    return result


def _compute_activity(uc, params, result):
    """Ionic strength correction (Davies equation)."""
    I = uc.ionic_strength_M
    z = abs(uc.guest_charge) if uc.guest_charge != 0 else abs(uc.host_charge)
    if I > 0 and z > 0:
        sqrt_I = math.sqrt(I)
        log_gamma = -0.509 * z**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)
        result.dg_activity = -5.71 * log_gamma
    return result


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_batch(entries, params=None):
    """Predict log Ka for a list of UniversalComplex entries.

    Returns list of PredictionResult.
    """
    if params is None:
        params = PhysicsParameters()

    results = []
    for uc in entries:
        try:
            r = predict(uc, params)
        except Exception as e:
            r = PredictionResult(
                name=uc.name, binding_mode=uc.binding_mode,
                log_Ka_exp=uc.log_Ka_exp, log_Ka_pred=0.0,
                dg_pred_kj=0.0, error=-uc.log_Ka_exp)
        results.append(r)

    return results


def compute_statistics(results):
    """Compute R², MAE, bias from a list of PredictionResult."""
    if not results:
        return {"r2": 0, "mae": 0, "bias": 0, "n": 0}

    import statistics
    exps = [r.log_Ka_exp for r in results]
    preds = [r.log_Ka_pred for r in results]
    errors = [r.error for r in results]
    abs_errors = [abs(e) for e in errors]

    n = len(results)
    mae = sum(abs_errors) / n
    bias = sum(errors) / n

    # R²
    mean_exp = sum(exps) / n
    ss_tot = sum((e - mean_exp)**2 for e in exps)
    ss_res = sum((p - e)**2 for p, e in zip(preds, exps))
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {"r2": round(r2, 4), "mae": round(mae, 2),
            "bias": round(bias, 2), "n": n}
