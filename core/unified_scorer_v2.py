"""
core/unified_scorer_v2.py — Phase 13b: Unified Scorer (No Routing Wall)

Single entry point: predict(uc) → PredictionResult
All energy terms fire independently with self-zeroing guards.
Delegates to calibrated backends:
  - scorer_frozen.predict_log_k() for metal coordination (17 terms → 1 bundled)
  - hg_scorer energy functions for host-guest (hydrophobic, dehydration, H-bond,
    π, conformational entropy, shape)
  - cross_modal_predictor energy functions for metal@CB[n] (ion-dipole,
    desolvation, portal size, CB dehydration, shape)

Backward compatibility: 644 regression tests must pass within ε=0.01 log K.
"""

import sys
import os
import math
from dataclasses import dataclass

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
for _sub in ('knowledge', 'core'):
    _p = os.path.join(_project_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.universal_schema import UniversalComplex

# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATED BACKENDS (imported at module level for speed)
# ═══════════════════════════════════════════════════════════════════════════
from core.scorer_frozen import predict_log_k as _metal_predict_log_k

from hg_scorer import (
    HG_PARAMS, LN10_RT,
    compute_guest_sasa, estimate_buried_sasa,
    dg_hydrophobic as _hg_dg_hydrophobic,
    dg_cavity_dehydration as _hg_dg_cavity_dehydration,
    dg_size_mismatch as _hg_dg_size_mismatch,
)
from knowledge.hg_hbond import compute_dg_hbond as _hg_compute_dg_hbond, HBOND_PARAMS
from knowledge.hg_pi import compute_dg_pi as _hg_compute_dg_pi, PI_PARAMS
from knowledge.hg_conf_shape import (
    compute_dg_conf_shape as _hg_compute_dg_conf_shape,
    CONF_SHAPE_PARAMS, HOST_CAVITY_VOLUME,
)
from knowledge.hg_dataset import HOST_DB as _HG_HOST_DB

from cross_modal_predictor import (
    dg_ion_dipole as _cm_dg_ion_dipole,
    dg_desolvation as _cm_dg_desolvation,
    dg_portal_size_match as _cm_dg_portal_size_match,
    dg_cb_dehydration as _cm_dg_cb_dehydration,
    dg_shape_complementarity as _cm_dg_shape_complementarity,
    CM_PARAMS,
)
from knowledge.cross_modal_dataset import CB_PORTAL_INFO


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """Full energy decomposition for one UniversalComplex."""
    name: str
    binding_mode: str
    log_Ka_exp: float
    log_Ka_pred: float
    dg_total_kj: float
    error: float

    # Metal coordination (bundled — 17 internal terms)
    dg_metal: float = 0.0

    # Host-guest inclusion
    dg_hydrophobic: float = 0.0
    dg_cavity_dehydration: float = 0.0
    dg_hbond: float = 0.0
    dg_pi: float = 0.0
    dg_conf_entropy: float = 0.0
    dg_shape: float = 0.0
    dg_size_mismatch: float = 0.0

    # Cross-modal (metal@host)
    dg_ion_dipole: float = 0.0
    dg_ion_desolv: float = 0.0
    dg_portal_size: float = 0.0
    dg_cm_dehydration: float = 0.0
    dg_cm_shape: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# HOST-TYPE INFERENCE MAPS
# Used when scoring from a UC that has host_name/host_type but no raw dict
# ═══════════════════════════════════════════════════════════════════════════

# Map host keys to hg_dataset HOST_DB keys
_HOST_KEY_MAP = {
    # Cyclodextrins
    "α-cyclodextrin": "alpha-CD", "alpha-CD": "alpha-CD",
    "β-cyclodextrin": "beta-CD",  "beta-CD": "beta-CD",
    "γ-cyclodextrin": "gamma-CD", "gamma-CD": "gamma-CD",
    # Cucurbiturils (HOST_REGISTRY full names + bracket variants)
    "cucurbit[5]uril": "CB5", "CB5": "CB5", "CB[5]": "CB5",
    "cucurbit[6]uril": "CB6", "CB6": "CB6", "CB[6]": "CB6",
    "cucurbit[7]uril": "CB7", "CB7": "CB7", "CB[7]": "CB7",
    "cucurbit[8]uril": "CB8", "CB8": "CB8", "CB[8]": "CB8",
    # Calixarenes
    "p-sulfonatocalix[4]arene": "calix4-SO3", "calix4-SO3": "calix4-SO3",
    "sulfonato-calix4arene": "calix4-SO3",
    # Pillararenes
    "pillar[5]arene": "pillar5", "pillar5": "pillar5",
    "pillar5arene": "pillar5",
    # Crown ethers / cryptands (no HOST_DB entries yet — mapped for future)
    "12-crown-4": "12-crown-4", "15-crown-5": "15-crown-5",
    "18-crown-6": "18-crown-6", "[2.2.2]cryptand": "cryptand-222",
    "cryptand-222": "cryptand-222",
}


def _resolve_host_key(uc):
    """Try to resolve a HOST_DB key from UC host_name."""
    for candidate in (uc.host_name, uc.host_type):
        if candidate in _HG_HOST_DB:
            return candidate
        if candidate in _HOST_KEY_MAP:
            return _HOST_KEY_MAP[candidate]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER: UC → hg_scorer dict format
# ═══════════════════════════════════════════════════════════════════════════

def _uc_to_hg_entry(uc, host_key):
    """Build a dict compatible with hg_scorer.predict_hg_log_ka()."""
    return {
        "name": uc.name,
        "host": host_key,
        "guest_smiles": uc.guest_smiles,
        "guest_charge": uc.guest_charge,
        "n_hbonds_portal": uc.n_hbonds_formed,
        "guest_has_cation": uc.guest_charge > 0,
        "log_Ka": uc.log_Ka_exp,
        "source": getattr(uc, 'source', ''),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER: UC → cross_modal_predictor dict format
# ═══════════════════════════════════════════════════════════════════════════

def _uc_to_cm_entry(uc, cb_host_key):
    """Build a dict compatible with cross_modal_predictor functions."""
    portal = CB_PORTAL_INFO.get(cb_host_key, {})
    return {
        "name": uc.name,
        "metal": uc.metal_formula,
        "cb_host": cb_host_key,
        "n_portal_carbonyls": portal.get("n_carbonyls", 0),
        "portal_radius_nm": portal.get("portal_radius_nm", 0.0),
        "cavity_sasa": portal.get("cavity_sasa", 0.0),
        "cavity_volume_A3": uc.cavity_volume_A3,
        "log_Ka": uc.log_Ka_exp,
        "mode": "cross_modal",
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTION ENGINE — NO ROUTING WALL
# ═══════════════════════════════════════════════════════════════════════════

def predict(uc, verbose=False):
    """Score any UniversalComplex. All terms self-zero when inputs absent.

    Args:
        uc: UniversalComplex (from auto_descriptor or manual construction)
        verbose: print energy decomposition

    Returns:
        PredictionResult with full term breakdown
    """
    result = PredictionResult(
        name=uc.name,
        binding_mode=uc.binding_mode,
        log_Ka_exp=uc.log_Ka_exp,
        log_Ka_pred=0.0,
        dg_total_kj=0.0,
        error=0.0,
    )

    # ── METAL COORDINATION (self-zeros if no metal) ──────────────────
    _compute_metal(uc, result)

    # ── HOST-GUEST INCLUSION (self-zeros if no cavity/guest) ─────────
    _compute_hg_terms(uc, result)

    # ── PROTEIN-LIGAND non-covalent (self-zeros if not metalloprotein) ─
    _compute_protein_ligand_terms(uc, result)

    # ── GENERAL PROTEIN-LIGAND (non-metal targets) ────────────────────
    _compute_general_pl_terms(uc, result)

    # ── CROSS-MODAL metal@host (self-zeros if no metal+cavity) ───────
    _compute_cm_terms(uc, result)

    # ── SUM AND CONVERT ──────────────────────────────────────────────
    dg_net = (result.dg_metal
              + result.dg_hydrophobic + result.dg_cavity_dehydration
              + result.dg_hbond + result.dg_pi
              + result.dg_conf_entropy + result.dg_shape
              + result.dg_size_mismatch
              + result.dg_ion_dipole + result.dg_ion_desolv
              + result.dg_portal_size + result.dg_cm_dehydration
              + result.dg_cm_shape)

    result.dg_total_kj = dg_net
    result.log_Ka_pred = -dg_net / LN10_RT
    result.error = result.log_Ka_pred - uc.log_Ka_exp

    if verbose:
        _print_decomposition(uc, result)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# METAL COORDINATION — delegates to scorer_frozen
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metal(uc, result):
    """Metal coordination: 17 physics terms bundled into dg_metal.

    Self-zeros if uc.metal_formula is empty or donor_subtypes is empty.
    """
    if not uc.metal_formula or not uc.donor_subtypes:
        return

    # Only fire for metal_coordination binding mode (not for metal@host
    # where the CM terms handle it). If binding_mode is cross_modal,
    # the metal scoring path should NOT fire — the CM scorer handles it.
    if uc.binding_mode == "cross_modal":
        return

    try:
        log_k = _metal_predict_log_k(
            uc.metal_formula,
            uc.donor_subtypes,
            chelate_rings=uc.chelate_rings,
            ring_sizes=uc.ring_sizes or None,
            pH=uc.ph,
            is_macrocyclic=uc.is_macrocyclic,
            cavity_radius_nm=uc.cavity_radius_nm if uc.cavity_radius_nm > 0 else None,
            n_ligand_molecules=uc.n_ligand_molecules,
            temperature_K=298.15,
        )
        # Convert log K back to ΔG for decomposition
        result.dg_metal = -log_k * LN10_RT
    except (ValueError, KeyError):
        result.dg_metal = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# HOST-GUEST TERMS — delegates to hg_scorer functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_hg_terms(uc, result):
    """Host-guest inclusion terms. Self-zeros if no guest SMILES or no host.

    Fires for binding_mode = host_guest_inclusion or mixed.
    Does NOT fire for cross_modal (metal@CB) — those are handled by CM terms.
    Does NOT fire for pure metal_coordination (no cavity).
    """
    if uc.binding_mode == "cross_modal":
        return
    if not uc.guest_smiles:
        return

    host_key = _resolve_host_key(uc)
    if host_key is None or host_key not in _HG_HOST_DB:
        return

    host = _HG_HOST_DB[host_key]
    hg_entry = _uc_to_hg_entry(uc, host_key)

    # Guest SASA
    try:
        sasa = compute_guest_sasa(uc.guest_smiles)
    except Exception:
        return

    guest_np = sasa["nonpolar_sasa"]
    cavity_sasa = host["cavity_sasa"]
    buried = estimate_buried_sasa(guest_np, cavity_sasa)

    # 1. Hydrophobic transfer
    result.dg_hydrophobic = _hg_dg_hydrophobic(buried, host["curvature_class"])

    # 2. Cavity dehydration
    result.dg_cavity_dehydration = _hg_dg_cavity_dehydration(buried, host_key)

    # 3. Size mismatch
    result.dg_size_mismatch = _hg_dg_size_mismatch(guest_np, cavity_sasa)

    # 4. H-bond network
    try:
        result.dg_hbond = _hg_compute_dg_hbond(hg_entry, _HG_HOST_DB)
    except Exception:
        result.dg_hbond = 0.0

    # 5. π-interactions
    try:
        result.dg_pi = _hg_compute_dg_pi(hg_entry, _HG_HOST_DB)
    except Exception:
        result.dg_pi = 0.0

    # 6+7. Conformational entropy + Shape complementarity
    try:
        dg_conf, dg_shape = _hg_compute_dg_conf_shape(hg_entry)
        result.dg_conf_entropy = dg_conf
        result.dg_shape = dg_shape
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# PROTEIN-LIGAND NON-COVALENT TERMS — Phase 14a + PL Calibration
# 5 physics descriptors + per-target offsets, fitted against 300 ChEMBL entries.
# 11 params total, 300 data = 27:1 ratio. Metal scorer untouched.
# ═══════════════════════════════════════════════════════════════════════════

PL_PARAMS = {
    "a_sasa":  -0.05825,   # kJ/mol per Å² buried SASA (2.3× HG γ_flat)
    "b_logP":   0.2011,    # log Ka per logP unit (lipophilic efficiency)
    "c_rot":   -0.1362,    # log Ka per rotatable bond (size proxy)
    "d_hb":     0.2453,    # log Ka per H-bond count (heuristic correction)
    "e_mw":     0.5277,    # log Ka per 100 Da (molecular size)
}

# Per-target offsets absorb systematic difference between metal scorer
# absolute prediction and protein-context binding. For novel targets
# without a calibrated offset, the model falls back to offset=0.
PL_TARGET_OFFSETS = {
    "ACE":         +3.360,
    "CA-II":       +0.441,
    "MMP-13":      +1.662,
    "MMP-7":       -0.092,
    "MMP-9":       +1.699,
    "Thermolysin": +0.947,
}


def _compute_protein_ligand_terms(uc, result):
    """Non-covalent terms for metalloprotein-ligand binding.

    Fires only for binding_mode == 'metalloprotein'.
    Uses 5 calibrated PL descriptors + per-target offset.
    Metal coordination term is computed separately by _compute_metal.

    Self-zeros if binding_mode != metalloprotein or guest properties absent.
    """
    if uc.binding_mode != "metalloprotein":
        return
    if not uc.guest_smiles or uc.guest_sasa_nonpolar_A2 <= 0:
        return

    p = PL_PARAMS

    # 1. Hydrophobic burial: SASA-based
    if uc.sasa_buried_A2 > 0:
        result.dg_hydrophobic = p["a_sasa"] * uc.sasa_buried_A2

    # 2–5. logP, rotors, H-bonds, MW → converted to kJ/mol
    dg_descriptors = (
        p["b_logP"] * uc.guest_logP * LN10_RT
        + p["c_rot"] * uc.guest_rotatable_bonds * LN10_RT
        + p["d_hb"] * uc.n_hbonds_formed * LN10_RT
        + p["e_mw"] * uc.guest_mw * 0.01
    )
    # Pack descriptor terms into conf_entropy (reuse existing result field)
    result.dg_conf_entropy = dg_descriptors

    # Per-target offset (falls back to 0 for unknown targets)
    target = uc.host_name
    offset = PL_TARGET_OFFSETS.get(target, 0.0)
    result.dg_shape = offset * LN10_RT


# ═══════════════════════════════════════════════════════════════════════════
# GENERAL PROTEIN-LIGAND (non-metal targets) — Phase 18
# 8 2D descriptors + per-target offsets, fitted against 2000 ChEMBL entries
# across 5 non-metal targets (DHFR, HIV protease, COX-2, trypsin, thrombin).
# No metal term. Binding driven entirely by non-covalent physics.
# ═══════════════════════════════════════════════════════════════════════════

GENERAL_PL_PARAMS = {
    "a_logP":      +0.1928,    # legacy fallback — used only for unknown targets
    "b_mw":        +0.1988,
    "c_rot":       -0.0140,
    "d_hbd":       -0.0987,
    "e_hba":       +0.1028,
    "f_tpsa":      +0.1022,
    "g_arom":      -0.0345,
    "h_fsp3":      +0.5548,
}

GENERAL_PL_TARGET_OFFSETS = {
    "COX-2":                        +4.239,
    "Dihydrofolate reductase":      +4.436,
    "HIV-1 protease":               +5.523,
    "Thrombin":                     +4.660,
    "Trypsin":                      +5.284,
}

# ── Per-target Ridge models (20 features each) ──
# Calibrated on ChEMBL data (400-800 entries per target, 5-fold CV).
# Features: 8 whole-molecule + 4 Gasteiger charge stats + 8 topological indices.
# Average within-target Pearson r = 0.548 (vs 0.345 for 8-feature global model).
PER_TARGET_PL_MODELS = {
    "COX-2": {
        "intercept": 5.4032,
        "coefficients": {
            "logP": 0.223883, "mw": -0.102092, "rot": 0.071854,
            "hbd": 0.065111, "hba": 0.256415, "tpsa": -1.157652,
            "arom": 0.025584, "fsp3": 1.251802,
            "q_mean": -0.352824, "q_std": 0.541457,
            "q_min": -1.560208, "q_max": -3.900301,
            "chi1": -0.684645, "chi2n": 0.188214,
            "bertz": 0.433266, "hk_alpha": -0.160272,
            "kappa2": -0.114779, "kappa3": 0.522968,
            "aliph_rings": 0.845295, "sat_rings": -0.036815,
        },
    },
    "Dihydrofolate reductase": {
        "intercept": 6.6077,
        "coefficients": {
            "logP": 0.15907, "mw": -0.592046, "rot": 0.042493,
            "hbd": 0.125532, "hba": 0.002836, "tpsa": 0.311327,
            "arom": 0.143698, "fsp3": 0.084559,
            "q_mean": -0.1935, "q_std": 0.336999,
            "q_min": -0.402494, "q_max": 0.89356,
            "chi1": -0.043946, "chi2n": -0.263065,
            "bertz": 0.211869, "hk_alpha": 0.40603,
            "kappa2": 0.115548, "kappa3": 0.107576,
            "aliph_rings": 0.367138, "sat_rings": 0.093623,
        },
    },
    "HIV-1 protease": {
        "intercept": 6.4152,
        "coefficients": {
            "logP": 0.075736, "mw": -0.531338, "rot": -0.014523,
            "hbd": 0.062451, "hba": 0.12781, "tpsa": -0.201139,
            "arom": -0.347659, "fsp3": 0.185748,
            "q_mean": -0.182852, "q_std": -0.920776,
            "q_min": 0.793644, "q_max": 0.910586,
            "chi1": 0.494475, "chi2n": -0.158262,
            "bertz": 0.061937, "hk_alpha": 0.081766,
            "kappa2": -0.065421, "kappa3": -0.303736,
            "aliph_rings": -0.23828, "sat_rings": -0.00139,
        },
    },
    "Trypsin": {
        "intercept": 3.6302,
        "coefficients": {
            "logP": 0.131537, "mw": 0.186642, "rot": 0.134934,
            "hbd": 0.042008, "hba": 0.253747, "tpsa": -0.093819,
            "arom": -0.822433, "fsp3": -1.285639,
            "q_mean": 0.572056, "q_std": 3.688903,
            "q_min": -4.030153, "q_max": 4.313006,
            "chi1": -0.43908, "chi2n": 0.523403,
            "bertz": 0.245497, "hk_alpha": -0.128856,
            "kappa2": 0.00444, "kappa3": -0.30535,
            "aliph_rings": -0.610561, "sat_rings": 0.065483,
        },
    },
    "Thrombin": {
        "intercept": 6.7873,
        "coefficients": {
            "logP": -0.000913, "mw": -0.609443, "rot": 0.281582,
            "hbd": -0.303052, "hba": -0.383938, "tpsa": 1.75087,
            "arom": 0.207892, "fsp3": 0.446049,
            "q_mean": -0.491461, "q_std": -1.874569,
            "q_min": -0.075337, "q_max": 0.516235,
            "chi1": 0.281265, "chi2n": -0.053339,
            "bertz": 0.270676, "hk_alpha": 0.94006,
            "kappa2": -0.39282, "kappa3": 0.141772,
            "aliph_rings": 0.225482, "sat_rings": -0.045803,
        },
    },
}


def _compute_general_pl_terms(uc, result):
    """Non-covalent scoring for non-metal protein targets.

    Fires only for binding_mode == 'protein_ligand_general'.
    Uses per-target Ridge model (20 descriptors) if available,
    falls back to global 8-descriptor model for unknown targets.
    """
    if uc.binding_mode != "protein_ligand_general":
        return
    if not uc.guest_smiles:
        return

    target = uc.host_name
    model = PER_TARGET_PL_MODELS.get(target)

    if model:
        # Per-target model: 20 descriptors
        c = model["coefficients"]
        log_ka_pred = model["intercept"]
        log_ka_pred += c["logP"] * uc.guest_logP
        log_ka_pred += c["mw"] * uc.guest_mw / 100.0
        log_ka_pred += c["rot"] * uc.guest_rotatable_bonds
        log_ka_pred += c["hbd"] * uc.guest_n_hbond_donors
        log_ka_pred += c["hba"] * uc.guest_n_hbond_acceptors
        log_ka_pred += c["tpsa"] * getattr(uc, 'guest_tpsa', 0.0) / 100.0
        log_ka_pred += c["arom"] * getattr(uc, 'guest_n_aromatic_rings', 0)
        log_ka_pred += c["fsp3"] * getattr(uc, 'guest_fsp3', 0.0)
        # Gasteiger charge statistics
        log_ka_pred += c["q_mean"] * getattr(uc, 'guest_q_mean', 0.0)
        log_ka_pred += c["q_std"] * getattr(uc, 'guest_q_std', 0.0)
        log_ka_pred += c["q_min"] * getattr(uc, 'guest_q_min', 0.0)
        log_ka_pred += c["q_max"] * getattr(uc, 'guest_q_max', 0.0)
        # Topological shape descriptors
        log_ka_pred += c["chi1"] * getattr(uc, 'guest_chi1', 0.0)
        log_ka_pred += c["chi2n"] * getattr(uc, 'guest_chi2n', 0.0)
        log_ka_pred += c["bertz"] * getattr(uc, 'guest_bertz', 0.0) / 100.0
        log_ka_pred += c["hk_alpha"] * getattr(uc, 'guest_hk_alpha', 0.0)
        log_ka_pred += c["kappa2"] * getattr(uc, 'guest_kappa2', 0.0)
        log_ka_pred += c["kappa3"] * getattr(uc, 'guest_kappa3', 0.0)
        log_ka_pred += c["aliph_rings"] * getattr(uc, 'guest_n_aliphatic_rings', 0)
        log_ka_pred += c["sat_rings"] * getattr(uc, 'guest_n_saturated_rings', 0)
    else:
        # Fallback: global 8-descriptor model for unknown targets
        p = GENERAL_PL_PARAMS
        offset = GENERAL_PL_TARGET_OFFSETS.get(target, 4.5)
        log_ka_pred = (
            p["a_logP"] * uc.guest_logP
            + p["b_mw"] * uc.guest_mw / 100.0
            + p["c_rot"] * uc.guest_rotatable_bonds
            + p["d_hbd"] * uc.guest_n_hbond_donors
            + p["e_hba"] * uc.guest_n_hbond_acceptors
            + p["f_tpsa"] * getattr(uc, 'guest_tpsa', 0.0) / 100.0
            + p["g_arom"] * getattr(uc, 'guest_n_aromatic_rings', 0)
            + p["h_fsp3"] * getattr(uc, 'guest_fsp3', 0.0)
            + offset
        )

    # Convert to dG and decompose into result fields
    dg_total = -log_ka_pred * LN10_RT

    # Decompose into interpretable buckets
    result.dg_hydrophobic = -(
        (model["coefficients"]["logP"] if model else GENERAL_PL_PARAMS["a_logP"])
        * uc.guest_logP) * LN10_RT
    result.dg_hbond = -(
        ((model["coefficients"]["hbd"] * uc.guest_n_hbond_donors
          + model["coefficients"]["hba"] * uc.guest_n_hbond_acceptors)
         if model else
         (GENERAL_PL_PARAMS["d_hbd"] * uc.guest_n_hbond_donors
          + GENERAL_PL_PARAMS["e_hba"] * uc.guest_n_hbond_acceptors))
    ) * LN10_RT
    result.dg_conf_entropy = dg_total - result.dg_hydrophobic - result.dg_hbond
    # Use dg_metal slot for the total (since no metal term fires)
    result.dg_metal = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-MODAL TERMS — delegates to cross_modal_predictor functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_cm_terms(uc, result):
    """Cross-modal terms for metal@host (e.g. cation@CB[n]).

    Self-zeros if no metal_formula or binding_mode != cross_modal.
    """
    if uc.binding_mode != "cross_modal":
        return
    if not uc.metal_formula:
        return

    # Resolve CB host key
    cb_key = _resolve_cm_host(uc)
    if cb_key is None:
        return

    cm_entry = _uc_to_cm_entry(uc, cb_key)

    try:
        result.dg_ion_dipole = _cm_dg_ion_dipole(cm_entry)
        result.dg_ion_desolv = _cm_dg_desolvation(cm_entry)
        result.dg_portal_size = _cm_dg_portal_size_match(cm_entry)
        result.dg_cm_dehydration = _cm_dg_cb_dehydration(cm_entry)
        result.dg_cm_shape = _cm_dg_shape_complementarity(cm_entry)
    except (ValueError, KeyError):
        pass


def _resolve_cm_host(uc):
    """Resolve a CB_PORTAL_INFO key from UC host_name."""
    for candidate in (uc.host_name, uc.host_type):
        if candidate in CB_PORTAL_INFO:
            return candidate
    # Try stripping brackets: "CB[7]" → "CB7"
    for candidate in (uc.host_name, uc.host_type):
        stripped = candidate.replace("[", "").replace("]", "")
        if stripped in CB_PORTAL_INFO:
            return stripped
    return None


# ═══════════════════════════════════════════════════════════════════════════
# VERBOSE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def _print_decomposition(uc, result):
    """Print full energy decomposition."""
    print(f"\n  {'='*60}")
    print(f"  {uc.name}  [{uc.binding_mode}]")
    print(f"  {'='*60}")

    terms = []
    if result.dg_metal != 0:
        terms.append(("Metal coord.", result.dg_metal))
    if result.dg_hydrophobic != 0:
        terms.append(("Hydrophobic", result.dg_hydrophobic))
    if result.dg_cavity_dehydration != 0:
        terms.append(("Cav.dehydr.", result.dg_cavity_dehydration))
    if result.dg_hbond != 0:
        terms.append(("H-bond net", result.dg_hbond))
    if result.dg_pi != 0:
        terms.append(("π-interact.", result.dg_pi))
    if result.dg_conf_entropy != 0:
        terms.append(("Conf.entropy", result.dg_conf_entropy))
    if result.dg_shape != 0:
        terms.append(("Shape compl.", result.dg_shape))
    if result.dg_size_mismatch != 0:
        terms.append(("Size mismatch", result.dg_size_mismatch))
    if result.dg_ion_dipole != 0:
        terms.append(("Ion-dipole", result.dg_ion_dipole))
    if result.dg_ion_desolv != 0:
        terms.append(("Ion desolv.", result.dg_ion_desolv))
    if result.dg_portal_size != 0:
        terms.append(("Portal size", result.dg_portal_size))
    if result.dg_cm_dehydration != 0:
        terms.append(("CM dehydr.", result.dg_cm_dehydration))
    if result.dg_cm_shape != 0:
        terms.append(("CM shape", result.dg_cm_shape))

    for label, val in terms:
        print(f"  {label:15s} {val:+8.2f} kJ/mol")
    print(f"  {'─'*30}")
    print(f"  {'ΔG total':15s} {result.dg_total_kj:+8.2f} kJ/mol")
    print(f"  {'log Ka pred':15s} {result.log_Ka_pred:+8.2f}")
    print(f"  {'log Ka exp':15s} {result.log_Ka_exp:+8.2f}")
    print(f"  {'error':15s} {result.error:+8.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION + STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def predict_batch(ucs, verbose=False):
    """Score a list of UniversalComplex entries.

    Returns list of PredictionResult.
    """
    results = []
    for uc in ucs:
        try:
            r = predict(uc, verbose=verbose)
        except Exception as e:
            r = PredictionResult(
                name=uc.name, binding_mode=uc.binding_mode,
                log_Ka_exp=uc.log_Ka_exp, log_Ka_pred=0.0,
                dg_total_kj=0.0, error=-uc.log_Ka_exp)
        results.append(r)
    return results


def compute_statistics(results):
    """Compute R², MAE, bias from PredictionResult list."""
    if not results:
        return {"r2": 0, "mae": 0, "bias": 0, "n": 0}

    exps = [r.log_Ka_exp for r in results]
    preds = [r.log_Ka_pred for r in results]
    errors = [r.error for r in results]
    n = len(results)

    mae = sum(abs(e) for e in errors) / n
    bias = sum(errors) / n

    mean_exp = sum(exps) / n
    ss_tot = sum((e - mean_exp)**2 for e in exps)
    ss_res = sum((p - e)**2 for p, e in zip(preds, exps))
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {"r2": round(r2, 4), "mae": round(mae, 3), "bias": round(bias, 3), "n": n}