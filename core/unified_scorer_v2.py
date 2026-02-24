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
from hg_hbond import compute_dg_hbond as _hg_compute_dg_hbond, HBOND_PARAMS
from hg_pi import compute_dg_pi as _hg_compute_dg_pi, PI_PARAMS
from hg_conf_shape import (
    compute_dg_conf_shape as _hg_compute_dg_conf_shape,
    CONF_SHAPE_PARAMS, HOST_CAVITY_VOLUME,
)
from hg_dataset import HOST_DB as _HG_HOST_DB

from cross_modal_predictor import (
    dg_ion_dipole as _cm_dg_ion_dipole,
    dg_desolvation as _cm_dg_desolvation,
    dg_portal_size_match as _cm_dg_portal_size_match,
    dg_cb_dehydration as _cm_dg_cb_dehydration,
    dg_shape_complementarity as _cm_dg_shape_complementarity,
    CM_PARAMS,
)
from cross_modal_dataset import CB_PORTAL_INFO


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
    "α-cyclodextrin": "alpha-CD", "alpha-CD": "alpha-CD",
    "β-cyclodextrin": "beta-CD",  "beta-CD": "beta-CD",
    "γ-cyclodextrin": "gamma-CD", "gamma-CD": "gamma-CD",
    "CB[6]": "CB6", "CB6": "CB6",
    "CB[7]": "CB7", "CB7": "CB7",
    "CB[8]": "CB8", "CB8": "CB8",
    "p-sulfonatocalix[4]arene": "calix4-SO3", "calix4-SO3": "calix4-SO3",
    "pillar[5]arene": "pillar5", "pillar5": "pillar5",
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