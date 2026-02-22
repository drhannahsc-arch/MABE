"""
core/modality_params.py -- Sprint 39: Modality-Branched Parameter Routing

Problem: Stage 2 (Sprint 38) optimized shared params against 21k protein-ligand
entries, moving neutral H-bond, shape, cation-pi, rotor entropy away from
host-guest optimal values.

    S37 params -> HG MAE=1.33, R2=0.48
    S38 params -> HG MAE=1.89, R2=-0.04  (regressed)
    S38 params -> PL MAE=1.0 (maintained with offsets)

Fix: Separate PhysicsParameters per modality. The predictor routes based on
binding_mode. Each modality keeps its own calibrated values for the universal
terms that diverged.

Metal coordination params are identical in both (frozen from Sprint 37a).
The 6 params that moved >30% all live in the universal terms (Phases 6-10).
"""

from core.universal_predictor import PhysicsParameters, predict, PredictionResult
from core.calibrated_params import get_calibrated_params as _get_s37_params
from core.calibrated_sprint38_final import (
    get_calibrated_params as _get_s38_params,
    TARGET_OFFSETS,
)
from core.calibrated_sprint40 import get_metal_params as _get_s40_metal


# =========================================================================
# HOST-GUEST PARAMS: Sprint 37a calibration (41 HG entries, 20 free)
# MAE=1.33, R2=0.48 on enriched host-guest
# =========================================================================

def get_host_guest_params():
    """S37a-calibrated params optimal for host-guest inclusion."""
    return _get_s37_params()


# =========================================================================
# PROTEIN-LIGAND PARAMS: Sprint 38 Stage 2 (21k entries, 18 free)
# MAE=1.0 with per-target offsets
# =========================================================================

def get_protein_ligand_params():
    """S38-calibrated params optimal for protein-ligand."""
    return _get_s38_params()


# =========================================================================
# METAL COORDINATION PARAMS: Same in both (frozen, needs NIST expansion)
# =========================================================================

def get_metal_params():
    """S40-calibrated metal params.
    
    Backsolve on 110 entries (48 seed + 62 Martell & Smith expansion).
    MAE=2.10, R2=0.86 (from MAE=15.11, R2=-0.45)
    """
    return _get_s40_metal()


# =========================================================================
# ROUTING
# =========================================================================

_MODALITY_MAP = {
    "host_guest_inclusion": get_host_guest_params,
    "synthetic_receptor": get_host_guest_params,
    "metal_coordination": get_metal_params,
    "protein_ligand": get_protein_ligand_params,
}


def get_params_for_modality(binding_mode):
    """Return the optimal PhysicsParameters for a given binding mode.

    Falls back to S38 params for unknown modalities.
    """
    getter = _MODALITY_MAP.get(binding_mode, get_protein_ligand_params)
    return getter()


def predict_routed(uc, params_override=None):
    """Predict with automatic modality-based parameter routing.

    If params_override is provided, uses that directly (for backsolve).
    Otherwise routes to the optimal param set for this binding_mode.

    For protein-ligand, applies per-target offsets.
    """
    if params_override is not None:
        params = params_override
    else:
        params = get_params_for_modality(uc.binding_mode)

    result = predict(uc, params)

    # Apply per-target offset for protein-ligand
    if uc.binding_mode == "protein_ligand":
        offset = TARGET_OFFSETS.get(uc.host_name, 0.0)
        result.log_Ka_pred += offset
        result.error = result.log_Ka_pred - result.log_Ka_exp
        result.dg_pred_kj = -result.log_Ka_pred * 5.71

    return result


def predict_routed_batch(entries, params_override=None):
    """Batch predict with routing."""
    results = []
    for uc in entries:
        try:
            r = predict_routed(uc, params_override)
        except Exception as e:
            r = PredictionResult(
                name=uc.name, binding_mode=uc.binding_mode,
                log_Ka_exp=uc.log_Ka_exp, log_Ka_pred=0.0,
                dg_pred_kj=0.0, error=-uc.log_Ka_exp)
        results.append(r)
    return results
