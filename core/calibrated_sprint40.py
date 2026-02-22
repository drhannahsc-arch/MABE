"""
core/calibrated_sprint40.py -- Sprint 40: Metal Recalibration + Physics Infrastructure

Results:
  Metal:    MAE 15.11 -> 2.10, R2=0.86 (110 entries, 32 free params)
  HG:       MAE 1.31, R2=0.48 (unchanged, routed to S37a params)
  PL:       MAE 1.05, R2=-0.05 (unchanged, routed to S38 params + offsets)
           New physics modules installed but PL params kept at S38 values.
           Within-target R2 requires 3D spatial features (future sprint).

New modules added but not yet PL-calibrated:
  - pka_estimator.py: 60% of PL compounds correctly charged
  - hbond_classifier.py: 3-tier donor/acceptor subtyping
  - sasa_burial.py: PC-based per-compound burial correction
  - metal_expansion.py: +62 entries from Martell & Smith

Architecture: modality_params.py routes to optimal params per binding_mode.
"""

from core.universal_predictor import PhysicsParameters
from core.calibrated_sprint38_final import (
    get_calibrated_params as _get_s38_params,
    TARGET_OFFSETS as PL_TARGET_OFFSETS,
)
from core.calibrated_params import get_calibrated_params as _get_s37_params


# =========================================================================
# METAL PARAMS: Backsolve on 110 entries (48 seed + 62 expansion)
# R2=0.86, MAE=2.10
# =========================================================================

def get_metal_params():
    """Sprint 40 metal-calibrated params."""
    p = PhysicsParameters()
    # Optimized exchange energies
    p.exchange_O_ether = -2.577
    p.exchange_O_hydroxyl = -18.369
    p.exchange_O_carboxylate = -1.008
    p.exchange_O_phenolate = -10.000
    p.exchange_O_hydroxamate = -20.822
    p.exchange_O_catecholate = -31.241
    p.exchange_N_amine = -12.204
    p.exchange_N_imine = -8.000
    p.exchange_N_pyridine = -10.965
    p.exchange_N_imidazole = -16.943
    p.exchange_S_thioether = -10.000
    p.exchange_S_thiosulfate = -41.436
    p.exchange_S_thiolate = -54.000
    p.exchange_S_dithiocarbamate = -10.000
    p.exchange_O_element = -4.000
    p.exchange_N_element = -6.000
    p.exchange_S_element = -18.000
    p.exchange_P_element = -15.000
    p.exchange_Cl_element = -3.000
    p.exchange_Br_element = -6.000
    p.exchange_I_element = -12.000
    # Optimized physics
    p.alpha_charge = -5.992
    p.base_f_monovalent = 0.020
    p.base_f_divalent = 0.005
    p.base_f_trivalent = 0.001
    p.epsilon_eff = 10.000
    p.chelate_base_d = -15.733
    p.chelate_base_s = -13.920
    p.k_strain = 80.0
    p.lfse_scale = 0.300
    p.jt_4coord = -20.040
    p.jt_6coord = -7.842
    p.dg_translational_per_mol = 2.000
    return p


# =========================================================================
# HOST-GUEST PARAMS: Sprint 37a (unchanged)
# =========================================================================

def get_host_guest_params():
    return _get_s37_params()


# =========================================================================
# PROTEIN-LIGAND PARAMS: Sprint 38 (unchanged pending 3D features)
# =========================================================================

def get_protein_ligand_params():
    return _get_s38_params()


def get_pl_target_offset(host_name):
    return PL_TARGET_OFFSETS.get(host_name, 0.0)
