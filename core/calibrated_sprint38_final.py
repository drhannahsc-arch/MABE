"""
Sprint 38: Consolidated Calibration -- Decoupled Architecture

Three calibration tiers:
  1. Metal coordination params: from Sprint 37a (frozen, needs NIST data for improvement)
  2. Host-guest params: from Sprint 37a seed backsolve (frozen)  
  3. Protein-ligand params: from Sprint 38 ChEMBL backsolve (18 free params + 6 offsets)

Performance:
  Protein-ligand:  MAE=1.05 log Ka, R2=-0.05 (21,521 compounds, 6 targets)
  Host-guest:      MAE=3.44 (41 entries, frozen from Sprint 37a)
  Metal:           MAE=15.11 (48 entries, requires NIST expansion)

Per-target protein-ligand:
  DHFR:            MAE=0.87, sigma=1.14 (n=8441)
  CA-II:           MAE=1.06, sigma=1.37 (n=7890)
  HIV-1 protease:  MAE=1.30, sigma=1.62 (n=2314)
  Thrombin:        MAE=1.33, sigma=1.68 (n=609)
  COX-2:           MAE=1.35, sigma=1.58 (n=1278)  R2=+0.34
  Trypsin:         MAE=1.38, sigma=1.58 (n=989)
"""

from core.universal_predictor import PhysicsParameters


# ===========================================================================
# PER-TARGET OFFSETS (log Ka units)
# Corrects systematic bias per protein active site.
# Applied AFTER physics prediction.
# ===========================================================================

TARGET_OFFSETS = {
    "Carbonic anhydrase II": +0.1198,
    "Thrombin": -0.0264,
    "Trypsin": +1.3030,
    "HIV-1 protease": -0.5915,
    "COX-2": +0.2361,
    "Dihydrofolate reductase": +0.0897,
}


# ===========================================================================
# CALIBRATED PHYSICS PARAMETERS
# Stage 1: Seed backsolve (99 entries, all 56 free) -> host-guest + metal base
# Stage 2: Joint backsolve (21,620 entries, 18 protein-relevant free) -> protein physics
# ===========================================================================

def get_calibrated_params():
    """Return Sprint 38 calibrated PhysicsParameters."""
    p = PhysicsParameters()

    # -- Metal coordination (frozen from Sprint 37a) ------------------
    p.exchange_O_ether = -1.000000
    p.exchange_O_hydroxyl = -5.000000
    p.exchange_O_carboxylate = -15.000000
    p.exchange_O_phenolate = -15.000000
    p.exchange_O_hydroxamate = -10.000000
    p.exchange_O_catecholate = -20.000000
    p.exchange_N_amine = -10.000000
    p.exchange_N_imine = -12.000000
    p.exchange_N_pyridine = -8.000000
    p.exchange_N_imidazole = -10.000000
    p.exchange_S_thioether = -5.000000
    p.exchange_S_thiosulfate = -8.000000
    p.exchange_S_thiolate = -30.000000
    p.exchange_S_dithiocarbamate = -25.000000
    p.exchange_O_element = -5.000000
    p.exchange_N_element = -10.000000
    p.exchange_S_element = -15.000000
    p.exchange_P_element = -5.000000
    p.exchange_Cl_element = -5.000000
    p.exchange_Br_element = -5.000000
    p.exchange_I_element = -5.000000
    p.alpha_charge = 1.500000
    p.base_f_monovalent = 0.300000
    p.base_f_divalent = 0.400000
    p.base_f_trivalent = 0.500000
    p.epsilon_eff = 4.000000
    p.chelate_base_d = 3.000000
    p.chelate_base_s = 1.000000
    p.k_strain = 2.000000
    p.k_macro_O = 3.000000
    p.k_macro_N = 4.000000
    p.k_macro_other = 2.000000
    p.sigma_cavity = 1.000000
    p.lfse_scale = 1.000000
    p.jt_4coord = -3.000000
    p.jt_6coord = -5.000000

    # -- Universal terms (Stage 2 optimized) --------------------------
    p.dg_translational_per_mol = 2.000000
    p.gamma_hydrophobic = 0.040000
    p.k_curvature = 3.000000
    p.polar_discount = 1.000000
    p.epsilon_hbond_neutral = -2.983000
    p.epsilon_hbond_charged = -12.320000
    p.epsilon_hbond_OH_pi = -2.000000
    p.epsilon_water_hbond = 2.847000
    p.k_cooperativity = 1.000000
    p.theta_half = 0.500000
    p.epsilon_pi_parallel = -1.277000
    p.epsilon_pi_T_shaped = -2.000000
    p.epsilon_CH_pi = -5.000000
    p.epsilon_cation_pi = -3.293000
    p.TdS_per_rotor = 1.000000
    p.ring_correction = 0.117000
    p.PC_optimal = 0.601000
    p.sigma_PC = 0.250000
    p.k_shape = -10.549000
    p.k_clash = 20.000000

    return p


def get_target_offset(host_name):
    """Return per-target offset for protein-ligand predictions."""
    return TARGET_OFFSETS.get(host_name, 0.0)


def predict_corrected(uc, params=None):
    """Predict with per-target offset correction.
    
    Use this for protein-ligand predictions.
    For host-guest and metal, offsets are 0 so this is equivalent to raw predict.
    """
    from core.universal_predictor import predict
    if params is None:
        params = get_calibrated_params()
    result = predict(uc, params)
    offset = get_target_offset(uc.host_name)
    result.log_Ka_pred += offset
    result.error = result.log_Ka_pred - result.log_Ka_exp
    result.dg_pred_kj = -result.log_Ka_pred * 5.71
    return result
