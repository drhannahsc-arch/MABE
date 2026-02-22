"""
core/calibrated_params.py — Sprint 37a: First Universal Calibration

Calibrated on 51 non-metal entries (41 host-guest + 10 protein-ligand).
Back-solve: 20 free parameters, lambda_reg=0.02, scipy trf.

Performance:
  Overall:  R2=0.338, MAE=1.65 log Ka
  HG only:  R2=0.475, MAE=1.33
  Protein:  R2=-1.048, MAE=2.97
"""
from core.universal_predictor import PhysicsParameters


def get_calibrated_params():
    """Return calibrated PhysicsParameters from seed back-solve."""
    p = PhysicsParameters()
    p.PC_optimal = 0.668786
    p.TdS_per_rotor = 3.275165
    p.dg_translational_per_mol = 2.000000
    p.epsilon_CH_pi = -1.535184
    p.epsilon_cation_pi = -15.000000
    p.epsilon_hbond_charged = -4.235699
    p.epsilon_hbond_neutral = -8.000000
    p.epsilon_water_hbond = 1.000000
    p.gamma_hydrophobic = 0.040000
    p.k_clash = 20.000000
    p.k_cooperativity = 1.432113
    p.k_curvature = 3.000000
    p.k_shape = -25.000000
    p.polar_discount = 1.000000
    p.ring_correction = 0.302413
    p.sigma_PC = 0.216484
    return p
