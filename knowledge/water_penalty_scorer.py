"""
MABE P20: Water Competition Penalty Scorer
============================================
Back-solved from MNSol v2012 (390 aqueous neutrals) + FreeSolv v0.52 (642 compounds).

Physics: When a polar group is buried in a binding pocket, it must break H-bonds
with surrounding water molecules. This desolvation cost is the dominant energy
barrier for binding in aqueous solution.

Parameters:
    eps_water_OH  = 5.3 kJ/mol per H-bond (from OH donor desolvation)
    eps_water_NH  = 5.0 kJ/mol per H-bond (from NH donor desolvation)
    eps_water_O   = 2.2 kJ/mol per H-bond (from O acceptor desolvation)
    eps_water_avg = 5.2 kJ/mol per H-bond (donor-weighted mean)

SASA-based coefficients (kJ/mol per A^2, for detailed scoring):
    gamma_OH_donor   = -1.097 (solvation; desolvation = +1.097)
    gamma_NH_donor   = -0.701
    gamma_O_acceptor = -0.210
    gamma_N_acceptor = -0.154

Calibration: R^2=0.695 (MNSol SASA), R^2=0.499 (FreeSolv tags)
Cross-dataset correlation: r=0.82 (N=407 matched compounds)

Sources:
    - MNSol v2012: doi:10.13020/3eks-j059 (CC-BY-NC-ND)
    - FreeSolv v0.52: doi:10.1021/acs.jced.7b00104 (CC-BY)
"""

# ============================================================
# Calibrated parameters from back-solve
# ============================================================

# Per-H-bond desolvation costs (kJ/mol)
# Combined estimate from MNSol SASA regression + FreeSolv group regression
EPS_WATER_OH = 5.3     # per H-bond disrupted when burying OH
EPS_WATER_NH = 5.0     # per H-bond disrupted when burying NH
EPS_WATER_O_ACC = 2.2  # per H-bond disrupted when burying C=O acceptor
EPS_WATER_AVG = 5.2    # donor-weighted mean (recommended default)

# Per-group total desolvation costs (kJ/mol)
DESOLV_PER_OH = 14.2   # total cost for one OH group (makes ~2.7 HBs with water)
DESOLV_PER_NH2 = 10.1  # total cost for one NH2 group (makes ~2 HBs)
DESOLV_PER_CO = 4.4    # total cost for one C=O group (makes ~2 HBs)

# SASA-based coefficients (kJ/mol/A^2) — for detailed atom-type scoring
# Sign convention: desolvation penalty (positive = costs energy)
GAMMA_DESOLV = {
    'aliphatic':  -0.0523,  # NEGATIVE: burying aliphatic RELEASES cavity energy
    'aromatic':    0.0893,  # POSITIVE: slight penalty for burying aromatic
    'OH_donor':    1.0971,  # per A^2 of OH surface buried
    'NH_donor':    0.7012,  # per A^2 of NH surface buried
    'O_acceptor':  0.2099,  # per A^2 of O acceptor surface buried
    'N_acceptor':  0.1536,  # per A^2 of N acceptor surface buried
    'halogen':     0.0088,  # nearly zero — halogens barely interact with water
    'sulfur':      0.0731,  # moderate
}


def compute_water_penalty(uc, params=None, result=None):
    """
    Compute the water competition penalty for burying polar groups.
    
    Self-zeroing: returns unchanged result if no H-bond data present.
    
    Modes:
      1. Simple (n_hbonds_displaced): uses EPS_WATER_AVG per H-bond
      2. Group-level (n_oh_buried, n_nh_buried, n_co_buried): uses per-group costs
      3. SASA-level (polar SASA buried): uses GAMMA_DESOLV coefficients
    
    Args:
        uc: UniversalComplex with populated H-bond / polar SASA fields
        params: optional parameter override dict
        result: ScoringResult to update
    
    Returns:
        Updated result with dg_water_penalty field
    """
    if result is None:
        result = {}
    
    eps = EPS_WATER_AVG if params is None else params.get('eps_water', EPS_WATER_AVG)
    
    # ── Self-zero gate ──
    n_displaced = getattr(uc, 'n_water_hbonds_displaced', 0)
    n_oh = getattr(uc, 'n_oh_buried', 0)
    n_nh = getattr(uc, 'n_nh_buried', 0)
    n_co = getattr(uc, 'n_co_buried', 0)
    sasa_polar = getattr(uc, 'sasa_polar_buried_A2', 0)
    
    if n_displaced <= 0 and n_oh + n_nh + n_co <= 0 and sasa_polar <= 0:
        result['dg_water_penalty_kJ'] = 0.0
        result['water_penalty_breakdown'] = {}
        return result
    
    # ── Mode 3: SASA-level (most detailed) ──
    if sasa_polar > 0:
        sasa_oh = getattr(uc, 'sasa_oh_buried_A2', 0)
        sasa_nh = getattr(uc, 'sasa_nh_buried_A2', 0)
        sasa_o_acc = getattr(uc, 'sasa_o_acceptor_buried_A2', 0)
        sasa_n_acc = getattr(uc, 'sasa_n_acceptor_buried_A2', 0)
        
        penalty = (GAMMA_DESOLV['OH_donor'] * sasa_oh +
                   GAMMA_DESOLV['NH_donor'] * sasa_nh +
                   GAMMA_DESOLV['O_acceptor'] * sasa_o_acc +
                   GAMMA_DESOLV['N_acceptor'] * sasa_n_acc)
        
        result['dg_water_penalty_kJ'] = round(penalty, 2)
        result['water_penalty_breakdown'] = {
            'mode': 'sasa',
            'oh_A2': sasa_oh, 'nh_A2': sasa_nh,
            'o_acc_A2': sasa_o_acc, 'n_acc_A2': sasa_n_acc,
            'penalty_kJ': round(penalty, 2),
        }
        return result
    
    # ── Mode 2: Group-level ──
    if n_oh + n_nh + n_co > 0:
        penalty = (DESOLV_PER_OH * n_oh +
                   DESOLV_PER_NH2 * n_nh +
                   DESOLV_PER_CO * n_co)
        
        result['dg_water_penalty_kJ'] = round(penalty, 2)
        result['water_penalty_breakdown'] = {
            'mode': 'group',
            'n_oh': n_oh, 'n_nh': n_nh, 'n_co': n_co,
            'penalty_kJ': round(penalty, 2),
        }
        return result
    
    # ── Mode 1: Simple count ──
    penalty = eps * n_displaced
    result['dg_water_penalty_kJ'] = round(penalty, 2)
    result['water_penalty_breakdown'] = {
        'mode': 'simple',
        'n_displaced': n_displaced,
        'eps_water': eps,
        'penalty_kJ': round(penalty, 2),
    }
    return result


# ============================================================
# Scorer integration hook for unified_scorer_v2.py
# ============================================================
def _compute_water_penalty(uc, params, result):
    """
    Drop-in function for unified_scorer_v2.py's _compute_* pattern.
    
    Self-zeros when no polar burial data present.
    Activates when uc has any of:
      - n_water_hbonds_displaced > 0
      - n_oh_buried, n_nh_buried, n_co_buried > 0
      - sasa_polar_buried_A2 > 0
    """
    return compute_water_penalty(uc, params, result)