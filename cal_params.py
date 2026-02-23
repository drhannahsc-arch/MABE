"""
cal_params.py — Parameter vector for MABE frozen scorer calibration.

Maps between a flat numpy/list vector (for scipy.optimize) and the
scorer_frozen.PARAMS dict + SUBTYPE_EXCHANGE + IRVING_WILLIAMS_BONUS.

Usage:
    from cal_params import PARAM_SPEC, apply_params, extract_params
    apply_params(x_vector)       # patches scorer_frozen in-place
    x = extract_params()         # reads current scorer state into vector
"""

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER SPECIFICATION
# Each entry: (name, target, key, initial, lower, upper)
#   target: "PARAMS" | "EXCHANGE" | "IW" (which dict to patch)
#   key: dict key to write
# ═══════════════════════════════════════════════════════════════════════════

PARAM_SPEC = [
    # ── Per-donor exchange energies (19 subtypes) ──
    # Values from scorer's current state (prior calibration round)
    ("ex_O_carboxylate",    "EXCHANGE", "O_carboxylate",     -6.36,  -25.0,   5.0),
    ("ex_O_phenolate",      "EXCHANGE", "O_phenolate",      -17.49,  -50.0,   0.0),
    ("ex_O_hydroxamate",    "EXCHANGE", "O_hydroxamate",     -8.42,  -50.0,   0.0),
    ("ex_O_catecholate",    "EXCHANGE", "O_catecholate",    -15.00,  -80.0,   0.0),
    ("ex_O_ether",          "EXCHANGE", "O_ether",            4.19,  -10.0,  15.0),
    ("ex_O_hydroxyl",       "EXCHANGE", "O_hydroxyl",        -2.00,  -15.0,   5.0),
    ("ex_N_amine",          "EXCHANGE", "N_amine",          -14.90,  -30.0,   0.0),
    ("ex_N_pyridine",       "EXCHANGE", "N_pyridine",        -9.23,  -30.0,   0.0),
    ("ex_N_imine",          "EXCHANGE", "N_imine",           -8.00,  -30.0,   0.0),
    ("ex_N_imidazole",      "EXCHANGE", "N_imidazole",       -7.40,  -25.0,   5.0),
    ("ex_N_amide",          "EXCHANGE", "N_amide",          -19.00,  -40.0,   0.0),
    ("ex_S_thiolate",       "EXCHANGE", "S_thiolate",       -15.80,  -50.0,   0.0),
    ("ex_S_dithiocarbamate","EXCHANGE", "S_dithiocarbamate", -9.30,  -40.0,   0.0),
    ("ex_S_thioether",      "EXCHANGE", "S_thioether",        5.00,  -20.0,  15.0),
    ("ex_S_thiosulfate",    "EXCHANGE", "S_thiosulfate",     -8.00,  -25.0,   5.0),
    ("ex_P_phosphine",      "EXCHANGE", "P_phosphine",      -20.00,  -50.0,   0.0),
    ("ex_Cl_chloride",      "EXCHANGE", "Cl_chloride",       -0.29,  -15.0,  10.0),
    ("ex_Br_bromide",       "EXCHANGE", "Br_bromide",        -6.00,  -25.0,   5.0),
    ("ex_I_iodide",         "EXCHANGE", "I_iodide",         -12.00,  -35.0,   0.0),

    # ── Irving-Williams bonuses (kJ/mol) ──
    ("iw_Mn2+", "IW", "Mn2+",  -5.0, -25.0,  0.0),
    ("iw_Fe2+", "IW", "Fe2+",  -9.0, -30.0,  0.0),
    ("iw_Co2+", "IW", "Co2+", -12.0, -35.0,  0.0),
    ("iw_Ni2+", "IW", "Ni2+", -15.0, -40.0,  0.0),
    ("iw_Cu2+", "IW", "Cu2+", -18.0, -45.0,  0.0),
    ("iw_Zn2+", "IW", "Zn2+",  -7.0, -25.0,  0.0),

    # ── Global PARAMS dict scalars ──
    ("charge_scale",    "PARAMS", "charge_scale",      -1.00, -15.0,    0.0),
    ("chelate_ring_d",  "PARAMS", "chelate_ring_d",   -10.84, -25.0,   -2.0),
    ("chelate_ring_z1", "PARAMS", "chelate_ring_z1",   -4.25, -15.0,   -0.5),
    ("chelate_ring_d0", "PARAMS", "chelate_ring_d0",   -5.82, -15.0,   -1.0),
    ("hsab_match",      "PARAMS", "hsab_match",        -1.50, -10.0,    0.0),
    ("hsab_mismatch",   "PARAMS", "hsab_mismatch",     10.00,   1.0,   30.0),
    ("desolv_base",     "PARAMS", "desolv_frac_base",   0.005,  0.001,  0.04),
    ("desolv_slope",    "PARAMS", "desolv_frac_slope",  0.001,  0.0001, 0.015),
    ("lfse_amp",        "PARAMS", "lfse_amp",           1.00,   0.1,   10.0),
    ("elec_zz_k",       "PARAMS", "elec_zz_k",        -2.68, -10.0,    0.0),
    ("trans_entropy",   "PARAMS", "trans_entropy",      5.50,   1.0,   15.0),
    ("macro_preorg",    "PARAMS", "macro_preorg",      -6.27, -25.0,    0.0),
    ("macro_cavity_k",  "PARAMS", "macro_cavity_k",    -5.03, -25.0,    0.0),
    ("rotor_cost",      "PARAMS", "rotor_cost",         2.00,   0.1,    5.0),
    ("freeze_chelate",  "PARAMS", "freeze_chelate",     0.50,   0.05,   0.95),
    ("repul_anionic",   "PARAMS", "repul_anionic",      1.67,   0.05,   6.0),
    ("repul_steric",    "PARAMS", "repul_steric",       0.15,   0.005,  1.0),
    ("jt_strong",       "PARAMS", "jt_strong",        -12.00, -30.0,   -1.0),
    ("jt_moderate",     "PARAMS", "jt_moderate",       -6.00, -18.0,   -0.5),
]

N_PARAMS = len(PARAM_SPEC)
PARAM_NAMES = [p[0] for p in PARAM_SPEC]


def get_x0():
    """Return initial parameter vector from current scorer state.

    Uses extract_params() to read live values, falling back to PARAM_SPEC
    defaults if scorer isn't available.
    """
    try:
        return extract_params()
    except Exception:
        return [p[3] for p in PARAM_SPEC]


def get_bounds():
    """Return (lower, upper) bound arrays."""
    lo = [p[4] for p in PARAM_SPEC]
    hi = [p[5] for p in PARAM_SPEC]
    return lo, hi


def apply_params(x):
    """Patch scorer_frozen module globals from flat parameter vector.

    Modifies scorer_frozen.SUBTYPE_EXCHANGE, scorer_frozen.IRVING_WILLIAMS_BONUS,
    and scorer_frozen.PARAMS in-place.
    """
    import scorer_frozen as sf

    for i, (name, target, key, *_) in enumerate(PARAM_SPEC):
        val = x[i]
        if target == "EXCHANGE":
            sf.SUBTYPE_EXCHANGE[key] = val
        elif target == "IW":
            sf.IRVING_WILLIAMS_BONUS[key] = val
        elif target == "PARAMS":
            sf.PARAMS[key] = val


def extract_params():
    """Read current scorer state into flat parameter vector."""
    import scorer_frozen as sf

    x = []
    for name, target, key, *_ in PARAM_SPEC:
        if target == "EXCHANGE":
            x.append(sf.SUBTYPE_EXCHANGE.get(key, 0.0))
        elif target == "IW":
            x.append(sf.IRVING_WILLIAMS_BONUS.get(key, 0.0))
        elif target == "PARAMS":
            x.append(sf.PARAMS.get(key, 0.0))
    return x


def print_params(x, label="Parameters"):
    """Pretty-print parameter vector with names and bounds."""
    lo, hi = get_bounds()
    print(f"\n  {label} ({N_PARAMS} values):")
    print(f"  {'Name':25s} {'Value':>8s} {'Low':>8s} {'High':>8s} {'AtBound':>8s}")
    print("  " + "─" * 60)
    for i, (name, target, key, init, lb, ub) in enumerate(PARAM_SPEC):
        at_bound = ""
        if abs(x[i] - lb) < 1e-6:
            at_bound = "→LOW"
        elif abs(x[i] - ub) < 1e-6:
            at_bound = "→HIGH"
        print(f"  {name:25s} {x[i]:8.3f} {lb:8.3f} {ub:8.3f} {at_bound:>8s}")


if __name__ == "__main__":
    x0 = get_x0()
    print_params(x0, "Initial values")
    lo, hi = get_bounds()
    print(f"\n  Total parameters: {N_PARAMS}")
    print(f"  Exchange energies: {sum(1 for p in PARAM_SPEC if p[1]=='EXCHANGE')}")
    print(f"  Irving-Williams:   {sum(1 for p in PARAM_SPEC if p[1]=='IW')}")
    print(f"  Global scalars:    {sum(1 for p in PARAM_SPEC if p[1]=='PARAMS')}")