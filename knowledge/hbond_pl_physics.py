"""
knowledge/hbond_pl_physics.py — Per-type H-bond scoring for protein-ligand

Replaces flat n_hb × eps with type-classified H-bond energies.

Each H-bond's net contribution depends on:
  1. Intrinsic strength (donor acidity × acceptor basicity)
  2. Water competition (donor and acceptor each lose ~1 water HB)
  3. Protein preorganization (pocket pre-shaped → reduced entropic cost)

Net ΔG_HB = intrinsic - water_donor_cost - water_acceptor_cost + preorganization

Parameter sources (non-binding, independent measurements):
  - Pace 2014, J. Mol. Biol. 426:1500 — 2818 HBs from protein folding ΔΔG
  - Fersht 1987, TIBS 12:301 — barnase double-mutant cycles
  - Abraham 1993, Chem. Soc. Rev. 22:73 — α/β scales (relative ordering)
  - Cabani 1981 — group solvation contributions (water competition)

These are NOT fitted against protein-ligand binding Ka. They are from
protein folding thermodynamics, gas-phase calorimetry, and solution
transfer experiments.
"""


# ═══════════════════════════════════════════════════════════════════════════
# PER-TYPE NET H-BOND ENERGIES (kJ/mol, negative = favorable)
# ═══════════════════════════════════════════════════════════════════════════

# Anchored to Pace 2014 experimental values where available.
# Filled with Abraham-ratio-scaled estimates for uncovered types.
# All values are NET (intrinsic - water competition + preorganization).

HBOND_TYPE_ENERGY = {
    # ── NEUTRAL DONOR → NEUTRAL ACCEPTOR ──────────────────────────
    # Pace 2014 Table 2, protein folding ΔΔG per H-bond

    "NH_backbone→O_carbonyl":   -4.7,  # Pace 2014: backbone amide NH to C=O
    "NH_sidechain→O_carbonyl":  -4.0,  # Pace 2014: Asn/Gln NH2 to C=O
    "OH→O_carbonyl":            -6.3,  # Pace 2014: Ser/Thr/Tyr OH to C=O
    "NH→O_hydroxyl":            -4.0,  # Pace 2014: NH to Ser/Thr OH
    "OH→O_hydroxyl":            -5.0,  # Pace estimate: OH to OH (interpolated)
    "OH→N_heterocycle":         -5.5,  # Abraham-ratio-scaled from OH→O_carbonyl
    "NH→N_heterocycle":         -4.2,  # Abraham-ratio-scaled from NH→O_carbonyl

    # ── CHARGE-ASSISTED ───────────────────────────────────────────
    # Fersht 1987: 15-20 kJ/mol. Pace 2014: 17.2 ± 2.1

    "NH3+→O_carboxylate":      -17.0,  # salt bridge (Pace 2014)
    "NH3+→O_carbonyl":          -8.5,  # charge-assisted but not full salt bridge
    "guanidinium→O_carboxylate":-15.0, # Arg-Asp/Glu (bidentate, per HB)
    "OH→O_carboxylate":         -9.0,  # Ser/Thr OH to COO- (charge-assisted)

    # ── WEAK / SPECIAL ────────────────────────────────────────────

    "SH→O_carbonyl":            -1.5,  # Cys thiol: weak donor (Abraham α=0.09)
    "CH→O_carbonyl":            -1.0,  # C-H···O (weak, real but small)
    "OH→S_thioether":           -2.5,  # OH to Met S (weak acceptor)
    "NH→S_thioether":           -2.0,  # NH to Met S
    "water_mediated":           -3.0,  # water bridge: weaker than direct

    # ── DEFAULTS ──────────────────────────────────────────────────

    "neutral":                  -4.5,  # generic neutral HB (Pace mean)
    "charge_assisted":         -12.0,  # generic charged HB
    "weak":                     -1.5,  # generic weak (CH, SH donors)
}

# Water competition costs per donor/acceptor type
# Source: Cabani 1981 group solvation, Abraham 1993 basicity
# These are ALREADY folded into the HBOND_TYPE_ENERGY values above.
# Listed here for documentation and if anyone wants raw decomposition.
WATER_COMPETITION = {
    "NH_amide_donor":    +6.0,  # kJ/mol: cost of removing 1 water from amide NH
    "OH_donor":          +8.0,  # cost of removing 1 water from OH
    "NH3+_donor":        +4.0,  # lower cost: charge RETAINS water even when H-bonded
    "O_carbonyl_acceptor": +5.0,  # cost of removing 1 water from C=O acceptor
    "O_carboxylate_acceptor": +3.0,  # lower: charge stabilizes remaining waters
}


# ═══════════════════════════════════════════════════════════════════════════
# H-BOND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_hbond(donor_type, acceptor_type, is_charged=False):
    """Classify a protein-ligand H-bond by donor and acceptor types.

    Args:
        donor_type: "NH_backbone", "NH_sidechain", "OH", "NH3+",
                    "guanidinium", "SH", "CH", "water"
        acceptor_type: "O_carbonyl", "O_hydroxyl", "O_carboxylate",
                       "N_heterocycle", "S_thioether"
        is_charged: True if either partner is formally charged

    Returns:
        (hbond_class_str, energy_kJ)
    """
    # Try specific combination first
    key = f"{donor_type}→{acceptor_type}"
    if key in HBOND_TYPE_ENERGY:
        return key, HBOND_TYPE_ENERGY[key]

    # Fall back to charged/neutral/weak defaults
    if is_charged or "+" in donor_type or "carboxylate" in acceptor_type:
        return "charge_assisted", HBOND_TYPE_ENERGY["charge_assisted"]

    if donor_type in ("SH", "CH"):
        return "weak", HBOND_TYPE_ENERGY["weak"]

    if donor_type == "water":
        return "water_mediated", HBOND_TYPE_ENERGY["water_mediated"]

    return "neutral", HBOND_TYPE_ENERGY["neutral"]


# ═══════════════════════════════════════════════════════════════════════════
# SCORING FROM CONTACT SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def score_hbond_network(hbond_list):
    """Score a list of classified H-bonds.

    Args:
        hbond_list: list of dicts, each with:
            - "donor_type": str
            - "acceptor_type": str
            - "is_charged": bool (optional, default False)
            OR
            - "type": str (pre-classified key from HBOND_TYPE_ENERGY)

    Returns:
        dict with:
            total_kJ: total H-bond energy (negative = favorable)
            per_hb: list of (class, energy_kJ) tuples
            n_neutral: count of neutral HBs
            n_charged: count of charged HBs
            n_weak: count of weak HBs
    """
    per_hb = []
    n_neutral = 0
    n_charged = 0
    n_weak = 0

    for hb in hbond_list:
        if "type" in hb:
            # Pre-classified
            hb_type = hb["type"]
            energy = HBOND_TYPE_ENERGY.get(hb_type, HBOND_TYPE_ENERGY["neutral"])
        else:
            donor = hb.get("donor_type", "neutral")
            acceptor = hb.get("acceptor_type", "O_carbonyl")
            charged = hb.get("is_charged", False)
            hb_type, energy = classify_hbond(donor, acceptor, charged)

        per_hb.append((hb_type, energy))

        if "charge" in hb_type or "+" in hb_type:
            n_charged += 1
        elif "weak" in hb_type or "SH" in hb_type or "CH" in hb_type:
            n_weak += 1
        else:
            n_neutral += 1

    total = sum(e for _, e in per_hb)

    return {
        "total_kJ": total,
        "per_hb": per_hb,
        "n_neutral": n_neutral,
        "n_charged": n_charged,
        "n_weak": n_weak,
        "n_total": len(per_hb),
    }


def score_hbond_simple(n_neutral=0, n_charged=0, n_weak=0, n_water_mediated=0):
    """Quick scoring from H-bond counts by category.

    For use when detailed donor/acceptor types aren't available.
    Uses category-average energies.

    Returns: total ΔG_HB in kJ/mol (negative = favorable)
    """
    return (n_neutral * HBOND_TYPE_ENERGY["neutral"]
            + n_charged * HBOND_TYPE_ENERGY["charge_assisted"]
            + n_weak * HBOND_TYPE_ENERGY["weak"]
            + n_water_mediated * HBOND_TYPE_ENERGY["water_mediated"])