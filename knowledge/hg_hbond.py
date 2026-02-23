"""
hg_hbond.py — H-bond network energy for MABE Phase 7.

Classifies and scores H-bonds at host-guest interfaces.
In aqueous solution, net H-bond energy = bonds formed - water H-bonds broken.

H-bond classes:
  1. neutral:         X-H···Y (OH, NH to C=O, ether, OH)    ~2-5 kJ/mol
  2. charge_assisted: X-H⁺···Y⁻ or X-H···Y⁻               ~8-15 kJ/mol
  3. water_penalty:   desolvating an H-bond donor/acceptor   ~3-6 kJ/mol

Zero for metal coordination (metal-donor bonds handled by scorer_frozen.py).
"""

# ═══════════════════════════════════════════════════════════════════════════
# H-BOND PARAMETERS (fitted by calibration)
# ═══════════════════════════════════════════════════════════════════════════
HBOND_PARAMS = {
    # Per-H-bond energies (kJ/mol, negative = favorable)
    "eps_neutral":          -3.0,     # neutral donor → neutral acceptor
    "eps_charge_assisted": -10.0,     # R-NH3+ → C=O (portal), charge-assisted
    "eps_oh_pi":            -1.5,     # O-H → π-system (weak, CD guest aromatic)

    # Water competition penalty (kJ/mol, positive = unfavorable)
    # Each H-bond formed at host-guest interface displaces ~1.5 water H-bonds
    # on average (some donor/acceptor sites had 2 waters, some 0)
    "water_penalty_per_hb":  3.5,     # cost per water H-bond broken

    # Effective fraction: how many water H-bonds broken per formed host-guest HB
    "water_displacement":    1.2,     # ~1-2 waters displaced per HB formed
}


# ═══════════════════════════════════════════════════════════════════════════
# H-BOND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_hbonds(entry: dict, host_db: dict) -> dict:
    """Classify H-bonds at the host-guest interface.

    Returns dict with counts:
        n_neutral: neutral H-bonds (guest OH/NH to host OH/ether)
        n_charge_assisted: charge-assisted (guest NH3+ to host C=O/SO3-)
        n_oh_pi: OH-to-pi interactions
        n_total: total H-bonds
        n_water_displaced: estimated water H-bonds broken
    """
    host = host_db[entry["host"]]
    portal = host["portal_type"]
    n_hb = entry.get("n_hbonds_portal", 0)
    is_cation = entry.get("guest_has_cation", False)
    guest_charge = entry.get("guest_charge", 0)

    n_neutral = 0
    n_charge_assisted = 0
    n_oh_pi = 0

    if n_hb > 0:
        if is_cation and portal in ("carbonyl", "sulfonate"):
            # R-NH3+ donating to C=O or SO3-: charge-assisted
            # Typically 2-3 of the portal H-bonds are charge-assisted,
            # remainder are neutral
            if guest_charge >= 2:
                # Dication: both ends charge-assisted
                n_charge_assisted = min(n_hb, 6)
            elif guest_charge == 1:
                # Monocation: ~2-3 charge-assisted at one portal
                n_charge_assisted = min(n_hb, 3)
            n_neutral = max(0, n_hb - n_charge_assisted)
        elif portal == "hydroxyl":
            # CD portals: all neutral (guest OH to host OH)
            n_neutral = n_hb
        else:
            # Default: neutral
            n_neutral = n_hb

    # OH-to-pi: guest aromatic ring near CD hydroxyl (from entry annotation)
    n_oh_pi = entry.get("n_oh_pi", 0)

    n_total = n_neutral + n_charge_assisted + n_oh_pi

    # Water displacement estimate
    # Each H-bond formed means the donor and acceptor each lose ~0.5-1 water H-bonds
    n_water_displaced = n_total * HBOND_PARAMS["water_displacement"]

    return {
        "n_neutral": n_neutral,
        "n_charge_assisted": n_charge_assisted,
        "n_oh_pi": n_oh_pi,
        "n_total": n_total,
        "n_water_displaced": n_water_displaced,
    }


def dg_hbond_net(hbond_counts: dict) -> float:
    """Compute net H-bond energy in kJ/mol.

    Net = H-bonds formed (favorable) - water displacement (unfavorable).
    """
    # Favorable: H-bonds formed
    dg_formed = (
        hbond_counts["n_neutral"] * HBOND_PARAMS["eps_neutral"]
        + hbond_counts["n_charge_assisted"] * HBOND_PARAMS["eps_charge_assisted"]
        + hbond_counts["n_oh_pi"] * HBOND_PARAMS["eps_oh_pi"]
    )

    # Unfavorable: water H-bonds broken
    dg_water = hbond_counts["n_water_displaced"] * HBOND_PARAMS["water_penalty_per_hb"]

    return dg_formed + dg_water


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════

def compute_dg_hbond(entry: dict, host_db: dict, verbose: bool = False) -> float:
    """Full pipeline: classify then score H-bonds."""
    counts = classify_hbonds(entry, host_db)
    dg = dg_hbond_net(counts)

    if verbose:
        print(f"  H-bonds: {counts['n_neutral']} neutral, "
              f"{counts['n_charge_assisted']} charge-assisted, "
              f"{counts['n_oh_pi']} OH-π")
        print(f"  Water displaced: {counts['n_water_displaced']:.1f}")
        print(f"  ΔG_hbond_net: {dg:+.1f} kJ/mol")

    return dg


if __name__ == "__main__":
    from hg_dataset import HG_DATA, HOST_DB

    print("Phase 7: H-Bond classification test")
    print("=" * 50)

    tests = [
        "CB7+adamantane-NH3+",    # charge-assisted at portal
        "bCD+1-adamantanol",      # neutral OH at portal
        "bCD+naphthalene",        # no H-bonds
        "CB6+hexanediamine-2H+",  # dication, many charge-assisted
        "aCD+1-hexanol",          # neutral OH
        "sCX4+tetramethyl-N+",    # cation to sulfonate
    ]

    for name in tests:
        entry = next((e for e in HG_DATA if e["name"] == name), None)
        if entry is None:
            print(f"\n{name}: NOT FOUND")
            continue
        print(f"\n{name} (log Ka={entry['log_Ka']:.1f}):")
        compute_dg_hbond(entry, HOST_DB, verbose=True)