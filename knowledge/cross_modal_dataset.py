"""
cross_modal_dataset.py — Cross-modal entries for MABE Phase 11b.

Metal cation@CB[n] binding constants from:
  Zhang, Grimm, Miskolczy, Biczók, Biedermann, Nau
  "Binding affinities of cucurbit[n]urils with cations"
  Chem. Commun. 2019, 55, 14131–14134. DOI: 10.1039/C9CC07687E

These are CROSS-MODAL because the metal cation:
  1. Coordinates portal C=O oxygens → metal scorer terms (exchange, charge, HSAB)
  2. Displaces high-energy cavity water → HG scorer terms (dehydration, shape)

Both scoring paths contribute to the same log Ka, creating parameter coupling
in joint optimization.

Donor model: CB portal C=O groups modeled as O_ether donors (weak carbonyl O).
Macrocyclic flag TRUE — portal is a preorganized ring of donors.
Portal radii from paper: 2.4, 3.9, 5.4, 6.9 Å diameter for CB5–8.
"""

import math

# ═══════════════════════════════════════════════════════════════════════════
# CB HOST PROPERTIES FOR CROSS-MODAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
# Extends HOST_DB for CB5 (not in original HG dataset)

CB_PORTAL_INFO = {
    "CB5": {
        "n_carbonyls": 5,
        "portal_diameter_A": 2.4,
        "portal_radius_nm": 0.12,
        "cavity_volume_A3": 82.0,       # Lagona 2005
        "cavity_sasa": 120.0,           # approximate
        "host_key": "CB5",
    },
    "CB6": {
        "n_carbonyls": 6,
        "portal_diameter_A": 3.9,
        "portal_radius_nm": 0.195,
        "cavity_volume_A3": 164.0,
        "cavity_sasa": 264.0,
        "host_key": "CB6",
    },
    "CB7": {
        "n_carbonyls": 7,
        "portal_diameter_A": 5.4,
        "portal_radius_nm": 0.27,
        "cavity_volume_A3": 279.0,
        "cavity_sasa": 418.0,
        "host_key": "CB7",
    },
    "CB8": {
        "n_carbonyls": 8,
        "portal_diameter_A": 6.9,
        "portal_radius_nm": 0.345,
        "cavity_volume_A3": 479.0,
        "cavity_sasa": 609.0,
        "host_key": "CB8",
    },
}


def _cm(name, metal, cb_host, log_Ka, source="Zhang2019"):
    """Create a cross-modal entry."""
    cb = CB_PORTAL_INFO[cb_host]
    # Ion properties from ionic radius
    # Metal ion volume = (4/3)π r³ where r is ionic radius
    # We'll compute this on the fly from METAL_DB
    return {
        "name": name,
        "mode": "cross_modal",
        "metal": metal,
        "cb_host": cb_host,
        "n_portal_carbonyls": cb["n_carbonyls"],
        "portal_radius_nm": cb["portal_radius_nm"],
        "cavity_volume_A3": cb["cavity_volume_A3"],
        "cavity_sasa": cb["cavity_sasa"],
        "log_Ka": log_Ka,
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-MODAL DATASET: Metal@CB[n]
# From Zhang et al. 2019, Table 1
# All measured in water at 298 K (CB5 at 283 K)
# Fluorescence/ITC displacement titrations, error ± 0.1–0.3
# ═══════════════════════════════════════════════════════════════════════════

CROSS_MODAL_DATA = [

    # ── CB5 (portal Ø 2.4 Å) ──
    _cm("CB5+Li+",   "Li+",  "CB5", 2.02),
    _cm("CB5+Na+",   "Na+",  "CB5", 3.94),
    _cm("CB5+K+",    "K+",   "CB5", 4.73),
    _cm("CB5+Rb+",   "Rb+",  "CB5", 3.22),
    _cm("CB5+Cs+",   "Cs+",  "CB5", 2.61),
    _cm("CB5+Mg2+",  "Mg2+", "CB5", 2.50),
    _cm("CB5+Ca2+",  "Ca2+", "CB5", 2.64),
    _cm("CB5+Sr2+",  "Sr2+", "CB5", 5.16),
    _cm("CB5+Ba2+",  "Ba2+", "CB5", 6.44),
    _cm("CB5+Ni2+",  "Ni2+", "CB5", 2.73),
    _cm("CB5+Fe3+",  "Fe3+", "CB5", 3.66),
    _cm("CB5+Yb3+",  "Yb3+", "CB5", 3.71),
    _cm("CB5+La3+",  "La3+", "CB5", 4.17),

    # ── CB6 (portal Ø 3.9 Å) ──
    _cm("CB6+Li+",   "Li+",  "CB6", 2.41),
    _cm("CB6+Na+",   "Na+",  "CB6", 3.89),
    _cm("CB6+K+",    "K+",   "CB6", 3.81),
    _cm("CB6+Rb+",   "Rb+",  "CB6", 4.30),
    _cm("CB6+Cs+",   "Cs+",  "CB6", 5.31),
    _cm("CB6+Ag+",   "Ag+",  "CB6", 3.87),
    _cm("CB6+Mg2+",  "Mg2+", "CB6", 2.57),
    _cm("CB6+Ca2+",  "Ca2+", "CB6", 4.22),
    _cm("CB6+Sr2+",  "Sr2+", "CB6", 4.91),
    _cm("CB6+Ba2+",  "Ba2+", "CB6", 5.29),
    _cm("CB6+Ni2+",  "Ni2+", "CB6", 2.59),
    _cm("CB6+Cu2+",  "Cu2+", "CB6", 2.88),
    _cm("CB6+Zn2+",  "Zn2+", "CB6", 2.45),
    _cm("CB6+Al3+",  "Al3+", "CB6", 3.81),
    _cm("CB6+Fe3+",  "Fe3+", "CB6", 5.17),
    _cm("CB6+Yb3+",  "Yb3+", "CB6", 3.50),
    _cm("CB6+La3+",  "La3+", "CB6", 4.16),

    # ── CB7 (portal Ø 5.4 Å) ──
    _cm("CB7+Li+",   "Li+",  "CB7", 2.34),
    _cm("CB7+Na+",   "Na+",  "CB7", 3.41),
    _cm("CB7+K+",    "K+",   "CB7", 3.46),
    _cm("CB7+Rb+",   "Rb+",  "CB7", 3.43),
    _cm("CB7+Cs+",   "Cs+",  "CB7", 3.50),
    _cm("CB7+Ag+",   "Ag+",  "CB7", 3.54),
    _cm("CB7+Mg2+",  "Mg2+", "CB7", 3.24),
    _cm("CB7+Ca2+",  "Ca2+", "CB7", 4.25),
    _cm("CB7+Sr2+",  "Sr2+", "CB7", 4.79),
    _cm("CB7+Ba2+",  "Ba2+", "CB7", 5.28),
    _cm("CB7+Ni2+",  "Ni2+", "CB7", 3.50),
    _cm("CB7+Cu2+",  "Cu2+", "CB7", 3.75),
    _cm("CB7+Zn2+",  "Zn2+", "CB7", 3.40),
    _cm("CB7+Al3+",  "Al3+", "CB7", 2.90),
    _cm("CB7+Fe3+",  "Fe3+", "CB7", 4.18),
    _cm("CB7+Yb3+",  "Yb3+", "CB7", 4.42),
    _cm("CB7+La3+",  "La3+", "CB7", 5.28),

    # ── CB8 (portal Ø 6.9 Å) ──
    _cm("CB8+Li+",   "Li+",  "CB8", 1.69),
    _cm("CB8+Na+",   "Na+",  "CB8", 2.49),
    _cm("CB8+K+",    "K+",   "CB8", 2.66),
    _cm("CB8+Rb+",   "Rb+",  "CB8", 2.64),
    _cm("CB8+Cs+",   "Cs+",  "CB8", 2.55),
    _cm("CB8+Ag+",   "Ag+",  "CB8", 2.32),
    _cm("CB8+Mg2+",  "Mg2+", "CB8", 2.72),
    _cm("CB8+Ca2+",  "Ca2+", "CB8", 3.31),
    _cm("CB8+Sr2+",  "Sr2+", "CB8", 3.63),
    _cm("CB8+Ba2+",  "Ba2+", "CB8", 3.95),
    _cm("CB8+Ni2+",  "Ni2+", "CB8", 2.73),
    _cm("CB8+Cu2+",  "Cu2+", "CB8", 2.86),
    _cm("CB8+Zn2+",  "Zn2+", "CB8", 2.67),
    _cm("CB8+Al3+",  "Al3+", "CB8", 2.90),
    _cm("CB8+Fe3+",  "Fe3+", "CB8", 3.00),
    _cm("CB8+Yb3+",  "Yb3+", "CB8", 3.44),
    _cm("CB8+La3+",  "La3+", "CB8", 3.76),
]


if __name__ == "__main__":
    print(f"Cross-modal dataset: {len(CROSS_MODAL_DATA)} entries")
    from collections import Counter
    host_counts = Counter(e["cb_host"] for e in CROSS_MODAL_DATA)
    metal_counts = Counter(e["metal"] for e in CROSS_MODAL_DATA)
    print(f"\nBy host:")
    for h, c in sorted(host_counts.items()):
        print(f"  {h}: {c}")
    print(f"\nBy metal ({len(metal_counts)} unique):")
    for m, c in sorted(metal_counts.items(), key=lambda x: -x[1]):
        print(f"  {m:6s}: {c}")
    ka_range = [e["log_Ka"] for e in CROSS_MODAL_DATA]
    print(f"\nlog Ka range: {min(ka_range):.2f} to {max(ka_range):.2f}")