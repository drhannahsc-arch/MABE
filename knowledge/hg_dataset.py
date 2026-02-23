"""
hg_dataset.py — Host-guest calibration dataset for MABE Phase 6.

Curated from Rekharsky & Inoue (Chem. Rev. 2007), Barrow et al. (Chem. Soc. Rev. 2015),
Szejtli (Chem. Rev. 1998), Isaacs (Chem. Commun. 2009), and primary literature.

All log Ka values at 25°C in aqueous solution (pH near neutral unless noted).
Guest SMILES for RDKit SASA computation.

Separate from cal_dataset.py (metal-only). Merged at Phase 11.
"""

# ═══════════════════════════════════════════════════════════════════════════
# HOST PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════
# cavity_diameter: inner diameter in Å (from crystal structures)
# cavity_depth: approximate depth in Å
# cavity_sasa: interior SASA in Å² (estimated from geometric model)
# portal_type: chemistry at entrance (affects H-bonding, Phase 7)
# curvature_class: "concave" (deep cavity), "shallow", "open"

HOST_DB = {
    "alpha-CD": {
        "full_name": "α-cyclodextrin",
        "cavity_diameter": 4.7,
        "cavity_depth": 7.9,
        "cavity_sasa": 174.0,   # π × 2.35 × 7.9 × 2 (cylinder interior)
        "portal_type": "hydroxyl",
        "curvature_class": "concave",
        "n_glucose": 6,
    },
    "beta-CD": {
        "full_name": "β-cyclodextrin",
        "cavity_diameter": 6.0,
        "cavity_depth": 7.9,
        "cavity_sasa": 285.0,
        "portal_type": "hydroxyl",
        "curvature_class": "concave",
        "n_glucose": 7,
    },
    "gamma-CD": {
        "full_name": "γ-cyclodextrin",
        "cavity_diameter": 7.5,
        "cavity_depth": 7.9,
        "cavity_sasa": 446.0,
        "portal_type": "hydroxyl",
        "curvature_class": "concave",
        "n_glucose": 8,
    },
    "CB6": {
        "full_name": "cucurbit[6]uril",
        "cavity_diameter": 5.8,
        "cavity_depth": 9.1,
        "cavity_sasa": 264.0,
        "portal_type": "carbonyl",
        "curvature_class": "concave",
        "n_glycoluril": 6,
    },
    "CB7": {
        "full_name": "cucurbit[7]uril",
        "cavity_diameter": 7.3,
        "cavity_depth": 9.1,
        "cavity_sasa": 418.0,
        "portal_type": "carbonyl",
        "curvature_class": "concave",
        "n_glycoluril": 7,
    },
    "CB8": {
        "full_name": "cucurbit[8]uril",
        "cavity_diameter": 8.8,
        "cavity_depth": 9.1,
        "cavity_sasa": 609.0,
        "portal_type": "carbonyl",
        "curvature_class": "concave",
        "n_glycoluril": 8,
    },
    "calix4-SO3": {
        "full_name": "p-sulfonatocalix[4]arene",
        "cavity_diameter": 5.0,
        "cavity_depth": 5.5,
        "cavity_sasa": 172.0,
        "portal_type": "sulfonate",
        "curvature_class": "shallow",
        "n_arene": 4,
    },
    "pillar5": {
        "full_name": "pillar[5]arene",
        "cavity_diameter": 4.7,
        "cavity_depth": 5.5,
        "cavity_sasa": 163.0,
        "portal_type": "methoxy",
        "curvature_class": "concave",
        "n_arene": 5,
    },
}


def _hg(name, host, guest_smiles, guest_charge, log_Ka, source,
        n_hbonds_portal=0, guest_has_cation=False):
    """Create a host-guest calibration entry.

    Args:
        name: descriptive label
        host: key into HOST_DB
        guest_smiles: SMILES for RDKit SASA computation
        guest_charge: net charge on guest (for electrostatic term)
        log_Ka: log10 of association constant
        source: literature citation
        n_hbonds_portal: estimated H-bonds at portal (for Phase 7 subtraction)
        guest_has_cation: guest bears cationic charge (ion-dipole at portal)
    """
    return {
        "name": name,
        "host": host,
        "guest_smiles": guest_smiles,
        "guest_charge": guest_charge,
        "log_Ka": log_Ka,
        "source": source,
        "n_hbonds_portal": n_hbonds_portal,
        "guest_has_cation": guest_has_cation,
    }


HG_DATA = [

    # ═════════════════════════════════════════════════════════════════
    # α-CYCLODEXTRIN (cavity Ø 4.7 Å, depth 7.9 Å)
    # Small cavity — only linear or small cyclic guests fit
    # ═════════════════════════════════════════════════════════════════
    _hg("aCD+1-butanol", "alpha-CD", "CCCCO", 0, 1.61,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+1-pentanol", "alpha-CD", "CCCCCO", 0, 2.06,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+1-hexanol", "alpha-CD", "CCCCCCO", 0, 2.30,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+1-heptanol", "alpha-CD", "CCCCCCCO", 0, 2.55,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+1-octanol", "alpha-CD", "CCCCCCCCO", 0, 2.51,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+cyclohexanol", "alpha-CD", "OC1CCCCC1", 0, 2.38,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+p-nitrophenol", "alpha-CD", "O=[N+]([O-])c1ccc(O)cc1", 0, 2.36,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+benzoic-acid", "alpha-CD", "OC(=O)c1ccccc1", 0, 1.83,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+phenol", "alpha-CD", "Oc1ccccc1", 0, 1.50,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+p-cresol", "alpha-CD", "Cc1ccc(O)cc1", 0, 1.74,
         "Rekharsky2007", n_hbonds_portal=1),

    # ═════════════════════════════════════════════════════════════════
    # β-CYCLODEXTRIN (cavity Ø 6.0 Å, depth 7.9 Å)
    # Workhorse — accommodates most single-ring and adamantyl guests
    # ═════════════════════════════════════════════════════════════════
    _hg("bCD+adamantane-COOH", "beta-CD", "OC(=O)C12CC3CC(CC(C3)C1)C2", 0, 3.97,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+1-adamantanol", "beta-CD", "OC12CC3CC(CC(C3)C1)C2", 0, 3.65,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+adamantane-NH3+", "beta-CD", "[NH3+]C12CC3CC(CC(C3)C1)C2", 1, 4.23,
         "Rekharsky2007", n_hbonds_portal=2, guest_has_cation=True),
    _hg("bCD+cyclohexanol", "beta-CD", "OC1CCCCC1", 0, 2.73,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+p-nitrophenol", "beta-CD", "O=[N+]([O-])c1ccc(O)cc1", 0, 2.69,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+1-butanol", "beta-CD", "CCCCO", 0, 1.58,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+1-hexanol", "beta-CD", "CCCCCCO", 0, 2.67,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+1-octanol", "beta-CD", "CCCCCCCCO", 0, 3.08,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+naphthalene", "beta-CD", "c1ccc2ccccc2c1", 0, 2.60,
         "Rekharsky2007"),
    _hg("bCD+benzoic-acid", "beta-CD", "OC(=O)c1ccccc1", 0, 1.82,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+phenol", "beta-CD", "Oc1ccccc1", 0, 1.78,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+p-cresol", "beta-CD", "Cc1ccc(O)cc1", 0, 2.10,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+toluene", "beta-CD", "Cc1ccccc1", 0, 1.55,
         "Rekharsky2007"),
    _hg("bCD+ibuprofen", "beta-CD", "CC(C)Cc1ccc(cc1)C(C)C(O)=O", 0, 3.21,
         "Loftsson2005", n_hbonds_portal=1),
    _hg("bCD+naproxen", "beta-CD", "COc1ccc2cc(ccc2c1)C(C)C(O)=O", 0, 3.12,
         "Loftsson2005", n_hbonds_portal=1),
    _hg("bCD+cholesterol", "beta-CD", "CC(CCCC(C)C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", 0, 2.41,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("bCD+pyrene", "beta-CD", "c1cc2ccc3cccc4ccc(c1)c2c34", 0, 2.11,
         "Connors1997"),

    # ═════════════════════════════════════════════════════════════════
    # γ-CYCLODEXTRIN (cavity Ø 7.5 Å, depth 7.9 Å)
    # Large cavity — fits polycyclic aromatics, loose for small guests
    # ═════════════════════════════════════════════════════════════════
    _hg("gCD+pyrene", "gamma-CD", "c1cc2ccc3cccc4ccc(c1)c2c34", 0, 3.45,
         "Connors1997"),
    _hg("gCD+naphthalene", "gamma-CD", "c1ccc2ccccc2c1", 0, 1.41,
         "Rekharsky2007"),
    _hg("gCD+anthracene", "gamma-CD", "c1ccc2cc3ccccc3cc2c1", 0, 3.04,
         "Connors1997"),
    _hg("gCD+1-adamantanol", "gamma-CD", "OC12CC3CC(CC(C3)C1)C2", 0, 2.08,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("gCD+cyclohexanol", "gamma-CD", "OC1CCCCC1", 0, 1.26,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("gCD+1-octanol", "gamma-CD", "CCCCCCCCO", 0, 2.18,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("gCD+phenol", "gamma-CD", "Oc1ccccc1", 0, 0.98,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("gCD+cholesterol", "gamma-CD", "CC(CCCC(C)C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", 0, 3.65,
         "Rekharsky2007", n_hbonds_portal=1),

    # ═════════════════════════════════════════════════════════════════
    # CB[7] (cavity Ø 7.3 Å, depth 9.1 Å)
    # Rigid, hydrophobic interior, carbonyl portals
    # ═════════════════════════════════════════════════════════════════
    _hg("CB7+adamantane-NH3+", "CB7",
         "[NH3+]C12CC3CC(CC(C3)C1)C2", 1, 14.3,
         "Liu2005", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB7+1-aminoadamantane", "CB7",
         "NC12CC3CC(CC(C3)C1)C2", 0, 12.6,
         "Liu2005", n_hbonds_portal=2),
    _hg("CB7+dimethyladamantane-NH3+", "CB7",
         "[NH3+]C12CC3CC(C)(CC(C3)C1)C2", 1, 12.0,
         "Isaacs2009", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB7+ferrocene-methyl-NH3+", "CB7",
         "[NH3+]CC1=CC=CC1", 1, 11.8,
         "Jeon2005", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB7+bicyclo222octane-NH3+", "CB7",
         "[NH3+]C12CCC(CC1)CC2", 1, 10.2,
         "Isaacs2009", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB7+cyclohexyl-NH3+", "CB7",
         "[NH3+]C1CCCCC1", 1, 7.9,
         "Isaacs2009", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB7+hexyl-NH3+", "CB7",
         "[NH3+]CCCCCC", 1, 6.1,
         "Isaacs2009", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB7+butyl-NH3+", "CB7",
         "[NH3+]CCCC", 1, 5.2,
         "Isaacs2009", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB7+propyl-NH3+", "CB7",
         "[NH3+]CCC", 1, 4.3,
         "Isaacs2009", n_hbonds_portal=3, guest_has_cation=True),

    # ═════════════════════════════════════════════════════════════════
    # CB[6] (cavity Ø 5.8 Å, depth 9.1 Å)
    # Smaller — mainly linear alkylammonium guests
    # ═════════════════════════════════════════════════════════════════
    _hg("CB6+hexanediamine-2H+", "CB6",
         "[NH3+]CCCCCC[NH3+]", 2, 8.1,
         "Mock1985", n_hbonds_portal=6, guest_has_cation=True),
    _hg("CB6+pentanediamine-2H+", "CB6",
         "[NH3+]CCCCC[NH3+]", 2, 7.3,
         "Mock1985", n_hbonds_portal=6, guest_has_cation=True),
    _hg("CB6+butanediamine-2H+", "CB6",
         "[NH3+]CCCC[NH3+]", 2, 6.7,
         "Mock1985", n_hbonds_portal=6, guest_has_cation=True),
    _hg("CB6+propanediamine-2H+", "CB6",
         "[NH3+]CCC[NH3+]", 2, 5.5,
         "Mock1985", n_hbonds_portal=6, guest_has_cation=True),
    _hg("CB6+butyl-NH3+", "CB6",
         "[NH3+]CCCC", 1, 4.8,
         "Mock1985", n_hbonds_portal=3, guest_has_cation=True),
    _hg("CB6+hexyl-NH3+", "CB6",
         "[NH3+]CCCCCC", 1, 5.6,
         "Mock1985", n_hbonds_portal=3, guest_has_cation=True),

    # ═════════════════════════════════════════════════════════════════
    # p-SULFONATOCALIX[4]ARENE (anionic host, shallow cavity)
    # ═════════════════════════════════════════════════════════════════
    _hg("sCX4+trimethyl-NH+", "calix4-SO3",
         "C[NH+](C)C", 1, 3.90,
         "Douteau2008", guest_has_cation=True),
    _hg("sCX4+tetramethyl-N+", "calix4-SO3",
         "C[N+](C)(C)C", 1, 4.53,
         "Douteau2008", guest_has_cation=True),
    _hg("sCX4+acetylcholine", "calix4-SO3",
         "CC(=O)OCC[N+](C)(C)C", 1, 4.26,
         "Douteau2008", guest_has_cation=True),
    _hg("sCX4+choline", "calix4-SO3",
         "OCC[N+](C)(C)C", 1, 3.72,
         "Douteau2008", n_hbonds_portal=1, guest_has_cation=True),
    _hg("sCX4+pyridinium", "calix4-SO3",
         "c1cc[nH+]cc1", 1, 2.80,
         "Douteau2008", guest_has_cation=True),
    _hg("sCX4+methyl-pyridinium", "calix4-SO3",
         "C[n+]1ccccc1", 1, 3.35,
         "Douteau2008", guest_has_cation=True),

    # ═════════════════════════════════════════════════════════════════
    # PILLAR[5]ARENE (neutral, tubular cavity)
    # ═════════════════════════════════════════════════════════════════
    _hg("P5+hexanediamine-2H+", "pillar5",
         "[NH3+]CCCCCC[NH3+]", 2, 4.1,
         "Ogoshi2012", n_hbonds_portal=4, guest_has_cation=True),
    _hg("P5+octanediamine-2H+", "pillar5",
         "[NH3+]CCCCCCCC[NH3+]", 2, 5.0,
         "Ogoshi2012", n_hbonds_portal=4, guest_has_cation=True),
    _hg("P5+butanediamine-2H+", "pillar5",
         "[NH3+]CCCC[NH3+]", 2, 3.2,
         "Ogoshi2012", n_hbonds_portal=4, guest_has_cation=True),
    _hg("P5+hexyl-NH3+", "pillar5",
         "[NH3+]CCCCCC", 1, 3.5,
         "Ogoshi2012", n_hbonds_portal=2, guest_has_cation=True),

    # ═════════════════════════════════════════════════════════════════
    # PHASE 7 MATCHED PAIRS: H-bond isolation
    # Non-H-bonding analogs of existing guests
    # ═════════════════════════════════════════════════════════════════

    # ── β-CD: H-bonding vs non-H-bonding matched guests ──
    # nitrobenzene (no OH) vs p-nitrophenol (has OH) — same ring, same NO2
    _hg("bCD+nitrobenzene", "beta-CD", "O=[N+]([O-])c1ccccc1", 0, 1.62,
         "Rekharsky2007"),
    # anisole (no OH) vs phenol (has OH) — same size aromatic
    _hg("bCD+anisole", "beta-CD", "COc1ccccc1", 0, 1.50,
         "Rekharsky2007"),
    # tert-butanol vs neopentane — matched size, ±OH
    _hg("bCD+neopentane", "beta-CD", "CC(C)(C)C", 0, 1.15,
         "Rekharsky2007"),
    _hg("bCD+tert-butanol", "beta-CD", "CC(C)(C)O", 0, 1.42,
         "Rekharsky2007", n_hbonds_portal=1),
    # cyclohexane vs cyclohexanol — classic pair
    _hg("bCD+cyclohexane", "beta-CD", "C1CCCCC1", 0, 2.04,
         "Rekharsky2007"),
    # adamantane (no HB) — matched to 1-adamantanol
    _hg("bCD+adamantane", "beta-CD", "C1C2CC3CC1CC(C2)C3", 0, 3.30,
         "Rekharsky2007"),
    # methyl benzoate (no COOH donor) vs benzoic acid
    _hg("bCD+methylbenzoate", "beta-CD", "COC(=O)c1ccccc1", 0, 1.85,
         "Rekharsky2007"),
    # 1-naphthol (HB) vs naphthalene (no HB)
    _hg("bCD+1-naphthol", "beta-CD", "Oc1cccc2ccccc12", 0, 2.90,
         "Rekharsky2007", n_hbonds_portal=1),

    # ── α-CD: matched pairs ──
    _hg("aCD+nitrobenzene", "alpha-CD", "O=[N+]([O-])c1ccccc1", 0, 1.37,
         "Rekharsky2007"),
    _hg("aCD+anisole", "alpha-CD", "COc1ccccc1", 0, 1.30,
         "Rekharsky2007"),
    _hg("aCD+1-naphthol", "alpha-CD", "Oc1cccc2ccccc12", 0, 2.50,
         "Rekharsky2007", n_hbonds_portal=1),
    _hg("aCD+naphthalene", "alpha-CD", "c1ccc2ccccc2c1", 0, 2.01,
         "Rekharsky2007"),

    # ── γ-CD: matched pairs ──
    _hg("gCD+nitrobenzene", "gamma-CD", "O=[N+]([O-])c1ccccc1", 0, 0.70,
         "Rekharsky2007"),

    # ── CB[7]: non-cationic guests (isolate charge-assisted HB) ──
    # These have ONLY hydrophobic driving force — no portal HB
    _hg("CB7+adamantane", "CB7",
         "C1C2CC3CC1CC(C2)C3", 0, 7.3,
         "Moghaddam2011"),
    _hg("CB7+adamantanone", "CB7",
         "O=C1C2CC3CC1CC(C2)C3", 0, 8.2,
         "Moghaddam2011"),
    _hg("CB7+cyclopentanone", "CB7",
         "O=C1CCCC1", 0, 3.7,
         "Moghaddam2011"),
    _hg("CB7+cyclohexanone", "CB7",
         "O=C1CCCCC1", 0, 5.1,
         "Moghaddam2011"),
    _hg("CB7+norbornane", "CB7",
         "C1CC2CC1CC2", 0, 5.8,
         "Moghaddam2011"),

    # ── CB[7]: neutral H-bond donors (OH, NH but not NH3+) ──
    _hg("CB7+1-adamantanol", "CB7",
         "OC12CC3CC(CC(C3)C1)C2", 0, 9.5,
         "Moghaddam2011", n_hbonds_portal=1),
    _hg("CB7+cyclohexanol", "CB7",
         "OC1CCCCC1", 0, 5.8,
         "Moghaddam2011", n_hbonds_portal=1),
]


if __name__ == "__main__":
    print(f"Host-guest dataset: {len(HG_DATA)} entries")
    from collections import Counter
    host_counts = Counter(e["host"] for e in HG_DATA)
    print(f"Hosts: {len(host_counts)}")
    for h, c in host_counts.most_common():
        print(f"  {h:15s} {c:3d} entries")
    ka_range = [e["log_Ka"] for e in HG_DATA]
    print(f"log Ka range: {min(ka_range):.2f} to {max(ka_range):.2f}")
    charged = sum(1 for e in HG_DATA if e["guest_charge"] != 0)
    print(f"Charged guests: {charged}/{len(HG_DATA)}")