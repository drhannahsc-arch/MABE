"""
core/seed_library.py — Sprint 37: Seed Validation Library

~120 hand-entered experimental binding constants from published sources.
Validates the pipeline before bulk database ingestion.

Sources cited per entry. All values at 25°C unless noted.
Units: log Ka (association constant, M⁻¹).

Categories:
  A. Metal coordination (existing 47 + expansion)
  B. Cyclodextrin host-guest (Rekharsky & Inoue 2007)
  C. Cucurbituril host-guest (Barrow et al. 2015, Isaacs group)
  D. Calixarene / pillararene (Schneider & Yatsimirsky 2008)
  E. Protein-ligand benchmark (PDBbind curated)
  F. Crown ether (Izatt 1991, existing + expansion)
"""
from core.universal_schema import UniversalComplex


def build_seed_library():
    """Return the full seed library as list[UniversalComplex]."""
    lib = []
    lib.extend(_metal_coordination())
    lib.extend(_cyclodextrin_host_guest())
    lib.extend(_cucurbituril_host_guest())
    lib.extend(_calixarene_host_guest())
    lib.extend(_protein_ligand())
    lib.extend(_crown_ether())
    print(f"  Seed library: {len(lib)} entries across "
          f"{len(set(uc.binding_mode for uc in lib))} modalities")
    return lib


# ═══════════════════════════════════════════════════════════════════════════
# A. METAL COORDINATION
# Sources: Martell & Smith, NIST SRD 46, Critical Stability Constants
# ═══════════════════════════════════════════════════════════════════════════

def _metal_coordination():
    """Existing 47 + 8 new metal complexes for gap coverage."""
    entries = [
        # === EDTA series (N2O4, 5-membered rings) ===
        _metal("Ca-EDTA", "Ca2+", 2, 0, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 10.7, "Martell & Smith"),
        _metal("Mg-EDTA", "Mg2+", 2, 0, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 8.7, "Martell & Smith"),
        _metal("Mn-EDTA", "Mn2+", 2, 5, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 13.9, "Martell & Smith"),
        _metal("Fe2-EDTA", "Fe2+", 2, 6, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 14.3, "Martell & Smith"),
        _metal("Co-EDTA", "Co2+", 2, 7, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 16.3, "Martell & Smith"),
        _metal("Ni-EDTA", "Ni2+", 2, 8, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 18.6, "Martell & Smith"),
        _metal("Cu-EDTA", "Cu2+", 2, 9, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 18.8, "Martell & Smith"),
        _metal("Zn-EDTA", "Zn2+", 2, 10, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 16.5, "Martell & Smith"),
        _metal("Pb-EDTA", "Pb2+", 2, 0, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 18.0, "Martell & Smith"),
        _metal("Cd-EDTA", "Cd2+", 2, 10, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 16.5, "Martell & Smith"),
        _metal("Fe3-EDTA", "Fe3+", 3, 5, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 25.1, "Martell & Smith"),
        _metal("Al-EDTA", "Al3+", 3, 0, ["N","N","O","O","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 5, 6, [5,5,5,5,5], 16.1, "Martell & Smith"),

        # === en/NH3 series ===
        _metal("Ni-en3", "Ni2+", 2, 8, ["N"]*6, ["N_amine"]*6,
               "borderline", 3, 6, [5,5,5], 18.3, "NIST 46.7"),
        _metal("Cu-en2", "Cu2+", 2, 9, ["N"]*4, ["N_amine"]*4,
               "borderline", 2, 4, [5,5], 19.6, "NIST 46.7",
               geometry="square_planar"),
        _metal("Zn-en3", "Zn2+", 2, 10, ["N"]*6, ["N_amine"]*6,
               "borderline", 3, 6, [5,5,5], 12.1, "NIST 46.7"),
        _metal("Co-en3", "Co2+", 2, 7, ["N"]*6, ["N_amine"]*6,
               "borderline", 3, 6, [5,5,5], 13.9, "NIST 46.7"),
        _metal("Ni-NH3_6", "Ni2+", 2, 8, ["N"]*6, ["N_amine"]*6,
               "borderline", 0, 6, [], 8.6, "NIST 46.7"),

        # === Soft donors ===
        _metal("Hg-cysteine2", "Hg2+", 2, 10, ["S","S","N","N"],
               ["S_thiolate","S_thiolate","N_amine","N_amine"],
               "soft", 0, 4, [], 38.0, "Martell & Smith"),
        _metal("Ag-thiosulfate2", "Ag+", 1, 10, ["S","S"],
               ["S_thiosulfate","S_thiosulfate"],
               "soft", 0, 2, [], 13.5, "NIST 46.7", geometry="linear"),
        _metal("Cd-cysteine", "Cd2+", 2, 10, ["S","N","O"],
               ["S_thiolate","N_amine","O_carboxylate"],
               "mixed", 1, 3, [5], 10.0, "Martell & Smith"),
        _metal("Pb-cysteine", "Pb2+", 2, 0, ["S","N","O"],
               ["S_thiolate","N_amine","O_carboxylate"],
               "mixed", 1, 3, [5], 12.0, "Martell & Smith"),

        # === Hard O-donors ===
        _metal("Fe3-catechol3", "Fe3+", 3, 5, ["O"]*6, ["O_catecholate"]*6,
               "hard", 3, 6, [5,5,5], 43.8, "Martell & Smith"),
        _metal("Fe3-acetohydroxamate3", "Fe3+", 3, 5, ["O"]*6, ["O_hydroxamate"]*6,
               "hard", 3, 6, [5,5,5], 28.3, "NIST 46.7"),
        _metal("Ca-citrate", "Ca2+", 2, 0, ["O","O","O"],
               ["O_carboxylate","O_carboxylate","O_hydroxyl"],
               "hard", 1, 3, [5], 3.5, "NIST 46.7"),
        _metal("Al-catechol3", "Al3+", 3, 0, ["O"]*6, ["O_catecholate"]*6,
               "hard", 3, 6, [5,5,5], 36.0, "Martell & Smith"),

        # === DTPA ===
        _metal("Gd-DTPA", "Gd3+", 3, 7,
               ["N","N","N","O","O","O","O","O"],
               ["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 6, 8, [5,5,5,5,5,5], 22.5, "Martell & Smith"),
        _metal("Cu-DTPA", "Cu2+", 2, 9,
               ["N","N","N","O","O","O","O","O"],
               ["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 6, 8, [5,5,5,5,5,5], 21.5, "Martell & Smith"),

        # === Bipy / phen ===
        _metal("Fe2-bipy3", "Fe2+", 2, 6, ["N"]*6, ["N_pyridine"]*6,
               "borderline", 3, 6, [5,5,5], 17.2, "NIST 46.7"),
        _metal("Ni-bipy3", "Ni2+", 2, 8, ["N"]*6, ["N_pyridine"]*6,
               "borderline", 3, 6, [5,5,5], 20.2, "NIST 46.7"),

        # === Glycinate ===
        _metal("Cu-glycine2", "Cu2+", 2, 9, ["N","N","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
               "mixed", 2, 4, [5,5], 15.1, "NIST 46.7"),
        _metal("Zn-glycine2", "Zn2+", 2, 10, ["N","N","O","O"],
               ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
               "mixed", 2, 4, [5,5], 9.0, "NIST 46.7"),

        # === NEW: expand coverage ===
        _metal("Cu-phen3", "Cu2+", 2, 9, ["N"]*6, ["N_pyridine"]*6,
               "borderline", 3, 6, [5,5,5], 21.0, "NIST 46.7",
               notes="Tris(phenanthroline), stronger π-backbond than bipy"),
        _metal("Fe3-DFO", "Fe3+", 3, 5, ["O"]*6, ["O_hydroxamate"]*6,
               "hard", 3, 6, [5,5,5], 30.6, "Martell & Smith",
               notes="Desferrioxamine B, siderophore"),
        _metal("Cu-IDA", "Cu2+", 2, 9, ["N","O","O"],
               ["N_amine","O_carboxylate","O_carboxylate"],
               "mixed", 2, 3, [5,5], 10.6, "Martell & Smith",
               notes="Iminodiacetate, tridentate"),
        _metal("Ni-IDA", "Ni2+", 2, 8, ["N","O","O"],
               ["N_amine","O_carboxylate","O_carboxylate"],
               "mixed", 2, 3, [5,5], 8.2, "Martell & Smith"),
        _metal("Zn-histidine", "Zn2+", 2, 10, ["N","N","O"],
               ["N_imidazole","N_amine","O_carboxylate"],
               "borderline", 2, 3, [5,5], 6.6, "Martell & Smith",
               notes="Bio-relevant coordination"),
        _metal("Cu-histidine", "Cu2+", 2, 9, ["N","N","O"],
               ["N_imidazole","N_amine","O_carboxylate"],
               "borderline", 2, 3, [5,5], 10.2, "Martell & Smith"),
        _metal("Ni-NTA", "Ni2+", 2, 8, ["N","O","O","O"],
               ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 3, 4, [5,5,5], 11.5, "Martell & Smith",
               notes="Nitrilotriacetate"),
        _metal("Cu-NTA", "Cu2+", 2, 9, ["N","O","O","O"],
               ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"],
               "mixed", 3, 4, [5,5,5], 12.7, "Martell & Smith"),
    ]
    return entries


def _metal(name, formula, charge, d_el, donors, subtypes, dtype, rings, dent,
           ring_sizes, log_K, source, geometry="octahedral",
           is_macrocyclic=False, cavity_radius_nm=0.0, notes=""):
    """Shorthand constructor for metal coordination entries."""
    return UniversalComplex(
        name=name,
        binding_mode="metal_coordination",
        log_Ka_exp=log_K,
        host_name=formula,
        host_type="metal_ion",
        host_charge=charge,
        guest_name=name.split("-", 1)[-1] if "-" in name else name,
        metal_formula=formula,
        metal_charge=charge,
        metal_d_electrons=d_el,
        donor_atoms=donors,
        donor_subtypes=subtypes,
        donor_type=dtype,
        chelate_rings=rings,
        denticity=dent,
        ring_sizes=ring_sizes,
        geometry=geometry,
        is_macrocyclic=is_macrocyclic,
        cavity_radius_nm=cavity_radius_nm,
        source=source,
        phase="Phase1-5",
        confidence="high",
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# B. CYCLODEXTRIN HOST-GUEST
# Source: Rekharsky & Inoue, Chem. Rev. 107, 3715 (2007), Table 3 & 5
# All at 25°C in water, log Ka values
# ═══════════════════════════════════════════════════════════════════════════

def _cyclodextrin_host_guest():
    """β-CD and α-CD binding constants from Rekharsky & Inoue 2007."""
    entries = [
        # ── β-CD ALKYL CHAIN SCAN (Phase 6 primary: isolates γ) ───────
        _hg("β-CD:1-propanol", "beta-CD", "1-propanol", "CCCO",
            1.58, "rekharsky_inoue_2007", series="bCD_alkyl",
            notes="n=3, smallest measurable alkanol"),
        _hg("β-CD:1-butanol", "beta-CD", "1-butanol", "CCCCO",
            1.85, "rekharsky_inoue_2007", series="bCD_alkyl"),
        _hg("β-CD:1-pentanol", "beta-CD", "1-pentanol", "CCCCCO",
            2.12, "rekharsky_inoue_2007", series="bCD_alkyl"),
        _hg("β-CD:1-hexanol", "beta-CD", "1-hexanol", "CCCCCCO",
            2.42, "rekharsky_inoue_2007", series="bCD_alkyl"),
        _hg("β-CD:1-heptanol", "beta-CD", "1-heptanol", "CCCCCCCO",
            2.58, "rekharsky_inoue_2007", series="bCD_alkyl"),
        _hg("β-CD:1-octanol", "beta-CD", "1-octanol", "CCCCCCCCO",
            2.65, "rekharsky_inoue_2007", series="bCD_alkyl",
            notes="Leveling off — chain extends beyond cavity"),

        # ── β-CD SHAPE SCAN (Phase 10: packing coefficient) ──────────
        _hg("β-CD:adamantane-1-carboxylate", "beta-CD", "adamantane-1-carboxylate",
            "OC(=O)C12CC3CC(CC(C3)C1)C2",
            4.25, "rekharsky_inoue_2007", series="bCD_shape",
            notes="Near-optimal fit, PC≈0.52"),
        _hg("β-CD:cyclohexanol", "beta-CD", "cyclohexanol", "OC1CCCCC1",
            2.55, "rekharsky_inoue_2007", series="bCD_shape"),
        _hg("β-CD:cyclopentanol", "beta-CD", "cyclopentanol", "OC1CCCC1",
            2.08, "rekharsky_inoue_2007", series="bCD_shape",
            notes="Smaller than optimal"),
        _hg("β-CD:adamantanol", "beta-CD", "1-adamantanol", "OC12CC3CC(CC(C3)C1)C2",
            3.45, "rekharsky_inoue_2007", series="bCD_shape"),
        _hg("β-CD:norbornane", "beta-CD", "norbornane", "C1CC2CC1CC2",
            2.20, "rekharsky_inoue_2007", series="bCD_shape",
            notes="Compact, undersized for β-CD"),

        # ── β-CD AROMATIC GUESTS (Phase 8: CH-π from CD walls) ────────
        _hg("β-CD:benzene", "beta-CD", "benzene", "c1ccccc1",
            1.56, "rekharsky_inoue_2007", series="bCD_aromatic"),
        _hg("β-CD:toluene", "beta-CD", "toluene", "Cc1ccccc1",
            2.20, "rekharsky_inoue_2007", series="bCD_aromatic"),
        _hg("β-CD:naphthalene", "beta-CD", "naphthalene", "c1ccc2ccccc2c1",
            2.60, "rekharsky_inoue_2007", series="bCD_aromatic"),
        _hg("β-CD:p-nitrophenol", "beta-CD", "p-nitrophenol", "Oc1ccc([N+](=O)[O-])cc1",
            2.72, "rekharsky_inoue_2007", series="bCD_aromatic",
            notes="H-bond donor OH, π-system, size match"),
        _hg("β-CD:p-nitrophenolate", "beta-CD", "p-nitrophenolate", "[O-]c1ccc([N+](=O)[O-])cc1",
            2.86, "rekharsky_inoue_2007", series="bCD_aromatic",
            notes="Anionic, tests electrostatic"),

        # ── α-CD CONTROLS (same guests, smaller cavity) ──────────────
        _hg("α-CD:1-butanol", "alpha-CD", "1-butanol", "CCCCO",
            1.60, "rekharsky_inoue_2007", series="aCD_alkyl",
            notes="Better fit than β-CD for short chains"),
        _hg("α-CD:1-hexanol", "alpha-CD", "1-hexanol", "CCCCCCO",
            2.26, "rekharsky_inoue_2007", series="aCD_alkyl"),
        _hg("α-CD:cyclohexanol", "alpha-CD", "cyclohexanol", "OC1CCCCC1",
            2.38, "rekharsky_inoue_2007", series="aCD_shape",
            notes="Tight fit in α-CD"),
        _hg("α-CD:adamantane-1-carboxylate", "alpha-CD", "adamantane-1-carboxylate",
            "OC(=O)C12CC3CC(CC(C3)C1)C2",
            2.05, "rekharsky_inoue_2007", series="aCD_shape",
            notes="Too large for α-CD, partial inclusion only"),

        # ── γ-CD CONTROLS (same guests, larger cavity) ────────────────
        _hg("γ-CD:adamantane-1-carboxylate", "gamma-CD", "adamantane-1-carboxylate",
            "OC(=O)C12CC3CC(CC(C3)C1)C2",
            2.90, "rekharsky_inoue_2007", series="gCD_shape",
            holdout=True, notes="Rattles in γ-CD, holdout for validation"),
        _hg("γ-CD:naphthalene", "gamma-CD", "naphthalene", "c1ccc2ccccc2c1",
            2.15, "rekharsky_inoue_2007", series="gCD_aromatic",
            holdout=True),
    ]
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# C. CUCURBITURIL HOST-GUEST
# Sources: Barrow et al. Chem. Rev. 115, 12320 (2015)
#          Cao et al. Angew. Chem. 2014
#          Liu et al. JACS 2005
# All at 25°C in water (or 50 mM NaOAc pH 4.74), log Ka values
# ═══════════════════════════════════════════════════════════════════════════

def _cucurbituril_host_guest():
    """CB[7] and CB[6] binding constants."""
    entries = [
        # ── CB[7] HYDROPHOBIC SERIES (Phase 6: cavity hydrophobic) ────
        _hg("CB7:adamantylammonium", "CB7", "1-aminoadamantane·H+",
            "[NH3+]C12CC3CC(CC(C3)C1)C2",
            14.3, "barrow_2015", series="CB7_hydro",
            guest_charge=1,
            n_hbonds_formed=3, hbond_types=["charge_assisted"]*3,
            notes="One of the strongest known host-guest Ka in water"),
        _hg("CB7:ferrocenemethyl-TMA", "CB7", "ferrocenemethyltrimethylammonium",
            "",  # Complex organometallic, no simple SMILES
            15.0, "barrow_2015", series="CB7_hydro",
            guest_charge=1, guest_volume_A3=200,
            notes="Strongest known host-guest: log Ka ~15"),
        _hg("CB7:trimethylsilylmethylammonium", "CB7", "TMSMA",
            "[NH3+]C[Si](C)(C)C",
            7.9, "barrow_2015", series="CB7_hydro", guest_charge=1),
        _hg("CB7:cyclohexylammonium", "CB7", "cyclohexylammonium",
            "[NH3+]C1CCCCC1",
            9.1, "barrow_2015", series="CB7_hydro", guest_charge=1,
            n_hbonds_formed=3, hbond_types=["charge_assisted"]*3),
        _hg("CB7:butylammonium", "CB7", "n-butylammonium",
            "[NH3+]CCCC",
            5.9, "barrow_2015", series="CB7_hydro", guest_charge=1,
            n_hbonds_formed=3, hbond_types=["charge_assisted"]*3),
        _hg("CB7:hexylammonium", "CB7", "n-hexylammonium",
            "[NH3+]CCCCCC",
            7.2, "barrow_2015", series="CB7_hydro", guest_charge=1,
            n_hbonds_formed=3, hbond_types=["charge_assisted"]*3),

        # ── CB[7] H-BOND SERIES (Phase 7: portal H-bonds) ────────────
        _hg("CB7:acetylcholine", "CB7", "acetylcholine",
            "CC(=O)OCC[N+](C)(C)C",
            5.5, "barrow_2015", series="CB7_hbond",
            guest_charge=1,
            notes="NMe3+ cation, minimal portal H-bonding"),
        _hg("CB7:p-xylenediammonium", "CB7", "p-xylenediammonium",
            "[NH3+]Cc1ccc(C[NH3+])cc1",
            9.0, "barrow_2015", series="CB7_hbond",
            guest_charge=2,
            n_hbonds_formed=6, hbond_types=["charge_assisted"]*6,
            notes="Both portals H-bonded, dicationic"),

        # ── CB[6] SERIES (smaller cavity) ─────────────────────────────
        _hg("CB6:hexanediamine-2H", "CB6", "1,6-hexanediamine·2H+",
            "[NH3+]CCCCCC[NH3+]",
            7.9, "barrow_2015", series="CB6_alkyl",
            guest_charge=2,
            n_hbonds_formed=6, hbond_types=["charge_assisted"]*6,
            notes="Threading complex, both portals"),
        _hg("CB6:putrescine-2H", "CB6", "putrescine·2H+",
            "[NH3+]CCCC[NH3+]",
            5.4, "barrow_2015", series="CB6_alkyl", guest_charge=2),
        _hg("CB6:cadaverine-2H", "CB6", "cadaverine·2H+",
            "[NH3+]CCCCC[NH3+]",
            7.2, "barrow_2015", series="CB6_alkyl", guest_charge=2,
            notes="Better chain-length match than putrescine"),
    ]
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# D. CALIXARENE / PILLARARENE
# Source: Schneider & Yatsimirsky, Chem. Soc. Rev. 37, 263 (2008)
#         Arena et al. various
# ═══════════════════════════════════════════════════════════════════════════

def _calixarene_host_guest():
    """Calixarene and pillararene binding data."""
    entries = [
        # ── p-Sulfonatocalix[4]arene cation-π series (Phase 8) ───────
        _hg("SC4A:NMe4", "sulfonato-calix4arene", "tetramethylammonium",
            "C[N+](C)(C)C",
            4.1, "schneider_yatsimirsky_2008", series="SC4A_cation_pi",
            guest_charge=1, n_aromatic_walls=4,
            n_pi_contacts=1, pi_contact_types=["cation_pi"],
            notes="Classic cation-π benchmark"),
        _hg("SC4A:choline", "sulfonato-calix4arene", "choline",
            "OCC[N+](C)(C)C",
            3.7, "schneider_yatsimirsky_2008", series="SC4A_cation_pi",
            guest_charge=1, n_aromatic_walls=4,
            n_pi_contacts=1, pi_contact_types=["cation_pi"]),
        _hg("SC4A:acetylcholine", "sulfonato-calix4arene", "acetylcholine",
            "CC(=O)OCC[N+](C)(C)C",
            3.5, "schneider_yatsimirsky_2008", series="SC4A_cation_pi",
            guest_charge=1, n_aromatic_walls=4,
            n_pi_contacts=1, pi_contact_types=["cation_pi"],
            notes="Bigger than choline, less deep insertion"),
        _hg("SC4A:arginine", "sulfonato-calix4arene", "arginine·H+",
            "NC(CCCNC(=[NH2+])N)C(=O)O",
            3.0, "schneider_yatsimirsky_2008", series="SC4A_cation_pi",
            guest_charge=1, n_aromatic_walls=4),
        _hg("SC4A:lysine", "sulfonato-calix4arene", "lysine·H+",
            "NCCCC(N)C(=O)O",
            2.6, "schneider_yatsimirsky_2008", series="SC4A_cation_pi",
            guest_charge=1, n_aromatic_walls=4),

        # ── Pillar[5]arene alkylammonium scan (Phase 6: hydrophobic) ──
        _hg("P5A:octylammonium", "pillar5arene", "n-octylammonium",
            "[NH3+]CCCCCCCC",
            4.2, "other", series="P5A_alkyl",
            guest_charge=1, n_aromatic_walls=5,
            notes="Threading through electron-rich cavity"),
        _hg("P5A:hexylammonium", "pillar5arene", "n-hexylammonium",
            "[NH3+]CCCCCC",
            3.5, "other", series="P5A_alkyl", guest_charge=1, n_aromatic_walls=5),
        _hg("P5A:butylammonium", "pillar5arene", "n-butylammonium",
            "[NH3+]CCCC",
            2.6, "other", series="P5A_alkyl", guest_charge=1, n_aromatic_walls=5),
    ]
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# E. PROTEIN-LIGAND
# Source: PDBbind v2020 refined set (curated, co-crystal + Kd)
#         Kuntz et al. 1999 (maximum affinity benchmarks)
# These are well-known benchmark complexes with precise Kd measurements.
# ═══════════════════════════════════════════════════════════════════════════

def _protein_ligand():
    """Well-characterized protein-ligand complexes."""
    entries = [
        # ── Serine proteases (well-studied, range of affinities) ──────
        _pl("trypsin:benzamidine", "3PTB", "benzamidine",
            "NC(=[NH2+])c1ccccc1",
            4.2, guest_charge=1, rotatable=1, hbd=2, hba=1, aro=1,
            n_hbonds=3, hbond_types=["charge_assisted","neutral","neutral"],
            notes="Classic benchmark, strong electrostatic"),
        _pl("trypsin:BPTI", "3OTJ", "BPTI",
            "", 13.0, notes="Protein-protein, extreme affinity"),

        # ── HIV protease (drug design benchmark) ──────────────────────
        _pl("HIV-PR:indinavir", "2BPX", "indinavir",
            "CC(C)(C)NC(=O)[C@H]1C[C@@H]2CCCC[C@@H]2CN1C[C@@H](O)[C@H](Cc1cccnc1)NC(=O)[C@@H](CC(=O)N)NC(=O)c1ccc2ccccc2n1",
            9.3, rotatable=12, hbd=4, hba=7, aro=3,
            n_hbonds=5, hbond_types=["neutral"]*5,
            notes="FDA-approved HIV drug, flexible"),
        _pl("HIV-PR:saquinavir", "1HXB", "saquinavir",
            "",
            8.5, rotatable=11, hbd=5, hba=7, aro=3,
            n_hbonds=6, hbond_types=["neutral"]*6),

        # ── Carbonic anhydrase (sulfonamide series, graded affinity) ──
        _pl("CAII:acetazolamide", "3HS4", "acetazolamide",
            "CC(=O)Nc1nnc(S(N)(=O)=O)s1",
            7.0, rotatable=2, hbd=2, hba=5, aro=1,
            n_hbonds=3, hbond_types=["neutral"]*3,
            series="CAII_sulfonamide",
            notes="Classic CAII inhibitor, Zn-binding sulfonamide"),
        _pl("CAII:benzenesulfonamide", "2WEJ", "benzenesulfonamide",
            "NS(=O)(=O)c1ccccc1",
            5.5, rotatable=1, hbd=1, hba=3, aro=1,
            n_hbonds=2, hbond_types=["neutral"]*2,
            series="CAII_sulfonamide"),

        # ── Thrombin (coagulation cascade, clinical target) ───────────
        _pl("thrombin:NAPAP", "1DWD", "NAPAP",
            "",
            8.5, rotatable=8, hbd=3, hba=5, aro=2,
            n_hbonds=4, hbond_types=["neutral"]*4),

        # ── Streptavidin-biotin (ultra-tight, upper limit test) ───────
        _pl("streptavidin:biotin", "3RY2", "biotin",
            "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12",
            15.0, rotatable=5, hbd=3, hba=4, aro=0,
            n_hbonds=8, hbond_types=["neutral"]*8,
            notes="Ka~10^15, extensive H-bond network"),

        # ── Lysozyme-NAG (simple, well-characterized) ─────────────────
        _pl("lysozyme:NAG3", "1HEW", "tri-N-acetylglucosamine",
            "",
            5.0, rotatable=10, hbd=6, hba=10, aro=0,
            n_hbonds=6, hbond_types=["neutral"]*6),

        # ── COX-2 (NSAID binding, clinical relevance) ─────────────────
        _pl("COX2:celecoxib", "3LN1", "celecoxib",
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
            7.8, rotatable=3, hbd=1, hba=5, aro=3,
            n_hbonds=2, hbond_types=["neutral"]*2,
            notes="Selective COX-2, sulfonamide + aromatic π"),
    ]
    return entries


def _pl(name, pdb_id, guest_name, smiles, pKd,
        guest_charge=0, rotatable=0, hbd=0, hba=0, aro=0,
        n_hbonds=0, hbond_types=None, series="", notes=""):
    """Shorthand for protein-ligand entry."""
    return UniversalComplex(
        name=name,
        binding_mode="protein_ligand",
        log_Ka_exp=pKd,  # pKd = -log(Kd) = log(Ka)
        host_name=name.split(":")[0],
        host_type="protein",
        host_pdb_id=pdb_id,
        guest_name=guest_name,
        guest_smiles=smiles,
        guest_charge=guest_charge,
        guest_rotatable_bonds=rotatable,
        guest_n_hbond_donors=hbd,
        guest_n_hbond_acceptors=hba,
        guest_n_aromatic_rings=aro,
        n_hbonds_formed=n_hbonds,
        hbond_types=hbond_types or [],
        source="pdbbind",
        source_id=pdb_id,
        series_id=series or f"pl_{pdb_id}",
        phase="Phase7",
        confidence="high",
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# F. CROWN ETHERS (expanded from existing Sprint 36 entries)
# Source: Izatt et al. Chem. Rev. 91, 1721 (1991), Martell & Smith
# ═══════════════════════════════════════════════════════════════════════════

def _crown_ether():
    """Crown ether and cryptand binding constants."""
    entries = [
        # ── 18-crown-6 size-match series ──────────────────────────────
        _metal("K-18crown6", "K+", 1, 0, ["O"]*6, ["O_ether"]*6,
               "hard", 0, 6, [], 2.0, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.134,
               notes="Size match: K+ (138pm) ≈ 18C6 (134pm)"),
        _metal("Na-18crown6", "Na+", 1, 0, ["O"]*6, ["O_ether"]*6,
               "hard", 0, 6, [], 0.8, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.134,
               notes="Na+ (102pm) too small for 18C6"),
        _metal("Rb-18crown6", "Rb+", 1, 0, ["O"]*6, ["O_ether"]*6,
               "hard", 0, 6, [], 1.5, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.134,
               notes="Rb+ (152pm) slightly too large"),
        _metal("Cs-18crown6", "Cs+", 1, 0, ["O"]*6, ["O_ether"]*6,
               "hard", 0, 6, [], 0.8, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.134,
               notes="Cs+ (167pm) much too large for 18C6"),

        # ── 15-crown-5 ───────────────────────────────────────────────
        _metal("Na-15crown5", "Na+", 1, 0, ["O"]*5, ["O_ether"]*5,
               "hard", 0, 5, [], 0.7, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.086,
               notes="Na+ (102pm) ≈ 15C5 (86pm) — fit"),
        _metal("K-15crown5", "K+", 1, 0, ["O"]*5, ["O_ether"]*5,
               "hard", 0, 5, [], 0.5, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.086,
               notes="K+ too large for 15C5"),

        # ── [2.2.2]cryptand (3D encapsulation, cryptate effect) ───────
        _metal("K-222crypt", "K+", 1, 0, ["O"]*6, ["O_ether"]*6,
               "hard", 0, 6, [], 5.4, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.140,
               notes="Cryptate effect: K+ in [2.2.2] — log K jumps from 2 to 5.4"),
        _metal("Na-222crypt", "Na+", 1, 0, ["O"]*6, ["O_ether"]*6,
               "hard", 0, 6, [], 3.9, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.140,
               notes="Na+ in [2.2.2]"),
        _metal("Ba-18crown6", "Ba2+", 2, 0, ["O"]*6, ["O_ether"]*6,
               "hard", 0, 6, [], 3.7, "Izatt 1991",
               is_macrocyclic=True, cavity_radius_nm=0.134,
               notes="Divalent in 18C6, charge effect test"),
    ]
    return entries


def _hg(name, host, guest, smiles, log_Ka, source, series="",
        guest_charge=0, n_hbonds_formed=0, hbond_types=None,
        n_aromatic_walls=0, n_pi_contacts=0, pi_contact_types=None,
        guest_volume_A3=0, holdout=False, notes=""):
    """Shorthand for host-guest entry."""
    return UniversalComplex(
        name=name,
        binding_mode="host_guest_inclusion",
        log_Ka_exp=log_Ka,
        host_name=host,
        guest_name=guest,
        guest_smiles=smiles,
        guest_charge=guest_charge,
        guest_volume_A3=guest_volume_A3,
        n_hbonds_formed=n_hbonds_formed,
        hbond_types=hbond_types or [],
        n_aromatic_walls=n_aromatic_walls,
        n_pi_contacts=n_pi_contacts,
        pi_contact_types=pi_contact_types or [],
        source=source,
        series_id=series,
        phase="Phase6",
        holdout=holdout,
        confidence="high",
        notes=notes,
    )
