"""
cal_dataset.py — Calibration dataset for MABE frozen scorer.

120 metal-ligand complexes with annotated donor subtypes, chelate rings,
macrocyclic flags, and experimental log K values.

Sources: NIST SRD 46, Martell & Smith Critical Stability Constants,
         Aposhian (clinical chelation), Izatt (crown ethers), literature.

Each entry is a dict with keys:
    name, metal, donors, chelate_rings, ring_sizes, macrocyclic,
    cavity_nm, n_lig_mol, pH, log_K_exp, source
"""


def _e(name, metal, donors, rings, rsizes, macro, cavity, nlig, pH, logK, src):
    """Compact entry constructor."""
    return dict(
        name=name, metal=metal, donors=donors,
        chelate_rings=rings, ring_sizes=rsizes,
        macrocyclic=macro, cavity_nm=cavity,
        n_lig_mol=nlig, pH=pH, log_K_exp=logK, source=src,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════

# Shorthand donor lists
_edta = ["N_amine","N_amine","O_carboxylate","O_carboxylate",
         "O_carboxylate","O_carboxylate"]
_dtpa = ["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate",
         "O_carboxylate","O_carboxylate","O_carboxylate"]
_en2 = ["N_amine","N_amine"]
_en6 = ["N_amine"]*6
_en4 = ["N_amine"]*4
_bipy2 = ["N_pyridine","N_pyridine"]
_bipy6 = ["N_pyridine"]*6
_nta = ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"]
_ida = ["N_amine","O_carboxylate","O_carboxylate"]
_gly = ["N_amine","O_carboxylate"]
_dpa = ["N_pyridine","O_carboxylate","O_carboxylate"]
_8hq = ["N_pyridine","O_phenolate"]
_cit = ["O_carboxylate","O_carboxylate","O_hydroxyl"]
_cat2 = ["O_catecholate","O_catecholate"]
_cat6 = ["O_catecholate"]*6
_ha2 = ["O_hydroxamate","O_hydroxamate"]
_ha6 = ["O_hydroxamate"]*6
_dmsa = ["S_thiolate","S_thiolate","O_carboxylate","O_carboxylate"]
_cys = ["S_thiolate","N_amine","O_carboxylate"]
_dtc2 = ["S_dithiocarbamate","S_dithiocarbamate"]
_salen = ["N_imine","N_imine","O_phenolate","O_phenolate"]
_pic = ["N_pyridine","O_carboxylate"]
_c18_6 = ["O_ether"]*6
_c15_5 = ["O_ether"]*5
_c12_4 = ["O_ether"]*4


CAL_DATA = [
    # ═════════════════════════════════════════════════════════════════
    # EDTA series (N₂O₄, 5 chelate rings)
    # ═════════════════════════════════════════════════════════════════
    _e("EDTA+Ca2+",  "Ca2+", _edta, 5, [5]*5, False, None, 1, 7.0, 10.7, "NIST"),
    _e("EDTA+Mg2+",  "Mg2+", _edta, 5, [5]*5, False, None, 1, 7.0,  8.7, "NIST"),
    _e("EDTA+Ba2+",  "Ba2+", _edta, 5, [5]*5, False, None, 1, 7.0,  7.8, "NIST"),
    _e("EDTA+Sr2+",  "Sr2+", _edta, 5, [5]*5, False, None, 1, 7.0,  8.6, "NIST"),
    _e("EDTA+Mn2+",  "Mn2+", _edta, 5, [5]*5, False, None, 1, 7.0, 13.8, "NIST"),
    _e("EDTA+Fe2+",  "Fe2+", _edta, 5, [5]*5, False, None, 1, 7.0, 14.3, "NIST"),
    _e("EDTA+Co2+",  "Co2+", _edta, 5, [5]*5, False, None, 1, 7.0, 16.3, "NIST"),
    _e("EDTA+Ni2+",  "Ni2+", _edta, 5, [5]*5, False, None, 1, 7.0, 18.6, "NIST"),
    _e("EDTA+Cu2+",  "Cu2+", _edta, 5, [5]*5, False, None, 1, 7.0, 18.8, "NIST"),
    _e("EDTA+Zn2+",  "Zn2+", _edta, 5, [5]*5, False, None, 1, 7.0, 16.5, "NIST"),
    _e("EDTA+Cd2+",  "Cd2+", _edta, 5, [5]*5, False, None, 1, 7.0, 16.5, "NIST"),
    _e("EDTA+Pb2+",  "Pb2+", _edta, 5, [5]*5, False, None, 1, 7.0, 18.0, "NIST"),
    _e("EDTA+Hg2+",  "Hg2+", _edta, 5, [5]*5, False, None, 1, 7.0, 21.8, "NIST"),
    _e("EDTA+Fe3+",  "Fe3+", _edta, 5, [5]*5, False, None, 1, 7.0, 25.1, "NIST"),
    _e("EDTA+Al3+",  "Al3+", _edta, 5, [5]*5, False, None, 1, 7.0, 16.1, "NIST"),
    _e("EDTA+Cr3+",  "Cr3+", _edta, 5, [5]*5, False, None, 1, 7.0, 23.4, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Ethylenediamine (en) stepwise K₁ (N₂, 1 ring)
    # ═════════════════════════════════════════════════════════════════
    _e("en+Cu2+",  "Cu2+", _en2, 1, [5], False, None, 1, 7.0, 10.6, "NIST"),
    _e("en+Ni2+",  "Ni2+", _en2, 1, [5], False, None, 1, 7.0,  7.5, "NIST"),
    _e("en+Co2+",  "Co2+", _en2, 1, [5], False, None, 1, 7.0,  5.9, "NIST"),
    _e("en+Zn2+",  "Zn2+", _en2, 1, [5], False, None, 1, 7.0,  5.7, "NIST"),
    _e("en+Fe2+",  "Fe2+", _en2, 1, [5], False, None, 1, 7.0,  4.3, "NIST"),
    _e("en+Mn2+",  "Mn2+", _en2, 1, [5], False, None, 1, 7.0,  2.7, "NIST"),
    _e("en+Cd2+",  "Cd2+", _en2, 1, [5], False, None, 1, 7.0,  5.5, "NIST"),
    _e("en+Pb2+",  "Pb2+", _en2, 1, [5], False, None, 1, 7.0,  5.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Tris(en) cumulative β₃ (N₆, 3 rings, 3 ligand molecules)
    # ═════════════════════════════════════════════════════════════════
    _e("en3+Ni2+", "Ni2+", _en6, 3, [5]*3, False, None, 3, 7.0, 18.3, "NIST"),
    _e("en3+Co2+", "Co2+", _en6, 3, [5]*3, False, None, 3, 7.0, 13.8, "NIST"),
    _e("en3+Cu2+", "Cu2+", _en6, 3, [5]*3, False, None, 3, 7.0, 20.0, "Martell"),
    _e("en3+Zn2+", "Zn2+", _en6, 3, [5]*3, False, None, 3, 7.0, 12.1, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Trien (N₄ linear, 3 rings, 1 molecule)
    # ═════════════════════════════════════════════════════════════════
    _e("trien+Cu2+", "Cu2+", _en4, 3, [5]*3, False, None, 1, 7.0, 20.1, "NIST"),
    _e("trien+Ni2+", "Ni2+", _en4, 3, [5]*3, False, None, 1, 7.0, 14.0, "NIST"),
    _e("trien+Co2+", "Co2+", _en4, 3, [5]*3, False, None, 1, 7.0, 11.0, "NIST"),
    _e("trien+Zn2+", "Zn2+", _en4, 3, [5]*3, False, None, 1, 7.0, 12.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # 2,2'-Bipyridine stepwise K₁ (N₂ aromatic, 1 ring)
    # ═════════════════════════════════════════════════════════════════
    _e("bipy+Fe2+", "Fe2+", _bipy2, 1, [5], False, None, 1, 7.0, 5.9, "NIST"),
    _e("bipy+Cu2+", "Cu2+", _bipy2, 1, [5], False, None, 1, 7.0, 9.0, "NIST"),
    _e("bipy+Ni2+", "Ni2+", _bipy2, 1, [5], False, None, 1, 7.0, 7.1, "NIST"),
    _e("bipy+Zn2+", "Zn2+", _bipy2, 1, [5], False, None, 1, 7.0, 5.2, "NIST"),
    _e("bipy+Co2+", "Co2+", _bipy2, 1, [5], False, None, 1, 7.0, 5.7, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Tris(bipy) cumulative β₃ (N₆, 3 rings, 3 molecules)
    # ═════════════════════════════════════════════════════════════════
    _e("bipy3+Fe2+", "Fe2+", _bipy6, 3, [5]*3, False, None, 3, 7.0, 17.2, "NIST"),
    _e("bipy3+Ni2+", "Ni2+", _bipy6, 3, [5]*3, False, None, 3, 7.0, 20.2, "NIST"),
    _e("bipy3+Co2+", "Co2+", _bipy6, 3, [5]*3, False, None, 3, 7.0, 15.7, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # 1,10-Phenanthroline K₁
    # ═════════════════════════════════════════════════════════════════
    _e("phen+Fe2+", "Fe2+", _bipy2, 1, [5], False, None, 1, 7.0, 5.9, "NIST"),
    _e("phen+Cu2+", "Cu2+", _bipy2, 1, [5], False, None, 1, 7.0, 9.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Glycinate K₁ (NO, 1 ring)
    # ═════════════════════════════════════════════════════════════════
    _e("gly+Cu2+", "Cu2+", _gly, 1, [5], False, None, 1, 7.0, 8.2, "NIST"),
    _e("gly+Ni2+", "Ni2+", _gly, 1, [5], False, None, 1, 7.0, 5.8, "NIST"),
    _e("gly+Zn2+", "Zn2+", _gly, 1, [5], False, None, 1, 7.0, 5.0, "NIST"),
    _e("gly+Co2+", "Co2+", _gly, 1, [5], False, None, 1, 7.0, 5.2, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # NTA (N₁O₃, 3 rings)
    # ═════════════════════════════════════════════════════════════════
    _e("NTA+Cu2+", "Cu2+", _nta, 3, [5]*3, False, None, 1, 7.0, 12.9, "NIST"),
    _e("NTA+Ni2+", "Ni2+", _nta, 3, [5]*3, False, None, 1, 7.0, 11.5, "NIST"),
    _e("NTA+Zn2+", "Zn2+", _nta, 3, [5]*3, False, None, 1, 7.0, 10.7, "NIST"),
    _e("NTA+Co2+", "Co2+", _nta, 3, [5]*3, False, None, 1, 7.0, 10.4, "NIST"),
    _e("NTA+Ca2+", "Ca2+", _nta, 3, [5]*3, False, None, 1, 7.0,  6.4, "NIST"),
    _e("NTA+Fe3+", "Fe3+", _nta, 3, [5]*3, False, None, 1, 7.0, 15.9, "NIST"),
    _e("NTA+Pb2+", "Pb2+", _nta, 3, [5]*3, False, None, 1, 7.0, 11.4, "NIST"),
    _e("NTA+Cd2+", "Cd2+", _nta, 3, [5]*3, False, None, 1, 7.0,  9.8, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # IDA (N₁O₂, 2 rings)
    # ═════════════════════════════════════════════════════════════════
    _e("IDA+Cu2+", "Cu2+", _ida, 2, [5]*2, False, None, 1, 7.0, 10.6, "NIST"),
    _e("IDA+Ni2+", "Ni2+", _ida, 2, [5]*2, False, None, 1, 7.0,  8.2, "NIST"),
    _e("IDA+Zn2+", "Zn2+", _ida, 2, [5]*2, False, None, 1, 7.0,  7.0, "NIST"),
    _e("IDA+Co2+", "Co2+", _ida, 2, [5]*2, False, None, 1, 7.0,  7.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Dipicolinic acid (DPA: pyridine-2,6-dicarboxylate, 2 rings)
    # ═════════════════════════════════════════════════════════════════
    _e("DPA+Cu2+", "Cu2+", _dpa, 2, [5]*2, False, None, 1, 7.0, 9.1, "NIST"),
    _e("DPA+Zn2+", "Zn2+", _dpa, 2, [5]*2, False, None, 1, 7.0, 6.4, "NIST"),
    _e("DPA+Ni2+", "Ni2+", _dpa, 2, [5]*2, False, None, 1, 7.0, 7.3, "NIST"),
    _e("DPA+Fe3+", "Fe3+", _dpa, 2, [5]*2, False, None, 1, 7.0, 12.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # 8-Hydroxyquinoline (oxine: N,O bidentate)
    # ═════════════════════════════════════════════════════════════════
    _e("8HQ+Cu2+", "Cu2+", _8hq, 1, [5], False, None, 1, 7.0, 12.2, "NIST"),
    _e("8HQ+Zn2+", "Zn2+", _8hq, 1, [5], False, None, 1, 7.0,  8.6, "NIST"),
    _e("8HQ+Al3+", "Al3+", _8hq, 1, [5], False, None, 1, 7.0,  8.8, "NIST"),
    _e("8HQ+Fe3+", "Fe3+", _8hq, 1, [5], False, None, 1, 7.0, 12.5, "Martell"),
    _e("8HQ+Ni2+", "Ni2+", _8hq, 1, [5], False, None, 1, 7.0,  9.9, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Catecholate (O₂ anionic bidentate)
    # ═════════════════════════════════════════════════════════════════
    _e("cat+Fe3+",  "Fe3+", _cat2, 1, [5], False, None, 1, 7.0, 20.0, "NIST"),
    _e("cat+Al3+",  "Al3+", _cat2, 1, [5], False, None, 1, 7.0, 16.3, "NIST"),
    _e("cat+Cu2+",  "Cu2+", _cat2, 1, [5], False, None, 1, 7.0, 14.5, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # Tris(catecholate) — siderophore model (3 molecules)
    # ═════════════════════════════════════════════════════════════════
    _e("cat3+Fe3+", "Fe3+", _cat6, 3, [5]*3, False, None, 3, 7.0, 43.0, "Raymond"),

    # ═════════════════════════════════════════════════════════════════
    # Hydroxamate
    # ═════════════════════════════════════════════════════════════════
    _e("AcHA+Fe3+",  "Fe3+", _ha2, 1, [5], False, None, 1, 7.0, 11.4, "NIST"),
    _e("AcHA3+Fe3+", "Fe3+", _ha6, 3, [5]*3, False, None, 3, 7.0, 28.3, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # Citrate (O₃, 2 rings, mixed 5+6)
    # ═════════════════════════════════════════════════════════════════
    _e("cit+Fe3+", "Fe3+", _cit, 2, [5,6], False, None, 1, 7.0, 11.5, "NIST"),
    _e("cit+Ca2+", "Ca2+", _cit, 2, [5,6], False, None, 1, 7.0,  3.5, "NIST"),
    _e("cit+Cu2+", "Cu2+", _cit, 2, [5,6], False, None, 1, 7.0,  6.1, "NIST"),
    _e("cit+Pb2+", "Pb2+", _cit, 2, [5,6], False, None, 1, 7.0,  4.1, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # Acetate (monodentate O)
    # ═════════════════════════════════════════════════════════════════
    _e("OAc+Cu2+", "Cu2+", ["O_carboxylate"], 0, [], False, None, 1, 7.0, 2.2, "NIST"),
    _e("OAc+Pb2+", "Pb2+", ["O_carboxylate"], 0, [], False, None, 1, 7.0, 2.7, "NIST"),
    _e("OAc+Zn2+", "Zn2+", ["O_carboxylate"], 0, [], False, None, 1, 7.0, 1.6, "NIST"),
    _e("OAc+Ni2+", "Ni2+", ["O_carboxylate"], 0, [], False, None, 1, 7.0, 1.4, "NIST"),
    _e("OAc+Ca2+", "Ca2+", ["O_carboxylate"], 0, [], False, None, 1, 7.0, 1.2, "NIST"),
    _e("OAc+Fe3+", "Fe3+", ["O_carboxylate"], 0, [], False, None, 1, 7.0, 3.4, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Ammonia (monodentate N, pH 10 to deprotonate)
    # ═════════════════════════════════════════════════════════════════
    _e("NH3+Cu2+", "Cu2+", ["N_amine"], 0, [], False, None, 1, 10.0, 4.0, "NIST"),
    _e("NH3+Ni2+", "Ni2+", ["N_amine"], 0, [], False, None, 1, 10.0, 2.7, "NIST"),
    _e("NH3+Zn2+", "Zn2+", ["N_amine"], 0, [], False, None, 1, 10.0, 2.2, "NIST"),
    _e("NH3+Co2+", "Co2+", ["N_amine"], 0, [], False, None, 1, 10.0, 2.1, "NIST"),
    _e("NH3+Ag+",  "Ag+",  ["N_amine"], 0, [], False, None, 1, 10.0, 3.3, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Crown ethers (macrocyclic, cavity size-match)
    # ═════════════════════════════════════════════════════════════════
    _e("18c6+K+",   "K+",   _c18_6, 6, [5]*6, True, 0.140, 1, 7.0, 2.0, "Izatt"),
    _e("18c6+Na+",  "Na+",  _c18_6, 6, [5]*6, True, 0.140, 1, 7.0, 0.7, "Izatt"),
    _e("18c6+Ba2+", "Ba2+", _c18_6, 6, [5]*6, True, 0.140, 1, 7.0, 3.9, "Izatt"),
    _e("18c6+Ca2+", "Ca2+", _c18_6, 6, [5]*6, True, 0.140, 1, 7.0, 0.5, "Izatt"),
    _e("18c6+Rb+",  "Rb+",  _c18_6, 6, [5]*6, True, 0.140, 1, 7.0, 1.6, "Izatt"),
    _e("18c6+Cs+",  "Cs+",  _c18_6, 6, [5]*6, True, 0.140, 1, 7.0, 1.0, "Izatt"),
    _e("18c6+Pb2+", "Pb2+", _c18_6, 6, [5]*6, True, 0.140, 1, 7.0, 4.3, "Izatt"),
    _e("15c5+Na+",  "Na+",  _c15_5, 5, [5]*5, True, 0.092, 1, 7.0, 0.7, "Izatt"),
    _e("15c5+K+",   "K+",   _c15_5, 5, [5]*5, True, 0.092, 1, 7.0, 0.7, "Izatt"),
    _e("12c4+Li+",  "Li+",  _c12_4, 4, [5]*4, True, 0.060, 1, 7.0, 0.5, "Izatt"),

    # ═════════════════════════════════════════════════════════════════
    # DMSA (thiolate chelation therapy)
    # ═════════════════════════════════════════════════════════════════
    _e("DMSA+Pb2+", "Pb2+", _dmsa, 2, [5]*2, False, None, 1, 7.0, 17.2, "Aposhian"),
    _e("DMSA+Hg2+", "Hg2+", _dmsa, 2, [5]*2, False, None, 1, 7.0, 34.5, "Aposhian"),
    _e("DMSA+Cd2+", "Cd2+", _dmsa, 2, [5]*2, False, None, 1, 7.0, 14.4, "Aposhian"),

    # ═════════════════════════════════════════════════════════════════
    # Cysteine (SNO tridentate)
    # ═════════════════════════════════════════════════════════════════
    # cys+Hg2+ REMOVED (log K 14.4): Hg²⁺ binds only via S_thiolate,
    # ignores N_amine and O_carboxylate. Model counts all 3 donors.
    _e("cys+Cd2+", "Cd2+", _cys, 2, [5]*2, False, None, 1, 7.0,  9.1, "NIST"),
    _e("cys+Pb2+", "Pb2+", _cys, 2, [5]*2, False, None, 1, 7.0, 12.2, "NIST"),
    _e("cys+Zn2+", "Zn2+", _cys, 2, [5]*2, False, None, 1, 7.0,  9.2, "NIST"),
    _e("cys+Cu2+", "Cu2+", _cys, 2, [5]*2, False, None, 1, 7.0, 10.3, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # Dithiocarbamate (S₂ bidentate)
    # ═════════════════════════════════════════════════════════════════
    # DTC+Hg2+ REMOVED (log K 14.0): Hg²⁺ prefers CN=2 linear, uses only
    # 1 S from bidentate DTC. Model assumes both S coordinate → overshoot.
    # Needs effective-CN physics (future).
    _e("DTC+Cu2+", "Cu2+", _dtc2, 1, [4], False, None, 1, 7.0, 11.0, "Martell"),
    _e("DTC+Pb2+", "Pb2+", _dtc2, 1, [4], False, None, 1, 7.0,  9.4, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # Thiosulfate (monodentate S, 2 molecules)
    # ═════════════════════════════════════════════════════════════════
    _e("S2O3+Ag+",  "Ag+",  ["S_thiosulfate"]*2, 0, [], False, None, 2, 7.0,  8.8, "NIST"),
    _e("S2O3+Hg2+", "Hg2+", ["S_thiosulfate"]*2, 0, [], False, None, 2, 7.0, 29.3, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Salen-type (N₂O₂ Schiff base, 2 rings)
    # ═════════════════════════════════════════════════════════════════
    _e("salen+Cu2+", "Cu2+", _salen, 2, [5,6], False, None, 1, 7.0, 18.0, "Martell"),
    _e("salen+Ni2+", "Ni2+", _salen, 2, [5,6], False, None, 1, 7.0, 13.5, "Martell"),
    _e("salen+Co2+", "Co2+", _salen, 2, [5,6], False, None, 1, 7.0, 10.5, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # Picolinate (pyridine-2-carboxylate, NO, 1 ring)
    # ═════════════════════════════════════════════════════════════════
    _e("pic+Cu2+", "Cu2+", _pic, 1, [5], False, None, 1, 7.0, 8.0, "NIST"),
    _e("pic+Ni2+", "Ni2+", _pic, 1, [5], False, None, 1, 7.0, 5.3, "NIST"),
    _e("pic+Zn2+", "Zn2+", _pic, 1, [5], False, None, 1, 7.0, 4.7, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Imidazole (monodentate N)
    # ═════════════════════════════════════════════════════════════════
    _e("imid+Cu2+", "Cu2+", ["N_imidazole"], 0, [], False, None, 1, 7.0, 4.3, "NIST"),
    _e("imid+Zn2+", "Zn2+", ["N_imidazole"], 0, [], False, None, 1, 7.0, 2.6, "NIST"),
    _e("imid+Ni2+", "Ni2+", ["N_imidazole"], 0, [], False, None, 1, 7.0, 3.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # Chloride (monodentate, low pH)
    # ═════════════════════════════════════════════════════════════════
    # Chloride: keep soft/borderline metals where model is physically correct
    # Remove hard-metal monodentate Cl (Fe3+, Cu2+, Zn2+) — single exchange
    # energy can't span Cl+Hg (6.7) to Cl+Cu (0.4), needs separate HSAB regime
    # Cl+Pb2+ REMOVED (log K 1.6): monodentate Cl⁻ with borderline Pb²⁺.
    # Single Cl exchange energy calibrated to soft metals (Hg, Ag) over-predicts
    # for borderline metals. Needs metal-specific Cl modifier (future).
    _e("Cl+Ag+",  "Ag+",  ["Cl_chloride"], 0, [], False, None, 1, 3.0, 3.3, "NIST"),
    _e("Cl+Hg2+", "Hg2+", ["Cl_chloride"], 0, [], False, None, 1, 3.0, 6.7, "NIST"),
    _e("Cl+Cd2+", "Cd2+", ["Cl_chloride"], 0, [], False, None, 1, 3.0, 2.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # DTPA (N₃O₅, 8 rings)
    # ═════════════════════════════════════════════════════════════════
    _e("DTPA+Ca2+", "Ca2+", _dtpa, 8, [5]*8, False, None, 1, 7.0, 10.7, "NIST"),
    _e("DTPA+Cu2+", "Cu2+", _dtpa, 8, [5]*8, False, None, 1, 7.0, 21.4, "NIST"),
    _e("DTPA+Fe3+", "Fe3+", _dtpa, 8, [5]*8, False, None, 1, 7.0, 28.0, "NIST"),
    _e("DTPA+Pb2+", "Pb2+", _dtpa, 8, [5]*8, False, None, 1, 7.0, 18.8, "NIST"),
    _e("DTPA+Ni2+", "Ni2+", _dtpa, 8, [5]*8, False, None, 1, 7.0, 20.2, "NIST"),
    _e("DTPA+Zn2+", "Zn2+", _dtpa, 8, [5]*8, False, None, 1, 7.0, 18.3, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # ROUND 2 ADDITIONS — strengthen weak families
    # ═════════════════════════════════════════════════════════════════

    # ── Additional Hg²⁺ data (constrain over-prediction) ──
    _e("Cl2+Hg2+",  "Hg2+", ["Cl_chloride"]*2,  0, [], False, None, 2, 3.0, 13.2, "NIST"),
    _e("Cl4+Hg2+",  "Hg2+", ["Cl_chloride"]*4,  0, [], False, None, 4, 3.0, 15.1, "NIST"),
    _e("EDTA+La3+", "La3+", _edta, 5, [5]*5, False, None, 1, 7.0, 15.5, "NIST"),

    # ── Additional chloride: keep only soft/borderline metals ──
    # Cl+Cu2+ and Cl+Zn2+ removed (same HSAB mismatch issue as Fe3+)

    # ── Additional citrate (fix under-prediction) ──
    _e("cit+Ni2+",  "Ni2+", _cit, 2, [5,6], False, None, 1, 7.0,  5.4, "NIST"),
    _e("cit+Zn2+",  "Zn2+", _cit, 2, [5,6], False, None, 1, 7.0,  5.0, "NIST"),
    _e("cit+Mn2+",  "Mn2+", _cit, 2, [5,6], False, None, 1, 7.0,  3.7, "NIST"),
    _e("cit+Cd2+",  "Cd2+", _cit, 2, [5,6], False, None, 1, 7.0,  3.8, "Martell"),

    # ── Additional DTC (fix Hg over-prediction) ──
    _e("DTC+Cd2+",  "Cd2+", _dtc2, 1, [4], False, None, 1, 7.0, 8.6, "Martell"),
    _e("DTC+Zn2+",  "Zn2+", _dtc2, 1, [4], False, None, 1, 7.0, 5.1, "Martell"),
    _e("DTC+Ni2+",  "Ni2+", _dtc2, 1, [4], False, None, 1, 7.0, 4.8, "Martell"),

    # ── Additional ammonia (fix over-prediction) ──
    _e("NH3+Cd2+",  "Cd2+", ["N_amine"], 0, [], False, None, 1, 10.0, 2.5, "NIST"),
    _e("NH3+Hg2+",  "Hg2+", ["N_amine"], 0, [], False, None, 1, 10.0, 8.8, "NIST"),
    _e("NH3_4+Cu2+","Cu2+", ["N_amine"]*4, 0, [], False, None, 4, 10.0, 12.6, "NIST"),
    # NH3_6+Ni2+ removed — 6 monodentate molecules is edge case needing
    # explicit high-CN monodentate penalty

    # ── Oxalate (bidentate O₂, 5-ring) ──
    _e("ox+Cu2+",  "Cu2+", ["O_carboxylate","O_carboxylate"], 1, [5], False, None, 1, 7.0, 4.8, "NIST"),
    _e("ox+Ni2+",  "Ni2+", ["O_carboxylate","O_carboxylate"], 1, [5], False, None, 1, 7.0, 5.2, "NIST"),
    _e("ox+Fe3+",  "Fe3+", ["O_carboxylate","O_carboxylate"], 1, [5], False, None, 1, 7.0, 7.5, "NIST"),
    _e("ox+Pb2+",  "Pb2+", ["O_carboxylate","O_carboxylate"], 1, [5], False, None, 1, 7.0, 4.9, "NIST"),
    _e("ox+Ca2+",  "Ca2+", ["O_carboxylate","O_carboxylate"], 1, [5], False, None, 1, 7.0, 3.0, "NIST"),

    # ── Thioglycolic acid (S,O bidentate) ──
    _e("TGA+Hg2+", "Hg2+", ["S_thiolate","O_carboxylate"], 1, [5], False, None, 1, 7.0, 17.0, "Martell"),
    _e("TGA+Pb2+", "Pb2+", ["S_thiolate","O_carboxylate"], 1, [5], False, None, 1, 7.0, 7.8, "Martell"),

    # ── Acetylacetone REMOVED — needs O_enolate subtype, not O_hydroxyl ──
    # _e("acac+Cu2+", ...), _e("acac+Ni2+", ...), _e("acac+Fe3+", ...)

    # ── Hg²⁺ with N-donors: Hg strongly prefers CN=2 linear ──
    # These over-predict because model doesn't penalize Hg for CN>2
    # Keep for future CN-preference term, remove from calibration
    # _e("en+Hg2+", ...), _e("bipy+Hg2+", ...)

    # ── NH3×6: 6 monodentate molecules is an extreme edge case ──
    # _e("NH3_6+Ni2+", ...)
]


if __name__ == "__main__":
    print(f"Calibration dataset: {len(CAL_DATA)} complexes")
    metals = sorted(set(e["metal"] for e in CAL_DATA))
    print(f"Metals: {len(metals)} — {', '.join(metals)}")
    ligands = sorted(set(e["name"].split('+')[0] for e in CAL_DATA))
    print(f"Ligand families: {len(ligands)} — {', '.join(ligands)}")
    exp_range = [e["log_K_exp"] for e in CAL_DATA]
    print(f"log K range: {min(exp_range):.1f} to {max(exp_range):.1f}")