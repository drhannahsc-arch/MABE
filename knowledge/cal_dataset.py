"""
cal_dataset.py — Calibration dataset for MABE frozen scorer.

500+ metal-ligand complexes from NIST SRD 46, Martell & Smith Critical
Stability Constants, and supplementary literature sources.

All log K values are thermodynamic formation constants (fully deprotonated
ligand convention) at 25°C, I = 0.1 M unless noted.

v3: Expanded from 147 to 500+ complexes.
"""


def _e(name, metal, donors, chelate_rings, ring_sizes,
       macrocyclic, cavity_nm, n_lig_mol, pH, log_K_exp, source):
    return {
        "name": name, "metal": metal, "donors": donors,
        "chelate_rings": chelate_rings, "ring_sizes": ring_sizes,
        "macrocyclic": macrocyclic, "cavity_nm": cavity_nm,
        "n_lig_mol": n_lig_mol, "pH": pH,
        "log_K_exp": log_K_exp, "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Donor set shorthands
# ═══════════════════════════════════════════════════════════════════════════
_edta = ["N_amine","N_amine","O_carboxylate","O_carboxylate",
         "O_carboxylate","O_carboxylate"]
_dtpa = ["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate",
         "O_carboxylate","O_carboxylate","O_carboxylate"]
_nta = ["N_amine","O_carboxylate","O_carboxylate","O_carboxylate"]
_ida = ["N_amine","O_carboxylate","O_carboxylate"]
_hedta = ["N_amine","N_amine","O_carboxylate","O_carboxylate",
          "O_carboxylate","O_hydroxyl"]
_egta = ["N_amine","N_amine","O_carboxylate","O_carboxylate",
         "O_carboxylate","O_carboxylate"]
_en2 = ["N_amine","N_amine"]
_en4 = ["N_amine"]*4
_en6 = ["N_amine"]*6
_dien = ["N_amine","N_amine","N_amine"]
_trien = ["N_amine"]*4
_gly = ["N_amine","O_carboxylate"]
_gly2 = ["N_amine","N_amine","O_carboxylate","O_carboxylate"]
_bipy2 = ["N_pyridine","N_pyridine"]
_bipy6 = ["N_pyridine"]*6
_phen2 = ["N_pyridine","N_pyridine"]
_phen6 = ["N_pyridine"]*6
_terpy = ["N_pyridine"]*3
_dpa = ["N_pyridine","O_carboxylate","O_carboxylate"]
_pic = ["N_pyridine","O_carboxylate"]
_8hq = ["N_pyridine","O_phenolate"]
_sal = ["O_carboxylate","O_phenolate"]
_salen = ["N_imine","N_imine","O_phenolate","O_phenolate"]
_cat2 = ["O_catecholate","O_catecholate"]
_cat6 = ["O_catecholate"]*6
_ox = ["O_carboxylate","O_carboxylate"]
_ox4 = ["O_carboxylate"]*4
_ox6 = ["O_carboxylate"]*6
_mal = ["O_carboxylate","O_carboxylate"]
_succ = ["O_carboxylate","O_carboxylate"]
_tart = ["O_carboxylate","O_carboxylate","O_hydroxyl","O_hydroxyl"]
_cit = ["O_carboxylate","O_carboxylate","O_hydroxyl"]
_oac = ["O_carboxylate"]
_aha = ["O_hydroxamate","O_hydroxamate"]
_aha6 = ["O_hydroxamate"]*6
_cys = ["S_thiolate","N_amine","O_carboxylate"]
_pen = ["S_thiolate","N_amine","O_carboxylate"]
_dmsa = ["S_thiolate","S_thiolate","O_carboxylate","O_carboxylate"]
_tga = ["S_thiolate","O_carboxylate"]
_dtc2 = ["S_dithiocarbamate","S_dithiocarbamate"]
_dmg = ["N_imine","N_imine","O_hydroxyl","O_hydroxyl"]
_cyclam = ["N_amine"]*4
_cyclen = ["N_amine"]*4
_dota = ["N_amine"]*4 + ["O_carboxylate"]*4
_nota = ["N_amine"]*3 + ["O_carboxylate"]*3
_18c6 = ["O_ether"]*6
_15c5 = ["O_ether"]*5
_12c4 = ["O_ether"]*4


CAL_DATA = [

    # ═════════════════════════════════════════════════════════════════
    # EDTA (N₂O₄, hexadentate, 5 chelate rings)
    # ═════════════════════════════════════════════════════════════════
    _e("EDTA+Ca2+", "Ca2+", _edta, 5, [5]*5, False, None, 1, 14.0, 10.7, "NIST"),
    _e("EDTA+Mg2+", "Mg2+", _edta, 5, [5]*5, False, None, 1, 14.0,  8.7, "NIST"),
    _e("EDTA+Ba2+", "Ba2+", _edta, 5, [5]*5, False, None, 1, 14.0,  7.8, "NIST"),
    _e("EDTA+Sr2+", "Sr2+", _edta, 5, [5]*5, False, None, 1, 14.0,  8.6, "NIST"),
    _e("EDTA+Mn2+", "Mn2+", _edta, 5, [5]*5, False, None, 1, 14.0, 13.8, "NIST"),
    _e("EDTA+Fe2+", "Fe2+", _edta, 5, [5]*5, False, None, 1, 14.0, 14.3, "NIST"),
    _e("EDTA+Co2+", "Co2+", _edta, 5, [5]*5, False, None, 1, 14.0, 16.3, "NIST"),
    _e("EDTA+Ni2+", "Ni2+", _edta, 5, [5]*5, False, None, 1, 14.0, 18.6, "NIST"),
    _e("EDTA+Cu2+", "Cu2+", _edta, 5, [5]*5, False, None, 1, 14.0, 18.8, "NIST"),
    _e("EDTA+Zn2+", "Zn2+", _edta, 5, [5]*5, False, None, 1, 14.0, 16.5, "NIST"),
    _e("EDTA+Cd2+", "Cd2+", _edta, 5, [5]*5, False, None, 1, 14.0, 16.5, "NIST"),
    _e("EDTA+Pb2+", "Pb2+", _edta, 5, [5]*5, False, None, 1, 14.0, 18.0, "NIST"),
    _e("EDTA+Hg2+", "Hg2+", _edta, 5, [5]*5, False, None, 1, 14.0, 21.8, "NIST"),
    _e("EDTA+Fe3+", "Fe3+", _edta, 5, [5]*5, False, None, 1, 14.0, 25.1, "NIST"),
    _e("EDTA+Al3+", "Al3+", _edta, 5, [5]*5, False, None, 1, 14.0, 16.1, "NIST"),
    _e("EDTA+Cr3+", "Cr3+", _edta, 5, [5]*5, False, None, 1, 14.0, 23.4, "NIST"),
    _e("EDTA+La3+", "La3+", _edta, 5, [5]*5, False, None, 1, 14.0, 15.5, "NIST"),
    _e("EDTA+Ce3+", "Ce3+", _edta, 5, [5]*5, False, None, 1, 14.0, 15.9, "NIST"),
    _e("EDTA+Nd3+", "Nd3+", _edta, 5, [5]*5, False, None, 1, 14.0, 16.6, "NIST"),
    _e("EDTA+Sm3+", "Sm3+", _edta, 5, [5]*5, False, None, 1, 14.0, 17.1, "NIST"),
    _e("EDTA+Eu3+", "Eu3+", _edta, 5, [5]*5, False, None, 1, 14.0, 17.3, "NIST"),
    _e("EDTA+Gd3+", "Gd3+", _edta, 5, [5]*5, False, None, 1, 14.0, 17.4, "NIST"),
    _e("EDTA+Tb3+", "Tb3+", _edta, 5, [5]*5, False, None, 1, 14.0, 17.9, "NIST"),
    _e("EDTA+Dy3+", "Dy3+", _edta, 5, [5]*5, False, None, 1, 14.0, 18.3, "NIST"),
    _e("EDTA+Er3+", "Er3+", _edta, 5, [5]*5, False, None, 1, 14.0, 18.8, "NIST"),
    _e("EDTA+Yb3+", "Yb3+", _edta, 5, [5]*5, False, None, 1, 14.0, 19.5, "NIST"),
    _e("EDTA+Lu3+", "Lu3+", _edta, 5, [5]*5, False, None, 1, 14.0, 19.8, "NIST"),
    _e("EDTA+VO2+", "VO2+", _edta, 5, [5]*5, False, None, 1, 14.0, 18.8, "NIST"),
    _e("EDTA+In3+", "In3+", _edta, 5, [5]*5, False, None, 1, 14.0, 24.9, "NIST"),
    _e("EDTA+Ga3+", "Ga3+", _edta, 5, [5]*5, False, None, 1, 14.0, 20.3, "NIST"),
    _e("EDTA+Bi3+", "Bi3+", _edta, 5, [5]*5, False, None, 1, 14.0, 22.8, "NIST"),
    _e("EDTA+Th4+", "Th4+", _edta, 5, [5]*5, False, None, 1, 14.0, 23.2, "NIST"),
    _e("EDTA+Sn2+", "Sn2+", _edta, 5, [5]*5, False, None, 1, 14.0, 18.3, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # DTPA (N₃O₅, octadentate, 8 rings)
    # ═════════════════════════════════════════════════════════════════
    _e("DTPA+Ca2+", "Ca2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 10.7, "NIST"),
    _e("DTPA+Cu2+", "Cu2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 21.4, "NIST"),
    _e("DTPA+Fe3+", "Fe3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 28.0, "NIST"),
    _e("DTPA+Pb2+", "Pb2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 18.8, "NIST"),
    _e("DTPA+Ni2+", "Ni2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 20.2, "NIST"),
    _e("DTPA+Zn2+", "Zn2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 18.3, "NIST"),
    _e("DTPA+Gd3+", "Gd3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 22.5, "NIST"),
    _e("DTPA+La3+", "La3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 19.5, "NIST"),
    _e("DTPA+Hg2+", "Hg2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 26.3, "NIST"),
    _e("DTPA+Cd2+", "Cd2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 19.3, "NIST"),
    _e("DTPA+Mn2+", "Mn2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 15.6, "NIST"),
    _e("DTPA+Co2+", "Co2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 18.6, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # HEDTA (N₂O₃(OH), hexadentate, 5 rings)
    # ═════════════════════════════════════════════════════════════════
    _e("HEDTA+Cu2+", "Cu2+", _hedta, 5, [5]*5, False, None, 1, 14.0, 17.4, "NIST"),
    _e("HEDTA+Ni2+", "Ni2+", _hedta, 5, [5]*5, False, None, 1, 14.0, 17.0, "NIST"),
    _e("HEDTA+Fe3+", "Fe3+", _hedta, 5, [5]*5, False, None, 1, 14.0, 19.8, "NIST"),
    _e("HEDTA+Ca2+", "Ca2+", _hedta, 5, [5]*5, False, None, 1, 14.0,  8.1, "NIST"),
    _e("HEDTA+Pb2+", "Pb2+", _hedta, 5, [5]*5, False, None, 1, 14.0, 15.5, "NIST"),
    _e("HEDTA+Zn2+", "Zn2+", _hedta, 5, [5]*5, False, None, 1, 14.0, 14.5, "NIST"),
    _e("HEDTA+Cd2+", "Cd2+", _hedta, 5, [5]*5, False, None, 1, 14.0, 13.1, "NIST"),
    _e("HEDTA+Co2+", "Co2+", _hedta, 5, [5]*5, False, None, 1, 14.0, 14.5, "NIST"),
    _e("HEDTA+Mn2+", "Mn2+", _hedta, 5, [5]*5, False, None, 1, 14.0, 10.6, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # EGTA (Ca-selective, larger backbone)
    # ═════════════════════════════════════════════════════════════════
    _e("EGTA+Ca2+", "Ca2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 10.9, "NIST"),
    _e("EGTA+Mg2+", "Mg2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0,  5.2, "NIST"),
    _e("EGTA+Cu2+", "Cu2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 17.7, "NIST"),
    _e("EGTA+Zn2+", "Zn2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 12.6, "NIST"),
    _e("EGTA+Pb2+", "Pb2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 14.7, "NIST"),
    _e("EGTA+Ni2+", "Ni2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 13.6, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # NTA (N₁O₃, tetradentate)
    # ═════════════════════════════════════════════════════════════════
    _e("NTA+Cu2+", "Cu2+", _nta, 3, [5]*3, False, None, 1, 14.0, 12.9, "NIST"),
    _e("NTA+Ni2+", "Ni2+", _nta, 3, [5]*3, False, None, 1, 14.0, 11.5, "NIST"),
    _e("NTA+Zn2+", "Zn2+", _nta, 3, [5]*3, False, None, 1, 14.0, 10.7, "NIST"),
    _e("NTA+Co2+", "Co2+", _nta, 3, [5]*3, False, None, 1, 14.0, 10.4, "NIST"),
    _e("NTA+Ca2+", "Ca2+", _nta, 3, [5]*3, False, None, 1, 14.0,  6.4, "NIST"),
    _e("NTA+Fe3+", "Fe3+", _nta, 3, [5]*3, False, None, 1, 14.0, 15.9, "NIST"),
    _e("NTA+Pb2+", "Pb2+", _nta, 3, [5]*3, False, None, 1, 14.0, 11.4, "NIST"),
    _e("NTA+Cd2+", "Cd2+", _nta, 3, [5]*3, False, None, 1, 14.0,  9.8, "NIST"),
    _e("NTA+Mn2+", "Mn2+", _nta, 3, [5]*3, False, None, 1, 14.0,  7.4, "NIST"),
    _e("NTA+Fe2+", "Fe2+", _nta, 3, [5]*3, False, None, 1, 14.0,  8.8, "NIST"),
    _e("NTA+La3+", "La3+", _nta, 3, [5]*3, False, None, 1, 14.0, 10.4, "NIST"),
    _e("NTA+Al3+", "Al3+", _nta, 3, [5]*3, False, None, 1, 14.0,  9.5, "NIST"),
    _e("NTA+Cr3+", "Cr3+", _nta, 3, [5]*3, False, None, 1, 14.0, 10.2, "NIST"),
    _e("NTA+Hg2+", "Hg2+", _nta, 3, [5]*3, False, None, 1, 14.0, 14.6, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # IDA (N₁O₂, tridentate)
    # ═════════════════════════════════════════════════════════════════
    _e("IDA+Cu2+", "Cu2+", _ida, 2, [5]*2, False, None, 1, 14.0, 10.6, "NIST"),
    _e("IDA+Ni2+", "Ni2+", _ida, 2, [5]*2, False, None, 1, 14.0,  8.2, "NIST"),
    _e("IDA+Zn2+", "Zn2+", _ida, 2, [5]*2, False, None, 1, 14.0,  7.0, "NIST"),
    _e("IDA+Co2+", "Co2+", _ida, 2, [5]*2, False, None, 1, 14.0,  7.0, "NIST"),
    _e("IDA+Fe3+", "Fe3+", _ida, 2, [5]*2, False, None, 1, 14.0, 10.7, "NIST"),
    _e("IDA+Mn2+", "Mn2+", _ida, 2, [5]*2, False, None, 1, 14.0,  4.7, "NIST"),
    _e("IDA+Pb2+", "Pb2+", _ida, 2, [5]*2, False, None, 1, 14.0,  7.3, "NIST"),
    _e("IDA+Cd2+", "Cd2+", _ida, 2, [5]*2, False, None, 1, 14.0,  5.7, "NIST"),
    _e("IDA+Ca2+", "Ca2+", _ida, 2, [5]*2, False, None, 1, 14.0,  2.6, "NIST"),
    _e("IDA+Fe2+", "Fe2+", _ida, 2, [5]*2, False, None, 1, 14.0,  5.8, "NIST"),
    _e("IDA+La3+", "La3+", _ida, 2, [5]*2, False, None, 1, 14.0,  5.1, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # POLYAMINES
    # ═════════════════════════════════════════════════════════════════
    # Ethylenediamine (en, N₂)
    _e("en+Cu2+", "Cu2+", _en2, 1, [5], False, None, 1, 14.0, 10.6, "NIST"),
    _e("en+Ni2+", "Ni2+", _en2, 1, [5], False, None, 1, 14.0,  7.5, "NIST"),
    _e("en+Co2+", "Co2+", _en2, 1, [5], False, None, 1, 14.0,  5.9, "NIST"),
    _e("en+Zn2+", "Zn2+", _en2, 1, [5], False, None, 1, 14.0,  5.7, "NIST"),
    _e("en+Fe2+", "Fe2+", _en2, 1, [5], False, None, 1, 14.0,  4.3, "NIST"),
    _e("en+Mn2+", "Mn2+", _en2, 1, [5], False, None, 1, 14.0,  2.7, "NIST"),
    _e("en+Cd2+", "Cd2+", _en2, 1, [5], False, None, 1, 14.0,  5.5, "NIST"),
    _e("en+Pb2+", "Pb2+", _en2, 1, [5], False, None, 1, 14.0,  5.0, "NIST"),
    # bis-en cumulative
    _e("en2+Cu2+", "Cu2+", _en4, 2, [5]*2, False, None, 2, 14.0, 19.6, "NIST"),
    _e("en2+Ni2+", "Ni2+", _en4, 2, [5]*2, False, None, 2, 14.0, 14.0, "NIST"),
    _e("en2+Co2+", "Co2+", _en4, 2, [5]*2, False, None, 2, 14.0, 10.7, "NIST"),
    _e("en2+Zn2+", "Zn2+", _en4, 2, [5]*2, False, None, 2, 14.0, 10.6, "NIST"),
    _e("en2+Cd2+", "Cd2+", _en4, 2, [5]*2, False, None, 2, 14.0, 10.1, "NIST"),
    _e("en2+Fe2+", "Fe2+", _en4, 2, [5]*2, False, None, 2, 14.0,  7.6, "NIST"),
    _e("en2+Mn2+", "Mn2+", _en4, 2, [5]*2, False, None, 2, 14.0,  4.1, "NIST"),
    # tris-en cumulative
    _e("en3+Ni2+", "Ni2+", _en6, 3, [5]*3, False, None, 3, 14.0, 18.3, "NIST"),
    _e("en3+Co2+", "Co2+", _en6, 3, [5]*3, False, None, 3, 14.0, 13.8, "NIST"),
    _e("en3+Cu2+", "Cu2+", _en6, 3, [5]*3, False, None, 3, 14.0, 20.0, "Martell"),
    _e("en3+Zn2+", "Zn2+", _en6, 3, [5]*3, False, None, 3, 14.0, 12.1, "NIST"),
    _e("en3+Cd2+", "Cd2+", _en6, 3, [5]*3, False, None, 3, 14.0, 12.0, "NIST"),
    _e("en3+Fe2+", "Fe2+", _en6, 3, [5]*3, False, None, 3, 14.0,  9.5, "NIST"),
    _e("en3+Mn2+", "Mn2+", _en6, 3, [5]*3, False, None, 3, 14.0,  5.7, "NIST"),
    # Diethylenetriamine (dien, N₃)
    _e("dien+Cu2+", "Cu2+", _dien, 2, [5]*2, False, None, 1, 14.0, 16.0, "NIST"),
    _e("dien+Ni2+", "Ni2+", _dien, 2, [5]*2, False, None, 1, 14.0, 10.7, "NIST"),
    _e("dien+Co2+", "Co2+", _dien, 2, [5]*2, False, None, 1, 14.0,  8.1, "NIST"),
    _e("dien+Zn2+", "Zn2+", _dien, 2, [5]*2, False, None, 1, 14.0,  8.9, "NIST"),
    _e("dien+Cd2+", "Cd2+", _dien, 2, [5]*2, False, None, 1, 14.0,  8.4, "NIST"),
    _e("dien+Mn2+", "Mn2+", _dien, 2, [5]*2, False, None, 1, 14.0,  4.9, "NIST"),
    _e("dien+Pb2+", "Pb2+", _dien, 2, [5]*2, False, None, 1, 14.0,  8.0, "Martell"),
    _e("dien+Fe2+", "Fe2+", _dien, 2, [5]*2, False, None, 1, 14.0,  7.2, "NIST"),
    # Triethylenetetramine (trien, N₄)
    _e("trien+Cu2+", "Cu2+", _trien, 3, [5]*3, False, None, 1, 14.0, 20.1, "NIST"),
    _e("trien+Ni2+", "Ni2+", _trien, 3, [5]*3, False, None, 1, 14.0, 14.0, "NIST"),
    _e("trien+Co2+", "Co2+", _trien, 3, [5]*3, False, None, 1, 14.0, 11.0, "NIST"),
    _e("trien+Zn2+", "Zn2+", _trien, 3, [5]*3, False, None, 1, 14.0, 12.0, "NIST"),
    _e("trien+Cd2+", "Cd2+", _trien, 3, [5]*3, False, None, 1, 14.0, 10.8, "NIST"),
    _e("trien+Mn2+", "Mn2+", _trien, 3, [5]*3, False, None, 1, 14.0,  4.9, "NIST"),
    _e("trien+Fe2+", "Fe2+", _trien, 3, [5]*3, False, None, 1, 14.0,  7.8, "NIST"),
    _e("trien+Pb2+", "Pb2+", _trien, 3, [5]*3, False, None, 1, 14.0, 10.3, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # GLYCINE family
    # ═════════════════════════════════════════════════════════════════
    _e("gly+Cu2+", "Cu2+", _gly, 1, [5], False, None, 1, 14.0, 8.2, "NIST"),
    _e("gly+Ni2+", "Ni2+", _gly, 1, [5], False, None, 1, 14.0, 5.8, "NIST"),
    _e("gly+Zn2+", "Zn2+", _gly, 1, [5], False, None, 1, 14.0, 5.0, "NIST"),
    _e("gly+Co2+", "Co2+", _gly, 1, [5], False, None, 1, 14.0, 5.2, "NIST"),
    _e("gly+Mn2+", "Mn2+", _gly, 1, [5], False, None, 1, 14.0, 2.9, "NIST"),
    _e("gly+Fe2+", "Fe2+", _gly, 1, [5], False, None, 1, 14.0, 4.3, "NIST"),
    _e("gly+Cd2+", "Cd2+", _gly, 1, [5], False, None, 1, 14.0, 4.3, "NIST"),
    _e("gly+Pb2+", "Pb2+", _gly, 1, [5], False, None, 1, 14.0, 5.5, "NIST"),
    _e("gly+Fe3+", "Fe3+", _gly, 1, [5], False, None, 1, 14.0, 10.0, "NIST"),
    _e("gly+Ca2+", "Ca2+", _gly, 1, [5], False, None, 1, 14.0, 1.4, "NIST"),
    # bis-glycine cumulative
    _e("gly2+Cu2+", "Cu2+", _gly2, 2, [5]*2, False, None, 2, 14.0, 15.0, "NIST"),
    _e("gly2+Ni2+", "Ni2+", _gly2, 2, [5]*2, False, None, 2, 14.0, 11.0, "NIST"),
    _e("gly2+Co2+", "Co2+", _gly2, 2, [5]*2, False, None, 2, 14.0,  9.5, "NIST"),
    _e("gly2+Zn2+", "Zn2+", _gly2, 2, [5]*2, False, None, 2, 14.0,  9.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # PYRIDINE DONORS
    # ═════════════════════════════════════════════════════════════════
    # Bipyridine (bipy, N₂)
    _e("bipy+Fe2+", "Fe2+", _bipy2, 1, [5], False, None, 1, 14.0, 5.9, "NIST"),
    _e("bipy+Cu2+", "Cu2+", _bipy2, 1, [5], False, None, 1, 14.0, 9.0, "NIST"),
    _e("bipy+Ni2+", "Ni2+", _bipy2, 1, [5], False, None, 1, 14.0, 7.1, "NIST"),
    _e("bipy+Zn2+", "Zn2+", _bipy2, 1, [5], False, None, 1, 14.0, 5.2, "NIST"),
    _e("bipy+Co2+", "Co2+", _bipy2, 1, [5], False, None, 1, 14.0, 5.7, "NIST"),
    _e("bipy+Mn2+", "Mn2+", _bipy2, 1, [5], False, None, 1, 14.0, 2.5, "NIST"),
    _e("bipy+Cd2+", "Cd2+", _bipy2, 1, [5], False, None, 1, 14.0, 4.2, "NIST"),
    # tris-bipy cumulative
    _e("bipy3+Fe2+", "Fe2+", _bipy6, 3, [5]*3, False, None, 3, 14.0, 17.2, "NIST"),
    _e("bipy3+Ni2+", "Ni2+", _bipy6, 3, [5]*3, False, None, 3, 14.0, 20.2, "NIST"),
    _e("bipy3+Co2+", "Co2+", _bipy6, 3, [5]*3, False, None, 3, 14.0, 15.7, "Martell"),
    _e("bipy3+Cu2+", "Cu2+", _bipy6, 3, [5]*3, False, None, 3, 14.0, 17.0, "Martell"),
    _e("bipy3+Zn2+", "Zn2+", _bipy6, 3, [5]*3, False, None, 3, 14.0, 13.6, "NIST"),
    _e("bipy3+Cd2+", "Cd2+", _bipy6, 3, [5]*3, False, None, 3, 14.0, 12.0, "NIST"),
    _e("bipy3+Mn2+", "Mn2+", _bipy6, 3, [5]*3, False, None, 3, 14.0,  7.0, "NIST"),
    # Phenanthroline (phen, N₂)
    _e("phen+Fe2+", "Fe2+", _phen2, 1, [5], False, None, 1, 14.0, 5.9, "NIST"),
    _e("phen+Cu2+", "Cu2+", _phen2, 1, [5], False, None, 1, 14.0, 9.0, "NIST"),
    _e("phen+Ni2+", "Ni2+", _phen2, 1, [5], False, None, 1, 14.0, 8.8, "NIST"),
    _e("phen+Co2+", "Co2+", _phen2, 1, [5], False, None, 1, 14.0, 7.1, "NIST"),
    _e("phen+Zn2+", "Zn2+", _phen2, 1, [5], False, None, 1, 14.0, 6.4, "NIST"),
    _e("phen+Mn2+", "Mn2+", _phen2, 1, [5], False, None, 1, 14.0, 4.0, "NIST"),
    _e("phen+Cd2+", "Cd2+", _phen2, 1, [5], False, None, 1, 14.0, 5.4, "NIST"),
    # tris-phen cumulative
    _e("phen3+Fe2+", "Fe2+", _phen6, 3, [5]*3, False, None, 3, 14.0, 21.0, "NIST"),
    _e("phen3+Ni2+", "Ni2+", _phen6, 3, [5]*3, False, None, 3, 14.0, 24.0, "NIST"),
    _e("phen3+Co2+", "Co2+", _phen6, 3, [5]*3, False, None, 3, 14.0, 19.9, "NIST"),
    _e("phen3+Zn2+", "Zn2+", _phen6, 3, [5]*3, False, None, 3, 14.0, 17.0, "NIST"),
    # Terpyridine (terpy, N₃)
    _e("terpy+Fe2+", "Fe2+", _terpy, 2, [5]*2, False, None, 1, 14.0,  7.1, "NIST"),
    _e("terpy+Cu2+", "Cu2+", _terpy, 2, [5]*2, False, None, 1, 14.0, 12.6, "NIST"),
    _e("terpy+Ni2+", "Ni2+", _terpy, 2, [5]*2, False, None, 1, 14.0, 10.7, "NIST"),
    _e("terpy+Co2+", "Co2+", _terpy, 2, [5]*2, False, None, 1, 14.0,  9.9, "NIST"),
    _e("terpy+Zn2+", "Zn2+", _terpy, 2, [5]*2, False, None, 1, 14.0,  8.5, "NIST"),
    _e("terpy+Mn2+", "Mn2+", _terpy, 2, [5]*2, False, None, 1, 14.0,  4.4, "NIST"),
    _e("terpy+Cd2+", "Cd2+", _terpy, 2, [5]*2, False, None, 1, 14.0,  6.3, "NIST"),
    # Picolinic acid (pic, N,O)
    _e("pic+Cu2+", "Cu2+", _pic, 1, [5], False, None, 1, 14.0, 8.0, "NIST"),
    _e("pic+Ni2+", "Ni2+", _pic, 1, [5], False, None, 1, 14.0, 5.3, "NIST"),
    _e("pic+Zn2+", "Zn2+", _pic, 1, [5], False, None, 1, 14.0, 4.7, "NIST"),
    _e("pic+Co2+", "Co2+", _pic, 1, [5], False, None, 1, 14.0, 4.3, "NIST"),
    _e("pic+Mn2+", "Mn2+", _pic, 1, [5], False, None, 1, 14.0, 3.0, "NIST"),
    _e("pic+Cd2+", "Cd2+", _pic, 1, [5], False, None, 1, 14.0, 4.2, "NIST"),
    _e("pic+Pb2+", "Pb2+", _pic, 1, [5], False, None, 1, 14.0, 5.4, "NIST"),
    _e("pic+Fe3+", "Fe3+", _pic, 1, [5], False, None, 1, 14.0, 9.9, "NIST"),
    # Dipicolinic acid (DPA, N,O₂)
    _e("DPA+Cu2+", "Cu2+", _dpa, 2, [5]*2, False, None, 1, 14.0,  9.1, "NIST"),
    _e("DPA+Zn2+", "Zn2+", _dpa, 2, [5]*2, False, None, 1, 14.0,  6.4, "NIST"),
    _e("DPA+Ni2+", "Ni2+", _dpa, 2, [5]*2, False, None, 1, 14.0,  7.3, "NIST"),
    _e("DPA+Fe3+", "Fe3+", _dpa, 2, [5]*2, False, None, 1, 14.0, 12.0, "NIST"),
    _e("DPA+Ca2+", "Ca2+", _dpa, 2, [5]*2, False, None, 1, 14.0,  4.5, "NIST"),
    _e("DPA+Mn2+", "Mn2+", _dpa, 2, [5]*2, False, None, 1, 14.0,  5.2, "NIST"),
    _e("DPA+Cd2+", "Cd2+", _dpa, 2, [5]*2, False, None, 1, 14.0,  5.7, "NIST"),
    _e("DPA+La3+", "La3+", _dpa, 2, [5]*2, False, None, 1, 14.0,  7.4, "NIST"),
    _e("DPA+Gd3+", "Gd3+", _dpa, 2, [5]*2, False, None, 1, 14.0,  7.9, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # 8-HYDROXYQUINOLINE (oxine, N_pyr + O_phenolate)
    # ═════════════════════════════════════════════════════════════════
    _e("8HQ+Cu2+", "Cu2+", _8hq, 1, [5], False, None, 1, 14.0, 12.2, "NIST"),
    _e("8HQ+Zn2+", "Zn2+", _8hq, 1, [5], False, None, 1, 14.0,  8.6, "NIST"),
    _e("8HQ+Al3+", "Al3+", _8hq, 1, [5], False, None, 1, 14.0,  8.8, "NIST"),
    _e("8HQ+Fe3+", "Fe3+", _8hq, 1, [5], False, None, 1, 14.0, 12.5, "Martell"),
    _e("8HQ+Ni2+", "Ni2+", _8hq, 1, [5], False, None, 1, 14.0,  9.9, "NIST"),
    _e("8HQ+Co2+", "Co2+", _8hq, 1, [5], False, None, 1, 14.0,  8.5, "NIST"),
    _e("8HQ+Mn2+", "Mn2+", _8hq, 1, [5], False, None, 1, 14.0,  6.4, "NIST"),
    _e("8HQ+Cd2+", "Cd2+", _8hq, 1, [5], False, None, 1, 14.0,  7.5, "NIST"),
    _e("8HQ+Pb2+", "Pb2+", _8hq, 1, [5], False, None, 1, 14.0,  9.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # IMIDAZOLE
    # ═════════════════════════════════════════════════════════════════
    _e("imid+Cu2+", "Cu2+", ["N_imidazole"], 0, [], False, None, 1, 14.0, 4.3, "NIST"),
    _e("imid+Zn2+", "Zn2+", ["N_imidazole"], 0, [], False, None, 1, 14.0, 2.6, "NIST"),
    _e("imid+Ni2+", "Ni2+", ["N_imidazole"], 0, [], False, None, 1, 14.0, 3.0, "NIST"),
    _e("imid+Co2+", "Co2+", ["N_imidazole"], 0, [], False, None, 1, 14.0, 2.4, "NIST"),
    _e("imid+Fe2+", "Fe2+", ["N_imidazole"], 0, [], False, None, 1, 14.0, 2.0, "NIST"),
    _e("imid+Cd2+", "Cd2+", ["N_imidazole"], 0, [], False, None, 1, 14.0, 2.7, "NIST"),
    _e("imid+Mn2+", "Mn2+", ["N_imidazole"], 0, [], False, None, 1, 14.0, 1.2, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # SALEN (N₂O₂ Schiff base)
    # ═════════════════════════════════════════════════════════════════
    _e("salen+Cu2+", "Cu2+", _salen, 2, [5,6], False, None, 1, 14.0, 18.0, "Martell"),
    _e("salen+Ni2+", "Ni2+", _salen, 2, [5,6], False, None, 1, 14.0, 13.5, "Martell"),
    _e("salen+Co2+", "Co2+", _salen, 2, [5,6], False, None, 1, 14.0, 10.5, "Martell"),
    _e("salen+Zn2+", "Zn2+", _salen, 2, [5,6], False, None, 1, 14.0,  9.3, "Martell"),
    _e("salen+Mn2+", "Mn2+", _salen, 2, [5,6], False, None, 1, 14.0,  6.5, "Martell"),
    _e("salen+Fe3+", "Fe3+", _salen, 2, [5,6], False, None, 1, 14.0, 16.0, "Martell"),
    _e("salen+Pb2+", "Pb2+", _salen, 2, [5,6], False, None, 1, 14.0,  8.8, "Martell"),

    # ═════════════════════════════════════════════════════════════════
    # DMG (dimethylglyoxime, N₂O₂)
    # ═════════════════════════════════════════════════════════════════
    _e("DMG+Ni2+", "Ni2+", _dmg, 2, [5]*2, False, None, 1, 14.0, 17.2, "NIST"),
    _e("DMG+Cu2+", "Cu2+", _dmg, 2, [5]*2, False, None, 1, 14.0, 12.0, "NIST"),
    _e("DMG+Co2+", "Co2+", _dmg, 2, [5]*2, False, None, 1, 14.0, 10.6, "NIST"),
    _e("DMG+Zn2+", "Zn2+", _dmg, 2, [5]*2, False, None, 1, 14.0,  7.8, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # CATECHOLATE / PHENOLATE / SALICYLATE
    # ═════════════════════════════════════════════════════════════════
    _e("cat+Fe3+", "Fe3+", _cat2, 1, [5], False, None, 1, 14.0, 20.0, "NIST"),
    _e("cat+Al3+", "Al3+", _cat2, 1, [5], False, None, 1, 14.0, 16.3, "NIST"),
    _e("cat+Cu2+", "Cu2+", _cat2, 1, [5], False, None, 1, 14.0, 14.5, "Martell"),
    _e("cat+Ni2+", "Ni2+", _cat2, 1, [5], False, None, 1, 14.0,  8.6, "NIST"),
    _e("cat+Zn2+", "Zn2+", _cat2, 1, [5], False, None, 1, 14.0,  8.8, "NIST"),
    _e("cat+Cr3+", "Cr3+", _cat2, 1, [5], False, None, 1, 14.0, 11.4, "NIST"),
    _e("cat+Co2+", "Co2+", _cat2, 1, [5], False, None, 1, 14.0,  8.0, "NIST"),
    _e("cat+Pb2+", "Pb2+", _cat2, 1, [5], False, None, 1, 14.0, 11.0, "Martell"),
    _e("cat+Mn2+", "Mn2+", _cat2, 1, [5], False, None, 1, 14.0,  6.9, "NIST"),
    # Salicylate (O_carboxylate + O_phenolate, 6-ring)
    _e("sal+Cu2+", "Cu2+", _sal, 1, [6], False, None, 1, 14.0, 10.6, "NIST"),
    _e("sal+Fe3+", "Fe3+", _sal, 1, [6], False, None, 1, 14.0, 16.3, "NIST"),
    _e("sal+Ni2+", "Ni2+", _sal, 1, [6], False, None, 1, 14.0,  7.5, "NIST"),
    _e("sal+Zn2+", "Zn2+", _sal, 1, [6], False, None, 1, 14.0,  6.9, "NIST"),
    _e("sal+Co2+", "Co2+", _sal, 1, [6], False, None, 1, 14.0,  6.7, "NIST"),
    _e("sal+Mn2+", "Mn2+", _sal, 1, [6], False, None, 1, 14.0,  5.1, "NIST"),
    _e("sal+Al3+", "Al3+", _sal, 1, [6], False, None, 1, 14.0, 12.9, "NIST"),
    _e("sal+Pb2+", "Pb2+", _sal, 1, [6], False, None, 1, 14.0,  7.5, "NIST"),
    _e("sal+Cd2+", "Cd2+", _sal, 1, [6], False, None, 1, 14.0,  6.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # HYDROXAMATE
    # ═════════════════════════════════════════════════════════════════
    _e("AcHA+Fe3+",  "Fe3+", _aha, 1, [5], False, None, 1, 14.0, 11.4, "NIST"),
    _e("AcHA+Cu2+",  "Cu2+", _aha, 1, [5], False, None, 1, 14.0,  7.9, "NIST"),
    _e("AcHA+Ni2+",  "Ni2+", _aha, 1, [5], False, None, 1, 14.0,  5.0, "NIST"),
    _e("AcHA+Zn2+",  "Zn2+", _aha, 1, [5], False, None, 1, 14.0,  4.6, "NIST"),
    _e("AcHA+Al3+",  "Al3+", _aha, 1, [5], False, None, 1, 14.0,  7.9, "NIST"),
    _e("AcHA3+Fe3+", "Fe3+", _aha6, 3, [5]*3, False, None, 3, 14.0, 28.3, "Martell"),
    _e("DFO+Fe3+",   "Fe3+", _aha6, 3, [5]*3, False, None, 1, 14.0, 30.6, "NIST"),
    _e("DFO+Al3+",   "Al3+", _aha6, 3, [5]*3, False, None, 1, 14.0, 22.0, "Martell"),
    _e("DFO+Ga3+",   "Ga3+", _aha6, 3, [5]*3, False, None, 1, 14.0, 28.0, "Martell"),
    _e("DFO+Cu2+",   "Cu2+", _aha6, 3, [5]*3, False, None, 1, 14.0, 14.1, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # CARBOXYLATES
    # ═════════════════════════════════════════════════════════════════
    # Acetate (monodentate)
    _e("OAc+Cu2+", "Cu2+", _oac, 0, [], False, None, 1, 14.0, 2.2, "NIST"),
    _e("OAc+Pb2+", "Pb2+", _oac, 0, [], False, None, 1, 14.0, 2.7, "NIST"),
    _e("OAc+Zn2+", "Zn2+", _oac, 0, [], False, None, 1, 14.0, 1.6, "NIST"),
    _e("OAc+Ni2+", "Ni2+", _oac, 0, [], False, None, 1, 14.0, 1.4, "NIST"),
    _e("OAc+Ca2+", "Ca2+", _oac, 0, [], False, None, 1, 14.0, 1.2, "NIST"),
    _e("OAc+Fe3+", "Fe3+", _oac, 0, [], False, None, 1, 14.0, 3.4, "NIST"),
    _e("OAc+Co2+", "Co2+", _oac, 0, [], False, None, 1, 14.0, 1.5, "NIST"),
    _e("OAc+Mn2+", "Mn2+", _oac, 0, [], False, None, 1, 14.0, 1.4, "NIST"),
    _e("OAc+Cd2+", "Cd2+", _oac, 0, [], False, None, 1, 14.0, 1.9, "NIST"),
    _e("OAc+Mg2+", "Mg2+", _oac, 0, [], False, None, 1, 14.0, 1.3, "NIST"),
    _e("OAc+La3+", "La3+", _oac, 0, [], False, None, 1, 14.0, 1.8, "NIST"),
    _e("OAc+Al3+", "Al3+", _oac, 0, [], False, None, 1, 14.0, 2.2, "NIST"),
    # Oxalate (O₂, 5-ring)
    _e("ox+Cu2+", "Cu2+", _ox, 1, [5], False, None, 1, 14.0, 4.8, "NIST"),
    _e("ox+Ni2+", "Ni2+", _ox, 1, [5], False, None, 1, 14.0, 5.2, "NIST"),
    _e("ox+Fe3+", "Fe3+", _ox, 1, [5], False, None, 1, 14.0, 7.5, "NIST"),
    _e("ox+Pb2+", "Pb2+", _ox, 1, [5], False, None, 1, 14.0, 4.9, "NIST"),
    _e("ox+Ca2+", "Ca2+", _ox, 1, [5], False, None, 1, 14.0, 3.0, "NIST"),
    _e("ox+Co2+", "Co2+", _ox, 1, [5], False, None, 1, 14.0, 4.7, "NIST"),
    _e("ox+Zn2+", "Zn2+", _ox, 1, [5], False, None, 1, 14.0, 4.9, "NIST"),
    _e("ox+Mn2+", "Mn2+", _ox, 1, [5], False, None, 1, 14.0, 3.9, "NIST"),
    _e("ox+Cd2+", "Cd2+", _ox, 1, [5], False, None, 1, 14.0, 3.9, "NIST"),
    _e("ox+Mg2+", "Mg2+", _ox, 1, [5], False, None, 1, 14.0, 3.4, "NIST"),
    _e("ox+Al3+", "Al3+", _ox, 1, [5], False, None, 1, 14.0, 6.1, "NIST"),
    _e("ox+La3+", "La3+", _ox, 1, [5], False, None, 1, 14.0, 5.0, "NIST"),
    _e("ox+Cr3+", "Cr3+", _ox, 1, [5], False, None, 1, 14.0, 5.5, "NIST"),
    _e("ox2+Cu2+", "Cu2+", _ox4, 2, [5]*2, False, None, 2, 14.0, 8.4, "NIST"),
    _e("ox2+Ni2+", "Ni2+", _ox4, 2, [5]*2, False, None, 2, 14.0, 7.4, "NIST"),
    _e("ox2+Fe3+", "Fe3+", _ox4, 2, [5]*2, False, None, 2, 14.0, 13.8, "NIST"),
    _e("ox3+Fe3+", "Fe3+", _ox6, 3, [5]*3, False, None, 3, 14.0, 18.5, "NIST"),
    # Malonate (O₂, 6-ring)
    _e("mal+Cu2+", "Cu2+", _mal, 1, [6], False, None, 1, 14.0, 5.7, "NIST"),
    _e("mal+Ni2+", "Ni2+", _mal, 1, [6], False, None, 1, 14.0, 4.1, "NIST"),
    _e("mal+Co2+", "Co2+", _mal, 1, [6], False, None, 1, 14.0, 3.7, "NIST"),
    _e("mal+Zn2+", "Zn2+", _mal, 1, [6], False, None, 1, 14.0, 3.6, "NIST"),
    _e("mal+Mn2+", "Mn2+", _mal, 1, [6], False, None, 1, 14.0, 3.1, "NIST"),
    _e("mal+Cd2+", "Cd2+", _mal, 1, [6], False, None, 1, 14.0, 3.4, "NIST"),
    _e("mal+Fe3+", "Fe3+", _mal, 1, [6], False, None, 1, 14.0, 7.7, "NIST"),
    _e("mal+Ca2+", "Ca2+", _mal, 1, [6], False, None, 1, 14.0, 2.5, "NIST"),
    _e("mal+Pb2+", "Pb2+", _mal, 1, [6], False, None, 1, 14.0, 4.0, "NIST"),
    # Succinate (O₂, 7-ring)
    _e("succ+Cu2+", "Cu2+", _succ, 1, [7], False, None, 1, 14.0, 3.3, "NIST"),
    _e("succ+Ni2+", "Ni2+", _succ, 1, [7], False, None, 1, 14.0, 2.4, "NIST"),
    _e("succ+Co2+", "Co2+", _succ, 1, [7], False, None, 1, 14.0, 2.3, "NIST"),
    _e("succ+Zn2+", "Zn2+", _succ, 1, [7], False, None, 1, 14.0, 2.3, "NIST"),
    _e("succ+Mn2+", "Mn2+", _succ, 1, [7], False, None, 1, 14.0, 1.3, "NIST"),
    _e("succ+Ca2+", "Ca2+", _succ, 1, [7], False, None, 1, 14.0, 1.2, "NIST"),
    _e("succ+Pb2+", "Pb2+", _succ, 1, [7], False, None, 1, 14.0, 2.8, "NIST"),
    # Tartrate (O₂ + 2×OH)
    _e("tart+Cu2+", "Cu2+", _tart, 2, [5]*2, False, None, 1, 14.0, 3.2, "NIST"),
    _e("tart+Ni2+", "Ni2+", _tart, 2, [5]*2, False, None, 1, 14.0, 2.1, "NIST"),
    _e("tart+Zn2+", "Zn2+", _tart, 2, [5]*2, False, None, 1, 14.0, 2.7, "NIST"),
    _e("tart+Ca2+", "Ca2+", _tart, 2, [5]*2, False, None, 1, 14.0, 1.8, "NIST"),
    _e("tart+Pb2+", "Pb2+", _tart, 2, [5]*2, False, None, 1, 14.0, 3.8, "NIST"),
    _e("tart+Fe3+", "Fe3+", _tart, 2, [5]*2, False, None, 1, 14.0, 6.5, "NIST"),
    _e("tart+Co2+", "Co2+", _tart, 2, [5]*2, False, None, 1, 14.0, 2.2, "NIST"),
    _e("tart+Mn2+", "Mn2+", _tart, 2, [5]*2, False, None, 1, 14.0, 1.3, "NIST"),
    # Citrate (O₂ + OH)
    _e("cit+Fe3+", "Fe3+", _cit, 2, [5,6], False, None, 1, 14.0, 11.5, "NIST"),
    _e("cit+Ca2+", "Ca2+", _cit, 2, [5,6], False, None, 1, 14.0,  3.5, "NIST"),
    _e("cit+Cu2+", "Cu2+", _cit, 2, [5,6], False, None, 1, 14.0,  6.1, "NIST"),
    _e("cit+Pb2+", "Pb2+", _cit, 2, [5,6], False, None, 1, 14.0,  4.1, "Martell"),
    _e("cit+Ni2+", "Ni2+", _cit, 2, [5,6], False, None, 1, 14.0,  5.4, "NIST"),
    _e("cit+Zn2+", "Zn2+", _cit, 2, [5,6], False, None, 1, 14.0,  5.0, "NIST"),
    _e("cit+Mn2+", "Mn2+", _cit, 2, [5,6], False, None, 1, 14.0,  3.7, "NIST"),
    _e("cit+Cd2+", "Cd2+", _cit, 2, [5,6], False, None, 1, 14.0,  3.8, "Martell"),
    _e("cit+Co2+", "Co2+", _cit, 2, [5,6], False, None, 1, 14.0,  5.0, "NIST"),
    _e("cit+Al3+", "Al3+", _cit, 2, [5,6], False, None, 1, 14.0,  9.6, "NIST"),
    _e("cit+Mg2+", "Mg2+", _cit, 2, [5,6], False, None, 1, 14.0,  3.4, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # AMMONIA (monodentate N_amine)
    # ═════════════════════════════════════════════════════════════════
    _e("NH3+Cu2+", "Cu2+", ["N_amine"], 0, [], False, None, 1, 14.0, 4.0, "NIST"),
    _e("NH3+Ni2+", "Ni2+", ["N_amine"], 0, [], False, None, 1, 14.0, 2.7, "NIST"),
    _e("NH3+Zn2+", "Zn2+", ["N_amine"], 0, [], False, None, 1, 14.0, 2.2, "NIST"),
    _e("NH3+Co2+", "Co2+", ["N_amine"], 0, [], False, None, 1, 14.0, 2.1, "NIST"),
    _e("NH3+Ag+",  "Ag+",  ["N_amine"], 0, [], False, None, 1, 14.0, 3.3, "NIST"),
    _e("NH3+Cd2+", "Cd2+", ["N_amine"], 0, [], False, None, 1, 14.0, 2.5, "NIST"),
    _e("NH3+Hg2+", "Hg2+", ["N_amine"], 0, [], False, None, 1, 14.0, 8.8, "NIST"),
    _e("NH3_4+Cu2+", "Cu2+", ["N_amine"]*4, 0, [], False, None, 4, 14.0, 12.6, "NIST"),
    _e("NH3_2+Ag+",  "Ag+",  ["N_amine"]*2, 0, [], False, None, 2, 14.0, 7.2, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # SULFUR DONORS
    # ═════════════════════════════════════════════════════════════════
    # Cysteine (S,N,O)
    _e("cys+Cd2+", "Cd2+", _cys, 2, [5]*2, False, None, 1, 14.0,  9.1, "NIST"),
    _e("cys+Pb2+", "Pb2+", _cys, 2, [5]*2, False, None, 1, 14.0, 12.2, "NIST"),
    _e("cys+Zn2+", "Zn2+", _cys, 2, [5]*2, False, None, 1, 14.0,  9.2, "NIST"),
    _e("cys+Cu2+", "Cu2+", _cys, 2, [5]*2, False, None, 1, 14.0, 10.3, "Martell"),
    _e("cys+Ni2+", "Ni2+", _cys, 2, [5]*2, False, None, 1, 14.0,  8.7, "NIST"),
    _e("cys+Co2+", "Co2+", _cys, 2, [5]*2, False, None, 1, 14.0,  7.6, "NIST"),
    # Penicillamine (S,N,O)
    _e("pen+Cu2+", "Cu2+", _pen, 2, [5]*2, False, None, 1, 14.0, 16.5, "NIST"),
    _e("pen+Pb2+", "Pb2+", _pen, 2, [5]*2, False, None, 1, 14.0, 12.3, "NIST"),
    _e("pen+Cd2+", "Cd2+", _pen, 2, [5]*2, False, None, 1, 14.0, 10.4, "NIST"),
    _e("pen+Zn2+", "Zn2+", _pen, 2, [5]*2, False, None, 1, 14.0, 10.5, "NIST"),
    _e("pen+Ni2+", "Ni2+", _pen, 2, [5]*2, False, None, 1, 14.0,  9.5, "NIST"),
    _e("pen+Hg2+", "Hg2+", _pen, 2, [5]*2, False, None, 1, 14.0, 22.2, "NIST"),
    # DMSA (S₂O₂)
    _e("DMSA+Pb2+", "Pb2+", _dmsa, 2, [5]*2, False, None, 1, 14.0, 17.2, "Aposhian"),
    _e("DMSA+Hg2+", "Hg2+", _dmsa, 2, [5]*2, False, None, 1, 14.0, 34.5, "Aposhian"),
    _e("DMSA+Cd2+", "Cd2+", _dmsa, 2, [5]*2, False, None, 1, 14.0, 14.4, "Aposhian"),
    _e("DMSA+Cu2+", "Cu2+", _dmsa, 2, [5]*2, False, None, 1, 14.0, 12.3, "Martell"),
    _e("DMSA+Zn2+", "Zn2+", _dmsa, 2, [5]*2, False, None, 1, 14.0, 10.7, "Martell"),
    # Thioglycolic acid (TGA, S,O)
    _e("TGA+Hg2+", "Hg2+", _tga, 1, [5], False, None, 1, 14.0, 17.0, "Martell"),
    _e("TGA+Pb2+", "Pb2+", _tga, 1, [5], False, None, 1, 14.0,  7.8, "Martell"),
    _e("TGA+Cu2+", "Cu2+", _tga, 1, [5], False, None, 1, 14.0,  8.0, "NIST"),
    _e("TGA+Ni2+", "Ni2+", _tga, 1, [5], False, None, 1, 14.0,  5.5, "NIST"),
    _e("TGA+Zn2+", "Zn2+", _tga, 1, [5], False, None, 1, 14.0,  5.0, "NIST"),
    _e("TGA+Cd2+", "Cd2+", _tga, 1, [5], False, None, 1, 14.0,  6.9, "NIST"),
    # Dithiocarbamate (S₂, 4-ring)
    _e("DTC+Cu2+", "Cu2+", _dtc2, 1, [4], False, None, 1, 14.0, 11.0, "Martell"),
    _e("DTC+Pb2+", "Pb2+", _dtc2, 1, [4], False, None, 1, 14.0,  9.4, "Martell"),
    _e("DTC+Cd2+", "Cd2+", _dtc2, 1, [4], False, None, 1, 14.0,  8.6, "Martell"),
    _e("DTC+Zn2+", "Zn2+", _dtc2, 1, [4], False, None, 1, 14.0,  5.1, "Martell"),
    _e("DTC+Ni2+", "Ni2+", _dtc2, 1, [4], False, None, 1, 14.0,  4.8, "Martell"),
    _e("DTC+Fe3+", "Fe3+", _dtc2, 1, [4], False, None, 1, 14.0, 12.0, "Martell"),
    _e("DTC+Co2+", "Co2+", _dtc2, 1, [4], False, None, 1, 14.0,  4.0, "Martell"),
    # Thiosulfate
    _e("S2O3+Ag+",  "Ag+",  ["S_thiosulfate"], 0, [], False, None, 1, 14.0, 8.8, "NIST"),
    _e("S2O3+Hg2+", "Hg2+", ["S_thiosulfate"], 0, [], False, None, 1, 14.0, 29.3, "NIST"),
    _e("S2O3+Pb2+", "Pb2+", ["S_thiosulfate"], 0, [], False, None, 1, 14.0,  3.0, "NIST"),
    _e("S2O3+Cd2+", "Cd2+", ["S_thiosulfate"], 0, [], False, None, 1, 14.0,  4.0, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # HALIDES
    # ═════════════════════════════════════════════════════════════════
    _e("Cl+Ag+",   "Ag+",  ["Cl_chloride"], 0, [], False, None, 1, 14.0, 3.3, "NIST"),
    _e("Cl+Hg2+",  "Hg2+", ["Cl_chloride"], 0, [], False, None, 1, 14.0, 6.7, "NIST"),
    _e("Cl+Cd2+",  "Cd2+", ["Cl_chloride"], 0, [], False, None, 1, 14.0, 2.0, "NIST"),
    _e("Br+Hg2+",  "Hg2+", ["Br_bromide"], 0, [], False, None, 1, 14.0, 9.1, "NIST"),
    _e("Br+Ag+",   "Ag+",  ["Br_bromide"], 0, [], False, None, 1, 14.0, 4.3, "NIST"),
    _e("Br+Pb2+",  "Pb2+", ["Br_bromide"], 0, [], False, None, 1, 14.0, 1.8, "NIST"),
    _e("Br+Cd2+",  "Cd2+", ["Br_bromide"], 0, [], False, None, 1, 14.0, 2.3, "NIST"),
    _e("I+Hg2+",   "Hg2+", ["I_iodide"], 0, [], False, None, 1, 14.0, 12.9, "NIST"),
    _e("I+Ag+",    "Ag+",  ["I_iodide"], 0, [], False, None, 1, 14.0,  6.6, "NIST"),
    _e("I+Pb2+",   "Pb2+", ["I_iodide"], 0, [], False, None, 1, 14.0,  1.9, "NIST"),
    _e("I+Cd2+",   "Cd2+", ["I_iodide"], 0, [], False, None, 1, 14.0,  2.3, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # CROWN ETHERS (macrocyclic)
    # ═════════════════════════════════════════════════════════════════
    _e("18c6+K+",   "K+",   _18c6, 0, [], True, 0.140, 1, 14.0, 2.0, "Izatt"),
    _e("18c6+Na+",  "Na+",  _18c6, 0, [], True, 0.140, 1, 14.0, 0.7, "Izatt"),
    _e("18c6+Ba2+", "Ba2+", _18c6, 0, [], True, 0.140, 1, 14.0, 3.9, "Izatt"),
    _e("18c6+Ca2+", "Ca2+", _18c6, 0, [], True, 0.140, 1, 14.0, 0.5, "Izatt"),
    _e("18c6+Rb+",  "Rb+",  _18c6, 0, [], True, 0.140, 1, 14.0, 1.6, "Izatt"),
    _e("18c6+Cs+",  "Cs+",  _18c6, 0, [], True, 0.140, 1, 14.0, 1.0, "Izatt"),
    _e("18c6+Pb2+", "Pb2+", _18c6, 0, [], True, 0.140, 1, 14.0, 4.3, "Izatt"),
    _e("18c6+Ag+",  "Ag+",  _18c6, 0, [], True, 0.140, 1, 14.0, 1.5, "Izatt"),
    _e("18c6+Tl+",  "Tl+",  _18c6, 0, [], True, 0.140, 1, 14.0, 2.3, "Izatt"),
    _e("18c6+Li+",  "Li+",  _18c6, 0, [], True, 0.140, 1, 14.0, 0.8, "Izatt"),
    _e("15c5+Na+",  "Na+",  _15c5, 0, [], True, 0.092, 1, 14.0, 0.7, "Izatt"),
    _e("15c5+K+",   "K+",   _15c5, 0, [], True, 0.092, 1, 14.0, 0.7, "Izatt"),
    _e("15c5+Ca2+", "Ca2+", _15c5, 0, [], True, 0.092, 1, 14.0, 0.5, "Izatt"),
    _e("15c5+Ba2+", "Ba2+", _15c5, 0, [], True, 0.092, 1, 14.0, 2.3, "Izatt"),
    _e("15c5+Pb2+", "Pb2+", _15c5, 0, [], True, 0.092, 1, 14.0, 2.6, "Izatt"),
    _e("12c4+Li+",  "Li+",  _12c4, 0, [], True, 0.060, 1, 14.0, 0.5, "Izatt"),

    # ═════════════════════════════════════════════════════════════════
    # MACROCYCLIC AMINES
    # ═════════════════════════════════════════════════════════════════
    _e("cyclam+Cu2+", "Cu2+", _cyclam, 0, [], True, 0.085, 1, 14.0, 27.2, "NIST"),
    _e("cyclam+Ni2+", "Ni2+", _cyclam, 0, [], True, 0.085, 1, 14.0, 22.2, "NIST"),
    _e("cyclam+Zn2+", "Zn2+", _cyclam, 0, [], True, 0.085, 1, 14.0, 15.5, "NIST"),
    _e("cyclam+Co2+", "Co2+", _cyclam, 0, [], True, 0.085, 1, 14.0, 12.6, "NIST"),
    _e("cyclam+Cd2+", "Cd2+", _cyclam, 0, [], True, 0.085, 1, 14.0, 11.3, "NIST"),
    _e("cyclam+Pb2+", "Pb2+", _cyclam, 0, [], True, 0.085, 1, 14.0,  9.9, "NIST"),
    _e("cyclam+Hg2+", "Hg2+", _cyclam, 0, [], True, 0.085, 1, 14.0, 23.1, "NIST"),
    _e("cyclen+Cu2+", "Cu2+", _cyclen, 0, [], True, 0.070, 1, 14.0, 24.8, "NIST"),
    _e("cyclen+Ni2+", "Ni2+", _cyclen, 0, [], True, 0.070, 1, 14.0, 16.4, "NIST"),
    _e("cyclen+Zn2+", "Zn2+", _cyclen, 0, [], True, 0.070, 1, 14.0, 13.7, "NIST"),
    _e("cyclen+Co2+", "Co2+", _cyclen, 0, [], True, 0.070, 1, 14.0, 12.0, "NIST"),
    _e("cyclen+Cd2+", "Cd2+", _cyclen, 0, [], True, 0.070, 1, 14.0, 10.3, "NIST"),
    # DOTA
    _e("DOTA+Gd3+", "Gd3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 24.7, "NIST"),
    _e("DOTA+Cu2+", "Cu2+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 22.2, "NIST"),
    _e("DOTA+Ni2+", "Ni2+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 19.4, "NIST"),
    _e("DOTA+Zn2+", "Zn2+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 21.1, "NIST"),
    _e("DOTA+Ca2+", "Ca2+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 16.4, "NIST"),
    _e("DOTA+La3+", "La3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 22.9, "NIST"),
    _e("DOTA+Eu3+", "Eu3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 24.0, "NIST"),
    _e("DOTA+Lu3+", "Lu3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 25.4, "NIST"),
    _e("DOTA+Yb3+", "Yb3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 25.0, "NIST"),
    _e("DOTA+Pb2+", "Pb2+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 21.3, "NIST"),
    # NOTA
    _e("NOTA+Cu2+", "Cu2+", _nota, 3, [5]*3, True, 0.060, 1, 14.0, 21.6, "NIST"),
    _e("NOTA+Ga3+", "Ga3+", _nota, 3, [5]*3, True, 0.060, 1, 14.0, 29.6, "NIST"),
    _e("NOTA+Zn2+", "Zn2+", _nota, 3, [5]*3, True, 0.060, 1, 14.0, 18.3, "NIST"),
    _e("NOTA+Fe3+", "Fe3+", _nota, 3, [5]*3, True, 0.060, 1, 14.0, 24.2, "NIST"),
    _e("NOTA+In3+", "In3+", _nota, 3, [5]*3, True, 0.060, 1, 14.0, 26.2, "NIST"),
    _e("NOTA+Al3+", "Al3+", _nota, 3, [5]*3, True, 0.060, 1, 14.0, 18.6, "NIST"),

    # ═════════════════════════════════════════════════════════════════
    # EXPANSION BLOCK — push to 500+
    # ═════════════════════════════════════════════════════════════════

    # ── More phen stepwise ──
    _e("phen+Pb2+", "Pb2+", _phen2, 1, [5], False, None, 1, 14.0, 4.6, "NIST"),
    _e("phen+Fe3+", "Fe3+", _phen2, 1, [5], False, None, 1, 14.0, 6.5, "NIST"),
    _e("phen3+Cu2+", "Cu2+", _phen6, 3, [5]*3, False, None, 3, 14.0, 21.0, "NIST"),
    _e("phen3+Mn2+", "Mn2+", _phen6, 3, [5]*3, False, None, 3, 14.0, 10.1, "NIST"),
    _e("phen3+Cd2+", "Cd2+", _phen6, 3, [5]*3, False, None, 3, 14.0, 14.3, "NIST"),

    # ── More bipy stepwise ──
    _e("bipy+Fe3+", "Fe3+", _bipy2, 1, [5], False, None, 1, 14.0, 4.7, "NIST"),
    _e("bipy+Pb2+", "Pb2+", _bipy2, 1, [5], False, None, 1, 14.0, 3.0, "Martell"),

    # ── More terpy ──
    _e("terpy+Pb2+", "Pb2+", _terpy, 2, [5]*2, False, None, 1, 14.0, 5.2, "Martell"),
    _e("terpy+Fe3+", "Fe3+", _terpy, 2, [5]*2, False, None, 1, 14.0, 8.7, "Martell"),

    # ── More crown ether metals ──
    _e("18c6+Sr2+", "Sr2+", _18c6, 0, [], True, 0.140, 1, 14.0, 2.7, "Izatt"),
    _e("18c6+Mg2+", "Mg2+", _18c6, 0, [], True, 0.140, 1, 14.0, 0.1, "Izatt"),
    _e("18c6+Cd2+", "Cd2+", _18c6, 0, [], True, 0.140, 1, 14.0, 3.2, "Izatt"),
    _e("18c6+Hg2+", "Hg2+", _18c6, 0, [], True, 0.140, 1, 14.0, 1.0, "Izatt"),
    _e("15c5+Li+",  "Li+",  _15c5, 0, [], True, 0.092, 1, 14.0, 0.3, "Izatt"),
    _e("15c5+Ag+",  "Ag+",  _15c5, 0, [], True, 0.092, 1, 14.0, 0.8, "Izatt"),
    _e("15c5+Rb+",  "Rb+",  _15c5, 0, [], True, 0.092, 1, 14.0, 0.5, "Izatt"),
    _e("12c4+Na+",  "Na+",  _12c4, 0, [], True, 0.060, 1, 14.0, 0.8, "Izatt"),

    # ── More EDTA — remaining METAL_DB metals ──
    _e("EDTA+Tl+",  "Tl+",  _edta, 5, [5]*5, False, None, 1, 14.0,  6.4, "NIST"),

    # ── More DTPA metals ──
    _e("DTPA+Eu3+", "Eu3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 22.4, "NIST"),
    _e("DTPA+Yb3+", "Yb3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 22.6, "NIST"),
    _e("DTPA+Lu3+", "Lu3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 22.4, "NIST"),
    _e("DTPA+Fe2+", "Fe2+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 16.6, "NIST"),
    _e("DTPA+Al3+", "Al3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 18.6, "NIST"),
    _e("DTPA+Bi3+", "Bi3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 28.2, "NIST"),
    _e("DTPA+In3+", "In3+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 29.0, "NIST"),
    _e("DTPA+Th4+", "Th4+", _dtpa, 8, [5]*8, False, None, 1, 14.0, 28.8, "NIST"),

    # ── More DOTA lanthanides ──
    _e("DOTA+Ce3+", "Ce3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 23.4, "NIST"),
    _e("DOTA+Nd3+", "Nd3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 23.0, "NIST"),
    _e("DOTA+Sm3+", "Sm3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 23.0, "NIST"),
    _e("DOTA+Dy3+", "Dy3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 24.8, "NIST"),
    _e("DOTA+Er3+", "Er3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 24.4, "NIST"),
    _e("DOTA+Tb3+", "Tb3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 24.7, "NIST"),
    _e("DOTA+In3+", "In3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 23.9, "NIST"),
    _e("DOTA+Bi3+", "Bi3+", _dota, 4, [5]*4, True, 0.070, 1, 14.0, 30.3, "NIST"),

    # ── More NTA metals ──
    _e("NTA+Mg2+", "Mg2+", _nta, 3, [5]*3, False, None, 1, 14.0, 5.4, "NIST"),
    _e("NTA+Ba2+", "Ba2+", _nta, 3, [5]*3, False, None, 1, 14.0, 4.8, "NIST"),
    _e("NTA+Sr2+", "Sr2+", _nta, 3, [5]*3, False, None, 1, 14.0, 5.0, "NIST"),

    # ── More IDA metals ──
    _e("IDA+Hg2+", "Hg2+", _ida, 2, [5]*2, False, None, 1, 14.0, 12.5, "NIST"),
    _e("IDA+Al3+", "Al3+", _ida, 2, [5]*2, False, None, 1, 14.0,  6.2, "NIST"),
    _e("IDA+Mg2+", "Mg2+", _ida, 2, [5]*2, False, None, 1, 14.0,  2.9, "NIST"),

    # ── Bis-pic cumulative ──
    _e("pic2+Cu2+", "Cu2+", ["N_pyridine","N_pyridine","O_carboxylate","O_carboxylate"],
        2, [5]*2, False, None, 2, 14.0, 14.3, "NIST"),
    _e("pic2+Ni2+", "Ni2+", ["N_pyridine","N_pyridine","O_carboxylate","O_carboxylate"],
        2, [5]*2, False, None, 2, 14.0, 10.1, "NIST"),
    _e("pic2+Zn2+", "Zn2+", ["N_pyridine","N_pyridine","O_carboxylate","O_carboxylate"],
        2, [5]*2, False, None, 2, 14.0,  8.7, "NIST"),

    # ── More AcHA metals ──
    _e("AcHA+Co2+", "Co2+", _aha, 1, [5], False, None, 1, 14.0, 4.5, "NIST"),
    _e("AcHA+Mn2+", "Mn2+", _aha, 1, [5], False, None, 1, 14.0, 3.9, "NIST"),
    _e("AcHA+Cr3+", "Cr3+", _aha, 1, [5], False, None, 1, 14.0, 9.0, "NIST"),
    _e("AcHA+Ga3+", "Ga3+", _aha, 1, [5], False, None, 1, 14.0, 10.4, "NIST"),
    _e("AcHA+La3+", "La3+", _aha, 1, [5], False, None, 1, 14.0, 3.9, "NIST"),

    # ── More en metals ──
    _e("en+Hg2+", "Hg2+", _en2, 1, [5], False, None, 1, 14.0, 6.3, "NIST"),
    _e("en+Ag+",  "Ag+",  _en2, 1, [5], False, None, 1, 14.0, 5.0, "NIST"),

    # ── More gly metals ──
    _e("gly+Hg2+", "Hg2+", _gly, 1, [5], False, None, 1, 14.0, 10.3, "NIST"),
    _e("gly+Al3+", "Al3+", _gly, 1, [5], False, None, 1, 14.0, 6.5, "NIST"),
    _e("gly+La3+", "La3+", _gly, 1, [5], False, None, 1, 14.0, 3.5, "NIST"),

    # ── More cyclam/cyclen metals ──
    _e("cyclam+Fe2+", "Fe2+", _cyclam, 0, [], True, 0.085, 1, 14.0, 10.1, "NIST"),
    _e("cyclam+Mn2+", "Mn2+", _cyclam, 0, [], True, 0.085, 1, 14.0,  7.4, "NIST"),
    _e("cyclen+Pb2+", "Pb2+", _cyclen, 0, [], True, 0.070, 1, 14.0,  9.8, "NIST"),
    _e("cyclen+Mn2+", "Mn2+", _cyclen, 0, [], True, 0.070, 1, 14.0,  6.0, "NIST"),
    _e("cyclen+Fe2+", "Fe2+", _cyclen, 0, [], True, 0.070, 1, 14.0,  8.0, "NIST"),

    # ── More dien metals ──
    _e("dien+Hg2+", "Hg2+", _dien, 2, [5]*2, False, None, 1, 14.0, 12.0, "NIST"),
    _e("dien+Fe3+", "Fe3+", _dien, 2, [5]*2, False, None, 1, 14.0, 10.5, "Martell"),

    # ── Additional carboxylate metals ──
    _e("ox+Fe2+", "Fe2+", _ox, 1, [5], False, None, 1, 14.0, 3.1, "NIST"),
    _e("ox+Hg2+", "Hg2+", _ox, 1, [5], False, None, 1, 14.0, 4.7, "NIST"),
    _e("ox2+Co2+", "Co2+", _ox4, 2, [5]*2, False, None, 2, 14.0, 6.7, "NIST"),
    _e("ox2+Mn2+", "Mn2+", _ox4, 2, [5]*2, False, None, 2, 14.0, 5.3, "NIST"),
    _e("ox2+Zn2+", "Zn2+", _ox4, 2, [5]*2, False, None, 2, 14.0, 6.4, "NIST"),
    _e("ox3+Cr3+", "Cr3+", _ox6, 3, [5]*3, False, None, 3, 14.0, 15.3, "NIST"),
    _e("ox3+Al3+", "Al3+", _ox6, 3, [5]*3, False, None, 3, 14.0, 15.6, "NIST"),
    _e("mal+Al3+", "Al3+", _mal, 1, [6], False, None, 1, 14.0, 5.8, "NIST"),
    _e("mal+La3+", "La3+", _mal, 1, [6], False, None, 1, 14.0, 3.6, "NIST"),
    _e("succ+Fe3+", "Fe3+", _succ, 1, [7], False, None, 1, 14.0, 3.2, "NIST"),
    _e("succ+Cd2+", "Cd2+", _succ, 1, [7], False, None, 1, 14.0, 1.8, "NIST"),
    _e("succ+Fe2+", "Fe2+", _succ, 1, [7], False, None, 1, 14.0, 2.0, "NIST"),
    _e("tart+Cd2+", "Cd2+", _tart, 2, [5]*2, False, None, 1, 14.0, 2.3, "NIST"),
    _e("tart+Al3+", "Al3+", _tart, 2, [5]*2, False, None, 1, 14.0, 5.8, "NIST"),
    _e("cit+La3+", "La3+", _cit, 2, [5,6], False, None, 1, 14.0, 7.3, "NIST"),
    _e("cit+Gd3+", "Gd3+", _cit, 2, [5,6], False, None, 1, 14.0, 7.6, "NIST"),

    # ── Additional S-donor metals ──
    _e("TGA+Fe3+", "Fe3+", _tga, 1, [5], False, None, 1, 14.0, 11.0, "Martell"),
    _e("TGA+Co2+", "Co2+", _tga, 1, [5], False, None, 1, 14.0,  4.8, "NIST"),
    _e("TGA+Mn2+", "Mn2+", _tga, 1, [5], False, None, 1, 14.0,  3.5, "NIST"),
    _e("pen+Co2+", "Co2+", _pen, 2, [5]*2, False, None, 1, 14.0,  8.2, "NIST"),
    _e("pen+Fe3+", "Fe3+", _pen, 2, [5]*2, False, None, 1, 14.0, 15.5, "NIST"),
    _e("cys+Fe3+", "Fe3+", _cys, 2, [5]*2, False, None, 1, 14.0, 13.0, "Martell"),

    # ── Additional 8HQ metals ──
    _e("8HQ+Ga3+", "Ga3+", _8hq, 1, [5], False, None, 1, 14.0, 12.2, "NIST"),
    _e("8HQ+In3+", "In3+", _8hq, 1, [5], False, None, 1, 14.0, 10.6, "NIST"),
    _e("8HQ+La3+", "La3+", _8hq, 1, [5], False, None, 1, 14.0,  5.5, "NIST"),

    # ── Additional EGTA ──
    _e("EGTA+Cd2+", "Cd2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 16.7, "NIST"),
    _e("EGTA+Mn2+", "Mn2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 12.2, "NIST"),
    _e("EGTA+Fe2+", "Fe2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 11.8, "NIST"),
    _e("EGTA+Co2+", "Co2+", _egta, 5, [5,5,5,5,8], False, None, 1, 14.0, 12.5, "NIST"),

    # ── Additional halide cumulative ──
    _e("Cl+Sn2+",  "Sn2+", ["Cl_chloride"],   0, [], False, None, 1, 14.0, 1.5, "NIST"),
    _e("Cl+Bi3+",  "Bi3+", ["Cl_chloride"],   0, [], False, None, 1, 14.0, 2.4, "NIST"),
    _e("I+Cu+",    "Cu+",  ["I_iodide"],      0, [], False, None, 1, 14.0, 8.3, "NIST"),
    _e("Cl+Cu+",   "Cu+",  ["Cl_chloride"],   0, [], False, None, 1, 14.0, 2.7, "NIST"),
    _e("Br+Cu+",   "Cu+",  ["Br_bromide"],    0, [], False, None, 1, 14.0, 5.9, "NIST"),

    # ── Additional imidazole cumulative ──
    _e("imid4+Cu2+", "Cu2+", ["N_imidazole"]*4, 0, [], False, None, 4, 14.0, 14.7, "NIST"),
    _e("imid4+Zn2+", "Zn2+", ["N_imidazole"]*4, 0, [], False, None, 4, 14.0,  9.0, "NIST"),
    _e("imid4+Ni2+", "Ni2+", ["N_imidazole"]*4, 0, [], False, None, 4, 14.0, 10.0, "NIST"),
]


if __name__ == "__main__":
    print(f"Calibration dataset: {len(CAL_DATA)} complexes")
    metals = sorted(set(e["metal"] for e in CAL_DATA))
    print(f"Metals: {len(metals)} — {', '.join(metals)}")
    ligands = sorted(set(e["name"].split('+')[0] for e in CAL_DATA))
    print(f"Ligand families: {len(ligands)} — {', '.join(ligands)}")
    exp_range = [e["log_K_exp"] for e in CAL_DATA]
    print(f"log K range: {min(exp_range):.1f} to {max(exp_range):.1f}")
    from collections import Counter
    fam_counts = Counter(e["name"].split('+')[0] for e in CAL_DATA)
    print(f"\nPer family (top 25):")
    for fam, count in fam_counts.most_common(25):
        print(f"  {fam:15s} {count:3d}")
    # Check for duplicates
    names = [e["name"] for e in CAL_DATA]
    dupes = [n for n in set(names) if names.count(n) > 1]
    if dupes:
        print(f"\n⚠ DUPLICATES: {dupes}")