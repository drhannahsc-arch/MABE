"""
macrocycle_metal_dataset.py — Phase 14b: Expanded macrocycle-metal binding data.

New macrocycle classes NOT in cal_dataset.py:
  1. Cryptands [2.1.1], [2.2.1], [2.2.2] — Lehn/Izatt compilations
  2. Aza-crown ethers (diaza-18-crown-6, triaza variants)
  3. Thia-crown ethers (mixed S/O donors)
  4. Water-soluble porphyrins (TPPS4)
  5. Lariat ethers (armed crowns)

All log K values: water, 25°C, I = 0.1 M unless noted.
Thermodynamic stability constants (fully deprotonated ligand convention).

Sources:
  Izatt, Pawlak, Bradshaw, Bruening. Chem. Rev. 1991, 91, 1721-2085.
  Izatt, Bradshaw, Nielsen, Lamb, Christensen. Chem. Rev. 1985, 85, 271-339.
  Lehn, Sauvage. J. Am. Chem. Soc. 1975, 97, 6700-6707.
  Dietrich, Lehn, Sauvage. Tetrahedron Lett. 1969, 10, 2889-2892.
  Arnaud-Neu, Delgado, Schwing-Weill. Pure Appl. Chem. 2003, 75, 71-102.
  Pasternack et al. Inorg. Chem. 1973, 12, 2606-2610.
  Hambright. "Chemistry of Water-Soluble Porphyrins" in Kadish, Smith, Guilard
    The Porphyrin Handbook, Vol. 3, 2000, pp 129-210.
  Cooper, Rawle. Struct. Bond. 1990, 72, 1-72. (thia-crowns)
  Blake, Schröder. Adv. Inorg. Chem. 1990, 35, 1-80. (thia-crowns)
"""


def _e(name, metal, donors, chelate_rings, ring_sizes,
       macrocyclic, cavity_nm, n_lig_mol, pH, log_K_exp, source,
       smiles=None, macrocycle_class=None):
    d = {
        "name": name, "metal": metal, "donors": donors,
        "chelate_rings": chelate_rings, "ring_sizes": ring_sizes,
        "macrocyclic": macrocyclic, "cavity_nm": cavity_nm,
        "n_lig_mol": n_lig_mol, "pH": pH,
        "log_K_exp": log_K_exp, "source": source,
    }
    if smiles:
        d["smiles"] = smiles
    if macrocycle_class:
        d["macrocycle_class"] = macrocycle_class
    return d


# ═══════════════════════════════════════════════════════════════════════════
# DONOR SETS
# ═══════════════════════════════════════════════════════════════════════════

# Cryptands: bridgehead N + bridging O
_crypt211 = ["N_amine", "N_amine", "O_ether", "O_ether", "O_ether"]   # 5 donors
_crypt221 = ["N_amine", "N_amine", "O_ether", "O_ether", "O_ether",
             "O_ether", "O_ether"]                                      # 7 donors — 2N + 5O
_crypt222 = ["N_amine", "N_amine", "O_ether", "O_ether", "O_ether",
             "O_ether", "O_ether", "O_ether"]                          # 8 donors — 2N + 6O

# Aza-crowns
_diaza18c6 = ["N_amine", "N_amine", "O_ether", "O_ether",
              "O_ether", "O_ether"]                                     # 2N + 4O
_triaza18c6 = ["N_amine", "N_amine", "N_amine",
               "O_ether", "O_ether", "O_ether"]                        # 3N + 3O

# Thia-crowns (S replaces O)
_9s3 = ["S_thioether", "S_thioether", "S_thioether"]                   # 9-ane-S3
_12s4 = ["S_thioether"] * 4                                            # 12-ane-S4
_18s6 = ["S_thioether"] * 6                                            # 18-ane-S6
_18s4o2 = ["S_thioether", "S_thioether", "S_thioether",
           "S_thioether", "O_ether", "O_ether"]                        # mixed S/O

# Porphyrins: 4 pyrrole-N donors
_porph4 = ["N_pyridine", "N_pyridine", "N_pyridine", "N_pyridine"]     # pyrrolic N ≈ pyridine-like

# SMILES
_SMILES_CRYPT222 = "C(COCCOCCN1CCOCCOCCN1CCOCC)OCC"  # approximate
_SMILES_DIAZA18C6 = "C(COCCNCCOCCNC)COCC"            # approximate


# ═══════════════════════════════════════════════════════════════════════════
# 1. CRYPTANDS
# ═══════════════════════════════════════════════════════════════════════════
# Cavity radii from Lehn: [2.1.1] ~0.08 nm, [2.2.1] ~0.11 nm, [2.2.2] ~0.14 nm
# Chelate rings: each bridge forms 5-membered chelate with metal
# Ring sizes: -CH2CH2-O-CH2CH2- bridges give 5-membered chelate rings

_IZ91 = "Izatt1991"
_LE75 = "Lehn1975"
_AN03 = "ArnaudNeu2003"

CRYPTAND_DATA = [
    # ── [2.1.1] — small cavity, Li+ selective ──
    # cavity ~0.08 nm, 5 donors (2N+3O), 5 chelate rings all size 5
    _e("[2.1.1]+Li+",  "Li+",  _crypt211, 5, [5,5,5,5,5], True, 0.08, 1, 14.0, 5.5, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.1.1]+Na+",  "Na+",  _crypt211, 5, [5,5,5,5,5], True, 0.08, 1, 14.0, 3.2, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.1.1]+K+",   "K+",   _crypt211, 5, [5,5,5,5,5], True, 0.08, 1, 14.0, 2.3, _IZ91,
       macrocycle_class="cryptand"),

    # ── [2.2.1] — medium cavity, Na+ selective ──
    # cavity ~0.11 nm, 7 donors (2N+5O), 7 chelate rings
    _e("[2.2.1]+Li+",  "Li+",  _crypt221, 7, [5]*7, True, 0.11, 1, 14.0, 4.3, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.2.1]+Na+",  "Na+",  _crypt221, 7, [5]*7, True, 0.11, 1, 14.0, 5.4, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.2.1]+K+",   "K+",   _crypt221, 7, [5]*7, True, 0.11, 1, 14.0, 3.9, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.2.1]+Ca2+", "Ca2+", _crypt221, 7, [5]*7, True, 0.11, 1, 14.0, 6.9, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.2.1]+Sr2+", "Sr2+", _crypt221, 7, [5]*7, True, 0.11, 1, 14.0, 7.4, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.2.1]+Ba2+", "Ba2+", _crypt221, 7, [5]*7, True, 0.11, 1, 14.0, 6.3, _IZ91,
       macrocycle_class="cryptand"),

    # ── [2.2.2] — large cavity, K+ selective ──
    # cavity ~0.14 nm, 8 donors (2N+6O), 8 chelate rings
    _e("[2.2.2]+Li+",  "Li+",  _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 1.0, _LE75,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Na+",  "Na+",  _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 3.9, _LE75,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+K+",   "K+",   _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 5.4, _LE75,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Rb+",  "Rb+",  _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 4.4, _LE75,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Cs+",  "Cs+",  _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 1.5, _LE75,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Ca2+", "Ca2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 4.4, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Sr2+", "Sr2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 8.0, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Ba2+", "Ba2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 9.5, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Pb2+", "Pb2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 12.6, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Ag+",  "Ag+",  _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 9.6, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Cd2+", "Cd2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 10.4, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Hg2+", "Hg2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 12.0, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Zn2+", "Zn2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 10.8, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Tl+",  "Tl+",  _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 6.3, _IZ91,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Cu2+", "Cu2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 12.4, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Ni2+", "Ni2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 10.2, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Co2+", "Co2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 8.8, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Mn2+", "Mn2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 5.0, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Fe2+", "Fe2+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 7.4, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+La3+", "La3+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 3.7, _AN03,
       macrocycle_class="cryptand"),
    _e("[2.2.2]+Gd3+", "Gd3+", _crypt222, 8, [5]*8, True, 0.14, 1, 14.0, 4.2, _AN03,
       macrocycle_class="cryptand"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 2. AZA-CROWN ETHERS
# ═══════════════════════════════════════════════════════════════════════════
# diaza-18-crown-6: replaces 2 O with NH in 18c6
# cavity ~0.14 nm (same as 18c6), but N donors add IW sensitivity

_DIE85 = "Dietrich1985"
_IZ85 = "Izatt1985"

AZA_CROWN_DATA = [
    # ── diaza-18-crown-6: 2N + 4O ──
    # Higher affinity for transition metals than 18c6 due to N donors
    _e("diaza-18c6+K+",   "K+",   _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 2.0, _IZ85,
       smiles=_SMILES_DIAZA18C6, macrocycle_class="aza-crown"),
    _e("diaza-18c6+Na+",  "Na+",  _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 1.0, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Ba2+", "Ba2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 3.3, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Pb2+", "Pb2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 7.0, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Cu2+", "Cu2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 6.5, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Ni2+", "Ni2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 4.8, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Zn2+", "Zn2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 4.3, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Co2+", "Co2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 3.5, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Cd2+", "Cd2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 5.0, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Ca2+", "Ca2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 1.2, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Hg2+", "Hg2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 10.3, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Ag+",  "Ag+",  _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 5.8, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Mn2+", "Mn2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 2.0, _IZ85,
       macrocycle_class="aza-crown"),
    _e("diaza-18c6+Fe2+", "Fe2+", _diaza18c6, 6, [5]*6, True, 0.14, 1, 14.0, 3.0, _IZ85,
       macrocycle_class="aza-crown"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 3. THIA-CROWN ETHERS
# ═══════════════════════════════════════════════════════════════════════════
# S donors: soft, favors Cu+, Ag+, Hg2+, Pd2+ over alkali/alkaline earth
# Key data from Cooper & Rawle (Struct. Bond. 1990) and Blake & Schröder

_CR90 = "Cooper1990"
_BS90 = "Blake1990"

THIA_CROWN_DATA = [
    # ── [9]aneS3 (9-thiacrown-3, trithia): 3 S donors ──
    # cavity ~0.07 nm
    _e("[9]aneS3+Cu2+", "Cu2+", _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 2.7, _CR90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Cu+",  "Cu+",  _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 5.5, _CR90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Ag+",  "Ag+",  _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 4.3, _CR90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Ni2+", "Ni2+", _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 1.8, _CR90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Co2+", "Co2+", _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 1.3, _CR90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Zn2+", "Zn2+", _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 1.5, _CR90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Hg2+", "Hg2+", _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 5.0, _BS90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Cd2+", "Cd2+", _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 3.0, _BS90,
       macrocycle_class="thia-crown"),
    _e("[9]aneS3+Pb2+", "Pb2+", _9s3, 3, [5,5,5], True, 0.07, 1, 14.0, 2.8, _BS90,
       macrocycle_class="thia-crown"),

    # ── [12]aneS4 (12-thiacrown-4): 4 S donors ──
    # cavity ~0.09 nm
    _e("[12]aneS4+Cu2+", "Cu2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 4.1, _CR90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Cu+",  "Cu+",  _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 8.0, _CR90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Ni2+", "Ni2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 4.8, _CR90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Ag+",  "Ag+",  _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 7.3, _CR90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Hg2+", "Hg2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 8.5, _BS90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Co2+", "Co2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 2.6, _CR90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Zn2+", "Zn2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 3.0, _CR90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Cd2+", "Cd2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 5.0, _BS90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Pb2+", "Pb2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 5.5, _BS90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Fe2+", "Fe2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 2.2, _CR90,
       macrocycle_class="thia-crown"),
    _e("[12]aneS4+Mn2+", "Mn2+", _12s4, 4, [5,5,5,5], True, 0.09, 1, 14.0, 1.5, _CR90,
       macrocycle_class="thia-crown"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 4. WATER-SOLUBLE PORPHYRINS
# ═══════════════════════════════════════════════════════════════════════════
# TPPS4 = meso-tetrakis(4-sulfonatophenyl)porphyrin
# 4 pyrrole N donors, planar macrocycle, cavity ~0.20 nm
# Metalation constants from Pasternack, Hambright, and references therein
# These are formation constants for M2+ + H2(TPPS4) ⇌ M(TPPS4) + 2H+
# Converted to log K (metal-porphyrin) using pKa corrections in original refs.

_PA73 = "Pasternack1973"
_HA00 = "Hambright2000"

PORPHYRIN_DATA = [
    # ── TPPS4: 4 pyrrolic N donors ──
    # cavity ~0.20 nm (tetrapyrrole hole)
    # All chelate rings 5-membered (N-C-C-C-N bridge in pyrrole pairs)
    _e("TPPS4+Cu2+", "Cu2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 16.0, _PA73,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Zn2+", "Zn2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 11.0, _PA73,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Co2+", "Co2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 12.5, _HA00,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Ni2+", "Ni2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 10.5, _PA73,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Mn2+", "Mn2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 8.0, _HA00,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Fe2+", "Fe2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 12.0, _HA00,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Cd2+", "Cd2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 9.0, _PA73,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Pb2+", "Pb2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 9.5, _HA00,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Hg2+", "Hg2+", _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 14.0, _HA00,
       macrocycle_class="porphyrin"),
    _e("TPPS4+Ag+",  "Ag+",  _porph4, 4, [5,5,5,5], True, 0.20, 1, 14.0, 6.0, _HA00,
       macrocycle_class="porphyrin"),
]


# ═══════════════════════════════════════════════════════════════════════════
# 5. LARIAT ETHERS (ARMED CROWNS)
# ═══════════════════════════════════════════════════════════════════════════
# Crown ether with pendant arm(s) containing additional donor(s)
# Increased selectivity and binding strength vs parent crown

_GO89 = "Gokel1989"

# N-pivot lariat: aza-crown with -CH2CH2OCH3 arm
_lariat_N18c6 = ["N_amine", "O_ether", "O_ether", "O_ether",
                 "O_ether", "O_ether", "O_ether"]  # arm O counts as 7th donor

LARIAT_DATA = [
    _e("lariat-aza-18c6+K+",  "K+",  _lariat_N18c6, 7, [5]*7, True, 0.14, 1, 14.0, 3.2, _GO89,
       macrocycle_class="lariat"),
    _e("lariat-aza-18c6+Na+", "Na+", _lariat_N18c6, 7, [5]*7, True, 0.14, 1, 14.0, 2.5, _GO89,
       macrocycle_class="lariat"),
    _e("lariat-aza-18c6+Ba2+","Ba2+",_lariat_N18c6, 7, [5]*7, True, 0.14, 1, 14.0, 4.8, _GO89,
       macrocycle_class="lariat"),
    _e("lariat-aza-18c6+Ca2+","Ca2+",_lariat_N18c6, 7, [5]*7, True, 0.14, 1, 14.0, 2.1, _GO89,
       macrocycle_class="lariat"),
    _e("lariat-aza-18c6+Sr2+","Sr2+",_lariat_N18c6, 7, [5]*7, True, 0.14, 1, 14.0, 3.5, _GO89,
       macrocycle_class="lariat"),
    _e("lariat-aza-18c6+Pb2+","Pb2+",_lariat_N18c6, 7, [5]*7, True, 0.14, 1, 14.0, 5.5, _GO89,
       macrocycle_class="lariat"),
]


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATE
# ═══════════════════════════════════════════════════════════════════════════

MACROCYCLE_METAL_DATA = (
    CRYPTAND_DATA
    + AZA_CROWN_DATA
    + THIA_CROWN_DATA
    + PORPHYRIN_DATA
    + LARIAT_DATA
)


# ═══════════════════════════════════════════════════════════════════════════
# MACROCYCLE PROPERTY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
# Extends HOST_DB concept for macrocycle size-selectivity modeling

MACROCYCLE_PROPS = {
    "[2.1.1]":       {"cavity_nm": 0.08, "n_donors": 5, "donor_types": "2N+3O",
                      "class": "cryptand", "flexibility": "rigid"},
    "[2.2.1]":       {"cavity_nm": 0.11, "n_donors": 7, "donor_types": "2N+5O",
                      "class": "cryptand", "flexibility": "rigid"},
    "[2.2.2]":       {"cavity_nm": 0.14, "n_donors": 8, "donor_types": "2N+6O",
                      "class": "cryptand", "flexibility": "rigid"},
    "diaza-18c6":    {"cavity_nm": 0.14, "n_donors": 6, "donor_types": "2N+4O",
                      "class": "aza-crown", "flexibility": "semi-rigid"},
    "[9]aneS3":      {"cavity_nm": 0.07, "n_donors": 3, "donor_types": "3S",
                      "class": "thia-crown", "flexibility": "semi-rigid"},
    "[12]aneS4":     {"cavity_nm": 0.09, "n_donors": 4, "donor_types": "4S",
                      "class": "thia-crown", "flexibility": "semi-rigid"},
    "TPPS4":         {"cavity_nm": 0.20, "n_donors": 4, "donor_types": "4N",
                      "class": "porphyrin", "flexibility": "rigid"},
    "lariat-aza-18c6":{"cavity_nm": 0.14, "n_donors": 7, "donor_types": "1N+6O",
                      "class": "lariat", "flexibility": "flexible"},
}


if __name__ == "__main__":
    from collections import Counter
    print(f"Phase 14b macrocycle-metal dataset")
    print(f"Total entries: {len(MACROCYCLE_METAL_DATA)}")
    by_class = Counter(e.get("macrocycle_class", "unknown") for e in MACROCYCLE_METAL_DATA)
    for cls, n in by_class.most_common():
        print(f"  {cls:15s}: {n}")
    metals = Counter(e["metal"] for e in MACROCYCLE_METAL_DATA)
    print(f"Unique metals: {len(metals)}")
    for m, c in metals.most_common(10):
        print(f"  {m:8s}: {c}")