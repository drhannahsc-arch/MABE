"""
mabe/glycan/dam_brewer_2002_data.py -- Extracted ITC data
==========================================================

Source: Dam TK, Brewer CF (2002) Chem. Rev. 102:387-429
DOI: 10.1021/cr000401x

All values at 27C (300K) unless noted. Units as published.
Converted to kJ/mol where needed (1 kcal = 4.184 kJ).
"""

# =====================================================================
# TABLE 2: ConA + multivalent sugars at 27C
# =====================================================================

CONA_BINDING_DATA = {
    'MeaMan': {
        'Ka': 1.2e4, 'dG_kcal': -5.6, 'dH_kcal': -8.4, 'TdS_kcal': -2.8,
        'n': 1.0, 'source': 'Table 2, Dam & Brewer 2002',
    },
    'trimannoside': {
        'Ka': 3.9e5, 'dG_kcal': -7.6, 'dH_kcal': -14.7, 'TdS_kcal': -7.1,
        'n': 1.0, 'source': 'Table 2',
    },
}

# =====================================================================
# TABLE 3: DGL + multivalent sugars at 27C
# =====================================================================

DGL_BINDING_DATA = {
    'MeaMan': {
        'Ka': 0.46e4, 'dG_kcal': -4.9, 'dH_kcal': -8.2, 'TdS_kcal': -3.3,
        'n': 1.0, 'source': 'Table 3',
    },
    'trimannoside': {
        'Ka': 1.22e6, 'dG_kcal': -8.3, 'dH_kcal': -16.2, 'TdS_kcal': -7.9,
        'n': 1.0, 'source': 'Table 3',
    },
}

# =====================================================================
# TABLE 11: WGA + GlcNAc oligomers (Bains et al. 1992)
# =====================================================================

WGA_BINDING_DATA = {
    'GlcNAc': {
        'Ka': 410, 'dG_kcal': -3.7, 'dH_kcal': -6.1, 'TdS_kcal': -2.4,
        'source': 'Table 11, Bains 1992',
    },
    '(GlcNAc)2': {
        'Ka': 5300, 'dG_kcal': -5.1, 'dH_kcal': -15.6, 'TdS_kcal': -10.5,
        'source': 'Table 11',
    },
    '(GlcNAc)3': {
        'Ka': 11100, 'dG_kcal': -5.5, 'dH_kcal': -19.4, 'TdS_kcal': -13.9,
        'source': 'Table 11',
    },
    '(GlcNAc)4': {
        'Ka': 12300, 'dG_kcal': -5.6, 'dH_kcal': -19.2, 'TdS_kcal': -13.6,
        'source': 'Table 11',
    },
    '(GlcNAc)5': {
        'Ka': 19100, 'dG_kcal': -5.8, 'dH_kcal': -18.2, 'TdS_kcal': -12.4,
        'source': 'Table 11',
    },
}

# =====================================================================
# TABLE 11: UDA + GlcNAc oligomers
# =====================================================================

UDA_BINDING_DATA = {
    '(GlcNAc)2': {
        'Ka': 800, 'dG_kcal': -3.9, 'dH_kcal': -4.7, 'TdS_kcal': -0.8,
        'source': 'Table 11, Lee 1998',
    },
    '(GlcNAc)3': {
        'Ka': 6200, 'dG_kcal': -5.1, 'dH_kcal': -6.3, 'TdS_kcal': -1.2,
        'source': 'Table 11',
    },
    '(GlcNAc)4': {
        'Ka': 14400, 'dG_kcal': -5.6, 'dH_kcal': -5.1, 'TdS_kcal': 0.5,
        'source': 'Table 11',
    },
    '(GlcNAc)5': {
        'Ka': 26500, 'dG_kcal': -5.9, 'dH_kcal': -5.1, 'TdS_kcal': 0.8,
        'source': 'Table 11',
    },
}

# =====================================================================
# TABLE 10: Hevein + GlcNAc oligomers (Asensio 2000)
# =====================================================================

HEVEIN_BINDING_DATA = {
    '(GlcNAc)2': {
        'Ka': 616, 'dG_kcal': -3.8, 'dH_kcal': -6.3, 'dS_cal_K': -8.4,
        'source': 'Table 10, Asensio 2000',
    },
    '(GlcNAc)3': {
        'Ka': 8525, 'dG_kcal': -5.4, 'dH_kcal': -8.3, 'dS_cal_K': -9.9,
        'source': 'Table 10',
    },
    '(GlcNAc)4': {
        'Ka': 10850, 'dG_kcal': -5.5, 'dH_kcal': -9.5, 'dS_cal_K': -13.4,
        'source': 'Table 10',
    },
    '(GlcNAc)5': {
        'Ka': 474000, 'dG_kcal': -7.8, 'dH_kcal': -9.6, 'dS_cal_K': -6.3,
        'source': 'Table 10',
    },
}

# =====================================================================
# GALECTIN-3 DATA (Bachhawat-Sikder 2001, via Dam & Brewer)
# =====================================================================

GALECTIN3_BINDING_DATA = {
    'lactose': {
        'Ka': 1160, 'dH_kcal': -4.8,
        'source': 'Bachhawat-Sikder 2001 via Dam & Brewer Section II.A.5',
    },
    'LacNAc': {
        'Ka_ratio_vs_lac': 7,  # 7-fold higher than lactose
        'ddH_vs_lac_kcal': -3.3,  # more favorable
        'source': 'Section II.A.5',
    },
}

# =====================================================================
# SBA (Soybean Agglutinin) DATA (Gupta 1996, via Dam & Brewer)
# =====================================================================

SBA_BINDING_DATA = {
    'MebGalNAc': {
        'dH_kcal': -13.9,
        'source': 'Section IV.A, Gupta 1996',
    },
    'MebGal': {
        'dH_kcal': -10.6,
        'source': 'Section IV.A',
        'note': 'GalNAc 3.3 kcal/mol more favorable than Gal',
    },
}

# =====================================================================
# KEY FINDINGS FOR MABE
# =====================================================================
# 1. Solvent reorganization = 25-100% of binding enthalpy
#    (Chervenak & Toone 1994)
# 2. ddH values for deoxy analogues are NONLINEAR -- do not scale
#    with number of H-bonds (Section III.A.1)
# 3. Altered ordered water networks between DGL and ConA explain
#    ddH differences despite identical contact residues (Section V)
# 4. Osmotic stress: 1 water released for trimannoside, 5 for mannose
# 5. WGA three-subsite model: affinity plateaus at (GlcNAc)3
# 6. Enthalpy-entropy compensation ubiquitous across all systems
"""
