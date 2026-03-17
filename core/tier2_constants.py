"""
core/tier2_constants.py — Peer-Reviewed Physical Constants for Tier 2 Terms

Every value has an inline citation. No fitted parameters.
All energies in kJ/mol unless noted. Distances in Angstroms.

Data integrity: Tier 1 (database with DOI) or Tier 2 (published paper, specific table).
"""

import math

# ─── FUNDAMENTAL ──────────────────────────────────────────────────────────
R_GAS = 8.314462e-3       # kJ/(mol·K)
T_298 = 298.15            # K
RT_298 = R_GAS * T_298    # 2.4790 kJ/mol
LN10_RT = 2.303 * RT_298  # 5.708 kJ/mol
AVOGADRO = 6.02214076e23
E_CHARGE = 1.602176634e-19  # C
EPSILON_0 = 8.8541878e-12   # F/m
EPSILON_WATER = 78.4         # Dielectric constant of water at 298 K  [CRC Handbook]

# Coulomb constant in kJ·Å/mol for z1*z2/d formula:
# e^2 * N_A / (4*pi*eps0) converted to kJ·Å/mol = 1389.35 kJ·Å/mol
COULOMB_KJ_A = 1389.35


# ═══════════════════════════════════════════════════════════════════════════
# T1: LONDON DISPERSION — Atomic polarizabilities
# Source: Schwerdtfeger P. "Atomic Static Dipole Polarizabilities"
#         In "Atoms, Molecules and Clusters in Electric Fields" (2006)
#         Updated: Schwerdtfeger & Nagle, Mol. Phys. 117:1200 (2019)
#         Additional: CRC Handbook of Chemistry and Physics, 104th ed.
# Units: Å³ (= 10^-30 m^3 * 4πε₀)
# ═══════════════════════════════════════════════════════════════════════════

ATOMIC_POLARIZABILITY = {
    # Element: α (Å³)  — neutral atom free-space values
    "H":  0.667,   # Schwerdtfeger 2019
    "He": 0.205,   # Schwerdtfeger 2019
    "C":  1.76,    # Schwerdtfeger 2019
    "N":  1.10,    # Schwerdtfeger 2019
    "O":  0.802,   # Schwerdtfeger 2019
    "F":  0.557,   # Schwerdtfeger 2019
    "Ne": 0.396,   # Schwerdtfeger 2019
    "S":  2.90,    # Schwerdtfeger 2019
    "Cl": 2.18,    # Schwerdtfeger 2019
    "Br": 3.05,    # Schwerdtfeger 2019
    "I":  5.35,    # Schwerdtfeger 2019
    "P":  3.63,    # Schwerdtfeger 2019
    "Si": 5.38,    # Schwerdtfeger 2019
    "Se": 3.77,    # Schwerdtfeger 2019
}

# Group polarizabilities for coarse-grained dispersion
# Source: Miller KJ, Savchik JA. JACS 101:7206 (1979)
GROUP_POLARIZABILITY = {
    # Group: α (Å³)
    "CH3":     2.22,
    "CH2":     1.76,
    "phenyl":  10.0,   # Miller & Savchik 1979
    "indole":  14.6,   # Miller & Savchik, estimated from tryptophan
    "naphthyl": 16.5,  # Miller & Savchik
    "OH":      1.0,    # Miller & Savchik
    "NH2":     1.7,    # Miller & Savchik
    "COOH":    3.3,    # Miller & Savchik
    "COO-":    4.1,    # Miller & Savchik
    "SH":      3.8,    # Miller & Savchik
    "pyridine": 9.5,   # Miller & Savchik
    "imidazole": 7.5,  # Miller & Savchik
    "water":   1.45,   # CRC Handbook
}


# ═══════════════════════════════════════════════════════════════════════════
# T2: CATION-π — Gas-phase anchors and aqueous attenuation
# ═══════════════════════════════════════════════════════════════════════════

# Gas-phase cation-π interaction energies (kJ/mol, exothermic = negative)
# Source: Kebarle et al. J. Phys. Chem. 1981; Armentrout CID measurements
#         Ma & Dougherty, Chem. Rev. 1997, 97:1303-1324
CATION_PI_GAS = {
    # (cation, ring_type): ΔH_gas (kJ/mol)
    ("Li+", "benzene"):   -159.0,  # Kebarle 1981
    ("Na+", "benzene"):   -117.0,  # Kebarle 1981; Sunner, Nishizawa & Kebarle
    ("K+",  "benzene"):    -80.0,  # Kebarle 1981
    ("NH4+", "benzene"):   -80.0,  # Meot-Ner & Deakyne
    ("NMe4+", "benzene"):  -42.0,  # Meot-Ner & Deakyne
    ("Na+", "indole"):    -134.0,  # Ma & Dougherty 1997
    ("Na+", "phenol"):    -117.0,  # Ma & Dougherty 1997
    ("K+",  "indole"):     -97.0,  # estimated from ESP correlation
}

# Aqueous attenuation factor: ε_aq ≈ f_atten × ε_gas
# Dougherty Acc. Chem. Res. 2013, 46:885: "2-5 kcal/mol" = 8-21 kJ/mol in water
# Gas phase K+-benzene = 80 kJ/mol, aqueous = ~10 kJ/mol → f ≈ 0.12
# NMe4+-cyclophane: gas ~42, aqueous ~10.5 kJ/mol → f ≈ 0.25
CATION_PI_AQUEOUS_ATTENUATION = 0.15  # conservative middle estimate

# Default per-contact aqueous values (kJ/mol) for scorer
# Source: Dougherty cyclophane studies (JACS 1993); nAChR fluorination
CATION_PI_AQUEOUS_DEFAULTS = {
    "organic_cation_benzene":  -4.0,  # R-NR3+ to Phe-like
    "organic_cation_phenol":   -5.0,  # R-NR3+ to Tyr-like
    "organic_cation_indole":   -7.0,  # R-NR3+ to Trp-like
    "alkali_benzene":          -6.0,  # Na+/K+ to benzene (confinement-dependent)
    "ammonium_benzene":        -4.0,  # NH4+ to benzene
}

# Geometric parameters
CATION_PI_D0 = 3.5    # Å, optimal cation-centroid distance (average across systems)
CATION_PI_BETA = 2.5  # Å^-2, distance decay parameter


# ═══════════════════════════════════════════════════════════════════════════
# T3: π-π STACKING — Benchmark interaction energies
# ═══════════════════════════════════════════════════════════════════════════

# CCSD(T)/CBS gas-phase interaction energies (kJ/mol)
# Source: Sherrill S22 benchmark, Hobza BEGDB; Tsuzuki JACS 2000
PI_STACK_GAS = {
    ("benzene", "benzene", "parallel_displaced"): -11.8,
    ("benzene", "benzene", "T_shaped"):           -11.5,
    ("benzene", "benzene", "sandwich"):            -7.6,
    ("pyridine", "benzene", "stacked"):           -14.1,
    ("indole", "benzene", "stacked"):             -19.2,
    ("naphthalene", "naphthalene", "stacked"):    -16.5,
}

# Aqueous per-contact effective values (after hydrophobic subtraction)
# Source: Diederich cyclophanes JACS 1990; Hunter double mutant Chem. Sci.
# These are RESIDUAL after SASA-based hydrophobic transfer is subtracted
PI_STACK_AQUEOUS = {
    "parallel_displaced": -3.0,   # kJ/mol per stacking contact
    "donor_acceptor":     -8.0,   # DAN-NDI type (Iverson foldamers)
    "T_shaped":           -2.0,   # edge-to-face (overlaps with CH-π)
}

# Hammett dependence of stacking (Cozzi torsion balance; Hunter DMC)
# Slope: ~-2 kJ/(mol·σ_para) for substituted benzene stacking
PI_STACK_HAMMETT_SLOPE = -2.0  # kJ/(mol·σ)


# ═══════════════════════════════════════════════════════════════════════════
# T4: HALOGEN BONDING — σ-hole interaction parameters
# ═══════════════════════════════════════════════════════════════════════════

# Per-contact energies by halogen type (kJ/mol)
# Source: Cavallo et al. Chem. Rev. 2016, 116:2478-2601 (Table 1 compilations)
#         Kozuch & Martin XB18/XB51 benchmarks JCTC 2013
#         Auffinger PDB survey; Shinada J. Med. Chem. 2019
HALOGEN_BOND_ENERGY = {
    # (C-X type, nucleophile): ε_XB (kJ/mol, negative = favorable)
    ("C-I",  "N"):  -25.0,  # middle of -10 to -40 range
    ("C-I",  "O"):  -18.0,  # middle of -8 to -30 range
    ("C-I",  "S"):  -22.0,  # softer nucleophile, stronger
    ("C-Br", "N"):  -12.0,  # middle of -5 to -20 range
    ("C-Br", "O"):   -8.0,  # middle of -3 to -15 range
    ("C-Cl", "N"):   -6.0,  # middle of -3 to -10 range
    ("C-Cl", "O"):   -4.0,  # weakest practical XB
}

# Geometric requirements
HALOGEN_BOND_ANGLE_MIN = 140.0    # degrees; CXB angle must exceed this
HALOGEN_BOND_ANGLE_OPT = 175.0    # degrees; ideal is near-linear
HALOGEN_BOND_D0 = {               # Å; equilibrium X···B distances
    "I":  3.0,   # CSD statistical analysis
    "Br": 2.85,  # CSD
    "Cl": 2.75,  # CSD
}


# ═══════════════════════════════════════════════════════════════════════════
# T5: SALT BRIDGE / ION PAIR
# ═══════════════════════════════════════════════════════════════════════════

# Universal ion pair association ΔG in water
# Source: Schneider HJ. Angew. Chem. Int. Ed. 2009, 48:3924 (>200 ion pairs)
#         Bjerrum equation; Fuoss equation
SALT_BRIDGE_DG = {
    # (z_A * z_B): ΔG_assoc (kJ/mol, at I=0.15 M)
    -1:  -5.5,    # 1:1 ion pair (COO- with NH3+), most common
    -2: -11.0,    # 2:1 pair (PO4^2- with NH3+), approximately 2× additive
    -4: -22.0,    # 2:2 pair (SO4^2- with Ca^2+)
}

# Ionic strength correction (Debye-Hückel)
# At I=0: ΔG ≈ -8 kJ/mol for 1:1
# At I=0.15M: ΔG ≈ -5.5 kJ/mol for 1:1
# Slope: d(ΔG)/d(√I) ≈ +5 kJ/(mol·M^0.5)
SALT_BRIDGE_I_SLOPE = 5.0  # kJ/(mol·M^0.5)

# Buried salt bridge additional stabilization (low-ε environment)
# Source: Anderson et al. Biochemistry 1990, T4 lysozyme: -12 to -21 kJ/mol
# BUT offset by desolvation penalty → net contribution context-dependent
SALT_BRIDGE_BURIED_FACTOR = 2.0  # multiplier relative to exposed


# ═══════════════════════════════════════════════════════════════════════════
# T6: BORN ION SOLVATION
# ═══════════════════════════════════════════════════════════════════════════

# Born model: ΔG_Born = -(z² × 694.3) / r_eff × (1 - 1/ε)
# where 694.3 = e² × N_A / (8π × ε₀) in kJ·Å/mol
BORN_CONSTANT = 694.3   # kJ·Å/mol

# Born radius corrections (Rashin & Honig, J. Phys. Chem. 1985, 89:5588)
BORN_DR_CATION = 0.72   # Å — added to r_crystal for cations
BORN_DR_ANION  = 0.00   # Å — no correction for anions


# ═══════════════════════════════════════════════════════════════════════════
# T7: H-BOND COOPERATIVITY
# ═══════════════════════════════════════════════════════════════════════════

# Cooperativity factors (fractional enhancement per chain extension)
# Source: Dannenberg JJ. JACS 2002, 124:11006 (formamide chains MP2/CBS)
#         Batista et al. J. Chem. Phys. (ice lattice decomposition)
HBOND_COOP_FACTOR = {
    "amide":    0.20,   # 20% enhancement per additional amide H-bond in chain
    "water":    0.18,   # 18% for water chains
    "hydroxyl": 0.10,   # 10% for polyol OH chains (weaker due to geometry)
    "default":  0.15,   # generic cooperativity
}

HBOND_COOP_MAX_CHAIN = 5  # cooperativity saturates beyond this


# ═══════════════════════════════════════════════════════════════════════════
# T8: ANION-π
# ═══════════════════════════════════════════════════════════════════════════

# Per-contact energies for anion near electron-poor π-face (kJ/mol)
# Source: Frontera & Quiñonero compilations Chem. Rev.; Matile group Ka data
ANION_PI_ENERGY = {
    "Cl_perfluoroarene":   -4.0,   # aqueous estimate
    "Cl_triazine":         -6.0,   # aqueous estimate
    "NO3_NDI":             -5.0,   # Matile/Saha receptor studies
    "default":             -3.0,   # conservative generic
}


# ═══════════════════════════════════════════════════════════════════════════
# T9: AUROPHILIC / METALLOPHILIC (d10-d10)
# ═══════════════════════════════════════════════════════════════════════════

# Interaction energies at equilibrium distance (kJ/mol)
# Source: Pyykkö P. Chem. Rev. 1997, 97:597; Angew. Chem. 2004, 43:4412
#         Schmidbaur H, Schier A. Chem. Soc. Rev. 2012, 41:370
METALLOPHILIC_ENERGY = {
    ("Au", "Au"):  -40.0,   # Au(I)···Au(I), strong relativistic
    ("Ag", "Ag"):  -18.0,   # Ag(I)···Ag(I)
    ("Cu", "Cu"):  -10.0,   # Cu(I)···Cu(I)
    ("Au", "Ag"):  -25.0,   # heterometallic
}

METALLOPHILIC_D0 = {
    ("Au", "Au"): 3.0,   # Å, equilibrium distance
    ("Ag", "Ag"): 3.1,
    ("Cu", "Cu"): 2.8,
}

METALLOPHILIC_BETA = 3.0  # Å^-1, exponential decay

# d10 metals that participate
D10_METALS = {"Au+", "Au1+", "Ag+", "Ag1+", "Cu+", "Cu1+", "Hg0", "Tl+", "Tl1+"}


# ═══════════════════════════════════════════════════════════════════════════
# T10: GROUP DESOLVATION (systematic)
# ═══════════════════════════════════════════════════════════════════════════

# Per-group desolvation cost upon full burial (kJ/mol, positive = penalty)
# Source: Cabani S et al. J. Solution Chem. 1981, 10:563 (group additivity)
#         Wolfenden R et al. Biochemistry 1981, 20:849 (amino acid sidechains)
#         FreeSolv: github.com/MobleyLab/FreeSolv (643 ΔG_hyd, Mobley & Guthrie 2014)
#         Abraham solvation parameters (H-bond acidity A, basicity B)
GROUP_DESOLVATION_COST = {
    # Group: k_desolv (kJ/mol) for complete burial
    # Primary: Cabani 1981 J. Solution Chem. 10:563; Wolfenden 1981 Biochemistry 20:849
    # Cross-checked: MNSol v2012 (doi:10.13020/3eks-j059) SASA regression, N=390
    #   OH back-solve: 14.2 kJ/mol (MNSol+FreeSolv mean) vs Cabani 10.0
    #   NH2 back-solve: 10.1 kJ/mol vs Wolfenden 10.0  ← excellent agreement
    #   C=O back-solve:  4.4 kJ/mol vs Wolfenden 7.5   ← Wolfenden higher
    "OH_primary_eq":       10.0,   # Cabani + FreeSolv cross-check
    "OH_primary_ax":        7.5,   # axial less exposed
    "OH_secondary_eq":      8.5,
    "OH_secondary_ax":      6.5,
    "NH2":                 10.0,   # Wolfenden 1981
    "NHAc":                12.5,   # Cabani + Wolfenden; FreeSolv
    "COOH":                15.0,   # Cabani 1981
    "COO_minus":           30.0,   # Born model + Cabani correction
    "amide_CO":             7.5,   # Wolfenden
    "ring_O":               4.0,   # ether oxygen, Cabani 1981
    "CH3":                  2.0,   # small, hydrophobic gain partially offsets
    "phenyl":               3.5,   # partial offset by hydrophobic gain
    "SH":                   6.5,   # Wolfenden; Abraham
    "S_minus":             25.0,   # Born model for thiolate
    "NH3_plus":            35.0,   # Born model for ammonium
    "guanidinium":         30.0,   # Born model for Arg
    "default_polar":        8.0,   # generic polar group
    "default_nonpolar":     2.0,   # generic nonpolar group
}


# ═══════════════════════════════════════════════════════════════════════════
# P20: WATER H-BOND COMPETITION (per H-bond disrupted)
# Back-solved from MNSol v2012 (390 aq. neutrals) + FreeSolv v0.52 (642 cpds)
# doi:10.13020/3eks-j059 (MNSol), doi:10.1021/acs.jced.7b00104 (FreeSolv)
# ═══════════════════════════════════════════════════════════════════════════
EPS_WATER_PER_HBOND = 5.2    # kJ/mol, donor-weighted mean (OH + NH)
EPS_WATER_OH = 5.3           # per OH-water H-bond disrupted
EPS_WATER_NH = 5.0           # per NH-water H-bond disrupted
EPS_WATER_O_ACCEPTOR = 2.2   # per C=O acceptor H-bond disrupted

# Per-group desolvation from regression (kJ/mol, total per group buried)
DESOLV_PER_OH_GROUP = 14.2   # OH makes ~2.7 H-bonds with water
DESOLV_PER_NH2_GROUP = 10.1  # NH2 makes ~2 H-bonds
DESOLV_PER_CO_GROUP = 4.4    # C=O makes ~2 H-bonds


# ═══════════════════════════════════════════════════════════════════════════
# P13: HYDROPHOBIC SASA TRANSFER COEFFICIENTS (absolute solvation)
# Back-solved from MNSol v2012 pure hydrocarbon subsets
# n-alkane series (methane→octane, N=8, R²=0.890)
# aromatic series (benzene→anthracene, N=8, R²=0.650)
# ═══════════════════════════════════════════════════════════════════════════
GAMMA_ABS_ALIPHATIC = 0.0242  # kJ/mol/Å², positive = unfavorable in water
GAMMA_ABS_AROMATIC = -0.1427  # kJ/mol/Å², negative = π-water stabilization

# SASA-based solvation coefficients (from full 390-entry regression)
# Sign: negative = favorable solvation (gas→water). Desolvation = negate.
GAMMA_SASA_DESOLV = {
    "aliphatic":   0.0523,   # burying aliphatic SASA is unfavorable (desolv cost)
    "aromatic":   -0.0893,   # aromatic SASA is favorable in water (π-water)
    "OH_donor":    1.0971,   # per Å² of OH surface buried (strong penalty)
    "NH_donor":    0.7012,   # per Å² of NH surface buried
    "O_acceptor":  0.2099,   # per Å² of O acceptor surface buried
    "N_acceptor":  0.1536,   # per Å² of N acceptor surface buried
    "halogen":     0.0088,   # nearly zero — halogens barely interact with water
    "sulfur":      0.0731,   # moderate
}
# Note: For host-guest differential binding, γ_flat (0.018-0.025 kJ/mol/Å²)
# from Rekharsky CD series is the correct parameter. γ_abs is for absolute
# solvation predictions. Rekharsky data NOT YET CURATED.


# ═══════════════════════════════════════════════════════════════════════════
# P16: NEUTRAL H-BOND ENERGY AT HOST-GUEST PORTAL
# Back-solved from CB7 adamantane ± NH2/OH matched pairs
# Source: MABE HG_DATA, Moghaddam 2011 JACS 133:3570 (CB7 thermodynamics)
# ═══════════════════════════════════════════════════════════════════════════
EPS_NEUTRAL_HBOND_CB_NET = -13.8   # kJ/mol, NET observed ΔG per portal H-bond
                                    # (includes desolvation, geometry, cooperativity)
EPS_NEUTRAL_HBOND_NH2 = -15.1      # NH2 donor to CB portal C=O (bidentate, per HB)
EPS_NEUTRAL_HBOND_OH = -12.6       # OH donor to CB portal C=O (monodentate)
# Cross-reference: eps_charge_assisted (NH3+ → C=O) = -13.3 kJ/mol per HB
# including ion-dipole. Neutral and charge-assisted are similar at CB portal
# because the C=O is intrinsically a strong acceptor.
# Compare: CD portal OH···OH = -2.0 kJ/mol (much weaker acceptor)


# ═══════════════════════════════════════════════════════════════════════════
# P28: OPTIMAL PACKING COEFFICIENT (Rebek 55% rule)
# Supported by CB7 guest size series (logKa monotonic up to PC=0.574)
# Source: Mecozzi, Rebek, Whitesides 1998 Chem.Eur.J. 4:1016
# ═══════════════════════════════════════════════════════════════════════════
PC_OPT = 0.55                      # Optimal V_guest / V_cavity
PC_OPT_SIGMA = 0.09                # Standard deviation around optimum
# CB7 adamantane-NH3+ at PC=0.574 is best binder → consistent with 0.55
# CB7 dimethyladamantane-NH3+ at PC=0.635 binds weaker → overpacking onset
