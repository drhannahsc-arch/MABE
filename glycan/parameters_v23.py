"""
MABE Glycan Module — Locked Parameters v2.3
All values from non-biological calibration sources (NIST, synthetic host-guest, ITC mutant panels).
Biological lectin data is holdout validation ONLY.

Version history:
  v2.1: eps_HB, beta_context locked (Fersht 1985, GLYCAM06 QM)
  v2.2: k_desolv series locked (Schwarz 1996, Jasra 1982)
  v2.3: eps_CH_pi residue-specific (Diehl 2024 ITC); eps_linker_net (WGA Bains 1992)
"""

# Hydrogen bond energy (effective, post-context-correction)
EPS_HB        = -5.0   # kJ/mol, intrinsic (Fersht 1985, Pace 2014)
BETA_CONTEXT  =  0.45  # dimensionless (GLYCAM06 QM, Kirschner 2008)
EPS_HB_EFF    = EPS_HB * BETA_CONTEXT  # = -2.25 kJ/mol

# CH-pi interaction energies (residue-specific)
# Source: Diehl et al. 2024, JACS Au 4:3028 (Galectin-3 W181 ITC mutant panel)
# Asensio 2013 literature cross-validation: Trp range 2.5-3.4 kJ/mol per contact
EPS_CH_PI_TRP = -3.5   # kJ/mol per contact (indole; Trp)   [LOCKED v2.3]
EPS_CH_PI_TYR = -1.9   # kJ/mol per contact (phenol/phenyl; Tyr, Phe) [LOCKED v2.3]
EPS_CH_PI_PHE = -1.9   # kJ/mol per contact (same as Tyr per Diehl 2024) [LOCKED v2.3]

# Linker net contribution per glycosidic bond
# Source: WGA (GlcNAc)3 -> (GlcNAc)4 plateau (Bains 1992 ITC)
EPS_LINKER_NET = -0.28  # kJ/mol per linkage [LOCKED v2.3]

# Desolvation penalties for OH burial (k_desolv per OH type)
# Source: Schwarz 1996, J. Solution Chemistry
K_DESOLV_EQ   = +2.4   # kJ/mol  equatorial OH (mean C1,C2 from Schwarz Table I) [LOCKED v2.2]
K_DESOLV_C6   = +11.2  # kJ/mol  primary OH (C6-CH2OH, Schwarz Table I) [LOCKED v2.2]
# Source: Jasra 1982 (Gal C2-ax analog)
K_DESOLV_AX   = +6.3   # kJ/mol  axial OH [PROVISIONAL v2.2 — needs Schwarz-corrected validation]
# Estimate pending direct measurement
K_DESOLV_NAC  = +8.5   # kJ/mol  N-acetyl burial [ESTIMATE v2.2]
# Carboxylate burial (Neu5Ac COO- at Siglec binding sites)
# Carboxylate is more strongly hydrated than primary OH (charge + 2 acceptors)
# Constrained: K_C6 < K_COO < 3*K_C6. Initial estimate from Marcus ion hydration
# analogy (COO- ~3x more hydrated than -CH2OH) tempered by partial charge screening.
K_DESOLV_COO  = +15.0  # kJ/mol  carboxylate burial [PROVISIONAL v2.4]

# Structural water bridge (G5)
# Conserved water molecules bridging sugar-OH to protein residues.
# Back-solved from osmotic stress data + galectin mutant series.
# Source: Clarke 2001 JACS; Chervenak 1995 ConA water displacement mutants.
EPS_WATER_BRIDGE = -3.5  # kJ/mol per conserved water [LOCKED v2.3]

# Residue-type CH-pi dispatch
CH_PI_EPS = {
    "Trp": EPS_CH_PI_TRP,
    "Tyr": EPS_CH_PI_TYR,
    "Phe": EPS_CH_PI_PHE,
    "anthracene": EPS_CH_PI_TYR,  # Synthetic TEM platform ≈ phenyl polarizability
    "none": 0.0,
}

# Desolvation lookup
K_DESOLV = {
    "K_EQ":  K_DESOLV_EQ,
    "K_AX":  K_DESOLV_AX,
    "K_C6":  K_DESOLV_C6,
    "K_NAC": K_DESOLV_NAC,
    "K_COO": K_DESOLV_COO,
}

# Davis receptor DG0 (re-anchored v5 with CH-pi explicit)
DG0_DAVIS = -19.30  # kJ/mol

# Per-scaffold DG0 (from anchor ligand, Phase 5 v5)
DG0 = {
    "ConA":  None,  # computed from Man anchor at runtime
    "WGA":   None,  # computed from GlcNAc anchor at runtime
    "PNA":   None,  # computed from Gal anchor at runtime
    "Gal3":  None,  # computed from Gal anchor at runtime
    "Davis": DG0_DAVIS,
}
