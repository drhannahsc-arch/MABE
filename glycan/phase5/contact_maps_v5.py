"""
MABE Glycan Phase 5 v5 — Contact Maps
Davis receptor: CH-π contacts explicit (Eades 2026, TEM anthracene platforms)
WGA secondary subsites: B (primary), C (subsite C), A (subsite A) corrected
All parameters v2.3 locked.
"""

# Davis receptor contact maps (Tromans/Eades GluHUT)
# Source: Eades et al. 2026 ChemRxiv; CH-pi confirmed from MD structural analysis
DAVIS_CONTACTS = {
    "Glc":   {"n_HB": 4, "n_CHP": 3, "buried": ["K_EQ","K_EQ","K_EQ","K_EQ"],
              "note": "4 equatorial OHs + 3 CH-pi (C1,C3,C5 axial Hs on beta-face, TEM platforms)"},
    "Gal":   {"n_HB": 2, "n_CHP": 1, "buried": ["K_EQ","K_AX","K_EQ","K_EQ"],
              "note": "C4-OH axial displaces sugar from beta-face; 2 HBs remain; 1 CH-pi survives (C1 partial)"},
    "Man":   {"n_HB": 2, "n_CHP": 1, "buried": ["K_AX","K_EQ","K_EQ","K_EQ"],
              "note": "C2-OH axial displaces sugar; symmetric to Gal case; near-identical pred dG"},
    "2dGlc": {"n_HB": 0, "n_CHP": 3, "buried": ["K_EQ","K_EQ","K_EQ"],
              "note": "C2-OH absent; 2 HBs lost (C2 urea arm); CH-pi intact (no beta-face disruption)"},
    "Fru":   {"n_HB": 1, "n_CHP": 0, "buried": ["K_AX","K_EQ","K_EQ"],
              "note": "Furanose geometry; no beta-face; 1 HB only. LOW confidence."},
    "GlcNAc":{"n_HB": 0, "n_CHP": 0, "buried": ["K_NAC","K_EQ","K_EQ"],
              "note": "NHAc steric clash prevents cavity entry. LOW confidence."},
}

# WGA contact maps (revised v5; Bains 1992 ITC)
# Subsite B (primary): n_HB=4, n_CHP=1(Tyr73), buried=[K_NAC,K_EQ]
# Subsite C (nonreducing GlcNAc): n_HB_add=+3, n_CHP_add=+1, buried_add=[K_EQ]
#   Key: beta1->4 glycosidic geometry flips nonreducing ring; NHAc is SOLVENT-EXPOSED at subsite C
# Subsite A (3rd GlcNAc): n_HB_add=+1, n_CHP_add=+1, buried_add=[K_EQ]
# Subsite D (4th GlcNAc): linker only; no additional contacts
WGA_CONTACTS = {
    "GlcNAc":    {"n_HB": 4, "n_CHP": 1, "buried": ["K_NAC","K_EQ"],       "n_linker": 0},
    "(GlcNAc)2": {"n_HB": 7, "n_CHP": 2, "buried": ["K_NAC","K_EQ","K_EQ"],"n_linker": 1},
    "(GlcNAc)3": {"n_HB": 8, "n_CHP": 3, "buried": ["K_NAC","K_EQ","K_EQ","K_EQ"],"n_linker": 2},
    "(GlcNAc)4": {"n_HB": 8, "n_CHP": 3, "buried": ["K_NAC","K_EQ","K_EQ","K_EQ"],"n_linker": 3},
}
