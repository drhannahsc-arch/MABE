"""
glycan/contact_maps.py — Contact maps for all scored scaffolds.

Each entry: {ligand: {n_HB, buried, n_CHP, res_type, n_linker}}
  n_HB      — direct H-bonds (integer)
  buried    — list of K_DESOLV keys for each buried OH
  n_CHP     — CH-π contacts (integer)
  res_type  — 'Trp' | 'Tyr' | 'Phe' | 'anthracene' | 'none'
  n_linker  — glycosidic bonds (integer)
  anchor    — True if this is the DG0 anchor (must match obs exactly)
  obs_dG    — observed ΔG in kJ/mol (from ITC; hold-out, never fitted)
  confidence— 'HIGH' | 'MEDIUM' | 'LOW'
  note      — source / structural justification

Sources:
  ConA:  PDB 5CNA (Naismith 1994), 1CVN (Loris 1994); Chervenak 1995 Biochemistry 34:5685
  WGA:   PDB 2UVO; Bains 1992 ITC
  PNA:   PDB 2PEL; Swaminathan 1998
  Gal3:  PDB 3GAL; Seetharaman 1998
  Davis: Eades et al. 2026 ChemRxiv DOI:10.26434/chemrxiv.10001540/v1; Davis 2012 Nat.Chem.
"""

# ══════════════════════════════════════════════════════════════════════════
# ConA (Concanavalin A) — mannose/glucose lectin
# Key CH-π: Tyr12 stacks against sugar beta-face
# PDB 5CNA: Man contacts O3-Asp208, O4-Arg228, O6-Asn14 (3 direct HBs)
# ══════════════════════════════════════════════════════════════════════════
CONA_CONTACTS = {
    "Man": {
        "n_HB": 3, "buried": ["K_EQ", "K_EQ", "K_C6"], "n_CHP": 1,
        "res_type": "Tyr", "n_linker": 0,
        "anchor": True, "obs_dG": -22.2, "confidence": "HIGH",
        "note": "5CNA: O3-Asp208, O4-Arg228, O6-Asn14; Tyr12 CH-pi; C3,C4 eq, C6 primary"
    },
    "Glc": {
        "n_HB": 2, "buried": ["K_EQ", "K_EQ", "K_C6"], "n_CHP": 1,
        "res_type": "Tyr", "n_linker": 0,
        "anchor": False, "obs_dG": -19.3, "confidence": "HIGH",
        "note": "C2-eq: O4 contact lost vs Man (4.4A crystal); 2 HBs remain; same Tyr12 CH-pi"
    },
    # diMan: Man1 (full Man contacts) + Man2 (branch contacts) + linker
    "1->2 diMan": {
        "n_HB": 8, "buried": ["K_EQ", "K_EQ", "K_C6", "K_EQ"], "n_CHP": 1,
        "res_type": "Tyr", "n_linker": 1,
        "anchor": False, "obs_dG": -28.5, "confidence": "HIGH",
        "note": "1CVN: Man1 full + Man2: 4 direct + 1 water-mediated HB (Thr15/Asp16); Naismith 1994"
    },
    "1->3 diMan": {
        "n_HB": 5, "buried": ["K_EQ", "K_EQ", "K_C6"], "n_CHP": 2,
        "res_type": "Tyr", "n_linker": 1,
        "anchor": False, "obs_dG": -25.1, "confidence": "HIGH",
        "note": "1CVN: Man1 full + Man2: 2 HBs (O3'/backbone) + Tyr12 extended CH-pi; Loris 1994"
    },
    "1->4 diMan": {
        "n_HB": 6, "buried": ["K_EQ", "K_EQ", "K_C6", "K_EQ"], "n_CHP": 2,
        "res_type": "Tyr", "n_linker": 1,
        "anchor": False, "obs_dG": -26.4, "confidence": "MEDIUM",
        "note": "1CVN: Man1 + Man2 at O4 position adds 3 HBs (moderate additional contacts)"
    },
    "1->6 diMan": {
        "n_HB": 3, "buried": ["K_EQ", "K_EQ", "K_C6"], "n_CHP": 1,
        "res_type": "Tyr", "n_linker": 1,
        "anchor": False, "obs_dG": -22.2, "confidence": "HIGH",
        "note": "1CVN: 1->6 arm extends to solvent (Loris 1994 Fig3); omega torsion stays mobile"
    },
    "triMan": {
        "n_HB": 9, "buried": ["K_EQ", "K_EQ", "K_C6", "K_EQ", "K_EQ"], "n_CHP": 2,
        "res_type": "Tyr", "n_linker": 2,
        "anchor": False, "obs_dG": -31.0, "confidence": "LOW",
        "note": "1CVN: Man1 full + Man2(1→3): 2 HB + CH-pi + Man3(1→6): minimal; Loris 1994"
    },
}

# ══════════════════════════════════════════════════════════════════════════
# WGA (Wheat Germ Agglutinin) — GlcNAc lectin
# Subsites B (primary), C, A, D per Bains 1992 ITC
# CH-π: Tyr73 at primary subsite B
# ══════════════════════════════════════════════════════════════════════════
WGA_CONTACTS = {
    "GlcNAc": {
        "n_HB": 4, "buried": ["K_NAC", "K_EQ"], "n_CHP": 1,
        "res_type": "Tyr", "n_linker": 0,
        "anchor": True, "obs_dG": -15.5, "confidence": "HIGH",
        "note": "2UVO subsite B: Tyr73 CH-pi; 4 HBs to GlcNAc; NHAc + C6 OH buried"
    },
    "(GlcNAc)2": {
        "n_HB": 7, "buried": ["K_NAC", "K_EQ", "K_EQ"], "n_CHP": 2,
        "res_type": "Tyr", "n_linker": 1,
        "anchor": False, "obs_dG": -21.3, "confidence": "HIGH",
        "note": "Subsite C: beta1->4 flips ring; NHAc solvent-exposed at C; +3 HBs, +1 Tyr, +1 EQ"
    },
    "(GlcNAc)3": {
        "n_HB": 8, "buried": ["K_NAC", "K_EQ", "K_EQ", "K_EQ"], "n_CHP": 3,
        "res_type": "Tyr", "n_linker": 2,
        "anchor": False, "obs_dG": -23.1, "confidence": "HIGH",
        "note": "Subsite A: +1 HB, +1 Tyr, +1 EQ buried"
    },
    "(GlcNAc)4": {
        "n_HB": 8, "buried": ["K_NAC", "K_EQ", "K_EQ", "K_EQ"], "n_CHP": 3,
        "res_type": "Tyr", "n_linker": 3,
        "anchor": False, "obs_dG": -23.4, "confidence": "HIGH",
        "note": "Subsite D: 4th unit adds only linker (no additional contacts; plateau in Bains ITC)"
    },
}

# ══════════════════════════════════════════════════════════════════════════
# PNA (Peanut Agglutinin) — Gal/GalNAc lectin
# CH-π: Trp132 stacks against galactose C3-H5 face
# PDB 2PEL; Swaminathan 1998
# ══════════════════════════════════════════════════════════════════════════
PNA_CONTACTS = {
    "Gal": {
        "n_HB": 3, "buried": ["K_AX", "K_EQ", "K_C6"], "n_CHP": 1,
        "res_type": "Trp", "n_linker": 0,
        "anchor": True, "obs_dG": -18.9, "confidence": "HIGH",
        "note": "2PEL: O3-Asp83, O4-Gly108, O6-His121; Trp132 CH-pi; C4 axial (Gal), C6 primary"
    },
    "GalNAc": {
        "n_HB": 2, "buried": ["K_NAC", "K_AX", "K_EQ"], "n_CHP": 1,
        "res_type": "Trp", "n_linker": 0,
        "anchor": False, "obs_dG": -20.1, "confidence": "MEDIUM",
        "note": "NHAc at C2: O4 HB retained, O6 HB lost; NHAc buried; Trp132 intact"
    },
}

# ══════════════════════════════════════════════════════════════════════════
# Galectin-3 (Gal3) — beta-galactoside lectin
# CH-π: Trp181 (key; Diehl/Kiessling 2024 JACS Au 4:3028 mutant panel)
# PDB 3GAL; Seetharaman 1998; Diehl 2024
# ══════════════════════════════════════════════════════════════════════════
GAL3_CONTACTS = {
    "Gal": {
        "n_HB": 4, "buried": ["K_AX", "K_EQ", "K_C6"], "n_CHP": 1,
        "res_type": "Trp", "n_linker": 0,
        "anchor": True, "obs_dG": -22.6, "confidence": "HIGH",
        "note": "3GAL: Arg144,His158,Asn160,Glu165; Trp181 CH-pi; C4 axial, C6 primary"
    },
    "LacNAc": {
        "n_HB": 8, "buried": ["K_AX", "K_EQ", "K_C6", "K_NAC", "K_EQ"], "n_CHP": 2,
        "res_type": "Trp", "n_linker": 1,
        "anchor": False, "obs_dG": -27.8, "confidence": "MEDIUM",
        "note": "Gal (primary) + GlcNAc (beta1->4); GlcNAc adds 3 HBs, NHAc buried, 1 Tyr CH-pi"
    },
}

# ══════════════════════════════════════════════════════════════════════════
# Davis synthetic receptor (GluHUT family, Eades/Tromans/Davis)
# CH-π: anthracene TEM platforms (2 platforms, ~3 contacts for equatorial-rich sugars)
# Anthracene ≈ Tyr/Phe in polarizability → use EPS_CH_PI_TYR
# Source: Eades et al. 2026 ChemRxiv; Davis 2012 Nat.Chem. 4:718
# DG0_DAVIS = -19.30 (locked v2.3, from Glc anchor)
# ══════════════════════════════════════════════════════════════════════════
DAVIS_CONTACTS = {
    "Glc": {
        "n_HB": 4, "buried": ["K_EQ", "K_EQ", "K_EQ", "K_EQ"], "n_CHP": 3,
        "res_type": "anthracene", "n_linker": 0,
        "anchor": True, "obs_dG": -24.4, "confidence": "HIGH",
        "note": "Eades 2026: 4 equatorial OHs in urea HB array; 3 CH-pi on beta-face (C1,C3,C5 axial Hs)"
    },
    "Gal": {
        "n_HB": 2, "buried": ["K_EQ", "K_AX", "K_EQ", "K_EQ"], "n_CHP": 1,
        "res_type": "anthracene", "n_linker": 0,
        "anchor": False, "obs_dG": -12.7, "confidence": "HIGH",
        "note": "C4-OH axial: disrupts beta-face alignment; 2 HBs lost; 2 CH-pi lost"
    },
    "Man": {
        "n_HB": 2, "buried": ["K_AX", "K_EQ", "K_EQ", "K_EQ"], "n_CHP": 1,
        "res_type": "anthracene", "n_linker": 0,
        "anchor": False, "obs_dG": -12.4, "confidence": "HIGH",
        "note": "C2-OH axial: symmetric to Gal case; near-identical prediction"
    },
    "2dGlc": {
        "n_HB": 0, "buried": ["K_EQ", "K_EQ", "K_EQ"], "n_CHP": 3,
        "res_type": "anthracene", "n_linker": 0,
        "anchor": False, "obs_dG": -16.9, "confidence": "HIGH",
        "note": "C2-OH absent: 2 urea-arm HBs lost; CH-pi beta-face intact (no axial disruption)"
    },
    "Fru": {
        "n_HB": 1, "buried": ["K_AX", "K_EQ", "K_EQ"], "n_CHP": 0,
        "res_type": "anthracene", "n_linker": 0,
        "anchor": False, "obs_dG": -9.9, "confidence": "LOW",
        "note": "Furanose: no beta-face; 1 HB only. Geometry uncertain."
    },
    "GlcNAc": {
        "n_HB": 0, "buried": ["K_NAC", "K_EQ", "K_EQ"], "n_CHP": 0,
        "res_type": "anthracene", "n_linker": 0,
        "anchor": False, "obs_dG": -6.0, "confidence": "LOW",
        "note": "NHAc steric clash prevents cavity entry. LOW confidence."
    },
}

# ── Registry ────────────────────────────────────────────────────────────

SCAFFOLD_CONTACTS = {
    "ConA":  CONA_CONTACTS,
    "WGA":   WGA_CONTACTS,
    "PNA":   PNA_CONTACTS,
    "Gal3":  GAL3_CONTACTS,
    "Davis": DAVIS_CONTACTS,
}

# Pre-anchored DG0 for Davis (v2.3 locked; others computed at runtime)
PREANCHORED_DG0 = {
    "Davis": -19.30,
}
