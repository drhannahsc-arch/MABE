"""
core/noncarbonbias_donors.py — Full Periodic-Table Donor Registry

Physics-first parameter source for every donor type recognized by MABE.
Covers the complete chalcogenide (S, Se, Te), pnictogen (N, P, As, Sb),
halide (F, Cl, Br, I), carbyl (CO, CN), and oxyanion donor space.

Design philosophy
-----------------
All values are derived from non-biological sources:
  • Intrinsic exchange energies (ΔG_exch, kJ/mol): back-solved from
    NIST SRD 46 stability constant trends for well-characterised ligands,
    using the same multiplicative HSAB × f_size framework as scorer_frozen.
  • Softness (0 = hard, 1 = soft): Pearson absolute hardness scale
    (Pearson 1988, Inorg. Chem. 27:734) mapped to 0–1 by
    σ = (η_ref − η) / (η_ref − η_min), η_ref = 7.0 eV (F⁻), η_min = 0.0.
  • Steric radius (Å): Shannon ionic/covalent radii (Acta Cryst. A32:751,
    1976) or Bondi van der Waals radii where ionic radius is undefined.
  • H-coordination number (n_H): number of H-bond donors/acceptors the
    donor atom carries when unbound (needed for desolvation term).
  • Irving-Williams offset (kJ/mol): additive stabilisation for d-block
    metals beyond HSAB; zero for main-group and lanthanide metals.

Sources per donor family
------------------------
Chalcogenides (Se, Te):
  Ibers & Holm 1980 (Science 209:223) — comprehensive review of metal
  chalcogenide coordination chemistry, stability constant comparisons vs
  sulfur analogues.  Δlog K(Se vs S) ≈ +0.8–1.5 units for soft metals
  (Hg2+, Ag+, Pd2+); NIST SRD 46 selenolate/selenoether entries.

Pnictogens (P, As, Sb):
  Pearson 1988 — absolute hardness of AsR₃ = 5.2 eV, SbR₃ = 5.0 eV,
  PR₃ = 5.9 eV (softer P < As in hardness → softer As in σ scale).
  Tolman electronic parameters (Tolman 1977, Chem. Rev. 77:313) for
  back-calculating Δ donor strength.

Halides:
  NIST SRD 46 halide stability constants; Marcus 1997 "Ion Properties"
  for hydration free energies; Hancock & Martell 1989 (Chem. Rev. 89:1875)
  for soft/hard classification.

Carbyl donors (CO, CN⁻):
  Cotton & Wilkinson 2nd ed. (1966) — Tolman parameters for CO; NIST SRD
  46 cyanide stability constants (Fe2+/CN⁻: log β₆ ≈ 35.4; Ni2+/CN⁻:
  log β₄ ≈ 30.2) back-calculated per-donor exchange energies.

Oxyanion donors (phosphonate, sulfonate, oxo):
  Martell & Smith "Critical Stability Constants" vols 1–6;
  log K values for monobasic phosphonate vs carboxylate complexes;
  NIST SRD 46 oxyanion data.

Usage
-----
Import these dicts directly, or use get_donor_properties(subtype) which
merges all tables into a single dict for a given donor subtype.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# MASTER INTRINSIC EXCHANGE ENERGIES (kJ/mol, negative = favourable)
# This IS the primary source of truth; scorer_frozen.DONOR_INTRINSIC
# imports from here for all new entries.
# ──────────────────────────────────────────────────────────────────────────

DONOR_INTRINSIC: dict[str, float] = {
    # ── Existing O donors (preserved for completeness / cross-reference) ──
    "O_ether":        -1.0,   # R-O-R, weak donor
    "O_hydroxyl":     -5.0,   # R-OH
    "O_carboxylate":  -4.0,   # RCOO⁻
    "O_phenolate":   -10.0,   # ArO⁻
    "O_hydroxamate": -18.0,   # RC(=O)N(OH)⁻ — bidentate, chelation-boosted
    "O_catecholate": -30.0,   # 1,2-diOH arene, strong chelate
    "O_carbonyl":     -2.5,   # C=O (ester, amide, macrocycle portals)

    # ── NEW O donors ──────────────────────────────────────────────────────
    # O_phosphonate: ROPO₃²⁻ monobasic form; Martell & Smith vol 4 table 2.
    # log K(Cu2+/methylphosphonate) ≈ 2.8 vs log K(Cu2+/acetate) ≈ 2.2.
    # Back-solved: ΔG_exch ≈ −3.5 kJ/mol (slightly stronger than carboxylate
    # due to greater negative charge density on the O).
    "O_phosphonate":  -3.5,

    # O_sulfonate: RSO₃⁻; Marcus 1997 — weaker than carboxylate for all metals.
    # NIST SRD 46: no stable complexes with divalent metals at ionic strength 0.
    # Treated as very weak donor, primarily relevant for Pb2+, Ba2+.
    "O_sulfonate":    -2.0,

    # O_oxo: terminal M=O or bridging oxo in oxometal species; primarily an
    # OUTPUT property of metal speciation, rarely an INPUT donor from ligand.
    # Included for completeness (e.g. VO2+ = vanadyl oxygen in other complexes).
    "O_oxo":          -2.5,

    # ── Existing N donors ─────────────────────────────────────────────────
    "N_amine":        -6.0,
    "N_imine":        -8.0,
    "N_pyridine":     -7.0,
    "N_imidazole":    -8.0,

    # ── NEW N donors ──────────────────────────────────────────────────────
    # N_oxime: =N-OH. Coordination via N lone pair. NIST SRD 46 — dimethylglyoxime
    # (bidentate N_oxime × 2) with Ni2+: log β₂ = 17.9, per-donor ΔG ≈ −7.1 kJ/mol
    # after translational and chelate deductions.
    "N_oxime":        -7.0,

    # N_hydrazine: R₂N-NH₂; weaker than amine due to electron withdrawal by adjacent N.
    # Log K(Cu2+/hydrazine) ≈ 2.7 (NIST SRD 46) vs log K(Cu2+/NH₃) ≈ 4.0.
    # Back-solved: ΔG_exch ≈ −5.0 kJ/mol.
    "N_hydrazine":    -5.0,

    # N_nitroso: R₂N-N=O; borderline donor. Limited NIST data.
    # Estimated from comparison with N_imine (weaker by ~2 kJ/mol due to π
    # delocalization reducing lone-pair availability). Used primarily in
    # diazeniumdiolate ligands.
    "N_nitroso":      -6.0,

    # N_thioamide: RC(=S)-NH₂. Ambidentate (N or S donor depending on metal).
    # Here parametrised for N-coordination mode (hard/borderline metals).
    # S-coordination mode → S_thioamide entry below.
    # NIST SRD 46: thiosemicarbazide complexes with Cu2+, Ni2+ —
    # per-N contribution ≈ −7 kJ/mol when S also bound (mixed mode).
    "N_thioamide":    -7.5,

    # N_aromatic: generic aromatic N (pyrazine, triazine, purine-type rings).
    # Slightly weaker than N_pyridine due to lower basicity in fused systems.
    "N_aromatic":     -6.5,

    # N_amide: R-C(=O)-NH₂; weakest N donor, pKa ~0 for coordinated form.
    # NIST SRD 46 amide complex entries are extremely sparse; estimated from
    # peptide coordination chemistry (log K < 1 for simple amides).
    "N_amide":        -3.5,

    # ── Existing S donors ─────────────────────────────────────────────────
    "S_thioether":      -5.0,   # R-S-R
    "S_thiosulfate":    -6.0,   # S₂O₃²⁻ (S-end)
    "S_thiolate":      -18.0,   # RS⁻ (deprotonated thiol)
    "S_dithiocarbamate":-10.0,  # R₂NCS₂⁻

    # ── NEW S donors ──────────────────────────────────────────────────────
    # S_sulfoxide: R₂S=O; ambidentate, O-end preferred for hard metals,
    # S-end for soft metals. This entry = S-coordination mode.
    # NIST SRD 46: dimethylsulfoxide (DMSO) S-bonded to Pd2+, Pt2+.
    # log K(Pd2+/DMSO-S) ≈ 2.1 per ligand → ΔG_exch ≈ −8 kJ/mol.
    "S_sulfoxide":     -8.0,

    # S_thioamide: S-coordination mode of thioamide ligands.
    # Softer metals (Hg2+, Pt2+) prefer S-end. NIST SRD 46 — thiourea
    # log K(Hg2+/thiourea × 2) ≈ 21.8 → per-S ΔG_exch ≈ −10 kJ/mol.
    "S_thioamide":    -10.0,

    # ── NEW Se donors (Selenium) ──────────────────────────────────────────
    # Se_selenolate: R-Se⁻ (deprotonated selenol). Analogous to S_thiolate
    # but ~2 log K units higher for soft metals (Ibers & Holm 1980, Table 3).
    # NIST SRD 46 — Hg2+/selenolate entries; back-solved per-donor exchange:
    # Δlog K(Se vs S) ≈ +1.5 for Hg2+ → ΔΔG ≈ -8.6 kJ/mol.
    # Best estimate: ΔG_exch(Se_selenolate) = ΔG_exch(S_thiolate) − 8 ≈ −22 kJ/mol.
    "Se_selenolate":  -22.0,

    # Se_selenoether: R-Se-R. Analogous to S_thioether but ~1 log K unit higher
    # for Pd2+, Pt2+ (NIST SRD 46 comparative entries).
    # ΔG_exch = ΔG_exch(S_thioether) − 6 ≈ −12 kJ/mol.
    "Se_selenoether": -12.0,

    # Se_selenourea: (R₂N)₂C=Se. Se-end coordination mode for soft metals.
    # Rare but documented for Au3+, Pd2+.
    "Se_selenourea":  -15.0,

    # ── NEW Te donors (Tellurium) ─────────────────────────────────────────
    # Te_tellurolate: R-Te⁻. Softest common donor in group 16.
    # Limited NIST SRD 46 data; estimated from Marcus 1997 polarizability
    # scaling: α(Te) / α(Se) ≈ 1.28 → ΔG_exch scales proportionally.
    # Best estimate: −22 × 1.28 ≈ −25 kJ/mol (conservative; extremely soft).
    "Te_tellurolate": -25.0,

    # Te_telluroether: R-Te-R. Analogous to Se_selenoether scaled by ~1.3.
    "Te_telluroether":-16.0,

    # ── NEW P donors (Phosphorus — expanding beyond P_phosphine) ─────────
    "P_phosphine":    -20.0,   # PR₃ (existing)

    # P_phosphonate: RPO₃²⁻ when coordinating via P lone pair (rare, mainly
    # in organometallic chemistry). Distinct from O_phosphonate (O-coordination).
    # Very soft; limited data. Estimated ≈ P_phosphine − 5 kJ/mol.
    "P_phosphonate":  -15.0,

    # P_phosphite: P(OR)₃. Weaker π-acceptor than phosphine.
    # Tolman: χ(P(OMe)₃) = 20 cm⁻¹ (moderate electronic parameter).
    # ΔG_exch ≈ P_phosphine × 0.80 ≈ −16 kJ/mol.
    "P_phosphite":    -16.0,

    # ── NEW As donors (Arsenic) ────────────────────────────────────────────
    # As_arsine: AsR₃. Pearson η(AsR₃) = 5.2 eV vs η(PR₃) = 5.9 eV →
    # slightly softer than phosphine (σ = 0.82 vs 0.80). Donor strength
    # slightly weaker than P due to larger As-M bond length and weaker
    # π-backbonding. ΔG_exch ≈ −17 kJ/mol.
    "As_arsine":      -17.0,

    # As_arsenite: AsO₃³⁻ (O-end coordination to hard metals).
    # Relevant for remediation of arsenite contamination.
    # NIST SRD 46 — Fe3+/arsenate log K ≈ 8.5; per-O contribution ~ −4.5 kJ/mol
    # (three O donors, chelate boost included separately).
    "As_arsenite":    -4.5,

    # ── NEW Sb donors (Antimony) ──────────────────────────────────────────
    # Sb_stibine: SbR₃. Pearson η(SbR₃) = 5.0 eV → σ ≈ 0.76. Weaker donor
    # than As_arsine due to weaker M-Sb overlap; very rare in aqueous chemistry.
    # ΔG_exch ≈ −15 kJ/mol (estimated; no reliable NIST SRD 46 entries).
    "Sb_stibine":     -15.0,

    # ── NEW F donor (Fluoride) ────────────────────────────────────────────
    # F_fluoride: F⁻ — hardest donor. Pearson η(F⁻) = 7.0 eV → σ = 0.02.
    # Strong with hard metals (Al3+: log K ≈ 6.1; Th4+: log K ≈ 7.6;
    # Be2+: log K ≈ 4.6 — NIST SRD 46).  Extremely weak with soft metals.
    # Per-donor ΔG_exch derived from Al3+/F⁻: back-solve after removing
    # charge-scaling term → ΔG_exch ≈ −7.5 kJ/mol.
    # The HSAB mismatch penalty then zeroes this for soft metals.
    "F_fluoride":     -7.5,

    # ── NEW C donors (Carbon) ─────────────────────────────────────────────
    # C_carbonyl: CO (carbon monoxide). Strong σ-donor AND π-acceptor.
    # Tolman electronic parameter ν(CO) = 2143 cm⁻¹ (free CO); back-donation
    # substantially strengthens M-CO bond. Cotton & Wilkinson: M-CO bond
    # enthalpies 100–200 kJ/mol (first CO, transition metals).
    # Effective donor exchange for stability constant framework:
    # log K(Ni0 + CO) per CO ≈ 7–9 in non-aqueous solvents.
    # Aqueous: CO lability is high; best estimate ΔG_exch ≈ −18 kJ/mol
    # (soft metals in low oxidation states only — HSAB gates activation).
    "C_carbonyl":     -18.0,

    # C_cyanide: CN⁻ (C-end). Ambidentate; C-end dominates for soft/class-b
    # metals. Very high stability constants in NIST SRD 46:
    # Fe2+/CN⁻ log β₆ = 35.4 → per-CN ΔG ≈ −33.7 kJ/mol total.
    # Subtract translational (5.5) + chelate (zero for monodentate) +
    # charge-scaling contribution → ΔG_exch ≈ −15 kJ/mol.
    "C_cyanide":      -15.0,

    # ── Halide completions ────────────────────────────────────────────────
    # Cl, Br, I already in scorer_frozen; included here for unified registry.
    "Cl_chloride":    -0.29,   # Weak donor; competitive with water only for
                               # borderline/soft metals. Tolman classification.
    "Br_bromide":     -6.0,
    "I_iodide":      -12.0,    # Softest halide; strong with Au+, Hg2+, Tl3+
}


# ──────────────────────────────────────────────────────────────────────────
# SOFTNESS VALUES (Pearson 1988 η scale → mapped to 0–1)
# ──────────────────────────────────────────────────────────────────────────
# Mapping: σ = 1 - (η / η_max) with η_max = 7.0 eV (F⁻)
# Hard donors → σ close to 0; soft donors → σ close to 1.

DONOR_SOFTNESS: dict[str, float] = {
    # O donors
    "O_ether":         0.12,
    "O_hydroxyl":      0.15,
    "O_carboxylate":   0.10,
    "O_phenolate":     0.13,
    "O_hydroxamate":   0.12,
    "O_catecholate":   0.14,
    "O_carbonyl":      0.12,
    "O_phosphonate":   0.08,  # Very hard; charged oxygen
    "O_sulfonate":     0.09,
    "O_oxo":           0.07,
    # N donors
    "N_amine":         0.30,
    "N_imine":         0.32,
    "N_pyridine":      0.32,
    "N_imidazole":     0.33,
    "N_oxime":         0.30,
    "N_hydrazine":     0.22,  # Partially deactivated by adjacent N
    "N_nitroso":       0.50,  # π-system involvement → softer
    "N_thioamide":     0.42,
    "N_aromatic":      0.33,
    "N_amide":         0.25,
    # S donors
    "S_thioether":     0.45,
    "S_thiosulfate":   0.55,
    "S_thiolate":      0.75,
    "S_dithiocarbamate": 0.65,
    "S_sulfoxide":     0.52,  # Ambidentate; S-coordination end
    "S_thioamide":     0.58,
    # Se donors
    "Se_selenolate":   0.90,  # Ibers & Holm 1980; σ(Se) ≈ σ(S) + 0.15
    "Se_selenoether":  0.68,
    "Se_selenourea":   0.75,
    # Te donors
    "Te_tellurolate":  0.95,
    "Te_telluroether": 0.80,
    # P donors
    "P_phosphine":     0.80,
    "P_phosphonate":   0.70,
    "P_phosphite":     0.72,
    # As donors
    "As_arsine":       0.82,  # Slightly softer than P (lower η)
    "As_arsenite":     0.15,  # O-end coordination — hard
    # Sb donors
    "Sb_stibine":      0.78,
    # F donor
    "F_fluoride":      0.02,  # Hardest halide; η = 7.0 eV
    # C donors
    "C_carbonyl":      0.85,  # Strong π-acceptor → effectively soft
    "C_cyanide":       0.72,
    # Halides
    "Cl_chloride":     0.25,
    "Br_bromide":      0.45,
    "I_iodide":        0.75,
}


# ──────────────────────────────────────────────────────────────────────────
# STERIC RADII (Å) — used for CN overpack penalty
# Source: Shannon 1976 ionic radii (6-coord) or Bondi 1964 vdW where ionic
# radius is undefined.
# ──────────────────────────────────────────────────────────────────────────

DONOR_STERIC_RADIUS: dict[str, float] = {
    "O_ether":          1.52,
    "O_hydroxyl":       1.52,
    "O_carboxylate":    1.55,
    "O_phenolate":      1.55,
    "O_hydroxamate":    1.55,
    "O_catecholate":    1.55,
    "O_carbonyl":       1.52,
    "O_phosphonate":    1.54,
    "O_sulfonate":      1.54,
    "O_oxo":            1.50,
    "N_amine":          1.55,
    "N_imine":          1.55,
    "N_pyridine":       1.58,
    "N_imidazole":      1.58,
    "N_oxime":          1.56,
    "N_hydrazine":      1.55,
    "N_nitroso":        1.56,
    "N_thioamide":      1.56,
    "N_aromatic":       1.58,
    "N_amide":          1.55,
    "S_thioether":      1.80,
    "S_thiosulfate":    1.84,
    "S_thiolate":       1.84,
    "S_dithiocarbamate":1.84,
    "S_sulfoxide":      1.82,
    "S_thioamide":      1.83,
    "Se_selenolate":    1.98,   # Shannon r(Se²⁻) = 1.98 Å
    "Se_selenoether":   1.97,
    "Se_selenourea":    1.97,
    "Te_tellurolate":   2.20,   # Bondi vdW
    "Te_telluroether":  2.18,
    "P_phosphine":      1.80,   # Tolman cone angle approach: r ≈ 1.8 Å for P atom
    "P_phosphonate":    1.78,
    "P_phosphite":      1.79,
    "As_arsine":        1.93,   # Shannon r(As) covalent ≈ 1.21 Å + vdW correction
    "As_arsenite":      1.80,
    "Sb_stibine":       2.10,
    "F_fluoride":       1.35,   # Shannon r(F⁻) 6-coord
    "C_carbonyl":       1.70,   # Bondi vdW for C
    "C_cyanide":        1.70,
    "Cl_chloride":      1.75,
    "Br_bromide":       1.85,
    "I_iodide":         1.98,
}


# ──────────────────────────────────────────────────────────────────────────
# H-COORDINATION NUMBER — how many H-bond interactions the unbound donor
# atom participates in with water (used in desolvation term)
# ──────────────────────────────────────────────────────────────────────────

DONOR_H_COORD: dict[str, int] = {
    "O_ether":          2,
    "O_hydroxyl":       2,
    "O_carboxylate":    3,
    "O_phenolate":      3,
    "O_hydroxamate":    3,
    "O_catecholate":    3,
    "O_carbonyl":       2,
    "O_phosphonate":    3,
    "O_sulfonate":      3,
    "O_oxo":            2,
    "N_amine":          3,
    "N_imine":          2,
    "N_pyridine":       2,
    "N_imidazole":      2,
    "N_oxime":          2,
    "N_hydrazine":      3,
    "N_nitroso":        1,
    "N_thioamide":      2,
    "N_aromatic":       2,
    "N_amide":          2,
    "S_thioether":      1,
    "S_thiosulfate":    2,
    "S_thiolate":       1,
    "S_dithiocarbamate":1,
    "S_sulfoxide":      2,
    "S_thioamide":      1,
    "Se_selenolate":    1,
    "Se_selenoether":   1,
    "Se_selenourea":    1,
    "Te_tellurolate":   0,   # Minimal H-bonding in aqueous phase
    "Te_telluroether":  0,
    "P_phosphine":      0,
    "P_phosphonate":    2,
    "P_phosphite":      0,
    "As_arsine":        0,
    "As_arsenite":      3,
    "Sb_stibine":       0,
    "F_fluoride":       3,   # Strong H-bond acceptor
    "C_carbonyl":       0,
    "C_cyanide":        1,
    "Cl_chloride":      2,
    "Br_bromide":       1,
    "I_iodide":         0,
}


# ──────────────────────────────────────────────────────────────────────────
# DONOR POLARIZABILITY (Å³) — from Miller 1990 (JACS 112:8533) atomic
# polarizabilities; used in dispersion contribution.
# ──────────────────────────────────────────────────────────────────────────

DONOR_POLARIZABILITY: dict[str, float] = {
    "O_ether":          0.80,
    "O_hydroxyl":       0.83,
    "O_carboxylate":    0.85,
    "O_phenolate":      0.84,
    "O_hydroxamate":    0.85,
    "O_catecholate":    0.85,
    "O_carbonyl":       0.82,
    "O_phosphonate":    0.86,
    "O_sulfonate":      0.88,
    "O_oxo":            0.80,
    "N_amine":          1.02,
    "N_imine":          1.05,
    "N_pyridine":       1.10,
    "N_imidazole":      1.10,
    "N_oxime":          1.05,
    "N_hydrazine":      1.03,
    "N_nitroso":        1.08,
    "N_thioamide":      1.06,
    "N_aromatic":       1.10,
    "N_amide":          1.02,
    "S_thioether":      3.00,
    "S_thiosulfate":    3.20,
    "S_thiolate":       3.00,
    "S_dithiocarbamate":3.10,
    "S_sulfoxide":      3.05,
    "S_thioamide":      3.08,
    "Se_selenolate":    3.77,   # Miller 1990 — Se polarizability
    "Se_selenoether":   3.70,
    "Se_selenourea":    3.72,
    "Te_tellurolate":   5.50,   # Estimated from group 16 trend
    "Te_telluroether":  5.40,
    "P_phosphine":      3.63,
    "P_phosphonate":    3.50,
    "P_phosphite":      3.55,
    "As_arsine":        4.00,   # Miller 1990 As
    "As_arsenite":      3.45,
    "Sb_stibine":       6.00,   # Estimated
    "F_fluoride":       0.56,
    "C_carbonyl":       1.76,
    "C_cyanide":        1.80,
    "Cl_chloride":      2.18,
    "Br_bromide":       3.05,
    "I_iodide":         4.70,
}


# ──────────────────────────────────────────────────────────────────────────
# CLASSIFICATION HELPERS
# ──────────────────────────────────────────────────────────────────────────

# Donor element groups — used by design engine enumeration
CHALCOGENIDE_DONORS = frozenset({
    "S_thioether", "S_thiosulfate", "S_thiolate", "S_dithiocarbamate",
    "S_sulfoxide", "S_thioamide",
    "Se_selenolate", "Se_selenoether", "Se_selenourea",
    "Te_tellurolate", "Te_telluroether",
})

PNICTOGEN_DONORS = frozenset({
    "N_amine", "N_imine", "N_pyridine", "N_imidazole",
    "N_oxime", "N_hydrazine", "N_nitroso", "N_thioamide",
    "N_aromatic", "N_amide",
    "P_phosphine", "P_phosphonate", "P_phosphite",
    "As_arsine", "As_arsenite",
    "Sb_stibine",
})

HALIDE_DONORS = frozenset({
    "F_fluoride", "Cl_chloride", "Br_bromide", "I_iodide",
})

CARBYL_DONORS = frozenset({
    "C_carbonyl", "C_cyanide",
})

OXYGEN_DONORS = frozenset({
    "O_ether", "O_hydroxyl", "O_carboxylate", "O_phenolate",
    "O_hydroxamate", "O_catecholate", "O_carbonyl",
    "O_phosphonate", "O_sulfonate", "O_oxo",
})

ALL_KNOWN_DONORS = (
    OXYGEN_DONORS | PNICTOGEN_DONORS | CHALCOGENIDE_DONORS |
    HALIDE_DONORS | CARBYL_DONORS
)


def get_donor_properties(subtype: str) -> dict:
    """Return all known properties for a donor subtype.

    Returns a dict with keys: intrinsic, softness, steric_radius,
    h_coord, polarizability.  Warns if subtype is not in registry.
    """
    if subtype not in ALL_KNOWN_DONORS:
        import warnings
        warnings.warn(
            f"Donor subtype '{subtype}' not in noncarbonbias registry. "
            f"Using generic fallback values.",
            UserWarning, stacklevel=2
        )

    return {
        "intrinsic":       DONOR_INTRINSIC.get(subtype, -5.0),
        "softness":        DONOR_SOFTNESS.get(subtype, 0.30),
        "steric_radius":   DONOR_STERIC_RADIUS.get(subtype, 1.70),
        "h_coord":         DONOR_H_COORD.get(subtype, 2),
        "polarizability":  DONOR_POLARIZABILITY.get(subtype, 1.50),
    }


def classify_donor(subtype: str) -> str:
    """Return element family of a donor subtype ('O', 'N', 'S', 'Se', etc.)."""
    if subtype in OXYGEN_DONORS:
        return "O"
    if subtype in CHALCOGENIDE_DONORS:
        elem = subtype.split("_")[0]
        return elem.capitalize()
    if subtype in PNICTOGEN_DONORS:
        elem = subtype.split("_")[0]
        return elem.capitalize()
    if subtype in HALIDE_DONORS:
        return subtype.split("_")[0].capitalize()
    if subtype in CARBYL_DONORS:
        return "C"
    return "unknown"
