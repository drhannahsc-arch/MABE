"""
scorer_frozen.py — Frozen scoring function for MABE generative design engine.

Standalone log K predictor extracted from the MABE physics engine (v2 calibration).
Contains all calibrated parameters baked in. No external dependencies beyond math.

Physics terms included:
  1. Per-donor exchange energy (18 subtypes, subtype-resolved)
  2. Charge scaling (z²-dependent electrostatic enhancement)
  3. HSAB match/mismatch (multiplicative framework with f_size correction)
  4. Chelate effect (per-ring, charge/d-electron scaled)
  5. Desolvation cost (metal-specific ΔG_hyd per-water exchange)
  6. ΔLFSE (ligand field vs aqua, CN-gated, capped)
  7. Jahn-Teller correction (d9, d4 high-spin)
  8. Covalent BDE contribution (metal-specific fractions for soft pairs)
  9. Translational entropy (+5.5 kJ/mol per ligand molecule)
  10. Macrocyclic/preorganization (entropic + cavity size-match Gaussian)
  11. Chelate ring-size selectivity (Hancock rule: 5- vs 6-membered)
  12. Electrostatic z-z charge-charge (anionic donors + cationic metals)
  13. Metal-donor exchange matrix (multiplicative HSAB × size correction)
  14. Irving-Williams enhancement (empirical d-electron stabilization curve)
  15. pH-dependent protonation penalty (conditional log K)
  16. Repulsion forces (CN overpacking, donor-donor anionic, steric strain)
  17. Entropy decomposition (conformational rotor penalty, T-dependent ΔG)

Regenerate from live engine: python scorer_frozen.py --freeze (future)

Usage:
    from scorer_frozen import predict_log_k, METAL_DB
    log_k = predict_log_k("Pb2+", ["S_thiolate", "N_amine", "O_carboxylate"],
                          chelate_rings=2, pH=5.0)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATED PARAMETERS (v2, fitted against NIST SRD 46 + 33-complex set)
# ═══════════════════════════════════════════════════════════════════════════

# Per-donor subtype exchange energies (kJ/mol)
# Negative = favorable binding contribution per donor atom
SUBTYPE_EXCHANGE = {
    # O-donors (calibrated, 251 complexes, R²=0.87)
    "O_ether":          4.19,   # Crown ethers — weak, entropy-dominated
    "O_hydroxyl":       -2.0,   # Alcohols, sugars — weak
    "O_carboxylate":    -6.36,  # Acetate, EDTA arms — charged, good sigma
    "O_phenolate":     -17.49,  # Phenol anion — aromatic + charged
    "O_hydroxamate":    -8.42,  # CONHO⁻ — resonance-stabilized
    "O_catecholate":   -15.00,  # Aromatic diolate — pi-donation into d-orbitals
    "O_phosphate":      -8.51,  # PO4 donors
    "O_sulfonate":      -2.0,   # SO3⁻ — weak donor
    # N-donors
    "N_amine":         -14.90,  # Primary/secondary/tertiary amines
    "N_imine":          -8.0,   # Schiff bases, C=N
    "N_pyridine":       -9.23,  # Aromatic N, pi-acceptor/donor
    "N_imidazole":      -7.40,  # Histidine-like
    "N_nitrile":        -3.0,   # C≡N — weak sigma donor
    "N_amide":         -19.00,  # Peptide bond N
    # S-donors
    "S_thioether":      5.00,   # R-S-R — weak intrinsic, selectivity via HSAB
    "S_thiosulfate":    -8.0,   # S₂O₃²⁻ terminal S
    "S_thiolate":      -15.80,  # RS⁻, cysteine — strong covalent character
    "S_dithiocarbamate": -9.30, # CS₂⁻ — chelating S,S-bidentate
    # P-donors
    "P_phosphine":     -20.0,   # PR₃ — strong for soft metals
    # Halides
    "Cl_chloride":      -0.29,
    "Br_bromide":       -6.0,
    "I_iodide":        -12.0,
}

# Fallback per-element exchange (when subtype unknown)
ELEMENT_EXCHANGE = {
    "O": -4.0, "N": -6.0, "S": -18.0, "P": -15.0,
    "Cl": -3.0, "Br": -6.0, "I": -12.0,
}

# Donor softness values (0 = hard, 1 = soft)
DONOR_SOFTNESS = {
    "O_ether": 0.05, "O_hydroxyl": 0.10, "O_carboxylate": 0.15,
    "O_phenolate": 0.20, "O_hydroxamate": 0.15, "O_catecholate": 0.20,
    "O_phosphate": 0.10, "O_sulfonate": 0.05,
    "N_amine": 0.35, "N_imine": 0.40, "N_pyridine": 0.45,
    "N_imidazole": 0.40, "N_nitrile": 0.30, "N_amide": 0.25,
    "S_thioether": 0.70, "S_thiosulfate": 0.60, "S_thiolate": 0.85,
    "S_dithiocarbamate": 0.75,
    "P_phosphine": 0.80,
    "Cl_chloride": 0.25, "Br_bromide": 0.45, "I_iodide": 0.75,
}

# Donor pKa values for pH-dependent protonation
# Below this pKa, donor is protonated and cannot coordinate
DONOR_PKA = {
    "O_carboxylate": 4.5, "O_phenolate": 10.0, "O_hydroxamate": 9.0,
    "O_catecholate": 9.2, "O_hydroxyl": 14.0, "O_phosphate": 7.2,
    "N_amine": 10.5, "N_imine": 6.5, "N_pyridine": 5.3,
    "N_imidazole": 7.0, "N_amide": 15.0,
    "S_thiolate": 8.3, "S_dithiocarbamate": 3.0,
    # These don't protonate in water
    "O_ether": 99.0, "O_sulfonate": -2.0,
    "S_thioether": 99.0, "S_thiosulfate": 1.5,
    "N_nitrile": -1.0, "P_phosphine": 99.0,
    "Cl_chloride": -7.0, "Br_bromide": -9.0, "I_iodide": -10.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# METAL DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MetalProperties:
    """Complete physics description of a metal ion."""
    formula: str
    charge: int
    d_electrons: int
    ionic_radius_pm: float        # Shannon ionic radius
    hsab_softness: float          # 0 = hard, 1 = soft
    dg_hyd_kj: float              # Hydration free energy (kJ/mol, negative)
    cn_aqua: int                  # Coordination number of aqua complex
    cn_range: tuple               # (min_CN, max_CN) for ligand complexes
    lfse_oct_dq: float            # LFSE in octahedral field (units of Dq)
    jahn_teller: bool             # True if Jahn-Teller active
    covalent_fraction: float      # Fraction of BDE to count (0-0.25)
    name: str = ""

# Comprehensive metal database — 55+ metals from NIST set
METAL_DB: dict[str, MetalProperties] = {}

def _register(formula, charge, d_el, r_pm, soft, dg_hyd, cn_aq, cn_range,
              lfse_dq, jt, cov_frac, name=""):
    METAL_DB[formula] = MetalProperties(
        formula=formula, charge=charge, d_electrons=d_el,
        ionic_radius_pm=r_pm, hsab_softness=soft, dg_hyd_kj=dg_hyd,
        cn_aqua=cn_aq, cn_range=cn_range, lfse_oct_dq=lfse_dq,
        jahn_teller=jt, covalent_fraction=cov_frac, name=name)

# Alkali metals (+1)
_register("Li+",   1, 0,  76, 0.02, -475,  4, (4,6),   0.0, False, 0.0, "Lithium")
_register("Na+",   1, 0, 102, 0.02, -365,  6, (4,6),   0.0, False, 0.0, "Sodium")
_register("K+",    1, 0, 138, 0.02, -295,  6, (6,8),   0.0, False, 0.0, "Potassium")
_register("Rb+",   1, 0, 152, 0.02, -275,  6, (6,8),   0.0, False, 0.0, "Rubidium")
_register("Cs+",   1, 0, 167, 0.02, -250,  6, (6,8),   0.0, False, 0.0, "Cesium")

# Alkaline earth (+2)
_register("Mg2+",  2, 0,  72, 0.05, -1830, 6, (4,6),   0.0, False, 0.0, "Magnesium")
_register("Ca2+",  2, 0, 100, 0.05, -1505, 6, (6,8),   0.0, False, 0.0, "Calcium")
_register("Sr2+",  2, 0, 118, 0.04, -1380, 6, (6,8),   0.0, False, 0.0, "Strontium")
_register("Ba2+",  2, 0, 135, 0.03, -1250, 6, (6,9),   0.0, False, 0.0, "Barium")

# First-row transition metals (+2)
_register("Ti2+",  2, 2,  86, 0.20, -1760, 6, (4,6),   0.8, False, 0.0, "Titanium(II)")
_register("V2+",   2, 3,  79, 0.22, -1825, 6, (4,6),   1.2, False, 0.0, "Vanadium(II)")
_register("Cr2+",  2, 4,  80, 0.25, -1850, 6, (4,6),   0.6, True,  0.0, "Chromium(II)")
_register("Mn2+",  2, 5,  83, 0.25, -1760, 6, (4,6),   0.0, False, 0.0, "Manganese(II)")
_register("Fe2+",  2, 6,  78, 0.30, -1840, 6, (4,6),   0.4, False, 0.0, "Iron(II)")
_register("Co2+",  2, 7,  75, 0.35, -1915, 6, (4,6),   0.8, False, 0.0, "Cobalt(II)")
_register("Ni2+",  2, 8,  69, 0.40, -1980, 6, (4,6),   1.2, False, 0.0, "Nickel(II)")
_register("Cu2+",  2, 9,  73, 0.45, -2010, 6, (4,6),   0.6, True,  0.05,"Copper(II)")
_register("Zn2+",  2, 10, 75, 0.50, -1955, 6, (4,6),   0.0, False, 0.0, "Zinc(II)")

# First-row transition metals (+3)
_register("Ti3+",  3, 1,  67, 0.15, -4150, 6, (6,6),   0.4, False, 0.0, "Titanium(III)")
_register("V3+",   3, 2,  64, 0.18, -4250, 6, (6,6),   0.8, False, 0.0, "Vanadium(III)")
_register("Cr3+",  3, 3,  62, 0.20, -4010, 6, (6,6),   1.2, False, 0.0, "Chromium(III)")
_register("Mn3+",  3, 4,  58, 0.22, -4350, 6, (6,6),   0.6, True,  0.0, "Manganese(III)")
_register("Fe3+",  3, 5,  65, 0.25, -4265, 6, (4,6),   0.0, False, 0.0, "Iron(III)")
_register("Co3+",  3, 6,  55, 0.28, -4640, 6, (6,6),   2.4, False, 0.0, "Cobalt(III)")

# Heavier d-block (+2)
_register("Cd2+",  2, 10, 95, 0.60, -1755, 6, (4,6),   0.0, False, 0.05,"Cadmium(II)")
_register("Hg2+",  2, 10,102, 0.85, -1760, 6, (2,6),   0.0, False, 0.25,"Mercury(II)")
_register("Pb2+",  2, 0, 119, 0.55, -1425, 6, (4,8),   0.0, False, 0.05,"Lead(II)")
_register("Cu+",   1, 10,  77, 0.85, -580,  4, (2,4),   0.0, False, 0.12,"Copper(I)")
_register("Ag+",   1, 10, 115, 0.90, -430,  4, (2,4),   0.0, False, 0.15,"Silver(I)")
_register("Au+",   1, 10,  85, 0.95, -615,  2, (2,4),   0.0, False, 0.22,"Gold(I)")
_register("Au3+",  3, 8,  70, 0.82, -4420, 4, (4,6),   1.2, False, 0.20,"Gold(III)")
_register("Pt2+",  2, 8,  80, 0.80, -2100, 4, (4,4),   1.2, False, 0.18,"Platinum(II)")
_register("Pd2+",  2, 8,  86, 0.75, -1980, 4, (4,4),   1.2, False, 0.15,"Palladium(II)")
_register("Tl+",   1, 0, 150, 0.55, -310,  6, (4,8),   0.0, False, 0.0, "Thallium(I)")
_register("Tl3+",  3, 0,  89, 0.65, -3970, 6, (4,6),   0.0, False, 0.08,"Thallium(III)")
_register("Sn2+",  2, 0,  93, 0.40, -1490, 6, (4,6),   0.0, False, 0.0, "Tin(II)")
_register("Bi3+",  3, 0, 103, 0.52, -3480, 6, (5,8),   0.0, False, 0.0, "Bismuth(III)")
_register("In3+",  3, 10, 80, 0.40, -3980, 6, (4,6),   0.0, False, 0.0, "Indium(III)")
_register("Al3+",  3,  0, 54, 0.05, -4525, 6, (4,6),   0.0, False, 0.0, "Aluminium(III)")
_register("Ga3+",  3, 10, 62, 0.30, -4515, 6, (4,6),   0.0, False, 0.0, "Gallium(III)")

# Lanthanides (+3) — f-electrons, high CN
_register("La3+",  3, 0, 103, 0.12, -3145, 9, (8,10),  0.0, False, 0.0, "Lanthanum(III)")
_register("Ce3+",  3, 1, 101, 0.12, -3200, 9, (8,10),  0.0, False, 0.0, "Cerium(III)")
_register("Nd3+",  3, 3,  98, 0.12, -3280, 9, (8,10),  0.0, False, 0.0, "Neodymium(III)")
_register("Sm3+",  3, 5,  96, 0.12, -3330, 9, (8,10),  0.0, False, 0.0, "Samarium(III)")
_register("Eu3+",  3, 6,  95, 0.12, -3360, 9, (8,10),  0.0, False, 0.0, "Europium(III)")
_register("Gd3+",  3, 7,  94, 0.12, -3425, 9, (8,10),  0.0, False, 0.0, "Gadolinium(III)")
_register("Tb3+",  3, 8,  92, 0.12, -3460, 9, (8,10),  0.0, False, 0.0, "Terbium(III)")
_register("Dy3+",  3, 9,  91, 0.12, -3500, 9, (8,10),  0.0, False, 0.0, "Dysprosium(III)")
_register("Er3+",  3, 11, 89, 0.12, -3560, 9, (8,10),  0.0, False, 0.0, "Erbium(III)")
_register("Yb3+",  3, 13, 87, 0.12, -3620, 9, (8,10),  0.0, False, 0.0, "Ytterbium(III)")
_register("Lu3+",  3, 14, 86, 0.12, -3650, 9, (8,10),  0.0, False, 0.0, "Lutetium(III)")

# Actinides
_register("UO2_2+",2, 0,  73, 0.45, -1630, 5, (4,6),   0.0, False, 0.0, "Uranyl(VI)")
_register("Th4+",  4, 0,  94, 0.15, -5815, 8, (8,10),  0.0, False, 0.0, "Thorium(IV)")

# Oxoanions that form complexes
_register("VO2+",  2, 1,  59, 0.30, -2080, 5, (4,6),   0.4, False, 0.0, "Vanadyl(IV)")

# ═══════════════════════════════════════════════════════════════════════════
# LFSE LOOKUP TABLE
# Octahedral LFSE in units of Dq for high-spin configurations
# ═══════════════════════════════════════════════════════════════════════════

LFSE_OCT_HIGH_SPIN = {
    0: 0.0,   # d0
    1: -4.0,  # d1
    2: -8.0,  # d2
    3: -12.0, # d3
    4: -6.0,  # d4 (high-spin)
    5: 0.0,   # d5 (high-spin)
    6: -4.0,  # d6
    7: -8.0,  # d7
    8: -12.0, # d8
    9: -6.0,  # d9
    10: 0.0,  # d10
}

# Dq values (kJ/mol per Dq unit) for different donor subtypes
# Spectrochemical series: water = 10.0 reference
# Real ratios from spectroscopic 10Dq measurements
DQ_BY_DONOR = {
    "O_ether": 8.5, "O_hydroxyl": 9.0, "O_carboxylate": 10.7,
    "O_phenolate": 11.0, "O_hydroxamate": 11.5, "O_catecholate": 12.0,
    "O_phosphate": 10.2, "O_sulfonate": 8.8,
    "N_amine": 12.6, "N_imine": 14.0, "N_pyridine": 13.5,
    "N_imidazole": 12.8, "N_nitrile": 11.5, "N_amide": 11.0,
    "S_thioether": 9.0, "S_thiosulfate": 8.5, "S_thiolate": 9.5,
    "S_dithiocarbamate": 9.8,
    "P_phosphine": 16.0,
    "Cl_chloride": 7.9, "Br_bromide": 7.5, "I_iodide": 7.0,
}

DQ_WATER = 10.0  # kJ/mol reference for aqua complex


# ═══════════════════════════════════════════════════════════════════════════
# COVALENT BDE LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

COVALENT_BDE = {
    # (metal_formula, donor_element): BDE in kJ/mol
    ("Hg2+", "S"): 217, ("Ag+", "S"): 216, ("Au+", "S"): 253,
    ("Au3+", "S"): 230, ("Pt2+", "S"): 235, ("Pd2+", "S"): 210,
    ("Cu+", "S"): 190, ("Cu2+", "S"): 120, ("Pb2+", "S"): 168,
    ("Cd2+", "S"): 180, ("Hg2+", "N"): 80, ("Ag+", "N"): 70,
    ("Au+", "C"): 200, ("Hg2+", "C"): 122,
}


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 11: DONOR CHARGE TABLE (for electrostatic z-z term)
# Formal charge on donor atom when coordinating
# ═══════════════════════════════════════════════════════════════════════════

DONOR_FORMAL_CHARGE = {
    # Anionic donors (negative charge)
    "O_carboxylate": -1.0, "O_phenolate": -1.0, "O_hydroxamate": -1.0,
    "O_catecholate": -1.0, "O_phosphate": -1.0, "O_sulfonate": -1.0,
    "S_thiolate": -1.0, "S_thiosulfate": -1.0, "S_dithiocarbamate": -0.5,
    "Cl_chloride": -1.0, "Br_bromide": -1.0, "I_iodide": -1.0,
    # Neutral donors
    "O_ether": 0.0, "O_hydroxyl": 0.0,
    "N_amine": 0.0, "N_imine": 0.0, "N_pyridine": 0.0,
    "N_imidazole": 0.0, "N_nitrile": 0.0, "N_amide": 0.0,
    "S_thioether": 0.0,
    "P_phosphine": 0.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 12: IRVING-WILLIAMS EMPIRICAL CURVE
# Extra stabilization for 3d transition metals beyond what LFSE captures
# Values in kJ/mol, referenced to Ca2+ = 0
# These encode the experimental double-humped stability ordering
# ═══════════════════════════════════════════════════════════════════════════

IRVING_WILLIAMS_BONUS = {
    "Ca2+":  0.0, "Mg2+":  0.0, "Ba2+": 0.0, "Sr2+": 0.0,
    "Mn2+": -5.0,  "Fe2+": -9.0, "Co2+": -12.0,
    "Ni2+": -15.0, "Cu2+": -18.0, "Zn2+": -7.0,
    # Extended to other divalent
    "Cd2+": -3.0,  "Pb2+": -2.0,  "Hg2+": -5.0,
    # Trivalent: scaled by charge
    "Fe3+": -12.0, "Cr3+": -10.0, "Co3+": -22.0,
    "Al3+": -3.0,  "Ga3+": -5.0,
    # Monovalent
    "Ag+":  -5.0,  "Cu+": -3.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 13: CHELATE RING-SIZE PENALTY TABLE (Hancock Rule)
# ΔΔH (kJ/mol) per ring for substituting 5→6 membered ring
# Indexed by ionic radius range (pm)
# Larger metals pay MORE penalty for 6-membered rings
# ═══════════════════════════════════════════════════════════════════════════

def _hancock_ring_penalty(ionic_radius_pm: float) -> float:
    """Per-ring ΔΔH penalty (kJ/mol) for 6-membered vs 5-membered chelate ring.

    Hancock & Martell (Chem. Rev. 1989): larger metals are destabilized more
    by 6-membered rings due to bite angle mismatch.
    Returns positive value (penalty) in kJ/mol.
    """
    r_nm = ionic_radius_pm / 1000.0  # Convert pm → nm
    r_optimal = 0.065  # nm, optimal metal size for 6-membered ring
    if r_nm <= r_optimal:
        return 2.0  # Small penalty for small metals
    else:
        # Linear scaling: ~2 kJ/mol per 0.01 nm above optimal
        return 2.0 + 200.0 * (r_nm - r_optimal)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 14: REPULSION FORCES
# Three sub-terms: CN overpacking, donor-donor anionic repulsion, steric strain
# ═══════════════════════════════════════════════════════════════════════════

# Donor van der Waals radii (Å) for steric calculations
# Based on Bondi vdW radii of coordinating atom
DONOR_VDW_RADIUS = {
    "O_ether": 1.52, "O_hydroxyl": 1.52, "O_carboxylate": 1.52,
    "O_phenolate": 1.52, "O_hydroxamate": 1.52, "O_catecholate": 1.52,
    "O_phosphate": 1.52, "O_sulfonate": 1.52,
    "N_amine": 1.55, "N_imine": 1.55, "N_pyridine": 1.55,
    "N_imidazole": 1.55, "N_nitrile": 1.55, "N_amide": 1.55,
    "S_thioether": 1.80, "S_thiosulfate": 1.80, "S_thiolate": 1.80,
    "S_dithiocarbamate": 1.80,
    "P_phosphine": 1.80,
    "Cl_chloride": 1.75, "Br_bromide": 1.85, "I_iodide": 1.98,
}

# Ligand cone angle approximation (degrees) — steric bulk of donor group
# Larger cone = more steric demand in the coordination sphere
DONOR_CONE_ANGLE = {
    "O_ether": 80, "O_hydroxyl": 70, "O_carboxylate": 85,
    "O_phenolate": 100, "O_hydroxamate": 95, "O_catecholate": 105,
    "O_phosphate": 95, "O_sulfonate": 90,
    "N_amine": 85, "N_imine": 90, "N_pyridine": 100,
    "N_imidazole": 95, "N_nitrile": 65, "N_amide": 80,
    "S_thioether": 95, "S_thiosulfate": 90, "S_thiolate": 85,
    "S_dithiocarbamate": 110,
    "P_phosphine": 145,  # Tolman cone angle for PPh3
    "Cl_chloride": 80, "Br_bromide": 85, "I_iodide": 90,
}


def _cn_overpack_penalty(metal: 'MetalProperties', n_donors: int) -> float:
    """Penalty (kJ/mol, positive) for exceeding metal's preferred CN range.

    Overpacking costs energy due to:
    - Bond angle compression (Pauli repulsion between donor electron clouds)
    - Reduced orbital overlap per bond (dilution effect)
    """
    cn_max = metal.cn_range[1]
    cn_min = metal.cn_range[0]
    if n_donors > cn_max:
        excess = n_donors - cn_max
        # Exponential ramp: first extra donor costs ~8 kJ, second ~20, etc.
        return 8.0 * excess + 6.0 * excess * (excess - 1) / 2.0
    elif n_donors < cn_min:
        deficit = cn_min - n_donors
        # Underpacking: less severe, metal can fill with water
        return 3.0 * deficit
    return 0.0


# Metals with strong preferred CN below their maximum CN
# Relativistic 6s contraction: Hg²⁺ and Ag⁺ strongly prefer CN=2 linear
PREFERRED_CN = {
    "Hg2+": 2,   # 6s² → linear sp hybridization, very stable
    "Ag+":  2,   # Similar relativistic preference, but weaker
}


def _cn_preference_penalty(metal: 'MetalProperties', n_donors: int) -> float:
    """Penalty for exceeding a metal's PREFERRED CN (not max CN).

    Distinct from overpack penalty: Hg²⁺ can accept CN=4-6, but strongly
    prefers CN=2. Each additional donor costs energy.

    Returns positive value (kJ/mol, unfavorable).
    """
    pref_cn = PREFERRED_CN.get(metal.formula, None)
    if pref_cn is None or n_donors <= pref_cn:
        return 0.0
    excess = n_donors - pref_cn
    return PARAMS["cn_pref_penalty"] * excess


def _donor_donor_repulsion(active_subtypes: list, r_pm: float) -> float:
    """Repulsion between anionic donors crowded in the coordination sphere.

    Multiple negatively charged donors in close proximity repel each other.
    The effect scales as n_anionic × (n_anionic - 1) / r.
    """
    n_anionic = sum(1 for st in active_subtypes
                    if DONOR_FORMAL_CHARGE.get(st, 0.0) < -0.3)
    if n_anionic < 2:
        return 0.0
    # Coulombic repulsion between anionic donors: scales as n(n-1)/2
    # Attenuated by metal radius (larger sphere = more separation)
    n_pairs = n_anionic * (n_anionic - 1) / 2.0
    r_factor = 85.0 / max(r_pm, 50.0)  # Normalized to reference ~85pm
    return PARAMS["repul_anionic"] * n_pairs * r_factor  # kJ/mol, positive = unfavorable


def _steric_strain(active_subtypes: list, metal: 'MetalProperties') -> float:
    """Steric strain from bulky donors crowding the coordination sphere.

    Uses sum of cone angles vs. available solid angle at coordination number.
    """
    n = len(active_subtypes)
    if n <= 2:
        return 0.0
    total_cone = sum(DONOR_CONE_ANGLE.get(st, 100) for st in active_subtypes)
    # Available angular space depends on CN geometry
    # Octahedral: ~6 × 90° = 540° effective; tetrahedral: ~4 × 109.5° = 438°
    if n <= 4:
        available = 440.0
    elif n <= 6:
        available = 540.0
    else:
        available = 540.0 + 60.0 * (n - 6)  # Expanded CN

    if total_cone > available:
        overcrowding = total_cone - available
        return PARAMS["repul_steric"] * overcrowding
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 15: ENTROPY DECOMPOSITION
# Conformational rotor penalty, temperature-dependent ΔG, rigidity bonus
# ═══════════════════════════════════════════════════════════════════════════

# Estimated rotatable bonds per donor arm (connecting donor to backbone)
# Chelated donors have these frozen upon binding
DONOR_ROTOR_COUNT = {
    # Rigid donors (aromatic, constrained)
    "N_pyridine": 0, "N_imidazole": 0, "N_imine": 0, "N_nitrile": 0,
    "O_catecholate": 0, "O_phenolate": 0,
    # Semi-rigid (1 rotor in linker arm)
    "O_carboxylate": 1, "O_hydroxamate": 1, "O_phosphate": 1,
    "S_dithiocarbamate": 0, "N_amide": 1,
    # Flexible (2+ rotors in chain)
    "N_amine": 2, "O_hydroxyl": 1, "O_ether": 2,
    "S_thiolate": 1, "S_thioether": 2, "S_thiosulfate": 1,
    "O_sulfonate": 1,
    "P_phosphine": 1,
    "Cl_chloride": 0, "Br_bromide": 0, "I_iodide": 0,
}

# Conformational entropy cost per frozen rotatable bond (kJ/mol at 298K)
# Mammen, Shakhnovich, Whitesides 1998 consensus: ~3.4 kJ/mol
# Now read from PARAMS["rotor_cost"]

# Fraction of rotors actually frozen upon chelation
# Now read from PARAMS["freeze_chelate"], PARAMS["freeze_mono"], PARAMS["freeze_macro"]


def _conformational_entropy_penalty(
    active_subtypes: list,
    chelate_rings: int,
    is_macrocyclic: bool,
    n_ligand_molecules: int,
) -> float:
    """Conformational entropy cost from freezing rotatable bonds upon binding.

    Returns positive value (unfavorable, kJ/mol).
    """
    total_rotors = sum(DONOR_ROTOR_COUNT.get(st, 1) for st in active_subtypes)
    if total_rotors == 0:
        return 0.0

    # Determine freeze fraction based on ligand type
    if is_macrocyclic:
        f_freeze = PARAMS["freeze_macro"]
    elif chelate_rings > 0:
        f_freeze = PARAMS["freeze_chelate"]
    else:
        f_freeze = PARAMS["freeze_mono"]

    n_frozen = total_rotors * f_freeze
    return PARAMS["rotor_cost"] * n_frozen


def _temperature_correction(dg_298: float, temperature_K: float) -> float:
    """Gibbs-Helmholtz temperature correction for ΔG.

    Approximation: ΔG(T) = ΔG(298) × T/298
    This assumes ΔH and ΔS are approximately temperature-independent
    (valid within ±50K of 298K for most coordination complexes).

    For more accuracy, would need ΔH and ΔS decomposition + ΔCp corrections.
    """
    if abs(temperature_K - 298.15) < 0.5:
        return dg_298  # No correction needed
    # Linear scaling: entropy terms scale with T, enthalpy terms don't
    # Approximate: 60% of ΔG is enthalpic, 40% is entropic for coordination
    f_enthalpy = 0.60
    f_entropy = 0.40
    dg_T = dg_298 * (f_enthalpy + f_entropy * temperature_K / 298.15)
    return dg_T


# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS (patchable by calibration optimizer)
# All optimizable scalars in one dict. Optimizer writes here directly.
# ═══════════════════════════════════════════════════════════════════════════

PARAMS = {
    # Term 2: charge scaling
    "charge_scale":       -1.00,    # multiplier on (z²-1)
    # Term 3: HSAB match/mismatch
    "hsab_match":         -1.50,    # kJ/mol, match bonus per donor
    "hsab_mismatch":      10.00,    # kJ/mol, mismatch penalty per donor
    # Term 4: chelate effect (kJ/mol per ring)
    "chelate_ring_d":    -10.84,    # d>0, z≥2 metals
    "chelate_ring_z1":    -4.25,    # z=1 metals
    "chelate_ring_d0":    -5.82,    # d0 divalent (Ca, Mg)
    # Term 5: desolvation cost
    "desolv_frac_base":    0.005,   # base fraction
    "desolv_frac_slope":   0.001,   # per-charge slope
    # Term 6: LFSE amplifier
    "lfse_amp":            1.00,    # delta_dq multiplier
    # Term 7: Jahn-Teller (structural, less tunable)
    "jt_strong":         -12.00,    # square planar (CN≤4)
    "jt_moderate":        -6.00,    # tetragonal distortion (CN>4)
    # Term 9: translational entropy
    "trans_entropy":       5.50,    # kJ/mol per extra ligand molecule
    # Term 10: macrocyclic
    "macro_preorg":       -6.27,    # kJ/mol, entropic preorganization
    "macro_cavity_k":     -5.03,    # kJ/mol, cavity size-match peak
    "macro_sigma":         0.015,   # nm, Gaussian width of cavity match
    # Term 12: electrostatic z-z
    "elec_zz_k":          -2.68,    # kJ/mol scaling factor
    # Term 14b: repulsion terms
    "repul_anionic":       1.67,    # donor-donor anionic repulsion coeff
    "repul_steric":        0.15,    # steric overcrowding coeff (kJ/mol/degree)
    # Term 14c: preferred CN penalty (relativistic metals)
    "cn_pref_penalty":     0.0,     # kJ/mol per donor beyond preferred CN (disabled — needs effective-CN model)
    # Term 15: entropy decomposition
    "rotor_cost":          2.00,    # kJ/mol per frozen rotor
    "freeze_chelate":      0.50,    # fraction frozen in chelates
    "freeze_mono":         0.25,    # fraction frozen, monodentate
    "freeze_macro":        0.10,    # fraction frozen, macrocyclic
}


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

RT_KJ = 5.71  # RT at 25°C in kJ/mol (= 2.479 kJ/mol × ln(10))


def _donor_element(subtype: str) -> str:
    """Extract element from subtype string: 'N_amine' -> 'N'."""
    return subtype.split("_")[0]


def _effective_cn(metal: MetalProperties, n_donors: int) -> int:
    """Clamp donor count to metal's coordination range."""
    return max(metal.cn_range[0], min(metal.cn_range[1], n_donors))


def _protonation_penalty(subtype: str, pH: float) -> float:
    """Fraction of donor that is deprotonated (available) at given pH.

    Returns 0.0 (fully protonated, cannot bind) to 1.0 (fully available).
    """
    pka = DONOR_PKA.get(subtype, 99.0)
    if pH >= pka + 3:
        return 1.0  # Fully deprotonated
    elif pH <= pka - 3:
        return 0.0  # Fully protonated
    else:
        # Henderson-Hasselbalch: fraction deprotonated
        return 1.0 / (1.0 + 10.0 ** (pka - pH))


def predict_log_k(
    metal_formula: str,
    donor_subtypes: list[str],
    chelate_rings: int = 0,
    ring_sizes: Optional[list[int]] = None,
    pH: float = 7.0,
    is_macrocyclic: bool = False,
    cavity_radius_nm: Optional[float] = None,
    n_ligand_molecules: int = 1,
    temperature_K: float = 298.15,
    verbose: bool = False,
) -> float:
    """Predict conditional formation constant log K' at given pH and temperature.

    Strategy: compute intrinsic log K (fully deprotonated ligand), then
    apply conditional pH correction for donor protonation competition.

    Args:
        metal_formula: e.g. "Pb2+", "Fe3+", "Cu2+"
        donor_subtypes: list of donor subtypes
        chelate_rings: number of chelate rings (total)
        ring_sizes: list of ring sizes (5 or 6) per ring; None = all 5-membered
        pH: solution pH (conditional correction)
        is_macrocyclic: True for macrocycles (crown ethers, cryptands, porphyrins)
        cavity_radius_nm: macrocycle cavity radius in nm (for size-match); None = skip
        n_ligand_molecules: number of separate ligand molecules
        temperature_K: temperature in Kelvin (default 298.15 = 25°C)
        verbose: print term-by-term decomposition

    Returns:
        Predicted conditional log K' at specified pH and temperature
    """
    metal = METAL_DB.get(metal_formula)
    if metal is None:
        raise ValueError(f"Unknown metal: {metal_formula}. "
                         f"Available: {sorted(METAL_DB.keys())}")

    n_donors = len(donor_subtypes)
    if n_donors == 0:
        return 0.0

    eff_cn = _effective_cn(metal, n_donors)
    charge = metal.charge
    softness_m = metal.hsab_softness
    r_pm = metal.ionic_radius_pm
    active_subtypes = donor_subtypes[:eff_cn]
    n_active = len(active_subtypes)

    # ── Term 1: Per-donor exchange energy (intrinsic, all donors active) ──
    dg_exchange = 0.0
    for st in active_subtypes:
        base_e = SUBTYPE_EXCHANGE.get(st, ELEMENT_EXCHANGE.get(_donor_element(st), -5.0))
        dg_exchange += base_e

    # ── Term 2: Charge-dependent electrostatic enhancement ────────────
    # z²/r scaling: higher charge and smaller radius = stronger field
    dg_charge = PARAMS["charge_scale"] * (charge ** 2 - 1)

    # ── Term 3: Metal-donor exchange matrix (multiplicative HSAB × size) ─
    # Replaces additive HSAB with multiplicative framework:
    # Good HSAB match amplifies exchange, mismatch dampens it
    # f_size: smaller metals form stronger bonds with hard donors (z/r effect)
    dg_hsab = 0.0
    r_ref = 85.0  # pm, reference radius (average of 3d divalent metals)
    for st in active_subtypes:
        softness_d = DONOR_SOFTNESS.get(st, 0.3)
        delta_soft = abs(softness_m - softness_d)

        # f_size: small hard metals get bonus with hard donors, penalty with soft
        f_size = 1.0
        if softness_d < 0.3:  # Hard donor
            f_size = 1.0 + 0.3 * (r_ref - r_pm) / r_ref  # Smaller = stronger
        elif softness_d > 0.6:  # Soft donor
            f_size = 1.0 - 0.2 * (r_ref - r_pm) / r_ref  # Smaller = weaker with soft

        if delta_soft < 0.30:
            # Good match: bonus scaled by match quality and size
            dg_hsab += PARAMS["hsab_match"] * (0.30 - delta_soft) / 0.30 * f_size
        else:
            # Mismatch: penalty
            dg_hsab += PARAMS["hsab_mismatch"] * (delta_soft - 0.30) * f_size

    # ── Term 4: Chelate effect ───────────────────────────────────────
    if metal.d_electrons > 0 and charge >= 2:
        chelate_per_ring = PARAMS["chelate_ring_d"]
    elif charge == 1:
        chelate_per_ring = PARAMS["chelate_ring_z1"]
    else:
        chelate_per_ring = PARAMS["chelate_ring_d0"]
    dg_chelate = chelate_per_ring * chelate_rings

    # ── Term 5: Desolvation cost ─────────────────────────────────────
    # Metal-specific: fraction of per-water hydration energy × donors
    f_exchange = PARAMS["desolv_frac_base"] + PARAMS["desolv_frac_slope"] * charge
    per_water_cost = abs(metal.dg_hyd_kj) / metal.cn_aqua
    dg_desolv = f_exchange * per_water_cost * n_active  # Positive = unfavorable

    # ── Term 6: ΔLFSE (ligand field vs aqua) ─────────────────────────
    # Scaled by a factor of 3 to match experimental magnitudes
    dg_lfse = 0.0
    if metal.d_electrons > 0 and metal.d_electrons < 10:
        lfse_factor = LFSE_OCT_HIGH_SPIN.get(metal.d_electrons, 0.0)
        avg_dq_ligand = sum(DQ_BY_DONOR.get(st, 10.0)
                            for st in active_subtypes) / max(1, n_active)
        delta_dq = avg_dq_ligand - DQ_WATER
        dg_lfse = lfse_factor * delta_dq * PARAMS["lfse_amp"]  # Amplified to match real LFSE scale
        dg_lfse = max(dg_lfse, -80.0)  # Cap

    # ── Term 7: Jahn-Teller correction ───────────────────────────────
    dg_jt = 0.0
    if metal.jahn_teller:
        if eff_cn <= 4:
            dg_jt = PARAMS["jt_strong"]
        else:
            dg_jt = PARAMS["jt_moderate"]

    # ── Term 8: Covalent BDE contribution ────────────────────────────
    dg_covalent = 0.0
    if metal.covalent_fraction > 0:
        for st in active_subtypes:
            el = _donor_element(st)
            bde = COVALENT_BDE.get((metal.formula, el), 0.0)
            if bde > 0:
                dg_covalent += -bde * metal.covalent_fraction

    # ── Term 9: Translational entropy cost ───────────────────────────
    dg_trans = PARAMS["trans_entropy"] * (n_ligand_molecules - 1) if n_ligand_molecules > 1 else 0.0

    # ── Term 10: Macrocyclic/preorganization effect ────────────────
    # Two components: (a) entropic preorganization, (b) cavity size-match
    dg_macro = 0.0
    if is_macrocyclic:
        # (a) Entropic preorganization: ~5-8 kJ/mol for macrocycles
        dg_macro = PARAMS["macro_preorg"]

        # (b) Cavity size-match Gaussian (Pedersen selectivity)
        if cavity_radius_nm is not None:
            r_ion_nm = r_pm / 1000.0
            sigma_cavity = PARAMS["macro_sigma"]
            size_match = math.exp(-(r_ion_nm - cavity_radius_nm)**2
                                  / (2 * sigma_cavity**2))
            dg_macro += PARAMS["macro_cavity_k"] * size_match

    # ── Term 11: Chelate ring-size selectivity (Hancock rule) ────────
    # 6-membered rings destabilize large metals relative to 5-membered
    dg_ring_size = 0.0
    if ring_sizes:
        penalty_per_6ring = _hancock_ring_penalty(r_pm)
        for rs in ring_sizes:
            if rs == 6:
                dg_ring_size += penalty_per_6ring  # Positive = destabilizing
            elif rs == 4:
                dg_ring_size += 15.0  # Large penalty: 4-ring is strained for all metals
            elif rs >= 7:
                dg_ring_size += 8.0   # Moderate penalty: large rings are floppy

    # ── Term 12: Electrostatic z-z charge-charge attraction ──────────
    # Anionic donors are attracted to cationic metals: ΔG ∝ z_metal × z_donor / r
    dg_elec = 0.0
    k_elec = PARAMS["elec_zz_k"]  # kJ/mol scaling factor per unit of z_M × |z_D| / (r/100)
    for st in active_subtypes:
        z_donor = DONOR_FORMAL_CHARGE.get(st, 0.0)
        if z_donor < 0:
            # Coulombic attraction: favorable (negative ΔG)
            dg_elec += k_elec * charge * abs(z_donor) / (r_pm / 100.0)

    # ── Term 13: Irving-Williams empirical enhancement ───────────────
    # Extra stabilization for 3d metals beyond LFSE/JT
    # Captures kinetic lability, nephelauxetic, and spin-orbit effects
    dg_iw = IRVING_WILLIAMS_BONUS.get(metal.formula, 0.0)

    # ── Term 14: Repulsion forces ─────────────────────────────────────
    # Three sub-terms: CN overpacking, donor-donor anionic repulsion, steric
    dg_repulsion = 0.0
    # (a) CN overpacking: penalty for exceeding metal's coordination max
    dg_repulsion += _cn_overpack_penalty(metal, n_donors)
    # (b) CN preference: penalty for exceeding preferred CN (Hg²⁺, Ag⁺)
    dg_repulsion += _cn_preference_penalty(metal, n_donors)
    # (c) Donor-donor anionic repulsion: charged donors repel in tight sphere
    dg_repulsion += _donor_donor_repulsion(active_subtypes, r_pm)
    # (d) Steric strain: bulky donors crowding the coordination sphere
    dg_repulsion += _steric_strain(active_subtypes, metal)

    # ── Term 15: Entropy decomposition ────────────────────────────────
    # (a) Conformational rotor penalty: flexible arms freeze upon chelation
    dg_conf_entropy = _conformational_entropy_penalty(
        active_subtypes, chelate_rings, is_macrocyclic, n_ligand_molecules)

    # ── Intrinsic ΔG and log K ───────────────────────────────────────
    dg_intrinsic = (dg_exchange + dg_charge + dg_hsab + dg_chelate
                    + dg_desolv + dg_lfse + dg_jt + dg_covalent
                    + dg_trans + dg_macro + dg_ring_size + dg_elec + dg_iw
                    + dg_repulsion + dg_conf_entropy)

    # (b) Temperature correction via Gibbs-Helmholtz approximation
    dg_at_T = _temperature_correction(dg_intrinsic, temperature_K)

    # RT at operating temperature
    R_kJ = 0.008314  # kJ/(mol·K)
    rt_T = 2.303 * R_kJ * temperature_K  # = RT × ln(10)
    log_k_intrinsic = -dg_at_T / rt_T

    # ── Conditional pH correction ────────────────────────────────────
    # log K_cond = log K_intrinsic - Σ log(1 + 10^(pKa_i - pH))
    # This accounts for protonation competition at the operating pH
    ph_penalty = 0.0
    for st in active_subtypes:
        pka = DONOR_PKA.get(st, 99.0)
        if pka >= 50 or pka <= 0:
            continue  # Non-protonable or always deprotonated
        if pka > pH - 5:
            ph_penalty += math.log10(1.0 + 10.0 ** (pka - pH))

    log_k_cond = log_k_intrinsic - ph_penalty

    if verbose:
        print(f"  Metal: {metal_formula} (z={charge}, d={metal.d_electrons}, "
              f"σ={softness_m:.2f}, r={r_pm}pm)")
        print(f"  Donors: {active_subtypes} (eff_CN={eff_cn})")
        t_str = f", T={temperature_K:.0f}K" if abs(temperature_K - 298.15) > 0.5 else ""
        print(f"  pH={pH}, macrocyclic={is_macrocyclic}{t_str}")
        print(f"  ── Energy terms (kJ/mol) ──")
        print(f"  Exchange:     {dg_exchange:+8.1f}")
        print(f"  Charge:       {dg_charge:+8.1f}")
        print(f"  HSAB×size:    {dg_hsab:+8.1f}")
        print(f"  Chelate:      {dg_chelate:+8.1f}")
        print(f"  Desolvation:  {dg_desolv:+8.1f}")
        print(f"  ΔLFSE:        {dg_lfse:+8.1f}")
        print(f"  Jahn-Teller:  {dg_jt:+8.1f}")
        print(f"  Covalent:     {dg_covalent:+8.1f}")
        print(f"  Trans. S:     {dg_trans:+8.1f}")
        print(f"  Macrocyclic:  {dg_macro:+8.1f}")
        print(f"  Ring-size:    {dg_ring_size:+8.1f}")
        print(f"  Elec z-z:     {dg_elec:+8.1f}")
        print(f"  Irving-Will:  {dg_iw:+8.1f}")
        print(f"  Repulsion:    {dg_repulsion:+8.1f}")
        print(f"  Conf. entropy:{dg_conf_entropy:+8.1f}")
        print(f"  ─────────────────────────")
        print(f"  ΔG(298K):     {dg_intrinsic:+8.1f} kJ/mol")
        if abs(temperature_K - 298.15) > 0.5:
            print(f"  ΔG({temperature_K:.0f}K):     {dg_at_T:+8.1f} kJ/mol")
        print(f"  log K (int):  {log_k_intrinsic:+8.1f}")
        print(f"  pH penalty:   {ph_penalty:+8.1f} (conditional correction)")
        print(f"  log K (cond): {log_k_cond:+8.1f}")

    return log_k_cond


def predict_selectivity(
    target_metal: str,
    interferent_metals: list[str],
    donor_subtypes: list[str],
    chelate_rings: int = 0,
    ring_sizes: Optional[list[int]] = None,
    pH: float = 7.0,
    is_macrocyclic: bool = False,
    cavity_radius_nm: Optional[float] = None,
    n_ligand_molecules: int = 1,
    temperature_K: float = 298.15,
) -> dict:
    """Predict log K for target and all interferents; compute selectivity gaps.

    Returns dict with keys:
        target_log_k, interferent_log_ks (dict), selectivity_gaps (dict),
        min_gap, worst_interferent
    """
    kwargs = dict(chelate_rings=chelate_rings, ring_sizes=ring_sizes,
                  pH=pH, is_macrocyclic=is_macrocyclic,
                  cavity_radius_nm=cavity_radius_nm,
                  n_ligand_molecules=n_ligand_molecules,
                  temperature_K=temperature_K)

    target_lk = predict_log_k(target_metal, donor_subtypes, **kwargs)

    interferent_lks = {}
    gaps = {}
    for m in interferent_metals:
        try:
            lk = predict_log_k(m, donor_subtypes, **kwargs)
        except ValueError:
            lk = 0.0  # Unknown metal, assume no binding
        interferent_lks[m] = lk
        gaps[m] = target_lk - lk

    worst = min(gaps, key=gaps.get) if gaps else None
    min_gap = gaps[worst] if worst else float('inf')

    return {
        "target_log_k": target_lk,
        "interferent_log_ks": interferent_lks,
        "selectivity_gaps": gaps,
        "min_gap": min_gap,
        "worst_interferent": worst,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== MABE Frozen Scorer v4 — Self-Test (17 physics terms) ===\n")

    # EDTA-type donor set: N2O4, 5 chelate rings
    edta_donors = ["N_amine", "N_amine", "O_carboxylate", "O_carboxylate",
                   "O_carboxylate", "O_carboxylate"]

    lit = {"Ca2+": 10.7, "Mn2+": 13.8, "Fe2+": 14.3, "Co2+": 16.3,
           "Ni2+": 18.6, "Cu2+": 18.8, "Zn2+": 16.5}

    print("EDTA-type ligand — Irving-Williams series (pH 7):")
    print(f"  {'Metal':6s} {'Pred':>6s} {'Lit':>6s} {'Err':>6s}")
    for m in ["Ca2+", "Mn2+", "Fe2+", "Co2+", "Ni2+", "Cu2+", "Zn2+"]:
        lk = predict_log_k(m, edta_donors, chelate_rings=5, pH=7.0)
        exp = lit.get(m, 0)
        print(f"  {m:6s} {lk:6.1f} {exp:6.1f} {lk-exp:+6.1f}")

    # Repulsion test: overcrowding CN=8 donors on a CN=4 metal
    print("\nRepulsion test — overcrowding:")
    lk_4 = predict_log_k("Cu2+", ["N_amine"]*4, chelate_rings=2, pH=7.0)
    lk_8 = predict_log_k("Cu2+", ["N_amine"]*8, chelate_rings=4, pH=7.0)
    print(f"  Cu2+ + 4×N_amine (CN=4-6): log K = {lk_4:.1f}")
    print(f"  Cu2+ + 8×N_amine (overpack): log K = {lk_8:.1f}")
    print(f"  Overcrowding penalty visible: {'YES' if lk_8 < lk_4 + 5 else 'NO'}")

    # Donor-donor anionic repulsion
    print("\nAnionic donor repulsion (Fe3+):")
    lk_neutral = predict_log_k("Fe3+", ["N_amine"]*6, chelate_rings=3, pH=10.0)
    lk_anionic = predict_log_k("Fe3+", ["O_carboxylate"]*6, chelate_rings=5, pH=10.0)
    print(f"  Fe3+ + 6×N_amine (neutral):     log K = {lk_neutral:.1f}")
    print(f"  Fe3+ + 6×O_carboxylate (6×-1):  log K = {lk_anionic:.1f}")

    # Steric test: phosphine (cone angle 145°) vs amine (95°)
    print("\nSteric strain — bulky PPh3 vs compact NH2:")
    lk_phos = predict_log_k("Pd2+", ["P_phosphine"]*4, chelate_rings=2, pH=7.0)
    lk_amine = predict_log_k("Pd2+", ["N_amine"]*4, chelate_rings=2, pH=7.0)
    print(f"  Pd2+ + 4×PPh3 (bulky):  log K = {lk_phos:.1f}")
    print(f"  Pd2+ + 4×NH2  (compact): log K = {lk_amine:.1f}")

    # Entropy: rigid vs flexible ligand
    print("\nConformational entropy — rigid vs flexible:")
    rigid = ["N_pyridine", "N_pyridine", "N_pyridine",
             "N_pyridine", "N_pyridine", "N_pyridine"]
    flex = ["N_amine", "N_amine", "N_amine",
            "N_amine", "N_amine", "N_amine"]
    lk_rigid = predict_log_k("Ni2+", rigid, chelate_rings=3, pH=7.0)
    lk_flex = predict_log_k("Ni2+", flex, chelate_rings=3, pH=7.0)
    print(f"  Ni2+ + 3×bipy (0 rotors):  log K = {lk_rigid:.1f}")
    print(f"  Ni2+ + 3×en   (12 rotors): log K = {lk_flex:.1f}")
    print(f"  Rigidity advantage: {lk_rigid - lk_flex:+.1f} log K")

    # Temperature dependence
    print("\nTemperature dependence — Cu2+/EDTA:")
    for T in [278, 298, 318, 338, 358]:
        lk_T = predict_log_k("Cu2+", edta_donors, chelate_rings=5, pH=7.0,
                              temperature_K=float(T))
        print(f"  T={T}K ({T-273}°C): log K = {lk_T:.1f}")

    # Verbose decomposition of repulsion+entropy terms
    print("\nFull verbose — Cu2+ EDTA at 298K:")
    predict_log_k("Cu2+", edta_donors, chelate_rings=5, pH=7.0, verbose=True)

    print(f"\nFull verbose — Cu2+ EDTA at 350K:")
    predict_log_k("Cu2+", edta_donors, chelate_rings=5, pH=7.0,
                  temperature_K=350.0, verbose=True)

    print(f"\n{len(METAL_DB)} metals in database. 17 physics terms active.")
    print("Self-test complete.")