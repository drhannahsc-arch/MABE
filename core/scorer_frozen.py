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
  14. pH-dependent protonation penalty (conditional log K)
  15. Irving-Williams enhancement (empirical d-electron stabilization curve)

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
    # O-donors
    "O_ether":          -1.0,   # Crown ethers, PEG — no charge, no pi
    "O_hydroxyl":       -5.0,   # Alcohols, sugars — weak
    "O_carboxylate":    -4.0,   # Acetate, EDTA arms — charged, good sigma
    "O_phenolate":     -10.0,   # Phenol anion — aromatic + charged
    "O_hydroxamate":   -18.0,   # CONHO⁻ — resonance-stabilized
    "O_catecholate":   -30.0,   # Aromatic diolate — pi-donation into d-orbitals
    "O_phosphate":      -6.0,   # PO4 donors
    "O_sulfonate":      -2.0,   # SO3⁻ — weak donor
    # N-donors
    "N_amine":          -6.0,   # Primary/secondary/tertiary amines
    "N_imine":          -8.0,   # Schiff bases, C=N
    "N_pyridine":       -7.0,   # Aromatic N, pi-acceptor/donor
    "N_imidazole":      -8.0,   # Histidine-like
    "N_nitrile":        -3.0,   # C≡N — weak sigma donor
    "N_amide":          -4.0,   # Peptide bond N — weak
    # S-donors
    "S_thioether":     -10.0,   # R-S-R, methionine-like
    "S_thiosulfate":    -6.0,   # S₂O₃²⁻ terminal S
    "S_thiolate":      -18.0,   # RS⁻, cysteine — strong covalent character
    "S_dithiocarbamate":-15.0,  # CS₂⁻ — chelating S,S-bidentate
    # P-donors
    "P_phosphine":     -15.0,   # PR₃ — strong for soft metals
    # Halides
    "Cl_chloride":      -3.0,
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
    verbose: bool = False,
) -> float:
    """Predict conditional formation constant log K' at given pH.

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
        verbose: print term-by-term decomposition

    Returns:
        Predicted conditional log K' at specified pH
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
    dg_charge = -5.0 * (charge ** 2 - 1)

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
            dg_hsab += -1.5 * (0.30 - delta_soft) / 0.30 * f_size
        else:
            # Mismatch: penalty
            dg_hsab += 10.0 * (delta_soft - 0.30) * f_size

    # ── Term 4: Chelate effect ───────────────────────────────────────
    if metal.d_electrons > 0 and charge >= 2:
        chelate_per_ring = -12.0  # kJ/mol, Schwarzenbach value
    elif charge == 1:
        chelate_per_ring = -6.0
    else:
        chelate_per_ring = -8.0   # d0 divalent (Ca, Mg)
    dg_chelate = chelate_per_ring * chelate_rings

    # ── Term 5: Desolvation cost ─────────────────────────────────────
    # Metal-specific: fraction of per-water hydration energy × donors
    f_exchange = 0.012 + 0.003 * charge  # 1.5% for +1, 1.8% for +2, 2.1% for +3
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
        dg_lfse = lfse_factor * delta_dq * 3.0  # Amplified to match real LFSE scale
        dg_lfse = max(dg_lfse, -80.0)  # Cap

    # ── Term 7: Jahn-Teller correction ───────────────────────────────
    dg_jt = 0.0
    if metal.jahn_teller:
        if eff_cn <= 4:
            dg_jt = -12.0  # Strong: square planar stabilization
        else:
            dg_jt = -6.0   # Moderate: elongated octahedron (tetragonal distortion)

    # ── Term 8: Covalent BDE contribution ────────────────────────────
    dg_covalent = 0.0
    if metal.covalent_fraction > 0:
        for st in active_subtypes:
            el = _donor_element(st)
            bde = COVALENT_BDE.get((metal.formula, el), 0.0)
            if bde > 0:
                dg_covalent += -bde * metal.covalent_fraction

    # ── Term 9: Translational entropy cost ───────────────────────────
    dg_trans = 5.5 * (n_ligand_molecules - 1) if n_ligand_molecules > 1 else 0.0

    # ── Term 10: Macrocyclic/preorganization effect ────────────────
    # Two components: (a) entropic preorganization, (b) cavity size-match
    dg_macro = 0.0
    if is_macrocyclic:
        # (a) Entropic preorganization: ~5-8 kJ/mol for macrocycles
        dg_macro = -6.0

        # (b) Cavity size-match Gaussian (Pedersen selectivity)
        if cavity_radius_nm is not None:
            r_ion_nm = r_pm / 1000.0
            sigma_cavity = 0.015  # nm, width of Gaussian
            size_match = math.exp(-(r_ion_nm - cavity_radius_nm)**2
                                  / (2 * sigma_cavity**2))
            dg_macro += -8.0 * size_match  # Up to -8 kJ/mol extra for perfect match

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
    k_elec = -1.8  # kJ/mol scaling factor per unit of z_M × |z_D| / (r/100)
    for st in active_subtypes:
        z_donor = DONOR_FORMAL_CHARGE.get(st, 0.0)
        if z_donor < 0:
            # Coulombic attraction: favorable (negative ΔG)
            dg_elec += k_elec * charge * abs(z_donor) / (r_pm / 100.0)

    # ── Term 13: Irving-Williams empirical enhancement ───────────────
    # Extra stabilization for 3d metals beyond LFSE/JT
    # Captures kinetic lability, nephelauxetic, and spin-orbit effects
    dg_iw = IRVING_WILLIAMS_BONUS.get(metal.formula, 0.0)

    # ── Intrinsic ΔG and log K ───────────────────────────────────────
    dg_intrinsic = (dg_exchange + dg_charge + dg_hsab + dg_chelate
                    + dg_desolv + dg_lfse + dg_jt + dg_covalent
                    + dg_trans + dg_macro + dg_ring_size + dg_elec + dg_iw)
    log_k_intrinsic = -dg_intrinsic / RT_KJ

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
        print(f"  pH={pH}, macrocyclic={is_macrocyclic}, "
              f"cavity={cavity_radius_nm}nm" if cavity_radius_nm else
              f"  pH={pH}, macrocyclic={is_macrocyclic}")
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
        print(f"  ─────────────────────────")
        print(f"  ΔG_intrinsic: {dg_intrinsic:+8.1f} kJ/mol")
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
) -> dict:
    """Predict log K for target and all interferents; compute selectivity gaps.

    Returns dict with keys:
        target_log_k, interferent_log_ks (dict), selectivity_gaps (dict),
        min_gap, worst_interferent
    """
    kwargs = dict(chelate_rings=chelate_rings, ring_sizes=ring_sizes,
                  pH=pH, is_macrocyclic=is_macrocyclic,
                  cavity_radius_nm=cavity_radius_nm,
                  n_ligand_molecules=n_ligand_molecules)

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
    print("=== MABE Frozen Scorer v3 — Self-Test (15 physics terms) ===\n")

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

    # Crown ether: 18-crown-6 (cavity = 0.13-0.16 nm)
    crown_donors = ["O_ether"] * 6
    print("\n18-Crown-6 with cavity size-match (pH 7):")
    print(f"  {'Metal':6s} {'Pred':>6s} {'Lit':>6s}")
    crown_lit = {"Na+": 0.7, "K+": 2.0, "Rb+": 1.6, "Cs+": 1.0,
                 "Ca2+": 0.5, "Ba2+": 3.9}
    for m in ["Na+", "K+", "Rb+", "Cs+", "Ca2+", "Ba2+"]:
        lk = predict_log_k(m, crown_donors, chelate_rings=6,
                           is_macrocyclic=True, cavity_radius_nm=0.14, pH=7.0)
        exp = crown_lit.get(m, 0)
        print(f"  {m:6s} {lk:6.1f} {exp:6.1f}")

    # Ring-size selectivity: 5-membered vs 6-membered for large vs small metal
    print("\nRing-size selectivity (Hancock rule):")
    en_donors = ["N_amine", "N_amine"]  # ethylenediamine
    for m in ["Cu2+", "Pb2+", "Ca2+"]:
        lk_5 = predict_log_k(m, en_donors, chelate_rings=1,
                              ring_sizes=[5], pH=7.0)
        lk_6 = predict_log_k(m, en_donors, chelate_rings=1,
                              ring_sizes=[6], pH=7.0)
        print(f"  {m:6s}  5-ring: {lk_5:+5.1f}  6-ring: {lk_6:+5.1f}  "
              f"Δ={lk_5-lk_6:+5.1f}")

    # Electrostatic z-z: Fe3+ with anionic vs neutral donors
    print("\nElectrostatic z-z effect (Fe3+):")
    predict_log_k("Fe3+", ["O_carboxylate"]*6, chelate_rings=5, pH=7, verbose=True)
    print()
    predict_log_k("Fe3+", ["N_amine"]*6, chelate_rings=3, pH=7, verbose=True)

    print(f"\nPb2+ selectivity over mine water (pH 5):")
    result = predict_selectivity(
        "Pb2+",
        ["Ca2+", "Mg2+", "Fe3+", "Zn2+", "Cu2+", "Mn2+"],
        ["S_thiolate", "N_amine", "N_amine", "O_carboxylate"],
        chelate_rings=3, pH=5.0)
    print(f"  Pb2+ log K = {result['target_log_k']:.1f}")
    for m, lk in result['interferent_log_ks'].items():
        gap = result['selectivity_gaps'][m]
        print(f"  {m:6s}  log K = {lk:5.1f}  gap = {gap:+5.1f}")
    print(f"  Worst: {result['worst_interferent']} "
          f"(gap = {result['min_gap']:+.1f})")

    print(f"\n{len(METAL_DB)} metals in database. 15 physics terms active.")
    print("Self-test complete.")