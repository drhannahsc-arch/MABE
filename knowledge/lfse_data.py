"""
knowledge/lfse_data.py - Ligand Field Stabilization Energy database.

Crystal field theory in 200 lines:

    When a transition metal ion is surrounded by ligands, the five d orbitals
    split in energy. The splitting pattern depends on geometry:

    Octahedral:  t2g (dxy, dxz, dyz) at -0.4 Dq_oct
                 eg  (dz², dx²-y²)   at +0.6 Dq_oct

    Tetrahedral: e   (dz², dx²-y²)   at -0.6 Dq_tet
                 t2  (dxy, dxz, dyz) at +0.4 Dq_tet
                 Dq_tet ≈ 4/9 × Dq_oct (always weaker)

    Square planar: dx²-y² highest, dxy next, dz² mid, dxz/dyz lowest
                   Strong splitting; d⁸ strongly favored

    LFSE = Σ(n_i × ε_i) where n_i = electron count in orbital i, ε_i = energy

    The LFSE difference between geometries drives geometry preference.
    For d⁸ (Ni²⁺), square planar LFSE >> octahedral LFSE with strong-field
    donors, so Ni²⁺ switches to square planar with sufficient field strength.

SPECTROCHEMICAL SERIES (weak → strong field):
    I⁻ < Br⁻ < S²⁻ < Cl⁻ < N₃⁻ < F⁻ < OH⁻ < ox²⁻ < H₂O
    < NCS⁻ < CH₃CN < py < NH₃ < en < bipy < phen < NO₂⁻
    < PPh₃ < CN⁻ < CO < NO⁺

For binder design:
    S donors: weak-to-intermediate field (except thiolate anion → moderate)
    O donors: weak field (water, carboxylate, hydroxyl)
    N donors: moderate-to-strong field (amine, pyridine, imidazole)
    P donors: strong field (phosphine)
    CN/CO: very strong field (cyanide, carbonyl)

CRITICAL DESIGN IMPLICATION:
    If you want selectivity for Ni²⁺ over Ca²⁺, use 4 strong-field N donors
    in a planar arrangement. Ni²⁺ d⁸ gains ~120-240 kJ/mol LFSE.
    Ca²⁺ d⁰ gains zero. That is a selectivity wall.
"""

from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# d-electron configurations for metal ions
# ═══════════════════════════════════════════════════════════════════════════

METAL_D_ELECTRONS = {
    # Transition metals (common oxidation states in MABE)
    # identity → {oxidation_state: d_count}
    "iron":     {3: 5, 2: 6},       # Fe³⁺ = d⁵, Fe²⁺ = d⁶
    "copper":   {2: 9, 1: 10},      # Cu²⁺ = d⁹, Cu⁺ = d¹⁰
    "nickel":   {2: 8},             # Ni²⁺ = d⁸
    "zinc":     {2: 10},            # Zn²⁺ = d¹⁰
    "cobalt":   {3: 6, 2: 7},       # Co³⁺ = d⁶, Co²⁺ = d⁷
    "manganese":{2: 5, 4: 3, 7: 0}, # Mn²⁺ = d⁵, Mn⁴⁺ = d³, Mn⁷⁺ = d⁰
    "chromium": {3: 3, 6: 0},       # Cr³⁺ = d³, Cr⁶⁺ = d⁰
    "cadmium":  {2: 10},            # Cd²⁺ = d¹⁰
    "silver":   {1: 10},            # Ag⁺ = d¹⁰

    # Post-transition / main group — no crystal field splitting
    # But stereochemically active lone pairs affect geometry
    "lead":     {2: -1},            # Pb²⁺ = [Xe]4f¹⁴5d¹⁰6s² — 6s² lone pair, not d-metal
    "mercury":  {2: -1},            # Hg²⁺ = [Xe]4f¹⁴5d¹⁰ (d¹⁰ + fully filled)
    # -1 sentinel: "post-d / s² lone pair, no crystal field splitting"

    # Hard cations — no d-electrons
    "calcium":  {2: 0},             # Ca²⁺ = d⁰
    "magnesium":{2: 0},             # Mg²⁺ = d⁰
    "sodium":   {1: 0},             # Na⁺ = d⁰
    "potassium":{1: 0},             # K⁺ = d⁰
    "aluminum": {3: 0},             # Al³⁺ = d⁰
    "barium":   {2: 0},             # Ba²⁺ = d⁰

    # Lanthanides (f-electrons, not d — use sentinel -2)
    "cerium":   {3: -2, 4: -2},     # Ce³⁺ = [Xe]4f¹ — f-electron, not crystal field
    "uranium":  {6: -2},            # UO₂²⁺ — actinide, not crystal field

    # Metalloids / oxyanions — no d-splitting
    "arsenate": {5: 0},
    "selenite": {4: 0},
    "chromate": {6: 0},

    # Gold — d⁸ for Au³⁺, d¹⁰ for Au⁺
    "gold":     {3: 8, 1: 10},      # Au³⁺ = d⁸ (strong square planar preference!)
}


# ═══════════════════════════════════════════════════════════════════════════
# Dq values (10Dq in kJ/mol) by donor type
# These are the ligand field splitting parameter for OCTAHEDRAL geometry
# Tetrahedral Dq ≈ 4/9 of octahedral
# ═══════════════════════════════════════════════════════════════════════════

# Representative 10Dq (octahedral) in kJ/mol for M²⁺ ions
# Actual values vary with metal, but the RATIOS between donors are consistent
# Source: Figgis & Hitchman, Lever, Shriver & Atkins
DQ_OCT_KJ = {
    # donor_atom → approximate 10Dq for a generic divalent transition metal
    "O": 100.0,     # water / carboxylate — weak field
    "N": 130.0,     # amine / pyridine — moderate field
    "S": 85.0,      # thiolate / thioether — weak-moderate
    "P": 150.0,     # phosphine — strong field
    "C": 170.0,     # cyanide / carbonyl — very strong field
    "electrostatic": 0.0,  # no covalent splitting
}

# Field strength classification (drives high-spin vs low-spin)
DONOR_FIELD_STRENGTH = {
    "S": "weak",        # S²⁻, thiolate: weak field
    "O": "weak",        # H₂O, COO⁻, OH⁻: weak field
    "N": "moderate",    # NH₃, py, im: moderate → can cause spin crossover in some d⁴-d⁷
    "P": "strong",      # PPh₃, phosphine: strong field
    "C": "strong",      # CN⁻, CO: very strong field
    "electrostatic": "none",
}

# Pairing energy (in kJ/mol) — cost of putting two electrons in same orbital
# Must overcome pairing energy to go low-spin
# Varies by metal; these are representative values
PAIRING_ENERGY = {
    "iron":     {3: 210, 2: 175},     # Fe³⁺ high P, Fe²⁺ moderate
    "cobalt":   {3: 200, 2: 190},     # Co³⁺ moderate, Co²⁺ high
    "nickel":   {2: 180},             # Ni²⁺ — only relevant for tetrahedral (rare low-spin oct)
    "copper":   {2: 170},             # Cu²⁺
    "manganese":{2: 250},             # Mn²⁺ — very high (half-filled stability)
    "chromium": {3: 230},             # Cr³⁺ — very high
}

# Jahn-Teller active configurations (in octahedral)
# These configurations have unequal eg occupancy → distortion mandatory
# d⁴ (t2g³ eg¹), d⁷ low-spin (t2g⁶ eg¹), d⁹ (t2g⁶ eg³)
JAHN_TELLER_ACTIVE_OCT = {
    (4, "high"): True,   # d⁴ high-spin: t2g³ eg¹
    (9, "high"): True,   # d⁹: t2g⁶ eg³ — ALWAYS (Cu²⁺)
    (9, "low"):  True,   # d⁹: same
    (7, "low"):  True,   # d⁷ low-spin: t2g⁶ eg¹
}

# Geometry-specific LFSE in units of Dq (these are exact from crystal field theory)
# For high-spin configurations:
LFSE_DQ_HIGHSPIN = {
    # d_count: {geometry: LFSE in units of Dq}
    0:  {"oct": 0.0,  "tet": 0.0,  "sq_planar": 0.0},
    1:  {"oct": -4.0, "tet": -2.67, "sq_planar": -5.14},
    2:  {"oct": -8.0, "tet": -5.34, "sq_planar": -10.28},
    3:  {"oct": -12.0,"tet": -3.56, "sq_planar": -12.28},
    4:  {"oct": -6.0, "tet": -1.78, "sq_planar": -9.14},
    5:  {"oct": 0.0,  "tet": 0.0,  "sq_planar": -0.0},
    6:  {"oct": -4.0, "tet": -2.67, "sq_planar": -5.14},
    7:  {"oct": -8.0, "tet": -5.34, "sq_planar": -10.28},
    8:  {"oct": -12.0,"tet": -3.56, "sq_planar": -24.56},
    9:  {"oct": -6.0, "tet": -1.78, "sq_planar": -21.42},
    10: {"oct": 0.0,  "tet": 0.0,  "sq_planar": 0.0},
}

# Low-spin corrections (only relevant for d⁴-d⁷ with strong field)
LFSE_DQ_LOWSPIN = {
    4:  {"oct": -16.0, "sq_planar": -17.42},
    5:  {"oct": -20.0, "sq_planar": -24.56},
    6:  {"oct": -24.0, "sq_planar": -24.56},
    7:  {"oct": -18.0, "sq_planar": -24.56},
}

# Stereochemically active lone pair data (post-transition metals)
# These don't have LFSE but have geometry preference from lone pair
LONE_PAIR_METALS = {
    "lead": {
        "lone_pair": "6s²",
        "effect": "hemidirected",
        "description": "Pb²⁺ 6s² lone pair occupies one coordination position. "
                       "Favors asymmetric coordination (hemidirected) at CN 4-6. "
                       "At CN > 8, lone pair becomes stereochemically inactive (holodirected).",
        "geometry_preference": {
            "hemidirected": -8.0,      # kJ/mol stabilization for asymmetric
            "square_pyramidal": -6.0,
            "trigonal_bipyramidal": -4.0,
            "octahedral": 0.0,         # lone pair squeezed — less favorable
            "holodirected": 2.0,       # high CN forces symmetric — slight penalty
        },
        "preferred_cn": [4, 5, 6],
    },
    "mercury": {
        "lone_pair": "5d¹⁰6s⁰",
        "effect": "relativistic",
        "description": "Hg²⁺ has fully filled d¹⁰ shell. Strong relativistic contraction "
                       "of 6s orbital drives linear 2-coordinate preference. "
                       "This is NOT LFSE — it is a relativistic effect.",
        "geometry_preference": {
            "linear": -15.0,           # strong preference
            "tetrahedral": -3.0,
            "octahedral": 0.0,
        },
        "preferred_cn": [2, 4],
    },
}


def get_d_electron_count(identity: str, oxidation_state: int) -> Optional[int]:
    """
    Get d-electron count for a metal ion.

    Returns:
        int >= 0: actual d-electron count (0-10)
        -1: post-transition metal with lone pair effects
        -2: f-block element (lanthanide/actinide)
        None: unknown metal
    """
    metal_data = METAL_D_ELECTRONS.get(identity.lower())
    if metal_data is None:
        return None
    return metal_data.get(oxidation_state, metal_data.get(abs(oxidation_state)))


def get_field_strength(donor_atoms: list[str]) -> str:
    """Classify overall field strength from donor set."""
    if not donor_atoms:
        return "weak"

    strengths = [DONOR_FIELD_STRENGTH.get(d, "weak") for d in donor_atoms]

    strong_count = strengths.count("strong")
    moderate_count = strengths.count("moderate")
    weak_count = strengths.count("weak")

    if strong_count >= len(donor_atoms) // 2:
        return "strong"
    elif moderate_count + strong_count >= len(donor_atoms) // 2:
        return "moderate"
    return "weak"


def average_dq(donor_atoms: list[str]) -> float:
    """Average 10Dq in kJ/mol from donor set."""
    if not donor_atoms:
        return 100.0  # default to water-like
    dqs = [DQ_OCT_KJ.get(d, 100.0) for d in donor_atoms]
    return sum(dqs) / len(dqs)


def is_high_spin(d_count: int, field_strength: str, identity: str,
                  oxidation_state: int) -> bool:
    """
    Determine if the complex is high-spin or low-spin.

    Only relevant for d⁴-d⁷. d¹-d³, d⁸-d¹⁰ always have the same
    configuration regardless of field strength.
    """
    if d_count < 4 or d_count > 7:
        return True  # not ambiguous

    if field_strength == "weak":
        return True  # weak field → always high-spin

    if field_strength == "strong":
        return False  # strong field → always low-spin

    # Moderate field: depends on metal-specific pairing energy
    # Heuristic: moderate field + 3+ charge → often low-spin (more compact d orbitals)
    if oxidation_state >= 3:
        return False  # higher charge → larger splitting → low-spin more likely
    return True  # +2 with moderate field → usually high-spin


def compute_lfse_dq(d_count: int, geometry: str, high_spin: bool) -> float:
    """
    Compute LFSE in units of Dq for a given d-electron count and geometry.
    """
    if d_count <= 0 or d_count > 10:
        return 0.0

    if not high_spin and d_count in LFSE_DQ_LOWSPIN:
        low_data = LFSE_DQ_LOWSPIN[d_count]
        if geometry in low_data:
            return low_data[geometry]

    hs_data = LFSE_DQ_HIGHSPIN.get(d_count, {})
    return hs_data.get(geometry, 0.0)


def is_jahn_teller(d_count: int, high_spin: bool) -> bool:
    """Check if configuration is Jahn-Teller active in octahedral."""
    spin_label = "high" if high_spin else "low"
    return JAHN_TELLER_ACTIVE_OCT.get((d_count, spin_label), False)
