"""
Cyclodextrin Knowledge Base.

Properties from BackSolve host registry + literature.
All cavity dimensions from crystallographic data.
Modification library from commercial availability + literature.

Data sources:
    - Rekharsky & Inoue (Chem. Rev. 1998) — thermodynamic compilations
    - Szejtli (Chem. Rev. 1998) — CD chemistry review
    - BackSolve Phase 6-10 calibration dataset
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CDHost:
    """Physical properties of one cyclodextrin variant."""

    name: str
    base_type: str                     # "alpha", "beta", "gamma"
    n_glucose: int

    # ── Cavity geometry (from crystallography) ──
    cavity_diameter_A: float           # internal diameter
    cavity_depth_A: float              # all CDs: ~7.9 Å
    cavity_volume_A3: float
    outer_diameter_A: float
    cavity_sasa_A2: float

    # ── Portal properties ──
    n_portal_HB_sites: int             # primary OH + secondary OH
    portal_diameter_A: float           # narrower face (primary OH)

    # ── Physicochemical ──
    mw: float                          # g/mol
    water_solubility_mM: float         # at 25°C
    pKa_secondary_OH: float            # ~12.1 for all CDs

    # ── Production ──
    commercial: bool
    cost_per_gram_usd: float           # approximate, bulk
    common_suppliers: list[str]

    # ── Modification state ──
    modification: str = "native"       # "native", "HP", "Me", "SBE", etc.
    modification_description: str = ""
    avg_degree_of_substitution: float = 0.0

    # ── BackSolve calibration ──
    backsolve_MAE_logK: float = 0.0    # MAE from Phase 6-10
    n_calibration_entries: int = 0


@dataclass(frozen=True)
class CDModification:
    """A chemical modification applicable to a CD."""

    name: str
    abbreviation: str
    description: str
    effect_on_cavity: str              # "widens", "deepens", "narrows", "unchanged"
    effect_on_solubility: str          # "increases", "decreases", "unchanged"
    effect_on_selectivity: str         # qualitative
    synthesis_complexity: str          # "trivial", "moderate", "complex"
    commercial_availability: bool
    typical_DS: float                  # degree of substitution
    applicable_to: list[str]           # ["alpha", "beta", "gamma"]


# ─────────────────────────────────────────────
# Native CD hosts (from BackSolve host registry)
# ─────────────────────────────────────────────

ALPHA_CD = CDHost(
    name="α-Cyclodextrin",
    base_type="alpha",
    n_glucose=6,
    cavity_diameter_A=4.7,
    cavity_depth_A=7.9,
    cavity_volume_A3=174.0,
    outer_diameter_A=14.6,
    cavity_sasa_A2=307.0,
    n_portal_HB_sites=12,   # 6×OH-2 + 6×OH-3
    portal_diameter_A=4.7,
    mw=972.84,
    water_solubility_mM=149.0,  # 145 mg/mL
    pKa_secondary_OH=12.1,
    commercial=True,
    cost_per_gram_usd=0.50,
    common_suppliers=["Sigma-Aldrich", "TCI", "Wacker"],
    backsolve_MAE_logK=0.62,
    n_calibration_entries=14,
)

BETA_CD = CDHost(
    name="β-Cyclodextrin",
    base_type="beta",
    n_glucose=7,
    cavity_diameter_A=6.0,
    cavity_depth_A=7.9,
    cavity_volume_A3=262.0,
    outer_diameter_A=15.4,
    cavity_sasa_A2=427.0,
    n_portal_HB_sites=14,
    portal_diameter_A=6.0,
    mw=1134.98,
    water_solubility_mM=16.3,  # 18.5 mg/mL — notoriously low
    pKa_secondary_OH=12.1,
    commercial=True,
    cost_per_gram_usd=0.15,
    common_suppliers=["Sigma-Aldrich", "TCI", "Wacker", "Roquette"],
    backsolve_MAE_logK=0.37,
    n_calibration_entries=25,
)

GAMMA_CD = CDHost(
    name="γ-Cyclodextrin",
    base_type="gamma",
    n_glucose=8,
    cavity_diameter_A=7.5,
    cavity_depth_A=7.9,
    cavity_volume_A3=427.0,
    outer_diameter_A=17.5,
    cavity_sasa_A2=590.0,
    n_portal_HB_sites=16,
    portal_diameter_A=7.5,
    mw=1297.12,
    water_solubility_mM=178.0,  # 232 mg/mL
    pKa_secondary_OH=12.1,
    commercial=True,
    cost_per_gram_usd=1.00,
    common_suppliers=["Sigma-Aldrich", "TCI", "Wacker"],
    backsolve_MAE_logK=0.61,
    n_calibration_entries=9,
)

# ─────────────────────────────────────────────
# Modified CDs
# ─────────────────────────────────────────────

HP_BETA_CD = CDHost(
    name="HP-β-Cyclodextrin",
    base_type="beta",
    n_glucose=7,
    cavity_diameter_A=6.0,  # cavity unchanged
    cavity_depth_A=8.5,     # slightly deeper due to HP groups
    cavity_volume_A3=290.0, # ~10% larger effective cavity
    outer_diameter_A=16.0,
    cavity_sasa_A2=440.0,
    n_portal_HB_sites=14,
    portal_diameter_A=6.2,
    mw=1396.0,  # average, DS-dependent
    water_solubility_mM=500.0,  # >>β-CD, major advantage
    pKa_secondary_OH=12.1,
    commercial=True,
    cost_per_gram_usd=0.80,
    common_suppliers=["Sigma-Aldrich", "Roquette", "Ashland"],
    modification="HP",
    modification_description="Hydroxypropyl substitution at O-2, O-3, O-6",
    avg_degree_of_substitution=4.5,
    backsolve_MAE_logK=0.50,  # estimated, fewer calibration entries
    n_calibration_entries=5,
)

ME_BETA_CD = CDHost(
    name="Me-β-Cyclodextrin",
    base_type="beta",
    n_glucose=7,
    cavity_diameter_A=6.0,
    cavity_depth_A=8.2,
    cavity_volume_A3=280.0,
    outer_diameter_A=15.8,
    cavity_sasa_A2=435.0,
    n_portal_HB_sites=7,    # methylation removes half the OH groups
    portal_diameter_A=6.0,
    mw=1331.36,  # heptakis(2,6-di-O-methyl)
    water_solubility_mM=570.0,
    pKa_secondary_OH=12.5,  # remaining OHs
    commercial=True,
    cost_per_gram_usd=2.00,
    common_suppliers=["Sigma-Aldrich", "TCI"],
    modification="Me",
    modification_description="Methylation at O-2 and O-6",
    avg_degree_of_substitution=12.0,  # ~12/21 positions
    n_calibration_entries=3,
)

SBE_BETA_CD = CDHost(
    name="SBE-β-Cyclodextrin",
    base_type="beta",
    n_glucose=7,
    cavity_diameter_A=6.0,
    cavity_depth_A=8.8,
    cavity_volume_A3=300.0,
    outer_diameter_A=16.5,
    cavity_sasa_A2=450.0,
    n_portal_HB_sites=14,
    portal_diameter_A=6.5,
    mw=2163.0,  # Captisol average
    water_solubility_mM=700.0,  # extremely soluble
    pKa_secondary_OH=12.1,
    commercial=True,
    cost_per_gram_usd=15.00,  # pharmaceutical grade, expensive
    common_suppliers=["Ligand/Captisol", "Sigma-Aldrich"],
    modification="SBE",
    modification_description="Sulfobutyl ether at O-6 (anionic)",
    avg_degree_of_substitution=6.5,
    n_calibration_entries=2,
)

# ─────────────────────────────────────────────
# All hosts indexed
# ─────────────────────────────────────────────

CD_HOSTS: dict[str, CDHost] = {
    "alpha-CD": ALPHA_CD,
    "beta-CD": BETA_CD,
    "gamma-CD": GAMMA_CD,
    "HP-beta-CD": HP_BETA_CD,
    "Me-beta-CD": ME_BETA_CD,
    "SBE-beta-CD": SBE_BETA_CD,
}

NATIVE_CDS: list[CDHost] = [ALPHA_CD, BETA_CD, GAMMA_CD]

ALL_CDS: list[CDHost] = list(CD_HOSTS.values())

# ─────────────────────────────────────────────
# Modification library
# ─────────────────────────────────────────────

CD_MODIFICATIONS: list[CDModification] = [
    CDModification(
        name="Hydroxypropylation",
        abbreviation="HP",
        description="Hydroxypropyl groups at O-2, O-3, O-6. "
                    "Dramatically improves water solubility while preserving cavity.",
        effect_on_cavity="slightly widens",
        effect_on_solubility="increases",
        effect_on_selectivity="slightly reduces (DS heterogeneity)",
        synthesis_complexity="moderate",
        commercial_availability=True,
        typical_DS=4.5,
        applicable_to=["alpha", "beta", "gamma"],
    ),
    CDModification(
        name="Methylation",
        abbreviation="Me",
        description="Methyl groups at O-2 and O-6. "
                    "Extends cavity depth, makes more hydrophobic interior.",
        effect_on_cavity="deepens",
        effect_on_solubility="increases",
        effect_on_selectivity="enhances hydrophobic selectivity",
        synthesis_complexity="moderate",
        commercial_availability=True,
        typical_DS=12.0,
        applicable_to=["alpha", "beta", "gamma"],
    ),
    CDModification(
        name="Sulfobutyl ether",
        abbreviation="SBE",
        description="Anionic sulfobutyl ether groups at O-6. "
                    "Adds electrostatic interactions, high solubility.",
        effect_on_cavity="extends with flexible arms",
        effect_on_solubility="increases",
        effect_on_selectivity="enhances cation binding via charge",
        synthesis_complexity="complex",
        commercial_availability=True,
        typical_DS=6.5,
        applicable_to=["beta"],
    ),
    CDModification(
        name="Amino substitution",
        abbreviation="NH2",
        description="Primary amine at C-6. Cationic at neutral pH. "
                    "Click chemistry handle via reductive amination.",
        effect_on_cavity="unchanged",
        effect_on_solubility="increases",
        effect_on_selectivity="adds anion binding capability",
        synthesis_complexity="moderate",
        commercial_availability=True,
        typical_DS=1.0,
        applicable_to=["alpha", "beta", "gamma"],
    ),
    CDModification(
        name="Tosylation (C6-OTs)",
        abbreviation="OTs",
        description="Tosylate at C-6 primary OH. Key intermediate for "
                    "azide, thiol, amine conversions. Click handle precursor.",
        effect_on_cavity="unchanged",
        effect_on_solubility="decreases",
        effect_on_selectivity="unchanged (handle, not binding modification)",
        synthesis_complexity="trivial",
        commercial_availability=True,
        typical_DS=1.0,
        applicable_to=["alpha", "beta", "gamma"],
    ),
    CDModification(
        name="Azide (C6-N3)",
        abbreviation="N3",
        description="Azide at C-6 via tosylate displacement. "
                    "CuAAC click handle for conjugation to scaffolds.",
        effect_on_cavity="unchanged",
        effect_on_solubility="slightly decreases",
        effect_on_selectivity="unchanged (conjugation handle)",
        synthesis_complexity="moderate",
        commercial_availability=True,
        typical_DS=1.0,
        applicable_to=["alpha", "beta", "gamma"],
    ),
]


# ─────────────────────────────────────────────
# BackSolve calibrated parameters for CD binding
# ─────────────────────────────────────────────

BACKSOLVE_CD_PARAMS = {
    # From Phase 6-10 joint calibration (R²=0.850, MAE=0.74)
    "gamma_hydrophobic": -0.0291,       # kJ/mol per Å² buried SASA
    "eps_neutral_hbond": -8.027,        # kJ/mol per neutral H-bond
    "eps_charge_assisted_hbond": -7.173,# kJ/mol per charge-assisted H-bond
    "water_penalty_per_hb": 3.995,      # kJ/mol per H-bond to displace water
    "water_displacement_bonus": 1.332,  # kJ/mol per high-energy water displaced
    "eps_rotor": 2.481,                 # kJ/mol per frozen rotor (penalty)
    "f_partial_freeze": 0.534,          # fraction of rotors that freeze
    "k_shape": -4.997,                  # shape complementarity coefficient
    "PC_optimal": 0.591,                # optimal packing coefficient (Rebek ~0.55)
    "sigma_PC": 0.050,                  # packing coefficient Gaussian width
}


def select_best_cd(
    guest_volume_A3: float,
    guest_is_charged: bool = False,
    require_water_soluble: bool = True,
    require_click_handle: bool = False,
) -> list[tuple[CDHost, float, str]]:
    """
    Quick CD selection from guest volume using Rebek's 55% packing rule.

    Returns list of (CDHost, packing_coefficient, rationale) sorted by
    how close PC is to optimal.
    """
    results = []
    candidates = ALL_CDS if not require_water_soluble else [
        cd for cd in ALL_CDS if cd.water_solubility_mM > 50.0
    ]

    for cd in candidates:
        pc = guest_volume_A3 / cd.cavity_volume_A3
        # Rebek's rule: optimal packing ~0.55-0.59
        deviation = abs(pc - BACKSOLVE_CD_PARAMS["PC_optimal"])

        rationale_parts = [f"PC={pc:.2f}"]
        if pc < 0.3:
            rationale_parts.append("guest too small — poor shape complementarity")
        elif pc > 0.8:
            rationale_parts.append("guest too large — steric clash likely")
        elif deviation < 0.05:
            rationale_parts.append("excellent packing (Rebek sweet spot)")
        elif deviation < 0.10:
            rationale_parts.append("good packing")
        else:
            rationale_parts.append("suboptimal packing")

        if guest_is_charged and cd.modification == "SBE":
            rationale_parts.append("SBE anionic arms complement cationic guest")

        results.append((cd, pc, "; ".join(rationale_parts)))

    # Sort by closeness to optimal PC
    results.sort(key=lambda x: abs(x[1] - BACKSOLVE_CD_PARAMS["PC_optimal"]))
    return results
