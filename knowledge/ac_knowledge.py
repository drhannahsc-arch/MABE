"""
Activated Carbon Adsorption Knowledge Base.

Data sources (all Tier 2 — peer-reviewed published literature):

1. Xiao & Thomas, Langmuir 2004, 20(11), 4566-4578.
   DOI: 10.1021/la049712j
   "Competitive Adsorption of Aqueous Metal Ions on an Oxidized
    Nanoporous Activated Carbon"
   - Competitive adsorption ordering: Hg2+ > Pb2+ > Cd2+ > Ca2+
   - Mechanism: ion exchange at acidic oxygen functional group sites

2. DuPont Water Solutions, "Filtrasorb Activated Carbon" technical data.
   General AC properties for granular activated carbon (GAC).

3. General activated carbon properties from IUPAC classifications
   and standard textbook values (pore size ranges, surface areas).

Physics basis:
    - Langmuir isotherm: qe = qmax * KL * Ce / (1 + KL * Ce)
    - Freundlich isotherm: qe = KF * Ce^(1/n)
    - Surface complexation: M2+ + ≡S-OH → ≡S-OM+ + H+
    - pH dependence via pHpzc (point of zero charge)

IMPORTANT: Langmuir parameters (qmax, KL) are MATERIAL-SPECIFIC.
They vary dramatically with AC source, activation method, surface
chemistry, and experimental conditions. The values here are
representative ranges for commercial GAC, NOT universal constants.
Every design MUST be validated with batch isotherm experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data provenance
# ─────────────────────────────────────────────

DATA_SOURCES = [
    {
        "id": "xiao_thomas_2004",
        "title": "Competitive Adsorption of Aqueous Metal Ions on an Oxidized Nanoporous Activated Carbon",
        "authors": "B. Xiao, K. M. Thomas",
        "journal": "Langmuir",
        "year": 2004,
        "volume": 20,
        "pages": "4566-4578",
        "doi": "10.1021/la049712j",
        "data_quality_tier": 2,
        "data_type": "competitive adsorption ordering, mechanism",
    },
    {
        "id": "general_ac_properties",
        "title": "Activated carbon pore classification (IUPAC)",
        "source": "IUPAC Recommendations 1994, Pure Appl. Chem. 66(8), 1739-1758",
        "data_quality_tier": 2,
        "data_type": "pore size classification",
    },
]


# ─────────────────────────────────────────────
# Competitive adsorption ordering
# From Xiao & Thomas 2004 (Langmuir)
# ─────────────────────────────────────────────

# Verified ordering for oxidized nanoporous AC.
# This ordering is consistent across single-ion and binary mixture experiments.
# Mechanism: ion exchange at acidic oxygen functional groups.
# Higher rank = stronger adsorption.
METAL_ADSORPTION_ORDER_OXIDIZED_AC = [
    # (ion, relative_rank, notes)
    ("Hg2+", 4, "Strongest — hydrolysis enhances surface interaction"),
    ("Pb2+", 3, "Strong — electronegativity + ionic radius favorable"),
    ("Cd2+", 2, "Moderate — intermediate hydration energy"),
    ("Ca2+", 1, "Weakest of tested divalents — high hydration energy"),
]

# General selectivity trends on activated carbon (well-established in literature):
# - Divalent > monovalent (charge effect)
# - Among divalents: correlates with electronegativity and hydration energy
# - pH strongly affects speciation and surface charge
# - Organic contaminants: correlate with hydrophobicity (log P)
GENERAL_SELECTIVITY_PRINCIPLES = {
    "charge_effect": "Higher charge → stronger electrostatic attraction",
    "electronegativity": "Higher electronegativity → stronger surface complexation",
    "hydration_energy": "Lower hydration energy → easier desolvation → better adsorption",
    "ionic_radius": "Optimal fit in micropores enhances selectivity",
    "pH_dependence": "Adsorption increases with pH until precipitation; pHpzc critical",
}


# ─────────────────────────────────────────────
# Activated carbon material properties
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class ACMaterialProfile:
    """Properties of an activated carbon type."""
    name: str
    source: str                        # "coconut shell", "coal", "wood", etc.
    activation: str                    # "steam", "chemical (H3PO4)", "chemical (ZnCl2)"
    bet_surface_area_m2_g: tuple[float, float]  # (min, max) typical range
    micropore_volume_cm3_g: float      # typical
    total_pore_volume_cm3_g: float     # typical
    ph_pzc: float                      # point of zero charge
    typical_capacity_heavy_metals_mg_g: tuple[float, float]  # (min, max) range
    cost_per_kg_usd: tuple[float, float]  # (min, max) range
    notes: str


# Representative AC profiles — these are RANGES, not point values
AC_PROFILES: dict[str, ACMaterialProfile] = {
    "coconut_shell_gac": ACMaterialProfile(
        name="Coconut Shell GAC",
        source="coconut shell",
        activation="steam",
        bet_surface_area_m2_g=(900.0, 1200.0),
        micropore_volume_cm3_g=0.45,
        total_pore_volume_cm3_g=0.55,
        ph_pzc=7.0,
        typical_capacity_heavy_metals_mg_g=(5.0, 50.0),
        cost_per_kg_usd=(1.5, 4.0),
        notes="High micropore fraction, good for small molecules and metals",
    ),
    "coal_gac": ACMaterialProfile(
        name="Coal-based GAC",
        source="bituminous coal",
        activation="steam",
        bet_surface_area_m2_g=(800.0, 1100.0),
        micropore_volume_cm3_g=0.35,
        total_pore_volume_cm3_g=0.60,
        ph_pzc=7.5,
        typical_capacity_heavy_metals_mg_g=(5.0, 40.0),
        cost_per_kg_usd=(1.0, 3.0),
        notes="Broad pore distribution, versatile",
    ),
    "wood_gac": ACMaterialProfile(
        name="Wood-based GAC",
        source="wood (hardwood)",
        activation="chemical (H3PO4)",
        bet_surface_area_m2_g=(1000.0, 1800.0),
        micropore_volume_cm3_g=0.30,
        total_pore_volume_cm3_g=0.80,
        ph_pzc=3.5,
        typical_capacity_heavy_metals_mg_g=(10.0, 80.0),
        cost_per_kg_usd=(2.0, 5.0),
        notes="High mesopore fraction, acidic pHpzc favors cation adsorption at lower pH",
    ),
    "oxidized_gac": ACMaterialProfile(
        name="Oxidized GAC",
        source="various (post-oxidation treatment)",
        activation="oxidation (HNO3, H2O2, or air)",
        bet_surface_area_m2_g=(600.0, 1000.0),
        micropore_volume_cm3_g=0.25,
        total_pore_volume_cm3_g=0.50,
        ph_pzc=3.0,
        typical_capacity_heavy_metals_mg_g=(20.0, 150.0),
        cost_per_kg_usd=(3.0, 8.0),
        notes="Enhanced surface oxygen groups → better metal adsorption. "
              "Lower pHpzc. Xiao & Thomas 2004 used this type.",
    ),
}


# ─────────────────────────────────────────────
# IUPAC Pore Classification
# ─────────────────────────────────────────────

PORE_CLASSIFICATION = {
    "micropore": {"diameter_A": (0, 20), "role": "Small molecules, metal ions"},
    "mesopore": {"diameter_A": (20, 500), "role": "Larger organic molecules, dyes"},
    "macropore": {"diameter_A": (500, float("inf")), "role": "Transport pores, access"},
}


# ─────────────────────────────────────────────
# Isotherm physics functions
# ─────────────────────────────────────────────

def langmuir_qe(Ce_mg_L: float, qmax_mg_g: float, KL_L_mg: float) -> float:
    """
    Langmuir isotherm: qe = qmax * KL * Ce / (1 + KL * Ce)

    Args:
        Ce_mg_L: equilibrium concentration (mg/L)
        qmax_mg_g: maximum monolayer capacity (mg/g)
        KL_L_mg: Langmuir constant (L/mg)

    Returns:
        qe: equilibrium adsorption capacity (mg/g)
    """
    if Ce_mg_L < 0 or qmax_mg_g <= 0 or KL_L_mg <= 0:
        return 0.0
    return qmax_mg_g * KL_L_mg * Ce_mg_L / (1.0 + KL_L_mg * Ce_mg_L)


def freundlich_qe(Ce_mg_L: float, KF: float, n: float) -> float:
    """
    Freundlich isotherm: qe = KF * Ce^(1/n)

    Args:
        Ce_mg_L: equilibrium concentration (mg/L)
        KF: Freundlich capacity constant (mg/g)(L/mg)^(1/n)
        n: Freundlich intensity constant (dimensionless, typically 1-10)

    Returns:
        qe: equilibrium adsorption capacity (mg/g)
    """
    if Ce_mg_L <= 0 or KF <= 0 or n <= 0:
        return 0.0
    return KF * Ce_mg_L ** (1.0 / n)


def langmuir_separation_factor(C0_mg_L: float, KL_L_mg: float) -> float:
    """
    Dimensionless separation factor RL = 1 / (1 + KL * C0).

    RL > 1: unfavorable
    RL = 1: linear
    0 < RL < 1: favorable
    RL = 0: irreversible
    """
    if C0_mg_L <= 0 or KL_L_mg <= 0:
        return 1.0
    return 1.0 / (1.0 + KL_L_mg * C0_mg_L)


def ph_adsorption_factor(pH: float, ph_pzc: float, target_charge: int) -> float:
    """
    Estimate pH effect on adsorption.

    For cations: adsorption increases when pH > pHpzc (surface negative).
    For anions: adsorption increases when pH < pHpzc (surface positive).

    Returns a 0-1 factor (1.0 = optimal pH, 0.0 = very unfavorable).
    This is a first-order approximation — real behavior is more complex.
    """
    if target_charge > 0:
        # Cation: want pH > pHpzc but below precipitation
        delta = pH - ph_pzc
        if delta < -3:
            return 0.1  # very unfavorable — surface positive
        elif delta < 0:
            return 0.3 + 0.2 * (delta + 3) / 3  # improving
        elif delta < 3:
            return 0.5 + 0.5 * delta / 3  # favorable
        else:
            return 0.9  # very favorable but may precipitate
    elif target_charge < 0:
        # Anion: want pH < pHpzc
        delta = ph_pzc - pH
        if delta < -3:
            return 0.1
        elif delta < 0:
            return 0.3 + 0.2 * (delta + 3) / 3
        elif delta < 3:
            return 0.5 + 0.5 * delta / 3
        else:
            return 0.9
    else:
        # Neutral: hydrophobic interaction, less pH dependent
        return 0.7


def recommend_ac_type(
    target_species: str,
    target_charge: int,
    target_mw: float,
) -> str:
    """
    Recommend AC type based on target properties.

    Heavy metals → oxidized GAC (enhanced surface groups)
    Small organics → coconut shell (high micropore)
    Large organics → wood-based (high mesopore)
    """
    if target_charge != 0:
        # Ionic species — surface chemistry matters most
        return "oxidized_gac"
    elif target_mw < 200:
        return "coconut_shell_gac"
    else:
        return "wood_gac"