"""
Ion Exchange Resin Selectivity Knowledge Base.

All data from:
    DuPont Water Solutions, "Ion Exchange Resins Selectivity"
    Tech Fact, Form No. 45-D01458-en, Rev. 2, November 2019.
    URL: https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/IER-Selectivity-TechFact-45-D01458-en.pdf

Data quality: Tier 2 (published manufacturer technical document, retrievable PDF).
Selectivity coefficients are relative affinities vs. H+ (cation) or OH- (anion) = 1.0.

Convention: Gaines-Thomas selectivity coefficients on sulfonated polystyrene
(SAC), carboxylic (WAC), or quaternary amine (SBA) resins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data provenance
# ─────────────────────────────────────────────

DATA_SOURCE = {
    "title": "Ion Exchange Resins Selectivity",
    "publisher": "DuPont Water Solutions",
    "form_number": "45-D01458-en",
    "revision": "Rev. 2",
    "date": "November 2019",
    "url": "https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/IER-Selectivity-TechFact-45-D01458-en.pdf",
    "data_quality_tier": 2,
    "verified_by": "web_fetch_2026-02-28",
    "verification_note": "Full PDF fetched and values extracted directly from document text.",
}


# ─────────────────────────────────────────────
# SAC resin: relative affinities vs H+ = 1.0
# Sulfonated polystyrene, varying DVB cross-linking
# DuPont Tech Fact Table 1 (page 1)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class SACSelectivity:
    """Selectivity coefficient for a cation on SAC resin at a given cross-linking."""
    ion: str
    charge: int
    dvb_4pct: float
    dvb_8pct: float
    dvb_10pct: float
    dvb_16pct: float


SAC_SELECTIVITY: list[SACSelectivity] = [
    # Monovalent cations
    SACSelectivity("Li+",   1,  0.76,  0.79,  0.77,  0.68),
    SACSelectivity("H+",    1,  1.00,  1.00,  1.00,  1.00),
    SACSelectivity("Na+",   1,  1.20,  1.56,  1.61,  1.62),
    SACSelectivity("NH4+",  1,  1.44,  2.01,  2.15,  2.27),
    SACSelectivity("K+",    1,  1.72,  2.28,  2.54,  3.06),
    SACSelectivity("Rb+",   1,  1.86,  2.49,  2.69,  3.14),
    SACSelectivity("Cs+",   1,  2.02,  2.56,  2.77,  3.17),
    SACSelectivity("Ag+",   1,  3.58,  6.70,  8.15, 15.60),
    SACSelectivity("Tl+",   1,  5.08,  9.76, 12.60, 19.40),
    # Divalent cations
    SACSelectivity("UO2_2+", 2, 1.79,  1.93,  2.00,  2.27),
    SACSelectivity("Mg2+",  2,  2.23,  2.59,  2.62,  2.39),
    SACSelectivity("Zn2+",  2,  2.37,  2.73,  2.77,  2.57),
    SACSelectivity("Co2+",  2,  2.45,  2.94,  2.92,  2.59),
    SACSelectivity("Cu2+",  2,  2.49,  3.03,  3.15,  3.03),
    SACSelectivity("Cd2+",  2,  2.55,  3.06,  3.23,  3.37),
    SACSelectivity("Ni2+",  2,  2.61,  3.09,  3.08,  2.76),
    SACSelectivity("Ca2+",  2,  3.14,  4.06,  4.42,  4.95),
    SACSelectivity("Sr2+",  2,  3.56,  5.13,  5.85,  6.87),
    SACSelectivity("Pb2+",  2,  4.97,  7.80,  8.92, 12.20),
    SACSelectivity("Ba2+",  2,  5.66,  9.06,  9.42, 14.20),
]

# Quick lookup: ion name → SACSelectivity
SAC_BY_ION: dict[str, SACSelectivity] = {s.ion: s for s in SAC_SELECTIVITY}


# ─────────────────────────────────────────────
# SBA resin: relative affinities vs OH- = 1.0
# Polystyrenic quaternary amine, Type 1 and Type 2
# DuPont Tech Fact Table 2 (page 2)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class SBASelectivity:
    """Selectivity coefficient for an anion on SBA resin."""
    ion: str
    charge: int
    type1: float
    type2: float


SBA_SELECTIVITY: list[SBASelectivity] = [
    SBASelectivity("OH-",              -1,    1.0,    1.0),
    SBASelectivity("benzene_sulfonate", -1, 500.0,   75.0),
    SBASelectivity("salicylate",       -1,  450.0,   65.0),
    SBASelectivity("citrate",          -3,  220.0,   23.0),
    SBASelectivity("I-",               -1,  175.0,   17.0),
    SBASelectivity("phenate",          -1,  110.0,   27.0),
    SBASelectivity("HSO4-",            -1,   85.0,   15.0),
    SBASelectivity("ClO3-",            -1,   74.0,   12.0),
    SBASelectivity("NO3-",             -1,   65.0,    8.0),
    SBASelectivity("Br-",              -1,   50.0,    6.0),
    SBASelectivity("CN-",              -1,   28.0,    3.0),
    SBASelectivity("HSO3-",            -1,   27.0,    3.0),
    SBASelectivity("BrO3-",            -1,   27.0,    3.0),
    SBASelectivity("NO2-",             -1,   24.0,    3.0),
    SBASelectivity("Cl-",              -1,   22.0,    2.3),
    SBASelectivity("HCO3-",            -1,    6.0,    1.2),
    SBASelectivity("IO3-",             -1,    5.5,    0.5),
    SBASelectivity("formate",          -1,    4.6,    0.5),
    SBASelectivity("acetate",          -1,    3.2,    0.5),
    SBASelectivity("propionate",       -1,    2.6,    0.3),
    SBASelectivity("F-",               -1,    1.6,    0.3),
    SBASelectivity("H2PO4-",           -1,    5.0,    0.5),
]

SBA_BY_ION: dict[str, SBASelectivity] = {s.ion: s for s in SBA_SELECTIVITY}


# ─────────────────────────────────────────────
# WAC resin: relative affinities vs Ca2+ = 1.0
# Carboxylic resin (R-COOH)
# DuPont Tech Fact Table 3 (page 3)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class WACSelectivity:
    """Selectivity coefficient for a cation on WAC resin (vs Ca2+ = 1.0)."""
    ion: str
    charge: int
    relative_to_Ca: float


WAC_SELECTIVITY: list[WACSelectivity] = [
    WACSelectivity("Mg2+", 2, 0.3),
    WACSelectivity("Sr2+", 2, 0.9),   # Listed as "< 1" — conservative 0.9
    WACSelectivity("Ba2+", 2, 0.9),   # Listed as "< 1" — conservative 0.9
    WACSelectivity("Ca2+", 2, 1.0),
    WACSelectivity("Cd2+", 2, 1.0),
    WACSelectivity("Ni2+", 2, 1.4),
    WACSelectivity("Zn2+", 2, 1.5),
    WACSelectivity("Co2+", 2, 1.9),
    WACSelectivity("Cu2+", 2, 2.0),
    WACSelectivity("Pb2+", 2, 1.5),   # Listed as "> 1" — conservative 1.5
]

WAC_BY_ION: dict[str, WACSelectivity] = {s.ion: s for s in WAC_SELECTIVITY}

# Note: WAC Sr2+ and Ba2+ are listed as "< 1" in the source.
# WAC Pb2+ is listed as "> 1". Conservative point estimates used.
# These should be flagged if used for quantitative predictions.
WAC_ESTIMATED_ENTRIES = {"Sr2+", "Ba2+", "Pb2+"}


# ─────────────────────────────────────────────
# Typical resin properties
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class ResinProfile:
    """Physical and operational characteristics of a resin class."""
    resin_type: str              # "SAC", "WAC", "SBA", "WBA"
    functional_group: str        # Chemical name
    typical_capacity_meq_mL: float  # wet-volume capacity
    ph_operating_range: tuple[float, float]
    regenerant: str
    typical_dvb_pct: float
    bead_size_range_mm: tuple[float, float]


RESIN_PROFILES: dict[str, ResinProfile] = {
    "SAC": ResinProfile(
        resin_type="SAC",
        functional_group="sulfonate (R-SO3H)",
        typical_capacity_meq_mL=1.8,   # typical for 8% DVB gel resin
        ph_operating_range=(0.0, 14.0),
        regenerant="HCl or H2SO4 (H-cycle), NaCl (Na-cycle)",
        typical_dvb_pct=8.0,
        bead_size_range_mm=(0.3, 1.2),
    ),
    "WAC": ResinProfile(
        resin_type="WAC",
        functional_group="carboxylate (R-COOH)",
        typical_capacity_meq_mL=3.5,   # higher capacity than SAC
        ph_operating_range=(5.0, 14.0),  # does not work in acidic pH
        regenerant="HCl or H2SO4",
        typical_dvb_pct=8.0,
        bead_size_range_mm=(0.3, 1.2),
    ),
    "SBA": ResinProfile(
        resin_type="SBA",
        functional_group="quaternary amine (R-NR3OH)",
        typical_capacity_meq_mL=1.2,
        ph_operating_range=(0.0, 14.0),
        regenerant="NaOH",
        typical_dvb_pct=8.0,
        bead_size_range_mm=(0.3, 1.2),
    ),
}


# ─────────────────────────────────────────────
# Lookup functions
# ─────────────────────────────────────────────

def get_sac_selectivity(
    ion: str,
    dvb_pct: float = 8.0,
) -> Optional[float]:
    """
    Get SAC selectivity coefficient for an ion at a given cross-linking.

    Returns None if ion not in database.
    Interpolates linearly between DVB percentages if not exact match.
    """
    entry = SAC_BY_ION.get(ion)
    if entry is None:
        return None

    dvb_levels = [4.0, 8.0, 10.0, 16.0]
    values = [entry.dvb_4pct, entry.dvb_8pct, entry.dvb_10pct, entry.dvb_16pct]

    if dvb_pct <= 4.0:
        return values[0]
    if dvb_pct >= 16.0:
        return values[3]

    # Linear interpolation
    for i in range(len(dvb_levels) - 1):
        if dvb_levels[i] <= dvb_pct <= dvb_levels[i + 1]:
            frac = (dvb_pct - dvb_levels[i]) / (dvb_levels[i + 1] - dvb_levels[i])
            return values[i] + frac * (values[i + 1] - values[i])

    return values[1]  # fallback: 8% DVB


def compute_separation_factor(
    target_ion: str,
    competing_ion: str,
    dvb_pct: float = 8.0,
) -> Optional[float]:
    """
    Compute separation factor α = K_target / K_competing for SAC resin.

    Returns None if either ion not in database.
    α > 1 means the resin prefers target over competitor.
    """
    k_target = get_sac_selectivity(target_ion, dvb_pct)
    k_competing = get_sac_selectivity(competing_ion, dvb_pct)
    if k_target is None or k_competing is None or k_competing == 0:
        return None
    return k_target / k_competing


def recommend_resin_type(target_ion: str, target_charge: int) -> str:
    """
    Recommend resin type based on target ion charge.

    Cations → SAC (broad pH) or WAC (high capacity, pH > 5).
    Anions → SBA.
    """
    if target_charge > 0:
        # Check if WAC has data for this ion — WAC has much higher capacity
        if target_ion in WAC_BY_ION:
            return "WAC_or_SAC"  # user decides based on pH
        return "SAC"
    elif target_charge < 0:
        return "SBA"
    else:
        return "CHELATING"  # neutral species need chelating resin