"""
Optical Interaction Knowledge Base.

Physics basis:
    - Beer-Lambert law: A = ε·c·l (absorbance = absorptivity × concentration × path)
    - Visible spectrum: 380-780 nm
    - Color perception: absorbed wavelength → perceived complementary color
    - Electronic transitions: d-d, charge transfer, π-π*, n-π*

Data sources (all Tier 1/2):
    1. NIST Atomic Spectra Database (wavelength standards)
    2. Visible spectrum → perceived color mapping (physics, not empirical)
    3. Common indicator dye properties (well-established in analytical chemistry)

KEY ADVANTAGE: Optical predictions are self-validating.
If we predict a compound absorbs at 450 nm (blue absorption → orange appearance),
the prediction is verified or falsified by looking at the sample.
This provides rapid feedback with zero instrumentation beyond human eyes.

Refractive index data: refractiveindex.info (Tier 1-equivalent, primary literature aggregator)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────
# Data provenance
# ─────────────────────────────────────────────

DATA_SOURCES = [
    {
        "id": "visible_spectrum_physics",
        "title": "Visible light wavelength-color correspondence",
        "source": "CIE standard observer / physics",
        "data_quality_tier": 1,
        "data_type": "wavelength to perceived color mapping",
        "note": "Derived from physics of human color perception, not empirical fit.",
    },
    {
        "id": "common_indicators",
        "title": "Common metal indicator dyes — absorption properties",
        "source": "Standard analytical chemistry references (Skoog, West, Holler)",
        "data_quality_tier": 2,
        "data_type": "molar absorptivity, wavelength, color change",
    },
    {
        "id": "refractiveindex_info",
        "title": "refractiveindex.info — Refractive index database",
        "url": "https://refractiveindex.info",
        "data_quality_tier": 1,
        "data_type": "n,k optical constants from primary literature",
        "note": "Aggregates published measurements with full citations.",
    },
]


# ─────────────────────────────────────────────
# Wavelength ↔ Color mapping (physics)
# ─────────────────────────────────────────────

# Absorbed wavelength → color absorbed → perceived (complementary) color
# This is physics, not empirical data.
WAVELENGTH_TO_COLOR: list[tuple[float, float, str, str]] = [
    # (λ_min, λ_max, absorbed_color, perceived_complementary)
    (380, 435, "violet",        "yellow-green"),
    (435, 480, "blue",          "orange"),
    (480, 500, "cyan",          "red"),
    (500, 560, "green",         "magenta/purple"),
    (560, 580, "yellow-green",  "violet"),
    (580, 595, "yellow",        "blue"),
    (595, 610, "orange",        "cyan-blue"),
    (610, 780, "red",           "cyan-green"),
]


def wavelength_to_absorbed_color(wavelength_nm: float) -> str:
    """Map wavelength to the name of the absorbed color."""
    for lmin, lmax, color, _ in WAVELENGTH_TO_COLOR:
        if lmin <= wavelength_nm < lmax:
            return color
    if wavelength_nm < 380:
        return "ultraviolet"
    return "infrared"


def wavelength_to_perceived_color(wavelength_nm: float) -> str:
    """Map absorbed wavelength to the perceived (complementary) color."""
    for lmin, lmax, _, perceived in WAVELENGTH_TO_COLOR:
        if lmin <= wavelength_nm < lmax:
            return perceived
    if wavelength_nm < 380:
        return "colorless (UV absorption)"
    return "colorless (IR absorption)"


def is_visible(wavelength_nm: float) -> bool:
    return 380 <= wavelength_nm <= 780


# ─────────────────────────────────────────────
# Beer-Lambert law
# ─────────────────────────────────────────────

def beer_lambert_absorbance(
    epsilon_L_mol_cm: float,
    concentration_M: float,
    pathlength_cm: float = 1.0,
) -> float:
    """
    A = ε·c·l

    Args:
        epsilon_L_mol_cm: molar absorptivity (L/(mol·cm))
        concentration_M: concentration (mol/L)
        pathlength_cm: optical path length (cm)

    Returns:
        Absorbance (dimensionless, typically 0-3)
    """
    if epsilon_L_mol_cm < 0 or concentration_M < 0 or pathlength_cm < 0:
        return 0.0
    return epsilon_L_mol_cm * concentration_M * pathlength_cm


def absorbance_to_transmittance(absorbance: float) -> float:
    """T = 10^(-A). Returns fraction (0-1)."""
    if absorbance < 0:
        return 1.0
    return 10.0 ** (-absorbance)


def transmittance_to_absorbance(transmittance: float) -> float:
    """A = -log10(T)."""
    if transmittance <= 0:
        return float("inf")
    if transmittance >= 1:
        return 0.0
    return -math.log10(transmittance)


def required_concentration_M(
    target_absorbance: float,
    epsilon_L_mol_cm: float,
    pathlength_cm: float = 1.0,
) -> float:
    """Back-calculate concentration needed for target absorbance."""
    if epsilon_L_mol_cm <= 0 or pathlength_cm <= 0:
        return float("inf")
    return target_absorbance / (epsilon_L_mol_cm * pathlength_cm)


# ─────────────────────────────────────────────
# Common chromophore/indicator properties
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class ChromophoreEntry:
    """Properties of a common chromophore."""
    name: str
    chromophore_class: str          # "porphyrin", "azo", "indicator", etc.
    lambda_max_nm: float            # Peak absorption wavelength
    epsilon_L_mol_cm: float         # Molar absorptivity at λmax
    color_absorbed: str
    color_perceived: str
    transition_type: str            # "d-d", "CT", "π-π*", "n-π*"
    solvent: str = "water"
    ph_sensitive: bool = False
    notes: str = ""


# Well-established values from analytical chemistry textbooks.
# These are representative/typical values, not unique to one measurement.
COMMON_CHROMOPHORES: list[ChromophoreEntry] = [
    # Metal-porphyrin complexes (MABE's strength)
    ChromophoreEntry(
        name="Free-base porphyrin (H2TPP)",
        chromophore_class="porphyrin",
        lambda_max_nm=419,      # Soret band
        epsilon_L_mol_cm=4.7e5, # Very high ε for Soret
        color_absorbed="blue-violet",
        color_perceived="yellow-orange",
        transition_type="π-π*",
        solvent="CHCl3",
        notes="Soret band. Q bands at 515, 550, 590, 645 nm.",
    ),
    ChromophoreEntry(
        name="Zn-TPP (zinc porphyrin)",
        chromophore_class="porphyrin",
        lambda_max_nm=421,
        epsilon_L_mol_cm=5.5e5,
        color_absorbed="blue-violet",
        color_perceived="pink-red",
        transition_type="π-π*",
        solvent="CHCl3",
        notes="Metalation shifts Soret ~2 nm, increases ε.",
    ),
    ChromophoreEntry(
        name="Cu-TPP (copper porphyrin)",
        chromophore_class="porphyrin",
        lambda_max_nm=416,
        epsilon_L_mol_cm=3.8e5,
        color_absorbed="violet",
        color_perceived="yellow-green",
        transition_type="π-π*",
        solvent="CHCl3",
        notes="Cu2+ causes hypsochromic shift.",
    ),

    # Common metal indicators
    ChromophoreEntry(
        name="Xylenol Orange",
        chromophore_class="indicator",
        lambda_max_nm=433,      # Free form
        epsilon_L_mol_cm=2.4e4,
        color_absorbed="blue",
        color_perceived="yellow (free) → red-violet (metal complex)",
        transition_type="CT",
        solvent="water",
        ph_sensitive=True,
        notes="λmax shifts to 570-580 nm on metal binding. Classic Pb2+ indicator.",
    ),
    ChromophoreEntry(
        name="Murexide",
        chromophore_class="indicator",
        lambda_max_nm=520,      # Free form
        epsilon_L_mol_cm=1.3e4,
        color_absorbed="green",
        color_perceived="red-violet (free) → yellow (Ca2+) → orange (Cu2+)",
        transition_type="CT",
        solvent="water",
        ph_sensitive=True,
        notes="Classic Ca2+, Cu2+, Ni2+ indicator.",
    ),
    ChromophoreEntry(
        name="1-(2-pyridylazo)-2-naphthol (PAN)",
        chromophore_class="azo",
        lambda_max_nm=470,
        epsilon_L_mol_cm=3.4e4,
        color_absorbed="blue",
        color_perceived="yellow (free) → red (metal complex)",
        transition_type="π-π* / CT",
        solvent="water/ethanol",
        ph_sensitive=True,
        notes="Versatile metal indicator. λmax shifts 20-50 nm on metal binding.",
    ),

    # Simple reference chromophores
    ChromophoreEntry(
        name="KMnO4 (permanganate)",
        chromophore_class="inorganic",
        lambda_max_nm=525,
        epsilon_L_mol_cm=2.4e3,
        color_absorbed="green",
        color_perceived="purple",
        transition_type="CT (LMCT)",
        solvent="water",
        notes="Classic Beer-Lambert reference. Color intensity proportional to [MnO4-].",
    ),
    ChromophoreEntry(
        name="CuSO4·5H2O (aqueous)",
        chromophore_class="inorganic",
        lambda_max_nm=810,
        epsilon_L_mol_cm=13.0,
        color_absorbed="near-IR",
        color_perceived="blue",
        transition_type="d-d",
        solvent="water",
        notes="Weak d-d absorption. Classic blue solution. Low ε typical for d-d.",
    ),
]

CHROMOPHORE_BY_NAME: dict[str, ChromophoreEntry] = {
    c.name: c for c in COMMON_CHROMOPHORES
}


# ─────────────────────────────────────────────
# Color matching utility
# ─────────────────────────────────────────────

def find_chromophores_for_color(
    target_color: str,
) -> list[ChromophoreEntry]:
    """Find chromophores that produce a target perceived color."""
    target = target_color.lower()
    matches = []
    for c in COMMON_CHROMOPHORES:
        if target in c.color_perceived.lower():
            matches.append(c)
    return matches


def find_chromophores_for_wavelength(
    wavelength_nm: float,
    tolerance_nm: float = 30.0,
) -> list[ChromophoreEntry]:
    """Find chromophores with λmax near the target wavelength."""
    matches = []
    for c in COMMON_CHROMOPHORES:
        if abs(c.lambda_max_nm - wavelength_nm) <= tolerance_nm:
            matches.append(c)
    return matches