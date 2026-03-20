"""
core/degradation_safety.py — Designed Degradability Module

Environmental safety scoring for capture-transform scaffolds.
Every scaffold gets a Degradation Safety Score (DSS) based on:
  - Environmental persistence half-life
  - Degradation product toxicity
  - Metal ion toxicity class
  - Organic fragment biodegradability

Hard constraints:
  - DSS < 0.3 → excluded from pipeline output
  - Cr-containing scaffolds → banned (Cr⁶⁺ carcinogenicity)
  - Free Pd²⁺ release without chelating mitigation → banned

Fail-safe linker architecture:
  Substrate → [click] → [FAIL-SAFE] → [click] → [Scaffold]
  The fail-safe is the designed weak point — breaks under environmental
  conditions, releasing scaffold for self-destruct via intrinsic lability.

References:
  - Levy-Booth et al., Soil Biol. Biochem. 2007 (DNA persistence)
  - Dell'Anno & Danovaro, Appl. Environ. Microbiol. 2005 (DNA in seawater)
  - Zimmermann et al., Environ. Toxicol. Chem. 2017 (Pd²⁺ aquatic toxicity)
  - EPA ECOTOX database (metal EC50 values)

Data tier: Tier 2 (DOI per toxicity value).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class MetalToxicity(str, Enum):
    """Metal ion toxicity classification for aquatic release."""
    NONE = "none"               # no metal (DNA, organic cage)
    BENIGN = "benign"           # Fe²⁺/Fe³⁺ — natural background, precipitates
    LOW = "low"                 # Zr⁴⁺ — very low solubility (ZrO₂)
    MODERATE_LOW = "moderate_low"  # Zn²⁺ — essential nutrient but toxic at high levels
    MODERATE = "moderate"       # Cu²⁺ — EC50 ~10-100 µg/L
    HIGH = "high"               # Pd²⁺ — EC50 ~13 µg/L
    BANNED = "banned"           # Cr (Cr⁶⁺ carcinogen under oxidizing conditions)


class FailSafeTrigger(str, Enum):
    """Environmental trigger for fail-safe linker cleavage."""
    ACID = "acid"                     # pH < 5 (acetal, ortho-ester, hydrazone)
    HYDROLYSIS = "hydrolysis"         # neutral water over weeks (ester)
    UV = "uv"                         # sunlight / UV (o-nitrobenzyl)
    DIOL = "diol"                     # fructose/glucose competition (boronate ester)
    REDUCING = "reducing"             # thiols in environment (disulfide)
    THERMAL = "thermal"               # retro-Diels-Alder at >60°C


class DeploymentMatrix(str, Enum):
    """Deployment environment for fail-safe linker selection."""
    MINE_DRAINAGE = "mine_drainage"       # pH 3-5, metals, sulfate
    WASTEWATER = "wastewater"             # pH 6-8, organics, nutrients
    SEAWATER = "seawater"                 # pH 8.1, UV at surface, high ionic
    GROUNDWATER = "groundwater"           # pH 6-7, dark, low organic
    FLUE_GAS = "flue_gas"                 # acidic condensate
    SURFACE_WATER = "surface_water"       # pH 6-8, UV, moderate organic
    TEXTILE = "textile"                   # sweat pH 5.5, sunlight, wash water
    AIR = "air"                           # direct air capture, UV exposure


# ═══════════════════════════════════════════════════════════════════════════
# Degradation profile
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DegradationProfile:
    """Environmental degradation profile for a scaffold system."""
    scaffold_system: str
    primary_trigger: str                  # "enzymatic", "pH", "hydrolysis", "UV"
    trigger_conditions: str               # "DNase in natural water", "pH < 5"
    half_life_description: str            # "4-24 hours in river water"
    half_life_hours: float                # numeric estimate for scoring
    degradation_products: list[str]       # ["nucleotides", "PO₄³⁻", "Mg²⁺"]
    metal_released: str                   # "none", "Fe²⁺", "Zn²⁺", "Pd²⁺"
    metal_toxicity: MetalToxicity
    fragment_biodegradable: bool
    recommended_failsafe: str             # "ester", "acetal", "boronate_ester"
    design_modifications: list[str]       # required mods for safe deployment
    hard_excluded: bool = False           # True → never deploy in field
    exclusion_reason: str = ""

    # Sub-scores (0-1, higher = safer)
    persistence_score: float = 0.0
    product_toxicity_score: float = 0.0
    metal_score: float = 0.0
    fragment_score: float = 0.0
    dss: float = 0.0                      # Degradation Safety Score (composite)

    def summary(self) -> str:
        status = "EXCLUDED" if self.hard_excluded else \
                 "SAFE" if self.dss >= 0.7 else \
                 "CAUTION" if self.dss >= 0.5 else "UNSAFE"
        lines = [
            f"Degradation Safety: {self.scaffold_system}",
            f"  DSS: {self.dss:.2f} / 1.00 [{status}]",
            f"  Self-destruct: {self.primary_trigger} ({self.trigger_conditions})",
            f"  Half-life: {self.half_life_description}",
            f"  Metal released: {self.metal_released} ({self.metal_toxicity.value})",
            f"  Fragments biodegradable: {self.fragment_biodegradable}",
            f"  Fail-safe linker: {self.recommended_failsafe}",
        ]
        if self.hard_excluded:
            lines.append(f"  ⚠ EXCLUDED: {self.exclusion_reason}")
        if self.design_modifications:
            lines.append(f"  Required modifications:")
            for m in self.design_modifications:
                lines.append(f"    - {m}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# DSS computation
# ═══════════════════════════════════════════════════════════════════════════

# Weights for DSS sub-scores
_W_PERSISTENCE = 0.25
_W_PRODUCT_TOX = 0.35
_W_METAL = 0.30
_W_FRAGMENT = 0.10


def _persistence_score(half_life_hours: float) -> float:
    """Score environmental persistence (higher = degrades faster = safer)."""
    if half_life_hours <= 24:
        return 1.0
    elif half_life_hours <= 168:       # 1 week
        return 0.8
    elif half_life_hours <= 672:       # 4 weeks
        return 0.5
    elif half_life_hours <= 8760:      # 1 year
        return 0.2
    else:
        return 0.0


def _metal_toxicity_score(toxicity: MetalToxicity) -> float:
    """Score metal ion toxicity (higher = less toxic = safer)."""
    return {
        MetalToxicity.NONE: 1.0,
        MetalToxicity.BENIGN: 0.9,
        MetalToxicity.LOW: 0.8,
        MetalToxicity.MODERATE_LOW: 0.7,
        MetalToxicity.MODERATE: 0.3,
        MetalToxicity.HIGH: 0.1,
        MetalToxicity.BANNED: 0.0,
    }[toxicity]


def compute_dss(profile: DegradationProfile) -> float:
    """Compute Degradation Safety Score from sub-scores."""
    profile.persistence_score = _persistence_score(profile.half_life_hours)
    profile.metal_score = _metal_toxicity_score(profile.metal_toxicity)
    # product_toxicity_score and fragment_score set during profile construction

    dss = (_W_PERSISTENCE * profile.persistence_score +
           _W_PRODUCT_TOX * profile.product_toxicity_score +
           _W_METAL * profile.metal_score +
           _W_FRAGMENT * profile.fragment_score)

    if profile.hard_excluded:
        dss = 0.0

    profile.dss = round(dss, 4)
    return profile.dss


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold degradation database
# ═══════════════════════════════════════════════════════════════════════════

def _build_profiles() -> dict[str, DegradationProfile]:
    """Build degradation profiles for all scaffold systems."""

    profiles = {}

    # ── DNA Origami ──
    p = DegradationProfile(
        scaffold_system="dna_origami",
        primary_trigger="enzymatic + dilution",
        trigger_conditions="DNase in natural water; Mg²⁺ dilution below 5 mM",
        half_life_description="4-24 hours in river water; 12-48 hours in seawater",
        half_life_hours=12.0,
        degradation_products=["nucleotides (A, T, G, C)", "deoxyribose", "phosphate"],
        metal_released="none",
        metal_toxicity=MetalToxicity.NONE,
        fragment_biodegradable=True,
        recommended_failsafe="ester",
        design_modifications=[],
        product_toxicity_score=1.0,
        fragment_score=1.0,
    )
    compute_dss(p)
    profiles["dna_origami"] = p

    # ── Nitschke Fe₄L₆ ──
    p = DegradationProfile(
        scaffold_system="nitschke_cage",
        primary_trigger="acid hydrolysis (imine)",
        trigger_conditions="pH < 5 → imine C=N hydrolysis; UV accelerates",
        half_life_description="Hours at pH < 5; days at pH 7",
        half_life_hours=48.0,
        degradation_products=["Fe²⁺", "aldehyde fragments", "amine fragments"],
        metal_released="Fe²⁺",
        metal_toxicity=MetalToxicity.BENIGN,
        fragment_biodegradable=True,
        recommended_failsafe="acetal",
        design_modifications=[
            "Use hydrazone variant for neutral-pH deployment (more stable operationally, still degrades)"
        ],
        product_toxicity_score=0.85,
        fragment_score=0.9,
    )
    compute_dss(p)
    profiles["nitschke_cage"] = p

    # ── Fujita Pd₆L₄ (standard — excluded) ──
    p = DegradationProfile(
        scaffold_system="fujita_cage_standard",
        primary_trigger="ligand exchange",
        trigger_conditions="pH < 3 protonation; slow exchange at neutral pH",
        half_life_description="Weeks-months in neutral water; minutes at pH < 3",
        half_life_hours=720.0,
        degradation_products=["Pd²⁺ (FREE)", "pyridyl ligand fragments"],
        metal_released="Pd²⁺",
        metal_toxicity=MetalToxicity.HIGH,
        fragment_biodegradable=True,
        recommended_failsafe="acetal",
        design_modifications=[
            "REQUIRED: self-chelating ligand design (NTA/EDTA group on ligand backbone)",
            "Without modification: Pd²⁺ release exceeds aquatic EC50 threshold"
        ],
        hard_excluded=True,
        exclusion_reason="Free Pd²⁺ release (EC50 = 13 µg/L Daphnia); requires self-chelating ligand modification",
        product_toxicity_score=0.2,
        fragment_score=0.8,
    )
    compute_dss(p)
    profiles["fujita_cage_standard"] = p

    # ── Fujita Pd₆L₄ (self-chelating ligand modification) ──
    p = DegradationProfile(
        scaffold_system="fujita_cage_chelating",
        primary_trigger="ligand exchange + self-chelation",
        trigger_conditions="Cage collapse → freed ligands re-chelate Pd²⁺ via built-in NTA group",
        half_life_description="Weeks in neutral water; hours at pH < 3",
        half_life_hours=720.0,
        degradation_products=["Pd-NTA complex (low bioavailability)", "pyridyl fragments"],
        metal_released="Pd²⁺ (chelated)",
        metal_toxicity=MetalToxicity.MODERATE,
        fragment_biodegradable=True,
        recommended_failsafe="acetal",
        design_modifications=[
            "Self-chelating ligand: NTA or EDTA group integrated into each ligand backbone",
            "Upon cage collapse, Pd²⁺ immediately re-chelated (log K_PdNTA = 14.1)"
        ],
        product_toxicity_score=0.5,
        fragment_score=0.8,
    )
    compute_dss(p)
    profiles["fujita_cage_chelating"] = p

    # ── MOF ZIF-8 (Zn) ──
    p = DegradationProfile(
        scaffold_system="mof_zif8",
        primary_trigger="acid dissolution",
        trigger_conditions="pH < 6 → Zn-imidazole bonds break; phosphate buffer accelerates",
        half_life_description="Hours at pH < 5; weeks at pH 7",
        half_life_hours=168.0,
        degradation_products=["Zn²⁺", "imidazole"],
        metal_released="Zn²⁺",
        metal_toxicity=MetalToxicity.MODERATE_LOW,
        fragment_biodegradable=True,
        recommended_failsafe="acetal",
        design_modifications=[],
        product_toxicity_score=0.75,
        fragment_score=0.9,
    )
    compute_dss(p)
    profiles["mof_zif8"] = p

    # ── MOF UiO-66 (Zr, standard — persistence concern) ──
    p = DegradationProfile(
        scaffold_system="mof_uio66_standard",
        primary_trigger="almost none (extremely stable)",
        trigger_conditions="Stable pH 1-12, to 500°C; resists all mild degradation",
        half_life_description="Years in environment",
        half_life_hours=50000.0,
        degradation_products=["ZrO₂ (insoluble)", "terephthalic acid"],
        metal_released="Zr⁴⁺ (as ZrO₂)",
        metal_toxicity=MetalToxicity.LOW,
        fragment_biodegradable=False,
        recommended_failsafe="acetal",
        design_modifications=[
            "REQUIRED: 30% missing-linker defects for environmental degradability",
            "Without defects: persists for years — unacceptable for field deployment"
        ],
        hard_excluded=True,
        exclusion_reason="Non-degradable without defect engineering; environmental persistence",
        product_toxicity_score=0.6,
        fragment_score=0.4,
    )
    compute_dss(p)
    profiles["mof_uio66_standard"] = p

    # ── MOF UiO-66 (Zr, defect-engineered) ──
    p = DegradationProfile(
        scaffold_system="mof_uio66_defective",
        primary_trigger="hydrolysis via defect sites",
        trigger_conditions="30% missing-linker defects → water ingress → weeks to degrade",
        half_life_description="2-8 weeks in neutral water",
        half_life_hours=672.0,
        degradation_products=["ZrO₂ (insoluble nanoparticles)", "terephthalic acid"],
        metal_released="Zr⁴⁺ (as ZrO₂)",
        metal_toxicity=MetalToxicity.LOW,
        fragment_biodegradable=False,
        recommended_failsafe="acetal",
        design_modifications=[
            "30% missing-linker defects via modulator synthesis (acetic acid modulator)",
            "Verify defect level by TGA before deployment"
        ],
        product_toxicity_score=0.6,
        fragment_score=0.4,
    )
    compute_dss(p)
    profiles["mof_uio66_defective"] = p

    # ── MOF MIL-101 (Cr) — BANNED ──
    p = DegradationProfile(
        scaffold_system="mof_mil101_cr",
        primary_trigger="n/a",
        trigger_conditions="n/a",
        half_life_description="n/a",
        half_life_hours=50000.0,
        degradation_products=["Cr³⁺ → potential Cr⁶⁺ under oxidizing conditions"],
        metal_released="Cr³⁺/Cr⁶⁺",
        metal_toxicity=MetalToxicity.BANNED,
        fragment_biodegradable=False,
        recommended_failsafe="n/a",
        design_modifications=[],
        hard_excluded=True,
        exclusion_reason="Cr⁶⁺ is a known carcinogen; Cr³⁺ oxidizes to Cr⁶⁺ under environmental conditions",
        product_toxicity_score=0.0,
        fragment_score=0.3,
    )
    compute_dss(p)
    profiles["mof_mil101_cr"] = p

    # ── POC (imine linkage) ──
    p = DegradationProfile(
        scaffold_system="poc_imine",
        primary_trigger="acid hydrolysis (imine)",
        trigger_conditions="pH < 5 → imine hydrolysis; similar to Nitschke",
        half_life_description="Hours at pH < 5; days-weeks at pH 7",
        half_life_hours=72.0,
        degradation_products=["aldehyde fragments", "amine fragments"],
        metal_released="none",
        metal_toxicity=MetalToxicity.NONE,
        fragment_biodegradable=True,
        recommended_failsafe="acetal",
        design_modifications=[],
        product_toxicity_score=0.85,
        fragment_score=0.9,
    )
    compute_dss(p)
    profiles["poc_imine"] = p

    # ── POC (boronate ester linkage) ──
    p = DegradationProfile(
        scaffold_system="poc_boronate",
        primary_trigger="diol competition + acid",
        trigger_conditions="Fructose/glucose in natural water; pH < 4",
        half_life_description="Days in natural water (fructose ~0.01 mM); hours at pH < 4",
        half_life_hours=120.0,
        degradation_products=["boronic acid", "diol fragments"],
        metal_released="none",
        metal_toxicity=MetalToxicity.NONE,
        fragment_biodegradable=True,
        recommended_failsafe="ester",
        design_modifications=[],
        product_toxicity_score=0.80,
        fragment_score=0.85,
    )
    compute_dss(p)
    profiles["poc_boronate"] = p

    # ── MOP Cu paddlewheel ──
    p = DegradationProfile(
        scaffold_system="mop_cu",
        primary_trigger="water hydrolysis",
        trigger_conditions="Cu-paddlewheel unstable in water; dissolves over days",
        half_life_description="Days in water",
        half_life_hours=72.0,
        degradation_products=["Cu²⁺", "carboxylate ligands"],
        metal_released="Cu²⁺",
        metal_toxicity=MetalToxicity.MODERATE,
        fragment_biodegradable=True,
        recommended_failsafe="ester",
        design_modifications=[
            "Limit deployment to low-volume or contained systems (Cu²⁺ is moderately toxic)"
        ],
        product_toxicity_score=0.5,
        fragment_score=0.85,
    )
    compute_dss(p)
    profiles["mop_cu"] = p

    # ── MOP Fe₃O cluster ──
    p = DegradationProfile(
        scaffold_system="mop_fe",
        primary_trigger="hydrolysis",
        trigger_conditions="Fe-carboxylate bonds hydrolyze in water over weeks",
        half_life_description="1-4 weeks in water",
        half_life_hours=336.0,
        degradation_products=["Fe³⁺ → Fe(OH)₃ (rust)", "carboxylate ligands"],
        metal_released="Fe³⁺",
        metal_toxicity=MetalToxicity.BENIGN,
        fragment_biodegradable=True,
        recommended_failsafe="ester",
        design_modifications=[],
        product_toxicity_score=0.80,
        fragment_score=0.85,
    )
    compute_dss(p)
    profiles["mop_fe"] = p

    return profiles


_PROFILES: dict[str, DegradationProfile] = _build_profiles()


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold system → degradation profile mapping
# ═══════════════════════════════════════════════════════════════════════════

# Maps cascade_scaffold.ScaffoldSystem values to degradation profile keys
_SCAFFOLD_MAP: dict[str, str] = {
    "dna_origami": "dna_origami",
    "fujita_cage": "fujita_cage_chelating",   # default to chelating (safe) variant
    "nitschke_cage": "nitschke_cage",
    "mof_cavity": "mof_zif8",                 # default to ZIF-8 (degradable)
    "poc": "poc_imine",                        # default to imine (degradable)
    "mop": "mop_fe",                           # default to Fe (benign)
}


def get_degradation_profile(
    scaffold_system: str,
    variant: Optional[str] = None,
) -> DegradationProfile:
    """Get degradation profile for a scaffold system.

    Args:
        scaffold_system: key from cascade_scaffold.ScaffoldSystem
        variant: specific variant (e.g., "standard" vs "chelating" for Fujita)

    Returns:
        DegradationProfile with pre-computed DSS.
    """
    if variant:
        key = f"{scaffold_system}_{variant}" if f"{scaffold_system}_{variant}" in _PROFILES else variant
        if key in _PROFILES:
            return _PROFILES[key]

    mapped = _SCAFFOLD_MAP.get(scaffold_system, scaffold_system)
    if mapped in _PROFILES:
        return _PROFILES[mapped]

    # Unknown scaffold — return conservative profile
    p = DegradationProfile(
        scaffold_system=scaffold_system,
        primary_trigger="unknown",
        trigger_conditions="No degradation data available",
        half_life_description="Unknown",
        half_life_hours=8760.0,
        degradation_products=["unknown"],
        metal_released="unknown",
        metal_toxicity=MetalToxicity.MODERATE,
        fragment_biodegradable=False,
        recommended_failsafe="ester",
        design_modifications=["Characterize degradation before field deployment"],
        product_toxicity_score=0.3,
        fragment_score=0.3,
    )
    compute_dss(p)
    return p


def list_profiles() -> list[str]:
    """Return all available degradation profile keys."""
    return list(_PROFILES.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Fail-safe linker database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FailSafeLinker:
    """A fail-safe linker for the tether stack."""
    name: str
    linker_id: str
    trigger: FailSafeTrigger
    trigger_conditions: str               # "pH < 5", "UV 365nm", "fructose"
    half_life_env_hours: float            # estimated half-life under trigger
    operational_stable: bool              # stable during normal operation?
    cleavage_products: list[str]
    products_toxic: bool
    reagent_description: str              # "N₃-PEG₂-acetal-PEG₂-DBCO"
    notes: str = ""


_FAILSAFE_LINKERS: dict[str, FailSafeLinker] = {
    "acetal": FailSafeLinker(
        name="Acetal fail-safe",
        linker_id="acetal",
        trigger=FailSafeTrigger.ACID,
        trigger_conditions="pH < 5 → acetal hydrolysis (hours)",
        half_life_env_hours=4.0,
        operational_stable=True,
        cleavage_products=["aldehyde", "alcohol", "PEG fragments"],
        products_toxic=False,
        reagent_description="N₃-PEG₂-O-CH(OMe)-O-PEG₂-DBCO",
        notes="Ideal for mine drainage (pH 3-5). Stable at pH > 6.",
    ),
    "ortho_ester": FailSafeLinker(
        name="Ortho-ester fail-safe",
        linker_id="ortho_ester",
        trigger=FailSafeTrigger.ACID,
        trigger_conditions="pH < 5 → ortho-ester hydrolysis (faster than acetal)",
        half_life_env_hours=2.0,
        operational_stable=True,
        cleavage_products=["ester", "alcohol"],
        products_toxic=False,
        reagent_description="N₃-PEG₂-orthoester-PEG₂-DBCO",
        notes="Faster cleavage than acetal. Use when rapid release needed.",
    ),
    "ester": FailSafeLinker(
        name="Ester fail-safe",
        linker_id="ester",
        trigger=FailSafeTrigger.HYDROLYSIS,
        trigger_conditions="Neutral water → slow ester hydrolysis (weeks-months)",
        half_life_env_hours=1000.0,
        operational_stable=True,
        cleavage_products=["carboxylic acid", "alcohol"],
        products_toxic=False,
        reagent_description="N₃-PEG₄-ester-PEG₄-DBCO",
        notes="Slowest fail-safe. Use for long operational lifetime + eventual degradation.",
    ),
    "photocleavable": FailSafeLinker(
        name="o-Nitrobenzyl photocleavable fail-safe",
        linker_id="photocleavable",
        trigger=FailSafeTrigger.UV,
        trigger_conditions="UV 365nm or sunlight → hours",
        half_life_env_hours=6.0,
        operational_stable=True,
        cleavage_products=["nitroso-aldehyde", "amine"],
        products_toxic=False,
        reagent_description="N₃-PEG₂-oNB-PEG₂-DBCO",
        notes="Ideal for systems that operate in dark but may be released to surface. "
              "Stable indefinitely in dark.",
    ),
    "boronate_ester": FailSafeLinker(
        name="Boronate ester fail-safe",
        linker_id="boronate_ester",
        trigger=FailSafeTrigger.DIOL,
        trigger_conditions="Fructose/glucose in natural water (0.01 mM) → days",
        half_life_env_hours=72.0,
        operational_stable=True,
        cleavage_products=["boronic acid", "diol"],
        products_toxic=False,
        reagent_description="N₃-PEG₂-boronate-PEG₂-DBCO",
        notes="Elegant: degradation triggered by natural sugar metabolites. "
              "Stable in clean/synthetic water, degrades in natural water.",
    ),
    "disulfide": FailSafeLinker(
        name="Disulfide fail-safe",
        linker_id="disulfide",
        trigger=FailSafeTrigger.REDUCING,
        trigger_conditions="Environmental thiols (glutathione, cysteine) → hours-days",
        half_life_env_hours=48.0,
        operational_stable=True,
        cleavage_products=["two thiols"],
        products_toxic=False,
        reagent_description="N₃-PEG₂-SS-PEG₂-DBCO",
        notes="Cleaved by biological reducing agents. Good for water with "
              "biological activity (wastewater, soil leachate).",
    ),
    "hydrazone": FailSafeLinker(
        name="Hydrazone fail-safe",
        linker_id="hydrazone",
        trigger=FailSafeTrigger.ACID,
        trigger_conditions="pH < 5 → hydrazone hydrolysis; UV accelerates",
        half_life_env_hours=8.0,
        operational_stable=True,
        cleavage_products=["aldehyde", "hydrazide"],
        products_toxic=False,
        reagent_description="N₃-PEG₂-hydrazone-PEG₂-DBCO",
        notes="Slightly more stable than acetal at neutral pH. "
              "Good intermediate between acetal (fast) and ester (slow).",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Fail-safe linker selection
# ═══════════════════════════════════════════════════════════════════════════

# Matrix → recommended fail-safe linker(s), ordered by preference
_MATRIX_LINKER_MAP: dict[str, list[str]] = {
    "mine_drainage":   ["acetal", "ortho_ester", "hydrazone"],
    "wastewater":      ["boronate_ester", "ester", "disulfide"],
    "seawater":        ["photocleavable", "ester", "boronate_ester"],
    "groundwater":     ["ester", "disulfide", "boronate_ester"],
    "flue_gas":        ["acetal", "ortho_ester"],
    "surface_water":   ["photocleavable", "boronate_ester", "ester"],
    "textile":         ["acetal", "photocleavable", "hydrazone"],
    "air":             ["photocleavable", "ester"],
}


def select_fail_safe_linker(
    deployment_matrix: str,
    scaffold_system: Optional[str] = None,
) -> FailSafeLinker:
    """Select the best fail-safe linker for a deployment environment.

    Args:
        deployment_matrix: key from DeploymentMatrix or string description
        scaffold_system: optional scaffold system for compatibility check

    Returns:
        Best FailSafeLinker for the conditions.
    """
    matrix_key = deployment_matrix.lower().replace(" ", "_")

    candidates = _MATRIX_LINKER_MAP.get(matrix_key, ["ester"])

    for linker_id in candidates:
        if linker_id in _FAILSAFE_LINKERS:
            return _FAILSAFE_LINKERS[linker_id]

    return _FAILSAFE_LINKERS["ester"]  # universal fallback


def get_linker(linker_id: str) -> Optional[FailSafeLinker]:
    """Look up a specific fail-safe linker by ID."""
    return _FAILSAFE_LINKERS.get(linker_id)


def list_linkers() -> list[str]:
    """Return all available fail-safe linker IDs."""
    return list(_FAILSAFE_LINKERS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Safety enforcement
# ═══════════════════════════════════════════════════════════════════════════

_DSS_HARD_EXCLUDE = 0.3
_DSS_WARNING = 0.5


@dataclass
class SafetyAssessment:
    """Safety assessment for one system design."""
    scaffold_system: str
    degradation: DegradationProfile
    fail_safe: FailSafeLinker
    deployment_matrix: str
    safe_for_deployment: bool
    warnings: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "APPROVED" if self.safe_for_deployment else "REJECTED"
        lines = [
            f"Safety Assessment: [{status}]",
            f"  Scaffold: {self.scaffold_system}",
            f"  DSS: {self.degradation.dss:.2f}",
            f"  Fail-safe: {self.fail_safe.name} ({self.fail_safe.trigger.value})",
            f"  Deployment: {self.deployment_matrix}",
        ]
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.alternatives:
            lines.append(f"  Alternatives: {', '.join(self.alternatives)}")
        return "\n".join(lines)


def assess_safety(
    scaffold_system: str,
    deployment_matrix: str = "surface_water",
    variant: Optional[str] = None,
) -> SafetyAssessment:
    """Full safety assessment for a scaffold in a deployment environment.

    Returns SafetyAssessment with approval/rejection, warnings, and alternatives.
    """
    degradation = get_degradation_profile(scaffold_system, variant)
    fail_safe = select_fail_safe_linker(deployment_matrix, scaffold_system)

    safe = True
    warnings = []
    alternatives = []

    # Hard exclusion
    if degradation.hard_excluded:
        safe = False
        warnings.append(degradation.exclusion_reason)
        # Suggest alternatives
        for key, prof in _PROFILES.items():
            if not prof.hard_excluded and prof.dss >= 0.7:
                alternatives.append(f"{key} (DSS={prof.dss:.2f})")
    elif degradation.dss < _DSS_HARD_EXCLUDE:
        safe = False
        warnings.append(f"DSS {degradation.dss:.2f} below minimum threshold {_DSS_HARD_EXCLUDE}")
    elif degradation.dss < _DSS_WARNING:
        warnings.append(f"DSS {degradation.dss:.2f} is marginal — consider alternatives")

    # Metal toxicity warnings
    if degradation.metal_toxicity in (MetalToxicity.HIGH, MetalToxicity.BANNED):
        if not degradation.hard_excluded:
            warnings.append(f"Metal released: {degradation.metal_released} "
                            f"({degradation.metal_toxicity.value} toxicity)")

    # Persistence warnings
    if degradation.half_life_hours > 672:  # > 4 weeks
        warnings.append(f"Persistence: {degradation.half_life_description} — "
                        f"may accumulate if released continuously")

    # Design modification warnings
    if degradation.design_modifications:
        for mod in degradation.design_modifications:
            if mod.startswith("REQUIRED"):
                warnings.append(f"Modification needed: {mod}")

    return SafetyAssessment(
        scaffold_system=scaffold_system,
        degradation=degradation,
        fail_safe=fail_safe,
        deployment_matrix=deployment_matrix,
        safe_for_deployment=safe,
        warnings=warnings,
        alternatives=alternatives[:3],
    )


def enforce_safety(
    scaffold_systems: list[str],
    deployment_matrix: str = "surface_water",
) -> list[SafetyAssessment]:
    """Assess and filter a list of scaffold systems for safety.

    Returns all assessments (safe and rejected). Use
    [a for a in results if a.safe_for_deployment] to get only approved.
    """
    return [assess_safety(s, deployment_matrix) for s in scaffold_systems]


def filter_safe_scaffolds(
    scaffold_systems: list[str],
    deployment_matrix: str = "surface_water",
) -> list[str]:
    """Return only scaffold systems that pass safety assessment."""
    assessments = enforce_safety(scaffold_systems, deployment_matrix)
    return [a.scaffold_system for a in assessments if a.safe_for_deployment]
