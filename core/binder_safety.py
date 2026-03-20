"""
core/binder_safety.py — Off-Target & Environmental Toxicity Screen

Screens de novo designed binders for:
  1. Essential metal depletion — will the binder strip Ca²⁺, Mg²⁺, Zn²⁺, Fe²⁺?
  2. Off-target selectivity — affinity ratio target vs competitors
  3. Environmental persistence — biodegradability from molecular structure
  4. Bioaccumulation potential — LogP-based
  5. Aquatic toxicity — structural alert screening
  6. Reactive group hazards — isocyanates, acrylates, epoxides, etc.

Hard constraints:
  - Binder that depletes Ca²⁺ or Mg²⁺ below physiological threshold → excluded
  - Selectivity ratio < 10 for target vs essential metal → warning
  - LogP > 5 (bioaccumulative) → warning
  - Known toxicophore present → warning or exclusion

Does NOT replace proper ecotoxicology testing. Provides computational
pre-screening to flag candidates before synthesis.

Data tier: Tier 2 (published HSAB data, stability constant patterns,
structural alert databases from Derek Nexus / Kroes TTC literature).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Essential metal database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EssentialMetal:
    """An essential biological metal that must not be depleted."""
    symbol: str
    name: str
    charge: int
    hsab_class: str              # "hard", "borderline", "soft"
    preferred_donors: list       # ["O", "N"], ["N", "S"], etc.
    physiological_range_uM: tuple[float, float]  # normal range in blood/tissue
    depletion_threshold_uM: float  # below this → clinical consequences
    coordination_number: int
    depletion_consequence: str
    typical_log_k_edta: float    # log K with EDTA (reference chelator)


_ESSENTIAL_METALS: dict[str, EssentialMetal] = {
    "Ca2+": EssentialMetal(
        "Ca", "calcium", 2, "hard", ["O"],
        (2200.0, 2600.0), 1800.0, 6,
        "Hypocalcemia: cardiac arrhythmia, tetany, seizures",
        10.7,
    ),
    "Mg2+": EssentialMetal(
        "Mg", "magnesium", 2, "hard", ["O", "N"],
        (700.0, 1000.0), 500.0, 6,
        "Hypomagnesemia: cardiac arrhythmia, muscle weakness, tremor",
        8.7,
    ),
    "Zn2+": EssentialMetal(
        "Zn", "zinc", 2, "borderline", ["N", "S", "O"],
        (10.0, 18.0), 7.0, 4,
        "Zinc depletion: immune suppression, wound healing impairment, taste loss",
        16.5,
    ),
    "Fe2+": EssentialMetal(
        "Fe", "iron (II)", 2, "borderline", ["N", "O", "S"],
        (10.0, 30.0), 5.0, 6,
        "Iron depletion: anemia, fatigue, cognitive impairment",
        14.3,
    ),
    "Fe3+": EssentialMetal(
        "Fe", "iron (III)", 3, "hard", ["O", "N"],
        (10.0, 30.0), 5.0, 6,
        "Iron depletion: anemia, fatigue, cognitive impairment",
        25.1,
    ),
    "Cu2+": EssentialMetal(
        "Cu", "copper", 2, "borderline", ["N", "S", "O"],
        (10.0, 25.0), 5.0, 4,
        "Copper depletion: anemia, neutropenia, neurodegeneration",
        18.8,
    ),
    "Mn2+": EssentialMetal(
        "Mn", "manganese", 2, "borderline", ["O", "N"],
        (0.1, 0.3), 0.05, 6,
        "Manganese depletion: bone/cartilage defects, impaired glucose metabolism",
        13.9,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# HSAB affinity estimation
# ═══════════════════════════════════════════════════════════════════════════

def _hsab_match(binder_donors: list[str], binder_hsab: str,
                metal: EssentialMetal) -> float:
    """Estimate relative affinity of binder for a metal via HSAB matching.

    Returns a score 0-1 where higher = stronger predicted binding.
    """
    # Donor atom overlap
    donor_overlap = len(set(binder_donors) & set(metal.preferred_donors))
    max_donors = max(len(binder_donors), len(metal.preferred_donors), 1)
    donor_score = donor_overlap / max_donors

    # HSAB match
    hsab_matrix = {
        ("hard", "hard"): 1.0,
        ("hard", "borderline"): 0.5,
        ("hard", "soft"): 0.1,
        ("borderline", "hard"): 0.5,
        ("borderline", "borderline"): 0.8,
        ("borderline", "soft"): 0.5,
        ("soft", "hard"): 0.1,
        ("soft", "borderline"): 0.5,
        ("soft", "soft"): 1.0,
    }
    hsab_score = hsab_matrix.get((binder_hsab, metal.hsab_class), 0.3)

    return 0.5 * donor_score + 0.5 * hsab_score


def estimate_off_target_log_k(
    binder_log_k_target: float,
    binder_donors: list[str],
    binder_hsab: str,
    binder_denticity: int,
    metal: EssentialMetal,
) -> float:
    """Estimate log K of binder with an essential metal.

    Uses HSAB matching + denticity correction relative to the
    binder's known affinity for its target.

    This is a rough estimate (±2 log units). Sufficient for screening,
    not for quantitative prediction.
    """
    match = _hsab_match(binder_donors, binder_hsab, metal)

    # Denticity correction: chelate effect adds ~1.5 log K per ring
    # Reference: EDTA (denticity 6) vs monodentate
    denticity_factor = min(binder_denticity, metal.coordination_number)

    # Estimate: scale from EDTA's log K for this metal
    # If binder is weaker HSAB match, proportionally lower log K
    estimated = metal.typical_log_k_edta * match * (denticity_factor / 6.0)

    return round(estimated, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Structural alerts for environmental toxicity
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StructuralAlert:
    """A molecular structural feature associated with toxicity."""
    name: str
    description: str
    severity: str          # "warning", "exclude"
    functional_groups: list  # groups that trigger this alert
    mechanism: str         # "reactive", "persistent", "bioaccumulative", "mutagenic"


_STRUCTURAL_ALERTS: list[StructuralAlert] = [
    # Reactive electrophiles — direct toxicity
    StructuralAlert(
        "isocyanate", "Isocyanate group (-N=C=O): respiratory sensitizer, reactive electrophile",
        "exclude", ["isocyanate", "NCO", "-N=C=O"], "reactive",
    ),
    StructuralAlert(
        "acyl_halide", "Acyl halide (-C(=O)X): highly reactive, corrosive",
        "exclude", ["acyl_chloride", "acyl_bromide", "C(=O)Cl", "C(=O)Br"], "reactive",
    ),
    StructuralAlert(
        "epoxide", "Epoxide: alkylating agent, mutagenic potential",
        "warning", ["epoxide", "oxirane"], "mutagenic",
    ),
    StructuralAlert(
        "acrylate", "Michael acceptor (acrylate/acrylonitrile): skin sensitizer",
        "warning", ["acrylate", "acrylamide", "acrylonitrile", "C=CC(=O)"], "reactive",
    ),
    StructuralAlert(
        "aldehyde", "Aldehyde: reactive, irritant, some are mutagenic (formaldehyde)",
        "warning", ["aldehyde", "CHO", "-C(=O)H"], "reactive",
    ),

    # Persistent organic pollutant features
    StructuralAlert(
        "polyfluorinated", "Polyfluorinated (≥3 C-F): PFAS-like persistence",
        "exclude", ["perfluoro", "PFAS", "CF3", "polyfluoro"], "persistent",
    ),
    StructuralAlert(
        "polychlorinated", "Polychlorinated (≥3 Cl on aromatic): PCB-like persistence",
        "exclude", ["polychloro", "PCB", "trichlorophenol"], "persistent",
    ),
    StructuralAlert(
        "polyaromatic", "Polycyclic aromatic (≥4 fused rings): mutagenic, persistent",
        "warning", ["PAH", "pyrene", "anthracene", "benzo[a]pyrene"], "mutagenic",
    ),

    # Bioaccumulation flags
    StructuralAlert(
        "long_alkyl", "Long alkyl chain (≥C12): surfactant, bioaccumulative",
        "warning", ["dodecyl", "stearyl", "C12+", "C16+", "C18+"], "bioaccumulative",
    ),

    # Heavy metal content in the binder itself
    StructuralAlert(
        "contains_hg", "Mercury in binder structure",
        "exclude", ["Hg", "mercury", "organomercury"], "toxic_metal",
    ),
    StructuralAlert(
        "contains_cd", "Cadmium in binder structure",
        "exclude", ["Cd", "cadmium"], "toxic_metal",
    ),
    StructuralAlert(
        "contains_tl", "Thallium in binder structure",
        "exclude", ["Tl", "thallium"], "toxic_metal",
    ),
    StructuralAlert(
        "contains_as", "Arsenic in binder structure (not arsenate target)",
        "warning", ["As", "arsenic", "arsenical"], "toxic_metal",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Binder specification for screening
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BinderSpec:
    """Properties of a designed binder for safety screening.

    Can be constructed from CandidateResult, CaptureTransformScore,
    or directly from binder design parameters.
    """
    name: str
    binder_type: str                 # "chelator", "enzyme_mimic", "MOF_site", etc.
    donor_atoms: list[str]           # ["N", "N", "O", "O"]
    hsab_class: str                  # "hard", "borderline", "soft", "mixed"
    denticity: int = 2               # number of donor atoms coordinating target
    target_metal: str = ""           # "Pb2+", "CO2", etc.
    target_log_k: float = 0.0       # estimated log K for target
    molecular_weight: float = 0.0    # Da
    logP: float = 0.0                # octanol-water partition coefficient
    functional_groups: list[str] = field(default_factory=list)  # ["amine", "thiol", "carboxylate"]
    contains_metal: str = ""         # metal in the binder itself ("Zn" for Zn-CA mimic)
    aromatic_rings: int = 0
    halogen_count: int = 0
    fluorine_count: int = 0
    smiles: str = ""                 # if available
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Safety assessment results
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OffTargetHit:
    """One predicted off-target binding interaction."""
    metal: str                       # "Ca2+", "Zn2+", etc.
    metal_name: str
    estimated_log_k: float
    selectivity_ratio: float         # 10^(log_k_target - log_k_offtarget)
    depletion_risk: str              # "none", "low", "moderate", "high", "critical"
    consequence: str
    notes: str = ""


@dataclass
class EnvironmentalFlag:
    """One environmental toxicity flag."""
    category: str                    # "persistence", "bioaccumulation", "aquatic_toxicity", "reactive"
    severity: str                    # "info", "warning", "exclude"
    description: str
    parameter: str = ""              # "LogP=4.2", "MW=650", etc.
    notes: str = ""


@dataclass
class BinderSafetyReport:
    """Complete safety screening report for one binder."""
    binder_name: str
    binder_spec: BinderSpec

    # Off-target screening
    off_target_hits: list[OffTargetHit]
    worst_selectivity_ratio: float       # lowest ratio (closest competitor)
    essential_metal_risk: str            # "none", "low", "moderate", "high", "critical"

    # Environmental screening
    environmental_flags: list[EnvironmentalFlag]
    persistence_class: str               # "readily_biodegradable", "inherently_biodegradable",
                                         # "persistent", "very_persistent"
    bioaccumulation_class: str           # "not_bioaccumulative", "bioaccumulative", "very_bioaccumulative"
    aquatic_toxicity_class: str          # "low", "moderate", "high", "very_high"

    # Composite
    safety_score: float = 0.0            # 0-1, higher = safer
    safe_for_deployment: bool = True
    exclusion_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "EXCLUDED" if not self.safe_for_deployment else \
                 "SAFE" if self.safety_score >= 0.7 else \
                 "CAUTION" if self.safety_score >= 0.4 else "UNSAFE"
        lines = [
            f"Binder Safety: {self.binder_name} [{status}]",
            f"  Safety score: {self.safety_score:.2f} / 1.00",
            f"  Off-target: {self.essential_metal_risk} risk "
            f"(worst selectivity: {self.worst_selectivity_ratio:.0f}×)",
            f"  Persistence: {self.persistence_class}",
            f"  Bioaccumulation: {self.bioaccumulation_class}",
            f"  Aquatic toxicity: {self.aquatic_toxicity_class}",
        ]
        if self.exclusion_reasons:
            for r in self.exclusion_reasons:
                lines.append(f"  ✗ EXCLUDE: {r}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.recommendations:
            for r in self.recommendations:
                lines.append(f"  → {r}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Off-target screening
# ═══════════════════════════════════════════════════════════════════════════

def screen_off_target(spec: BinderSpec) -> list[OffTargetHit]:
    """Screen binder for off-target binding to essential metals.

    Returns list of OffTargetHit for each essential metal with
    predicted binding affinity and selectivity ratio.
    """
    hits = []

    for metal_key, metal in _ESSENTIAL_METALS.items():
        # Skip if the target IS this essential metal (e.g., Zn²⁺ capture)
        if spec.target_metal and metal.symbol in spec.target_metal:
            continue

        # Estimate off-target log K
        log_k_off = estimate_off_target_log_k(
            spec.target_log_k, spec.donor_atoms, spec.hsab_class,
            spec.denticity, metal,
        )

        # Selectivity ratio: 10^(log_k_target - log_k_off)
        delta_log_k = spec.target_log_k - log_k_off
        selectivity = 10.0 ** max(0, delta_log_k)

        # Depletion risk
        if log_k_off > 12 and selectivity < 10:
            risk = "critical"
        elif log_k_off > 10 and selectivity < 100:
            risk = "high"
        elif log_k_off > 8 and selectivity < 1000:
            risk = "moderate"
        elif log_k_off > 5:
            risk = "low"
        else:
            risk = "none"

        hits.append(OffTargetHit(
            metal=metal_key,
            metal_name=metal.name,
            estimated_log_k=log_k_off,
            selectivity_ratio=round(selectivity, 1),
            depletion_risk=risk,
            consequence=metal.depletion_consequence,
            notes=f"HSAB match: {_hsab_match(spec.donor_atoms, spec.hsab_class, metal):.2f}",
        ))

    return hits


# ═══════════════════════════════════════════════════════════════════════════
# Environmental screening
# ═══════════════════════════════════════════════════════════════════════════

def _screen_structural_alerts(spec: BinderSpec) -> list[EnvironmentalFlag]:
    """Screen for known toxic structural features."""
    flags = []

    all_groups = " ".join(spec.functional_groups).lower()
    if spec.smiles:
        all_groups += " " + spec.smiles.lower()
    if spec.contains_metal:
        all_groups += " " + spec.contains_metal.lower()
    all_groups += " " + spec.name.lower()

    for alert in _STRUCTURAL_ALERTS:
        triggered = any(g.lower() in all_groups for g in alert.functional_groups)
        if triggered:
            flags.append(EnvironmentalFlag(
                category=alert.mechanism,
                severity=alert.severity,
                description=alert.description,
            ))

    return flags


def _assess_persistence(spec: BinderSpec) -> tuple[str, list[EnvironmentalFlag]]:
    """Estimate environmental persistence from molecular structure.

    Uses simplified biodegradability rules:
    - Low MW (<300), no halogens, no fused aromatics → readily biodegradable
    - Moderate MW, few halogens → inherently biodegradable
    - High MW, many halogens, fused aromatics → persistent
    - Fluorinated → very persistent (PFAS-like)
    """
    flags = []

    # Fluorine count
    if spec.fluorine_count >= 3:
        flags.append(EnvironmentalFlag(
            "persistence", "exclude",
            f"Polyfluorinated ({spec.fluorine_count} F atoms): PFAS-like persistence",
            f"F_count={spec.fluorine_count}",
        ))
        return "very_persistent", flags

    # Halogen count
    if spec.halogen_count >= 3:
        flags.append(EnvironmentalFlag(
            "persistence", "warning",
            f"Polyhalogenated ({spec.halogen_count} halogens): reduced biodegradability",
            f"halogen_count={spec.halogen_count}",
        ))

    # Aromatic ring count
    if spec.aromatic_rings >= 4:
        flags.append(EnvironmentalFlag(
            "persistence", "warning",
            f"Polycyclic aromatic ({spec.aromatic_rings} rings): slow biodegradation",
            f"aromatic_rings={spec.aromatic_rings}",
        ))

    # Classification
    if spec.fluorine_count >= 3:
        cls = "very_persistent"
    elif spec.halogen_count >= 3 or spec.aromatic_rings >= 4:
        cls = "persistent"
    elif spec.molecular_weight > 500 or spec.halogen_count >= 1 or spec.aromatic_rings >= 2:
        cls = "inherently_biodegradable"
    else:
        cls = "readily_biodegradable"

    return cls, flags


def _assess_bioaccumulation(spec: BinderSpec) -> tuple[str, list[EnvironmentalFlag]]:
    """Estimate bioaccumulation potential from LogP.

    REACH criteria:
    - LogP < 3.0: not bioaccumulative
    - LogP 3.0-4.5: potentially bioaccumulative
    - LogP > 4.5: bioaccumulative
    - LogP > 5.0: very bioaccumulative
    """
    flags = []
    logp = spec.logP

    if logp > 5.0:
        cls = "very_bioaccumulative"
        flags.append(EnvironmentalFlag(
            "bioaccumulation", "warning",
            f"LogP={logp:.1f} > 5.0: very bioaccumulative (REACH vB criterion)",
            f"LogP={logp:.1f}",
        ))
    elif logp > 4.5:
        cls = "bioaccumulative"
        flags.append(EnvironmentalFlag(
            "bioaccumulation", "warning",
            f"LogP={logp:.1f} > 4.5: bioaccumulative (REACH B criterion)",
            f"LogP={logp:.1f}",
        ))
    elif logp > 3.0:
        cls = "not_bioaccumulative"
        flags.append(EnvironmentalFlag(
            "bioaccumulation", "info",
            f"LogP={logp:.1f}: borderline — monitor in chronic exposure scenarios",
            f"LogP={logp:.1f}",
        ))
    else:
        cls = "not_bioaccumulative"

    return cls, flags


def _assess_aquatic_toxicity(spec: BinderSpec) -> tuple[str, list[EnvironmentalFlag]]:
    """Estimate aquatic toxicity class from molecular properties.

    Uses Verhaar classification (baseline narcosis) + structural alerts:
    - Baseline narcosis: -log LC50 ≈ 0.84 × LogP - 1.39 (fish)
    - Reactive chemicals: 10× more toxic than baseline
    - Specifically acting: varies
    """
    flags = []
    logp = spec.logP

    # Baseline narcosis estimate (Verhaar)
    neg_log_lc50 = 0.84 * logp - 1.39  # mmol/L scale
    estimated_lc50_mg_L = 10 ** (-neg_log_lc50) * spec.molecular_weight if spec.molecular_weight > 0 else 1000

    # Reactive group correction
    reactive_groups = {"isocyanate", "epoxide", "acrylate", "acyl_chloride",
                       "aldehyde", "acyl_halide", "acrylonitrile"}
    has_reactive = any(g.lower() in reactive_groups for g in spec.functional_groups)
    if has_reactive:
        estimated_lc50_mg_L /= 10.0  # 10× more toxic
        flags.append(EnvironmentalFlag(
            "aquatic_toxicity", "warning",
            "Reactive functional group → estimated 10× increase in aquatic toxicity",
        ))

    # Classification (EU CLP / GHS)
    if estimated_lc50_mg_L <= 1.0:
        cls = "very_high"
        flags.append(EnvironmentalFlag(
            "aquatic_toxicity", "warning",
            f"Estimated LC50 ≈ {estimated_lc50_mg_L:.1f} mg/L (Category 1: very toxic to aquatic life)",
            f"est_LC50={estimated_lc50_mg_L:.1f} mg/L",
        ))
    elif estimated_lc50_mg_L <= 10.0:
        cls = "high"
        flags.append(EnvironmentalFlag(
            "aquatic_toxicity", "warning",
            f"Estimated LC50 ≈ {estimated_lc50_mg_L:.0f} mg/L (Category 2: toxic to aquatic life)",
            f"est_LC50={estimated_lc50_mg_L:.0f} mg/L",
        ))
    elif estimated_lc50_mg_L <= 100.0:
        cls = "moderate"
    else:
        cls = "low"

    return cls, flags


# ═══════════════════════════════════════════════════════════════════════════
# Main screening function
# ═══════════════════════════════════════════════════════════════════════════

def screen_binder(spec: BinderSpec) -> BinderSafetyReport:
    """Complete safety screening of a designed binder.

    Runs all screens and produces a composite safety report with
    score, flags, exclusion reasons, and recommendations.
    """
    # ── Off-target screening ──
    off_target = screen_off_target(spec)
    worst_sel = min((h.selectivity_ratio for h in off_target), default=float('inf'))
    worst_risk = "none"
    for h in off_target:
        if h.depletion_risk == "critical":
            worst_risk = "critical"
            break
        elif h.depletion_risk == "high" and worst_risk not in ("critical",):
            worst_risk = "high"
        elif h.depletion_risk == "moderate" and worst_risk not in ("critical", "high"):
            worst_risk = "moderate"
        elif h.depletion_risk == "low" and worst_risk == "none":
            worst_risk = "low"

    # ── Environmental screening ──
    structural_flags = _screen_structural_alerts(spec)
    persistence_cls, persistence_flags = _assess_persistence(spec)
    bioaccum_cls, bioaccum_flags = _assess_bioaccumulation(spec)
    aquatic_cls, aquatic_flags = _assess_aquatic_toxicity(spec)

    all_env_flags = structural_flags + persistence_flags + bioaccum_flags + aquatic_flags

    # ── Exclusion logic ──
    exclusions = []
    warnings = []
    recommendations = []

    # Off-target exclusions
    critical_metals = [h for h in off_target if h.depletion_risk == "critical"]
    if critical_metals:
        for h in critical_metals:
            exclusions.append(
                f"Critical {h.metal_name} depletion risk: estimated log K = {h.estimated_log_k:.1f}, "
                f"selectivity only {h.selectivity_ratio:.0f}× over target")

    high_risk_metals = [h for h in off_target if h.depletion_risk == "high"]
    for h in high_risk_metals:
        warnings.append(
            f"{h.metal_name} binding: log K ≈ {h.estimated_log_k:.1f}, "
            f"selectivity {h.selectivity_ratio:.0f}× ({h.consequence})")

    moderate_risk = [h for h in off_target if h.depletion_risk == "moderate"]
    for h in moderate_risk:
        warnings.append(
            f"Moderate {h.metal_name} affinity: log K ≈ {h.estimated_log_k:.1f}")

    # Environmental exclusions
    for flag in all_env_flags:
        if flag.severity == "exclude":
            exclusions.append(flag.description)
        elif flag.severity == "warning":
            warnings.append(flag.description)

    # Recommendations
    if worst_risk in ("high", "critical"):
        recommendations.append(
            "Redesign donor set to reduce essential metal affinity: "
            "e.g., replace N donors with S (shifts HSAB soft, away from Ca²⁺/Mg²⁺)")
    if worst_risk in ("moderate",):
        recommendations.append(
            "Verify selectivity experimentally with ITC against Ca²⁺, Mg²⁺, Zn²⁺ panel")
    if bioaccum_cls in ("bioaccumulative", "very_bioaccumulative"):
        recommendations.append(
            "Add hydrophilic groups (PEG, sulfonate, carboxylate) to reduce LogP")
    if persistence_cls in ("persistent", "very_persistent"):
        recommendations.append(
            "Incorporate hydrolyzable linkage (ester, imine) for environmental degradation")
    if aquatic_cls in ("very_high", "high"):
        recommendations.append(
            "Conduct formal aquatic toxicity testing (OECD 203) before scale-up")

    # ── Composite safety score ──
    # Off-target score (0-1)
    offtarget_score = 1.0
    if worst_risk == "critical":
        offtarget_score = 0.0
    elif worst_risk == "high":
        offtarget_score = 0.2
    elif worst_risk == "moderate":
        offtarget_score = 0.5
    elif worst_risk == "low":
        offtarget_score = 0.8

    # Persistence score
    persist_score = {
        "readily_biodegradable": 1.0,
        "inherently_biodegradable": 0.7,
        "persistent": 0.3,
        "very_persistent": 0.0,
    }.get(persistence_cls, 0.5)

    # Bioaccumulation score
    bioaccum_score = {
        "not_bioaccumulative": 1.0,
        "bioaccumulative": 0.3,
        "very_bioaccumulative": 0.1,
    }.get(bioaccum_cls, 0.5)

    # Aquatic toxicity score
    aquatic_score = {
        "low": 1.0,
        "moderate": 0.6,
        "high": 0.3,
        "very_high": 0.1,
    }.get(aquatic_cls, 0.5)

    # Weighted composite
    safety_score = (0.40 * offtarget_score +
                    0.20 * persist_score +
                    0.15 * bioaccum_score +
                    0.25 * aquatic_score)

    safe = len(exclusions) == 0 and safety_score >= 0.3

    return BinderSafetyReport(
        binder_name=spec.name,
        binder_spec=spec,
        off_target_hits=off_target,
        worst_selectivity_ratio=round(worst_sel, 1),
        essential_metal_risk=worst_risk,
        environmental_flags=all_env_flags,
        persistence_class=persistence_cls,
        bioaccumulation_class=bioaccum_cls,
        aquatic_toxicity_class=aquatic_cls,
        safety_score=round(safety_score, 4),
        safe_for_deployment=safe,
        exclusion_reasons=exclusions,
        warnings=warnings,
        recommendations=recommendations,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: screen from capture-transform pipeline parameters
# ═══════════════════════════════════════════════════════════════════════════

def screen_capture_element(
    name: str,
    target_formula: str,
    donor_atoms: list[str],
    hsab_class: str = "borderline",
    denticity: int = 2,
    target_log_k: float = 10.0,
    molecular_weight: float = 300.0,
    logP: float = 1.0,
    functional_groups: Optional[list[str]] = None,
    contains_metal: str = "",
) -> BinderSafetyReport:
    """Screen a capture element from the capture-transform pipeline.

    Convenience wrapper that builds a BinderSpec and runs full screening.
    """
    spec = BinderSpec(
        name=name,
        binder_type="capture_element",
        donor_atoms=donor_atoms,
        hsab_class=hsab_class,
        denticity=denticity,
        target_metal=target_formula,
        target_log_k=target_log_k,
        molecular_weight=molecular_weight,
        logP=logP,
        functional_groups=functional_groups or [],
        contains_metal=contains_metal,
    )
    return screen_binder(spec)


# ═══════════════════════════════════════════════════════════════════════════
# Batch screening
# ═══════════════════════════════════════════════════════════════════════════

def screen_batch(specs: list[BinderSpec]) -> list[BinderSafetyReport]:
    """Screen multiple binder designs. Returns all reports (safe and excluded)."""
    return [screen_binder(spec) for spec in specs]


def filter_safe_binders(specs: list[BinderSpec]) -> list[BinderSpec]:
    """Return only binder specs that pass safety screening."""
    reports = screen_batch(specs)
    return [r.binder_spec for r in reports if r.safe_for_deployment]
