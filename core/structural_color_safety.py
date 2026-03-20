"""
core/structural_color_safety.py — Structural Color Component Safety

Screens every material in a structural color design for:
  1. Nanoparticle-specific hazards (lung deposition, dissolution, ROS)
  2. Absorber toxicity (carbon black IARC 2B, PDA benign, azo dyes carcinogenic)
  3. Matrix/binder leaching (monomers, plasticizers, catalysts)
  4. Chromophore structural alerts (anthracyclines, azo dyes, heavy metal pigments)
  5. Application-specific exposure routes (inhalation, dermal, oral, aquatic)

The doxorubicin principle: a beautiful color does not mean a safe material.
Anthracyclines are brilliant red and also DNA intercalators / cardiotoxins.
Azo dyes are cheap vivid colors and also produce carcinogenic aromatic amines
on reductive cleavage. Cadmium pigments are stunning yellows/reds and cause
kidney failure. Structural color replaces ALL of these with geometry — but
the absorber, particle, and matrix materials still need screening.

Hard constraints:
  - IARC Group 1 carcinogens in particle/absorber → excluded
  - Known reproductive toxins → excluded for textile/wearable
  - Acute oral toxicity LD50 < 50 mg/kg → excluded for textile
  - Nano-ZnO in textile (dissolves in sweat → Zn²⁺ at toxic levels) → warning

Data tier: Tier 2 (IARC monographs, GHS classifications, published nano-tox data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Exposure routes by application
# ═══════════════════════════════════════════════════════════════════════════

class ExposureRoute(str, Enum):
    INHALATION = "inhalation"           # during spray application, sanding, weathering
    DERMAL = "dermal"                   # skin contact (textile, wearable)
    ORAL = "oral"                       # mouthing (infant textile), accidental ingestion
    AQUATIC = "aquatic"                 # wash water, rain runoff, leachate
    OCULAR = "ocular"                   # splash during application


_APPLICATION_ROUTES: dict[str, list[ExposureRoute]] = {
    "facade_panel":       [ExposureRoute.INHALATION, ExposureRoute.AQUATIC],
    "roof_coating":       [ExposureRoute.INHALATION, ExposureRoute.AQUATIC],
    "wall_tile":          [ExposureRoute.INHALATION],
    "textile_coating":    [ExposureRoute.DERMAL, ExposureRoute.ORAL, ExposureRoute.AQUATIC],
    "protective_garment": [ExposureRoute.DERMAL, ExposureRoute.INHALATION],
    "smart_textile":      [ExposureRoute.DERMAL, ExposureRoute.ORAL, ExposureRoute.AQUATIC],
    "paint":              [ExposureRoute.INHALATION, ExposureRoute.DERMAL, ExposureRoute.AQUATIC],
    "cosmetic":           [ExposureRoute.DERMAL, ExposureRoute.ORAL, ExposureRoute.OCULAR],
}


# ═══════════════════════════════════════════════════════════════════════════
# Material safety profiles
# ═══════════════════════════════════════════════════════════════════════════

class IARCGroup(str, Enum):
    """IARC carcinogenicity classification."""
    GROUP_1 = "1"           # carcinogenic to humans
    GROUP_2A = "2A"         # probably carcinogenic
    GROUP_2B = "2B"         # possibly carcinogenic
    GROUP_3 = "3"           # not classifiable
    NOT_CLASSIFIED = "NC"   # not evaluated


@dataclass(frozen=True)
class MaterialSafety:
    """Safety profile for one structural color material."""
    material_id: str
    name: str
    category: str                    # "particle", "absorber", "matrix", "chromophore"

    # Regulatory classification
    iarc_group: IARCGroup
    ghs_hazards: tuple               # ("H302 Harmful if swallowed", ...)

    # Toxicity data
    ld50_oral_mg_kg: float           # rat oral LD50 (-1 if no data)
    lc50_inhalation_mg_m3: float     # rat 4h inhalation LC50 (-1 if no data)
    ec50_aquatic_mg_L: float         # Daphnia/fish 48h EC50 (-1 if no data)

    # Nano-specific hazards
    nano_lung_hazard: bool           # ultrafine particles deposit in alveoli
    nano_dissolution: bool           # dissolves releasing toxic ions
    nano_ros_generation: bool        # generates reactive oxygen species

    # Leaching
    leaches_in_water: bool           # dissolves or releases components in water
    leaches_in_acid: bool            # dissolves below pH ~5 (sweat, stomach)

    # Overall classification
    safe_for_building: bool
    safe_for_textile: bool
    safe_for_cosmetic: bool

    # Fields with defaults
    nano_notes: str = ""
    leach_products: tuple = ()       # what leaches out
    leach_notes: str = ""
    excluded_applications: tuple = ()  # ("textile", "cosmetic")
    warnings_for: tuple = ()         # ("inhalation during spray",)
    source: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Material safety database
# ═══════════════════════════════════════════════════════════════════════════

_MATERIALS: dict[str, MaterialSafety] = {

    # ── Particle materials ──

    "SiO2": MaterialSafety(
        "SiO2", "Amorphous silica (Stöber)", "particle",
        IARCGroup.GROUP_3,
        ("H335 May cause respiratory irritation (dust)",),
        ld50_oral_mg_kg=3160.0,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,  # insoluble, minimal aquatic toxicity
        nano_lung_hazard=True,
        nano_dissolution=False,
        nano_ros_generation=False,
        nano_notes="Amorphous silica: lower lung hazard than crystalline. "
                   "Stöber silica is high purity amorphous — NOT quartz (Group 1). "
                   "Wear dust mask during handling. Embedded in binder = no dust exposure in use.",
        leaches_in_water=False,
        leaches_in_acid=False,
        safe_for_building=True,
        safe_for_textile=True,
        safe_for_cosmetic=True,
        source="IARC Monograph 68 (1997); ECETOC JACC 51",
    ),

    "TiO2_rutile": MaterialSafety(
        "TiO2_rutile", "Titanium dioxide (rutile)", "particle",
        IARCGroup.GROUP_2B,
        ("H351 Suspected of causing cancer (inhalation of dust)",),
        ld50_oral_mg_kg=-1,  # >10000 mg/kg (practically non-toxic orally)
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=True,
        nano_dissolution=False,
        nano_ros_generation=True,
        nano_notes="IARC 2B is based on inhalation of TiO₂ dust in rat studies. "
                   "Photocatalytic ROS generation under UV — a feature for self-cleaning "
                   "but a hazard for lung tissue. Embedded in polymer matrix eliminates "
                   "inhalation risk during normal use. Spray application requires RPE.",
        leaches_in_water=False,
        leaches_in_acid=False,
        warnings_for=("Wear RPE during spray application", "ROS generation under UV — avoid bare nano-TiO₂ on skin"),
        safe_for_building=True,
        safe_for_textile=True,  # OK when embedded in binder
        safe_for_cosmetic=True,  # widely used in sunscreen (rutile form)
        source="IARC Monograph 93 (2010); FDA sunscreen monograph",
    ),

    "TiO2_anatase": MaterialSafety(
        "TiO2_anatase", "Titanium dioxide (anatase)", "particle",
        IARCGroup.GROUP_2B,
        ("H351 Suspected of causing cancer (inhalation)",),
        ld50_oral_mg_kg=-1,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=True,
        nano_dissolution=False,
        nano_ros_generation=True,
        nano_notes="Anatase is MORE photocatalytically active than rutile → more ROS. "
                   "Prefer rutile for applications with UV exposure + skin contact.",
        leaches_in_water=False,
        leaches_in_acid=False,
        warnings_for=("Higher ROS than rutile — prefer rutile for textile/cosmetic",
                       "Wear RPE during spray application"),
        safe_for_building=True,
        safe_for_textile=True,  # if embedded
        safe_for_cosmetic=False,  # anatase not recommended for cosmetic (FDA prefers rutile)
        source="IARC Monograph 93 (2010)",
    ),

    "ZnS": MaterialSafety(
        "ZnS", "Zinc sulfide (sphalerite)", "particle",
        IARCGroup.NOT_CLASSIFIED,
        ("H302 Harmful if swallowed", "H332 Harmful if inhaled"),
        ld50_oral_mg_kg=7500.0,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=1.0,  # Zn²⁺ released → aquatic toxic
        nano_lung_hazard=True,
        nano_dissolution=True,
        nano_ros_generation=False,
        nano_notes="ZnS dissolves in acid (pH < 5) releasing Zn²⁺ and H₂S. "
                   "Zn²⁺ is acutely toxic to aquatic organisms (EC50 ~ 1 mg/L). "
                   "H₂S is toxic gas. NOT suitable for acidic matrices or textile (sweat).",
        leaches_in_water=False,
        leaches_in_acid=True,
        leach_products=("Zn²⁺", "H₂S (trace, at low pH)"),
        leach_notes="pH < 5 → dissolution. Sweat pH = 5.5 → borderline.",
        excluded_applications=("cosmetic"),
        warnings_for=("Acid dissolution releases Zn²⁺ — avoid acidic environments",
                       "Not recommended for textile in direct skin contact without sealing"),
        safe_for_building=True,
        safe_for_textile=False,  # sweat dissolution concern
        safe_for_cosmetic=False,
        source="ECHA registration; Heinlaan et al., Chemosphere 2008",
    ),

    "BaTiO3": MaterialSafety(
        "BaTiO3", "Barium titanate", "particle",
        IARCGroup.NOT_CLASSIFIED,
        ("H302 Harmful if swallowed (Ba²⁺ release)",),
        ld50_oral_mg_kg=-1,  # Ba²⁺ compounds: LD50 varies 100-500 mg/kg for soluble Ba
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=True,
        nano_dissolution=True,
        nano_ros_generation=False,
        nano_notes="BaTiO₃ is sparingly soluble but can release Ba²⁺ in acidic conditions. "
                   "Ba²⁺ is toxic (hypokalemia, cardiac effects). "
                   "Use only in sealed/embedded form — not as free nanoparticles.",
        leaches_in_water=False,
        leaches_in_acid=True,
        leach_products=("Ba²⁺"),
        leach_notes="Acid dissolution releases barium — toxic if ingested.",
        excluded_applications=("textile_coating", "cosmetic", "smart_textile"),
        safe_for_building=True,
        safe_for_textile=False,  # Ba²⁺ leaching into sweat
        safe_for_cosmetic=False,
        source="ECHA; Ba²⁺ toxicology literature",
    ),

    "polystyrene": MaterialSafety(
        "polystyrene", "Polystyrene spheres", "particle",
        IARCGroup.GROUP_3,
        (),
        ld50_oral_mg_kg=-1,  # inert polymer
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,  # insoluble; microplastic concern
        nano_lung_hazard=False,
        nano_dissolution=False,
        nano_ros_generation=False,
        nano_notes="PS is chemically inert. Concern is microplastic persistence, "
                   "not acute toxicity. Residual styrene monomer should be < 0.5% by GMP.",
        leaches_in_water=False,
        leaches_in_acid=False,
        leach_products=("styrene monomer (trace, from incomplete polymerization)"),
        leach_notes="Styrene monomer: IARC 2A. Properly polymerized PS has < 0.01% residual.",
        warnings_for=("Microplastic persistence in aquatic environments",
                       "Ensure residual styrene monomer < 0.5% (GMP standard)"),
        safe_for_building=True,
        safe_for_textile=True,
        safe_for_cosmetic=False,
        source="IARC Monograph 82 (2002); EFSA 2014 on styrene migration",
    ),

    "PMMA": MaterialSafety(
        "PMMA", "Poly(methyl methacrylate)", "particle",
        IARCGroup.GROUP_3,
        (),
        ld50_oral_mg_kg=-1,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=False,
        nano_dissolution=False,
        nano_ros_generation=False,
        nano_notes="PMMA is bioinert — used in bone cement, dental prosthetics, contact lenses.",
        leaches_in_water=False,
        leaches_in_acid=False,
        leach_products=("MMA monomer (trace)"),
        leach_notes="Residual MMA: skin sensitizer. Below 0.1% in properly cured PMMA.",
        safe_for_building=True,
        safe_for_textile=True,
        safe_for_cosmetic=True,
        source="IARC; FDA medical device classification",
    ),

    # ── Absorber materials ──

    "carbon_black": MaterialSafety(
        "carbon_black", "Carbon black", "absorber",
        IARCGroup.GROUP_2B,
        ("H351 Suspected of causing cancer (inhalation, rat lung overload)",),
        ld50_oral_mg_kg=-1,  # >10000 (practically non-toxic orally)
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,  # insoluble
        nano_lung_hazard=True,
        nano_dissolution=False,
        nano_ros_generation=False,
        nano_notes="IARC 2B based on rat inhalation (lung overload mechanism). "
                   "Embedded in polymer matrix: no inhalation exposure in use. "
                   "Carbon black is in every car tire, printer toner, and mascara — "
                   "the hazard is inhalation of free dust, not embedded particles.",
        leaches_in_water=False,
        leaches_in_acid=False,
        warnings_for=("Wear RPE during handling of dry powder",
                       "No concern when embedded in binder matrix"),
        safe_for_building=True,
        safe_for_textile=True,  # embedded
        safe_for_cosmetic=True,  # FDA-approved colorant (D&C standard)
        source="IARC Monograph 93 (2010); FDA 21 CFR 73.2030",
    ),

    "melanin": MaterialSafety(
        "melanin", "Melanin / polydopamine (PDA)", "absorber",
        IARCGroup.NOT_CLASSIFIED,
        (),
        ld50_oral_mg_kg=-1,  # bioidentical — no toxicity
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=False,
        nano_dissolution=False,
        nano_ros_generation=False,
        nano_notes="Melanin is the pigment in human skin and hair. "
                   "Polydopamine is a synthetic melanin analog. Both are biocompatible, "
                   "biodegradable, and non-toxic. The IDEAL absorber from a safety standpoint.",
        leaches_in_water=False,
        leaches_in_acid=False,
        safe_for_building=True,
        safe_for_textile=True,
        safe_for_cosmetic=True,
        source="Liu et al., ACS Nano 2013; PDA biocompatibility review",
    ),

    "iron_oxide": MaterialSafety(
        "iron_oxide", "Iron oxide (Fe₂O₃ / Fe₃O₄)", "absorber",
        IARCGroup.GROUP_3,
        (),
        ld50_oral_mg_kg=10000.0,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=True,
        nano_dissolution=False,
        nano_ros_generation=False,
        nano_notes="Iron oxide NPs are FDA-approved MRI contrast agents (Feridex). "
                   "Siderosis (lung iron deposition) from chronic inhalation of dust, "
                   "but this is a welding hazard, not a coatings hazard when embedded.",
        leaches_in_water=False,
        leaches_in_acid=True,
        leach_products=("Fe²⁺/Fe³⁺ (benign — natural background metal)"),
        safe_for_building=True,
        safe_for_textile=True,
        safe_for_cosmetic=True,
        source="IARC; FDA approved colorant CI 77491/77492/77499",
    ),

    # ── Matrix / binder materials ──

    "silicone": MaterialSafety(
        "silicone", "Silicone (PDMS)", "matrix",
        IARCGroup.NOT_CLASSIFIED,
        (),
        ld50_oral_mg_kg=-1,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=False, nano_dissolution=False, nano_ros_generation=False,
        leaches_in_water=False, leaches_in_acid=False,
        leach_products=("cyclic siloxanes (D4, D5 — trace)"),
        leach_notes="D4/D5 siloxanes: PBT concern at high levels. "
                    "Cured PDMS releases minimal volatiles.",
        safe_for_building=True, safe_for_textile=True, safe_for_cosmetic=True,
        source="ECHA SVHC evaluation; SCCS opinion on D4/D5",
    ),

    "polyurethane": MaterialSafety(
        "polyurethane", "Polyurethane (flexible)", "matrix",
        IARCGroup.NOT_CLASSIFIED,
        (),
        ld50_oral_mg_kg=-1,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=False, nano_dissolution=False, nano_ros_generation=False,
        leaches_in_water=False, leaches_in_acid=False,
        leach_products=("residual isocyanate monomer (MDI/TDI — trace if properly cured)"),
        leach_notes="UNCURED isocyanate is a severe respiratory sensitizer. "
                    "Fully cured PU is inert. Ensure full cure before deployment.",
        warnings_for=("Ensure full cure — residual isocyanate is toxic",
                       "Wear RPE during application of uncured PU"),
        safe_for_building=True, safe_for_textile=True, safe_for_cosmetic=False,
        source="ECHA; OSHA PEL for MDI/TDI",
    ),

    "epoxy": MaterialSafety(
        "epoxy", "Epoxy resin", "matrix",
        IARCGroup.NOT_CLASSIFIED,
        ("H317 May cause allergic skin reaction (uncured)",),
        ld50_oral_mg_kg=-1,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=False, nano_dissolution=False, nano_ros_generation=False,
        leaches_in_water=False, leaches_in_acid=False,
        leach_products=("BPA (from DGEBA epoxy — trace)", "residual hardener (amine)"),
        leach_notes="BPA is an endocrine disruptor. Use BPA-free epoxy for "
                    "food-contact or textile applications. Cured epoxy is inert.",
        warnings_for=("Uncured epoxy: skin sensitizer", "BPA migration from DGEBA-based epoxy"),
        excluded_applications=("textile_coating", "cosmetic"),
        safe_for_building=True, safe_for_textile=False, safe_for_cosmetic=False,
        source="ECHA; EFSA BPA opinion 2015",
    ),

    "PVA": MaterialSafety(
        "PVA", "Polyvinyl alcohol", "matrix",
        IARCGroup.NOT_CLASSIFIED,
        (),
        ld50_oral_mg_kg=-1,  # GRAS status
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=False, nano_dissolution=False, nano_ros_generation=False,
        leaches_in_water=True,  # PVA is water-soluble
        leaches_in_acid=False,
        leach_products=("PVA polymer (non-toxic, biodegradable)"),
        leach_notes="PVA dissolves in water — not suitable for wet environments unless crosslinked.",
        warnings_for=("Water-soluble: crosslink for wet applications"),
        safe_for_building=True, safe_for_textile=True, safe_for_cosmetic=True,
        source="FDA GRAS; JECFA evaluation",
    ),

    "CNC": MaterialSafety(
        "CNC", "Cellulose nanocrystals", "particle",
        IARCGroup.NOT_CLASSIFIED,
        (),
        ld50_oral_mg_kg=-1,  # cellulose is dietary fiber
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=-1,
        nano_lung_hazard=True,  # any nanofiber → lung concern
        nano_dissolution=False,
        nano_ros_generation=False,
        nano_notes="CNC is cellulose — the most abundant biopolymer on earth. "
                   "Nanofiber lung concern is precautionary (no chronic data yet). "
                   "Biodegradable. Non-toxic orally.",
        leaches_in_water=False, leaches_in_acid=False,
        warnings_for=("Nanofiber precautionary lung concern — wear mask during dry handling"),
        safe_for_building=True, safe_for_textile=True, safe_for_cosmetic=True,
        source="Endes et al., J. Nanobiotechnol. 2016; NCC safety review",
    ),

    # ── Chromophore / dye structural alerts ──
    # These are NOT used in structural color but might be proposed as absorbers

    "azo_dye": MaterialSafety(
        "azo_dye", "Azo dye (generic)", "chromophore",
        IARCGroup.GROUP_2A,  # many specific azo dyes are Group 1
        ("H350 May cause cancer (aromatic amine release on reductive cleavage)",
         "H341 Suspected of causing genetic defects"),
        ld50_oral_mg_kg=500.0,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=10.0,
        nano_lung_hazard=False, nano_dissolution=True, nano_ros_generation=False,
        leaches_in_water=True, leaches_in_acid=True,
        leach_products=("aromatic amines (carcinogenic)", "sulfonated fragments"),
        leach_notes="Azo bond cleaves under reducing conditions (gut bacteria, "
                    "anaerobic sediment) releasing aromatic amines — many are carcinogens. "
                    "THIS IS WHY STRUCTURAL COLOR EXISTS: to replace these.",
        excluded_applications=("textile_coating", "cosmetic", "smart_textile",
                                "protective_garment", "wall_tile"),
        safe_for_building=False, safe_for_textile=False, safe_for_cosmetic=False,
        source="IARC Monograph 99 (2010); EU REACH Annex XVII entry 43",
    ),

    "anthracycline": MaterialSafety(
        "anthracycline", "Anthracycline chromophore (doxorubicin class)", "chromophore",
        IARCGroup.GROUP_2A,
        ("H350 May cause cancer", "H361 Suspected reproductive toxin",
         "H370 Causes damage to heart (cardiotoxicity)"),
        ld50_oral_mg_kg=20.0,  # doxorubicin: extremely toxic
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=0.1,
        nano_lung_hazard=False, nano_dissolution=True, nano_ros_generation=True,
        nano_notes="DNA intercalator. Topoisomerase II poison. Cardiotoxic. "
                   "Beautiful red color is irrelevant — this is a cytotoxic drug, "
                   "not a structural color material.",
        leaches_in_water=True, leaches_in_acid=True,
        leach_products=("intact anthracycline (DNA intercalator)"),
        excluded_applications=("ALL"),
        safe_for_building=False, safe_for_textile=False, safe_for_cosmetic=False,
        source="IARC; WHO Essential Medicines List (as chemotherapy, not colorant)",
    ),

    "cadmium_pigment": MaterialSafety(
        "cadmium_pigment", "Cadmium pigment (CdS, CdSe)", "chromophore",
        IARCGroup.GROUP_1,
        ("H350 Causes cancer (lung)", "H341 Genetic defects",
         "H361fd Reproductive toxin", "H330 Fatal if inhaled"),
        ld50_oral_mg_kg=88.0,
        lc50_inhalation_mg_m3=0.5,
        ec50_aquatic_mg_L=0.01,
        nano_lung_hazard=True, nano_dissolution=True, nano_ros_generation=True,
        leaches_in_water=True, leaches_in_acid=True,
        leach_products=("Cd²⁺ (nephrotoxin, carcinogen)"),
        excluded_applications=("ALL"),
        safe_for_building=False, safe_for_textile=False, safe_for_cosmetic=False,
        source="IARC Monograph 100C (2012); EU REACH restriction",
    ),

    "lead_pigment": MaterialSafety(
        "lead_pigment", "Lead pigment (PbCrO₄, Pb₃O₄)", "chromophore",
        IARCGroup.GROUP_1,
        ("H350 Causes cancer", "H360 Reproductive toxin",
         "H332+H302 Harmful", "H410 Very toxic to aquatic life"),
        ld50_oral_mg_kg=-1,
        lc50_inhalation_mg_m3=-1,
        ec50_aquatic_mg_L=0.01,
        nano_lung_hazard=True, nano_dissolution=True, nano_ros_generation=False,
        leaches_in_water=True, leaches_in_acid=True,
        leach_products=("Pb²⁺ (neurotoxin)", "CrO₄²⁻ (carcinogen)"),
        excluded_applications=("ALL"),
        safe_for_building=False, safe_for_textile=False, safe_for_cosmetic=False,
        source="IARC; banned in most jurisdictions",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Design screening
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentFlag:
    """One safety flag for a design component."""
    component: str                   # "SiO2 particle", "carbon_black absorber"
    category: str                    # "carcinogen", "nano_hazard", "leaching", "persistence"
    severity: str                    # "info", "warning", "exclude"
    description: str
    route: str = ""                  # exposure route if relevant
    mitigation: str = ""             # what to do about it


@dataclass
class StructuralColorSafetyReport:
    """Complete safety report for a structural color design."""
    design_description: str
    application: str
    exposure_routes: list[ExposureRoute]

    # Per-component analysis
    components_assessed: list[str]
    component_flags: list[ComponentFlag]

    # Summary
    safe_for_application: bool = True
    exclusion_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    safety_score: float = 0.0        # 0-1

    # Safe alternatives
    alternatives: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "EXCLUDED" if not self.safe_for_application else \
                 "SAFE" if self.safety_score >= 0.8 else \
                 "CAUTION" if self.safety_score >= 0.5 else "UNSAFE"
        lines = [
            f"Structural Color Safety: [{status}]",
            f"  Application: {self.application}",
            f"  Exposure routes: {', '.join(r.value for r in self.exposure_routes)}",
            f"  Components: {', '.join(self.components_assessed)}",
            f"  Safety score: {self.safety_score:.2f}",
        ]
        if self.exclusion_reasons:
            for r in self.exclusion_reasons:
                lines.append(f"  ✗ {r}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.recommendations:
            for r in self.recommendations:
                lines.append(f"  → {r}")
        if self.alternatives:
            lines.append(f"  Alternatives: {'; '.join(self.alternatives)}")
        return "\n".join(lines)


def screen_design(
    particle_material: str,
    absorber_material: str = "carbon_black",
    matrix_material: str = "air",
    application: str = "building_panel",
    additional_components: Optional[list[str]] = None,
) -> StructuralColorSafetyReport:
    """Screen a structural color design for all safety hazards.

    Args:
        particle_material: "SiO2", "TiO2_rutile", "ZnS", etc.
        absorber_material: "carbon_black", "melanin", "iron_oxide", or chromophore name
        matrix_material: "silicone", "polyurethane", "epoxy", "PVA", "air"
        application: from multiphysics_profiles application IDs
        additional_components: any extra materials to screen

    Returns:
        StructuralColorSafetyReport with per-component analysis.
    """
    routes = _APPLICATION_ROUTES.get(application, [ExposureRoute.INHALATION])

    components = [particle_material, absorber_material]
    if matrix_material and matrix_material != "air":
        components.append(matrix_material)
    if additional_components:
        components.extend(additional_components)

    flags = []
    exclusions = []
    warnings = []
    recommendations = []
    component_scores = []

    for comp_id in components:
        mat = _MATERIALS.get(comp_id)
        if mat is None:
            flags.append(ComponentFlag(
                comp_id, "unknown", "warning",
                f"Material '{comp_id}' not in safety database — cannot assess",
                mitigation="Obtain SDS and conduct manual review",
            ))
            warnings.append(f"Unknown material: {comp_id}")
            component_scores.append(0.5)
            continue

        comp_score = 1.0

        # IARC classification
        if mat.iarc_group == IARCGroup.GROUP_1:
            flags.append(ComponentFlag(
                comp_id, "carcinogen", "exclude",
                f"{mat.name}: IARC Group 1 carcinogen",
                mitigation="Replace with non-carcinogenic alternative",
            ))
            exclusions.append(f"{mat.name} is IARC Group 1 (carcinogenic to humans)")
            comp_score = 0.0
        elif mat.iarc_group == IARCGroup.GROUP_2A:
            flags.append(ComponentFlag(
                comp_id, "carcinogen", "exclude",
                f"{mat.name}: IARC Group 2A (probably carcinogenic)",
                mitigation="Replace with safer alternative",
            ))
            exclusions.append(f"{mat.name} is IARC Group 2A (probably carcinogenic)")
            comp_score = 0.0
        elif mat.iarc_group == IARCGroup.GROUP_2B:
            # 2B: possibly carcinogenic — application-dependent
            if ExposureRoute.INHALATION in routes and mat.nano_lung_hazard:
                flags.append(ComponentFlag(
                    comp_id, "carcinogen", "warning",
                    f"{mat.name}: IARC 2B + inhalation route — ensure embedding in matrix",
                    route="inhalation",
                    mitigation="Embed in polymer binder; wear RPE during spray application",
                ))
                warnings.append(f"{mat.name}: IARC 2B — ensure no free dust exposure")
                comp_score *= 0.7

        # Nano-specific hazards
        if mat.nano_lung_hazard and ExposureRoute.INHALATION in routes:
            flags.append(ComponentFlag(
                comp_id, "nano_hazard", "warning",
                f"{mat.name}: nano lung deposition risk during handling",
                route="inhalation",
                mitigation="Embed in matrix; wear RPE during application",
            ))

        if mat.nano_dissolution:
            if ExposureRoute.DERMAL in routes or ExposureRoute.ORAL in routes:
                flags.append(ComponentFlag(
                    comp_id, "nano_hazard", "warning",
                    f"{mat.name}: nano dissolution releases {', '.join(mat.leach_products)}",
                    route="dermal/oral",
                    mitigation="Seal in polymer matrix; verify no leaching by migration test",
                ))
                comp_score *= 0.6

        if mat.nano_ros_generation and ExposureRoute.DERMAL in routes:
            flags.append(ComponentFlag(
                comp_id, "nano_hazard", "warning",
                f"{mat.name}: photocatalytic ROS generation on UV exposure",
                route="dermal",
                mitigation="Use rutile (not anatase); coat with silica shell to suppress ROS",
            ))
            comp_score *= 0.8

        # Leaching
        if mat.leaches_in_acid and ExposureRoute.DERMAL in routes:
            flags.append(ComponentFlag(
                comp_id, "leaching", "warning",
                f"{mat.name}: dissolves in acid (sweat pH 5.5) → {', '.join(mat.leach_products)}",
                route="dermal",
                mitigation="Seal particles in acid-resistant binder",
            ))
            comp_score *= 0.5

        if mat.leaches_in_water and ExposureRoute.AQUATIC in routes:
            flags.append(ComponentFlag(
                comp_id, "leaching", "info",
                f"{mat.name}: leaches into water → {', '.join(mat.leach_products)}",
                route="aquatic",
            ))

        # Application-specific exclusions
        if application in mat.excluded_applications or "ALL" in mat.excluded_applications:
            flags.append(ComponentFlag(
                comp_id, "application_excluded", "exclude",
                f"{mat.name}: excluded for {application} application",
                mitigation="Use alternative material",
            ))
            exclusions.append(f"{mat.name} excluded for {application}")
            comp_score = 0.0

        # Application-specific safety
        if "textile" in application and not mat.safe_for_textile:
            flags.append(ComponentFlag(
                comp_id, "application_excluded", "exclude",
                f"{mat.name}: not safe for textile application",
            ))
            exclusions.append(f"{mat.name} not textile-safe")
            comp_score = 0.0
        elif "cosmetic" in application and not mat.safe_for_cosmetic:
            flags.append(ComponentFlag(
                comp_id, "application_excluded", "exclude",
                f"{mat.name}: not safe for cosmetic application",
            ))
            exclusions.append(f"{mat.name} not cosmetic-safe")
            comp_score = 0.0

        component_scores.append(comp_score)

    # Generate alternatives for excluded components
    alternatives = []
    for exc in exclusions:
        if "absorber" in exc.lower() or "azo" in exc.lower() or "anthracycline" in exc.lower():
            alternatives.append("melanin/PDA absorber (bioidentical, non-toxic)")
            alternatives.append("iron_oxide absorber (FDA-approved, natural)")
        if "cadmium" in exc.lower() or "lead" in exc.lower():
            alternatives.append("Structural color replaces toxic pigments entirely")
        if "ZnS" in exc:
            alternatives.append("SiO2 particles (amorphous, non-dissolving)")
        if "BaTiO3" in exc:
            alternatives.append("TiO2_rutile (high n, non-dissolving, lower toxicity)")

    # Recommendations
    if any(mat.nano_lung_hazard for comp in components
           if (mat := _MATERIALS.get(comp)) is not None) and \
       ExposureRoute.INHALATION in routes:
        recommendations.append(
            "All nanoparticles require RPE during dry handling and spray application")
    if "textile" in application:
        recommendations.append(
            "Migration test (EN 71-3 or similar) recommended for skin-contact textiles")
    if ExposureRoute.AQUATIC in routes:
        recommendations.append(
            "Wash water testing for leached metals/particles recommended before scale-up")

    # Composite score
    if component_scores:
        safety_score = sum(component_scores) / len(component_scores)
    else:
        safety_score = 0.5

    safe = len(exclusions) == 0 and safety_score >= 0.3

    return StructuralColorSafetyReport(
        design_description=f"{particle_material} + {absorber_material} in {matrix_material}",
        application=application,
        exposure_routes=routes,
        components_assessed=components,
        component_flags=flags,
        safe_for_application=safe,
        exclusion_reasons=exclusions,
        warnings=warnings,
        recommendations=recommendations,
        safety_score=round(safety_score, 4),
        alternatives=list(set(alternatives)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def get_material_safety(material_id: str) -> Optional[MaterialSafety]:
    """Look up safety profile for a material."""
    return _MATERIALS.get(material_id)


def list_safe_particles(application: str = "building_panel") -> list[str]:
    """List particle materials safe for an application."""
    safe = []
    for mid, mat in _MATERIALS.items():
        if mat.category != "particle":
            continue
        if "textile" in application and not mat.safe_for_textile:
            continue
        if "cosmetic" in application and not mat.safe_for_cosmetic:
            continue
        if application in mat.excluded_applications or "ALL" in mat.excluded_applications:
            continue
        if mat.iarc_group in (IARCGroup.GROUP_1, IARCGroup.GROUP_2A):
            continue
        safe.append(mid)
    return safe


def list_safe_absorbers(application: str = "building_panel") -> list[str]:
    """List absorber materials safe for an application."""
    safe = []
    for mid, mat in _MATERIALS.items():
        if mat.category != "absorber":
            continue
        if "textile" in application and not mat.safe_for_textile:
            continue
        if application in mat.excluded_applications or "ALL" in mat.excluded_applications:
            continue
        if mat.iarc_group in (IARCGroup.GROUP_1, IARCGroup.GROUP_2A):
            continue
        safe.append(mid)
    return safe
