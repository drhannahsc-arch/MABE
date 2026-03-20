"""
core/harvesting_safety.py -- Safety Screening for Energy Harvesting Materials

Extends the structural color safety system with safety profiles for
energy harvesting materials: PV absorbers, thermoelectrics, piezoelectrics,
and transparent conductors.

Hard excludes:
  - PZT (lead zirconate titanate) for ANY application -- Pb content
  - Perovskite MAPbI3 for textile -- Pb leaching into sweat
  - CdTe for ANY application -- Cd toxicity (already banned in optical DB)

This module does NOT modify core/structural_color_safety.py (zero regression).
It provides a parallel `screen_harvesting_design()` entry point.

Phase 6 of the Energy Harvesting module.
Data tier: Tier 2 (values from REACH, IARC, literature).

References:
  - IARC Monographs (various volumes)
  - REACH Regulation (EC) No 1907/2006
  - Babayigit et al., Nat. Mater. 2016, 15, 247 (perovskite Pb toxicity)
  - Grandjean & Herz, J. Toxicol. Environ. Health A 2015, 78, 1029 (Pb in consumer products)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# Import shared types from existing safety module
from core.structural_color_safety import IARCGroup


# ---------------------------------------------------------------------------
# Harvesting material safety dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HarvestingMaterialSafety:
    """Safety profile for one energy harvesting material."""
    material_id: str
    name: str
    category: str                   # "pv_absorber", "thermoelectric", "piezo", "conductor"
    iarc_group: IARCGroup
    key_concern: str
    contains_lead: bool
    contains_cadmium: bool
    encapsulation_required: bool
    safe_for_building: bool
    safe_for_textile: bool
    hard_exclude_all: bool          # banned for ALL applications
    hard_exclude_textile: bool      # banned for textile only
    source: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Harvesting materials safety database
# ---------------------------------------------------------------------------

_HARVESTING_MATERIALS = {
    # PV absorbers
    "perovskite_MAPbI3": HarvestingMaterialSafety(
        material_id="perovskite_MAPbI3",
        name="Methylammonium lead iodide perovskite",
        category="pv_absorber",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Pb content; Pb leaching in rain/moisture",
        contains_lead=True,
        contains_cadmium=False,
        encapsulation_required=True,
        safe_for_building=True,     # with encapsulation
        safe_for_textile=False,     # Pb leaching into sweat
        hard_exclude_all=False,
        hard_exclude_textile=True,
        source="Babayigit et al. Nat. Mater. 2016, 15, 247",
        notes="Requires hermetic encapsulation to prevent Pb leaching. "
              "Rain exposure dissolves MA+ and releases Pb2+.",
    ),
    "perovskite_CsAgBiBr": HarvestingMaterialSafety(
        material_id="perovskite_CsAgBiBr",
        name="Cesium silver bismuth bromide double perovskite",
        category="pv_absorber",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Pb-free; Bi low toxicity",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=False,
        safe_for_building=True,
        safe_for_textile=True,
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="Slavney et al. J. Am. Chem. Soc. 2017, 139, 5015",
        notes="Lead-free alternative. Bismuth has low mammalian toxicity.",
    ),
    "organic_PV": HarvestingMaterialSafety(
        material_id="organic_PV",
        name="Organic photovoltaic absorbers (PM6:Y6, PBDB-T:ITIC)",
        category="pv_absorber",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Low concern; organic degradation products",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=False,
        safe_for_building=True,
        safe_for_textile=True,
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="Zimmermann et al. Adv. Energy Mater. 2022, 12, 2103692",
        notes="Carbon-based; degradation products are benign organic fragments.",
    ),

    # Thermoelectric
    "Bi2Te3": HarvestingMaterialSafety(
        material_id="Bi2Te3",
        name="Bismuth telluride",
        category="thermoelectric",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Te concern; encapsulation required",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=True,
        safe_for_building=True,
        safe_for_textile=True,     # with encapsulation
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="REACH Regulation (EC) 1907/2006; Te not classified carcinogenic",
        notes="Tellurium has moderate toxicity; encapsulation prevents exposure. "
              "Bi is low toxicity (Pepto-Bismol).",
    ),

    # Piezoelectric
    "PVDF": HarvestingMaterialSafety(
        material_id="PVDF",
        name="Polyvinylidene fluoride",
        category="piezo",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Fluoropolymer but NOT PFAS",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=False,
        safe_for_building=True,
        safe_for_textile=True,
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="OECD SIDS assessment; EPA PFAS definition excludes polymers",
        notes="Polymer-bound fluorine does NOT release PFOA/PFOS. "
              "Not a PFAS under EPA or OECD definitions. "
              "Stable to >300C; no HF release under normal conditions.",
    ),
    "P_VDF_TrFE": HarvestingMaterialSafety(
        material_id="P_VDF_TrFE",
        name="Poly(vinylidene fluoride-co-trifluoroethylene)",
        category="piezo",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Same as PVDF; fluoropolymer, not PFAS",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=False,
        safe_for_building=True,
        safe_for_textile=True,
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="OECD SIDS assessment",
        notes="Copolymer of VDF and TrFE. Same safety profile as PVDF.",
    ),

    # Conductors
    "PEDOT_PSS": HarvestingMaterialSafety(
        material_id="PEDOT_PSS",
        name="Poly(3,4-ethylenedioxythiophene):poly(styrene sulfonate)",
        category="conductor",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Benign; water-processable",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=False,
        safe_for_building=True,
        safe_for_textile=True,
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="Heraeus CLEVIOS safety data sheet; no GHS hazards",
        notes="Water-based dispersion. Used in bioelectronics with skin contact.",
    ),
    "AgNW": HarvestingMaterialSafety(
        material_id="AgNW",
        name="Silver nanowire network",
        category="conductor",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Ag nanoparticle concern at high concentrations",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=False,
        safe_for_building=True,
        safe_for_textile=True,
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="REACH; Ag compounds aquatic H400",
        notes="Ag+ ions are toxic to aquatic organisms. Embedded in matrix = low release. "
              "Avoid unencapsulated AgNW in aquatic-exposure applications.",
    ),
    "MXene_Ti3C2": HarvestingMaterialSafety(
        material_id="MXene_Ti3C2",
        name="MXene Ti3C2Tx",
        category="conductor",
        iarc_group=IARCGroup.NOT_CLASSIFIED,
        key_concern="Emerging material; limited tox data; Ti is benign",
        contains_lead=False,
        contains_cadmium=False,
        encapsulation_required=False,
        safe_for_building=True,
        safe_for_textile=True,
        hard_exclude_all=False,
        hard_exclude_textile=False,
        source="Jastrzebska et al. ACS Nano 2017, 11, 10834",
        notes="Ti is biocompatible (medical implants). Surface terminations (OH, F, O) "
              "may vary. Early cytotoxicity studies show low concern.",
    ),

    # ---- HARD EXCLUDES ----
    "PZT": HarvestingMaterialSafety(
        material_id="PZT",
        name="Lead zirconate titanate",
        category="piezo",
        iarc_group=IARCGroup.GROUP_2A,
        key_concern="Lead content (>60 wt% PbO)",
        contains_lead=True,
        contains_cadmium=False,
        encapsulation_required=True,
        safe_for_building=False,
        safe_for_textile=False,
        hard_exclude_all=True,
        hard_exclude_textile=True,
        source="EU RoHS Directive 2011/65/EU; IARC Monograph 87",
        notes="Contains >60 wt% PbO. Excluded from all MABE applications. "
              "Use BaTiO3, KNN, PVDF, or ZnO instead.",
    ),
    "CdTe": HarvestingMaterialSafety(
        material_id="CdTe",
        name="Cadmium telluride",
        category="pv_absorber",
        iarc_group=IARCGroup.GROUP_1,
        key_concern="Cadmium: Group 1 carcinogen",
        contains_lead=False,
        contains_cadmium=True,
        encapsulation_required=True,
        safe_for_building=False,
        safe_for_textile=False,
        hard_exclude_all=True,
        hard_exclude_textile=True,
        source="IARC Monograph 100C (2012); EU RoHS",
        notes="Cd is IARC Group 1 (carcinogenic). CdTe releases Cd2+ on damage. "
              "Excluded from all MABE applications.",
    ),
}


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------

def get_harvesting_material_safety(material_id: str) -> HarvestingMaterialSafety:
    """Look up safety profile for a harvesting material."""
    if material_id not in _HARVESTING_MATERIALS:
        raise KeyError(
            f"Unknown harvesting material '{material_id}'. "
            f"Available: {sorted(_HARVESTING_MATERIALS.keys())}"
        )
    return _HARVESTING_MATERIALS[material_id]


def list_harvesting_materials() -> List[str]:
    """Return sorted list of all harvesting material IDs in the safety DB."""
    return sorted(_HARVESTING_MATERIALS.keys())


def list_safe_harvesting_materials(
    category: Optional[str] = None,
    application: str = "building",
) -> List[str]:
    """
    List harvesting materials safe for a given application.

    Parameters
    ----------
    category : str or None
        Filter by category ("pv_absorber", "thermoelectric", "piezo", "conductor").
    application : str
        "building" or "textile".
    """
    results = []
    for mid, mat in _HARVESTING_MATERIALS.items():
        if mat.hard_exclude_all:
            continue
        if category and mat.category != category:
            continue
        if application == "textile" and (mat.hard_exclude_textile or not mat.safe_for_textile):
            continue
        if application == "building" and not mat.safe_for_building:
            continue
        results.append(mid)
    return sorted(results)


# ---------------------------------------------------------------------------
# Safety screening
# ---------------------------------------------------------------------------

@dataclass
class HarvestingSafetyFlag:
    """One safety flag for a harvesting material."""
    component: str              # material_id
    category: str               # "lead", "cadmium", "encapsulation", "emerging"
    severity: str               # "info", "warning", "exclude"
    description: str
    mitigation: str = ""


@dataclass
class HarvestingSafetyReport:
    """Complete safety report for a harvesting design."""
    application: str
    components_assessed: List[str]
    flags: List[HarvestingSafetyFlag]
    safe_for_application: bool
    exclusion_reasons: List[str]
    warnings: List[str]
    recommendations: List[str]

    def summary(self) -> str:
        status = "SAFE" if self.safe_for_application else "EXCLUDED"
        lines = [
            f"Harvesting Safety: [{status}]",
            f"  Application: {self.application}",
            f"  Components: {', '.join(self.components_assessed)}",
        ]
        for r in self.exclusion_reasons:
            lines.append(f"  EXCLUDE: {r}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for r in self.recommendations:
            lines.append(f"  REC: {r}")
        return "\n".join(lines)


def screen_harvesting_design(
    pv_material: str,
    te_material: str = "Bi2Te3",
    piezo_material: str = "PVDF",
    conductor: str = "PEDOT_PSS",
    application: str = "building",
) -> HarvestingSafetyReport:
    """
    Screen a harvesting design for safety.

    Checks all specified materials against the safety database.
    Returns a report with flags, exclusions, warnings, and recommendations.

    Parameters
    ----------
    pv_material : str
        PV absorber material (use "organic_PV" for organic materials,
        or specific perovskite names).
    te_material : str
        Thermoelectric material.
    piezo_material : str
        Piezoelectric material.
    conductor : str
        Conductor material.
    application : str
        "building" or "textile".

    Returns
    -------
    HarvestingSafetyReport
    """
    # Map PV material names to safety DB keys
    pv_safety_key = pv_material
    if pv_material.startswith("organic_"):
        pv_safety_key = "organic_PV"

    components = [pv_safety_key, te_material, piezo_material, conductor]
    assessed = []
    flags = []
    exclusions = []
    warnings = []
    recommendations = []
    safe = True

    for comp in components:
        if comp not in _HARVESTING_MATERIALS:
            # Unknown material — flag as info, don't exclude
            flags.append(HarvestingSafetyFlag(
                comp, "unknown", "info",
                f"Material '{comp}' not in harvesting safety database",
                mitigation="Verify safety data independently",
            ))
            warnings.append(f"'{comp}' not in safety database; verify independently")
            assessed.append(comp)
            continue

        mat = _HARVESTING_MATERIALS[comp]
        assessed.append(comp)

        # Hard excludes
        if mat.hard_exclude_all:
            flags.append(HarvestingSafetyFlag(
                comp, "hard_exclude", "exclude",
                f"{mat.name}: {mat.key_concern}",
            ))
            exclusions.append(f"{mat.name} excluded from ALL applications: {mat.key_concern}")
            safe = False
            continue

        if application == "textile" and mat.hard_exclude_textile:
            flags.append(HarvestingSafetyFlag(
                comp, "textile_exclude", "exclude",
                f"{mat.name}: {mat.key_concern} — excluded for textile",
            ))
            exclusions.append(f"{mat.name} excluded for textile: {mat.key_concern}")
            safe = False
            continue

        # Warnings
        if mat.contains_lead:
            flags.append(HarvestingSafetyFlag(
                comp, "lead", "warning",
                f"{mat.name} contains lead",
                mitigation="Hermetic encapsulation required; monitor for damage",
            ))
            warnings.append(f"{mat.name}: contains Pb — encapsulation critical")

        if mat.encapsulation_required:
            flags.append(HarvestingSafetyFlag(
                comp, "encapsulation", "warning",
                f"{mat.name} requires encapsulation",
                mitigation="Use hermetic barrier layers",
            ))
            recommendations.append(f"Encapsulate {mat.name} to prevent environmental release")

        if "emerging" in mat.key_concern.lower() or "limited" in mat.key_concern.lower():
            flags.append(HarvestingSafetyFlag(
                comp, "emerging", "info",
                f"{mat.name}: {mat.key_concern}",
                mitigation="Monitor literature for updated toxicology data",
            ))

    return HarvestingSafetyReport(
        application=application,
        components_assessed=assessed,
        flags=flags,
        safe_for_application=safe,
        exclusion_reasons=exclusions,
        warnings=warnings,
        recommendations=recommendations,
    )
