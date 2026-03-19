"""
core/substrate_anchor.py — Substrate Anchor Adapter

Generates click-chemistry tethering protocols for attaching capture elements
(molecular binders, MOF particles, cage scaffolds, nanoparticles) to
macroscopic deployment substrates (plastic netting, silica beads, glass, etc.).

Three-layer protocol:
  Layer A: Surface activation (substrate → reactive functional group)
  Layer B: Click handle installation (reactive group → click-ready surface)
  Layer C: Capture element coupling (click-ready surface + capture element → tethered system)

Design constraints:
  - SPAAC (DBCO-azide) is default for Cu-sensitive capture elements
  - CuAAC allowed only when capture element is Cu-tolerant
  - All protocols must work in aqueous conditions at ambient temperature
  - Protocols must be specific enough to be orderable/executable

Connects to:
  - core/click_attachment.py (binder-side handle analysis)
  - core/transform_enumerator.py (ClickCompatibility constraint)
  - core/candidate.py (ImmobilizationOption output)
  - core/inorganic_surfaces.py (silica/glass surface properties)

Data tier: Tier 2 (DOI + reference per protocol step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class SubstrateType(str, Enum):
    """Macroscopic deployment substrates."""
    SILICA_BEADS = "silica_beads"
    GLASS_SLIDE = "glass_slide"
    GLASS_BEADS = "glass_beads"
    PE_NETTING = "pe_netting"            # polyethylene
    PP_NETTING = "pp_netting"            # polypropylene
    NYLON_NETTING = "nylon_netting"
    PVDF_MEMBRANE = "pvdf_membrane"
    PS_BEADS = "ps_beads"                # polystyrene
    STAINLESS_STEEL_MESH = "stainless_steel_mesh"
    CARBON_FIBER_MESH = "carbon_fiber_mesh"
    CELLULOSE_FILTER = "cellulose_filter"


class ActivationMethod(str, Enum):
    """How to activate the substrate surface."""
    SILANIZATION_APTES = "silanization_aptes"    # → -NH₂
    SILANIZATION_GPTMS = "silanization_gptms"    # → epoxide
    SILANIZATION_MPTMS = "silanization_mptms"    # → -SH
    PLASMA_OXIDATION = "plasma_oxidation"        # → -COOH/-OH (on polymers)
    UV_OZONE = "uv_ozone"                        # → -COOH/-OH (on polymers)
    RADIATION_GRAFTING = "radiation_grafting"     # graft acrylic acid → -COOH
    PARTIAL_HYDROLYSIS = "partial_hydrolysis"     # nylon → expose -NH₂
    DEHYDROFLUORINATION = "dehydrofluorination"  # PVDF → C=C → thiol-ene
    PHOSPHONIC_ACID = "phosphonic_acid"          # metals → -PO₃H₂
    CATECHOL_ANCHOR = "catechol_anchor"          # metals → catechol (mussel-inspired)
    EDC_NHS = "edc_nhs"                          # -COOH → -NHS ester
    PERIODATE_OXIDATION = "periodate_oxidation"  # cellulose → -CHO (aldehyde)
    NONE = "none"                                # already has usable groups


class ClickChemistry(str, Enum):
    """Click coupling reaction for handle installation."""
    SPAAC = "spaac"                    # DBCO + azide, no catalyst
    CUAAC = "cuaac"                    # terminal alkyne + azide, Cu(I) catalyst
    THIOL_MALEIMIDE = "thiol_maleimide"  # thiol + maleimide, no catalyst
    OXIME_LIGATION = "oxime_ligation"    # aminooxy + aldehyde, no catalyst
    NHS_AMINE = "nhs_amine"              # NHS ester + amine, no catalyst

    @property
    def requires_copper(self) -> bool:
        return self == ClickChemistry.CUAAC

    @property
    def aqueous_compatible(self) -> bool:
        return True  # all of these work in water

    @property
    def ambient_temperature(self) -> bool:
        return True  # all work at RT


class CaptureElementType(str, Enum):
    """What kind of capture element is being tethered."""
    MOLECULAR_BINDER = "molecular_binder"        # small molecule / chelator
    MOF_PARTICLE = "mof_particle"                # 100nm-10µm MOF crystals
    METAL_OXIDE_NP = "metal_oxide_np"            # ZrO₂, FeOOH, TiO₂ NPs
    DNA_ORIGAMI_CAGE = "dna_origami_cage"        # ATHENA/DAEDALUS cage
    COORDINATION_CAGE = "coordination_cage"      # Fujita/Nitschke cage
    POLYMER_CAGE = "polymer_cage"                # POC
    ENZYME_MIMIC = "enzyme_mimic"                # Zn-CA mimic, etc.
    PHOTOCATALYST = "photocatalyst"              # TiO₂, g-C₃N₄
    SULFIDE_NP = "sulfide_np"                    # FeS₂ for heavy metal capture
    ZVI_NP = "zvi_np"                            # zerovalent iron


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubstrateProperties:
    """Physical and chemical properties of a substrate material."""
    substrate_type: SubstrateType
    name: str
    native_surface: str              # "Si-OH", "inert C-H", "-NH-", "C-F", etc.
    surface_area_m2_g: float         # available surface area
    mechanical_form: str             # "beads 50-200µm", "netting 1mm mesh", "flat slide"
    chemical_stability_ph: tuple[float, float]  # pH range for stability
    max_temperature_c: float         # thermal stability limit
    cost_usd_per_m2: float           # approximate surface cost
    notes: str = ""


@dataclass
class ActivationProtocol:
    """Step-by-step protocol for activating a substrate surface."""
    method: ActivationMethod
    resulting_group: str             # "-NH₂", "epoxide", "-SH", "-COOH", "-CHO"
    site_density_per_nm2: float      # estimated reactive sites after activation

    # Protocol steps
    reagents: list[str]              # what you need
    solvent: str                     # reaction medium
    temperature_c: float             # reaction temperature
    time_hours: float                # reaction time
    steps: list[str]                 # ordered procedure steps

    # Quality
    reproducibility: str             # "high", "moderate", "variable"
    literature_ref: str              # DOI or textbook reference
    notes: str = ""


@dataclass
class HandleInstallation:
    """Protocol for converting activated surface to click-ready surface."""
    click_chemistry: ClickChemistry
    handle_installed: str            # "azide (-N₃)", "DBCO", "alkyne", "maleimide"
    complementary_group: str         # what the capture element needs

    # Protocol
    reagent: str                     # "NHS-PEG₄-azide", "NHS-DBCO", etc.
    reagent_concentration: str       # e.g. "1 mM in PBS"
    solvent: str
    temperature_c: float
    time_hours: float
    steps: list[str]

    # Economics
    reagent_cost_per_mmol: float     # USD
    commercial_source: str           # "Click Chemistry Tools", "Sigma", etc.
    catalog_example: str             # example catalog number

    literature_ref: str
    notes: str = ""


@dataclass
class CouplingProtocol:
    """Protocol for coupling capture element to click-ready substrate."""
    capture_element_type: CaptureElementType
    capture_element_handle: str      # "DBCO", "azide", "thiol", etc.

    # Protocol
    concentration: str               # e.g. "10 µM in PBS"
    solvent: str
    temperature_c: float
    time_hours: float
    steps: list[str]

    # Expected performance
    coupling_efficiency_pct: float   # estimated % of surface sites coupled
    surface_density: str             # "~10¹² elements/cm²" or "1 per 100 nm²"

    literature_ref: str
    notes: str = ""


@dataclass
class TetherProtocol:
    """Complete three-layer tethering protocol.

    This is the full output: substrate → activate → handle → capture element.
    """
    # Identity
    substrate: SubstrateProperties
    capture_element_type: CaptureElementType
    click_chemistry: ClickChemistry

    # Three layers
    activation: ActivationProtocol
    handle: HandleInstallation
    coupling: CouplingProtocol

    # Compatibility
    cu_safe: bool                    # True if no Cu exposure to capture element
    aqueous_protocol: bool = True    # entire protocol in water-compatible conditions
    ambient_temperature: bool = True # no step >40°C after element coupling

    # Summary
    total_steps: int = 0
    estimated_time_hours: float = 0.0
    estimated_cost_per_cm2: float = 0.0

    # Warnings
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Tether Protocol: {self.substrate.name} → {self.capture_element_type.value}",
            f"  Click chemistry: {self.click_chemistry.value} ({'Cu-free' if self.cu_safe else 'Cu catalyst'})",
            f"  Activation: {self.activation.method.value} → {self.activation.resulting_group}",
            f"  Handle: {self.handle.handle_installed} (complementary: {self.handle.complementary_group})",
            f"  Total steps: {self.total_steps}",
            f"  Estimated time: {self.estimated_time_hours:.1f} hours",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {'; '.join(self.warnings)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Substrate database — hardcoded properties
# ═══════════════════════════════════════════════════════════════════════════

_SUBSTRATES: dict[SubstrateType, SubstrateProperties] = {
    SubstrateType.SILICA_BEADS: SubstrateProperties(
        substrate_type=SubstrateType.SILICA_BEADS,
        name="Silica beads (50-200 µm)",
        native_surface="Si-OH (silanol)",
        surface_area_m2_g=300.0,  # typical mesoporous silica
        mechanical_form="Packed bed or mesh-retained beads",
        chemical_stability_ph=(1.0, 12.0),
        max_temperature_c=800.0,
        cost_usd_per_m2=0.01,
        notes="Most versatile substrate. Compatible with silanization, high surface area.",
    ),
    SubstrateType.GLASS_SLIDE: SubstrateProperties(
        substrate_type=SubstrateType.GLASS_SLIDE,
        name="Borosilicate glass slide",
        native_surface="Si-OH (silanol)",
        surface_area_m2_g=0.001,  # flat, geometric only
        mechanical_form="Flat slide (25 × 75 mm typical)",
        chemical_stability_ph=(1.0, 12.0),
        max_temperature_c=500.0,
        cost_usd_per_m2=5.0,
        notes="Low surface area. Best for sensor/diagnostic applications, not bulk capture.",
    ),
    SubstrateType.GLASS_BEADS: SubstrateProperties(
        substrate_type=SubstrateType.GLASS_BEADS,
        name="Glass beads (100-500 µm)",
        native_surface="Si-OH (silanol)",
        surface_area_m2_g=0.5,
        mechanical_form="Packed bed or mesh-retained beads",
        chemical_stability_ph=(1.0, 12.0),
        max_temperature_c=500.0,
        cost_usd_per_m2=0.5,
        notes="Robust, cheap, reusable. Lower surface area than silica beads.",
    ),
    SubstrateType.PE_NETTING: SubstrateProperties(
        substrate_type=SubstrateType.PE_NETTING,
        name="Polyethylene netting (1-5 mm mesh)",
        native_surface="Inert C-H (polyolefin)",
        surface_area_m2_g=0.1,
        mechanical_form="Open mesh netting, roll or panel",
        chemical_stability_ph=(1.0, 13.0),
        max_temperature_c=80.0,
        cost_usd_per_m2=2.0,
        notes="Cheap, flexible, UV-resistant grades available. Requires plasma activation.",
    ),
    SubstrateType.PP_NETTING: SubstrateProperties(
        substrate_type=SubstrateType.PP_NETTING,
        name="Polypropylene netting (1-5 mm mesh)",
        native_surface="Inert C-H (polyolefin)",
        surface_area_m2_g=0.1,
        mechanical_form="Open mesh netting, roll or panel",
        chemical_stability_ph=(1.0, 13.0),
        max_temperature_c=100.0,
        cost_usd_per_m2=1.5,
        notes="Slightly more rigid than PE. Same activation chemistry.",
    ),
    SubstrateType.NYLON_NETTING: SubstrateProperties(
        substrate_type=SubstrateType.NYLON_NETTING,
        name="Nylon-6,6 netting (0.5-2 mm mesh)",
        native_surface="-NH- (amide backbone)",
        surface_area_m2_g=0.2,
        mechanical_form="Open mesh netting",
        chemical_stability_ph=(2.0, 11.0),
        max_temperature_c=120.0,
        cost_usd_per_m2=3.0,
        notes="Has amide N-H accessible by partial hydrolysis. Easier activation than polyolefins.",
    ),
    SubstrateType.PVDF_MEMBRANE: SubstrateProperties(
        substrate_type=SubstrateType.PVDF_MEMBRANE,
        name="PVDF membrane (0.2-5 µm pore)",
        native_surface="C-F (fluoropolymer)",
        surface_area_m2_g=10.0,
        mechanical_form="Flat or hollow fiber membrane",
        chemical_stability_ph=(1.0, 13.0),
        max_temperature_c=140.0,
        cost_usd_per_m2=50.0,
        notes="Excellent chemical resistance. Dehydrofluorination creates C=C for thiol-ene.",
    ),
    SubstrateType.PS_BEADS: SubstrateProperties(
        substrate_type=SubstrateType.PS_BEADS,
        name="Polystyrene beads (10-500 µm)",
        native_surface="Aromatic C-H (benzene ring)",
        surface_area_m2_g=1.0,  # porous PS up to 50 m²/g
        mechanical_form="Beads, packed or suspended",
        chemical_stability_ph=(1.0, 13.0),
        max_temperature_c=80.0,
        cost_usd_per_m2=0.1,
        notes="Available with pre-functionalized variants (chloromethyl, amino, carboxy).",
    ),
    SubstrateType.STAINLESS_STEEL_MESH: SubstrateProperties(
        substrate_type=SubstrateType.STAINLESS_STEEL_MESH,
        name="Stainless steel mesh (316L, 100-500 µm)",
        native_surface="Cr₂O₃/Fe₂O₃ passive oxide layer",
        surface_area_m2_g=0.01,
        mechanical_form="Woven mesh, panel or roll",
        chemical_stability_ph=(2.0, 12.0),
        max_temperature_c=600.0,
        cost_usd_per_m2=20.0,
        notes="Conductive. Enables electrochemical pathways (future adapter). Phosphonic acid anchoring.",
    ),
    SubstrateType.CARBON_FIBER_MESH: SubstrateProperties(
        substrate_type=SubstrateType.CARBON_FIBER_MESH,
        name="Carbon fiber mesh / cloth",
        native_surface="Graphitic C (some edge -COOH/-OH)",
        surface_area_m2_g=1.0,
        mechanical_form="Woven cloth or felt",
        chemical_stability_ph=(1.0, 13.0),
        max_temperature_c=400.0,
        cost_usd_per_m2=30.0,
        notes="Conductive. Edge sites functionalize more easily than basal plane.",
    ),
    SubstrateType.CELLULOSE_FILTER: SubstrateProperties(
        substrate_type=SubstrateType.CELLULOSE_FILTER,
        name="Cellulose filter paper / membrane",
        native_surface="-OH (glucose hydroxyl groups)",
        surface_area_m2_g=5.0,
        mechanical_form="Paper or membrane sheet",
        chemical_stability_ph=(3.0, 10.0),
        max_temperature_c=150.0,
        cost_usd_per_m2=0.5,
        notes="Cheap, biodegradable. Periodate oxidation gives -CHO for oxime ligation.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Activation protocols per substrate
# ═══════════════════════════════════════════════════════════════════════════

def _silica_glass_activation(method: str = "aptes") -> ActivationProtocol:
    """Silanization protocols for silica and glass substrates."""
    if method == "aptes":
        return ActivationProtocol(
            method=ActivationMethod.SILANIZATION_APTES,
            resulting_group="-NH₂",
            site_density_per_nm2=3.0,  # Vansant et al., 1995
            reagents=["APTES (3-aminopropyltriethoxysilane)", "toluene (anhydrous) or ethanol/water 95:5"],
            solvent="toluene (anhydrous) or ethanol/water",
            temperature_c=70.0,
            time_hours=4.0,
            steps=[
                "Clean substrate: piranha (3:1 H₂SO₄:H₂O₂) 30 min, rinse 3× DI water, dry N₂",
                "Immerse in 2% (v/v) APTES in anhydrous toluene, 70°C, 4 h under N₂",
                "Rinse: toluene 3×, ethanol 3×, DI water 3×",
                "Cure: 110°C oven, 1 h (crosslinks siloxane bonds)",
                "Verify: ninhydrin test on beads (purple = amine present)",
            ],
            reproducibility="high",
            literature_ref="Howarter & Youngblood, Langmuir 2006, 22, 11142; Vansant et al., 1995",
            notes="Most common and well-characterized silane. Gives primary amine for NHS coupling.",
        )
    elif method == "gptms":
        return ActivationProtocol(
            method=ActivationMethod.SILANIZATION_GPTMS,
            resulting_group="epoxide",
            site_density_per_nm2=2.5,
            reagents=["GPTMS (3-glycidoxypropyltrimethoxysilane)", "toluene (anhydrous)"],
            solvent="toluene (anhydrous)",
            temperature_c=70.0,
            time_hours=6.0,
            steps=[
                "Clean substrate: piranha 30 min, rinse 3× DI water, dry N₂",
                "Immerse in 2% (v/v) GPTMS in anhydrous toluene, 70°C, 6 h under N₂",
                "Rinse: toluene 3×, acetone 1×, dry N₂",
                "Use immediately or store desiccated at 4°C (epoxides hydrolyze slowly)",
            ],
            reproducibility="high",
            literature_ref="Acres et al., J. Phys. Chem. C 2012, 116, 6289",
            notes="Epoxide ring opens with azide-amine for direct azide installation.",
        )
    else:  # mptms
        return ActivationProtocol(
            method=ActivationMethod.SILANIZATION_MPTMS,
            resulting_group="-SH",
            site_density_per_nm2=2.0,
            reagents=["MPTMS (3-mercaptopropyltrimethoxysilane)", "toluene (anhydrous)"],
            solvent="toluene (anhydrous)",
            temperature_c=70.0,
            time_hours=4.0,
            steps=[
                "Clean substrate: piranha 30 min, rinse 3× DI water, dry N₂",
                "Immerse in 2% (v/v) MPTMS in anhydrous toluene, 70°C, 4 h under N₂",
                "Rinse: toluene 3×, ethanol 3×, dry N₂",
                "Verify: Ellman's test (DTNB, yellow = thiol present)",
                "Store under N₂ (thiols oxidize in air)",
            ],
            reproducibility="moderate",
            literature_ref="De Palma et al., Chem. Mater. 2007, 19, 1821",
            notes="Gives thiol for maleimide coupling. Sensitive to oxidation.",
        )


def _polyolefin_activation() -> ActivationProtocol:
    """Plasma/UV-ozone activation for PE and PP netting."""
    return ActivationProtocol(
        method=ActivationMethod.PLASMA_OXIDATION,
        resulting_group="-COOH / -OH (mixed)",
        site_density_per_nm2=1.5,  # varies with treatment time
        reagents=["O₂ plasma (or air plasma)", "EDC (1-ethyl-3-(3-dimethylaminopropyl)carbodiimide)",
                  "NHS (N-hydroxysuccinimide)"],
        solvent="air (plasma) then MES buffer pH 6.0 (EDC/NHS)",
        temperature_c=25.0,
        time_hours=1.0,
        steps=[
            "O₂ plasma treatment: 50 W, 100 mTorr, 5 min (or air plasma, 10 min)",
            "  Alternative: UV-ozone cleaner, 30 min exposure",
            "Immediately after plasma: immerse in 50 mM EDC + 50 mM NHS in MES pH 6.0, RT, 30 min",
            "  This converts -COOH to -NHS ester (stable for ~4 h at RT)",
            "Rinse: MES buffer 2×",
            "Proceed directly to handle installation (NHS ester is reactive)",
        ],
        reproducibility="moderate",
        literature_ref="Grace & Gerenser, J. Dispersion Sci. Technol. 2003, 24, 305; "
                        "Goddard & Hotchkiss, Prog. Polym. Sci. 2007, 32, 698",
        notes="Must proceed to handle installation within 4 h (NHS ester hydrolysis). "
              "Contact angle measurement confirms activation (drop from >90° to <50°).",
    )


def _nylon_activation() -> ActivationProtocol:
    """Partial hydrolysis to expose free -NH₂ on nylon."""
    return ActivationProtocol(
        method=ActivationMethod.PARTIAL_HYDROLYSIS,
        resulting_group="-NH₂ (free amine from cleaved amide)",
        site_density_per_nm2=0.8,
        reagents=["3M HCl", "DI water"],
        solvent="3M HCl (aqueous)",
        temperature_c=40.0,
        time_hours=2.0,
        steps=[
            "Immerse nylon netting in 3M HCl, 40°C, 2 h",
            "  This partially hydrolyzes surface amide bonds: -CO-NH- → -COOH + H₂N-",
            "Rinse: DI water 5× until pH neutral",
            "Dry under N₂ or vacuum",
            "  Free -NH₂ groups now available for NHS-ester or direct click conjugation",
        ],
        reproducibility="moderate",
        literature_ref="Jia & McCarthy, Langmuir 2002, 18, 683",
        notes="Over-hydrolysis weakens the nylon. Monitor tensile strength if critical.",
    )


def _pvdf_activation() -> ActivationProtocol:
    """Dehydrofluorination of PVDF → C=C for thiol-ene."""
    return ActivationProtocol(
        method=ActivationMethod.DEHYDROFLUORINATION,
        resulting_group="C=C (vinyl, from HF elimination)",
        site_density_per_nm2=1.0,
        reagents=["NaOH or KOH (10% w/v)", "DI water",
                  "thiol-PEG-azide (for thiol-ene click)"],
        solvent="NaOH/water then neat thiol-ene",
        temperature_c=60.0,
        time_hours=3.0,
        steps=[
            "Immerse PVDF membrane in 10% NaOH, 60°C, 2 h",
            "  Eliminates HF from -CH₂-CF₂- → -CH=CF- (conjugated vinyl)",
            "Rinse: DI water 5× until pH neutral",
            "Thiol-ene coupling: apply thiol-PEG₄-azide (neat or 10% in ethanol)",
            "  UV irradiation (365 nm, 10 mW/cm²) 30 min, or thermal (60°C, 2 h)",
            "  Thiol radical adds across C=C → covalent S-C bond → azide handle installed",
            "Rinse: ethanol 2×, DI water 3×",
        ],
        reproducibility="moderate",
        literature_ref="Hester et al., Macromolecules 2002, 35, 7652; "
                        "Liu et al., J. Membr. Sci. 2011, 369, 1",
        notes="Color change (white → brown/tan) confirms dehydrofluorination. "
              "Thiol-ene installs azide handle in same step as activation.",
    )


def _metal_mesh_activation(method: str = "phosphonic") -> ActivationProtocol:
    """Phosphonic acid or catechol anchoring for metal substrates."""
    if method == "phosphonic":
        return ActivationProtocol(
            method=ActivationMethod.PHOSPHONIC_ACID,
            resulting_group="-PO₃H₂ (with distal -NH₂ or -N₃)",
            site_density_per_nm2=2.0,
            reagents=["11-aminoundecylphosphonic acid (or azido-PEG-phosphonic acid)",
                      "ethanol/water 1:1"],
            solvent="ethanol/water 1:1",
            temperature_c=60.0,
            time_hours=12.0,
            steps=[
                "Clean metal mesh: sonicate in acetone 10 min, then ethanol 10 min",
                "Oxidize surface: O₂ plasma 5 min or immerse in 1M NaOH 30 min → rinse",
                "Immerse in 1 mM phosphonic acid derivative in ethanol/water 1:1, 60°C, 12 h",
                "Rinse: ethanol 3×, DI water 3×",
                "Cure: 120°C oven, 1 h (condenses P-O-M bonds)",
                "  Free -NH₂ or -N₃ at distal end now available for click",
            ],
            reproducibility="high",
            literature_ref="Guerrero et al., Chem. Mater. 2001, 13, 4367; "
                            "Textor et al., Langmuir 2000, 16, 3257",
            notes="Phosphonic acid SAMs on metal oxides are among the most stable organic coatings.",
        )
    else:  # catechol
        return ActivationProtocol(
            method=ActivationMethod.CATECHOL_ANCHOR,
            resulting_group="catechol-PEG-NH₂ (mussel-inspired)",
            site_density_per_nm2=1.5,
            reagents=["dopamine-PEG-NH₂ or dopamine-PEG-N₃",
                      "Tris buffer pH 8.5"],
            solvent="10 mM Tris-HCl pH 8.5",
            temperature_c=25.0,
            time_hours=4.0,
            steps=[
                "Clean metal mesh: sonicate in ethanol 10 min",
                "Immerse in 1 mg/mL dopamine-PEG-NH₂ in 10 mM Tris pH 8.5, RT, 4 h",
                "  Catechol chelates to metal oxide surface spontaneously",
                "Rinse: DI water 3×",
                "  Free -NH₂ or -N₃ at PEG terminus ready for click",
            ],
            reproducibility="moderate",
            literature_ref="Lee et al., Science 2007, 318, 426 (mussel-inspired)",
            notes="Works on almost any surface (metal, oxide, polymer) but coating can be patchy.",
        )


def _cellulose_activation() -> ActivationProtocol:
    """Periodate oxidation of cellulose → aldehyde groups."""
    return ActivationProtocol(
        method=ActivationMethod.PERIODATE_OXIDATION,
        resulting_group="-CHO (aldehyde, from vicinal diol cleavage)",
        site_density_per_nm2=2.0,
        reagents=["Sodium periodate (NaIO₄)", "DI water"],
        solvent="DI water",
        temperature_c=25.0,
        time_hours=2.0,
        steps=[
            "Immerse cellulose in 10 mM NaIO₄ in DI water, RT, 2 h, in the dark",
            "  Periodate cleaves C2-C3 vicinal diol → two aldehyde groups per glucose",
            "Rinse: DI water 5×",
            "  -CHO groups now available for oxime ligation (aminooxy-capture-element)",
            "  Or: reductive amination with amine-bearing capture element + NaBH₃CN",
        ],
        reproducibility="high",
        literature_ref="Kim et al., Cellulose 2004, 11, 207; Malaprade reaction",
        notes="Degree of oxidation controlled by periodate concentration and time. "
              "Over-oxidation degrades cellulose backbone.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Handle installation database
# ═══════════════════════════════════════════════════════════════════════════

def _handle_for_amine_surface(click: ClickChemistry) -> HandleInstallation:
    """Install click handle on an -NH₂ activated surface."""
    if click == ClickChemistry.SPAAC:
        return HandleInstallation(
            click_chemistry=ClickChemistry.SPAAC,
            handle_installed="-N₃ (azide)",
            complementary_group="DBCO on capture element",
            reagent="NHS-PEG₄-azide",
            reagent_concentration="1 mM in PBS pH 7.4",
            solvent="PBS pH 7.4",
            temperature_c=25.0,
            time_hours=2.0,
            steps=[
                "Dissolve NHS-PEG₄-azide in PBS pH 7.4 to 1 mM",
                "Incubate activated substrate in reagent solution, RT, 2 h",
                "Rinse: PBS 3×, DI water 2×",
                "Surface now bears -NH-CO-PEG₄-N₃ (azide-terminated)",
            ],
            reagent_cost_per_mmol=50.0,
            commercial_source="Click Chemistry Tools, BroadPharm, Sigma-Aldrich",
            catalog_example="CCT-1065 (Click Chemistry Tools)",
            literature_ref="Agard et al., ACS Chem. Biol. 2006, 1, 644 (SPAAC)",
        )
    elif click == ClickChemistry.CUAAC:
        return HandleInstallation(
            click_chemistry=ClickChemistry.CUAAC,
            handle_installed="-N₃ (azide)",
            complementary_group="terminal alkyne on capture element",
            reagent="NHS-PEG₄-azide",
            reagent_concentration="1 mM in PBS pH 7.4",
            solvent="PBS pH 7.4",
            temperature_c=25.0,
            time_hours=2.0,
            steps=[
                "Dissolve NHS-PEG₄-azide in PBS pH 7.4 to 1 mM",
                "Incubate activated substrate, RT, 2 h",
                "Rinse: PBS 3×, DI water 2×",
                "Surface now bears azide for CuAAC",
            ],
            reagent_cost_per_mmol=50.0,
            commercial_source="Click Chemistry Tools, Sigma-Aldrich",
            catalog_example="CCT-1065",
            literature_ref="Rostovtsev et al., Angew. Chem. Int. Ed. 2002, 41, 2596",
            notes="Same handle as SPAAC, but coupling step uses Cu(I) catalyst.",
        )
    elif click == ClickChemistry.NHS_AMINE:
        return HandleInstallation(
            click_chemistry=ClickChemistry.NHS_AMINE,
            handle_installed="-NH₂ (the surface amine itself)",
            complementary_group="NHS ester on capture element",
            reagent="None — surface amine is the handle",
            reagent_concentration="N/A",
            solvent="PBS pH 7.4",
            temperature_c=25.0,
            time_hours=0.0,
            steps=["No handle installation needed — surface -NH₂ reacts directly with NHS-ester "
                   "on capture element"],
            reagent_cost_per_mmol=0.0,
            commercial_source="N/A",
            catalog_example="N/A",
            literature_ref="Hermanson, Bioconjugate Techniques, 3rd ed., 2013",
            notes="Simplest path. NHS-ester must be on the capture element side.",
        )
    else:
        # DBCO on surface, azide on element (reversed orientation)
        return HandleInstallation(
            click_chemistry=ClickChemistry.SPAAC,
            handle_installed="DBCO",
            complementary_group="azide (-N₃) on capture element",
            reagent="NHS-PEG₄-DBCO",
            reagent_concentration="0.5 mM in PBS pH 7.4",
            solvent="PBS pH 7.4",
            temperature_c=25.0,
            time_hours=2.0,
            steps=[
                "Dissolve NHS-PEG₄-DBCO in PBS pH 7.4 to 0.5 mM",
                "Incubate activated substrate, RT, 2 h (dark — DBCO is light-sensitive)",
                "Rinse: PBS 3× (dark)",
                "Surface now bears -NH-CO-PEG₄-DBCO",
            ],
            reagent_cost_per_mmol=200.0,
            commercial_source="Click Chemistry Tools, BroadPharm",
            catalog_example="CCT-A102 (Click Chemistry Tools)",
            literature_ref="Agard et al., ACS Chem. Biol. 2006, 1, 644",
            notes="DBCO is more expensive than azide. Use this orientation if capture element "
                  "already has azide (e.g., azide-functionalized MOF particles).",
        )


def _handle_for_thiol_surface() -> HandleInstallation:
    """Install maleimide handle on -SH surface."""
    return HandleInstallation(
        click_chemistry=ClickChemistry.THIOL_MALEIMIDE,
        handle_installed="-SH (the surface thiol itself)",
        complementary_group="maleimide on capture element",
        reagent="None — surface thiol reacts directly with maleimide-capture-element",
        reagent_concentration="N/A",
        solvent="PBS pH 6.5-7.0 (maleimide-thiol optimal pH)",
        temperature_c=25.0,
        time_hours=0.0,
        steps=["No handle installation needed — surface -SH reacts directly with maleimide "
               "on capture element at pH 6.5-7.0, RT"],
        reagent_cost_per_mmol=0.0,
        commercial_source="N/A",
        catalog_example="N/A",
        literature_ref="Hermanson, Bioconjugate Techniques, 3rd ed., 2013",
        notes="Fastest coupling (~k = 500 M⁻¹s⁻¹ at pH 7). Thiol must be reduced (not disulfide).",
    )


def _handle_for_epoxide_surface() -> HandleInstallation:
    """Convert epoxide surface to azide via ring-opening with azide-amine."""
    return HandleInstallation(
        click_chemistry=ClickChemistry.SPAAC,
        handle_installed="-N₃ (azide, from epoxide ring-opening)",
        complementary_group="DBCO on capture element",
        reagent="sodium azide (NaN₃) or 11-azido-3,6,9-trioxaundecan-1-amine",
        reagent_concentration="100 mM NaN₃ in DMF/water 1:1, or 10 mM azido-PEG-amine in water",
        solvent="DMF/water (NaN₃) or water (azido-PEG-amine)",
        temperature_c=60.0,
        time_hours=12.0,
        steps=[
            "Option A (NaN₃): immerse in 100 mM NaN₃ in DMF/H₂O 1:1, 60°C, 12 h",
            "  Azide opens epoxide → -CH(OH)-CH₂-N₃",
            "Option B (azido-PEG-amine): immerse in 10 mM in water, 60°C, 6 h",
            "  Amine opens epoxide → -CH(OH)-CH₂-NH-PEG-N₃",
            "Rinse: DMF 2× (if A), then water 3×",
            "CAUTION: NaN₃ is toxic. Use in fume hood, no metal spatulas.",
        ],
        reagent_cost_per_mmol=5.0,
        commercial_source="Sigma-Aldrich (NaN₃), BroadPharm (azido-PEG-amine)",
        catalog_example="S2002 (Sigma, NaN₃); BP-20419 (BroadPharm)",
        literature_ref="Sun et al., Bioconjugate Chem. 2006, 17, 52",
        notes="Option B (azido-PEG-amine) is safer and gives a PEG spacer.",
    )


def _handle_for_aldehyde_surface() -> HandleInstallation:
    """Oxime ligation on aldehyde surface (cellulose-specific)."""
    return HandleInstallation(
        click_chemistry=ClickChemistry.OXIME_LIGATION,
        handle_installed="-CHO (the surface aldehyde itself)",
        complementary_group="aminooxy (-ONH₂) on capture element",
        reagent="None — surface aldehyde reacts directly with aminooxy-capture-element",
        reagent_concentration="N/A",
        solvent="acetate buffer pH 4.5 (oxime ligation optimal)",
        temperature_c=25.0,
        time_hours=0.0,
        steps=["No handle installation needed — surface -CHO reacts directly with "
               "aminooxy (-ONH₂) on capture element at pH 4.5, RT. "
               "Optional: add 10 mM aniline catalyst for faster ligation."],
        reagent_cost_per_mmol=0.0,
        commercial_source="N/A",
        catalog_example="N/A",
        literature_ref="Dirksen & Dawson, Bioconjugate Chem. 2008, 19, 2543",
        notes="Oxime bond is stable at physiological pH. Reversible only at pH <3.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Routing logic
# ═══════════════════════════════════════════════════════════════════════════

# Substrate → compatible activation methods
_ACTIVATION_ROUTES: dict[SubstrateType, list[tuple[ActivationMethod, str]]] = {
    SubstrateType.SILICA_BEADS: [
        (ActivationMethod.SILANIZATION_APTES, "aptes"),
        (ActivationMethod.SILANIZATION_GPTMS, "gptms"),
        (ActivationMethod.SILANIZATION_MPTMS, "mptms"),
    ],
    SubstrateType.GLASS_SLIDE: [
        (ActivationMethod.SILANIZATION_APTES, "aptes"),
        (ActivationMethod.SILANIZATION_GPTMS, "gptms"),
        (ActivationMethod.SILANIZATION_MPTMS, "mptms"),
    ],
    SubstrateType.GLASS_BEADS: [
        (ActivationMethod.SILANIZATION_APTES, "aptes"),
        (ActivationMethod.SILANIZATION_GPTMS, "gptms"),
    ],
    SubstrateType.PE_NETTING: [
        (ActivationMethod.PLASMA_OXIDATION, "plasma"),
        (ActivationMethod.UV_OZONE, "uv_ozone"),
    ],
    SubstrateType.PP_NETTING: [
        (ActivationMethod.PLASMA_OXIDATION, "plasma"),
        (ActivationMethod.RADIATION_GRAFTING, "radiation"),
    ],
    SubstrateType.NYLON_NETTING: [
        (ActivationMethod.PARTIAL_HYDROLYSIS, "hydrolysis"),
        (ActivationMethod.PLASMA_OXIDATION, "plasma"),
    ],
    SubstrateType.PVDF_MEMBRANE: [
        (ActivationMethod.DEHYDROFLUORINATION, "dehf"),
    ],
    SubstrateType.PS_BEADS: [
        (ActivationMethod.PLASMA_OXIDATION, "plasma"),
    ],
    SubstrateType.STAINLESS_STEEL_MESH: [
        (ActivationMethod.PHOSPHONIC_ACID, "phosphonic"),
        (ActivationMethod.CATECHOL_ANCHOR, "catechol"),
    ],
    SubstrateType.CARBON_FIBER_MESH: [
        (ActivationMethod.PLASMA_OXIDATION, "plasma"),
        (ActivationMethod.CATECHOL_ANCHOR, "catechol"),
    ],
    SubstrateType.CELLULOSE_FILTER: [
        (ActivationMethod.PERIODATE_OXIDATION, "periodate"),
    ],
}


def _get_activation(substrate_type: SubstrateType, preferred_method: Optional[str] = None) -> ActivationProtocol:
    """Get activation protocol for a substrate."""
    routes = _ACTIVATION_ROUTES.get(substrate_type, [])
    if not routes:
        raise ValueError(f"No activation route for {substrate_type.value}")

    # Use preferred method if specified
    if preferred_method:
        for method, key in routes:
            if key == preferred_method or method.value == preferred_method:
                return _dispatch_activation(method, key)

    # Default: first route
    method, key = routes[0]
    return _dispatch_activation(method, key)


def _dispatch_activation(method: ActivationMethod, key: str) -> ActivationProtocol:
    """Dispatch to correct activation protocol builder."""
    if method in (ActivationMethod.SILANIZATION_APTES, ActivationMethod.SILANIZATION_GPTMS,
                  ActivationMethod.SILANIZATION_MPTMS):
        return _silica_glass_activation(key)
    elif method in (ActivationMethod.PLASMA_OXIDATION, ActivationMethod.UV_OZONE,
                    ActivationMethod.RADIATION_GRAFTING):
        return _polyolefin_activation()
    elif method == ActivationMethod.PARTIAL_HYDROLYSIS:
        return _nylon_activation()
    elif method == ActivationMethod.DEHYDROFLUORINATION:
        return _pvdf_activation()
    elif method == ActivationMethod.PHOSPHONIC_ACID:
        return _metal_mesh_activation("phosphonic")
    elif method == ActivationMethod.CATECHOL_ANCHOR:
        return _metal_mesh_activation("catechol")
    elif method == ActivationMethod.PERIODATE_OXIDATION:
        return _cellulose_activation()
    else:
        raise ValueError(f"No activation protocol for {method.value}")


def _get_handle(resulting_group: str, click: ClickChemistry) -> HandleInstallation:
    """Get handle installation protocol based on activated surface and desired click."""
    if "-NH₂" in resulting_group or "amine" in resulting_group.lower():
        return _handle_for_amine_surface(click)
    elif "epoxide" in resulting_group.lower():
        return _handle_for_epoxide_surface()
    elif "-SH" in resulting_group or "thiol" in resulting_group.lower():
        return _handle_for_thiol_surface()
    elif "-CHO" in resulting_group or "aldehyde" in resulting_group.lower():
        return _handle_for_aldehyde_surface()
    elif "-COOH" in resulting_group or "NHS" in resulting_group:
        # Plasma-activated polyolefin → already has NHS ester after EDC/NHS step
        return _handle_for_amine_surface(click)  # NHS reacts with amine on element
    elif "catechol" in resulting_group.lower() or "phosphonic" in resulting_group.lower():
        # Metal anchors already have -NH₂ or -N₃ at distal end
        return _handle_for_amine_surface(click)
    elif "C=C" in resulting_group:
        # PVDF — thiol-ene already installs azide in activation step
        return HandleInstallation(
            click_chemistry=ClickChemistry.SPAAC,
            handle_installed="-N₃ (from thiol-ene with azido-thiol in activation)",
            complementary_group="DBCO on capture element",
            reagent="Already installed during PVDF activation step",
            reagent_concentration="N/A",
            solvent="N/A",
            temperature_c=25.0,
            time_hours=0.0,
            steps=["Azide was installed during thiol-ene step of PVDF activation"],
            reagent_cost_per_mmol=0.0,
            commercial_source="N/A",
            catalog_example="N/A",
            literature_ref="See PVDF activation protocol",
        )
    else:
        raise ValueError(f"No handle protocol for surface group: {resulting_group}")


# ═══════════════════════════════════════════════════════════════════════════
# Coupling protocol generator
# ═══════════════════════════════════════════════════════════════════════════

def _coupling_protocol(click: ClickChemistry, element_type: CaptureElementType,
                       handle_on_surface: str) -> CouplingProtocol:
    """Generate coupling protocol for attaching capture element to click-ready surface."""

    # Determine what handle the capture element needs
    if "N₃" in handle_on_surface or "azide" in handle_on_surface.lower():
        element_handle = "DBCO" if click == ClickChemistry.SPAAC else "terminal alkyne"
    elif "DBCO" in handle_on_surface:
        element_handle = "azide (-N₃)"
    elif "SH" in handle_on_surface or "thiol" in handle_on_surface.lower():
        element_handle = "maleimide"
    elif "NH₂" in handle_on_surface or "amine" in handle_on_surface.lower():
        element_handle = "NHS ester"
    elif "CHO" in handle_on_surface or "aldehyde" in handle_on_surface.lower():
        element_handle = "aminooxy (-ONH₂)"
    else:
        element_handle = "complementary click group"

    # Element-specific protocols
    if element_type in (CaptureElementType.MOLECULAR_BINDER, CaptureElementType.ENZYME_MIMIC):
        return CouplingProtocol(
            capture_element_type=element_type,
            capture_element_handle=element_handle,
            concentration="10-100 µM in PBS pH 7.4",
            solvent="PBS pH 7.4",
            temperature_c=25.0,
            time_hours=4.0,
            steps=[
                f"Dissolve {element_handle}-functionalized capture element at 10-100 µM in PBS",
                "Incubate click-ready substrate in solution, RT, 4 h (gentle agitation)",
                "Rinse: PBS 3×, DI water 2×",
                "  Covalent bond formed — element permanently tethered",
            ],
            coupling_efficiency_pct=60.0,
            surface_density="~10¹³ molecules/cm² (monolayer)",
            literature_ref="Jewett & Bertozzi, Chem. Soc. Rev. 2010, 39, 1272 (SPAAC review)",
        )

    elif element_type in (CaptureElementType.MOF_PARTICLE, CaptureElementType.METAL_OXIDE_NP,
                          CaptureElementType.PHOTOCATALYST, CaptureElementType.SULFIDE_NP,
                          CaptureElementType.ZVI_NP):
        return CouplingProtocol(
            capture_element_type=element_type,
            capture_element_handle=element_handle,
            concentration="1-10 mg/mL particle suspension in PBS pH 7.4",
            solvent="PBS pH 7.4",
            temperature_c=25.0,
            time_hours=12.0,
            steps=[
                f"Functionalize particle surface with {element_handle} handle:",
                "  For MOF: post-synthetic modification of surface carboxylates/amines",
                "  For metal oxide NP: silanization (APTES or MPTMS) then handle installation",
                "  For ZVI NP: catechol-PEG-handle coating",
                "Suspend functionalized particles at 1-10 mg/mL in PBS",
                "Incubate click-ready substrate in suspension, RT, 12 h (gentle rocking)",
                "Rinse: PBS 3×, DI water 2× (to remove unbound particles)",
                "  Particles covalently tethered to substrate via click bond",
            ],
            coupling_efficiency_pct=30.0,
            surface_density="~10⁸-10¹⁰ particles/cm² (depends on particle size)",
            literature_ref="Biju, Chem. Soc. Rev. 2014, 43, 744 (NP functionalization review)",
            notes="Particle functionalization is a pre-step. Surface -COOH or -NH₂ on particles "
                  "are converted to click handles using same NHS-PEG-click reagents.",
        )

    elif element_type == CaptureElementType.DNA_ORIGAMI_CAGE:
        return CouplingProtocol(
            capture_element_type=element_type,
            capture_element_handle=element_handle,
            concentration="1-10 nM cage in folding buffer (TAE/Mg²⁺)",
            solvent="1× TAE + 12.5 mM MgCl₂",
            temperature_c=25.0,
            time_hours=6.0,
            steps=[
                f"Design 3-5 exterior staple overhangs with 5'-{element_handle} modification",
                "  All on ONE face of cage (oriented attachment)",
                "  Order modified staples from IDT (~$50-100 per staple)",
                "Fold cage with standard ATHENA protocol (include modified staples)",
                "Purify: agarose gel or PEG precipitation",
                "Incubate click-ready substrate in 1-10 nM cage solution, RT, 6 h",
                "Rinse: folding buffer 3× (gentle — do not shear cages)",
                "  Cages covalently tethered with interior pores facing solution",
            ],
            coupling_efficiency_pct=40.0,
            surface_density="~10⁹-10¹⁰ cages/cm² (hexagonal packing limit ~10¹¹)",
            literature_ref="Gopinath et al., Nature 2016, 535, 401 (DNA origami placement)",
            notes="Oriented attachment ensures interior capture modules face outward. "
                  "Modified staples carry the click handle on the exterior face only.",
        )

    elif element_type in (CaptureElementType.COORDINATION_CAGE, CaptureElementType.POLYMER_CAGE):
        return CouplingProtocol(
            capture_element_type=element_type,
            capture_element_handle=element_handle,
            concentration="0.1-1 mM cage in DMSO/water or water",
            solvent="DMSO/water 1:9 or water (cage-dependent)",
            temperature_c=25.0,
            time_hours=8.0,
            steps=[
                f"Synthesize cage with one exohedral ligand bearing {element_handle}",
                "  E.g., for Fujita Pd₆L₄: one ligand carries -PEG-azide at para position",
                "Dissolve cage at 0.1-1 mM",
                "Incubate click-ready substrate, RT, 8 h",
                "Rinse: solvent 3×, DI water 2×",
            ],
            coupling_efficiency_pct=50.0,
            surface_density="~10¹²-10¹³ cages/cm² (molecular monolayer)",
            literature_ref="Fujita et al., Chem. Soc. Rev. 2009, 38, 1753",
            notes="Exohedral handle must not disrupt self-assembly. "
                  "Para-substitution on one ligand typically tolerated.",
        )

    else:
        return CouplingProtocol(
            capture_element_type=element_type,
            capture_element_handle=element_handle,
            concentration="Variable",
            solvent="PBS pH 7.4",
            temperature_c=25.0,
            time_hours=4.0,
            steps=[f"Couple {element_handle}-functionalized element to click-ready substrate"],
            coupling_efficiency_pct=50.0,
            surface_density="Varies by element",
            literature_ref="",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def get_substrate(substrate_type: SubstrateType) -> SubstrateProperties:
    """Look up substrate properties."""
    if substrate_type not in _SUBSTRATES:
        raise ValueError(f"Unknown substrate: {substrate_type.value}")
    return _SUBSTRATES[substrate_type]


def list_substrates() -> list[SubstrateProperties]:
    """Return all known substrates."""
    return list(_SUBSTRATES.values())


def compatible_click_chemistries(substrate_type: SubstrateType,
                                 cu_tolerant: bool = True) -> list[ClickChemistry]:
    """List click chemistries compatible with a substrate + Cu tolerance constraint."""
    # All substrates support SPAAC and thiol-maleimide
    result = [ClickChemistry.SPAAC, ClickChemistry.THIOL_MALEIMIDE]
    if cu_tolerant:
        result.append(ClickChemistry.CUAAC)

    # Cellulose specifically supports oxime ligation
    if substrate_type == SubstrateType.CELLULOSE_FILTER:
        result.append(ClickChemistry.OXIME_LIGATION)

    # All amine-surface substrates support direct NHS-amine
    routes = _ACTIVATION_ROUTES.get(substrate_type, [])
    for method, _ in routes:
        if method in (ActivationMethod.SILANIZATION_APTES, ActivationMethod.PARTIAL_HYDROLYSIS,
                      ActivationMethod.PHOSPHONIC_ACID, ActivationMethod.CATECHOL_ANCHOR):
            if ClickChemistry.NHS_AMINE not in result:
                result.append(ClickChemistry.NHS_AMINE)
            break

    return result


def generate_tether_protocol(
    substrate_type: SubstrateType,
    capture_element_type: CaptureElementType,
    cu_tolerant: bool = True,
    preferred_click: Optional[ClickChemistry] = None,
    preferred_activation: Optional[str] = None,
) -> TetherProtocol:
    """Generate a complete three-layer tethering protocol.

    Args:
        substrate_type: What macroscopic support to use
        capture_element_type: What capture element to tether
        cu_tolerant: Whether the capture element tolerates Cu exposure
        preferred_click: Specific click chemistry (or auto-select)
        preferred_activation: Specific activation method (or auto-select)

    Returns:
        Complete TetherProtocol with activation, handle, and coupling steps.
    """
    substrate = get_substrate(substrate_type)

    # Select click chemistry
    if preferred_click:
        click = preferred_click
    elif not cu_tolerant:
        click = ClickChemistry.SPAAC
    else:
        click = ClickChemistry.SPAAC  # default even when Cu OK — safer

    # Validate Cu tolerance
    warnings = []
    if click.requires_copper and not cu_tolerant:
        warnings.append("WARNING: CuAAC selected but capture element is Cu-sensitive. "
                        "Switching to SPAAC.")
        click = ClickChemistry.SPAAC

    # Get activation protocol
    activation = _get_activation(substrate_type, preferred_activation)

    # Special handling: PVDF activation includes handle installation
    # Cellulose periodate → oxime is also a combined step
    if substrate_type == SubstrateType.PVDF_MEMBRANE:
        click = ClickChemistry.SPAAC  # PVDF activation produces azide directly
    if substrate_type == SubstrateType.CELLULOSE_FILTER:
        click = ClickChemistry.OXIME_LIGATION  # cellulose aldehyde → oxime

    # Get handle installation
    handle = _get_handle(activation.resulting_group, click)

    # Get coupling protocol
    coupling = _coupling_protocol(click, capture_element_type, handle.handle_installed)

    # Compute totals
    total_steps = (len(activation.steps) + len(handle.steps) + len(coupling.steps))
    total_time = activation.time_hours + handle.time_hours + coupling.time_hours

    return TetherProtocol(
        substrate=substrate,
        capture_element_type=capture_element_type,
        click_chemistry=click,
        activation=activation,
        handle=handle,
        coupling=coupling,
        cu_safe=not click.requires_copper,
        total_steps=total_steps,
        estimated_time_hours=total_time,
        warnings=warnings,
    )


def generate_all_protocols(
    substrate_type: SubstrateType,
    capture_element_type: CaptureElementType,
    cu_tolerant: bool = True,
) -> list[TetherProtocol]:
    """Generate ALL viable tethering protocols for a substrate + element pair.

    Returns multiple protocols (different activation methods × click chemistries)
    so the user can choose based on available equipment and reagents.
    """
    protocols = []
    routes = _ACTIVATION_ROUTES.get(substrate_type, [])

    for method, key in routes:
        for click in compatible_click_chemistries(substrate_type, cu_tolerant):
            try:
                p = generate_tether_protocol(
                    substrate_type=substrate_type,
                    capture_element_type=capture_element_type,
                    cu_tolerant=cu_tolerant,
                    preferred_click=click,
                    preferred_activation=key,
                )
                protocols.append(p)
            except (ValueError, KeyError):
                continue  # skip incompatible combinations

    # Deduplicate by (activation method, click chemistry)
    seen = set()
    unique = []
    for p in protocols:
        key = (p.activation.method.value, p.click_chemistry.value)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def protocol_report(protocol: TetherProtocol) -> str:
    """Human-readable full protocol report."""
    p = protocol
    lines = [
        f"{'=' * 70}",
        f"TETHERING PROTOCOL",
        f"{'=' * 70}",
        f"Substrate: {p.substrate.name}",
        f"Capture element: {p.capture_element_type.value}",
        f"Click chemistry: {p.click_chemistry.value} ({'Cu-free' if p.cu_safe else 'USES Cu CATALYST'})",
        f"Total steps: {p.total_steps}",
        f"Estimated time: {p.estimated_time_hours:.1f} hours",
        "",
    ]

    if p.warnings:
        for w in p.warnings:
            lines.append(f"  !! {w}")
        lines.append("")

    lines.append(f"--- LAYER A: SURFACE ACTIVATION ---")
    lines.append(f"Method: {p.activation.method.value}")
    lines.append(f"Result: {p.activation.resulting_group}")
    lines.append(f"Site density: {p.activation.site_density_per_nm2:.1f} sites/nm²")
    lines.append(f"Reagents: {', '.join(p.activation.reagents)}")
    lines.append(f"Conditions: {p.activation.solvent}, {p.activation.temperature_c}°C, {p.activation.time_hours} h")
    for i, step in enumerate(p.activation.steps, 1):
        lines.append(f"  {i}. {step}")
    lines.append(f"Ref: {p.activation.literature_ref}")
    lines.append("")

    lines.append(f"--- LAYER B: HANDLE INSTALLATION ---")
    lines.append(f"Handle: {p.handle.handle_installed}")
    lines.append(f"Complementary group needed on element: {p.handle.complementary_group}")
    if p.handle.reagent != "None" and "N/A" not in p.handle.reagent:
        lines.append(f"Reagent: {p.handle.reagent} ({p.handle.reagent_concentration})")
        lines.append(f"Source: {p.handle.commercial_source} ({p.handle.catalog_example})")
    for i, step in enumerate(p.handle.steps, 1):
        lines.append(f"  {i}. {step}")
    lines.append(f"Ref: {p.handle.literature_ref}")
    lines.append("")

    lines.append(f"--- LAYER C: ELEMENT COUPLING ---")
    lines.append(f"Element handle: {p.coupling.capture_element_handle}")
    lines.append(f"Concentration: {p.coupling.concentration}")
    lines.append(f"Conditions: {p.coupling.solvent}, {p.coupling.temperature_c}°C, {p.coupling.time_hours} h")
    for i, step in enumerate(p.coupling.steps, 1):
        lines.append(f"  {i}. {step}")
    lines.append(f"Coupling efficiency: ~{p.coupling.coupling_efficiency_pct:.0f}%")
    lines.append(f"Surface density: {p.coupling.surface_density}")
    lines.append(f"Ref: {p.coupling.literature_ref}")

    return "\n".join(lines)
