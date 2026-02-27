"""
Functionalized Lignin Adapter — Class D (Bulk Sorbent).

Takes an InteractionGeometrySpec and designs a functionalized lignin sorbent.
The spec drives functional group selection through the same HSAB and donor
subtype logic used by precision binders.

Design logic:
    1. Spec donor types → functional group selection (HSAB-matched)
    2. Spec cavity size → target loading density (groups/g)
    3. Application context → backbone grade + crosslinker selection
    4. Physics predicts capacity from: group_density × site_Ka × accessibility
    5. FabSpec includes functionalization chemistry, not just "use lignin"

Physics connection:
    - Same HSAB donor routing as crown ether adapter
    - Same BackSolve donor subtypes (O_carboxylate, S_thiolate, N_amine, etc.)
    - Capacity = f(functional_group_density, per_site_Ka, accessibility)
    - Selectivity from donor preference ratios

Data sources:
    - Guo X et al. (Chem. Eng. J. 2020) — lignin-based heavy metal sorbents
    - Wang J, Chen C (Biotechnol. Adv. 2009) — biosorbent review
    - Ge Y et al. (J. Environ. Chem. Eng. 2016) — modified lignin for Pb removal
    - Suhas et al. (Bioresour. Technol. 2007) — lignin as sorbent
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.adapters.base import RealizationAdapter, ValidationReport
from mabe.realization.models import (
    ApplicationContext,
    CavityDimensions,
    DeviationReport,
    FabricationSpec,
    InteractionGeometrySpec,
    RealizationScore,
    Solvent,
)
from mabe.realization.registry.material_registry import MaterialCapability


# ─────────────────────────────────────────────
# Functional group library (physics-driven)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class FunctionalGroup:
    """A graftable functional group for lignin functionalization."""

    name: str
    abbreviation: str
    donor_atom: str                    # "O", "N", "S"
    donor_subtype: str                 # BackSolve subtype
    hsab_affinity: str                 # "hard", "borderline", "soft"

    # ── Performance ──
    typical_loading_mmol_per_g: float  # achievable grafting density
    typical_qmax_mg_per_g: float       # for Pb²⁺ as reference
    typical_logK_site: float           # per-site binding constant
    selectivity_metals: list[str]      # metals this group prefers

    # ── Chemistry ──
    grafting_reagent: str
    grafting_conditions: str
    grafting_yield: float              # 0–1

    # ── Cost ──
    reagent_cost_per_kg_usd: float
    process_cost_per_kg_sorbent_usd: float


FUNCTIONAL_GROUPS: dict[str, FunctionalGroup] = {
    "carboxylate": FunctionalGroup(
        name="Carboxylate (–COOH)",
        abbreviation="COOH",
        donor_atom="O",
        donor_subtype="O_carboxylate",
        hsab_affinity="hard",
        typical_loading_mmol_per_g=3.5,
        typical_qmax_mg_per_g=45.0,
        typical_logK_site=3.2,
        selectivity_metals=["Ca2+", "Mg2+", "La3+", "UO2_2+", "Cu2+"],
        grafting_reagent="chloroacetic acid (ClCH2COOH)",
        grafting_conditions="NaOH (40%), 60°C, 4h",
        grafting_yield=0.75,
        reagent_cost_per_kg_usd=2.00,
        process_cost_per_kg_sorbent_usd=5.00,
    ),
    "phosphonate": FunctionalGroup(
        name="Phosphonate (–PO3H2)",
        abbreviation="PO3H2",
        donor_atom="O",
        donor_subtype="O_phosphonate",
        hsab_affinity="hard",
        typical_loading_mmol_per_g=2.0,
        typical_qmax_mg_per_g=55.0,
        typical_logK_site=4.0,
        selectivity_metals=["UO2_2+", "La3+", "Ca2+", "Fe3+"],
        grafting_reagent="phosphorous acid + formaldehyde (Mannich)",
        grafting_conditions="85°C, 6h, aqueous",
        grafting_yield=0.60,
        reagent_cost_per_kg_usd=8.00,
        process_cost_per_kg_sorbent_usd=15.00,
    ),
    "amine": FunctionalGroup(
        name="Amine (–NH2)",
        abbreviation="NH2",
        donor_atom="N",
        donor_subtype="N_amine",
        hsab_affinity="borderline",
        typical_loading_mmol_per_g=4.0,
        typical_qmax_mg_per_g=65.0,
        typical_logK_site=3.8,
        selectivity_metals=["Cu2+", "Ni2+", "Zn2+", "Co2+", "Pb2+"],
        grafting_reagent="ethylenediamine (EDA)",
        grafting_conditions="Mannich reaction: HCHO + EDA, 70°C, 8h",
        grafting_yield=0.70,
        reagent_cost_per_kg_usd=3.00,
        process_cost_per_kg_sorbent_usd=8.00,
    ),
    "iminodiacetate": FunctionalGroup(
        name="Iminodiacetate (–N(CH2COOH)2)",
        abbreviation="IDA",
        donor_atom="N",
        donor_subtype="N_amine",
        hsab_affinity="borderline",
        typical_loading_mmol_per_g=2.5,
        typical_qmax_mg_per_g=80.0,
        typical_logK_site=5.0,
        selectivity_metals=["Cu2+", "Ni2+", "Zn2+", "Co2+", "Pb2+"],
        grafting_reagent="chloroacetic acid + ammonia",
        grafting_conditions="Two-step: amination then carboxymethylation",
        grafting_yield=0.55,
        reagent_cost_per_kg_usd=5.00,
        process_cost_per_kg_sorbent_usd=12.00,
    ),
    "dithiocarbamate": FunctionalGroup(
        name="Dithiocarbamate (–NCS2⁻)",
        abbreviation="DTC",
        donor_atom="S",
        donor_subtype="S_thiolate",
        hsab_affinity="soft",
        typical_loading_mmol_per_g=2.8,
        typical_qmax_mg_per_g=120.0,  # excellent for Pb, Hg, Cd
        typical_logK_site=5.5,
        selectivity_metals=["Pb2+", "Hg2+", "Cd2+", "Ag+", "Cu2+"],
        grafting_reagent="CS2 + amine (on aminated lignin)",
        grafting_conditions="CS2 in NaOH, 25°C, 2h (on pre-aminated lignin)",
        grafting_yield=0.65,
        reagent_cost_per_kg_usd=6.00,
        process_cost_per_kg_sorbent_usd=18.00,
    ),
    "thiol": FunctionalGroup(
        name="Thiol (–SH)",
        abbreviation="SH",
        donor_atom="S",
        donor_subtype="S_thiolate",
        hsab_affinity="soft",
        typical_loading_mmol_per_g=2.0,
        typical_qmax_mg_per_g=95.0,
        typical_logK_site=5.2,
        selectivity_metals=["Hg2+", "Pb2+", "Cd2+", "Ag+"],
        grafting_reagent="thioglycolic acid or 2-mercaptoethanol",
        grafting_conditions="Esterification with phenolic OH, 80°C, 6h",
        grafting_yield=0.50,
        reagent_cost_per_kg_usd=12.00,
        process_cost_per_kg_sorbent_usd=22.00,
    ),
}


# ─────────────────────────────────────────────
# Lignin backbone library
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class LigninBackbone:
    """Lignin source/grade with base properties."""

    name: str
    source: str                        # "kraft", "organosolv", "soda", "lignosulfonate"
    phenolic_OH_mmol_per_g: float
    aliphatic_OH_mmol_per_g: float
    COOH_mmol_per_g: float
    mw_avg: float                      # number-average MW
    cost_per_kg_usd: float
    water_soluble: bool
    suppliers: list[str]


LIGNIN_BACKBONES: dict[str, LigninBackbone] = {
    "kraft": LigninBackbone(
        name="Kraft lignin",
        source="kraft",
        phenolic_OH_mmol_per_g=4.2,
        aliphatic_OH_mmol_per_g=2.8,
        COOH_mmol_per_g=0.6,
        mw_avg=5000,
        cost_per_kg_usd=0.50,
        water_soluble=False,
        suppliers=["Domtar", "Stora Enso", "West Fraser"],
    ),
    "organosolv": LigninBackbone(
        name="Organosolv lignin",
        source="organosolv",
        phenolic_OH_mmol_per_g=3.8,
        aliphatic_OH_mmol_per_g=3.5,
        COOH_mmol_per_g=0.3,
        mw_avg=3000,
        cost_per_kg_usd=3.00,
        water_soluble=False,
        suppliers=["Suzano", "Lignol"],
    ),
    "lignosulfonate": LigninBackbone(
        name="Lignosulfonate",
        source="lignosulfonate",
        phenolic_OH_mmol_per_g=2.0,
        aliphatic_OH_mmol_per_g=1.5,
        COOH_mmol_per_g=0.8,
        mw_avg=20000,
        cost_per_kg_usd=0.30,
        water_soluble=True,
        suppliers=["Borregaard", "Sappi"],
    ),
}


# ─────────────────────────────────────────────
# Crosslinker library
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Crosslinker:
    """Crosslinker for insolubilizing lignin."""

    name: str
    mechanism: str                     # "Schiff base", "epoxide ring opening", etc.
    pH_stability: tuple[float, float]
    thermal_limit_C: float
    cost_per_kg_usd: float
    typical_wt_pct: float              # typical loading


CROSSLINKERS: dict[str, Crosslinker] = {
    "glutaraldehyde": Crosslinker(
        name="Glutaraldehyde",
        mechanism="Schiff base with amine groups",
        pH_stability=(2.0, 10.0),
        thermal_limit_C=120.0,
        cost_per_kg_usd=5.00,
        typical_wt_pct=15.0,
    ),
    "epichlorohydrin": Crosslinker(
        name="Epichlorohydrin",
        mechanism="Epoxide ring opening with –OH",
        pH_stability=(1.0, 12.0),
        thermal_limit_C=150.0,
        cost_per_kg_usd=3.00,
        typical_wt_pct=20.0,
    ),
    "formaldehyde": Crosslinker(
        name="Formaldehyde",
        mechanism="Methylene bridge with phenolic ring",
        pH_stability=(1.0, 13.0),
        thermal_limit_C=200.0,
        cost_per_kg_usd=1.00,
        typical_wt_pct=10.0,
    ),
}


# ─────────────────────────────────────────────
# HSAB routing: spec donors → functional groups
# ─────────────────────────────────────────────

# Maps cation HSAB class to preferred functional group
HSAB_GROUP_ROUTING: dict[str, list[str]] = {
    "hard": ["carboxylate", "phosphonate"],
    "borderline": ["amine", "iminodiacetate"],
    "soft": ["dithiocarbamate", "thiol"],
}

# Simplified cation HSAB lookup (subset for bulk sorbent targets)
CATION_HSAB: dict[str, str] = {
    "Li+": "hard", "Na+": "hard", "K+": "hard",
    "Mg2+": "hard", "Ca2+": "hard", "Sr2+": "hard", "Ba2+": "hard",
    "Fe3+": "hard", "La3+": "hard", "UO2_2+": "hard",
    "Cu2+": "borderline", "Zn2+": "borderline", "Ni2+": "borderline",
    "Co2+": "borderline", "Mn2+": "borderline",
    "Pb2+": "soft", "Cd2+": "soft", "Hg2+": "soft", "Ag+": "soft",
}


def select_functional_group(
    spec: InteractionGeometrySpec,
) -> tuple[FunctionalGroup, str]:
    """
    Select functional group from spec donor types via HSAB routing.

    Uses same logic as precision adapters: donor atom types in spec
    indicate required HSAB class.
    """
    donor_types = spec.required_donor_types

    # Infer HSAB preference from donor types
    if "S" in donor_types:
        hsab_class = "soft"
    elif "N" in donor_types and "O" not in donor_types:
        hsab_class = "borderline"
    elif "N" in donor_types and "O" in donor_types:
        hsab_class = "borderline"
    elif "O" in donor_types:
        hsab_class = "hard"
    else:
        # Default: check cavity size to guess cation
        hsab_class = _guess_hsab_from_cavity(spec)

    group_names = HSAB_GROUP_ROUTING.get(hsab_class, ["carboxylate"])
    selected = FUNCTIONAL_GROUPS[group_names[0]]

    rationale = (
        f"HSAB routing: {hsab_class} → {selected.abbreviation} "
        f"({selected.donor_subtype}). "
        f"Donor types in spec: {donor_types or 'none (inferred from cavity)'}."
    )

    return selected, rationale


def _guess_hsab_from_cavity(spec: InteractionGeometrySpec) -> str:
    """
    Guess HSAB class from cavity size when donor types aren't specified.

    Small cavity → likely transition metal (borderline).
    Medium cavity → likely divalent main group (hard/soft depends).
    Large cavity → likely alkali/alkaline earth (hard).
    """
    diameter = spec.cavity_dimensions.max_internal_diameter_A
    if diameter < 1.5:
        return "borderline"  # small TM
    elif diameter > 3.0:
        return "hard"  # large alkali
    else:
        return "borderline"  # default


# ─────────────────────────────────────────────
# LigninFabSpec
# ─────────────────────────────────────────────

@dataclass
class LigninFabSpec(FabricationSpec):
    """Fabrication spec for functionalized lignin sorbent."""

    # ── Backbone ──
    backbone: str = ""
    backbone_source: str = ""
    backbone_cost_per_kg: float = 0.0

    # ── Functionalization ──
    functional_group: str = ""
    functional_group_abbreviation: str = ""
    donor_subtype: str = ""
    hsab_class: str = ""
    target_loading_mmol_per_g: float = 0.0
    grafting_reagent: str = ""
    grafting_conditions: str = ""

    # ── Crosslinker ──
    crosslinker: str = ""
    crosslinker_wt_pct: float = 0.0
    pH_stability: tuple[float, float] = (2.0, 10.0)

    # ── Predicted performance ──
    predicted_qmax_mg_per_g: float = 0.0
    predicted_qmax_mmol_per_g: float = 0.0
    predicted_logK_site: float = 0.0
    predicted_selectivity_vs: dict[str, float] = field(default_factory=dict)

    # ── Economics ──
    sorbent_cost_per_kg: float = 0.0
    predicted_cost_per_kg_removed: float = 0.0
    predicted_regeneration_cycles: int = 0

    # ── Process design ──
    recommended_column_config: str = ""
    contact_time_min: float = 0.0
    optimal_pH: float = 0.0


# ─────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────

class FunctionalizedLigninAdapter(RealizationAdapter):
    """
    Designs functionalized lignin sorbents from InteractionGeometrySpec.

    Class D (Bulk Sorbent):
        - Physics drives functional group selection (HSAB routing)
        - Capacity from group_density × site_Ka × accessibility
        - Cost metric is $/kg of target removed
        - Same donor subtype logic as precision adapters
    """

    def __init__(self, capability: Optional[MaterialCapability] = None):
        if capability is None:
            capability = _make_lignin_capability()
        super().__init__(capability)

    def estimate_fidelity(
        self,
        spec: InteractionGeometrySpec,
    ) -> RealizationScore:
        """Quick score for the ranker."""

        func_group, rationale = select_functional_group(spec)

        # Physics fidelity: how well can we approximate the spec geometry?
        # Bulk sorbents have distributed sites — modest fidelity but high capacity
        physics_fidelity = 0.3  # base for any sorbent

        # Bonus if donor types match well
        if spec.required_donor_types:
            donor_match = len(
                spec.required_donor_types & {func_group.donor_atom}
            ) / max(1, len(spec.required_donor_types))
            physics_fidelity += donor_match * 0.2

        # Capacity estimate
        capacity = func_group.typical_loading_mmol_per_g * 0.8  # 80% accessibility

        # Selectivity from group preference
        selectivity = 10.0 ** (func_group.typical_logK_site - 2.0)  # vs generic competitor

        # Cost per kg removed
        sorbent_cost = (
            LIGNIN_BACKBONES["kraft"].cost_per_kg_usd +
            func_group.process_cost_per_kg_sorbent_usd
        )
        qmax_kg = func_group.typical_qmax_mg_per_g / 1000.0  # mg/g → g/g → kg/kg
        regen_cycles = 15
        cost_per_kg = sorbent_cost / (qmax_kg * regen_cycles) if qmax_kg > 0 else 999.0

        advantages = [
            f"Lignin: $0.50/kg renewable backbone",
            f"{func_group.abbreviation} group: {func_group.typical_qmax_mg_per_g:.0f} mg/g capacity",
            "Scalable to tonnes",
        ]
        limitations = [
            "Distributed binding sites — lower selectivity than precision binders",
            f"Physics fidelity {physics_fidelity:.2f} (bulk approximation)",
        ]

        deviation = DeviationReport(
            material_system="functionalized_lignin",
            element_deviations_A=[],
            max_deviation_A=1.0,
            mean_deviation_A=1.0,
        )

        return RealizationScore(
            material_system="functionalized_lignin",
            adapter_id="FunctionalizedLigninAdapter",
            deviation_from_ideal=deviation,
            physics_fidelity=physics_fidelity,
            synthetic_accessibility=0.90,
            cost_score=0.90,
            scalability=0.95,
            operating_condition_compatibility=0.80,
            reusability_score=0.60,
            # Bulk sorbent fields
            capacity_mmol_per_g=capacity,
            selectivity_factor=selectivity,
            throughput_L_per_h_per_kg=50.0,  # typical packed column
            regenerability_cycles=regen_cycles,
            cost_per_kg_processed=cost_per_kg,
            physics_class="bulk_sorbent",
            composite_score=0.0,
            confidence=0.70,
            advantages=advantages,
            limitations=limitations,
            feasible=True,
        )

    def design(
        self,
        spec: InteractionGeometrySpec,
    ) -> LigninFabSpec:
        """Full lignin sorbent design driven by spec physics."""

        spec_hash = hashlib.md5(str(spec).encode()).hexdigest()[:12]

        # ── Step 1: Functional group from HSAB routing ──
        func_group, group_rationale = select_functional_group(spec)

        # ── Step 2: Backbone selection ──
        backbone = LIGNIN_BACKBONES["kraft"]  # cheapest, most available
        if func_group.hsab_affinity == "soft":
            # Need more phenolic OH for thiol/DTC grafting
            if LIGNIN_BACKBONES["kraft"].phenolic_OH_mmol_per_g < 3.0:
                backbone = LIGNIN_BACKBONES["organosolv"]

        # ── Step 3: Crosslinker selection ──
        if func_group.donor_atom == "N":
            crosslinker = CROSSLINKERS["glutaraldehyde"]
        else:
            crosslinker = CROSSLINKERS["epichlorohydrin"]

        # ── Step 4: Capacity prediction ──
        loading = func_group.typical_loading_mmol_per_g * func_group.grafting_yield
        accessibility = 0.80  # 80% of grafted sites accessible
        effective_loading = loading * accessibility
        qmax_mmol = effective_loading
        # Convert to mg/g using Pb²⁺ MW as reference (207.2 g/mol)
        target_mw = 207.2  # default Pb²⁺; spec could override
        qmax_mg = qmax_mmol * target_mw

        # ── Step 5: Cost calculation ──
        sorbent_cost = backbone.cost_per_kg_usd + func_group.process_cost_per_kg_sorbent_usd
        regen_cycles = 15
        qmax_kg_per_kg = qmax_mg / 1e6  # mg/g → kg/kg
        cost_per_kg_removed = (
            sorbent_cost / (qmax_kg_per_kg * regen_cycles)
            if qmax_kg_per_kg > 0 else 999.0
        )

        # ── Step 6: Synthesis steps ──
        steps = [
            f"Procure {backbone.name} (${backbone.cost_per_kg_usd}/kg) from {backbone.suppliers[0]}",
        ]

        # Functionalization may be multi-step
        if func_group.abbreviation == "DTC":
            steps.append(
                f"Step 1 — Amination: Mannich reaction "
                f"(HCHO + EDA, 70°C, 8h) to introduce –NH₂ groups"
            )
            steps.append(
                f"Step 2 — DTC grafting: {func_group.grafting_conditions}"
            )
        else:
            steps.append(
                f"Functionalize: {func_group.grafting_reagent}, "
                f"{func_group.grafting_conditions}"
            )

        steps.append(
            f"Crosslink: {crosslinker.name} at {crosslinker.typical_wt_pct:.0f} wt%, "
            f"{crosslinker.mechanism}"
        )
        steps.append("Wash (water, ethanol), dry (60°C, 12h), grind to 0.5–1.0 mm")
        steps.append(
            "Characterize: FTIR (confirm functional groups), "
            "BET surface area, elemental analysis (N or S content)"
        )

        # ── Step 7: Validation plan ──
        validation = [
            f"Batch adsorption isotherm: 10–500 ppm target in pH-buffered solution",
            "Langmuir/Freundlich fit → qmax, KL",
            "Kinetics: pseudo-second-order, contact time optimization",
            "Column breakthrough: 10 bed volumes, measure Ct/C0",
        ]
        if regen_cycles > 1:
            validation.append(
                f"Regeneration: 0.1M HNO₃ wash, measure capacity over {regen_cycles} cycles"
            )

        # pH for optimal binding
        if func_group.hsab_affinity == "hard":
            optimal_pH = 5.5
        elif func_group.hsab_affinity == "soft":
            optimal_pH = 5.0
        else:
            optimal_pH = 5.5

        expected = {
            "qmax_mg_per_g": qmax_mg,
            "qmax_mmol_per_g": qmax_mmol,
            "logK_site": func_group.typical_logK_site,
            "optimal_pH": optimal_pH,
            "contact_time_min": 60.0,
            "sorbent_cost_per_kg": sorbent_cost,
            "cost_per_kg_removed_usd": cost_per_kg_removed,
        }

        return LigninFabSpec(
            material_system="functionalized_lignin",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=CavityDimensions(
                volume_A3=0, aperture_A=0, depth_A=0, max_internal_diameter_A=0,
            ),
            predicted_deviation_from_ideal_A=1.0,
            synthesis_steps=steps,
            estimated_yield=func_group.grafting_yield,
            estimated_cost_per_unit=sorbent_cost,
            estimated_time="2–3 days",
            validation_experiments=validation,
            expected_observables=expected,
            # Lignin-specific
            backbone=backbone.name,
            backbone_source=backbone.source,
            backbone_cost_per_kg=backbone.cost_per_kg_usd,
            functional_group=func_group.name,
            functional_group_abbreviation=func_group.abbreviation,
            donor_subtype=func_group.donor_subtype,
            hsab_class=func_group.hsab_affinity,
            target_loading_mmol_per_g=loading,
            grafting_reagent=func_group.grafting_reagent,
            grafting_conditions=func_group.grafting_conditions,
            crosslinker=crosslinker.name,
            crosslinker_wt_pct=crosslinker.typical_wt_pct,
            pH_stability=crosslinker.pH_stability,
            predicted_qmax_mg_per_g=qmax_mg,
            predicted_qmax_mmol_per_g=qmax_mmol,
            predicted_logK_site=func_group.typical_logK_site,
            sorbent_cost_per_kg=sorbent_cost,
            predicted_cost_per_kg_removed=cost_per_kg_removed,
            predicted_regeneration_cycles=regen_cycles,
            recommended_column_config="Packed bed, 0.5–1.0 mm particles, EBCT=5 min",
            contact_time_min=60.0,
            optimal_pH=optimal_pH,
        )

    def validate_design(self, fab: FabricationSpec) -> ValidationReport:
        if not isinstance(fab, LigninFabSpec):
            return ValidationReport(
                valid=False, issues=["Not a LigninFabSpec"], warnings=[],
            )

        issues = []
        warnings = []

        if fab.target_loading_mmol_per_g < 0.5:
            issues.append("Loading too low — insufficient capacity")

        if fab.predicted_qmax_mg_per_g < 10:
            warnings.append(f"Low capacity ({fab.predicted_qmax_mg_per_g:.0f} mg/g)")

        if fab.predicted_cost_per_kg_removed > 100:
            warnings.append(
                f"High cost (${fab.predicted_cost_per_kg_removed:.0f}/kg removed)"
            )

        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            confidence=0.70,
        )


def _make_lignin_capability() -> MaterialCapability:
    return MaterialCapability(
        system_id="functionalized_lignin",
        physics_class="bulk_sorbent",
        adapter_class="FunctionalizedLigninAdapter",
        min_pocket_size_nm=0.0,
        max_pocket_size_nm=0.0,  # not a cavity system
        achievable_symmetries=["none"],
        max_donor_count=1000,  # distributed sites
        donor_types_available=["O", "N", "S"],
        positioning_precision_A=5.0,  # bulk average
        rigidity_range=("flexible", "semi-rigid"),
        pH_stability=(1.0, 12.0),
        thermal_stability_K=(273.0, 470.0),
        solvent_compatibility=["aqueous"],
        min_practical_scale="g",
        max_practical_scale="tonne",
        cost_per_unit_range=(0.50, 25.0),
        typical_synthesis_time="2–3 days",
        literature_validation_rate=0.75,
        literature_examples=500,
        design_tools_available=[],
        known_strengths=[
            "Cheapest backbone ($0.50/kg kraft lignin)",
            "Renewable (paper industry byproduct)",
            "HSAB-tunable functionalization (same physics as precision)",
            "Scalable to tonnes",
            "High capacity (50–120 mg/g for heavy metals)",
        ],
        known_limitations=[
            "Distributed sites — lower selectivity than defined cavities",
            "Batch-to-batch variability in lignin properties",
            "Capacity degrades over regeneration cycles",
        ],
    )
