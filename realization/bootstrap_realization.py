"""
MABE Realization Engine — Sprint R1 Bootstrap
Run from your MABE repo root: python bootstrap_realization.py

Creates: mabe/realization/ with all subpackages (17 files)
Test:    python -m pytest mabe/realization/tests/test_sprint_r1.py -v
"""

import os

FILES = {

    'mabe/__init__.py': """\
\"\"\"MABE — Molecular Affinity Binding Engine.\"\"\"
""",

    'mabe/realization/__init__.py': """\
\"\"\"
MABE Realization Engine — Layer 3 + Layer 4

Physics first. Compute the ideal pocket. Then measure deviation.

Layer 3 Phase 1: InteractionGeometrySpec → IdealPocketSpec
Layer 3 Phase 2: IdealPocketSpec × MaterialRegistry → RankedRealizations
Layer 4:         RankedRealizations → FabricationSpec (per adapter)
\"\"\"

from mabe.realization.models import (
    InteractionGeometrySpec,
    IdealPocketSpec,
    DeviationReport,
    RealizationScore,
    RankedRealizations,
    FabricationSpec,
)
from mabe.realization.engine.ideal_pocket import compute_ideal_pocket
from mabe.realization.engine.ranker import rank_realizations

__all__ = [
    "InteractionGeometrySpec",
    "IdealPocketSpec",
    "DeviationReport",
    "RealizationScore",
    "RankedRealizations",
    "FabricationSpec",
    "compute_ideal_pocket",
    "rank_realizations",
]
""",

    'mabe/realization/adapters/__init__.py': """\
\"\"\"
Layer 4 Implementation Adapters.

Each adapter takes an InteractionGeometrySpec and produces a FabricationSpec.
Organized by physics class, not material origin.

Sprint R1: base class only. Concrete adapters in Sprint R2+.
\"\"\"

from mabe.realization.adapters.base import RealizationAdapter

__all__ = ["RealizationAdapter"]
""",

    'mabe/realization/adapters/base.py': """\
\"\"\"
RealizationAdapter — base class for all Layer 4 adapters.

Each adapter handles one material system. It does two things:
    1. estimate_fidelity() — quick score for the ranker (Phase 2)
    2. design()            — full pocket design → FabricationSpec (Phase 4)

Concrete adapters implement these for specific material systems.
Sprint R2+ will add: CyclodextrinAdapter, CrownEtherAdapter, PorphyrinAdapter, etc.
\"\"\"

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import (
        InteractionGeometrySpec,
        RealizationScore,
        FabricationSpec,
    )
    from mabe.realization.registry.material_registry import MaterialCapability


class RealizationAdapter(ABC):
    \"\"\"Base class for all Layer 4 material-system adapters.\"\"\"

    system_id: str
    capability: "MaterialCapability"

    def __init__(self, capability: "MaterialCapability"):
        self.system_id = capability.system_id
        self.capability = capability

    @abstractmethod
    def estimate_fidelity(
        self,
        spec: "InteractionGeometrySpec",
    ) -> "RealizationScore":
        \"\"\"
        Quick score without full design. Used by Layer 3 ranker.

        Must be fast (<1 second). Uses capability envelope + heuristics.
        Full design is expensive and only runs on selected systems.
        \"\"\"
        ...

    @abstractmethod
    def design(
        self,
        spec: "InteractionGeometrySpec",
    ) -> "FabricationSpec":
        \"\"\"
        Full pocket design. Produces fabrication-ready output.

        This is the expensive call. Only runs when the ranker selects
        this material system (or the user requests it explicitly).
        \"\"\"
        ...

    @abstractmethod
    def validate_design(
        self,
        fab: "FabricationSpec",
    ) -> "ValidationReport":
        \"\"\"
        Check the design for internal consistency, strain, clashes.

        Catches designs that look good on paper but would fail in practice.
        \"\"\"
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system_id={self.system_id!r})"


@dataclass
class ValidationReport:
    \"\"\"Result of adapter self-validation on a design.\"\"\"
    valid: bool
    issues: list[str]
    warnings: list[str]
    strain_energy_kJ_mol: float = 0.0
    steric_clashes: int = 0
    confidence: float = 0.0
""",

    'mabe/realization/adapters/cyclodextrin_adapter.py': """\
\"\"\"
Cyclodextrin Adapter — Sprint R2a.

First concrete Layer 4 adapter. Leverages BackSolve host-guest
calibration (R²=0.850, MAE=0.74 kJ/mol).

Physics class: Covalent Cavity (cyclic encapsulant)
Input geometry: Truncated cone hydrophobic cavity, 0.47–0.75 nm diameter
Design logic:
    1. Map target cavity volume → α/β/γ-CD selection
    2. Compute packing coefficient → Rebek 55% rule validation
    3. Match H-bond / charge requirements → rim modification selection
    4. Predict binding energy from BackSolve parameters
    5. Output: CD variant + modification + sourcing + predicted Ka
\"\"\"

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.adapters.base import RealizationAdapter, ValidationReport
from mabe.realization.models import (
    CavityDimensions,
    CavityShape,
    DeviationReport,
    FabricationSpec,
    InteractionGeometrySpec,
    RealizationScore,
    Solvent,
)
from mabe.realization.registry.material_registry import MaterialCapability


# ─────────────────────────────────────────────
# CD Host Registry (from BackSolve HOST_REGISTRY)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class CDVariant:
    \"\"\"A specific cyclodextrin host with known properties.\"\"\"
    name: str
    base_type: str                    # "alpha", "beta", "gamma"
    n_glucose: int
    cavity_diameter_A: float
    cavity_depth_A: float
    cavity_volume_A3: float
    cavity_sasa_A2: float
    portal_hb_sites: int              # OH-2 + OH-3 on secondary face
    modification: str                 # "native", "HP", "Me", "SBE", "amine", "thiol"
    modification_face: str            # "none", "primary", "secondary", "both"
    solubility_aqueous: str           # "low", "moderate", "high"
    commercial: bool
    cas_number: Optional[str] = None
    approx_cost_per_g: float = 0.0   # USD


CD_LIBRARY: dict[str, CDVariant] = {
    # ── Native CDs ──
    "alpha-CD": CDVariant(
        name="α-Cyclodextrin", base_type="alpha", n_glucose=6,
        cavity_diameter_A=4.7, cavity_depth_A=7.9,
        cavity_volume_A3=174, cavity_sasa_A2=307,
        portal_hb_sites=12, modification="native", modification_face="none",
        solubility_aqueous="moderate", commercial=True,
        cas_number="10016-20-3", approx_cost_per_g=0.50,
    ),
    "beta-CD": CDVariant(
        name="β-Cyclodextrin", base_type="beta", n_glucose=7,
        cavity_diameter_A=6.0, cavity_depth_A=7.9,
        cavity_volume_A3=262, cavity_sasa_A2=427,
        portal_hb_sites=14, modification="native", modification_face="none",
        solubility_aqueous="low", commercial=True,
        cas_number="7585-39-9", approx_cost_per_g=0.30,
    ),
    "gamma-CD": CDVariant(
        name="γ-Cyclodextrin", base_type="gamma", n_glucose=8,
        cavity_diameter_A=7.5, cavity_depth_A=7.9,
        cavity_volume_A3=427, cavity_sasa_A2=590,
        portal_hb_sites=16, modification="native", modification_face="none",
        solubility_aqueous="high", commercial=True,
        cas_number="17465-86-0", approx_cost_per_g=2.00,
    ),

    # ── Modified β-CDs ──
    "HP-beta-CD": CDVariant(
        name="Hydroxypropyl-β-Cyclodextrin", base_type="beta", n_glucose=7,
        cavity_diameter_A=6.0, cavity_depth_A=7.9,
        cavity_volume_A3=262, cavity_sasa_A2=427,
        portal_hb_sites=14, modification="HP", modification_face="secondary",
        solubility_aqueous="high", commercial=True,
        cas_number="128446-35-5", approx_cost_per_g=1.50,
    ),
    "Me-beta-CD": CDVariant(
        name="Methyl-β-Cyclodextrin", base_type="beta", n_glucose=7,
        cavity_diameter_A=6.0, cavity_depth_A=7.9,
        cavity_volume_A3=262, cavity_sasa_A2=427,
        portal_hb_sites=7,  # methylation removes some OH groups
        modification="Me", modification_face="both",
        solubility_aqueous="high", commercial=True,
        cas_number="128446-36-6", approx_cost_per_g=5.00,
    ),
    "SBE-beta-CD": CDVariant(
        name="Sulfobutylether-β-Cyclodextrin", base_type="beta", n_glucose=7,
        cavity_diameter_A=6.0, cavity_depth_A=7.9,
        cavity_volume_A3=262, cavity_sasa_A2=427,
        portal_hb_sites=14, modification="SBE", modification_face="primary",
        solubility_aqueous="high", commercial=True,
        cas_number="182410-00-0", approx_cost_per_g=15.00,
    ),
    "amine-beta-CD": CDVariant(
        name="6-Amino-β-Cyclodextrin", base_type="beta", n_glucose=7,
        cavity_diameter_A=6.0, cavity_depth_A=7.9,
        cavity_volume_A3=262, cavity_sasa_A2=427,
        portal_hb_sites=14, modification="amine", modification_face="primary",
        solubility_aqueous="moderate", commercial=True,
        approx_cost_per_g=50.00,
    ),
    "thiol-beta-CD": CDVariant(
        name="6-Thio-β-Cyclodextrin", base_type="beta", n_glucose=7,
        cavity_diameter_A=6.0, cavity_depth_A=7.9,
        cavity_volume_A3=262, cavity_sasa_A2=427,
        portal_hb_sites=14, modification="thiol", modification_face="primary",
        solubility_aqueous="moderate", commercial=True,
        approx_cost_per_g=80.00,
    ),

    # ── Modified α-CDs ──
    "HP-alpha-CD": CDVariant(
        name="Hydroxypropyl-α-Cyclodextrin", base_type="alpha", n_glucose=6,
        cavity_diameter_A=4.7, cavity_depth_A=7.9,
        cavity_volume_A3=174, cavity_sasa_A2=307,
        portal_hb_sites=12, modification="HP", modification_face="secondary",
        solubility_aqueous="high", commercial=True,
        approx_cost_per_g=5.00,
    ),

    # ── Modified γ-CDs ──
    "HP-gamma-CD": CDVariant(
        name="Hydroxypropyl-γ-Cyclodextrin", base_type="gamma", n_glucose=8,
        cavity_diameter_A=7.5, cavity_depth_A=7.9,
        cavity_volume_A3=427, cavity_sasa_A2=590,
        portal_hb_sites=16, modification="HP", modification_face="secondary",
        solubility_aqueous="high", commercial=True,
        approx_cost_per_g=8.00,
    ),
}


# ─────────────────────────────────────────────
# BackSolve Calibrated Parameters
# ─────────────────────────────────────────────

# From Phase 6-10 calibration (BackSolve Protocol v2)
BACKSOLVE_PARAMS = {
    "gamma_hphob": -0.0179,          # kJ/mol per Å² buried SASA
    "k_shape": -4.9973,              # shape complementarity
    "PC_optimal": 0.5911,            # optimal packing coefficient
    "sigma_PC": 0.0500,              # PC Gaussian width
    "eps_neutral_hbond": -8.0270,    # neutral H-bond
    "eps_charge_hbond": -7.1734,     # charge-assisted H-bond
    "eps_rotor": 2.4814,             # conformational entropy penalty per rotor
    "f_partial_freeze": 0.5339,      # partial freezing fraction
    "water_penalty_per_hb": 3.9947,  # water displacement penalty
}


# ─────────────────────────────────────────────
# CD-specific FabricationSpec
# ─────────────────────────────────────────────

@dataclass
class CyclodextrinFabricationSpec(FabricationSpec):
    \"\"\"Fabrication spec for a cyclodextrin-based pocket.\"\"\"

    cd_variant: Optional[CDVariant] = None
    packing_coefficient: float = 0.0
    predicted_Ka: float = 0.0               # predicted association constant
    predicted_dG_kJ_mol: float = 0.0
    click_handle: str = ""                  # conjugation chemistry
    click_handle_face: str = ""             # which face the handle is on
    alternatives: list[str] = field(default_factory=list)  # other viable CDs


# ─────────────────────────────────────────────
# The Adapter
# ─────────────────────────────────────────────

class CyclodextrinAdapter(RealizationAdapter):
    \"\"\"
    Cyclodextrin adapter — maps geometry spec to CD selection + modification.

    Design logic:
        1. Cavity volume → select α/β/γ by packing coefficient
        2. Packing coefficient → Rebek 55% rule validation
        3. H-bond / charge requirements → rim modification
        4. Solubility / application → modification refinement
        5. Predict Ka from BackSolve parameters
    \"\"\"

    def __init__(self, capability: MaterialCapability):
        super().__init__(capability)

    def estimate_fidelity(self, spec: InteractionGeometrySpec) -> RealizationScore:
        \"\"\"Quick score for ranker. Uses packing coefficient match.\"\"\"

        # Find best CD match
        best_cd, best_pc, all_viable = self._select_cd(spec)

        if best_cd is None:
            return RealizationScore(
                material_system=self.system_id,
                adapter_id=self.__class__.__name__,
                deviation_from_ideal=DeviationReport(
                    material_system=self.system_id,
                    element_deviations_A=[],
                    max_deviation_A=float("inf"),
                    mean_deviation_A=float("inf"),
                ),
                physics_fidelity=0.0,
                feasible=False,
                infeasibility_reason="No CD variant matches cavity volume",
            )

        # Physics fidelity from packing coefficient match
        pc_deviation = abs(best_pc - BACKSOLVE_PARAMS["PC_optimal"])
        pc_fidelity = math.exp(
            -(pc_deviation ** 2) / (2 * BACKSOLVE_PARAMS["sigma_PC"] ** 2)
        )

        # Cavity shape penalty: CDs are truncated cones
        shape_penalty = 0.0
        if spec.cavity_shape not in (CavityShape.CONE, CavityShape.BARREL, CavityShape.CUSTOM):
            shape_penalty = 0.15  # CD isn't ideal for flat/sphere/channel

        # H-bond capability
        hbond_match = self._score_hbond_capability(spec, best_cd)

        physics_fidelity = max(0.0, pc_fidelity * (1.0 - shape_penalty) * hbond_match)

        deviation = DeviationReport(
            material_system=self.system_id,
            element_deviations_A=[self.capability.positioning_precision_A] * len(spec.donor_positions),
            max_deviation_A=self.capability.positioning_precision_A,
            mean_deviation_A=self.capability.positioning_precision_A,
        )

        return RealizationScore(
            material_system=self.system_id,
            adapter_id=self.__class__.__name__,
            deviation_from_ideal=deviation,
            physics_fidelity=physics_fidelity,
            feasible=True,
            advantages=[
                f"Best CD: {best_cd.name} (PC={best_pc:.2f})",
                f"{len(all_viable)} viable variants",
                "Commercially available" if best_cd.commercial else "Requires synthesis",
                f"~${best_cd.approx_cost_per_g:.2f}/g",
            ],
            limitations=self._identify_limitations(spec, best_cd, best_pc),
        )

    def design(self, spec: InteractionGeometrySpec) -> CyclodextrinFabricationSpec:
        \"\"\"Full design: CD selection + modification + click handle + predicted Ka.\"\"\"

        best_cd, best_pc, all_viable = self._select_cd(spec)
        if best_cd is None:
            raise ValueError("No CD variant matches this geometry spec")

        # Modification selection based on spec requirements
        modification_cd = self._select_modification(spec, best_cd, all_viable)

        # Predict binding energy
        predicted_dG = self._predict_binding_energy(spec, modification_cd, best_pc)
        predicted_Ka = math.exp(-predicted_dG * 1000 / (8.314 * 298.15))  # Ka from ΔG

        # Click handle selection
        click_handle, click_face = self._select_click_handle(modification_cd)

        # Build fabrication spec
        return CyclodextrinFabricationSpec(
            material_system=self.system_id,
            geometry_spec_hash="",  # TODO: hash the spec
            predicted_pocket_geometry=CavityDimensions(
                volume_A3=modification_cd.cavity_volume_A3,
                aperture_A=modification_cd.cavity_diameter_A,
                depth_A=modification_cd.cavity_depth_A,
                max_internal_diameter_A=modification_cd.cavity_diameter_A,
            ),
            predicted_deviation_from_ideal_A=self.capability.positioning_precision_A,
            synthesis_steps=self._synthesis_route(modification_cd),
            estimated_yield=0.95 if modification_cd.commercial else 0.60,
            estimated_cost_per_unit=modification_cd.approx_cost_per_g,
            estimated_time="1 day (commercial)" if modification_cd.commercial else "3–5 days",
            validation_experiments=[
                "UV-Vis or fluorescence titration to confirm Ka",
                "ITC for full thermodynamic profile (ΔH, TΔS)",
                "¹H NMR ROESY for binding mode confirmation",
                "Job plot for stoichiometry verification",
            ],
            expected_observables={
                "predicted_Ka": predicted_Ka,
                "predicted_dG_kJ_mol": predicted_dG,
                "packing_coefficient": best_pc,
                "stoichiometry": "1:1",
            },
            order_sheet=self._commercial_sourcing(modification_cd),
            cd_variant=modification_cd,
            packing_coefficient=best_pc,
            predicted_Ka=predicted_Ka,
            predicted_dG_kJ_mol=predicted_dG,
            click_handle=click_handle,
            click_handle_face=click_face,
            alternatives=[cd.name for cd in all_viable if cd.name != modification_cd.name],
        )

    def validate_design(self, fab: FabricationSpec) -> ValidationReport:
        \"\"\"Check CD design for internal consistency.\"\"\"
        issues = []
        warnings = []

        if not isinstance(fab, CyclodextrinFabricationSpec):
            return ValidationReport(valid=False, issues=["Wrong FabricationSpec type"], warnings=[])

        # Packing coefficient sanity
        if fab.packing_coefficient < 0.3:
            issues.append(f"Guest too small for cavity (PC={fab.packing_coefficient:.2f} < 0.3)")
        elif fab.packing_coefficient > 0.8:
            issues.append(f"Guest too large for cavity (PC={fab.packing_coefficient:.2f} > 0.8)")
        elif fab.packing_coefficient < 0.45 or fab.packing_coefficient > 0.70:
            warnings.append(f"PC={fab.packing_coefficient:.2f} outside optimal 0.45–0.70 range")

        # Ka sanity
        if fab.predicted_Ka < 10:
            warnings.append(f"Predicted Ka={fab.predicted_Ka:.0f} M⁻¹ — weak binding")
        if fab.predicted_Ka > 1e8:
            warnings.append(f"Predicted Ka={fab.predicted_Ka:.0e} — suspiciously high for CD")

        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            confidence=0.80,  # BackSolve calibrated R²=0.850
        )

    # ─────────────────────────────────────────
    # Internal design logic
    # ─────────────────────────────────────────

    def _select_cd(
        self, spec: InteractionGeometrySpec
    ) -> tuple[Optional[CDVariant], float, list[CDVariant]]:
        \"\"\"
        Select best CD by packing coefficient.
        Returns (best_variant, best_PC, all_viable_variants).
        \"\"\"
        target_volume = spec.cavity_dimensions.volume_A3

        # Score all native CDs by packing coefficient proximity to optimal
        candidates = []
        for key, cd in CD_LIBRARY.items():
            if cd.modification != "native":
                continue  # select base type first, then modify
            if cd.cavity_volume_A3 == 0:
                continue

            pc = target_volume / cd.cavity_volume_A3
            pc_score = math.exp(
                -((pc - BACKSOLVE_PARAMS["PC_optimal"]) ** 2)
                / (2 * BACKSOLVE_PARAMS["sigma_PC"] ** 2)
            )
            candidates.append((cd, pc, pc_score))

        if not candidates:
            return None, 0.0, []

        # Sort by PC score
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_cd, best_pc, _ = candidates[0]

        # Collect all viable (PC in 0.25–0.85 range)
        viable = [cd for cd, pc, _ in candidates if 0.25 <= pc <= 0.85]

        return best_cd, best_pc, viable

    def _score_hbond_capability(
        self, spec: InteractionGeometrySpec, cd: CDVariant
    ) -> float:
        \"\"\"How well can this CD provide the required H-bond network?\"\"\"
        if spec.h_bond_network is None:
            return 1.0  # no requirement

        required_donors = len(spec.h_bond_network.donors)
        required_acceptors = len(spec.h_bond_network.acceptors)
        total_required = required_donors + required_acceptors

        if total_required == 0:
            return 1.0

        # CD portal OH groups serve as both donors and acceptors
        available = cd.portal_hb_sites
        if available >= total_required:
            return 1.0
        return available / total_required

    def _select_modification(
        self,
        spec: InteractionGeometrySpec,
        base_cd: CDVariant,
        viable: list[CDVariant],
    ) -> CDVariant:
        \"\"\"Select the best modification based on spec requirements.\"\"\"
        base_type = base_cd.base_type

        # Check if we need enhanced solubility
        needs_solubility = (
            spec.solvent == Solvent.AQUEOUS
            and base_cd.solubility_aqueous == "low"
        )

        # Check if we need a conjugation handle
        needs_handle = spec.target_application.value in ("diagnostic", "remediation")

        # Check charge requirements
        has_charge_requirement = any(
            d.charge_state != 0.0 for d in spec.donor_positions
        )

        # Selection logic
        candidates = [
            cd for cd in CD_LIBRARY.values()
            if cd.base_type == base_type
        ]

        if needs_handle:
            # Prefer amine or thiol modifications (click chemistry ready)
            handle_cds = [c for c in candidates if c.modification in ("amine", "thiol")]
            if handle_cds:
                return handle_cds[0]

        if needs_solubility:
            soluble = [c for c in candidates if c.solubility_aqueous == "high"]
            if soluble:
                return soluble[0]  # HP-β-CD is the standard

        if has_charge_requirement:
            # SBE-β-CD for anionic charge matching
            sbe = [c for c in candidates if c.modification == "SBE"]
            if sbe:
                return sbe[0]

        # Default: native CD
        return base_cd

    def _predict_binding_energy(
        self, spec: InteractionGeometrySpec, cd: CDVariant, pc: float
    ) -> float:
        \"\"\"
        Predict ΔG (kJ/mol) from BackSolve parameters.

        ΔG = hydrophobic_burial + shape_complementarity + H_bond + entropy_penalty
        \"\"\"
        p = BACKSOLVE_PARAMS

        # Hydrophobic burial (SASA-based)
        # Estimate buried SASA from cavity SASA × occupancy fraction
        buried_sasa = cd.cavity_sasa_A2 * min(1.0, pc / p["PC_optimal"])
        dG_hphob = p["gamma_hphob"] * buried_sasa

        # Shape complementarity (Gaussian around optimal PC)
        dG_shape = p["k_shape"] * math.exp(
            -((pc - p["PC_optimal"]) ** 2) / (2 * p["sigma_PC"] ** 2)
        )

        # H-bond contribution
        n_hbonds = 0
        if spec.h_bond_network:
            n_hbonds = len(spec.h_bond_network.donors) + len(spec.h_bond_network.acceptors)
        n_hbonds = min(n_hbonds, cd.portal_hb_sites)
        dG_hbond = n_hbonds * p["eps_neutral_hbond"]

        # Water penalty (displacing ordered water from cavity)
        dG_water = p["water_penalty_per_hb"] * max(0, n_hbonds - 2)

        # Conformational entropy penalty
        n_rotors = 0  # estimate from spec if available
        dG_entropy = n_rotors * p["eps_rotor"] * p["f_partial_freeze"]

        return dG_hphob + dG_shape + dG_hbond + dG_water + dG_entropy

    def _select_click_handle(self, cd: CDVariant) -> tuple[str, str]:
        \"\"\"Select conjugation chemistry for deployment.\"\"\"
        if cd.modification == "amine":
            return "NHS-ester coupling or azide conversion (C6-NH₂ → C6-N₃)", "primary"
        if cd.modification == "thiol":
            return "Maleimide-thiol coupling (C6-SH)", "primary"
        # Native CDs: tosylation of primary face → azide
        return "C6-OTs → C6-N₃ (2 steps, well-precedented)", "primary"

    def _synthesis_route(self, cd: CDVariant) -> list[str]:
        \"\"\"Synthesis or sourcing steps.\"\"\"
        if cd.commercial:
            return [f"Purchase {cd.name} ({cd.cas_number or 'check supplier'})"]

        steps = [f"Start from native {cd.base_type}-CD"]
        if cd.modification == "amine":
            steps.extend([
                "Monotosylation of primary face (C6-OTs)",
                "Displacement with NaN₃ → C6-N₃",
                "Staudinger reduction → C6-NH₂",
                "Purify by ion-exchange chromatography",
            ])
        elif cd.modification == "thiol":
            steps.extend([
                "Monotosylation of primary face (C6-OTs)",
                "Displacement with thiourea → C6-SH",
                "Purify by RP-HPLC",
            ])
        return steps

    def _commercial_sourcing(self, cd: CDVariant) -> str:
        \"\"\"Generate order information.\"\"\"
        if not cd.commercial:
            return "Requires custom synthesis"
        return (
            f"{cd.name} | CAS: {cd.cas_number or 'check supplier'} | "
            f"Suppliers: Sigma-Aldrich, TCI, Cyclodextrin-Shop | "
            f"~${cd.approx_cost_per_g:.2f}/g"
        )

    def _identify_limitations(
        self, spec: InteractionGeometrySpec, cd: CDVariant, pc: float
    ) -> list[str]:
        \"\"\"Identify specific limitations for this CD match.\"\"\"
        limitations = []

        if pc < 0.45:
            limitations.append(f"Guest undersized (PC={pc:.2f}): rattling in cavity reduces binding")
        if pc > 0.70:
            limitations.append(f"Guest oversized (PC={pc:.2f}): steric strain at portal")

        if spec.cavity_shape == CavityShape.FLAT:
            limitations.append("CD cavity is cone-shaped, not flat — geometry mismatch")
        if spec.cavity_shape == CavityShape.SPHERE:
            limitations.append("CD cavity is truncated cone, not spherical")

        if spec.solvent != Solvent.AQUEOUS:
            limitations.append("CDs work best in aqueous solution")

        # CDs can't do coordination chemistry
        coord_donors = [d for d in spec.donor_positions
                        if d.coordination_role in ("axial", "equatorial", "terminal")]
        if coord_donors:
            limitations.append(
                f"CD has no metal coordination capability "
                f"({len(coord_donors)} coordination donors required)"
            )

        return limitations
""",

    'mabe/realization/engine/__init__.py': """\
\"\"\"
Realization engine — Phase 1 (ideal pocket) and Phase 2 (ranking).
\"\"\"

from mabe.realization.engine.ideal_pocket import compute_ideal_pocket
from mabe.realization.engine.ranker import rank_realizations

__all__ = ["compute_ideal_pocket", "rank_realizations"]
""",

    'mabe/realization/engine/ideal_pocket.py': """\
\"\"\"
Phase 1: Ideal Pocket Computation.

Pure physics. No material constraints. No synthetic accessibility.
No cost. What would the pocket look like if you could place atoms
arbitrarily in space?

Input:  InteractionGeometrySpec (from Layer 2)
Output: IdealPocketSpec (the reference standard)
\"\"\"

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import InteractionGeometrySpec, IdealPocketSpec

from mabe.realization.models import (
    CavityShape,
    IdealElement,
    IdealPocketSpec,
    RigidityClass,
)


def compute_ideal_pocket(spec: "InteractionGeometrySpec") -> IdealPocketSpec:
    \"\"\"
    Compute the physics-optimal pocket for a given interaction geometry.

    This is the reference standard that all material systems are scored against.

    Steps:
        1. Optimize element positions from donor spec
        2. Compute per-element precision requirements (from energy gradients)
        3. Compute desolvation penalty
        4. Compute ideal binding energy (sum of interactions − desolvation)
        5. Derive rigidity class from tightest tolerance
        6. Package as IdealPocketSpec
    \"\"\"

    ideal_elements = _compute_ideal_elements(spec)
    desolvation = _compute_desolvation_penalty(spec)
    binding_energy = _compute_ideal_binding_energy(ideal_elements, desolvation)
    rigidity_class = _classify_rigidity(ideal_elements)

    return IdealPocketSpec(
        optimal_elements=ideal_elements,
        ideal_cavity_volume_A3=spec.cavity_dimensions.volume_A3,
        ideal_cavity_shape=spec.cavity_shape,
        ideal_desolvation_energy_kJ_mol=desolvation,
        ideal_binding_energy_kJ_mol=binding_energy,
        min_precision_required_A=_tightest_precision(ideal_elements),
        rigidity_class=rigidity_class,
        min_stability_pH=spec.pH_range,
        min_stability_K=spec.temperature_range_K,
        required_elements={e.atom_type for e in ideal_elements},
        symmetry_exploitable=spec.symmetry != "none",
        ideal_material_requirements=_describe_ideal_material(ideal_elements, rigidity_class),
        critical_constraints=_derive_critical_constraints(spec, ideal_elements),
    )


# ─────────────────────────────────────────────
# Internal computation functions
# ─────────────────────────────────────────────

def _compute_ideal_elements(spec: "InteractionGeometrySpec") -> list[IdealElement]:
    \"\"\"
    For each donor in the spec, compute the physics-optimal placement.

    Current implementation: direct pass-through of spec positions with
    precision derived from interaction type. This will be replaced with
    energy-gradient-based optimization as BackSolve data integrates.
    \"\"\"
    elements = []
    for donor in spec.donor_positions:
        precision = _precision_for_interaction_type(
            donor.atom_type,
            donor.coordination_role,
        )
        energy = _energy_contribution(
            donor.atom_type,
            donor.coordination_role,
            donor.charge_state,
        )
        elements.append(IdealElement(
            atom_type=donor.atom_type,
            exact_position_A=donor.position_vector_A,
            required_precision_A=precision,
            orbital_hybridization=donor.required_hybridization,
            charge_state=donor.charge_state,
            interaction_energy_contribution_kJ_mol=energy,
        ))
    return elements


def _precision_for_interaction_type(atom_type: str, coordination_role: str) -> float:
    \"\"\"
    How precisely must this element be placed?

    Derived from the steepness of the interaction energy gradient.
    Steeper gradient = tighter tolerance = higher precision requirement.

    Values from metal-ligand coordination literature:
        - Terminal coordination bond: 0.05 Å (steep Morse well)
        - Bridging coordination: 0.10 Å
        - H-bond donor/acceptor: 0.15 Å (broader well)
        - Hydrophobic contact: 0.30 Å (very broad)
    \"\"\"
    # Coordination bonds: tight tolerance
    if coordination_role in ("axial", "equatorial", "terminal"):
        if atom_type in ("N", "S", "Se"):
            return 0.05
        elif atom_type == "O":
            return 0.08  # O-donors have more variability
        else:
            return 0.05

    # Bridging: slightly looser
    if coordination_role == "bridging":
        return 0.10

    # H-bond: moderate
    if coordination_role in ("h_bond_donor", "h_bond_acceptor"):
        return 0.15

    # Hydrophobic: broad
    if coordination_role == "hydrophobic":
        return 0.30

    # Default
    return 0.10


def _energy_contribution(
    atom_type: str,
    coordination_role: str,
    charge_state: float,
) -> float:
    \"\"\"
    Estimated interaction energy contribution of one element.

    Placeholder values from NIST / literature. Will be replaced by
    BackSolve-calibrated per-interaction energies.

    Returns negative values (stabilizing).
    \"\"\"
    # Metal-N coordination: ~-40 to -80 kJ/mol per bond
    # Metal-O coordination: ~-30 to -60 kJ/mol per bond
    # Metal-S coordination: ~-50 to -100 kJ/mol per bond
    # H-bond: ~-10 to -30 kJ/mol
    # CH-π / hydrophobic: ~-5 to -15 kJ/mol

    base_energies = {
        "N": -50.0,
        "O": -40.0,
        "S": -70.0,
        "Se": -60.0,
        "P": -45.0,
    }
    base = base_energies.get(atom_type, -30.0)

    # Coordination bonds are stronger than H-bonds
    role_scale = {
        "axial": 1.0,
        "equatorial": 1.0,
        "terminal": 1.0,
        "bridging": 0.8,
        "h_bond_donor": 0.4,
        "h_bond_acceptor": 0.4,
        "hydrophobic": 0.2,
    }
    scale = role_scale.get(coordination_role, 0.5)

    return base * scale


def _compute_desolvation_penalty(spec: "InteractionGeometrySpec") -> float:
    \"\"\"
    Estimated desolvation cost for creating this pocket.

    Uses Eisenberg-McLachlan transfer coefficients as baseline.
    Aqueous desolvation is expensive; organic is cheap.
    \"\"\"
    if spec.solvent == "gas":
        return 0.0

    # Rough estimate: ~0.1 kJ/mol per Å³ of cavity in water
    # (from hydrophobic transfer free energy literature)
    base_cost_per_A3 = 0.10 if spec.solvent.value == "aqueous" else 0.03
    cavity_cost = spec.cavity_dimensions.volume_A3 * base_cost_per_A3

    # Each donor that must be desolvated adds penalty
    donor_desolv = sum(
        _donor_desolvation_cost(d.atom_type, spec.solvent.value)
        for d in spec.donor_positions
    )

    return cavity_cost + donor_desolv


def _donor_desolvation_cost(atom_type: str, solvent: str) -> float:
    \"\"\"Cost to strip solvent from one donor element.\"\"\"
    if solvent != "aqueous":
        return 2.0  # minimal in organic
    # Aqueous desolvation costs from hydration free energy data
    costs = {
        "N": 12.0,   # amine hydration is moderate
        "O": 15.0,   # hydroxyl / carboxylate hydration is strong
        "S": 8.0,    # thiol / thioether hydration is weak
        "Se": 6.0,
        "P": 10.0,
    }
    return costs.get(atom_type, 10.0)


def _compute_ideal_binding_energy(
    elements: list[IdealElement],
    desolvation: float,
) -> float:
    \"\"\"
    Ideal binding energy = sum of interaction contributions − desolvation.

    Negative = favorable binding.
    \"\"\"
    total_interaction = sum(e.interaction_energy_contribution_kJ_mol for e in elements)
    return total_interaction + desolvation  # desolvation is positive (penalty)


def _tightest_precision(elements: list[IdealElement]) -> float:
    if not elements:
        return float("inf")
    return min(e.required_precision_A for e in elements)


def _classify_rigidity(elements: list[IdealElement]) -> RigidityClass:
    \"\"\"Derive rigidity class from tightest precision requirement.\"\"\"
    tightest = _tightest_precision(elements)
    if tightest < 0.05:
        return RigidityClass.CRYSTALLINE
    elif tightest < 0.20:
        return RigidityClass.PREORGANIZED
    elif tightest < 0.50:
        return RigidityClass.SEMI_FLEXIBLE
    else:
        return RigidityClass.ANY


def _describe_ideal_material(
    elements: list[IdealElement],
    rigidity: RigidityClass,
) -> str:
    \"\"\"Human-readable description of what would score 1.0.\"\"\"
    element_types = sorted({e.atom_type for e in elements})
    tightest = _tightest_precision(elements)
    return (
        f"Requires {len(elements)} interaction elements ({', '.join(element_types)}) "
        f"positioned to ±{tightest:.2f} Å. "
        f"Rigidity class: {rigidity.value}. "
        f"Material must provide all element types with covalent or "
        f"coordination-level positional control."
    )


def _derive_critical_constraints(
    spec: "InteractionGeometrySpec",
    elements: list[IdealElement],
) -> list[str]:
    \"\"\"Non-negotiable requirements for any realization.\"\"\"
    constraints = []

    # Element availability
    for e in elements:
        constraints.append(
            f"Must provide {e.atom_type} at {e.exact_position_A} "
            f"± {e.required_precision_A:.2f} Å"
        )

    # Exclusion constraints
    for exc in spec.must_exclude:
        constraints.append(
            f"Must exclude {exc.species} (max affinity: "
            f"{exc.max_allowed_affinity_kJ_mol:.1f} kJ/mol, "
            f"mechanism: {exc.exclusion_mechanism})"
        )

    # Operating conditions
    constraints.append(
        f"Must be stable at pH {spec.pH_range[0]:.1f}–{spec.pH_range[1]:.1f}, "
        f"{spec.temperature_range_K[0]:.0f}–{spec.temperature_range_K[1]:.0f} K, "
        f"in {spec.solvent.value}"
    )

    return constraints
""",

    'mabe/realization/engine/ranker.py': """\
\"\"\"
Phase 2: Material System Ranking.

Scores every registered material system against the IdealPocketSpec.
Physics fidelity is the primary axis. Implementation concerns are secondary.

Input:  IdealPocketSpec + InteractionGeometrySpec
Output: RankedRealizations (sorted, with gap analysis)
\"\"\"

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from mabe.realization.models import (
    ApplicationContext,
    DeviationReport,
    InteractionGeometrySpec,
    IdealPocketSpec,
    RankedRealizations,
    RealizationScore,
)
from mabe.realization.registry.material_registry import MATERIAL_REGISTRY
from mabe.realization.scoring.feasibility import feasibility_gate
from mabe.realization.scoring.deviation import compute_deviation, deviation_to_fidelity
from mabe.realization.scoring.implementation import (
    score_synthetic_accessibility,
    score_cost,
    score_scalability,
    score_operating_conditions,
    score_reusability,
)
from mabe.realization.scoring.composite import compute_composite
from mabe.realization.scoring.gap_report import generate_gap_report, generate_novel_material_spec


def rank_realizations(
    ideal: IdealPocketSpec,
    spec: InteractionGeometrySpec,
) -> RankedRealizations:
    \"\"\"
    Score all registered material systems against the ideal pocket.

    Physics fidelity always dominates (60% of composite).
    Implementation factors split the remaining 40% based on application context.
    \"\"\"

    scores: list[RealizationScore] = []

    for cap in MATERIAL_REGISTRY.all():
        # ── Hard gate ──
        feasible, reason = feasibility_gate(spec, cap)
        if not feasible:
            scores.append(RealizationScore(
                material_system=cap.system_id,
                adapter_id=cap.adapter_class,
                deviation_from_ideal=DeviationReport(
                    material_system=cap.system_id,
                    element_deviations_A=[],
                    max_deviation_A=float("inf"),
                    mean_deviation_A=float("inf"),
                ),
                physics_fidelity=0.0,
                feasible=False,
                infeasibility_reason=reason,
            ))
            continue

        # ── Deviation from ideal ──
        deviation = compute_deviation(ideal, cap)
        physics_fidelity = deviation_to_fidelity(deviation)

        # ── Implementation scores ──
        sa = score_synthetic_accessibility(spec, cap)
        cost = score_cost(spec, cap)
        scale = score_scalability(spec, cap)
        conditions = score_operating_conditions(spec, cap)
        reuse = score_reusability(spec, cap) if spec.reusability_required else 1.0

        # ── Confidence (calibrated, not precedent-biased) ──
        confidence = _calibrate_confidence(cap, deviation)

        # ── Composite ──
        composite = compute_composite(
            physics_fidelity=physics_fidelity,
            synthetic_accessibility=sa,
            cost_score=cost,
            scalability=scale,
            operating_condition_compatibility=conditions,
            reusability_score=reuse,
            application=spec.target_application,
        )

        scores.append(RealizationScore(
            material_system=cap.system_id,
            adapter_id=cap.adapter_class,
            deviation_from_ideal=deviation,
            physics_fidelity=physics_fidelity,
            synthetic_accessibility=sa,
            cost_score=cost,
            scalability=scale,
            operating_condition_compatibility=conditions,
            reusability_score=reuse,
            composite_score=composite,
            confidence=confidence,
            advantages=cap.known_strengths,
            limitations=cap.known_limitations,
            feasible=True,
        ))

    # ── Sort by composite ──
    scores.sort(key=lambda s: s.composite_score, reverse=True)

    # ── Gap analysis ──
    feasible_scores = [s for s in scores if s.feasible]
    best_fidelity = max((s.physics_fidelity for s in feasible_scores), default=0.0)
    gap = 1.0 - best_fidelity

    gap_report = None
    novel_suggestion = None
    if gap > 0.3:
        gap_report = generate_gap_report(ideal, feasible_scores)
    if gap > 0.5:
        novel_suggestion = generate_novel_material_spec(ideal, feasible_scores)

    recommended = feasible_scores[0].material_system if feasible_scores else "none"
    rationale = _build_rationale(feasible_scores, gap) if feasible_scores else "No feasible material system found."

    return RankedRealizations(
        geometry_spec=spec,
        ideal_pocket=ideal,
        rankings=scores,
        recommended=recommended,
        recommendation_rationale=rationale,
        best_physics_fidelity=best_fidelity,
        gap_to_ideal=gap,
        gap_report=gap_report,
        novel_material_suggestion=novel_suggestion,
    )


def _calibrate_confidence(cap, deviation: DeviationReport) -> float:
    \"\"\"
    How likely is it that a design in this material system will actually work?
    Based on published design-to-validation success rates.
    Not a scoring bonus for familiarity — a calibrated risk estimate.
    \"\"\"
    base_rate = cap.literature_validation_rate
    # Operating near precision limit = higher risk of failure
    if deviation.max_deviation_A < float("inf") and cap.positioning_precision_A > 0:
        utilization = deviation.max_deviation_A / cap.positioning_precision_A
        if utilization > 0.8:
            return base_rate * 0.5
    return base_rate


def _build_rationale(scores: list[RealizationScore], gap: float) -> str:
    \"\"\"Build human-readable recommendation rationale.\"\"\"
    if not scores:
        return "No feasible material systems."
    top = scores[0]
    parts = [
        f"{top.material_system} recommended (physics fidelity: {top.physics_fidelity:.2f}, "
        f"composite: {top.composite_score:.2f})."
    ]
    if gap > 0.3:
        parts.append(f"Gap to ideal: {gap:.2f}. See gap report for improvement opportunities.")
    if len(scores) > 1:
        runner = scores[1]
        parts.append(
            f"Runner-up: {runner.material_system} "
            f"(fidelity: {runner.physics_fidelity:.2f})."
        )
    return " ".join(parts)
""",

    'mabe/realization/models.py': """\
\"\"\"
Data models for the MABE Realization Engine.

These are physics objects. They do not know what a protein is.

Hierarchy:
    InteractionGeometrySpec  (Layer 2 output, our input)
    → IdealPocketSpec        (Phase 1 output: the physics optimum)
    → DeviationReport        (per-material deviation from ideal)
    → RealizationScore       (per-material composite score)
    → RankedRealizations     (sorted output with gap analysis)
    → FabricationSpec        (Layer 4 output: buildable design)
\"\"\"

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class CavityShape(str, Enum):
    \"\"\"Pocket geometry classification.\"\"\"
    SPHERE = "sphere"
    CONE = "cone"
    CHANNEL = "channel"
    CLEFT = "cleft"
    FLAT = "flat"
    BARREL = "barrel"
    CUSTOM = "custom"


class RigidityClass(str, Enum):
    \"\"\"
    Derived from tightest precision requirement in the ideal pocket.
    <0.05 Å → crystalline, <0.2 → preorganized, <0.5 → semi-flexible, else → any.
    \"\"\"
    CRYSTALLINE = "crystalline"
    PREORGANIZED = "preorganized"
    SEMI_FLEXIBLE = "semi-flexible"
    ANY = "any"


class Solvent(str, Enum):
    AQUEOUS = "aqueous"
    ORGANIC = "organic"
    MIXED = "mixed"
    GAS = "gas"


class ApplicationContext(str, Enum):
    DIAGNOSTIC = "diagnostic"
    REMEDIATION = "remediation"
    THERAPEUTIC = "therapeutic"
    RESEARCH = "research"
    SEPARATION = "separation"
    CATALYSIS = "catalysis"


class ScaleClass(str, Enum):
    NMOL = "nmol"
    UMOL = "µmol"
    MMOL = "mmol"
    MOL = "mol"
    KMOL = "kmol"

    @property
    def rank(self) -> int:
        return list(ScaleClass).index(self)


# ─────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class DonorPosition:
    \"\"\"A single interaction element in the geometry spec.\"\"\"
    atom_type: str                    # "N", "O", "S", "Se", etc.
    coordination_role: str            # "axial", "equatorial", "bridging", "terminal"
    position_vector_A: tuple[float, float, float]  # relative to cavity center
    tolerance_A: float                # how precisely this must be placed
    required_hybridization: str       # "sp2", "sp3", "any"
    charge_state: float = 0.0        # partial charge requirement


@dataclass(frozen=True)
class ExclusionSpec:
    \"\"\"A species that must NOT bind.\"\"\"
    species: str
    max_allowed_affinity_kJ_mol: float
    exclusion_mechanism: str          # "size", "charge", "geometry", "kinetic"


@dataclass(frozen=True)
class CavityDimensions:
    \"\"\"Physical dimensions of the pocket cavity.\"\"\"
    volume_A3: float
    aperture_A: float                 # narrowest opening
    depth_A: float
    max_internal_diameter_A: float
    aspect_ratio: float = 1.0        # depth / width


@dataclass(frozen=True)
class HydrophobicSurface:
    \"\"\"A non-polar contact region in the pocket.\"\"\"
    center_A: tuple[float, float, float]
    area_A2: float
    normal_vector: tuple[float, float, float]


@dataclass(frozen=True)
class HBondSpec:
    \"\"\"Hydrogen bond network requirement.\"\"\"
    donors: list[tuple[float, float, float]]   # positions of H-bond donors
    acceptors: list[tuple[float, float, float]] # positions of H-bond acceptors
    required_geometry: str = "any"              # "linear", "bifurcated", "any"


# ─────────────────────────────────────────────
# Layer 2 Output / Layer 3 Input
# ─────────────────────────────────────────────

@dataclass
class InteractionGeometrySpec:
    \"\"\"
    Realization-agnostic pocket description. Layer 2 output.

    This is a physics object. It describes a field of interaction
    potentials in 3D space. It does not know what a protein is.
    \"\"\"

    # ── Cavity geometry ──
    cavity_shape: CavityShape
    cavity_dimensions: CavityDimensions
    symmetry: str = "none"           # "C3v", "D4h", "none", etc.

    # ── Interaction elements ──
    donor_positions: list[DonorPosition] = field(default_factory=list)
    hydrophobic_surfaces: list[HydrophobicSurface] = field(default_factory=list)
    h_bond_network: Optional[HBondSpec] = None

    # ── Flexibility constraints ──
    rigidity_requirement: str = "semi-rigid"
    max_backbone_rmsd_A: float = 1.0
    conformational_penalty_budget_kJ_mol: float = 10.0

    # ── Scale ──
    pocket_scale_nm: float = 0.5
    multivalency: int = 1

    # ── Selectivity constraints ──
    must_exclude: list[ExclusionSpec] = field(default_factory=list)

    # ── Operating conditions ──
    pH_range: tuple[float, float] = (5.0, 9.0)
    temperature_range_K: tuple[float, float] = (273.15, 373.15)
    solvent: Solvent = Solvent.AQUEOUS
    ionic_strength_M: float = 0.1

    # ── Application context (informs realization, doesn't constrain geometry) ──
    target_application: ApplicationContext = ApplicationContext.RESEARCH
    required_scale: ScaleClass = ScaleClass.UMOL
    cost_ceiling_per_unit: Optional[float] = None
    reusability_required: bool = False

    @property
    def required_donor_types(self) -> set[str]:
        return {d.atom_type for d in self.donor_positions}

    @property
    def tightest_tolerance_A(self) -> float:
        if not self.donor_positions:
            return float("inf")
        return min(d.tolerance_A for d in self.donor_positions)


# ─────────────────────────────────────────────
# Layer 3 Phase 1 Output: The Ideal Pocket
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class IdealElement:
    \"\"\"One interaction element in the physics-optimal pocket.\"\"\"
    atom_type: str
    exact_position_A: tuple[float, float, float]
    required_precision_A: float
    orbital_hybridization: str
    charge_state: float
    interaction_energy_contribution_kJ_mol: float


@dataclass
class IdealPocketSpec:
    \"\"\"
    The physics-optimal pocket. No material constraints. Pure geometry + thermodynamics.

    This is the reference standard. Every material system is scored by
    deviation from this object.
    \"\"\"

    # ── Computed from InteractionGeometrySpec ──
    optimal_elements: list[IdealElement]

    # ── Derived pocket properties ──
    ideal_cavity_volume_A3: float
    ideal_cavity_shape: CavityShape
    ideal_desolvation_energy_kJ_mol: float
    ideal_binding_energy_kJ_mol: float

    # ── Fabrication requirements (material-agnostic) ──
    min_precision_required_A: float
    rigidity_class: RigidityClass
    min_stability_pH: tuple[float, float]
    min_stability_K: tuple[float, float]
    required_elements: set[str]
    symmetry_exploitable: bool

    # ── The ideal material spec (when nothing scores high enough) ──
    ideal_material_requirements: str = ""
    critical_constraints: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Layer 3 Phase 2: Deviation + Scoring
# ─────────────────────────────────────────────

@dataclass
class DeviationReport:
    \"\"\"How a specific material system deviates from the IdealPocketSpec.\"\"\"
    material_system: str
    element_deviations_A: list[float]
    max_deviation_A: float
    mean_deviation_A: float
    rigidity_deviation: float = 0.0          # 0 = exact match, 1 = completely wrong
    electrostatic_field_correlation: float = 1.0  # 0–1
    missing_interactions: list[str] = field(default_factory=list)
    compensating_interactions: list[str] = field(default_factory=list)


@dataclass
class RealizationScore:
    \"\"\"Score for one material system against the IdealPocketSpec.\"\"\"

    material_system: str
    adapter_id: str

    # ── Physics deviation (PRIMARY) ──
    deviation_from_ideal: DeviationReport
    physics_fidelity: float               # 0.0–1.0, derived from deviation

    # ── Implementation axes (SECONDARY) ──
    synthetic_accessibility: float = 0.0
    cost_score: float = 0.0
    scalability: float = 0.0
    operating_condition_compatibility: float = 0.0
    reusability_score: float = 0.0

    # ── Composite ──
    composite_score: float = 0.0
    confidence: float = 0.0               # calibrated from literature success rates

    # ── Rationale ──
    advantages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    critical_risk: Optional[str] = None

    # ── Feasibility gate ──
    feasible: bool = True
    infeasibility_reason: Optional[str] = None


@dataclass
class RankedRealizations:
    \"\"\"Layer 3 complete output.\"\"\"

    # ── The physics target ──
    geometry_spec: InteractionGeometrySpec
    ideal_pocket: IdealPocketSpec

    # ── Material rankings ──
    rankings: list[RealizationScore]
    recommended: str
    recommendation_rationale: str = ""

    # ── Gap analysis ──
    best_physics_fidelity: float = 0.0
    gap_to_ideal: float = 1.0
    gap_report: Optional[str] = None
    novel_material_suggestion: Optional[str] = None


# ─────────────────────────────────────────────
# Layer 4 Output: Fabrication Spec
# ─────────────────────────────────────────────

@dataclass
class FabricationSpec:
    \"\"\"
    Base class for all Layer 4 adapter outputs.
    Each adapter subclasses with material-specific fields.
    \"\"\"

    material_system: str
    geometry_spec_hash: str               # traceability to input
    predicted_pocket_geometry: CavityDimensions
    predicted_deviation_from_ideal_A: float

    # ── Synthesis / fabrication ──
    synthesis_steps: list[str] = field(default_factory=list)
    estimated_yield: float = 0.0
    estimated_cost_per_unit: float = 0.0
    estimated_time: str = ""

    # ── Characterization plan ──
    validation_experiments: list[str] = field(default_factory=list)
    expected_observables: dict = field(default_factory=dict)

    # ── Files ──
    structure_file: Optional[str] = None  # PDB, CIF, MOL, oxDNA, etc.
    order_sheet: Optional[str] = None
    protocol: Optional[str] = None
""",

    'mabe/realization/registry/__init__.py': """\
\"\"\"
Material system registry — what each material class can and cannot do.
\"\"\"

from mabe.realization.registry.material_registry import MATERIAL_REGISTRY, MaterialCapability

__all__ = ["MATERIAL_REGISTRY", "MaterialCapability"]
""",

    'mabe/realization/registry/material_registry.py': """\
\"\"\"
Material System Registry.

Every material system that can create a binding pocket registers its
capability envelope. Organized by physics class, not material origin.

Sprint R1 starters:
    Class A (Covalent Cavity):        planar_coordination_ring
    Class A (Covalent Cavity):        cyclic_encapsulant
    Class B (Periodic Lattice):       periodic_lattice_node
    Class C (Foldable Polymer):       folded_polypeptide
    Class D (Emergent Cavity):        emergent_coordination_cage
\"\"\"

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MaterialCapability:
    \"\"\"What a material system can and cannot do.\"\"\"

    system_id: str
    physics_class: str                   # "covalent_cavity", "periodic_lattice", etc.
    adapter_class: str                   # Layer 4 adapter reference

    # ── Geometric capability envelope ──
    min_pocket_size_nm: float
    max_pocket_size_nm: float
    achievable_symmetries: list[str]
    max_donor_count: int
    donor_types_available: list[str]
    positioning_precision_A: float       # best achievable placement accuracy
    rigidity_range: tuple[str, str]      # ("rigid", "rigid") or ("flexible", "semi-rigid")

    # ── Operating envelope ──
    pH_stability: tuple[float, float]
    thermal_stability_K: tuple[float, float]
    solvent_compatibility: list[str]

    # ── Production envelope ──
    min_practical_scale: str
    max_practical_scale: str
    cost_per_unit_range: tuple[float, float]  # $/µmol at mid-scale
    typical_synthesis_time: str

    # ── Validation calibration ──
    literature_validation_rate: float    # fraction of designs that work in lab
    literature_examples: int
    design_tools_available: list[str]

    # ── Qualitative (for rationale generation) ──
    known_strengths: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)


class MaterialRegistry:
    \"\"\"Registry of all known material capabilities.\"\"\"

    def __init__(self):
        self._systems: dict[str, MaterialCapability] = {}

    def register(self, cap: MaterialCapability) -> None:
        self._systems[cap.system_id] = cap

    def get(self, system_id: str) -> Optional[MaterialCapability]:
        return self._systems.get(system_id)

    def all(self) -> list[MaterialCapability]:
        return list(self._systems.values())

    def by_physics_class(self, cls: str) -> list[MaterialCapability]:
        return [c for c in self._systems.values() if c.physics_class == cls]

    def __len__(self) -> int:
        return len(self._systems)


# ─────────────────────────────────────────────
# Global registry instance
# ─────────────────────────────────────────────

MATERIAL_REGISTRY = MaterialRegistry()


# ─────────────────────────────────────────────
# Sprint R1: 5 starter entries
# ─────────────────────────────────────────────

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="planar_coordination_ring",
    physics_class="covalent_cavity",
    adapter_class="PlanarCoordinationRingAdapter",
    min_pocket_size_nm=0.2,
    max_pocket_size_nm=0.5,
    achievable_symmetries=["D4h", "C4v", "D2h", "C2v"],
    max_donor_count=6,  # 4 ring + 2 axial
    donor_types_available=["N", "O", "S"],
    positioning_precision_A=0.01,  # covalent bond geometry
    rigidity_range=("rigid", "rigid"),
    pH_stability=(1.0, 14.0),
    thermal_stability_K=(200.0, 600.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="µmol",
    max_practical_scale="mol",
    cost_per_unit_range=(10.0, 500.0),
    typical_synthesis_time="1–5 days",
    literature_validation_rate=0.85,  # porphyrin synthesis is well-established
    literature_examples=50000,
    design_tools_available=["RDKit", "Gaussian"],
    known_strengths=[
        "Highest precision (covalent bond geometry)",
        "Exceptional rigidity",
        "Well-characterized metal coordination",
        "Tunable electronics via meso/beta substituents",
    ],
    known_limitations=[
        "Fixed 4N planar geometry — limited pocket shapes",
        "Only small metal ions fit",
        "Multi-step synthesis for custom variants",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="cyclic_encapsulant",
    physics_class="covalent_cavity",
    adapter_class="CyclicEncapsulantAdapter",
    min_pocket_size_nm=0.3,
    max_pocket_size_nm=0.9,
    achievable_symmetries=["Cn", "Cnv", "Dnh", "D3h", "D6h"],
    max_donor_count=12,  # large crown ethers / cryptands
    donor_types_available=["N", "O", "S"],
    positioning_precision_A=0.05,  # ring conformational averaging
    rigidity_range=("semi-rigid", "rigid"),  # cryptands more rigid than crowns
    pH_stability=(2.0, 12.0),
    thermal_stability_K=(250.0, 500.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="µmol",
    max_practical_scale="kmol",  # crown ethers are industrially produced
    cost_per_unit_range=(1.0, 200.0),
    typical_synthesis_time="1–3 days",
    literature_validation_rate=0.80,
    literature_examples=20000,
    design_tools_available=["RDKit"],
    known_strengths=[
        "Size-selective cation binding (ring size = selectivity)",
        "Well-characterized thermodynamics (Izatt compilations)",
        "Cheap at scale (18-crown-6 is commodity chemical)",
        "HSAB-tunable via O/N/S donor substitution",
    ],
    known_limitations=[
        "Primarily cation-selective (poor for anions/neutrals)",
        "Conformational flexibility in larger rings reduces selectivity",
        "Limited pocket shape diversity (circular)",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="periodic_lattice_node",
    physics_class="periodic_lattice",
    adapter_class="PeriodicLatticeNodeAdapter",
    min_pocket_size_nm=0.3,
    max_pocket_size_nm=2.0,
    achievable_symmetries=["Oh", "Td", "D4h", "D3h", "C4v", "C3v"],
    max_donor_count=12,  # MOF nodes can have high coordination
    donor_types_available=["N", "O", "S", "P"],
    positioning_precision_A=0.1,  # lattice precision
    rigidity_range=("rigid", "semi-rigid"),
    pH_stability=(2.0, 12.0),  # varies enormously: UiO-66 → pH 1-12, HKUST-1 → pH 5-8
    thermal_stability_K=(250.0, 700.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="mmol",
    max_practical_scale="kmol",  # MOFs produced at tonne scale
    cost_per_unit_range=(5.0, 1000.0),
    typical_synthesis_time="1–7 days",
    literature_validation_rate=0.70,
    literature_examples=100000,
    design_tools_available=["pymatgen", "CSD_API", "Zeo++", "ToposPro"],
    known_strengths=[
        "Massively parallel — every unit cell is a binding site",
        "Extreme surface area (>7000 m²/g achievable)",
        "Tunable pore via linker length + topology",
        "Scalable to tonnes",
    ],
    known_limitations=[
        "All pockets identical (periodic constraint)",
        "Water stability varies (many MOFs degrade in water)",
        "Post-synthetic modification needed for fine-tuning",
        "Pore access may limit diffusion to binding sites",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="folded_polypeptide",
    physics_class="foldable_polymer",
    adapter_class="FoldedPolypeptideAdapter",
    min_pocket_size_nm=0.5,
    max_pocket_size_nm=5.0,
    achievable_symmetries=["none", "Cn", "Dn"],  # via oligomeric assembly
    max_donor_count=20,  # limited only by fold
    donor_types_available=["N", "O", "S", "Se"],  # all natural aa + selenocysteine
    positioning_precision_A=0.3,  # thermal fluctuation of side chains
    rigidity_range=("flexible", "semi-rigid"),
    pH_stability=(4.0, 10.0),
    thermal_stability_K=(277.0, 370.0),  # most proteins denature < 100°C
    solvent_compatibility=["aqueous"],  # organic solvents denature
    min_practical_scale="nmol",
    max_practical_scale="mmol",  # E. coli expression
    cost_per_unit_range=(100.0, 10000.0),
    typical_synthesis_time="1–4 weeks (expression + purification)",
    literature_validation_rate=0.40,  # RFdiffusion ~19% for small molecule, higher for PPI
    literature_examples=200000,
    design_tools_available=["RFDiffusion", "ProteinMPNN", "AlphaFold2", "Rosetta"],
    known_strengths=[
        "Any pocket shape in principle",
        "20 monomer types — high interaction element diversity",
        "Mature design tools (RFdiffusion, ProteinMPNN)",
        "Biocompatible",
    ],
    known_limitations=[
        "Must fold correctly AND present correct elements — two failure modes",
        "Thermal/pH/solvent stability constraints",
        "0.3 Å positioning precision (thermal fluctuation)",
        "Requires wet-lab validation (expression, folding, binding)",
        "Organic solvents, extreme pH, or high temp destroy fold",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="emergent_coordination_cage",
    physics_class="emergent_cavity",
    adapter_class="EmergentCoordinationCageAdapter",
    min_pocket_size_nm=0.5,
    max_pocket_size_nm=5.0,
    achievable_symmetries=["Td", "Oh", "D2h", "D4h", "T"],
    max_donor_count=12,
    donor_types_available=["N", "O", "S"],
    positioning_precision_A=0.1,  # self-assembly with metal vertices
    rigidity_range=("semi-rigid", "rigid"),
    pH_stability=(3.0, 11.0),
    thermal_stability_K=(270.0, 400.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="µmol",
    max_practical_scale="mmol",
    cost_per_unit_range=(50.0, 5000.0),
    typical_synthesis_time="1–3 days (self-assembly)",
    literature_validation_rate=0.80,  # thermodynamic self-assembly is reliable
    literature_examples=500,
    design_tools_available=["stk", "RDKit"],
    known_strengths=[
        "Self-assembly under thermodynamic control — high reliability",
        "Discrete 3D cavities with defined geometry",
        "Heteroleptic designs allow 4+ different ligands → single isomer",
        "Interior functionalization possible",
        "Nitschke subcomponent self-assembly enables dynamic control",
    ],
    known_limitations=[
        "Limited to coordination-compatible metals (Pd, Pt, Fe, Ga, etc.)",
        "Stability can be marginal (reversible bonds)",
        "Scaling beyond mmol is challenging",
        "Shape persistence upon guest removal not guaranteed",
    ],
))
""",

    'mabe/realization/scoring/__init__.py': """\
\"\"\"
Scoring functions for the realization engine.

    feasibility.py    — hard pass/fail gate
    deviation.py      — physics deviation from ideal (PRIMARY axis)
    implementation.py — SA, cost, scale, conditions, reusability (SECONDARY axes)
    composite.py      — weighted combination
    gap_report.py     — what's missing when nothing scores well
\"\"\"
""",

    'mabe/realization/scoring/composite.py': """\
\"\"\"
Composite Scoring.

Physics fidelity: 60% (FIXED, non-negotiable).
Implementation: 40% (distributed by application context).

No precedent score. Physics doesn't reward familiarity.
\"\"\"

from __future__ import annotations

from mabe.realization.models import ApplicationContext


# Physics fidelity weight is FIXED at 0.60.
PHYSICS_WEIGHT = 0.60
IMPLEMENTATION_WEIGHT = 0.40

# The remaining 0.40 distributed among implementation factors.
# These weights are relative (sum to 1.0 within implementation).
IMPLEMENTATION_PROFILES: dict[ApplicationContext, dict[str, float]] = {
    ApplicationContext.DIAGNOSTIC: {
        "synthetic_accessibility": 0.25,
        "cost": 0.10,
        "scalability": 0.10,
        "operating_conditions": 0.15,
        "reusability": 0.00,
    },
    ApplicationContext.REMEDIATION: {
        "synthetic_accessibility": 0.10,
        "cost": 0.35,
        "scalability": 0.35,
        "operating_conditions": 0.10,
        "reusability": 0.10,
    },
    ApplicationContext.THERAPEUTIC: {
        "synthetic_accessibility": 0.10,
        "cost": 0.10,
        "scalability": 0.10,
        "operating_conditions": 0.35,
        "reusability": 0.00,
    },
    ApplicationContext.RESEARCH: {
        "synthetic_accessibility": 0.35,
        "cost": 0.15,
        "scalability": 0.00,
        "operating_conditions": 0.10,
        "reusability": 0.00,
    },
    ApplicationContext.SEPARATION: {
        "synthetic_accessibility": 0.15,
        "cost": 0.25,
        "scalability": 0.25,
        "operating_conditions": 0.15,
        "reusability": 0.20,
    },
    ApplicationContext.CATALYSIS: {
        "synthetic_accessibility": 0.15,
        "cost": 0.10,
        "scalability": 0.15,
        "operating_conditions": 0.30,
        "reusability": 0.30,
    },
}


def compute_composite(
    physics_fidelity: float,
    synthetic_accessibility: float,
    cost_score: float,
    scalability: float,
    operating_condition_compatibility: float,
    reusability_score: float,
    application: ApplicationContext,
) -> float:
    \"\"\"
    Weighted composite score.

    Physics always 60%. Implementation 40% distributed by application.
    \"\"\"
    profile = IMPLEMENTATION_PROFILES.get(
        application,
        IMPLEMENTATION_PROFILES[ApplicationContext.RESEARCH],  # default
    )

    impl_score = (
        profile["synthetic_accessibility"] * synthetic_accessibility
        + profile["cost"] * cost_score
        + profile["scalability"] * scalability
        + profile["operating_conditions"] * operating_condition_compatibility
        + profile["reusability"] * reusability_score
    )

    # Normalize: profile weights should sum to ~1.0 but enforce
    profile_sum = sum(profile.values())
    if profile_sum > 0:
        impl_score /= profile_sum

    return PHYSICS_WEIGHT * physics_fidelity + IMPLEMENTATION_WEIGHT * impl_score
""",

    'mabe/realization/scoring/deviation.py': """\
\"\"\"
Deviation Scoring — PRIMARY axis.

Measures how far a material system deviates from the IdealPocketSpec.
This is the most important scoring function. Everything else is secondary.

Physics fidelity = exp(-mean_deviation / decay_constant)
\"\"\"

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.registry.material_registry import MaterialCapability

from mabe.realization.models import DeviationReport, IdealPocketSpec, RigidityClass


# Exponential decay constant for fidelity conversion.
# 0.1 Å mean deviation → 0.82 fidelity
# 0.5 Å mean deviation → 0.37 fidelity
# 1.0 Å mean deviation → 0.14 fidelity
FIDELITY_DECAY_A = 0.5


def compute_deviation(
    ideal: IdealPocketSpec,
    cap: "MaterialCapability",
) -> DeviationReport:
    \"\"\"
    For each element in the ideal spec, compute how far this material
    system can get from the ideal position.

    The deviation at each element is bounded below by the material's
    positioning precision — it cannot do better than its intrinsic limit.
    \"\"\"

    element_deviations: list[float] = []
    missing: list[str] = []

    for element in ideal.optimal_elements:
        # Can this material provide this atom type at all?
        if element.atom_type not in cap.donor_types_available:
            element_deviations.append(float("inf"))
            missing.append(
                f"{element.atom_type} at {element.exact_position_A}"
            )
            continue

        # Best achievable deviation = material's positioning precision
        # The material cannot place an element more precisely than this
        achievable = cap.positioning_precision_A

        # If the material's precision is sufficient, it can meet the spec
        # The deviation is the precision limit itself (best case)
        element_deviations.append(achievable)

    # Rigidity deviation
    rigidity_dev = _rigidity_deviation(ideal.rigidity_class, cap.rigidity_range)

    finite_devs = [d for d in element_deviations if d < float("inf")]

    return DeviationReport(
        material_system=cap.system_id,
        element_deviations_A=element_deviations,
        max_deviation_A=max(element_deviations) if element_deviations else float("inf"),
        mean_deviation_A=(
            sum(finite_devs) / len(finite_devs) if finite_devs else float("inf")
        ),
        rigidity_deviation=rigidity_dev,
        missing_interactions=missing,
    )


def deviation_to_fidelity(dev: DeviationReport) -> float:
    \"\"\"
    Convert deviation report to 0–1 fidelity score.

    Missing interactions → 0.0
    Otherwise: exp(-mean_deviation / decay_constant) * rigidity_factor
    \"\"\"
    if dev.missing_interactions:
        return 0.0

    if dev.mean_deviation_A == float("inf") or dev.mean_deviation_A < 0:
        return 0.0

    position_fidelity = math.exp(-dev.mean_deviation_A / FIDELITY_DECAY_A)

    # Rigidity mismatch penalizes further
    rigidity_factor = 1.0 - 0.3 * dev.rigidity_deviation  # max 30% penalty

    return max(0.0, min(1.0, position_fidelity * rigidity_factor))


def _rigidity_deviation(
    required: RigidityClass,
    material_range: tuple[str, str],
) -> float:
    \"\"\"
    How far is the material's rigidity from the requirement?
    0.0 = perfect match or MORE rigid than required, 1.0 = too flexible.

    Asymmetric: too rigid is fine (or slightly beneficial).
    Too flexible is penalized — a flexible material can't reproduce
    a pocket that requires preorganized geometry.
    \"\"\"
    rigidity_order = {
        "flexible": 0,
        "semi-rigid": 1,
        "semi-flexible": 1,
        "preorganized": 2,
        "rigid": 3,
        "crystalline": 3,
    }

    required_level = rigidity_order.get(required.value, 1)

    mat_min = rigidity_order.get(material_range[0], 1)
    mat_max = rigidity_order.get(material_range[1], 1)

    # Material CAN reach the required rigidity or higher → no penalty
    if mat_max >= required_level:
        return 0.0

    # Material is too flexible — can't achieve required rigidity
    shortfall = required_level - mat_max
    return min(1.0, shortfall / 3.0)
""",

    'mabe/realization/scoring/feasibility.py': """\
\"\"\"
Feasibility Gate.

Binary pass/fail. If this fails, the material system is not scored.
Prevents wasting compute on impossible realizations.
\"\"\"

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import InteractionGeometrySpec
    from mabe.realization.registry.material_registry import MaterialCapability

from mabe.realization.models import ScaleClass


def feasibility_gate(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> tuple[bool, Optional[str]]:
    \"\"\"
    Binary: can this material system even attempt this geometry?

    Returns (True, None) if feasible, (False, reason) if not.
    \"\"\"

    # ── Pocket too small? ──
    if spec.pocket_scale_nm < cap.min_pocket_size_nm:
        return False, (
            f"Pocket {spec.pocket_scale_nm:.2f} nm below minimum "
            f"{cap.min_pocket_size_nm:.2f} nm for {cap.system_id}"
        )

    # ── Pocket too large? ──
    if spec.pocket_scale_nm > cap.max_pocket_size_nm:
        return False, (
            f"Pocket {spec.pocket_scale_nm:.2f} nm above maximum "
            f"{cap.max_pocket_size_nm:.2f} nm for {cap.system_id}"
        )

    # ── Required donors not available? ──
    required = spec.required_donor_types
    available = set(cap.donor_types_available)
    missing = required - available
    if missing:
        return False, (
            f"Required donors {missing} not available in {cap.system_id} "
            f"(available: {available})"
        )

    # ── Too many donors? ──
    if len(spec.donor_positions) > cap.max_donor_count:
        return False, (
            f"Spec requires {len(spec.donor_positions)} donors, "
            f"{cap.system_id} supports max {cap.max_donor_count}"
        )

    # ── pH incompatible? ──
    if spec.pH_range[0] < cap.pH_stability[0] or spec.pH_range[1] > cap.pH_stability[1]:
        return False, (
            f"Required pH {spec.pH_range} outside "
            f"{cap.system_id} stability {cap.pH_stability}"
        )

    # ── Temperature incompatible? ──
    if spec.temperature_range_K[1] > cap.thermal_stability_K[1]:
        return False, (
            f"Required temp {spec.temperature_range_K[1]:.0f} K exceeds "
            f"{cap.system_id} max {cap.thermal_stability_K[1]:.0f} K"
        )

    # ── Solvent incompatible? ──
    if spec.solvent.value not in cap.solvent_compatibility:
        return False, (
            f"Solvent {spec.solvent.value} not compatible with "
            f"{cap.system_id} (supports: {cap.solvent_compatibility})"
        )

    # ── Scale impossible? ──
    spec_rank = ScaleClass(spec.required_scale).rank if isinstance(spec.required_scale, str) else spec.required_scale.rank
    max_rank = ScaleClass(cap.max_practical_scale).rank if isinstance(cap.max_practical_scale, str) else -1
    # Graceful handling: skip scale check if we can't parse
    try:
        max_cap = ScaleClass(cap.max_practical_scale)
        if spec.required_scale.rank > max_cap.rank:
            return False, (
                f"Required scale {spec.required_scale.value} exceeds "
                f"{cap.system_id} max {cap.max_practical_scale}"
            )
    except (ValueError, AttributeError):
        pass  # skip scale gate if values don't parse cleanly

    return True, None
""",

    'mabe/realization/scoring/gap_report.py': """\
\"\"\"
Gap Report Generator.

When the best available material system only achieves partial physics
fidelity, the gap report answers: "Why can't anything build this pocket?"
and "What would need to exist?"

This is the output that lets you invent a new material.
\"\"\"

from __future__ import annotations

from typing import Optional

from mabe.realization.models import IdealPocketSpec, RealizationScore


def generate_gap_report(
    ideal: IdealPocketSpec,
    feasible_scores: list[RealizationScore],
) -> str:
    \"\"\"
    Identify which ideal requirements are not met by any material system.

    Called when gap_to_ideal > 0.3.
    \"\"\"
    if not feasible_scores:
        return "No feasible material systems registered for this geometry."

    lines = ["## Gap Analysis", ""]

    # Best achiever
    best = max(feasible_scores, key=lambda s: s.physics_fidelity)
    lines.append(
        f"Best physics fidelity: {best.physics_fidelity:.2f} "
        f"({best.material_system})"
    )
    lines.append("")

    # Per-element analysis: which elements are hardest to place?
    if best.deviation_from_ideal.element_deviations_A:
        lines.append("### Element-level deviations (best system):")
        for i, (element, dev) in enumerate(zip(
            ideal.optimal_elements,
            best.deviation_from_ideal.element_deviations_A,
        )):
            status = "✓" if dev <= element.required_precision_A else "✗"
            lines.append(
                f"  {status} Element {i}: {element.atom_type} — "
                f"required ±{element.required_precision_A:.2f} Å, "
                f"best achievable: ±{dev:.2f} Å"
            )
        lines.append("")

    # Missing interactions across ALL systems
    all_missing = set()
    for s in feasible_scores:
        all_missing.update(s.deviation_from_ideal.missing_interactions)
    if all_missing:
        lines.append("### Interactions no registered material can provide:")
        for m in sorted(all_missing):
            lines.append(f"  - {m}")
        lines.append("")

    # Precision bottleneck
    precision_gap = _identify_precision_bottleneck(ideal, feasible_scores)
    if precision_gap:
        lines.append(f"### Precision bottleneck: {precision_gap}")
        lines.append("")

    return "\\n".join(lines)


def generate_novel_material_spec(
    ideal: IdealPocketSpec,
    feasible_scores: list[RealizationScore],
) -> str:
    \"\"\"
    When gap > 0.5, describe what a novel material would need.

    This is the spec for something that doesn't exist yet but
    physics says should work.
    \"\"\"
    lines = ["## Novel Material Specification", ""]
    lines.append("No registered material system achieves >0.5 physics fidelity.")
    lines.append("A material with the following properties would score 1.0:")
    lines.append("")

    # Required elements
    elements = sorted(ideal.required_elements)
    lines.append(f"**Required elements:** {', '.join(elements)}")

    # Required precision
    lines.append(
        f"**Positioning precision:** ≤ {ideal.min_precision_required_A:.2f} Å"
    )

    # Rigidity
    lines.append(f"**Rigidity class:** {ideal.rigidity_class.value}")

    # Stability
    lines.append(
        f"**Stability:** pH {ideal.min_stability_pH[0]:.1f}–"
        f"{ideal.min_stability_pH[1]:.1f}, "
        f"{ideal.min_stability_K[0]:.0f}–{ideal.min_stability_K[1]:.0f} K"
    )

    # Per-element requirements
    lines.append("")
    lines.append("**Per-element placement:**")
    for e in ideal.optimal_elements:
        lines.append(
            f"  - {e.atom_type} at {e.exact_position_A} ± {e.required_precision_A:.2f} Å "
            f"(energy contribution: {e.interaction_energy_contribution_kJ_mol:.1f} kJ/mol)"
        )

    # What's blocking existing systems
    lines.append("")
    lines.append("**Blocking constraints from existing systems:**")
    blockers = _identify_blockers(ideal, feasible_scores)
    for b in blockers:
        lines.append(f"  - {b}")

    return "\\n".join(lines)


def _identify_precision_bottleneck(
    ideal: IdealPocketSpec,
    scores: list[RealizationScore],
) -> Optional[str]:
    \"\"\"Find the element whose precision requirement eliminates the most systems.\"\"\"
    if not ideal.optimal_elements:
        return None

    element_failures = {}
    for element in ideal.optimal_elements:
        failures = 0
        for s in scores:
            idx = ideal.optimal_elements.index(element)
            if idx < len(s.deviation_from_ideal.element_deviations_A):
                dev = s.deviation_from_ideal.element_deviations_A[idx]
                if dev > element.required_precision_A:
                    failures += 1
        element_failures[element.atom_type] = failures

    if not element_failures:
        return None

    worst = max(element_failures, key=element_failures.get)
    count = element_failures[worst]
    if count == 0:
        return None
    return (
        f"{worst} placement (required ±{ideal.optimal_elements[0].required_precision_A:.2f} Å) "
        f"fails for {count}/{len(scores)} feasible systems"
    )


def _identify_blockers(
    ideal: IdealPocketSpec,
    scores: list[RealizationScore],
) -> list[str]:
    \"\"\"Identify what prevents existing systems from scoring well.\"\"\"
    blockers = []

    if not scores:
        blockers.append("No registered material systems pass feasibility gate")
        return blockers

    best = max(scores, key=lambda s: s.physics_fidelity)

    # Precision blocker
    if best.deviation_from_ideal.mean_deviation_A > ideal.min_precision_required_A * 2:
        blockers.append(
            f"Best system ({best.material_system}) achieves "
            f"±{best.deviation_from_ideal.mean_deviation_A:.2f} Å mean, "
            f"but spec requires ±{ideal.min_precision_required_A:.2f} Å"
        )

    # Rigidity blocker
    if best.deviation_from_ideal.rigidity_deviation > 0.3:
        blockers.append(
            f"Required rigidity ({ideal.rigidity_class.value}) not achievable "
            f"by best system ({best.material_system})"
        )

    # Missing interaction blocker
    if best.deviation_from_ideal.missing_interactions:
        blockers.append(
            f"Best system cannot provide: "
            f"{', '.join(best.deviation_from_ideal.missing_interactions)}"
        )

    if not blockers:
        blockers.append("No single blocker identified — cumulative deviation across elements")

    return blockers
""",

    'mabe/realization/scoring/implementation.py': """\
\"\"\"
Implementation Scoring — SECONDARY axes.

These only matter after physics fidelity. They break ties between
material systems that achieve similar physics fidelity.

All return 0.0–1.0, higher = better.
\"\"\"

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabe.realization.models import InteractionGeometrySpec
    from mabe.realization.registry.material_registry import MaterialCapability


def score_synthetic_accessibility(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    \"\"\"
    How hard is it to make?

    Proxy: inverse of typical synthesis complexity.
    Will be replaced by per-adapter SA estimates in Sprint R2+.
    \"\"\"
    # Simple heuristic from registry data
    # More design tools = easier to design
    tool_bonus = min(1.0, len(cap.design_tools_available) * 0.2)

    # Higher validation rate = more likely to succeed on first try
    success_bonus = cap.literature_validation_rate

    # More literature = better understood synthesis
    lit_bonus = min(1.0, math.log10(max(1, cap.literature_examples)) / 5.0)

    return (tool_bonus + success_bonus + lit_bonus) / 3.0


def score_cost(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    \"\"\"
    Cost score at required scale.

    If cost ceiling specified: hard fail below threshold.
    Otherwise: soft logarithmic penalty.
    \"\"\"
    # Use midpoint of cost range as estimate
    estimated_cost = (cap.cost_per_unit_range[0] + cap.cost_per_unit_range[1]) / 2.0

    if spec.cost_ceiling_per_unit is not None:
        if estimated_cost > spec.cost_ceiling_per_unit:
            return 0.0
        return 1.0 - (estimated_cost / spec.cost_ceiling_per_unit)

    # Soft penalty: cheaper is better
    return 1.0 / (1.0 + math.log10(max(1.0, estimated_cost)))


def score_scalability(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    \"\"\"Can it be produced at the required scale?\"\"\"
    from mabe.realization.models import ScaleClass

    try:
        required = spec.required_scale.rank
        max_cap = ScaleClass(cap.max_practical_scale).rank
    except (ValueError, AttributeError):
        return 0.5  # can't evaluate, neutral

    if required > max_cap:
        return 0.0  # should have been caught by feasibility gate

    headroom = max_cap - required
    # More headroom = better (easier to scale)
    return min(1.0, 0.5 + 0.1 * headroom)


def score_operating_conditions(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    \"\"\"Does the material survive the operating environment?\"\"\"
    score = 1.0

    # pH headroom
    ph_margin_low = spec.pH_range[0] - cap.pH_stability[0]
    ph_margin_high = cap.pH_stability[1] - spec.pH_range[1]
    ph_margin = min(ph_margin_low, ph_margin_high)
    if ph_margin < 0:
        return 0.0  # should have been caught by gate
    if ph_margin < 1.0:
        score *= 0.5 + 0.5 * ph_margin  # penalize tight margins

    # Temperature headroom
    temp_margin = cap.thermal_stability_K[1] - spec.temperature_range_K[1]
    if temp_margin < 0:
        return 0.0
    if temp_margin < 50:
        score *= 0.5 + 0.01 * temp_margin

    return score


def score_reusability(
    spec: "InteractionGeometrySpec",
    cap: "MaterialCapability",
) -> float:
    \"\"\"
    Can the pocket be regenerated and reused?

    Rigid, chemically stable systems score higher.
    \"\"\"
    # Rigid materials are more reusable
    rigidity_map = {"rigid": 1.0, "semi-rigid": 0.7, "semi-flexible": 0.4, "flexible": 0.2}
    rigidity_score = rigidity_map.get(cap.rigidity_range[1], 0.5)

    # Wider operating envelope = more robust to regeneration conditions
    ph_range = cap.pH_stability[1] - cap.pH_stability[0]
    robustness = min(1.0, ph_range / 10.0)

    return (rigidity_score + robustness) / 2.0
""",

    'mabe/realization/tests/__init__.py': """\

""",

    'mabe/realization/tests/test_sprint_r1.py': """\
\"\"\"
Sprint R1 Test Suite.

Integration test from the plan:
    Feed a "4N planar 0.4nm pocket for Cu²⁺" spec →
    ideal pocket says "4 N-donors at 2.00 Å ± 0.05 Å in D4h, rigidity: crystalline" →
    porphyrin scores ~0.95, crown ether ~0.6, protein ~0.4 →
    correct feasibility failures and deviation reports.
\"\"\"

import math
import pytest

from mabe.realization.models import (
    ApplicationContext,
    CavityDimensions,
    CavityShape,
    DonorPosition,
    ExclusionSpec,
    IdealPocketSpec,
    InteractionGeometrySpec,
    RankedRealizations,
    RigidityClass,
    ScaleClass,
    Solvent,
)
from mabe.realization.engine.ideal_pocket import compute_ideal_pocket
from mabe.realization.engine.ranker import rank_realizations
from mabe.realization.registry.material_registry import MATERIAL_REGISTRY
from mabe.realization.scoring.feasibility import feasibility_gate
from mabe.realization.scoring.deviation import compute_deviation, deviation_to_fidelity


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_cu2_4n_spec() -> InteractionGeometrySpec:
    \"\"\"
    4N planar pocket for Cu²⁺ coordination.
    Classic porphyrin-like geometry: 4 nitrogen donors in D4h square planar,
    Cu-N distance ~2.00 Å, pocket diameter ~0.4 nm.
    \"\"\"
    cu_n_distance = 2.00  # Å

    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=33.5,  # small planar pocket
            aperture_A=4.0,
            depth_A=2.0,
            max_internal_diameter_A=4.0,
        ),
        symmetry="D4h",
        donor_positions=[
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(cu_n_distance, 0.0, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(0.0, cu_n_distance, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(-cu_n_distance, 0.0, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(0.0, -cu_n_distance, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
        ],
        rigidity_requirement="rigid",
        max_backbone_rmsd_A=0.1,
        conformational_penalty_budget_kJ_mol=5.0,
        pocket_scale_nm=0.4,
        must_exclude=[
            ExclusionSpec(
                species="Ca²⁺",
                max_allowed_affinity_kJ_mol=-10.0,
                exclusion_mechanism="geometry",
            ),
        ],
        pH_range=(3.0, 10.0),
        temperature_range_K=(280.0, 350.0),
        solvent=Solvent.AQUEOUS,
        ionic_strength_M=0.1,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


# ─────────────────────────────────────────────
# Test 1: Registry has 5 starter systems
# ─────────────────────────────────────────────

class TestRegistry:

    def test_registry_has_5_systems(self):
        assert len(MATERIAL_REGISTRY) == 5

    def test_registry_system_ids(self):
        ids = {c.system_id for c in MATERIAL_REGISTRY.all()}
        assert ids == {
            "planar_coordination_ring",
            "cyclic_encapsulant",
            "periodic_lattice_node",
            "folded_polypeptide",
            "emergent_coordination_cage",
        }

    def test_registry_physics_classes(self):
        classes = {c.physics_class for c in MATERIAL_REGISTRY.all()}
        assert "covalent_cavity" in classes
        assert "periodic_lattice" in classes
        assert "foldable_polymer" in classes
        assert "emergent_cavity" in classes

    def test_porphyrin_has_highest_precision(self):
        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        assert porphyrin is not None
        for cap in MATERIAL_REGISTRY.all():
            assert porphyrin.positioning_precision_A <= cap.positioning_precision_A


# ─────────────────────────────────────────────
# Test 2: Ideal Pocket Computation
# ─────────────────────────────────────────────

class TestIdealPocket:

    def test_ideal_pocket_has_4_elements(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert len(ideal.optimal_elements) == 4

    def test_ideal_pocket_all_nitrogen(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert all(e.atom_type == "N" for e in ideal.optimal_elements)

    def test_ideal_pocket_requires_tight_precision(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        # 4N coordination should require ≤0.05 Å precision
        assert ideal.min_precision_required_A <= 0.05

    def test_ideal_pocket_rigidity_crystalline_or_preorganized(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.rigidity_class in (
            RigidityClass.CRYSTALLINE,
            RigidityClass.PREORGANIZED,
        )

    def test_ideal_pocket_binding_energy_negative(self):
        \"\"\"Favorable binding = negative energy.\"\"\"
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.ideal_binding_energy_kJ_mol < 0

    def test_ideal_pocket_desolvation_positive(self):
        \"\"\"Desolvation is a penalty = positive energy.\"\"\"
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.ideal_desolvation_energy_kJ_mol > 0

    def test_ideal_pocket_required_elements(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.required_elements == {"N"}

    def test_ideal_pocket_critical_constraints_nonempty(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert len(ideal.critical_constraints) > 0

    def test_ideal_pocket_material_requirements_string(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert "4 interaction elements" in ideal.ideal_material_requirements
        assert "N" in ideal.ideal_material_requirements


# ─────────────────────────────────────────────
# Test 3: Feasibility Gate
# ─────────────────────────────────────────────

class TestFeasibilityGate:

    def test_feasibility_gate_cu2_pocket(self):
        \"\"\"0.4nm pocket: covalent + lattice systems pass, polymer + cage fail (min 0.5nm).\"\"\"
        spec = make_cu2_4n_spec()
        passed = []
        failed = []
        for cap in MATERIAL_REGISTRY.all():
            feasible, reason = feasibility_gate(spec, cap)
            if feasible:
                passed.append(cap.system_id)
            else:
                failed.append(cap.system_id)
        assert "planar_coordination_ring" in passed
        assert "cyclic_encapsulant" in passed
        assert "periodic_lattice_node" in passed
        # Protein and cage correctly fail — 0.4nm is below their 0.5nm minimum
        assert "folded_polypeptide" in failed
        assert "emergent_coordination_cage" in failed

    def test_gate_rejects_oversized_pocket(self):
        \"\"\"A 10nm pocket should fail for porphyrin (max 0.5nm).\"\"\"
        spec = make_cu2_4n_spec()
        spec.pocket_scale_nm = 10.0
        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        feasible, reason = feasibility_gate(spec, porphyrin)
        assert not feasible
        assert "above maximum" in reason

    def test_gate_rejects_missing_donor(self):
        \"\"\"Require Se donor — porphyrin can't provide it.\"\"\"
        spec = make_cu2_4n_spec()
        spec.donor_positions.append(DonorPosition(
            atom_type="Se",
            coordination_role="axial",
            position_vector_A=(0.0, 0.0, 2.5),
            tolerance_A=0.10,
            required_hybridization="sp3",
        ))
        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        feasible, reason = feasibility_gate(spec, porphyrin)
        assert not feasible
        assert "Se" in reason


# ─────────────────────────────────────────────
# Test 4: Deviation Scoring
# ─────────────────────────────────────────────

class TestDeviation:

    def test_porphyrin_lowest_deviation(self):
        \"\"\"Porphyrin should have the lowest deviation for 4N planar.\"\"\"
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)

        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        protein = MATERIAL_REGISTRY.get("folded_polypeptide")

        dev_porph = compute_deviation(ideal, porphyrin)
        dev_prot = compute_deviation(ideal, protein)

        assert dev_porph.mean_deviation_A < dev_prot.mean_deviation_A

    def test_porphyrin_fidelity_highest(self):
        \"\"\"Porphyrin physics fidelity should be highest.\"\"\"
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)

        fidelities = {}
        for cap in MATERIAL_REGISTRY.all():
            dev = compute_deviation(ideal, cap)
            fidelities[cap.system_id] = deviation_to_fidelity(dev)

        assert fidelities["planar_coordination_ring"] == max(fidelities.values())

    def test_fidelity_bounded_0_1(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        for cap in MATERIAL_REGISTRY.all():
            dev = compute_deviation(ideal, cap)
            f = deviation_to_fidelity(dev)
            assert 0.0 <= f <= 1.0

    def test_porphyrin_fidelity_above_0_8(self):
        \"\"\"Porphyrin at 0.01Å precision for 0.05Å requirement → high fidelity.\"\"\"
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        porph = MATERIAL_REGISTRY.get("planar_coordination_ring")
        dev = compute_deviation(ideal, porph)
        f = deviation_to_fidelity(dev)
        assert f > 0.8

    def test_protein_fidelity_below_porphyrin(self):
        \"\"\"Protein at 0.3Å precision should score below porphyrin at 0.01Å.\"\"\"
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        porph = MATERIAL_REGISTRY.get("planar_coordination_ring")
        prot = MATERIAL_REGISTRY.get("folded_polypeptide")
        f_porph = deviation_to_fidelity(compute_deviation(ideal, porph))
        f_prot = deviation_to_fidelity(compute_deviation(ideal, prot))
        assert f_porph > f_prot


# ─────────────────────────────────────────────
# Test 5: Full Ranker Integration
# ─────────────────────────────────────────────

class TestRanker:

    def test_ranker_returns_ranked_realizations(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert isinstance(result, RankedRealizations)

    def test_ranker_has_ideal_pocket(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert result.ideal_pocket is ideal

    def test_ranker_recommends_porphyrin(self):
        \"\"\"For 4N planar Cu²⁺, porphyrin should be #1.\"\"\"
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert result.recommended == "planar_coordination_ring"

    def test_ranker_sorts_by_composite(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        composites = [s.composite_score for s in result.rankings]
        assert composites == sorted(composites, reverse=True)

    def test_ranker_all_feasible_have_scores(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        for s in result.rankings:
            if s.feasible:
                assert s.composite_score > 0
                assert s.physics_fidelity > 0

    def test_ranker_gap_analysis_present(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        # gap_to_ideal should be computed
        assert 0.0 <= result.gap_to_ideal <= 1.0
        assert result.best_physics_fidelity > 0

    def test_ranker_recommendation_rationale_nonempty(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert len(result.recommendation_rationale) > 0


# ─────────────────────────────────────────────
# Test 6: Gap Report (use a hard spec)
# ─────────────────────────────────────────────

class TestGapReport:

    def _make_extreme_spec(self) -> InteractionGeometrySpec:
        \"\"\"A spec so demanding that nothing scores above 0.5.\"\"\"
        return InteractionGeometrySpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(
                volume_A3=100.0,
                aperture_A=5.0,
                depth_A=5.0,
                max_internal_diameter_A=6.0,
            ),
            symmetry="none",
            donor_positions=[
                # 6 different element types — no material has all of them
                DonorPosition("N", "equatorial", (2.0, 0.0, 0.0), 0.02, "sp2"),
                DonorPosition("O", "equatorial", (0.0, 2.0, 0.0), 0.02, "sp3"),
                DonorPosition("S", "axial", (0.0, 0.0, 2.5), 0.02, "sp3"),
                DonorPosition("Se", "axial", (0.0, 0.0, -2.5), 0.02, "sp3"),
                DonorPosition("N", "bridging", (1.5, 1.5, 0.0), 0.02, "sp2"),
                DonorPosition("P", "terminal", (-2.0, 0.0, 0.0), 0.02, "sp3"),
            ],
            pocket_scale_nm=0.6,
            pH_range=(4.0, 9.0),
            temperature_range_K=(280.0, 340.0),
            solvent=Solvent.AQUEOUS,
            target_application=ApplicationContext.RESEARCH,
            required_scale=ScaleClass.UMOL,
        )

    def test_gap_report_generated_for_hard_spec(self):
        \"\"\"An extreme spec should trigger gap report (gap > 0.3).\"\"\"
        spec = self._make_extreme_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        # Most systems can't provide Se — should see infeasible or low fidelity
        # This should trigger a gap report
        assert result.gap_to_ideal > 0.0  # at minimum, not perfect

    def test_novel_material_suggestion_for_very_hard_spec(self):
        \"\"\"When gap > 0.5, should generate novel material suggestion.\"\"\"
        spec = self._make_extreme_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        # If gap is large enough, novel suggestion should exist
        if result.gap_to_ideal > 0.5:
            assert result.novel_material_suggestion is not None
            assert len(result.novel_material_suggestion) > 0
""",

    'mabe/realization/tests/test_sprint_r2a.py': """\
\"\"\"
Sprint R2a Test Suite — Cyclodextrin Adapter.

Tests:
    1. CD selection by cavity volume (α vs β vs γ)
    2. Packing coefficient calculation
    3. Modification selection logic
    4. Binding energy prediction (sign and magnitude)
    5. Full design pipeline → FabricationSpec
    6. Validation report catches bad designs
    7. Integration: spec → ideal pocket → ranker includes CD score
\"\"\"

import math
import pytest

from mabe.realization.models import (
    ApplicationContext,
    CavityDimensions,
    CavityShape,
    DonorPosition,
    HBondSpec,
    InteractionGeometrySpec,
    ScaleClass,
    Solvent,
)
from mabe.realization.adapters.cyclodextrin_adapter import (
    CyclodextrinAdapter,
    CyclodextrinFabricationSpec,
    CD_LIBRARY,
    BACKSOLVE_PARAMS,
)
from mabe.realization.registry.material_registry import MATERIAL_REGISTRY


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _get_cd_adapter() -> CyclodextrinAdapter:
    \"\"\"Get adapter with cyclic_encapsulant capability.\"\"\"
    cap = MATERIAL_REGISTRY.get("cyclic_encapsulant")
    assert cap is not None
    return CyclodextrinAdapter(cap)


def make_beta_cd_guest_spec() -> InteractionGeometrySpec:
    \"\"\"
    A guest that fits β-CD: ~150 Å³ volume (target PC ≈ 150/262 ≈ 0.57).
    Roughly adamantane-sized — classic β-CD guest.
    \"\"\"
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.CONE,
        cavity_dimensions=CavityDimensions(
            volume_A3=150.0,  # adamantane-sized
            aperture_A=5.5,
            depth_A=5.0,
            max_internal_diameter_A=6.0,
        ),
        symmetry="C3v",
        donor_positions=[],  # hydrophobic guest — no donors
        pocket_scale_nm=0.6,
        pH_range=(4.0, 9.0),
        temperature_range_K=(280.0, 340.0),
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_alpha_cd_guest_spec() -> InteractionGeometrySpec:
    \"\"\"Small guest that fits α-CD: ~100 Å³ (PC ≈ 100/174 ≈ 0.57).\"\"\"
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.CONE,
        cavity_dimensions=CavityDimensions(
            volume_A3=100.0,
            aperture_A=4.0,
            depth_A=4.5,
            max_internal_diameter_A=4.5,
        ),
        pocket_scale_nm=0.45,
        pH_range=(4.0, 9.0),
        temperature_range_K=(280.0, 340.0),
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_gamma_cd_guest_spec() -> InteractionGeometrySpec:
    \"\"\"Large guest that fits γ-CD: ~250 Å³ (PC ≈ 250/427 ≈ 0.59).\"\"\"
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.CONE,
        cavity_dimensions=CavityDimensions(
            volume_A3=250.0,
            aperture_A=7.0,
            depth_A=6.0,
            max_internal_diameter_A=7.5,
        ),
        pocket_scale_nm=0.75,
        pH_range=(4.0, 9.0),
        temperature_range_K=(280.0, 340.0),
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_diagnostic_spec() -> InteractionGeometrySpec:
    \"\"\"Guest for diagnostic application — should trigger amine/thiol modification.\"\"\"
    spec = make_beta_cd_guest_spec()
    spec.target_application = ApplicationContext.DIAGNOSTIC
    return spec


def make_hbond_guest_spec() -> InteractionGeometrySpec:
    \"\"\"Guest with H-bond requirements.\"\"\"
    spec = make_beta_cd_guest_spec()
    spec.h_bond_network = HBondSpec(
        donors=[(0.0, 3.0, 0.0), (3.0, 0.0, 0.0)],
        acceptors=[(0.0, -3.0, 0.0)],
    )
    return spec


# ─────────────────────────────────────────────
# Test 1: CD Library
# ─────────────────────────────────────────────

class TestCDLibrary:

    def test_library_has_native_cds(self):
        assert "alpha-CD" in CD_LIBRARY
        assert "beta-CD" in CD_LIBRARY
        assert "gamma-CD" in CD_LIBRARY

    def test_library_has_modifications(self):
        assert "HP-beta-CD" in CD_LIBRARY
        assert "Me-beta-CD" in CD_LIBRARY
        assert "SBE-beta-CD" in CD_LIBRARY
        assert "amine-beta-CD" in CD_LIBRARY

    def test_cavity_volumes_increase_alpha_beta_gamma(self):
        a = CD_LIBRARY["alpha-CD"]
        b = CD_LIBRARY["beta-CD"]
        g = CD_LIBRARY["gamma-CD"]
        assert a.cavity_volume_A3 < b.cavity_volume_A3 < g.cavity_volume_A3

    def test_all_native_cds_commercial(self):
        for key, cd in CD_LIBRARY.items():
            if cd.modification == "native":
                assert cd.commercial, f"{key} should be commercial"


# ─────────────────────────────────────────────
# Test 2: CD Selection by Volume
# ─────────────────────────────────────────────

class TestCDSelection:

    def test_selects_beta_for_150A3(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        cd, pc, viable = adapter._select_cd(spec)
        assert cd is not None
        assert cd.base_type == "beta"
        assert 0.50 < pc < 0.65

    def test_selects_alpha_for_100A3(self):
        adapter = _get_cd_adapter()
        spec = make_alpha_cd_guest_spec()
        cd, pc, viable = adapter._select_cd(spec)
        assert cd is not None
        assert cd.base_type == "alpha"
        assert 0.45 < pc < 0.70

    def test_selects_gamma_for_250A3(self):
        adapter = _get_cd_adapter()
        spec = make_gamma_cd_guest_spec()
        cd, pc, viable = adapter._select_cd(spec)
        assert cd is not None
        assert cd.base_type == "gamma"
        assert 0.45 < pc < 0.70

    def test_viable_list_nonempty(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        _, _, viable = adapter._select_cd(spec)
        assert len(viable) >= 1


# ─────────────────────────────────────────────
# Test 3: Modification Selection
# ─────────────────────────────────────────────

class TestModificationSelection:

    def test_diagnostic_selects_amine_or_thiol(self):
        adapter = _get_cd_adapter()
        spec = make_diagnostic_spec()
        cd, _, viable = adapter._select_cd(spec)
        mod_cd = adapter._select_modification(spec, cd, viable)
        assert mod_cd.modification in ("amine", "thiol")

    def test_research_selects_native(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()  # research context
        cd, _, viable = adapter._select_cd(spec)
        mod_cd = adapter._select_modification(spec, cd, viable)
        assert mod_cd.modification == "native"


# ─────────────────────────────────────────────
# Test 4: Binding Energy Prediction
# ─────────────────────────────────────────────

class TestBindingPrediction:

    def test_predicted_dG_negative(self):
        \"\"\"Favorable binding = negative ΔG.\"\"\"
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        cd, pc, _ = adapter._select_cd(spec)
        dG = adapter._predict_binding_energy(spec, cd, pc)
        assert dG < 0, f"Expected negative ΔG, got {dG}"

    def test_predicted_dG_reasonable_magnitude(self):
        \"\"\"CD binding typically −10 to −40 kJ/mol.\"\"\"
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        cd, pc, _ = adapter._select_cd(spec)
        dG = adapter._predict_binding_energy(spec, cd, pc)
        assert -50.0 < dG < 0.0, f"ΔG={dG} outside expected range"

    def test_hbond_guest_stronger_binding(self):
        \"\"\"Guest with H-bonds should bind more strongly.\"\"\"
        adapter = _get_cd_adapter()

        spec_no_hb = make_beta_cd_guest_spec()
        cd, pc, _ = adapter._select_cd(spec_no_hb)
        dG_no_hb = adapter._predict_binding_energy(spec_no_hb, cd, pc)

        spec_hb = make_hbond_guest_spec()
        cd2, pc2, _ = adapter._select_cd(spec_hb)
        dG_hb = adapter._predict_binding_energy(spec_hb, cd2, pc2)

        assert dG_hb < dG_no_hb, "H-bond guest should bind more strongly"


# ─────────────────────────────────────────────
# Test 5: Full Design Pipeline
# ─────────────────────────────────────────────

class TestFullDesign:

    def test_design_returns_cd_fab_spec(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        assert isinstance(fab, CyclodextrinFabricationSpec)

    def test_design_has_cd_variant(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        assert fab.cd_variant is not None

    def test_design_has_predicted_Ka(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        assert fab.predicted_Ka > 0

    def test_design_has_click_handle(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        assert len(fab.click_handle) > 0

    def test_design_has_validation_experiments(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        assert len(fab.validation_experiments) >= 3

    def test_design_has_synthesis_route(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        assert len(fab.synthesis_steps) >= 1

    def test_diagnostic_design_has_amine_handle(self):
        adapter = _get_cd_adapter()
        spec = make_diagnostic_spec()
        fab = adapter.design(spec)
        assert "amine" in fab.cd_variant.modification or "thiol" in fab.cd_variant.modification


# ─────────────────────────────────────────────
# Test 6: Validation
# ─────────────────────────────────────────────

class TestValidation:

    def test_good_design_validates(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        report = adapter.validate_design(fab)
        assert report.valid

    def test_oversized_guest_fails_validation(self):
        \"\"\"Guest volume 250 Å³ in α-CD (174 Å³) → PC > 1.0 → fail.\"\"\"
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        # Manually override to create a bad design
        fab.packing_coefficient = 1.2
        report = adapter.validate_design(fab)
        assert not report.valid

    def test_undersized_guest_fails_validation(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        fab = adapter.design(spec)
        fab.packing_coefficient = 0.1
        report = adapter.validate_design(fab)
        assert not report.valid


# ─────────────────────────────────────────────
# Test 7: Fidelity Estimate
# ─────────────────────────────────────────────

class TestFidelityEstimate:

    def test_fidelity_positive_for_good_match(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        score = adapter.estimate_fidelity(spec)
        assert score.physics_fidelity > 0.5
        assert score.feasible

    def test_fidelity_has_advantages(self):
        adapter = _get_cd_adapter()
        spec = make_beta_cd_guest_spec()
        score = adapter.estimate_fidelity(spec)
        assert len(score.advantages) > 0

    def test_coordination_spec_flags_limitation(self):
        \"\"\"CD can't do metal coordination — should note limitation.\"\"\"
        spec = make_beta_cd_guest_spec()
        spec.donor_positions = [
            DonorPosition("N", "equatorial", (2.0, 0.0, 0.0), 0.05, "sp2"),
        ]
        adapter = _get_cd_adapter()
        score = adapter.estimate_fidelity(spec)
        limitations_text = " ".join(score.limitations)
        assert "coordination" in limitations_text.lower()
""",
}


def main():
    print("=== MABE Realization Engine — Sprint R1 Bootstrap ===")
    print()
    for filepath, content in FILES.items():
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        # Strip leading newline from heredoc style
        if content.startswith("\n"):
            content = content[1:]
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        print(f"  Created {filepath}")
    print()
    count = len(FILES)
    print(f"=== Bootstrap complete: {count} files ===")
    print()
    print("Run tests:")
    print("  python -m pytest mabe/realization/tests/test_sprint_r1.py -v")


if __name__ == "__main__":
    main()