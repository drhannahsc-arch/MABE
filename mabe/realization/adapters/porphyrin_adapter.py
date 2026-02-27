"""
Porphyrin / Phthalocyanine Adapter — Layer 4 Implementation.

Takes an InteractionGeometrySpec describing a planar 4N coordination pocket
and designs a metalloporphyrin realization.

Design logic:
    1. Identify target metal from spec (4N planar, ionic radius)
    2. Select porphyrin core (TPP, OEP, TPFPP, Pc, etc.)
    3. Choose meso-substituents for electronics + conjugation
    4. Add axial ligands if spec requires 5th/6th coordination
    5. Produce PorphyrinFabSpec with full design + synthesis route

Physics connection:
    - M-N bond distance → BackSolve metal coordination (R²=0.908)
    - LFSE → d-electron configuration
    - Size-match → core hole radius vs ionic radius
    - Hammett σ → substituent electronic tuning
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.adapters.base import RealizationAdapter, ValidationReport
from mabe.realization.adapters.porphyrin_knowledge import (
    ALL_CORES,
    AXIAL_LIGANDS,
    BACKSOLVE_METAL_PARAMS,
    MESO_SUBSTITUENTS,
    METAL_PORPH_DB,
    AxialLigand,
    MesoSubstituent,
    MetalPorphyrinEntry,
    PorphyrinCore,
    metal_porphyrin_size_match,
    predict_metalation_dG,
    select_core_for_metal,
)
from mabe.realization.models import (
    ApplicationContext,
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
# PorphyrinFabSpec
# ─────────────────────────────────────────────

@dataclass
class PorphyrinFabSpec(FabricationSpec):
    """Fabrication spec for a metalloporphyrin pocket."""

    # ── Core selection ──
    selected_core: str = ""
    core_abbreviation: str = ""
    core_type: str = ""
    core_hole_radius_A: float = 0.0

    # ── Metal ──
    target_metal: str = ""
    metal_ionic_radius_A: float = 0.0
    metal_coordination: int = 0
    metal_spin_state: str = ""

    # ── Size match ──
    size_match_score: float = 0.0
    mn_bond_predicted_A: float = 0.0
    in_plane: bool = True

    # ── Substituents ──
    meso_substituent: str = ""
    meso_rationale: str = ""
    hammett_sigma_sum: float = 0.0

    # ── Axial ligands ──
    axial_ligands: list[str] = field(default_factory=list)
    axial_rationale: str = ""

    # ── Binding prediction ──
    predicted_dG_metalation_kJ_mol: float = 0.0
    predicted_logK_metalation: float = 0.0
    kinetic_class: str = ""

    # ── Synthesis ──
    synthesis_route: str = ""
    metalation_protocol: str = ""

    # ── Sourcing ──
    supplier: str = ""
    catalog_note: str = ""


# ─────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────

class PorphyrinAdapter(RealizationAdapter):
    """
    Designs metalloporphyrin binding pockets.

    Class A (Covalent Cavity / Planar Coordination Ring):
        - 4N planar pocket, D4h symmetry
        - Core hole ~2.01 Å (porphyrin) or ~1.92 Å (Pc)
        - Design freedom: meso-substituents, axial ligands, core type
        - Highest positioning precision (0.01 Å)
        - Limited to metals that fit the ~2 Å core hole
    """

    def __init__(self, capability: Optional[MaterialCapability] = None):
        if capability is None:
            from mabe.realization.registry.material_registry import MATERIAL_REGISTRY
            capability = MATERIAL_REGISTRY.get("planar_coordination_ring")
            if capability is None:
                capability = _make_porph_capability()
        super().__init__(capability)

    def estimate_fidelity(
        self,
        spec: InteractionGeometrySpec,
    ) -> RealizationScore:
        """
        Quick score: does this spec match a 4N planar coordination pocket?
        """

        metal_symbol = self._identify_target_metal(spec)
        metal = METAL_PORPH_DB.get(metal_symbol)

        if metal is None:
            return self._infeasible_score(
                f"Metal {metal_symbol} not in porphyrin database"
            )

        # ── Size match across all cores ──
        ranked = select_core_for_metal(metal_symbol)
        if not ranked:
            return self._infeasible_score("No suitable core")

        best_core, sm, _rationale = ranked[0]

        # ── Physics fidelity ──
        physics_fidelity = sm * 0.5

        # Geometry match: spec must be planar 4N
        n_N_donors = sum(
            1 for d in spec.donor_positions if d.atom_type == "N"
        )
        if n_N_donors >= 4:
            physics_fidelity += 0.3
        elif n_N_donors >= 2:
            physics_fidelity += 0.1

        # Symmetry bonus
        if spec.cavity_shape == CavityShape.FLAT:
            physics_fidelity += 0.15
        elif spec.cavity_shape == CavityShape.SPHERE:
            physics_fidelity += 0.05

        physics_fidelity = max(0.0, min(1.0, physics_fidelity))

        # ── Deviation ──
        mn_dev = abs(metal.typical_MN_bond_A - best_core.ideal_MN_bond_A)
        deviation = DeviationReport(
            material_system="porphyrin",
            element_deviations_A=[mn_dev] * 4,
            max_deviation_A=mn_dev,
            mean_deviation_A=mn_dev,
        )

        # ── Advantages / limitations ──
        advantages = []
        limitations = []

        advantages.append(f"Highest positioning precision (0.01 Å)")
        if best_core.commercial:
            advantages.append(
                f"{best_core.abbreviation} commercial "
                f"(${best_core.cost_per_gram_usd}/g)"
            )
        if metal.logK_metalation > 15:
            advantages.append(
                f"Strong metalation (logK={metal.logK_metalation:.0f})"
            )
        if sm > 0.8:
            advantages.append("Excellent metal-cavity size match")

        if not metal.in_plane:
            limitations.append(
                f"{metal.metal} sits {metal.displacement_A:.2f} Å out-of-plane"
            )
        if metal.kinetic_class == "labile":
            limitations.append("Labile metalation — demetalation risk in acid")
        if metal.ionic_radius_A > 0.80:
            limitations.append("Large ion — significant ring distortion expected")

        return RealizationScore(
            material_system="porphyrin",
            adapter_id="PorphyrinAdapter",
            deviation_from_ideal=deviation,
            physics_fidelity=physics_fidelity,
            synthetic_accessibility=0.80,
            cost_score=0.70,
            scalability=0.80,
            operating_condition_compatibility=0.85,
            reusability_score=0.7,
            composite_score=0.0,
            confidence=0.85,
            advantages=advantages,
            limitations=limitations,
            feasible=True,
        )

    def design(
        self,
        spec: InteractionGeometrySpec,
    ) -> PorphyrinFabSpec:
        """
        Full metalloporphyrin design.

        1. Identify target metal
        2. Select porphyrin core by size match
        3. Choose meso-substituent (electronics + function)
        4. Add axial ligands if needed
        5. Predict metalation affinity
        """

        spec_hash = hashlib.md5(str(spec).encode()).hexdigest()[:12]
        metal_symbol = self._identify_target_metal(spec)
        metal = METAL_PORPH_DB.get(metal_symbol)

        if metal is None:
            return self._empty_fab(spec_hash, f"Metal {metal_symbol} unknown")

        # ── Step 1: Core selection ──
        ranked = select_core_for_metal(metal_symbol)
        if not ranked:
            return self._empty_fab(spec_hash, "No suitable core")

        best_core, sm, _core_rationale = ranked[0]

        # ── Step 2: Meso-substituent selection ──
        meso, meso_rationale = self._select_meso(metal, spec)

        # ── Step 3: Axial ligand selection ──
        axial_list, axial_rationale = self._select_axial(metal, spec)

        # ── Step 4: Predict metalation ──
        hammett_sum = meso.hammett_sigma * 4  # 4 meso positions
        dG_met = predict_metalation_dG(
            metal, best_core,
            meso_sigma=hammett_sum,
            n_axial=len(axial_list),
        )
        RT = 8.314e-3 * 298.15
        logK_met = -dG_met / (2.303 * RT)

        # ── Step 5: Synthesis route ──
        steps = [
            f"Synthesize {best_core.abbreviation} via {best_core.synthesis_route}",
        ]
        if best_core.commercial:
            steps[0] = f"Procure {best_core.abbreviation} from {best_core.common_suppliers[0]}"

        # Metalation protocol
        if metal.kinetic_class == "labile":
            metalation = (
                f"Dissolve {best_core.abbreviation} in DMF, add {metal.metal} "
                f"acetate (5 eq), stir 60°C, 2h. Monitor by UV-vis (Soret shift)."
            )
        elif metal.kinetic_class == "intermediate":
            metalation = (
                f"Reflux {best_core.abbreviation} with {metal.metal} chloride "
                f"(10 eq) in DMF, 4-12h. Column purification (silica, CHCl3)."
            )
        else:  # inert
            metalation = (
                f"Reflux {best_core.abbreviation} with {metal.metal} salt "
                f"in glacial AcOH, 12-24h. Extended reaction time for inert metal."
            )

        steps.append(metalation)
        steps.append(
            "Purify: column chromatography (silica gel, CHCl3/MeOH)"
        )
        steps.append(
            "Characterize: UV-vis (Soret + Q bands), MALDI-TOF, 1H NMR"
        )

        if axial_list:
            axial_names = ", ".join(a.name for a in axial_list)
            steps.append(
                f"Axial coordination: add {axial_names} "
                f"(stoichiometric) in solution"
            )

        # ── Step 6: Validation plan ──
        validation = [
            f"UV-vis: Soret band shift confirms metalation ({metal.metal})",
            "MALDI-TOF: confirm M-porphyrin MW",
        ]
        if axial_list:
            validation.append(
                "EPR/NMR: confirm axial coordination geometry"
            )
        if spec.target_application in (
            ApplicationContext.DIAGNOSTIC,
            ApplicationContext.SEPARATION,
        ):
            validation.append(
                f"ITC: binding affinity for {metal.metal} in target matrix"
            )

        expected = {
            "logK_metalation": logK_met,
            "dG_kJ_mol": dG_met,
            "soret_band_nm": 420 if best_core.core_type == "porphyrin" else 680,
            "coordination_number": metal.preferred_coordination,
            "spin_state": metal.spin_state,
        }

        return PorphyrinFabSpec(
            material_system="porphyrin",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=CavityDimensions(
                volume_A3=4/3 * math.pi * best_core.core_hole_radius_A ** 3,
                aperture_A=best_core.core_hole_radius_A * 2,
                depth_A=3.4 if not axial_list else 6.0,
                max_internal_diameter_A=best_core.core_hole_radius_A * 2,
            ),
            predicted_deviation_from_ideal_A=abs(
                metal.typical_MN_bond_A - best_core.ideal_MN_bond_A
            ),
            synthesis_steps=steps,
            estimated_yield=0.60 if metal.kinetic_class == "labile" else 0.40,
            estimated_cost_per_unit=best_core.cost_per_gram_usd * 2.0,
            estimated_time="1 day" if metal.kinetic_class == "labile" else "2–3 days",
            validation_experiments=validation,
            expected_observables=expected,
            # Porphyrin-specific
            selected_core=best_core.name,
            core_abbreviation=best_core.abbreviation,
            core_type=best_core.core_type,
            core_hole_radius_A=best_core.core_hole_radius_A,
            target_metal=metal.metal,
            metal_ionic_radius_A=metal.ionic_radius_A,
            metal_coordination=metal.preferred_coordination,
            metal_spin_state=metal.spin_state,
            size_match_score=sm,
            mn_bond_predicted_A=metal.typical_MN_bond_A,
            in_plane=metal.in_plane,
            meso_substituent=meso.abbreviation,
            meso_rationale=meso_rationale,
            hammett_sigma_sum=hammett_sum,
            axial_ligands=[a.abbreviation for a in axial_list],
            axial_rationale=axial_rationale,
            predicted_dG_metalation_kJ_mol=dG_met,
            predicted_logK_metalation=logK_met,
            kinetic_class=metal.kinetic_class,
            synthesis_route=best_core.synthesis_route,
            metalation_protocol=metalation,
            supplier=(
                best_core.common_suppliers[0] if best_core.commercial
                else "custom synthesis"
            ),
            catalog_note=f"{best_core.abbreviation}, {best_core.cost_per_gram_usd}$/g",
        )

    def validate_design(self, fab: FabricationSpec) -> ValidationReport:
        """Check porphyrin design for internal consistency."""

        if not isinstance(fab, PorphyrinFabSpec):
            return ValidationReport(
                valid=False,
                issues=["Not a PorphyrinFabSpec"],
                warnings=[],
            )

        issues = []
        warnings = []

        # Size match
        if fab.size_match_score < 0.3:
            issues.append(
                f"Size match {fab.size_match_score:.2f} too poor — "
                "severe ring distortion expected"
            )

        # Out-of-plane displacement
        if not fab.in_plane:
            warnings.append(
                f"{fab.target_metal} sits out-of-plane — "
                "may affect selectivity"
            )

        # Axial count vs metal preference
        metal = METAL_PORPH_DB.get(fab.target_metal)
        if metal and len(fab.axial_ligands) > metal.max_axial_count:
            issues.append(
                f"{fab.target_metal} max axial={metal.max_axial_count}, "
                f"but {len(fab.axial_ligands)} specified"
            )

        # Labile metalation warning
        if fab.kinetic_class == "labile":
            warnings.append(
                "Labile metalation: risk of demetalation below pH 3"
            )

        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            confidence=0.85,
        )

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _identify_target_metal(self, spec: InteractionGeometrySpec) -> str:
        """
        Identify target metal from spec.

        Strategy:
            1. Match cavity size to known M-N bond distances
            2. Disambiguate by coordination number (count of donor positions)
        """
        spec_radius = spec.cavity_dimensions.max_internal_diameter_A / 2.0
        spec_coord = len(spec.donor_positions)

        best_match = "Cu2+"
        best_score = float("inf")

        for symbol, metal in METAL_PORPH_DB.items():
            # Distance penalty
            delta = abs(metal.typical_MN_bond_A - spec_radius)
            # Coordination mismatch penalty
            coord_penalty = abs(metal.preferred_coordination - spec_coord) * 0.1
            score = delta + coord_penalty
            if score < best_score:
                best_score = score
                best_match = symbol

        return best_match

    def _select_meso(
        self,
        metal: MetalPorphyrinEntry,
        spec: InteractionGeometrySpec,
    ) -> tuple[MesoSubstituent, str]:
        """Select meso-substituent based on metal electronics + application."""

        # Default to phenyl
        selected = MESO_SUBSTITUENTS["phenyl"]
        rationale = "Standard TPP — well-characterized, commercial"

        # Application-driven conjugation handle
        if spec.target_application in (
            ApplicationContext.DIAGNOSTIC,
            ApplicationContext.SEPARATION,
        ):
            selected = MESO_SUBSTITUENTS["4-carboxyphenyl"]
            rationale = (
                "4-COOH-Ph: conjugation handle for scaffold attachment "
                "(EDC/NHS coupling). Water-soluble."
            )
            return selected, rationale

        # Electronic tuning for electron-rich metals (d8+)
        if metal.d_electrons >= 8:
            selected = MESO_SUBSTITUENTS["pentafluorophenyl"]
            rationale = (
                f"C6F5: electron-poor meso for d{metal.d_electrons} metal. "
                "Stabilizes electron-rich center, raises redox potential."
            )
            return selected, rationale

        # Electron-rich core for early/low-d metals
        if metal.d_electrons <= 4 and metal.charge >= 3:
            selected = MESO_SUBSTITUENTS["4-aminophenyl"]
            rationale = (
                f"4-NH2-Ph: electron-rich meso for high-charge M{metal.charge}+. "
                "Accelerates metalation, lowers redox potential."
            )
            return selected, rationale

        return selected, rationale

    def _select_axial(
        self,
        metal: MetalPorphyrinEntry,
        spec: InteractionGeometrySpec,
    ) -> tuple[list[AxialLigand], str]:
        """Select axial ligands based on metal coordination preference."""

        if metal.max_axial_count == 0:
            return [], "4-coordinate metal — no axial ligands needed"

        # Check if spec has axial donor positions
        axial_donors = [
            d for d in spec.donor_positions
            if d.coordination_role == "axial"
        ]

        if not axial_donors and metal.preferred_coordination <= 4:
            return [], "Spec does not require axial coordination"

        # Select based on metal preference
        n_axial = min(
            metal.max_axial_count,
            max(1, metal.preferred_coordination - 4),
        )

        # Pick best axial ligand for this metal
        ligand_scores = []
        for lig in AXIAL_LIGANDS.values():
            score = lig.typical_logK_axial
            # Bonus for matching metal's typical axial ligands
            if lig.abbreviation in metal.axial_ligands_typical:
                score += 2.0
            # Aqueous penalty for non-labile if in water
            ligand_scores.append((lig, score))

        ligand_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [ligand_scores[0][0]] * n_axial

        lig_names = ", ".join(l.abbreviation for l in selected)
        rationale = (
            f"{metal.metal} prefers {metal.preferred_coordination}-coordinate: "
            f"{n_axial} axial {lig_names} selected "
            f"(logK_axial={selected[0].typical_logK_axial:.1f})"
        )

        return selected, rationale

    def _infeasible_score(self, reason: str) -> RealizationScore:
        return RealizationScore(
            material_system="porphyrin",
            adapter_id="PorphyrinAdapter",
            deviation_from_ideal=DeviationReport(
                material_system="porphyrin",
                element_deviations_A=[],
                max_deviation_A=float("inf"),
                mean_deviation_A=float("inf"),
            ),
            physics_fidelity=0.0,
            feasible=False,
            infeasibility_reason=reason,
        )

    def _empty_fab(self, spec_hash: str, reason: str) -> PorphyrinFabSpec:
        return PorphyrinFabSpec(
            material_system="porphyrin",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=CavityDimensions(0, 0, 0, 0),
            predicted_deviation_from_ideal_A=float("inf"),
            synthesis_steps=[f"DESIGN FAILED: {reason}"],
        )


def _make_porph_capability() -> MaterialCapability:
    return MaterialCapability(
        system_id="porphyrin",
        physics_class="covalent_cavity",
        adapter_class="PorphyrinAdapter",
        min_pocket_size_nm=0.38,
        max_pocket_size_nm=0.41,
        achievable_symmetries=["D4h", "C4v", "C2v"],
        max_donor_count=6,
        donor_types_available=["N"],
        positioning_precision_A=0.01,
        rigidity_range=("rigid", "rigid"),
        pH_stability=(1.0, 14.0),
        thermal_stability_K=(273.0, 600.0),
        solvent_compatibility=["aqueous", "organic", "mixed"],
        min_practical_scale="µmol",
        max_practical_scale="mol",
        cost_per_unit_range=(5.0, 50.0),
        typical_synthesis_time="1–3 days",
        literature_validation_rate=0.85,
        literature_examples=50000,
        design_tools_available=["RDKit"],
        known_strengths=[
            "Highest precision covalent cavity (0.01 Å)",
            "BackSolve-calibrated (R²=0.908)",
            "Rigid D4h geometry — highly preorganized",
            "Vast literature + commercial availability",
        ],
        known_limitations=[
            "4N planar only — single geometry class",
            "Large metals (Pb2+) cause severe distortion",
            "Selectivity via cavity size is limited (fixed ~2 Å hole)",
        ],
    )
