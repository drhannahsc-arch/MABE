"""
Cyclodextrin Adapter — Layer 4 Implementation.

Takes an InteractionGeometrySpec describing a hydrophobic inclusion pocket
and designs a cyclodextrin-based realization.

Design logic:
    1. Map spec cavity volume → α/β/γ-CD by packing coefficient
    2. Score each CD variant (native + modified) against spec
    3. Select modifications based on solubility, charge, conjugation needs
    4. Produce CyclodextrinFabSpec with selection rationale + sourcing

Physics connection:
    - Packing coefficient → BackSolve Phase 10 (shape complementarity)
    - Hydrophobic SASA → BackSolve Phase 6 (γ_hydrophobic)
    - H-bond count → BackSolve Phase 7 (ε_neutral, ε_charge_assisted)
    - Rotor freezing → BackSolve Phase 9 (ε_rotor, f_partial)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.adapters.base import RealizationAdapter, ValidationReport
from mabe.realization.adapters.cyclodextrin_knowledge import (
    ALL_CDS,
    BACKSOLVE_CD_PARAMS,
    CD_HOSTS,
    CD_MODIFICATIONS,
    CDHost,
    CDModification,
    NATIVE_CDS,
    select_best_cd,
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
# CyclodextrinFabSpec — CD-specific fabrication output
# ─────────────────────────────────────────────

@dataclass
class CyclodextrinFabSpec(FabricationSpec):
    """Fabrication spec for a cyclodextrin-based pocket."""

    # ── CD selection ──
    selected_cd: str = ""
    base_type: str = ""
    cavity_diameter_A: float = 0.0
    cavity_volume_A3: float = 0.0

    # ── Packing analysis ──
    packing_coefficient: float = 0.0
    packing_quality: str = ""

    # ── Modification ──
    modifications: list[str] = field(default_factory=list)
    modification_rationale: list[str] = field(default_factory=list)

    # ── Binding prediction (from BackSolve params) ──
    predicted_dG_kJ_mol: float = 0.0
    predicted_logK: float = 0.0
    prediction_confidence: str = ""

    # ── Conjugation ──
    click_handle: str = ""
    click_handle_synthesis: str = ""

    # ── Sourcing ──
    supplier: str = ""
    catalog_note: str = ""


# ─────────────────────────────────────────────
# Adapter implementation
# ─────────────────────────────────────────────

class CyclodextrinAdapter(RealizationAdapter):
    """
    Designs cyclodextrin-based binding pockets.

    CDs are Class A (Covalent Cavity) systems:
        - 0.47–0.83 nm cavity diameter (α/β/γ)
        - Truncated cone shape
        - Hydrophobic interior, hydrophilic exterior
        - Driven by hydrophobic burial + H-bonds at portals
        - Rebek packing coefficient ~0.55 is optimal
    """

    def __init__(self, capability: Optional[MaterialCapability] = None):
        if capability is None:
            from mabe.realization.registry.material_registry import MATERIAL_REGISTRY
            capability = MATERIAL_REGISTRY.get("cyclic_encapsulant")
            if capability is None:
                capability = _make_cd_capability()
        super().__init__(capability)

    def estimate_fidelity(
        self,
        spec: InteractionGeometrySpec,
    ) -> RealizationScore:
        """
        Quick score: can a CD host this guest geometry?

        Primary check: cavity volume → packing coefficient.
        Secondary: operating conditions, solvent compatibility.
        """

        # ── Packing coefficient scan across all CDs ──
        best_pc_dev = float("inf")
        best_cd = None
        best_pc = 0.0

        target_vol = spec.cavity_dimensions.volume_A3

        for cd in ALL_CDS:
            pc = target_vol / cd.cavity_volume_A3
            pc_dev = abs(pc - BACKSOLVE_CD_PARAMS["PC_optimal"])
            if pc_dev < best_pc_dev:
                best_pc_dev = pc_dev
                best_cd = cd
                best_pc = pc

        # ── Physics fidelity from packing coefficient ──
        sigma = BACKSOLVE_CD_PARAMS["sigma_PC"]
        pc_optimal = BACKSOLVE_CD_PARAMS["PC_optimal"]
        pc_fidelity = math.exp(-0.5 * ((best_pc - pc_optimal) / sigma) ** 2)

        # ── Cavity shape compatibility ──
        shape_compat = {
            CavityShape.CONE: 1.0,
            CavityShape.SPHERE: 0.9,
            CavityShape.BARREL: 0.7,
            CavityShape.CUSTOM: 0.6,
            CavityShape.CLEFT: 0.3,
            CavityShape.FLAT: 0.2,
            CavityShape.CHANNEL: 0.1,
        }
        shape_score = shape_compat.get(spec.cavity_shape, 0.5)

        # ── Combine into physics fidelity ──
        physics_fidelity = pc_fidelity * 0.7 + shape_score * 0.3

        # ── Size gate ──
        if best_pc < 0.2 or best_pc > 0.9:
            physics_fidelity *= 0.1

        # ── Solvent gate ──
        if spec.solvent not in (Solvent.AQUEOUS, Solvent.MIXED):
            physics_fidelity *= 0.1

        # ── Deviation report ──
        spec_diameter = spec.cavity_dimensions.max_internal_diameter_A
        cd_diameter = best_cd.cavity_diameter_A if best_cd else 0.0
        diameter_dev = abs(spec_diameter - cd_diameter)

        deviation = DeviationReport(
            material_system="cyclodextrin",
            element_deviations_A=[diameter_dev],
            max_deviation_A=diameter_dev,
            mean_deviation_A=diameter_dev,
        )

        # ── Advantages / limitations ──
        advantages = []
        limitations = []

        if best_cd:
            if best_cd.commercial:
                advantages.append(
                    f"{best_cd.name} commercially available "
                    f"(${best_cd.cost_per_gram_usd}/g)"
                )
            if best_cd.water_solubility_mM > 100:
                advantages.append(
                    f"Good water solubility ({best_cd.water_solubility_mM:.0f} mM)"
                )
            if best_cd.backsolve_MAE_logK < 0.5:
                advantages.append(
                    f"Well-calibrated prediction "
                    f"(MAE={best_cd.backsolve_MAE_logK:.2f} log K)"
                )
            if best_pc < 0.4:
                limitations.append(f"Guest may be too small (PC={best_pc:.2f})")
            if best_pc > 0.75:
                limitations.append(f"Guest may be too large (PC={best_pc:.2f})")
            if spec.solvent == Solvent.ORGANIC:
                limitations.append("CDs require aqueous environment")
            if (best_cd.base_type == "beta"
                    and best_cd.modification == "native"):
                limitations.append("Native β-CD has low water solubility (16 mM)")

        return RealizationScore(
            material_system="cyclodextrin",
            adapter_id="CyclodextrinAdapter",
            deviation_from_ideal=deviation,
            physics_fidelity=max(0.0, min(1.0, physics_fidelity)),
            synthetic_accessibility=0.95,
            cost_score=0.90,
            scalability=0.95,
            operating_condition_compatibility=(
                0.9 if spec.solvent == Solvent.AQUEOUS else 0.2
            ),
            reusability_score=0.3,
            composite_score=0.0,
            confidence=(
                0.85 if best_cd and best_cd.n_calibration_entries > 10 else 0.60
            ),
            advantages=advantages,
            limitations=limitations,
            feasible=0.15 <= best_pc <= 0.95,
            infeasibility_reason=(
                f"No CD has suitable cavity (best PC={best_pc:.2f})"
                if best_pc < 0.15 or best_pc > 0.95 else None
            ),
        )

    def design(
        self,
        spec: InteractionGeometrySpec,
    ) -> CyclodextrinFabSpec:
        """
        Full CD pocket design.

        1. Rank all CDs by packing coefficient
        2. Apply modification logic
        3. Predict binding affinity
        4. Specify conjugation handle
        5. Output fab spec
        """

        target_vol = spec.cavity_dimensions.volume_A3
        spec_hash = hashlib.md5(str(spec).encode()).hexdigest()[:12]

        # ── Step 1: Rank native CDs by packing (no solubility filter) ──
        ranked = select_best_cd(
            guest_volume_A3=target_vol,
            require_water_soluble=False,  # pick best physics first
        )

        if not ranked:
            return self._empty_fab(spec_hash, "No suitable CD found")

        # Find best native CD (not modified)
        best_cd, best_pc, _rationale = ranked[0]
        for cd, pc, rat in ranked:
            if cd.modification == "native":
                best_cd, best_pc, _rationale = cd, pc, rat
                break

        # ── Step 2: Modification selection ──
        modifications = []
        mod_rationale = []

        # Solubility fix: if native CD has low solubility in aqueous app
        if (best_cd.modification == "native"
                and best_cd.water_solubility_mM < 50
                and spec.solvent in (Solvent.AQUEOUS, Solvent.MIXED)):
            modifications.append("HP")
            mod_rationale.append(
                "Hydroxypropylation recommended: native "
                f"{best_cd.base_type}-CD solubility "
                f"({best_cd.water_solubility_mM:.0f} mM) may limit application. "
                f"HP-{best_cd.base_type}-CD provides >500 mM with minimal "
                "cavity change."
            )
            # Switch to HP variant if available
            hp_key = f"HP-{best_cd.base_type}-CD"
            if hp_key in CD_HOSTS:
                best_cd = CD_HOSTS[hp_key]
                best_pc = target_vol / best_cd.cavity_volume_A3

        # Charge complementarity
        charged_donors = [
            d for d in spec.donor_positions if d.charge_state != 0
        ]
        if charged_donors and best_cd.base_type == "beta":
            if any(d.charge_state > 0 for d in charged_donors):
                if best_cd.modification != "SBE":
                    modifications.append("SBE")
                    mod_rationale.append(
                        "SBE modification: anionic sulfobutyl arms for "
                        "electrostatic complementarity with cationic guest."
                    )

        # Conjugation handle for deployment applications
        click_handle = "none"
        click_synthesis = ""
        if spec.target_application in (
            ApplicationContext.DIAGNOSTIC,
            ApplicationContext.REMEDIATION,
            ApplicationContext.SEPARATION,
        ):
            modifications.append("N3")
            mod_rationale.append(
                "C6-azide handle for CuAAC click conjugation to scaffold."
            )
            click_handle = "C6-azide"
            click_synthesis = (
                "1. Monotosylation: CD + TsCl (1.0 eq), NaOH, 0°C, 2h\n"
                "2. Azide displacement: OTs-CD + NaN3 (3 eq), DMF, 80°C, 12h\n"
                "3. Purification: dialysis (MWCO 500) + lyophilization"
            )

        # ── Step 3: Binding prediction ──
        predicted_dG, predicted_logK, confidence = self._predict_binding(
            best_cd, target_vol, spec
        )

        # ── Step 4: Packing quality label ──
        pc_dev = abs(best_pc - BACKSOLVE_CD_PARAMS["PC_optimal"])
        if pc_dev < 0.03:
            packing_quality = "excellent"
        elif pc_dev < 0.08:
            packing_quality = "good"
        elif pc_dev < 0.15:
            packing_quality = "suboptimal"
        else:
            packing_quality = "poor"

        # ── Step 5: Characterization plan ──
        validation = [
            "ITC for Ka, ΔH, ΔS",
            "1H NMR titration (ROESY for inclusion geometry)",
        ]
        if click_handle != "none":
            validation.append("FTIR: azide stretch at 2100 cm⁻¹")
            validation.append("MALDI-TOF: confirm mono-substitution MW")

        expected = {
            "Ka_M_inv": 10 ** predicted_logK if predicted_logK < 10 else 1e10,
            "dG_kJ_mol": predicted_dG,
            "stoichiometry": "1:1",
        }

        # ── Step 6: Synthesis steps ──
        steps = [f"Procure {best_cd.name} from {best_cd.common_suppliers[0]}"]
        if "HP" in modifications and best_cd.modification != "HP":
            steps.append("HP modification: propylene oxide, NaOH, 40°C, 24h")
        if "N3" in modifications:
            steps.append("Monotosylation: TsCl (1.0 eq), NaOH, 0°C, 2h")
            steps.append("Azide displacement: NaN3 (3 eq), DMF, 80°C, 12h")
            steps.append("Purification: dialysis + lyophilization")
        if not any(m in modifications for m in ["HP", "Me", "SBE", "NH2", "N3"]):
            steps.append("Use as-is (native CD)")

        return CyclodextrinFabSpec(
            material_system="cyclodextrin",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=CavityDimensions(
                volume_A3=best_cd.cavity_volume_A3,
                aperture_A=best_cd.portal_diameter_A,
                depth_A=best_cd.cavity_depth_A,
                max_internal_diameter_A=best_cd.cavity_diameter_A,
            ),
            predicted_deviation_from_ideal_A=abs(
                spec.cavity_dimensions.volume_A3 - best_cd.cavity_volume_A3
            ) ** (1/3),
            synthesis_steps=steps,
            estimated_yield=0.85 if not modifications else 0.60,
            estimated_cost_per_unit=best_cd.cost_per_gram_usd * (
                1.0 + 0.5 * len(modifications)
            ),
            estimated_time="1 day" if not modifications else "3–5 days",
            validation_experiments=validation,
            expected_observables=expected,
            selected_cd=best_cd.name,
            base_type=best_cd.base_type,
            cavity_diameter_A=best_cd.cavity_diameter_A,
            cavity_volume_A3=best_cd.cavity_volume_A3,
            packing_coefficient=best_pc,
            packing_quality=packing_quality,
            modifications=modifications,
            modification_rationale=mod_rationale,
            predicted_dG_kJ_mol=predicted_dG,
            predicted_logK=predicted_logK,
            prediction_confidence=confidence,
            click_handle=click_handle,
            click_handle_synthesis=click_synthesis,
            supplier=best_cd.common_suppliers[0],
            catalog_note=f"{best_cd.name}, ≥97%, {best_cd.mw:.0f} g/mol",
        )

    def validate_design(self, fab: FabricationSpec) -> ValidationReport:
        """Check CD design for internal consistency."""

        if not isinstance(fab, CyclodextrinFabSpec):
            return ValidationReport(
                valid=False,
                issues=["Not a CyclodextrinFabSpec"],
                warnings=[],
            )

        issues = []
        warnings = []

        if fab.packing_coefficient < 0.2:
            issues.append(
                f"PC {fab.packing_coefficient:.2f} too low — "
                "guest will not be retained"
            )
        elif fab.packing_coefficient > 0.85:
            issues.append(
                f"PC {fab.packing_coefficient:.2f} too high — "
                "steric clash prevents inclusion"
            )

        if fab.packing_coefficient < 0.4:
            warnings.append(
                f"Low packing ({fab.packing_coefficient:.2f}): "
                "weak binding expected"
            )
        if fab.packing_coefficient > 0.7:
            warnings.append(
                f"High packing ({fab.packing_coefficient:.2f}): "
                "slow kinetics possible"
            )

        if "SBE" in fab.modifications and fab.base_type != "beta":
            issues.append("SBE modification only available for β-CD")

        if fab.predicted_dG_kJ_mol > 0:
            warnings.append("Predicted ΔG > 0: unfavorable binding")

        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            confidence=0.85 if fab.prediction_confidence == "high" else 0.60,
        )

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _predict_binding(
        self,
        cd: CDHost,
        guest_volume_A3: float,
        spec: InteractionGeometrySpec,
    ) -> tuple[float, float, str]:
        """
        Predict binding ΔG using BackSolve calibrated parameters.

        Returns (ΔG kJ/mol, log K, confidence).
        """
        params = BACKSOLVE_CD_PARAMS

        # ── Hydrophobic burial ──
        guest_radius = (3 * guest_volume_A3 / (4 * math.pi)) ** (1/3)
        estimated_sasa = 4 * math.pi * guest_radius ** 2
        burial_fraction = min(
            1.0, cd.cavity_depth_A / (2 * guest_radius + 1)
        )
        buried_sasa = estimated_sasa * burial_fraction
        dG_hydrophobic = params["gamma_hydrophobic"] * buried_sasa

        # ── Shape complementarity ──
        pc = guest_volume_A3 / cd.cavity_volume_A3
        pc_penalty = params["k_shape"] * math.exp(
            -0.5 * ((pc - params["PC_optimal"]) / params["sigma_PC"]) ** 2
        )

        # ── H-bond estimate ──
        n_hbonds = 0
        if spec.h_bond_network and spec.h_bond_network.donors:
            n_hbonds = min(len(spec.h_bond_network.donors), 3)
        dG_hbond = n_hbonds * params["eps_neutral_hbond"]
        dG_water = n_hbonds * params["water_penalty_per_hb"]

        # ── Conformational entropy ──
        n_rotors = len([
            d for d in spec.donor_positions
            if d.coordination_role in ("h_bond_donor", "h_bond_acceptor")
        ])
        dG_entropy = params["eps_rotor"] * n_rotors * params["f_partial_freeze"]

        # ── Total ──
        dG = dG_hydrophobic + pc_penalty + dG_hbond + dG_water + dG_entropy

        RT = 8.314e-3 * 298.15
        logK = -dG / (2.303 * RT)

        if cd.n_calibration_entries >= 15:
            confidence = "high"
        elif cd.n_calibration_entries >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        return dG, logK, confidence

    def _empty_fab(self, spec_hash: str, reason: str) -> CyclodextrinFabSpec:
        """Return an empty fab spec when design fails."""
        return CyclodextrinFabSpec(
            material_system="cyclodextrin",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=CavityDimensions(0, 0, 0, 0),
            predicted_deviation_from_ideal_A=float("inf"),
            synthesis_steps=[f"DESIGN FAILED: {reason}"],
        )


def _make_cd_capability() -> MaterialCapability:
    """Build a CD-specific MaterialCapability."""
    return MaterialCapability(
        system_id="cyclodextrin",
        physics_class="covalent_cavity",
        adapter_class="CyclodextrinAdapter",
        min_pocket_size_nm=0.47,
        max_pocket_size_nm=0.83,
        achievable_symmetries=["Cn", "Cnv"],
        max_donor_count=16,
        donor_types_available=["O"],
        positioning_precision_A=0.10,
        rigidity_range=("semi-rigid", "rigid"),
        pH_stability=(2.0, 12.0),
        thermal_stability_K=(273.0, 500.0),
        solvent_compatibility=["aqueous", "mixed"],
        min_practical_scale="µmol",
        max_practical_scale="kmol",
        cost_per_unit_range=(0.15, 15.0),
        typical_synthesis_time="1–5 days",
        literature_validation_rate=0.80,
        literature_examples=20000,
        design_tools_available=["RDKit"],
        known_strengths=[
            "Commodity chemical",
            "BackSolve-calibrated (MAE=0.37 log K for β-CD)",
            "Click-handle conjugation well-precedented",
        ],
        known_limitations=[
            "Truncated cone shape only",
            "Aqueous-only",
            "Limited selectivity for hydrophilic guests",
        ],
    )
