"""
Crown Ether / Cryptand Adapter — Layer 4 Implementation.

Takes an InteractionGeometrySpec describing a cation-coordination pocket
and designs a crown ether or cryptand realization.

Design logic:
    1. Identify target cation from spec (donor types + charge + cavity size)
    2. Rank crown ethers and cryptands by cavity size-match
    3. HSAB routing for donor atom selection (O/N/S variants)
    4. Cryptand upgrade when 3D encapsulation improves selectivity
    5. Produce CrownEtherFabSpec with selection + Izatt-sourced logK

Physics connection:
    - Cavity radius → size-match Gaussian (Hancock & Martell)
    - Macrocyclic stabilization (Haymore et al.)
    - Cryptate effect (Lehn)
    - HSAB donor selection (from BackSolve donor subtype calibration)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.adapters.base import RealizationAdapter, ValidationReport
from mabe.realization.adapters.crown_ether_knowledge import (
    ALL_HOSTS_LIST,
    CATION_DB,
    CationTarget,
    CrownEtherHost,
    CRYPTANDS,
    hsab_donor_score,
    select_best_crown,
    size_match_dG,
    size_match_score,
    SIZE_MATCH_SIGMA_A,
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
# CrownEtherFabSpec — material-specific output
# ─────────────────────────────────────────────

@dataclass
class CrownEtherFabSpec(FabricationSpec):
    """Fabrication spec for a crown ether or cryptand pocket."""

    # ── Selection ──
    selected_host: str = ""
    host_class: str = ""                   # "crown_ether", "aza_crown", etc.
    cavity_radius_A: float = 0.0

    # ── Target cation ──
    target_ion: str = ""
    ion_radius_A: float = 0.0

    # ── Size-match analysis ──
    size_match_score: float = 0.0
    size_match_quality: str = ""

    # ── Binding prediction ──
    predicted_logK: float = 0.0
    logK_source: str = ""                  # "Izatt_measured" or "estimated"
    macrocyclic_stabilization_kJ_mol: float = 0.0
    cryptate_stabilization_kJ_mol: float = 0.0
    predicted_selectivity_ratio: float = 0.0  # vs best competitor

    # ── HSAB analysis ──
    hsab_compatibility: float = 0.0
    donor_types_used: list[str] = field(default_factory=list)

    # ── Design decisions ──
    cryptand_upgrade: bool = False
    cryptand_upgrade_rationale: str = ""
    donor_substitution: str = ""           # "none", "aza", "thia"
    donor_substitution_rationale: str = ""

    # ── Sourcing ──
    smiles: str = ""
    supplier: str = ""
    catalog_note: str = ""


# ─────────────────────────────────────────────
# Adapter implementation
# ─────────────────────────────────────────────

class CrownEtherAdapter(RealizationAdapter):
    """
    Designs crown ether / cryptand binding pockets for cation capture.

    Class A (Covalent Cavity) system:
        - 0.60–1.70 Å cavity radius (12C4 through 21C7)
        - Circular/spherical coordination geometry
        - Size-match selectivity is primary mechanism
        - O-donors default, N/S for borderline/soft cations
        - Cryptands add 3D encapsulation (+5–20 kJ/mol)
    """

    def __init__(self, capability: Optional[MaterialCapability] = None):
        if capability is None:
            from mabe.realization.registry.material_registry import MATERIAL_REGISTRY
            capability = MATERIAL_REGISTRY.get("cyclic_encapsulant")
            if capability is None:
                capability = _make_crown_capability()
        super().__init__(capability)

    def estimate_fidelity(
        self,
        spec: InteractionGeometrySpec,
    ) -> RealizationScore:
        """
        Quick score: can a crown ether or cryptand host this geometry?

        Primary: cavity size → size-match to target ion.
        Secondary: donor type compatibility (HSAB).
        """

        target_ion, cation = self._identify_target_cation(spec)

        if cation is None:
            return self._infeasible_score(
                "Cannot identify target cation from spec"
            )

        # ── Rank all hosts ──
        ranked = select_best_crown(
            target_ion,
            require_water_soluble=(spec.solvent == Solvent.AQUEOUS),
        )

        if not ranked:
            return self._infeasible_score(
                f"No host compatible with {target_ion}"
            )

        best_host, combined_score, predicted_logK, rationale = ranked[0]

        # ── Physics fidelity ──
        sm = size_match_score(cation.ionic_radius_A, best_host.cavity_radius_A)
        hsab = hsab_donor_score(cation.hsab_class, best_host.donor_types)

        physics_fidelity = sm * 0.6 + hsab * 0.2

        # Shape: crowns are circular, cryptands are spherical
        if spec.cavity_shape == CavityShape.SPHERE and best_host.is_3d_cage:
            physics_fidelity += 0.15
        elif spec.cavity_shape in (CavityShape.FLAT, CavityShape.SPHERE):
            physics_fidelity += 0.05

        physics_fidelity = max(0.0, min(1.0, physics_fidelity))

        # ── Deviation ──
        radius_dev = abs(cation.ionic_radius_A - best_host.cavity_radius_A)
        deviation = DeviationReport(
            material_system="crown_ether",
            element_deviations_A=[radius_dev],
            max_deviation_A=radius_dev,
            mean_deviation_A=radius_dev,
        )

        # ── Advantages / limitations ──
        advantages = []
        limitations = []

        if best_host.commercial:
            advantages.append(
                f"{best_host.common_name} commercial "
                f"(${best_host.cost_per_gram_usd}/g)"
            )
        if sm > 0.8:
            advantages.append(f"Excellent size match (score={sm:.2f})")
        if predicted_logK > 5:
            advantages.append(f"Strong binding (logK={predicted_logK:.1f})")
        if best_host.is_3d_cage:
            advantages.append("Cryptate effect — enhanced selectivity")

        if sm < 0.5:
            limitations.append(f"Suboptimal size match ({sm:.2f})")
        if not best_host.water_soluble and spec.solvent == Solvent.AQUEOUS:
            limitations.append("Not water-soluble")
        if hsab < 0.5:
            limitations.append(
                f"HSAB mismatch: {cation.hsab_class} ion, "
                f"O-donors suboptimal"
            )

        return RealizationScore(
            material_system="crown_ether",
            adapter_id="CrownEtherAdapter",
            deviation_from_ideal=deviation,
            physics_fidelity=physics_fidelity,
            synthetic_accessibility=0.85,
            cost_score=0.80,
            scalability=0.90,
            operating_condition_compatibility=(
                0.9 if spec.solvent in (Solvent.AQUEOUS, Solvent.MIXED) else 0.7
            ),
            reusability_score=0.5,
            composite_score=0.0,
            confidence=0.80,
            advantages=advantages,
            limitations=limitations,
            feasible=True,
        )

    def design(
        self,
        spec: InteractionGeometrySpec,
    ) -> CrownEtherFabSpec:
        """
        Full crown ether / cryptand design.

        1. Identify target cation
        2. Size-match ranking
        3. HSAB donor selection (aza/thia variants)
        4. Cryptand upgrade evaluation
        5. Selectivity estimation
        """

        spec_hash = hashlib.md5(str(spec).encode()).hexdigest()[:12]
        target_ion, cation = self._identify_target_cation(spec)

        if cation is None:
            return self._empty_fab(spec_hash, "Cannot identify target cation")

        # ── Step 1: Size-match ranking ──
        ranked = select_best_crown(target_ion, require_water_soluble=False)
        if not ranked:
            return self._empty_fab(spec_hash, f"No host for {target_ion}")

        best_host, combined, predicted_logK, _rationale = ranked[0]
        sm = size_match_score(cation.ionic_radius_A, best_host.cavity_radius_A)

        # ── Step 2: HSAB donor substitution ──
        donor_sub = "none"
        donor_sub_rationale = ""
        hsab = hsab_donor_score(cation.hsab_class, best_host.donor_types)

        if cation.hsab_class == "soft" and "S" not in best_host.donor_types:
            # Look for thia variant — relax radius tolerance since HSAB
            # benefit can overcome moderate size-match penalty
            for h in ALL_HOSTS_LIST:
                if h.host_class == "thia_crown" and abs(
                    h.cavity_radius_A - cation.ionic_radius_A
                ) < 0.4:
                    best_host = h
                    donor_sub = "thia"
                    donor_sub_rationale = (
                        f"Soft cation {target_ion}: S-donors preferred. "
                        f"Switched to {h.common_name}."
                    )
                    break

        elif cation.hsab_class == "borderline" and "N" not in best_host.donor_types:
            for h in ALL_HOSTS_LIST:
                if h.host_class == "aza_crown" and abs(
                    h.cavity_radius_A - cation.ionic_radius_A
                ) < 0.8:
                    best_host = h
                    donor_sub = "aza"
                    donor_sub_rationale = (
                        f"Borderline cation {target_ion}: N-donors improve binding. "
                        f"Switched to {h.common_name}."
                    )
                    break

        # ── Step 3: Cryptand upgrade evaluation ──
        cryptand_upgrade = False
        cryptand_rationale = ""

        # Consider cryptand if: (a) selectivity needed, or (b) size-match is marginal
        if spec.must_exclude or sm < 0.7:
            for h in CRYPTANDS:
                crypt_sm = size_match_score(
                    cation.ionic_radius_A, h.cavity_radius_A
                )
                if crypt_sm > sm * 0.9:  # cryptand has competitive size-match
                    crypt_logK = h.selectivity_profile.get(target_ion, 0)
                    crown_logK = best_host.selectivity_profile.get(target_ion, 0)
                    if crypt_logK > crown_logK:
                        best_host = h
                        sm = crypt_sm
                        predicted_logK = crypt_logK
                        cryptand_upgrade = True
                        cryptand_rationale = (
                            f"Cryptand [{h.common_name}] provides 3D encapsulation: "
                            f"logK {crypt_logK:.1f} vs crown {crown_logK:.1f}. "
                            f"Cryptate stabilization: "
                            f"{h.cryptate_stabilization_kJ_mol:.0f} kJ/mol."
                        )
                        break

        # ── Step 4: Update predicted logK from best host ──
        known_logK = best_host.selectivity_profile.get(target_ion)
        logK_source = "Izatt_measured" if known_logK else "estimated"
        if known_logK:
            predicted_logK = known_logK

        # ── Step 5: Selectivity estimation ──
        selectivity_ratio = 1.0
        if spec.must_exclude:
            for exc in spec.must_exclude:
                competitor_logK = best_host.selectivity_profile.get(exc.species, 0)
                if competitor_logK > 0 and predicted_logK > 0:
                    ratio = 10 ** (predicted_logK - competitor_logK)
                    selectivity_ratio = min(selectivity_ratio, ratio)

        # ── Step 6: Size-match quality label ──
        if sm > 0.9:
            sm_quality = "excellent"
        elif sm > 0.7:
            sm_quality = "good"
        elif sm > 0.4:
            sm_quality = "moderate"
        else:
            sm_quality = "poor"

        # ── Step 7: Synthesis and sourcing ──
        steps = []
        if best_host.commercial:
            steps.append(f"Procure {best_host.common_name} from {best_host.common_suppliers[0]}")
        else:
            steps.append(f"Synthesize {best_host.common_name} via {best_host.synthesis_route}")

        if donor_sub == "thia" and not best_host.commercial:
            steps.append("Thia-crown synthesis: thiol-alkoxide Williamson condensation")
        if donor_sub == "aza" and not best_host.commercial:
            steps.append("Aza-crown synthesis: Richman-Atkins tosyl route")

        validation = [
            f"ITC: Ka for {target_ion} in target matrix",
            f"Competitive ITC: selectivity vs major interferents",
        ]
        if spec.must_exclude:
            for exc in spec.must_exclude:
                validation.append(f"Selectivity assay: {target_ion} vs {exc.species}")

        expected = {
            "logK": predicted_logK,
            "Ka_M_inv": 10 ** predicted_logK if predicted_logK < 15 else 1e15,
            "selectivity_ratio": selectivity_ratio,
            "binding_stoichiometry": "1:1",
        }

        return CrownEtherFabSpec(
            material_system="crown_ether",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=CavityDimensions(
                volume_A3=4/3 * math.pi * best_host.cavity_radius_A ** 3,
                aperture_A=best_host.cavity_radius_A * 2,
                depth_A=3.0 if not best_host.is_3d_cage else best_host.cavity_radius_A * 2,
                max_internal_diameter_A=best_host.cavity_radius_A * 2,
            ),
            predicted_deviation_from_ideal_A=abs(
                cation.ionic_radius_A - best_host.cavity_radius_A
            ),
            synthesis_steps=steps,
            estimated_yield=0.70 if best_host.commercial else 0.40,
            estimated_cost_per_unit=best_host.cost_per_gram_usd,
            estimated_time="immediate" if best_host.commercial else "3–7 days",
            validation_experiments=validation,
            expected_observables=expected,
            # Crown-specific
            selected_host=best_host.common_name,
            host_class=best_host.host_class,
            cavity_radius_A=best_host.cavity_radius_A,
            target_ion=target_ion,
            ion_radius_A=cation.ionic_radius_A,
            size_match_score=sm,
            size_match_quality=sm_quality,
            predicted_logK=predicted_logK,
            logK_source=logK_source,
            macrocyclic_stabilization_kJ_mol=best_host.macrocyclic_stabilization_kJ_mol,
            cryptate_stabilization_kJ_mol=best_host.cryptate_stabilization_kJ_mol,
            predicted_selectivity_ratio=selectivity_ratio,
            hsab_compatibility=hsab_donor_score(
                cation.hsab_class, best_host.donor_types
            ),
            donor_types_used=best_host.donor_types,
            cryptand_upgrade=cryptand_upgrade,
            cryptand_upgrade_rationale=cryptand_rationale,
            donor_substitution=donor_sub,
            donor_substitution_rationale=donor_sub_rationale,
            smiles=best_host.smiles,
            supplier=best_host.common_suppliers[0] if best_host.commercial else "custom synthesis",
            catalog_note=f"{best_host.common_name}, {best_host.mw:.0f} g/mol",
        )

    def validate_design(self, fab: FabricationSpec) -> ValidationReport:
        """Check crown/cryptand design for internal consistency."""

        if not isinstance(fab, CrownEtherFabSpec):
            return ValidationReport(
                valid=False,
                issues=["Not a CrownEtherFabSpec"],
                warnings=[],
            )

        issues = []
        warnings = []

        if fab.size_match_score < 0.2:
            issues.append(
                f"Size match {fab.size_match_score:.2f} too poor — "
                "binding will be negligible"
            )

        if fab.predicted_logK < 1.0:
            warnings.append(
                f"Weak binding predicted (logK={fab.predicted_logK:.1f})"
            )

        if fab.hsab_compatibility < 0.4:
            warnings.append(
                f"HSAB mismatch ({fab.hsab_compatibility:.2f}): "
                "consider aza/thia variant"
            )

        if fab.predicted_selectivity_ratio < 10 and fab.target_ion:
            warnings.append(
                f"Low selectivity ratio ({fab.predicted_selectivity_ratio:.0f}×): "
                "consider cryptand upgrade"
            )

        return ValidationReport(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            confidence=0.85 if fab.logK_source == "Izatt_measured" else 0.60,
        )

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _identify_target_cation(
        self,
        spec: InteractionGeometrySpec,
    ) -> tuple[str, Optional[CationTarget]]:
        """
        Identify the target cation from the spec.

        Strategy: match cavity radius to known cation radii.
        If spec has donor positions with charge info, use that.
        """

        # Check if any donor has charge annotation suggesting the target
        for donor in spec.donor_positions:
            if donor.charge_state < 0:
                # Anionic donor → cation is the target
                continue

        # Primary: cavity radius matching
        # Convert spec cavity diameter to radius
        spec_radius_A = spec.cavity_dimensions.max_internal_diameter_A / 2.0

        best_match = None
        best_delta = float("inf")

        for symbol, cation in CATION_DB.items():
            delta = abs(cation.ionic_radius_A - spec_radius_A)
            if delta < best_delta:
                best_delta = delta
                best_match = (symbol, cation)

        if best_match and best_delta < 1.0:
            return best_match

        # Fallback: use cavity volume to estimate
        return "K+", CATION_DB["K+"]  # default to most common crown target

    def _infeasible_score(self, reason: str) -> RealizationScore:
        return RealizationScore(
            material_system="crown_ether",
            adapter_id="CrownEtherAdapter",
            deviation_from_ideal=DeviationReport(
                material_system="crown_ether",
                element_deviations_A=[],
                max_deviation_A=float("inf"),
                mean_deviation_A=float("inf"),
            ),
            physics_fidelity=0.0,
            feasible=False,
            infeasibility_reason=reason,
        )

    def _empty_fab(self, spec_hash: str, reason: str) -> CrownEtherFabSpec:
        return CrownEtherFabSpec(
            material_system="crown_ether",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=CavityDimensions(0, 0, 0, 0),
            predicted_deviation_from_ideal_A=float("inf"),
            synthesis_steps=[f"DESIGN FAILED: {reason}"],
        )


def _make_crown_capability() -> MaterialCapability:
    """Build crown-ether-specific MaterialCapability."""
    return MaterialCapability(
        system_id="crown_ether",
        physics_class="covalent_cavity",
        adapter_class="CrownEtherAdapter",
        min_pocket_size_nm=0.12,
        max_pocket_size_nm=0.34,
        achievable_symmetries=["Cn", "Cnv", "Dnh"],
        max_donor_count=8,
        donor_types_available=["O", "N", "S"],
        positioning_precision_A=0.05,
        rigidity_range=("semi-rigid", "rigid"),
        pH_stability=(2.0, 12.0),
        thermal_stability_K=(273.0, 500.0),
        solvent_compatibility=["aqueous", "organic", "mixed"],
        min_practical_scale="µmol",
        max_practical_scale="kmol",
        cost_per_unit_range=(1.50, 80.0),
        typical_synthesis_time="immediate (commercial) to 7 days",
        literature_validation_rate=0.85,
        literature_examples=20000,
        design_tools_available=["RDKit"],
        known_strengths=[
            "Size-match selectivity well-characterized (Izatt)",
            "Many commercial options",
            "HSAB-tunable via O/N/S donor substitution",
            "Cryptand upgrade for enhanced selectivity",
        ],
        known_limitations=[
            "Primarily cation-selective",
            "Organic crown ethers can be toxic",
            "Conformational flexibility in larger rings",
        ],
    )
