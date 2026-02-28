"""
Activated Carbon Adapter — Surface Paradigm.

Takes a SurfaceInteractionSpec and designs an AC adsorption solution.
Uses isotherm physics (Langmuir/Freundlich) and surface chemistry
principles to predict performance and recommend AC type.

Physics connection:
    - Langmuir isotherm governs monolayer capacity
    - Freundlich isotherm for heterogeneous surfaces
    - pH vs pHpzc governs surface charge → electrostatic contribution
    - Competitive adsorption ordering from Xiao & Thomas 2004
    - Pore size distribution governs accessibility

This adapter does NOT use cavity geometry, donor positions, or any
pocket-paradigm physics. It operates entirely in the surface paradigm.

CRITICAL CALIBRATION NOTE:
    Isotherm parameters (qmax, KL, KF, n) are MATERIAL-SPECIFIC.
    This adapter provides feasibility assessment and design guidance,
    NOT precise capacity predictions. Every design requires batch
    isotherm validation with the actual AC material.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.models import (
    ApplicationContext,
    FabricationSpec,
    SurfaceInteractionSpec,
    SurfaceMechanism,
    BaseMaterial,
    IsothermModel,
    Solvent,
)
from mabe.realization.adapters.ac_knowledge import (
    DATA_SOURCES,
    METAL_ADSORPTION_ORDER_OXIDIZED_AC,
    AC_PROFILES,
    PORE_CLASSIFICATION,
    langmuir_qe,
    freundlich_qe,
    langmuir_separation_factor,
    ph_adsorption_factor,
    recommend_ac_type,
)


# ─────────────────────────────────────────────
# Fabrication spec for activated carbon
# ─────────────────────────────────────────────

@dataclass
class ACFabricationSpec(FabricationSpec):
    """Layer 4 output for an activated carbon adsorption design."""

    # ── Material selection ──
    ac_type: str = ""                     # "coconut_shell_gac", "oxidized_gac", etc.
    ac_source: str = ""
    activation_method: str = ""

    # ── Physical properties ──
    required_surface_area_m2_g: float = 0.0
    recommended_pore_type: str = ""       # "micropore", "mesopore"
    recommended_ph_pzc: float = 0.0

    # ── Performance predictions ──
    estimated_capacity_range_mg_g: tuple[float, float] = (0.0, 0.0)
    ph_suitability: float = 0.0           # 0-1 factor
    competitive_adsorption_rank: int = 0  # vs. known competitors

    # ── Operational parameters ──
    recommended_dose_g_L: float = 0.0
    recommended_contact_time_min: float = 0.0
    recommended_pH: float = 0.0

    # ── Calibration status ──
    calibration_status: str = "uncalibrated"  # per data gate requirement
    calibration_notes: list[str] = field(default_factory=list)

    # ── Data provenance ──
    data_sources: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────

class ActivatedCarbonAdapter:
    """
    Designs an activated carbon adsorption solution for a SurfaceInteractionSpec.

    Scores based on:
        - Target compatibility with AC surface chemistry
        - pH suitability (pHpzc relationship)
        - Competitive adsorption position
        - Capacity range from literature
        - Material availability and cost
    """

    system_id = "activated_carbon"
    supported_spec_types = ["surface"]

    def score(self, spec: SurfaceInteractionSpec) -> dict:
        """
        Score feasibility and performance of AC adsorption approach.

        Returns dict with scoring components and feasibility assessment.
        """
        # ── Select AC type ──
        ac_key = self._select_ac(spec)
        profile = AC_PROFILES.get(ac_key)
        if profile is None:
            return self._infeasible(f"No AC profile for {ac_key}")

        # ── pH suitability ──
        mid_ph = sum(spec.pH_range) / 2.0
        ph_factor = ph_adsorption_factor(mid_ph, profile.ph_pzc, spec.target_charge)

        # ── Surface area check ──
        sa_score = self._score_surface_area(spec, profile)

        # ── Capacity estimation ──
        cap_score, cap_range = self._score_capacity(spec, profile)

        # ── Competitive adsorption ranking ──
        comp_score, comp_rank = self._score_competition(spec)

        # ── Mechanism compatibility ──
        mech_score = self._score_mechanism(spec)

        # ── Composite ──
        if spec.target_application == ApplicationContext.REMEDIATION:
            composite = (
                0.30 * cap_score
                + 0.25 * ph_factor
                + 0.20 * comp_score
                + 0.15 * sa_score
                + 0.10 * mech_score
            )
        else:
            composite = (
                0.25 * cap_score
                + 0.25 * comp_score
                + 0.20 * ph_factor
                + 0.15 * mech_score
                + 0.15 * sa_score
            )

        # ── Confidence ──
        # AC is well-studied but parameters are material-specific
        confidence = self._compute_confidence(spec, profile)

        return {
            "feasible": True,
            "ac_type": ac_key,
            "ac_profile": profile.name,
            "capacity_score": round(cap_score, 3),
            "estimated_capacity_range_mg_g": cap_range,
            "ph_suitability": round(ph_factor, 3),
            "surface_area_score": round(sa_score, 3),
            "competition_score": round(comp_score, 3),
            "competition_rank": comp_rank,
            "mechanism_score": round(mech_score, 3),
            "composite_score": round(composite, 3),
            "confidence": confidence,
            "calibration_status": "uncalibrated",
            "calibration_notes": [
                "Capacity predictions are ranges based on AC class, not material-specific isotherms.",
                "Batch isotherm experiment required before design finalization.",
            ],
        }

    def design(self, spec: SurfaceInteractionSpec) -> ACFabricationSpec:
        """Produce a fabrication spec for the AC adsorption design."""
        score_result = self.score(spec)

        ac_key = score_result.get("ac_type", "coal_gac")
        profile = AC_PROFILES.get(ac_key)
        cap_range = score_result.get("estimated_capacity_range_mg_g", (5.0, 50.0))

        # ── Dose calculation (from capacity range midpoint) ──
        mid_cap = (cap_range[0] + cap_range[1]) / 2.0
        if mid_cap > 0 and spec.initial_concentration_mg_L > 0:
            # dose (g/L) = C0 / qe (assuming target removal)
            target_removal = max(spec.target_removal_efficiency, 0.9)
            dose = spec.initial_concentration_mg_L * target_removal / mid_cap
        else:
            dose = 5.0  # default starting dose

        # ── pH recommendation ──
        if profile and spec.target_charge > 0:
            rec_ph = max(profile.ph_pzc + 1.0, spec.pH_range[0])
            rec_ph = min(rec_ph, spec.pH_range[1])
        elif profile and spec.target_charge < 0:
            rec_ph = min(profile.ph_pzc - 1.0, spec.pH_range[1])
            rec_ph = max(rec_ph, spec.pH_range[0])
        else:
            rec_ph = 7.0

        # ── Pore type recommendation ──
        if spec.target_mw_g_mol < 200 or spec.target_charge != 0:
            pore_type = "micropore"
        else:
            pore_type = "mesopore"

        spec_hash = hashlib.md5(
            f"{spec.target_species}:{ac_key}:{spec.mechanism.value}".encode()
        ).hexdigest()[:12]

        synthesis_steps = [
            f"Select {profile.name if profile else 'GAC'} ({profile.source if profile else 'commercial'})",
            f"Wash AC with deionized water to remove fines",
            f"Dry at 105°C for 24h",
            f"Perform batch isotherm: 0.1-10 g/L dose, "
            f"{spec.target_species} at {spec.initial_concentration_mg_L:.0f} mg/L, pH {rec_ph:.1f}",
            f"Fit Langmuir and Freundlich models to determine actual qmax and KL",
        ]

        if profile and "oxidized" in ac_key:
            synthesis_steps.insert(1, "Optional: oxidize with HNO3 (5M, 80°C, 4h) to enhance surface groups")

        validation = [
            f"Batch isotherm: at least 6 concentration points at pH {rec_ph:.1f}",
            f"pH edge experiment: adsorption vs. pH (2-12) at C₀ = {spec.initial_concentration_mg_L:.0f} mg/L",
            f"Kinetics: time series at optimal dose to determine contact time",
        ]
        if spec.competing_species:
            validation.append(
                f"Competitive batch test: {spec.target_species} + "
                f"{', '.join(spec.competing_species[:3])} at equimolar concentrations"
            )
        if spec.reusability_required:
            validation.append("Regeneration test: 0.1M HCl or NaOH, 5 adsorption-desorption cycles")

        return ACFabricationSpec(
            material_system="activated_carbon",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=None,  # No pocket — surface paradigm
            predicted_deviation_from_ideal_A=0.0,  # N/A
            ac_type=ac_key,
            ac_source=profile.source if profile else "commercial",
            activation_method=profile.activation if profile else "steam",
            required_surface_area_m2_g=max(
                spec.min_surface_area_m2_g,
                profile.bet_surface_area_m2_g[0] if profile else 800.0,
            ),
            recommended_pore_type=pore_type,
            recommended_ph_pzc=profile.ph_pzc if profile else 7.0,
            estimated_capacity_range_mg_g=cap_range,
            ph_suitability=score_result.get("ph_suitability", 0.5),
            competitive_adsorption_rank=score_result.get("competition_rank", 0),
            recommended_dose_g_L=round(dose, 2),
            recommended_contact_time_min=60.0,  # standard starting point
            recommended_pH=round(rec_ph, 1),
            calibration_status="uncalibrated",
            calibration_notes=score_result.get("calibration_notes", []),
            synthesis_steps=synthesis_steps,
            validation_experiments=validation,
            estimated_cost_per_unit=self._estimate_cost(ac_key, spec),
            estimated_time="3-5 days (isotherm + kinetics characterization)",
            data_sources=DATA_SOURCES,
        )

    # ─────────────────────────────────────────
    # Internal methods
    # ─────────────────────────────────────────

    def _select_ac(self, spec: SurfaceInteractionSpec) -> str:
        """Select AC type from spec or auto-recommend."""
        if spec.base_material is not None:
            material_map = {
                BaseMaterial.ACTIVATED_CARBON: "coal_gac",
            }
            return material_map.get(spec.base_material, "coal_gac")

        return recommend_ac_type(
            spec.target_species, spec.target_charge, spec.target_mw_g_mol,
        )

    def _score_surface_area(
        self, spec: SurfaceInteractionSpec, profile: "ACMaterialProfile",
    ) -> float:
        """Score: does the AC meet surface area requirements?"""
        if spec.min_surface_area_m2_g <= 0:
            return 1.0  # no requirement

        typical_sa = (profile.bet_surface_area_m2_g[0] + profile.bet_surface_area_m2_g[1]) / 2.0
        if typical_sa >= spec.min_surface_area_m2_g:
            return 1.0
        ratio = typical_sa / spec.min_surface_area_m2_g
        return max(0.0, ratio)

    def _score_capacity(
        self, spec: SurfaceInteractionSpec, profile: "ACMaterialProfile",
    ) -> tuple[float, tuple[float, float]]:
        """Score capacity: does AC class have sufficient capacity?"""
        cap_range = profile.typical_capacity_heavy_metals_mg_g

        if spec.target_capacity_mg_g <= 0:
            return 0.7, cap_range  # no specific requirement → moderate score

        mid_cap = (cap_range[0] + cap_range[1]) / 2.0
        if mid_cap >= spec.target_capacity_mg_g:
            return 1.0, cap_range
        elif cap_range[1] >= spec.target_capacity_mg_g:
            return 0.7, cap_range  # achievable at upper end
        else:
            ratio = cap_range[1] / spec.target_capacity_mg_g
            return max(0.0, round(ratio, 3)), cap_range

    def _score_competition(
        self, spec: SurfaceInteractionSpec,
    ) -> tuple[float, int]:
        """Score: where does target rank in competitive adsorption?"""
        # Check if target is in the Xiao & Thomas ordering
        order_map = {entry[0]: entry[1] for entry in METAL_ADSORPTION_ORDER_OXIDIZED_AC}

        rank = order_map.get(spec.target_species, 0)
        if rank == 0:
            # Unknown species — can't score competition
            return 0.5, 0

        max_rank = max(entry[1] for entry in METAL_ADSORPTION_ORDER_OXIDIZED_AC)

        # Check competitors
        competitor_ranks = []
        for comp in spec.competing_species:
            comp_rank = order_map.get(comp, 0)
            if comp_rank > 0:
                competitor_ranks.append(comp_rank)

        if not competitor_ranks:
            # No competitor data — score by absolute rank
            return round(rank / max_rank, 3), rank

        # Score by worst-case: target must outcompete all competitors
        worst_competitor = max(competitor_ranks)
        if rank > worst_competitor:
            return 1.0, rank  # target preferred
        elif rank == worst_competitor:
            return 0.5, rank  # tied
        else:
            return 0.2, rank  # competitor preferred — AC may not work

    def _score_mechanism(self, spec: SurfaceInteractionSpec) -> float:
        """Score mechanism compatibility with AC."""
        # AC supports all surface mechanisms to varying degrees
        mechanism_scores = {
            SurfaceMechanism.CHEMISORPTION: 0.9,
            SurfaceMechanism.SURFACE_COMPLEXATION: 0.85,
            SurfaceMechanism.PHYSISORPTION: 0.8,
            SurfaceMechanism.ELECTROSTATIC: 0.7,
            SurfaceMechanism.ION_EXCHANGE_SURFACE: 0.75,
        }
        return mechanism_scores.get(spec.mechanism, 0.5)

    def _compute_confidence(
        self, spec: SurfaceInteractionSpec, profile: "ACMaterialProfile",
    ) -> float:
        """
        Confidence: AC is well-studied technology but material-specific.

        Base confidence is LOWER than IX because isotherm parameters
        are not transferable between AC products.
        """
        base = 0.60  # lower than IX (0.85) because material-specific

        # Bonus if target is in known competitive ordering
        order_map = {entry[0]: entry[1] for entry in METAL_ADSORPTION_ORDER_OXIDIZED_AC}
        if spec.target_species in order_map:
            base += 0.10

        # Penalty for unknowns
        if spec.target_charge == 0 and spec.target_mw_g_mol > 500:
            base -= 0.10  # large neutral organics — less predictable

        return max(0.1, round(base, 2))

    def _estimate_cost(self, ac_key: str, spec: SurfaceInteractionSpec) -> float:
        """Rough cost per kg of AC."""
        profile = AC_PROFILES.get(ac_key)
        if profile:
            return (profile.cost_per_kg_usd[0] + profile.cost_per_kg_usd[1]) / 2.0
        return 3.0

    def _infeasible(self, reason: str) -> dict:
        return {
            "feasible": False,
            "reason": reason,
            "capacity_score": 0.0,
            "ph_suitability": 0.0,
            "surface_area_score": 0.0,
            "competition_score": 0.0,
            "mechanism_score": 0.0,
            "composite_score": 0.0,
            "confidence": 0.0,
        }