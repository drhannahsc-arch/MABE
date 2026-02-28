"""
Ion Exchange Resin Adapter — Network Paradigm.

Takes a NetworkInteractionSpec and designs an IX resin solution.
Uses selectivity coefficients from DuPont published data (Tier 2)
to predict separation factors and recommend resin configuration.

Physics connection:
    - Selectivity coefficient K_A/B governs equilibrium ion distribution
    - Separation factor α = K_target / K_competitor
    - DVB cross-linking controls selectivity magnitude + kinetics tradeoff
    - Capacity from resin type (SAC ~1.8 meq/mL, WAC ~3.5 meq/mL)
    - Donnan exclusion provides charge-based selectivity

This adapter does NOT use cavity geometry, donor positions, or any
pocket-paradigm physics. It operates entirely in the network paradigm.

Data source:
    DuPont Tech Fact 45-D01458-en, Rev. 2 (November 2019)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.models import (
    ApplicationContext,
    FabricationSpec,
    NetworkInteractionSpec,
    NetworkMechanism,
    ResinType,
    Solvent,
)
from mabe.realization.adapters.ix_resin_knowledge import (
    DATA_SOURCE,
    SAC_BY_ION,
    SBA_BY_ION,
    WAC_BY_ION,
    WAC_ESTIMATED_ENTRIES,
    RESIN_PROFILES,
    get_sac_selectivity,
    compute_separation_factor,
    recommend_resin_type,
)


# ─────────────────────────────────────────────
# Fabrication spec for IX resin
# ─────────────────────────────────────────────

@dataclass
class IXResinFabricationSpec(FabricationSpec):
    """Layer 4 output for an ion exchange resin design."""

    # ── Resin selection ──
    resin_type: str = ""               # "SAC", "WAC", "SBA"
    functional_group: str = ""
    recommended_dvb_pct: float = 8.0
    form: str = ""                     # "H+", "Na+", "OH-", "Cl-"

    # ── Performance predictions ──
    selectivity_target_vs_H: float = 0.0
    separation_factors: dict = field(default_factory=dict)  # competitor → α
    predicted_capacity_meq_mL: float = 0.0
    predicted_capacity_mg_g: float = 0.0

    # ── Operational parameters ──
    recommended_flow_rate_BV_h: float = 0.0
    regenerant: str = ""
    regenerant_concentration: str = ""
    estimated_BV_to_breakthrough: int = 0

    # ── Data provenance ──
    data_source: dict = field(default_factory=dict)
    data_quality_notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────

class IXResinAdapter:
    """
    Designs an IX resin solution for a NetworkInteractionSpec.

    Scores the target ion's selectivity vs. all competing species,
    recommends resin type and cross-linking, and predicts performance.
    """

    system_id = "ion_exchange_resin"
    supported_spec_types = ["network"]

    def score(self, spec: NetworkInteractionSpec) -> dict:
        """
        Score feasibility and performance of IX approach.

        Returns dict with:
            - feasible: bool
            - selectivity_score: 0.0-1.0 (target preference over competitors)
            - capacity_score: 0.0-1.0 (capacity vs. requirement)
            - kinetics_score: 0.0-1.0 (DVB tradeoff)
            - composite_score: 0.0-1.0
            - confidence: 0.0-1.0
            - rationale: str
        """
        if spec.mechanism not in (
            NetworkMechanism.ION_EXCHANGE,
            NetworkMechanism.CHELATING_RESIN,
        ):
            return {
                "feasible": False,
                "reason": f"IX adapter handles ion_exchange/chelating_resin, not {spec.mechanism.value}",
                "selectivity_score": 0.0,
                "capacity_score": 0.0,
                "kinetics_score": 0.0,
                "composite_score": 0.0,
                "confidence": 0.0,
            }

        # ── Determine resin type ──
        resin_type, form = self._select_resin(spec)
        if resin_type is None:
            return {
                "feasible": False,
                "reason": f"No selectivity data for {spec.target_species}",
                "selectivity_score": 0.0,
                "capacity_score": 0.0,
                "kinetics_score": 0.0,
                "composite_score": 0.0,
                "confidence": 0.0,
            }

        # ── Selectivity scoring ──
        sel_score, sep_factors, data_notes = self._score_selectivity(
            spec, resin_type
        )

        # ── Capacity scoring ──
        cap_score, predicted_cap = self._score_capacity(spec, resin_type)

        # ── Kinetics (DVB tradeoff) ──
        kin_score = self._score_kinetics(spec)

        # ── pH compatibility ──
        ph_ok = self._check_ph(spec, resin_type)
        if not ph_ok:
            return {
                "feasible": False,
                "reason": f"{resin_type} resin incompatible with pH range {spec.pH_range}",
                "selectivity_score": sel_score,
                "capacity_score": cap_score,
                "kinetics_score": kin_score,
                "composite_score": 0.0,
                "confidence": 0.0,
            }

        # ── Composite ──
        # Weights: selectivity dominates for targeted removal,
        # capacity matters for bulk treatment
        if spec.target_application == ApplicationContext.REMEDIATION:
            composite = 0.40 * sel_score + 0.35 * cap_score + 0.25 * kin_score
        else:
            composite = 0.50 * sel_score + 0.30 * cap_score + 0.20 * kin_score

        # ── Confidence ──
        confidence = self._compute_confidence(spec, resin_type, data_notes)

        return {
            "feasible": True,
            "resin_type": resin_type,
            "form": form,
            "selectivity_score": sel_score,
            "separation_factors": sep_factors,
            "capacity_score": cap_score,
            "predicted_capacity_meq_mL": predicted_cap,
            "kinetics_score": kin_score,
            "composite_score": composite,
            "confidence": confidence,
            "data_quality_notes": data_notes,
        }

    def design(self, spec: NetworkInteractionSpec) -> IXResinFabricationSpec:
        """
        Produce a fabrication spec for the IX resin design.
        """
        score_result = self.score(spec)

        resin_type = score_result.get("resin_type", "SAC")
        form = score_result.get("form", "H+")
        profile = RESIN_PROFILES.get(resin_type)

        # ── DVB recommendation ──
        dvb = self._recommend_dvb(spec, resin_type)

        # ── Capacity prediction ──
        cap_meq = score_result.get("predicted_capacity_meq_mL", 0.0)
        # Convert meq/mL to mg/g using target molecular weight estimate
        mw_estimate = self._estimate_target_mw(spec.target_species, spec.target_charge)
        cap_mg_g = cap_meq * mw_estimate / abs(spec.target_charge) if spec.target_charge != 0 else 0.0

        # ── Selectivity for target vs H+ ──
        sel_vs_H = 0.0
        if resin_type == "SAC":
            sel_vs_H = get_sac_selectivity(spec.target_species, dvb) or 0.0

        # ── Build fabrication spec ──
        spec_hash = hashlib.md5(
            f"{spec.target_species}:{resin_type}:{dvb}".encode()
        ).hexdigest()[:12]

        synthesis_steps = [
            f"Select {resin_type} resin, {dvb}% DVB cross-linking, gel type",
            f"Condition resin to {form} form using {profile.regenerant if profile else 'standard regenerant'}",
            f"Pack column with resin bed (BV as required by throughput)",
            f"Pre-equilibrate with feed water matrix at pH {spec.pH_range[0]:.1f}-{spec.pH_range[1]:.1f}",
        ]

        validation = [
            f"Batch isotherm: verify selectivity coefficient for {spec.target_species} at target conditions",
            f"Column breakthrough: measure BV to 10% breakthrough at design flow rate",
            f"Regeneration efficiency: verify >90% recovery over 5 cycles",
        ]
        if spec.competing_species:
            validation.append(
                f"Competitive loading: verify separation from {', '.join(spec.competing_species[:3])}"
            )

        return IXResinFabricationSpec(
            material_system="ion_exchange_resin",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=None,  # No pocket — this is network paradigm
            predicted_deviation_from_ideal_A=0.0,  # N/A for network
            resin_type=resin_type,
            functional_group=profile.functional_group if profile else "",
            recommended_dvb_pct=dvb,
            form=form,
            selectivity_target_vs_H=sel_vs_H,
            separation_factors=score_result.get("separation_factors", {}),
            predicted_capacity_meq_mL=cap_meq,
            predicted_capacity_mg_g=cap_mg_g,
            recommended_flow_rate_BV_h=10.0,  # standard starting point
            regenerant=profile.regenerant if profile else "",
            regenerant_concentration="4-6% HCl or 2-4% H2SO4 (H-cycle), 10% NaCl (Na-cycle)",
            estimated_BV_to_breakthrough=max(100, spec.throughput_BV_per_cycle),
            synthesis_steps=synthesis_steps,
            validation_experiments=validation,
            estimated_cost_per_unit=self._estimate_cost(resin_type, spec.required_scale),
            estimated_time="1-2 days (column setup + conditioning)",
            data_source=DATA_SOURCE,
            data_quality_notes=score_result.get("data_quality_notes", []),
        )

    # ─────────────────────────────────────────
    # Internal methods
    # ─────────────────────────────────────────

    def _select_resin(
        self, spec: NetworkInteractionSpec
    ) -> tuple[Optional[str], str]:
        """Select resin type and ionic form based on target."""
        if spec.resin_type is not None:
            # User specified
            rt = spec.resin_type.value.split("_")[0].upper()
            # Map ResinType enum to our keys
            type_map = {
                "STRONG": "SAC" if spec.target_charge > 0 else "SBA",
                "WEAK": "WAC" if spec.target_charge > 0 else "WBA",
                "CHELATING": "CHELATING",
            }
            for prefix, resin_key in type_map.items():
                if spec.resin_type.value.startswith(prefix.lower()):
                    return resin_key, "H+" if spec.target_charge > 0 else "OH-"
            # Fallback based on enum name
            if "cation" in spec.resin_type.value:
                return "SAC", "H+"
            elif "anion" in spec.resin_type.value:
                return "SBA", "OH-"

        # Auto-select based on target charge
        if spec.target_charge > 0:
            if spec.target_species in SAC_BY_ION:
                return "SAC", "H+"
            return None, ""
        elif spec.target_charge < 0:
            if spec.target_species in SBA_BY_ION:
                return "SBA", "OH-"
            return None, ""
        else:
            return None, ""

    def _score_selectivity(
        self,
        spec: NetworkInteractionSpec,
        resin_type: str,
    ) -> tuple[float, dict, list[str]]:
        """
        Score selectivity of target vs. all competing species.

        Returns (score, separation_factors_dict, data_quality_notes).
        """
        sep_factors = {}
        data_notes = []
        dvb = spec.crosslink_pct

        if resin_type == "SAC":
            k_target = get_sac_selectivity(spec.target_species, dvb)
            if k_target is None:
                return 0.0, {}, [f"{spec.target_species} not in SAC database"]

            for comp in spec.competing_species:
                alpha = compute_separation_factor(spec.target_species, comp, dvb)
                if alpha is not None:
                    sep_factors[comp] = round(alpha, 3)
                else:
                    data_notes.append(f"{comp} not in SAC database — separation unknown")

        elif resin_type == "WAC":
            entry = WAC_BY_ION.get(spec.target_species)
            if entry is None:
                return 0.0, {}, [f"{spec.target_species} not in WAC database"]
            k_target = entry.relative_to_Ca

            if spec.target_species in WAC_ESTIMATED_ENTRIES:
                data_notes.append(
                    f"{spec.target_species} WAC selectivity is estimated (source says '<1' or '>1')"
                )

            for comp in spec.competing_species:
                comp_entry = WAC_BY_ION.get(comp)
                if comp_entry is not None:
                    alpha = k_target / comp_entry.relative_to_Ca if comp_entry.relative_to_Ca > 0 else 0.0
                    sep_factors[comp] = round(alpha, 3)
                    if comp in WAC_ESTIMATED_ENTRIES:
                        data_notes.append(f"{comp} WAC selectivity is estimated")
                else:
                    data_notes.append(f"{comp} not in WAC database")

        elif resin_type == "SBA":
            entry = SBA_BY_ION.get(spec.target_species)
            if entry is None:
                return 0.0, {}, [f"{spec.target_species} not in SBA database"]
            k_target = entry.type1  # default to Type 1

            for comp in spec.competing_species:
                comp_entry = SBA_BY_ION.get(comp)
                if comp_entry is not None:
                    alpha = k_target / comp_entry.type1 if comp_entry.type1 > 0 else 0.0
                    sep_factors[comp] = round(alpha, 3)
                else:
                    data_notes.append(f"{comp} not in SBA database")
        else:
            return 0.0, {}, [f"No selectivity data for resin type {resin_type}"]

        # Score: based on worst-case separation factor
        if not sep_factors:
            # No competitors specified — can only score based on absolute selectivity
            # Higher K means stronger retention
            score = min(1.0, k_target / 10.0)  # normalize: K=10 → score 1.0
        else:
            min_alpha = min(sep_factors.values()) if sep_factors else 0.0
            if min_alpha <= 0:
                score = 0.0
            elif min_alpha >= 5.0:
                score = 1.0   # α ≥ 5 is excellent separation
            else:
                score = min_alpha / 5.0  # linear scale

        return round(score, 3), sep_factors, data_notes

    def _score_capacity(
        self, spec: NetworkInteractionSpec, resin_type: str,
    ) -> tuple[float, float]:
        """Score capacity: can this resin handle the required loading?"""
        profile = RESIN_PROFILES.get(resin_type)
        if profile is None:
            return 0.0, 0.0

        predicted_cap = profile.typical_capacity_meq_mL

        if spec.target_capacity_meq_per_mL <= 0:
            # No specific requirement — full score
            return 1.0, predicted_cap

        ratio = predicted_cap / spec.target_capacity_meq_per_mL
        if ratio >= 1.0:
            score = 1.0
        elif ratio >= 0.5:
            score = ratio  # linear penalty
        else:
            score = 0.0  # fundamentally insufficient

        return round(score, 3), predicted_cap

    def _score_kinetics(self, spec: NetworkInteractionSpec) -> float:
        """
        Score kinetics: higher DVB = higher selectivity but slower kinetics.

        Tradeoff curve: 4% DVB = fast, low selectivity; 16% = slow, high selectivity.
        8% DVB is the standard compromise.
        """
        dvb = spec.crosslink_pct
        if dvb <= 8.0:
            return 1.0  # good kinetics
        elif dvb <= 12.0:
            return 0.8  # acceptable
        else:
            return 0.6  # slow but still functional

    def _check_ph(
        self, spec: NetworkInteractionSpec, resin_type: str,
    ) -> bool:
        """Check pH compatibility."""
        profile = RESIN_PROFILES.get(resin_type)
        if profile is None:
            return True  # unknown resin, don't block

        req_low, req_high = spec.pH_range
        cap_low, cap_high = profile.ph_operating_range

        return req_low >= cap_low and req_high <= cap_high

    def _compute_confidence(
        self,
        spec: NetworkInteractionSpec,
        resin_type: str,
        data_notes: list[str],
    ) -> float:
        """
        Confidence calibration.

        IX is a mature technology — base confidence is high.
        Penalties for: estimated data, missing competitor data, extreme conditions.
        """
        base = 0.85  # IX design-to-validation rate is very high

        # Penalty for estimated/missing data
        n_issues = len([n for n in data_notes if "estimated" in n or "not in" in n])
        base -= n_issues * 0.05

        # Penalty for high ionic strength (competition effects harder to predict)
        if spec.ionic_strength_M > 0.5:
            base -= 0.10

        return max(0.1, round(base, 2))

    def _recommend_dvb(
        self, spec: NetworkInteractionSpec, resin_type: str,
    ) -> float:
        """Recommend DVB cross-linking."""
        if spec.crosslink_pct > 0:
            return spec.crosslink_pct

        # Default logic: heavy metals prefer higher DVB for selectivity
        if resin_type == "SAC" and spec.target_charge == 2:
            return 10.0  # higher selectivity for divalents
        return 8.0  # standard

    def _estimate_target_mw(self, species: str, charge: int) -> float:
        """Rough MW estimate for capacity conversion. Not for precision work."""
        mw_table = {
            "Li+": 6.9, "Na+": 23.0, "K+": 39.1, "NH4+": 18.0,
            "Rb+": 85.5, "Cs+": 132.9, "Ag+": 107.9, "Tl+": 204.4,
            "Mg2+": 24.3, "Ca2+": 40.1, "Sr2+": 87.6, "Ba2+": 137.3,
            "Zn2+": 65.4, "Cu2+": 63.5, "Ni2+": 58.7, "Co2+": 58.9,
            "Cd2+": 112.4, "Pb2+": 207.2, "UO2_2+": 270.0,
            "Cl-": 35.5, "NO3-": 62.0, "SO4_2-": 96.1, "HCO3-": 61.0,
            "F-": 19.0, "Br-": 79.9, "I-": 126.9,
        }
        return mw_table.get(species, 100.0)

    def _estimate_cost(self, resin_type: str, scale) -> float:
        """Rough cost estimate in $/L of resin."""
        cost_per_L = {"SAC": 15.0, "WAC": 25.0, "SBA": 30.0}
        return cost_per_L.get(resin_type, 20.0)