"""
adapters/peptide_adapter.py - Peptide chelator adapter for MABE.

Searches curated peptide library for metal-binding peptides matching the target.
Includes phytochelatins, metallothionein fragments, His-tags, and selected peptides.
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)
from knowledge.peptide_library import PEPTIDE_LIBRARY


def _target_matches_peptide(entry: dict, target_identity: str) -> bool:
    identity = target_identity.lower().strip()
    for alias, ion in entry["target_aliases"].items():
        if alias in identity or identity in alias:
            return True
    return False


def _get_kd_for_target(entry: dict, target_identity: str):
    identity = target_identity.lower().strip()
    for alias, ion in entry["target_aliases"].items():
        if alias in identity or identity in alias:
            if ion in entry["kd_data"]:
                return entry["kd_data"][ion], ion
    return None, None


class PeptideAdapter(ToolAdapter):

    @property
    def name(self) -> str:
        return "peptide_chelator"

    @property
    def version(self) -> str:
        return "0.4.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [Capability(
            description="Search validated metal-binding peptides: phytochelatins, metallothioneins, His-tags, selected peptides",
            target_types=["metal_ion"],
            interaction_types=["coordination", "chelation"],
            output_types=["peptide_sequence", "binding_constant"],
        )]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        matches = [e for e in PEPTIDE_LIBRARY if _target_matches_peptide(e, problem.target.identity)]
        if matches:
            return ContributionAssessment(
                can_contribute=True, relevance=0.75,
                what_it_would_do=f"Search {len(matches)} metal-binding peptides for {problem.target.identity}",
                what_part_of_problem="molecular recognition (peptide modality)",
                estimated_compute_time="instant",
                limitations=["Limited to curated library", "Does not design novel peptides"],
            )
        return ContributionAssessment(
            can_contribute=False, relevance=0.0,
            what_it_would_do=f"No peptides in library for {problem.target.identity}",
            what_part_of_problem="none",
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        target = problem.target
        matrix = problem.matrix
        candidates = []

        for entry in PEPTIDE_LIBRARY:
            if not _target_matches_peptide(entry, target.identity):
                continue

            kd, ion_matched = _get_kd_for_target(entry, target.identity)

            # pH check
            ph_ok = True
            ph_note = ""
            if matrix.ph is not None:
                low, high = entry["ph_range"]
                if not (low <= matrix.ph <= high):
                    ph_ok = False
                    ph_note = f"Matrix pH {matrix.ph} outside optimal range ({low}-{high})"

            # HSAB match
            target_hsab = target.electronic.hardness_softness or "unknown"
            hsab_match = 1.0
            if target_hsab != "unknown":
                match_map = {
                    ("soft", "soft"): 1.0, ("soft", "borderline"): 0.6, ("soft", "hard"): 0.2,
                    ("borderline", "borderline"): 0.8, ("borderline", "soft"): 0.6, ("borderline", "hard"): 0.6,
                    ("hard", "hard"): 1.0, ("hard", "borderline"): 0.6, ("hard", "soft"): 0.2,
                }
                hsab_match = match_map.get((entry["donor_type"], target_hsab), 0.5)

            # Probability
            base_prob = 0.5
            if kd is not None:
                if kd < 0.01:
                    base_prob = 0.85
                elif kd < 0.1:
                    base_prob = 0.75
                elif kd < 1.0:
                    base_prob = 0.65
                elif kd < 10.0:
                    base_prob = 0.55
                else:
                    base_prob = 0.40
            base_prob *= hsab_match
            if not ph_ok:
                base_prob *= 0.5
            if entry["environmental_tested"]:
                base_prob = min(base_prob + 0.10, 0.95)
            base_prob = max(0.05, min(0.95, round(base_prob, 2)))

            # Confidence
            if kd is not None and entry["environmental_tested"]:
                confidence = "moderate"
                conf_reason = f"Kd = {kd} uM for {ion_matched} (measured). Tested in environmental matrix."
            elif kd is not None:
                confidence = "moderate" if hsab_match > 0.6 else "low"
                conf_reason = f"Kd = {kd} uM for {ion_matched} (measured). Lab buffer validation."
            else:
                confidence = "low"
                conf_reason = f"Target in peptide's metal list but no specific Kd measured."

            # Failure modes
            failure_modes = []
            if entry["donor_type"] == "soft":
                failure_modes.append("Thiol oxidation in aerobic conditions (Cys disulfide formation)")
            if not ph_ok and ph_note:
                failure_modes.append(ph_note)
            if entry["class"] == "metallothionein":
                failure_modes.append("Proteolytic degradation in matrices with microbial activity")
            if not failure_modes:
                failure_modes.append("No major failure modes identified")

            # Improvements
            improvements = []
            if entry["donor_type"] == "soft":
                improvements.append("Handle under N2/argon to prevent Cys oxidation")
            if not ph_ok and matrix.ph is not None:
                mid = (entry["ph_range"][0] + entry["ph_range"][1]) / 2
                improvements.append(f"Buffer to pH {mid:.1f}")
            if entry["class"] == "his_tag":
                improvements.append("Elute with 250 mM imidazole for clean release and reuse")

            # Build structure description
            kd_str = f"Kd={kd} uM for {ion_matched}" if kd else "Kd not measured for this target"
            struct_desc = (
                f"{entry['sequence']} ({entry['full_sequence']}) | "
                f"{entry['length']} residues, {'/'.join(entry['donor_atoms'])} donors, "
                f"{kd_str}"
            )

            # Immobilization
            immob = []
            for handle in entry["conjugation"]:
                immob.append(ImmobilizationOption(
                    substrate="nylon netting or silica beads",
                    attachment_chemistry=handle,
                    click_handle=handle.split("(")[0].strip(),
                    effect_on_binding="Minimal with C6+ linker arm",
                ))

            # Cross-domain
            other_apps = []
            if entry["class"] == "phytochelatin":
                other_apps.append(ApplicationConnection(
                    domain="bioremediation",
                    description="Express in bacteria/plants for in vivo metal sequestration",
                    what_would_change="Clone encoding gene into expression vector",
                    confidence="strong",
                ))
            if entry["class"] == "his_tag":
                other_apps.append(ApplicationConnection(
                    domain="protein_purification",
                    description="Same chemistry used in IMAC purification - massive existing infrastructure",
                    what_would_change="Already standard technology",
                    confidence="strong",
                ))

            candidates.append(CandidateResult(
                rank=0,
                name=f"{entry['name']} for {target.identity}",
                description=(
                    f"{entry['name']} - a {entry['class']} ({entry['length']} residues) with "
                    f"{'/'.join(entry['donor_atoms'])} donor atoms ({entry['donor_type']}). "
                    f"{entry['notes']}"
                ),
                modality="peptide_chelator",
                source_tool="peptide_chelator",
                structure_description=struct_desc,
                performance=PerformancePrediction(
                    probability_of_success=base_prob,
                    confidence=confidence,
                    confidence_reasoning=conf_reason,
                    sensitive_to=[f"pH range: {entry['ph_range'][0]}-{entry['ph_range'][1]}"],
                    failure_modes=failure_modes,
                    what_improves_odds=improvements if improvements else ["Conditions appear favorable"],
                    selectivity_threats=[],
                ),
                evidence=EvidenceProfile(
                    source_type="hybrid" if kd else "computational_prediction",
                    literature_references=[f"DOI: {entry['doi']} ({entry['year']})"],
                    computational_method="Library search + HSAB/Kd scoring",
                    what_would_validate=f"Measure Kd in target matrix by ITC or SPR",
                ),
                accessibility=AccessibilityProfile(
                    estimated_cost=entry["cost_per_mg"] + "/mg (peptide synthesis service)",
                    equipment_required=["analytical balance", "pH meter"],
                    community_lab_feasible=entry["community_lab"],
                    reusability_cycles=entry["reusability"],
                    waste_generated="Peptide is biodegradable - minimal waste",
                    end_of_life="Fully biodegradable",
                ),
                immobilization_options=immob,
                other_applications=other_apps,
            ))

        candidates.sort(key=lambda c: c.performance.probability_of_success, reverse=True)
        return candidates
