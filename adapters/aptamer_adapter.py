"""
adapters/aptamer_adapter.py - DNA/RNA aptamer adapter for MABE.

Aptamers differ from DNAzymes: they bind targets without catalytic cleavage.
No modification needed for capture mode. Selected through SELEX.
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)
from knowledge.peptide_library import APTAMER_LIBRARY


def _target_matches_aptamer(entry: dict, target_identity: str) -> bool:
    identity = target_identity.lower().strip()
    for alias in entry["target_aliases"]:
        if alias in identity or identity in alias:
            return True
    return False


class AptamerAdapter(ToolAdapter):

    @property
    def name(self) -> str:
        return "aptamer"

    @property
    def version(self) -> str:
        return "0.4.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [Capability(
            description="Search SELEX-validated DNA/RNA aptamers that bind targets without cleaving",
            target_types=["metal_ion", "small_molecule"],
            interaction_types=["binding", "folding"],
            output_types=["nucleic_acid_sequence", "binding_constant"],
        )]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        matches = [e for e in APTAMER_LIBRARY if _target_matches_aptamer(e, problem.target.identity)]
        if matches:
            return ContributionAssessment(
                can_contribute=True, relevance=0.7,
                what_it_would_do=f"Search {len(matches)} validated aptamer(s) for {problem.target.identity}",
                what_part_of_problem="molecular recognition (aptamer modality)",
                estimated_compute_time="instant",
                limitations=["Limited to curated library", "Capture-ready (no modification needed)"],
            )
        return ContributionAssessment(
            can_contribute=False, relevance=0.0,
            what_it_would_do=f"No aptamer in library for {problem.target.identity}",
            what_part_of_problem="none",
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        target = problem.target
        matrix = problem.matrix
        candidates = []

        for entry in APTAMER_LIBRARY:
            if not _target_matches_aptamer(entry, target.identity):
                continue

            ph_ok = True
            if matrix.ph is not None:
                low, high = entry["ph_range"]
                ph_ok = low <= matrix.ph <= high

            base_prob = 0.6
            if entry["kd_um"] < 0.1:
                base_prob = 0.80
            elif entry["kd_um"] < 1.0:
                base_prob = 0.70
            elif entry["kd_um"] < 10.0:
                base_prob = 0.55
            if entry["capture_ready"]:
                base_prob += 0.05
            if entry["environmental_tested"]:
                base_prob += 0.10
            if not ph_ok:
                base_prob *= 0.5
            base_prob = max(0.05, min(0.95, round(base_prob, 2)))

            confidence = "moderate" if entry["environmental_tested"] else "low"
            conf_reason = f"Kd = {entry['kd_um']} uM. {'Environmental tested.' if entry['environmental_tested'] else 'Lab buffer only.'}"

            failure_modes = []
            if not ph_ok:
                failure_modes.append(f"Matrix pH outside optimal range ({entry['ph_range'][0]}-{entry['ph_range'][1]})")
            failure_modes.append("DNase degradation in environmental matrices - consider phosphorothioate backbone")
            if not failure_modes:
                failure_modes = ["No major failure modes identified"]

            # Selectivity threats
            threats = []
            for sel in entry.get("selectivity", []):
                for comp in matrix.competing_species:
                    if comp.identity.lower() == sel.get("competitor_alias", "").lower():
                        if sel["fold"] < 20:
                            threats.append(f"{comp.identity} - selectivity only {sel['fold']}-fold")

            immob = []
            for handle in entry["conjugation"]:
                substrate = "streptavidin beads" if "biotin" in handle.lower() else "nylon netting or silica"
                immob.append(ImmobilizationOption(
                    substrate=substrate,
                    attachment_chemistry=handle,
                    click_handle=handle.split("(")[0].strip(),
                    effect_on_binding="Minimal - terminus attachment, binding site internal",
                ))

            candidates.append(CandidateResult(
                rank=0,
                name=f"{entry['name']} for {target.identity}",
                description=f"{entry['name']} - a SELEX-selected DNA aptamer. Kd = {entry['kd_um']} uM. Capture-ready (binds without cleaving). {entry['notes']}",
                modality="dna_aptamer",
                source_tool="aptamer",
                structure_description=f"{entry['sequence']} | Kd={entry['kd_um']} uM",
                performance=PerformancePrediction(
                    probability_of_success=base_prob,
                    confidence=confidence,
                    confidence_reasoning=conf_reason,
                    sensitive_to=[f"pH range: {entry['ph_range'][0]}-{entry['ph_range'][1]}"],
                    failure_modes=failure_modes,
                    what_improves_odds=["Use phosphorothioate backbone for nuclease resistance"],
                    selectivity_threats=threats if threats else ["No major selectivity threats"],
                ),
                evidence=EvidenceProfile(
                    source_type="literature_validated" if entry["environmental_tested"] else "hybrid",
                    literature_references=[f"DOI: {entry['doi']} ({entry['year']})"],
                    computational_method="Library search",
                    what_would_validate="Test in target matrix, measure Kd shift",
                ),
                accessibility=AccessibilityProfile(
                    estimated_cost=entry["cost_per_synthesis"] + " per synthesis",
                    equipment_required=["thermal cycler (optional)", "gel electrophoresis"],
                    community_lab_feasible=True,
                    reusability_cycles=15,
                    waste_generated="DNA is biodegradable",
                    end_of_life="Fully biodegradable",
                ),
                immobilization_options=immob,
                other_applications=[ApplicationConnection(
                    domain="diagnostic",
                    description=f"Aptamer sensor for {entry['primary_target']} detection",
                    what_would_change="Add fluorophore/quencher for signal readout",
                    confidence="strong",
                )],
            ))

        candidates.sort(key=lambda c: c.performance.probability_of_success, reverse=True)
        return candidates
