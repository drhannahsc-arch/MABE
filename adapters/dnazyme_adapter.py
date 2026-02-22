"""
adapters/dnazyme_adapter.py - DNAzyme and DNA motif adapter for MABE.

Searches curated DNAzyme library for sequences matching the target metal.
Returns candidates with real Kd values, selectivity data, and literature evidence.

DNAzymes are unique: they are selected through SELEX (evolution), not designed
computationally. So this adapter searches existing validated sequences rather
than generating new ones. The physics reasoning is in matching the target
to the right DNAzyme, evaluating matrix compatibility, and identifying
modifications needed for capture vs sensing mode.
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)
from knowledge.dnazyme_library import DNAZYME_LIBRARY


def _target_matches(entry: dict, target_identity: str) -> bool:
    """Check if a DNAzyme library entry matches the target."""
    identity = target_identity.lower().strip()
    # Check primary target
    if identity in entry["primary_target"].lower():
        return True
    # Check aliases
    for alias in entry["target_aliases"]:
        if alias in identity or identity in alias:
            return True
    return False


def _ph_compatible(entry: dict, ph: float) -> tuple[bool, str]:
    """Check if matrix pH is compatible."""
    if ph is None:
        return True, ""
    low, high = entry["ph_range"]
    if low <= ph <= high:
        return True, ""
    elif abs(ph - low) <= 0.5 or abs(ph - high) <= 0.5:
        return True, f"pH {ph} is marginal (optimal: {low}-{high})"
    else:
        return False, f"pH {ph} is outside operational range ({low}-{high})"


def _ionic_strength_compatible(entry: dict, ionic_mm: float) -> tuple[bool, str]:
    """Check ionic strength compatibility."""
    if ionic_mm is None:
        return True, ""
    low, high = entry["ionic_strength_range_mm"]
    if low <= ionic_mm <= high:
        return True, ""
    else:
        return False, f"Ionic strength {ionic_mm} mM outside range ({low}-{high} mM)"


def _check_incompatibilities(entry: dict, matrix) -> list[str]:
    """Check for fatal or performance-reducing incompatibilities."""
    issues = []
    matrix_desc = (matrix.description + " " + matrix.notes).lower()

    for incompat in entry["incompatibilities"]:
        condition = incompat["condition"].lower()
        # Check if matrix description hints at the incompatible condition
        keywords = condition.replace("-", " ").replace("_", " ").split()
        for kw in keywords:
            if len(kw) > 3 and kw in matrix_desc:
                severity_label = "FATAL" if incompat["severity"] == "fatal" else "Warning"
                issues.append(f"{severity_label}: {incompat['condition']} - {incompat['effect']}")
                break

    return issues


def _selectivity_vs_matrix(entry: dict, competing_species: list) -> list[str]:
    """Evaluate selectivity against actual competing species in the matrix."""
    threats = []
    for comp in competing_species:
        comp_name = comp.identity.lower()
        for sel in entry["selectivity"]:
            if comp_name == sel.get("competitor_alias", "").lower():
                fold = sel["fold"]
                if fold < 10:
                    threats.append(
                        f"{comp.identity} ({comp.formula}) - POOR selectivity ({fold}-fold) "
                        f"at {comp.concentration_mm} mM in matrix"
                    )
                elif fold < 100:
                    threats.append(
                        f"{comp.identity} ({comp.formula}) - moderate selectivity ({fold}-fold), "
                        f"may interfere at {comp.concentration_mm} mM"
                    )
                # >100-fold is generally fine, don't flag
    return threats


def _compute_confidence(entry: dict, ph_ok: bool, is_ok: bool,
                         threats: list, issues: list) -> tuple[str, str]:
    """Determine confidence level and reasoning."""
    tier = entry["validation_tier"]
    env_tested = entry["environmental_tested"]

    if any("FATAL" in i for i in issues):
        return "speculative", "Fatal incompatibility with matrix conditions detected."

    if tier >= 4 and env_tested and ph_ok and is_ok and not threats:
        return "high", (
            f"Validation tier {tier} (environmental matrix tested). "
            f"Kd = {entry['kd_um']} uM ({entry['kd_confidence']}). "
            f"Matrix conditions compatible."
        )
    elif tier >= 3 and ph_ok and is_ok:
        confidence = "moderate" if not threats else "low"
        return confidence, (
            f"Validation tier {tier} ({'environmental tested' if env_tested else 'lab buffer only'}). "
            f"Kd = {entry['kd_um']} uM. "
            f"{'Some selectivity concerns.' if threats else 'Matrix compatible.'}"
        )
    else:
        return "low", (
            f"Validation tier {tier}. Limited validation data. "
            f"Kd = {entry['kd_um']} uM ({entry['kd_confidence']}). "
            f"{'Matrix issues detected.' if issues else ''}"
        )


class DNAzymeAdapter(ToolAdapter):
    """
    Searches curated DNAzyme/DNA motif library for validated metal-binding sequences.
    """

    @property
    def name(self) -> str:
        return "dnazyme"

    @property
    def version(self) -> str:
        return "0.3.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                description="Search validated DNAzyme and DNA motif sequences for metal-selective binding",
                target_types=["metal_ion"],
                interaction_types=["catalytic_pocket", "mismatch_pair", "coordination"],
                output_types=["nucleic_acid_sequence", "binding_constant", "selectivity_panel"],
            ),
        ]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        # Check if any library entry matches the target
        matches = [e for e in DNAZYME_LIBRARY if _target_matches(e, problem.target.identity)]
        if matches:
            return ContributionAssessment(
                can_contribute=True,
                relevance=0.9,
                what_it_would_do=(
                    f"Search {len(matches)} validated DNAzyme/DNA motif sequences "
                    f"for {problem.target.identity} with real Kd values and selectivity data"
                ),
                what_part_of_problem="molecular recognition (nucleic acid modality)",
                estimated_compute_time="instant",
                limitations=[
                    "Limited to existing SELEX-validated sequences",
                    "Capture mode requires modification of sensor sequences",
                    "Does not generate novel sequences (would need SELEX)",
                ],
            )
        else:
            return ContributionAssessment(
                can_contribute=False,
                relevance=0.0,
                what_it_would_do=f"No validated DNAzyme found for {problem.target.identity}",
                what_part_of_problem="none",
                limitations=[f"No DNAzyme in library for {problem.target.identity}. Could be discovered via SELEX."],
            )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        target = problem.target
        matrix = problem.matrix
        candidates = []

        for entry in DNAZYME_LIBRARY:
            if not _target_matches(entry, target.identity):
                continue

            # Evaluate compatibility
            ph_ok, ph_note = _ph_compatible(entry, matrix.ph)
            is_ok, is_note = _ionic_strength_compatible(entry, matrix.ionic_strength_mm)
            issues = _check_incompatibilities(entry, matrix)
            threats = _selectivity_vs_matrix(entry, matrix.competing_species)

            # Skip if fatal incompatibility
            fatal = any("FATAL" in i for i in issues)

            # Confidence
            confidence, conf_reason = _compute_confidence(
                entry, ph_ok, is_ok, threats, issues
            )

            # Probability
            base_prob = 0.7 if entry["capture_validated"] else 0.5
            if entry["validation_tier"] >= 4:
                base_prob += 0.15
            elif entry["validation_tier"] >= 3:
                base_prob += 0.10
            if entry["environmental_tested"]:
                base_prob += 0.10
            if not ph_ok:
                base_prob -= 0.25
            if not is_ok:
                base_prob -= 0.15
            if fatal:
                base_prob = 0.05
            for t in threats:
                if "POOR" in t:
                    base_prob -= 0.20
                elif "moderate" in t:
                    base_prob -= 0.10
            base_prob = max(0.05, min(0.95, base_prob))

            # Failure modes
            failure_modes = []
            if not entry["capture_validated"]:
                failure_modes.append(
                    f"Capture mode not yet validated - sensor modification "
                    f"({entry['capture_modification'] or 'unknown'}) needs experimental confirmation"
                )
            if ph_note:
                failure_modes.append(ph_note)
            if is_note:
                failure_modes.append(is_note)
            for issue in issues:
                failure_modes.append(issue)
            if not failure_modes:
                failure_modes.append("No major failure modes identified")

            # Improvements
            improvements = []
            if not ph_ok and matrix.ph is not None:
                mid_ph = (entry["ph_range"][0] + entry["ph_range"][1]) / 2
                improvements.append(f"Buffer to pH {mid_ph:.1f} for optimal DNAzyme function")
            if threats:
                improvements.append("Pre-treatment to reduce competing ion concentrations")
            if not entry["capture_validated"]:
                improvements.append("Validate capture modification before scaling")

            # Evidence
            lit_refs = [f"DOI: {entry['doi']} ({entry['year']}, {entry['lab']})"]
            if entry["environmental_tested"]:
                evidence_type = "literature_validated"
            elif entry["validation_tier"] >= 3:
                evidence_type = "hybrid"
            else:
                evidence_type = "computational_prediction"

            # Selectivity description
            sel_str = ", ".join(
                f"{s['competitor']} ({s['fold']}x)"
                for s in entry["selectivity"][:3]
            )

            # Description
            is_motif = entry["modality"] == "dna_motif"
            modality_label = "DNA motif" if is_motif else "DNAzyme"
            capture_note = (
                "Capture-ready (no modification needed)." if entry["capture_validated"]
                else f"Sensor mode - requires modification for capture: {entry['capture_modification'] or 'TBD'}."
            )

            description = (
                f"{entry['name']} - a {modality_label} with validated selectivity for "
                f"{entry['primary_target']}. Kd = {entry['kd_um']} uM. "
                f"Selectivity: {sel_str}. {capture_note} "
                f"{entry['notes']}"
            )

            # Cost estimate
            seq_len = len(entry["sequence"].replace("5\'", "").replace("3\'", "").replace("-", ""))
            base_cost = max(20, seq_len * 0.5)  # ~$0.50/base for modified oligos
            if entry["capture_modification"] and "OMe" in str(entry["capture_modification"]):
                base_cost += 50  # 2'-OMe modification surcharge
            cost_str = f"~${base_cost:.0f} per synthesis (modified oligo)"

            # Immobilization
            immob_options = []
            for handle in entry["conjugation_handles"]:
                if "amine" in handle.lower():
                    immob_options.append(ImmobilizationOption(
                        substrate="nylon netting or glass beads",
                        attachment_chemistry="NHS-amine coupling",
                        click_handle=handle,
                        effect_on_binding="Minimal - attachment at terminus, binding pocket internal",
                    ))
                if "thiol" in handle.lower():
                    immob_options.append(ImmobilizationOption(
                        substrate="gold surface or maleimide-functionalized beads",
                        attachment_chemistry="Thiol-gold or maleimide-thiol coupling",
                        click_handle=handle,
                        effect_on_binding="Minimal - standard DNA surface chemistry",
                    ))
                if "biotin" in handle.lower():
                    immob_options.append(ImmobilizationOption(
                        substrate="streptavidin-coated beads or plates",
                        attachment_chemistry="Biotin-streptavidin",
                        click_handle=handle,
                        effect_on_binding="None - biotin-streptavidin is orthogonal to metal binding",
                    ))

            # Cross-domain connections
            other_apps = [
                ApplicationConnection(
                    domain="diagnostic",
                    description=(
                        f"Original sensor function: fluorescent or electrochemical "
                        f"{entry['primary_target']} detection at nM-uM levels"
                    ),
                    what_would_change="Use native sequence without capture modification, add fluorophore/quencher pair",
                    confidence="strong",
                ),
            ]
            if entry["capture_validated"]:
                other_apps.append(ApplicationConnection(
                    domain="nanocage_interior",
                    description="Attach to interior of DNA origami nanocage as selective recognition element",
                    what_would_change="Extend with staple-compatible overhang sequence",
                    confidence="plausible",
                ))

            candidates.append(CandidateResult(
                rank=0,
                name=f"{entry['name']} for {target.identity}",
                description=description,
                modality=modality_label.lower().replace(" ", "_"),
                source_tool="dnazyme",
                structure_description=f"{entry['sequence']} | Kd={entry['kd_um']} uM, {entry['stoichiometry']}",
                performance=PerformancePrediction(
                    probability_of_success=round(base_prob, 2),
                    confidence=confidence,
                    confidence_reasoning=conf_reason,
                    sensitive_to=[
                        f"pH range: {entry['ph_range'][0]}-{entry['ph_range'][1]}",
                        f"Temperature: {entry['temp_range_c'][0]}-{entry['temp_range_c'][1]} C",
                        f"Ionic strength: {entry['ionic_strength_range_mm'][0]}-{entry['ionic_strength_range_mm'][1]} mM",
                    ],
                    failure_modes=failure_modes,
                    what_improves_odds=improvements if improvements else ["Conditions appear favorable"],
                    selectivity_threats=threats if threats else ["No major selectivity threats from matrix species"],
                ),
                evidence=EvidenceProfile(
                    source_type=evidence_type,
                    literature_references=lit_refs,
                    computational_method="Library search with matrix compatibility evaluation",
                    what_would_validate=(
                        f"Test capture modification in target matrix. Measure Kd shift vs native sensor. "
                        f"Selectivity panel in actual mine/environmental water."
                    ),
                ),
                accessibility=AccessibilityProfile(
                    estimated_cost=cost_str,
                    equipment_required=["thermal cycler (optional)", "gel electrophoresis", "UV-Vis or fluorimeter"],
                    community_lab_feasible=True,
                    reusability_cycles=20 if entry["capture_validated"] else 10,
                    waste_generated="DNA is biodegradable - minimal waste",
                    end_of_life="Fully biodegradable (nucleic acid)",
                ),
                immobilization_options=immob_options,
                other_applications=other_apps,
            ))

        # Sort by probability
        candidates.sort(key=lambda c: c.performance.probability_of_success, reverse=True)
        return candidates
