"""
adapters/dummy_adapter.py - Mock adapter that returns realistic fake results.
Proves the pipeline works end-to-end. Replace with real adapters one at a time.
"""

from __future__ import annotations

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import (
    CandidateResult, PerformancePrediction, EvidenceProfile,
    AccessibilityProfile, ImmobilizationOption, ApplicationConnection,
)


class DummyAdapter(ToolAdapter):

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [Capability(
            description="Generates mock candidates for pipeline testing",
            target_types=["metal_ion", "small_molecule", "protein"],
            interaction_types=["coordination", "encapsulation", "covalent"],
            output_types=["molecular_structure"],
        )]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        return ContributionAssessment(
            can_contribute=True,
            relevance=0.5,
            what_it_would_do="Generate mock candidates for pipeline testing",
            what_part_of_problem="all (mock data)",
            estimated_compute_time="instant",
            limitations=["This is a dummy adapter - results are not real predictions"],
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        target = problem.target.identity
        hsab = problem.target.electronic.hardness_softness or "unknown"

        candidates = []

        # Candidate 1: Chelator
        candidates.append(CandidateResult(
            rank=0,
            name=f"Thiol-functionalized chelator for {target}",
            description=(
                f"A small-molecule chelator with thiol donor groups designed for "
                f"{target} ({problem.target.formula}). Thiol sulfur donors provide "
                f"soft-base coordination suitable for {hsab} metal centers. "
                f"Predicted selectivity driven by ionic radius matching and "
                f"charge density complementarity."
            ),
            modality="chelator",
            source_tool="dummy",
            structure_description="[mock] HSCH2CH2N(CH2COOH)CH2CH2SH - dithiol-NTA hybrid",
            performance=PerformancePrediction(
                probability_of_success=0.65,
                confidence="moderate",
                confidence_reasoning=(
                    "Thiol-metal coordination is well-established for soft/borderline metals. "
                    "Selectivity over competing ions in complex matrices is less certain."
                ),
                sensitive_to=[
                    "pH below 3 - thiol protonation reduces donor availability",
                    "Oxidizing conditions - disulfide formation deactivates thiol groups",
                ],
                failure_modes=[
                    "Thiol oxidation in aerobic mine water (dissolved O2 > 2 mg/L)",
                    "Biofouling of thiol groups by natural organic matter",
                ],
                what_improves_odds=[
                    "Pre-filter to remove dissolved organics",
                    "Deoxygenate water or add mild reducing agent (ascorbate)",
                    "Buffer pH to 5.0-6.0 for optimal thiol availability",
                ],
                selectivity_threats=[
                    "Copper(II) competes strongly for thiol donors",
                ],
            ),
            evidence=EvidenceProfile(
                source_type="computational_prediction",
                computational_method="Mock - rule-based HSAB matching",
                what_would_validate="ITC binding assay with target vs top 3 competing ions",
            ),
            accessibility=AccessibilityProfile(
                estimated_cost="~$30/g synthesized, commodity reagents",
                equipment_required=["fume hood", "rotary evaporator", "pH meter"],
                community_lab_feasible=True,
                reusability_cycles=50,
                waste_generated="Spent eluent containing captured metal - recover by electrodeposition",
                end_of_life="Biodegradable organic backbone",
            ),
            immobilization_options=[
                ImmobilizationOption(
                    substrate="nylon netting",
                    attachment_chemistry="NHS-amine coupling to nylon surface amines",
                    click_handle="primary amine on chelator backbone",
                    effect_on_binding="~15% reduction in binding kinetics, selectivity unchanged",
                ),
                ImmobilizationOption(
                    substrate="glass beads (column packing)",
                    attachment_chemistry="silanization + NHS-amine coupling",
                    click_handle="primary amine on chelator backbone",
                    effect_on_binding="Minimal - glass surface is inert, full rotational freedom on C6 linker",
                ),
            ],
            other_applications=[
                ApplicationConnection(
                    domain="therapeutic",
                    description=f"If selectivity for {target} holds in biological fluid, could serve as blood detoxification agent",
                    what_would_change="Add PEG linker for renal clearance, test in serum matrix",
                    confidence="plausible",
                ),
            ],
        ))

        # Candidate 2: Designed protein
        candidates.append(CandidateResult(
            rank=0,
            name=f"De novo protein binding pocket for {target}",
            description=(
                f"A computationally designed protein (~80 residues) with an interior "
                f"binding pocket shaped for {target} ({problem.target.formula}). "
                f"Pocket positions coordinating residues (Cys, His, Asp) to "
                f"complement the target charge density and coordination geometry."
            ),
            modality="designed_protein",
            source_tool="dummy",
            structure_description="[mock] 80-residue 4-helix bundle, Cys2His2 binding site",
            performance=PerformancePrediction(
                probability_of_success=0.45,
                confidence="low",
                confidence_reasoning=(
                    "De novo protein design for metal binding is emerging but less validated. "
                    "RFDiffusion3 shows success for protein-protein but metal pockets less tested."
                ),
                sensitive_to=[
                    "Temperature above 60C - protein denaturation",
                    "Extreme pH (<3 or >10) - loss of tertiary structure",
                ],
                failure_modes=[
                    "Protein misfolding during expression",
                    "Insufficient selectivity - pocket may accept similar-sized ions",
                    "Proteolytic degradation in matrices with microbial activity",
                ],
                what_improves_odds=[
                    "Express multiple designs and screen - expect ~20% success rate",
                    "Thermostabilize with disulfide bridges",
                    "PEGylate surface to resist proteolysis",
                ],
                selectivity_threats=[
                    "Similar-radius ions with same charge may fit the pocket",
                ],
            ),
            evidence=EvidenceProfile(
                source_type="computational_prediction",
                computational_method="Mock - RFDiffusion3 backbone + ProteinMPNN sequence",
                what_would_validate="Express in E. coli, measure Kd by ITC against target and 5 competitors",
            ),
            accessibility=AccessibilityProfile(
                estimated_cost="~$500 for gene synthesis + expression, then ~$5/mg protein",
                equipment_required=["molecular biology lab", "FPLC or gravity column", "incubator"],
                community_lab_feasible=False,
                reusability_cycles=200,
                waste_generated="Minimal - protein is biodegradable",
                end_of_life="Fully biodegradable",
            ),
            immobilization_options=[
                ImmobilizationOption(
                    substrate="nylon netting",
                    attachment_chemistry="NHS-amine coupling to surface lysines",
                    click_handle="surface-exposed lysine residues",
                    effect_on_binding="Minimal if binding pocket is on opposite face",
                ),
            ],
            other_applications=[
                ApplicationConnection(
                    domain="diagnostic",
                    description=f"Protein binder as recognition element in electrochemical sensor for {target}",
                    what_would_change="Add redox-active label, optimize for signal generation",
                    confidence="strong",
                ),
            ],
        ))

        # Candidate 3: Nanocage
        candidates.append(CandidateResult(
            rank=0,
            name=f"Interior-decorated DNA nanocage for {target}",
            description=(
                f"A 40nm wireframe DNA origami icosahedron with interior-facing staple "
                f"overhangs functionalized with {target}-selective recognition chemistry. "
                f"Pore size (~8nm) admits target while excluding larger competing complexes. "
                f"Loaded cages have altered charge/magnetic profile for field-driven sorting."
            ),
            modality="nanocage",
            source_tool="dummy",
            structure_description="[mock] 40nm DNA origami icosahedron, M13mp18 scaffold, 200 staples, 12 interior sites",
            performance=PerformancePrediction(
                probability_of_success=0.35,
                confidence="low",
                confidence_reasoning=(
                    "DNA origami cages are structurally well-established. Interior decoration "
                    "for selective metal capture is novel - limited published validation."
                ),
                sensitive_to=[
                    "Ionic strength below 5mM - cage destabilization",
                    "DNase activity in environmental matrices",
                    "Temperature above 50C",
                ],
                failure_modes=[
                    "Cage assembly yield may be low (<50%)",
                    "Interior binding sites may be sterically occluded after folding",
                    "Pore size may not provide sufficient selectivity alone",
                ],
                what_improves_odds=[
                    "Optimize Mg2+ concentration for cage stability",
                    "Use 6HB edges for tighter pore control",
                    "Add PEG passivation to exterior",
                ],
                selectivity_threats=[
                    "Small competing ions pass through same pores - interior chemistry must discriminate",
                ],
            ),
            evidence=EvidenceProfile(
                source_type="computational_prediction",
                computational_method="Mock - ATHENA cage geometry + rule-based interior placement",
                what_would_validate="Fold cage, verify by TEM, measure uptake by ICP-MS",
            ),
            accessibility=AccessibilityProfile(
                estimated_cost="~$2000 first assembly (staple set), then ~$50/assembly",
                equipment_required=["thermal cycler", "gel electrophoresis", "TEM access"],
                community_lab_feasible=False,
                reusability_cycles=10,
                waste_generated="DNA is biodegradable - minimal waste",
                end_of_life="Fully biodegradable",
            ),
            immobilization_options=[
                ImmobilizationOption(
                    substrate="magnetic beads (batch pulldown)",
                    attachment_chemistry="Biotin-streptavidin on exterior staple overhangs",
                    click_handle="3'-biotin on selected exterior staples",
                    effect_on_binding="None - interior chemistry independent of exterior attachment",
                ),
            ],
            other_applications=[
                ApplicationConnection(
                    domain="drug_delivery",
                    description="Same cage with interior therapeutic payload and exterior targeting ligands",
                    what_would_change="Swap interior chemistry to drug encapsulation, add targeting aptamers",
                    confidence="strong",
                ),
                ApplicationConnection(
                    domain="manufacturing",
                    description=f"Loaded cages with {target} cargo have distinct field profile for patterned deposition",
                    what_would_change="Design exterior charge profile for electrophoretic deposition",
                    confidence="speculative",
                ),
            ],
        ))

        # Rank by public-good-weighted score
        for c in candidates:
            reuse_score = min((c.accessibility.reusability_cycles or 0) / 100, 1.0)
            cost_score = 1.0 if c.accessibility.community_lab_feasible else 0.3
            c._sort_score = (
                c.performance.probability_of_success * 0.5 +
                cost_score * 0.3 +
                reuse_score * 0.2
            )

        candidates.sort(key=lambda c: c._sort_score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1

        return candidates
