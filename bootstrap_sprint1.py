"""
MABE Sprint 1 Bootstrap
========================
Run this ONE file and it creates the entire project.

Usage:
    cd Documents
    mkdir mabe
    cd mabe
    python bootstrap_sprint1.py
    python tests/test_sprint1.py
    python main.py
"""

import os

def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print()
print("  MABE Sprint 1 — Bootstrap")
print("  " + "=" * 40)
print()

# ═══════════════════════════════════════════════════════════════════════════
# __init__.py files
# ═══════════════════════════════════════════════════════════════════════════

for pkg in ["core", "adapters", "conversation", "knowledge", "tests"]:
    write_file(f"{pkg}/__init__.py", "")

# ═══════════════════════════════════════════════════════════════════════════
# core/problem.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/problem.py", '''"""
core/problem.py - MABE internal representation of molecular design problems.

Everything here describes physics, not chemistry categories.
A Problem is a question about energy landscapes, not about binders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ElectronicDescription:
    """Electronic structure relevant to molecular interactions."""
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    polarizability: Optional[float] = None
    electronegativity: Optional[float] = None
    hardness_softness: Optional[str] = None
    donor_atoms: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class HydrationDescription:
    """How the target interacts with water."""
    hydrated_radius_angstrom: Optional[float] = None
    dehydration_energy_kj_mol: Optional[float] = None
    coordination_number_water: Optional[int] = None
    notes: str = ""


@dataclass
class RedoxState:
    """One possible oxidation state and its properties."""
    oxidation_state: int
    formula: str
    stable_ph_range: Optional[tuple[float, float]] = None
    standard_potential_v: Optional[float] = None
    notes: str = ""


@dataclass
class MagneticDescription:
    """Magnetic properties - determines field response when captured."""
    type: str = "diamagnetic"
    susceptibility: Optional[float] = None
    unpaired_electrons: int = 0


@dataclass
class SizeDescription:
    """Size at different levels."""
    ionic_radius_angstrom: Optional[float] = None
    hydrated_radius_angstrom: Optional[float] = None
    vdw_radius_angstrom: Optional[float] = None
    molecular_weight: Optional[float] = None


@dataclass
class TargetSpecies:
    """A molecular species described by its physics, not its name."""
    identity: str
    formula: str
    charge: float
    geometry: str
    electronic: ElectronicDescription = field(default_factory=ElectronicDescription)
    hydration: HydrationDescription = field(default_factory=HydrationDescription)
    redox_states: list[RedoxState] = field(default_factory=list)
    magnetic: MagneticDescription = field(default_factory=MagneticDescription)
    size: SizeDescription = field(default_factory=SizeDescription)
    notes: str = ""

    def summary(self) -> str:
        parts = [f"{self.identity} ({self.formula}), charge {self.charge:+.0f}"]
        if self.geometry:
            parts.append(self.geometry)
        if self.electronic.hardness_softness:
            parts.append(f"HSAB: {self.electronic.hardness_softness}")
        if self.size.ionic_radius_angstrom:
            parts.append(f"r={self.size.ionic_radius_angstrom} A")
        return ", ".join(parts)


@dataclass
class CompetingSpecies:
    """Something else in the matrix that could interfere."""
    identity: str
    formula: str
    concentration_mm: float
    charge: float = 0.0
    notes: str = ""


@dataclass
class Matrix:
    """The full physical/chemical environment."""
    description: str = ""
    ph: Optional[float] = None
    temperature_c: float = 25.0
    ionic_strength_mm: Optional[float] = None
    redox_potential_mv: Optional[float] = None
    competing_species: list[CompetingSpecies] = field(default_factory=list)
    flow_rate_l_min: Optional[float] = None
    pressure_atm: float = 1.0
    notes: str = ""


@dataclass
class Outcome:
    """What energy landscape trajectory the user wants. Open-ended, not an enum."""
    description: str
    reversible: Optional[bool] = None
    trigger: Optional[str] = None
    product: Optional[str] = None
    destination: Optional[str] = None
    notes: str = ""


@dataclass
class Exclusion:
    """Something that must NOT happen. Repulsion is half the design."""
    description: str
    reason: str = ""


@dataclass
class Constraints:
    """Real-world limits. Encodes values: cost, accessibility, waste, reusability."""
    max_cost_per_unit: Optional[str] = None
    required_reusability_cycles: Optional[int] = None
    no_environmental_release: bool = True
    available_equipment: list[str] = field(default_factory=list)
    scale: str = "lab"
    exclusions: list[Exclusion] = field(default_factory=list)
    notes: str = ""


@dataclass
class Problem:
    """A molecular design problem expressed in physics terms. MABE's core unit."""
    target: TargetSpecies
    matrix: Matrix
    desired_outcome: Outcome
    constraints: Constraints = field(default_factory=Constraints)
    original_query: str = ""
    assumptions_made: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Target: {self.target.summary()}",
            f"Matrix: {self.matrix.description or 'unspecified'}",
            f"Desired outcome: {self.desired_outcome.description}",
        ]
        if self.constraints.exclusions:
            excl = ", ".join(e.description for e in self.constraints.exclusions)
            lines.append(f"Must NOT: {excl}")
        if self.assumptions_made:
            lines.append(f"Assumptions: {'; '.join(self.assumptions_made)}")
        return "\\n".join(lines)
''')

# ═══════════════════════════════════════════════════════════════════════════
# core/candidate.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/candidate.py", '''"""
core/candidate.py - What MABE returns: candidate solutions with honest uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerformancePrediction:
    """Probabilistic assessment. Never a single number."""
    probability_of_success: float
    confidence: str
    confidence_reasoning: str
    sensitive_to: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    what_improves_odds: list[str] = field(default_factory=list)
    selectivity_threats: list[str] = field(default_factory=list)


@dataclass
class EvidenceProfile:
    """What is known vs predicted. MABE never hides this distinction."""
    source_type: str
    literature_references: list[str] = field(default_factory=list)
    computational_method: str = ""
    what_would_validate: str = ""
    notes: str = ""


@dataclass
class AccessibilityProfile:
    """Values-aligned assessment. Public good first."""
    estimated_cost: str
    equipment_required: list[str] = field(default_factory=list)
    community_lab_feasible: bool = False
    reusability_cycles: Optional[int] = None
    waste_generated: str = ""
    end_of_life: str = ""


@dataclass
class ImmobilizationOption:
    """How this candidate attaches to a physical substrate. Nothing released to environment."""
    substrate: str
    attachment_chemistry: str
    click_handle: str
    effect_on_binding: str
    notes: str = ""


@dataclass
class ApplicationConnection:
    """A cross-domain application this design could address."""
    domain: str
    description: str
    what_would_change: str
    confidence: str


@dataclass
class CandidateResult:
    """One possible solution to the user's problem."""
    rank: int
    name: str
    description: str
    modality: str
    source_tool: str
    structure_description: str
    performance: PerformancePrediction
    evidence: EvidenceProfile
    accessibility: AccessibilityProfile
    immobilization_options: list[ImmobilizationOption] = field(default_factory=list)
    other_applications: list[ApplicationConnection] = field(default_factory=list)

    def short_summary(self) -> str:
        conf = self.performance.confidence
        prob = f"{self.performance.probability_of_success:.0%}"
        cost = self.accessibility.estimated_cost
        return (
            f"#{self.rank}: {self.name} ({self.modality})\\n"
            f"   Success probability: {prob} ({conf}) | Cost: {cost} | "
            f"Evidence: {self.evidence.source_type}"
        )

    def full_report(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  #{self.rank}: {self.name}",
            f"{'=' * 60}",
            f"",
            f"  {self.description}",
            f"",
            f"  Modality: {self.modality}",
            f"  Structure: {self.structure_description}",
            f"  Generated by: {self.source_tool}",
            f"",
            f"-- Performance --",
            f"  Probability of success: {self.performance.probability_of_success:.0%}",
            f"  Confidence: {self.performance.confidence}",
            f"  Reasoning: {self.performance.confidence_reasoning}",
        ]

        if self.performance.failure_modes:
            lines.append(f"")
            lines.append(f"  What could go wrong:")
            for fm in self.performance.failure_modes:
                lines.append(f"    - {fm}")

        if self.performance.what_improves_odds:
            lines.append(f"")
            lines.append(f"  What improves your odds:")
            for imp in self.performance.what_improves_odds:
                lines.append(f"    - {imp}")

        if self.performance.selectivity_threats:
            lines.append(f"")
            lines.append(f"  Selectivity threats:")
            for st in self.performance.selectivity_threats:
                lines.append(f"    - {st}")

        lines.extend([
            f"",
            f"-- Evidence --",
            f"  Type: {self.evidence.source_type}",
        ])
        if self.evidence.literature_references:
            lines.append(f"  Literature:")
            for ref in self.evidence.literature_references:
                lines.append(f"    - {ref}")
        if self.evidence.computational_method:
            lines.append(f"  Method: {self.evidence.computational_method}")
        if self.evidence.what_would_validate:
            lines.append(f"  To validate: {self.evidence.what_would_validate}")

        lines.extend([
            f"",
            f"-- Accessibility --",
            f"  Cost: {self.accessibility.estimated_cost}",
            f"  Equipment: {', '.join(self.accessibility.equipment_required) or 'basic lab'}",
            f"  Community lab feasible: {'Yes' if self.accessibility.community_lab_feasible else 'No'}",
        ])
        if self.accessibility.reusability_cycles:
            lines.append(f"  Reusability: ~{self.accessibility.reusability_cycles} cycles")
        if self.accessibility.waste_generated:
            lines.append(f"  Waste: {self.accessibility.waste_generated}")
        if self.accessibility.end_of_life:
            lines.append(f"  End of life: {self.accessibility.end_of_life}")

        if self.immobilization_options:
            lines.append(f"")
            lines.append(f"-- Immobilization Options --")
            for opt in self.immobilization_options:
                lines.append(f"  - {opt.substrate} via {opt.attachment_chemistry}")
                lines.append(f"    Handle: {opt.click_handle}")
                lines.append(f"    Effect on binding: {opt.effect_on_binding}")

        if self.other_applications:
            lines.append(f"")
            lines.append(f"-- What Else Could This Do --")
            for app in self.other_applications:
                lines.append(f"  - [{app.confidence}] {app.domain}: {app.description}")
                lines.append(f"    Would need: {app.what_would_change}")

        return "\\n".join(lines)
''')

# ═══════════════════════════════════════════════════════════════════════════
# adapters/base.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("adapters/base.py", '''"""
adapters/base.py - Universal tool adapter interface and registry.
Any molecular design tool connects to MABE by implementing ToolAdapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from core.problem import Problem
from core.candidate import CandidateResult


@dataclass
class Capability:
    """What a tool can do, described in physics terms."""
    description: str
    target_types: list[str] = field(default_factory=list)
    interaction_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)


@dataclass
class ContributionAssessment:
    """A tool's honest self-assessment of whether it can help."""
    can_contribute: bool
    relevance: float
    what_it_would_do: str
    what_part_of_problem: str
    estimated_compute_time: str = "unknown"
    limitations: list[str] = field(default_factory=list)


class ToolAdapter(ABC):
    """Universal interface for connecting any tool to MABE."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> list[Capability]: ...

    @abstractmethod
    def assess_contribution(self, problem: Problem) -> ContributionAssessment: ...

    @abstractmethod
    def generate_candidates(self, problem: Problem) -> list[CandidateResult]: ...

    def is_available(self) -> bool:
        return True

    def __repr__(self):
        return f"<{self.name} v{self.version}>"


class ToolRegistry:
    """The web. All connected tools register here."""

    def __init__(self):
        self._adapters: dict[str, ToolAdapter] = {}

    def register(self, adapter: ToolAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def unregister(self, name: str) -> None:
        self._adapters.pop(name, None)

    def get(self, name: str) -> Optional[ToolAdapter]:
        return self._adapters.get(name)

    def all_adapters(self) -> list[ToolAdapter]:
        return list(self._adapters.values())

    def available_adapters(self) -> list[ToolAdapter]:
        return [a for a in self._adapters.values() if a.is_available()]

    def find_contributors(self, problem: Problem) -> list[tuple[ToolAdapter, ContributionAssessment]]:
        contributors = []
        for adapter in self.available_adapters():
            assessment = adapter.assess_contribution(problem)
            if assessment.can_contribute:
                contributors.append((adapter, assessment))
        contributors.sort(key=lambda x: x[1].relevance, reverse=True)
        return contributors

    def summary(self) -> str:
        if not self._adapters:
            return "No tools connected."
        lines = ["Connected tools:"]
        for adapter in self._adapters.values():
            status = "+" if adapter.is_available() else "x"
            caps = len(adapter.capabilities)
            lines.append(f"  {status} {adapter.name} v{adapter.version} ({caps} capabilities)")
        return "\\n".join(lines)


registry = ToolRegistry()
''')

# ═══════════════════════════════════════════════════════════════════════════
# adapters/dummy_adapter.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("adapters/dummy_adapter.py", '''"""
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
''')

# ═══════════════════════════════════════════════════════════════════════════
# core/orchestrator.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/orchestrator.py", '''"""
core/orchestrator.py - The brain of MABE.
Decomposes, routes, assembles, ranks with values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.problem import Problem
from core.candidate import CandidateResult
from adapters.base import ToolRegistry


@dataclass
class OrchestrationResult:
    problem_summary: str
    candidates: list[CandidateResult]
    tools_consulted: list[str]
    tools_declined: list[tuple[str, str]]
    assumptions: list[str]
    notes: list[str] = field(default_factory=list)


class Orchestrator:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def solve(self, problem: Problem) -> OrchestrationResult:
        contributors = self.registry.find_contributors(problem)
        tools_consulted = []
        tools_declined = []

        for adapter in self.registry.available_adapters():
            assessment = adapter.assess_contribution(problem)
            if not assessment.can_contribute:
                tools_declined.append((adapter.name, "Not relevant"))

        all_candidates: list[CandidateResult] = []
        for adapter, assessment in contributors:
            tools_consulted.append(f"{adapter.name}: {assessment.what_it_would_do}")
            candidates = adapter.generate_candidates(problem)
            all_candidates.extend(candidates)

        all_candidates = self._rank_candidates(all_candidates)
        for i, candidate in enumerate(all_candidates):
            candidate.rank = i + 1

        notes = []
        if not contributors:
            notes.append("No connected tools could contribute to this problem.")
        if problem.assumptions_made:
            notes.append("MABE made assumptions - review these, changing one may change results.")

        return OrchestrationResult(
            problem_summary=problem.summary(),
            candidates=all_candidates,
            tools_consulted=tools_consulted,
            tools_declined=tools_declined,
            assumptions=problem.assumptions_made,
            notes=notes,
        )

    def _rank_candidates(self, candidates: list[CandidateResult]) -> list[CandidateResult]:
        """Rank by values: 40% performance, 25% accessibility, 20% reusability, 15% evidence."""
        def score(c: CandidateResult) -> float:
            perf = c.performance.probability_of_success
            access = 0.8 if c.accessibility.community_lab_feasible else 0.3
            cycles = c.accessibility.reusability_cycles or 0
            reuse = min(cycles / 100, 1.0)
            evidence_map = {"literature_validated": 1.0, "hybrid": 0.7, "computational_prediction": 0.4}
            evidence = evidence_map.get(c.evidence.source_type, 0.3)
            return perf * 0.40 + access * 0.25 + reuse * 0.20 + evidence * 0.15

        candidates.sort(key=score, reverse=True)
        return candidates
''')

# ═══════════════════════════════════════════════════════════════════════════
# conversation/decomposer.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("conversation/decomposer.py", '''"""
conversation/decomposer.py - Translates natural language into a Problem.
Sprint 1: Simple keyword matching. The point is proving the pipeline.
"""

from __future__ import annotations

from core.problem import (
    Problem, TargetSpecies, ElectronicDescription, HydrationDescription,
    RedoxState, MagneticDescription, SizeDescription,
    Matrix, CompetingSpecies, Outcome, Constraints, Exclusion,
)

KNOWN_TARGETS = {
    "selenite": TargetSpecies(
        identity="selenite", formula="SeO3(2-)", charge=-2.0,
        geometry="trigonal pyramidal",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=2.55, donor_atoms=["O"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.8, dehydration_energy_kj_mol=1080.0),
        redox_states=[
            RedoxState(6, "SeO4(2-)"), RedoxState(4, "SeO3(2-)"),
            RedoxState(0, "Se(0)", notes="elemental"), RedoxState(-2, "Se(2-)"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=2.39, molecular_weight=126.96),
    ),
    "lead": TargetSpecies(
        identity="lead", formula="Pb(2+)", charge=2.0,
        geometry="variable 4-8 coordinate",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=2.33, donor_atoms=["O","N","S"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.01, dehydration_energy_kj_mol=1481.0, coordination_number_water=9),
        redox_states=[RedoxState(2, "Pb(2+)"), RedoxState(0, "Pb(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=1.19, hydrated_radius_angstrom=4.01, molecular_weight=207.2),
    ),
    "nickel": TargetSpecies(
        identity="nickel", formula="Ni(2+)", charge=2.0,
        geometry="octahedral (preferred), square planar",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91, donor_atoms=["N","O","S"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        redox_states=[RedoxState(2, "Ni(2+)"), RedoxState(0, "Ni(0)")],
        magnetic=MagneticDescription(type="paramagnetic", unpaired_electrons=2),
        size=SizeDescription(ionic_radius_angstrom=0.69, hydrated_radius_angstrom=4.04, molecular_weight=58.69),
    ),
    "gold": TargetSpecies(
        identity="gold", formula="Au(3+)", charge=3.0,
        geometry="square planar",
        electronic=ElectronicDescription(hardness_softness="soft", electronegativity=2.54, donor_atoms=["S","P","C"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.5, dehydration_energy_kj_mol=4690.0),
        redox_states=[RedoxState(3, "Au(3+)"), RedoxState(1, "Au(+)"), RedoxState(0, "Au(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=0.85, molecular_weight=196.97),
    ),
    "copper": TargetSpecies(
        identity="copper", formula="Cu(2+)", charge=2.0,
        geometry="Jahn-Teller distorted octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.90, donor_atoms=["N","O","S"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.19, dehydration_energy_kj_mol=2100.0, coordination_number_water=6),
        redox_states=[RedoxState(2, "Cu(2+)"), RedoxState(1, "Cu(+)"), RedoxState(0, "Cu(0)")],
        magnetic=MagneticDescription(type="paramagnetic", unpaired_electrons=1),
        size=SizeDescription(ionic_radius_angstrom=0.73, hydrated_radius_angstrom=4.19, molecular_weight=63.55),
    ),
}

KNOWN_MATRICES = {
    "mine": Matrix(
        description="Acid mine drainage - typical BC/Canadian mine site",
        ph=3.5, temperature_c=12.0, ionic_strength_mm=50.0, redox_potential_mv=400.0,
        competing_species=[
            CompetingSpecies("sulfate", "SO4(2-)", 500.0, -2.0),
            CompetingSpecies("calcium", "Ca(2+)", 200.0, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 100.0, 2.0),
            CompetingSpecies("iron", "Fe(3+)", 50.0, 3.0),
        ],
    ),
    "river": Matrix(
        description="Freshwater river", ph=7.2, temperature_c=15.0, ionic_strength_mm=5.0,
        competing_species=[
            CompetingSpecies("calcium", "Ca(2+)", 40.0, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 15.0, 2.0),
        ],
    ),
    "ocean": Matrix(
        description="Seawater", ph=8.1, temperature_c=18.0, ionic_strength_mm=700.0,
        competing_species=[
            CompetingSpecies("sodium", "Na(+)", 468000.0, 1.0),
            CompetingSpecies("chloride", "Cl(-)", 546000.0, -1.0),
            CompetingSpecies("magnesium", "Mg(2+)", 52800.0, 2.0),
        ],
    ),
}


def decompose(user_input: str) -> Problem:
    text = user_input.lower().strip()
    assumptions = []

    target = None
    for name, species in KNOWN_TARGETS.items():
        if name in text:
            target = species
            break

    if target is None:
        target = TargetSpecies(identity="unknown target", formula="?", charge=0.0, geometry="unknown")
        assumptions.append(f"Could not identify a specific target in: '{user_input}'. Using generic target.")

    matrix = Matrix()
    for keyword, known_matrix in KNOWN_MATRICES.items():
        if keyword in text:
            matrix = known_matrix
            break

    if not matrix.description:
        matrix.description = "unspecified matrix"
        assumptions.append("No matrix specified - using default conditions (pH 7, 25C, freshwater)")
        matrix.ph = 7.0
        matrix.temperature_c = 25.0
        matrix.ionic_strength_mm = 10.0

    outcome_desc = "capture target"
    constraints = Constraints()

    if "release" in text or "feedstock" in text or "recover" in text:
        outcome_desc = "capture and release as clean feedstock"
        constraints.required_reusability_cycles = 20
    elif "transform" in text or "convert" in text:
        outcome_desc = "capture and transform to useful product"
    elif "place" in text or "deposit" in text or "chip" in text:
        outcome_desc = "selective extraction and precision placement"
    elif "remove" in text or "clean" in text:
        outcome_desc = "remove from environment"
        constraints.required_reusability_cycles = 50

    outcome = Outcome(description=outcome_desc)
    constraints.no_environmental_release = True
    constraints.exclusions.append(
        Exclusion("No nanomaterial release to environment", "MABE default - public good")
    )

    return Problem(
        target=target, matrix=matrix, desired_outcome=outcome,
        constraints=constraints, original_query=user_input, assumptions_made=assumptions,
    )
''')

# ═══════════════════════════════════════════════════════════════════════════
# conversation/interface.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("conversation/interface.py", '''"""
conversation/interface.py - MABE conversational CLI.
"""

from __future__ import annotations

from core.orchestrator import Orchestrator, OrchestrationResult
from core.candidate import CandidateResult
from conversation.decomposer import decompose
from adapters.base import ToolRegistry


def print_banner():
    print()
    print("  +----------------------------------------------------------+")
    print("  |                                                          |")
    print("  |    MABE - Modality-Agnostic Binder Engine                |")
    print("  |    Universal Molecular Interaction Design Platform        |")
    print("  |                                                          |")
    print("  |    Public good first. Honest uncertainty. No waste.       |")
    print("  |                                                          |")
    print("  +----------------------------------------------------------+")
    print()


def print_result(result: OrchestrationResult):
    print()
    print("=" * 60)
    print("  MABE ANALYSIS")
    print("=" * 60)
    print()

    print("  Your problem (as MABE understands it):")
    for line in result.problem_summary.split("\\n"):
        print(f"    {line}")
    print()

    if result.assumptions:
        print("  ! Assumptions MABE made (review these):")
        for a in result.assumptions:
            print(f"    - {a}")
        print()

    for note in result.notes:
        print(f"  i {note}")
        print()

    if result.tools_consulted:
        print(f"  Tools consulted: {len(result.tools_consulted)}")
        for tc in result.tools_consulted:
            print(f"    - {tc}")
        print()

    if not result.candidates:
        print("  No candidates found.")
        return

    print(f"  Top {len(result.candidates)} candidates (ranked by public-good-weighted score):")
    print()
    for c in result.candidates:
        print(f"  {c.short_summary()}")
        print()
    print("=" * 60)
    print()


def explore_candidate(candidate: CandidateResult):
    print()
    print(candidate.full_report())
    print()


def run_interactive(registry: ToolRegistry):
    print_banner()
    orchestrator = Orchestrator(registry)
    print(f"  {registry.summary()}")
    print()
    print("  Describe your problem in plain language.")
    print("  Type 'quit' to exit, or a number to explore a candidate.")
    print()

    last_result = None

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\n\\n  Goodbye.\\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\\n  Goodbye.\\n")
            break

        if user_input.isdigit() and last_result and last_result.candidates:
            idx = int(user_input) - 1
            if 0 <= idx < len(last_result.candidates):
                explore_candidate(last_result.candidates[idx])
                print(f"  Enter a number to explore another, or describe a new problem.")
                print()
                continue
            else:
                print(f"  No candidate #{user_input}. Choose 1-{len(last_result.candidates)}.")
                continue

        problem = decompose(user_input)
        result = orchestrator.solve(problem)
        last_result = result
        print_result(result)
        n = len(result.candidates) if result.candidates else 0
        print(f"  Enter a number (1-{n}) to explore a candidate in detail.")
        print()


def run_single_query(registry: ToolRegistry, query: str):
    orchestrator = Orchestrator(registry)
    problem = decompose(query)
    result = orchestrator.solve(problem)
    print_result(result)
    if result.candidates:
        for c in result.candidates:
            explore_candidate(c)
''')

# ═══════════════════════════════════════════════════════════════════════════
# main.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("main.py", '''"""
MABE - Modality-Agnostic Binder Engine
Run with: python main.py
Or: python main.py "selenite capture from mine water"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.base import ToolRegistry
from adapters.dummy_adapter import DummyAdapter
from conversation.interface import run_interactive, run_single_query


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(DummyAdapter())
    # Future:
    # registry.register(RDKitAdapter())
    # registry.register(RFD3Adapter())
    # registry.register(ATHENAAdapter())
    return registry


def main():
    registry = build_registry()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(registry, query)
    else:
        run_interactive(registry)


if __name__ == "__main__":
    main()
''')

# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint1.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint1.py", '''"""
tests/test_sprint1.py - End-to-end test for Sprint 1 skeleton.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.problem import Problem, TargetSpecies, Matrix, Outcome, Constraints
from core.candidate import CandidateResult
from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose, KNOWN_TARGETS


def test_target_species():
    se = KNOWN_TARGETS["selenite"]
    assert se.charge == -2.0
    assert se.geometry == "trigonal pyramidal"
    assert se.electronic.hardness_softness == "borderline"
    assert len(se.redox_states) == 4

    ni = KNOWN_TARGETS["nickel"]
    assert ni.magnetic.type == "paramagnetic"
    assert ni.magnetic.unpaired_electrons == 2
    print("  + TargetSpecies - physics descriptions correct")


def test_decompose_selenite_mine():
    problem = decompose("I need selenite capture from a mine in BC")
    assert problem.target.identity == "selenite"
    assert problem.target.charge == -2.0
    assert problem.matrix.ph == 3.5
    assert problem.constraints.no_environmental_release is True
    print(f"  + Decompose - selenite from mine detected")


def test_decompose_lead_release():
    problem = decompose("capture lead from mine water and release as feedstock")
    assert problem.target.identity == "lead"
    assert "release" in problem.desired_outcome.description.lower()
    assert problem.constraints.required_reusability_cycles >= 20
    print(f"  + Decompose - capture/release outcome detected")


def test_decompose_unknown():
    problem = decompose("I need to capture unobtanium from the atmosphere")
    assert problem.target.identity == "unknown target"
    assert len(problem.assumptions_made) > 0
    print(f"  + Decompose - unknown target handled with assumptions")


def test_registry():
    registry = ToolRegistry()
    registry.register(DummyAdapter())
    assert len(registry.available_adapters()) == 1
    problem = decompose("lead capture from mine")
    contributors = registry.find_contributors(problem)
    assert len(contributors) > 0
    print(f"  + Registry - {len(contributors)} tool(s) contribute")


def test_orchestrator():
    registry = ToolRegistry()
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)
    problem = decompose("selenite capture and release from mine water in BC")
    result = orchestrator.solve(problem)

    assert len(result.candidates) > 0
    for i, c in enumerate(result.candidates):
        assert c.rank == i + 1
    for c in result.candidates:
        assert c.performance.confidence in ("high", "moderate", "low", "speculative")
        assert len(c.performance.failure_modes) > 0
        assert c.evidence.source_type
        assert c.accessibility.estimated_cost
    has_connections = any(len(c.other_applications) > 0 for c in result.candidates)
    assert has_connections
    has_immob = any(len(c.immobilization_options) > 0 for c in result.candidates)
    assert has_immob
    print(f"  + Orchestrator - {len(result.candidates)} candidates, ranked, with uncertainty")


def test_values_ranking():
    registry = ToolRegistry()
    registry.register(DummyAdapter())
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    chelator = None
    nanocage = None
    for c in result.candidates:
        if "chelator" in c.modality.lower():
            chelator = c
        if "nanocage" in c.modality.lower():
            nanocage = c

    if chelator and nanocage:
        assert chelator.rank < nanocage.rank
        print(f"  + Values - accessible chelator ranks above expensive nanocage")
    else:
        print(f"  + Values - ranking test skipped")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 1 - Skeleton Tests")
    print("  " + "=" * 40)
    print()

    test_target_species()
    test_decompose_selenite_mine()
    test_decompose_lead_release()
    test_decompose_unknown()
    test_registry()
    test_orchestrator()
    test_values_ranking()

    print()
    print("  All tests passed.")
    print()
''')

# ═══════════════════════════════════════════════════════════════════════════

print()
print("  Done! Files created:")
print()
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f != "bootstrap_sprint1.py" and not f.endswith(".pyc"):
            print(f"    {os.path.join(root, f)}")

print()
print("  Next steps:")
print("    python tests/test_sprint1.py    (run tests)")
print("    python main.py                  (interactive mode)")
print('    python main.py "selenite capture from mine water in BC"')
print()