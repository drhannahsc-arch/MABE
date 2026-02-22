"""
MABE Sprint 5 Bootstrap - Cross-Domain Connection Engine
=========================================================
Teaches the orchestrator to see beyond the user's question.

Every binder that captures a metal IS ALSO a recognition element.
Pair it with a readout and you replace mass spec.

    cd Documents\\mabe
    python bootstrap_sprint5.py
    python tests\\test_sprint5.py
    python main.py "lead capture from mine water"
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
print("  MABE Sprint 5 - Cross-Domain Connection Engine")
print("  " + "=" * 40)
print()

# ═══════════════════════════════════════════════════════════════════════════
# core/connections.py — The brain that sees what the user didn't ask
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/connections.py", '''"""
core/connections.py - Cross-domain connection engine.

Given a candidate binder, this module discovers applications the user
did not ask about. The key insight:

    Any molecule that selectively CAPTURES a target
    can also selectively DETECT that target.

    Capture + readout = diagnostic.
    Diagnostic at $5 instead of mass spec at $500.

This is MABE's value multiplication layer. A user asks for mine water
remediation and gets back candidates that ALSO enable:
- Field diagnostics (lateral flow, electrochemical)
- Research tools (cheap alternative to ICP-MS)
- Therapeutic chelation
- Environmental monitoring networks
- Quality control in manufacturing

Connection rules are grounded in physics, not speculation.
Each connection explains WHAT would change and HOW HARD it is.
"""

from __future__ import annotations
from core.candidate import CandidateResult, ApplicationConnection


# ═══════════════════════════════════════════════════════════════════════════
# Connection rules — each rule examines a candidate and may add connections
# ═══════════════════════════════════════════════════════════════════════════

def _diagnostic_from_capture(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    CORE INSIGHT: Any selective binder is a recognition element for a sensor.

    Capture binder + signal readout = diagnostic that replaces mass spec.

    Readout options depend on modality:
    - DNA-based (dnazyme, aptamer, dna_motif): fluorophore/quencher, lateral flow, electrochemical
    - Peptide: conjugate to reporter enzyme or nanoparticle, electrochemical
    - Chelator: colorimetric (metal-indicator displacement), electrochemical
    - Protein: ELISA-like sandwich, SPR, electrochemical
    """
    connections = []
    modality = candidate.modality.lower()
    prob = candidate.performance.probability_of_success
    cost = candidate.accessibility.estimated_cost

    # Only generate diagnostic connections for candidates with reasonable binding
    if prob < 0.2:
        return connections

    # ── DNA-based readouts ────────────────────────────────────────
    if modality in ("dnazyme", "dna_aptamer", "dna_motif"):
        connections.append(ApplicationConnection(
            domain="field_diagnostic",
            description=(
                f"Lateral flow strip for {target_name} detection. "
                f"Conjugate binder to gold nanoparticle reporter + capture line. "
                f"Visual yes/no readout in 15 minutes, no equipment needed. "
                f"Estimated cost: $2-5 per test vs $50-500 for ICP-MS."
            ),
            what_would_change=(
                "Add biotin to one end, gold-NP-conjugated complementary strand to other. "
                "Print capture/test lines on nitrocellulose. Standard lateral flow manufacturing."
            ),
            confidence="strong" if prob > 0.4 else "plausible",
        ))
        connections.append(ApplicationConnection(
            domain="electrochemical_sensor",
            description=(
                f"Electrochemical sensor for {target_name}. "
                f"Immobilize binder on screen-printed electrode, measure current change on binding. "
                f"Quantitative, reusable, field-deployable with handheld potentiostat ($200). "
                f"Detection limit typically 10-100x better than visual lateral flow."
            ),
            what_would_change=(
                "Thiolate DNA onto gold electrode surface. Add redox reporter (methylene blue or ferrocene). "
                "Binding event changes electron transfer distance, measurable by square wave voltammetry."
            ),
            confidence="strong" if prob > 0.4 else "plausible",
        ))
        if modality == "dnazyme":
            connections.append(ApplicationConnection(
                domain="fluorescent_sensor",
                description=(
                    f"Fluorescent turn-on sensor for {target_name}. "
                    f"Use native DNAzyme (without capture modification) with fluorophore-quencher pair. "
                    f"Metal binding triggers cleavage, separating F from Q, generating signal. "
                    f"Works in microplate reader or field fluorimeter. Sub-nM detection."
                ),
                what_would_change=(
                    "Use unmodified (native) DNAzyme sequence. Add FAM to substrate 5' end, "
                    "dabcyl quencher to 3' end. No capture modification needed — this IS the native function."
                ),
                confidence="strong",
            ))

    # ── Peptide readouts ──────────────────────────────────────────
    if modality == "peptide_chelator":
        connections.append(ApplicationConnection(
            domain="field_diagnostic",
            description=(
                f"Colorimetric test strip for {target_name}. "
                f"Immobilize peptide on test strip, add indicator dye that is displaced by metal binding. "
                f"Color change visible by eye. Estimated cost: $1-3 per test."
            ),
            what_would_change=(
                "Conjugate peptide to cellulose strip via NHS chemistry. "
                "Pre-load with weak indicator (e.g., PAR or zincon) that changes color when displaced by target."
            ),
            confidence="plausible",
        ))
        connections.append(ApplicationConnection(
            domain="electrochemical_sensor",
            description=(
                f"Peptide-modified electrode for {target_name} quantification. "
                f"Self-assembled monolayer of peptide on gold electrode. "
                f"Metal binding changes impedance, measurable by EIS. Reusable."
            ),
            what_would_change=(
                "Add C-terminal Cys for gold-thiol SAM formation. "
                "Measure electrochemical impedance spectroscopy before/after sample exposure."
            ),
            confidence="plausible",
        ))

    # ── Chelator readouts ─────────────────────────────────────────
    if modality == "chelator":
        connections.append(ApplicationConnection(
            domain="field_diagnostic",
            description=(
                f"Indicator displacement assay for {target_name}. "
                f"Pre-complex chelator with colored indicator (weak binding). "
                f"Target metal displaces indicator, producing visible color change. "
                f"Classic analytical chemistry, works in a test tube. Cost: <$1."
            ),
            what_would_change=(
                "Select indicator with lower binding constant than chelator for target. "
                "Calibrate color change vs concentration. Can be read by phone camera for quantification."
            ),
            confidence="strong",
        ))

    # ── Protein readouts ──────────────────────────────────────────
    if modality in ("designed_protein", "nanobody"):
        connections.append(ApplicationConnection(
            domain="field_diagnostic",
            description=(
                f"Protein-based sensor for {target_name}. "
                f"Sandwich format: capture protein on surface, detection protein with reporter. "
                f"ELISA-like but for metals. Quantitative."
            ),
            what_would_change=(
                "Engineer two variants: one for surface capture, one for detection with HRP label. "
                "Requires protein engineering but leverages existing ELISA infrastructure."
            ),
            confidence="plausible",
        ))

    return connections


def _research_tool_from_capture(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    Research tool connections: cheap alternatives to expensive analytical methods.
    """
    connections = []
    prob = candidate.performance.probability_of_success
    if prob < 0.3:
        return connections

    connections.append(ApplicationConnection(
        domain="research_tool",
        description=(
            f"Low-cost {target_name} quantification for research labs. "
            f"Replace ICP-MS ($50-500/sample) with binder-based assay ($1-10/sample). "
            f"Enables high-throughput screening, field studies, and resource-limited settings. "
            f"Particularly valuable for: environmental surveys, dose-response studies, "
            f"process optimization, and continuous monitoring."
        ),
        what_would_change=(
            "Validate binder response curve against ICP-MS standards. "
            "Package as kit: binder + readout reagents + calibration standards. "
            "Cross-validate with 20+ real samples to establish correlation."
        ),
        confidence="plausible" if prob > 0.5 else "speculative",
    ))

    return connections


def _dna_tag_encoding(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    DNA tag attachment: encode binding events as DNA sequences for multiplexed readout.
    """
    connections = []
    modality = candidate.modality.lower()

    if modality not in ("dnazyme", "dna_aptamer", "dna_motif", "peptide_chelator"):
        return connections

    if candidate.performance.probability_of_success < 0.3:
        return connections

    if modality in ("dnazyme", "dna_aptamer", "dna_motif"):
        connections.append(ApplicationConnection(
            domain="multiplexed_diagnostics",
            description=(
                f"DNA-barcoded multiplexed detection. "
                f"Each binder carries a unique DNA barcode tag. "
                f"Pool multiple binders for different metals in one sample. "
                f"After binding, read all barcodes by sequencing or hybridization array. "
                f"One test, many metals. Replaces multi-element ICP-MS panel."
            ),
            what_would_change=(
                "Extend binder with unique 20-nt barcode + universal primer site. "
                "After capture, wash and elute. PCR amplify barcodes. "
                "Read by Nanopore sequencing ($2/sample) or lateral flow array."
            ),
            confidence="plausible",
        ))
    else:
        # Peptide with click-attached DNA tag
        connections.append(ApplicationConnection(
            domain="multiplexed_diagnostics",
            description=(
                f"Click-chemistry DNA tag for multiplexed {target_name} detection. "
                f"Attach unique DNA barcode to peptide binder via click chemistry (azide-DBCO). "
                f"After capture, read barcode. Enables multi-analyte panels from single sample."
            ),
            what_would_change=(
                "Add azide-modified lysine to peptide. Click to DBCO-DNA barcode. "
                "Standard bioconjugation, well-established protocols."
            ),
            confidence="plausible",
        ))

    return connections


def _monitoring_network(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    Environmental monitoring network from field-deployable sensors.
    """
    connections = []
    prob = candidate.performance.probability_of_success
    if prob < 0.4:
        return connections

    modality = candidate.modality.lower()
    if modality in ("dnazyme", "dna_aptamer", "dna_motif", "chelator"):
        connections.append(ApplicationConnection(
            domain="monitoring_network",
            description=(
                f"Distributed {target_name} monitoring network. "
                f"Deploy binder-based sensors at multiple points along waterway. "
                f"Continuous or periodic sampling with phone-camera readout. "
                f"Community-operated water quality monitoring at $5-20 per test point. "
                f"Replaces periodic lab sampling ($200-1000/visit)."
            ),
            what_would_change=(
                "Package sensor in waterproof housing with sample inlet. "
                "Develop phone app for color/fluorescence reading and GPS logging. "
                "Train community members for sample collection (15 min protocol)."
            ),
            confidence="plausible",
        ))

    return connections


def _therapeutic_from_capture(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    Therapeutic chelation from environmental capture binders.
    """
    connections = []
    prob = candidate.performance.probability_of_success
    modality = candidate.modality.lower()

    if prob < 0.4:
        return connections

    # Only some modalities have therapeutic potential
    if modality in ("chelator", "peptide_chelator"):
        notes = candidate.description.lower()
        if "fda" in notes:
            connections.append(ApplicationConnection(
                domain="therapeutic",
                description=(
                    f"This compound is FDA-approved for clinical use. "
                    f"Direct path to therapeutic {target_name} chelation in poisoning cases."
                ),
                what_would_change="Existing clinical protocols apply. Dosing optimization for specific indication.",
                confidence="strong",
            ))
        elif modality == "peptide_chelator":
            connections.append(ApplicationConnection(
                domain="therapeutic",
                description=(
                    f"Peptide chelator for therapeutic {target_name} removal. "
                    f"Peptides are biodegradable with predictable pharmacokinetics. "
                    f"Phytochelatins are natural plant defense — low toxicity expected."
                ),
                what_would_change=(
                    "Add PEG for renal clearance. Toxicity studies. "
                    "Formulation for oral or IV delivery. Long regulatory path."
                ),
                confidence="speculative",
            ))

    return connections


# ═══════════════════════════════════════════════════════════════════════════
# Main engine: run all connection rules on a candidate
# ═══════════════════════════════════════════════════════════════════════════

ALL_CONNECTION_RULES = [
    _diagnostic_from_capture,
    _research_tool_from_capture,
    _dna_tag_encoding,
    _monitoring_network,
    _therapeutic_from_capture,
]


def discover_connections(candidate: CandidateResult, target_name: str) -> list[ApplicationConnection]:
    """
    Run all connection rules on a candidate.
    Returns new connections that weren't already present.
    """
    existing_domains = {(a.domain, a.description[:50]) for a in candidate.other_applications}
    new_connections = []

    for rule in ALL_CONNECTION_RULES:
        connections = rule(candidate, target_name)
        for conn in connections:
            key = (conn.domain, conn.description[:50])
            if key not in existing_domains:
                new_connections.append(conn)
                existing_domains.add(key)

    return new_connections
''')

# ═══════════════════════════════════════════════════════════════════════════
# Update orchestrator to use connection engine
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/orchestrator.py", '''"""
core/orchestrator.py - The brain of MABE.
Decomposes, routes, assembles, ranks with values, discovers connections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.problem import Problem
from core.candidate import CandidateResult
from core.connections import discover_connections
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

        # STEP: Discover cross-domain connections for every candidate
        target_name = problem.target.identity
        for candidate in all_candidates:
            new_connections = discover_connections(candidate, target_name)
            candidate.other_applications.extend(new_connections)

        all_candidates = self._rank_candidates(all_candidates)
        for i, candidate in enumerate(all_candidates):
            candidate.rank = i + 1

        notes = []
        if not contributors:
            notes.append("No connected tools could contribute to this problem.")
        if problem.assumptions_made:
            notes.append("MABE made assumptions - review these, changing one may change results.")

        # Count total connections discovered
        total_connections = sum(len(c.other_applications) for c in all_candidates)
        if total_connections > 0:
            notes.append(
                f"MABE discovered {total_connections} cross-domain applications across all candidates. "
                f"Explore individual candidates to see diagnostic, research tool, and monitoring possibilities."
            )

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
# tests/test_sprint5.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint5.py", '''"""
tests/test_sprint5.py - Cross-domain connection engine tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation.decomposer_patch import patch_targets
patch_targets()

from core.orchestrator import Orchestrator
from core.connections import discover_connections
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from adapters.dummy_adapter import DummyAdapter
from conversation.decomposer import decompose


def _build_full_registry():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available():
        registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    registry.register(DummyAdapter())
    return registry


def test_dnazyme_gets_diagnostic_connections():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    # Find the GR-5 DNAzyme candidate
    gr5 = None
    for c in result.candidates:
        if "GR-5" in c.name:
            gr5 = c
            break

    assert gr5 is not None, "GR-5 should be in results"

    domains = [a.domain for a in gr5.other_applications]
    assert "field_diagnostic" in domains, f"GR-5 should have field diagnostic connection, got: {domains}"
    assert "electrochemical_sensor" in domains, f"GR-5 should have electrochemical connection"
    assert "fluorescent_sensor" in domains, f"GR-5 DNAzyme should have fluorescent sensor (native function)"
    print(f"  + GR-5 DNAzyme: {len(gr5.other_applications)} connections including diagnostics")


def test_chelator_gets_indicator_displacement():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    edta = None
    for c in result.candidates:
        if "EDTA" in c.name:
            edta = c
            break

    assert edta is not None
    domains = [a.domain for a in edta.other_applications]
    assert "field_diagnostic" in domains, f"EDTA should have indicator displacement diagnostic"
    print(f"  + EDTA: {len(edta.other_applications)} connections including indicator displacement")


def test_peptide_gets_diagnostic():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    peptides = [c for c in result.candidates if c.modality == "peptide_chelator"]
    if peptides:
        domains = [a.domain for a in peptides[0].other_applications]
        assert "field_diagnostic" in domains or "electrochemical_sensor" in domains
        print(f"  + Peptide {peptides[0].name}: {len(peptides[0].other_applications)} connections")


def test_research_tool_connection():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_research_tool = False
    for c in result.candidates:
        for a in c.other_applications:
            if a.domain == "research_tool":
                has_research_tool = True
                assert "ICP-MS" in a.description or "mass spec" in a.description.lower()
                break

    assert has_research_tool, "At least one candidate should have research tool connection"
    print(f"  + Research tool connections found (ICP-MS replacement)")


def test_dna_barcode_multiplexing():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_multiplex = False
    for c in result.candidates:
        for a in c.other_applications:
            if a.domain == "multiplexed_diagnostics":
                has_multiplex = True
                assert "barcode" in a.description.lower()
                break

    assert has_multiplex, "DNA-based candidates should have multiplexed diagnostic connection"
    print(f"  + DNA barcode multiplexing connections found")


def test_monitoring_network():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_monitoring = False
    for c in result.candidates:
        for a in c.other_applications:
            if a.domain == "monitoring_network":
                has_monitoring = True
                break

    assert has_monitoring, "Should have monitoring network connections"
    print(f"  + Monitoring network connections found")


def test_total_connections_reported():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    total = sum(len(c.other_applications) for c in result.candidates)
    assert total > 10, f"Expected 10+ total connections, got {total}"

    # The note should mention connections
    has_connection_note = any("cross-domain" in n.lower() for n in result.notes)
    assert has_connection_note, "Orchestrator should report total connections in notes"
    print(f"  + Total cross-domain connections: {total}")


def test_mercury_gets_lateral_flow():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("mercury removal from river water")
    result = orchestrator.solve(problem)

    has_lateral_flow = False
    for c in result.candidates:
        for a in c.other_applications:
            if "lateral flow" in a.description.lower():
                has_lateral_flow = True
                break

    assert has_lateral_flow, "Mercury binders should suggest lateral flow strip diagnostic"
    print(f"  + Mercury: lateral flow diagnostic discovered")


def test_connections_mention_cost_savings():
    registry = _build_full_registry()
    orchestrator = Orchestrator(registry)
    problem = decompose("lead capture from mine water")
    result = orchestrator.solve(problem)

    has_cost_comparison = False
    for c in result.candidates:
        for a in c.other_applications:
            if "ICP-MS" in a.description or "$" in a.description:
                has_cost_comparison = True
                break

    assert has_cost_comparison, "Connections should mention cost savings vs traditional methods"
    print(f"  + Cost comparison to mass spec included in connections")


if __name__ == "__main__":
    print()
    print("  MABE Sprint 5 - Cross-Domain Connection Tests")
    print("  " + "=" * 40)
    print()

    test_dnazyme_gets_diagnostic_connections()
    test_chelator_gets_indicator_displacement()
    test_peptide_gets_diagnostic()
    test_research_tool_connection()
    test_dna_barcode_multiplexing()
    test_monitoring_network()
    test_total_connections_reported()
    test_mercury_gets_lateral_flow()
    test_connections_mention_cost_savings()

    print()
    print("  All Sprint 5 tests passed.")
    print()
''')

print()
print("  Done! New/updated files:")
print("    core/connections.py        (cross-domain connection engine)")
print("    core/orchestrator.py       (updated - runs connection discovery)")
print("    tests/test_sprint5.py      (9 tests)")
print()
print("  Next steps:")
print("    python tests\\test_sprint5.py")
print('    python main.py "lead capture from mine water"')
print("    (then type a number to explore a candidate's connections)")
print()