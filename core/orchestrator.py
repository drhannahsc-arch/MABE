"""
core/orchestrator.py - The brain of MABE.
Now produces BOTH flat candidates AND composite assemblies.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from core.problem import Problem
from core.candidate import CandidateResult
from core.assembly import BinderAssembly
from core.connections import discover_connections
import core.assembly_composer as _assembly_composer_module
from adapters.base import ToolRegistry


@dataclass
class OrchestrationResult:
    problem_summary: str
    candidates: list[CandidateResult]
    assemblies: list[BinderAssembly]
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

        # Discover cross-domain connections
        target_name = problem.target.identity
        for candidate in all_candidates:
            new_connections = discover_connections(candidate, target_name)
            candidate.other_applications.extend(new_connections)

        # Rank flat candidates
        all_candidates = self._rank_candidates(all_candidates)
        for i, candidate in enumerate(all_candidates):
            candidate.rank = i + 1

        # COMPOSE ASSEMBLIES from top recognition candidates
        assemblies = _assembly_composer_module.compose_assemblies(all_candidates, problem, max_assemblies=6)

        notes = []
        if not contributors:
            notes.append("No connected tools could contribute to this problem.")
        if problem.assumptions_made:
            notes.append("MABE made assumptions - review these.")

        total_connections = sum(len(c.other_applications) for c in all_candidates)
        if total_connections > 0:
            notes.append(f"{total_connections} cross-domain applications discovered. Explore candidates for details.")

        if assemblies:
            notes.append(
                f"{len(assemblies)} composite binder assemblies designed: "
                f"recognition chemistry + structural constraint + selectivity filter + release mechanism."
            )

        return OrchestrationResult(
            problem_summary=problem.summary(),
            candidates=all_candidates,
            assemblies=assemblies,
            tools_consulted=tools_consulted,
            tools_declined=tools_declined,
            assumptions=problem.assumptions_made,
            notes=notes,
        )

    def _rank_candidates(self, candidates: list[CandidateResult]) -> list[CandidateResult]:
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
