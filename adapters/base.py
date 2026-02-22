"""
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
        return "\n".join(lines)


registry = ToolRegistry()
