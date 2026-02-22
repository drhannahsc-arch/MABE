"""
conversation/interface.py - MABE conversational CLI.
Now displays both flat candidates and composite assemblies.
"""

from __future__ import annotations

from core.orchestrator import Orchestrator, OrchestrationResult
from core.candidate import CandidateResult
from core.assembly import BinderAssembly
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
    for line in result.problem_summary.split("\n"):
        print(f"    {line}")
    print()

    if result.assumptions:
        print("  ! Assumptions MABE made:")
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

    # ── COMPOSITE ASSEMBLIES ─────────────────────────────────────
    if result.assemblies:
        print("=" * 60)
        print("  COMPOSITE BINDER ASSEMBLIES")
        print("  (recognition + structure + selectivity + release)")
        print("=" * 60)
        print()
        for i, a in enumerate(result.assemblies):
            print(f"  A{i+1}: {a.name}")
            print(f"      Score: {a.composite_score:.0%} ({a.confidence}) | Cost: {a.estimated_cost}")
            print(f"      Structure: {a.structure.type} | Release: {a.release.trigger}")
            print()

    # ── FLAT CANDIDATES ──────────────────────────────────────────
    if result.candidates:
        n = len(result.candidates)
        print(f"  Top {n} recognition chemistry candidates:")
        print()
        for c in result.candidates[:10]:  # Show top 10
            print(f"  {c.short_summary()}")
            print()

    print("=" * 60)
    print()


def explore_assembly(assembly: BinderAssembly):
    print()
    print(assembly.full_report())
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
    print("  Type A1-A9 to explore an assembly, 1-N for a candidate, 'quit' to exit.")
    print()

    last_result = None

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye.\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye.\n")
            break

        # Explore assembly
        if user_input.upper().startswith("A") and last_result and last_result.assemblies:
            try:
                idx = int(user_input[1:]) - 1
                if 0 <= idx < len(last_result.assemblies):
                    explore_assembly(last_result.assemblies[idx])
                    continue
            except ValueError:
                pass

        # Explore candidate
        if user_input.isdigit() and last_result and last_result.candidates:
            idx = int(user_input) - 1
            if 0 <= idx < len(last_result.candidates):
                explore_candidate(last_result.candidates[idx])
                continue

        problem = decompose(user_input)
        result = orchestrator.solve(problem)
        last_result = result
        print_result(result)


def run_single_query(registry: ToolRegistry, query: str):
    orchestrator = Orchestrator(registry)
    problem = decompose(query)
    result = orchestrator.solve(problem)
    print_result(result)
    if result.assemblies:
        print()
        print("  ASSEMBLY DETAILS:")
        for a in result.assemblies:
            explore_assembly(a)
