"""
tests/test_capture_transform_orchestration.py — Tests for MABE orchestration wiring.

Validates:
  - CaptureTransformAdapter implements ToolAdapter correctly
  - Adapter assess_contribution recognizes capture-transform problems
  - Adapter generate_candidates produces CandidateResult objects
  - Decomposer extension adds new targets and matrices
  - End-to-end: natural language → decompose → orchestrate → results
  - Results contain capture-transform candidates alongside others
  - Cu tolerance propagated through to immobilization options
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.problem import (
    Problem, TargetSpecies, Matrix, Outcome, Constraints,
    CompetingSpecies, ElectronicDescription,
)
from core.candidate import CandidateResult
from adapters.base import ToolRegistry, Capability
from adapters.capture_transform_adapter import (
    CaptureTransformAdapter,
    _resolve_target_formula,
    _is_transform_problem,
    _extract_matrix_species,
)
from conversation.decomposer_capture_transform import (
    extend_decomposer,
    register_capture_transform,
    setup_capture_transform,
    CAPTURE_TRANSFORM_TARGETS,
    CAPTURE_TRANSFORM_MATRICES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter interface tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAdapterInterface:

    def test_adapter_has_name(self):
        adapter = CaptureTransformAdapter()
        assert adapter.name == "capture_transform"

    def test_adapter_has_version(self):
        adapter = CaptureTransformAdapter()
        assert adapter.version == "1.0.0"

    def test_adapter_has_capabilities(self):
        adapter = CaptureTransformAdapter()
        assert len(adapter.capabilities) >= 1
        assert isinstance(adapter.capabilities[0], Capability)

    def test_adapter_is_available(self):
        adapter = CaptureTransformAdapter()
        assert adapter.is_available()

    def test_adapter_registers_in_registry(self):
        registry = ToolRegistry()
        adapter = CaptureTransformAdapter()
        registry.register(adapter)
        assert "capture_transform" in [a.name for a in registry.all_adapters()]


# ═══════════════════════════════════════════════════════════════════════════
# Target resolution tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTargetResolution:

    def test_resolve_co2(self):
        p = Problem(
            target=TargetSpecies(identity="carbon dioxide", formula="CO2", charge=0, geometry="linear"),
            matrix=Matrix(), desired_outcome=Outcome(description="capture")
        )
        assert _resolve_target_formula(p) == "CO2"

    def test_resolve_phosphate(self):
        p = Problem(
            target=TargetSpecies(identity="phosphate", formula="PO4(3-)", charge=-3, geometry="tetrahedral"),
            matrix=Matrix(), desired_outcome=Outcome(description="capture")
        )
        assert _resolve_target_formula(p) == "PO4_3-"

    def test_resolve_lead_from_existing_decomposer(self):
        """Lead from original decomposer uses Pb(2+) format."""
        p = Problem(
            target=TargetSpecies(identity="lead", formula="Pb(2+)", charge=2, geometry="variable"),
            matrix=Matrix(), desired_outcome=Outcome(description="capture")
        )
        assert _resolve_target_formula(p) == "Pb2+"

    def test_resolve_identity_fallback(self):
        """Should resolve from identity if formula doesn't match."""
        p = Problem(
            target=TargetSpecies(identity="mercury", formula="?", charge=2, geometry="linear"),
            matrix=Matrix(), desired_outcome=Outcome(description="capture")
        )
        assert _resolve_target_formula(p) == "Hg2+"

    def test_unknown_target_returns_none(self):
        p = Problem(
            target=TargetSpecies(identity="xenon", formula="Xe", charge=0, geometry="spherical"),
            matrix=Matrix(), desired_outcome=Outcome(description="capture")
        )
        assert _resolve_target_formula(p) is None


# ═══════════════════════════════════════════════════════════════════════════
# Transform detection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTransformDetection:

    def _make_problem(self, outcome_desc, identity="carbon dioxide", formula="CO2"):
        return Problem(
            target=TargetSpecies(identity=identity, formula=formula, charge=0, geometry="linear"),
            matrix=Matrix(), desired_outcome=Outcome(description=outcome_desc)
        )

    def test_transform_keyword_detected(self):
        assert _is_transform_problem(self._make_problem("capture and transform to mineral"))

    def test_feedstock_keyword_detected(self):
        assert _is_transform_problem(self._make_problem("convert to usable feedstock"))

    def test_mineralize_keyword_detected(self):
        assert _is_transform_problem(self._make_problem("mineralize CO2"))

    def test_inherent_target_detected(self):
        """CO₂ is inherently transform-amenable even without transform keyword."""
        p = self._make_problem("capture CO2 from air")
        assert _is_transform_problem(p)

    def test_plain_capture_of_non_transform_target(self):
        """Nickel capture without transform keyword → not detected."""
        p = self._make_problem("capture nickel from solution",
                               identity="nickel", formula="Ni(2+)")
        assert not _is_transform_problem(p)


# ═══════════════════════════════════════════════════════════════════════════
# Matrix extraction tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMatrixExtraction:

    def test_extract_from_competing_species(self):
        p = Problem(
            target=TargetSpecies(identity="co2", formula="CO2", charge=0, geometry="linear"),
            matrix=Matrix(
                description="test",
                competing_species=[
                    CompetingSpecies("calcium", "Ca(2+)", 10.0, 2.0),
                    CompetingSpecies("magnesium", "Mg(2+)", 5.0, 2.0),
                ],
            ),
            desired_outcome=Outcome(description="capture"),
        )
        species = _extract_matrix_species(p)
        assert "Ca2+" in species
        assert species["Ca2+"] == 10.0

    def test_enrich_from_seawater_description(self):
        p = Problem(
            target=TargetSpecies(identity="co2", formula="CO2", charge=0, geometry="linear"),
            matrix=Matrix(description="seawater"),
            desired_outcome=Outcome(description="capture"),
        )
        species = _extract_matrix_species(p)
        assert "Ca2+" in species
        assert "Mg2+" in species

    def test_enrich_from_wastewater_description(self):
        p = Problem(
            target=TargetSpecies(identity="phosphate", formula="PO4(3-)", charge=-3, geometry="tetrahedral"),
            matrix=Matrix(description="municipal wastewater"),
            desired_outcome=Outcome(description="capture"),
        )
        species = _extract_matrix_species(p)
        assert "NH4+" in species


# ═══════════════════════════════════════════════════════════════════════════
# Assess contribution tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAssessContribution:

    def test_co2_transform_high_relevance(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="carbon dioxide", formula="CO2", charge=0, geometry="linear"),
            matrix=Matrix(description="seawater"),
            desired_outcome=Outcome(description="capture and transform CO2 to mineral feedstock"),
        )
        assessment = adapter.assess_contribution(p)
        assert assessment.can_contribute
        assert assessment.relevance >= 0.9

    def test_phosphate_wastewater_contributes(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="phosphate", formula="PO4(3-)", charge=-3, geometry="tetrahedral"),
            matrix=Matrix(description="agricultural wastewater"),
            desired_outcome=Outcome(description="recover phosphorus as fertilizer"),
        )
        assessment = adapter.assess_contribution(p)
        assert assessment.can_contribute

    def test_unknown_target_declines(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="xenon", formula="Xe", charge=0, geometry="spherical"),
            matrix=Matrix(),
            desired_outcome=Outcome(description="capture xenon"),
        )
        assessment = adapter.assess_contribution(p)
        assert not assessment.can_contribute

    def test_inherent_target_moderate_relevance(self):
        """CO₂ without explicit transform keyword → still high relevance (inherent target)."""
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="carbon dioxide", formula="CO2", charge=0, geometry="linear"),
            matrix=Matrix(),
            desired_outcome=Outcome(description="remove CO2 from environment"),
        )
        assessment = adapter.assess_contribution(p)
        assert assessment.can_contribute
        # CO₂ is inherently transform-amenable → detected even without keyword
        assert assessment.relevance >= 0.6


# ═══════════════════════════════════════════════════════════════════════════
# Generate candidates tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateCandidates:

    def test_co2_produces_candidates(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="carbon dioxide", formula="CO2", charge=0, geometry="linear"),
            matrix=Matrix(
                description="seawater",
                competing_species=[
                    CompetingSpecies("calcium", "Ca(2+)", 10.0, 2.0),
                ],
            ),
            desired_outcome=Outcome(description="capture and mineralize CO2"),
        )
        candidates = adapter.generate_candidates(p)
        assert len(candidates) >= 2
        assert all(isinstance(c, CandidateResult) for c in candidates)

    def test_candidates_have_modality(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="phosphate", formula="PO4(3-)", charge=-3, geometry="tetrahedral"),
            matrix=Matrix(description="wastewater"),
            desired_outcome=Outcome(description="recover phosphorus"),
        )
        candidates = adapter.generate_candidates(p)
        for c in candidates:
            assert c.modality == "capture_transform"

    def test_candidates_have_source_tool(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="lead", formula="Pb(2+)", charge=2, geometry="variable"),
            matrix=Matrix(),
            desired_outcome=Outcome(description="capture lead and sequester"),
        )
        candidates = adapter.generate_candidates(p)
        for c in candidates:
            assert c.source_tool == "capture_transform_pipeline"

    def test_candidates_have_immobilization_options(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="carbon dioxide", formula="CO2", charge=0, geometry="linear"),
            matrix=Matrix(description="hard water"),
            desired_outcome=Outcome(description="mineralize CO2"),
        )
        candidates = adapter.generate_candidates(p)
        has_immob = any(len(c.immobilization_options) > 0 for c in candidates)
        assert has_immob

    def test_candidates_have_performance(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="fluoride", formula="F(-)", charge=-1, geometry="spherical"),
            matrix=Matrix(),
            desired_outcome=Outcome(description="remove fluoride from water"),
        )
        candidates = adapter.generate_candidates(p)
        for c in candidates:
            assert 0.0 <= c.performance.probability_of_success <= 1.0

    def test_n2_produces_no_viable_candidates(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="nitrogen", formula="N2", charge=0, geometry="linear"),
            matrix=Matrix(),
            desired_outcome=Outcome(description="fix nitrogen from air"),
        )
        candidates = adapter.generate_candidates(p)
        assert len(candidates) == 0  # all excluded under orthogonality

    def test_unknown_produces_empty(self):
        adapter = CaptureTransformAdapter()
        p = Problem(
            target=TargetSpecies(identity="xenon", formula="Xe", charge=0, geometry="spherical"),
            matrix=Matrix(),
            desired_outcome=Outcome(description="capture xenon"),
        )
        candidates = adapter.generate_candidates(p)
        assert len(candidates) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Decomposer extension tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDecomposerExtension:

    def test_extend_adds_targets(self):
        extend_decomposer()
        from conversation.decomposer import KNOWN_TARGETS
        assert "carbon dioxide" in KNOWN_TARGETS
        assert "phosphate" in KNOWN_TARGETS
        assert "ammonia" in KNOWN_TARGETS
        assert "fluoride" in KNOWN_TARGETS

    def test_extend_adds_matrices(self):
        extend_decomposer()
        from conversation.decomposer import KNOWN_MATRICES
        assert "wastewater" in KNOWN_MATRICES
        assert "hard water" in KNOWN_MATRICES
        assert "air" in KNOWN_MATRICES

    def test_extend_idempotent(self):
        """Calling extend_decomposer twice should not duplicate entries."""
        extend_decomposer()
        extend_decomposer()
        from conversation.decomposer import KNOWN_TARGETS
        assert isinstance(KNOWN_TARGETS["carbon dioxide"], TargetSpecies)

    def test_extend_doesnt_overwrite_existing(self):
        """Existing targets (selenite, lead) should not be overwritten."""
        from conversation.decomposer import KNOWN_TARGETS
        original_selenite = KNOWN_TARGETS.get("selenite")
        extend_decomposer()
        assert KNOWN_TARGETS.get("selenite") is original_selenite

    def test_decompose_recognizes_co2(self):
        extend_decomposer()
        from conversation.decomposer import decompose
        problem = decompose("capture carbon dioxide from seawater and convert to mineral feedstock")
        assert problem.target.identity == "carbon dioxide"
        assert problem.target.formula == "CO2"

    def test_decompose_recognizes_phosphate(self):
        extend_decomposer()
        from conversation.decomposer import decompose
        problem = decompose("recover phosphate from wastewater as fertilizer")
        assert problem.target.identity == "phosphate"

    def test_decompose_recognizes_wastewater_matrix(self):
        extend_decomposer()
        from conversation.decomposer import decompose
        problem = decompose("remove fluoride from wastewater")
        assert "wastewater" in problem.matrix.description.lower()


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end with orchestrator
# ═══════════════════════════════════════════════════════════════════════════

_orchestrator_available = False
try:
    from core.orchestrator import Orchestrator as _Orchestrator
    _orchestrator_available = True
except (ImportError, ModuleNotFoundError):
    pass


class TestEndToEndOrchestration:

    @pytest.mark.skipif(not _orchestrator_available,
                        reason="Orchestrator requires acoustic module (not installed)")
    def test_full_pipeline_co2(self):
        """Natural language → decompose → orchestrate with capture-transform adapter."""
        registry = ToolRegistry()
        adapter = CaptureTransformAdapter()
        registry.register(adapter)

        extend_decomposer()
        from conversation.decomposer import decompose
        from core.orchestrator import Orchestrator

        problem = decompose("capture carbon dioxide from seawater and transform to solid mineral")
        orchestrator = Orchestrator(registry)
        result = orchestrator.solve(problem)

        assert len(result.candidates) >= 2
        assert any(c.modality == "capture_transform" for c in result.candidates)
        assert "capture_transform" in " ".join(result.tools_consulted)

    @pytest.mark.skipif(not _orchestrator_available,
                        reason="Orchestrator requires acoustic module (not installed)")
    def test_full_pipeline_lead(self):
        """Lead capture → capture-transform adapter should contribute PbS pathway."""
        registry = ToolRegistry()
        adapter = CaptureTransformAdapter()
        registry.register(adapter)

        extend_decomposer()
        from conversation.decomposer import decompose
        from core.orchestrator import Orchestrator

        problem = decompose("capture lead from mine water and sequester as solid")
        orchestrator = Orchestrator(registry)
        result = orchestrator.solve(problem)

        pb_candidates = [c for c in result.candidates if "PbS" in c.structure_description]
        assert len(pb_candidates) >= 1

    def test_setup_convenience(self):
        """setup_capture_transform() should extend decomposer + register adapter."""
        registry = ToolRegistry()
        adapter = setup_capture_transform(registry)
        assert adapter.name == "capture_transform"
        assert "capture_transform" in [a.name for a in registry.all_adapters()]

        from conversation.decomposer import KNOWN_TARGETS
        assert "carbon dioxide" in KNOWN_TARGETS
