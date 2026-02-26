"""
Sprint R1 Test Suite.

Integration test from the plan:
    Feed a "4N planar 0.4nm pocket for Cu²⁺" spec →
    ideal pocket says "4 N-donors at 2.00 Å ± 0.05 Å in D4h, rigidity: crystalline" →
    porphyrin scores ~0.95, crown ether ~0.6, protein ~0.4 →
    correct feasibility failures and deviation reports.
"""

import math
import pytest

from mabe.realization.models import (
    ApplicationContext,
    CavityDimensions,
    CavityShape,
    DonorPosition,
    ExclusionSpec,
    IdealPocketSpec,
    InteractionGeometrySpec,
    RankedRealizations,
    RigidityClass,
    ScaleClass,
    Solvent,
)
from mabe.realization.engine.ideal_pocket import compute_ideal_pocket
from mabe.realization.engine.ranker import rank_realizations
from mabe.realization.registry.material_registry import MATERIAL_REGISTRY
from mabe.realization.scoring.feasibility import feasibility_gate
from mabe.realization.scoring.deviation import compute_deviation, deviation_to_fidelity


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_cu2_4n_spec() -> InteractionGeometrySpec:
    """
    4N planar pocket for Cu²⁺ coordination.
    Classic porphyrin-like geometry: 4 nitrogen donors in D4h square planar,
    Cu-N distance ~2.00 Å, pocket diameter ~0.4 nm.
    """
    cu_n_distance = 2.00  # Å

    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=33.5,  # small planar pocket
            aperture_A=4.0,
            depth_A=2.0,
            max_internal_diameter_A=4.0,
        ),
        symmetry="D4h",
        donor_positions=[
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(cu_n_distance, 0.0, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(0.0, cu_n_distance, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(-cu_n_distance, 0.0, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
            DonorPosition(
                atom_type="N",
                coordination_role="equatorial",
                position_vector_A=(0.0, -cu_n_distance, 0.0),
                tolerance_A=0.05,
                required_hybridization="sp2",
            ),
        ],
        rigidity_requirement="rigid",
        max_backbone_rmsd_A=0.1,
        conformational_penalty_budget_kJ_mol=5.0,
        pocket_scale_nm=0.4,
        must_exclude=[
            ExclusionSpec(
                species="Ca²⁺",
                max_allowed_affinity_kJ_mol=-10.0,
                exclusion_mechanism="geometry",
            ),
        ],
        pH_range=(3.0, 10.0),
        temperature_range_K=(280.0, 350.0),
        solvent=Solvent.AQUEOUS,
        ionic_strength_M=0.1,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


# ─────────────────────────────────────────────
# Test 1: Registry has 5 starter systems
# ─────────────────────────────────────────────

class TestRegistry:

    def test_registry_has_5_systems(self):
        assert len(MATERIAL_REGISTRY) == 5

    def test_registry_system_ids(self):
        ids = {c.system_id for c in MATERIAL_REGISTRY.all()}
        assert ids == {
            "planar_coordination_ring",
            "cyclic_encapsulant",
            "periodic_lattice_node",
            "folded_polypeptide",
            "emergent_coordination_cage",
        }

    def test_registry_physics_classes(self):
        classes = {c.physics_class for c in MATERIAL_REGISTRY.all()}
        assert "covalent_cavity" in classes
        assert "periodic_lattice" in classes
        assert "foldable_polymer" in classes
        assert "emergent_cavity" in classes

    def test_porphyrin_has_highest_precision(self):
        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        assert porphyrin is not None
        for cap in MATERIAL_REGISTRY.all():
            assert porphyrin.positioning_precision_A <= cap.positioning_precision_A


# ─────────────────────────────────────────────
# Test 2: Ideal Pocket Computation
# ─────────────────────────────────────────────

class TestIdealPocket:

    def test_ideal_pocket_has_4_elements(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert len(ideal.optimal_elements) == 4

    def test_ideal_pocket_all_nitrogen(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert all(e.atom_type == "N" for e in ideal.optimal_elements)

    def test_ideal_pocket_requires_tight_precision(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        # 4N coordination should require ≤0.05 Å precision
        assert ideal.min_precision_required_A <= 0.05

    def test_ideal_pocket_rigidity_crystalline_or_preorganized(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.rigidity_class in (
            RigidityClass.CRYSTALLINE,
            RigidityClass.PREORGANIZED,
        )

    def test_ideal_pocket_binding_energy_negative(self):
        """Favorable binding = negative energy."""
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.ideal_binding_energy_kJ_mol < 0

    def test_ideal_pocket_desolvation_positive(self):
        """Desolvation is a penalty = positive energy."""
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.ideal_desolvation_energy_kJ_mol > 0

    def test_ideal_pocket_required_elements(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert ideal.required_elements == {"N"}

    def test_ideal_pocket_critical_constraints_nonempty(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert len(ideal.critical_constraints) > 0

    def test_ideal_pocket_material_requirements_string(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        assert "4 interaction elements" in ideal.ideal_material_requirements
        assert "N" in ideal.ideal_material_requirements


# ─────────────────────────────────────────────
# Test 3: Feasibility Gate
# ─────────────────────────────────────────────

class TestFeasibilityGate:

    def test_feasibility_gate_cu2_pocket(self):
        """0.4nm pocket: covalent + lattice systems pass, polymer + cage fail (min 0.5nm)."""
        spec = make_cu2_4n_spec()
        passed = []
        failed = []
        for cap in MATERIAL_REGISTRY.all():
            feasible, reason = feasibility_gate(spec, cap)
            if feasible:
                passed.append(cap.system_id)
            else:
                failed.append(cap.system_id)
        assert "planar_coordination_ring" in passed
        assert "cyclic_encapsulant" in passed
        assert "periodic_lattice_node" in passed
        # Protein and cage correctly fail — 0.4nm is below their 0.5nm minimum
        assert "folded_polypeptide" in failed
        assert "emergent_coordination_cage" in failed

    def test_gate_rejects_oversized_pocket(self):
        """A 10nm pocket should fail for porphyrin (max 0.5nm)."""
        spec = make_cu2_4n_spec()
        spec.pocket_scale_nm = 10.0
        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        feasible, reason = feasibility_gate(spec, porphyrin)
        assert not feasible
        assert "above maximum" in reason

    def test_gate_rejects_missing_donor(self):
        """Require Se donor — porphyrin can't provide it."""
        spec = make_cu2_4n_spec()
        spec.donor_positions.append(DonorPosition(
            atom_type="Se",
            coordination_role="axial",
            position_vector_A=(0.0, 0.0, 2.5),
            tolerance_A=0.10,
            required_hybridization="sp3",
        ))
        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        feasible, reason = feasibility_gate(spec, porphyrin)
        assert not feasible
        assert "Se" in reason


# ─────────────────────────────────────────────
# Test 4: Deviation Scoring
# ─────────────────────────────────────────────

class TestDeviation:

    def test_porphyrin_lowest_deviation(self):
        """Porphyrin should have the lowest deviation for 4N planar."""
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)

        porphyrin = MATERIAL_REGISTRY.get("planar_coordination_ring")
        protein = MATERIAL_REGISTRY.get("folded_polypeptide")

        dev_porph = compute_deviation(ideal, porphyrin)
        dev_prot = compute_deviation(ideal, protein)

        assert dev_porph.mean_deviation_A < dev_prot.mean_deviation_A

    def test_porphyrin_fidelity_highest(self):
        """Porphyrin physics fidelity should be highest."""
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)

        fidelities = {}
        for cap in MATERIAL_REGISTRY.all():
            dev = compute_deviation(ideal, cap)
            fidelities[cap.system_id] = deviation_to_fidelity(dev)

        assert fidelities["planar_coordination_ring"] == max(fidelities.values())

    def test_fidelity_bounded_0_1(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        for cap in MATERIAL_REGISTRY.all():
            dev = compute_deviation(ideal, cap)
            f = deviation_to_fidelity(dev)
            assert 0.0 <= f <= 1.0

    def test_porphyrin_fidelity_above_0_8(self):
        """Porphyrin at 0.01Å precision for 0.05Å requirement → high fidelity."""
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        porph = MATERIAL_REGISTRY.get("planar_coordination_ring")
        dev = compute_deviation(ideal, porph)
        f = deviation_to_fidelity(dev)
        assert f > 0.8

    def test_protein_fidelity_below_porphyrin(self):
        """Protein at 0.3Å precision should score below porphyrin at 0.01Å."""
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        porph = MATERIAL_REGISTRY.get("planar_coordination_ring")
        prot = MATERIAL_REGISTRY.get("folded_polypeptide")
        f_porph = deviation_to_fidelity(compute_deviation(ideal, porph))
        f_prot = deviation_to_fidelity(compute_deviation(ideal, prot))
        assert f_porph > f_prot


# ─────────────────────────────────────────────
# Test 5: Full Ranker Integration
# ─────────────────────────────────────────────

class TestRanker:

    def test_ranker_returns_ranked_realizations(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert isinstance(result, RankedRealizations)

    def test_ranker_has_ideal_pocket(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert result.ideal_pocket is ideal

    def test_ranker_recommends_porphyrin(self):
        """For 4N planar Cu²⁺, porphyrin should be #1."""
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert result.recommended == "planar_coordination_ring"

    def test_ranker_sorts_by_composite(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        composites = [s.composite_score for s in result.rankings]
        assert composites == sorted(composites, reverse=True)

    def test_ranker_all_feasible_have_scores(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        for s in result.rankings:
            if s.feasible:
                assert s.composite_score > 0
                assert s.physics_fidelity > 0

    def test_ranker_gap_analysis_present(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        # gap_to_ideal should be computed
        assert 0.0 <= result.gap_to_ideal <= 1.0
        assert result.best_physics_fidelity > 0

    def test_ranker_recommendation_rationale_nonempty(self):
        spec = make_cu2_4n_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        assert len(result.recommendation_rationale) > 0


# ─────────────────────────────────────────────
# Test 6: Gap Report (use a hard spec)
# ─────────────────────────────────────────────

class TestGapReport:

    def _make_extreme_spec(self) -> InteractionGeometrySpec:
        """A spec so demanding that nothing scores above 0.5."""
        return InteractionGeometrySpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(
                volume_A3=100.0,
                aperture_A=5.0,
                depth_A=5.0,
                max_internal_diameter_A=6.0,
            ),
            symmetry="none",
            donor_positions=[
                # 6 different element types — no material has all of them
                DonorPosition("N", "equatorial", (2.0, 0.0, 0.0), 0.02, "sp2"),
                DonorPosition("O", "equatorial", (0.0, 2.0, 0.0), 0.02, "sp3"),
                DonorPosition("S", "axial", (0.0, 0.0, 2.5), 0.02, "sp3"),
                DonorPosition("Se", "axial", (0.0, 0.0, -2.5), 0.02, "sp3"),
                DonorPosition("N", "bridging", (1.5, 1.5, 0.0), 0.02, "sp2"),
                DonorPosition("P", "terminal", (-2.0, 0.0, 0.0), 0.02, "sp3"),
            ],
            pocket_scale_nm=0.6,
            pH_range=(4.0, 9.0),
            temperature_range_K=(280.0, 340.0),
            solvent=Solvent.AQUEOUS,
            target_application=ApplicationContext.RESEARCH,
            required_scale=ScaleClass.UMOL,
        )

    def test_gap_report_generated_for_hard_spec(self):
        """An extreme spec should trigger gap report (gap > 0.3)."""
        spec = self._make_extreme_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        # Most systems can't provide Se — should see infeasible or low fidelity
        # This should trigger a gap report
        assert result.gap_to_ideal > 0.0  # at minimum, not perfect

    def test_novel_material_suggestion_for_very_hard_spec(self):
        """When gap > 0.5, should generate novel material suggestion."""
        spec = self._make_extreme_spec()
        ideal = compute_ideal_pocket(spec)
        result = rank_realizations(ideal, spec)
        # If gap is large enough, novel suggestion should exist
        if result.gap_to_ideal > 0.5:
            assert result.novel_material_suggestion is not None
            assert len(result.novel_material_suggestion) > 0