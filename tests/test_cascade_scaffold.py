"""
tests/test_cascade_scaffold.py — Tests for cascade scaffold adapter.

Validates:
  - Physics models (confinement, diffusion, pore selectivity, retention)
  - Scaffold database completeness
  - Scaffold scoring against cascade specs
  - Scaffold ranking and recommendation
  - Module stoichiometry optimization
  - Cascade pattern library (Pattern 1: CO₂, Pattern 2: struvite)
  - Scale-dependent recommendations (DNA origami vs MOF)
  - Report generation
"""

import sys
import os
import math
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.cascade_scaffold import (
    # Physics
    confined_concentration_mM,
    confinement_enhancement,
    diffusion_time_ns,
    pore_selectivity,
    intermediate_retention,
    # Models
    CascadeModule,
    CascadeSpec,
    PoreSpec,
    ModuleRole,
    ScaffoldSystem,
    ScaleClass,
    ScaffoldScore,
    StoichiometryResult,
    # Functions
    score_scaffold,
    rank_scaffolds,
    recommend_scaffold,
    optimize_stoichiometry,
    cascade_report,
    # Pattern library
    build_cascade_spec_pattern1_co2,
    build_cascade_spec_pattern2_struvite,
    # Database
    _SCAFFOLDS,
)
from core.transform_enumerator import ClickCompatibility


# ═══════════════════════════════════════════════════════════════════════════
# Physics model tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConfinedConcentration:

    def test_single_molecule_in_20nm_cage(self):
        """1 molecule in a 20 nm cage should give ~mM concentration."""
        conc = confined_concentration_mM(1, 20.0)
        assert 0.1 < conc < 100.0  # order-of-magnitude check

    def test_more_molecules_higher_concentration(self):
        c1 = confined_concentration_mM(1, 20.0)
        c5 = confined_concentration_mM(5, 20.0)
        assert c5 == pytest.approx(5 * c1, rel=0.01)

    def test_smaller_cage_higher_concentration(self):
        c_big = confined_concentration_mM(1, 40.0)
        c_small = confined_concentration_mM(1, 20.0)
        assert c_small > c_big

    def test_zero_diameter_returns_zero(self):
        assert confined_concentration_mM(1, 0.0) == 0.0


class TestConfinementEnhancement:

    def test_enhancement_positive(self):
        """Confinement should always enhance vs dilute bulk."""
        enh = confinement_enhancement(20.0, 0.001)
        assert enh > 1.0

    def test_smaller_cage_bigger_enhancement(self):
        e20 = confinement_enhancement(20.0, 0.001)
        e10 = confinement_enhancement(10.0, 0.001)
        assert e10 > e20

    def test_enhancement_scales_with_bulk(self):
        """Lower bulk concentration → higher enhancement."""
        e_dilute = confinement_enhancement(20.0, 0.0001)
        e_conc = confinement_enhancement(20.0, 0.01)
        assert e_dilute > e_conc


class TestDiffusionTime:

    def test_short_distance_fast(self):
        """2 nm gap → sub-nanosecond for small ion."""
        t = diffusion_time_ns(2.0, 0.2)
        assert t < 10.0  # should be sub-ns to single-digit ns

    def test_longer_distance_slower(self):
        t_short = diffusion_time_ns(2.0, 0.2)
        t_long = diffusion_time_ns(10.0, 0.2)
        assert t_long > t_short

    def test_zero_distance_zero_time(self):
        assert diffusion_time_ns(0.0) == 0.0

    def test_larger_species_slower(self):
        t_small = diffusion_time_ns(5.0, 0.1)
        t_big = diffusion_time_ns(5.0, 0.5)
        assert t_big > t_small


class TestPoreSelectivity:

    def test_smaller_target_than_competitor_positive_selectivity(self):
        """If target is smaller than competitor, pore should favor target."""
        sel = pore_selectivity(0.2, 0.4, 0.5)
        assert sel > 1.0

    def test_competitor_excluded(self):
        """If competitor radius >= pore radius, selectivity → ∞."""
        sel = pore_selectivity(0.2, 0.5, 0.5)
        assert sel == float('inf')

    def test_equal_sizes_unity(self):
        """Same size target and competitor → selectivity = 1."""
        sel = pore_selectivity(0.3, 0.3, 1.0)
        assert sel == pytest.approx(1.0, abs=0.01)

    def test_zero_pore_unity(self):
        assert pore_selectivity(0.2, 0.3, 0.0) == 1.0


class TestIntermediateRetention:

    def test_large_intermediate_perfect_retention(self):
        """Intermediate bigger than pore → retention = 1."""
        ret = intermediate_retention(0.5, 0.3, 10.0)
        assert ret == 1.0

    def test_small_intermediate_lower_retention(self):
        """Small intermediate through large pore → lower retention."""
        ret = intermediate_retention(0.1, 1.0, 100.0)
        assert ret < 1.0

    def test_fast_transform_better_retention(self):
        """Faster transformation (shorter diffusion time) → better retention."""
        ret_fast = intermediate_retention(0.15, 0.5, 1.0)
        ret_slow = intermediate_retention(0.15, 0.5, 1000.0)
        assert ret_fast >= ret_slow


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold database tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScaffoldDatabase:

    def test_all_systems_have_properties(self):
        for system in ScaffoldSystem:
            assert system in _SCAFFOLDS
            props = _SCAFFOLDS[system]
            assert props.name != ""
            assert props.cavity_diameter_range_nm[1] >= props.cavity_diameter_range_nm[0]

    def test_dna_origami_largest_cavity(self):
        dna = _SCAFFOLDS[ScaffoldSystem.DNA_ORIGAMI]
        mof = _SCAFFOLDS[ScaffoldSystem.MOF_CAVITY]
        assert dna.cavity_diameter_range_nm[1] > mof.cavity_diameter_range_nm[1]

    def test_mof_scales_to_industrial(self):
        mof = _SCAFFOLDS[ScaffoldSystem.MOF_CAVITY]
        assert ScaleClass.INDUSTRIAL in mof.viable_scales

    def test_dna_origami_lab_only(self):
        dna = _SCAFFOLDS[ScaffoldSystem.DNA_ORIGAMI]
        assert ScaleClass.INDUSTRIAL not in dna.viable_scales
        assert ScaleClass.LAB in dna.viable_scales

    def test_dna_origami_highest_addressability(self):
        dna = _SCAFFOLDS[ScaffoldSystem.DNA_ORIGAMI]
        assert dna.addressability == "individual"


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold scoring tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScaffoldScoring:

    @pytest.fixture
    def simple_spec(self):
        from core.transform_enumerator import enumerate_transformations
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0})
        return CascadeSpec(
            name="test cascade",
            description="test",
            cascade_pattern="Pattern 1",
            target_formula="CO₂",
            product=products[0],
            modules=[
                CascadeModule("m1", "capture", ModuleRole.CAPTURE, "X", "test"),
                CascadeModule("m2", "activate", ModuleRole.ACTIVATION, "Y", "test"),
            ],
            min_interior_volume_nm3=10.0,
        )

    def test_score_has_composite(self, simple_spec):
        score = score_scaffold(ScaffoldSystem.DNA_ORIGAMI, simple_spec)
        assert 0.0 <= score.composite <= 1.0

    def test_all_scaffolds_scorable(self, simple_spec):
        for system in ScaffoldSystem:
            score = score_scaffold(system, simple_spec)
            assert isinstance(score, ScaffoldScore)

    def test_infeasible_if_volume_too_small(self):
        from core.transform_enumerator import enumerate_transformations
        products = enumerate_transformations("CO2")
        spec = CascadeSpec(
            name="huge volume", description="test",
            cascade_pattern="test",
            target_formula="CO₂",
            product=products[0],
            modules=[CascadeModule("m1", "x", ModuleRole.CAPTURE, "X", "test")],
            min_interior_volume_nm3=1e6,  # impossible
        )
        score = score_scaffold(ScaffoldSystem.FUJITA_CAGE, spec)
        assert not score.feasible

    def test_infeasible_if_too_many_modules(self):
        from core.transform_enumerator import enumerate_transformations
        products = enumerate_transformations("CO2")
        spec = CascadeSpec(
            name="too many", description="test",
            cascade_pattern="test",
            target_formula="CO₂",
            product=products[0],
            modules=[
                CascadeModule(f"m{i}", f"mod{i}", ModuleRole.CAPTURE, "X", "test",
                              stoichiometric_ratio=100.0)
                for i in range(10)
            ],
        )
        score = score_scaffold(ScaffoldSystem.NITSCHKE_CAGE, spec)
        assert not score.feasible

    def test_confinement_factor_positive(self, simple_spec):
        score = score_scaffold(ScaffoldSystem.DNA_ORIGAMI, simple_spec)
        assert score.confinement_factor > 1.0

    def test_diffusion_time_reasonable(self, simple_spec):
        score = score_scaffold(ScaffoldSystem.DNA_ORIGAMI, simple_spec)
        assert score.diffusion_time_ns > 0.0
        assert score.diffusion_time_ns < 1e6  # not geologic time


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold ranking tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScaffoldRanking:

    def test_ranking_sorted_by_composite(self):
        spec = build_cascade_spec_pattern1_co2()
        ranked = rank_scaffolds(spec)
        composites = [s.composite for s in ranked]
        assert composites == sorted(composites, reverse=True)

    def test_ranking_returns_all_systems(self):
        spec = build_cascade_spec_pattern1_co2()
        ranked = rank_scaffolds(spec)
        assert len(ranked) == len(ScaffoldSystem)

    def test_mof_ranks_high_for_field_scale(self):
        """MOF should rank highest for field-scale deployment."""
        spec = build_cascade_spec_pattern1_co2()
        spec.target_scale = ScaleClass.FIELD
        ranked = rank_scaffolds(spec)
        feasible = [s for s in ranked if s.feasible]
        # MOF should be in top 2 for field scale
        top2_systems = [s.scaffold.system for s in feasible[:2]]
        assert ScaffoldSystem.MOF_CAVITY in top2_systems

    def test_dna_origami_ranks_high_for_diagnostic(self):
        """DNA origami should rank well for diagnostic scale."""
        spec = build_cascade_spec_pattern1_co2()
        spec.target_scale = ScaleClass.DIAGNOSTIC
        ranked = rank_scaffolds(spec)
        feasible = [s for s in ranked if s.feasible]
        dna_ranks = [i for i, s in enumerate(feasible)
                     if s.scaffold.system == ScaffoldSystem.DNA_ORIGAMI]
        assert len(dna_ranks) >= 1
        assert dna_ranks[0] < 3  # top 3


# ═══════════════════════════════════════════════════════════════════════════
# Recommendation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRecommendation:

    def test_recommend_returns_score_and_rationale(self):
        spec = build_cascade_spec_pattern1_co2()
        best, rationale = recommend_scaffold(spec)
        assert isinstance(best, ScaffoldScore)
        assert len(rationale) > 0

    def test_field_scale_mentions_mof(self):
        spec = build_cascade_spec_pattern1_co2()
        spec.target_scale = ScaleClass.FIELD
        _, rationale = recommend_scaffold(spec)
        assert "MOF" in rationale or "mof" in rationale.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Module stoichiometry tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStoichiometry:

    def test_basic_optimization(self):
        spec = build_cascade_spec_pattern1_co2()
        result = optimize_stoichiometry(spec, available_positions=20)
        assert result.total_modules <= 20
        assert result.total_modules > 0
        assert result.limiting_module != ""

    def test_respects_position_budget(self):
        spec = build_cascade_spec_pattern2_struvite()
        result = optimize_stoichiometry(spec, available_positions=10)
        assert result.total_modules <= 10

    def test_excess_positions_go_to_capture(self):
        """Excess positions should be allocated to capture modules."""
        spec = build_cascade_spec_pattern1_co2()
        result_small = optimize_stoichiometry(spec, available_positions=10)
        result_large = optimize_stoichiometry(spec, available_positions=50)
        # Capture module ("zn_ca") should have more in the larger budget
        assert result_large.optimized_ratios.get("zn_ca", 0) >= \
               result_small.optimized_ratios.get("zn_ca", 0)

    def test_capacity_positive(self):
        spec = build_cascade_spec_pattern1_co2()
        result = optimize_stoichiometry(spec, available_positions=20)
        assert result.capacity_per_scaffold > 0

    def test_empty_spec(self):
        from core.transform_enumerator import enumerate_transformations
        products = enumerate_transformations("CO2")
        spec = CascadeSpec(
            name="empty", description="test", cascade_pattern="test",
            target_formula="CO₂", product=products[0], modules=[],
        )
        result = optimize_stoichiometry(spec, available_positions=20)
        assert result.total_modules == 0


# ═══════════════════════════════════════════════════════════════════════════
# Cascade pattern library tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPatternLibrary:

    def test_pattern1_co2_builds(self):
        spec = build_cascade_spec_pattern1_co2()
        assert spec.cascade_pattern == "Pattern 1"
        assert spec.target_formula == "CO₂"
        assert len(spec.modules) == 3

    def test_pattern1_has_capture_module(self):
        spec = build_cascade_spec_pattern1_co2()
        captures = [m for m in spec.modules if m.role == ModuleRole.CAPTURE]
        assert len(captures) >= 1
        assert "Zn" in captures[0].name or "CA" in captures[0].name

    def test_pattern1_has_co_reactant(self):
        spec = build_cascade_spec_pattern1_co2()
        crs = [m for m in spec.modules if m.role == ModuleRole.CO_REACTANT]
        assert len(crs) >= 1

    def test_pattern1_needs_copper_free(self):
        """Zn-CA mimic requires SPAAC → spec should flag copper-free."""
        spec = build_cascade_spec_pattern1_co2()
        assert spec.needs_copper_free

    def test_pattern2_struvite_builds(self):
        spec = build_cascade_spec_pattern2_struvite()
        assert spec.cascade_pattern == "Pattern 2"
        assert len(spec.modules) == 3

    def test_pattern2_has_dual_capture(self):
        """Pattern 2 should have two capture modules (PO₄ + NH₄)."""
        spec = build_cascade_spec_pattern2_struvite()
        captures = [m for m in spec.modules if m.role == ModuleRole.CAPTURE]
        assert len(captures) == 2

    def test_pattern2_has_pore_spec(self):
        spec = build_cascade_spec_pattern2_struvite()
        assert spec.pore_spec is not None
        assert spec.pore_spec.target_hydrated_radius_nm < spec.pore_spec.competitor_hydrated_radius_nm


# ═══════════════════════════════════════════════════════════════════════════
# Integration test: full pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_pattern1_rank_and_optimize(self):
        """Full pipeline: build spec → rank scaffolds → optimize stoichiometry."""
        spec = build_cascade_spec_pattern1_co2()
        ranked = rank_scaffolds(spec)
        best = ranked[0]
        assert best.feasible

        # Optimize for best scaffold
        positions = best.scaffold.module_positions[1]
        stoich = optimize_stoichiometry(spec, positions)
        assert stoich.total_modules > 0
        assert stoich.capacity_per_scaffold > 0

    def test_pattern2_rank_and_optimize(self):
        spec = build_cascade_spec_pattern2_struvite()
        ranked = rank_scaffolds(spec)
        feasible = [s for s in ranked if s.feasible]
        assert len(feasible) >= 3  # most scaffolds should work

    def test_field_scale_full_pipeline(self):
        """Field-scale struvite should recommend MOF."""
        spec = build_cascade_spec_pattern2_struvite()
        spec.target_scale = ScaleClass.FIELD
        best, rationale = recommend_scaffold(spec)
        assert best.feasible
        # At field scale, MOF should dominate
        ranked = rank_scaffolds(spec)
        top_feasible = [s for s in ranked if s.feasible][0]
        assert top_feasible.scaffold.system == ScaffoldSystem.MOF_CAVITY


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

class TestReporting:

    def test_report_not_empty(self):
        spec = build_cascade_spec_pattern1_co2()
        ranked = rank_scaffolds(spec)
        report = cascade_report(spec, ranked)
        assert "Cascade:" in report
        assert "Pattern 1" in report

    def test_report_includes_all_scaffolds(self):
        spec = build_cascade_spec_pattern1_co2()
        ranked = rank_scaffolds(spec)
        report = cascade_report(spec, ranked)
        for system in ScaffoldSystem:
            name = _SCAFFOLDS[system].name
            # At least part of the name should appear
            assert any(word in report for word in name.split()[:3])

    def test_report_shows_confinement(self):
        spec = build_cascade_spec_pattern1_co2()
        ranked = rank_scaffolds(spec)
        report = cascade_report(spec, ranked)
        assert "Confinement" in report
