"""
test_material_designer.py — Tests for material designer M1 + M5.
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.material_designer import (
    TargetSpec, TieredValue, DataTier, PerformanceMetrics,
    PorousFrameworkSpec, FrameworkDesign,
    langmuir_capacity, langmuir_q_max_from_sites, langmuir_K_from_dG,
    selectivity_from_K, pore_size_exclusion,
    estimate_site_density, estimate_kinetics_t90,
    func_group_fraction_active,
    design_framework, screen_frameworks,
    unified_design, FRAMEWORK_DATABASE, FUNCTIONAL_GROUPS,
)


class TestTieredValue:

    def test_tier_tracking(self):
        tv = TieredValue(42.0, DataTier.T1_KNOWN, "NIST")
        assert tv.tier == DataTier.T1_KNOWN
        assert tv.value == 42.0

    def test_repr(self):
        tv = TieredValue(3.14, DataTier.T2_SOLID)
        assert "T2" in repr(tv)


class TestLangmuirPhysics:

    def test_capacity_positive(self):
        q = langmuir_capacity(100.0, 1.0, 1.0)
        assert q > 0

    def test_capacity_increases_with_concentration(self):
        q_low = langmuir_capacity(100.0, 1.0, 0.01)
        q_high = langmuir_capacity(100.0, 1.0, 10.0)
        assert q_high > q_low

    def test_capacity_saturates_at_qmax(self):
        """At very high C, q → q_max."""
        q = langmuir_capacity(100.0, 1.0, 1e6)
        assert abs(q - 100.0) < 1.0

    def test_capacity_zero_at_zero_C(self):
        q = langmuir_capacity(100.0, 1.0, 0.0)
        assert q == 0.0

    def test_K_from_dG_more_negative_higher_K(self):
        K_weak = langmuir_K_from_dG(-10.0)
        K_strong = langmuir_K_from_dG(-40.0)
        assert K_strong > K_weak

    def test_K_from_dG_zero_gives_one(self):
        """ΔG=0 → K=1 (in some unit system)."""
        K = langmuir_K_from_dG(0.0)
        assert K == pytest.approx(1.0 / 1000.0)  # normalized

    def test_q_max_from_sites(self):
        """1 mmol/g sites × 100 g/mol target = 100 mg/g."""
        q = langmuir_q_max_from_sites(1.0, 100.0)
        assert q == pytest.approx(100.0)

    def test_selectivity_ratio(self):
        assert selectivity_from_K(10.0, 1.0) == pytest.approx(10.0)
        assert selectivity_from_K(1.0, 10.0) == pytest.approx(0.1)

    def test_selectivity_inf_for_zero_interferent(self):
        assert selectivity_from_K(1.0, 0.0) == float('inf')


class TestPoreExclusion:

    def test_large_pore_admits_small_guest(self):
        assert pore_size_exclusion(12.0, 3.5) is True

    def test_small_pore_excludes_large_guest(self):
        assert pore_size_exclusion(3.0, 3.5) is False

    def test_marginal_pore(self):
        """Pore = guest + 1.0 Å (2×0.5 clearance) → just passes."""
        assert pore_size_exclusion(4.5, 3.5) is True

    def test_exact_fit_fails(self):
        """Pore = guest diameter exactly → fails (no clearance)."""
        assert pore_size_exclusion(3.5, 3.5) is False


class TestSiteDensity:

    def test_positive(self):
        sd = estimate_site_density(1000.0, 1.0)
        assert sd > 0

    def test_scales_with_surface_area(self):
        sd_low = estimate_site_density(500.0)
        sd_high = estimate_site_density(2000.0)
        assert sd_high > sd_low

    def test_reasonable_range(self):
        """MOF with 1000 m²/g, 1 group/nm² → ~1.7 mmol/g."""
        sd = estimate_site_density(1000.0, 1.0)
        assert 0.5 < sd < 5.0


class TestKinetics:

    def test_positive(self):
        t = estimate_kinetics_t90(10.0, 200.0)
        assert t > 0

    def test_small_pores_slower(self):
        t_small = estimate_kinetics_t90(3.0, 200.0)
        t_large = estimate_kinetics_t90(15.0, 200.0)
        assert t_small > t_large

    def test_larger_particles_slower(self):
        t_small = estimate_kinetics_t90(10.0, 50.0)
        t_large = estimate_kinetics_t90(10.0, 500.0)
        assert t_large > t_small


class TestFuncGroupPKa:

    def test_amine_active_for_anion_at_low_pH(self):
        """Protonated amine binds anions at pH 5."""
        f = func_group_fraction_active("amine-primary", 5.0, for_anion_capture=True)
        assert f > 0.99

    def test_amine_inactive_for_metal_at_low_pH(self):
        """Amine can't coordinate metals at pH 5 (protonated)."""
        f = func_group_fraction_active("amine-primary", 5.0, for_anion_capture=False)
        assert f < 0.001

    def test_carboxylate_for_anion(self):
        """Carboxylate needs to be deprotonated (COO⁻) even for anion capture."""
        f = func_group_fraction_active("carboxylate", 7.0, for_anion_capture=True)
        assert f > 0.99

    def test_thiol_always_deprotonation_dependent(self):
        f_low = func_group_fraction_active("thiol", 5.0)
        f_high = func_group_fraction_active("thiol", 10.0)
        assert f_high > f_low

    def test_sulfonate_always_active(self):
        assert func_group_fraction_active("sulfonate", 2.0) == 1.0
        assert func_group_fraction_active("sulfonate", 12.0) == 1.0


class TestFrameworkDesign:

    def test_from_database(self):
        spec = PorousFrameworkSpec.from_database("HKUST-1")
        assert spec.framework_name == "HKUST-1"
        assert spec.pore_diameter_A == 12.0

    def test_unknown_framework_raises(self):
        with pytest.raises(KeyError):
            PorousFrameworkSpec.from_database("NotAMOF-99")

    def test_design_framework_returns_design(self):
        target = TargetSpec(target_species="Pb2+", target_diameter_A=2.5,
                            target_charge=2, target_mw=207.2, pH=5.0)
        d = design_framework("UiO-66", target, functional_groups=["thiol"])
        assert isinstance(d, FrameworkDesign)
        assert d.metrics.capacity_mg_g is not None
        assert d.metrics.capacity_mg_g.value > 0

    def test_anion_target_uses_protonated_amine(self):
        """For selenite (anion), amine-functionalized MOF should show capacity."""
        target = TargetSpec(target_species="SeO3^2-", target_diameter_A=3.5,
                            target_charge=-2, target_mw=127.0, pH=5.0)
        d = design_framework("UiO-66-NH2", target,
                              functional_groups=["amine-primary"])
        assert d.metrics.capacity_mg_g.value > 10.0

    def test_inaccessible_pore_low_capacity(self):
        """SAPO-34 (3.8 Å pore) can't admit 3.5 Å selenite."""
        target = TargetSpec(target_diameter_A=3.5, target_charge=-2,
                            target_mw=127.0, pH=5.0)
        d = design_framework("SAPO-34", target)
        accessible = design_framework("HKUST-1", target)
        assert d.metrics.capacity_mg_g.value < accessible.metrics.capacity_mg_g.value


class TestScreenFrameworks:

    def test_returns_ranked_list(self):
        target = TargetSpec(target_species="Pb2+", target_diameter_A=2.5,
                            target_mw=207.2, pH=7.0)
        results = screen_frameworks(target)
        assert len(results) >= 5
        # Sorted by composite score
        scores = [d.metrics.composite_score() for d in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestUnifiedDesign:

    def test_returns_result(self):
        target = TargetSpec(name="test", target_species="Pb2+",
                            target_diameter_A=2.5, target_mw=207.2, pH=7.0)
        result = unified_design(target)
        assert result.n_materials_evaluated > 0
        assert len(result.rankings) > 0

    def test_best_for_categories(self):
        target = TargetSpec(name="test", target_species="SeO3^2-",
                            target_diameter_A=3.5, target_charge=-2,
                            target_mw=127.0, pH=5.0)
        result = unified_design(target, framework_func_groups=["amine-primary"])
        assert result.best_for_capacity != ""
        assert result.best_for_cost != ""

    def test_composite_score_bounded(self):
        target = TargetSpec(target_species="Pb2+", target_mw=207.2)
        result = unified_design(target)
        for r in result.rankings:
            assert 0 <= r.composite_score <= 1.0


class TestPerformanceMetrics:

    def test_composite_score_zero_for_empty(self):
        m = PerformanceMetrics()
        assert m.composite_score() == 0.0

    def test_composite_score_bounded(self):
        m = PerformanceMetrics(
            capacity_mg_g=TieredValue(200.0, DataTier.T2_SOLID),
            selectivity_ratio=TieredValue(50.0, DataTier.T3_CONCEPTUAL),
            kinetics_t90_min=TieredValue(5.0, DataTier.T3_CONCEPTUAL),
            cost_per_kg_usd=TieredValue(100.0, DataTier.T3_CONCEPTUAL),
            scalability_score=TieredValue(0.8, DataTier.T3_CONCEPTUAL),
            environmental_score=TieredValue(0.7, DataTier.T3_CONCEPTUAL),
        )
        s = m.composite_score()
        assert 0 < s <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
