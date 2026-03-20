"""
tests/test_harvesting_pareto.py -- Tests for 4-objective harvesting Pareto optimizer.

Validates:
  - 4-D dominance logic
  - Pareto front extraction
  - Grid search enumeration
  - Weighted recommendation selection
  - Power objective correctness (negate = maximize)
  - Physical sanity
  - Edge cases
"""

import sys
import os
import math
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.harvesting_pareto import (
    HarvestDesignPoint,
    HarvestParetoResult,
    _dominates_4d,
    extract_pareto_front_4d,
    optimize_harvesting,
)
from core.energy_harvesting import EnvironmentSpec


# -----------------------------------------------------------------------
# Dominance logic
# -----------------------------------------------------------------------

class TestDominance4D:

    def test_strictly_better_dominates(self):
        a = HarvestDesignPoint("A", 300, "A", "A",
                               obj_color_delta_e=1.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.6, obj_power_neg_W_m2=-10.0)
        b = HarvestDesignPoint("B", 300, "B", "B",
                               obj_color_delta_e=5.0, obj_thermal_neg_R=-0.05,
                               obj_acoustic_neg_nrc=-0.3, obj_power_neg_W_m2=-5.0)
        assert _dominates_4d(a, b)
        assert not _dominates_4d(b, a)

    def test_equal_does_not_dominate(self):
        a = HarvestDesignPoint("A", 300, "A", "A",
                               obj_color_delta_e=5.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.5, obj_power_neg_W_m2=-7.0)
        b = HarvestDesignPoint("A", 300, "A", "A",
                               obj_color_delta_e=5.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.5, obj_power_neg_W_m2=-7.0)
        assert not _dominates_4d(a, b)

    def test_tradeoff_no_dominance(self):
        """One better on power, other better on color -> neither dominates."""
        a = HarvestDesignPoint("A", 300, "A", "A",
                               obj_color_delta_e=2.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.5, obj_power_neg_W_m2=-5.0)
        b = HarvestDesignPoint("B", 300, "B", "B",
                               obj_color_delta_e=8.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.5, obj_power_neg_W_m2=-15.0)
        assert not _dominates_4d(a, b)
        assert not _dominates_4d(b, a)


# -----------------------------------------------------------------------
# Pareto front extraction
# -----------------------------------------------------------------------

class TestParetoFront:

    def test_single_design_is_pareto(self):
        d = HarvestDesignPoint("A", 300, "A", "A",
                               obj_power_neg_W_m2=-10.0)
        front = extract_pareto_front_4d([d])
        assert len(front) == 1
        assert front[0].is_pareto

    def test_dominated_excluded(self):
        a = HarvestDesignPoint("A", 300, "A", "A",
                               obj_color_delta_e=1.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.6, obj_power_neg_W_m2=-10.0)
        b = HarvestDesignPoint("B", 300, "B", "B",
                               obj_color_delta_e=5.0, obj_thermal_neg_R=-0.05,
                               obj_acoustic_neg_nrc=-0.3, obj_power_neg_W_m2=-5.0)
        front = extract_pareto_front_4d([a, b])
        assert len(front) == 1
        assert front[0].pv_material == "A"

    def test_tradeoff_both_on_front(self):
        a = HarvestDesignPoint("A", 300, "A", "A",
                               obj_color_delta_e=2.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.5, obj_power_neg_W_m2=-5.0)
        b = HarvestDesignPoint("B", 300, "B", "B",
                               obj_color_delta_e=8.0, obj_thermal_neg_R=-0.1,
                               obj_acoustic_neg_nrc=-0.5, obj_power_neg_W_m2=-15.0)
        front = extract_pareto_front_4d([a, b])
        assert len(front) == 2


# -----------------------------------------------------------------------
# Grid search optimizer
# -----------------------------------------------------------------------

class TestOptimizer:

    def test_runs_with_defaults(self):
        result = optimize_harvesting()
        assert isinstance(result, HarvestParetoResult)
        assert result.n_designs > 0

    def test_grid_size_correct(self):
        """With specific inputs, grid should be n_pv * n_t * n_te * n_piezo."""
        result = optimize_harvesting(
            pv_materials=["organic_PM6Y6", "amorphous_Si"],
            pv_thicknesses_nm=[300.0],
            te_materials=["Bi2Te3"],
            piezo_materials=["PVDF"],
        )
        assert result.n_designs == 2  # 2 PV * 1 thickness * 1 TE * 1 piezo

    def test_full_grid_size(self):
        """Default grid: 6 PV * 3 thicknesses * 5 TE * 5 piezo = 450."""
        result = optimize_harvesting()
        assert result.n_designs == 6 * 3 * 5 * 5

    def test_pareto_front_not_empty(self):
        result = optimize_harvesting()
        assert result.n_pareto > 0

    def test_recommendation_exists(self):
        result = optimize_harvesting()
        assert result.recommended is not None
        assert result.recommended.is_pareto

    def test_power_objective_negative(self):
        """Power objective should be negative (negate to maximize)."""
        result = optimize_harvesting()
        for d in result.all_designs:
            assert d.obj_power_neg_W_m2 <= 0

    def test_higher_irradiance_higher_power(self):
        """More sun should shift the front to higher power."""
        r_low = optimize_harvesting(
            environment=EnvironmentSpec(solar_irradiance_W_m2=100.0),
            pv_materials=["organic_PM6Y6"],
            pv_thicknesses_nm=[300.0],
            te_materials=["Bi2Te3"],
            piezo_materials=["PVDF"],
        )
        r_high = optimize_harvesting(
            environment=EnvironmentSpec(solar_irradiance_W_m2=800.0),
            pv_materials=["organic_PM6Y6"],
            pv_thicknesses_nm=[300.0],
            te_materials=["Bi2Te3"],
            piezo_materials=["PVDF"],
        )
        # Higher irradiance -> more negative power objective (more power)
        assert r_high.recommended.obj_power_neg_W_m2 < r_low.recommended.obj_power_neg_W_m2

    def test_custom_weights(self):
        """Power-heavy weights should select high-power design."""
        result = optimize_harvesting(
            weights=(0.0, 0.0, 0.0, 1.0),
            pv_materials=["organic_PM6Y6", "perovskite_MAPbI3"],
            pv_thicknesses_nm=[300.0],
            te_materials=["Bi2Te3", "SnSe"],
            piezo_materials=["PVDF"],
        )
        # Recommended should have the best (most negative) power
        all_powers = [d.obj_power_neg_W_m2 for d in result.pareto_front]
        assert result.recommended.obj_power_neg_W_m2 == min(all_powers)

    def test_power_budget_populated(self):
        result = optimize_harvesting(
            pv_materials=["organic_PM6Y6"],
            pv_thicknesses_nm=[300.0],
            te_materials=["Bi2Te3"],
            piezo_materials=["PVDF"],
        )
        d = result.all_designs[0]
        assert d.power_budget is not None
        assert d.power_budget.total_W_m2 > 0

    def test_summary_string(self):
        result = optimize_harvesting(
            pv_materials=["organic_PM6Y6"],
            pv_thicknesses_nm=[300.0],
            te_materials=["Bi2Te3"],
            piezo_materials=["PVDF"],
        )
        s = result.summary()
        assert "Harvest Pareto" in s


# -----------------------------------------------------------------------
# Physical sanity
# -----------------------------------------------------------------------

class TestPhysicalSanity:

    def test_perovskite_highest_power(self):
        """MAPbI3 (highest PCE) should yield highest PV power."""
        result = optimize_harvesting(
            pv_materials=["organic_PM6Y6", "perovskite_MAPbI3"],
            pv_thicknesses_nm=[300.0],
            te_materials=["Bi2Te3"],
            piezo_materials=["PVDF"],
        )
        powers = {d.pv_material: d.power_budget.pv_W_m2 for d in result.all_designs}
        assert powers["perovskite_MAPbI3"] > powers["organic_PM6Y6"]

    def test_all_powers_positive(self):
        result = optimize_harvesting()
        for d in result.all_designs:
            assert d.power_budget.total_W_m2 >= 0
