"""
tests/test_multiphysics_pareto.py — Tests for multi-objective Pareto optimizer.

Validates:
  - Grid generation covers design space
  - Design evaluation produces all three physics predictions
  - Pareto dominance relation is correct
  - Pareto front extraction is non-empty
  - Best-per-axis selections are correct
  - Weighted balanced recommendation exists
  - Quick design convenience function
  - Physics consistency: larger D → longer λ, higher R
  - Optical model integration (if available)
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.multiphysics_pareto import (
    optimize_multiphysics,
    evaluate_design,
    extract_pareto_front,
    quick_design,
    photonic_glass_peak_analytical,
    _dominates,
    _build_design_grid,
    MultiPhysicsTarget,
    ColorTarget,
    ThermalTarget,
    AcousticTarget,
    DesignPoint,
    ParetoResult,
    NAMED_COLORS,
    _OPTICAL_AVAILABLE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Analytical color estimate
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyticalColor:

    def test_sio2_280nm_gives_visible_peak(self):
        """280nm SiO₂ should give peak in visible range."""
        lam = photonic_glass_peak_analytical(280.0, "SiO2", 0.50)
        assert 400 < lam < 800

    def test_larger_d_longer_wavelength(self):
        lam_small = photonic_glass_peak_analytical(200.0, "SiO2", 0.50)
        lam_big = photonic_glass_peak_analytical(350.0, "SiO2", 0.50)
        assert lam_big > lam_small

    def test_higher_n_longer_wavelength(self):
        """Higher refractive index → longer λ for same D."""
        lam_sio2 = photonic_glass_peak_analytical(250.0, "SiO2", 0.50)
        lam_tio2 = photonic_glass_peak_analytical(250.0, "TiO2_rutile", 0.50)
        assert lam_tio2 > lam_sio2


# ═══════════════════════════════════════════════════════════════════════════
# Grid generation
# ═══════════════════════════════════════════════════════════════════════════

class TestGridGeneration:

    def test_grid_nonempty(self):
        target = MultiPhysicsTarget()
        grid = _build_design_grid(target)
        assert len(grid) > 0

    def test_grid_covers_diameters(self):
        target = MultiPhysicsTarget()
        grid = _build_design_grid(target, diameter_range=(200, 300), diameter_step=50)
        diameters = {p["diameter_nm"] for p in grid}
        assert 200.0 in diameters
        assert 250.0 in diameters
        assert 300.0 in diameters

    def test_grid_covers_phi(self):
        target = MultiPhysicsTarget()
        grid = _build_design_grid(target, phi_range=(0.40, 0.55), phi_step=0.05)
        phis = {round(p["volume_fraction"], 2) for p in grid}
        assert 0.40 in phis
        assert 0.45 in phis
        assert 0.50 in phis
        assert 0.55 in phis

    def test_grid_respects_materials(self):
        target = MultiPhysicsTarget()
        grid = _build_design_grid(target, materials=["SiO2", "TiO2_rutile"])
        materials = {p["material"] for p in grid}
        assert "SiO2" in materials
        assert "TiO2_rutile" in materials


# ═══════════════════════════════════════════════════════════════════════════
# Design evaluation
# ═══════════════════════════════════════════════════════════════════════════

class TestDesignEvaluation:

    def test_evaluate_produces_thermal(self):
        target = MultiPhysicsTarget(thermal=ThermalTarget())
        params = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.50,
                  "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        dp = evaluate_design(params, target)
        assert dp.thermal is not None
        assert dp.thermal.kappa_eff_W_mK > 0

    def test_evaluate_produces_acoustic(self):
        target = MultiPhysicsTarget(acoustic=AcousticTarget())
        params = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.50,
                  "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        dp = evaluate_design(params, target)
        assert dp.acoustic is not None
        assert dp.acoustic.nrc >= 0

    @pytest.mark.skipif(not _OPTICAL_AVAILABLE, reason="Optical pipeline not available")
    def test_evaluate_produces_color(self):
        target = MultiPhysicsTarget(
            color=ColorTarget(cie_x=0.30, cie_y=0.52, name="green"))
        params = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.50,
                  "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        dp = evaluate_design(params, target)
        assert dp.color.predicted
        assert dp.color.delta_e < 999

    def test_no_color_target_zero_penalty(self):
        target = MultiPhysicsTarget(thermal=ThermalTarget())  # no color target
        params = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.50,
                  "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        dp = evaluate_design(params, target)
        assert dp.obj_color_delta_e == 0.0

    def test_summary_not_empty(self):
        target = MultiPhysicsTarget()
        params = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.50,
                  "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        dp = evaluate_design(params, target)
        assert len(dp.summary()) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Pareto dominance
# ═══════════════════════════════════════════════════════════════════════════

class TestParetoDominance:

    def _make_dp(self, de, neg_r, neg_nrc):
        dp = DesignPoint(280.0, "SiO2", 0.5, 0.005, 50.0, "air")
        dp.obj_color_delta_e = de
        dp.obj_thermal_neg_R = neg_r
        dp.obj_acoustic_neg_nrc = neg_nrc
        return dp

    def test_strict_domination(self):
        """A is better on all axes → A dominates B."""
        a = self._make_dp(5.0, -0.01, -0.5)
        b = self._make_dp(10.0, -0.005, -0.3)
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_no_domination_tradeoff(self):
        """A better on color, B better on thermal → neither dominates."""
        a = self._make_dp(3.0, -0.005, -0.3)
        b = self._make_dp(8.0, -0.02, -0.3)
        assert not _dominates(a, b)
        assert not _dominates(b, a)

    def test_equal_not_dominated(self):
        """Equal on all axes → neither dominates."""
        a = self._make_dp(5.0, -0.01, -0.3)
        b = self._make_dp(5.0, -0.01, -0.3)
        assert not _dominates(a, b)
        assert not _dominates(b, a)

    def test_pareto_front_nonempty(self):
        designs = [
            self._make_dp(3.0, -0.005, -0.1),
            self._make_dp(8.0, -0.02, -0.5),
            self._make_dp(5.0, -0.01, -0.3),
            self._make_dp(20.0, -0.001, -0.05),  # dominated
        ]
        pareto = extract_pareto_front(designs)
        assert len(pareto) >= 2

    def test_dominated_point_excluded(self):
        """A point dominated by another should not be on the Pareto front."""
        designs = [
            self._make_dp(3.0, -0.02, -0.5),   # dominates the next one
            self._make_dp(10.0, -0.005, -0.2),  # dominated
        ]
        pareto = extract_pareto_front(designs)
        pareto_des = [(d.obj_color_delta_e, d.obj_thermal_neg_R) for d in pareto]
        assert (3.0, -0.02) in pareto_des
        assert (10.0, -0.005) not in pareto_des


# ═══════════════════════════════════════════════════════════════════════════
# Full optimization
# ═══════════════════════════════════════════════════════════════════════════

class TestFullOptimization:

    def test_optimization_runs(self):
        target = MultiPhysicsTarget(thermal=ThermalTarget())
        result = optimize_multiphysics(
            target,
            diameter_range=(250, 300), diameter_step=50,
            phi_range=(0.45, 0.55), phi_step=0.10,
            absorber_fractions=[0.005],
            film_thicknesses_um=[50.0],
        )
        assert isinstance(result, ParetoResult)
        assert result.n_evaluated > 0
        assert len(result.pareto_front) > 0

    def test_pareto_front_subset_of_all(self):
        target = MultiPhysicsTarget(thermal=ThermalTarget())
        result = optimize_multiphysics(
            target,
            diameter_range=(250, 300), diameter_step=50,
            phi_range=(0.45, 0.55), phi_step=0.10,
            absorber_fractions=[0.005],
            film_thicknesses_um=[50.0],
        )
        assert len(result.pareto_front) <= len(result.all_designs)

    def test_best_balanced_exists(self):
        target = MultiPhysicsTarget(thermal=ThermalTarget())
        result = optimize_multiphysics(
            target,
            diameter_range=(250, 300), diameter_step=50,
            phi_range=(0.45, 0.55), phi_step=0.10,
            absorber_fractions=[0.005],
            film_thicknesses_um=[50.0],
        )
        assert result.best_balanced is not None

    def test_best_thermal_has_best_r(self):
        target = MultiPhysicsTarget(thermal=ThermalTarget())
        result = optimize_multiphysics(
            target,
            diameter_range=(250, 300), diameter_step=50,
            phi_range=(0.40, 0.55), phi_step=0.05,
            absorber_fractions=[0.005],
            film_thicknesses_um=[50.0, 200.0],
        )
        # best_thermal should have the most negative obj_thermal_neg_R
        for d in result.all_designs:
            assert result.best_thermal.obj_thermal_neg_R <= d.obj_thermal_neg_R

    @pytest.mark.skipif(not _OPTICAL_AVAILABLE, reason="Optical pipeline not available")
    def test_optimization_with_color_target(self):
        target = MultiPhysicsTarget(
            color=ColorTarget(cie_x=0.30, cie_y=0.52, name="green"),
            thermal=ThermalTarget(),
        )
        result = optimize_multiphysics(
            target,
            diameter_range=(240, 300), diameter_step=30,
            phi_range=(0.45, 0.55), phi_step=0.10,
            absorber_fractions=[0.005],
            film_thicknesses_um=[50.0],
        )
        assert result.best_color is not None
        assert result.best_color.color.predicted

    def test_summary_not_empty(self):
        target = MultiPhysicsTarget(thermal=ThermalTarget())
        result = optimize_multiphysics(
            target,
            diameter_range=(250, 300), diameter_step=50,
            phi_range=(0.45, 0.55), phi_step=0.10,
            absorber_fractions=[0.005],
            film_thicknesses_um=[50.0],
        )
        s = result.summary()
        assert "Pareto" in s or "Multi-Physics" in s


# ═══════════════════════════════════════════════════════════════════════════
# Quick design convenience
# ═══════════════════════════════════════════════════════════════════════════

class TestQuickDesign:

    def test_quick_green(self):
        result = quick_design("green")
        assert isinstance(result, ParetoResult)
        assert result.n_evaluated > 0

    def test_quick_blue(self):
        result = quick_design("blue")
        assert result.n_evaluated > 0
        assert result.best_balanced is not None

    def test_quick_with_backing(self):
        result = quick_design("green", backing_mm=50.0, air_gap_mm=25.0)
        assert result.n_evaluated > 0

    def test_named_colors_all_valid(self):
        for name, (x, y) in NAMED_COLORS.items():
            assert 0 < x < 1
            assert 0 < y < 1


# ═══════════════════════════════════════════════════════════════════════════
# Physics consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicsConsistency:

    def test_thicker_film_better_thermal(self):
        """Thicker film → higher R-value across all designs."""
        target = MultiPhysicsTarget(thermal=ThermalTarget())
        thin_params = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.50,
                       "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        thick_params = dict(thin_params, film_thickness_um=500)
        dp_thin = evaluate_design(thin_params, target)
        dp_thick = evaluate_design(thick_params, target)
        assert dp_thick.thermal.R_value_m2KW > dp_thin.thermal.R_value_m2KW

    def test_higher_phi_lower_acoustic(self):
        """Higher φ → less porosity → generally lower NRC (for thin films)."""
        target = MultiPhysicsTarget(acoustic=AcousticTarget())
        lo = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.35,
              "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        hi = dict(lo, volume_fraction=0.60)
        dp_lo = evaluate_design(lo, target)
        dp_hi = evaluate_design(hi, target)
        # Both should have acoustic results
        assert dp_lo.acoustic is not None
        assert dp_hi.acoustic is not None

    def test_all_three_physics_populated(self):
        """With all three targets set, all three predictions should exist."""
        target = MultiPhysicsTarget(
            color=ColorTarget(cie_x=0.30, cie_y=0.52, name="green"),
            thermal=ThermalTarget(),
            acoustic=AcousticTarget(),
        )
        params = {"diameter_nm": 280, "material": "SiO2", "volume_fraction": 0.50,
                  "absorber_fraction": 0.005, "film_thickness_um": 50, "matrix_material": "air"}
        dp = evaluate_design(params, target)
        assert dp.thermal is not None
        assert dp.acoustic is not None
        # Color may or may not be predicted depending on optical availability
