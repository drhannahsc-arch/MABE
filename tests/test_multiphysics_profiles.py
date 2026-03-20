"""
tests/test_multiphysics_profiles.py — Tests for application profiles.

Validates:
  - All 6 profiles build correctly
  - Each profile produces valid MultiPhysicsTarget
  - Color override works (named + explicit xy)
  - Optimizer runs for each profile and produces results
  - Weights differ by application (building = thermal-heavy, textile = color-heavy)
  - Form factor constraints propagated (thickness, flexibility, mass)
  - Backing structure populated for panel profiles
  - design_for_application() convenience entry point
  - compare_applications() cross-application comparison
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.multiphysics_profiles import (
    list_applications,
    get_profile,
    design_for_application,
    compare_applications,
    ApplicationProfile,
    ApplicationComparison,
    _PROFILES,
)
from core.multiphysics_pareto import (
    ParetoResult,
    MultiPhysicsTarget,
    NAMED_COLORS,
    _OPTICAL_AVAILABLE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Profile registry
# ═══════════════════════════════════════════════════════════════════════════

class TestProfileRegistry:

    def test_six_applications_available(self):
        apps = list_applications()
        assert len(apps) == 6

    def test_all_expected_ids(self):
        apps = list_applications()
        expected = {"facade_panel", "roof_coating", "wall_tile",
                    "textile_coating", "protective_garment", "smart_textile"}
        assert set(apps) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown application"):
            get_profile("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════
# Profile construction
# ═══════════════════════════════════════════════════════════════════════════

class TestProfileConstruction:

    @pytest.mark.parametrize("app_id", list(_PROFILES.keys()))
    def test_profile_builds(self, app_id):
        profile = get_profile(app_id)
        assert isinstance(profile, ApplicationProfile)
        assert profile.application_id == app_id
        assert profile.name != ""

    @pytest.mark.parametrize("app_id", list(_PROFILES.keys()))
    def test_profile_has_category(self, app_id):
        profile = get_profile(app_id)
        assert profile.category in ("building", "wearable")

    @pytest.mark.parametrize("app_id", list(_PROFILES.keys()))
    def test_profile_converts_to_target(self, app_id):
        profile = get_profile(app_id)
        target = profile.to_target()
        assert isinstance(target, MultiPhysicsTarget)
        assert target.application == app_id

    @pytest.mark.parametrize("app_id", list(_PROFILES.keys()))
    def test_profile_summary(self, app_id):
        profile = get_profile(app_id)
        s = profile.summary()
        assert profile.name in s
        assert len(s) > 50

    def test_facade_has_backing(self):
        p = get_profile("facade_panel")
        assert p.backing_thickness_mm > 0
        assert p.air_gap_mm > 0

    def test_textile_is_flexible(self):
        p = get_profile("textile_coating")
        assert p.must_be_flexible
        assert p.washable
        assert p.max_thickness_mm <= 1.0

    def test_roof_thermal_dominant_weight(self):
        p = get_profile("roof_coating")
        assert p.weights[1] > p.weights[0]  # thermal > color

    def test_textile_color_dominant_weight(self):
        p = get_profile("textile_coating")
        assert p.weights[0] > p.weights[1]  # color > thermal

    def test_wall_tile_acoustic_weight_high(self):
        p = get_profile("wall_tile")
        assert p.weights[2] >= 0.30  # acoustic significant

    def test_smart_textile_has_pnipam(self):
        p = get_profile("smart_textile")
        assert "PNIPAM_swollen" in p.matrix_materials
        assert "PNIPAM_collapsed" in p.matrix_materials

    def test_protective_has_thermal_constraint(self):
        p = get_profile("protective_garment")
        assert p.thermal_target is not None
        assert p.thermal_target.max_kappa_W_mK is not None

    def test_facade_has_fire_rating(self):
        p = get_profile("facade_panel")
        assert p.fire_rating in ("A1", "A2")

    def test_roof_water_resistant(self):
        p = get_profile("roof_coating")
        assert p.water_resistant


# ═══════════════════════════════════════════════════════════════════════════
# Color override
# ═══════════════════════════════════════════════════════════════════════════

class TestColorOverride:

    def test_named_color_override(self):
        p = get_profile("wall_tile", color_name="red")
        assert p.color_target.name == "red"
        assert p.color_target.cie_x == pytest.approx(0.64, abs=0.01)

    def test_xy_override(self):
        p = get_profile("facade_panel", color_xy=(0.35, 0.40))
        assert p.color_target.cie_x == pytest.approx(0.35, abs=0.01)
        assert p.color_target.cie_y == pytest.approx(0.40, abs=0.01)

    def test_xy_overrides_name(self):
        p = get_profile("facade_panel", color_name="blue", color_xy=(0.50, 0.30))
        assert p.color_target.cie_x == pytest.approx(0.50, abs=0.01)

    def test_no_override_uses_default(self):
        p = get_profile("wall_tile")
        assert p.color_target is not None
        # Wall tile default is green
        assert p.color_target.cie_y > 0.4


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer integration
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizerIntegration:

    def test_wall_tile_produces_result(self):
        result = design_for_application("wall_tile", color="green")
        assert isinstance(result, ParetoResult)
        assert result.n_evaluated > 0
        assert result.best_balanced is not None

    def test_textile_produces_result(self):
        result = design_for_application("textile_coating", color="blue")
        assert isinstance(result, ParetoResult)
        assert result.n_evaluated > 0

    def test_facade_has_pareto_front(self):
        result = design_for_application("facade_panel", color="white")
        assert len(result.pareto_front) >= 1

    def test_roof_produces_result(self):
        result = design_for_application("roof_coating")
        assert result.n_evaluated > 0

    def test_smart_textile_evaluates_both_states(self):
        """Smart textile should evaluate PNIPAM swollen and collapsed."""
        result = design_for_application("smart_textile", color="green")
        # Should have designs with both matrix materials
        matrices = {d.matrix_material for d in result.all_designs}
        assert "PNIPAM_swollen" in matrices
        assert "PNIPAM_collapsed" in matrices

    def test_protective_produces_result(self):
        result = design_for_application("protective_garment", color="orange")
        assert result.n_evaluated > 0

    def test_result_summary(self):
        result = design_for_application("wall_tile", color="green")
        s = result.summary()
        assert "Pareto" in s or "Multi-Physics" in s


# ═══════════════════════════════════════════════════════════════════════════
# Cross-application comparison
# ═══════════════════════════════════════════════════════════════════════════

class TestComparison:

    def test_compare_two_applications(self):
        comp = compare_applications("green", ["wall_tile", "textile_coating"])
        assert isinstance(comp, ApplicationComparison)
        assert "wall_tile" in comp.results
        assert "textile_coating" in comp.results

    def test_comparison_summary(self):
        comp = compare_applications("blue", ["wall_tile", "textile_coating"])
        s = comp.summary()
        assert "blue" in s.lower() or "Blue" in s
        assert "Wall" in s or "wall" in s
        assert "Textile" in s or "textile" in s

    def test_same_color_different_designs(self):
        """Same color target → different optimal D/φ/t per application."""
        comp = compare_applications("green", ["facade_panel", "textile_coating"])
        facade_best = comp.results["facade_panel"].best_balanced
        textile_best = comp.results["textile_coating"].best_balanced
        # Facade should have thicker film than textile
        if facade_best and textile_best:
            assert facade_best.film_thickness_um >= textile_best.film_thickness_um


# ═══════════════════════════════════════════════════════════════════════════
# Material constraints
# ═══════════════════════════════════════════════════════════════════════════

class TestMaterialConstraints:

    def test_facade_uses_silicone_matrix(self):
        p = get_profile("facade_panel")
        assert "silicone" in p.matrix_materials

    def test_textile_uses_polyurethane(self):
        p = get_profile("textile_coating")
        assert "polyurethane" in p.matrix_materials

    def test_wall_tile_air_matrix(self):
        p = get_profile("wall_tile")
        assert "air" in p.matrix_materials

    def test_facade_includes_tio2(self):
        p = get_profile("facade_panel")
        assert "TiO2_rutile" in p.materials

    def test_roof_large_diameter_range(self):
        """Cool roof: large particles for broadband reflectance."""
        p = get_profile("roof_coating")
        assert p.diameter_range[0] >= 280
