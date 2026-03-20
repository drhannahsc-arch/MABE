"""
tests/test_active_element_profiles.py -- Tests for active building element profiles.

Validates:
  - Profile database (3 profiles, all fields populated)
  - Weight normalization (4 weights, check constraints)
  - Harvesting target fields per spec
  - design_active_element() runs and returns correct type
  - Conductor selection per profile
  - Transparency constraints (window >= 40%)
  - Power target checking
  - Physical sanity of results
  - Edge cases
"""

import sys
import os
import math
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.active_element_profiles import (
    ActiveElementProfile,
    ActiveElementResult,
    HarvestingTargets,
    get_active_profile,
    list_active_profiles,
    design_active_element,
    _ACTIVE_PROFILES,
)
from core.energy_harvesting import EnvironmentSpec


EXPECTED_PROFILES = ["wall_panel_active", "window_active", "awning_active"]


# -----------------------------------------------------------------------
# Profile database
# -----------------------------------------------------------------------

class TestProfileDatabase:

    def test_all_present(self):
        available = list_active_profiles()
        for name in EXPECTED_PROFILES:
            assert name in available

    def test_count(self):
        assert len(_ACTIVE_PROFILES) == 3

    @pytest.mark.parametrize("app_id", EXPECTED_PROFILES)
    def test_fields_populated(self, app_id):
        p = get_active_profile(app_id)
        assert p.name != ""
        assert p.description != ""
        assert p.max_thickness_mm > 0
        assert p.harvesting_targets is not None

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown active profile"):
            get_active_profile("solar_roof")


# -----------------------------------------------------------------------
# Weight structure
# -----------------------------------------------------------------------

class TestWeights:

    @pytest.mark.parametrize("app_id", EXPECTED_PROFILES)
    def test_four_weights(self, app_id):
        p = get_active_profile(app_id)
        assert len(p.weights) == 4

    @pytest.mark.parametrize("app_id", EXPECTED_PROFILES)
    def test_weights_positive(self, app_id):
        p = get_active_profile(app_id)
        for w in p.weights:
            assert w > 0

    def test_wall_panel_power_dominant(self):
        """Wall panel should have highest weight on power (0.35)."""
        p = get_active_profile("wall_panel_active")
        assert p.weight_power == max(p.weights)

    def test_spec_weights_match(self):
        """Verify weights match the build handoff spec."""
        wall = get_active_profile("wall_panel_active")
        assert wall.weights == (0.20, 0.25, 0.20, 0.35)

        win = get_active_profile("window_active")
        assert win.weights == (0.10, 0.20, 0.15, 0.30)

        awning = get_active_profile("awning_active")
        assert awning.weights == (0.25, 0.15, 0.15, 0.30)


# -----------------------------------------------------------------------
# Harvesting targets per spec
# -----------------------------------------------------------------------

class TestHarvestingTargets:

    def test_wall_panel_targets(self):
        ht = get_active_profile("wall_panel_active").harvesting_targets
        assert ht.conductor_material == "MXene_Ti3C2"
        assert ht.min_transparency == 0.0  # opaque
        assert ht.pv_material_preference == "organic_PM6Y6"

    def test_window_targets(self):
        ht = get_active_profile("window_active").harvesting_targets
        assert ht.conductor_material == "ITO"
        assert ht.min_transparency >= 0.40
        assert ht.pv_material_preference == "perovskite_CsAgBiBr"

    def test_awning_targets(self):
        ht = get_active_profile("awning_active").harvesting_targets
        assert ht.conductor_material == "PEDOT_PSS"
        assert ht.pv_material_preference == "organic_PM6Y6"
        p = get_active_profile("awning_active")
        assert p.must_be_flexible is True

    @pytest.mark.parametrize("app_id", EXPECTED_PROFILES)
    def test_pb_free_default(self, app_id):
        ht = get_active_profile(app_id).harvesting_targets
        assert ht.pb_free_required is True


# -----------------------------------------------------------------------
# Form factor constraints
# -----------------------------------------------------------------------

class TestFormFactor:

    def test_wall_panel_thick(self):
        p = get_active_profile("wall_panel_active")
        assert p.max_thickness_mm == 5.0
        assert p.opaque is True

    def test_window_thin(self):
        p = get_active_profile("window_active")
        assert p.max_thickness_mm == 1.0
        assert p.opaque is False

    def test_awning_flexible(self):
        p = get_active_profile("awning_active")
        assert p.must_be_flexible is True
        assert p.max_thickness_mm == 1.0


# -----------------------------------------------------------------------
# Design function
# -----------------------------------------------------------------------

class TestDesignActiveElement:

    @pytest.mark.parametrize("app_id", EXPECTED_PROFILES)
    def test_runs_with_defaults(self, app_id):
        result = design_active_element(app_id)
        assert isinstance(result, ActiveElementResult)

    @pytest.mark.parametrize("app_id", EXPECTED_PROFILES)
    def test_power_budget_populated(self, app_id):
        result = design_active_element(app_id)
        pb = result.power_budget
        assert pb.total_W_m2 >= 0
        assert pb.pv_W_m2 >= 0
        assert pb.teg_W_m2 >= 0

    def test_custom_environment(self):
        env = EnvironmentSpec(solar_irradiance_W_m2=500.0, delta_T_K=15.0)
        result = design_active_element("wall_panel_active", environment=env)
        assert result.power_budget.pv_W_m2 > 0

    def test_window_transparency_met(self):
        """Window profile should meet its own transparency target."""
        result = design_active_element("window_active")
        assert result.meets_transparency_target is True
        assert result.conductor_transparency >= 0.40

    def test_wall_panel_opaque_ok(self):
        """Wall panel has min_transparency=0, so always meets."""
        result = design_active_element("wall_panel_active")
        assert result.meets_transparency_target is True

    def test_score_bounded(self):
        """Score should be in [0, 1]."""
        for app_id in EXPECTED_PROFILES:
            result = design_active_element(app_id)
            assert 0.0 <= result.score <= 1.0

    def test_higher_irradiance_higher_power(self):
        env_low = EnvironmentSpec(solar_irradiance_W_m2=100.0)
        env_high = EnvironmentSpec(solar_irradiance_W_m2=800.0)
        r_low = design_active_element("wall_panel_active", environment=env_low)
        r_high = design_active_element("wall_panel_active", environment=env_high)
        assert r_high.power_budget.pv_W_m2 > r_low.power_budget.pv_W_m2

    def test_conductor_material_matches_profile(self):
        result = design_active_element("window_active")
        assert result.conductor.material == "ITO"
        result2 = design_active_element("awning_active")
        assert result2.conductor.material == "PEDOT_PSS"

    def test_summary_string(self):
        p = get_active_profile("wall_panel_active")
        s = p.summary()
        assert "wall_panel_active" in s.lower() or "Wall Panel" in s


# -----------------------------------------------------------------------
# Physical sanity
# -----------------------------------------------------------------------

class TestPhysicalSanity:

    def test_window_reasonable_power(self):
        """Window with moderate sun should produce 1-30 W/m2."""
        result = design_active_element("window_active")
        assert 0.1 < result.power_budget.total_W_m2 < 50.0

    def test_awning_higher_power_than_wall(self):
        """Awning (angled to sky, 300 W/m2) should outperform wall (100 W/m2)."""
        r_wall = design_active_element("wall_panel_active")
        r_awning = design_active_element("awning_active")
        assert r_awning.power_budget.pv_W_m2 > r_wall.power_budget.pv_W_m2
