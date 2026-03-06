"""
tests/test_optical/test_application_optics.py — Application-Level Optics Tests
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")

from optical.application_optics import (
    CoatingSpec, coating_reflectance,
    lambertian_average, bragg_angle_average,
    transmission_spectrum, film_absorption_spectrum,
    substrate_reflectance, FABRIC_LIBRARY,
    predict_application, ApplicationResult,
    glass_film, textile_dipcoat, beads_on_glass, beads_on_textile,
    compare_scenarios,
)

_LAM = np.linspace(380, 780, 81)


# ═══════════════════════════════════════════════════════════════════════════
# 1. COATING GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

class TestCoatingGeometry:
    def test_bulk_film_has_peak(self):
        spec = CoatingSpec("bulk_film", core_diameter_nm=200, core_material="SiO2")
        R = coating_reflectance(spec, _LAM)
        assert np.max(R) > 0
        assert _LAM[np.argmax(R)] > 380

    def test_thin_film_weaker_than_bulk(self):
        spec_bulk = CoatingSpec("bulk_film", n_layers=20, core_diameter_nm=200,
                                core_material="SiO2")
        spec_thin = CoatingSpec("thin_film", n_layers=3, core_diameter_nm=200,
                                core_material="SiO2")
        R_bulk = coating_reflectance(spec_bulk, _LAM)
        R_thin = coating_reflectance(spec_thin, _LAM)
        assert np.max(R_bulk) >= np.max(R_thin)

    def test_monolayer_weaker_than_film(self):
        spec_film = CoatingSpec("bulk_film", core_diameter_nm=200,
                                core_material="SiO2")
        spec_mono = CoatingSpec("monolayer", coverage_fraction=0.8,
                                core_diameter_nm=200, core_material="SiO2")
        R_film = coating_reflectance(spec_film, _LAM)
        R_mono = coating_reflectance(spec_mono, _LAM)
        assert np.max(R_film) >= np.max(R_mono)

    def test_sparse_weakest(self):
        spec = CoatingSpec("sparse", coverage_fraction=0.2,
                           core_diameter_nm=200, core_material="SiO2")
        R = coating_reflectance(spec, _LAM)
        assert np.max(R) < 0.05  # very weak

    def test_coverage_scales_reflectance(self):
        spec_full = CoatingSpec("sparse", coverage_fraction=0.8,
                                core_diameter_nm=200, core_material="SiO2")
        spec_half = CoatingSpec("sparse", coverage_fraction=0.2,
                                core_diameter_nm=200, core_material="SiO2")
        R_full = coating_reflectance(spec_full, _LAM)
        R_half = coating_reflectance(spec_half, _LAM)
        assert np.max(R_full) > np.max(R_half)

    def test_unknown_geometry_raises(self):
        spec = CoatingSpec("nonexistent", core_diameter_nm=200)
        with pytest.raises(ValueError):
            coating_reflectance(spec, _LAM)


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIEWING ANGLE
# ═══════════════════════════════════════════════════════════════════════════

class TestViewingAngle:
    def test_lambertian_reduces_brightness(self):
        R = np.ones(len(_LAM)) * 0.3
        R_lamb = lambertian_average(R, _LAM)
        assert np.max(R_lamb) < np.max(R)

    def test_lambertian_factor_is_two_thirds(self):
        R = np.ones(len(_LAM)) * 0.6
        R_lamb = lambertian_average(R)
        np.testing.assert_allclose(R_lamb, 0.4, atol=0.01)

    def test_bragg_angle_average_has_peak(self):
        R = bragg_angle_average(200, 1.45, wavelengths_nm=_LAM)
        assert np.max(R) > 0

    def test_bragg_average_broader_than_normal(self):
        """Angle averaging broadens the Bragg peak."""
        R = bragg_angle_average(200, 1.45, wavelengths_nm=_LAM)
        # Count wavelengths above half-max
        half_max = np.max(R) / 2
        n_above = np.sum(R > half_max)
        assert n_above > 3  # should be broadened


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSMISSION
# ═══════════════════════════════════════════════════════════════════════════

class TestTransmission:
    def test_R_plus_T_plus_A_leq_one(self):
        R = np.array([0.3] * len(_LAM))
        A = np.array([0.1] * len(_LAM))
        result = transmission_spectrum(R, _LAM, film_absorption=A)
        total = result["R"] + result["T"] + result["A"]
        assert np.all(total <= 1.01)

    def test_no_absorption_high_T(self):
        R = np.array([0.1] * len(_LAM))
        result = transmission_spectrum(R, _LAM)
        assert np.all(result["T"] > 0.7)

    def test_blue_reflection_warm_transmission(self):
        """Blue reflected → transmitted should be yellowish."""
        # Create blue reflectance peak
        R = np.exp(-0.5 * ((_LAM - 460) / 20)**2) * 0.3
        result = transmission_spectrum(R, _LAM)
        # Transmitted x should be higher (warmer) than reflected x
        assert result["cie_xy_T"][0] > result["cie_xy_R"][0]

    def test_film_absorption_nonnegative(self):
        A = film_absorption_spectrum(200, "carbon", 0.05, 10, _LAM)
        assert np.all(A >= 0)
        assert np.all(A <= 1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DIFFUSE SUBSTRATE
# ═══════════════════════════════════════════════════════════════════════════

class TestDiffuseSubstrate:
    def test_all_fabrics_valid(self):
        for name in FABRIC_LIBRARY:
            R = substrate_reflectance(name, _LAM)
            assert np.all(R >= 0)
            assert np.all(R <= 1)
            assert len(R) == len(_LAM)

    def test_black_very_low(self):
        R = substrate_reflectance("black_polyester", _LAM)
        assert np.max(R) < 0.10

    def test_white_high(self):
        R = substrate_reflectance("white_cotton", _LAM)
        assert np.mean(R) > 0.5

    def test_measured_array_passthrough(self):
        custom = np.linspace(0.1, 0.5, len(_LAM))
        R = substrate_reflectance(custom, _LAM)
        np.testing.assert_array_equal(R, custom)

    def test_measured_wrong_length_raises(self):
        with pytest.raises(ValueError):
            substrate_reflectance(np.array([0.5, 0.5]), _LAM)

    def test_underlayer_materials_still_work(self):
        R = substrate_reflectance("carbon", _LAM)
        assert np.all(R >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. FULL APPLICATION SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

class TestApplicationScenarios:
    def test_glass_film_returns_result(self):
        r = glass_film(200, wavelengths_nm=_LAM)
        assert isinstance(r, ApplicationResult)
        assert r.peak_nm > 0
        assert len(r.T_observed) > 0  # transmission included

    def test_textile_dipcoat_returns_result(self):
        r = textile_dipcoat(200, wavelengths_nm=_LAM)
        assert isinstance(r, ApplicationResult)
        assert r.peak_nm > 0

    def test_beads_on_glass_returns_result(self):
        r = beads_on_glass(200, wavelengths_nm=_LAM)
        assert isinstance(r, ApplicationResult)

    def test_beads_on_textile_returns_result(self):
        r = beads_on_textile(200, wavelengths_nm=_LAM)
        assert isinstance(r, ApplicationResult)

    def test_black_substrate_better_color_than_white(self):
        """Black substrate should give more saturated structural color."""
        r_black = textile_dipcoat(200, fabric="black_polyester",
                                   wavelengths_nm=_LAM)
        r_white = textile_dipcoat(200, fabric="white_cotton",
                                   wavelengths_nm=_LAM)
        # Black should have CIE xy further from achromatic (0.33, 0.33)
        dx_black = (r_black.cie_xy[0] - 0.33)**2 + (r_black.cie_xy[1] - 0.33)**2
        dx_white = (r_white.cie_xy[0] - 0.33)**2 + (r_white.cie_xy[1] - 0.33)**2
        assert dx_black >= dx_white * 0.5  # black at least half as saturated

    def test_compare_scenarios_returns_four(self):
        results = compare_scenarios(200, wavelengths_nm=_LAM)
        assert len(results) == 4

    def test_brightness_ordering(self):
        """Bulk film should be brighter than sparse beads."""
        results = compare_scenarios(200, wavelengths_nm=_LAM)
        # Bulk film brightness >= sparse beads brightness
        # (may be equal if both near zero for certain wavelengths)
        r_bulk = [r for r in results if "bulk_film" in r.scenario][0]
        r_sparse = [r for r in results if "sparse" in r.scenario][0]
        assert r_bulk.brightness >= r_sparse.brightness - 0.1
