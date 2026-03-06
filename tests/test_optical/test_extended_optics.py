"""
tests/test_optical/test_extended_optics.py — Tests for M6b, base layer, Janus sphere
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")
pytest.importorskip("scipy")

from optical.photonic_glass import photonic_glass_reflectance

_LAM = np.linspace(380, 780, 81)


# ═══════════════════════════════════════════════════════════════════════════
# MULTIPLE SCATTERING (Module 6b)
# ═══════════════════════════════════════════════════════════════════════════

from optical.multiple_scattering import (
    transport_properties, ms_correction_factor,
    photonic_glass_reflectance_ms, peak_comparison,
)


class TestMultipleScattering:
    def test_transport_properties_returns_dict(self):
        tp = transport_properties(200, "SiO2", 1.0, 0.55, 500)
        assert "l_sca" in tp
        assert "l_star" in tp
        assert "g" in tp
        assert tp["l_sca"] > 0
        assert tp["l_star"] >= tp["l_sca"]  # l* >= l_sca always

    def test_correction_factor_near_unity_thin_film(self):
        f = ms_correction_factor(200, "SiO2", 1.0, 0.55, 500,
                                  film_thickness_layers=2)
        assert 1.0 <= f < 1.02  # very thin: almost no correction

    def test_correction_factor_increases_with_thickness(self):
        f5 = ms_correction_factor(200, "SiO2", 1.0, 0.55, 500,
                                   film_thickness_layers=5)
        f50 = ms_correction_factor(200, "SiO2", 1.0, 0.55, 500,
                                    film_thickness_layers=50)
        assert f50 >= f5

    def test_ms_reflectance_has_peak(self):
        R = photonic_glass_reflectance_ms(200, "SiO2", 1.0, 0.55, _LAM)
        assert np.max(R) > 0
        peak = _LAM[np.argmax(R)]
        assert 380 < peak < 780

    def test_ms_absolute_reflectance_bounded(self):
        R = photonic_glass_reflectance_ms(200, "SiO2", 1.0, 0.55, _LAM,
                                           film_thickness_layers=20,
                                           absolute=True)
        assert 0 < np.max(R) <= 0.5  # physically reasonable absolute R

    def test_ms_peak_redshift_vs_ss(self):
        """MS correction should redshift or maintain peak (never blueshift)."""
        r = peak_comparison(200, "polystyrene", 1.0, 0.55)
        assert r["shift_nm"] >= 0  # redshift or zero


# ═══════════════════════════════════════════════════════════════════════════
# BASE LAYER
# ═══════════════════════════════════════════════════════════════════════════

from optical.base_layer import (
    effective_substrate_reflectance,
    photonic_glass_on_base_layer,
    multilevel_stack_reflectance,
    BASE_LAYER_MATERIALS,
    base_layer_n,
)


class TestBaseLayer:
    def test_no_base_layer_matches_standard(self):
        """base_material='none' should give same result as standard coupling."""
        from optical.underlayer_coupling import photonic_glass_on_substrate
        R_film = photonic_glass_reflectance(200, "SiO2", 1.0, 0.55, _LAM)
        R_std = photonic_glass_on_substrate(R_film, _LAM, substrate="carbon")
        R_none = photonic_glass_on_base_layer(R_film, _LAM,
                                               base_material="none",
                                               substrate_material="carbon")
        np.testing.assert_allclose(R_none, R_std, atol=0.01)

    def test_pda_base_modifies_spectrum(self):
        """PDA base layer should change CIE coordinates vs no base."""
        R_film = photonic_glass_reflectance(200, "SiO2", 1.0, 0.55, _LAM)
        R_none = photonic_glass_on_base_layer(R_film, _LAM,
                                               base_material="none",
                                               substrate_material="carbon")
        R_pda = photonic_glass_on_base_layer(R_film, _LAM,
                                              base_material="polydopamine",
                                              base_thickness_nm=50,
                                              substrate_material="carbon")
        # Should be different spectra
        assert not np.allclose(R_none, R_pda, atol=0.001)

    def test_tio2_base_high_n_interference(self):
        """High-n TiO2 base layer should show stronger spectral modification."""
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY
        R_film = photonic_glass_reflectance(200, "SiO2", 1.0, 0.55, _LAM)
        R_none = photonic_glass_on_base_layer(R_film, _LAM,
                                               base_material="none",
                                               substrate_material="carbon")
        R_tio2 = photonic_glass_on_base_layer(R_film, _LAM,
                                               base_material="sol_gel_TiO2",
                                               base_thickness_nm=100,
                                               substrate_material="carbon")
        X1, Y1, Z1 = spectrum_to_XYZ(R_none, _LAM)
        X2, Y2, Z2 = spectrum_to_XYZ(R_tio2, _LAM)
        x1, y1, _ = XYZ_to_xyY(X1, Y1, Z1)
        x2, y2, _ = XYZ_to_xyY(X2, Y2, Z2)
        # CIE should differ
        assert abs(x1 - x2) > 0.01 or abs(y1 - y2) > 0.01

    def test_effective_substrate_reflectance_bounded(self):
        R = effective_substrate_reflectance("polydopamine", 50, "carbon", _LAM)
        assert np.all(R >= 0)
        assert np.all(R <= 1)

    def test_all_base_materials_valid(self):
        for mat in BASE_LAYER_MATERIALS:
            n = base_layer_n(mat, 550)
            assert abs(n) > 0

    def test_multilevel_stack(self):
        """Multilevel stack should produce valid reflectance."""
        R_film = photonic_glass_reflectance(200, "SiO2", 1.0, 0.55, _LAM)
        R = multilevel_stack_reflectance(R_film, _LAM,
            layers=[("polydopamine", 30), ("sol_gel_SiO2", 50)],
            substrate_material="carbon")
        assert np.all(R >= 0)
        assert np.all(R <= 1)
        assert np.max(R) > 0


# ═══════════════════════════════════════════════════════════════════════════
# JANUS SPHERE
# ═══════════════════════════════════════════════════════════════════════════

from optical.janus_sphere import (
    janus_Q_back, janus_photonic_glass_reflectance, two_way_color,
)


class TestJanusSphere:
    def test_symmetric_janus_random_is_average(self):
        """Random orientation Q_back should be mean of cap_up and cap_down."""
        Q_up = janus_Q_back(200, "SiO2", "Au", 20, 1.0, 500,
                             orientation="cap_up")
        Q_down = janus_Q_back(200, "SiO2", "Au", 20, 1.0, 500,
                               orientation="cap_down")
        Q_rand = janus_Q_back(200, "SiO2", "Au", 20, 1.0, 500,
                               orientation="random")
        expected = 0.5 * Q_up + 0.5 * Q_down
        assert abs(Q_rand - expected) < 1e-6

    def test_no_cap_matches_homogeneous(self):
        """Zero-thickness cap should give same Q_back for both orientations."""
        Q_up = janus_Q_back(200, "SiO2", "SiO2", 0.0, 1.0, 500,
                             orientation="cap_up")
        Q_down = janus_Q_back(200, "SiO2", "SiO2", 0.0, 1.0, 500,
                               orientation="cap_down")
        # Should be very similar (not identical due to 0-thickness shell)
        assert abs(Q_up - Q_down) / max(Q_up, Q_down, 1e-10) < 0.1

    def test_janus_pg_has_peak(self):
        R = janus_photonic_glass_reflectance(200, "SiO2", "Au", 20, 1.0,
                                              0.55, _LAM)
        assert np.max(R) > 0
        peak = _LAM[np.argmax(R)]
        assert 380 < peak < 780

    def test_two_way_color_produces_delta_e(self):
        r = two_way_color(200, "SiO2", "TiO2_rutile", 30,
                          wavelengths_nm=_LAM)
        assert "delta_E_two_way" in r
        assert r["delta_E_two_way"] > 0  # TiO2 cap creates measurable asymmetry

    def test_magnetic_janus_large_delta_e(self):
        """Fe2O3 cap on PS should give large two-way ΔE (magnetic switchable)."""
        r = two_way_color(250, "polystyrene", "Fe2O3", 25,
                          wavelengths_nm=_LAM)
        assert r["delta_E_two_way"] > 20  # should be very pronounced

    def test_two_way_contains_all_orientations(self):
        r = two_way_color(200, "SiO2", "Au", 20, wavelengths_nm=_LAM)
        for orient in ["cap_up", "cap_down", "random"]:
            assert orient in r
            assert "peak_nm" in r[orient]
            assert "cie_xy" in r[orient]
            assert "Lab" in r[orient]
            assert "sRGB" in r[orient]

    def test_cap_up_vs_cap_down_different_spectra(self):
        """Different orientations should give different spectra."""
        R_up = janus_photonic_glass_reflectance(200, "SiO2", "TiO2_rutile",
                                                 30, 1.0, 0.55, _LAM,
                                                 orientation="cap_up")
        R_down = janus_photonic_glass_reflectance(200, "SiO2", "TiO2_rutile",
                                                   30, 1.0, 0.55, _LAM,
                                                   orientation="cap_down")
        assert not np.allclose(R_up, R_down, atol=0.01)
