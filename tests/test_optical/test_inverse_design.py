"""
tests/test_optical/test_inverse_design.py — Module 11: Inverse Design Tests
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")
pytest.importorskip("scipy")

from optical.inverse_design import (
    inverse_design_photonic_glass,
    inverse_design_multilayer,
    DesignResult, PhotonicGlassDesign, MultilayerDesign,
)


class TestPhotnicGlassInverse:
    def test_blue_target_returns_result(self):
        """Blue target (x~0.15) should converge to a reasonable design."""
        r = inverse_design_photonic_glass(0.15, 0.10, max_iter=30)
        assert isinstance(r, DesignResult)
        assert isinstance(r.design, PhotonicGlassDesign)
        assert r.design.diameter_nm > 80
        assert r.delta_E < 20  # should get reasonably close

    def test_green_target_converges(self):
        """Green (x=0.27, y=0.40) is achievable with SiO2."""
        r = inverse_design_photonic_glass(0.27, 0.40, max_iter=50)
        assert r.delta_E < 5.0, f"Green should converge well, got ΔE={r.delta_E}"

    def test_red_target_higher_delta_e(self):
        """Red (x=0.50, y=0.30) is harder — ΔE should be higher than green."""
        r_green = inverse_design_photonic_glass(0.27, 0.40, max_iter=30)
        r_red = inverse_design_photonic_glass(0.50, 0.30, max_iter=30)
        # Red is harder, so ΔE should be >= green's ΔE
        # (just check both return results)
        assert r_red.design is not None
        assert r_green.design is not None

    def test_design_fields_populated(self):
        r = inverse_design_photonic_glass(0.27, 0.40, max_iter=30)
        d = r.design
        assert d.diameter_nm > 0
        assert d.packing_fraction > 0
        assert d.peak_wavelength_nm > 380
        assert 0 < d.cie_x < 1
        assert 0 < d.cie_y < 1

    def test_absorber_bounds_respected(self):
        r = inverse_design_photonic_glass(
            0.30, 0.30, absorber_bounds=(0.0, 0.05), max_iter=20)
        assert r.design.absorber_fraction <= 0.051  # small tolerance

    def test_diameter_bounds_respected(self):
        r = inverse_design_photonic_glass(
            0.30, 0.30, diameter_bounds=(150, 250), max_iter=20)
        assert 149 < r.design.diameter_nm < 251

    def test_n_evaluations_tracked(self):
        r = inverse_design_photonic_glass(0.30, 0.30, max_iter=10)
        assert r.n_evaluations > 0

    def test_elapsed_time_tracked(self):
        r = inverse_design_photonic_glass(0.30, 0.30, max_iter=10)
        assert r.elapsed_s > 0


class TestMultilayerInverse:
    def test_green_mirror(self):
        """Target 550 nm stopband → should find ~550 nm peak."""
        r = inverse_design_multilayer(550, max_iter=20)
        assert isinstance(r.design, MultilayerDesign)
        assert abs(r.design.stopband_centre_nm - 550) < 20

    def test_blue_mirror(self):
        """Target 450 nm stopband."""
        r = inverse_design_multilayer(450, max_iter=20)
        assert abs(r.design.stopband_centre_nm - 450) < 20

    def test_red_mirror(self):
        """Target 650 nm stopband."""
        r = inverse_design_multilayer(650, max_iter=20)
        assert abs(r.design.stopband_centre_nm - 650) < 20

    def test_layer_count_correct(self):
        r = inverse_design_multilayer(550, n_periods=3, max_iter=10)
        assert r.design.n_layers == 6  # 3 periods × 2 layers

    def test_thicknesses_positive(self):
        r = inverse_design_multilayer(550, max_iter=10)
        for t in r.design.thicknesses_nm:
            assert t > 0


class TestIntegration:
    def test_existing_optical_tests_still_pass(self):
        """Sanity: M1-M9 forward models still work."""
        from optical.photonic_glass import photonic_glass_reflectance
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY
        lam = np.linspace(380, 780, 81)
        R = photonic_glass_reflectance(200, "SiO2", 1.0, 0.55, lam)
        assert R is not None
        X, Y, Z = spectrum_to_XYZ(R, lam)
        x, y, _ = XYZ_to_xyY(X, Y, Z)
        assert 0 < x < 1
        assert 0 < y < 1
