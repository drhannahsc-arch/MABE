"""
test_surface_optics.py — Tests for substrate library + optical coupling.
"""
import pytest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.surface_optics import (
    SUBSTRATE_LIBRARY, SubstrateEntry,
    substrate_reflectance_spectrum,
    kubelka_munk_coupling, estimate_dye_transmittance,
    fresnel_interface_reflectance, air_dye_interface_reflectance,
    predict_color_on_substrate, compare_substrates, ColorOnSubstrateResult,
    young_contact_angle, work_of_adhesion,
    owens_wendt_gamma_sl, predict_adhesion,
    _LAM,
)


class TestSubstrateLibrary:

    def test_count(self):
        assert len(SUBSTRATE_LIBRARY) >= 16

    def test_all_have_source(self):
        for name, s in SUBSTRATE_LIBRARY.items():
            assert len(s.source) > 10, f"{name}: missing source"

    def test_all_have_category(self):
        cats = set()
        for name, s in SUBSTRATE_LIBRARY.items():
            assert s.category in ("textile", "glass", "metal", "polymer",
                                   "paper", "ceramic"), f"{name}: bad category"
            cats.add(s.category)
        # Should cover multiple categories
        assert len(cats) >= 5

    def test_all_have_n_substrate(self):
        for name, s in SUBSTRATE_LIBRARY.items():
            assert s.n_substrate is not None and s.n_substrate > 1.0, \
                f"{name}: missing/invalid n"

    def test_metals_have_k(self):
        for name, s in SUBSTRATE_LIBRARY.items():
            if s.category == "metal":
                assert s.k_substrate is not None, f"{name}: metal needs k"

    def test_reflectance_bounded(self):
        for name, s in SUBSTRATE_LIBRARY.items():
            if s.reflectance_vis is not None:
                assert 0.0 <= s.reflectance_vis <= 1.0, f"{name}: R out of bounds"

    def test_black_substrates_low_R(self):
        for name in ["cotton_black", "polyester_black"]:
            s = SUBSTRATE_LIBRARY[name]
            assert s.reflectance_vis < 0.10

    def test_white_substrates_high_R(self):
        for name in ["cotton_white", "paper_white"]:
            s = SUBSTRATE_LIBRARY[name]
            assert s.reflectance_vis > 0.70

    def test_click_compatible_have_groups(self):
        for name, s in SUBSTRATE_LIBRARY.items():
            if s.click_compatible:
                assert len(s.functional_groups) > 0, f"{name}: click but no groups"


class TestSubstrateSpectrum:

    def test_returns_spectrum(self):
        lam, R = substrate_reflectance_spectrum("cotton_white")
        assert len(R) == len(_LAM)

    def test_white_high_R(self):
        _, R = substrate_reflectance_spectrum("cotton_white")
        assert np.mean(R) > 0.7

    def test_black_low_R(self):
        _, R = substrate_reflectance_spectrum("cotton_black")
        assert np.mean(R) < 0.15

    def test_unknown_returns_default(self):
        _, R = substrate_reflectance_spectrum("UNKNOWN_SUBSTRATE")
        assert len(R) == len(_LAM)


class TestKubelkaMunk:

    def test_opaque_dye_hides_substrate(self):
        """Highly reflective dye should dominate over substrate."""
        R_dye = np.full(81, 0.8)
        T_dye = np.full(81, 0.0)  # opaque
        R_sub = np.full(81, 0.1)  # dark substrate
        R_total = kubelka_munk_coupling(R_dye, T_dye, R_sub)
        assert np.allclose(R_total, 0.8, atol=0.01)

    def test_transparent_dye_shows_substrate(self):
        """Transparent dye: substrate dominates."""
        R_dye = np.full(81, 0.0)
        T_dye = np.full(81, 1.0)
        R_sub = np.full(81, 0.9)
        R_total = kubelka_munk_coupling(R_dye, T_dye, R_sub)
        assert np.mean(R_total) > 0.8

    def test_black_substrate_preserves_dye(self):
        """Black substrate (R=0): only dye color shows."""
        R_dye = np.full(81, 0.3)
        T_dye = np.full(81, 0.7)
        R_sub = np.full(81, 0.0)
        R_total = kubelka_munk_coupling(R_dye, T_dye, R_sub)
        assert np.allclose(R_total, R_dye, atol=0.01)

    def test_white_substrate_washes_out(self):
        """White substrate (R=1): dye color diluted by substrate bleed-through."""
        R_dye = 0.25 * np.exp(-0.5 * ((_LAM - 530) / 25) ** 2) + 0.05
        T_dye = estimate_dye_transmittance(R_dye)
        R_black = np.full(81, 0.0)
        R_white = np.full(81, 0.9)
        R_on_black = kubelka_munk_coupling(R_dye, T_dye, R_black)
        R_on_white = kubelka_munk_coupling(R_dye, T_dye, R_white)
        # On white: higher overall reflectance (washed out)
        assert np.mean(R_on_white) > np.mean(R_on_black)

    def test_output_bounded(self):
        R_dye = np.random.rand(81) * 0.5
        T_dye = 1.0 - R_dye
        R_sub = np.random.rand(81)
        R_total = kubelka_munk_coupling(R_dye, T_dye, R_sub)
        assert np.all(R_total >= 0)
        assert np.all(R_total <= 1)


class TestFresnelInterface:

    def test_same_n_zero_reflection(self):
        assert fresnel_interface_reflectance(1.5, 1.5) == pytest.approx(0.0)

    def test_air_glass_about_4_percent(self):
        R = air_dye_interface_reflectance(1.5)
        assert 0.03 < R < 0.05

    def test_higher_contrast_more_reflection(self):
        R_low = fresnel_interface_reflectance(1.0, 1.3)
        R_high = fresnel_interface_reflectance(1.0, 2.5)
        assert R_high > R_low


class TestPredictColorOnSubstrate:

    def test_returns_result(self):
        R_dye = 0.25 * np.exp(-0.5 * ((_LAM - 530) / 25) ** 2) + 0.05
        r = predict_color_on_substrate(R_dye, "cotton_black")
        assert isinstance(r, ColorOnSubstrateResult)

    def test_black_substrate_closer_to_target(self):
        """Dye on black substrate should match target better than on white."""
        R_dye = 0.25 * np.exp(-0.5 * ((_LAM - 530) / 25) ** 2) + 0.05
        target = (55.0, -40.0, 20.0)
        r_black = predict_color_on_substrate(R_dye, "cotton_black", target_Lab=target)
        r_white = predict_color_on_substrate(R_dye, "cotton_white", target_Lab=target)
        if r_black.delta_E_from_target is not None and r_white.delta_E_from_target is not None:
            assert r_black.delta_E_from_target < r_white.delta_E_from_target

    def test_substrate_shifts_color(self):
        """White substrate should shift perceived color more than black."""
        R_dye = 0.25 * np.exp(-0.5 * ((_LAM - 530) / 25) ** 2) + 0.05
        r_black = predict_color_on_substrate(R_dye, "cotton_black")
        r_white = predict_color_on_substrate(R_dye, "cotton_white")
        if r_black.delta_E_from_alone is not None and r_white.delta_E_from_alone is not None:
            assert r_white.delta_E_from_alone > r_black.delta_E_from_alone

    def test_compare_substrates_returns_list(self):
        R_dye = 0.25 * np.exp(-0.5 * ((_LAM - 530) / 25) ** 2) + 0.05
        results = compare_substrates(R_dye, target_Lab=(55, -40, 20))
        assert len(results) >= 10
        # Should be sorted by ΔE from target
        if all(r.delta_E_from_target is not None for r in results):
            for i in range(len(results) - 1):
                assert results[i].delta_E_from_target <= results[i + 1].delta_E_from_target


class TestSurfaceAdhesion:

    def test_young_contact_angle(self):
        """Water on low-energy surface (PDMS): θ > 90°."""
        theta = young_contact_angle(20.0, 72.8, 90.0)
        assert theta > 80

    def test_young_high_energy_surface(self):
        """Water on glass: cos(θ) = (γ_s - γ_sl)/γ_l.
        γ_s=75, γ_l=72.8, γ_sl=5 → cos(θ)=0.96 → θ≈16°."""
        theta = young_contact_angle(75.0, 72.8, 5.0)
        assert theta < 30

    def test_work_of_adhesion_positive(self):
        W = work_of_adhesion(50.0, 35.0, 10.0)
        assert W > 0

    def test_owens_wendt_symmetric(self):
        """Same material on both sides → γ_sl = 0."""
        gamma_sl = owens_wendt_gamma_sl(30.0, 5.0, 30.0, 5.0)
        assert abs(gamma_sl) < 0.1

    def test_predict_adhesion_returns_dict(self):
        result = predict_adhesion("cotton_white")
        assert "W_a_mJ_m2" in result
        assert result["W_a_mJ_m2"] > 0

    def test_glass_better_adhesion_than_pdms(self):
        """Glass (high γ_s) should have better adhesion than PDMS (low γ_s)."""
        r_glass = predict_adhesion("soda_lime_glass")
        r_pdms = predict_adhesion("PDMS")
        if r_glass.get("W_a_mJ_m2") and r_pdms.get("W_a_mJ_m2"):
            assert r_glass["W_a_mJ_m2"] > r_pdms["W_a_mJ_m2"]

    def test_click_compatible_noted(self):
        r = predict_adhesion("cotton_white")
        assert r["click_compatible"] is True
        r2 = predict_adhesion("PDMS")
        assert r2["click_compatible"] is False

    def test_unknown_substrate(self):
        r = predict_adhesion("FAKE_SUBSTRATE")
        assert "error" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
