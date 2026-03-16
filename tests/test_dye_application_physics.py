"""
test_dye_application_physics.py — Tests for S1-S4 dye application physics.
"""
import pytest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dye_application_physics import (
    # S1
    capillary_number, spreading_coefficient, lucas_washburn_penetration,
    drying_time_thin_film, film_uniformity_from_roughness,
    # S2
    bragg_angle_shift, iridescence_index, angle_dependent_spectrum,
    brdf_components, BRDFComponents,
    # S3
    polydispersity_broadening, broadened_spectrum,
    km_scattering_absorption, km_reflectance_from_KS,
    km_thickness_dependent_R,
    # S4
    uv_degradation_rate, photon_dose_to_deltaE,
    years_to_noticeable_fade, UV_SENSITIVITY,
    humidity_induced_shift, thermal_expansion_shift, THERMAL_EXPANSION,
    strain_induced_shift,
    subtractive_mix_km, additive_fluorescence_mix, gamut_check,
    _LAM,
)


# ═══════════════════════════════════════════════════════════════════════════
# S1: Wetting & Film Formation
# ═══════════════════════════════════════════════════════════════════════════

class TestWetting:

    def test_capillary_number_low_for_dipping(self):
        """Dip coating at 1 cm/s: Ca << 1 (surface tension dominates)."""
        Ca = capillary_number(0.01, 8.9e-4, 0.0728)
        assert Ca < 0.01

    def test_capillary_number_high_for_fast_coating(self):
        Ca_slow = capillary_number(0.001, 8.9e-4, 0.0728)
        Ca_fast = capillary_number(1.0, 8.9e-4, 0.0728)
        assert Ca_fast > Ca_slow

    def test_spreading_positive_on_high_energy(self):
        """Glass (γ=75) with water (γ=73): should spread (S > 0 for low γ_sl)."""
        S = spreading_coefficient(75.0, 30.0, 10.0)
        assert S > 0

    def test_spreading_negative_on_low_energy(self):
        """PDMS (γ=20) with water (γ=73): won't spread."""
        S = spreading_coefficient(20.0, 73.0, 50.0)
        assert S < 0

    def test_lucas_washburn_positive(self):
        L = lucas_washburn_penetration(20e-6, 30.0, 8.9e-4, 0.0728, 1.0)
        assert L > 0

    def test_lucas_washburn_deeper_with_time(self):
        L1 = lucas_washburn_penetration(20e-6, 30.0, 8.9e-4, 0.0728, 1.0)
        L10 = lucas_washburn_penetration(20e-6, 30.0, 8.9e-4, 0.0728, 10.0)
        assert L10 > L1

    def test_lucas_washburn_no_wetting(self):
        """Contact angle > 90° → no penetration."""
        L = lucas_washburn_penetration(20e-6, 110.0, 8.9e-4, 0.0728, 1.0)
        assert L == 0.0

    def test_drying_time_positive(self):
        t = drying_time_thin_film(100e-6)
        assert t > 0

    def test_drying_faster_at_low_humidity(self):
        t_dry = drying_time_thin_film(100e-6, RH=0.2)
        t_humid = drying_time_thin_film(100e-6, RH=0.8)
        assert t_dry < t_humid

    def test_uniformity_thick_film(self):
        """Thick film on smooth substrate: high uniformity."""
        u = film_uniformity_from_roughness(1000.0, 0.01)
        assert u > 0.99

    def test_uniformity_thin_on_rough(self):
        """Thin film on rough substrate: poor uniformity."""
        u = film_uniformity_from_roughness(50.0, 20.0)
        assert u < 0.1


# ═══════════════════════════════════════════════════════════════════════════
# S2: Angle-Dependent / BRDF
# ═══════════════════════════════════════════════════════════════════════════

class TestAngleDependence:

    def test_normal_incidence_unchanged(self):
        lam = bragg_angle_shift(530.0, 1.35, 0.0)
        assert lam == pytest.approx(530.0)

    def test_blue_shift_with_angle(self):
        """Bragg peak blue-shifts at angle."""
        lam_30 = bragg_angle_shift(530.0, 1.35, 30.0)
        assert lam_30 < 530.0

    def test_larger_angle_more_shift(self):
        lam_30 = bragg_angle_shift(530.0, 1.35, 30.0)
        lam_60 = bragg_angle_shift(530.0, 1.35, 60.0)
        assert lam_60 < lam_30

    def test_higher_n_less_shift(self):
        """Higher n_eff → less angle-dependent (more angle-independent)."""
        lam_low_n = bragg_angle_shift(530.0, 1.2, 45.0)
        lam_high_n = bragg_angle_shift(530.0, 2.0, 45.0)
        shift_low = 530.0 - lam_low_n
        shift_high = 530.0 - lam_high_n
        assert shift_high < shift_low

    def test_iridescence_index_bounded(self):
        I = iridescence_index(530.0, 1.35)
        assert 0.0 <= I <= 1.0

    def test_iridescence_index_zero_at_high_n(self):
        """Very high n_eff → essentially angle-independent."""
        I = iridescence_index(530.0, 10.0)
        assert I < 0.01

    def test_angle_spectrum_shifts(self):
        _, R_0 = angle_dependent_spectrum(530.0, 1.35, theta_deg=0)
        _, R_45 = angle_dependent_spectrum(530.0, 1.35, theta_deg=45)
        peak_0 = _LAM[np.argmax(R_0)]
        peak_45 = _LAM[np.argmax(R_45)]
        assert peak_45 < peak_0


class TestBRDF:

    def test_components_sum(self):
        b = brdf_components(0.3, 0.1, 1.5, 0.0)
        assert b.total > 0
        assert b.total <= 1.0

    def test_specular_increases_with_angle(self):
        b_0 = brdf_components(0.3, 0.1, 1.5, 0.0)
        b_60 = brdf_components(0.3, 0.1, 1.5, 60.0)
        assert b_60.specular > b_0.specular

    def test_structural_dominates_on_dark(self):
        """On dark substrate, structural component should dominate diffuse."""
        b = brdf_components(0.3, 0.02, 1.5, 0.0)
        assert b.structural > b.diffuse


# ═══════════════════════════════════════════════════════════════════════════
# S3: Spectral Broadening + Multiple Scattering
# ═══════════════════════════════════════════════════════════════════════════

class TestSpectralBroadening:

    def test_monodisperse_unchanged(self):
        """CV=0 → no broadening."""
        fwhm = polydispersity_broadening(530.0, 30.0, 0.0)
        assert fwhm == pytest.approx(30.0)

    def test_polydisperse_broader(self):
        fwhm_mono = polydispersity_broadening(530.0, 30.0, 0.0)
        fwhm_poly = polydispersity_broadening(530.0, 30.0, 0.10)
        assert fwhm_poly > fwhm_mono

    def test_high_cv_very_broad(self):
        fwhm = polydispersity_broadening(530.0, 30.0, 0.20)
        assert fwhm > 100  # very broad

    def test_broadened_spectrum_shape(self):
        lam, R = broadened_spectrum(530.0, 30.0, 0.05)
        assert len(R) == len(_LAM)
        assert np.max(R) > 0.1


class TestKubelkaMunkCoefficients:

    def test_km_white_zero_KS(self):
        """R=1 → K/S = 0."""
        KS, _ = km_scattering_absorption(1.0)
        assert KS == pytest.approx(0.0)

    def test_km_black_high_KS(self):
        """R near 0 → very high K/S."""
        KS, _ = km_scattering_absorption(0.01)
        assert KS > 10

    def test_km_inverse_consistent(self):
        """KM → K/S → inverse KM → should recover R."""
        R_orig = 0.35
        KS, _ = km_scattering_absorption(R_orig)
        R_recovered = km_reflectance_from_KS(KS)
        assert abs(R_recovered - R_orig) < 0.01

    def test_km_thickness_thin_shows_substrate(self):
        """Very thin layer → substrate shows through."""
        R = km_thickness_dependent_R(K=1.0, S=2.0, d=0.001, R_substrate=0.9)
        assert R > 0.8  # nearly substrate color

    def test_km_thickness_thick_hides_substrate(self):
        """Very thick layer → independent of substrate."""
        R_white = km_thickness_dependent_R(K=1.0, S=2.0, d=100.0, R_substrate=0.9)
        R_black = km_thickness_dependent_R(K=1.0, S=2.0, d=100.0, R_substrate=0.1)
        assert abs(R_white - R_black) < 0.05


# ═══════════════════════════════════════════════════════════════════════════
# S4: Durability / Environmental / Color Mixing
# ═══════════════════════════════════════════════════════════════════════════

class TestDurability:

    def test_arrhenius_positive(self):
        k = uv_degradation_rate(80.0, 300.0)
        assert k > 0

    def test_arrhenius_faster_at_high_T(self):
        k_cold = uv_degradation_rate(80.0, 280.0)
        k_hot = uv_degradation_rate(80.0, 340.0)
        assert k_hot > k_cold

    def test_photon_dose_zero_at_zero(self):
        assert photon_dose_to_deltaE(0.0) == 0.0

    def test_photon_dose_increases(self):
        dE1 = photon_dose_to_deltaE(10.0)
        dE2 = photon_dose_to_deltaE(100.0)
        assert dE2 > dE1

    def test_years_to_fade_cupc(self):
        """CuPc (industrial blue): should last >10 years outdoor."""
        yrs = years_to_noticeable_fade(UV_SENSITIVITY["CuPc"])
        assert yrs > 10

    def test_years_to_fade_fluorescein(self):
        """Fluorescein: poor stability, < 1 year outdoor."""
        yrs = years_to_noticeable_fade(UV_SENSITIVITY["fluorescein"])
        assert yrs < 1.0

    def test_uv_table_has_entries(self):
        assert len(UV_SENSITIVITY) >= 12


class TestEnvironmental:

    def test_humidity_shift_dry(self):
        """Below critical RH: no shift."""
        lam = humidity_induced_shift(1.30, 1.40, 530.0, 0.3)
        assert lam == pytest.approx(530.0)

    def test_humidity_shift_wet(self):
        """Above critical RH: redshift (n increases)."""
        lam = humidity_induced_shift(1.30, 1.40, 530.0, 0.95)
        assert lam > 530.0

    def test_thermal_expansion_ps(self):
        """PS at +50K: slight redshift."""
        lam = thermal_expansion_shift(530.0, THERMAL_EXPANSION["polystyrene"], 50.0)
        assert lam > 530.0

    def test_thermal_contraction(self):
        """Cooling: slight blueshift."""
        lam = thermal_expansion_shift(530.0, THERMAL_EXPANSION["polystyrene"], -50.0)
        assert lam < 530.0

    def test_thermal_table_has_entries(self):
        assert len(THERMAL_EXPANSION) >= 10

    def test_strain_blueshift(self):
        """Stretching (ν < 0.5): transverse compression → blueshift."""
        lam = strain_induced_shift(530.0, 0.10, 0.3)
        assert lam < 530.0

    def test_strain_zero_at_nu_half(self):
        """Incompressible (ν=0.5): no transverse strain → no shift."""
        lam = strain_induced_shift(530.0, 0.10, 0.5)
        assert lam == pytest.approx(530.0)


class TestColorMixing:

    def test_subtractive_single_unchanged(self):
        """Single colorant → same spectrum."""
        R = 0.3 * np.exp(-0.5 * ((_LAM - 530) / 25) ** 2) + 0.1
        R_mixed = subtractive_mix_km([R])
        assert np.allclose(R_mixed, R, atol=0.01)

    def test_subtractive_darker(self):
        """Mixing two colorants → darker (lower average R)."""
        R1 = 0.3 * np.exp(-0.5 * ((_LAM - 500) / 30) ** 2) + 0.1
        R2 = 0.3 * np.exp(-0.5 * ((_LAM - 600) / 30) ** 2) + 0.1
        R_mixed = subtractive_mix_km([R1, R2])
        assert np.mean(R_mixed) < np.mean(R1)

    def test_additive_fluorescence(self):
        """Additive mixing: two emission peaks → broader combined."""
        em1 = np.exp(-0.5 * ((_LAM - 500) / 15) ** 2)
        em2 = np.exp(-0.5 * ((_LAM - 600) / 15) ** 2)
        mixed = additive_fluorescence_mix([em1, em2])
        assert np.max(mixed) == pytest.approx(1.0)

    def test_gamut_check_returns_dict(self):
        result = gamut_check((50, -40, 20))
        assert "chroma" in result
        assert "in_sRGB_approx" in result
        assert result["chroma"] > 0

    def test_gamut_neutral(self):
        result = gamut_check((50, 0, 0))
        assert result["saturation"] == "near-neutral"

    def test_gamut_saturated(self):
        result = gamut_check((50, -80, 50))
        assert result["saturation"] == "highly saturated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
