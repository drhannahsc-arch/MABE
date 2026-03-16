"""
test_molecular_photonics.py — Tests for 6 light-redirection mechanisms.
"""
import pytest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.molecular_photonics import (
    # L1
    FLUOROPHORE_LIBRARY, stokes_shift_energy, lippert_mataga_shift,
    strickler_berg_lifetime, emission_spectrum,
    # L2
    rayleigh_cross_section, rayleigh_wavelength_dependence,
    raman_shifted_wavelength, RAMAN_SHIFTS_CM1,
    # L3
    drude_specific_rotation, cd_spectrum_gaussian, SPECIFIC_ROTATIONS,
    # L4
    photonic_bandgap_1D, photonic_dos_1D, quarter_wave_design,
    bandgap_reflectance_spectrum,
    # L5
    drude_dielectric, lspr_wavelength, lspr_shift_per_RIU,
    plasmon_absorption_spectrum, METAL_DRUDE_PARAMS,
    # L6
    shg_wavelength, millers_rule_chi2, kerr_coefficient,
    two_photon_absorption_wavelength, CHI2_VALUES_PM_V, KERR_N2_CM2_W,
    _LAM,
)


# ═══════════════════════════════════════════════════════════════════════════
# L1: Fluorescence
# ═══════════════════════════════════════════════════════════════════════════

class TestFluorescenceData:

    def test_library_count(self):
        assert len(FLUOROPHORE_LIBRARY) >= 12

    def test_all_have_source(self):
        for name, f in FLUOROPHORE_LIBRARY.items():
            assert len(f.source) > 10, f"{name}: missing source"

    def test_stokes_shift_positive(self):
        for name, f in FLUOROPHORE_LIBRARY.items():
            assert f.stokes_shift_nm > 0, f"{name}: Stokes shift must be positive"
            assert f.em_max_nm > f.abs_max_nm, f"{name}: emission must be redder"

    def test_quantum_yield_bounded(self):
        for name, f in FLUOROPHORE_LIBRARY.items():
            assert 0 < f.quantum_yield <= 1.0, f"{name}: QY out of range"

    def test_fluorescein_high_qy(self):
        assert FLUOROPHORE_LIBRARY["fluorescein"].quantum_yield > 0.9


class TestFluorescencePhysics:

    def test_stokes_energy_positive(self):
        dE = stokes_shift_energy(490, 514)
        assert dE > 0

    def test_lippert_mataga_positive_in_polar(self):
        shift = lippert_mataga_shift(10.0, 4.0, 0.3)  # water-like
        assert shift > 0

    def test_lippert_zero_in_nonpolar(self):
        shift = lippert_mataga_shift(10.0, 4.0, 0.0)
        assert shift == 0.0

    def test_strickler_berg_returns_ns(self):
        tau = strickler_berg_lifetime(76000, 3000, 1.33, 514)
        assert 0.1 < tau < 100  # nanoseconds range for organic

    def test_emission_spectrum_shape(self):
        lam, em = emission_spectrum(490, 24, 40.0, 0.93)
        assert len(em) == len(_LAM)
        peak_idx = np.argmax(em)
        assert abs(_LAM[peak_idx] - 514) < 10


# ═══════════════════════════════════════════════════════════════════════════
# L2: Rayleigh / Raman
# ═══════════════════════════════════════════════════════════════════════════

class TestRayleigh:

    def test_cross_section_positive(self):
        sigma = rayleigh_cross_section(50.0, 1.46, 1.0, 550.0)
        assert sigma > 0

    def test_lambda_minus_4(self):
        """Blue light scatters more than red."""
        I_blue = rayleigh_wavelength_dependence(450.0)
        I_red = rayleigh_wavelength_dependence(650.0)
        assert I_blue > I_red

    def test_lambda_minus_4_ratio(self):
        """450/650 ratio should be (650/450)⁴ ≈ 4.34."""
        ratio = rayleigh_wavelength_dependence(450.0) / rayleigh_wavelength_dependence(650.0)
        expected = (650.0 / 450.0) ** 4
        assert abs(ratio - expected) < 0.01


class TestRaman:

    def test_stokes_redder(self):
        """Stokes Raman: scattered light is redder (longer λ)."""
        lam_scat = raman_shifted_wavelength(532.0, 1000.0)
        assert lam_scat > 532.0

    def test_larger_shift_redder(self):
        lam_small = raman_shifted_wavelength(532.0, 500.0)
        lam_large = raman_shifted_wavelength(532.0, 3000.0)
        assert lam_large > lam_small

    def test_raman_table_has_common(self):
        assert "C-H_stretch" in RAMAN_SHIFTS_CM1
        assert "C=O_stretch" in RAMAN_SHIFTS_CM1
        assert "O-H_stretch" in RAMAN_SHIFTS_CM1


# ═══════════════════════════════════════════════════════════════════════════
# L3: Optical Rotation / CD
# ═══════════════════════════════════════════════════════════════════════════

class TestOpticalRotation:

    def test_drude_nonzero(self):
        rot = drude_specific_rotation([(400.0, 100.0)], 589.0)
        assert rot != 0.0

    def test_opposite_enantiomers(self):
        """Opposite signs for enantiomers."""
        assert SPECIFIC_ROTATIONS["D-glucose"] > 0
        assert SPECIFIC_ROTATIONS["L-glucose"] < 0
        assert abs(SPECIFIC_ROTATIONS["D-glucose"]) == abs(SPECIFIC_ROTATIONS["L-glucose"])

    def test_cd_spectrum_shape(self):
        lam, cd = cd_spectrum_gaussian([(500.0, 10.0, 30.0)])
        assert len(cd) == len(_LAM)
        peak_idx = np.argmax(np.abs(cd))
        assert abs(_LAM[peak_idx] - 500) < 10


# ═══════════════════════════════════════════════════════════════════════════
# L4: Photonic Bandgap
# ═══════════════════════════════════════════════════════════════════════════

class TestPhotonicBandgap:

    def test_gap_center(self):
        gap = photonic_bandgap_1D(2.5, 1.46, 50.0, 90.0)
        assert 400 < gap["center_nm"] < 700

    def test_higher_contrast_wider_gap(self):
        gap_low = photonic_bandgap_1D(1.6, 1.46, 50.0, 90.0)
        gap_high = photonic_bandgap_1D(2.5, 1.46, 50.0, 90.0)
        assert gap_high["gap_width_nm"] > gap_low["gap_width_nm"]

    def test_zero_contrast_no_gap(self):
        gap = photonic_bandgap_1D(1.5, 1.5, 50.0, 50.0)
        assert gap["gap_width_nm"] == pytest.approx(0.0, abs=0.01)

    def test_dos_zero_inside_gap(self):
        gap = photonic_bandgap_1D(2.5, 1.46, 50.0, 90.0)
        dos = photonic_dos_1D(gap["center_nm"], gap["center_nm"], gap["gap_width_nm"])
        assert dos == 0.0

    def test_dos_enhanced_at_edge(self):
        gap = photonic_bandgap_1D(2.5, 1.46, 50.0, 90.0)
        edge = gap["center_nm"] + gap["gap_width_nm"] / 2.0 + 5.0
        dos = photonic_dos_1D(edge, gap["center_nm"], gap["gap_width_nm"])
        assert dos > 1.0  # enhanced

    def test_quarter_wave_design(self):
        d_h, d_l = quarter_wave_design(530.0, 2.5, 1.46)
        assert d_h == pytest.approx(530.0 / (4 * 2.5))
        assert d_l == pytest.approx(530.0 / (4 * 1.46))

    def test_bandgap_reflectance_has_peak(self):
        lam, R = bandgap_reflectance_spectrum(2.5, 1.46, 5, 530.0)
        assert np.max(R) > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# L5: Plasmon Resonance
# ═══════════════════════════════════════════════════════════════════════════

class TestPlasmonResonance:

    def test_drude_dielectric_complex(self):
        eps = drude_dielectric(500.0, 9.0, 0.07, 9.5)
        assert isinstance(eps, complex)

    def test_drude_negative_real_at_lspr(self):
        """Metal ε₁ should be negative in visible (below plasma freq)."""
        eps = drude_dielectric(500.0, 9.0, 0.07, 9.5)
        assert eps.real < 0

    def test_au_lspr_in_visible(self):
        lam = lspr_wavelength(1.33, "Au")
        assert 400 < lam < 600

    def test_ag_lspr_bluer_than_au(self):
        lam_au = lspr_wavelength(1.33, "Au")
        lam_ag = lspr_wavelength(1.33, "Ag")
        assert lam_ag < lam_au

    def test_lspr_redshifts_with_n(self):
        lam_air = lspr_wavelength(1.0, "Au")
        lam_water = lspr_wavelength(1.33, "Au")
        assert lam_water > lam_air

    def test_sensitivity_positive(self):
        sens = lspr_shift_per_RIU("Au")
        assert sens > 50  # Au spheres: ~100 nm/RIU

    def test_plasmon_spectrum_has_peak(self):
        lam, C = plasmon_absorption_spectrum("Au", 20.0, 1.33)
        assert np.max(C) == pytest.approx(1.0)
        peak_nm = _LAM[np.argmax(C)]
        assert 400 < peak_nm < 600

    def test_metal_params_complete(self):
        for metal in ["Au", "Ag", "Cu", "Al"]:
            assert metal in METAL_DRUDE_PARAMS


# ═══════════════════════════════════════════════════════════════════════════
# L6: Nonlinear Optics
# ═══════════════════════════════════════════════════════════════════════════

class TestNonlinearOptics:

    def test_shg_halves_wavelength(self):
        assert shg_wavelength(1064.0) == pytest.approx(532.0)

    def test_millers_rule_higher_n_higher_chi2(self):
        chi_low = millers_rule_chi2(1.5, 1.5)
        chi_high = millers_rule_chi2(2.5, 2.5)
        assert chi_high > chi_low

    def test_chi2_table_has_key_crystals(self):
        for crystal in ["LiNbO3", "KTP", "BBO"]:
            assert crystal in CHI2_VALUES_PM_V

    def test_kerr_conversion(self):
        n2_SI = kerr_coefficient(1.5, 3e-16)
        assert n2_SI == pytest.approx(3e-20)

    def test_kerr_table_has_common(self):
        assert "fused_silica" in KERR_N2_CM2_W
        assert "water" in KERR_N2_CM2_W

    def test_two_photon_threshold(self):
        """Si bandgap 1.12 eV: TPA threshold ≈ 2213 nm."""
        lam = two_photon_absorption_wavelength(1.12)
        assert abs(lam - 2214) < 5

    def test_tpa_wider_gap_shorter_threshold(self):
        lam_si = two_photon_absorption_wavelength(1.12)
        lam_gaas = two_photon_absorption_wavelength(1.42)
        assert lam_gaas < lam_si


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
