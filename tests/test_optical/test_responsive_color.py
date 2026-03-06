"""
tests/test_optical/test_responsive_color.py — Stimulus-Responsive Color Tests
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")

from optical.responsive_color import (
    magnetic_response, thermal_response, photochromic_response,
    electric_response, assembly_comparison,
    StimulusResponse, ColorState, pnipam_length_at_T,
    _langevin, PNIPAM_SPACERS, PHOTOCHROMES,
)

_LAM = np.linspace(380, 780, 81)


class TestLangevin:
    def test_zero_field(self):
        assert abs(_langevin(0.0)) < 1e-6

    def test_large_field_saturates(self):
        assert abs(_langevin(100.0) - 1.0) < 0.02

    def test_monotonic(self):
        vals = [_langevin(x) for x in [0, 1, 5, 10, 50]]
        assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))


class TestPNIPAM:
    def test_below_lcst_swollen(self):
        L, n = pnipam_length_at_T("PNIPAM_25", 20.0)
        sp = PNIPAM_SPACERS["PNIPAM_25"]
        assert L > sp.L_collapsed_nm * 1.5  # much longer than collapsed

    def test_above_lcst_collapsed(self):
        L, n = pnipam_length_at_T("PNIPAM_25", 50.0)
        sp = PNIPAM_SPACERS["PNIPAM_25"]
        assert L < sp.L_swollen_nm * 0.6  # much shorter than swollen

    def test_transition_at_lcst(self):
        L_low, _ = pnipam_length_at_T("PNIPAM_25", 20.0)
        L_lcst, _ = pnipam_length_at_T("PNIPAM_25", 32.0)
        L_high, _ = pnipam_length_at_T("PNIPAM_25", 50.0)
        assert L_low > L_lcst > L_high

    def test_unknown_spacer_raises(self):
        with pytest.raises(ValueError):
            pnipam_length_at_T("PNIPAM_999", 25.0)


class TestMagneticResponse:
    def test_produces_states(self):
        r = magnetic_response(250, "polystyrene", "Fe2O3", 25,
                               B_fields_mT=[0, 100], wavelengths_nm=_LAM)
        assert isinstance(r, StimulusResponse)
        assert r.mechanism == "magnetic"
        assert len(r.states) == 2

    def test_field_changes_color(self):
        r = magnetic_response(250, "polystyrene", "Fe2O3", 25,
                               B_fields_mT=[0, 500], wavelengths_nm=_LAM)
        assert r.delta_E_range > 0

    def test_zero_field_is_random(self):
        r = magnetic_response(250, "polystyrene", "Fe2O3", 25,
                               B_fields_mT=[0], wavelengths_nm=_LAM)
        assert "f_cap=0.50" in r.states[0].label


class TestThermalResponse:
    def test_produces_states(self):
        r = thermal_response(200, "SiO2", "APTES", "PNIPAM_25", "SPAAC",
                              n_medium=1.33,
                              temperatures_C=[20, 40], wavelengths_nm=_LAM)
        assert r.mechanism == "thermal_PNIPAM"
        assert len(r.states) == 2

    def test_temperature_shifts_peak(self):
        r = thermal_response(200, "SiO2", "APTES", "PNIPAM_50", "SPAAC",
                              n_medium=1.33,
                              temperatures_C=[20, 50], wavelengths_nm=_LAM)
        assert r.delta_lambda_nm > 0

    def test_cold_redder_than_hot(self):
        """Below LCST (swollen, larger D_eff) → longer wavelength."""
        r = thermal_response(200, "SiO2", "APTES", "PNIPAM_50", "SPAAC",
                              n_medium=1.33,
                              temperatures_C=[20, 50], wavelengths_nm=_LAM)
        assert r.states[0].peak_nm >= r.states[1].peak_nm


class TestPhotochromicResponse:
    def test_azobenzene_produces_states(self):
        r = photochromic_response(200, "SiO2", "APTES", "azobenzene", "SPAAC",
                                   wavelengths_nm=_LAM)
        assert "azobenzene" in r.mechanism
        assert len(r.states) >= 2

    def test_spiropyran_produces_states(self):
        r = photochromic_response(200, "SiO2", "APTES", "spiropyran", "SPAAC",
                                   wavelengths_nm=_LAM)
        assert "spiropyran" in r.mechanism

    def test_unknown_photochrome_raises(self):
        with pytest.raises(ValueError):
            photochromic_response(200, "SiO2", "APTES", "nonexistent", "SPAAC")


class TestElectricResponse:
    def test_produces_states(self):
        r = electric_response(200, "SiO2", "TiO2_rutile", 30,
                               E_fields_Vmm=[0, 500], wavelengths_nm=_LAM)
        assert r.mechanism == "electric_DEP"
        assert len(r.states) == 2

    def test_field_changes_color(self):
        r = electric_response(200, "SiO2", "TiO2_rutile", 30,
                               E_fields_Vmm=[0, 1000], wavelengths_nm=_LAM)
        assert r.delta_E_range > 0

    def test_high_dielectric_contrast(self):
        """TiO2 cap (ε=86) vs SiO2 base (ε=3.9) — large Δε."""
        r = electric_response(200, "SiO2", "TiO2_rutile", 30,
                               E_fields_Vmm=[0, 500], wavelengths_nm=_LAM)
        assert "Δε=82" in r.notes


class TestAssemblyComparison:
    def test_produces_two_states(self):
        r = assembly_comparison(200, "SiO2", wavelengths_nm=_LAM)
        assert r.mechanism == "assembly_directed"
        assert len(r.states) == 2

    def test_opal_vs_glass_different(self):
        r = assembly_comparison(200, "SiO2", wavelengths_nm=_LAM)
        assert r.delta_E_range > 0
        labels = [s.label for s in r.states]
        assert "ordered_opal" in labels
        assert "disordered_glass" in labels

    def test_bragg_noted(self):
        r = assembly_comparison(200, "SiO2", wavelengths_nm=_LAM)
        assert "Bragg=" in r.notes


class TestColorStateStructure:
    def test_states_have_required_fields(self):
        r = thermal_response(200, "SiO2", "APTES", "PNIPAM_25", "SPAAC",
                              n_medium=1.33,
                              temperatures_C=[25], wavelengths_nm=_LAM)
        s = r.states[0]
        assert s.peak_nm > 0
        assert len(s.cie_xy) == 2
        assert len(s.Lab) == 3
        assert len(s.sRGB) == 3
        assert len(s.R_spectrum) == len(_LAM)
