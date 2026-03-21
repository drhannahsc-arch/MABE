"""
tests/test_switchable_window.py -- Tests for switchable opacity window design.
"""

import sys
import os
import math
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.switchable_window import (
    design_conductive, ConductiveLayer, TCO_MATERIALS,
    design_electrochromic, ElectrochromicLayer,
    design_photochromic, PhotochromicLayer,
    design_magnetochromic, MagnetochromicLayer, _langevin,
    design_switchable_window, SwitchableTargets, SwitchableWindowDesign,
)


# -----------------------------------------------------------------------
# Conductive layer
# -----------------------------------------------------------------------

class TestConductiveLayer:

    def test_meets_target_R(self):
        c = design_conductive(target_R_sq=15.0)
        assert c.sheet_resistance_ohm_sq <= 15.0

    def test_vis_transmittance_high(self):
        c = design_conductive(target_R_sq=15.0)
        assert c.vis_transmittance > 0.80

    def test_indium_free_preferred(self):
        c = design_conductive(prefer_indium_free=True)
        assert c.indium_free

    def test_ito_available(self):
        c = design_conductive(prefer_indium_free=False)
        assert c.material in TCO_MATERIALS

    def test_lower_R_thicker_film(self):
        c_low = design_conductive(target_R_sq=5.0)
        c_high = design_conductive(target_R_sq=50.0)
        assert c_low.thickness_nm >= c_high.thickness_nm

    def test_lower_R_lower_T(self):
        c_low = design_conductive(target_R_sq=5.0)
        c_high = design_conductive(target_R_sq=50.0)
        assert c_low.vis_transmittance <= c_high.vis_transmittance


# -----------------------------------------------------------------------
# Electrochromic
# -----------------------------------------------------------------------

class TestElectrochromic:

    def test_clear_state(self):
        ec = design_electrochromic(T_clear=0.70, T_dark=0.05)
        assert ec.T_clear == 0.70

    def test_dark_state_low(self):
        ec = design_electrochromic(T_clear=0.70, T_dark=0.05)
        assert ec.T_dark < 0.10

    def test_voltage_reasonable(self):
        ec = design_electrochromic()
        assert 0.5 <= ec.switching_voltage_V <= 3.5

    def test_switching_time_positive(self):
        ec = design_electrochromic()
        assert ec.switching_time_s > 0

    def test_larger_area_slower(self):
        ec_small = design_electrochromic(area_m2=0.1)
        ec_large = design_electrochromic(area_m2=10.0)
        assert ec_large.switching_time_s > ec_small.switching_time_s

    def test_modulation_range(self):
        ec = design_electrochromic(T_clear=0.70, T_dark=0.05)
        assert ec.modulation_range > 0.5

    def test_delta_od_positive(self):
        ec = design_electrochromic(T_clear=0.70, T_dark=0.05)
        assert ec.delta_OD > 0

    def test_cycle_life(self):
        ec = design_electrochromic()
        assert ec.cycle_life >= 10000


# -----------------------------------------------------------------------
# Photochromic
# -----------------------------------------------------------------------

class TestPhotochromic:

    def test_clear_state(self):
        pc = design_photochromic(T_clear=0.70)
        assert pc.T_clear == 0.70

    def test_dark_state_low(self):
        pc = design_photochromic(T_clear=0.70)
        assert pc.T_dark < 0.20

    def test_darkening_fast(self):
        pc = design_photochromic()
        assert pc.darkening_tau_s < 60  # under 1 minute

    def test_bleaching_moderate(self):
        pc = design_photochromic()
        assert pc.bleaching_tau_s < 600  # under 10 minutes

    def test_uv_threshold(self):
        pc = design_photochromic()
        assert pc.uv_threshold_mW_cm2 > 0

    def test_agcl_material(self):
        pc = design_photochromic(material="AgCl_nanoparticle")
        assert pc.material == "AgCl_nanoparticle"


# -----------------------------------------------------------------------
# Magnetochromic
# -----------------------------------------------------------------------

class TestMagnetochromic:

    def test_langevin_zero(self):
        assert abs(_langevin(0.0)) < 0.01

    def test_langevin_large(self):
        assert _langevin(100.0) > 0.98

    def test_langevin_monotonic(self):
        assert _langevin(1.0) < _langevin(5.0) < _langevin(20.0)

    def test_zero_field_clear(self):
        mc = design_magnetochromic(target_field_T=0.0001)
        assert mc.T_at_field > mc.T_zero_field * 0.9  # barely changes

    def test_high_field_dark(self):
        mc = design_magnetochromic(target_field_T=1.0)
        assert mc.T_at_field < mc.T_zero_field * 0.5

    def test_alignment_at_01T(self):
        mc = design_magnetochromic(target_field_T=0.1)
        assert mc.alignment_fraction > 0.3

    def test_fast_response(self):
        mc = design_magnetochromic()
        assert mc.response_time_ms < 100  # sub-second


# -----------------------------------------------------------------------
# Switchable window design
# -----------------------------------------------------------------------

class TestSwitchableWindow:

    def test_electrochromic_meets_targets(self):
        d = design_switchable_window(SwitchableTargets(
            switching_mechanism="electrochromic"))
        assert d.meets_targets
        assert d.T_clear_state > 0.50
        assert d.T_dark_state < 0.10

    def test_electrochromic_has_conductor(self):
        d = design_switchable_window(SwitchableTargets(
            switching_mechanism="electrochromic"))
        assert d.conductor is not None

    def test_photochromic_meets_targets(self):
        d = design_switchable_window(SwitchableTargets(
            switching_mechanism="photochromic"))
        assert d.meets_targets

    def test_photochromic_no_conductor(self):
        d = design_switchable_window(SwitchableTargets(
            switching_mechanism="photochromic"))
        assert d.conductor is None

    def test_magnetochromic_low_field(self):
        d = design_switchable_window(SwitchableTargets(
            switching_mechanism="magnetochromic"))
        assert d.magnetochromic is not None
        assert d.magnetochromic.alignment_fraction > 0

    def test_base_window_properties(self):
        d = design_switchable_window()
        assert d.base_uv_block > 90
        assert d.base_ir_R > 0.80
        assert d.base_u_value < 2.5

    def test_iridescence_present(self):
        d = design_switchable_window(SwitchableTargets(iridescent=True))
        assert len(d.iridescence_sweep) > 0

    def test_summary_string(self):
        d = design_switchable_window()
        assert "Switchable" in d.summary
        assert "Mechanism" in d.summary

    def test_modulation_range(self):
        d = design_switchable_window(SwitchableTargets(
            switching_mechanism="electrochromic"))
        assert d.modulation_range > 0.30

    def test_indium_free_conductor(self):
        d = design_switchable_window(SwitchableTargets(
            switching_mechanism="electrochromic",
            prefer_indium_free=True))
        assert d.conductor.indium_free
