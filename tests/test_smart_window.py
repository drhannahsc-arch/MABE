"""
tests/test_smart_window.py -- Tests for smart window coating design.
"""

import sys
import os
import math
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.smart_window import (
    uv_transmittance, vis_average_transmittance,
    MultilayerStack, calculate_u_value,
    design_uv_block, design_ir_reflector, design_low_e, design_self_clean,
    design_window, WindowTargets, WindowDesign,
    _wavelength_to_color, MATERIALS,
)


class TestUVAbsorption:

    def test_tio2_blocks_uv(self):
        T = uv_transmittance("TiO2", 200, 350)
        assert T < 0.2

    def test_tio2_transparent_vis(self):
        T = uv_transmittance("TiO2", 200, 550)
        assert T > 0.99

    def test_thicker_blocks_more(self):
        assert uv_transmittance("TiO2", 500, 350) < uv_transmittance("TiO2", 100, 350)

    def test_vis_average_high(self):
        assert vis_average_transmittance("TiO2", 200) > 0.95


class TestMultilayerStack:

    def test_reflectance_increases_with_pairs(self):
        assert MultilayerStack(2.49, 1.46, 7, 1000).peak_reflectance > MultilayerStack(2.49, 1.46, 3, 1000).peak_reflectance

    def test_5_pair_high_reflectance(self):
        assert MultilayerStack(2.49, 1.46, 5, 1000).peak_reflectance > 0.95

    def test_bandwidth_positive(self):
        bw = MultilayerStack(2.49, 1.46, 5, 1000).bandwidth_fraction
        assert 0.2 < bw < 0.5

    def test_angle_shift_blue(self):
        s = MultilayerStack(2.49, 1.46, 5, 1000)
        assert s.angle_shifted_edges(60)[0] < s.band_edges_nm[0]

    def test_ir_stack_transmits_vis(self):
        assert MultilayerStack(2.49, 1.46, 5, 1200).vis_transmittance(0) > 0.9


class TestThermal:

    def test_single_pane_high_u(self):
        assert calculate_u_value(n_panes=1, emissivity=0.84).u_value > 5.0

    def test_double_clear(self):
        u = calculate_u_value(n_panes=2, emissivity=0.84).u_value
        assert 2.5 < u < 3.5

    def test_low_e_reduces_u(self):
        assert calculate_u_value(n_panes=2, emissivity=0.10).u_value < calculate_u_value(n_panes=2, emissivity=0.84).u_value

    def test_triple_better(self):
        assert calculate_u_value(n_panes=3, emissivity=0.10).u_value < calculate_u_value(n_panes=2, emissivity=0.10).u_value

    def test_ag_low_e(self):
        assert calculate_u_value(n_panes=2, emissivity=0.04).u_value < 1.5


class TestLayerDesign:

    def test_uv_block(self):
        uv = design_uv_block(95.0)
        assert uv.uv_block_pct > 90 and uv.vis_transmittance > 0.90

    def test_ir_broadband(self):
        ir, _ = design_ir_reflector(broadband_metal=True)
        assert ir.ir_reflectance > 0.80 and ir.emissivity < 0.10

    def test_ir_dielectric(self):
        _, stack = design_ir_reflector(broadband_metal=False, n_pairs=5)
        assert stack.peak_reflectance > 0.95

    def test_low_e(self):
        le = design_low_e("ITO")
        assert le.vis_transmittance > 0.95 and le.emissivity < 0.15

    def test_self_clean(self):
        sc = design_self_clean()
        assert sc.water_contact_angle > 100 and sc.vis_transmittance > 0.95


class TestIridescence:

    def test_normal_incidence_clear(self):
        d = design_window(WindowTargets(iridescent=True))
        assert d.color_normal == "clear"
        assert "clear" in d.iridescence_sweep.get(0, "").lower()

    def test_oblique_shows_color(self):
        d = design_window(WindowTargets(iridescent=True))
        has_color = any("red" in d.iridescence_sweep.get(a, "") or
                        "orange" in d.iridescence_sweep.get(a, "")
                        for a in [45, 60, 75])
        assert has_color

    def test_zero_vis_penalty_normal(self):
        d_with = design_window(WindowTargets(iridescent=True))
        d_without = design_window(WindowTargets(iridescent=False))
        assert abs(d_with.total_vis_transmittance - d_without.total_vis_transmittance) < 0.05

    def test_sweep_is_angle_dependent(self):
        d = design_window(WindowTargets(iridescent=True))
        assert "clear" in d.iridescence_sweep.get(0, "").lower()
        # At 75 deg should show color
        assert "clear" not in d.iridescence_sweep.get(75, "").lower()


class TestWindowDesign:

    def test_default_meets_targets(self):
        assert design_window().meets_targets

    def test_vis_high(self):
        assert design_window().total_vis_transmittance > 0.70

    def test_uv_blocked(self):
        assert design_window(WindowTargets(uv_block_pct=95)).total_uv_block_pct > 90

    def test_ir_reflected(self):
        assert design_window().total_ir_reflectance > 0.80

    def test_u_value(self):
        assert design_window(WindowTargets(u_value_target=2.0)).thermal.u_value < 2.2

    def test_triple_pane(self):
        assert design_window(WindowTargets(n_panes=3, u_value_target=1.0)).thermal.u_value < 1.1

    def test_self_clean_present(self):
        d = design_window(WindowTargets(self_cleaning=True))
        assert any(l.function == "self_clean" for l in d.layers)

    def test_no_self_clean(self):
        d = design_window(WindowTargets(self_cleaning=False))
        assert not any(l.function == "self_clean" for l in d.layers)

    def test_no_iridescence(self):
        d = design_window(WindowTargets(iridescent=False))
        assert not any(l.function == "iridescent" for l in d.layers)
        assert len(d.iridescence_sweep) == 0

    def test_summary(self):
        s = design_window().summary
        assert "Smart Window" in s and "U-value" in s

    def test_iridescent_adds_ir(self):
        d_i = design_window(WindowTargets(iridescent=True))
        d_n = design_window(WindowTargets(iridescent=False))
        assert d_i.total_ir_reflectance >= d_n.total_ir_reflectance
