"""
tests/test_window_orientation.py -- Tests for orientation-aware window coating.
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.window_orientation import (
    Orientation, OrientedWindowTargets, OrientedWindowDesign,
    design_oriented_window, design_adhesion,
    glass_uv_transmittance, glass_vis_transmittance,
    _layer_order,
)


# -----------------------------------------------------------------------
# Glass model
# -----------------------------------------------------------------------

class TestGlassModel:

    def test_blocks_deep_uv(self):
        assert glass_uv_transmittance(300, 4.0) < 0.30

    def test_passes_vis(self):
        assert glass_uv_transmittance(550, 4.0) > 0.90

    def test_transition_at_320(self):
        T = glass_uv_transmittance(320, 4.0)
        assert 0.30 < T < 0.60

    def test_thicker_glass_blocks_more_uv(self):
        T_thin = glass_uv_transmittance(340, 3.0)
        T_thick = glass_uv_transmittance(340, 6.0)
        assert T_thick < T_thin

    def test_vis_average(self):
        T = glass_vis_transmittance(4.0)
        assert 0.88 < T < 0.97


# -----------------------------------------------------------------------
# Layer ordering rules
# -----------------------------------------------------------------------

class TestLayerRules:

    def test_exterior_has_self_clean(self):
        r = _layer_order(Orientation.EXTERIOR)
        assert r["self_clean"] is True

    def test_interior_no_self_clean(self):
        r = _layer_order(Orientation.INTERIOR)
        assert r["self_clean"] is False

    def test_interior_uv_pre_filtered(self):
        r = _layer_order(Orientation.INTERIOR)
        assert r["uv_pre_filtered"] is True

    def test_exterior_no_pre_filter(self):
        r = _layer_order(Orientation.EXTERIOR)
        assert r["uv_pre_filtered"] is False

    def test_interior_iridescence_seen_by_room(self):
        r = _layer_order(Orientation.INTERIOR)
        assert r["iridescence_viewer"] == "room"

    def test_exterior_iridescence_seen_by_street(self):
        r = _layer_order(Orientation.EXTERIOR)
        assert r["iridescence_viewer"] == "street"

    def test_igu_no_self_clean(self):
        r = _layer_order(Orientation.IGU_SURFACE_2)
        assert r["self_clean"] is False

    def test_igu_no_adhesion(self):
        r = _layer_order(Orientation.IGU_SURFACE_2)
        assert r["adhesion"] is False


# -----------------------------------------------------------------------
# Adhesion
# -----------------------------------------------------------------------

class TestAdhesion:

    def test_adhesion_transparent(self):
        a = design_adhesion()
        assert a.vis_transmittance > 0.99

    def test_adhesion_thin(self):
        a = design_adhesion()
        assert a.thickness_nm < 20


# -----------------------------------------------------------------------
# Exterior orientation
# -----------------------------------------------------------------------

class TestExterior:

    def test_meets_targets(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.EXTERIOR))
        assert d.meets_targets

    def test_has_self_clean(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.EXTERIOR))
        assert d.surfaces[0].has_self_clean

    def test_iridescence_by_street(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.EXTERIOR, iridescent=True))
        assert d.iridescence_seen_by == "street"


# -----------------------------------------------------------------------
# Interior orientation
# -----------------------------------------------------------------------

class TestInterior:

    def test_meets_targets(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR))
        assert d.meets_targets

    def test_no_self_clean(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR))
        assert not d.surfaces[0].has_self_clean

    def test_iridescence_by_room(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR, iridescent=True))
        assert d.iridescence_seen_by == "room"

    def test_glass_pre_filters_uv(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR))
        assert d.glass_uv_block_at_350 > 10  # glass blocks some UV

    def test_uv_block_includes_glass(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR, uv_block_pct=95))
        assert d.total_uv_block_pct > 95  # glass + coating exceeds target

    def test_has_adhesion(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR))
        assert d.surfaces[0].has_adhesion

    def test_interior_switchable(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR, switchable=True,
            switching_mechanism="electrochromic"))
        assert d.switching_mechanism == "electrochromic"
        assert d.T_clear_state > 0
        assert d.T_dark_state < d.T_clear_state


# -----------------------------------------------------------------------
# IGU orientation
# -----------------------------------------------------------------------

class TestIGU:

    def test_meets_targets(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.IGU_SURFACE_2))
        assert d.meets_targets

    def test_no_self_clean_no_adhesion(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.IGU_SURFACE_2))
        s = d.surfaces[0]
        assert not s.has_self_clean
        assert not s.has_adhesion


# -----------------------------------------------------------------------
# Dual orientation
# -----------------------------------------------------------------------

class TestDual:

    def test_two_surfaces(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.DUAL))
        assert len(d.surfaces) == 2

    def test_exterior_surface_first(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.DUAL))
        assert d.surfaces[0].orientation == Orientation.EXTERIOR

    def test_interior_surface_second(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.DUAL))
        assert d.surfaces[1].orientation == Orientation.INTERIOR

    def test_higher_ir_than_single(self):
        d_single = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.EXTERIOR))
        d_dual = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.DUAL))
        assert d_dual.total_ir_R >= d_single.total_ir_R

    def test_dual_switchable(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.DUAL, switchable=True,
            switching_mechanism="electrochromic"))
        # Interior surface should be switchable
        assert d.surfaces[1].switchable

    def test_dual_iridescence_on_exterior(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.DUAL, exterior_iridescent=True))
        assert d.iridescence_seen_by == "street"

    def test_vis_is_product(self):
        d = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.DUAL))
        # Total T should be roughly product of surfaces * glass
        expected = d.surfaces[0].vis_transmittance * d.glass_vis_T * d.surfaces[1].vis_transmittance
        assert abs(d.total_vis_T - expected) < 0.05


# -----------------------------------------------------------------------
# Cross-orientation comparisons
# -----------------------------------------------------------------------

class TestCrossOrientation:

    def test_interior_higher_uv_block(self):
        """Interior benefits from glass pre-filtering."""
        d_ext = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.EXTERIOR))
        d_int = design_oriented_window(OrientedWindowTargets(
            orientation=Orientation.INTERIOR))
        assert d_int.total_uv_block_pct >= d_ext.total_uv_block_pct

    def test_summary_contains_orientation(self):
        for orient in [Orientation.EXTERIOR, Orientation.INTERIOR, Orientation.DUAL]:
            d = design_oriented_window(OrientedWindowTargets(orientation=orient))
            assert orient.value in d.summary
