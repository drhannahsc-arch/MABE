"""
Tests for glycan/athena_staple_spec.py -- DNA origami pulldown staple specs.
"""

import pytest
from glycan.athena_staple_spec import (
    design_origami_pulldown,
    list_cage_types,
    list_presets,
    CAGE_GEOMETRIES,
    PULLDOWN_PRESETS,
    OrigamiPulldownSpec,
)


# ── Basic design ────────────────────────────────────────────────────────

class TestDesignBasic:
    def test_default_design_returns_spec(self):
        spec = design_origami_pulldown("Man")
        assert isinstance(spec, OrigamiPulldownSpec)

    def test_summary_is_string(self):
        spec = design_origami_pulldown("Man")
        assert isinstance(spec.summary, str)
        assert "Man" in spec.summary

    def test_order_sheet_has_entries(self):
        spec = design_origami_pulldown("Man")
        sheet = spec.order_sheet
        assert len(sheet) >= 4  # sugar + magnetic + passivation + unmodified + scaffold
        roles = {e["role"] for e in sheet}
        assert "sugar" in roles
        assert "magnetic" in roles
        assert "scaffold" in roles

    def test_unknown_cage_raises(self):
        with pytest.raises(ValueError, match="Unknown cage"):
            design_origami_pulldown("Man", cage_type="nonexistent")

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            design_origami_pulldown("Man", config_preset="nonexistent")


# ── Staple assignment ───────────────────────────────────────────────────

class TestStapleAssignment:
    def test_staple_counts_sum_to_exterior(self):
        spec = design_origami_pulldown("Man", cage_type="octahedron_DX_30")
        cage = CAGE_GEOMETRIES["octahedron_DX_30"]
        ext_total = spec.n_sugar_staples + spec.n_magnetic_staples + spec.n_passivation_staples + spec.n_reporter_staples
        assert ext_total == cage.n_exterior_staples

    def test_unmodified_equals_interior(self):
        spec = design_origami_pulldown("Man", cage_type="octahedron_DX_30")
        cage = CAGE_GEOMETRIES["octahedron_DX_30"]
        assert spec.n_unmodified_staples == cage.n_interior_staples

    def test_cell_capture_preset_50pct_sugar(self):
        spec = design_origami_pulldown("Man", cage_type="octahedron_DX_30", config_preset="cell_capture")
        cage = CAGE_GEOMETRIES["octahedron_DX_30"]
        assert spec.n_sugar_staples == int(cage.n_exterior_staples * 0.50)

    def test_high_avidity_more_sugar(self):
        cell = design_origami_pulldown("Man", config_preset="cell_capture")
        avid = design_origami_pulldown("Man", config_preset="high_avidity")
        assert avid.n_sugar_staples > cell.n_sugar_staples

    def test_reporter_present_in_diagnostic(self):
        spec = design_origami_pulldown("Man", config_preset="diagnostic")
        assert spec.n_reporter_staples > 0
        assert spec.reporter_mod is not None

    def test_no_reporter_in_environmental(self):
        spec = design_origami_pulldown("Man", config_preset="environmental")
        assert spec.n_reporter_staples == 0


# ── Click chemistry ─────────────────────────────────────────────────────

class TestClickChemistry:
    def test_spaac_uses_dbco(self):
        spec = design_origami_pulldown("Man", click_chemistry="SPAAC")
        assert "DBCO" in spec.sugar_mod.modification_3prime

    def test_cuaac_uses_alkyne(self):
        spec = design_origami_pulldown("Man", click_chemistry="CuAAC")
        assert "alkyne" in spec.sugar_mod.modification_3prime

    def test_spaac_more_expensive(self):
        spaac = design_origami_pulldown("Man", click_chemistry="SPAAC")
        cuaac = design_origami_pulldown("Man", click_chemistry="CuAAC")
        assert spaac.sugar_mod.cost_per_staple_usd > cuaac.sugar_mod.cost_per_staple_usd

    def test_spaac_biocompatible_note(self):
        spec = design_origami_pulldown("Man", click_chemistry="SPAAC")
        assert any("copper-free" in n for n in spec.notes)


# ── Cost ────────────────────────────────────────────────────────────────

class TestCost:
    def test_total_cost_positive(self):
        spec = design_origami_pulldown("Man")
        assert spec.total_estimated_cost_usd > 0

    def test_larger_cage_more_expensive(self):
        small = design_origami_pulldown("Man", cage_type="tetrahedron_DX_30")
        large = design_origami_pulldown("Man", cage_type="icosahedron_DX_30")
        assert large.total_estimated_cost_usd > small.total_estimated_cost_usd

    def test_order_sheet_costs_sum(self):
        spec = design_origami_pulldown("Man")
        sheet_total = sum(e["cost_total"] for e in spec.order_sheet)
        # Sheet total = staple + scaffold cost. Total includes reagent + beads too.
        assert sheet_total <= spec.total_estimated_cost_usd

    def test_cost_in_documented_range(self):
        """Octahedron DX 30 should be $2000-15000 range (SPAAC is expensive)."""
        spec = design_origami_pulldown("Man", cage_type="octahedron_DX_30")
        assert 1000 < spec.total_estimated_cost_usd < 20000
        # CuAAC version should be cheaper
        spec_cu = design_origami_pulldown("Man", cage_type="octahedron_DX_30", click_chemistry="CuAAC")
        assert spec_cu.total_estimated_cost_usd < spec.total_estimated_cost_usd


# ── Sugar spacing and valency ───────────────────────────────────────────

class TestSpacingValency:
    def test_sugar_spacing_positive(self):
        spec = design_origami_pulldown("Man")
        assert spec.sugar_spacing_nm > 0

    def test_more_sugars_tighter_spacing(self):
        cell = design_origami_pulldown("Man", config_preset="cell_capture")
        avid = design_origami_pulldown("Man", config_preset="high_avidity")
        assert avid.sugar_spacing_nm < cell.sugar_spacing_nm

    def test_effective_valency_positive(self):
        spec = design_origami_pulldown("Man")
        assert spec.effective_valency >= 1

    def test_larger_cage_more_valency(self):
        small = design_origami_pulldown("Man", cage_type="tetrahedron_DX_30")
        large = design_origami_pulldown("Man", cage_type="icosahedron_DX_30")
        assert large.effective_valency >= small.effective_valency


# ── Different sugars ────────────────────────────────────────────────────

class TestSugarVariants:
    def test_neu5ac_c9(self):
        spec = design_origami_pulldown("Neu5Ac", click_position="C9")
        assert spec.sugar == "Neu5Ac"
        assert spec.click_position == "C9"
        assert "Neu5Ac" in spec.sugar_mod.notes

    def test_gal_c1(self):
        spec = design_origami_pulldown("Gal", click_position="C1")
        assert spec.sugar == "Gal"

    def test_man_for_macrophage(self):
        spec = design_origami_pulldown("Man", click_position="C1")
        assert spec.sugar == "Man"


# ── Cage types ──────────────────────────────────────────────────────────

class TestCageTypes:
    def test_all_cage_types_work(self):
        for ct in list_cage_types():
            spec = design_origami_pulldown("Man", cage_type=ct)
            assert spec.n_sugar_staples > 0

    def test_all_presets_work(self):
        for preset in list_presets():
            spec = design_origami_pulldown("Man", config_preset=preset)
            assert isinstance(spec, OrigamiPulldownSpec)

    def test_icosahedron_most_staples(self):
        specs = {ct: design_origami_pulldown("Man", cage_type=ct) for ct in list_cage_types()}
        ico = specs["icosahedron_DX_30"]
        for name, spec in specs.items():
            assert ico.n_sugar_staples >= spec.n_sugar_staples, \
                f"Icosahedron should have most sugar staples, but {name} has more"
