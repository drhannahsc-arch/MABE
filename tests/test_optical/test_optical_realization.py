"""
test_optical_realization.py -- Tests for Layer 3/4 optical realization adapters.

Tests:
  - PhotonicGlassAdapter: scoring + full design with Stoeber protocol
  - BraggOpalAdapter: scoring + full design with vertical deposition
  - TMMMultilayerAdapter: scoring + full design with e-beam stack
  - OpticalRanker: ranks adapters, selects correct system per angular behavior
  - FabSpec content: synthesis steps, materials lists, costs, validation plans
  - Integration: FieldInteractionSpec -> ranker -> FabSpec end-to-end
"""

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import (
    AngularBehavior, ApplicationContext, FieldInteractionSpec,
    FieldType, FieldResponse, Solvent,
)
from optical.optical_realization import (
    PhotonicGlassAdapter, BraggOpalAdapter, TMMMultilayerAdapter,
    PhotonicGlassFabSpec, BraggOpalFabSpec, TMMMultilayerFabSpec,
    OpticalRealizationScore, OpticalRankerOutput,
    rank_optical_realizations, ALL_ADAPTERS,
    STOBER_CONDITIONS, SUPPLIERS,
    _interpolate_stober, _estimate_cost_photonic_glass,
)


# ── Fixtures ──

def make_blue_non_iridescent():
    return FieldInteractionSpec(
        field_type=FieldType.ELECTROMAGNETIC,
        target_wavelength_nm=470.0,
        target_x=0.15, target_y=0.10,
        angular_behavior=AngularBehavior.NON_IRIDESCENT,
        substrate="glass",
        target_application=ApplicationContext.RESEARCH,
    )


def make_green_iridescent():
    return FieldInteractionSpec(
        field_type=FieldType.ELECTROMAGNETIC,
        target_wavelength_nm=540.0,
        target_x=0.20, target_y=0.40,
        angular_behavior=AngularBehavior.IRIDESCENT,
        substrate="glass",
        target_application=ApplicationContext.RESEARCH,
    )


def make_red_non_iridescent():
    return FieldInteractionSpec(
        field_type=FieldType.ELECTROMAGNETIC,
        target_wavelength_nm=650.0,
        target_x=0.50, target_y=0.30,
        angular_behavior=AngularBehavior.NON_IRIDESCENT,
        substrate="glass",
        target_application=ApplicationContext.RESEARCH,
    )


def make_ir_filter():
    return FieldInteractionSpec(
        field_type=FieldType.ELECTROMAGNETIC,
        target_wavelength_nm=1000.0,
        angular_behavior=AngularBehavior.DIRECTIONAL,
        substrate="glass",
        target_application=ApplicationContext.RESEARCH,
    )


# ── Stoeber database ──

class TestStoberDatabase:

    def test_conditions_cover_range(self):
        diameters = [s[0] for s in STOBER_CONDITIONS]
        assert min(diameters) <= 100
        assert max(diameters) >= 500

    def test_interpolate_within_range(self):
        teos, nh3, etoh, h2o = _interpolate_stober(200)
        assert teos > 0
        assert nh3 > 0
        assert etoh > 0
        assert abs(teos + nh3 + etoh + h2o - 200) < 5  # ~200 mL total

    def test_interpolate_at_boundary(self):
        teos_lo, _, _, _ = _interpolate_stober(50)
        teos_hi, _, _, _ = _interpolate_stober(800)
        assert teos_hi > teos_lo  # more TEOS for larger particles

    def test_interpolate_between_points(self):
        teos_175, _, _, _ = _interpolate_stober(175)
        teos_150, _, _, _ = _interpolate_stober(150)
        teos_200, _, _, _ = _interpolate_stober(200)
        assert teos_150 < teos_175 < teos_200

    def test_supplier_database(self):
        assert "TEOS" in SUPPLIERS
        assert "NH4OH" in SUPPLIERS
        assert SUPPLIERS["TEOS"]["price_per_L"] > 0


# ── PhotonicGlassAdapter ──

class TestPhotnicGlassAdapter:

    def setup_method(self):
        self.adapter = PhotonicGlassAdapter()

    def test_blue_scores_high(self):
        score = self.adapter.estimate_fidelity(make_blue_non_iridescent())
        assert score.feasible
        assert score.angular_match > 0.8
        assert score.gamut_coverage > 0.8
        assert score.composite > 0.5

    def test_red_gamut_low(self):
        score = self.adapter.estimate_fidelity(make_red_non_iridescent())
        assert score.gamut_coverage < 0.3  # red problem

    def test_iridescent_angular_mismatch(self):
        score = self.adapter.estimate_fidelity(make_green_iridescent())
        assert score.angular_match < 0.3

    def test_design_produces_fab_spec(self):
        fab = self.adapter.design(make_blue_non_iridescent())
        assert isinstance(fab, PhotonicGlassFabSpec)

    def test_design_has_stoeber_params(self):
        fab = self.adapter.design(make_blue_non_iridescent())
        assert fab.stober_TEOS_mL > 0
        assert fab.stober_NH3_mL > 0
        assert fab.stober_EtOH_mL > 0
        assert 100 < fab.target_diameter_nm < 300

    def test_design_has_synthesis_steps(self):
        fab = self.adapter.design(make_blue_non_iridescent())
        assert len(fab.synthesis_steps) >= 5
        assert any("TEOS" in s for s in fab.synthesis_steps)
        assert any("DLS" in s or "centrifuge" in s.lower() for s in fab.synthesis_steps)

    def test_design_has_materials_list(self):
        fab = self.adapter.design(make_blue_non_iridescent())
        assert len(fab.materials_list) >= 3
        assert any("TEOS" in m for m in fab.materials_list)

    def test_design_has_cost(self):
        fab = self.adapter.design(make_blue_non_iridescent())
        assert 5 < fab.estimated_cost_usd < 200

    def test_design_has_validation(self):
        fab = self.adapter.design(make_blue_non_iridescent())
        assert len(fab.validation_plan) >= 3
        assert any("DLS" in v for v in fab.validation_plan)


# ── BraggOpalAdapter ──

class TestBraggOpalAdapter:

    def setup_method(self):
        self.adapter = BraggOpalAdapter()

    def test_iridescent_scores_high(self):
        score = self.adapter.estimate_fidelity(make_green_iridescent())
        assert score.feasible
        assert score.angular_match > 0.8
        assert score.gamut_coverage > 0.8

    def test_non_iridescent_mismatch(self):
        score = self.adapter.estimate_fidelity(make_blue_non_iridescent())
        assert score.angular_match < 0.3

    def test_full_gamut(self):
        """Bragg can do red — no red problem."""
        score = self.adapter.estimate_fidelity(make_red_non_iridescent())
        # Gamut is fine even for red (it's the angular that's wrong)
        assert score.gamut_coverage > 0.5

    def test_design_produces_fab_spec(self):
        fab = self.adapter.design(make_green_iridescent())
        assert isinstance(fab, BraggOpalFabSpec)

    def test_design_has_vertical_deposition(self):
        fab = self.adapter.design(make_green_iridescent())
        assert any("vertical" in s.lower() or "Vertical" in s for s in fab.synthesis_steps)
        assert fab.substrate_orientation == "vertical"
        assert fab.deposition_temperature_C > 40

    def test_design_pdi_requirement(self):
        fab = self.adapter.design(make_green_iridescent())
        assert fab.required_PDI <= 0.05
        assert any("PDI" in s and "5%" in s for s in fab.synthesis_steps)

    def test_design_predicts_peak(self):
        fab = self.adapter.design(make_green_iridescent())
        assert 400 < fab.expected_peak_nm < 700


# ── TMMMultilayerAdapter ──

class TestTMMMultilayerAdapter:

    def setup_method(self):
        self.adapter = TMMMultilayerAdapter()

    def test_directional_scores_high(self):
        score = self.adapter.estimate_fidelity(make_ir_filter())
        assert score.angular_match > 0.8

    def test_full_gamut(self):
        score = self.adapter.estimate_fidelity(make_red_non_iridescent())
        assert score.gamut_coverage > 0.9

    def test_low_fab_accessibility(self):
        score = self.adapter.estimate_fidelity(make_blue_non_iridescent())
        assert score.fabrication_accessibility < 0.5  # needs vacuum

    def test_high_durability(self):
        score = self.adapter.estimate_fidelity(make_blue_non_iridescent())
        assert score.durability > 0.8

    def test_design_produces_layer_stack(self):
        fab = self.adapter.design(make_green_iridescent())
        assert isinstance(fab, TMMMultilayerFabSpec)
        assert fab.n_layers >= 4
        assert len(fab.layer_materials) == fab.n_layers
        assert len(fab.layer_thicknesses_nm) == fab.n_layers

    def test_design_quarter_wave(self):
        spec = FieldInteractionSpec(
            target_wavelength_nm=550.0,
            angular_behavior=AngularBehavior.DIRECTIONAL,
        )
        fab = self.adapter.design(spec)
        assert fab.design_type == "quarter_wave"
        assert fab.stopband_center_nm == 550.0
        assert fab.total_thickness_nm > 0


# ── Ranker ──

class TestOpticalRanker:

    def test_three_adapters_registered(self):
        assert len(ALL_ADAPTERS) == 3

    def test_non_iridescent_recommends_photonic_glass(self):
        result = rank_optical_realizations(make_blue_non_iridescent())
        assert result.recommended == "photonic_glass"
        assert len(result.scores) == 3

    def test_iridescent_recommends_bragg(self):
        result = rank_optical_realizations(make_green_iridescent())
        assert result.recommended == "bragg_opal"

    def test_directional_recommends_tmm(self):
        result = rank_optical_realizations(make_ir_filter())
        assert result.recommended == "tmm_multilayer"

    def test_scores_sorted_descending(self):
        result = rank_optical_realizations(make_blue_non_iridescent())
        composites = [s.composite for s in result.scores]
        assert composites == sorted(composites, reverse=True)

    def test_has_rationale(self):
        result = rank_optical_realizations(make_blue_non_iridescent())
        assert len(result.recommendation_rationale) > 0

    def test_run_design_produces_fab_spec(self):
        result = rank_optical_realizations(make_blue_non_iridescent(), run_design=True)
        assert result.fab_spec is not None
        assert isinstance(result.fab_spec, PhotonicGlassFabSpec)

    def test_iridescent_design_produces_bragg_fab(self):
        result = rank_optical_realizations(make_green_iridescent(), run_design=True)
        assert result.fab_spec is not None
        assert isinstance(result.fab_spec, BraggOpalFabSpec)


# ── End-to-end ──

class TestEndToEnd:

    def test_field_spec_to_fab_spec_pipeline(self):
        """FieldInteractionSpec -> ranker -> design -> FabSpec with protocols."""
        spec = make_blue_non_iridescent()
        result = rank_optical_realizations(spec, run_design=True)

        assert result.recommended == "photonic_glass"
        fab = result.fab_spec
        assert isinstance(fab, PhotonicGlassFabSpec)
        assert fab.target_diameter_nm > 0
        assert fab.stober_TEOS_mL > 0
        assert len(fab.synthesis_steps) >= 5
        assert len(fab.materials_list) >= 3
        assert fab.estimated_cost_usd > 0
        assert len(fab.validation_plan) >= 3
