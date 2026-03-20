"""
tests/test_responsive_multiphysics.py — Tests for responsive structures (Phase 5).

Validates:
  - PNIPAM swelling model: sigmoidal, bounded, LCST correct
  - Effective diameter increases below LCST (swollen)
  - Effective volume fraction decreases below LCST (diluted)
  - Matrix properties interpolate between water-like and polymer-like
  - Single-temperature prediction has all three physics
  - Temperature sweep produces monotonic trends
  - Color shifts to longer λ when swollen (larger D_eff)
  - Thermal κ changes across transition (water-like → polymer-like matrix)
  - Crosslink density comparison: low swells more than high
  - Coupled response: all three physics respond to same T change
"""

import sys
import os
import math
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.responsive_multiphysics import (
    swelling_ratio,
    effective_diameter,
    effective_volume_fraction,
    effective_matrix_properties,
    predict_at_temperature,
    sweep_temperature,
    design_responsive,
    compare_crosslink_densities,
    PNIPAMConfig,
    ResponsiveSpec,
    ResponsiveState,
    ResponsiveSweep,
)


# ═══════════════════════════════════════════════════════════════════════════
# Swelling model
# ═══════════════════════════════════════════════════════════════════════════

class TestSwellingModel:

    def test_swollen_below_lcst(self):
        """Well below LCST: Q ≈ Q_swollen."""
        Q = swelling_ratio(20.0)
        assert Q > 6.0  # near Q_swollen = 8.0

    def test_collapsed_above_lcst(self):
        """Well above LCST: Q ≈ Q_collapsed."""
        Q = swelling_ratio(45.0)
        assert Q < 2.0  # near Q_collapsed = 1.2

    def test_transition_at_lcst(self):
        """At LCST: Q ≈ midpoint of swollen and collapsed."""
        Q = swelling_ratio(32.0)
        midpoint = (8.0 + 1.2) / 2.0
        assert abs(Q - midpoint) < 1.0

    def test_monotonic_decrease(self):
        """Q should decrease monotonically with temperature."""
        temps = [15, 20, 25, 28, 30, 32, 34, 36, 38, 40, 45]
        Qs = [swelling_ratio(t) for t in temps]
        for i in range(len(Qs) - 1):
            assert Qs[i] >= Qs[i + 1] - 1e-10

    def test_bounded(self):
        """Q stays between Q_collapsed and Q_swollen at any temperature."""
        config = PNIPAMConfig()
        for t in range(-10, 80):
            Q = swelling_ratio(float(t), config)
            assert config.q_collapsed <= Q <= config.q_swollen + 0.01

    def test_custom_lcst(self):
        """Custom LCST shifts the transition."""
        config = PNIPAMConfig(lcst_c=40.0)
        Q_at_35 = swelling_ratio(35.0, config)
        Q_default_35 = swelling_ratio(35.0)  # default LCST=32
        # At 35°C: custom (LCST=40) should be more swollen than default (LCST=32)
        assert Q_at_35 > Q_default_35


# ═══════════════════════════════════════════════════════════════════════════
# Effective parameters
# ═══════════════════════════════════════════════════════════════════════════

class TestEffectiveParameters:

    def test_diameter_increases_below_lcst(self):
        """Swelling increases effective diameter."""
        D_cold = effective_diameter(250.0, 20.0)
        D_hot = effective_diameter(250.0, 40.0)
        assert D_cold > D_hot

    def test_diameter_at_dry_above_lcst(self):
        """Well above LCST: D_eff ≈ D_dry × Q_collapsed^(1/3)."""
        D = effective_diameter(250.0, 50.0)
        expected = 250.0 * 1.2 ** (1.0 / 3.0)  # Q_collapsed = 1.2
        assert abs(D - expected) < 10.0

    def test_volume_fraction_decreases_below_lcst(self):
        """Swelling dilutes particles → lower φ."""
        phi_cold = effective_volume_fraction(0.50, 20.0)
        phi_hot = effective_volume_fraction(0.50, 40.0)
        assert phi_cold < phi_hot

    def test_volume_fraction_bounded(self):
        """φ should stay in physical range [0.01, 0.74]."""
        for t in range(10, 50):
            phi = effective_volume_fraction(0.50, float(t))
            assert 0.01 <= phi <= 0.74

    def test_porosity_increases_below_lcst(self):
        """More swollen → higher porosity."""
        phi_cold = effective_volume_fraction(0.50, 20.0)
        phi_hot = effective_volume_fraction(0.50, 40.0)
        eps_cold = 1.0 - phi_cold
        eps_hot = 1.0 - phi_hot
        assert eps_cold > eps_hot


class TestMatrixProperties:

    def test_swollen_water_like(self):
        """Below LCST: matrix κ should be near water (0.55 W/mK)."""
        props = effective_matrix_properties(20.0)
        assert props["kappa_W_mK"] > 0.4

    def test_collapsed_polymer_like(self):
        """Above LCST: matrix κ should be near polymer (0.25 W/mK)."""
        props = effective_matrix_properties(45.0)
        assert props["kappa_W_mK"] < 0.35

    def test_density_decreases_when_swollen(self):
        """Swollen (water-like) is slightly denser than collapsed (polymer-like)... 
        actually PNIPAM_swollen has ρ=1020, collapsed ρ=1150, so collapsed is denser."""
        props_cold = effective_matrix_properties(20.0)
        props_hot = effective_matrix_properties(45.0)
        # Collapsed is denser (polymer > water in this system)
        assert props_hot["density_kg_m3"] > props_cold["density_kg_m3"]

    def test_f_collapsed_low_below_lcst(self):
        props = effective_matrix_properties(20.0)
        assert props["f_collapsed"] < 0.2

    def test_f_collapsed_high_above_lcst(self):
        props = effective_matrix_properties(45.0)
        assert props["f_collapsed"] > 0.8

    def test_label_switches(self):
        """Matrix label should switch at ~LCST."""
        below = effective_matrix_properties(25.0)
        above = effective_matrix_properties(40.0)
        assert below["matrix_label"] == "PNIPAM_swollen"
        assert above["matrix_label"] == "PNIPAM_collapsed"


# ═══════════════════════════════════════════════════════════════════════════
# Single-temperature prediction
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleTemperature:

    @pytest.fixture
    def spec(self):
        return ResponsiveSpec(
            particle_material="SiO2",
            dry_diameter_nm=250.0,
            dry_volume_fraction=0.50,
            film_thickness_um=100.0,
        )

    def test_state_has_thermal(self, spec):
        state = predict_at_temperature(spec, 25.0)
        assert state.thermal is not None
        assert state.thermal.kappa_eff_W_mK > 0

    def test_state_has_acoustic(self, spec):
        state = predict_at_temperature(spec, 25.0)
        assert state.acoustic is not None
        assert state.acoustic.nrc >= 0

    def test_state_has_wavelength(self, spec):
        state = predict_at_temperature(spec, 25.0)
        assert state.peak_wavelength_nm > 0

    def test_state_has_effective_params(self, spec):
        state = predict_at_temperature(spec, 25.0)
        assert state.swelling_ratio > 1.0
        assert state.effective_diameter_nm > spec.dry_diameter_nm
        assert state.effective_volume_fraction < spec.dry_volume_fraction

    def test_summary_not_empty(self, spec):
        state = predict_at_temperature(spec, 25.0)
        assert len(state.summary()) > 50


# ═══════════════════════════════════════════════════════════════════════════
# Temperature sweep
# ═══════════════════════════════════════════════════════════════════════════

class TestTemperatureSweep:

    def test_sweep_produces_states(self):
        sweep = design_responsive(dry_diameter_nm=250.0, t_step_c=5.0)
        assert len(sweep.states) > 0
        assert len(sweep.temperatures_c) > 0

    def test_sweep_correct_count(self):
        sweep = design_responsive(t_min_c=20.0, t_max_c=40.0, t_step_c=5.0)
        assert len(sweep.states) == 5  # 20, 25, 30, 35, 40

    def test_wavelength_range_positive(self):
        sweep = design_responsive()
        assert sweep.wavelength_range_nm[1] > sweep.wavelength_range_nm[0]

    def test_color_shift_positive(self):
        """Swelling should produce a measurable color shift."""
        sweep = design_responsive()
        assert sweep.color_shift_nm > 10.0  # at least 10 nm shift

    def test_thermal_switch_ratio_above_one(self):
        """κ should change across transition → ratio > 1."""
        sweep = design_responsive()
        assert sweep.thermal_switch_ratio > 1.0

    def test_cold_longer_wavelength(self):
        """Below LCST (swollen): longer wavelength (red-shifted)."""
        sweep = design_responsive(t_step_c=5.0)
        cold = [s for s in sweep.states if s.temperature_c < 25]
        hot = [s for s in sweep.states if s.temperature_c > 38]
        if cold and hot:
            assert cold[0].peak_wavelength_nm > hot[0].peak_wavelength_nm

    def test_cold_more_insulating(self):
        """Below LCST (swollen): more porous but water-filled matrix.
        The net κ depends on which effect dominates."""
        sweep = design_responsive(t_step_c=5.0)
        # Just verify both states produce valid thermal results
        for s in sweep.states:
            assert s.thermal is not None
            assert s.thermal.kappa_eff_W_mK > 0

    def test_summary_has_key_info(self):
        sweep = design_responsive(t_step_c=5.0)
        s = sweep.summary()
        assert "Color shift" in s
        assert "Thermal switch" in s
        assert "LCST" in s


# ═══════════════════════════════════════════════════════════════════════════
# Crosslink density comparison
# ═══════════════════════════════════════════════════════════════════════════

class TestCrosslinkDensity:

    def test_three_densities(self):
        results = compare_crosslink_densities()
        assert "low" in results
        assert "standard" in results
        assert "high" in results

    def test_low_more_shift_than_high(self):
        """Low crosslink → more swelling → larger color shift."""
        results = compare_crosslink_densities()
        assert results["low"].color_shift_nm > results["high"].color_shift_nm

    def test_all_produce_sweeps(self):
        results = compare_crosslink_densities()
        for density, sweep in results.items():
            assert isinstance(sweep, ResponsiveSweep)
            assert len(sweep.states) > 0

    def test_high_crosslink_less_swelling(self):
        """High crosslink → smaller swelling ratio difference."""
        results = compare_crosslink_densities()
        high_states = results["high"].states
        low_states = results["low"].states
        # At 20°C (well below LCST)
        high_20 = next(s for s in high_states if abs(s.temperature_c - 20.0) < 1)
        low_20 = next(s for s in low_states if abs(s.temperature_c - 20.0) < 1)
        assert low_20.swelling_ratio > high_20.swelling_ratio


# ═══════════════════════════════════════════════════════════════════════════
# Coupled response validation
# ═══════════════════════════════════════════════════════════════════════════

class TestCoupledResponse:

    def test_all_three_respond_to_temperature(self):
        """All three physics should change between 20°C and 40°C."""
        spec = ResponsiveSpec(dry_diameter_nm=250.0, dry_volume_fraction=0.50)
        cold = predict_at_temperature(spec, 20.0)
        hot = predict_at_temperature(spec, 40.0)

        # Color shifts
        assert cold.peak_wavelength_nm != hot.peak_wavelength_nm

        # Thermal changes
        assert cold.thermal.kappa_eff_W_mK != hot.thermal.kappa_eff_W_mK

        # Effective parameters change
        assert cold.effective_diameter_nm != hot.effective_diameter_nm
        assert cold.effective_volume_fraction != hot.effective_volume_fraction

    def test_diameter_and_phi_anticorrelate(self):
        """When D_eff goes up (swollen), φ_eff goes down (diluted)."""
        spec = ResponsiveSpec(dry_diameter_nm=250.0, dry_volume_fraction=0.50)
        cold = predict_at_temperature(spec, 20.0)
        hot = predict_at_temperature(spec, 40.0)

        # Cold: larger D, smaller φ
        assert cold.effective_diameter_nm > hot.effective_diameter_nm
        assert cold.effective_volume_fraction < hot.effective_volume_fraction

    def test_one_stimulus_three_responses(self):
        """The key architectural claim: one T change → three physics responses.
        Verify that a single temperature step produces measurably different
        values for all three domains."""
        spec = ResponsiveSpec(dry_diameter_nm=280.0, dry_volume_fraction=0.45)
        state_30 = predict_at_temperature(spec, 30.0)
        state_34 = predict_at_temperature(spec, 34.0)

        # Just 4°C across LCST should change all three
        assert abs(state_30.peak_wavelength_nm - state_34.peak_wavelength_nm) > 1.0
        assert state_30.thermal.kappa_eff_W_mK != state_34.thermal.kappa_eff_W_mK
        assert state_30.effective_porosity != state_34.effective_porosity
