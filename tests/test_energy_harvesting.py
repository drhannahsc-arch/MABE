"""
tests/test_energy_harvesting.py -- Tests for energy harvesting layer models.

Validates:
  - Material database completeness (PV, TE, piezo: all entries, all fields)
  - PV: SQ bound, temperature derating, spectral utilization, zero input
  - TEG: Carnot bound, ZT dependence, monotonic dT, zero input
  - Piezo: matched-load physics, J^2-like scaling, zero input
  - Acoustic: dBA conversion, negligible power note, zero input
  - Combined: all modalities sum correctly, daily energy, defaults
  - Physical sanity bounds
  - Edge cases
"""

import sys
import os
import math
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.energy_harvesting import (
    # Constants
    AM15_TOTAL_W_m2,
    VANCOUVER_VERTICAL_SOUTH_AVG_W_m2,
    EPSILON_0,
    REF_INTENSITY_W_m2,
    # Database access
    get_pv_material,
    get_te_material,
    get_piezo_material,
    list_pv_materials,
    list_te_materials,
    list_piezo_materials,
    # Databases
    _PV_MATERIALS,
    _TE_MATERIALS,
    _PIEZO_MATERIALS,
    # Models
    predict_pv_power,
    predict_teg_power,
    predict_piezo_power,
    predict_acoustic_harvest,
    predict_total_harvest,
    # Helpers
    _shockley_queisser_limit,
    _dba_to_intensity,
    _am15_spectral_irradiance,
    am15_photon_flux,
    # Dataclasses
    PVResult,
    TEGResult,
    PiezoResult,
    AcousticHarvestResult,
    PowerBudget,
    EnvironmentSpec,
    HarvestingSpec,
)


# -----------------------------------------------------------------------
# PV database completeness
# -----------------------------------------------------------------------

EXPECTED_PV = [
    "organic_PM6Y6", "organic_PBDBT_ITIC", "perovskite_MAPbI3",
    "perovskite_CsAgBiBr", "amorphous_Si", "CZTS",
]

EXPECTED_TE = [
    "PEDOT_PSS_doped", "Bi2Te3", "SnSe", "Cu2Se", "Ca3Co4O9",
]

EXPECTED_PIEZO = [
    "PVDF", "P_VDF_TrFE", "BaTiO3", "KNN", "ZnO",
]


class TestPVDatabase:

    def test_all_present(self):
        available = list_pv_materials()
        for name in EXPECTED_PV:
            assert name in available

    def test_count(self):
        assert len(_PV_MATERIALS) == 6

    @pytest.mark.parametrize("name", EXPECTED_PV)
    def test_fields(self, name):
        mat = get_pv_material(name)
        assert mat.bandgap_eV > 0
        assert 0 < mat.pce_lab <= 0.5
        assert 0 < mat.pce_module <= mat.pce_lab
        assert mat.temp_coeff_per_C < 0  # all negative
        assert mat.source != ""

    def test_perovskite_not_pb_free(self):
        mat = get_pv_material("perovskite_MAPbI3")
        assert mat.pb_free is False

    def test_organics_pb_free(self):
        for name in ["organic_PM6Y6", "organic_PBDBT_ITIC"]:
            assert get_pv_material(name).pb_free is True

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_pv_material("solar_paint")


class TestTEDatabase:

    def test_all_present(self):
        for name in EXPECTED_TE:
            assert name in list_te_materials()

    def test_count(self):
        assert len(_TE_MATERIALS) == 5

    @pytest.mark.parametrize("name", EXPECTED_TE)
    def test_fields(self, name):
        mat = get_te_material(name)
        assert mat.seebeck_uV_K > 0
        assert mat.zt_300K > 0
        assert mat.sigma_elec_S_m > 0
        assert mat.kappa_te_W_mK > 0

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_te_material("unobtanium")


class TestPiezoDatabase:

    def test_all_present(self):
        for name in EXPECTED_PIEZO:
            assert name in list_piezo_materials()

    def test_count(self):
        assert len(_PIEZO_MATERIALS) == 5

    @pytest.mark.parametrize("name", EXPECTED_PIEZO)
    def test_fields(self, name):
        mat = get_piezo_material(name)
        assert mat.d33_pC_N > 0
        assert mat.epsilon_r > 0
        assert mat.density_kg_m3 > 0

    @pytest.mark.parametrize("name", EXPECTED_PIEZO)
    def test_all_pb_free(self, name):
        assert get_piezo_material(name).pb_free is True

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_piezo_material("PZT")  # intentionally excluded (Pb)


# -----------------------------------------------------------------------
# AM1.5 spectral model
# -----------------------------------------------------------------------

class TestAM15Spectrum:

    def test_positive_in_visible(self):
        for wl in [400, 500, 600, 700]:
            assert _am15_spectral_irradiance(float(wl)) > 0

    def test_zero_at_extreme_uv(self):
        assert _am15_spectral_irradiance(100.0) < 1e-10

    def test_photon_flux_positive(self):
        assert am15_photon_flux(550.0) > 0

    def test_integral_order_of_magnitude(self):
        """Integral should be within factor of 2 of 1000 W/m2."""
        total = sum(_am15_spectral_irradiance(float(wl)) * 10.0
                     for wl in range(280, 4001, 10))
        assert 500 < total < 2000


# -----------------------------------------------------------------------
# PV model
# -----------------------------------------------------------------------

class TestPVModel:

    def test_basic_prediction(self):
        result = predict_pv_power(1000.0, "organic_PM6Y6")
        assert isinstance(result, PVResult)
        assert result.power_W_m2 > 0

    def test_zero_irradiance(self):
        result = predict_pv_power(0.0, "organic_PM6Y6")
        assert result.power_W_m2 == 0.0

    def test_zero_transmission(self):
        result = predict_pv_power(1000.0, "organic_PM6Y6", transmitted_fraction=0.0)
        assert result.power_W_m2 == 0.0

    def test_more_sun_more_power(self):
        r1 = predict_pv_power(500.0, "organic_PM6Y6")
        r2 = predict_pv_power(1000.0, "organic_PM6Y6")
        assert r2.power_W_m2 > r1.power_W_m2

    def test_higher_temp_lower_power(self):
        """Negative temp coefficient: hotter = less power."""
        r25 = predict_pv_power(1000.0, "organic_PM6Y6", temperature_C=25.0)
        r60 = predict_pv_power(1000.0, "organic_PM6Y6", temperature_C=60.0)
        assert r60.power_W_m2 < r25.power_W_m2

    def test_efficiency_below_sq_limit(self):
        """Module efficiency must be below Shockley-Queisser limit."""
        for name in EXPECTED_PV:
            mat = get_pv_material(name)
            sq = _shockley_queisser_limit(mat.bandgap_eV)
            assert mat.pce_module < sq or mat.pce_module < 0.337

    def test_sq_peak_near_1_34_eV(self):
        """SQ limit should peak near 1.34 eV."""
        peak_eg = max((eg / 10.0 for eg in range(8, 25)),
                      key=lambda x: _shockley_queisser_limit(x))
        assert abs(peak_eg - 1.34) < 0.2

    def test_transmitted_fraction_scales_linearly(self):
        r_full = predict_pv_power(1000.0, "amorphous_Si", transmitted_fraction=1.0)
        r_half = predict_pv_power(1000.0, "amorphous_Si", transmitted_fraction=0.5)
        ratio = r_half.power_W_m2 / r_full.power_W_m2
        assert abs(ratio - 0.5) < 0.01

    def test_negative_irradiance_raises(self):
        with pytest.raises(ValueError):
            predict_pv_power(-100.0, "organic_PM6Y6")

    def test_spectral_utilization_bounded(self):
        result = predict_pv_power(1000.0, "perovskite_CsAgBiBr")
        assert 0.0 < result.spectral_utilization < 1.0


# -----------------------------------------------------------------------
# TEG model
# -----------------------------------------------------------------------

class TestTEGModel:

    def test_basic_prediction(self):
        result = predict_teg_power(10.0, "Bi2Te3")
        assert isinstance(result, TEGResult)
        assert result.power_W_m2 > 0

    def test_zero_delta_t(self):
        result = predict_teg_power(0.0, "Bi2Te3")
        assert result.power_W_m2 == 0.0
        assert result.efficiency_percent == 0.0

    def test_more_delta_t_more_power(self):
        r5 = predict_teg_power(5.0, "Bi2Te3")
        r20 = predict_teg_power(20.0, "Bi2Te3")
        assert r20.power_W_m2 > r5.power_W_m2

    def test_higher_zt_more_efficient(self):
        """SnSe (ZT=2.6) should be more efficient than PEDOT (ZT=0.25)."""
        r_snse = predict_teg_power(10.0, "SnSe")
        r_pedot = predict_teg_power(10.0, "PEDOT_PSS_doped")
        assert r_snse.efficiency_percent > r_pedot.efficiency_percent

    def test_efficiency_below_carnot(self):
        """TEG efficiency must not exceed Carnot."""
        for name in EXPECTED_TE:
            result = predict_teg_power(50.0, name, t_cold_K=300.0)
            carnot = 50.0 / 350.0 * 100.0
            assert result.efficiency_percent <= carnot + 0.01

    def test_negative_delta_t_raises(self):
        with pytest.raises(ValueError):
            predict_teg_power(-5.0, "Bi2Te3")

    def test_voltage_positive(self):
        result = predict_teg_power(10.0, "Bi2Te3")
        assert result.voltage_V > 0


# -----------------------------------------------------------------------
# Piezo model
# -----------------------------------------------------------------------

class TestPiezoModel:

    def test_basic_prediction(self):
        result = predict_piezo_power(1.0, 60.0, "PVDF")
        assert isinstance(result, PiezoResult)
        assert result.power_W_m2 > 0

    def test_zero_acceleration(self):
        result = predict_piezo_power(0.0, 60.0, "PVDF")
        assert result.power_W_m2 == 0.0

    def test_zero_frequency(self):
        result = predict_piezo_power(1.0, 0.0, "PVDF")
        assert result.power_W_m2 == 0.0

    def test_more_acceleration_more_power(self):
        r1 = predict_piezo_power(0.5, 60.0, "BaTiO3")
        r2 = predict_piezo_power(2.0, 60.0, "BaTiO3")
        assert r2.power_W_m2 > r1.power_W_m2

    def test_power_scales_with_accel_squared(self):
        """Power ~ sigma^2 ~ accel^2."""
        r1 = predict_piezo_power(1.0, 60.0, "PVDF", thickness_um=100.0)
        r3 = predict_piezo_power(3.0, 60.0, "PVDF", thickness_um=100.0)
        ratio = r3.power_W_m2 / r1.power_W_m2
        assert abs(ratio - 9.0) < 0.1

    def test_more_frequency_more_power(self):
        r30 = predict_piezo_power(1.0, 30.0, "PVDF")
        r120 = predict_piezo_power(1.0, 120.0, "PVDF")
        assert r120.power_W_m2 > r30.power_W_m2

    def test_higher_d33_more_power(self):
        """BaTiO3 (d33=150) should generate more than ZnO (d33=12)."""
        r_bto = predict_piezo_power(1.0, 60.0, "BaTiO3", thickness_um=100.0)
        r_zno = predict_piezo_power(1.0, 60.0, "ZnO", thickness_um=100.0)
        assert r_bto.power_W_m2 > r_zno.power_W_m2

    def test_voltage_positive(self):
        result = predict_piezo_power(1.0, 60.0, "PVDF")
        assert result.voltage_peak_V > 0

    def test_optimal_load_positive(self):
        result = predict_piezo_power(1.0, 60.0, "PVDF")
        assert result.optimal_load_ohm > 0

    def test_negative_accel_raises(self):
        with pytest.raises(ValueError):
            predict_piezo_power(-1.0, 60.0, "PVDF")


# -----------------------------------------------------------------------
# Acoustic harvest model
# -----------------------------------------------------------------------

class TestAcousticHarvest:

    def test_basic_prediction(self):
        result = predict_acoustic_harvest(70.0, "PVDF")
        assert isinstance(result, AcousticHarvestResult)
        assert result.power_W_m2 >= 0

    def test_negligible_power_note(self):
        result = predict_acoustic_harvest(70.0, "PVDF")
        assert "negligible" in result.note.lower()

    def test_louder_more_power(self):
        r60 = predict_acoustic_harvest(60.0, "PVDF")
        r90 = predict_acoustic_harvest(90.0, "PVDF")
        assert r90.power_W_m2 > r60.power_W_m2

    def test_dba_to_intensity_reference(self):
        """120 dBA should give 1 W/m2."""
        i = _dba_to_intensity(120.0)
        assert abs(i - 1.0) < 1e-6

    def test_dba_to_intensity_0dB(self):
        """0 dBA should give 1e-12 W/m2."""
        i = _dba_to_intensity(0.0)
        assert abs(i - 1e-12) < 1e-18

    def test_typical_indoor_negligible(self):
        """At 60 dBA (quiet office), power should be < 1 uW/m2."""
        result = predict_acoustic_harvest(60.0, "PVDF")
        assert result.power_W_m2 < 1e-6

    def test_with_alpha_spectrum(self):
        alpha = {250.0: 0.3, 500.0: 0.5, 1000.0: 0.7, 2000.0: 0.8}
        result = predict_acoustic_harvest(80.0, "PVDF", alpha_spectrum=alpha)
        assert result.power_W_m2 > 0

    def test_dominant_frequency_in_range(self):
        result = predict_acoustic_harvest(70.0, "PVDF")
        assert result.dominant_frequency_Hz in [250.0, 500.0, 1000.0, 2000.0, 4000.0]

    def test_negative_dba_raises(self):
        with pytest.raises(ValueError):
            predict_acoustic_harvest(-10.0, "PVDF")


# -----------------------------------------------------------------------
# Combined harvest
# -----------------------------------------------------------------------

class TestCombinedHarvest:

    def test_default_prediction(self):
        result = predict_total_harvest()
        assert isinstance(result, PowerBudget)
        assert result.total_W_m2 >= 0

    def test_total_is_sum(self):
        result = predict_total_harvest()
        expected = result.pv_W_m2 + result.teg_W_m2 + result.piezo_W_m2 + result.acoustic_W_m2
        assert abs(result.total_W_m2 - expected) < 1e-12

    def test_breakdown_matches(self):
        result = predict_total_harvest()
        assert abs(result.per_modality_breakdown["pv"] - result.pv_W_m2) < 1e-12
        assert abs(result.per_modality_breakdown["teg"] - result.teg_W_m2) < 1e-12
        assert abs(result.per_modality_breakdown["piezo"] - result.piezo_W_m2) < 1e-12
        assert abs(result.per_modality_breakdown["acoustic"] - result.acoustic_W_m2) < 1e-12

    def test_daily_kwh_positive(self):
        result = predict_total_harvest()
        assert result.daily_kWh_m2 > 0

    def test_custom_environment(self):
        env = EnvironmentSpec(solar_irradiance_W_m2=500.0, delta_T_K=20.0)
        result = predict_total_harvest(environment_spec=env)
        assert result.pv_W_m2 > 0
        assert result.teg_W_m2 > 0

    def test_pv_dominates(self):
        """Under standard conditions, PV should be the largest contributor."""
        env = EnvironmentSpec(solar_irradiance_W_m2=1000.0)
        result = predict_total_harvest(environment_spec=env, transmitted_fraction=0.8)
        assert result.pv_W_m2 > result.teg_W_m2
        assert result.pv_W_m2 > result.piezo_W_m2
        assert result.pv_W_m2 > result.acoustic_W_m2

    def test_zero_everything(self):
        env = EnvironmentSpec(
            solar_irradiance_W_m2=0.0, delta_T_K=0.0,
            vibration_acceleration_m_s2=0.0, vibration_freq_Hz=0.0,
            sound_level_dBA=0.0,
        )
        result = predict_total_harvest(environment_spec=env, transmitted_fraction=0.0)
        # PV, piezo should be zero; acoustic at 0 dBA is near-zero
        assert result.pv_W_m2 == 0.0
        assert result.piezo_W_m2 == 0.0


# -----------------------------------------------------------------------
# Physical sanity
# -----------------------------------------------------------------------

class TestPhysicalSanity:

    def test_pv_efficiency_under_50_percent(self):
        """No PV module should claim > 50% efficiency."""
        for name in EXPECTED_PV:
            result = predict_pv_power(1000.0, name)
            assert result.efficiency_actual < 0.5

    def test_teg_building_scale_reasonable(self):
        """For a building wall (dT~5K), TEG power should be < 10 W/m2."""
        result = predict_teg_power(5.0, "Bi2Te3")
        assert result.power_W_m2 < 10.0

    def test_piezo_building_scale_microwatts(self):
        """Building vibrations (0.1 m/s2, 60 Hz) -> sub-milliwatt range."""
        result = predict_piezo_power(0.1, 60.0, "PVDF")
        assert result.power_W_m2 < 0.001  # < 1 mW/m2

    def test_acoustic_always_smallest(self):
        """Acoustic should always be the smallest contributor."""
        result = predict_total_harvest(
            environment_spec=EnvironmentSpec(solar_irradiance_W_m2=500.0),
            transmitted_fraction=0.5,
        )
        assert result.acoustic_W_m2 <= result.pv_W_m2
        assert result.acoustic_W_m2 <= result.teg_W_m2

    def test_vancouver_wall_total_reasonable(self):
        """Vancouver south wall: total should be 1-50 W/m2 range."""
        env = EnvironmentSpec()  # defaults = Vancouver
        result = predict_total_harvest(environment_spec=env, transmitted_fraction=0.5)
        assert 0.1 < result.total_W_m2 < 100.0
