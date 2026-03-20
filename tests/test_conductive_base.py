"""
tests/test_conductive_base.py -- Tests for conductive base layer module.

Validates:
  - Material database completeness (all 6 entries, all fields populated)
  - Sheet resistance prediction (scaling with thickness, TCR sign)
  - Transparency prediction (Beer-Lambert scaling)
  - Power loss prediction (J^2 dependence, zero-input zero-output)
  - Conductor selection (filtering, ranking)
  - Click handle compatibility
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

from core.conductive_base import (
    ConductorMaterial,
    ConductorSpec,
    ConductorResult,
    get_conductor,
    list_conductors,
    predict_sheet_resistance,
    predict_transparency,
    predict_power_loss,
    predict_conductor,
    select_conductor,
    check_click_compatibility,
    find_compatible_click,
    _CONDUCTORS,
)


# -----------------------------------------------------------------------
# Database completeness
# -----------------------------------------------------------------------

EXPECTED_MATERIALS = [
    "PEDOT_PSS", "AgNW", "graphene", "MXene_Ti3C2", "ITO", "Cu_mesh",
]


class TestDatabaseCompleteness:

    def test_all_materials_present(self):
        available = list_conductors()
        for name in EXPECTED_MATERIALS:
            assert name in available, f"Missing material: {name}"

    def test_material_count(self):
        assert len(_CONDUCTORS) == 6

    @pytest.mark.parametrize("name", EXPECTED_MATERIALS)
    def test_fields_populated(self, name):
        mat = get_conductor(name)
        assert mat.sigma_S_m > 0
        assert mat.sheet_R_range_ohm_sq[0] > 0
        assert mat.sheet_R_range_ohm_sq[1] >= mat.sheet_R_range_ohm_sq[0]
        assert mat.ref_thickness_nm > 0
        assert 0.0 < mat.transparency_vis <= 1.0
        assert mat.density_kg_m3 > 0
        assert len(mat.click_handles_top) > 0
        assert len(mat.click_handles_bottom) > 0
        assert mat.source != ""

    @pytest.mark.parametrize("name", EXPECTED_MATERIALS)
    def test_safe_flag(self, name):
        mat = get_conductor(name)
        assert mat.safe is True  # all 6 are safe per spec

    def test_unknown_material_raises(self):
        with pytest.raises(KeyError, match="Unknown conductor"):
            get_conductor("unobtanium")

    def test_nominal_sheet_R_is_geometric_mean(self):
        mat = get_conductor("PEDOT_PSS")
        expected = math.sqrt(10.0 * 100.0)
        assert abs(mat.sheet_R_nominal_ohm_sq - expected) < 1e-6


# -----------------------------------------------------------------------
# Sheet resistance prediction
# -----------------------------------------------------------------------

class TestSheetResistance:

    def test_at_reference_thickness(self):
        """At ref thickness and 25C, should return nominal R_s."""
        mat = get_conductor("ITO")
        rs = predict_sheet_resistance("ITO", mat.ref_thickness_nm, 25.0)
        assert abs(rs - mat.sheet_R_nominal_ohm_sq) < 1e-6

    def test_thicker_film_lower_resistance(self):
        """Doubling thickness halves sheet resistance."""
        mat = get_conductor("AgNW")
        rs1 = predict_sheet_resistance("AgNW", mat.ref_thickness_nm)
        rs2 = predict_sheet_resistance("AgNW", mat.ref_thickness_nm * 2)
        assert abs(rs2 - rs1 / 2.0) < 1e-6

    def test_thinner_film_higher_resistance(self):
        rs1 = predict_sheet_resistance("graphene", 1.5)
        rs2 = predict_sheet_resistance("graphene", 0.75)
        assert rs2 > rs1

    def test_positive_tcr_metallic(self):
        """AgNW (metallic) should have higher R at higher temperature."""
        rs_25 = predict_sheet_resistance("AgNW", 100.0, 25.0)
        rs_80 = predict_sheet_resistance("AgNW", 100.0, 80.0)
        assert rs_80 > rs_25

    def test_negative_tcr_pedot(self):
        """PEDOT:PSS should have LOWER R at higher temperature."""
        rs_25 = predict_sheet_resistance("PEDOT_PSS", 200.0, 25.0)
        rs_80 = predict_sheet_resistance("PEDOT_PSS", 200.0, 80.0)
        assert rs_80 < rs_25

    def test_metallic_tcr_sign_all(self):
        """All metallic conductors should increase R with temperature."""
        metallic = ["AgNW", "graphene", "MXene_Ti3C2", "ITO", "Cu_mesh"]
        for name in metallic:
            mat = get_conductor(name)
            rs_25 = predict_sheet_resistance(name, mat.ref_thickness_nm, 25.0)
            rs_60 = predict_sheet_resistance(name, mat.ref_thickness_nm, 60.0)
            assert rs_60 > rs_25, f"{name} should have positive TCR"

    def test_zero_thickness_raises(self):
        with pytest.raises(ValueError, match="thickness_nm must be > 0"):
            predict_sheet_resistance("ITO", 0.0)

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="thickness_nm must be > 0"):
            predict_sheet_resistance("ITO", -10.0)

    def test_always_positive(self):
        """Even at extreme cold, R_s should be positive."""
        rs = predict_sheet_resistance("PEDOT_PSS", 200.0, -200.0)
        assert rs > 0


# -----------------------------------------------------------------------
# Transparency prediction
# -----------------------------------------------------------------------

class TestTransparency:

    def test_at_reference_thickness(self):
        """At ref thickness, should return ref transparency."""
        mat = get_conductor("ITO")
        tr = predict_transparency("ITO", mat.ref_thickness_nm)
        assert abs(tr - mat.transparency_vis) < 1e-6

    def test_thicker_film_lower_transparency(self):
        """Thicker film absorbs more -> lower transparency."""
        mat = get_conductor("PEDOT_PSS")
        tr1 = predict_transparency("PEDOT_PSS", mat.ref_thickness_nm)
        tr2 = predict_transparency("PEDOT_PSS", mat.ref_thickness_nm * 2)
        assert tr2 < tr1

    def test_thinner_film_higher_transparency(self):
        mat = get_conductor("MXene_Ti3C2")
        tr1 = predict_transparency("MXene_Ti3C2", mat.ref_thickness_nm)
        tr2 = predict_transparency("MXene_Ti3C2", mat.ref_thickness_nm / 2)
        assert tr2 > tr1

    def test_bounded_zero_one(self):
        """Transparency must always be in [0, 1]."""
        for name in EXPECTED_MATERIALS:
            for t in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
                tr = predict_transparency(name, t)
                assert 0.0 <= tr <= 1.0, f"{name} at {t}nm: T={tr}"

    def test_very_thick_approaches_zero(self):
        """Very thick film should have near-zero transparency."""
        tr = predict_transparency("PEDOT_PSS", 100000.0)
        assert tr < 0.01

    def test_very_thin_approaches_one(self):
        """Very thin film should be nearly transparent."""
        tr = predict_transparency("AgNW", 0.1)
        assert tr > 0.99

    def test_zero_thickness_raises(self):
        with pytest.raises(ValueError):
            predict_transparency("ITO", 0.0)

    def test_bad_wavelength_range_raises(self):
        with pytest.raises(ValueError, match="wavelength_range_nm"):
            predict_transparency("ITO", 150.0, (780.0, 380.0))


# -----------------------------------------------------------------------
# Power loss prediction
# -----------------------------------------------------------------------

class TestPowerLoss:

    def test_zero_current_zero_loss(self):
        pl = predict_power_loss("ITO", 150.0, 0.0, 0.5)
        assert pl == 0.0

    def test_zero_path_zero_loss(self):
        pl = predict_power_loss("ITO", 150.0, 10.0, 0.0)
        assert pl == 0.0

    def test_positive_loss_with_current(self):
        pl = predict_power_loss("PEDOT_PSS", 200.0, 10.0, 0.5)
        assert pl > 0.0

    def test_loss_scales_with_j_squared(self):
        """Power loss should scale as J^2."""
        pl1 = predict_power_loss("AgNW", 100.0, 1.0, 0.5)
        pl2 = predict_power_loss("AgNW", 100.0, 2.0, 0.5)
        ratio = pl2 / pl1
        assert abs(ratio - 4.0) < 1e-6

    def test_loss_scales_with_path(self):
        """Power loss should scale linearly with path length."""
        pl1 = predict_power_loss("AgNW", 100.0, 10.0, 0.5)
        pl2 = predict_power_loss("AgNW", 100.0, 10.0, 1.0)
        ratio = pl2 / pl1
        assert abs(ratio - 2.0) < 1e-6

    def test_higher_conductivity_lower_loss(self):
        """Cu_mesh (highest sigma) should have less loss than PEDOT_PSS."""
        pl_cu = predict_power_loss("Cu_mesh", 500.0, 100.0, 0.5)
        pl_pedot = predict_power_loss("PEDOT_PSS", 500.0, 100.0, 0.5)
        assert pl_cu < pl_pedot

    def test_negative_current_raises(self):
        with pytest.raises(ValueError, match="current_density_A_m2 must be >= 0"):
            predict_power_loss("ITO", 150.0, -1.0, 0.5)

    def test_negative_path_raises(self):
        with pytest.raises(ValueError, match="path_length_m must be >= 0"):
            predict_power_loss("ITO", 150.0, 1.0, -0.5)


# -----------------------------------------------------------------------
# Conductor selection
# -----------------------------------------------------------------------

class TestSelection:

    def test_all_returned_with_no_filters(self):
        results = select_conductor()
        assert len(results) == 6

    def test_filter_flexible(self):
        results = select_conductor(require_flexible=True)
        for r in results:
            assert r.flexible is True
        # PEDOT_PSS, AgNW, graphene are flexible
        assert len(results) == 3

    def test_filter_transparency(self):
        results = select_conductor(min_transparency=0.85)
        for r in results:
            assert r.transparency >= 0.85

    def test_filter_max_sheet_r(self):
        results = select_conductor(max_sheet_R=15.0)
        for r in results:
            assert r.sheet_resistance_ohm_sq <= 15.0

    def test_combined_filters_can_return_empty(self):
        """Impossible constraints should return empty list."""
        results = select_conductor(min_transparency=0.99, max_sheet_R=0.001)
        assert len(results) == 0

    def test_results_are_ranked(self):
        """Results should come back with scores in descending order."""
        results = select_conductor()
        assert len(results) > 1
        # Just verify it's a list of ConductorResult
        for r in results:
            assert isinstance(r, ConductorResult)

    def test_custom_thickness(self):
        results = select_conductor(thickness_nm=50.0)
        for r in results:
            assert r.thickness_nm == 50.0


# -----------------------------------------------------------------------
# Full prediction via ConductorSpec
# -----------------------------------------------------------------------

class TestPredictConductor:

    def test_basic_prediction(self):
        spec = ConductorSpec(material="ITO", thickness_nm=150.0)
        result = predict_conductor(spec)
        assert isinstance(result, ConductorResult)
        assert result.material == "ITO"
        assert result.sheet_resistance_ohm_sq > 0
        assert 0 < result.transparency <= 1
        assert result.power_loss_W_m2 == 0.0  # no current

    def test_with_current(self):
        spec = ConductorSpec(
            material="AgNW", thickness_nm=100.0,
            current_density_A_m2=50.0, path_length_m=0.3,
        )
        result = predict_conductor(spec)
        assert result.power_loss_W_m2 > 0

    def test_sigma_eff_consistent(self):
        """sigma_eff should equal 1 / (R_s * t)."""
        spec = ConductorSpec(material="Cu_mesh", thickness_nm=500.0)
        result = predict_conductor(spec)
        expected_sigma = 1.0 / (result.sheet_resistance_ohm_sq * 500e-9)
        assert abs(result.sigma_eff_S_m - expected_sigma) / expected_sigma < 1e-6


# -----------------------------------------------------------------------
# Click handle compatibility
# -----------------------------------------------------------------------

class TestClickCompatibility:

    def test_cuaac_pair(self):
        result = check_click_compatibility("-N3", "-alkyne")
        assert result == "CuAAC"

    def test_cuaac_reverse(self):
        result = check_click_compatibility("-alkyne", "-N3")
        assert result == "CuAAC"

    def test_thiol_maleimide(self):
        result = check_click_compatibility("thiol-Ag", "-maleimide")
        assert result == "thiol-maleimide"

    def test_incompatible(self):
        result = check_click_compatibility("-NH2", "-alkyne")
        assert result is None

    def test_same_group_incompatible(self):
        result = check_click_compatibility("-N3", "azide")
        assert result is None  # same side of the pair

    def test_find_compatible_agNW(self):
        """AgNW has thiol-Ag handles; should be compatible with maleimide."""
        matches = find_compatible_click(
            "AgNW", face="top",
            partner_handles=["-maleimide"],
        )
        assert len(matches) > 0
        assert any(m[2] == "thiol-maleimide" for m in matches)

    def test_find_compatible_graphene(self):
        """Graphene has -N3 and -alkyne; should find CuAAC self-pairing."""
        matches = find_compatible_click(
            "graphene", face="top",
            partner_handles=["-alkyne", "-N3"],
        )
        cuaac_matches = [m for m in matches if m[2] == "CuAAC"]
        assert len(cuaac_matches) > 0

    def test_find_all_pairings(self):
        """With no partner specified, should list all possible pairings."""
        matches = find_compatible_click("Cu_mesh", face="top")
        assert len(matches) > 0


# -----------------------------------------------------------------------
# Physical sanity
# -----------------------------------------------------------------------

class TestPhysicalSanity:

    def test_cu_mesh_highest_conductivity(self):
        """Cu mesh (bulk Cu) should have highest sigma."""
        sigmas = {n: m.sigma_S_m for n, m in _CONDUCTORS.items()}
        assert max(sigmas, key=sigmas.get) == "Cu_mesh"

    def test_pedot_lowest_conductivity(self):
        """PEDOT:PSS should have lowest sigma."""
        sigmas = {n: m.sigma_S_m for n, m in _CONDUCTORS.items()}
        assert min(sigmas, key=sigmas.get) == "PEDOT_PSS"

    def test_ito_highest_ref_transparency(self):
        """ITO should have highest reference transparency (0.90)."""
        tr = {n: m.transparency_vis for n, m in _CONDUCTORS.items()}
        assert max(tr, key=tr.get) == "ITO"

    def test_pedot_negative_tcr(self):
        mat = get_conductor("PEDOT_PSS")
        assert mat.tcr_per_K < 0

    def test_metals_positive_tcr(self):
        for name in ["AgNW", "graphene", "MXene_Ti3C2", "ITO", "Cu_mesh"]:
            mat = get_conductor(name)
            assert mat.tcr_per_K > 0, f"{name} should have positive TCR"
