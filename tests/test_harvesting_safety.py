"""
tests/test_harvesting_safety.py -- Tests for harvesting material safety screening.

Validates:
  - Material database completeness (11 entries: 9 safe + 2 hard excludes)
  - Hard excludes: PZT (all), CdTe (all), MAPbI3 (textile only)
  - Safe material listing by category and application
  - screen_harvesting_design() correct flags and report
  - PVDF/P_VDF_TrFE NOT flagged as PFAS
  - Encapsulation warnings
  - Unknown material handling
  - Physical sanity
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.harvesting_safety import (
    HarvestingMaterialSafety,
    HarvestingSafetyFlag,
    HarvestingSafetyReport,
    get_harvesting_material_safety,
    list_harvesting_materials,
    list_safe_harvesting_materials,
    screen_harvesting_design,
    _HARVESTING_MATERIALS,
)


EXPECTED_MATERIALS = [
    "perovskite_MAPbI3", "perovskite_CsAgBiBr", "organic_PV",
    "Bi2Te3", "PVDF", "P_VDF_TrFE",
    "PEDOT_PSS", "AgNW", "MXene_Ti3C2",
    "PZT", "CdTe",
]


# -----------------------------------------------------------------------
# Database completeness
# -----------------------------------------------------------------------

class TestDatabase:

    def test_all_present(self):
        available = list_harvesting_materials()
        for name in EXPECTED_MATERIALS:
            assert name in available

    def test_count(self):
        assert len(_HARVESTING_MATERIALS) == 11

    @pytest.mark.parametrize("name", EXPECTED_MATERIALS)
    def test_fields_populated(self, name):
        mat = get_harvesting_material_safety(name)
        assert mat.name != ""
        assert mat.category != ""
        assert mat.key_concern != ""
        assert mat.source != ""

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_harvesting_material_safety("gallium_arsenide")


# -----------------------------------------------------------------------
# Hard excludes
# -----------------------------------------------------------------------

class TestHardExcludes:

    def test_pzt_excluded_all(self):
        mat = get_harvesting_material_safety("PZT")
        assert mat.hard_exclude_all is True
        assert mat.contains_lead is True

    def test_cdte_excluded_all(self):
        mat = get_harvesting_material_safety("CdTe")
        assert mat.hard_exclude_all is True
        assert mat.contains_cadmium is True

    def test_mapbi3_excluded_textile_only(self):
        mat = get_harvesting_material_safety("perovskite_MAPbI3")
        assert mat.hard_exclude_all is False
        assert mat.hard_exclude_textile is True
        assert mat.safe_for_building is True
        assert mat.safe_for_textile is False

    def test_mapbi3_contains_lead(self):
        mat = get_harvesting_material_safety("perovskite_MAPbI3")
        assert mat.contains_lead is True

    def test_pzt_not_in_safe_building(self):
        safe = list_safe_harvesting_materials(category="piezo", application="building")
        assert "PZT" not in safe

    def test_cdte_not_in_safe_building(self):
        safe = list_safe_harvesting_materials(category="pv_absorber", application="building")
        assert "CdTe" not in safe

    def test_mapbi3_in_safe_building(self):
        safe = list_safe_harvesting_materials(category="pv_absorber", application="building")
        assert "perovskite_MAPbI3" in safe

    def test_mapbi3_not_in_safe_textile(self):
        safe = list_safe_harvesting_materials(category="pv_absorber", application="textile")
        assert "perovskite_MAPbI3" not in safe


# -----------------------------------------------------------------------
# Safe material listing
# -----------------------------------------------------------------------

class TestSafeListing:

    def test_safe_pv_building(self):
        safe = list_safe_harvesting_materials(category="pv_absorber", application="building")
        assert "organic_PV" in safe
        assert "perovskite_CsAgBiBr" in safe
        assert "perovskite_MAPbI3" in safe

    def test_safe_piezo_building(self):
        safe = list_safe_harvesting_materials(category="piezo", application="building")
        assert "PVDF" in safe
        assert "P_VDF_TrFE" in safe
        assert "PZT" not in safe

    def test_safe_conductor_textile(self):
        safe = list_safe_harvesting_materials(category="conductor", application="textile")
        assert "PEDOT_PSS" in safe
        assert "AgNW" in safe
        assert "MXene_Ti3C2" in safe

    def test_no_category_filter(self):
        safe = list_safe_harvesting_materials(application="building")
        # Should include materials from all categories
        assert len(safe) >= 9  # all non-excluded


# -----------------------------------------------------------------------
# PVDF not PFAS
# -----------------------------------------------------------------------

class TestPVDFNotPFAS:

    def test_pvdf_safe_all(self):
        mat = get_harvesting_material_safety("PVDF")
        assert mat.safe_for_building is True
        assert mat.safe_for_textile is True
        assert mat.hard_exclude_all is False

    def test_pvdf_notes_mention_not_pfas(self):
        mat = get_harvesting_material_safety("PVDF")
        assert "NOT PFAS" in mat.notes or "not PFAS" in mat.notes.lower() or "NOT PFAS" in mat.key_concern

    def test_pvdf_trfe_same_safety(self):
        pvdf = get_harvesting_material_safety("PVDF")
        trfe = get_harvesting_material_safety("P_VDF_TrFE")
        assert pvdf.safe_for_building == trfe.safe_for_building
        assert pvdf.safe_for_textile == trfe.safe_for_textile


# -----------------------------------------------------------------------
# Screening function
# -----------------------------------------------------------------------

class TestScreening:

    def test_safe_design(self):
        report = screen_harvesting_design(
            pv_material="organic_PM6Y6",
            piezo_material="PVDF",
            conductor="PEDOT_PSS",
            application="building",
        )
        assert isinstance(report, HarvestingSafetyReport)
        assert report.safe_for_application is True
        assert len(report.exclusion_reasons) == 0

    def test_pzt_excluded(self):
        report = screen_harvesting_design(
            pv_material="organic_PM6Y6",
            piezo_material="PZT",
            conductor="PEDOT_PSS",
            application="building",
        )
        assert report.safe_for_application is False
        assert any("PZT" in r or "Lead zirconate" in r for r in report.exclusion_reasons)

    def test_mapbi3_building_ok(self):
        report = screen_harvesting_design(
            pv_material="perovskite_MAPbI3",
            application="building",
        )
        assert report.safe_for_application is True
        # Should have encapsulation warning
        assert any("encapsul" in w.lower() for w in report.warnings + report.recommendations)

    def test_mapbi3_textile_excluded(self):
        report = screen_harvesting_design(
            pv_material="perovskite_MAPbI3",
            application="textile",
        )
        assert report.safe_for_application is False

    def test_bi2te3_encapsulation_warning(self):
        report = screen_harvesting_design(
            pv_material="organic_PM6Y6",
            te_material="Bi2Te3",
            application="building",
        )
        assert report.safe_for_application is True
        assert any("Bi2Te3" in r or "bismuth" in r.lower()
                    for r in report.recommendations)

    def test_organic_pv_mapping(self):
        """organic_PM6Y6 should map to organic_PV safety key."""
        report = screen_harvesting_design(pv_material="organic_PM6Y6")
        assert "organic_PV" in report.components_assessed

    def test_unknown_material_warning(self):
        report = screen_harvesting_design(
            pv_material="magic_solar_paint",
            application="building",
        )
        assert report.safe_for_application is True  # unknown doesn't exclude, just warns
        assert any("not in safety database" in w for w in report.warnings)

    def test_summary_string(self):
        report = screen_harvesting_design(pv_material="organic_PM6Y6")
        s = report.summary()
        assert "SAFE" in s or "EXCLUDED" in s

    def test_all_components_assessed(self):
        report = screen_harvesting_design(
            pv_material="organic_PM6Y6",
            te_material="Bi2Te3",
            piezo_material="PVDF",
            conductor="PEDOT_PSS",
        )
        assert len(report.components_assessed) == 4
