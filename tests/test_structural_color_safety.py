"""
tests/test_structural_color_safety.py — Tests for structural color component safety.

Validates:
  - Material database completeness (all optical pipeline materials covered)
  - IARC Group 1 carcinogens excluded from all applications
  - Azo dyes and anthracyclines excluded
  - Cadmium and lead pigments excluded
  - SiO₂ + melanin + silicone = safest combination (passes everything)
  - ZnS excluded for textile (acid dissolution in sweat)
  - BaTiO₃ excluded for textile (Ba²⁺ leaching)
  - TiO₂ 2B warning but allowed when embedded
  - Carbon black 2B allowed when embedded
  - Application-specific exposure route assignment
  - Design screening produces actionable report
  - Alternatives suggested for excluded components
  - Safe material lists correct per application
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.structural_color_safety import (
    screen_design,
    get_material_safety,
    list_safe_particles,
    list_safe_absorbers,
    MaterialSafety,
    IARCGroup,
    ExposureRoute,
    StructuralColorSafetyReport,
    _MATERIALS,
    _APPLICATION_ROUTES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Material database
# ═══════════════════════════════════════════════════════════════════════════

class TestMaterialDatabase:

    def test_optical_pipeline_materials_covered(self):
        """All materials used in colloidal_thermal should have safety profiles."""
        required = ["SiO2", "TiO2_rutile", "TiO2_anatase", "ZnS", "BaTiO3",
                     "polystyrene", "PMMA", "melanin", "carbon_black", "CNC"]
        for mat_id in required:
            assert mat_id in _MATERIALS, f"Missing safety profile for {mat_id}"

    def test_absorbers_covered(self):
        absorbers = [m for m in _MATERIALS.values() if m.category == "absorber"]
        assert len(absorbers) >= 3  # carbon_black, melanin, iron_oxide

    def test_dangerous_chromophores_covered(self):
        chromophores = [m for m in _MATERIALS.values() if m.category == "chromophore"]
        assert len(chromophores) >= 3  # azo, anthracycline, cadmium, lead

    def test_all_have_iarc(self):
        for mid, mat in _MATERIALS.items():
            assert isinstance(mat.iarc_group, IARCGroup), f"{mid} missing IARC"

    def test_all_have_source(self):
        for mid, mat in _MATERIALS.items():
            assert mat.source != "" or mat.category == "matrix", f"{mid} missing source"


# ═══════════════════════════════════════════════════════════════════════════
# Hard exclusions
# ═══════════════════════════════════════════════════════════════════════════

class TestHardExclusions:

    def test_cadmium_pigment_group_1(self):
        mat = _MATERIALS["cadmium_pigment"]
        assert mat.iarc_group == IARCGroup.GROUP_1
        assert not mat.safe_for_building
        assert not mat.safe_for_textile

    def test_lead_pigment_group_1(self):
        mat = _MATERIALS["lead_pigment"]
        assert mat.iarc_group == IARCGroup.GROUP_1

    def test_azo_dye_excluded(self):
        mat = _MATERIALS["azo_dye"]
        assert mat.iarc_group in (IARCGroup.GROUP_1, IARCGroup.GROUP_2A)
        assert not mat.safe_for_textile

    def test_anthracycline_excluded(self):
        mat = _MATERIALS["anthracycline"]
        assert mat.ld50_oral_mg_kg is not None
        assert mat.ld50_oral_mg_kg < 50  # extremely toxic
        assert "ALL" in mat.excluded_applications

    def test_cadmium_screen_rejected(self):
        report = screen_design("SiO2", "cadmium_pigment", "silicone", "building_panel")
        assert not report.safe_for_application

    def test_azo_screen_rejected(self):
        report = screen_design("SiO2", "azo_dye", "silicone", "textile_coating")
        assert not report.safe_for_application

    def test_anthracycline_screen_rejected(self):
        report = screen_design("SiO2", "anthracycline", "silicone", "wall_tile")
        assert not report.safe_for_application


# ═══════════════════════════════════════════════════════════════════════════
# Safe combinations
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeCombinations:

    def test_sio2_melanin_silicone_building(self):
        """Gold standard: SiO₂ + melanin + silicone → safe for everything."""
        report = screen_design("SiO2", "melanin", "silicone", "building_panel")
        assert report.safe_for_application
        assert report.safety_score >= 0.7

    def test_sio2_melanin_pu_textile(self):
        """SiO₂ + melanin + polyurethane → safe for textile."""
        report = screen_design("SiO2", "melanin", "polyurethane", "textile_coating")
        assert report.safe_for_application
        assert report.safety_score >= 0.7

    def test_sio2_carbon_black_silicone_building(self):
        """SiO₂ + carbon black + silicone → safe for building (2B embedded)."""
        report = screen_design("SiO2", "carbon_black", "silicone", "building_panel")
        assert report.safe_for_application

    def test_pmma_melanin_textile(self):
        """PMMA + melanin → bioinert combo, safe for textile."""
        report = screen_design("PMMA", "melanin", "polyurethane", "textile_coating")
        assert report.safe_for_application

    def test_iron_oxide_absorber_safe(self):
        """Iron oxide absorber: FDA-approved, safe everywhere."""
        report = screen_design("SiO2", "iron_oxide", "silicone", "textile_coating")
        assert report.safe_for_application


# ═══════════════════════════════════════════════════════════════════════════
# Application-specific restrictions
# ═══════════════════════════════════════════════════════════════════════════

class TestApplicationRestrictions:

    def test_zns_excluded_textile(self):
        """ZnS dissolves in sweat → excluded for textile."""
        report = screen_design("ZnS", "melanin", "polyurethane", "textile_coating")
        assert not report.safe_for_application
        assert any("Zinc sulfide" in r or "ZnS" in r or "textile" in r.lower()
                    for r in report.exclusion_reasons)

    def test_batio3_excluded_textile(self):
        """BaTiO₃ releases Ba²⁺ in acid → excluded for textile."""
        report = screen_design("BaTiO3", "melanin", "silicone", "textile_coating")
        assert not report.safe_for_application

    def test_epoxy_excluded_textile(self):
        """Epoxy: BPA concern → excluded for textile."""
        report = screen_design("SiO2", "melanin", "epoxy", "textile_coating")
        assert not report.safe_for_application

    def test_zns_ok_building(self):
        """ZnS is fine for building (no sweat contact)."""
        report = screen_design("ZnS", "melanin", "silicone", "building_panel")
        assert report.safe_for_application

    def test_batio3_ok_building(self):
        """BaTiO₃ is fine for building (sealed in matrix)."""
        report = screen_design("BaTiO3", "melanin", "silicone", "facade_panel")
        assert report.safe_for_application


# ═══════════════════════════════════════════════════════════════════════════
# Nano-specific hazards
# ═══════════════════════════════════════════════════════════════════════════

class TestNanoHazards:

    def test_tio2_2b_warning_not_exclusion(self):
        """TiO₂ is 2B — warning for inhalation, not excluded when embedded."""
        report = screen_design("TiO2_rutile", "melanin", "silicone", "building_panel")
        assert report.safe_for_application
        # Should have a warning about 2B
        assert any("2B" in w or "IARC" in w for w in report.warnings)

    def test_tio2_anatase_ros_warning_textile(self):
        """TiO₂ anatase + textile → ROS warning."""
        report = screen_design("TiO2_anatase", "melanin", "polyurethane", "textile_coating")
        flags = [f for f in report.component_flags if "ROS" in f.description]
        assert len(flags) > 0

    def test_melanin_no_nano_hazards(self):
        """Melanin/PDA: no nano hazards whatsoever."""
        mat = _MATERIALS["melanin"]
        assert not mat.nano_lung_hazard
        assert not mat.nano_dissolution
        assert not mat.nano_ros_generation


# ═══════════════════════════════════════════════════════════════════════════
# Report structure
# ═══════════════════════════════════════════════════════════════════════════

class TestReportStructure:

    def test_report_has_exposure_routes(self):
        report = screen_design("SiO2", "melanin", "silicone", "textile_coating")
        assert ExposureRoute.DERMAL in report.exposure_routes

    def test_report_has_components(self):
        report = screen_design("SiO2", "melanin", "silicone", "building_panel")
        assert "SiO2" in report.components_assessed
        assert "melanin" in report.components_assessed
        assert "silicone" in report.components_assessed

    def test_report_summary(self):
        report = screen_design("SiO2", "melanin", "silicone", "building_panel")
        s = report.summary()
        assert "SAFE" in s or "CAUTION" in s or "EXCLUDED" in s

    def test_excluded_report_has_alternatives(self):
        report = screen_design("SiO2", "azo_dye", "silicone", "textile_coating")
        assert len(report.alternatives) > 0
        assert any("melanin" in a.lower() for a in report.alternatives)

    def test_safety_score_bounded(self):
        report = screen_design("SiO2", "melanin", "silicone", "building_panel")
        assert 0.0 <= report.safety_score <= 1.0

    def test_unknown_material_warned(self):
        report = screen_design("SiO2", "mystery_absorber_xyz", "silicone", "building_panel")
        assert any("mystery_absorber_xyz" in w for w in report.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# Safe material lists
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeMaterialLists:

    def test_safe_particles_building(self):
        safe = list_safe_particles("building_panel")
        assert "SiO2" in safe
        assert "TiO2_rutile" in safe
        assert "PMMA" in safe

    def test_safe_particles_textile(self):
        safe = list_safe_particles("textile_coating")
        assert "SiO2" in safe
        assert "PMMA" in safe
        assert "ZnS" not in safe  # sweat dissolution
        assert "BaTiO3" not in safe  # Ba²⁺ leaching

    def test_safe_absorbers_building(self):
        safe = list_safe_absorbers("building_panel")
        assert "melanin" in safe
        assert "carbon_black" in safe
        assert "iron_oxide" in safe

    def test_safe_absorbers_textile(self):
        safe = list_safe_absorbers("textile_coating")
        assert "melanin" in safe
        assert "iron_oxide" in safe

    def test_no_chromophores_in_safe_lists(self):
        """Toxic chromophores should never appear in safe lists."""
        for app in ["building_panel", "textile_coating"]:
            particles = list_safe_particles(app)
            absorbers = list_safe_absorbers(app)
            assert "cadmium_pigment" not in particles
            assert "azo_dye" not in absorbers
            assert "anthracycline" not in absorbers
            assert "lead_pigment" not in absorbers


# ═══════════════════════════════════════════════════════════════════════════
# The doxorubicin test
# ═══════════════════════════════════════════════════════════════════════════

class TestDoxorubicinPrinciple:

    def test_beautiful_red_still_toxic(self):
        """The module's raison d'être: a beautiful color can be lethal.
        Anthracycline is brilliant red. It also kills cells."""
        mat = _MATERIALS["anthracycline"]
        assert mat.iarc_group == IARCGroup.GROUP_2A
        assert mat.ld50_oral_mg_kg < 50
        assert mat.ec50_aquatic_mg_L < 1.0
        assert "ALL" in mat.excluded_applications

    def test_structural_color_is_the_answer(self):
        """The safe alternative: structural color from geometry, not chemistry.
        SiO₂ + melanin: both are found in nature, both are non-toxic,
        and together they produce any structural color in the visible spectrum."""
        sio2 = _MATERIALS["SiO2"]
        mel = _MATERIALS["melanin"]
        assert sio2.iarc_group == IARCGroup.GROUP_3
        assert mel.iarc_group == IARCGroup.NOT_CLASSIFIED
        assert sio2.safe_for_textile
        assert mel.safe_for_textile
        assert mel.safe_for_cosmetic
        report = screen_design("SiO2", "melanin", "silicone", "textile_coating")
        assert report.safe_for_application
        assert report.safety_score >= 0.7
