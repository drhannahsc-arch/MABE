"""
tests/test_inorganic_surfaces.py — Inorganic Surface Adapter Tests
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inorganic_surfaces import (
    SURFACE_MATERIALS, SurfaceMaterial,
    predict_surface_binding, SurfaceBindingResult,
    recommend_surface, compare_surfaces_for_target,
)


class TestMaterialDatabase:
    def test_all_materials_valid(self):
        for key, mat in SURFACE_MATERIALS.items():
            assert isinstance(mat, SurfaceMaterial)
            assert mat.name
            assert mat.surface_area_m2_g > 0

    def test_has_silica_variants(self):
        silica_keys = [k for k in SURFACE_MATERIALS if "SiO2" in k]
        assert len(silica_keys) >= 4  # native, APTES, MPTMS, DTPA

    def test_has_iron_oxides(self):
        fe_keys = [k for k in SURFACE_MATERIALS if "Fe" in k]
        assert len(fe_keys) >= 3  # Fe2O3, Fe3O4, FeOOH

    def test_has_clays(self):
        assert "montmorillonite" in SURFACE_MATERIALS
        assert "kaolinite" in SURFACE_MATERIALS

    def test_has_zeolites(self):
        zeo_keys = [k for k in SURFACE_MATERIALS if "zeolite" in k or "clinoptilolite" in k]
        assert len(zeo_keys) >= 3

    def test_magnetite_is_magnetic(self):
        assert SURFACE_MATERIALS["Fe3O4"].magnetic

    def test_titania_is_photocatalytic(self):
        assert SURFACE_MATERIALS["TiO2"].photocatalytic


class TestPredictBinding:
    def test_pb_on_hydroxyapatite_strong(self):
        r = predict_surface_binding("Pb2+", "hydroxyapatite")
        assert r.log_K_surface > 5  # literature: very strong

    def test_hg_on_thiol_silica_very_strong(self):
        r = predict_surface_binding("Hg2+", "SiO2_MPTMS")
        assert r.log_K_surface > 8  # HSAB soft-soft

    def test_hg_on_native_silica_weak(self):
        r = predict_surface_binding("Hg2+", "SiO2")
        assert r.log_K_surface < 0  # poor match

    def test_cu_on_aptes_moderate(self):
        r = predict_surface_binding("Cu2+", "SiO2_APTES")
        assert 1 < r.log_K_surface < 8

    def test_returns_binding_result(self):
        r = predict_surface_binding("Cu2+", "Fe2O3")
        assert isinstance(r, SurfaceBindingResult)
        assert r.target == "Cu2+"
        assert r.surface_name

    def test_capacity_positive(self):
        r = predict_surface_binding("Pb2+", "SiO2_DTPA")
        assert r.capacity_mg_g > 0
        assert r.capacity_mmol_g > 0

    def test_mechanism_correct_for_clay(self):
        r = predict_surface_binding("Cs+", "montmorillonite")
        assert r.mechanism == "ion_exchange"

    def test_mechanism_correct_for_grafted(self):
        r = predict_surface_binding("Cu2+", "SiO2_APTES")
        assert r.mechanism == "grafted_chelation"

    def test_mechanism_correct_for_oxide(self):
        r = predict_surface_binding("Pb2+", "Fe2O3")
        assert r.mechanism == "inner_sphere"

    def test_unknown_surface_raises(self):
        with pytest.raises(ValueError):
            predict_surface_binding("Cu2+", "unobtanium")

    def test_selectivity_computed(self):
        r = predict_surface_binding("Pb2+", "SiO2_MPTMS",
                                     competitors=["Ca2+", "Zn2+"])
        assert "Ca2+" in r.selectivity_vs
        assert "Zn2+" in r.selectivity_vs

    def test_magnetic_flag_propagates(self):
        r = predict_surface_binding("Pb2+", "Fe3O4")
        assert r.magnetic_separation


class TestSelectivityRankings:
    """Verify that known selectivity preferences emerge from the model."""

    def test_thiol_prefers_hg_over_cu(self):
        r_hg = predict_surface_binding("Hg2+", "SiO2_MPTMS")
        r_cu = predict_surface_binding("Cu2+", "SiO2_MPTMS")
        assert r_hg.log_K_surface > r_cu.log_K_surface

    def test_hap_prefers_pb_over_zn(self):
        """Hydroxyapatite: Pb >> Zn (forms pyromorphite)."""
        r_pb = predict_surface_binding("Pb2+", "hydroxyapatite")
        r_zn = predict_surface_binding("Zn2+", "hydroxyapatite")
        assert r_pb.log_K_surface > r_zn.log_K_surface + 2

    def test_clinoptilolite_prefers_cs_over_na(self):
        r_cs = predict_surface_binding("Cs+", "clinoptilolite")
        r_na = predict_surface_binding("Na+", "clinoptilolite")
        assert r_cs.log_K_surface > r_na.log_K_surface

    def test_iron_oxide_prefers_pb_over_zn(self):
        r_pb = predict_surface_binding("Pb2+", "Fe2O3")
        r_zn = predict_surface_binding("Zn2+", "Fe2O3")
        assert r_pb.log_K_surface > r_zn.log_K_surface

    def test_phosphonate_prefers_fe3_over_zn(self):
        """Phosphonate: hard trivalent >> soft divalent."""
        r_fe = predict_surface_binding("Fe3+", "SiO2_phosphonate")
        r_zn = predict_surface_binding("Zn2+", "SiO2_phosphonate")
        assert r_fe.log_K_surface > r_zn.log_K_surface


class TestRecommendation:
    def test_pb_top_recommendation(self):
        """Hydroxyapatite or grafted silica should top the Pb list."""
        results = recommend_surface("Pb2+", pH=6.0)
        assert len(results) > 0
        top = results[0]
        assert top.log_K_surface > 3

    def test_hg_top_is_thiol(self):
        results = recommend_surface("Hg2+", pH=5.0)
        assert "thiol" in results[0].surface_name.lower()

    def test_cs_top_is_zeolite_or_clay(self):
        results = recommend_surface("Cs+", pH=7.0)
        assert any(word in results[0].surface_name.lower()
                   for word in ["zeolite", "clinoptilolite", "montmorillonite"])

    def test_magnetic_filter(self):
        results = recommend_surface("Pb2+", require_magnetic=True)
        for r in results:
            assert r.magnetic_separation

    def test_cost_filter(self):
        results = recommend_surface("Cu2+", max_cost_per_kg=10)
        for r in results:
            assert r.cost_per_kg <= 10

    def test_capacity_filter(self):
        results = recommend_surface("Pb2+", min_capacity_mg_g=50)
        for r in results:
            assert r.capacity_mg_g >= 50


class TestCostRealism:
    def test_natural_minerals_cheap(self):
        assert SURFACE_MATERIALS["clinoptilolite"].cost_usd_per_kg < 2
        assert SURFACE_MATERIALS["kaolinite"].cost_usd_per_kg < 2

    def test_mesoporous_silica_expensive(self):
        assert SURFACE_MATERIALS["MCM-41"].cost_usd_per_kg > 100

    def test_grafted_more_than_native(self):
        assert (SURFACE_MATERIALS["SiO2_APTES"].cost_usd_per_kg >
                SURFACE_MATERIALS["SiO2"].cost_usd_per_kg)
