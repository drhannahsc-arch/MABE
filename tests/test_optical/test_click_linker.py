"""
tests/test_optical/test_click_linker.py — Module 10: Click Chemistry Linker Tests
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")

from optical.click_linker import (
    ANCHORS, SPACERS, CLICK_PAIRS,
    compute_chain, interparticle_gap, effective_diameter, effective_packing,
    recommend_attachment, functionalized_photonic_glass, compare_linkers,
    AttachmentChain,
)

_LAM = np.linspace(380, 780, 81)


class TestChainComputation:
    def test_sio2_aptes_peg4_spaac(self):
        c = compute_chain("SiO2", "APTES", "PEG4", "SPAAC")
        assert c.compatible
        assert c.L_anchor_nm == 0.8
        assert c.L_spacer_nm == 1.4
        assert c.L_click_half_nm == 0.9
        assert abs(c.L_total_per_side_nm - 3.1) < 0.01

    def test_tio2_catechol(self):
        c = compute_chain("TiO2_rutile", "catechol", "none", "SPAAC")
        assert c.compatible
        assert c.L_anchor_nm == 0.6

    def test_fe2o3_phosphonate(self):
        c = compute_chain("Fe2O3", "phosphonate", "PEG4", "SPAAC")
        assert c.compatible

    def test_carbon_diazonium(self):
        c = compute_chain("carbon", "diazonium", "PEG4", "CuAAC")
        assert c.compatible

    def test_au_thiol(self):
        c = compute_chain("Au", "thiol", "PEG4", "thiol_maleimide")
        assert c.compatible

    def test_ps_direct_azide(self):
        c = compute_chain("polystyrene", "direct_azide", "none", "SPAAC")
        assert c.compatible
        assert c.L_anchor_nm == 0.0  # no anchor needed

    def test_incompatible_silane_on_carbon(self):
        c = compute_chain("carbon", "APTES", "PEG4", "SPAAC")
        assert not c.compatible
        assert "not compatible" in c.incompatibility_reason

    def test_incompatible_thiol_on_sio2(self):
        c = compute_chain("SiO2", "thiol", "none", "SPAAC")
        assert not c.compatible

    def test_unknown_anchor(self):
        c = compute_chain("SiO2", "unknown_anchor", "PEG4", "SPAAC")
        assert not c.compatible

    def test_shell_n_reasonable(self):
        c = compute_chain("SiO2", "APTES", "PEG4", "SPAAC")
        assert 1.4 < c.n_shell_effective < 1.6


class TestGeometry:
    def test_interparticle_gap_symmetric(self):
        c = compute_chain("SiO2", "APTES", "PEG4", "SPAAC")
        gap = interparticle_gap(c)
        assert abs(gap - 2 * c.L_total_per_side_nm) < 0.01

    def test_effective_diameter_larger_than_core(self):
        c = compute_chain("SiO2", "APTES", "PEG4", "SPAAC")
        D_eff = effective_diameter(200, c)
        assert D_eff > 200

    def test_effective_packing_less_than_actual(self):
        c = compute_chain("SiO2", "APTES", "PEG4", "SPAAC")
        phi_eff = effective_packing(0.55, 200, c)
        assert phi_eff < 0.55

    def test_longer_spacer_larger_gap(self):
        c_short = compute_chain("SiO2", "APTES", "none", "SPAAC")
        c_long = compute_chain("SiO2", "APTES", "PEG12", "SPAAC")
        assert interparticle_gap(c_long) > interparticle_gap(c_short)


class TestRecommendation:
    def test_sio2_has_recommendations(self):
        recs = recommend_attachment("SiO2")
        assert len(recs) > 0
        assert all(c.compatible for c in recs)

    def test_tio2_has_recommendations(self):
        recs = recommend_attachment("TiO2_rutile")
        assert len(recs) > 0

    def test_fe2o3_has_recommendations(self):
        recs = recommend_attachment("Fe2O3")
        assert len(recs) > 0

    def test_carbon_has_recommendations(self):
        recs = recommend_attachment("carbon")
        assert len(recs) > 0

    def test_au_has_recommendations(self):
        recs = recommend_attachment("Au")
        assert len(recs) > 0

    def test_ps_has_recommendations(self):
        recs = recommend_attachment("polystyrene")
        assert len(recs) > 0

    def test_sorted_by_length(self):
        recs = recommend_attachment("SiO2")
        lengths = [c.L_total_per_side_nm for c in recs]
        assert lengths == sorted(lengths)

    def test_prefer_short(self):
        recs_all = recommend_attachment("SiO2", prefer_short=False)
        recs_short = recommend_attachment("SiO2", prefer_short=True)
        assert len(recs_short) <= len(recs_all)


class TestFunctionalizedPG:
    def test_sio2_produces_spectrum(self):
        r = functionalized_photonic_glass(200, "SiO2", "APTES", "PEG4",
                                           "SPAAC", 1.0, 0.55, _LAM)
        assert "R" in r
        assert "peak_nm" in r
        assert "peak_shift_nm" in r
        assert np.max(r["R"]) > 0

    def test_linker_redshifts_peak(self):
        """Longer linker → redshift (larger effective diameter)."""
        r_short = functionalized_photonic_glass(200, "SiO2", "APTES", "none",
                                                 "SPAAC", 1.0, 0.55, _LAM)
        r_long = functionalized_photonic_glass(200, "SiO2", "APTES", "PEG12",
                                                "SPAAC", 1.0, 0.55, _LAM)
        assert r_long["peak_nm"] >= r_short["peak_nm"]

    def test_tio2_works(self):
        r = functionalized_photonic_glass(200, "TiO2_rutile", "catechol",
                                           "PEG4", "SPAAC", 1.0, 0.55, _LAM)
        assert r["peak_nm"] > 380

    def test_fe2o3_works(self):
        r = functionalized_photonic_glass(200, "Fe2O3", "phosphonate",
                                           "PEG4", "SPAAC", 1.0, 0.55, _LAM)
        assert r["peak_nm"] > 380

    def test_carbon_works(self):
        r = functionalized_photonic_glass(200, "carbon", "pyrene",
                                           "none", "SPAAC", 1.0, 0.55, _LAM)
        assert r["peak_nm"] > 380

    def test_incompatible_raises(self):
        with pytest.raises(ValueError):
            functionalized_photonic_glass(200, "carbon", "APTES", "PEG4",
                                          "SPAAC", 1.0, 0.55, _LAM)


class TestCompareLinkers:
    def test_compare_returns_results(self):
        results = compare_linkers(200, "SiO2", wavelengths_nm=_LAM)
        assert len(results) > 0

    def test_compare_sorted_by_shift(self):
        results = compare_linkers(200, "SiO2", wavelengths_nm=_LAM)
        shifts = [abs(r["peak_shift_nm"]) for r in results]
        assert shifts == sorted(shifts)
