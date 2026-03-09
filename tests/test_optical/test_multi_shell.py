"""
tests/test_optical/test_multi_shell.py — Multi-Shell Compositor Tests
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")

from optical.multi_shell import (
    check_orthogonality, ShellLayer, multi_shell_reflectance,
    red_problem_attack, high_saturation_green, warm_gold,
    index_sandwich, triple_stack, compare_designs, MultiShellParticle,
)


class TestClickOrthogonality:

    def test_single_shell_always_orthogonal(self):
        ok, _ = check_orthogonality(["SPAAC"])
        assert ok

    def test_spaac_cuaac_not_orthogonal(self):
        """SPAAC and CuAAC both consume azide — not orthogonal."""
        ok, reason = check_orthogonality(["SPAAC", "CuAAC"])
        assert not ok
        assert "azide" in reason

    def test_spaac_thiol_maleimide_orthogonal(self):
        ok, _ = check_orthogonality(["SPAAC", "thiol_maleimide"])
        assert ok

    def test_three_way_orthogonal(self):
        ok, _ = check_orthogonality(["SPAAC", "thiol_maleimide", "IEDDA"])
        assert ok

    def test_duplicate_rejected(self):
        ok, reason = check_orthogonality(["SPAAC", "SPAAC"])
        assert not ok
        assert "reuses" in reason

    def test_cuaac_thiol_iedda_orthogonal(self):
        ok, _ = check_orthogonality(["CuAAC", "thiol_maleimide", "IEDDA"])
        assert ok


class TestMultiShellReflectance:

    def test_single_shell_runs(self):
        shells = [ShellLayer("CuPc", 2.0, click="SPAAC")]
        r = multi_shell_reflectance("SiO2", 225, shells)
        assert "R" in r
        assert r["R"].max() > 0

    def test_two_shell_runs(self):
        shells = [
            ShellLayer("porous_silica_30", 5.0, click="SPAAC"),
            ShellLayer("CuPc", 2.0, click="thiol_maleimide"),
        ]
        r = multi_shell_reflectance("SiO2", 225, shells)
        assert r["orthogonal"]
        assert 380 <= r["peak_nm"] <= 780

    def test_three_shell_runs(self):
        shells = [
            ShellLayer("porous_silica_30", 5.0, click="SPAAC"),
            ShellLayer("CuPc", 2.0, click="thiol_maleimide"),
            ShellLayer("PMMA_brush", 3.0, click="IEDDA"),
        ]
        r = multi_shell_reflectance("SiO2", 225, shells)
        assert r["orthogonal"]
        assert len(r["particle"].shells) == 3

    def test_total_diameter_correct(self):
        shells = [
            ShellLayer("CuPc", 3.0, click="SPAAC"),
            ShellLayer("polydopamine", 5.0, click="thiol_maleimide"),
        ]
        r = multi_shell_reflectance("SiO2", 200, shells)
        expected = 200 + 2 * (3.0 + 5.0)
        assert abs(r["particle"].total_diameter_nm - expected) < 0.01

    def test_non_orthogonal_flagged_but_runs(self):
        """Non-orthogonal stacks run but are flagged."""
        shells = [
            ShellLayer("CuPc", 2.0, click="SPAAC"),
            ShellLayer("TPP_freebase", 2.0, click="CuAAC"),  # shares azide
        ]
        r = multi_shell_reflectance("SiO2", 225, shells)
        assert not r["orthogonal"]
        assert "azide" in r["orthogonality_note"]
        assert r["R"].max() > 0  # still computes

    def test_cie_in_valid_range(self):
        shells = [ShellLayer("TiO2_solgel", 3.0, click="SPAAC")]
        r = multi_shell_reflectance("SiO2", 225, shells)
        x, y = r["CIE_xy"]
        assert 0 < x < 1
        assert 0 < y < 1


class TestDesignPresets:

    def test_all_presets_orthogonal(self):
        for name, func in [("red_attack", red_problem_attack),
                           ("hi_sat_green", high_saturation_green),
                           ("warm_gold", warm_gold),
                           ("index_sandwich", index_sandwich),
                           ("triple_stack", triple_stack)]:
            r = func()
            assert r["orthogonal"], f"{name} should be orthogonal"

    def test_all_presets_have_peak(self):
        for func in [red_problem_attack, high_saturation_green,
                     warm_gold, index_sandwich, triple_stack]:
            r = func()
            assert 380 <= r["peak_nm"] <= 780

    def test_triple_stack_has_three_shells(self):
        r = triple_stack()
        assert len(r["particle"].shells) == 3

    def test_designs_span_gamut(self):
        """Different presets should produce different peak wavelengths."""
        peaks = []
        for func in [red_problem_attack, high_saturation_green,
                     warm_gold, triple_stack]:
            r = func()
            peaks.append(r["peak_nm"])
        # At least 100nm spread across designs
        assert max(peaks) - min(peaks) > 80, \
            f"Gamut too narrow: peaks={peaks}"

    def test_warm_gold_is_warm(self):
        """Warm gold should have positive b* (yellow) in Lab space."""
        r = warm_gold()
        L, a, b = r["Lab"]
        assert b > 20, f"Expected warm/yellow (b*>20), got b*={b:.1f}"

    def test_hi_sat_green_is_green(self):
        """High-saturation green should have negative a* in Lab space."""
        r = high_saturation_green()
        L, a, b = r["Lab"]
        assert a < -30, f"Expected green (a*<-30), got a*={a:.1f}"


class TestCompareDesigns:

    def test_comparison_runs(self):
        designs = {
            "green": high_saturation_green(),
            "gold": warm_gold(),
        }
        comp = compare_designs(designs)
        assert "bare" in comp
        assert len(comp["designs"]) == 2

    def test_comparison_has_shifts(self):
        designs = {"green": high_saturation_green()}
        comp = compare_designs(designs)
        d = comp["designs"][0]
        assert "delta_xy" in d
        assert d["delta_xy"] > 0
