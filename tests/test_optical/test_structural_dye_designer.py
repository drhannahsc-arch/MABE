"""
tests/test_optical/test_structural_dye_designer.py — Inverse Design Tests
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")
pytest.importorskip("scipy")

from optical.structural_dye_designer import (
    design_color_on_substrate, design_named_color, print_designs,
    DesignCandidate, _evaluate_design, _scan_design_space,
    _COLOR_TARGETS,
)


class TestForwardEvaluation:

    def test_bare_returns_result(self):
        result = _evaluate_design(225, "SiO2", [], "textile_black",
                                   "few_layer", 3, 0.80)
        assert result is not None
        Lab, peak = result
        assert len(Lab) == 3
        assert 380 <= peak <= 780

    def test_single_shell_returns_result(self):
        result = _evaluate_design(225, "SiO2", [("CuPc", 2.0)],
                                   "glass", "monolayer", 1, 0.70)
        assert result is not None

    def test_two_shell_returns_result(self):
        result = _evaluate_design(
            225, "SiO2",
            [("porous_silica_30", 3.0), ("CuPc", 2.0)],
            "textile_black", "thick_film", 10, 0.55,
        )
        assert result is not None

    def test_different_diameters_different_color(self):
        r1 = _evaluate_design(150, "SiO2", [], "textile_black",
                               "thick_film", 10, 0.55)
        r2 = _evaluate_design(280, "SiO2", [], "textile_black",
                               "thick_film", 10, 0.55)
        assert r1 is not None and r2 is not None
        # Different diameters → different peak
        assert r1[1] != r2[1]


class TestScanPhase:

    def test_scan_returns_sorted_candidates(self):
        candidates = _scan_design_space(
            (55, -50, 30), "SiO2", "textile_black",
            "thick_film", 10, 0.55,
            n_diameters=3,
        )
        assert len(candidates) > 10
        # Sorted by ΔE ascending
        deltas = [c[0] for c in candidates]
        assert deltas == sorted(deltas)

    def test_scan_includes_bare(self):
        candidates = _scan_design_space(
            (50, 0, 0), "SiO2", "textile_black",
            "few_layer", 3, 0.80, n_diameters=3,
        )
        bare_count = sum(1 for _, c in candidates if not c.shells)
        assert bare_count > 0, "Scan should include bare particles"

    def test_scan_includes_two_shell(self):
        candidates = _scan_design_space(
            (50, 0, 0), "SiO2", "textile_black",
            "few_layer", 3, 0.80, n_diameters=3,
        )
        two_shell = sum(1 for _, c in candidates if len(c.shells) == 2)
        assert two_shell > 0, "Scan should include two-shell combos"


class TestFullDesign:

    def test_design_returns_result(self):
        result = design_color_on_substrate(
            target_Lab=(50, -30, 20),
            substrate="textile_black",
            regime="thick_film", n_layers=10,
            n_top_refine=2, max_refine_iter=20,
            verbose=False,
        )
        assert "designs" in result
        assert len(result["designs"]) > 0
        assert result["designs"][0].delta_E < 999

    def test_designs_sorted_by_delta_e(self):
        result = design_color_on_substrate(
            target_Lab=(50, -30, 20),
            substrate="textile_black",
            regime="thick_film", n_layers=10,
            n_top_refine=3, max_refine_iter=20,
            verbose=False,
        )
        deltas = [d.delta_E for d in result["designs"]]
        assert deltas == sorted(deltas)

    def test_named_color_green(self):
        result = design_named_color(
            "green", substrate="textile_black",
            regime="thick_film", n_layers=10,
            n_top_refine=2, max_refine_iter=20,
            verbose=False,
        )
        assert len(result["designs"]) > 0
        # Best design should have negative a* (green direction)
        best = result["designs"][0]
        assert best.Lab[1] < 0, f"Green target: a* should be negative, got {best.Lab[1]}"

    def test_named_color_blue(self):
        result = design_named_color(
            "blue", substrate="glass",
            regime="few_layer", n_layers=3,
            n_top_refine=2, max_refine_iter=20,
            verbose=False,
        )
        best = result["designs"][0]
        assert best.Lab[2] < 0, f"Blue target: b* should be negative, got {best.Lab[2]}"

    def test_design_includes_timing(self):
        result = design_color_on_substrate(
            target_Lab=(50, 0, 0),
            substrate="textile_black",
            n_top_refine=1, max_refine_iter=10,
            verbose=False,
        )
        assert result["scan_time_s"] > 0
        assert result["total_time_s"] > 0

    def test_all_named_colors_exist(self):
        expected = ["red", "green", "blue", "yellow", "orange",
                    "purple", "teal", "gold"]
        for name in expected:
            assert name in _COLOR_TARGETS, f"Missing named color: {name}"


class TestDesignCandidate:

    def test_shell_description_bare(self):
        c = DesignCandidate("SiO2", 225, [], [], "glass", "monolayer", 1, 0.8)
        assert c.shell_description() == "bare"

    def test_shell_description_single(self):
        c = DesignCandidate("SiO2", 225, [("CuPc", 2.0)], ["SPAAC"],
                            "glass", "monolayer", 1, 0.8)
        assert "CuPc" in c.shell_description()

    def test_shell_description_multi(self):
        c = DesignCandidate("SiO2", 225,
                            [("porous_silica_30", 5.0), ("CuPc", 2.0)],
                            ["SPAAC", "thiol_maleimide"],
                            "glass", "few_layer", 3, 0.8)
        desc = c.shell_description()
        assert "→" in desc
        assert "porous_silica_30" in desc
        assert "CuPc" in desc
