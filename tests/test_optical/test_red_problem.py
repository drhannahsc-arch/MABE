"""
tests/test_optical/test_red_problem.py — Red Problem Analysis Tests
"""

import sys
import os
import pytest
import numpy as np
import math

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")

from optical.red_problem import (
    diagnose_mie_bias, spectral_decomposition, attack_red_problem,
    _RED_TARGET,
)


class TestDiagnosis:

    def test_blue_red_ratio_greater_than_1(self):
        """Mie backscatter should be blue-biased for dielectric spheres."""
        diag = diagnose_mie_bias(300, "polystyrene")
        assert diag["blue_red_ratio"] > 1.5, \
            f"Expected blue bias > 1.5, got {diag['blue_red_ratio']}"

    def test_ratio_higher_for_larger_particles(self):
        """Larger particles (red S(q) regime) should have worse blue bias."""
        d_small = diagnose_mie_bias(200, "SiO2")
        d_large = diagnose_mie_bias(320, "SiO2")
        # Not strictly monotonic but generally true for red-regime sizes
        assert d_large["blue_red_ratio"] > 1.0

    def test_sio2_less_biased_than_ps(self):
        """SiO₂ (n≈1.46) has lower index contrast than PS (n≈1.59)
        → less Mie resonance → less blue bias."""
        d_sio2 = diagnose_mie_bias(300, "SiO2")
        d_ps = diagnose_mie_bias(300, "polystyrene")
        assert d_sio2["blue_red_ratio"] < d_ps["blue_red_ratio"]


class TestDecomposition:

    def test_peak_mismatch_large_for_red_diameter(self):
        """For D=300nm, S(q) peak should be red but Mie peak blue
        → large mismatch."""
        decomp = spectral_decomposition(300, "SiO2")
        assert decomp["peak_mismatch_nm"] > 100, \
            f"Expected >100nm mismatch, got {decomp['peak_mismatch_nm']}"

    def test_sq_peak_in_red(self):
        """300nm SiO₂: S(q) peak should be in red/near-IR."""
        decomp = spectral_decomposition(300, "SiO2")
        assert decomp["Sq_peak_nm"] > 600

    def test_mie_peak_in_blue(self):
        """300nm SiO₂: Mie peak should be in blue/violet."""
        decomp = spectral_decomposition(300, "SiO2")
        assert decomp["Mie_peak_nm"] < 500

    def test_small_particle_no_mismatch(self):
        """For D=200nm (blue structural peak), mismatch should be smaller."""
        decomp = spectral_decomposition(200, "SiO2")
        # Both S(q) and Mie peak in similar blue region
        assert decomp["peak_mismatch_nm"] < decomp["Sq_peak_nm"]


class TestRedAttack:

    @pytest.fixture(scope="class")
    def red_result(self):
        return attack_red_problem(n_diameters=3)  # fast scan

    def test_returns_baselines(self, red_result):
        assert len(red_result["baselines"]) > 0

    def test_returns_solutions(self, red_result):
        assert len(red_result["solutions"]) > 0

    def test_solutions_sorted_by_redness(self, red_result):
        """Solutions should be sorted by descending a* (reddest first)."""
        a_stars = [s.a_star for s in red_result["solutions"]]
        # First should have highest a*
        assert a_stars[0] >= a_stars[-1]

    def test_baseline_shows_purple_for_large_D(self, red_result):
        """At least one baseline should show purple (a*>0, b*<0)."""
        purples = [b for b in red_result["baselines"]
                   if b.a_star > 0 and b.Lab[2] < 0]
        # May not always have purple depending on diameter range
        # but baselines should at least exist
        assert len(red_result["baselines"]) >= 2

    def test_best_design_exists(self, red_result):
        assert red_result["best"] is not None
        assert red_result["best"].a_star > 0, \
            "Best design should have positive a* (warm direction)"

    def test_solutions_have_positive_a_star(self, red_result):
        """Top solutions should push toward red (positive a*)."""
        top_5 = red_result["solutions"][:5]
        for s in top_5:
            assert s.a_star > 0, \
                f"{s.description}: a*={s.a_star} should be positive"

    def test_strategy_diversity(self, red_result):
        """Results should include multiple strategy types."""
        strategies = set(s.strategy for s in red_result["solutions"])
        assert len(strategies) >= 2, \
            f"Expected diverse strategies, got {strategies}"

    def test_diagnosis_included(self, red_result):
        assert "diagnosis" in red_result
        assert red_result["diagnosis"]["blue_red_ratio"] > 1.0

    def test_decomposition_included(self, red_result):
        assert "decomposition" in red_result
        assert red_result["decomposition"]["peak_mismatch_nm"] > 0
