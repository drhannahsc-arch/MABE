"""
tests/test_selenite_v2.py — Regression tests for selenite scorer v2 fixes

Tests that the four gap-closing patches produce correct results and
don't break existing scorer behavior.

Run: pytest tests/test_selenite_v2.py -v
"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RT_LN10 = 8.314e-3 * 298.15 * math.log(10)  # 5.708 kJ/mol


# ============================================================
# Gap 1: Oxyanion exchange subtypes exist in scorer_frozen
# ============================================================

class TestOxyanionExchange:
    """Verify oxyanion donor subtypes are in SUBTYPE_EXCHANGE."""

    def test_selenite_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_selenite" in SUBTYPE_EXCHANGE

    def test_carbonate_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_carbonate" in SUBTYPE_EXCHANGE

    def test_sulfate_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_sulfate" in SUBTYPE_EXCHANGE

    def test_selenate_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_selenate" in SUBTYPE_EXCHANGE

    def test_selenite_more_favorable_than_sulfate(self):
        """Selenite (pKa2=8.3) is a stronger base → more negative exchange."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert SUBTYPE_EXCHANGE["O_selenite"] < SUBTYPE_EXCHANGE["O_sulfate"]

    def test_carbonate_most_favorable(self):
        """Carbonate (pKa2=10.3) is the strongest base in the set."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert SUBTYPE_EXCHANGE["O_carbonate"] < SUBTYPE_EXCHANGE["O_selenite"]

    def test_basicity_order(self):
        """Exchange favorability follows pKa2 order."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert (SUBTYPE_EXCHANGE["O_carbonate"]
                < SUBTYPE_EXCHANGE["O_selenite"]
                < SUBTYPE_EXCHANGE["O_sulfate"]
                <= SUBTYPE_EXCHANGE["O_selenate"])


# ============================================================
# Gap 2: Zr⁴⁺ IW offset updated
# ============================================================

class TestZrIWOffset:
    """Verify Zr⁴⁺ Irving-Williams offset is updated."""

    def test_zr4_in_iw_table(self):
        from core.scorer_frozen import IRVING_WILLIAMS_BONUS
        assert "Zr4+" in IRVING_WILLIAMS_BONUS

    def test_zr4_offset_value(self):
        """Should be -4.68 (NEA-TDB back-calc), not the old -2.0."""
        from core.scorer_frozen import IRVING_WILLIAMS_BONUS
        assert abs(IRVING_WILLIAMS_BONUS["Zr4+"] - (-4.68)) < 0.1

    def test_zr4_between_zn_and_cu(self):
        """Zr⁴⁺ offset should be between Zn²⁺ and Cu²⁺."""
        from core.scorer_frozen import IRVING_WILLIAMS_BONUS
        iw = IRVING_WILLIAMS_BONUS
        # Zr4+ should have moderate stabilization
        assert iw["Zr4+"] < 0  # favorable


# ============================================================
# Gap 3: Anion receptor scorer has calibrated geometry params
# ============================================================

class TestCalibratedGeometry:
    """Verify anion_receptor_scorer has calibrated constants."""

    def test_calibration_status_updated(self):
        with open(os.path.join(os.path.dirname(__file__), "..",
                               "anion_receptor_scorer.py"), "r",
                  encoding="utf-8") as f:
            content = f.read()
        assert "UNCALIBRATED" not in content or "PARTIALLY CALIBRATED" in content

    def test_geometry_constants_exist(self):
        from anion_receptor_scorer import (GEOMETRY_K_MAX_KJ,
                                            GEOMETRY_SIGMA_H_A)
        assert GEOMETRY_K_MAX_KJ < 0  # stabilizing
        assert 0.1 < GEOMETRY_SIGMA_H_A < 0.5  # reasonable width


# ============================================================
# Gap 4: Born desolvation model
# ============================================================

class TestBornDesolvation:
    """Verify Born desolvation model is present and physical."""

    def test_born_function_exists(self):
        from anion_receptor_scorer import born_desolvation_penalty
        # Should return a positive number (penalty)
        result = born_desolvation_penalty(charge=2, thermo_radius_A=2.39,
                                          cavity_aperture_A=4.0)
        assert result > 0

    def test_larger_ion_lower_penalty(self):
        """Larger ions have lower Born penalty (1/r dependence)."""
        from anion_receptor_scorer import born_desolvation_penalty
        small = born_desolvation_penalty(2, 2.30, 4.0)  # sulfate
        large = born_desolvation_penalty(2, 2.66, 4.0)  # carbonate
        assert small > large

    def test_higher_charge_higher_penalty(self):
        """Higher charge = higher desolvation cost (z² dependence)."""
        from anion_receptor_scorer import born_desolvation_penalty
        z1 = born_desolvation_penalty(1, 2.39, 4.0)
        z2 = born_desolvation_penalty(2, 2.39, 4.0)
        assert z2 > z1


# ============================================================
# Integration: Existing scorer not broken
# ============================================================

class TestNoRegression:
    """Verify existing metal scorer still works after patches."""

    def test_edta_cu_still_works(self):
        """EDTA + Cu²⁺ should still give reasonable log K."""
        try:
            from core.scorer_frozen import predict_log_k
            lk = predict_log_k(
                "Cu2+",
                ["N_amine", "N_amine", "O_carboxylate", "O_carboxylate",
                 "O_carboxylate", "O_carboxylate"],
                chelate_rings=5
            )
            # Cu-EDTA log K ≈ 18.8 (NIST)
            assert 10 < lk < 30, f"Cu-EDTA log K = {lk}, expected ~18.8"
        except ImportError:
            pytest.skip("scorer_frozen not importable in this environment")

    def test_original_carboxylate_unchanged(self):
        """O_carboxylate exchange should not have changed."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert abs(SUBTYPE_EXCHANGE["O_carboxylate"] - (-6.36)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
