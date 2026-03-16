"""
tests/test_entropy_of_mixing.py — Tests for translational/rotational entropy loss
"""

import pytest
import sys
import os

_mabe_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _mabe_root not in sys.path:
    sys.path.insert(0, _mabe_root)

from knowledge.entropy_of_mixing import (
    total_mixing_entropy, mixing_entropy_quick,
    CONSENSUS_MIXING_ENTROPY, VIB_RECOVERY_TOTAL, _A, _B,
)


class TestConsensusValues:

    def test_fragment_in_range(self):
        """Fragment (MW=150) gives ~25 kJ/mol (Zhou & Gilson lower bound)."""
        r = total_mixing_entropy(150)
        assert 22.0 < r["gross_kJ"] < 28.0, f"MW=150: {r['gross_kJ']:.1f}"

    def test_drug_in_range(self):
        """Drug-like (MW=350) gives ~28 kJ/mol (consensus midpoint)."""
        r = total_mixing_entropy(350)
        assert 25.0 < r["gross_kJ"] < 32.0, f"MW=350: {r['gross_kJ']:.1f}"

    def test_large_in_range(self):
        """Large molecule (MW=800) gives ~32 kJ/mol (Zhou & Gilson upper)."""
        r = total_mixing_entropy(800)
        assert 29.0 < r["gross_kJ"] < 35.0, f"MW=800: {r['gross_kJ']:.1f}"

    def test_increases_with_mw(self):
        """Entropy cost increases with molecular weight (more DOF to lose)."""
        r150 = total_mixing_entropy(150)["gross_kJ"]
        r500 = total_mixing_entropy(500)["gross_kJ"]
        r800 = total_mixing_entropy(800)["gross_kJ"]
        assert r150 < r500 < r800

    def test_weak_mw_dependence(self):
        """MW dependence is weak: <10 kJ/mol across full drug range."""
        r150 = total_mixing_entropy(150)["gross_kJ"]
        r800 = total_mixing_entropy(800)["gross_kJ"]
        assert r800 - r150 < 10.0

    def test_all_positive(self):
        """Gross entropy cost is always positive (unfavorable)."""
        for mw in [50, 100, 200, 500, 1000]:
            r = total_mixing_entropy(mw)
            assert r["gross_kJ"] > 0

    def test_net_positive(self):
        """Net cost (after vib recovery) is still positive for drugs."""
        r = total_mixing_entropy(350)
        assert r["net_kJ"] > 0, "Net should be positive even after vib recovery"

    def test_vib_recovery_reasonable(self):
        """Vibrational recovery is 15-20 kJ/mol (6 modes × ~3 kJ)."""
        assert 12.0 < VIB_RECOVERY_TOTAL < 24.0


class TestQuickAPI:

    def test_default(self):
        """Default gives ~28 kJ/mol."""
        v = mixing_entropy_quick()
        assert 25.0 < v < 32.0

    def test_category_ordering(self):
        """Categories are ordered: fragment < drug < peptide."""
        frag = mixing_entropy_quick(category="fragment")
        drug = mixing_entropy_quick(category="drug_like")
        pep = mixing_entropy_quick(category="peptide")
        assert frag < drug < pep

    def test_mw_overrides_category(self):
        """MW-based calculation used when mw_Da provided."""
        v1 = mixing_entropy_quick(mw_Da=300)
        v2 = mixing_entropy_quick(category="drug_like")
        # Both should be close but not identical (MW=300 vs MW~350 for category)
        assert abs(v1 - v2) < 3.0

    def test_vib_recovery_flag(self):
        """With vib recovery, value is lower."""
        gross = mixing_entropy_quick(mw_Da=350, include_vib_recovery=False)
        net = mixing_entropy_quick(mw_Da=350, include_vib_recovery=True)
        assert net < gross
        assert abs(gross - net - VIB_RECOVERY_TOTAL) < 0.01


class TestDecomposition:

    def test_gross_equals_net_plus_recovery(self):
        """gross = net + vib_recovery."""
        r = total_mixing_entropy(400)
        assert abs(r["gross_kJ"] - r["net_kJ"] - r["vib_recovery_kJ"]) < 0.01

    def test_small_mw_clamped(self):
        """Very small MW is clamped, doesn't crash."""
        r = total_mixing_entropy(10)
        assert r["gross_kJ"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])