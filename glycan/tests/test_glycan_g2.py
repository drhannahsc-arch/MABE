"""
glycan/tests/test_glycan_g2.py — G2 conformational entropy validation.

Tests that the physics-first TdS model produces correct ordering,
self-zeros for monosaccharides, and reasonable magnitudes.
"""
import pytest
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)


class TestG2PhysicsValues:
    """Verify TdS values from Mammen × QM flexibility factor."""

    def test_ordering_alpha16_gt_alpha14(self):
        from glycan.scorer import G2_TDS_PER_LINKAGE
        assert G2_TDS_PER_LINKAGE["alpha1-6"] > G2_TDS_PER_LINKAGE["alpha1-4"], \
            "α1→6 must have higher entropy cost (3 torsions vs 2)"

    def test_ordering_alpha14_gt_alpha23(self):
        from glycan.scorer import G2_TDS_PER_LINKAGE
        assert G2_TDS_PER_LINKAGE["alpha1-4"] > G2_TDS_PER_LINKAGE["alpha2-3"], \
            "α1→4 must be more flexible than α2→3 (carboxylate-restricted)"

    def test_beta14_range(self):
        from glycan.scorer import G2_TDS_PER_LINKAGE
        tds = G2_TDS_PER_LINKAGE["beta1-4"]
        assert 2.0 < tds < 8.0, f"β1→4 TdS={tds} outside physical range"

    def test_alpha16_has_three_torsions(self):
        """α1→6 should have much higher TdS due to omega torsion."""
        from glycan.scorer import G2_TDS_PER_LINKAGE
        assert G2_TDS_PER_LINKAGE["alpha1-6"] > 8.0, \
            "α1→6 should be > 8 kJ/mol (3 torsions)"

    def test_all_linkages_positive(self):
        from glycan.scorer import G2_TDS_PER_LINKAGE
        for name, val in G2_TDS_PER_LINKAGE.items():
            assert val > 0, f"{name}: TdS must be positive (entropy cost)"


class TestG2SelfZero:
    """Monosaccharides must get zero conformational entropy."""

    def test_monosaccharide_zero(self):
        from glycan.scorer import _compute_g2_entropy
        assert _compute_g2_entropy("Man") == 0.0
        assert _compute_g2_entropy("Glc") == 0.0
        assert _compute_g2_entropy("GlcNAc") == 0.0

    def test_unknown_ligand_zero(self):
        from glycan.scorer import _compute_g2_entropy
        assert _compute_g2_entropy("UnknownSugar") == 0.0


class TestG2Oligosaccharides:
    """Verify oligosaccharide entropy penalties."""

    def test_disaccharide_nonzero(self):
        from glycan.scorer import _compute_g2_entropy
        assert _compute_g2_entropy("(GlcNAc)2") > 0

    def test_trisaccharide_gt_disaccharide(self):
        from glycan.scorer import _compute_g2_entropy
        tri = _compute_g2_entropy("(GlcNAc)3")
        di = _compute_g2_entropy("(GlcNAc)2")
        assert tri > di, f"tri={tri} should be > di={di}"

    def test_trimannoside_includes_branch(self):
        from glycan.scorer import _compute_g2_entropy, G2_BRANCH_PENALTY
        tds = _compute_g2_entropy("triMan")
        assert tds > 15.0, f"triMan TdS={tds}, expected >15 (2 linkages + branch)"

    def test_16_linkage_most_expensive(self):
        from glycan.scorer import _compute_g2_entropy
        a16 = _compute_g2_entropy("1->6 diMan")
        a13 = _compute_g2_entropy("1->3 diMan")
        assert a16 > a13, f"1→6 ({a16}) should cost more than 1→3 ({a13})"


class TestG2ScorerIntegration:
    """Verify G2 fires correctly through the scorer pipeline."""

    def test_monosaccharide_unchanged(self):
        from glycan.scorer import GlycanScorer
        scorer = GlycanScorer()
        p = scorer.score("ConA", "Man")
        assert p.dG_conf == 0.0, "Monosaccharide must have zero G2"
        assert abs(p.residual) < 0.01, "Anchor must be exact"

    def test_oligosaccharide_has_conf(self):
        from glycan.scorer import GlycanScorer
        scorer = GlycanScorer()
        p = scorer.score("WGA", "(GlcNAc)2")
        assert p.dG_conf > 0, f"dG_conf={p.dG_conf}, expected positive"
