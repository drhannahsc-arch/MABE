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
    """Verify TdS values from CCCBDB/GLYCAM conformer analysis (v2)."""

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
        assert 1.0 < tds < 4.0, f"β1→4 TdS={tds} outside CCCBDB range (RT×ln(2) ≈ 1.7)"

    def test_alpha16_highest(self):
        """α1→6 should be highest due to extra omega torsion (3 rotamers)."""
        from glycan.scorer import G2_TDS_PER_LINKAGE
        assert G2_TDS_PER_LINKAGE["alpha1-6"] > 5.0, \
            "α1→6 should be > 5 kJ/mol (RT×ln(12) ≈ 6.2)"
        # Must be at least 1.5× higher than α1→3 (extra torsion)
        assert G2_TDS_PER_LINKAGE["alpha1-6"] > 1.5 * G2_TDS_PER_LINKAGE["alpha1-3"]

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
        # triMan: 1→3 linkage (frozen, 5 contacts) + 1→6 linkage (partial, 2 contacts) + branch
        assert tds > 4.0, f"triMan TdS={tds}, expected >4 (two linkages + branch)"
        assert tds < 8.0, f"triMan TdS={tds}, expected <8 (contact gating limits penalty)"

    def test_16_raw_tds_highest(self):
        """Raw (pre-flexibility) TdS for α1→6 must be highest."""
        from glycan.scorer import G2_TDS_PER_LINKAGE
        assert G2_TDS_PER_LINKAGE["alpha1-6"] > G2_TDS_PER_LINKAGE["alpha1-3"], \
            "Raw α1→6 TdS must exceed α1→3 (3 torsions vs 2)"

    def test_16_effective_lower_in_cona(self):
        """Effective TdS for ConA 1→6 diMan is ZERO — downstream Man has no contacts."""
        from glycan.scorer import _compute_g2_entropy
        a16 = _compute_g2_entropy("1->6 diMan")
        a13 = _compute_g2_entropy("1->3 diMan")
        assert a16 == 0.0, f"1→6 diMan: downstream Man has 0 contacts → G2 must be 0, got {a16}"
        assert a13 > 0, f"1→3 diMan should have nonzero G2"

    def test_contact_gating_zero_contacts(self):
        """Linkage with zero downstream contacts → zero G2 penalty."""
        from glycan.scorer import _compute_g2_entropy
        # (GlcNAc)4: 4th unit adds 0 contacts → same G2 as (GlcNAc)3
        g3 = _compute_g2_entropy("(GlcNAc)3")
        g4 = _compute_g2_entropy("(GlcNAc)4")
        assert g4 == g3, f"(GlcNAc)4 G2={g4} should equal (GlcNAc)3 G2={g3} (4th unit uncontacted)"


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
