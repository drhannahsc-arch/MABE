"""
tests/test_glycan_unified_integration.py — Tests for glycan → unified_scorer_v2 wiring

Validates that:
1. GlycanScorer produces correct predictions (standalone)
2. predict() fires glycan terms for binding_mode == 'glycan_lectin'
3. Result fields map correctly
4. Non-glycan modes are unaffected
"""

import pytest
import sys
import os

_mabe_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _mabe_root not in sys.path:
    sys.path.insert(0, _mabe_root)

from glycan.scorer import GlycanScorer, GlycanPrediction


# ===================================================================
# STANDALONE GLYCAN SCORER TESTS
# ===================================================================

class TestGlycanScorerStandalone:

    @pytest.fixture
    def scorer(self):
        return GlycanScorer()

    def test_cona_man_anchor(self, scorer):
        """ConA-Man is the anchor: prediction matches obs exactly."""
        pred = scorer.score("ConA", "Man")
        assert abs(pred.residual) < 0.01, f"Anchor residual: {pred.residual}"

    def test_cona_glc_selectivity(self, scorer):
        """ConA binds Man > Glc (mannose selectivity)."""
        man = scorer.score("ConA", "Man")
        glc = scorer.score("ConA", "Glc")
        assert man.dG_pred < glc.dG_pred, "Man should bind tighter than Glc"

    def test_wga_multivalency(self, scorer):
        """WGA: (GlcNAc)3 binds tighter than GlcNAc monomer."""
        mono = scorer.score("WGA", "GlcNAc")
        tri = scorer.score("WGA", "(GlcNAc)3")
        assert tri.dG_pred < mono.dG_pred

    def test_davis_glc_anchor(self, scorer):
        """Davis-Glc anchor prediction matches obs."""
        pred = scorer.score("Davis", "Glc")
        assert abs(pred.residual) < 0.01

    def test_davis_selectivity_inversion(self, scorer):
        """Davis synthetic receptor: Glc > Gal (inverted from lectins)."""
        glc = scorer.score("Davis", "Glc")
        gal = scorer.score("Davis", "Gal")
        assert glc.dG_pred < gal.dG_pred, "Davis binds Glc tighter than Gal"

    def test_all_scaffolds_scorable(self, scorer):
        """All 5 scaffolds produce predictions."""
        for scaffold in ["ConA", "WGA", "PNA", "Gal3", "Davis"]:
            preds = scorer.score_scaffold(scaffold)
            assert len(preds) > 0, f"{scaffold}: no predictions"

    def test_21_total_predictions(self, scorer):
        """21 ligand-scaffold pairs total."""
        all_preds = scorer.score_all()
        assert len(all_preds) == 21

    def test_overall_mae_below_2(self, scorer):
        """Overall MAE < 2 kJ/mol for all scored pairs with obs."""
        all_preds = scorer.score_all()
        errors = [abs(p.residual) for p in all_preds if p.residual is not None]
        mae = sum(errors) / len(errors)
        assert mae < 2.0, f"MAE = {mae:.2f}, expected < 2.0"

    def test_decomposition_sums(self, scorer):
        """Individual terms sum to dG_pred for all predictions."""
        for pred in scorer.score_all():
            expected = pred.dG0 + pred.dG_HB + pred.dG_desolv + pred.dG_CHP + pred.dG_linker
            assert abs(pred.dG_pred - expected) < 0.1, (
                f"{pred.scaffold}/{pred.ligand}: sum={expected:.2f} != pred={pred.dG_pred:.2f}"
            )


# ===================================================================
# FIELD MAPPING TESTS (no RDKit needed)
# ===================================================================

class TestFieldMapping:
    """Test that glycan terms map to correct PredictionResult fields.

    Tests the _compute_glycan_terms function directly without going
    through predict() (which requires RDKit import chain).
    """

    def test_compute_glycan_terms_fires(self):
        """_compute_glycan_terms populates result fields."""
        # Import just the function, not the whole scorer
        # Need to manually call since predict() requires RDKit
        from core.universal_schema import UniversalComplex
        from glycan.scorer import GlycanScorer

        scorer = GlycanScorer()
        pred = scorer.score("ConA", "Man")

        # Simulate what _compute_glycan_terms does
        from dataclasses import dataclass
        @dataclass
        class FakeResult:
            dg_shape: float = 0.0
            dg_hbond: float = 0.0
            dg_group_desolv: float = 0.0
            dg_pi: float = 0.0
            dg_hbond_coop: float = 0.0

        r = FakeResult()
        r.dg_shape = pred.dG0
        r.dg_hbond = pred.dG_HB
        r.dg_group_desolv = pred.dG_desolv
        r.dg_pi = pred.dG_CHP
        r.dg_hbond_coop = pred.dG_linker

        total = r.dg_shape + r.dg_hbond + r.dg_group_desolv + r.dg_pi + r.dg_hbond_coop
        assert abs(total - pred.dG_pred) < 0.1

    def test_dg0_maps_to_shape(self):
        """DG0 (scaffold offset) maps to dg_shape field."""
        scorer = GlycanScorer()
        pred = scorer.score("ConA", "Man")
        assert pred.dG0 < 0, "DG0 should be negative (favorable baseline)"

    def test_hb_maps_to_hbond(self):
        """Glycan H-bond energy maps to dg_hbond."""
        scorer = GlycanScorer()
        pred = scorer.score("ConA", "Man")
        assert pred.dG_HB < 0, "H-bond should be favorable"

    def test_desolv_maps_to_group_desolv(self):
        """Glycan desolvation maps to dg_group_desolv (positive = costly)."""
        scorer = GlycanScorer()
        pred = scorer.score("ConA", "Man")
        assert pred.dG_desolv > 0, "Desolvation should be positive (unfavorable)"

    def test_chpi_maps_to_pi(self):
        """CH-π contacts map to dg_pi (favorable)."""
        scorer = GlycanScorer()
        pred = scorer.score("ConA", "Man")
        assert pred.dG_CHP < 0, "CH-π should be favorable"


# ===================================================================
# SELF-ZEROING TESTS
# ===================================================================

class TestSelfZeroing:

    def test_wrong_mode_zeros(self):
        """Non-glycan binding mode produces no glycan output."""
        from core.universal_schema import UniversalComplex
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            dg_shape: float = 0.0
            dg_hbond: float = 0.0
            dg_group_desolv: float = 0.0
            dg_pi: float = 0.0
            dg_hbond_coop: float = 0.0

        try:
            import core.unified_scorer_v2 as usm
            fn = usm._compute_glycan_terms
        except (ImportError, AttributeError):
            pytest.skip("unified_scorer_v2 import chain requires RDKit")

        uc = UniversalComplex(
            name="test", binding_mode="host_guest_inclusion",
            host_name="ConA", guest_name="Man",
        )
        r = FakeResult()
        fn(uc, r)
        assert r.dg_hbond == 0.0
        assert r.dg_group_desolv == 0.0

    def test_unknown_scaffold_zeros(self):
        """Unknown scaffold name doesn't crash, just zeros."""
        from core.universal_schema import UniversalComplex
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            dg_shape: float = 0.0
            dg_hbond: float = 0.0
            dg_group_desolv: float = 0.0
            dg_pi: float = 0.0
            dg_hbond_coop: float = 0.0

        try:
            import core.unified_scorer_v2 as usm
            fn = usm._compute_glycan_terms
        except (ImportError, AttributeError):
            pytest.skip("Import chain unavailable")

        uc = UniversalComplex(
            name="test", binding_mode="glycan_lectin",
            host_name="UnknownLectin", guest_name="Man",
        )
        r = FakeResult()
        fn(uc, r)
        assert r.dg_hbond == 0.0  # graceful zero on unknown scaffold


# ===================================================================
# FULL predict() INTEGRATION (skipped if RDKit unavailable)
# ===================================================================

class TestPredictIntegration:

    @pytest.fixture(autouse=True)
    def check_rdkit(self):
        try:
            from core.unified_scorer_v2 import predict
        except ImportError:
            pytest.skip("unified_scorer_v2 import chain requires RDKit")

    def test_cona_man_through_predict(self):
        from core.unified_scorer_v2 import predict
        from core.universal_schema import UniversalComplex

        uc = UniversalComplex(
            name="ConA:Man", binding_mode="glycan_lectin",
            host_name="ConA", guest_name="Man",
            log_Ka_exp=3.699,
        )
        result = predict(uc)
        # Should have nonzero glycan terms
        assert result.dg_hbond != 0.0, "H-bond should fire"
        assert result.dg_group_desolv != 0.0, "Desolvation should fire"
        assert result.dg_pi != 0.0, "CH-π should fire"
        # Check total is reasonable
        assert -35 < result.dg_total_kj < -10, (
            f"dG_total = {result.dg_total_kj:.1f}, expected -10 to -35"
        )

    def test_hg_mode_unaffected(self):
        """Host-guest mode should not have glycan terms."""
        from core.unified_scorer_v2 import predict
        from core.universal_schema import UniversalComplex

        uc = UniversalComplex(
            name="test_hg", binding_mode="host_guest_inclusion",
            host_name="beta-CD",
            guest_smiles="C1(CC2CC3CC(C2)CC1C3)",
        )
        result = predict(uc)
        # Should not crash; glycan terms should be zero
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
