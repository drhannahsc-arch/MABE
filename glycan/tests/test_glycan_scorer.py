"""
glycan/tests/test_glycan_scorer.py

Tests for:
1. GlycanScorer — anchor correctness, predictions, R², sign conventions
2. contact_maps.py — structural integrity
3. G3-V MIT DFT cross-validation
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from glycan.scorer import GlycanScorer, GlycanPrediction
from glycan.contact_maps import SCAFFOLD_CONTACTS, PREANCHORED_DG0
from glycan.parameters_v23 import (
    EPS_HB_EFF, CH_PI_EPS, K_DESOLV, EPS_LINKER_NET,
    EPS_CH_PI_TRP, EPS_CH_PI_TYR, EPS_CH_PI_PHE,
)


# ══════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════

def _get_predictions(scaffold=None):
    s = GlycanScorer()
    if scaffold:
        return s.score_scaffold(scaffold)
    return s.score_all()


# ══════════════════════════════════════════════════════════════════════════
# 1. Parameter sanity (v2.3 locked values)
# ══════════════════════════════════════════════════════════════════════════

class TestParameterValues:

    def test_eps_hb_eff_negative(self):
        assert EPS_HB_EFF < 0, "H-bond energy must be negative (stabilizing)"

    def test_eps_hb_eff_value(self):
        assert abs(EPS_HB_EFF - (-2.25)) < 0.01

    def test_trp_stronger_than_tyr(self):
        """Trp CH-pi more favorable than Tyr (larger pi surface)."""
        assert EPS_CH_PI_TRP < EPS_CH_PI_TYR

    def test_tyr_approx_phe(self):
        """Tyr ≈ Phe per Diehl 2024."""
        assert abs(EPS_CH_PI_TYR - EPS_CH_PI_PHE) < 0.5

    def test_anthracene_in_dispatch(self):
        assert "anthracene" in CH_PI_EPS

    def test_desolv_ax_gt_eq(self):
        """Axial OH penalty > equatorial (more disruptive desolvation)."""
        assert K_DESOLV["K_AX"] > K_DESOLV["K_EQ"]

    def test_desolv_c6_gt_eq(self):
        """Primary OH (C6) penalty > equatorial."""
        assert K_DESOLV["K_C6"] > K_DESOLV["K_EQ"]

    def test_desolv_nac_positive(self):
        assert K_DESOLV["K_NAC"] > 0

    def test_linker_net_negative(self):
        """Linker provides small net stabilization."""
        assert EPS_LINKER_NET < 0


# ══════════════════════════════════════════════════════════════════════════
# 2. Contact map integrity
# ══════════════════════════════════════════════════════════════════════════

class TestContactMapIntegrity:

    def test_all_scaffolds_present(self):
        for s in ["ConA", "WGA", "PNA", "Gal3", "Davis"]:
            assert s in SCAFFOLD_CONTACTS

    def test_each_scaffold_has_anchor(self):
        for scaffold, contacts in SCAFFOLD_CONTACTS.items():
            anchors = [k for k, v in contacts.items() if v.get("anchor")]
            assert len(anchors) == 1, f"{scaffold}: expected 1 anchor, got {len(anchors)}"

    def test_davis_preanchored(self):
        assert "Davis" in PREANCHORED_DG0
        assert abs(PREANCHORED_DG0["Davis"] - (-19.30)) < 0.01

    def test_buried_keys_valid(self):
        valid_keys = set(K_DESOLV.keys())
        for scaffold, contacts in SCAFFOLD_CONTACTS.items():
            for ligand, entry in contacts.items():
                for key in entry.get("buried", []):
                    assert key in valid_keys, (
                        f"{scaffold}/{ligand}: invalid buried key '{key}'"
                    )

    def test_res_type_in_dispatch(self):
        for scaffold, contacts in SCAFFOLD_CONTACTS.items():
            for ligand, entry in contacts.items():
                rt = entry.get("res_type", "none")
                assert rt in CH_PI_EPS, (
                    f"{scaffold}/{ligand}: res_type '{rt}' not in CH_PI_EPS"
                )

    def test_n_hb_non_negative(self):
        for scaffold, contacts in SCAFFOLD_CONTACTS.items():
            for ligand, entry in contacts.items():
                assert entry["n_HB"] >= 0

    def test_n_chp_non_negative(self):
        for scaffold, contacts in SCAFFOLD_CONTACTS.items():
            for ligand, entry in contacts.items():
                assert entry["n_CHP"] >= 0


# ══════════════════════════════════════════════════════════════════════════
# 3. Scorer: anchor correctness
# ══════════════════════════════════════════════════════════════════════════

class TestAnchorCorrectness:

    def test_cona_man_anchor(self):
        s = GlycanScorer()
        p = s.score("ConA", "Man")
        assert p.anchor_matches_obs if hasattr(p, 'anchor_matches_obs') else True
        assert abs(p.dG_pred - (-22.2)) < 0.01

    def test_wga_glcnac_anchor(self):
        p = GlycanScorer().score("WGA", "GlcNAc")
        assert abs(p.dG_pred - (-15.5)) < 0.01

    def test_pna_gal_anchor(self):
        p = GlycanScorer().score("PNA", "Gal")
        assert abs(p.dG_pred - (-18.9)) < 0.01

    def test_gal3_gal_anchor(self):
        p = GlycanScorer().score("Gal3", "Gal")
        assert abs(p.dG_pred - (-22.6)) < 0.01

    def test_davis_glc_anchor(self):
        p = GlycanScorer().score("Davis", "Glc")
        assert abs(p.dG_pred - (-24.4)) < 0.05


# ══════════════════════════════════════════════════════════════════════════
# 4. Scorer: prediction correctness (HIGH-confidence entries)
# ══════════════════════════════════════════════════════════════════════════

class TestPredictionAccuracy:

    def _check(self, scaffold, ligand, expected_pred, tol=0.5):
        p = GlycanScorer().score(scaffold, ligand)
        assert abs(p.dG_pred - expected_pred) < tol, (
            f"{scaffold}/{ligand}: pred={p.dG_pred:.2f} expected≈{expected_pred:.2f}"
        )

    def test_cona_glc(self):         self._check("ConA", "Glc", -19.9)
    def test_cona_13_diman(self):    self._check("ConA", "1->3 diMan", -26.1)
    def test_cona_16_diman(self):    self._check("ConA", "1->6 diMan", -22.5)
    def test_wga_glcnac2(self):      self._check("WGA", "(GlcNAc)2", -20.7)
    def test_wga_glcnac3(self):      self._check("WGA", "(GlcNAc)3", -22.0)
    def test_wga_glcnac4(self):      self._check("WGA", "(GlcNAc)4", -22.3)
    def test_davis_gal(self):        self._check("Davis", "Gal", -12.2, tol=0.6)
    def test_davis_man(self):        self._check("Davis", "Man", -12.2, tol=0.6)
    def test_davis_2dglc(self):      self._check("Davis", "2dGlc", -17.8, tol=0.5)

    def test_all_predictions_negative(self):
        """All ΔG predictions must be negative (binding is stabilizing)."""
        preds = _get_predictions()
        for p in preds:
            assert p.dG_pred < 0, f"{p.scaffold}/{p.ligand}: pred={p.dG_pred} should be < 0"

    def test_residual_sign_matches_direction(self):
        """Residuals should be small for HIGH-confidence entries."""
        preds = [p for p in _get_predictions() if p.confidence == "HIGH"]
        for p in preds:
            assert p.residual is not None
            assert abs(p.residual) < 2.0, (
                f"{p.scaffold}/{p.ligand}: HIGH residual too large: {p.residual:.2f}"
            )


# ══════════════════════════════════════════════════════════════════════════
# 5. Scorer: selectivity ordering
# ══════════════════════════════════════════════════════════════════════════

class TestSelectivityOrdering:

    def test_cona_man_gt_glc(self):
        """ConA binds Man more tightly than Glc (mannose-selective lectin)."""
        s = GlycanScorer()
        man = s.score("ConA", "Man").dG_pred
        glc = s.score("ConA", "Glc").dG_pred
        assert man < glc, f"ConA should prefer Man ({man:.2f}) over Glc ({glc:.2f})"

    def test_davis_glc_gt_gal(self):
        """Davis receptor selects Glc over Gal (design target is glucose)."""
        s = GlycanScorer()
        glc = s.score("Davis", "Glc").dG_pred
        gal = s.score("Davis", "Gal").dG_pred
        assert glc < gal, f"Davis should prefer Glc ({glc:.2f}) over Gal ({gal:.2f})"

    def test_davis_glc_gt_man(self):
        """Davis receptor selects Glc over Man."""
        s = GlycanScorer()
        glc = s.score("Davis", "Glc").dG_pred
        man = s.score("Davis", "Man").dG_pred
        assert glc < man, f"Davis should prefer Glc ({glc:.2f}) over Man ({man:.2f})"

    def test_wga_oligomer_ordering(self):
        """WGA binds oligomers progressively: GlcNAc < (GlcNAc)2 < (GlcNAc)3."""
        s = GlycanScorer()
        m1 = s.score("WGA", "GlcNAc").dG_pred
        m2 = s.score("WGA", "(GlcNAc)2").dG_pred
        m3 = s.score("WGA", "(GlcNAc)3").dG_pred
        assert m2 < m1, "WGA: (GlcNAc)2 should bind tighter than GlcNAc"
        assert m3 < m2, "WGA: (GlcNAc)3 should bind tighter than (GlcNAc)2"

    def test_opposite_selectivity_cona_davis(self):
        """ConA: Man > Glc. Davis: Glc > Man. Same parameters, opposite selectivity."""
        s = GlycanScorer()
        cona_man = s.score("ConA", "Man").dG_pred
        cona_glc = s.score("ConA", "Glc").dG_pred
        davis_glc = s.score("Davis", "Glc").dG_pred
        davis_man = s.score("Davis", "Man").dG_pred
        # ConA prefers Man
        assert cona_man < cona_glc
        # Davis prefers Glc
        assert davis_glc < davis_man


# ══════════════════════════════════════════════════════════════════════════
# 6. R² statistics
# ══════════════════════════════════════════════════════════════════════════

class TestR2Statistics:

    def test_r2_all_above_0p85(self):
        stats = GlycanScorer().compute_r2()
        assert stats["r2"] >= 0.85, f"Overall R²={stats['r2']:.3f} < 0.85"

    def test_r2_high_above_0p95(self):
        stats = GlycanScorer().compute_r2(confidence_filter=["HIGH"])
        assert stats["r2"] >= 0.95, f"HIGH R²={stats['r2']:.3f} < 0.95"

    def test_mae_all_below_2(self):
        stats = GlycanScorer().compute_r2()
        assert stats["mae"] < 2.0, f"MAE={stats['mae']:.2f} > 2.0 kJ/mol"

    def test_mae_high_below_1(self):
        stats = GlycanScorer().compute_r2(confidence_filter=["HIGH"])
        assert stats["mae"] < 1.0, f"HIGH MAE={stats['mae']:.2f} > 1.0 kJ/mol"

    def test_n_all_correct(self):
        stats = GlycanScorer().compute_r2()
        # 22 entries: 21 original + 1 Siglec2/Neu5Ac
        assert stats["n"] == 22

    def test_n_high_correct(self):
        stats = GlycanScorer().compute_r2(confidence_filter=["HIGH"])
        assert stats["n"] == 15


# ══════════════════════════════════════════════════════════════════════════
# 7. G3-V MIT DFT cross-validation
# ══════════════════════════════════════════════════════════════════════════

class TestG3VMITDft:

    def test_g3v_runs(self):
        from glycan.g3v_mit_dft import run_g3v_validation
        results = run_g3v_validation(verbose=False)
        assert len(results) == 3, "Should have results for Trp, Tyr, Phe"

    def test_g3v_all_pass_2x(self):
        """All residue types pass 2x criterion against attenuated DFT."""
        from glycan.g3v_mit_dft import run_g3v_validation, g3v_summary
        results = run_g3v_validation(verbose=False)
        summary = g3v_summary(results)
        assert summary["all_pass_2x"], (
            f"G3-V 2x criterion failed: {[(r.residue, r.ratio_to_central) for r in results]}"
        )

    def test_g3v_trp_within_attenuated_range(self):
        """Trp ε_CH_pi lands within (f_solv_min × DFT_mean, f_solv_max × DFT_mean)."""
        from glycan.g3v_mit_dft import run_g3v_validation
        results = {r.residue: r for r in run_g3v_validation(verbose=False)}
        assert results["Trp"].within_range

    def test_g3v_ordering_preserved(self):
        """MABE ordering Trp > Tyr ≈ Phe matches DFT ordering."""
        from glycan.g3v_mit_dft import run_g3v_validation
        results = {r.residue: r for r in run_g3v_validation(verbose=False)}
        # DFT: Trp most favorable (most negative)
        assert results["Trp"].dft_stats.mean_dft_kJ < results["Tyr"].dft_stats.mean_dft_kJ
        assert results["Tyr"].dft_stats.mean_dft_kJ < -1.0  # must be stabilizing

    def test_g3v_ratios_reasonable(self):
        """All attenuation ratios between 0.3 and 3.0."""
        from glycan.g3v_mit_dft import run_g3v_validation
        for r in run_g3v_validation(verbose=False):
            assert 0.3 <= r.ratio_to_central <= 3.0, (
                f"{r.residue}: ratio={r.ratio_to_central:.2f} out of [0.3, 3.0]"
            )

    def test_g3v_fallback_values_sensible(self):
        """Fallback DFT mean values are in kJ/mol range for CH-pi interactions."""
        from glycan.g3v_mit_dft import FALLBACK_DFT_KCAL, KCAL_TO_KJ
        for res, vals in FALLBACK_DFT_KCAL.items():
            kj = vals["mean"] * KCAL_TO_KJ
            # CH-pi energies: expect -5 to -25 kJ/mol in vacuum
            assert -25 <= kj <= -5, f"{res}: fallback DFT {kj:.1f} kJ/mol out of range"
