"""
Tests for glycan/pulldown_selector.py -- cell type -> pulldown design pipeline.
"""

import pytest
from glycan.pulldown_selector import (
    recommend_pulldown,
    list_cell_types,
    list_aliases,
    _resolve_cell_type,
    CELL_LECTIN_DB,
    PulldownRecommendation,
)


# ── Cell type resolution ────────────────────────────────────────────────

class TestCellTypeResolution:
    def test_canonical_key(self):
        assert _resolve_cell_type("macrophage_m2") == "macrophage_m2"

    def test_alias(self):
        assert _resolve_cell_type("macrophage") == "macrophage_m2"
        assert _resolve_cell_type("t cell") == "t_cell"
        assert _resolve_cell_type("NK") == "nk_cell"
        assert _resolve_cell_type("liver") == "hepatocyte"
        assert _resolve_cell_type("cancer") == "tumor_epithelial"

    def test_case_insensitive(self):
        assert _resolve_cell_type("Macrophage") == "macrophage_m2"
        assert _resolve_cell_type("T Cell") == "t_cell"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown cell type"):
            _resolve_cell_type("alien_cell")

    def test_list_cell_types(self):
        types = list_cell_types()
        assert "macrophage_m2" in types
        assert "t_cell" in types
        assert len(types) >= 8


# ── Macrophage M2 (full quantitative path) ──────────────────────────────

class TestMacrophageM2:
    def test_returns_recommendations(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        assert len(recs) >= 2  # CD206 + Gal3

    def test_has_scored_designs(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        scored = [r for r in recs if r.dG_pred is not None]
        assert len(scored) >= 1

    def test_has_feasible_linker(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        feasible = [r for r in recs if r.linker is not None and r.linker.feasible]
        assert len(feasible) >= 1

    def test_cd206_uses_cona_proxy(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        cd206 = [r for r in recs if "CD206" in r.lectin or "Mannose" in r.lectin]
        assert len(cd206) >= 1
        assert cd206[0].scorer_proxy == "ConA"

    def test_sorted_by_composite(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        scores = [r.composite_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_density_propagates(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        cd206 = [r for r in recs if "CD206" in r.lectin or "Mannose" in r.lectin]
        assert cd206[0].lectin_density == 50000


# ── T cell ──────────────────────────────────────────────────────────────

class TestTCell:
    def test_returns_gal3(self):
        recs = recommend_pulldown("t_cell", bead_diameter_nm=50)
        assert len(recs) >= 1
        assert any("Galectin" in r.lectin for r in recs)

    def test_gal3_uses_gal(self):
        recs = recommend_pulldown("t_cell", bead_diameter_nm=50)
        gal3 = [r for r in recs if "Galectin-3" in r.lectin]
        assert gal3[0].sugar == "Gal"

    def test_alias_works(self):
        recs_alias = recommend_pulldown("t cell", bead_diameter_nm=50)
        recs_canon = recommend_pulldown("t_cell", bead_diameter_nm=50)
        assert len(recs_alias) == len(recs_canon)


# ── Activated T cell (mixed: scored + qualitative) ──────────────────────

class TestActivatedTCell:
    def test_has_both_scored_and_qualitative(self):
        recs = recommend_pulldown("t_cell_activated", bead_diameter_nm=50)
        scored = [r for r in recs if r.dG_pred is not None]
        qualitative = [r for r in recs if r.dG_pred is None]
        assert len(scored) >= 1   # Gal3
        assert len(qualitative) >= 1  # CD62L (no proxy)

    def test_qualitative_has_low_confidence(self):
        recs = recommend_pulldown("t_cell_activated", bead_diameter_nm=50)
        qualitative = [r for r in recs if r.dG_pred is None]
        for r in qualitative:
            assert r.confidence == "LOW"


# ── Hepatocyte (ASGPR with proxy) ──────────────────────────────────────

class TestHepatocyte:
    def test_returns_asgpr(self):
        recs = recommend_pulldown("hepatocyte", bead_diameter_nm=50)
        assert any("ASGPR" in r.lectin for r in recs)

    def test_pna_proxy(self):
        recs = recommend_pulldown("hepatocyte", bead_diameter_nm=50)
        asgpr = [r for r in recs if "ASGPR" in r.lectin]
        assert asgpr[0].scorer_proxy == "PNA"
        assert asgpr[0].proxy_confidence == "MEDIUM"

    def test_high_density(self):
        recs = recommend_pulldown("hepatocyte", bead_diameter_nm=50)
        asgpr = [r for r in recs if "ASGPR" in r.lectin]
        assert asgpr[0].lectin_density == 500000


# ── B cell (qualitative only — no proxy) ────────────────────────────────

class TestBCell:
    def test_qualitative_only(self):
        recs = recommend_pulldown("b_cell", bead_diameter_nm=50)
        assert len(recs) >= 1
        assert recs[0].dG_pred is None
        assert recs[0].confidence == "LOW"

    def test_default_c1_position(self):
        recs = recommend_pulldown("b_cell", bead_diameter_nm=50)
        assert "C1" in recs[0].position


# ── NK cell ─────────────────────────────────────────────────────────────

class TestNKCell:
    def test_wga_proxy(self):
        recs = recommend_pulldown("nk_cell", bead_diameter_nm=50)
        assert len(recs) >= 1
        assert recs[0].scorer_proxy == "WGA"
        assert recs[0].sugar == "GlcNAc"


# ── Tumor epithelial ────────────────────────────────────────────────────

class TestTumor:
    def test_galectin_targets(self):
        recs = recommend_pulldown("tumor_epithelial", bead_diameter_nm=50)
        assert len(recs) >= 2
        assert all("Galectin" in r.lectin for r in recs)

    def test_alias_cancer(self):
        recs = recommend_pulldown("cancer", bead_diameter_nm=50)
        assert len(recs) >= 2


# ── Summary and composite score ─────────────────────────────────────────

class TestOutput:
    def test_summary_is_string(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        for r in recs:
            assert isinstance(r.summary, str)
            assert len(r.summary) > 10

    def test_composite_positive(self):
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        for r in recs:
            assert r.composite_score > 0

    def test_all_have_sugar_and_position(self):
        for cell_type in list_cell_types():
            recs = recommend_pulldown(cell_type, bead_diameter_nm=50)
            for r in recs:
                assert r.sugar is not None
                assert r.position is not None

    def test_all_have_source(self):
        for cell_type in list_cell_types():
            recs = recommend_pulldown(cell_type, bead_diameter_nm=50)
            for r in recs:
                assert r.source != ""


# ── Bead size propagation ───────────────────────────────────────────────

class TestBeadSize:
    def test_larger_bead_longer_peg(self):
        small = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        large = recommend_pulldown("macrophage_m2", bead_diameter_nm=500)
        # Find matching entries (same lectin + position)
        for s in small:
            if s.linker and s.linker.feasible:
                for l in large:
                    if l.lectin == s.lectin and l.position == s.position and l.linker:
                        assert l.linker.peg_n_recommended > s.linker.peg_n_recommended
                        break
