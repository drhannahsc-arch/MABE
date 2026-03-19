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
from glycan.bead_linker_design import is_oligosaccharide


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
    def test_scored_with_siglec2_proxy(self):
        """B cell now has Siglec2 proxy — should be scored, not qualitative."""
        recs = recommend_pulldown("b_cell", bead_diameter_nm=50)
        assert len(recs) >= 1
        assert recs[0].dG_pred is not None
        assert recs[0].scorer_proxy == "Siglec2"
        assert recs[0].confidence == "MEDIUM"

    def test_neu5ac_sugar(self):
        """B cell should recommend Neu5Ac."""
        recs = recommend_pulldown("b_cell", bead_diameter_nm=50)
        assert recs[0].sugar == "Neu5Ac"

    def test_has_feasible_linker(self):
        """B cell Neu5Ac should have feasible linker design (C8 or C9)."""
        recs = recommend_pulldown("b_cell", bead_diameter_nm=50)
        feasible = [r for r in recs if r.linker and r.linker.feasible]
        assert len(feasible) >= 1


# ── NK cell ─────────────────────────────────────────────────────────────

class TestNKCell:
    def test_wga_proxy(self):
        recs = recommend_pulldown("nk_cell", bead_diameter_nm=50)
        assert len(recs) >= 1
        assert recs[0].scorer_proxy == "WGA"
        # Now recommends best binder (oligosaccharide) over monomer
        assert recs[0].sugar in ("GlcNAc", "(GlcNAc)2", "(GlcNAc)3", "(GlcNAc)4")

    def test_oligo_ranked_above_mono(self):
        """Oligosaccharide should score higher than monomer for WGA."""
        recs = recommend_pulldown("nk_cell", bead_diameter_nm=50)
        oligo = [r for r in recs if is_oligosaccharide(r.sugar)]
        mono = [r for r in recs if not is_oligosaccharide(r.sugar) and r.dG_pred is not None]
        if oligo and mono:
            assert oligo[0].composite_score >= mono[0].composite_score


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


# ── Oligosaccharide-specific tests ──────────────────────────────────────

class TestOligosaccharideDetection:
    def test_mono_not_oligo(self):
        for name in ["Man", "Glc", "Gal", "GlcNAc", "GalNAc", "Fru", "2dGlc"]:
            assert not is_oligosaccharide(name), f"{name} should not be oligo"

    def test_oligo_detected(self):
        for name in ["1->2 diMan", "1->3 diMan", "(GlcNAc)2", "(GlcNAc)3",
                      "triMan", "LacNAc", "1->6 diMan"]:
            assert is_oligosaccharide(name), f"{name} should be oligo"


class TestOligoRecommendations:
    def test_cona_recommends_oligo_over_mono(self):
        """ConA: 1->2 diMan (-26.5) should rank above Man (-22.2)."""
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        cona_recs = [r for r in recs if r.scorer_proxy == "ConA"]
        if len(cona_recs) >= 2:
            # Top ConA rec should be an oligo
            top = cona_recs[0]
            assert is_oligosaccharide(top.sugar), \
                f"Top ConA rec should be oligo, got {top.sugar}"

    def test_oligo_uses_reducing_end(self):
        """Oligosaccharide recommendations use C1_reducing position."""
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        oligo_recs = [r for r in recs if is_oligosaccharide(r.sugar)]
        for r in oligo_recs:
            assert r.position == "C1_reducing", \
                f"Oligo {r.sugar} should use C1_reducing, got {r.position}"

    def test_oligo_has_feasible_linker(self):
        """Reducing-end attachment should always be feasible for reasonable beads."""
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        oligo_recs = [r for r in recs if is_oligosaccharide(r.sugar)]
        for r in oligo_recs:
            assert r.linker is not None
            assert r.linker.feasible, f"Oligo {r.sugar} linker should be feasible"
            assert r.linker.exit_class == "axial_out"

    def test_oligo_has_stronger_dg(self):
        """Best oligo should have more negative dG than best mono for same lectin."""
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        cona_recs = [r for r in recs if r.scorer_proxy == "ConA" and r.dG_pred is not None]
        oligo = [r for r in cona_recs if is_oligosaccharide(r.sugar)]
        mono = [r for r in cona_recs if not is_oligosaccharide(r.sugar)]
        if oligo and mono:
            assert oligo[0].dG_pred < mono[0].dG_pred  # more negative = stronger

    def test_wga_includes_chitooligomers(self):
        """WGA/NK cell should include (GlcNAc)n oligomers."""
        recs = recommend_pulldown("nk_cell", bead_diameter_nm=50)
        oligos = [r for r in recs if is_oligosaccharide(r.sugar)]
        assert len(oligos) >= 1
        assert any("GlcNAc" in r.sugar for r in oligos)

    def test_multiple_ligands_per_lectin(self):
        """Should return multiple ligands per lectin (up to max_ligands_per_lectin)."""
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50, max_ligands_per_lectin=3)
        cona_sugars = {r.sugar for r in recs if r.scorer_proxy == "ConA"}
        assert len(cona_sugars) >= 2  # at least one oligo + one mono

    def test_reducing_end_note_in_summary(self):
        """Oligo summaries should mention reducing end."""
        recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
        oligo_recs = [r for r in recs if is_oligosaccharide(r.sugar)]
        for r in oligo_recs:
            assert any("reducing" in n.lower() for n in r.notes), \
                f"Oligo {r.sugar} notes should mention reducing end"
