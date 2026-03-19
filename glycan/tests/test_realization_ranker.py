"""
Tests for glycan/realization_ranker.py -- fabrication specs for immune cell pulldown.
"""

import pytest
from glycan.realization_ranker import (
    rank_all_options,
    list_scaffolds,
    list_binders,
    list_glycan_cell_types,
    SCAFFOLDS,
    STRATEGY_B_BINDERS,
    CELL_GLYCAN_PROFILES,
    _resolve_glycan_cell_type,
    _dG_to_Kd_uM,
    FabricationSpec,
)


# ── Cell type resolution ────────────────────────────────────────────────

class TestCellTypeResolution:
    def test_canonical(self):
        assert _resolve_glycan_cell_type("macrophage_m2") == "macrophage_m2"

    def test_alias(self):
        assert _resolve_glycan_cell_type("cancer") == "tumor_epithelial"
        assert _resolve_glycan_cell_type("t cell") == "t_cell"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _resolve_glycan_cell_type("alien_cell")


# ── Data completeness ───────────────────────────────────────────────────

class TestDataCompleteness:
    def test_all_cell_types_have_glycan_profiles(self):
        for ct in list_glycan_cell_types():
            profiles = CELL_GLYCAN_PROFILES[ct]
            assert len(profiles) >= 1, f"{ct} has no glycan profiles"

    def test_all_scaffolds_have_strategy(self):
        for name, s in SCAFFOLDS.items():
            assert len(s.strategies) >= 1, f"{name} has no strategy"
            assert s.cost_per_test_usd > 0

    def test_all_binders_have_targets(self):
        for name, b in STRATEGY_B_BINDERS.items():
            assert len(b.target_glycans) >= 1, f"{name} has no target glycans"
            assert b.estimated_Kd_uM > 0

    def test_scaffolds_cover_both_strategies(self):
        a_scaffolds = [s for s in SCAFFOLDS.values() if "A" in s.strategies]
        b_scaffolds = [s for s in SCAFFOLDS.values() if "B" in s.strategies]
        assert len(a_scaffolds) >= 3
        assert len(b_scaffolds) >= 5

    def test_glycan_profiles_have_sources(self):
        for ct, profiles in CELL_GLYCAN_PROFILES.items():
            for p in profiles:
                assert p.source != "", f"{ct}/{p.glycan} missing source"


# ── Strategy A (sugar -> lectin) ────────────────────────────────────────

class TestStrategyA:
    def test_macrophage_has_strategy_a(self):
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        a_opts = [o for o in opts if o.strategy == "A"]
        assert len(a_opts) >= 1

    def test_strategy_a_has_sugar_binder(self):
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        a_opts = [o for o in opts if o.strategy == "A"]
        for o in a_opts:
            assert o.binder is not None
            assert o.estimated_Kd_nM > 0


# ── Strategy B (synthetic lectin -> glycan) ─────────────────────────────

class TestStrategyB:
    def test_macrophage_has_strategy_b(self):
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        b_opts = [o for o in opts if o.strategy == "B"]
        assert len(b_opts) >= 1

    def test_tumor_has_pna_option(self):
        """Tumors display Tn (GalNAc) -> PNA lectin should be an option."""
        opts = rank_all_options("tumor_epithelial", bead_diameter_nm=50)
        pna = [o for o in opts if "PNA" in o.binder_display]
        assert len(pna) >= 1

    def test_b_cell_has_sna_option(self):
        """B cells display alpha2-6 Sia -> SNA lectin should appear."""
        opts = rank_all_options("b_cell", bead_diameter_nm=50)
        sna = [o for o in opts if "SNA" in o.binder_display]
        assert len(sna) >= 1

    def test_boronic_acid_appears_for_all(self):
        """Boronic acid is broad-spectrum -> should appear for every cell type."""
        for ct in list_glycan_cell_types():
            opts = rank_all_options(ct, bead_diameter_nm=50)
            ba = [o for o in opts if "boronic" in o.binder.lower()]
            assert len(ba) >= 1, f"No boronic acid option for {ct}"

    def test_plant_lectin_cheap(self):
        """Plant lectin + bead should be cheapest Strategy B option."""
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        b_opts = [o for o in opts if o.strategy == "B"]
        lectin_opts = [o for o in b_opts if "lectin" in o.scaffold.lower() or "lectin" in o.binder.lower()]
        if lectin_opts:
            cheapest = min(lectin_opts, key=lambda o: o.cost_per_test_usd)
            assert cheapest.cost_per_test_usd <= 2.0


# ── Ranking and output ──────────────────────────────────────────────────

class TestRanking:
    def test_sorted_by_composite(self):
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        scores = [o.composite_score for o in opts]
        assert scores == sorted(scores, reverse=True)

    def test_all_have_positive_score(self):
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        for o in opts:
            assert o.composite_score > 0

    def test_summary_is_string(self):
        opts = rank_all_options("t_cell", bead_diameter_nm=50)
        for o in opts[:5]:
            assert isinstance(o.summary, str)
            assert len(o.summary) > 20

    def test_both_strategies_present(self):
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        strategies = {o.strategy for o in opts}
        assert "A" in strategies
        assert "B" in strategies

    def test_athena_option_exists(self):
        """DNA origami option should be flagged as ATHENA-compatible."""
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        athena = [o for o in opts if o.athena_compatible]
        assert len(athena) >= 1

    def test_rfd_option_exists(self):
        """Designed mini-protein should be flagged as RFdiffusion-compatible."""
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        rfd = [o for o in opts if o.rfd_compatible]
        assert len(rfd) >= 1


# ── Physics checks ──────────────────────────────────────────────────────

class TestPhysics:
    def test_dG_to_Kd_conversion(self):
        # dG = -RT ln(Ka) = RT ln(Kd)
        # At -22.2 kJ/mol: Ka = 7600 M-1, Kd = 131 uM
        Kd = _dG_to_Kd_uM(-22.2)
        assert 100 < Kd < 200

    def test_multivalent_enhancement_helps(self):
        """Higher-valency scaffolds should show lower effective Kd."""
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        a_opts = [o for o in opts if o.strategy == "A" and o.binder == opts[0].binder
                  if opts[0].strategy == "A"]
        if len(a_opts) >= 2:
            # Sort by valency
            a_opts.sort(key=lambda o: o.effective_valency)
            # Higher valency should give lower Kd_effective (usually)
            if a_opts[0].effective_valency < a_opts[-1].effective_valency:
                assert a_opts[-1].estimated_Kd_effective_nM <= a_opts[0].estimated_Kd_effective_nM

    def test_effective_kd_lower_than_monovalent(self):
        """Multivalent Kd should be <= monovalent Kd (within rounding)."""
        opts = rank_all_options("macrophage_m2", bead_diameter_nm=50)
        for o in opts:
            assert o.estimated_Kd_effective_nM <= o.estimated_Kd_nM * 1.001


# ── Cross-cell-type ─────────────────────────────────────────────────────

class TestCrossCellType:
    def test_all_cell_types_produce_options(self):
        for ct in list_glycan_cell_types():
            opts = rank_all_options(ct, bead_diameter_nm=50)
            assert len(opts) >= 1, f"No options for {ct}"

    def test_tumor_has_many_options(self):
        """Tumors have rich glycan profile -> many options."""
        opts = rank_all_options("tumor_epithelial", bead_diameter_nm=50)
        assert len(opts) >= 10
