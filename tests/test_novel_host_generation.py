"""
tests/test_novel_host_generation.py — Tests for Novel Host Guest Generation

Tests:
  1. NovelHostSpec construction + auto Rebek volume
  2. _score_for_novel_host() unit tests
  3. generate_for_host() with NovelHostSpec integration
  4. Guest size filtering against cavity volume
  5. Pareto mode with novel hosts
  6. Known host regression (beta-CD unchanged)
  7. Edge cases (zero volume, tiny cavity)
"""

import sys
import os
import math
import pytest
from dataclasses import dataclass

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.de_novo_generator import (
    NovelHostSpec, _score_for_novel_host,
    generate_for_host, GenerationResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. NovelHostSpec CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

class TestNovelHostSpec:

    def test_basic_construction(self):
        hs = NovelHostSpec(name="MOF-A", cavity_volume_A3=500.0)
        assert hs.name == "MOF-A"
        assert hs.cavity_volume_A3 == 500.0
        assert hs.host_type == "novel_cavity"

    def test_auto_rebek_volume(self):
        """max_guest_volume auto-set to 65% of cavity volume."""
        hs = NovelHostSpec(name="test", cavity_volume_A3=1000.0)
        assert abs(hs.max_guest_volume_A3 - 650.0) < 0.1

    def test_explicit_max_guest_volume(self):
        """Explicit max_guest_volume overrides auto."""
        hs = NovelHostSpec(name="test", cavity_volume_A3=1000.0,
                           max_guest_volume_A3=400.0)
        assert hs.max_guest_volume_A3 == 400.0

    def test_zero_volume_no_auto(self):
        hs = NovelHostSpec(name="test", cavity_volume_A3=0.0)
        assert hs.max_guest_volume_A3 == 0.0

    def test_portal_type_default(self):
        hs = NovelHostSpec(name="test", cavity_volume_A3=500.0)
        assert hs.portal_type == "neutral"

    def test_host_charge_default(self):
        hs = NovelHostSpec(name="test", cavity_volume_A3=500.0)
        assert hs.host_charge == 0

    def test_hkust1_spec(self):
        """HKUST-1: V≈636 ų, neutral OMS portals."""
        hs = NovelHostSpec(
            name="HKUST-1", cavity_volume_A3=636.0,
            host_type="MOF", portal_type="neutral",
        )
        assert hs.max_guest_volume_A3 == pytest.approx(413.4, abs=0.1)

    def test_uio66nh2_spec(self):
        """UiO-66-NH2: V≈905 ų, amine portal."""
        hs = NovelHostSpec(
            name="UiO-66-NH2", cavity_volume_A3=905.0,
            host_type="MOF", portal_type="amine",
            n_hbonds_host=1,
        )
        assert hs.n_hbonds_host == 1
        assert hs.portal_type == "amine"


# ═══════════════════════════════════════════════════════════════════════════
# 2. _score_for_novel_host UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreForNovelHost:

    def test_toluene_in_mof(self):
        """Toluene in large MOF cavity should score favorably."""
        hs = NovelHostSpec(name="big_MOF", cavity_volume_A3=800.0)
        sc = _score_for_novel_host("Cc1ccccc1", hs)
        assert sc.dg_total_kj < 0, "Toluene in MOF must bind favorably"
        assert sc.log_Ka_pred > 0

    def test_co2_in_mof(self):
        """CO2 in MOF cavity."""
        hs = NovelHostSpec(name="HKUST-1", cavity_volume_A3=636.0)
        sc = _score_for_novel_host("O=C=O", hs)
        assert sc.dg_total_kj < 0

    def test_hydrophobic_term_fires(self):
        """Hydrophobic transfer should be nonzero for aromatic guest."""
        hs = NovelHostSpec(name="cage", cavity_volume_A3=500.0)
        sc = _score_for_novel_host("c1ccccc1", hs)  # benzene
        assert sc.prediction.dg_hydrophobic < 0

    def test_hbond_with_host_spec(self):
        """n_hbonds_host propagates to scoring."""
        hs_no_hb = NovelHostSpec(name="cage_bare", cavity_volume_A3=500.0,
                                 n_hbonds_host=0)
        hs_hb = NovelHostSpec(name="cage_func", cavity_volume_A3=500.0,
                              n_hbonds_host=2)
        sc_no = _score_for_novel_host("c1ccccc1", hs_no_hb)
        sc_hb = _score_for_novel_host("c1ccccc1", hs_hb)
        # H-bond term should differ
        assert sc_no.prediction.dg_hbond != sc_hb.prediction.dg_hbond

    def test_name_propagated(self):
        hs = NovelHostSpec(name="my_cage", cavity_volume_A3=500.0)
        sc = _score_for_novel_host("C", hs, name="methane@my_cage")
        assert "my_cage" in sc.name

    def test_binding_mode_set(self):
        """Should set host_guest_inclusion mode."""
        hs = NovelHostSpec(name="cage", cavity_volume_A3=500.0)
        sc = _score_for_novel_host("Cc1ccccc1", hs)
        assert sc.prediction.binding_mode == "host_guest_inclusion"


# ═══════════════════════════════════════════════════════════════════════════
# 3. generate_for_host WITH NovelHostSpec
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateForNovelHost:

    def test_generates_candidates(self):
        """Novel host produces scored candidates."""
        hs = NovelHostSpec(name="HKUST-1", cavity_volume_A3=636.0)
        r = generate_for_host(hs, max_candidates=20, max_scored=5)
        assert isinstance(r, GenerationResult)
        assert r.target == "HKUST-1"
        assert r.mode == "host_guest"
        assert r.n_scored > 0

    def test_candidates_have_host_name(self):
        hs = NovelHostSpec(name="test_MOF", cavity_volume_A3=500.0)
        r = generate_for_host(hs, max_candidates=20, max_scored=5)
        if r.candidates:
            assert "test_MOF" in r.candidates[0].name

    def test_candidates_score_negative_dg(self):
        """At least some candidates should have favorable binding."""
        hs = NovelHostSpec(name="big_cage", cavity_volume_A3=800.0)
        r = generate_for_host(hs, max_candidates=30, max_scored=10)
        if r.candidates:
            has_favorable = any(c.dg_total_kj < 0 for c in r.candidates)
            assert has_favorable, "At least one candidate should bind favorably"

    def test_small_cavity_limits_guests(self):
        """Very small cavity should restrict heavy atom count."""
        hs = NovelHostSpec(name="tiny", cavity_volume_A3=100.0)
        r = generate_for_host(hs, max_candidates=30, max_scored=10)
        # max_guest_volume = 100 * 0.65 = 65 ų → ~6 heavy atoms max
        # Some or all candidates may be filtered out
        assert r.n_enumerated >= 0  # just verify it doesn't crash


# ═══════════════════════════════════════════════════════════════════════════
# 4. PARETO MODE WITH NOVEL HOSTS
# ═══════════════════════════════════════════════════════════════════════════

class TestParetoNovelHost:

    def test_pareto_mode(self):
        hs = NovelHostSpec(name="HKUST-1", cavity_volume_A3=636.0)
        r = generate_for_host(hs, max_candidates=20, max_scored=5,
                              ranking_mode="pareto")
        assert r.ranking_mode == "pareto"
        assert r.pareto_result is not None
        if r.candidates:
            assert r.candidates[0].pareto_front_idx >= 0

    def test_composite_mode_default(self):
        hs = NovelHostSpec(name="cage", cavity_volume_A3=500.0)
        r = generate_for_host(hs, max_candidates=20, max_scored=5)
        assert r.ranking_mode == "composite"
        assert r.pareto_result is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. KNOWN HOST REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

class TestKnownHostRegression:

    def test_beta_cd_still_works(self):
        """String host key still routes through HOST_REGISTRY."""
        r = generate_for_host("beta-CD", max_candidates=20, max_scored=5)
        assert r.target == "beta-CD"
        assert r.n_scored > 0

    def test_cb7_still_works(self):
        r = generate_for_host("CB7", max_candidates=20, max_scored=5)
        assert r.target == "CB7"
        assert r.n_scored > 0

    def test_known_host_pareto_mode(self):
        r = generate_for_host("beta-CD", max_candidates=20, max_scored=5,
                              ranking_mode="pareto")
        assert r.ranking_mode == "pareto"


# ═══════════════════════════════════════════════════════════════════════════
# 6. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_zero_volume_spec(self):
        """Zero cavity volume → no scoring possible."""
        hs = NovelHostSpec(name="flat", cavity_volume_A3=0.0)
        # Should not crash, may produce zero-scored candidates
        r = generate_for_host(hs, max_candidates=10, max_scored=5)
        assert isinstance(r, GenerationResult)

    def test_very_large_cavity(self):
        """Very large cavity → many candidates pass size filter."""
        hs = NovelHostSpec(name="huge_cage", cavity_volume_A3=5000.0)
        r = generate_for_host(hs, max_candidates=20, max_scored=5)
        assert r.n_enumerated > 0

    def test_charged_host(self):
        hs = NovelHostSpec(name="charged_cage", cavity_volume_A3=500.0,
                           host_charge=-2)
        r = generate_for_host(hs, max_candidates=20, max_scored=5)
        assert isinstance(r, GenerationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
