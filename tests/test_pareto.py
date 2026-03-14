"""
tests/test_pareto.py — Tests for Multi-Objective Pareto Optimization

Tests:
  1. Dominance logic (unit)
  2. Non-dominated sorting (unit)
  3. Crowding distance (unit)
  4. Pareto ranking (unit with synthetic data)
  5. Objective sign handling (maximize vs minimize)
  6. Edge cases (empty, single, all-identical)
  7. Integration: generate_candidates with ranking_mode="pareto"
  8. Integration: generate_and_screen with ranking_mode="pareto"
  9. Custom objectives
  10. Regression: existing composite ranking unchanged
"""

import sys
import os
import math
import pytest
from dataclasses import dataclass, field

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.pareto import (
    dominates, fast_non_dominated_sort, crowding_distance,
    pareto_rank, print_pareto,
    Objective, ParetoResult, ParetoCandidate,
    DEFAULT_OBJECTIVES, AFFINITY_SA_OBJECTIVES,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: synthetic candidate for unit tests
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FakeCandidate:
    log_Ka_pred: float = 0.0
    min_gap: float = 0.0
    sa_score_val: float = 5.0
    smiles: str = ""
    name: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 1. DOMINANCE LOGIC
# ═══════════════════════════════════════════════════════════════════════════

class TestDominance:

    def test_clear_dominance(self):
        """a better in all objectives → dominates b."""
        assert dominates((5, 5, 5), (4, 4, 4)) is True

    def test_equal_no_dominance(self):
        """Identical vectors → no dominance."""
        assert dominates((5, 5, 5), (5, 5, 5)) is False

    def test_one_worse_no_dominance(self):
        """a worse in one objective → does not dominate b."""
        assert dominates((5, 5, 3), (4, 4, 4)) is False

    def test_one_better_rest_equal(self):
        """a better in one, equal in rest → dominates."""
        assert dominates((5, 5, 6), (5, 5, 5)) is True

    def test_asymmetric(self):
        """If a dominates b, b does not dominate a."""
        a = (10, 8, 6)
        b = (9, 7, 5)
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_two_objectives(self):
        """Works for 2-objective case."""
        assert dominates((10, 5), (9, 4)) is True
        assert dominates((10, 4), (9, 5)) is False  # tradeoff

    def test_single_objective(self):
        assert dominates((10,), (9,)) is True
        assert dominates((9,), (10,)) is False


# ═══════════════════════════════════════════════════════════════════════════
# 2. NON-DOMINATED SORT
# ═══════════════════════════════════════════════════════════════════════════

class TestNonDominatedSort:

    def test_simple_three_fronts(self):
        """Three clearly separated fronts."""
        vecs = [
            (10, 10),  # 0: front 0
            (9, 9),    # 1: front 1
            (8, 8),    # 2: front 2
        ]
        fronts = fast_non_dominated_sort(vecs)
        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]

    def test_tradeoff_same_front(self):
        """Two candidates with tradeoffs are both on front 0."""
        vecs = [
            (10, 5),   # 0: high affinity, low selectivity
            (5, 10),   # 1: low affinity, high selectivity
        ]
        fronts = fast_non_dominated_sort(vecs)
        assert len(fronts[0]) == 2
        assert set(fronts[0]) == {0, 1}

    def test_one_dominated(self):
        """One candidate dominated by both others."""
        vecs = [
            (10, 5),
            (5, 10),
            (4, 4),   # dominated by both
        ]
        fronts = fast_non_dominated_sort(vecs)
        assert set(fronts[0]) == {0, 1}
        assert fronts[1] == [2]

    def test_empty_input(self):
        assert fast_non_dominated_sort([]) == []

    def test_single_input(self):
        fronts = fast_non_dominated_sort([(5, 5)])
        assert fronts[0] == [0]

    def test_all_identical(self):
        """All identical → all on front 0 (none dominates another)."""
        vecs = [(5, 5)] * 4
        fronts = fast_non_dominated_sort(vecs)
        assert len(fronts[0]) == 4

    def test_three_objectives(self):
        """3D Pareto front."""
        vecs = [
            (10, 5, 3),   # 0
            (5, 10, 3),   # 1
            (5, 5, 10),   # 2
            (4, 4, 4),    # 3: dominated by 0
        ]
        fronts = fast_non_dominated_sort(vecs)
        assert set(fronts[0]) == {0, 1, 2}
        assert fronts[1] == [3]


# ═══════════════════════════════════════════════════════════════════════════
# 3. CROWDING DISTANCE
# ═══════════════════════════════════════════════════════════════════════════

class TestCrowdingDistance:

    def test_two_members_both_inf(self):
        """Front with 2 members → both get infinite distance."""
        vecs = [(10, 5), (5, 10)]
        cd = crowding_distance([0, 1], vecs)
        assert cd[0] == float('inf')
        assert cd[1] == float('inf')

    def test_boundary_inf(self):
        """Boundary members of a front get infinite distance."""
        vecs = [(10, 2), (7, 5), (4, 8)]
        cd = crowding_distance([0, 1, 2], vecs)
        # Extremes in each objective get inf
        assert cd[0] == float('inf')
        assert cd[2] == float('inf')
        # Middle gets finite
        assert cd[1] > 0 and cd[1] < float('inf')

    def test_middle_gets_finite(self):
        """Interior member gets a finite positive distance."""
        vecs = [(1, 10), (5, 5), (10, 1)]
        cd = crowding_distance([0, 1, 2], vecs)
        assert cd[1] > 0
        assert cd[1] < float('inf')

    def test_single_member(self):
        vecs = [(5, 5)]
        cd = crowding_distance([0], vecs)
        assert cd[0] == float('inf')


# ═══════════════════════════════════════════════════════════════════════════
# 4. PARETO RANKING (FULL PIPELINE)
# ═══════════════════════════════════════════════════════════════════════════

class TestParetoRank:

    def test_basic_ranking(self):
        """3 Pareto-optimal + 2 dominated."""
        fakes = [
            FakeCandidate(log_Ka_pred=20, min_gap=5, sa_score_val=3),
            FakeCandidate(log_Ka_pred=15, min_gap=10, sa_score_val=4),
            FakeCandidate(log_Ka_pred=18, min_gap=8, sa_score_val=2),
            FakeCandidate(log_Ka_pred=14, min_gap=4, sa_score_val=6),
            FakeCandidate(log_Ka_pred=12, min_gap=3, sa_score_val=7),
        ]
        result = pareto_rank(fakes)
        assert result.n_pareto_optimal == 3
        assert len(result.fronts) >= 2

    def test_all_pareto_optimal(self):
        """All candidates are non-dominated (all tradeoffs)."""
        fakes = [
            FakeCandidate(log_Ka_pred=10, min_gap=1, sa_score_val=1),
            FakeCandidate(log_Ka_pred=1, min_gap=10, sa_score_val=1),
            FakeCandidate(log_Ka_pred=1, min_gap=1, sa_score_val=0.1),
        ]
        result = pareto_rank(fakes)
        assert result.n_pareto_optimal == 3
        assert len(result.fronts) == 1

    def test_single_candidate(self):
        fakes = [FakeCandidate(log_Ka_pred=10, min_gap=5, sa_score_val=3)]
        result = pareto_rank(fakes)
        assert result.n_pareto_optimal == 1
        assert result.candidates[0].pareto_rank == 1

    def test_empty_input(self):
        result = pareto_rank([])
        assert result.n_pareto_optimal == 0
        assert result.candidates == []

    def test_pareto_rank_1_is_front_0(self):
        """All rank-1 candidates must be on front 0."""
        fakes = [
            FakeCandidate(log_Ka_pred=20, min_gap=5, sa_score_val=3),
            FakeCandidate(log_Ka_pred=5, min_gap=20, sa_score_val=4),
            FakeCandidate(log_Ka_pred=10, min_gap=2, sa_score_val=8),
        ]
        result = pareto_rank(fakes)
        for pc in result.candidates:
            if pc.front == 0:
                assert pc.pareto_rank <= result.n_pareto_optimal

    def test_front_ordering(self):
        """All front-0 candidates ranked before front-1."""
        fakes = [
            FakeCandidate(log_Ka_pred=20, min_gap=5, sa_score_val=3),
            FakeCandidate(log_Ka_pred=15, min_gap=10, sa_score_val=4),
            FakeCandidate(log_Ka_pred=5, min_gap=2, sa_score_val=8),
        ]
        result = pareto_rank(fakes)
        ranks_by_front = {}
        for pc in result.candidates:
            ranks_by_front.setdefault(pc.front, []).append(pc.pareto_rank)
        if len(ranks_by_front) > 1:
            max_front0 = max(ranks_by_front[0])
            min_front1 = min(ranks_by_front[1])
            assert max_front0 < min_front1

    def test_objectives_dict_populated(self):
        """Each ParetoCandidate has objective values."""
        fakes = [
            FakeCandidate(log_Ka_pred=20, min_gap=5, sa_score_val=3),
            FakeCandidate(log_Ka_pred=15, min_gap=10, sa_score_val=4),
        ]
        result = pareto_rank(fakes)
        for pc in result.candidates:
            assert "affinity" in pc.objectives
            assert "selectivity" in pc.objectives
            assert "synthesizability" in pc.objectives
            assert pc.objectives["affinity"] == fakes[pc.index].log_Ka_pred
            assert pc.objectives["synthesizability"] == fakes[pc.index].sa_score_val


# ═══════════════════════════════════════════════════════════════════════════
# 5. OBJECTIVE SIGN HANDLING
# ═══════════════════════════════════════════════════════════════════════════

class TestObjectiveSigns:

    def test_maximize_positive(self):
        """Maximize objective: higher is better."""
        obj = Objective(name="aff", extract=lambda c: c.log_Ka_pred, maximize=True)
        c = FakeCandidate(log_Ka_pred=10)
        assert obj.signed_value(c) == 10

    def test_minimize_negated(self):
        """Minimize objective: lower is better → negated internally."""
        obj = Objective(name="sa", extract=lambda c: c.sa_score_val, maximize=False)
        c = FakeCandidate(sa_score_val=3)
        assert obj.signed_value(c) == -3

    def test_low_sa_dominates_high_sa(self):
        """Candidate with lower SA score should dominate (SA is minimized)."""
        good = FakeCandidate(log_Ka_pred=10, min_gap=5, sa_score_val=2)
        bad = FakeCandidate(log_Ka_pred=10, min_gap=5, sa_score_val=8)
        result = pareto_rank([good, bad])
        # good dominates bad (same aff/sel, better SA)
        assert result.n_pareto_optimal == 1
        assert result.candidates[0].index == 0  # good


# ═══════════════════════════════════════════════════════════════════════════
# 6. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_all_identical_candidates(self):
        """All identical → all on front 0, no dominance."""
        fakes = [FakeCandidate(log_Ka_pred=10, min_gap=5, sa_score_val=3)] * 5
        result = pareto_rank(fakes)
        assert result.n_pareto_optimal == 5

    def test_two_candidates_one_dominates(self):
        a = FakeCandidate(log_Ka_pred=20, min_gap=10, sa_score_val=2)
        b = FakeCandidate(log_Ka_pred=10, min_gap=5, sa_score_val=5)
        result = pareto_rank([a, b])
        assert result.n_pareto_optimal == 1
        assert result.candidates[0].index == 0

    def test_hypervolume_nonnegative(self):
        fakes = [
            FakeCandidate(log_Ka_pred=20, min_gap=5, sa_score_val=3),
            FakeCandidate(log_Ka_pred=15, min_gap=10, sa_score_val=4),
        ]
        result = pareto_rank(fakes)
        assert result.hypervolume >= 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. CUSTOM OBJECTIVES
# ═══════════════════════════════════════════════════════════════════════════

class TestCustomObjectives:

    def test_four_objectives(self):
        """User-defined 4-objective ranking."""
        @dataclass
        class ExtCandidate:
            log_Ka_pred: float = 0.0
            min_gap: float = 0.0
            sa_score_val: float = 5.0
            novelty_score: float = 0.0

        custom_objs = [
            Objective("affinity", lambda c: c.log_Ka_pred, True),
            Objective("selectivity", lambda c: c.min_gap, True),
            Objective("SA", lambda c: c.sa_score_val, False),
            Objective("novelty", lambda c: c.novelty_score, True),
        ]
        candidates = [
            ExtCandidate(20, 5, 3, 0.9),
            ExtCandidate(15, 10, 4, 0.5),
            ExtCandidate(18, 8, 2, 0.7),
        ]
        result = pareto_rank(candidates, objectives=custom_objs)
        assert result.n_pareto_optimal >= 1
        assert "novelty" in result.objective_names

    def test_single_objective_degenerates_to_sort(self):
        """With one objective, Pareto rank == sort order."""
        objs = [Objective("aff", lambda c: c.log_Ka_pred, True)]
        fakes = [
            FakeCandidate(log_Ka_pred=10),
            FakeCandidate(log_Ka_pred=20),
            FakeCandidate(log_Ka_pred=15),
        ]
        result = pareto_rank(fakes, objectives=objs)
        # Each dominates the next → n fronts == n candidates
        assert result.candidates[0].index == 1  # highest aff
        assert result.candidates[1].index == 2
        assert result.candidates[2].index == 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. AUTO-SELECT: 2 vs 3 objectives
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoSelect:

    def test_with_selectivity_uses_3_obj(self):
        """When min_gap != 0, auto-selects 3-objective mode."""
        fakes = [
            FakeCandidate(log_Ka_pred=20, min_gap=5, sa_score_val=3),
            FakeCandidate(log_Ka_pred=15, min_gap=10, sa_score_val=4),
        ]
        result = pareto_rank(fakes)
        assert "selectivity" in result.objective_names

    def test_without_selectivity_uses_2_obj(self):
        """When all min_gap == 0, auto-selects 2-objective mode."""
        fakes = [
            FakeCandidate(log_Ka_pred=20, min_gap=0, sa_score_val=3),
            FakeCandidate(log_Ka_pred=15, min_gap=0, sa_score_val=4),
        ]
        result = pareto_rank(fakes)
        assert "selectivity" not in result.objective_names
        assert "affinity" in result.objective_names
        assert "synthesizability" in result.objective_names


# ═══════════════════════════════════════════════════════════════════════════
# 9. INTEGRATION: generate_candidates with ranking_mode="pareto"
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationGenerateCandidates:

    @pytest.fixture(autouse=True)
    def _skip_if_slow(self):
        """These tests run RDKit + scorer, ~30s each."""
        pass

    def test_pareto_mode_populates_fields(self):
        """Pareto ranking mode fills pareto_* fields on candidates."""
        from core.de_novo_generator import generate_candidates
        result = generate_candidates("Cu2+", max_candidates=30,
                                     max_scored=10,
                                     ranking_mode="pareto")
        assert result.ranking_mode == "pareto"
        assert result.pareto_result is not None
        if result.candidates:
            c = result.candidates[0]
            assert c.pareto_front_idx >= 0
            assert c.pareto_rank >= 1
            assert "affinity" in c.pareto_objectives

    def test_composite_mode_still_works(self):
        """Default composite mode unchanged."""
        from core.de_novo_generator import generate_candidates
        result = generate_candidates("Cu2+", max_candidates=30,
                                     max_scored=10,
                                     ranking_mode="composite")
        assert result.ranking_mode == "composite"
        assert result.pareto_result is None
        if result.candidates:
            assert result.candidates[0].pareto_front_idx == -1


# ═══════════════════════════════════════════════════════════════════════════
# 10. INTEGRATION: generate_and_screen with ranking_mode="pareto"
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationGenerateAndScreen:

    def test_pareto_selectivity_screening(self):
        """Pareto ranking with 3 objectives through selectivity pipeline."""
        from core.de_novo_generator import generate_and_screen
        result = generate_and_screen(
            "Pb2+", ["Ca2+"],
            max_candidates=30, max_scored=10,
            ranking_mode="pareto",
        )
        assert result.ranking_mode == "pareto"
        assert result.pareto_result is not None
        if result.candidates:
            assert result.pareto_result.n_pareto_optimal >= 1
            # All front-0 candidates ranked before front-1
            front0_ranks = [c.pareto_rank for c in result.candidates
                            if c.pareto_front_idx == 0]
            front1_ranks = [c.pareto_rank for c in result.candidates
                            if c.pareto_front_idx == 1]
            if front0_ranks and front1_ranks:
                assert max(front0_ranks) < min(front1_ranks)


# ═══════════════════════════════════════════════════════════════════════════
# 11. REGRESSION: existing tests still pass
# ═══════════════════════════════════════════════════════════════════════════

class TestRegression:

    def test_default_mode_is_composite(self):
        """generate_candidates defaults to composite ranking."""
        from core.de_novo_generator import generate_candidates
        result = generate_candidates("Cu2+", max_candidates=20,
                                     max_scored=5)
        assert result.ranking_mode == "composite"

    def test_generate_and_screen_default_is_composite(self):
        """generate_and_screen defaults to composite ranking."""
        from core.de_novo_generator import generate_and_screen
        result = generate_and_screen("Pb2+", ["Ca2+"],
                                     max_candidates=20, max_scored=5)
        assert result.ranking_mode == "composite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
