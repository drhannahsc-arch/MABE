"""
tests/test_glycan_g2_conformational.py -- G2 conformational entropy tests
"""
import pytest, math
from mabe.glycan.conformational import (
    LINKAGE_REGISTRY, K_BRANCH_PENALTY, compute_conformational_entropy, _compute_TdS,
    BETA_1_4, BETA_1_3, ALPHA_1_3, ALPHA_1_4, ALPHA_1_6, ALPHA_2_3, R, T,
)
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
from mabe.glycan.contact_map import cona_mannose_pocket, cona_trimannoside

class TestLinkageProfiles:
    def test_all_8_linkages_present(self):
        assert len(LINKAGE_REGISTRY) == 8
    def test_populations_sum_to_one(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert abs(sum(p.populations) - 1.0) < 0.01, n
    def test_all_positive(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert all(x > 0 for x in p.populations), n
    def test_n_minima_matches(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert len(p.populations) == p.n_minima, n
    def test_tds_in_range(self):
        for n, p in LINKAGE_REGISTRY.items():
            assert 0.5 < p.TdS_freeze_kj < 7.0, f"{n}: {p.TdS_freeze_kj}"
    def test_alpha16_3_torsions(self):
        assert ALPHA_1_6.n_torsions == 3
    def test_others_2_torsions(self):
        for n, p in LINKAGE_REGISTRY.items():
            if n != 'alpha1-6':
                assert p.n_torsions == 2, n

class TestPhysicalOrdering:
    def test_alpha16_most_flexible(self):
        for n, p in LINKAGE_REGISTRY.items():
            if n != 'alpha1-6':
                assert ALPHA_1_6.TdS_freeze_kj > p.TdS_freeze_kj, n
    def test_alpha16_gt_alpha14(self):
        assert ALPHA_1_6.TdS_freeze_kj > ALPHA_1_4.TdS_freeze_kj
    def test_alpha14_gt_alpha23(self):
        assert ALPHA_1_4.TdS_freeze_kj > ALPHA_2_3.TdS_freeze_kj

class TestTdSComputation:
    def test_single_state_zero(self):
        assert abs(_compute_TdS([1.0])) < 1e-10
    def test_two_equal(self):
        assert abs(_compute_TdS([0.5, 0.5]) - R * T * math.log(2)) < 0.01
    def test_three_equal(self):
        assert abs(_compute_TdS([1/3, 1/3, 1/3]) - R * T * math.log(3)) < 0.01
    def test_more_states_more_entropy(self):
        assert _compute_TdS([1/3]*3) > _compute_TdS([0.5, 0.5])
    def test_skewed_less(self):
        assert _compute_TdS([0.5, 0.5]) > _compute_TdS([0.9, 0.1])

class TestScoring:
    def test_mono_zero(self):
        assert compute_conformational_entropy(None)['TdS_total'] == 0.0
    def test_empty_zero(self):
        assert compute_conformational_entropy([])['TdS_total'] == 0.0
    def test_single(self):
        r = compute_conformational_entropy(['beta1-4'])
        assert r['TdS_total'] == BETA_1_4.TdS_freeze_kj
    def test_additive(self):
        r = compute_conformational_entropy(['alpha1-3', 'alpha1-6'])
        assert abs(r['TdS_total'] - (ALPHA_1_3.TdS_freeze_kj + ALPHA_1_6.TdS_freeze_kj)) < 0.01
    def test_branch(self):
        a = compute_conformational_entropy(['alpha1-3'], n_branch_points=0)
        b = compute_conformational_entropy(['alpha1-3'], n_branch_points=1)
        assert b['TdS_total'] - a['TdS_total'] == K_BRANCH_PENALTY
    def test_unknown_fallback(self):
        r = compute_conformational_entropy(['unknown'])
        assert 4.0 < r['TdS_total'] < 7.0

class TestScorerIntegration:
    def test_mono_no_conf(self):
        assert compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket()).dG_conf_entropy == 0.0
    def test_tri_has_conf(self):
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cona_trimannoside())
        assert 6.0 < r.dG_conf_entropy < 9.0
    def test_tri_less_favorable(self):
        m = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket(), beta_context=0.45)
        t = compute_glycan_terms(ALPHA_D_MANNOSE, cona_trimannoside(), beta_context=0.45)
        assert (t.dG_conf_entropy - m.dG_conf_entropy) > 5.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
