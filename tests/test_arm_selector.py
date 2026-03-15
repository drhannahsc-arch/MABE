"""
test_arm_selector.py — Tests for arm selection module.
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arm_selector import (
    score_arm, select_arms, select_arms_for_rankings,
    electronic_score, guest_electronic_needs,
    arm_pka_score, arm_pka_penalty,
    steric_compatible, scaffold_size_class, arm_size,
    geometry_match_score, get_arm_geometry,
    ArmScore, ArmAssignment, ArmGeometry,
    _get_arm_library,
)
from core.scaffold_designer import ScaffoldDescriptor, GuestSpec, ScaffoldRanking


class TestElectronicCompatibility:

    def test_anion_guest_prefers_nh_donors(self):
        """Anion guests need H-bond donor arms (NH, OH)."""
        needs = guest_electronic_needs(GuestSpec(charge=-2, n_hb_acceptors=3))
        assert "anion_hb_donors" in needs

    def test_aromatic_guest_needs_pi(self):
        needs = guest_electronic_needs(GuestSpec(n_aromatic_rings=2))
        assert "pi_stacking_arms" in needs

    def test_pyrrole_high_for_anion(self):
        """Pyrrole (NH donor) should score well for anion guests."""
        needs = ["anion_hb_donors"]
        score = electronic_score(["N_pyrrole"], needs)
        assert score > 0.5

    def test_thioether_poor_for_anion(self):
        """Thioether (no H-bond donor) poor for anion binding."""
        needs = ["anion_hb_donors"]
        score = electronic_score(["S_thioether"], needs)
        assert score < 0.3

    def test_no_donors_low_score(self):
        """Arm with no donors gets low electronic score."""
        needs = ["hb_donor_arms"]
        score = electronic_score([], needs)
        assert score < 0.2


class TestPKaFiltering:

    def test_pyrrole_always_active(self):
        """Pyrrole NH: always_active, score = 1.0 at any pH."""
        score = arm_pka_score(["N_pyrrole"], 5.0)
        assert score > 0.99

    def test_amine_dead_at_ph5(self):
        """Amine (pKa 10): protonated at pH 5, inactive."""
        score = arm_pka_score(["N_amine"], 5.0)
        assert score < 0.01

    def test_carboxylate_good_at_ph7(self):
        score = arm_pka_score(["O_carboxylate"], 7.0)
        assert score > 0.99

    def test_carboxylate_moderate_at_ph4(self):
        score = arm_pka_score(["O_carboxylate"], 4.0)
        assert 0.4 < score < 0.6

    def test_penalty_higher_for_mismatched_ph(self):
        p_ok = arm_pka_penalty(["O_carboxylate"], 7.0)
        p_bad = arm_pka_penalty(["O_carboxylate"], 2.0)
        assert p_bad > p_ok


class TestStericCompatibility:

    def test_tight_scaffold_rejects_large_arm(self):
        assert not steric_compatible(3.0, "anthracenyl")

    def test_wide_scaffold_accepts_all(self):
        assert steric_compatible(10.0, "anthracenyl")
        assert steric_compatible(10.0, "aminomethyl")

    def test_medium_scaffold_accepts_medium(self):
        assert steric_compatible(6.0, "2-pyridyl")

    def test_size_class_boundaries(self):
        assert scaffold_size_class(3.0) == "tight"
        assert scaffold_size_class(6.0) == "medium"
        assert scaffold_size_class(10.0) == "wide"


class TestGeometryMatching:

    def test_matching_geometry_high_score(self):
        ag = ArmGeometry(name="test", donor_reach_A=3.0,
                         donor_angle_deg=15.0, n_rotors=1)
        score = geometry_match_score(ag, 6.0, 55.0, 3.5)
        assert score > 0.3

    def test_rigid_arm_bonus_in_preorganized(self):
        rigid = ArmGeometry(name="rigid", donor_reach_A=3.5,
                            donor_angle_deg=15.0, n_rotors=0, is_rigid=True)
        flex = ArmGeometry(name="flex", donor_reach_A=3.5,
                           donor_angle_deg=15.0, n_rotors=4)
        s_rigid = geometry_match_score(rigid, 6.0, 55.0, 3.5)
        s_flex = geometry_match_score(flex, 6.0, 55.0, 3.5)
        assert s_rigid >= s_flex

    def test_geometry_score_varies_with_angle(self):
        """Geometry score should change with convergence angle.
        
        Note: geometry_match_score measures SPATIAL reach fit, not
        convergence preference. Convergence preference is handled
        separately by Phase B thermo_score (K_CONVERGENCE_PENALTY).
        This test just verifies the score is angle-dependent.
        """
        ag = ArmGeometry(name="test", donor_reach_A=3.0,
                         donor_angle_deg=15.0, n_rotors=1)
        scores = [geometry_match_score(ag, 6.0, angle, 3.5)
                  for angle in [30, 60, 90, 120, 150]]
        # Scores should vary (not all identical)
        assert max(scores) > min(scores)


class TestArmSelection:

    def _make_sd(self, name="test", **kwargs):
        defaults = dict(smiles="", category="aromatic", n_sites=2,
                        d_arm_A=6.0, theta_conv_deg=55.0, rigidity_index=0.7,
                        n_rotors_bridge=2, V_cavity_est_A3=70.0)
        defaults.update(kwargs)
        return ScaffoldDescriptor(name=name, **defaults)

    def test_select_returns_assignment(self):
        sd = self._make_sd()
        guest = GuestSpec(diameter_A=3.5, charge=-2, n_hb_acceptors=3, pH=5.0)
        a = select_arms(sd, guest)
        assert isinstance(a, ArmAssignment)
        assert len(a.best_arms) == 2
        assert len(a.arm_scores) > 0

    def test_selenite_avoids_amines(self):
        """At pH 5, amines are protonated → should not be top-ranked."""
        sd = self._make_sd()
        guest = GuestSpec(name="selenite", diameter_A=3.5, charge=-2,
                          n_hb_acceptors=3, pH=5.0)
        a = select_arms(sd, guest)
        top_3 = [s.arm_name for s in a.arm_scores[:3]]
        assert "aminomethyl" not in top_3
        assert "aminoethyl" not in top_3

    def test_selenite_prefers_nh_donors(self):
        """Selenite (anion) should prefer pyrrole/urea NH donors."""
        sd = self._make_sd()
        guest = GuestSpec(name="selenite", diameter_A=3.5, charge=-2,
                          n_hb_acceptors=3, pH=5.0)
        a = select_arms(sd, guest)
        top_5_names = [s.arm_name for s in a.arm_scores[:5]]
        nh_donors = {"pyrrole", "indolyl", "urea-NH2", "bis-urea", "ethanol"}
        assert any(n in nh_donors for n in top_5_names)

    def test_neutral_ph_allows_amines(self):
        """At pH 12, amines are free base → should score well."""
        sd = self._make_sd()
        guest = GuestSpec(diameter_A=3.5, n_hb_acceptors=2, pH=12.0)
        a = select_arms(sd, guest)
        amine_scores = [s for s in a.arm_scores if s.arm_name == "aminomethyl"]
        if amine_scores:
            assert amine_scores[0].pka_score > 0.9

    def test_fallback_library_has_arms(self):
        arms = _get_arm_library()
        assert len(arms) >= 20

    def test_score_arm_returns_armscore(self):
        guest = GuestSpec(diameter_A=3.5, charge=-1, pH=7.0)
        s = score_arm("pyrrole", ["N_pyrrole"], "borderline",
                      guest, 6.0, 55.0, 7.0)
        assert isinstance(s, ArmScore)
        assert 0 <= s.composite <= 1.0


class TestBatchSelection:

    def _make_ranking(self, name, **kwargs):
        defaults = dict(smiles="", category="aromatic", n_sites=2,
                        d_arm_A=6.0, theta_conv_deg=55.0, rigidity_index=0.7,
                        n_rotors_bridge=2, V_cavity_est_A3=70.0)
        defaults.update(kwargs)
        sd = ScaffoldDescriptor(name=name, **defaults)
        from core.scaffold_designer import compute_scaffold_requirements, rank_one_scaffold
        guest = GuestSpec(diameter_A=3.5, charge=-2, pH=5.0)
        return rank_one_scaffold(sd, guest)

    def test_batch_returns_list(self):
        rankings = [self._make_ranking("a"), self._make_ranking("b")]
        guest = GuestSpec(diameter_A=3.5, charge=-2, pH=5.0)
        assignments = select_arms_for_rankings(rankings, guest)
        assert len(assignments) == 2
        assert all(isinstance(a, ArmAssignment) for a in assignments)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
