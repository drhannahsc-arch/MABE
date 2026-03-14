"""
test_linkage_entropy.py — Phase G2 conformational entropy tests.

Tests:
  - Physics constraints (ordering, range, positivity)
  - Rotamer entropy calculator correctness
  - Linkage table completeness
  - Cross-checks against MABE HG parameters
  - Integration with glycan scorer
  - Regression: all G1 tests still pass (if available)

NO binding data used in any test.
"""
import pytest
import numpy as np
import sys
import os

# Add parent to path for standalone execution
sys.path.insert(0, os.path.dirname(__file__))

from linkage_entropy import (
    rotamer_entropy,
    rotamer_entropy_from_barrier,
    get_TdS_freeze,
    score_conformational_entropy,
    get_branch_penalty,
    update_glycan_params_with_G2,
    build_linkage_entropy_table,
    LINKAGE_ENTROPY_TABLE,
    RT_STD,
    T_STD,
)


# ═══════════════════════════════════════════════════════════════════════════
# Core physics tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRotamerEntropy:
    """Test the fundamental rotamer_entropy function."""

    def test_single_state_zero_entropy(self):
        """A single rotamer has zero conformational entropy."""
        assert rotamer_entropy([1.0]) == pytest.approx(0.0, abs=1e-10)

    def test_two_equal_states(self):
        """Two equally populated rotamers: TΔS = RT×ln(2)."""
        expected = RT_STD * np.log(2)
        assert rotamer_entropy([0.5, 0.5]) == pytest.approx(expected, rel=1e-6)

    def test_three_equal_states(self):
        """Three equally populated rotamers: TΔS = RT×ln(3)."""
        expected = RT_STD * np.log(3)
        assert rotamer_entropy([1/3, 1/3, 1/3]) == pytest.approx(expected, rel=1e-3)

    def test_always_non_negative(self):
        """TΔS_freeze is always ≥ 0 (entropy of a distribution)."""
        test_pops = [
            [1.0],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.99, 0.01],
            [0.5, 0.3, 0.2],
            [0.7, 0.2, 0.1],
        ]
        for pops in test_pops:
            assert rotamer_entropy(pops) >= 0.0

    def test_more_states_more_entropy(self):
        """More equally populated states → higher entropy."""
        TdS_2 = rotamer_entropy([0.5, 0.5])
        TdS_3 = rotamer_entropy([1/3, 1/3, 1/3])
        assert TdS_3 > TdS_2

    def test_skewed_less_than_uniform(self):
        """Skewed populations have less entropy than uniform."""
        TdS_uniform = rotamer_entropy([0.5, 0.5])
        TdS_skewed = rotamer_entropy([0.9, 0.1])
        assert TdS_skewed < TdS_uniform

    def test_populations_must_sum_to_one(self):
        """Populations that don't sum to ~1 should raise."""
        with pytest.raises(ValueError, match="sum to"):
            rotamer_entropy([0.5, 0.3])  # sums to 0.8

    def test_temperature_scaling(self):
        """TΔS scales linearly with temperature."""
        TdS_298 = rotamer_entropy([0.5, 0.5], T=298.15)
        TdS_310 = rotamer_entropy([0.5, 0.5], T=310.0)
        assert TdS_310 / TdS_298 == pytest.approx(310.0 / 298.15, rel=1e-4)


class TestBarrierEntropy:
    """Test entropy computation from barrier heights."""

    def test_symmetric_3fold_gives_ln3(self):
        """A symmetric 3-fold barrier gives TΔS = RT×ln(3) regardless of height."""
        expected = RT_STD * np.log(3)
        for V0 in [5.0, 11.38, 20.0, 40.0]:
            TdS, pops = rotamer_entropy_from_barrier(V0, 3)
            assert TdS == pytest.approx(expected, rel=0.01), \
                f"V0={V0}: expected {expected:.2f}, got {TdS:.2f}"

    def test_dme_barrier_value(self):
        """DME barrier (NIST: 11.38 kJ/mol, 3-fold) gives expected entropy."""
        TdS, pops = rotamer_entropy_from_barrier(11.38, 3)
        # 3-fold symmetric → RT×ln(3)
        assert TdS == pytest.approx(RT_STD * np.log(3), rel=0.01)
        assert len(pops) == 3
        assert all(abs(p - 1/3) < 0.01 for p in pops)

    def test_1fold_single_well(self):
        """1-fold barrier has only 1 well → TΔS ≈ 0."""
        TdS, pops = rotamer_entropy_from_barrier(20.0, 1)
        assert TdS < 0.1  # essentially zero
        assert len(pops) == 1

    def test_negative_barrier_raises(self):
        with pytest.raises(ValueError):
            rotamer_entropy_from_barrier(-5.0, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Linkage table physics constraints
# ═══════════════════════════════════════════════════════════════════════════

class TestLinkageTablePhysics:
    """Test that the linkage entropy table satisfies physical constraints."""

    def test_all_values_in_range(self):
        """All TΔS_freeze values in [1.5, 6.0] kJ/mol (convergence criterion)."""
        for key, le in LINKAGE_ENTROPY_TABLE.items():
            assert 1.5 <= le.TdS_freeze_kJmol <= 6.0, \
                f"{key}: TΔS={le.TdS_freeze_kJmol:.2f} outside [1.5, 6.0]"

    def test_ordering_1_6_gt_1_4(self):
        """α1→6 > α1→4 (extra ω torsion)."""
        assert get_TdS_freeze("a1-6") > get_TdS_freeze("a1-4")

    def test_ordering_1_4_gt_1_3(self):
        """α1→4 > α1→3 (more ψ flexibility)."""
        assert get_TdS_freeze("a1-4") > get_TdS_freeze("a1-3")

    def test_ordering_1_6_gt_all_others(self):
        """α1→6 has the highest entropy cost (3 torsions vs 2)."""
        a16 = get_TdS_freeze("a1-6")
        for key in ["b1-4", "b1-3", "a1-4", "a1-3", "a1-2", "a2-3"]:
            assert a16 > get_TdS_freeze(key), \
                f"a1-6 ({a16:.2f}) should be > {key} ({get_TdS_freeze(key):.2f})"

    def test_1_6_has_three_torsions(self):
        """1→6 linkage should have φ, ψ, AND ω."""
        le = LINKAGE_ENTROPY_TABLE["a1-6"]
        assert le.n_torsions == 3
        torsion_names = [t.name for t in le.torsions]
        assert "omega" in torsion_names

    def test_non_1_6_has_two_torsions(self):
        """Non-1→6 linkages should have exactly φ and ψ."""
        for key in ["b1-4", "b1-3", "a1-4", "a1-3", "a1-2", "a2-3"]:
            le = LINKAGE_ENTROPY_TABLE[key]
            assert le.n_torsions == 2, f"{key} has {le.n_torsions} torsions"

    def test_mean_below_mammen_consensus(self):
        """Mean per-linkage TΔS should be below Mammen 3.4×2=6.8 kJ/mol.
        
        Glycosidic torsions are pre-restricted by anomeric effect,
        so should have LESS entropy per torsion than generic rotors.
        """
        all_TdS = [le.TdS_freeze_kJmol for le in LINKAGE_ENTROPY_TABLE.values()]
        mean = np.mean(all_TdS)
        mammen_two_rotors = 3.4 * 2  # 6.8 kJ/mol for two standard rotors
        assert mean < mammen_two_rotors, \
            f"Mean {mean:.2f} should be < Mammen 2-rotor {mammen_two_rotors:.2f}"

    def test_per_torsion_below_eps_rotor(self):
        """Average per-torsion entropy should be below MABE eps_rotor (2.48 kJ/mol).
        
        Glycosidic torsions (restricted by anomeric effect) should have
        less entropy per torsion than generic rotatable bonds.
        """
        eps_rotor = 2.48  # MABE HG Phase 9 value
        # Exclude 1→6 (has ω which is NMR-calibrated, special case)
        two_torsion_TdS = [
            le.TdS_freeze_kJmol / 2.0
            for key, le in LINKAGE_ENTROPY_TABLE.items()
            if key != "a1-6"
        ]
        avg_per_torsion = np.mean(two_torsion_TdS)
        assert avg_per_torsion < eps_rotor, \
            f"Avg per torsion {avg_per_torsion:.2f} should be < eps_rotor {eps_rotor}"

    def test_omega_close_to_eps_rotor(self):
        """ω torsion (3 NMR rotamers) should be comparable to eps_rotor.
        
        ω is the least restricted glycosidic torsion (no anomeric effect),
        so its entropy should be closest to a generic rotor.
        """
        le = LINKAGE_ENTROPY_TABLE["a1-6"]
        omega_t = [t for t in le.torsions if t.name == "omega"][0]
        TdS_omega = rotamer_entropy(omega_t.populations)
        eps_rotor = 2.48
        # Should be within 50% of eps_rotor
        assert abs(TdS_omega - eps_rotor) / eps_rotor < 0.50, \
            f"ω TΔS ({TdS_omega:.2f}) too far from eps_rotor ({eps_rotor})"


# ═══════════════════════════════════════════════════════════════════════════
# Table completeness and access
# ═══════════════════════════════════════════════════════════════════════════

class TestLinkageTableCompleteness:
    """Test that all required linkage types are present and accessible."""

    REQUIRED_LINKAGES = ["b1-4", "b1-3", "a1-4", "a1-3", "a1-2", "a1-6", "a2-3"]

    def test_all_required_linkages_present(self):
        for lt in self.REQUIRED_LINKAGES:
            assert lt in LINKAGE_ENTROPY_TABLE, f"Missing linkage type: {lt}"

    def test_get_TdS_freeze_all_types(self):
        for lt in self.REQUIRED_LINKAGES:
            val = get_TdS_freeze(lt)
            assert isinstance(val, float)
            assert val > 0

    def test_unknown_linkage_raises(self):
        with pytest.raises(KeyError):
            get_TdS_freeze("x9-9")

    def test_all_torsions_have_sources(self):
        """Every torsion definition should have a non-empty source citation."""
        for key, le in LINKAGE_ENTROPY_TABLE.items():
            for t in le.torsions:
                assert len(t.source) > 10, \
                    f"{key}/{t.name}: missing source citation"

    def test_all_torsions_have_tiers(self):
        """Every torsion should have tier 1, 2, or 3."""
        for key, le in LINKAGE_ENTROPY_TABLE.items():
            for t in le.torsions:
                assert t.tier in [1, 2, 3], \
                    f"{key}/{t.name}: invalid tier {t.tier}"

    def test_omega_is_tier_1(self):
        """ω torsion populations are from NMR → should be Tier 1."""
        le = LINKAGE_ENTROPY_TABLE["a1-6"]
        omega_t = [t for t in le.torsions if t.name == "omega"][0]
        assert omega_t.tier == 1


# ═══════════════════════════════════════════════════════════════════════════
# Scoring function tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScoringFunction:
    """Test the glycan conformational entropy scoring function."""

    def test_single_linkage(self):
        """Single linkage gives that linkage's TΔS."""
        for lt in ["b1-4", "a1-3", "a1-6"]:
            expected = get_TdS_freeze(lt)
            assert score_conformational_entropy([lt]) == pytest.approx(expected)

    def test_additive_two_linkages(self):
        """Two linkages give sum of individual TΔS values."""
        lt1, lt2 = "b1-4", "a1-3"
        expected = get_TdS_freeze(lt1) + get_TdS_freeze(lt2)
        assert score_conformational_entropy([lt1, lt2]) == pytest.approx(expected)

    def test_empty_linkages_zero(self):
        """No linkages (monosaccharide) → zero conformational penalty."""
        assert score_conformational_entropy([]) == pytest.approx(0.0)

    def test_branch_penalty_additive(self):
        """Branch penalty adds on top of linkage entropy."""
        base = score_conformational_entropy(["b1-4"], n_branches=0)
        with_branch = score_conformational_entropy(["b1-4"], n_branches=1)
        assert with_branch > base
        assert with_branch - base == pytest.approx(get_branch_penalty(1))

    def test_chitobiose_vs_isomaltose(self):
        """Chitobiose (β1→4) should have lower entropy cost than isomaltose (α1→6).
        
        Same composition, different linkage. 1→6 has extra ω → more entropy to lose.
        This directly tests Prediction 4B from the glycan plan.
        """
        TdS_chitobiose = score_conformational_entropy(["b1-4"])
        TdS_isomaltose = score_conformational_entropy(["a1-6"])
        assert TdS_isomaltose > TdS_chitobiose

    def test_mannotriose_double_mannobiose(self):
        """Mannotriose (2 linkages) penalty ≈ 2× mannobiose (1 linkage).
        
        Additivity check for Prediction 4A.
        """
        TdS_mono_to_di = score_conformational_entropy(["a1-3"])
        TdS_mono_to_tri = score_conformational_entropy(["a1-3", "a1-3"])
        assert TdS_mono_to_tri == pytest.approx(2 * TdS_mono_to_di, rel=1e-10)

    def test_biantennary_core(self):
        """Biantennary N-glycan core has predictable total entropy.
        
        Core: GlcNAc-β1→4-GlcNAc-β1→4-Man + two branches (α1→3, α1→6).
        Total: 2× b1-4 + 1× a1-3 + 1× a1-6 + 1 branch.
        """
        linkages = ["b1-4", "b1-4", "a1-3", "a1-6"]
        total = score_conformational_entropy(linkages, n_branches=1)
        expected = (2 * get_TdS_freeze("b1-4") +
                    get_TdS_freeze("a1-3") +
                    get_TdS_freeze("a1-6") +
                    get_branch_penalty(1))
        assert total == pytest.approx(expected)
        # Should be between 8 and 15 kJ/mol for a 5-residue core
        assert 8.0 < total < 15.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration with glycan params
# ═══════════════════════════════════════════════════════════════════════════

class TestGlycanParamsIntegration:
    """Test integration with the glycan scorer parameter dict."""

    def test_update_populates_eps_conf(self):
        params = {"eps_conf": 0.0}
        updated = update_glycan_params_with_G2(params)
        assert updated["eps_conf"] > 0
        assert "eps_conf_source" in updated

    def test_update_includes_linkage_table(self):
        params = {}
        updated = update_glycan_params_with_G2(params)
        assert "linkage_entropy_table" in updated
        table = updated["linkage_entropy_table"]
        assert "b1-4" in table
        assert "a1-6" in table

    def test_mean_eps_conf_in_range(self):
        """Mean eps_conf should be in [1.5, 5.0] kJ/mol."""
        params = {}
        updated = update_glycan_params_with_G2(params)
        assert 1.5 <= updated["eps_conf"] <= 5.0

    def test_does_not_clobber_other_params(self):
        """G2 update should preserve existing parameters."""
        params = {"k_desolv_eq": 5.2, "eps_CH_pi": -2.9, "eps_HB": -6.0}
        updated = update_glycan_params_with_G2(params)
        assert updated["k_desolv_eq"] == 5.2
        assert updated["eps_CH_pi"] == -2.9
        assert updated["eps_HB"] == -6.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
