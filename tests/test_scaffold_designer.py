"""
test_scaffold_designer.py — Tests for scaffold designer Phases A-C.

Split into:
  - Physics tests (no rdkit): equations, pKa, requirements
  - Descriptor tests (rdkit required): 3D extraction
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scaffold_designer import (
    GuestSpec, ScaffoldDescriptor, ScaffoldRequirements, ScaffoldRanking,
    compute_scaffold_requirements, compute_thermo_score,
    donor_fraction_active, compute_pka_adjustment, effective_donor_count,
    rank_one_scaffold, rank_scaffolds_for_guest, gap_analysis,
    PKA_TABLE, RT_STD, EPS_ROTOR,
    K_SIZE_MISMATCH, K_CONVERGENCE_PENALTY, K_RIGID_BONUS,
)


# ═══════════════════════════════════════════════════════════════════════════
# GuestSpec
# ═══════════════════════════════════════════════════════════════════════════

class TestGuestSpec:

    def test_diameter_from_volume(self):
        """Volume → diameter auto-computed (sphere)."""
        g = GuestSpec(volume_A3=100.0)
        expected_d = 2.0 * (3.0 * 100.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        assert abs(g.diameter_A - expected_d) < 0.01

    def test_volume_from_diameter(self):
        g = GuestSpec(diameter_A=5.0)
        expected_v = (4.0 / 3.0) * math.pi * 2.5 ** 3
        assert abs(g.volume_A3 - expected_v) < 0.1

    def test_anion_auto_detected(self):
        g = GuestSpec(charge=-2)
        assert g.is_anion is True

    def test_neutral_not_anion(self):
        g = GuestSpec(charge=0)
        assert g.is_anion is False


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold Requirements (Phase B)
# ═══════════════════════════════════════════════════════════════════════════

class TestScaffoldRequirements:

    def test_d_arm_scales_with_guest_diameter(self):
        small = compute_scaffold_requirements(GuestSpec(diameter_A=3.0))
        large = compute_scaffold_requirements(GuestSpec(diameter_A=6.0))
        assert large.d_arm_required_A > small.d_arm_required_A

    def test_d_arm_includes_vdw_clearance(self):
        """d_arm = guest_d + 2×1.5 Å clearance."""
        req = compute_scaffold_requirements(GuestSpec(diameter_A=4.0))
        assert req.d_arm_required_A == pytest.approx(7.0)

    def test_rebek_cavity_volume(self):
        """V_cavity = V_guest / 0.55."""
        req = compute_scaffold_requirements(GuestSpec(volume_A3=55.0))
        assert req.V_cavity_required_A3 == pytest.approx(100.0)

    def test_convergence_always_true(self):
        req = compute_scaffold_requirements(GuestSpec())
        assert req.convergence_required is True

    def test_hb_complementarity(self):
        """Guest acceptors → receptor needs donors."""
        g = GuestSpec(n_hb_donors=2, n_hb_acceptors=3)
        req = compute_scaffold_requirements(g)
        assert req.min_hb_donors == 3   # complement guest's acceptors
        assert req.min_hb_acceptors == 2  # complement guest's donors

    def test_anion_prefers_cationic(self):
        g = GuestSpec(charge=-1)
        req = compute_scaffold_requirements(g)
        assert req.prefer_cationic is True

    def test_aromatic_guest_prefers_aromatic_walls(self):
        g = GuestSpec(n_aromatic_rings=2)
        req = compute_scaffold_requirements(g)
        assert req.prefer_aromatic_walls is True


# ═══════════════════════════════════════════════════════════════════════════
# Thermodynamic Scoring (Phase B)
# ═══════════════════════════════════════════════════════════════════════════

class TestThermoScoring:

    def _make_sd(self, **kwargs):
        defaults = dict(name="test", smiles="", category="linear", n_sites=2,
                        d_arm_A=6.0, n_rotors_bridge=2, rigidity_index=0.33,
                        V_cavity_est_A3=80.0, theta_conv_deg=60.0,
                        n_backbone_hb_donors=0, n_backbone_hb_acceptors=0)
        defaults.update(kwargs)
        return ScaffoldDescriptor(**defaults)

    def test_rigid_better_than_flexible(self):
        guest = GuestSpec(diameter_A=3.0, volume_A3=40.0)
        req = compute_scaffold_requirements(guest)
        rigid = self._make_sd(n_rotors_bridge=0, rigidity_index=1.0)
        flex = self._make_sd(n_rotors_bridge=6, rigidity_index=0.14)
        dG_rigid = compute_thermo_score(rigid, req, guest)
        dG_flex = compute_thermo_score(flex, req, guest)
        assert dG_rigid < dG_flex  # rigid is more favorable

    def test_size_match_better_than_mismatch(self):
        guest = GuestSpec(diameter_A=3.0)
        req = compute_scaffold_requirements(guest)
        # req.d_arm_required = 6.0 Å
        matched = self._make_sd(d_arm_A=6.0)
        mismatched = self._make_sd(d_arm_A=12.0)
        dG_match = compute_thermo_score(matched, req, guest)
        dG_mismatch = compute_thermo_score(mismatched, req, guest)
        assert dG_match < dG_mismatch

    def test_convergent_better_for_encapsulation(self):
        guest = GuestSpec(diameter_A=3.0)
        req = compute_scaffold_requirements(guest)
        assert req.convergence_required
        conv = self._make_sd(theta_conv_deg=60.0)
        div = self._make_sd(theta_conv_deg=150.0)
        dG_conv = compute_thermo_score(conv, req, guest)
        dG_div = compute_thermo_score(div, req, guest)
        assert dG_conv < dG_div  # convergent is better

    def test_flexibility_penalty_proportional_to_rotors(self):
        guest = GuestSpec(diameter_A=3.0)
        req = compute_scaffold_requirements(guest)
        sd2 = self._make_sd(n_rotors_bridge=2, rigidity_index=0.33)
        sd4 = self._make_sd(n_rotors_bridge=4, rigidity_index=0.2)
        dG2 = compute_thermo_score(sd2, req, guest)
        dG4 = compute_thermo_score(sd4, req, guest)
        diff = dG4 - dG2
        expected_diff = EPS_ROTOR * 2  # 2 extra rotors
        assert abs(diff - expected_diff) < 0.5  # allow rigidity index diff

    def test_aromatic_bonus_for_aromatic_guest(self):
        guest = GuestSpec(diameter_A=3.0, n_aromatic_rings=2)
        req = compute_scaffold_requirements(guest)
        aromatic = self._make_sd(category="aromatic")
        linear = self._make_sd(category="linear")
        dG_arom = compute_thermo_score(aromatic, req, guest)
        dG_lin = compute_thermo_score(linear, req, guest)
        assert dG_arom < dG_lin

    def test_hb_donors_help_anion_binding(self):
        """Scaffold with NH donors should score better for anion guest."""
        guest = GuestSpec(charge=-2, n_hb_acceptors=3)
        req = compute_scaffold_requirements(guest)
        with_donors = self._make_sd(n_backbone_hb_donors=3)
        no_donors = self._make_sd(n_backbone_hb_donors=0)
        dG_with = compute_thermo_score(with_donors, req, guest)
        dG_without = compute_thermo_score(no_donors, req, guest)
        assert dG_with < dG_without


# ═══════════════════════════════════════════════════════════════════════════
# pKa Adjustment (Phase C)
# ═══════════════════════════════════════════════════════════════════════════

class TestPKa:

    def test_carboxylate_active_at_neutral_pH(self):
        f = donor_fraction_active("O_carboxylate", 7.4)
        assert f > 0.99

    def test_carboxylate_partially_active_at_pH4(self):
        f = donor_fraction_active("O_carboxylate", 4.0)
        assert 0.4 < f < 0.6  # pKa = 4.0, so ~50%

    def test_carboxylate_inactive_at_pH2(self):
        f = donor_fraction_active("O_carboxylate", 2.0)
        assert f < 0.02

    def test_amine_mostly_protonated_at_neutral_pH(self):
        """Amine (pKa 10) is mostly protonated at pH 7 → poor for coordination."""
        f = donor_fraction_active("N_amine", 7.0)
        assert f < 0.01  # ~0.001, only 0.1% free base

    def test_amine_inactive_at_pH12(self):
        """Amine protonated → can't coordinate at very high pH? 
        No — amine is active (unprotonated) at high pH. At low pH, protonated → inactive."""
        f_low = donor_fraction_active("N_amine", 3.0)
        f_high = donor_fraction_active("N_amine", 12.0)
        # At pH 3: amine is protonated (RNH3+) → inactive for coordination
        # Wait — direction is "protonates" meaning active = unprotonated
        # At pH 3 << pKa 10: 1/(1+10^(3-10)) = 1/(1+10^-7) ≈ 1.0
        # That's wrong for coordination... 
        # Actually: "protonates" in our convention means the INACTIVE form is protonated
        # f = 1/(1+10^(pH-pKa)). At pH 3, pKa 10: f = 1/(1+10^-7) ≈ 1.0
        # This is correct: at pH 3, amine IS protonated (RNH3+), but our formula gives f≈1
        # because the formula says "fraction in UNprotonated form" = 1/(1+10^(pH-pKa))
        # At pH 3, pKa 10: exponent = 3-10 = -7, f = 1/(1+10^-7) ≈ 1.0
        # This is the DEPROTONATED fraction — which is the free base, active for coordination
        # But at pH 3, the amine IS protonated... so f should be LOW
        # 
        # The issue: for "protonates" direction, the formula should be
        # f_active = 1/(1+10^(pKa-pH)) (fraction in unprotonated/active form)
        # At pH 3, pKa 10: 1/(1+10^7) ≈ 0. That's correct.
        #
        # Let me check what the code actually does...
        # Code says: protonates → f = 1/(1+10^(pH-pKa))
        # pH=3, pKa=10: 1/(1+10^(3-10)) = 1/(1+10^-7) ≈ 1.0
        # That means at pH 3, amine is 100% active — WRONG for metal coordination
        # The formula is inverted.
        #
        # Fix: protonates should use f = 1/(1+10^(pKa-pH))
        # pH 3, pKa 10: 1/(1+10^7) ≈ 0 — correct (protonated, inactive)
        # pH 12, pKa 10: 1/(1+10^-2) ≈ 0.99 — correct (free base, active)
        assert f_high > f_low  # high pH amine is free base → active

    def test_pyridine_half_active_at_pka(self):
        """At pH = pKa, fraction active = 0.5."""
        f = donor_fraction_active("N_pyridine", 5.2)
        assert abs(f - 0.5) < 0.01

    def test_thioether_always_active(self):
        for pH in [1.0, 5.0, 7.0, 14.0]:
            assert donor_fraction_active("S_thioether", pH) == 1.0

    def test_unknown_donor_always_active(self):
        assert donor_fraction_active("UNKNOWN_TYPE", 5.0) == 1.0

    def test_phenolate_dead_at_low_pH(self):
        f = donor_fraction_active("O_phenolate", 3.0)
        assert f < 0.001

    def test_phenolate_active_at_high_pH(self):
        f = donor_fraction_active("O_phenolate", 12.0)
        assert f > 0.99

    def test_hydroxyl_always_active(self):
        """Hydroxyl (ROH) is always active (pKa too high to matter)."""
        for pH in [3.0, 7.0, 12.0]:
            f = donor_fraction_active("O_hydroxyl", pH)
            assert f == 1.0

    def test_pyrrole_always_active(self):
        """Pyrrole NH is always active (pKa too high to matter)."""
        for pH in [3.0, 7.0, 12.0]:
            f = donor_fraction_active("N_pyrrole", pH)
            assert f == 1.0

    def test_guanidinium_always_active(self):
        """Guanidinium (pKa 13.5) is always active at practical pH."""
        for pH in [3.0, 7.0, 11.0]:
            f = donor_fraction_active("N_guanidinium", pH)
            assert f == 1.0

    def test_compute_pka_adjustment_penalty_positive(self):
        """pKa penalty is always >= 0."""
        donors = ["O_carboxylate", "N_amine"]
        dG, _ = compute_pka_adjustment(donors, 7.0)
        assert dG >= 0.0

    def test_severe_penalty_at_wrong_pH(self):
        """Phenolate at pH 3 should have huge penalty."""
        dG, fracs = compute_pka_adjustment(["O_phenolate"], 3.0)
        assert dG > 15.0  # severe

    def test_effective_donor_count(self):
        """2 carboxylates (active at pH 7) + 1 amine (mostly protonated at pH 7)."""
        donors = ["O_carboxylate", "O_carboxylate", "N_amine"]
        n = effective_donor_count(donors, 7.0)
        # Carboxylates ~1.0 each, amine ~0.001 → total ~2.0
        assert 1.9 < n < 2.1

    def test_effective_donors_reduced_at_low_pH(self):
        """Carboxylate donors lose effectiveness at low pH."""
        donors = ["O_carboxylate", "O_carboxylate"]
        n_neutral = effective_donor_count(donors, 7.0)
        n_acid = effective_donor_count(donors, 2.0)
        assert n_neutral > n_acid


# ═══════════════════════════════════════════════════════════════════════════
# Combined Ranking
# ═══════════════════════════════════════════════════════════════════════════

class TestRanking:

    def _make_sd(self, name, **kwargs):
        defaults = dict(smiles="", category="linear", n_sites=2,
                        d_arm_A=6.0, n_rotors_bridge=2, rigidity_index=0.33,
                        V_cavity_est_A3=80.0, theta_conv_deg=60.0,
                        n_backbone_hb_donors=0, n_backbone_hb_acceptors=0)
        defaults.update(kwargs)
        return ScaffoldDescriptor(name=name, **defaults)

    def test_rank_one_returns_ranking(self):
        sd = self._make_sd("test")
        guest = GuestSpec(diameter_A=3.0)
        r = rank_one_scaffold(sd, guest)
        assert isinstance(r, ScaffoldRanking)
        assert r.dG_total != 0.0

    def test_rank_scaffolds_sorted(self):
        sds = [
            self._make_sd("rigid", n_rotors_bridge=0, rigidity_index=1.0),
            self._make_sd("flex", n_rotors_bridge=8, rigidity_index=0.11),
            self._make_sd("medium", n_rotors_bridge=3, rigidity_index=0.25),
        ]
        guest = GuestSpec(diameter_A=3.0)
        ranked = rank_scaffolds_for_guest(guest, sds)
        # Best (lowest dG) should be first
        for i in range(len(ranked) - 1):
            assert ranked[i].dG_total <= ranked[i + 1].dG_total

    def test_gap_analysis_flags(self):
        sds = [
            self._make_sd("too_small", d_arm_A=2.0),
            self._make_sd("too_big", d_arm_A=20.0),
            self._make_sd("ok", d_arm_A=6.0),
        ]
        guest = GuestSpec(diameter_A=3.0)  # req d_arm = 6.0
        ranked = rank_scaffolds_for_guest(guest, sds)
        gaps = gap_analysis(ranked, guest)
        assert "too_small" in gaps or "too_large" in gaps

    def test_pka_affects_ranking(self):
        """Same scaffold, different pH → different score."""
        sd = self._make_sd("test_pka")
        g_neutral = GuestSpec(diameter_A=3.0, pH=7.0)
        g_acid = GuestSpec(diameter_A=3.0, pH=2.0)
        r_neutral = rank_one_scaffold(sd, g_neutral, arm_donor_types=["O_carboxylate"])
        r_acid = rank_one_scaffold(sd, g_acid, arm_donor_types=["O_carboxylate"])
        # At pH 2, carboxylate is protonated → penalty
        assert r_acid.dG_pka > r_neutral.dG_pka

    def test_selenite_example(self):
        """Selenite at pH 5 should prefer convergent, H-bond-rich scaffolds."""
        selenite = GuestSpec(name="selenite", diameter_A=3.5, volume_A3=45.0,
                             charge=-2, n_hb_acceptors=3, pH=5.0)
        good = self._make_sd("good_receptor",
                             d_arm_A=6.5, theta_conv_deg=50.0,
                             rigidity_index=0.8, n_rotors_bridge=1,
                             n_backbone_hb_donors=3, category="aromatic")
        bad = self._make_sd("bad_receptor",
                            d_arm_A=15.0, theta_conv_deg=160.0,
                            rigidity_index=0.1, n_rotors_bridge=8,
                            n_backbone_hb_donors=0)
        r_good = rank_one_scaffold(good, selenite)
        r_bad = rank_one_scaffold(bad, selenite)
        assert r_good.dG_total < r_bad.dG_total


# ═══════════════════════════════════════════════════════════════════════════
# Descriptor extraction (rdkit required)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@pytest.mark.skipif(not HAS_RDKIT, reason="Requires rdkit")
class TestDescriptorExtraction:

    def test_extract_ethylenediamine(self):
        from core.scaffold_designer import extract_descriptor
        from core.de_novo_generator import BACKBONE_LIBRARY
        en = [b for b in BACKBONE_LIBRARY if b.name == "ethylenediamine"][0]
        sd = extract_descriptor(en)
        assert sd.name == "ethylenediamine"
        assert sd.n_sites == 2
        assert sd.d_arm_A > 2.0   # at least 2 Å between sites
        assert sd.d_arm_A < 10.0  # not unreasonably large
        assert sd.rigidity_index > 0  # has some rotors
        assert sd.mw > 0

    def test_extract_all(self):
        from core.scaffold_designer import extract_all_descriptors
        descs = extract_all_descriptors()
        assert len(descs) >= 50  # we have 65 backbones
        # All should have names
        names = [d.name for d in descs]
        assert "ethylenediamine" in names

    def test_aromatic_more_rigid(self):
        from core.scaffold_designer import extract_descriptor
        from core.de_novo_generator import BACKBONE_LIBRARY
        en = [b for b in BACKBONE_LIBRARY if b.name == "ethylenediamine"][0]
        pyr = [b for b in BACKBONE_LIBRARY if b.name == "2,6-disubstituted-pyridine"][0]
        sd_en = extract_descriptor(en)
        sd_pyr = extract_descriptor(pyr)
        assert sd_pyr.rigidity_index >= sd_en.rigidity_index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
