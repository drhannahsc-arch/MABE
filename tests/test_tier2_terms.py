"""
tests/test_tier2_terms.py — Tier 2 Interaction Terms Tests

1. Self-zero: all 10 terms return 0.0 for a bare UniversalComplex
2. Regression: existing 644 entries are unaffected (Tier 2 = 0.0 for all)
3. Smoke: known interactions produce physically reasonable values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge'))

import math
import pytest

from core.universal_schema import UniversalComplex
from core.tier2_terms import (
    compute_dispersion_upgraded, compute_cation_pi, compute_pi_stack,
    compute_halogen_bond, compute_salt_bridge, compute_born_solvation,
    compute_hbond_cooperativity, compute_anion_pi, compute_metallophilic,
    compute_group_desolvation, compute_water_penalty,
    compute_all_tier2, tier2_total, TIER2_RESULT_FIELDS,
)


# ═══════════════════════════════════════════════════════════════════════════
# MOCK RESULT — mimics PredictionResult with Tier 2 fields
# ═══════════════════════════════════════════════════════════════════════════

class MockResult:
    """Minimal mock of PredictionResult with Tier 2 fields."""
    def __init__(self):
        for f in TIER2_RESULT_FIELDS:
            setattr(self, f, 0.0)
        # Also mock existing fields that Tier 2 reads
        self.dg_hbond = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: SELF-ZERO — bare UC produces all zeros
# ═══════════════════════════════════════════════════════════════════════════

class TestSelfZero:
    """Every Tier 2 term must return 0.0 for a default UniversalComplex."""

    def setup_method(self):
        self.uc = UniversalComplex(name="bare_test")
        self.result = MockResult()

    def test_dispersion_self_zero(self):
        compute_dispersion_upgraded(self.uc, self.result)
        assert self.result.dg_dispersion_t2 == 0.0

    def test_cation_pi_self_zero(self):
        compute_cation_pi(self.uc, self.result)
        assert self.result.dg_cation_pi == 0.0

    def test_pi_stack_self_zero(self):
        compute_pi_stack(self.uc, self.result)
        assert self.result.dg_pi_stack == 0.0

    def test_halogen_bond_self_zero(self):
        compute_halogen_bond(self.uc, self.result)
        assert self.result.dg_halogen_bond == 0.0

    def test_salt_bridge_self_zero(self):
        compute_salt_bridge(self.uc, self.result)
        assert self.result.dg_salt_bridge == 0.0

    def test_born_solvation_self_zero(self):
        compute_born_solvation(self.uc, self.result)
        assert self.result.dg_born_solvation == 0.0

    def test_hbond_coop_self_zero(self):
        compute_hbond_cooperativity(self.uc, self.result)
        assert self.result.dg_hbond_coop == 0.0

    def test_anion_pi_self_zero(self):
        compute_anion_pi(self.uc, self.result)
        assert self.result.dg_anion_pi == 0.0

    def test_metallophilic_self_zero(self):
        compute_metallophilic(self.uc, self.result)
        assert self.result.dg_metallophilic == 0.0

    def test_group_desolv_self_zero(self):
        compute_group_desolvation(self.uc, self.result)
        assert self.result.dg_group_desolv == 0.0

    def test_all_tier2_self_zero(self):
        compute_all_tier2(self.uc, self.result)
        assert tier2_total(self.result) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: EXISTING ENTRY TYPES — metal, HG, CM all get zero Tier 2
# ═══════════════════════════════════════════════════════════════════════════

class TestExistingEntriesZero:
    """Tier 2 terms must be 0.0 for entries shaped like existing calibration data.

    These UC objects have the same field population as real CAL_DATA / HG_DATA /
    CROSS_MODAL_DATA entries — none have Tier 2 descriptor fields populated.
    """

    def test_metal_entry_zero(self):
        uc = UniversalComplex(
            name="Cu-EDTA",
            binding_mode="metal_coordination",
            metal_formula="Cu2+",
            donor_subtypes=["N_amine", "N_amine", "O_carboxylate", "O_carboxylate"],
            chelate_rings=5,
            log_Ka_exp=18.8,
        )
        result = MockResult()
        compute_all_tier2(uc, result)
        assert tier2_total(result) == 0.0

    def test_hg_entry_zero(self):
        uc = UniversalComplex(
            name="beta-CD:adamantane",
            binding_mode="host_guest_inclusion",
            host_name="beta-CD",
            guest_smiles="C1C2CC3CC1CC(C2)C3",
            guest_charge=0,
            n_hbonds_formed=0,
            log_Ka_exp=4.3,
        )
        result = MockResult()
        compute_all_tier2(uc, result)
        assert tier2_total(result) == 0.0

    def test_cm_entry_zero(self):
        uc = UniversalComplex(
            name="Na+@CB7",
            binding_mode="cross_modal",
            metal_formula="Na+",
            host_name="CB7",
            cavity_volume_A3=279.0,
            log_Ka_exp=3.2,
        )
        result = MockResult()
        compute_all_tier2(uc, result)
        assert tier2_total(result) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: SMOKE — known interactions produce expected ranges
# ═══════════════════════════════════════════════════════════════════════════

class TestSmoke:
    """Verify that populated Tier 2 fields produce physically reasonable values."""

    def test_cation_pi_reasonable(self):
        """NMe4+ near Trp indole → expect -4 to -10 kJ/mol."""
        uc = UniversalComplex(name="cation_pi_test")
        uc.n_cation_pi_contacts = 1
        uc.cation_pi_type = "organic_cation_indole"
        result = MockResult()
        compute_cation_pi(uc, result)
        assert -12.0 < result.dg_cation_pi < -2.0, \
            f"Cation-pi should be -2 to -12 kJ/mol, got {result.dg_cation_pi}"

    def test_pi_stack_reasonable(self):
        """One parallel displaced stacking contact → expect -2 to -5 kJ/mol."""
        uc = UniversalComplex(name="pi_stack_test")
        uc.n_pi_stack_contacts = 1
        uc.pi_stack_type = "parallel_displaced"
        result = MockResult()
        compute_pi_stack(uc, result)
        assert -6.0 < result.dg_pi_stack < -1.0

    def test_halogen_bond_iodine(self):
        """C-I···N halogen bond → expect -15 to -35 kJ/mol."""
        uc = UniversalComplex(name="xb_test")
        uc.n_halogen_bonds = 1
        uc.halogen_bond_type = "C-I"
        uc.halogen_bond_nucleophile = "N"
        uc.halogen_bond_angle = 175.0
        result = MockResult()
        compute_halogen_bond(uc, result)
        assert -35.0 < result.dg_halogen_bond < -10.0

    def test_halogen_bond_angle_kills(self):
        """Halogen bond at bad angle (<140°) → zero."""
        uc = UniversalComplex(name="xb_bad_angle")
        uc.n_halogen_bonds = 1
        uc.halogen_bond_type = "C-I"
        uc.halogen_bond_nucleophile = "N"
        uc.halogen_bond_angle = 120.0  # too bent
        result = MockResult()
        compute_halogen_bond(uc, result)
        assert result.dg_halogen_bond == 0.0

    def test_salt_bridge_reasonable(self):
        """One COO-/NH3+ salt bridge → expect -4 to -8 kJ/mol."""
        uc = UniversalComplex(name="sb_test", ionic_strength_M=0.15)
        uc.n_salt_bridges = 1
        uc.salt_bridge_z_product = -1
        result = MockResult()
        compute_salt_bridge(uc, result)
        assert -10.0 < result.dg_salt_bridge < -3.0

    def test_born_solvation_Na(self):
        """Na+ (r=1.02 Å, z=+1) → Born ΔG should be ~-375 kJ/mol."""
        uc = UniversalComplex(name="born_test")
        uc.guest_formal_charge = 1
        uc.guest_charge = 1
        uc.guest_ion_radius_A = 1.02
        uc.has_marcus_hydration_dg = False
        result = MockResult()
        compute_born_solvation(uc, result)
        # Born for Na+: -(1 * 694.3) / (1.02 + 0.72) * (1 - 1/78.4)
        #             = -694.3 / 1.74 * 0.9872 ≈ -394 kJ/mol
        assert -500.0 < result.dg_born_solvation < -300.0, \
            f"Born for Na+ should be ~-394 kJ/mol, got {result.dg_born_solvation}"

    def test_born_skips_when_marcus(self):
        """If Marcus data available, Born term should not fire (avoid double-count)."""
        uc = UniversalComplex(name="born_skip")
        uc.guest_formal_charge = 2
        uc.guest_charge = 2
        uc.guest_ion_radius_A = 0.73
        uc.has_marcus_hydration_dg = True
        result = MockResult()
        compute_born_solvation(uc, result)
        assert result.dg_born_solvation == 0.0

    def test_hbond_cooperativity(self):
        """Chain of 3 amide H-bonds → ~20% enhancement on each beyond first."""
        uc = UniversalComplex(name="coop_test")
        uc.max_hbond_chain_length = 3
        uc.hbond_chain_type = "amide"
        uc.n_hbonds_formed = 3
        result = MockResult()
        result.dg_hbond = -15.0  # 3 × -5 kJ/mol
        compute_hbond_cooperativity(uc, result)
        # coop = 0.20, chain=3 → correction = 0.20 × 2 × (-5) = -2.0
        assert -3.0 < result.dg_hbond_coop < -1.0

    def test_metallophilic_au_au(self):
        """One Au(I)···Au(I) contact → expect -30 to -50 kJ/mol."""
        uc = UniversalComplex(name="aurophilic_test")
        uc.n_d10_d10_contacts = 1
        uc.metallophilic_pair = ("Au", "Au")
        uc.metallophilic_distance_A = 3.0
        result = MockResult()
        compute_metallophilic(uc, result)
        assert -55.0 < result.dg_metallophilic < -20.0

    def test_group_desolvation(self):
        """Two primary OH groups fully buried → ~20 kJ/mol penalty."""
        uc = UniversalComplex(name="desolv_test")
        uc.buried_groups = [
            {"type": "OH_primary_eq", "burial_fraction": 1.0},
            {"type": "OH_primary_eq", "burial_fraction": 1.0},
        ]
        result = MockResult()
        compute_group_desolvation(uc, result)
        assert 15.0 < result.dg_group_desolv < 25.0  # positive = penalty

    def test_group_desolvation_partial(self):
        """Partially buried group → proportional cost."""
        uc = UniversalComplex(name="desolv_partial")
        uc.buried_groups = [
            {"type": "OH_primary_eq", "burial_fraction": 0.5},
        ]
        result = MockResult()
        compute_group_desolvation(uc, result)
        assert 4.0 < result.dg_group_desolv < 6.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: ADDITIVITY — multiple terms sum correctly
# ═══════════════════════════════════════════════════════════════════════════

class TestAdditivity:
    """Multiple Tier 2 terms should sum correctly via tier2_total()."""

    def test_multi_term_sum(self):
        uc = UniversalComplex(name="multi_test")
        uc.n_cation_pi_contacts = 1
        uc.cation_pi_type = "organic_cation_benzene"
        uc.n_salt_bridges = 1
        uc.salt_bridge_z_product = -1

        result = MockResult()
        compute_all_tier2(uc, result)

        total = tier2_total(result)
        assert total < 0.0, "Combined favorable interactions should be negative"
        # Each individually negative → sum should be more negative than either alone
        assert total == pytest.approx(
            result.dg_cation_pi + result.dg_salt_bridge, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════════════
# T11: WATER PENALTY (P20 back-solve)
# ═══════════════════════════════════════════════════════════════════════════

def _make_empty_uc():
    return UniversalComplex(name="water_penalty_test")

def _make_result():
    return MockResult()

def _sample_ucs():
    """Return a few sample UCs that represent existing calibration entries."""
    # Metal entry: has metal_formula but no SASA burial
    m = UniversalComplex(name="Cu-EDTA_test", binding_mode="metal",
                         metal_formula="Cu2+")
    m.donor_subtypes = ["O_carboxylate"] * 4
    # HG entry: has host but no polar SASA burial fields
    h = UniversalComplex(name="betaCD-adamantane_test", binding_mode="host_guest")
    h.host_name = "beta-CD"
    h.sasa_buried_A2 = 80.0
    # CM entry
    c = UniversalComplex(name="Cs@CB7_test", binding_mode="cross_modal")
    c.metal_formula = "Cs+"
    c.host_name = "CB[7]"
    return [m, h, c]


class TestWaterPenalty:
    """Tests for T11 compute_water_penalty (SASA-based water competition)."""

    def test_self_zero_no_data(self):
        """No SASA or H-bond count -> zero contribution."""
        uc = _make_empty_uc()
        result = _make_result()
        compute_water_penalty(uc, result)
        assert result.dg_water_penalty == 0.0

    def test_sasa_mode_oh(self):
        """OH burial via SASA produces positive penalty."""
        uc = _make_empty_uc()
        uc.sasa_oh_buried_A2 = 10.0  # ~1 OH group
        result = _make_result()
        compute_water_penalty(uc, result)
        # 10 A^2 x 1.097 kJ/mol/A^2 ~ 11.0 kJ/mol
        assert result.dg_water_penalty > 9.0
        assert result.dg_water_penalty < 13.0

    def test_sasa_mode_nh(self):
        """NH burial via SASA produces positive penalty."""
        uc = _make_empty_uc()
        uc.sasa_nh_buried_A2 = 18.0  # ~1 NH2 group
        result = _make_result()
        compute_water_penalty(uc, result)
        # 18 A^2 x 0.701 kJ/mol/A^2 ~ 12.6 kJ/mol
        assert result.dg_water_penalty > 10.0
        assert result.dg_water_penalty < 15.0

    def test_sasa_mode_o_acceptor(self):
        """O acceptor burial via SASA produces smaller penalty."""
        uc = _make_empty_uc()
        uc.sasa_o_acceptor_buried_A2 = 14.0  # ~1 C=O
        result = _make_result()
        compute_water_penalty(uc, result)
        # 14 A^2 x 0.210 kJ/mol/A^2 ~ 2.9 kJ/mol
        assert result.dg_water_penalty > 1.5
        assert result.dg_water_penalty < 5.0

    def test_hbond_count_mode(self):
        """H-bond count mode when no SASA data."""
        uc = _make_empty_uc()
        uc.n_water_hbonds_displaced = 3
        result = _make_result()
        compute_water_penalty(uc, result)
        # 3 x 5.2 = 15.6 kJ/mol
        assert abs(result.dg_water_penalty - 15.6) < 0.1

    def test_sasa_overrides_hbond_count(self):
        """SASA mode takes priority over H-bond count."""
        uc = _make_empty_uc()
        uc.sasa_oh_buried_A2 = 10.0
        uc.n_water_hbonds_displaced = 100  # should be ignored
        result = _make_result()
        compute_water_penalty(uc, result)
        # Should use SASA mode (~11 kJ), not hbond count mode (520 kJ)
        assert result.dg_water_penalty < 20.0

    def test_existing_entries_unaffected(self):
        """Existing metal/HG/CM entries have no SASA burial -> zero T11."""
        for uc in _sample_ucs():
            result = _make_result()
            compute_water_penalty(uc, result)
            assert result.dg_water_penalty == 0.0, \
                f"{uc.name}: dg_water_penalty should be 0 for existing entries"

    def test_combined_polar_sasa(self):
        """Multiple polar types sum correctly."""
        uc = _make_empty_uc()
        uc.sasa_oh_buried_A2 = 9.5    # 1 OH
        uc.sasa_nh_buried_A2 = 18.0   # 1 NH2
        uc.sasa_o_acceptor_buried_A2 = 14.0  # 1 C=O
        result = _make_result()
        compute_water_penalty(uc, result)
        # 9.5x1.097 + 18x0.701 + 14x0.210 ~ 10.4 + 12.6 + 2.9 = 25.9
        assert result.dg_water_penalty > 22.0
        assert result.dg_water_penalty < 30.0

    def test_in_tier2_total(self):
        """Water penalty included in tier2_total sum."""
        uc = _make_empty_uc()
        uc.n_water_hbonds_displaced = 2
        result = _make_result()
        compute_all_tier2(uc, result)
        total = tier2_total(result)
        assert total >= result.dg_water_penalty
        assert result.dg_water_penalty > 0
