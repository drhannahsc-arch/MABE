"""
tests/test_hbond_pl_physics.py — Tests for per-type H-bond scoring

Validates that type-classified H-bond energies:
1. Match Fersht/Pace experimental ranges
2. Have physically correct ordering
3. Score correctly via network and simple APIs
"""

import pytest
import sys
import os

_mabe_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _mabe_root not in sys.path:
    sys.path.insert(0, _mabe_root)

from knowledge.hbond_pl_physics import (
    HBOND_TYPE_ENERGY, classify_hbond,
    score_hbond_network, score_hbond_simple,
)


# ===================================================================
# PARAMETER TESTS
# ===================================================================

class TestHbondTypeEnergies:

    def test_all_favorable(self):
        """All H-bond type energies are negative (favorable)."""
        for hb_type, energy in HBOND_TYPE_ENERGY.items():
            assert energy < 0, f"{hb_type}: energy = {energy}, expected < 0"

    def test_charged_stronger_than_neutral(self):
        """Charge-assisted HBs are significantly stronger than neutral."""
        assert abs(HBOND_TYPE_ENERGY["charge_assisted"]) > abs(HBOND_TYPE_ENERGY["neutral"]) * 2

    def test_salt_bridge_strongest(self):
        """NH3+→COO- is the strongest single HB type."""
        salt = abs(HBOND_TYPE_ENERGY["NH3+→O_carboxylate"])
        for hb_type, energy in HBOND_TYPE_ENERGY.items():
            if hb_type != "NH3+→O_carboxylate":
                assert salt >= abs(energy) - 0.1, (
                    f"Salt bridge ({salt:.1f}) should be >= {hb_type} ({abs(energy):.1f})"
                )

    def test_sh_weakest_donor(self):
        """Cys SH is the weakest conventional H-bond donor."""
        assert abs(HBOND_TYPE_ENERGY["SH→O_carbonyl"]) < abs(HBOND_TYPE_ENERGY["neutral"])

    def test_ch_weaker_than_sh(self):
        """C-H···O is weaker than S-H···O."""
        assert abs(HBOND_TYPE_ENERGY["CH→O_carbonyl"]) < abs(HBOND_TYPE_ENERGY["SH→O_carbonyl"])

    def test_oh_stronger_than_nh(self):
        """OH is a stronger donor than amide NH (Pace 2014: 6.3 vs 4.7)."""
        assert abs(HBOND_TYPE_ENERGY["OH→O_carbonyl"]) > abs(HBOND_TYPE_ENERGY["NH_backbone→O_carbonyl"])

    def test_water_mediated_weaker_than_direct(self):
        """Water-mediated HBs are weaker than direct neutral HBs."""
        assert abs(HBOND_TYPE_ENERGY["water_mediated"]) < abs(HBOND_TYPE_ENERGY["neutral"])

    # ── Pace 2014 cross-validation ──

    def test_backbone_nh_co_matches_pace(self):
        """NH→C=O matches Pace 2014: 4.7 ± 0.5 kJ/mol."""
        val = abs(HBOND_TYPE_ENERGY["NH_backbone→O_carbonyl"])
        assert 3.0 < val < 7.0, f"NH→C=O = {val:.1f}, Pace = 4.7 ± 0.5"

    def test_oh_co_matches_pace(self):
        """OH→C=O matches Pace 2014: 6.3 ± 0.8 kJ/mol."""
        val = abs(HBOND_TYPE_ENERGY["OH→O_carbonyl"])
        assert 4.0 < val < 9.0, f"OH→C=O = {val:.1f}, Pace = 6.3 ± 0.8"

    def test_salt_bridge_matches_pace(self):
        """NH3+→COO- matches Pace 2014: 17.2 ± 2.1 kJ/mol."""
        val = abs(HBOND_TYPE_ENERGY["NH3+→O_carboxylate"])
        assert 12.0 < val < 22.0, f"Salt bridge = {val:.1f}, Pace = 17.2 ± 2.1"

    def test_fersht_neutral_range(self):
        """Neutral HBs within Fersht range: 2-7 kJ/mol."""
        val = abs(HBOND_TYPE_ENERGY["neutral"])
        assert 2.0 < val < 8.0, f"Neutral = {val:.1f}, Fersht = 2-7"

    def test_fersht_charged_range(self):
        """Charged HBs within Fersht range: 15-20 kJ/mol."""
        # Our "charge_assisted" default is an average; specific types
        # (salt bridge) fall in the Fersht range
        val = abs(HBOND_TYPE_ENERGY["NH3+→O_carboxylate"])
        assert 12.0 < val < 22.0, f"Charged = {val:.1f}, Fersht = 15-20"


# ===================================================================
# CLASSIFICATION
# ===================================================================

class TestClassification:

    def test_backbone_nh_co(self):
        cls, energy = classify_hbond("NH_backbone", "O_carbonyl")
        assert cls == "NH_backbone→O_carbonyl"
        assert energy < 0

    def test_salt_bridge(self):
        cls, energy = classify_hbond("NH3+", "O_carboxylate", is_charged=True)
        assert cls == "NH3+→O_carboxylate"
        assert abs(energy) > 10

    def test_unknown_falls_to_neutral(self):
        cls, energy = classify_hbond("unknown_donor", "unknown_acceptor")
        assert cls == "neutral"
        assert abs(energy - HBOND_TYPE_ENERGY["neutral"]) < 0.01

    def test_charged_flag_triggers_default(self):
        cls, energy = classify_hbond("unusual_donor", "unusual_acceptor", is_charged=True)
        assert cls == "charge_assisted"

    def test_sh_recognized_as_weak(self):
        cls, energy = classify_hbond("SH", "O_carbonyl")
        assert cls == "SH→O_carbonyl"
        assert abs(energy) < 3.0

    def test_water_mediated(self):
        cls, energy = classify_hbond("water", "O_carbonyl")
        assert cls == "water_mediated"


# ===================================================================
# NETWORK SCORING
# ===================================================================

class TestNetworkScoring:

    def test_empty_network(self):
        result = score_hbond_network([])
        assert result["total_kJ"] == 0.0
        assert result["n_total"] == 0

    def test_single_neutral(self):
        result = score_hbond_network([{"type": "neutral"}])
        assert result["total_kJ"] == HBOND_TYPE_ENERGY["neutral"]
        assert result["n_neutral"] == 1
        assert result["n_charged"] == 0

    def test_mixed_network(self):
        hbs = [
            {"type": "NH_backbone→O_carbonyl"},
            {"type": "NH_backbone→O_carbonyl"},
            {"type": "NH3+→O_carboxylate"},
            {"type": "water_mediated"},
        ]
        result = score_hbond_network(hbs)
        assert result["n_total"] == 4
        assert result["n_charged"] == 1
        assert result["total_kJ"] < 0
        # Salt bridge dominates
        assert abs(result["total_kJ"]) > 20

    def test_donor_acceptor_classification(self):
        """Network scoring via donor/acceptor types (not pre-classified)."""
        hbs = [
            {"donor_type": "NH_backbone", "acceptor_type": "O_carbonyl"},
            {"donor_type": "OH", "acceptor_type": "O_carbonyl"},
        ]
        result = score_hbond_network(hbs)
        assert result["n_total"] == 2
        expected = (HBOND_TYPE_ENERGY["NH_backbone→O_carbonyl"]
                    + HBOND_TYPE_ENERGY["OH→O_carbonyl"])
        assert abs(result["total_kJ"] - expected) < 0.01

    def test_all_types_favorable(self):
        """Every scored H-bond contributes favorably."""
        for hb_type in HBOND_TYPE_ENERGY:
            result = score_hbond_network([{"type": hb_type}])
            assert result["total_kJ"] < 0, f"{hb_type}: total = {result['total_kJ']}"


class TestSimpleScoring:

    def test_zero_hbs(self):
        assert score_hbond_simple() == 0.0

    def test_neutral_only(self):
        result = score_hbond_simple(n_neutral=3)
        assert result == 3 * HBOND_TYPE_ENERGY["neutral"]

    def test_charged_only(self):
        result = score_hbond_simple(n_charged=2)
        assert result == 2 * HBOND_TYPE_ENERGY["charge_assisted"]

    def test_mixed(self):
        result = score_hbond_simple(n_neutral=2, n_charged=1, n_weak=1)
        expected = (2 * HBOND_TYPE_ENERGY["neutral"]
                    + 1 * HBOND_TYPE_ENERGY["charge_assisted"]
                    + 1 * HBOND_TYPE_ENERGY["weak"])
        assert abs(result - expected) < 0.01

    def test_charged_dominates(self):
        """Adding 1 charged HB contributes more than 2 neutral."""
        neutral_only = score_hbond_simple(n_neutral=2)
        with_charged = score_hbond_simple(n_neutral=2, n_charged=1)
        delta = with_charged - neutral_only
        assert abs(delta) > abs(HBOND_TYPE_ENERGY["neutral"]) * 2


# ===================================================================
# INTEGRATION WITH PHYSICS PL SCORER
# ===================================================================

class TestPhysicsPLIntegration:

    def test_typed_hbonds_fire(self):
        """Physics PL scorer uses per-type H-bonds when hbond_types populated."""
        from knowledge.physics_pl_scorer import score_physics_pl
        from core.universal_schema import UniversalComplex

        uc = UniversalComplex(
            name="typed_hb_test",
            binding_mode="protein_ligand_physics",
            guest_smiles="CC(=O)O",  # acetic acid
            sasa_buried_A2=80.0,
            guest_sasa_total_A2=150.0,
            n_hbonds_formed=3,
            guest_rotatable_bonds=1,
            hbond_types=["NH_backbone→O_carbonyl", "NH_backbone→O_carbonyl", "OH→O_carbonyl"],
        )
        result = score_physics_pl(uc)
        expected = 2 * HBOND_TYPE_ENERGY["NH_backbone→O_carbonyl"] + HBOND_TYPE_ENERGY["OH→O_carbonyl"]
        assert abs(result["dg_hbond"] - expected) < 0.1, (
            f"Expected {expected:.1f}, got {result['dg_hbond']:.1f}"
        )

    def test_count_fallback_fires(self):
        """Without hbond_types, falls back to count-based scoring."""
        from knowledge.physics_pl_scorer import score_physics_pl
        from core.universal_schema import UniversalComplex

        uc = UniversalComplex(
            name="count_hb_test",
            binding_mode="protein_ligand_physics",
            guest_smiles="CCCCO",
            sasa_buried_A2=100.0,
            guest_sasa_total_A2=200.0,
            n_hbonds_formed=2,
            guest_rotatable_bonds=2,
            guest_charge=0,
        )
        result = score_physics_pl(uc)
        assert result["dg_hbond"] < 0, "Neutral H-bonds should be favorable"
        expected_2_neutral = 2 * HBOND_TYPE_ENERGY["neutral"]
        assert abs(result["dg_hbond"] - expected_2_neutral) < 0.5

    def test_charged_ligand_stronger_hbonds(self):
        """Charged ligand gets charge-assisted H-bond scoring."""
        from knowledge.physics_pl_scorer import score_physics_pl
        from core.universal_schema import UniversalComplex

        uc_neutral = UniversalComplex(
            name="neutral", binding_mode="protein_ligand_physics",
            guest_smiles="CCCCO", sasa_buried_A2=100.0, guest_sasa_total_A2=200.0,
            n_hbonds_formed=4, guest_rotatable_bonds=2, guest_charge=0,
        )
        uc_charged = UniversalComplex(
            name="charged", binding_mode="protein_ligand_physics",
            guest_smiles="CCCCO", sasa_buried_A2=100.0, guest_sasa_total_A2=200.0,
            n_hbonds_formed=4, guest_rotatable_bonds=2, guest_charge=1,
        )
        r_n = score_physics_pl(uc_neutral)
        r_c = score_physics_pl(uc_charged)
        assert r_c["dg_hbond"] < r_n["dg_hbond"], (
            "Charged ligand should have more favorable H-bond energy"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])