"""
tests/test_electrostatic_solvation.py — Tests for electrostatic solvation model

Validates:
1. Born model reproduces Marcus 1994 experimental hydration energies
2. Group-additive solvation has correct signs and ordering
3. Effective dielectric model behaves physically
4. Integration with physics PL scorer
"""

import pytest
import sys
import os
import math

_mabe_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _mabe_root not in sys.path:
    sys.path.insert(0, _mabe_root)

from knowledge.electrostatic_solvation import (
    born_solvation_energy, born_desolvation_cost,
    MARCUS_BORN_RADII, MARCUS_DG_HYDR, BORN_FACTOR, EPSILON_WATER,
    GROUP_ELEC_SOLVATION, compute_group_elec_desolvation,
    effective_dielectric, compute_electrostatic_desolvation,
)


# ===================================================================
# BORN MODEL TESTS
# ===================================================================

class TestBornModel:

    def test_born_energy_negative_in_water(self):
        """Born solvation in water is favorable (negative)."""
        dG = born_solvation_energy(1, 3.0, 1.0, 78.4)
        assert dG < 0

    def test_born_zero_for_neutral(self):
        """Zero charge → zero Born energy."""
        assert born_solvation_energy(0, 3.0) == 0.0

    def test_born_zero_for_zero_radius(self):
        """Zero radius → zero (guard)."""
        assert born_solvation_energy(1, 0.0) == 0.0

    def test_born_scales_with_charge_squared(self):
        """Divalent ion has 4× the Born energy of monovalent."""
        dG1 = born_solvation_energy(1, 3.0, 1.0, 78.4)
        dG2 = born_solvation_energy(2, 3.0, 1.0, 78.4)
        assert abs(dG2 / dG1 - 4.0) < 0.01

    def test_born_inversely_proportional_to_radius(self):
        """Smaller ion has more negative (stronger) solvation."""
        dG_small = born_solvation_energy(1, 2.0, 1.0, 78.4)
        dG_large = born_solvation_energy(1, 4.0, 1.0, 78.4)
        assert abs(dG_small) > abs(dG_large)
        assert abs(dG_small / dG_large - 2.0) < 0.01

    def test_marcus_radii_reproduce_hydration(self):
        """Marcus effective radii reproduce experimental ΔG_hydr exactly."""
        ION_CHARGES = {
            "Li+": 1, "Na+": 1, "K+": 1, "NH4+": 1,
            "Mg2+": 2, "Ca2+": 2, "Zn2+": 2, "Fe3+": 3,
            "F-": -1, "Cl-": -1, "Br-": -1, "I-": -1,
            "COO-": -1, "RNH3+": 1,
        }
        for ion, dG_exp in MARCUS_DG_HYDR.items():
            r_eff = MARCUS_BORN_RADII.get(ion)
            if r_eff is None:
                continue
            q = ION_CHARGES.get(ion, 0)
            if q == 0:
                continue
            dG_born = born_solvation_energy(q, r_eff, 1.0, EPSILON_WATER)
            assert abs(dG_born - dG_exp) < abs(dG_exp) * 0.02, (
                f"{ion}: Born={dG_born:.0f}, Marcus={dG_exp}, err={abs(dG_born-dG_exp):.0f}"
            )


class TestBornDesolvation:

    def test_desolvation_always_positive(self):
        """Burying a charge always costs energy (positive)."""
        for ion in ["Na+", "Cl-", "COO-", "RNH3+"]:
            cost = born_desolvation_cost(ion, epsilon_pocket=8.0)
            assert cost > 0, f"{ion}: desolvation cost = {cost:.1f}, expected > 0"

    def test_divalent_more_costly_than_monovalent(self):
        """Burying Mg2+ costs more than Na+."""
        cost_na = born_desolvation_cost("Na+", 8.0)
        cost_mg = born_desolvation_cost("Mg2+", 8.0)
        assert cost_mg > cost_na * 2

    def test_lower_dielectric_higher_cost(self):
        """Lower pocket ε → higher desolvation cost."""
        cost_8 = born_desolvation_cost("Na+", 8.0)
        cost_4 = born_desolvation_cost("Na+", 4.0)
        assert cost_4 > cost_8

    def test_high_dielectric_low_cost(self):
        """Near-water dielectric → near-zero cost."""
        cost = born_desolvation_cost("Na+", 60.0)
        assert cost < 10, f"ε=60 cost = {cost:.1f}, expected < 10 kJ/mol"

    def test_unknown_ion_returns_zero(self):
        """Unknown ion type returns zero (no crash)."""
        assert born_desolvation_cost("XYZ", 8.0) == 0.0


# ===================================================================
# GROUP-ADDITIVE TESTS
# ===================================================================

class TestGroupAdditive:

    def test_all_solvation_favorable(self):
        """All group electrostatic solvation values are negative."""
        for group, dG in GROUP_ELEC_SOLVATION.items():
            assert dG < 0, f"{group}: ΔG = {dG}, expected < 0"

    def test_charged_groups_strongest(self):
        """Charged groups have the largest (most negative) solvation."""
        charged = [abs(GROUP_ELEC_SOLVATION[g]) for g in
                   ["NH3+", "COO-", "guanidinium+", "phosphate2-"]]
        neutral = [abs(GROUP_ELEC_SOLVATION[g]) for g in
                   ["OH", "C=O_amide", "NH_amide"]]
        assert min(charged) > max(neutral) * 3

    def test_oh_more_polar_than_sh(self):
        """|ΔG_elec(OH)| > |ΔG_elec(SH)|: OH is more polar."""
        assert abs(GROUP_ELEC_SOLVATION["OH"]) > abs(GROUP_ELEC_SOLVATION["SH"])

    def test_amide_carbonyl_more_polar_than_ester(self):
        """Amide C=O more polar than ester C=O."""
        assert abs(GROUP_ELEC_SOLVATION["C=O_amide"]) > abs(GROUP_ELEC_SOLVATION["C=O_ester"])

    def test_desolvation_cost_positive(self):
        """Burying polar groups has positive cost."""
        groups = ["OH", "NH_amide", "C=O_amide"]
        cost = compute_group_elec_desolvation(groups, epsilon_pocket=8.0)
        assert cost > 0, f"Group desolvation cost = {cost:.1f}, expected > 0"

    def test_empty_groups_zero(self):
        """No groups → zero cost."""
        assert compute_group_elec_desolvation([]) == 0.0

    def test_charged_group_high_cost(self):
        """Burying a charge has very high cost."""
        charged_cost = compute_group_elec_desolvation(["COO-"], 8.0)
        neutral_cost = compute_group_elec_desolvation(["OH"], 8.0)
        assert charged_cost > neutral_cost * 5

    def test_burial_fraction_scales(self):
        """Partial burial reduces cost proportionally."""
        full = compute_group_elec_desolvation(
            [{"type": "OH", "burial_fraction": 1.0}], 8.0)
        half = compute_group_elec_desolvation(
            [{"type": "OH", "burial_fraction": 0.5}], 8.0)
        assert abs(half / full - 0.5) < 0.01


# ===================================================================
# DIELECTRIC MODEL TESTS
# ===================================================================

class TestEffectiveDielectric:

    def test_surface_high_dielectric(self):
        """Surface (burial=0) has high ε (near water)."""
        eps = effective_dielectric(0.0)
        assert eps > 30

    def test_buried_low_dielectric(self):
        """Fully buried (burial=1) has low ε."""
        eps = effective_dielectric(1.0)
        assert eps < 6

    def test_monotonically_decreasing(self):
        """ε decreases monotonically with burial fraction."""
        prev = effective_dielectric(0.0)
        for f in [0.2, 0.4, 0.6, 0.8, 1.0]:
            curr = effective_dielectric(f)
            assert curr <= prev, f"ε({f}) = {curr:.1f} > ε({f-0.2:.1f}) = {prev:.1f}"
            prev = curr

    def test_clamped_input(self):
        """Out-of-range inputs are clamped."""
        assert effective_dielectric(-0.5) == effective_dielectric(0.0)
        assert effective_dielectric(1.5) == effective_dielectric(1.0)


# ===================================================================
# INTEGRATION TESTS
# ===================================================================

class TestCombinedScoring:

    def test_neutral_ligand_small_cost(self):
        """Neutral ligand with polar groups has moderate cost."""
        groups = [
            {"type": "OH", "burial_fraction": 0.8},
            {"type": "C=O_amide", "burial_fraction": 0.5},
        ]
        cost = compute_electrostatic_desolvation(groups, burial_fraction=0.6)
        assert 0 < cost < 20, f"Neutral ligand cost = {cost:.1f}"

    def test_charged_ligand_from_uc(self):
        """Charged UC triggers Born desolvation."""
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(
            name="test_charged",
            binding_mode="protein_ligand_physics",
            guest_smiles="CC(=O)[O-]",  # acetate
            guest_charge=-1,
            sasa_buried_A2=100.0,
            guest_sasa_total_A2=150.0,
        )
        cost = compute_electrostatic_desolvation(uc, burial_fraction=0.7)
        assert cost > 10, f"Charged ligand cost = {cost:.1f}, expected > 10"


class TestPhysicsPLIntegration:

    def test_born_solvation_fires(self):
        """Physics PL scorer populates dg_born_solvation for charged ligand."""
        from knowledge.physics_pl_scorer import score_physics_pl
        from core.universal_schema import UniversalComplex

        uc = UniversalComplex(
            name="charged_test",
            binding_mode="protein_ligand_physics",
            guest_smiles="CC(=O)[O-]",
            guest_charge=-1,
            sasa_buried_A2=100.0,
            guest_sasa_total_A2=150.0,
            n_hbonds_formed=2,
            guest_rotatable_bonds=1,
        )
        result = score_physics_pl(uc)
        assert result["dg_born_solvation"] > 0, (
            f"Born solvation should be positive (unfavorable) for charged ligand, "
            f"got {result['dg_born_solvation']:.2f}"
        )

    def test_neutral_ligand_small_born(self):
        """Neutral ligand has zero or small Born solvation."""
        from knowledge.physics_pl_scorer import score_physics_pl
        from core.universal_schema import UniversalComplex

        uc = UniversalComplex(
            name="neutral_test",
            binding_mode="protein_ligand_physics",
            guest_smiles="CCCCO",
            guest_charge=0,
            sasa_buried_A2=100.0,
            guest_sasa_total_A2=200.0,
            n_hbonds_formed=1,
            guest_rotatable_bonds=3,
        )
        result = score_physics_pl(uc)
        # Neutral ligand: Born should be zero (no formal charge)
        # Group-additive may fire if buried_groups populated
        assert result["dg_born_solvation"] >= 0

    def test_five_term_decomposition(self):
        """Physics PL scorer terms sum to total."""
        from knowledge.physics_pl_scorer import score_physics_pl
        from core.universal_schema import UniversalComplex

        uc = UniversalComplex(
            name="five_term_test",
            binding_mode="protein_ligand_physics",
            guest_smiles="CC(=O)[O-]",
            guest_charge=-1,
            guest_mw=59.0,
            sasa_buried_A2=120.0,
            guest_sasa_total_A2=180.0,
            guest_sasa_nonpolar_A2=80.0,
            n_hbonds_formed=3,
            guest_rotatable_bonds=2,
        )
        r = score_physics_pl(uc)
        expected = (r["dg_desolv"] + r["dg_hydrophobic"] + r["dg_hbond"]
                    + r["dg_conf_entropy"] + r["dg_born_solvation"]
                    + r.get("dg_mixing_entropy", 0.0)
                    + r.get("dg_dispersion", 0.0)
                    + r.get("dg_pocket_desolv", 0.0)
                    + r.get("dg_water_displacement", 0.0)
                    + r.get("dg_preorganization", 0.0))
        assert abs(r["dg_total"] - expected) < 0.01, (
            f"Sum ({expected:.2f}) != total ({r['dg_total']:.2f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
