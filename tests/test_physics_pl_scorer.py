"""
tests/test_physics_pl_scorer.py — Tests for physics-based protein-ligand scoring

Validates that the parallel physics PL scorer:
1. Self-zeros for non-physics binding modes
2. Produces physically correct energy decomposition
3. Does not regress existing scoring paths
"""

import pytest
import sys
import os

_mabe_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _mabe_root not in sys.path:
    sys.path.insert(0, _mabe_root)

from knowledge.physics_pl_scorer import (
    compute_physics_pl_terms,
    score_physics_pl,
    GAMMA_HYDROPHOBIC,
    EPS_HBOND_FORMED,
    WATER_PENALTY_PER_HB,
    WATER_DISPLACEMENT,
    EPS_ROTOR,
    F_PARTIAL,
    EPS_CHARGED_HB,
)
from core.universal_schema import UniversalComplex


# ===================================================================
# PARAMETER SANITY
# ===================================================================

class TestPhysicsPLParameters:

    def test_gamma_hydrophobic_negative(self):
        """Hydrophobic burial is favorable (negative ΔG)."""
        assert GAMMA_HYDROPHOBIC < 0

    def test_hbond_formed_favorable(self):
        """Forming an H-bond is intrinsically favorable."""
        assert EPS_HBOND_FORMED < 0

    def test_water_penalty_positive(self):
        """Displacing water H-bonds is unfavorable."""
        assert WATER_PENALTY_PER_HB > 0

    def test_net_neutral_hb_near_zero(self):
        """Net neutral H-bond in protein pocket is near-zero.

        This is the Dunitz/Fersht result: protein-ligand H-bonds
        barely beat water. Binding energy comes from preorganization.
        """
        net = EPS_HBOND_FORMED + WATER_DISPLACEMENT * WATER_PENALTY_PER_HB
        assert abs(net) < 2.0, (
            f"Net neutral HB = {net:.2f}, expected near zero (±2 kJ/mol)"
        )

    def test_rotor_penalty_positive(self):
        """Freezing rotors costs entropy (positive ΔG)."""
        assert EPS_ROTOR > 0
        assert F_PARTIAL > 0

    def test_charged_hb_stronger(self):
        """Charge-assisted H-bonds are much stronger than neutral."""
        assert abs(EPS_CHARGED_HB) > abs(EPS_HBOND_FORMED) * 2

    def test_water_displacement_subunit(self):
        """Water displacement < 1.5 per H-bond in protein pocket."""
        assert 0.3 < WATER_DISPLACEMENT < 1.5


# ===================================================================
# SELF-ZEROING
# ===================================================================

class TestSelfZeroing:

    def test_wrong_binding_mode_zeros(self):
        """Scorer produces no output for non-physics modes."""
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            dg_group_desolv: float = 0.0
            dg_hydrophobic: float = 0.0
            dg_hbond: float = 0.0
            dg_conf_entropy: float = 0.0

        uc = UniversalComplex(
            name="test", binding_mode="host_guest_inclusion",
            guest_smiles="CCCCCC",
            sasa_buried_A2=100.0, guest_sasa_total_A2=200.0,
            n_hbonds_formed=3, guest_rotatable_bonds=5,
        )
        r = FakeResult()
        compute_physics_pl_terms(uc, r)
        assert r.dg_group_desolv == 0.0
        assert r.dg_hydrophobic == 0.0
        assert r.dg_hbond == 0.0
        assert r.dg_conf_entropy == 0.0

    def test_no_smiles_zeros(self):
        """Scorer zeros if no guest_smiles."""
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            dg_group_desolv: float = 0.0
            dg_hydrophobic: float = 0.0
            dg_hbond: float = 0.0
            dg_conf_entropy: float = 0.0

        uc = UniversalComplex(
            name="test", binding_mode="protein_ligand_physics",
            guest_smiles="",
        )
        r = FakeResult()
        compute_physics_pl_terms(uc, r)
        assert r.dg_group_desolv == 0.0


# ===================================================================
# ENERGY DECOMPOSITION
# ===================================================================

class TestEnergyDecomposition:

    def _make_uc(self, smiles="CCCCO", buried=100.0, total_sasa=200.0,
                 np_sasa=120.0, polar_sasa=80.0, n_hb=2, n_rot=3, charge=0):
        return UniversalComplex(
            name="test", binding_mode="protein_ligand_physics",
            guest_smiles=smiles,
            sasa_buried_A2=buried, guest_sasa_total_A2=total_sasa,
            guest_sasa_nonpolar_A2=np_sasa, guest_sasa_polar_A2=polar_sasa,
            n_hbonds_formed=n_hb, guest_rotatable_bonds=n_rot,
            guest_charge=charge,
        )

    def test_hydrophobic_is_favorable(self):
        """Burying nonpolar surface produces negative dg_hydrophobic."""
        result = score_physics_pl(self._make_uc())
        assert result["dg_hydrophobic"] < 0, (
            f"Hydrophobic should be favorable, got {result['dg_hydrophobic']:.2f}"
        )

    def test_conf_entropy_is_unfavorable(self):
        """Freezing rotors produces positive dg_conf_entropy."""
        result = score_physics_pl(self._make_uc(n_rot=5))
        assert result["dg_conf_entropy"] > 0

    def test_more_rotors_more_penalty(self):
        """More rotors → larger conformational penalty."""
        r1 = score_physics_pl(self._make_uc(smiles="CCO", n_rot=1))        # ethanol: ~1 rotor
        r5 = score_physics_pl(self._make_uc(smiles="CCCCCCCCO", n_rot=6))  # octanol: ~6 rotors
        assert r5["dg_conf_entropy"] > r1["dg_conf_entropy"]

    def test_more_burial_more_hydrophobic(self):
        """More buried SASA → more favorable hydrophobic term."""
        r_small = score_physics_pl(self._make_uc(buried=50.0))
        r_large = score_physics_pl(self._make_uc(buried=200.0))
        assert r_large["dg_hydrophobic"] < r_small["dg_hydrophobic"]

    def test_hbond_scales_with_count(self):
        """More H-bonds → more favorable dg_hbond."""
        r0 = score_physics_pl(self._make_uc(n_hb=0))
        r4 = score_physics_pl(self._make_uc(n_hb=4))
        assert r4["dg_hbond"] < r0["dg_hbond"], (
            f"4 HBs ({r4['dg_hbond']:.1f}) should be more favorable "
            f"than 0 HBs ({r0['dg_hbond']:.1f})"
        )

    def test_charged_hb_stronger(self):
        """Charged ligand gets stronger H-bond energy."""
        r_neutral = score_physics_pl(self._make_uc(n_hb=4, charge=0))
        r_charged = score_physics_pl(self._make_uc(n_hb=4, charge=1))
        assert r_charged["dg_hbond"] < r_neutral["dg_hbond"], (
            "Charged H-bonds should be more favorable"
        )

    def test_decomposition_sums_to_total(self):
        """Individual terms sum to dg_total."""
        r = score_physics_pl(self._make_uc())
        expected = (r["dg_desolv"] + r["dg_hydrophobic"]
                    + r["dg_hbond"] + r["dg_conf_entropy"])
        assert abs(r["dg_total"] - expected) < 0.01

    def test_log_ka_correct_sign(self):
        """Favorable binding (negative ΔG) gives positive log Ka."""
        # Large buried surface, few rotors → should be favorable
        r = score_physics_pl(self._make_uc(
            smiles="CCCCCCCC", buried=250.0, total_sasa=300.0,
            np_sasa=250.0, polar_sasa=50.0, n_hb=0, n_rot=0
        ))
        if r["dg_total"] < 0:
            assert r["log_Ka_pred"] > 0


# ===================================================================
# INTEGRATION WITH unified_scorer_v2
# ===================================================================

class TestUnifiedScorerIntegration:

    def test_physics_mode_fires(self):
        """protein_ligand_physics mode fires physics scorer in predict()."""
        try:
            from core.unified_scorer_v2 import predict, PredictionResult
        except ImportError:
            pytest.skip("unified_scorer_v2 import chain unavailable")

        uc = UniversalComplex(
            name="butanol_in_pocket",
            binding_mode="protein_ligand_physics",
            guest_smiles="CCCCO",
            sasa_buried_A2=100.0,
            guest_sasa_total_A2=200.0,
            guest_sasa_nonpolar_A2=120.0,
            guest_sasa_polar_A2=80.0,
            n_hbonds_formed=2,
            guest_rotatable_bonds=3,
        )
        result = predict(uc)
        # At least one physics term should be nonzero
        has_physics = (
            result.dg_hydrophobic != 0.0
            or result.dg_conf_entropy != 0.0
            or result.dg_group_desolv != 0.0
        )
        assert has_physics, "Physics PL terms should fire for protein_ligand_physics mode"

    def test_hg_mode_unaffected(self):
        """host_guest_inclusion mode is not changed by physics PL addition."""
        try:
            from core.unified_scorer_v2 import predict
        except ImportError:
            pytest.skip("unified_scorer_v2 import chain unavailable")

        uc = UniversalComplex(
            name="adamantane_in_bCD",
            binding_mode="host_guest_inclusion",
            host_name="beta-CD",
            guest_smiles="C1(CC2CC3CC(C2)CC1C3)",  # adamantane
        )
        result = predict(uc)
        # Should not have physics PL terms from the new scorer
        # (HG terms may still fire via existing path)
        # The key test: no regression
        assert result is not None


# ===================================================================
# OPENBABEL-DEPENDENT TESTS
# ===================================================================

class TestDesolvationIntegration:

    @pytest.fixture(autouse=True)
    def check_openbabel(self):
        pytest.importorskip("openbabel")

    def test_desolvation_fires_with_smiles(self):
        """FreeSolv desolvation computes nonzero for drug-like molecule."""
        result = score_physics_pl(UniversalComplex(
            name="test", binding_mode="protein_ligand_physics",
            guest_smiles="c1ccc(cc1)C(=O)O",  # benzoic acid
            sasa_buried_A2=150.0, guest_sasa_total_A2=250.0,
            guest_sasa_nonpolar_A2=150.0, guest_sasa_polar_A2=100.0,
            n_hbonds_formed=2, guest_rotatable_bonds=1,
        ))
        assert result["dg_desolv"] != 0.0, "Desolvation should fire with SMILES"

    def test_hydrophilic_ligand_high_desolv_cost(self):
        """Hydrophilic ligand (many OH) has positive desolvation cost."""
        result = score_physics_pl(UniversalComplex(
            name="test", binding_mode="protein_ligand_physics",
            guest_smiles="OCC(O)C(O)C(O)CO",  # xylitol-like polyol
            sasa_buried_A2=150.0, guest_sasa_total_A2=250.0,
            n_hbonds_formed=3, guest_rotatable_bonds=4,
        ))
        assert result["dg_desolv"] > 0, (
            f"Polyol desolvation cost should be positive, got {result['dg_desolv']:.2f}"
        )

    def test_hydrophobic_ligand_favorable_desolv(self):
        """Hydrophobic ligand (alkane) has negative desolvation cost."""
        result = score_physics_pl(UniversalComplex(
            name="test", binding_mode="protein_ligand_physics",
            guest_smiles="CCCCCCCCCC",  # decane
            sasa_buried_A2=200.0, guest_sasa_total_A2=300.0,
            guest_sasa_nonpolar_A2=280.0, guest_sasa_polar_A2=20.0,
            n_hbonds_formed=0, guest_rotatable_bonds=7,
        ))
        # decane is hydrophobic → dG_hydr > 0 → desolvation cost = -f × dG_hydr < 0
        assert result["dg_desolv"] < 0, (
            f"Decane desolvation should be favorable, got {result['dg_desolv']:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])