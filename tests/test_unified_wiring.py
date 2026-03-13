"""
tests/test_unified_wiring.py — Verify glycan wiring + routing relaxation
=========================================================================

Tests:
  1. Glycan fields exist on UniversalComplex
  2. Glycan fields exist on PredictionResult
  3. predict() self-zeros glycan for non-glycan entry
  4. predict() fires glycan for glycan entry (ConA-mannose)
  5. Regression: existing HG/metal entries unchanged
  6. Routing: metalloprotein fires on data-presence (no hard gate)
  7. Routing: general PL fires on data-presence (no hard gate)
"""

import pytest
import sys
import os

# Ensure project root on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestGlycanFieldsExist:
    """Schema and result have glycan fields."""

    def test_uc_has_sugar_property_card(self):
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(name="test")
        assert hasattr(uc, 'sugar_property_card')
        assert uc.sugar_property_card is None

    def test_uc_has_glycan_contact_map(self):
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(name="test")
        assert hasattr(uc, 'glycan_contact_map')
        assert uc.glycan_contact_map is None

    def test_uc_has_beta_context(self):
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(name="test")
        assert hasattr(uc, 'beta_context')
        assert uc.beta_context is None

    def test_result_has_glycan_total(self):
        from core.unified_scorer_v2 import PredictionResult
        r = PredictionResult(name="t", binding_mode="x",
                             log_Ka_exp=0, log_Ka_pred=0,
                             dg_total_kj=0, error=0)
        assert hasattr(r, 'dg_glycan_total')
        assert r.dg_glycan_total == 0.0

    def test_result_has_all_glycan_fields(self):
        from core.unified_scorer_v2 import PredictionResult
        r = PredictionResult(name="t", binding_mode="x",
                             log_Ka_exp=0, log_Ka_pred=0,
                             dg_total_kj=0, error=0)
        for field in ['dg_glycan_polar_desolv', 'dg_glycan_hbond',
                      'dg_glycan_conf_entropy', 'dg_glycan_ch_pi',
                      'dg_glycan_structural_water', 'dg_glycan_ca_coordination',
                      'dg_glycan_total']:
            assert hasattr(r, field), f"Missing {field}"
            assert getattr(r, field) == 0.0


class TestGlycanSelfZero:
    """Glycan terms must be zero when no glycan data present."""

    def test_hg_entry_zero_glycan(self):
        """A standard HG entry should have zero glycan contribution."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="beta-CD:adamantane",
            binding_mode="host_guest_inclusion",
            host_name="beta-CD",
            guest_smiles="C1C2CC3CC1CC(C2)C3",
            log_Ka_exp=4.26,
        )
        result = predict(uc)
        assert result.dg_glycan_total == 0.0

    def test_metal_entry_zero_glycan(self):
        """A metal coordination entry should have zero glycan."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="Cu-EDTA",
            binding_mode="metal_coordination",
            metal_formula="Cu2+",
            metal_charge=2,
            metal_d_electrons=9,
            donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate",
                           "O_carboxylate","O_carboxylate"],
            chelate_rings=5,
            ring_sizes=[5,5,5,5,5],
            denticity=6,
            log_Ka_exp=18.8,
        )
        result = predict(uc)
        assert result.dg_glycan_total == 0.0


class TestGlycanFires:
    """Glycan terms fire when glycan data present."""

    def test_cona_mannose_fires(self):
        """ConA + mannose with glycan data should produce nonzero glycan energy."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict

        try:
            from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
            from mabe.glycan.contact_map import cona_mannose_pocket
        except ImportError:
            pytest.skip("Glycan module not installed")

        uc = UniversalComplex(
            name="ConA:mannose",
            binding_mode="lectin_glycan",
            log_Ka_exp=3.91,  # log(8200)
            sugar_property_card=ALPHA_D_MANNOSE,
            glycan_contact_map=cona_mannose_pocket(),
            beta_context=0.45,
        )
        result = predict(uc)
        assert result.dg_glycan_total != 0.0
        assert result.dg_glycan_total < 0  # should be favorable
        assert result.dg_glycan_polar_desolv > 0  # desolvation is unfavorable
        assert result.dg_glycan_hbond < 0  # H-bonds are favorable

    def test_glycan_adds_to_total(self):
        """Glycan energy should be included in dg_total_kj."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict

        try:
            from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
            from mabe.glycan.contact_map import cona_mannose_pocket
        except ImportError:
            pytest.skip("Glycan module not installed")

        uc = UniversalComplex(
            name="ConA:mannose",
            binding_mode="lectin_glycan",
            log_Ka_exp=3.91,
            sugar_property_card=ALPHA_D_MANNOSE,
            glycan_contact_map=cona_mannose_pocket(),
        )
        result = predict(uc)
        # dg_total should include glycan
        # For a pure glycan entry, total should equal glycan_total
        # (other terms self-zero since no guest_smiles/metal/host)
        assert abs(result.dg_total_kj - result.dg_glycan_total) < 0.01


