"""
tests/test_glycan_integration.py — Integration tests for glycan module
======================================================================

Tests:
  1. Parameter integrity (values match documented sources)
  2. Sugar property cards (correct hydroxyl counts and positions)
  3. Contact map construction (ConA known contacts)
  4. Scorer correctness (ConA deoxy series predictions)
  5. Unified scorer integration (self-zero for non-glycan entries)
  6. Regression guard (glycan terms zero for metal/HG entries)
"""

import pytest
import math

from mabe.glycan.params import GlycanParams, GLYCAN_PARAMS, SASA_DESOLV_PER_POSITION
from mabe.glycan.sugar_properties import (
    get_sugar_card, ALPHA_D_MANNOSE, ALPHA_D_GLUCOSE, ALPHA_D_GALACTOSE,
    ALPHA_D_GLCNAC, ALPHA_D_GALNAC, ALPHA_L_FUCOSE,
)
from mabe.glycan.contact_map import GlycanContactMap, OHContact, cona_mannose_pocket
from mabe.glycan.scorer import compute_glycan_terms, GlycanScoreDecomposition
from mabe.glycan.unified_scorer_extension import compute_glycan_contribution


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterIntegrity:
    """Verify parameter values match documented sources."""

    def test_k_desolv_OH(self):
        assert GLYCAN_PARAMS.k_desolv_OH == 3.97

    def test_k_hbond_dG(self):
        assert GLYCAN_PARAMS.k_hbond_dG == -2.00

    def test_k_hbond_dH(self):
        assert GLYCAN_PARAMS.k_hbond_dH == -7.35

    def test_dCp_per_OH(self):
        assert GLYCAN_PARAMS.dCp_per_OH == -52.0

    def test_eps_CH_pi_locked(self):
        assert GLYCAN_PARAMS.eps_CH_pi == -2.5

    def test_beta_context(self):
        assert GLYCAN_PARAMS.beta_context_default == 0.45

    def test_sasa_table_complete(self):
        expected = {'C1_anomeric', 'C2_equatorial', 'C2_axial',
                    'C3_equatorial', 'C4_equatorial', 'C4_axial', 'C6_primary'}
        assert set(SASA_DESOLV_PER_POSITION.keys()) == expected

    def test_sasa_all_positive(self):
        for k, v in SASA_DESOLV_PER_POSITION.items():
            assert v > 0, f"{k} must be positive"


# ═══════════════════════════════════════════════════════════════════════════
# SUGAR PROPERTY CARDS
# ═══════════════════════════════════════════════════════════════════════════

class TestSugarProperties:

    def test_mannose_has_5_hydroxyls(self):
        assert len(ALPHA_D_MANNOSE.hydroxyls) == 5

    def test_mannose_c2_axial(self):
        c2 = [h for h in ALPHA_D_MANNOSE.hydroxyls if h.position == 'C2'][0]
        assert c2.orientation == 'axial'

    def test_glucose_all_equatorial_except_anomeric(self):
        for h in ALPHA_D_GLUCOSE.hydroxyls:
            if h.position != 'C1':
                assert h.orientation in ('equatorial', 'primary'), \
                    f"Glucose {h.position} should be equatorial/primary"

    def test_galactose_c4_axial(self):
        c4 = [h for h in ALPHA_D_GALACTOSE.hydroxyls if h.position == 'C4'][0]
        assert c4.orientation == 'axial'

    def test_glcnac_has_4_hydroxyls(self):
        """GlcNAc: C2 replaced by NAc → 4 OH groups."""
        assert len(ALPHA_D_GLCNAC.hydroxyls) == 4
        assert ALPHA_D_GLCNAC.has_NAc

    def test_fucose_no_c6_oh(self):
        """Fucose is 6-deoxy → no C6-OH."""
        positions = [h.position for h in ALPHA_L_FUCOSE.hydroxyls]
        assert 'C6' not in positions

    def test_sugar_library_lookup(self):
        card = get_sugar_card('aMan')
        assert card.three_letter == 'Man'


# ═══════════════════════════════════════════════════════════════════════════
# CONTACT MAP
# ═══════════════════════════════════════════════════════════════════════════

class TestContactMap:

    def test_cona_pocket_5_positions(self):
        cm = cona_mannose_pocket()
        assert len(cm.oh_contacts) == 5

    def test_cona_essential_positions(self):
        cm = cona_mannose_pocket()
        essential = [c.position for c in cm.oh_contacts if c.n_hbonds >= 2]
        assert set(essential) == {'C3', 'C4', 'C6'}

    def test_cona_nonessential_positions(self):
        cm = cona_mannose_pocket()
        noness = [c.position for c in cm.oh_contacts if c.n_hbonds == 0]
        assert set(noness) == {'C1', 'C2'}

    def test_cona_no_ch_pi(self):
        cm = cona_mannose_pocket()
        assert cm.total_ch_pi == 0

    def test_cona_hbonds_per_oh(self):
        cm = cona_mannose_pocket()
        assert cm.n_hbonds_per_oh == [0, 0, 2, 2, 2]


# ═══════════════════════════════════════════════════════════════════════════
# SCORER — ConA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestGlycanScorer:

    def test_self_zero_no_sugar(self):
        result = compute_glycan_terms(None, cona_mannose_pocket())
        assert result.dG_total == 0.0

    def test_self_zero_no_contacts(self):
        result = compute_glycan_terms(ALPHA_D_MANNOSE, None)
        assert result.dG_total == 0.0

    def test_cona_essentiality(self):
        result = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket())
        assert result.n_essential == 3
        assert result.n_nonessential == 2
        assert result.essentiality_map['C3'] == 'essential'
        assert result.essentiality_map['C4'] == 'essential'
        assert result.essentiality_map['C6'] == 'essential'
        assert result.essentiality_map['C1'] == 'nonessential'
        assert result.essentiality_map['C2'] == 'nonessential'

    def test_cona_mannose_score_negative(self):
        """Overall score should be favorable (negative) — Man binds ConA."""
        result = compute_glycan_terms(
            ALPHA_D_MANNOSE, cona_mannose_pocket(),
            beta_context=0.45
        )
        assert result.dG_total < 0, f"Man@ConA should be favorable, got {result.dG_total}"

    def test_deoxy_c3_costly(self):
        """Removing C3-OH (2 H-bonds) should make binding much worse."""
        full = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket(),
                                    beta_context=0.45)
        # Create a modified contact map with C3 set to 0 H-bonds
        cm_deoxy = cona_mannose_pocket()
        cm_deoxy.oh_contacts[2] = OHContact('C3', n_hbonds=0, is_solvent_exposed=True)

        deoxy = compute_glycan_terms(ALPHA_D_MANNOSE, cm_deoxy, beta_context=0.45)
        ddg = deoxy.dG_total - full.dG_total
        assert ddg > 3.0, f"Removing C3-OH should cost >3 kJ/mol, got {ddg:.2f}"

    def test_deoxy_c2_cheap(self):
        """Removing C2-OH (0 H-bonds) should have minimal effect."""
        full = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket(),
                                    beta_context=0.45)
        cm_deoxy = cona_mannose_pocket()
        cm_deoxy.oh_contacts[1] = OHContact('C2', n_hbonds=0, is_solvent_exposed=True)

        deoxy = compute_glycan_terms(ALPHA_D_MANNOSE, cm_deoxy, beta_context=0.45)
        ddg = abs(deoxy.dG_total - full.dG_total)
        # C2 already has 0 H-bonds, so removing it should cost nothing
        assert ddg < 0.5, f"Removing C2-OH (no contacts) should be near-zero, got {ddg:.2f}"

    def test_enthalpy_vs_free_energy_hbond(self):
        """k_hbond_dH gives stronger effect than k_hbond_dG."""
        result_dh = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket(),
                                          use_enthalpy_hbond=True)
        result_dg = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket(),
                                          use_enthalpy_hbond=False)
        # ΔH H-bonds are stronger → more favorable total
        assert result_dh.dG_total < result_dg.dG_total

    def test_ch_pi_adds_stabilization(self):
        """Adding CH-π contacts should make score more favorable."""
        cm = cona_mannose_pocket()
        result_no_pi = compute_glycan_terms(ALPHA_D_MANNOSE, cm)

        from mabe.glycan.contact_map import CHPiContact
        cm_with_pi = cona_mannose_pocket()
        cm_with_pi.ch_pi_contacts = [
            CHPiContact(sugar_hydrogens=['C1-H', 'C3-H', 'C5-H'],
                       receptor_residue='Trp181', n_CH_contacts=3)
        ]
        result_with_pi = compute_glycan_terms(ALPHA_D_MANNOSE, cm_with_pi)

        assert result_with_pi.dG_total < result_no_pi.dG_total
        assert abs(result_with_pi.dG_ch_pi - 3 * (-2.5)) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED SCORER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestUnifiedScorerIntegration:

    def test_self_zero_for_metal_entry(self):
        """A metal coordination entry has no glycan fields → zero contribution."""
        class FakeUC:
            pass  # no sugar_property_card, no glycan_contact_map

        uc = FakeUC()
        result = {'dg_total': -50.0}  # existing metal score
        result = compute_glycan_contribution(uc, result)

        assert result['dg_glycan_polar_desolv'] == 0.0
        assert result['dg_glycan_hbond'] == 0.0
        assert result['dg_glycan_ch_pi'] == 0.0
        assert result['dg_total'] == -50.0  # unchanged

    def test_self_zero_for_hg_entry(self):
        """A host-guest entry has no glycan fields → zero contribution."""
        class FakeUC:
            sugar_property_card = None
            glycan_contact_map = None

        uc = FakeUC()
        result = {'dg_total': -25.0}
        result = compute_glycan_contribution(uc, result)

        assert result['dg_glycan_polar_desolv'] == 0.0
        assert result['dg_total'] == -25.0

    def test_glycan_entry_adds_to_total(self):
        """A glycan entry with data should produce non-zero contribution."""
        class FakeUC:
            sugar_property_card = ALPHA_D_MANNOSE
            glycan_contact_map = cona_mannose_pocket()
            beta_context = 0.45

        uc = FakeUC()
        result = {'dg_total': 0.0}
        result = compute_glycan_contribution(uc, result)

        assert result['dg_total'] != 0.0
        assert result['glycan_n_essential'] == 3
        assert result['glycan_n_nonessential'] == 2


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION GUARD
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionGuard:
    """
    These tests verify that adding glycan terms does NOT change
    predictions for existing metal and host-guest entries.
    
    The actual 644-entry regression requires the full repo.
    These tests verify the self-zero mechanism that guarantees it.
    """

    def test_no_glycan_fields_means_zero(self):
        """Any object without glycan fields → zero glycan contribution."""
        for _ in range(10):  # multiple fake entries
            class FakeUC:
                pass
            uc = FakeUC()
            result = {'dg_total': -42.0}
            result = compute_glycan_contribution(uc, result)
            assert result['dg_total'] == -42.0

    def test_none_glycan_fields_means_zero(self):
        """Explicitly None glycan fields → zero."""
        class FakeUC:
            sugar_property_card = None
            glycan_contact_map = None
            beta_context = None
        uc = FakeUC()
        result = {'dg_total': -100.0}
        result = compute_glycan_contribution(uc, result)
        assert result['dg_total'] == -100.0

    def test_glycan_terms_all_present_in_result(self):
        """Even for non-glycan entries, result dict has glycan keys (= 0)."""
        class FakeUC:
            pass
        uc = FakeUC()
        result = {}
        result = compute_glycan_contribution(uc, result)
        assert 'dg_glycan_polar_desolv' in result
        assert 'dg_glycan_hbond' in result
        assert 'dg_glycan_ch_pi' in result
        assert 'dg_glycan_structural_water' in result
        assert 'dg_glycan_ca_coordination' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
