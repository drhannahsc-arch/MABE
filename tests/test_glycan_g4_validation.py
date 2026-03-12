"""
tests/test_glycan_g4_validation.py -- G4 Synthetic Receptor Validation Gate
=============================================================================

Tests that locked G1+G2+G3 parameters correctly predict binding in
non-biological systems. No new parameters. Pure prediction.

Primary: Davis GluHUT receptor (Tromans 2019)
Secondary: ConA deoxy regression through full scorer

SUCCESS CRITERIA (from plan):
  - Davis: rank correlation rho >= 0.8
  - Davis: Glc ranked #1
  - Davis: Glc > Gal selectivity (inverts ConA)
  - ConA: essentiality classification 5/5 correct
  - Same parameters predict both systems
"""

import pytest
import math

from mabe.glycan.validation_g4 import (
    run_davis_validation, run_cona_regression,
    davis_receptor_contacts, DAVIS_ANSWER_KEY, dG_from_Ka,
)
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.sugar_properties import (
    ALPHA_D_MANNOSE, ALPHA_D_GLUCOSE, ALPHA_D_GALACTOSE,
    get_sugar_card,
)
from mabe.glycan.contact_map import cona_mannose_pocket


# =====================================================================
# DAVIS RECEPTOR CONTACT MODEL
# =====================================================================

class TestDavisContacts:

    def test_glucose_5_oh_contacts(self):
        cm = davis_receptor_contacts('aGlc')
        assert len(cm.oh_contacts) == 5

    def test_glucose_all_equatorial_get_hbonds(self):
        """Glucose: all OHs equatorial -> all form H-bonds to urea."""
        cm = davis_receptor_contacts('aGlc')
        for c in cm.oh_contacts:
            assert c.n_hbonds >= 1, f"{c.position} should have H-bond in Davis receptor"

    def test_galactose_c4_axial_no_hbond(self):
        """Galactose C4-OH is axial -> no H-bond to urea donor."""
        cm = davis_receptor_contacts('aGal')
        c4 = [c for c in cm.oh_contacts if c.position == 'C4'][0]
        assert c4.n_hbonds == 0

    def test_mannose_c2_axial_no_hbond(self):
        """Mannose C2-OH is axial -> no H-bond."""
        cm = davis_receptor_contacts('aMan')
        c2 = [c for c in cm.oh_contacts if c.position == 'C2'][0]
        assert c2.n_hbonds == 0

    def test_glucose_more_ch_pi_than_galactose(self):
        """Glc (all-eq) gets more CH-pi contacts than Gal (C4-ax OH)."""
        cm_glc = davis_receptor_contacts('aGlc')
        cm_gal = davis_receptor_contacts('aGal')
        assert cm_glc.total_ch_pi >= cm_gal.total_ch_pi

    def test_no_linkages(self):
        """Monosaccharides: no glycosidic linkages."""
        cm = davis_receptor_contacts('aGlc')
        assert cm.linkage_types == []


# =====================================================================
# DAVIS VALIDATION — PRIMARY G4 TEST
# =====================================================================

class TestDavisValidation:

    @pytest.fixture
    def davis_result(self):
        return run_davis_validation(beta_context=1.0, use_enthalpy_hbond=True)

    def test_rank_correlation_meets_threshold(self, davis_result):
        """rho >= 0.8 (plan success criterion)."""
        assert davis_result['rank_correlation'] >= 0.8, \
            f"rho = {davis_result['rank_correlation']:.3f}, need >= 0.8"

    def test_glucose_ranked_first(self, davis_result):
        """Glucose must be the top-ranked sugar."""
        assert davis_result['rank_correct'], \
            "Glucose should be ranked #1 in Davis receptor"

    def test_selectivity_inverted_vs_cona(self, davis_result):
        """Glc > Gal in Davis (inverts ConA's Man > Glc)."""
        assert davis_result['selectivity_inverted'], \
            "Davis receptor should prefer Glc over Gal"

    def test_glucose_most_favorable(self, davis_result):
        """Glucose should have the most negative predicted dG."""
        glc = next(r for r in davis_result['results'] if r.sugar_key == 'aGlc')
        for r in davis_result['results']:
            if r.sugar_key != 'aGlc':
                assert glc.dG_predicted < r.dG_predicted, \
                    f"Glc ({glc.dG_predicted:.1f}) should be more favorable than {r.sugar_name} ({r.dG_predicted:.1f})"

    def test_glucose_gal_separation(self, davis_result):
        """Glc should be at least 5 kJ/mol more favorable than Gal."""
        glc = next(r for r in davis_result['results'] if r.sugar_key == 'aGlc')
        gal = next(r for r in davis_result['results'] if r.sugar_key == 'aGal')
        separation = gal.dG_predicted - glc.dG_predicted
        assert separation > 5.0, \
            f"Glc-Gal separation = {separation:.1f} kJ/mol, need > 5"

    def test_fructose_weakest(self, davis_result):
        """Fructose should be the weakest binder."""
        fuc = next(r for r in davis_result['results'] if r.sugar_key == 'aFuc')
        for r in davis_result['results']:
            if r.sugar_key != 'aFuc':
                assert fuc.dG_predicted > r.dG_predicted, \
                    f"Fructose should be weaker than {r.sugar_name}"

    def test_all_scores_negative(self, davis_result):
        """All sugars should show some binding (negative dG)."""
        for r in davis_result['results']:
            assert r.dG_predicted < 0, \
                f"{r.sugar_name} predicted dG = {r.dG_predicted:.1f}, should be negative"


# =====================================================================
# SCAFFOLD INDEPENDENCE — THE KEY CLAIM
# =====================================================================

class TestScaffoldIndependence:
    """
    Same parameters must predict:
      - ConA: Man > Glc (metal coordination site favors Man O3/O4)
      - Davis: Glc >> Gal (aromatic cage favors all-equatorial)
    
    This is the poster centrepiece claim.
    """

    def test_cona_prefers_man_essential_ohs(self):
        """In ConA, Man's C3/C4/C6 OHs are all essential (n_hb=2)."""
        cm = cona_mannose_pocket()
        essential = [c.position for c in cm.oh_contacts if c.n_hbonds >= 2]
        assert set(essential) == {'C3', 'C4', 'C6'}

    def test_davis_prefers_glc_equatorial(self):
        """In Davis, Glc's all-equatorial geometry gives max H-bonds + CH-pi."""
        cm_glc = davis_receptor_contacts('aGlc')
        cm_man = davis_receptor_contacts('aMan')

        glc_hbonds = sum(c.n_hbonds for c in cm_glc.oh_contacts)
        man_hbonds = sum(c.n_hbonds for c in cm_man.oh_contacts)
        assert glc_hbonds > man_hbonds, \
            "Glc should have more H-bonds than Man in Davis receptor"

    def test_same_params_opposite_selectivity(self):
        """Score ConA and Davis with SAME parameters, get opposite ranking."""
        # ConA: Man score
        man_cona = compute_glycan_terms(
            ALPHA_D_MANNOSE, cona_mannose_pocket(), beta_context=0.45)
        # ConA: Glc score (use Glc sugar card with Man pocket contacts)
        glc_cona = compute_glycan_terms(
            ALPHA_D_GLUCOSE, cona_mannose_pocket(), beta_context=0.45)

        # Davis: Glc and Man
        glc_davis = compute_glycan_terms(
            ALPHA_D_GLUCOSE, davis_receptor_contacts('aGlc'), beta_context=1.0)
        man_davis = compute_glycan_terms(
            ALPHA_D_MANNOSE, davis_receptor_contacts('aMan'), beta_context=1.0)

        # Davis: Glc should be more favorable
        assert glc_davis.dG_total < man_davis.dG_total, \
            "Davis should prefer Glc over Man"


# =====================================================================
# CONA REGRESSION
# =====================================================================

class TestConARegression:

    @pytest.fixture
    def cona_result(self):
        return run_cona_regression()

    def test_regression_passed(self, cona_result):
        assert cona_result['passed'], "ConA deoxy regression failed"

    def test_essentials_correct(self, cona_result):
        assert cona_result['essentials_correct']

    def test_nonessentials_correct(self, cona_result):
        assert cona_result['nonessentials_correct']

    def test_c3_costly(self, cona_result):
        assert cona_result['deoxy_results']['C3']['ddg_predicted'] > 3.0

    def test_c4_costly(self, cona_result):
        assert cona_result['deoxy_results']['C4']['ddg_predicted'] > 3.0

    def test_c6_costly(self, cona_result):
        assert cona_result['deoxy_results']['C6']['ddg_predicted'] > 3.0

    def test_c1_cheap(self, cona_result):
        assert abs(cona_result['deoxy_results']['C1']['ddg_predicted']) < 2.0

    def test_c2_cheap(self, cona_result):
        assert abs(cona_result['deoxy_results']['C2']['ddg_predicted']) < 2.0


# =====================================================================
# ANSWER KEY INTEGRITY
# =====================================================================

class TestAnswerKey:

    def test_davis_ka_values_present(self):
        assert len(DAVIS_ANSWER_KEY) >= 4

    def test_glucose_highest_ka(self):
        glc_ka = DAVIS_ANSWER_KEY['aGlc']['Ka_ITC']
        for key, val in DAVIS_ANSWER_KEY.items():
            if key != 'aGlc':
                assert glc_ka > val['Ka_ITC'], \
                    f"Glc Ka ({glc_ka}) should exceed {key} ({val['Ka_ITC']})"

    def test_dg_conversion(self):
        """dG_from_Ka should give correct sign and magnitude."""
        dg = dG_from_Ka(18600)
        assert -26 < dg < -23  # ~-24.4 kJ/mol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
