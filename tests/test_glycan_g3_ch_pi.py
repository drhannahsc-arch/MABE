"""
tests/test_glycan_g3_ch_pi.py -- G3 CH-pi interaction tests
=============================================================

Tests:
  1. Parameter integrity (eps_CH_pi, aromatic weights)
  2. Geometric weighting (distance, angle)
  3. Sugar face profiles (axial CH counts, stacking rules)
  4. Scoring correctness (simple + geometry-weighted)
  5. Aromatic hierarchy (Trp > Tyr > Phe > His)
  6. Hevein cross-check (2-3 contacts x Trp = 5-7.5 kJ/mol)
  7. Integration with scorer (simple and detailed paths)
"""

import pytest
import math

from mabe.glycan.ch_pi import (
    EPS_CH_PI, AROMATIC_WEIGHT, D_OPTIMAL, D_SIGMA, D_CUTOFF,
    f_distance, f_angle, score_ch_pi, CHPiContact,
    CH_PI_PROFILES, get_ch_pi_profile, estimate_ch_pi_contacts,
)
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
from mabe.glycan.contact_map import (
    GlycanContactMap, OHContact,
    CHPiContact as ContactMapCHPi,
    cona_mannose_pocket,
)


# =====================================================================
# PARAMETER INTEGRITY
# =====================================================================

class TestParameters:

    def test_eps_ch_pi_locked(self):
        assert EPS_CH_PI == -2.5

    def test_eps_in_literature_range(self):
        assert -4.0 < EPS_CH_PI < -1.0

    def test_aromatic_hierarchy(self):
        assert AROMATIC_WEIGHT['Trp'] > AROMATIC_WEIGHT['Tyr']
        assert AROMATIC_WEIGHT['Tyr'] > AROMATIC_WEIGHT['Phe']
        assert AROMATIC_WEIGHT['Phe'] > AROMATIC_WEIGHT['His']

    def test_trp_is_reference(self):
        assert AROMATIC_WEIGHT['Trp'] == 1.0

    def test_all_weights_positive(self):
        for aa, w in AROMATIC_WEIGHT.items():
            assert 0 < w <= 1.0, f"{aa}: weight {w}"


# =====================================================================
# GEOMETRIC WEIGHTING
# =====================================================================

class TestGeometricWeighting:

    def test_distance_optimal(self):
        assert f_distance(D_OPTIMAL) > 0.99

    def test_distance_cutoff(self):
        assert f_distance(D_CUTOFF + 0.1) == 0.0

    def test_distance_too_close(self):
        assert f_distance(1.5) == 0.0

    def test_distance_monotonic_from_optimal(self):
        """Weight decreases as distance moves from optimal."""
        assert f_distance(3.5) > f_distance(4.0)
        assert f_distance(3.5) > f_distance(3.0)

    def test_distance_symmetric_around_optimal(self):
        """Approximately symmetric Gaussian."""
        w_near = f_distance(D_OPTIMAL - 0.3)
        w_far = f_distance(D_OPTIMAL + 0.3)
        assert abs(w_near - w_far) < 0.01

    def test_angle_perpendicular(self):
        assert f_angle(0) > 0.99

    def test_angle_parallel(self):
        assert f_angle(90) == 0.0

    def test_angle_45_deg(self):
        assert abs(f_angle(45) - 0.5) < 0.01

    def test_angle_monotonic(self):
        assert f_angle(0) > f_angle(30) > f_angle(60) > f_angle(89)


# =====================================================================
# SUGAR FACE PROFILES
# =====================================================================

class TestSugarFaces:

    def test_mannose_alpha_face_3ch(self):
        p = CH_PI_PROFILES['aMan']
        assert p.alpha_face.n_axial_CH == 3

    def test_mannose_beta_face_blocked(self):
        """Man C2-axial OH blocks beta face."""
        p = CH_PI_PROFILES['aMan']
        assert p.beta_face.stackable is False

    def test_glucose_alpha_face_3ch(self):
        p = CH_PI_PROFILES['aGlc']
        assert p.alpha_face.n_axial_CH == 3

    def test_galactose_alpha_face_3ch(self):
        p = CH_PI_PROFILES['aGal']
        assert p.alpha_face.n_axial_CH == 3

    def test_galactose_beta_blocked(self):
        """Gal C4-axial OH blocks beta face."""
        p = CH_PI_PROFILES['aGal']
        assert p.beta_face.stackable is False

    def test_bGlcNAc_sandwich(self):
        """beta-GlcNAc can be sandwiched (WGA/hevein)."""
        p = CH_PI_PROFILES['bGlcNAc']
        assert p.sandwich_possible is True
        assert p.alpha_face.stackable and p.beta_face.stackable

    def test_bGlc_sandwich(self):
        """beta-Glc (all eq) can be sandwiched."""
        p = CH_PI_PROFILES['bGlc']
        assert p.sandwich_possible is True

    def test_best_face(self):
        p = CH_PI_PROFILES['aMan']
        assert p.best_face.face == 'alpha'

    def test_all_profiles_have_max_contacts(self):
        for key, p in CH_PI_PROFILES.items():
            assert p.max_simultaneous_contacts >= 1, f"{key} has 0 max contacts"


# =====================================================================
# SCORING
# =====================================================================

class TestScoring:

    def test_empty_contacts_zero(self):
        r = score_ch_pi([])
        assert r['dG_ch_pi'] == 0.0
        assert r['n_contacts'] == 0

    def test_single_trp_ideal(self):
        """One contact with Trp at ideal geometry."""
        c = [CHPiContact('C3-H', 'Trp181', 'Trp')]
        r = score_ch_pi(c, use_geometry=False)
        assert abs(r['dG_ch_pi'] - (-2.5)) < 0.01

    def test_single_phe_weaker(self):
        """Phe contact weaker than Trp."""
        c_trp = [CHPiContact('C3-H', 'Trp181', 'Trp')]
        c_phe = [CHPiContact('C3-H', 'Phe181', 'Phe')]
        r_trp = score_ch_pi(c_trp, use_geometry=False)
        r_phe = score_ch_pi(c_phe, use_geometry=False)
        assert r_trp['dG_ch_pi'] < r_phe['dG_ch_pi']  # more negative = stronger

    def test_three_contacts_additive(self):
        """Three Trp contacts = 3 x eps."""
        contacts = [
            CHPiContact('C1-H', 'Trp181', 'Trp'),
            CHPiContact('C3-H', 'Trp181', 'Trp'),
            CHPiContact('C5-H', 'Trp181', 'Trp'),
        ]
        r = score_ch_pi(contacts, use_geometry=False)
        assert abs(r['dG_ch_pi'] - 3 * (-2.5)) < 0.01

    def test_distance_weighting_reduces(self):
        """Contact at 4.2 A weaker than at 3.5 A."""
        c_ideal = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=3.5)]
        c_far = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=4.2)]
        r_ideal = score_ch_pi(c_ideal)
        r_far = score_ch_pi(c_far)
        assert r_ideal['dG_ch_pi'] < r_far['dG_ch_pi']  # ideal more favorable

    def test_angle_weighting_reduces(self):
        """45 deg deviation weaker than perpendicular."""
        c_perp = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=3.5, angle_deviation_deg=0)]
        c_45 = [CHPiContact('C3-H', 'Trp', 'Trp', distance_A=3.5, angle_deviation_deg=45)]
        r_perp = score_ch_pi(c_perp)
        r_45 = score_ch_pi(c_45)
        assert r_perp['dG_ch_pi'] < r_45['dG_ch_pi']

    def test_aromatic_breakdown(self):
        contacts = [
            CHPiContact('C1-H', 'Trp181', 'Trp'),
            CHPiContact('C3-H', 'Tyr100', 'Tyr'),
        ]
        r = score_ch_pi(contacts, use_geometry=False)
        assert 'Trp181' in r['aromatic_breakdown']
        assert 'Tyr100' in r['aromatic_breakdown']

    def test_hevein_crosscheck(self):
        """Hevein double-Trp stacking: 2 aromatics x 3 CH each = ~15 kJ/mol.
        Asensio reports 6.3-8.4 kJ/mol total — but that's per stacking event,
        not per CH contact. At 3 contacts per Trp: 3 x -2.5 = -7.5. Consistent."""
        contacts = [
            CHPiContact(f'C{i}-H', 'Trp21', 'Trp')
            for i in [1, 3, 5]
        ]
        r = score_ch_pi(contacts, use_geometry=False)
        assert -8.5 < r['dG_ch_pi'] < -6.0  # consistent with Asensio 6.3-8.4


# =====================================================================
# CONTACT ESTIMATION
# =====================================================================

class TestContactEstimation:

    def test_mannose_one_trp(self):
        contacts = estimate_ch_pi_contacts('aMan', [{'residue': 'Trp181', 'type': 'Trp'}])
        assert len(contacts) == 3  # alpha face: C1-H, C3-H, C5-H

    def test_mannose_beta_face_blocked(self):
        contacts = estimate_ch_pi_contacts('aMan',
            [{'residue': 'Trp181', 'type': 'Trp', 'face': 'beta'}])
        assert len(contacts) == 3  # falls back to best (alpha)

    def test_unknown_sugar_empty(self):
        contacts = estimate_ch_pi_contacts('unknown', [{'residue': 'Trp', 'type': 'Trp'}])
        assert len(contacts) == 0

    def test_no_aromatics_empty(self):
        contacts = estimate_ch_pi_contacts('aMan', [])
        assert len(contacts) == 0


# =====================================================================
# SCORER INTEGRATION
# =====================================================================

class TestScorerIntegration:

    def test_cona_no_ch_pi(self):
        """ConA has no aromatic stacking — G3 should be zero."""
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cona_mannose_pocket())
        assert r.dG_ch_pi == 0.0

    def test_simple_count_path(self):
        """Simple count path: 3 contacts x -2.5 = -7.5."""
        cm = cona_mannose_pocket()
        cm.ch_pi_contacts = [
            ContactMapCHPi(['C1-H', 'C3-H', 'C5-H'], 'Trp181', 3)
        ]
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cm)
        assert abs(r.dG_ch_pi - (-7.5)) < 0.01

    def test_detailed_path(self):
        """Detailed path with aromatic weighting."""
        from mabe.glycan.ch_pi import CHPiContact as DetailedContact
        cm = cona_mannose_pocket()
        cm.ch_pi_contacts = [
            ContactMapCHPi(['C1-H', 'C3-H', 'C5-H'], 'Trp181', 3)
        ]
        cm.detailed_ch_pi_contacts = [
            DetailedContact('C1-H', 'Trp181', 'Trp'),
            DetailedContact('C3-H', 'Trp181', 'Trp'),
            DetailedContact('C5-H', 'Trp181', 'Trp'),
        ]
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cm)
        # Trp weight = 1.0, no geometry = ideal: 3 x -2.5 = -7.5
        assert abs(r.dG_ch_pi - (-7.5)) < 0.01

    def test_detailed_tyr_weaker_than_trp(self):
        """Tyr contacts give less stabilization than Trp."""
        from mabe.glycan.ch_pi import CHPiContact as DetailedContact
        cm_trp = cona_mannose_pocket()
        cm_trp.detailed_ch_pi_contacts = [
            DetailedContact('C3-H', 'Trp181', 'Trp'),
        ]
        cm_trp.ch_pi_contacts = [ContactMapCHPi(['C3-H'], 'Trp181', 1)]

        cm_tyr = cona_mannose_pocket()
        cm_tyr.detailed_ch_pi_contacts = [
            DetailedContact('C3-H', 'Tyr100', 'Tyr'),
        ]
        cm_tyr.ch_pi_contacts = [ContactMapCHPi(['C3-H'], 'Tyr100', 1)]

        r_trp = compute_glycan_terms(ALPHA_D_MANNOSE, cm_trp)
        r_tyr = compute_glycan_terms(ALPHA_D_MANNOSE, cm_tyr)
        assert r_trp.dG_ch_pi < r_tyr.dG_ch_pi  # Trp more negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
