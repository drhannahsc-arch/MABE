"""
tests/test_glycan_g5_structural_water.py -- G5 structural water tests
======================================================================

Tests:
  1. Parameter integrity (eps_water_bridge, norm)
  2. Scoring correctness (zero, single, multiple waters)
  3. Water estimation from SASA
  4. Osmotic stress data consistency
  5. Scorer integration (ConA monosaccharide vs trimannoside)
  6. Regression: existing G1-G4 tests unaffected
"""

import pytest

from mabe.glycan.structural_water import (
    EPS_WATER_BRIDGE, N_WATER_BRIDGE_NORM,
    compute_structural_water_energy, estimate_conserved_waters,
    OSMOTIC_STRESS_DATA, CONSERVED_WATERS,
)
from mabe.glycan.params import GLYCAN_PARAMS
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
from mabe.glycan.contact_map import cona_mannose_pocket, cona_trimannoside


# =====================================================================
# PARAMETER INTEGRITY
# =====================================================================

class TestParameters:

    def test_eps_water_bridge_value(self):
        assert EPS_WATER_BRIDGE == -3.5

    def test_eps_in_range(self):
        """Literature range: -2 to -8 kJ/mol per water."""
        assert -8.0 < EPS_WATER_BRIDGE < -1.0

    def test_params_match(self):
        """GlycanParams singleton should match module constant."""
        assert GLYCAN_PARAMS.eps_water_bridge == -3.5

    def test_norm_positive(self):
        assert N_WATER_BRIDGE_NORM > 0

    def test_norm_physically_reasonable(self):
        """0.01-0.03 waters per A^2 is reasonable for polar interfaces."""
        assert 0.005 < N_WATER_BRIDGE_NORM < 0.05


# =====================================================================
# SCORING
# =====================================================================

class TestScoring:

    def test_zero_waters(self):
        r = compute_structural_water_energy(0)
        assert r['dG_water'] == 0.0
        assert r['n_waters'] == 0

    def test_one_water(self):
        r = compute_structural_water_energy(1)
        assert abs(r['dG_water'] - (-3.5)) < 0.01

    def test_three_waters(self):
        r = compute_structural_water_energy(3)
        assert abs(r['dG_water'] - (-10.5)) < 0.01

    def test_negative_waters(self):
        r = compute_structural_water_energy(-1)
        assert r['dG_water'] == 0.0

    def test_favorable(self):
        """Structural waters always contribute favorably."""
        r = compute_structural_water_energy(2)
        assert r['dG_water'] < 0


# =====================================================================
# SASA ESTIMATION
# =====================================================================

class TestSASAEstimation:

    def test_zero_sasa(self):
        assert estimate_conserved_waters(0) == 0

    def test_negative_sasa(self):
        assert estimate_conserved_waters(-10) == 0

    def test_100_A2(self):
        """100 A^2 polar SASA at 0.015 norm = 1 water."""
        n = estimate_conserved_waters(100.0)
        assert n == 1

    def test_200_A2(self):
        n = estimate_conserved_waters(200.0)
        assert n >= 2

    def test_scaling(self):
        """More SASA = more waters."""
        n1 = estimate_conserved_waters(50.0)
        n2 = estimate_conserved_waters(200.0)
        assert n2 >= n1


# =====================================================================
# OSMOTIC STRESS DATA
# =====================================================================

class TestOsmoticStressData:

    def test_data_present(self):
        assert len(OSMOTIC_STRESS_DATA) >= 4

    def test_mannose_most_waters_released(self):
        """Mannose releases the most waters (5)."""
        man = next(d for d in OSMOTIC_STRESS_DATA if d.ligand == 'D-mannose')
        assert man.n_waters_released == 5

    def test_trimannoside_fewest_released(self):
        """Trimannoside retains most water (only 1 released)."""
        tri = next(d for d in OSMOTIC_STRESS_DATA if d.ligand == 'trimannoside')
        assert tri.n_waters_released == 1

    def test_ordering(self):
        """Man(5) > disaccharides(3) > trimannoside(1)."""
        man = next(d for d in OSMOTIC_STRESS_DATA if d.ligand == 'D-mannose')
        tri = next(d for d in OSMOTIC_STRESS_DATA if d.ligand == 'trimannoside')
        assert man.n_waters_released > tri.n_waters_released

    def test_conserved_water_39(self):
        """Water 39 in ConA is strictly conserved."""
        w39 = CONSERVED_WATERS.get('ConA_W39')
        assert w39 is not None
        assert w39.conservation == 'strict'
        assert 'Asn14' in w39.anchored_by


# =====================================================================
# SCORER INTEGRATION
# =====================================================================

class TestScorerIntegration:

    def test_monosaccharide_has_water(self):
        """ConA monosaccharide pocket has 1 conserved water."""
        cm = cona_mannose_pocket()
        assert cm.n_conserved_waters == 1

    def test_trimannoside_has_water(self):
        cm = cona_trimannoside()
        assert cm.n_conserved_waters == 1

    def test_water_contributes_to_score(self):
        """Structural water should make score more favorable."""
        # Score with water
        cm_with = cona_mannose_pocket()
        cm_with.n_conserved_waters = 1
        r_with = compute_glycan_terms(ALPHA_D_MANNOSE, cm_with)

        # Score without water
        cm_without = cona_mannose_pocket()
        cm_without.n_conserved_waters = 0
        r_without = compute_glycan_terms(ALPHA_D_MANNOSE, cm_without)

        assert r_with.dG_total < r_without.dG_total
        water_contribution = r_with.dG_structural_water - r_without.dG_structural_water
        assert abs(water_contribution - (-3.5)) < 0.01

    def test_no_water_receptor_zero(self):
        """Davis receptor has n_conserved_waters=0, so water term = 0."""
        from mabe.glycan.validation_g4 import davis_receptor_contacts
        cm = davis_receptor_contacts('aGlc')
        r = compute_glycan_terms(ALPHA_D_MANNOSE, cm)
        assert r.dG_structural_water == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
