"""
Tests for glycan/bead_linker_design.py -- linker + bead geometry for pulldown.

Tests cover:
  - Minimum linker length scales with bead size
  - Entropy penalty < 2 kJ/mol at recommended PEG length
  - Entropy diverges when linker is too short
  - ESSENTIAL positions are infeasible
  - C2 warning propagates for ConA
  - Multivalent enhancement scales with bead size
  - Full pulldown design for MACS and Dynabeads
"""

import math
import pytest

from glycan.bead_linker_design import (
    compute_min_linker_length,
    compute_linker_entropy,
    peg_contour_length,
    peg_units_for_length,
    recommend_peg_length,
    estimate_multivalent_enhancement,
    design_pulldown,
    PEG_MONOMER_LENGTH_A,
)


# ── Linker length ───────────────────────────────────────────────────────

class TestMinLinkerLength:
    def test_scales_with_bead_size(self):
        """Larger bead -> longer linker."""
        small = compute_min_linker_length("ConA", "Man", "C1", bead_diameter_nm=50)
        large = compute_min_linker_length("ConA", "Man", "C1", bead_diameter_nm=1000)
        assert large["L_min_A"] > small["L_min_A"]

    def test_bead_radius_dominates(self):
        """For beads >> pocket depth, L_min ≈ bead_radius."""
        geom = compute_min_linker_length("ConA", "Man", "C1", bead_diameter_nm=1000)
        bead_radius = geom["bead_radius_A"]
        # L_min should be within 2x of bead radius (pocket + clearance are small)
        assert geom["L_min_A"] < 2.0 * bead_radius

    def test_equatorial_exit_longer(self):
        """Equatorial exit needs ~40% longer path than axial."""
        axial = compute_min_linker_length("ConA", "Man", "C1", bead_diameter_nm=50)
        equat = compute_min_linker_length("ConA", "Man", "C2", bead_diameter_nm=50)
        # Equatorial has exit penalty 1.4x on pocket depth component
        assert equat["L_min_A"] > axial["L_min_A"]

    def test_davis_deeper_pocket(self):
        """Davis (10 A pocket) needs slightly longer linker than ConA (8 A)."""
        cona = compute_min_linker_length("ConA", "Glc", "C1", bead_diameter_nm=50)
        davis = compute_min_linker_length("Davis", "Glc", "C1", bead_diameter_nm=50)
        assert davis["L_min_A"] > cona["L_min_A"]

    def test_unknown_position_raises(self):
        with pytest.raises(ValueError):
            compute_min_linker_length("ConA", "Man", "C3", bead_diameter_nm=50)


# ── Entropy ─────────────────────────────────────────────────────────────

class TestLinkerEntropy:
    def test_long_linker_low_entropy(self):
        """L_contour >> L_min -> near-zero entropy cost."""
        ddG = compute_linker_entropy(L_contour_A=10000, L_min_A=500)
        assert ddG < 0.1  # kJ/mol

    def test_tight_linker_high_entropy(self):
        """L_contour barely > L_min -> large entropy cost."""
        ddG = compute_linker_entropy(L_contour_A=550, L_min_A=500)
        assert ddG > 5.0  # kJ/mol

    def test_taut_linker_diverges(self):
        """L_min >= L_contour -> infinite (returned as 50 or inf)."""
        ddG = compute_linker_entropy(L_contour_A=500, L_min_A=500)
        assert ddG == float('inf')

    def test_ratio_dependence(self):
        """Entropy increases monotonically with L_min/L_contour."""
        ddG_low = compute_linker_entropy(L_contour_A=1000, L_min_A=300)
        ddG_high = compute_linker_entropy(L_contour_A=1000, L_min_A=700)
        assert ddG_high > ddG_low


# ── PEG sizing ──────────────────────────────────────────────────────────

class TestPEGSizing:
    def test_peg_contour_length(self):
        """PEG_100 = 100 * 3.5 = 350 A."""
        assert peg_contour_length(100) == 350.0

    def test_peg_units_for_length(self):
        """Need ceil(350/3.5) = 100 units for 350 A."""
        assert peg_units_for_length(350.0) == 100

    def test_peg_units_rounds_up(self):
        """351 A needs 101 units."""
        assert peg_units_for_length(351.0) == 101


# ── Linker recommendation ──────────────────────────────────────────────

class TestRecommendPEG:
    def test_macs_bead_feasible(self):
        """MACS beads (50 nm) should be feasible at ConA C1."""
        d = recommend_peg_length("ConA", "Man", "C1", bead_diameter_nm=50)
        assert d.feasible
        assert d.classification == "CANDIDATE"
        assert d.ddG_entropy_kJ < 2.0
        assert d.peg_n_recommended > 0

    def test_dynabead_feasible(self):
        """Dynabeads (1000 nm) should be feasible with longer PEG."""
        d = recommend_peg_length("ConA", "Man", "C1", bead_diameter_nm=1000)
        assert d.feasible
        assert d.peg_n_recommended > 100

    def test_essential_position_infeasible(self):
        """ESSENTIAL position -> not feasible."""
        d = recommend_peg_length("ConA", "Man", "C3", bead_diameter_nm=50)
        assert not d.feasible
        assert "ESSENTIAL" in d.infeasibility_reason

    def test_ring_position_infeasible(self):
        """C5 ring carbon -> not feasible."""
        d = recommend_peg_length("ConA", "Man", "C5", bead_diameter_nm=50)
        assert not d.feasible
        assert "ring carbon" in d.infeasibility_reason

    def test_c2_warning_propagates(self):
        """ConA C2 should carry the Schwarz warning."""
        d = recommend_peg_length("ConA", "Man", "C2", bead_diameter_nm=50)
        assert d.warning is not None
        assert "Schwarz" in d.warning

    def test_entropy_under_threshold(self):
        """At recommended PEG length, entropy should be < 2 kJ/mol."""
        for scaffold, ligand, pos in [("ConA", "Man", "C1"),
                                       ("Davis", "Glc", "C1"),
                                       ("PNA", "Gal", "C1"),
                                       ("Gal3", "Gal", "C1")]:
            d = recommend_peg_length(scaffold, ligand, pos, bead_diameter_nm=50)
            assert d.ddG_entropy_kJ < 2.0, \
                f"{scaffold}/{ligand} {pos}: entropy {d.ddG_entropy_kJ} >= 2.0"

    def test_larger_bead_longer_peg(self):
        """Larger bead -> longer recommended PEG."""
        small = recommend_peg_length("ConA", "Man", "C1", bead_diameter_nm=50)
        large = recommend_peg_length("ConA", "Man", "C1", bead_diameter_nm=500)
        assert large.peg_n_recommended > small.peg_n_recommended


# ── Multivalent enhancement ─────────────────────────────────────────────

class TestMultivalent:
    def test_larger_bead_more_contacts(self):
        """Larger bead -> more receptor contacts."""
        small = estimate_multivalent_enhancement(bead_diameter_nm=50)
        large = estimate_multivalent_enhancement(bead_diameter_nm=500)
        assert large.n_receptors_under_bead >= small.n_receptors_under_bead

    def test_enhancement_positive(self):
        """Any bead with contacts should show enhancement."""
        mv = estimate_multivalent_enhancement(bead_diameter_nm=100)
        assert mv.enhancement_log10 >= 0

    def test_enhancement_capped(self):
        """Enhancement should not exceed 10^6."""
        mv = estimate_multivalent_enhancement(bead_diameter_nm=5000)
        assert mv.enhancement_log10 <= 6.0

    def test_higher_density_more_receptors(self):
        """Higher receptor density -> more contacts."""
        low = estimate_multivalent_enhancement(bead_diameter_nm=100,
                                                receptor_density_per_um2=100)
        high = estimate_multivalent_enhancement(bead_diameter_nm=100,
                                                 receptor_density_per_um2=10000)
        assert high.n_receptors_under_bead >= low.n_receptors_under_bead


# ── Full pulldown design ────────────────────────────────────────────────

class TestPulldownDesign:
    def test_full_design_macs(self):
        """Full design for MACS bead pulldown via ConA/Man C1."""
        d = design_pulldown("ConA", "Man", "C1", bead_diameter_nm=50)
        assert d.linker.feasible
        assert d.multivalent.effective_valency >= 1
        assert "PEG" in d.summary

    def test_full_design_essential_blocked(self):
        """Essential position blocked in full design."""
        d = design_pulldown("ConA", "Man", "C4", bead_diameter_nm=50)
        assert not d.linker.feasible
        assert "NOT FEASIBLE" in d.summary

    def test_davis_c1_only_option(self):
        """Davis only has C1 as CANDIDATE; design should work."""
        d = design_pulldown("Davis", "Glc", "C1", bead_diameter_nm=50)
        assert d.linker.feasible
        assert d.linker.exit_class == "axial_out"
