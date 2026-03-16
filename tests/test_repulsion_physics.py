"""
test_repulsion_physics.py — Tests for 5-mechanism repulsion scoring.
"""
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repulsion_physics import (
    steric_repulsion, electrostatic_repulsion, donnan_repulsion_dG,
    hydrophobic_mismatch, hsab_mismatch, hsab_mismatch_for_site,
    geometric_frustration,
    score_repulsion, MaterialSiteSpec, RepulsionBreakdown,
    selectivity_from_differential_repulsion, SelectivityDecomposition,
    back_calculate_repulsion_from_selectivity,
    IONIC_RADII_A, HYDRATION_ENTHALPY_kJ, HSAB_HARDNESS_EV,
    DONOR_HARDNESS_EV, PREFERRED_CN,
    RT_STD,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data Tables
# ═══════════════════════════════════════════════════════════════════════════

class TestDataTables:

    def test_ionic_radii_positive(self):
        for sp, r in IONIC_RADII_A.items():
            assert r > 0, f"{sp}: radius must be positive"

    def test_hydration_enthalpy_negative(self):
        """Hydration is exothermic for all ions."""
        for sp, h in HYDRATION_ENTHALPY_kJ.items():
            assert h < 0, f"{sp}: hydration enthalpy must be negative"

    def test_hsab_hardness_positive(self):
        for sp, eta in HSAB_HARDNESS_EV.items():
            assert eta > 0, f"{sp}: hardness must be positive"

    def test_hard_ions_higher_eta(self):
        """Hard acids (Na+, Mg2+) should have higher η than soft (Ag+, Cu+)."""
        assert HSAB_HARDNESS_EV["Na+"] > HSAB_HARDNESS_EV["Ag+"]
        assert HSAB_HARDNESS_EV["Mg2+"] > HSAB_HARDNESS_EV["Cu+"]

    def test_preferred_cn_nonempty(self):
        for sp, cns in PREFERRED_CN.items():
            assert len(cns) > 0, f"{sp}: must have preferred CN"


# ═══════════════════════════════════════════════════════════════════════════
# R1: Steric Repulsion
# ═══════════════════════════════════════════════════════════════════════════

class TestStericRepulsion:

    def test_perfect_fit_minimal(self):
        """Species radius ≈ cavity radius → small repulsion."""
        dG = steric_repulsion(1.0, 1.0)
        assert dG < 1.0

    def test_oversized_species_large_repulsion(self):
        """Species > cavity → steep wall."""
        dG = steric_repulsion(2.0, 1.0)
        assert dG > 10.0

    def test_undersized_species_mild(self):
        """Species < cavity → mild loose-fit penalty."""
        dG = steric_repulsion(0.5, 2.0)
        assert 0 < dG < 5.0

    def test_larger_mismatch_more_repulsion(self):
        dG_small = steric_repulsion(1.2, 1.0)
        dG_large = steric_repulsion(2.0, 1.0)
        assert dG_large > dG_small

    def test_zero_cavity_no_repulsion(self):
        assert steric_repulsion(1.0, 0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# R2: Electrostatic Repulsion
# ═══════════════════════════════════════════════════════════════════════════

class TestElectrostaticRepulsion:

    def test_same_sign_repulsive(self):
        dG = electrostatic_repulsion(2, 1)
        assert dG > 0

    def test_opposite_sign_zero(self):
        """Opposite charges → attractive, function returns 0 (repulsive part only)."""
        dG = electrostatic_repulsion(-2, 1)
        assert dG == 0.0

    def test_higher_charge_stronger(self):
        dG1 = electrostatic_repulsion(1, 1)
        dG2 = electrostatic_repulsion(2, 1)
        assert dG2 > dG1

    def test_larger_distance_weaker(self):
        dG_close = electrostatic_repulsion(1, 1, r_A=2.0)
        dG_far = electrostatic_repulsion(1, 1, r_A=5.0)
        assert dG_close > dG_far


class TestDonnanRepulsion:

    def test_co_ion_repelled(self):
        """Same-sign species is repelled by Donnan."""
        dG = donnan_repulsion_dG(z_species=1, z_fixed=1,
                                   Q_meq_mL=2.0, C_ext_mM=1.0)
        assert dG > 0

    def test_counter_ion_not_repelled(self):
        """Opposite-sign species is NOT repelled."""
        dG = donnan_repulsion_dG(z_species=-1, z_fixed=1,
                                   Q_meq_mL=2.0, C_ext_mM=1.0)
        assert dG == 0.0

    def test_higher_Q_more_repulsion(self):
        dG_low = donnan_repulsion_dG(1, 1, 1.0, 1.0)
        dG_high = donnan_repulsion_dG(1, 1, 5.0, 1.0)
        assert dG_high > dG_low


# ═══════════════════════════════════════════════════════════════════════════
# R3: Hydrophobic Mismatch
# ═══════════════════════════════════════════════════════════════════════════

class TestHydrophobicMismatch:

    def test_strongly_hydrated_more_repulsion(self):
        """Mg²⁺ (ΔH_hyd=-1920) should suffer more than Cs⁺ (-263)."""
        dG_mg = hydrophobic_mismatch(-1920, 0.5)
        dG_cs = hydrophobic_mismatch(-263, 0.5)
        assert dG_mg > dG_cs

    def test_more_hydrophobic_cavity_more_repulsion(self):
        dG_polar = hydrophobic_mismatch(-1000, 0.1)
        dG_hydrophobic = hydrophobic_mismatch(-1000, 0.9)
        assert dG_hydrophobic > dG_polar

    def test_zero_hydrophobicity_no_mismatch(self):
        dG = hydrophobic_mismatch(-1000, 0.0)
        assert dG == 0.0

    def test_back_calc_li_k_zeolite(self):
        """Li/K selectivity in hydrophobic zeolite.
        |ΔH_hyd(Li)|-|ΔH_hyd(K)| = 520-321 = 199 kJ/mol.
        At f=0.3: ΔΔG = 0.05 × 199 × 0.3 ≈ 3 kJ/mol → ~1.2 log K.
        Published zeolite Li/K selectivity: α ≈ 0.5 → log α ≈ -0.3 (K preferred).
        Our model direction is correct: Li is MORE repelled."""
        dG_li = hydrophobic_mismatch(-520, 0.3)
        dG_k = hydrophobic_mismatch(-321, 0.3)
        assert dG_li > dG_k  # Li more repelled


# ═══════════════════════════════════════════════════════════════════════════
# R4: HSAB Mismatch
# ═══════════════════════════════════════════════════════════════════════════

class TestHSABMismatch:

    def test_hard_on_soft_repulsive(self):
        """Ca²⁺ (η=19.7) on S_thiol (η=5.0) → large mismatch."""
        dG = hsab_mismatch(19.7, 5.0)
        assert dG > 10.0

    def test_soft_on_soft_minimal(self):
        """Hg²⁺ (η=7.7) on S_thiol (η=5.0) → small mismatch."""
        dG = hsab_mismatch(7.7, 5.0)
        assert dG < 1.0

    def test_symmetric(self):
        """HSAB mismatch is symmetric: |Δη|² is the same both ways."""
        dG1 = hsab_mismatch(20.0, 5.0)
        dG2 = hsab_mismatch(5.0, 20.0)
        assert dG1 == pytest.approx(dG2)

    def test_identical_hardness_zero(self):
        dG = hsab_mismatch(10.0, 10.0)
        assert dG == 0.0

    def test_for_site_uses_best_donor(self):
        """With multiple donors, uses the best (minimum mismatch)."""
        # Pb2+ (η=8.5) with [S_thiol(5.0), O_carboxylate(15.0)]
        dG = hsab_mismatch_for_site("Pb2+", ["S_thiol", "O_carboxylate"])
        dG_thiol_only = hsab_mismatch(8.5, 5.0)
        dG_carb_only = hsab_mismatch(8.5, 15.0)
        assert dG == pytest.approx(dG_thiol_only)  # thiol is better match

    def test_no_donors_penalized(self):
        dG = hsab_mismatch_for_site("Ca2+", [])
        assert dG > 5.0

    def test_back_calc_thiol_hg_ca(self):
        """Thiol selectivity Hg/Ca: η(Hg)=7.7, η(Ca)=19.7, η(S)=5.0.
        ΔG(Ca-S) = 0.08×14.7² = 17.3 kJ/mol.
        ΔG(Hg-S) = 0.08×2.7² = 0.6 kJ/mol.
        ΔΔG = 16.7 kJ/mol → ~3 log K. Published: 3-5 log K. Consistent."""
        dG_hg = hsab_mismatch(HSAB_HARDNESS_EV["Hg2+"], DONOR_HARDNESS_EV["S_thiol"])
        dG_ca = hsab_mismatch(HSAB_HARDNESS_EV["Ca2+"], DONOR_HARDNESS_EV["S_thiol"])
        ddG = dG_ca - dG_hg
        log_alpha = ddG / (2.303 * RT_STD)
        assert 2.0 < log_alpha < 6.0  # published: 3-5 log units


# ═══════════════════════════════════════════════════════════════════════════
# R5: Geometric Frustration
# ═══════════════════════════════════════════════════════════════════════════

class TestGeometricFrustration:

    def test_matching_cn_zero_strain(self):
        """Ni²⁺ prefers CN=6, offered CN=6 → no strain."""
        dG = geometric_frustration("Ni2+", 6, "octahedral")
        assert dG < 1.0

    def test_mismatched_cn_positive(self):
        """Hg²⁺ prefers CN=2, offered CN=6 → large strain."""
        dG = geometric_frustration("Hg2+", 6, "octahedral")
        assert dG > 10.0

    def test_wrong_geometry_penalized(self):
        """Pd²⁺ prefers square planar, offered octahedral → penalty."""
        dG_sq = geometric_frustration("Pd2+", 4, "square_planar")
        dG_oct = geometric_frustration("Pd2+", 4, "octahedral")
        assert dG_oct > dG_sq

    def test_unknown_species_zero(self):
        dG = geometric_frustration("UNKNOWN99+", 6, "octahedral")
        assert dG == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Combined Scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestCombinedRepulsion:

    def _make_site(self, **kwargs):
        defaults = dict(cavity_radius_A=3.0, site_charge=0,
                        hydrophobicity=0.3,
                        donor_types=["O_carboxylate", "N_amine"],
                        offered_CN=6, offered_geometry="octahedral")
        defaults.update(kwargs)
        return MaterialSiteSpec(**defaults)

    def test_score_returns_breakdown(self):
        site = self._make_site()
        rb = score_repulsion(site, "Ca2+")
        assert isinstance(rb, RepulsionBreakdown)
        assert rb.dG_total_repulsion >= 0

    def test_all_five_mechanisms_computed(self):
        site = self._make_site()
        rb = score_repulsion(site, "Pb2+")
        assert rb.dG_steric >= 0
        assert rb.dG_electrostatic >= 0
        assert rb.dG_hydrophobic_mismatch >= 0
        assert rb.dG_hsab_mismatch >= 0
        assert rb.dG_geometric_frustration >= 0

    def test_dominant_mechanism_correct(self):
        """On a thiol site, HSAB should dominate for hard ions."""
        site = self._make_site(donor_types=["S_thiol"], hydrophobicity=0.1)
        rb = score_repulsion(site, "Mg2+")
        assert rb.dominant_mechanism == "HSAB"

    def test_soft_metal_low_repulsion_on_soft_site(self):
        site = self._make_site(donor_types=["S_thiol", "S_thioether"],
                               offered_CN=4, offered_geometry="tetrahedral")
        rb_hg = score_repulsion(site, "Hg2+")
        rb_ca = score_repulsion(site, "Ca2+")
        assert rb_hg.dG_total_repulsion < rb_ca.dG_total_repulsion


# ═══════════════════════════════════════════════════════════════════════════
# Selectivity from Differential Repulsion
# ═══════════════════════════════════════════════════════════════════════════

class TestDifferentialSelectivity:

    def test_returns_decomposition(self):
        site = MaterialSiteSpec(cavity_radius_A=3.0,
                                 donor_types=["S_thiol"],
                                 offered_CN=4,
                                 offered_geometry="tetrahedral")
        sd = selectivity_from_differential_repulsion(
            site, "Pb2+", "Ca2+", -25.0, -25.0
        )
        assert isinstance(sd, SelectivityDecomposition)

    def test_thiol_selects_pb_over_ca(self):
        """Thiol site should be selective for Pb²⁺ over Ca²⁺."""
        site = MaterialSiteSpec(cavity_radius_A=3.0,
                                 donor_types=["S_thiol"],
                                 hydrophobicity=0.3,
                                 offered_CN=4,
                                 offered_geometry="tetrahedral")
        sd = selectivity_from_differential_repulsion(
            site, "Pb2+", "Ca2+", -25.0, -25.0
        )
        assert sd.ddG_selectivity < 0  # negative = selective for target
        assert sd.log_selectivity > 0  # positive log α

    def test_equal_attraction_selectivity_from_repulsion(self):
        """With equal ΔG_attract, all selectivity comes from repulsion."""
        site = MaterialSiteSpec(donor_types=["S_thiol"])
        sd = selectivity_from_differential_repulsion(
            site, "Hg2+", "Mg2+", -25.0, -25.0
        )
        # Mg²⁺ is much more repelled → selectivity for Hg²⁺
        assert sd.log_selectivity > 5.0


class TestBackCalculation:

    def test_positive_alpha_positive_repulsion(self):
        """α > 1 with equal attraction → interferent is more repelled."""
        ddG_rep = back_calculate_repulsion_from_selectivity(
            alpha_published=100.0,
            dG_attract_target=-25.0,
            dG_attract_interferent=-25.0,
        )
        assert ddG_rep < 0  # interferent net more repelled → negative ΔΔG_rep

    def test_zero_selectivity_zero_repulsion(self):
        """α = 1 with equal attraction → zero differential repulsion."""
        ddG_rep = back_calculate_repulsion_from_selectivity(1.0, -25.0, -25.0)
        assert abs(ddG_rep) < 0.01

    def test_back_calc_consistent_with_forward(self):
        """Back-calculated repulsion should be consistent with SAC table."""
        # Pb²⁺/Ca²⁺ on SAC: α = 5.0/3.9 ≈ 1.28 (from Helfferich)
        ddG = back_calculate_repulsion_from_selectivity(
            alpha_published=1.28,
            dG_attract_target=-20.0,
            dG_attract_interferent=-20.0,
        )
        # Small ΔΔG because selectivity is modest
        assert abs(ddG) < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
