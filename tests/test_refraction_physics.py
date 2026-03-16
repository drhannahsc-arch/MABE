"""
test_refraction_physics.py — Tests for 5-module refraction physics.
"""
import pytest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.refraction_physics import (
    # F1: Kramers-Kronig
    kramers_kronig_delta_n, chromophore_nk_spectrum,
    # F2: Fresnel
    fresnel_rs, fresnel_rp, fresnel_reflectance, fresnel_reflectance_spectrum,
    brewster_angle, critical_angle,
    # F3: Graded-index
    GradedIndexProfile, discretize_graded_index, anti_reflection_design,
    # F4: Lorentz-Lorenz
    lorentz_lorenz_n, predict_n_from_composition, predict_n_from_bonds,
    validate_lorentz_lorenz, BOND_POLARIZABILITY_A3, MOLECULAR_POLARIZABILITY_A3,
    # F5: Effective medium
    looyenga_ema, hashin_shtrikman_bounds, hashin_shtrikman_n_bounds,
    composite_shell_n,
    # Common
    _LAM,
)


# ═══════════════════════════════════════════════════════════════════════════
# F1: Kramers-Kronig
# ═══════════════════════════════════════════════════════════════════════════

class TestKramersKronig:

    def test_returns_array(self):
        k = np.zeros_like(_LAM)
        k[40] = 0.1  # spike at ~580nm
        dn = kramers_kronig_delta_n(k)
        assert len(dn) == len(_LAM)

    def test_absorption_produces_dispersion(self):
        """Non-zero k → non-zero Δn (KK mandates this)."""
        k = np.zeros_like(_LAM)
        sigma = 10.0 / 2.355
        k = 0.01 * np.exp(-0.5 * ((_LAM - 550) / sigma) ** 2)
        dn = kramers_kronig_delta_n(k)
        assert np.max(np.abs(dn)) > 0

    def test_zero_k_zero_dn(self):
        """No absorption → no dispersion."""
        k = np.zeros_like(_LAM)
        dn = kramers_kronig_delta_n(k)
        assert np.allclose(dn, 0.0, atol=1e-10)

    def test_anomalous_dispersion_shape(self):
        """Near absorption peak: Δn should change sign (anomalous dispersion).
        Below peak: Δn > 0, above peak: Δn < 0 (or vice versa)."""
        k = 0.05 * np.exp(-0.5 * ((_LAM - 550) / 10) ** 2)
        dn = kramers_kronig_delta_n(k)
        # Should have both positive and negative regions
        assert np.min(dn) < 0
        assert np.max(dn) > 0

    def test_chromophore_nk_returns_pair(self):
        n, k = chromophore_nk_spectrum("CuPc", n_baseline=1.60)
        assert len(n) == len(_LAM)
        assert len(k) == len(_LAM)
        assert np.mean(n) > 1.0  # n should be around baseline


# ═══════════════════════════════════════════════════════════════════════════
# F2: Fresnel
# ═══════════════════════════════════════════════════════════════════════════

class TestFresnel:

    def test_normal_incidence_air_glass(self):
        """Air→glass at normal: R = ((n-1)/(n+1))² = ((1.5-1)/(1.5+1))² ≈ 0.04."""
        R = fresnel_reflectance(1.0, 1.5, 0.0)
        expected = ((1.5 - 1) / (1.5 + 1)) ** 2
        assert abs(R - expected) < 0.001

    def test_same_medium_zero_reflection(self):
        R = fresnel_reflectance(1.5, 1.5, 0.0)
        assert R < 1e-10

    def test_reflectance_increases_with_angle(self):
        R_0 = fresnel_reflectance(1.0, 1.5, 0.0)
        R_60 = fresnel_reflectance(1.0, 1.5, 60.0)
        assert R_60 > R_0

    def test_grazing_incidence_near_one(self):
        R = fresnel_reflectance(1.0, 1.5, 89.0)
        assert R > 0.9

    def test_brewster_angle_value(self):
        theta_B = brewster_angle(1.0, 1.5)
        expected = math.degrees(math.atan(1.5))
        assert abs(theta_B - expected) < 0.1

    def test_rp_zero_at_brewster(self):
        """At Brewster's angle, R_p = 0."""
        theta_B = brewster_angle(1.0, 1.5)
        R_p = fresnel_reflectance(1.0, 1.5, theta_B, "p")
        assert R_p < 1e-6

    def test_critical_angle_exists_for_dense_to_rare(self):
        theta_c = critical_angle(1.5, 1.0)
        assert theta_c is not None
        expected = math.degrees(math.asin(1.0 / 1.5))
        assert abs(theta_c - expected) < 0.1

    def test_no_critical_angle_for_rare_to_dense(self):
        assert critical_angle(1.0, 1.5) is None

    def test_total_internal_reflection(self):
        """Beyond critical angle, R = 1 (TIR)."""
        theta_c = critical_angle(1.5, 1.0)
        R = fresnel_reflectance(1.5, 1.0, theta_c + 1.0)
        assert R > 0.99

    def test_spectrum_shape(self):
        n1 = np.full(81, 1.0)
        n2 = np.linspace(1.3, 1.7, 81)
        R = fresnel_reflectance_spectrum(n1, n2, 0.0)
        assert len(R) == 81
        # Higher n2 → higher R
        assert R[-1] > R[0]


# ═══════════════════════════════════════════════════════════════════════════
# F3: Graded-Index
# ═══════════════════════════════════════════════════════════════════════════

class TestGradedIndex:

    def test_linear_endpoints(self):
        p = GradedIndexProfile(n_inner=1.5, n_outer=1.0, profile_type="linear")
        assert p.n_at_x(0.0) == pytest.approx(1.5)
        assert p.n_at_x(1.0) == pytest.approx(1.0)

    def test_linear_midpoint(self):
        p = GradedIndexProfile(n_inner=1.5, n_outer=1.0, profile_type="linear")
        assert p.n_at_x(0.5) == pytest.approx(1.25)

    def test_quintic_endpoints(self):
        p = GradedIndexProfile(n_inner=1.5, n_outer=1.0, profile_type="quintic")
        assert p.n_at_x(0.0) == pytest.approx(1.5)
        assert p.n_at_x(1.0) == pytest.approx(1.0)

    def test_quintic_smooth(self):
        """Quintic should be smoother at endpoints than linear."""
        p = GradedIndexProfile(n_inner=1.5, n_outer=1.0, profile_type="quintic")
        # Derivative at x=0 should be ~0 for quintic (zero at endpoints)
        dn_dx = (p.n_at_x(0.01) - p.n_at_x(0.0)) / 0.01
        assert abs(dn_dx) < 0.1  # near-zero slope at boundary

    def test_exponential_monotonic(self):
        p = GradedIndexProfile(n_inner=1.5, n_outer=1.0, profile_type="exponential")
        n_vals = [p.n_at_x(x) for x in np.linspace(0, 1, 20)]
        for i in range(len(n_vals) - 1):
            assert n_vals[i + 1] <= n_vals[i]  # monotonically decreasing

    def test_discretize_count(self):
        p = GradedIndexProfile(total_thickness_nm=100.0)
        layers = discretize_graded_index(p, n_sublayers=10)
        assert len(layers) == 10

    def test_discretize_thickness_sum(self):
        p = GradedIndexProfile(total_thickness_nm=100.0)
        layers = discretize_graded_index(p, n_sublayers=10)
        total = sum(t for _, t in layers)
        assert abs(total - 100.0) < 0.01

    def test_anti_reflection_design(self):
        ar = anti_reflection_design(1.46, 1.0, 550.0)
        assert ar.n_inner == pytest.approx(1.46)
        assert ar.n_outer == pytest.approx(1.0)
        assert ar.total_thickness_nm > 100  # ~223nm for SiO2

    def test_rugate_oscillates(self):
        p = GradedIndexProfile(n_inner=1.4, n_outer=1.2, profile_type="rugate",
                                total_thickness_nm=500.0,
                                rugate_period_nm=100.0, rugate_amplitude=0.1)
        n_vals = [p.n_at_x(x) for x in np.linspace(0, 1, 100)]
        # Should oscillate — check that it crosses the linear baseline
        n_linear = [1.4 + (1.2 - 1.4) * x for x in np.linspace(0, 1, 100)]
        crossings = sum(1 for i in range(len(n_vals) - 1)
                        if (n_vals[i] - n_linear[i]) * (n_vals[i+1] - n_linear[i+1]) < 0)
        assert crossings >= 4  # should cross multiple times


# ═══════════════════════════════════════════════════════════════════════════
# F4: Lorentz-Lorenz
# ═══════════════════════════════════════════════════════════════════════════

class TestLorentzLorenz:

    def test_zero_polarizability_returns_one(self):
        assert lorentz_lorenz_n(0.0, 1e28) == pytest.approx(1.0)

    def test_higher_alpha_higher_n(self):
        N = 3e28
        n_low = lorentz_lorenz_n(2.0, N)
        n_high = lorentz_lorenz_n(10.0, N)
        assert n_high > n_low

    def test_higher_density_higher_n(self):
        alpha = 5.0
        n_sparse = lorentz_lorenz_n(alpha, 1e27)
        n_dense = lorentz_lorenz_n(alpha, 3e28)
        assert n_dense > n_sparse

    def test_water_accurate(self):
        """Water: α=1.45 ų, MW=18, ρ=1.0 → n≈1.33."""
        n = predict_n_from_composition(1.45, 18.015, 1.0)
        assert abs(n - 1.33) < 0.01

    def test_pmma_accurate(self):
        """PMMA: α=8.95, MW=100.12, ρ=1.18 → n≈1.49."""
        n = predict_n_from_composition(8.95, 100.12, 1.18)
        assert abs(n - 1.49) < 0.10

    def test_polystyrene_accurate(self):
        n = predict_n_from_composition(14.5, 104.15, 1.05)
        assert abs(n - 1.59) < 0.10

    def test_bond_additive(self):
        """Simple alkane: predict n from bond polarizabilities."""
        # Methane: 4 C-H bonds
        n = predict_n_from_bonds({"C-H": 4}, 16.04, 0.42)
        assert n > 1.0

    def test_aromatic_higher_n(self):
        """Aromatic bonds have higher polarizability → higher n."""
        n_aliphatic = predict_n_from_bonds({"C-C": 5, "C-H": 12}, 72.15, 0.63)
        n_aromatic = predict_n_from_bonds({"aromatic_C": 6, "C-H": 6}, 78.11, 0.88)
        assert n_aromatic > n_aliphatic

    def test_bond_table_has_common_bonds(self):
        essential = ["C-C", "C=C", "C-H", "C-O", "C=O", "C-N", "O-H", "N-H"]
        for bond in essential:
            assert bond in BOND_POLARIZABILITY_A3

    def test_molecular_table_has_key_materials(self):
        for mat in ["SiO2", "TiO2", "water", "PMMA_monomer", "styrene"]:
            assert mat in MOLECULAR_POLARIZABILITY_A3

    def test_validation_molecular_materials_accurate(self):
        """Molecular materials should predict n within 0.10."""
        results = validate_lorentz_lorenz()
        for name, n_pub, n_pred, err, cat in results:
            if cat == "molecular":
                assert err < 0.10, f"{name}: error {err:.3f} exceeds 0.10"


# ═══════════════════════════════════════════════════════════════════════════
# F5: Effective Medium
# ═══════════════════════════════════════════════════════════════════════════

class TestEffectiveMedium:

    def test_looyenga_pure_a(self):
        n = looyenga_ema(1.5, 1.0, 1.0)
        assert abs(n - 1.5) < 0.01

    def test_looyenga_pure_b(self):
        n = looyenga_ema(1.5, 1.0, 0.0)
        assert abs(n - 1.0) < 0.01

    def test_looyenga_intermediate(self):
        n = looyenga_ema(1.5, 1.0, 0.5)
        assert 1.0 < n < 1.5

    def test_looyenga_symmetric(self):
        """Looyenga should be symmetric in A,B when swapped with fractions."""
        n1 = looyenga_ema(1.5, 1.2, 0.4)
        n2 = looyenga_ema(1.2, 1.5, 0.6)
        assert abs(n1 - n2) < 0.01

    def test_hs_bounds_bracket(self):
        """HS bounds should bracket Looyenga."""
        eps_lo, eps_hi = hashin_shtrikman_bounds(1.5**2, 1.0**2, 0.5)
        n_lo, n_hi = math.sqrt(eps_lo), math.sqrt(eps_hi)
        n_L = looyenga_ema(1.5, 1.0, 0.5)
        assert n_lo <= n_L <= n_hi

    def test_hs_bounds_narrow_for_similar_n(self):
        """Similar components → narrow bounds."""
        n_lo, n_hi = hashin_shtrikman_n_bounds(1.5, 1.4, 0.5)
        assert n_hi - n_lo < 0.01

    def test_hs_bounds_wide_for_different_n(self):
        """Very different components → wide bounds."""
        n_lo, n_hi = hashin_shtrikman_n_bounds(2.5, 1.0, 0.5)
        assert n_hi - n_lo > 0.1

    def test_porous_silica_looyenga(self):
        """30% porous SiO₂: n should be ~1.29-1.32."""
        n = looyenga_ema(1.46, 1.0, 0.7)
        assert 1.25 < n < 1.35

    def test_composite_shell_n(self):
        n = composite_shell_n([("SiO2", 0.7), ("air", 0.3)])
        assert 1.2 < n < 1.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
