"""
tests/test_colloidal_thermal.py — Tests for colloidal thermal conductivity model.

Validates:
  - Maxwell-Garnett baseline against analytical limits
  - Hasselman-Johnson with Kapitza interface resistance
  - Core-shell extension
  - Material and Kapitza databases
  - Interface density computation
  - R-value and thermal resistance
  - Bridge function from optical pipeline parameters
  - Physical sanity (κ_eff between matrix and particle, interfaces reduce κ)
  - Still et al. 2008 validation target for SiO₂ colloidal crystal
"""

import sys
import os
import math
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.colloidal_thermal import (
    maxwell_garnett,
    hasselman_johnson,
    hasselman_johnson_core_shell,
    interface_density,
    predict_thermal,
    predict_from_optical,
    predict_core_shell,
    get_material,
    get_kapitza,
    list_materials,
    list_kapitza_pairs,
    ColloidalThermalSpec,
    ThermalResult,
    ThermalMaterial,
    _MATERIALS,
    _estimate_kapitza_amm,
)


# ═══════════════════════════════════════════════════════════════════════════
# Maxwell-Garnett analytical tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxwellGarnett:

    def test_zero_phi_returns_matrix(self):
        """φ = 0 → pure matrix."""
        assert maxwell_garnett(10.0, 1.0, 0.0) == pytest.approx(1.0, rel=0.001)

    def test_dilute_limit(self):
        """Very low φ → close to matrix κ."""
        result = maxwell_garnett(10.0, 1.0, 0.01)
        assert 1.0 < result < 1.1

    def test_high_contrast_particles_increase_kappa(self):
        """κ_p >> κ_m → κ_eff > κ_m."""
        result = maxwell_garnett(100.0, 0.1, 0.3)
        assert result > 0.1

    def test_low_conductivity_particles_decrease_kappa(self):
        """κ_p < κ_m → κ_eff < κ_m (e.g., air pores in solid)."""
        result = maxwell_garnett(0.026, 1.0, 0.3)
        assert result < 1.0

    def test_equal_conductivities(self):
        """κ_p = κ_m → κ_eff = κ_m regardless of φ."""
        assert maxwell_garnett(1.0, 1.0, 0.5) == pytest.approx(1.0, rel=0.001)

    def test_zero_matrix_kappa(self):
        """κ_m = 0 → κ_eff = 0 (disconnected particles can't conduct)."""
        assert maxwell_garnett(10.0, 0.0, 0.3) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Hasselman-Johnson tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHasselmanJohnson:

    def test_zero_kapitza_equals_mg(self):
        """R_K = 0 → reverts to Maxwell-Garnett."""
        mg = maxwell_garnett(1.3, 0.15, 0.5)
        hj = hasselman_johnson(1.3, 0.15, 0.5, 125e-9, 0.0)
        assert hj == pytest.approx(mg, rel=0.01)

    def test_kapitza_reduces_kappa(self):
        """Adding Kapitza resistance should reduce κ_eff vs no-interface case."""
        hj_no_R = hasselman_johnson(1.3, 0.15, 0.5, 125e-9, 0.0)
        hj_with_R = hasselman_johnson(1.3, 0.15, 0.5, 125e-9, 5e-8)
        assert hj_with_R < hj_no_R

    def test_higher_kapitza_lower_kappa(self):
        """Higher R_K → lower κ_eff."""
        hj_low = hasselman_johnson(1.3, 0.15, 0.5, 125e-9, 1e-8)
        hj_high = hasselman_johnson(1.3, 0.15, 0.5, 125e-9, 1e-7)
        assert hj_high < hj_low

    def test_smaller_particles_more_interfaces(self):
        """Smaller particles at same φ → more interfaces → lower κ_eff."""
        hj_big = hasselman_johnson(1.3, 0.15, 0.5, 200e-9, 5e-8)
        hj_small = hasselman_johnson(1.3, 0.15, 0.5, 50e-9, 5e-8)
        assert hj_small < hj_big

    def test_kappa_eff_positive(self):
        """κ_eff should always be positive."""
        result = hasselman_johnson(1.3, 0.026, 0.5, 125e-9, 1e-6)
        assert result > 0.0

    def test_silica_in_air_typical_range(self):
        """SiO₂ spheres in air at φ=0.5: κ_eff should be 0.01–0.5 W/mK."""
        result = hasselman_johnson(1.3, 0.026, 0.5, 125e-9, 1e-8)
        assert 0.01 < result < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Core-shell extension tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCoreShellHJ:

    def test_zero_shell_equals_solid(self):
        """Shell thickness = 0 → same as solid particle HJ."""
        hj_solid = hasselman_johnson(1.3, 0.15, 0.5, 125e-9, 5e-8)
        hj_cs = hasselman_johnson_core_shell(
            kappa_core=1.3, kappa_shell=1.3,  # shell = core
            kappa_matrix=0.15,
            core_radius_m=125e-9, shell_thickness_m=0.0,
            phi=0.5,
            R_kapitza_core_shell=0.0,
            R_kapitza_shell_matrix=5e-8,
        )
        # Should be close (not exact due to zero-shell path)
        assert abs(hj_cs - hj_solid) / hj_solid < 0.2

    def test_soft_shell_reduces_kappa(self):
        """Soft polymer shell (low κ) around hard core → lower κ_eff than solid particles."""
        hj_solid = hasselman_johnson(1.3, 0.026, 0.5, 150e-9, 5e-8)
        hj_cs = hasselman_johnson_core_shell(
            kappa_core=1.3, kappa_shell=0.15,  # silicone shell
            kappa_matrix=0.026,
            core_radius_m=100e-9, shell_thickness_m=50e-9,
            phi=0.5,
            R_kapitza_core_shell=3e-8,
            R_kapitza_shell_matrix=5e-8,
        )
        assert hj_cs < hj_solid

    def test_two_interfaces_lower_than_one(self):
        """Core-shell with R_K at both interfaces → lower κ than single interface."""
        hj_one = hasselman_johnson(1.3, 0.026, 0.5, 150e-9, 5e-8)
        hj_two = hasselman_johnson_core_shell(
            kappa_core=1.3, kappa_shell=1.3,
            kappa_matrix=0.026,
            core_radius_m=100e-9, shell_thickness_m=50e-9,
            phi=0.5,
            R_kapitza_core_shell=5e-8,  # additional inner interface
            R_kapitza_shell_matrix=5e-8,
        )
        assert hj_two < hj_one

    def test_core_shell_positive(self):
        result = hasselman_johnson_core_shell(
            1.3, 0.15, 0.026, 100e-9, 50e-9, 0.5, 3e-8, 5e-8)
        assert result > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Interface density tests
# ═══════════════════════════════════════════════════════════════════════════

class TestInterfaceDensity:

    def test_smaller_particles_more_interfaces(self):
        n_big = interface_density(500.0, 0.5)
        n_small = interface_density(100.0, 0.5)
        assert n_small > n_big

    def test_higher_phi_more_interfaces(self):
        n_low = interface_density(250.0, 0.2)
        n_high = interface_density(250.0, 0.6)
        assert n_high > n_low

    def test_typical_colloidal_order_of_magnitude(self):
        """250 nm particles at φ=0.5 → ~3×10⁶ interfaces/m."""
        n = interface_density(250.0, 0.5)
        assert 1e6 < n < 1e7

    def test_zero_diameter_zero_interfaces(self):
        assert interface_density(0.0, 0.5) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Material database tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMaterialDatabase:

    def test_sio2_in_database(self):
        mat = get_material("SiO2")
        assert mat is not None
        assert mat.kappa_W_mK == pytest.approx(1.3, rel=0.1)

    def test_air_in_database(self):
        mat = get_material("air")
        assert mat is not None
        assert mat.kappa_W_mK == pytest.approx(0.026, rel=0.1)

    def test_impedance_positive(self):
        for name, mat in _MATERIALS.items():
            assert mat.impedance_MRayl > 0, f"{name} has zero impedance"

    def test_diffusivity_positive(self):
        for name, mat in _MATERIALS.items():
            assert mat.diffusivity_m2_s > 0, f"{name} has zero diffusivity"

    def test_list_materials_complete(self):
        names = list_materials()
        assert "SiO2" in names
        assert "air" in names
        assert "silicone" in names
        assert len(names) >= 15


# ═══════════════════════════════════════════════════════════════════════════
# Kapitza database tests
# ═══════════════════════════════════════════════════════════════════════════

class TestKapitzaDatabase:

    def test_sio2_silicone_exists(self):
        R_K = get_kapitza("SiO2", "silicone")
        assert R_K > 0
        assert 1e-9 < R_K < 1e-5

    def test_reverse_order_same_result(self):
        R_ab = get_kapitza("SiO2", "silicone")
        R_ba = get_kapitza("silicone", "SiO2")
        assert R_ab == R_ba

    def test_unknown_pair_uses_amm(self):
        """Unknown pair should fall back to AMM estimate."""
        R_K = get_kapitza("ZnS", "polyurethane")
        assert R_K > 0

    def test_amm_estimate_physical(self):
        """AMM estimate should be in physical range [10⁻⁹, 10⁻⁵)."""
        R_K = _estimate_kapitza_amm("BaTiO3", "PMMA")
        assert 1e-9 <= R_K < 1e-5

    def test_list_kapitza_pairs_nonempty(self):
        pairs = list_kapitza_pairs()
        assert len(pairs) >= 15

    def test_air_interfaces_highest_resistance(self):
        """Gas-solid interfaces should have highest R_K."""
        R_sio2_air = get_kapitza("SiO2", "air")
        R_sio2_sil = get_kapitza("SiO2", "silicone")
        assert R_sio2_air > R_sio2_sil


# ═══════════════════════════════════════════════════════════════════════════
# Full prediction tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPrediction:

    def test_silica_in_air_basic(self):
        """Standard photonic glass: SiO₂ in air."""
        spec = ColloidalThermalSpec(
            particle_material="SiO2",
            particle_diameter_nm=250.0,
            volume_fraction=0.50,
            matrix_material="air",
            film_thickness_um=50.0,
        )
        result = predict_thermal(spec)
        assert isinstance(result, ThermalResult)
        assert result.kappa_eff_W_mK > 0
        assert result.R_value_m2KW > 0
        assert result.model == "hasselman_johnson"

    def test_kappa_eff_between_bounds(self):
        """κ_eff should be between κ_air and κ_SiO2."""
        result = predict_from_optical(250.0, "SiO2", 0.5, "air", 50.0)
        assert result.kappa_bulk_matrix < result.kappa_eff_W_mK
        # With Kapitza, κ_eff can be below MG prediction
        assert result.kappa_eff_W_mK < result.kappa_bulk_particle

    def test_hasselman_lower_than_mg(self):
        """HJ with Kapitza should give lower κ than Maxwell-Garnett."""
        result = predict_from_optical(250.0, "SiO2", 0.5, "silicone", 50.0)
        assert result.kappa_hasselman_johnson_W_mK <= result.kappa_maxwell_garnett_W_mK

    def test_kapitza_fraction_positive(self):
        """Interface resistance should be a measurable fraction of total R."""
        result = predict_from_optical(250.0, "SiO2", 0.5, "silicone", 50.0)
        assert result.R_kapitza_total_fraction > 0.0

    def test_thicker_film_higher_r_value(self):
        """Thicker film → higher R-value."""
        thin = predict_from_optical(250.0, "SiO2", 0.5, "air", 10.0)
        thick = predict_from_optical(250.0, "SiO2", 0.5, "air", 100.0)
        assert thick.R_value_m2KW > thin.R_value_m2KW

    def test_core_shell_prediction(self):
        """Core-shell (SiO₂ core + silicone shell) in air."""
        result = predict_core_shell(
            core_material="SiO2",
            core_diameter_nm=200.0,
            shell_material="silicone",
            shell_thickness_nm=25.0,
            volume_fraction=0.50,
            matrix_material="air",
            film_thickness_um=50.0,
        )
        assert result.model == "core_shell_hjm"
        assert result.kappa_eff_W_mK > 0
        assert result.R_kapitza_core_shell > 0
        assert result.R_kapitza_shell_matrix > 0

    def test_core_shell_lower_kappa_than_solid(self):
        """Core-shell with soft shell → lower κ than solid particles."""
        solid = predict_from_optical(250.0, "SiO2", 0.5, "air", 50.0)
        cs = predict_core_shell("SiO2", 200.0, "silicone", 25.0, 0.5, "air", 50.0)
        assert cs.kappa_eff_W_mK < solid.kappa_eff_W_mK

    def test_n_layers_autocomputed(self):
        spec = ColloidalThermalSpec(
            particle_material="SiO2",
            particle_diameter_nm=250.0,
            volume_fraction=0.50,
            matrix_material="air",
            film_thickness_um=50.0,
        )
        assert spec.n_layers == 200  # 50000 nm / 250 nm

    def test_summary_not_empty(self):
        result = predict_from_optical(250.0, "SiO2", 0.5, "air", 50.0)
        s = result.summary()
        assert "κ_eff" in s
        assert "R-value" in s

    def test_delta_t_present(self):
        result = predict_from_optical(250.0, "SiO2", 0.5, "air", 50.0)
        assert result.delta_T_surface_C is not None


# ═══════════════════════════════════════════════════════════════════════════
# Validation target: Still et al. PRB 2008
# ═══════════════════════════════════════════════════════════════════════════

class TestStillValidation:
    """Still et al. (Phys. Rev. B 2008, 78, 125426) measured κ of SiO₂
    colloidal crystals (FCC, sintered contacts).

    Measured κ_eff ≈ 0.03–0.12 W/mK for 300–1000 nm SiO₂ opals.
    Key finding: κ dominated by contact resistance, not bulk SiO₂.

    Our model should produce values in this range for similar parameters.
    """

    def test_still_300nm_opal(self):
        """300 nm SiO₂ FCC opal: κ should be 0.01–0.2 W/mK."""
        result = predict_from_optical(300.0, "SiO2", 0.74, "air", 100.0)
        assert 0.01 < result.kappa_eff_W_mK < 0.2, \
            f"κ_eff = {result.kappa_eff_W_mK} outside Still et al. range"

    def test_still_interface_dominated(self):
        """In Still's measurement, contact/interface resistance dominates."""
        result = predict_from_optical(300.0, "SiO2", 0.74, "air", 100.0)
        assert result.R_kapitza_total_fraction > 0.2, \
            f"Kapitza fraction {result.R_kapitza_total_fraction} too low — " \
            f"should be interface-dominated per Still et al."

    def test_still_size_dependence(self):
        """Smaller particles → lower κ (more interfaces per unit length)."""
        k_300 = predict_from_optical(300.0, "SiO2", 0.74, "air", 100.0).kappa_eff_W_mK
        k_600 = predict_from_optical(600.0, "SiO2", 0.74, "air", 100.0).kappa_eff_W_mK
        assert k_300 < k_600


# ═══════════════════════════════════════════════════════════════════════════
# Multi-physics bridge tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiPhysicsBridge:
    """Test that the thermal model uses the same parameters as the optical pipeline."""

    def test_color_parameters_are_thermal_parameters(self):
        """The particle spec for green structural color also predicts thermal κ."""
        # Green photonic glass: ~280 nm SiO₂, φ=0.50
        result = predict_from_optical(
            particle_diameter_nm=280.0,
            particle_material="SiO2",
            volume_fraction=0.50,
            matrix_material="air",
            film_thickness_um=50.0,
        )
        # Should produce a valid thermal prediction
        assert result.kappa_eff_W_mK > 0
        assert result.R_value_m2KW > 0

    def test_tio2_gives_lower_kappa_interface(self):
        """TiO₂ has higher ΔZ vs polymer → more interface scattering → lower κ_eff
        relative to its bulk κ than SiO₂."""
        sio2 = predict_from_optical(250.0, "SiO2", 0.5, "silicone", 50.0)
        tio2 = predict_from_optical(250.0, "TiO2_rutile", 0.5, "silicone", 50.0)
        # TiO₂ has higher bulk κ but more interface scattering
        # Reduction ratio should be smaller for TiO₂
        assert tio2.reduction_vs_particle < sio2.reduction_vs_particle

    def test_responsive_pnipam(self):
        """PNIPAM matrix: swollen vs collapsed should give different κ."""
        swollen = predict_from_optical(250.0, "SiO2", 0.3, "PNIPAM_swollen", 50.0)
        collapsed = predict_from_optical(250.0, "SiO2", 0.5, "PNIPAM_collapsed", 50.0)
        # Different κ values (different matrix + different φ)
        assert swollen.kappa_eff_W_mK != collapsed.kappa_eff_W_mK
