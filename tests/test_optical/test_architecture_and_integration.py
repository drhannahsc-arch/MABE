"""
test_architecture_and_integration.py — Module 12 + Sprint 5

Module 12: Architecture isomorphism (FieldInteractionSpec ↔ DiscretePocketSpec)
Sprint 5:  End-to-end integration ("blue structural color on glass + SPAAC")
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models import (
    AngularBehavior, ApplicationContext, CavityDimensions, CavityShape,
    DiscretePocketSpec, DonorPosition, FieldInteractionSpec, FieldResponse,
    FieldType, InteractionParadigm, InteractionSpec, Polarization, Solvent,
)
from optical.architecture_demo import (
    PhysicsResult, dispatch_physics, get_realization_options,
    make_blue_structural_color_spec, make_cu2_pocket_spec, run_isomorphism_demo,
)


# ═══════════════════════════════════════════════
# MODULE 12: Architecture Isomorphism
# ═══════════════════════════════════════════════

class TestFieldInteractionSpec:

    def test_inherits_from_interaction_spec(self):
        assert isinstance(FieldInteractionSpec(), InteractionSpec)

    def test_spec_type_is_field(self):
        assert FieldInteractionSpec().spec_type == InteractionParadigm.FIELD.value

    def test_discrete_pocket_spec_type_is_pocket(self):
        spec = DiscretePocketSpec(cavity_shape=CavityShape.SPHERE,
                                  cavity_dimensions=CavityDimensions(10, 2, 3, 2))
        assert spec.spec_type == InteractionParadigm.POCKET.value

    def test_both_are_interaction_spec(self):
        mol = make_cu2_pocket_spec()
        opt = make_blue_structural_color_spec()
        assert isinstance(mol, InteractionSpec)
        assert isinstance(opt, InteractionSpec)
        assert mol.spec_type != opt.spec_type

    def test_field_spec_has_design_variables(self):
        assert "particle_diameter_nm" in FieldInteractionSpec().design_variables()

    def test_field_spec_has_scoring_axes(self):
        assert "spectral_match" in FieldInteractionSpec().scoring_axes()

    def test_field_spec_default_values(self):
        spec = FieldInteractionSpec()
        assert spec.field_type == FieldType.ELECTROMAGNETIC
        assert spec.angular_behavior == AngularBehavior.NON_IRIDESCENT
        assert spec.polarization == Polarization.UNPOLARIZED

    def test_field_spec_custom_values(self):
        spec = make_blue_structural_color_spec()
        assert spec.target_wavelength_nm == 470.0
        assert spec.target_x == 0.15
        assert "SiO2" in spec.allowed_materials

    def test_angular_behavior_enum(self):
        assert AngularBehavior.IRIDESCENT.value == "iridescent"
        assert AngularBehavior.NON_IRIDESCENT.value == "non_iridescent"


class TestDispatchIsomorphism:

    def test_dispatch_molecular(self):
        result = dispatch_physics(make_cu2_pocket_spec())
        assert result.solver_name == "unified_metal_scorer"
        assert result.spec_type == "pocket"

    def test_dispatch_optical(self):
        result = dispatch_physics(make_blue_structural_color_spec())
        assert result.solver_name == "optical_forward_model"
        assert result.spec_type == "field"

    def test_dispatch_unknown_raises(self):
        class UnknownSpec(InteractionSpec):
            @property
            def spec_type(self): return "unknown_paradigm"
        with pytest.raises(NotImplementedError, match="unknown_paradigm"):
            dispatch_physics(UnknownSpec())

    def test_molecular_result_has_dG(self):
        result = dispatch_physics(make_cu2_pocket_spec())
        assert result.primary_metric_name == "ΔG_bind (kJ/mol)"
        assert result.primary_metric < 0
        assert result.feasible

    def test_optical_result_has_CIE(self):
        result = dispatch_physics(make_blue_structural_color_spec())
        assert "CIE_x" in result.secondary_metrics
        assert "CIE_y" in result.secondary_metrics
        assert "sRGB" in result.secondary_metrics

    def test_dispatch_preserves_spec_type(self):
        assert dispatch_physics(make_cu2_pocket_spec()).spec_type == "pocket"
        assert dispatch_physics(make_blue_structural_color_spec()).spec_type == "field"


class TestRealizationOptions:

    def test_molecular_options(self):
        names = [o.name for o in get_realization_options(make_cu2_pocket_spec())]
        for n in ["crown_ether", "cyclodextrin", "porphyrin", "functionalized_lignin"]:
            assert n in names

    def test_optical_options(self):
        names = [o.name for o in get_realization_options(make_blue_structural_color_spec())]
        for n in ["photonic_glass", "bragg_opal", "tmm_multilayer"]:
            assert n in names

    def test_no_cross_contamination(self):
        mol = {o.name for o in get_realization_options(make_cu2_pocket_spec())}
        opt = {o.name for o in get_realization_options(make_blue_structural_color_spec())}
        assert mol.isdisjoint(opt)


class TestIsomorphismDemo:

    def test_demo_runs(self):
        result = run_isomorphism_demo()
        assert "molecular" in result and "optical" in result

    def test_demo_spec_types(self):
        r = run_isomorphism_demo()
        assert r["molecular"]["spec_type"] == "pocket"
        assert r["optical"]["spec_type"] == "field"

    def test_demo_isomorphism_check(self):
        c = run_isomorphism_demo()["isomorphism_check"]
        assert c["same_dispatch_function"] is True
        assert c["same_base_class"] is True
        assert "no isinstance()" in c["no_domain_branching_in_dispatch"]


# ═══════════════════════════════════════════════
# SPRINT 5: End-to-End Integration
# ═══════════════════════════════════════════════

class TestEndToEndBlueCoating:
    """
    "Design a non-iridescent blue structural color coating for glass
    substrate using silica particles with SPAAC click-directed assembly."
    """

    def test_refractive_index_provides_SiO2(self):
        from optical.refractive_index import n_real
        n = n_real("SiO2", 470.0)
        assert 1.4 < n < 1.5

    def test_forward_model_produces_blue(self):
        from optical.photonic_glass import photonic_glass_reflectance
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_sRGB

        wl = np.linspace(380, 780, 81)
        R = photonic_glass_reflectance(200.0, "SiO2", 1.0, 0.50, wl)
        X, Y, Z = spectrum_to_XYZ(R, wl)
        x, y, _ = XYZ_to_xyY(X, Y, Z)
        r, g, b = XYZ_to_sRGB(X, Y, Z)
        assert b > r, f"Expected blue dominant, got sRGB=({r:.2f},{g:.2f},{b:.2f})"
        assert b > g

    def test_click_linker_modifies_effective_diameter(self):
        from optical.click_linker import compute_chain, effective_diameter
        chain = compute_chain("SiO2", anchor="APTES", spacer="PEG4", click="SPAAC")
        assert chain.L_total_per_side_nm > 1.0  # SPAAC adds several nm
        d_eff = effective_diameter(200.0, chain)
        assert d_eff > 200.0

    def test_core_shell_mie_with_click_shell(self):
        from optical.core_shell_mie import mie_coated_efficiencies
        result = mie_coated_efficiencies(200.0, 210.0, 1.46, 1.50, 1.0, 470.0)
        assert result["Q_ext"] > 0
        assert result["Q_ext"] < 10

    def test_underlayer_coupling(self):
        from optical.underlayer_coupling import coupled_reflectance
        R_coupled = coupled_reflectance(R_film=0.3, R_under=0.1)
        assert 0 <= R_coupled <= 1.0

    def test_cie_color_conversion(self):
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_sRGB
        wl = np.linspace(380, 780, 81)
        R = np.ones_like(wl) * 0.5
        X, Y, Z = spectrum_to_XYZ(R, wl)
        x, y, _ = XYZ_to_xyY(X, Y, Z)
        assert abs(x - 0.3127) < 0.01  # D65 white point
        assert abs(y - 0.3290) < 0.01

    def test_inverse_design_finds_blue(self):
        from optical.inverse_design import inverse_design_photonic_glass
        result = inverse_design_photonic_glass(0.15, 0.10, sphere_material="SiO2")
        assert result.design is not None
        d = result.design.diameter_nm
        assert 140 < d < 260, f"Expected ~170-220 nm, got {d:.0f} nm"

    def test_bragg_opal_prediction(self):
        from optical.bragg_opal import bragg_opal
        peak = bragg_opal(220.0, material="SiO2")
        # 1.633 * 220 * n_eff ≈ 480-500 nm
        assert 400 < peak < 600, f"Bragg peak {peak:.0f} nm out of range"

    def test_tmm_energy_conservation(self):
        from optical.tmm import tmm_reflectance
        stack = [("air", 0), ("TiO2_rutile", 68.75), ("SiO2", 91.67),
                 ("TiO2_rutile", 68.75), ("SiO2", 91.67), ("SiO2", 0)]
        R, T = tmm_reflectance(stack, 550.0)
        assert abs(R + T - 1.0) < 1e-6, f"R+T={R+T}"

    def test_full_pipeline_blue_on_glass(self):
        """End-to-end: blue structural color on glass with SiO₂ + SPAAC."""
        from optical.photonic_glass import photonic_glass_reflectance
        from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_sRGB
        from optical.click_linker import compute_chain, effective_diameter

        # Click linker geometry
        chain = compute_chain("SiO2", anchor="APTES", spacer="PEG4", click="SPAAC")
        d_core = 190.0
        d_total = effective_diameter(d_core, chain)

        # Forward model
        wl = np.linspace(380, 780, 81)
        R = photonic_glass_reflectance(d_total, "SiO2", 1.0, 0.50, wl)

        # CIE color
        X, Y, Z = spectrum_to_XYZ(R, wl)
        x, y, _ = XYZ_to_xyY(X, Y, Z)
        r, g, b = XYZ_to_sRGB(X, Y, Z)

        # Should be blue
        assert b > r, f"Expected blue: sRGB=({r:.2f},{g:.2f},{b:.2f})"
        assert x < 0.30, f"CIE x={x:.3f}, expected <0.30 for blue"

        # Design is physically realizable
        assert 150 < d_core < 300
        assert d_total > d_core

    def test_red_target_fails_gracefully(self):
        from optical.inverse_design import inverse_design_photonic_glass
        result = inverse_design_photonic_glass(0.50, 0.30, sphere_material="SiO2")
        # Standard photonic glass can't make saturated red
        if result.converged:
            assert result.delta_E > 3.0 or result.design.diameter_nm > 280

    def test_field_spec_drives_full_pipeline(self):
        """FieldInteractionSpec dispatches through optical pipeline end-to-end."""
        spec = make_blue_structural_color_spec()
        result = dispatch_physics(spec)
        assert result.feasible
        assert result.spec_type == "field"
        assert result.solver_name == "optical_forward_model"
        # Verify optical output is physically reasonable
        m = result.secondary_metrics
        assert 0 < m["CIE_x"] < 1
        assert 0 < m["CIE_y"] < 1
        assert m["particle_diameter_nm"] > 0
        assert m["peak_reflectance"] > 0
