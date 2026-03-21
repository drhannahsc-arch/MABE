"""
tests/test_photonic_assembly.py -- Tests for photonic nanoparticle assembly.
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.photonic_assembly import (
    bragg_color_ordered, glass_color_disordered,
    diameter_for_ordered_color, diameter_for_glass_color,
    angle_shift_ordered, sedimentation_velocity, sedimentation_time_hours,
    design_photonic_particles, photonic_structural_color_layer,
    nanoparticle_monomer, AssemblyConditions,
    PARTICLE_MATERIALS, PhotonicDesign,
)
from core.assembly_engine import design_material, Topology


class TestBraggColor:

    def test_blue_opal(self):
        wl, color = bragg_color_ordered(212, 1.46)
        assert 440 < wl < 500
        assert color == "blue"

    def test_red_opal(self):
        wl, color = bragg_color_ordered(294, 1.46)
        assert 620 < wl < 680
        assert color == "red"

    def test_higher_n_smaller_particle(self):
        d_sio2 = diameter_for_ordered_color(530, 1.46)
        d_tio2 = diameter_for_ordered_color(530, 2.49)
        assert d_tio2 < d_sio2

    def test_roundtrip(self):
        for target in [420, 470, 530, 580, 650]:
            d = diameter_for_ordered_color(target, 1.46)
            wl, _ = bragg_color_ordered(d, 1.46)
            assert abs(wl - target) < 5


class TestGlassColor:

    def test_blue_glass(self):
        wl, color = glass_color_disordered(198, 1.46)
        assert 440 < wl < 500

    def test_angle_independent(self):
        d = design_photonic_particles("blue", ordered=False)
        assert not d.angle_dependent

    def test_roundtrip(self):
        for target in [470, 530, 650]:
            d = diameter_for_glass_color(target, 1.46)
            wl, _ = glass_color_disordered(d, 1.46)
            assert abs(wl - target) < 5


class TestAngleShift:

    def test_normal_no_shift(self):
        shifted = angle_shift_ordered(470, 0, 1.36)
        assert abs(shifted - 470) < 1

    def test_oblique_blue_shifts(self):
        shifted = angle_shift_ordered(470, 45, 1.36)
        assert shifted < 470

    def test_larger_angle_more_shift(self):
        s30 = angle_shift_ordered(470, 30, 1.36)
        s60 = angle_shift_ordered(470, 60, 1.36)
        assert s60 < s30


class TestSedimentation:

    def test_positive_velocity(self):
        v = sedimentation_velocity(200, 2.2)
        assert v > 0

    def test_heavier_faster(self):
        v_sio2 = sedimentation_velocity(200, 2.2)
        v_tio2 = sedimentation_velocity(200, 4.23)
        assert v_tio2 > v_sio2

    def test_larger_faster(self):
        v_small = sedimentation_velocity(100, 2.2)
        v_large = sedimentation_velocity(300, 2.2)
        assert v_large > v_small

    def test_time_positive(self):
        t = sedimentation_time_hours(200, 1.0, 2.2)
        assert t > 0


class TestAssemblyConditions:

    def test_monodisperse_ordered(self):
        c = AssemblyConditions(dispersity_pct=3.0)
        assert c.ordered

    def test_polydisperse_disordered(self):
        c = AssemblyConditions(dispersity_pct=15.0)
        assert not c.ordered

    def test_spray_disordered(self):
        c = AssemblyConditions(dispersity_pct=3.0, assembly_method="spray")
        assert not c.ordered

    def test_absorber_disrupts(self):
        c = AssemblyConditions(dispersity_pct=3.0, absorber_loading=0.2)
        assert not c.ordered


class TestDesignPhotonicParticles:

    def test_blue_ordered(self):
        d = design_photonic_particles("blue", "SiO2", ordered=True)
        assert d.color == "blue"
        assert d.ordered
        assert d.angle_dependent
        assert d.diameter_nm > 100

    def test_green_glass(self):
        d = design_photonic_particles("green", "SiO2", ordered=False)
        assert d.color == "green"
        assert not d.ordered
        assert not d.angle_dependent

    def test_has_monomer(self):
        d = design_photonic_particles("blue")
        assert d.monomer is not None
        assert d.monomer.valence > 0

    def test_has_material_design(self):
        d = design_photonic_particles("blue")
        assert d.material_design is not None
        assert d.material_design.topology is not None

    def test_angle_sweep_ordered(self):
        d = design_photonic_particles("blue", ordered=True)
        assert len(d.angle_sweep) >= 4
        # Normal incidence should be blue
        assert d.angle_sweep[0][1] == "blue"

    def test_no_angle_sweep_glass(self):
        d = design_photonic_particles("blue", ordered=False)
        assert len(d.angle_sweep) == 0

    def test_tio2_smaller_particles(self):
        d_sio2 = design_photonic_particles("blue", "SiO2")
        d_tio2 = design_photonic_particles("blue", "TiO2")
        assert d_tio2.diameter_nm < d_sio2.diameter_nm

    def test_summary_string(self):
        d = design_photonic_particles("blue")
        assert "Photonic" in d.summary
        assert "blue" in d.summary.lower()

    def test_all_colors_valid(self):
        for color in ["violet", "blue", "green", "yellow", "orange", "red"]:
            d = design_photonic_particles(color)
            assert d.color == color
            assert d.diameter_nm > 50


class TestMonomerSpec:

    def test_fcc_symmetry(self):
        m = nanoparticle_monomer("SiO2", 200, AssemblyConditions(dispersity_pct=3))
        assert m.symmetry == "Oh"

    def test_disordered_c1(self):
        m = nanoparticle_monomer("SiO2", 200, AssemblyConditions(dispersity_pct=15))
        assert m.symmetry == "C1"

    def test_12_contacts_ordered(self):
        m = nanoparticle_monomer("SiO2", 200, AssemblyConditions(dispersity_pct=3))
        assert m.valence == 12

    def test_6_contacts_disordered(self):
        m = nanoparticle_monomer("SiO2", 200, AssemblyConditions(dispersity_pct=15))
        assert m.valence == 6

    def test_feeds_into_assembly_engine(self):
        m = nanoparticle_monomer("SiO2", 215)
        d = design_material(m)
        assert d.topology is not None
        assert d.properties is not None


class TestCoatingIntegration:

    def test_structural_color_params(self):
        p = photonic_structural_color_layer("blue", "SiO2", ordered=True)
        assert p["color"] == "blue"
        assert p["particle_diameter_nm"] > 100
        assert p["layer_thickness_um"] > 0

    def test_all_colors_produce_params(self):
        for color in ["blue", "green", "red"]:
            p = photonic_structural_color_layer(color)
            assert p["peak_wavelength_nm"] > 400
            assert p["photonic_design"] is not None


class TestMaterialDB:

    def test_all_materials_valid(self):
        for name, mat in PARTICLE_MATERIALS.items():
            assert mat.n_refractive > 1.0
            assert mat.density_g_cm3 > 0
            assert mat.size_range_nm[0] < mat.size_range_nm[1]

    def test_tio2_highest_n(self):
        assert PARTICLE_MATERIALS["TiO2"].n_refractive > PARTICLE_MATERIALS["SiO2"].n_refractive
