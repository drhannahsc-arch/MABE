"""
tests/test_pfas_free_coating.py -- Tests for PFAS-free multi-layer coating design.
"""

import sys
import os
import math
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.pfas_free_coating import (
    young_contact_angle, cassie_baxter_angle, wenzel_angle,
    laplace_entry_pressure, re_entrant_oil_angle,
    bragg_wavelength, particle_size_for_color, color_from_particle_size,
    wvtr_estimate,
    design_primer, design_omniphobic, design_structural_color,
    design_breathability, design_durability,
    design_coating, CoatingTargets, CoatingStack,
    SUBSTRATES, SubstrateSpec, SubstrateType,
    GAMMA_WATER, GAMMA_HEXADECANE,
)


# -----------------------------------------------------------------------
# Contact angle physics
# -----------------------------------------------------------------------

class TestContactAngle:

    def test_young_hydrophobic(self):
        # PDMS gamma=20: should be > 90 for water
        theta = young_contact_angle(20.0, GAMMA_WATER)
        assert theta > 90

    def test_young_hydrophilic(self):
        # Glass gamma=250: should be < 90 for water
        theta = young_contact_angle(250.0, GAMMA_WATER)
        assert theta < 90

    def test_young_oil_wets_silicone(self):
        # Oil (27.5) on silicone (20): theta_Y < 90 (wetting)
        theta = young_contact_angle(20.0, GAMMA_HEXADECANE)
        assert theta < 90

    def test_cassie_increases_angle(self):
        theta_y = 110
        theta_cb = cassie_baxter_angle(theta_y, 0.1)
        assert theta_cb > theta_y

    def test_cassie_f1_equals_young(self):
        theta_y = 110
        theta_cb = cassie_baxter_angle(theta_y, 1.0)
        assert abs(theta_cb - theta_y) < 0.1

    def test_wenzel_amplifies(self):
        # Hydrophobic surface: Wenzel makes more hydrophobic
        theta_y = 110
        theta_w = wenzel_angle(theta_y, 2.0)
        assert theta_w > theta_y

    def test_wenzel_hydrophilic_gets_more(self):
        # Hydrophilic surface: Wenzel makes more hydrophilic
        theta_y = 70
        theta_w = wenzel_angle(theta_y, 2.0)
        assert theta_w < theta_y


# -----------------------------------------------------------------------
# Re-entrant oil repulsion
# -----------------------------------------------------------------------

class TestReEntrant:

    def test_oil_repulsion_with_reentrant(self):
        # Silicone (gamma=20): theta_Y_oil = 58 (wets)
        # Re-entrant psi=75 with f=0.1 should give > 130
        theta = re_entrant_oil_angle(20.0, 75.0, 0.1)
        assert theta > 120

    def test_shallow_reentrant_fails(self):
        # psi=10 (nearly flat) should not achieve Cassie
        theta = re_entrant_oil_angle(20.0, 10.0, 0.1)
        assert theta < 90  # falls to Young's angle

    def test_already_nonwetting(self):
        # If theta_Y > 90, re-entrant just adds stability
        theta = re_entrant_oil_angle(10.0, 75.0, 0.1)
        assert theta > 140

    def test_no_fluorine_needed(self):
        # Demonstrate: silicone (gamma=20) + re-entrant achieves oil CA > 130
        theta = re_entrant_oil_angle(20.0, 75.0, 0.1)
        assert theta > 130, f"Oil CA = {theta}, need > 130 without fluorine"


# -----------------------------------------------------------------------
# Laplace pressure
# -----------------------------------------------------------------------

class TestLaplacePressure:

    def test_hydrophobic_positive_lep(self):
        # CA > 90: positive LEP (liquid repelled)
        lep = laplace_entry_pressure(GAMMA_WATER / 1000, 150.0, 1e-6)
        assert lep > 0

    def test_hydrophilic_zero_lep(self):
        # CA < 90: no barrier
        lep = laplace_entry_pressure(GAMMA_WATER / 1000, 60.0, 1e-6)
        assert lep == 0.0

    def test_smaller_pore_higher_lep(self):
        lep_small = laplace_entry_pressure(GAMMA_WATER / 1000, 150.0, 0.1e-6)
        lep_large = laplace_entry_pressure(GAMMA_WATER / 1000, 150.0, 1.0e-6)
        assert lep_small > lep_large


# -----------------------------------------------------------------------
# Structural color
# -----------------------------------------------------------------------

class TestStructuralColor:

    def test_bragg_basic(self):
        # lambda = 2*n*d
        wl = bragg_wavelength(1.3, 200.0)
        assert abs(wl - 520.0) < 1.0

    def test_particle_size_blue(self):
        d = particle_size_for_color(470)
        assert 180 < d < 260

    def test_particle_size_red(self):
        d = particle_size_for_color(650)
        assert 250 < d < 350

    def test_roundtrip_color(self):
        for target_wl in [420, 470, 530, 580, 650]:
            d = particle_size_for_color(target_wl)
            actual_wl, color = color_from_particle_size(d)
            assert abs(actual_wl - target_wl) < 5

    def test_color_names(self):
        _, c_blue = color_from_particle_size(particle_size_for_color(470))
        _, c_red = color_from_particle_size(particle_size_for_color(650))
        assert c_blue == "blue"
        assert c_red == "red"


# -----------------------------------------------------------------------
# Breathability
# -----------------------------------------------------------------------

class TestBreathability:

    def test_wvtr_positive(self):
        wvtr = wvtr_estimate(0.5e-6, 0.5, 10e-6)
        assert wvtr > 0

    def test_thinner_membrane_higher_wvtr(self):
        wvtr_thin = wvtr_estimate(0.5e-6, 0.5, 5e-6)
        wvtr_thick = wvtr_estimate(0.5e-6, 0.5, 50e-6)
        assert wvtr_thin > wvtr_thick

    def test_higher_porosity_higher_wvtr(self):
        wvtr_low = wvtr_estimate(0.5e-6, 0.2, 10e-6)
        wvtr_high = wvtr_estimate(0.5e-6, 0.8, 10e-6)
        assert wvtr_high > wvtr_low


# -----------------------------------------------------------------------
# Layer design
# -----------------------------------------------------------------------

class TestLayerDesign:

    def test_primer_for_glass(self):
        p = design_primer(SUBSTRATES["glass"])
        assert "silane" in p.chemistry

    def test_primer_for_cotton(self):
        p = design_primer(SUBSTRATES["cotton"])
        assert p.gamma_surface > 30

    def test_omniphobic_no_fluorine(self):
        o = design_omniphobic(150, 130)
        assert "PTFE" not in o.chemistry
        assert "fluoro" not in o.chemistry.lower()
        assert o.water_contact_angle > 140

    def test_omniphobic_has_reentrant(self):
        o = design_omniphobic(150, 130)
        assert o.re_entrant_angle_deg > 60

    def test_structural_color_blue(self):
        sc = design_structural_color("blue")
        assert sc.color == "blue"
        assert 400 < sc.peak_wavelength_nm < 500
        assert sc.particle_diameter_nm > 0

    def test_breathable_membrane(self):
        b = design_breathability(10000, 50)
        assert b.wvtr > 0
        assert b.lep_water_kPa > 0
        assert b.pore_radius_um > 0


# -----------------------------------------------------------------------
# Full coating design
# -----------------------------------------------------------------------

class TestCoatingDesign:

    def test_textile_jacket(self):
        d = design_coating(
            SUBSTRATES["polyester"],
            CoatingTargets(water_contact_angle=150, oil_contact_angle=130,
                          color="blue", wvtr=10000, durable=True))
        assert d.meets_targets
        assert d.stack.color == "blue"
        assert d.stack.water_contact_angle > 140
        assert d.stack.oil_contact_angle > 115

    def test_glass_cookware(self):
        d = design_coating(
            SUBSTRATES["glass"],
            CoatingTargets(water_contact_angle=110, oil_contact_angle=110,
                          color="green", wvtr=0, durable=True))
        assert d.meets_targets
        assert d.stack.color == "green"

    def test_steel_surface(self):
        d = design_coating(
            SUBSTRATES["stainless_steel"],
            CoatingTargets(water_contact_angle=160, oil_contact_angle=140,
                          color="", wvtr=0, durable=True))
        assert d.meets_targets
        assert d.stack.water_contact_angle > 150

    def test_cotton_tent(self):
        d = design_coating(
            SUBSTRATES["cotton"],
            CoatingTargets(water_contact_angle=150, oil_contact_angle=120,
                          color="", wvtr=5000, durable=True))
        assert d.meets_targets
        assert d.stack.wvtr > 4000

    def test_film_no_durability(self):
        d = design_coating(
            SUBSTRATES["PP_film"],
            CoatingTargets(water_contact_angle=130, color="red", durable=False))
        assert d.meets_targets
        # No durability layer
        assert not any(l.layer_type.value == "durability" for l in d.stack.layers)

    def test_layer_count(self):
        d = design_coating(
            SUBSTRATES["polyester"],
            CoatingTargets(water_contact_angle=150, oil_contact_angle=130,
                          color="blue", wvtr=10000, durable=True))
        # Should have: primer + color + breathability + omniphobic + durability = 5
        assert len(d.stack.layers) == 5

    def test_no_pfas_anywhere(self):
        for sub_name, sub in SUBSTRATES.items():
            d = design_coating(sub,
                CoatingTargets(water_contact_angle=150, oil_contact_angle=130))
            for layer in d.stack.layers:
                assert "PTFE" not in layer.chemistry
                assert "PFAS" not in layer.chemistry.lower()
                assert "fluoro" not in layer.chemistry.lower() or "fluorine_free" in layer.chemistry

    def test_summary_string(self):
        d = design_coating(SUBSTRATES["polyester"],
            CoatingTargets(water_contact_angle=150, color="blue"))
        assert "PFAS-Free" in d.summary
        assert "polyester" in d.summary
