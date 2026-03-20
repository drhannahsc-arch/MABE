"""
tests/test_click_interface.py -- Tests for click interface electrical model.

Validates:
  - Bond database completeness (5 entries, all fields)
  - Interface resistance prediction (area scaling, physical ordering)
  - Stack loss prediction (I^2*R, multi-interface sum)
  - Default harvesting stack
  - Physical sanity (covalent < vdW, conjugated < saturated)
  - Edge cases
"""

import sys
import os
import math
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.click_interface import (
    ClickBond,
    ClickInterfaceSpec,
    InterfaceResult,
    StackLossResult,
    get_click_bond,
    list_click_bonds,
    predict_interface_resistance,
    predict_stack_loss,
    default_harvesting_stack,
    _CLICK_BONDS,
)


EXPECTED_BONDS = ["SPAAC", "CuAAC", "thiol-maleimide", "Diels-Alder", "van_der_Waals"]


# -----------------------------------------------------------------------
# Database completeness
# -----------------------------------------------------------------------

class TestBondDatabase:

    def test_all_present(self):
        available = list_click_bonds()
        for name in EXPECTED_BONDS:
            assert name in available

    def test_count(self):
        assert len(_CLICK_BONDS) == 5

    @pytest.mark.parametrize("name", EXPECTED_BONDS)
    def test_fields_populated(self, name):
        bond = get_click_bond(name)
        assert bond.R_contact_ohm_cm2 > 0
        assert bond.R_kapitza_m2K_W > 0
        assert bond.source != ""
        assert bond.bond_type == name

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown bond type"):
            get_click_bond("hydrogen_bond")

    def test_spec_values_match_table(self):
        """Verify R_contact values match the build handoff spec."""
        assert get_click_bond("SPAAC").R_contact_ohm_cm2 == 1e-4
        assert get_click_bond("CuAAC").R_contact_ohm_cm2 == 1e-5
        assert get_click_bond("thiol-maleimide").R_contact_ohm_cm2 == 1e-5
        assert get_click_bond("Diels-Alder").R_contact_ohm_cm2 == 1e-3
        assert get_click_bond("van_der_Waals").R_contact_ohm_cm2 == 1e-2

    def test_kapitza_values_match_table(self):
        assert get_click_bond("SPAAC").R_kapitza_m2K_W == 5e-8
        assert get_click_bond("CuAAC").R_kapitza_m2K_W == 5e-8
        assert get_click_bond("thiol-maleimide").R_kapitza_m2K_W == 3e-8
        assert get_click_bond("Diels-Alder").R_kapitza_m2K_W == 8e-8
        assert get_click_bond("van_der_Waals").R_kapitza_m2K_W == 1e-6


# -----------------------------------------------------------------------
# Interface resistance prediction
# -----------------------------------------------------------------------

class TestInterfaceResistance:

    def test_basic_prediction(self):
        r = predict_interface_resistance("CuAAC", area_m2=1.0)
        assert r > 0

    def test_larger_area_lower_resistance(self):
        """R = R_contact / area, so doubling area halves R."""
        r1 = predict_interface_resistance("CuAAC", area_m2=1.0)
        r2 = predict_interface_resistance("CuAAC", area_m2=2.0)
        assert abs(r2 - r1 / 2.0) < 1e-15

    def test_small_area_high_resistance(self):
        r = predict_interface_resistance("CuAAC", area_m2=1e-4)  # 1 cm2
        # R = 1e-5 ohm*cm2 / 1 cm2 = 1e-5 ohm
        assert abs(r - 1e-5) < 1e-12

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match="area_m2 must be > 0"):
            predict_interface_resistance("CuAAC", area_m2=0.0)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="area_m2 must be > 0"):
            predict_interface_resistance("CuAAC", area_m2=-1.0)


# -----------------------------------------------------------------------
# Stack loss prediction
# -----------------------------------------------------------------------

class TestStackLoss:

    def test_single_interface(self):
        result = predict_stack_loss(["CuAAC"], current_A=1.0, area_m2=1.0)
        assert isinstance(result, StackLossResult)
        assert result.total_power_loss_W > 0
        assert len(result.per_interface) == 1

    def test_three_interfaces(self):
        stack = default_harvesting_stack()
        result = predict_stack_loss(stack, current_A=1.0, area_m2=1.0)
        assert len(result.per_interface) == 3

    def test_total_r_is_sum(self):
        stack = ["SPAAC", "CuAAC", "thiol-maleimide"]
        result = predict_stack_loss(stack, current_A=1.0, area_m2=1.0)
        sum_r = sum(iface.electrical_R_ohm for iface in result.per_interface)
        assert abs(result.total_electrical_R_ohm - sum_r) < 1e-15

    def test_power_is_i_squared_r(self):
        stack = ["CuAAC"]
        result = predict_stack_loss(stack, current_A=2.0, area_m2=1.0)
        expected = 4.0 * result.total_electrical_R_ohm
        assert abs(result.total_power_loss_W - expected) < 1e-15

    def test_zero_current_zero_loss(self):
        result = predict_stack_loss(["CuAAC"], current_A=0.0, area_m2=1.0)
        assert result.total_power_loss_W == 0.0

    def test_loss_scales_with_current_squared(self):
        r1 = predict_stack_loss(["CuAAC"], current_A=1.0, area_m2=1.0)
        r3 = predict_stack_loss(["CuAAC"], current_A=3.0, area_m2=1.0)
        ratio = r3.total_power_loss_W / r1.total_power_loss_W
        assert abs(ratio - 9.0) < 1e-10

    def test_negative_current_raises(self):
        with pytest.raises(ValueError, match="current_A must be >= 0"):
            predict_stack_loss(["CuAAC"], current_A=-1.0)

    def test_kapitza_populated(self):
        result = predict_stack_loss(["SPAAC"], current_A=1.0, area_m2=1.0)
        assert result.per_interface[0].thermal_R_kapitza_m2K_W == 5e-8


# -----------------------------------------------------------------------
# Default stack
# -----------------------------------------------------------------------

class TestDefaultStack:

    def test_three_interfaces(self):
        stack = default_harvesting_stack()
        assert len(stack) == 3

    def test_correct_order(self):
        stack = default_harvesting_stack()
        assert stack == ["SPAAC", "CuAAC", "thiol-maleimide"]

    def test_all_valid(self):
        for bt in default_harvesting_stack():
            get_click_bond(bt)  # should not raise


# -----------------------------------------------------------------------
# Physical sanity
# -----------------------------------------------------------------------

class TestPhysicalSanity:

    def test_covalent_lower_R_than_vdw(self):
        """All covalent bonds should have lower contact R than van der Waals."""
        vdw = get_click_bond("van_der_Waals").R_contact_ohm_cm2
        for name in ["SPAAC", "CuAAC", "thiol-maleimide", "Diels-Alder"]:
            bond = get_click_bond(name)
            assert bond.R_contact_ohm_cm2 < vdw, f"{name} should be < vdW"

    def test_covalent_lower_kapitza_than_vdw(self):
        """Covalent bonds should have lower thermal R than van der Waals."""
        vdw = get_click_bond("van_der_Waals").R_kapitza_m2K_W
        for name in ["SPAAC", "CuAAC", "thiol-maleimide", "Diels-Alder"]:
            bond = get_click_bond(name)
            assert bond.R_kapitza_m2K_W < vdw, f"{name} should be < vdW"

    def test_conjugated_lower_R_than_saturated(self):
        """Conjugated bonds (triazoles) should conduct better than Diels-Alder."""
        da = get_click_bond("Diels-Alder").R_contact_ohm_cm2
        for name in ["SPAAC", "CuAAC"]:
            bond = get_click_bond(name)
            assert bond.R_contact_ohm_cm2 < da, f"{name} should be < Diels-Alder"

    def test_stack_loss_negligible_for_building(self):
        """Interface losses in a 1 m2 panel at typical current should be < 1 mW."""
        # Typical PV current: ~7 W/m2 at ~0.5V -> ~14 A/m2... but that's current density
        # At panel level with conductor collecting, total current ~ 14 A for 1m2
        # But interface loss is across the full area, so R is very small
        result = predict_stack_loss(
            default_harvesting_stack(), current_A=14.0, area_m2=1.0,
        )
        assert result.total_power_loss_W < 0.001  # < 1 mW
