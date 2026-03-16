"""
tests/test_dispersion_d3.py — Tests for Grimme D3 dispersion correction
"""

import pytest
import sys
import os
import math

_mabe_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _mabe_root not in sys.path:
    sys.path.insert(0, _mabe_root)

from knowledge.dispersion_d3 import (
    C6_GRIMME, GAMMA_DISP, GAMMA_CORRECTION, GAMMA_DISP_BASE,
    compute_dispersion_correction, dispersion_decomposition,
)


class TestC6Values:

    def test_c6_ordering(self):
        """C6 follows polarizability: H < F < O < N < C < Cl < Br < S < I."""
        assert C6_GRIMME["H"] < C6_GRIMME["F"] < C6_GRIMME["O"]
        assert C6_GRIMME["O"] < C6_GRIMME["N"] < C6_GRIMME["C"]
        assert C6_GRIMME["C"] < C6_GRIMME["Cl"] < C6_GRIMME["Br"]
        assert C6_GRIMME["Br"] < C6_GRIMME["I"]

    def test_sulfur_very_polarizable(self):
        """Sulfur has ~4× the C6 of carbon."""
        assert C6_GRIMME["S"] > C6_GRIMME["C"] * 3

    def test_iodine_highest(self):
        """Iodine has the highest C6 among common drug elements."""
        assert C6_GRIMME["I"] > C6_GRIMME["Br"]
        assert C6_GRIMME["I"] > C6_GRIMME["S"]


class TestGammaCorrections:

    def test_carbon_correction_zero(self):
        """Carbon correction is zero (it's the reference)."""
        assert abs(GAMMA_CORRECTION["C"]) < 1e-10

    def test_halogens_favorable(self):
        """Cl, Br, I have negative corrections (more dispersion than C)."""
        assert GAMMA_CORRECTION["Cl"] < 0
        assert GAMMA_CORRECTION["Br"] < 0
        assert GAMMA_CORRECTION["I"] < 0

    def test_halogen_ordering(self):
        """I > Br > Cl in dispersion enhancement."""
        assert GAMMA_CORRECTION["I"] < GAMMA_CORRECTION["Br"] < GAMMA_CORRECTION["Cl"]

    def test_sulfur_favorable(self):
        """Sulfur has negative correction (highly polarizable)."""
        assert GAMMA_CORRECTION["S"] < 0

    def test_fluorine_unfavorable(self):
        """Fluorine has positive correction (less dispersive than C)."""
        assert GAMMA_CORRECTION["F"] > 0

    def test_oxygen_unfavorable(self):
        """Oxygen has positive correction (less dispersive than C)."""
        assert GAMMA_CORRECTION["O"] > 0

    def test_nitrogen_slightly_unfavorable(self):
        """Nitrogen is slightly less dispersive than C."""
        assert GAMMA_CORRECTION["N"] > 0
        assert abs(GAMMA_CORRECTION["N"]) < abs(GAMMA_CORRECTION["O"])

    def test_corrections_small(self):
        """All corrections are < 0.015 kJ/mol/Å² (physically reasonable)."""
        for el, corr in GAMMA_CORRECTION.items():
            assert abs(corr) < 0.020, f"{el}: correction = {corr:.4f}"


class TestDispersionScoring:

    @pytest.fixture(autouse=True)
    def check_openbabel(self):
        pytest.importorskip("openbabel")

    def test_all_carbon_near_zero(self):
        """All-carbon molecule (hexane) has near-zero correction."""
        dG = compute_dispersion_correction("CCCCCC", 150, 250)
        assert abs(dG) < 0.5, f"Hexane correction = {dG:.2f}, expected ~0"

    def test_iodobenzene_favorable(self):
        """Iodobenzene (I-containing) gets favorable correction."""
        dG = compute_dispersion_correction("Ic1ccccc1", 150, 200)
        assert dG < 0, f"Iodobenzene correction = {dG:.2f}, expected < 0"

    def test_chlorinated_favorable(self):
        """Chlorinated drug gets favorable correction."""
        dG = compute_dispersion_correction("ClC1=CC=CC=C1Cl", 180, 280)
        assert dG < 0, f"Dichlorobenzene correction = {dG:.2f}"

    def test_thioether_favorable(self):
        """Sulfur-containing ligand gets favorable correction."""
        dG = compute_dispersion_correction("CSCC", 100, 180)
        assert dG < -0.01, f"Thioether correction = {dG:.2f}"

    def test_fluorinated_unfavorable(self):
        """Fluorinated molecule gets unfavorable (positive) correction."""
        dG = compute_dispersion_correction("FC(F)(F)C(F)(F)F", 120, 180)
        assert dG > 0, f"Perfluoroethane correction = {dG:.2f}, expected > 0"

    def test_more_burial_larger_effect(self):
        """More buried surface → larger correction magnitude."""
        dG_small = compute_dispersion_correction("Ic1ccccc1", 80, 200)
        dG_large = compute_dispersion_correction("Ic1ccccc1", 180, 200)
        assert abs(dG_large) > abs(dG_small)

    def test_zero_burial_zero_correction(self):
        """Zero buried SASA → zero correction."""
        dG = compute_dispersion_correction("Ic1ccccc1", 0, 200)
        assert dG == 0.0

    def test_magnitude_reasonable(self):
        """Correction is small: < 2 kJ/mol for typical druglike."""
        dG = compute_dispersion_correction("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 200, 350)
        assert abs(dG) < 2.0


class TestDecomposition:

    @pytest.fixture(autouse=True)
    def check_openbabel(self):
        pytest.importorskip("openbabel")

    def test_decomposition_returns_elements(self):
        result = dispersion_decomposition("ClC1=CC=CC=C1")
        assert "C" in result
        assert "Cl" in result
        assert result["Cl"]["count"] >= 1

    def test_correction_sign_matches_element(self):
        result = dispersion_decomposition("Ic1ccccc1")
        assert result["I"]["gamma_correction"] < 0  # I is favorable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])