"""
tests/test_optical/test_surface_optics.py — Surface Optics Tests

Validates structural dye deployment on real substrates.
"""

import sys
import os
import pytest
import numpy as np
import math

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

pytest.importorskip("miepython")

from optical.surface_optics import (
    surface_reflectance, structural_dye_on_surface,
    compare_substrates, structure_factor_2D_hex,
    DeploymentSpec, SUBSTRATES, _substrate_R,
)


class TestSubstrateDatabase:

    def test_all_substrates_exist(self):
        expected = ["glass", "steel", "aluminum", "concrete",
                    "textile_black", "textile_white", "wood",
                    "plastic_clear", "plastic_white", "plastic_black"]
        for name in expected:
            assert name in SUBSTRATES, f"Missing substrate: {name}"

    def test_substrate_R_bounded(self):
        lam = np.linspace(380, 780, 81)
        for name in SUBSTRATES:
            R = _substrate_R(name, lam)
            assert np.all(R >= 0), f"{name} has negative R"
            assert np.all(R <= 1), f"{name} has R > 1"

    def test_black_textile_low_R(self):
        lam = np.linspace(380, 780, 81)
        R = _substrate_R("textile_black", lam)
        assert R.mean() < 0.10, "Black textile should have low R"

    def test_white_textile_high_R(self):
        lam = np.linspace(380, 780, 81)
        R = _substrate_R("textile_white", lam)
        assert R.mean() > 0.60, "White textile should have high R"

    def test_glass_low_R(self):
        """Glass has low Fresnel R (~4%) at near-index-matched interface."""
        lam = np.linspace(380, 780, 81)
        R = _substrate_R("glass", lam, film_n=1.35)
        assert R.mean() < 0.10


class TestStructureFactor2D:

    def test_has_peak(self):
        """2D hex S(q) should have a peak near G1 = 4π/(√3 × a)."""
        a = 225  # nm
        G1 = 4 * math.pi / (math.sqrt(3) * a)
        q_arr = np.linspace(0.01, 0.15, 500)
        S_arr = [structure_factor_2D_hex(q, a, domain_size=15) for q in q_arr]
        peak_q = q_arr[np.argmax(S_arr)]
        assert abs(peak_q - G1) / G1 < 0.15, \
            f"2D S(q) peak at {peak_q:.4f}, expected near G1={G1:.4f}"

    def test_baseline_near_unity(self):
        """Away from peaks, S₂D ≈ 1 (uncorrelated)."""
        a = 225
        # q far from any peak
        S_low = structure_factor_2D_hex(0.001, a, 15)
        assert abs(S_low - 1.0) < 0.5


class TestDeploymentRegimes:

    def test_sparse_runs(self):
        dep = DeploymentSpec("textile_black", "sparse", coverage=0.20)
        r = surface_reflectance(225, complex(1.46, 0), dep)
        assert "R_total" in r
        assert r["R_total"].max() > 0

    def test_monolayer_runs(self):
        dep = DeploymentSpec("textile_black", "monolayer", coverage=0.80)
        r = surface_reflectance(225, complex(1.46, 0), dep)
        assert r["R_total"].max() > 0

    def test_few_layer_runs(self):
        dep = DeploymentSpec("glass", "few_layer", n_layers=3)
        r = surface_reflectance(225, complex(1.46, 0), dep)
        assert r["R_total"].max() > 0

    def test_thick_film_runs(self):
        dep = DeploymentSpec("concrete", "thick_film", n_layers=15)
        r = surface_reflectance(225, complex(1.46, 0), dep)
        assert r["R_total"].max() > 0

    def test_thick_film_more_saturated_than_sparse(self):
        """More layers → more structural color → larger a* deviation from neutral."""
        dep_sparse = DeploymentSpec("textile_black", "sparse", coverage=0.30)
        dep_thick = DeploymentSpec("textile_black", "thick_film", n_layers=10)

        r_sparse = surface_reflectance(225, complex(1.46, 0), dep_sparse)
        r_thick = surface_reflectance(225, complex(1.46, 0), dep_thick)

        # Chroma = sqrt(a*² + b*²)
        L_s, a_s, b_s = r_sparse["Lab"]
        L_t, a_t, b_t = r_thick["Lab"]
        chroma_sparse = math.sqrt(a_s**2 + b_s**2)
        chroma_thick = math.sqrt(a_t**2 + b_t**2)
        assert chroma_thick > chroma_sparse, \
            f"Thick film chroma={chroma_thick:.1f} should exceed sparse={chroma_sparse:.1f}"


class TestSubstrateEffects:

    def test_black_more_saturated_than_white(self):
        """Black substrate → more saturated structural color (Iwata 2017)."""
        dep_black = DeploymentSpec("textile_black", "few_layer", n_layers=3)
        dep_white = DeploymentSpec("textile_white", "few_layer", n_layers=3)

        r_black = surface_reflectance(225, complex(1.46, 0), dep_black)
        r_white = surface_reflectance(225, complex(1.46, 0), dep_white)

        L_b, a_b, b_b = r_black["Lab"]
        L_w, a_w, b_w = r_white["Lab"]
        chroma_black = math.sqrt(a_b**2 + b_b**2)
        chroma_white = math.sqrt(a_w**2 + b_w**2)
        assert chroma_black > chroma_white, \
            "Black substrate should give higher chroma"

    def test_glass_has_transmission(self):
        """Glass substrate should produce transmission spectrum."""
        dep = DeploymentSpec("glass", "few_layer", n_layers=3)
        r = surface_reflectance(225, complex(1.46, 0), dep)
        assert "T_total" in r, "Glass should have transmission"
        assert "transmission_sRGB" in r

    def test_opaque_no_transmission(self):
        """Opaque substrates should not have transmission."""
        dep = DeploymentSpec("concrete", "few_layer", n_layers=3)
        r = surface_reflectance(225, complex(1.46, 0), dep)
        assert "T_total" not in r


class TestMultiShellOnSurface:

    def test_structural_dye_on_surface_runs(self):
        from optical.multi_shell import ShellLayer
        shells = [ShellLayer("CuPc", 2.0, click="SPAAC")]
        dep = DeploymentSpec("textile_black", "few_layer", n_layers=3)
        r = structural_dye_on_surface("SiO2", 225, shells, dep)
        assert r["peak_nm"] > 380

    def test_total_diameter_propagated(self):
        from optical.multi_shell import ShellLayer
        shells = [ShellLayer("CuPc", 3.0, click="SPAAC")]
        dep = DeploymentSpec("glass", "monolayer")
        r = structural_dye_on_surface("SiO2", 200, shells, dep)
        assert abs(r["total_diameter_nm"] - 206.0) < 0.1

    def test_compare_substrates_runs(self):
        results = compare_substrates(substrates=["textile_black", "glass"])
        assert len(results) == 2
        assert all("R_total" in r for r in results)
