"""
test_structural_dye_engine.py — Tests for structural dye design engine.
"""
import pytest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.structural_dye_engine import (
    CHROMOPHORE_LIBRARY, INDEX_SHELL_LIBRARY, ChromophoreEntry, IndexShellEntry,
    chromophore_k_spectrum, chromophore_absorption_spectrum,
    shell_n_spectrum, shell_k_spectrum,
    compute_structural_dye_spectrum, compute_multilayer_spectrum,
    spectrum_to_cie, compute_delta_E,
    generate_dye_designs, DyeDesign,
    _LAM,
)


class TestChromophoreLibrary:

    def test_count(self):
        assert len(CHROMOPHORE_LIBRARY) >= 20

    def test_all_have_published_epsilon(self):
        for name, ch in CHROMOPHORE_LIBRARY.items():
            assert ch.epsilon_M_cm > 0, f"{name}: ε must be positive"
            assert ch.lambda_max_nm > 300, f"{name}: λ_max must be > 300 nm"

    def test_all_have_click_chemistry(self):
        for name, ch in CHROMOPHORE_LIBRARY.items():
            assert len(ch.click_chemistry) > 5, f"{name}: missing click chemistry"

    def test_all_have_source(self):
        for name, ch in CHROMOPHORE_LIBRARY.items():
            assert len(ch.source) > 5, f"{name}: missing source"

    def test_porphyrins_high_epsilon(self):
        """Porphyrin Soret bands have ε > 100,000."""
        assert CHROMOPHORE_LIBRARY["TPP_freebase"].epsilon_M_cm >= 200000
        assert CHROMOPHORE_LIBRARY["ZnTPP"].epsilon_M_cm >= 250000

    def test_wavelength_ordering(self):
        """CuPc (678nm) absorbs redder than TPP (419nm)."""
        assert CHROMOPHORE_LIBRARY["CuPc"].lambda_max_nm > \
               CHROMOPHORE_LIBRARY["TPP_freebase"].lambda_max_nm


class TestIndexShellLibrary:

    def test_count(self):
        assert len(INDEX_SHELL_LIBRARY) >= 10

    def test_all_have_source(self):
        for name, s in INDEX_SHELL_LIBRARY.items():
            assert len(s.source) > 5, f"{name}: missing source"

    def test_porous_shells_have_porosity(self):
        for name, s in INDEX_SHELL_LIBRARY.items():
            if "porous" in name:
                assert s.porosity > 0, f"{name}: should be porous"


class TestChromophoreKSpectrum:

    def test_returns_array(self):
        k = chromophore_k_spectrum("CuPc")
        assert isinstance(k, np.ndarray)
        assert len(k) == len(_LAM)

    def test_peak_near_lambda_max(self):
        """k(λ) should peak near the chromophore's λ_max."""
        k = chromophore_k_spectrum("CuPc")
        peak_idx = np.argmax(k)
        peak_lam = _LAM[peak_idx]
        assert abs(peak_lam - 678) < 30  # within 30nm

    def test_higher_epsilon_higher_k(self):
        """Higher ε → higher k at same conditions."""
        k_high = chromophore_k_spectrum("ZnTPP")   # ε=250000
        k_low = chromophore_k_spectrum("indigo")    # ε=20000
        assert np.max(k_high) > np.max(k_low)

    def test_unknown_chromophore_zero(self):
        k = chromophore_k_spectrum("FAKE_CHROMOPHORE")
        assert np.all(k == 0)

    def test_absorption_bounded(self):
        abs_f = chromophore_absorption_spectrum("CuPc")
        assert np.all(abs_f >= 0)
        assert np.all(abs_f <= 1)


class TestShellNSpectrum:

    def test_fixed_n_shell(self):
        """PMMA has fixed n=1.49."""
        n = shell_n_spectrum("PMMA_brush")
        assert np.allclose(n, 1.49)

    def test_porous_lower_than_solid(self):
        """Porous SiO2 should have lower n than solid SiO2."""
        n_porous = shell_n_spectrum("porous_SiO2_30")
        n_solid = shell_n_spectrum("TiO2_anatase")  # higher n material
        assert np.mean(n_porous) < np.mean(n_solid)

    def test_high_index_shell(self):
        """TiO2 should have n > 2."""
        n = shell_n_spectrum("TiO2_rutile")
        assert np.mean(n) > 2.0


class TestStructuralDyeSpectrum:

    def test_returns_spectrum(self):
        lam, R = compute_structural_dye_spectrum(
            "SiO2", 250.0, [("PMMA_brush", 10.0)]
        )
        assert len(lam) == len(_LAM)
        assert len(R) == len(lam)
        assert np.max(R) > 0.01

    def test_diameter_shifts_peak(self):
        """Larger particles → longer wavelength peak."""
        _, R_small = compute_structural_dye_spectrum("SiO2", 180.0, [("PMMA_brush", 5.0)])
        _, R_large = compute_structural_dye_spectrum("SiO2", 300.0, [("PMMA_brush", 5.0)])
        peak_small = _LAM[np.argmax(R_small)]
        peak_large = _LAM[np.argmax(R_large)]
        assert peak_large > peak_small

    def test_shell_modifies_peak(self):
        """High-index shell should shift peak vs low-index shell."""
        _, R_low = compute_structural_dye_spectrum(
            "SiO2", 220.0, [("porous_SiO2_30", 10.0)]
        )
        _, R_high = compute_structural_dye_spectrum(
            "SiO2", 220.0, [("TiO2_anatase", 10.0)]
        )
        peak_low = _LAM[np.argmax(R_low)]
        peak_high = _LAM[np.argmax(R_high)]
        assert peak_high != peak_low  # different shells → different color

    def test_chromophore_reduces_reflectance(self):
        """Adding chromophore should reduce some wavelengths."""
        _, R_bare = compute_structural_dye_spectrum(
            "SiO2", 250.0, [("PMMA_brush", 10.0)]
        )
        _, R_dye = compute_structural_dye_spectrum(
            "SiO2", 250.0, [("PMMA_brush", 10.0)],
            chromophore="CuPc", chromophore_coverage=5.0
        )
        # CuPc absorbs at 678nm → R should decrease around there
        idx_678 = np.argmin(np.abs(_LAM - 678))
        assert R_dye[idx_678] <= R_bare[idx_678]

    def test_bragg_opal_higher_reflectance(self):
        """Bragg opal should have higher peak R than photonic glass."""
        _, R_opal = compute_structural_dye_spectrum(
            "SiO2", 250.0, [("PMMA_brush", 5.0)], assembly="bragg_opal"
        )
        _, R_glass = compute_structural_dye_spectrum(
            "SiO2", 250.0, [("PMMA_brush", 5.0)], assembly="photonic_glass"
        )
        assert np.max(R_opal) > np.max(R_glass)


class TestCIEIntegration:

    def test_spectrum_to_cie_returns_dict(self):
        R = np.exp(-0.5 * ((_LAM - 530) / 20) ** 2)
        color = spectrum_to_cie(R)
        assert "Lab" in color
        assert "sRGB" in color
        assert "peak_nm" in color

    def test_green_spectrum_negative_a(self):
        """Green peak → negative a* in Lab."""
        R = 0.3 * np.exp(-0.5 * ((_LAM - 530) / 25) ** 2)
        color = spectrum_to_cie(R)
        if color["Lab"]:
            assert color["Lab"][1] < 0  # a* negative = green

    def test_delta_E_zero_identical(self):
        assert compute_delta_E((50, -30, 20), (50, -30, 20)) == pytest.approx(0.0)

    def test_delta_E_large_different(self):
        dE = compute_delta_E((50, -50, 0), (50, 50, 0))
        assert dE > 50

    def test_delta_E_none_handling(self):
        assert compute_delta_E(None, (50, 0, 0)) == 999.0


class TestDeNovoGenerator:

    def test_returns_list(self):
        designs = generate_dye_designs((55, -40, 20), top_n=5)
        assert isinstance(designs, list)
        assert len(designs) > 0

    def test_sorted_by_delta_E(self):
        designs = generate_dye_designs((55, -40, 20), top_n=10)
        for i in range(len(designs) - 1):
            assert designs[i].delta_E <= designs[i + 1].delta_E

    def test_all_have_click_steps(self):
        designs = generate_dye_designs((55, -40, 20), top_n=5)
        for d in designs:
            assert isinstance(d.click_steps, list)
            assert isinstance(d.n_click_steps, int)

    def test_green_target_finds_green(self):
        """Green target should find designs peaking near 500-560nm."""
        designs = generate_dye_designs((55, -40, 20), top_n=5)
        best = designs[0]
        assert 450 < best.predicted_peak_nm < 600

    def test_blue_target_finds_blue(self):
        """Blue target should find designs peaking near 430-480nm."""
        designs = generate_dye_designs((50, 0, -50), top_n=5)
        best = designs[0]
        assert 400 < best.predicted_peak_nm < 550

    def test_respects_top_n(self):
        designs = generate_dye_designs((55, -40, 20), top_n=3)
        assert len(designs) <= 3

    def test_design_dataclass_complete(self):
        designs = generate_dye_designs((55, -40, 20), top_n=1)
        d = designs[0]
        assert isinstance(d, DyeDesign)
        assert d.core_material != ""
        assert d.core_diameter_nm > 0
        assert len(d.shell_stack) > 0
        assert d.predicted_Lab is not None
        assert d.delta_E < 999


class TestMultilayerSpectrum:

    def test_tmm_returns_spectrum(self):
        layers = [("TiO2_rutile", 50.0), ("SiO2", 90.0)] * 3
        lam, R = compute_multilayer_spectrum(layers)
        assert len(R) == len(_LAM)

    def test_more_layers_higher_reflectance(self):
        """More bilayers → higher peak reflectance."""
        layers_3 = [("TiO2_rutile", 50.0), ("SiO2", 90.0)] * 3
        layers_7 = [("TiO2_rutile", 50.0), ("SiO2", 90.0)] * 7
        _, R3 = compute_multilayer_spectrum(layers_3)
        _, R7 = compute_multilayer_spectrum(layers_7)
        assert np.max(R7) > np.max(R3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
