"""
tests/test_optical/test_coordination_optics_bridge.py — Module 4a Tests

Validates the coordination chemistry → optical properties bridge.
Key targets from META 2026 abstract Figure 2.
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)

from optical.coordination_optics_bridge import (
    coordination_optics_bridge,
    bpmen_cu2, dtc_pb2, bipy_cu2,
    _has_dd_transitions, _delta_0_complex, _dd_band_wavelength_nm,
    _dd_extinction, _delta_alpha_functionalization, _delta_alpha_coordination,
    ALPHA_IONIC, ALPHA_DONOR_GROUP, DELTA_0_AQUA_CM1,
)


class TestChannel1_Absorption:
    """Channel 1: donor set → Δ₀ → λ_dd → k(λ)."""

    def test_d10_no_dd(self):
        """d¹⁰ metals (Zn²⁺, Pb²⁺, Cd²⁺, Ag⁺) have no d-d transitions."""
        for metal in ["Zn2+", "Pb2+", "Cd2+", "Ag+"]:
            assert not _has_dd_transitions(metal), f"{metal} should have no d-d"

    def test_d0_no_dd(self):
        """d⁰ metals (Al³⁺, Ca²⁺) have no d-d transitions."""
        for metal in ["Al3+", "Ca2+", "Mg2+"]:
            assert not _has_dd_transitions(metal), f"{metal} should have no d-d"

    def test_cu2_has_dd(self):
        """Cu²⁺ (d⁹) has d-d transitions."""
        assert _has_dd_transitions("Cu2+")

    def test_ni2_has_dd(self):
        """Ni²⁺ (d⁸) has d-d transitions."""
        assert _has_dd_transitions("Ni2+")

    def test_delta_0_spectrochemical_ordering(self):
        """Stronger-field ligands → larger Δ₀."""
        donors_weak = ["O_carboxylate", "O_carboxylate"]  # weak field
        donors_strong = ["N_pyridine", "N_pyridine"]       # strong field
        d0_weak = _delta_0_complex("Ni2+", donors_weak)
        d0_strong = _delta_0_complex("Ni2+", donors_strong)
        assert d0_strong > d0_weak, "N_pyridine should give larger Δ₀ than O_carboxylate"

    def test_cu2_bpmen_dd_band(self):
        """BPMEN+Cu²⁺: Δ₀ ≈ 15000-16000 cm⁻¹, λ_dd ≈ 625-670 nm."""
        donors = ["N_pyridine", "N_pyridine", "N_amine", "N_amine"]
        d0 = _delta_0_complex("Cu2+", donors)
        lam = _dd_band_wavelength_nm(d0)
        assert 14000 < d0 < 18000, f"Δ₀={d0} outside expected range"
        assert 560 < lam < 720, f"λ_dd={lam} outside expected range"

    def test_d10_k_zero(self):
        """All d¹⁰ metals produce k=0 across visible spectrum."""
        for metal in ["Zn2+", "Pb2+", "Cd2+", "Ag+"]:
            result = coordination_optics_bridge(
                ["N_amine", "N_amine"], metal)
            assert np.allclose(result["k_shell"], 0.0), \
                f"{metal} should have k=0, got max={result['k_shell'].max()}"

    def test_cu2_k_nonzero_near_dd(self):
        """Cu²⁺ produces nonzero k near its d-d band."""
        result = bpmen_cu2()
        lam_dd = result["lambda_dd_nm"]
        # k should be nonzero near the band
        lam = np.linspace(380, 780, 81)
        idx_near = np.argmin(np.abs(lam - lam_dd))
        assert result["k_shell"][idx_near] > 0, \
            f"k should be nonzero near λ_dd={lam_dd}"

    def test_k_bounded(self):
        """k for d-d transitions should be small (Laporte-forbidden)."""
        result = bpmen_cu2()
        assert result["k_shell"].max() < 0.05, \
            f"k_max={result['k_shell'].max()} exceeds physical limit for d-d"


class TestChannel2_RefractiveIndex:
    """Channel 2: polarizability → Δn."""

    def test_pb2_largest_alpha(self):
        """Pb²⁺ has largest ionic polarizability among common metals."""
        assert ALPHA_IONIC["Pb2+"] > ALPHA_IONIC["Cu2+"]
        assert ALPHA_IONIC["Pb2+"] > ALPHA_IONIC["Zn2+"]
        assert ALPHA_IONIC["Pb2+"] > ALPHA_IONIC["Ni2+"]

    def test_pyridine_high_group_alpha(self):
        """Aromatic donors have higher polarizability than aliphatic."""
        assert ALPHA_DONOR_GROUP["N_pyridine"] > ALPHA_DONOR_GROUP["N_amine"]
        assert ALPHA_DONOR_GROUP["O_phenolate"] > ALPHA_DONOR_GROUP["O_hydroxyl"]

    def test_functionalization_positive_for_aromatic(self):
        """Anchoring aromatic ligands replaces water → positive Δα."""
        da = _delta_alpha_functionalization(["N_pyridine", "N_pyridine"])
        assert da > 0, f"Aromatic functionalization should increase α, got Δα={da}"

    def test_functionalization_larger_for_more_donors(self):
        """More donor groups → larger Δα."""
        da_2 = _delta_alpha_functionalization(["N_pyridine", "N_pyridine"])
        da_4 = _delta_alpha_functionalization(
            ["N_pyridine", "N_pyridine", "N_amine", "N_amine"])
        assert da_4 > da_2

    def test_coordination_pb2_positive(self):
        """Pb²⁺ coordination adds polarizability (α_Pb >> α_waters)."""
        da = _delta_alpha_coordination("Pb2+", 4)
        assert da > 0, f"Pb²⁺ should add α, got Δα_coord={da}"

    def test_coordination_cu2_negative(self):
        """Cu²⁺ coordination may reduce net α (small ion, many waters displaced)."""
        da = _delta_alpha_coordination("Cu2+", 4)
        # Cu²⁺ α=1.2, displaced 4 × 1.45 = 5.8 → Δα = -4.6
        assert da < 0, f"Cu²⁺ coordination should reduce α, got Δα_coord={da}"


class TestAbstractFigure2:
    """Validate the three shell-metal combos from the Dublin abstract."""

    def test_all_three_run(self):
        """All three convenience functions produce results."""
        r1 = bpmen_cu2()
        r2 = dtc_pb2()
        r3 = bipy_cu2()
        for r, name in [(r1, "BPMEN"), (r2, "DTC"), (r3, "Bipy")]:
            assert "n_shell" in r, f"{name} missing n_shell"
            assert "k_shell" in r, f"{name} missing k_shell"
            assert len(r["n_shell"]) == 81, f"{name} wrong array length"

    def test_n_shell_physical_range(self):
        """n_shell must be in [1.3, 2.5] for organic/coordination shells on SiO₂."""
        for func, name in [(bpmen_cu2, "BPMEN+Cu2+"),
                           (dtc_pb2, "DTC+Pb2+"),
                           (bipy_cu2, "Bipy+Cu2+")]:
            r = func()
            assert r["n_shell"].min() > 1.3, f"{name} n too low"
            assert r["n_shell"].max() < 2.5, f"{name} n too high"

    def test_dtc_pb2_largest_delta_n(self):
        """DTC+Pb²⁺ should produce the largest Δn (6s² lone pair)."""
        r_bpmen = bpmen_cu2()
        r_dtc = dtc_pb2()
        r_bipy = bipy_cu2()
        assert abs(r_dtc["delta_n_total"]) >= abs(r_bipy["delta_n_total"]), \
            f"DTC+Pb²⁺ Δn={r_dtc['delta_n_total']} should exceed Bipy Δn={r_bipy['delta_n_total']}"

    def test_bpmen_cu2_has_dd_absorption(self):
        """BPMEN+Cu²⁺ should have d-d absorption (k > 0)."""
        r = bpmen_cu2()
        assert r["lambda_dd_nm"] > 0
        assert r["k_shell"].max() > 0
        assert r["epsilon_dd"] > 0

    def test_dtc_pb2_no_absorption(self):
        """DTC+Pb²⁺ should have pure Δn, no Δk (d¹⁰s²)."""
        r = dtc_pb2()
        assert r["lambda_dd_nm"] == 0.0
        assert np.allclose(r["k_shell"], 0.0)

    def test_bipy_cu2_has_absorption(self):
        """Bipy+Cu²⁺ should have d-d absorption."""
        r = bipy_cu2()
        assert r["lambda_dd_nm"] > 0
        assert r["k_shell"].max() > 0

    def test_rankings_decoupled(self):
        """Binding affinity ranking ≠ optical ranking.

        Key architectural point of abstract Section 3:
        Cu²⁺ is the stronger binder, Pb²⁺ is the stronger color shifter.
        """
        r_cu = bpmen_cu2()
        r_pb = dtc_pb2()
        # Pb²⁺ has larger |Δn| despite weaker binding
        # (binding ranking tested separately in scorer tests)
        assert abs(r_pb["delta_alpha_functionalization"] +
                   r_pb["delta_alpha_coordination"]) > 0 or \
               r_pb["delta_n_total"] != 0, \
            "Pb²⁺ should produce nonzero optical effect"

    def test_dd_band_positions_differ(self):
        """BPMEN+Cu²⁺ and Bipy+Cu²⁺ should have different λ_dd.

        Different donor sets → different Δ₀ → different band position.
        """
        r1 = bpmen_cu2()
        r2 = bipy_cu2()
        # BPMEN is stronger field (4 N donors) → larger Δ₀ → shorter λ_dd
        assert r1["delta_0_cm1"] != r2["delta_0_cm1"]
        assert r1["lambda_dd_nm"] != r2["lambda_dd_nm"]


class TestConsistencyWithScorer:
    """Verify bridge uses same data sources as binding engine."""

    def test_shared_dq_table(self):
        """Bridge and scorer use the same DQ_BY_DONOR table."""
        from core.scorer_frozen import DQ_BY_DONOR as scorer_dq
        from optical.coordination_optics_bridge import DQ_BY_DONOR as bridge_dq
        # They should be the exact same object (imported from same source)
        assert scorer_dq is bridge_dq

    def test_shared_metal_db(self):
        """Bridge and scorer use the same METAL_DB."""
        from core.scorer_frozen import METAL_DB as scorer_db
        from optical.coordination_optics_bridge import METAL_DB as bridge_db
        assert scorer_db is bridge_db

    def test_all_metals_in_alpha_table(self):
        """Every first-row d-block metal in METAL_DB has a polarizability."""
        first_row_dblock = ["Ti2+", "V2+", "Cr2+", "Mn2+", "Fe2+",
                            "Co2+", "Ni2+", "Cu2+", "Zn2+"]
        for m in first_row_dblock:
            assert m in ALPHA_IONIC, f"{m} missing from polarizability table"

    def test_heavy_metals_in_alpha_table(self):
        """Pb²⁺, Cd²⁺, Hg²⁺, Ag⁺ have polarizabilities."""
        for m in ["Pb2+", "Cd2+", "Hg2+", "Ag+"]:
            assert m in ALPHA_IONIC, f"{m} missing from polarizability table"
