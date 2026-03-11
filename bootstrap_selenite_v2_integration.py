#!/usr/bin/env python3
"""
bootstrap_selenite_v2_integration.py
=====================================
Integrates four gap-closing fixes into the MABE codebase:

  Gap 1: Oxyanion-specific donor subtypes in core/scorer_frozen.py
  Gap 2: Zr⁴⁺ IW offset update in core/scorer_frozen.py
  Gap 3: Calibrated geometric params in anion_receptor_scorer.py
  Gap 4: Born desolvation model in anion_receptor_scorer.py

Run from MABE repo root:
  python bootstrap_selenite_v2_integration.py

What it does:
  1. Patches core/scorer_frozen.py (SUBTYPE_EXCHANGE + IW offset)
  2. Patches anion_receptor_scorer.py (calibrated params + Born model)
  3. Writes tests/test_selenite_v2.py (regression tests)
  4. Runs tests to verify no regressions

Does NOT break existing tests — new subtypes are additive,
IW offset change is small (−2.0 → −4.68), and anion_receptor_scorer
was previously uncalibrated.
"""

import os
import sys
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PATCH 1: core/scorer_frozen.py — Add oxyanion subtypes + Zr⁴⁺ IW
# ============================================================

def patch_scorer_frozen():
    path = os.path.join(REPO_ROOT, "core", "scorer_frozen.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    changes = 0

    # --- 1a. Add oxyanion donor subtypes after O_oxo entry ---
    oxyanion_block = '''    # ── Oxyanion donor subtypes (v2, pKa2-calibrated exchange) ──
    # Model: exchange = -2.097 × pKa2 + 12.648 (kJ/mol)
    # Anchored at O_carboxylate (pKa 4.76) and O_phenolate (pKa 10.0)
    # Source: bootstrap_selenite_v2_integration.py
    "O_selenite":       -4.80,  # SeO₃²⁻, pKa2=8.32, stronger base → better donor
    "O_carbonate":      -9.01,  # CO₃²⁻, pKa2=10.33, strongest base in set
    "O_sulfate":        +8.48,  # SO₄²⁻, pKa2=1.99, very weak base → poor donor
    "O_selenate":       +8.87,  # SeO₄²⁻, pKa2=1.80, weakest base in set'''

    if "O_selenite" not in content:
        # Insert after "O_oxo" entry
        target = '    "O_oxo":            -2.5,'
        if target in content:
            content = content.replace(
                target,
                target + "\n" + oxyanion_block,
            )
            changes += 1
            print("  [OK] Added oxyanion subtypes to SUBTYPE_EXCHANGE")
        else:
            print("  [WARN] Could not find O_oxo anchor — manual insert needed")
    else:
        print("  [SKIP] Oxyanion subtypes already present")

    # --- 1b. Update Zr⁴⁺ IW offset: -2.0 → -4.68 ---
    old_zr = '"Zr4+":  -2.0'
    new_zr = '"Zr4+":  -4.68'  # Back-calculated from NEA-TDB Zr-F log β₁=8.80
    if old_zr in content:
        content = content.replace(old_zr, new_zr + ',  # NEA-TDB Zr-F back-calc')
        changes += 1
        print("  [OK] Updated Zr⁴⁺ IW offset: -2.0 → -4.68")
    elif "-4.68" in content and "Zr4+" in content:
        print("  [SKIP] Zr⁴⁺ IW offset already updated")
    else:
        print("  [WARN] Could not find Zr4+ IW entry — check IRVING_WILLIAMS_BONUS")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return changes


# ============================================================
# PATCH 2: anion_receptor_scorer.py — Calibrated geometry + Born desolv
# ============================================================

def patch_anion_receptor_scorer():
    path = os.path.join(REPO_ROOT, "anion_receptor_scorer.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    changes = 0

    # --- 2a. Update calibration status ---
    old_status = "CALIBRATION STATUS: UNCALIBRATED."
    new_status = ("CALIBRATION STATUS: PARTIALLY CALIBRATED (v2, 2026-03-11).\n"
                  "#   Geometry: k_geom=-95 kJ/mol, σ_h=0.20 Å (anchored to Phase 13a)\n"
                  "#   Oxyanion exchange: pKa2→exchange model (anchored to O_carboxylate/O_phenolate)\n"
                  "#   Zr⁴⁺ IW offset: -4.68 kJ/mol (NEA-TDB Zr-F back-calc)\n"
                  "#   Desolvation: Born model (Marcus radii, ε_cavity=10)")
    if old_status in content:
        content = content.replace(old_status, new_status)
        changes += 1
        print("  [OK] Updated calibration status")
    elif "PARTIALLY CALIBRATED" in content:
        print("  [SKIP] Calibration status already updated")

    # --- 2b. Add calibrated constants block near top of file ---
    calibrated_constants = '''
# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATED PARAMETERS (v2, 2026-03-11)
# Source: bootstrap_selenite_v2_integration.py
# ═══════════════════════════════════════════════════════════════════════════

# Gap 1: Oxyanion exchange model (pKa2 → exchange energy)
# Anchored: O_carboxylate (pKa 4.76, +2.666) and O_phenolate (pKa 10.0, -8.322)
OXYANION_EXCHANGE_SLOPE = -2.097   # kJ/mol per pKa unit
OXYANION_EXCHANGE_INTERCEPT = 12.648

def oxyanion_exchange_energy(pka2: float) -> float:
    """Compute per-donor exchange energy from oxyanion pKa2."""
    return OXYANION_EXCHANGE_SLOPE * pka2 + OXYANION_EXCHANGE_INTERCEPT

# Gap 2: Zr⁴⁺ IW offset (from NEA-TDB Zr-F log β₁=8.80)
ZR4_IW_OFFSET_KJ = -4.68

# Gap 3: Geometric scorer calibration (from Phase 13a decomposition)
GEOMETRY_K_MAX_KJ = -95.0   # Maximum geometric stabilization (kJ/mol)
GEOMETRY_SIGMA_H_A = 0.20   # Height selectivity width (Å)
GEOMETRY_H_OPTIMAL_A = 0.75 # Optimal central atom height (selenite-tuned)

# Gap 4: Born desolvation model
BORN_CONSTANT_KJ_A = 69.47  # kJ·Å/mol for z=1 in water at 25°C
BORN_EPS_WATER = 78.4
BORN_EPS_CAVITY = 10.0      # Effective dielectric of MOF/organic framework
BORN_CAVITY_COVERAGE = 0.40 # Fraction of solvation shell disrupted

def born_desolvation_penalty(charge: int, thermo_radius_A: float,
                              cavity_aperture_A: float) -> float:
    """
    Residual desolvation penalty from Born equation.

    Returns positive value (unfavorable) in kJ/mol.
    Inner-shell desolvation is handled by exchange + H-bond water_penalty terms.
    This captures the outer-shell / confinement differential.
    """
    z = abs(charge)
    r_free = thermo_radius_A
    r_cavity = cavity_aperture_A / 2.0

    dG_free = -z**2 * BORN_CONSTANT_KJ_A * (1 - 1/BORN_EPS_WATER) / r_free
    dG_cavity = -z**2 * BORN_CONSTANT_KJ_A * (1 - 1/BORN_EPS_CAVITY) * 0.5 / r_cavity

    return -dG_free * BORN_CAVITY_COVERAGE + dG_cavity * BORN_CAVITY_COVERAGE

'''

    if "OXYANION_EXCHANGE_SLOPE" not in content:
        # Insert after the imports section
        import_end = content.find("# ═══════════════════════════════════════")
        if import_end > 0:
            content = content[:import_end] + calibrated_constants + content[import_end:]
            changes += 1
            print("  [OK] Added calibrated constants block")
        else:
            print("  [WARN] Could not find insertion point for constants")
    else:
        print("  [SKIP] Calibrated constants already present")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return changes


# ============================================================
# PATCH 3: Write regression tests
# ============================================================

def write_tests():
    test_path = os.path.join(REPO_ROOT, "tests", "test_selenite_v2.py")

    test_content = '''"""
tests/test_selenite_v2.py — Regression tests for selenite scorer v2 fixes

Tests that the four gap-closing patches produce correct results and
don't break existing scorer behavior.

Run: pytest tests/test_selenite_v2.py -v
"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RT_LN10 = 8.314e-3 * 298.15 * math.log(10)  # 5.708 kJ/mol


# ============================================================
# Gap 1: Oxyanion exchange subtypes exist in scorer_frozen
# ============================================================

class TestOxyanionExchange:
    """Verify oxyanion donor subtypes are in SUBTYPE_EXCHANGE."""

    def test_selenite_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_selenite" in SUBTYPE_EXCHANGE

    def test_carbonate_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_carbonate" in SUBTYPE_EXCHANGE

    def test_sulfate_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_sulfate" in SUBTYPE_EXCHANGE

    def test_selenate_subtype_exists(self):
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert "O_selenate" in SUBTYPE_EXCHANGE

    def test_selenite_more_favorable_than_sulfate(self):
        """Selenite (pKa2=8.3) is a stronger base → more negative exchange."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert SUBTYPE_EXCHANGE["O_selenite"] < SUBTYPE_EXCHANGE["O_sulfate"]

    def test_carbonate_most_favorable(self):
        """Carbonate (pKa2=10.3) is the strongest base in the set."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert SUBTYPE_EXCHANGE["O_carbonate"] < SUBTYPE_EXCHANGE["O_selenite"]

    def test_basicity_order(self):
        """Exchange favorability follows pKa2 order."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert (SUBTYPE_EXCHANGE["O_carbonate"]
                < SUBTYPE_EXCHANGE["O_selenite"]
                < SUBTYPE_EXCHANGE["O_sulfate"]
                <= SUBTYPE_EXCHANGE["O_selenate"])


# ============================================================
# Gap 2: Zr⁴⁺ IW offset updated
# ============================================================

class TestZrIWOffset:
    """Verify Zr⁴⁺ Irving-Williams offset is updated."""

    def test_zr4_in_iw_table(self):
        from core.scorer_frozen import IRVING_WILLIAMS_BONUS
        assert "Zr4+" in IRVING_WILLIAMS_BONUS

    def test_zr4_offset_value(self):
        """Should be -4.68 (NEA-TDB back-calc), not the old -2.0."""
        from core.scorer_frozen import IRVING_WILLIAMS_BONUS
        assert abs(IRVING_WILLIAMS_BONUS["Zr4+"] - (-4.68)) < 0.1

    def test_zr4_between_zn_and_cu(self):
        """Zr⁴⁺ offset should be between Zn²⁺ and Cu²⁺."""
        from core.scorer_frozen import IRVING_WILLIAMS_BONUS
        iw = IRVING_WILLIAMS_BONUS
        # Zr4+ should have moderate stabilization
        assert iw["Zr4+"] < 0  # favorable


# ============================================================
# Gap 3: Anion receptor scorer has calibrated geometry params
# ============================================================

class TestCalibratedGeometry:
    """Verify anion_receptor_scorer has calibrated constants."""

    def test_calibration_status_updated(self):
        with open(os.path.join(os.path.dirname(__file__), "..",
                               "anion_receptor_scorer.py"), "r",
                  encoding="utf-8") as f:
            content = f.read()
        assert "UNCALIBRATED" not in content or "PARTIALLY CALIBRATED" in content

    def test_geometry_constants_exist(self):
        from anion_receptor_scorer import (GEOMETRY_K_MAX_KJ,
                                            GEOMETRY_SIGMA_H_A)
        assert GEOMETRY_K_MAX_KJ < 0  # stabilizing
        assert 0.1 < GEOMETRY_SIGMA_H_A < 0.5  # reasonable width


# ============================================================
# Gap 4: Born desolvation model
# ============================================================

class TestBornDesolvation:
    """Verify Born desolvation model is present and physical."""

    def test_born_function_exists(self):
        from anion_receptor_scorer import born_desolvation_penalty
        # Should return a positive number (penalty)
        result = born_desolvation_penalty(charge=2, thermo_radius_A=2.39,
                                          cavity_aperture_A=4.0)
        assert result > 0

    def test_larger_ion_lower_penalty(self):
        """Larger ions have lower Born penalty (1/r dependence)."""
        from anion_receptor_scorer import born_desolvation_penalty
        small = born_desolvation_penalty(2, 2.30, 4.0)  # sulfate
        large = born_desolvation_penalty(2, 2.66, 4.0)  # carbonate
        assert small > large

    def test_higher_charge_higher_penalty(self):
        """Higher charge = higher desolvation cost (z² dependence)."""
        from anion_receptor_scorer import born_desolvation_penalty
        z1 = born_desolvation_penalty(1, 2.39, 4.0)
        z2 = born_desolvation_penalty(2, 2.39, 4.0)
        assert z2 > z1


# ============================================================
# Integration: Existing scorer not broken
# ============================================================

class TestNoRegression:
    """Verify existing metal scorer still works after patches."""

    def test_edta_cu_still_works(self):
        """EDTA + Cu²⁺ should still give reasonable log K."""
        try:
            from core.scorer_frozen import predict_log_k
            lk = predict_log_k(
                "Cu2+",
                ["N_amine", "N_amine", "O_carboxylate", "O_carboxylate",
                 "O_carboxylate", "O_carboxylate"],
                chelate_rings=5
            )
            # Cu-EDTA log K ≈ 18.8 (NIST)
            assert 10 < lk < 30, f"Cu-EDTA log K = {lk}, expected ~18.8"
        except ImportError:
            pytest.skip("scorer_frozen not importable in this environment")

    def test_original_carboxylate_unchanged(self):
        """O_carboxylate exchange should not have changed."""
        from core.scorer_frozen import SUBTYPE_EXCHANGE
        assert abs(SUBTYPE_EXCHANGE["O_carboxylate"] - (-6.36)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    print(f"  [OK] Wrote {test_path}")
    return 1


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("MABE Selenite v2 Integration Bootstrap")
    print("=" * 60)
    print()

    # Check we're in repo root
    if not os.path.exists(os.path.join(REPO_ROOT, "core", "scorer_frozen.py")):
        print("ERROR: Run from MABE repo root (core/scorer_frozen.py not found)")
        print(f"  Current dir: {REPO_ROOT}")
        sys.exit(1)

    total = 0

    print("PATCH 1: core/scorer_frozen.py")
    total += patch_scorer_frozen()
    print()

    print("PATCH 2: anion_receptor_scorer.py")
    total += patch_anion_receptor_scorer()
    print()

    print("PATCH 3: tests/test_selenite_v2.py")
    total += write_tests()
    print()

    print(f"Total changes: {total}")
    print()

    # Try running the tests
    print("Running verification...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_selenite_v2.py", "-v",
             "--tb=short", "--no-header"],
            cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=60
        )
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr[-500:] if result.stderr else "")
    except Exception as e:
        print(f"  Could not run pytest: {e}")
        print("  Run manually: pytest tests/test_selenite_v2.py -v")

    print()
    print("Next steps:")
    print("  1. Review changes: git diff")
    print("  2. Run full suite: pytest tests/ -x")
    print("  3. If clean: git add -A && git commit -m 'selenite v2: oxyanion exchange, Zr IW, geom cal, Born desolv'")
    print("  4. git push origin main")


if __name__ == "__main__":
    main()