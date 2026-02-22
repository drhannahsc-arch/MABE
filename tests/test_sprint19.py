"""tests/test_sprint19.py — Sprint 19: Spin State + Strong/Weak Field LFSE (15 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.spin_state import (
    predict_spin_state, compute_lfse_for_geometry,
    _get_10dq, _get_pairing_energy,
)

# === SPIN STATE PREDICTION ===
def test_fe2_weak_field_high_spin():
    """Fe2+ d6 with weak-field water → high-spin, 4 unpaired."""
    ss = predict_spin_state("Fe2+", 6, ["water"])
    assert ss.spin_state == "high_spin"
    assert ss.unpaired_electrons == 4
    assert ss.magnetic_moment_bm > 4.5  # √(4×6) = 4.90 BM
    print(f"  \u2705 test_fe2_weak_field_hs: {ss.spin_state}, μ={ss.magnetic_moment_bm:.2f} BM, "
          f"unpaired={ss.unpaired_electrons}")

def test_fe2_strong_field_low_spin():
    """Fe2+ d6 with strong-field bipyridyl → low-spin, 0 unpaired."""
    ss = predict_spin_state("Fe2+", 6, ["bipyridyl", "bipyridyl", "bipyridyl"])
    assert ss.spin_state == "low_spin"
    assert ss.unpaired_electrons == 0
    assert ss.magnetic_moment_bm == 0.0  # Diamagnetic!
    assert abs(ss.lfse_oct_kj) > abs(predict_spin_state("Fe2+", 6, ["water"]).lfse_oct_kj)
    print(f"  \u2705 test_fe2_strong_field_ls: {ss.spin_state}, μ={ss.magnetic_moment_bm:.2f} BM, "
          f"LFSE={ss.lfse_oct_kj:.1f} kJ/mol")

def test_fe3_weak_field_high_spin():
    """Fe3+ d5 with water → high-spin, 5 unpaired, LFSE=0."""
    ss = predict_spin_state("Fe3+", 5, ["water"])
    assert ss.spin_state == "high_spin"
    assert ss.unpaired_electrons == 5
    assert ss.lfse_oct_kj == 0.0  # d5 HS has zero LFSE
    print(f"  \u2705 test_fe3_weak_field_hs: {ss.spin_state}, LFSE={ss.lfse_oct_kj:.1f}, "
          f"unpaired={ss.unpaired_electrons}")

def test_fe3_strong_field_low_spin():
    """Fe3+ d5 with cyanide → low-spin, 1 unpaired."""
    ss = predict_spin_state("Fe3+", 5, ["cyanide"] * 6)
    assert ss.spin_state == "low_spin"
    assert ss.unpaired_electrons == 1
    assert ss.magnetic_moment_bm < 2.5  # √(1×3) = 1.73 BM
    assert ss.lfse_oct_kj < -100  # Significant LFSE
    print(f"  \u2705 test_fe3_strong_field_ls: {ss.spin_state}, μ={ss.magnetic_moment_bm:.2f} BM, "
          f"LFSE={ss.lfse_oct_kj:.1f} kJ/mol")

def test_co2_weak_vs_strong():
    """Co2+ d7 should change spin state with field strength."""
    weak = predict_spin_state("Co2+", 7, ["water"])
    strong = predict_spin_state("Co2+", 7, ["cyanide"] * 6)
    assert weak.spin_state == "high_spin"
    assert weak.unpaired_electrons == 3
    assert strong.spin_state == "low_spin"
    assert strong.unpaired_electrons == 1
    print(f"  \u2705 test_co2_weak_vs_strong: water={weak.spin_state}({weak.unpaired_electrons}), "
          f"CN-={strong.spin_state}({strong.unpaired_electrons})")

def test_ni2_no_spin_choice():
    """Ni2+ d8 has no spin-state ambiguity."""
    ss = predict_spin_state("Ni2+", 8, ["imidazole"] * 4)
    assert ss.spin_state == "no_choice"
    assert ss.d_electrons == 8
    print(f"  \u2705 test_ni2_no_choice: {ss.spin_state}, LFSE_oct={ss.lfse_oct_kj:.1f}")

def test_d10_zero_lfse():
    """d10 metals should have zero LFSE regardless of ligands."""
    for metal in ["Zn2+", "Ag+"]:
        ss = predict_spin_state(metal, 10, ["imidazole"] * 4)
        assert ss.lfse_oct_kj == 0.0
        assert ss.lfse_tet_kj == 0.0
    print(f"  \u2705 test_d10_zero_lfse: Zn2+ and Ag+ both LFSE=0")

def test_cr3_always_high_lfse():
    """Cr3+ d3 has 3 unpaired regardless, but large LFSE."""
    ss = predict_spin_state("Cr3+", 3, ["imidazole"] * 6)
    assert ss.unpaired_electrons == 3
    assert ss.lfse_oct_kj < -100  # Strong LFSE
    print(f"  \u2705 test_cr3_high_lfse: unpaired=3, LFSE={ss.lfse_oct_kj:.1f}")

def test_au3_large_10dq():
    """Au3+ (3rd row) should have very large 10Dq."""
    ten_dq = _get_10dq("Au3+", ["thiolate"] * 4)
    assert ten_dq > 300  # 3rd row + field strength
    print(f"  \u2705 test_au3_large_10dq: 10Dq={ten_dq:.0f} kJ/mol")

def test_mn2_hs_zero_lfse():
    """Mn2+ d5 high-spin → LFSE = 0 (half-filled shell)."""
    ss = predict_spin_state("Mn2+", 5, ["water"])
    assert ss.spin_state == "high_spin"
    assert ss.lfse_oct_kj == 0.0
    assert ss.unpaired_electrons == 5
    print(f"  \u2705 test_mn2_hs_zero_lfse: LFSE={ss.lfse_oct_kj}, unpaired={ss.unpaired_electrons}")

# === GEOMETRY-SPECIFIC LFSE ===
def test_lfse_square_planar_d8():
    """d8 square planar should have very large LFSE."""
    r = compute_lfse_for_geometry("Ni2+", 8, "square_planar", ["bipyridyl"] * 2)
    assert r.lfse_kj < -100  # Very favorable
    assert r.spin_state == "low_spin"  # d8 square planar is always low-spin
    assert r.unpaired_electrons == 0  # Diamagnetic
    print(f"  \u2705 test_lfse_sq_planar_d8: LFSE={r.lfse_kj:.1f}, μ={r.magnetic_moment_bm:.2f} BM")

def test_lfse_octahedral_vs_tetrahedral():
    """Octahedral LFSE should be larger than tetrahedral."""
    oct = compute_lfse_for_geometry("Ni2+", 8, "octahedral", ["imidazole"] * 6)
    tet = compute_lfse_for_geometry("Ni2+", 8, "tetrahedral", ["imidazole"] * 4)
    assert abs(oct.lfse_kj) > abs(tet.lfse_kj)
    print(f"  \u2705 test_lfse_oct_vs_tet: oct={oct.lfse_kj:.1f} vs tet={tet.lfse_kj:.1f}")

def test_jahn_teller_cu2():
    """Cu2+ d9 should show strong Jahn-Teller."""
    r = compute_lfse_for_geometry("Cu2+", 9, "octahedral", ["imidazole"] * 6)
    assert r.jahn_teller == "strong"
    print(f"  \u2705 test_jahn_teller_cu2: JT={r.jahn_teller}")

def test_magnetic_moment_fe2_comparison():
    """Fe2+ magnetic moment should differ dramatically HS vs LS."""
    hs = predict_spin_state("Fe2+", 6, ["water"])
    ls = predict_spin_state("Fe2+", 6, ["bipyridyl"] * 3)
    assert hs.magnetic_moment_bm > 4.0   # ~4.90 BM
    assert ls.magnetic_moment_bm == 0.0   # Diamagnetic
    print(f"  \u2705 test_magnetic_fe2: HS μ={hs.magnetic_moment_bm:.2f}, LS μ={ls.magnetic_moment_bm:.2f} BM")

def test_spectrochemical_series_ordering():
    """10Dq should follow spectrochemical series: I- < Br- < Cl- < O < N < CN-."""
    series = ["iodide", "bromide", "chloride", "water", "imidazole", "cyanide"]
    dqs = [_get_10dq("Fe2+", [lig] * 6) for lig in series]
    for i in range(len(dqs) - 1):
        assert dqs[i] <= dqs[i + 1], \
            f"Spectrochemical violation: {series[i]}({dqs[i]:.0f}) > {series[i+1]}({dqs[i+1]:.0f})"
    print(f"  \u2705 test_spectrochemical_series: {' < '.join(f'{s}({d:.0f})' for s, d in zip(series, dqs))}")

if __name__ == "__main__":
    print("\n\U0001f9ea Sprint 19 \u2014 Spin State + Strong/Weak Field LFSE\n")
    print("Spin State Prediction:")
    test_fe2_weak_field_high_spin(); test_fe2_strong_field_low_spin()
    test_fe3_weak_field_high_spin(); test_fe3_strong_field_low_spin()
    test_co2_weak_vs_strong(); test_ni2_no_spin_choice()
    test_d10_zero_lfse(); test_cr3_always_high_lfse()
    test_au3_large_10dq(); test_mn2_hs_zero_lfse()
    print("\nGeometry-Specific LFSE:")
    test_lfse_square_planar_d8(); test_lfse_octahedral_vs_tetrahedral()
    test_jahn_teller_cu2(); test_magnetic_moment_fe2_comparison()
    test_spectrochemical_series_ordering()
    print("\n\u2705 All Sprint 19 tests passed! (15/15)")
    print("\n\U0001f389 SPIN STATE ENGINE OPERATIONAL\n")

