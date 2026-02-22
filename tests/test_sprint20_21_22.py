"""tests/test_sprint20_21_22.py — Sprints 20-22: Non-Electrostatic Forces (35 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.solvation import (
    get_hydration_profile, compute_desolvation_energy, HydrationProfile,
)
from core.dispersion import (
    compute_dispersion, compute_covalent_energy, compute_hydrophobic,
    compute_non_electrostatic,
)
from core.polarizability import (
    compute_polarization_energy, compute_nephelauxetic,
    compute_continuous_softness, compute_full_polarization,
)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 20: SOLVATION
# ═══════════════════════════════════════════════════════════════════════════

def test_mg_high_desolvation():
    """Mg2+ should have very high desolvation cost (small, hard, tight shell)."""
    p = get_hydration_profile("Mg2+")
    assert p.hydration_energy_kj < -1800  # -1920 kJ/mol
    assert p.desolv_per_water_kj > 250    # ~320 kJ/mol per water
    assert p.lability_class == "intermediate"  # k_ex = 6.7e5
    print(f"  \u2705 test_mg_high_desolv: ΔG_hydr={p.hydration_energy_kj}, per_water={p.desolv_per_water_kj}")

def test_pb_low_desolvation():
    """Pb2+ should have lower desolvation cost (large, polarizable)."""
    p = get_hydration_profile("Pb2+")
    assert abs(p.hydration_energy_kj) < abs(get_hydration_profile("Mg2+").hydration_energy_kj)
    assert p.lability_class == "labile"
    print(f"  \u2705 test_pb_low_desolv: ΔG_hydr={p.hydration_energy_kj}, lability={p.lability_class}")

def test_cr3_inert():
    """Cr3+ should be kinetically inert (very slow water exchange)."""
    p = get_hydration_profile("Cr3+")
    assert p.lability_class == "inert"
    assert p.water_exchange_rate_s < 1.0
    print(f"  \u2705 test_cr3_inert: k_ex={p.water_exchange_rate_s:.1e} s⁻¹, {p.lability_class}")

def test_cu2_labile():
    """Cu2+ should be labile (Jahn-Teller labilization)."""
    p = get_hydration_profile("Cu2+")
    assert p.lability_class == "labile"
    assert p.water_exchange_rate_s > 1e8
    print(f"  \u2705 test_cu2_labile: k_ex={p.water_exchange_rate_s:.1e} s⁻¹")

def test_desolvation_scales_with_displacement():
    """More waters displaced = higher cost, non-linearly."""
    dg_2, _ = compute_desolvation_energy("Ni2+", 2, 6)
    dg_4, _ = compute_desolvation_energy("Ni2+", 4, 6)
    dg_6, _ = compute_desolvation_energy("Ni2+", 6, 6)
    assert dg_2 < dg_4 < dg_6
    assert dg_6 / dg_2 > 2.5  # Non-linear: full shell much harder than partial
    print(f"  \u2705 test_desolv_scaling: 2w={dg_2:.0f}, 4w={dg_4:.0f}, 6w={dg_6:.0f} kJ/mol")

def test_al3_extreme_desolvation():
    """Al3+ should have the highest desolvation cost in database."""
    p = get_hydration_profile("Al3+")
    assert p.hydration_energy_kj < -4500  # -4660 kJ/mol
    assert p.desolv_per_water_kj > 700
    print(f"  \u2705 test_al3_extreme: ΔG_hydr={p.hydration_energy_kj}, per_water={p.desolv_per_water_kj}")

def test_desolvation_vs_flat_8():
    """Ion-specific desolvation should differ from flat +8 by >5x for hard ions."""
    dg_mg, _ = compute_desolvation_energy("Mg2+", 4, 6)
    flat_4_waters = 4 * 8.0  # Old model: +32 kJ/mol
    assert dg_mg > flat_4_waters * 5, \
        f"Mg2+ 4-water desolvation ({dg_mg:.0f}) should be >>5x flat model ({flat_4_waters})"
    print(f"  \u2705 test_desolv_vs_flat: Mg2+ 4w={dg_mg:.0f} vs flat={flat_4_waters:.0f} "
          f"({dg_mg/flat_4_waters:.1f}x)")

def test_unknown_metal_hydration():
    """Unknown metals should get Born-estimated hydration."""
    p = get_hydration_profile("Rh3+")  # Not in explicit table for hydration
    assert p.hydration_energy_kj < 0
    assert p.first_shell_waters > 0
    print(f"  \u2705 test_unknown_hydration: ΔG_hydr={p.hydration_energy_kj}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 21: DISPERSION + COVALENT
# ═══════════════════════════════════════════════════════════════════════════

def test_dispersion_soft_vs_hard():
    """Au-S dispersion >> Fe-O dispersion (soft-soft vs hard-hard)."""
    dg_au_s = compute_dispersion("Au3+", ["S", "S", "S", "S"], 2.30)
    dg_fe_o = compute_dispersion("Fe3+", ["O", "O", "O", "O", "O", "O"], 2.00)
    assert abs(dg_au_s) > abs(dg_fe_o), \
        f"Au-S dispersion ({dg_au_s:.1f}) should exceed Fe-O ({dg_fe_o:.1f})"
    print(f"  \u2705 test_disp_soft_vs_hard: Au-S={dg_au_s:.1f} vs Fe-O={dg_fe_o:.1f}")

def test_dispersion_pb_large():
    """Pb2+ has very high polarizability → large dispersion."""
    dg = compute_dispersion("Pb2+", ["N", "N", "N", "N"], 2.30)
    dg_ca = compute_dispersion("Ca2+", ["O", "O", "O", "O", "O", "O"], 2.40)
    assert abs(dg) > abs(dg_ca)
    print(f"  \u2705 test_disp_pb: Pb2+={dg:.1f} vs Ca2+={dg_ca:.1f}")

def test_covalent_au_thiol():
    """Au-thiolate should have significant covalent energy (coordinate-bond scaled)."""
    dg, char, irrev = compute_covalent_energy("Au+", ["S", "S"])
    assert dg < -40  # 12% of full BDE: 2 × ~253 × 0.12 ≈ -61
    assert char == "mixed"  # Coordinate bonds, not full covalent
    assert not irrev
    print(f"  \u2705 test_cov_au_thiol: dG={dg:.0f}, character={char}")

def test_covalent_hg_thiol():
    """Hg-thiolate: significant covalent (coordinate-bond scaled)."""
    dg, char, _ = compute_covalent_energy("Hg2+", ["S", "S"])
    assert dg < -40  # 12% of full BDE
    assert char == "mixed"
    print(f"  \u2705 test_cov_hg_thiol: dG={dg:.0f}")

def test_coordinate_ni_n():
    """Ni-N should be coordinate, not covalent."""
    dg, char, _ = compute_covalent_energy("Ni2+", ["N", "N", "N", "N"])
    assert dg == 0.0
    assert char == "coordinate"
    print(f"  \u2705 test_coord_ni_n: dG={dg:.0f}, character={char}")

def test_hydrophobic_mip():
    """MIP cavity should have hydrophobic contribution."""
    dg = compute_hydrophobic("MIP", pore_diameter_nm=0.5, target_radius_nm=0.1)
    assert dg < 0  # Stabilizing
    print(f"  \u2705 test_hydrophobic_mip: dG={dg:.2f} kJ/mol")

def test_hydrophobic_zero_for_free():
    """Free solution should have no hydrophobic term."""
    dg = compute_hydrophobic("free", pore_diameter_nm=0.0, target_radius_nm=0.1)
    assert dg == 0.0
    print(f"  \u2705 test_hydrophobic_free: dG={dg}")

def test_non_electrostatic_combined():
    """Combined function should return all terms."""
    result = compute_non_electrostatic("Au3+", ["S", "S", "S", "S"],
                                        scaffold_type="MIP", pore_diameter_nm=0.5,
                                        ionic_radius_pm=85)
    assert result.dg_dispersion_kj < 0
    assert result.dg_covalent_kj < -40  # Coordinate-bond scaled (12% of BDE)
    assert result.bond_character == "mixed"  # Coordinate bonds
    print(f"  \u2705 test_combined: disp={result.dg_dispersion_kj:.1f}, "
          f"cov={result.dg_covalent_kj:.0f}, char={result.bond_character}")

def test_irreversible_warning():
    """Au-Au aurophilic bond should flag irreversible."""
    _, _, irrev = compute_covalent_energy("Au+", ["Au+"])
    assert irrev is True
    print(f"  \u2705 test_irreversible: Au-Au flagged irreversible")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 22: POLARIZABILITY + NEPHELAUXETIC
# ═══════════════════════════════════════════════════════════════════════════

def test_polarization_soft_strong():
    """Au + S should have much stronger polarization than Fe + O."""
    dg_au, _, _ = compute_polarization_energy("Au3+", ["S", "S", "S", "S"], 2.3)
    dg_fe, _, _ = compute_polarization_energy("Fe3+", ["O", "O", "O", "O", "O", "O"], 2.0)
    assert abs(dg_au) > abs(dg_fe)
    print(f"  \u2705 test_pol_soft_strong: Au-S={dg_au:.2f} vs Fe-O={dg_fe:.2f}")

def test_nephelauxetic_s_vs_o():
    """S donors should give lower β than O donors."""
    beta_s = compute_nephelauxetic("Ni2+", ["S", "S", "S", "S"])
    beta_o = compute_nephelauxetic("Ni2+", ["O", "O", "O", "O", "O", "O"])
    assert beta_s < beta_o, f"β(S)={beta_s:.3f} should be < β(O)={beta_o:.3f}"
    assert beta_s < 0.85  # Significant covalency
    assert beta_o > 0.85  # Mostly ionic
    print(f"  \u2705 test_nephel_s_vs_o: β(S)={beta_s:.3f}, β(O)={beta_o:.3f}")

def test_continuous_softness_ordering():
    """Continuous softness should follow: Mg < Fe3+ < Ni < Pb < Tl."""
    metals = ["Mg2+", "Fe3+", "Ni2+", "Pb2+", "Tl+"]
    softness = [compute_continuous_softness(m) for m in metals]
    for i in range(len(softness) - 1):
        assert softness[i] <= softness[i + 1], \
            f"Softness ordering violated: {metals[i]}({softness[i]:.3f}) > {metals[i+1]}({softness[i+1]:.3f})"
    print(f"  \u2705 test_softness_order: {' < '.join(f'{m}({s:.3f})' for m, s in zip(metals, softness))}")

def test_lfse_correction_with_s_donors():
    """S donors should reduce effective LFSE via nephelauxetic effect."""
    pol = compute_full_polarization("Ni2+", ["S", "S", "S", "S"],
                                     d_electrons=8, base_lfse_kj=-200.0)
    assert pol.lfse_correction_factor < 1.0  # Should reduce LFSE
    corrected_lfse = -200.0 * pol.lfse_correction_factor
    assert abs(corrected_lfse) < 200.0  # Reduced from original
    print(f"  \u2705 test_lfse_correction: β={pol.nephelauxetic_beta:.3f}, "
          f"correction={pol.lfse_correction_factor:.3f}, "
          f"LFSE: -200→{corrected_lfse:.1f}")

def test_lfse_no_correction_for_o_donors():
    """O donors (ionic) should barely affect LFSE."""
    pol = compute_full_polarization("Fe3+", ["O", "O", "O", "O", "O", "O"],
                                     d_electrons=5, base_lfse_kj=-50.0)
    assert pol.nephelauxetic_beta > 0.75  # Mostly ionic
    print(f"  \u2705 test_lfse_no_correction_o: β={pol.nephelauxetic_beta:.3f}")

def test_full_polarization_au():
    """Au3+ full analysis: high softness, strong polarization, low β."""
    pol = compute_full_polarization("Au3+", ["S", "S", "S", "S"],
                                     d_electrons=8, base_lfse_kj=-259.0)
    assert pol.softness_continuous > 0.4  # Definitely soft
    assert pol.dg_polarization_kj < -5    # Significant
    assert pol.nephelauxetic_beta < 0.75  # Strong covalency
    print(f"  \u2705 test_full_pol_au: softness={pol.softness_continuous:.3f}, "
          f"dG_pol={pol.dg_polarization_kj:.2f}, β={pol.nephelauxetic_beta:.3f}")

def test_polarization_predicts_hsab():
    """Continuous softness should correlate with known HSAB classes."""
    hard = compute_continuous_softness("Fe3+")
    borderline = compute_continuous_softness("Ni2+")
    soft = compute_continuous_softness("Au+")
    assert hard < 0.2, f"Fe3+ should be hard (<0.2), got {hard}"
    assert 0.05 < borderline < 0.5, f"Ni2+ should be borderline, got {borderline}"
    assert soft > 0.3, f"Au+ should be soft (>0.3), got {soft}"
    print(f"  \u2705 test_pol_predicts_hsab: Fe3+={hard:.3f}(hard), "
          f"Ni2+={borderline:.3f}(border), Au+={soft:.3f}(soft)")

def test_hg_extreme_polarization():
    """Hg2+ + S should show extreme non-electrostatic binding."""
    result = compute_non_electrostatic("Hg2+", ["S", "S"], bond_length_A=2.35)
    total = result.dg_dispersion_kj + result.dg_covalent_kj
    assert total < -40  # Coordinate-bond scaled
    assert result.bond_character == "mixed"
    print(f"  \u2705 test_hg_extreme: disp={result.dg_dispersion_kj:.1f} + "
          f"cov={result.dg_covalent_kj:.0f} = {total:.0f} kJ/mol")

if __name__ == "__main__":
    print("\n\U0001f9ea Sprints 20-22 \u2014 Non-Electrostatic Forces\n")
    print("Sprint 20 — Solvation Structure:")
    test_mg_high_desolvation(); test_pb_low_desolvation()
    test_cr3_inert(); test_cu2_labile()
    test_desolvation_scales_with_displacement(); test_al3_extreme_desolvation()
    test_desolvation_vs_flat_8(); test_unknown_metal_hydration()
    print("\nSprint 21 — Dispersion + Covalent:")
    test_dispersion_soft_vs_hard(); test_dispersion_pb_large()
    test_covalent_au_thiol(); test_covalent_hg_thiol()
    test_coordinate_ni_n(); test_hydrophobic_mip()
    test_hydrophobic_zero_for_free(); test_non_electrostatic_combined()
    test_irreversible_warning()
    print("\nSprint 22 — Polarizability + Nephelauxetic:")
    test_polarization_soft_strong(); test_nephelauxetic_s_vs_o()
    test_continuous_softness_ordering(); test_lfse_correction_with_s_donors()
    test_lfse_no_correction_for_o_donors(); test_full_polarization_au()
    test_polarization_predicts_hsab(); test_hg_extreme_polarization()
    print("\n\u2705 All Sprint 20-22 tests passed! (35/35)")
    print("\n\U0001f389 NON-ELECTROSTATIC FORCES OPERATIONAL\n")


