"""
optical/red_problem.py — The Red Problem: Analysis and Solutions

The most important unsolved challenge in angle-independent structural color:
pure red is structurally difficult because Mie backscattering from individual
dielectric spheres preferentially scatters blue light, overwhelming the
structural resonance when tuned to red wavelengths.

This module:
  1. DIAGNOSES the red problem quantitatively (Mie blue/red ratio)
  2. DECOMPOSES the spectrum into S(q) structural and Mie form factor contributions
  3. SYSTEMATICALLY ATTACKS the problem with 5 physically distinct strategies
  4. PRODUCES a ranked table of red-achievable designs

The 5 attack strategies:
  A. Low-n shell (porous silica): Reduces effective n_particle → weakens
     Mie resonances. The cavity-like modes that cause blue backscatter
     become weaker when the sphere-medium index contrast decreases.

  B. Blue-selective absorber (TPP Soret band at 419nm): Directly kills
     the Mie blue contamination. Does NOT absorb red (unlike CuPc at 678nm).

  C. Combined A+B: Low-n inner shell + TPP blue absorber outer shell.
     Attacks both causes — weakened Mie AND absorbed blue residual.

  D. Fe₂O₃ underlayer: Iron oxide selectively absorbs blue/green transmitted
     light and reflects red back through the film. The substrate becomes a
     spectral filter that aids the structural resonance.

  E. High-n shell + selective absorber: TiO₂ shell (n≈2.3) sharpens the
     S(q) peak via increased index contrast. Combined with TPP to suppress
     Mie blue. Sharper peak + cleaner background.

This is the novel computational prediction for META 2026.
"""

import math
import numpy as np
from dataclasses import dataclass

from optical.refractive_index import n_complex, n_real
from optical.mie_scattering import mie_efficiencies
from optical.structure_factor import structure_factor_PY
from optical.photonic_glass import photonic_glass_reflectance
from optical.multi_shell import ShellLayer, multi_shell_reflectance
from optical.surface_optics import (
    surface_reflectance, structural_dye_on_surface, DeploymentSpec,
)
from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, XYZ_to_sRGB, cie_delta_E,
)


_LAM = np.linspace(380, 780, 201)

# Target: pure structural red in CIE Lab
_RED_TARGET = (45, 55, 35)  # L*=45, a*=+55, b*=+35 (saturated warm red)


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC TOOLS
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_mie_bias(diameter_nm, material="polystyrene", n_medium=1.0):
    """Quantify the Mie blue/red backscattering bias for a given particle.

    Returns:
        dict with Q_back at blue/green/red wavelengths, bias ratios, and
        the critical q_back/q_Sq overlap analysis.
    """
    Q = {}
    for lam_nm in [420, 450, 480, 520, 550, 600, 650, 700]:
        n_p = n_complex(material, lam_nm)
        eff = mie_efficiencies(diameter_nm, n_p, n_medium, lam_nm)
        Q[lam_nm] = eff["Q_back"]

    blue_avg = (Q[420] + Q[450] + Q[480]) / 3
    red_avg = (Q[600] + Q[650] + Q[700]) / 3
    bias = blue_avg / max(red_avg, 1e-10)

    return {
        "Q_back_spectrum": Q,
        "blue_avg_Q_back": blue_avg,
        "red_avg_Q_back": red_avg,
        "blue_red_ratio": bias,
        "diameter_nm": diameter_nm,
        "material": material,
    }


def spectral_decomposition(diameter_nm, material="polystyrene",
                            n_medium=1.0, packing_fraction=0.55):
    """Decompose photonic glass reflectance into Mie and S(q) contributions.

    R(λ) ∝ S(q_back) × Q_back(λ)

    Returns both individually so you can see where each dominates.
    """
    R_total = np.zeros(len(_LAM))
    Mie_component = np.zeros(len(_LAM))
    Sq_component = np.zeros(len(_LAM))

    for i, lam in enumerate(_LAM):
        n_p = n_complex(material, lam)
        n_p_real = n_real(material, lam)
        n_eff = math.sqrt(packing_fraction * n_p_real**2 +
                          (1 - packing_fraction) * n_medium**2)

        q_back = 4 * math.pi * n_eff / lam
        Sq = structure_factor_PY(q_back, diameter_nm, packing_fraction)

        eff = mie_efficiencies(diameter_nm, n_p, n_medium, lam)
        Q_back = eff["Q_back"]

        Mie_component[i] = Q_back
        Sq_component[i] = Sq
        R_total[i] = Q_back * Sq

    # Normalize each to peak=1
    for arr in [Mie_component, Sq_component, R_total]:
        mx = arr.max()
        if mx > 0:
            arr /= mx

    Sq_peak_lam = float(_LAM[np.argmax(Sq_component)])
    Mie_peak_lam = float(_LAM[np.argmax(Mie_component)])
    R_peak_lam = float(_LAM[np.argmax(R_total)])

    return {
        "wavelengths": _LAM,
        "R_total": R_total,
        "Mie_component": Mie_component,
        "Sq_component": Sq_component,
        "Sq_peak_nm": Sq_peak_lam,
        "Mie_peak_nm": Mie_peak_lam,
        "R_peak_nm": R_peak_lam,
        "peak_mismatch_nm": abs(Sq_peak_lam - Mie_peak_lam),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ATTACK STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RedSolution:
    """A candidate solution to the red problem."""
    strategy: str
    description: str
    core_diameter_nm: float
    core_material: str
    shells: list           # List of ShellLayer
    substrate: str
    regime: str

    # Results
    peak_nm: float = 0.0
    Lab: tuple = (0.0, 0.0, 0.0)
    sRGB: tuple = (0, 0, 0)
    CIE_xy: tuple = (0.0, 0.0)
    delta_E_from_red: float = 999.0
    a_star: float = 0.0    # positive a* = red
    red_purity: float = 0.0  # a* / sqrt(a*² + b*²)


def _evaluate_red(core_diameter, core_material, shells, substrate, regime,
                   n_layers=10, coverage=0.80):
    """Evaluate a design and compute red-relevant metrics."""
    dep = DeploymentSpec(substrate=substrate, regime=regime,
                         n_layers=n_layers, coverage=coverage)

    if shells:
        r = structural_dye_on_surface(
            core_material, core_diameter, shells, dep,
            wavelengths_nm=_LAM,
        )
    else:
        n_p = n_real(core_material, 550)
        r = surface_reflectance(core_diameter, complex(n_p, 0), dep,
                                 wavelengths_nm=_LAM)

    L, a, b = r["Lab"]
    chroma = math.sqrt(a**2 + b**2)
    red_purity = a / chroma if chroma > 1 else 0.0
    dE = cie_delta_E(_RED_TARGET, r["Lab"])

    return {
        "peak_nm": r["peak_nm"],
        "Lab": r["Lab"],
        "sRGB": r["sRGB"],
        "CIE_xy": r["CIE_xy"],
        "delta_E": dE,
        "a_star": a,
        "red_purity": red_purity,
    }


def _make_solution(strategy, description, D, material, shells,
                    substrate, regime, result):
    """Build a RedSolution from evaluation result."""
    return RedSolution(
        strategy=strategy,
        description=description,
        core_diameter_nm=D,
        core_material=material,
        shells=shells,
        substrate=substrate,
        regime=regime,
        peak_nm=result["peak_nm"],
        Lab=result["Lab"],
        sRGB=result["sRGB"],
        CIE_xy=result["CIE_xy"],
        delta_E_from_red=result["delta_E"],
        a_star=result["a_star"],
        red_purity=result["red_purity"],
    )


def attack_red_problem(substrate="textile_black", regime="thick_film",
                        core_material="SiO2",
                        diameter_range=(240, 340), n_diameters=6):
    """Systematically search for structural red using all 5 strategies.

    Returns:
        dict with:
          baseline: Bare particle results showing the problem
          solutions: List of RedSolution, sorted by ΔE from red target
          best: The single best design
          diagnosis: Mie bias data for representative diameter
    """
    diameters = np.linspace(diameter_range[0], diameter_range[1], n_diameters)
    all_solutions = []

    # ── BASELINE: bare particles (shows the problem) ─────────────────────
    baselines = []
    for D in diameters:
        try:
            result = _evaluate_red(D, core_material, [], substrate, regime)
            sol = _make_solution("baseline", f"Bare {core_material} {D:.0f}nm",
                                  D, core_material, [], substrate, regime, result)
            baselines.append(sol)
        except Exception:
            continue

    # ── STRATEGY A: Low-n porous shell ───────────────────────────────────
    for porosity_key, porosity_label in [("porous_silica_30", "30% porous"),
                                          ("porous_silica_50", "50% porous")]:
        for thickness in [5.0, 10.0]:
            for D in diameters:
                try:
                    shells = [ShellLayer(porosity_key, thickness, click="SPAAC")]
                    result = _evaluate_red(D, core_material, shells, substrate, regime)
                    desc = f"A: {porosity_label} {thickness:.0f}nm shell"
                    sol = _make_solution("A_low_n", desc, D, core_material,
                                          shells, substrate, regime, result)
                    all_solutions.append(sol)
                except Exception:
                    continue

    # ── STRATEGY B: Blue-selective absorber (TPP Soret at 419nm) ─────────
    for chrom, label in [("TPP_freebase", "TPP (Soret 419nm)"),
                          ("fluorescein", "fluorescein (490nm)")]:
        for thickness in [2.0, 5.0]:
            for cov in [1.5, 3.0]:
                for D in diameters:
                    try:
                        shells = [ShellLayer(chrom, thickness, click="SPAAC",
                                             coverage_nm2=cov)]
                        result = _evaluate_red(D, core_material, shells,
                                                substrate, regime)
                        desc = f"B: {label} {thickness:.0f}nm σ={cov:.1f}"
                        sol = _make_solution("B_blue_absorber", desc,
                                              D, core_material, shells,
                                              substrate, regime, result)
                        all_solutions.append(sol)
                    except Exception:
                        continue

    # ── STRATEGY C: Low-n + blue absorber (combined attack) ──────────────
    for porosity_key in ["porous_silica_30", "porous_silica_50"]:
        for D in diameters:
            try:
                shells = [
                    ShellLayer(porosity_key, 8.0, click="SPAAC"),
                    ShellLayer("TPP_freebase", 2.0, click="thiol_maleimide",
                               coverage_nm2=2.0),
                ]
                result = _evaluate_red(D, core_material, shells,
                                        substrate, regime)
                desc = f"C: {porosity_key} + TPP blue absorber"
                sol = _make_solution("C_combined", desc, D, core_material,
                                      shells, substrate, regime, result)
                all_solutions.append(sol)
            except Exception:
                continue

    # ── STRATEGY D: Fe₂O₃ underlayer (spectral recycling) ───────────────
    # Use Fe₂O₃ as substrate instead of black
    for D in diameters:
        try:
            # Bare particle on Fe₂O₃
            result = _evaluate_red(D, core_material, [], "Fe2O3", regime)
            desc = f"D: Fe₂O₃ underlayer (bare)"
            sol = _make_solution("D_underlayer", desc, D, core_material,
                                  [], "Fe2O3", regime, result)
            all_solutions.append(sol)
        except Exception:
            continue

    for D in diameters:
        try:
            # TPP shell on Fe₂O₃ underlayer
            shells = [ShellLayer("TPP_freebase", 3.0, click="SPAAC",
                                  coverage_nm2=2.0)]
            result = _evaluate_red(D, core_material, shells, "Fe2O3", regime)
            desc = f"D: TPP shell + Fe₂O₃ underlayer"
            sol = _make_solution("D_underlayer", desc, D, core_material,
                                  shells, "Fe2O3", regime, result)
            all_solutions.append(sol)
        except Exception:
            continue

    # ── STRATEGY E: High-n shell + blue absorber ─────────────────────────
    for D in diameters:
        try:
            shells = [
                ShellLayer("TiO2_solgel", 3.0, click="SPAAC"),
                ShellLayer("TPP_freebase", 2.0, click="thiol_maleimide",
                           coverage_nm2=2.0),
            ]
            result = _evaluate_red(D, core_material, shells, substrate, regime)
            desc = f"E: TiO₂ high-n + TPP blue absorber"
            sol = _make_solution("E_high_n_absorber", desc, D, core_material,
                                  shells, substrate, regime, result)
            all_solutions.append(sol)
        except Exception:
            continue

    # ── Sort by red quality ──────────────────────────────────────────────
    # Primary: highest a* (reddest). Secondary: lowest ΔE from target.
    all_solutions.sort(key=lambda s: (-s.a_star, s.delta_E_from_red))

    # Diagnosis for representative diameter
    mid_D = float(diameters[len(diameters)//2])
    diagnosis = diagnose_mie_bias(mid_D, core_material)
    decomp = spectral_decomposition(mid_D, core_material)

    best = all_solutions[0] if all_solutions else None

    return {
        "baselines": baselines,
        "solutions": all_solutions,
        "best": best,
        "diagnosis": diagnosis,
        "decomposition": decomp,
        "n_evaluated": len(all_solutions),
        "target_Lab": _RED_TARGET,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_red_analysis(result):
    """Pretty-print the red problem analysis."""
    diag = result["diagnosis"]
    decomp = result["decomposition"]

    print()
    print("  THE RED PROBLEM — MABE Computational Analysis")
    print("  " + "=" * 48)
    print()

    # Diagnosis
    print(f"  Mie Backscattering Bias ({diag['material']} "
          f"D={diag['diameter_nm']:.0f}nm):")
    print(f"    Blue avg Q_back:  {diag['blue_avg_Q_back']:.4f}")
    print(f"    Red avg Q_back:   {diag['red_avg_Q_back']:.4f}")
    print(f"    Blue/Red ratio:   {diag['blue_red_ratio']:.1f}×")
    print()

    # Decomposition
    print(f"  Spectral Decomposition:")
    print(f"    S(q) structural peak: {decomp['Sq_peak_nm']:.0f}nm")
    print(f"    Mie form factor peak: {decomp['Mie_peak_nm']:.0f}nm")
    print(f"    Peak mismatch: {decomp['peak_mismatch_nm']:.0f}nm — "
          f"{'PROBLEM' if decomp['peak_mismatch_nm'] > 100 else 'OK'}")
    print()

    # Baselines
    print("  BASELINE (bare particles — showing the problem):")
    print(f"  {'D':>6} {'peak':>6} {'a*':>6} {'b*':>6} {'sRGB':>13} {'Verdict':>10}")
    print("  " + "-" * 55)
    for b in result["baselines"]:
        L, a, bv = b.Lab
        sr, sg, sb = b.sRGB
        ri, gi, bi = int(sr*255), int(sg*255), int(sb*255)
        verdict = "RED" if a > 30 and bv > 0 else ("purple" if a > 0 and bv < 0 else "other")
        print(f"  {b.core_diameter_nm:>5.0f}nm {b.peak_nm:>5.0f}nm {a:>+6.1f} "
              f"{bv:>+6.1f} ({ri:3d},{gi:3d},{bi:3d}) {verdict:>10}")

    # Solutions (top 15)
    print()
    print("  SOLUTIONS (sorted by reddest a*):")
    print(f"  {'Strategy':>5} {'Description':<40} {'D':>5} {'a*':>6} "
          f"{'b*':>6} {'ΔE':>5} {'sRGB':>13}")
    print("  " + "-" * 90)
    for s in result["solutions"][:15]:
        L, a, b = s.Lab
        sr, sg, sb = s.sRGB
        ri, gi, bi = int(sr*255), int(sg*255), int(sb*255)
        desc = s.description[:38]
        print(f"  {s.strategy[:5]:>5} {desc:<40} {s.core_diameter_nm:>4.0f}nm "
              f"{a:>+6.1f} {b:>+6.1f} {s.delta_E_from_red:>5.1f} "
              f"({ri:3d},{gi:3d},{bi:3d})")

    if result["best"]:
        print()
        best = result["best"]
        print(f"  BEST DESIGN: {best.description}")
        print(f"    Core: {best.core_material} {best.core_diameter_nm:.0f}nm")
        print(f"    Shells: {' → '.join(s.shell_type for s in best.shells) if best.shells else 'bare'}")
        print(f"    Substrate: {best.substrate}")
        L, a, b = best.Lab
        print(f"    Lab: L*={L:.0f} a*={a:+.0f} b*={b:+.0f}")
        print(f"    a* (redness): {a:+.1f}  (target: +55)")
        print(f"    ΔE from red target: {best.delta_E_from_red:.1f}")
        sr, sg, sb = best.sRGB
        print(f"    sRGB: ({int(sr*255)},{int(sg*255)},{int(sb*255)})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = attack_red_problem()
    print_red_analysis(result)
