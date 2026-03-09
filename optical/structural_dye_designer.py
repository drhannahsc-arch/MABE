"""
optical/structural_dye_designer.py — Inverse Design for Structural Dyes

Given a target color and a substrate, find the optimal particle design:
  core diameter + shell stack + deployment parameters → target CIE color

Two-phase approach:
  Phase 1 (SCAN): Evaluate all single-shell and curated two-shell combos
    across a diameter grid. Coarse but covers the full design space.
    ~2000 forward evaluations, ~15 seconds.

  Phase 2 (REFINE): Take top candidates from scan, run continuous
    optimization on diameter + shell thickness(es) + coverage via
    scipy differential_evolution. ~200 evals per candidate, ~2s each.

The output is a ranked list of physically realizable structural dye designs,
each with:
  - Core material + diameter
  - Shell stack (with click chemistry)
  - Deployment parameters (regime, coverage)
  - Predicted color (CIE xy, Lab, sRGB)
  - ΔE from target
  - Orthogonality status

Usage:
  result = design_color_on_substrate(
      target_Lab=(60, -50, 30),   # green
      substrate="concrete",
      regime="few_layer",
  )
  print_designs(result)
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from optical.shell_library import (
    shell_optical_properties, available_shells, CHROMOPHORES, INDEX_SHELLS,
)
from optical.multi_shell import (
    ShellLayer, multi_shell_reflectance, check_orthogonality,
)
from optical.surface_optics import (
    surface_reflectance, structural_dye_on_surface,
    DeploymentSpec, SUBSTRATES,
)
from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, XYZ_to_sRGB, cie_delta_E,
)
from optical.refractive_index import n_real


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DesignCandidate:
    """A candidate structural dye design with predicted color."""
    core_material: str
    core_diameter_nm: float
    shells: list                   # List of (shell_type, thickness_nm) tuples
    click_sequence: list           # Click chemistry for each shell
    substrate: str
    regime: str
    n_layers: int
    coverage: float

    # Predicted
    peak_nm: float = 0.0
    CIE_xy: tuple = (0.0, 0.0)
    Lab: tuple = (0.0, 0.0, 0.0)
    sRGB: tuple = (0, 0, 0)
    delta_E: float = 999.0         # ΔE from target
    orthogonal: bool = True

    def shell_description(self):
        if not self.shells:
            return "bare"
        return " → ".join(f"{s[0]}({s[1]:.1f}nm)" for s in self.shells)


# ═══════════════════════════════════════════════════════════════════════════
# CURATED TWO-SHELL COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════
# Not all 272 pairs are physically interesting. Curate the useful ones.

_TWO_SHELL_COMBOS = [
    # (inner, outer, click_inner, click_outer) — all orthogonal
    # Index + chromophore (spectral shaping on index-tuned base)
    ("porous_silica_30", "CuPc", "SPAAC", "thiol_maleimide"),
    ("porous_silica_30", "TPP_freebase", "SPAAC", "thiol_maleimide"),
    ("porous_silica_30", "disperse_red_1", "SPAAC", "thiol_maleimide"),
    ("porous_silica_30", "rhodamine_B", "SPAAC", "thiol_maleimide"),
    ("porous_silica_50", "CuPc", "SPAAC", "thiol_maleimide"),
    ("porous_silica_50", "TPP_freebase", "SPAAC", "thiol_maleimide"),

    # High-n + chromophore
    ("TiO2_solgel", "CuPc", "SPAAC", "thiol_maleimide"),
    ("TiO2_solgel", "TPP_freebase", "SPAAC", "thiol_maleimide"),

    # Chromophore + chromophore (complementary absorption)
    ("CuPc", "TPP_freebase", "SPAAC", "thiol_maleimide"),
    ("CuPc", "disperse_red_1", "SPAAC", "thiol_maleimide"),
    ("TPP_freebase", "rhodamine_B", "SPAAC", "thiol_maleimide"),
    ("disperse_red_1", "fluorescein", "SPAAC", "thiol_maleimide"),

    # Chromophore + protective
    ("CuPc", "PMMA_brush", "SPAAC", "thiol_maleimide"),
    ("TPP_freebase", "PMMA_brush", "SPAAC", "thiol_maleimide"),
    ("disperse_red_1", "PMMA_brush", "SPAAC", "thiol_maleimide"),

    # Polydopamine combos (adhesion + absorption)
    ("polydopamine", "CuPc", "SPAAC", "thiol_maleimide"),
    ("polydopamine", "TPP_freebase", "SPAAC", "thiol_maleimide"),
    ("TPP_freebase", "polydopamine", "SPAAC", "thiol_maleimide"),
    ("CuPc", "polydopamine", "SPAAC", "thiol_maleimide"),

    # Index sandwich
    ("porous_silica_30", "TiO2_solgel", "SPAAC", "thiol_maleimide"),
    ("porous_silica_50", "TiO2_solgel", "SPAAC", "thiol_maleimide"),
]


# ═══════════════════════════════════════════════════════════════════════════
# FAST FORWARD MODEL (for scanning)
# ═══════════════════════════════════════════════════════════════════════════

_LAM_COARSE = np.linspace(380, 780, 81)  # 5nm steps — fast


def _evaluate_design(core_diameter, core_material, shell_specs,
                     substrate, regime, n_layers, coverage,
                     packing_fraction=0.55):
    """Fast forward evaluation → Lab color.

    Args:
        core_diameter: Core diameter (nm)
        core_material: Core material
        shell_specs: List of (shell_type, thickness_nm) tuples
        substrate, regime, n_layers, coverage: Deployment params

    Returns:
        (Lab_tuple, peak_nm) or None if evaluation fails
    """
    try:
        dep = DeploymentSpec(
            substrate=substrate, regime=regime,
            n_layers=n_layers, coverage=coverage,
            packing_fraction=packing_fraction,
        )

        if not shell_specs:
            # Bare particle
            n_p = n_real(core_material, 550)
            r = surface_reflectance(
                core_diameter, complex(n_p, 0), dep,
                wavelengths_nm=_LAM_COARSE,
            )
        else:
            shells = []
            clicks = ["SPAAC", "thiol_maleimide", "IEDDA", "CuAAC"]
            for i, (stype, thick) in enumerate(shell_specs):
                click = clicks[i % len(clicks)]
                shells.append(ShellLayer(stype, thick, click=click))

            r = structural_dye_on_surface(
                core_material, core_diameter, shells, dep,
                wavelengths_nm=_LAM_COARSE,
            )

        return (r["Lab"], r["peak_nm"])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: SCAN
# ═══════════════════════════════════════════════════════════════════════════

def _scan_design_space(target_Lab, core_material, substrate, regime,
                        n_layers, coverage, diameter_range=(120, 350),
                        n_diameters=8, shell_thickness=2.0):
    """Scan all shell configurations across a diameter grid.

    Returns list of (delta_E, DesignCandidate) sorted by ΔE.
    """
    all_shells = available_shells()
    single_shells = all_shells["chromophores"] + all_shells["index_modifications"]
    diameters = np.linspace(diameter_range[0], diameter_range[1], n_diameters)

    candidates = []

    # ── Bare particles ────────────────────────────────────────────────────
    for D in diameters:
        result = _evaluate_design(D, core_material, [],
                                   substrate, regime, n_layers, coverage)
        if result is None:
            continue
        Lab, peak = result
        dE = cie_delta_E(target_Lab, Lab)
        candidates.append((dE, DesignCandidate(
            core_material=core_material, core_diameter_nm=D,
            shells=[], click_sequence=[],
            substrate=substrate, regime=regime,
            n_layers=n_layers, coverage=coverage,
            peak_nm=peak, Lab=Lab, delta_E=dE,
        )))

    # ── Single-shell ──────────────────────────────────────────────────────
    for stype in single_shells:
        for D in diameters:
            result = _evaluate_design(
                D, core_material, [(stype, shell_thickness)],
                substrate, regime, n_layers, coverage,
            )
            if result is None:
                continue
            Lab, peak = result
            dE = cie_delta_E(target_Lab, Lab)
            candidates.append((dE, DesignCandidate(
                core_material=core_material, core_diameter_nm=D,
                shells=[(stype, shell_thickness)],
                click_sequence=["SPAAC"],
                substrate=substrate, regime=regime,
                n_layers=n_layers, coverage=coverage,
                peak_nm=peak, Lab=Lab, delta_E=dE,
            )))

    # ── Two-shell combos ──────────────────────────────────────────────────
    for inner, outer, click1, click2 in _TWO_SHELL_COMBOS:
        for D in diameters:
            result = _evaluate_design(
                D, core_material,
                [(inner, shell_thickness), (outer, shell_thickness)],
                substrate, regime, n_layers, coverage,
            )
            if result is None:
                continue
            Lab, peak = result
            dE = cie_delta_E(target_Lab, Lab)
            candidates.append((dE, DesignCandidate(
                core_material=core_material, core_diameter_nm=D,
                shells=[(inner, shell_thickness), (outer, shell_thickness)],
                click_sequence=[click1, click2],
                substrate=substrate, regime=regime,
                n_layers=n_layers, coverage=coverage,
                peak_nm=peak, Lab=Lab, delta_E=dE,
                orthogonal=check_orthogonality([click1, click2])[0],
            )))

    candidates.sort(key=lambda x: x[0])
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: REFINE
# ═══════════════════════════════════════════════════════════════════════════

def _refine_candidate(candidate, target_Lab, diameter_range=(100, 400),
                       thickness_range=(0.5, 8.0), coverage_range=(0.3, 0.95),
                       max_iter=80, seed=42):
    """Refine a candidate design via continuous optimization.

    Optimizes: core_diameter, shell_thickness(es), coverage.
    """
    from scipy.optimize import differential_evolution

    n_shells = len(candidate.shells)

    # Build bounds: [diameter, thickness_1, ..., thickness_n, coverage]
    bounds = [diameter_range]
    for _ in range(n_shells):
        bounds.append(thickness_range)
    bounds.append(coverage_range)

    shell_types = [s[0] for s in candidate.shells]
    n_evals = [0]

    def _obj(params):
        n_evals[0] += 1
        D = params[0]
        thicknesses = params[1:1+n_shells]
        cov = params[-1]

        shell_specs = [(st, t) for st, t in zip(shell_types, thicknesses)]
        result = _evaluate_design(
            D, candidate.core_material, shell_specs,
            candidate.substrate, candidate.regime,
            candidate.n_layers, cov,
        )
        if result is None:
            return 999.0
        Lab, _ = result
        return cie_delta_E(target_Lab, Lab)

    try:
        opt = differential_evolution(
            _obj, bounds,
            maxiter=max_iter,
            seed=seed,
            tol=0.01,
            polish=True,
        )

        D_opt = opt.x[0]
        thick_opt = list(opt.x[1:1+n_shells])
        cov_opt = opt.x[-1]

        # Re-evaluate at optimum for full result
        shell_specs = [(st, t) for st, t in zip(shell_types, thick_opt)]
        result = _evaluate_design(
            D_opt, candidate.core_material, shell_specs,
            candidate.substrate, candidate.regime,
            candidate.n_layers, cov_opt,
        )

        if result is not None:
            Lab, peak = result
            refined = DesignCandidate(
                core_material=candidate.core_material,
                core_diameter_nm=round(D_opt, 1),
                shells=[(st, round(t, 1)) for st, t in zip(shell_types, thick_opt)],
                click_sequence=candidate.click_sequence,
                substrate=candidate.substrate,
                regime=candidate.regime,
                n_layers=candidate.n_layers,
                coverage=round(cov_opt, 3),
                peak_nm=peak,
                Lab=Lab,
                delta_E=round(opt.fun, 1),
                orthogonal=candidate.orthogonal,
            )

            # Fill CIE_xy and sRGB
            dep = DeploymentSpec(
                substrate=refined.substrate, regime=refined.regime,
                n_layers=refined.n_layers, coverage=refined.coverage,
            )
            lam = np.linspace(380, 780, 201)
            if not refined.shells:
                n_p = n_real(refined.core_material, 550)
                r = surface_reflectance(
                    refined.core_diameter_nm, complex(n_p, 0), dep,
                    wavelengths_nm=lam,
                )
            else:
                shells = []
                for i, (stype, thick) in enumerate(refined.shells):
                    shells.append(ShellLayer(
                        stype, thick,
                        click=refined.click_sequence[i] if i < len(refined.click_sequence) else "SPAAC",
                    ))
                r = structural_dye_on_surface(
                    refined.core_material, refined.core_diameter_nm,
                    shells, dep, wavelengths_nm=lam,
                )

            refined.CIE_xy = r["CIE_xy"]
            refined.sRGB = r["sRGB"]
            refined.peak_nm = r["peak_nm"]
            refined.Lab = r["Lab"]
            refined.delta_E = round(cie_delta_E(target_Lab, r["Lab"]), 1)
            return refined

    except Exception:
        pass

    return candidate  # return unrefined if optimization fails


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def design_color_on_substrate(
    target_Lab: tuple = None,
    target_CIE_xy: tuple = None,
    substrate: str = "concrete",
    regime: str = "few_layer",
    n_layers: int = 3,
    coverage: float = 0.80,
    core_material: str = "SiO2",
    diameter_range: tuple = (120, 350),
    n_top_refine: int = 5,
    max_refine_iter: int = 80,
    verbose: bool = True,
):
    """Design a structural dye to produce a target color on a given substrate.

    Provide EITHER target_Lab OR target_CIE_xy.

    Args:
        target_Lab: (L*, a*, b*) target color
        target_CIE_xy: (x, y) target chromaticity (converted to Lab internally)
        substrate: Target substrate name
        regime: Deployment regime
        n_layers: Number of particle layers
        coverage: Surface coverage fraction
        core_material: Particle core material
        diameter_range: (min_nm, max_nm) for core diameter
        n_top_refine: How many top candidates to refine
        max_refine_iter: Max optimizer iterations per candidate
        verbose: Print progress

    Returns:
        dict with:
          target: target Lab
          substrate: substrate name
          designs: list of DesignCandidate, sorted by ΔE
          scan_time_s: Phase 1 time
          refine_time_s: Phase 2 time
          total_time_s: Total time
    """
    # Convert target
    if target_Lab is None and target_CIE_xy is not None:
        # Approximate Lab from xy (assume Y=50 for mid-lightness)
        x, y = target_CIE_xy
        if y > 0:
            Y = 50
            X = x * Y / y
            Z = (1 - x - y) * Y / y
            from optical.cie_color import XYZ_to_Lab as _XYZ_to_Lab
            target_Lab = _XYZ_to_Lab(X, Y, Z)
        else:
            target_Lab = (50, 0, 0)
    elif target_Lab is None:
        raise ValueError("Provide either target_Lab or target_CIE_xy")

    if verbose:
        print(f"  Target: L*={target_Lab[0]:.0f} a*={target_Lab[1]:.0f} "
              f"b*={target_Lab[2]:.0f} on {substrate} ({regime}, {n_layers} layers)")

    # ── Phase 1: Scan ────────────────────────────────────────────────────
    t0 = time.time()
    candidates = _scan_design_space(
        target_Lab, core_material, substrate, regime,
        n_layers, coverage, diameter_range=diameter_range,
    )
    t_scan = time.time() - t0

    if verbose:
        n_total = len(candidates)
        print(f"  Phase 1 scan: {n_total} configurations in {t_scan:.1f}s")
        if candidates:
            print(f"  Best scan ΔE = {candidates[0][0]:.1f} "
                  f"({candidates[0][1].shell_description()})")

    # ── Phase 2: Refine top candidates ───────────────────────────────────
    t1 = time.time()
    refined = []
    seen_types = set()

    for dE, cand in candidates[:min(n_top_refine * 3, len(candidates))]:
        # Deduplicate by shell type combination
        sig = tuple(s[0] for s in cand.shells)
        if sig in seen_types and len(refined) >= n_top_refine:
            continue
        seen_types.add(sig)

        if verbose:
            print(f"  Refining: {cand.shell_description()} "
                  f"(scan ΔE={dE:.1f})...", end="", flush=True)

        r = _refine_candidate(
            cand, target_Lab,
            diameter_range=diameter_range,
            max_iter=max_refine_iter,
        )
        refined.append(r)

        if verbose:
            print(f" → ΔE={r.delta_E:.1f}")

        if len(refined) >= n_top_refine:
            break

    t_refine = time.time() - t1
    t_total = time.time() - t0

    # Sort by ΔE
    refined.sort(key=lambda c: c.delta_E)

    if verbose:
        print(f"  Phase 2 refine: {len(refined)} candidates in {t_refine:.1f}s")
        print(f"  Total: {t_total:.1f}s")

    return {
        "target_Lab": target_Lab,
        "substrate": substrate,
        "regime": regime,
        "designs": refined,
        "scan_time_s": round(t_scan, 1),
        "refine_time_s": round(t_refine, 1),
        "total_time_s": round(t_total, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_designs(result):
    """Pretty-print inverse design results."""
    L, a, b = result["target_Lab"]
    print()
    print(f"  Target: L*={L:.0f} a*={a:.0f} b*={b:.0f}  "
          f"on {result['substrate']} ({result['regime']})")
    print(f"  Search: {result['total_time_s']:.1f}s")
    print()
    print(f"  {'#':>2} {'ΔE':>5} {'Shell stack':<35} {'D':>5} "
          f"{'peak':>5} {'L*':>4} {'a*':>5} {'b*':>5} {'sRGB':>13} {'orth':>4}")
    print("  " + "-" * 90)

    for i, d in enumerate(result["designs"]):
        desc = d.shell_description()
        if len(desc) > 33:
            desc = desc[:30] + "..."
        L, a, b = d.Lab
        sr, sg, sb = d.sRGB
        ri, gi, bi = int(sr*255), int(sg*255), int(sb*255)
        orth = "✓" if d.orthogonal else "✗"
        print(f"  {i+1:>2} {d.delta_E:>5.1f} {desc:<35} {d.core_diameter_nm:>4.0f}nm "
              f"{d.peak_nm:>4.0f}nm {L:>4.0f} {a:>+5.0f} {b:>+5.0f} "
              f"({ri:3d},{gi:3d},{bi:3d}) {orth:>4}")

    print()
    if result["designs"]:
        best = result["designs"][0]
        print(f"  Best design: {best.shell_description()}")
        print(f"    Core: {best.core_material} {best.core_diameter_nm:.0f}nm")
        print(f"    Coverage: {best.coverage:.0%}")
        print(f"    ΔE from target: {best.delta_E:.1f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: NAMED COLOR TARGETS
# ═══════════════════════════════════════════════════════════════════════════

_COLOR_TARGETS = {
    "red":          (50, 60, 40),
    "green":        (55, -60, 40),
    "blue":         (40, 10, -60),
    "yellow":       (85, -10, 80),
    "orange":       (70, 40, 70),
    "purple":       (35, 50, -50),
    "teal":         (55, -35, -15),
    "gold":         (75, 5, 65),
    "forest_green": (40, -40, 25),
    "sky_blue":     (60, -10, -35),
}


def design_named_color(color_name, substrate="textile_black", **kwargs):
    """Design a structural dye for a named color target.

    Available colors: red, green, blue, yellow, orange, purple, teal,
    gold, forest_green, sky_blue.
    """
    target = _COLOR_TARGETS.get(color_name.lower())
    if target is None:
        raise ValueError(f"Unknown color: {color_name}. "
                         f"Available: {sorted(_COLOR_TARGETS.keys())}")
    return design_color_on_substrate(target_Lab=target, substrate=substrate, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("  MABE Structural Dye Designer — Inverse Design")
    print("  " + "=" * 48)
    print()

    result = design_named_color("green", substrate="concrete")
    print_designs(result)
