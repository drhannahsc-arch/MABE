"""
optical/surface_optics.py — Structural Dyes on Real Substrates

Computes the observed color when structural dye particles are deposited
on real-world substrates: glass, steel, concrete, textile, plastic, wood.

The physics changes fundamentally between deployment regimes:

  SPARSE (<30% coverage): Isolated particles. Pure Mie backscatter
    modulated by substrate reflection. No inter-particle interference.
    Color from single-particle resonance × substrate spectral shaping.

  MONOLAYER (30-100%): 2D hexagonal packing on the surface. Structure
    factor is 2D, not 3D. Peak position differs from bulk photonic glass.
    Substrate reflection adds coherent interference (Fabry-Perot-like).

  FEW-LAYER (2-5 layers): Transition regime. Partial 3D S(q) develops.
    Film thickness → discrete interference fringes visible. Substrate
    coupling strongest here (light bounces through thin film).

  THICK FILM (>5 layers): Approaches bulk photonic glass. S(q) fully
    3D. Substrate effect attenuated by multiple scattering. Standard
    M6 model applies.

Substrate library:
  glass:     Soda-lime, n ≈ 1.52, transparent → transmission mode available
  steel:     Fe optical constants, highly reflective → strong image effect
  aluminum:  Al from M1 database, >85% R in visible
  concrete:  Diffuse R ≈ 0.35-0.55 (gray), broadband
  textile_white:  Diffuse R ≈ 0.70-0.85, cotton/polyester
  textile_black:  Diffuse R ≈ 0.03-0.08
  textile_dyed:   User-specified base R(λ)
  wood:      Diffuse, warm-toned R(λ)
  plastic_clear: n ≈ 1.49 (PMMA/PC), transparent
  plastic_white: Diffuse R ≈ 0.80, TiO₂-filled polymer
  plastic_black: Diffuse R ≈ 0.05, carbon-filled

All substrates are scored as deployment targets — the same structural
dye particle may look different on every substrate.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

from optical.refractive_index import n_complex, n_real
from optical.mie_scattering import mie_efficiencies
from optical.structure_factor import structure_factor_PY
from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, XYZ_to_sRGB, cie_delta_E,
)


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTRATE DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Substrate:
    """Optical description of a deployment substrate."""
    name: str
    substrate_type: str         # "specular", "diffuse", "transparent"
    n_substrate: float = 1.5    # Real refractive index (for specular)
    k_substrate: float = 0.0    # Imaginary part (for metals)
    diffuse_R: float = 0.5      # Broadband diffuse reflectance
    spectral_R: Optional[dict] = None  # {wavelength_nm: R} for spectral substrates
    transmission: bool = False  # Can we see through it?
    notes: str = ""


SUBSTRATES = {
    # ── Transparent ──
    "glass": Substrate("glass", "transparent", n_substrate=1.52,
                       transmission=True,
                       notes="Soda-lime glass. Color visible in reflection AND transmission"),
    "plastic_clear": Substrate("plastic_clear", "transparent", n_substrate=1.49,
                                transmission=True,
                                notes="PMMA or polycarbonate"),

    # ── Specular metals ──
    "steel": Substrate("steel", "specular",
                       notes="Use Fe optical constants from M1 (interpolated)"),
    "aluminum": Substrate("aluminum", "specular",
                          notes="Use Al optical constants from M1"),
    "gold_surface": Substrate("gold_surface", "specular",
                               notes="Use Au optical constants from M1"),

    # ── Diffuse opaque ──
    "concrete": Substrate("concrete", "diffuse", diffuse_R=0.45,
                          spectral_R={380: 0.35, 450: 0.40, 500: 0.43,
                                      550: 0.45, 600: 0.47, 650: 0.48,
                                      700: 0.50, 750: 0.52},
                          notes="Portland cement gray. Slightly warm-shifting"),
    "textile_white": Substrate("textile_white", "diffuse", diffuse_R=0.80,
                                spectral_R={380: 0.70, 450: 0.78, 500: 0.82,
                                            550: 0.83, 600: 0.82, 650: 0.80,
                                            700: 0.78, 750: 0.76},
                                notes="White cotton or polyester"),
    "textile_black": Substrate("textile_black", "diffuse", diffuse_R=0.05,
                                notes="Black fabric. Best for saturation (Iwata 2017)"),
    "wood": Substrate("wood", "diffuse", diffuse_R=0.40,
                       spectral_R={380: 0.10, 450: 0.18, 500: 0.30,
                                   550: 0.40, 600: 0.48, 650: 0.52,
                                   700: 0.55, 750: 0.58},
                       notes="Light wood (pine/maple). Warm-toned"),
    "plastic_white": Substrate("plastic_white", "diffuse", diffuse_R=0.82,
                                notes="TiO₂-filled polymer"),
    "plastic_black": Substrate("plastic_black", "diffuse", diffuse_R=0.05,
                                notes="Carbon-filled polymer"),
}


def _substrate_R(substrate_name, wavelengths_nm, film_n=1.35):
    """Compute substrate reflectance R_sub(λ) seen from the film side.

    For specular metals: Fresnel from film/metal interface.
    For diffuse: interpolated spectral or constant broadband R.
    For transparent: Fresnel at film/substrate interface.
    """
    sub = SUBSTRATES.get(substrate_name)
    if sub is None:
        # Try as a material in the refractive index database (metals)
        try:
            R = np.zeros(len(wavelengths_nm))
            for i, lam in enumerate(wavelengths_nm):
                n_s = n_complex(substrate_name, lam)
                r = (complex(film_n) - n_s) / (complex(film_n) + n_s)
                R[i] = min(abs(r)**2, 1.0)
            return R
        except ValueError:
            return np.full(len(wavelengths_nm), 0.04)  # default glass-like

    wavelengths_nm = np.asarray(wavelengths_nm)

    if sub.substrate_type == "specular":
        # Use M1 optical constants for metals
        R = np.zeros(len(wavelengths_nm))
        mat_name = sub.name.replace("_surface", "")
        if mat_name == "steel":
            mat_name = "Fe2O3"  # approximate — real steel would use Fe data
        try:
            for i, lam in enumerate(wavelengths_nm):
                n_s = n_complex(mat_name, lam)
                r = (complex(film_n) - n_s) / (complex(film_n) + n_s)
                R[i] = min(abs(r)**2, 1.0)
        except ValueError:
            # Fallback: high reflectance for metals
            R = np.full(len(wavelengths_nm), 0.60)
        return R

    elif sub.substrate_type == "diffuse":
        if sub.spectral_R is not None:
            # Interpolate spectral data
            lams = sorted(sub.spectral_R.keys())
            vals = [sub.spectral_R[l] for l in lams]
            R = np.interp(wavelengths_nm, lams, vals)
        else:
            R = np.full(len(wavelengths_nm), sub.diffuse_R)
        return np.clip(R, 0.0, 1.0)

    elif sub.substrate_type == "transparent":
        # Fresnel at film/substrate interface
        R = np.zeros(len(wavelengths_nm))
        for i, lam in enumerate(wavelengths_nm):
            r = (film_n - sub.n_substrate) / (film_n + sub.n_substrate)
            R[i] = abs(r)**2
        return R

    return np.full(len(wavelengths_nm), 0.04)


# ═══════════════════════════════════════════════════════════════════════════
# 2D STRUCTURE FACTOR (MONOLAYER HEXAGONAL)
# ═══════════════════════════════════════════════════════════════════════════

def structure_factor_2D_hex(q, lattice_const_nm, domain_size=10):
    """2D hexagonal structure factor for a monolayer.

    For a 2D hexagonal lattice with lattice constant a (= particle diameter
    for touching spheres), the reciprocal lattice vectors give peaks at:
      |G₁| = 4π / (√3 × a)  (first-order)

    In a finite domain of N particles, the peaks broaden. We model this
    as a Gaussian-broadened sum of Bragg peaks.

    Args:
        q: Scattering wavevector (nm⁻¹)
        lattice_const_nm: Lattice constant = center-to-center distance (nm)
        domain_size: Number of particles across domain (controls peak width)

    Returns:
        float: S₂D(q) value
    """
    a = lattice_const_nm
    if a <= 0:
        return 1.0

    # First-order reciprocal lattice vector magnitude
    G1 = 4 * math.pi / (math.sqrt(3) * a)

    # Peak width from finite domain: Δq ≈ 2π / (domain_size × a)
    sigma_q = 2 * math.pi / (domain_size * a) / 2.355  # FWHM → σ

    # Sum first few Bragg orders (hexagonal: |G| = G1 × √(h²+hk+k²))
    # First order: (1,0), (0,1), (1,1̄) → |G| = G1
    # Second order: (1,1), (2,0) → |G| = G1 × √3
    # Third order: (2,1) → |G| = G1 × √7  (skip for speed)
    S = 1.0  # baseline (no correlation = 1)

    for G_mult, degeneracy in [(1.0, 6), (math.sqrt(3), 6)]:
        G = G1 * G_mult
        peak = degeneracy * math.exp(-0.5 * ((q - G) / sigma_q)**2)
        S += peak / (domain_size**2)  # normalize by N

    return max(S, 0.01)


# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT REGIMES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeploymentSpec:
    """How structural dye particles are deployed on a substrate."""
    substrate: str              # Key in SUBSTRATES or material name
    regime: str                 # "sparse", "monolayer", "few_layer", "thick_film"
    n_layers: int = 1           # Number of particle layers
    coverage: float = 0.80      # Surface coverage fraction (0-1)
    domain_size: int = 15       # Hexagonal domain size (particles) for monolayer
    packing_fraction: float = 0.55  # 3D packing for few_layer/thick_film


def surface_reflectance(
    particle_diameter_nm: float,
    n_particle: complex,
    deployment: DeploymentSpec,
    wavelengths_nm: np.ndarray = None,
    n_medium: float = 1.0,
):
    """Compute reflectance of structural dye particles on a real substrate.

    Dispatches to the appropriate physics model based on deployment regime,
    then couples with substrate reflectance.

    Args:
        particle_diameter_nm: Total particle diameter (core + shells)
        n_particle: Complex effective refractive index of composite particle
                    (can be a scalar for constant n, or will be evaluated per-λ)
        deployment: DeploymentSpec
        wavelengths_nm: Wavelength array
        n_medium: Medium above particles (1.0 for air)

    Returns:
        dict with R, peak_nm, CIE_xy, Lab, sRGB, transmission (if applicable)
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 201)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    D = particle_diameter_nm
    dep = deployment

    # Get substrate R(λ)
    # Estimate effective film n for substrate coupling
    n_p_real = abs(n_particle.real) if isinstance(n_particle, complex) else n_particle
    n_eff_film = math.sqrt(dep.packing_fraction * n_p_real**2 +
                           (1 - dep.packing_fraction) * n_medium**2)
    R_sub = _substrate_R(dep.substrate, wavelengths_nm, film_n=n_eff_film)

    R_film = np.zeros(len(wavelengths_nm))

    for i, lam in enumerate(wavelengths_nm):
        n_p = n_particle if isinstance(n_particle, complex) else complex(n_particle, 0)

        eff = mie_efficiencies(D, n_p, n_medium, lam)
        Q_back = eff["Q_back"]

        if dep.regime == "sparse":
            # Isolated particles: R = coverage × Q_back
            # No inter-particle interference
            R_film[i] = dep.coverage * Q_back

        elif dep.regime == "monolayer":
            # 2D hexagonal structure factor
            lattice_a = D / dep.coverage**0.5  # effective spacing from coverage
            n_eff = math.sqrt(dep.coverage * n_p_real**2 +
                              (1 - dep.coverage) * n_medium**2)
            q_back = 4 * math.pi * n_eff / lam

            S2D = structure_factor_2D_hex(q_back, lattice_a, dep.domain_size)
            R_film[i] = dep.coverage * Q_back * S2D

        elif dep.regime == "few_layer":
            # Transition: partial 3D structure factor
            # Weight between 2D (1 layer) and full 3D (N layers)
            alpha = min(1.0, (dep.n_layers - 1) / 4.0)  # 0→1 over 1-5 layers

            n_eff = math.sqrt(dep.packing_fraction * n_p_real**2 +
                              (1 - dep.packing_fraction) * n_medium**2)
            q_back = 4 * math.pi * n_eff / lam

            lattice_a = D
            S2D = structure_factor_2D_hex(q_back, lattice_a, dep.domain_size)
            S3D = structure_factor_PY(q_back, D, dep.packing_fraction)
            S_eff = (1 - alpha) * S2D + alpha * S3D

            R_film[i] = dep.packing_fraction * Q_back * S_eff

            # Thickness-dependent absolute R scaling
            # Thin films have lower R than bulk
            thickness_factor = 1 - math.exp(-dep.n_layers / 3.0)
            R_film[i] *= thickness_factor

        elif dep.regime == "thick_film":
            # Full 3D photonic glass
            n_eff = math.sqrt(dep.packing_fraction * n_p_real**2 +
                              (1 - dep.packing_fraction) * n_medium**2)
            q_back = 4 * math.pi * n_eff / lam

            S3D = structure_factor_PY(q_back, D, dep.packing_fraction)
            C_back = (math.pi / 4) * D**2 * Q_back
            R_film[i] = C_back * S3D

    # Normalize film reflectance
    Rmax = R_film.max()
    if Rmax > 0:
        # Scale to physically reasonable absolute R for each regime
        target_peak_R = {
            "sparse": 0.02 * dep.coverage,
            "monolayer": 0.05 * dep.coverage,
            "few_layer": 0.03 * dep.n_layers,
            "thick_film": 0.20,
        }
        peak_R = min(target_peak_R.get(dep.regime, 0.10), 0.40)
        R_film = R_film * (peak_R / Rmax)

    # ── Couple with substrate ────────────────────────────────────────────
    # Two-pass model: light → film → substrate → film → observer
    T_film = 1.0 - R_film
    T_film = np.clip(T_film, 0.0, 1.0)

    # Substrate-recycled light (Fabry-Perot)
    R_total = R_film + T_film**2 * R_sub / (1 - R_film * R_sub + 1e-10)
    R_total = np.clip(R_total, 0.0, 1.0)

    # ── Colorimetry ──────────────────────────────────────────────────────
    X, Y, Z = spectrum_to_XYZ(R_total, wavelengths_nm)
    x, y, Y_lum = XYZ_to_xyY(X, Y, Z)
    Lab = XYZ_to_Lab(X, Y, Z)
    srgb = XYZ_to_sRGB(X, Y, Z)
    peak_nm = float(wavelengths_nm[np.argmax(R_total)])

    result = {
        "R_total": R_total,
        "R_film": R_film,
        "R_substrate": R_sub,
        "wavelengths_nm": wavelengths_nm,
        "peak_nm": peak_nm,
        "CIE_xy": (x, y),
        "Lab": Lab,
        "sRGB": srgb,
        "substrate": dep.substrate,
        "regime": dep.regime,
    }

    # Transmission for transparent substrates
    sub_obj = SUBSTRATES.get(dep.substrate)
    if sub_obj and sub_obj.transmission:
        T_total = T_film * (1 - R_sub)
        T_total = np.clip(T_total, 0.0, 1.0)
        Xt, Yt, Zt = spectrum_to_XYZ(T_total, wavelengths_nm)
        xt, yt, _ = XYZ_to_xyY(Xt, Yt, Zt)
        Lab_t = XYZ_to_Lab(Xt, Yt, Zt)
        srgb_t = XYZ_to_sRGB(Xt, Yt, Zt)
        result["T_total"] = T_total
        result["transmission_CIE_xy"] = (xt, yt)
        result["transmission_Lab"] = Lab_t
        result["transmission_sRGB"] = srgb_t
        result["transmission_peak_nm"] = float(
            wavelengths_nm[np.argmin(T_total)])  # minimum T = complement of R peak

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-SHELL PARTICLE ON SURFACE
# ═══════════════════════════════════════════════════════════════════════════

def structural_dye_on_surface(
    core_material: str,
    core_diameter_nm: float,
    shells: list,
    deployment: DeploymentSpec,
    wavelengths_nm: np.ndarray = None,
    n_medium: float = 1.0,
):
    """Deploy multi-shell structural dye particles on a real substrate.

    Connects the multi-shell compositor (optical/multi_shell.py) with
    the surface optics model.

    Args:
        core_material: Core material name
        core_diameter_nm: Core diameter (nm)
        shells: List of ShellLayer objects (from multi_shell module)
        deployment: DeploymentSpec (substrate + regime)
        wavelengths_nm: Wavelength array
        n_medium: Medium above particles

    Returns:
        dict: Same as surface_reflectance, plus particle details
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 201)

    from optical.multi_shell import ShellLayer
    from optical.shell_library import shell_optical_properties
    from optical.core_shell_mie import _effective_n_core_shell

    # Total particle diameter
    total_shell_t = sum(s.thickness_nm for s in shells)
    d_total = core_diameter_nm + 2 * total_shell_t

    # Compute wavelength-dependent effective particle n
    # Use mid-visible as reference for the surface_reflectance call
    # (surface_reflectance does per-λ Mie internally)
    r_core = core_diameter_nm / 2.0

    # Get shell optical properties
    for shell in shells:
        if not hasattr(shell, 'n_array') or shell.n_array is None:
            props = shell_optical_properties(
                shell.shell_type, wavelengths_nm,
                surface_coverage_nm2=shell.coverage_nm2,
                shell_thickness_nm=shell.thickness_nm,
            )
            shell.n_array = props["n_shell"]
            shell.k_array = props["k_shell"]

    # For surface_reflectance we need per-wavelength effective n
    # Build array of effective n_particle(λ)
    n_eff_arr = np.zeros(len(wavelengths_nm), dtype=complex)
    for i, lam in enumerate(wavelengths_nm):
        n_core = n_complex(core_material, lam)
        n_inner = n_core
        r_inner = r_core
        for shell in shells:
            n_s = complex(shell.n_array[i], shell.k_array[i])
            r_outer = r_inner + shell.thickness_nm
            n_inner = _effective_n_core_shell(n_inner, n_s, r_inner, r_outer)
            r_inner = r_outer
        n_eff_arr[i] = n_inner

    # Use median real part as the scalar n for the surface model
    # (the per-λ Mie inside surface_reflectance uses scalar n)
    n_median = complex(float(np.median(np.real(n_eff_arr))),
                        float(np.median(np.abs(np.imag(n_eff_arr)))))

    result = surface_reflectance(
        d_total, n_median, deployment,
        wavelengths_nm=wavelengths_nm,
        n_medium=n_medium,
    )

    result["core_material"] = core_material
    result["core_diameter_nm"] = core_diameter_nm
    result["total_diameter_nm"] = d_total
    result["n_shells"] = len(shells)
    result["n_particle_median"] = n_median

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SUBSTRATE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def compare_substrates(
    core_material: str = "SiO2",
    core_diameter_nm: float = 225,
    shells: list = None,
    substrates: list = None,
    regime: str = "few_layer",
    n_layers: int = 3,
    coverage: float = 0.80,
):
    """Compare the same structural dye on multiple substrates.

    Returns comparison table showing how substrate changes observed color.
    """
    if substrates is None:
        substrates = ["textile_black", "textile_white", "glass",
                      "concrete", "steel", "plastic_white", "wood"]

    lam = np.linspace(380, 780, 201)
    results = []

    for sub_name in substrates:
        dep = DeploymentSpec(
            substrate=sub_name, regime=regime,
            n_layers=n_layers, coverage=coverage,
        )

        if shells:
            r = structural_dye_on_surface(
                core_material, core_diameter_nm, shells, dep,
                wavelengths_nm=lam,
            )
        else:
            # Bare particles
            n_p = n_real(core_material, 550)
            r = surface_reflectance(
                core_diameter_nm, complex(n_p, 0), dep,
                wavelengths_nm=lam,
            )

        results.append(r)

    return results


def print_substrate_comparison(results):
    """Pretty-print substrate comparison."""
    print(f"  {'Substrate':<18} {'Regime':<12} {'peak':>6} "
          f"{'L*':>5} {'a*':>6} {'b*':>6} {'sRGB':>15}")
    print("  " + "-" * 70)
    for r in results:
        L, a, b = r["Lab"]
        sr, sg, sb = r["sRGB"]
        ri, gi, bi = int(sr*255), int(sg*255), int(sb*255)
        print(f"  {r['substrate']:<18} {r['regime']:<12} {r['peak_nm']:>5.0f}nm "
              f"{L:>5.0f} {a:>+6.1f} {b:>+6.1f} ({ri:3d},{gi:3d},{bi:3d})")


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("  MABE Surface Optics — Structural Dyes on Real Substrates")
    print("  " + "=" * 56)
    print()

    # Bare SiO₂ on different substrates
    print("  Bare 225nm SiO₂ particles, 3-layer coating:")
    print()
    results = compare_substrates(regime="few_layer", n_layers=3)
    print_substrate_comparison(results)

    # Multi-shell on substrates
    print()
    print("  CuPc-coated SiO₂ (2nm shell), 3-layer coating:")
    print()
    from optical.multi_shell import ShellLayer
    shells = [ShellLayer("CuPc", 2.0, click="SPAAC", coverage_nm2=2.0)]
    results2 = compare_substrates(shells=shells, regime="few_layer", n_layers=3)
    print_substrate_comparison(results2)
