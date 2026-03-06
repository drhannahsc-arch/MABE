"""
optical/application_optics.py — Application-Level Optical Model

Bridges the gap between particle-level physics (Mie, PG, TMM) and
what an observer actually sees on a real substrate.

Four physics layers:

1. COATING GEOMETRY — how particles sit on the surface
   - bulk_film:  ≥5 layers, full PG model applies (M6)
   - thin_film:  2-4 layers, reduced R, MS correction matters (M6b)
   - monolayer:  single layer of particles, NO structure factor —
                 single-particle Mie backscatter × coverage fraction
   - sparse:     sub-monolayer, isolated beads — pure Mie × f_coverage

2. VIEWING ANGLE INTEGRATION — Lambertian averaging for diffuse surfaces
   - flat_specular: single angle (glass, mirror-like film)
   - lambertian:    cosθ-weighted integral over hemisphere (textile, matte)
   - Photonic glass is ~Lambertian (angle-independent structural color)
   - Bragg opal is specular (angle-dependent, iridescent)

3. TRANSMISSION MODE — for transparent substrates (glass)
   - T(λ) = 1 - R(λ) - A(λ)
   - Useful for: stained-glass-like effects, coated windows, lab filters
   - A(λ) from Beer-Lambert through film + absorber

4. DIFFUSE SUBSTRATE — arbitrary measured R(λ) as input
   - Replace Fresnel R_under with a measured/specified curve
   - Models real fabric: black polyester, white cotton, dyed silk, etc.
   - Built-in library of common textile reflectance approximations

References:
  Gao et al. 2017, J. Nanopart. Res. (textile structural color)
  Iwata et al. 2017, Adv. Mater. 29:1605050 (substrate comparison)
  Vogel et al. 2015, PNAS 112:10845 (monolayer colloidal crystals)
"""

import sys
import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, cie_delta_E, XYZ_to_sRGB,
)

_LAM_DEFAULT = np.linspace(380, 780, 81)


# ═══════════════════════════════════════════════════════════════════════════
# 1. COATING GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoatingSpec:
    """Specification of how particles are arranged on the surface."""
    geometry: str               # "bulk_film", "thin_film", "monolayer", "sparse"
    n_layers: int = 10          # Number of particle layers (for film geometries)
    coverage_fraction: float = 1.0  # Surface coverage (0-1), for monolayer/sparse
    core_diameter_nm: float = 200.0
    core_material: str = "SiO2"
    n_medium: float = 1.0
    packing_fraction: float = 0.55  # In-plane packing for film; N/A for sparse

    # Optional: linker chain (from click_linker.py)
    shell_thickness_nm: float = 0.0
    shell_n: float = 1.46

    # Optional: absorber
    absorber_fraction: float = 0.0
    absorber_material: str = "carbon"


def coating_reflectance(spec: CoatingSpec,
                        wavelengths_nm: np.ndarray = None) -> np.ndarray:
    """Compute reflectance spectrum for a given coating geometry.

    Dispatches to the appropriate model based on spec.geometry:
      bulk_film:  → M6 photonic glass (full PG model)
      thin_film:  → M6b with thickness parameter
      monolayer:  → single-particle Mie × coverage × packing enhancement
      sparse:     → single-particle Mie × coverage (no collective effects)

    Returns R(λ) array (absolute, not normalized to 1).
    """
    if wavelengths_nm is None:
        wavelengths_nm = _LAM_DEFAULT

    if spec.geometry == "bulk_film":
        return _bulk_film_R(spec, wavelengths_nm)
    elif spec.geometry == "thin_film":
        return _thin_film_R(spec, wavelengths_nm)
    elif spec.geometry == "monolayer":
        return _monolayer_R(spec, wavelengths_nm)
    elif spec.geometry == "sparse":
        return _sparse_R(spec, wavelengths_nm)
    else:
        raise ValueError(f"Unknown geometry: {spec.geometry}. "
                         f"Options: bulk_film, thin_film, monolayer, sparse")


def _bulk_film_R(spec, lam):
    """Bulk film (≥5 layers): full photonic glass model, absolute R."""
    from optical.multiple_scattering import photonic_glass_reflectance_ms
    D_eff = spec.core_diameter_nm + 2 * spec.shell_thickness_nm
    R = photonic_glass_reflectance_ms(
        D_eff, spec.core_material, spec.n_medium, spec.packing_fraction,
        lam, absorber_fraction=spec.absorber_fraction,
        absorber_material=spec.absorber_material,
        film_thickness_layers=spec.n_layers,
        absolute=True)
    return R * spec.coverage_fraction


def _thin_film_R(spec, lam):
    """Thin film (2-4 layers): MS-corrected with reduced optical depth."""
    from optical.multiple_scattering import photonic_glass_reflectance_ms
    D_eff = spec.core_diameter_nm + 2 * spec.shell_thickness_nm
    R = photonic_glass_reflectance_ms(
        D_eff, spec.core_material, spec.n_medium, spec.packing_fraction,
        lam, absorber_fraction=spec.absorber_fraction,
        absorber_material=spec.absorber_material,
        film_thickness_layers=max(2, spec.n_layers),
        absolute=True)
    return R * spec.coverage_fraction


def _monolayer_R(spec, lam):
    """Monolayer: particles in a single layer, close-packed or partial.

    Physics: at monolayer, inter-particle distance = D (touching).
    Structure factor still applies in-plane but with reduced effect.
    Use a damped S(q): S_mono(q) ≈ 1 + (S_PY(q) - 1) × damping_factor
    where damping ≈ 0.3 for monolayer (incomplete destructive interference).
    """
    from optical.mie_scattering import mie_efficiencies
    from optical.refractive_index import n_complex, n_real
    from optical.structure_factor import structure_factor_PY

    D_eff = spec.core_diameter_nm + 2 * spec.shell_thickness_nm
    R = np.zeros(len(lam))

    # Monolayer damping factor: collective effects reduced
    damping = 0.3 * min(1.0, spec.coverage_fraction / 0.5)

    for i, wl in enumerate(lam):
        n_sph = n_real(spec.core_material, wl)
        n_eff = math.sqrt(spec.packing_fraction * n_sph**2
                          + (1 - spec.packing_fraction) * spec.n_medium**2)

        # Mie backscattering
        n_sph_c = n_complex(spec.core_material, wl)
        if spec.shell_thickness_nm > 0.1:
            from optical.core_shell_mie import mie_coated_efficiencies
            r_core = spec.core_diameter_nm / 2
            r_total = r_core + spec.shell_thickness_nm
            try:
                eff = mie_coated_efficiencies(r_core, r_total, n_sph_c,
                                               complex(spec.shell_n, 0),
                                               spec.n_medium, wl)
            except Exception:
                eff = mie_efficiencies(D_eff, n_sph_c, spec.n_medium, wl)
        else:
            eff = mie_efficiencies(D_eff, n_sph_c, spec.n_medium, wl)

        Q_back = eff["Q_back"]
        C_back = (math.pi / 4) * D_eff**2 * Q_back

        # Damped structure factor
        q_back = 4 * math.pi * n_eff / wl
        S_full = structure_factor_PY(q_back, D_eff, spec.packing_fraction)
        S_mono = 1.0 + damping * (S_full - 1.0)

        # Number density per unit area: N/A = f_cov / (π/4 × D²)
        N_per_area = spec.coverage_fraction / ((math.pi / 4) * (D_eff * 1e-9)**2)

        # Reflectance from monolayer = N/A × C_back × S_mono
        # Scale to dimensionless: multiply by (D_eff in meters)² for area normalization
        R[i] = spec.coverage_fraction * Q_back * S_mono

        # Absorber attenuation (single-pass through one layer)
        if spec.absorber_fraction > 0:
            n_abs_c = n_complex(spec.absorber_material, wl)
            k_abs = spec.absorber_fraction * n_abs_c.imag
            attenuation = math.exp(-4 * math.pi * k_abs * D_eff / wl)
            R[i] *= attenuation

    # Scale to physically reasonable absolute R for monolayer
    # Monolayer R is much lower than bulk film: typical ~0.01-0.05
    Rmax = R.max()
    if Rmax > 0:
        R = R * (0.05 / Rmax)  # peak ~5% for full-coverage monolayer
    return R


def _sparse_R(spec, lam):
    """Sparse coating: isolated particles, no collective effects.

    Pure single-particle Mie backscattering × coverage fraction.
    No structure factor (S(q) = 1 for isolated particles).
    """
    from optical.mie_scattering import mie_efficiencies
    from optical.refractive_index import n_complex

    D_eff = spec.core_diameter_nm + 2 * spec.shell_thickness_nm
    R = np.zeros(len(lam))

    for i, wl in enumerate(lam):
        n_sph_c = n_complex(spec.core_material, wl)
        if spec.shell_thickness_nm > 0.1:
            from optical.core_shell_mie import mie_coated_efficiencies
            try:
                eff = mie_coated_efficiencies(
                    spec.core_diameter_nm / 2,
                    spec.core_diameter_nm / 2 + spec.shell_thickness_nm,
                    n_sph_c, complex(spec.shell_n, 0),
                    spec.n_medium, wl)
            except Exception:
                eff = mie_efficiencies(D_eff, n_sph_c, spec.n_medium, wl)
        else:
            eff = mie_efficiencies(D_eff, n_sph_c, spec.n_medium, wl)

        R[i] = spec.coverage_fraction * eff["Q_back"]

    # Scale to physical R: sparse is very weak
    Rmax = R.max()
    if Rmax > 0:
        R = R * (0.02 * spec.coverage_fraction / Rmax)
    return R


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIEWING ANGLE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def lambertian_average(R_normal: np.ndarray,
                       wavelengths_nm: np.ndarray = None,
                       n_angles: int = 9) -> np.ndarray:
    """Cosine-weighted (Lambertian) average over viewing hemisphere.

    For angle-independent structural color (photonic glass), this mainly
    affects brightness, not hue. For iridescent surfaces (opal), this
    washes out the color (each angle contributes a different λ_peak).

    For photonic glass: R_lambertian ≈ R_normal × (2/3)
    (exact Lambertian integral of cosθ over hemisphere = 2/3 for
    angle-independent reflectance).

    For Bragg opal: need to compute R(λ,θ) at each angle and average.

    Args:
        R_normal:      Reflectance at normal incidence (0°)
        wavelengths_nm: Wavelength array (used for Bragg shift if needed)
        n_angles:      Number of angle bins for numerical integration

    Returns:
        R_lambertian(λ) — cosine-weighted average reflectance
    """
    # For photonic glass (angle-independent): simple cosθ integral
    # ∫₀^{π/2} R(λ) × cosθ × sinθ dθ / ∫₀^{π/2} cosθ × sinθ dθ
    # = R(λ) × 1 (if R is angle-independent)
    # But the detected intensity scales with cosθ (Lambert's law),
    # and we're collecting over the hemisphere, so the average is ~2/3.
    # This is a brightness correction, not a color correction.
    return R_normal * (2.0 / 3.0)


def bragg_angle_average(diameter_nm: float, n_sphere: float,
                        n_medium: float = 1.0,
                        fill_fraction: float = 0.74,
                        wavelengths_nm: np.ndarray = None,
                        n_angles: int = 9) -> np.ndarray:
    """Angle-averaged reflectance for a Bragg opal.

    At angle θ from normal, Bragg peak shifts:
      λ(θ) = λ₀ × √(1 - sin²θ/n_eff²)

    Lambertian integration produces a broad, washed-out reflection —
    this is why thin opal films look white from a distance but show
    vivid color at specific angles.

    Returns R_averaged(λ) array.
    """
    if wavelengths_nm is None:
        wavelengths_nm = _LAM_DEFAULT

    from optical.bragg_opal import bragg_opal, n_eff_volume_average

    n_eff = n_eff_volume_average(n_sphere, n_medium, fill_fraction)
    lambda_0 = bragg_opal(diameter_nm, n_sphere, n_medium, fill_fraction)

    # Bragg bandwidth (approximate): Δλ/λ₀ ≈ (4/π) × arcsin((nH-nL)/(nH+nL))
    delta_n = abs(n_sphere - n_medium)
    sum_n = n_sphere + n_medium
    bandwidth_frac = (4 / math.pi) * math.asin(delta_n / max(sum_n, 0.01))
    sigma = lambda_0 * bandwidth_frac / 2.35  # FWHM → σ

    # Angular integration
    angles = np.linspace(0, math.pi / 2 * 0.95, n_angles)  # 0 to ~86°
    weights = np.cos(angles) * np.sin(angles)  # Lambertian weight
    weights /= weights.sum()

    R_total = np.zeros(len(wavelengths_nm))
    for theta, w in zip(angles, weights):
        sin_theta = math.sin(theta)
        # Bragg shift with angle
        factor = math.sqrt(max(0, 1 - sin_theta**2 / n_eff**2))
        lambda_theta = lambda_0 * factor

        # Gaussian reflectance peak at this angle
        R_angle = np.exp(-0.5 * ((wavelengths_nm - lambda_theta)
                                  / max(sigma, 1))**2)
        R_total += w * R_angle

    # Scale: peak R of opal ~0.3-0.8 at normal, washed out to ~0.1-0.3
    R_total = R_total / R_total.max() * 0.15 if R_total.max() > 0 else R_total
    return R_total


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSMISSION MODE
# ═══════════════════════════════════════════════════════════════════════════

def transmission_spectrum(R_film: np.ndarray,
                          wavelengths_nm: np.ndarray,
                          film_absorption: np.ndarray = None,
                          substrate_T: float = 0.92) -> dict:
    """Compute transmission through a coated transparent substrate.

    T(λ) = (1 - R_film(λ) - A_film(λ)) × T_substrate

    For glass: T_substrate ≈ 0.92 (4% Fresnel loss per surface × 2).
    For the photonic glass film: R_film from M6, A from absorber.

    The transmitted color is complementary to the reflected color:
    if reflected light is blue, transmitted light appears yellow-orange.

    Args:
        R_film:          Film reflectance spectrum
        wavelengths_nm:  Wavelength array
        film_absorption: Film absorption spectrum A(λ); if None, computed
                         as A = 0 (non-absorbing film)
        substrate_T:     Bare substrate transmittance (default: glass ~0.92)

    Returns:
        dict with:
          T: transmission spectrum
          R: reflection spectrum (passed through)
          A: absorption spectrum
          cie_xy_T, Lab_T, sRGB_T: transmitted color
          cie_xy_R, Lab_R, sRGB_R: reflected color
    """
    if film_absorption is None:
        film_absorption = np.zeros(len(wavelengths_nm))

    # Ensure physical bounds
    R = np.clip(R_film, 0, 1)
    A = np.clip(film_absorption, 0, 1 - R)
    T = (1 - R - A) * substrate_T
    T = np.clip(T, 0, 1)

    # Colors
    def _extract(spectrum):
        X, Y, Z = spectrum_to_XYZ(spectrum, wavelengths_nm)
        x, y, _ = XYZ_to_xyY(X, Y, Z)
        Lab = XYZ_to_Lab(X, Y, Z)
        srgb = XYZ_to_sRGB(X, Y, Z)
        srgb_int = tuple(max(0, min(255, int(round(c * 255)))) for c in srgb)
        return (round(x, 4), round(y, 4)), tuple(round(v, 1) for v in Lab), srgb_int

    xy_T, Lab_T, sRGB_T = _extract(T)
    xy_R, Lab_R, sRGB_R = _extract(R)

    return {
        "T": T,
        "R": R,
        "A": A,
        "cie_xy_T": xy_T,
        "Lab_T": Lab_T,
        "sRGB_T": sRGB_T,
        "cie_xy_R": xy_R,
        "Lab_R": Lab_R,
        "sRGB_R": sRGB_R,
    }


def film_absorption_spectrum(diameter_nm, absorber_material, absorber_fraction,
                             film_layers, wavelengths_nm):
    """Compute absorption spectrum A(λ) for a photonic glass film.

    Beer-Lambert through the film:
      A(λ) = 1 - exp(-4π k_eff L / λ)
    where k_eff = f_abs × k_absorber, L = n_layers × D.
    """
    from optical.refractive_index import n_complex

    L_nm = film_layers * diameter_nm
    A = np.zeros(len(wavelengths_nm))
    for i, wl in enumerate(wavelengths_nm):
        k_abs = 0.0
        if absorber_fraction > 0:
            n_abs_c = n_complex(absorber_material, wl)
            k_abs = absorber_fraction * n_abs_c.imag
        alpha = 4 * math.pi * k_abs / wl  # absorption coefficient (1/nm)
        A[i] = 1 - math.exp(-alpha * L_nm)
    return np.clip(A, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DIFFUSE SUBSTRATE — ARBITRARY MEASURED R(λ)
# ═══════════════════════════════════════════════════════════════════════════

# Built-in approximate reflectance curves for common textiles
# These are simplified piecewise-linear approximations of measured data.
# Real applications should use measured R(λ) from a spectrophotometer.

def _fabric_reflectance(name, wavelengths_nm):
    """Built-in fabric reflectance approximations."""
    lam = wavelengths_nm

    if name == "black_polyester":
        # Very low, ~3-5% flat (absorbs most light)
        return np.full(len(lam), 0.04)

    elif name == "white_cotton":
        # High, ~70-80%, slightly less in UV
        R = np.where(lam < 420, 0.60, 0.78)
        return R

    elif name == "raw_linen":
        # Yellowish: absorbs blue, reflects red
        R = 0.3 + 0.3 * (lam - 380) / 400
        return np.clip(R, 0.25, 0.65)

    elif name == "red_dyed_cotton":
        # Absorbs blue/green, reflects red
        R = np.where(lam < 580, 0.05, 0.05 + 0.45 * (lam - 580) / 200)
        return np.clip(R, 0.05, 0.50)

    elif name == "blue_dyed_cotton":
        # Reflects blue, absorbs red
        R_blue = np.exp(-0.5 * ((lam - 460) / 40)**2) * 0.35 + 0.05
        return np.clip(R_blue, 0.05, 0.40)

    elif name == "green_dyed_cotton":
        R_green = np.exp(-0.5 * ((lam - 530) / 50)**2) * 0.30 + 0.05
        return np.clip(R_green, 0.05, 0.35)

    elif name == "denim":
        # Indigo: absorbs orange/red, reflects blue
        R = np.where(lam < 500, 0.10 + 0.10 * (lam - 380) / 120,
                     0.20 - 0.12 * (lam - 500) / 280)
        return np.clip(R, 0.05, 0.22)

    elif name == "black_satin":
        # Very smooth black, ~2-3%
        return np.full(len(lam), 0.025)

    else:
        raise ValueError(f"Unknown fabric: {name}. Options: "
                         f"black_polyester, white_cotton, raw_linen, "
                         f"red_dyed_cotton, blue_dyed_cotton, "
                         f"green_dyed_cotton, denim, black_satin")


FABRIC_LIBRARY = [
    "black_polyester", "white_cotton", "raw_linen",
    "red_dyed_cotton", "blue_dyed_cotton", "green_dyed_cotton",
    "denim", "black_satin",
]


def substrate_reflectance(substrate: Union[str, np.ndarray],
                          wavelengths_nm: np.ndarray) -> np.ndarray:
    """Get substrate reflectance spectrum.

    Args:
        substrate: either a fabric name (str) or a measured R(λ) array
        wavelengths_nm: wavelength array

    Returns:
        R_substrate(λ) array
    """
    if isinstance(substrate, np.ndarray):
        if len(substrate) != len(wavelengths_nm):
            raise ValueError(f"Substrate array length {len(substrate)} "
                             f"!= wavelength array length {len(wavelengths_nm)}")
        return np.clip(substrate, 0, 1)
    elif isinstance(substrate, str):
        # Try built-in fabric library first
        try:
            return _fabric_reflectance(substrate, wavelengths_nm)
        except ValueError:
            pass
        # Try underlayer module materials (carbon, Fe2O3, etc.)
        from optical.underlayer_coupling import underlayer_reflectance
        return underlayer_reflectance(substrate, wavelengths_nm)
    else:
        raise TypeError(f"substrate must be str or np.ndarray, got {type(substrate)}")


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE APPLICATION MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ApplicationResult:
    """Full application-level optical prediction."""
    scenario: str           # "glass_film", "textile_dipcoat", "beads_on_glass", "beads_on_textile"
    R_observed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    T_observed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peak_nm: float = 0.0
    cie_xy: tuple = (0.0, 0.0)
    Lab: tuple = (0.0, 0.0, 0.0)
    sRGB: tuple = (0, 0, 0)
    # Transmission color (glass only)
    cie_xy_T: tuple = (0.0, 0.0)
    Lab_T: tuple = (0.0, 0.0, 0.0)
    sRGB_T: tuple = (0, 0, 0)
    brightness: float = 0.0     # Y luminance (0-100)
    notes: str = ""


def predict_application(
    coating: CoatingSpec,
    substrate: Union[str, np.ndarray] = "black_polyester",
    viewing: str = "lambertian",
    include_transmission: bool = False,
    wavelengths_nm: np.ndarray = None,
) -> ApplicationResult:
    """Full application prediction: coating + substrate + viewing → observed color.

    Args:
        coating:              CoatingSpec defining the particle arrangement
        substrate:            Substrate name or measured R(λ) array
        viewing:              "specular" or "lambertian"
        include_transmission: If True, also compute T(λ) (glass scenarios)
        wavelengths_nm:       Wavelength array

    Returns:
        ApplicationResult with observed R, color, and optionally T
    """
    if wavelengths_nm is None:
        wavelengths_nm = _LAM_DEFAULT

    # Step 1: Coating reflectance (particle physics)
    R_coating = coating_reflectance(coating, wavelengths_nm)

    # Step 2: Substrate coupling (underlayer recycling)
    R_sub = substrate_reflectance(substrate, wavelengths_nm)
    from optical.underlayer_coupling import coupled_reflectance
    R_total = coupled_reflectance(R_coating, R_sub)

    # Step 3: Viewing angle integration
    if viewing == "lambertian":
        R_observed = lambertian_average(R_total, wavelengths_nm)
    else:
        R_observed = R_total

    # Step 4: Extract color
    X, Y, Z = spectrum_to_XYZ(R_observed, wavelengths_nm)
    x, y, Y_lum = XYZ_to_xyY(X, Y, Z)
    Lab = XYZ_to_Lab(X, Y, Z)
    srgb = XYZ_to_sRGB(X, Y, Z)
    srgb_int = tuple(max(0, min(255, int(round(c * 255)))) for c in srgb)
    peak = float(wavelengths_nm[np.argmax(R_observed)])

    result = ApplicationResult(
        scenario=f"{coating.geometry}+{substrate if isinstance(substrate, str) else 'measured'}",
        R_observed=R_observed,
        peak_nm=peak,
        cie_xy=(round(x, 4), round(y, 4)),
        Lab=tuple(round(v, 1) for v in Lab),
        sRGB=srgb_int,
        brightness=round(Y_lum, 1),
    )

    # Step 5: Transmission (optional, for glass)
    if include_transmission:
        A_film = film_absorption_spectrum(
            coating.core_diameter_nm, coating.absorber_material,
            coating.absorber_fraction, coating.n_layers, wavelengths_nm)
        t_result = transmission_spectrum(R_coating, wavelengths_nm,
                                          film_absorption=A_film)
        result.T_observed = t_result["T"]
        result.cie_xy_T = t_result["cie_xy_T"]
        result.Lab_T = t_result["Lab_T"]
        result.sRGB_T = t_result["sRGB_T"]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO SHORTCUTS
# ═══════════════════════════════════════════════════════════════════════════

def glass_film(diameter_nm, material="SiO2", n_layers=10, phi=0.55,
               absorber_fraction=0.0, substrate="carbon",
               wavelengths_nm=None, **kwargs):
    """Predict: thick photonic glass film on glass substrate."""
    spec = CoatingSpec("bulk_film", n_layers=n_layers,
                       core_diameter_nm=diameter_nm, core_material=material,
                       packing_fraction=phi, absorber_fraction=absorber_fraction,
                       **kwargs)
    return predict_application(spec, substrate, "specular",
                                include_transmission=True,
                                wavelengths_nm=wavelengths_nm)


def textile_dipcoat(diameter_nm, material="SiO2", n_layers=3, phi=0.50,
                    coverage=0.7, fabric="black_polyester",
                    absorber_fraction=0.01, wavelengths_nm=None, **kwargs):
    """Predict: dip-coated photonic glass on textile."""
    spec = CoatingSpec("thin_film", n_layers=n_layers,
                       core_diameter_nm=diameter_nm, core_material=material,
                       packing_fraction=phi, coverage_fraction=coverage,
                       absorber_fraction=absorber_fraction, **kwargs)
    return predict_application(spec, fabric, "lambertian",
                                wavelengths_nm=wavelengths_nm)


def beads_on_glass(diameter_nm, material="SiO2", coverage=0.6,
                   substrate="carbon", wavelengths_nm=None, **kwargs):
    """Predict: monolayer of beads on glass substrate."""
    spec = CoatingSpec("monolayer", coverage_fraction=coverage,
                       core_diameter_nm=diameter_nm, core_material=material,
                       **kwargs)
    return predict_application(spec, substrate, "specular",
                                include_transmission=True,
                                wavelengths_nm=wavelengths_nm)


def beads_on_textile(diameter_nm, material="SiO2", coverage=0.3,
                     fabric="black_polyester", wavelengths_nm=None, **kwargs):
    """Predict: sparse beads attached to textile fibers."""
    spec = CoatingSpec("sparse", coverage_fraction=coverage,
                       core_diameter_nm=diameter_nm, core_material=material,
                       **kwargs)
    return predict_application(spec, fabric, "lambertian",
                                wavelengths_nm=wavelengths_nm)


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON ACROSS SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

def compare_scenarios(diameter_nm, material="SiO2",
                      wavelengths_nm=None):
    """Compare all four application scenarios for the same particles.

    Returns list of ApplicationResult, one per scenario.
    """
    if wavelengths_nm is None:
        wavelengths_nm = _LAM_DEFAULT

    results = []
    results.append(glass_film(diameter_nm, material, wavelengths_nm=wavelengths_nm))
    results.append(textile_dipcoat(diameter_nm, material, wavelengths_nm=wavelengths_nm))
    results.append(beads_on_glass(diameter_nm, material, wavelengths_nm=wavelengths_nm))
    results.append(beads_on_textile(diameter_nm, material, wavelengths_nm=wavelengths_nm))
    return results


def print_comparison(results):
    """Pretty-print scenario comparison."""
    print()
    print(f"  MABE Application Scenario Comparison")
    print(f"  {'Scenario':35s}  {'Peak':>5s}  {'CIE xy':14s}  {'Y':>5s}  sRGB")
    print(f"  {'─'*85}")
    for r in results:
        print(f"  {r.scenario:35s}  {r.peak_nm:5.0f}  "
              f"({r.cie_xy[0]:.3f},{r.cie_xy[1]:.3f})  "
              f"{r.brightness:5.1f}  {r.sRGB}")
        if r.T_observed is not None and len(r.T_observed) > 0:
            print(f"    (transmitted: ({r.cie_xy_T[0]:.3f},{r.cie_xy_T[1]:.3f})  "
                  f"sRGB_T={r.sRGB_T})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MABE Application Optics — Self-Test")
    print("=" * 70)

    lam = np.linspace(380, 780, 81)

    # Compare all four scenarios for 200nm SiO2
    print("\n--- SiO2 200nm across all scenarios ---")
    results = compare_scenarios(200, "SiO2", wavelengths_nm=lam)
    print_comparison(results)

    # Textile on different fabrics
    print("--- 200nm SiO2 on different fabrics ---")
    for fabric in FABRIC_LIBRARY:
        r = textile_dipcoat(200, fabric=fabric, wavelengths_nm=lam)
        print(f"  {fabric:20s}: peak={r.peak_nm:.0f}nm  "
              f"xy=({r.cie_xy[0]:.3f},{r.cie_xy[1]:.3f})  "
              f"Y={r.brightness:.1f}  sRGB={r.sRGB}")
    print()

    # Glass transmission
    print("--- Glass film: reflected vs transmitted color ---")
    r = glass_film(200, absorber_fraction=0.02, wavelengths_nm=lam)
    print(f"  Reflected:   xy=({r.cie_xy[0]:.3f},{r.cie_xy[1]:.3f}) sRGB={r.sRGB}")
    print(f"  Transmitted: xy=({r.cie_xy_T[0]:.3f},{r.cie_xy_T[1]:.3f}) sRGB_T={r.sRGB_T}")
