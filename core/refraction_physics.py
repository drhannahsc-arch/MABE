"""
refraction_physics.py — Refractive Index Physics from First Principles

Five modules for computing, predicting, and manipulating refractive indices:

  F1. Kramers-Kronig: k(λ) → Δn(λ) anomalous dispersion
  F2. Fresnel: R(θ, polarization) at dielectric interfaces
  F3. Graded-index: continuously varying n(r) → sub-layer TMM
  F4. Lorentz-Lorenz: molecular polarizability → n
  F5. Effective medium: Bruggeman, Looyenga, Hashin-Shtrikman bounds

All equations T2 (established electromagnetic theory).
Published polarizabilities T1.

References:
  Born M, Wolf E. Principles of Optics, 7th ed. Cambridge 1999.
  Jackson JD. Classical Electrodynamics, 3rd ed. Wiley 1999.
  Bohren CF, Huffman DR. Absorption and Scattering of Light by Small Particles. Wiley 1983.
  Kramers HA. Atti Cong. Intern. Fisici, Como 1927, 2, 545.
  Kronig R. J. Opt. Soc. Am. 1926, 12, 547.
  Lorentz HA. Ann. Phys. 1880, 9, 641.
  Lorenz L. Ann. Phys. 1880, 11, 70.
  Fresnel A. Mém. Acad. Sci. 1823, 11, 393.
  Hashin Z, Shtrikman S. J. Appl. Phys. 1962, 33, 3125.
  Looyenga H. Physica 1965, 31, 401.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

C_LIGHT = 2.998e8          # m/s
HBAR_EV_S = 6.582e-16      # ℏ in eV·s
EPS_0 = 8.854e-12          # F/m (vacuum permittivity)
N_AVO = 6.022e23           # mol⁻¹
PI = math.pi
_LAM = np.linspace(380, 780, 81)  # default visible grid


# ═══════════════════════════════════════════════════════════════════════════
# F1. Kramers-Kronig: k(λ) → Δn(λ)
# ═══════════════════════════════════════════════════════════════════════════
#
# The Kramers-Kronig relations connect the real and imaginary parts of
# any causal response function. For refractive index:
#
#   n(ω) - 1 = (2/π) P∫₀^∞ ω'k(ω') / (ω'² - ω²) dω'
#
# Where P = Cauchy principal value.
# This means: any absorption feature (k > 0) NECESSARILY produces a
# refractive index anomaly (Δn ≠ 0) nearby — anomalous dispersion.
#
# Ref: Bohren & Huffman Ch. 9; Born & Wolf Ch. 2.

def kramers_kronig_delta_n(k_spectrum: np.ndarray,
                             wavelengths_nm: np.ndarray = None) -> np.ndarray:
    """Compute anomalous dispersion Δn(λ) from k(λ) via Kramers-Kronig.

    Uses the subtractive Kramers-Kronig (sKK) with numerical integration.
    The result is the CHANGE in n due to the absorption feature, not the
    total n (add to baseline n to get full n).

    Numerical method: trapezoidal integration with principal value
    handled by symmetric exclusion of the singularity.

    Parameters
    ----------
    k_spectrum : np.ndarray
        Extinction coefficient k(λ). Same grid as wavelengths_nm.
    wavelengths_nm : np.ndarray
        Wavelength grid (nm). Default: 380-780nm, 5nm steps.

    Returns
    -------
    np.ndarray
        Δn(λ) — anomalous dispersion contribution.

    Physics tier: T2 (Kramers 1927, Kronig 1926, exact for linear media).
    """
    if wavelengths_nm is None:
        wavelengths_nm = _LAM

    # Convert to energy (eV) for integration — more numerically stable
    E = 1239.84 / wavelengths_nm  # eV, from E = hc/λ
    N = len(E)
    delta_n = np.zeros(N)

    for i in range(N):
        # Cauchy principal value integral
        # P∫ E'·k(E') / (E'² - E²) dE'
        integrand = np.zeros(N)
        for j in range(N):
            if i == j:
                continue
            denom = E[j]**2 - E[i]**2
            if abs(denom) < 1e-20:
                continue
            integrand[j] = E[j] * k_spectrum[j] / denom

        # Trapezoidal integration
        dE = np.gradient(E)
        delta_n[i] = (2.0 / PI) * np.sum(integrand * np.abs(dE))

    return delta_n


def chromophore_nk_spectrum(name: str, surface_coverage_nm2: float = 3.0,
                              shell_thickness_nm: float = 5.0,
                              n_baseline: float = 1.50) -> Tuple[np.ndarray, np.ndarray]:
    """Full complex refractive index for a chromophore shell.

    Returns (n(λ), k(λ)) where n includes:
      - baseline n from shell matrix
      - Kramers-Kronig anomalous dispersion from chromophore absorption

    Parameters
    ----------
    name : str
        Chromophore name from structural_dye_engine.CHROMOPHORE_LIBRARY.
    surface_coverage_nm2 : float
        Molecules per nm².
    shell_thickness_nm : float
        Shell thickness (nm).
    n_baseline : float
        Baseline refractive index of the shell matrix.

    Returns
    -------
    (n_array, k_array) on _LAM grid.

    Physics tier: T2 (Beer-Lambert + Kramers-Kronig).
    """
    try:
        from core.structural_dye_engine import chromophore_k_spectrum as _get_k
    except ImportError:
        return np.full_like(_LAM, n_baseline, dtype=float), np.zeros_like(_LAM)

    k = _get_k(name, surface_coverage_nm2, shell_thickness_nm)
    delta_n = kramers_kronig_delta_n(k)

    n = np.full_like(_LAM, n_baseline, dtype=float) + delta_n
    return n, k


# ═══════════════════════════════════════════════════════════════════════════
# F2. Fresnel Coefficients
# ═══════════════════════════════════════════════════════════════════════════
#
# Fresnel equations: amplitude reflection/transmission at a planar
# interface between two media. Foundation of all thin-film optics.
#
# Ref: Born & Wolf Ch. 1; Jackson Ch. 7; Hecht, Optics Ch. 4.

def fresnel_rs(n1: float, n2: float, theta_i_rad: float) -> complex:
    """Fresnel reflection coefficient, s-polarization.

    r_s = (n1·cosθ_i - n2·cosθ_t) / (n1·cosθ_i + n2·cosθ_t)

    where θ_t from Snell's law: n1·sinθ_i = n2·sinθ_t

    Parameters
    ----------
    n1, n2 : float
        Refractive indices (real part; for absorbing media use complex version).
    theta_i_rad : float
        Angle of incidence (radians).

    Returns
    -------
    complex
        Amplitude reflection coefficient r_s.

    Physics tier: T2 (Fresnel 1823, exact for planar interfaces).
    """
    cos_i = math.cos(theta_i_rad)
    sin_i = math.sin(theta_i_rad)

    # Snell's law: cosθ_t
    sin_t_sq = (n1 / n2 * sin_i) ** 2
    if sin_t_sq > 1.0:
        # Total internal reflection
        cos_t = 1j * math.sqrt(sin_t_sq - 1.0)
    else:
        cos_t = math.sqrt(1.0 - sin_t_sq)

    num = n1 * cos_i - n2 * cos_t
    den = n1 * cos_i + n2 * cos_t
    if abs(den) < 1e-30:
        return 0.0
    return num / den


def fresnel_rp(n1: float, n2: float, theta_i_rad: float) -> complex:
    """Fresnel reflection coefficient, p-polarization.

    r_p = (n2·cosθ_i - n1·cosθ_t) / (n2·cosθ_i + n1·cosθ_t)

    Physics tier: T2.
    """
    cos_i = math.cos(theta_i_rad)
    sin_i = math.sin(theta_i_rad)

    sin_t_sq = (n1 / n2 * sin_i) ** 2
    if sin_t_sq > 1.0:
        cos_t = 1j * math.sqrt(sin_t_sq - 1.0)
    else:
        cos_t = math.sqrt(1.0 - sin_t_sq)

    num = n2 * cos_i - n1 * cos_t
    den = n2 * cos_i + n1 * cos_t
    if abs(den) < 1e-30:
        return 0.0
    return num / den


def fresnel_reflectance(n1: float, n2: float,
                          theta_i_deg: float = 0.0,
                          polarization: str = "unpolarized") -> float:
    """Intensity reflectance at a single interface.

    R = |r|² for s or p polarization.
    Unpolarized: R = (R_s + R_p) / 2.

    Parameters
    ----------
    n1, n2 : float
        Refractive indices.
    theta_i_deg : float
        Angle of incidence (degrees).
    polarization : str
        "s", "p", or "unpolarized".

    Returns
    -------
    float
        Intensity reflectance R (0-1).

    Physics tier: T2 (Fresnel equations).
    """
    theta_rad = math.radians(theta_i_deg)

    if polarization == "s":
        r = fresnel_rs(n1, n2, theta_rad)
        return abs(r) ** 2
    elif polarization == "p":
        r = fresnel_rp(n1, n2, theta_rad)
        return abs(r) ** 2
    else:
        Rs = abs(fresnel_rs(n1, n2, theta_rad)) ** 2
        Rp = abs(fresnel_rp(n1, n2, theta_rad)) ** 2
        return (Rs + Rp) / 2.0


def brewster_angle(n1: float, n2: float) -> float:
    """Brewster's angle: θ_B = arctan(n2/n1).

    At Brewster's angle, R_p = 0 (only s-polarized light reflects).

    Returns angle in degrees.
    Physics tier: T2.
    """
    if n1 <= 0:
        return 0.0
    return math.degrees(math.atan(n2 / n1))


def critical_angle(n1: float, n2: float) -> Optional[float]:
    """Critical angle for total internal reflection.

    θ_c = arcsin(n2/n1), only exists when n1 > n2.

    Returns angle in degrees, or None if no TIR possible.
    Physics tier: T2.
    """
    if n1 <= n2:
        return None  # no TIR from low-n to high-n
    ratio = n2 / n1
    if ratio > 1.0:
        return None
    return math.degrees(math.asin(ratio))


def fresnel_reflectance_spectrum(n1_array: np.ndarray, n2_array: np.ndarray,
                                   theta_i_deg: float = 0.0) -> np.ndarray:
    """Fresnel reflectance R(λ) at a single interface.

    For wavelength-dependent n1(λ), n2(λ).

    Returns unpolarized R(λ).
    Physics tier: T2.
    """
    R = np.zeros(len(n1_array))
    for i in range(len(n1_array)):
        R[i] = fresnel_reflectance(float(n1_array[i]), float(n2_array[i]),
                                     theta_i_deg, "unpolarized")
    return R


# ═══════════════════════════════════════════════════════════════════════════
# F3. Graded-Index Shells
# ═══════════════════════════════════════════════════════════════════════════
#
# A continuously varying n(r) profile discretized into sub-layers for TMM.
# Key application: anti-reflection coatings (gradient from n_core to n_air),
# and moth-eye type nanostructured surfaces.
#
# Ref: Macleod HA. Thin-Film Optical Filters. CRC 2010.

@dataclass
class GradedIndexProfile:
    """Defines a graded-index shell profile n(x) where x ∈ [0, 1]."""
    name: str = ""
    n_inner: float = 1.46      # n at inner surface (core side)
    n_outer: float = 1.0       # n at outer surface (medium side)
    total_thickness_nm: float = 100.0
    profile_type: str = "linear"  # "linear", "quintic", "exponential", "rugate"
    # For rugate: sinusoidal n modulation
    rugate_period_nm: float = 0.0
    rugate_amplitude: float = 0.0

    def n_at_x(self, x: float) -> float:
        """Refractive index at fractional position x (0=inner, 1=outer).

        Parameters
        ----------
        x : float
            Position 0 to 1.

        Returns
        -------
        float
            n at position x.
        """
        x = max(0.0, min(1.0, x))
        dn = self.n_outer - self.n_inner

        if self.profile_type == "linear":
            return self.n_inner + dn * x

        elif self.profile_type == "quintic":
            # Quintic polynomial: smooth transition with zero derivative at ends
            # f(x) = 10x³ - 15x⁴ + 6x⁵
            f = 10 * x**3 - 15 * x**4 + 6 * x**5
            return self.n_inner + dn * f

        elif self.profile_type == "exponential":
            # Exponential taper: n(x) = n_inner × (n_outer/n_inner)^x
            if self.n_inner <= 0:
                return self.n_outer
            ratio = self.n_outer / self.n_inner
            if ratio <= 0:
                return self.n_inner
            return self.n_inner * ratio ** x

        elif self.profile_type == "rugate":
            # Sinusoidal modulation on linear base
            base = self.n_inner + dn * x
            if self.rugate_period_nm > 0:
                phase = 2 * PI * x * self.total_thickness_nm / self.rugate_period_nm
                base += self.rugate_amplitude * math.sin(phase)
            return base

        return self.n_inner + dn * x  # fallback linear


def discretize_graded_index(profile: GradedIndexProfile,
                              n_sublayers: int = 20) -> List[Tuple[float, float]]:
    """Discretize a graded-index profile into TMM sub-layers.

    Returns list of (n, thickness_nm) pairs for TMM computation.

    Parameters
    ----------
    profile : GradedIndexProfile
    n_sublayers : int
        Number of discrete sub-layers.

    Returns
    -------
    list of (n_value, thickness_nm)

    Physics tier: T2 (exact in the limit of many sub-layers).
    """
    t_sub = profile.total_thickness_nm / n_sublayers
    layers = []
    for i in range(n_sublayers):
        x = (i + 0.5) / n_sublayers  # midpoint of sub-layer
        n = profile.n_at_x(x)
        layers.append((n, t_sub))
    return layers


def anti_reflection_design(n_substrate: float, n_medium: float = 1.0,
                             target_wavelength_nm: float = 550.0,
                             profile_type: str = "quintic",
                             total_thickness_nm: float = 0.0) -> GradedIndexProfile:
    """Design an anti-reflection graded-index coating.

    Optimal thickness ≈ λ/(2·n_avg) for a half-wave matching layer.
    Graded profile eliminates reflections across a broad band.

    Parameters
    ----------
    n_substrate : float
        Substrate refractive index.
    n_medium : float
        Surrounding medium (1.0 for air).
    target_wavelength_nm : float
        Center design wavelength.
    profile_type : str
        Profile shape.
    total_thickness_nm : float
        Override thickness. 0 = auto (λ/2n_avg).

    Returns
    -------
    GradedIndexProfile
    """
    n_avg = (n_substrate + n_medium) / 2.0
    if total_thickness_nm <= 0:
        total_thickness_nm = target_wavelength_nm / (2.0 * n_avg)

    return GradedIndexProfile(
        name=f"AR-{profile_type}-{target_wavelength_nm:.0f}nm",
        n_inner=n_substrate,
        n_outer=n_medium,
        total_thickness_nm=total_thickness_nm,
        profile_type=profile_type,
    )


# ═══════════════════════════════════════════════════════════════════════════
# F4. Lorentz-Lorenz: Molecular Polarizability → n
# ═══════════════════════════════════════════════════════════════════════════
#
# The Lorentz-Lorenz (Clausius-Mossotti) equation connects microscopic
# molecular polarizability α to macroscopic refractive index n:
#
#   (n² - 1)/(n² + 2) = (N·α)/(3ε₀)
#
# where N = number density (molecules/m³).
# This enables predicting n for NOVEL materials from atomic/bond
# polarizabilities without measured data.
#
# Ref: Born & Wolf Ch. 2; Böttcher CJF. Theory of Electric Polarisation. Elsevier 1973.

# Published bond polarizabilities (Å³)
# Source: Miller KJ. JACS 1990, 112, 8533; CRC Handbook
BOND_POLARIZABILITY_A3: Dict[str, float] = {
    "C-C": 0.53,
    "C=C": 1.65,
    "C≡C": 2.04,
    "C-H": 0.65,
    "C-O": 0.58,
    "C=O": 1.02,
    "C-N": 0.55,
    "C=N": 1.47,
    "C≡N": 1.94,
    "C-S": 1.32,
    "C-F": 0.44,
    "C-Cl": 1.50,
    "C-Br": 1.96,
    "O-H": 0.59,
    "N-H": 0.62,
    "S-H": 1.35,
    "Si-O": 0.72,
    "P=O": 1.20,
    "P-O": 0.68,
    "S=O": 1.10,
    "aromatic_C": 1.07,  # per aromatic C atom (delocalized)
}

# Published molecular polarizabilities (Å³)
# Source: CRC Handbook of Chemistry and Physics; Miller 1990
MOLECULAR_POLARIZABILITY_A3: Dict[str, float] = {
    "SiO2": 4.84,          # per formula unit
    "TiO2": 7.40,
    "ZnO": 4.50,
    "Al2O3": 8.30,
    "Si3N4": 11.2,
    "CaCO3": 7.20,
    "MgO": 3.30,
    "PMMA_monomer": 8.95,  # methyl methacrylate
    "styrene": 14.5,
    "ethylene": 4.25,
    "water": 1.45,
    "benzene": 10.0,
    "naphthalene": 17.5,
    "anthracene": 25.4,
    "dopamine": 15.0,      # estimated from structure
}


def lorentz_lorenz_n(polarizability_A3: float,
                       number_density_m3: float) -> float:
    """Compute refractive index from Lorentz-Lorenz equation.

    (n² - 1)/(n² + 2) = N·α / (3ε₀)

    where α must be in SI units (F·m²).

    Parameters
    ----------
    polarizability_A3 : float
        Molecular polarizability in ų (10⁻³⁰ m³).
    number_density_m3 : float
        Number density N (molecules/m³).

    Returns
    -------
    float
        Refractive index n.

    Physics tier: T2 (Lorentz 1880, Lorenz 1880).
    """
    # Convert α from ų to SI: 1 ų = 1.1127e-40 F·m²
    alpha_SI = polarizability_A3 * 1.1127e-40  # F·m²

    # Lorentz-Lorenz parameter
    LL = number_density_m3 * alpha_SI / (3.0 * EPS_0)

    if LL >= 1.0:
        # Unphysical — polarizability too high for this density
        return 3.0  # cap at n=3

    if LL <= 0:
        return 1.0

    # Solve: (n²-1)/(n²+2) = LL → n² = (1 + 2·LL)/(1 - LL)
    n_sq = (1.0 + 2.0 * LL) / (1.0 - LL)
    return math.sqrt(max(1.0, n_sq))


def predict_n_from_composition(molecular_polarizability_A3: float,
                                  molecular_weight: float,
                                  density_g_cm3: float) -> float:
    """Predict n for a material from its molecular properties.

    Computes number density from MW and density, then applies Lorentz-Lorenz.

    Parameters
    ----------
    molecular_polarizability_A3 : float
        Molecular/formula unit polarizability (ų).
    molecular_weight : float
        g/mol.
    density_g_cm3 : float
        Bulk density (g/cm³).

    Returns
    -------
    float
        Predicted refractive index.

    Physics tier: T2.
    """
    # Number density: N = ρ·N_A / MW  (molecules/m³)
    density_kg_m3 = density_g_cm3 * 1000.0
    N = density_kg_m3 * N_AVO / (molecular_weight * 1e-3)  # per m³

    return lorentz_lorenz_n(molecular_polarizability_A3, N)


def predict_n_from_bonds(bond_counts: Dict[str, int],
                           molecular_weight: float,
                           density_g_cm3: float) -> float:
    """Predict n from bond-additive polarizabilities.

    Sum bond polarizabilities → total molecular α → Lorentz-Lorenz.

    Parameters
    ----------
    bond_counts : dict
        {bond_type: count} e.g. {"C-C": 3, "C-H": 8, "C=O": 1}
    molecular_weight : float
        g/mol.
    density_g_cm3 : float
        Bulk density (g/cm³).

    Returns
    -------
    float
        Predicted refractive index.

    Physics tier: T2 (bond-additive model, Miller 1990).
    """
    alpha_total = 0.0
    for bond, count in bond_counts.items():
        alpha_bond = BOND_POLARIZABILITY_A3.get(bond, 0.65)  # default C-H
        alpha_total += alpha_bond * count

    return predict_n_from_composition(alpha_total, molecular_weight, density_g_cm3)


# Validation: check against known materials
def validate_lorentz_lorenz():
    """Validate Lorentz-Lorenz against known n values.

    Lorentz-Lorenz is accurate for molecular materials (organic polymers,
    liquids) where the static polarizability ≈ optical polarizability.

    For ionic crystals (TiO₂, ZnO), the published α values are static
    (DC) polarizabilities which include ionic displacement contributions
    absent at optical frequencies. These materials require optical-frequency
    α values (back-calculated from known n) for accurate LL prediction.

    The validation correctly shows: molecular materials → excellent,
    ionic crystals → known limitation documented here.
    """
    known = [
        # Molecular materials — LL works well
        # (name, α_A3, MW, ρ, n_published, category)
        ("water", 1.45, 18.015, 1.00, 1.33, "molecular"),
        ("PMMA", 8.95, 100.12, 1.18, 1.49, "molecular"),
        ("polystyrene", 14.5, 104.15, 1.05, 1.59, "molecular"),

        # Ionic crystals — LL overestimates (static α > optical α)
        # Included to document the known limitation, not as validation
        ("SiO2 (amorphous)", 4.84, 60.08, 2.20, 1.46, "ionic"),
        ("TiO2 (rutile)", 7.40, 79.87, 4.23, 2.61, "ionic"),
        ("ZnO", 4.50, 81.38, 5.61, 1.95, "ionic"),
    ]
    results = []
    for name, alpha, mw, rho, n_pub, category in known:
        n_pred = predict_n_from_composition(alpha, mw, rho)
        err = abs(n_pred - n_pub)
        results.append((name, n_pub, n_pred, err, category))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# F5. Effective Medium: Advanced Mixing Rules
# ═══════════════════════════════════════════════════════════════════════════
#
# Beyond Maxwell-Garnett (already in material_designer M3).
# Adding Looyenga and Hashin-Shtrikman spectral bounds.

def looyenga_ema(n_a: float, n_b: float, f_a: float) -> float:
    """Looyenga effective medium approximation.

    n_eff^(2/3) = f_a · n_a^(2/3) + (1-f_a) · n_b^(2/3)

    Empirically excellent for porous dielectrics and composites.
    No distinguished matrix — symmetric in A and B.

    Parameters
    ----------
    n_a, n_b : float
        Refractive indices of components.
    f_a : float
        Volume fraction of component A.

    Returns
    -------
    float
        Effective refractive index.

    Physics tier: T2 (Looyenga 1965).
    """
    f_b = 1.0 - f_a
    n_eff_23 = f_a * n_a ** (2.0/3.0) + f_b * n_b ** (2.0/3.0)
    return max(1.0, n_eff_23 ** 1.5)


def hashin_shtrikman_bounds(eps_a: float, eps_b: float,
                              f_a: float) -> Tuple[float, float]:
    """Hashin-Shtrikman bounds on effective dielectric constant.

    Tightest possible bounds for any isotropic two-phase composite.
    Lower bound: component with lower ε is the matrix.
    Upper bound: component with higher ε is the matrix.

    Parameters
    ----------
    eps_a, eps_b : float
        Dielectric constants (ε = n²).
    f_a : float
        Volume fraction of component A.

    Returns
    -------
    (eps_lower, eps_upper)

    Physics tier: T2 (Hashin & Shtrikman 1962, rigorous variational bounds).
    """
    f_b = 1.0 - f_a
    eps_lo, eps_hi = min(eps_a, eps_b), max(eps_a, eps_b)
    f_lo = f_a if eps_a <= eps_b else f_b
    f_hi = 1.0 - f_lo

    # Lower bound: low-ε phase is matrix
    if eps_lo > 0:
        HS_lower = eps_lo + f_hi / (1.0/(eps_hi - eps_lo) + f_lo/(3.0*eps_lo))
    else:
        HS_lower = 0.0

    # Upper bound: high-ε phase is matrix
    if eps_hi > 0:
        HS_upper = eps_hi + f_lo / (1.0/(eps_lo - eps_hi) + f_hi/(3.0*eps_hi))
    else:
        HS_upper = eps_hi

    return (max(1.0, HS_lower), max(1.0, HS_upper))


def hashin_shtrikman_n_bounds(n_a: float, n_b: float,
                                f_a: float) -> Tuple[float, float]:
    """Hashin-Shtrikman bounds on effective refractive index.

    Converts n → ε = n², applies HS bounds, converts back.
    """
    eps_lo, eps_hi = hashin_shtrikman_bounds(n_a**2, n_b**2, f_a)
    return (math.sqrt(max(1.0, eps_lo)), math.sqrt(max(1.0, eps_hi)))


def composite_shell_n(components: List[Tuple[str, float]],
                        method: str = "looyenga") -> float:
    """Compute effective n for a multi-component shell.

    Parameters
    ----------
    components : list of (material_name, volume_fraction)
        e.g. [("SiO2", 0.7), ("air", 0.3)]
    method : str
        "looyenga", "maxwell_garnett", or "hashin_shtrikman_avg".

    Returns
    -------
    float
        Effective refractive index at 550nm.
    """
    # Get n values
    ns = []
    fracs = []
    for mat, f in components:
        try:
            from optical.refractive_index import n_real
            n = n_real(mat, 550.0)
        except (ImportError, Exception):
            n_map = {"SiO2": 1.46, "TiO2_rutile": 2.61, "air": 1.0,
                      "water": 1.33, "polystyrene": 1.59, "PMMA": 1.49,
                      "ZnO": 1.95}
            n = n_map.get(mat, 1.50)
        ns.append(n)
        fracs.append(f)

    if len(ns) < 2:
        return ns[0] if ns else 1.50

    if method == "looyenga":
        # Multi-component Looyenga: n_eff^(2/3) = Σ f_i · n_i^(2/3)
        n_eff_23 = sum(f * n ** (2.0/3.0) for f, n in zip(fracs, ns))
        return max(1.0, n_eff_23 ** 1.5)

    elif method == "hashin_shtrikman_avg":
        # Sequential pairwise HS, average of bounds
        n_eff = ns[0]
        f_acc = fracs[0]
        for i in range(1, len(ns)):
            f_rel = fracs[i] / (f_acc + fracs[i])
            n_lo, n_hi = hashin_shtrikman_n_bounds(n_eff, ns[i], 1.0 - f_rel)
            n_eff = (n_lo + n_hi) / 2.0
            f_acc += fracs[i]
        return n_eff

    else:  # maxwell_garnett
        # Use first component as matrix
        from core.material_designer import maxwell_garnett
        n_matrix = ns[0]
        for i in range(1, len(ns)):
            n_matrix = math.sqrt(maxwell_garnett(n_matrix**2, ns[i]**2, fracs[i]))
        return n_matrix


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_lorentz_lorenz_validation():
    """Print validation of Lorentz-Lorenz against known materials."""
    results = validate_lorentz_lorenz()
    print(f"\n{'Material':<25} {'n_pub':>6} {'n_pred':>6} {'Error':>6} {'Category'}")
    print("-" * 60)
    for name, n_pub, n_pred, err, cat in results:
        flag = "✓" if err < 0.10 else "△ (static α)"
        print(f"{name:<25} {n_pub:6.3f} {n_pred:6.3f} {err:6.3f} {cat} {flag}")


if __name__ == "__main__":
    print("=" * 70)
    print("Refraction Physics — 5-Module Refractive Index Engine")
    print("=" * 70)

    # F1: Kramers-Kronig demo
    print("\n--- F1: Kramers-Kronig ---")
    n_kk, k_kk = chromophore_nk_spectrum("CuPc", n_baseline=1.60)
    idx_peak = np.argmax(k_kk)
    print(f"CuPc shell: peak k={k_kk[idx_peak]:.4f} at {_LAM[idx_peak]:.0f}nm")
    print(f"  Δn range: {(n_kk - 1.60).min():.4f} to {(n_kk - 1.60).max():.4f}")

    # F2: Fresnel demo
    print("\n--- F2: Fresnel ---")
    for angle in [0, 30, 60]:
        R = fresnel_reflectance(1.0, 1.46, angle)
        print(f"  Air→SiO₂ at {angle}°: R={R:.4f}")
    theta_B = brewster_angle(1.0, 1.46)
    print(f"  Brewster angle: {theta_B:.1f}°")

    # F3: Graded index
    print("\n--- F3: Graded Index ---")
    ar = anti_reflection_design(1.46, 1.0, 550.0)
    layers = discretize_graded_index(ar, n_sublayers=5)
    print(f"  AR coating for SiO₂: {ar.total_thickness_nm:.1f}nm, {len(layers)} sublayers")
    for n, t in layers:
        print(f"    n={n:.3f}, t={t:.1f}nm")

    # F4: Lorentz-Lorenz validation
    print("\n--- F4: Lorentz-Lorenz Validation ---")
    print_lorentz_lorenz_validation()

    # F5: Effective medium
    print("\n--- F5: Effective Medium ---")
    for f_air in [0.0, 0.1, 0.3, 0.5]:
        n_L = looyenga_ema(1.46, 1.0, 1.0 - f_air)
        n_lo, n_hi = hashin_shtrikman_n_bounds(1.46, 1.0, 1.0 - f_air)
        print(f"  SiO₂ + {f_air:.0%} air: Looyenga n={n_L:.3f}, HS=[{n_lo:.3f}, {n_hi:.3f}]")
