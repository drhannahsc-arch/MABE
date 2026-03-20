"""
core/colloidal_acoustic.py — Colloidal Acoustic Absorption Model

Predicts acoustic absorption coefficient α(f), NRC, and STC of colloidal
assemblies from the same particle parameters that determine structural color
and thermal conductivity.

Physics models:
  - Johnson-Champoux-Allard (JCA): rigorous porous media model
  - Delany-Bazley: empirical model for fibrous/granular materials
  - Colloidal geometry → JCA parameters (porosity, flow resistivity,
    tortuosity, viscous/thermal characteristic lengths)
  - Acoustic transfer matrix for layered structures (film + air gap + backing)
  - NRC (Noise Reduction Coefficient) and STC (Sound Transmission Class)

Bridge to optical pipeline:
  Sphere diameter D and volume fraction φ determine:
  - Structural color (λ_peak ∝ D)
  - Thermal conductivity (κ_eff from Hasselman-Johnson, Phase 1)
  - Acoustic absorption (this module — α from JCA with D, φ-derived parameters)

Key references:
  - Johnson et al., J. Fluid Mech. 1987, 176, 379 (viscous drag in porous media)
  - Champoux & Allard, J. Appl. Phys. 1991, 70, 1975 (thermal effects)
  - Allard & Atalla, "Propagation of Sound in Porous Media", 2nd ed. 2009
  - Delany & Bazley, Appl. Acoust. 1970, 3, 105 (empirical model)
  - Kozeny-Carman: flow resistivity from sphere packing geometry

Data tier: Tier 2 (DOI per model equation).
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

PI = math.pi
RHO_AIR = 1.21          # kg/m³ at 20°C
C_AIR = 343.0            # m/s speed of sound at 20°C
Z_AIR = RHO_AIR * C_AIR  # 415 Pa·s/m — characteristic impedance of air
ETA_AIR = 1.81e-5        # Pa·s — dynamic viscosity of air at 20°C
GAMMA_AIR = 1.4          # ratio of specific heats for air
PR_AIR = 0.71            # Prandtl number for air
KAPPA_AIR = 0.026        # W/(m·K) thermal conductivity of air
CP_AIR = 1005.0          # J/(kg·K) specific heat at constant pressure


# ═══════════════════════════════════════════════════════════════════════════
# Colloidal geometry → porous media parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PorousMediaParams:
    """JCA model parameters derived from colloidal geometry."""
    porosity: float                    # ε = 1 - φ
    flow_resistivity_Pa_s_m2: float    # σ (Kozeny-Carman)
    tortuosity: float                  # α∞ (geometric path length ratio)
    viscous_char_length_m: float       # Λ (viscous effects length scale)
    thermal_char_length_m: float       # Λ' (thermal effects length scale)
    source: str = "colloidal_geometry"


def colloidal_to_jca(
    particle_diameter_nm: float,
    volume_fraction: float,
    packing_type: str = "random",
) -> PorousMediaParams:
    """Derive JCA porous media parameters from colloidal geometry.

    Uses Kozeny-Carman for flow resistivity and geometric relations
    for tortuosity and characteristic lengths of sphere packings.

    Args:
        particle_diameter_nm: sphere diameter in nm
        volume_fraction: φ (volume fraction of particles)
        packing_type: "random", "fcc", "bcc"

    Returns:
        PorousMediaParams for JCA model input.
    """
    D_m = particle_diameter_nm * 1e-9
    phi = volume_fraction
    eps = 1.0 - phi  # porosity

    if eps <= 0 or D_m <= 0:
        return PorousMediaParams(0.0, 1e12, 1.0, 1e-9, 1e-9)

    # ── Flow resistivity (Kozeny-Carman) ──
    # σ = 180 × η × (1-ε)² / (ε³ × D²)
    # Standard Kozeny-Carman for spheres. The constant 180 is Ergun's value.
    # Ref: Carman, Trans. Inst. Chem. Eng. 1937; Ergun, Chem. Eng. Prog. 1952
    sigma = 180.0 * ETA_AIR * (1.0 - eps)**2 / (eps**3 * D_m**2)

    # ── Tortuosity ──
    # For random sphere packing: α∞ ≈ 1 + (1-ε)/(2ε)
    # This is the Berryman (1980) relation for unconsolidated granular media.
    # Ref: Berryman, J. Math. Phys. 1980, 21, 2569
    if packing_type in ("fcc", "hcp"):
        # Ordered packing: slightly lower tortuosity
        alpha_inf = 1.0 + 0.4 * (1.0 - eps) / eps
    else:
        alpha_inf = 1.0 + 0.5 * (1.0 - eps) / eps

    # ── Viscous characteristic length (Λ) ──
    # For sphere packing: Λ ≈ D × ε / (3(1-ε))
    # This is the hydraulic radius approximation.
    # Ref: Johnson et al., J. Fluid Mech. 1987, 176, 379
    lambda_v = D_m * eps / (3.0 * (1.0 - eps))

    # ── Thermal characteristic length (Λ') ──
    # For spheres: Λ' ≈ 2Λ (thermal boundary layer thicker than viscous)
    # Ref: Champoux & Allard, J. Appl. Phys. 1991, 70, 1975
    lambda_t = 2.0 * lambda_v

    return PorousMediaParams(
        porosity=eps,
        flow_resistivity_Pa_s_m2=sigma,
        tortuosity=alpha_inf,
        viscous_char_length_m=lambda_v,
        thermal_char_length_m=lambda_t,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Johnson-Champoux-Allard (JCA) model
# ═══════════════════════════════════════════════════════════════════════════

def _jca_effective_density(freq_hz: float, params: PorousMediaParams) -> complex:
    """Effective dynamic density of air in porous medium (JCA).

    Ref: Johnson et al., J. Fluid Mech. 1987, 176, 379.

    ρ_eff(ω) = (α∞ × ρ₀ / ε) × [1 + σε/(j×ω×ρ₀×α∞) × √(1 + j×4×α∞²×η×ρ₀×ω / (σ²×Λ²×ε²))]
    """
    omega = 2.0 * PI * freq_hz
    if omega <= 0:
        return complex(RHO_AIR * params.tortuosity / params.porosity, 0)

    eps = params.porosity
    sigma = params.flow_resistivity_Pa_s_m2
    alpha_inf = params.tortuosity
    lam = params.viscous_char_length_m

    # Dimensionless frequency parameter
    X = 4.0 * alpha_inf**2 * ETA_AIR * RHO_AIR * omega / (sigma**2 * lam**2 * eps**2)

    # Johnson's dynamic tortuosity
    G_omega = cmath.sqrt(1.0 + 1j * X)
    rho_eff = (alpha_inf * RHO_AIR / eps) * (1.0 + (sigma * eps) / (1j * omega * RHO_AIR * alpha_inf) * G_omega)

    return rho_eff


def _jca_effective_bulk_modulus(freq_hz: float, params: PorousMediaParams) -> complex:
    """Effective dynamic bulk modulus of air in porous medium (JCA).

    Ref: Champoux & Allard, J. Appl. Phys. 1991, 70, 1975.

    K_eff(ω) = γP₀/ε / [γ - (γ-1) / (1 + 8η/(j×Λ'²×Pr×ρ₀×ω) × √(1 + j×ρ₀×ω×Pr×Λ'²/(16η)))]
    """
    omega = 2.0 * PI * freq_hz
    if omega <= 0:
        return complex(GAMMA_AIR * 101325.0 / params.porosity, 0)

    eps = params.porosity
    lam_prime = params.thermal_char_length_m
    P0 = 101325.0  # atmospheric pressure Pa

    # Thermal correction
    X_prime = RHO_AIR * omega * PR_AIR * lam_prime**2 / (16.0 * ETA_AIR)
    G_prime = cmath.sqrt(1.0 + 1j * X_prime)

    thermal_factor = 1.0 + (8.0 * ETA_AIR) / (1j * lam_prime**2 * PR_AIR * RHO_AIR * omega) * G_prime

    K_eff = (GAMMA_AIR * P0 / eps) / (GAMMA_AIR - (GAMMA_AIR - 1.0) / thermal_factor)

    return K_eff


def jca_surface_impedance(
    freq_hz: float,
    params: PorousMediaParams,
    thickness_m: float,
    backing: str = "rigid",
) -> complex:
    """Surface impedance of a porous layer on a backing.

    Z_s = -j × Z_c × cot(k_c × d)  (rigid backing)
    Z_s = -j × Z_c × tan(k_c × d)  (open backing — not standard)

    where Z_c = √(ρ_eff × K_eff) is the characteristic impedance in the porous medium,
    k_c = ω × √(ρ_eff / K_eff) is the complex wavenumber.
    """
    rho_eff = _jca_effective_density(freq_hz, params)
    K_eff = _jca_effective_bulk_modulus(freq_hz, params)

    Z_c = cmath.sqrt(rho_eff * K_eff)
    k_c = 2.0 * PI * freq_hz * cmath.sqrt(rho_eff / K_eff)

    kd = k_c * thickness_m

    # Overflow protection: when |Im(kd)| is large, cmath.sin/cos overflow.
    # Physically, large |Im(kd)| means the layer is acoustically thick —
    # sound is fully attenuated before reaching the backing.
    # In this limit, Z_s → Z_c (characteristic impedance of the porous medium).
    if abs(kd.imag) > 500:
        return Z_c

    try:
        if backing == "rigid":
            sin_kd = cmath.sin(kd)
            cos_kd = cmath.cos(kd)
            if abs(sin_kd) < 1e-15:
                return complex(1e12, 0)
            Z_s = -1j * Z_c * cos_kd / sin_kd
        else:
            cos_kd = cmath.cos(kd)
            sin_kd = cmath.sin(kd)
            if abs(cos_kd) < 1e-15:
                return complex(1e12, 0)
            Z_s = -1j * Z_c * sin_kd / cos_kd
    except (OverflowError, ValueError):
        # Fallback for any remaining overflow: thick-layer limit
        return Z_c

    return Z_s


def jca_absorption_coefficient(
    freq_hz: float,
    params: PorousMediaParams,
    thickness_m: float,
    backing: str = "rigid",
) -> float:
    """Normal-incidence absorption coefficient α at frequency f.

    α = 1 - |R|² where R = (Z_s - Z_air) / (Z_s + Z_air)
    """
    Z_s = jca_surface_impedance(freq_hz, params, thickness_m, backing)

    R = (Z_s - Z_AIR) / (Z_s + Z_AIR)
    alpha = 1.0 - abs(R)**2

    return max(0.0, min(1.0, alpha))


# ═══════════════════════════════════════════════════════════════════════════
# Delany-Bazley empirical model (simpler alternative)
# ═══════════════════════════════════════════════════════════════════════════

def delany_bazley_absorption(
    freq_hz: float,
    flow_resistivity_Pa_s_m2: float,
    thickness_m: float,
) -> float:
    """Delany-Bazley empirical absorption model.

    Simple empirical model for fibrous/granular absorbers.
    Less accurate than JCA but requires only flow resistivity.

    Ref: Delany & Bazley, Appl. Acoust. 1970, 3, 105.
    Valid for: 0.01 < ρ₀f/σ < 1.0
    """
    if freq_hz <= 0 or flow_resistivity_Pa_s_m2 <= 0:
        return 0.0

    X = RHO_AIR * freq_hz / flow_resistivity_Pa_s_m2

    # Characteristic impedance (Delany-Bazley)
    Z_c_real = RHO_AIR * C_AIR * (1.0 + 0.0571 * X**(-0.754))
    Z_c_imag = -RHO_AIR * C_AIR * 0.0870 * X**(-0.732)
    Z_c = complex(Z_c_real, Z_c_imag)

    # Propagation constant
    k_real = (2.0 * PI * freq_hz / C_AIR) * (1.0 + 0.0978 * X**(-0.700))
    k_imag = -(2.0 * PI * freq_hz / C_AIR) * 0.1890 * X**(-0.595)
    k_c = complex(k_real, k_imag)

    # Surface impedance (rigid backing)
    kd = k_c * thickness_m
    sin_kd = cmath.sin(kd)
    cos_kd = cmath.cos(kd)
    if abs(sin_kd) < 1e-15:
        return 0.0
    Z_s = -1j * Z_c * cos_kd / sin_kd

    # Absorption coefficient
    R = (Z_s - Z_AIR) / (Z_s + Z_AIR)
    alpha = 1.0 - abs(R)**2

    return max(0.0, min(1.0, alpha))


# ═══════════════════════════════════════════════════════════════════════════
# Layered structure: acoustic transfer matrix
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AcousticLayer:
    """One layer in a multi-layer acoustic structure."""
    name: str
    thickness_m: float
    layer_type: str            # "porous", "air_gap", "rigid"
    jca_params: Optional[PorousMediaParams] = None  # for porous layers


def _transfer_matrix_porous(freq_hz: float, layer: AcousticLayer) -> tuple:
    """2×2 transfer matrix for a porous layer."""
    params = layer.jca_params
    rho_eff = _jca_effective_density(freq_hz, params)
    K_eff = _jca_effective_bulk_modulus(freq_hz, params)

    Z_c = cmath.sqrt(rho_eff * K_eff)
    k_c = 2.0 * PI * freq_hz * cmath.sqrt(rho_eff / K_eff)
    kd = k_c * layer.thickness_m

    # Overflow protection for thick/high-resistivity layers
    if abs(kd.imag) > 500:
        # Thick-layer limit: transmission → 0, acts as impedance-matched termination
        # Return identity-like matrix that contributes Z_c to the surface impedance
        return (complex(1, 0), Z_c, 1.0 / Z_c, complex(1, 0))

    try:
        cos_kd = cmath.cos(kd)
        sin_kd = cmath.sin(kd)
    except (OverflowError, ValueError):
        return (complex(1, 0), Z_c, 1.0 / Z_c, complex(1, 0))

    T11 = cos_kd
    T12 = 1j * Z_c * sin_kd
    T21 = 1j * sin_kd / Z_c
    T22 = cos_kd

    return (T11, T12, T21, T22)


def _transfer_matrix_air(freq_hz: float, thickness_m: float) -> tuple:
    """2×2 transfer matrix for an air gap."""
    k0 = 2.0 * PI * freq_hz / C_AIR
    kd = k0 * thickness_m

    cos_kd = cmath.cos(kd)
    sin_kd = cmath.sin(kd)

    T11 = cos_kd
    T12 = 1j * Z_AIR * sin_kd
    T21 = 1j * sin_kd / Z_AIR
    T22 = cos_kd

    return (T11, T12, T21, T22)


def _multiply_matrices(A: tuple, B: tuple) -> tuple:
    """Multiply two 2×2 matrices stored as (T11, T12, T21, T22)."""
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    return (
        a11 * b11 + a12 * b21,
        a11 * b12 + a12 * b22,
        a21 * b11 + a22 * b21,
        a21 * b12 + a22 * b22,
    )


def layered_absorption(
    freq_hz: float,
    layers: list[AcousticLayer],
) -> float:
    """Absorption coefficient for a multi-layer structure on rigid backing.

    Layers ordered from front (sound-incident) to back (rigid wall).
    Uses acoustic transfer matrix method.
    """
    if not layers or freq_hz <= 0:
        return 0.0

    # Build total transfer matrix
    T_total = (complex(1, 0), complex(0, 0), complex(0, 0), complex(1, 0))  # identity

    for layer in layers:
        if layer.layer_type == "porous" and layer.jca_params:
            T_layer = _transfer_matrix_porous(freq_hz, layer)
        elif layer.layer_type == "air_gap":
            T_layer = _transfer_matrix_air(freq_hz, layer.thickness_m)
        else:
            continue  # rigid or unknown — skip

        T_total = _multiply_matrices(T_total, T_layer)

    # Surface impedance from transfer matrix (rigid backing: v=0 at back)
    # Z_s = T11 / T21
    T11, T12, T21, T22 = T_total
    if abs(T21) < 1e-20:
        return 0.0

    Z_s = T11 / T21

    # Absorption coefficient
    R = (Z_s - Z_AIR) / (Z_s + Z_AIR)
    alpha = 1.0 - abs(R)**2

    return max(0.0, min(1.0, alpha))


# ═══════════════════════════════════════════════════════════════════════════
# NRC and STC computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_nrc(alpha_spectrum: dict[float, float]) -> float:
    """Noise Reduction Coefficient: average α at 250, 500, 1000, 2000 Hz.

    NRC = (α_250 + α_500 + α_1000 + α_2000) / 4
    Rounded to nearest 0.05 per ASTM C423.
    """
    nrc_freqs = [250.0, 500.0, 1000.0, 2000.0]
    values = []
    for f in nrc_freqs:
        if f in alpha_spectrum:
            values.append(alpha_spectrum[f])
        else:
            # Interpolate from nearest frequencies
            freqs_sorted = sorted(alpha_spectrum.keys())
            if f <= freqs_sorted[0]:
                values.append(alpha_spectrum[freqs_sorted[0]])
            elif f >= freqs_sorted[-1]:
                values.append(alpha_spectrum[freqs_sorted[-1]])
            else:
                for i in range(len(freqs_sorted) - 1):
                    if freqs_sorted[i] <= f <= freqs_sorted[i + 1]:
                        f1, f2 = freqs_sorted[i], freqs_sorted[i + 1]
                        t = (f - f1) / (f2 - f1)
                        val = alpha_spectrum[f1] * (1 - t) + alpha_spectrum[f2] * t
                        values.append(val)
                        break

    if not values:
        return 0.0

    nrc = sum(values) / len(values)
    # Round to nearest 0.05
    return round(nrc * 20) / 20


def compute_stc_estimate(thickness_m: float, surface_density_kg_m2: float) -> int:
    """Estimate Sound Transmission Class from mass law.

    STC ≈ 20 × log₁₀(m × f) - 47  (mass law at 500 Hz reference)

    This is a rough estimate. Proper STC requires transmission loss
    measurement across 16 frequencies. Mass law gives the baseline.

    Ref: Sharp, Noise Control Eng. J. 1978, 11, 53.
    """
    if surface_density_kg_m2 <= 0:
        return 0

    f_ref = 500.0  # Hz
    TL_500 = 20.0 * math.log10(surface_density_kg_m2 * f_ref) - 47.0
    stc = max(0, int(round(TL_500)))
    return stc


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColloidalAcousticSpec:
    """Input: colloidal assembly for acoustic prediction.

    Same particle parameters as optical pipeline + colloidal_thermal.
    """
    particle_diameter_nm: float
    volume_fraction: float = 0.50
    packing_type: str = "random"

    # Film
    film_thickness_um: float = 50.0

    # Backing structure (optional)
    air_gap_mm: float = 0.0          # air gap behind film (mm)
    backing_layers: list[AcousticLayer] = field(default_factory=list)

    # Surface density (for STC estimate)
    particle_density_kg_m3: float = 2200.0  # SiO₂ default
    matrix_density_kg_m3: float = 1.21      # air default


@dataclass
class AcousticResult:
    """Output: acoustic absorption prediction."""
    # Spectrum
    alpha_spectrum: dict[float, float]   # {freq_Hz: absorption_coefficient}
    frequencies_hz: list[float]

    # Summary metrics
    nrc: float                           # Noise Reduction Coefficient
    alpha_250: float
    alpha_500: float
    alpha_1000: float
    alpha_2000: float

    # STC estimate
    stc_estimate: int = 0
    surface_density_kg_m2: float = 0.0

    # JCA parameters used
    jca_params: Optional[PorousMediaParams] = None

    # Configuration
    total_thickness_mm: float = 0.0
    has_air_gap: bool = False

    # Confidence
    model: str = "jca"
    confidence: float = 0.5
    confidence_notes: str = ""

    def summary(self) -> str:
        lines = [
            f"Colloidal Acoustic Absorption:",
            f"  NRC = {self.nrc:.2f}",
            f"  α(250) = {self.alpha_250:.3f}  α(500) = {self.alpha_500:.3f}",
            f"  α(1k) = {self.alpha_1000:.3f}  α(2k) = {self.alpha_2000:.3f}",
            f"  STC estimate: {self.stc_estimate}",
            f"  Total thickness: {self.total_thickness_mm:.1f} mm",
            f"  Model: {self.model}",
        ]
        if self.jca_params:
            lines.append(f"  Porosity: {self.jca_params.porosity:.2f}")
            lines.append(f"  Flow resistivity: {self.jca_params.flow_resistivity_Pa_s_m2:.0f} Pa·s/m²")
            lines.append(f"  Tortuosity: {self.jca_params.tortuosity:.2f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main prediction function
# ═══════════════════════════════════════════════════════════════════════════

_STANDARD_FREQS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
_DETAILED_FREQS = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]


def predict_acoustic(
    spec: ColloidalAcousticSpec,
    frequencies: Optional[list[float]] = None,
) -> AcousticResult:
    """Predict acoustic absorption of a colloidal assembly.

    Args:
        spec: ColloidalAcousticSpec with particle geometry and film spec
        frequencies: list of frequencies to evaluate (Hz). Defaults to standard set.

    Returns:
        AcousticResult with α spectrum, NRC, STC.
    """
    if frequencies is None:
        frequencies = list(_DETAILED_FREQS)

    # ── Derive JCA parameters from colloidal geometry ──
    jca_params = colloidal_to_jca(
        spec.particle_diameter_nm,
        spec.volume_fraction,
        spec.packing_type,
    )

    # ── Build layer stack ──
    film_thickness_m = spec.film_thickness_um * 1e-6
    layers = [
        AcousticLayer(
            name="colloidal_film",
            thickness_m=film_thickness_m,
            layer_type="porous",
            jca_params=jca_params,
        )
    ]

    if spec.air_gap_mm > 0:
        layers.append(AcousticLayer(
            name="air_gap",
            thickness_m=spec.air_gap_mm * 1e-3,
            layer_type="air_gap",
        ))

    # Add any additional backing layers
    layers.extend(spec.backing_layers)

    # ── Compute absorption spectrum ──
    alpha_spectrum = {}
    for f in frequencies:
        if len(layers) == 1 and not spec.backing_layers:
            # Single porous layer on rigid backing — use direct JCA
            alpha = jca_absorption_coefficient(f, jca_params, film_thickness_m, "rigid")
        else:
            # Multi-layer — use transfer matrix
            alpha = layered_absorption(f, layers)
        alpha_spectrum[f] = round(alpha, 4)

    # ── NRC ──
    nrc = compute_nrc(alpha_spectrum)

    # ── Key frequencies ──
    def _get_alpha(f):
        if f in alpha_spectrum:
            return alpha_spectrum[f]
        # Interpolate
        fs = sorted(alpha_spectrum.keys())
        for i in range(len(fs) - 1):
            if fs[i] <= f <= fs[i + 1]:
                t = (f - fs[i]) / (fs[i + 1] - fs[i])
                return alpha_spectrum[fs[i]] * (1 - t) + alpha_spectrum[fs[i + 1]] * t
        return 0.0

    # ── Surface density for STC ──
    total_t_m = film_thickness_m + spec.air_gap_mm * 1e-3
    surface_density = (spec.particle_density_kg_m3 * spec.volume_fraction +
                       spec.matrix_density_kg_m3 * (1 - spec.volume_fraction)) * film_thickness_m

    stc = compute_stc_estimate(total_t_m, surface_density)

    # ── Total thickness ──
    total_mm = spec.film_thickness_um / 1000.0 + spec.air_gap_mm
    for layer in spec.backing_layers:
        total_mm += layer.thickness_m * 1000.0

    # ── Confidence ──
    if spec.particle_diameter_nm > 100 and spec.particle_diameter_nm < 1000:
        confidence = 0.7
        conf_notes = "JCA model well-validated for granular media in this size range"
    elif spec.particle_diameter_nm < 100:
        confidence = 0.5
        conf_notes = "Very small particles: JCA continuum assumption may break down"
    else:
        confidence = 0.6
        conf_notes = "Large particles: well into granular regime"

    if spec.air_gap_mm > 0:
        conf_notes += "; air gap modeled via acoustic TMM"

    return AcousticResult(
        alpha_spectrum=alpha_spectrum,
        frequencies_hz=frequencies,
        nrc=nrc,
        alpha_250=_get_alpha(250.0),
        alpha_500=_get_alpha(500.0),
        alpha_1000=_get_alpha(1000.0),
        alpha_2000=_get_alpha(2000.0),
        stc_estimate=stc,
        surface_density_kg_m2=round(surface_density, 4),
        jca_params=jca_params,
        total_thickness_mm=round(total_mm, 2),
        has_air_gap=spec.air_gap_mm > 0,
        model="jca_tmm" if len(layers) > 1 else "jca",
        confidence=confidence,
        confidence_notes=conf_notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: predict from optical pipeline parameters
# ═══════════════════════════════════════════════════════════════════════════

def predict_from_optical(
    particle_diameter_nm: float,
    volume_fraction: float = 0.50,
    film_thickness_um: float = 50.0,
    air_gap_mm: float = 0.0,
    particle_density_kg_m3: float = 2200.0,
) -> AcousticResult:
    """Predict acoustic absorption from optical pipeline parameters.

    Same bridge function pattern as colloidal_thermal.predict_from_optical().
    """
    spec = ColloidalAcousticSpec(
        particle_diameter_nm=particle_diameter_nm,
        volume_fraction=volume_fraction,
        film_thickness_um=film_thickness_um,
        air_gap_mm=air_gap_mm,
        particle_density_kg_m3=particle_density_kg_m3,
    )
    return predict_acoustic(spec)


def predict_with_backing(
    particle_diameter_nm: float,
    volume_fraction: float = 0.50,
    film_thickness_um: float = 50.0,
    backing_thickness_mm: float = 50.0,
    backing_flow_resistivity: float = 10000.0,
    air_gap_mm: float = 25.0,
) -> AcousticResult:
    """Predict for colloidal film + fibrous backing + air gap.

    Typical building panel: colored colloidal surface + glass wool + air gap.
    """
    # Backing JCA params (fibrous absorber approximation)
    backing_params = PorousMediaParams(
        porosity=0.95,
        flow_resistivity_Pa_s_m2=backing_flow_resistivity,
        tortuosity=1.06,
        viscous_char_length_m=100e-6,  # 100 µm typical for glass wool
        thermal_char_length_m=200e-6,
        source="fibrous_absorber_standard",
    )

    backing_layer = AcousticLayer(
        name="fibrous_backing",
        thickness_m=backing_thickness_mm * 1e-3,
        layer_type="porous",
        jca_params=backing_params,
    )

    spec = ColloidalAcousticSpec(
        particle_diameter_nm=particle_diameter_nm,
        volume_fraction=volume_fraction,
        film_thickness_um=film_thickness_um,
        air_gap_mm=air_gap_mm,
        particle_density_kg_m3=2200.0,
        backing_layers=[backing_layer],
    )
    return predict_acoustic(spec)
