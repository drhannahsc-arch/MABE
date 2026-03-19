"""
thermal/ir_properties.py — Infrared Optical Properties & Thermal Radiation

Extends the optical refractive_index.py to the mid-IR (2-25 μm).
In this regime, phonon absorption bands dominate: SiO₂ at 9.7 μm,
Si₃N₄ at 11 μm, PDMS at 9-12 μm. These are the key materials
for radiative cooling — they emit selectively in the atmospheric
transparency window (8-13 μm).

The isomorphism:
  Visible: n(λ) contrast + Mie/TMM → structural color
  IR:      n(λ) + k(λ) contrast + TMM → selective thermal emission
  Acoustic: Z contrast + ATMM → sound blocking

Same transfer matrix. Different wavelengths. Different physics payoff.

Sources:
  SiO₂:  Kitamura 2007, Appl. Opt. 46:8118 (Tier 1)
  Si₃N₄: Kischkat 2012, Appl. Opt. 51:6789 (Tier 2)
  PDMS:  Querry 1987 (Tier 2)
  Metals: Ordal 1985, Appl. Opt. 24:4493 (Tier 1 — Drude model)
  Atmospheric: MODTRAN / Berk 1999 (simplified model)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

H_PLANCK = 6.626e-34    # J·s
C_LIGHT = 2.998e8       # m/s
K_BOLTZ = 1.381e-23     # J/K
SIGMA_SB = 5.670e-8     # W/(m²·K⁴) Stefan-Boltzmann


# ═══════════════════════════════════════════════════════════════════════════
# PLANCK BLACKBODY SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════

def planck_spectral_radiance(wavelength_um: float, temperature_K: float) -> float:
    """Planck spectral radiance B(λ,T) in W/(m²·sr·μm).

    The thermal emission spectrum. Peak at λ_max = 2898/T μm (Wien's law).
    At 300K: peak at ~9.7 μm — right in the atmospheric window.
    """
    lam_m = wavelength_um * 1e-6
    if lam_m <= 0 or temperature_K <= 0:
        return 0.0
    exponent = H_PLANCK * C_LIGHT / (lam_m * K_BOLTZ * temperature_K)
    if exponent > 500:
        return 0.0
    denom = math.exp(exponent) - 1
    if denom <= 0:
        return 0.0
    B = (2 * H_PLANCK * C_LIGHT**2 / lam_m**5) / denom
    # Convert from W/(m²·sr·m) to W/(m²·sr·μm)
    return B * 1e-6


def planck_spectrum(wavelengths_um: np.ndarray, temperature_K: float) -> np.ndarray:
    """Vectorized Planck spectrum."""
    return np.array([planck_spectral_radiance(w, temperature_K) for w in wavelengths_um])


def wien_peak_um(temperature_K: float) -> float:
    """Wien's displacement law: peak emission wavelength in μm."""
    return 2898.0 / temperature_K


def total_radiated_power(temperature_K: float) -> float:
    """Total hemispherical radiated power (W/m²) from Stefan-Boltzmann."""
    return SIGMA_SB * temperature_K**4


# ═══════════════════════════════════════════════════════════════════════════
# ATMOSPHERIC TRANSMISSION WINDOW
# ═══════════════════════════════════════════════════════════════════════════

def atmospheric_transmission(wavelength_um: float) -> float:
    """Simplified atmospheric transmission τ(λ) for clear sky at sea level.

    The atmosphere is opaque to IR everywhere EXCEPT the 8-13 μm window.
    Objects that emit strongly in this window can radiate heat directly
    to outer space (~3K) even during the day — radiative cooling.

    Simplified model based on MODTRAN with standard atmosphere:
    - 0-7 μm: H₂O absorption, low transmission
    - 8-13 μm: atmospheric window, high transmission
    - 13-25 μm: CO₂ + H₂O absorption, low transmission
    - 9.5-10 μm: O₃ absorption dip within the window
    """
    lam = wavelength_um

    # Below 3 μm: solar region, treat separately
    if lam < 3.0:
        return 0.7  # approximate average visible/NIR

    # 3-5 μm: partial window (CO₂ absorption at 4.3 μm)
    if 3.0 <= lam < 5.0:
        if 4.0 <= lam <= 4.6:
            return 0.05  # CO₂ 4.3 μm band
        return 0.4

    # 5-8 μm: H₂O absorption
    if 5.0 <= lam < 8.0:
        return 0.1

    # 8-13 μm: atmospheric window
    if 8.0 <= lam <= 13.0:
        # O₃ dip at 9.6 μm
        if 9.3 <= lam <= 10.0:
            o3_depth = 0.3 * math.exp(-((lam - 9.6) / 0.3)**2)
            return 0.85 - o3_depth
        return 0.85

    # 13-25 μm: CO₂ + H₂O
    if 13.0 < lam <= 25.0:
        return 0.1

    return 0.05


def atmospheric_window_spectrum(wavelengths_um: np.ndarray) -> np.ndarray:
    """Vectorized atmospheric transmission."""
    return np.array([atmospheric_transmission(w) for w in wavelengths_um])


# ═══════════════════════════════════════════════════════════════════════════
# IR OPTICAL CONSTANTS DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IRMaterial:
    """IR optical properties for a material.

    For simple materials: constant n, k across IR.
    For phonon-active materials: Lorentzian resonances.
    """
    name: str
    n_background: float = 1.5       # real part away from resonances
    k_background: float = 0.0       # absorption away from resonances
    # Phonon resonances: (center_um, strength, width_um)
    phonon_resonances: list = None
    category: str = ""

    def __post_init__(self):
        if self.phonon_resonances is None:
            self.phonon_resonances = []

    def n_k(self, wavelength_um: float) -> Tuple[float, float]:
        """Compute n and k at a given IR wavelength."""
        n = self.n_background
        k = self.k_background

        for center, strength, width in self.phonon_resonances:
            # Lorentzian absorption: k peaks at center
            delta = (wavelength_um - center) / (width / 2)
            k_contrib = strength / (1 + delta**2)
            k += k_contrib
            # Kramers-Kronig: n perturbation from absorption
            n -= strength * delta / (1 + delta**2) * 0.5

        return max(0.5, n), max(0.0, k)

    def emissivity(self, wavelength_um: float, thickness_um: float = 1.0) -> float:
        """Approximate emissivity from Beer-Lambert: ε = 1 - exp(-4πk·d/λ)."""
        _, k = self.n_k(wavelength_um)
        if k <= 0 or thickness_um <= 0:
            return 0.0
        alpha = 4 * math.pi * k / wavelength_um  # absorption coefficient (1/μm)
        return 1.0 - math.exp(-alpha * thickness_um)


# ── Material database ─────────────────────────────────────────────────

IR_DB = {}

def _add_ir(name, n_bg, k_bg, phonons, category=""):
    IR_DB[name] = IRMaterial(name, n_bg, k_bg, phonons, category)

# Phonon-active dielectrics (key radiative cooling materials)
_add_ir("SiO2", 1.40, 0.0, [
    (9.7, 2.5, 1.5),     # Si-O-Si asymmetric stretch — THE radiative cooling band
    (12.5, 0.8, 1.0),    # Si-O-Si symmetric stretch
    (21.0, 0.5, 2.0),    # Si-O-Si bending
], "dielectric")

_add_ir("Si3N4", 2.00, 0.0, [
    (11.5, 1.8, 2.0),    # Si-N stretch
    (8.5, 0.3, 1.0),     # secondary mode
], "dielectric")

_add_ir("PDMS", 1.40, 0.0, [
    (9.5, 1.5, 1.5),     # Si-O-Si stretch
    (12.0, 1.0, 1.0),    # Si-CH₃ deformation
    (7.9, 0.8, 0.5),     # Si-C stretch
], "polymer")

_add_ir("PMMA", 1.48, 0.0, [
    (5.8, 1.2, 0.3),     # C=O stretch
    (8.4, 0.6, 0.5),     # C-O-C stretch
    (11.5, 0.3, 1.0),    # C-O stretch
], "polymer")

_add_ir("polyethylene", 1.52, 0.0, [
    (6.8, 0.3, 0.3),     # CH₂ deformation
    (13.8, 0.2, 0.5),    # CH₂ rocking
], "polymer")

# Metals (Drude — high reflectivity in IR)
_add_ir("aluminum", 1.0, 50.0, [], "metal")  # essentially a mirror
_add_ir("silver", 0.5, 80.0, [], "metal")
_add_ir("gold_ir", 1.0, 40.0, [], "metal")

# Simple dielectrics (weak/no phonon features in 8-13 μm)
_add_ir("ZnSe", 2.40, 0.0, [], "dielectric")   # IR window material
_add_ir("BaF2", 1.45, 0.0, [], "dielectric")
_add_ir("CaF2", 1.40, 0.0, [], "dielectric")
_add_ir("Ge", 4.00, 0.0, [], "dielectric")      # transparent 2-16 μm

# 2D materials
_add_ir("hBN_ir", 1.65, 0.0, [
    (7.3, 2.0, 0.5),     # out-of-plane phonon (Reststrahlen)
    (12.8, 1.5, 0.5),    # in-plane phonon
], "2D")

_add_ir("TiO2_ir", 2.20, 0.0, [
    (14.0, 1.0, 2.0),    # Ti-O phonon
], "dielectric")

# Air / vacuum
_add_ir("air_ir", 1.00, 0.0, [], "medium")
_add_ir("vacuum", 1.00, 0.0, [], "medium")


def get_ir_material(name: str) -> IRMaterial:
    """Look up IR material by name."""
    if name not in IR_DB:
        raise KeyError(f"Unknown IR material: {name}. Available: {sorted(IR_DB.keys())}")
    return IR_DB[name]


# ═══════════════════════════════════════════════════════════════════════════
# EMISSIVITY SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════

def emissivity_spectrum(
    material: str,
    thickness_um: float,
    wavelengths_um: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute emissivity spectrum ε(λ) for a material film.

    Returns (wavelengths_um, emissivity array).
    """
    if wavelengths_um is None:
        wavelengths_um = np.linspace(2, 25, 500)

    mat = get_ir_material(material)
    eps = np.array([mat.emissivity(w, thickness_um) for w in wavelengths_um])
    return wavelengths_um, eps


# ═══════════════════════════════════════════════════════════════════════════
# RADIATIVE COOLING POWER
# ═══════════════════════════════════════════════════════════════════════════

def radiative_cooling_power(
    emissivity_func,
    T_surface_K: float = 300.0,
    T_ambient_K: float = 300.0,
    T_sky_K: float = 270.0,
) -> dict:
    """
    Compute net radiative cooling power for an emissive surface.

    P_cool = P_rad(surface) - P_atm(absorbed) - P_solar(absorbed)

    P_rad = ∫ ε(λ) × B(λ, T_surface) × π dλ  [hemispherical]
    P_atm = ∫ ε(λ) × (1-τ_atm(λ)) × B(λ, T_ambient) × π dλ
    P_solar ≈ 0 (assume perfect solar reflector or nighttime)

    This is the radiative cooling design metric. Maximize P_cool.

    Args:
        emissivity_func: callable(wavelength_um) → emissivity (0-1)
        T_surface_K: surface temperature
        T_ambient_K: ambient air temperature
        T_sky_K: effective sky temperature (clear sky ~270K)

    Returns dict with P_rad, P_atm, P_cool, and per-band breakdown.
    """
    wavelengths = np.linspace(2, 25, 1000)
    dw = wavelengths[1] - wavelengths[0]

    P_rad = 0.0
    P_atm = 0.0
    P_window = 0.0  # contribution from atmospheric window only

    for w in wavelengths:
        eps = emissivity_func(w)
        tau = atmospheric_transmission(w)
        B_surface = planck_spectral_radiance(w, T_surface_K)
        B_ambient = planck_spectral_radiance(w, T_ambient_K)

        # Emitted power
        p_emit = eps * math.pi * B_surface * dw
        P_rad += p_emit

        # Absorbed atmospheric radiation
        p_atm = eps * (1 - tau) * math.pi * B_ambient * dw
        P_atm += p_atm

        # Window contribution
        if 8.0 <= w <= 13.0:
            P_window += p_emit

    P_cool = P_rad - P_atm

    return {
        'P_rad_W_m2': P_rad,
        'P_atm_absorbed_W_m2': P_atm,
        'P_cool_net_W_m2': P_cool,
        'P_window_W_m2': P_window,
        'window_fraction': P_window / P_rad if P_rad > 0 else 0,
        'T_surface_K': T_surface_K,
        'T_ambient_K': T_ambient_K,
        'blackbody_P_W_m2': total_radiated_power(T_surface_K),
    }


if __name__ == "__main__":
    print("═" * 70)
    print("THERMAL IR PROPERTIES & RADIATIVE COOLING")
    print("═" * 70)

    # Wien peak at various temperatures
    print("\nWien peak wavelength:")
    for T in [200, 273, 300, 400, 500, 1000, 5778]:
        print(f"  {T:5d} K → λ_max = {wien_peak_um(T):6.1f} μm"
              f"  ({'visible' if T > 3000 else 'mid-IR' if T > 200 else 'far-IR'})")

    # Material emissivity in atmospheric window
    print(f"\nEmissivity at 10 μm (atmospheric window center):")
    for mat_name in ["SiO2", "Si3N4", "PDMS", "PMMA", "polyethylene",
                      "aluminum", "hBN_ir"]:
        mat = get_ir_material(mat_name)
        for thick in [1, 10, 100]:
            eps = mat.emissivity(10.0, thick)
            print(f"  {mat_name:>15s} @ {thick:>4d} μm thick: ε = {eps:.3f}")

    # Radiative cooling power
    print(f"\nRadiative cooling power (T=300K, clear sky):")
    for mat_name, thick in [("SiO2", 50), ("PDMS", 100), ("Si3N4", 50),
                             ("PMMA", 100), ("polyethylene", 50)]:
        mat = get_ir_material(mat_name)
        result = radiative_cooling_power(
            emissivity_func=lambda w, m=mat, t=thick: m.emissivity(w, t),
        )
        print(f"  {mat_name:>15s} ({thick} μm): P_cool = {result['P_cool_net_W_m2']:.1f} W/m²"
              f"  (window fraction: {result['window_fraction']:.0%})")

    # Ideal blackbody in window only
    def ideal_window_emitter(w):
        return 1.0 if 8.0 <= w <= 13.0 else 0.0

    result_ideal = radiative_cooling_power(ideal_window_emitter)
    print(f"\n  IDEAL window emitter: P_cool = {result_ideal['P_cool_net_W_m2']:.1f} W/m²")

    result_bb = radiative_cooling_power(lambda w: 1.0)
    print(f"  Full blackbody:       P_cool = {result_bb['P_cool_net_W_m2']:.1f} W/m²")