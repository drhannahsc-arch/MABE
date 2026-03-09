"""
optical/structure_factor.py — Module 5: Percus-Yevick Structure Factor

Analytical solution for S(q) of a hard-sphere fluid at packing fraction φ.
Uses the Wertheim/Thiele closed-form for the direct correlation function c(r),
with numerically evaluated Fourier transform for robustness.

S(q) peak position → structural resonance wavelength
S(q) peak width → color bandwidth

Validation:
  S(q) peak at φ=0.55 at qD ≈ 7.0-7.5
  S(0) = (1-φ)⁴ / (1+2φ)² for compressibility
"""

import math
import numpy as np


def structure_factor_PY(q, diameter_nm, packing_fraction):
    """Compute Percus-Yevick hard-sphere structure factor at wavevector q.

    Uses Wertheim-Thiele closed-form c(r) with numerical FT.

    Args:
        q: Scattering wavevector in nm⁻¹ (scalar)
        diameter_nm: Hard sphere diameter in nm
        packing_fraction: Volume fraction φ (0 < φ < 0.64)

    Returns:
        float: S(q) value
    """
    phi = packing_fraction
    D = diameter_nm

    if phi <= 0 or phi >= 1:
        return 1.0

    k = q * D  # dimensionless wavevector

    if abs(k) < 1e-10:
        # S(0) = compressibility limit
        return (1 - phi)**4 / (1 + 2 * phi)**2

    # PY direct correlation function coefficients
    # c(r) = -(α + β(r/D) + γ(r/D)³) for r < D, 0 otherwise
    alpha = (1 + 2 * phi)**2 / (1 - phi)**4
    beta = -6 * phi * (1 + phi / 2)**2 / (1 - phi)**4
    gamma = phi * alpha / 2.0

    # Fourier transform via analytical integrals
    # c̃(q) = -4πD³ ∫₀¹ [α + βs + γs³] s² (sin(ks)/(ks)) ds
    # = -(4πD³/k) ∫₀¹ [α s + β s² + γ s⁴] sin(ks) ds
    #
    # Analytical integrals:
    # I1 = ∫₀¹ s sin(ks) ds = (sin k - k cos k) / k²
    # I2 = ∫₀¹ s² sin(ks) ds
    # I4 = ∫₀¹ s⁴ sin(ks) ds

    sk = math.sin(k)
    ck = math.cos(k)
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k

    # I1 = ∫₀¹ s sin(ks) ds
    I1 = (sk - k * ck) / k2

    # I2 = ∫₀¹ s² sin(ks) ds
    I2 = (2 * k * sk + (2 - k2) * ck - 2) / k3

    # I4 = ∫₀¹ s⁴ sin(ks) ds
    I4 = ((4 * k3 - 24 * k) * sk
          - (k4 - 12 * k2 + 24) * ck + 24) / k5

    # c̃(q) = -(4πD³/k) × (α I1 + β I2 + γ I4)
    c_tilde_over_vol = -(4 * math.pi / k) * (alpha * I1 + beta * I2 + gamma * I4)

    # Number density
    rho = 6 * phi / (math.pi * D**3)

    # ρ c̃(q) = ρ × D³ × c̃/D³ ... but c_tilde already has D³ factor
    # Actually: c̃(q) = -(4πD³/k)(α I1 + β I2 + γ I4)
    # So ρ c̃(q) = (6φ/(πD³)) × (-4πD³/k)(α I1 + β I2 + γ I4)
    #            = -24φ/k × (α I1 + β I2 + γ I4)

    rho_c_tilde = -24 * phi / k * (alpha * I1 + beta * I2 + gamma * I4)

    # S(q) = 1 / (1 - ρ c̃(q))
    denom = 1.0 - rho_c_tilde
    if abs(denom) < 1e-10:
        denom = 1e-10
    Sq = 1.0 / denom

    return max(Sq, 0.01)


def structure_factor_peak(diameter_nm, packing_fraction, q_min=None, q_max=None):
    """Find the S(q) peak position and height.

    Args:
        diameter_nm: Sphere diameter in nm
        packing_fraction: Volume fraction φ

    Returns:
        dict: q_peak (nm⁻¹), S_peak
    """
    D = diameter_nm
    # First peak is near qD ≈ 2π (for dense packing, shifts to ~7.0-7.5)
    if q_min is None:
        q_min = 3.0 / D
    if q_max is None:
        q_max = 15.0 / D

    q_arr = np.linspace(q_min, q_max, 500)
    S_arr = np.array([structure_factor_PY(q, D, packing_fraction) for q in q_arr])

    idx = np.argmax(S_arr)
    q_peak = float(q_arr[idx])
    S_peak = float(S_arr[idx])

    return {
        "q_peak": q_peak,
        "S_peak": S_peak,
    }
