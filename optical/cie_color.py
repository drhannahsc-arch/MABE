"""
optical/cie_color.py — Module 9: CIE 1931 Color Calculation

Convert reflectance spectrum R(λ) to human-perceived color using CIE 1931 XYZ
color matching functions. Standard illuminant D65.

Functions:
  spectrum_to_XYZ(R, wavelengths) → (X, Y, Z)
  XYZ_to_xyY(X, Y, Z) → (x, y, Y)
  XYZ_to_Lab(X, Y, Z) → (L*, a*, b*)
  XYZ_to_sRGB(X, Y, Z) → (r, g, b)  in [0,1]
  cie_delta_E(Lab1, Lab2) → float

Validation:
  White (flat R=1) → (x,y) = (0.3127, 0.3290) under D65 to ±0.001
"""

import math
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# CIE 1931 2° OBSERVER COLOR MATCHING FUNCTIONS + D65 ILLUMINANT
# Tabulated at 5 nm intervals, 380-780 nm (81 points)
# Source: CIE 15:2004 standard, publicly available
# ═══════════════════════════════════════════════════════════════════════════

_CIE_WAVELENGTHS = np.arange(380, 785, 5, dtype=float)  # 81 points

# x̄(λ) — CIE 1931 2° observer
_CIE_X = np.array([
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310,
    0.023190, 0.043510, 0.077630, 0.134380, 0.214770,
    0.283900, 0.328500, 0.348280, 0.348060, 0.336200,
    0.318700, 0.290800, 0.251100, 0.195360, 0.142100,
    0.095640, 0.058010, 0.032010, 0.014700, 0.004900,
    0.002400, 0.009300, 0.029100, 0.063270, 0.109600,
    0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
    0.512050, 0.594500, 0.678400, 0.762100, 0.842500,
    0.916300, 0.978600, 1.026300, 1.056700, 1.062200,
    1.045600, 1.002600, 0.938400, 0.854450, 0.751400,
    0.642400, 0.541900, 0.447900, 0.360800, 0.283500,
    0.218700, 0.164900, 0.121200, 0.087400, 0.063600,
    0.046770, 0.032900, 0.022700, 0.015840, 0.011359,
    0.008111, 0.005790, 0.004109, 0.002899, 0.002049,
    0.001440, 0.001000, 0.000690, 0.000476, 0.000332,
    0.000235, 0.000166, 0.000117, 0.000083, 0.000059,
    0.000042,
])

# ȳ(λ) — CIE 1931 2° observer
_CIE_Y = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396,
    0.000640, 0.001210, 0.002180, 0.004000, 0.007300,
    0.011600, 0.016840, 0.023000, 0.029800, 0.038000,
    0.048000, 0.060000, 0.073900, 0.090980, 0.112600,
    0.139020, 0.169300, 0.208020, 0.258600, 0.323000,
    0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
    0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
    1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000,
    0.566800, 0.503000, 0.441200, 0.381000, 0.321000,
    0.265000, 0.217000, 0.175000, 0.138200, 0.107000,
    0.081600, 0.061000, 0.044580, 0.032000, 0.023200,
    0.017000, 0.011920, 0.008210, 0.005723, 0.004102,
    0.002929, 0.002091, 0.001484, 0.001047, 0.000740,
    0.000520, 0.000361, 0.000249, 0.000172, 0.000120,
    0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015,
])

# z̄(λ) — CIE 1931 2° observer
_CIE_Z = np.array([
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850,
    0.110200, 0.207400, 0.371300, 0.645600, 1.039050,
    1.385600, 1.622960, 1.747060, 1.782600, 1.772110,
    1.744100, 1.669200, 1.528100, 1.287640, 1.041900,
    0.812950, 0.616200, 0.465180, 0.353300, 0.272000,
    0.212300, 0.158200, 0.111700, 0.078250, 0.057250,
    0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
    0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
    0.001650, 0.001400, 0.001100, 0.001000, 0.000800,
    0.000600, 0.000340, 0.000240, 0.000190, 0.000100,
    0.000050, 0.000030, 0.000020, 0.000010, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000,
])

# D65 illuminant SPD (relative, at 5 nm intervals 380-780)
_D65 = np.array([
    49.9755, 52.3118, 54.6482, 68.7015, 82.7549,
    87.1204, 91.4860, 92.4589, 93.4318, 90.0570,
    86.6823, 95.7736, 104.8650, 110.9360, 117.0080,
    117.4100, 117.8120, 116.3360, 114.8610, 115.3920,
    115.9230, 112.3670, 108.8110, 109.0820, 109.3540,
    108.5780, 107.8020, 106.2960, 104.7900, 106.2390,
    107.6890, 106.0470, 104.4050, 104.2250, 104.0460,
    102.0230, 100.0000, 98.1671, 96.3342, 96.0611,
    95.7880, 92.2368, 88.6856, 89.3459, 90.0062,
    89.8026, 89.5991, 88.6489, 87.6987, 85.4936,
    83.2886, 83.4939, 83.6992, 81.8630, 80.0268,
    80.1207, 80.2146, 81.2462, 82.2778, 80.2810,
    78.2842, 74.0027, 69.7213, 70.6652, 71.6091,
    72.9790, 74.3490, 67.9765, 61.6040, 65.7448,
    69.8856, 72.4863, 75.0870, 69.3398, 63.5927,
    55.0054, 46.4182, 56.6118, 66.8054, 65.0941,
    63.3828,
])


def _interp_to_grid(wavelengths_nm, values, target_wavelengths):
    """Interpolate spectral data to target wavelength grid."""
    return np.interp(target_wavelengths, wavelengths_nm, values)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def spectrum_to_XYZ(R, wavelengths_nm, illuminant="D65"):
    """Convert reflectance spectrum to CIE XYZ tristimulus values.

    Args:
        R: Reflectance spectrum array
        wavelengths_nm: Corresponding wavelength array in nm
        illuminant: Illuminant name (only "D65" supported)

    Returns:
        tuple: (X, Y, Z) tristimulus values
    """
    R = np.asarray(R, dtype=float)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    # Interpolate R to the CIE standard grid (5 nm, 380-780)
    R_std = _interp_to_grid(wavelengths_nm, R, _CIE_WAVELENGTHS)

    # Illuminant
    S = _D65

    # Normalization constant k = 1 / Σ(ȳ × S × Δλ)
    dlam = 5.0  # nm step
    k = 1.0 / np.sum(_CIE_Y * S * dlam)

    X = k * np.sum(_CIE_X * R_std * S * dlam)
    Y = k * np.sum(_CIE_Y * R_std * S * dlam)
    Z = k * np.sum(_CIE_Z * R_std * S * dlam)

    return (float(X), float(Y), float(Z))


def XYZ_to_xyY(X, Y, Z):
    """Convert CIE XYZ to chromaticity coordinates (x, y, Y).

    Args:
        X, Y, Z: Tristimulus values

    Returns:
        tuple: (x, y, Y_luminance)
    """
    total = X + Y + Z
    if total < 1e-10:
        return (0.3127, 0.3290, 0.0)  # D65 white point
    x = X / total
    y = Y / total
    return (float(x), float(y), float(Y))


def XYZ_to_Lab(X, Y, Z, Xn=0.95047, Yn=1.0, Zn=1.08883):
    """Convert CIE XYZ to CIELAB (L*, a*, b*).

    Reference white: D65 (Xn=0.95047, Yn=1.0, Zn=1.08883)

    Args:
        X, Y, Z: Tristimulus values
        Xn, Yn, Zn: Reference white tristimulus

    Returns:
        tuple: (L*, a*, b*)
    """
    def f(t):
        delta = 6.0 / 29.0
        if t > delta**3:
            return t**(1.0 / 3.0)
        else:
            return t / (3 * delta**2) + 4.0 / 29.0

    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return (float(L), float(a), float(b))


def XYZ_to_sRGB(X, Y, Z):
    """Convert CIE XYZ to sRGB (linear then gamma-corrected).

    Returns values in [0, 1] range, clipped.

    Args:
        X, Y, Z: Tristimulus values

    Returns:
        tuple: (R, G, B) in [0, 1]
    """
    # Linear sRGB from XYZ (D65)
    r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(c):
        if c <= 0.0031308:
            return max(0.0, 12.92 * c)
        else:
            return min(1.0, 1.055 * c**(1.0 / 2.4) - 0.055)

    return (gamma(r_lin), gamma(g_lin), gamma(b_lin))


def cie_delta_E(Lab1, Lab2):
    """Compute CIE76 color difference ΔE between two Lab colors.

    ΔE = √((L₁-L₂)² + (a₁-a₂)² + (b₁-b₂)²)

    Args:
        Lab1: (L*, a*, b*) tuple
        Lab2: (L*, a*, b*) tuple

    Returns:
        float: ΔE color difference
    """
    dL = Lab1[0] - Lab2[0]
    da = Lab1[1] - Lab2[1]
    db = Lab1[2] - Lab2[2]
    return math.sqrt(dL**2 + da**2 + db**2)
