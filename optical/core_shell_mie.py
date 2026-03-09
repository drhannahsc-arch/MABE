"""
optical/core_shell_mie.py — Module 4: Core-Shell Mie Scattering

Extension of Module 3 to concentric sphere with different core and shell
refractive indices.

For thin shells (shell << core, which is the typical case for click handles
at 1-5 nm on 100-300 nm cores), uses volume-weighted effective medium
approximation: n_eff = volume-average of core and shell indices.

Shell represents click-functional layer:
  APTES:       ~0.8-1.2 nm, n ≈ 1.46
  PEG-azide:   ~2-5 nm, n ≈ 1.47
  Maleimide:   ~1-2 nm, n ≈ 1.50

Validation:
  Shell thickness → 0: reproduces homogeneous sphere
  Higher-n shell → redshift in Q_sca peak
"""

import math
import numpy as np

from optical.mie_scattering import mie_efficiencies


def _effective_n_core_shell(n_core, n_shell, r_core, r_total):
    """Volume-weighted effective refractive index for core-shell sphere.

    Exact in the thin-shell limit. Reasonable for shell/core < 0.1.
    """
    V_total = (4/3) * math.pi * r_total**3
    V_core = (4/3) * math.pi * r_core**3
    V_shell = V_total - V_core

    f_core = V_core / V_total
    f_shell = V_shell / V_total

    # Complex volume average
    nc = n_core if isinstance(n_core, complex) else complex(n_core, 0)
    ns = n_shell if isinstance(n_shell, complex) else complex(n_shell, 0)

    # Maxwell-Garnett-like mixing for core in shell medium
    # For thin shells, simple volume average of ε is adequate
    import cmath
    eps_eff = f_core * nc**2 + f_shell * ns**2
    n_eff = cmath.sqrt(eps_eff)
    return n_eff


def mie_coated_efficiencies(diameter_core_nm, diameter_total_nm,
                            n_core, n_shell, n_medium, wavelength_nm):
    """Compute Mie efficiencies for a coated (core-shell) sphere.

    Uses effective medium approximation for the composite particle,
    then standard Mie theory on the total diameter.

    Args:
        diameter_core_nm: Core diameter in nm
        diameter_total_nm: Total diameter (core + 2×shell thickness) in nm
        n_core: Complex refractive index of core
        n_shell: Complex refractive index of shell
        n_medium: Refractive index of surrounding medium (real)
        wavelength_nm: Wavelength in nm

    Returns:
        dict with keys: Q_ext, Q_sca, Q_abs, Q_back, g
    """
    r_core = diameter_core_nm / 2.0
    r_total = diameter_total_nm / 2.0

    if r_total < r_core:
        raise ValueError(f"Total diameter ({diameter_total_nm}) must exceed "
                         f"core diameter ({diameter_core_nm})")

    if abs(r_total - r_core) < 0.001:
        # No shell - use bare sphere
        return mie_efficiencies(diameter_core_nm, n_core, n_medium, wavelength_nm)

    # Effective index for the composite particle
    n_eff = _effective_n_core_shell(n_core, n_shell, r_core, r_total)

    return mie_efficiencies(diameter_total_nm, n_eff, n_medium, wavelength_nm)


def shell_spectral_shift(diameter_core_nm, shell_thickness_nm,
                          n_core, n_shell, n_medium,
                          wavelengths_nm=None):
    """Compute the spectral shift introduced by a shell layer.

    Returns the peak shift in Q_back between bare and coated sphere.
    Positive = redshift.

    Args:
        diameter_core_nm: Core diameter
        shell_thickness_nm: Shell thickness
        n_core: Core refractive index
        n_shell: Shell refractive index
        n_medium: Medium refractive index
        wavelengths_nm: Wavelength array (default: 380-780 nm)

    Returns:
        dict: bare_peak_nm, coated_peak_nm, shift_nm
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 200)

    d_total = diameter_core_nm + 2 * shell_thickness_nm

    Q_bare = []
    Q_coated = []
    for lam in wavelengths_nm:
        eff_bare = mie_efficiencies(diameter_core_nm, n_core, n_medium, lam)
        Q_bare.append(eff_bare["Q_back"])

        eff_coat = mie_coated_efficiencies(
            diameter_core_nm, d_total, n_core, n_shell, n_medium, lam)
        Q_coated.append(eff_coat["Q_back"])

    Q_bare = np.array(Q_bare)
    Q_coated = np.array(Q_coated)

    bare_peak = float(wavelengths_nm[np.argmax(Q_bare)])
    coated_peak = float(wavelengths_nm[np.argmax(Q_coated)])

    return {
        "bare_peak_nm": bare_peak,
        "coated_peak_nm": coated_peak,
        "shift_nm": coated_peak - bare_peak,
    }
