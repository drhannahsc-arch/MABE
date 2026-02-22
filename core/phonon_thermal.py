"""
core/phonon_thermal.py — Sprint 29c: Phonon-Mediated Thermal Ejection

Solid-state binders (zeolites, MOFs, MIPs) have binding sites that
vibrate. At high temperatures, phonon population increases and ions
can be shaken out of binding sites. This is distinct from solution-phase
Gibbs-Helmholtz — it's a lattice dynamics effect.

Physics:
  Debye model: <u²> = (3ħ²T)/(mk_BΘ_D²) × [Φ(Θ_D/T)/T + Θ_D/4T]
  Lindemann criterion: melting/ejection when <u²>^(1/2) > 0.1 × d_nn
  Phonon occupation: n(ω) = 1/(exp(ħω/kT) - 1)
  Thermal ejection rate: k_ej = ν₀ × exp(-E_bind / kT) × (1 + n_phonon)
"""
from dataclasses import dataclass
import math


@dataclass
class PhononThermalProfile:
    """Phonon-mediated thermal stability of binding site."""
    material_type: str
    debye_temp_K: float             # Θ_D: higher = stiffer lattice
    mean_displacement_A: float      # <u²>^(1/2) at operating temp
    lindemann_ratio: float          # <u²>^(1/2) / d_nn (>0.1 = unstable)
    thermal_ejection_rate_s: float  # Rate of ion ejection at operating T
    thermal_stability_class: str    # "stable", "marginal", "unstable"
    max_operating_temp_C: float     # T above which binding degrades >50%
    activation_energy_kj: float     # E_a for thermal desorption
    phonon_enhancement_factor: float  # How much phonons boost ejection vs Arrhenius
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# DEBYE TEMPERATURE DATABASE (K)
# Higher Θ_D = stiffer lattice = more thermally stable binding
# ═══════════════════════════════════════════════════════════════════════════

_DEBYE_TEMPS = {
    "zeolite":          450,   # Silicate framework: stiff
    "zeolite_Y":        430,
    "zeolite_ZSM5":     470,
    "mesoporous_silica": 500,  # Amorphous SiO2: very stiff
    "MOF":              150,   # Organic linkers: flexible
    "MOF_UiO66":        200,   # Zr-MOF: stiffer than average
    "MOF_MIL101":       120,   # Cr-MOF: flexible
    "MIP":              100,   # Polymer: soft
    "COF":              130,   # Organic framework: moderate
    "LDH":              350,   # Layered hydroxide: moderately stiff
    "carbon_nanotube":  1000,  # sp2 carbon: extremely stiff
    "coordination_cage": 80,   # Molecular: very soft
    "dendrimer":        60,    # Molecular: softest
    "dna_origami":      50,    # Soft in solution (not truly solid-state)
}

_K_BOLTZMANN = 1.381e-23
_HBAR = 1.055e-34


def compute_mean_displacement(debye_temp_K, ion_mass_amu, temp_K):
    """Mean-square displacement from Debye model.

    <u²> = (3ħ²T) / (m × k_B × Θ_D²) for T >> Θ_D (classical limit)
    For T << Θ_D (quantum): <u²> = (3ħ²)/(4m×k_B×Θ_D) (zero-point motion)
    """
    m_kg = ion_mass_amu * 1.661e-27  # amu to kg

    if debye_temp_K <= 0 or m_kg <= 0:
        return 0.0

    # High-T limit (classical): <u²> = 3kT / (mω_D²) where ω_D = kΘ_D/ħ
    omega_d = _K_BOLTZMANN * debye_temp_K / _HBAR
    u2_classical = 3 * _K_BOLTZMANN * temp_K / (m_kg * omega_d**2)

    # Zero-point motion
    u2_zp = 3 * _HBAR / (2 * m_kg * omega_d)

    # Interpolate
    u2 = u2_classical + u2_zp

    return math.sqrt(u2) * 1e10  # Convert m to Å


def compute_thermal_ejection(binding_energy_kj, debye_temp_K, temp_K,
                               attempt_frequency_hz=1e12):
    """Compute rate of thermal ejection from binding site.

    k_ej = ν₀ × exp(-E_bind / kT) × (1 + n̄)
    where n̄ is the mean phonon occupation number at the relevant frequency.

    The phonon enhancement factor (1 + n̄) accounts for how lattice
    vibrations assist the ejection — this is what distinguishes solid-state
    from solution-phase temperature dependence.
    """
    if binding_energy_kj <= 0 or temp_K <= 0:
        return 0.0, 1.0

    E_j = binding_energy_kj * 1000 / 6.022e23  # kJ/mol → J/molecule
    kT = _K_BOLTZMANN * temp_K

    # Arrhenius part
    arrhenius = attempt_frequency_hz * math.exp(-E_j / kT) if E_j / kT < 500 else 0.0

    # Phonon enhancement: n̄ = 1/(exp(ħω/kT) - 1) at Debye frequency
    omega_d = _K_BOLTZMANN * debye_temp_K / _HBAR
    x = _HBAR * omega_d / kT
    if x > 500:
        n_phonon = 0.0
    elif x < 0.01:
        n_phonon = kT / (_HBAR * omega_d)  # Classical limit
    else:
        n_phonon = 1.0 / (math.exp(x) - 1.0)

    enhancement = 1.0 + n_phonon
    k_ej = arrhenius * enhancement

    return k_ej, round(enhancement, 3)


def analyze_phonon_stability(material_type, binding_energy_kj=50.0,
                               ion_mass_amu=60.0, operating_temp_C=25.0,
                               nearest_neighbor_A=2.5):
    """Full phonon-mediated stability analysis."""
    key = material_type.lower().replace(" ", "_")
    debye = None
    for k in _DEBYE_TEMPS:
        if k in key or key in k:
            debye = _DEBYE_TEMPS[k]
            break
    if debye is None:
        debye = 200  # Default

    temp_K = operating_temp_C + 273.15

    u_rms = compute_mean_displacement(debye, ion_mass_amu, temp_K)
    lindemann = u_rms / nearest_neighbor_A if nearest_neighbor_A > 0 else 0

    k_ej, enhancement = compute_thermal_ejection(binding_energy_kj, debye, temp_K)

    # Stability classification
    if lindemann < 0.05 and k_ej < 1e-4:
        stability = "stable"
    elif lindemann < 0.10 and k_ej < 1.0:
        stability = "marginal"
    else:
        stability = "unstable"

    # Find max operating temperature (where k_ej crosses 1e-3 s⁻¹)
    max_temp = operating_temp_C
    for t_test in range(int(operating_temp_C) + 10, 500, 10):
        k_test, _ = compute_thermal_ejection(binding_energy_kj, debye, t_test + 273.15)
        if k_test > 1e-3:
            max_temp = t_test - 10
            break
    else:
        max_temp = 500  # Stable up to at least 500°C

    notes_parts = []
    if enhancement > 2.0:
        notes_parts.append(f"Strong phonon enhancement ({enhancement:.1f}×): "
                           f"lattice vibrations significantly assist ejection")
    if stability == "unstable":
        notes_parts.append(f"Binding unstable at {operating_temp_C}°C. "
                           f"Use stiffer scaffold (Θ_D > {debye} K)")
    if debye < 100:
        notes_parts.append("Very soft lattice — consider zeolite or silica alternative")

    return PhononThermalProfile(
        material_type=material_type,
        debye_temp_K=debye,
        mean_displacement_A=round(u_rms, 4),
        lindemann_ratio=round(lindemann, 4),
        thermal_ejection_rate_s=k_ej,
        thermal_stability_class=stability,
        max_operating_temp_C=max_temp,
        activation_energy_kj=binding_energy_kj,
        phonon_enhancement_factor=enhancement,
        notes="; ".join(notes_parts),
    )

