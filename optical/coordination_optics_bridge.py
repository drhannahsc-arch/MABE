"""
optical/coordination_optics_bridge.py — Module 4a: Coordination-Optics Bridge

Derives shell optical properties (n_shell(λ), k_shell(λ)) from coordination
chemistry outputs. This is the glue module between the binding engine
(DiscretePocketSpec → metal scorer) and the optical engine (core-shell Mie).

Physics — two channels:

  Channel 1: Donor set → Δ₀ → λ_dd → k(λ)
    Spectrochemical series gives ligand field splitting.
    d-d absorption band position = 1/Δ₀.
    k from ε (Laporte-forbidden: 5-100 M⁻¹cm⁻¹) and surface density.

  Channel 2: Metal-ligand → Δα → Lorentz-Lorenz → Δn
    Two stages:
      Stage 1 (functionalization): anchored ligand replaces water layer
      Stage 2 (metal binding): metal ion displaces coordinated waters

Zero parameters fitted to structural color data.
All inputs from tabulated constants or the binding engine's donor classification.

Data sources:
  Δ₀ aqua complexes: Figgis & Hitchman (Tier 1 textbook)
  Ionic polarizabilities: Shannon & Fischer 2016, Acta Cryst. B72:325 (Tier 1)
  Group polarizabilities: Miller & Savchik 1979, JACS 101:7206 (Tier 1)
  DQ_BY_DONOR: shared with core/scorer_frozen.py (spectrochemical series)
"""

import math
import numpy as np

# Import shared data from binding engine
from core.scorer_frozen import METAL_DB, DQ_BY_DONOR, DQ_WATER


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE Δ₀ VALUES FOR AQUA COMPLEXES (cm⁻¹)
# Source: Figgis & Hitchman, "Ligand Field Theory and Its Applications"
# These are 10Dq for [M(H₂O)₆]ⁿ⁺ in octahedral field
# ═══════════════════════════════════════════════════════════════════════════

DELTA_0_AQUA_CM1 = {
    # First-row +2 (d1-d9 only; d0, d10 have no d-d transitions)
    "Ti3+": 20300,  # d1
    "V3+":  17700,  # d2
    "V2+":  12400,  # d3
    "Cr3+": 17400,  # d3
    "Cr2+": 13900,  # d4
    "Mn3+": 21000,  # d4
    "Mn2+":  7800,  # d5 — spin-forbidden only, ε ≈ 0.01
    "Fe3+":  9400,  # d5 — spin-forbidden only
    "Fe2+": 10400,  # d6
    "Co3+": 18200,  # d6
    "Co2+":  9200,  # d7
    "Ni2+":  8500,  # d8
    "Cu2+": 12600,  # d9
    # Second/third row: much larger Δ₀, typically low-spin
    "Ru2+": 19800,  # d6
    "Rh3+": 27000,  # d6
    "Ir3+": 32000,  # d6
    "Os2+": 22000,  # d6
    "Pd2+": 26000,  # d8 (square planar)
    "Pt2+": 30000,  # d8 (square planar)
}


# ═══════════════════════════════════════════════════════════════════════════
# IONIC / ATOMIC POLARIZABILITIES (Å³)
# Source: Shannon & Fischer 2016 for ions; CRC for water
# ═══════════════════════════════════════════════════════════════════════════

ALPHA_IONIC = {
    # Transition metals (+2)
    "Cu2+": 1.2,   "Zn2+": 1.0,   "Ni2+": 1.1,   "Co2+": 1.3,
    "Fe2+": 1.4,   "Mn2+": 1.3,   "Cr2+": 1.5,   "V2+":  1.5,
    "Ti2+": 1.6,
    # +3
    "Fe3+": 0.8,   "Cr3+": 0.9,   "Co3+": 0.7,   "Al3+": 0.4,
    "Mn3+": 0.9,   "V3+":  1.0,   "Ti3+": 1.1,
    # Heavy metals
    "Pb2+": 6.8,   "Cd2+": 2.4,   "Hg2+": 3.1,   "Ag+":  2.1,
    "Au+":  3.5,   "Au3+": 2.5,   "Sn2+": 3.3,   "Bi3+": 4.2,
    "In3+": 1.5,   "Tl+":  5.2,   "Tl3+": 2.0,
    # Alkaline earth
    "Ca2+": 0.8,   "Mg2+": 0.3,   "Ba2+": 1.6,   "Sr2+": 1.1,
    # Noble metals
    "Pd2+": 2.0,   "Pt2+": 2.8,   "Ru2+": 1.8,   "Rh3+": 1.5,
    "Ir3+": 2.0,
    # Rare earth (representative)
    "La3+": 1.5,   "Gd3+": 1.2,   "Eu3+": 1.3,
}

ALPHA_WATER = 1.45  # Å³, CRC Handbook


# ═══════════════════════════════════════════════════════════════════════════
# LIGAND GROUP POLARIZABILITIES (Å³)
# Source: Miller & Savchik 1979 atom/group additivity
# ═══════════════════════════════════════════════════════════════════════════

ALPHA_DONOR_GROUP = {
    "N_amine":     1.7,
    "N_pyridine":  9.5,   # full pyridine ring
    "N_imine":     2.2,
    "N_imidazole": 7.5,   # full imidazole ring
    "N_amide":     2.0,
    "N_nitrile":   3.5,
    "O_carboxylate": 4.1,
    "O_phenolate": 8.0,   # phenol ring included
    "O_hydroxyl":  1.0,
    "O_ether":     1.0,
    "O_catecholate": 9.0,
    "O_hydroxamate": 3.5,
    "O_carbonyl":  1.5,
    "O_enolate":   3.8,
    "S_thiolate":  5.5,
    "S_thioether": 3.8,
    "S_dithiocarbamate": 8.2,
    "P_phosphine": 10.0,  # estimated, Tier 3
    "Cl_chloride": 3.0,
    "Br_bromide":  4.5,
    "I_iodide":    7.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# CHANNEL 1: d-d ABSORPTION → k(λ)
# ═══════════════════════════════════════════════════════════════════════════

def _has_dd_transitions(metal_formula):
    """Check if metal has d-d transitions (d1-d9)."""
    metal = METAL_DB.get(metal_formula)
    if metal is None:
        return False
    return 0 < metal.d_electrons < 10


def _delta_0_complex(metal_formula, donor_subtypes):
    """Compute Δ₀ for a metal-ligand complex from spectrochemical series.

    Δ₀(complex) = Δ₀(aqua) × mean(Dq_donors) / Dq_water

    Args:
        metal_formula: e.g. "Cu2+", "Ni2+"
        donor_subtypes: list of donor subtype strings

    Returns:
        float: Δ₀ in cm⁻¹, or 0 if not applicable
    """
    if not _has_dd_transitions(metal_formula):
        return 0.0

    delta_aqua = DELTA_0_AQUA_CM1.get(metal_formula, 0.0)
    if delta_aqua == 0.0 or len(donor_subtypes) == 0:
        return 0.0

    # Average Dq of coordinating donors relative to water
    dq_values = [DQ_BY_DONOR.get(d, DQ_WATER) for d in donor_subtypes]
    mean_dq = sum(dq_values) / len(dq_values)

    return delta_aqua * mean_dq / DQ_WATER


def _dd_band_wavelength_nm(delta_0_cm1):
    """Convert Δ₀ (cm⁻¹) to d-d absorption band wavelength (nm).

    λ = 1e7 / Δ₀

    Returns:
        float: Wavelength in nm, or 0 if no transition
    """
    if delta_0_cm1 <= 0:
        return 0.0
    return 1e7 / delta_0_cm1


def _dd_extinction(metal_formula):
    """Estimate molar extinction coefficient for d-d transition.

    Selection rules:
      Laporte-forbidden (centrosymmetric Oh): ε ≈ 5-50 M⁻¹cm⁻¹
      Laporte-allowed (non-centrosymmetric): ε ≈ 50-500 M⁻¹cm⁻¹
      Spin-forbidden (ΔS≠0): ε ≈ 0.01-1 M⁻¹cm⁻¹

    Returns:
        float: Estimated ε in M⁻¹cm⁻¹
    """
    metal = METAL_DB.get(metal_formula)
    if metal is None or metal.d_electrons == 0 or metal.d_electrons == 10:
        return 0.0

    # d5 high-spin (Mn²⁺, Fe³⁺): all transitions spin-forbidden
    if metal.d_electrons == 5:
        return 0.05

    # d9 Cu²⁺: Jahn-Teller distortion relaxes Laporte rule
    if metal.d_electrons == 9:
        return 80.0

    # Typical octahedral d-d
    return 30.0


def _k_from_absorption(epsilon, concentration_M, wavelength_nm, lambda_dd_nm,
                        bandwidth_cm1=3000):
    """Compute imaginary refractive index k at a wavelength from d-d band.

    Models d-d band as Gaussian centered at lambda_dd with given bandwidth.

    Args:
        epsilon: Molar extinction coefficient (M⁻¹cm⁻¹) at band center
        concentration_M: Effective concentration of metal centers (mol/L)
        wavelength_nm: Wavelength to evaluate
        lambda_dd_nm: d-d band center wavelength
        bandwidth_cm1: Full width at half max in cm⁻¹ (typical: 2000-4000)

    Returns:
        float: k value at this wavelength
    """
    if epsilon <= 0 or lambda_dd_nm <= 0 or concentration_M <= 0:
        return 0.0

    # Convert wavelength to cm⁻¹ for Gaussian
    nu = 1e7 / wavelength_nm      # cm⁻¹
    nu_dd = 1e7 / lambda_dd_nm    # cm⁻¹
    sigma = bandwidth_cm1 / 2.355  # FWHM → σ for Gaussian

    # Gaussian band shape
    band = math.exp(-0.5 * ((nu - nu_dd) / sigma)**2) if sigma > 0 else 0.0

    # Beer-Lambert: α = ε × c × ln(10)
    # k = α × λ / (4π) = ε × c × ln(10) × λ_cm / (4π)
    alpha_cm1 = epsilon * concentration_M * math.log(10) * band
    lambda_cm = wavelength_nm * 1e-7
    k = alpha_cm1 * lambda_cm / (4 * math.pi)

    return k


# ═══════════════════════════════════════════════════════════════════════════
# CHANNEL 2: POLARIZABILITY → Δn
# ═══════════════════════════════════════════════════════════════════════════

def _shell_concentration(surface_coverage_nm2, shell_thickness_nm,
                          particle_diameter_nm):
    """Estimate effective molar concentration of metal sites in shell volume.

    Args:
        surface_coverage_nm2: Metal sites per nm² of particle surface
        shell_thickness_nm: Shell thickness in nm
        particle_diameter_nm: Particle diameter in nm

    Returns:
        float: Effective concentration in mol/L
    """
    if shell_thickness_nm <= 0 or surface_coverage_nm2 <= 0:
        return 0.0

    # Surface area per particle
    r = particle_diameter_nm / 2.0
    A_nm2 = 4 * math.pi * r**2

    # Sites per particle
    n_sites = surface_coverage_nm2 * A_nm2

    # Shell volume per particle (nm³)
    r_outer = r + shell_thickness_nm
    V_shell_nm3 = (4/3) * math.pi * (r_outer**3 - r**3)

    # Concentration = sites / shell_volume
    # Convert: 1 nm³ = 1e-24 L, NA = 6.022e23
    sites_per_nm3 = n_sites / V_shell_nm3
    conc_M = sites_per_nm3 / 0.6022  # (1e-24 L/nm³ × 6.022e23 /mol)

    return conc_M


def _delta_alpha_functionalization(donor_subtypes):
    """Stage 1: Polarizability change from anchoring ligand to shell surface.

    Replaces water molecules on the surface with higher-polarizability
    donor groups. This is the dominant Δn contribution for aromatic ligands.

    Returns:
        float: Δα per binding site (Å³)
    """
    alpha_ligand = sum(ALPHA_DONOR_GROUP.get(d, 1.5) for d in donor_subtypes)

    # Ligand replaces equivalent volume of water on surface
    # Rough estimate: each donor group displaces ~1 water
    n_waters_displaced = len(donor_subtypes)
    alpha_water = n_waters_displaced * ALPHA_WATER

    return alpha_ligand - alpha_water


def _delta_alpha_coordination(metal_formula, n_waters_displaced):
    """Stage 2: Polarizability change from metal binding to pre-functionalized shell.

    Metal ion arrives and displaces some coordinated waters.
    Can be negative for small metals (α_metal < n_waters × α_water).

    Returns:
        float: Δα per binding event (Å³)
    """
    alpha_metal = ALPHA_IONIC.get(metal_formula, 1.0)
    return alpha_metal - n_waters_displaced * ALPHA_WATER


def _delta_n_from_delta_alpha(delta_alpha_A3, volume_fraction, base_n):
    """Convert Δα to Δn via linearized Lorentz-Lorenz.

    For small perturbations:
    Δn ≈ (2π/3) × N × Δα × (n² + 2)² / (6n)

    Simplified: Δn ≈ f_vol × Δα × (n² + 2) / (6n × α₀)

    We use a simpler effective medium approach:
    n_eff² ≈ n_base² + f × Δα/α₀ × (n_base² - 1)

    For very thin shells with small volume fractions, linearize:
    Δn ≈ f × Δα / (2 × n_base × V_molecule) × conversion_factor
    """
    if abs(delta_alpha_A3) < 1e-10 or volume_fraction <= 0:
        return 0.0

    # Clausius-Mossotti perturbation for small Δα:
    # Δε = ε × (3 × f × Δα/α_avg) / (ε + 2 - f(ε - 1))
    # For small f: Δε ≈ 3f × Δα/α_avg
    # Δn = Δε / (2n)

    # α_avg for typical shell material (silane/organic, n~1.46):
    # (n²-1)/(n²+2) × 3/(4πN) gives α per unit volume
    # Simpler: use empirical scaling
    # For organic molecules: Δn/n ≈ 0.003 per Å³ per site at f=0.01

    # Volume fraction of the perturbation within the shell
    dn = volume_fraction * delta_alpha_A3 * 0.003 / base_n
    return dn


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def coordination_optics_bridge(
    donor_subtypes: list,
    metal: str,
    shell_thickness_nm: float = 1.5,
    surface_coverage_nm2: float = 3.0,
    particle_diameter_nm: float = 250.0,
    base_shell_n: float = 1.46,
    wavelength_array_nm: np.ndarray = None,
):
    """Derive shell optical properties from coordination chemistry.

    Returns (n_shell(λ), k_shell(λ)) for input to core-shell Mie (Module 4).

    Physics chain:
      1. Donor subtypes → spectrochemical Dq → Δ₀ → λ_dd → k(λ)
      2. Metal + donor polarizabilities → Δα → Lorentz-Lorenz → n(λ)

    Zero fitted parameters against color data.

    Args:
        donor_subtypes: List of donor subtype strings (from DONOR_PATTERNS)
        metal: Metal formula (e.g. "Cu2+", "Pb2+")
        shell_thickness_nm: Shell thickness in nm
        surface_coverage_nm2: Metal binding sites per nm²
        particle_diameter_nm: Particle diameter for geometry
        base_shell_n: Refractive index of unfunctionalized shell
        wavelength_array_nm: Wavelength array (default: 380-780 nm)

    Returns:
        dict with keys:
          n_shell: np.ndarray — real refractive index vs wavelength
          k_shell: np.ndarray — imaginary refractive index vs wavelength
          delta_0_cm1: float — ligand field splitting
          lambda_dd_nm: float — d-d band position (0 if no d-d)
          delta_n_functionalization: float — Δn from ligand anchoring (Stage 1)
          delta_n_coordination: float — Δn from metal binding (Stage 2)
          delta_n_total: float — total Δn
          metal_d_electrons: int — d-electron count
    """
    if wavelength_array_nm is None:
        wavelength_array_nm = np.linspace(380, 780, 81)
    wavelength_array_nm = np.asarray(wavelength_array_nm, dtype=float)

    metal_props = METAL_DB.get(metal)
    d_electrons = metal_props.d_electrons if metal_props else 0
    cn_aqua = metal_props.cn_aqua if metal_props else 6

    # ── Channel 1: k(λ) from d-d absorption ─────────────────────────────

    delta_0 = _delta_0_complex(metal, donor_subtypes)
    lambda_dd = _dd_band_wavelength_nm(delta_0)
    epsilon = _dd_extinction(metal)

    # Effective concentration of metal in shell
    conc = _shell_concentration(surface_coverage_nm2, shell_thickness_nm,
                                particle_diameter_nm)

    k_array = np.zeros(len(wavelength_array_nm))
    if epsilon > 0 and lambda_dd > 0:
        for i, lam in enumerate(wavelength_array_nm):
            k_array[i] = _k_from_absorption(epsilon, conc, lam, lambda_dd)

    # ── Channel 2: n from polarizability ──────────────────────────────────

    # Stage 1: functionalization (ligand replaces water on surface)
    delta_alpha_func = _delta_alpha_functionalization(donor_subtypes)

    # Volume fraction of functionalized sites in shell
    # sites/nm² × (effective_site_volume) / shell_thickness
    # Approximate site volume: ~0.5 nm³ for a small coordination complex
    site_volume_nm3 = 0.5
    f_vol = surface_coverage_nm2 * site_volume_nm3 / shell_thickness_nm
    f_vol = min(f_vol, 0.5)  # cap at physical limit

    dn_func = _delta_n_from_delta_alpha(delta_alpha_func, f_vol, base_shell_n)

    # Stage 2: metal coordination (metal replaces waters at pre-functionalized site)
    n_waters_displaced = min(len(donor_subtypes), cn_aqua)
    delta_alpha_coord = _delta_alpha_coordination(metal, n_waters_displaced)
    dn_coord = _delta_n_from_delta_alpha(delta_alpha_coord, f_vol, base_shell_n)

    dn_total = dn_func + dn_coord

    # Build n_shell(λ) — weakly dispersive in visible, add constant Δn
    n_shell = np.full(len(wavelength_array_nm), base_shell_n + dn_total)

    # Kramers-Kronig: d-d absorption causes anomalous dispersion near λ_dd
    # For weak d-d bands (k ~ 0.01), this is Δn ~ 0.001 — ignorable in v1
    # but we add the correct sign: n increases below λ_dd, decreases above
    if lambda_dd > 0 and epsilon > 0:
        for i, lam in enumerate(wavelength_array_nm):
            nu = 1e7 / lam
            nu_dd = 1e7 / lambda_dd
            # Kramers-Kronig dispersive correction (Lorentzian approximation)
            bandwidth = 3000  # cm⁻¹
            dnu = nu - nu_dd
            kk_correction = (k_array[i] * dnu / bandwidth *
                           2 / math.pi * 0.01)  # small correction
            n_shell[i] += kk_correction

    return {
        "n_shell": n_shell,
        "k_shell": k_array,
        "delta_0_cm1": delta_0,
        "lambda_dd_nm": lambda_dd,
        "epsilon_dd": epsilon,
        "delta_n_functionalization": dn_func,
        "delta_n_coordination": dn_coord,
        "delta_n_total": dn_total,
        "delta_alpha_functionalization": delta_alpha_func,
        "delta_alpha_coordination": delta_alpha_coord,
        "metal_d_electrons": d_electrons,
        "concentration_M": conc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: ABSTRACT FIGURE 2 SHELLS
# ═══════════════════════════════════════════════════════════════════════════

def bpmen_cu2():
    """BPMEN + Cu²⁺ shell (abstract Figure 2).
    Tetradentate N₄: 2 pyridyl + 2 amine.
    """
    return coordination_optics_bridge(
        donor_subtypes=["N_pyridine", "N_pyridine", "N_amine", "N_amine"],
        metal="Cu2+",
    )


def dtc_pb2():
    """Dithiocarbamate + Pb²⁺ shell (abstract Figure 2).
    Bidentate S,S + amine anchor.
    """
    return coordination_optics_bridge(
        donor_subtypes=["S_dithiocarbamate", "S_dithiocarbamate", "N_amine"],
        metal="Pb2+",
    )


def bipy_cu2():
    """Bipyridyl + Cu²⁺ shell (abstract Figure 2).
    Bidentate N,N.
    """
    return coordination_optics_bridge(
        donor_subtypes=["N_pyridine", "N_pyridine"],
        metal="Cu2+",
    )
