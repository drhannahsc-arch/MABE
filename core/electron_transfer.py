"""
core/electron_transfer.py — Sprint 25: Marcus Electron Transfer + Radiation Stability

Quantitative electron transfer rate prediction via Marcus theory.
Replaces qualitative "redox favorable/unfavorable" with actual rate
constants. Adds radiation stability scoring for nuclear applications.

Physics:
  k_ET = (2π/ħ) × |H_DA|² × (1/√(4πλkT)) × exp(-(ΔG° + λ)² / 4λkT)
  where:
    λ = reorganization energy (inner-sphere + outer-sphere)
    H_DA = electronic coupling between donor and acceptor
    ΔG° = driving force (from Nernst/redox potentials)
"""
from dataclasses import dataclass, field
import math


_K_BOLTZMANN = 1.381e-23   # J/K
_HBAR = 1.055e-34          # J·s
_EV_TO_KJ = 96.485         # kJ/mol per eV
_EV_TO_J = 1.602e-19       # J per eV


@dataclass
class MarcusResult:
    """Marcus theory electron transfer prediction."""
    dg_driving_kj: float            # ΔG° driving force
    lambda_total_kj: float          # Total reorganization energy
    lambda_inner_kj: float          # Inner-sphere (bond length changes)
    lambda_outer_kj: float          # Outer-sphere (solvent reorganization)
    h_da_ev: float                  # Electronic coupling
    k_et_s: float                   # Electron transfer rate constant (s⁻¹)
    regime: str                     # "normal", "activationless", "inverted"
    activation_energy_kj: float     # (ΔG° + λ)² / 4λ
    half_life_s: float              # ln(2) / k_ET
    is_adiabatic: bool              # H_DA > 0.025 eV
    notes: str = ""

@dataclass
class RadiationStability:
    """Radiation damage assessment for binder materials."""
    material_type: str
    dose_rate_gy_hr: float          # Expected dose rate
    critical_dose_gy: float         # Dose at which material fails
    operational_lifetime_hr: float  # critical_dose / dose_rate
    operational_lifetime_days: float
    degradation_mechanism: str
    stability_rating: str           # "excellent", "good", "moderate", "poor", "unsuitable"
    rad_resistant_alternatives: list
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# REORGANIZATION ENERGY DATABASE
# ═══════════════════════════════════════════════════════════════════════════

# Inner-sphere reorganization (bond length changes upon ET)
# λ_inner in eV, from crystallographic Δd values
_LAMBDA_INNER = {
    # Self-exchange reactions
    "Fe3+/Fe2+": 1.20,     # Large: d5→d6 changes bond lengths by ~0.14 Å
    "Co3+/Co2+": 1.80,     # Very large: LS d6→HS d7, massive geometry change
    "Ru3+/Ru2+": 0.40,     # Small: 4d orbitals, less distortion
    "Os3+/Os2+": 0.30,     # Very small: 5d
    "Cu2+/Cu+":  0.80,     # JT distortion change
    "Mn3+/Mn2+": 0.60,
    "Ni3+/Ni2+": 0.50,
    "Cr3+/Cr2+": 1.50,     # Large: inert d3 → labile d4
    # Reduction to metal
    "Au3+/Au0":  0.50,     # Surface deposition, moderate inner sphere
    "Au+/Au0":   0.30,
    "Ag+/Ag0":   0.35,
    "Hg2+/Hg0":  0.60,
    "Pt2+/Pt0":  0.45,
    "Pd2+/Pd0":  0.40,
    # Oxyanion reductions
    "Cr6+/Cr3+": 2.00,     # Massive: tetrahedral CrO4²⁻ → octahedral Cr3+
    "U6+/U4+":   0.80,     # UO2²⁺ → UO2(s)
    "Se4+/Se0":  1.20,     # SeO3²⁻ → Se(0) trigonal → elemental
    "As5+/As3+": 0.90,
}

# Outer-sphere reorganization from Marcus continuum model
# λ_outer = (Δe²/4πε₀)(1/2r_D + 1/2r_A - 1/d)(1/ε_∞ - 1/ε_s)
# Typical values 0.5-1.5 eV in water
_LAMBDA_OUTER_WATER = 1.0  # eV, typical for aqueous


# ═══════════════════════════════════════════════════════════════════════════
# ELECTRONIC COUPLING DATABASE
# ═══════════════════════════════════════════════════════════════════════════

# H_DA in eV — depends on donor-acceptor distance and bridge
_COUPLING = {
    "direct_contact":     0.10,    # Metal-ligand direct orbital overlap
    "thiol_bridge":       0.05,    # Through-bond via S
    "pyridine_bridge":    0.03,    # Through aromatic N
    "carboxylate_bridge": 0.02,    # Through COO⁻
    "water_bridge":       0.005,   # Through-space/water, very weak
    "protein_bridge":     0.001,   # Long-range through protein
    "dna_bridge":         0.0005,  # Through DNA backbone, very weak
}


# ═══════════════════════════════════════════════════════════════════════════
# MARCUS THEORY CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

def compute_marcus_rate(
    dg_driving_ev,
    lambda_inner_ev=None,
    lambda_outer_ev=None,
    h_da_ev=None,
    redox_pair="",
    bridge_type="direct_contact",
    temp_c=25.0,
):
    """Compute Marcus theory electron transfer rate.

    Args:
        dg_driving_ev: ΔG° in eV (negative = thermodynamically favorable)
        lambda_inner_ev: Inner-sphere reorganization (or looked up from pair)
        lambda_outer_ev: Outer-sphere reorganization (default: water)
        h_da_ev: Electronic coupling (or looked up from bridge type)
        redox_pair: e.g. "Fe3+/Fe2+" for database lookup
        bridge_type: e.g. "thiol_bridge" for coupling lookup
        temp_c: Temperature

    Returns:
        MarcusResult
    """
    T = temp_c + 273.15
    kT_ev = _K_BOLTZMANN * T / _EV_TO_J  # kT in eV
    kT_j = _K_BOLTZMANN * T

    # Resolve parameters
    if lambda_inner_ev is None:
        lambda_inner_ev = _LAMBDA_INNER.get(redox_pair, 0.70)
    if lambda_outer_ev is None:
        lambda_outer_ev = _LAMBDA_OUTER_WATER
    if h_da_ev is None:
        h_da_ev = _COUPLING.get(bridge_type, 0.01)

    lambda_total_ev = lambda_inner_ev + lambda_outer_ev

    # Activation energy: ΔG‡ = (ΔG° + λ)² / (4λ)
    dg_barrier_ev = (dg_driving_ev + lambda_total_ev)**2 / (4.0 * lambda_total_ev)

    # Determine regime
    ratio = abs(dg_driving_ev) / lambda_total_ev
    if ratio < 0.8:
        regime = "normal"
    elif ratio < 1.2:
        regime = "activationless"  # ΔG° ≈ -λ → barrier ≈ 0
    else:
        regime = "inverted"  # |ΔG°| > λ → rate decreases!

    # Marcus rate: k = (2π/ħ) × H_DA² / √(4πλkT) × exp(-ΔG‡/kT)
    prefactor = (2.0 * math.pi / _HBAR)
    h_da_j = h_da_ev * _EV_TO_J
    lambda_total_j = lambda_total_ev * _EV_TO_J
    dg_barrier_j = dg_barrier_ev * _EV_TO_J

    nuclear_factor = 1.0 / math.sqrt(4.0 * math.pi * lambda_total_j * kT_j)
    exponential = math.exp(-dg_barrier_j / kT_j) if dg_barrier_j / kT_j < 500 else 0.0

    k_et = prefactor * h_da_j**2 * nuclear_factor * exponential

    # Cap at physically meaningful range
    k_et = min(k_et, 1e13)  # Vibration frequency limit

    half_life = math.log(2) / k_et if k_et > 1e-30 else 1e30
    is_adiabatic = h_da_ev > 0.025

    notes_parts = []
    if regime == "inverted":
        notes_parts.append("Marcus inverted region: rate DECREASES despite larger driving force")
    if regime == "activationless":
        notes_parts.append("Near activationless: ΔG° ≈ -λ, maximum rate")
    if not is_adiabatic:
        notes_parts.append("Non-adiabatic: weak electronic coupling")
    if k_et < 1:
        notes_parts.append(f"Very slow ET: t½ = {half_life:.0f} s")
    elif k_et > 1e6:
        notes_parts.append("Fast ET: kinetically competent for capture")

    return MarcusResult(
        dg_driving_kj=round(dg_driving_ev * _EV_TO_KJ, 2),
        lambda_total_kj=round(lambda_total_ev * _EV_TO_KJ, 2),
        lambda_inner_kj=round(lambda_inner_ev * _EV_TO_KJ, 2),
        lambda_outer_kj=round(lambda_outer_ev * _EV_TO_KJ, 2),
        h_da_ev=h_da_ev,
        k_et_s=k_et,
        regime=regime,
        activation_energy_kj=round(dg_barrier_ev * _EV_TO_KJ, 2),
        half_life_s=half_life,
        is_adiabatic=is_adiabatic,
        notes="; ".join(notes_parts),
    )


def predict_reductive_capture_rate(metal_formula, reductant_type="thiol",
                                     ph=7.0, temp_c=25.0):
    """Predict rate of reductive metal capture.

    Combines Nernst driving force with Marcus theory to get actual rate.
    """
    # Standard reduction potentials (V vs SHE)
    _E0 = {
        "Au3+": 1.50, "Au+": 1.69, "Ag+": 0.80, "Hg2+": 0.85,
        "Cu2+": 0.34, "Pt2+": 1.18, "Pd2+": 0.95,
        "Fe3+": 0.77, "Cr6+": 1.33, "UO2_2+": 0.27,
    }
    # Reductant potentials (V vs SHE)
    _E_RED = {
        "thiol": 0.25,       # RS⁻/RSSR: ~0.25 V
        "ascorbate": 0.06,   # Dehydroascorbate/ascorbate
        "citrate": 0.00,     # Approximate
        "NaBH4": -0.48,      # Very strong reductant
        "Fe2+": 0.77,        # Fe3+/Fe2+
        "sulfide": -0.48,    # S/S²⁻
        "zero_valent_iron": -0.44,
    }

    e0_metal = _E0.get(metal_formula, 0.5)
    e0_red = _E_RED.get(reductant_type, 0.25)

    # Driving force: ΔG° = -nF(E_metal - E_reductant)
    # Assume n=1 for rate-determining step
    dg_ev = -(e0_metal - e0_red)  # In V ≈ eV for 1-electron

    # Get appropriate redox pair
    pair_map = {
        "Au3+": "Au3+/Au0", "Au+": "Au+/Au0", "Ag+": "Ag+/Ag0",
        "Hg2+": "Hg2+/Hg0", "Fe3+": "Fe3+/Fe2+", "Cu2+": "Cu2+/Cu+",
        "Cr6+": "Cr6+/Cr3+", "UO2_2+": "U6+/U4+",
        "Pt2+": "Pt2+/Pt0", "Pd2+": "Pd2+/Pd0",
    }
    redox_pair = pair_map.get(metal_formula, "")

    # Bridge type from reductant
    bridge_map = {
        "thiol": "thiol_bridge", "ascorbate": "carboxylate_bridge",
        "citrate": "carboxylate_bridge", "NaBH4": "direct_contact",
        "Fe2+": "water_bridge", "sulfide": "direct_contact",
        "zero_valent_iron": "direct_contact",
    }
    bridge = bridge_map.get(reductant_type, "water_bridge")

    return compute_marcus_rate(dg_ev, redox_pair=redox_pair,
                                bridge_type=bridge, temp_c=temp_c)


# ═══════════════════════════════════════════════════════════════════════════
# RADIATION STABILITY
# ═══════════════════════════════════════════════════════════════════════════

# Critical dose (Gy) at which material loses >50% function
_RADIATION_TOLERANCE = {
    # Inorganic — highly stable
    "zeolite":           {"dose_gy": 1e8,  "mechanism": "Framework stable; ion exchange capacity retained",
                          "rating": "excellent"},
    "mof":               {"dose_gy": 1e5,  "mechanism": "Organic linkers radiolyzed; framework collapses",
                          "rating": "moderate"},
    "mesoporous_silica":  {"dose_gy": 1e8,  "mechanism": "SiO2 framework radiation-hard",
                          "rating": "excellent"},
    "ldh":               {"dose_gy": 5e7,  "mechanism": "Hydroxide layers stable; interlayer anions may exchange",
                          "rating": "good"},
    "carbon_nanotube":    {"dose_gy": 1e7,  "mechanism": "sp2 carbon radiation-resistant",
                          "rating": "good"},
    # Organic — moderate
    "mip":               {"dose_gy": 1e5,  "mechanism": "Polymer chain scission; cavity distortion",
                          "rating": "moderate"},
    "cof":               {"dose_gy": 5e4,  "mechanism": "Organic framework radiolysis",
                          "rating": "poor"},
    "dendrimer":         {"dose_gy": 1e4,  "mechanism": "Branch cleavage; loss of terminal groups",
                          "rating": "poor"},
    # Biological — sensitive
    "dna_origami":       {"dose_gy": 1e2,  "mechanism": "Strand breaks, base oxidation, structural collapse",
                          "rating": "unsuitable"},
    "aptamer":           {"dose_gy": 1e2,  "mechanism": "Base damage, strand cleavage",
                          "rating": "unsuitable"},
    "peptide":           {"dose_gy": 1e3,  "mechanism": "Backbone cleavage, side chain oxidation",
                          "rating": "poor"},
    "coordination_cage": {"dose_gy": 1e6,  "mechanism": "Metal nodes stable; organic linkers vulnerable",
                          "rating": "good"},
}

# Alternatives for radiation environments
_RAD_ALTERNATIVES = {
    "unsuitable": ["zeolite", "mesoporous_silica", "LDH"],
    "poor": ["zeolite", "mesoporous_silica", "coordination_cage", "carbon_nanotube"],
    "moderate": ["zeolite", "mesoporous_silica"],
}


def assess_radiation_stability(material_type, dose_rate_gy_hr=0.0,
                                 is_nuclear_application=False):
    """Assess radiation stability of a binder material.

    Args:
        material_type: Scaffold/binder type
        dose_rate_gy_hr: Expected radiation dose rate
        is_nuclear_application: If True, applies conservative thresholds
    """
    # Normalize material type
    mat_key = material_type.lower().replace(" ", "_")
    for key in _RADIATION_TOLERANCE:
        if key in mat_key:
            mat_key = key
            break

    data = _RADIATION_TOLERANCE.get(mat_key)
    if data is None:
        return RadiationStability(
            material_type, dose_rate_gy_hr, 1e6, 1e6/max(0.01, dose_rate_gy_hr),
            1e6/max(0.01, dose_rate_gy_hr)/24,
            "Unknown", "unknown", [], "No radiation data for this material")

    critical = data["dose_gy"]
    if is_nuclear_application:
        critical *= 0.5  # Safety factor

    if dose_rate_gy_hr > 0:
        lifetime_hr = critical / dose_rate_gy_hr
    else:
        lifetime_hr = 1e12  # Effectively infinite

    lifetime_days = lifetime_hr / 24.0

    alternatives = _RAD_ALTERNATIVES.get(data["rating"], [])

    notes = ""
    if data["rating"] in ("unsuitable", "poor") and is_nuclear_application:
        notes = f"NOT RECOMMENDED for nuclear applications. Use: {', '.join(alternatives[:3])}"
    elif lifetime_days < 30 and dose_rate_gy_hr > 0:
        notes = f"WARNING: only {lifetime_days:.0f} days operational lifetime at {dose_rate_gy_hr} Gy/hr"

    return RadiationStability(
        material_type=material_type,
        dose_rate_gy_hr=dose_rate_gy_hr,
        critical_dose_gy=critical,
        operational_lifetime_hr=round(lifetime_hr, 1),
        operational_lifetime_days=round(lifetime_days, 1),
        degradation_mechanism=data["mechanism"],
        stability_rating=data["rating"],
        rad_resistant_alternatives=alternatives,
        notes=notes,
    )

