"""
core/mass_transport.py — Sprint 24: Mass Transport + Diffusion Coupling

Replaces heuristic P_enter with physics-based diffusion modeling.
Stokes-Einstein for bulk, hindered diffusion in pores, Thiele modulus
for diffusion vs reaction limitation, and time-to-capture prediction.

Physics:
  D_bulk = kT / (6πηr_h)                           (Stokes-Einstein)
  D_pore = D_bulk × (1-λ)² × (1 - 2.104λ + ...)   (Renkin hindered)
  φ = L√(k_rxn/D_eff)                              (Thiele modulus)
  t_90 = capacity / (flux × area × 0.9)             (Time to 90% loading)
"""
from dataclasses import dataclass
import math


@dataclass
class DiffusionProfile:
    """Diffusion characterization of a target in a scaffold."""
    d_bulk_m2_s: float              # Stokes-Einstein bulk diffusion
    d_pore_m2_s: float              # Hindered pore diffusion
    hindrance_factor: float         # D_pore / D_bulk (0-1)
    lambda_ratio: float             # r_ion / r_pore (0 = unhindered, >1 = excluded)
    transport_regime: str           # "unhindered", "hindered", "severely_hindered", "excluded"
    thiele_modulus: float           # φ: <1 reaction-limited, >1 diffusion-limited
    effectiveness_factor: float     # η: fraction of scaffold actually used
    rate_limiting_step: str         # "reaction", "pore_diffusion", "external_mass_transfer"
    notes: str = ""

@dataclass
class CaptureKinetics:
    """Time-dependent capture prediction."""
    time_to_50pct_s: float          # Time to 50% loading
    time_to_90pct_s: float          # Time to 90% loading
    time_to_99pct_s: float          # Time to 99% loading
    initial_flux_mol_m2_s: float    # Initial capture rate per unit area
    equilibrium_loading_pct: float  # Predicted final loading
    breakthrough_time_s: float      # For column mode: when target appears in effluent
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

_K_BOLTZMANN = 1.381e-23   # J/K
_VISCOSITY_WATER_25C = 8.9e-4  # Pa·s
_AVOGADRO = 6.022e23


# ═══════════════════════════════════════════════════════════════════════════
# DIFFUSION COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════

def stokes_einstein(hydrated_radius_nm, temp_c=25.0, viscosity=None):
    """Bulk diffusion coefficient via Stokes-Einstein.

    D = kT / (6πηr_h)

    Returns D in m²/s. Typical values: 0.5-2 × 10⁻⁹ m²/s for metal ions.
    """
    r_m = hydrated_radius_nm * 1e-9
    T = temp_c + 273.15
    eta = viscosity if viscosity else _VISCOSITY_WATER_25C * (293.15 / T)  # Approx T correction

    D = _K_BOLTZMANN * T / (6.0 * math.pi * eta * r_m)
    return D


def hindered_diffusion(d_bulk, ion_radius_nm, pore_radius_nm):
    """Hindered diffusion in a cylindrical pore (Renkin equation).

    D_pore = D_bulk × (1-λ)² × (1 - 2.104λ + 2.089λ³ - 0.948λ⁵)
    where λ = r_ion / r_pore

    Returns (D_pore, hindrance_factor, lambda_ratio, regime)
    """
    if pore_radius_nm <= 0:
        return d_bulk, 1.0, 0.0, "unhindered"

    lam = ion_radius_nm / pore_radius_nm

    if lam >= 1.0:
        return 0.0, 0.0, lam, "excluded"

    if lam > 0.95:
        return d_bulk * 1e-6, 1e-6, lam, "excluded"

    # Renkin equation
    steric = (1.0 - lam)**2
    hydro = 1.0 - 2.104 * lam + 2.089 * lam**3 - 0.948 * lam**5
    hindrance = steric * hydro
    hindrance = max(0.0, min(1.0, hindrance))

    d_pore = d_bulk * hindrance

    if lam < 0.2:
        regime = "unhindered"
    elif lam < 0.5:
        regime = "hindered"
    elif lam < 0.9:
        regime = "severely_hindered"
    else:
        regime = "excluded"

    return d_pore, hindrance, lam, regime


def compute_thiele_modulus(d_eff, k_rxn_s, particle_radius_m):
    """Thiele modulus: φ = R × √(k/D_eff).

    φ < 0.3: reaction-limited (all scaffold interior is used)
    0.3 < φ < 3: transitional
    φ > 3: diffusion-limited (only outer shell of scaffold is active)

    Returns (φ, effectiveness_factor η)
    """
    if d_eff <= 0 or k_rxn_s <= 0 or particle_radius_m <= 0:
        return 0.0, 1.0

    phi = particle_radius_m * math.sqrt(k_rxn_s / d_eff)

    # Effectiveness factor for sphere: η = (1/φ) × (1/tanh(3φ) - 1/(3φ))
    if phi < 0.01:
        eta = 1.0
    elif phi > 50:
        eta = 1.0 / phi
    else:
        try:
            eta = (1.0 / phi) * (1.0 / math.tanh(3.0 * phi) - 1.0 / (3.0 * phi))
        except (OverflowError, ZeroDivisionError):
            eta = 1.0 / phi

    eta = max(0.001, min(1.0, eta))

    return round(phi, 3), round(eta, 4)


# ═══════════════════════════════════════════════════════════════════════════
# FULL TRANSPORT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_transport(
    hydrated_radius_nm, pore_diameter_nm,
    particle_diameter_um=10.0,
    k_on_M_s=1e6,
    temp_c=25.0,
    target_charge=2,
):
    """Full mass transport analysis for target in scaffold.

    Args:
        hydrated_radius_nm: Hydrated radius of target ion
        pore_diameter_nm: Scaffold pore diameter
        particle_diameter_um: Scaffold particle size (for Thiele)
        k_on_M_s: Binding on-rate (M⁻¹s⁻¹)
        temp_c: Temperature
        target_charge: For electromigration correction

    Returns:
        DiffusionProfile
    """
    pore_radius_nm = pore_diameter_nm / 2.0

    d_bulk = stokes_einstein(hydrated_radius_nm, temp_c)
    d_pore, hindrance, lam, regime = hindered_diffusion(
        d_bulk, hydrated_radius_nm, pore_radius_nm)

    # Thiele modulus
    particle_r_m = particle_diameter_um * 1e-6 / 2.0
    # Convert k_on (M⁻¹s⁻¹) to pseudo-first-order k (s⁻¹) at 1 µM target
    k_pseudo = k_on_M_s * 1e-6  # s⁻¹ at 1 µM
    d_eff = d_pore if d_pore > 0 else d_bulk * 0.001

    phi, eta = compute_thiele_modulus(d_eff, k_pseudo, particle_r_m)

    # Rate-limiting step
    if regime == "excluded":
        rate_limit = "pore_exclusion"
    elif phi > 3:
        rate_limit = "pore_diffusion"
    elif phi < 0.3:
        rate_limit = "reaction"
    else:
        rate_limit = "mixed"

    notes_parts = []
    if regime == "excluded":
        notes_parts.append(f"Target (r={hydrated_radius_nm:.2f} nm) EXCLUDED from "
                           f"pore (r={pore_radius_nm:.2f} nm)")
    elif regime == "severely_hindered":
        notes_parts.append(f"Severely hindered transport: D_pore/D_bulk = {hindrance:.4f}")
    if phi > 3:
        notes_parts.append(f"Diffusion-limited (φ={phi:.1f}): only {eta*100:.0f}% "
                           f"of scaffold interior is effective")

    return DiffusionProfile(
        d_bulk_m2_s=d_bulk, d_pore_m2_s=d_pore,
        hindrance_factor=round(hindrance, 6),
        lambda_ratio=round(lam, 3),
        transport_regime=regime,
        thiele_modulus=phi, effectiveness_factor=eta,
        rate_limiting_step=rate_limit,
        notes="; ".join(notes_parts),
    )


def predict_capture_time(
    target_conc_uM, capacity_mmol_g, material_g_per_L,
    k_on_M_s=1e6, effectiveness=1.0,
    flow_rate_mL_min=0.0, column_volume_mL=0.0,
):
    """Predict time to reach various loading levels.

    For batch mode: simple kinetic model
    For column mode: breakthrough prediction

    Args:
        target_conc_uM: Target concentration in solution
        capacity_mmol_g: Material capacity
        material_g_per_L: Amount of material per liter
        k_on_M_s: Binding on-rate
        effectiveness: Thiele effectiveness factor
        flow_rate_mL_min: >0 for column mode
        column_volume_mL: Column volume for breakthrough
    """
    # Total capacity in solution
    cap_total_uM = capacity_mmol_g * material_g_per_L * 1000  # µM equivalent

    if cap_total_uM <= 0:
        return CaptureKinetics(0, 0, 0, 0, 0, 0, "No capacity")

    # Effective k_on accounting for transport
    k_eff = k_on_M_s * effectiveness
    conc_M = target_conc_uM * 1e-6

    # Pseudo-first-order rate
    k_pseudo = k_eff * conc_M  # s⁻¹

    if k_pseudo <= 0:
        return CaptureKinetics(1e12, 1e12, 1e12, 0, 0, 0, "Zero rate")

    # Time to reach fraction f: t = -ln(1-f) / k_pseudo (first-order approximation)
    t_50 = -math.log(0.5) / k_pseudo
    t_90 = -math.log(0.1) / k_pseudo
    t_99 = -math.log(0.01) / k_pseudo

    # Initial flux
    flux = k_eff * conc_M**2 * 1000  # mol/m²/s (order of magnitude)

    # Equilibrium loading
    eq_load = min(100.0, (target_conc_uM / cap_total_uM) * 100) if cap_total_uM > 0 else 0

    # Breakthrough for column
    breakthrough = 0.0
    if flow_rate_mL_min > 0 and column_volume_mL > 0:
        bed_volumes = cap_total_uM / max(0.001, target_conc_uM)
        breakthrough = bed_volumes * column_volume_mL / flow_rate_mL_min * 60  # seconds

    notes = ""
    if t_90 > 3600:
        notes = f"Slow: {t_90/3600:.1f} hours to 90% capture"
    elif t_90 < 60:
        notes = f"Fast: {t_90:.0f} seconds to 90% capture"

    return CaptureKinetics(
        time_to_50pct_s=round(t_50, 1),
        time_to_90pct_s=round(t_90, 1),
        time_to_99pct_s=round(t_99, 1),
        initial_flux_mol_m2_s=flux,
        equilibrium_loading_pct=round(eq_load, 1),
        breakthrough_time_s=round(breakthrough, 1),
        notes=notes,
    )

