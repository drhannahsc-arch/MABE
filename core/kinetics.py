"""
core/kinetics.py - Kinetic response dynamics.

Thermodynamics says WHERE the system goes. Kinetics says HOW FAST.

Key quantities:
    k_on:  association rate constant (M⁻¹s⁻¹)
           How fast does the target bind? Diffusion-limited? Reaction-limited?
    k_off: dissociation rate constant (s⁻¹)
           How fast does it fall off? Determines residence time.
    τ_res: residence time = 1/k_off (seconds)
           How long does target stay bound?
    t_eq:  time to reach equilibrium
    C_50:  concentration for 50% binding (like IC50/EC50)
    θ:     fractional occupancy at given [target] (Langmuir isotherm)

These connect thermodynamic ΔG to real-world capture performance.

Key relationships:
    K_eq = k_on / k_off           (thermodynamic-kinetic bridge)
    k_off = k_on / K_eq           (from ΔG → K_eq → k_off)
    τ_res = 1 / k_off
    θ = [T] / ([T] + Kd)          (Langmuir saturation)
    t_eq ≈ 4 / (k_on × [T] + k_off)  (time to 98% equilibrium)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.thermodynamics import BindingThermodynamics, R_GAS
from core.hydrodynamics import HydrodynamicProfile
from core.orbital_binding import OrbitalAnalysis
from core.assembly import StructuralConstraint, InteriorDesign
from core.problem import Problem


@dataclass
class KineticProfile:
    """Complete kinetic characterization of a binding interaction."""
    # Rate constants
    k_on_M_s: float = 0.0           # association rate (M⁻¹s⁻¹)
    k_off_s: float = 0.0            # dissociation rate (s⁻¹)

    # Derived
    residence_time_s: float = 0.0    # τ = 1/k_off
    half_life_s: float = 0.0         # t½ = ln(2)/k_off
    time_to_equilibrium_s: float = 0.0  # time to ~98% equilibrium

    # Saturation
    kd_um: float = 0.0              # dissociation constant (µM)
    fractional_occupancy: float = 0.0  # θ at target concentration
    c50_um: float = 0.0              # concentration for 50% occupancy

    # Regime
    rate_limiting_step: str = ""     # "diffusion", "reaction", "transport"

    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"k_on = {self.k_on_M_s:.1e} M⁻¹s⁻¹ | k_off = {self.k_off_s:.1e} s⁻¹",
            f"Residence time: {_format_time(self.residence_time_s)} | t_eq: {_format_time(self.time_to_equilibrium_s)}",
            f"Kd = {self.kd_um:.2f} µM | Occupancy: {self.fractional_occupancy:.0%} at working [target]",
            f"Rate-limiting: {self.rate_limiting_step}",
        ]
        return "\n".join(parts)


def _format_time(t_s: float) -> str:
    if t_s < 1e-6:
        return f"{t_s*1e9:.0f} ns"
    elif t_s < 1e-3:
        return f"{t_s*1e6:.0f} µs"
    elif t_s < 1:
        return f"{t_s*1e3:.0f} ms"
    elif t_s < 60:
        return f"{t_s:.1f} s"
    elif t_s < 3600:
        return f"{t_s/60:.1f} min"
    elif t_s < 86400:
        return f"{t_s/3600:.1f} hr"
    else:
        return f"{t_s/86400:.1f} days"


def compute_kinetics(thermo: BindingThermodynamics,
                      hydro: HydrodynamicProfile,
                      orbital: OrbitalAnalysis,
                      structure: StructuralConstraint,
                      interior: InteriorDesign,
                      problem: Problem) -> KineticProfile:
    """
    Compute kinetic profile from thermodynamic and hydrodynamic data.
    """
    breakdown = []
    temp_k = thermo.temperature_k
    RT = R_GAS * temp_k

    # ── k_on estimation ───────────────────────────────────────────
    # Diffusion-limited k_on (Smoluchowski): k_diff = 4π × D × r × N_A
    # For ion-small molecule encounter: ~10⁹ M⁻¹s⁻¹ in water
    D = hydro.diffusion_coeff_m2s
    r_encounter = 5e-10  # ~5 Å encounter radius
    k_diff = 4 * math.pi * D * r_encounter * 6.022e23  # M⁻¹s⁻¹
    breakdown.append(f"Diffusion-limited k_on (Smoluchowski): {k_diff:.1e} M⁻¹s⁻¹")

    # Reaction probability per encounter: from activation barrier
    # ΔG‡ estimated from orbital analysis + desolvation
    dG_barrier = 0.0

    # Desolvation barrier: must strip water to bind
    # Barrier ~ fraction of desolvation penalty (Marcus theory-like)
    dG_barrier += abs(thermo.dG_desolv) * 0.3  # ~30% of desolv as barrier

    # Orbital correction: favorable charge transfer lowers barrier
    if orbital.dft_data_available and orbital.charge_transfer_dG_kj < 0:
        ct_reduction = abs(orbital.charge_transfer_dG_kj) * 0.2
        dG_barrier -= ct_reduction
        breakdown.append(f"CT barrier reduction: -{ct_reduction:.1f} kJ/mol")

    dG_barrier = max(0.0, dG_barrier)

    # Transition state theory: k_reaction = (kT/h) × exp(-ΔG‡/RT)
    # For our purposes: probability of reaction per encounter
    p_react = math.exp(-dG_barrier / RT) if dG_barrier / RT < 500 else 0.0
    breakdown.append(f"Activation barrier: {dG_barrier:.1f} kJ/mol → P(react) = {p_react:.3f}")

    # Effective k_on = k_diff × P(react) × transport_factor
    transport = hydro.transport_limitation_factor
    k_on = k_diff * p_react * transport

    # Structure-specific k_on adjustments
    if structure.type == "none":
        pass  # free in solution, no correction
    elif structure.type in ("zeolite", "mof") and structure.pore_size_nm and structure.pore_size_nm < 1.0:
        # Tight pores: on-rate limited by pore entry, not encounter
        k_on *= 0.1
        breakdown.append("Tight-pore correction: k_on × 0.1")
    elif structure.type == "mip":
        # MIP cavities: limited access, rebinding enhances apparent rate
        k_on *= 0.5
        breakdown.append("MIP access correction: k_on × 0.5")

    # Minimum physical k_on
    k_on = max(k_on, 1.0)  # at least 1 M⁻¹s⁻¹
    breakdown.append(f"Effective k_on = {k_on:.1e} M⁻¹s⁻¹")

    # ── k_off from thermodynamic-kinetic bridge ───────────────────
    K_eq = thermo.K_eq
    if K_eq > 0:
        k_off = k_on / K_eq
    else:
        k_off = 1e6  # very fast dissociation

    k_off = max(k_off, 1e-10)  # physical minimum
    breakdown.append(f"k_off = k_on/K_eq = {k_off:.1e} s⁻¹")

    # ── Derived quantities ────────────────────────────────────────
    tau_res = 1.0 / k_off
    t_half = math.log(2) / k_off

    # Kd
    kd_M = k_off / k_on if k_on > 0 else 1.0
    kd_um = kd_M * 1e6

    # Target concentration (estimate from matrix)
    # Typical: mine water metals 0.01-10 mM
    target_conc_um = 100.0  # default 100 µM
    for comp in problem.matrix.competing_species:
        if comp.identity.lower() == problem.target.identity.lower():
            target_conc_um = comp.concentration_mm * 1000.0
            break

    # Langmuir fractional occupancy: θ = [T] / ([T] + Kd)
    theta = target_conc_um / (target_conc_um + kd_um) if (target_conc_um + kd_um) > 0 else 0.0

    # Time to equilibrium: t_eq ≈ 4 / (k_on × [T] + k_off)
    target_M = target_conc_um * 1e-6
    t_eq = 4.0 / (k_on * target_M + k_off) if (k_on * target_M + k_off) > 0 else 1e6

    breakdown.append(f"τ_res = {_format_time(tau_res)} | t_eq = {_format_time(t_eq)}")
    breakdown.append(f"Kd = {kd_um:.2f} µM | θ = {theta:.0%} at {target_conc_um:.0f} µM")

    # Rate-limiting step
    if transport < 0.3:
        rate_limiting = "transport (target can't reach binding site fast enough)"
    elif p_react < 0.01:
        rate_limiting = "reaction (high activation barrier, slow desolvation)"
    elif k_on < 1e3:
        rate_limiting = "association (slow on-rate limits capture)"
    else:
        rate_limiting = "diffusion (fast kinetics, performance near thermodynamic limit)"

    return KineticProfile(
        k_on_M_s=round(k_on, 1),
        k_off_s=k_off,
        residence_time_s=tau_res,
        half_life_s=t_half,
        time_to_equilibrium_s=t_eq,
        kd_um=round(kd_um, 4),
        fractional_occupancy=round(theta, 4),
        c50_um=round(kd_um, 4),
        rate_limiting_step=rate_limiting,
        breakdown=breakdown,
    )
