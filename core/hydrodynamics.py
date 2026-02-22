"""
core/hydrodynamics.py - Mass transport and hydrodynamic analysis.

The thermodynamics say WHETHER binding is favorable.
The hydrodynamics say WHETHER the target GETS THERE.

Key physics:
    D = kT / (6πηr)                     Stokes-Einstein diffusion
    Pe = vL/D                            Péclet number (advection/diffusion)
    J_pore = D × ΔC × A_pore / L_pore   Fick's law pore entry flux
    τ_res = V_cage / J_exit              Residence time inside cage
    k_mt = D / δ                         Mass transfer coefficient

Regimes:
    Pe < 1:   diffusion-dominated (well-mixed, batch capture)
    Pe > 1:   advection-dominated (flow-through, column)
    Pe >> 100: need to design for mass transfer limitation

If the binder is thermodynamically perfect but the target can't
diffuse through the pore fast enough, the system is transport-limited
and real performance is much worse than equilibrium prediction.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.assembly import StructuralConstraint, InteriorDesign
from core.problem import Problem


# Physical constants
K_BOLTZMANN = 1.381e-23   # J/K
AVOGADRO = 6.022e23
ETA_WATER_25C = 8.9e-4    # Pa·s (dynamic viscosity of water at 25°C)


def _viscosity_at_temp(temp_c: float) -> float:
    """Water viscosity (Pa·s) at given temperature. Arrhenius approximation."""
    # η(T) ≈ η(25°C) × exp(1650 × (1/T - 1/298.15))
    T = temp_c + 273.15
    return ETA_WATER_25C * math.exp(1650.0 * (1.0 / T - 1.0 / 298.15))


@dataclass
class HydrodynamicProfile:
    """Mass transport analysis for a binder assembly."""

    # Target diffusion
    diffusion_coeff_m2s: float = 0.0        # Stokes-Einstein D
    target_hydrated_radius_m: float = 0.0

    # Flow regime
    peclet_number: Optional[float] = None   # Pe = vL/D
    flow_regime: str = "diffusion"           # diffusion, mixed, advection

    # Pore transport (for cages, zeolites, mesoporous)
    pore_entry_rate_per_s: Optional[float] = None  # molecules entering per second
    pore_diffusion_limited: bool = False
    pore_restriction_factor: float = 1.0     # Renkin correction for hindered diffusion

    # Residence time
    residence_time_s: Optional[float] = None  # time target spends in capture zone
    capture_probability: float = 1.0          # P(capture) given entry

    # Mass transfer
    mass_transfer_coeff_ms: Optional[float] = None   # k_mt = D / δ
    transport_limitation_factor: float = 1.0  # 1.0 = no limitation, <1 = transport-limited

    # Breakdown
    hydro_breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"D = {self.diffusion_coeff_m2s:.2e} m²/s",
            f"Flow regime: {self.flow_regime}" + (f" (Pe = {self.peclet_number:.1f})" if self.peclet_number else ""),
        ]
        if self.pore_entry_rate_per_s is not None:
            parts.append(f"Pore entry rate: {self.pore_entry_rate_per_s:.1e} /s")
        if self.pore_diffusion_limited:
            parts.append("⚠ PORE DIFFUSION LIMITED — real capture slower than equilibrium")
        if self.residence_time_s is not None:
            parts.append(f"Residence time in capture zone: {self.residence_time_s:.2e} s")
        parts.append(f"Transport limitation factor: {self.transport_limitation_factor:.2f}")
        return "\n".join(parts)


def compute_hydrodynamics(structure: StructuralConstraint,
                           interior: InteriorDesign,
                           problem: Problem) -> HydrodynamicProfile:
    """
    Full hydrodynamic analysis for a binder assembly.
    """
    target = problem.target
    temp_c = problem.matrix.temperature_c or 25.0
    temp_k = temp_c + 273.15
    eta = _viscosity_at_temp(temp_c)
    breakdown = []

    # ── Target diffusion coefficient (Stokes-Einstein) ────────────
    r_hyd_angstrom = 2.0  # default
    if target.hydration and target.hydration.hydrated_radius_angstrom:
        r_hyd_angstrom = target.hydration.hydrated_radius_angstrom
    r_hyd_m = r_hyd_angstrom * 1e-10

    D = K_BOLTZMANN * temp_k / (6.0 * math.pi * eta * r_hyd_m)
    breakdown.append(
        f"Stokes-Einstein: D = kT/(6πηr) = {D:.2e} m²/s "
        f"(r_hyd = {r_hyd_angstrom:.1f} Å, T = {temp_c:.0f}°C, η = {eta:.2e} Pa·s)"
    )

    # ── Flow regime (Péclet number) ───────────────────────────────
    pe = None
    flow_regime = "diffusion"

    flow_rate = problem.matrix.flow_rate_l_min
    if flow_rate and flow_rate > 0:
        # Assume a characteristic length scale based on structure
        if structure.type in ("mesoporous_silica", "zeolite", "mof", "cof"):
            # Packed bed: characteristic length ~ particle diameter ~ 100 µm
            L_char = 100e-6  # 100 µm
        elif structure.type in ("dna_origami_cage", "protein_cage", "coordination_cage"):
            # Nanoscale: characteristic length ~ cage diameter
            if structure.interior_volume_nm3:
                L_char = (structure.interior_volume_nm3 ** (1.0/3.0)) * 1e-9
            else:
                L_char = 40e-9  # 40 nm default
        else:
            L_char = 1e-3  # 1 mm default for macro structures

        # Rough velocity estimate: flow through characteristic cross-section
        # Assume 1 cm² flow area for packed bed
        flow_m3s = flow_rate * 1e-3 / 60.0  # L/min → m³/s
        area = 1e-4  # 1 cm²
        v = flow_m3s / area

        pe = v * L_char / D
        if pe < 1:
            flow_regime = "diffusion"
        elif pe < 100:
            flow_regime = "mixed"
        else:
            flow_regime = "advection"

        breakdown.append(f"Pe = vL/D = {pe:.1f} → {flow_regime}-dominated")
    else:
        breakdown.append("No flow rate specified — assuming diffusion-dominated (batch)")

    # ── Pore transport ────────────────────────────────────────────
    pore_entry_rate = None
    pore_limited = False
    restriction = 1.0

    if structure.pore_size_nm and structure.pore_size_nm > 0:
        r_pore_m = structure.pore_size_nm * 1e-9 / 2.0
        r_target_m = r_hyd_m

        # Renkin correction: hindered diffusion in pores
        # λ = r_target / r_pore
        lam = r_target_m / r_pore_m if r_pore_m > 0 else 1.0

        if lam >= 1.0:
            restriction = 0.0
            pore_limited = True
            breakdown.append(
                f"TARGET EXCLUDED: hydrated radius ({r_hyd_angstrom:.1f} Å) ≥ "
                f"pore radius ({structure.pore_size_nm * 10 / 2:.1f} Å). "
                f"Target cannot enter structure."
            )
        elif lam > 0.9:
            restriction = 0.01
            pore_limited = True
            breakdown.append(f"Near-exclusion: λ = {lam:.2f}.")
        else:
            # Renkin equation: D_pore/D_bulk = (1-λ)² × (1 - 2.104λ + 2.089λ³)
            restriction = (1.0 - lam)**2 * (1.0 - 2.104*lam + 2.089*lam**3)
            restriction = max(0.0, restriction)

            # Channel geometry correction: continuous channels have less restriction
            # than single apertures (Renkin was derived for the latter)
            if structure.geometry in ("hexagonal_channels", "channel_intersection"):
                restriction = restriction ** 0.6
                breakdown.append("Channel geometry: restriction softened vs single-pore Renkin")

            D_pore = D * restriction

            # Fick's law: J = D_pore × ΔC × A_pore / L_pore
            # Assume ΔC = 1 mM = 6e20 molecules/m³
            # A_pore = π × r_pore²
            # L_pore = wall thickness ~ 2-5 nm for DNA origami, more for zeolites
            A_pore = math.pi * r_pore_m**2
            L_pore = max(2e-9, structure.pore_size_nm * 0.5e-9)  # rough wall thickness
            dC = 6.022e20  # 1 mM in molecules/m³

            pore_entry_rate = D_pore * dC * A_pore / L_pore
            pore_limited = restriction < 0.3

            breakdown.append(
                f"Pore transport: λ = {lam:.2f}, Renkin correction = {restriction:.3f}, "
                f"D_pore = {D_pore:.2e} m²/s"
            )
            if pore_limited:
                breakdown.append(
                    f"⚠ Hindered diffusion: pore restricts transport to {restriction:.0%} of bulk"
                )

    # ── Residence time in capture zone ────────────────────────────
    residence_time = None
    if structure.interior_volume_nm3 and structure.interior_volume_nm3 > 0 and not (restriction == 0.0):
        V_cage_m3 = structure.interior_volume_nm3 * 1e-27
        # Exit rate ~ D × A_pore / V_cage
        if structure.pore_size_nm and structure.pore_size_nm > 0:
            A_exit = math.pi * (structure.pore_size_nm * 1e-9 / 2.0)**2
            D_eff = D * restriction
            exit_rate = D_eff * A_exit / V_cage_m3 if V_cage_m3 > 0 else 1e30
            residence_time = 1.0 / exit_rate if exit_rate > 0 else float('inf')
            breakdown.append(f"Residence time: {residence_time:.2e} s")

    # ── Capture probability during residence ──────────────────────
    capture_prob = 1.0
    if residence_time is not None and interior.total_binding_sites > 0:
        # Simple: if many binding sites and long residence, capture is ~certain
        # If short residence and few sites, capture depends on on-rate
        # Rough: P_capture ≈ 1 - exp(-k_on × [sites] × τ)
        # Typical k_on ~ 10⁶ M⁻¹s⁻¹ for small molecule-metal
        k_on = 1e6  # M⁻¹s⁻¹
        site_conc_M = interior.total_binding_sites / (V_cage_m3 * AVOGADRO) if V_cage_m3 > 0 else 1.0
        capture_prob = 1.0 - math.exp(-k_on * site_conc_M * residence_time)
        capture_prob = max(0.0, min(1.0, capture_prob))
        breakdown.append(f"Capture probability during residence: {capture_prob:.2%}")
    elif structure.type == "none":
        capture_prob = 1.0  # free in solution, no transport limitation
        breakdown.append("Free in solution — no transport limitation")
    elif structure.type in ("mesoporous_silica", "graphene_oxide", "silica_np"):
        # High surface area, open access
        capture_prob = 0.95
        breakdown.append("High surface area, open geometry — minimal transport limitation")

    # ── Overall transport limitation factor ───────────────────────
    transport_factor = restriction * capture_prob
    if restriction == 0.0:
        transport_factor = 0.0
    elif structure.type == "none":
        transport_factor = 1.0
    elif structure.type in ("mesoporous_silica", "graphene_oxide", "silica_np"):
        transport_factor = max(0.5, restriction * 0.95)

    breakdown.append(f"Transport limitation factor: {transport_factor:.3f}")

    return HydrodynamicProfile(
        diffusion_coeff_m2s=D,
        target_hydrated_radius_m=r_hyd_m,
        peclet_number=pe,
        flow_regime=flow_regime,
        pore_entry_rate_per_s=pore_entry_rate,
        pore_diffusion_limited=pore_limited,
        pore_restriction_factor=round(restriction, 4),
        residence_time_s=residence_time,
        capture_probability=round(capture_prob, 4),
        mass_transfer_coeff_ms=D / 100e-6 if pe and pe > 1 else None,  # boundary layer ~ 100µm
        transport_limitation_factor=round(transport_factor, 4),
        hydro_breakdown=breakdown,
    )
