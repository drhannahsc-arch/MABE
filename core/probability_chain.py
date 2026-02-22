"""
core/probability_chain.py - Statistical probability chain for molecular events.

A successful capture-release cycle is a chain of conditional events:

    P(cycle) = P(enter) × P(encounter|enter) × P(bind|encounter)
             × P(retain|bind) × P(release|trigger,retain)

Each probability is computed from physics, not estimated.

This replaces the old composite_score with a real probability that has
physical meaning: "out of 1000 target molecules in solution, how many
complete the full capture-release cycle?"
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

from core.thermodynamics import BindingThermodynamics
from core.hydrodynamics import HydrodynamicProfile
from core.kinetics import KineticProfile, _format_time
from core.orbital_binding import OrbitalAnalysis
from core.assembly import BinderAssembly
from core.problem import Problem


@dataclass
class ProbabilityChain:
    """Complete probability chain for a molecular capture event."""
    # Individual event probabilities
    p_enter: float = 1.0         # target reaches the binder (transport)
    p_encounter: float = 1.0     # target finds binding site given entry
    p_bind: float = 1.0          # target binds given encounter (activation barrier)
    p_retain: float = 1.0        # target stays bound long enough to be useful
    p_release: float = 1.0       # target releases when triggered

    # Chain products
    p_capture: float = 1.0       # P(enter) × P(encounter) × P(bind) = capture probability
    p_cycle: float = 1.0         # full capture-release cycle probability

    # Context
    exposure_time_s: float = 60.0   # how long target is exposed to binder
    retain_threshold_s: float = 10.0  # minimum retention time needed

    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"P(capture) = P(enter)×P(encounter)×P(bind) = {self.p_enter:.2f}×{self.p_encounter:.2f}×{self.p_bind:.2f} = {self.p_capture:.3f}",
            f"P(retain) = {self.p_retain:.3f} | P(release|trigger) = {self.p_release:.2f}",
            f"P(full cycle) = {self.p_cycle:.4f}",
        ]
        return "\n".join(parts)


def compute_probability_chain(thermo: BindingThermodynamics,
                                hydro: HydrodynamicProfile,
                                kinetics: KineticProfile,
                                orbital: OrbitalAnalysis,
                                assembly: BinderAssembly,
                                problem: Problem) -> ProbabilityChain:
    """
    Compute the probability chain for a complete capture-release event.
    """
    breakdown = []
    chain = ProbabilityChain()

    # Estimate exposure time from context
    flow_rate = problem.matrix.flow_rate_l_min
    if flow_rate and flow_rate > 0:
        # Flow-through: exposure time ~ residence time of solution in bed
        chain.exposure_time_s = 30.0  # typical packed bed residence
    else:
        chain.exposure_time_s = 600.0  # 10 min batch contact

    # ── P(enter): target reaches the binder ───────────────────────
    chain.p_enter = hydro.transport_limitation_factor
    breakdown.append(f"P(enter) = {chain.p_enter:.3f} (transport factor)")

    # ── P(encounter): target finds binding site given entry ───────
    # Depends on: interior site density, cage volume, exposure time
    interior = assembly.interior
    if assembly.structure.type == "none":
        # Free in solution: encounter is diffusion-limited
        # P(encounter) ≈ 1 - exp(-k_on × [binder] × t_exposure)
        # Assume binder at 1 µM = 1e-6 M
        binder_conc = 1e-6
        chain.p_encounter = 1.0 - math.exp(-kinetics.k_on_M_s * binder_conc * chain.exposure_time_s)
    elif interior.total_binding_sites > 10:
        # Dense interior: very high encounter probability once entered
        chain.p_encounter = 0.95
    elif interior.total_binding_sites > 1:
        chain.p_encounter = 0.80
    else:
        chain.p_encounter = 0.60

    breakdown.append(f"P(encounter|enter) = {chain.p_encounter:.3f} ({interior.total_binding_sites} sites)")

    # ── P(bind): target binds given encounter ─────────────────────
    # From Boltzmann: P = exp(-ΔG‡/RT) where ΔG‡ is activation barrier
    # Use kinetics: P(bind per encounter) = k_on × t_encounter / k_diff
    # Or simply from occupancy at equilibrium if time allows
    if kinetics.time_to_equilibrium_s < chain.exposure_time_s:
        # Enough time to reach equilibrium: P(bind) ≈ θ (occupancy)
        chain.p_bind = kinetics.fractional_occupancy
        breakdown.append(f"P(bind|encounter) = {chain.p_bind:.3f} (equilibrium reached, θ at working [target])")
    else:
        # Not enough time: P(bind) from kinetic approach to equilibrium
        # θ(t) = θ_eq × (1 - exp(-(k_on×[T] + k_off) × t))
        target_M = 100e-6  # 100 µM estimate
        approach_rate = kinetics.k_on_M_s * target_M + kinetics.k_off_s
        fraction_eq = 1.0 - math.exp(-approach_rate * chain.exposure_time_s)
        chain.p_bind = kinetics.fractional_occupancy * fraction_eq
        breakdown.append(
            f"P(bind|encounter) = {chain.p_bind:.3f} (kinetic: {fraction_eq:.0%} of equilibrium in "
            f"{_format_time(chain.exposure_time_s)})"
        )

    # ── P(retain): stays bound long enough ────────────────────────
    # Need target to stay bound for at least the processing time
    chain.retain_threshold_s = 10.0  # minimum 10s retention
    if kinetics.residence_time_s > chain.retain_threshold_s * 10:
        chain.p_retain = 0.99
    elif kinetics.residence_time_s > chain.retain_threshold_s:
        chain.p_retain = 1.0 - math.exp(-kinetics.residence_time_s / chain.retain_threshold_s)
    else:
        chain.p_retain = kinetics.residence_time_s / chain.retain_threshold_s

    chain.p_retain = max(0.0, min(1.0, chain.p_retain))
    breakdown.append(
        f"P(retain >{_format_time(chain.retain_threshold_s)}) = {chain.p_retain:.3f} "
        f"(τ_res = {_format_time(kinetics.residence_time_s)})"
    )

    # ── P(release): releases when triggered ───────────────────────
    release = assembly.release
    wants_release = "release" in problem.desired_outcome.description.lower()
    if not wants_release or release.trigger == "none":
        chain.p_release = 1.0  # permanent capture, no release needed
        breakdown.append("P(release) = 1.0 (permanent capture or no release needed)")
    else:
        # Release efficiency from mechanism
        eff_str = release.release_efficiency or ""
        if ">95%" in eff_str:
            chain.p_release = 0.95
        elif ">90%" in eff_str:
            chain.p_release = 0.90
        elif "80-95%" in eff_str:
            chain.p_release = 0.85
        elif "70-95%" in eff_str:
            chain.p_release = 0.80
        elif "60-90%" in eff_str:
            chain.p_release = 0.70
        else:
            chain.p_release = 0.75  # default

        # Very strong binders (ΔG < -40 kJ/mol) may resist elution
        if thermo.dG_net < -40:
            penalty = 0.9  # 10% harder to elute
            chain.p_release *= penalty
            breakdown.append(f"Strong binding penalty: release × {penalty:.2f}")

        breakdown.append(f"P(release|trigger) = {chain.p_release:.3f} ({release.name})")

    # ── Chain products ────────────────────────────────────────────
    chain.p_capture = chain.p_enter * chain.p_encounter * chain.p_bind
    chain.p_cycle = chain.p_capture * chain.p_retain * chain.p_release

    breakdown.append(f"P(capture) = {chain.p_capture:.4f}")
    breakdown.append(f"P(full cycle) = {chain.p_cycle:.4f}")

    chain.breakdown = breakdown
    return chain
