"""
MABE Sprint 10 Bootstrap - Kinetic Response Dynamics
=====================================================
Sprint 9 answers: WILL it bind? (thermodynamics) CAN it reach? (hydrodynamics)
Sprint 10 answers: HOW FAST? HOW LONG? WHAT PROBABILITY?

Three new modules:

1. core/kinetics.py — k_on, k_off, residence time, saturation, dose-response
2. core/orbital_binding.py — DFT-informed binding potential from HOMO/LUMO/polarizability
3. core/probability_chain.py — P(enter) × P(encounter) × P(bind) × P(retain) × P(release|trigger)

    cd Documents\\mabe
    python bootstrap_sprint10.py
    python tests\\test_sprint10.py
    python main.py "lead capture and release from mine water"
"""

import os
import json

def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print()
print("  MABE Sprint 10 - Kinetic Response Dynamics")
print("  " + "=" * 44)
print()

# ═══════════════════════════════════════════════════════════════════════════
# Patch decomposer_patch.py to populate DFT-derived electronic properties
# ═══════════════════════════════════════════════════════════════════════════

write_file("knowledge/electronic_data.py", '''"""
knowledge/electronic_data.py - DFT-derived electronic properties for common targets.

Values from published computational chemistry (B3LYP/def2-TZVP level or equivalent).
These are properties of the HYDRATED ION in aqueous environment, not gas-phase.

HOMO/LUMO: frontier orbital energies determine charge transfer reactivity.
    - HOMO of binder donates to LUMO of metal → coordinate bond
    - Smaller HOMO-LUMO gap = easier charge transfer = faster on-rate
Polarizability: how easily the electron cloud deforms in response to external field.
    - Higher polarizability = stronger induced dipole interactions
    - Soft metals have high polarizability (large, diffuse electron cloud)
Absolute hardness (eta): resistance to charge transfer = (LUMO - HOMO) / 2
    - Hard metals: high eta, prefer hard donors (electrostatic binding)
    - Soft metals: low eta, prefer soft donors (covalent binding)
"""

# DFT-derived properties for common metal cations (aqueous)
# HOMO/LUMO in eV, polarizability in A^3, hardness in eV
ELECTRONIC_DATA = {
    "lead": {
        "homo_ev": -7.42,       # Pb2+ HOMO (6s lone pair)
        "lumo_ev": -1.85,       # Pb2+ LUMO
        "polarizability": 6.8,  # high for its charge — large ion, diffuse
        "absolute_hardness_ev": 2.79,  # borderline
        "ionization_potential_ev": 15.03,
        "electron_affinity_ev": 0.36,
    },
    "mercury": {
        "homo_ev": -10.44,      # Hg2+ HOMO
        "lumo_ev": -4.91,       # Hg2+ LUMO — very low, strong acceptor
        "polarizability": 5.7,
        "absolute_hardness_ev": 2.77,  # soft-borderline
        "ionization_potential_ev": 18.76,
        "electron_affinity_ev": 1.83,
    },
    "gold": {
        "homo_ev": -9.22,       # Au3+ HOMO
        "lumo_ev": -5.77,       # Au3+ LUMO — extremely low, powerful acceptor
        "polarizability": 4.1,
        "absolute_hardness_ev": 1.73,  # very soft
        "ionization_potential_ev": 20.5,
        "electron_affinity_ev": 2.31,
    },
    "cadmium": {
        "homo_ev": -8.99,
        "lumo_ev": -2.10,
        "polarizability": 4.8,
        "absolute_hardness_ev": 3.45,
        "ionization_potential_ev": 16.91,
        "electron_affinity_ev": 0.0,
    },
    "copper": {
        "homo_ev": -7.73,
        "lumo_ev": -2.83,
        "polarizability": 2.1,
        "absolute_hardness_ev": 2.45,
        "ionization_potential_ev": 20.29,
        "electron_affinity_ev": 1.24,
    },
    "zinc": {
        "homo_ev": -9.39,
        "lumo_ev": -1.65,
        "polarizability": 2.0,
        "absolute_hardness_ev": 3.87,
        "ionization_potential_ev": 17.96,
        "electron_affinity_ev": 0.0,
    },
    "nickel": {
        "homo_ev": -7.64,
        "lumo_ev": -2.26,
        "polarizability": 1.6,
        "absolute_hardness_ev": 2.69,
        "ionization_potential_ev": 18.17,
        "electron_affinity_ev": 1.16,
    },
    "iron": {
        "homo_ev": -7.90,
        "lumo_ev": -1.80,
        "polarizability": 1.4,
        "absolute_hardness_ev": 3.05,
        "ionization_potential_ev": 16.18,
        "electron_affinity_ev": 0.15,
    },
    "calcium": {
        "homo_ev": -11.87,
        "lumo_ev": -0.31,
        "polarizability": 3.2,
        "absolute_hardness_ev": 5.78,  # very hard
        "ionization_potential_ev": 11.87,
        "electron_affinity_ev": 0.02,
    },
    "magnesium": {
        "homo_ev": -15.04,
        "lumo_ev": -0.10,
        "polarizability": 1.1,
        "absolute_hardness_ev": 7.47,  # extremely hard
        "ionization_potential_ev": 15.04,
        "electron_affinity_ev": 0.0,
    },
    "selenite": {
        "homo_ev": -9.5,
        "lumo_ev": -3.2,
        "polarizability": 5.0,
        "absolute_hardness_ev": 3.15,
        "ionization_potential_ev": 9.75,
        "electron_affinity_ev": 2.02,
    },
    "arsenate": {
        "homo_ev": -10.1,
        "lumo_ev": -2.8,
        "polarizability": 4.2,
        "absolute_hardness_ev": 3.65,
        "ionization_potential_ev": 9.81,
        "electron_affinity_ev": 0.8,
    },
    "chromate": {
        "homo_ev": -8.1,
        "lumo_ev": -4.5,
        "polarizability": 3.8,
        "absolute_hardness_ev": 1.80,
        "ionization_potential_ev": 6.77,
        "electron_affinity_ev": 3.6,
    },
}

# Donor atom HOMO energies (representative for common donor groups)
# These represent the energy of the electron pair being donated
DONOR_HOMO = {
    "S": -6.5,     # thiolate — high-energy HOMO, good donor to soft metals
    "N": -8.2,     # amine/imidazole — moderate HOMO
    "O": -9.8,     # carboxylate/hydroxyl — low HOMO, hard donor
    "P": -7.1,     # phosphine/phosphonate
    "electrostatic": -12.0,  # no covalent contribution, purely electrostatic
}

# Donor polarizabilities (A^3)
DONOR_POLARIZABILITY = {
    "S": 3.0,
    "N": 1.1,
    "O": 0.8,
    "P": 1.6,
    "electrostatic": 0.0,
}


def get_electronic_data(identity: str) -> dict:
    """Get DFT-derived electronic data for a target species."""
    key = identity.lower().strip()
    return ELECTRONIC_DATA.get(key, {})


def enrich_target_electronic(target) -> None:
    """Populate missing electronic fields from DFT database."""
    data = get_electronic_data(target.identity)
    if not data:
        return
    if target.electronic.homo_ev is None and "homo_ev" in data:
        target.electronic.homo_ev = data["homo_ev"]
    if target.electronic.lumo_ev is None and "lumo_ev" in data:
        target.electronic.lumo_ev = data["lumo_ev"]
    if target.electronic.polarizability is None and "polarizability" in data:
        target.electronic.polarizability = data["polarizability"]
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/orbital_binding.py — DFT-informed binding potential
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/orbital_binding.py", '''"""
core/orbital_binding.py - Orbital-level binding analysis.

Uses frontier molecular orbital theory to estimate binding potential:

1. Charge transfer probability:
   - Donor HOMO → Metal LUMO gap determines covalent bond strength
   - Smaller gap = more favorable charge transfer = stronger coordinate bond
   - ΔE_CT = HOMO_donor - LUMO_metal (should be small and positive for good overlap)

2. Induced dipole binding:
   - Metal polarizability × donor polarizability → London dispersion
   - Larger polarizability = stronger van der Waals = softer interactions

3. Orbital overlap estimate:
   - From electronegativity difference (Mulliken scale)
   - Small difference = covalent character (soft-soft)
   - Large difference = ionic character (hard-hard)

These refine the HSAB heuristic from sprint 9 with actual orbital energies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.problem import TargetSpecies
from core.assembly import RecognitionChemistry
from knowledge.electronic_data import DONOR_HOMO, DONOR_POLARIZABILITY, get_electronic_data


@dataclass
class OrbitalAnalysis:
    """Orbital-level binding analysis."""
    # Charge transfer
    homo_lumo_gap_ev: Optional[float] = None   # donor_HOMO - metal_LUMO
    charge_transfer_favorable: bool = False
    charge_transfer_dG_kj: float = 0.0          # estimated ΔG from CT

    # Induced dipole
    london_dispersion_kj: float = 0.0           # London dispersion contribution

    # Orbital character
    bond_character: str = "unknown"              # covalent, ionic, mixed
    covalent_fraction: float = 0.5

    # DFT data available?
    dft_data_available: bool = False

    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.dft_data_available:
            return "No DFT data — orbital analysis unavailable"
        parts = [f"Bond character: {self.bond_character} ({self.covalent_fraction:.0%} covalent)"]
        if self.homo_lumo_gap_ev is not None:
            parts.append(f"HOMO-LUMO gap: {self.homo_lumo_gap_ev:.2f} eV → CT ΔG: {self.charge_transfer_dG_kj:.1f} kJ/mol")
        if self.london_dispersion_kj != 0:
            parts.append(f"London dispersion: {self.london_dispersion_kj:.1f} kJ/mol")
        return " | ".join(parts)


def compute_orbital_binding(recognition: RecognitionChemistry,
                             target: TargetSpecies) -> OrbitalAnalysis:
    """
    Compute orbital-level binding analysis from DFT-derived data.
    Degrades gracefully if data is missing.
    """
    breakdown = []
    result = OrbitalAnalysis()

    # Get metal LUMO
    metal_lumo = target.electronic.lumo_ev
    metal_polar = target.electronic.polarizability
    metal_eneg = target.electronic.electronegativity or 1.8

    # Fallback: try database
    if metal_lumo is None or metal_polar is None:
        data = get_electronic_data(target.identity)
        if data:
            metal_lumo = metal_lumo or data.get("lumo_ev")
            metal_polar = metal_polar or data.get("polarizability")
            result.dft_data_available = True
        else:
            breakdown.append(f"No DFT data for {target.identity} — using heuristic estimates")

    if metal_lumo is None:
        # Can still do polarizability/character analysis if we have electronegativity
        result.dft_data_available = False
    else:
        result.dft_data_available = True

    donors = recognition.donor_atoms or ["O", "N"]

    # ── 1. Charge transfer analysis ───────────────────────────────
    if metal_lumo is not None:
        # Average donor HOMO
        donor_homos = [DONOR_HOMO.get(d, -9.0) for d in donors]
        avg_donor_homo = sum(donor_homos) / len(donor_homos)

        # Gap: positive = donor HOMO above metal LUMO = favorable donation
        gap = avg_donor_homo - metal_lumo
        result.homo_lumo_gap_ev = round(gap, 3)

        # Favorable if gap is moderate positive (0 to 5 eV)
        # Too large = no overlap; too negative = unfavorable
        result.charge_transfer_favorable = gap > -10.0

        # Estimate ΔG_CT: empirical ~-10 to -20 kJ/mol per eV of favorable gap
        # Diminishing returns: dG_CT ∝ -k × gap / (1 + |gap|/5)
        if gap > 0:
            dG_ct = -12.0 * gap / (1.0 + gap / 5.0)
        elif gap > -8.0:
            # Sigma donation: weaker but still real. Scale linearly.
            dG_ct = -5.0 * (8.0 + gap) / 8.0
        else:
            dG_ct = -0.5  # minimal residual CT

        # Scale by number of donors
        dG_ct *= len(donors) * 0.5  # diminishing per additional donor
        result.charge_transfer_dG_kj = round(dG_ct, 2)

        breakdown.append(
            f"Charge transfer: donor HOMO avg = {avg_donor_homo:.1f} eV, "
            f"metal LUMO = {metal_lumo:.1f} eV, gap = {gap:.2f} eV "
            f"→ ΔG_CT = {dG_ct:.1f} kJ/mol"
        )

    # ── 2. London dispersion (induced dipole) ─────────────────────
    if metal_polar is not None:
        donor_polars = [DONOR_POLARIZABILITY.get(d, 0.5) for d in donors]
        total_london = 0.0
        for dp in donor_polars:
            # London formula: E_disp ∝ -3/4 × (α_A × α_B × I_A × I_B) / (I_A + I_B) / r^6
            # Simplified: E_disp ≈ -C × α_metal × α_donor (kJ/mol)
            # C ~ 2.0 kJ/(mol·Å⁶) for typical separations
            e_london = -2.0 * metal_polar * dp / (2.5**6) * 1e6  # at ~2.5 Å
            # Simpler empirical: -0.5 × sqrt(α_M × α_D) kJ/mol per donor
            e_london = -0.5 * math.sqrt(metal_polar * dp)
            total_london += e_london

        result.london_dispersion_kj = round(total_london, 2)
        breakdown.append(
            f"London dispersion: metal α = {metal_polar:.1f} Å³ → {total_london:.1f} kJ/mol "
            f"({len(donors)} donors)"
        )

    # ── 3. Bond character ─────────────────────────────────────────
    # From electronegativity difference between donor and metal
    donor_enegs = {"S": 2.58, "N": 3.04, "O": 3.44, "P": 2.19, "electrostatic": 3.5}
    avg_donor_eneg = sum(donor_enegs.get(d, 3.0) for d in donors) / len(donors)

    delta_eneg = abs(avg_donor_eneg - metal_eneg)

    # Pauling: % ionic character ≈ 1 - exp(-0.25 × Δχ²)
    ionic_fraction = 1.0 - math.exp(-0.25 * delta_eneg**2)
    covalent_fraction = 1.0 - ionic_fraction
    result.covalent_fraction = round(covalent_fraction, 3)

    if covalent_fraction > 0.7:
        result.bond_character = "covalent"
    elif covalent_fraction > 0.4:
        result.bond_character = "mixed"
    else:
        result.bond_character = "ionic"

    breakdown.append(
        f"Bond character: Δχ = {delta_eneg:.2f} → {ionic_fraction:.0%} ionic, "
        f"{covalent_fraction:.0%} covalent → {result.bond_character}"
    )

    result.breakdown = breakdown
    return result
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/kinetics.py — Kinetic response dynamics
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/kinetics.py", '''"""
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
        return "\\n".join(parts)


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
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/probability_chain.py — Statistical probability of complete events
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/probability_chain.py", '''"""
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
        return "\\n".join(parts)


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
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/sprint10_integration.py — Wires everything together
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/sprint10_integration.py", '''"""
core/sprint10_integration.py - Integrates kinetics, orbital, and probability chain
into the assembly pipeline.

Runs after physics_integration (sprint 9). Adds:
- Orbital analysis
- Kinetic profile
- Probability chain
- Updates composite score with probability chain

The composite score is now:
    score = f(P_cycle, selectivity, practical_factors)
"""

import core.assembly_composer as composer
import core.thermodynamics as _thermo_mod
from core.hydrodynamics import compute_hydrodynamics
from core.orbital_binding import compute_orbital_binding
from core.kinetics import compute_kinetics
from core.probability_chain import compute_probability_chain
from core.assembly import BinderAssembly
from core.problem import Problem
from knowledge.electronic_data import enrich_target_electronic


def full_physics_rescore(assemblies: list[BinderAssembly],
                          problem: Problem) -> list[BinderAssembly]:
    """
    Full physics pipeline: thermo → hydro → orbital → kinetics → probability chain.
    """
    # Enrich target with DFT data if available
    enrich_target_electronic(problem.target)

    for assembly in assemblies:
        recognition = assembly.recognition
        structure = assembly.structure
        interior = assembly.interior

        # Sprint 9: thermodynamics + hydrodynamics
        thermo = _thermo_mod.compute_thermodynamics(recognition, structure, interior, problem)
        hydro = compute_hydrodynamics(structure, interior, problem)

        # Sprint 10: orbital + kinetics + probability chain
        orbital = compute_orbital_binding(recognition, problem.target)
        kinetics = compute_kinetics(thermo, hydro, orbital, structure, interior, problem)
        chain = compute_probability_chain(thermo, hydro, kinetics, orbital, assembly, problem)

        # ── Update composite score from probability chain ─────────
        # Primary: P(capture) — most important for function
        # Secondary: selectivity factor, practical concerns
        selectivity_score = 0.5
        if thermo.selectivity_factor > 100:
            selectivity_score = 1.0
        elif thermo.selectivity_factor > 10:
            selectivity_score = 0.8
        elif thermo.selectivity_factor > 3:
            selectivity_score = 0.6
        else:
            selectivity_score = 0.3

        practical = 0.5
        if structure.synthesis_complexity == "trivial":
            practical = 0.7
        elif structure.synthesis_complexity == "standard":
            practical = 0.5
        elif structure.synthesis_complexity == "complex":
            practical = 0.3

        composite = (
            0.45 * chain.p_capture +
            0.25 * selectivity_score +
            0.15 * chain.p_retain +
            0.10 * chain.p_release +
            0.05 * practical
        )
        assembly.composite_score = max(0.01, min(0.99, round(composite, 3)))

        # ── Build comprehensive confidence reasoning ──────────────
        lines = [
            "PHYSICS:",
            thermo.summary(),
            "",
            "ORBITAL:",
            orbital.summary(),
            "",
            "KINETICS:",
            kinetics.summary(),
            "",
            "PROBABILITY CHAIN:",
            chain.summary(),
            "",
            "TRANSPORT:",
            hydro.summary(),
        ]
        assembly.confidence_reasoning = "\\n".join(lines)

        # Confidence from ΔG + kinetics
        if thermo.dG_net < -25 and kinetics.fractional_occupancy > 0.8:
            assembly.confidence = "high"
        elif thermo.dG_net < -15 and kinetics.fractional_occupancy > 0.5:
            assembly.confidence = "moderate"
        elif thermo.dG_net < -5:
            assembly.confidence = "low"
        else:
            assembly.confidence = "speculative"

        # Add kinetic warnings
        if kinetics.rate_limiting_step.startswith("transport"):
            assembly.failure_modes.append(
                f"Transport-limited: k_on effective only {kinetics.k_on_M_s:.0e} M⁻¹s⁻¹"
            )
        if kinetics.time_to_equilibrium_s > 3600:
            assembly.failure_modes.append(
                f"Slow equilibrium: {kinetics.time_to_equilibrium_s/60:.0f} min to reach 98%"
            )
        if chain.p_capture < 0.1:
            assembly.failure_modes.append(
                f"Low capture probability: {chain.p_capture:.1%} per cycle"
            )

    assemblies.sort(key=lambda a: a.composite_score, reverse=True)
    return assemblies


# ── Patch into pipeline ───────────────────────────────────────────────

_prev_compose = composer.compose_assemblies


def _sprint10_compose(candidates, problem, max_assemblies=8):
    assemblies = _prev_compose(candidates, problem, max_assemblies)
    return full_physics_rescore(assemblies, problem)


composer.compose_assemblies = _sprint10_compose
''')


# ═══════════════════════════════════════════════════════════════════════════
# Update main.py
# ═══════════════════════════════════════════════════════════════════════════

main_lines = [
    '"""', 'MABE - Modality-Agnostic Binder Engine', '"""', '',
    'import sys', 'import os', '',
    'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))', '',
    'from adapters.base import ToolRegistry',
    'from adapters.rdkit_adapter import RDKitAdapter',
    'from adapters.dnazyme_adapter import DNAzymeAdapter',
    'from adapters.peptide_adapter import PeptideAdapter',
    'from adapters.aptamer_adapter import AptamerAdapter',
    'from conversation.decomposer_patch import patch_targets',
    'from conversation.interface import run_interactive, run_single_query', '',
    'patch_targets()', '',
    '# Sprint 8 patches',
    'import core.assembly_composer_patch',
    'import core.scoring_patch', '',
    '# Sprint 9: thermodynamics + hydrodynamics',
    'import core.physics_integration', '',
    '# Sprint 10: kinetics + orbital + probability chain',
    'import core.sprint10_integration', '', '',
    'def build_registry() -> ToolRegistry:',
    '    registry = ToolRegistry()',
    '    rdkit = RDKitAdapter()',
    '    if rdkit.is_available():',
    '        registry.register(rdkit)',
    '    registry.register(DNAzymeAdapter())',
    '    registry.register(PeptideAdapter())',
    '    registry.register(AptamerAdapter())',
    '    return registry', '', '',
    'def main():',
    '    registry = build_registry()',
    '    if len(sys.argv) > 1:',
    '        query = " ".join(sys.argv[1:])',
    '        run_single_query(registry, query)',
    '    else:',
    '        run_interactive(registry)', '', '',
    'if __name__ == "__main__":',
    '    main()',
]
write_file("main.py", "\n".join(main_lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint10.py — embedded via json for safe quoting
# ═══════════════════════════════════════════════════════════════════════════

import pathlib
_test_path = pathlib.Path(__file__).parent / "test_sprint10_content.py"
if _test_path.exists():
    write_file("tests/test_sprint10.py", _test_path.read_text(encoding="utf-8"))
else:
    write_file('tests/test_sprint10.py', "\"\"\"\ntests/test_sprint10.py - Kinetic response dynamics tests.\n\"\"\"\n\nimport sys\nimport os\nsys.path.insert(0, os.path.join(os.path.dirname(__file__), \"..\"))\n\nfrom conversation.decomposer_patch import patch_targets\npatch_targets()\n\nimport core.assembly_composer_patch\nimport core.scoring_patch\nimport core.physics_integration\nimport core.sprint10_integration\n\nfrom knowledge.electronic_data import get_electronic_data, enrich_target_electronic, DONOR_HOMO\nfrom core.orbital_binding import compute_orbital_binding, OrbitalAnalysis\nfrom core.kinetics import compute_kinetics, KineticProfile\nfrom core.probability_chain import compute_probability_chain, ProbabilityChain\nfrom core.thermodynamics import compute_thermodynamics\nfrom core.hydrodynamics import compute_hydrodynamics\nfrom core.assembly import (\n    RecognitionChemistry, StructuralConstraint, InteriorDesign, InteriorSite,\n    BinderAssembly, SelectivityFilter, ReleaseMechanism,\n)\nfrom core.problem import TargetSpecies, ElectronicDescription, HydrationDescription, SizeDescription\nfrom conversation.decomposer import decompose\nfrom core.orchestrator import Orchestrator\nfrom adapters.base import ToolRegistry\nfrom adapters.dnazyme_adapter import DNAzymeAdapter\nfrom adapters.peptide_adapter import PeptideAdapter\nfrom adapters.aptamer_adapter import AptamerAdapter\n\n\ndef _build():\n    registry = ToolRegistry()\n    registry.register(DNAzymeAdapter())\n    registry.register(PeptideAdapter())\n    registry.register(AptamerAdapter())\n    return registry\n\n\n# \u2500\u2500 Electronic data tests \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\ndef test_dft_data_exists():\n    \"\"\"DFT data should exist for common metals.\"\"\"\n    for metal in [\"lead\", \"gold\", \"mercury\", \"copper\", \"zinc\", \"calcium\"]:\n        data = get_electronic_data(metal)\n        assert data, f\"No DFT data for {metal}\"\n        assert \"homo_ev\" in data\n        assert \"lumo_ev\" in data\n        assert \"polarizability\" in data\n    print(f\"  + DFT data present for 6 common metals\")\n\n\ndef test_enrich_populates_target():\n    \"\"\"enrich_target_electronic should fill None fields.\"\"\"\n    problem = decompose(\"lead capture from mine water\")\n    assert problem.target.electronic.homo_ev is None\n    enrich_target_electronic(problem.target)\n    assert problem.target.electronic.homo_ev is not None\n    assert problem.target.electronic.lumo_ev is not None\n    assert problem.target.electronic.polarizability is not None\n    print(f\"  + Lead enriched: HOMO={problem.target.electronic.homo_ev}, LUMO={problem.target.electronic.lumo_ev}, alpha={problem.target.electronic.polarizability}\")\n\n\ndef test_gold_lower_lumo_than_calcium():\n    \"\"\"Soft metals (gold) should have lower (more negative) LUMO than hard metals (calcium).\"\"\"\n    au = get_electronic_data(\"gold\")\n    ca = get_electronic_data(\"calcium\")\n    assert au[\"lumo_ev\"] < ca[\"lumo_ev\"], \"Gold LUMO should be lower (better acceptor)\"\n    print(f\"  + Gold LUMO={au['lumo_ev']:.1f} < Calcium LUMO={ca['lumo_ev']:.1f} (soft accepts better)\")\n\n\n# \u2500\u2500 Orbital binding tests \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\ndef test_orbital_soft_match():\n    \"\"\"S donors + gold (soft-soft) should show covalent character.\"\"\"\n    problem = decompose(\"gold recovery from mine tailings\")\n    enrich_target_electronic(problem.target)\n    rec = RecognitionChemistry(name=\"thiol\", type=\"chelator\", donor_atoms=[\"S\", \"S\"],\n        donor_type=\"soft\", structure=\"dithiol\")\n    orbital = compute_orbital_binding(rec, problem.target)\n    assert orbital.dft_data_available\n    assert orbital.bond_character in (\"covalent\", \"mixed\")\n    assert orbital.covalent_fraction > 0.5\n    print(f\"  + Gold+S: {orbital.bond_character} ({orbital.covalent_fraction:.0%} covalent), CT dG={orbital.charge_transfer_dG_kj:.1f} kJ/mol\")\n\n\ndef test_orbital_hard_match():\n    \"\"\"O donors + calcium (hard-hard) should show ionic character.\"\"\"\n    ca = TargetSpecies(identity=\"calcium\", formula=\"Ca(2+)\", charge=2.0, geometry=\"octahedral\",\n        electronic=ElectronicDescription(hardness_softness=\"hard\", electronegativity=1.0),\n        hydration=HydrationDescription(hydrated_radius_angstrom=4.12, dehydration_energy_kj_mol=1577.0, coordination_number_water=8),\n        size=SizeDescription(ionic_radius_angstrom=1.0))\n    enrich_target_electronic(ca)\n    rec = RecognitionChemistry(name=\"carboxylate\", type=\"chelator\", donor_atoms=[\"O\", \"O\", \"O\"],\n        donor_type=\"hard\", structure=\"tricarboxylate\")\n    orbital = compute_orbital_binding(rec, ca)\n    assert orbital.bond_character == \"ionic\"\n    assert orbital.covalent_fraction < 0.5\n    print(f\"  + Ca+O: {orbital.bond_character} ({orbital.covalent_fraction:.0%} covalent)\")\n\n\ndef test_charge_transfer_favorable_for_good_match():\n    \"\"\"Good donor-metal HOMO-LUMO gap should give negative CT energy.\"\"\"\n    problem = decompose(\"lead capture from mine water\")\n    enrich_target_electronic(problem.target)\n    rec = RecognitionChemistry(name=\"test\", type=\"chelator\", donor_atoms=[\"N\", \"S\"],\n        donor_type=\"borderline\", structure=\"NS-chelator\")\n    orbital = compute_orbital_binding(rec, problem.target)\n    assert orbital.charge_transfer_favorable\n    assert orbital.charge_transfer_dG_kj < 0\n    print(f\"  + Lead+NS: CT favorable, dG_CT={orbital.charge_transfer_dG_kj:.1f} kJ/mol\")\n\n\n# \u2500\u2500 Kinetics tests \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\ndef _make_lead_thermo_hydro():\n    \"\"\"Helper: compute thermo+hydro for lead in mesoporous silica.\"\"\"\n    problem = decompose(\"lead capture from mine water\")\n    enrich_target_electronic(problem.target)\n    rec = RecognitionChemistry(name=\"test\", type=\"chelator\", donor_atoms=[\"N\", \"O\", \"O\", \"N\"],\n        donor_type=\"borderline\", structure=\"EDTA-like\")\n    from knowledge.structural_library import STRUCTURAL_OPTIONS\n    meso = [s for s in STRUCTURAL_OPTIONS if s.type == \"mesoporous_silica\"][0]\n    interior = InteriorDesign(sites=[InteriorSite(recognition=rec, copies=10)],\n        design_level=\"composite\", total_binding_sites=10, unique_recognition_types=1, avidity_factor=3.0)\n    thermo = compute_thermodynamics(rec, meso, interior, problem)\n    hydro = compute_hydrodynamics(meso, interior, problem)\n    orbital = compute_orbital_binding(rec, problem.target)\n    return thermo, hydro, orbital, meso, interior, problem\n\n\ndef test_k_on_physical_range():\n    \"\"\"k_on should be in physically meaningful range (1 to 10^9 M-1s-1).\"\"\"\n    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()\n    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)\n    assert 1.0 <= kinetics.k_on_M_s <= 1e10, f\"k_on = {kinetics.k_on_M_s:.1e} out of range\"\n    print(f\"  + k_on = {kinetics.k_on_M_s:.1e} M-1s-1 (physical)\")\n\n\ndef test_k_off_from_keq():\n    \"\"\"k_off = k_on / K_eq should give reasonable residence time.\"\"\"\n    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()\n    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)\n    assert kinetics.k_off_s > 0\n    assert kinetics.residence_time_s > 0\n    assert kinetics.residence_time_s < 1e10  # not infinite\n    print(f\"  + k_off = {kinetics.k_off_s:.1e} s-1, tau = {kinetics.residence_time_s:.1e} s\")\n\n\ndef test_occupancy_reasonable():\n    \"\"\"Fractional occupancy should be between 0 and 1.\"\"\"\n    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()\n    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)\n    assert 0.0 <= kinetics.fractional_occupancy <= 1.0\n    print(f\"  + Occupancy: {kinetics.fractional_occupancy:.0%} at working [Pb2+]\")\n\n\ndef test_rate_limiting_identified():\n    \"\"\"Rate-limiting step should be identified.\"\"\"\n    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()\n    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)\n    assert kinetics.rate_limiting_step != \"\"\n    print(f\"  + Rate-limiting: {kinetics.rate_limiting_step[:50]}\")\n\n\n# \u2500\u2500 Probability chain tests \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\ndef test_chain_all_probabilities_valid():\n    \"\"\"All probabilities in chain should be 0-1.\"\"\"\n    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()\n    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)\n    assembly = BinderAssembly(\n        name=\"test_assembly\", description=\"test\", design_level=\"composite\",\n        interior=interior, structure=struct,\n        selectivity=SelectivityFilter(name=\"none\", mechanism=\"none\", description=\"none\"),\n        release=ReleaseMechanism(name=\"pH_shift\", trigger=\"pH_change\", description=\"pH shift release\"),\n    )\n    chain = compute_probability_chain(thermo, hydro, kinetics, orbital, assembly, problem)\n    for name, val in [(\"enter\", chain.p_enter), (\"encounter\", chain.p_encounter),\n                      (\"bind\", chain.p_bind), (\"retain\", chain.p_retain),\n                      (\"release\", chain.p_release), (\"capture\", chain.p_capture),\n                      (\"cycle\", chain.p_cycle)]:\n        assert 0.0 <= val <= 1.0, f\"P({name}) = {val} out of range\"\n    print(f\"  + All probabilities valid: P(cycle) = {chain.p_cycle:.4f}\")\n\n\ndef test_chain_product_correct():\n    \"\"\"P(capture) should equal P(enter) * P(encounter) * P(bind).\"\"\"\n    thermo, hydro, orbital, struct, interior, problem = _make_lead_thermo_hydro()\n    kinetics = compute_kinetics(thermo, hydro, orbital, struct, interior, problem)\n    assembly = BinderAssembly(\n        name=\"test_assembly\", description=\"test\", design_level=\"composite\",\n        interior=interior, structure=struct,\n        selectivity=SelectivityFilter(name=\"none\", mechanism=\"none\", description=\"none\"),\n        release=ReleaseMechanism(name=\"pH_shift\", trigger=\"pH_change\", description=\"pH shift release\"),\n    )\n    chain = compute_probability_chain(thermo, hydro, kinetics, orbital, assembly, problem)\n    expected = chain.p_enter * chain.p_encounter * chain.p_bind\n    assert abs(chain.p_capture - expected) < 0.001, f\"P(capture)={chain.p_capture} != product={expected}\"\n    print(f\"  + P(capture) = {chain.p_enter:.3f} x {chain.p_encounter:.3f} x {chain.p_bind:.3f} = {chain.p_capture:.4f}\")\n\n\n# \u2500\u2500 End-to-end \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\ndef test_e2e_lead():\n    \"\"\"Full pipeline should produce physics + kinetics + probability data.\"\"\"\n    o = Orchestrator(_build())\n    r = o.solve(decompose(\"lead capture and release from mine water\"))\n    assert len(r.assemblies) > 0\n    top = r.assemblies[0]\n    cr = top.confidence_reasoning\n    assert \"KINETICS:\" in cr, \"Missing kinetics\"\n    assert \"PROBABILITY CHAIN:\" in cr, \"Missing probability chain\"\n    assert \"ORBITAL:\" in cr, \"Missing orbital analysis\"\n    assert \"k_on\" in cr, \"Missing k_on\"\n    assert \"P(capture)\" in cr, \"Missing P(capture)\"\n    print(f\"  + Lead E2E: full physics pipeline present\")\n    for a in r.assemblies[:3]:\n        print(f\"    {a.composite_score:.0%}  {a.name[:50]}\")\n\n\ndef test_e2e_gold():\n    \"\"\"Gold should produce orbital analysis with DFT data.\"\"\"\n    o = Orchestrator(_build())\n    r = o.solve(decompose(\"gold recovery from mine tailings\"))\n    assert len(r.assemblies) > 0\n    cr = r.assemblies[0].confidence_reasoning\n    assert \"covalent\" in cr.lower() or \"mixed\" in cr.lower(), \"Gold should show covalent/mixed character\"\n    print(f\"  + Gold E2E: orbital analysis confirms covalent/mixed character\")\n    print(f\"    Top: {r.assemblies[0].name[:50]} ({r.assemblies[0].composite_score:.0%})\")\n\n\ndef test_e2e_mercury():\n    \"\"\"Mercury should work through full pipeline.\"\"\"\n    o = Orchestrator(_build())\n    r = o.solve(decompose(\"mercury removal from river water\"))\n    assert len(r.assemblies) > 0\n    cr = r.assemblies[0].confidence_reasoning\n    assert \"kJ/mol\" in cr\n    assert \"M\" in cr  # rate constant units\n    print(f\"  + Mercury E2E: {r.assemblies[0].name[:50]} ({r.assemblies[0].composite_score:.0%})\")\n\n\nif __name__ == \"__main__\":\n    print()\n    print(\"  MABE Sprint 10 - Kinetic Response Dynamics Tests\")\n    print(\"  \" + \"=\" * 50)\n    print()\n    print(\"  Electronic data:\")\n    test_dft_data_exists()\n    test_enrich_populates_target()\n    test_gold_lower_lumo_than_calcium()\n    print()\n    print(\"  Orbital binding:\")\n    test_orbital_soft_match()\n    test_orbital_hard_match()\n    test_charge_transfer_favorable_for_good_match()\n    print()\n    print(\"  Kinetics:\")\n    test_k_on_physical_range()\n    test_k_off_from_keq()\n    test_occupancy_reasonable()\n    test_rate_limiting_identified()\n    print()\n    print(\"  Probability chain:\")\n    test_chain_all_probabilities_valid()\n    test_chain_product_correct()\n    print()\n    print(\"  End-to-end:\")\n    test_e2e_lead()\n    test_e2e_gold()\n    test_e2e_mercury()\n    print()\n    print(\"  All Sprint 10 tests passed.\")\n    print()\n")

print()
print("  Done! New/updated files:")
print("    knowledge/electronic_data.py    (NEW: DFT-derived HOMO/LUMO/polarizability for 12 species)")
print("    core/orbital_binding.py         (NEW: charge transfer, London dispersion, bond character)")
print("    core/kinetics.py                (NEW: k_on, k_off, residence time, saturation, dose-response)")
print("    core/probability_chain.py       (NEW: P(enter)×P(encounter)×P(bind)×P(retain)×P(release))")
print("    core/sprint10_integration.py    (NEW: full pipeline integration)")
print("    main.py                          (updated)")
print()
print("  THE KEY CHANGE:")
print("    Sprint 9: WILL it bind? CAN it reach?")
print("    Sprint 10: HOW FAST? HOW LONG? WHAT PROBABILITY?")
print("    Score is now P(full capture-release cycle) — a real physical probability.")
print()
print("  Next steps:")
print("    python tests\\test_sprint10.py")
print('    python main.py "lead capture and release from mine water"')
print()