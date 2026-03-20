"""
core/responsive_multiphysics.py — Responsive Multi-Physics Structures

Continuous temperature-dependent model for thermoresponsive colloidal
assemblies where a single stimulus (temperature) simultaneously changes:
  - Structural color (inter-particle spacing → peak wavelength shift)
  - Thermal conductivity (matrix properties + porosity change)
  - Acoustic absorption (porosity + flow resistivity change)

Primary system: PNIPAM (poly(N-isopropylacrylamide)) hydrogel matrix
with colloidal particles (SiO₂, PS).

PNIPAM physics:
  LCST = 32°C (lower critical solution temperature)
  Below LCST: hydrogel swollen with water, large inter-particle spacing
  Above LCST: hydrogel collapsed (hydrophobic), small spacing
  Transition width: ~2-5°C (not a sharp step)

The swelling ratio Q(T) is the master variable. Everything else follows:
  D_eff(T) = D_dry × Q(T)^(1/3)     → color shift
  φ_eff(T) = φ_dry / Q(T)           → thermal + acoustic change
  κ_matrix(T) interpolates water↔polymer → thermal change
  ε(T) = 1 - φ_eff(T)              → acoustic change

Phase 5 of the Multi-Physics Structural Element module.

Key references:
  - Schild, Prog. Polym. Sci. 1992, 17, 163 (PNIPAM review)
  - Takeoka et al., Angew. Chem. 2003, 42, 3246 (PNIPAM photonic crystal)
  - Weissman et al., Science 1996, 274, 959 (tunable colloidal crystal)
  - Hu et al., Adv. Mater. 2012, 24, OP131 (angle-independent responsive color)

Data tier: Tier 2 (published swelling curves + DOI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.colloidal_thermal import (
    predict_from_optical as thermal_predict,
    ThermalResult,
    _MATERIALS,
)
from core.colloidal_acoustic import (
    predict_from_optical as acoustic_predict,
    AcousticResult,
)

# Optical — soft dependency
try:
    import numpy as np
    from core.multiphysics_pareto import (
        _predict_color,
        ColorTarget,
        ColorResult,
        photonic_glass_peak_analytical,
    )
    _PARETO_AVAILABLE = True
except ImportError:
    _PARETO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# PNIPAM swelling model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PNIPAMConfig:
    """Configuration for PNIPAM hydrogel swelling behavior."""
    lcst_c: float = 32.0                # lower critical solution temperature (°C)
    transition_width_c: float = 3.0      # width of the swelling transition
    q_swollen: float = 8.0               # volumetric swelling ratio below LCST
    q_collapsed: float = 1.2             # volumetric swelling ratio above LCST
    crosslink_density: str = "standard"  # "low", "standard", "high"

    def __post_init__(self):
        # Adjust swelling ratio by crosslink density
        if self.crosslink_density == "low":
            self.q_swollen = min(self.q_swollen * 1.5, 15.0)
        elif self.crosslink_density == "high":
            self.q_swollen = max(self.q_swollen * 0.6, 2.0)


def swelling_ratio(
    temperature_c: float,
    config: Optional[PNIPAMConfig] = None,
) -> float:
    """Volumetric swelling ratio Q(T) for PNIPAM hydrogel.

    Uses a sigmoidal model fitted to experimental swelling curves:
      Q(T) = Q_collapsed + (Q_swollen - Q_collapsed) / (1 + exp((T - LCST) / w))

    where w controls transition sharpness.

    Ref: Schild, Prog. Polym. Sci. 1992; Hu et al., Adv. Mater. 2012.
    """
    if config is None:
        config = PNIPAMConfig()

    Q_s = config.q_swollen
    Q_c = config.q_collapsed
    T_lcst = config.lcst_c
    w = config.transition_width_c

    # Sigmoidal: high Q below LCST, low Q above
    exponent = (temperature_c - T_lcst) / w
    # Clamp to prevent overflow
    exponent = max(-50.0, min(50.0, exponent))
    Q = Q_c + (Q_s - Q_c) / (1.0 + math.exp(exponent))

    return Q


def effective_diameter(
    dry_diameter_nm: float,
    temperature_c: float,
    config: Optional[PNIPAMConfig] = None,
) -> float:
    """Effective inter-particle spacing (≈ diameter) at temperature T.

    D_eff = D_dry × Q(T)^(1/3)

    In a PNIPAM-embedded colloidal assembly, the hydrogel swelling
    pushes particles apart, increasing the effective lattice spacing.
    """
    Q = swelling_ratio(temperature_c, config)
    return dry_diameter_nm * Q ** (1.0 / 3.0)


def effective_volume_fraction(
    dry_volume_fraction: float,
    temperature_c: float,
    config: Optional[PNIPAMConfig] = None,
) -> float:
    """Effective volume fraction at temperature T.

    φ_eff = φ_dry / Q(T)

    As hydrogel swells, the same number of particles occupies more volume,
    so volume fraction decreases.
    """
    Q = swelling_ratio(temperature_c, config)
    phi = dry_volume_fraction / Q
    return max(0.01, min(0.74, phi))  # physical bounds


def effective_matrix_properties(
    temperature_c: float,
    config: Optional[PNIPAMConfig] = None,
) -> dict:
    """Interpolated matrix thermal properties between swollen and collapsed.

    Below LCST: matrix is mostly water (high κ, high ρ)
    Above LCST: matrix is mostly polymer (low κ, low ρ)
    Transition: linear interpolation weighted by swelling ratio.
    """
    if config is None:
        config = PNIPAMConfig()

    Q = swelling_ratio(temperature_c, config)
    Q_s = config.q_swollen
    Q_c = config.q_collapsed

    # Fraction toward collapsed state (0 = fully swollen, 1 = fully collapsed)
    if Q_s - Q_c > 0:
        f_collapsed = 1.0 - (Q - Q_c) / (Q_s - Q_c)
    else:
        f_collapsed = 0.5
    f_collapsed = max(0.0, min(1.0, f_collapsed))

    swollen = _MATERIALS["PNIPAM_swollen"]
    collapsed = _MATERIALS["PNIPAM_collapsed"]

    def _lerp(a, b, t):
        return a + (b - a) * t

    return {
        "kappa_W_mK": _lerp(swollen.kappa_W_mK, collapsed.kappa_W_mK, f_collapsed),
        "density_kg_m3": _lerp(swollen.density_kg_m3, collapsed.density_kg_m3, f_collapsed),
        "v_sound_m_s": _lerp(swollen.v_sound_m_s, collapsed.v_sound_m_s, f_collapsed),
        "cp_J_kgK": _lerp(swollen.cp_J_kgK, collapsed.cp_J_kgK, f_collapsed),
        "matrix_label": "PNIPAM_swollen" if f_collapsed < 0.5 else "PNIPAM_collapsed",
        "f_collapsed": round(f_collapsed, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResponsiveSpec:
    """Specification for a responsive colloidal assembly."""
    # Particle (dry state)
    particle_material: str = "SiO2"
    dry_diameter_nm: float = 250.0       # particle D in collapsed/dry hydrogel
    dry_volume_fraction: float = 0.50    # φ in collapsed state

    # Film
    film_thickness_um: float = 100.0     # total film thickness (dry)

    # PNIPAM configuration
    pnipam: PNIPAMConfig = field(default_factory=PNIPAMConfig)

    # Color target (optional — for ΔE tracking)
    color_target: Optional[ColorTarget] = None


@dataclass
class ResponsiveState:
    """Multi-physics state at one temperature."""
    temperature_c: float

    # Effective parameters
    swelling_ratio: float
    effective_diameter_nm: float
    effective_volume_fraction: float
    effective_porosity: float
    matrix_f_collapsed: float

    # Color
    peak_wavelength_nm: float = 0.0
    color: Optional[ColorResult] = None

    # Thermal
    thermal: Optional[ThermalResult] = None

    # Acoustic
    acoustic: Optional[AcousticResult] = None

    def summary(self) -> str:
        lines = [
            f"  T = {self.temperature_c:.1f}°C | Q = {self.swelling_ratio:.2f} | "
            f"D_eff = {self.effective_diameter_nm:.0f} nm | "
            f"φ_eff = {self.effective_volume_fraction:.3f} | "
            f"ε = {self.effective_porosity:.3f}",
        ]
        lines.append(f"    λ_peak = {self.peak_wavelength_nm:.0f} nm")
        if self.thermal:
            lines.append(f"    κ = {self.thermal.kappa_eff_W_mK:.4f} W/mK | "
                         f"R = {self.thermal.R_value_m2KW:.4f} m²K/W")
        if self.acoustic:
            lines.append(f"    NRC = {self.acoustic.nrc:.2f}")
        return "\n".join(lines)


@dataclass
class ResponsiveSweep:
    """Complete temperature sweep of a responsive structure."""
    spec: ResponsiveSpec
    states: list[ResponsiveState]
    temperatures_c: list[float]

    # Derived ranges
    wavelength_range_nm: tuple[float, float] = (0.0, 0.0)
    kappa_range_W_mK: tuple[float, float] = (0.0, 0.0)
    nrc_range: tuple[float, float] = (0.0, 0.0)

    # Transition characterization
    color_shift_nm: float = 0.0          # total wavelength shift across transition
    thermal_switch_ratio: float = 1.0    # κ_hot / κ_cold
    lcst_c: float = 32.0

    def summary(self) -> str:
        lines = [
            f"Responsive Multi-Physics Sweep",
            f"  Particle: {self.spec.dry_diameter_nm:.0f} nm {self.spec.particle_material} "
            f"(dry φ = {self.spec.dry_volume_fraction:.2f})",
            f"  LCST: {self.lcst_c:.1f}°C",
            f"  Temperature range: {self.temperatures_c[0]:.0f} – {self.temperatures_c[-1]:.0f}°C",
            f"  Color shift: {self.color_shift_nm:.0f} nm "
            f"({self.wavelength_range_nm[0]:.0f} – {self.wavelength_range_nm[1]:.0f} nm)",
            f"  Thermal switch: {self.thermal_switch_ratio:.2f}× "
            f"(κ: {self.kappa_range_W_mK[0]:.4f} – {self.kappa_range_W_mK[1]:.4f} W/mK)",
            f"  NRC range: {self.nrc_range[0]:.3f} – {self.nrc_range[1]:.3f}",
            "",
        ]

        # Show key states
        lines.append("Key states:")
        for s in self.states:
            if abs(s.temperature_c - 20.0) < 0.5 or \
               abs(s.temperature_c - self.lcst_c) < 0.5 or \
               abs(s.temperature_c - 40.0) < 0.5:
                lines.append(s.summary())

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Core prediction: state at one temperature
# ═══════════════════════════════════════════════════════════════════════════

def predict_at_temperature(
    spec: ResponsiveSpec,
    temperature_c: float,
) -> ResponsiveState:
    """Predict all three physics at one temperature."""

    Q = swelling_ratio(temperature_c, spec.pnipam)
    D_eff = effective_diameter(spec.dry_diameter_nm, temperature_c, spec.pnipam)
    phi_eff = effective_volume_fraction(spec.dry_volume_fraction, temperature_c, spec.pnipam)
    eps_eff = 1.0 - phi_eff
    mat_props = effective_matrix_properties(temperature_c, spec.pnipam)

    # ── Color ──
    lam_peak = photonic_glass_peak_analytical(
        D_eff, spec.particle_material, phi_eff
    ) if _PARETO_AVAILABLE else 2.0 * 1.3 * D_eff  # crude fallback

    color_result = None
    if _PARETO_AVAILABLE and spec.color_target:
        color_result = _predict_color(
            D_eff, spec.particle_material, phi_eff, 0.005, spec.color_target)

    # ── Thermal ──
    # Use whichever PNIPAM label is closer to current state
    matrix_label = mat_props["matrix_label"]
    thermal_result = thermal_predict(
        D_eff, spec.particle_material, phi_eff,
        matrix_label, spec.film_thickness_um,
    )

    # ── Acoustic ──
    acoustic_result = acoustic_predict(
        D_eff, phi_eff, spec.film_thickness_um,
    )

    return ResponsiveState(
        temperature_c=temperature_c,
        swelling_ratio=round(Q, 4),
        effective_diameter_nm=round(D_eff, 2),
        effective_volume_fraction=round(phi_eff, 4),
        effective_porosity=round(eps_eff, 4),
        matrix_f_collapsed=mat_props["f_collapsed"],
        peak_wavelength_nm=round(lam_peak, 1),
        color=color_result,
        thermal=thermal_result,
        acoustic=acoustic_result,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Temperature sweep
# ═══════════════════════════════════════════════════════════════════════════

def sweep_temperature(
    spec: ResponsiveSpec,
    t_min_c: float = 15.0,
    t_max_c: float = 45.0,
    t_step_c: float = 1.0,
) -> ResponsiveSweep:
    """Sweep temperature and predict coupled multi-physics response.

    Returns a ResponsiveSweep with states at each temperature step.
    """
    temperatures = []
    t = t_min_c
    while t <= t_max_c + 0.01:
        temperatures.append(round(t, 2))
        t += t_step_c

    states = []
    for t in temperatures:
        state = predict_at_temperature(spec, t)
        states.append(state)

    if not states:
        return ResponsiveSweep(spec, [], [], lcst_c=spec.pnipam.lcst_c)

    # Extract ranges
    wavelengths = [s.peak_wavelength_nm for s in states]
    kappas = [s.thermal.kappa_eff_W_mK for s in states if s.thermal]
    nrcs = [s.acoustic.nrc for s in states if s.acoustic]

    lam_range = (min(wavelengths), max(wavelengths))
    kap_range = (min(kappas), max(kappas)) if kappas else (0.0, 0.0)
    nrc_range = (min(nrcs), max(nrcs)) if nrcs else (0.0, 0.0)

    color_shift = lam_range[1] - lam_range[0]
    thermal_ratio = kap_range[1] / kap_range[0] if kap_range[0] > 0 else 1.0

    return ResponsiveSweep(
        spec=spec,
        states=states,
        temperatures_c=temperatures,
        wavelength_range_nm=lam_range,
        kappa_range_W_mK=kap_range,
        nrc_range=nrc_range,
        color_shift_nm=round(color_shift, 1),
        thermal_switch_ratio=round(thermal_ratio, 4),
        lcst_c=spec.pnipam.lcst_c,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def design_responsive(
    particle_material: str = "SiO2",
    dry_diameter_nm: float = 250.0,
    dry_volume_fraction: float = 0.50,
    film_thickness_um: float = 100.0,
    lcst_c: float = 32.0,
    transition_width_c: float = 3.0,
    q_swollen: float = 8.0,
    q_collapsed: float = 1.2,
    t_min_c: float = 15.0,
    t_max_c: float = 45.0,
    t_step_c: float = 1.0,
) -> ResponsiveSweep:
    """Design a responsive multi-physics structure with custom parameters.

    Example:
        sweep = design_responsive(dry_diameter_nm=250, dry_volume_fraction=0.50)
        print(sweep.summary())
        # Shows color shift, thermal switch ratio, NRC range across 15-45°C
    """
    spec = ResponsiveSpec(
        particle_material=particle_material,
        dry_diameter_nm=dry_diameter_nm,
        dry_volume_fraction=dry_volume_fraction,
        film_thickness_um=film_thickness_um,
        pnipam=PNIPAMConfig(
            lcst_c=lcst_c,
            transition_width_c=transition_width_c,
            q_swollen=q_swollen,
            q_collapsed=q_collapsed,
        ),
    )
    return sweep_temperature(spec, t_min_c, t_max_c, t_step_c)


def compare_crosslink_densities(
    dry_diameter_nm: float = 250.0,
    dry_volume_fraction: float = 0.50,
) -> dict[str, ResponsiveSweep]:
    """Compare low/standard/high crosslink density PNIPAM.

    Higher crosslink density → less swelling → smaller color shift
    but faster response time and better mechanical stability.
    """
    results = {}
    for density in ("low", "standard", "high"):
        spec = ResponsiveSpec(
            dry_diameter_nm=dry_diameter_nm,
            dry_volume_fraction=dry_volume_fraction,
            pnipam=PNIPAMConfig(crosslink_density=density),
        )
        results[density] = sweep_temperature(spec)
    return results
