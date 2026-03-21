"""
core/switchable_window.py -- Switchable opacity for smart window coatings.

Extends smart_window.py with active layers that modulate transparency:
  1. Conductive layer (TCO/thin metal) — enables electrochromic switching
  2. Electrochromic (WO3/NiO) — voltage-driven clear↔dark
  3. Photochromic (AgCl/WO3 NP) — UV-activated darkening
  4. Magnetochromic (Fe3O4 chain) — magnetic field shifts Bragg peak

Physics:
  Drude:     R_sheet = rho / t, T = exp(-alpha*t)
  Faraday:   Q = CE * delta_OD, T_dark = T_clear * 10^(-delta_OD)
  Kinetics:  tau_switch = R_sheet * C_ionic * A
  Langevin:  L(x) = coth(x) - 1/x, x = mu*B/(kT)
  Beer-Lambert: photochromic darkening

Entry points:
  design_conductive(target_R_sq) -> ConductiveLayer
  design_electrochromic(T_clear, T_dark, ...) -> ElectrochromicLayer
  design_photochromic(...) -> PhotochromicLayer
  design_magnetochromic(...) -> MagnetochromicLayer
  design_switchable_window(targets) -> SwitchableWindowDesign
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

K_BOLTZMANN = 1.381e-23   # J/K
FARADAY = 96485.0          # C/mol
ROOM_TEMP_K = 300.0
KT_ROOM = K_BOLTZMANN * ROOM_TEMP_K  # ~4.14e-21 J


# ---------------------------------------------------------------------------
# TCO material database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TCOMaterial:
    """Transparent conducting oxide properties."""
    name: str
    resistivity_ohm_cm: float   # bulk resistivity
    alpha_vis_cm: float          # VIS absorption coefficient
    n_vis: float                 # refractive index at 550nm
    max_thickness_nm: float      # practical deposition limit
    cost_relative: float         # 1.0 = ITO baseline
    indium_free: bool


TCO_MATERIALS: Dict[str, TCOMaterial] = {
    "ITO":  TCOMaterial("ITO",  2e-4, 2000, 1.90, 500, 1.0, False),
    "FTO":  TCOMaterial("FTO",  5e-4, 3000, 1.95, 600, 0.3, True),
    "AZO":  TCOMaterial("AZO",  4e-4, 2500, 1.90, 400, 0.2, True),
    "Ag_mesh": TCOMaterial("Ag_mesh", 1e-5, 500, 0.13, 50, 0.5, True),
}


# ---------------------------------------------------------------------------
# Conductive layer
# ---------------------------------------------------------------------------

@dataclass
class ConductiveLayer:
    """Transparent conductive layer for electrochromic switching."""
    material: str
    thickness_nm: float
    sheet_resistance_ohm_sq: float
    vis_transmittance: float
    resistivity: float
    indium_free: bool
    notes: str = ""


def design_conductive(
    target_R_sq: float = 15.0,
    prefer_indium_free: bool = True,
) -> ConductiveLayer:
    """Design TCO layer meeting sheet resistance target.

    R_sheet = rho / t
    T = exp(-alpha * t)

    Lower R needs thicker film, which reduces transmittance.
    """
    # Pick material
    if prefer_indium_free:
        candidates = [(n, m) for n, m in TCO_MATERIALS.items() if m.indium_free]
    else:
        candidates = list(TCO_MATERIALS.items())

    best = None
    best_T = 0.0

    for name, mat in candidates:
        # Required thickness: t = rho / R_sheet
        t_cm = mat.resistivity_ohm_cm / target_R_sq
        t_nm = t_cm / 1e-7

        if t_nm > mat.max_thickness_nm:
            continue  # can't get low enough R with this material
        if t_nm < 10:
            t_nm = 10  # practical minimum

        T = math.exp(-mat.alpha_vis_cm * t_nm * 1e-7)
        actual_R = mat.resistivity_ohm_cm / (t_nm * 1e-7)

        if T > best_T:
            best = (name, mat, t_nm, T, actual_R)
            best_T = T

    if best is None:
        # Fallback to ITO
        mat = TCO_MATERIALS["ITO"]
        t_nm = min(mat.max_thickness_nm, mat.resistivity_ohm_cm / (target_R_sq * 1e-7))
        T = math.exp(-mat.alpha_vis_cm * t_nm * 1e-7)
        actual_R = mat.resistivity_ohm_cm / (t_nm * 1e-7)
        best = ("ITO", mat, t_nm, T, actual_R)

    name, mat, t_nm, T, actual_R = best

    return ConductiveLayer(
        material=name,
        thickness_nm=round(t_nm, 0),
        sheet_resistance_ohm_sq=round(actual_R, 1),
        vis_transmittance=round(T, 3),
        resistivity=mat.resistivity_ohm_cm,
        indium_free=mat.indium_free,
        notes=f"{name} {t_nm:.0f}nm, R_sq={actual_R:.1f} Ohm/sq",
    )


# ---------------------------------------------------------------------------
# Electrochromic layer
# ---------------------------------------------------------------------------

@dataclass
class ElectrochromicLayer:
    """WO3/NiO electrochromic layer pair."""
    T_clear: float                # transmittance in clear state
    T_dark: float                 # transmittance in dark state
    delta_OD: float               # optical density change
    coloration_efficiency: float  # cm2/C
    charge_density_mC_cm2: float  # charge needed per area
    switching_voltage_V: float
    switching_time_s: float       # for given area and conductor
    conductor_R_sq: float         # sheet resistance of conductor
    area_cm2: float
    thickness_wo3_nm: float
    thickness_nio_nm: float
    cycle_life: int = 50000       # typical
    notes: str = ""

    @property
    def modulation_range(self) -> float:
        """Transmittance swing."""
        return self.T_clear - self.T_dark


def design_electrochromic(
    T_clear: float = 0.65,
    T_dark: float = 0.05,
    conductor_R_sq: float = 15.0,
    area_m2: float = 1.0,
) -> ElectrochromicLayer:
    """Design electrochromic layer for target clear/dark transmittance.

    Physics:
      delta_OD = -log10(T_dark / T_clear)
      Q = delta_OD / CE  (charge per cm2)
      Switching time ~ R_sheet * C_ionic * Area

    WO3 coloration efficiency: 40-60 cm2/C (state of art ~120 cm2/C)
    """
    CE = 50.0  # cm2/C, conservative WO3

    # Required optical density change
    if T_dark <= 0 or T_clear <= 0:
        delta_OD = 2.0
    else:
        delta_OD = -math.log10(T_dark / T_clear)

    # Charge per area
    Q_C_cm2 = delta_OD / CE
    Q_mC_cm2 = Q_C_cm2 * 1000

    # WO3 thickness: ~300-500nm typical for full switching
    # Thicker = more charge capacity = deeper coloration
    t_wo3 = 300 + delta_OD * 100  # nm, scales with required OD
    t_wo3 = min(600, max(200, t_wo3))

    # NiO counter-electrode: ~200-300nm
    t_nio = t_wo3 * 0.6

    # Switching voltage: 1-3V for Li+ intercalation
    voltage = 1.0 + delta_OD * 0.5
    voltage = min(3.0, max(0.5, voltage))

    # Switching time: tau = R * C * A
    area_cm2 = area_m2 * 1e4
    C_ionic = 0.005  # F/cm2 (typical WO3 ionic capacitance)
    tau = conductor_R_sq * C_ionic * math.sqrt(area_cm2)
    # sqrt(area) because current flows from edge; effective path ~ sqrt(A)
    tau = max(1.0, tau)

    return ElectrochromicLayer(
        T_clear=T_clear,
        T_dark=round(T_clear * 10 ** (-delta_OD), 4),
        delta_OD=round(delta_OD, 2),
        coloration_efficiency=CE,
        charge_density_mC_cm2=round(Q_mC_cm2, 1),
        switching_voltage_V=round(voltage, 1),
        switching_time_s=round(tau, 1),
        conductor_R_sq=conductor_R_sq,
        area_cm2=area_cm2,
        thickness_wo3_nm=round(t_wo3, 0),
        thickness_nio_nm=round(t_nio, 0),
        notes=(f"WO3({t_wo3:.0f}nm)/LiAlF4/NiO({t_nio:.0f}nm). "
               f"CE={CE:.0f}cm2/C, Q={Q_mC_cm2:.1f}mC/cm2, "
               f"V={voltage:.1f}V, tau={tau:.0f}s"),
    )


# ---------------------------------------------------------------------------
# Photochromic layer
# ---------------------------------------------------------------------------

@dataclass
class PhotochromicLayer:
    """UV-activated photochromic layer."""
    material: str                  # e.g. "AgCl_nanoparticle", "WO3_photochromic"
    thickness_nm: float
    T_clear: float
    T_dark: float                  # under UV illumination at steady state
    darkening_tau_s: float         # time constant to darken under UV
    bleaching_tau_s: float         # time constant to bleach (no UV)
    uv_threshold_mW_cm2: float    # minimum UV intensity to trigger
    notes: str = ""

    @property
    def modulation_range(self) -> float:
        return self.T_clear - self.T_dark


def design_photochromic(
    T_clear: float = 0.70,
    T_dark_target: float = 0.15,
    material: str = "AgCl_nanoparticle",
) -> PhotochromicLayer:
    """Design photochromic layer.

    AgCl nanoparticles: UV -> Ag0 clusters form, absorb VIS.
    Thermal relaxation re-oxidizes Ag0 -> AgCl (bleaching).

    Physics:
      alpha_dark = alpha_0 * (1 - exp(-UV_dose / dose_threshold))
      T_dark = T_clear * exp(-alpha_dark * d)
    """
    if material == "AgCl_nanoparticle":
        # AgCl NPs in glass matrix
        thickness = 500.0  # nm
        alpha_dark_max = 5e4  # cm-1 at full darkening
        darkening_tau = 15.0  # seconds
        bleaching_tau = 180.0  # seconds (3 min)
        uv_threshold = 1.0  # mW/cm2

        # Compute actual T_dark at max darkening
        d_cm = thickness * 1e-7
        T_dark = T_clear * math.exp(-alpha_dark_max * d_cm)
    elif material == "WO3_photochromic":
        thickness = 300.0
        alpha_dark_max = 3e4
        darkening_tau = 30.0
        bleaching_tau = 300.0
        uv_threshold = 2.0
        d_cm = thickness * 1e-7
        T_dark = T_clear * math.exp(-alpha_dark_max * d_cm)
    else:
        thickness = 400.0
        T_dark = T_dark_target
        darkening_tau = 20.0
        bleaching_tau = 240.0
        uv_threshold = 1.5

    T_dark = max(0.01, T_dark)

    return PhotochromicLayer(
        material=material,
        thickness_nm=thickness,
        T_clear=T_clear,
        T_dark=round(T_dark, 4),
        darkening_tau_s=darkening_tau,
        bleaching_tau_s=bleaching_tau,
        uv_threshold_mW_cm2=uv_threshold,
        notes=(f"{material} {thickness:.0f}nm. "
               f"Darken tau={darkening_tau:.0f}s, bleach tau={bleaching_tau:.0f}s. "
               f"UV threshold={uv_threshold:.1f}mW/cm2"),
    )


# ---------------------------------------------------------------------------
# Magnetochromic layer
# ---------------------------------------------------------------------------

@dataclass
class MagnetochromicLayer:
    """Magnetic-field-switchable opacity layer."""
    material: str                     # e.g. "Fe3O4_chains"
    particle_diameter_nm: float
    thickness_um: float
    T_zero_field: float               # transmittance with no field
    T_at_field: float                 # transmittance at target field
    target_field_T: float             # Tesla
    alignment_fraction: float         # Langevin L(x) at target field
    response_time_ms: float           # milliseconds
    notes: str = ""

    @property
    def modulation_range(self) -> float:
        return abs(self.T_zero_field - self.T_at_field)


def _langevin(x: float) -> float:
    """Langevin function L(x) = coth(x) - 1/x."""
    if abs(x) < 0.01:
        return x / 3.0
    return 1.0 / math.tanh(x) - 1.0 / x


def design_magnetochromic(
    target_field_T: float = 0.1,
    particle_nm: float = 15.0,
    T_zero_field: float = 0.60,
) -> MagnetochromicLayer:
    """Design magnetochromic layer using Fe3O4 nanoparticle chains.

    Physics:
      Magnetic NPs in polymer matrix form chains under applied field.
      Chain alignment follows Langevin function: L(x) = coth(x) - 1/x
      where x = mu*B/(kT), mu = moment of NP.

      When aligned: chains form periodic structure → Bragg diffraction
      → reduced transmission at specific wavelengths.

      When random (B=0): isotropic scattering → moderate baseline T.

    Fe3O4 NP (15nm): mu ~ 1e-19 J/T (single domain)
    Response time: ~1-10 ms (viscous relaxation)
    """
    # Magnetic moment scales with volume: mu = M_s * V
    # M_s (Fe3O4) = 4.8e5 A/m
    M_s = 4.8e5  # A/m
    r_m = particle_nm * 1e-9 / 2
    V = (4 / 3) * math.pi * r_m ** 3
    mu = M_s * V  # magnetic moment in J/T (= A*m2)

    # Langevin parameter
    x = mu * target_field_T / KT_ROOM
    L = _langevin(x)

    # Transmittance reduction: aligned chains scatter/diffract more
    # Model: T(B) = T_0 * (1 - f_scatter * L(x))
    # f_scatter: max scattering fraction when fully aligned
    f_scatter = 0.7  # at full alignment, block 70% of remaining light
    T_field = T_zero_field * (1.0 - f_scatter * L)
    T_field = max(0.01, T_field)

    # Response time: Brownian relaxation
    # tau_B = 3*eta*V_hydro / (kT)
    eta = 1e-3  # Pa.s (water-like polymer)
    V_hydro = V * 8  # hydrodynamic volume ~ 2x particle volume (with coating)
    tau_B = 3 * eta * V_hydro / KT_ROOM
    tau_ms = tau_B * 1000

    # Layer thickness: 5-20 um (particle suspension)
    thickness = 10.0  # um

    return MagnetochromicLayer(
        material="Fe3O4_chains",
        particle_diameter_nm=particle_nm,
        thickness_um=thickness,
        T_zero_field=T_zero_field,
        T_at_field=round(T_field, 4),
        target_field_T=target_field_T,
        alignment_fraction=round(L, 3),
        response_time_ms=round(tau_ms, 2),
        notes=(f"Fe3O4 {particle_nm:.0f}nm NPs, mu={mu:.1e}J/T. "
               f"At B={target_field_T}T: L={L:.3f} ({L*100:.0f}% aligned), "
               f"tau={tau_ms:.1f}ms"),
    )


# ---------------------------------------------------------------------------
# Switchable window targets
# ---------------------------------------------------------------------------

@dataclass
class SwitchableTargets:
    """Targets for switchable smart window."""
    # Base window targets
    uv_block_pct: float = 95.0
    ir_reflectance: float = 0.80
    vis_transmittance_min: float = 0.70
    iridescent: bool = True
    self_cleaning: bool = True
    u_value_target: float = 2.0
    n_panes: int = 2
    glass_mm: float = 4.0
    gap_mm: float = 12.0
    # Switchable targets
    T_clear: float = 0.60          # clear state transmittance
    T_dark: float = 0.05           # dark state transmittance
    switching_mechanism: str = "electrochromic"  # or "photochromic", "magnetochromic"
    prefer_indium_free: bool = True
    window_area_m2: float = 1.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Switchable window design
# ---------------------------------------------------------------------------

@dataclass
class SwitchableWindowDesign:
    """Complete switchable smart window design."""
    # Base window (from smart_window.py)
    base_layers: List = field(default_factory=list)
    base_vis_T: float = 0.0
    base_uv_block: float = 0.0
    base_ir_R: float = 0.0
    base_u_value: float = 0.0
    iridescence_sweep: Dict[int, str] = field(default_factory=dict)
    # Conductive
    conductor: Optional[ConductiveLayer] = None
    # Switching layer
    electrochromic: Optional[ElectrochromicLayer] = None
    photochromic: Optional[PhotochromicLayer] = None
    magnetochromic: Optional[MagnetochromicLayer] = None
    # Computed states
    T_clear_state: float = 0.0
    T_dark_state: float = 0.0
    switching_mechanism: str = ""
    switching_energy: str = ""      # voltage, UV, or field
    switching_time: str = ""
    meets_targets: bool = False
    target_checks: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)

    @property
    def modulation_range(self) -> float:
        return self.T_clear_state - self.T_dark_state

    @property
    def summary(self) -> str:
        lines = [
            f"Switchable Smart Window Design",
            f"  Mechanism: {self.switching_mechanism}",
            f"  Clear state: T={self.T_clear_state:.0%}",
            f"  Dark state:  T={self.T_dark_state:.0%}",
            f"  Modulation:  {self.modulation_range:.0%}",
            f"  Switching: {self.switching_energy}",
            f"  Response: {self.switching_time}",
        ]
        if self.conductor:
            lines.append(f"  Conductor: {self.conductor.material} "
                        f"R={self.conductor.sheet_resistance_ohm_sq:.0f}Ohm/sq "
                        f"T={self.conductor.vis_transmittance:.0%} "
                        f"{'(In-free)' if self.conductor.indium_free else '(ITO)'}")
        lines.append(f"  Base window:")
        lines.append(f"    VIS T={self.base_vis_T:.0%}, UV block={self.base_uv_block:.0f}%, "
                    f"IR R={self.base_ir_R:.0%}, U={self.base_u_value:.2f}W/m2K")
        if self.iridescence_sweep:
            lines.append(f"  Iridescence:")
            for a in sorted(self.iridescence_sweep):
                lines.append(f"    {a:3d}deg -> {self.iridescence_sweep[a]}")
        lines.append(f"  Layers: {len(self.base_layers)} base + conductor + switch")
        lines.append(f"  Meets targets: {self.meets_targets}")
        return "\n".join(lines)


def design_switchable_window(
    targets: Optional[SwitchableTargets] = None,
) -> SwitchableWindowDesign:
    """Design complete switchable smart window.

    Builds base window (UV, IR, iridescent, self-clean) then adds
    conductive layer + switching mechanism.
    """
    if targets is None:
        targets = SwitchableTargets()

    # 1. Build base window
    from core.smart_window import design_window, WindowTargets

    base_targets = WindowTargets(
        uv_block_pct=targets.uv_block_pct,
        ir_reflectance=targets.ir_reflectance,
        vis_transmittance_min=targets.vis_transmittance_min,
        iridescent=targets.iridescent,
        self_cleaning=targets.self_cleaning,
        u_value_target=targets.u_value_target,
        n_panes=targets.n_panes,
        glass_mm=targets.glass_mm,
        gap_mm=targets.gap_mm,
    )
    base = design_window(base_targets)

    # 2. Conductive layer (needed for electrochromic)
    conductor = None
    if targets.switching_mechanism == "electrochromic":
        conductor = design_conductive(
            target_R_sq=15.0,
            prefer_indium_free=targets.prefer_indium_free,
        )

    # 3. Switching layer
    # The switching layer T_clear/T_dark are ITS OWN transmittances.
    # Total = base_T * conductor_T * switching_T
    # So switching_T_clear = target_T_clear / (base_T * conductor_T)
    T_base = base.total_vis_transmittance
    T_cond = conductor.vis_transmittance if conductor else 1.0
    T_passive = T_base * T_cond

    switch_T_clear = min(0.95, targets.T_clear / max(T_passive, 0.1))
    switch_T_dark = max(0.01, targets.T_dark / max(T_passive, 0.1))

    ec = None
    pc = None
    mc = None

    if targets.switching_mechanism == "electrochromic":
        R_sq = conductor.sheet_resistance_ohm_sq if conductor else 15.0
        ec = design_electrochromic(
            T_clear=switch_T_clear,
            T_dark=switch_T_dark,
            conductor_R_sq=R_sq,
            area_m2=targets.window_area_m2,
        )
    elif targets.switching_mechanism == "photochromic":
        pc = design_photochromic(
            T_clear=switch_T_clear,
            T_dark_target=switch_T_dark,
        )
    elif targets.switching_mechanism == "magnetochromic":
        mc = design_magnetochromic(
            target_field_T=0.1,
            T_zero_field=switch_T_clear,
        )

    # 4. Compute clear and dark total states
    if ec:
        T_clear = T_passive * ec.T_clear
        T_dark = T_passive * ec.T_dark
        mechanism = "electrochromic"
        energy = f"{ec.switching_voltage_V:.1f}V DC"
        time = f"{ec.switching_time_s:.0f}s"
    elif pc:
        T_clear = T_passive * pc.T_clear
        T_dark = T_passive * pc.T_dark
        mechanism = "photochromic"
        energy = f"UV >{pc.uv_threshold_mW_cm2:.1f}mW/cm2"
        time = f"darken {pc.darkening_tau_s:.0f}s, bleach {pc.bleaching_tau_s:.0f}s"
    elif mc:
        T_clear = T_passive * mc.T_zero_field
        T_dark = T_passive * mc.T_at_field
        mechanism = "magnetochromic"
        energy = f"{mc.target_field_T:.2f}T magnetic field"
        time = f"{mc.response_time_ms:.1f}ms"
    else:
        T_clear = T_passive
        T_dark = T_passive
        mechanism = "none"
        energy = "N/A"
        time = "N/A"

    # 5. Target checks
    checks = {}
    checks["T_clear"] = (targets.T_clear, T_clear,
                          T_clear >= targets.T_clear * 0.85)
    checks["T_dark"] = (targets.T_dark, T_dark,
                         T_dark <= targets.T_dark * 1.5)
    checks["UV_block"] = (targets.uv_block_pct, base.total_uv_block_pct,
                           base.total_uv_block_pct >= targets.uv_block_pct * 0.95)
    checks["IR_reflectance"] = (targets.ir_reflectance, base.total_ir_reflectance,
                                 base.total_ir_reflectance >= targets.ir_reflectance * 0.9)
    checks["U_value"] = (targets.u_value_target, base.thermal.u_value,
                          base.thermal.u_value <= targets.u_value_target * 1.1)

    meets = all(v[2] for v in checks.values())

    return SwitchableWindowDesign(
        base_layers=base.layers,
        base_vis_T=base.total_vis_transmittance,
        base_uv_block=base.total_uv_block_pct,
        base_ir_R=base.total_ir_reflectance,
        base_u_value=base.thermal.u_value,
        iridescence_sweep=base.iridescence_sweep,
        conductor=conductor,
        electrochromic=ec,
        photochromic=pc,
        magnetochromic=mc,
        T_clear_state=round(T_clear, 3),
        T_dark_state=round(T_dark, 3),
        switching_mechanism=mechanism,
        switching_energy=energy,
        switching_time=time,
        meets_targets=meets,
        target_checks=checks,
    )
