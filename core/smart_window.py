"""
core/smart_window.py -- Smart window coating design.

Integrated module: UV block, low-e IR reflector, angle-dependent structural
color, self-cleaning exterior, and thermal U-value calculation.

Key physics insight: A single quarter-wave multilayer designed with its
reflection band in the near-IR provides:
  1. Heat rejection (IR reflection >80%)
  2. Visible transparency (VIS transmission >70%)
  3. Angle-dependent color: at oblique incidence the reflection band
     blue-shifts (Bragg: lambda(theta) = lambda_0 * sqrt(1 - sin2(theta)/n_eff2)),
     pulling the short-wavelength edge into the visible. Clear looking out
     (normal incidence), colored looking at from street (oblique).

Physics (textbook, no fitted parameters):
  Beer-Lambert:     T = exp(-alpha * d)
  Quarter-wave:     d_layer = lambda_center / (4 * n)
  Reflectance:      R = ((nH/nL)^2N - 1)^2 / ((nH/nL)^2N + 1)^2
  Bandwidth:        delta_lambda/lambda = (4/pi)*arcsin((nH-nL)/(nH+nL))
  Angle shift:      lambda(theta) = lambda_0 * sqrt(1 - sin2(theta_ext)/n_eff2)
  U-value:          1/U = 1/h_ext + d/k + 1/h_cavity + d/k + 1/h_int
  Emissivity:       h_rad = 4 * epsilon * sigma * T^3

Imports from:
  core.pfas_free_coating  (self-cleaning layer)
  core.assembly_engine    (MonomerSpec for nanoparticle assemblies)

Entry point:
  design_window(targets) -> WindowDesign
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

SIGMA = 5.67e-8     # Stefan-Boltzmann, W/m2/K4
K_GLASS = 1.0       # thermal conductivity of soda-lime glass, W/m/K
H_EXT = 23.0        # exterior convective coefficient, W/m2/K (wind)
H_INT = 8.0         # interior convective coefficient, W/m2/K (still air)
H_CONV_GAP = 1.5    # natural convection in 12mm air gap, W/m2/K


# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpticalMaterial:
    """Optical properties of a coating material."""
    name: str
    n: float             # refractive index at 550nm
    band_gap_eV: float   # optical band gap
    alpha_uv: float      # absorption coefficient at 350nm, cm-1
    alpha_vis: float      # absorption coefficient at 550nm, cm-1
    emissivity: float     # thermal emissivity (for IR)


MATERIALS: Dict[str, OpticalMaterial] = {
    "TiO2":   OpticalMaterial("TiO2",   2.49, 3.2, 1e5, 10.0, 0.90),
    "ZnO":    OpticalMaterial("ZnO",    2.00, 3.3, 8e4, 5.0,  0.90),
    "SiO2":   OpticalMaterial("SiO2",   1.46, 9.0, 0.1, 0.01, 0.84),
    "Si3N4":  OpticalMaterial("Si3N4",  2.05, 5.0, 10.0, 0.1, 0.85),
    "SnO2":   OpticalMaterial("SnO2",   2.00, 3.6, 5e4, 2.0,  0.15),  # TCO, low-e
    "ITO":    OpticalMaterial("ITO",    1.90, 3.5, 3e4, 1.0,  0.10),  # TCO, low-e
    "Ag":     OpticalMaterial("Ag",     0.13, 0.0, 1e6, 1e6,  0.02),  # metal, lowest-e
    "glass":  OpticalMaterial("glass",  1.52, 4.0, 100, 0.1,  0.84),
}


# ---------------------------------------------------------------------------
# UV absorption (Beer-Lambert)
# ---------------------------------------------------------------------------

def uv_transmittance(material: str, thickness_nm: float,
                     wavelength_nm: float = 350.0) -> float:
    """Transmittance at given wavelength through a thin film.

    T = exp(-alpha * d)
    alpha interpolated between UV and VIS values using band gap.
    """
    mat = MATERIALS.get(material)
    if mat is None:
        return 1.0

    d_cm = thickness_nm * 1e-7

    # Interpolate alpha based on wavelength relative to band gap
    lambda_gap = 1240.0 / mat.band_gap_eV  # band gap wavelength in nm
    if wavelength_nm < lambda_gap:
        # Below band gap: strong absorption
        # Exponential tail: alpha = alpha_uv * exp(-k*(wl - 350))
        decay = max(0, (wavelength_nm - 350) / (lambda_gap - 350))
        alpha = mat.alpha_uv * math.exp(-5 * decay)  # drops ~150x over gap range
    else:
        # Above band gap: transparent
        alpha = mat.alpha_vis

    T = math.exp(-alpha * d_cm)
    return max(0.0, min(1.0, T))


def vis_average_transmittance(material: str, thickness_nm: float) -> float:
    """Average transmittance across VIS spectrum (400-700nm)."""
    total = 0.0
    n_pts = 7
    for wl in range(400, 701, 50):
        total += uv_transmittance(material, thickness_nm, wl)
    return total / n_pts


# ---------------------------------------------------------------------------
# Quarter-wave multilayer (IR reflector)
# ---------------------------------------------------------------------------

@dataclass
class MultilayerStack:
    """Quarter-wave dielectric multilayer for selective reflection."""
    n_high: float          # refractive index of high-n layer
    n_low: float           # refractive index of low-n layer
    n_pairs: int           # number of H/L pairs
    center_wavelength_nm: float  # design wavelength (center of reflection band)
    mat_high: str = "TiO2"
    mat_low: str = "SiO2"

    @property
    def d_high_nm(self) -> float:
        """Quarter-wave thickness of high-n layer."""
        return self.center_wavelength_nm / (4 * self.n_high)

    @property
    def d_low_nm(self) -> float:
        """Quarter-wave thickness of low-n layer."""
        return self.center_wavelength_nm / (4 * self.n_low)

    @property
    def total_thickness_nm(self) -> float:
        return self.n_pairs * (self.d_high_nm + self.d_low_nm)

    @property
    def peak_reflectance(self) -> float:
        """Peak reflectance at center wavelength (normal incidence)."""
        ratio = (self.n_high / self.n_low) ** (2 * self.n_pairs)
        return ((ratio - 1) / (ratio + 1)) ** 2

    @property
    def bandwidth_fraction(self) -> float:
        """Fractional bandwidth: delta_lambda / lambda_center."""
        nH, nL = self.n_high, self.n_low
        return (4 / math.pi) * math.asin((nH - nL) / (nH + nL))

    @property
    def band_edges_nm(self) -> Tuple[float, float]:
        """Short and long wavelength edges of reflection band."""
        hw = self.bandwidth_fraction / 2
        return (
            self.center_wavelength_nm * (1 - hw),
            self.center_wavelength_nm * (1 + hw),
        )

    @property
    def n_eff(self) -> float:
        """Effective refractive index of the multilayer."""
        return math.sqrt(0.5 * self.n_high ** 2 + 0.5 * self.n_low ** 2)

    def angle_shifted_edges(self, theta_ext_deg: float) -> Tuple[float, float]:
        """Reflection band edges at external angle theta (degrees).

        Bragg: lambda(theta) = lambda_0 * sqrt(1 - sin2(theta)/n_eff2)
        """
        theta_rad = math.radians(theta_ext_deg)
        sin_ratio = math.sin(theta_rad) / self.n_eff
        if abs(sin_ratio) >= 1:
            return (0.0, 0.0)
        shift = math.sqrt(1 - sin_ratio ** 2)
        e_short, e_long = self.band_edges_nm
        return (e_short * shift, e_long * shift)

    def reflectance_at_wavelength(self, wavelength_nm: float,
                                  theta_ext_deg: float = 0.0) -> float:
        """Approximate reflectance at a specific wavelength and angle.

        Simple model: R = peak_R inside band, 0 outside.
        """
        edges = self.angle_shifted_edges(theta_ext_deg)
        if edges[0] <= wavelength_nm <= edges[1]:
            return self.peak_reflectance
        return 0.0

    def vis_transmittance(self, theta_ext_deg: float = 0.0) -> float:
        """Average VIS transmittance (400-700nm) at given angle."""
        total = 0.0
        n_pts = 7
        for wl in range(400, 701, 50):
            R = self.reflectance_at_wavelength(wl, theta_ext_deg)
            total += (1.0 - R)
        return total / n_pts

    def ir_reflectance(self, theta_ext_deg: float = 0.0) -> float:
        """Average IR reflectance (800-2500nm) at given angle."""
        total = 0.0
        n_pts = 0
        for wl in range(800, 2501, 100):
            total += self.reflectance_at_wavelength(wl, theta_ext_deg)
            n_pts += 1
        return total / max(n_pts, 1)

    def color_at_angle(self, theta_ext_deg: float) -> Tuple[Optional[float], str]:
        """Reflected color at given external angle.

        If the short edge of the reflection band falls in visible (380-750nm),
        that wavelength dominates the reflected color.
        """
        short_edge, _ = self.angle_shifted_edges(theta_ext_deg)
        if 380 <= short_edge <= 750:
            return short_edge, _wavelength_to_color(short_edge)
        return None, "none"


def _wavelength_to_color(wl: float) -> str:
    if wl < 380: return "UV"
    if wl < 450: return "violet"
    if wl < 495: return "blue"
    if wl < 570: return "green"
    if wl < 590: return "yellow"
    if wl < 620: return "orange"
    if wl < 750: return "red"
    return "IR"


# ---------------------------------------------------------------------------
# Thermal U-value
# ---------------------------------------------------------------------------

@dataclass
class ThermalResult:
    """Thermal performance of a glazing unit."""
    u_value: float             # W/m2/K
    r_total: float             # total thermal resistance, m2K/W
    h_cavity: float            # cavity heat transfer coefficient, W/m2/K
    emissivity_used: float
    shgc: float = 0.0         # solar heat gain coefficient (0-1)
    description: str = ""


def calculate_u_value(
    glass_thickness_mm: float = 4.0,
    gap_mm: float = 12.0,
    emissivity: float = 0.84,
    n_panes: int = 2,
    t_mean_C: float = 10.0,
) -> ThermalResult:
    """Calculate U-value for insulated glazing unit.

    Standard: EN 673 simplified method.

    Args:
        glass_thickness_mm: each pane thickness
        gap_mm: air gap between panes
        emissivity: low-e coating emissivity (on cavity surface)
        n_panes: 2 = double, 3 = triple
        t_mean_C: mean temperature for radiative calculation
    """
    T_K = t_mean_C + 273.15
    d_glass = glass_thickness_mm / 1000  # m

    # Radiative heat transfer in cavity
    h_rad = 4 * emissivity * SIGMA * T_K ** 3

    # Convective heat transfer in cavity (simplified)
    h_conv = H_CONV_GAP  # natural convection
    h_cavity = h_rad + h_conv

    # Build resistance chain
    R = 1 / H_EXT  # exterior surface resistance

    for i in range(n_panes):
        R += d_glass / K_GLASS
        if i < n_panes - 1:
            R += 1 / h_cavity

    R += 1 / H_INT  # interior surface resistance

    U = 1 / R

    # SHGC estimate: solar transmission * (1 + U/h_ext fraction absorbed)
    # Simplified: for low-e glass, SHGC ~ 0.6-0.7 (clear) or 0.25-0.4 (selective)
    shgc = 0.7 * (0.84 / max(emissivity, 0.02)) ** 0.1  # crude scaling
    shgc = min(0.85, max(0.15, shgc))

    desc_parts = [f"{n_panes}-pane", f"{glass_thickness_mm:.0f}mm glass",
                  f"{gap_mm:.0f}mm gap", f"eps={emissivity:.2f}"]

    return ThermalResult(
        u_value=round(U, 2),
        r_total=round(R, 3),
        h_cavity=round(h_cavity, 2),
        emissivity_used=emissivity,
        shgc=round(shgc, 2),
        description=", ".join(desc_parts),
    )


# ---------------------------------------------------------------------------
# Window layer types
# ---------------------------------------------------------------------------

@dataclass
class WindowLayer:
    """One functional layer in the window coating stack."""
    name: str
    function: str               # "uv_block", "ir_reflector", "self_clean", "low_e"
    material: str
    thickness_nm: float
    vis_transmittance: float    # at normal incidence
    uv_block_pct: float = 0.0  # % UV blocked at 350nm
    ir_reflectance: float = 0.0
    emissivity: float = 0.84
    water_contact_angle: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Layer design functions
# ---------------------------------------------------------------------------

def design_uv_block(
    target_uv_block_pct: float = 95.0,
    material: str = "TiO2",
) -> WindowLayer:
    """Design UV-blocking layer. Must maintain VIS transparency."""
    mat = MATERIALS.get(material, MATERIALS["TiO2"])

    # Find thickness that blocks target % UV at 350nm
    target_T = 1.0 - target_uv_block_pct / 100.0
    # T = exp(-alpha * d) → d = -ln(T) / alpha
    if target_T > 0 and mat.alpha_uv > 0:
        d_cm = -math.log(target_T) / mat.alpha_uv
        d_nm = d_cm / 1e-7
    else:
        d_nm = 200.0  # default

    d_nm = max(20, min(1000, d_nm))

    uv_T = uv_transmittance(material, d_nm, 350)
    vis_T = vis_average_transmittance(material, d_nm)

    return WindowLayer(
        name=f"uv_block_{material}",
        function="uv_block",
        material=material,
        thickness_nm=round(d_nm, 0),
        vis_transmittance=round(vis_T, 3),
        uv_block_pct=round((1 - uv_T) * 100, 1),
        notes=f"{d_nm:.0f}nm {material}, band gap={mat.band_gap_eV:.1f}eV",
    )


def design_ir_reflector(
    center_wavelength_nm: float = 1100.0,
    n_pairs: int = 5,
    mat_high: str = "TiO2",
    mat_low: str = "SiO2",
    broadband_metal: bool = False,
) -> Tuple[WindowLayer, MultilayerStack]:
    """Design IR reflector for heat rejection.

    Two modes:
    - Dielectric (default): narrow-band quarter-wave stack, very selective
    - Broadband metal: glass/TiO2/Ag/TiO2 — standard commercial low-e
      Ag layer gives broadband IR reflectance + low emissivity
    """
    nH = MATERIALS[mat_high].n
    nL = MATERIALS[mat_low].n

    stack = MultilayerStack(
        n_high=nH, n_low=nL, n_pairs=n_pairs,
        center_wavelength_nm=center_wavelength_nm,
        mat_high=mat_high, mat_low=mat_low,
    )

    if broadband_metal:
        # Commercial low-e: TiO2(30nm) / Ag(10nm) / TiO2(30nm)
        # Broadband: reflects ~80-95% of 780-2500nm
        # VIS: ~75-85% transmittance
        thickness = 70.0  # nm total
        ir_R = 0.85  # broadband average
        vis_T = 0.80
        emissivity = 0.04  # Ag is extremely low-e

        return WindowLayer(
            name="ir_reflector_Ag_low_e",
            function="ir_reflector",
            material="TiO2/Ag/TiO2",
            thickness_nm=thickness,
            vis_transmittance=vis_T,
            ir_reflectance=ir_R,
            emissivity=emissivity,
            notes=("TiO2(30)/Ag(10)/TiO2(30) broadband low-e. "
                   f"IR R=85%, VIS T=80%, eps={emissivity:.2f}"),
        ), stack
    else:
        vis_T = stack.vis_transmittance(0)
        ir_R = stack.ir_reflectance(0)
        emissivity = 0.4  # dielectric multilayer

        return WindowLayer(
            name=f"ir_reflector_{n_pairs}pair",
            function="ir_reflector",
            material=f"{mat_high}/{mat_low}",
            thickness_nm=round(stack.total_thickness_nm, 0),
            vis_transmittance=round(vis_T, 3),
            ir_reflectance=round(ir_R, 3),
            emissivity=emissivity,
            notes=(f"{n_pairs} pairs, center={center_wavelength_nm:.0f}nm, "
                   f"band={stack.band_edges_nm[0]:.0f}-{stack.band_edges_nm[1]:.0f}nm, "
                   f"R_peak={stack.peak_reflectance:.1%}"),
        ), stack


def design_low_e(material: str = "ITO") -> WindowLayer:
    """Design low-emissivity layer for thermal insulation."""
    mat = MATERIALS.get(material, MATERIALS["ITO"])
    # ITO/SnO2: ~20-50nm gives low-e while maintaining transparency
    d_nm = 30.0
    vis_T = vis_average_transmittance(material, d_nm)

    return WindowLayer(
        name=f"low_e_{material}",
        function="low_e",
        material=material,
        thickness_nm=d_nm,
        vis_transmittance=round(vis_T, 3),
        emissivity=mat.emissivity,
        notes=f"{d_nm:.0f}nm {material}, eps={mat.emissivity:.2f}",
    )


def design_self_clean() -> WindowLayer:
    """Design self-cleaning exterior using hydrophobic coating.

    Imports omniphobic design from pfas_free_coating for consistency.
    Window self-cleaning needs CA > 110 (not superhydrophobic).
    """
    from core.pfas_free_coating import design_omniphobic

    omni = design_omniphobic(target_water_ca=115.0, target_oil_ca=0)

    return WindowLayer(
        name="self_clean_silicone",
        function="self_clean",
        material=omni.chemistry,
        thickness_nm=200.0,  # ~200nm conformal coating
        vis_transmittance=0.98,  # thin silicone is VIS-transparent
        water_contact_angle=omni.water_contact_angle,
        notes=f"CA={omni.water_contact_angle:.0f}, PDMS-based, no fluorine",
    )


# ---------------------------------------------------------------------------
# Target specification
# ---------------------------------------------------------------------------

@dataclass
class WindowTargets:
    """Desired properties for smart window coating."""
    uv_block_pct: float = 95.0          # % UV blocked at 350nm
    ir_reflectance: float = 0.80        # fraction (0-1)
    vis_transmittance_min: float = 0.70  # minimum VIS transparency
    iridescent: bool = True              # angle-dependent color from IR edge
    self_cleaning: bool = True
    u_value_target: float = 2.0         # W/m2/K
    n_panes: int = 2                    # double or triple glazing
    glass_mm: float = 4.0
    gap_mm: float = 12.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Full window design
# ---------------------------------------------------------------------------

@dataclass
class WindowDesign:
    """Complete smart window coating design."""
    layers: List[WindowLayer] = field(default_factory=list)
    multilayer_stack: Optional[MultilayerStack] = None
    iridescent_stack: Optional[MultilayerStack] = None
    thermal: Optional[ThermalResult] = None
    targets: Optional[WindowTargets] = None
    # Computed
    total_vis_transmittance: float = 0.0
    total_uv_block_pct: float = 0.0
    total_ir_reflectance: float = 0.0
    color_normal: str = "clear"
    iridescence_sweep: Dict[int, str] = field(default_factory=dict)  # angle -> color
    meets_targets: bool = False
    target_checks: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)

    @property
    def total_thickness_nm(self) -> float:
        return sum(l.thickness_nm for l in self.layers)

    @property
    def summary(self) -> str:
        lines = [
            f"Smart Window Coating Design",
            f"  Total coating thickness: {self.total_thickness_nm:.0f}nm "
            f"({self.total_thickness_nm/1000:.1f}um)",
            f"  VIS transmittance (normal): {self.total_vis_transmittance:.0%}",
            f"  UV block: {self.total_uv_block_pct:.0f}%",
            f"  IR reflectance: {self.total_ir_reflectance:.0%}",
            f"  Normal view: {self.color_normal}",
        ]
        if self.iridescence_sweep:
            lines.append(f"  Iridescence:")
            for angle in sorted(self.iridescence_sweep):
                lines.append(f"    {angle:3d}deg → {self.iridescence_sweep[angle]}")
        if self.thermal:
            lines.append(f"  U-value: {self.thermal.u_value:.2f} W/m2K "
                        f"(eps={self.thermal.emissivity_used:.2f})")
            lines.append(f"  SHGC: {self.thermal.shgc:.2f}")
        lines.append(f"  Layers:")
        for i, l in enumerate(self.layers):
            lines.append(f"    {i+1}. {l.name} ({l.thickness_nm:.0f}nm, "
                        f"VIS T={l.vis_transmittance:.0%})")
        lines.append(f"  Meets targets: {self.meets_targets}")
        return "\n".join(lines)


def design_window(targets: Optional[WindowTargets] = None) -> WindowDesign:
    """Design complete smart window coating.

    Layer order (glass surface outward):
      1. Low-e layer (thermal insulation, cavity side of inner pane)
      2. UV block (near glass surface)
      3. IR reflector — broadband Ag (heat rejection)
      4. Iridescent dielectric stack — short band edge at ~780nm,
         blue-shifts into visible at oblique angles. Zero VIS penalty
         at normal incidence (entire band is in IR).
      5. Self-cleaning exterior (outermost)

    Iridescence physics: lambda(theta) = lambda_0 * sqrt(1 - sin²θ/n_eff²)
    With n_eff=2.04 (TiO2/SiO2), shift is ~6% at 45°, ~10% at 60°, ~12% at 75°.
    Edge at 780nm → 732nm (red) at 45°, 706nm (red) at 60°, 687nm (red) at 75°.
    Visible as warm red-to-orange iridescent sheen at glancing angles.
    """
    if targets is None:
        targets = WindowTargets()

    layers = []

    # 1. Low-e layer (cavity side)
    low_e = design_low_e("ITO")
    layers.append(low_e)

    # 2. UV block
    uv = design_uv_block(targets.uv_block_pct)
    layers.append(uv)

    # 3. Broadband IR reflector (Ag) for heat rejection
    ir_layer, ml_stack_ir = design_ir_reflector(
        center_wavelength_nm=1200.0, n_pairs=5, broadband_metal=True)
    layers.append(ir_layer)

    # 4. Iridescent dielectric stack
    iridescent_stack = None
    if targets.iridescent:
        # Design: short band edge at 780nm (just above visible)
        # center = short_edge / (1 - bw/2)
        nH = MATERIALS["TiO2"].n
        nL = MATERIALS["SiO2"].n
        bw_frac = (4 / math.pi) * math.asin((nH - nL) / (nH + nL))
        edge_target = 780.0  # nm, just above visible
        center_iridescent = edge_target / (1 - bw_frac / 2)

        iridescent_stack = MultilayerStack(
            n_high=nH, n_low=nL, n_pairs=3,
            center_wavelength_nm=center_iridescent,
            mat_high="TiO2", mat_low="SiO2",
        )

        # VIS transmittance: at normal, entire band is in IR → no VIS loss
        vis_T_iridescent = iridescent_stack.vis_transmittance(0)

        irid_layer = WindowLayer(
            name="iridescent_dielectric",
            function="iridescent",
            material="TiO2/SiO2",
            thickness_nm=round(iridescent_stack.total_thickness_nm, 0),
            vis_transmittance=round(vis_T_iridescent, 3),
            ir_reflectance=round(iridescent_stack.ir_reflectance(0), 3),
            emissivity=0.5,
            notes=(f"3-pair dielectric, center={center_iridescent:.0f}nm. "
                   f"Band edge at 780nm shifts into visible at oblique angles. "
                   f"No VIS penalty at normal incidence."),
        )
        layers.append(irid_layer)

    # 5. Self-cleaning exterior
    if targets.self_cleaning:
        sc = design_self_clean()
        layers.append(sc)

    # --- Compute aggregate properties ---

    total_vis_T = 1.0
    for l in layers:
        total_vis_T *= l.vis_transmittance

    total_uv_pass = 1.0
    for l in layers:
        total_uv_pass *= (1.0 - l.uv_block_pct / 100.0)
    total_uv_block = (1.0 - total_uv_pass) * 100.0

    total_ir_R = ir_layer.ir_reflectance
    # Add iridescent stack IR contribution (additive in the band overlap)
    if iridescent_stack:
        total_ir_R = 1.0 - (1.0 - total_ir_R) * (1.0 - iridescent_stack.ir_reflectance(0))

    # Iridescence sweep
    iridescence = {}
    color_normal = "clear"
    if iridescent_stack:
        for angle in [0, 15, 30, 45, 60, 75]:
            short_edge, _ = iridescent_stack.angle_shifted_edges(angle)
            if short_edge < 750:
                color = _wavelength_to_color(short_edge)
                vis_T_angle = iridescent_stack.vis_transmittance(angle)
                vis_loss = round((1 - vis_T_angle) * 100, 0)
                iridescence[angle] = f"{color} (edge={short_edge:.0f}nm, VIS loss={vis_loss:.0f}%)"
            else:
                iridescence[angle] = "clear (band in IR)"
        color_normal = "clear"  # at 0 deg, band is entirely in IR

    # Thermal
    min_eps = min(l.emissivity for l in layers)
    thermal = calculate_u_value(
        glass_thickness_mm=targets.glass_mm,
        gap_mm=targets.gap_mm,
        emissivity=min_eps,
        n_panes=targets.n_panes,
    )

    # Target checks
    checks = {}
    checks["VIS_transmittance"] = (
        targets.vis_transmittance_min,
        total_vis_T,
        total_vis_T >= targets.vis_transmittance_min,
    )
    checks["UV_block"] = (
        targets.uv_block_pct,
        total_uv_block,
        total_uv_block >= targets.uv_block_pct * 0.95,
    )
    checks["IR_reflectance"] = (
        targets.ir_reflectance,
        total_ir_R,
        total_ir_R >= targets.ir_reflectance * 0.9,
    )
    checks["U_value"] = (
        targets.u_value_target,
        thermal.u_value,
        thermal.u_value <= targets.u_value_target * 1.1,
    )
    if targets.iridescent:
        # Check that at least some angle shows visible color
        has_visible = any("clear" not in v for v in iridescence.values())
        checks["iridescence"] = (0, 0, has_visible)

    meets = all(v[2] for v in checks.values())

    return WindowDesign(
        layers=layers,
        multilayer_stack=ml_stack_ir,
        iridescent_stack=iridescent_stack,
        thermal=thermal,
        targets=targets,
        total_vis_transmittance=round(total_vis_T, 3),
        total_uv_block_pct=round(total_uv_block, 1),
        total_ir_reflectance=round(total_ir_R, 3),
        color_normal=color_normal,
        iridescence_sweep=iridescence,
        meets_targets=meets,
        target_checks=checks,
    )
