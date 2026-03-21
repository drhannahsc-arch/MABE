"""
core/window_orientation.py -- Orientation-aware window coating design.

Wraps smart_window.py and switchable_window.py with directional logic:
  - EXTERIOR: coating on outside surface (original behavior)
  - INTERIOR: coating on room-side surface (retrofit, practical)
  - IGU_SURFACE_2: on inner face of outer pane (standard for new IGU)
  - IGU_SURFACE_3: on outer face of inner pane (alternative IGU position)
  - DUAL: coatings on both exterior and interior surfaces

Physics changes by orientation:
  - Glass pre-filters UV for interior coatings (sigmoidal cutoff at 320nm)
  - Low-e position: room-facing for interior, cavity-facing for IGU
  - Self-clean only on exterior; adhesion layer on glass-contact side
  - Iridescence visibility: exterior = seen from street, interior = seen from room
  - Switchable layer position affects which direction gets darkened

Glass model:
  T_glass(wl) = 0.92 / (1 + exp(-(wl - 320)/15))
  4mm soda-lime: blocks <300nm hard, 50% at 320nm, 90% at 380nm+
  The 0.92 factor accounts for ~8% Fresnel reflection (2 surfaces, n=1.52)

Entry point:
  design_oriented_window(targets) -> OrientedWindowDesign
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Glass optical model
# ---------------------------------------------------------------------------

def glass_uv_transmittance(wavelength_nm: float, thickness_mm: float = 4.0) -> float:
    """UV transmittance of soda-lime glass.

    Sigmoidal absorption edge at ~320nm, with thickness scaling.
    T = fresnel_factor / (1 + exp(-(wl - cutoff) / width))
    """
    # Cutoff shifts slightly with thickness (thicker blocks more)
    cutoff = 320.0 + (thickness_mm - 4.0) * 2.0  # ~2nm per mm
    width = 15.0  # transition width
    fresnel = 1.0 - 0.04 * (thickness_mm / 4.0)  # ~4% per surface, 2 surfaces for 4mm
    fresnel = max(0.80, min(0.96, fresnel))

    T = fresnel / (1.0 + math.exp(-(wavelength_nm - cutoff) / width))
    return max(0.0, min(1.0, T))


def glass_vis_transmittance(thickness_mm: float = 4.0) -> float:
    """Average VIS transmittance of clear soda-lime glass."""
    total = 0.0
    n = 7
    for wl in range(400, 701, 50):
        total += glass_uv_transmittance(wl, thickness_mm)
    return round(total / n, 3)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

class Orientation(Enum):
    """Where the coating sits relative to glass and room."""
    EXTERIOR = "exterior"               # outside surface of outermost pane
    INTERIOR = "interior"               # room-side surface (retrofit)
    IGU_SURFACE_2 = "igu_surface_2"     # inner face of outer pane (standard IGU low-e)
    IGU_SURFACE_3 = "igu_surface_3"     # outer face of inner pane
    DUAL = "dual"                       # coatings on both exterior + interior


# ---------------------------------------------------------------------------
# Surface spec
# ---------------------------------------------------------------------------

@dataclass
class SurfaceCoating:
    """Coating stack for one surface of the glazing."""
    orientation: Orientation
    layers: List[dict] = field(default_factory=list)  # WindowLayer-like dicts
    vis_transmittance: float = 1.0
    uv_block_pct: float = 0.0
    ir_reflectance: float = 0.0
    emissivity: float = 0.84
    has_self_clean: bool = False
    has_adhesion: bool = False
    has_iridescent: bool = False
    iridescence_viewer: str = ""   # "street" or "room"
    switchable: bool = False
    switching_mechanism: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Layer ordering rules
# ---------------------------------------------------------------------------

def _layer_order(orientation: Orientation) -> dict:
    """Return layer ordering and configuration for given orientation.

    Each orientation specifies:
      - Order from glass surface outward (toward air/room/cavity)
      - Whether self-clean or adhesion is used
      - Where low-e faces
      - Who sees the iridescence
      - UV pre-filtering by glass (yes if light passes through glass first)
    """
    rules = {
        Orientation.EXTERIOR: {
            "order": ["adhesion", "uv_block", "ir_reflector", "iridescent",
                      "switchable", "self_clean"],
            "glass_contact": "adhesion",    # first layer bonds to glass
            "air_contact": "self_clean",    # outermost = self-clean
            "low_e_faces": "glass",         # reflects room IR back through glass
            "iridescence_viewer": "street", # street sees reflected color
            "uv_pre_filtered": False,       # sun hits coating directly
            "self_clean": True,
            "adhesion": True,
        },
        Orientation.INTERIOR: {
            "order": ["adhesion", "ir_reflector", "uv_block", "iridescent",
                      "switchable", "low_e_room_facing"],
            "glass_contact": "adhesion",    # bonds to room-side of glass
            "air_contact": "low_e",         # faces room (reflects room IR back)
            "low_e_faces": "room",          # room-facing for thermal retention
            "iridescence_viewer": "room",   # occupant sees iridescence on their window
            "uv_pre_filtered": True,        # glass already absorbed short UV
            "self_clean": False,            # interior stays clean
            "adhesion": True,
        },
        Orientation.IGU_SURFACE_2: {
            "order": ["uv_block", "ir_reflector", "iridescent", "low_e_cavity"],
            "glass_contact": None,          # deposited during manufacturing
            "air_contact": None,            # faces cavity, not air
            "low_e_faces": "cavity",        # standard position
            "iridescence_viewer": "room",   # looking through inner pane
            "uv_pre_filtered": True,        # outer pane pre-filters
            "self_clean": False,
            "adhesion": False,
        },
        Orientation.IGU_SURFACE_3: {
            "order": ["low_e_cavity", "ir_reflector", "uv_block", "iridescent"],
            "glass_contact": None,
            "air_contact": None,
            "low_e_faces": "cavity",
            "iridescence_viewer": "room",
            "uv_pre_filtered": True,
            "self_clean": False,
            "adhesion": False,
        },
    }
    return rules.get(orientation, rules[Orientation.EXTERIOR])


# ---------------------------------------------------------------------------
# Adhesion layer
# ---------------------------------------------------------------------------

@dataclass
class AdhesionLayer:
    """Transparent adhesion layer for bonding coating to glass."""
    chemistry: str
    thickness_nm: float
    vis_transmittance: float
    notes: str = ""


def design_adhesion() -> AdhesionLayer:
    """Silane-based adhesion promoter for glass bonding."""
    return AdhesionLayer(
        chemistry="aminopropylsilane",
        thickness_nm=5.0,
        vis_transmittance=0.999,
        notes="Self-assembled monolayer, covalent Si-O-Si bond to glass",
    )


# ---------------------------------------------------------------------------
# Oriented window targets
# ---------------------------------------------------------------------------

@dataclass
class OrientedWindowTargets:
    """Targets for orientation-aware window coating."""
    # Orientation
    orientation: Orientation = Orientation.INTERIOR
    # Base performance
    uv_block_pct: float = 95.0
    ir_reflectance: float = 0.80
    vis_transmittance_min: float = 0.70
    iridescent: bool = True
    u_value_target: float = 2.0
    n_panes: int = 2
    glass_mm: float = 4.0
    gap_mm: float = 12.0
    # Switchable
    switchable: bool = False
    switching_mechanism: str = "electrochromic"
    T_clear: float = 0.60
    T_dark: float = 0.05
    # Dual-side specific
    exterior_self_clean: bool = True
    exterior_iridescent: bool = True
    interior_switchable: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Oriented window design
# ---------------------------------------------------------------------------

@dataclass
class OrientedWindowDesign:
    """Complete orientation-aware window coating design."""
    orientation: Orientation
    surfaces: List[SurfaceCoating] = field(default_factory=list)
    # Glass properties
    glass_mm: float = 4.0
    glass_vis_T: float = 0.92
    glass_uv_block_at_350: float = 0.0
    # Total performance (product of glass + all surfaces)
    total_vis_T: float = 0.0
    total_uv_block_pct: float = 0.0
    total_ir_R: float = 0.0
    u_value: float = 0.0
    # Iridescence
    iridescence_sweep: Dict[int, str] = field(default_factory=dict)
    iridescence_seen_by: str = ""   # "street", "room", or "both"
    # Switchable
    T_clear_state: float = 0.0
    T_dark_state: float = 0.0
    switching_mechanism: str = ""
    switching_info: str = ""
    # Checks
    meets_targets: bool = False
    target_checks: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        lines = [
            f"Oriented Window Coating Design",
            f"  Orientation: {self.orientation.value}",
            f"  Glass: {self.glass_mm:.0f}mm soda-lime (VIS T={self.glass_vis_T:.0%})",
        ]
        for i, surf in enumerate(self.surfaces):
            lines.append(f"  Surface {i+1}: {surf.orientation.value}")
            lines.append(f"    VIS T={surf.vis_transmittance:.0%}, "
                        f"UV block={surf.uv_block_pct:.0f}%, "
                        f"IR R={surf.ir_reflectance:.0%}, "
                        f"eps={surf.emissivity:.2f}")
            if surf.has_self_clean:
                lines.append(f"    Self-cleaning: yes")
            if surf.has_adhesion:
                lines.append(f"    Adhesion: silane monolayer")
            if surf.has_iridescent:
                lines.append(f"    Iridescence visible to: {surf.iridescence_viewer}")
            if surf.switchable:
                lines.append(f"    Switchable: {surf.switching_mechanism}")
            lines.append(f"    Layers: {len(surf.layers)}")
        lines.extend([
            f"  Total performance:",
            f"    VIS T (normal): {self.total_vis_T:.0%}",
            f"    UV block: {self.total_uv_block_pct:.0f}%",
            f"    IR reflectance: {self.total_ir_R:.0%}",
            f"    U-value: {self.u_value:.2f} W/m2K",
        ])
        if self.iridescence_sweep:
            lines.append(f"    Iridescence ({self.iridescence_seen_by}):")
            for a in sorted(self.iridescence_sweep):
                lines.append(f"      {a:3d}deg -> {self.iridescence_sweep[a]}")
        if self.switching_mechanism:
            lines.append(f"    Switchable: {self.switching_mechanism}")
            lines.append(f"    Clear: {self.T_clear_state:.0%}  Dark: {self.T_dark_state:.0%}")
            lines.append(f"    {self.switching_info}")
        lines.append(f"  Meets targets: {self.meets_targets}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Design engine
# ---------------------------------------------------------------------------

def design_oriented_window(
    targets: Optional[OrientedWindowTargets] = None,
) -> OrientedWindowDesign:
    """Design window coating with orientation awareness.

    Handles EXTERIOR, INTERIOR, IGU positions, and DUAL (both sides).
    """
    if targets is None:
        targets = OrientedWindowTargets()

    if targets.orientation == Orientation.DUAL:
        return _design_dual(targets)

    return _design_single_surface(targets)


def _design_single_surface(targets: OrientedWindowTargets) -> OrientedWindowDesign:
    """Design coating for a single surface."""
    from core.smart_window import (
        design_window, WindowTargets,
        design_uv_block, design_ir_reflector, design_low_e, design_self_clean,
    )

    rules = _layer_order(targets.orientation)

    # Glass pre-filtering
    glass_T_vis = glass_vis_transmittance(targets.glass_mm)
    glass_T_uv350 = glass_uv_transmittance(350.0, targets.glass_mm)

    # Adjust UV block target: if glass pre-filters, coating needs less
    if rules["uv_pre_filtered"]:
        # Glass blocks (1 - glass_T_uv350) of UV at 350nm
        glass_uv_block = (1.0 - glass_T_uv350) * 100.0
        remaining_uv_to_block = targets.uv_block_pct - glass_uv_block
        coating_uv_target = max(0.0, remaining_uv_to_block / glass_T_uv350 * 100.0)
        coating_uv_target = min(99.0, max(50.0, coating_uv_target))
    else:
        coating_uv_target = targets.uv_block_pct

    # Build base window design (handles UV, IR, iridescent)
    base_targets = WindowTargets(
        uv_block_pct=coating_uv_target,
        ir_reflectance=targets.ir_reflectance,
        vis_transmittance_min=targets.vis_transmittance_min / glass_T_vis,  # coating-only target
        iridescent=targets.iridescent,
        self_cleaning=rules["self_clean"],
        u_value_target=targets.u_value_target,
        n_panes=targets.n_panes,
        glass_mm=targets.glass_mm,
        gap_mm=targets.gap_mm,
    )
    base = design_window(base_targets)

    # Build surface coating descriptor
    surface = SurfaceCoating(
        orientation=targets.orientation,
        layers=[{"name": l.name, "function": l.function,
                 "thickness_nm": l.thickness_nm,
                 "vis_T": l.vis_transmittance}
                for l in base.layers],
        vis_transmittance=base.total_vis_transmittance,
        uv_block_pct=base.total_uv_block_pct,
        ir_reflectance=base.total_ir_reflectance,
        emissivity=min(l.emissivity for l in base.layers),
        has_self_clean=rules["self_clean"],
        has_adhesion=rules["adhesion"],
        has_iridescent=targets.iridescent,
        iridescence_viewer=rules["iridescence_viewer"],
    )

    # Switchable
    switching_info = ""
    T_clear = 0.0
    T_dark = 0.0
    if targets.switchable:
        from core.switchable_window import (
            design_switchable_window, SwitchableTargets,
        )
        sw_targets = SwitchableTargets(
            uv_block_pct=coating_uv_target,
            ir_reflectance=targets.ir_reflectance,
            iridescent=targets.iridescent,
            self_cleaning=rules["self_clean"],
            u_value_target=targets.u_value_target,
            n_panes=targets.n_panes,
            glass_mm=targets.glass_mm,
            gap_mm=targets.gap_mm,
            T_clear=targets.T_clear,
            T_dark=targets.T_dark,
            switching_mechanism=targets.switching_mechanism,
            window_area_m2=1.0,
        )
        sw = design_switchable_window(sw_targets)
        surface.switchable = True
        surface.switching_mechanism = targets.switching_mechanism
        T_clear = sw.T_clear_state * glass_T_vis
        T_dark = sw.T_dark_state * glass_T_vis
        switching_info = f"{sw.switching_energy}, {sw.switching_time}"

    # Total performance (glass × coating)
    total_vis_T = glass_T_vis * surface.vis_transmittance
    # UV block: combined glass + coating
    combined_uv_pass = glass_T_uv350 * (1.0 - surface.uv_block_pct / 100.0)
    total_uv_block = (1.0 - combined_uv_pass) * 100.0

    # Thermal: use coating emissivity
    from core.smart_window import calculate_u_value
    thermal = calculate_u_value(
        glass_thickness_mm=targets.glass_mm,
        gap_mm=targets.gap_mm,
        emissivity=surface.emissivity,
        n_panes=targets.n_panes,
    )

    # Iridescence
    iridescence = base.iridescence_sweep
    seen_by = rules["iridescence_viewer"] if targets.iridescent else ""

    # Target checks
    checks = {}
    checks["VIS_T"] = (targets.vis_transmittance_min, total_vis_T,
                        total_vis_T >= targets.vis_transmittance_min * 0.95)
    checks["UV_block"] = (targets.uv_block_pct, total_uv_block,
                           total_uv_block >= targets.uv_block_pct * 0.95)
    checks["IR_R"] = (targets.ir_reflectance, surface.ir_reflectance,
                       surface.ir_reflectance >= targets.ir_reflectance * 0.9)
    checks["U_value"] = (targets.u_value_target, thermal.u_value,
                          thermal.u_value <= targets.u_value_target * 1.1)

    meets = all(v[2] for v in checks.values())

    return OrientedWindowDesign(
        orientation=targets.orientation,
        surfaces=[surface],
        glass_mm=targets.glass_mm,
        glass_vis_T=glass_T_vis,
        glass_uv_block_at_350=round((1 - glass_T_uv350) * 100, 1),
        total_vis_T=round(total_vis_T, 3),
        total_uv_block_pct=round(total_uv_block, 1),
        total_ir_R=round(surface.ir_reflectance, 3),
        u_value=thermal.u_value,
        iridescence_sweep=iridescence,
        iridescence_seen_by=seen_by,
        T_clear_state=round(T_clear, 3) if targets.switchable else 0.0,
        T_dark_state=round(T_dark, 3) if targets.switchable else 0.0,
        switching_mechanism=targets.switching_mechanism if targets.switchable else "",
        switching_info=switching_info,
        meets_targets=meets,
        target_checks=checks,
    )


def _design_dual(targets: OrientedWindowTargets) -> OrientedWindowDesign:
    """Design coatings on both exterior and interior surfaces.

    Exterior: self-clean + iridescent (aesthetic from street)
    Interior: low-e + switchable (thermal + privacy control)
    Both contribute to UV/IR blocking (additive).
    """
    from core.smart_window import (
        design_window, WindowTargets, design_self_clean,
        design_uv_block, design_ir_reflector, design_low_e,
        calculate_u_value,
    )

    glass_T_vis = glass_vis_transmittance(targets.glass_mm)
    glass_T_uv350 = glass_uv_transmittance(350.0, targets.glass_mm)

    # --- Exterior surface ---
    ext_targets = WindowTargets(
        uv_block_pct=targets.uv_block_pct,
        ir_reflectance=targets.ir_reflectance * 0.6,  # split IR duty
        vis_transmittance_min=0.85,  # exterior needs high T (interior adds more loss)
        iridescent=targets.exterior_iridescent,
        self_cleaning=targets.exterior_self_clean,
        u_value_target=targets.u_value_target,
        n_panes=targets.n_panes,
        glass_mm=targets.glass_mm,
        gap_mm=targets.gap_mm,
    )
    ext_base = design_window(ext_targets)

    ext_surface = SurfaceCoating(
        orientation=Orientation.EXTERIOR,
        layers=[{"name": l.name, "function": l.function,
                 "thickness_nm": l.thickness_nm, "vis_T": l.vis_transmittance}
                for l in ext_base.layers],
        vis_transmittance=ext_base.total_vis_transmittance,
        uv_block_pct=ext_base.total_uv_block_pct,
        ir_reflectance=ext_base.total_ir_reflectance,
        emissivity=min(l.emissivity for l in ext_base.layers),
        has_self_clean=targets.exterior_self_clean,
        has_adhesion=True,
        has_iridescent=targets.exterior_iridescent,
        iridescence_viewer="street",
    )

    # --- Interior surface ---
    # After glass pre-filtering, less UV remains
    remaining_uv = (1.0 - ext_surface.uv_block_pct / 100.0) * glass_T_uv350
    int_uv_target = max(50.0, (1.0 - targets.uv_block_pct / 100.0 - remaining_uv) * 100 / max(remaining_uv, 0.01))
    int_uv_target = min(95.0, int_uv_target)

    int_targets = WindowTargets(
        uv_block_pct=int_uv_target,
        ir_reflectance=targets.ir_reflectance * 0.5,
        vis_transmittance_min=0.85,
        iridescent=False,  # iridescence on exterior only
        self_cleaning=False,  # interior
        u_value_target=targets.u_value_target,
        n_panes=targets.n_panes,
        glass_mm=targets.glass_mm,
        gap_mm=targets.gap_mm,
    )
    int_base = design_window(int_targets)

    int_surface = SurfaceCoating(
        orientation=Orientation.INTERIOR,
        layers=[{"name": l.name, "function": l.function,
                 "thickness_nm": l.thickness_nm, "vis_T": l.vis_transmittance}
                for l in int_base.layers],
        vis_transmittance=int_base.total_vis_transmittance,
        uv_block_pct=int_base.total_uv_block_pct,
        ir_reflectance=int_base.total_ir_reflectance,
        emissivity=min(l.emissivity for l in int_base.layers),
        has_self_clean=False,
        has_adhesion=True,
        has_iridescent=False,
        iridescence_viewer="",
    )

    # Switchable on interior
    T_clear = 0.0
    T_dark = 0.0
    switching_info = ""
    if targets.interior_switchable and targets.switchable:
        from core.switchable_window import (
            design_switchable_window, SwitchableTargets,
        )
        sw = design_switchable_window(SwitchableTargets(
            switching_mechanism=targets.switching_mechanism,
            T_clear=targets.T_clear,
            T_dark=targets.T_dark,
            iridescent=False,
            self_cleaning=False,
        ))
        int_surface.switchable = True
        int_surface.switching_mechanism = targets.switching_mechanism
        # Total clear/dark through full stack
        T_passive = ext_surface.vis_transmittance * glass_T_vis * int_surface.vis_transmittance
        T_clear = sw.T_clear_state / sw.base_vis_T * T_passive  # scale to real passive stack
        T_dark = sw.T_dark_state / sw.base_vis_T * T_passive
        switching_info = f"{sw.switching_energy}, {sw.switching_time}"

    # --- Total performance ---
    total_vis_T = ext_surface.vis_transmittance * glass_T_vis * int_surface.vis_transmittance

    # UV: multiplicative pass-through
    uv_pass = (1 - ext_surface.uv_block_pct / 100) * glass_T_uv350 * (1 - int_surface.uv_block_pct / 100)
    total_uv_block = (1 - uv_pass) * 100

    # IR: additive (1 - (1-R1)(1-R2))
    total_ir_R = 1 - (1 - ext_surface.ir_reflectance) * (1 - int_surface.ir_reflectance)

    # Thermal: use lowest emissivity from either surface
    min_eps = min(ext_surface.emissivity, int_surface.emissivity)
    thermal = calculate_u_value(
        glass_thickness_mm=targets.glass_mm,
        gap_mm=targets.gap_mm,
        emissivity=min_eps,
        n_panes=targets.n_panes,
    )

    # Checks
    checks = {}
    checks["VIS_T"] = (targets.vis_transmittance_min, total_vis_T,
                        total_vis_T >= targets.vis_transmittance_min * 0.90)
    checks["UV_block"] = (targets.uv_block_pct, total_uv_block,
                           total_uv_block >= targets.uv_block_pct * 0.95)
    checks["IR_R"] = (targets.ir_reflectance, total_ir_R,
                       total_ir_R >= targets.ir_reflectance * 0.9)
    checks["U_value"] = (targets.u_value_target, thermal.u_value,
                          thermal.u_value <= targets.u_value_target * 1.1)
    meets = all(v[2] for v in checks.values())

    return OrientedWindowDesign(
        orientation=Orientation.DUAL,
        surfaces=[ext_surface, int_surface],
        glass_mm=targets.glass_mm,
        glass_vis_T=glass_T_vis,
        glass_uv_block_at_350=round((1 - glass_T_uv350) * 100, 1),
        total_vis_T=round(total_vis_T, 3),
        total_uv_block_pct=round(total_uv_block, 1),
        total_ir_R=round(total_ir_R, 3),
        u_value=thermal.u_value,
        iridescence_sweep=ext_base.iridescence_sweep if targets.exterior_iridescent else {},
        iridescence_seen_by="street" if targets.exterior_iridescent else "",
        T_clear_state=round(T_clear, 3) if targets.switchable else 0.0,
        T_dark_state=round(T_dark, 3) if targets.switchable else 0.0,
        switching_mechanism=targets.switching_mechanism if targets.switchable else "",
        switching_info=switching_info,
        meets_targets=meets,
        target_checks=checks,
    )
