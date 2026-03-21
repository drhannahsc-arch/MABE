"""
core/pfas_free_coating.py -- PFAS-free multi-layer coating design.

Modular coating stack with physics scoring at each layer.
Substrate-agnostic: textile, rigid, flexible film.
No fluorine chemistry — omniphobicity from re-entrant geometry.

Physics (textbook, no fitted parameters):
  Young's equation:      cos(theta_Y) = 2*gamma_s/gamma_l - 1
  Cassie-Baxter:         cos(theta_CB) = f*(cos(theta_Y)+1) - 1
  Wenzel:                cos(theta_W) = r*cos(theta_Y)
  Laplace pressure:      dP = 2*gamma*|cos(theta)|/r_pore
  Bragg condition:        lambda = 2*n_eff*d
  WVTR estimate:          from pore size + porosity + Fick's law

Layer types:
  1. Primer        - adhesion to substrate
  2. Omniphobic    - water + oil repulsion (re-entrant texture)
  3. Structural color - photonic nanostructure (Bragg/Mie)
  4. Breathability - vapor-permeable, liquid-proof membrane
  5. Durability    - abrasion-resistant topcoat

Entry point:
  design_coating(substrate, targets) -> CoatingDesign
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

GAMMA_WATER = 72.8        # mN/m at 20C
GAMMA_HEXADECANE = 27.5   # mN/m (model oil)
GAMMA_OLIVE_OIL = 33.0    # mN/m
D_WATER_VAPOR = 2.5e-5    # m2/s diffusivity of water vapor in air (25C)
MOLAR_MASS_WATER = 0.018  # kg/mol

# Surface energies of coating chemistries (mN/m)
_CHEMISTRY_GAMMA: Dict[str, float] = {
    "PTFE":           18.0,    # reference (PFAS)
    "PDMS_silicone":  20.0,    # polydimethylsiloxane
    "paraffin_wax":   25.0,
    "stearic_acid":   24.0,    # C18 SAM
    "octadecylsilane": 22.0,   # OTS SAM on SiO2
    "carnauba_wax":   26.0,
    "polyethylene":   31.0,
    "polystyrene":    40.0,
    "glass":          250.0,   # clean glass (hydrophilic)
    "cotton_raw":     44.0,
    "polyester":      43.0,
    "nylon":          46.0,
    "aluminum":       200.0,
    "steel":          250.0,
}


# ---------------------------------------------------------------------------
# Contact angle physics
# ---------------------------------------------------------------------------

def young_contact_angle(gamma_surface: float, gamma_liquid: float) -> float:
    """Young's equation: flat surface contact angle (degrees).

    Uses Fox equation approximation:
    cos(theta_Y) = 2*(gamma_s/gamma_l) - 1
    """
    if gamma_liquid <= 0:
        return 0.0
    cos_y = min(1.0, max(-1.0, 2.0 * gamma_surface / gamma_liquid - 1.0))
    return math.degrees(math.acos(cos_y))


def cassie_baxter_angle(theta_young_deg: float, solid_fraction: float) -> float:
    """Cassie-Baxter equation for textured surface.

    cos(theta_CB) = f*(cos(theta_Y) + 1) - 1
    f = solid fraction contacting the drop (0 < f <= 1)
    """
    f = max(0.01, min(1.0, solid_fraction))
    cos_y = math.cos(math.radians(theta_young_deg))
    cos_cb = f * (cos_y + 1) - 1
    cos_cb = max(-1.0, min(1.0, cos_cb))
    return math.degrees(math.acos(cos_cb))


def wenzel_angle(theta_young_deg: float, roughness_factor: float) -> float:
    """Wenzel equation for rough surface (liquid fills grooves).

    cos(theta_W) = r * cos(theta_Y)
    r = actual area / projected area (r >= 1)
    """
    r = max(1.0, roughness_factor)
    cos_y = math.cos(math.radians(theta_young_deg))
    cos_w = r * cos_y
    cos_w = max(-1.0, min(1.0, cos_w))
    return math.degrees(math.acos(cos_w))


def laplace_entry_pressure(gamma_liquid: float, contact_angle_deg: float,
                           pore_radius_m: float) -> float:
    """Laplace pressure for liquid entry into pore (Pa).

    dP = 2 * gamma * |cos(theta)| / r
    Positive when theta > 90 (liquid repelled).
    """
    if pore_radius_m <= 0:
        return float('inf')
    theta_rad = math.radians(contact_angle_deg)
    cos_t = math.cos(theta_rad)
    if cos_t >= 0:
        return 0.0  # wetting, no barrier
    return 2.0 * (gamma_liquid / 1000.0) * abs(cos_t) / pore_radius_m


def re_entrant_oil_angle(gamma_surface: float, psi_deg: float,
                         solid_fraction: float) -> float:
    """Oil contact angle on re-entrant texture (Tuteja 2007, Science 318:1618).

    Key insight: re-entrant geometry pins the contact line on the overhang,
    making the Cassie-Baxter state METASTABLE even when theta_Y < 90.
    The contact angle equation is the same (Cassie-Baxter), but the state
    that was thermodynamically inaccessible on flat geometry becomes
    accessible on re-entrant geometry.

    Stability condition (Tuteja's T* > 0):
    The meniscus must advance past the overhang before it can wet.
    This requires psi > (90 - theta_Y), approximately.

    Returns Cassie-Baxter angle if re-entrant is sufficient, else Young's.
    """
    theta_y = young_contact_angle(gamma_surface, GAMMA_HEXADECANE)

    if theta_y > 90:
        # Already non-wetting, re-entrant just adds stability
        return cassie_baxter_angle(theta_y, solid_fraction)

    # Check if re-entrant angle is sufficient to pin the meniscus
    # Tuteja criterion: psi must exceed (90 - theta_Y)
    critical_psi = 90.0 - theta_y
    if psi_deg < critical_psi:
        # Re-entrant too shallow — Cassie state not stable
        return theta_y  # falls to Wenzel (wetting)

    # Re-entrant sufficient: Cassie-Baxter applies
    # cos(theta_CB) = f*(cos(theta_Y) + 1) - 1
    return cassie_baxter_angle(theta_y, solid_fraction)


# ---------------------------------------------------------------------------
# Bragg structural color
# ---------------------------------------------------------------------------

def bragg_wavelength(n_eff: float, d_nm: float) -> float:
    """First-order Bragg peak: lambda = 2 * n_eff * d (nm)."""
    return 2.0 * n_eff * d_nm


def particle_size_for_color(target_wavelength_nm: float,
                            n_particle: float = 1.5,
                            n_medium: float = 1.0,
                            packing_fraction: float = 0.64) -> float:
    """Particle diameter (nm) needed for target color.

    For close-packed spheres: d_layer = diameter * 0.816 (FCC 111 spacing)
    n_eff = sqrt(packing * n_particle^2 + (1-packing) * n_medium^2)
    lambda = 2 * n_eff * d_layer
    """
    n_eff = math.sqrt(packing_fraction * n_particle ** 2
                      + (1 - packing_fraction) * n_medium ** 2)
    d_layer = target_wavelength_nm / (2.0 * n_eff)
    diameter = d_layer / 0.816  # FCC 111 → sphere diameter
    return round(diameter, 1)


def color_from_particle_size(diameter_nm: float,
                             n_particle: float = 1.5,
                             n_medium: float = 1.0,
                             packing_fraction: float = 0.64) -> Tuple[float, str]:
    """Predicted peak wavelength and color name from particle diameter."""
    n_eff = math.sqrt(packing_fraction * n_particle ** 2
                      + (1 - packing_fraction) * n_medium ** 2)
    d_layer = diameter_nm * 0.816
    wavelength = 2.0 * n_eff * d_layer

    # Color name from wavelength
    if wavelength < 380:
        color = "UV"
    elif wavelength < 450:
        color = "violet"
    elif wavelength < 495:
        color = "blue"
    elif wavelength < 570:
        color = "green"
    elif wavelength < 590:
        color = "yellow"
    elif wavelength < 620:
        color = "orange"
    elif wavelength < 750:
        color = "red"
    else:
        color = "IR"

    return round(wavelength, 1), color


# ---------------------------------------------------------------------------
# Breathability physics
# ---------------------------------------------------------------------------

def wvtr_estimate(pore_radius_m: float, porosity: float,
                  membrane_thickness_m: float,
                  delta_rh: float = 0.5) -> float:
    """Water vapor transmission rate estimate (g/m2/day).

    Simplified Fick's law through porous membrane:
    WVTR = D_eff * delta_C / thickness * 86400
    D_eff = D_air * porosity / tortuosity
    delta_C = delta_RH * C_sat(37C)

    Good outdoor jacket: WVTR > 5000-10000 g/m2/day
    Gore-Tex: ~15000 g/m2/day
    """
    if membrane_thickness_m <= 0 or porosity <= 0:
        return 0.0

    # Tortuosity estimate (Bruggeman)
    tortuosity = porosity ** (-0.5)

    # Effective diffusivity
    if pore_radius_m < 50e-9:
        # Knudsen regime: D_K = (d_pore/3) * sqrt(8RT/piM)
        d_k = (2 * pore_radius_m / 3) * math.sqrt(8 * 8.314 * 310 / (math.pi * MOLAR_MASS_WATER))
        d_eff = porosity * d_k / tortuosity
    else:
        # Molecular diffusion regime
        d_eff = D_WATER_VAPOR * porosity / tortuosity

    # Concentration gradient: C_sat at 37C ~ 44 g/m3 (skin temp)
    c_sat = 44.0e-3  # kg/m3
    delta_c = delta_rh * c_sat

    # Flux: J = D_eff * delta_C / thickness  (kg/m2/s)
    flux = d_eff * delta_c / membrane_thickness_m

    # Convert to g/m2/day
    wvtr = flux * 1000 * 86400
    return round(wvtr, 0)


# ---------------------------------------------------------------------------
# Substrate specification
# ---------------------------------------------------------------------------

class SubstrateType(Enum):
    TEXTILE = "textile"
    RIGID = "rigid"
    FILM = "film"


@dataclass
class SubstrateSpec:
    """Physical properties of the substrate."""
    name: str
    substrate_type: SubstrateType
    material: str                    # e.g. "cotton", "polyester", "glass", "PET"
    gamma_surface: float             # surface energy, mN/m
    roughness_rms_um: float = 1.0   # RMS roughness in micrometers
    porosity: float = 0.0           # bulk porosity (0-1), for textiles
    pore_size_um: float = 0.0       # mean pore diameter
    flexible: bool = True
    max_process_temp_C: float = 200.0
    notes: str = ""


# Pre-defined substrates
SUBSTRATES: Dict[str, SubstrateSpec] = {
    "cotton": SubstrateSpec("cotton", SubstrateType.TEXTILE, "cotton",
                            44.0, roughness_rms_um=5.0, porosity=0.6,
                            pore_size_um=20.0, max_process_temp_C=180),
    "polyester": SubstrateSpec("polyester", SubstrateType.TEXTILE, "polyester",
                               43.0, roughness_rms_um=3.0, porosity=0.5,
                               pore_size_um=15.0, max_process_temp_C=200),
    "nylon": SubstrateSpec("nylon", SubstrateType.TEXTILE, "nylon",
                            46.0, roughness_rms_um=3.0, porosity=0.5,
                            pore_size_um=15.0, max_process_temp_C=180),
    "glass": SubstrateSpec("glass", SubstrateType.RIGID, "glass",
                            250.0, roughness_rms_um=0.01, porosity=0.0,
                            flexible=False, max_process_temp_C=500),
    "aluminum": SubstrateSpec("aluminum", SubstrateType.RIGID, "aluminum",
                               200.0, roughness_rms_um=0.5, porosity=0.0,
                               flexible=False, max_process_temp_C=400),
    "stainless_steel": SubstrateSpec("stainless_steel", SubstrateType.RIGID, "steel",
                                      250.0, roughness_rms_um=0.3, porosity=0.0,
                                      flexible=False, max_process_temp_C=600),
    "PET_film": SubstrateSpec("PET_film", SubstrateType.FILM, "PET",
                               43.0, roughness_rms_um=0.05, porosity=0.0,
                               flexible=True, max_process_temp_C=150),
    "PP_film": SubstrateSpec("PP_film", SubstrateType.FILM, "polypropylene",
                              30.0, roughness_rms_um=0.1, porosity=0.0,
                              flexible=True, max_process_temp_C=130),
}


# ---------------------------------------------------------------------------
# Coating layer
# ---------------------------------------------------------------------------

class LayerType(Enum):
    PRIMER = "primer"
    OMNIPHOBIC = "omniphobic"
    STRUCTURAL_COLOR = "structural_color"
    BREATHABILITY = "breathability"
    DURABILITY = "durability"


@dataclass
class CoatingLayer:
    """One functional layer in the coating stack."""
    name: str
    layer_type: LayerType
    chemistry: str                 # coating material name
    thickness_um: float            # layer thickness in micrometers
    gamma_surface: float           # surface energy of this layer, mN/m

    # Texture parameters (for omniphobic layer)
    solid_fraction: float = 1.0    # Cassie-Baxter f (1 = flat)
    re_entrant_angle_deg: float = 0.0  # overhang angle (0 = no re-entrant)
    pore_radius_um: float = 0.0   # pore size in this layer
    porosity: float = 0.0

    # Structural color
    particle_diameter_nm: float = 0.0
    n_particle: float = 1.5
    peak_wavelength_nm: float = 0.0
    color: str = ""

    # Computed scores
    water_contact_angle: float = 0.0
    oil_contact_angle: float = 0.0
    lep_water_kPa: float = 0.0    # liquid entry pressure for water
    lep_oil_kPa: float = 0.0
    wvtr: float = 0.0             # g/m2/day (if breathable)

    notes: str = ""


# ---------------------------------------------------------------------------
# Layer design functions
# ---------------------------------------------------------------------------

def design_primer(substrate: SubstrateSpec) -> CoatingLayer:
    """Design adhesion primer matched to substrate."""
    # Pick chemistry that bridges substrate γ to coating γ
    if substrate.gamma_surface > 100:
        # High energy (glass, metal): silane coupling agent
        chem = "aminosilane"
        gamma = 35.0
        thickness = 0.05  # ~50 nm SAM
    elif substrate.gamma_surface > 40:
        # Medium (textiles, polymers): plasma + primer
        chem = "acrylic_primer"
        gamma = 38.0
        thickness = 1.0
    else:
        # Low energy (PP, PE): corona treatment + primer
        chem = "corona_treated_primer"
        gamma = 36.0
        thickness = 0.5

    return CoatingLayer(
        name=f"primer_{chem}",
        layer_type=LayerType.PRIMER,
        chemistry=chem,
        thickness_um=thickness,
        gamma_surface=gamma,
        notes=f"Bridges substrate gamma={substrate.gamma_surface:.0f} to coating",
    )


def design_omniphobic(
    target_water_ca: float = 150.0,
    target_oil_ca: float = 130.0,
) -> CoatingLayer:
    """Design omniphobic layer using re-entrant texture.

    Key physics:
    - Water repulsion: silicone chemistry (gamma~20) + texture
    - Oil repulsion: REQUIRES re-entrant geometry (Tuteja 2007)
    - No fluorine needed if re-entrant angle > 60 deg
    """
    # Base chemistry: PDMS silicone (gamma = 20 mN/m)
    gamma_s = 20.0
    chemistry = "PDMS_silicone"

    # Young's angle on flat surface
    theta_y_water = young_contact_angle(gamma_s, GAMMA_WATER)
    theta_y_oil = young_contact_angle(gamma_s, GAMMA_HEXADECANE)

    # Need to find solid fraction f for target CAs
    # Use the MORE DEMANDING liquid (usually oil) to set f
    # Water: cos(theta_CB) = f*(cos(theta_Y_water)+1) - 1
    # Oil (re-entrant): cos(theta_CB) = f*(cos(theta_Y_oil)+1) - 1
    cos_target_w = math.cos(math.radians(target_water_ca))
    cos_y_w = math.cos(math.radians(theta_y_water))

    if (cos_y_w + 1) > 0.01:
        f_water = (cos_target_w + 1) / (cos_y_w + 1)
    else:
        f_water = 0.05

    f_needed = max(0.02, min(1.0, f_water))

    # If oil repulsion requested, check if oil demands lower f
    if target_oil_ca > 0:
        cos_target_o = math.cos(math.radians(target_oil_ca))
        cos_y_o = math.cos(math.radians(theta_y_oil))
        if (cos_y_o + 1) > 0.01:
            f_oil = (cos_target_o + 1) / (cos_y_o + 1)
        else:
            f_oil = 0.05
        f_oil = max(0.02, min(1.0, f_oil))
        f_needed = min(f_needed, f_oil)  # use the tighter constraint

    # Re-entrant angle for oil repulsion
    # Need to find psi such that re_entrant_oil_angle > target_oil_ca
    # Empirical: psi ~ 70 deg works for most oils with f < 0.2
    psi = 70.0  # degrees
    if target_oil_ca > 120:
        psi = 75.0  # steeper overhang for high oil CA
    if target_oil_ca > 140:
        psi = 80.0

    # Compute actual angles
    water_ca = cassie_baxter_angle(theta_y_water, f_needed)
    oil_ca = re_entrant_oil_angle(gamma_s, psi, f_needed)

    # Pore size for texture (between pillars)
    # Typical: 1-10 um pillars, spacing ~ 2-20 um
    pore_r = 5.0  # um, inter-pillar gap

    # Laplace entry pressure
    lep_w = laplace_entry_pressure(GAMMA_WATER / 1000, water_ca, pore_r * 1e-6) / 1000
    lep_o = laplace_entry_pressure(GAMMA_HEXADECANE / 1000, oil_ca, pore_r * 1e-6) / 1000

    return CoatingLayer(
        name="omniphobic_reentrant",
        layer_type=LayerType.OMNIPHOBIC,
        chemistry=chemistry,
        thickness_um=10.0,
        gamma_surface=gamma_s,
        solid_fraction=round(f_needed, 3),
        re_entrant_angle_deg=psi,
        pore_radius_um=pore_r,
        water_contact_angle=round(water_ca, 1),
        oil_contact_angle=round(oil_ca, 1),
        lep_water_kPa=round(lep_w, 1),
        lep_oil_kPa=round(lep_o, 1),
        notes=(f"Re-entrant geometry (psi={psi:.0f}deg) for oil repulsion. "
               f"Young water={theta_y_water:.0f}, oil={theta_y_oil:.0f}. "
               f"No fluorine chemistry."),
    )


def design_structural_color(
    target_color: str = "blue",
    particle_material: str = "SiO2",
) -> CoatingLayer:
    """Design structural color layer from photonic nanostructure.

    Uses Bragg diffraction from close-packed nanoparticles.
    No dye migration, no fading (color from structure, not chemistry).
    """
    # Target wavelengths
    color_wavelength = {
        "violet": 420, "blue": 470, "cyan": 500,
        "green": 530, "yellow": 580, "orange": 600,
        "red": 650,
    }
    target_wl = color_wavelength.get(target_color.lower(), 530)

    # Particle refractive index
    n_map = {
        "SiO2": 1.46, "polystyrene": 1.59, "PMMA": 1.49,
        "TiO2": 2.49, "ZnO": 2.00, "melanin": 1.74,
    }
    n_p = n_map.get(particle_material, 1.50)

    # Calculate required particle size
    diameter = particle_size_for_color(target_wl, n_particle=n_p)

    # Verify: what color do we actually get?
    actual_wl, actual_color = color_from_particle_size(diameter, n_particle=n_p)

    # Layer thickness: need ~10-20 layers of particles for good color
    n_layers = 15
    layer_thickness_um = diameter * n_layers * 0.816 / 1000  # nm -> um

    return CoatingLayer(
        name=f"structural_color_{actual_color}",
        layer_type=LayerType.STRUCTURAL_COLOR,
        chemistry=particle_material,
        thickness_um=round(layer_thickness_um, 1),
        gamma_surface=_CHEMISTRY_GAMMA.get(particle_material, 30.0),
        particle_diameter_nm=diameter,
        n_particle=n_p,
        peak_wavelength_nm=actual_wl,
        color=actual_color,
        notes=(f"{diameter:.0f}nm {particle_material} particles, "
               f"{n_layers} layers. lambda={actual_wl:.0f}nm ({actual_color}). "
               f"Angle-dependent iridescence."),
    )


def design_breathability(
    target_wvtr: float = 10000.0,
    target_lep_kPa: float = 50.0,
) -> CoatingLayer:
    """Design breathable membrane (vapor-permeable, liquid-proof).

    Key: pore size must be small enough to repel liquid (Laplace)
    but large enough for vapor diffusion.
    Sweet spot: 0.1-1.0 um pores.
    """
    # Hydrophobic pore walls
    gamma_s = 20.0  # PDMS
    theta_y = young_contact_angle(gamma_s, GAMMA_WATER)

    # Find pore size that gives target LEP
    # LEP = 2*gamma*|cos(theta)| / r
    cos_t = math.cos(math.radians(theta_y))
    if cos_t >= 0:
        pore_r_m = 1e-6  # fallback, shouldn't happen for hydrophobic
    else:
        # Solve: r = 2*gamma*|cos_t| / LEP
        lep_pa = target_lep_kPa * 1000
        pore_r_m = 2 * (GAMMA_WATER / 1000) * abs(cos_t) / lep_pa

    pore_r_um = pore_r_m * 1e6
    porosity = 0.5  # typical for expanded PTFE or electrospun membrane

    # Thickness for target WVTR
    # Start thick and thin out until WVTR target is met
    chosen_thickness = 50.0  # default thick (um)
    for thickness_test in [5, 10, 15, 20, 30, 50, 100]:
        wvtr_test = wvtr_estimate(pore_r_m, porosity, thickness_test * 1e-6)
        if wvtr_test >= target_wvtr * 0.8:
            chosen_thickness = thickness_test
            break

    actual_wvtr = wvtr_estimate(pore_r_m, porosity, chosen_thickness * 1e-6)
    # Cap at realistic values (Gore-Tex max ~25000)
    actual_wvtr = min(actual_wvtr, 30000.0)
    actual_lep = laplace_entry_pressure(
        GAMMA_WATER / 1000, theta_y, pore_r_m) / 1000

    return CoatingLayer(
        name="breathable_membrane",
        layer_type=LayerType.BREATHABILITY,
        chemistry="PDMS_microporous",
        thickness_um=chosen_thickness,
        gamma_surface=gamma_s,
        pore_radius_um=round(pore_r_um, 3),
        porosity=porosity,
        lep_water_kPa=round(actual_lep, 1),
        wvtr=round(actual_wvtr, 0),
        notes=(f"Pore radius={pore_r_um:.3f}um, porosity={porosity:.0%}. "
               f"LEP={actual_lep:.0f}kPa, WVTR={actual_wvtr:.0f}g/m2/day"),
    )


def design_durability(flexible: bool = True, over_omniphobic: bool = False) -> CoatingLayer:
    """Design abrasion-resistant topcoat.

    When over an omniphobic layer, must use low-gamma chemistry
    to preserve water/oil repulsion through the conformal topcoat.
    """
    if over_omniphobic:
        # Must maintain low gamma to preserve Cassie-Baxter state
        chem = "fluorine_free_silicone_hardcoat"
        gamma = 21.0
        thickness = 0.5  # thin conformal
        notes = "Conformal silicone hardcoat, preserves texture. No fluorine."
    elif flexible:
        chem = "silicone_urethane_hybrid"
        gamma = 28.0
        thickness = 2.0
        notes = "Flexible crosslinked hybrid, pencil hardness ~2H"
    else:
        chem = "sol_gel_SiO2"
        gamma = 30.0
        thickness = 1.0
        notes = "Hard sol-gel silica, pencil hardness ~6H"

    return CoatingLayer(
        name=f"topcoat_{chem}",
        layer_type=LayerType.DURABILITY,
        chemistry=chem,
        thickness_um=thickness,
        gamma_surface=gamma,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Coating stack
# ---------------------------------------------------------------------------

@dataclass
class CoatingStack:
    """Ordered list of coating layers on a substrate."""
    substrate: SubstrateSpec
    layers: List[CoatingLayer] = field(default_factory=list)

    @property
    def total_thickness_um(self) -> float:
        return sum(l.thickness_um for l in self.layers)

    @property
    def top_layer(self) -> Optional[CoatingLayer]:
        return self.layers[-1] if self.layers else None

    @property
    def water_contact_angle(self) -> float:
        """Effective water CA. Thin durability topcoat preserves underlying texture."""
        omni = None
        for layer in self.layers:
            if layer.layer_type == LayerType.OMNIPHOBIC:
                omni = layer
        if omni is not None:
            # Omniphobic texture determines CA; topcoat is conformal
            # Adjust for topcoat chemistry if present
            top = self.top_layer
            if top and top.layer_type == LayerType.DURABILITY:
                # Conformal: use topcoat gamma but omniphobic geometry
                theta_y = young_contact_angle(top.gamma_surface, GAMMA_WATER)
                return cassie_baxter_angle(theta_y, omni.solid_fraction)
            return omni.water_contact_angle
        if self.layers:
            return young_contact_angle(self.layers[-1].gamma_surface, GAMMA_WATER)
        return young_contact_angle(self.substrate.gamma_surface, GAMMA_WATER)

    @property
    def oil_contact_angle(self) -> float:
        """Effective oil CA. Re-entrant geometry preserved through conformal topcoat."""
        omni = None
        for layer in self.layers:
            if layer.layer_type == LayerType.OMNIPHOBIC:
                omni = layer
        if omni is not None:
            top = self.top_layer
            if top and top.layer_type == LayerType.DURABILITY:
                gamma_top = top.gamma_surface
                return re_entrant_oil_angle(
                    gamma_top, omni.re_entrant_angle_deg, omni.solid_fraction)
            return omni.oil_contact_angle
        return 0.0

    @property
    def wvtr(self) -> float:
        """WVTR limited by least permeable layer."""
        wvtrs = [l.wvtr for l in self.layers if l.wvtr > 0]
        return min(wvtrs) if wvtrs else 0.0

    @property
    def color(self) -> str:
        for layer in self.layers:
            if layer.layer_type == LayerType.STRUCTURAL_COLOR:
                return layer.color
        return "none"

    def compatibility_check(self) -> List[str]:
        """Check inter-layer compatibility."""
        issues = []
        for i in range(1, len(self.layers)):
            prev = self.layers[i - 1]
            curr = self.layers[i]
            # Surface energy mismatch: coating won't wet
            if prev.gamma_surface < curr.gamma_surface * 0.5:
                issues.append(
                    f"Layer {i}: {curr.name} (gamma={curr.gamma_surface:.0f}) "
                    f"may not wet {prev.name} (gamma={prev.gamma_surface:.0f})")
        return issues


# ---------------------------------------------------------------------------
# Target specification
# ---------------------------------------------------------------------------

@dataclass
class CoatingTargets:
    """Desired properties for the coating design."""
    water_contact_angle: float = 150.0    # degrees
    oil_contact_angle: float = 130.0      # degrees (0 = don't care)
    color: str = ""                       # "blue", "red", etc. ("" = no color)
    particle_material: str = "SiO2"       # for structural color
    wvtr: float = 0.0                     # g/m2/day (0 = not breathable)
    lep_kPa: float = 50.0                # liquid entry pressure
    durable: bool = True
    notes: str = ""


# ---------------------------------------------------------------------------
# Full design pipeline
# ---------------------------------------------------------------------------

@dataclass
class CoatingDesign:
    """Complete coating design output."""
    stack: CoatingStack
    targets: CoatingTargets
    meets_targets: bool = False
    target_checks: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        lines = [
            f"PFAS-Free Coating Design",
            f"  Substrate: {self.stack.substrate.name} ({self.stack.substrate.substrate_type.value})",
            f"  Total thickness: {self.stack.total_thickness_um:.1f} um",
            f"  Water CA: {self.stack.water_contact_angle:.0f} deg",
            f"  Oil CA: {self.stack.oil_contact_angle:.0f} deg",
        ]
        if self.stack.color != "none":
            lines.append(f"  Color: {self.stack.color}")
        if self.stack.wvtr > 0:
            lines.append(f"  WVTR: {self.stack.wvtr:.0f} g/m2/day")
        lines.append(f"  Layers:")
        for i, layer in enumerate(self.stack.layers):
            lines.append(f"    {i+1}. {layer.name} ({layer.thickness_um:.1f}um, "
                        f"gamma={layer.gamma_surface:.0f})")
        issues = self.stack.compatibility_check()
        if issues:
            lines.append(f"  Warnings:")
            for w in issues:
                lines.append(f"    ! {w}")
        lines.append(f"  Meets all targets: {self.meets_targets}")
        return "\n".join(lines)


def design_coating(
    substrate: SubstrateSpec,
    targets: CoatingTargets,
) -> CoatingDesign:
    """
    Design a complete PFAS-free coating stack.

    Layer order (bottom to top):
    1. Primer (always)
    2. Structural color (if requested)
    3. Breathability membrane (if WVTR > 0)
    4. Omniphobic (if water/oil repulsion needed)
    5. Durability topcoat (if requested)
    """
    layers = []

    # 1. Primer
    layers.append(design_primer(substrate))

    # 2. Structural color (below omniphobic so it's protected)
    if targets.color:
        layers.append(design_structural_color(targets.color, targets.particle_material))

    # 3. Breathability membrane
    if targets.wvtr > 0:
        layers.append(design_breathability(targets.wvtr, targets.lep_kPa))

    # 4. Omniphobic surface
    if targets.water_contact_angle > 90 or targets.oil_contact_angle > 0:
        layers.append(design_omniphobic(
            targets.water_contact_angle,
            targets.oil_contact_angle,
        ))

    # 5. Durability
    if targets.durable:
        has_omni = any(l.layer_type == LayerType.OMNIPHOBIC for l in layers)
        layers.append(design_durability(
            flexible=substrate.flexible, over_omniphobic=has_omni))

    stack = CoatingStack(substrate=substrate, layers=layers)

    # Check targets
    checks = {}
    checks["water_CA"] = (
        targets.water_contact_angle,
        stack.water_contact_angle,
        stack.water_contact_angle >= targets.water_contact_angle * 0.95,
    )
    if targets.oil_contact_angle > 0:
        checks["oil_CA"] = (
            targets.oil_contact_angle,
            stack.oil_contact_angle,
            stack.oil_contact_angle >= targets.oil_contact_angle * 0.90,
        )
    if targets.color:
        checks["color"] = (0, 0, stack.color.lower() == targets.color.lower())
    if targets.wvtr > 0:
        checks["WVTR"] = (
            targets.wvtr,
            stack.wvtr,
            stack.wvtr >= targets.wvtr * 0.8,
        )

    meets_all = all(v[2] for v in checks.values())

    return CoatingDesign(
        stack=stack,
        targets=targets,
        meets_targets=meets_all,
        target_checks=checks,
    )
