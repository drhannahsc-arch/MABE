"""
core/photonic_assembly.py -- Nanoparticle self-assembly for structural color.

Connects assembly_engine (MonomerSpec, topology) to structural color layers
in pfas_free_coating and smart_window.

Two assembly modes:
  1. Ordered (FCC opal): monodisperse spheres → close-packed crystal → Bragg color
     lambda = 2 * n_eff * d_111, angle-dependent (iridescent)
  2. Disordered (photonic glass): polydisperse spheres → short-range order only
     lambda ~ 1.86 * n_eff * d, angle-INdependent (matte structural color)

Physics (all textbook):
  Bragg:          lambda = 2 * n_eff * d * 0.816  (FCC 111)
  n_eff:          sqrt(phi * n_p^2 + (1-phi) * n_m^2)
  Photonic glass: lambda_peak ~ 1.86 * n_eff * d  (Percus-Yevick first peak)
  DLVO:           V = V_vdW + V_electrostatic  (determines assembly kinetics)
  Stokes:         v_sed = 2*r^2*(rho_p - rho_m)*g / (9*eta)  (sedimentation rate)

Particle materials (from refractive index DB):
  SiO2:  n=1.46, rho=2.2 g/cm3, cheap, easy Stober synthesis
  PS:    n=1.59, rho=1.05, emulsion polymerization
  PMMA:  n=1.49, rho=1.18, emulsion polymerization
  TiO2:  n=2.49, rho=4.23, high contrast but absorbs UV
  ZnO:   n=2.00, rho=5.61, UV-active
  melanin: n=1.74, rho=1.4, bio-derived, broadband absorber

Entry points:
  design_photonic_particles(target_color, ordered) -> PhotonicDesign
  nanoparticle_monomer(material, diameter) -> MonomerSpec
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.assembly_engine import (
    Face, FaceRole, InteractionMode, MonomerSpec,
    design_material, MaterialDesign, Topology,
)
from core.pfas_free_coating import (
    particle_size_for_color, color_from_particle_size,
)


# ---------------------------------------------------------------------------
# Particle material database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParticleMaterial:
    """Physical properties of a nanoparticle material."""
    name: str
    n_refractive: float    # at 550nm
    density_g_cm3: float
    synthesis: str         # synthesis method
    size_range_nm: Tuple[float, float]  # achievable size range
    dispersity_pct: float  # typical size dispersity (CV%)
    cost_relative: float   # 1.0 = SiO2 baseline
    uv_absorbing: bool     # absorbs UV (< 380nm)
    notes: str = ""


PARTICLE_MATERIALS: Dict[str, ParticleMaterial] = {
    "SiO2": ParticleMaterial(
        "SiO2", 1.46, 2.20, "Stober", (50, 800), 3.0, 1.0, False,
        "Standard, cheap, tunable. Stober gives excellent monodispersity."),
    "polystyrene": ParticleMaterial(
        "polystyrene", 1.59, 1.05, "emulsion_polymerization", (100, 1000), 2.0, 0.5, False,
        "Higher n than SiO2, lighter. Excellent monodispersity."),
    "PMMA": ParticleMaterial(
        "PMMA", 1.49, 1.18, "emulsion_polymerization", (100, 500), 3.0, 0.6, False,
        "Similar to PS but lower n."),
    "TiO2": ParticleMaterial(
        "TiO2", 2.49, 4.23, "sol_gel", (10, 300), 10.0, 2.0, True,
        "Very high n, strong color. UV absorber. Hard to make monodisperse."),
    "ZnO": ParticleMaterial(
        "ZnO", 2.00, 5.61, "precipitation", (20, 200), 15.0, 1.5, True,
        "High n, UV-active. Difficult monodispersity."),
    "melanin": ParticleMaterial(
        "melanin", 1.74, 1.40, "oxidative_polymerization", (100, 500), 5.0, 3.0, True,
        "Bio-derived, broadband absorber, high n. Natures structural color."),
}


# ---------------------------------------------------------------------------
# Assembly physics
# ---------------------------------------------------------------------------

def bragg_color_ordered(diameter_nm: float, n_particle: float,
                        n_medium: float = 1.0,
                        packing_fraction: float = 0.74) -> Tuple[float, str]:
    """Bragg color from FCC ordered opal.

    FCC 111 spacing: d_111 = diameter * sqrt(2/3) = diameter * 0.816
    lambda = 2 * n_eff * d_111
    """
    n_eff = math.sqrt(packing_fraction * n_particle ** 2
                      + (1 - packing_fraction) * n_medium ** 2)
    d_111 = diameter_nm * 0.816
    wavelength = 2 * n_eff * d_111
    color = _wl_to_color(wavelength)
    return round(wavelength, 1), color


def glass_color_disordered(diameter_nm: float, n_particle: float,
                           n_medium: float = 1.0,
                           packing_fraction: float = 0.55) -> Tuple[float, str]:
    """Color from photonic glass (disordered, short-range order).

    Percus-Yevick first structure factor peak:
    lambda_peak ~ 1.86 * n_eff * d
    This peak is angle-independent (no long-range order).
    """
    n_eff = math.sqrt(packing_fraction * n_particle ** 2
                      + (1 - packing_fraction) * n_medium ** 2)
    wavelength = 1.86 * n_eff * diameter_nm
    color = _wl_to_color(wavelength)
    return round(wavelength, 1), color


def diameter_for_ordered_color(target_wavelength_nm: float,
                               n_particle: float = 1.46,
                               n_medium: float = 1.0,
                               packing_fraction: float = 0.74) -> float:
    """Particle diameter needed for target Bragg color (ordered opal)."""
    n_eff = math.sqrt(packing_fraction * n_particle ** 2
                      + (1 - packing_fraction) * n_medium ** 2)
    d_111 = target_wavelength_nm / (2 * n_eff)
    diameter = d_111 / 0.816
    return round(diameter, 1)


def diameter_for_glass_color(target_wavelength_nm: float,
                             n_particle: float = 1.46,
                             n_medium: float = 1.0,
                             packing_fraction: float = 0.55) -> float:
    """Particle diameter needed for target photonic glass color."""
    n_eff = math.sqrt(packing_fraction * n_particle ** 2
                      + (1 - packing_fraction) * n_medium ** 2)
    diameter = target_wavelength_nm / (1.86 * n_eff)
    return round(diameter, 1)


def angle_shift_ordered(wavelength_0: float, theta_ext_deg: float,
                        n_eff: float = 1.30) -> float:
    """Bragg peak shift with viewing angle for ordered opal.

    lambda(theta) = lambda_0 * sqrt(1 - sin^2(theta_ext)/n_eff^2)
    """
    theta_rad = math.radians(theta_ext_deg)
    sin_ratio = math.sin(theta_rad) / n_eff
    if abs(sin_ratio) >= 1:
        return 0.0
    return wavelength_0 * math.sqrt(1 - sin_ratio ** 2)


def _wl_to_color(wl: float) -> str:
    if wl < 380: return "UV"
    if wl < 450: return "violet"
    if wl < 495: return "blue"
    if wl < 570: return "green"
    if wl < 590: return "yellow"
    if wl < 620: return "orange"
    if wl < 750: return "red"
    return "IR"


# ---------------------------------------------------------------------------
# Assembly conditions
# ---------------------------------------------------------------------------

@dataclass
class AssemblyConditions:
    """Conditions that determine ordered vs disordered assembly."""
    dispersity_pct: float = 3.0     # particle CV%. <5% -> ordered, >10% -> glass
    assembly_method: str = "sedimentation"  # sedimentation, spin_coating, spray
    temperature_C: float = 25.0
    solvent: str = "water"
    surfactant: bool = False
    absorber_loading: float = 0.0   # fraction of carbon black or melanin (0-0.3)

    @property
    def ordered(self) -> bool:
        """Predict whether assembly will be ordered or disordered."""
        if self.dispersity_pct > 8:
            return False  # too polydisperse for crystallization
        if self.assembly_method == "spray":
            return False  # spray → kinetically trapped glass
        if self.absorber_loading > 0.1:
            return False  # absorber disrupts order
        return True


# ---------------------------------------------------------------------------
# Sedimentation and DLVO
# ---------------------------------------------------------------------------

def sedimentation_velocity(diameter_nm: float, rho_particle: float,
                           rho_medium: float = 1.0,
                           eta_Pa_s: float = 1e-3) -> float:
    """Stokes sedimentation velocity (m/s).

    v = 2*r^2*(rho_p - rho_m)*g / (9*eta)
    """
    r_m = diameter_nm * 1e-9 / 2
    g = 9.81
    delta_rho = (rho_particle - rho_medium) * 1000  # g/cm3 -> kg/m3
    v = 2 * r_m ** 2 * delta_rho * g / (9 * eta_Pa_s)
    return v  # m/s


def sedimentation_time_hours(diameter_nm: float, height_cm: float,
                             rho_particle: float) -> float:
    """Time for particles to sediment through given height."""
    v = sedimentation_velocity(diameter_nm, rho_particle)
    if v <= 0:
        return float('inf')
    t_s = (height_cm / 100) / v
    return t_s / 3600


def dlvo_barrier_kT(diameter_nm: float, surface_charge_mV: float = -40.0,
                    ionic_strength_mM: float = 1.0) -> float:
    """Estimate DLVO energy barrier in kT units.

    Simplified: balance of van der Waals attraction and electrostatic repulsion.
    V_max ~ (epsilon * psi^2 * a) / (4 * kappa) - A_H * a / (12 * D_min)

    Higher barrier = more stable colloidal suspension = slower assembly.
    """
    a_m = diameter_nm * 1e-9 / 2  # radius in meters
    psi = surface_charge_mV * 1e-3  # V
    epsilon = 80 * 8.85e-12  # water permittivity
    kT = 4.11e-21  # at 300K

    # Debye length: kappa^-1 = sqrt(epsilon*kT / (2*e^2*N_A*I))
    e = 1.6e-19
    N_A = 6.022e23
    I_mol_m3 = ionic_strength_mM  # mM = mol/m3
    if I_mol_m3 > 0:
        kappa = math.sqrt(2 * e ** 2 * N_A * I_mol_m3 / (epsilon * kT))
    else:
        kappa = 1e6  # nm scale

    # Electrostatic repulsion at contact
    V_elec = epsilon * psi ** 2 * a_m / 4  # simplified
    # Hamaker constant for SiO2 in water: ~0.8e-20 J
    A_H = 0.8e-20
    D_min = 0.3e-9  # minimum approach distance
    V_vdw = -A_H * a_m / (12 * D_min)

    V_barrier = (V_elec + V_vdw) / kT
    return round(max(0, V_barrier), 1)


# ---------------------------------------------------------------------------
# Nanoparticle MonomerSpec
# ---------------------------------------------------------------------------

def nanoparticle_monomer(
    material: str = "SiO2",
    diameter_nm: float = 215.0,
    conditions: Optional[AssemblyConditions] = None,
) -> MonomerSpec:
    """Create a MonomerSpec for a nanoparticle.

    Sphere with 12 nearest-neighbor contacts (FCC) or 6 (random).
    Each contact point is a face with van der Waals interaction.
    """
    if conditions is None:
        conditions = AssemblyConditions()

    mat = PARTICLE_MATERIALS.get(material, PARTICLE_MATERIALS["SiO2"])
    ordered = conditions.ordered

    # Volume
    r_nm = diameter_nm / 2
    vol_A3 = (4 / 3) * math.pi * (r_nm * 10) ** 3  # nm -> A

    # MW estimate from density
    vol_cm3 = (4 / 3) * math.pi * (r_nm * 1e-7) ** 3
    mass_g = vol_cm3 * mat.density_g_cm3
    mw_kDa = mass_g * 6.022e23 / 1000

    # Contact faces: FCC has 12 nearest neighbors
    n_contacts = 12 if ordered else 6
    # Contact area ~ pi * (a_contact)^2, where a_contact ~ sqrt(D * deformation)
    # For hard spheres: contact area ~ 0 (point contact)
    # For soft/coated: effective area ~ 100 A^2 per contact
    contact_area = 100.0  # A^2

    faces = []
    for i in range(n_contacts):
        faces.append(Face(
            name=f"contact_{i+1}",
            role=FaceRole.STRUCTURAL,
            interaction=InteractionMode.VAN_DER_WAALS,
            n_contacts=1,
            area_A2=contact_area,
            complementary_to="self",
            notes=f"{'FCC' if ordered else 'random'} nearest neighbor",
        ))

    # Symmetry
    symmetry = "Oh" if ordered else "C1"
    rigidity = 1.0  # hard sphere

    return MonomerSpec(
        name=f"{material}_{diameter_nm:.0f}nm",
        smiles="",  # nanoparticle, no molecular SMILES
        faces=faces,
        molecular_weight=round(mw_kDa * 1000, 0),  # in Da for consistency
        monomer_volume_A3=round(vol_A3, 0),
        symmetry=symmetry,
        rigidity=rigidity,
    )


# ---------------------------------------------------------------------------
# Photonic design result
# ---------------------------------------------------------------------------

@dataclass
class PhotonicDesign:
    """Complete photonic nanoparticle assembly design."""
    # Particle
    material: str
    diameter_nm: float
    n_particle: float
    # Assembly
    ordered: bool
    packing_fraction: float
    n_eff: float
    # Color
    peak_wavelength_nm: float
    color: str
    angle_dependent: bool
    # Angle sweep (ordered only)
    angle_sweep: Dict[int, Tuple[float, str]] = field(default_factory=dict)
    # Assembly conditions
    conditions: Optional[AssemblyConditions] = None
    sedimentation_hours: float = 0.0
    dlvo_barrier_kT: float = 0.0
    # Absorber
    absorber: str = ""
    absorber_loading: float = 0.0
    # MonomerSpec + MaterialDesign
    monomer: Optional[MonomerSpec] = None
    material_design: Optional[MaterialDesign] = None
    notes: str = ""

    @property
    def summary(self) -> str:
        mode = "Ordered opal (FCC)" if self.ordered else "Photonic glass (disordered)"
        lines = [
            f"Photonic Nanoparticle Design",
            f"  Material: {self.material} (n={self.n_particle:.2f})",
            f"  Diameter: {self.diameter_nm:.0f}nm",
            f"  Assembly: {mode}",
            f"  Packing: {self.packing_fraction:.0%}",
            f"  n_eff: {self.n_eff:.2f}",
            f"  Peak: {self.peak_wavelength_nm:.0f}nm ({self.color})",
            f"  Angle-dependent: {self.angle_dependent}",
        ]
        if self.angle_sweep:
            lines.append(f"  Angle sweep:")
            for a in sorted(self.angle_sweep):
                wl, c = self.angle_sweep[a]
                lines.append(f"    {a:3d}deg: {wl:.0f}nm ({c})")
        if self.absorber:
            lines.append(f"  Absorber: {self.absorber} at {self.absorber_loading:.0%}")
        lines.append(f"  Sedimentation: {self.sedimentation_hours:.0f}h (1cm)")
        lines.append(f"  DLVO barrier: {self.dlvo_barrier_kT:.0f} kT")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main design function
# ---------------------------------------------------------------------------

_COLOR_WAVELENGTHS = {
    "violet": 420, "blue": 470, "cyan": 500,
    "green": 530, "yellow": 580, "orange": 600, "red": 650,
}


def design_photonic_particles(
    target_color: str = "blue",
    material: str = "SiO2",
    ordered: bool = True,
    absorber: str = "",
    absorber_loading: float = 0.0,
    conditions: Optional[AssemblyConditions] = None,
) -> PhotonicDesign:
    """Design nanoparticles for target structural color.

    Args:
        target_color: "blue", "green", "red", etc.
        material: particle material from PARTICLE_MATERIALS
        ordered: True = FCC opal (iridescent), False = photonic glass (matte)
        absorber: "carbon_black" or "melanin" for saturated color
        absorber_loading: fraction (0-0.3)
        conditions: assembly conditions (auto-generated if None)
    """
    target_wl = _COLOR_WAVELENGTHS.get(target_color.lower(), 530)
    mat = PARTICLE_MATERIALS.get(material, PARTICLE_MATERIALS["SiO2"])

    if conditions is None:
        conditions = AssemblyConditions(
            dispersity_pct=mat.dispersity_pct if ordered else max(10, mat.dispersity_pct),
            absorber_loading=absorber_loading,
        )

    # Compute required diameter
    if ordered:
        phi = 0.74  # FCC
        diameter = diameter_for_ordered_color(target_wl, mat.n_refractive,
                                              packing_fraction=phi)
        peak_wl, color = bragg_color_ordered(diameter, mat.n_refractive,
                                             packing_fraction=phi)
    else:
        phi = 0.55  # random close packing
        diameter = diameter_for_glass_color(target_wl, mat.n_refractive,
                                           packing_fraction=phi)
        peak_wl, color = glass_color_disordered(diameter, mat.n_refractive,
                                                packing_fraction=phi)

    # Check size range
    if diameter < mat.size_range_nm[0] or diameter > mat.size_range_nm[1]:
        notes = (f"WARNING: {diameter:.0f}nm outside synthesis range "
                 f"{mat.size_range_nm[0]:.0f}-{mat.size_range_nm[1]:.0f}nm for {material}")
    else:
        notes = ""

    n_eff = math.sqrt(phi * mat.n_refractive ** 2 + (1 - phi) * 1.0 ** 2)

    # Angle sweep for ordered
    angle_sweep = {}
    if ordered:
        for theta in [0, 15, 30, 45, 60]:
            shifted_wl = angle_shift_ordered(peak_wl, theta, n_eff)
            shifted_color = _wl_to_color(shifted_wl)
            angle_sweep[theta] = (round(shifted_wl, 0), shifted_color)

    # Sedimentation time
    sed_hours = sedimentation_time_hours(diameter, 1.0, mat.density_g_cm3)

    # DLVO barrier
    barrier = dlvo_barrier_kT(diameter)

    # Build MonomerSpec
    monomer = nanoparticle_monomer(material, diameter, conditions)
    mat_design = design_material(monomer)

    return PhotonicDesign(
        material=material,
        diameter_nm=diameter,
        n_particle=mat.n_refractive,
        ordered=ordered,
        packing_fraction=phi,
        n_eff=round(n_eff, 3),
        peak_wavelength_nm=peak_wl,
        color=color,
        angle_dependent=ordered,
        angle_sweep=angle_sweep,
        conditions=conditions,
        sedimentation_hours=round(sed_hours, 1),
        dlvo_barrier_kT=barrier,
        absorber=absorber,
        absorber_loading=absorber_loading,
        monomer=monomer,
        material_design=mat_design,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Integration with structural color layers
# ---------------------------------------------------------------------------

def photonic_structural_color_layer(
    target_color: str = "blue",
    material: str = "SiO2",
    ordered: bool = True,
    n_layers: int = 15,
) -> dict:
    """Generate parameters for structural color layer in pfas_free_coating.

    Returns dict compatible with design_structural_color() inputs.
    """
    design = design_photonic_particles(target_color, material, ordered)

    layer_thickness_um = design.diameter_nm * n_layers * 0.816 / 1000

    return {
        "particle_material": material,
        "particle_diameter_nm": design.diameter_nm,
        "n_particle": design.n_particle,
        "peak_wavelength_nm": design.peak_wavelength_nm,
        "color": design.color,
        "ordered": ordered,
        "angle_dependent": design.angle_dependent,
        "layer_thickness_um": round(layer_thickness_um, 2),
        "n_eff": design.n_eff,
        "packing_fraction": design.packing_fraction,
        "photonic_design": design,
    }
