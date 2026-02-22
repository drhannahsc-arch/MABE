"""
core/surface_magnetic.py — Sprint 28: Surface Energy + Magnetic Properties

Surface energy and wetting: determines whether target solution actually
reaches binding sites on deployed materials. Magnetic force: enables
magnetic bead separation for paramagnetic complexes.

Physics:
  Contact angle: cos(θ) = (γ_SV - γ_SL) / γ_LV  (Young's equation)
  Wetting criterion: θ < 90° for spontaneous wetting
  Magnetic force: F = (χ_m × V × B × ∇B) / μ₀
  Magnetic separation: particle velocity from Stokes drag balance
"""
from dataclasses import dataclass
import math


@dataclass
class SurfaceProfile:
    """Surface energy and wettability of a scaffold material."""
    material_type: str
    surface_energy_mj_m2: float     # γ_SV (mJ/m²)
    contact_angle_water_deg: float  # θ with water
    wettability: str                # "hydrophilic", "hydrophobic", "superhydrophilic"
    spontaneous_wetting: bool       # θ < 90°
    capillary_pressure_kpa: float   # For porous materials
    surface_treatment: str          # Recommended treatment if needed
    notes: str = ""

@dataclass
class MagneticProfile:
    """Magnetic properties for separation applications."""
    complex_magnetic_moment_bm: float  # μ in Bohr magnetons
    unpaired_electrons: int
    paramagnetic: bool
    volume_susceptibility: float    # χ_v (dimensionless, SI)
    magnetic_force_fn_m: float      # F/V at B=1T, ∇B=10 T/m in fN
    separation_velocity_um_s: float # In gradient field
    separation_feasible: bool
    separation_time_min: float      # Time to separate 1 cm
    bead_recommendation: str        # Type of magnetic bead to use
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# SURFACE ENERGY DATABASE
# Sources: van Oss (1994), Owens-Wendt, measured values
# ═══════════════════════════════════════════════════════════════════════════

_SURFACE_DATA = {
    # material: (γ_SV mJ/m², θ_water °, treatment_if_needed)
    "zeolite":           (250, 15,  "None — inherently hydrophilic"),
    "zeolite_Y":         (250, 15,  "None — inherently hydrophilic"),
    "zeolite_ZSM5":      (200, 25,  "None — hydrophilic framework"),
    "mesoporous_silica":  (180, 20,  "None — silanol groups provide wettability"),
    "mesoporous_silica_MCM41": (180, 20, "None"),
    "MOF":               (120, 45,  "Depends on linker; some are hydrophobic"),
    "MOF_UiO66":         (130, 40,  "Moderately hydrophilic Zr-oxo nodes"),
    "MOF_MIL101":        (110, 50,  "Large pores compensate moderate wettability"),
    "LDH":               (200, 20,  "None — charged layers attract water"),
    "mip":               (45,  85,  "Polymer surface often hydrophobic. Consider hydrophilic monomer"),
    "cof":               (60,  75,  "Aromatic framework somewhat hydrophobic"),
    "coordination_cage": (80,  60,  "Depends on exterior ligands"),
    "carbon_nanotube":   (30,  110, "HYDROPHOBIC. Requires oxidation or surfactant coating"),
    "dendrimer_PAMAM_G4": (55, 35,  "Amine termini provide hydrophilicity"),
    "dna_origami":       (300, 5,   "None — DNA is highly hydrophilic"),
    "dna_origami_icosahedron": (300, 5, "None"),
    "dna_origami_tetrahedron": (300, 5, "None"),
    "aptamer":           (280, 8,   "None — nucleic acid is hydrophilic"),
    "peptide":           (150, 30,  "Depends on sequence; charged residues help"),
    "gold_nanoparticle": (1000, 60, "Bare Au is hydrophilic; thiol SAMs make it hydrophobic"),
    "iron_oxide_bead":   (100, 40,  "Moderate; polymer coating determines wettability"),
}


def get_surface_profile(material_type, pore_diameter_nm=0.0):
    """Get surface energy and wettability for a scaffold material."""
    key = material_type.lower().replace(" ", "_")

    # Try exact match, then partial
    data = _SURFACE_DATA.get(key)
    if data is None:
        for k in _SURFACE_DATA:
            if k in key or key in k:
                data = _SURFACE_DATA[k]
                break

    if data is None:
        return SurfaceProfile(material_type, 100, 60, "moderate", True, 0, "Unknown material")

    gamma, theta, treatment = data

    if theta < 10:
        wettability = "superhydrophilic"
    elif theta < 90:
        wettability = "hydrophilic"
    else:
        wettability = "hydrophobic"

    spontaneous = theta < 90

    # Capillary pressure for porous materials (Washburn equation)
    # P_cap = 2γcos(θ) / r
    cap_pressure = 0.0
    if pore_diameter_nm > 0:
        r_m = pore_diameter_nm * 1e-9 / 2.0
        gamma_lv = 72.8e-3  # Water surface tension N/m at 25°C
        theta_rad = math.radians(theta)
        if r_m > 0:
            cap_pressure = 2 * gamma_lv * math.cos(theta_rad) / r_m / 1000  # kPa

    notes = ""
    if not spontaneous:
        notes = f"θ={theta}° — aqueous solution will NOT spontaneously wet this material. " \
                f"Treatment: {treatment}"
    elif pore_diameter_nm > 0 and cap_pressure > 0:
        notes = f"Capillary-driven infiltration at {cap_pressure:.0f} kPa"

    return SurfaceProfile(
        material_type=material_type,
        surface_energy_mj_m2=gamma,
        contact_angle_water_deg=theta,
        wettability=wettability,
        spontaneous_wetting=spontaneous,
        capillary_pressure_kpa=round(cap_pressure, 1),
        surface_treatment=treatment,
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAGNETIC PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

_MU_BOHR = 9.274e-24   # J/T
_MU_0 = 4 * math.pi * 1e-7  # T·m/A


def compute_magnetic_properties(
    unpaired_electrons, particle_diameter_um=1.0,
    field_strength_T=1.0, gradient_T_per_m=10.0,
):
    """Compute magnetic properties and separation feasibility.

    Args:
        unpaired_electrons: From spin_state module
        particle_diameter_um: Scaffold/bead particle size
        field_strength_T: Applied magnetic field
        gradient_T_per_m: Field gradient (higher = faster separation)
    """
    # Spin-only magnetic moment
    n = unpaired_electrons
    mu_bm = math.sqrt(n * (n + 2)) if n > 0 else 0.0
    paramagnetic = n > 0

    if not paramagnetic:
        return MagneticProfile(
            0.0, 0, False, 0.0, 0.0, 0.0, False, 1e12, "N/A",
            "Diamagnetic complex — no magnetic separation without magnetic bead carrier")

    # Volume susceptibility (paramagnetic, Curie law at 298K)
    # χ_v = n_ions × μ₀ × μ² / (3 × k_B × T × V)
    # For a single ion in a particle:
    mu_j = mu_bm * _MU_BOHR
    kT = 1.381e-23 * 298.15
    chi_molar = _MU_0 * mu_j**2 / (3 * kT)  # Per ion

    # Volume of particle
    r_m = particle_diameter_um * 1e-6 / 2.0
    V_particle = (4/3) * math.pi * r_m**3

    # Force on particle: F = (χ × V × B × ∇B) / μ₀
    # For paramagnetic bead loaded with ions:
    # Assume ~1000 ions per particle for realistic loading
    n_ions_per_particle = 1000
    chi_particle = chi_molar * n_ions_per_particle

    F = chi_particle * V_particle * field_strength_T * gradient_T_per_m / _MU_0
    F_fn = F * 1e15  # Convert to femtonewtons

    # Stokes drag balance: F_mag = 6πηrv → v = F/(6πηr)
    eta = 8.9e-4  # Water viscosity Pa·s
    v = F / (6 * math.pi * eta * r_m) if r_m > 0 else 0.0
    v_um_s = v * 1e6

    # Time to traverse 1 cm
    t_1cm = 0.01 / v if v > 0 else 1e12
    t_min = t_1cm / 60

    # Feasibility
    feasible = v_um_s > 1.0 and t_min < 60  # >1 µm/s, <1 hour

    bead = "N/A"
    if not feasible and unpaired_electrons > 0:
        bead = "Fe3O4 bead (1-5 µm) with surface-conjugated binder"
        feasible_with_bead = True
    elif feasible:
        bead = "Direct magnetic separation possible"
    else:
        bead = "Fe3O4 bead recommended"

    notes = ""
    if t_min > 60 and paramagnetic:
        notes = (f"Intrinsic paramagnetism too weak for direct separation "
                 f"(v={v_um_s:.2f} µm/s). Use superparamagnetic Fe₃O₄ beads "
                 f"(χ ~ 10⁴× larger) with surface-conjugated binder.")

    return MagneticProfile(
        complex_magnetic_moment_bm=round(mu_bm, 2),
        unpaired_electrons=n,
        paramagnetic=paramagnetic,
        volume_susceptibility=chi_particle,
        magnetic_force_fn_m=round(F_fn, 4),
        separation_velocity_um_s=round(v_um_s, 4),
        separation_feasible=feasible,
        separation_time_min=round(t_min, 2),
        bead_recommendation=bead,
        notes=notes,
    )


def recommend_magnetic_strategy(unpaired_electrons, scaffold_type,
                                  target_formula=""):
    """High-level recommendation for magnetic-based capture/separation."""
    mag = compute_magnetic_properties(unpaired_electrons)

    if unpaired_electrons == 0:
        return {
            "strategy": "magnetic_bead_conjugation",
            "rationale": "Diamagnetic complex requires external magnetic carrier",
            "bead_type": "Fe₃O₄@SiO₂ (core-shell, 1-5 µm)",
            "conjugation": f"Surface-functionalize bead with {scaffold_type} binder",
            "separation": "Standard MACS (magnetic-activated cell sorting) protocol",
            "field": "Permanent magnet (0.5 T) or electromagnet",
        }

    if mag.separation_feasible:
        return {
            "strategy": "direct_magnetic_separation",
            "rationale": f"μ={mag.complex_magnetic_moment_bm:.1f} BM, "
                         f"v={mag.separation_velocity_um_s:.1f} µm/s — "
                         f"intrinsic paramagnetism sufficient",
            "separation_time": f"{mag.separation_time_min:.1f} min per cm",
            "field": "1 T, 10 T/m gradient",
        }

    return {
        "strategy": "magnetic_bead_assisted",
        "rationale": f"μ={mag.complex_magnetic_moment_bm:.1f} BM — paramagnetic but "
                     f"insufficient for direct separation at particle scale",
        "bead_type": "Fe₃O₄@SiO₂ or Fe₃O₄@polymer",
        "conjugation": f"Conjugate {scaffold_type} binder to bead surface",
        "advantage": "Bead provides 10⁴× stronger magnetic response than "
                     "intrinsic paramagnetism",
    }

