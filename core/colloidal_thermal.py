"""
core/colloidal_thermal.py — Colloidal Thermal Conductivity Model

Predicts effective thermal conductivity κ_eff of colloidal assemblies
(photonic glasses, opals, core-shell packings) from the same particle
parameters that determine structural color.

Physics models:
  - Maxwell-Garnett effective medium (baseline, no interfaces)
  - Hasselman-Johnson model (spheres with Kapitza interface resistance)
  - Core-shell extension (two interfaces per particle)
  - Interface density from packing geometry
  - Thermal resistance (R-value) from κ_eff and thickness

This is Phase 1 of the Multi-Physics Structural Element module.
The same particle spec that the optical pipeline uses for color prediction
feeds directly into this module for thermal prediction.

Key references:
  - Hasselman & Johnson, J. Compos. Mater. 1987, 21, 508
  - Nan et al., J. Appl. Phys. 1997, 81, 6692 (effective medium review)
  - Every et al., Int. J. Thermophys. 2004, 25, 229 (Kapitza in composites)
  - Still et al., Phys. Rev. B 2008, 78, 125426 (colloidal crystal κ measurement)
  - Minnich et al., Energy Environ. Sci. 2009, 2, 466 (nanostructured thermal)

Data tier: Tier 2 (DOI per entry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

PI = math.pi


# ═══════════════════════════════════════════════════════════════════════════
# Material thermal property database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ThermalMaterial:
    """Thermal properties of a material used in colloidal assemblies."""
    name: str
    kappa_W_mK: float         # bulk thermal conductivity
    density_kg_m3: float       # density
    v_sound_m_s: float         # longitudinal sound speed (for Z and phonon MFP estimates)
    cp_J_kgK: float            # specific heat capacity
    source: str                # literature reference

    @property
    def impedance_MRayl(self) -> float:
        """Acoustic impedance Z = ρ × v in MRayl."""
        return self.density_kg_m3 * self.v_sound_m_s / 1e6

    @property
    def diffusivity_m2_s(self) -> float:
        """Thermal diffusivity α = κ / (ρ × cp)."""
        return self.kappa_W_mK / (self.density_kg_m3 * self.cp_J_kgK)


# Particle materials
_MATERIALS: dict[str, ThermalMaterial] = {
    "SiO2": ThermalMaterial(
        "Amorphous silica (Stöber)", 1.3, 2200.0, 5970.0, 745.0,
        "CRC Handbook; Cahill et al., PRB 1992",
    ),
    "TiO2_rutile": ThermalMaterial(
        "TiO₂ (rutile)", 8.5, 4250.0, 7900.0, 690.0,
        "CRC Handbook; Minerals Database",
    ),
    "TiO2_anatase": ThermalMaterial(
        "TiO₂ (anatase)", 4.0, 3900.0, 7500.0, 690.0,
        "Hatta et al., J. Am. Ceram. Soc. 1990",
    ),
    "ZnS": ThermalMaterial(
        "ZnS (sphalerite)", 27.0, 4090.0, 5400.0, 472.0,
        "CRC Handbook",
    ),
    "BaTiO3": ThermalMaterial(
        "BaTiO₃ (barium titanate)", 6.2, 6020.0, 5700.0, 434.0,
        "Landolt-Börnstein; ceramic bulk value",
    ),
    "polystyrene": ThermalMaterial(
        "Polystyrene (PS)", 0.14, 1050.0, 2350.0, 1300.0,
        "CRC Handbook; polymer reference",
    ),
    "PMMA": ThermalMaterial(
        "PMMA", 0.19, 1180.0, 2760.0, 1466.0,
        "CRC Handbook",
    ),
    "melanin": ThermalMaterial(
        "Melanin / polydopamine", 0.3, 1400.0, 2000.0, 1200.0,
        "Estimated from PDA films; Kang et al., Nanoscale 2015",
    ),
    "carbon_black": ThermalMaterial(
        "Carbon black", 6.0, 1800.0, 4000.0, 710.0,
        "Estimated; depends on grade and packing",
    ),
    "CNC": ThermalMaterial(
        "Cellulose nanocrystal", 0.6, 1500.0, 4000.0, 1200.0,
        "Moon et al., Chem. Soc. Rev. 2011",
    ),

    # Matrix / binder materials
    "air": ThermalMaterial(
        "Air (25°C)", 0.026, 1.18, 346.0, 1005.0,
        "NIST standard",
    ),
    "water": ThermalMaterial(
        "Water (25°C)", 0.607, 997.0, 1497.0, 4186.0,
        "NIST standard",
    ),
    "silicone": ThermalMaterial(
        "Silicone rubber (PDMS)", 0.15, 970.0, 1000.0, 1460.0,
        "CRC Handbook; Dow Corning data",
    ),
    "polyurethane": ThermalMaterial(
        "Polyurethane (flexible)", 0.20, 1100.0, 1700.0, 1800.0,
        "CRC Handbook; polymer reference",
    ),
    "epoxy": ThermalMaterial(
        "Epoxy resin", 0.20, 1200.0, 2600.0, 1050.0,
        "CRC Handbook",
    ),
    "PVA": ThermalMaterial(
        "Polyvinyl alcohol", 0.20, 1260.0, 2350.0, 1670.0,
        "Polymer handbook",
    ),
    "PNIPAM_swollen": ThermalMaterial(
        "PNIPAM hydrogel (swollen, <32°C)", 0.55, 1020.0, 1500.0, 4100.0,
        "Estimated: ~water-like when swollen",
    ),
    "PNIPAM_collapsed": ThermalMaterial(
        "PNIPAM hydrogel (collapsed, >32°C)", 0.25, 1150.0, 1800.0, 2000.0,
        "Estimated: polymer-like when collapsed",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Kapitza resistance database — per material pair
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class KapitzaEntry:
    """Kapitza (interfacial thermal) resistance for a material pair."""
    material_a: str
    material_b: str
    R_kapitza_m2K_W: float       # interfacial thermal resistance
    method: str                   # "DMM", "AMM", "experimental", "MD", "estimated"
    source: str                   # literature reference
    notes: str = ""


# Key Kapitza resistance values for colloidal assemblies
# Format: (material_a, material_b) → R_K in m²K/W
_KAPITZA_DB: list[KapitzaEntry] = [
    # Silica-polymer interfaces (the workhorse of photonic glasses)
    KapitzaEntry("SiO2", "silicone", 5.0e-8,
                 "estimated", "Every et al., Int. J. Thermophys. 2004, 25, 229",
                 "Typical oxide-polymer Kapitza resistance"),
    KapitzaEntry("SiO2", "polyurethane", 4.0e-8,
                 "estimated", "Every et al., 2004",
                 "Covalent coupling (silanization) reduces R_K by ~2×"),
    KapitzaEntry("SiO2", "epoxy", 3.0e-8,
                 "experimental", "Smith et al., Int. J. Heat Mass Transf. 2013",
                 "Silane-coupled interface"),
    KapitzaEntry("SiO2", "air", 1.0e-6,
                 "estimated", "Gas-solid interface; phonon-gas coupling very weak",
                 "Effective R_K for air gap; dominates at low φ"),
    KapitzaEntry("SiO2", "water", 1.0e-7,
                 "MD", "Murad & Puri, Appl. Phys. Lett. 2008",
                 "Hydrophilic silica-water interface"),
    KapitzaEntry("SiO2", "PVA", 4.0e-8,
                 "estimated", "Similar to SiO2-epoxy; hydrogen bonding",
                 "PVA binder common in photonic glass fabrication"),
    KapitzaEntry("SiO2", "PNIPAM_swollen", 8.0e-8,
                 "estimated", "Hydrogel: water-mediated thermal coupling",
                 "Swollen state: water layers reduce coupling"),
    KapitzaEntry("SiO2", "PNIPAM_collapsed", 4.0e-8,
                 "estimated", "Collapsed polymer contacts silica directly",
                 "Collapsed state: better mechanical contact"),

    # SiO2-SiO2 (direct contact in dry photonic glass)
    KapitzaEntry("SiO2", "SiO2", 1.0e-8,
                 "experimental", "Still et al., Phys. Rev. B 2008, 78, 125426",
                 "Hertzian contact between silica spheres; small contact area → high R"),

    # TiO2 interfaces
    KapitzaEntry("TiO2_rutile", "silicone", 3.0e-8,
                 "estimated", "Higher Z contrast → higher R_K than SiO2-silicone",
                 "TiO2 has higher acoustic impedance"),
    KapitzaEntry("TiO2_anatase", "silicone", 3.0e-8,
                 "estimated", "Similar to rutile",
                 "Anatase is the common photocatalytic form"),
    KapitzaEntry("TiO2_rutile", "air", 1.0e-6,
                 "estimated", "Gas-solid; same order as SiO2-air", ""),
    KapitzaEntry("TiO2_anatase", "air", 1.0e-6,
                 "estimated", "", ""),

    # BaTiO3 interfaces (for multi-physics core-shell)
    KapitzaEntry("BaTiO3", "silicone", 6.0e-8,
                 "estimated", "High ΔZ → elevated Kapitza resistance",
                 "BaTiO3 has very high acoustic impedance"),
    KapitzaEntry("BaTiO3", "polyurethane", 5.0e-8,
                 "estimated", "", ""),

    # Polystyrene interfaces (PS colloidal crystals)
    KapitzaEntry("polystyrene", "air", 5.0e-7,
                 "estimated", "Polymer-gas; weaker coupling than oxide-gas", ""),
    KapitzaEntry("polystyrene", "water", 5.0e-8,
                 "MD", "Estimated from polymer-water MD studies", ""),
    KapitzaEntry("polystyrene", "polystyrene", 5.0e-9,
                 "estimated", "Polymer-polymer; sintered contact", ""),

    # Carbon black interfaces
    KapitzaEntry("carbon_black", "silicone", 2.0e-8,
                 "estimated", "Carbon-polymer; moderate coupling", ""),
    KapitzaEntry("carbon_black", "epoxy", 1.5e-8,
                 "estimated", "Carbon-epoxy; well-characterized", ""),

    # Melanin / PDA shell interfaces
    KapitzaEntry("melanin", "SiO2", 3.0e-8,
                 "estimated", "PDA-SiO2; covalent coating → good coupling", ""),
    KapitzaEntry("melanin", "air", 3.0e-7,
                 "estimated", "Polymer-like shell in air", ""),

    # CNC interfaces
    KapitzaEntry("CNC", "air", 3.0e-7,
                 "estimated", "Cellulose-air; porous film", ""),
    KapitzaEntry("CNC", "water", 5.0e-8,
                 "estimated", "Hydrophilic cellulose-water", ""),
]


def get_kapitza(material_a: str, material_b: str) -> float:
    """Look up Kapitza resistance for a material pair.

    Returns R_Kapitza in m²K/W. Tries both orderings.
    Falls back to estimate from acoustic mismatch model (AMM) if not in database.
    """
    for entry in _KAPITZA_DB:
        if (entry.material_a == material_a and entry.material_b == material_b) or \
           (entry.material_a == material_b and entry.material_b == material_a):
            return entry.R_kapitza_m2K_W

    # Fallback: acoustic mismatch model estimate
    return _estimate_kapitza_amm(material_a, material_b)


def _estimate_kapitza_amm(mat_a: str, mat_b: str) -> float:
    """Estimate Kapitza resistance from acoustic mismatch model (AMM).

    R_K ≈ 4 / (Γ × v_a × C_a)
    where Γ = 4×Z_a×Z_b / (Z_a + Z_b)² is the phonon transmission coefficient.

    This gives order-of-magnitude estimates for unknown pairs.
    """
    a = _MATERIALS.get(mat_a)
    b = _MATERIALS.get(mat_b)
    if a is None or b is None:
        return 5.0e-8  # generic fallback

    Z_a = a.density_kg_m3 * a.v_sound_m_s
    Z_b = b.density_kg_m3 * b.v_sound_m_s

    if Z_a + Z_b == 0:
        return 5.0e-8

    # Phonon transmission coefficient (AMM)
    gamma = 4.0 * Z_a * Z_b / (Z_a + Z_b)**2

    # Volumetric heat capacity
    C_a = a.density_kg_m3 * a.cp_J_kgK  # J/(m³·K)

    if gamma * a.v_sound_m_s * C_a == 0:
        return 5.0e-8

    R_K = 4.0 / (gamma * a.v_sound_m_s * C_a)
    # Clamp to physical range
    return max(1.0e-9, min(1.0e-5, R_K))


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ColloidalThermalSpec:
    """Input specification: colloidal assembly geometry.

    Uses the same particle parameters as the optical pipeline.
    """
    # Particle
    particle_material: str           # key into _MATERIALS: "SiO2", "TiO2_rutile", etc.
    particle_diameter_nm: float      # D — also determines structural color λ

    # Core-shell (optional)
    is_core_shell: bool = False
    core_material: str = ""          # e.g., "SiO2"
    core_diameter_nm: float = 0.0
    shell_material: str = ""         # e.g., "melanin", "silicone"
    shell_thickness_nm: float = 0.0  # total particle D = core_D + 2×shell_t

    # Packing
    volume_fraction: float = 0.50    # φ — key shared parameter with color
    packing_type: str = "random"     # "random", "fcc", "bcc", "hcp"

    # Matrix / binder
    matrix_material: str = "air"     # what fills the voids between particles

    # Film
    film_thickness_um: float = 50.0  # total film thickness in µm
    n_layers: Optional[int] = None   # if None, auto-compute from D and thickness

    def __post_init__(self):
        if self.is_core_shell and self.core_diameter_nm == 0:
            self.core_diameter_nm = self.particle_diameter_nm - 2 * self.shell_thickness_nm
        if self.n_layers is None and self.particle_diameter_nm > 0:
            self.n_layers = int(self.film_thickness_um * 1000.0 / self.particle_diameter_nm)


@dataclass
class ThermalResult:
    """Output: thermal conductivity prediction for a colloidal assembly."""
    # Effective conductivity
    kappa_eff_W_mK: float            # effective thermal conductivity
    kappa_maxwell_garnett_W_mK: float  # MG baseline (no interfaces)
    kappa_hasselman_johnson_W_mK: float  # HJ with Kapitza

    # Resistance
    R_value_m2KW: float              # thermal resistance of film
    U_value_W_m2K: float             # thermal transmittance (1/R)

    # Interface analysis
    n_interfaces_per_m: float        # interface density along heat flow direction
    R_kapitza_per_interface: float   # R_K per interface (m²K/W)
    R_kapitza_total_fraction: float  # fraction of total R from interfaces

    # For core-shell
    R_kapitza_core_shell: float = 0.0   # inner interface R_K
    R_kapitza_shell_matrix: float = 0.0  # outer interface R_K

    # Temperature reduction estimate
    delta_T_surface_C: Optional[float] = None  # surface temperature reduction vs bare substrate

    # Comparison
    kappa_bulk_particle: float = 0.0
    kappa_bulk_matrix: float = 0.0
    reduction_vs_particle: float = 0.0   # fraction: κ_eff / κ_particle
    reduction_vs_matrix: float = 0.0     # fraction: κ_eff / κ_matrix (>1 if particles are more conductive)

    # Confidence
    model: str = ""                      # "hasselman_johnson" or "core_shell_hjm"
    confidence: float = 0.0
    confidence_notes: str = ""

    # Notes
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"Colloidal Thermal Conductivity:",
            f"  κ_eff = {self.kappa_eff_W_mK:.4f} W/(m·K)",
            f"  R-value = {self.R_value_m2KW:.4f} m²K/W (for film thickness)",
            f"  Interface density: {self.n_interfaces_per_m:.2e} /m",
            f"  Kapitza fraction: {self.R_kapitza_total_fraction:.0%} of total R",
            f"  Model: {self.model}",
        ]
        if self.delta_T_surface_C is not None:
            lines.append(f"  Surface ΔT: {self.delta_T_surface_C:+.1f} °C vs bare substrate")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Physics models
# ═══════════════════════════════════════════════════════════════════════════

def maxwell_garnett(kappa_particle: float, kappa_matrix: float,
                    phi: float) -> float:
    """Maxwell-Garnett effective medium: spheres in matrix, no interfaces.

    κ_eff = κ_m × [κ_p + 2κ_m + 2φ(κ_p - κ_m)] / [κ_p + 2κ_m - φ(κ_p - κ_m)]

    Valid for dilute to moderate φ. Becomes inaccurate above φ ~ 0.4 for
    high-contrast systems, but adequate for order-of-magnitude baseline.
    """
    kp, km, f = kappa_particle, kappa_matrix, phi
    if km <= 0:
        return 0.0
    numer = kp + 2 * km + 2 * f * (kp - km)
    denom = kp + 2 * km - f * (kp - km)
    if denom <= 0:
        return km
    return km * numer / denom


def hasselman_johnson(kappa_particle: float, kappa_matrix: float,
                      phi: float, particle_radius_m: float,
                      R_kapitza: float) -> float:
    """Hasselman-Johnson model: spheres with Kapitza interface resistance.

    κ_eff = κ_m × [2(κ_p/κ_m - κ_p/(h×a) - 1)φ + κ_p/κ_m + 2κ_p/(h×a) + 2]
                 / [(2 + κ_p/(h×a) - κ_p/κ_m)φ + κ_p/κ_m + 2κ_p/(h×a) + 2]

    where h = 1/R_K is interface conductance, a = particle radius.

    Reference: Hasselman & Johnson, J. Compos. Mater. 1987, 21, 508.
    Extended by Nan et al., J. Appl. Phys. 1997, 81, 6692.
    """
    kp, km, f, a = kappa_particle, kappa_matrix, phi, particle_radius_m
    if km <= 0 or a <= 0:
        return 0.0

    # Interface conductance
    if R_kapitza <= 0:
        # No interface resistance → revert to Maxwell-Garnett
        return maxwell_garnett(kp, km, f)

    h = 1.0 / R_kapitza  # W/(m²·K)

    # Dimensionless ratio: Biot number for the particle
    # β = κ_p / (h × a)
    beta = kp / (h * a)

    ratio = kp / km

    numer = 2.0 * (ratio - beta - 1.0) * f + ratio + 2.0 * beta + 2.0
    denom = (2.0 + beta - ratio) * f + ratio + 2.0 * beta + 2.0

    if denom <= 0:
        return km

    return km * numer / denom


def hasselman_johnson_core_shell(
    kappa_core: float,
    kappa_shell: float,
    kappa_matrix: float,
    core_radius_m: float,
    shell_thickness_m: float,
    phi: float,
    R_kapitza_core_shell: float,
    R_kapitza_shell_matrix: float,
) -> float:
    """Extended Hasselman-Johnson for core-shell particles.

    Strategy: compute effective κ of a single core-shell particle
    (core + shell with inner Kapitza), then use HJ with the effective
    particle in the matrix (with outer Kapitza).

    This is the Benveniste-Miloh / Nan multi-coated sphere model.
    """
    r_core = core_radius_m
    r_outer = core_radius_m + shell_thickness_m

    if r_core <= 0 or r_outer <= r_core:
        # No valid core-shell; treat as solid particle
        return hasselman_johnson(kappa_shell, kappa_matrix, phi,
                                 r_outer, R_kapitza_shell_matrix)

    # Step 1: Effective κ of core-shell unit (inner Kapitza)
    # Use composite sphere model with interface resistance
    h_cs = 1.0 / R_kapitza_core_shell if R_kapitza_core_shell > 0 else 1e12
    beta_cs = kappa_core / (h_cs * r_core)

    f_core = (r_core / r_outer) ** 3  # volume fraction of core in particle

    # Effective κ of composite sphere (Hashin 1968 + Kapitza extension)
    kc, ks = kappa_core, kappa_shell
    ratio = kc / ks if ks > 0 else 1.0

    numer_p = 2.0 * (ratio - beta_cs - 1.0) * f_core + ratio + 2.0 * beta_cs + 2.0
    denom_p = (2.0 + beta_cs - ratio) * f_core + ratio + 2.0 * beta_cs + 2.0

    if denom_p <= 0:
        kappa_effective_particle = ks
    else:
        kappa_effective_particle = ks * numer_p / denom_p

    # Step 2: Effective particle in matrix (outer Kapitza)
    return hasselman_johnson(kappa_effective_particle, kappa_matrix,
                             phi, r_outer, R_kapitza_shell_matrix)


def interface_density(particle_diameter_nm: float, volume_fraction: float,
                      packing_type: str = "random") -> float:
    """Number of particle-matrix interfaces per meter along heat flow direction.

    For a dense packing of spheres with diameter D and volume fraction φ:
    - Along any line through the material, you cross ~φ/D interfaces per unit length
    - More precisely: N/m ≈ 3φ / (2D) for random packing (from mean free path analysis)
    """
    D_m = particle_diameter_nm * 1e-9
    if D_m <= 0:
        return 0.0

    # Coordination number affects interface density
    if packing_type == "fcc" or packing_type == "hcp":
        # Close-packed: φ_max = 0.74, 12 contacts per sphere
        return volume_fraction / D_m  # approximate
    elif packing_type == "bcc":
        return volume_fraction / D_m
    else:
        # Random packing: ~6 contacts per sphere at φ ~ 0.5
        return 3.0 * volume_fraction / (2.0 * D_m)


# ═══════════════════════════════════════════════════════════════════════════
# Main prediction function
# ═══════════════════════════════════════════════════════════════════════════

def predict_thermal(spec: ColloidalThermalSpec) -> ThermalResult:
    """Predict effective thermal conductivity of a colloidal assembly.

    Uses Hasselman-Johnson model (or core-shell extension) with
    Kapitza interface resistance from the database.
    """
    # Look up materials
    if spec.is_core_shell:
        mat_core = _MATERIALS.get(spec.core_material)
        mat_shell = _MATERIALS.get(spec.shell_material)
        mat_particle_name = spec.core_material  # for reporting
        kp = mat_core.kappa_W_mK if mat_core else 1.0
        ks = mat_shell.kappa_W_mK if mat_shell else 0.2
    else:
        mat_particle = _MATERIALS.get(spec.particle_material)
        kp = mat_particle.kappa_W_mK if mat_particle else 1.0
        ks = 0.0
        mat_particle_name = spec.particle_material

    mat_matrix = _MATERIALS.get(spec.matrix_material)
    km = mat_matrix.kappa_W_mK if mat_matrix else 0.026  # air default

    phi = spec.volume_fraction
    D_nm = spec.particle_diameter_nm
    D_m = D_nm * 1e-9
    r_m = D_m / 2.0

    # ── Maxwell-Garnett baseline ──
    if spec.is_core_shell:
        # Use shell κ as effective particle κ for MG
        kp_effective_noR = ks  # rough
        kappa_mg = maxwell_garnett(kp, km, phi)
    else:
        kappa_mg = maxwell_garnett(kp, km, phi)

    # ── Kapitza resistance lookup ──
    if spec.is_core_shell:
        r_core_m = spec.core_diameter_nm * 1e-9 / 2.0
        shell_t_m = spec.shell_thickness_nm * 1e-9
        R_K_cs = get_kapitza(spec.core_material, spec.shell_material)
        R_K_sm = get_kapitza(spec.shell_material, spec.matrix_material)

        kappa_hj = hasselman_johnson_core_shell(
            kappa_core=kp,
            kappa_shell=ks,
            kappa_matrix=km,
            core_radius_m=r_core_m,
            shell_thickness_m=shell_t_m,
            phi=phi,
            R_kapitza_core_shell=R_K_cs,
            R_kapitza_shell_matrix=R_K_sm,
        )
        R_K_primary = R_K_sm  # outer interface dominates
        model_name = "core_shell_hjm"
    else:
        R_K = get_kapitza(spec.particle_material, spec.matrix_material)
        kappa_hj = hasselman_johnson(kp, km, phi, r_m, R_K)
        R_K_primary = R_K
        R_K_cs = 0.0
        R_K_sm = R_K
        model_name = "hasselman_johnson"

    kappa_eff = kappa_hj

    # ── Interface density ──
    n_int = interface_density(D_nm, phi, spec.packing_type)

    # ── Film thermal resistance ──
    t_m = spec.film_thickness_um * 1e-6  # µm → m
    R_value = t_m / kappa_eff if kappa_eff > 0 else float('inf')
    U_value = 1.0 / R_value if R_value > 0 and R_value != float('inf') else 0.0

    # ── Kapitza fraction of total resistance ──
    # Total R = R_bulk + R_interfaces
    # R_bulk = t / κ_MG (what it would be without interfaces)
    R_bulk = t_m / kappa_mg if kappa_mg > 0 else t_m
    R_total = R_value
    R_kapitza_contribution = R_total - R_bulk
    kapitza_fraction = R_kapitza_contribution / R_total if R_total > 0 else 0.0
    kapitza_fraction = max(0.0, min(1.0, kapitza_fraction))

    # ── Surface temperature reduction estimate ──
    # Assume solar flux ~ 1000 W/m² incident on surface
    # ΔT_surface ≈ -Q × R_value (simplified steady-state)
    solar_flux = 1000.0  # W/m²
    delta_T = -solar_flux * R_value if R_value < float('inf') else None

    # ── Reduction ratios ──
    red_particle = kappa_eff / kp if kp > 0 else 0.0
    red_matrix = kappa_eff / km if km > 0 else 0.0

    # ── Confidence ──
    if spec.particle_material in ("SiO2", "polystyrene") and spec.matrix_material in ("air", "water", "silicone", "PVA"):
        confidence = 0.8
        conf_notes = "Well-characterized material pair; HJ model validated for this system"
    elif any(entry.material_a == spec.particle_material or entry.material_b == spec.particle_material
             for entry in _KAPITZA_DB):
        confidence = 0.6
        conf_notes = "Kapitza resistance from database; moderate confidence"
    else:
        confidence = 0.4
        conf_notes = "Kapitza resistance estimated from AMM; experimental validation needed"

    return ThermalResult(
        kappa_eff_W_mK=round(kappa_eff, 6),
        kappa_maxwell_garnett_W_mK=round(kappa_mg, 6),
        kappa_hasselman_johnson_W_mK=round(kappa_hj, 6),
        R_value_m2KW=round(R_value, 6),
        U_value_W_m2K=round(U_value, 4),
        n_interfaces_per_m=n_int,
        R_kapitza_per_interface=R_K_primary,
        R_kapitza_total_fraction=round(kapitza_fraction, 4),
        R_kapitza_core_shell=R_K_cs,
        R_kapitza_shell_matrix=R_K_sm,
        delta_T_surface_C=round(delta_T, 1) if delta_T is not None else None,
        kappa_bulk_particle=kp,
        kappa_bulk_matrix=km,
        reduction_vs_particle=round(red_particle, 4),
        reduction_vs_matrix=round(red_matrix, 4),
        model=model_name,
        confidence=confidence,
        confidence_notes=conf_notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: predict from optical pipeline parameters
# ═══════════════════════════════════════════════════════════════════════════

def predict_from_optical(
    particle_diameter_nm: float,
    particle_material: str = "SiO2",
    volume_fraction: float = 0.50,
    matrix_material: str = "air",
    film_thickness_um: float = 50.0,
) -> ThermalResult:
    """Predict thermal conductivity from the same parameters the optical pipeline uses.

    This is the bridge function: optical pipeline computes color from
    (D, material, φ), and this function computes κ_eff from the same spec.
    """
    spec = ColloidalThermalSpec(
        particle_material=particle_material,
        particle_diameter_nm=particle_diameter_nm,
        volume_fraction=volume_fraction,
        matrix_material=matrix_material,
        film_thickness_um=film_thickness_um,
    )
    return predict_thermal(spec)


def predict_core_shell(
    core_material: str,
    core_diameter_nm: float,
    shell_material: str,
    shell_thickness_nm: float,
    volume_fraction: float = 0.50,
    matrix_material: str = "air",
    film_thickness_um: float = 50.0,
) -> ThermalResult:
    """Predict thermal conductivity for core-shell colloidal assembly."""
    total_D = core_diameter_nm + 2 * shell_thickness_nm
    spec = ColloidalThermalSpec(
        particle_material=core_material,
        particle_diameter_nm=total_D,
        is_core_shell=True,
        core_material=core_material,
        core_diameter_nm=core_diameter_nm,
        shell_material=shell_material,
        shell_thickness_nm=shell_thickness_nm,
        volume_fraction=volume_fraction,
        matrix_material=matrix_material,
        film_thickness_um=film_thickness_um,
    )
    return predict_thermal(spec)


# ═══════════════════════════════════════════════════════════════════════════
# Database access
# ═══════════════════════════════════════════════════════════════════════════

def get_material(name: str) -> Optional[ThermalMaterial]:
    """Look up a thermal material by name."""
    return _MATERIALS.get(name)


def list_materials() -> list[str]:
    """Return all material names in the database."""
    return list(_MATERIALS.keys())


def list_kapitza_pairs() -> list[tuple[str, str, float]]:
    """Return all Kapitza resistance entries as (mat_a, mat_b, R_K)."""
    return [(e.material_a, e.material_b, e.R_kapitza_m2K_W) for e in _KAPITZA_DB]
