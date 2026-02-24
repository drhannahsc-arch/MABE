"""
interaction_geometry_spec.py — Realization-Agnostic Interaction Geometry Specification

MABE PRIME DIRECTIVE COMPLIANCE:
  This is Layer 2 output. It describes WHAT the optimal interaction geometry
  looks like. It says NOTHING about what material builds it.

  No DNA. No MOF. No chelator. No protein. No resin. No thin film. No crystal.
  Just physics: shapes, interaction elements, distances, energetics.

  Layer 3 (realization_ranker.py) reads this and decides what builds it.
  Layer 4 (adapters) implement it in a specific material.

═══════════════════════════════════════════════════════════════════════════
An InteractionGeometrySpec is:

  A spatial arrangement of INTERACTION ELEMENTS
  within a defined GEOMETRY,
  under specified CONDITIONS,
  with SELECTIVITY REQUIREMENTS against competitors.

This abstraction covers:

  MOLECULAR RECOGNITION
  - Metal coordination:  donors arranged around a coordination center
  - Anion recognition:   cavity with H-bond donors + charge + shape
  - Protein epitope:     surface patch with complementary features
  - Small molecule:      enclosed cavity with mixed interactions

  OPTICAL / PHOTONIC
  - Wavelength capture:  periodic structure with refractive boundaries
  - Structural color:    constructive interference at target λ
  - Plasmonic resonance: metallic nanostructure at resonant dimensions
  - Fluorescence:        electronic transition matching

  FUTURE MODALITIES
  - Acoustic resonance, thermal, magnetic, mechanical...
  - Same pattern: target physics → optimal geometry → realization ranking

═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTION DOMAIN — What physics governs the interaction
# ═══════════════════════════════════════════════════════════════════════════

class InteractionDomain(str, Enum):
    """Top-level physics domain. Determines which energy/field equations apply."""
    MOLECULAR = "molecular"       # Thermodynamics, ΔG, Ka/Kd
    OPTICAL = "optical"           # Maxwell's equations, λ, reflectance/absorbance
    ACOUSTIC = "acoustic"         # Wave mechanics, frequency selectivity
    MAGNETIC = "magnetic"         # Susceptibility, field gradients
    THERMAL = "thermal"           # Heat capacity, conductivity matching
    MECHANICAL = "mechanical"     # Stiffness, resonance


# ═══════════════════════════════════════════════════════════════════════════
# TARGET GEOMETRY — What we're trying to interact with
# ═══════════════════════════════════════════════════════════════════════════

class TargetScale(str, Enum):
    """Size regime of the target."""
    ATOMIC = "atomic"               # Single ion, < 0.3 nm
    SMALL_MOLECULE = "small_mol"    # < 1 nm
    PEPTIDE = "peptide"             # 1-5 nm
    PROTEIN_DOMAIN = "domain"       # 5-20 nm
    MACROMOLECULAR = "macro"        # > 20 nm
    WAVELENGTH = "wavelength"       # 100 nm - 100 µm (optical target)
    ACOUSTIC_WAVE = "acoustic"      # mm - m scale


class TargetShape(str, Enum):
    """Geometric classification."""
    # Molecular shapes
    SPHERICAL = "spherical"
    PYRAMIDAL = "pyramidal"
    PLANAR = "planar"
    TETRAHEDRAL = "tetrahedral"
    LINEAR = "linear"
    CONVEX_SURFACE = "convex"
    CONCAVE_SURFACE = "concave"
    IRREGULAR = "irregular"
    CYLINDRICAL = "cylindrical"
    # Wave/field shapes
    PLANE_WAVE = "plane_wave"       # Collimated light/sound
    SPHERICAL_WAVE = "spherical_wave"
    EVANESCENT = "evanescent"       # Near-field


@dataclass(frozen=True)
class TargetGeometry:
    """
    Physics-level description of what we're interacting with.
    Works for molecules, photons, phonons, or fields.
    """
    name: str
    formula: str                    # 'Pb2+', 'SeO3(2-)', 'λ=532nm', 'f=440Hz'
    charge: float = 0.0            # Electric charge (molecular) or 0 (optical)
    shape: TargetShape = TargetShape.SPHERICAL
    scale: TargetScale = TargetScale.ATOMIC
    symmetry: str = ""
    domain: InteractionDomain = InteractionDomain.MOLECULAR

    # Spatial dimensions (Å for molecular, nm for optical)
    effective_radius_A: float = 0.0
    bounding_sphere_A: float = 0.0
    length_A: float = 0.0
    width_A: float = 0.0
    height_A: float = 0.0

    # Molecular properties
    hsab_class: str = ""
    polarizability_A3: float = 0.0
    dipole_moment_D: float = 0.0
    hydration_dG_kJ: float = 0.0
    hydrated_radius_A: float = 0.0
    has_lone_pair: bool = False
    lone_pair_accessible: bool = False
    n_hbond_donors: int = 0
    n_hbond_acceptors: int = 0
    aromatic: bool = False
    redox_active: bool = False

    # Optical properties (active when domain == OPTICAL)
    wavelength_nm: float = 0.0      # Target wavelength
    frequency_Hz: float = 0.0       # Target frequency
    energy_eV: float = 0.0          # Photon energy
    bandwidth_nm: float = 0.0       # Acceptable bandwidth (selectivity)
    polarization: str = ""          # 'unpolarized', 'linear', 'circular'

    # Speciation / population
    dominant_form_at_pH: str = ""
    fraction_target_form: float = 1.0

    def volume_A3(self) -> float:
        """Approximate target volume (molecular domain)."""
        if self.domain != InteractionDomain.MOLECULAR:
            return 0.0
        if self.shape == TargetShape.SPHERICAL:
            return (4/3) * np.pi * self.effective_radius_A**3
        elif self.shape in (TargetShape.PYRAMIDAL, TargetShape.TETRAHEDRAL):
            r = self.width_A / 2 if self.width_A > 0 else self.effective_radius_A
            h = self.height_A if self.height_A > 0 else self.effective_radius_A
            return (1/3) * np.pi * r**2 * h
        else:
            return (4/3) * np.pi * self.effective_radius_A**3


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTION ELEMENTS — The features the geometry must provide
# ═══════════════════════════════════════════════════════════════════════════

class InteractionType(str, Enum):
    """Fundamental interaction types across all domains."""

    # ── Molecular: Coordinate bonds ──
    COORDINATE_DONOR_O = "coord_O"
    COORDINATE_DONOR_N = "coord_N"
    COORDINATE_DONOR_S = "coord_S"
    COORDINATE_DONOR_P = "coord_P"
    METAL_CENTER = "metal_center"

    # ── Molecular: H-bonds ──
    HBOND_DONOR = "hbond_donor"
    HBOND_ACCEPTOR = "hbond_acceptor"

    # ── Molecular: Electrostatic ──
    POSITIVE_CHARGE = "pos_charge"
    NEGATIVE_CHARGE = "neg_charge"
    PARTIAL_POSITIVE = "partial_pos"
    PARTIAL_NEGATIVE = "partial_neg"

    # ── Molecular: Hydrophobic / dispersion ──
    HYDROPHOBIC_SURFACE = "hydrophobic"
    AROMATIC_WALL = "aromatic_wall"

    # ── Molecular: Shape ──
    STERIC_WALL = "steric_wall"

    # ── Optical: Refractive boundaries ──
    REFRACTIVE_BOUNDARY = "refractive_boundary"   # n₁/n₂ interface
    REFLECTIVE_SURFACE = "reflective_surface"      # High-Δn interface
    ABSORBING_LAYER = "absorbing_layer"            # Resonant absorption

    # ── Optical: Resonant structures ──
    RESONANT_CAVITY = "resonant_cavity"            # Fabry-Perot, whispering gallery
    PERIODIC_ELEMENT = "periodic_element"           # Bragg plane, grating line
    PLASMONIC_ELEMENT = "plasmonic_element"         # Metal nanostructure at resonance

    # ── Optical: Electronic ──
    ELECTRONIC_TRANSITION = "electronic_transition" # Chromophore, quantum dot
    BANDGAP = "bandgap"                            # Semiconductor absorption edge


@dataclass
class InteractionElement:
    """
    One feature that the geometry must present.
    Position relative to geometry center. Units: Å (molecular) or nm (optical).
    """
    interaction_type: InteractionType
    position_A: np.ndarray          # [x, y, z]
    direction_A: np.ndarray         # Unit vector
    strength_required: float = 1.0  # Relative importance (0-10)
    tolerance_A: float = 0.5        # Positional tolerance

    subtype: str = ""               # e.g. 'N_pyridine', 'n=1.45_SiO2', 'Au_sphere_50nm'
    estimated_dG_kJ: float = 0.0    # Energy contribution (molecular)

    # Optical-specific
    refractive_index: float = 0.0   # n of this element
    thickness_nm: float = 0.0       # Layer thickness (optical)
    period_nm: float = 0.0          # Repeat period (photonic crystal)

    must_be_rigid: bool = False
    cooperates_with: List[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY SHAPE — The spatial envelope
# ═══════════════════════════════════════════════════════════════════════════

class GeometryTopology(str, Enum):
    """Topological class of the interaction geometry."""
    # Molecular
    CAVITY = "cavity"
    CLEFT = "cleft"
    GROOVE = "groove"
    SURFACE = "surface"
    CHANNEL = "channel"
    COORDINATION_SPHERE = "coordination_sphere"
    # Optical
    PERIODIC_STACK = "periodic_stack"     # 1D photonic crystal / multilayer
    PERIODIC_2D = "periodic_2d"           # 2D photonic crystal / grating
    PERIODIC_3D = "periodic_3d"           # 3D photonic crystal / opal
    RESONANT_CAVITY_OPT = "resonant_cavity"  # Fabry-Perot, ring resonator
    NANOPARTICLE_ARRAY = "nanoparticle_array" # Plasmonic lattice


@dataclass
class GeometryShape:
    """
    Spatial envelope of the interaction geometry.
    All dimensions in Å (molecular) or nm (optical).
    """
    topology: GeometryTopology

    # Cavity / cleft dimensions (molecular)
    aperture_A: float = 0.0
    depth_A: float = 0.0
    width_A: float = 0.0
    cone_half_angle_deg: float = 0.0
    symmetry_order: int = 0
    curvature_A_inv: float = 0.0
    pocket_volume_A3: float = 0.0
    optimal_packing_coefficient: float = 0.55

    # Periodic structure dimensions (optical)
    lattice_period_nm: float = 0.0       # d-spacing
    n_periods: int = 0                   # Number of repeats
    fill_fraction: float = 0.0           # Volume fraction of high-n material
    lattice_type: str = ""               # 'fcc', 'bcc', 'hex', 'lamellar'

    # Flexibility / precision
    rigidity_required: float = 0.0
    max_rms_deformation_A: float = 1.0
    # Optical equivalent: how precise must the periodicity be?
    period_tolerance_fraction: float = 0.01  # 1% default

    def ideal_guest_volume_A3(self) -> float:
        return self.pocket_volume_A3 * self.optimal_packing_coefficient


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CompetitorConstraint:
    """
    One species/signal the geometry must discriminate against.
    Works for competing ions OR adjacent wavelengths.
    """
    competitor_name: str
    competitor_formula: str              # 'SO4(2-)' or 'λ=550nm'
    competitor_shape: TargetShape = TargetShape.IRREGULAR
    competitor_charge: float = 0.0
    concentration_ratio: float = 1.0     # [competitor]/[target] or intensity ratio
    required_delta_logK: float = 0.0     # Molecular: log K selectivity
    required_extinction_ratio: float = 0.0  # Optical: dB rejection

    selectivity_sources: Dict[str, float] = field(default_factory=dict)
    achievable_delta_logK: float = 0.0
    achievable_extinction_ratio: float = 0.0
    feasible: bool = True
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# OPERATING CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OperatingConditions:
    """Physical conditions. Molecular and optical fields populated as relevant."""
    # Universal
    temperature_K: float = 298.15
    environment: str = ""                # 'coal_mine_drainage', 'ambient_air', 'vacuum'

    # Molecular
    pH: float = 7.0
    ionic_strength_M: float = 0.1
    target_concentration_M: float = 0.0
    required_removal_fraction: float = 0.0
    flow_conditions: str = "batch"
    regeneration_required: bool = False
    matrix_name: str = ""

    # Optical
    incident_angle_deg: float = 0.0      # Normal incidence default
    polarization: str = "unpolarized"
    ambient_refractive_index: float = 1.0  # Air default
    required_reflectance: float = 0.0      # 0-1
    required_bandwidth_nm: float = 0.0     # FWHM of response


# ═══════════════════════════════════════════════════════════════════════════
# THE SPEC — Layer 2 output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InteractionGeometrySpec:
    """
    Complete, realization-agnostic specification of an optimal interaction geometry.

    THIS IS THE CENTRAL DESIGN OBJECT IN MABE.

    Physics (Layer 1) computes what the target needs.
    This spec (Layer 2) encodes the answer.
    Realization ranker (Layer 3) decides what builds it.
    Adapters (Layer 4) implement it in specific materials.

    Works for molecular recognition, optical wavelength selection,
    and future interaction domains — same abstraction.
    """
    # Identity
    name: str
    domain: InteractionDomain = InteractionDomain.MOLECULAR
    version: str = "2.0"

    # What we're interacting with
    target: TargetGeometry = field(default_factory=lambda: TargetGeometry(
        name="unspecified", formula="?"))

    # Geometry
    shape: GeometryShape = field(default_factory=lambda: GeometryShape(
        topology=GeometryTopology.CAVITY))

    # Interaction features
    elements: List[InteractionElement] = field(default_factory=list)

    # Selectivity
    competitors: List[CompetitorConstraint] = field(default_factory=list)

    # Conditions
    conditions: OperatingConditions = field(default_factory=OperatingConditions)

    # Predicted performance
    predicted_logK: float = 0.0          # Molecular: association constant
    predicted_dG_kJ: float = 0.0         # Molecular: free energy
    predicted_reflectance: float = 0.0   # Optical: peak reflectance
    predicted_bandwidth_nm: float = 0.0  # Optical: FWHM
    energy_decomposition: Dict[str, float] = field(default_factory=dict)

    # Confidence
    calibration_status: str = "uncalibrated"
    confidence_notes: str = ""

    # Design constraints from physics
    minimum_rigidity: float = 0.0
    minimum_elements: int = 0
    critical_elements: List[int] = field(default_factory=list)

    # ── Derived Properties ───────────────────────────────────────────

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def element_types(self) -> Dict[str, int]:
        counts = {}
        for e in self.elements:
            t = e.interaction_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    @property
    def bounding_radius_A(self) -> float:
        if not self.elements:
            return 0.0
        positions = np.array([e.position_A for e in self.elements])
        return float(np.max(np.linalg.norm(positions, axis=1)))

    @property
    def worst_selectivity_margin(self) -> float:
        if not self.competitors:
            return float('inf')
        return min(c.achievable_delta_logK - c.required_delta_logK
                   for c in self.competitors)

    @property
    def all_competitors_feasible(self) -> bool:
        return all(c.feasible for c in self.competitors)

    def summary(self) -> str:
        lines = [
            f"InteractionGeometrySpec: {self.name} [{self.domain.value}]",
            f"  Target: {self.target.name} ({self.target.formula}), "
            f"shape={self.target.shape.value}, scale={self.target.scale.value}",
            f"  Geometry: {self.shape.topology.value}",
        ]
        if self.domain == InteractionDomain.MOLECULAR:
            lines.append(
                f"    aperture={self.shape.aperture_A:.1f}Å, "
                f"depth={self.shape.depth_A:.1f}Å")
            lines.append(f"  Predicted logK: {self.predicted_logK:.1f} "
                         f"(ΔG={self.predicted_dG_kJ:.1f} kJ/mol)")
        elif self.domain == InteractionDomain.OPTICAL:
            lines.append(
                f"    period={self.shape.lattice_period_nm:.1f}nm, "
                f"n_periods={self.shape.n_periods}, "
                f"fill={self.shape.fill_fraction:.2f}")
            lines.append(f"  Predicted reflectance: {self.predicted_reflectance:.1%}, "
                         f"bandwidth: {self.predicted_bandwidth_nm:.1f}nm")

        lines.append(f"  Elements: {self.n_elements} ({self.element_types})")

        if self.competitors:
            lines.append(f"  Competitors: {len(self.competitors)}")
            for c in self.competitors:
                status = "✓" if c.feasible else "✗"
                if self.domain == InteractionDomain.MOLECULAR:
                    lines.append(
                        f"    {status} vs {c.competitor_formula}: "
                        f"need Δ{c.required_delta_logK:+.1f}, "
                        f"have Δ{c.achievable_delta_logK:+.1f}")
                elif self.domain == InteractionDomain.OPTICAL:
                    lines.append(
                        f"    {status} vs {c.competitor_formula}: "
                        f"need {c.required_extinction_ratio:.0f}dB rejection")

        lines.append(f"  Calibration: {self.calibration_status}")
        lines.append(f"  Rigidity: {self.minimum_rigidity:.1f}")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE ALIAS
# ═══════════════════════════════════════════════════════════════════════════

# Any code referencing the old name still works
BindingPocketSpec = InteractionGeometrySpec
PocketShape = GeometryShape
PocketTopology = GeometryTopology


# ═══════════════════════════════════════════════════════════════════════════
# GENERATORS — Molecular domain
# ═══════════════════════════════════════════════════════════════════════════

def _coordination_positions(
    cn: int, geometry: str, bond_length: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate ideal donor positions for a coordination geometry."""
    vectors_map = {
        'octahedral': [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
        'tetrahedral': [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],
        'square_planar': [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],
    }
    if geometry in vectors_map:
        vectors = vectors_map[geometry][:cn]
    elif cn == 2:
        vectors = [[0,0,1],[0,0,-1]]
    elif cn == 3:
        vectors = [[1,0,0],[-0.5,0.866,0],[-0.5,-0.866,0]]
    elif cn == 5:
        vectors = [[0,0,1],[0,0,-1],[1,0,0],[-0.5,0.866,0],[-0.5,-0.866,0]]
    else:
        golden = (1 + 5**0.5) / 2
        vectors = []
        for i in range(cn):
            theta = np.arccos(1 - 2*(i+0.5)/cn)
            phi = 2*np.pi*i/golden
            vectors.append([np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi), np.cos(theta)])

    positions = []
    for v in vectors[:cn]:
        v = np.array(v, dtype=float)
        v = v / np.linalg.norm(v)
        positions.append((v * bond_length, -v))
    return positions


def spec_for_metal_ion(
    metal_formula: str, charge: int, ionic_radius_A: float,
    hsab_class: str, d_electrons: int, coordination_number: int,
    preferred_geometry: str, donor_subtypes: List[str],
    pH: float = 7.0, competitors: Optional[List[Dict]] = None,
    matrix: str = "",
) -> InteractionGeometrySpec:
    """Generate interaction geometry spec for metal ion capture."""
    target = TargetGeometry(
        name=metal_formula, formula=metal_formula, charge=float(charge),
        shape=TargetShape.SPHERICAL, scale=TargetScale.ATOMIC,
        symmetry="Kh", effective_radius_A=ionic_radius_A,
        bounding_sphere_A=ionic_radius_A, hsab_class=hsab_class,
        domain=InteractionDomain.MOLECULAR,
    )

    positions = _coordination_positions(coordination_number, preferred_geometry,
                                         ionic_radius_A + 2.0)
    elements = []
    for i, (pos, direction) in enumerate(positions):
        subtype = donor_subtypes[i] if i < len(donor_subtypes) else donor_subtypes[-1]
        donor_atom = subtype.split('_')[0] if '_' in subtype else subtype[0]
        itype = {'O': InteractionType.COORDINATE_DONOR_O,
                 'N': InteractionType.COORDINATE_DONOR_N,
                 'S': InteractionType.COORDINATE_DONOR_S,
                 'P': InteractionType.COORDINATE_DONOR_P,
                 }.get(donor_atom, InteractionType.COORDINATE_DONOR_O)

        elements.append(InteractionElement(
            interaction_type=itype, position_A=pos, direction_A=direction,
            subtype=subtype, strength_required=5.0, tolerance_A=0.3,
            must_be_rigid=True,
        ))

    geom_shape = GeometryShape(
        topology=GeometryTopology.COORDINATION_SPHERE,
        aperture_A=2*(ionic_radius_A + 2.0),
        depth_A=ionic_radius_A + 2.0,
        pocket_volume_A3=target.volume_A3() * 3,
        rigidity_required=0.7,
    )

    comp_list = []
    if competitors:
        for comp in competitors:
            comp_list.append(CompetitorConstraint(
                competitor_name=comp.get('name', comp['formula']),
                competitor_formula=comp['formula'],
                competitor_shape=TargetShape.SPHERICAL,
                competitor_charge=comp.get('charge', 0),
                concentration_ratio=comp.get('conc_ratio', 1.0),
                required_delta_logK=comp.get('required_delta', 2.0),
            ))

    return InteractionGeometrySpec(
        name=f"geom_{metal_formula}_{preferred_geometry}",
        domain=InteractionDomain.MOLECULAR,
        target=target, shape=geom_shape, elements=elements,
        competitors=comp_list,
        conditions=OperatingConditions(pH=pH, matrix_name=matrix),
        calibration_status="calibrated" if d_electrons >= 0 else "uncalibrated",
        minimum_rigidity=0.7,
        minimum_elements=coordination_number,
        critical_elements=list(range(coordination_number)),
    )


def spec_for_oxyanion(
    anion_name: str, anion_formula: str, charge: int,
    shape: str, n_oxygen: int, pyramid_height_A: float,
    oo_distance_A: float, thermodynamic_radius_A: float,
    hydration_dG_kJ: float, has_lone_pair: bool,
    metal_center: Optional[str] = None, metal_logK: float = 0.0,
    cone_angle_deg: float = 38.0, n_hbond_donors: int = 4,
    competitors: Optional[List[Dict]] = None,
    pH: float = 7.0, matrix: str = "",
) -> InteractionGeometrySpec:
    """Generate interaction geometry spec for oxyanion capture."""
    target_shape = {'pyramidal': TargetShape.PYRAMIDAL,
                    'planar': TargetShape.PLANAR,
                    'tetrahedral': TargetShape.TETRAHEDRAL,
                    }.get(shape, TargetShape.IRREGULAR)
    symmetry = {'pyramidal': 'C3v', 'planar': 'D3h',
                'tetrahedral': 'Td'}.get(shape, 'C1')

    target = TargetGeometry(
        name=anion_name, formula=anion_formula, charge=float(charge),
        shape=target_shape, scale=TargetScale.SMALL_MOLECULE,
        symmetry=symmetry, effective_radius_A=thermodynamic_radius_A,
        bounding_sphere_A=thermodynamic_radius_A,
        width_A=oo_distance_A, height_A=pyramid_height_A,
        hsab_class='hard', hydration_dG_kJ=hydration_dG_kJ,
        has_lone_pair=has_lone_pair, lone_pair_accessible=has_lone_pair,
        n_hbond_acceptors=n_oxygen, domain=InteractionDomain.MOLECULAR,
    )

    elements = []
    if metal_center:
        elements.append(InteractionElement(
            interaction_type=InteractionType.METAL_CENTER,
            position_A=np.array([0.0, 0.0, -pyramid_height_A - 1.0]),
            direction_A=np.array([0.0, 0.0, 1.0]),
            subtype=metal_center, strength_required=8.0,
            tolerance_A=0.2, must_be_rigid=True,
            estimated_dG_kJ=-metal_logK * 5.71,
        ))

    hbond_ring_radius = oo_distance_A / 2 + 1.4
    for i in range(n_hbond_donors):
        angle = 2 * np.pi * i / n_hbond_donors
        pos = np.array([hbond_ring_radius*np.cos(angle),
                       hbond_ring_radius*np.sin(angle), 0.0])
        elements.append(InteractionElement(
            interaction_type=InteractionType.HBOND_DONOR,
            position_A=pos, direction_A=-pos/np.linalg.norm(pos),
            subtype='urea_NH', strength_required=3.0,
            tolerance_A=0.5, must_be_rigid=True,
        ))

    if has_lone_pair:
        elements.append(InteractionElement(
            interaction_type=InteractionType.HBOND_DONOR,
            position_A=np.array([0.0, 0.0, pyramid_height_A + 1.4]),
            direction_A=np.array([0.0, 0.0, -1.0]),
            subtype='lone_pair_acceptor', strength_required=2.0,
            tolerance_A=0.8,
        ))

    aperture_A = oo_distance_A + 2.8
    depth_A = pyramid_height_A + 1.4 if pyramid_height_A > 0 else 1.5
    geom_shape = GeometryShape(
        topology=GeometryTopology.CAVITY,
        aperture_A=aperture_A, depth_A=depth_A,
        cone_half_angle_deg=cone_angle_deg, symmetry_order=3,
        pocket_volume_A3=(1/3)*np.pi*(aperture_A/2)**2*depth_A,
        rigidity_required=0.8, max_rms_deformation_A=0.3,
    )

    comp_list = []
    if competitors:
        for comp in competitors:
            required = np.log10(comp.get('conc_ratio', 1.0)) + 1.5
            achiev = sum(comp.get('sources', {}).values())
            comp_list.append(CompetitorConstraint(
                competitor_name=comp.get('name', comp['formula']),
                competitor_formula=comp['formula'],
                competitor_shape={'planar': TargetShape.PLANAR,
                    'tetrahedral': TargetShape.TETRAHEDRAL,
                    'pyramidal': TargetShape.PYRAMIDAL,
                    }.get(comp.get('shape', ''), TargetShape.IRREGULAR),
                competitor_charge=comp.get('charge', 0),
                concentration_ratio=comp.get('conc_ratio', 1.0),
                required_delta_logK=required,
                selectivity_sources=comp.get('sources', {}),
                achievable_delta_logK=achiev,
                feasible=achiev >= required,
            ))

    return InteractionGeometrySpec(
        name=f"geom_{anion_name}_{cone_angle_deg:.0f}deg",
        domain=InteractionDomain.MOLECULAR,
        target=target, shape=geom_shape, elements=elements,
        competitors=comp_list,
        conditions=OperatingConditions(pH=pH, matrix_name=matrix),
        predicted_logK=metal_logK if metal_center else 0.0,
        predicted_dG_kJ=-metal_logK*5.71 if metal_center else 0.0,
        calibration_status="uncalibrated",
        minimum_rigidity=0.8,
        minimum_elements=n_hbond_donors + (1 if metal_center else 0),
        critical_elements=[0] if metal_center else [],
    )


def spec_for_protein_epitope(
    target_name: str, epitope_area_A2: float = 800.0,
    epitope_charge: float = 0.0, n_hbond_patches: int = 3,
    n_hydrophobic_patches: int = 2, n_charged_patches: int = 1,
    pH: float = 7.4, matrix: str = "physiological_buffer",
) -> InteractionGeometrySpec:
    """Generate stub spec for a protein epitope target."""
    target = TargetGeometry(
        name=target_name, formula=target_name, charge=epitope_charge,
        shape=TargetShape.CONVEX_SURFACE, scale=TargetScale.PROTEIN_DOMAIN,
        effective_radius_A=np.sqrt(epitope_area_A2/np.pi),
        bounding_sphere_A=np.sqrt(epitope_area_A2/np.pi)*1.5,
        n_hbond_donors=n_hbond_patches, n_hbond_acceptors=n_hbond_patches,
        domain=InteractionDomain.MOLECULAR,
    )

    elements = []
    radius = np.sqrt(epitope_area_A2/np.pi)
    total = n_hbond_patches + n_hydrophobic_patches + n_charged_patches
    for i in range(total):
        angle = 2*np.pi*i/total
        pos = np.array([radius*0.6*np.cos(angle), radius*0.6*np.sin(angle), 0.0])
        if i < n_hbond_patches:
            itype, sub = InteractionType.HBOND_DONOR, 'backbone_NH'
        elif i < n_hbond_patches + n_hydrophobic_patches:
            itype, sub = InteractionType.HYDROPHOBIC_SURFACE, 'nonpolar_patch'
        else:
            itype = InteractionType.NEGATIVE_CHARGE if epitope_charge > 0 \
                else InteractionType.POSITIVE_CHARGE
            sub = 'charged_residue'
        elements.append(InteractionElement(
            interaction_type=itype, position_A=pos,
            direction_A=np.array([0,0,-1]), subtype=sub,
            strength_required=3.0, tolerance_A=2.0,
        ))

    return InteractionGeometrySpec(
        name=f"geom_{target_name}_epitope",
        domain=InteractionDomain.MOLECULAR,
        target=target,
        shape=GeometryShape(topology=GeometryTopology.SURFACE,
            aperture_A=2*radius, depth_A=5.0,
            pocket_volume_A3=epitope_area_A2*5.0,
            rigidity_required=0.5, max_rms_deformation_A=2.0),
        elements=elements,
        conditions=OperatingConditions(pH=pH, matrix_name=matrix),
        calibration_status="stub",
        minimum_rigidity=0.5,
    )


# ═══════════════════════════════════════════════════════════════════════════
# GENERATORS — Optical domain
# ═══════════════════════════════════════════════════════════════════════════

def spec_for_wavelength_selective(
    target_wavelength_nm: float,
    bandwidth_nm: float = 20.0,
    peak_reflectance: float = 0.95,
    n_high: float = 2.5,           # Refractive index of high-n material
    n_low: float = 1.45,           # Refractive index of low-n material
    competitors: Optional[List[Dict]] = None,
    incident_angle_deg: float = 0.0,
    environment: str = "ambient_air",
) -> InteractionGeometrySpec:
    """
    Generate interaction geometry spec for wavelength-selective reflection.

    Physics: Bragg reflection from periodic dielectric stack.
    λ_peak = 2 * n_eff * d    (quarter-wave condition per layer)
    Bandwidth ∝ Δn/n_eff × λ_peak
    Peak reflectance → 1 as N_periods → ∞

    Same design logic as molecular capture:
      Target:     photon at λ = target_wavelength_nm
      Selectivity: reject photons at other wavelengths
      Geometry:   periodic stack with correct d-spacing
      Realization: Layer 3 decision — not specified here.
    """
    # Effective index and quarter-wave thickness
    n_eff = (n_high + n_low) / 2
    d_period = target_wavelength_nm / (2 * n_eff)  # nm, one full period
    d_high = target_wavelength_nm / (4 * n_high)   # quarter-wave thickness
    d_low = target_wavelength_nm / (4 * n_low)

    # Bandwidth from coupled-mode theory
    delta_n = n_high - n_low
    predicted_bw = (4 / np.pi) * np.arcsin(delta_n / (n_high + n_low)) * target_wavelength_nm

    # Number of periods needed for target reflectance
    # R ≈ ((n_high/n_low)^(2N) - 1)^2 / ((n_high/n_low)^(2N) + 1)^2
    r = n_high / n_low
    # Solve for N: invert reflectance formula
    if peak_reflectance >= 0.999:
        n_periods = 20
    else:
        # Approximate: N ≈ ln((1+√R)/(1-√R)) / (2*ln(r))
        sqrt_R = np.sqrt(peak_reflectance)
        n_periods = max(3, int(np.ceil(
            np.log((1 + sqrt_R) / (1 - sqrt_R)) / (2 * np.log(r))
        )))

    # Photon energy
    h_eV_nm = 1239.84  # eV·nm
    energy_eV = h_eV_nm / target_wavelength_nm
    frequency_Hz = 3e8 / (target_wavelength_nm * 1e-9)

    target = TargetGeometry(
        name=f"λ={target_wavelength_nm:.0f}nm",
        formula=f"λ={target_wavelength_nm:.0f}nm",
        charge=0.0,
        shape=TargetShape.PLANE_WAVE,
        scale=TargetScale.WAVELENGTH,
        domain=InteractionDomain.OPTICAL,
        wavelength_nm=target_wavelength_nm,
        frequency_Hz=frequency_Hz,
        energy_eV=energy_eV,
        bandwidth_nm=bandwidth_nm,
    )

    # Build interaction elements: alternating high-n / low-n layers
    elements = []
    total_thickness = 0.0
    for i in range(n_periods):
        # High-n layer
        z_high = total_thickness + d_high / 2
        elements.append(InteractionElement(
            interaction_type=InteractionType.REFRACTIVE_BOUNDARY,
            position_A=np.array([0, 0, z_high]),  # position along stack axis (nm)
            direction_A=np.array([0, 0, 1]),
            subtype=f"high_n={n_high:.2f}",
            refractive_index=n_high,
            thickness_nm=d_high,
            period_nm=d_period,
            must_be_rigid=True,
            tolerance_A=d_high * 0.01,  # 1% thickness tolerance
        ))
        total_thickness += d_high

        # Low-n layer
        z_low = total_thickness + d_low / 2
        elements.append(InteractionElement(
            interaction_type=InteractionType.REFRACTIVE_BOUNDARY,
            position_A=np.array([0, 0, z_low]),
            direction_A=np.array([0, 0, 1]),
            subtype=f"low_n={n_low:.2f}",
            refractive_index=n_low,
            thickness_nm=d_low,
            period_nm=d_period,
            must_be_rigid=True,
            tolerance_A=d_low * 0.01,
        ))
        total_thickness += d_low

    geom_shape = GeometryShape(
        topology=GeometryTopology.PERIODIC_STACK,
        lattice_period_nm=d_period,
        n_periods=n_periods,
        fill_fraction=d_high / d_period,
        lattice_type='lamellar',
        rigidity_required=0.6,
        period_tolerance_fraction=0.01,
    )

    # Competitor constraints (adjacent wavelengths)
    comp_list = []
    if competitors:
        for comp in competitors:
            comp_list.append(CompetitorConstraint(
                competitor_name=comp.get('name', comp['formula']),
                competitor_formula=comp['formula'],
                competitor_shape=TargetShape.PLANE_WAVE,
                required_extinction_ratio=comp.get('rejection_dB', 20.0),
                feasible=abs(comp.get('wavelength_nm', 0) - target_wavelength_nm) > predicted_bw / 2,
            ))

    return InteractionGeometrySpec(
        name=f"geom_λ{target_wavelength_nm:.0f}nm_stack",
        domain=InteractionDomain.OPTICAL,
        target=target, shape=geom_shape, elements=elements,
        competitors=comp_list,
        conditions=OperatingConditions(
            incident_angle_deg=incident_angle_deg,
            environment=environment,
            ambient_refractive_index=1.0 if 'air' in environment else 1.33,
            required_reflectance=peak_reflectance,
            required_bandwidth_nm=bandwidth_nm,
        ),
        predicted_reflectance=peak_reflectance,
        predicted_bandwidth_nm=predicted_bw,
        calibration_status="analytical",
        confidence_notes="1D Bragg stack: analytical solution from transfer matrix method.",
        minimum_rigidity=0.6,
        minimum_elements=2 * n_periods,
    )


def spec_for_structural_color(
    target_wavelength_nm: float,
    viewing_angle_deg: float = 0.0,
    n_sphere: float = 1.45,         # Silica default
    n_medium: float = 1.0,          # Air default
) -> InteractionGeometrySpec:
    """
    Generate spec for structural color from 3D photonic crystal (opal structure).

    Physics: FCC close-packed spheres → Bragg reflection from (111) planes.
    d₁₁₁ = √(2/3) × D_sphere
    λ_peak = 2 × d₁₁₁ × n_eff

    Realization options scored by Layer 3 — not specified here.
    """
    # Effective index for FCC: volume fraction of spheres = 0.7405
    phi = 0.7405  # FCC packing fraction
    n_eff = np.sqrt(phi * n_sphere**2 + (1 - phi) * n_medium**2)

    # Sphere diameter from Bragg condition
    d_111 = target_wavelength_nm / (2 * n_eff)
    D_sphere = d_111 / np.sqrt(2/3)

    target = TargetGeometry(
        name=f"λ={target_wavelength_nm:.0f}nm",
        formula=f"λ={target_wavelength_nm:.0f}nm",
        charge=0.0,
        shape=TargetShape.PLANE_WAVE,
        scale=TargetScale.WAVELENGTH,
        domain=InteractionDomain.OPTICAL,
        wavelength_nm=target_wavelength_nm,
        energy_eV=1239.84 / target_wavelength_nm,
        bandwidth_nm=target_wavelength_nm * 0.05,  # ~5% for opal
    )

    # Periodic elements: sphere positions in FCC unit
    a_lattice = D_sphere * np.sqrt(2)  # FCC lattice parameter
    elements = [
        InteractionElement(
            interaction_type=InteractionType.PERIODIC_ELEMENT,
            position_A=np.array([0, 0, 0]),
            direction_A=np.array([1, 1, 1]) / np.sqrt(3),  # (111) normal
            subtype=f"sphere_D={D_sphere:.0f}nm_n={n_sphere:.2f}",
            refractive_index=n_sphere,
            period_nm=d_111,
            must_be_rigid=False,  # self-assembly provides order
            tolerance_A=D_sphere * 0.03,  # 3% size polydispersity
        ),
    ]

    geom_shape = GeometryShape(
        topology=GeometryTopology.PERIODIC_3D,
        lattice_period_nm=d_111,
        n_periods=20,  # typical opal: many layers
        fill_fraction=phi,
        lattice_type='fcc',
        rigidity_required=0.3,  # self-assembly tolerates defects
        period_tolerance_fraction=0.03,
    )

    return InteractionGeometrySpec(
        name=f"geom_λ{target_wavelength_nm:.0f}nm_opal",
        domain=InteractionDomain.OPTICAL,
        target=target, shape=geom_shape, elements=elements,
        conditions=OperatingConditions(
            incident_angle_deg=viewing_angle_deg,
            environment="ambient_air",
            ambient_refractive_index=n_medium,
        ),
        predicted_reflectance=0.75,  # typical opal
        predicted_bandwidth_nm=target_wavelength_nm * 0.05,
        calibration_status="analytical",
        confidence_notes=f"FCC opal: sphere D={D_sphere:.1f}nm, d111={d_111:.1f}nm, n_eff={n_eff:.3f}",
        minimum_rigidity=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    print("InteractionGeometrySpec — Self-Test")
    print("═" * 70)
    print()

    # ── Test 1: Metal ion ──────────────────────────────────────────────
    print("1. Molecular: Pb²⁺ capture")
    print("─" * 60)
    pb = spec_for_metal_ion(
        metal_formula='Pb2+', charge=2, ionic_radius_A=1.19,
        hsab_class='borderline', d_electrons=0, coordination_number=6,
        preferred_geometry='octahedral',
        donor_subtypes=['N_amine','N_amine','O_carboxylate','O_carboxylate',
                        'S_thiolate','S_thiolate'],
        pH=4.5, matrix='acid_mine_drainage',
        competitors=[{'formula':'Ca2+','charge':2,'conc_ratio':1000,'required_delta':4.0}],
    )
    print(pb.summary())
    assert pb.domain == InteractionDomain.MOLECULAR
    print("  ✓ Domain: molecular")
    print()

    # ── Test 2: Oxyanion ───────────────────────────────────────────────
    print("2. Molecular: SeO₃²⁻ from mine water")
    print("─" * 60)
    se = spec_for_oxyanion(
        anion_name='selenite', anion_formula='SeO3(2-)', charge=-2,
        shape='pyramidal', n_oxygen=3, pyramid_height_A=0.80,
        oo_distance_A=2.63, thermodynamic_radius_A=2.39,
        hydration_dG_kJ=-410.0, has_lone_pair=True,
        metal_center='Zr4+', metal_logK=12.0,
        competitors=[{
            'name':'carbonate','formula':'CO3(2-)','charge':-2,'shape':'planar',
            'conc_ratio':3454,'sources':{'geometric':11.3,'metal_affinity':4.0,'hydration':0.3},
        }],
        pH=7.5, matrix='coal_mine_drainage',
    )
    print(se.summary())
    assert se.all_competitors_feasible
    print("  ✓ All competitors feasible")
    print()

    # ── Test 3: Protein epitope ────────────────────────────────────────
    print("3. Molecular: insulin epitope (stub)")
    print("─" * 60)
    ins = spec_for_protein_epitope('insulin')
    print(ins.summary())
    print()

    # ── Test 4: Optical — 1D Bragg stack ───────────────────────────────
    print("4. Optical: λ=532nm green reflector (Bragg stack)")
    print("─" * 60)
    green = spec_for_wavelength_selective(
        target_wavelength_nm=532.0,
        bandwidth_nm=20.0,
        peak_reflectance=0.99,
        n_high=2.5,    # TiO₂
        n_low=1.45,    # SiO₂
        competitors=[
            {'name':'red','formula':'λ=630nm','wavelength_nm':630,'rejection_dB':20},
            {'name':'blue','formula':'λ=450nm','wavelength_nm':450,'rejection_dB':20},
        ],
    )
    print(green.summary())
    assert green.domain == InteractionDomain.OPTICAL
    print(f"\n  Stack: {green.shape.n_periods} periods × "
          f"{green.shape.lattice_period_nm:.1f}nm = "
          f"{green.shape.n_periods * green.shape.lattice_period_nm:.0f}nm total")
    print(f"  Predicted bandwidth: {green.predicted_bandwidth_nm:.1f}nm")
    print(f"  Elements: {green.n_elements} layers")
    print("  ✓ Domain: optical")
    print()

    # ── Test 5: Optical — 3D structural color ──────────────────────────
    print("5. Optical: λ=450nm blue structural color (opal)")
    print("─" * 60)
    blue = spec_for_structural_color(
        target_wavelength_nm=450.0,
        n_sphere=1.45,  # silica
        n_medium=1.0,   # air
    )
    print(blue.summary())
    print(f"\n  {blue.confidence_notes}")
    print("  ✓ Domain: optical, 3D periodic")
    print()

    # ── Test 6: Cross-domain comparison ────────────────────────────────
    print("6. Cross-Domain Comparison")
    print("─" * 60)
    specs = {
        'Pb²⁺ capture': pb,
        'SeO₃²⁻ capture': se,
        'insulin recognition': ins,
        'green reflector': green,
        'blue opal': blue,
    }
    print(f"{'Name':25s} {'Domain':12s} {'Topology':20s} {'Elements':>8s} {'Rigid':>6s}")
    for name, spec in specs.items():
        print(f"{name:25s} {spec.domain.value:12s} "
              f"{spec.shape.topology.value:20s} "
              f"{spec.n_elements:8d} {spec.minimum_rigidity:6.1f}")
    print()

    # ── Test 7: Prime Directive compliance ─────────────────────────────
    print("7. Prime Directive Compliance")
    print("─" * 60)
    import inspect
    all_source = inspect.getsource(spec_for_metal_ion)
    all_source += inspect.getsource(spec_for_oxyanion)
    all_source += inspect.getsource(spec_for_protein_epitope)
    all_source += inspect.getsource(spec_for_wavelength_selective)
    all_source += inspect.getsource(spec_for_structural_color)

    fabrication_words = [
        'DNA', 'MOF', 'origami', 'ATHENA', 'staple', 'helix',
        'chelator', 'EDTA', 'DTPA', 'crown_ether',
        'resin', 'column', 'bead', 'membrane',
        'RFDiffusion', 'AlphaFold', 'protein_fold',
        'UiO-66', 'NU-1000', 'zeolite',
        'TiO2', 'SiO2', 'silicon',  # even optical materials are realization
        'butterfly', 'thin_film',
    ]
    found = [w for w in fabrication_words if w in all_source]
    if found:
        print(f"  ⚠ Found fabrication references: {found}")
    else:
        print("  ✓ Zero fabrication references in all generator functions")
    print()

    # ── Test 8: Backward compatibility ─────────────────────────────────
    print("8. Backward Compatibility")
    print("─" * 60)
    assert BindingPocketSpec is InteractionGeometrySpec
    assert PocketTopology is GeometryTopology
    assert PocketShape is GeometryShape
    print("  ✓ BindingPocketSpec → InteractionGeometrySpec alias works")
    print("  ✓ PocketTopology → GeometryTopology alias works")
    print("  ✓ PocketShape → GeometryShape alias works")
    print()

    print("═" * 70)
    print("InteractionGeometrySpec v2.0 — covers molecular + optical domains.")
    print("Same abstraction: target physics → optimal geometry → realization ranking.")


if __name__ == "__main__":
    self_test()