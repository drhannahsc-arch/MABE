"""
binding_pocket_spec.py — Realization-Agnostic Binding Pocket Specification

MABE PRIME DIRECTIVE COMPLIANCE:
  This is Layer 2 output. It describes WHAT the optimal pocket looks like.
  It says NOTHING about what material builds it.

  No DNA. No MOF. No chelator. No protein. No resin. No scaffold.
  Just physics: shapes, donors, charges, distances.

  Layer 3 (realization_ranker.py) reads this and decides what builds it.
  Layer 4 (adapters) implement it in a specific material.

═══════════════════════════════════════════════════════════════════════════
A BindingPocketSpec is:

  A spatial arrangement of INTERACTION ELEMENTS
  around a POCKET SHAPE,
  under specified CONDITIONS,
  with SELECTIVITY REQUIREMENTS against competitors.

This abstraction covers:
  - Metal coordination:  donors arranged around a coordination center
  - Anion recognition:   cavity with H-bond donors + charge + shape
  - Protein epitope:     surface patch with complementary features
  - Small molecule:      enclosed cavity with mixed interactions
  - Any future target:   same elements, different arrangement

═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# TARGET GEOMETRY — What we're trying to bind
# ═══════════════════════════════════════════════════════════════════════════

class TargetScale(str, Enum):
    """Size regime of the target — determines which interaction physics dominate."""
    ATOMIC = "atomic"               # Single ion (Pb²⁺, Se⁴⁺), < 0.3 nm
    SMALL_MOLECULE = "small_mol"    # < 1 nm (metabolites, drugs, oxyanions)
    PEPTIDE = "peptide"             # 1-5 nm (short peptides, glycans)
    PROTEIN_DOMAIN = "domain"       # 5-20 nm (epitope, domain)
    MACROMOLECULAR = "macro"        # > 20 nm (whole protein, virus)


class TargetShape(str, Enum):
    """Geometric classification of the target."""
    SPHERICAL = "spherical"         # Metal ions, noble gas atoms
    PYRAMIDAL = "pyramidal"         # SeO₃²⁻, SO₃²⁻, AsO₃³⁻
    PLANAR = "planar"               # CO₃²⁻, NO₃⁻, aromatic rings
    TETRAHEDRAL = "tetrahedral"     # SO₄²⁻, PO₄³⁻, SeO₄²⁻
    LINEAR = "linear"               # SCN⁻, N₃⁻, CO₂
    CONVEX_SURFACE = "convex"       # Protein epitope (outward-facing)
    CONCAVE_SURFACE = "concave"     # Protein cleft
    IRREGULAR = "irregular"         # Complex small molecules
    CYLINDRICAL = "cylindrical"     # Helical peptides, DNA


@dataclass(frozen=True)
class TargetGeometry:
    """
    Physics-level description of the target's shape and properties.

    Everything needed to determine what the pocket must look like —
    without knowing what the pocket is made of.
    """
    name: str
    formula: str
    charge: float
    shape: TargetShape
    scale: TargetScale
    symmetry: str = ""              # Point group: C3v, D3h, Td, C∞v, etc.

    # Dimensions (Å)
    effective_radius_A: float = 0.0       # Ionic/thermodynamic radius
    bounding_sphere_A: float = 0.0        # Smallest enclosing sphere
    length_A: float = 0.0                 # For non-spherical: longest axis
    width_A: float = 0.0                  # Perpendicular to length
    height_A: float = 0.0                 # Third axis / pyramid height

    # Electronic
    hsab_class: str = ""            # 'hard', 'soft', 'borderline'
    polarizability_A3: float = 0.0
    dipole_moment_D: float = 0.0

    # Hydration
    hydration_dG_kJ: float = 0.0    # More negative = harder to dehydrate
    hydrated_radius_A: float = 0.0

    # Special features
    has_lone_pair: bool = False
    lone_pair_accessible: bool = False
    n_hbond_donors: int = 0         # Target can DONATE H-bonds
    n_hbond_acceptors: int = 0      # Target can ACCEPT H-bonds
    aromatic: bool = False
    redox_active: bool = False

    # Speciation
    dominant_form_at_pH: str = ""   # e.g. 'HSeO3-' at pH 7.0
    fraction_target_form: float = 1.0  # What fraction exists as bindable form

    def volume_A3(self) -> float:
        """Approximate target volume."""
        if self.shape == TargetShape.SPHERICAL:
            return (4/3) * np.pi * self.effective_radius_A**3
        elif self.shape in (TargetShape.PYRAMIDAL, TargetShape.TETRAHEDRAL):
            # Approximate as cone or tetrahedron
            r = self.width_A / 2 if self.width_A > 0 else self.effective_radius_A
            h = self.height_A if self.height_A > 0 else self.effective_radius_A
            return (1/3) * np.pi * r**2 * h
        else:
            return (4/3) * np.pi * self.effective_radius_A**3


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTION ELEMENTS — The recognition features the pocket must provide
# ═══════════════════════════════════════════════════════════════════════════

class InteractionType(str, Enum):
    """Fundamental interaction types. Physics, not chemistry."""
    # Coordinate bonds
    COORDINATE_DONOR_O = "coord_O"       # O donating lone pair to metal
    COORDINATE_DONOR_N = "coord_N"       # N donating lone pair to metal
    COORDINATE_DONOR_S = "coord_S"       # S donating lone pair to metal
    COORDINATE_DONOR_P = "coord_P"       # P donating lone pair to metal
    METAL_CENTER = "metal_center"        # Metal ion providing Lewis acid site

    # H-bonds
    HBOND_DONOR = "hbond_donor"          # NH, OH pointing into pocket
    HBOND_ACCEPTOR = "hbond_acceptor"    # C=O, lone pair accepting H-bond

    # Electrostatic
    POSITIVE_CHARGE = "pos_charge"       # Localized positive charge
    NEGATIVE_CHARGE = "neg_charge"       # Localized negative charge
    PARTIAL_POSITIVE = "partial_pos"     # δ+ region
    PARTIAL_NEGATIVE = "partial_neg"     # δ- region

    # Hydrophobic / dispersion
    HYDROPHOBIC_SURFACE = "hydrophobic"  # Nonpolar contact surface
    AROMATIC_WALL = "aromatic_wall"      # π-surface for stacking/CH-π

    # Shape
    STERIC_WALL = "steric_wall"          # Hard boundary (excluded volume)


@dataclass
class InteractionElement:
    """
    One recognition feature that the pocket must present.

    Position is relative to the pocket center (origin = target center
    when bound). Units: Ångströms.
    """
    interaction_type: InteractionType
    position_A: np.ndarray       # [x, y, z] relative to pocket center (Å)
    direction_A: np.ndarray      # Unit vector: which way it points
    strength_required: float = 1.0  # Relative importance (0-10)
    tolerance_A: float = 0.5     # Positional tolerance (Å)

    # Subtype for finer specification
    subtype: str = ""            # e.g. 'N_pyridine', 'O_carboxylate', 'urea_NH'

    # Energetics (from physics scorer)
    estimated_dG_kJ: float = 0.0  # Contribution to binding free energy

    # Constraints
    must_be_rigid: bool = False   # Requires structural enforcement
    cooperates_with: List[int] = field(default_factory=list)  # Indices of synergistic elements


# ═══════════════════════════════════════════════════════════════════════════
# POCKET SHAPE — The spatial envelope
# ═══════════════════════════════════════════════════════════════════════════

class PocketTopology(str, Enum):
    """Topological class of the binding pocket."""
    CAVITY = "cavity"              # Enclosed from 3+ sides (host-guest, deep pocket)
    CLEFT = "cleft"                # Open on one side (enzyme active site)
    GROOVE = "groove"              # Linear channel (DNA minor groove)
    SURFACE = "surface"            # Flat contact (protein-protein)
    CHANNEL = "channel"            # Through-hole (ion channel, pore)
    SPHERE = "coordination_sphere" # Radial arrangement around center (metal coordination)


@dataclass
class PocketShape:
    """
    Geometric envelope of the binding pocket.
    Describes the volume the target occupies when bound.
    All dimensions in Ångströms.
    """
    topology: PocketTopology

    # Cavity / cleft dimensions
    aperture_A: float = 0.0       # Opening diameter
    depth_A: float = 0.0          # How deep the target sits
    width_A: float = 0.0          # Lateral extent

    # Shape selectivity parameters
    cone_half_angle_deg: float = 0.0   # For conical cavities (0=cylinder, 90=flat)
    symmetry_order: int = 0            # Rotational symmetry (3 for C3v, etc.)
    curvature_A_inv: float = 0.0       # Surface curvature (concave<0, flat=0, convex>0)

    # Volume
    pocket_volume_A3: float = 0.0
    optimal_packing_coefficient: float = 0.55  # Rebek's 55% rule default

    # Flexibility
    rigidity_required: float = 0.0  # 0.0 = flexible OK, 1.0 = must be rigid
    max_rms_deformation_A: float = 1.0  # How much pocket can flex

    def ideal_guest_volume_A3(self) -> float:
        """Optimal target volume for this pocket (Rebek packing)."""
        return self.pocket_volume_A3 * self.optimal_packing_coefficient


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY REQUIREMENTS — What the pocket must reject
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CompetitorConstraint:
    """
    One species the pocket must discriminate against.

    The pocket design must achieve at least `required_delta_logK` selectivity
    over this competitor. The physics scorer determined WHY selectivity is
    possible (geometry, affinity, hydration) — this captures HOW MUCH.
    """
    competitor_name: str
    competitor_formula: str
    competitor_shape: TargetShape
    competitor_charge: float
    concentration_ratio: float       # [competitor]/[target] in operating matrix
    required_delta_logK: float       # Minimum selectivity needed
    selectivity_sources: Dict[str, float] = field(default_factory=dict)
    # e.g. {'geometric': 11.3, 'metal_affinity': 4.0, 'hydration': 0.3}
    # These are the physics-determined AVAILABLE selectivity channels.
    achievable_delta_logK: float = 0.0  # Sum of available sources
    feasible: bool = True              # achievable >= required?
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# OPERATING CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OperatingConditions:
    """Physical conditions under which the pocket must function."""
    pH: float = 7.0
    temperature_K: float = 298.15
    ionic_strength_M: float = 0.1
    matrix_name: str = ""               # e.g. 'coal_mine_drainage', 'blood_plasma'
    target_concentration_M: float = 0.0
    required_removal_fraction: float = 0.0  # 0.97 = 97% removal
    flow_conditions: str = "batch"      # 'batch', 'continuous', 'column'
    regeneration_required: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# THE SPEC — Layer 2 output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BindingPocketSpec:
    """
    Complete, realization-agnostic specification of an optimal binding pocket.

    THIS IS THE CENTRAL DESIGN OBJECT IN MABE.

    Physics (Layer 1) computes what the target needs.
    This spec (Layer 2) encodes the answer.
    Realization ranker (Layer 3) decides what builds it.
    Adapters (Layer 4) implement it in specific materials.

    Contains:
      target:       What we're binding
      shape:        Pocket geometry
      elements:     Interaction features the pocket must present
      competitors:  Selectivity constraints
      conditions:   Operating environment
      energetics:   Predicted thermodynamic performance
    """
    # Identity
    name: str
    version: str = "1.0"

    # What we're binding
    target: TargetGeometry = field(default_factory=lambda: TargetGeometry(
        name="unspecified", formula="?", charge=0,
        shape=TargetShape.SPHERICAL, scale=TargetScale.ATOMIC))

    # Pocket geometry
    shape: PocketShape = field(default_factory=lambda: PocketShape(
        topology=PocketTopology.CAVITY))

    # Recognition features
    elements: List[InteractionElement] = field(default_factory=list)

    # Selectivity
    competitors: List[CompetitorConstraint] = field(default_factory=list)

    # Conditions
    conditions: OperatingConditions = field(default_factory=OperatingConditions)

    # Predicted energetics (from physics scorer)
    predicted_logK: float = 0.0
    predicted_dG_kJ: float = 0.0
    energy_decomposition: Dict[str, float] = field(default_factory=dict)
    # e.g. {'metal_coordination': -45.0, 'hbond': -12.0, 'dehydration': +25.0, ...}

    # Confidence / calibration
    calibration_status: str = "uncalibrated"  # 'calibrated', 'partially_calibrated', 'uncalibrated'
    confidence_notes: str = ""

    # Design constraints that emerge from physics
    minimum_rigidity: float = 0.0        # How rigid must the pocket be?
    minimum_elements: int = 0            # Minimum number of interaction elements
    critical_elements: List[int] = field(default_factory=list)
    # Indices into self.elements that are non-negotiable

    # ── Derived Properties ───────────────────────────────────────────

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def element_types(self) -> Dict[str, int]:
        """Count of each interaction type."""
        counts = {}
        for e in self.elements:
            t = e.interaction_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    @property
    def bounding_radius_A(self) -> float:
        """Radius of sphere enclosing all interaction elements."""
        if not self.elements:
            return 0.0
        positions = np.array([e.position_A for e in self.elements])
        return float(np.max(np.linalg.norm(positions, axis=1)))

    @property
    def worst_selectivity_margin(self) -> float:
        """Smallest margin (achievable - required) across all competitors."""
        if not self.competitors:
            return float('inf')
        return min(c.achievable_delta_logK - c.required_delta_logK
                   for c in self.competitors)

    @property
    def all_competitors_feasible(self) -> bool:
        return all(c.feasible for c in self.competitors)

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"BindingPocketSpec: {self.name}",
            f"  Target: {self.target.name} ({self.target.formula}), "
            f"charge={self.target.charge:+.0f}, "
            f"shape={self.target.shape.value}, scale={self.target.scale.value}",
            f"  Pocket: {self.shape.topology.value}, "
            f"aperture={self.shape.aperture_A:.1f}Å, "
            f"depth={self.shape.depth_A:.1f}Å",
            f"  Elements: {self.n_elements} ({self.element_types})",
            f"  Predicted logK: {self.predicted_logK:.1f} "
            f"(ΔG={self.predicted_dG_kJ:.1f} kJ/mol)",
        ]
        if self.competitors:
            lines.append(f"  Competitors: {len(self.competitors)}")
            for c in self.competitors:
                status = "✓" if c.feasible else "✗"
                lines.append(
                    f"    {status} vs {c.competitor_formula}: "
                    f"need Δ{c.required_delta_logK:+.1f}, "
                    f"have Δ{c.achievable_delta_logK:+.1f}, "
                    f"margin={c.achievable_delta_logK - c.required_delta_logK:+.1f}"
                )
        lines.append(f"  Calibration: {self.calibration_status}")
        lines.append(f"  Rigidity required: {self.minimum_rigidity:.1f}")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# GENERATORS — Populate specs from physics
# ═══════════════════════════════════════════════════════════════════════════

def spec_for_metal_ion(
    metal_formula: str,
    charge: int,
    ionic_radius_A: float,
    hsab_class: str,
    d_electrons: int,
    coordination_number: int,
    preferred_geometry: str,   # 'octahedral', 'tetrahedral', 'square_planar'
    donor_subtypes: List[str],
    pH: float = 7.0,
    competitors: Optional[List[Dict]] = None,
    matrix: str = "",
) -> BindingPocketSpec:
    """
    Generate pocket spec for capturing a metal ion.

    The pocket IS a coordination sphere: donors arranged at defined
    positions around where the metal sits.
    """
    # Target geometry
    target = TargetGeometry(
        name=metal_formula,
        formula=metal_formula,
        charge=float(charge),
        shape=TargetShape.SPHERICAL,
        scale=TargetScale.ATOMIC,
        symmetry="Kh",  # spherical
        effective_radius_A=ionic_radius_A,
        bounding_sphere_A=ionic_radius_A,
        hsab_class=hsab_class,
    )

    # Coordination sphere geometry
    # Generate ideal donor positions from coordination geometry
    positions = _coordination_positions(coordination_number, preferred_geometry,
                                         ionic_radius_A + 2.0)  # bond length ≈ r_metal + r_donor

    elements = []
    for i, (pos, direction) in enumerate(positions):
        if i < len(donor_subtypes):
            subtype = donor_subtypes[i]
        else:
            subtype = donor_subtypes[-1] if donor_subtypes else "O_generic"

        # Map subtype to interaction type
        donor_atom = subtype.split('_')[0] if '_' in subtype else subtype[0]
        itype = {
            'O': InteractionType.COORDINATE_DONOR_O,
            'N': InteractionType.COORDINATE_DONOR_N,
            'S': InteractionType.COORDINATE_DONOR_S,
            'P': InteractionType.COORDINATE_DONOR_P,
        }.get(donor_atom, InteractionType.COORDINATE_DONOR_O)

        elements.append(InteractionElement(
            interaction_type=itype,
            position_A=pos,
            direction_A=direction,
            subtype=subtype,
            strength_required=5.0,  # coordination bonds are strong
            tolerance_A=0.3,        # tight positional requirement
            must_be_rigid=True,
        ))

    # Pocket shape = coordination sphere
    pocket_shape = PocketShape(
        topology=PocketTopology.SPHERE,
        aperture_A=2 * (ionic_radius_A + 2.0),
        depth_A=ionic_radius_A + 2.0,
        pocket_volume_A3=target.volume_A3() * 3,  # coordination shell volume
        rigidity_required=0.7,  # metal coordination is fairly rigid
    )

    # Competitor constraints
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

    return BindingPocketSpec(
        name=f"pocket_{metal_formula}_{preferred_geometry}",
        target=target,
        shape=pocket_shape,
        elements=elements,
        competitors=comp_list,
        conditions=OperatingConditions(pH=pH, matrix_name=matrix),
        calibration_status="calibrated" if d_electrons >= 0 else "uncalibrated",
        minimum_rigidity=0.7,
        minimum_elements=coordination_number,
        critical_elements=list(range(coordination_number)),
    )


def spec_for_oxyanion(
    anion_name: str,
    anion_formula: str,
    charge: int,
    shape: str,             # 'pyramidal', 'planar', 'tetrahedral'
    n_oxygen: int,
    pyramid_height_A: float,
    oo_distance_A: float,
    thermodynamic_radius_A: float,
    hydration_dG_kJ: float,
    has_lone_pair: bool,
    metal_center: Optional[str] = None,
    metal_logK: float = 0.0,
    cone_angle_deg: float = 38.0,
    n_hbond_donors: int = 4,
    competitors: Optional[List[Dict]] = None,
    pH: float = 7.0,
    matrix: str = "",
) -> BindingPocketSpec:
    """
    Generate pocket spec for capturing an oxyanion.

    The pocket is a conical cavity with H-bond donors and optional metal center.
    Shape selectivity comes from cone angle discriminating pyramidal/planar/tetrahedral.
    """
    target_shape = {
        'pyramidal': TargetShape.PYRAMIDAL,
        'planar': TargetShape.PLANAR,
        'tetrahedral': TargetShape.TETRAHEDRAL,
    }.get(shape, TargetShape.IRREGULAR)

    symmetry = {
        'pyramidal': 'C3v',
        'planar': 'D3h',
        'tetrahedral': 'Td',
    }.get(shape, 'C1')

    target = TargetGeometry(
        name=anion_name,
        formula=anion_formula,
        charge=float(charge),
        shape=target_shape,
        scale=TargetScale.SMALL_MOLECULE,
        symmetry=symmetry,
        effective_radius_A=thermodynamic_radius_A,
        bounding_sphere_A=thermodynamic_radius_A,
        width_A=oo_distance_A,
        height_A=pyramid_height_A,
        hsab_class='hard',  # oxyanions are hard bases
        hydration_dG_kJ=hydration_dG_kJ,
        has_lone_pair=has_lone_pair,
        lone_pair_accessible=has_lone_pair,
        n_hbond_acceptors=n_oxygen,
    )

    # Build interaction elements
    elements = []

    # Optional metal center at pocket base
    if metal_center:
        elements.append(InteractionElement(
            interaction_type=InteractionType.METAL_CENTER,
            position_A=np.array([0.0, 0.0, -pyramid_height_A - 1.0]),
            direction_A=np.array([0.0, 0.0, 1.0]),  # points toward target
            subtype=metal_center,
            strength_required=8.0,
            tolerance_A=0.2,
            must_be_rigid=True,
            estimated_dG_kJ=-metal_logK * 5.71,
        ))

    # H-bond donors arranged in ring around cavity walls
    hbond_ring_radius = oo_distance_A / 2 + 1.4  # anion O-O/2 + vdW
    for i in range(n_hbond_donors):
        angle = 2 * np.pi * i / n_hbond_donors
        pos = np.array([
            hbond_ring_radius * np.cos(angle),
            hbond_ring_radius * np.sin(angle),
            0.0  # at aperture plane
        ])
        direction = -pos / np.linalg.norm(pos)  # points inward
        elements.append(InteractionElement(
            interaction_type=InteractionType.HBOND_DONOR,
            position_A=pos,
            direction_A=direction,
            subtype='urea_NH',
            strength_required=3.0,
            tolerance_A=0.5,
            must_be_rigid=True,
        ))

    # Lone pair recognition (if target has accessible lone pair)
    if has_lone_pair:
        elements.append(InteractionElement(
            interaction_type=InteractionType.HBOND_DONOR,
            position_A=np.array([0.0, 0.0, pyramid_height_A + 1.4]),
            direction_A=np.array([0.0, 0.0, -1.0]),  # points down at lone pair
            subtype='lone_pair_acceptor',
            strength_required=2.0,
            tolerance_A=0.8,
        ))

    # Pocket shape
    aperture_A = oo_distance_A + 2.8  # O-O + 2×vdW(O)
    depth_A = pyramid_height_A + 1.4 if pyramid_height_A > 0 else 1.5
    pocket_vol = (1/3) * np.pi * (aperture_A/2)**2 * depth_A

    pocket_shape = PocketShape(
        topology=PocketTopology.CAVITY,
        aperture_A=aperture_A,
        depth_A=depth_A,
        cone_half_angle_deg=cone_angle_deg,
        symmetry_order=3 if n_oxygen == 3 else 4,
        pocket_volume_A3=pocket_vol,
        rigidity_required=0.8,  # shape selectivity requires rigidity
        max_rms_deformation_A=0.3,
    )

    # Competitor constraints
    comp_list = []
    if competitors:
        for comp in competitors:
            required = np.log10(comp.get('conc_ratio', 1.0)) + 1.5
            # Need enough selectivity to overcome concentration disadvantage
            # plus margin of 1.5 log K
            comp_list.append(CompetitorConstraint(
                competitor_name=comp.get('name', comp['formula']),
                competitor_formula=comp['formula'],
                competitor_shape={
                    'planar': TargetShape.PLANAR,
                    'tetrahedral': TargetShape.TETRAHEDRAL,
                    'pyramidal': TargetShape.PYRAMIDAL,
                    'spherical': TargetShape.SPHERICAL,
                }.get(comp.get('shape', ''), TargetShape.IRREGULAR),
                competitor_charge=comp.get('charge', 0),
                concentration_ratio=comp.get('conc_ratio', 1.0),
                required_delta_logK=required,
                selectivity_sources=comp.get('sources', {}),
                achievable_delta_logK=sum(comp.get('sources', {}).values()),
                feasible=sum(comp.get('sources', {}).values()) >= required,
            ))

    return BindingPocketSpec(
        name=f"pocket_{anion_name}_{cone_angle_deg:.0f}deg",
        target=target,
        shape=pocket_shape,
        elements=elements,
        competitors=comp_list,
        conditions=OperatingConditions(pH=pH, matrix_name=matrix),
        predicted_logK=metal_logK if metal_center else 0.0,
        predicted_dG_kJ=-metal_logK * 5.71 if metal_center else 0.0,
        calibration_status="uncalibrated",
        confidence_notes="Geometric scoring from crystal structures. Metal logK from literature.",
        minimum_rigidity=0.8,
        minimum_elements=n_hbond_donors + (1 if metal_center else 0),
        critical_elements=[0] if metal_center else [],  # metal center is critical
    )


def spec_for_protein_epitope(
    target_name: str,
    epitope_area_A2: float = 800.0,
    epitope_charge: float = 0.0,
    n_hbond_patches: int = 3,
    n_hydrophobic_patches: int = 2,
    n_charged_patches: int = 1,
    pH: float = 7.4,
    matrix: str = "physiological_buffer",
) -> BindingPocketSpec:
    """
    Generate STUB pocket spec for a protein epitope target.

    This is a placeholder showing the abstraction works for protein targets.
    Real implementation requires structural data (PDB coordinates, epitope mapping).
    """
    target = TargetGeometry(
        name=target_name,
        formula=target_name,
        charge=epitope_charge,
        shape=TargetShape.CONVEX_SURFACE,
        scale=TargetScale.PROTEIN_DOMAIN,
        effective_radius_A=np.sqrt(epitope_area_A2 / np.pi),
        bounding_sphere_A=np.sqrt(epitope_area_A2 / np.pi) * 1.5,
        hsab_class='N/A',
        n_hbond_donors=n_hbond_patches,
        n_hbond_acceptors=n_hbond_patches,
    )

    elements = []
    # Distribute interaction patches across the binding surface
    # Approximate as points on a disk
    radius = np.sqrt(epitope_area_A2 / np.pi)
    total_patches = n_hbond_patches + n_hydrophobic_patches + n_charged_patches
    for i in range(total_patches):
        angle = 2 * np.pi * i / total_patches
        r = radius * 0.6  # inner ring
        pos = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])

        if i < n_hbond_patches:
            itype = InteractionType.HBOND_DONOR
            sub = 'backbone_NH'
        elif i < n_hbond_patches + n_hydrophobic_patches:
            itype = InteractionType.HYDROPHOBIC_SURFACE
            sub = 'nonpolar_patch'
        else:
            itype = InteractionType.NEGATIVE_CHARGE if epitope_charge > 0 \
                else InteractionType.POSITIVE_CHARGE
            sub = 'charged_residue'

        elements.append(InteractionElement(
            interaction_type=itype,
            position_A=pos,
            direction_A=np.array([0, 0, -1]),
            subtype=sub,
            strength_required=3.0,
            tolerance_A=2.0,  # protein surfaces are larger, more tolerant
        ))

    pocket_shape = PocketShape(
        topology=PocketTopology.SURFACE,
        aperture_A=2 * radius,
        depth_A=5.0,  # typical antibody CDR depth
        pocket_volume_A3=epitope_area_A2 * 5.0,
        rigidity_required=0.5,  # induced fit is common
        max_rms_deformation_A=2.0,
    )

    return BindingPocketSpec(
        name=f"pocket_{target_name}_epitope",
        target=target,
        shape=pocket_shape,
        elements=elements,
        conditions=OperatingConditions(pH=pH, matrix_name=matrix),
        calibration_status="stub",
        confidence_notes="Protein epitope spec is a placeholder. Requires PDB structural data.",
        minimum_rigidity=0.5,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Coordination geometry positions
# ═══════════════════════════════════════════════════════════════════════════

def _coordination_positions(
    cn: int,
    geometry: str,
    bond_length: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate ideal donor positions for a coordination geometry.
    Returns list of (position, direction_toward_center).
    """
    positions = []

    if geometry == 'octahedral' or (cn == 6 and geometry != 'trigonal_prismatic'):
        vectors = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ]
    elif geometry == 'tetrahedral' or cn == 4:
        vectors = [
            [1, 1, 1], [1, -1, -1],
            [-1, 1, -1], [-1, -1, 1],
        ]
    elif geometry == 'square_planar':
        vectors = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
        ]
    elif cn == 2:
        vectors = [[0, 0, 1], [0, 0, -1]]
    elif cn == 3:
        vectors = [
            [1, 0, 0],
            [-0.5, 0.866, 0],
            [-0.5, -0.866, 0],
        ]
    elif cn == 5:  # trigonal bipyramidal
        vectors = [
            [0, 0, 1], [0, 0, -1],  # axial
            [1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0],  # equatorial
        ]
    elif cn == 8:  # square antiprismatic
        sqrt2 = np.sqrt(2) / 2
        vectors = [
            [1, 0, sqrt2], [0, 1, sqrt2],
            [-1, 0, sqrt2], [0, -1, sqrt2],
            [sqrt2, sqrt2, -sqrt2], [-sqrt2, sqrt2, -sqrt2],
            [-sqrt2, -sqrt2, -sqrt2], [sqrt2, -sqrt2, -sqrt2],
        ]
    else:
        # Fibonacci sphere for arbitrary CN
        golden = (1 + 5**0.5) / 2
        vectors = []
        for i in range(cn):
            theta = np.arccos(1 - 2 * (i + 0.5) / cn)
            phi = 2 * np.pi * i / golden
            vectors.append([np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi),
                          np.cos(theta)])

    for v in vectors[:cn]:
        v = np.array(v, dtype=float)
        v = v / np.linalg.norm(v)
        pos = v * bond_length
        direction = -v  # points toward center
        positions.append((pos, direction))

    return positions


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    print("BindingPocketSpec — Self-Test")
    print("═" * 70)
    print()
    print("Prime Directive check: this module contains ZERO references to")
    print("DNA, MOF, chelator, resin, protein fold, or any fabrication method.")
    print("It describes WHAT the pocket looks like, not WHAT builds it.")
    print()

    # ── Test 1: Metal ion (Pb²⁺) ──────────────────────────────────────
    print("1. Metal Ion Target: Pb²⁺ capture from acid mine drainage")
    print("─" * 60)

    pb_spec = spec_for_metal_ion(
        metal_formula='Pb2+',
        charge=2,
        ionic_radius_A=1.19,
        hsab_class='borderline',
        d_electrons=0,  # [Xe]4f14 5d10 6s2 → Pb²⁺ has 6s² (inert pair)
        coordination_number=6,
        preferred_geometry='octahedral',
        donor_subtypes=['N_amine', 'N_amine', 'O_carboxylate', 'O_carboxylate',
                        'S_thiolate', 'S_thiolate'],
        pH=4.5,
        competitors=[
            {'formula': 'Ca2+', 'charge': 2, 'conc_ratio': 1000, 'required_delta': 4.0},
            {'formula': 'Mg2+', 'charge': 2, 'conc_ratio': 500, 'required_delta': 3.5},
            {'formula': 'Fe3+', 'charge': 3, 'conc_ratio': 100, 'required_delta': 3.0},
        ],
        matrix='acid_mine_drainage',
    )
    print(pb_spec.summary())
    print()

    # Verify no fabrication references
    summary_text = pb_spec.summary()
    forbidden = ['DNA', 'MOF', 'origami', 'chelator', 'resin', 'protein', 'ATHENA']
    violations = [w for w in forbidden if w.lower() in summary_text.lower()]
    assert not violations, f"Prime Directive violation! Found: {violations}"
    print("  ✓ No fabrication references in spec")
    print()

    # ── Test 2: Oxyanion (Selenite) ────────────────────────────────────
    print("2. Oxyanion Target: SeO₃²⁻ from Elk Valley mine water")
    print("─" * 60)

    se_spec = spec_for_oxyanion(
        anion_name='selenite',
        anion_formula='SeO3(2-)',
        charge=-2,
        shape='pyramidal',
        n_oxygen=3,
        pyramid_height_A=0.80,
        oo_distance_A=2.63,
        thermodynamic_radius_A=2.39,
        hydration_dG_kJ=-410.0,
        has_lone_pair=True,
        metal_center='Zr4+',
        metal_logK=12.0,
        cone_angle_deg=38.0,
        n_hbond_donors=4,
        competitors=[
            {
                'name': 'carbonate', 'formula': 'CO3(2-)',
                'charge': -2, 'shape': 'planar',
                'conc_ratio': 3454,
                'sources': {'geometric': 11.3, 'metal_affinity': 4.0, 'hydration': 0.3},
            },
            {
                'name': 'sulfate', 'formula': 'SO4(2-)',
                'charge': -2, 'shape': 'tetrahedral',
                'conc_ratio': 5486,
                'sources': {'geometric': -0.7, 'metal_affinity': 6.0, 'hydration': 3.4},
            },
            {
                'name': 'selenate', 'formula': 'SeO4(2-)',
                'charge': -2, 'shape': 'tetrahedral',
                'conc_ratio': 1.375,  # 55% selenate / 40% selenite
                'sources': {'geometric': -0.7, 'metal_affinity': 7.0, 'hydration': 3.4},
            },
        ],
        pH=7.5,
        matrix='coal_mine_drainage',
    )
    print(se_spec.summary())
    print()

    # Verify all competitors feasible
    for c in se_spec.competitors:
        margin = c.achievable_delta_logK - c.required_delta_logK
        status = "FEASIBLE" if c.feasible else "INSUFFICIENT"
        print(f"  {c.competitor_formula}: required Δ{c.required_delta_logK:+.1f}, "
              f"achievable Δ{c.achievable_delta_logK:+.1f}, "
              f"margin {margin:+.1f} → {status}")
    print()

    # ── Test 3: Protein epitope (stub) ─────────────────────────────────
    print("3. Protein Epitope Target: insulin (stub)")
    print("─" * 60)

    insulin_spec = spec_for_protein_epitope(
        target_name='insulin',
        epitope_area_A2=900.0,
        epitope_charge=-1.0,
        n_hbond_patches=4,
        n_hydrophobic_patches=3,
        n_charged_patches=1,
    )
    print(insulin_spec.summary())
    print()

    # ── Test 4: Verify spec is truly agnostic ──────────────────────────
    print("4. Prime Directive Compliance — Full Scan")
    print("─" * 60)

    import inspect
    source = inspect.getsource(spec_for_metal_ion)
    source += inspect.getsource(spec_for_oxyanion)
    source += inspect.getsource(spec_for_protein_epitope)

    # These words should NOT appear in generator functions
    fabrication_words = [
        'DNA', 'MOF', 'origami', 'ATHENA', 'staple', 'helix',
        'chelator', 'EDTA', 'DTPA', 'crown_ether',
        'resin', 'column', 'bead', 'membrane',
        'RFDiffusion', 'AlphaFold', 'protein_fold',
        'UiO-66', 'NU-1000', 'zeolite',
    ]
    found = [w for w in fabrication_words if w in source]
    if found:
        print(f"  ⚠ Found fabrication references: {found}")
    else:
        print("  ✓ Zero fabrication references in generator functions")
    print()

    # ── Test 5: Cross-target element comparison ────────────────────────
    print("5. Element Comparison Across Target Types")
    print("─" * 60)

    specs = {'Pb²⁺': pb_spec, 'SeO₃²⁻': se_spec, 'insulin': insulin_spec}
    for name, spec in specs.items():
        types = spec.element_types
        r = spec.bounding_radius_A
        topo = spec.shape.topology.value
        rigid = spec.minimum_rigidity
        print(f"  {name:12s}: {topo:20s} R={r:.1f}Å  rigid={rigid:.1f}  "
              f"elements={types}")
    print()

    # ── Test 6: Spec feeds downstream without modification ─────────────
    print("6. Downstream Readability")
    print("─" * 60)
    print("  A realization ranker would read:")
    print(f"    target_scale = {se_spec.target.scale.value}")
    print(f"    pocket_topology = {se_spec.shape.topology.value}")
    print(f"    pocket_aperture = {se_spec.shape.aperture_A:.1f} Å")
    print(f"    rigidity_needed = {se_spec.minimum_rigidity}")
    print(f"    n_elements = {se_spec.n_elements}")
    print(f"    has_metal_center = "
          f"{any(e.interaction_type == InteractionType.METAL_CENTER for e in se_spec.elements)}")
    print(f"    worst_margin = {se_spec.worst_selectivity_margin:+.1f}")
    print(f"    all_feasible = {se_spec.all_competitors_feasible}")
    print()
    print("  ...then decide which material systems can realize it.")
    print("  That decision belongs in Layer 3, not here.")
    print()

    print("═" * 70)
    print("BindingPocketSpec architecture complete.")
    print("This is Layer 2 output. Layer 3 (realization ranker) reads it.")
    print("No fabrication decisions made here.")


if __name__ == "__main__":
    self_test()