"""
Data models for the MABE Realization Engine.

These are physics objects. They do not know what a protein is.

Hierarchy:
    InteractionGeometrySpec  (Layer 2 output, our input)
    → IdealPocketSpec        (Phase 1 output: the physics optimum)
    → DeviationReport        (per-material deviation from ideal)
    → RealizationScore       (per-material composite score)
    → RankedRealizations     (sorted output with gap analysis)
    → FabricationSpec        (Layer 4 output: buildable design)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class CavityShape(str, Enum):
    """Pocket geometry classification."""
    SPHERE = "sphere"
    CONE = "cone"
    CHANNEL = "channel"
    CLEFT = "cleft"
    FLAT = "flat"
    BARREL = "barrel"
    CUSTOM = "custom"


class RigidityClass(str, Enum):
    """
    Derived from tightest precision requirement in the ideal pocket.
    <0.05 Å → crystalline, <0.2 → preorganized, <0.5 → semi-flexible, else → any.
    """
    CRYSTALLINE = "crystalline"
    PREORGANIZED = "preorganized"
    SEMI_FLEXIBLE = "semi-flexible"
    ANY = "any"


class Solvent(str, Enum):
    AQUEOUS = "aqueous"
    ORGANIC = "organic"
    MIXED = "mixed"
    GAS = "gas"


class ApplicationContext(str, Enum):
    DIAGNOSTIC = "diagnostic"
    REMEDIATION = "remediation"
    THERAPEUTIC = "therapeutic"
    RESEARCH = "research"
    SEPARATION = "separation"
    CATALYSIS = "catalysis"


class ScaleClass(str, Enum):
    NMOL = "nmol"
    UMOL = "µmol"
    MMOL = "mmol"
    MOL = "mol"
    KMOL = "kmol"

    @property
    def rank(self) -> int:
        return list(ScaleClass).index(self)


# ─────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class DonorPosition:
    """A single interaction element in the geometry spec."""
    atom_type: str                    # "N", "O", "S", "Se", etc.
    coordination_role: str            # "axial", "equatorial", "bridging", "terminal"
    position_vector_A: tuple[float, float, float]  # relative to cavity center
    tolerance_A: float                # how precisely this must be placed
    required_hybridization: str       # "sp2", "sp3", "any"
    charge_state: float = 0.0        # partial charge requirement


@dataclass(frozen=True)
class ExclusionSpec:
    """A species that must NOT bind."""
    species: str
    max_allowed_affinity_kJ_mol: float
    exclusion_mechanism: str          # "size", "charge", "geometry", "kinetic"


@dataclass(frozen=True)
class CavityDimensions:
    """Physical dimensions of the pocket cavity."""
    volume_A3: float
    aperture_A: float                 # narrowest opening
    depth_A: float
    max_internal_diameter_A: float
    aspect_ratio: float = 1.0        # depth / width


@dataclass(frozen=True)
class HydrophobicSurface:
    """A non-polar contact region in the pocket."""
    center_A: tuple[float, float, float]
    area_A2: float
    normal_vector: tuple[float, float, float]


@dataclass(frozen=True)
class HBondSpec:
    """Hydrogen bond network requirement."""
    donors: list[tuple[float, float, float]]   # positions of H-bond donors
    acceptors: list[tuple[float, float, float]] # positions of H-bond acceptors
    required_geometry: str = "any"              # "linear", "bifurcated", "any"


# ─────────────────────────────────────────────
# Layer 2 Output / Layer 3 Input
# ─────────────────────────────────────────────

@dataclass
class InteractionGeometrySpec:
    """
    Realization-agnostic pocket description. Layer 2 output.

    This is a physics object. It describes a field of interaction
    potentials in 3D space. It does not know what a protein is.
    """

    # ── Cavity geometry ──
    cavity_shape: CavityShape
    cavity_dimensions: CavityDimensions
    symmetry: str = "none"           # "C3v", "D4h", "none", etc.

    # ── Interaction elements ──
    donor_positions: list[DonorPosition] = field(default_factory=list)
    hydrophobic_surfaces: list[HydrophobicSurface] = field(default_factory=list)
    h_bond_network: Optional[HBondSpec] = None

    # ── Flexibility constraints ──
    rigidity_requirement: str = "semi-rigid"
    max_backbone_rmsd_A: float = 1.0
    conformational_penalty_budget_kJ_mol: float = 10.0

    # ── Scale ──
    pocket_scale_nm: float = 0.5
    multivalency: int = 1

    # ── Selectivity constraints ──
    must_exclude: list[ExclusionSpec] = field(default_factory=list)

    # ── Operating conditions ──
    pH_range: tuple[float, float] = (5.0, 9.0)
    temperature_range_K: tuple[float, float] = (273.15, 373.15)
    solvent: Solvent = Solvent.AQUEOUS
    ionic_strength_M: float = 0.1

    # ── Application context (informs realization, doesn't constrain geometry) ──
    target_application: ApplicationContext = ApplicationContext.RESEARCH
    required_scale: ScaleClass = ScaleClass.UMOL
    cost_ceiling_per_unit: Optional[float] = None
    reusability_required: bool = False

    @property
    def required_donor_types(self) -> set[str]:
        return {d.atom_type for d in self.donor_positions}

    @property
    def tightest_tolerance_A(self) -> float:
        if not self.donor_positions:
            return float("inf")
        return min(d.tolerance_A for d in self.donor_positions)


# ─────────────────────────────────────────────
# Layer 3 Phase 1 Output: The Ideal Pocket
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class IdealElement:
    """One interaction element in the physics-optimal pocket."""
    atom_type: str
    exact_position_A: tuple[float, float, float]
    required_precision_A: float
    orbital_hybridization: str
    charge_state: float
    interaction_energy_contribution_kJ_mol: float


@dataclass
class IdealPocketSpec:
    """
    The physics-optimal pocket. No material constraints. Pure geometry + thermodynamics.

    This is the reference standard. Every material system is scored by
    deviation from this object.
    """

    # ── Computed from InteractionGeometrySpec ──
    optimal_elements: list[IdealElement]

    # ── Derived pocket properties ──
    ideal_cavity_volume_A3: float
    ideal_cavity_shape: CavityShape
    ideal_desolvation_energy_kJ_mol: float
    ideal_binding_energy_kJ_mol: float

    # ── Fabrication requirements (material-agnostic) ──
    min_precision_required_A: float
    rigidity_class: RigidityClass
    min_stability_pH: tuple[float, float]
    min_stability_K: tuple[float, float]
    required_elements: set[str]
    symmetry_exploitable: bool

    # ── The ideal material spec (when nothing scores high enough) ──
    ideal_material_requirements: str = ""
    critical_constraints: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Layer 3 Phase 2: Deviation + Scoring
# ─────────────────────────────────────────────

@dataclass
class DeviationReport:
    """How a specific material system deviates from the IdealPocketSpec."""
    material_system: str
    element_deviations_A: list[float]
    max_deviation_A: float
    mean_deviation_A: float
    rigidity_deviation: float = 0.0          # 0 = exact match, 1 = completely wrong
    electrostatic_field_correlation: float = 1.0  # 0–1
    missing_interactions: list[str] = field(default_factory=list)
    compensating_interactions: list[str] = field(default_factory=list)


@dataclass
class RealizationScore:
    """Score for one material system against the IdealPocketSpec."""

    material_system: str
    adapter_id: str

    # ── Physics deviation (PRIMARY) ──
    deviation_from_ideal: DeviationReport
    physics_fidelity: float               # 0.0–1.0, derived from deviation

    # ── Implementation axes (SECONDARY — precision binders) ──
    synthetic_accessibility: float = 0.0
    cost_score: float = 0.0
    scalability: float = 0.0
    operating_condition_compatibility: float = 0.0
    reusability_score: float = 0.0

    # ── Bulk sorbent axes (populated by Class D adapters) ──
    capacity_mmol_per_g: float = 0.0       # qmax from isotherm
    selectivity_factor: float = 0.0        # α = Kd_target / Kd_competitor
    throughput_L_per_h_per_kg: float = 0.0 # column flow at breakthrough
    regenerability_cycles: int = 0         # sorption/desorption cycles
    cost_per_kg_processed: float = 0.0     # $/kg of target removed

    # ── Physics class tag ──
    physics_class: str = ""                # "covalent_cavity", "bulk_sorbent", etc.

    # ── Composite ──
    composite_score: float = 0.0
    confidence: float = 0.0               # calibrated from literature success rates

    # ── Rationale ──
    advantages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    critical_risk: Optional[str] = None

    # ── Feasibility gate ──
    feasible: bool = True
    infeasibility_reason: Optional[str] = None


@dataclass
class RankedRealizations:
    """Layer 3 complete output."""

    # ── The physics target ──
    geometry_spec: InteractionGeometrySpec
    ideal_pocket: IdealPocketSpec

    # ── Material rankings ──
    rankings: list[RealizationScore]
    recommended: str
    recommendation_rationale: str = ""

    # ── Gap analysis ──
    best_physics_fidelity: float = 0.0
    gap_to_ideal: float = 1.0
    gap_report: Optional[str] = None
    novel_material_suggestion: Optional[str] = None


# ─────────────────────────────────────────────
# Layer 4 Output: Fabrication Spec
# ─────────────────────────────────────────────

@dataclass
class FabricationSpec:
    """
    Base class for all Layer 4 adapter outputs.
    Each adapter subclasses with material-specific fields.
    """

    material_system: str
    geometry_spec_hash: str               # traceability to input
    predicted_pocket_geometry: CavityDimensions
    predicted_deviation_from_ideal_A: float

    # ── Synthesis / fabrication ──
    synthesis_steps: list[str] = field(default_factory=list)
    estimated_yield: float = 0.0
    estimated_cost_per_unit: float = 0.0
    estimated_time: str = ""

    # ── Characterization plan ──
    validation_experiments: list[str] = field(default_factory=list)
    expected_observables: dict = field(default_factory=dict)

    # ── Files ──
    structure_file: Optional[str] = None  # PDB, CIF, MOL, oxDNA, etc.
    order_sheet: Optional[str] = None
    protocol: Optional[str] = None
