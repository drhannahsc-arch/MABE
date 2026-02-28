"""
Data models for the MABE Realization Engine.

These are physics objects. They do not know what a protein is.

Hierarchy:
    InteractionSpec              (polymorphic base for all Layer 2 outputs)
    ├─ DiscretePocketSpec        (cavity with positioned donors — the original)
    │  alias: InteractionGeometrySpec (backward compat)
    ├─ NetworkInteractionSpec    (future: SAPs, hydrogels, IX resins)
    ├─ SurfaceInteractionSpec    (future: activated carbon, lignin, clays)
    ├─ BulkInteractionSpec       (future: ILs, DES, extraction)
    ├─ FieldInteractionSpec      (future: photonic, acoustic)
    └─ CompositeInteractionSpec  (future: multi-paradigm materials)

    InteractionSpec
    → IdealSpec                  (paradigm-specific physics optimum)
    → DeviationReport            (per-material deviation from ideal)
    → RealizationScore           (per-material composite score)
    → RankedRealizations         (sorted output with gap analysis)
    → FabricationSpec            (Layer 4 output: buildable design)
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


class InteractionParadigm(str, Enum):
    """Which physics paradigm governs the interaction."""
    POCKET = "pocket"           # Discrete cavity with positioned donors
    NETWORK = "network"         # Crosslinked polymer (SAP, hydrogel, IX resin)
    SURFACE = "surface"         # Distributed sites on a material surface
    BULK = "bulk"               # Volume phenomenon (partitioning, dissolution)
    FIELD = "field"             # Electromagnetic, acoustic, thermal
    COMPOSITE = "composite"     # Multiple paradigms combined


# ─────────────────────────────────────────────
# Base class for all Layer 2 outputs
# ─────────────────────────────────────────────

class InteractionSpec:
    """
    Polymorphic base for all Layer 2 interaction specifications.

    NOT a dataclass — avoids field-ordering inheritance issues.
    Each subtype (DiscretePocketSpec, NetworkInteractionSpec, etc.)
    is a dataclass that inherits from this and defines its own fields.

    This class provides:
        - spec_type property for polymorphic dispatch
        - isinstance() checking across the type hierarchy
        - Common interface that Layer 3 can depend on

    The universal fields (pH_range, temperature_range_K, solvent,
    ionic_strength_M, must_exclude, target_application, required_scale,
    cost_ceiling_per_unit, reusability_required) live on each subtype
    individually. This is deliberate: dataclass inheritance with mixed
    default/non-default fields across parent/child is fragile. Each
    paradigm owns its full field set.
    """

    @property
    def spec_type(self) -> str:
        """Which interaction paradigm this spec represents."""
        raise NotImplementedError(
            f"{type(self).__name__} must define spec_type"
        )


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
class DiscretePocketSpec(InteractionSpec):
    """
    Interaction via a defined cavity with positioned interaction elements.

    This is a physics object. It describes a field of interaction
    potentials in 3D space. It does not know what a protein is.

    Applies to: chelators, macrocycles, protein pockets, cage interiors,
    MOF nodes, zeolite cages, MIP cavities, cucurbiturils, cryptands.
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
    def spec_type(self) -> str:
        return InteractionParadigm.POCKET.value

    @property
    def required_donor_types(self) -> set[str]:
        return {d.atom_type for d in self.donor_positions}

    @property
    def tightest_tolerance_A(self) -> float:
        if not self.donor_positions:
            return float("inf")
        return min(d.tolerance_A for d in self.donor_positions)


# Backward compatibility — all existing code uses this name.
InteractionGeometrySpec = DiscretePocketSpec


# ─────────────────────────────────────────────
# Network Interaction Paradigm
# ─────────────────────────────────────────────

class ResinType(str, Enum):
    """Ion exchange resin functional group class."""
    SAC = "strong_acid_cation"    # R-SO3H (sulfonate)
    WAC = "weak_acid_cation"      # R-COOH (carboxylate)
    SBA = "strong_base_anion"     # R-NR3OH (quaternary amine)
    WBA = "weak_base_anion"       # R-NR2 (tertiary amine)
    CHELATING = "chelating"       # IDA, BPA, etc.


class NetworkMechanism(str, Enum):
    """Primary mechanism of the network interaction."""
    ION_EXCHANGE = "ion_exchange"         # Donnan + selectivity coefficients
    OSMOTIC_SWELLING = "osmotic_swelling" # Flory-Rehner + polyelectrolyte
    CHELATING_RESIN = "chelating_resin"   # Fixed chelators on polymer backbone
    ABSORPTION = "absorption"             # Bulk uptake into gel phase


@dataclass
class NetworkInteractionSpec(InteractionSpec):
    """
    Interaction via a crosslinked polymer network.

    Applies to: ion exchange resins, superabsorbent polymers, hydrogels,
    chelating resins, functionalized polymer beads.

    The physics is Donnan equilibrium + selectivity coefficients for IX,
    Flory-Rehner + rubber elasticity for swelling networks.
    This is NOT a pocket — there is no defined cavity geometry.
    """

    # ── Target species ──
    target_species: str = ""                  # "Pb2+", "Ca2+", "NO3-", etc.
    target_charge: int = 0                    # +2, -1, etc.
    target_hydrated_radius_A: float = 0.0     # Governs IX selectivity

    # ── Competing species (matrix) ──
    competing_species: list[str] = field(default_factory=list)  # ["Na+", "Ca2+", ...]
    competing_concentrations_mM: list[float] = field(default_factory=list)

    # ── Network mechanism ──
    mechanism: NetworkMechanism = NetworkMechanism.ION_EXCHANGE

    # ── IX-specific design variables ──
    resin_type: Optional[ResinType] = None
    crosslink_pct: float = 8.0               # % DVB for styrenic resins
    target_capacity_meq_per_mL: float = 0.0  # desired exchange capacity
    target_selectivity_over: str = ""         # reference counter-ion ("Na+", "H+")
    min_selectivity_coefficient: float = 1.0  # K_target/K_reference

    # ── Swelling-specific design variables ──
    target_swelling_ratio: float = 0.0        # Q = swollen_mass / dry_mass
    target_modulus_kPa: float = 0.0           # mechanical integrity
    target_mesh_size_A: float = 0.0           # pore size in network

    # ── Performance requirements ──
    target_capacity_mg_per_g: float = 0.0     # mass-based capacity
    min_removal_efficiency: float = 0.0       # 0.0-1.0
    throughput_BV_per_cycle: int = 0          # bed volumes before breakthrough

    # ── Selectivity constraints ──
    must_exclude: list[ExclusionSpec] = field(default_factory=list)

    # ── Operating conditions ──
    pH_range: tuple[float, float] = (1.0, 14.0)
    temperature_range_K: tuple[float, float] = (273.15, 353.15)
    solvent: Solvent = Solvent.AQUEOUS
    ionic_strength_M: float = 0.1

    # ── Application context ──
    target_application: ApplicationContext = ApplicationContext.REMEDIATION
    required_scale: ScaleClass = ScaleClass.MOL
    cost_ceiling_per_unit: Optional[float] = None
    reusability_required: bool = True          # IX resins are almost always regenerated

    @property
    def spec_type(self) -> str:
        return InteractionParadigm.NETWORK.value


# ─────────────────────────────────────────────
# Surface Interaction Paradigm
# ─────────────────────────────────────────────

class SurfaceMechanism(str, Enum):
    """Primary mechanism of the surface interaction."""
    PHYSISORPTION = "physisorption"           # van der Waals, pore filling
    CHEMISORPTION = "chemisorption"           # Covalent/ionic bond to surface site
    SURFACE_COMPLEXATION = "surface_complexation"  # Metal-oxide coordination
    ELECTROSTATIC = "electrostatic"           # Coulombic attraction to charged surface
    ION_EXCHANGE_SURFACE = "ion_exchange_surface"  # Exchange at surface sites


class IsothermModel(str, Enum):
    """Which isotherm model governs the equilibrium."""
    LANGMUIR = "langmuir"           # Monolayer, homogeneous
    FREUNDLICH = "freundlich"       # Multilayer, heterogeneous
    BET = "bet"                     # Multilayer gas-phase
    DUBININ_RADUSHKEVICH = "dr"     # Micropore filling


class BaseMaterial(str, Enum):
    """Surface material class."""
    ACTIVATED_CARBON = "activated_carbon"
    BIOCHAR = "biochar"
    LIGNIN = "lignin"
    CLAY = "clay"
    ZEOLITE_SURFACE = "zeolite_surface"      # Surface mode (not pocket mode)
    CELLULOSE = "cellulose"
    GRAPHENE_OXIDE = "graphene_oxide"
    METAL_OXIDE = "metal_oxide"              # Fe2O3, Al2O3, MnO2, etc.


@dataclass
class SurfaceInteractionSpec(InteractionSpec):
    """
    Interaction via distributed sites on a material surface.

    Applies to: activated carbon, biochar, lignin sorbents, clay minerals,
    functionalized cellulose, graphene oxide, metal oxide adsorbents.

    The physics is adsorption isotherms (Langmuir/Freundlich) +
    surface complexation + pore diffusion. There is no defined cavity —
    binding occurs at distributed functional group sites on a surface.
    """

    # ── Target species ──
    target_species: str = ""                  # "Pb2+", "phenol", "PFOS", etc.
    target_charge: int = 0
    target_mw_g_mol: float = 0.0              # molecular weight
    target_hydrophobicity: float = 0.0        # log P (for organics)

    # ── Competing species (matrix) ──
    competing_species: list[str] = field(default_factory=list)
    competing_concentrations_mM: list[float] = field(default_factory=list)

    # ── Surface mechanism ──
    mechanism: SurfaceMechanism = SurfaceMechanism.CHEMISORPTION
    isotherm_model: IsothermModel = IsothermModel.LANGMUIR

    # ── Material constraints ──
    base_material: Optional[BaseMaterial] = None  # None = auto-select
    min_surface_area_m2_g: float = 0.0            # BET requirement
    target_pore_size_A: float = 0.0               # for size-selective adsorption

    # ── Performance requirements ──
    target_capacity_mg_g: float = 0.0             # qmax requirement
    target_removal_efficiency: float = 0.0        # 0.0-1.0 at given C₀
    initial_concentration_mg_L: float = 0.0       # C₀ for design
    target_contact_time_min: float = 0.0          # kinetics requirement

    # ── Functional group requirements ──
    required_functional_groups: list[str] = field(default_factory=list)
    # e.g., ["carboxyl", "phenol", "amine"] — surface chemistry needed

    # ── Selectivity constraints ──
    must_exclude: list[ExclusionSpec] = field(default_factory=list)

    # ── Operating conditions ──
    pH_range: tuple[float, float] = (2.0, 12.0)
    temperature_range_K: tuple[float, float] = (273.15, 353.15)
    solvent: Solvent = Solvent.AQUEOUS
    ionic_strength_M: float = 0.1

    # ── Application context ──
    target_application: ApplicationContext = ApplicationContext.REMEDIATION
    required_scale: ScaleClass = ScaleClass.MOL
    cost_ceiling_per_unit: Optional[float] = None
    reusability_required: bool = False

    @property
    def spec_type(self) -> str:
        return InteractionParadigm.SURFACE.value


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

    # ── Computed from DiscretePocketSpec (InteractionGeometrySpec) ──
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