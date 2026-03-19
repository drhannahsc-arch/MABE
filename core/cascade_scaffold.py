"""
core/cascade_scaffold.py — Cascade Scaffold Adapter

Designs and scores 3D scaffold-mediated capture-transform cascades where
multiple heterogeneous functional modules are positioned within a confined
volume to execute sequential chemistry:

  Module A (capture) → Module B (activation/co-reactant) → Module C (nucleation)

The scaffold (DNA origami, MOF, coordination cage, etc.) provides:
  1. Confinement: enhances intermediate concentration by 100-10000×
  2. Proximity coupling: sub-µs diffusion between modules at <5 nm spacing
  3. Pore selectivity: geometric pre-filtering of species entering the scaffold
  4. Module stoichiometry: architectural control of module ratios

Physics models:
  - Confined diffusion (Einstein-Smoluchowski in spherical cavity)
  - Pore selectivity (hindered transport theory, Renkin 1954)
  - Confinement concentration enhancement (single-molecule-in-cavity)
  - Module stoichiometry optimization (limiting reagent analysis)

Connects to:
  - core/dna_origami_pocket.py (MolecularModule, CagePreset for DNA scaffolds)
  - core/capture_transform_scorer.py (CascadeBenefit output)
  - core/transform_enumerator.py (TransformationProduct input)
  - adapters/coordination_cage_adapter.py (Fujita/Nitschke cages)
  - adapters/mof_adapter.py (MOF cavity systems)

Data tier: Tier 2 (published physics + DOI references).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.transform_enumerator import (
    TransformationProduct,
    CoReactantSource,
    TurnoverMode,
    ProductPhase,
    ClickCompatibility,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

AVOGADRO = 6.022e23
BOLTZMANN_J = 1.381e-23     # J/K
R_kJ = 8.314e-3             # kJ/(mol·K)
T_STD = 298.15              # K
PI = math.pi
WATER_VISCOSITY_PA_S = 8.9e-4  # Pa·s at 25°C


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class ModuleRole(str, Enum):
    """Functional role of a module in a cascade."""
    CAPTURE = "capture"               # selective target recognition
    ACTIVATION = "activation"         # catalytic activation of bound target
    CO_REACTANT = "co_reactant"       # pre-loaded co-reactant source
    NUCLEATION = "nucleation"         # product nucleation template
    ELECTRON_RELAY = "electron_relay" # photoelectron transfer (Pattern 3)
    PRODUCT_TRAP = "product_trap"     # retains product, prevents back-reaction
    PORE_GATE = "pore_gate"           # controls pore selectivity


class ScaffoldSystem(str, Enum):
    """Material system for the cascade scaffold."""
    DNA_ORIGAMI = "dna_origami"           # 10-100 nm, max positional control
    FUJITA_CAGE = "fujita_cage"           # 1-5 nm, Pd₆L₄ / Pd₁₂L₂₄
    NITSCHKE_CAGE = "nitschke_cage"       # 1-3 nm, Fe₄L₆ tetrahedra
    MOF_CAVITY = "mof_cavity"             # 0.5-3 nm per cavity, infinite lattice
    POROUS_ORGANIC_CAGE = "poc"           # 1-5 nm, solution-processable
    METAL_ORGANIC_POLYHEDRON = "mop"      # 1-5 nm, discrete and soluble


class ScaleClass(str, Enum):
    """Deployment scale."""
    DIAGNOSTIC = "diagnostic"     # µL-mL, proof of concept
    LAB = "lab"                   # mL-L
    PILOT = "pilot"               # L-m³
    FIELD = "field"               # m³-ML/day
    INDUSTRIAL = "industrial"     # ML/day+


# ═══════════════════════════════════════════════════════════════════════════
# Data models — Cascade specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CascadeModule:
    """One functional module in a capture-transform cascade.

    Extends the concept from dna_origami_pocket.MolecularModule with
    capture-transform-specific roles and co-reactant chemistry.
    """
    module_id: str
    name: str                        # "Zn-CA mimic", "Ca²⁺-EDTA chelate"
    role: ModuleRole
    functional_group: str            # "Zn²⁺-cyclen", "EDTA-Ca", "-SO₃H"
    description: str                 # how this module functions in the cascade

    # Chemistry
    click_compatibility: ClickCompatibility = ClickCompatibility.SPAAC_ONLY
    conjugation_handle: str = "azide-SPAAC"  # how it attaches to scaffold

    # Spatial requirements
    functional_radius_nm: float = 0.5   # how far the active group extends
    required_orientation: str = "inward"  # "inward", "outward", "lateral"
    min_spacing_from_partner_nm: float = 1.0  # minimum distance to paired module
    max_spacing_from_partner_nm: float = 5.0  # maximum distance to paired module

    # Stoichiometry
    stoichiometric_ratio: float = 1.0  # relative to capture module (capture = 1.0)

    # Economics
    estimated_cost_per_module_usd: float = 10.0

    notes: str = ""


@dataclass
class PoreSpec:
    """Pore selectivity specification for the cascade scaffold."""
    target_hydrated_radius_nm: float         # target must fit through
    competitor_hydrated_radius_nm: float      # competitor should be excluded
    pore_diameter_nm: float                  # scaffold pore diameter
    selectivity_factor: float = 1.0          # calculated from hindered transport
    notes: str = ""


@dataclass
class CascadeSpec:
    """Complete specification for a capture-transform cascade.

    This is the Layer 2 output for cascade systems — analogous to
    InteractionGeometrySpec but for multi-module capture-transform.
    """
    # Identity
    name: str                        # "CO₂ mineralization cascade"
    description: str
    cascade_pattern: str             # "Pattern 1", "Pattern 2", etc.

    # Target
    target_formula: str
    product: TransformationProduct

    # Modules
    modules: list[CascadeModule]

    # Pore requirements
    pore_spec: Optional[PoreSpec] = None

    # Geometry constraints
    min_interior_volume_nm3: float = 10.0    # minimum cage interior
    max_interior_volume_nm3: float = 10000.0
    required_module_positions: int = 0       # auto-computed from modules

    # Operating conditions
    ph_range: tuple[float, float] = (5.0, 9.0)
    temperature_range_c: tuple[float, float] = (5.0, 45.0)
    aqueous: bool = True

    # Scale target
    target_scale: ScaleClass = ScaleClass.LAB

    def __post_init__(self):
        if self.required_module_positions == 0:
            self.required_module_positions = sum(
                max(1, int(m.stoichiometric_ratio)) for m in self.modules
            )

    @property
    def total_modules(self) -> int:
        return sum(max(1, int(m.stoichiometric_ratio)) for m in self.modules)

    @property
    def capture_modules(self) -> list[CascadeModule]:
        return [m for m in self.modules if m.role == ModuleRole.CAPTURE]

    @property
    def needs_copper_free(self) -> bool:
        return any(m.click_compatibility == ClickCompatibility.SPAAC_ONLY
                   for m in self.modules)


# ═══════════════════════════════════════════════════════════════════════════
# Physics models
# ═══════════════════════════════════════════════════════════════════════════

def confined_concentration_mM(
    n_molecules: int,
    cavity_diameter_nm: float,
) -> float:
    """Effective molar concentration of n molecules in a spherical cavity.

    At the single-molecule level in a 20 nm cage, one molecule gives ~mM
    effective concentration — orders of magnitude above bulk dilute solutions.

    Args:
        n_molecules: number of molecules in the cavity
        cavity_diameter_nm: cavity diameter in nm

    Returns:
        Effective concentration in mM.
    """
    if cavity_diameter_nm <= 0:
        return 0.0
    r_m = cavity_diameter_nm / 2.0 * 1e-9  # radius in meters
    volume_L = (4.0 / 3.0) * PI * r_m**3 * 1000.0  # liters
    if volume_L <= 0:
        return 0.0
    molar = n_molecules / (AVOGADRO * volume_L)
    return molar * 1e3  # mM


def confinement_enhancement(
    cavity_diameter_nm: float,
    bulk_concentration_mM: float = 0.001,  # typical dilute target
) -> float:
    """Fold enhancement of effective concentration from confinement.

    Returns ratio: confined_concentration / bulk_concentration.
    """
    confined = confined_concentration_mM(1, cavity_diameter_nm)
    if bulk_concentration_mM <= 0:
        return confined / 0.001  # assume 1 µM bulk as default
    return confined / bulk_concentration_mM


def diffusion_time_ns(
    distance_nm: float,
    diffusing_species_radius_nm: float = 0.2,  # small ion
    temperature_c: float = 25.0,
    viscosity_pa_s: float = WATER_VISCOSITY_PA_S,
) -> float:
    """Mean first-passage time for diffusion across distance in confined volume.

    Uses Einstein-Smoluchowski: <t> = d² / (6D)
    where D = kT / (6πηr) (Stokes-Einstein)

    Args:
        distance_nm: distance between modules in nm
        diffusing_species_radius_nm: hydrodynamic radius of intermediate
        temperature_c: temperature in °C
        viscosity_pa_s: solvent viscosity

    Returns:
        Diffusion time in nanoseconds.
    """
    if distance_nm <= 0:
        return 0.0

    T = temperature_c + 273.15
    r_m = diffusing_species_radius_nm * 1e-9
    d_m = distance_nm * 1e-9

    # Stokes-Einstein diffusion coefficient
    D = BOLTZMANN_J * T / (6.0 * PI * viscosity_pa_s * r_m)

    # Mean first-passage time
    t_s = d_m**2 / (6.0 * D)
    return t_s * 1e9  # convert to nanoseconds


def pore_selectivity(
    target_radius_nm: float,
    competitor_radius_nm: float,
    pore_radius_nm: float,
) -> float:
    """Selectivity factor from hindered pore transport.

    Uses Renkin equation (1954) for hindered diffusion:
      Φ = (1 - λ)² × (1 - 2.104λ + 2.089λ³ - 0.948λ⁵)
    where λ = solute_radius / pore_radius

    Returns selectivity = Φ_target / Φ_competitor.
    A value >1 means the pore favors target over competitor.
    """
    if pore_radius_nm <= 0:
        return 1.0

    def renkin(solute_r, pore_r):
        lam = solute_r / pore_r
        if lam >= 1.0:
            return 0.0  # completely excluded
        phi = (1.0 - lam)**2 * (1.0 - 2.104 * lam + 2.089 * lam**3 - 0.948 * lam**5)
        return max(0.0, phi)

    phi_target = renkin(target_radius_nm, pore_radius_nm)
    phi_competitor = renkin(competitor_radius_nm, pore_radius_nm)

    if phi_competitor <= 0:
        return float('inf') if phi_target > 0 else 1.0

    return phi_target / phi_competitor


def intermediate_retention(
    intermediate_radius_nm: float,
    pore_radius_nm: float,
    diffusion_to_transform_ns: float,
) -> float:
    """Fraction of intermediate retained in cage before it escapes through pore.

    Competes: intermediate diffusion to next module (productive) vs
    intermediate escape through pore (loss).

    Simplified model: retention ≈ 1 - Φ_pore × (t_escape / t_transform)
    where t_escape ∝ cage_diameter / (D × Φ_pore).

    Returns retention fraction (0-1). Higher = more intermediate stays inside.
    """
    phi = max(0.001, (1.0 - intermediate_radius_nm / max(0.1, pore_radius_nm))**2)

    # If intermediate is bigger than pore → perfect retention
    if intermediate_radius_nm >= pore_radius_nm:
        return 1.0

    # Retention decreases as pore permeability increases
    # and as transform time increases
    # Simple exponential model: retention = exp(-k_escape / k_transform)
    # k_escape ∝ phi, k_transform ∝ 1/t_transform
    if diffusion_to_transform_ns <= 0:
        return 1.0
    k_ratio = phi * diffusion_to_transform_ns / 100.0  # normalized
    retention = math.exp(-k_ratio)
    return min(1.0, max(0.0, retention))


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffoldProperties:
    """Properties of a scaffold system for hosting cascades."""
    system: ScaffoldSystem
    name: str
    cavity_diameter_range_nm: tuple[float, float]  # (min, max) achievable
    module_positions: tuple[int, int]               # (min, max) addressable positions
    pore_tunable: bool                              # can pore size be controlled?
    aqueous_stable: bool
    addressability: str          # "individual" (DNA), "lattice" (MOF), "ligand-class" (cage)
    orthogonal_chemistries: int  # number of orthogonal conjugation chemistries

    # Scale
    viable_scales: list[ScaleClass]
    cost_per_unit: str           # approximate

    # Limitations
    ph_range: tuple[float, float]
    temperature_max_c: float

    notes: str = ""


_SCAFFOLDS: dict[ScaffoldSystem, ScaffoldProperties] = {
    ScaffoldSystem.DNA_ORIGAMI: ScaffoldProperties(
        system=ScaffoldSystem.DNA_ORIGAMI,
        name="DNA origami wireframe (ATHENA/DAEDALUS)",
        cavity_diameter_range_nm=(10.0, 100.0),
        module_positions=(20, 200),
        pore_tunable=True,
        aqueous_stable=True,
        addressability="individual",
        orthogonal_chemistries=4,  # NHS-amine, SPAAC, thiol-maleimide, hybridization
        viable_scales=[ScaleClass.DIAGNOSTIC, ScaleClass.LAB],
        cost_per_unit="$500-2000 per batch (nmol)",
        ph_range=(5.0, 9.0),
        temperature_max_c=60.0,
        notes="Maximum positional control. Each staple independently addressable. "
              "Cost-prohibitive above lab scale.",
    ),
    ScaffoldSystem.FUJITA_CAGE: ScaffoldProperties(
        system=ScaffoldSystem.FUJITA_CAGE,
        name="Fujita Pd₆L₄ / Pd₁₂L₂₄ cage",
        cavity_diameter_range_nm=(1.0, 5.0),
        module_positions=(4, 24),
        pore_tunable=False,  # pore = face opening, fixed by topology
        aqueous_stable=True,
        addressability="ligand-class",
        orthogonal_chemistries=2,  # endohedral + exohedral on ligand
        viable_scales=[ScaleClass.LAB, ScaleClass.PILOT],
        cost_per_unit="$50-500 per mmol",
        ph_range=(3.0, 10.0),
        temperature_max_c=80.0,
        notes="Self-assembled from Pd²⁺ + pyridyl ligands. Electron-deficient cavity "
              "prefers electron-rich guests. Endohedral functionalization via ligand design.",
    ),
    ScaffoldSystem.NITSCHKE_CAGE: ScaffoldProperties(
        system=ScaffoldSystem.NITSCHKE_CAGE,
        name="Nitschke Fe₄L₆ tetrahedron",
        cavity_diameter_range_nm=(1.0, 3.0),
        module_positions=(6, 10),
        pore_tunable=False,
        aqueous_stable=True,
        addressability="ligand-class",
        orthogonal_chemistries=2,
        viable_scales=[ScaleClass.LAB, ScaleClass.PILOT],
        cost_per_unit="$100-1000 per mmol",
        ph_range=(4.0, 9.0),
        temperature_max_c=70.0,
        notes="Subcomponent self-assembly (aldehyde + amine + Fe²⁺). Modular — "
              "swap aldehyde/amine components to change cavity.",
    ),
    ScaffoldSystem.MOF_CAVITY: ScaffoldProperties(
        system=ScaffoldSystem.MOF_CAVITY,
        name="MOF cavity (UiO-66, MIL-101, etc.)",
        cavity_diameter_range_nm=(0.5, 3.0),
        module_positions=(2, 12),  # per cavity; infinite cavities in lattice
        pore_tunable=True,  # linker length, post-synthetic modification
        aqueous_stable=True,  # depends on MOF; UiO-66 and MIL-101 are stable
        addressability="lattice",
        orthogonal_chemistries=3,  # PSM, linker functionalization, defect engineering
        viable_scales=[ScaleClass.LAB, ScaleClass.PILOT, ScaleClass.FIELD, ScaleClass.INDUSTRIAL],
        cost_per_unit="$10-100 per kg",
        ph_range=(1.0, 12.0),  # UiO-66 specific
        temperature_max_c=300.0,
        notes="Every cavity is a reactor. Infinite lattice = massive parallelism. "
              "Scalable to tonnes. The bulk-scale translation of DNA origami designs.",
    ),
    ScaffoldSystem.POROUS_ORGANIC_CAGE: ScaffoldProperties(
        system=ScaffoldSystem.POROUS_ORGANIC_CAGE,
        name="Porous organic cage (POC)",
        cavity_diameter_range_nm=(1.0, 5.0),
        module_positions=(4, 12),
        pore_tunable=True,  # vertex/edge functionalization
        aqueous_stable=True,  # some are; depends on linkage chemistry
        addressability="vertex",
        orthogonal_chemistries=2,
        viable_scales=[ScaleClass.LAB, ScaleClass.PILOT],
        cost_per_unit="$200-2000 per gram",
        ph_range=(3.0, 11.0),
        temperature_max_c=150.0,
        notes="Solution-processable. Can be incorporated into membranes. "
              "Imine or boronate ester linkages.",
    ),
    ScaffoldSystem.METAL_ORGANIC_POLYHEDRON: ScaffoldProperties(
        system=ScaffoldSystem.METAL_ORGANIC_POLYHEDRON,
        name="Metal-organic polyhedron (MOP)",
        cavity_diameter_range_nm=(1.0, 5.0),
        module_positions=(4, 24),
        pore_tunable=False,
        aqueous_stable=True,  # Cu-paddlewheel MOPs are aqueous-compatible
        addressability="ligand-class",
        orthogonal_chemistries=2,
        viable_scales=[ScaleClass.LAB, ScaleClass.PILOT],
        cost_per_unit="$100-500 per mmol",
        ph_range=(3.0, 10.0),
        temperature_max_c=100.0,
        notes="Discrete, soluble analogs of MOFs. Can be embedded in membranes "
              "or drop-cast onto surfaces.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold scoring
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffoldScore:
    """Score for one scaffold system against a cascade spec."""
    scaffold: ScaffoldProperties
    spec: CascadeSpec

    # Geometric compatibility
    volume_compatible: bool = True
    positions_sufficient: bool = True
    pore_compatible: bool = True

    # Physics scores (0-1)
    confinement_factor: float = 1.0
    diffusion_time_ns: float = 0.0
    intermediate_retention: float = 1.0
    pore_selectivity: float = 1.0

    # Practical scores
    scale_match: bool = True
    cost_feasibility: float = 1.0
    addressability_score: float = 1.0

    # Composite
    composite: float = 0.0
    feasible: bool = True
    infeasibility_reason: Optional[str] = None

    # Rationale
    advantages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def score_scaffold(
    scaffold_system: ScaffoldSystem,
    spec: CascadeSpec,
    target_hydrated_radius_nm: float = 0.2,
    competitor_hydrated_radius_nm: float = 0.3,
    bulk_target_concentration_mM: float = 0.001,
    module_spacing_nm: float = 3.0,
) -> ScaffoldScore:
    """Score a scaffold system against a cascade specification."""

    scaffold = _SCAFFOLDS[scaffold_system]
    score = ScaffoldScore(scaffold=scaffold, spec=spec)

    # ── Volume compatibility ──
    min_vol = spec.min_interior_volume_nm3
    # Cavity volume from diameter range
    min_cav_vol = (4.0 / 3.0) * PI * (scaffold.cavity_diameter_range_nm[0] / 2.0)**3
    max_cav_vol = (4.0 / 3.0) * PI * (scaffold.cavity_diameter_range_nm[1] / 2.0)**3

    # Lattice systems (MOFs) have infinite copies of small cavities.
    # Volume requirement is about module positioning, not total throughput.
    # For lattice systems, check that per-cavity volume fits the MODULE footprint,
    # not the total cascade volume requirement.
    is_lattice = scaffold.addressability == "lattice"

    if not is_lattice and max_cav_vol < min_vol:
        score.volume_compatible = False
        score.feasible = False
        score.infeasibility_reason = (
            f"Max cavity volume ({max_cav_vol:.1f} nm³) < required ({min_vol:.1f} nm³)")

    # ── Position count ──
    required = spec.required_module_positions
    # Lattice systems: positions per cavity are limited, but total positions
    # across the lattice are unlimited. Check that distinct module TYPES fit
    # in one cavity (positions per cavity >= number of module types).
    if is_lattice:
        n_module_types = len(spec.modules)
        if scaffold.module_positions[1] < n_module_types:
            score.positions_sufficient = False
            score.feasible = False
            score.infeasibility_reason = (
                f"Positions per cavity ({scaffold.module_positions[1]}) "
                f"< distinct module types ({n_module_types})")
    else:
        if scaffold.module_positions[1] < required:
            score.positions_sufficient = False
            score.feasible = False
            score.infeasibility_reason = (
                f"Max positions ({scaffold.module_positions[1]}) < required ({required})")

    # ── Confinement enhancement ──
    typical_diameter = sum(scaffold.cavity_diameter_range_nm) / 2.0
    score.confinement_factor = confinement_enhancement(
        typical_diameter, bulk_target_concentration_mM)

    # ── Diffusion time ──
    score.diffusion_time_ns = diffusion_time_ns(
        module_spacing_nm, target_hydrated_radius_nm)

    # ── Pore selectivity ──
    if spec.pore_spec and scaffold.pore_tunable:
        score.pore_selectivity = pore_selectivity(
            spec.pore_spec.target_hydrated_radius_nm,
            spec.pore_spec.competitor_hydrated_radius_nm,
            spec.pore_spec.pore_diameter_nm / 2.0,
        )
        score.pore_compatible = score.pore_selectivity > 1.0
    elif spec.pore_spec and not scaffold.pore_tunable:
        score.pore_compatible = True  # can't tune but still admits target
        score.pore_selectivity = 1.0  # no enhancement from pore

    # ── Intermediate retention ──
    pore_r_nm = typical_diameter / 4.0  # rough estimate of pore radius
    intermediate_r_nm = 0.15  # typical small ion intermediate
    score.intermediate_retention = intermediate_retention(
        intermediate_r_nm, pore_r_nm, score.diffusion_time_ns)

    # ── Scale match ──
    score.scale_match = spec.target_scale in scaffold.viable_scales

    # ── Addressability ──
    if scaffold.addressability == "individual":
        score.addressability_score = 1.0
    elif scaffold.addressability == "vertex":
        score.addressability_score = 0.7
    elif scaffold.addressability == "ligand-class":
        score.addressability_score = 0.5
    elif scaffold.addressability == "lattice":
        score.addressability_score = 0.3
    else:
        score.addressability_score = 0.3

    # ── Cost feasibility ──
    if spec.target_scale in (ScaleClass.FIELD, ScaleClass.INDUSTRIAL):
        if scaffold_system == ScaffoldSystem.DNA_ORIGAMI:
            score.cost_feasibility = 0.05  # prohibitive at scale
        elif scaffold_system == ScaffoldSystem.MOF_CAVITY:
            score.cost_feasibility = 0.95  # MOFs scale to tonnes
        else:
            score.cost_feasibility = 0.4
    else:
        score.cost_feasibility = 0.8  # all are feasible at lab/pilot

    # ── Advantages / limitations ──
    if scaffold.addressability == "individual":
        score.advantages.append("Each module position independently addressable")
    if scaffold_system == ScaffoldSystem.MOF_CAVITY:
        score.advantages.append("Every cavity is a reactor — massive parallelism")
        score.advantages.append("Scalable to tonnes")
    if score.confinement_factor > 100:
        score.advantages.append(
            f"Confinement enhancement: {score.confinement_factor:.0f}×")
    if score.pore_selectivity > 2.0:
        score.advantages.append(
            f"Pore selectivity: {score.pore_selectivity:.1f}× over competitor")
    if not score.scale_match:
        score.limitations.append(
            f"Not viable at {spec.target_scale.value} scale")
    if score.cost_feasibility < 0.2:
        score.limitations.append("Cost-prohibitive at target scale")
    if scaffold.orthogonal_chemistries < len(set(m.role for m in spec.modules)):
        score.limitations.append(
            f"Only {scaffold.orthogonal_chemistries} orthogonal chemistries "
            f"but {len(set(m.role for m in spec.modules))} distinct module roles")

    # ── Composite ──
    if not score.feasible:
        score.composite = 0.0
    else:
        composite = 0.0
        # Confinement (25%) — log-scale, cap at 10000×
        conf_score = min(1.0, math.log10(max(1.0, score.confinement_factor)) / 4.0)
        composite += 0.25 * conf_score
        # Scale match (30%) — DOMINANT for field/industrial targets
        if score.scale_match:
            composite += 0.30
        else:
            # Harsh penalty: 0.05 if wrong scale (almost eliminates candidate)
            composite += 0.05
        # Addressability (15%)
        composite += 0.15 * score.addressability_score
        # Cost (15%)
        composite += 0.15 * score.cost_feasibility
        # Pore selectivity (10%)
        pore_score = min(1.0, max(0.0, (score.pore_selectivity - 1.0) / 5.0))
        composite += 0.10 * pore_score
        # Intermediate retention (5%)
        composite += 0.05 * score.intermediate_retention

        score.composite = round(composite, 4)

    return score


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold selection
# ═══════════════════════════════════════════════════════════════════════════

def rank_scaffolds(
    spec: CascadeSpec,
    target_hydrated_radius_nm: float = 0.2,
    competitor_hydrated_radius_nm: float = 0.3,
    bulk_concentration_mM: float = 0.001,
) -> list[ScaffoldScore]:
    """Rank all scaffold systems for a cascade specification.

    Returns list sorted by composite score (descending).
    Infeasible scaffolds included at end with composite = 0.
    """
    scores = []
    for system in ScaffoldSystem:
        score = score_scaffold(
            scaffold_system=system,
            spec=spec,
            target_hydrated_radius_nm=target_hydrated_radius_nm,
            competitor_hydrated_radius_nm=competitor_hydrated_radius_nm,
            bulk_target_concentration_mM=bulk_concentration_mM,
        )
        scores.append(score)

    scores.sort(key=lambda s: s.composite, reverse=True)
    return scores


def recommend_scaffold(
    spec: CascadeSpec,
    **kwargs,
) -> tuple[ScaffoldScore, str]:
    """Recommend best scaffold with rationale.

    Returns (best_score, rationale_string).
    """
    ranked = rank_scaffolds(spec, **kwargs)
    feasible = [s for s in ranked if s.feasible]

    if not feasible:
        return ranked[0], "No scaffold is feasible for this cascade specification."

    best = feasible[0]
    rationale_parts = [f"Recommended: {best.scaffold.name}"]
    if best.advantages:
        rationale_parts.append(f"Advantages: {'; '.join(best.advantages)}")
    if best.limitations:
        rationale_parts.append(f"Limitations: {'; '.join(best.limitations)}")

    # Scale-specific recommendation
    if spec.target_scale in (ScaleClass.FIELD, ScaleClass.INDUSTRIAL):
        mof = next((s for s in feasible if s.scaffold.system == ScaffoldSystem.MOF_CAVITY), None)
        if mof and mof.composite > 0:
            rationale_parts.append(
                f"At {spec.target_scale.value} scale, MOF cavity "
                f"(composite: {mof.composite:.3f}) is strongly recommended "
                f"for cost and scalability.")

    if spec.target_scale == ScaleClass.DIAGNOSTIC:
        dna = next((s for s in feasible if s.scaffold.system == ScaffoldSystem.DNA_ORIGAMI), None)
        if dna and dna.composite > 0:
            rationale_parts.append(
                "At diagnostic scale, DNA origami provides maximum positional control "
                "for proof-of-concept and geometry optimization.")

    return best, " | ".join(rationale_parts)


# ═══════════════════════════════════════════════════════════════════════════
# Module stoichiometry optimization
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StoichiometryResult:
    """Result of module stoichiometry optimization."""
    optimized_ratios: dict[str, int]  # module_id → count
    total_modules: int
    limiting_module: str              # which module is the bottleneck
    capacity_per_scaffold: int        # how many target molecules per scaffold unit
    notes: str = ""


def optimize_stoichiometry(
    spec: CascadeSpec,
    available_positions: int,
) -> StoichiometryResult:
    """Optimize module counts to maximize throughput within position budget.

    Strategy: start from stoichiometric ratios in spec, then allocate
    remaining positions to the limiting module.
    """
    if not spec.modules:
        return StoichiometryResult({}, 0, "none", 0)

    # Start with minimum 1 of each
    counts = {m.module_id: max(1, int(m.stoichiometric_ratio)) for m in spec.modules}
    total = sum(counts.values())

    if total > available_positions:
        # Can't fit even the minimum — scale down proportionally
        scale = available_positions / total
        counts = {mid: max(1, int(c * scale)) for mid, c in counts.items()}
        total = sum(counts.values())

    # Allocate remaining positions to capture modules (throughput bottleneck)
    remaining = available_positions - total
    capture_ids = [m.module_id for m in spec.modules if m.role == ModuleRole.CAPTURE]
    if capture_ids and remaining > 0:
        per_capture = remaining // len(capture_ids)
        for cid in capture_ids:
            counts[cid] += per_capture
            remaining -= per_capture

    # Any leftover → first co-reactant module (second bottleneck)
    coreactant_ids = [m.module_id for m in spec.modules if m.role == ModuleRole.CO_REACTANT]
    if coreactant_ids and remaining > 0:
        counts[coreactant_ids[0]] += remaining

    # Identify limiting module
    # Limiting = the module whose count, divided by its stoichiometric ratio,
    # gives the lowest value
    ratios = {}
    for m in spec.modules:
        effective = counts[m.module_id] / max(0.1, m.stoichiometric_ratio)
        ratios[m.module_id] = effective
    limiting = min(ratios, key=ratios.get)

    capacity = int(ratios[limiting])  # capacity is limited by the bottleneck

    return StoichiometryResult(
        optimized_ratios=counts,
        total_modules=sum(counts.values()),
        limiting_module=limiting,
        capacity_per_scaffold=capacity,
        notes=f"Limiting module: {limiting} (effective ratio: {ratios[limiting]:.1f})",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cascade pattern library
# ═══════════════════════════════════════════════════════════════════════════

def build_cascade_spec_pattern1_co2(
    matrix_has_calcium: bool = True,
) -> CascadeSpec:
    """Pattern 1: CO₂ → Catalytic Activation → Mineralization.

    Zn-CA mimic + Ca²⁺ chelate + nucleation template → CaCO₃.
    """
    from core.transform_enumerator import enumerate_transformations
    products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0} if matrix_has_calcium else {})
    calcite = next((p for p in products if "calcite" in p.name.lower()), products[0])

    return CascadeSpec(
        name="CO₂ mineralization cascade (Pattern 1)",
        description="Zn-CA mimic captures and hydrates CO₂. Ca²⁺ chelate releases Ca²⁺. "
                    "Nucleation template seeds CaCO₃ crystal growth.",
        cascade_pattern="Pattern 1",
        target_formula="CO₂",
        product=calcite,
        modules=[
            CascadeModule(
                "zn_ca", "Zn-CA mimic", ModuleRole.CAPTURE,
                "Zn²⁺-cyclen", "Catalyzes CO₂ → HCO₃⁻ hydration",
                click_compatibility=ClickCompatibility.SPAAC_ONLY,
                stoichiometric_ratio=3.0,  # 3 per face
                estimated_cost_per_module_usd=50.0,
            ),
            CascadeModule(
                "ca_chelate", "Ca²⁺-EDTA chelate", ModuleRole.CO_REACTANT,
                "EDTA-Ca", "Releases Ca²⁺ as HCO₃⁻ accumulates (Le Chatelier)",
                click_compatibility=ClickCompatibility.CUAAC_OK,
                stoichiometric_ratio=2.0,
                estimated_cost_per_module_usd=5.0,
            ),
            CascadeModule(
                "nucleation", "Carboxylate nucleation template", ModuleRole.NUCLEATION,
                "-COOH cluster", "Seeds CaCO₃ crystal growth",
                click_compatibility=ClickCompatibility.CUAAC_OK,
                stoichiometric_ratio=1.0,
                estimated_cost_per_module_usd=2.0,
            ),
        ],
        pore_spec=PoreSpec(
            target_hydrated_radius_nm=0.17,  # CO₂ kinetic diameter 3.3 Å
            competitor_hydrated_radius_nm=0.20,
            pore_diameter_nm=2.0,
        ),
        min_interior_volume_nm3=50.0,
    )


def build_cascade_spec_pattern2_struvite() -> CascadeSpec:
    """Pattern 2: Dual-target co-capture → Struvite.

    Zr-PO₄ capture + sulfonic acid NH₄⁺ capture + Mg²⁺ source → struvite.
    """
    from core.transform_enumerator import enumerate_transformations
    products = enumerate_transformations("PO4_3-", matrix_species={"NH4+": 5.0, "Mg2+": 2.0})
    struvite = next((p for p in products if "struvite" in p.name.lower()), products[0])

    return CascadeSpec(
        name="Struvite co-capture cascade (Pattern 2)",
        description="Zr(IV) captures PO₄³⁻. Sulfonic acid captures NH₄⁺. "
                    "Pre-loaded Mg²⁺ crystallizes struvite.",
        cascade_pattern="Pattern 2",
        target_formula="PO₄³⁻ + NH₄⁺",
        product=struvite,
        modules=[
            CascadeModule(
                "zr_po4", "Zr(IV) oxide cluster", ModuleRole.CAPTURE,
                "ZrO₂·nH₂O", "Selective phosphate adsorption (discriminates SO₄²⁻)",
                click_compatibility=ClickCompatibility.CUAAC_OK,
                stoichiometric_ratio=3.0,
            ),
            CascadeModule(
                "acid_nh4", "Sulfonic acid group", ModuleRole.CAPTURE,
                "-SO₃H", "NH₄⁺ capture via acid-base proton transfer",
                click_compatibility=ClickCompatibility.CUAAC_OK,
                stoichiometric_ratio=3.0,
            ),
            CascadeModule(
                "mg_source", "Mg²⁺ chelate", ModuleRole.CO_REACTANT,
                "EDTA-Mg", "Releases Mg²⁺ as PO₄³⁻ and NH₄⁺ accumulate",
                click_compatibility=ClickCompatibility.CUAAC_OK,
                stoichiometric_ratio=2.0,
            ),
        ],
        pore_spec=PoreSpec(
            target_hydrated_radius_nm=0.34,   # HPO₄²⁻
            competitor_hydrated_radius_nm=0.38, # SO₄²⁻
            pore_diameter_nm=2.0,
        ),
        min_interior_volume_nm3=20.0,  # modules only; crystal grows on/around scaffold
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public API — Reporting
# ═══════════════════════════════════════════════════════════════════════════

def cascade_report(spec: CascadeSpec, scores: list[ScaffoldScore]) -> str:
    """Human-readable cascade scaffold ranking report."""
    lines = [
        f"Cascade: {spec.name}",
        f"Pattern: {spec.cascade_pattern}",
        f"Modules: {spec.total_modules} ({len(spec.modules)} types)",
        f"Target scale: {spec.target_scale.value}",
        f"",
        f"Scaffold Rankings:",
        f"{'=' * 60}",
    ]

    for i, s in enumerate(scores, 1):
        status = "FEASIBLE" if s.feasible else "INFEASIBLE"
        lines.append(f"  {i}. [{status}] {s.scaffold.name}")
        lines.append(f"     Composite: {s.composite:.4f}")
        lines.append(f"     Confinement: {s.confinement_factor:.0f}×")
        lines.append(f"     Diffusion: {s.diffusion_time_ns:.1f} ns between modules")
        lines.append(f"     Retention: {s.intermediate_retention:.2f}")
        if s.pore_selectivity != 1.0:
            lines.append(f"     Pore selectivity: {s.pore_selectivity:.2f}×")
        if s.advantages:
            for a in s.advantages:
                lines.append(f"     + {a}")
        if s.limitations:
            for l in s.limitations:
                lines.append(f"     - {l}")
        if s.infeasibility_reason:
            lines.append(f"     REASON: {s.infeasibility_reason}")
        lines.append("")

    return "\n".join(lines)
