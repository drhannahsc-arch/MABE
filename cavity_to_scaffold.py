"""
cavity_to_scaffold.py — Binding Pocket → DNA Scaffold Translator

Bridges the anion_receptor_scorer (which outputs abstract CavityGeometry)
to the ATHENA/NanoCage interface (which needs 3D attachment point coordinates
on interior staple termini).

═══════════════════════════════════════════════════════════════════════════
CRITICAL SCALE HIERARCHY
═══════════════════════════════════════════════════════════════════════════

The binding pocket operates at TWO scales:

  SCALE 1 — Ångström (0.2–0.5 nm): The receptor pocket itself
    Built INTO a molecular module (small molecule or short peptide).
    Example: Zr-phosphonate node with 4 pendant urea H-bond donors.
    The cone cavity, donor geometry, metal coordination — all internal
    to the module. This is organic/inorganic chemistry, not DNA design.

  SCALE 2 — Nanometer (1–20 nm): Scaffold-enforced arrangement
    DNA origami positions modules at defined coordinates inside the cage.
    Each interior staple terminus carries one functional module.
    The scaffold provides:
      - Co-orientation of multiple modules (convergent binding zone)
      - Inter-module spacing (avidity, cooperative capture)
      - Structural rigidity (maintains geometry in flow)
      - Multivalency (N pockets per cage = capacity)
      - Recoverability (cage-level magnetic tags, filtration)

  Analogy:
    MODULE = amino acid side chain with its own chemistry
    SCAFFOLD = protein fold that positions side chains to form the active site
    CAGE = the protein — folds, captures, gets recycled

DNA origami staple spacing constraints:
  - Along one helix: 10.5 bp/turn = 3.57 nm per helical turn
  - Adjacent parallel helices (DX): ~2.0-2.5 nm center-to-center
  - Staple crossover spacing: ~1.5-2.0 nm
  - Practical minimum inter-attachment distance: ~2 nm
  - Interior overhang extensions: 0-15 nm (0-~45 nt ssDNA)
  - Overhang tip positional uncertainty: ±1-2 nm (ssDNA is floppy)
    → Can be rigidified with dsDNA extensions (±0.3 nm) at cost of length

═══════════════════════════════════════════════════════════════════════════
CALIBRATION STATUS: DESIGN FRAMEWORK — No experimental validation yet.
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Import from sibling module
try:
    from anion_receptor_scorer import (
        CavityGeometry, OxyanionGeometry, SELENITE, CARBONATE, SULFATE, SELENATE,
        selenite_optimal_cavity, OXYANION_DB
    )
except ImportError:
    pass  # Allow standalone testing


# ═══════════════════════════════════════════════════════════════════════════
# SCALE 1: MOLECULAR MODULES (Å-level pocket geometry)
# ═══════════════════════════════════════════════════════════════════════════

class ModuleType(str, Enum):
    """Molecular module types that provide Å-level binding geometry."""
    ZR_PHOSPHONATE_UREA = "Zr_phosphonate_urea"   # Zr⁴⁺ node + urea H-bond donors
    FE_CATECHOLATE       = "Fe_catecholate"         # Fe³⁺ with catechol coordination
    LA_DOTA_VARIANT      = "La_DOTA_variant"        # Lanthanide macrocyclic w/ pendant arms
    UREA_TRIPOD          = "urea_tripod"            # Pure H-bond receptor (no metal)
    ZR_MOF_NODE          = "Zr_MOF_node"            # Zr₆O₄(OH)₄ cluster analog
    CUSTOM               = "custom"


@dataclass(frozen=True)
class MolecularModule:
    """
    A small molecule/cluster that provides the Å-level binding pocket.

    The module itself IS the receptor. The DNA scaffold just holds it
    in place and co-orients multiple copies.
    """
    name: str
    module_type: ModuleType
    # Internal pocket geometry (what the module provides)
    pocket_aperture_A: float     # Å — effective opening diameter
    pocket_depth_A: float        # Å — cavity depth
    pocket_cone_angle_deg: float # degrees — shape selectivity
    n_hbond_donors: int          # H-bond donors pointing into pocket
    metal_center: Optional[str]  # Metal ion at pocket base (or None)
    # Physical properties
    molecular_weight: float      # Da
    diameter_nm: float           # overall module diameter (nm)
    linker_attachment: str       # How it connects to DNA staple overhang
    linker_length_nm: float      # nm — distance from DNA to active pocket
    # Performance (from anion_receptor_scorer or literature)
    target_anion: str            # Primary target
    logK_target: float           # Predicted/measured log K for target
    selectivity_vs_carbonate: float  # Δlog K
    selectivity_vs_sulfate: float    # Δlog K
    # Synthesis
    availability: str            # 'commercial', 'literature_synthesis', 'novel_design'
    reference: str               # DOI or description


# ═══════════════════════════════════════════════════════════════════════════
# MODULE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

# Module 1: Zr-phosphonate node with urea arms
# Inspired by NU-1000 Zr₆ node, but as a discrete molecular cluster
# attached to DNA via phosphonate-thiol or NHS-amine linker
ZR_UREA_MODULE = MolecularModule(
    name="Zr-phosphonate-tetraurea",
    module_type=ModuleType.ZR_PHOSPHONATE_UREA,
    pocket_aperture_A=4.0,       # matches SeO₃ O-O + vdW
    pocket_depth_A=2.2,          # pyramid height + vdW
    pocket_cone_angle_deg=38.0,  # from anion_receptor_scorer optimization
    n_hbond_donors=4,
    metal_center='Zr4+',
    molecular_weight=850.0,      # estimated for Zr(phosphonate)₂(urea-arm)₄
    diameter_nm=1.5,
    linker_attachment='NHS_amine',  # to amine-modified staple overhang
    linker_length_nm=1.0,
    target_anion='SeO3',
    logK_target=12.0,            # Zr⁴⁺-selenite from scorer
    selectivity_vs_carbonate=15.6,  # from anion_receptor_scorer
    selectivity_vs_sulfate=8.7,
    availability='novel_design',
    reference='Designed from NU-1000 Zr6 node chemistry + urea anion receptors',
)

# Module 2: Urea tripod (no metal — pure H-bond receptor)
# Based on Gale/Sessler anion receptor chemistry
# Three urea arms converging to a central cavity
UREA_TRIPOD_MODULE = MolecularModule(
    name="tris(urea)-tripod",
    module_type=ModuleType.UREA_TRIPOD,
    pocket_aperture_A=3.8,
    pocket_depth_A=2.0,
    pocket_cone_angle_deg=35.0,
    n_hbond_donors=6,            # 3 urea × 2 NH each
    metal_center=None,
    molecular_weight=520.0,
    diameter_nm=1.2,
    linker_attachment='NHS_amine',
    linker_length_nm=0.8,
    target_anion='SeO3',
    logK_target=4.5,             # H-bond only, weaker than metal-assisted
    selectivity_vs_carbonate=6.0, # geometry provides all selectivity
    selectivity_vs_sulfate=2.0,   # poor — no metal center advantage
    availability='literature_synthesis',
    reference='Gale et al. Coord Chem Rev 2003; Sessler anion binding 2006',
)

# Module 3: Fe-catecholate (for comparison — flat surface, no selectivity)
FE_CATECHOL_MODULE = MolecularModule(
    name="Fe-catecholate-surface",
    module_type=ModuleType.FE_CATECHOLATE,
    pocket_aperture_A=5.0,
    pocket_depth_A=1.5,
    pocket_cone_angle_deg=75.0,  # flat — no geometric selectivity
    n_hbond_donors=2,
    metal_center='Fe3+',
    molecular_weight=350.0,
    diameter_nm=0.8,
    linker_attachment='catechol_direct',
    linker_length_nm=0.5,
    target_anion='SeO3',
    logK_target=10.0,            # strong binding, no selectivity
    selectivity_vs_carbonate=0.0, # Fe³⁺ binds both equally
    selectivity_vs_sulfate=6.0,
    availability='commercial',
    reference='Iron oxide surface chemistry; dopamine-PEG linkers',
)

MODULE_LIBRARY = {
    'Zr_urea': ZR_UREA_MODULE,
    'urea_tripod': UREA_TRIPOD_MODULE,
    'Fe_catechol': FE_CATECHOL_MODULE,
}


# ═══════════════════════════════════════════════════════════════════════════
# SCALE 2: DNA SCAFFOLD POSITIONING (nm-level)
# ═══════════════════════════════════════════════════════════════════════════

class CageType(str, Enum):
    TETRAHEDRON = "tetrahedron"    # 4 faces, 6 edges, 4 vertices
    OCTAHEDRON  = "octahedron"     # 8 faces, 12 edges, 6 vertices
    CUBE        = "cube"           # 6 faces, 12 edges, 8 vertices
    ICOSAHEDRON = "icosahedron"    # 20 faces, 30 edges, 12 vertices


@dataclass
class CageSpec:
    """DNA origami cage specification."""
    cage_type: CageType
    edge_length_nm: float       # nm per edge
    edge_type: str              # 'DX' (2-helix) or '6HB' (6-helix bundle)
    n_edges: int = 0
    n_faces: int = 0
    n_vertices: int = 0
    interior_volume_nm3: float = 0.0
    n_interior_staples: int = 0 # from ATHENA

    def __post_init__(self):
        topology = {
            CageType.TETRAHEDRON: (6, 4, 4),
            CageType.OCTAHEDRON:  (12, 8, 6),
            CageType.CUBE:        (12, 6, 8),
            CageType.ICOSAHEDRON: (30, 20, 12),
        }
        self.n_edges, self.n_faces, self.n_vertices = topology[self.cage_type]
        # Approximate interior volume (nm³)
        L = self.edge_length_nm
        vol_factor = {
            CageType.TETRAHEDRON: 0.1178,   # L³/(6√2)
            CageType.OCTAHEDRON:  0.4714,   # L³√2/3
            CageType.CUBE:        1.0,
            CageType.ICOSAHEDRON: 2.1817,   # (5(3+√5)/12)L³
        }
        self.interior_volume_nm3 = vol_factor[self.cage_type] * L**3

        # Estimate interior staple count
        # ~2 staples per nm of edge, half face interior
        staples_per_edge = max(1, int(L / 3.5))  # one per helical turn
        self.n_interior_staples = staples_per_edge * self.n_edges // 2


@dataclass
class AttachmentPoint3D:
    """A specific position inside the cage where a module can attach."""
    id: str
    position_nm: np.ndarray    # [x, y, z] in nm, cage-centered coords
    normal_nm: np.ndarray      # direction pointing INTO cage interior
    edge_id: int               # which edge this staple is on
    staple_index: int          # index along edge
    overhang_length_nt: int    # nucleotides of ssDNA overhang into interior
    overhang_length_nm: float  # effective reach into interior (nm)
    rigidified: bool           # dsDNA overhang (more precise, less flexible)
    positional_uncertainty_nm: float  # ±nm of tip position


@dataclass
class BindingPocketDesign:
    """
    A complete binding pocket design = module + scaffold positioning.

    Multiple modules arranged by the cage scaffold form the capture site.
    Can be single-module (each module acts independently) or multi-module
    (convergent arrangement creates cooperative recognition).
    """
    name: str
    module: MolecularModule
    arrangement: str            # 'independent', 'convergent_pair', 'convergent_triad'
    n_modules_per_pocket: int
    inter_module_spacing_nm: float  # center-to-center distance between modules
    attachment_points: List[AttachmentPoint3D]
    # Pocket-level properties (emerge from arrangement)
    effective_logK: float       # combined affinity
    effective_selectivity_CO3: float
    effective_selectivity_SO4: float
    n_pockets_per_cage: int     # how many copies fit
    capacity_ions_per_cage: int # total capture capacity


# ═══════════════════════════════════════════════════════════════════════════
# TRANSLATOR: CavityGeometry → BindingPocketDesign
# ═══════════════════════════════════════════════════════════════════════════

def select_module(cavity: CavityGeometry) -> MolecularModule:
    """
    Select best molecular module for the cavity specification.

    Matches cavity parameters to module library.
    """
    best_module = None
    best_score = -np.inf

    for name, module in MODULE_LIBRARY.items():
        score = 0.0
        # Size match
        size_diff = abs(module.pocket_aperture_A - cavity.aperture_radius * 2) / 4.0
        score -= 3.0 * size_diff**2
        # Depth match
        depth_diff = abs(module.pocket_depth_A - cavity.depth) / 2.0
        score -= 3.0 * depth_diff**2
        # Cone match
        cone_diff = abs(module.pocket_cone_angle_deg - cavity.cone_angle) / 45.0
        score -= 5.0 * cone_diff**2
        # Metal match
        if cavity.metal_center and module.metal_center:
            if cavity.metal_center == module.metal_center:
                score += 5.0  # strong bonus for matching metal
        elif cavity.metal_center and not module.metal_center:
            score -= 3.0  # penalty for missing metal
        # H-bond donor count
        donor_diff = abs(module.n_hbond_donors - cavity.n_hbond_donors) / 4.0
        score -= 2.0 * donor_diff**2

        if score > best_score:
            best_score = score
            best_module = module

    return best_module


def compute_arrangement(
    module: MolecularModule,
    cage: CageSpec,
    target: str = 'SeO3',
) -> dict:
    """
    Determine optimal module arrangement within the cage.

    Decision tree:
    1. Module has metal center + H-bond donors → independent sites work.
       Each module is a complete receptor. Maximize n_modules.
    2. Module has only H-bond donors (no metal) → convergent pairs/triads
       needed for sufficient affinity. Reduces capacity but increases
       selectivity per pocket.
    3. Module has only metal (flat surface) → independent but unselective.
       Quantity over quality.
    """
    if module.metal_center and module.n_hbond_donors >= 3:
        # Complete receptor — independent sites
        arrangement = 'independent'
        n_per_pocket = 1
        # Module spacing: just avoid steric clash
        min_spacing = module.diameter_nm + 1.0  # nm
    elif module.n_hbond_donors >= 4 and not module.metal_center:
        # H-bond only — pair with a metal module, or use convergent arrangement
        arrangement = 'convergent_pair'
        n_per_pocket = 2
        # Spacing: close enough for shared anion recognition
        min_spacing = 2.0  # nm — modules face each other
    else:
        # Metal only / weak — independent, brute force
        arrangement = 'independent'
        n_per_pocket = 1
        min_spacing = module.diameter_nm + 0.5

    # How many pockets fit in the cage?
    # Limited by interior staple count and steric constraints
    staples_available = cage.n_interior_staples
    staples_per_pocket = n_per_pocket  # each module needs one staple
    n_pockets = staples_available // staples_per_pocket

    # Further limit by volume: modules can't overlap
    # Approximate: cube root of volume / module diameter
    linear_capacity = cage.interior_volume_nm3**(1/3) / min_spacing
    volume_limit = int(linear_capacity**3)
    n_pockets = min(n_pockets, volume_limit)

    # Cooperative boost for convergent arrangements
    if arrangement == 'convergent_pair':
        coop_boost = 1.5  # log K units from avidity
    elif arrangement == 'convergent_triad':
        coop_boost = 2.5
    else:
        coop_boost = 0.0

    return {
        'arrangement': arrangement,
        'n_per_pocket': n_per_pocket,
        'min_spacing_nm': min_spacing,
        'n_pockets': n_pockets,
        'capacity_ions': n_pockets * n_per_pocket,
        'coop_boost_logK': coop_boost,
        'effective_logK': module.logK_target + coop_boost,
    }


def generate_attachment_positions(
    cage: CageSpec,
    n_positions: int,
    inward_reach_nm: float = 3.0,
    rigidified: bool = False,
) -> List[AttachmentPoint3D]:
    """
    Generate 3D attachment point coordinates inside a cage.

    Distributes positions approximately evenly across interior staple
    termini. In production, ATHENA provides exact coordinates — this
    is the geometric approximation for design-phase scoring.
    """
    points = []
    # Place points on sphere of radius = edge_length * inscribed_fraction - reach
    L = cage.edge_length_nm
    inscribed_factors = {
        CageType.TETRAHEDRON: 0.204,   # r_in/L
        CageType.OCTAHEDRON:  0.408,
        CageType.CUBE:        0.500,
        CageType.ICOSAHEDRON: 0.756,
    }
    r_inscribed = inscribed_factors[cage.cage_type] * L
    r_shell = max(1.0, r_inscribed - inward_reach_nm)

    # Fibonacci sphere for even distribution
    golden_ratio = (1 + 5**0.5) / 2
    for i in range(n_positions):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_positions)
        phi = 2 * np.pi * i / golden_ratio

        x = r_shell * np.sin(theta) * np.cos(phi)
        y = r_shell * np.sin(theta) * np.sin(phi)
        z = r_shell * np.cos(theta)

        pos = np.array([x, y, z])
        normal = -pos / np.linalg.norm(pos)  # points inward

        uncertainty = 0.3 if rigidified else 1.5  # dsDNA vs ssDNA

        points.append(AttachmentPoint3D(
            id=f"AP_{i:03d}",
            position_nm=pos,
            normal_nm=normal,
            edge_id=i % cage.n_edges,
            staple_index=i,
            overhang_length_nt=int(inward_reach_nm / 0.34) if not rigidified
                               else int(inward_reach_nm / 0.34),
            overhang_length_nm=inward_reach_nm,
            rigidified=rigidified,
            positional_uncertainty_nm=uncertainty,
        ))

    return points


def design_binding_pocket(
    cavity: CavityGeometry,
    cage: CageSpec,
    target_anion: str = 'SeO3',
    prefer_module: Optional[str] = None,
) -> BindingPocketDesign:
    """
    MAIN TRANSLATOR FUNCTION.

    CavityGeometry (from anion_receptor_scorer)
      → MolecularModule selection
        → Arrangement computation
          → 3D attachment point generation
            → BindingPocketDesign

    This is the bridge between physics scoring and structural design.
    """
    # 1. Select module
    if prefer_module and prefer_module in MODULE_LIBRARY:
        module = MODULE_LIBRARY[prefer_module]
    else:
        module = select_module(cavity)

    # 2. Compute arrangement
    arr = compute_arrangement(module, cage, target_anion)

    # 3. Generate attachment positions
    points = generate_attachment_positions(
        cage,
        n_positions=arr['capacity_ions'],
        inward_reach_nm=module.linker_length_nm + module.diameter_nm / 2,
        rigidified=True,  # prefer rigid overhangs for ion capture precision
    )

    # 4. Assign to pockets
    pocket_points = points[:arr['n_pockets'] * arr['n_per_pocket']]

    # 5. Assemble design
    design = BindingPocketDesign(
        name=f"{module.name}@{cage.cage_type.value}_{cage.edge_length_nm}nm",
        module=module,
        arrangement=arr['arrangement'],
        n_modules_per_pocket=arr['n_per_pocket'],
        inter_module_spacing_nm=arr['min_spacing_nm'],
        attachment_points=pocket_points,
        effective_logK=arr['effective_logK'],
        effective_selectivity_CO3=module.selectivity_vs_carbonate,
        effective_selectivity_SO4=module.selectivity_vs_sulfate,
        n_pockets_per_cage=arr['n_pockets'],
        capacity_ions_per_cage=arr['capacity_ions'],
    )

    return design


# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT CALCULATOR: Cage quantity for Elk Valley scale
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeploymentSpec:
    """Scale-up calculation for field deployment."""
    target_removal_ug_L: float       # Se to remove per liter
    flow_rate_L_per_day: float       # treatment volume
    pocket_design: BindingPocketDesign
    cage_spec: CageSpec

    @property
    def Se_atoms_per_L(self) -> float:
        """Se atoms to capture per liter of water."""
        Se_mol_per_L = self.target_removal_ug_L / 1e6 / 79.0  # g/mol
        return Se_mol_per_L * 6.022e23

    @property
    def cages_per_L(self) -> float:
        """Cages needed per liter, assuming 1 cycle."""
        cap = max(1, self.pocket_design.capacity_ions_per_cage)
        return self.Se_atoms_per_L / cap

    @property
    def cages_per_day(self) -> float:
        return self.cages_per_L * self.flow_rate_L_per_day

    @property
    def cage_mass_kg_per_day(self) -> float:
        """Mass of DNA origami needed per day."""
        # M13 scaffold: 7249 nt, ~2.4 MDa → ~4e-18 g per cage
        cage_mass_g = 4e-18
        return self.cages_per_day * cage_mass_g / 1000

    @property
    def module_mass_kg_per_day(self) -> float:
        """Mass of molecular modules needed per day."""
        n_modules = self.cages_per_day * self.pocket_design.capacity_ions_per_cage
        module_mass_g = self.pocket_design.module.molecular_weight / 6.022e23
        return n_modules * module_mass_g / 1000

    def summary(self) -> str:
        lines = [
            f"Deployment: {self.pocket_design.name}",
            f"  Target removal: {self.target_removal_ug_L} µg/L Se",
            f"  Flow rate: {self.flow_rate_L_per_day:.2e} L/day",
            f"  Se atoms/L: {self.Se_atoms_per_L:.2e}",
            f"  Cage capacity: {self.pocket_design.capacity_ions_per_cage} ions/cage",
            f"  Cages/L: {self.cages_per_L:.2e}",
            f"  Cages/day: {self.cages_per_day:.2e}",
            f"  DNA mass/day: {self.cage_mass_kg_per_day:.1f} kg",
            f"  Module mass/day: {self.module_mass_kg_per_day:.3f} kg",
        ]
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def self_test():
    print("Cavity-to-Scaffold Translator — Self-Test")
    print("═" * 70)
    print()

    # 1. Module selection
    print("1. Module Selection from Cavity Parameters")
    print("─" * 60)
    from anion_receptor_scorer import selenite_optimal_cavity
    cavity = selenite_optimal_cavity('Zr4+')
    module = select_module(cavity)
    print(f"  Cavity: aperture={cavity.aperture_radius*2:.1f}Å, "
          f"depth={cavity.depth:.1f}Å, cone={cavity.cone_angle:.0f}°, "
          f"metal={cavity.metal_center}")
    print(f"  → Selected module: {module.name}")
    print(f"    pocket: {module.pocket_aperture_A:.1f}Å × {module.pocket_depth_A:.1f}Å, "
          f"cone={module.pocket_cone_angle_deg:.0f}°")
    print(f"    metal: {module.metal_center}, donors: {module.n_hbond_donors}")
    print(f"    logK(SeO₃): {module.logK_target}, "
          f"ΔCO₃={module.selectivity_vs_carbonate:+.1f}, "
          f"ΔSO₄={module.selectivity_vs_sulfate:+.1f}")
    print()

    # 2. Cage comparison
    print("2. Cage Geometry Comparison")
    print("─" * 60)
    cages = {
        'Tetra 30nm DX':  CageSpec(CageType.TETRAHEDRON, 30.0, 'DX'),
        'Octa 30nm DX':   CageSpec(CageType.OCTAHEDRON, 30.0, 'DX'),
        'Octa 40nm DX':   CageSpec(CageType.OCTAHEDRON, 40.0, 'DX'),
        'Octa 50nm 6HB':  CageSpec(CageType.OCTAHEDRON, 50.0, '6HB'),
        'Icosa 30nm DX':  CageSpec(CageType.ICOSAHEDRON, 30.0, 'DX'),
    }
    print(f"{'Cage':20s} {'Vol(nm³)':>10s} {'Staples':>8s} {'Edges':>6s} {'Faces':>6s}")
    for name, cage in cages.items():
        print(f"{name:20s} {cage.interior_volume_nm3:10.0f} "
              f"{cage.n_interior_staples:8d} "
              f"{cage.n_edges:6d} {cage.n_faces:6d}")
    print()

    # 3. Full pocket design for each cage
    print("3. Binding Pocket Designs: Zr-urea module")
    print("─" * 60)
    for name, cage in cages.items():
        design = design_binding_pocket(cavity, cage, 'SeO3')
        print(f"  {name}:")
        print(f"    Module: {design.module.name}")
        print(f"    Arrangement: {design.arrangement}")
        print(f"    Pockets/cage: {design.n_pockets_per_cage}")
        print(f"    Ions/cage: {design.capacity_ions_per_cage}")
        print(f"    Effective logK: {design.effective_logK:.1f}")
        print(f"    ΔCO₃: {design.effective_selectivity_CO3:+.1f}, "
              f"ΔSO₄: {design.effective_selectivity_SO4:+.1f}")
        print()

    # 4. Elk Valley deployment scale
    print("4. Elk Valley Deployment Scale")
    print("─" * 60)
    octa_40 = CageSpec(CageType.OCTAHEDRON, 40.0, 'DX')
    design = design_binding_pocket(cavity, octa_40, 'SeO3')

    # Elk Valley: 75 µg/L → 2 µg/L, ~150 ML/day
    deploy = DeploymentSpec(
        target_removal_ug_L=73.0,  # 75-2
        flow_rate_L_per_day=150e6,  # 150 million L/day
        pocket_design=design,
        cage_spec=octa_40,
    )
    print(deploy.summary())
    print()

    # Reality check
    dna_cost_per_kg = 1e6  # very rough: $1M/kg for origami-grade DNA at scale
    daily_dna_cost = deploy.cage_mass_kg_per_day * dna_cost_per_kg
    print(f"  Estimated DNA cost: ${daily_dna_cost:.0e}/day")
    print()

    if daily_dna_cost > 1e6:
        print("  ⚠ DNA COST PROHIBITIVE at current origami production scale.")
        print("  Cage-based approach is viable for:")
        print("    - Lab-scale selenium removal (mL-L volumes)")
        print("    - High-value selective recovery (REE from tailings)")
        print("    - Diagnostic selenium speciation")
        print("    - Proof-of-concept for receptor geometry optimization")
        print()
        print("  For Elk Valley 150 ML/day treatment volume, the physics")
        print("  insights (Zr⁴⁺ center, conical cavity, geometric selectivity)")
        print("  should be implemented in BULK MATERIALS:")
        print("    - Zr-MOFs with urea-functionalized linkers (UiO-66-urea)")
        print("    - Zr-phosphonate layered materials")
        print("    - Zr-doped ion exchange resins with H-bond cavities")
        print("  MABE's role: score which bulk material geometry is optimal.")
    else:
        print("  ✓ DNA cost within industrial treatment range.")
    print()

    # 5. Module comparison in same cage
    print("5. Module Comparison (Octa 40nm DX)")
    print("─" * 60)
    for mod_name in MODULE_LIBRARY:
        design = design_binding_pocket(cavity, octa_40, 'SeO3', prefer_module=mod_name)
        print(f"  {mod_name:15s}: pockets={design.n_pockets_per_cage:4d}, "
              f"logK={design.effective_logK:5.1f}, "
              f"ΔCO₃={design.effective_selectivity_CO3:+.1f}, "
              f"ΔSO₄={design.effective_selectivity_SO4:+.1f}, "
              f"capacity={design.capacity_ions_per_cage} ions")
    print()

    # 6. Design hierarchy summary
    print("6. Design Hierarchy")
    print("─" * 60)
    print("""
  SCALE          WHAT                        WHO PROVIDES
  ─────────────────────────────────────────────────────────────
  0.2-0.5 nm     Receptor pocket geometry     Molecular module
                  (cone angle, H-bonds,        (organic synthesis)
                   metal coordination)

  1-5 nm         Module positioning           DNA scaffold
                  (inter-module spacing,        (ATHENA staple design)
                   co-orientation, rigidity)

  10-50 nm       Cage architecture            Cage geometry engine
                  (capacity, pore size,         (MABE + ATHENA)
                   recovery handles)

  µm-mm          Bead/column/membrane         Deployment substrate
                  (flow integration,            (engineering)
                   regeneration, scale-up)

  The MABE anion_receptor_scorer operates at Scale 0.
  This translator bridges Scale 0 → Scale 1 → Scale 2.
  Scale 3 is the interface_extraction module (already built).
    """)

    print("═" * 70)
    print("Translator architecture complete.")
    print("Next steps:")
    print("  1. Calibrate module logK against published Zr-MOF adsorption isotherms")
    print("  2. Integrate with ATHENA adapter for real staple coordinate generation")
    print("  3. Add convergent multi-module pocket optimization")
    print("  4. Bridge to bulk material design (MOF/resin geometry mapping)")


if __name__ == "__main__":
    self_test()