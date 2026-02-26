"""
Material System Registry.

Every material system that can create a binding pocket registers its
capability envelope. Organized by physics class, not material origin.

Sprint R1 starters:
    Class A (Covalent Cavity):        planar_coordination_ring
    Class A (Covalent Cavity):        cyclic_encapsulant
    Class B (Periodic Lattice):       periodic_lattice_node
    Class C (Foldable Polymer):       folded_polypeptide
    Class D (Emergent Cavity):        emergent_coordination_cage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MaterialCapability:
    """What a material system can and cannot do."""

    system_id: str
    physics_class: str                   # "covalent_cavity", "periodic_lattice", etc.
    adapter_class: str                   # Layer 4 adapter reference

    # ── Geometric capability envelope ──
    min_pocket_size_nm: float
    max_pocket_size_nm: float
    achievable_symmetries: list[str]
    max_donor_count: int
    donor_types_available: list[str]
    positioning_precision_A: float       # best achievable placement accuracy
    rigidity_range: tuple[str, str]      # ("rigid", "rigid") or ("flexible", "semi-rigid")

    # ── Operating envelope ──
    pH_stability: tuple[float, float]
    thermal_stability_K: tuple[float, float]
    solvent_compatibility: list[str]

    # ── Production envelope ──
    min_practical_scale: str
    max_practical_scale: str
    cost_per_unit_range: tuple[float, float]  # $/µmol at mid-scale
    typical_synthesis_time: str

    # ── Validation calibration ──
    literature_validation_rate: float    # fraction of designs that work in lab
    literature_examples: int
    design_tools_available: list[str]

    # ── Qualitative (for rationale generation) ──
    known_strengths: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)


class MaterialRegistry:
    """Registry of all known material capabilities."""

    def __init__(self):
        self._systems: dict[str, MaterialCapability] = {}

    def register(self, cap: MaterialCapability) -> None:
        self._systems[cap.system_id] = cap

    def get(self, system_id: str) -> Optional[MaterialCapability]:
        return self._systems.get(system_id)

    def all(self) -> list[MaterialCapability]:
        return list(self._systems.values())

    def by_physics_class(self, cls: str) -> list[MaterialCapability]:
        return [c for c in self._systems.values() if c.physics_class == cls]

    def __len__(self) -> int:
        return len(self._systems)


# ─────────────────────────────────────────────
# Global registry instance
# ─────────────────────────────────────────────

MATERIAL_REGISTRY = MaterialRegistry()


# ─────────────────────────────────────────────
# Sprint R1: 5 starter entries
# ─────────────────────────────────────────────

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="planar_coordination_ring",
    physics_class="covalent_cavity",
    adapter_class="PlanarCoordinationRingAdapter",
    min_pocket_size_nm=0.2,
    max_pocket_size_nm=0.5,
    achievable_symmetries=["D4h", "C4v", "D2h", "C2v"],
    max_donor_count=6,  # 4 ring + 2 axial
    donor_types_available=["N", "O", "S"],
    positioning_precision_A=0.01,  # covalent bond geometry
    rigidity_range=("rigid", "rigid"),
    pH_stability=(1.0, 14.0),
    thermal_stability_K=(200.0, 600.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="µmol",
    max_practical_scale="mol",
    cost_per_unit_range=(10.0, 500.0),
    typical_synthesis_time="1–5 days",
    literature_validation_rate=0.85,  # porphyrin synthesis is well-established
    literature_examples=50000,
    design_tools_available=["RDKit", "Gaussian"],
    known_strengths=[
        "Highest precision (covalent bond geometry)",
        "Exceptional rigidity",
        "Well-characterized metal coordination",
        "Tunable electronics via meso/beta substituents",
    ],
    known_limitations=[
        "Fixed 4N planar geometry — limited pocket shapes",
        "Only small metal ions fit",
        "Multi-step synthesis for custom variants",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="cyclic_encapsulant",
    physics_class="covalent_cavity",
    adapter_class="CyclicEncapsulantAdapter",
    min_pocket_size_nm=0.3,
    max_pocket_size_nm=0.9,
    achievable_symmetries=["Cn", "Cnv", "Dnh", "D3h", "D6h"],
    max_donor_count=12,  # large crown ethers / cryptands
    donor_types_available=["N", "O", "S"],
    positioning_precision_A=0.05,  # ring conformational averaging
    rigidity_range=("semi-rigid", "rigid"),  # cryptands more rigid than crowns
    pH_stability=(2.0, 12.0),
    thermal_stability_K=(250.0, 500.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="µmol",
    max_practical_scale="kmol",  # crown ethers are industrially produced
    cost_per_unit_range=(1.0, 200.0),
    typical_synthesis_time="1–3 days",
    literature_validation_rate=0.80,
    literature_examples=20000,
    design_tools_available=["RDKit"],
    known_strengths=[
        "Size-selective cation binding (ring size = selectivity)",
        "Well-characterized thermodynamics (Izatt compilations)",
        "Cheap at scale (18-crown-6 is commodity chemical)",
        "HSAB-tunable via O/N/S donor substitution",
    ],
    known_limitations=[
        "Primarily cation-selective (poor for anions/neutrals)",
        "Conformational flexibility in larger rings reduces selectivity",
        "Limited pocket shape diversity (circular)",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="periodic_lattice_node",
    physics_class="periodic_lattice",
    adapter_class="PeriodicLatticeNodeAdapter",
    min_pocket_size_nm=0.3,
    max_pocket_size_nm=2.0,
    achievable_symmetries=["Oh", "Td", "D4h", "D3h", "C4v", "C3v"],
    max_donor_count=12,  # MOF nodes can have high coordination
    donor_types_available=["N", "O", "S", "P"],
    positioning_precision_A=0.1,  # lattice precision
    rigidity_range=("rigid", "semi-rigid"),
    pH_stability=(2.0, 12.0),  # varies enormously: UiO-66 → pH 1-12, HKUST-1 → pH 5-8
    thermal_stability_K=(250.0, 700.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="mmol",
    max_practical_scale="kmol",  # MOFs produced at tonne scale
    cost_per_unit_range=(5.0, 1000.0),
    typical_synthesis_time="1–7 days",
    literature_validation_rate=0.70,
    literature_examples=100000,
    design_tools_available=["pymatgen", "CSD_API", "Zeo++", "ToposPro"],
    known_strengths=[
        "Massively parallel — every unit cell is a binding site",
        "Extreme surface area (>7000 m²/g achievable)",
        "Tunable pore via linker length + topology",
        "Scalable to tonnes",
    ],
    known_limitations=[
        "All pockets identical (periodic constraint)",
        "Water stability varies (many MOFs degrade in water)",
        "Post-synthetic modification needed for fine-tuning",
        "Pore access may limit diffusion to binding sites",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="folded_polypeptide",
    physics_class="foldable_polymer",
    adapter_class="FoldedPolypeptideAdapter",
    min_pocket_size_nm=0.5,
    max_pocket_size_nm=5.0,
    achievable_symmetries=["none", "Cn", "Dn"],  # via oligomeric assembly
    max_donor_count=20,  # limited only by fold
    donor_types_available=["N", "O", "S", "Se"],  # all natural aa + selenocysteine
    positioning_precision_A=0.3,  # thermal fluctuation of side chains
    rigidity_range=("flexible", "semi-rigid"),
    pH_stability=(4.0, 10.0),
    thermal_stability_K=(277.0, 370.0),  # most proteins denature < 100°C
    solvent_compatibility=["aqueous"],  # organic solvents denature
    min_practical_scale="nmol",
    max_practical_scale="mmol",  # E. coli expression
    cost_per_unit_range=(100.0, 10000.0),
    typical_synthesis_time="1–4 weeks (expression + purification)",
    literature_validation_rate=0.40,  # RFdiffusion ~19% for small molecule, higher for PPI
    literature_examples=200000,
    design_tools_available=["RFDiffusion", "ProteinMPNN", "AlphaFold2", "Rosetta"],
    known_strengths=[
        "Any pocket shape in principle",
        "20 monomer types — high interaction element diversity",
        "Mature design tools (RFdiffusion, ProteinMPNN)",
        "Biocompatible",
    ],
    known_limitations=[
        "Must fold correctly AND present correct elements — two failure modes",
        "Thermal/pH/solvent stability constraints",
        "0.3 Å positioning precision (thermal fluctuation)",
        "Requires wet-lab validation (expression, folding, binding)",
        "Organic solvents, extreme pH, or high temp destroy fold",
    ],
))

MATERIAL_REGISTRY.register(MaterialCapability(
    system_id="emergent_coordination_cage",
    physics_class="emergent_cavity",
    adapter_class="EmergentCoordinationCageAdapter",
    min_pocket_size_nm=0.5,
    max_pocket_size_nm=5.0,
    achievable_symmetries=["Td", "Oh", "D2h", "D4h", "T"],
    max_donor_count=12,
    donor_types_available=["N", "O", "S"],
    positioning_precision_A=0.1,  # self-assembly with metal vertices
    rigidity_range=("semi-rigid", "rigid"),
    pH_stability=(3.0, 11.0),
    thermal_stability_K=(270.0, 400.0),
    solvent_compatibility=["aqueous", "organic", "mixed"],
    min_practical_scale="µmol",
    max_practical_scale="mmol",
    cost_per_unit_range=(50.0, 5000.0),
    typical_synthesis_time="1–3 days (self-assembly)",
    literature_validation_rate=0.80,  # thermodynamic self-assembly is reliable
    literature_examples=500,
    design_tools_available=["stk", "RDKit"],
    known_strengths=[
        "Self-assembly under thermodynamic control — high reliability",
        "Discrete 3D cavities with defined geometry",
        "Heteroleptic designs allow 4+ different ligands → single isomer",
        "Interior functionalization possible",
        "Nitschke subcomponent self-assembly enables dynamic control",
    ],
    known_limitations=[
        "Limited to coordination-compatible metals (Pd, Pt, Fe, Ga, etc.)",
        "Stability can be marginal (reversible bonds)",
        "Scaling beyond mmol is challenging",
        "Shape persistence upon guest removal not guaranteed",
    ],
))
