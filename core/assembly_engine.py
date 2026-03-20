"""
core/assembly_engine.py -- Self-assembly design for stacked and framework materials.

Architecture:
  MonomerSpec    -> defines a building block with typed faces
  StackingEngine -> scores face-face interactions using MABE energy terms
  AssemblyGrammar -> predicts topology from monomer valence + interaction geometry
  MaterialPropertyPredictor -> porosity, surface area, stability from topology

All energy terms reuse calibrated MABE values:
  pi-pi stacking:    -4.0 kJ/mol per contact (knowledge/hg_pi.py)
  CH-pi:             -1.9 to -2.9 kJ/mol (glycan scorer)
  H-bond:            -2.25 kJ/mol (G1 calibration)
  Coordination:      -5.7 kJ/mol per bond (from metal scorer log Ka calibration)
  Hydrophobic:       -3.5 kJ/mol per nm2 buried SASA (unified scorer)
  Covalent (click):  irreversible, scored as locked

Entry point:
  design_material(target, ...) -> MaterialDesign
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Interaction modes (reusing MABE calibrated energies)
# ---------------------------------------------------------------------------

class InteractionMode(Enum):
    """Face-face interaction physics."""
    PI_PI = "pi_pi"               # aromatic stacking
    CH_PI = "ch_pi"               # CH-pi (sugar-like)
    HBOND_NETWORK = "hbond_net"   # cooperative H-bond array (urea tape, amide)
    COORDINATION = "coordination" # metal-ligand (MOF nodes)
    HYDROPHOBIC = "hydrophobic"   # solvophobic burial
    COVALENT = "covalent"         # irreversible (click, condensation)
    VAN_DER_WAALS = "vdw"         # weak, non-specific


# Energy per contact (kJ/mol), calibrated from MABE
_INTERACTION_ENERGY: Dict[InteractionMode, float] = {
    InteractionMode.PI_PI:          -4.0,   # hg_pi.py eps_pi_stack
    InteractionMode.CH_PI:          -2.4,   # average of -1.9 and -2.9
    InteractionMode.HBOND_NETWORK:  -2.25,  # G1 EPS_HB_EFF
    InteractionMode.COORDINATION:   -5.7,   # ~1 log Ka unit per bond
    InteractionMode.HYDROPHOBIC:    -3.5,   # per nm2 buried SASA
    InteractionMode.COVALENT:       -40.0,  # effectively irreversible
    InteractionMode.VAN_DER_WAALS:  -0.5,   # kT-scale
}

# Geometric directionality: how precise the alignment must be
_DIRECTIONALITY: Dict[InteractionMode, float] = {
    InteractionMode.PI_PI:          0.8,    # strong preference for parallel/offset
    InteractionMode.CH_PI:          0.6,    # moderate directionality
    InteractionMode.HBOND_NETWORK:  0.9,    # highly directional
    InteractionMode.COORDINATION:   0.95,   # very directional (d-orbital)
    InteractionMode.HYDROPHOBIC:    0.2,    # isotropic
    InteractionMode.COVALENT:       1.0,    # locked geometry
    InteractionMode.VAN_DER_WAALS:  0.1,    # isotropic
}


# ---------------------------------------------------------------------------
# Face specification
# ---------------------------------------------------------------------------

class FaceRole(Enum):
    """What this face does in the material."""
    CAPTURE = "capture"         # binds target analyte
    STRUCTURAL = "structural"   # holds material together
    LINKER = "linker"           # connects to support / other modules
    FUNCTIONAL = "functional"   # catalytic, sensing, etc.
    INERT = "inert"             # passivated, non-interacting


@dataclass
class Face:
    """One interaction face of a monomer."""
    name: str
    role: FaceRole
    interaction: InteractionMode
    n_contacts: int = 1           # number of interaction points
    area_A2: float = 50.0         # face area in Angstrom^2
    normal_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    complementary_to: str = ""    # name of face it pairs with (if specific)
    smarts: str = ""              # SMARTS defining the face chemistry
    notes: str = ""

    @property
    def dG_per_contact(self) -> float:
        return _INTERACTION_ENERGY.get(self.interaction, -1.0)

    @property
    def dG_total(self) -> float:
        return self.n_contacts * self.dG_per_contact

    @property
    def directionality(self) -> float:
        return _DIRECTIONALITY.get(self.interaction, 0.5)


# ---------------------------------------------------------------------------
# Monomer specification
# ---------------------------------------------------------------------------

@dataclass
class MonomerSpec:
    """A self-assembling building block with typed faces."""
    name: str
    smiles: str = ""
    faces: List[Face] = field(default_factory=list)
    molecular_weight: float = 0.0
    monomer_volume_A3: float = 0.0
    symmetry: str = "C1"          # point group: C1, C2, C3, C4, D2, D3, Td, Oh
    rigidity: float = 0.5         # 0 = fully flexible, 1 = rigid

    @property
    def valence(self) -> int:
        """Number of structural faces (determines topology)."""
        return sum(1 for f in self.faces if f.role == FaceRole.STRUCTURAL)

    @property
    def n_capture_faces(self) -> int:
        return sum(1 for f in self.faces if f.role == FaceRole.CAPTURE)

    @property
    def n_linker_faces(self) -> int:
        return sum(1 for f in self.faces if f.role == FaceRole.LINKER)

    def structural_faces(self) -> List[Face]:
        return [f for f in self.faces if f.role == FaceRole.STRUCTURAL]

    def face_by_name(self, name: str) -> Optional[Face]:
        for f in self.faces:
            if f.name == name:
                return f
        return None


# ---------------------------------------------------------------------------
# Stacking engine
# ---------------------------------------------------------------------------

@dataclass
class StackingScore:
    """Energy of a face-face stacking interaction."""
    face_a: str
    face_b: str
    interaction: InteractionMode
    n_contacts: int
    dG_interaction: float    # kJ/mol per pair
    dG_desolvation: float    # desolvation cost
    dG_net: float            # net energy
    reversible: bool
    notes: str = ""


class StackingEngine:
    """Score face-face interactions using MABE calibrated energies."""

    # Desolvation cost per face area (kJ/mol per A2)
    DESOLV_PER_A2 = 0.015   # ~1.5 kJ/mol per 100 A2 face

    @staticmethod
    def score_pair(face_a: Face, face_b: Face) -> StackingScore:
        """Score the interaction between two faces."""
        # Check compatibility
        if face_a.interaction != face_b.interaction:
            # Mixed interactions: use weaker
            mode = face_a.interaction
            if abs(face_a.dG_per_contact) < abs(face_b.dG_per_contact):
                mode = face_a.interaction
            else:
                mode = face_b.interaction
        else:
            mode = face_a.interaction

        n_contacts = min(face_a.n_contacts, face_b.n_contacts)
        dG_int = n_contacts * _INTERACTION_ENERGY.get(mode, -1.0)

        # Desolvation: both faces lose solvent
        avg_area = (face_a.area_A2 + face_b.area_A2) / 2
        dG_desolv = avg_area * StackingEngine.DESOLV_PER_A2

        dG_net = dG_int + dG_desolv
        reversible = mode != InteractionMode.COVALENT

        return StackingScore(
            face_a=face_a.name, face_b=face_b.name,
            interaction=mode, n_contacts=n_contacts,
            dG_interaction=round(dG_int, 2),
            dG_desolvation=round(dG_desolv, 2),
            dG_net=round(dG_net, 2),
            reversible=reversible,
        )

    @staticmethod
    def score_monomer_assembly(monomer: MonomerSpec) -> List[StackingScore]:
        """Score all structural face-face pairs in a monomer self-assembly."""
        structural = monomer.structural_faces()
        scores = []
        for i, fa in enumerate(structural):
            for fb in structural[i + 1:]:
                scores.append(StackingEngine.score_pair(fa, fb))
        # Self-complementary faces: same face stacking with itself
        for fa in structural:
            if fa.complementary_to == fa.name or fa.complementary_to == "self":
                scores.append(StackingEngine.score_pair(fa, fa))
        return scores


# ---------------------------------------------------------------------------
# Topology prediction (assembly grammar)
# ---------------------------------------------------------------------------

class Topology(Enum):
    """Predicted material topology from monomer valence + geometry."""
    DIMER = "dimer"                # 0D, two monomers
    CHAIN_1D = "chain_1D"          # 1D tape/chain
    LADDER_1D = "ladder_1D"        # 1D double-stranded
    SHEET_2D = "sheet_2D"          # 2D layer
    HONEYCOMB_2D = "honeycomb_2D"  # 2D hexagonal
    FRAMEWORK_3D = "framework_3D"  # 3D porous (MOF-like)
    CAGE_0D = "cage_0D"            # discrete cage
    TUBE_1D = "tube_1D"            # 1D nanotube
    CAPSULE_0D = "capsule_0D"      # closed capsule


@dataclass
class TopologyPrediction:
    """Result of topology prediction."""
    topology: Topology
    dimensionality: int           # 0, 1, 2, 3
    n_monomers_per_unit: int      # how many monomers in repeating unit
    coordination_number: int       # connections per monomer
    periodicity: str              # "finite", "1D-periodic", "2D-periodic", "3D-periodic"
    confidence: str = "HIGH"
    notes: str = ""
    # Geometric parameters
    repeat_distance_A: float = 0.0
    pore_diameter_A: float = 0.0
    wall_thickness_A: float = 0.0


class AssemblyGrammar:
    """Predict topology from monomer specification."""

    # Valence -> topology mapping (primary lookup)
    # Key: (structural_valence, symmetry_class)
    _TOPOLOGY_RULES = {
        # Ditopic monomers (2 structural faces)
        (2, "linear"): Topology.CHAIN_1D,
        (2, "bent"):   Topology.CHAIN_1D,  # zigzag chain
        (2, "C2"):     Topology.CHAIN_1D,

        # Tritopic monomers (3 structural faces)
        (3, "C3"):     Topology.HONEYCOMB_2D,
        (3, "planar"): Topology.SHEET_2D,
        (3, "C1"):     Topology.SHEET_2D,

        # Tetratopic (4 structural faces)
        (4, "Td"):     Topology.FRAMEWORK_3D,
        (4, "D4h"):    Topology.SHEET_2D,
        (4, "C4"):     Topology.SHEET_2D,
        (4, "square"): Topology.SHEET_2D,

        # Hexatopic (6 structural faces)
        (6, "Oh"):     Topology.FRAMEWORK_3D,

        # Special: 1 structural face = dimer
        (1, "any"):    Topology.DIMER,
    }

    @staticmethod
    def predict(monomer: MonomerSpec) -> TopologyPrediction:
        """Predict assembly topology from monomer specification."""
        valence = monomer.valence
        symmetry = monomer.symmetry

        # Map symmetry to class
        sym_class = AssemblyGrammar._symmetry_class(symmetry, valence)

        # Primary lookup
        topo = AssemblyGrammar._TOPOLOGY_RULES.get(
            (valence, sym_class),
            AssemblyGrammar._TOPOLOGY_RULES.get(
                (valence, "C1"),
                None
            )
        )

        # Fallback rules
        if topo is None:
            if valence <= 1:
                topo = Topology.DIMER
            elif valence <= 2:
                topo = Topology.CHAIN_1D
            elif valence == 3:
                topo = Topology.SHEET_2D
            elif valence >= 4:
                topo = Topology.FRAMEWORK_3D

        # Cage detection: rigid + specific valence + highly directional + COORDINATION/COVALENT
        # Cages require SELF-COMPLEMENTARY assembly (monomer pairs with itself)
        # Monomers that connect to EXTERNAL nodes (metal_node, etc.) form periodic lattices
        structural = monomer.structural_faces()
        cage_interactions = {InteractionMode.COORDINATION, InteractionMode.COVALENT}
        has_cage_chemistry = any(f.interaction in cage_interactions for f in structural)

        # Check if faces are self-complementary vs external-node-targeting
        connects_to_external = any(
            f.complementary_to and f.complementary_to not in ("self", "", f.name)
            and not any(f2.name == f.complementary_to for f2 in monomer.faces)
            for f in structural
        )

        if (valence in (2, 3) and monomer.rigidity > 0.7
                and has_cage_chemistry
                and not connects_to_external
                and all(f.directionality > 0.7 for f in structural)):
            if valence == 2:
                topo = Topology.CAPSULE_0D
            elif valence == 3:
                topo = Topology.CAGE_0D

        # Framework override: 4+ coordination sites with high directionality → 3D
        if (valence >= 4
                and has_cage_chemistry
                and monomer.rigidity > 0.7):
            topo = Topology.FRAMEWORK_3D

        dim = {
            Topology.DIMER: 0, Topology.CAPSULE_0D: 0, Topology.CAGE_0D: 0,
            Topology.CHAIN_1D: 1, Topology.LADDER_1D: 1, Topology.TUBE_1D: 1,
            Topology.SHEET_2D: 2, Topology.HONEYCOMB_2D: 2,
            Topology.FRAMEWORK_3D: 3,
        }.get(topo, 0)

        periodicity = {0: "finite", 1: "1D-periodic", 2: "2D-periodic",
                        3: "3D-periodic"}.get(dim, "finite")

        # Geometric estimates
        repeat_dist = 0.0
        pore_diam = 0.0

        if topo in (Topology.CHAIN_1D, Topology.LADDER_1D):
            # Chain repeat ~ monomer length along stacking axis
            repeat_dist = (monomer.monomer_volume_A3 ** (1 / 3)) * 1.5
        elif topo in (Topology.SHEET_2D, Topology.HONEYCOMB_2D):
            repeat_dist = (monomer.monomer_volume_A3 ** (1 / 3)) * 2.0
            if topo == Topology.HONEYCOMB_2D:
                # Pore ~ 2 * arm_length
                pore_diam = repeat_dist * 0.8
        elif topo == Topology.FRAMEWORK_3D:
            repeat_dist = (monomer.monomer_volume_A3 ** (1 / 3)) * 2.5
            pore_diam = repeat_dist * 0.6

        confidence = "HIGH" if (valence, sym_class) in AssemblyGrammar._TOPOLOGY_RULES else "MEDIUM"

        return TopologyPrediction(
            topology=topo,
            dimensionality=dim,
            n_monomers_per_unit=max(1, valence),
            coordination_number=valence,
            periodicity=periodicity,
            confidence=confidence,
            repeat_distance_A=round(repeat_dist, 1),
            pore_diameter_A=round(pore_diam, 1),
            notes=f"valence={valence} symmetry={symmetry}",
        )

    @staticmethod
    def _symmetry_class(symmetry: str, valence: int) -> str:
        """Map point group to topology-relevant class."""
        if symmetry in ("Td", "Oh", "D3", "D4h"):
            return symmetry
        if symmetry in ("C3", "D3h", "C3v"):
            return "C3"
        if symmetry in ("C4", "D4h", "C4v", "S4"):
            return "C4"
        if symmetry in ("C2", "C2v", "D2", "D2h"):
            return "C2"
        if valence == 2:
            return "linear"
        if valence == 4:
            return "square"
        return "C1"


# ---------------------------------------------------------------------------
# Material property prediction
# ---------------------------------------------------------------------------

@dataclass
class MaterialProperties:
    """Predicted properties of a self-assembled material."""
    # Porosity
    porosity_fraction: float = 0.0    # void fraction (0-1)
    pore_diameter_A: float = 0.0
    bet_surface_area_m2g: float = 0.0  # BET surface area estimate

    # Mechanical
    interaction_density: float = 0.0   # contacts per monomer
    dG_per_monomer: float = 0.0        # kJ/mol cohesive energy
    thermal_stability: str = ""        # qualitative: low/moderate/high
    reversible: bool = True

    # Functional
    capture_density: float = 0.0       # capture faces per nm2
    functional_loading: float = 0.0    # functional groups per nm2

    notes: str = ""


class MaterialPropertyPredictor:
    """Estimate material properties from topology + monomer specification."""

    @staticmethod
    def predict(monomer: MonomerSpec, topology: TopologyPrediction,
                stacking_scores: Optional[List[StackingScore]] = None) -> MaterialProperties:

        # Cohesive energy: sum of all structural face interactions
        if stacking_scores:
            dG_per_monomer = sum(s.dG_net for s in stacking_scores)
        else:
            structural = monomer.structural_faces()
            dG_per_monomer = sum(f.dG_total for f in structural)

        # Porosity: depends on topology
        if topology.topology == Topology.FRAMEWORK_3D:
            # 3D frameworks: porosity from pore size relative to repeat unit
            if topology.repeat_distance_A > 0 and topology.pore_diameter_A > 0:
                pore_frac = (topology.pore_diameter_A / topology.repeat_distance_A) ** 2
                porosity = min(0.9, max(0.1, pore_frac))
            else:
                porosity = 0.5  # typical MOF
        elif topology.topology == Topology.HONEYCOMB_2D:
            porosity = 0.4
        elif topology.topology in (Topology.CAGE_0D, Topology.CAPSULE_0D):
            # Discrete cages: internal void
            porosity = 0.3
        elif topology.topology in (Topology.SHEET_2D,):
            porosity = 0.1  # interlayer gaps only
        else:
            porosity = 0.05

        # Surface area estimate
        # BET ~ SASA_monomer * porosity_factor * (1/MW) * Avogadro / 1e18
        # Simplified: typical MOF 500-4000 m2/g, scale by porosity
        mw = max(monomer.molecular_weight, 100.0)
        if topology.dimensionality >= 2:
            bet = porosity * 5000.0 * (200.0 / mw)  # scale inversely with MW
            bet = min(4000.0, max(50.0, bet))
        elif topology.dimensionality == 1:
            bet = porosity * 1000.0 * (200.0 / mw)
            bet = min(1000.0, max(10.0, bet))
        else:
            bet = 0.0  # discrete species, no BET

        # Thermal stability
        abs_dG = abs(dG_per_monomer)
        if any(f.interaction == InteractionMode.COVALENT for f in monomer.structural_faces()):
            stability = "high"
        elif abs_dG > 30:
            stability = "high"
        elif abs_dG > 15:
            stability = "moderate"
        else:
            stability = "low"

        reversible = all(
            f.interaction != InteractionMode.COVALENT
            for f in monomer.structural_faces()
        )

        # Capture/functional density
        unit_area_nm2 = (topology.repeat_distance_A / 10.0) ** 2 if topology.repeat_distance_A > 0 else 1.0
        capture_density = monomer.n_capture_faces / max(unit_area_nm2, 0.1)
        functional_count = sum(1 for f in monomer.faces if f.role == FaceRole.FUNCTIONAL)
        functional_density = functional_count / max(unit_area_nm2, 0.1)

        return MaterialProperties(
            porosity_fraction=round(porosity, 3),
            pore_diameter_A=topology.pore_diameter_A,
            bet_surface_area_m2g=round(bet, 0),
            interaction_density=monomer.valence,
            dG_per_monomer=round(dG_per_monomer, 2),
            thermal_stability=stability,
            reversible=reversible,
            capture_density=round(capture_density, 2),
            functional_loading=round(functional_density, 2),
        )


# ---------------------------------------------------------------------------
# Material design result
# ---------------------------------------------------------------------------

@dataclass
class MaterialDesign:
    """Complete material design output."""
    monomer: MonomerSpec
    topology: TopologyPrediction
    properties: MaterialProperties
    stacking_scores: List[StackingScore] = field(default_factory=list)
    # Design quality
    meets_target: bool = False
    target_property: str = ""
    target_value: float = 0.0
    achieved_value: float = 0.0

    @property
    def summary(self) -> str:
        lines = [
            f"Material: {self.monomer.name}",
            f"  Topology: {self.topology.topology.value} ({self.topology.dimensionality}D)",
            f"  Periodicity: {self.topology.periodicity}",
            f"  Pore: {self.topology.pore_diameter_A:.1f} A",
            f"  Porosity: {self.properties.porosity_fraction:.1%}",
            f"  BET: {self.properties.bet_surface_area_m2g:.0f} m2/g",
            f"  dG/monomer: {self.properties.dG_per_monomer:.1f} kJ/mol",
            f"  Stability: {self.properties.thermal_stability}",
            f"  Reversible: {self.properties.reversible}",
        ]
        if self.properties.capture_density > 0:
            lines.append(f"  Capture density: {self.properties.capture_density:.1f} sites/nm2")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monomer library (common building blocks)
# ---------------------------------------------------------------------------

def urea_tape_monomer(name: str = "anthracene-diurea") -> MonomerSpec:
    """Davis-type: anthracene core + 2 urea faces → 1D H-bond tape."""
    return MonomerSpec(
        name=name,
        smiles="NC(=O)Nc1ccc2cc3ccc(NC(N)=O)cc3cc2c1",
        faces=[
            Face("urea_top", FaceRole.STRUCTURAL, InteractionMode.HBOND_NETWORK,
                 n_contacts=4, area_A2=40.0, normal_vector=(0, 0, 1),
                 complementary_to="urea_bottom",
                 smarts="[NX3]C(=O)[NX3]", notes="cooperative urea H-bond"),
            Face("urea_bottom", FaceRole.STRUCTURAL, InteractionMode.HBOND_NETWORK,
                 n_contacts=4, area_A2=40.0, normal_vector=(0, 0, -1),
                 complementary_to="urea_top"),
            Face("aromatic_face", FaceRole.CAPTURE, InteractionMode.CH_PI,
                 n_contacts=3, area_A2=80.0, normal_vector=(1, 0, 0),
                 notes="anthracene CH-pi for Glc binding"),
        ],
        molecular_weight=294.3,
        monomer_volume_A3=250.0,
        symmetry="C2",
        rigidity=0.8,
    )


def tripodal_linker_monomer(name: str = "trisubst-benzene") -> MonomerSpec:
    """Tritopic linker → 2D honeycomb with metal nodes."""
    return MonomerSpec(
        name=name,
        smiles="c1cc(C(=O)O)cc(C(=O)O)c1C(=O)O",
        faces=[
            Face("carboxylate_1", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                 n_contacts=2, area_A2=20.0, complementary_to="metal_node"),
            Face("carboxylate_2", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                 n_contacts=2, area_A2=20.0, complementary_to="metal_node"),
            Face("carboxylate_3", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                 n_contacts=2, area_A2=20.0, complementary_to="metal_node"),
        ],
        molecular_weight=210.1,
        monomer_volume_A3=180.0,
        symmetry="C3",
        rigidity=0.9,
    )


def mof_paddle_wheel(name: str = "Cu-paddlewheel") -> MonomerSpec:
    """Cu2(COO)4 paddle-wheel node → 3D framework with ditopic linker."""
    return MonomerSpec(
        name=name,
        smiles="",  # coordination complex, no simple SMILES
        faces=[
            Face("equatorial_1", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                 n_contacts=2, area_A2=15.0),
            Face("equatorial_2", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                 n_contacts=2, area_A2=15.0),
            Face("equatorial_3", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                 n_contacts=2, area_A2=15.0),
            Face("equatorial_4", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                 n_contacts=2, area_A2=15.0),
            Face("axial_capture", FaceRole.CAPTURE, InteractionMode.COORDINATION,
                 n_contacts=1, area_A2=10.0, notes="open metal site for guest"),
        ],
        molecular_weight=310.0,
        monomer_volume_A3=200.0,
        symmetry="D4h",
        rigidity=1.0,
    )


def pi_stacking_monomer(name: str = "pyrene-core") -> MonomerSpec:
    """Large aromatic core → 1D columnar stack."""
    return MonomerSpec(
        name=name,
        smiles="c1cc2ccc3cccc4ccc(c1)c2c34",
        faces=[
            Face("pi_top", FaceRole.STRUCTURAL, InteractionMode.PI_PI,
                 n_contacts=2, area_A2=120.0, normal_vector=(0, 0, 1),
                 complementary_to="pi_bottom"),
            Face("pi_bottom", FaceRole.STRUCTURAL, InteractionMode.PI_PI,
                 n_contacts=2, area_A2=120.0, normal_vector=(0, 0, -1),
                 complementary_to="pi_top"),
            Face("edge_functional", FaceRole.FUNCTIONAL, InteractionMode.VAN_DER_WAALS,
                 n_contacts=1, area_A2=30.0),
        ],
        molecular_weight=202.3,
        monomer_volume_A3=190.0,
        symmetry="D2",
        rigidity=1.0,
    )


# ---------------------------------------------------------------------------
# Unified design pipeline
# ---------------------------------------------------------------------------

def design_material(
    monomer: MonomerSpec,
) -> MaterialDesign:
    """
    Full material design from monomer specification.

    Pipeline:
      1. Score all structural face-face interactions
      2. Predict topology from valence + symmetry
      3. Estimate material properties
    """
    stacking = StackingEngine.score_monomer_assembly(monomer)
    topology = AssemblyGrammar.predict(monomer)
    properties = MaterialPropertyPredictor.predict(monomer, topology, stacking)

    return MaterialDesign(
        monomer=monomer,
        topology=topology,
        properties=properties,
        stacking_scores=stacking,
    )


# ---------------------------------------------------------------------------
# Assembly demand (integrates with core/demand_generator.py)
# ---------------------------------------------------------------------------

def assembly_demand(
    target_topology: str = "framework_3D",
    target_porosity: float = 0.5,
    capture_mode: Optional[str] = None,
) -> dict:
    """
    Generate a demand specification for material assembly.

    Args:
        target_topology: desired topology ("chain_1D", "sheet_2D",
                         "honeycomb_2D", "framework_3D", "cage_0D")
        target_porosity: desired void fraction (0-1)
        capture_mode: optional capture interaction ("ch_pi", "coordination", etc.)

    Returns:
        dict with recommended monomer parameters.
    """
    # Topology -> required valence
    topo_valence = {
        "dimer": 1, "chain_1D": 2, "ladder_1D": 2,
        "sheet_2D": 3, "honeycomb_2D": 3,
        "framework_3D": 4, "cage_0D": 3, "capsule_0D": 2,
    }
    valence = topo_valence.get(target_topology, 2)

    # Topology -> recommended symmetry
    topo_symmetry = {
        "chain_1D": "C2", "honeycomb_2D": "C3",
        "sheet_2D": "C3", "framework_3D": "Td",
        "cage_0D": "C3", "capsule_0D": "C2",
    }
    symmetry = topo_symmetry.get(target_topology, "C1")

    # Higher porosity -> larger linker arms, lower density
    # BET ~ porosity * 5000 * (200/MW)
    # So: MW ~ 200 * porosity * 5000 / target_BET
    recommended_mw = 150.0 + target_porosity * 300.0  # rough

    # Rigidity needed for porous materials
    rigidity = 0.9 if target_porosity > 0.3 else 0.5

    # Interaction type for structural faces
    if target_topology == "framework_3D":
        structural_interaction = "coordination"
    elif target_topology in ("chain_1D", "capsule_0D"):
        structural_interaction = "hbond_net"
    else:
        structural_interaction = "coordination"

    return {
        "target_topology": target_topology,
        "required_valence": valence,
        "recommended_symmetry": symmetry,
        "recommended_mw_range": (recommended_mw * 0.7, recommended_mw * 1.3),
        "recommended_rigidity": rigidity,
        "structural_interaction": structural_interaction,
        "capture_mode": capture_mode,
        "target_porosity": target_porosity,
    }
