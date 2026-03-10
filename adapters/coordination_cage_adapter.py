"""
adapters/coordination_cage_adapter.py — Self-Assembled Coordination Cage Adapter

Maps InteractionGeometrySpec to Pd/Fe/Ga/Co coordination cage designs:
    pocket geometry → cage topology → metal selection → linker design

Physics basis:
    - Metal vertex (Pd²⁺, Fe²⁺, Ga³⁺, etc.) defines coordination geometry
    - Ligand angles define cage topology (M₂L₄, M₄L₆, M₆L₄, M₈L₆, M₁₂L₂₄)
    - Cavity volume from vertex count + edge length
    - Interior chemistry from ligand panel functionalization
    - Guest binding: hydrophobic encapsulation + electrostatic (charged cages)

Cage types (from Fujita, Nitschke, Clever, Ward, Crowley):
    - Pd₂L₄: lantern/barrel. 2 Pd²⁺ + 4 ditopic ligands. Small cavity.
    - M₄L₆: tetrahedron. 4 metals + 6 ditopic ligands. Medium cavity.
    - M₆L₄: octahedron. 6 metals + 4 tritopic ligands. Large cavity.
    - M₈L₆: cube. 8 metals + 6 tetratopic ligands. Large cavity.
    - M₁₂L₂₄: cuboctahedron. Fujita sphere. Very large cavity.

Does NOT:
    - Run MD on cage structures
    - Use fitted parameters against cage binding data
    - Assume specific ligand synthesis (proposes specs)
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# CAGE VERTEX METALS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CageMetal:
    """Metal vertex for self-assembled cage."""
    name: str
    formula: str
    charge: int
    preferred_geometry: str       # "square_planar", "octahedral", "tetrahedral"
    coordination_number: int      # at vertex
    lability: str                 # "inert", "labile" — determines self-assembly speed
    water_compatible: bool
    cage_charge_per_vertex: int   # charge contributed per vertex
    hardness: str
    notes: str = ""


CAGE_METALS = [
    CageMetal("Pd2+", "Pd2+", 2, "square_planar", 4, "labile", True, 2,
              "soft", "Fujita cages. Fastest self-assembly. Water-compatible."),
    CageMetal("Pt2+", "Pt2+", 2, "square_planar", 4, "inert", True, 2,
              "soft", "Kinetically inert Pd analog. More stable but slower assembly."),
    CageMetal("Fe2+", "Fe2+", 2, "octahedral", 6, "labile", True, 2,
              "borderline", "Nitschke subcomponent self-assembly. Imine + Fe²⁺."),
    CageMetal("Co2+", "Co2+", 2, "octahedral", 6, "labile", True, 2,
              "borderline", "Similar to Fe²⁺ but different redox behavior."),
    CageMetal("Ga3+", "Ga3+", 3, "octahedral", 6, "labile", True, 3,
              "hard", "Raymond cages. High charge → strong electrostatic binding."),
    CageMetal("Zn2+", "Zn2+", 2, "tetrahedral", 4, "labile", True, 2,
              "borderline", "Tetrahedral cages. Flexible geometry."),
    CageMetal("Cu2+", "Cu2+", 2, "square_planar", 4, "labile", True, 2,
              "borderline", "Jahn-Teller distorted. Less predictable geometry."),
]

_METAL_BY_NAME = {m.name: m for m in CAGE_METALS}


# ═══════════════════════════════════════════════════════════════════════════
# CAGE TOPOLOGY LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CageTopology:
    """A self-assembled cage topology."""
    name: str
    formula: str                  # e.g. "M2L4", "M4L6"
    n_metals: int
    n_ligands: int
    ligand_connectivity: int      # ditopic=2, tritopic=3, tetratopic=4
    metal_geometry: str           # required at vertex
    polyhedron: str               # "lantern", "tetrahedron", "octahedron", etc.

    # Cavity properties (for unit edge = 1 nm)
    cavity_volume_per_nm3_edge: float  # V / edge³
    pore_aperture_fraction: float      # pore_d / edge_length
    n_faces: int                       # number of open faces (for guest entry)
    cage_charge_factor: int            # total charge = n_metals × metal_charge

    # Interior chemistry
    interior_panel: str           # "aromatic" (electron-deficient), "aliphatic", "mixed"
    encapsulation_mode: str       # "hydrophobic", "electrostatic", "both"

    # Practical
    self_assembly_yield: float    # typical literature yield
    water_soluble: bool
    exemplar: str                 # known cage example
    notes: str = ""


CAGE_TOPOLOGIES = [
    CageTopology(
        "M2L4_lantern", "M2L4", 2, 4, 2, "square_planar", "lantern",
        0.4, 0.3, 4, 2, "aromatic", "hydrophobic", 0.90, True,
        "Fujita Pd2L4",
        "Smallest cage. 4+ charge. Binds neutral aromatics in water.",
    ),
    CageTopology(
        "M4L6_tetrahedron", "M4L6", 4, 6, 2, "octahedral", "tetrahedron",
        0.12, 0.4, 4, 2, "aromatic", "both", 0.85, True,
        "Nitschke Fe4L6",
        "Tetrahedral. 8+ charge. Subcomponent self-assembly.",
    ),
    CageTopology(
        "M4L6_Ga_tetrahedron", "M4L6", 4, 6, 2, "octahedral", "tetrahedron",
        0.12, 0.4, 4, 3, "aromatic", "electrostatic", 0.80, True,
        "Raymond Ga4L6",
        "12− charge (catecholate). Encapsulates cations. Enzyme-like.",
    ),
    CageTopology(
        "M6L4_octahedron", "M6L4", 6, 4, 3, "square_planar", "octahedron",
        0.47, 0.5, 8, 2, "aromatic", "hydrophobic", 0.85, True,
        "Fujita Pd6L4",
        "Large cavity. 12+ charge. Encapsulates multiple guests.",
    ),
    CageTopology(
        "M8L6_cube", "M8L6", 8, 6, 4, "square_planar", "cube",
        1.0, 0.6, 6, 2, "aromatic", "hydrophobic", 0.70, True,
        "Fujita Pd8L6 (rare)",
        "Cubic cavity. Very large.",
    ),
    CageTopology(
        "M12L24_sphere", "M12L24", 12, 24, 2, "square_planar", "cuboctahedron",
        2.5, 0.25, 8, 2, "aromatic", "hydrophobic", 0.75, True,
        "Fujita M12L24",
        "Giant sphere. 24+ charge. Encapsulates proteins.",
    ),
    CageTopology(
        "M2L3_helicate", "M2L3", 2, 3, 2, "octahedral", "trigonal_prism",
        0.08, 0.25, 3, 2, "aromatic", "both", 0.90, True,
        "Hannon Fe2L3",
        "Triple helicate. Chiral. Small cavity.",
    ),
    CageTopology(
        "M4L4_Zn_tetrahedron", "M4L4", 4, 4, 3, "tetrahedral", "tetrahedron",
        0.12, 0.4, 4, 2, "mixed", "hydrophobic", 0.80, True,
        "Clever Zn4L4",
        "Tetrahedral with tritopic ligands at faces.",
    ),
]

_TOPOLOGY_BY_NAME = {t.name: t for t in CAGE_TOPOLOGIES}


# ═══════════════════════════════════════════════════════════════════════════
# LIGAND PANEL FUNCTIONAL GROUPS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EndohedralGroup:
    """A functional group pointing into the cage interior."""
    name: str
    provides: str            # "hb_donor", "hb_acceptor", "aromatic", "hydrophobic"
    donor_type: str          # specific donor: "N_pyridine", "O_hydroxyl", etc.
    attachment: str           # where on ligand: "panel_center", "panel_edge", "vertex_adjacent"
    steric_cost: str          # "none", "minor", "major"


ENDOHEDRAL_GROUPS = [
    EndohedralGroup("inward_OH", "hb_donor", "O_hydroxyl", "panel_center", "none"),
    EndohedralGroup("inward_NH2", "hb_donor", "N_amine", "panel_center", "none"),
    EndohedralGroup("inward_urea", "hb_donor", "N_urea", "panel_center", "minor"),
    EndohedralGroup("inward_pyridyl", "hb_acceptor", "N_pyridine", "panel_edge", "none"),
    EndohedralGroup("inward_catechol", "hb_donor", "O_catecholate", "panel_center", "minor"),
    EndohedralGroup("inward_fluorine", "hydrophobic", "F", "panel_center", "none"),
    EndohedralGroup("inward_nitro", "hb_acceptor", "O_nitro", "panel_center", "minor"),
    EndohedralGroup("inward_methyl", "hydrophobic", "C", "panel_edge", "none"),
]


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CageDesign:
    """A scored coordination cage design."""
    topology: CageTopology
    metal: CageMetal
    edge_length_nm: float
    endohedral_groups: list = field(default_factory=list)  # list[EndohedralGroup]

    # Computed
    cavity_volume_nm3: float = 0.0
    cavity_volume_A3: float = 0.0
    pore_diameter_nm: float = 0.0
    total_charge: int = 0
    n_interior_donors: int = 0

    # Scores
    size_match: float = 0.0
    donor_match: float = 0.0
    stability_score: float = 0.0
    composite_score: float = 0.0

    # Practical
    assembly_conditions: str = ""
    estimated_cost_usd: float = 0.0
    notes: str = ""

    @property
    def formula(self):
        return f"{self.metal.formula}{self.topology.n_metals}L{self.topology.n_ligands}"


def design_cage_for_guest(
    spec,
    guest_volume_A3: float = 0.0,
    guest_max_dim_A: float = 20.0,
    guest_charge: int = 0,
    require_water: bool = True,
    max_designs: int = 5,
) -> list:
    """Design coordination cage candidates for a guest.

    Args:
        spec: InteractionGeometrySpec
        guest_volume_A3: guest molecular volume
        guest_max_dim_A: guest max dimension
        guest_charge: guest formal charge (for electrostatic complementarity)
        require_water: require water-soluble cage
        max_designs: max results

    Returns:
        list[CageDesign] sorted by composite score.
    """
    guest_vol_nm3 = guest_volume_A3 / 1000.0

    designs = []

    for topo in CAGE_TOPOLOGIES:
        if require_water and not topo.water_soluble:
            continue

        # Compatible metals
        compatible_metals = [
            m for m in CAGE_METALS
            if m.preferred_geometry == topo.metal_geometry
            and (not require_water or m.water_compatible)
        ]

        for metal in compatible_metals:
            # Compute edge length to fit guest
            # V_cavity = vol_factor × edge³ → edge = (V_guest / (packing × vol_factor))^(1/3)
            target_packing = 0.55  # Rebek
            if topo.cavity_volume_per_nm3_edge > 0.001:
                target_vol = guest_vol_nm3 / target_packing
                edge_nm = (target_vol / topo.cavity_volume_per_nm3_edge) ** (1/3)
            else:
                edge_nm = 1.5

            # Clamp to practical range (0.5-3.0 nm ligand span)
            edge_nm = max(0.5, min(3.0, edge_nm))

            # Actual cavity at this edge length
            cav_vol = topo.cavity_volume_per_nm3_edge * edge_nm ** 3
            pore_d = topo.pore_aperture_fraction * edge_nm

            # Size match
            if cav_vol > 0.001:
                packing = guest_vol_nm3 / cav_vol
                size_match = max(0, 1.0 - abs(packing - 0.55) / 0.55)
            else:
                size_match = 0.0

            # Pore check: guest must enter
            guest_min_d_nm = guest_max_dim_A / 10.0 * 0.4  # min cross-section ~40% of max
            if pore_d < guest_min_d_nm:
                size_match *= 0.3  # severe penalty

            # Donor match: select endohedral groups
            endo_groups = _select_endohedral(spec, topo)
            n_donors = len(endo_groups)
            n_needed = len(spec.donor_positions)
            donor_match = min(1.0, n_donors / max(n_needed, 1)) if n_needed > 0 else 0.5

            # Electrostatic complementarity
            total_charge = topo.cage_charge_factor * metal.charge * topo.n_metals
            if guest_charge != 0:
                # Opposite charges attract
                if (total_charge > 0 and guest_charge < 0) or \
                   (total_charge < 0 and guest_charge > 0):
                    donor_match += 0.3
                elif total_charge * guest_charge > 0:
                    donor_match -= 0.2  # like charges repel

            # Stability
            stab = topo.self_assembly_yield

            composite = (
                size_match * 3.0 +
                donor_match * 4.0 +
                stab * 2.0
            )

            # Assembly conditions
            if metal.formula in ("Pd2+", "Pt2+"):
                conditions = f"{metal.formula} + ligand in D₂O or DMSO, RT, 1-24h"
            elif metal.formula in ("Fe2+", "Co2+"):
                conditions = f"Aldehyde + amine + {metal.formula} subcomponent assembly, CH₃CN, RT, 24h"
            elif metal.formula == "Ga3+":
                conditions = f"Catecholate ligand + {metal.formula}, K₂CO₃, MeOH/H₂O, 70°C, 12h"
            else:
                conditions = f"{metal.formula} + ligand, MeCN, RT-60°C, 24h"

            designs.append(CageDesign(
                topology=topo,
                metal=metal,
                edge_length_nm=round(edge_nm, 2),
                endohedral_groups=endo_groups,
                cavity_volume_nm3=cav_vol,
                cavity_volume_A3=cav_vol * 1000,
                pore_diameter_nm=round(pore_d, 2),
                total_charge=total_charge,
                n_interior_donors=n_donors,
                size_match=size_match,
                donor_match=min(1.0, donor_match),
                stability_score=stab,
                composite_score=composite,
                assembly_conditions=conditions,
                estimated_cost_usd=200 + topo.n_ligands * 50,
            ))

    designs.sort(key=lambda d: d.composite_score, reverse=True)
    return designs[:max_designs]


def _select_endohedral(spec, topo):
    """Select endohedral functional groups based on pocket spec donors."""
    selected = []
    max_groups = topo.n_ligands  # one group per ligand panel max

    for dp in spec.donor_positions:
        if len(selected) >= max_groups:
            break
        if dp.charge_state < 0:
            # Pocket needs acceptor → add acceptor group
            for eg in ENDOHEDRAL_GROUPS:
                if eg.provides == "hb_acceptor" and eg not in selected:
                    selected.append(eg)
                    break
        elif dp.charge_state > 0:
            # Pocket needs donor → add donor group
            for eg in ENDOHEDRAL_GROUPS:
                if eg.provides == "hb_donor" and eg not in selected:
                    selected.append(eg)
                    break

    # Fill remaining with hydrophobic if guest has nonpolar regions
    for hs in spec.hydrophobic_surfaces[:max_groups - len(selected)]:
        for eg in ENDOHEDRAL_GROUPS:
            if eg.provides == "hydrophobic" and eg not in selected:
                selected.append(eg)
                break

    return selected
