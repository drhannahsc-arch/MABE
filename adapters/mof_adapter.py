"""
adapters/mof_adapter.py — Metal-Organic Framework Adapter for MABE

Maps InteractionGeometrySpec to MOF designs:
    pocket geometry → topology selection → node chemistry → linker design → PSM

Physics basis:
    - MOF nodes provide metal coordination sites (Zr₆O₄, Cu₂, Zn₄O, etc.)
    - Linkers set pocket dimensions (ditopic, tritopic, tetratopic)
    - Topology determines cavity shape and pore connectivity
    - Post-synthetic modification (PSM) tunes donor chemistry
    - Pore size gates guest access (size selectivity for free)

Every MOF unit cell is a binding pocket → massive parallelism at bulk scale.
This is the remediation-scale answer when DNA origami is too expensive.

Does NOT:
    - Run DFT on MOF structures (flags for pymatgen/VASP refinement)
    - Use fitted parameters against MOF binding data
    - Access CoRE MOF database at runtime (uses embedded topology catalog)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import CandidateResult


# ═══════════════════════════════════════════════════════════════════════════
# MOF NODE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MOFNode:
    """A secondary building unit (SBU) — the metal cluster at MOF vertices."""
    name: str
    formula: str                  # e.g. "Zr6O4(OH)4"
    metal: str                    # primary metal
    coordination_number: int      # connections to linkers
    node_geometry: str            # "octahedral", "paddlewheel", "tetrahedral", etc.
    donor_atoms_per_site: list    # what each open site provides
    stability: str                # "high", "medium", "low"
    water_stable: bool
    acid_stable: bool
    thermal_limit_C: int
    hardness: str                 # HSAB: "hard", "borderline", "soft"
    notes: str = ""


NODE_LIBRARY = [
    MOFNode("Zr6-oxo", "Zr6O4(OH)4", "Zr", 12, "cuboctahedral",
            ["O_hydroxyl", "O_carboxylate"], "high", True, True, 500, "hard",
            "UiO-series. Most chemically robust MOF node. 12-connected."),
    MOFNode("Zr6-oxo-8", "Zr6O4(OH)4", "Zr", 8, "cubic",
            ["O_hydroxyl", "O_carboxylate"], "high", True, True, 500, "hard",
            "8-connected Zr node (NU-1000, MOF-545). Larger pores than UiO."),
    MOFNode("Cu-paddlewheel", "Cu2(COO)4", "Cu", 4, "paddlewheel",
            ["O_carboxylate"], "medium", False, False, 300, "borderline",
            "HKUST-1. Open metal site after activation. Lewis acid."),
    MOFNode("Zn4O", "Zn4O(COO)6", "Zn", 6, "octahedral",
            ["O_carboxylate"], "medium", False, False, 400, "borderline",
            "MOF-5 / IRMOF series. Not water-stable but huge diversity."),
    MOFNode("Fe3-oxo", "Fe3O(COO)6", "Fe", 6, "trigonal_prismatic",
            ["O_carboxylate", "O_hydroxyl"], "high", True, False, 350, "hard",
            "MIL-series (MIL-100, MIL-101). Very large pores. Redox active."),
    MOFNode("Al-oxo", "Al(OH)(COO)2", "Al", 6, "octahedral_chain",
            ["O_hydroxyl", "O_carboxylate"], "high", True, True, 500, "hard",
            "MIL-53(Al), CAU series. Chain SBU. Breathing behavior."),
    MOFNode("Ti-oxo", "Ti8O8(OH)4", "Ti", 8, "cubic",
            ["O_hydroxyl"], "high", True, True, 450, "hard",
            "MIL-125. Photocatalytically active. Hard Lewis acid."),
    MOFNode("Cr-oxo", "Cr3O(COO)6", "Cr", 6, "trigonal_prismatic",
            ["O_hydroxyl", "O_carboxylate"], "high", True, True, 400, "hard",
            "MIL-101(Cr). Among the most stable MOFs known."),
    MOFNode("Cu-triazolate", "Cu3(triaz)3", "Cu", 3, "triangular",
            ["N_triazole"], "medium", True, False, 250, "soft",
            "Soft-metal MOF. Thiol/thioether guests preferred."),
    MOFNode("Zn-imidazolate", "Zn(Im)2", "Zn", 4, "tetrahedral",
            ["N_imidazole"], "high", True, False, 500, "borderline",
            "ZIF series (ZIF-8, ZIF-67). Zeolitic topology. Very stable."),
]

_NODE_BY_NAME = {n.name: n for n in NODE_LIBRARY}


# ═══════════════════════════════════════════════════════════════════════════
# MOF LINKER LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MOFLinker:
    """An organic linker connecting MOF nodes."""
    name: str
    smiles: str
    connectivity: int             # ditopic=2, tritopic=3, tetratopic=4
    length_nm: float              # end-to-end distance
    functional_groups: list       # available for PSM: ["NH2", "OH", "Br", etc.]
    aromatic: bool
    notes: str = ""


LINKER_LIBRARY = [
    # Ditopic (2-connected)
    MOFLinker("BDC", "OC(=O)c1ccc(C(=O)O)cc1", 2, 0.69, [],
              True, "1,4-benzenedicarboxylate. UiO-66 linker."),
    MOFLinker("BDC-NH2", "OC(=O)c1ccc(C(=O)O)c(N)c1", 2, 0.69,
              ["NH2"], True, "2-aminoterephthalate. PSM handle on UiO-66."),
    MOFLinker("BDC-OH", "OC(=O)c1ccc(C(=O)O)c(O)c1", 2, 0.69,
              ["OH"], True, "2-hydroxyterephthalate."),
    MOFLinker("BDC-Br", "OC(=O)c1ccc(C(=O)O)c(Br)c1", 2, 0.69,
              ["Br"], True, "2-bromoterephthalate. Click-ready via Sonogashira."),
    MOFLinker("NDC", "OC(=O)c1ccc2cc(C(=O)O)ccc2c1", 2, 0.94,
              [], True, "2,6-naphthalenedicarboxylate. Longer, more hydrophobic."),
    MOFLinker("BPDC", "OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1", 2, 1.15,
              [], True, "Biphenyldicarboxylate. UiO-67 linker. Expanded pore."),
    MOFLinker("fumarate", "OC(=O)/C=C/C(=O)O", 2, 0.50,
              [], False, "Fumarate. Short linker. Dense MOF."),
    MOFLinker("oxalate", "OC(=O)C(=O)O", 2, 0.35,
              [], False, "Oxalate. Shortest dicarboxylate."),

    # Tritopic (3-connected)
    MOFLinker("BTC", "OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1", 3, 0.60,
              [], True, "Trimesate. HKUST-1 / MIL-100 linker."),
    MOFLinker("TATB", "Nc1cc(N)cc(N)c1", 3, 0.50,
              ["NH2", "NH2", "NH2"], True, "Triaminobenzene. All-amine tritopic."),

    # Tetratopic (4-connected)
    MOFLinker("TCPP", "OC(=O)c1ccc(-c2cc(-c3ccc(C(=O)O)cc3)c(-c3ccc(C(=O)O)cc3)c(-c3ccc(C(=O)O)cc3)c2)cc1",
              4, 1.60, [], True, "Tetrakis(carboxyphenyl)porphyrin. PCN-222 linker."),

    # Imidazolate (for ZIFs)
    MOFLinker("mIm", "Cc1ncc[nH]1", 2, 0.30,
              [], True, "2-methylimidazole. ZIF-8 linker."),
    MOFLinker("bIm", "c1ccc2[nH]cnc2c1", 2, 0.35,
              [], True, "Benzimidazole. ZIF-7 linker."),
]

_LINKER_BY_NAME = {l.name: l for l in LINKER_LIBRARY}


# ═══════════════════════════════════════════════════════════════════════════
# MOF TOPOLOGY PRESETS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MOFTopology:
    """A known MOF topology with computed properties."""
    name: str
    node: str                     # node name from NODE_LIBRARY
    linker: str                   # linker name from LINKER_LIBRARY
    topology_code: str            # RCSR code: "fcu", "pcu", "sod", etc.
    cavity_diameter_nm: float
    pore_aperture_nm: float       # narrowest window
    surface_area_m2_g: float      # BET surface area
    cavity_shape: str             # "octahedral", "tetrahedral", "spherical", etc.
    water_stable: bool
    exemplar: str                 # known MOF name
    notes: str = ""


MOF_TOPOLOGIES = [
    MOFTopology("UiO-66", "Zr6-oxo", "BDC", "fcu", 0.80, 0.60,
                1200, "octahedral", True, "UiO-66",
                "Benchmark MOF. Defect engineering creates open Zr sites."),
    MOFTopology("UiO-66-NH2", "Zr6-oxo", "BDC-NH2", "fcu", 0.80, 0.60,
                1050, "octahedral", True, "UiO-66-NH2",
                "Amine-functionalized. PSM: amide coupling, diazotization."),
    MOFTopology("UiO-67", "Zr6-oxo", "BPDC", "fcu", 1.20, 0.80,
                2400, "octahedral", True, "UiO-67",
                "Expanded pore UiO. Fits larger guests."),
    MOFTopology("HKUST-1", "Cu-paddlewheel", "BTC", "tbo", 0.90, 0.65,
                1800, "spherical", False, "HKUST-1",
                "Open Cu sites after activation. Strong Lewis acid."),
    MOFTopology("MIL-101-Cr", "Cr-oxo", "BDC", "mtn", 2.90, 1.20,
                3500, "spherical", True, "MIL-101(Cr)",
                "Giant pore MOF. Very stable. Industrial candidate."),
    MOFTopology("MIL-101-NH2", "Cr-oxo", "BDC-NH2", "mtn", 2.90, 1.20,
                2800, "spherical", True, "MIL-101-NH2(Cr)",
                "Amine-functionalized MIL-101. PSM-ready."),
    MOFTopology("ZIF-8", "Zn-imidazolate", "mIm", "sod", 1.16, 0.34,
                1600, "spherical", True, "ZIF-8",
                "Sodalite cage. 3.4 Å aperture gates molecular sieving."),
    MOFTopology("MOF-808", "Zr6-oxo-8", "BTC", "spn", 1.80, 1.00,
                2000, "spherical", True, "MOF-808",
                "6-connected Zr + BTC. Very open. Formate-capped sites for PSM."),
    MOFTopology("NU-1000", "Zr6-oxo-8", "TCPP", "csq", 3.10, 1.20,
                2400, "hexagonal", True, "NU-1000",
                "Mesoporous Zr-MOF. Porphyrin linkers."),
    MOFTopology("PCN-222", "Zr6-oxo-8", "TCPP", "she", 3.70, 1.50,
                2200, "hexagonal", True, "PCN-222",
                "Largest Zr-porphyrin MOF. Metalloporphyrin active sites."),

    # ── Selenite-relevant: Zr + urea functionalized ──
    MOFTopology("UiO-66-urea", "Zr6-oxo", "BDC-NH2", "fcu", 0.70, 0.50,
                900, "octahedral", True, "UiO-66-NH2 + urea PSM",
                "HYPOTHETICAL: UiO-66-NH2 post-synthetically modified with "
                "isocyanate to install urea groups. Each linker carries one urea "
                "(2 NH donors) pointing into octahedral cavity. 12-connected Zr node "
                "provides hard Lewis acid site. Combined: Zr coordination + urea H-bond "
                "cavity = selenite receptor architecture from Phase 13a. "
                "Reduced pore size from urea groups (~0.7 nm vs 0.8 nm native)."),
    MOFTopology("MOF-808-urea", "Zr6-oxo-8", "BTC", "spn", 1.60, 0.90,
                1700, "spherical", True, "MOF-808 + urea PSM",
                "HYPOTHETICAL: MOF-808 with formate caps replaced by urea-bearing "
                "modulators. 6 open Zr sites per node, each with adjacent urea donor. "
                "Larger cavity than UiO-66-urea — may suit selenate (tetrahedral) "
                "better than selenite (pyramidal). Aqueous stable."),
]

_TOPOLOGY_BY_NAME = {t.name: t for t in MOF_TOPOLOGIES}


# ═══════════════════════════════════════════════════════════════════════════
# PSM (POST-SYNTHETIC MODIFICATION) LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PSMReaction:
    """A post-synthetic modification reaction for MOF linkers."""
    name: str
    handle_required: str          # functional group on linker: "NH2", "OH", "Br", etc.
    product_group: str            # what it becomes after PSM
    provides_donor: str           # donor type it adds: "N_amide", "O_hydroxyl", etc.
    conditions: str
    notes: str = ""


PSM_LIBRARY = [
    PSMReaction("urea_formation", "NH2", "urea", "N_urea",
                "isocyanate in DMF, RT, 24h",
                "Selenite receptor model. Bidentate H-bond donor."),
    PSMReaction("amide_coupling", "NH2", "amide", "N_amide",
                "acyl chloride in DMF, 60°C, 12h",
                "General PSM. Introduces any carboxylic acid functionality."),
    PSMReaction("thiol_graft", "NH2", "thiol", "S_thiolate",
                "2-iminothiolane in MeOH, RT, 6h",
                "Adds soft-metal binding site. Hg/Pb/Cd selective."),
    PSMReaction("azide_click", "Br", "triazole", "N_triazole",
                "NaN3 then CuAAC with alkyne, RT",
                "Click chemistry PSM. Modular."),
    PSMReaction("catechol_graft", "NH2", "catechol", "O_catecholate",
                "3,4-dihydroxybenzaldehyde, MeOH, RT, 24h (Schiff base)",
                "Adds bidentate O donor. Hard metal selective."),
    PSMReaction("phosphonate_graft", "OH", "phosphonate", "O_phosphonate",
                "POCl3, then hydrolysis",
                "Strong hard-metal binder."),
    PSMReaction("sulfonamide_graft", "NH2", "sulfonamide", "N_sulfonamide",
                "sulfonyl chloride in DMF, RT",
                "H-bond donor. Quinone-complementary."),
    PSMReaction("hydroxamic_acid", "NH2", "hydroxamate", "O_hydroxamate",
                "hydroxylamine + acid chloride, DMF, RT",
                "Bidentate O,O chelator. Fe3+ selective."),
]


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MOFDesign:
    """A scored MOF design for a given pocket spec."""
    topology_name: str
    node: MOFNode
    linker: MOFLinker
    topology: MOFTopology
    psm: list = field(default_factory=list)   # list[PSMReaction] applied

    # Scores
    cavity_match_score: float = 0.0    # how well cavity matches spec
    donor_match_score: float = 0.0     # how well donors match spec
    stability_score: float = 0.0       # chemical stability
    scalability_score: float = 0.0     # industrial feasibility
    composite_score: float = 0.0

    # Predicted properties
    predicted_pore_nm: float = 0.0
    predicted_cavity_nm: float = 0.0
    guest_accessible: bool = True
    n_binding_sites_per_g: float = 0.0  # from surface area + site density

    # Synthesis
    synthesis_route: str = ""
    estimated_cost_per_kg: float = 0.0
    notes: str = ""


def design_mof_for_guest(
    spec,
    guest_volume_A3: float = 0.0,
    guest_max_dim_A: float = 20.0,
    require_water_stable: bool = True,
    application: str = "remediation",
    max_designs: int = 5,
) -> list:
    """Design MOF candidates for a given InteractionGeometrySpec.

    Args:
        spec: InteractionGeometrySpec
        guest_volume_A3: guest molecular volume
        guest_max_dim_A: guest largest dimension
        require_water_stable: filter for aqueous stability
        application: "remediation", "separation", "sensing"
        max_designs: max designs to return

    Returns:
        list[MOFDesign] sorted by composite score.
    """
    guest_vol_nm3 = guest_volume_A3 / 1000.0
    guest_d_nm = guest_max_dim_A / 10.0

    # Required donor types from spec
    spec_donors = set()
    for dp in spec.donor_positions:
        if dp.charge_state < 0:
            spec_donors.add("acceptor")
        else:
            spec_donors.add("donor")

    designs = []

    for topo in MOF_TOPOLOGIES:
        if require_water_stable and not topo.water_stable:
            continue

        node = _NODE_BY_NAME.get(topo.node)
        linker = _LINKER_BY_NAME.get(topo.linker)
        if not node or not linker:
            continue

        # Cavity size match
        cav_match = _cavity_match(topo, guest_vol_nm3, guest_d_nm)

        # Guest must fit through pore
        if topo.pore_aperture_nm < guest_d_nm * 0.5:
            accessible = False
        else:
            accessible = True

        # Donor matching: does the MOF provide what the pocket spec needs?
        # Node open metal sites → Lewis acid (acceptor for guest donors)
        # Linker functional groups + PSM → H-bond donors/acceptors
        donor_match, psm_applied = _donor_match(node, linker, spec)

        # Stability
        stab = 1.0 if node.stability == "high" else 0.6 if node.stability == "medium" else 0.3

        # Scalability: commodity linkers score higher
        scalab = 0.8 if linker.name in ("BDC", "BDC-NH2", "BTC", "fumarate", "mIm") else 0.5

        # Composite
        composite = (
            cav_match * 3.0 +
            donor_match * 4.0 +
            stab * 2.0 +
            scalab * 1.0 +
            (1.0 if accessible else -5.0)
        )

        # Binding site density: ~1 site per node, SA-weighted
        mw_est = 500 + linker.connectivity * 150  # rough formula unit MW
        sites_per_g = (6.022e23 / mw_est) if mw_est > 0 else 0
        sites_per_g_mmol = sites_per_g / 6.022e20  # mmol/g

        designs.append(MOFDesign(
            topology_name=topo.name,
            node=node,
            linker=linker,
            topology=topo,
            psm=psm_applied,
            cavity_match_score=cav_match,
            donor_match_score=donor_match,
            stability_score=stab,
            scalability_score=scalab,
            composite_score=composite,
            predicted_pore_nm=topo.pore_aperture_nm,
            predicted_cavity_nm=topo.cavity_diameter_nm,
            guest_accessible=accessible,
            n_binding_sites_per_g=sites_per_g_mmol,
            synthesis_route=f"Solvothermal: {node.formula} + {linker.name} in DMF, 120°C, 24h",
            estimated_cost_per_kg=50.0 if scalab > 0.7 else 200.0,
        ))

    designs.sort(key=lambda d: d.composite_score, reverse=True)
    return designs[:max_designs]


def _cavity_match(topo, guest_vol_nm3, guest_d_nm):
    """Score cavity size match (0-1). Optimal: guest fills 40-70% of cavity."""
    cav_vol = (4/3) * math.pi * (topo.cavity_diameter_nm / 2) ** 3
    if cav_vol < 0.001:
        return 0.0
    packing = guest_vol_nm3 / cav_vol
    # Optimal packing 0.4-0.7 (Rebek extended to MOFs)
    if 0.3 <= packing <= 0.8:
        return 1.0 - abs(packing - 0.55) / 0.55
    elif packing < 0.3:
        return 0.3  # too much empty space
    else:
        return max(0, 0.5 - (packing - 0.8))  # too tight


def _donor_match(node, linker, spec):
    """Score donor chemistry match and propose PSM if needed."""
    score = 0.0
    psm_applied = []
    n_spec_donors = len(spec.donor_positions)
    if n_spec_donors == 0:
        return 0.5, []  # no specific donor requirements

    matched = 0

    # Node provides open metal sites → Lewis acid acceptor
    # Matches spec donor positions with negative charge (pocket acceptor = guest donor receiver)
    n_acceptors_needed = sum(1 for dp in spec.donor_positions if dp.charge_state <= 0)
    if n_acceptors_needed > 0 and node.coordination_number > 4:
        matched += min(n_acceptors_needed, 2)

    # Linker functional groups for H-bond donors
    n_donors_needed = sum(1 for dp in spec.donor_positions if dp.charge_state >= 0)
    if n_donors_needed > 0:
        for fg in linker.functional_groups:
            if fg in ("NH2", "OH"):
                matched += 1
        # If native groups insufficient, check PSM
        if matched < n_spec_donors and linker.functional_groups:
            for psm in PSM_LIBRARY:
                if psm.handle_required in linker.functional_groups:
                    psm_applied.append(psm)
                    matched += 1
                    if matched >= n_spec_donors:
                        break
        # If linker has no native groups but does have PSM handles
        elif matched < n_spec_donors and not linker.functional_groups:
            pass  # no PSM possible without handles

    score = min(1.0, matched / max(n_spec_donors, 1))
    return score, psm_applied


# ═══════════════════════════════════════════════════════════════════════════
# TOOL ADAPTER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class MOFAdapter(ToolAdapter):
    """MABE adapter for metal-organic framework design."""

    @property
    def name(self) -> str:
        return "mof_adapter"

    @property
    def version(self) -> str:
        return "0.2.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                description="Design MOF pore geometry for selective molecular capture",
                target_types=["metal_ion", "anion", "small_molecule", "gas"],
                interaction_types=["metal_coordination", "h_bond", "shape_selective",
                                   "lewis_acid", "electrostatic"],
                output_types=["topology_spec", "synthesis_conditions", "pore_properties"],
            ),
            Capability(
                description="Rank MOF topologies with PSM for target binding",
                target_types=["metal_ion", "anion", "small_molecule"],
                interaction_types=["metal_coordination", "h_bond"],
                output_types=["ranked_topologies", "psm_protocol"],
            ),
        ]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        """Assess whether a MOF can help with this problem."""
        target = problem.target
        relevance = 0.0

        # MOFs excel at: metal coordination + cavity selectivity, bulk scale
        if hasattr(target, 'size') and hasattr(target.size, 'vdw_radius'):
            r_nm = target.size.vdw_radius
            if 0.05 < r_nm < 1.5:  # MOF pore range
                relevance += 0.3

        # Metal coordination available → MOF nodes provide Lewis acid sites
        if hasattr(target, 'electronic_structure'):
            if hasattr(target.electronic_structure, 'hsab_class'):
                if target.electronic_structure.hsab_class == 'hard':
                    relevance += 0.2  # Hard oxyanions → Zr/Fe/Al nodes

        # Scale: MOFs are bulk materials
        if hasattr(problem, 'constraints'):
            scale = getattr(problem.constraints, 'required_scale', 'g')
            if scale in ('kg', 'tonne'):
                relevance += 0.3  # MOFs scale to tonnes

        # Selectivity needed
        if hasattr(problem, 'exclusions') and len(problem.exclusions) > 0:
            relevance += 0.1

        can = relevance > 0.3

        return ContributionAssessment(
            can_contribute=can,
            relevance=relevance,
            what_it_would_do=(
                "Design a metal-organic framework with node chemistry matched "
                "to target coordination and pore geometry matched to target shape. "
                "MOFs provide metal Lewis acid sites + tunable organic cavities at "
                "industrial scale."
            ),
            what_part_of_problem="Selective capture via metal coordination + cavity geometry",
            estimated_compute_time="<1 second (catalog lookup + scoring)",
            limitations=[
                "Cavity size limited to topology (discrete options, not continuous)",
                "PSM conversion may be incomplete (<100% of linkers modified)",
                "Some MOFs degrade in strongly acidic or basic conditions",
                "Pore blockage possible with high-loading adsorbates",
            ],
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        """Generate MOF candidates for a MABE Problem."""
        target = problem.target

        # Extract guest dimensions
        guest_vol = 50.0  # default Å³
        guest_dim = 5.0   # default Å
        if hasattr(target, 'size'):
            if hasattr(target.size, 'vdw_radius'):
                r = target.size.vdw_radius * 10  # nm → Å
                guest_vol = (4/3) * math.pi * r**3
                guest_dim = 2 * r

        # Check aqueous
        water = True
        if hasattr(problem, 'matrix'):
            solvent = str(getattr(problem.matrix, 'solvent', 'aqueous')).lower()
            water = 'aqueous' in solvent or 'water' in solvent

        # Build a minimal spec-like object for the design function
        class _MinSpec:
            def __init__(self):
                self.donor_positions = []
        spec = _MinSpec()

        designs = design_mof_for_guest(
            spec=spec,
            guest_volume_A3=guest_vol,
            guest_max_dim_A=guest_dim,
            require_water_stable=water,
        )

        results = []
        for d in designs:
            psm_names = [p.name for p in d.psm] if d.psm else []
            results.append(CandidateResult(
                name=d.topology_name,
                description=(
                    f"MOF: {d.node.formula} + {d.linker.name} "
                    f"({d.topology.topology_code}, "
                    f"cavity {d.predicted_cavity_nm:.1f} nm)"
                    + (f" + PSM: {', '.join(psm_names)}" if psm_names else "")
                ),
                score=d.composite_score,
                details={
                    "cavity_nm": d.predicted_cavity_nm,
                    "pore_aperture_nm": d.predicted_pore_nm,
                    "water_stable": d.topology.water_stable,
                    "node_metal": d.node.metal,
                    "psm_applied": psm_names,
                    "cost_per_kg": d.estimated_cost_per_kg,
                },
            ))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MOF ADAPTER — Selenite Receptor Design Demo")
    print("=" * 70)
    print()

    # Build a minimal spec for selenite
    # SeO₃²⁻: pyramidal, ~50 Å³, max dimension ~5 Å
    # Needs: hard Lewis acid node (Zr) + H-bond donors (urea) + water stable
    class _DonorPos:
        def __init__(self, cs):
            self.charge_state = cs

    class _Spec:
        def __init__(self):
            # Selenite presents 3 anionic O donors → receptor needs 3 acceptors
            # Plus we want H-bond donors (urea NH) → 4 donor positions
            self.donor_positions = [
                _DonorPos(-1),  # O acceptor site 1
                _DonorPos(-1),  # O acceptor site 2
                _DonorPos(+1),  # H-bond donor site 1 (urea)
                _DonorPos(+1),  # H-bond donor site 2 (urea)
                _DonorPos(+1),  # H-bond donor site 3 (urea)
                _DonorPos(+1),  # H-bond donor site 4 (urea)
            ]

    spec = _Spec()

    designs = design_mof_for_guest(
        spec=spec,
        guest_volume_A3=50.0,     # selenite ~50 Å³
        guest_max_dim_A=5.0,      # ~5 Å across
        require_water_stable=True,
        application="remediation",
        max_designs=10,
    )

    print(f"Candidates for selenite receptor (aqueous, Zr node + H-bond cavity):\n")
    print(f"{'Rank':<5} {'Topology':<16} {'Cavity':>7} {'Pore':>6} "
          f"{'CavM':>5} {'DonM':>5} {'Comp':>6} {'PSM':<25}")
    print("-" * 78)

    for i, d in enumerate(designs, 1):
        psm_str = ", ".join(p.name for p in d.psm) if d.psm else "none"
        print(f"  {i:<3} {d.topology_name:<16} {d.predicted_cavity_nm:>6.2f}nm "
              f"{d.predicted_pore_nm:>5.2f}nm "
              f"{d.cavity_match_score:>5.2f} {d.donor_match_score:>5.2f} "
              f"{d.composite_score:>6.2f} {psm_str:<25}")

    print()
    if designs:
        top = designs[0]
        print(f"Top pick: {top.topology_name}")
        print(f"  Node: {top.node.formula} ({top.node.metal}, {top.node.hardness})")
        print(f"  Linker: {top.linker.name}")
        print(f"  Cavity: {top.predicted_cavity_nm} nm")
        print(f"  Water stable: {top.topology.water_stable}")
        if top.psm:
            print(f"  PSM: {[p.name for p in top.psm]}")
            for p in top.psm:
                print(f"    {p.name}: {p.handle_required} → {p.product_group} "
                      f"({p.conditions})")
        print(f"  Notes: {top.topology.notes}")
        print(f"  Synthesis: {top.synthesis_route}")
        print(f"  Est. cost: ${top.estimated_cost_per_kg}/kg")
    print()

    # Summary stats
    print(f"Libraries: {len(NODE_LIBRARY)} nodes, {len(LINKER_LIBRARY)} linkers, "
          f"{len(MOF_TOPOLOGIES)} topologies, {len(PSM_LIBRARY)} PSM reactions")