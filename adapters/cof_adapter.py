"""
adapters/cof_adapter.py — Covalent Organic Framework Adapter for MABE

Maps InteractionGeometrySpec to COF designs:
    pocket geometry → topology selection → monomer chemistry → linkage type →
    pore wall functionalization → synthesis conditions

Physics basis:
    - COFs are crystalline porous polymers built from organic monomers
    - No metal nodes (unlike MOFs) — all-covalent linkages
    - Pore geometry set by monomer geometry + topology (hexagonal, square, kagome)
    - Pore wall chemistry provides binding sites (H-bond donors/acceptors,
      electrostatic surfaces, hydrophobic pockets)
    - Linkage type controls stability: β-ketoenamine > imine > boronate
    - Functionalization via monomer design (pre-synthetic) or PSM (post-synthetic)

Compared to MOFs:
    - No metal coordination sites → pure H-bond/shape/hydrophobic selectivity
    - Generally larger pores (1.0–5.0 nm vs 0.3–2.0 nm for MOFs)
    - Better chemical stability in water (β-ketoenamine COFs survive pH 1–14)
    - Lower density (all light elements: C, H, N, O, B)
    - Crystallinity is harder to achieve (Yaghi group AI agent, JACS 2026)

Selenite relevance:
    - Urea-functionalized COF pores can provide the H-bond cavity
    - Cone geometry achievable through pore-wall curvature + functional group
      orientation
    - If combined with Zr-node MOF hybrid, gets both metal coordination
      and organic cavity selectivity
    - Pure COF approach: H-bond selectivity without metal center.
      Loses the metal-exchange channel but retains geometry + H-bond channels.

Does NOT:
    - Run DFT on COF structures
    - Access any database at runtime (uses embedded monomer/topology catalog)
    - Predict crystallinity (flags for experimental optimization)

References:
    Côté et al. Science 310:1166 (2005) — first COF (Yaghi)
    Kandambeth et al. JACS 134:19524 (2012) — β-ketoenamine COFs
    Ahn, Wang et al. JACS 2026, DOI: 10.1021/jacs.5c23233 — AI-driven COF synthesis
    Lohse & Bein, Adv. Funct. Mater. 28:1705553 (2018) — COF design principles review
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from adapters.base import ToolAdapter, Capability, ContributionAssessment
from core.problem import Problem
from core.candidate import CandidateResult


# ═══════════════════════════════════════════════════════════════════════════
# COF LINKAGE CHEMISTRY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class COFLinkage:
    """A covalent linkage type connecting COF monomers."""
    name: str
    bond_type: str              # "imine", "boronate", "beta_ketoenamine", "triazine", etc.
    formation_reaction: str     # e.g. "amine + aldehyde → imine + H₂O"
    reversibility: str          # "reversible", "quasi-reversible", "irreversible"
    water_stable: bool
    acid_stable: bool           # pH < 3
    base_stable: bool           # pH > 11
    thermal_limit_C: int
    crystallinity_ease: str     # "high", "medium", "low"
    notes: str = ""


LINKAGE_LIBRARY = [
    COFLinkage("imine", "C=N", "amine + aldehyde → imine + H₂O",
               "reversible", False, False, True, 300, "high",
               "Most common COF linkage. Good crystallinity via error correction. "
               "Hydrolyzes in water — not suitable for aqueous remediation without PSM."),

    COFLinkage("beta_ketoenamine", "C=C-NH", "amine + β-ketoaldehyde → enamine-keto",
               "quasi-reversible", True, True, True, 350, "medium",
               "Gold standard for aqueous stability. Forms via imine intermediate "
               "then tautomerizes to keto-enamine. Survives pH 1–14. "
               "Kandambeth et al. JACS 2012."),

    COFLinkage("boronate_ester", "B-O", "boronic acid + diol → boronate + H₂O",
               "reversible", False, False, False, 200, "high",
               "Original COF linkage (Côté 2005). Excellent crystallinity. "
               "Unstable in water. Useful for gas-phase or non-aqueous applications."),

    COFLinkage("triazine", "C₃N₃", "nitrile trimerization → triazine ring",
               "irreversible", True, True, True, 500, "low",
               "Covalent triazine frameworks (CTFs). Extremely stable. "
               "Harsh synthesis conditions (ionothermal, 400°C). Poor crystallinity "
               "from irreversibility — no error correction during growth."),

    COFLinkage("imide", "C(=O)-N-C(=O)", "amine + dianhydride → polyimide",
               "irreversible", True, True, False, 400, "low",
               "Polyimide-linked COFs. High thermal stability. "
               "Two-step: first form polyamic acid, then cyclize."),

    COFLinkage("hydrazone", "C=N-NH", "hydrazide + aldehyde → hydrazone",
               "reversible", True, False, True, 280, "high",
               "Better water stability than imine. Good crystallinity. "
               "Moderate acid sensitivity."),

    COFLinkage("azine", "C=N-N=C", "hydrazine + aldehyde → azine",
               "reversible", True, False, True, 300, "medium",
               "Symmetric double-Schiff-base. Fluorescent COFs."),

    COFLinkage("olefin", "C=C", "Knoevenagel or aldol → vinyl",
               "irreversible", True, True, True, 400, "low",
               "sp2-carbon linked COFs. Ultra-stable. Very challenging crystallinity."),
]

_LINKAGE_BY_NAME = {l.name: l for l in LINKAGE_LIBRARY}


# ═══════════════════════════════════════════════════════════════════════════
# COF MONOMER LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class COFMonomer:
    """An organic building block for COF construction."""
    name: str
    role: str                   # "node" (≥3 connections) or "linker" (2 connections)
    connectivity: int           # number of reactive functional groups
    geometry: str               # "trigonal", "tetrahedral", "linear", "square"
    reactive_group: str         # "amine", "aldehyde", "boronic_acid", "diol", "nitrile"
    approx_size_nm: float       # longest dimension
    functional_groups: list     # additional groups on the monomer body
    commercial: bool            # readily available?
    notes: str = ""


MONOMER_LIBRARY = [
    # ── Amine nodes (trigonal, 3-connected) ──
    COFMonomer("TAPB", "node", 3, "trigonal", "amine", 1.2,
               [], True,
               "1,3,5-tris(4-aminophenyl)benzene. Workhorse trigonal amine."),
    COFMonomer("TAPA", "node", 3, "trigonal", "amine", 0.9,
               [], True,
               "Tris(4-aminophenyl)amine. Smaller trigonal amine with N center."),
    COFMonomer("TAPB-urea", "node", 3, "trigonal", "amine", 1.4,
               ["urea_NH"], False,
               "TAPB with urea groups on each arm. Provides H-bond donors "
               "pointing into pore. Custom synthesis required."),
    COFMonomer("TAPB-SO3", "node", 3, "trigonal", "amine", 1.3,
               ["sulfonate"], False,
               "TAPB with sulfonate groups. Anionic pore walls for cation capture."),
    COFMonomer("melamine", "node", 3, "trigonal", "amine", 0.4,
               ["triazine_core"], True,
               "1,3,5-triazine-2,4,6-triamine. Small, rigid, H-bond rich."),

    # ── Amine linkers (linear, 2-connected) ──
    COFMonomer("PDA", "linker", 2, "linear", "amine", 0.6,
               [], True,
               "p-Phenylenediamine. Short linear diamine."),
    COFMonomer("BZ", "linker", 2, "linear", "amine", 1.0,
               [], True,
               "Benzidine (4,4'-diaminobiphenyl). Medium linear diamine."),
    COFMonomer("DHBD", "linker", 2, "linear", "amine", 1.1,
               ["hydroxyl"], True,
               "2,5-Dihydroxybenzidine. For β-ketoenamine linkage. "
               "Hydroxyl groups provide H-bond donors on pore walls."),

    # ── Aldehyde nodes (trigonal, 3-connected) ──
    COFMonomer("TFB", "node", 3, "trigonal", "aldehyde", 0.7,
               [], True,
               "1,3,5-Triformylbenzene. Small trigonal aldehyde."),
    COFMonomer("TFP", "node", 3, "trigonal", "aldehyde", 0.7,
               ["hydroxyl"], True,
               "1,3,5-Triformylphloroglucinol. THE β-ketoenamine aldehyde. "
               "Hydroxyl groups enable tautomerization to keto-enamine."),
    COFMonomer("TFPB", "node", 3, "trigonal", "aldehyde", 1.2,
               [], True,
               "1,3,5-Tris(4-formylphenyl)benzene. Larger trigonal aldehyde."),

    # ── Aldehyde linkers (linear, 2-connected) ──
    COFMonomer("TA", "linker", 2, "linear", "aldehyde", 0.6,
               [], True,
               "Terephthalaldehyde. Short linear dialdehyde."),
    COFMonomer("BPDA", "linker", 2, "linear", "aldehyde", 1.0,
               [], True,
               "Biphenyl-4,4'-dicarbaldehyde. Medium linear dialdehyde."),
    COFMonomer("BPDA-urea", "linker", 2, "linear", "aldehyde", 1.2,
               ["urea_NH"], False,
               "BPDA with pendant urea groups. H-bond donors for anion recognition. "
               "Custom synthesis."),

    # ── Tetrahedral nodes (4-connected, for 3D COFs) ──
    COFMonomer("TAPM", "node", 4, "tetrahedral", "amine", 0.8,
               [], True,
               "Tetrakis(4-aminophenyl)methane. For 3D COFs."),
    COFMonomer("TAPA-Si", "node", 4, "tetrahedral", "amine", 0.9,
               [], False,
               "Tetrakis(4-aminophenyl)silane. Si center, 3D COF node."),

    # ── Square nodes (4-connected planar) ──
    COFMonomer("TAP-porphyrin", "node", 4, "square", "amine", 1.5,
               ["porphyrin_core"], False,
               "Tetra(aminophenyl)porphyrin. Metalloporphyrin core provides "
               "metal coordination INSIDE the COF strut — hybrid approach."),
]

_MONOMER_BY_NAME = {m.name: m for m in MONOMER_LIBRARY}


# ═══════════════════════════════════════════════════════════════════════════
# COF TOPOLOGY PRESETS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class COFTopology:
    """A known COF architecture with characterized properties."""
    name: str
    node_monomer: str           # from MONOMER_LIBRARY
    linker_monomer: str         # from MONOMER_LIBRARY
    linkage: str                # from LINKAGE_LIBRARY
    dimensionality: str         # "2D" or "3D"
    lattice: str                # "hexagonal", "square", "kagome", "diamond", etc.
    pore_diameter_nm: float     # accessible pore diameter
    surface_area_m2_g: float    # BET surface area
    pore_wall_groups: list      # functional groups lining the pore
    water_stable: bool
    notes: str = ""


COF_TOPOLOGIES = [
    # ── Benchmark 2D COFs ──
    COFTopology("TpPa-1", "TFP", "PDA", "beta_ketoenamine", "2D", "hexagonal",
                1.8, 535, ["keto_NH", "keto_C=O"], True,
                "Kandambeth 2012 benchmark. Survives boiling acid/base. "
                "Moderate surface area. NH and C=O groups line pore walls."),

    COFTopology("TpBD", "TFP", "BZ", "beta_ketoenamine", "2D", "hexagonal",
                2.4, 960, ["keto_NH", "keto_C=O"], True,
                "Larger pore β-ketoenamine COF. Higher surface area."),

    COFTopology("TpDHBD", "TFP", "DHBD", "beta_ketoenamine", "2D", "hexagonal",
                2.2, 800, ["keto_NH", "keto_C=O", "hydroxyl"], True,
                "Hydroxyl-functionalized pore walls. Additional H-bond donors. "
                "Relevant for anion recognition."),

    COFTopology("TAPB-TFB-imine", "TAPB", "TFB", "imine", "2D", "hexagonal",
                3.2, 1250, [], False,
                "Large pore imine COF. High surface area but water-unstable."),

    COFTopology("TAPB-TFP-keto", "TAPB", "TFP", "beta_ketoenamine", "2D", "hexagonal",
                3.0, 1100, ["keto_NH", "keto_C=O"], True,
                "Large pore β-ketoenamine. Water stable. Combinable with "
                "functional monomer variants."),

    # ── Functionalized for anion binding ──
    COFTopology("TpPa-urea", "TFP", "PDA", "beta_ketoenamine", "2D", "hexagonal",
                1.5, 400, ["keto_NH", "keto_C=O", "urea_NH"], True,
                "HYPOTHETICAL: PDA replaced with urea-PDA variant. "
                "Urea NH donors point into pore for selenite H-bonding. "
                "Reduced pore size from urea groups. "
                "β-ketoenamine backbone provides aqueous stability."),

    COFTopology("TAPB-urea-TFP", "TAPB-urea", "TFP", "beta_ketoenamine",
                "2D", "hexagonal",
                2.5, 750, ["keto_NH", "keto_C=O", "urea_NH"], True,
                "HYPOTHETICAL: TAPB-urea node provides 3 urea groups per node "
                "pointing into hexagonal pore. Each pore ringed by 6 ureas = "
                "12 NH donors. Directly maps to selenite receptor design. "
                "Retains β-ketoenamine stability."),

    # ── 3D COFs ──
    COFTopology("COF-300", "TAPM", "TA", "imine", "3D", "diamond",
                0.9, 1360, [], False,
                "Interpenetrated diamond topology. Small pores from interpenetration."),

    COFTopology("COF-320", "TAPM", "BPDA", "imine", "3D", "diamond",
                1.2, 2400, [], False,
                "Non-interpenetrated diamond. Very high surface area. Water-unstable."),

    # ── Metalloporphyrin hybrid ──
    COFTopology("COF-porph-Zr", "TAP-porphyrin", "TFP", "beta_ketoenamine",
                "2D", "square",
                2.0, 600, ["keto_NH", "keto_C=O", "porphyrin_metal"], True,
                "HYPOTHETICAL: Porphyrin node metallated with Zr. Combines "
                "COF pore geometry (H-bond walls) with metal coordination "
                "(porphyrin Zr center). Hybrid metal + organic approach for "
                "selenite binding. Best of both worlds if synthesis works."),
]

_TOPOLOGY_BY_NAME = {t.name: t for t in COF_TOPOLOGIES}


# ═══════════════════════════════════════════════════════════════════════════
# COF DESIGN OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class COFDesign:
    """A proposed COF realization of an InteractionGeometrySpec."""
    topology_name: str
    node_monomer: str
    linker_monomer: str
    linkage_type: str
    dimensionality: str
    pore_diameter_nm: float
    pore_wall_groups: list
    water_stable: bool

    # Scoring against the geometry spec
    pore_size_match: float      # 0–1, how well pore matches target cavity
    hbond_donor_count: int      # NH donors available per pore
    hbond_match: float          # 0–1, match to spec H-bond requirements
    stability_match: float      # 0–1, operating condition compatibility
    synthetic_accessibility: float  # 0–1

    # Flags
    is_hypothetical: bool = False
    requires_custom_monomer: bool = False
    notes: str = ""

    @property
    def overall_score(self) -> float:
        """Weighted composite score."""
        return (0.30 * self.pore_size_match +
                0.25 * self.hbond_match +
                0.25 * self.stability_match +
                0.20 * self.synthetic_accessibility)


# ═══════════════════════════════════════════════════════════════════════════
# COF DESIGN LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def design_cof_for_target(
    target_cavity_nm: float,
    required_hbond_donors: int = 0,
    required_hbond_acceptors: int = 0,
    water_required: bool = True,
    acid_required: bool = False,
    temperature_max_C: int = 100,
    has_metal_center: bool = False,
    functional_groups: Optional[list] = None,
) -> list[COFDesign]:
    """
    Design COF candidates for a target binding geometry.

    Parameters
    ----------
    target_cavity_nm : float
        Target pore diameter in nm.
    required_hbond_donors : int
        Minimum NH donors per pore needed for binding.
    required_hbond_acceptors : int
        Minimum acceptors per pore needed.
    water_required : bool
        Must survive aqueous conditions.
    acid_required : bool
        Must survive pH < 3.
    temperature_max_C : int
        Maximum operating temperature.
    has_metal_center : bool
        Whether metal coordination is needed (routes to hybrid COFs).
    functional_groups : list, optional
        Specific functional groups needed on pore walls (e.g. ["urea_NH"]).

    Returns
    -------
    list[COFDesign]
        Ranked candidate COFs, best first.
    """
    candidates = []

    for topo in COF_TOPOLOGIES:
        linkage = _LINKAGE_BY_NAME.get(topo.linkage)
        if not linkage:
            continue

        # ── Stability gates ──
        if water_required and not topo.water_stable:
            continue
        if acid_required and linkage and not linkage.acid_stable:
            continue
        if temperature_max_C > linkage.thermal_limit_C:
            continue

        # ── Metal center gate ──
        if has_metal_center and "porphyrin_metal" not in topo.pore_wall_groups:
            continue
        if not has_metal_center and "porphyrin_metal" in topo.pore_wall_groups:
            pass  # allow but don't prefer

        # ── Pore size match ──
        size_diff = abs(topo.pore_diameter_nm - target_cavity_nm)
        pore_match = max(0.0, 1.0 - size_diff / target_cavity_nm) if target_cavity_nm > 0 else 0.5

        # ── H-bond donor count ──
        nh_per_pore = 0
        for grp in topo.pore_wall_groups:
            if grp == "urea_NH":
                # Hexagonal: 6 edges per pore, urea on each → 12 NH
                # Square: 4 edges per pore → 8 NH
                if topo.lattice == "hexagonal":
                    nh_per_pore += 12
                elif topo.lattice == "square":
                    nh_per_pore += 8
                else:
                    nh_per_pore += 6
            elif grp == "keto_NH":
                # β-ketoenamine: 1 NH per linkage, 6 linkages per hex pore
                if topo.lattice == "hexagonal":
                    nh_per_pore += 6
                else:
                    nh_per_pore += 4
            elif grp == "hydroxyl":
                nh_per_pore += 3  # partial H-bond donors

        hbond_match = 1.0
        if required_hbond_donors > 0:
            hbond_match = min(1.0, nh_per_pore / required_hbond_donors)

        # ── Functional group match ──
        if functional_groups:
            matched = sum(1 for fg in functional_groups if fg in topo.pore_wall_groups)
            fg_match = matched / len(functional_groups)
            hbond_match *= fg_match

        # ── Stability score ──
        stability = 1.0
        if linkage:
            if linkage.water_stable:
                stability *= 1.0
            else:
                stability *= 0.3
            if linkage.crystallinity_ease == "high":
                stability *= 1.0
            elif linkage.crystallinity_ease == "medium":
                stability *= 0.8
            else:
                stability *= 0.5

        # ── Synthetic accessibility ──
        node = _MONOMER_BY_NAME.get(topo.node_monomer)
        linker = _MONOMER_BY_NAME.get(topo.linker_monomer)
        sa = 1.0
        custom = False
        hypothetical = "HYPOTHETICAL" in topo.notes

        if node and not node.commercial:
            sa *= 0.5
            custom = True
        if linker and not linker.commercial:
            sa *= 0.5
            custom = True
        if hypothetical:
            sa *= 0.7

        design = COFDesign(
            topology_name=topo.name,
            node_monomer=topo.node_monomer,
            linker_monomer=topo.linker_monomer,
            linkage_type=topo.linkage,
            dimensionality=topo.dimensionality,
            pore_diameter_nm=topo.pore_diameter_nm,
            pore_wall_groups=list(topo.pore_wall_groups),
            water_stable=topo.water_stable,
            pore_size_match=pore_match,
            hbond_donor_count=nh_per_pore,
            hbond_match=hbond_match,
            stability_match=stability,
            synthetic_accessibility=sa,
            is_hypothetical=hypothetical,
            requires_custom_monomer=custom,
            notes=topo.notes,
        )
        candidates.append(design)

    candidates.sort(key=lambda d: d.overall_score, reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# TOOL ADAPTER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class COFAdapter(ToolAdapter):
    """MABE adapter for covalent organic framework design."""

    @property
    def name(self) -> str:
        return "cof_adapter"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> list[Capability]:
        return [
            Capability(
                description="Design COF pore geometry for selective molecular capture",
                target_types=["anion", "small_molecule", "gas"],
                interaction_types=["h_bond", "shape_selective", "electrostatic",
                                   "hydrophobic_cavity"],
                output_types=["monomer_spec", "synthesis_conditions", "pore_properties"],
            ),
            Capability(
                description="Rank COF topologies by pore-target geometric match",
                target_types=["anion", "small_molecule"],
                interaction_types=["shape_selective"],
                output_types=["ranked_topologies"],
            ),
        ]

    def assess_contribution(self, problem: Problem) -> ContributionAssessment:
        """Assess whether a COF can help with this problem."""

        # COFs are good for: shape-selective capture, gas separation,
        # aqueous anion capture (if β-ketoenamine), catalysis
        target = problem.target
        relevance = 0.0

        # Check if target fits COF pore regime (0.5–5.0 nm)
        if hasattr(target, 'size') and hasattr(target.size, 'vdw_radius'):
            r_nm = target.size.vdw_radius
            if 0.1 < r_nm < 2.5:
                relevance += 0.3

        # Check if shape selectivity is needed
        if hasattr(problem, 'exclusions') and len(problem.exclusions) > 0:
            relevance += 0.3  # selectivity needed → COF geometry helps

        # Check operating conditions
        if hasattr(problem, 'constraints'):
            scale = getattr(problem.constraints, 'required_scale', 'g')
            if scale in ('kg', 'tonne'):
                relevance += 0.2  # COFs scale better than DNA origami

        # Check if aqueous
        if hasattr(problem, 'matrix'):
            if 'aqueous' in str(getattr(problem.matrix, 'solvent', '')).lower():
                relevance += 0.1  # β-ketoenamine COFs work in water

        can = relevance > 0.3

        return ContributionAssessment(
            can_contribute=can,
            relevance=relevance,
            what_it_would_do=(
                "Design a covalent organic framework with pore geometry "
                "matched to the target. COFs provide shape-selective cavities "
                "at bulk scale without metal nodes."
            ),
            what_part_of_problem="Selective molecular capture via pore geometry",
            estimated_compute_time="<1 second (catalog lookup + scoring)",
            limitations=[
                "No metal coordination sites (unless hybrid porphyrin-COF)",
                "Crystallinity not guaranteed — experimental optimization needed",
                "Large pores (>1 nm) — less effective for very small targets",
                "Custom monomers for functionalized variants require synthesis",
            ],
        )

    def generate_candidates(self, problem: Problem) -> list[CandidateResult]:
        """Generate COF candidates for a MABE Problem."""
        # Extract target dimensions
        target = problem.target
        cavity_nm = 0.5  # default
        if hasattr(target, 'size') and hasattr(target.size, 'vdw_radius'):
            cavity_nm = target.size.vdw_radius * 2.5  # pore ≈ 2.5× target radius

        # Extract H-bond requirements
        n_hbd = 0
        if hasattr(target, 'hydration') and hasattr(target.hydration, 'n_hbond_sites'):
            n_hbd = target.hydration.n_hbond_sites

        # Check aqueous
        water = False
        if hasattr(problem, 'matrix'):
            water = 'aqueous' in str(getattr(problem.matrix, 'solvent', '')).lower()

        designs = design_cof_for_target(
            target_cavity_nm=cavity_nm,
            required_hbond_donors=n_hbd,
            water_required=water,
        )

        results = []
        for d in designs[:5]:  # top 5
            results.append(CandidateResult(
                name=d.topology_name,
                description=(
                    f"COF: {d.node_monomer} + {d.linker_monomer} "
                    f"({d.linkage_type}, {d.dimensionality} {d.pore_diameter_nm:.1f} nm pore)"
                ),
                score=d.overall_score,
                details={
                    "pore_diameter_nm": d.pore_diameter_nm,
                    "linkage": d.linkage_type,
                    "water_stable": d.water_stable,
                    "nh_donors_per_pore": d.hbond_donor_count,
                    "is_hypothetical": d.is_hypothetical,
                    "requires_custom_monomer": d.requires_custom_monomer,
                },
            ))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("COF ADAPTER — Selenite Receptor Design Demo")
    print("=" * 70)
    print()

    # Selenite receptor requirements from Phase 13a
    designs = design_cof_for_target(
        target_cavity_nm=0.8,       # selenite fits ~0.5 nm, pore ~0.8 for access
        required_hbond_donors=6,    # need NH donors for oxyanion H-bonding
        water_required=True,        # Elk Valley = aqueous
        acid_required=False,        # pH 7-8.5
        functional_groups=["urea_NH"],  # directional H-bond donors
    )

    print(f"Candidates for selenite receptor (aqueous, H-bond cavity):\n")
    print(f"{'Rank':<5} {'Topology':<20} {'Pore':>5} {'NH':>4} {'Score':>6} "
          f"{'Water':>6} {'Custom':>7} {'Hypo':>5}")
    print("-" * 65)

    for i, d in enumerate(designs, 1):
        print(f"  {i:<3} {d.topology_name:<20} {d.pore_diameter_nm:>4.1f}nm "
              f"{d.hbond_donor_count:>4} {d.overall_score:>6.3f} "
              f"{'yes' if d.water_stable else 'no':>6} "
              f"{'yes' if d.requires_custom_monomer else 'no':>7} "
              f"{'yes' if d.is_hypothetical else 'no':>5}")

    print()
    if designs:
        top = designs[0]
        print(f"Top pick: {top.topology_name}")
        print(f"  Node: {top.node_monomer}")
        print(f"  Linker: {top.linker_monomer}")
        print(f"  Linkage: {top.linkage_type}")
        print(f"  Pore: {top.pore_diameter_nm} nm, {top.dimensionality}")
        print(f"  NH donors/pore: {top.hbond_donor_count}")
        print(f"  Water stable: {top.water_stable}")
        print(f"  Notes: {top.notes}")
    print()

    # Compare: what if no H-bond donors required? (gas separation use case)
    print("=" * 70)
    print("COF ADAPTER — CO₂ Capture Design Demo (no H-bond requirement)")
    print("=" * 70)
    print()

    co2_designs = design_cof_for_target(
        target_cavity_nm=1.5,
        required_hbond_donors=0,
        water_required=False,
        acid_required=False,
    )

    for i, d in enumerate(co2_designs[:3], 1):
        print(f"  {i}. {d.topology_name} — {d.pore_diameter_nm} nm pore, "
              f"score {d.overall_score:.3f}")

    print()
    print(f"Monomer library: {len(MONOMER_LIBRARY)} entries")
    print(f"Linkage library: {len(LINKAGE_LIBRARY)} entries")
    print(f"Topology presets: {len(COF_TOPOLOGIES)} entries")