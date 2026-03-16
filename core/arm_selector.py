"""
arm_selector.py — Geometry-Aware Arm Selection for Scaffold Design

Given a scaffold descriptor and a guest specification, selects and ranks
optimal arm combinations from the ARM_LIBRARY based on:

1. Electronic compatibility: arm donor type matches guest requirements
2. pKa compatibility: arm donors active at working pH
3. Steric compatibility: arm size appropriate for scaffold site geometry
4. Geometric matching: arm donor projection aligns with guest binding vector

Standalone usage:
    from core.arm_selector import select_arms, ArmAssignment
    assignments = select_arms(scaffold_desc, guest_spec)

Auto-wired into design_scaffold() via design_with_arms().

All selection logic from first principles. No fitting to binding data.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import product as cartesian_product


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

R_kJ = 8.314e-3
T_STD = 298.15
RT_STD = R_kJ * T_STD
PI = math.pi


# ═══════════════════════════════════════════════════════════════════════════
# New Anion/Oxoanion Arms
# ═══════════════════════════════════════════════════════════════════════════
#
# These are added to ARM_LIBRARY via patch. Defined here for reference and
# so the selector knows about them even if the patch hasn't been applied.

ANION_ARMS_DEFINITIONS = [
    # (name, smiles, donor_subtypes, donor_element, hardness, category, notes)
    ("bis-urea", "[*]NC(=O)NC(=O)N",
     ["N_urea", "N_urea", "O_carbonyl", "O_carbonyl"], "N", "hard",
     "anion-binding",
     "4 NH donors + 2 C=O, strong oxoanion cleft. "
     "Ref: Gale PA. Chem. Soc. Rev. 2010, 39, 3746"),

    ("amidinium", "[*]C(=N)N",
     ["N_amine", "N_amine"], "N", "hard",
     "anion-binding",
     "Cationic at neutral pH, charge-assisted H-bond donor pair. "
     "Ref: Schmidtchen FP. Top. Curr. Chem. 2005, 255, 1"),

    ("triazole-NH", "[*]c1cn[nH]n1",
     ["N_pyrrole", "N_pyridine"], "N", "borderline",
     "anion-binding",
     "1,2,3-Triazole: NH donor + two N acceptors, click-accessible. "
     "Ref: Meudtner RM, Hecht S. Angew. Chem. Int. Ed. 2008, 47, 4926"),

    ("indole-amide", "[*]NC(=O)c1c[nH]c2ccccc12",
     ["N_amide", "N_pyrrole", "O_carbonyl"], "N", "borderline",
     "anion-binding",
     "Indole NH + amide NH convergent donors, large pi surface. "
     "Ref: Gale PA. Acc. Chem. Res. 2006, 39, 465"),

    ("amino-thiourea", "[*]NC(=S)NN",
     ["N_amine", "N_amine", "S_thiolate"], "N", "borderline",
     "anion-binding",
     "Thiourea NH donors + terminal hydrazine, strong H-bond array. "
     "Ref: Zhang Z, Schreiner PR. Chem. Soc. Rev. 2009, 38, 1187"),

    ("cyanuric-NH", "[*]Nc1nc(N)[nH]c(=O)n1",
     ["N_amine", "N_amine", "O_carbonyl"], "N", "hard",
     "anion-binding",
     "Cyanuric acid derivative: 2 NH donors + C=O, planar, H-bond array. "
     "Ref: Sato K. JACS 2003, 125, 8066"),

    ("hydroxamic-extended", "[*]CCC(=O)NO",
     ["O_hydroxamate", "O_hydroxamate"], "O", "hard",
     "anion-binding",
     "Extended hydroxamate with propyl spacer, better reach into cavity. "
     "Ref: Codd R. Coord. Chem. Rev. 2008, 252, 1387"),

    ("amino-pyridine", "[*]Nc1ccccn1",
     ["N_amine", "N_pyridine"], "N", "borderline",
     "anion-binding",
     "2-Aminopyridine: NH donor + pyridine N, bidentate for oxoanions. "
     "Ref: Kang SO, Begum RA, Bowman-James K. Angew. Chem. Int. Ed. 2006, 45, 7882"),

    ("phosphoryl-NH", "[*]NP(=O)(O)O",
     ["N_amine", "O_phosphate", "O_phosphate"], "N", "hard",
     "anion-binding",
     "Phosphoramide: NH donor + phosphoryl O acceptors, binds oxoanions. "
     "Ref: Tobey SL, Anslyn EV. JACS 2003, 125, 14807"),

    ("chromone-hydroxyl", "[*]c1cc(=O)c2ccccc2o1",
     ["O_carbonyl", "O_hydroxyl"], "O", "hard",
     "anion-binding",
     "Chromone: C=O + ring O, planar H-bond acceptor, aromatic wall. "
     "Ref: Ghosh S, Keillor JW. JACS 2019, 141, 18822"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Compatibility Matrix
# ═══════════════════════════════════════════════════════════════════════════

# Guest requirement → compatible donor types
# These map what the GUEST needs to what the ARM provides.

GUEST_DONOR_COMPATIBILITY = {
    # Guest is anion (needs H-bond donors from receptor arms)
    "anion_hb_donors": [
        "N_urea", "N_amine", "N_amide", "N_pyrrole",
        "O_hydroxyl", "N_sulfonamide",
    ],
    # Guest is H-bond acceptor (needs donor arms)
    "hb_donor_arms": [
        "N_amine", "N_urea", "N_amide", "N_pyrrole",
        "O_hydroxyl", "O_carboxylate",
    ],
    # Guest is H-bond donor (needs acceptor arms)
    "hb_acceptor_arms": [
        "N_pyridine", "N_imidazole", "O_carbonyl",
        "O_carboxylate", "O_phosphate", "N_nitrile",
    ],
    # Guest is aromatic (needs pi-stacking arms)
    "pi_stacking_arms": [
        "naphthyl", "anthracenyl", "indolyl", "pyrrole",
        "N_pyridine", "N_imidazole",
    ],
    # Guest is cation (needs anionic/electron-rich arms)
    "cation_binding": [
        "O_carboxylate", "O_phenolate", "O_phosphate",
        "O_sulfonate", "S_thiolate",
    ],
}

# Steric size categories for arms
# Small arms fit tight scaffolds; large arms need wide scaffolds
ARM_SIZE_CLASS = {
    # name → ("small", "medium", "large") based on heavy atom count
    "aminomethyl": "small", "aminoethyl": "small",
    "thiol-methyl": "small", "thiol-ethyl": "small",
    "thioether-methyl": "small", "ethanol": "small",
    "nitrile": "small", "H-cap": "small",
    "acetic-acid": "small", "2-pyridyl": "medium",
    "2-pyridylmethyl": "medium", "picolylamine": "medium",
    "imidazolylmethyl": "medium", "phenol": "medium",
    "oxime-simple": "small", "amidoxime": "medium",
    "hydrazide": "medium", "carbamoylmethyl": "small",
    "propionic-acid": "medium", "phosphonate": "medium",
    "acetohydroxamate": "medium", "hydroxypyridinone": "medium",
    "squaramide": "medium", "salicylaldimine": "medium",
    "catechol": "medium", "bisphosphonate": "large",
    "8-hydroxyquinolinyl": "large", "benzimidazolyl": "medium",
    "sulfonate": "medium", "diphenylphosphino": "large",
    "dithiocarbamate-NMe": "medium", "dithiocarbamate-NH": "medium",
    "thiosemicarbazone": "medium", "thiourea": "small",
    "aminothiadiazole": "medium", "thioether-propyl": "medium",
    "thioacetate": "medium", "mercaptoacetate": "medium",
    "pyridine-2-thiol": "medium", "thiadiazole-thiol": "medium",
    "indolyl": "medium", "pyrrole": "small", "naphthyl": "large",
    "anthracenyl": "large", "urea-NH2": "small", "guanidinium": "small",
    "sulfonamide-NH": "medium", "nitrophenyl": "medium",
    # New anion arms
    "bis-urea": "medium", "amidinium": "small",
    "triazole-NH": "small", "indole-amide": "large",
    "amino-thiourea": "medium", "cyanuric-NH": "medium",
    "hydroxamic-extended": "medium", "amino-pyridine": "medium",
    "phosphoryl-NH": "medium", "chromone-hydroxyl": "large",
}

# Scaffold site size → compatible arm sizes
STERIC_COMPAT = {
    # (scaffold_d_arm class, arm_size) → compatible
    ("tight", "small"): True,    # d_arm < 5 Å
    ("tight", "medium"): True,
    ("tight", "large"): False,   # large arms clash in tight scaffolds
    ("medium", "small"): True,   # d_arm 5-8 Å
    ("medium", "medium"): True,
    ("medium", "large"): True,
    ("wide", "small"): True,     # d_arm > 8 Å
    ("wide", "medium"): True,
    ("wide", "large"): True,
}


def scaffold_size_class(d_arm_A: float) -> str:
    """Classify scaffold arm separation."""
    if d_arm_A <= 0:
        return "medium"  # unknown → permissive
    if d_arm_A < 5.0:
        return "tight"
    elif d_arm_A < 8.0:
        return "medium"
    else:
        return "wide"


def arm_size(arm_name: str) -> str:
    """Look up arm steric size class."""
    return ARM_SIZE_CLASS.get(arm_name, "medium")


def steric_compatible(scaffold_d_arm: float, arm_name: str) -> bool:
    """Check steric compatibility of an arm with a scaffold."""
    sc = scaffold_size_class(scaffold_d_arm)
    ac = arm_size(arm_name)
    return STERIC_COMPAT.get((sc, ac), True)


# ═══════════════════════════════════════════════════════════════════════════
# pKa-Aware Arm Filtering
# ═══════════════════════════════════════════════════════════════════════════

def arm_pka_score(arm_donor_types: List[str], pH: float) -> float:
    """Score arm donor availability at working pH.

    Returns mean fraction active across all donor types on the arm.
    1.0 = all donors fully active. 0.0 = all donors dead.

    Uses Henderson-Hasselbalch from scaffold_designer Phase C.
    """
    from core.scaffold_designer import donor_fraction_active

    if not arm_donor_types:
        return 1.0  # no donors → no pKa issue (e.g., naphthyl)

    fractions = [donor_fraction_active(dt, pH) for dt in arm_donor_types]
    return sum(fractions) / len(fractions)


def arm_pka_penalty(arm_donor_types: List[str], pH: float) -> float:
    """pKa penalty in kJ/mol for maintaining arm donors in active form."""
    from core.scaffold_designer import compute_pka_adjustment
    dG, _ = compute_pka_adjustment(arm_donor_types, pH)
    return dG


# ═══════════════════════════════════════════════════════════════════════════
# Electronic Compatibility
# ═══════════════════════════════════════════════════════════════════════════

def electronic_score(arm_donor_types: List[str],
                     guest_needs: List[str]) -> float:
    """Score how well an arm's donor types match guest requirements.

    Returns 0.0–1.0 match score.
    1.0 = all guest needs met by this arm's donors.
    0.0 = no overlap.
    """
    if not guest_needs:
        return 0.5  # no specific need → neutral

    # Collect all compatible donor types for the guest
    compatible = set()
    for need in guest_needs:
        if need in GUEST_DONOR_COMPATIBILITY:
            compatible.update(GUEST_DONOR_COMPATIBILITY[need])

    if not compatible:
        return 0.5

    # What fraction of the arm's donors are compatible?
    if not arm_donor_types:
        return 0.1  # pure hydrophobic arm — low match for most guests

    n_match = sum(1 for dt in arm_donor_types if dt in compatible)
    return n_match / len(arm_donor_types)


def guest_electronic_needs(guest) -> List[str]:
    """Determine electronic requirements from a GuestSpec."""
    needs = []
    if guest.is_anion:
        needs.append("anion_hb_donors")
    if guest.n_hb_acceptors > 0:
        needs.append("hb_donor_arms")
    if guest.n_hb_donors > 0:
        needs.append("hb_acceptor_arms")
    if guest.n_aromatic_rings > 0:
        needs.append("pi_stacking_arms")
    if guest.charge > 0:
        needs.append("cation_binding")
    return needs


# ═══════════════════════════════════════════════════════════════════════════
# Geometry-Aware Arm Placement
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ArmGeometry:
    """Geometric descriptor for an arm's donor projection.

    Extracted from 3D embedding (RDKit) or estimated from topology.
    """
    name: str
    n_heavy: int = 0
    donor_reach_A: float = 0.0     # distance from attachment to donor atom
    donor_angle_deg: float = 0.0   # angle of donor projection from backbone axis
    n_rotors: int = 0              # rotatable bonds between attachment and donor
    is_rigid: bool = False         # arm has no rotatable bonds

    @property
    def flexibility(self) -> float:
        """0 = rigid, 1 = fully flexible."""
        return min(1.0, self.n_rotors / 5.0)


# Precomputed arm geometries (estimated from topology, no RDKit needed)
# donor_reach ≈ n_bonds × 1.5 Å; angle ≈ 0° for linear, 60-120° for branched

_ARM_GEOMETRY_ESTIMATES: Dict[str, Tuple[float, float, int]] = {
    # (reach_A, angle_deg, n_rotors)
    "aminomethyl": (2.5, 0.0, 1),
    "aminoethyl": (3.8, 0.0, 2),
    "acetic-acid": (3.8, 0.0, 2),
    "propionic-acid": (5.2, 0.0, 3),
    "ethanol": (3.8, 0.0, 2),
    "phenol": (4.0, 30.0, 1),
    "2-pyridyl": (3.5, 15.0, 0),
    "2-pyridylmethyl": (5.0, 15.0, 1),
    "picolylamine": (6.0, 20.0, 2),
    "imidazolylmethyl": (4.5, 15.0, 1),
    "catechol": (4.0, 30.0, 1),
    "phosphonate": (4.0, 0.0, 2),
    "thiol-methyl": (2.5, 0.0, 1),
    "thiol-ethyl": (3.8, 0.0, 2),
    "thioether-methyl": (2.5, 0.0, 1),
    "thioether-propyl": (5.2, 0.0, 3),
    "urea-NH2": (3.5, 30.0, 1),
    "guanidinium": (3.0, 0.0, 1),
    "bis-urea": (5.5, 30.0, 2),
    "amidinium": (2.8, 0.0, 1),
    "triazole-NH": (3.5, 15.0, 0),
    "indole-amide": (6.0, 30.0, 1),
    "amino-thiourea": (4.5, 20.0, 2),
    "amino-pyridine": (4.5, 15.0, 1),
    "pyrrole": (3.0, 15.0, 0),
    "indolyl": (4.0, 15.0, 0),
    "naphthyl": (5.0, 0.0, 0),
    "anthracenyl": (6.5, 0.0, 0),
    "sulfonamide-NH": (4.0, 30.0, 1),
    "thiourea": (3.5, 20.0, 1),
    "dithiocarbamate-NMe": (3.0, 30.0, 0),
    "dithiocarbamate-NH": (3.0, 30.0, 0),
    "8-hydroxyquinolinyl": (5.0, 20.0, 0),
    "salicylaldimine": (5.0, 15.0, 1),
    "hydroxamic-extended": (6.5, 0.0, 3),
    "cyanuric-NH": (4.5, 30.0, 1),
    "chromone-hydroxyl": (5.5, 20.0, 0),
    "phosphoryl-NH": (4.5, 30.0, 1),
    "mercaptoacetate": (4.0, 0.0, 2),
}


def get_arm_geometry(arm_name: str) -> ArmGeometry:
    """Get arm geometry from precomputed table or estimate."""
    if arm_name in _ARM_GEOMETRY_ESTIMATES:
        reach, angle, n_rot = _ARM_GEOMETRY_ESTIMATES[arm_name]
        return ArmGeometry(
            name=arm_name,
            donor_reach_A=reach,
            donor_angle_deg=angle,
            n_rotors=n_rot,
            is_rigid=(n_rot == 0),
        )
    # Default: medium reach, flexible
    return ArmGeometry(name=arm_name, donor_reach_A=4.0,
                       donor_angle_deg=15.0, n_rotors=2)


def geometry_match_score(arm_geo: ArmGeometry,
                          scaffold_d_arm: float,
                          scaffold_theta_conv: float,
                          guest_diameter: float) -> float:
    """Score geometric compatibility of arm placement on scaffold.

    Evaluates whether the arm's donor, when mounted on the scaffold,
    can reach the guest binding zone.

    Physics:
        effective_reach = d_arm/2 + donor_reach × cos(angle_mismatch)
        match = 1.0 when effective_reach ≈ guest_radius + vdW

    Parameters
    ----------
    arm_geo : ArmGeometry
    scaffold_d_arm : float
        Scaffold arm separation (Å).
    scaffold_theta_conv : float
        Scaffold convergence angle (degrees).
    guest_diameter : float
        Guest diameter (Å).

    Returns
    -------
    float
        Geometric match score 0.0–1.0.
    """
    if scaffold_d_arm <= 0 or guest_diameter <= 0:
        return 0.5  # unknown geometry → neutral

    # Target: arm donor should reach approximately guest center
    guest_radius = guest_diameter / 2.0
    half_d_arm = scaffold_d_arm / 2.0

    # How far does the arm project toward the guest?
    # For convergent scaffolds (theta < 90°), arm points inward
    # For divergent (theta > 90°), arm points outward
    if scaffold_theta_conv < 90.0:
        # Convergent: arm projects inward by donor_reach × cos(theta/2)
        proj_angle_rad = math.radians(scaffold_theta_conv / 2.0)
        effective_inward = arm_geo.donor_reach_A * math.cos(proj_angle_rad)
    else:
        # Divergent: arm projects outward — negative for encapsulation
        effective_inward = -arm_geo.donor_reach_A * 0.5

    # Total reach from scaffold center to donor
    total_reach = half_d_arm - effective_inward  # gap between donor and center

    # Ideal: donor is at guest_radius + vdW contact (~1.5 Å)
    ideal_gap = guest_radius + 1.5
    mismatch = abs(total_reach - ideal_gap)

    # Gaussian scoring: perfect match = 1.0, decays with mismatch
    sigma = 2.0  # Å, tolerance
    score = math.exp(-0.5 * (mismatch / sigma) ** 2)

    # Bonus for rigid arms in preorganized scaffolds
    if arm_geo.is_rigid and scaffold_theta_conv < 70.0:
        score = min(1.0, score * 1.1)

    # Penalty for very long flexible arms (entropy cost)
    if arm_geo.n_rotors > 3:
        score *= 0.9

    return score


# ═══════════════════════════════════════════════════════════════════════════
# Arm Selection Engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ArmScore:
    """Scored arm candidate for a scaffold–guest pair."""
    arm_name: str
    donor_types: List[str]
    hardness: str
    electronic_score: float = 0.0    # 0–1, donor type match
    pka_score: float = 0.0           # 0–1, donor availability at pH
    steric_ok: bool = True           # passes steric filter
    geometry_score: float = 0.0      # 0–1, spatial match
    repulsion_score: float = 0.0     # 0–1, differential repulsion selectivity
    pka_penalty_kJ: float = 0.0      # kJ/mol, pKa cost
    composite: float = 0.0           # weighted combination

    @property
    def viable(self) -> bool:
        return self.steric_ok and self.pka_score > 0.05


@dataclass
class ArmAssignment:
    """Optimal arm assignment for a scaffold–guest pair."""
    scaffold_name: str
    guest_name: str
    arm_scores: List[ArmScore]        # all scored arms, ranked
    best_arms: List[str]              # top arm per site (length = n_sites)
    best_combo_score: float = 0.0     # composite score of best assignment
    n_sites: int = 2
    pH: float = 7.4


def _arm_repulsion_score(arm_donor_types: List[str],
                         target_species: str,
                         interferent_species: List[str],
                         cavity_radius_A: float = 3.0) -> float:
    """Score how well an arm's donors differentially repel interferents.

    Returns 0–1. Higher = this arm repels interferents more than the target.
    1.0 = strong differential repulsion (great for selectivity).
    0.0 = no differential repulsion (interferents bind equally).

    Uses HSAB mismatch as the primary mechanism — donor hardness
    determines which species are repelled.
    """
    try:
        from core.repulsion_physics import (
            hsab_mismatch_for_site, HSAB_HARDNESS_EV, IONIC_RADII_A,
            steric_repulsion, hydrophobic_mismatch, HYDRATION_ENTHALPY_kJ,
        )
    except ImportError:
        return 0.5  # neutral if repulsion module unavailable

    if not interferent_species or not arm_donor_types:
        return 0.5  # no interferents or no donors → neutral

    # Compute repulsion for target
    repel_target = hsab_mismatch_for_site(target_species, arm_donor_types)

    # Compute repulsion for each interferent, take the average
    repel_intfs = []
    for intf in interferent_species:
        r = hsab_mismatch_for_site(intf, arm_donor_types)
        # Add steric component if radii available
        r_sp = IONIC_RADII_A.get(intf, 1.0)
        r += steric_repulsion(r_sp, cavity_radius_A) * 0.1  # scaled down
        # Add hydrophobic mismatch component
        h = HYDRATION_ENTHALPY_kJ.get(intf, -500.0)
        r += hydrophobic_mismatch(h, 0.3) * 0.05  # mild contribution
        repel_intfs.append(r)

    if not repel_intfs:
        return 0.5

    avg_repel_intf = sum(repel_intfs) / len(repel_intfs)

    # Differential: interferents should be MORE repelled than target
    # ΔΔG = avg_repel_intf - repel_target
    # Normalize: 10 kJ/mol differential → score 0.8, 0 → 0.5, -10 → 0.2
    ddG = avg_repel_intf - repel_target
    score = 0.5 + 0.03 * ddG  # 0.03 per kJ/mol
    return max(0.0, min(1.0, score))


def score_arm(arm_name: str, arm_donor_types: List[str], arm_hardness: str,
              guest, scaffold_d_arm: float, scaffold_theta_conv: float,
              pH: float,
              w_electronic: float = 0.30,
              w_pka: float = 0.25,
              w_geometry: float = 0.20,
              w_repulsion: float = 0.25) -> ArmScore:
    """Score a single arm for a scaffold–guest pair.

    Parameters
    ----------
    arm_name : str
    arm_donor_types : list of str
    arm_hardness : str
    guest : GuestSpec
    scaffold_d_arm : float (Å)
    scaffold_theta_conv : float (degrees)
    pH : float
    w_electronic, w_pka, w_geometry, w_repulsion : float
        Weights for composite score (must sum to 1.0).

    Returns
    -------
    ArmScore
    """
    needs = guest_electronic_needs(guest)
    e_score = electronic_score(arm_donor_types, needs)
    p_score = arm_pka_score(arm_donor_types, pH)
    p_penalty = arm_pka_penalty(arm_donor_types, pH)

    is_steric_ok = steric_compatible(scaffold_d_arm, arm_name)

    arm_geo = get_arm_geometry(arm_name)
    g_score = geometry_match_score(
        arm_geo, scaffold_d_arm, scaffold_theta_conv, guest.diameter_A
    )

    # Repulsion: how well does this arm differentially repel interferents?
    interferents = getattr(guest, 'interferent_species', []) or []
    r_score = _arm_repulsion_score(
        arm_donor_types,
        getattr(guest, 'target_species', guest.name) if hasattr(guest, 'target_species') else guest.name,
        interferents,
        cavity_radius_A=scaffold_d_arm / 2.0 if scaffold_d_arm > 0 else 3.0,
    )

    composite = (w_electronic * e_score +
                 w_pka * p_score +
                 w_geometry * g_score +
                 w_repulsion * r_score)

    # Steric veto
    if not is_steric_ok:
        composite *= 0.1

    return ArmScore(
        arm_name=arm_name,
        donor_types=arm_donor_types,
        hardness=arm_hardness,
        electronic_score=e_score,
        pka_score=p_score,
        steric_ok=is_steric_ok,
        geometry_score=g_score,
        repulsion_score=r_score,
        pka_penalty_kJ=p_penalty,
        composite=composite,
    )



# ═══════════════════════════════════════════════════════════════════════════
# Fallback Arm Registry (no rdkit needed)
# Used when ARM_LIBRARY can't be imported (rdkit absent).
# Minimal dataclass to mirror Arm interface.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _FallbackArm:
    name: str
    smiles: str
    donor_subtypes: list
    donor_element: str
    hardness: str
    category: str = ""

_FALLBACK_ARMS = [
    _FallbackArm("acetic-acid", "[*]CC(=O)O", ["O_carboxylate"], "O", "hard"),
    _FallbackArm("propionic-acid", "[*]CCC(=O)O", ["O_carboxylate"], "O", "hard"),
    _FallbackArm("ethanol", "[*]CCO", ["O_hydroxyl"], "O", "hard"),
    _FallbackArm("phenol", "[*]c1ccc(O)cc1", ["O_phenolate"], "O", "hard"),
    _FallbackArm("acetohydroxamate", "[*]CC(=O)NO", ["O_hydroxamate", "O_hydroxamate"], "O", "hard"),
    _FallbackArm("phosphonate", "[*]CP(=O)(O)O", ["O_phosphate"], "O", "hard"),
    _FallbackArm("2-pyridyl", "[*]c1ccccn1", ["N_pyridine"], "N", "borderline"),
    _FallbackArm("2-pyridylmethyl", "[*]Cc1ccccn1", ["N_pyridine"], "N", "borderline"),
    _FallbackArm("imidazolylmethyl", "[*]Cc1cnc[nH]1", ["N_imidazole"], "N", "borderline"),
    _FallbackArm("aminomethyl", "[*]CN", ["N_amine"], "N", "borderline"),
    _FallbackArm("aminoethyl", "[*]CCN", ["N_amine"], "N", "borderline"),
    _FallbackArm("nitrile", "[*]C#N", ["N_nitrile"], "N", "borderline"),
    _FallbackArm("thiol-methyl", "[*]CS", ["S_thiolate"], "S", "soft"),
    _FallbackArm("thiol-ethyl", "[*]CCS", ["S_thiolate"], "S", "soft"),
    _FallbackArm("thioether-methyl", "[*]CSC", ["S_thioether"], "S", "soft"),
    _FallbackArm("thiourea", "[*]NC(=S)N", ["S_thiolate", "N_amine"], "S", "soft"),
    _FallbackArm("urea-NH2", "[*]NC(=O)N", ["N_amine", "O_carbonyl"], "N", "hard"),
    _FallbackArm("guanidinium", "[*]NC(=N)N", ["N_amine"], "N", "hard"),
    _FallbackArm("pyrrole", "[*]c1ccc[nH]1", ["N_pyrrole"], "N", "borderline"),
    _FallbackArm("indolyl", "[*]c1c[nH]c2ccccc12", ["N_pyrrole"], "N", "borderline"),
    _FallbackArm("sulfonamide-NH", "[*]NS(=O)(=O)C", ["N_amine"], "N", "borderline"),
    _FallbackArm("naphthyl", "[*]c1cccc2ccccc12", [], "C", "borderline"),
    _FallbackArm("H-cap", "[*]C", [], "C", "borderline"),
    # New anion arms
    _FallbackArm("bis-urea", "[*]NC(=O)NC(=O)N", ["N_urea", "N_urea", "O_carbonyl", "O_carbonyl"], "N", "hard"),
    _FallbackArm("amidinium", "[*]C(=N)N", ["N_amine", "N_amine"], "N", "hard"),
    _FallbackArm("triazole-NH", "[*]c1cn[nH]n1", ["N_pyrrole", "N_pyridine"], "N", "borderline"),
    _FallbackArm("indole-amide", "[*]NC(=O)c1c[nH]c2ccccc12", ["N_amide", "N_pyrrole", "O_carbonyl"], "N", "borderline"),
    _FallbackArm("amino-thiourea", "[*]NC(=S)NN", ["N_amine", "N_amine", "S_thiolate"], "N", "borderline"),
    _FallbackArm("amino-pyridine", "[*]Nc1ccccn1", ["N_amine", "N_pyridine"], "N", "borderline"),
    _FallbackArm("hydroxamic-extended", "[*]CCC(=O)NO", ["O_hydroxamate", "O_hydroxamate"], "O", "hard"),
    _FallbackArm("chromone-hydroxyl", "[*]c1cc(=O)c2ccccc2o1", ["O_carbonyl", "O_hydroxyl"], "O", "hard"),
]


def _get_arm_library():
    """Get arm library — prefer de_novo_generator, fallback to builtin."""
    try:
        from core.de_novo_generator import ARM_LIBRARY
        return list(ARM_LIBRARY)
    except (ImportError, ModuleNotFoundError):
        return list(_FALLBACK_ARMS)


def select_arms(scaffold_desc, guest,
                arms=None, pH: float = None,
                top_n: int = 10) -> ArmAssignment:
    """Select optimal arms for a scaffold–guest pair.

    Scores all arms from ARM_LIBRARY against the scaffold geometry
    and guest requirements, returns ranked list and best assignment.

    Parameters
    ----------
    scaffold_desc : ScaffoldDescriptor
        Scaffold with extracted geometry (from Phase A).
    guest : GuestSpec
        Target guest specification.
    arms : list, optional
        Override arm library. If None, imports ARM_LIBRARY.
    pH : float, optional
        Working pH. If None, uses guest.pH.
    top_n : int
        Number of top arms to return.

    Returns
    -------
    ArmAssignment
        Ranked arms and best combination.
    """
    if arms is None:
        arms = _get_arm_library()

    if pH is None:
        pH = guest.pH

    d_arm = scaffold_desc.d_arm_A
    theta = scaffold_desc.theta_conv_deg
    n_sites = scaffold_desc.n_sites

    # Score all arms
    scored = []
    for arm in arms:
        s = score_arm(
            arm.name, arm.donor_subtypes, arm.hardness,
            guest, d_arm, theta, pH
        )
        scored.append(s)

    # Sort by composite (best first)
    scored.sort(key=lambda s: s.composite, reverse=True)

    # Best arm per site: for now, use top N distinct arms
    # (could do combinatorial optimization for multi-site, but greedy is fine)
    best = []
    seen_types = set()
    for s in scored:
        if not s.viable:
            continue
        # Prefer diversity: don't repeat same donor element unless it's top-1
        if len(best) >= n_sites:
            break
        best.append(s.arm_name)

    # Fill remaining sites if not enough viable arms
    while len(best) < n_sites:
        best.append(scored[0].arm_name if scored else "H-cap")

    combo_score = sum(s.composite for s in scored[:n_sites]) / max(1, n_sites)

    return ArmAssignment(
        scaffold_name=scaffold_desc.name,
        guest_name=guest.name,
        arm_scores=scored[:top_n],
        best_arms=best[:n_sites],
        best_combo_score=combo_score,
        n_sites=n_sites,
        pH=pH,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Batch Selection (for design_scaffold integration)
# ═══════════════════════════════════════════════════════════════════════════

def select_arms_for_rankings(rankings, guest,
                              arms=None, pH: float = None) -> List[ArmAssignment]:
    """Select arms for each scaffold in a ranking list.

    Parameters
    ----------
    rankings : list of ScaffoldRanking
        From rank_scaffolds_for_guest().
    guest : GuestSpec
    arms : list, optional
    pH : float, optional

    Returns
    -------
    List of ArmAssignment, same order as rankings.
    """
    assignments = []
    for r in rankings:
        a = select_arms(r.descriptor, guest, arms=arms, pH=pH)
        assignments.append(a)
    return assignments


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_arm_assignment(assignment: ArmAssignment, top_n: int = 8):
    """Print arm assignment summary."""
    print(f"\n  Arm Selection: {assignment.scaffold_name} → {assignment.guest_name} "
          f"(pH {assignment.pH})")
    print(f"  Best combo: [{', '.join(assignment.best_arms)}] "
          f"(score={assignment.best_combo_score:.3f})")

    print(f"\n  {'Rank':>4} {'Arm':<25} {'Elec':>5} {'pKa':>5} {'Geom':>5} "
          f"{'Repul':>5} {'Total':>6} {'Steric':>6} {'pKa kJ':>7}")
    print("  " + "-" * 88)

    for i, s in enumerate(assignment.arm_scores[:top_n]):
        steric_str = "OK" if s.steric_ok else "FAIL"
        print(f"  {i+1:4d} {s.arm_name:<25} {s.electronic_score:5.2f} "
              f"{s.pka_score:5.2f} {s.geometry_score:5.2f} "
              f"{s.repulsion_score:5.2f} "
              f"{s.composite:6.3f} {steric_str:>6} {s.pka_penalty_kJ:+7.1f}")


if __name__ == "__main__":
    from core.scaffold_designer import ScaffoldDescriptor, GuestSpec

    print("=" * 70)
    print("Arm Selector — Standalone Demo")
    print("=" * 70)

    # Test: selenite at pH 5
    sd = ScaffoldDescriptor(
        name="isophthalamide", smiles="", category="aromatic", n_sites=2,
        d_arm_A=6.0, theta_conv_deg=55.0, rigidity_index=0.7,
        n_rotors_bridge=2, V_cavity_est_A3=70.0,
        n_backbone_hb_donors=2, n_backbone_hb_acceptors=3,
    )
    guest = GuestSpec(name="selenite", diameter_A=3.5, volume_A3=45.0,
                      charge=-2, n_hb_acceptors=3, pH=5.0)

    assignment = select_arms(sd, guest)
    print_arm_assignment(assignment, top_n=12)
