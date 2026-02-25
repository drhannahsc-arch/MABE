"""
realization_ranker/geometric_fidelity/biopolymer.py

Geometric fidelity scorers for biological and framework material systems.

Each function answers: "Can this material system reproduce the donor geometry
that Layer 2 specified?"

Physics basis per system:
  - Protein: Ramachandran + loop polymer physics + side chain donor reach
  - DNA origami: B-form helix geometry + staple positional precision
  - MOF: Node coordination geometry + linker length
  - Aptamer: NUPACK folding thermodynamics (low confidence for 3D)
"""

import math
from typing import Optional

from ..epistemic import EpistemicScore, EpistemicBasis
from .material_geometry import (
    PROTEIN_CA_CA_DISTANCE_NM,
    POSITIONAL_PRECISION_NM,
    SIDECHAIN_DONOR_REACH_NM,
    DONOR_TO_RESIDUE,
    DNA_STAPLE_POSITION_SIGMA_NM,
    DNA_HELIX_DIAMETER_NM,
    MOF_NODE_GEOMETRIES,
    NATURAL_CAVITY_RANGE_NM,
    MAX_SIMULTANEOUS_DONORS,
)


# ═══════════════════════════════════════════════════════════════════════════
# PROTEIN
# ═══════════════════════════════════════════════════════════════════════════

def score_protein_geometric_fidelity(
    required_donor_distances_nm: list[float],
    required_donor_types: list[str],
    required_cavity_nm: float,
) -> EpistemicScore:
    """
    Score whether a protein fold can present the required donor geometry.

    Physics:
      - Loop spanning: d_max = 0.38n nm (extended), d_typical = 0.38√n nm (coil)
      - Side chain reach: each residue type has a known reach from Cα
      - Multiple donors converging: need enough loop length to span all pairs

    Parameters
    ----------
    required_donor_distances_nm : list[float]
        Pairwise distances between required donor positions (from Layer 2).
    required_donor_types : list[str]
        Donor subtypes needed (e.g. "O_carboxylate", "N_imidazole").
    required_cavity_nm : float
        Cavity diameter from Layer 2.
    """

    # Check 1: Can any residues provide each required donor?
    donor_feasibility = []
    for dt in required_donor_types:
        residues = DONOR_TO_RESIDUE.get(dt, [])
        if residues:
            # Best reach among providing residues
            max_reach = max(SIDECHAIN_DONOR_REACH_NM.get(r, 0.3) for r in residues)
            donor_feasibility.append(1.0)
        else:
            donor_feasibility.append(0.0)  # No natural amino acid provides this

    if not donor_feasibility:
        return EpistemicScore(
            value=0.0,
            basis=EpistemicBasis.PHYSICS_DERIVED,
            equation="No donors requested",
            uncertainty=0.0,
        )

    donor_score = sum(donor_feasibility) / len(donor_feasibility)

    # Check 2: Can loops span the required distances?
    # For each pairwise distance, compute minimum residues needed
    span_scores = []
    for d_nm in required_donor_distances_nm:
        if d_nm <= 0:
            span_scores.append(1.0)
            continue
        # Random coil model: n_residues = (d / 0.38)²
        n_needed_coil = (d_nm / PROTEIN_CA_CA_DISTANCE_NM) ** 2
        # Extended: n_residues = d / 0.38
        n_needed_extended = d_nm / PROTEIN_CA_CA_DISTANCE_NM

        if n_needed_extended <= 5:
            span_scores.append(1.0)    # Very achievable
        elif n_needed_coil <= 30:
            span_scores.append(0.9)    # Normal loop
        elif n_needed_coil <= 100:
            span_scores.append(0.6)    # Long loop, less certain
        else:
            span_scores.append(0.3)    # Would need a large domain

    span_score = min(span_scores) if span_scores else 1.0  # Weakest link

    # Check 3: Cavity size within protein range
    cav_range = NATURAL_CAVITY_RANGE_NM["protein"]
    if cav_range[0] <= required_cavity_nm <= cav_range[1]:
        cav_score = 1.0
    else:
        dist_from_range = min(
            abs(required_cavity_nm - cav_range[0]),
            abs(required_cavity_nm - cav_range[1]),
        )
        cav_score = math.exp(-(dist_from_range ** 2) / (2 * 0.5 ** 2))

    fidelity = 0.4 * donor_score + 0.3 * span_score + 0.3 * cav_score

    return EpistemicScore(
        value=min(1.0, fidelity),
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=(
            "fidelity = 0.4×donor_availability + 0.3×loop_spanning "
            "+ 0.3×cavity_size_match. Loop: n = (d/0.38)² (coil model)"
        ),
        uncertainty=0.10,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DNA ORIGAMI
# ═══════════════════════════════════════════════════════════════════════════

def score_dna_origami_geometric_fidelity(
    required_donor_distances_nm: list[float],
    required_cavity_nm: float,
    required_donor_count: int,
) -> EpistemicScore:
    """
    Score whether DNA origami can position functional groups at required geometry.

    Physics:
      - Staple positioning precision: σ ≈ 1.5 nm
      - Minimum feature spacing: ~2.25 nm (helix diameter)
      - Interior cavity limited by cage design (2–50 nm)
      - Donors are conjugated to staple extensions, not native to DNA

    Note: DNA origami positions donor-carrying molecules, not donors directly.
    Geometric fidelity depends on whether required spacings are compatible
    with helical geometry constraints.
    """

    sigma = DNA_STAPLE_POSITION_SIGMA_NM
    min_spacing = DNA_HELIX_DIAMETER_NM  # Can't be closer than one helix

    # Check 1: Are required spacings achievable?
    spacing_scores = []
    for d_nm in required_donor_distances_nm:
        if d_nm < min_spacing:
            # Required spacing below physical minimum
            # Penalty based on how far below
            penalty = math.exp(-((min_spacing - d_nm) ** 2) / (2 * 0.5 ** 2))
            spacing_scores.append(penalty)
        else:
            # Gaussian uncertainty from positional precision
            # Can we hit d_nm ± sigma?
            relative_error = sigma / d_nm if d_nm > 0 else 1.0
            spacing_scores.append(math.exp(-(relative_error ** 2) / 2))

    spacing_score = min(spacing_scores) if spacing_scores else 1.0

    # Check 2: Cavity size
    cav_range = NATURAL_CAVITY_RANGE_NM["DNA_origami"]
    if cav_range[0] <= required_cavity_nm <= cav_range[1]:
        cav_score = 1.0
    elif required_cavity_nm < cav_range[0]:
        cav_score = 0.1  # DNA origami can't form sub-2nm cavities
    else:
        cav_score = 0.5  # Very large cavities possible but less controlled

    # Check 3: Donor count — each needs a staple extension
    max_d = MAX_SIMULTANEOUS_DONORS["DNA_origami"]
    if required_donor_count <= max_d:
        donor_score = 1.0
    else:
        donor_score = max(0.2, max_d / required_donor_count)

    fidelity = 0.5 * spacing_score + 0.3 * cav_score + 0.2 * donor_score

    return EpistemicScore(
        value=min(1.0, fidelity),
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=(
            f"Staple precision σ={sigma} nm. "
            f"Min spacing={min_spacing} nm (helix diameter). "
            "fidelity = 0.5×spacing + 0.3×cavity + 0.2×donors"
        ),
        uncertainty=0.15,
        note="Assumes standard DX wireframe origami. Solid-wall designs have tighter precision.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# MOF LATTICE
# ═══════════════════════════════════════════════════════════════════════════

def score_mof_geometric_fidelity(
    required_cavity_nm: float,
    required_donor_count: int,
    required_coordination_geometry: Optional[str] = None,
) -> EpistemicScore:
    """
    Score whether a MOF lattice can reproduce the required pocket geometry.

    Physics:
      - Node geometry fixed by metal SBU (secondary building unit)
      - Pore size determined by linker length (calculable from molecular mechanics)
      - Crystallographic precision: ± 0.005 nm

    Data sources:
      - CoRE MOF 2019 database (free, ~14K structures)
      - RCSR for topology enumeration
      - Zeo++ for pore analysis from CIF
    """

    best_match_score = 0.0
    best_match_name = "none"

    for sbu_name, (coord_num, geom_label, typical_pore) in MOF_NODE_GEOMETRIES.items():
        # Pore size match
        pore_diff = abs(required_cavity_nm - typical_pore)
        pore_score = math.exp(-(pore_diff ** 2) / (2 * 0.3 ** 2))

        # Coordination geometry match
        if required_coordination_geometry:
            geom_score = 1.0 if geom_label == required_coordination_geometry else 0.3
        else:
            geom_score = 0.7  # Unknown requirement, partial credit

        # Donor count vs node coordination number
        if required_donor_count <= coord_num:
            donor_score = 1.0
        else:
            donor_score = max(0.2, coord_num / required_donor_count)

        combined = 0.4 * pore_score + 0.3 * geom_score + 0.3 * donor_score

        if combined > best_match_score:
            best_match_score = combined
            best_match_name = sbu_name

    return EpistemicScore(
        value=min(1.0, best_match_score),
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=(
            "Scored against known SBU archetypes. "
            "Pore match = exp(-Δd²/0.18). "
            f"Best match: {best_match_name}"
        ),
        data_source="MOF_NODE_GEOMETRIES (from published crystal structures); "
                     "expand via CoRE MOF 2019 / COD API queries",
        uncertainty=0.12,
        note="Scored against 6 archetype SBUs. Live CoRE MOF query would improve coverage.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# APTAMER / DNAZYME
# ═══════════════════════════════════════════════════════════════════════════

def score_aptamer_geometric_fidelity(
    required_cavity_nm: float,
    required_donor_count: int,
) -> EpistemicScore:
    """
    Score aptamer geometric fidelity.

    Honest assessment: aptamer 3D structure prediction is LOW CONFIDENCE.
    NUPACK gives good 2D folding thermodynamics, but 3D positioning of
    donor atoms in loops/bulges remains poorly predicted for novel sequences.

    This is flagged as heuristic. Physics basis exists for 2D, not 3D.
    """

    # Cavity check — aptamers form pockets in the 0.5–3 nm range
    cav_range = NATURAL_CAVITY_RANGE_NM["aptamer"]
    if cav_range[0] <= required_cavity_nm <= cav_range[1]:
        cav_score = 0.7  # Can form cavities in range, but 3D control is uncertain
    else:
        cav_score = 0.2

    # Donor count — limited by loop/bulge size
    max_d = MAX_SIMULTANEOUS_DONORS["aptamer"]
    if required_donor_count <= max_d:
        donor_score = 0.6  # Can present donors but positioning uncertain
    else:
        donor_score = 0.2

    fidelity = 0.5 * cav_score + 0.5 * donor_score

    return EpistemicScore(
        value=min(1.0, fidelity),
        basis=EpistemicBasis.HEURISTIC_ESTIMATE,
        equation="Cavity range check + donor count feasibility",
        data_source="NUPACK 2D folding thermodynamics (3D extrapolation uncertain)",
        uncertainty=0.25,
        note=(
            "Best guess, more data required. Aptamer 3D structure prediction "
            "is low confidence for novel sequences. NUPACK gives reliable 2D "
            "folding ΔG but not 3D donor positions."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH — route to appropriate scorer by realization type
# ═══════════════════════════════════════════════════════════════════════════

def score_geometric_fidelity(
    realization_type: str,
    required_cavity_nm: float,
    required_donor_count: int,
    required_donor_types: list[str] = None,
    required_donor_distances_nm: list[float] = None,
    required_coordination_geometry: str = None,
) -> EpistemicScore:
    """
    Dispatch geometric fidelity scoring to the appropriate material scorer.

    This is the main entry point from the ranker.
    """

    if required_donor_types is None:
        required_donor_types = []
    if required_donor_distances_nm is None:
        required_donor_distances_nm = []

    if realization_type in ("protein", "antibody_CDR", "peptide"):
        return score_protein_geometric_fidelity(
            required_donor_distances_nm,
            required_donor_types,
            required_cavity_nm,
        )
    elif realization_type == "DNA_origami":
        return score_dna_origami_geometric_fidelity(
            required_donor_distances_nm,
            required_cavity_nm,
            required_donor_count,
        )
    elif realization_type == "MOF":
        return score_mof_geometric_fidelity(
            required_cavity_nm,
            required_donor_count,
            required_coordination_geometry,
        )
    elif realization_type in ("aptamer", "dnazyme"):
        return score_aptamer_geometric_fidelity(
            required_cavity_nm,
            required_donor_count,
        )
    else:
        # Use class-level scorer for all other types
        from .small_molecule import score_class_geometric_fidelity
        return score_class_geometric_fidelity(
            realization_type,
            required_cavity_nm,
            required_donor_count,
            set(required_donor_types),
        )