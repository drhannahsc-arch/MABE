"""
realization_ranker/scale_feasibility.py

Scale feasibility scoring — can this be manufactured at the required scale?

Honest assessment: this axis is the MOST heuristic in Layer 3.
Physics can bound some quantities (diffusion-limited production rates,
coupling yields at scale) but practical manufacturing feasibility is
largely engineering knowledge, not first-principles physics.

Every score here is tagged with its epistemic basis.
"""

import math
from typing import Optional

from .epistemic import EpistemicScore, EpistemicBasis


# ═══════════════════════════════════════════════════════════════════════════
# Scale targets — what quantity does the application need?
# ═══════════════════════════════════════════════════════════════════════════

SCALE_TIERS = {
    "research":     1e-6,    # mg quantities
    "diagnostic":   1e-3,    # g quantities (per device)
    "therapeutic":  1e-1,    # 100 mg per dose × thousands of doses
    "pilot":        1e0,     # kg quantities
    "industrial":   1e3,     # tonnes
}


# ═══════════════════════════════════════════════════════════════════════════
# Class-level scale feasibility — honestly flagged
# ═══════════════════════════════════════════════════════════════════════════

def score_scale_feasibility(
    realization_type: str,
    target_scale: str = "pilot",
    oligo_length: Optional[int] = None,
    staple_count: Optional[int] = None,
) -> EpistemicScore:
    """
    Score manufacturing scale feasibility.

    Parameters
    ----------
    realization_type : str
        Material system class.
    target_scale : str
        One of: "research", "diagnostic", "therapeutic", "pilot", "industrial"
    oligo_length : int, optional
        For nucleic acid realizations — enables yield-at-scale calculation.
    staple_count : int, optional
        For DNA origami — enables staple production estimate.
    """

    scale_kg = SCALE_TIERS.get(target_scale, 1.0)

    # ─── Physics-grounded cases ───

    # Oligonucleotides: coupling yield × synthesis throughput is calculable
    if realization_type in ("aptamer", "dnazyme") and oligo_length:
        return _score_oligo_scale(oligo_length, scale_kg)

    # DNA origami: staple count × oligo cost/throughput
    if realization_type == "DNA_origami":
        return _score_origami_scale(staple_count or 200, scale_kg)

    # ─── Heuristic cases ───
    return _class_level_scale_score(realization_type, target_scale)


def _score_oligo_scale(oligo_length: int, scale_kg: float) -> EpistemicScore:
    """
    Oligo synthesis scale — partially physics-based.

    Modern phosphoramidite synthesis:
    - Column: ~1 μmol/synthesis → ~10 mg of 50-mer
    - Large-scale: up to 1 mmol → ~10 g per batch
    - Industrial (Twist/IDT): can produce kg quantities of short oligos
    """
    # Mass per synthesis (g) at 1 μmol scale
    mw_per_nt = 330  # g/mol average
    mass_per_umol = oligo_length * mw_per_nt * 1e-6  # grams

    # Scale factor: how many synthesis batches needed?
    batches_needed = (scale_kg * 1000) / mass_per_umol  # at 1 μmol scale
    # Large-scale can do 1000× per batch
    batches_at_large_scale = batches_needed / 1000

    if batches_at_large_scale < 1:
        score = 0.95  # Single large-scale batch
    elif batches_at_large_scale < 10:
        score = 0.8
    elif batches_at_large_scale < 100:
        score = 0.5
    else:
        score = 0.2  # Would require industrial oligo production

    return EpistemicScore(
        value=score,
        basis=EpistemicBasis.PHYSICS_DERIVED,
        equation=(
            f"MW={oligo_length * mw_per_nt} g/mol. "
            f"Need {scale_kg:.1g} kg → {batches_at_large_scale:.0f} large-scale batches. "
            "Coupling yield factored separately in SA."
        ),
        uncertainty=0.10,
        note="Assumes access to commercial oligo synthesis (IDT/Twist).",
    )


def _score_origami_scale(staple_count: int, scale_kg: float) -> EpistemicScore:
    """
    DNA origami scale — mostly heuristic with some physics grounding.

    Current state of the art:
    - Lab scale: μg to low mg (sufficient for diagnostics)
    - Biotechnological production: mg scale demonstrated (Praetorius et al. 2017)
    - kg scale: not yet demonstrated for origami
    """
    # Each origami structure ≈ 5 MDa → 5e6 g/mol
    # 1 μmol of origami = 5 kg — this is enormous
    # Typical lab production: ~1 nmol → ~5 μg

    if scale_kg < 1e-6:
        score = 0.8   # Research: mg scale is achievable
    elif scale_kg < 1e-3:
        score = 0.5   # g scale: push the boundary
    else:
        score = 0.15  # kg+ scale: not demonstrated

    # More staples = harder to produce at scale (more oligos to order)
    staple_penalty = math.exp(-staple_count / 500)
    score *= (0.7 + 0.3 * staple_penalty)

    return EpistemicScore(
        value=min(1.0, score),
        basis=EpistemicBasis.HEURISTIC_ESTIMATE,
        equation=f"staples={staple_count}, target={scale_kg:.1g} kg",
        note=(
            "Best guess, more data required. DNA origami at kg scale is undemonstrated. "
            "Biotechnological production (phage-based) may enable larger scales."
        ),
        uncertainty=0.25,
    )


def _class_level_scale_score(realization_type: str, target_scale: str) -> EpistemicScore:
    """Heuristic class-level scale scores — all flagged."""

    # Scale score matrix: (realization_type, target_scale) → score
    # Higher = more feasible at that scale
    SCALE_MATRIX = {
        #                       research  diagnostic  therapeutic  pilot  industrial
        "small_molecule":       (0.95,    0.90,       0.85,       0.80,  0.70),
        "chelator":             (0.95,    0.90,       0.85,       0.80,  0.75),
        "porphyrin":            (0.90,    0.85,       0.70,       0.60,  0.40),
        "crown_ether":          (0.95,    0.90,       0.80,       0.75,  0.60),
        "peptide":              (0.90,    0.85,       0.70,       0.50,  0.20),
        "protein":              (0.80,    0.60,       0.50,       0.30,  0.10),
        "antibody_CDR":         (0.75,    0.50,       0.40,       0.20,  0.05),
        "aptamer":              (0.95,    0.90,       0.80,       0.60,  0.30),
        "dnazyme":              (0.95,    0.90,       0.80,       0.60,  0.30),
        "DNA_origami":          (0.70,    0.40,       0.15,       0.05,  0.01),
        "MOF":                  (0.85,    0.70,       0.50,       0.40,  0.25),
        "crystal":              (0.80,    0.60,       0.40,       0.30,  0.20),
        "ion_exchange_resin":   (0.95,    0.95,       0.90,       0.90,  0.85),
    }

    scale_idx = {
        "research": 0, "diagnostic": 1, "therapeutic": 2,
        "pilot": 3, "industrial": 4,
    }.get(target_scale, 3)

    scores = SCALE_MATRIX.get(realization_type)
    if scores is None:
        return EpistemicScore(
            value=0.5,
            basis=EpistemicBasis.HEURISTIC_ESTIMATE,
            note=f"Unknown realization type: {realization_type}. Best guess, more data required.",
            uncertainty=0.3,
        )

    return EpistemicScore(
        value=scores[scale_idx],
        basis=EpistemicBasis.HEURISTIC_ESTIMATE,
        equation=f"Class-level estimate for {realization_type} at {target_scale} scale",
        note=(
            "Best guess, more data required. Scale feasibility depends on specific "
            "design, not just material class. Scores based on current manufacturing "
            "state-of-the-art as of 2024-2025."
        ),
        uncertainty=0.20,
    )