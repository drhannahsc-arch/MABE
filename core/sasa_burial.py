"""
core/sasa_burial.py -- Sprint 40: Per-Compound SASA Burial Fraction

Problem: annotate_protein_pocket() applies a flat sasa_burial_fraction
to all compounds in a target. A 150 Da fragment and a 600 Da peptidomimetic
get the same burial. This kills within-target ranking because the
hydrophobic term can't differentiate.

Fix: Modulate burial fraction by:
  1. Compound SASA relative to pocket surface area (from volume^(2/3))
  2. Molecular shape (compact = better burial, elongated = worse)
  3. Flexibility penalty (more rotors = conformational mismatch)

The result: each compound gets a unique sasa_buried_A2 value,
enabling the hydrophobic term to rank compounds within a target.
"""

import math


def estimate_per_compound_burial(uc, pocket=None):
    """Add per-compound CORRECTION to existing SASA burial.

    The flat burial (pocket_fraction * guest_sasa_nonpolar) captures
    the first-order effect well. This function adds a second-order
    correction based on packing coefficient — the one compound-specific
    property that captures pocket fit quality.

    Design: multiplicative correction centered on 1.0.
    - Well-fitting PC (~0.55): burial *= 1.10 (10% bonus)
    - Poor fit (very small/large): burial *= 0.85 (15% penalty)
    - This preserves the existing signal while adding discrimination.

    Args:
        uc: UniversalComplex with sasa_buried_A2 already set
        pocket: ProteinPocket (optional, not currently used)

    Returns:
        uc with adjusted sasa_buried_A2
    """
    if uc.sasa_buried_A2 <= 0:
        return uc
    if uc.binding_mode != "protein_ligand":
        return uc

    # Get packing coefficient (set by fast_enrich + annotate_protein_pocket)
    pc = uc.packing_coefficient
    if pc <= 0 or uc.cavity_volume_A3 <= 0:
        return uc

    # Gaussian correction centered on optimal PC
    pc_optimal = 0.55
    sigma_pc = 0.20
    pc_match = math.exp(-((pc - pc_optimal) ** 2) / (2 * sigma_pc ** 2))

    # Correction: 0.85 at poor fit, 1.10 at optimal fit
    correction = 0.85 + 0.25 * pc_match

    uc.sasa_buried_A2 = round(uc.sasa_buried_A2 * correction, 1)

    return uc
