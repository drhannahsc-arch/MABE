"""
core/coordination_generator_v2.py — Sprint 16 rework: Diverse Coordination Generation

Fixes:
1. HSAB softness from polarizability module (continuous, not categorical)
2. Multiple donor strategies per metal (not just one)
3. Forced diversity: hard, borderline, soft, and mixed donor sets
4. Dedup by (donor_set_hash, scaffold_type)

Replaces coordination_generator.py in the pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional
import itertools

from core.polarizability import compute_full_polarization


# ═══════════════════════════════════════════════════════════════════════════
# METAL DATABASE (unchanged structure, updated softness integration)
# ═══════════════════════════════════════════════════════════════════════════

# Keep the original lookups for properties that don't change
METAL_D_ELECTRONS = {
    "Na+": 0, "K+": 0, "Ca2+": 0, "Mg2+": 0, "Ba2+": 0, "Al3+": 0,
    "Cr3+": 3, "Mn2+": 5, "Fe2+": 6, "Fe3+": 5, "Co2+": 7, "Co3+": 6,
    "Ni2+": 8, "Cu2+": 9, "Cu+": 10, "Zn2+": 10, "Pd2+": 8, "Ag+": 10,
    "Cd2+": 10, "La3+": 0, "Ce3+": 1, "Gd3+": 0, "Au+": 10, "Au3+": 8, "Pt2+": 8,
    "Hg2+": 10, "Pb2+": 0, "Tl+": 0, "Bi3+": 0, "UO2_2+": 0,
    "Ru3+": 5, "Rh3+": 6, "Ir3+": 6, "Os3+": 5,
}

PREFERRED_CN = {
    "Na+": [6], "K+": [8], "Ca2+": [6, 8], "Mg2+": [6], "Ba2+": [8],
    "Al3+": [6], "Cr3+": [6], "Mn2+": [6], "Fe2+": [6], "Fe3+": [6],
    "Co2+": [6, 4], "Co3+": [6], "Ni2+": [6, 4], "Cu2+": [4, 6],
    "Cu+": [4, 2], "Zn2+": [4, 6], "Pd2+": [4], "Pt2+": [4],
    "Ag+": [2, 4], "Cd2+": [6, 4], "Au+": [2], "Au3+": [4],
    "Hg2+": [2, 4], "Pb2+": [4, 6], "Tl+": [6], "Bi3+": [6],
    "UO2_2+": [6], "La3+": [9, 8], "Ce3+": [9, 8], "Gd3+": [8, 9],
    "Ru3+": [6], "Rh3+": [6], "Ir3+": [6],
}

SPECIAL_EFFECTS = {
    "Cu2+": ["jahn_teller"], "Cu+": ["linear_preference"],
    "Pb2+": ["hemidirected_lone_pair", "stereochemically_active_6s2"],
    "Tl+": ["hemidirected_lone_pair"], "Bi3+": ["hemidirected_lone_pair"],
    "Pd2+": ["strong_square_planar"], "Pt2+": ["strong_square_planar"],
    "Au3+": ["strong_square_planar"], "Au+": ["linear_preference"],
    "Hg2+": ["linear_preference"], "Ag+": ["linear_preference"],
    "UO2_2+": ["trans_dioxo_equatorial"],
}

_IONIC_RADII = {
    "Na+": 102, "K+": 138, "Ca2+": 100, "Mg2+": 72, "Ba2+": 135,
    "Al3+": 54, "Cr3+": 62, "Mn2+": 83, "Fe2+": 78, "Fe3+": 65,
    "Co2+": 75, "Co3+": 55, "Ni2+": 69, "Cu2+": 73, "Cu+": 77,
    "Zn2+": 74, "Pd2+": 86, "Pt2+": 80, "Ag+": 115, "Cd2+": 95,
    "Au+": 137, "Au3+": 85, "Hg2+": 102, "Pb2+": 119, "Tl+": 150,
    "Bi3+": 103, "La3+": 103, "Ce3+": 101, "Gd3+": 94, "UO2_2+": 73,
    "Ru3+": 68, "Rh3+": 67, "Ir3+": 68,
}

# HSAB: use as FALLBACK only — primary softness comes from polarizability
METAL_HSAB_SOFTNESS = {
    "Na+": 0.05, "K+": 0.04, "Ca2+": 0.08, "Mg2+": 0.06, "Ba2+": 0.07,
    "Al3+": 0.10, "Cr3+": 0.18, "Mn2+": 0.20, "Fe2+": 0.22, "Fe3+": 0.12,
    "Co2+": 0.25, "Co3+": 0.15, "Ni2+": 0.24, "Cu2+": 0.35, "Cu+": 0.65,
    "Zn2+": 0.28, "Pd2+": 0.70, "Pt2+": 0.72, "Ag+": 0.75, "Cd2+": 0.50,
    "Au+": 0.90, "Au3+": 0.85, "Hg2+": 0.85, "Pb2+": 0.55, "Tl+": 0.50,
    "Bi3+": 0.45, "La3+": 0.08, "Ce3+": 0.09, "Gd3+": 0.10, "UO2_2+": 0.15,
}

LFSE_GEOMETRY_PREFERENCES = {
    0: [("flexible", 0.0, 0.0)],
    3: [("octahedral", 96.0, 25.0)],
    5: [("octahedral", 0.0, 5.0)],
    6: [("octahedral", 48.0, 10.0)],
    7: [("octahedral", 96.0, 15.0)],
    8: [("square_planar", 240.0, 60.0), ("octahedral", 144.0, 30.0)],
    9: [("tetragonal_elongated", 132.0, 35.0), ("square_planar", 144.0, 40.0)],
    10: [("tetrahedral", 0.0, 0.0), ("octahedral", 0.0, 0.0)],
}


# ═══════════════════════════════════════════════════════════════════════════
# DONOR STRATEGY ENGINE — generates DIVERSE donor sets
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DonorRequirement:
    donor_atom: str
    bond_length_A: float
    bond_length_tolerance_A: float
    hsab_softness: float
    required: bool

@dataclass
class CoordinationEnvironment:
    target_identity: str
    target_formula: str
    coordination_number: int
    geometry: str
    donors: list
    lfse_stabilization_kj: float
    geometry_preference_kj: float
    hsab_match_score: float
    charge: int
    special_constraints: list
    rationale: str
    rank: int = 0
    donor_strategy: str = ""  # NEW: "hard", "borderline", "soft", "mixed_NO", "mixed_NS", "mixed_OS"


_DONOR_COV_RADII = {"O": 0.66, "N": 0.71, "S": 1.05, "P": 1.07, "Cl": 0.99}
_DONOR_SOFTNESS = {"O": 0.15, "N": 0.40, "S": 0.80, "P": 0.75}


def _get_continuous_softness(formula, d_electrons=0):
    """Get continuous softness from polarizability module."""
    try:
        pol = compute_full_polarization(formula, ["N", "N"], d_electrons=d_electrons)
        return pol.softness_continuous
    except Exception:
        return METAL_HSAB_SOFTNESS.get(formula, 0.3)


def _make_donors(atoms, ionic_radius_pm):
    """Create DonorRequirement list from atom symbols."""
    donors = []
    for a in atoms:
        bl = (ionic_radius_pm + _DONOR_COV_RADII.get(a, 0.70) * 100) / 100.0
        donors.append(DonorRequirement(a, round(bl, 2), 0.15,
                                        _DONOR_SOFTNESS.get(a, 0.3), True))
    return donors


def _hsab_match(metal_softness, donors):
    avg = sum(_DONOR_SOFTNESS.get(d, 0.3) for d in donors) / max(1, len(donors))
    return round(1.0 - abs(metal_softness - avg), 3)


def _resolve_geometry(base_geom, cn, specials):
    """Resolve geometry from LFSE preference + special effects."""
    if "strong_square_planar" in specials and cn == 4:
        return "square_planar"
    if "linear_preference" in specials and cn == 2:
        return "linear"
    if "hemidirected_lone_pair" in specials:
        if cn <= 4: return "hemidirected_seesaw"
        return "hemidirected_octahedral"
    if "jahn_teller" in specials:
        return "tetragonal_elongated" if cn == 6 else base_geom
    if "trans_dioxo_equatorial" in specials:
        return "pentagonal_bipyramidal_equatorial"
    cn_geom = {2: "linear", 4: "tetrahedral", 6: "octahedral", 8: "square_antiprismatic", 9: "tricapped_trigonal"}
    if base_geom in ("flexible",):
        return cn_geom.get(cn, "octahedral")
    return base_geom


def generate_coordination_environments(target_identity, target_formula, charge=2,
    d_electrons=None, hsab_softness=None, ionic_radius_pm=None,
    special_effects_override=None, max_candidates=6):
    """Generate DIVERSE coordination environments for a target metal.

    Key improvement: generates multiple DONOR STRATEGIES spanning the
    relevant chemical space, not just the single best-match strategy.
    """
    if d_electrons is None:
        d_electrons = METAL_D_ELECTRONS.get(target_formula, 0)
    if ionic_radius_pm is None:
        ionic_radius_pm = _IONIC_RADII.get(target_formula, 80.0)
    specials = special_effects_override or SPECIAL_EFFECTS.get(target_formula, [])
    preferred_cns = PREFERRED_CN.get(target_formula, [6])
    lfse_prefs = LFSE_GEOMETRY_PREFERENCES.get(d_electrons, [("flexible", 0.0, 0.0)])

    # Use continuous softness from polarizability, not categorical HSAB
    softness = hsab_softness if hsab_softness is not None else _get_continuous_softness(target_formula, d_electrons)

    candidates = []
    seen = set()  # Dedup key: (strategy, cn, geometry)

    # === STRATEGY 1: Best-match homogeneous ===
    if softness >= 0.6:
        primary_strategies = [("soft", ["S"]), ("mixed_NS", ["N", "S"]), ("borderline", ["N"])]
    elif softness >= 0.3:
        primary_strategies = [("borderline", ["N"]), ("mixed_NO", ["N", "O"]), ("mixed_NS", ["N", "S"])]
    elif softness >= 0.15:
        primary_strategies = [("hard", ["O"]), ("mixed_NO", ["N", "O"]), ("borderline", ["N"])]
    else:
        primary_strategies = [("hard", ["O"]), ("mixed_NO", ["N", "O"])]

    # === Always include at least one of each relevant strategy ===
    # For soft metals: also try hard donors (EDTA binds everything)
    if softness >= 0.4:
        if ("hard", ["O"]) not in primary_strategies:
            primary_strategies.append(("hard", ["O"]))
    # For hard metals: also try borderline (amines coordinate well)
    if softness < 0.3:
        if ("borderline", ["N"]) not in primary_strategies:
            primary_strategies.append(("borderline", ["N"]))

    for strategy_name, donor_pool in primary_strategies:
        for cn in preferred_cns[:2]:  # Top 2 CNs
            for geom_base, lfse_kj, pref_kj in lfse_prefs[:2]:  # Top 2 geometries
                geom = _resolve_geometry(geom_base, cn, specials)
                if geom is None:
                    continue

                dedup_key = (strategy_name, cn, geom)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Build donor atoms
                if len(donor_pool) == 1:
                    atoms = donor_pool * cn
                else:
                    # Mixed: distribute evenly
                    n_a = cn // 2
                    n_b = cn - n_a
                    atoms = [donor_pool[0]] * n_a + [donor_pool[1]] * n_b

                donors = _make_donors(atoms, ionic_radius_pm)
                match = _hsab_match(softness, atoms)

                rationale = (f"{strategy_name.capitalize()} strategy for {target_formula} "
                             f"(softness={softness:.2f}): {'+'.join(set(atoms))} donors, "
                             f"CN={cn}, {geom}")

                candidates.append(CoordinationEnvironment(
                    target_identity, target_formula, cn, geom, donors,
                    lfse_kj, pref_kj, match, charge, specials, rationale,
                    donor_strategy=strategy_name))

    # Sort by match + LFSE + preference
    candidates.sort(
        key=lambda c: c.hsab_match_score * 50 + c.lfse_stabilization_kj + c.geometry_preference_kj,
        reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return candidates[:max_candidates]

