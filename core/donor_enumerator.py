"""
core/donor_enumerator.py - Generative Coordination Engine, Layer 2.
From CoordinationEnvironment -> DonorArrangement candidates.
"""
from dataclasses import dataclass, field
from typing import Optional
import itertools
from knowledge.donor_chemistry import LIGAND_TEMPLATES, get_ligands_matching

@dataclass
class PositionedDonor:
    ligand_name: str
    donor_atom: str
    donor_count: int
    bond_length_A: float
    hsab_softness: float
    pka_coordinating: float
    charge_when_bound: int
    field_strength_dq: float
    steric_bulk: str
    ph_stable_range: tuple
    notes: str = ""

@dataclass
class DonorArrangement:
    target_formula: str
    coordination_environment: str
    geometry: str
    positioned_donors: list
    total_donor_count: int
    total_charge: int
    effective_denticity: int
    chelate_rings: int
    required_spacing_nm: float
    spacing_tolerance_nm: float
    ph_working_range: tuple
    hsab_match_score: float
    diversity_score: float
    synthetic_feasibility: float
    rationale: str
    rank: int = 0

def enumerate_donor_arrangements(coord_env, working_ph=7.0, max_arrangements=8,
                                  allow_mixed_donors=True, min_synthetic_feasibility=0.3):
    donor_groups = {}
    for d in coord_env.donors:
        donor_groups.setdefault(d.donor_atom, []).append(d)

    ligand_options = {}
    for da, reqs in donor_groups.items():
        target_soft = reqs[0].hsab_softness
        compat = get_ligands_matching(da, working_ph, target_soft, tolerance=0.35)
        if not compat:
            compat = [t for t in LIGAND_TEMPLATES
                      if t.donor_atom == da
                      and t.ph_stable_range[0] <= working_ph <= t.ph_stable_range[1]]
        ligand_options[da] = compat

    arrangements = []

    # Homogeneous
    for da, ligands in ligand_options.items():
        count = len(donor_groups[da])
        for lig in ligands:
            arr = _build_arrangement(coord_env, {da: [(lig, count)]}, working_ph, "homogeneous")
            if arr and arr.synthetic_feasibility >= min_synthetic_feasibility:
                arrangements.append(arr)

    # Mixed
    if allow_mixed_donors:
        for da, ligands in ligand_options.items():
            count = len(donor_groups[da])
            if count >= 2 and len(ligands) >= 2:
                for la, lb in itertools.combinations(ligands[:4], 2):
                    split = count // 2
                    arr = _build_arrangement(coord_env,
                        {da: [(la, split), (lb, count - split)]}, working_ph, "mixed")
                    if arr and arr.synthetic_feasibility >= min_synthetic_feasibility:
                        arrangements.append(arr)

    # Multidentate
    for lig in [t for t in LIGAND_TEMPLATES if t.donor_count >= 2]:
        if lig.donor_atom in donor_groups:
            if lig.ph_stable_range[0] <= working_ph <= lig.ph_stable_range[1]:
                count = len(donor_groups[lig.donor_atom])
                n_lig = max(1, count // lig.donor_count)
                arr = _build_arrangement(coord_env,
                    {lig.donor_atom: [(lig, n_lig)]}, working_ph, "multidentate")
                if arr and arr.synthetic_feasibility >= min_synthetic_feasibility:
                    arrangements.append(arr)

    for arr in arrangements:
        arr._score = (arr.hsab_match_score * 40 + arr.synthetic_feasibility * 30
                      + arr.chelate_rings * 5 + arr.diversity_score * 10)
    arrangements.sort(key=lambda a: a._score, reverse=True)
    for i, arr in enumerate(arrangements):
        arr.rank = i + 1
        if hasattr(arr, "_score"): delattr(arr, "_score")
    return arrangements[:max_arrangements]

def _build_arrangement(coord_env, ligand_assignments, working_ph, strategy):
    positioned = []
    ph_ranges = []
    for da, assignments in ligand_assignments.items():
        for lig, count in assignments:
            matching_reqs = [d for d in coord_env.donors if d.donor_atom == da]
            bl = matching_reqs[0].bond_length_A if matching_reqs else 2.0
            for _ in range(count):
                positioned.append(PositionedDonor(
                    lig.name, lig.donor_atom, lig.donor_count, bl,
                    lig.hsab_softness, lig.pka_coordinating, lig.charge_when_bound,
                    lig.field_strength_dq, lig.steric_bulk, lig.ph_stable_range, lig.notes))
                ph_ranges.append(lig.ph_stable_range)
    if not positioned: return None
    total_charge = sum(p.charge_when_bound for p in positioned)
    total_donors = sum(p.donor_count for p in positioned)
    chelate_rings = sum(1 for p in positioned if p.donor_count >= 2)
    ph_min = max(r[0] for r in ph_ranges)
    ph_max = min(r[1] for r in ph_ranges)
    if ph_min > ph_max: return None
    metal_softness = sum(d.hsab_softness for d in coord_env.donors) / len(coord_env.donors)
    avg_softness = sum(p.hsab_softness for p in positioned) / len(positioned)
    hsab_match = 1.0 - abs(metal_softness - avg_softness)
    unique_ligs = len(set(p.ligand_name for p in positioned))
    diversity = min(1.0, (unique_ligs - 1) * 0.3) if unique_ligs > 1 else 0.0
    avg_access = sum(next((t.synthetic_accessibility for t in LIGAND_TEMPLATES
                           if t.name == p.ligand_name), 0.5) for p in positioned) / len(positioned)
    feasibility = max(0.0, min(1.0, avg_access - max(0, (unique_ligs - 2) * 0.1)))
    gf_map = {"octahedral": 1.41, "tetrahedral": 1.63, "square_planar": 1.41,
              "linear": 2.0, "trigonal_bipyramidal": 1.41, "tetragonal_elongated": 1.41,
              "hemidirected_seesaw": 1.41, "hemidirected_octahedral": 1.41}
    avg_bond = sum(p.bond_length_A for p in positioned) / len(positioned)
    spacing_nm = avg_bond * gf_map.get(coord_env.geometry, 1.41) / 10.0
    lig_desc = ", ".join(f"{c}x{n}" for n, c in sorted(
        {p.ligand_name: sum(1 for q in positioned if q.ligand_name == p.ligand_name)
         for p in positioned}.items()))
    return DonorArrangement(
        coord_env.target_formula, f"{coord_env.geometry}_CN{coord_env.coordination_number}",
        coord_env.geometry, positioned, total_donors, total_charge, total_donors,
        chelate_rings, round(spacing_nm, 3), 0.05, (ph_min, ph_max),
        round(hsab_match, 3), round(diversity, 3), round(feasibility, 3),
        f"{strategy.capitalize()}: {lig_desc} in {coord_env.geometry}. "
        f"Denticity: {total_donors}. Chelate rings: {chelate_rings}. pH: {ph_min:.1f}-{ph_max:.1f}.")


