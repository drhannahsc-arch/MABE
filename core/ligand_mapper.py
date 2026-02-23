"""
ligand_mapper.py — Map abstract donor sets to real purchasable ligands.

Takes design_engine.py output (ranked donor sets with scores) and maps each
to real molecules from ligand_library.py, ranked by structural match quality.

Usage:
    from ligand_mapper import map_design_result, map_donor_set

    # From design engine output:
    result = map_design_result(engine_result)

    # Direct query:
    matches = map_donor_set(
        ["S_thiolate","S_thiolate","O_carboxylate","O_carboxylate"],
        chelate_rings=2, macrocyclic=False
    )
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from ligand_library import (
    LIGAND_DB, Ligand, search_donors, get_by_exact_donors,
    get_by_category, validate_library,
)


@dataclass
class LigandMatch:
    """A real ligand matched to an abstract donor set."""
    ligand: Ligand
    score: float               # Overall match score (0-1)
    donor_match: float         # Donor set similarity (Jaccard)
    topology_match: float      # Chelation topology similarity
    notes: list[str]           # Match quality annotations


@dataclass
class MappedDesign:
    """A complete design with real ligand options."""
    donor_set: list[str]       # Original abstract donor set
    target_metal: str          # Target metal
    predicted_log_k: float     # Predicted stability
    selectivity_gap: float     # Min gap vs interferents
    matches: list[LigandMatch] # Ranked real ligand options
    multi_ligand: list[tuple[Ligand, Ligand]]  # 2-ligand combinations (if no single match)


def _topology_score(ligand: Ligand, target_rings: int, target_macro: bool) -> float:
    """Score how well ligand chelation topology matches the target."""
    score = 1.0

    # Macrocyclic match
    if target_macro and not ligand.macrocyclic:
        score *= 0.5  # Wanted macrocyclic, got open-chain
    elif not target_macro and ligand.macrocyclic:
        score *= 0.7  # Got macrocyclic bonus we didn't ask for (still useful)

    # Chelate ring count similarity
    if target_rings > 0 and ligand.chelate_rings > 0:
        ratio = min(ligand.chelate_rings, target_rings) / max(ligand.chelate_rings, target_rings)
        score *= (0.5 + 0.5 * ratio)
    elif target_rings > 0 and ligand.chelate_rings == 0:
        score *= 0.4  # Wanted chelates, got monodentate
    elif target_rings == 0 and ligand.chelate_rings > 0:
        score *= 0.8  # Got chelates we didn't need (usually fine)

    return score


def _match_annotations(ligand: Ligand, query_donors: list[str],
                        target_rings: int, target_macro: bool) -> list[str]:
    """Generate human-readable match quality notes."""
    notes = []
    q = Counter(query_donors)
    l = Counter(ligand.donors)

    # Missing donors
    missing = q - l
    if missing:
        for st, count in missing.items():
            notes.append(f"missing {count}× {st}")

    # Extra donors
    extra = l - q
    if extra:
        for st, count in extra.items():
            notes.append(f"extra {count}× {st}")

    # Topology
    if target_rings > ligand.chelate_rings:
        notes.append(f"fewer rings ({ligand.chelate_rings} vs {target_rings} wanted)")
    elif ligand.chelate_rings > target_rings:
        notes.append(f"more rings ({ligand.chelate_rings} vs {target_rings} wanted)")

    if target_macro and not ligand.macrocyclic:
        notes.append("not macrocyclic (wanted macrocyclic)")
    elif not target_macro and ligand.macrocyclic:
        notes.append("macrocyclic (preorganization bonus)")

    if not notes:
        notes.append("exact match")

    return notes


def map_donor_set(
    donors: list[str],
    chelate_rings: int = 0,
    macrocyclic: bool = False,
    top_n: int = 5,
    min_score: float = 0.15,
) -> list[LigandMatch]:
    """Map an abstract donor set to ranked real ligands.

    Args:
        donors: List of donor subtypes from design engine
        chelate_rings: Expected number of chelate rings
        macrocyclic: Whether macrocyclic is desired
        top_n: Maximum results to return
        min_score: Minimum combined score threshold

    Returns:
        Ranked list of LigandMatch objects
    """
    # Get donor similarity scores
    donor_results = search_donors(donors)

    matches = []
    for lig, donor_score in donor_results:
        topo_score = _topology_score(lig, chelate_rings, macrocyclic)
        combined = donor_score * 0.7 + topo_score * 0.3
        if combined >= min_score:
            notes = _match_annotations(lig, donors, chelate_rings, macrocyclic)
            matches.append(LigandMatch(
                ligand=lig,
                score=round(combined, 3),
                donor_match=round(donor_score, 3),
                topology_match=round(topo_score, 3),
                notes=notes,
            ))

    matches.sort(key=lambda m: -m.score)
    return matches[:top_n]


def _find_combinations(donors: list[str], top_n: int = 3) -> list[tuple[Ligand, Ligand]]:
    """Find 2-ligand combinations that together cover the target donor set.

    For cases where no single ligand matches well (e.g., mixed S+N+O sets).
    """
    target = Counter(donors)
    target_total = sum(target.values())
    combos = []

    # Try all pairs
    for i, lig_a in enumerate(LIGAND_DB):
        count_a = Counter(lig_a.donors)
        # What's left after lig_a?
        remaining = target - count_a
        remaining_positive = +remaining  # only positive counts
        if sum(remaining_positive.values()) == 0:
            continue  # lig_a alone covers it (handled by single match)
        if sum(remaining_positive.values()) >= target_total:
            continue  # lig_a covers nothing

        for lig_b in LIGAND_DB[i:]:
            count_b = Counter(lig_b.donors)
            # How much does a + b cover?
            covered = count_a + count_b
            still_missing = target - covered
            coverage = 1.0 - sum((+still_missing).values()) / target_total
            if coverage >= 0.8:
                # Penalty for excess
                excess = covered - target
                excess_count = sum((+excess).values())
                penalty = 0.1 * excess_count
                combo_score = coverage - penalty
                if combo_score > 0.5:
                    combos.append((lig_a, lig_b, combo_score))

    combos.sort(key=lambda x: -x[2])
    return [(a, b) for a, b, s in combos[:top_n]]


def map_design_result(engine_result: dict, top_n: int = 5) -> list[MappedDesign]:
    """Map a complete design engine result to real ligands.

    Args:
        engine_result: Output from design_engine.py (dict with 'ranked' key)
        top_n: Max ligand matches per donor set

    Returns:
        List of MappedDesign objects, one per ranked donor set
    """
    mapped = []

    for entry in engine_result.get("ranked", []):
        donors = entry.get("donors", [])
        metal = entry.get("target", engine_result.get("target", ""))
        log_k = entry.get("log_k", 0.0)
        gap = entry.get("min_gap", 0.0)
        rings = entry.get("chelate_rings", 0)
        macro = entry.get("macrocyclic", False)

        # Find single-ligand matches
        matches = map_donor_set(donors, rings, macro, top_n)

        # If no good single match, try combinations
        combos = []
        if not matches or matches[0].score < 0.6:
            combos = _find_combinations(donors)

        mapped.append(MappedDesign(
            donor_set=donors,
            target_metal=metal,
            predicted_log_k=log_k,
            selectivity_gap=gap,
            matches=matches,
            multi_ligand=combos,
        ))

    return mapped


def print_mapped(mapped: list[MappedDesign]):
    """Pretty-print mapped design results."""
    for i, md in enumerate(mapped):
        print(f"\n{'═' * 70}")
        print(f"Design #{i+1}: {md.target_metal}")
        print(f"  Donors: {md.donor_set}")
        print(f"  Pred log K = {md.predicted_log_k:.1f}, "
              f"selectivity gap = {md.selectivity_gap:+.1f}")
        print(f"  {'─' * 60}")

        if md.matches:
            print(f"  Single-ligand matches:")
            for j, m in enumerate(md.matches):
                lig = m.ligand
                flag = "★" if m.score >= 0.8 else "◆" if m.score >= 0.5 else "○"
                print(f"    {flag} {lig.name:30s} score={m.score:.2f} "
                      f"(donor={m.donor_match:.2f} topo={m.topology_match:.2f})")
                print(f"      SMILES: {lig.smiles}")
                print(f"      MW={lig.mw or '?'}, CAS={lig.cas or '?'}, "
                      f"{'Commercial' if lig.commercial else 'Research'}")
                if m.notes != ["exact match"]:
                    print(f"      Notes: {'; '.join(m.notes)}")
        else:
            print(f"  No single-ligand matches")

        if md.multi_ligand:
            print(f"  2-ligand combinations:")
            for a, b in md.multi_ligand:
                print(f"    ◇ {a.name} + {b.name}")
                print(f"      Donors: {a.donors} + {b.donors}")


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Validate library first
    print("SMILES validation:")
    validate_library()

    # Test 1: EDTA-like donor set
    print("\n" + "=" * 70)
    print("Test 1: EDTA-type donor set (N₂O₄)")
    matches = map_donor_set(
        ["N_amine","N_amine","O_carboxylate","O_carboxylate",
         "O_carboxylate","O_carboxylate"],
        chelate_rings=5, macrocyclic=False,
    )
    for m in matches:
        print(f"  {m.ligand.name:25s} score={m.score:.3f}  "
              f"{'; '.join(m.notes)}")

    # Test 2: Thiolate-heavy set (Hg/Pb chelation)
    print("\n" + "=" * 70)
    print("Test 2: Thiolate set (S₂O₂, Pb chelation)")
    matches = map_donor_set(
        ["S_thiolate","S_thiolate","O_carboxylate","O_carboxylate"],
        chelate_rings=2, macrocyclic=False,
    )
    for m in matches:
        print(f"  {m.ligand.name:25s} score={m.score:.3f}  "
              f"{'; '.join(m.notes)}")

    # Test 3: Crown ether set (macrocyclic O₆)
    print("\n" + "=" * 70)
    print("Test 3: Crown ether set (macrocyclic O₆)")
    matches = map_donor_set(
        ["O_ether"]*6,
        chelate_rings=6, macrocyclic=True,
    )
    for m in matches:
        print(f"  {m.ligand.name:25s} score={m.score:.3f}  "
              f"{'; '.join(m.notes)}")

    # Test 4: Siderophore-like (catecholate O₆)
    print("\n" + "=" * 70)
    print("Test 4: Siderophore set (catecholate×6)")
    matches = map_donor_set(
        ["O_catecholate"]*6,
        chelate_rings=3, macrocyclic=False,
    )
    for m in matches:
        print(f"  {m.ligand.name:25s} score={m.score:.3f}  "
              f"{'; '.join(m.notes)}")

    # Test 5: Mixed soft donors (S_thiolate + N_pyridine)
    print("\n" + "=" * 70)
    print("Test 5: Mixed soft donors (S₂N₂, borderline metals)")
    matches = map_donor_set(
        ["S_thiolate","S_thiolate","N_pyridine","N_pyridine"],
        chelate_rings=2, macrocyclic=False,
    )
    for m in matches:
        print(f"  {m.ligand.name:25s} score={m.score:.3f}  "
              f"{'; '.join(m.notes)}")

    if not matches or matches[0].score < 0.6:
        print("  Trying 2-ligand combinations:")
        combos = _find_combinations(
            ["S_thiolate","S_thiolate","N_pyridine","N_pyridine"])
        for a, b in combos:
            print(f"    ◇ {a.name} + {b.name}")

    # Test 6: Phosphine set (Pd catalysis)
    print("\n" + "=" * 70)
    print("Test 6: Diphosphine (P₂, Pd cross-coupling)")
    matches = map_donor_set(
        ["P_phosphine","P_phosphine"],
        chelate_rings=1, macrocyclic=False,
    )
    for m in matches:
        print(f"  {m.ligand.name:25s} score={m.score:.3f}  "
              f"{'; '.join(m.notes)}")