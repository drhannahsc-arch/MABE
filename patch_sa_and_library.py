"""
patch_sa_and_library.py — Fix SA differentiation + diversify enumeration + add soft-metal scaffolds

Patches core/de_novo_generator.py with:
1. Improved sa_score() that differentiates small linear molecules
2. Round-robin enumerate_molecules() that samples across all backbones
3. New soft-metal-oriented backbones and arms

Run from MABE root:
    python patch_sa_and_library.py
    python -m pytest tests/test_de_novo_generator.py -v
    python demo_pareto_pb_selectivity.py
"""
import os
import sys
import re

def main():
    dng = os.path.join("core", "de_novo_generator.py")
    if not os.path.exists(dng):
        print(f"ERROR: {dng} not found. Run from MABE root.")
        sys.exit(1)

    with open(dng, "r", encoding="utf-8") as f:
        content = f.read()

    n_patches = 0

    # ═══════════════════════════════════════════════════════════════════
    # PATCH 1: SA Score — add functional group complexity + MW baseline
    # ═══════════════════════════════════════════════════════════════════

    old_sa = '''    # Complexity score
    complexity = (
        1.0                                    # baseline
        + n_rings * 0.3                        # each ring adds difficulty
        + n_spiro * 1.0                        # spiro centers are hard
        + n_bridgehead * 1.0                   # bridgeheads are hard
        + n_stereo * 0.5                       # stereocenters need control
        + macro_penalty
        + size_penalty
        + max(0, n_rotatable - 10) * 0.1       # very flexible = harder to purify
    )'''

    new_sa = '''    # Heteroatom diversity: more types of heteroatoms = harder synthesis
    het_types = set()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ("C", "H"):
            het_types.add(sym)
    n_het_types = len(het_types)

    # Functional group complexity from SMILES patterns
    smiles_for_fg = Chem.MolToSmiles(mol)
    fg_penalty = 0.0
    if "P(=O)" in smiles_for_fg or "P=O" in smiles_for_fg:
        fg_penalty += 0.6         # phosphonate/phosphine oxide
    if "C(=S)" in smiles_for_fg or "C=S" in smiles_for_fg:
        fg_penalty += 0.4         # thioamide/dithiocarbamate
    if "S(=O)(=O)" in smiles_for_fg:
        fg_penalty += 0.5         # sulfonamide/sulfonate
    if "N=C" in smiles_for_fg or "C=N" in smiles_for_fg:
        fg_penalty += 0.2         # imine/oxime
    if "B(O)" in smiles_for_fg:
        fg_penalty += 0.5         # boronic acid
    if "N=N" in smiles_for_fg:
        fg_penalty += 0.3         # hydrazide/azo

    # MW-based baseline (heavier molecules generally harder to make)
    mw = Descriptors.MolWt(mol)
    mw_term = mw / 200.0  # ~1.0 for MW=200, ~1.5 for MW=300

    # Complexity score
    complexity = (
        mw_term                                # MW-scaled baseline (not flat 1.0)
        + n_rings * 0.3                        # each ring adds difficulty
        + n_spiro * 1.0                        # spiro centers are hard
        + n_bridgehead * 1.0                   # bridgeheads are hard
        + n_stereo * 0.5                       # stereocenters need control
        + macro_penalty
        + size_penalty
        + max(0, n_rotatable - 10) * 0.1       # very flexible = harder to purify
        + max(0, n_het_types - 2) * 0.3        # heteroatom diversity penalty
        + fg_penalty                           # functional group complexity
    )'''

    if old_sa in content:
        content = content.replace(old_sa, new_sa)
        n_patches += 1
        print("  [1/3] SA score differentiation: PATCHED")
    else:
        print("  [1/3] SA score: SKIPPED (already patched or not found)")

    # ═══════════════════════════════════════════════════════════════════
    # PATCH 2: Round-robin enumeration across backbones
    # ═══════════════════════════════════════════════════════════════════

    old_enum = '''    for bb in backbones:
        # Generate all arm combinations for this backbone
        arm_combos = cartesian_product(compatible_arms, repeat=bb.n_sites)

        for arm_tuple in arm_combos:
            if len(results) >= max_candidates:
                return results

            smiles, mol = assemble(bb, list(arm_tuple))
            if smiles is None:
                continue

            # Dedup by canonical SMILES
            if smiles in seen:
                continue
            seen.add(smiles)

            # Property filter
            if not passes_filter(smiles, mol, pfilter):
                continue

            sa = sa_score(mol)
            arm_names = [a.name for a in arm_tuple]
            results.append((smiles, bb.name, arm_names, sa))

    return results'''

    new_enum = '''    # Round-robin across backbones to ensure scaffold diversity.
    # Each backbone gets a budget, then we cycle until max_candidates.
    import random as _rng
    _rng.seed(42)

    # Build combo iterators per backbone
    bb_combos = {}
    for bb in backbones:
        combos = list(cartesian_product(compatible_arms, repeat=bb.n_sites))
        _rng.shuffle(combos)
        bb_combos[bb.name] = (bb, combos, 0)  # (backbone, shuffled combos, index)

    # Round-robin: take N from each backbone per round
    PER_ROUND = max(3, max_candidates // (len(backbones) * 5))
    active_bbs = list(bb_combos.keys())
    _rng.shuffle(active_bbs)

    while len(results) < max_candidates and active_bbs:
        next_active = []
        for bb_name in active_bbs:
            if len(results) >= max_candidates:
                break
            bb, combos, idx = bb_combos[bb_name]
            added = 0
            while idx < len(combos) and added < PER_ROUND:
                arm_tuple = combos[idx]
                idx += 1

                smiles, mol = assemble(bb, list(arm_tuple))
                if smiles is None:
                    continue

                if smiles in seen:
                    continue
                seen.add(smiles)

                if not passes_filter(smiles, mol, pfilter):
                    continue

                sa = sa_score(mol)
                arm_names = [a.name for a in arm_tuple]
                results.append((smiles, bb.name, arm_names, sa))
                added += 1

            bb_combos[bb_name] = (bb, combos, idx)
            if idx < len(combos):
                next_active.append(bb_name)
        active_bbs = next_active

    return results'''

    if old_enum in content:
        content = content.replace(old_enum, new_enum)
        n_patches += 1
        print("  [2/3] Round-robin enumeration: PATCHED")
    else:
        print("  [2/3] Enumeration: SKIPPED (already patched or not found)")

    # ═══════════════════════════════════════════════════════════════════
    # PATCH 3: Add soft-metal backbones + arms
    # ═══════════════════════════════════════════════════════════════════

    # Find insertion point: just before the end of BACKBONE_LIBRARY
    # Look for the last Backbone entry followed by ]
    bb_insert_marker = '    Backbone("biphenyl-cleft"'
    arm_insert_marker = '    Arm("H-cap"'

    new_backbones = '''
    # ── SOFT-METAL SELECTIVE (added for Pb/Hg/Cd selectivity) ────────────
    Backbone("bis-thioether-en", "[1*]SCCSC[2*]", 2, "linear",
             "S2N0 podand, soft-metal selective via thioether donors"),
    Backbone("NS2-triamine", "[1*]NCCSCCS[2*]", 2, "linear",
             "NS2 mixed donor, borderline-soft selectivity"),
    Backbone("dithiol-propyl", "[1*]SCCCS[2*]", 2, "linear",
             "S2 dithiol, highly soft-selective"),
    Backbone("pyridine-2-thiol-6-subst", "[1*]c1cccc(S)n1", 1, "aromatic",
             "Pyridine-thiol NS donor, borderline-soft"),
    Backbone("thiophene-2,5-disubst", "[1*]c1ccc([2*])s1", 2, "aromatic",
             "Thiophene S-donor platform, soft metal selective"),
    Backbone("thioether-crown-S3", "[1*]SCCSCCSC[2*]", 2, "macrocyclic",
             "Trithia macrocyclic motif, high Pb/Hg selectivity"),
    Backbone("NS2-macrocycle", "[1*]N1CCSCCSCC1[2*]", 2, "macrocyclic",
             "NS2 macrocycle, Pb²⁺ / Cd²⁺ selective"),'''

    new_arms = '''
    # ── SOFT DONOR ARMS (added for Pb/Hg/Cd selectivity) ────────────────
    Arm("thioether-propyl", "[*]CCCS", ["S_thioether"], "S", "soft",
        "thioether, propyl spacer"),
    Arm("thioacetate", "[*]CC(=S)O", ["S_thioamide", "O_carboxylate"], "S", "soft",
        "thioacetate, bidentate S/O"),
    Arm("mercaptoacetate", "[*]SCC(=O)O", ["S_thiol", "O_carboxylate"], "S", "soft",
        "mercaptoacetic acid arm, S+O bidentate"),
    Arm("pyridine-2-thiol", "[*]c1ccccn1S", ["N_pyridine", "S_thiol"], "S", "borderline",
        "NS bidentate, borderline-soft"),
    Arm("thiadiazole-thiol", "[*]c1nnc(S)s1", ["S_thiol", "N_imine"], "S", "soft",
        "1,3,4-thiadiazole-2-thiol, strong soft donor"),'''

    if bb_insert_marker in content and "bis-thioether-en" not in content:
        # Find the line with biphenyl-cleft and insert after the full entry
        idx = content.index(bb_insert_marker)
        # Find the end of that Backbone(...) entry
        close_idx = content.index("),", idx) + 2
        content = content[:close_idx] + new_backbones + content[close_idx:]
        n_patches += 1
        print("  [3a/3] Soft-metal backbones: ADDED (7 new)")
    else:
        if "bis-thioether-en" in content:
            print("  [3a/3] Soft-metal backbones: already present")
        else:
            print("  [3a/3] Backbone insert point not found: SKIPPED")

    if arm_insert_marker in content and "thioether-propyl" not in content:
        idx = content.index(arm_insert_marker)
        content = content[:idx] + new_arms + "\n    " + content[idx:]
        n_patches += 1
        print("  [3b/3] Soft-metal arms: ADDED (5 new)")
    else:
        if "thioether-propyl" in content:
            print("  [3b/3] Soft-metal arms: already present")
        else:
            print("  [3b/3] Arm insert point not found: SKIPPED")

    # ── Write ─────────────────────────────────────────────────────────

    with open(dng, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nApplied {n_patches} patches to {dng}")
    print("Run: python -m pytest tests\\test_de_novo_generator.py -v")
    print("Then: python demo_pareto_pb_selectivity.py")


if __name__ == "__main__":
    main()