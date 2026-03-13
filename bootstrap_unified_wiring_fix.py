#!/usr/bin/env python3
"""
bootstrap_unified_wiring_fix.py -- Fix routing guards

Fixes 5 test failures from routing relaxation.

Logic: keep original binding_mode gates as primary trigger,
add data-presence as secondary ONLY for new/unhandled binding modes
(lectin_glycan, synthetic_receptor, unknown, etc.). Existing modes
route exactly as before = zero regression.

Run from C:\dev\MABE:
  python bootstrap_unified_wiring_fix.py
  python -m pytest tests/ -v
"""

import os
import sys


def main():
    root = os.getcwd()
    scorer_path = os.path.join(root, "core", "unified_scorer_v2.py")

    if not os.path.exists(scorer_path):
        print(f"ERROR: {scorer_path} not found. Run from MABE root.")
        sys.exit(1)

    text = open(scorer_path, "r", encoding="utf-8").read()
    changed = False

    # FIX 1: Metalloprotein PL terms
    old_metal = (
        '    # Data-presence guard (replaces binding_mode hard gate)\n'
        '    if not (uc.metal_formula and uc.guest_smiles and uc.guest_sasa_nonpolar_A2 > 0):\n'
        '        return'
    )
    new_metal = (
        '    # Fires for metalloprotein binding_mode OR data-presence on new modes\n'
        '    is_metalloprotein_mode = (uc.binding_mode == "metalloprotein")\n'
        '    _HANDLED_MODES = {"metal_coordination", "host_guest_inclusion",\n'
        '                      "cross_modal", "protein_ligand_general"}\n'
        '    has_metal_pl_data = (uc.binding_mode not in _HANDLED_MODES\n'
        '                         and uc.metal_formula and uc.guest_smiles\n'
        '                         and uc.guest_sasa_nonpolar_A2 > 0)\n'
        '    if not (is_metalloprotein_mode or has_metal_pl_data):\n'
        '        return'
    )
    if old_metal in text:
        text = text.replace(old_metal, new_metal)
        print("  Fixed: metalloprotein routing guard")
        changed = True

    # FIX 2: General PL terms
    old_pl = (
        '    # Data-presence guard (replaces binding_mode hard gate)\n'
        '    if not (uc.guest_smiles and uc.host_pdb_id and not uc.metal_formula):\n'
        '        return'
    )
    new_pl = (
        '    # Fires for protein_ligand_general binding_mode OR data-presence on new modes\n'
        '    is_general_pl_mode = (uc.binding_mode == "protein_ligand_general")\n'
        '    _HANDLED_MODES_PL = {"metal_coordination", "metalloprotein",\n'
        '                          "host_guest_inclusion", "cross_modal"}\n'
        '    has_general_pl_data = (uc.binding_mode not in _HANDLED_MODES_PL\n'
        '                           and uc.guest_smiles and uc.host_pdb_id\n'
        '                           and not uc.metal_formula)\n'
        '    if not (is_general_pl_mode or has_general_pl_data):\n'
        '        return'
    )
    if old_pl in text:
        text = text.replace(old_pl, new_pl)
        print("  Fixed: general PL routing guard")
        changed = True

    if changed:
        with open(scorer_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("\nRun: python -m pytest tests/ -v")
    else:
        print("  No matching patterns found. Already fixed or different code state.")


if __name__ == "__main__":
    main()