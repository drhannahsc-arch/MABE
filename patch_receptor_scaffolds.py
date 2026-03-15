"""
patch_receptor_scaffolds.py — Expand receptor/host scaffold library

Adds cage-like, deep-cavity, anion-binding, aromatic box, and capsule-forming
backbones to RECEPTOR_BACKBONE_LIBRARY, plus complementary receptor arms.

All scaffolds are published, synthetically precedented structures.

Run from MABE root:
    python patch_receptor_scaffolds.py
    python -m pytest tests/test_de_novo_generator.py -v
"""
import os
import sys


def main():
    dng = os.path.join("core", "de_novo_generator.py")
    if not os.path.exists(dng):
        print(f"ERROR: {dng} not found. Run from MABE root.")
        sys.exit(1)

    with open(dng, "r", encoding="utf-8") as f:
        content = f.read()

    n_patches = 0

    # ═══════════════════════════════════════════════════════════════════
    # PATCH 1: New receptor backbones
    # ═══════════════════════════════════════════════════════════════════

    # Insert before the closing ] of RECEPTOR_BACKBONE_LIBRARY
    # Use the last entry as anchor
    bb_anchor = 'Backbone("NS2-macrocycle"'

    new_receptor_bbs = '''
    # ── DEEP CLEFTS & AROMATIC BOXES ─────────────────────────────────────
    Backbone("diphenylmethane-cleft",
             "[1*]c1ccc(Cc2ccc([2*])cc2)cc1",
             2, "aromatic",
             "Diphenylmethane: hinged aromatic cleft, ~120 deg angle. "
             "Ref: Zimmerman HJ. Chem. Rev. 1997, 97, 1681"),
    Backbone("fluorene-platform",
             "[1*]c1ccc2c(c1)Cc1cc([2*])ccc1-2",
             2, "aromatic",
             "Fluorene: rigid 9H-fluorene platform, coplanar arms. "
             "Ref: Cram DJ. Science 1988, 240, 760"),
    Backbone("carbazole-cleft",
             "[1*]c1ccc2c(c1)[nH]c1cc([2*])ccc12",
             2, "aromatic",
             "Carbazole: NH donor at hinge + two aromatic walls. "
             "Ref: Etter MC. JACS 1990, 112, 8415"),
    Backbone("acridine-cleft",
             "[1*]c1ccc2cc3ccc([2*])cc3nc2c1",
             2, "aromatic",
             "Acridine: N-heterocycle hinge, fluorescent, intercalator geometry. "
             "Ref: Albert A. The Acridines, 2nd ed., Arnold 1966"),
    Backbone("terphenyl-spacer",
             "[1*]c1ccc(-c2ccc(-c3ccc([2*])cc3)cc2)cc1",
             2, "aromatic",
             "Terphenyl: extended linear spacer, ~15 A arm separation. "
             "Ref: Hamilton AD. Chem. Rev. 1997, 97, 1669"),
    Backbone("diphenylamine-cleft",
             "[1*]c1ccc(Nc2ccc([2*])cc2)cc1",
             2, "aromatic",
             "Diphenylamine: NH at hinge, electron-rich walls, ~120 deg. "
             "Ref: Anslyn EV. JACS 2005, 127, 15566"),
    Backbone("bis-naphthyl-methane",
             "[1*]c1ccc2ccccc2c1Cc1c([2*])ccc2ccccc12",
             2, "aromatic",
             "Bis(naphthyl)methane: large aromatic cleft for PAH guests. "
             "Ref: Klärner FG. Angew. Chem. Int. Ed. 2001, 40, 3635"),

    # ── ANION-BINDING SCAFFOLDS ──────────────────────────────────────────
    Backbone("thiourea-cleft",
             "[1*]NC(=S)NC(=S)N[2*]",
             2, "linear",
             "Bis-thiourea: 4 NH donors, strong anion/oxoanion binding. "
             "Ref: Gale PA. Chem. Commun. 2005, 3761"),
    Backbone("bis-amidopyridine",
             "[1*]NC(=O)c1cccc(C(=O)N[2*])n1",
             2, "aromatic",
             "2,6-bis(amido)pyridine: Hamilton-type receptor, 3 convergent H-bond "
             "donors for barbiturate/carboxylate guests. "
             "Ref: Hamilton AD. JACS 1988, 110, 1318"),
    Backbone("bis-sulfonamide-arene",
             "[1*]NS(=O)(=O)c1cccc(S(=O)(=O)N[2*])c1",
             2, "aromatic",
             "Bis-sulfonamide: acidic NH donors, anion cleft. "
             "Ref: Gale PA. Coord. Chem. Rev. 2003, 240, 191"),
    Backbone("bis-pyrrole-methane",
             "[1*]c1ccc([nH]1)Cc1ccc([2*])[nH]1",
             2, "aromatic",
             "Dipyrromethane: calix[4]pyrrole precursor, 2 NH donors converging. "
             "Ref: Sessler JL. Angew. Chem. Int. Ed. 1996, 35, 2380"),
    Backbone("guanidinium-cleft",
             "[1*]NC(=N)NC(=N)N[2*]",
             2, "linear",
             "Bis-guanidinium: cationic H-bond donor array, strong oxoanion binding. "
             "Ref: Schmidtchen FP. Chem. Rev. 1997, 97, 1609"),

    # ── CAPSULE-FORMING & 3D ENCAPSULATION ───────────────────────────────
    Backbone("tris-urea-tripod",
             "[1*]NC(=O)CCN(CCNC(=O)[2*])CCNC(=O)[3*]",
             3, "branched",
             "Tris(ureido)amine: 3 convergent urea NH pairs, cage-like H-bond "
             "capsule when dimerized. Ref: Rebek J. JACS 1996, 118, 2545"),
    Backbone("triamino-cyclohexane",
             "[1*]N[C@H]1C[C@@H](N[2*])C[C@H](N[3*])C1",
             3, "cyclic",
             "cis,cis-1,3,5-triaminocyclohexane: preorganized tripodal receptor, "
             "C3v symmetry. Ref: Steed JW. Supramol. Chem. 2000, 12, 129"),
    Backbone("benzene-tricarboxamide-extended",
             "[1*]CNC(=O)c1cc(C(=O)NC[2*])cc(C(=O)NC[3*])c1",
             3, "aromatic",
             "Extended trimesic triamide: methylene spacers give deeper cavity. "
             "Ref: Meijer EW. Chem. Rev. 2001, 101, 3893"),'''

    if bb_anchor in content and "diphenylmethane-cleft" not in content:
        idx = content.index(bb_anchor)
        close_idx = content.index("),", idx) + 2
        content = content[:close_idx] + new_receptor_bbs + content[close_idx:]
        n_patches += 1
        print("  [1/3] Receptor backbones: ADDED (15 new)")
    elif "diphenylmethane-cleft" in content:
        print("  [1/3] Receptor backbones: already present")
    else:
        print("  [1/3] Receptor backbone anchor not found: SKIPPED")

    # ═══════════════════════════════════════════════════════════════════
    # PATCH 2: New receptor-complementary arms
    # ═══════════════════════════════════════════════════════════════════

    arm_anchor = '    Arm("H-cap"'

    new_receptor_arms = '''    # ── RECEPTOR-SPECIFIC ARMS (aromatic panels, H-bond arrays) ────────────
    Arm("indolyl", "[*]c1c[nH]c2ccccc12",
        ["N_pyrrole"], "N", "borderline",
        "Indole: NH donor + large aromatic surface, tryptophan mimic"),
    Arm("pyrrole", "[*]c1ccc[nH]1",
        ["N_pyrrole"], "N", "borderline",
        "Pyrrole: NH donor for anion binding, calixpyrrole unit"),
    Arm("naphthyl", "[*]c1cccc2ccccc12",
        [], "C", "borderline",
        "Naphthyl: large aromatic panel for pi-stacking / hydrophobic wall"),
    Arm("anthracenyl", "[*]c1ccc2cc3ccccc3cc2c1",
        [], "C", "borderline",
        "Anthracenyl: large pi surface, fluorescent reporter + stacking"),
    Arm("urea-NH2", "[*]NC(=O)N",
        ["N_amine", "O_carbonyl"], "N", "hard",
        "Terminal urea: 2 NH donors + C=O acceptor, anion binding arm"),
    Arm("guanidinium", "[*]NC(=N)N",
        ["N_amine"], "N", "hard",
        "Guanidinium: cationic, charge-assisted H-bond donor for oxoanions"),
    Arm("sulfonamide-NH", "[*]NS(=O)(=O)C",
        ["N_amine"], "N", "borderline",
        "Sulfonamide: acidic NH donor (pKa ~10), anion recognition"),
    Arm("nitrophenyl", "[*]c1ccc([N+](=O)[O-])cc1",
        [], "C", "borderline",
        "p-Nitrophenyl: electron-poor aromatic, CT stacking with electron-rich guests"),
'''

    if arm_anchor in content and "indolyl" not in content:
        idx = content.index(arm_anchor)
        content = content[:idx] + new_receptor_arms + "    " + content[idx:]
        n_patches += 1
        print("  [2/3] Receptor arms: ADDED (8 new)")
    elif "indolyl" in content:
        print("  [2/3] Receptor arms: already present")
    else:
        print("  [2/3] Arm anchor not found: SKIPPED")

    # ═══════════════════════════════════════════════════════════════════
    # PATCH 3: Update _COMPLEMENTARY_ARM_MAP with new arms
    # ═══════════════════════════════════════════════════════════════════

    # Add new arms to the complementary map
    map_anchor = '    "hydrophobic": ['

    old_hydrophobic = '''    "hydrophobic": [
        # Arms with hydrophobic character
        ("thioether-methyl", 0.5),
        ("thiol-ethyl", 0.4),
        ("H-cap", 0.3),                # leave site unfunctionalized
    ],'''

    new_hydrophobic = '''    "hydrophobic": [
        # Arms with hydrophobic character
        ("thioether-methyl", 0.5),
        ("thiol-ethyl", 0.4),
        ("naphthyl", 0.85),            # large hydrophobic aromatic wall
        ("anthracenyl", 0.9),          # largest aromatic panel
        ("H-cap", 0.3),               # leave site unfunctionalized
    ],
    "anion": [
        # Arms that bind anions (carboxylate, phosphate, halide)
        ("pyrrole", 0.85),             # NH donor, calixpyrrole-like
        ("indolyl", 0.8),              # indole NH donor
        ("urea-NH2", 0.9),            # strong NH donor array
        ("guanidinium", 0.95),         # charge-assisted, best for oxoanions
        ("sulfonamide-NH", 0.75),      # acidic NH
        ("squaramide", 0.85),          # acidic NH + planar
        ("thiourea", 0.8),             # 2x NH donors
    ],
    "electron_poor_aromatic": [
        # Arms for charge-transfer stacking with electron-rich guests
        ("nitrophenyl", 0.9),          # strong electron-withdrawing
        ("sulfonamide-NH", 0.5),       # weakly electron-poor ring
        ("2-pyridyl", 0.6),            # moderately electron-poor
    ],
    "electron_rich_aromatic": [
        # Arms for CT stacking with electron-poor guests (quinones, NDI)
        ("catechol", 0.9),             # electron-rich, strong CT with quinones
        ("indolyl", 0.85),             # electron-rich heterocycle
        ("phenol", 0.7),               # moderately electron-rich
        ("naphthyl", 0.6),             # neutral aromatic
    ],'''

    if old_hydrophobic in content and '"anion"' not in content:
        content = content.replace(old_hydrophobic, new_hydrophobic)
        n_patches += 1
        print("  [3/3] Complementary arm map: EXPANDED (4 new feature types)")
    elif '"anion"' in content:
        print("  [3/3] Complementary arm map: already expanded")
    else:
        print("  [3/3] Complementary arm map anchor not found: SKIPPED")

    # ── Write ─────────────────────────────────────────────────────────

    with open(dng, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nApplied {n_patches} patches to {dng}")

    # Count totals
    import re
    bb_count = len(re.findall(r'Backbone\("', content))
    arm_count = len(re.findall(r'Arm\("', content))
    receptor_bb = content.count("Backbone(") - content.index("RECEPTOR_BACKBONE_LIBRARY")
    print(f"\nLibrary totals: {bb_count} backbones, {arm_count} arms")
    print("Run: python -m pytest tests\\test_de_novo_generator.py -v")


if __name__ == "__main__":
    main()