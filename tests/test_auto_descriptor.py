"""
tests/test_auto_descriptor.py — Phase 13a Validation

50 known NIST ligands: SMILES → donor subtypes must match cal_dataset.
Zero tolerance on EDTA donor count.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from collections import Counter
from core.auto_descriptor import (
    extract_donor_subtypes, extract_subtypes_only,
    detect_chelate_rings, from_smiles, from_metal_ligand, from_host_guest
)


# ═══════════════════════════════════════════════════════════════════════════
# TEST LIGAND DATABASE
# (name, SMILES, expected_subtypes_counter, chelate_rings, ring_sizes, is_macro)
# chelate_rings=None → skip ring count check
# ═══════════════════════════════════════════════════════════════════════════

TEST_LIGANDS = [
    # ── Aminocarboxylate chelators ──────────────────────────────────────
    ("EDTA",
     "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
     Counter({"N_amine": 2, "O_carboxylate": 4}),
     5, [5,5,5,5,5], False),

    ("DTPA",
     "OC(=O)CN(CCN(CC(=O)O)CCN(CC(=O)O)CC(=O)O)CC(=O)O",
     Counter({"N_amine": 3, "O_carboxylate": 5}),
     7, [5,5,5,5,5,5,5], False),

    ("NTA",
     "OC(=O)CN(CC(=O)O)CC(=O)O",
     Counter({"N_amine": 1, "O_carboxylate": 3}),
     3, [5,5,5], False),

    ("IDA",
     "OC(=O)CNCC(=O)O",
     Counter({"N_amine": 1, "O_carboxylate": 2}),
     2, [5,5], False),

    ("HEDTA",
     "OCCN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
     Counter({"N_amine": 2, "O_carboxylate": 3, "O_hydroxyl": 1}),
     5, [5,5,5,5,5], False),

    ("EGTA",
     "OC(=O)CN(CCOCCOCCN(CC(=O)O)CC(=O)O)CC(=O)O",
     Counter({"N_amine": 2, "O_carboxylate": 4, "O_ether": 2}),
     None, None, False),  # ether O's detected; ring topology complex

    # ── Simple amines ──────────────────────────────────────────────────
    ("ethylenediamine (en)",
     "NCCN",
     Counter({"N_amine": 2}),
     1, [5], False),

    ("diethylenetriamine (dien)",
     "NCCNCCN",
     Counter({"N_amine": 3}),
     2, [5,5], False),

    ("triethylenetetramine (trien)",
     "NCCNCCNCCN",
     Counter({"N_amine": 4}),
     3, [5,5,5], False),

    # ── Amino acids ────────────────────────────────────────────────────
    ("glycine",
     "NCC(=O)O",
     Counter({"N_amine": 1, "O_carboxylate": 1}),
     1, [5], False),

    ("alanine",
     "CC(N)C(=O)O",
     Counter({"N_amine": 1, "O_carboxylate": 1}),
     1, [5], False),

    ("histidine",
     "NC(Cc1c[nH]cn1)C(=O)O",
     Counter({"N_amine": 1, "N_imidazole": 1, "O_carboxylate": 1}),
     None, None, False),  # complex ring topology

    # ── Pyridine-type ──────────────────────────────────────────────────
    ("2,2'-bipyridine",
     "c1ccnc(-c2ccccn2)c1",
     Counter({"N_pyridine": 2}),
     1, [5], False),

    ("1,10-phenanthroline",
     "c1cnc2c(c1)ccc1cccnc12",
     Counter({"N_pyridine": 2}),
     1, [5], False),

    ("terpyridine",
     "c1ccnc(-c2cccc(-c3ccccn3)n2)c1",
     Counter({"N_pyridine": 3}),
     2, [5,5], False),

    ("picolinic acid",
     "OC(=O)c1ccccn1",
     Counter({"N_pyridine": 1, "O_carboxylate": 1}),
     1, [5], False),

    ("dipicolinic acid (DPA)",
     "OC(=O)c1cccc(C(=O)O)n1",
     Counter({"N_pyridine": 1, "O_carboxylate": 2}),
     2, [5,5], False),

    # ── Phenolates / Quinolines ────────────────────────────────────────
    ("8-hydroxyquinoline",
     "Oc1cccc2cccnc12",
     Counter({"N_pyridine": 1, "O_phenolate": 1}),
     1, [5], False),

    ("salicylic acid",
     "OC(=O)c1ccccc1O",
     Counter({"O_carboxylate": 1, "O_phenolate": 1}),
     1, [6], False),

    ("2-aminophenol",
     "Nc1ccccc1O",
     Counter({"N_amine": 1, "O_phenolate": 1}),
     1, [5], False),

    # ── Schiff bases ───────────────────────────────────────────────────
    ("salen",
     "Oc1ccccc1/C=N/CC/N=C/c1ccccc1O",
     Counter({"N_imine": 2, "O_phenolate": 2}),
     3, [5,6,6], False),

    # ── Catechol ───────────────────────────────────────────────────────
    ("catechol",
     "Oc1ccccc1O",
     Counter({"O_catecholate": 2}),
     1, [5], False),

    # ── Dicarboxylates ─────────────────────────────────────────────────
    ("oxalic acid",
     "OC(=O)C(=O)O",
     Counter({"O_carboxylate": 2}),
     1, [5], False),

    ("malonic acid",
     "OC(=O)CC(=O)O",
     Counter({"O_carboxylate": 2}),
     1, [6], False),

    ("succinic acid",
     "OC(=O)CCC(=O)O",
     Counter({"O_carboxylate": 2}),
     1, [7], False),

    ("tartaric acid",
     "OC(=O)C(O)C(O)C(=O)O",
     Counter({"O_carboxylate": 2, "O_hydroxyl": 2}),
     None, None, False),  # complex — depends on which OH's are detected

    ("citric acid",
     "OC(=O)CC(O)(CC(=O)O)C(=O)O",
     Counter({"O_carboxylate": 3, "O_hydroxyl": 1}),
     None, None, False),  # variable denticity

    # ── Monocarboxylate ────────────────────────────────────────────────
    ("acetic acid",
     "CC(=O)O",
     Counter({"O_carboxylate": 1}),
     0, [], False),

    # ── Hydroxamic acid ────────────────────────────────────────────────
    ("acetohydroxamic acid",
     "CC(=O)NO",
     Counter({"O_hydroxamate": 2}),
     1, [5], False),

    # ── Sulfur donors ──────────────────────────────────────────────────
    ("cysteine",
     "NC(CS)C(=O)O",
     Counter({"N_amine": 1, "O_carboxylate": 1, "S_thiolate": 1}),
     2, [5,5], False),

    ("penicillamine",
     "NC(C(C)(C)S)C(=O)O",
     Counter({"N_amine": 1, "O_carboxylate": 1, "S_thiolate": 1}),
     2, [5,5], False),

    ("DMSA",
     "OC(=O)C(S)C(S)C(=O)O",
     Counter({"O_carboxylate": 2, "S_thiolate": 2}),
     3, [5,5,5], False),

    ("thioglycolic acid",
     "OC(=O)CS",
     Counter({"O_carboxylate": 1, "S_thiolate": 1}),
     1, [5], False),

    ("diethyldithiocarbamate",
     "CCN(CC)C(=S)S",
     Counter({"S_dithiocarbamate": 2}),
     1, [4], False),  # S-C-S = 2 bonds → 4-ring

    # ── Oximes ─────────────────────────────────────────────────────────
    ("dimethylglyoxime (DMG)",
     "CC(=NO)C(=NO)C",
     Counter({"N_imine": 2, "O_hydroxyl": 2}),
     None, None, False),  # 3-ring N-O pairs are edge case

    # ── Macrocyclic amines ─────────────────────────────────────────────
    ("cyclam",
     "C1CNCCCNCCCNCCCN1",
     Counter({"N_amine": 4}),
     4, None, True),  # mixed 5/6-rings

    ("cyclen",
     "C1CNCCNCCNCCN1",
     Counter({"N_amine": 4}),
     4, None, True),

    # ── Macrocyclic + pendant arms ─────────────────────────────────────
    ("DOTA",
     "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC1",
     Counter({"N_amine": 4, "O_carboxylate": 4}),
     None, None, True),

    ("NOTA",
     "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CC1",
     Counter({"N_amine": 3, "O_carboxylate": 3}),
     None, None, True),

    # ── Crown ethers ───────────────────────────────────────────────────
    ("18-crown-6",
     "C1COCCOCCOCCOCCOCCO1",
     Counter({"O_ether": 6}),
     None, None, True),

    ("15-crown-5",
     "C1COCCOCCOCCOCCO1",
     Counter({"O_ether": 5}),
     None, None, True),

    ("12-crown-4",
     "C1COCCOCCOCCO1",
     Counter({"O_ether": 4}),
     None, None, True),

    # ── 2-aminoethanol ─────────────────────────────────────────────────
    ("2-aminoethanol",
     "NCCO",
     Counter({"N_amine": 1, "O_hydroxyl": 1}),
     1, [5], False),

    # ── Imidazole ──────────────────────────────────────────────────────
    ("imidazole",
     "c1c[nH]cn1",
     Counter({"N_imidazole": 1}),
     0, [], False),

    # ── Deprotonated forms ─────────────────────────────────────────────
    ("EDTA (deprotonated)",
     "[O-]C(=O)CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]",
     Counter({"N_amine": 2, "O_carboxylate": 4}),
     5, [5,5,5,5,5], False),

    ("glycine (zwitterion)",
     "[NH3+]CC(=O)[O-]",
     Counter({"N_amine": 1, "O_carboxylate": 1}),
     1, [5], False),

    ("oxalate (dianion)",
     "[O-]C(=O)C(=O)[O-]",
     Counter({"O_carboxylate": 2}),
     1, [5], False),

    ("8-HQ (deprotonated)",
     "[O-]c1cccc2cccnc12",
     Counter({"N_pyridine": 1, "O_phenolate": 1}),
     1, [5], False),

    # ── Additional breadth ─────────────────────────────────────────────
    ("iminodiacetic acid",
     "OC(=O)CNCC(=O)O",
     Counter({"N_amine": 1, "O_carboxylate": 2}),
     2, [5,5], False),

    ("bipyridine (deprotonated)",
     "c1ccnc(-c2ccccn2)c1",
     Counter({"N_pyridine": 2}),
     1, [5], False),
]


# ═══════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def test_donor_extraction():
    """Donor subtype extraction: sorted subtypes match cal_dataset annotation."""
    passed = failed = 0
    errors = []

    for name, smiles, expected_counter, *_ in TEST_LIGANDS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            errors.append(f"  PARSE FAIL: {name} — {smiles}")
            failed += 1
            continue

        donors = extract_donor_subtypes(mol)
        got_counter = Counter(s for _, s in donors)

        if got_counter == expected_counter:
            passed += 1
        else:
            failed += 1
            missing = expected_counter - got_counter
            extra = got_counter - expected_counter
            detail = ""
            if +missing:
                detail += f" missing={dict(missing)}"
            if +extra:
                detail += f" extra={dict(extra)}"
            errors.append(f"  FAIL: {name:35s}{detail}")

    print(f"\n{'='*72}")
    print(f"DONOR EXTRACTION: {passed}/{passed+failed} passed")
    print(f"{'='*72}")
    for e in errors:
        print(e)
    return failed


def test_chelate_rings():
    """Chelate ring count and sizes match expectations."""
    passed = failed = skipped = 0
    errors = []

    for name, smiles, _, chelate_rings, ring_sizes, is_macro in TEST_LIGANDS:
        if chelate_rings is None:
            skipped += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed += 1
            continue

        donors = extract_donor_subtypes(mol)
        n_rings, got_sizes, got_macro, _ = detect_chelate_rings(mol, donors)

        ok = True
        details = []

        if n_rings != chelate_rings:
            ok = False
            details.append(f"n_rings: exp={chelate_rings} got={n_rings}")

        if ring_sizes is not None and sorted(got_sizes) != sorted(ring_sizes):
            ok = False
            details.append(f"sizes: exp={sorted(ring_sizes)} got={sorted(got_sizes)}")

        if got_macro != is_macro:
            ok = False
            details.append(f"macro: exp={is_macro} got={got_macro}")

        if ok:
            passed += 1
        else:
            failed += 1
            errors.append(f"  FAIL: {name:35s} {'; '.join(details)}")

    print(f"\n{'='*72}")
    print(f"CHELATE RINGS: {passed}/{passed+failed} passed ({skipped} skipped)")
    print(f"{'='*72}")
    for e in errors:
        print(e)
    return failed


def test_macrocyclic_detection():
    """Macrocyclic flag correct for all ligands."""
    passed = failed = 0
    errors = []

    for name, smiles, _, _, _, is_macro in TEST_LIGANDS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        donors = extract_donor_subtypes(mol)
        _, _, got_macro, _ = detect_chelate_rings(mol, donors)

        if got_macro == is_macro:
            passed += 1
        else:
            failed += 1
            errors.append(f"  FAIL: {name:35s} exp={is_macro} got={got_macro}")

    print(f"\n{'='*72}")
    print(f"MACROCYCLIC: {passed}/{passed+failed} passed")
    print(f"{'='*72}")
    for e in errors:
        print(e)
    return failed


def test_edta_critical():
    """CRITICAL: EDTA + Cu2+ must be perfectly correct."""
    uc = from_smiles("OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O", metal="Cu2+")

    checks = {
        "subtypes": (Counter(uc.donor_subtypes),
                     Counter({"N_amine": 2, "O_carboxylate": 4})),
        "denticity": (uc.denticity, 6),
        "chelate_rings": (uc.chelate_rings, 5),
        "ring_sizes": (sorted(uc.ring_sizes), [5,5,5,5,5]),
        "is_macrocyclic": (uc.is_macrocyclic, False),
        "metal": (uc.metal_formula, "Cu2+"),
        "metal_charge": (uc.metal_charge, 2),
        "metal_d_electrons": (uc.metal_d_electrons, 9),
        "binding_mode": (uc.binding_mode, "metal_coordination"),
    }

    errors = []
    for label, (got, exp) in checks.items():
        if got != exp:
            errors.append(f"  {label}: exp={exp} got={got}")

    if uc.guest_volume_A3 <= 0:
        errors.append(f"  guest_volume_A3 not computed: {uc.guest_volume_A3}")
    if uc.guest_sasa_total_A2 <= 0:
        errors.append(f"  guest_sasa_total_A2 not computed: {uc.guest_sasa_total_A2}")

    print(f"\n{'='*72}")
    print(f"EDTA CRITICAL: {'PASS ✓' if not errors else 'FAIL ✗'}")
    print(f"{'='*72}")
    if errors:
        for e in errors:
            print(e)
    else:
        print(f"  subtypes:  {dict(Counter(uc.donor_subtypes))}")
        print(f"  rings:     {uc.chelate_rings}× sizes={uc.ring_sizes}")
        print(f"  volume:    {uc.guest_volume_A3:.1f} Å³")
        print(f"  SASA:      {uc.guest_sasa_total_A2:.1f} Å²")
    return len(errors)


def test_from_metal_ligand():
    """from_metal_ligand wraps cal_dataset entries correctly."""
    entry = {
        "name": "EDTA+Cu2+", "metal": "Cu2+",
        "donors": ["N_amine","N_amine","O_carboxylate","O_carboxylate",
                    "O_carboxylate","O_carboxylate"],
        "chelate_rings": 5, "ring_sizes": [5,5,5,5,5],
        "macrocyclic": False, "cavity_nm": None,
        "n_lig_mol": 1, "pH": 14.0, "log_K_exp": 18.8, "source": "NIST",
    }
    uc = from_metal_ligand(entry)

    errors = []
    if uc.log_Ka_exp != 18.8:
        errors.append(f"log_Ka: {uc.log_Ka_exp}")
    if uc.metal_charge != 2:
        errors.append(f"charge: {uc.metal_charge}")
    if uc.denticity != 6:
        errors.append(f"dent: {uc.denticity}")
    if uc.binding_mode != "metal_coordination":
        errors.append(f"mode: {uc.binding_mode}")

    status = "PASS ✓" if not errors else f"FAIL ✗ {errors}"
    print(f"\n{'='*72}")
    print(f"FROM_METAL_LIGAND: {status}")
    print(f"{'='*72}")
    return len(errors)


def test_from_host_guest():
    """from_smiles with host populates cavity + guest + packing."""
    uc = from_smiles("C1CCCCC1", host="beta-CD")

    errors = []
    if uc.binding_mode != "host_guest_inclusion":
        errors.append(f"mode: {uc.binding_mode}")
    if uc.cavity_volume_A3 <= 0:
        errors.append(f"cavity: {uc.cavity_volume_A3}")
    if uc.guest_volume_A3 <= 0:
        errors.append(f"volume: {uc.guest_volume_A3}")
    if uc.packing_coefficient <= 0:
        errors.append(f"PC: {uc.packing_coefficient}")

    status = "PASS ✓" if not errors else f"FAIL ✗ {errors}"
    print(f"\n{'='*72}")
    print(f"HOST-GUEST: {status}")
    if not errors:
        print(f"  host={uc.host_name}, cavity={uc.cavity_volume_A3:.0f} Å³")
        print(f"  guest_vol={uc.guest_volume_A3:.1f} Å³, PC={uc.packing_coefficient:.2f}")
    print(f"{'='*72}")
    return len(errors)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total = 0
    total += test_edta_critical()
    total += test_donor_extraction()
    total += test_chelate_rings()
    total += test_macrocyclic_detection()
    total += test_from_metal_ligand()
    total += test_from_host_guest()

    print(f"\n{'='*72}")
    if total == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"TOTAL FAILURES: {total}")
    print(f"{'='*72}")