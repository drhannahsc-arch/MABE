"""
hotfix_unified_wiring_tests.py — Fix test_unified_wiring.py to match current glycan implementation

The current glycan wiring maps glycan terms to EXISTING PredictionResult fields:
    DG0       → dg_shape
    dG_HB     → dg_hbond
    dG_desolv → dg_group_desolv
    dG_CHP    → dg_pi
    dG_linker → dg_hbond_coop

An earlier wiring attempt expected DEDICATED glycan fields (dg_glycan_total, etc.)
which were never added to PredictionResult. This hotfix updates the tests.

Run: python hotfix_unified_wiring_tests.py
From: MABE repo root
"""
import os, sys

if os.path.exists("core/universal_schema.py"):
    ROOT = "."
elif os.path.exists("../core/universal_schema.py"):
    ROOT = ".."
else:
    print("ERROR: Run from MABE repo root"); sys.exit(1)

path = os.path.join(ROOT, "tests/test_unified_wiring.py")
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# ── Fix 1: TestGlycanFieldsExist ──
# These tests check for dg_glycan_total and dg_glycan_polar_desolv
# which don't exist. Replace with checks for the actual fields used.

old_glycan_total = '''def test_result_has_glycan_total'''
old_glycan_fields = '''def test_result_has_all_glycan_fields'''

if "dg_glycan_total" in text:
    # Replace the entire TestGlycanFieldsExist class
    # Find and replace the failing test methods

    # Strategy: replace references to dg_glycan_total with dg_shape
    # and dg_glycan_polar_desolv with dg_group_desolv
    text = text.replace("dg_glycan_total", "dg_shape")
    text = text.replace("dg_glycan_polar_desolv", "dg_group_desolv")
    text = text.replace("dg_glycan_hbond", "dg_hbond")
    text = text.replace("dg_glycan_chpi", "dg_pi")
    text = text.replace("dg_glycan_desolv", "dg_group_desolv")
    text = text.replace("dg_glycan_linker", "dg_hbond_coop")
    text = text.replace("dg_glycan_conf", "dg_conf_entropy")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"DONE: Replaced dg_glycan_* references with actual field names in {path}")
    print("  dg_glycan_total → dg_shape")
    print("  dg_glycan_polar_desolv → dg_group_desolv")
    print("  dg_glycan_hbond → dg_hbond")
    print("  dg_glycan_chpi → dg_pi")
    print("  dg_glycan_linker → dg_hbond_coop")
else:
    print("No dg_glycan_* references found — already fixed or different test version.")