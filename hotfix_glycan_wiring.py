"""
hotfix_glycan_wiring.py — Adds missing _compute_glycan_terms to unified_scorer_v2.py

Run: python hotfix_glycan_wiring.py
From: MABE repo root
"""
import os, sys

if os.path.exists("core/universal_schema.py"):
    ROOT = "."
elif os.path.exists("../core/universal_schema.py"):
    ROOT = ".."
else:
    print("ERROR: Run from MABE repo root"); sys.exit(1)

path = os.path.join(ROOT, "core/unified_scorer_v2.py")
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# ── Check if function already defined ──
if "def _compute_glycan_terms" in text:
    print("_compute_glycan_terms already defined. No changes needed.")
    sys.exit(0)

# ── Find insertion point: just before _compute_cm_terms ──
MARKER = "def _compute_cm_terms(uc, result):"
if MARKER not in text:
    print(f"ERROR: Could not find '{MARKER}' in unified_scorer_v2.py")
    sys.exit(1)

FUNCTION = '''
# ═══════════════════════════════════════════════════════════════════════════
# GLYCAN-LECTIN SCORING — delegates to glycan/scorer.py
# ═══════════════════════════════════════════════════════════════════════════

def _compute_glycan_terms(uc, result):
    """Score glycan-lectin binding from contact maps + physics parameters.

    Fires only for binding_mode == 'glycan_lectin'.
    Self-zeros if glycan scorer or contact map not available.

    Maps GlycanPrediction terms to PredictionResult fields:
        DG0       -> dg_shape (scaffold geometry baseline)
        dG_HB     -> dg_hbond (H-bonds at interface)
        dG_desolv -> dg_group_desolv (OH burial desolvation)
        dG_CHP    -> dg_pi (CH-pi contacts)
        dG_linker -> dg_hbond_coop (linker cooperativity)
    """
    if uc.binding_mode != "glycan_lectin":
        return

    scaffold = uc.host_name
    ligand = uc.guest_name

    if not scaffold or not ligand:
        return

    try:
        from glycan.scorer import GlycanScorer
        scorer = GlycanScorer()
        pred = scorer.score(scaffold, ligand)
    except (ImportError, ValueError, KeyError):
        return

    # Map glycan terms to PredictionResult fields
    result.dg_shape = pred.dG0
    result.dg_hbond = pred.dG_HB
    result.dg_group_desolv = pred.dG_desolv
    result.dg_pi = pred.dG_CHP
    result.dg_hbond_coop = pred.dG_linker


'''

text = text.replace(MARKER, FUNCTION + MARKER)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("DONE: Added _compute_glycan_terms() to unified_scorer_v2.py")
print(f"  Inserted before {MARKER}")