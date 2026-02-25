"""
metalloprotein_dataset.py — Phase 14a: Metalloprotein-ligand binding data.

300 curated entries across 6 Zn-dependent metalloenzymes from ChEMBL:
  1. Carbonic anhydrase II (CA-II) — His3-Zn-OH active site
  2. MMP-9  — His3-Zn-OH2 active site
  3. MMP-13 — His3-Zn-OH2 active site
  4. MMP-7  — His3-Zn-OH2 active site
  5. Thermolysin — His2Glu-Zn-OH2 active site
  6. ACE — His2Glu-Zn-OH2 active site

All entries have:
  - Ki (nM) converted to log Ka = -log10(Ki_M)
  - Canonical SMILES from ChEMBL
  - Metal-binding group (MBG) classified by SMARTS
  - MBG donor subtypes for metal coordination scoring

Sources:
  ChEMBL database (www.ebi.ac.uk/chembl), accessed 2026-02-24.
  Ki values with standard_relation='=' and standard_units='nM'.
  Deduplicated by canonical SMILES per target.
"""

import json
import os

# ═══════════════════════════════════════════════════════════════════════════
# TARGET REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

TARGET_REGISTRY = {
    "CA-II": {
        "chembl_id": "CHEMBL205",
        "metal": "Zn2+",
        "coord_motif": "His3-Zn-OH",
        "residues": ["His94", "His96", "His119"],
        "protein_donors": ["N_imidazole", "N_imidazole", "N_imidazole"],
        "description": "Carbonic anhydrase II — catalytic Zn bound by 3 His + water/hydroxide",
    },
    "MMP-9": {
        "chembl_id": "CHEMBL321",
        "metal": "Zn2+",
        "coord_motif": "His3-Zn-OH2",
        "residues": ["His226", "His230", "His236"],
        "protein_donors": ["N_imidazole", "N_imidazole", "N_imidazole"],
        "description": "Matrix metalloproteinase 9 — catalytic Zn bound by 3 His + water",
    },
    "MMP-13": {
        "chembl_id": "CHEMBL333",
        "metal": "Zn2+",
        "coord_motif": "His3-Zn-OH2",
        "residues": ["His222", "His226", "His232"],
        "protein_donors": ["N_imidazole", "N_imidazole", "N_imidazole"],
        "description": "Matrix metalloproteinase 13 — catalytic Zn bound by 3 His + water",
    },
    "MMP-7": {
        "chembl_id": "CHEMBL340",
        "metal": "Zn2+",
        "coord_motif": "His3-Zn-OH2",
        "residues": ["His218", "His222", "His228"],
        "protein_donors": ["N_imidazole", "N_imidazole", "N_imidazole"],
        "description": "Matrix metalloproteinase 7 — catalytic Zn bound by 3 His + water",
    },
    "Thermolysin": {
        "chembl_id": "CHEMBL1865",
        "metal": "Zn2+",
        "coord_motif": "His2Glu-Zn-OH2",
        "residues": ["His142", "His146", "Glu166"],
        "protein_donors": ["N_imidazole", "N_imidazole", "O_carboxylate"],
        "description": "Thermolysin — catalytic Zn bound by 2 His + Glu + water",
    },
    "ACE": {
        "chembl_id": "CHEMBL4801",
        "metal": "Zn2+",
        "coord_motif": "His2Glu-Zn-OH2",
        "residues": ["His383", "His387", "Glu411"],
        "protein_donors": ["N_imidazole", "N_imidazole", "O_carboxylate"],
        "description": "Angiotensin-converting enzyme — catalytic Zn bound by 2 His + Glu + water",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# METAL-BINDING GROUP (MBG) DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

MBG_DONOR_MAP = {
    "sulfonamide":  ["N_amide"],                          # deprotonated -SO2NH- → Zn-N
    "hydroxamate":  ["O_hydroxamate", "O_hydroxamate"],   # bidentate C(=O)N(OH) → Zn
    "carboxylate":  ["O_carboxylate"],                    # monodentate -COO⁻ → Zn
    "thiol":        ["S_thiolate"],                       # deprotonated -SH → Zn-S
    "phosphonate":  ["O_carboxylate", "O_carboxylate"],   # bidentate -PO3²⁻
    "phosphinate":  ["O_carboxylate"],                    # monodentate
    "boronate":     ["O_hydroxyl", "O_hydroxyl"],         # tetrahedral boronate
    "unknown":      [],                                    # unclassified MBG
}


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ENTRIES
# ═══════════════════════════════════════════════════════════════════════════

def _load_entries():
    """Load classified entries from JSON sidecar."""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "metalloprotein_entries.json")
    with open(json_path) as f:
        return json.load(f)


def _build_metalloprotein_entry(raw):
    """Convert raw ChEMBL entry to cal_dataset-compatible format."""
    target_info = TARGET_REGISTRY.get(raw["target"], {})
    mbg = raw.get("mbg", "unknown")
    mbg_donors = MBG_DONOR_MAP.get(mbg, [])
    protein_donors = target_info.get("protein_donors", [])

    return {
        "name": raw["name"],
        "target": raw["target"],
        "metal": raw["metal"],
        "smiles": raw["smiles"],
        "ki_nM": raw["ki_nM"],
        "log_Ka_exp": raw["log_Ka"],
        "mbg": mbg,
        "mbg_donors": mbg_donors,
        "protein_donors": protein_donors,
        "coord_motif": target_info.get("coord_motif", ""),
        "chembl_id": raw.get("chembl_id", ""),
        "source": "ChEMBL",
    }


# Lazy-loaded dataset
_METALLOPROTEIN_DATA = None

def get_metalloprotein_data():
    global _METALLOPROTEIN_DATA
    if _METALLOPROTEIN_DATA is None:
        raw = _load_entries()
        _METALLOPROTEIN_DATA = [_build_metalloprotein_entry(e) for e in raw]
    return _METALLOPROTEIN_DATA


# Convenience: direct import
try:
    METALLOPROTEIN_DATA = get_metalloprotein_data()
except FileNotFoundError:
    METALLOPROTEIN_DATA = []  # JSON not yet generated


if __name__ == "__main__":
    from collections import Counter
    data = get_metalloprotein_data()
    print(f"Phase 14a metalloprotein dataset")
    print(f"Total entries: {len(data)}")

    by_target = Counter(e["target"] for e in data)
    print("\nBy target:")
    for t, n in by_target.most_common():
        print(f"  {t:15s}: {n}")

    by_mbg = Counter(e["mbg"] for e in data)
    print("\nBy MBG:")
    for m, n in by_mbg.most_common():
        donors = MBG_DONOR_MAP.get(m, [])
        print(f"  {m:15s}: {n:3d}  donors={donors}")

    lks = [e["log_Ka_exp"] for e in data]
    print(f"\nlog Ka range: {min(lks):.1f} to {max(lks):.1f}")
    print(f"log Ka mean:  {sum(lks)/len(lks):.1f}")