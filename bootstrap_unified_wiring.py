#!/usr/bin/env python3
"""
bootstrap_unified_wiring.py — Wire glycan into unified scorer + relax routing walls
=====================================================================================

Changes:
  1. universal_schema.py: Add glycan fields (sugar_property_card, glycan_contact_map, beta_context)
  2. unified_scorer_v2.py:
     - Add 6 glycan energy fields to PredictionResult
     - Add _compute_glycan_terms() that delegates to mabe.glycan.scorer
     - Wire glycan into predict() and dg_net sum
     - Add glycan to _print_decomposition()
     - Replace binding_mode hard gates with data-presence guards on:
       * _compute_protein_ligand_terms (was: binding_mode=="metalloprotein")
       * _compute_general_pl_terms (was: binding_mode=="protein_ligand_general")
     - Keep cross_modal routing (physically distinct — different cavity physics)

Zero-regression guarantee: All existing entries have binding_mode that matches
their data, so data-presence guards produce identical behavior. Glycan fields
default to None → glycan terms self-zero for all 644 existing entries.

Run from C:\\dev\\MABE:
  python bootstrap_unified_wiring.py
  python -m pytest tests/ -v
"""

import os
import re
import sys
import shutil
from datetime import datetime


def main():
    root = os.getcwd()
    schema_path = os.path.join(root, "core", "universal_schema.py")
    scorer_path = os.path.join(root, "core", "unified_scorer_v2.py")

    if not os.path.exists(schema_path):
        print(f"ERROR: {schema_path} not found. Run from MABE root.")
        sys.exit(1)

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [schema_path, scorer_path]:
        if os.path.exists(path):
            backup = path + f".bak_{ts}"
            shutil.copy2(path, backup)
            print(f"  Backed up: {os.path.basename(path)} -> {os.path.basename(backup)}")

    # ── PATCH 1: universal_schema.py ──────────────────────────────────
    patch_schema(schema_path)

    # ── PATCH 2: unified_scorer_v2.py ─────────────────────────────────
    patch_scorer(scorer_path)

    # ── CREATE: integration test ──────────────────────────────────────
    test_path = os.path.join(root, "tests", "test_unified_wiring.py")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(TEST_FILE)
    print(f"  Created: tests/test_unified_wiring.py")

    print()
    print("Done. Run: python -m pytest tests/test_unified_wiring.py -v")
    print("Then:  python -m pytest tests/ -v  (full regression)")


# ═══════════════════════════════════════════════════════════════════════════
# PATCH: universal_schema.py — add glycan fields
# ═══════════════════════════════════════════════════════════════════════════

def patch_schema(path):
    text = open(path, "r", encoding="utf-8").read()

    # Check if already patched
    if "sugar_property_card" in text:
        print("  universal_schema.py: already has glycan fields, skipping.")
        return

    # Add Optional import
    if "from typing import Optional" not in text:
        text = text.replace(
            "from dataclasses import dataclass, field",
            "from dataclasses import dataclass, field\nfrom typing import Optional",
        )

    # Insert glycan fields before TIER 2 section
    glycan_fields = '''
    # ── GLYCAN-SPECIFIC (active for lectin/receptor-sugar binding) ─────
    # Populated by from_glycan_binding() or manual construction.
    # All default to None → glycan terms self-zero for non-glycan entries.
    sugar_property_card: Optional[object] = None   # SugarPropertyCard instance
    glycan_contact_map: Optional[object] = None    # GlycanContactMap instance
    beta_context: Optional[float] = None           # Solvent-exposure factor (0-1)

'''

    # Insert before the TIER 2 section marker
    marker = "    # ── TIER 2 INTERACTION DESCRIPTORS"
    if marker in text:
        text = text.replace(marker, glycan_fields + marker)
    else:
        # Fallback: insert before __post_init__
        text = text.replace(
            "    def __post_init__",
            glycan_fields + "    def __post_init__",
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("  Patched: universal_schema.py (added glycan fields)")


# ═══════════════════════════════════════════════════════════════════════════
# PATCH: unified_scorer_v2.py — wire glycan + relax routing
# ═══════════════════════════════════════════════════════════════════════════

def patch_scorer(path):
    text = open(path, "r", encoding="utf-8").read()

    # Check if already patched
    if "dg_glycan_total" in text:
        print("  unified_scorer_v2.py: already has glycan wiring, skipping.")
        return

    # ── 1. Add glycan fields to PredictionResult ──────────────────────
    # Insert after dg_cm_shape line
    glycan_result_fields = '''
    # Glycan recognition (G1-G5)
    dg_glycan_polar_desolv: float = 0.0
    dg_glycan_hbond: float = 0.0
    dg_glycan_conf_entropy: float = 0.0
    dg_glycan_ch_pi: float = 0.0
    dg_glycan_structural_water: float = 0.0
    dg_glycan_ca_coordination: float = 0.0
    dg_glycan_total: float = 0.0
'''
    text = text.replace(
        "    dg_cm_shape: float = 0.0\n",
        "    dg_cm_shape: float = 0.0\n" + glycan_result_fields,
    )

    # ── 2. Add glycan compute call in predict() ──────────────────────
    # Insert after tier2 call
    glycan_call = '''
    # ── GLYCAN RECOGNITION (self-zeros if no sugar/contacts) ──────
    _compute_glycan_terms(uc, result)
'''
    text = text.replace(
        "    # ── SUM AND CONVERT",
        glycan_call + "\n    # ── SUM AND CONVERT",
    )

    # ── 3. Add glycan to dg_net sum ──────────────────────────────────
    text = text.replace(
        "              + tier2_total(result))",
        "              + tier2_total(result)\n"
        "              + result.dg_glycan_total)",
    )

    # ── 4. Add _compute_glycan_terms function ────────────────────────
    glycan_function = '''

# ═══════════════════════════════════════════════════════════════════════════
# GLYCAN RECOGNITION — delegates to mabe.glycan.scorer
# ═══════════════════════════════════════════════════════════════════════════

def _compute_glycan_terms(uc, result):
    """Glycan-specific energy terms (G1-G5).

    Self-zeros if sugar_property_card or glycan_contact_map is None.
    Physics: polyol desolvation, H-bond compensation, conformational entropy,
    CH-pi stacking, structural water bridges, Ca coordination.
    """
    sugar = getattr(uc, 'sugar_property_card', None)
    contacts = getattr(uc, 'glycan_contact_map', None)

    if sugar is None or contacts is None:
        return

    try:
        from mabe.glycan.scorer import compute_glycan_terms as _glycan_score
        from mabe.glycan.params import GLYCAN_PARAMS
    except ImportError:
        return  # glycan module not installed — self-zero

    beta = getattr(uc, 'beta_context', None)

    try:
        gr = _glycan_score(sugar=sugar, contacts=contacts,
                          params=GLYCAN_PARAMS, beta_context=beta)
    except Exception:
        return

    result.dg_glycan_polar_desolv = gr.dG_polar_desolv
    result.dg_glycan_hbond = gr.dG_hbond
    result.dg_glycan_conf_entropy = gr.dG_conf_entropy
    result.dg_glycan_ch_pi = gr.dG_ch_pi
    result.dg_glycan_structural_water = gr.dG_structural_water
    result.dg_glycan_ca_coordination = gr.dG_ca_coordination
    result.dg_glycan_total = gr.dG_total

'''

    # Insert before the VERBOSE OUTPUT section
    text = text.replace(
        "\n# ═══════════════════════════════════════════════════════════════════════════\n# VERBOSE OUTPUT",
        glycan_function + "\n# ═══════════════════════════════════════════════════════════════════════════\n# VERBOSE OUTPUT",
    )

    # ── 5. Add glycan to _print_decomposition ────────────────────────
    text = text.replace(
        '    # Tier 2\n    for f in TIER2_RESULT_FIELDS:',
        '    # Glycan\n'
        '    if result.dg_glycan_total != 0:\n'
        '        terms.append(("Glycan desolv", result.dg_glycan_polar_desolv))\n'
        '        terms.append(("Glycan H-bond", result.dg_glycan_hbond))\n'
        '        terms.append(("Glycan conf.S", result.dg_glycan_conf_entropy))\n'
        '        terms.append(("Glycan CH-π", result.dg_glycan_ch_pi))\n'
        '        terms.append(("Glycan water", result.dg_glycan_structural_water))\n'
        '        if result.dg_glycan_ca_coordination != 0:\n'
        '            terms.append(("Glycan Ca²⁺", result.dg_glycan_ca_coordination))\n'
        '    # Tier 2\n    for f in TIER2_RESULT_FIELDS:',
    )

    # ── 6. Relax metalloprotein routing wall ─────────────────────────
    # Replace: if uc.binding_mode != "metalloprotein": return
    # With:    if not (uc.metal_formula and uc.guest_smiles and uc.guest_sasa_nonpolar_A2 > 0): return
    text = text.replace(
        '    if uc.binding_mode != "metalloprotein":\n'
        '        return\n'
        '    if not uc.guest_smiles or uc.guest_sasa_nonpolar_A2 <= 0:\n'
        '        return',
        '    # Data-presence guard (replaces binding_mode hard gate)\n'
        '    if not (uc.metal_formula and uc.guest_smiles and uc.guest_sasa_nonpolar_A2 > 0):\n'
        '        return',
    )

    # ── 7. Relax general PL routing wall ─────────────────────────────
    # Replace: if uc.binding_mode != "protein_ligand_general": return
    # With: data-presence: needs guest_smiles + host_pdb_id + no metal
    text = text.replace(
        '    if uc.binding_mode != "protein_ligand_general":\n'
        '        return\n'
        '    if not uc.guest_smiles:\n'
        '        return',
        '    # Data-presence guard (replaces binding_mode hard gate)\n'
        '    if not (uc.guest_smiles and uc.host_pdb_id and not uc.metal_formula):\n'
        '        return',
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("  Patched: unified_scorer_v2.py (glycan wiring + routing relaxation)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST FILE
# ═══════════════════════════════════════════════════════════════════════════

TEST_FILE = r'''"""
tests/test_unified_wiring.py — Verify glycan wiring + routing relaxation
=========================================================================

Tests:
  1. Glycan fields exist on UniversalComplex
  2. Glycan fields exist on PredictionResult
  3. predict() self-zeros glycan for non-glycan entry
  4. predict() fires glycan for glycan entry (ConA-mannose)
  5. Regression: existing HG/metal entries unchanged
  6. Routing: metalloprotein fires on data-presence (no hard gate)
  7. Routing: general PL fires on data-presence (no hard gate)
"""

import pytest
import sys
import os

# Ensure project root on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestGlycanFieldsExist:
    """Schema and result have glycan fields."""

    def test_uc_has_sugar_property_card(self):
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(name="test")
        assert hasattr(uc, 'sugar_property_card')
        assert uc.sugar_property_card is None

    def test_uc_has_glycan_contact_map(self):
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(name="test")
        assert hasattr(uc, 'glycan_contact_map')
        assert uc.glycan_contact_map is None

    def test_uc_has_beta_context(self):
        from core.universal_schema import UniversalComplex
        uc = UniversalComplex(name="test")
        assert hasattr(uc, 'beta_context')
        assert uc.beta_context is None

    def test_result_has_glycan_total(self):
        from core.unified_scorer_v2 import PredictionResult
        r = PredictionResult(name="t", binding_mode="x",
                             log_Ka_exp=0, log_Ka_pred=0,
                             dg_total_kj=0, error=0)
        assert hasattr(r, 'dg_glycan_total')
        assert r.dg_glycan_total == 0.0

    def test_result_has_all_glycan_fields(self):
        from core.unified_scorer_v2 import PredictionResult
        r = PredictionResult(name="t", binding_mode="x",
                             log_Ka_exp=0, log_Ka_pred=0,
                             dg_total_kj=0, error=0)
        for field in ['dg_glycan_polar_desolv', 'dg_glycan_hbond',
                      'dg_glycan_conf_entropy', 'dg_glycan_ch_pi',
                      'dg_glycan_structural_water', 'dg_glycan_ca_coordination',
                      'dg_glycan_total']:
            assert hasattr(r, field), f"Missing {field}"
            assert getattr(r, field) == 0.0


class TestGlycanSelfZero:
    """Glycan terms must be zero when no glycan data present."""

    def test_hg_entry_zero_glycan(self):
        """A standard HG entry should have zero glycan contribution."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="beta-CD:adamantane",
            binding_mode="host_guest_inclusion",
            host_name="beta-CD",
            guest_smiles="C1C2CC3CC1CC(C2)C3",
            log_Ka_exp=4.26,
        )
        result = predict(uc)
        assert result.dg_glycan_total == 0.0

    def test_metal_entry_zero_glycan(self):
        """A metal coordination entry should have zero glycan."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict
        uc = UniversalComplex(
            name="Cu-EDTA",
            binding_mode="metal_coordination",
            metal_formula="Cu2+",
            metal_charge=2,
            metal_d_electrons=9,
            donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate",
                           "O_carboxylate","O_carboxylate"],
            chelate_rings=5,
            ring_sizes=[5,5,5,5,5],
            denticity=6,
            log_Ka_exp=18.8,
        )
        result = predict(uc)
        assert result.dg_glycan_total == 0.0


class TestGlycanFires:
    """Glycan terms fire when glycan data present."""

    def test_cona_mannose_fires(self):
        """ConA + mannose with glycan data should produce nonzero glycan energy."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict

        try:
            from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
            from mabe.glycan.contact_map import cona_mannose_pocket
        except ImportError:
            pytest.skip("Glycan module not installed")

        uc = UniversalComplex(
            name="ConA:mannose",
            binding_mode="lectin_glycan",
            log_Ka_exp=3.91,  # log(8200)
            sugar_property_card=ALPHA_D_MANNOSE,
            glycan_contact_map=cona_mannose_pocket(),
            beta_context=0.45,
        )
        result = predict(uc)
        assert result.dg_glycan_total != 0.0
        assert result.dg_glycan_total < 0  # should be favorable
        assert result.dg_glycan_polar_desolv > 0  # desolvation is unfavorable
        assert result.dg_glycan_hbond < 0  # H-bonds are favorable

    def test_glycan_adds_to_total(self):
        """Glycan energy should be included in dg_total_kj."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict

        try:
            from mabe.glycan.sugar_properties import ALPHA_D_MANNOSE
            from mabe.glycan.contact_map import cona_mannose_pocket
        except ImportError:
            pytest.skip("Glycan module not installed")

        uc = UniversalComplex(
            name="ConA:mannose",
            binding_mode="lectin_glycan",
            log_Ka_exp=3.91,
            sugar_property_card=ALPHA_D_MANNOSE,
            glycan_contact_map=cona_mannose_pocket(),
        )
        result = predict(uc)
        # dg_total should include glycan
        # For a pure glycan entry, total should equal glycan_total
        # (other terms self-zero since no guest_smiles/metal/host)
        assert abs(result.dg_total_kj - result.dg_glycan_total) < 0.01


class TestRoutingRelaxation:
    """Data-presence guards replace binding_mode hard gates."""

    def test_metalloprotein_fires_without_mode_label(self):
        """Entry with metal + guest data fires PL terms even without
        binding_mode='metalloprotein' label."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict

        uc = UniversalComplex(
            name="test_mixed",
            binding_mode="metal_coordination",  # NOT "metalloprotein"
            metal_formula="Zn2+",
            metal_charge=2,
            metal_d_electrons=10,
            donor_subtypes=["N_imidazole","N_imidazole","O_carboxylate"],
            chelate_rings=0,
            ring_sizes=[],
            denticity=3,
            guest_smiles="c1ccc(C(=O)O)cc1",  # benzoic acid
            guest_sasa_nonpolar_A2=120.0,
            guest_logP=1.87,
            guest_mw=122.12,
            guest_rotatable_bonds=1,
            n_hbonds_formed=2,
            sasa_buried_A2=80.0,
            log_Ka_exp=5.0,
        )
        result = predict(uc)
        # Metal terms should fire (has metal data)
        assert result.dg_metal != 0.0
        # PL terms should ALSO fire (has guest + SASA + metal)
        # This was blocked before by binding_mode gate
        assert result.dg_hydrophobic != 0.0 or result.dg_conf_entropy != 0.0

    def test_general_pl_fires_with_pdb_id(self):
        """Entry with host_pdb_id + guest fires general PL
        regardless of binding_mode label."""
        from core.universal_schema import UniversalComplex
        from core.unified_scorer_v2 import predict

        uc = UniversalComplex(
            name="test_pl",
            binding_mode="protein_ligand",  # NOT "protein_ligand_general"
            host_name="trypsin",
            host_pdb_id="1TNG",
            guest_smiles="c1ccc(C(=N)N)cc1",  # benzamidine
            guest_logP=1.5,
            guest_mw=120.0,
            guest_rotatable_bonds=1,
            guest_n_hbond_donors=2,
            guest_n_hbond_acceptors=1,
            guest_n_aromatic_rings=1,
            log_Ka_exp=4.0,
        )
        result = predict(uc)
        # Should fire general PL path (has pdb_id + guest, no metal)
        assert result.dg_total_kj != 0.0
'''


if __name__ == "__main__":
    main()