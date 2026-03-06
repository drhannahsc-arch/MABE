"""
tests/test_general_pl.py — Phase 18: General protein-ligand scorer tests.
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import pytest
pytest.importorskip("rdkit")

from core.auto_descriptor import from_protein_ligand
from core.unified_scorer_v2 import predict


def test_from_protein_ligand_basic():
    """from_protein_ligand creates valid UC with general PL mode."""
    uc = from_protein_ligand("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COX-2", name="Ibuprofen")
    assert uc.binding_mode == "protein_ligand_general"
    assert uc.host_name == "COX-2"
    assert uc.guest_logP > 0
    assert uc.guest_mw > 100
    assert uc.guest_tpsa > 0
    assert uc.guest_fsp3 > 0


def test_general_pl_predict():
    """General PL scorer produces non-zero prediction."""
    uc = from_protein_ligand("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COX-2")
    r = predict(uc)
    assert r.log_Ka_pred > 3.0  # ibuprofen is a real COX-2 inhibitor
    assert r.log_Ka_pred < 12.0
    assert r.dg_total_kj < 0  # favorable binding


def test_no_metal_term():
    """General PL mode should not fire metal coordination."""
    uc = from_protein_ligand("c1ccccc1", "COX-2")
    r = predict(uc)
    # dg_metal is repurposed for target offset, but no metal coordination
    assert uc.metal_formula == ""
    assert uc.donor_subtypes == []


def test_different_targets_different_predictions():
    """Same ligand, different targets → different predictions (offset differs)."""
    smi = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    r_cox = predict(from_protein_ligand(smi, "COX-2"))
    r_hiv = predict(from_protein_ligand(smi, "HIV-1 protease"))
    # Different target offsets should produce different predictions
    assert abs(r_cox.log_Ka_pred - r_hiv.log_Ka_pred) > 0.5


def test_unknown_target_fallback():
    """Unknown target uses default offset, doesn't crash."""
    uc = from_protein_ligand("c1ccccc1", "Novel_Target_123")
    r = predict(uc)
    assert r.log_Ka_pred > 0  # should produce some prediction


def test_no_regression_metal():
    """Metal scoring untouched by general PL addition."""
    from core.auto_descriptor import from_smiles
    uc = from_smiles("OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O", metal="Cu2+", pH=14)
    r = predict(uc)
    # EDTA+Cu2+ should still be ~24.9 (frozen reference)
    assert abs(r.log_Ka_pred - 24.89) < 0.1


def test_no_regression_hg():
    """Host-guest scoring untouched."""
    from core.auto_descriptor import from_smiles
    uc = from_smiles("C1C2CC3CC1CC(C2)C3", host="beta-CD")
    r = predict(uc)
    # Adamantane in beta-CD should be ~4.25
    assert abs(r.log_Ka_pred - 4.25) < 0.2
