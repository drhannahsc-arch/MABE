"""
tests/test_design_api.py — Unified design API tests.
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import pytest
pytest.importorskip("rdkit")

from core.design_api import score, design, list_targets, _classify_target


def test_classify_metal():
    assert _classify_target("Cu2+") == "metal"
    assert _classify_target("Pb2+") == "metal"


def test_classify_host():
    assert _classify_target("beta-CD") == "host"
    assert _classify_target("CB7") == "host"


def test_classify_protein():
    assert _classify_target("COX-2") == "protein"
    assert _classify_target("EGFR") == "protein"


def test_score_metal():
    r = score("NCCN", "Cu2+")
    assert r["modality"] == "metal"
    assert r["log_Ka_pred"] > 0


def test_score_host():
    r = score("C1C2CC3CC1CC(C2)C3", "beta-CD")
    assert r["modality"] == "host_guest"
    assert 2.0 < r["log_Ka_pred"] < 8.0


def test_score_protein():
    r = score("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COX-2")
    assert r["modality"] == "protein"
    assert r["log_Ka_pred"] > 3.0


def test_design_rank_metal():
    r = design("Cu2+", smiles=["NCCN", "CC(=O)NO"], names=["en", "AcHA"])
    assert r["mode"] == "metal_ranking"
    assert r["result"].n_scored == 2


def test_design_rank_protein():
    r = design("EGFR", smiles=["c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"],
               names=["benzene", "ibuprofen"])
    assert r["mode"] == "protein_ranking"
    assert len(r["candidates"]) == 2


def test_design_cage():
    r = design("Hg2+", mode="cage", interferents=["Zn2+"], top_n=3)
    assert r["mode"] == "cage_design"
    assert r["n_designs"] > 0


def test_design_host_guest():
    r = design("beta-CD", smiles=["c1ccccc1", "CCCCCC"], names=["benzene", "hexane"])
    assert r["mode"] == "host_guest_ranking"
    assert r["result"].n_scored == 2


def test_unknown_target_fallback():
    r = score("c1ccccc1", "NovelTarget123")
    assert r["modality"] == "unknown_protein"
    assert "log_Ka_pred" in r
