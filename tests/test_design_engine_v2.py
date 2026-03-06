"""
tests/test_design_engine_v2.py — Phase 17: Generalized design engine tests.
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import pytest
pytest.importorskip("rdkit")

from core.design_engine_v2 import (
    score_one, rank_binders, selectivity_screen, rank_hosts,
)


def test_score_one_metal():
    """score_one returns valid prediction for metal chelator."""
    sc = score_one("OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
                   metal="Cu2+", name="EDTA")
    assert sc.log_Ka_pred > 10.0  # EDTA+Cu2+ is strong
    assert sc.dg_total_kj < 0     # favorable binding
    assert sc.name == "EDTA"


def test_score_one_hg():
    """score_one returns valid prediction for host-guest."""
    sc = score_one("C1C2CC3CC1CC(C2)C3", host="beta-CD", name="adamantane")
    assert 2.0 < sc.log_Ka_pred < 8.0  # literature ~4.3
    assert sc.prediction.dg_hydrophobic < 0  # hydrophobic favorable


def test_rank_binders():
    """rank_binders returns sorted results."""
    smiles = [
        "OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",  # EDTA
        "NCCN",                                       # en
        "CC(=O)NO",                                   # AcHA
    ]
    names = ["EDTA", "en", "AcHA"]
    result = rank_binders("Cu2+", smiles, names=names, pH=7.4)

    assert result.n_scored == 3
    assert result.n_failed == 0
    assert result.candidates[0].name == "EDTA"  # strongest
    # Descending order
    for i in range(len(result.candidates) - 1):
        assert result.candidates[i].log_Ka_pred >= result.candidates[i+1].log_Ka_pred


def test_selectivity_screen():
    """selectivity_screen ranks by min_gap."""
    smiles = [
        "OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",  # EDTA (nonselective)
        "c1ccc2c(c1)c1cccnc1nc2",                   # phen (borderline donor)
    ]
    names = ["EDTA", "phen"]
    result = selectivity_screen("Pb2+", ["Ca2+"], smiles, names=names, pH=5.0)

    assert result.n_scored == 2
    assert result.mode == "selectivity"
    # Both should have selectivity gaps calculated
    for c in result.candidates:
        assert "Ca2+" in c.interferent_scores
        assert "Ca2+" in c.selectivity_gaps
        assert c.grade in ("A", "B", "C", "D", "F")


def test_rank_hosts():
    """rank_hosts correctly orders beta-CD guests."""
    smiles = [
        "C1C2CC3CC1CC(C2)C3",   # adamantane (best beta-CD guest)
        "c1ccccc1",              # benzene (weak)
    ]
    names = ["adamantane", "benzene"]
    result = rank_hosts("beta-CD", smiles, names=names)

    assert result.n_scored == 2
    assert result.mode == "host_guest"
    assert result.candidates[0].name == "adamantane"


def test_invalid_smiles_handled():
    """Invalid SMILES should increment n_failed, not crash."""
    smiles = ["INVALID_SMILES", "NCCN"]
    result = rank_binders("Cu2+", smiles, pH=7.4)
    assert result.n_scored == 1
    assert result.n_failed == 1
    assert len(result.errors) == 1


def test_multi_ligand():
    """n_ligand_molecules works in ranking."""
    smiles = ["CC(=O)CC(C)=O"]  # acac
    r1 = rank_binders("Fe3+", smiles, n_ligand_molecules=1, pH=14)
    r3 = rank_binders("Fe3+", smiles, n_ligand_molecules=3, pH=14)
    # Fe(acac)3 should be much stronger than Fe(acac)
    assert r3.candidates[0].log_Ka_pred > r1.candidates[0].log_Ka_pred


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED DISCOVERY TESTS
# ═══════════════════════════════════════════════════════════════════════════

from core.design_engine_v2 import discover_binders, discover_guests


def test_discover_binders_metal():
    """discover_binders returns mixed library + generated candidates."""
    r = discover_binders("Cu2+", top_n=10,
                          max_generate=30, max_score_generated=5)
    assert r.n_scored > 0
    assert "discover" in r.mode
    sources = set(c.source for c in r.candidates)
    # Should have at least library hits
    assert "library" in sources


def test_discover_binders_has_both_sources():
    """With generation on, both library and generated should appear."""
    r = discover_binders("Cu2+", generate=True, top_n=50,
                          max_generate=50, max_score_generated=10)
    sources = set(c.source for c in r.candidates)
    assert "library" in sources
    assert "generated" in sources


def test_discover_binders_no_generate():
    """With generate=False, only library candidates returned."""
    r = discover_binders("Cu2+", generate=False, top_n=10)
    sources = set(c.source for c in r.candidates)
    assert sources == {"library"}


def test_discover_binders_selectivity():
    """discover_binders with interferents does selectivity screening."""
    r = discover_binders("Pb2+", interferents=["Ca2+"], pH=5.0,
                          top_n=10, max_generate=30, max_score_generated=5)
    assert "selectivity" in r.mode
    for c in r.candidates:
        assert c.grade in ("A", "B", "C", "D", "F")
        assert "Ca2+" in c.interferent_scores or c.source == "generated"


def test_discover_binders_ranked():
    """Candidates are ranked by position."""
    r = discover_binders("Cu2+", top_n=10,
                          max_generate=30, max_score_generated=5)
    ranks = [c.rank for c in r.candidates]
    assert ranks == list(range(1, len(ranks) + 1))


def test_discover_binders_no_duplicates():
    """No duplicate SMILES in output."""
    r = discover_binders("Cu2+", top_n=50,
                          max_generate=100, max_score_generated=20)
    smiles = [c.smiles for c in r.candidates]
    assert len(smiles) == len(set(smiles))


def test_discover_guests():
    """discover_guests returns mixed results for beta-CD."""
    r = discover_guests("beta-CD", top_n=10,
                         max_generate=30, max_score_generated=5)
    assert r.n_scored > 0
    assert "host_guest" in r.mode


def test_discover_guests_has_sources():
    """discover_guests tags candidates with source."""
    r = discover_guests("beta-CD", top_n=30,
                         max_generate=50, max_score_generated=10)
    sources = set(c.source for c in r.candidates)
    assert len(sources) >= 1  # at least library or generated
