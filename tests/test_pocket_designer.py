"""
tests/test_pocket_designer.py — Pocket design engine tests.
"""

import sys
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import pytest
import numpy as np

from core.pocket_designer import (
    design_pocket, CAGE_TOPOLOGIES, select_donors_for_target,
    score_cage_design, CageDesignSpec,
)


def test_topologies_valid():
    """All cage topologies have consistent geometry."""
    for name, topo in CAGE_TOPOLOGIES.items():
        assert topo.n_vertices == len(topo.vertices)
        assert topo.n_edges == len(topo.edges)
        assert topo.inscribed_radius > 0
        # Edges connect valid vertex indices
        for i, j in topo.edges:
            assert 0 <= i < topo.n_vertices
            assert 0 <= j < topo.n_vertices


def test_topology_scaling():
    """Cavity scales with edge length."""
    topo = CAGE_TOPOLOGIES["octahedron"]
    g1 = topo.at_edge_length(1.0)
    g2 = topo.at_edge_length(2.0)
    assert g2["cavity_radius_nm"] == pytest.approx(2 * g1["cavity_radius_nm"], rel=0.01)
    assert g2["cavity_volume_A3"] == pytest.approx(8 * g1["cavity_volume_A3"], rel=0.01)


def test_donor_selection_hsab():
    """HSAB donor selection returns soft donors for soft metals."""
    # Hg2+ is soft → should get S or P donors near top
    sets, ranked = select_donors_for_target("Hg2+", n_donors=4)
    soft_donors = {"S_thiolate", "S_thioether", "S_dithiocarbamate", "P_phosphine"}
    top_donor = ranked[0][0]
    assert top_donor in soft_donors or "S_" in top_donor or "P_" in top_donor

    # Ca2+ is hard → should get O donors near top
    sets_ca, ranked_ca = select_donors_for_target("Ca2+", n_donors=4)
    hard_donors = {"O_carboxylate", "O_phenolate", "O_hydroxamate", "O_phosphate"}
    top_ca = ranked_ca[0][0]
    assert "O_" in top_ca


def test_design_pocket_runs():
    """design_pocket returns valid CageDesignSpec list."""
    designs, elapsed = design_pocket("Cu2+", pH=7.0, max_results=5,
                                      edge_lengths=[1.0], donor_counts=[4])
    assert len(designs) > 0
    assert elapsed > 0
    d = designs[0]
    assert isinstance(d, CageDesignSpec)
    assert d.target_metal == "Cu2+"
    assert d.target_log_ka > 0
    assert d.cavity_volume_A3 > 0
    assert d.rank == 1


def test_selectivity_scoring():
    """Selectivity screen produces gaps."""
    designs, _ = design_pocket("Pb2+", interferents=["Ca2+"],
                                pH=5.0, max_results=5,
                                edge_lengths=[1.0], donor_counts=[4])
    assert len(designs) > 0
    d = designs[0]
    assert "Ca2+" in d.interferent_scores
    assert "Ca2+" in d.selectivity_gaps
    assert d.selectivity_grade in ("A", "B", "C", "D", "F")


def test_athena_export():
    """ATHENA export produces required fields."""
    designs, _ = design_pocket("Zn2+", max_results=1,
                                edge_lengths=[2.0], donor_counts=[4])
    assert len(designs) > 0
    athena = designs[0].to_athena_input()
    assert "topology" in athena
    assert "edge_length_nm" in athena
    assert "vertices" in athena
    assert "edges" in athena
    assert "donor_positions" in athena
    assert "donor_types" in athena
    assert "predicted_log_ka" in athena


def test_multiple_topologies():
    """Multiple cage topologies are explored."""
    designs, _ = design_pocket("Cu2+", pH=7.0, max_results=20,
                                edge_lengths=[1.0])
    topo_names = set(d.topology_name for d in designs)
    assert len(topo_names) >= 2  # at least 2 different topologies
