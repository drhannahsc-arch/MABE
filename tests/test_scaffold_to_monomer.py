"""
tests/test_scaffold_to_monomer.py -- Tests for multisite backbone → MonomerSpec bridge.
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

pytestmark = pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")

from core.scaffold_to_monomer import (
    backbone_to_monomer, all_monomer_specs,
    _analyze_backbone_chemistry, _generate_faces,
)
from core.multisite_assembler import ALL_MULTISITE_BACKBONES, get_multisite_backbones
from core.assembly_engine import (
    FaceRole, InteractionMode, design_material, Topology,
)


# -----------------------------------------------------------------------
# Chemistry detection
# -----------------------------------------------------------------------

class TestChemistryDetection:

    def test_davis_detects_urea(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bis-anthracene-diurea"][0]
        chem = _analyze_backbone_chemistry(bb.smiles)
        assert chem.get("urea", 0) >= 2

    def test_davis_detects_aromatic(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bis-anthracene-diurea"][0]
        chem = _analyze_backbone_chemistry(bb.smiles)
        assert chem.get("aromatic_atoms", 0) >= 10

    def test_nta_detects_carboxylate(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "nitrilotriacetic"][0]
        chem = _analyze_backbone_chemistry(bb.smiles)
        assert chem.get("carboxylate", 0) >= 2

    def test_bipy_detects_pyridine(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bipy-disubst"][0]
        chem = _analyze_backbone_chemistry(bb.smiles)
        assert chem.get("pyridine_N", 0) >= 2

    def test_salen_detects_imine_and_phenol(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "salen"][0]
        chem = _analyze_backbone_chemistry(bb.smiles)
        assert chem.get("imine", 0) >= 1
        assert chem.get("phenol", 0) >= 1

    def test_crown_detects_ether(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "18-crown-6-disubst"][0]
        chem = _analyze_backbone_chemistry(bb.smiles)
        assert chem.get("crown_ether", 0) >= 3


# -----------------------------------------------------------------------
# MonomerSpec generation
# -----------------------------------------------------------------------

class TestMonomerGeneration:

    def test_all_backbones_produce_monomers(self):
        for bb in ALL_MULTISITE_BACKBONES:
            mono = backbone_to_monomer(bb)
            assert mono.name == bb.name
            assert len(mono.faces) > 0

    def test_valence_matches_sites(self):
        for bb in ALL_MULTISITE_BACKBONES:
            mono = backbone_to_monomer(bb)
            assert mono.valence == bb.n_sites

    def test_davis_has_capture_face(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bis-anthracene-diurea"][0]
        mono = backbone_to_monomer(bb)
        assert mono.n_capture_faces >= 1

    def test_davis_hbond_structural(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bis-anthracene-diurea"][0]
        mono = backbone_to_monomer(bb)
        structural = mono.structural_faces()
        modes = {f.interaction for f in structural}
        assert InteractionMode.HBOND_NETWORK in modes

    def test_cyclam_coordination(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "cyclam"][0]
        mono = backbone_to_monomer(bb)
        structural = mono.structural_faces()
        assert all(f.interaction == InteractionMode.COORDINATION for f in structural)

    def test_cyclam_symmetry(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "cyclam"][0]
        mono = backbone_to_monomer(bb)
        assert mono.symmetry in ("C4", "D4h")

    def test_tripodal_c3_symmetry(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "TREN"][0]
        mono = backbone_to_monomer(bb)
        assert mono.symmetry == "C3"

    def test_coordination_faces_target_metal(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bipy-disubst"][0]
        mono = backbone_to_monomer(bb)
        for f in mono.structural_faces():
            if f.interaction == InteractionMode.COORDINATION:
                assert f.complementary_to == "metal_node"


# -----------------------------------------------------------------------
# Topology predictions
# -----------------------------------------------------------------------

class TestTopologyPredictions:

    def test_cyclam_framework(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "cyclam"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.FRAMEWORK_3D

    def test_davis_chain(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bis-anthracene-diurea"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.CHAIN_1D

    def test_tren_honeycomb(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "TREN"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.HONEYCOMB_2D

    def test_salen_chain(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "salen"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.CHAIN_1D

    def test_crown_chain(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "18-crown-6-disubst"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.CHAIN_1D

    def test_edta_framework(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "EDTA-core"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.FRAMEWORK_3D

    def test_bipy_chain(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "bipy-disubst"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.CHAIN_1D

    def test_terpy_honeycomb(self):
        bb = [b for b in ALL_MULTISITE_BACKBONES if b.name == "terpy-trisubst"][0]
        d = design_material(backbone_to_monomer(bb))
        assert d.topology.topology == Topology.HONEYCOMB_2D


# -----------------------------------------------------------------------
# Full catalog
# -----------------------------------------------------------------------

class TestFullCatalog:

    def test_all_20_produce_designs(self):
        results = all_monomer_specs()
        assert len(results) == 20

    def test_all_have_topology(self):
        results = all_monomer_specs()
        for mono, design in results:
            assert design.topology is not None

    def test_all_have_properties(self):
        results = all_monomer_specs()
        for mono, design in results:
            assert design.properties is not None

    def test_frameworks_are_porous(self):
        results = all_monomer_specs()
        frameworks = [(m, d) for m, d in results
                      if d.topology.topology == Topology.FRAMEWORK_3D]
        assert len(frameworks) >= 3
        for m, d in frameworks:
            assert d.properties.porosity_fraction > 0.2

    def test_chains_low_porosity(self):
        results = all_monomer_specs()
        chains = [(m, d) for m, d in results
                  if d.topology.topology == Topology.CHAIN_1D]
        assert len(chains) >= 3
        for m, d in chains:
            assert d.properties.porosity_fraction < 0.2
