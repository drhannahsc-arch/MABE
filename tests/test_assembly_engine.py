"""
tests/test_assembly_engine.py -- Tests for self-assembly design module.

Tests:
  - Face and monomer specifications
  - Stacking energy scoring
  - Topology prediction (chain, honeycomb, framework, cage)
  - Material property estimation
  - Assembly demand generation
  - Library monomers
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.assembly_engine import (
    Face, FaceRole, InteractionMode, MonomerSpec,
    StackingEngine, StackingScore,
    AssemblyGrammar, Topology, TopologyPrediction,
    MaterialPropertyPredictor, MaterialProperties,
    design_material, MaterialDesign,
    assembly_demand,
    urea_tape_monomer, tripodal_linker_monomer,
    mof_paddle_wheel, pi_stacking_monomer,
    _INTERACTION_ENERGY, _DIRECTIONALITY,
)


# -----------------------------------------------------------------------
# Interaction physics
# -----------------------------------------------------------------------

class TestInteractionPhysics:

    def test_all_modes_have_energy(self):
        for mode in InteractionMode:
            assert mode in _INTERACTION_ENERGY

    def test_all_modes_have_directionality(self):
        for mode in InteractionMode:
            assert mode in _DIRECTIONALITY

    def test_energies_are_negative(self):
        for mode, e in _INTERACTION_ENERGY.items():
            assert e < 0, f"{mode}: energy should be negative (favorable)"

    def test_covalent_strongest(self):
        assert abs(_INTERACTION_ENERGY[InteractionMode.COVALENT]) > abs(
            _INTERACTION_ENERGY[InteractionMode.PI_PI])

    def test_coordination_stronger_than_hbond(self):
        assert abs(_INTERACTION_ENERGY[InteractionMode.COORDINATION]) > abs(
            _INTERACTION_ENERGY[InteractionMode.HBOND_NETWORK])


# -----------------------------------------------------------------------
# Face specification
# -----------------------------------------------------------------------

class TestFace:

    def test_dG_per_contact(self):
        f = Face("test", FaceRole.STRUCTURAL, InteractionMode.PI_PI, n_contacts=2)
        assert f.dG_per_contact == -4.0

    def test_dG_total(self):
        f = Face("test", FaceRole.STRUCTURAL, InteractionMode.HBOND_NETWORK, n_contacts=4)
        assert abs(f.dG_total - (-2.25 * 4)) < 0.01

    def test_directionality(self):
        f_hb = Face("hb", FaceRole.STRUCTURAL, InteractionMode.HBOND_NETWORK)
        f_hp = Face("hp", FaceRole.STRUCTURAL, InteractionMode.HYDROPHOBIC)
        assert f_hb.directionality > f_hp.directionality


# -----------------------------------------------------------------------
# Monomer specification
# -----------------------------------------------------------------------

class TestMonomerSpec:

    def test_valence(self):
        m = urea_tape_monomer()
        assert m.valence == 2  # 2 structural faces

    def test_capture_faces(self):
        m = urea_tape_monomer()
        assert m.n_capture_faces == 1  # aromatic face

    def test_structural_faces(self):
        m = mof_paddle_wheel()
        assert m.valence == 4

    def test_face_by_name(self):
        m = urea_tape_monomer()
        f = m.face_by_name("aromatic_face")
        assert f is not None
        assert f.role == FaceRole.CAPTURE


# -----------------------------------------------------------------------
# Stacking engine
# -----------------------------------------------------------------------

class TestStackingEngine:

    def test_score_pair_same_mode(self):
        fa = Face("a", FaceRole.STRUCTURAL, InteractionMode.PI_PI, n_contacts=2, area_A2=100)
        fb = Face("b", FaceRole.STRUCTURAL, InteractionMode.PI_PI, n_contacts=2, area_A2=100)
        sc = StackingEngine.score_pair(fa, fb)
        assert sc.dG_interaction < 0
        assert sc.dG_desolvation > 0
        assert sc.n_contacts == 2

    def test_score_pair_net_favorable(self):
        fa = Face("a", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                  n_contacts=4, area_A2=20)
        fb = Face("b", FaceRole.STRUCTURAL, InteractionMode.COORDINATION,
                  n_contacts=4, area_A2=20)
        sc = StackingEngine.score_pair(fa, fb)
        assert sc.dG_net < 0  # strong coordination should overcome desolvation

    def test_reversibility(self):
        fa = Face("a", FaceRole.STRUCTURAL, InteractionMode.COVALENT, n_contacts=1)
        fb = Face("b", FaceRole.STRUCTURAL, InteractionMode.COVALENT, n_contacts=1)
        sc = StackingEngine.score_pair(fa, fb)
        assert not sc.reversible

    def test_monomer_assembly_scoring(self):
        m = urea_tape_monomer()
        scores = StackingEngine.score_monomer_assembly(m)
        assert len(scores) >= 1
        # Should have complementary face score
        total = sum(s.dG_net for s in scores)
        assert total < 0


# -----------------------------------------------------------------------
# Topology prediction
# -----------------------------------------------------------------------

class TestTopologyPrediction:

    def test_urea_tape_chain(self):
        m = urea_tape_monomer()
        t = AssemblyGrammar.predict(m)
        assert t.topology == Topology.CHAIN_1D
        assert t.dimensionality == 1

    def test_tripodal_honeycomb(self):
        m = tripodal_linker_monomer()
        t = AssemblyGrammar.predict(m)
        assert t.topology == Topology.HONEYCOMB_2D
        assert t.dimensionality == 2

    def test_paddlewheel_framework(self):
        m = mof_paddle_wheel()
        t = AssemblyGrammar.predict(m)
        assert t.topology == Topology.FRAMEWORK_3D
        assert t.dimensionality == 3

    def test_pyrene_chain(self):
        m = pi_stacking_monomer()
        t = AssemblyGrammar.predict(m)
        assert t.topology == Topology.CHAIN_1D

    def test_cage_from_self_complementary(self):
        m = MonomerSpec(
            name="cage-former",
            faces=[
                Face("a", FaceRole.STRUCTURAL, InteractionMode.COVALENT, n_contacts=1),
                Face("b", FaceRole.STRUCTURAL, InteractionMode.COVALENT, n_contacts=1),
                Face("c", FaceRole.STRUCTURAL, InteractionMode.COVALENT, n_contacts=1),
            ],
            symmetry="C3", rigidity=0.9,
        )
        t = AssemblyGrammar.predict(m)
        assert t.topology == Topology.CAGE_0D

    def test_monovalent_dimer(self):
        m = MonomerSpec(
            name="dimer",
            faces=[Face("a", FaceRole.STRUCTURAL, InteractionMode.HYDROPHOBIC)],
        )
        t = AssemblyGrammar.predict(m)
        assert t.topology == Topology.DIMER

    def test_periodicity_correct(self):
        for m, expected_period in [
            (urea_tape_monomer(), "1D-periodic"),
            (tripodal_linker_monomer(), "2D-periodic"),
            (mof_paddle_wheel(), "3D-periodic"),
        ]:
            t = AssemblyGrammar.predict(m)
            assert t.periodicity == expected_period


# -----------------------------------------------------------------------
# Material property prediction
# -----------------------------------------------------------------------

class TestMaterialProperties:

    def test_framework_porous(self):
        m = mof_paddle_wheel()
        d = design_material(m)
        assert d.properties.porosity_fraction > 0.2

    def test_chain_low_porosity(self):
        m = urea_tape_monomer()
        d = design_material(m)
        assert d.properties.porosity_fraction < 0.2

    def test_framework_has_bet(self):
        m = mof_paddle_wheel()
        d = design_material(m)
        assert d.properties.bet_surface_area_m2g > 100

    def test_stability_scales_with_energy(self):
        # Coordination stronger than pi-pi
        m_coord = mof_paddle_wheel()
        m_pi = pi_stacking_monomer()
        d_coord = design_material(m_coord)
        d_pi = design_material(m_pi)
        stab_rank = {"low": 0, "moderate": 1, "high": 2}
        assert stab_rank[d_coord.properties.thermal_stability] >= stab_rank[d_pi.properties.thermal_stability]

    def test_capture_density(self):
        m = urea_tape_monomer()  # has 1 capture face
        d = design_material(m)
        assert d.properties.capture_density > 0

    def test_covalent_irreversible(self):
        m = MonomerSpec(
            name="cov",
            faces=[
                Face("a", FaceRole.STRUCTURAL, InteractionMode.COVALENT, n_contacts=1),
                Face("b", FaceRole.STRUCTURAL, InteractionMode.COVALENT, n_contacts=1),
            ],
            rigidity=0.9,
        )
        d = design_material(m)
        assert not d.properties.reversible


# -----------------------------------------------------------------------
# Full design pipeline
# -----------------------------------------------------------------------

class TestDesignPipeline:

    def test_returns_material_design(self):
        m = urea_tape_monomer()
        d = design_material(m)
        assert isinstance(d, MaterialDesign)
        assert d.monomer is m
        assert d.topology is not None
        assert d.properties is not None

    def test_summary_string(self):
        d = design_material(urea_tape_monomer())
        assert "chain_1D" in d.summary
        assert "anthracene-diurea" in d.summary

    def test_stacking_scores_populated(self):
        d = design_material(urea_tape_monomer())
        assert len(d.stacking_scores) > 0


# -----------------------------------------------------------------------
# Assembly demand
# -----------------------------------------------------------------------

class TestAssemblyDemand:

    def test_framework_demand(self):
        d = assembly_demand("framework_3D", 0.5)
        assert d["required_valence"] == 4
        assert d["recommended_symmetry"] == "Td"
        assert d["structural_interaction"] == "coordination"

    def test_chain_demand(self):
        d = assembly_demand("chain_1D", 0.1)
        assert d["required_valence"] == 2
        assert d["structural_interaction"] == "hbond_net"

    def test_honeycomb_demand(self):
        d = assembly_demand("honeycomb_2D", 0.4)
        assert d["required_valence"] == 3

    def test_porosity_affects_mw(self):
        d_low = assembly_demand("framework_3D", 0.2)
        d_high = assembly_demand("framework_3D", 0.8)
        assert d_high["recommended_mw_range"][0] > d_low["recommended_mw_range"][0]

    def test_capture_mode_passthrough(self):
        d = assembly_demand("framework_3D", 0.5, capture_mode="ch_pi")
        assert d["capture_mode"] == "ch_pi"
