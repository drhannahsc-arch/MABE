"""
tests/test_modular_stack.py -- Tests for property predictor, multisite assembler,
reaction assembler, and construct assembler.
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


# -----------------------------------------------------------------------
# Property predictor
# -----------------------------------------------------------------------

from core.property_predictor import (
    predict_properties, filter_properties, PropertyGate,
    predict_batch, filter_batch, MolecularProperties,
)

class TestPropertyPredictor:

    def test_benzene_basic(self):
        p = predict_properties("c1ccccc1")
        assert p.valid
        assert 75 < p.molecular_weight < 80
        assert 1.0 < p.logP < 2.5
        assert p.solubility_class == "moderate"

    def test_glucose_soluble(self):
        p = predict_properties("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
        assert p.valid
        assert p.logP < 0
        assert p.solubility_class == "very_soluble"

    def test_anthracene_poor_sol(self):
        p = predict_properties("c1ccc2cc3ccccc3cc2c1")
        assert p.valid
        assert p.solubility_class in ("poor", "insoluble")

    def test_aspirin_pka(self):
        p = predict_properties("CC(=O)Oc1ccccc1C(=O)O")
        assert p.valid
        assert p.strongest_acidic_pka < 6  # carboxylic acid ~4
        assert "carboxylic_acid" in str(p.acidic_groups)

    def test_amine_pka(self):
        p = predict_properties("NCCN")
        assert p.valid
        assert p.strongest_basic_pka > 8

    def test_ester_hydrolyzable(self):
        p = predict_properties("CC(=O)OCC")
        assert p.valid
        assert not p.aqueous_stable
        assert "ester" in p.hydrolyzable_groups

    def test_lipinski(self):
        p = predict_properties("c1ccccc1")
        assert p.lipinski_pass

    def test_invalid_smiles(self):
        p = predict_properties("not_a_molecule")
        assert not p.valid

    def test_property_gate_pass(self):
        gate = PropertyGate(max_mw=200, max_logP=3)
        ok, props, reasons = filter_properties("c1ccccc1", gate)
        assert ok

    def test_property_gate_fail(self):
        gate = PropertyGate(max_mw=50)
        ok, props, reasons = filter_properties("c1ccccc1", gate)
        assert not ok
        assert any("MW" in r for r in reasons)

    def test_batch(self):
        results = predict_batch(["c1ccccc1", "CCO", "INVALID"])
        assert len(results) == 3
        assert results[0].valid
        assert results[1].valid
        assert not results[2].valid


# -----------------------------------------------------------------------
# Multisite assembler
# -----------------------------------------------------------------------

from core.multisite_assembler import (
    get_multisite_backbones, validate_backbones,
    TRIPODAL_BACKBONES, MACROCYCLIC_BACKBONES,
    CAGE_BACKBONES, TETRADENTATE_BACKBONES,
    ALL_MULTISITE_BACKBONES,
    multisite_enumerate,
)

class TestMultisiteAssembler:

    def test_all_valid(self):
        v = validate_backbones()
        for name, (ok, err) in v.items():
            assert ok, f"{name}: {err}"

    def test_total_count(self):
        assert len(ALL_MULTISITE_BACKBONES) == 20

    def test_tripodal_3site(self):
        bbs = get_multisite_backbones(n_sites=3, categories=["tripodal"])
        assert len(bbs) >= 3
        for bb in bbs:
            assert bb.n_sites == 3

    def test_macrocyclic_count(self):
        bbs = get_multisite_backbones(categories=["macrocyclic"])
        assert len(bbs) >= 5

    def test_cage_count(self):
        bbs = get_multisite_backbones(categories=["cage"])
        assert len(bbs) >= 3

    def test_davis_cage_present(self):
        bbs = get_multisite_backbones(categories=["cage"])
        names = {b.name for b in bbs}
        assert "bis-anthracene-diurea" in names

    def test_tetradentate(self):
        bbs = get_multisite_backbones(n_sites=4)
        assert len(bbs) >= 3

    def test_cyclam_4site(self):
        bbs = get_multisite_backbones(categories=["macrocyclic"])
        cyclam = [b for b in bbs if b.name == "cyclam"]
        assert len(cyclam) == 1
        assert cyclam[0].n_sites == 4

    def test_enumerate_works(self):
        raw = multisite_enumerate(n_sites=3, categories=["tripodal"],
                                  max_candidates=10, hsab_filter=False)
        assert len(raw) > 0


# -----------------------------------------------------------------------
# Reaction assembler
# -----------------------------------------------------------------------

from core.reaction_assembler import (
    identify_reactions, validate_synthesis, annotate_candidate,
    REACTION_LIBRARY, SynthesisRoute,
)

class TestReactionAssembler:

    def test_library_populated(self):
        assert len(REACTION_LIBRARY) >= 12

    def test_urea_detected(self):
        # Molecule with urea: NC(=O)Nc1ccccc1
        rxns = identify_reactions("NC(=O)Nc1ccccc1")
        names = {a.reaction_name for a in rxns}
        assert "urea_formation" in names

    def test_amide_detected(self):
        rxns = identify_reactions("CC(=O)Nc1ccccc1")
        names = {a.reaction_name for a in rxns}
        assert "amide_coupling" in names

    def test_synthesis_route(self):
        synth = validate_synthesis("NC(=O)Nc1ccc2cc3ccccc3cc2c1")
        assert synth.valid
        assert synth.annotation_coverage > 0
        assert len(synth.reactions_used) > 0

    def test_benzene_no_assembly_bonds(self):
        synth = validate_synthesis("c1ccccc1")
        assert synth.valid
        # Benzene has no non-ring bonds to annotate
        assert synth.annotation_coverage == 1.0  # 0/0 = 1.0

    def test_annotate_candidate(self):
        ac = annotate_candidate(
            "NC(=O)Nc1ccc2cc3ccccc3cc2c1",
            min_synthesis_coverage=0.3,
        )
        assert ac.properties is not None
        assert ac.synthesis is not None
        assert ac.passes_synthesis_check


# -----------------------------------------------------------------------
# Construct assembler
# -----------------------------------------------------------------------

from core.construct_assembler import (
    assemble_construct, ConstructSpec, suggest_construct,
    compatible_pairs, LINKER_LIBRARY, CLICK_HANDLE_LIBRARY,
    SUPPORT_LIBRARY,
)

class TestConstructAssembler:

    def test_linker_library(self):
        assert len(LINKER_LIBRARY) >= 7

    def test_click_library(self):
        assert len(CLICK_HANDLE_LIBRARY) >= 5

    def test_support_library(self):
        assert len(SUPPORT_LIBRARY) >= 5

    def test_compatible_pairs(self):
        pairs = compatible_pairs()
        assert len(pairs) >= 4
        # azide + DBCO should be compatible
        azide_dbco = [p for p in pairs if p[0] == "azide" and "DBCO" in p[1]]
        assert len(azide_dbco) >= 1

    def test_basic_assembly(self):
        spec = ConstructSpec(
            recognition_smiles="[*]c1ccccc1",
            linker="PEG4",
            click_handle="azide",
            support="DBCO-Fe3O4",
        )
        result = assemble_construct(spec)
        assert result.soluble_valid
        assert result.molecular_weight > 0
        assert result.click_compatible

    def test_anthracene_construct(self):
        spec = ConstructSpec(
            recognition_smiles="[*]c1ccc2cc3ccccc3cc2c1",
            linker="PEG4",
            click_handle="azide",
            support="DBCO-Fe3O4",
            target="Glc",
        )
        result = assemble_construct(spec)
        assert result.soluble_valid
        assert result.click_compatible
        assert result.support_magnetic
        assert "magnetic_pulldown" in result.readout_options

    def test_incompatible_click_support(self):
        spec = ConstructSpec(
            recognition_smiles="[*]c1ccccc1",
            linker="PEG4",
            click_handle="azide",
            support="NH2-Fe3O4",  # NH2 doesn't react with azide
        )
        result = assemble_construct(spec)
        assert not result.click_compatible

    def test_suggest_construct(self):
        spec = suggest_construct("[*]c1ccccc1", target="Glc")
        assert spec.linker == "PEG4"
        assert spec.click_handle == "azide"
        assert "DBCO" in spec.support

    def test_no_attachment_point(self):
        spec = ConstructSpec(
            recognition_smiles="c1ccccc1",  # no [*]
            linker="PEG4",
            click_handle="azide",
            support="DBCO-Fe3O4",
        )
        result = assemble_construct(spec)
        assert len(result.errors) > 0

    def test_direct_linker(self):
        spec = ConstructSpec(
            recognition_smiles="[*]c1ccccc1",
            linker="none",
            click_handle="azide",
            support="DBCO-Fe3O4",
        )
        result = assemble_construct(spec)
        assert result.soluble_valid

    def test_biotin_streptavidin(self):
        spec = ConstructSpec(
            recognition_smiles="[*]c1ccccc1",
            linker="PEG4",
            click_handle="biotin",
            support="streptavidin-Fe3O4",
        )
        result = assemble_construct(spec)
        assert result.click_compatible
        assert result.click_reaction == "biotin-SA"
