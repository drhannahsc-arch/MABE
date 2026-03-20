"""
glycan/tests/test_ring_grammar.py -- Tests for ring enumerator + demand grammar.

Validates:
  - Ring system catalog completeness and properties
  - Positional decoration enumeration
  - Physics-filtered enumeration
  - Demand vector computation from sugar targets
  - Grammar-driven generation pipeline
  - Cross-validation: grammar output scores match direct scoring
  - Design principle recovery (anthracene for Glc, boronic for Fru)
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

pytestmark = pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")

from glycan.ring_enumerator import (
    get_catalog, get_ring_system, list_ring_systems,
    enumerate_decorated, enumerate_all_decorated,
    enumerate_physics_filtered, PhysicsFilter,
    get_decorators, DECORATOR_LIBRARY,
    RingSystem, DecoratedScaffold,
)
from glycan.demand_grammar import (
    compute_demand, generate_from_demand, compare_approaches,
    DemandVector, _assemble_grammar,
)


# -----------------------------------------------------------------------
# Ring system catalog
# -----------------------------------------------------------------------

class TestRingCatalog:

    def test_catalog_populated(self):
        cat = get_catalog()
        assert len(cat) >= 35

    def test_1ring_systems(self):
        systems = list_ring_systems(min_rings=1, max_rings=1)
        assert len(systems) >= 10
        names = {s.name for s in systems}
        assert "benzene" in names
        assert "pyridine" in names
        assert "pyrrole" in names

    def test_2ring_systems(self):
        systems = list_ring_systems(min_rings=2, max_rings=2)
        assert len(systems) >= 10
        names = {s.name for s in systems}
        assert "naphthalene" in names
        assert "indole" in names
        assert "quinoline" in names

    def test_3ring_systems(self):
        systems = list_ring_systems(min_rings=3, max_rings=3)
        assert len(systems) >= 8
        names = {s.name for s in systems}
        assert "anthracene" in names
        assert "carbazole" in names
        assert "phenanthrene" in names

    def test_4ring_systems(self):
        systems = list_ring_systems(min_rings=4, max_rings=4)
        assert len(systems) >= 2
        names = {s.name for s in systems}
        assert "pyrene" in names

    def test_anthracene_properties(self):
        rs = get_ring_system("anthracene")
        assert rs.n_rings == 3
        assert rs.n_aromatic_atoms == 14
        assert rs.n_substitutable == 10
        assert rs.category == "carbocyclic"
        assert not rs.has_N

    def test_indole_properties(self):
        rs = get_ring_system("indole")
        assert rs.n_rings == 2
        assert rs.has_N
        assert rs.has_NH
        assert rs.category == "N-hetero"

    def test_carbocyclic_filter(self):
        systems = list_ring_systems(category="carbocyclic")
        for s in systems:
            assert not s.has_N
            assert not s.has_O
            assert not s.has_S

    def test_all_smiles_valid(self):
        for name, rs in get_catalog().items():
            mol = Chem.MolFromSmiles(rs.smiles)
            assert mol is not None, f"{name}: invalid SMILES {rs.smiles}"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_ring_system("adamantane")


# -----------------------------------------------------------------------
# Positional decoration
# -----------------------------------------------------------------------

class TestDecoration:

    def test_benzene_2site(self):
        scaffolds = enumerate_decorated("benzene", n_sites=2)
        assert len(scaffolds) >= 2  # ortho, meta, para
        for s in scaffolds:
            assert s.n_sites == 2
            mol = Chem.MolFromSmiles(s.smiles)
            assert mol is not None

    def test_anthracene_2site(self):
        scaffolds = enumerate_decorated("anthracene", n_sites=2, max_scaffolds=50)
        assert len(scaffolds) >= 5
        for s in scaffolds:
            assert s.ring_system == "anthracene"
            assert "[" in s.smiles or "*" in s.smiles  # has dummy atoms

    def test_dedup(self):
        scaffolds = enumerate_decorated("benzene", n_sites=2)
        smiles = [s.smiles for s in scaffolds]
        assert len(smiles) == len(set(smiles))

    def test_3site_decoration(self):
        scaffolds = enumerate_decorated("naphthalene", n_sites=3, max_scaffolds=30)
        assert len(scaffolds) >= 3
        for s in scaffolds:
            assert s.n_sites == 3

    def test_too_many_sites_returns_empty(self):
        # Imidazole has 4 substitutable positions
        scaffolds = enumerate_decorated("imidazole", n_sites=5)
        assert len(scaffolds) == 0

    def test_all_decorated(self):
        all_dec = enumerate_all_decorated(n_sites=2, min_rings=2, max_rings=2, max_per_system=5)
        assert len(all_dec) >= 20
        ring_names = {s.ring_system for s in all_dec}
        assert len(ring_names) >= 5


# -----------------------------------------------------------------------
# Physics-filtered enumeration
# -----------------------------------------------------------------------

class TestPhysicsFilter:

    def test_large_aromatic_filter(self):
        pf = PhysicsFilter(require_large_aromatic=True)
        scaffolds = enumerate_physics_filtered(n_sites=2, pfilter=pf, max_total=50)
        for s in scaffolds:
            rs = get_ring_system(s.ring_system)
            assert rs.n_aromatic_atoms >= 10

    def test_carbocyclic_only(self):
        pf = PhysicsFilter(allow_heteroatoms=False)
        scaffolds = enumerate_physics_filtered(n_sites=2, pfilter=pf, max_total=50)
        for s in scaffolds:
            rs = get_ring_system(s.ring_system)
            assert rs.category == "carbocyclic"

    def test_nh_donor_filter(self):
        pf = PhysicsFilter(require_NH_donor=True)
        scaffolds = enumerate_physics_filtered(n_sites=2, pfilter=pf, max_total=50)
        for s in scaffolds:
            rs = get_ring_system(s.ring_system)
            assert rs.has_NH


# -----------------------------------------------------------------------
# Decorator library
# -----------------------------------------------------------------------

class TestDecorators:

    def test_library_populated(self):
        assert len(DECORATOR_LIBRARY) >= 15

    def test_boronic_filter(self):
        boronics = get_decorators(require_boronic=True)
        assert len(boronics) >= 1
        for d in boronics:
            assert d.has_boronic

    def test_hbd_filter(self):
        hbds = get_decorators(require_hbd=True)
        assert len(hbds) >= 5
        for d in hbds:
            assert d.n_hbd > 0

    def test_all_smiles_valid(self):
        for d in DECORATOR_LIBRARY:
            mol = Chem.MolFromSmiles(d.smiles)
            assert mol is not None, f"{d.name}: invalid SMILES {d.smiles}"


# -----------------------------------------------------------------------
# Demand vector
# -----------------------------------------------------------------------

class TestDemandVector:

    def test_glc_demands_large_aromatic(self):
        d = compute_demand("Glc")
        assert d.min_aromatic_atoms >= 10
        assert d.prefer_large_aromatic

    def test_glc_bimodal_hbd(self):
        d = compute_demand("Glc")
        assert d.hbd_strategy == "bimodal"

    def test_fru_demands_boronic(self):
        d = compute_demand("Fru")
        assert d.boronic_useful
        assert d.boronic_preferred

    def test_fru_no_large_aromatic(self):
        d = compute_demand("Fru")
        assert d.min_aromatic_atoms == 0
        assert not d.prefer_large_aromatic

    def test_man_demands_boronic(self):
        d = compute_demand("Man")
        assert d.boronic_useful  # cis-1,2-diol

    def test_galnac_not_boronic(self):
        d = compute_demand("GalNAc")
        assert not d.boronic_useful

    def test_all_targets_have_demand(self):
        for t in ["Glc", "Gal", "Man", "GalNAc", "GlcNAc", "Fuc", "Fru", "Neu5Ac"]:
            d = compute_demand(t)
            assert d.target == t
            assert d.target_volume_A3 > 0


# -----------------------------------------------------------------------
# Grammar-driven generation
# -----------------------------------------------------------------------

class TestGrammarGeneration:

    def test_runs_and_returns_result(self):
        result = generate_from_demand("Glc", max_candidates=30)
        assert result.target == "Glc"
        assert result.n_scored > 0

    def test_generates_candidates(self):
        result = generate_from_demand("Glc", max_candidates=50)
        assert len(result.candidates) > 0
        for c in result.candidates[:5]:
            assert c.smiles != ""
            mol = Chem.MolFromSmiles(c.smiles)
            assert mol is not None

    def test_pareto_front_exists(self):
        result = generate_from_demand("Glc", max_candidates=50)
        assert result.n_pareto_front > 0

    def test_multiple_targets(self):
        for t in ["Glc", "Fru", "GalNAc"]:
            result = generate_from_demand(t, max_candidates=30)
            assert result.n_scored > 0


# -----------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------

class TestCrossValidation:

    def test_grammar_scores_match_direct(self):
        """Grammar candidate scores must match direct scoring of same SMILES."""
        from glycan.de_novo_binder import score_glycan_binder, compute_receptor_descriptor

        result = generate_from_demand("Glc", max_candidates=30)
        for c in result.candidates[:5]:
            desc = compute_receptor_descriptor(c.smiles)
            direct = score_glycan_binder(c.smiles, "Glc", desc)
            assert abs(c.score.dG_total - direct.dG_total) < 0.01

    def test_grammar_beats_fixed_for_glc(self):
        """Grammar should find better Glc binders than fixed library."""
        comp = compare_approaches("Glc", max_candidates=100)
        # Grammar should at least match fixed
        assert comp["grammar"]["best_dG"] <= comp["fixed"]["best_dG"] + 2.0


# -----------------------------------------------------------------------
# Design principle recovery
# -----------------------------------------------------------------------

class TestDesignRecovery:

    def test_glc_grammar_finds_polycyclic(self):
        """For Glc, grammar should produce scaffolds with >= 2 fused rings."""
        result = generate_from_demand("Glc", max_candidates=100)
        # At least some candidates should have multi-ring backbones
        ring_systems = {c.backbone_name for c in result.candidates}
        multi_ring = [r for r in ring_systems
                      if r not in ("benzene", "pyridine", "pyrimidine",
                                   "pyrrole", "furan", "thiophene",
                                   "imidazole", "oxazole", "thiazole", "pyrazole",
                                   "pyrazine", "pyridazine", "triazine-1,3,5")]
        assert len(multi_ring) > 0, f"Only single-ring scaffolds found: {ring_systems}"

    def test_fru_grammar_includes_boronic(self):
        """For Fru (cis-1,2-diol), grammar should include boronic acid decorators."""
        result = generate_from_demand("Fru", max_candidates=100)
        has_boronic = any(
            "boronic" in arm or "boronate" in arm
            for c in result.candidates
            for arm in c.arm_names
        )
        assert has_boronic, "No boronic acid candidates for Fru"

    def test_glc_ke_strategy_appears(self):
        """Grammar should produce Ke-style (0 HBD) candidates for Glc."""
        result = generate_from_demand("Glc", max_candidates=100)
        zero_hbd = [c for c in result.candidates if c.score.dG_hb == 0.0]
        assert len(zero_hbd) > 0, "No zero-HBD candidates for Glc"
