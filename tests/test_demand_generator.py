"""
tests/test_demand_generator.py -- Tests for core demand generator.

Validates:
  - Core ring enumerator (catalog, decoration, physics filter, conversion)
  - Metal demand vectors (HSAB logic)
  - Host-guest demand vectors (Rebek packing)
  - Glycan demand delegation
  - Unified generate_from_demand pipeline (all three modalities)
  - Backbone/Arm conversion for existing pipeline compatibility
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

from core.ring_enumerator import (
    get_catalog, get_ring_system, list_ring_systems,
    enumerate_decorated, enumerate_physics_filtered,
    PhysicsFilter, DECORATOR_LIBRARY,
    grammar_backbones, grammar_arms,
    scaffold_to_backbone, decorator_to_arm,
    get_decorators,
)
from core.demand_generator import (
    metal_demand, host_guest_demand, glycan_demand,
    generate_from_demand, DemandVector,
)


# -----------------------------------------------------------------------
# Core ring enumerator at core level
# -----------------------------------------------------------------------

class TestCoreRingEnumerator:

    def test_catalog_at_core(self):
        cat = get_catalog()
        assert len(cat) >= 35

    def test_anthracene_at_core(self):
        rs = get_ring_system("anthracene")
        assert rs.n_rings == 3
        assert rs.n_aromatic_atoms == 14

    def test_decoration_at_core(self):
        d = enumerate_decorated("benzene", n_sites=2)
        assert len(d) >= 2

    def test_physics_filter_at_core(self):
        pf = PhysicsFilter(require_large_aromatic=True)
        scaffolds = enumerate_physics_filtered(n_sites=2, pfilter=pf, max_total=10)
        for s in scaffolds:
            rs = get_ring_system(s.ring_system)
            assert rs.n_aromatic_atoms >= 10


# -----------------------------------------------------------------------
# Backbone/Arm conversion
# -----------------------------------------------------------------------

class TestConversion:

    def test_scaffold_to_backbone(self):
        d = enumerate_decorated("anthracene", n_sites=2, max_scaffolds=1)
        bb = scaffold_to_backbone(d[0])
        assert bb.n_sites == 2
        assert "ring:" in bb.name
        assert bb.category == "cyclic"

    def test_decorator_to_arm(self):
        dec = DECORATOR_LIBRARY[0]
        arm = decorator_to_arm(dec)
        assert arm.name.startswith("dec:")
        assert arm.smiles == dec.smiles

    def test_grammar_backbones(self):
        bbs = grammar_backbones(n_sites=2, max_total=10)
        assert len(bbs) > 0
        for bb in bbs:
            assert bb.n_sites == 2

    def test_grammar_arms(self):
        arms = grammar_arms()
        assert len(arms) >= 20

    def test_assembly_works(self):
        """Grammar backbones + arms assemble through existing pipeline."""
        from core.de_novo_generator import assemble
        bbs = grammar_backbones(n_sites=2, max_total=3)
        arms = grammar_arms()
        assembled = 0
        for bb in bbs:
            for arm in arms[:3]:
                smi, mol = assemble(bb, [arm, arm])
                if smi is not None:
                    assembled += 1
        assert assembled > 0

    def test_enumerate_molecules_accepts_grammar(self):
        """Grammar backbones/arms feed into existing enumerate_molecules."""
        from core.de_novo_generator import enumerate_molecules
        bbs = grammar_backbones(n_sites=2, max_total=5)
        arms = grammar_arms(categories=["HBD", "HBA"])
        raw = enumerate_molecules(backbones=bbs, arms=arms, max_candidates=10,
                                  hsab_filter=False)
        assert len(raw) > 0


# -----------------------------------------------------------------------
# Decorator library (unified)
# -----------------------------------------------------------------------

class TestUnifiedDecorators:

    def test_has_metal_donors(self):
        donors = get_decorators(categories=["donor"])
        assert len(donors) >= 5
        # Should have both N and S donors
        elements = {d.donor_element for d in donors}
        assert "N" in elements
        assert "S" in elements

    def test_has_glycan_features(self):
        hbd = get_decorators(require_hbd=True)
        assert len(hbd) >= 5
        boronic = get_decorators(require_boronic=True)
        assert len(boronic) >= 1

    def test_hardness_filter(self):
        hard = get_decorators(hardness="hard")
        for d in hard:
            assert d.hardness == "hard" or d.hardness == ""

    def test_all_smiles_valid(self):
        for d in DECORATOR_LIBRARY:
            mol = Chem.MolFromSmiles(d.smiles)
            assert mol is not None, f"{d.name}: invalid {d.smiles}"


# -----------------------------------------------------------------------
# Metal demand
# -----------------------------------------------------------------------

class TestMetalDemand:

    def test_cu2_borderline(self):
        d = metal_demand("Cu2+")
        assert d.mode == "metal"
        assert d.required_hardness is None  # borderline accepts all
        assert d.require_NH  # borderline prefers N-heterocycles

    def test_fe3_hard(self):
        d = metal_demand("Fe3+")
        assert d.required_hardness == "hard"
        assert d.required_donor_element == "O"

    def test_ag_soft(self):
        d = metal_demand("Ag+")
        assert d.required_hardness == "soft"
        assert d.required_donor_element == "S"

    def test_pb2_borderline(self):
        d = metal_demand("Pb2+")
        assert d.required_hardness is None  # borderline


# -----------------------------------------------------------------------
# Host-guest demand
# -----------------------------------------------------------------------

class TestHostGuestDemand:

    def test_beta_cd(self):
        d = host_guest_demand("beta-CD")
        assert d.mode == "host_guest"
        # Rebek: optimal guest ~55% of 262 A3 = ~144 A3
        assert 130 < d.target_volume_A3 < 160

    def test_alpha_cd_smaller(self):
        d_a = host_guest_demand("alpha-CD")
        d_b = host_guest_demand("beta-CD")
        assert d_a.target_volume_A3 < d_b.target_volume_A3

    def test_custom_volume(self):
        d = host_guest_demand("custom", cavity_volume=500.0)
        assert 250 < d.target_volume_A3 < 350


# -----------------------------------------------------------------------
# Glycan demand delegation
# -----------------------------------------------------------------------

class TestGlycanDemand:

    def test_glc_demand(self):
        d = glycan_demand("Glc")
        assert d.mode == "glycan"
        assert d.min_aromatic_atoms >= 10
        assert d.hbd_strategy == "bimodal"

    def test_fru_boronic(self):
        d = glycan_demand("Fru")
        assert d.require_boronic


# -----------------------------------------------------------------------
# Unified generation
# -----------------------------------------------------------------------

class TestUnifiedGeneration:

    def test_metal_generation(self):
        r = generate_from_demand("metal", "Cu2+", max_candidates=50, max_scored=10)
        assert r.n_enumerated > 0
        # May have 0 scored if scorer fails on some, but enumeration works
        assert r.mode == "metal"

    def test_host_guest_generation(self):
        r = generate_from_demand("host_guest", "beta-CD", max_candidates=50, max_scored=10)
        assert r.n_enumerated > 0
        assert r.mode == "host_guest"

    def test_glycan_generation(self):
        r = generate_from_demand("glycan", "Glc", max_candidates=50)
        assert r.n_scored > 0
        assert r.n_pareto_front > 0

    def test_metal_produces_scored(self):
        r = generate_from_demand("metal", "Cu2+", max_candidates=100, max_scored=20)
        assert r.n_scored > 0
        assert r.candidates[0].log_Ka_pred > 0

    def test_host_guest_produces_scored(self):
        r = generate_from_demand("host_guest", "beta-CD", max_candidates=100, max_scored=20)
        assert r.n_scored > 0
        assert r.candidates[0].log_Ka_pred > 0


# -----------------------------------------------------------------------
# Cross-modality consistency
# -----------------------------------------------------------------------

class TestCrossModality:

    def test_same_ring_catalog(self):
        """Core and glycan ring catalogs are identical."""
        from glycan.ring_enumerator import get_catalog as glycan_cat
        core_cat = get_catalog()
        g_cat = glycan_cat()
        assert set(core_cat.keys()) == set(g_cat.keys())

    def test_glycan_imports_from_core(self):
        """Glycan re-export works."""
        from glycan.ring_enumerator import RingSystem, DecoratedScaffold
        from core.ring_enumerator import RingSystem as CoreRS
        assert RingSystem is CoreRS
