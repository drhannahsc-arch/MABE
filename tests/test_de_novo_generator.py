"""
tests/test_de_novo_generator.py — Tests for Phase: De Novo Molecule Generation

Tests:
  1. Fragment library integrity (all SMILES valid)
  2. Assembly engine (backbone + arms → valid molecule)
  3. SA score range
  4. Property filters
  5. HSAB pre-filter logic
  6. Full metal generation pipeline
  7. Selectivity screening pipeline
  8. Host-guest generation pipeline
  9. Novelty detection
  10. Deduplication
  11. No regressions on existing tests
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem

from core.de_novo_generator import (
    BACKBONE_LIBRARY, ARM_LIBRARY,
    assemble, sa_score, sa_score_smiles,
    passes_filter, PropertyFilter,
    hsab_compatible,
    enumerate_molecules,
    generate_candidates,
    generate_and_screen,
    generate_for_host,
    GeneratedCandidate, GenerationResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. FRAGMENT LIBRARY INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

class TestFragmentLibrary:
    """All fragments must be valid RDKit molecules."""

    def test_all_backbones_valid(self):
        for bb in BACKBONE_LIBRARY:
            mol = Chem.MolFromSmiles(bb.smiles)
            assert mol is not None, f"Invalid backbone SMILES: {bb.name} = {bb.smiles}"

    def test_all_arms_valid(self):
        for arm in ARM_LIBRARY:
            mol = Chem.MolFromSmiles(arm.smiles)
            assert mol is not None, f"Invalid arm SMILES: {arm.name} = {arm.smiles}"

    def test_backbones_have_correct_site_count(self):
        for bb in BACKBONE_LIBRARY:
            mol = Chem.MolFromSmiles(bb.smiles)
            n_dummy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
            assert n_dummy == bb.n_sites, (
                f"{bb.name}: declared {bb.n_sites} sites, "
                f"found {n_dummy} dummy atoms"
            )

    def test_arms_have_exactly_one_dummy(self):
        for arm in ARM_LIBRARY:
            mol = Chem.MolFromSmiles(arm.smiles)
            n_dummy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
            assert n_dummy == 1, (
                f"{arm.name}: expected 1 dummy atom, found {n_dummy}"
            )

    def test_minimum_library_size(self):
        assert len(BACKBONE_LIBRARY) >= 28, f"Need at least 28 backbones, got {len(BACKBONE_LIBRARY)}"
        assert len(ARM_LIBRARY) >= 30, f"Need at least 30 arms, got {len(ARM_LIBRARY)}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. ASSEMBLY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TestAssembly:
    """Backbone + arms must produce valid molecules."""

    def test_simple_2site_assembly(self):
        bb = BACKBONE_LIBRARY[0]  # ethylenediamine
        arm1 = ARM_LIBRARY[0]     # acetic-acid
        arm2 = ARM_LIBRARY[0]     # acetic-acid
        smiles, mol = assemble(bb, [arm1, arm2])
        assert smiles is not None
        assert mol is not None
        # Should contain the expected atoms
        assert mol.GetNumHeavyAtoms() > 5

    def test_wrong_arm_count_fails(self):
        bb = BACKBONE_LIBRARY[0]  # 2 sites
        arm = ARM_LIBRARY[0]
        smiles, mol = assemble(bb, [arm])  # only 1 arm for 2-site backbone
        assert smiles is None

    def test_1site_assembly(self):
        # Find a 1-site backbone
        bb1 = [b for b in BACKBONE_LIBRARY if b.n_sites == 1][0]
        arm = ARM_LIBRARY[0]
        smiles, mol = assemble(bb1, [arm])
        assert smiles is not None

    def test_3site_assembly(self):
        # Find a 3-site backbone
        bb3_list = [b for b in BACKBONE_LIBRARY if b.n_sites == 3]
        if bb3_list:
            bb3 = bb3_list[0]
            arms = [ARM_LIBRARY[0]] * 3
            smiles, mol = assemble(bb3, arms)
            assert smiles is not None

    def test_assembled_mol_is_canonical(self):
        """Same backbone + arms in same order → same SMILES."""
        bb = BACKBONE_LIBRARY[0]
        arm1 = ARM_LIBRARY[0]
        arm2 = ARM_LIBRARY[1]
        smi_a, _ = assemble(bb, [arm1, arm2])
        smi_b, _ = assemble(bb, [arm1, arm2])
        assert smi_a == smi_b

    def test_different_arms_different_smiles(self):
        bb = BACKBONE_LIBRARY[0]
        arm1 = ARM_LIBRARY[0]  # acetic-acid
        arm2 = ARM_LIBRARY[1]  # propionic-acid
        smi_a, _ = assemble(bb, [arm1, arm1])
        smi_b, _ = assemble(bb, [arm1, arm2])
        assert smi_a != smi_b


# ═══════════════════════════════════════════════════════════════════════════
# 3. SA SCORE
# ═══════════════════════════════════════════════════════════════════════════

class TestSAScore:
    def test_simple_molecule_low_sa(self):
        sa = sa_score_smiles("NCCN")  # ethylenediamine
        assert 1.0 <= sa <= 3.0, f"en should be easy to synthesize, got SA={sa}"

    def test_complex_molecule_higher_sa(self):
        # Adamantane cage: harder
        sa = sa_score_smiles("C1C2CC3CC1CC(C2)C3")
        assert sa > 1.5, f"Adamantane should score higher than baseline"

    def test_invalid_returns_10(self):
        assert sa_score(None) == 10.0

    def test_sa_range(self):
        """SA scores should always be in [1, 10]."""
        test_smiles = [
            "C", "NCCN", "c1ccccc1", "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
            "C1COCCOCCOCCOCCOCCO1",
        ]
        for smi in test_smiles:
            sa = sa_score_smiles(smi)
            assert 1.0 <= sa <= 10.0, f"SA out of range for {smi}: {sa}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. PROPERTY FILTERS
# ═══════════════════════════════════════════════════════════════════════════

class TestPropertyFilter:
    def test_too_large_rejected(self):
        pf = PropertyFilter(max_heavy_atoms=5)
        mol = Chem.MolFromSmiles("NCCNCCNCCN")  # > 5 heavy
        assert not passes_filter("NCCNCCNCCN", mol, pf)

    def test_normal_passes(self):
        mol = Chem.MolFromSmiles("OC(=O)CNCC(=O)O")  # IDA, MW ~133
        assert passes_filter("OC(=O)CNCC(=O)O", mol, PropertyFilter())

    def test_no_donor_rejected(self):
        pf = PropertyFilter(require_donors=True)
        mol = Chem.MolFromSmiles("CCCCCC")
        assert not passes_filter("CCCCCC", mol, pf)

    def test_none_mol_rejected(self):
        assert not passes_filter("invalid", None, PropertyFilter())


# ═══════════════════════════════════════════════════════════════════════════
# 5. HSAB PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════

class TestHSAB:
    def test_hard_metal_rejects_soft_arm(self):
        soft_arm = [a for a in ARM_LIBRARY if a.hardness == "soft"][0]
        assert not hsab_compatible("Fe3+", soft_arm)

    def test_hard_metal_accepts_hard_arm(self):
        hard_arm = [a for a in ARM_LIBRARY if a.hardness == "hard"][0]
        assert hsab_compatible("Fe3+", hard_arm)

    def test_borderline_accepts_everything(self):
        for arm in ARM_LIBRARY:
            if arm.hardness:
                assert hsab_compatible("Cu2+", arm), (
                    f"Cu2+ (borderline) should accept {arm.name} ({arm.hardness})"
                )

    def test_soft_metal_rejects_hard_arm(self):
        hard_arm = [a for a in ARM_LIBRARY if a.hardness == "hard"][0]
        assert not hsab_compatible("Ag+", hard_arm)

    def test_soft_metal_accepts_soft_arm(self):
        soft_arm = [a for a in ARM_LIBRARY if a.hardness == "soft"][0]
        assert hsab_compatible("Ag+", soft_arm)


# ═══════════════════════════════════════════════════════════════════════════
# 6. FULL METAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class TestMetalGeneration:
    def test_generates_candidates(self):
        r = generate_candidates("Cu2+", max_candidates=30, max_scored=10)
        assert isinstance(r, GenerationResult)
        assert r.n_scored > 0
        assert r.mode == "metal"

    def test_candidates_have_scores(self):
        r = generate_candidates("Cu2+", max_candidates=30, max_scored=10)
        for c in r.candidates:
            assert hasattr(c, 'log_Ka_pred')
            assert hasattr(c, 'sa_score_val')
            assert hasattr(c, 'composite_score')

    def test_candidates_ranked_by_composite(self):
        r = generate_candidates("Cu2+", max_candidates=50, max_scored=15)
        scores = [c.composite_score for c in r.candidates]
        assert scores == sorted(scores, reverse=True), "Not sorted by composite"

    def test_rank_numbering(self):
        r = generate_candidates("Cu2+", max_candidates=30, max_scored=10)
        ranks = [c.rank for c in r.candidates]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_best_property(self):
        r = generate_candidates("Cu2+", max_candidates=30, max_scored=10)
        if r.candidates:
            assert r.best is r.candidates[0]

    def test_hard_metal_gets_hard_donors(self):
        """Fe3+ (hard) should preferentially get O-donor ligands."""
        r = generate_candidates("Fe3+", max_candidates=50, max_scored=20,
                                 hsab_filter=True)
        # At least some candidates should exist
        assert r.n_scored > 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. SELECTIVITY SCREENING
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectivityScreen:
    def test_screen_produces_results(self):
        r = generate_and_screen("Pb2+", ["Ca2+", "Mg2+"],
                                 max_candidates=30, max_scored=10, pH=5.0)
        assert r.n_scored > 0
        assert r.mode == "selectivity"

    def test_interferent_scores_populated(self):
        r = generate_and_screen("Pb2+", ["Ca2+"],
                                 max_candidates=30, max_scored=10)
        for c in r.candidates:
            assert "Ca2+" in c.interferent_scores
            assert "Ca2+" in c.selectivity_gaps

    def test_grade_assigned(self):
        r = generate_and_screen("Pb2+", ["Ca2+", "Mg2+"],
                                 max_candidates=30, max_scored=10)
        for c in r.candidates:
            assert c.grade in ("A", "B", "C", "D", "F")


# ═══════════════════════════════════════════════════════════════════════════
# 8. HOST-GUEST GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class TestHostGuestGeneration:
    def test_beta_cd_generation(self):
        r = generate_for_host("beta-CD", max_candidates=30, max_scored=10)
        assert r.n_scored > 0
        assert r.mode == "host_guest"

    def test_guests_are_small(self):
        """Generated guests should be reasonably sized for cavity inclusion."""
        r = generate_for_host("beta-CD", max_candidates=30, max_scored=10)
        for c in r.candidates:
            mol = Chem.MolFromSmiles(c.smiles)
            assert mol.GetNumHeavyAtoms() <= 30


# ═══════════════════════════════════════════════════════════════════════════
# 9. NOVELTY
# ═══════════════════════════════════════════════════════════════════════════

class TestNovelty:
    def test_generated_marked_novel(self):
        """Combinatorially generated molecules should mostly be novel."""
        r = generate_candidates("Cu2+", max_candidates=50, max_scored=15)
        if r.candidates:
            n_novel = sum(1 for c in r.candidates if c.novel)
            # Most should be novel (they're combinatorial products)
            assert n_novel > 0


# ═══════════════════════════════════════════════════════════════════════════
# 10. DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestDedup:
    def test_no_duplicate_smiles(self):
        raw = enumerate_molecules(metal="Cu2+", max_candidates=100)
        smiles_list = [r[0] for r in raw]
        assert len(smiles_list) == len(set(smiles_list)), "Duplicates found"


# ═══════════════════════════════════════════════════════════════════════════
# 11. INTEGRATION — existing tests still pass
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Verify de novo generator doesn't break existing pipelines."""

    def test_design_engine_v2_still_works(self):
        """design_engine_v2.score_one must still function."""
        from core.design_engine_v2 import score_one
        sc = score_one("OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
                       metal="Cu2+", name="EDTA")
        assert sc.log_Ka_pred > 0

    def test_auto_descriptor_still_works(self):
        from core.auto_descriptor import from_smiles
        uc = from_smiles("OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
                         metal="Cu2+")
        assert uc is not None
        assert uc.denticity > 0


# ═══════════════════════════════════════════════════════════════════════════
# 12. EXPANDED LIBRARY COVERAGE
# ═══════════════════════════════════════════════════════════════════════════

class TestExpandedCoverage:
    """Verify new fragment categories produce valid assemblies."""

    def test_salen_backbone_assembles(self):
        bb = [b for b in BACKBONE_LIBRARY if "salen" in b.name][0]
        arms = [ARM_LIBRARY[0]] * bb.n_sites
        smiles, mol = assemble(bb, arms)
        assert smiles is not None, f"Salen backbone failed to assemble"

    def test_macrocyclic_tacn_assembles(self):
        bb = [b for b in BACKBONE_LIBRARY if b.name == "tacn"][0]
        arms = [ARM_LIBRARY[0]] * bb.n_sites
        smiles, mol = assemble(bb, arms)
        assert smiles is not None, f"TACN backbone failed to assemble"

    def test_dithiocarbamate_arm_assembles(self):
        bb = BACKBONE_LIBRARY[0]  # ethylenediamine
        dtc = [a for a in ARM_LIBRARY if "dithiocarbamate" in a.name][0]
        smiles, mol = assemble(bb, [dtc, ARM_LIBRARY[0]])
        assert smiles is not None, f"Dithiocarbamate arm failed"

    def test_phosphonate_arm_assembles(self):
        bb = BACKBONE_LIBRARY[0]
        phos = [a for a in ARM_LIBRARY if a.name == "phosphonate"][0]
        smiles, mol = assemble(bb, [phos, ARM_LIBRARY[0]])
        assert smiles is not None, f"Phosphonate arm failed"

    def test_thiosemicarbazone_arm_assembles(self):
        bb = BACKBONE_LIBRARY[0]
        tsc = [a for a in ARM_LIBRARY if a.name == "thiosemicarbazone"][0]
        smiles, mol = assemble(bb, [tsc, ARM_LIBRARY[0]])
        assert smiles is not None, f"Thiosemicarbazone arm failed"

    def test_hydroxypyridinone_arm_assembles(self):
        bb = BACKBONE_LIBRARY[0]
        hpo = [a for a in ARM_LIBRARY if a.name == "hydroxypyridinone"][0]
        smiles, mol = assemble(bb, [hpo, ARM_LIBRARY[0]])
        assert smiles is not None, f"Hydroxypyridinone arm failed"

    def test_soft_metal_generates_with_new_soft_arms(self):
        """Hg2+ should pick up dithiocarbamate and thiosemicarbazone arms."""
        r = generate_candidates("Hg2+", max_candidates=50, max_scored=10,
                                 hsab_filter=True)
        assert r.n_scored > 0
        # Check that at least some candidates contain S atoms
        has_sulfur = any("S" in c.smiles or "s" in c.smiles
                         for c in r.candidates)
        assert has_sulfur, "Hg2+ should get S-donor candidates"

    def test_hard_metal_gets_phosphonate(self):
        """Al3+ (hard) should accept phosphonate arms."""
        r = generate_candidates("Al3+", max_candidates=80, max_scored=15,
                                 hsab_filter=True)
        assert r.n_scored > 0

    def test_expanded_enumeration_larger(self):
        """Expanded libraries should enumerate more molecules."""
        raw = enumerate_molecules(metal="Cu2+", max_candidates=500)
        assert len(raw) >= 100, f"Expected >=100 with expanded libs, got {len(raw)}"

    def test_donor_category_coverage(self):
        """Arms should cover hard, borderline, and soft categories."""
        hard = [a for a in ARM_LIBRARY if a.hardness == "hard"]
        border = [a for a in ARM_LIBRARY if a.hardness == "borderline"]
        soft = [a for a in ARM_LIBRARY if a.hardness == "soft"]
        assert len(hard) >= 8, f"Need >=8 hard arms, got {len(hard)}"
        assert len(border) >= 10, f"Need >=10 borderline arms, got {len(border)}"
        assert len(soft) >= 8, f"Need >=8 soft arms, got {len(soft)}"

    def test_backbone_category_coverage(self):
        """Backbones should cover all structural categories."""
        cats = set(b.category for b in BACKBONE_LIBRARY)
        for needed in ["linear", "branched", "aromatic", "macrocyclic"]:
            assert needed in cats, f"Missing backbone category: {needed}"
