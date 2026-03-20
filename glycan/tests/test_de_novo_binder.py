"""
glycan/tests/test_de_novo_binder.py -- Tests for glycan de novo binder design.

Validates:
  - Sugar target database (8 entries, all fields)
  - Receptor descriptor computation (aromatic rings, HBD, boronic acid)
  - Scoring physics (CH-pi, H-bond, desolvation, boronic, shape)
  - Selectivity scoring (Glc-selective receptor prefers Glc over Gal/Man)
  - Generator pipeline runs and returns correct types
  - Pareto front extraction
  - Known design principles (Davis-type receptors)
  - Physical sanity
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

from glycan.de_novo_binder import (
    SugarTarget,
    get_sugar_target,
    list_sugar_targets,
    compute_receptor_descriptor,
    score_glycan_binder,
    selectivity_score,
    generate_glycan_binders,
    GlycanBinderSpec,
    GlycanBinderResult,
    BinderScore,
    ReceptorDescriptor,
    _SUGAR_TARGETS,
    EPS_CHP_AROMATIC,
    EPS_CHP_LARGE_AROMATIC,
)


EXPECTED_SUGARS = ["Glc", "Gal", "Man", "GlcNAc", "GalNAc", "Fuc", "Fru", "Neu5Ac"]


# -----------------------------------------------------------------------
# Sugar target database
# -----------------------------------------------------------------------

class TestSugarTargets:

    def test_all_present(self):
        for name in EXPECTED_SUGARS:
            assert name in list_sugar_targets()

    def test_count(self):
        assert len(_SUGAR_TARGETS) == 8

    @pytest.mark.parametrize("name", EXPECTED_SUGARS)
    def test_fields_populated(self, name):
        s = get_sugar_target(name)
        assert s.molecular_volume_A3 > 0
        assert s.ring_type in ("pyranose", "furanose")

    def test_glc_all_equatorial(self):
        s = get_sugar_target("Glc")
        assert s.n_eq_OH == 4
        assert s.n_ax_OH == 0
        assert s.n_axial_CH == 3

    def test_gal_has_axial(self):
        s = get_sugar_target("Gal")
        assert s.n_ax_OH == 1

    def test_man_has_cis_12_diol(self):
        s = get_sugar_target("Man")
        assert s.has_cis_12_diol is True

    def test_fru_is_furanose(self):
        s = get_sugar_target("Fru")
        assert s.ring_type == "furanose"
        assert s.has_cis_12_diol is True

    def test_glcnac_has_nac(self):
        s = get_sugar_target("GlcNAc")
        assert s.n_NAc == 1

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_sugar_target("ribose")


# -----------------------------------------------------------------------
# Receptor descriptor
# -----------------------------------------------------------------------

class TestReceptorDescriptor:

    def test_benzene(self):
        desc = compute_receptor_descriptor("c1ccccc1")
        assert desc.valid
        assert desc.n_aromatic_rings == 1
        assert desc.n_boronic_acid == 0

    def test_naphthalene(self):
        desc = compute_receptor_descriptor("c1ccc2ccccc2c1")
        assert desc.valid
        assert desc.n_aromatic_rings >= 2

    def test_urea_detection(self):
        desc = compute_receptor_descriptor("NC(=O)Nc1ccccc1")
        assert desc.valid
        assert desc.n_urea_groups >= 1

    def test_boronic_acid_detection(self):
        desc = compute_receptor_descriptor("OB(O)c1ccccc1")
        assert desc.valid
        assert desc.n_boronic_acid >= 1

    def test_invalid_smiles(self):
        desc = compute_receptor_descriptor("not_a_molecule")
        assert not desc.valid

    def test_molecular_weight(self):
        desc = compute_receptor_descriptor("c1ccc(O)cc1")  # phenol
        assert desc.valid
        assert 90 < desc.molecular_weight < 100


# -----------------------------------------------------------------------
# Scoring physics
# -----------------------------------------------------------------------

class TestScoringPhysics:

    def test_aromatic_receptor_gets_chpi(self):
        """Aromatic receptor should get CH-pi bonus."""
        bs = score_glycan_binder("c1ccccc1", "Glc")
        assert bs.dG_chpi < 0

    def test_no_aromatic_no_chpi(self):
        """Non-aromatic receptor gets no CH-pi."""
        bs = score_glycan_binder("CCCCCC", "Glc")
        assert bs.dG_chpi == 0.0

    def test_more_axial_ch_more_chpi(self):
        """Glc (3 axial CH) should get more CH-pi than Gal (2 axial CH)."""
        aromatic = "c1ccc2cc3ccccc3cc2c1"  # anthracene
        bs_glc = score_glycan_binder(aromatic, "Glc")
        bs_gal = score_glycan_binder(aromatic, "Gal")
        assert bs_glc.dG_chpi < bs_gal.dG_chpi  # more negative = more CH-pi

    def test_hb_donors_help(self):
        """Receptor with HB donors should get HB bonus."""
        # Phenol (1 HBD) vs benzene (0 HBD)
        bs_phenol = score_glycan_binder("c1ccc(O)cc1", "Glc")
        bs_benzene = score_glycan_binder("c1ccccc1", "Glc")
        assert bs_phenol.dG_hb <= bs_benzene.dG_hb  # more negative

    def test_boronic_acid_bonus_for_diol(self):
        """Boronic acid should get bonus for sugars with cis-1,2-diol."""
        pba = "OB(O)c1ccccc1"
        bs_man = score_glycan_binder(pba, "Man")  # has cis-1,2-diol
        bs_glc = score_glycan_binder(pba, "Glc")  # no cis-1,2-diol
        assert bs_man.dG_boronic < bs_glc.dG_boronic  # Man gets bigger bonus

    def test_desolvation_positive(self):
        """Desolvation penalty should be non-negative."""
        bs = score_glycan_binder("NC(=O)Nc1ccc(O)c(O)c1", "Glc")
        assert bs.dG_desolv >= 0

    def test_shape_penalty_non_negative(self):
        bs = score_glycan_binder("c1ccccc1", "Glc")
        assert bs.dG_shape >= 0

    def test_total_is_sum_of_terms(self):
        bs = score_glycan_binder("c1ccc(O)cc1", "Glc")
        expected = (bs.dG_chpi + bs.dG_hb + bs.dG_desolv
                    + bs.dG_shape + bs.dG_boronic + bs.dG_flexibility
                    + bs.dG_axial_clash)
        assert abs(bs.dG_total - expected) < 0.01


# -----------------------------------------------------------------------
# Selectivity
# -----------------------------------------------------------------------

class TestSelectivity:

    def test_selectivity_returns_map(self):
        sel, sel_map = selectivity_score("c1ccccc1", "Glc")
        assert isinstance(sel_map, dict)
        assert "Gal" in sel_map

    def test_glc_selective_receptor(self):
        """Aromatic receptor should prefer Glc over Gal/Man (more axial CH)."""
        aromatic = "c1ccc2cc3ccccc3cc2c1"  # anthracene
        sel, sel_map = selectivity_score(aromatic, "Glc")
        assert sel_map["Gal"] < 0  # negative = Glc preferred
        assert sel_map["Man"] < 0

    def test_boronic_prefers_fru(self):
        """Boronic acid should prefer Fru (best cis-1,2-diol)."""
        pba = "OB(O)c1ccccc1"
        _, sel_map = selectivity_score(pba, "Fru")
        # Fru has cis-1,2-diol: should be preferred over Glc (no cis-1,2)
        assert sel_map["Glc"] < 0  # Fru preferred

    def test_custom_competitor_panel(self):
        sel, sel_map = selectivity_score(
            "c1ccccc1", "Glc", competitors=["Gal", "Man"],
        )
        assert len(sel_map) == 2
        assert "Gal" in sel_map
        assert "Man" in sel_map


# -----------------------------------------------------------------------
# Generator pipeline
# -----------------------------------------------------------------------

class TestGenerator:

    def test_runs_and_returns_result(self):
        result = generate_glycan_binders(target="Glc")
        assert isinstance(result, GlycanBinderResult)
        assert result.target == "Glc"

    def test_generates_candidates(self):
        spec = GlycanBinderSpec(target="Glc", max_candidates=20)
        result = generate_glycan_binders(spec=spec)
        assert result.n_scored > 0
        assert len(result.candidates) > 0

    def test_pareto_front_exists(self):
        spec = GlycanBinderSpec(target="Glc", max_candidates=30)
        result = generate_glycan_binders(spec=spec)
        assert result.n_pareto_front > 0
        assert result.candidates[0].pareto_front == 0

    def test_best_is_pareto_optimal(self):
        spec = GlycanBinderSpec(target="Glc", max_candidates=30)
        result = generate_glycan_binders(spec=spec)
        if result.best:
            assert result.best.pareto_rank == 0

    def test_candidates_have_smiles(self):
        spec = GlycanBinderSpec(target="Glc", max_candidates=20)
        result = generate_glycan_binders(spec=spec)
        for c in result.candidates[:5]:
            assert c.smiles != ""
            assert Chem.MolFromSmiles(c.smiles) is not None

    def test_candidates_have_backbone_and_arms(self):
        spec = GlycanBinderSpec(target="Glc", max_candidates=20)
        result = generate_glycan_binders(spec=spec)
        for c in result.candidates[:5]:
            assert c.backbone_name != ""
            assert len(c.arm_names) > 0

    def test_different_targets_different_results(self):
        r_glc = generate_glycan_binders(target="Glc")
        r_gal = generate_glycan_binders(target="GalNAc")
        if r_glc.best and r_gal.best:
            # Same SMILES but different scores
            bs_glc = r_glc.best.score
            bs_gal = r_gal.best.score
            assert bs_glc.target != bs_gal.target

    def test_summary_string(self):
        spec = GlycanBinderSpec(target="Glc", max_candidates=10)
        result = generate_glycan_binders(spec=spec)
        s = result.summary()
        assert "Glc" in s
        assert "Pareto" in s

    def test_elapsed_time_populated(self):
        spec = GlycanBinderSpec(target="Glc", max_candidates=10)
        result = generate_glycan_binders(spec=spec)
        assert result.elapsed_s > 0


# -----------------------------------------------------------------------
# Design principle validation
# -----------------------------------------------------------------------

class TestDesignPrinciples:

    def test_aromatic_surface_matters_for_glc(self):
        """Larger aromatic surface should score better for Glc (3 axial CH)."""
        benzene_score = score_glycan_binder("c1ccccc1", "Glc")
        anthracene_score = score_glycan_binder("c1ccc2cc3ccccc3cc2c1", "Glc")
        assert anthracene_score.dG_chpi <= benzene_score.dG_chpi

    def test_urea_hbonds_help(self):
        """Urea groups add H-bond donors, improving binding."""
        no_urea = score_glycan_binder("c1ccccc1", "Glc")
        with_urea = score_glycan_binder("NC(=O)Nc1ccccc1", "Glc")
        assert with_urea.dG_hb <= no_urea.dG_hb

    def test_nac_sugars_harder_to_bind(self):
        """GlcNAc should have larger shape penalty than Glc (bulkier)."""
        receptor = "c1ccc2cc3ccccc3cc2c1"  # anthracene (pure CH-pi, no HBD)
        bs_glc = score_glycan_binder(receptor, "Glc")
        bs_glcnac = score_glycan_binder(receptor, "GlcNAc")
        # GlcNAc is 195 A3 vs Glc 160 A3 -> larger shape penalty
        assert bs_glcnac.dG_shape > bs_glc.dG_shape

    def test_furanose_no_chpi(self):
        """Fru (furanose, 0 axial CH) should get no CH-pi."""
        bs = score_glycan_binder("c1ccc2cc3ccccc3cc2c1", "Fru")
        assert bs.dG_chpi == 0.0


# -----------------------------------------------------------------------
# Physical sanity
# -----------------------------------------------------------------------

class TestPhysicalSanity:

    def test_eps_chp_negative(self):
        assert EPS_CHP_AROMATIC < 0
        assert EPS_CHP_LARGE_AROMATIC < 0

    def test_large_aromatic_stronger(self):
        assert EPS_CHP_LARGE_AROMATIC < EPS_CHP_AROMATIC  # more negative

    def test_glc_best_chpi_target(self):
        """Glc should be the best CH-pi target (most axial CH)."""
        receptor = "c1ccc2cc3ccccc3cc2c1"
        scores = {}
        for name in EXPECTED_SUGARS:
            bs = score_glycan_binder(receptor, name)
            scores[name] = bs.dG_chpi
        best = min(scores, key=scores.get)
        assert best == "Glc" or best == "GlcNAc"  # both have 3 axial CH
