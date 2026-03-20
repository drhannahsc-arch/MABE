"""
glycan/tests/test_glycan_expansion.py -- Tests for expanded prediction set.

Validates:
  - DGL scaffold: anchor, selectivity (Man > Glc), present in registry
  - ConA deoxy-glucose series: non-binders have no obs_dG, 1-deoxy quantitative
  - ConA fluoro-glucose series: 1-F-Glc and 2-F-Glc quantitative
  - ConA maltose/isomaltose: present, reasonable predictions
  - Cross-scaffold consistency: DGL Man > Glc matches ConA Man > Glc
  - Non-binder predictions are weaker than parent Glc
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from glycan.scorer import GlycanScorer
from glycan.contact_maps import SCAFFOLD_CONTACTS


# -----------------------------------------------------------------------
# DGL scaffold
# -----------------------------------------------------------------------

class TestDGLScaffold:

    def test_dgl_in_registry(self):
        assert "DGL" in SCAFFOLD_CONTACTS

    def test_dgl_has_anchor(self):
        contacts = SCAFFOLD_CONTACTS["DGL"]
        anchors = [k for k, v in contacts.items() if v.get("anchor")]
        assert len(anchors) == 1
        assert anchors[0] == "Man"

    def test_dgl_man_anchor_exact(self):
        s = GlycanScorer()
        p = s.score("DGL", "Man")
        assert p.dG_pred == p.dG_obs  # anchor: pred == obs
        assert p.dG_obs == -20.3

    def test_dgl_man_gt_glc(self):
        """DGL shares ConA selectivity: Man > Glc."""
        s = GlycanScorer()
        man = s.score("DGL", "Man").dG_pred
        glc = s.score("DGL", "Glc").dG_pred
        assert man < glc

    def test_dgl_glc_residual_small(self):
        s = GlycanScorer()
        p = s.score("DGL", "Glc")
        assert abs(p.residual) < 1.5

    def test_dgl_has_four_ligands(self):
        assert len(SCAFFOLD_CONTACTS["DGL"]) == 4

    def test_dgl_triman_present(self):
        s = GlycanScorer()
        p = s.score("DGL", "triMan")
        assert p.dG_obs == -33.5  # Chervenak 1995

    def test_dgl_triman_outlier_noted(self):
        """DGL triMan is a known outlier (reversed preference vs ConA)."""
        s = GlycanScorer()
        p = s.score("DGL", "triMan")
        # Large residual expected — DGL binds triMan 2.8x tighter than ConA
        # but our contact map is based on ConA geometry
        assert abs(p.residual) > 2.0


# -----------------------------------------------------------------------
# ConA deoxy-glucose series
# -----------------------------------------------------------------------

class TestConADeoxySeries:

    def test_1deoxy_quantitative(self):
        s = GlycanScorer()
        p = s.score("ConA", "1-deoxy-Glc")
        assert p.dG_obs == -16.2
        assert abs(p.residual) < 1.5

    @pytest.mark.parametrize("ligand", [
        "2-deoxy-Glc", "3-deoxy-Glc", "4-deoxy-Glc", "6-deoxy-Glc",
    ])
    def test_nonbinder_no_obs(self, ligand):
        """Non-binders should have obs_dG = None."""
        s = GlycanScorer()
        p = s.score("ConA", ligand)
        assert p.dG_obs is None
        assert p.residual is None

    @pytest.mark.parametrize("ligand", [
        "2-deoxy-Glc", "3-deoxy-Glc", "4-deoxy-Glc", "6-deoxy-Glc",
    ])
    def test_nonbinder_fewer_contacts_than_parent(self, ligand):
        """Non-binders should have fewer HBs and CH-pi than Glc parent.
        
        Note: the anchor-based scorer correctly predicts contact energetics
        but non-binders don't enter the pocket at all. The useful test is
        binary: critical contacts (C3-Asp208, C4-Arg228, C6-Asn14) are lost.
        """
        contacts = SCAFFOLD_CONTACTS["ConA"]
        parent_hb = contacts["Glc"]["n_HB"]
        parent_chp = contacts["Glc"]["n_CHP"]
        nb_hb = contacts[ligand]["n_HB"]
        nb_chp = contacts[ligand]["n_CHP"]
        total_parent = parent_hb + parent_chp
        total_nb = nb_hb + nb_chp
        assert total_nb < total_parent, (
            f"{ligand}: contacts={total_nb} should be < Glc={total_parent}"
        )

    def test_1deoxy_weaker_than_glc(self):
        """1-deoxy-Glc binds but weaker than Glc."""
        s = GlycanScorer()
        glc = s.score("ConA", "Glc").dG_pred
        d1 = s.score("ConA", "1-deoxy-Glc").dG_pred
        assert d1 > glc


# -----------------------------------------------------------------------
# ConA fluoro-glucose series
# -----------------------------------------------------------------------

class TestConAFluoroSeries:

    def test_1f_glc_quantitative(self):
        s = GlycanScorer()
        p = s.score("ConA", "1-F-Glc")
        assert p.dG_obs == -20.4
        assert abs(p.residual) < 1.5

    def test_2f_glc_quantitative(self):
        s = GlycanScorer()
        p = s.score("ConA", "2-F-Glc")
        assert p.dG_obs == -18.7
        assert abs(p.residual) < 2.0

    def test_1f_stronger_than_1deoxy(self):
        """1-F-Glc retains positioning (Ka=3750 vs 690 for 1-deoxy)."""
        s = GlycanScorer()
        f1 = s.score("ConA", "1-F-Glc").dG_pred
        d1 = s.score("ConA", "1-deoxy-Glc").dG_pred
        assert f1 < d1


# -----------------------------------------------------------------------
# ConA maltose / isomaltose
# -----------------------------------------------------------------------

class TestConADisaccharides:

    def test_maltose_present(self):
        s = GlycanScorer()
        p = s.score("ConA", "maltose")
        assert p.dG_obs == -18.3

    def test_isomaltose_present(self):
        s = GlycanScorer()
        p = s.score("ConA", "isomaltose")
        assert p.dG_obs == -18.4

    def test_maltose_residual_reasonable(self):
        s = GlycanScorer()
        p = s.score("ConA", "maltose")
        assert abs(p.residual) < 2.0

    def test_isomaltose_residual_reasonable(self):
        s = GlycanScorer()
        p = s.score("ConA", "isomaltose")
        assert abs(p.residual) < 2.0

    def test_maltose_weaker_than_diman(self):
        """Maltose (Glc-Glc) should bind weaker than diMan (Man-Man)."""
        s = GlycanScorer()
        malt = s.score("ConA", "maltose").dG_pred
        diman = s.score("ConA", "1->3 diMan").dG_pred
        assert malt > diman

    def test_maltose_has_conformational_entropy(self):
        """Maltose is a disaccharide — should have G2 penalty."""
        s = GlycanScorer()
        p = s.score("ConA", "maltose")
        assert p.dG_conf > 0


# -----------------------------------------------------------------------
# Cross-scaffold consistency
# -----------------------------------------------------------------------

class TestCrossScaffold:

    def test_dgl_cona_same_selectivity(self):
        """DGL and ConA have same monosaccharide selectivity."""
        s = GlycanScorer()
        cona_ratio = s.score("ConA", "Man").dG_pred - s.score("ConA", "Glc").dG_pred
        dgl_ratio = s.score("DGL", "Man").dG_pred - s.score("DGL", "Glc").dG_pred
        # Same direction (Man tighter than Glc in both)
        assert cona_ratio < 0
        assert dgl_ratio < 0

    def test_seven_scaffolds_total(self):
        """Should now have 7 scaffolds in registry."""
        assert len(SCAFFOLD_CONTACTS) == 7

    def test_total_predictions_35(self):
        """Total prediction count should be 35."""
        s = GlycanScorer()
        preds = s.score_all()
        assert len(preds) == 35
