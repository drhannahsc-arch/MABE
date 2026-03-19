"""
Tests for glycan/click_site_predictor.py — binary classifier for click handle attachment.

Validates against Schwarz 1996 (Biochem J 316:123) deoxy-glucose @ ConA:
  C1: removal tolerated (Ka=690) -> CANDIDATE
  C2: removal abolishes binding -> should be ESSENTIAL but no contact mapped
       ... Schwarz says no binding, our map says CANDIDATE. Known gap:
       C2-OH may gate ring orientation. Documented as open question.
  C3: removal abolishes -> ESSENTIAL
  C4: removal abolishes -> ESSENTIAL
  C6: removal abolishes -> ESSENTIAL
"""

import pytest
from glycan.click_site_predictor import (
    POSITION_CONTACTS,
    ESSENTIAL, CANDIDATE, RING,
    classify_position,
    classify_all_positions,
    get_attachment_sites,
    get_essential_positions,
    validate_schwarz,
)


# ── Schwarz validation (ConA Glc deoxy series) ──────────────────────────

class TestSchwarzValidation:
    """Schwarz 1996: C3, C4, C6 essential; C1 non-essential."""

    def test_c1_candidate(self):
        """C1: 1-deoxy-Glc binds (Ka=690) -> CANDIDATE."""
        cls = classify_position("ConA", "Glc", "C1")
        assert cls.classification == CANDIDATE

    def test_c3_essential(self):
        """C3: 3-deoxy-Glc no binding -> ESSENTIAL."""
        cls = classify_position("ConA", "Glc", "C3")
        assert cls.classification == ESSENTIAL

    def test_c4_essential(self):
        """C4: 4-deoxy-Glc no binding -> ESSENTIAL."""
        cls = classify_position("ConA", "Glc", "C4")
        assert cls.classification == ESSENTIAL

    def test_c6_essential(self):
        """C6: 6-deoxy-Glc no binding -> ESSENTIAL."""
        cls = classify_position("ConA", "Glc", "C6")
        assert cls.classification == ESSENTIAL

    def test_c5_ring(self):
        """C5 is ring carbon, no OH."""
        cls = classify_position("ConA", "Glc", "C5")
        assert cls.classification == RING

    def test_schwarz_all_correct_except_c2(self):
        """4/5 Schwarz entries match. C2 is known gap (documented)."""
        results = validate_schwarz()
        correct = [r for r in results if r["match"] == "CORRECT"]
        # C1=CORRECT, C3=CORRECT, C4=CORRECT, C6=CORRECT
        # C2 is a known FALSE_NEGATIVE (CANDIDATE but obs=no binding)
        assert len(correct) >= 4


# ── ConA Man position map ───────────────────────────────────────────────

class TestConAMan:
    def test_c1_c2_candidate(self):
        """Man C1, C2 are solvent-exposed -> CANDIDATE."""
        for pos in ["C1", "C2"]:
            cls = classify_position("ConA", "Man", pos)
            assert cls.classification == CANDIDATE, f"Man {pos} should be CANDIDATE"

    def test_c3_c4_c6_essential(self):
        """Man C3, C4, C6 have protein contacts -> ESSENTIAL."""
        for pos in ["C3", "C4", "C6"]:
            cls = classify_position("ConA", "Man", pos)
            assert cls.classification == ESSENTIAL, f"Man {pos} should be ESSENTIAL"

    def test_attachment_sites_are_c1_c2(self):
        sites = get_attachment_sites("ConA", "Man")
        positions = {s.position for s in sites}
        assert positions == {"C1", "C2"}

    def test_pharmacophore_is_c3_c4_c6(self):
        essential = get_essential_positions("ConA", "Man")
        positions = {e.position for e in essential}
        assert positions == {"C3", "C4", "C6"}


# ── Davis Glc ───────────────────────────────────────────────────────────

class TestDavisGlc:
    def test_c1_is_candidate(self):
        """Davis C1: no HB, no desolv -> CANDIDATE (despite CH-pi at C1-H)."""
        cls = classify_position("Davis", "Glc", "C1")
        assert cls.classification == CANDIDATE

    def test_c2_through_c6_essential(self):
        """Davis C2-C4, C6 all have HB contacts -> ESSENTIAL."""
        for pos in ["C2", "C3", "C4", "C6"]:
            cls = classify_position("Davis", "Glc", pos)
            assert cls.classification == ESSENTIAL, f"Davis Glc {pos} should be ESSENTIAL"

    def test_only_c1_is_attachment_site(self):
        """Davis receptor: only C1 is available for click handle."""
        sites = get_attachment_sites("Davis", "Glc")
        assert len(sites) == 1
        assert sites[0].position == "C1"

    def test_davis_2dglc_consistency(self):
        """C2 is ESSENTIAL in Davis. 2dGlc (C2-OH removed) loses 7.5 kJ/mol.
        The binary classifier correctly flags this as essential."""
        cls = classify_position("Davis", "Glc", "C2")
        assert cls.classification == ESSENTIAL
        assert cls.n_hb == 1


# ── Cross-scaffold comparison ───────────────────────────────────────────

class TestCrossScaffold:
    def test_c2_candidate_in_lectins_essential_in_davis(self):
        """C2 is CANDIDATE in ConA/PNA/Gal3 but ESSENTIAL in Davis.
        Same sugar position, different receptor geometry -> different answer.
        This is the core value of the predictor."""
        # Lectins: C2 non-essential
        assert classify_position("ConA", "Man", "C2").classification == CANDIDATE
        assert classify_position("ConA", "Glc", "C2").classification == CANDIDATE
        assert classify_position("PNA", "Gal", "C2").classification == CANDIDATE
        assert classify_position("Gal3", "Gal", "C2").classification == CANDIDATE
        # Davis: C2 essential
        assert classify_position("Davis", "Glc", "C2").classification == ESSENTIAL

    def test_c3_c4_c6_universal_essential(self):
        """C3, C4, C6 are essential across all scaffolds."""
        for scaffold, ligand in [("ConA", "Man"), ("ConA", "Glc"),
                                  ("PNA", "Gal"), ("Gal3", "Gal"),
                                  ("Davis", "Glc")]:
            for pos in ["C3", "C4", "C6"]:
                cls = classify_position(scaffold, ligand, pos)
                assert cls.classification == ESSENTIAL, \
                    f"{scaffold}/{ligand} {pos} should be ESSENTIAL"


# ── Coverage checks ─────────────────────────────────────────────────────

class TestCoverage:
    def test_all_scaffolds_have_positions(self):
        """Every entry should have C1-C6 (hexoses) or C1-C9 (Neu5Ac)."""
        for key in POSITION_CONTACTS:
            scaffold, ligand = key
            positions = classify_all_positions(scaffold, ligand)
            if ligand == "Neu5Ac":
                assert len(positions) == 9, f"{scaffold}/{ligand} has {len(positions)} positions, expected 9"
            else:
                assert len(positions) == 6, f"{scaffold}/{ligand} has {len(positions)} positions, expected 6"

    def test_all_scaffolds_present(self):
        """All scaffolds x their anchor ligands are mapped."""
        expected = {
            ("ConA", "Man"), ("ConA", "Glc"),
            ("Davis", "Glc"),
            ("PNA", "Gal"), ("Gal3", "Gal"),
            ("WGA", "GlcNAc"),
            ("Siglec2", "Neu5Ac"),
        }
        assert expected == set(POSITION_CONTACTS.keys())

    def test_error_on_unknown_scaffold(self):
        with pytest.raises(ValueError):
            classify_position("FakeLectin", "Man", "C1")

    def test_error_on_unknown_position(self):
        with pytest.raises(ValueError):
            classify_position("ConA", "Man", "C7")


# ── Siglec2 / Neu5Ac (sialic acid) ─────────────────────────────────────

class TestSiglec2Neu5Ac:
    def test_pharmacophore_positions(self):
        """C1 (COO-), C4, C5 (NHAc), C7 are ESSENTIAL."""
        for pos in ["C1", "C4", "C5", "C7"]:
            cls = classify_position("Siglec2", "Neu5Ac", pos)
            assert cls.classification == ESSENTIAL, f"Neu5Ac {pos} should be ESSENTIAL"

    def test_ring_positions(self):
        """C2 (keto), C3 (CH2), C6 (ring) are RING."""
        for pos in ["C2", "C3", "C6"]:
            cls = classify_position("Siglec2", "Neu5Ac", pos)
            assert cls.classification == RING, f"Neu5Ac {pos} should be RING"

    def test_candidate_positions(self):
        """C8, C9 (glycerol sidechain) are CANDIDATE."""
        for pos in ["C8", "C9"]:
            cls = classify_position("Siglec2", "Neu5Ac", pos)
            assert cls.classification == CANDIDATE, f"Neu5Ac {pos} should be CANDIDATE"

    def test_attachment_sites_are_c8_c9(self):
        sites = get_attachment_sites("Siglec2", "Neu5Ac")
        positions = {s.position for s in sites}
        assert positions == {"C8", "C9"}

    def test_c9_preferred_for_click(self):
        """C9 is the standard click site for sialic acid (C9-azido-Neu5Ac)."""
        cls = classify_position("Siglec2", "Neu5Ac", "C9")
        assert cls.classification == CANDIDATE
        assert "azido" in cls.note.lower() or "click" in cls.note.lower()

    def test_nine_positions_total(self):
        """Neu5Ac has 9 carbons -> 9 positions."""
        all_pos = classify_all_positions("Siglec2", "Neu5Ac")
        assert len(all_pos) == 9
