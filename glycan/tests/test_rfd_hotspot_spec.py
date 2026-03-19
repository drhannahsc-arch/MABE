"""
Tests for glycan/rfd_hotspot_spec.py -- RFdiffusion hotspot generation.
"""

import pytest
from glycan.rfd_hotspot_spec import (
    generate_hotspot_spec,
    list_available_pharmacophores,
    HotspotSpec,
    HotspotResidue,
    _KNOWN_PDB_RESIDUES,
)


# ── Basic generation ────────────────────────────────────────────────────

class TestBasicGeneration:
    def test_cona_man_returns_spec(self):
        spec = generate_hotspot_spec("ConA", "Man")
        assert isinstance(spec, HotspotSpec)

    def test_summary_is_string(self):
        spec = generate_hotspot_spec("ConA", "Man")
        assert isinstance(spec.summary, str)
        assert "ConA" in spec.summary

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            generate_hotspot_spec("FakeLectin", "Man")

    def test_all_pharmacophores_work(self):
        for scaffold, ligand in list_available_pharmacophores():
            spec = generate_hotspot_spec(scaffold, ligand)
            assert isinstance(spec, HotspotSpec)


# ── ConA/Man pharmacophore ──────────────────────────────────────────────

class TestConAMan:
    def test_has_hb_and_chp(self):
        spec = generate_hotspot_spec("ConA", "Man")
        assert spec.n_hb_contacts >= 3  # C3, C4, C6
        assert spec.n_chp_contacts >= 1  # Tyr12

    def test_c3_asp(self):
        """C3 should map to ASP (Asp208 in 5CNA)."""
        spec = generate_hotspot_spec("ConA", "Man")
        c3 = [hs for hs in spec.hotspots if hs.position_label == "C3"]
        assert len(c3) == 1
        assert c3[0].residue_type_3 == "ASP"

    def test_c6_asn(self):
        """C6 should map to ASN (Asn14 in 5CNA)."""
        spec = generate_hotspot_spec("ConA", "Man")
        c6 = [hs for hs in spec.hotspots if hs.position_label == "C6"]
        assert len(c6) == 1
        assert c6[0].residue_type_3 == "ASN"

    def test_chp_is_tyr(self):
        """CH-pi should be TYR (Tyr12 in 5CNA)."""
        spec = generate_hotspot_spec("ConA", "Man")
        chp = [hs for hs in spec.hotspots if "chp" in hs.position_label]
        assert len(chp) >= 1
        assert chp[0].residue_type_3 == "TYR"

    def test_pdb_validation_present(self):
        """ConA/Man should have PDB-validated hotspots."""
        spec = generate_hotspot_spec("ConA", "Man")
        assert spec.n_validated >= 2  # at least C3-ASP, C6-ASN should match


# ── Gal3/Gal pharmacophore ──────────────────────────────────────────────

class TestGal3Gal:
    def test_has_trp_chp(self):
        """Gal3 uses Trp181 for CH-pi, not Tyr."""
        spec = generate_hotspot_spec("Gal3", "Gal")
        chp = [hs for hs in spec.hotspots if "chp" in hs.position_label]
        assert len(chp) >= 1
        assert chp[0].residue_type_3 == "TRP"

    def test_c4_axial_maps_to_arg(self):
        """Gal C4-OH is axial -> should map to ARG (Arg144 in 1KJL)."""
        spec = generate_hotspot_spec("Gal3", "Gal")
        c4 = [hs for hs in spec.hotspots if hs.position_label == "C4"]
        assert len(c4) == 1
        assert c4[0].residue_type_3 == "ARG"


# ── Siglec2/Neu5Ac ─────────────────────────────────────────────────────

class TestSiglec2Neu5Ac:
    def test_salt_bridge_at_c1(self):
        """C1 carboxylate -> salt bridge to Arg120."""
        spec = generate_hotspot_spec("Siglec2", "Neu5Ac")
        c1 = [hs for hs in spec.hotspots if hs.position_label == "C1"]
        assert len(c1) == 1
        assert c1[0].interaction_type == "salt_bridge"
        assert c1[0].residue_type_3 == "ARG"

    def test_nhac_at_c5(self):
        """C5 NHAc -> amide contact (ASN/GLN)."""
        spec = generate_hotspot_spec("Siglec2", "Neu5Ac")
        c5 = [hs for hs in spec.hotspots if hs.position_label == "C5"]
        assert len(c5) == 1
        assert c5[0].residue_type_3 in ("ASN", "GLN")


# ── Coordinate geometry ─────────────────────────────────────────────────

class TestCoordinates:
    def test_hb_residues_at_correct_distance(self):
        """HB hotspot residues should be ~2.8 A from sugar OH."""
        spec = generate_hotspot_spec("ConA", "Man")
        for hs in spec.hotspots:
            if hs.interaction_type in ("HB_acceptor", "HB_donor", "salt_bridge"):
                assert abs(hs.distance_to_sugar_A - 2.8) < 0.1

    def test_chp_residues_above_ring(self):
        """CH-pi hotspots should be above the ring plane (positive Z)."""
        spec = generate_hotspot_spec("ConA", "Man")
        chp = [hs for hs in spec.hotspots if hs.interaction_type == "CH_pi"]
        for hs in chp:
            assert hs.coord_z > 3.0  # well above ring plane

    def test_coords_not_all_zero(self):
        """No hotspot should be at origin."""
        for scaffold, ligand in [("ConA", "Man"), ("Gal3", "Gal"), ("Siglec2", "Neu5Ac")]:
            spec = generate_hotspot_spec(scaffold, ligand)
            for hs in spec.hotspots:
                assert not (hs.coord_x == 0 and hs.coord_y == 0 and hs.coord_z == 0)


# ── RFdiffusion output format ──────────────────────────────────────────

class TestOutputFormat:
    def test_contig_string_format(self):
        spec = generate_hotspot_spec("ConA", "Man")
        contig = spec.contig_string
        assert contig.startswith("[")
        assert "hotspots:" in contig

    def test_pdb_lines_format(self):
        spec = generate_hotspot_spec("ConA", "Man")
        lines = spec.hotspot_pdb_lines
        assert len(lines) == len(spec.hotspots)
        for line in lines:
            assert line.startswith("ATOM")
            assert len(line) >= 54  # minimum PDB line length

    def test_pdb_lines_have_correct_residues(self):
        spec = generate_hotspot_spec("ConA", "Man")
        lines = spec.hotspot_pdb_lines
        for i, (line, hs) in enumerate(zip(lines, spec.hotspots)):
            assert hs.residue_type_3 in line


# ── PDB validation coverage ────────────────────────────────────────────

class TestValidation:
    def test_cona_man_mostly_validated(self):
        """ConA/Man should have high validation rate."""
        spec = generate_hotspot_spec("ConA", "Man")
        assert spec.n_validated >= 3  # ASP208, ASN14, TYR12

    def test_validation_notes_propagate(self):
        spec = generate_hotspot_spec("ConA", "Man")
        validated = [hs for hs in spec.hotspots if hs.pdb_validation and "MATCH" in hs.pdb_validation]
        for hs in validated:
            assert hs.confidence == "high"
            assert "5CNA" in hs.pdb_validation
