"""
Tests for glycan/daedalus_bridge.py -- pyDAEDALUS integration.

Uses pre-generated CSV fixtures from pyDAEDALUS runs on
tetrahedron, octahedron, and icosahedron PLY files.
"""

import pytest
import os

from glycan.daedalus_bridge import (
    parse_staples_csv,
    assign_modifications,
    design_from_csv,
    DAEDALUSDesign,
    ModifiedStaple,
    ParsedStaple,
    list_standard_cages,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    return os.path.join(FIXTURES, name)


# ── CSV parsing ─────────────────────────────────────────────────────────

class TestCSVParsing:
    def test_parse_tetrahedron(self):
        staples, scaffold, scaf_len = parse_staples_csv(_fixture("tetrahedron_staples.csv"))
        assert len(staples) == 10
        assert scaf_len > 0
        assert len(scaffold) == scaf_len

    def test_parse_octahedron(self):
        staples, scaffold, scaf_len = parse_staples_csv(_fixture("octahedron_staples.csv"))
        assert len(staples) == 24

    def test_parse_icosahedron(self):
        staples, scaffold, scaf_len = parse_staples_csv(_fixture("icosahedron_staples.csv"))
        assert len(staples) == 54

    def test_vertex_edge_classification(self):
        staples, _, _ = parse_staples_csv(_fixture("tetrahedron_staples.csv"))
        v = [s for s in staples if s.is_vertex]
        e = [s for s in staples if s.is_edge]
        assert len(v) == 4
        assert len(e) == 6

    def test_octahedron_vertex_edge_counts(self):
        staples, _, _ = parse_staples_csv(_fixture("octahedron_staples.csv"))
        v = [s for s in staples if s.is_vertex]
        e = [s for s in staples if s.is_edge]
        assert len(v) == 12
        assert len(e) == 12

    def test_sequences_are_dna(self):
        staples, _, _ = parse_staples_csv(_fixture("octahedron_staples.csv"))
        for s in staples:
            assert all(c in "ACGT" for c in s.sequence), f"{s.name} has non-DNA chars"

    def test_vertex_staples_longer_than_edge(self):
        staples, _, _ = parse_staples_csv(_fixture("octahedron_staples.csv"))
        v_lens = [s.length for s in staples if s.is_vertex]
        e_lens = [s.length for s in staples if s.is_edge]
        assert min(v_lens) > max(e_lens), "Vertex staples should be longer than edge staples"


# ── Modification assignment ─────────────────────────────────────────────

class TestModificationAssignment:
    def test_all_staples_assigned(self):
        staples, _, _ = parse_staples_csv(_fixture("octahedron_staples.csv"))
        modified = assign_modifications(staples, sugar="Man")
        assert len(modified) == len(staples)

    def test_fraction_allocation(self):
        staples, _, _ = parse_staples_csv(_fixture("octahedron_staples.csv"))
        modified = assign_modifications(staples, sugar_fraction=0.50,
                                        magnetic_fraction=0.30,
                                        passivation_fraction=0.15,
                                        reporter_fraction=0.05)
        roles = {}
        for ms in modified:
            roles[ms.role] = roles.get(ms.role, 0) + 1
        assert roles.get("sugar", 0) == 12   # 50% of 24
        assert roles.get("magnetic", 0) == 7  # 30% of 24
        assert roles.get("reporter", 0) == 1  # 5% of 24

    def test_vertex_staples_get_sugar_first(self):
        """With prefer_vertex_for_sugar=True, vertex staples should be sugar."""
        staples, _, _ = parse_staples_csv(_fixture("octahedron_staples.csv"))
        modified = assign_modifications(staples, sugar_fraction=0.50,
                                        prefer_vertex_for_sugar=True)
        sugar_mods = [ms for ms in modified if ms.role == "sugar"]
        # All or most sugar mods should be on vertex staples
        v_sugar = sum(1 for ms in sugar_mods if ms.staple_type == "V")
        assert v_sugar >= len(sugar_mods) * 0.5  # at least half

    def test_spaac_uses_dbco(self):
        staples, _, _ = parse_staples_csv(_fixture("tetrahedron_staples.csv"))
        modified = assign_modifications(staples, click_chemistry="SPAAC")
        sugar_mods = [ms for ms in modified if ms.role == "sugar"]
        for ms in sugar_mods:
            assert "DBCO" in ms.modification_3prime

    def test_cuaac_uses_alkyne(self):
        staples, _, _ = parse_staples_csv(_fixture("tetrahedron_staples.csv"))
        modified = assign_modifications(staples, click_chemistry="CuAAC")
        sugar_mods = [ms for ms in modified if ms.role == "sugar"]
        for ms in sugar_mods:
            assert "alkyne" in ms.modification_3prime

    def test_cuaac_cheaper_than_spaac(self):
        staples, _, _ = parse_staples_csv(_fixture("octahedron_staples.csv"))
        spaac = assign_modifications(staples, click_chemistry="SPAAC")
        cuaac = assign_modifications(staples, click_chemistry="CuAAC")
        spaac_cost = sum(ms.cost_usd for ms in spaac)
        cuaac_cost = sum(ms.cost_usd for ms in cuaac)
        assert cuaac_cost < spaac_cost


# ── IDT order string ────────────────────────────────────────────────────

class TestIDTFormat:
    def test_sugar_staple_has_extension(self):
        staples, _, _ = parse_staples_csv(_fixture("tetrahedron_staples.csv"))
        modified = assign_modifications(staples, click_chemistry="SPAAC")
        sugar = [ms for ms in modified if ms.role == "sugar"][0]
        idt = sugar.idt_order_string
        assert "TTTTTTTTTTTTTTTTTTTT" in idt  # poly-T extension
        assert "DBCO" in idt

    def test_magnetic_staple_has_biotin(self):
        staples, _, _ = parse_staples_csv(_fixture("tetrahedron_staples.csv"))
        modified = assign_modifications(staples)
        mag = [ms for ms in modified if ms.role == "magnetic"][0]
        idt = mag.idt_order_string
        assert "Biotin" in idt

    def test_total_length_includes_extension(self):
        staples, _, _ = parse_staples_csv(_fixture("tetrahedron_staples.csv"))
        modified = assign_modifications(staples)
        sugar = [ms for ms in modified if ms.role == "sugar"][0]
        assert sugar.total_length == sugar.length + 20  # 20-nt poly-T extension


# ── Full design from CSV ────────────────────────────────────────────────

class TestDesignFromCSV:
    def test_tetrahedron_design(self):
        design = design_from_csv(_fixture("tetrahedron_staples.csv"), sugar="Man")
        assert isinstance(design, DAEDALUSDesign)
        assert design.n_staples == 10
        assert design.n_vertex_staples == 4
        assert design.n_edge_staples == 6

    def test_octahedron_design(self):
        design = design_from_csv(_fixture("octahedron_staples.csv"), sugar="Gal")
        assert design.n_staples == 24

    def test_icosahedron_design(self):
        design = design_from_csv(_fixture("icosahedron_staples.csv"), sugar="Neu5Ac")
        assert design.n_staples == 54
        assert design.scaffold_length > 0

    def test_summary_format(self):
        design = design_from_csv(_fixture("octahedron_staples.csv"))
        assert isinstance(design.summary, str)
        assert "DAEDALUS" in design.summary
        assert "24" in design.summary  # n_staples

    def test_order_sheet_has_all_entries(self):
        design = design_from_csv(_fixture("octahedron_staples.csv"))
        sheet = design.idt_order_sheet
        # n_staples + 1 scaffold
        assert len(sheet) == design.n_staples + 1
        assert sheet[-1]["role"] == "scaffold"

    def test_cost_positive(self):
        design = design_from_csv(_fixture("octahedron_staples.csv"))
        assert design.total_cost_usd > 0

    def test_larger_cage_more_expensive(self):
        tet = design_from_csv(_fixture("tetrahedron_staples.csv"))
        ico = design_from_csv(_fixture("icosahedron_staples.csv"))
        assert ico.total_cost_usd > tet.total_cost_usd

    def test_all_three_cages(self):
        for name in ["tetrahedron_staples.csv", "octahedron_staples.csv", "icosahedron_staples.csv"]:
            design = design_from_csv(_fixture(name))
            assert design.n_sugar > 0
            assert design.n_magnetic > 0


# ── Standard cages list ─────────────────────────────────────────────────

class TestStandardCages:
    def test_list_has_common_polyhedra(self):
        cages = list_standard_cages()
        assert "tetrahedron" in cages
        assert "octahedron" in cages
        assert "icosahedron" in cages
        assert "cube" in cages
