"""
tests/test_design_engine_expansion.py

Tests for Phase 2 design engine expansion:
1. donor_enumerator.py — HSAB pools, new archetypes, CHELATE_PAIRS
2. design_engine.py — new METAL_ALIASES, MATRIX_PRESETS, design_mode
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────────────────
# 1. Donor enumerator expansion
# ──────────────────────────────────────────────────────────────────────────

class TestHSABPools:

    def test_soft_pool_contains_selenolate(self):
        from core.donor_enumerator import SOFT_METAL_SUBTYPES
        assert "Se_selenolate" in SOFT_METAL_SUBTYPES

    def test_soft_pool_contains_phosphine(self):
        from core.donor_enumerator import SOFT_METAL_SUBTYPES
        assert "P_phosphine" in SOFT_METAL_SUBTYPES

    def test_hard_pool_contains_fluoride(self):
        from core.donor_enumerator import HARD_METAL_SUBTYPES
        assert "F_fluoride" in HARD_METAL_SUBTYPES

    def test_hard_pool_contains_carboxylate(self):
        from core.donor_enumerator import HARD_METAL_SUBTYPES
        assert "O_carboxylate" in HARD_METAL_SUBTYPES

    def test_soft_pool_excludes_fluoride(self):
        from core.donor_enumerator import SOFT_METAL_SUBTYPES
        assert "F_fluoride" not in SOFT_METAL_SUBTYPES

    def test_hard_pool_excludes_selenolate(self):
        from core.donor_enumerator import HARD_METAL_SUBTYPES
        assert "Se_selenolate" not in HARD_METAL_SUBTYPES

    def test_get_subtypes_soft_metal(self):
        """Hg2+ (softness=0.85) should get soft pool."""
        from core.donor_enumerator import _get_subtypes_for_metal, SOFT_METAL_SUBTYPES
        pool = _get_subtypes_for_metal("Hg2+")
        assert "Se_selenolate" in pool
        assert "P_phosphine" in pool

    def test_get_subtypes_hard_metal(self):
        """Al3+ (softness=0.05) should get hard pool."""
        from core.donor_enumerator import _get_subtypes_for_metal, HARD_METAL_SUBTYPES
        pool = _get_subtypes_for_metal("Al3+")
        assert "F_fluoride" in pool
        assert "O_carboxylate" in pool

    def test_allowed_subtypes_override(self):
        """allowed_subtypes should bypass HSAB inference."""
        from core.donor_enumerator import _get_subtypes_for_metal
        custom = ["N_amine", "O_carboxylate"]
        pool = _get_subtypes_for_metal("Hg2+", allowed_subtypes=custom)
        assert pool == custom


class TestNewArchetypes:

    def _get_archetypes_by_name(self, substring):
        from core.donor_enumerator import ARCHETYPES
        return [a for a in ARCHETYPES if substring.lower() in a.archetype.lower()]

    def test_selenocysteine_archetype_present(self):
        hits = self._get_archetypes_by_name("selenocysteine")
        assert len(hits) >= 1, "Selenocysteine-type archetype missing"
        hit = hits[0]
        assert "Se_selenolate" in hit.donor_subtypes
        assert "N_amine" in hit.donor_subtypes

    def test_bis_selenolate_archetype_present(self):
        hits = self._get_archetypes_by_name("bis(selenolate)")
        assert len(hits) >= 1

    def test_selenacrown_archetype_present(self):
        hits = self._get_archetypes_by_name("selenacrown")
        assert len(hits) >= 1
        hit = hits[0]
        assert hit.is_macrocyclic

    def test_phosphine_binder_archetype_present(self):
        hits = self._get_archetypes_by_name("bisphosphine")
        assert len(hits) >= 1

    def test_hexacyanometallate_archetype_present(self):
        hits = self._get_archetypes_by_name("hexacyano")
        assert len(hits) >= 1
        hit = hits[0]
        assert all(s == "C_cyanide" for s in hit.donor_subtypes)

    def test_oxime_archetype_present(self):
        hits = self._get_archetypes_by_name("dimethylglyoxime")
        assert len(hits) >= 1

    def test_fluoride_archetype_present(self):
        hits = self._get_archetypes_by_name("fluoride")
        assert len(hits) >= 1

    def test_thiacrown_archetype_present(self):
        hits = self._get_archetypes_by_name("[12]anes")
        assert len(hits) >= 1


class TestNewChelatePairs:

    def test_se_n_chelate_pair(self):
        from core.donor_enumerator import CHELATE_PAIRS
        assert ("Se_selenolate", "N_amine") in CHELATE_PAIRS

    def test_p_s_chelate_pair(self):
        from core.donor_enumerator import CHELATE_PAIRS
        assert ("P_phosphine", "S_thiolate") in CHELATE_PAIRS

    def test_p_se_chelate_pair(self):
        from core.donor_enumerator import CHELATE_PAIRS
        assert ("P_phosphine", "Se_selenolate") in CHELATE_PAIRS

    def test_as_as_chelate_pair(self):
        from core.donor_enumerator import CHELATE_PAIRS
        assert ("As_arsine", "As_arsine") in CHELATE_PAIRS

    def test_oxime_oxime_chelate_pair(self):
        from core.donor_enumerator import CHELATE_PAIRS
        assert ("N_oxime", "N_oxime") in CHELATE_PAIRS


class TestEnumeratorHSABIntegration:

    def test_hg_enumeration_includes_selenolate(self):
        """Hg2+ enumeration should include Se_selenolate in at least one candidate."""
        from core.donor_enumerator import enumerate_donor_sets
        candidates = enumerate_donor_sets("Hg2+", pH=7.0, max_candidates=100)
        has_se = any(
            "Se_selenolate" in ds.donor_subtypes for ds in candidates
        )
        assert has_se, "Hg2+ enumeration should include selenolate candidates"

    def test_al_enumeration_includes_fluoride(self):
        """Al3+ enumeration should include F_fluoride in at least one candidate."""
        from core.donor_enumerator import enumerate_donor_sets
        candidates = enumerate_donor_sets("Al3+", pH=7.0, max_candidates=100)
        has_f = any(
            "F_fluoride" in ds.donor_subtypes for ds in candidates
        )
        assert has_f, "Al3+ enumeration should include fluoride candidates"

    def test_pd_enumeration_includes_phosphine(self):
        """Pd2+ (softness=0.75) enumeration should include phosphine."""
        from core.donor_enumerator import enumerate_donor_sets
        candidates = enumerate_donor_sets("Pd2+", pH=7.0, max_candidates=100)
        has_p = any(
            "P_phosphine" in ds.donor_subtypes for ds in candidates
        )
        assert has_p, "Pd2+ enumeration should include phosphine candidates"

    def test_enumeration_no_crash_new_metals(self):
        """New metals should enumerate without exceptions."""
        from core.donor_enumerator import enumerate_donor_sets
        for metal in ["Ru2+", "Rh3+", "Zr4+", "Be2+"]:
            candidates = enumerate_donor_sets(metal, pH=7.0, max_candidates=50)
            assert len(candidates) > 0, f"No candidates for {metal}"


# ──────────────────────────────────────────────────────────────────────────
# 2. Design engine expansion
# ──────────────────────────────────────────────────────────────────────────

class TestMetalAliasesExpanded:

    def test_ruthenium_alias(self):
        from core.design_engine import resolve_metal
        assert resolve_metal("ruthenium") == "Ru2+"

    def test_rhodium_alias(self):
        from core.design_engine import resolve_metal
        assert resolve_metal("rhodium") == "Rh3+"

    def test_zirconium_alias(self):
        from core.design_engine import resolve_metal
        assert resolve_metal("zirconium") == "Zr4+"

    def test_beryllium_alias(self):
        from core.design_engine import resolve_metal
        assert resolve_metal("beryllium") == "Be2+"

    def test_iridium_alias(self):
        from core.design_engine import resolve_metal
        assert resolve_metal("iridium") == "Ir3+"

    def test_tungsten_alias(self):
        from core.design_engine import resolve_metal
        assert resolve_metal("tungsten") == "W4+"


class TestMatrixPresetsExpanded:

    def test_precious_metals_leach_preset(self):
        from core.design_engine import MATRIX_PRESETS
        assert "precious_metals_leach" in MATRIX_PRESETS
        assert "Cu2+" in MATRIX_PRESETS["precious_metals_leach"]

    def test_e_waste_preset(self):
        from core.design_engine import MATRIX_PRESETS
        assert "e_waste" in MATRIX_PRESETS
        assert "Pb2+" in MATRIX_PRESETS["e_waste"]

    def test_nuclear_waste_preset(self):
        from core.design_engine import MATRIX_PRESETS
        assert "nuclear_waste" in MATRIX_PRESETS
        assert "UO2_2+" in MATRIX_PRESETS["nuclear_waste"]

    def test_rare_earth_preset(self):
        from core.design_engine import MATRIX_PRESETS
        assert "rare_earth" in MATRIX_PRESETS

    def test_parse_matrix_with_new_preset(self):
        from core.design_engine import parse_matrix
        pH, metals = parse_matrix("e_waste pH 3")
        assert abs(pH - 3.0) < 0.01
        assert "Pb2+" in metals
        assert "Cu2+" in metals


class TestDesignMode:

    def test_soft_mode_hg(self):
        """design_mode='soft' for Hg2+ should enumerate Se donors."""
        from core.design_engine import design_binder
        result = design_binder(
            "Hg2+",
            interferents=["Ca2+", "Mg2+"],
            pH=7.0,
            top_n=5,
            max_enumerate=80,
            design_mode="soft",
        )
        assert result.n_enumerated > 0
        # At least one note should mention soft mode
        assert any("soft" in n.lower() for n in result.notes)

    def test_hard_mode_al(self):
        """design_mode='hard' for Al3+ should enumerate F/O donors."""
        from core.design_engine import design_binder
        result = design_binder(
            "Al3+",
            interferents=["Ca2+", "Mg2+"],
            pH=5.0,
            top_n=5,
            max_enumerate=80,
            design_mode="hard",
        )
        assert result.n_enumerated > 0
        assert any("hard" in n.lower() for n in result.notes)

    def test_cross_modal_mode(self):
        """design_mode='cross_modal' should only return macrocyclic candidates."""
        from core.design_engine import design_binder
        result = design_binder(
            "Ba2+",
            interferents=["Ca2+", "Na+", "K+"],
            pH=7.0,
            top_n=5,
            max_enumerate=50,
            design_mode="cross_modal",
        )
        # All candidates must be macrocyclic archetypes
        for c in result.candidates:
            assert c.donor_set.is_macrocyclic, (
                f"cross_modal mode returned non-macrocyclic: {c.donor_set}"
            )

    def test_auto_mode_default(self):
        """design_mode='auto' should complete without error for any metal."""
        from core.design_engine import design_binder
        result = design_binder(
            "Pb2+",
            matrix="mine_water pH 5",
            top_n=5,
            max_enumerate=100,
            design_mode="auto",
        )
        assert result.n_scored > 0

    def test_precious_metals_leach_design(self):
        """Full design run using new precious_metals_leach matrix."""
        from core.design_engine import design_binder
        result = design_binder(
            "Au+",
            matrix="precious_metals_leach pH 2",
            top_n=5,
            max_enumerate=100,
            design_mode="soft",
        )
        assert result.n_enumerated > 0
        assert result.target_metal == "Au+"

    def test_new_metal_design_run(self):
        """Rh3+ design should complete and return sensible results."""
        from core.design_engine import design_binder
        result = design_binder(
            "Rh3+",
            interferents=["Fe3+", "Co2+", "Ni2+"],
            pH=5.0,
            top_n=5,
            max_enumerate=80,
        )
        assert result.n_enumerated > 0
        assert not any("failed" in n.lower() and "exception" in n.lower()
                       for n in result.notes)

    def test_result_grades_valid(self):
        """All grade values in result must be A/B/C/D/F."""
        from core.design_engine import design_binder
        result = design_binder(
            "Cu2+",
            interferents=["Ca2+", "Mg2+", "Zn2+"],
            pH=7.0, top_n=10, max_enumerate=100,
        )
        for c in result.candidates:
            assert c.selectivity_grade in ("A", "B", "C", "D", "F"), (
                f"Invalid grade: {c.selectivity_grade}"
            )
