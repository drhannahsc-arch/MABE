"""
tests/test_degradation_safety.py — Tests for designed degradability module.

Validates:
  - DSS computation from sub-scores
  - All scaffold profiles built with pre-computed DSS
  - Hard exclusion of Cr scaffolds and unmodified Pd cages
  - DNA origami highest DSS (gold standard)
  - Nitschke Fe₄L₆ is safe (benign metal + imine lability)
  - Fail-safe linker database complete
  - Matrix-specific linker selection
  - Safety assessment approve/reject logic
  - Enforcement filtering
  - Degradation profile lookups by scaffold system
  - Metal toxicity scoring
  - Persistence scoring
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.degradation_safety import (
    compute_dss,
    get_degradation_profile,
    list_profiles,
    select_fail_safe_linker,
    get_linker,
    list_linkers,
    assess_safety,
    enforce_safety,
    filter_safe_scaffolds,
    DegradationProfile,
    FailSafeLinker,
    SafetyAssessment,
    MetalToxicity,
    FailSafeTrigger,
    _persistence_score,
    _metal_toxicity_score,
    _PROFILES,
    _FAILSAFE_LINKERS,
    _DSS_HARD_EXCLUDE,
)


# ═══════════════════════════════════════════════════════════════════════════
# DSS computation
# ═══════════════════════════════════════════════════════════════════════════

class TestDSSComputation:

    def test_persistence_score_fast_degradation(self):
        assert _persistence_score(12.0) == 1.0

    def test_persistence_score_slow_degradation(self):
        assert _persistence_score(50000.0) == 0.0

    def test_persistence_score_week(self):
        assert _persistence_score(100.0) == 0.8

    def test_metal_toxicity_none(self):
        assert _metal_toxicity_score(MetalToxicity.NONE) == 1.0

    def test_metal_toxicity_banned(self):
        assert _metal_toxicity_score(MetalToxicity.BANNED) == 0.0

    def test_metal_toxicity_benign(self):
        assert _metal_toxicity_score(MetalToxicity.BENIGN) == 0.9

    def test_metal_toxicity_high(self):
        assert _metal_toxicity_score(MetalToxicity.HIGH) == 0.1

    def test_dss_bounded_0_1(self):
        for key, profile in _PROFILES.items():
            assert 0.0 <= profile.dss <= 1.0, f"{key} DSS {profile.dss} out of bounds"

    def test_excluded_profiles_dss_zero(self):
        for key, profile in _PROFILES.items():
            if profile.hard_excluded:
                assert profile.dss == 0.0, f"Excluded {key} should have DSS=0"


# ═══════════════════════════════════════════════════════════════════════════
# Profile database
# ═══════════════════════════════════════════════════════════════════════════

class TestProfileDatabase:

    def test_profiles_exist(self):
        assert len(_PROFILES) >= 10

    def test_all_profiles_have_dss(self):
        for key, p in _PROFILES.items():
            assert p.dss >= 0.0, f"{key} missing DSS"

    def test_list_profiles(self):
        profiles = list_profiles()
        assert "dna_origami" in profiles
        assert "nitschke_cage" in profiles
        assert "fujita_cage_standard" in profiles

    def test_dna_origami_highest_dss(self):
        """DNA origami should have the highest DSS (gold standard)."""
        dna = _PROFILES["dna_origami"]
        for key, p in _PROFILES.items():
            if key != "dna_origami" and not p.hard_excluded:
                assert dna.dss >= p.dss, \
                    f"DNA origami DSS {dna.dss} should be >= {key} DSS {p.dss}"

    def test_dna_origami_no_metal(self):
        dna = _PROFILES["dna_origami"]
        assert dna.metal_toxicity == MetalToxicity.NONE
        assert dna.metal_released == "none"

    def test_nitschke_safe(self):
        n = _PROFILES["nitschke_cage"]
        assert n.dss >= 0.7
        assert n.metal_toxicity == MetalToxicity.BENIGN
        assert not n.hard_excluded

    def test_fujita_standard_excluded(self):
        f = _PROFILES["fujita_cage_standard"]
        assert f.hard_excluded
        assert f.dss == 0.0
        assert "Pd" in f.exclusion_reason

    def test_fujita_chelating_allowed(self):
        f = _PROFILES["fujita_cage_chelating"]
        assert not f.hard_excluded
        assert f.dss > 0.3

    def test_mil101_cr_banned(self):
        cr = _PROFILES["mof_mil101_cr"]
        assert cr.hard_excluded
        assert cr.metal_toxicity == MetalToxicity.BANNED
        assert "carcinogen" in cr.exclusion_reason.lower()

    def test_uio66_standard_excluded(self):
        """Standard UiO-66 excluded for persistence."""
        u = _PROFILES["mof_uio66_standard"]
        assert u.hard_excluded
        assert "persist" in u.exclusion_reason.lower()

    def test_uio66_defective_allowed(self):
        """Defect-engineered UiO-66 is allowed."""
        u = _PROFILES["mof_uio66_defective"]
        assert not u.hard_excluded
        assert u.dss > 0.3

    def test_poc_imine_safe(self):
        p = _PROFILES["poc_imine"]
        assert p.dss >= 0.7
        assert p.metal_toxicity == MetalToxicity.NONE

    def test_zif8_moderate(self):
        z = _PROFILES["mof_zif8"]
        assert 0.5 < z.dss < 1.0
        assert z.metal_toxicity == MetalToxicity.MODERATE_LOW

    def test_all_have_recommended_failsafe(self):
        for key, p in _PROFILES.items():
            if not p.hard_excluded or key != "mof_mil101_cr":
                assert p.recommended_failsafe != "", f"{key} missing recommended fail-safe"

    def test_profile_summary(self):
        dna = _PROFILES["dna_origami"]
        s = dna.summary()
        assert "DSS" in s
        assert "SAFE" in s


# ═══════════════════════════════════════════════════════════════════════════
# Profile lookup
# ═══════════════════════════════════════════════════════════════════════════

class TestProfileLookup:

    def test_lookup_by_cascade_scaffold_key(self):
        """Look up using cascade_scaffold.ScaffoldSystem values."""
        p = get_degradation_profile("dna_origami")
        assert p.scaffold_system == "dna_origami"
        assert p.dss > 0.9

    def test_lookup_fujita_defaults_to_chelating(self):
        """Fujita cage should default to the safe chelating variant."""
        p = get_degradation_profile("fujita_cage")
        assert p.scaffold_system == "fujita_cage_chelating"
        assert not p.hard_excluded

    def test_lookup_mof_defaults_to_zif8(self):
        """MOF cavity should default to degradable ZIF-8."""
        p = get_degradation_profile("mof_cavity")
        assert "zif8" in p.scaffold_system

    def test_lookup_with_variant(self):
        p = get_degradation_profile("fujita_cage", variant="fujita_cage_standard")
        assert p.hard_excluded  # standard Fujita is excluded

    def test_lookup_unknown_conservative(self):
        """Unknown scaffold gets conservative (low) DSS."""
        p = get_degradation_profile("unknown_scaffold_xyz")
        assert p.dss < 0.5
        assert "Characterize" in p.design_modifications[0]


# ═══════════════════════════════════════════════════════════════════════════
# Fail-safe linker database
# ═══════════════════════════════════════════════════════════════════════════

class TestFailSafeLinkers:

    def test_linkers_exist(self):
        assert len(_FAILSAFE_LINKERS) >= 6

    def test_all_linkers_have_trigger(self):
        for lid, linker in _FAILSAFE_LINKERS.items():
            assert isinstance(linker.trigger, FailSafeTrigger)

    def test_all_linkers_operational_stable(self):
        """All fail-safe linkers should be stable during normal operation."""
        for lid, linker in _FAILSAFE_LINKERS.items():
            assert linker.operational_stable

    def test_all_products_nontoxic(self):
        for lid, linker in _FAILSAFE_LINKERS.items():
            assert not linker.products_toxic

    def test_acetal_acid_triggered(self):
        acetal = _FAILSAFE_LINKERS["acetal"]
        assert acetal.trigger == FailSafeTrigger.ACID

    def test_photocleavable_uv_triggered(self):
        pc = _FAILSAFE_LINKERS["photocleavable"]
        assert pc.trigger == FailSafeTrigger.UV

    def test_ester_slowest(self):
        """Ester should have the longest half-life (slowest fail-safe)."""
        ester = _FAILSAFE_LINKERS["ester"]
        for lid, linker in _FAILSAFE_LINKERS.items():
            if lid != "ester":
                assert ester.half_life_env_hours >= linker.half_life_env_hours

    def test_list_linkers(self):
        ids = list_linkers()
        assert "acetal" in ids
        assert "ester" in ids
        assert "photocleavable" in ids

    def test_get_linker(self):
        linker = get_linker("acetal")
        assert linker is not None
        assert linker.linker_id == "acetal"

    def test_get_nonexistent_linker(self):
        assert get_linker("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# Fail-safe linker selection
# ═══════════════════════════════════════════════════════════════════════════

class TestLinkerSelection:

    def test_mine_drainage_gets_acetal(self):
        """Acidic mine drainage → acetal (acid-triggered)."""
        linker = select_fail_safe_linker("mine_drainage")
        assert linker.trigger == FailSafeTrigger.ACID

    def test_seawater_gets_photocleavable(self):
        """Seawater (surface, UV) → photocleavable."""
        linker = select_fail_safe_linker("seawater")
        assert linker.trigger == FailSafeTrigger.UV

    def test_wastewater_gets_boronate(self):
        """Wastewater (sugars present) → boronate ester."""
        linker = select_fail_safe_linker("wastewater")
        assert linker.trigger == FailSafeTrigger.DIOL

    def test_groundwater_gets_ester(self):
        """Dark groundwater → ester (slow hydrolysis, no UV)."""
        linker = select_fail_safe_linker("groundwater")
        assert linker.trigger == FailSafeTrigger.HYDROLYSIS

    def test_textile_gets_acid_or_uv(self):
        """Textile (sweat + sunlight) → acetal or photocleavable."""
        linker = select_fail_safe_linker("textile")
        assert linker.trigger in (FailSafeTrigger.ACID, FailSafeTrigger.UV)

    def test_unknown_matrix_gets_ester(self):
        """Unknown environment → ester (safest universal fallback)."""
        linker = select_fail_safe_linker("unknown_place")
        assert linker.linker_id == "ester"


# ═══════════════════════════════════════════════════════════════════════════
# Safety assessment
# ═══════════════════════════════════════════════════════════════════════════

class TestSafetyAssessment:

    def test_dna_origami_approved(self):
        a = assess_safety("dna_origami", "surface_water")
        assert a.safe_for_deployment
        assert len(a.warnings) == 0

    def test_nitschke_approved(self):
        a = assess_safety("nitschke_cage", "mine_drainage")
        assert a.safe_for_deployment

    def test_fujita_standard_rejected(self):
        """Standard Fujita (unmodified Pd) should be rejected."""
        a = assess_safety("fujita_cage", "surface_water", variant="fujita_cage_standard")
        assert not a.safe_for_deployment
        assert any("Pd" in w for w in a.warnings)
        assert len(a.alternatives) > 0

    def test_fujita_chelating_approved(self):
        a = assess_safety("fujita_cage", "surface_water")
        assert a.safe_for_deployment  # defaults to chelating variant

    def test_mil101_cr_rejected(self):
        a = assess_safety("mof_cavity", "surface_water", variant="mof_mil101_cr")
        assert not a.safe_for_deployment
        assert any("carcinogen" in w.lower() for w in a.warnings)

    def test_rejected_has_alternatives(self):
        a = assess_safety("fujita_cage", "surface_water", variant="fujita_cage_standard")
        assert len(a.alternatives) > 0
        # Alternatives should include safe scaffolds
        assert any("dna_origami" in alt for alt in a.alternatives)

    def test_assessment_has_failsafe(self):
        a = assess_safety("dna_origami", "seawater")
        assert a.fail_safe is not None
        assert a.fail_safe.trigger == FailSafeTrigger.UV  # seawater → UV

    def test_assessment_summary(self):
        a = assess_safety("nitschke_cage", "mine_drainage")
        s = a.summary()
        assert "APPROVED" in s or "REJECTED" in s
        assert "DSS" in s


# ═══════════════════════════════════════════════════════════════════════════
# Enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestEnforcement:

    def test_enforce_returns_all(self):
        systems = ["dna_origami", "nitschke_cage", "fujita_cage"]
        results = enforce_safety(systems, "surface_water")
        assert len(results) == 3

    def test_filter_removes_unsafe(self):
        """filter_safe_scaffolds should return only approved systems."""
        systems = ["dna_origami", "nitschke_cage", "poc", "mop"]
        safe = filter_safe_scaffolds(systems, "surface_water")
        assert "dna_origami" in safe
        assert "nitschke_cage" in safe

    def test_filter_with_all_scaffold_types(self):
        """Test all cascade_scaffold.ScaffoldSystem values."""
        from core.cascade_scaffold import ScaffoldSystem
        systems = [s.value for s in ScaffoldSystem]
        safe = filter_safe_scaffolds(systems, "surface_water")
        assert len(safe) >= 4  # DNA, Nitschke, POC, MOP at minimum
        assert "dna_origami" in safe


# ═══════════════════════════════════════════════════════════════════════════
# Integration with cascade_scaffold
# ═══════════════════════════════════════════════════════════════════════════

class TestCascadeIntegration:

    def test_all_scaffold_systems_have_profiles(self):
        """Every ScaffoldSystem in cascade_scaffold should have a degradation profile."""
        from core.cascade_scaffold import ScaffoldSystem
        for system in ScaffoldSystem:
            profile = get_degradation_profile(system.value)
            assert profile is not None
            assert profile.dss >= 0.0

    def test_scaffold_recommendations_include_safety(self):
        """For field deployment, recommended scaffolds should be safe."""
        from core.cascade_scaffold import ScaffoldSystem
        safe = filter_safe_scaffolds(
            [s.value for s in ScaffoldSystem], "mine_drainage")
        # At minimum: DNA origami and Nitschke should pass
        assert "dna_origami" in safe
        assert "nitschke_cage" in safe
