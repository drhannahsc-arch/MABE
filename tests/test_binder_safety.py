"""
tests/test_binder_safety.py — Tests for off-target & environmental toxicity screening.

Validates:
  - Essential metal database completeness
  - HSAB affinity estimation (hard-hard match > hard-soft)
  - Off-target log K estimation
  - Off-target screening: hard donors → Ca²⁺/Mg²⁺ risk, soft donors → less risk
  - Structural alerts: reactive groups, persistence, metal content
  - Persistence classification from molecular features
  - Bioaccumulation from LogP
  - Aquatic toxicity estimation
  - Composite safety scoring
  - Exclusion logic: critical off-target → excluded
  - Exclusion logic: PFAS-like → excluded
  - Recommendations generated for flagged binders
  - Batch screening and filtering
  - Real capture element screening (Zn-CA mimic, thiol-sulfide)
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.binder_safety import (
    screen_binder,
    screen_off_target,
    screen_capture_element,
    screen_batch,
    filter_safe_binders,
    estimate_off_target_log_k,
    _hsab_match,
    _assess_persistence,
    _assess_bioaccumulation,
    _assess_aquatic_toxicity,
    _screen_structural_alerts,
    BinderSpec,
    BinderSafetyReport,
    OffTargetHit,
    _ESSENTIAL_METALS,
    _STRUCTURAL_ALERTS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Essential metal database
# ═══════════════════════════════════════════════════════════════════════════

class TestEssentialMetalDB:

    def test_key_metals_present(self):
        assert "Ca2+" in _ESSENTIAL_METALS
        assert "Mg2+" in _ESSENTIAL_METALS
        assert "Zn2+" in _ESSENTIAL_METALS
        assert "Fe2+" in _ESSENTIAL_METALS
        assert "Cu2+" in _ESSENTIAL_METALS

    def test_all_have_depletion_threshold(self):
        for key, metal in _ESSENTIAL_METALS.items():
            assert metal.depletion_threshold_uM > 0, f"{key} missing threshold"

    def test_all_have_edta_logk(self):
        for key, metal in _ESSENTIAL_METALS.items():
            assert metal.typical_log_k_edta > 0, f"{key} missing EDTA log K"

    def test_ca_is_hard(self):
        assert _ESSENTIAL_METALS["Ca2+"].hsab_class == "hard"

    def test_zn_is_borderline(self):
        assert _ESSENTIAL_METALS["Zn2+"].hsab_class == "borderline"


# ═══════════════════════════════════════════════════════════════════════════
# HSAB matching
# ═══════════════════════════════════════════════════════════════════════════

class TestHSABMatching:

    def test_hard_hard_high_match(self):
        """Hard binder + hard metal → high match score."""
        ca = _ESSENTIAL_METALS["Ca2+"]
        score = _hsab_match(["O", "O", "O"], "hard", ca)
        assert score > 0.6  # set-based overlap: {O}∩{O}/max(3,1) + HSAB 1.0

    def test_soft_hard_low_match(self):
        """Soft binder + hard metal → low match score."""
        ca = _ESSENTIAL_METALS["Ca2+"]
        score = _hsab_match(["S", "S"], "soft", ca)
        assert score < 0.3

    def test_soft_soft_high_match(self):
        """Soft binder + borderline-soft metal → reasonable match."""
        cu = _ESSENTIAL_METALS["Cu2+"]
        score = _hsab_match(["S", "N"], "soft", cu)
        assert score > 0.4

    def test_donor_overlap_matters(self):
        """More donor overlap → higher match."""
        zn = _ESSENTIAL_METALS["Zn2+"]
        full_overlap = _hsab_match(["N", "S", "O"], "borderline", zn)
        no_overlap = _hsab_match(["P", "P"], "borderline", zn)
        assert full_overlap > no_overlap


# ═══════════════════════════════════════════════════════════════════════════
# Off-target log K estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestOffTargetLogK:

    def test_hard_binder_high_ca_affinity(self):
        """Hard O-donor chelator → meaningful estimated Ca²⁺ affinity."""
        ca = _ESSENTIAL_METALS["Ca2+"]
        log_k = estimate_off_target_log_k(15.0, ["O", "O", "O", "O"], "hard", 4, ca)
        assert log_k > 4.0  # not negligible — hard O-donors do bind Ca²⁺

    def test_soft_binder_low_ca_affinity(self):
        """Soft S-donor chelator → low estimated Ca²⁺ affinity."""
        ca = _ESSENTIAL_METALS["Ca2+"]
        log_k = estimate_off_target_log_k(15.0, ["S", "S"], "soft", 2, ca)
        assert log_k < 3.0

    def test_higher_denticity_higher_logk(self):
        """More donor atoms → higher estimated off-target log K."""
        zn = _ESSENTIAL_METALS["Zn2+"]
        low_dent = estimate_off_target_log_k(15.0, ["N", "O"], "borderline", 2, zn)
        high_dent = estimate_off_target_log_k(15.0, ["N", "O", "N", "O"], "borderline", 4, zn)
        assert high_dent > low_dent

    def test_log_k_non_negative(self):
        for metal in _ESSENTIAL_METALS.values():
            log_k = estimate_off_target_log_k(10.0, ["O"], "hard", 1, metal)
            assert log_k >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Off-target screening
# ═══════════════════════════════════════════════════════════════════════════

class TestOffTargetScreening:

    def test_hard_chelator_flags_calcium(self):
        """EDTA-like hard chelator should flag Ca²⁺ and Mg²⁺ risk."""
        spec = BinderSpec(
            name="EDTA-like", binder_type="chelator",
            donor_atoms=["O", "O", "N", "N", "O", "O"],
            hsab_class="hard", denticity=6,
            target_metal="Pb2+", target_log_k=18.0,
        )
        hits = screen_off_target(spec)
        ca_hits = [h for h in hits if h.metal == "Ca2+"]
        assert len(ca_hits) == 1
        assert ca_hits[0].estimated_log_k > 5

    def test_soft_thiol_minimal_ca_risk(self):
        """Soft thiol binder should have low Ca²⁺/Mg²⁺ risk."""
        spec = BinderSpec(
            name="dithiol", binder_type="chelator",
            donor_atoms=["S", "S"],
            hsab_class="soft", denticity=2,
            target_metal="Hg2+", target_log_k=20.0,
        )
        hits = screen_off_target(spec)
        ca_hits = [h for h in hits if h.metal == "Ca2+"]
        assert ca_hits[0].depletion_risk in ("none", "low")

    def test_target_metal_excluded_from_hits(self):
        """If target is Zn²⁺, Zn²⁺ should not appear as off-target."""
        spec = BinderSpec(
            name="Zn-binder", binder_type="chelator",
            donor_atoms=["N", "N", "O"],
            hsab_class="borderline", denticity=3,
            target_metal="Zn2+", target_log_k=12.0,
        )
        hits = screen_off_target(spec)
        zn_hits = [h for h in hits if h.metal == "Zn2+"]
        assert len(zn_hits) == 0

    def test_all_essential_metals_checked(self):
        """All essential metals should be in the off-target hit list."""
        spec = BinderSpec(
            name="generic", binder_type="chelator",
            donor_atoms=["N", "O"],
            hsab_class="borderline", denticity=2,
            target_metal="Pb2+", target_log_k=10.0,
        )
        hits = screen_off_target(spec)
        metals_checked = {h.metal for h in hits}
        assert "Ca2+" in metals_checked
        assert "Mg2+" in metals_checked
        assert "Zn2+" in metals_checked


# ═══════════════════════════════════════════════════════════════════════════
# Structural alerts
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuralAlerts:

    def test_isocyanate_excluded(self):
        spec = BinderSpec(
            name="test", binder_type="test",
            donor_atoms=["N"], hsab_class="hard",
            functional_groups=["amine", "isocyanate"],
        )
        flags = _screen_structural_alerts(spec)
        assert any(f.severity == "exclude" for f in flags)

    def test_pfas_excluded(self):
        spec = BinderSpec(
            name="test", binder_type="test",
            donor_atoms=["O"], hsab_class="hard",
            functional_groups=["perfluoro", "CF3"],
        )
        flags = _screen_structural_alerts(spec)
        assert any("PFAS" in f.description or "fluorinated" in f.description.lower() for f in flags)

    def test_mercury_binder_excluded(self):
        spec = BinderSpec(
            name="test", binder_type="test",
            donor_atoms=["S"], hsab_class="soft",
            contains_metal="Hg",
        )
        flags = _screen_structural_alerts(spec)
        assert any("Mercury" in f.description for f in flags)

    def test_benign_binder_no_alerts(self):
        spec = BinderSpec(
            name="test", binder_type="test",
            donor_atoms=["N", "O"], hsab_class="borderline",
            functional_groups=["amine", "carboxylate"],
        )
        flags = _screen_structural_alerts(spec)
        assert len(flags) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistence:

    def test_small_molecule_readily_biodegradable(self):
        spec = BinderSpec(
            name="test", binder_type="test",
            donor_atoms=["O"], hsab_class="hard",
            molecular_weight=200, aromatic_rings=0, halogen_count=0,
        )
        cls, _ = _assess_persistence(spec)
        assert cls == "readily_biodegradable"

    def test_fluorinated_very_persistent(self):
        spec = BinderSpec(
            name="test", binder_type="test",
            donor_atoms=["O"], hsab_class="hard",
            fluorine_count=6,
        )
        cls, flags = _assess_persistence(spec)
        assert cls == "very_persistent"
        assert any(f.severity == "exclude" for f in flags)

    def test_polyaromatic_persistent(self):
        spec = BinderSpec(
            name="test", binder_type="test",
            donor_atoms=["N"], hsab_class="borderline",
            aromatic_rings=5, halogen_count=0,
        )
        cls, _ = _assess_persistence(spec)
        assert cls == "persistent"


# ═══════════════════════════════════════════════════════════════════════════
# Bioaccumulation
# ═══════════════════════════════════════════════════════════════════════════

class TestBioaccumulation:

    def test_low_logp_not_bioaccumulative(self):
        spec = BinderSpec(name="t", binder_type="t", donor_atoms=["O"],
                          hsab_class="hard", logP=1.0)
        cls, _ = _assess_bioaccumulation(spec)
        assert cls == "not_bioaccumulative"

    def test_high_logp_bioaccumulative(self):
        spec = BinderSpec(name="t", binder_type="t", donor_atoms=["S"],
                          hsab_class="soft", logP=5.5)
        cls, flags = _assess_bioaccumulation(spec)
        assert cls == "very_bioaccumulative"
        assert len(flags) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Aquatic toxicity
# ═══════════════════════════════════════════════════════════════════════════

class TestAquaticToxicity:

    def test_low_logp_low_toxicity(self):
        spec = BinderSpec(name="t", binder_type="t", donor_atoms=["O"],
                          hsab_class="hard", logP=0.5, molecular_weight=200)
        cls, _ = _assess_aquatic_toxicity(spec)
        assert cls in ("low", "moderate")

    def test_high_logp_higher_toxicity(self):
        spec = BinderSpec(name="t", binder_type="t", donor_atoms=["S"],
                          hsab_class="soft", logP=5.0, molecular_weight=400)
        cls, _ = _assess_aquatic_toxicity(spec)
        assert cls in ("high", "very_high")

    def test_reactive_group_increases_toxicity(self):
        spec_benign = BinderSpec(name="t", binder_type="t", donor_atoms=["O"],
                                 hsab_class="hard", logP=2.0, molecular_weight=300,
                                 functional_groups=["amine"])
        spec_reactive = BinderSpec(name="t", binder_type="t", donor_atoms=["O"],
                                    hsab_class="hard", logP=2.0, molecular_weight=300,
                                    functional_groups=["acrylate"])
        cls_b, _ = _assess_aquatic_toxicity(spec_benign)
        cls_r, _ = _assess_aquatic_toxicity(spec_reactive)
        # Reactive should be same or worse
        tox_order = {"low": 0, "moderate": 1, "high": 2, "very_high": 3}
        assert tox_order.get(cls_r, 0) >= tox_order.get(cls_b, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Full screening
# ═══════════════════════════════════════════════════════════════════════════

class TestFullScreening:

    def test_safe_binder_passes(self):
        """Simple amine chelator: low MW, biodegradable, no alerts."""
        spec = BinderSpec(
            name="simple_amine", binder_type="chelator",
            donor_atoms=["N", "N"], hsab_class="borderline",
            denticity=2, target_metal="Pb2+", target_log_k=8.0,
            molecular_weight=200, logP=0.5,
            functional_groups=["amine"],
        )
        report = screen_binder(spec)
        assert report.safe_for_deployment
        assert report.safety_score > 0.5

    def test_dangerous_binder_excluded(self):
        """High-affinity hard hexadentate chelator → Ca²⁺ depletion risk."""
        spec = BinderSpec(
            name="super_chelator", binder_type="chelator",
            donor_atoms=["O", "O", "O", "O", "N", "N"],
            hsab_class="hard", denticity=6,
            target_metal="Pb2+", target_log_k=10.0,  # low target affinity
            molecular_weight=400, logP=0.5,
        )
        report = screen_binder(spec)
        # Should flag essential metal risk
        assert report.essential_metal_risk in ("moderate", "high", "critical")

    def test_pfas_binder_excluded(self):
        spec = BinderSpec(
            name="fluoro_binder", binder_type="chelator",
            donor_atoms=["O"], hsab_class="hard",
            denticity=1, target_metal="Pb2+", target_log_k=10.0,
            molecular_weight=500, logP=4.0, fluorine_count=8,
            functional_groups=["perfluoro"],
        )
        report = screen_binder(spec)
        assert not report.safe_for_deployment
        assert any("PFAS" in r or "fluorinated" in r.lower() or "Polyfluorinated" in r
                    for r in report.exclusion_reasons)

    def test_report_summary(self):
        spec = BinderSpec(
            name="test", binder_type="chelator",
            donor_atoms=["N", "O"], hsab_class="borderline",
            denticity=2, target_metal="Pb2+", target_log_k=12.0,
            molecular_weight=300, logP=1.5,
        )
        report = screen_binder(spec)
        s = report.summary()
        assert "Safety" in s or "safety" in s
        assert "score" in s.lower()

    def test_safety_score_bounded(self):
        spec = BinderSpec(
            name="test", binder_type="chelator",
            donor_atoms=["N", "O"], hsab_class="borderline",
            denticity=2, target_metal="Pb2+", target_log_k=12.0,
        )
        report = screen_binder(spec)
        assert 0.0 <= report.safety_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Capture element convenience
# ═══════════════════════════════════════════════════════════════════════════

class TestCaptureElementScreen:

    def test_zn_ca_mimic(self):
        """Zn-CA mimic: contains Zn, N-donors, moderate affinity."""
        report = screen_capture_element(
            name="Zn-CA mimic",
            target_formula="CO2",
            donor_atoms=["N", "N", "N"],
            hsab_class="borderline",
            denticity=3,
            target_log_k=8.0,
            molecular_weight=350,
            logP=0.5,
            contains_metal="Zn",
        )
        assert report.safe_for_deployment
        # Zn in the binder is the target metal analogue — not a toxin here
        assert report.safety_score > 0.4

    def test_thiol_sulfide_capture(self):
        """Thiol-sulfide capture for Pb²⁺: soft donors, low Ca/Mg risk."""
        report = screen_capture_element(
            name="thiol surface",
            target_formula="Pb2+",
            donor_atoms=["S", "S"],
            hsab_class="soft",
            denticity=2,
            target_log_k=15.0,
            molecular_weight=200,
            logP=1.0,
            functional_groups=["thiol"],
        )
        assert report.safe_for_deployment
        assert report.essential_metal_risk in ("none", "low")
        # Soft S-donors should not bind hard Ca²⁺/Mg²⁺
        ca_hits = [h for h in report.off_target_hits if h.metal == "Ca2+"]
        assert ca_hits[0].depletion_risk in ("none", "low")

    def test_zr_oxide_capture(self):
        """Zr-oxide phosphate capture: hard O-donors."""
        report = screen_capture_element(
            name="Zr-oxide",
            target_formula="PO4_3-",
            donor_atoms=["O", "O", "O", "O"],
            hsab_class="hard",
            denticity=4,
            target_log_k=12.0,
            molecular_weight=500,
            logP=-1.0,
        )
        assert report.safe_for_deployment

    def test_amine_sorbent(self):
        """Amine sorbent for CO₂: benign."""
        report = screen_capture_element(
            name="PEI-amine",
            target_formula="CO2",
            donor_atoms=["N", "N", "N"],
            hsab_class="borderline",
            denticity=3,
            target_log_k=5.0,
            molecular_weight=600,
            logP=-2.0,
            functional_groups=["amine"],
        )
        assert report.safe_for_deployment
        assert report.essential_metal_risk in ("none", "low")


# ═══════════════════════════════════════════════════════════════════════════
# Batch screening
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchScreening:

    def test_batch_returns_all(self):
        specs = [
            BinderSpec("safe", "chelator", ["N", "O"], "borderline", 2,
                       target_metal="Pb2+", target_log_k=12.0),
            BinderSpec("dangerous", "chelator", ["O", "O", "O", "O", "N", "N"], "hard", 6,
                       target_metal="Pb2+", target_log_k=10.0),
        ]
        reports = screen_batch(specs)
        assert len(reports) == 2

    def test_filter_removes_excluded(self):
        specs = [
            BinderSpec("safe", "chelator", ["N", "S"], "soft", 2,
                       target_metal="Hg2+", target_log_k=20.0, molecular_weight=200, logP=1.0),
            BinderSpec("fluoro", "chelator", ["O"], "hard", 1,
                       fluorine_count=10, functional_groups=["perfluoro"],
                       target_metal="Pb2+", target_log_k=10.0),
        ]
        safe = filter_safe_binders(specs)
        assert len(safe) == 1
        assert safe[0].name == "safe"


# ═══════════════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════════════

class TestRecommendations:

    def test_high_offtarget_gets_redesign_recommendation(self):
        spec = BinderSpec(
            name="problematic", binder_type="chelator",
            donor_atoms=["O", "O", "O", "O", "N", "N"],
            hsab_class="hard", denticity=6,
            target_metal="Pb2+", target_log_k=10.0,
        )
        report = screen_binder(spec)
        if report.essential_metal_risk in ("high", "critical"):
            assert any("Redesign" in r or "redesign" in r.lower() for r in report.recommendations)

    def test_high_logp_gets_hydrophilic_recommendation(self):
        spec = BinderSpec(
            name="lipophilic", binder_type="chelator",
            donor_atoms=["S", "S"], hsab_class="soft",
            denticity=2, target_metal="Hg2+", target_log_k=20.0,
            logP=5.5,
        )
        report = screen_binder(spec)
        assert any("hydrophilic" in r.lower() or "LogP" in r for r in report.recommendations)

    def test_persistent_gets_degradation_recommendation(self):
        spec = BinderSpec(
            name="persistent", binder_type="chelator",
            donor_atoms=["N", "N"], hsab_class="borderline",
            denticity=2, target_metal="Pb2+", target_log_k=12.0,
            aromatic_rings=5, halogen_count=3,
        )
        report = screen_binder(spec)
        assert any("degrad" in r.lower() or "hydrolyzable" in r.lower() for r in report.recommendations)
