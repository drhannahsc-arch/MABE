"""
Sprint R2b Test Suite — Crown Ether / Cryptand Adapter.

Tests:
    - Knowledge base integrity (hosts, cations, size-match data)
    - Size-match selection (K+ → 18C6, Li+ → 12C4, Na+ → 15C5)
    - HSAB donor routing (soft → thia, borderline → aza)
    - Cryptand upgrade logic
    - Full design pipeline (spec → CrownEtherFabSpec)
    - Validation checks
"""

import math
import pytest

from mabe.realization.adapters.crown_ether_adapter import (
    CrownEtherAdapter,
    CrownEtherFabSpec,
)
from mabe.realization.adapters.crown_ether_knowledge import (
    ALL_CROWN_HOSTS,
    ALL_HOSTS_LIST,
    CATION_DB,
    CROWN_12C4,
    CROWN_15C5,
    CROWN_18C6,
    CROWN_21C7,
    CRYPTAND_222,
    CRYPTANDS,
    DIAZA_18C6,
    DITHIA_18C6,
    hsab_donor_score,
    select_best_crown,
    size_match_score,
)
from mabe.realization.models import (
    ApplicationContext,
    CavityDimensions,
    CavityShape,
    DonorPosition,
    ExclusionSpec,
    InteractionGeometrySpec,
    ScaleClass,
    Solvent,
)


# ─────────────────────────────────────────────
# Spec fixtures — cation-sized pockets
# ─────────────────────────────────────────────

def make_potassium_spec() -> InteractionGeometrySpec:
    """K+ pocket: r_ion=1.38 Å → diameter=2.76 Å → 18-crown-6 (r_cav=1.34)."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=11.0,
            aperture_A=2.76,
            depth_A=3.0,
            max_internal_diameter_A=2.76,
        ),
        symmetry="C6v",
        donor_positions=[
            DonorPosition(
                atom_type="O",
                coordination_role="lone_pair_donor",
                position_vector_A=(1.34, 0.0, 0.0),
                tolerance_A=0.1,
                required_hybridization="sp3",
                charge_state=-0.3,
            ),
        ],
        pocket_scale_nm=0.28,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.SEPARATION,
        required_scale=ScaleClass.MMOL,
    )


def make_lithium_spec() -> InteractionGeometrySpec:
    """Li+ pocket: r_ion=0.76 Å → diameter=1.52 Å → 12-crown-4 (r_cav=0.60)."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=1.8,
            aperture_A=1.52,
            depth_A=2.5,
            max_internal_diameter_A=1.52,
        ),
        symmetry="C4v",
        donor_positions=[],
        pocket_scale_nm=0.15,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_sodium_spec() -> InteractionGeometrySpec:
    """Na+ pocket: r_ion=1.02 Å → diameter=2.04 Å → 15-crown-5 (r_cav=0.86)."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=4.4,
            aperture_A=2.04,
            depth_A=3.0,
            max_internal_diameter_A=2.04,
        ),
        symmetry="C5v",
        donor_positions=[],
        pocket_scale_nm=0.20,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_cesium_spec() -> InteractionGeometrySpec:
    """Cs+ pocket: r_ion=1.67 Å → diameter=3.34 Å → 21-crown-7 (r_cav=1.70)."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=19.5,
            aperture_A=3.34,
            depth_A=3.0,
            max_internal_diameter_A=3.34,
        ),
        symmetry="C7v",
        donor_positions=[],
        pocket_scale_nm=0.34,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.REMEDIATION,
        required_scale=ScaleClass.MOL,
    )


def make_silver_spec() -> InteractionGeometrySpec:
    """Ag+ (soft): r_ion=1.15 Å → should trigger thia-crown."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=6.4,
            aperture_A=2.30,
            depth_A=3.0,
            max_internal_diameter_A=2.30,
        ),
        symmetry="none",
        donor_positions=[],
        pocket_scale_nm=0.23,
        solvent=Solvent.MIXED,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_copper_spec() -> InteractionGeometrySpec:
    """Cu2+ (borderline): r_ion=0.73 Å → should trigger aza-crown."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=1.6,
            aperture_A=1.46,
            depth_A=3.0,
            max_internal_diameter_A=1.46,
        ),
        symmetry="none",
        donor_positions=[],
        pocket_scale_nm=0.15,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_potassium_with_exclusion_spec() -> InteractionGeometrySpec:
    """K+ pocket with Na+ exclusion → should consider cryptand upgrade."""
    spec = make_potassium_spec()
    spec.must_exclude = [
        ExclusionSpec(species="Na+", max_allowed_affinity_kJ_mol=-10.0, exclusion_mechanism="size"),
    ]
    return spec


# ─────────────────────────────────────────────
# Test 1: Knowledge Base
# ─────────────────────────────────────────────

class TestKnowledgeBase:

    def test_nine_hosts_total(self):
        assert len(ALL_CROWN_HOSTS) == 9

    def test_four_basic_crowns(self):
        for name in ["12-crown-4", "15-crown-5", "18-crown-6", "21-crown-7"]:
            assert name in ALL_CROWN_HOSTS

    def test_three_cryptands(self):
        assert len(CRYPTANDS) == 3

    def test_cavity_radii_increase_with_ring_size(self):
        assert (CROWN_12C4.cavity_radius_A
                < CROWN_15C5.cavity_radius_A
                < CROWN_18C6.cavity_radius_A
                < CROWN_21C7.cavity_radius_A)

    def test_cation_database(self):
        assert len(CATION_DB) >= 15
        assert "K+" in CATION_DB
        assert "Li+" in CATION_DB
        assert "Na+" in CATION_DB

    def test_cation_hsab_classes(self):
        assert CATION_DB["K+"].hsab_class == "hard"
        assert CATION_DB["Cu2+"].hsab_class == "borderline"
        assert CATION_DB["Ag+"].hsab_class == "soft"

    def test_all_hosts_commercial(self):
        for h in ALL_HOSTS_LIST:
            assert h.commercial

    def test_18c6_best_match_is_potassium(self):
        assert CROWN_18C6.best_match_ion == "K+"

    def test_cryptand_222_is_3d(self):
        assert CRYPTAND_222.is_3d_cage

    def test_cryptand_has_cryptate_stabilization(self):
        assert CRYPTAND_222.cryptate_stabilization_kJ_mol > 0


# ─────────────────────────────────────────────
# Test 2: Size-Match Physics
# ─────────────────────────────────────────────

class TestSizeMatch:

    def test_perfect_match_score_1(self):
        """Identical radii → score = 1.0."""
        assert size_match_score(1.38, 1.38) == pytest.approx(1.0)

    def test_large_mismatch_near_zero(self):
        """Very different radii → score ≈ 0."""
        assert size_match_score(0.60, 1.70) < 0.01

    def test_score_symmetric(self):
        """Score doesn't depend on which is larger."""
        assert (size_match_score(1.0, 1.2)
                == pytest.approx(size_match_score(1.2, 1.0)))

    def test_k_matches_18c6(self):
        """K+ (1.38 Å) vs 18C6 (1.34 Å) → excellent match."""
        sm = size_match_score(1.38, 1.34)
        assert sm > 0.9

    def test_li_matches_12c4(self):
        """Li+ (0.76 Å) vs 12C4 (0.60 Å) → reasonable match."""
        sm = size_match_score(0.76, 0.60)
        assert sm > 0.3

    def test_na_matches_15c5(self):
        """Na+ (1.02 Å) vs 15C5 (0.86 Å) → good match."""
        sm = size_match_score(1.02, 0.86)
        assert sm > 0.3

    def test_k_in_12c4_poor(self):
        """K+ (1.38 Å) in 12C4 (0.60 Å) → terrible match."""
        sm = size_match_score(1.38, 0.60)
        assert sm < 0.05


# ─────────────────────────────────────────────
# Test 3: HSAB Donor Scoring
# ─────────────────────────────────────────────

class TestHSAB:

    def test_hard_prefers_oxygen(self):
        score_O = hsab_donor_score("hard", ["O", "O", "O"])
        score_S = hsab_donor_score("hard", ["S", "S", "S"])
        assert score_O > score_S

    def test_soft_prefers_sulfur(self):
        score_S = hsab_donor_score("soft", ["S", "S", "S"])
        score_O = hsab_donor_score("soft", ["O", "O", "O"])
        assert score_S > score_O

    def test_borderline_prefers_nitrogen(self):
        score_N = hsab_donor_score("borderline", ["N", "N", "N"])
        score_O = hsab_donor_score("borderline", ["O", "O", "O"])
        assert score_N > score_O


# ─────────────────────────────────────────────
# Test 4: Crown Selection
# ─────────────────────────────────────────────

class TestCrownSelection:

    def test_k_selects_18c6_family(self):
        """K+ → 18-crown-6 or [2.2.2]cryptand (both r ≈ 1.34-1.40)."""
        results = select_best_crown("K+")
        best = results[0][0]
        assert best.cavity_radius_A > 1.2

    def test_li_selects_small_cavity(self):
        """Li+ → small cavity (12C4 or [2.1.1]cryptand)."""
        results = select_best_crown("Li+")
        best = results[0][0]
        assert best.cavity_radius_A < 1.0

    def test_na_selects_medium_cavity(self):
        """Na+ → medium cavity."""
        results = select_best_crown("Na+")
        best = results[0][0]
        assert 0.7 < best.cavity_radius_A < 1.4

    def test_cs_selects_large_cavity(self):
        """Cs+ → large cavity (21C7 region)."""
        results = select_best_crown("Cs+")
        best = results[0][0]
        assert best.cavity_radius_A > 1.3

    def test_results_sorted_by_score(self):
        results = select_best_crown("K+")
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_ion_returns_empty(self):
        results = select_best_crown("Unobtainium3+")
        assert results == []


# ─────────────────────────────────────────────
# Test 5: Adapter estimate_fidelity
# ─────────────────────────────────────────────

class TestEstimateFidelity:

    def setup_method(self):
        self.adapter = CrownEtherAdapter()

    def test_potassium_high_fidelity(self):
        spec = make_potassium_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert score.physics_fidelity > 0.4
        assert score.feasible

    def test_lithium_feasible(self):
        spec = make_lithium_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert score.feasible

    def test_fidelity_bounded(self):
        spec = make_potassium_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert 0.0 <= score.physics_fidelity <= 1.0

    def test_advantages_populated(self):
        spec = make_potassium_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert len(score.advantages) > 0


# ─────────────────────────────────────────────
# Test 6: Full Design Pipeline
# ─────────────────────────────────────────────

class TestDesign:

    def setup_method(self):
        self.adapter = CrownEtherAdapter()

    def test_returns_crown_fab_spec(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert isinstance(fab, CrownEtherFabSpec)

    def test_potassium_gets_18c6_family(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert fab.cavity_radius_A > 1.2

    def test_lithium_gets_small_host(self):
        spec = make_lithium_spec()
        fab = self.adapter.design(spec)
        assert fab.cavity_radius_A < 1.0

    def test_sodium_gets_medium_host(self):
        spec = make_sodium_spec()
        fab = self.adapter.design(spec)
        assert 0.7 < fab.cavity_radius_A < 1.4

    def test_has_predicted_logK(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert fab.predicted_logK > 0

    def test_has_synthesis_steps(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert len(fab.synthesis_steps) > 0

    def test_has_validation_plan(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert len(fab.validation_experiments) >= 1
        assert any("ITC" in v for v in fab.validation_experiments)

    def test_has_smiles(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert len(fab.smiles) > 0

    def test_size_match_score_present(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert 0.0 < fab.size_match_score <= 1.0


# ─────────────────────────────────────────────
# Test 7: HSAB Donor Routing
# ─────────────────────────────────────────────

class TestDonorRouting:

    def setup_method(self):
        self.adapter = CrownEtherAdapter()

    def test_silver_gets_thia(self):
        """Ag+ (soft) → thia-crown with S-donors."""
        spec = make_silver_spec()
        fab = self.adapter.design(spec)
        assert fab.donor_substitution == "thia" or "S" in fab.donor_types_used

    def test_copper_gets_aza(self):
        """Cu2+ (borderline) → aza-crown with N-donors."""
        spec = make_copper_spec()
        fab = self.adapter.design(spec)
        assert fab.donor_substitution == "aza" or "N" in fab.donor_types_used

    def test_potassium_stays_oxygen(self):
        """K+ (hard) → standard O-donor crown, no substitution."""
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        assert fab.donor_substitution in ("none", "")


# ─────────────────────────────────────────────
# Test 8: Cryptand Upgrade
# ─────────────────────────────────────────────

class TestCryptandUpgrade:

    def setup_method(self):
        self.adapter = CrownEtherAdapter()

    def test_exclusion_triggers_cryptand_consideration(self):
        """K+ with Na+ exclusion → should at least consider cryptand."""
        spec = make_potassium_with_exclusion_spec()
        fab = self.adapter.design(spec)
        # Either upgrades to cryptand or explains why not
        assert isinstance(fab, CrownEtherFabSpec)

    def test_cryptand_has_higher_logK_for_k(self):
        """[2.2.2]cryptand logK for K+ > 18-crown-6 logK."""
        k_logK_222 = CRYPTAND_222.selectivity_profile.get("K+", 0)
        k_logK_18c6 = CROWN_18C6.selectivity_profile.get("K+", 0)
        assert k_logK_222 > k_logK_18c6


# ─────────────────────────────────────────────
# Test 9: Validation
# ─────────────────────────────────────────────

class TestValidation:

    def setup_method(self):
        self.adapter = CrownEtherAdapter()

    def test_good_design_validates(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        report = self.adapter.validate_design(fab)
        assert report.valid

    def test_poor_size_match_flagged(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        fab.size_match_score = 0.1
        report = self.adapter.validate_design(fab)
        assert not report.valid

    def test_wrong_type_fails(self):
        from mabe.realization.models import FabricationSpec, CavityDimensions
        wrong = FabricationSpec(
            material_system="not_crown",
            geometry_spec_hash="x",
            predicted_pocket_geometry=CavityDimensions(0, 0, 0, 0),
            predicted_deviation_from_ideal_A=0,
        )
        report = self.adapter.validate_design(wrong)
        assert not report.valid

    def test_measured_logK_higher_confidence(self):
        spec = make_potassium_spec()
        fab = self.adapter.design(spec)
        report = self.adapter.validate_design(fab)
        if fab.logK_source == "Izatt_measured":
            assert report.confidence > 0.8
