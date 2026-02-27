"""
Sprint R3 Test Suite — Adapter Registry Integration.

Tests:
    - Adapter registry auto-discovery
    - Weight profiles (application-aware)
    - Composite scoring across physics classes
    - End-to-end ranking: precision vs bulk sorbent
    - Application context switching
    - Lignin adapter design pipeline
    - HSAB routing for bulk materials
"""

import pytest

from mabe.realization.adapter_registry import (
    ADAPTER_REGISTRY,
    AdapterRegistry,
    register_builtins,
)
from mabe.realization.adapters.lignin_adapter import (
    CATION_HSAB,
    FUNCTIONAL_GROUPS,
    FunctionalizedLigninAdapter,
    LigninFabSpec,
    LIGNIN_BACKBONES,
    select_functional_group,
)
from mabe.realization.ranker import (
    WEIGHT_PROFILES,
    AdapterRanker,
    WeightProfile,
    compute_composite,
    get_weight_profile,
)
from mabe.realization.models import (
    ApplicationContext,
    CavityDimensions,
    CavityShape,
    DeviationReport,
    DonorPosition,
    InteractionGeometrySpec,
    RealizationScore,
    ScaleClass,
    Solvent,
)


# ─────────────────────────────────────────────
# Spec fixtures
# ─────────────────────────────────────────────

def make_organic_guest_spec() -> InteractionGeometrySpec:
    """Medium organic guest → β-CD territory."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=155.0,
            aperture_A=6.0,
            depth_A=7.9,
            max_internal_diameter_A=6.0,
        ),
        symmetry="none",
        donor_positions=[],
        pocket_scale_nm=0.60,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_potassium_spec() -> InteractionGeometrySpec:
    """K⁺ pocket for crown ether."""
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
                atom_type="O", coordination_role="equatorial",
                position_vector_A=(1.34, 0.0, 0.0),
                tolerance_A=0.1, required_hybridization="sp3",
            ),
        ],
        pocket_scale_nm=0.28,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_cu2_4N_spec() -> InteractionGeometrySpec:
    """Cu²⁺ 4N planar → porphyrin territory."""
    donors = [
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(1.98, 0.0, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(0.0, 1.98, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(-1.98, 0.0, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(0.0, -1.98, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
    ]
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=33.0,
            aperture_A=3.96,
            depth_A=3.4,
            max_internal_diameter_A=3.96,
        ),
        symmetry="D4h",
        donor_positions=donors,
        pocket_scale_nm=0.40,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_pb2_remediation_spec() -> InteractionGeometrySpec:
    """Pb²⁺ at remediation scale → bulk sorbent should win."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=15.0,
            aperture_A=2.38,
            depth_A=3.0,
            max_internal_diameter_A=2.38,
        ),
        symmetry="none",
        donor_positions=[
            DonorPosition(
                atom_type="S", coordination_role="terminal",
                position_vector_A=(1.19, 0.0, 0.0),
                tolerance_A=0.5, required_hybridization="any",
            ),
        ],
        pocket_scale_nm=0.24,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.REMEDIATION,
        required_scale=ScaleClass.MOL,
    )


def make_soft_metal_spec() -> InteractionGeometrySpec:
    """Soft metal with S-donors → should trigger thiol/DTC in lignin."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=10.0,
            aperture_A=2.00,
            depth_A=3.0,
            max_internal_diameter_A=2.00,
        ),
        symmetry="none",
        donor_positions=[
            DonorPosition(
                atom_type="S", coordination_role="terminal",
                position_vector_A=(1.0, 0.0, 0.0),
                tolerance_A=0.3, required_hybridization="any",
            ),
        ],
        pocket_scale_nm=0.20,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.REMEDIATION,
        required_scale=ScaleClass.MOL,
    )


def make_hard_metal_spec() -> InteractionGeometrySpec:
    """Hard metal with O-donors → should trigger carboxylate in lignin."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=10.0,
            aperture_A=2.40,
            depth_A=3.0,
            max_internal_diameter_A=2.40,
        ),
        symmetry="none",
        donor_positions=[
            DonorPosition(
                atom_type="O", coordination_role="terminal",
                position_vector_A=(1.2, 0.0, 0.0),
                tolerance_A=0.3, required_hybridization="any",
            ),
        ],
        pocket_scale_nm=0.24,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.REMEDIATION,
        required_scale=ScaleClass.MOL,
    )


# ─────────────────────────────────────────────
# Test 1: Adapter Registry
# ─────────────────────────────────────────────

class TestAdapterRegistry:

    def test_global_registry_populated(self):
        assert len(ADAPTER_REGISTRY) >= 4  # CD, crown, porphyrin, lignin

    def test_builtins_registered(self):
        assert "cyclodextrin" in ADAPTER_REGISTRY
        assert "crown_ether" in ADAPTER_REGISTRY
        assert "porphyrin" in ADAPTER_REGISTRY
        assert "functionalized_lignin" in ADAPTER_REGISTRY

    def test_instantiate_all(self):
        adapters = ADAPTER_REGISTRY.instantiate_all()
        assert len(adapters) >= 4
        # Check registry has all expected keys
        assert "cyclodextrin" in ADAPTER_REGISTRY
        assert "functionalized_lignin" in ADAPTER_REGISTRY

    def test_fresh_registry(self):
        fresh = AdapterRegistry()
        assert len(fresh) == 0
        register_builtins(fresh)
        assert len(fresh) >= 4

    def test_instantiate_returns_same_instance(self):
        a1 = ADAPTER_REGISTRY.instantiate("cyclodextrin")
        a2 = ADAPTER_REGISTRY.instantiate("cyclodextrin")
        assert a1 is a2

    def test_instantiate_unknown_returns_none(self):
        assert ADAPTER_REGISTRY.instantiate("unobtainium") is None


# ─────────────────────────────────────────────
# Test 2: Weight Profiles
# ─────────────────────────────────────────────

class TestWeightProfiles:

    def test_four_profiles_exist(self):
        assert len(WEIGHT_PROFILES) == 4
        for name in ("research", "diagnostic", "separation", "remediation"):
            assert name in WEIGHT_PROFILES

    def test_weights_sum_to_1(self):
        for name, profile in WEIGHT_PROFILES.items():
            assert profile.total() == pytest.approx(1.0, abs=0.01), (
                f"{name} weights sum to {profile.total()}"
            )

    def test_remediation_weights_capacity_heavily(self):
        p = WEIGHT_PROFILES["remediation"]
        assert p.w_capacity > p.w_physics_fidelity

    def test_research_weights_fidelity_heavily(self):
        p = WEIGHT_PROFILES["research"]
        assert p.w_physics_fidelity > p.w_capacity

    def test_get_weight_profile_mapping(self):
        p = get_weight_profile(ApplicationContext.REMEDIATION)
        assert p.name == "remediation"
        p = get_weight_profile(ApplicationContext.RESEARCH)
        assert p.name == "research"


# ─────────────────────────────────────────────
# Test 3: Composite Scoring
# ─────────────────────────────────────────────

class TestCompositeScoring:

    def _make_precision_score(self) -> RealizationScore:
        return RealizationScore(
            material_system="crown_ether",
            adapter_id="CrownEtherAdapter",
            deviation_from_ideal=DeviationReport(
                "crown_ether", [], 0.1, 0.1,
            ),
            physics_fidelity=0.85,
            synthetic_accessibility=0.80,
            cost_score=0.70,
            scalability=0.50,
            operating_condition_compatibility=0.85,
            reusability_score=0.40,
            selectivity_factor=50.0,
            physics_class="covalent_cavity",
        )

    def _make_bulk_score(self) -> RealizationScore:
        return RealizationScore(
            material_system="functionalized_lignin",
            adapter_id="FunctionalizedLigninAdapter",
            deviation_from_ideal=DeviationReport(
                "functionalized_lignin", [], 1.0, 1.0,
            ),
            physics_fidelity=0.35,
            synthetic_accessibility=0.90,
            cost_score=0.90,
            scalability=0.95,
            operating_condition_compatibility=0.80,
            reusability_score=0.60,
            capacity_mmol_per_g=2.5,
            selectivity_factor=20.0,
            throughput_L_per_h_per_kg=50.0,
            regenerability_cycles=15,
            cost_per_kg_processed=8.0,
            physics_class="bulk_sorbent",
        )

    def test_research_prefers_precision(self):
        weights = WEIGHT_PROFILES["research"]
        precision = compute_composite(self._make_precision_score(), weights)
        bulk = compute_composite(self._make_bulk_score(), weights)
        assert precision > bulk

    def test_remediation_prefers_bulk(self):
        weights = WEIGHT_PROFILES["remediation"]
        precision = compute_composite(self._make_precision_score(), weights)
        bulk = compute_composite(self._make_bulk_score(), weights)
        assert bulk > precision

    def test_composite_bounded_0_1(self):
        weights = WEIGHT_PROFILES["research"]
        score = self._make_precision_score()
        c = compute_composite(score, weights)
        assert 0.0 <= c <= 1.0


# ─────────────────────────────────────────────
# Test 4: End-to-End Ranker
# ─────────────────────────────────────────────

class TestRankerEndToEnd:

    def setup_method(self):
        self.adapters = ADAPTER_REGISTRY.instantiate_all()
        self.ranker = AdapterRanker(self.adapters)

    def test_rank_returns_output(self):
        spec = make_potassium_spec()
        result = self.ranker.rank(spec)
        assert result.recommended_system != "none"
        assert len(result.all_rankings) >= 4

    def test_organic_guest_prefers_cd(self):
        """Organic guest + research → CD should rank high."""
        spec = make_organic_guest_spec()
        result = self.ranker.rank(spec, ApplicationContext.RESEARCH)
        # CD should be in top 2
        top_systems = [r.material_system for r in result.all_rankings[:2]]
        assert "cyclodextrin" in top_systems

    def test_pb2_remediation_prefers_lignin(self):
        """Pb²⁺ at remediation scale → lignin should win."""
        spec = make_pb2_remediation_spec()
        result = self.ranker.rank(spec, ApplicationContext.REMEDIATION)
        assert result.recommended_system == "functionalized_lignin"

    def test_pb2_research_prefers_precision(self):
        """Same Pb²⁺ spec but research → precision binder should win."""
        spec = make_pb2_remediation_spec()
        spec.target_application = ApplicationContext.RESEARCH
        result = self.ranker.rank(spec, ApplicationContext.RESEARCH)
        assert result.recommended_system != "functionalized_lignin"

    def test_groups_by_physics_class(self):
        spec = make_pb2_remediation_spec()
        result = self.ranker.rank(spec)
        class_names = [g.physics_class for g in result.groups]
        assert "bulk_sorbent" in class_names
        assert "covalent_cavity" in class_names

    def test_each_group_has_best(self):
        spec = make_pb2_remediation_spec()
        result = self.ranker.rank(spec)
        for group in result.groups:
            if group.rankings:
                assert group.best is not None

    def test_all_scores_have_composite(self):
        spec = make_potassium_spec()
        result = self.ranker.rank(spec)
        for score in result.all_rankings:
            assert score.composite_score > 0

    def test_rationale_present(self):
        spec = make_potassium_spec()
        result = self.ranker.rank(spec)
        assert len(result.recommendation_rationale) > 0


# ─────────────────────────────────────────────
# Test 5: Lignin HSAB Routing
# ─────────────────────────────────────────────

class TestLigninHSABRouting:

    def test_s_donors_select_soft_group(self):
        """S-donors in spec → dithiocarbamate or thiol."""
        spec = make_soft_metal_spec()
        group, rationale = select_functional_group(spec)
        assert group.hsab_affinity == "soft"
        assert group.donor_atom == "S"

    def test_o_donors_select_hard_group(self):
        """O-donors in spec → carboxylate or phosphonate."""
        spec = make_hard_metal_spec()
        group, rationale = select_functional_group(spec)
        assert group.hsab_affinity == "hard"
        assert group.donor_atom == "O"

    def test_n_donors_select_borderline(self):
        """N-donors → amine or IDA."""
        spec = make_cu2_4N_spec()
        group, rationale = select_functional_group(spec)
        assert group.hsab_affinity == "borderline"
        assert group.donor_atom == "N"


# ─────────────────────────────────────────────
# Test 6: Lignin Design Pipeline
# ─────────────────────────────────────────────

class TestLigninDesign:

    def setup_method(self):
        self.adapter = FunctionalizedLigninAdapter()

    def test_returns_lignin_fab_spec(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert isinstance(fab, LigninFabSpec)

    def test_soft_metal_gets_dtc(self):
        """Pb²⁺ with S-donors → DTC functionalization."""
        spec = make_soft_metal_spec()
        fab = self.adapter.design(spec)
        assert fab.functional_group_abbreviation in ("DTC", "SH")
        assert fab.donor_subtype == "S_thiolate"

    def test_hard_metal_gets_carboxylate(self):
        """Hard metal with O-donors → carboxylate."""
        spec = make_hard_metal_spec()
        fab = self.adapter.design(spec)
        assert fab.functional_group_abbreviation in ("COOH", "PO3H2")

    def test_has_synthesis_steps(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert len(fab.synthesis_steps) >= 4

    def test_has_grafting_chemistry(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert len(fab.grafting_reagent) > 0
        assert len(fab.grafting_conditions) > 0

    def test_has_crosslinker(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert len(fab.crosslinker) > 0

    def test_has_capacity_prediction(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert fab.predicted_qmax_mg_per_g > 0
        assert fab.predicted_qmax_mmol_per_g > 0

    def test_has_cost_prediction(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert fab.sorbent_cost_per_kg > 0
        assert fab.predicted_cost_per_kg_removed > 0

    def test_has_validation_plan(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert len(fab.validation_experiments) >= 3
        assert any("isotherm" in v for v in fab.validation_experiments)

    def test_backbone_is_kraft(self):
        """Cheapest backbone should be selected by default."""
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        assert "kraft" in fab.backbone.lower() or "raft" in fab.backbone.lower()

    def test_validation_passes(self):
        spec = make_pb2_remediation_spec()
        fab = self.adapter.design(spec)
        report = self.adapter.validate_design(fab)
        assert report.valid


# ─────────────────────────────────────────────
# Test 7: Application Context Switching
# ─────────────────────────────────────────────

class TestApplicationSwitching:
    """Same spec, different applications → different winners."""

    def setup_method(self):
        self.adapters = ADAPTER_REGISTRY.instantiate_all()
        self.ranker = AdapterRanker(self.adapters)

    def test_remediation_vs_research_different_winner(self):
        """Pb²⁺ spec: remediation → lignin, research → precision."""
        spec = make_pb2_remediation_spec()

        result_rem = self.ranker.rank(spec, ApplicationContext.REMEDIATION)
        result_res = self.ranker.rank(spec, ApplicationContext.RESEARCH)

        assert result_rem.recommended_system == "functionalized_lignin"
        assert result_res.recommended_system != "functionalized_lignin"

    def test_composite_changes_with_application(self):
        """Lignin's lead over precision changes with application context."""
        spec = make_pb2_remediation_spec()

        result_rem = self.ranker.rank(spec, ApplicationContext.REMEDIATION)
        result_res = self.ranker.rank(spec, ApplicationContext.RESEARCH)

        # Get lignin and best precision scores in each context
        lignin_rem = next(
            (r for r in result_rem.all_rankings
             if r.material_system == "functionalized_lignin"), None
        )
        lignin_res = next(
            (r for r in result_res.all_rankings
             if r.material_system == "functionalized_lignin"), None
        )
        best_precision_rem = next(
            (r for r in result_rem.all_rankings
             if r.physics_class == "covalent_cavity"), None
        )
        best_precision_res = next(
            (r for r in result_res.all_rankings
             if r.physics_class == "covalent_cavity"), None
        )

        assert lignin_rem is not None
        assert lignin_res is not None
        assert best_precision_rem is not None
        assert best_precision_res is not None

        # Lignin's lead over precision should be bigger in remediation
        gap_rem = lignin_rem.composite_score - best_precision_rem.composite_score
        gap_res = lignin_res.composite_score - best_precision_res.composite_score
        assert gap_rem > gap_res


# ─────────────────────────────────────────────
# Test 8: Functional Group Knowledge Base
# ─────────────────────────────────────────────

class TestFunctionalGroupKnowledge:

    def test_six_groups(self):
        assert len(FUNCTIONAL_GROUPS) >= 6

    def test_groups_cover_hsab(self):
        hsab_classes = {g.hsab_affinity for g in FUNCTIONAL_GROUPS.values()}
        assert "hard" in hsab_classes
        assert "borderline" in hsab_classes
        assert "soft" in hsab_classes

    def test_dtc_is_soft(self):
        assert FUNCTIONAL_GROUPS["dithiocarbamate"].hsab_affinity == "soft"

    def test_carboxylate_is_hard(self):
        assert FUNCTIONAL_GROUPS["carboxylate"].hsab_affinity == "hard"

    def test_amine_is_borderline(self):
        assert FUNCTIONAL_GROUPS["amine"].hsab_affinity == "borderline"

    def test_three_backbones(self):
        assert len(LIGNIN_BACKBONES) >= 3

    def test_kraft_is_practical_default(self):
        """Kraft lignin is the default backbone (best balance of cost + properties)."""
        kraft = LIGNIN_BACKBONES["kraft"]
        assert kraft.phenolic_OH_mmol_per_g > 3.0  # good grafting potential
        assert kraft.cost_per_kg_usd < 1.00  # cheap
