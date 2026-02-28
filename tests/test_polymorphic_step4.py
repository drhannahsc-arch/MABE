"""
Tests for Step 4: SurfaceInteractionSpec + Activated Carbon Adapter.

Verifies:
    - SurfaceInteractionSpec inherits from InteractionSpec
    - spec_type == "surface"
    - No pocket or network fields leak in
    - AC knowledge data integrity
    - AC adapter scoring via isotherm physics
    - AC adapter produces valid fabrication specs
    - Competitive adsorption ordering matches Xiao & Thomas 2004
    - Calibration status correctly reports "uncalibrated"
    - Isotherm math is correct
"""

import pytest
import math

from mabe.realization.models import (
    InteractionSpec,
    InteractionParadigm,
    DiscretePocketSpec,
    NetworkInteractionSpec,
    SurfaceInteractionSpec,
    SurfaceMechanism,
    BaseMaterial,
    IsothermModel,
    ApplicationContext,
    ScaleClass,
    Solvent,
)
from mabe.realization.adapters.ac_knowledge import (
    DATA_SOURCES,
    METAL_ADSORPTION_ORDER_OXIDIZED_AC,
    AC_PROFILES,
    PORE_CLASSIFICATION,
    langmuir_qe,
    freundlich_qe,
    langmuir_separation_factor,
    ph_adsorption_factor,
    recommend_ac_type,
)
from mabe.realization.adapters.ac_adapter import (
    ActivatedCarbonAdapter,
    ACFabricationSpec,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def pb_surface_spec():
    """Pb²⁺ removal from water using surface adsorption."""
    return SurfaceInteractionSpec(
        target_species="Pb2+",
        target_charge=2,
        target_mw_g_mol=207.2,
        competing_species=["Ca2+", "Cd2+"],
        competing_concentrations_mM=[20.0, 1.0],
        mechanism=SurfaceMechanism.CHEMISORPTION,
        isotherm_model=IsothermModel.LANGMUIR,
        initial_concentration_mg_L=100.0,
        target_removal_efficiency=0.95,
        pH_range=(4.0, 7.0),
        target_application=ApplicationContext.REMEDIATION,
    )


@pytest.fixture
def cd_surface_spec():
    """Cd²⁺ removal — weaker adsorption than Pb²⁺ per Xiao & Thomas."""
    return SurfaceInteractionSpec(
        target_species="Cd2+",
        target_charge=2,
        target_mw_g_mol=112.4,
        competing_species=["Pb2+"],
        mechanism=SurfaceMechanism.SURFACE_COMPLEXATION,
        initial_concentration_mg_L=50.0,
        pH_range=(5.0, 8.0),
    )


@pytest.fixture
def adapter():
    return ActivatedCarbonAdapter()


# ─────────────────────────────────────────────
# Type hierarchy
# ─────────────────────────────────────────────

class TestSurfaceSpecHierarchy:

    def test_is_interaction_spec(self, pb_surface_spec):
        assert isinstance(pb_surface_spec, InteractionSpec)

    def test_is_not_discrete_pocket(self, pb_surface_spec):
        assert not isinstance(pb_surface_spec, DiscretePocketSpec)

    def test_is_not_network(self, pb_surface_spec):
        assert not isinstance(pb_surface_spec, NetworkInteractionSpec)

    def test_spec_type_is_surface(self, pb_surface_spec):
        assert pb_surface_spec.spec_type == "surface"
        assert pb_surface_spec.spec_type == InteractionParadigm.SURFACE.value

    def test_has_no_cavity_fields(self, pb_surface_spec):
        assert not hasattr(pb_surface_spec, "cavity_shape")
        assert not hasattr(pb_surface_spec, "donor_positions")

    def test_has_no_network_fields(self, pb_surface_spec):
        assert not hasattr(pb_surface_spec, "crosslink_pct")
        assert not hasattr(pb_surface_spec, "resin_type")

    def test_has_surface_fields(self, pb_surface_spec):
        assert hasattr(pb_surface_spec, "mechanism")
        assert hasattr(pb_surface_spec, "isotherm_model")
        assert hasattr(pb_surface_spec, "base_material")
        assert hasattr(pb_surface_spec, "min_surface_area_m2_g")
        assert hasattr(pb_surface_spec, "target_capacity_mg_g")

    def test_three_paradigms_coexist(self, pb_surface_spec):
        """All three spec types are distinct and all are InteractionSpec."""
        from mabe.realization.models import CavityShape, CavityDimensions
        pocket = DiscretePocketSpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        network = NetworkInteractionSpec(
            target_species="Pb2+", target_charge=2,
        )
        surface = pb_surface_spec

        assert pocket.spec_type == "pocket"
        assert network.spec_type == "network"
        assert surface.spec_type == "surface"

        for s in [pocket, network, surface]:
            assert isinstance(s, InteractionSpec)

        # All distinct types
        assert type(pocket) is not type(network)
        assert type(network) is not type(surface)
        assert type(pocket) is not type(surface)


# ─────────────────────────────────────────────
# Isotherm math
# ─────────────────────────────────────────────

class TestIsothermMath:

    def test_langmuir_at_zero(self):
        assert langmuir_qe(0.0, 100.0, 0.1) == 0.0

    def test_langmuir_at_saturation(self):
        """At very high Ce, qe → qmax."""
        qe = langmuir_qe(1e6, 100.0, 0.1)
        assert qe == pytest.approx(100.0, rel=0.01)

    def test_langmuir_midpoint(self):
        """At Ce = 1/KL, qe = qmax/2."""
        KL = 0.1
        Ce = 1.0 / KL  # = 10 mg/L
        qe = langmuir_qe(Ce, 100.0, KL)
        assert qe == pytest.approx(50.0, rel=0.01)

    def test_langmuir_negative_inputs(self):
        assert langmuir_qe(-1.0, 100.0, 0.1) == 0.0
        assert langmuir_qe(10.0, -100.0, 0.1) == 0.0
        assert langmuir_qe(10.0, 100.0, -0.1) == 0.0

    def test_freundlich_at_zero(self):
        assert freundlich_qe(0.0, 10.0, 2.0) == 0.0

    def test_freundlich_basic(self):
        """KF=10, n=2, Ce=4 → qe = 10 * 4^0.5 = 20."""
        qe = freundlich_qe(4.0, 10.0, 2.0)
        assert qe == pytest.approx(20.0, rel=0.01)

    def test_freundlich_negative_inputs(self):
        assert freundlich_qe(-1.0, 10.0, 2.0) == 0.0
        assert freundlich_qe(4.0, -10.0, 2.0) == 0.0

    def test_separation_factor_favorable(self):
        """RL between 0 and 1 is favorable."""
        RL = langmuir_separation_factor(100.0, 0.1)
        assert 0.0 < RL < 1.0

    def test_separation_factor_approaches_zero_at_high_C0(self):
        RL = langmuir_separation_factor(1e6, 0.1)
        assert RL < 0.01

    def test_separation_factor_approaches_one_at_low_C0(self):
        RL = langmuir_separation_factor(0.001, 0.1)
        assert RL > 0.99


# ─────────────────────────────────────────────
# pH adsorption factor
# ─────────────────────────────────────────────

class TestpHFactor:

    def test_cation_above_pzc_favorable(self):
        """Cation at pH > pHpzc → favorable (surface negative)."""
        factor = ph_adsorption_factor(pH=8.0, ph_pzc=3.0, target_charge=2)
        assert factor > 0.7

    def test_cation_below_pzc_unfavorable(self):
        """Cation at pH << pHpzc → unfavorable (surface positive)."""
        factor = ph_adsorption_factor(pH=1.0, ph_pzc=7.0, target_charge=2)
        assert factor < 0.3

    def test_anion_below_pzc_favorable(self):
        factor = ph_adsorption_factor(pH=3.0, ph_pzc=7.0, target_charge=-1)
        assert factor > 0.7

    def test_neutral_moderate(self):
        factor = ph_adsorption_factor(pH=7.0, ph_pzc=7.0, target_charge=0)
        assert 0.5 < factor < 0.9


# ─────────────────────────────────────────────
# AC Knowledge data integrity
# ─────────────────────────────────────────────

class TestACKnowledge:

    def test_data_sources_documented(self):
        assert len(DATA_SOURCES) >= 2
        dois = [s.get("doi", "") for s in DATA_SOURCES]
        assert "10.1021/la049712j" in dois

    def test_competitive_ordering_has_4_metals(self):
        assert len(METAL_ADSORPTION_ORDER_OXIDIZED_AC) == 4

    def test_hg_strongest(self):
        """Hg2+ has highest rank per Xiao & Thomas 2004."""
        hg = [e for e in METAL_ADSORPTION_ORDER_OXIDIZED_AC if e[0] == "Hg2+"][0]
        assert hg[1] == 4

    def test_ordering_hg_pb_cd_ca(self):
        """Verified: Hg2+ > Pb2+ > Cd2+ > Ca2+."""
        order = {e[0]: e[1] for e in METAL_ADSORPTION_ORDER_OXIDIZED_AC}
        assert order["Hg2+"] > order["Pb2+"] > order["Cd2+"] > order["Ca2+"]

    def test_four_ac_profiles(self):
        assert len(AC_PROFILES) == 4
        assert "coconut_shell_gac" in AC_PROFILES
        assert "coal_gac" in AC_PROFILES
        assert "wood_gac" in AC_PROFILES
        assert "oxidized_gac" in AC_PROFILES

    def test_oxidized_gac_lowest_pzc(self):
        """Oxidized AC has lowest pHpzc (most acidic surface)."""
        ox = AC_PROFILES["oxidized_gac"]
        for key, profile in AC_PROFILES.items():
            if key != "oxidized_gac":
                assert ox.ph_pzc <= profile.ph_pzc

    def test_oxidized_gac_highest_capacity(self):
        """Oxidized AC has highest metal capacity range."""
        ox = AC_PROFILES["oxidized_gac"]
        for key, profile in AC_PROFILES.items():
            if key != "oxidized_gac":
                assert ox.typical_capacity_heavy_metals_mg_g[1] >= \
                       profile.typical_capacity_heavy_metals_mg_g[1]

    def test_iupac_pore_classification(self):
        assert PORE_CLASSIFICATION["micropore"]["diameter_A"] == (0, 20)
        assert PORE_CLASSIFICATION["mesopore"]["diameter_A"] == (20, 500)

    def test_recommend_metals_get_oxidized(self):
        assert recommend_ac_type("Pb2+", 2, 207.2) == "oxidized_gac"

    def test_recommend_small_organics_get_coconut(self):
        assert recommend_ac_type("phenol", 0, 94.1) == "coconut_shell_gac"

    def test_recommend_large_organics_get_wood(self):
        assert recommend_ac_type("humic_acid", 0, 2000.0) == "wood_gac"


# ─────────────────────────────────────────────
# AC Adapter: scoring
# ─────────────────────────────────────────────

class TestACAdapterScoring:

    def test_pb_feasible(self, adapter, pb_surface_spec):
        result = adapter.score(pb_surface_spec)
        assert result["feasible"] is True

    def test_pb_selects_oxidized(self, adapter, pb_surface_spec):
        result = adapter.score(pb_surface_spec)
        assert result["ac_type"] == "oxidized_gac"

    def test_pb_has_capacity_range(self, adapter, pb_surface_spec):
        result = adapter.score(pb_surface_spec)
        low, high = result["estimated_capacity_range_mg_g"]
        assert low > 0
        assert high > low

    def test_pb_outcompetes_cd(self, adapter, pb_surface_spec):
        """Pb2+ rank > Cd2+ rank per Xiao & Thomas ordering."""
        result = adapter.score(pb_surface_spec)
        assert result["competition_rank"] == 3  # Pb is rank 3
        assert result["competition_score"] == 1.0  # outcompetes Ca2+ and Cd2+

    def test_cd_outcompeted_by_pb(self, adapter, cd_surface_spec):
        """Cd2+ loses to Pb2+ in competition."""
        result = adapter.score(cd_surface_spec)
        assert result["competition_rank"] == 2
        assert result["competition_score"] < 0.5  # outcompeted by Pb2+

    def test_composite_bounded(self, adapter, pb_surface_spec):
        result = adapter.score(pb_surface_spec)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_calibration_status_uncalibrated(self, adapter, pb_surface_spec):
        result = adapter.score(pb_surface_spec)
        assert result["calibration_status"] == "uncalibrated"

    def test_calibration_notes_present(self, adapter, pb_surface_spec):
        result = adapter.score(pb_surface_spec)
        notes = result["calibration_notes"]
        assert len(notes) >= 1
        assert any("isotherm" in n.lower() for n in notes)

    def test_confidence_lower_than_ix(self, adapter, pb_surface_spec):
        """AC confidence should be lower than IX (0.85) because material-specific."""
        result = adapter.score(pb_surface_spec)
        assert result["confidence"] < 0.85


# ─────────────────────────────────────────────
# AC Adapter: design
# ─────────────────────────────────────────────

class TestACAdapterDesign:

    def test_returns_ac_fab_spec(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert isinstance(fab, ACFabricationSpec)

    def test_material_system(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert fab.material_system == "activated_carbon"

    def test_no_pocket_geometry(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert fab.predicted_pocket_geometry is None

    def test_has_ac_type(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert fab.ac_type == "oxidized_gac"

    def test_has_capacity_range(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        low, high = fab.estimated_capacity_range_mg_g
        assert high > low > 0

    def test_has_recommended_dose(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert fab.recommended_dose_g_L > 0

    def test_has_recommended_ph(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert 2.0 <= fab.recommended_pH <= 12.0

    def test_has_synthesis_steps(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert len(fab.synthesis_steps) >= 4

    def test_synthesis_includes_isotherm(self, adapter, pb_surface_spec):
        """Design MUST include isotherm characterization step."""
        fab = adapter.design(pb_surface_spec)
        steps_text = " ".join(fab.synthesis_steps).lower()
        assert "isotherm" in steps_text

    def test_has_validation(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert len(fab.validation_experiments) >= 3

    def test_validation_includes_ph_edge(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        val_text = " ".join(fab.validation_experiments).lower()
        assert "ph" in val_text

    def test_has_data_provenance(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert len(fab.data_sources) >= 1

    def test_calibration_uncalibrated(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert fab.calibration_status == "uncalibrated"

    def test_cost_positive(self, adapter, pb_surface_spec):
        fab = adapter.design(pb_surface_spec)
        assert fab.estimated_cost_per_unit > 0.0


# ─────────────────────────────────────────────
# Backward compat
# ─────────────────────────────────────────────

class TestBackwardCompat:

    def test_all_existing_tests_structure(self):
        """Verify imports from models still work."""
        from mabe.realization.models import (
            InteractionGeometrySpec,
            InteractionSpec,
            NetworkInteractionSpec,
            SurfaceInteractionSpec,
            IdealPocketSpec,
            RealizationScore,
        )
        assert InteractionGeometrySpec is DiscretePocketSpec
        assert issubclass(NetworkInteractionSpec, InteractionSpec)
        assert issubclass(SurfaceInteractionSpec, InteractionSpec)