"""
Tests for Step 1: Polymorphic InteractionSpec hierarchy.

Verifies:
    - InteractionSpec is the base class
    - DiscretePocketSpec inherits from InteractionSpec
    - InteractionGeometrySpec is an alias for DiscretePocketSpec
    - spec_type property works
    - InteractionParadigm enum is correct
    - isinstance() checks work across the hierarchy
    - All existing construction patterns still work
    - No field changes — pure reparenting
"""

import pytest

from mabe.realization.models import (
    # New types
    InteractionSpec,
    InteractionParadigm,
    DiscretePocketSpec,
    # Backward-compat alias
    InteractionGeometrySpec,
    # Sub-components (unchanged)
    CavityShape,
    CavityDimensions,
    DonorPosition,
    ExclusionSpec,
    HydrophobicSurface,
    HBondSpec,
    Solvent,
    ApplicationContext,
    ScaleClass,
    # Downstream types (unchanged)
    IdealPocketSpec,
    DeviationReport,
    RealizationScore,
    RankedRealizations,
    FabricationSpec,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def minimal_pocket():
    """Minimal valid DiscretePocketSpec."""
    return DiscretePocketSpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=100.0,
            aperture_A=5.0,
            depth_A=5.0,
            max_internal_diameter_A=6.0,
        ),
    )


@pytest.fixture
def full_pocket():
    """Fully-specified DiscretePocketSpec with all fields."""
    return DiscretePocketSpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=33.5,
            aperture_A=4.0,
            depth_A=2.0,
            max_internal_diameter_A=4.0,
        ),
        symmetry="D4h",
        donor_positions=[
            DonorPosition("N", "equatorial", (2.0, 0.0, 0.0), 0.05, "sp2"),
            DonorPosition("N", "equatorial", (0.0, 2.0, 0.0), 0.05, "sp2"),
            DonorPosition("N", "equatorial", (-2.0, 0.0, 0.0), 0.05, "sp2"),
            DonorPosition("N", "equatorial", (0.0, -2.0, 0.0), 0.05, "sp2"),
        ],
        rigidity_requirement="rigid",
        max_backbone_rmsd_A=0.1,
        pocket_scale_nm=0.4,
        must_exclude=[
            ExclusionSpec("Zn2+", -10.0, "geometry"),
        ],
        pH_range=(2.0, 12.0),
        temperature_range_K=(250.0, 500.0),
        solvent=Solvent.AQUEOUS,
        ionic_strength_M=0.1,
        target_application=ApplicationContext.REMEDIATION,
        required_scale=ScaleClass.MOL,
        cost_ceiling_per_unit=100.0,
        reusability_required=True,
    )


# ─────────────────────────────────────────────
# Type hierarchy tests
# ─────────────────────────────────────────────

class TestTypeHierarchy:
    """Verify the inheritance chain is correct."""

    def test_discrete_pocket_is_interaction_spec(self, minimal_pocket):
        assert isinstance(minimal_pocket, InteractionSpec)

    def test_interaction_geometry_spec_is_alias(self):
        assert InteractionGeometrySpec is DiscretePocketSpec

    def test_alias_instance_is_interaction_spec(self):
        spec = InteractionGeometrySpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        assert isinstance(spec, InteractionSpec)
        assert isinstance(spec, DiscretePocketSpec)
        assert isinstance(spec, InteractionGeometrySpec)

    def test_interaction_spec_is_not_instantiable_alone(self):
        """Base class exists but spec_type raises NotImplementedError."""
        base = InteractionSpec()
        with pytest.raises(NotImplementedError):
            _ = base.spec_type

    def test_discrete_pocket_spec_type(self, minimal_pocket):
        assert minimal_pocket.spec_type == "pocket"

    def test_spec_type_matches_paradigm_enum(self, minimal_pocket):
        assert minimal_pocket.spec_type == InteractionParadigm.POCKET.value


# ─────────────────────────────────────────────
# InteractionParadigm enum tests
# ─────────────────────────────────────────────

class TestInteractionParadigm:
    """Verify the paradigm enum covers planned subtypes."""

    def test_six_paradigms(self):
        assert len(InteractionParadigm) == 6

    def test_pocket_exists(self):
        assert InteractionParadigm.POCKET.value == "pocket"

    def test_network_exists(self):
        assert InteractionParadigm.NETWORK.value == "network"

    def test_surface_exists(self):
        assert InteractionParadigm.SURFACE.value == "surface"

    def test_bulk_exists(self):
        assert InteractionParadigm.BULK.value == "bulk"

    def test_field_exists(self):
        assert InteractionParadigm.FIELD.value == "field"

    def test_composite_exists(self):
        assert InteractionParadigm.COMPOSITE.value == "composite"

    def test_paradigm_is_str(self):
        """Can be used directly in string comparisons."""
        assert InteractionParadigm.POCKET == "pocket"


# ─────────────────────────────────────────────
# Backward compatibility tests
# ─────────────────────────────────────────────

class TestBackwardCompat:
    """Every existing construction pattern must still work."""

    def test_construct_via_old_name(self):
        spec = InteractionGeometrySpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        assert spec.cavity_shape == CavityShape.SPHERE

    def test_construct_via_new_name(self):
        spec = DiscretePocketSpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        assert spec.cavity_shape == CavityShape.SPHERE

    def test_both_names_produce_same_type(self):
        old = InteractionGeometrySpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        new = DiscretePocketSpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        assert type(old) is type(new)

    def test_required_donor_types_property(self, full_pocket):
        assert full_pocket.required_donor_types == {"N"}

    def test_tightest_tolerance_property(self, full_pocket):
        assert full_pocket.tightest_tolerance_A == 0.05

    def test_tightest_tolerance_empty_donors(self, minimal_pocket):
        assert minimal_pocket.tightest_tolerance_A == float("inf")

    def test_all_defaults_populated(self, minimal_pocket):
        """Minimal construction fills all defaults."""
        assert minimal_pocket.symmetry == "none"
        assert minimal_pocket.rigidity_requirement == "semi-rigid"
        assert minimal_pocket.max_backbone_rmsd_A == 1.0
        assert minimal_pocket.pocket_scale_nm == 0.5
        assert minimal_pocket.multivalency == 1
        assert minimal_pocket.pH_range == (5.0, 9.0)
        assert minimal_pocket.solvent == Solvent.AQUEOUS
        assert minimal_pocket.ionic_strength_M == 0.1
        assert minimal_pocket.target_application == ApplicationContext.RESEARCH
        assert minimal_pocket.required_scale == ScaleClass.UMOL
        assert minimal_pocket.cost_ceiling_per_unit is None
        assert minimal_pocket.reusability_required is False

    def test_all_enums_unchanged(self):
        """Existing enums still importable and correct."""
        assert CavityShape.SPHERE.value == "sphere"
        assert Solvent.AQUEOUS.value == "aqueous"
        assert ApplicationContext.REMEDIATION.value == "remediation"
        assert ScaleClass.KMOL.rank == 4

    def test_downstream_types_unchanged(self):
        """IdealPocketSpec, DeviationReport, etc. still importable."""
        assert IdealPocketSpec is not None
        assert DeviationReport is not None
        assert RealizationScore is not None
        assert RankedRealizations is not None
        assert FabricationSpec is not None


# ─────────────────────────────────────────────
# Full field round-trip
# ─────────────────────────────────────────────

class TestFieldIntegrity:
    """Verify every field on DiscretePocketSpec is accessible."""

    def test_all_pocket_fields_accessible(self, full_pocket):
        assert full_pocket.cavity_shape == CavityShape.FLAT
        assert full_pocket.cavity_dimensions.volume_A3 == 33.5
        assert full_pocket.symmetry == "D4h"
        assert len(full_pocket.donor_positions) == 4
        assert full_pocket.rigidity_requirement == "rigid"
        assert full_pocket.max_backbone_rmsd_A == 0.1
        assert full_pocket.pocket_scale_nm == 0.4
        assert len(full_pocket.must_exclude) == 1
        assert full_pocket.pH_range == (2.0, 12.0)
        assert full_pocket.temperature_range_K == (250.0, 500.0)
        assert full_pocket.solvent == Solvent.AQUEOUS
        assert full_pocket.target_application == ApplicationContext.REMEDIATION
        assert full_pocket.required_scale == ScaleClass.MOL
        assert full_pocket.cost_ceiling_per_unit == 100.0
        assert full_pocket.reusability_required is True

    def test_spec_type_always_pocket(self, full_pocket):
        assert full_pocket.spec_type == "pocket"


# ─────────────────────────────────────────────
# Future extensibility smoke test
# ─────────────────────────────────────────────

class TestExtensibility:
    """Verify that new subtypes can be created without breaking anything."""

    def test_custom_subtype_inherits_spec_type_requirement(self):
        """A subtype that doesn't define spec_type raises on access."""

        class BrokenSpec(InteractionSpec):
            pass

        broken = BrokenSpec()
        with pytest.raises(NotImplementedError):
            _ = broken.spec_type

    def test_custom_subtype_with_spec_type(self):
        """A subtype that defines spec_type works."""

        class StubNetworkSpec(InteractionSpec):
            @property
            def spec_type(self) -> str:
                return "network"

        stub = StubNetworkSpec()
        assert stub.spec_type == "network"
        assert isinstance(stub, InteractionSpec)

    def test_polymorphic_dispatch_sketch(self, minimal_pocket):
        """isinstance checks enable paradigm routing."""

        def route(spec: InteractionSpec) -> str:
            if isinstance(spec, DiscretePocketSpec):
                return "pocket_pipeline"
            return "unknown"

        assert route(minimal_pocket) == "pocket_pipeline"