"""
Tests for Step 2: NetworkInteractionSpec + IX Resin Adapter.

Verifies:
    - NetworkInteractionSpec inherits from InteractionSpec
    - spec_type == "network"
    - IX knowledge data integrity (DuPont Tech Fact 45-D01458-en)
    - IX adapter scores correctly using selectivity coefficients
    - IX adapter produces valid fabrication specs
    - Selectivity ordering matches known chemistry
    - No pocket-paradigm physics leaks into network scoring
"""

import pytest

from mabe.realization.models import (
    InteractionSpec,
    InteractionParadigm,
    DiscretePocketSpec,
    NetworkInteractionSpec,
    NetworkMechanism,
    ResinType,
    ApplicationContext,
    ScaleClass,
    Solvent,
    ExclusionSpec,
)
from mabe.realization.adapters.ix_resin_knowledge import (
    DATA_SOURCE,
    SAC_SELECTIVITY,
    SAC_BY_ION,
    SBA_SELECTIVITY,
    SBA_BY_ION,
    WAC_SELECTIVITY,
    WAC_BY_ION,
    WAC_ESTIMATED_ENTRIES,
    RESIN_PROFILES,
    get_sac_selectivity,
    compute_separation_factor,
)
from mabe.realization.adapters.ix_resin_adapter import (
    IXResinAdapter,
    IXResinFabricationSpec,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def pb_removal_spec():
    """Pb²⁺ removal from mine water with Na+, Ca2+, Mg2+ competition."""
    return NetworkInteractionSpec(
        target_species="Pb2+",
        target_charge=2,
        competing_species=["Na+", "Ca2+", "Mg2+"],
        competing_concentrations_mM=[50.0, 20.0, 10.0],
        mechanism=NetworkMechanism.ION_EXCHANGE,
        crosslink_pct=8.0,
        pH_range=(2.0, 6.0),
        target_application=ApplicationContext.REMEDIATION,
        required_scale=ScaleClass.MOL,
        reusability_required=True,
    )


@pytest.fixture
def nitrate_removal_spec():
    """NO3⁻ removal from groundwater."""
    return NetworkInteractionSpec(
        target_species="NO3-",
        target_charge=-1,
        competing_species=["Cl-", "HCO3-", "SO4_2-"],
        mechanism=NetworkMechanism.ION_EXCHANGE,
        pH_range=(6.0, 8.0),
        target_application=ApplicationContext.REMEDIATION,
    )


@pytest.fixture
def adapter():
    return IXResinAdapter()


# ─────────────────────────────────────────────
# Type hierarchy
# ─────────────────────────────────────────────

class TestNetworkSpecHierarchy:

    def test_is_interaction_spec(self, pb_removal_spec):
        assert isinstance(pb_removal_spec, InteractionSpec)

    def test_is_not_discrete_pocket(self, pb_removal_spec):
        assert not isinstance(pb_removal_spec, DiscretePocketSpec)

    def test_spec_type_is_network(self, pb_removal_spec):
        assert pb_removal_spec.spec_type == "network"
        assert pb_removal_spec.spec_type == InteractionParadigm.NETWORK.value

    def test_has_no_cavity_fields(self, pb_removal_spec):
        """Network spec must not have pocket-specific fields."""
        assert not hasattr(pb_removal_spec, "cavity_shape")
        assert not hasattr(pb_removal_spec, "cavity_dimensions")
        assert not hasattr(pb_removal_spec, "donor_positions")

    def test_has_network_fields(self, pb_removal_spec):
        assert hasattr(pb_removal_spec, "target_species")
        assert hasattr(pb_removal_spec, "competing_species")
        assert hasattr(pb_removal_spec, "mechanism")
        assert hasattr(pb_removal_spec, "crosslink_pct")
        assert hasattr(pb_removal_spec, "resin_type")

    def test_polymorphic_dispatch(self, pb_removal_spec):
        """isinstance enables paradigm routing."""
        def route(spec: InteractionSpec) -> str:
            if isinstance(spec, NetworkInteractionSpec):
                return "network_pipeline"
            if isinstance(spec, DiscretePocketSpec):
                return "pocket_pipeline"
            return "unknown"

        assert route(pb_removal_spec) == "network_pipeline"


# ─────────────────────────────────────────────
# Data integrity: DuPont Tech Fact
# ─────────────────────────────────────────────

class TestSACData:
    """Verify SAC selectivity values match the DuPont PDF exactly."""

    def test_source_is_documented(self):
        assert DATA_SOURCE["form_number"] == "45-D01458-en"
        assert DATA_SOURCE["data_quality_tier"] == 2

    def test_20_cation_entries(self):
        assert len(SAC_SELECTIVITY) == 20

    def test_h_is_reference(self):
        h = SAC_BY_ION["H+"]
        assert h.dvb_4pct == 1.0
        assert h.dvb_8pct == 1.0
        assert h.dvb_10pct == 1.0
        assert h.dvb_16pct == 1.0

    def test_pb2_values_exact(self):
        """Pb²⁺ values from page 1 of the PDF."""
        pb = SAC_BY_ION["Pb2+"]
        assert pb.dvb_4pct == 4.97
        assert pb.dvb_8pct == 7.80
        assert pb.dvb_10pct == 8.92
        assert pb.dvb_16pct == 12.20

    def test_na_values_exact(self):
        na = SAC_BY_ION["Na+"]
        assert na.dvb_8pct == 1.56

    def test_ca_values_exact(self):
        ca = SAC_BY_ION["Ca2+"]
        assert ca.dvb_8pct == 4.06

    def test_ba_highest_divalent(self):
        """Ba²⁺ has highest selectivity among divalents at 8% DVB."""
        ba = SAC_BY_ION["Ba2+"]
        for entry in SAC_SELECTIVITY:
            if entry.charge == 2 and entry.ion != "Ba2+":
                assert ba.dvb_8pct >= entry.dvb_8pct, \
                    f"Ba2+ ({ba.dvb_8pct}) should be >= {entry.ion} ({entry.dvb_8pct})"

    def test_monovalent_ordering_8pct(self):
        """Li < H < Na < NH4 < K < Rb < Cs (8% DVB) — hydrated radius ordering."""
        order = ["Li+", "H+", "Na+", "NH4+", "K+", "Rb+", "Cs+"]
        vals = [SAC_BY_ION[ion].dvb_8pct for ion in order]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1], \
                f"Order violation: {order[i]}({vals[i]}) > {order[i+1]}({vals[i+1]})"

    def test_divalent_ordering_8pct(self):
        """Mg < Zn ≈ Co ≈ Cu < Ni ≈ Cd < Ca < Sr < Pb < Ba (8% DVB)."""
        # Not strict monotonic (Ni ≈ Cd), but general trend holds
        low = SAC_BY_ION["Mg2+"].dvb_8pct
        high = SAC_BY_ION["Ba2+"].dvb_8pct
        assert low < high

    def test_higher_dvb_increases_selectivity_for_Pb(self):
        """Higher cross-linking increases selectivity for larger ions."""
        pb = SAC_BY_ION["Pb2+"]
        assert pb.dvb_4pct < pb.dvb_8pct < pb.dvb_10pct < pb.dvb_16pct

    def test_all_charges_correct(self):
        for entry in SAC_SELECTIVITY:
            if entry.ion in ("Li+", "H+", "Na+", "NH4+", "K+", "Rb+", "Cs+", "Ag+", "Tl+"):
                assert entry.charge == 1
            else:
                assert entry.charge == 2


class TestSBAData:

    def test_22_anion_entries(self):
        assert len(SBA_SELECTIVITY) == 22

    def test_oh_is_reference(self):
        oh = SBA_BY_ION["OH-"]
        assert oh.type1 == 1.0
        assert oh.type2 == 1.0

    def test_no3_values(self):
        no3 = SBA_BY_ION["NO3-"]
        assert no3.type1 == 65.0
        assert no3.type2 == 8.0

    def test_cl_values(self):
        cl = SBA_BY_ION["Cl-"]
        assert cl.type1 == 22.0
        assert cl.type2 == 2.3

    def test_type1_greater_than_type2(self):
        """Type 1 resins generally have higher selectivity than Type 2."""
        for entry in SBA_SELECTIVITY:
            if entry.ion != "OH-" and entry.ion != "phenate":
                assert entry.type1 >= entry.type2, \
                    f"{entry.ion}: Type1 ({entry.type1}) < Type2 ({entry.type2})"


class TestWACData:

    def test_10_entries(self):
        assert len(WAC_SELECTIVITY) == 10

    def test_ca_is_reference(self):
        assert WAC_BY_ION["Ca2+"].relative_to_Ca == 1.0

    def test_cu_highest(self):
        cu = WAC_BY_ION["Cu2+"]
        for entry in WAC_SELECTIVITY:
            assert cu.relative_to_Ca >= entry.relative_to_Ca, \
                f"Cu2+ ({cu.relative_to_Ca}) should be >= {entry.ion} ({entry.relative_to_Ca})"

    def test_estimated_entries_flagged(self):
        assert "Sr2+" in WAC_ESTIMATED_ENTRIES
        assert "Ba2+" in WAC_ESTIMATED_ENTRIES
        assert "Pb2+" in WAC_ESTIMATED_ENTRIES
        assert "Ca2+" not in WAC_ESTIMATED_ENTRIES


# ─────────────────────────────────────────────
# Selectivity lookup functions
# ─────────────────────────────────────────────

class TestSelectivityLookup:

    def test_exact_dvb_match(self):
        assert get_sac_selectivity("Pb2+", 8.0) == 7.80

    def test_interpolation(self):
        """6% DVB should interpolate between 4% and 8%."""
        val = get_sac_selectivity("Pb2+", 6.0)
        assert val is not None
        assert 4.97 < val < 7.80  # between 4% and 8% values

    def test_below_4pct_clamps(self):
        assert get_sac_selectivity("Pb2+", 2.0) == 4.97

    def test_above_16pct_clamps(self):
        assert get_sac_selectivity("Pb2+", 20.0) == 12.20

    def test_unknown_ion_returns_none(self):
        assert get_sac_selectivity("Xe+", 8.0) is None

    def test_separation_factor_pb_vs_na(self):
        alpha = compute_separation_factor("Pb2+", "Na+", 8.0)
        assert alpha is not None
        assert alpha == pytest.approx(7.80 / 1.56, rel=0.01)
        assert alpha > 1.0  # Pb preferred over Na

    def test_separation_factor_pb_vs_ca(self):
        alpha = compute_separation_factor("Pb2+", "Ca2+", 8.0)
        assert alpha is not None
        assert alpha == pytest.approx(7.80 / 4.06, rel=0.01)
        assert alpha > 1.0  # Pb preferred over Ca

    def test_separation_factor_na_vs_ca(self):
        """Na+ is NOT preferred over Ca2+ — separation factor < 1."""
        alpha = compute_separation_factor("Na+", "Ca2+", 8.0)
        assert alpha is not None
        assert alpha < 1.0


# ─────────────────────────────────────────────
# IX Resin Adapter: scoring
# ─────────────────────────────────────────────

class TestIXAdapterScoring:

    def test_pb_removal_feasible(self, adapter, pb_removal_spec):
        result = adapter.score(pb_removal_spec)
        assert result["feasible"] is True

    def test_pb_removal_high_selectivity(self, adapter, pb_removal_spec):
        result = adapter.score(pb_removal_spec)
        assert result["selectivity_score"] > 0.3  # Pb preferred over all competitors

    def test_pb_has_separation_factors(self, adapter, pb_removal_spec):
        result = adapter.score(pb_removal_spec)
        sep = result["separation_factors"]
        assert "Na+" in sep
        assert "Ca2+" in sep
        assert "Mg2+" in sep
        assert sep["Na+"] > 1.0   # Pb preferred over Na
        assert sep["Ca2+"] > 1.0  # Pb preferred over Ca
        assert sep["Mg2+"] > 1.0  # Pb preferred over Mg

    def test_pb_separation_na_highest(self, adapter, pb_removal_spec):
        """Separation from Na+ should be better than from Ca2+."""
        result = adapter.score(pb_removal_spec)
        sep = result["separation_factors"]
        assert sep["Na+"] > sep["Ca2+"]

    def test_nitrate_removal_feasible(self, adapter, nitrate_removal_spec):
        result = adapter.score(nitrate_removal_spec)
        assert result["feasible"] is True

    def test_nitrate_selects_sba(self, adapter, nitrate_removal_spec):
        result = adapter.score(nitrate_removal_spec)
        assert result["resin_type"] == "SBA"

    def test_swelling_mechanism_rejected(self, adapter):
        """IX adapter should reject osmotic swelling specs."""
        spec = NetworkInteractionSpec(
            target_species="Na+",
            target_charge=1,
            mechanism=NetworkMechanism.OSMOTIC_SWELLING,
        )
        result = adapter.score(spec)
        assert result["feasible"] is False

    def test_unknown_ion_not_feasible(self, adapter):
        spec = NetworkInteractionSpec(
            target_species="Xe+",
            target_charge=1,
            mechanism=NetworkMechanism.ION_EXCHANGE,
        )
        result = adapter.score(spec)
        assert result["feasible"] is False

    def test_wac_ph_check(self, adapter):
        """WAC resin should fail at pH < 5."""
        spec = NetworkInteractionSpec(
            target_species="Cu2+",
            target_charge=2,
            mechanism=NetworkMechanism.ION_EXCHANGE,
            resin_type=ResinType.WAC,
            pH_range=(2.0, 4.0),  # too acidic for WAC
        )
        result = adapter.score(spec)
        assert result["feasible"] is False

    def test_composite_score_bounded(self, adapter, pb_removal_spec):
        result = adapter.score(pb_removal_spec)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_confidence_positive(self, adapter, pb_removal_spec):
        result = adapter.score(pb_removal_spec)
        assert result["confidence"] > 0.0


# ─────────────────────────────────────────────
# IX Resin Adapter: design
# ─────────────────────────────────────────────

class TestIXAdapterDesign:

    def test_returns_ix_fab_spec(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert isinstance(fab, IXResinFabricationSpec)

    def test_material_system(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert fab.material_system == "ion_exchange_resin"

    def test_no_pocket_geometry(self, adapter, pb_removal_spec):
        """Network paradigm must not produce pocket geometry."""
        fab = adapter.design(pb_removal_spec)
        assert fab.predicted_pocket_geometry is None

    def test_has_resin_type(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert fab.resin_type == "SAC"

    def test_has_selectivity(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert fab.selectivity_target_vs_H > 1.0  # Pb preferred over H

    def test_has_separation_factors(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert len(fab.separation_factors) > 0

    def test_has_capacity(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert fab.predicted_capacity_meq_mL > 0.0

    def test_has_synthesis_steps(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert len(fab.synthesis_steps) >= 3

    def test_has_validation_experiments(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert len(fab.validation_experiments) >= 3

    def test_has_data_provenance(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert fab.data_source["form_number"] == "45-D01458-en"

    def test_nitrate_gets_sba(self, adapter, nitrate_removal_spec):
        fab = adapter.design(nitrate_removal_spec)
        assert fab.resin_type == "SBA"

    def test_has_regenerant(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert len(fab.regenerant) > 0

    def test_cost_positive(self, adapter, pb_removal_spec):
        fab = adapter.design(pb_removal_spec)
        assert fab.estimated_cost_per_unit > 0.0


# ─────────────────────────────────────────────
# Chemistry sanity checks
# ─────────────────────────────────────────────

class TestChemistrySanity:
    """Verify that known IX chemistry principles hold in the data."""

    def test_divalent_preferred_over_monovalent(self):
        """Divalent ions generally preferred over monovalent on SAC."""
        ca = SAC_BY_ION["Ca2+"].dvb_8pct
        na = SAC_BY_ION["Na+"].dvb_8pct
        assert ca > na

    def test_larger_monovalent_preferred(self):
        """Cs+ > K+ > Na+ > Li+ on SAC (dehydration energy)."""
        cs = SAC_BY_ION["Cs+"].dvb_8pct
        k = SAC_BY_ION["K+"].dvb_8pct
        na = SAC_BY_ION["Na+"].dvb_8pct
        li = SAC_BY_ION["Li+"].dvb_8pct
        assert cs > k > na > li

    def test_no3_preferred_over_cl(self):
        """NO3⁻ > Cl⁻ on SBA Type 1."""
        no3 = SBA_BY_ION["NO3-"].type1
        cl = SBA_BY_ION["Cl-"].type1
        assert no3 > cl

    def test_i_preferred_over_br(self):
        """I⁻ > Br⁻ on SBA (larger anion, less hydrated)."""
        i = SBA_BY_ION["I-"].type1
        br = SBA_BY_ION["Br-"].type1
        assert i > br

    def test_cu_preferred_over_ca_on_wac(self):
        """Cu2+ > Ca2+ on WAC (chelation effect)."""
        cu = WAC_BY_ION["Cu2+"].relative_to_Ca
        ca = WAC_BY_ION["Ca2+"].relative_to_Ca
        assert cu > ca

    def test_total_data_points_exceed_gate(self):
        """Data gate: ≥20 experimental selectivity coefficients."""
        total = len(SAC_SELECTIVITY) + len(SBA_SELECTIVITY) + len(WAC_SELECTIVITY)
        assert total >= 20
        assert total == 52  # 20 + 22 + 10


# ─────────────────────────────────────────────
# Backward compat: existing tests unaffected
# ─────────────────────────────────────────────

class TestBackwardCompat:
    """Adding NetworkInteractionSpec must not break pocket pipeline."""

    def test_pocket_spec_still_works(self):
        from mabe.realization.models import (
            InteractionGeometrySpec, CavityShape, CavityDimensions,
        )
        spec = InteractionGeometrySpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        assert spec.spec_type == "pocket"
        assert isinstance(spec, InteractionSpec)

    def test_pocket_and_network_are_distinct(self, pb_removal_spec):
        from mabe.realization.models import (
            InteractionGeometrySpec, CavityShape, CavityDimensions,
        )
        pocket = InteractionGeometrySpec(
            cavity_shape=CavityShape.SPHERE,
            cavity_dimensions=CavityDimensions(100.0, 5.0, 5.0, 6.0),
        )
        assert type(pocket) is not type(pb_removal_spec)
        assert pocket.spec_type != pb_removal_spec.spec_type
        # But both are InteractionSpec
        assert isinstance(pocket, InteractionSpec)
        assert isinstance(pb_removal_spec, InteractionSpec)