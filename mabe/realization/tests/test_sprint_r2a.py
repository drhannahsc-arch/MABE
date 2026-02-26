"""
Sprint R2a Test Suite — Cyclodextrin Adapter.

Tests:
    - Knowledge base integrity (CD properties, modifications)
    - Packing coefficient selection (α/β/γ by guest volume)
    - Full design pipeline (spec → CyclodextrinFabSpec)
    - Modification logic (solubility fix, charge, click handle)
    - Binding prediction (uses BackSolve params)
    - Validation (PC bounds, internal consistency)
"""

import math
import pytest

from mabe.realization.adapters.cyclodextrin_adapter import (
    CyclodextrinAdapter,
    CyclodextrinFabSpec,
)
from mabe.realization.adapters.cyclodextrin_knowledge import (
    ALPHA_CD,
    ALL_CDS,
    BACKSOLVE_CD_PARAMS,
    BETA_CD,
    CD_HOSTS,
    CD_MODIFICATIONS,
    GAMMA_CD,
    HP_BETA_CD,
    NATIVE_CDS,
    select_best_cd,
)
from mabe.realization.models import (
    ApplicationContext,
    CavityDimensions,
    CavityShape,
    DonorPosition,
    InteractionGeometrySpec,
    ScaleClass,
    Solvent,
)


# ─────────────────────────────────────────────
# Spec fixtures
# ─────────────────────────────────────────────

def make_beta_cd_guest_spec() -> InteractionGeometrySpec:
    """
    Guest that fits β-CD: ~155 Å³ volume.
    PC = 155/262 = 0.59 — right in the Rebek sweet spot.
    Think: adamantane, p-nitrophenol, ibuprofen-sized.
    """
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=155.0,
            aperture_A=6.0,
            depth_A=7.0,
            max_internal_diameter_A=6.5,
        ),
        symmetry="none",
        donor_positions=[],
        pocket_scale_nm=0.6,
        pH_range=(4.0, 9.0),
        temperature_range_K=(288.0, 310.0),
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_alpha_cd_guest_spec() -> InteractionGeometrySpec:
    """Small guest that fits α-CD: ~100 Å³ (PC = 100/174 = 0.57)."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=100.0,
            aperture_A=4.5,
            depth_A=6.0,
            max_internal_diameter_A=4.5,
        ),
        symmetry="none",
        pocket_scale_nm=0.47,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_gamma_cd_guest_spec() -> InteractionGeometrySpec:
    """Large guest that fits γ-CD: ~250 Å³ (PC = 250/427 = 0.59)."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=250.0,
            aperture_A=7.0,
            depth_A=7.5,
            max_internal_diameter_A=7.5,
        ),
        symmetry="none",
        pocket_scale_nm=0.75,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_diagnostic_spec() -> InteractionGeometrySpec:
    """β-CD guest with diagnostic application → should get click handle."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=155.0,
            aperture_A=6.0,
            depth_A=7.0,
            max_internal_diameter_A=6.5,
        ),
        symmetry="none",
        pocket_scale_nm=0.6,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.DIAGNOSTIC,
        required_scale=ScaleClass.UMOL,
    )


def make_cationic_guest_spec() -> InteractionGeometrySpec:
    """Cationic guest → should trigger SBE modification."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=155.0,
            aperture_A=6.0,
            depth_A=7.0,
            max_internal_diameter_A=6.5,
        ),
        symmetry="none",
        donor_positions=[
            DonorPosition(
                atom_type="N",
                coordination_role="h_bond_donor",
                position_vector_A=(0.0, 3.0, 0.0),
                tolerance_A=0.5,
                required_hybridization="sp3",
                charge_state=1.0,
            ),
        ],
        pocket_scale_nm=0.6,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_too_large_spec() -> InteractionGeometrySpec:
    """Guest too large for any CD (800 Å³, PC > 1.0 for all)."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.SPHERE,
        cavity_dimensions=CavityDimensions(
            volume_A3=800.0,
            aperture_A=12.0,
            depth_A=10.0,
            max_internal_diameter_A=12.0,
        ),
        symmetry="none",
        pocket_scale_nm=1.2,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


# ─────────────────────────────────────────────
# Test 1: Knowledge Base
# ─────────────────────────────────────────────

class TestKnowledgeBase:

    def test_three_native_cds(self):
        assert len(NATIVE_CDS) == 3

    def test_six_total_cds(self):
        assert len(ALL_CDS) == 6

    def test_alpha_smallest_cavity(self):
        assert ALPHA_CD.cavity_volume_A3 < BETA_CD.cavity_volume_A3 < GAMMA_CD.cavity_volume_A3

    def test_cavity_diameters_increase(self):
        assert ALPHA_CD.cavity_diameter_A < BETA_CD.cavity_diameter_A < GAMMA_CD.cavity_diameter_A

    def test_beta_cd_low_solubility(self):
        """β-CD notoriously has low water solubility."""
        assert BETA_CD.water_solubility_mM < 20

    def test_hp_beta_high_solubility(self):
        """HP-β-CD solves the solubility problem."""
        assert HP_BETA_CD.water_solubility_mM > 400

    def test_all_cds_commercial(self):
        for cd in ALL_CDS:
            assert cd.commercial

    def test_backsolve_params_present(self):
        assert "gamma_hydrophobic" in BACKSOLVE_CD_PARAMS
        assert "PC_optimal" in BACKSOLVE_CD_PARAMS
        assert "k_shape" in BACKSOLVE_CD_PARAMS

    def test_pc_optimal_near_rebek(self):
        """Rebek's 55% rule: optimal PC ≈ 0.55."""
        assert 0.50 < BACKSOLVE_CD_PARAMS["PC_optimal"] < 0.65

    def test_modifications_library(self):
        assert len(CD_MODIFICATIONS) >= 5
        abbrevs = {m.abbreviation for m in CD_MODIFICATIONS}
        assert "HP" in abbrevs
        assert "Me" in abbrevs
        assert "N3" in abbrevs


# ─────────────────────────────────────────────
# Test 2: Packing Coefficient Selection
# ─────────────────────────────────────────────

class TestPackingSelection:

    def test_beta_volume_selects_beta(self):
        """155 Å³ guest → β-CD (PC ≈ 0.59)."""
        results = select_best_cd(155.0)
        best_cd, pc, _ = results[0]
        assert best_cd.base_type == "beta"
        assert 0.45 < pc < 0.70

    def test_small_volume_selects_alpha(self):
        """100 Å³ guest → α-CD (PC ≈ 0.57)."""
        results = select_best_cd(100.0)
        best_cd, pc, _ = results[0]
        assert best_cd.base_type == "alpha"
        assert 0.45 < pc < 0.70

    def test_large_volume_selects_gamma(self):
        """250 Å³ guest → γ-CD (PC ≈ 0.59)."""
        results = select_best_cd(250.0)
        best_cd, pc, _ = results[0]
        assert best_cd.base_type == "gamma"

    def test_results_sorted_by_pc_optimality(self):
        """Results should be sorted by closeness to optimal PC."""
        results = select_best_cd(155.0)
        deviations = [
            abs(pc - BACKSOLVE_CD_PARAMS["PC_optimal"])
            for _, pc, _ in results
        ]
        assert deviations == sorted(deviations)

    def test_water_soluble_filter(self):
        """With require_water_soluble, native β-CD excluded."""
        results_all = select_best_cd(155.0, require_water_soluble=False)
        results_soluble = select_best_cd(155.0, require_water_soluble=True)
        # Native β-CD (16 mM) should be excluded from soluble list
        soluble_names = {cd.name for cd, _, _ in results_soluble}
        assert "β-Cyclodextrin" not in soluble_names


# ─────────────────────────────────────────────
# Test 3: Adapter estimate_fidelity
# ─────────────────────────────────────────────

class TestEstimateFidelity:

    def setup_method(self):
        self.adapter = CyclodextrinAdapter()

    def test_good_guest_high_fidelity(self):
        spec = make_beta_cd_guest_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert score.physics_fidelity > 0.5
        assert score.feasible

    def test_too_large_guest_low_fidelity(self):
        spec = make_too_large_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert score.physics_fidelity < 0.3

    def test_organic_solvent_penalized(self):
        spec = make_beta_cd_guest_spec()
        spec.solvent = Solvent.ORGANIC
        score = self.adapter.estimate_fidelity(spec)
        assert score.physics_fidelity < 0.2

    def test_fidelity_bounded(self):
        spec = make_beta_cd_guest_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert 0.0 <= score.physics_fidelity <= 1.0

    def test_advantages_populated(self):
        spec = make_beta_cd_guest_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert len(score.advantages) > 0


# ─────────────────────────────────────────────
# Test 4: Full Design Pipeline
# ─────────────────────────────────────────────

class TestDesign:

    def setup_method(self):
        self.adapter = CyclodextrinAdapter()

    def test_design_returns_cd_fab_spec(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert isinstance(fab, CyclodextrinFabSpec)

    def test_design_selects_beta_for_155A3(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert fab.base_type == "beta"

    def test_design_selects_alpha_for_100A3(self):
        spec = make_alpha_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert fab.base_type == "alpha"

    def test_design_selects_gamma_for_250A3(self):
        spec = make_gamma_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert fab.base_type == "gamma"

    def test_design_has_packing_coefficient(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert 0.3 < fab.packing_coefficient < 0.8

    def test_design_has_synthesis_steps(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert len(fab.synthesis_steps) > 0

    def test_design_has_validation_plan(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert len(fab.validation_experiments) >= 2
        assert any("ITC" in v for v in fab.validation_experiments)

    def test_design_has_supplier(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert len(fab.supplier) > 0

    def test_design_predicts_binding(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert fab.predicted_dG_kJ_mol != 0.0
        assert fab.predicted_logK != 0.0

    def test_good_packing_negative_dG(self):
        """Guest in Rebek sweet spot should have favorable (negative) ΔG."""
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert fab.predicted_dG_kJ_mol < 0


# ─────────────────────────────────────────────
# Test 5: Modification Logic
# ─────────────────────────────────────────────

class TestModifications:

    def setup_method(self):
        self.adapter = CyclodextrinAdapter()

    def test_beta_cd_gets_hp_modification(self):
        """Native β-CD selected → should auto-recommend HP for solubility."""
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert "HP" in fab.modifications

    def test_hp_rationale_mentions_solubility(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert any("solubility" in r.lower() for r in fab.modification_rationale)

    def test_diagnostic_gets_click_handle(self):
        spec = make_diagnostic_spec()
        fab = self.adapter.design(spec)
        assert fab.click_handle == "C6-azide"
        assert "N3" in fab.modifications

    def test_research_no_click_handle(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert fab.click_handle == "none"

    def test_cationic_guest_gets_sbe(self):
        spec = make_cationic_guest_spec()
        fab = self.adapter.design(spec)
        assert "SBE" in fab.modifications

    def test_alpha_cd_no_hp(self):
        """α-CD has good solubility natively — no HP needed."""
        spec = make_alpha_cd_guest_spec()
        fab = self.adapter.design(spec)
        assert "HP" not in fab.modifications


# ─────────────────────────────────────────────
# Test 6: Validation
# ─────────────────────────────────────────────

class TestValidation:

    def setup_method(self):
        self.adapter = CyclodextrinAdapter()

    def test_good_design_validates(self):
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        report = self.adapter.validate_design(fab)
        assert report.valid

    def test_extreme_pc_fails_validation(self):
        """Manually set PC to extreme value → should flag issue."""
        spec = make_beta_cd_guest_spec()
        fab = self.adapter.design(spec)
        fab.packing_coefficient = 0.95
        report = self.adapter.validate_design(fab)
        assert not report.valid
        assert any("too high" in i for i in report.issues)

    def test_wrong_type_fails(self):
        from mabe.realization.models import FabricationSpec, CavityDimensions
        wrong = FabricationSpec(
            material_system="not_cd",
            geometry_spec_hash="x",
            predicted_pocket_geometry=CavityDimensions(0, 0, 0, 0),
            predicted_deviation_from_ideal_A=0,
        )
        report = self.adapter.validate_design(wrong)
        assert not report.valid
