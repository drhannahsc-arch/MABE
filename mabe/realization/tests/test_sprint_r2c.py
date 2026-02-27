"""
Sprint R2c Test Suite — Porphyrin Adapter.

Tests:
    - Knowledge base integrity (cores, metals, substituents, axial ligands)
    - Metal identification from spec
    - Core selection (size-match physics)
    - Meso-substituent selection (Hammett, application-driven)
    - Axial ligand selection (coordination number driven)
    - Full design pipeline (spec → PorphyrinFabSpec)
    - Validation checks
"""

import math
import pytest

from mabe.realization.adapters.porphyrin_adapter import (
    PorphyrinAdapter,
    PorphyrinFabSpec,
)
from mabe.realization.adapters.porphyrin_knowledge import (
    ALL_CORES,
    AXIAL_LIGANDS,
    BACKSOLVE_METAL_PARAMS,
    MESO_SUBSTITUENTS,
    METAL_PORPH_DB,
    OEP,
    PC,
    TPFPP,
    TPP,
    metal_porphyrin_size_match,
    predict_metalation_dG,
    select_core_for_metal,
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
# Spec fixtures — 4N planar pockets
# ─────────────────────────────────────────────

def _make_4N_donor_positions(bond_A: float = 2.00) -> list[DonorPosition]:
    """4 N donors in square planar arrangement."""
    return [
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(bond_A, 0.0, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(0.0, bond_A, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(-bond_A, 0.0, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
        DonorPosition(
            atom_type="N", coordination_role="equatorial",
            position_vector_A=(0.0, -bond_A, 0.0),
            tolerance_A=0.05, required_hybridization="sp2",
        ),
    ]


def make_cu2_spec() -> InteractionGeometrySpec:
    """Cu²⁺: 4-coordinate, in-plane, M-N ~1.98 Å → diameter ~3.96."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=33.0,
            aperture_A=3.96,
            depth_A=3.4,
            max_internal_diameter_A=3.96,
        ),
        symmetry="D4h",
        donor_positions=_make_4N_donor_positions(1.98),
        pocket_scale_nm=0.40,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_fe3_spec() -> InteractionGeometrySpec:
    """Fe³⁺: 6-coordinate, needs axial ligands, M-N ~2.04 Å."""
    donors = _make_4N_donor_positions(2.04)
    # Add 2 axial positions
    donors.append(DonorPosition(
        atom_type="N", coordination_role="axial",
        position_vector_A=(0.0, 0.0, 2.10),
        tolerance_A=0.1, required_hybridization="sp2",
    ))
    donors.append(DonorPosition(
        atom_type="N", coordination_role="axial",
        position_vector_A=(0.0, 0.0, -2.10),
        tolerance_A=0.1, required_hybridization="sp2",
    ))
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=35.0,
            aperture_A=4.08,
            depth_A=6.0,
            max_internal_diameter_A=4.08,
        ),
        symmetry="D4h",
        donor_positions=donors,
        pocket_scale_nm=0.40,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_zn2_spec() -> InteractionGeometrySpec:
    """Zn²⁺: 5-coordinate, one axial, M-N ~2.04 Å."""
    donors = _make_4N_donor_positions(2.04)
    donors.append(DonorPosition(
        atom_type="N", coordination_role="axial",
        position_vector_A=(0.0, 0.0, 2.15),
        tolerance_A=0.1, required_hybridization="sp2",
    ))
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=35.0,
            aperture_A=4.08,
            depth_A=5.0,
            max_internal_diameter_A=4.08,
        ),
        symmetry="C4v",
        donor_positions=donors,
        pocket_scale_nm=0.40,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_ni2_spec() -> InteractionGeometrySpec:
    """Ni²⁺: strictly 4-coordinate, no axial, M-N ~1.93 Å."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=30.0,
            aperture_A=3.86,
            depth_A=3.4,
            max_internal_diameter_A=3.86,
        ),
        symmetry="D4h",
        donor_positions=_make_4N_donor_positions(1.93),
        pocket_scale_nm=0.39,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.RESEARCH,
        required_scale=ScaleClass.UMOL,
    )


def make_diagnostic_cu2_spec() -> InteractionGeometrySpec:
    """Cu²⁺ for diagnostic → should get COOH conjugation handle."""
    spec = make_cu2_spec()
    spec.target_application = ApplicationContext.DIAGNOSTIC
    return spec


def make_pb2_spec() -> InteractionGeometrySpec:
    """Pb²⁺: very large (0.98 Å), out-of-plane → stress test."""
    return InteractionGeometrySpec(
        cavity_shape=CavityShape.FLAT,
        cavity_dimensions=CavityDimensions(
            volume_A3=50.0,
            aperture_A=4.68,
            depth_A=4.0,
            max_internal_diameter_A=4.68,
        ),
        symmetry="C4v",
        donor_positions=_make_4N_donor_positions(2.34),
        pocket_scale_nm=0.47,
        solvent=Solvent.AQUEOUS,
        target_application=ApplicationContext.REMEDIATION,
        required_scale=ScaleClass.MMOL,
    )


# ─────────────────────────────────────────────
# Test 1: Knowledge Base
# ─────────────────────────────────────────────

class TestKnowledgeBase:

    def test_six_cores(self):
        assert len(ALL_CORES) == 6

    def test_twelve_metals(self):
        assert len(METAL_PORPH_DB) >= 12

    def test_seven_substituents(self):
        assert len(MESO_SUBSTITUENTS) >= 7

    def test_eight_axial_ligands(self):
        assert len(AXIAL_LIGANDS) >= 8

    def test_tpp_is_D4h(self):
        assert TPP.cavity_symmetry == "D4h"

    def test_all_cores_have_4_pyrrole_N(self):
        for core in ALL_CORES.values():
            assert core.n_pyrrole_N == 4

    def test_core_hole_around_2A(self):
        for core in ALL_CORES.values():
            assert 1.8 < core.core_hole_radius_A < 2.2

    def test_pc_smaller_than_tpp(self):
        """Phthalocyanine has smaller cavity than porphyrin."""
        assert PC.core_hole_radius_A < TPP.core_hole_radius_A

    def test_backsolve_params(self):
        assert "N_pyrrole_exchange_kJ_mol" in BACKSOLVE_METAL_PARAMS
        assert "lfse_scale" in BACKSOLVE_METAL_PARAMS

    def test_cu2_is_labile(self):
        assert METAL_PORPH_DB["Cu2+"].kinetic_class == "labile"

    def test_fe3_is_inert(self):
        assert METAL_PORPH_DB["Fe3+"].kinetic_class == "inert"

    def test_ni2_no_axial(self):
        assert METAL_PORPH_DB["Ni2+"].max_axial_count == 0

    def test_fe3_wants_6_coord(self):
        assert METAL_PORPH_DB["Fe3+"].preferred_coordination == 6

    def test_pb2_out_of_plane(self):
        assert not METAL_PORPH_DB["Pb2+"].in_plane
        assert METAL_PORPH_DB["Pb2+"].displacement_A > 0.5


# ─────────────────────────────────────────────
# Test 2: Size-Match Physics
# ─────────────────────────────────────────────

class TestSizeMatch:

    def test_cu2_good_match(self):
        """Cu²⁺ (0.57 Å) is a classic porphyrin metal."""
        sm = metal_porphyrin_size_match(0.57, TPP.core_hole_radius_A)
        assert sm > 0.5

    def test_pb2_poor_match(self):
        """Pb²⁺ (0.98 Å) is much too large."""
        sm = metal_porphyrin_size_match(0.98, TPP.core_hole_radius_A)
        assert sm < 0.3

    def test_core_selection_returns_ranked(self):
        results = select_core_for_metal("Cu2+")
        assert len(results) == len(ALL_CORES)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_core_selection_unknown_metal(self):
        results = select_core_for_metal("Unobtainium99+")
        assert results == []


# ─────────────────────────────────────────────
# Test 3: Metalation Prediction
# ─────────────────────────────────────────────

class TestMetalationPrediction:

    def test_dG_negative_for_cu2(self):
        """Cu²⁺ metalation should be favorable (ΔG < 0)."""
        metal = METAL_PORPH_DB["Cu2+"]
        dG = predict_metalation_dG(metal, TPP)
        assert dG < 0

    def test_ewg_raises_dG(self):
        """Electron-withdrawing meso → less favorable metalation."""
        metal = METAL_PORPH_DB["Cu2+"]
        dG_neutral = predict_metalation_dG(metal, TPP, meso_sigma=0.0)
        dG_ewg = predict_metalation_dG(metal, TPP, meso_sigma=1.0)
        # Hammett rho is negative → positive sigma makes dG less negative
        assert dG_ewg > dG_neutral

    def test_axial_stabilizes(self):
        """Adding axial ligands should make ΔG more negative."""
        metal = METAL_PORPH_DB["Fe3+"]
        dG_no_ax = predict_metalation_dG(metal, TPP, n_axial=0)
        dG_ax = predict_metalation_dG(metal, TPP, n_axial=2)
        assert dG_ax < dG_no_ax


# ─────────────────────────────────────────────
# Test 4: Adapter estimate_fidelity
# ─────────────────────────────────────────────

class TestEstimateFidelity:

    def setup_method(self):
        self.adapter = PorphyrinAdapter()

    def test_cu2_high_fidelity(self):
        spec = make_cu2_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert score.physics_fidelity > 0.5
        assert score.feasible

    def test_fidelity_bounded(self):
        spec = make_cu2_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert 0.0 <= score.physics_fidelity <= 1.0

    def test_advantages_present(self):
        spec = make_cu2_spec()
        score = self.adapter.estimate_fidelity(spec)
        assert len(score.advantages) > 0


# ─────────────────────────────────────────────
# Test 5: Full Design Pipeline
# ─────────────────────────────────────────────

class TestDesign:

    def setup_method(self):
        self.adapter = PorphyrinAdapter()

    def test_returns_porphyrin_fab_spec(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert isinstance(fab, PorphyrinFabSpec)

    def test_cu2_gets_porphyrin_core(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert fab.core_type in ("porphyrin", "phthalocyanine")

    def test_cu2_is_4_coordinate(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert fab.metal_coordination == 4

    def test_fe3_gets_axial_ligands(self):
        """Fe³⁺ prefers 6-coord → should get axial ligands."""
        spec = make_fe3_spec()
        fab = self.adapter.design(spec)
        assert len(fab.axial_ligands) >= 1

    def test_zn2_gets_one_axial(self):
        """Zn²⁺ prefers 5-coord → should get 1 axial."""
        spec = make_zn2_spec()
        fab = self.adapter.design(spec)
        assert len(fab.axial_ligands) == 1

    def test_ni2_no_axial(self):
        """Ni²⁺ is strictly 4-coord → no axial."""
        spec = make_ni2_spec()
        fab = self.adapter.design(spec)
        assert len(fab.axial_ligands) == 0

    def test_has_synthesis_steps(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert len(fab.synthesis_steps) >= 3

    def test_has_metalation_protocol(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert len(fab.metalation_protocol) > 0

    def test_has_predicted_dG(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert fab.predicted_dG_metalation_kJ_mol < 0

    def test_has_logK(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert fab.predicted_logK_metalation > 0

    def test_has_validation_plan(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        assert len(fab.validation_experiments) >= 2


# ─────────────────────────────────────────────
# Test 6: Substituent Selection
# ─────────────────────────────────────────────

class TestSubstituents:

    def setup_method(self):
        self.adapter = PorphyrinAdapter()

    def test_diagnostic_gets_cooh(self):
        """Diagnostic application → COOH conjugation handle."""
        spec = make_diagnostic_cu2_spec()
        fab = self.adapter.design(spec)
        assert "COOH" in fab.meso_substituent or "4-COOH" in fab.meso_substituent

    def test_ni2_d8_gets_electron_poor(self):
        """Ni²⁺ (d8) → electron-poor meso (C6F5)."""
        spec = make_ni2_spec()
        fab = self.adapter.design(spec)
        assert fab.meso_substituent in ("C6F5", "Ph")  # C6F5 preferred for d8+

    def test_fe3_defaults_to_standard(self):
        """Fe³⁺ (d5) → standard phenyl (d5 triggers neither EDG nor EWG path)."""
        spec = make_fe3_spec()
        fab = self.adapter.design(spec)
        # Fe3+ d5 HS: neither d<=4 nor d>=8 → default Ph
        assert fab.meso_substituent == "Ph"


# ─────────────────────────────────────────────
# Test 7: Edge Cases
# ─────────────────────────────────────────────

class TestEdgeCases:

    def setup_method(self):
        self.adapter = PorphyrinAdapter()

    def test_pb2_out_of_plane(self):
        """Pb²⁺ is very large — should flag out-of-plane."""
        spec = make_pb2_spec()
        fab = self.adapter.design(spec)
        assert not fab.in_plane

    def test_pb2_validation_warns(self):
        spec = make_pb2_spec()
        fab = self.adapter.design(spec)
        report = self.adapter.validate_design(fab)
        assert len(report.warnings) > 0


# ─────────────────────────────────────────────
# Test 8: Validation
# ─────────────────────────────────────────────

class TestValidation:

    def setup_method(self):
        self.adapter = PorphyrinAdapter()

    def test_good_design_validates(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        report = self.adapter.validate_design(fab)
        assert report.valid

    def test_bad_size_match_flags(self):
        spec = make_cu2_spec()
        fab = self.adapter.design(spec)
        fab.size_match_score = 0.1
        report = self.adapter.validate_design(fab)
        assert not report.valid

    def test_wrong_type_fails(self):
        from mabe.realization.models import FabricationSpec, CavityDimensions
        wrong = FabricationSpec(
            material_system="not_porph",
            geometry_spec_hash="x",
            predicted_pocket_geometry=CavityDimensions(0, 0, 0, 0),
            predicted_deviation_from_ideal_A=0,
        )
        report = self.adapter.validate_design(wrong)
        assert not report.valid

    def test_too_many_axials_flagged(self):
        """Ni²⁺ with axial ligands → should flag issue."""
        spec = make_ni2_spec()
        fab = self.adapter.design(spec)
        fab.axial_ligands = ["Im", "Py"]  # Ni2+ max is 0
        report = self.adapter.validate_design(fab)
        assert not report.valid
