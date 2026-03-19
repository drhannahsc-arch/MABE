"""
tests/test_capture_transform_scorer.py — Tests for capture-transform scoring module.

Validates:
  - Energy decomposition (ΔG_bind + ΔG_transform = ΔG_total)
  - Rate classification from activation barriers
  - Co-reactant availability assessment
  - Product accumulation modeling
  - Regeneration assessment
  - Cascade benefit estimation
  - Composite scoring and ranking
  - End-to-end pipeline (enumerate → score)
  - Gold/Silver/Bronze tier classification
  - Critical risk detection
"""

import sys
import os
import pytest
import math

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.capture_transform_scorer import (
    score_capture_transform,
    score_all_products,
    evaluate_capture_transform,
    scoring_report,
    CaptureTransformScore,
    CoReactantAvailability,
    ProductAccumulationModel,
    RegenerationAssessment,
    CascadeBenefit,
    _classify_rate,
    _assess_co_reactant,
    _estimate_accumulation,
    _assess_regeneration,
    _assess_cascade,
    _compute_composite,
)
from core.transform_enumerator import (
    enumerate_transformations,
    TransformationProduct,
    CoReactantSource,
    CoReactantSpec,
    EnergyInput,
    TurnoverMode,
    ProductPhase,
    CaptureSiteChemistry,
    ClickCompatibility,
)


# ═══════════════════════════════════════════════════════════════════════════
# Rate classification
# ═══════════════════════════════════════════════════════════════════════════

class TestRateClassification:

    def test_low_barrier_is_fast(self):
        assert _classify_rate(20.0) == "fast"

    def test_moderate_barrier(self):
        assert _classify_rate(50.0) == "moderate"

    def test_high_barrier_is_slow(self):
        assert _classify_rate(70.0) == "slow"

    def test_very_high_barrier(self):
        assert _classify_rate(100.0) == "very_slow"

    def test_zero_barrier_is_fast(self):
        assert _classify_rate(0.0) == "fast"

    def test_none_barrier_is_fast(self):
        assert _classify_rate(None) == "fast"


# ═══════════════════════════════════════════════════════════════════════════
# Co-reactant availability
# ═══════════════════════════════════════════════════════════════════════════

class TestCoReactantAvailability:

    def test_no_co_reactant_not_limiting(self):
        a = _assess_co_reactant("site", CoReactantSource.NONE, 0.0)
        assert a.limiting is False
        assert a.rate_factor == 1.0

    def test_matrix_native_excess(self):
        a = _assess_co_reactant("Ca2+", CoReactantSource.MATRIX_NATIVE, 10.0, 0.1)
        assert a.limiting is False
        assert a.rate_factor == 1.0

    def test_matrix_native_substoichiometric(self):
        a = _assess_co_reactant("Ca2+", CoReactantSource.MATRIX_NATIVE, 0.01, 0.1)
        assert a.limiting is True
        assert a.rate_factor < 1.0

    def test_matrix_native_absent(self):
        a = _assess_co_reactant("Ca2+", CoReactantSource.MATRIX_NATIVE, 0.0, 0.1)
        assert a.limiting is True

    def test_externally_supplied_always_limiting(self):
        a = _assess_co_reactant("H2", CoReactantSource.EXTERNALLY_SUPPLIED, 0.0)
        assert a.limiting is True
        assert a.rate_factor == 0.0

    def test_solar_has_reduced_rate(self):
        a = _assess_co_reactant("hv", CoReactantSource.SOLAR_PHOTOCATALYTIC, 0.0)
        assert a.limiting is False
        assert a.rate_factor < 1.0

    def test_preloaded_not_detected_partial_credit(self):
        a = _assess_co_reactant("Mg2+", CoReactantSource.SUBSTRATE_PRELOADED, 0.0, 0.1)
        assert a.limiting is False  # preloaded is OK even without matrix detection
        assert 0.0 < a.rate_factor < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Product accumulation
# ═══════════════════════════════════════════════════════════════════════════

class TestProductAccumulation:

    def test_dissolved_product_no_saturation(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0, product_phase=ProductPhase.DISSOLVED,
        )
        acc = _estimate_accumulation(p)
        assert acc.max_loading_mmol_per_g == float('inf')
        assert acc.fouling_risk == "low"

    def test_gas_product_no_saturation(self):
        p = TransformationProduct(
            name="test", formula="N2", target_formula="NO3-",
            dg_rxn_kj_mol=-50.0, product_phase=ProductPhase.GAS,
        )
        acc = _estimate_accumulation(p)
        assert acc.max_loading_mmol_per_g == float('inf')

    def test_solid_precipitate_has_capacity(self):
        p = TransformationProduct(
            name="CaCO3", formula="CaCO₃", target_formula="CO₂",
            dg_rxn_kj_mol=-47.7, ksp_log=-8.48,
            product_phase=ProductPhase.SOLID_PRECIPITATE,
        )
        acc = _estimate_accumulation(p)
        assert acc.max_loading_mmol_per_g > 0
        assert acc.max_loading_mmol_per_g < float('inf')
        assert acc.time_to_saturation_hours > 0

    def test_very_insoluble_product_crystal_growth(self):
        """Very low Ksp should trigger crystal growth model with higher capacity."""
        p = TransformationProduct(
            name="HAp", formula="Ca5(PO4)3OH", target_formula="PO4",
            dg_rxn_kj_mol=-120.0, ksp_log=-58.0,
            product_phase=ProductPhase.SOLID_PRECIPITATE,
        )
        acc = _estimate_accumulation(p)
        assert acc.capacity_model == "crystal_growth"
        assert acc.loading_curve_type == "nucleation_growth"

    def test_bound_to_site_langmuir(self):
        p = TransformationProduct(
            name="ZrP", formula="Zr(HPO4)2", target_formula="PO4",
            dg_rxn_kj_mol=-95.0,
            product_phase=ProductPhase.BOUND_TO_SITE,
        )
        acc = _estimate_accumulation(p)
        assert acc.loading_curve_type == "langmuir"


# ═══════════════════════════════════════════════════════════════════════════
# Regeneration assessment
# ═══════════════════════════════════════════════════════════════════════════

class TestRegeneration:

    def test_catalytic_is_regenerable(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0, turnover=TurnoverMode.CATALYTIC,
        )
        regen = _assess_regeneration(p)
        assert regen.regenerable is True
        assert regen.estimated_cycles >= 100

    def test_stoichiometric_cheap_not_regenerable(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0, turnover=TurnoverMode.STOICHIOMETRIC_CHEAP,
        )
        regen = _assess_regeneration(p)
        assert regen.regenerable is False
        assert regen.estimated_cycles == 1

    def test_stoichiometric_expensive_zero_cycles(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0, turnover=TurnoverMode.STOICHIOMETRIC_EXPENSIVE,
        )
        regen = _assess_regeneration(p)
        assert regen.regenerable is False
        assert regen.estimated_cycles == 0


# ═══════════════════════════════════════════════════════════════════════════
# Cascade benefit
# ═══════════════════════════════════════════════════════════════════════════

class TestCascadeBenefit:

    def test_no_benefit_when_not_flagged(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0,
            benefits_from_confinement=False,
        )
        cb = _assess_cascade(p)
        assert cb.benefits is False
        assert cb.confined_concentration_factor == 1.0

    def test_benefit_when_flagged(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0,
            benefits_from_confinement=True,
            cascade_notes="Pattern 2 in architecture.",
        )
        cb = _assess_cascade(p)
        assert cb.benefits is True
        assert cb.confined_concentration_factor > 1.0

    def test_pattern_extracted_from_notes(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0,
            benefits_from_confinement=True,
            cascade_notes="Gold-standard cascade: Pattern 2. Cage is self-contained.",
        )
        cb = _assess_cascade(p)
        assert cb.cascade_pattern == "Pattern 2"

    def test_pore_selectivity_boost(self):
        p = TransformationProduct(
            name="test", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-50.0,
            benefits_from_confinement=True,
        )
        cb = _assess_cascade(p)
        assert cb.pore_selectivity_boost > 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Full single-pathway scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestSinglePathwayScoring:

    def test_energy_decomposition(self):
        """ΔG_total = ΔG_bind + ΔG_transform."""
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 2.0})
        calcite = [p for p in products if "calcite" in p.name.lower()][0]
        score = score_capture_transform(
            product=calcite,
            dg_bind_kj=-15.0,
        )
        expected_total = -15.0 + calcite.dg_rxn_kj_mol
        assert abs(score.dg_total_kj - expected_total) < 0.01

    def test_log_ka_total_consistent(self):
        """log Ka total should be consistent with ΔG_total via LN10_RT."""
        products = enumerate_transformations("PO4_3-")
        p = products[0]
        score = score_capture_transform(product=p, dg_bind_kj=-20.0)
        expected_log_ka = -score.dg_total_kj / (2.303 * 8.314e-3 * 298.15)
        assert abs(score.log_ka_total - round(expected_log_ka, 2)) < 0.1

    def test_gold_pathway_detected(self):
        """CaCO₃ via CA mimic in seawater should be gold."""
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 10.0})
        calcite = [p for p in products if "calcite" in p.name.lower()][0]
        score = score_capture_transform(
            product=calcite, dg_bind_kj=-15.0,
            matrix_species={"Ca2+": 10.0},
        )
        assert score.is_gold
        assert score.is_viable

    def test_n2_excluded_not_viable(self):
        """N₂ fixation should score 0 orthogonality and not be viable."""
        products = enumerate_transformations("N2")
        score = score_capture_transform(product=products[0], dg_bind_kj=0.0)
        assert score.orthogonality_score == 0.0
        assert not score.is_viable

    def test_co_reactant_assessment_populated(self):
        """Co-reactant assessments should be populated for products with co-reactants."""
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0})
        calcite = [p for p in products if "calcite" in p.name.lower()][0]
        score = score_capture_transform(
            product=calcite, dg_bind_kj=-15.0,
            matrix_species={"Ca2+": 5.0},
        )
        assert len(score.co_reactant_assessments) >= 1

    def test_cascade_benefit_present_for_co2(self):
        """CO₂ calcite pathway should show cascade benefit."""
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0})
        calcite = [p for p in products if "calcite" in p.name.lower()][0]
        score = score_capture_transform(product=calcite, dg_bind_kj=-15.0)
        assert score.cascade is not None
        assert score.cascade.benefits is True

    def test_advantages_populated(self):
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0})
        calcite = [p for p in products if "calcite" in p.name.lower()][0]
        score = score_capture_transform(
            product=calcite, dg_bind_kj=-15.0,
            matrix_species={"Ca2+": 5.0},
        )
        assert len(score.advantages) >= 1

    def test_critical_risk_on_endergonic(self):
        """Endergonic pathway without energy input should flag critical risk."""
        p = TransformationProduct(
            name="test_endergonic", formula="X", target_formula="Y",
            dg_rxn_kj_mol=50.0,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        p.orthogonality_score = 0.0
        score = score_capture_transform(product=p, dg_bind_kj=-10.0)
        assert score.critical_risk is not None
        assert "endergonic" in score.critical_risk.lower() or "unfavorable" in score.critical_risk.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Composite scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositeScoring:

    def test_composite_bounded_0_1(self):
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0})
        for p in products:
            score = score_capture_transform(product=p, dg_bind_kj=-15.0,
                                           matrix_species={"Ca2+": 5.0})
            assert 0.0 <= score.composite_score <= 1.0

    def test_gold_scores_higher_than_silver(self):
        """Gold-tier pathways should composite-score higher than silver."""
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0})
        scores = score_all_products(products, dg_bind_kj=-15.0,
                                    matrix_species={"Ca2+": 5.0})
        gold = [s for s in scores if s.is_gold]
        non_gold = [s for s in scores if not s.is_gold and s.orthogonality_score > 0]
        if gold and non_gold:
            assert max(s.composite_score for s in gold) >= max(s.composite_score for s in non_gold)

    def test_stronger_dg_scores_higher(self):
        """More negative ΔG_total should produce higher composite (all else equal)."""
        p = TransformationProduct(
            name="strong", formula="X", target_formula="Y",
            dg_rxn_kj_mol=-100.0,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        p.orthogonality_score = 0.9
        score_strong = score_capture_transform(product=p, dg_bind_kj=-20.0)

        p2 = TransformationProduct(
            name="weak", formula="X2", target_formula="Y",
            dg_rxn_kj_mol=-10.0,
            energy_input=EnergyInput.NONE,
            turnover=TurnoverMode.CATALYTIC,
        )
        p2.orthogonality_score = 0.9
        score_weak = score_capture_transform(product=p2, dg_bind_kj=-20.0)

        assert score_strong.composite_score > score_weak.composite_score


# ═══════════════════════════════════════════════════════════════════════════
# Batch scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchScoring:

    def test_score_all_products_sorted(self):
        products = enumerate_transformations("PO4_3-", matrix_species={"Ca2+": 3.0, "NH4+": 5.0, "Mg2+": 2.0})
        scores = score_all_products(products, dg_bind_kj=-25.0,
                                    matrix_species={"Ca2+": 3.0, "NH4+": 5.0, "Mg2+": 2.0})
        composites = [s.composite_score for s in scores]
        assert composites == sorted(composites, reverse=True)

    def test_score_all_products_count_matches(self):
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 2.0})
        scores = score_all_products(products, dg_bind_kj=-15.0)
        assert len(scores) == len(products)


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_evaluate_co2_seawater(self):
        """Full pipeline: CO₂ in seawater → scored ranked products."""
        scores = evaluate_capture_transform(
            target_formula="CO2",
            matrix_species={"Ca2+": 10.0, "Mg2+": 53.0},
            dg_bind_kj=-15.0,
        )
        assert len(scores) >= 3  # calcite, magnesite, amine, photocatalytic
        assert scores[0].composite_score >= scores[-1].composite_score
        assert scores[0].is_viable

    def test_evaluate_phosphate_wastewater(self):
        """Full pipeline: phosphate in wastewater → all viable."""
        scores = evaluate_capture_transform(
            target_formula="PO4_3-",
            matrix_species={"Ca2+": 3.0, "Mg2+": 2.0, "NH4+": 5.0},
            dg_bind_kj=-25.0,
        )
        assert len(scores) >= 3
        viable = [s for s in scores if s.is_viable]
        assert len(viable) == len(scores)  # all phosphate pathways are viable

    def test_evaluate_lead(self):
        scores = evaluate_capture_transform("Pb2+", dg_bind_kj=-30.0)
        assert len(scores) >= 1
        assert "PbS" in scores[0].product.formula

    def test_evaluate_unknown_empty(self):
        scores = evaluate_capture_transform("Xe")
        assert len(scores) == 0

    def test_evaluate_n2_excluded(self):
        scores = evaluate_capture_transform("N2")
        assert len(scores) >= 1
        assert all(not s.is_viable for s in scores)


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

class TestReporting:

    def test_report_not_empty(self):
        scores = evaluate_capture_transform(
            "CO2", matrix_species={"Ca2+": 5.0}, dg_bind_kj=-15.0)
        report = scoring_report(scores)
        assert "Capture-Transform Scoring" in report
        assert "pathway" in report.lower()

    def test_report_empty_scores(self):
        report = scoring_report([])
        assert "No capture-transform" in report

    def test_summary_method(self):
        scores = evaluate_capture_transform(
            "CO2", matrix_species={"Ca2+": 5.0}, dg_bind_kj=-15.0)
        for s in scores:
            summary = s.summary()
            assert "ΔG" in summary
            assert s.product.formula in summary


# ═══════════════════════════════════════════════════════════════════════════
# Confidence and deployment
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceAndDeployment:

    def test_ksp_based_high_confidence(self):
        """Products with Ksp should get higher confidence."""
        products = enumerate_transformations("Pb2+")
        score = score_capture_transform(product=products[0], dg_bind_kj=-30.0)
        assert score.confidence >= 0.7

    def test_gold_pathway_field_scale(self):
        """Gold-standard catalytic pathway should be field-deployable."""
        products = enumerate_transformations("CO2", matrix_species={"Ca2+": 5.0})
        calcite = [p for p in products if "calcite" in p.name.lower()][0]
        score = score_capture_transform(
            product=calcite, dg_bind_kj=-15.0,
            matrix_species={"Ca2+": 5.0},
        )
        assert score.deployment_scale == "field"
