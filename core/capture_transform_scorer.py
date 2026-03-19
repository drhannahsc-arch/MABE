"""
core/capture_transform_scorer.py — Capture-Transform Scoring Module

Extends MABE's binding scorer with transform-specific energy terms:
  - ΔG_transform: free energy of the covalent transformation
  - Co-reactant availability: is the transformation co-reactant-limited?
  - Product accumulation: capacity and fouling prediction
  - Regeneration: cycle life and energy cost
  - Orthogonality: energy/reagent self-sufficiency
  - Cascade coupling: benefit from 3D scaffold confinement

This is NOT a replacement for unified_scorer_v2. It is an ADDITIONAL
scoring layer that evaluates capture-transform viability on top of
the existing binding affinity prediction.

Pipeline:
  1. unified_scorer_v2.predict(uc) → binding ΔG, selectivity
  2. transform_enumerator.enumerate() → candidate products
  3. THIS MODULE: score each (binding site × product) pair for
     total driving force, kinetic feasibility, and deployment viability

Connects to:
  - core/transform_enumerator.py (TransformationProduct input)
  - core/substrate_anchor.py (TetherProtocol for deployment scoring)
  - core/unified_scorer_v2.py (binding ΔG input)
  - mabe/realization/models.py (InteractionGeometrySpec extension)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.transform_enumerator import (
    TransformationProduct,
    CoReactantSource,
    EnergyInput,
    TurnoverMode,
    ProductPhase,
    ClickCompatibility,
    compute_orthogonality,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

R_kJ = 8.314e-3       # kJ/(mol·K)
T_STD = 298.15         # K
RT_STD = R_kJ * T_STD  # 2.4790 kJ/mol
LN10_RT = 2.303 * RT_STD  # 5.709 kJ/mol — converts log K ↔ ΔG


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoReactantAvailability:
    """Assessment of whether co-reactant supply limits the transformation."""
    species: str                     # "Ca²⁺", "Mg²⁺", "hν"
    source: CoReactantSource
    concentration_mM: float          # in the matrix (0 for non-solution)
    required_mM: float               # estimated minimum for reasonable rate
    limiting: bool = False           # True if co-reactant is the bottleneck
    rate_factor: float = 1.0         # 0-1 multiplier on transformation rate
    notes: str = ""


@dataclass
class ProductAccumulationModel:
    """Prediction of how product accumulates and when it blocks further capture."""
    product_phase: ProductPhase
    capacity_model: str              # "surface_limited", "pore_limited", "crystal_growth", "dissolved"

    # Capacity
    max_loading_mmol_per_g: float    # maximum product loading on substrate
    time_to_saturation_hours: float  # at expected capture rate
    loading_curve_type: str          # "langmuir", "linear", "nucleation_growth"

    # Fouling
    fouling_risk: str                # "low", "moderate", "high"
    fouling_mechanism: str           # "pore_blockage", "surface_burial", "crystal_overgrowth", "none"
    fouling_mitigation: str          # "periodic wash", "replace substrate", "product harvesting"

    notes: str = ""


@dataclass
class RegenerationAssessment:
    """Can the capture site be regenerated after product harvest?"""
    regenerable: bool
    turnover: TurnoverMode
    regeneration_method: str         # "water wash", "acid wash", "thermal", "replace substrate"
    energy_per_cycle_kj: float       # energy cost per regeneration cycle
    estimated_cycles: int            # number of capture/harvest cycles before degradation
    degradation_mechanism: str       # "hydrolysis", "oxidation", "product poisoning", "none"
    cost_per_cycle: str              # estimated $ per regeneration
    notes: str = ""


@dataclass
class CascadeBenefit:
    """Assessment of whether a 3D scaffold improves this pathway."""
    benefits: bool
    confined_concentration_factor: float  # fold increase in intermediate concentration
    pore_selectivity_boost: float         # multiplicative selectivity enhancement from pore
    cascade_pattern: str                  # "Pattern 1", "Pattern 2", etc. or "none"
    required_module_spacing_nm: float     # optimal inter-module distance
    stoichiometric_bottleneck: str        # which module limits throughput
    notes: str = ""


@dataclass
class CaptureTransformScore:
    """Complete scoring of one capture-transform pathway.

    This is the output of the capture-transform scorer.
    One per (capture site × transformation product × substrate) combination.
    """

    # ── Identity ──
    name: str                        # "Zn-CA mimic → CaCO₃ on silica beads"
    target_formula: str              # "CO₂"
    product: TransformationProduct   # from transform_enumerator

    # ── Binding score (from upstream scorer) ──
    dg_bind_kj: float                # ΔG of initial target binding (from unified_scorer_v2)
    log_ka_bind: float               # log Ka for binding step
    selectivity_bind: float          # selectivity ratio vs primary competitor

    # ── Transform score (new terms) ──
    dg_transform_kj: float           # ΔG of covalent transformation
    dg_total_kj: float               # dg_bind + dg_transform
    log_ka_total: float              # effective log Ka for full pathway

    # ── Kinetic feasibility ──
    activation_barrier_kj: float     # estimated Ea for transformation
    rate_class: str                  # "fast" (<1 min), "moderate" (1-60 min),
                                     # "slow" (1-24 h), "very_slow" (>24 h)
    kinetic_notes: str = ""

    # ── Co-reactant assessment ──
    co_reactant_assessments: list[CoReactantAvailability] = field(default_factory=list)
    co_reactant_limited: bool = False

    # ── Product accumulation ──
    accumulation: Optional[ProductAccumulationModel] = None

    # ── Regeneration ──
    regeneration: Optional[RegenerationAssessment] = None

    # ── Cascade benefit ──
    cascade: Optional[CascadeBenefit] = None

    # ── Orthogonality ──
    orthogonality_score: float = 0.0

    # ── Deployment viability ──
    deployment_scale: str = ""       # "lab", "pilot", "field", "industrial"
    substrate_compatible: bool = True
    click_compatible: bool = True

    # ── Composite ──
    composite_score: float = 0.0
    confidence: float = 0.0
    confidence_basis: str = ""

    # ── Rationale ──
    advantages: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    critical_risk: Optional[str] = None

    @property
    def is_viable(self) -> bool:
        """Is this pathway thermodynamically viable and orthogonal?"""
        return self.dg_total_kj < 0 and self.orthogonality_score > 0.0

    @property
    def is_gold(self) -> bool:
        """Gold-standard: spontaneous, matrix-native co-reactant, catalytic, passive."""
        return self.orthogonality_score >= 0.8

    def summary(self) -> str:
        tier = "GOLD" if self.is_gold else "SILVER" if self.orthogonality_score >= 0.4 else \
               "BRONZE" if self.orthogonality_score > 0 else "EXCLUDED"
        lines = [
            f"[{tier}] {self.name}",
            f"  Target: {self.target_formula} → {self.product.formula}",
            f"  ΔG_bind = {self.dg_bind_kj:+.1f}, ΔG_transform = {self.dg_transform_kj:+.1f}, "
            f"ΔG_total = {self.dg_total_kj:+.1f} kJ/mol",
            f"  Rate: {self.rate_class}",
            f"  Orthogonality: {self.orthogonality_score:.4f}",
            f"  Composite: {self.composite_score:.4f}",
        ]
        if self.co_reactant_limited:
            lines.append(f"  WARNING: co-reactant limited")
        if self.cascade and self.cascade.benefits:
            lines.append(f"  Cascade benefit: {self.cascade.cascade_pattern} "
                         f"(confinement ×{self.cascade.confined_concentration_factor:.0f})")
        if self.critical_risk:
            lines.append(f"  RISK: {self.critical_risk}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Scoring functions
# ═══════════════════════════════════════════════════════════════════════════

def _classify_rate(activation_barrier_kj: float, temperature_c: float = 25.0) -> str:
    """Classify transformation rate from activation barrier.

    Uses Eyring equation approximation:
      k ≈ (kT/h) × exp(-Ea/RT)
    Thresholds calibrated to:
      Ea < 40 kJ/mol → fast (seconds to minutes)
      40-60 → moderate (minutes to 1 hour)
      60-80 → slow (hours)
      >80 → very slow (>24 h or impractical without catalyst)
    """
    if activation_barrier_kj is None or activation_barrier_kj <= 0:
        return "fast"
    if activation_barrier_kj < 40.0:
        return "fast"
    elif activation_barrier_kj < 60.0:
        return "moderate"
    elif activation_barrier_kj < 80.0:
        return "slow"
    else:
        return "very_slow"


def _assess_co_reactant(
    species: str,
    source: CoReactantSource,
    matrix_concentration_mM: float,
    stoichiometric_requirement: float = 0.1,  # minimum mM for reasonable rate
) -> CoReactantAvailability:
    """Assess whether a co-reactant is limiting."""

    if source == CoReactantSource.NONE:
        return CoReactantAvailability(
            species=species, source=source,
            concentration_mM=0.0, required_mM=0.0,
            limiting=False, rate_factor=1.0,
            notes="No co-reactant needed — site itself is the reactant",
        )

    if source == CoReactantSource.SOLAR_PHOTOCATALYTIC:
        return CoReactantAvailability(
            species=species, source=source,
            concentration_mM=0.0, required_mM=0.0,
            limiting=False, rate_factor=0.3,  # solar limited by flux, not concentration
            notes="Rate limited by solar flux, not co-reactant concentration",
        )

    if source == CoReactantSource.EXTERNALLY_SUPPLIED:
        return CoReactantAvailability(
            species=species, source=source,
            concentration_mM=0.0, required_mM=stoichiometric_requirement,
            limiting=True, rate_factor=0.0,
            notes="Externally supplied — fails orthogonality",
        )

    # Matrix-native or substrate-preloaded
    if matrix_concentration_mM <= 0:
        # Not detected in matrix — must be preloaded
        if source == CoReactantSource.MATRIX_NATIVE:
            return CoReactantAvailability(
                species=species, source=source,
                concentration_mM=0.0, required_mM=stoichiometric_requirement,
                limiting=True, rate_factor=0.1,
                notes=f"{species} listed as matrix-native but not detected in matrix",
            )
        else:
            return CoReactantAvailability(
                species=species, source=source,
                concentration_mM=0.0, required_mM=stoichiometric_requirement,
                limiting=False, rate_factor=0.5,
                notes="Substrate-preloaded — will deplete over time",
            )

    # Available in matrix — assess if limiting
    ratio = matrix_concentration_mM / max(0.001, stoichiometric_requirement)
    if ratio >= 10.0:
        rate_factor = 1.0
        limiting = False
        notes = f"{species} in large excess ({ratio:.0f}× stoichiometric)"
    elif ratio >= 1.0:
        rate_factor = 0.8
        limiting = False
        notes = f"{species} adequate ({ratio:.1f}× stoichiometric)"
    else:
        rate_factor = ratio
        limiting = True
        notes = f"{species} sub-stoichiometric ({ratio:.2f}×) — rate-limiting"

    return CoReactantAvailability(
        species=species, source=source,
        concentration_mM=matrix_concentration_mM,
        required_mM=stoichiometric_requirement,
        limiting=limiting, rate_factor=rate_factor,
        notes=notes,
    )


def _estimate_accumulation(
    product: TransformationProduct,
    substrate_surface_area_m2_g: float = 300.0,  # silica beads default
    capture_rate_mmol_per_g_per_h: float = 0.01,
) -> ProductAccumulationModel:
    """Estimate product accumulation model."""

    phase = product.product_phase

    if phase == ProductPhase.DISSOLVED:
        return ProductAccumulationModel(
            product_phase=phase,
            capacity_model="dissolved",
            max_loading_mmol_per_g=float('inf'),
            time_to_saturation_hours=float('inf'),
            loading_curve_type="linear",
            fouling_risk="low",
            fouling_mechanism="none",
            fouling_mitigation="Collect in wash water",
            notes="Product dissolves — no accumulation on substrate",
        )

    if phase == ProductPhase.GAS:
        return ProductAccumulationModel(
            product_phase=phase,
            capacity_model="dissolved",
            max_loading_mmol_per_g=float('inf'),
            time_to_saturation_hours=float('inf'),
            loading_curve_type="linear",
            fouling_risk="low",
            fouling_mechanism="none",
            fouling_mitigation="Product escapes as gas",
        )

    # Solid precipitate or bound to site
    # Estimate capacity from surface area
    # Typical monolayer: ~5 µmol/m² = 0.005 mmol/m²
    site_density_mmol_per_m2 = 0.005
    max_loading = site_density_mmol_per_m2 * substrate_surface_area_m2_g

    # For crystal growth (e.g., CaCO₃, struvite), capacity is higher
    if product.ksp_log is not None and product.ksp_log < -10:
        # Very insoluble — crystal grows beyond monolayer
        max_loading *= 10.0  # crystal growth extends capacity
        loading_type = "nucleation_growth"
        fouling = "moderate"
        fouling_mech = "crystal_overgrowth"
    elif phase == ProductPhase.BOUND_TO_SITE:
        loading_type = "langmuir"
        fouling = "low"
        fouling_mech = "surface_burial"
    else:
        loading_type = "langmuir"
        fouling = "moderate"
        fouling_mech = "pore_blockage"

    time_sat = max_loading / max(1e-6, capture_rate_mmol_per_g_per_h)

    return ProductAccumulationModel(
        product_phase=phase,
        capacity_model="crystal_growth" if loading_type == "nucleation_growth" else "surface_limited",
        max_loading_mmol_per_g=round(max_loading, 4),
        time_to_saturation_hours=round(time_sat, 1),
        loading_curve_type=loading_type,
        fouling_risk=fouling,
        fouling_mechanism=fouling_mech,
        fouling_mitigation="Periodic scraping/dissolution" if fouling != "low" else "Replace when saturated",
    )


def _assess_regeneration(product: TransformationProduct) -> RegenerationAssessment:
    """Assess regeneration potential."""
    turnover = product.turnover

    if turnover == TurnoverMode.CATALYTIC:
        return RegenerationAssessment(
            regenerable=True,
            turnover=turnover,
            regeneration_method="Product harvest only — site regenerates automatically",
            energy_per_cycle_kj=0.0,
            estimated_cycles=1000,
            degradation_mechanism="hydrolysis" if product.energy_input == EnergyInput.NONE else "photodegradation",
            cost_per_cycle="~$0 (automatic)",
            notes="Catalytic site is not consumed. Product may need physical removal.",
        )
    elif turnover == TurnoverMode.STOICHIOMETRIC_CHEAP:
        return RegenerationAssessment(
            regenerable=False,
            turnover=turnover,
            regeneration_method="Replace substrate when sites exhausted",
            energy_per_cycle_kj=0.0,
            estimated_cycles=1,  # single use per site
            degradation_mechanism="none",
            cost_per_cycle="Substrate replacement cost",
            notes="Stoichiometric: capture sites consumed. Substrate must be replaced. "
                  "Product is itself valuable — not waste.",
        )
    else:
        return RegenerationAssessment(
            regenerable=False,
            turnover=turnover,
            regeneration_method="Not economically viable",
            energy_per_cycle_kj=0.0,
            estimated_cycles=0,
            degradation_mechanism="cost",
            cost_per_cycle="Prohibitive",
        )


def _assess_cascade(product: TransformationProduct) -> CascadeBenefit:
    """Assess whether 3D scaffold confinement improves this pathway."""

    if not product.benefits_from_confinement:
        return CascadeBenefit(
            benefits=False,
            confined_concentration_factor=1.0,
            pore_selectivity_boost=1.0,
            cascade_pattern="none",
            required_module_spacing_nm=0.0,
            stoichiometric_bottleneck="N/A",
        )

    # Estimate confinement enhancement
    # Inside a 20 nm cage, one molecule gives ~mM effective concentration
    # Bulk: µM or less for dilute targets
    # Enhancement factor: ~1000× for 20 nm cage, ~100× for 40 nm
    cage_diameter_nm = 20.0  # default assumption
    cage_volume_L = (4.0 / 3.0) * math.pi * (cage_diameter_nm / 2.0 * 1e-9) ** 3 * 1000.0
    # One molecule in this volume:
    avogadro = 6.022e23
    one_molecule_molar = 1.0 / (avogadro * cage_volume_L)
    # Convert to mM
    one_molecule_mM = one_molecule_molar * 1e3
    # Bulk typical: ~0.001 mM
    bulk_mM = 0.001
    confinement_factor = one_molecule_mM / bulk_mM if bulk_mM > 0 else 1000.0
    confinement_factor = min(10000.0, max(1.0, confinement_factor))

    # Pore selectivity boost estimate
    # Assume ~2× for generic size-based exclusion
    pore_boost = 2.0

    # Parse cascade pattern from notes
    pattern = "none"
    if product.cascade_notes:
        for p in ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Pattern 5"]:
            if p in product.cascade_notes:
                pattern = p
                break

    return CascadeBenefit(
        benefits=True,
        confined_concentration_factor=round(confinement_factor, 0),
        pore_selectivity_boost=pore_boost,
        cascade_pattern=pattern,
        required_module_spacing_nm=3.0,  # typical optimal for nm-scale cascades
        stoichiometric_bottleneck="co-reactant module" if product.co_reactants else "capture site",
        notes=product.cascade_notes[:200] if product.cascade_notes else "",
    )


def _compute_composite(score: CaptureTransformScore) -> float:
    """Compute composite score for ranking.

    Weighting:
      40% — thermodynamic driving force (ΔG_total)
      20% — orthogonality
      15% — kinetic feasibility (rate class)
      10% — co-reactant availability
      10% — cascade benefit
       5% — regeneration potential
    """
    composite = 0.0

    # Thermodynamic (40%) — scale: ΔG ≤ -100 → full credit
    if score.dg_total_kj < 0:
        thermo = min(1.0, abs(score.dg_total_kj) / 100.0)
    else:
        thermo = 0.0
    composite += 0.40 * thermo

    # Orthogonality (20%)
    composite += 0.20 * score.orthogonality_score

    # Kinetic feasibility (15%)
    rate_scores = {"fast": 1.0, "moderate": 0.7, "slow": 0.3, "very_slow": 0.05}
    composite += 0.15 * rate_scores.get(score.rate_class, 0.3)

    # Co-reactant availability (10%)
    if not score.co_reactant_limited:
        composite += 0.10
    else:
        # Partial credit based on worst rate factor
        if score.co_reactant_assessments:
            worst = min(a.rate_factor for a in score.co_reactant_assessments)
            composite += 0.10 * worst

    # Cascade benefit (10%)
    if score.cascade and score.cascade.benefits:
        composite += 0.10 * min(1.0, score.cascade.confined_concentration_factor / 1000.0)

    # Regeneration (5%)
    if score.regeneration and score.regeneration.regenerable:
        cycle_credit = min(1.0, score.regeneration.estimated_cycles / 100.0)
        composite += 0.05 * cycle_credit
    elif score.regeneration and score.regeneration.turnover == TurnoverMode.STOICHIOMETRIC_CHEAP:
        composite += 0.025  # partial credit for cheap stoichiometric

    return round(min(1.0, composite), 4)


# ═══════════════════════════════════════════════════════════════════════════
# Main scoring function
# ═══════════════════════════════════════════════════════════════════════════

def score_capture_transform(
    product: TransformationProduct,
    dg_bind_kj: float = 0.0,
    log_ka_bind: float = 0.0,
    selectivity_bind: float = 1.0,
    matrix_species: Optional[dict[str, float]] = None,
    substrate_surface_area_m2_g: float = 300.0,
    temperature_c: float = 25.0,
) -> CaptureTransformScore:
    """Score a single capture-transform pathway.

    Args:
        product: TransformationProduct from transform_enumerator
        dg_bind_kj: ΔG of initial target binding (from unified_scorer_v2 or estimate)
        log_ka_bind: log Ka for binding step
        selectivity_bind: selectivity ratio vs primary competitor
        matrix_species: {formula: concentration_mM} for co-reactant assessment
        substrate_surface_area_m2_g: substrate surface area for accumulation model
        temperature_c: operating temperature

    Returns:
        CaptureTransformScore with all assessment dimensions.
    """
    if matrix_species is None:
        matrix_species = {}

    # ── Thermodynamics ──
    dg_transform = product.dg_rxn_kj_mol
    dg_total = dg_bind_kj + dg_transform
    log_ka_total = -dg_total / LN10_RT if dg_total != 0 else 0.0

    # ── Kinetics ──
    ea = product.activation_barrier_kj_mol if product.activation_barrier_kj_mol else 50.0
    rate_class = _classify_rate(ea, temperature_c)

    # ── Co-reactant assessment ──
    cr_assessments = []
    cr_limited = False
    for cr in product.co_reactants:
        if cr.source == CoReactantSource.NONE:
            continue  # site itself is the reactant
        conc = matrix_species.get(cr.identity, 0.0)
        assessment = _assess_co_reactant(
            species=cr.identity,
            source=cr.source,
            matrix_concentration_mM=conc,
        )
        cr_assessments.append(assessment)
        if assessment.limiting:
            cr_limited = True

    # ── Product accumulation ──
    accumulation = _estimate_accumulation(
        product, substrate_surface_area_m2_g
    )

    # ── Regeneration ──
    regeneration = _assess_regeneration(product)

    # ── Cascade benefit ──
    cascade = _assess_cascade(product)

    # ── Build name ──
    site_name = product.capture_sites[0].name if product.capture_sites else "unknown site"
    name = f"{site_name} → {product.formula}"

    # ── Advantages / limitations ──
    advantages = []
    limitations = []

    if dg_total < -50:
        advantages.append(f"Strong thermodynamic driving force (ΔG = {dg_total:+.1f} kJ/mol)")
    if product.turnover == TurnoverMode.CATALYTIC:
        advantages.append("Catalytic — capture site regenerates automatically")
    if product.orthogonality_score >= 0.8:
        advantages.append("Gold-standard orthogonality — no external inputs needed")
    if cascade and cascade.benefits:
        advantages.append(f"Benefits from 3D scaffold confinement ({cascade.cascade_pattern})")
    if product.product_phase == ProductPhase.SOLID_PRECIPITATE and product.harvestable:
        advantages.append(f"Harvestable solid product: {product.feedstock_value}")

    if cr_limited:
        limiting = [a.species for a in cr_assessments if a.limiting]
        limitations.append(f"Co-reactant limited: {', '.join(limiting)}")
    if rate_class == "very_slow":
        limitations.append("Very slow kinetics — may need catalytic enhancement")
    if accumulation and accumulation.fouling_risk == "high":
        limitations.append(f"High fouling risk: {accumulation.fouling_mechanism}")
    if not regeneration.regenerable and regeneration.turnover == TurnoverMode.STOICHIOMETRIC_CHEAP:
        limitations.append("Stoichiometric — substrate replacement needed when exhausted")

    # ── Critical risk ──
    critical_risk = None
    if dg_total > 0 and product.energy_input == EnergyInput.NONE:
        critical_risk = "Endergonic without energy input — thermodynamically unfavorable"
    elif cr_limited and all(a.rate_factor < 0.1 for a in cr_assessments if a.limiting):
        critical_risk = "Severely co-reactant limited — may not function in this matrix"

    # ── Confidence ──
    # Based on data source quality
    if product.ksp_log is not None:
        confidence = 0.8  # Ksp-based predictions are well-grounded
        confidence_basis = "Based on published Ksp and NIST thermochemical data"
    elif "NIST" in product.dg_source or "Chase" in product.dg_source:
        confidence = 0.7
        confidence_basis = "Based on NIST-JANAF thermochemical data"
    else:
        confidence = 0.5
        confidence_basis = "Based on literature estimates — experimental validation needed"

    # ── Assemble score ──
    score = CaptureTransformScore(
        name=name,
        target_formula=product.target_formula,
        product=product,
        dg_bind_kj=dg_bind_kj,
        log_ka_bind=log_ka_bind,
        selectivity_bind=selectivity_bind,
        dg_transform_kj=dg_transform,
        dg_total_kj=dg_total,
        log_ka_total=round(log_ka_total, 2),
        activation_barrier_kj=ea,
        rate_class=rate_class,
        co_reactant_assessments=cr_assessments,
        co_reactant_limited=cr_limited,
        accumulation=accumulation,
        regeneration=regeneration,
        cascade=cascade,
        orthogonality_score=product.orthogonality_score,
        deployment_scale=_estimate_scale(product),
        advantages=advantages,
        limitations=limitations,
        critical_risk=critical_risk,
        confidence=confidence,
        confidence_basis=confidence_basis,
    )

    score.composite_score = _compute_composite(score)
    return score


def _estimate_scale(product: TransformationProduct) -> str:
    """Estimate deployment scale viability."""
    if product.turnover == TurnoverMode.CATALYTIC and product.orthogonality_score >= 0.8:
        return "field"  # passive, catalytic → field deployable
    elif product.orthogonality_score >= 0.4:
        return "pilot"
    elif product.orthogonality_score > 0:
        return "lab"
    else:
        return "lab"  # excluded pathways are lab-only investigation


# ═══════════════════════════════════════════════════════════════════════════
# Batch scoring
# ═══════════════════════════════════════════════════════════════════════════

def score_all_products(
    products: list[TransformationProduct],
    dg_bind_kj: float = 0.0,
    log_ka_bind: float = 0.0,
    selectivity_bind: float = 1.0,
    matrix_species: Optional[dict[str, float]] = None,
    substrate_surface_area_m2_g: float = 300.0,
    temperature_c: float = 25.0,
) -> list[CaptureTransformScore]:
    """Score all transformation products for a target.

    Typically called after transform_enumerator.enumerate_transformations().
    Returns scores sorted by composite score (descending).
    """
    scores = []
    for product in products:
        score = score_capture_transform(
            product=product,
            dg_bind_kj=dg_bind_kj,
            log_ka_bind=log_ka_bind,
            selectivity_bind=selectivity_bind,
            matrix_species=matrix_species,
            substrate_surface_area_m2_g=substrate_surface_area_m2_g,
            temperature_c=temperature_c,
        )
        scores.append(score)

    scores.sort(key=lambda s: s.composite_score, reverse=True)
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end convenience
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_capture_transform(
    target_formula: str,
    matrix_species: Optional[dict[str, float]] = None,
    dg_bind_kj: float = -20.0,
    substrate_surface_area_m2_g: float = 300.0,
    temperature_c: float = 25.0,
) -> list[CaptureTransformScore]:
    """End-to-end: enumerate products → score all → return ranked.

    Convenience function combining transform_enumerator + this scorer.
    """
    from core.transform_enumerator import enumerate_transformations

    products = enumerate_transformations(
        target_formula=target_formula,
        matrix_species=matrix_species,
        temperature_c=temperature_c,
    )

    return score_all_products(
        products=products,
        dg_bind_kj=dg_bind_kj,
        matrix_species=matrix_species,
        substrate_surface_area_m2_g=substrate_surface_area_m2_g,
        temperature_c=temperature_c,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def scoring_report(scores: list[CaptureTransformScore]) -> str:
    """Human-readable multi-pathway scoring report."""
    if not scores:
        return "No capture-transform pathways scored."

    lines = [
        f"Capture-Transform Scoring: {len(scores)} pathway(s)",
        f"{'=' * 60}",
        "",
    ]

    for i, s in enumerate(scores, 1):
        lines.append(f"{i}. {s.summary()}")
        if s.advantages:
            for a in s.advantages:
                lines.append(f"     + {a}")
        if s.limitations:
            for l in s.limitations:
                lines.append(f"     - {l}")
        lines.append(f"     Confidence: {s.confidence:.0%} ({s.confidence_basis})")
        lines.append(f"     Deployment: {s.deployment_scale}")
        lines.append("")

    return "\n".join(lines)
