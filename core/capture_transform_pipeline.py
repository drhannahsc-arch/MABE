"""
core/capture_transform_pipeline.py — End-to-End Capture-Transform Pipeline

Single entry point: design_capture_transform_system()

Takes: target formula, matrix, deployment constraints
Returns: ranked system designs with scored pathways, substrate protocols,
         and scaffold recommendations

Wires together:
  1. core/transform_enumerator.py    → enumerate products
  2. core/capture_transform_scorer.py → score pathways
  3. core/substrate_anchor.py         → generate tether protocols
  4. core/cascade_scaffold.py         → design cascades, rank scaffolds

This is the orchestration layer for the capture-transform module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.transform_enumerator import (
    enumerate_transformations,
    TransformationProduct,
    ClickCompatibility,
)
from core.capture_transform_scorer import (
    score_all_products,
    CaptureTransformScore,
)
from core.substrate_anchor import (
    SubstrateType,
    CaptureElementType,
    generate_tether_protocol,
    TetherProtocol,
)
from core.cascade_scaffold import (
    CascadeSpec,
    CascadeModule,
    PoreSpec,
    ModuleRole,
    ScaffoldSystem,
    ScaleClass,
    ScaffoldScore,
    StoichiometryResult,
    rank_scaffolds,
    recommend_scaffold,
    optimize_stoichiometry,
)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SystemDesign:
    """One complete capture-transform system design.

    Combines a scored pathway with substrate tethering and optional
    cascade scaffold recommendation.
    """
    # Pathway
    pathway_score: CaptureTransformScore

    # Substrate tethering
    tether_protocol: Optional[TetherProtocol] = None

    # Cascade (if applicable)
    cascade_spec: Optional[CascadeSpec] = None
    scaffold_recommendation: Optional[ScaffoldScore] = None
    scaffold_rationale: str = ""
    stoichiometry: Optional[StoichiometryResult] = None

    # Composite ranking
    system_score: float = 0.0

    def summary(self) -> str:
        lines = [self.pathway_score.summary()]
        if self.tether_protocol:
            lines.append(f"  Substrate: {self.tether_protocol.substrate.name}")
            lines.append(f"  Click: {self.tether_protocol.click_chemistry.value} "
                         f"({'Cu-free' if self.tether_protocol.cu_safe else 'Cu catalyst'})")
        if self.scaffold_recommendation and self.scaffold_recommendation.feasible:
            lines.append(f"  Scaffold: {self.scaffold_recommendation.scaffold.name}")
            lines.append(f"  Confinement: {self.scaffold_recommendation.confinement_factor:.0f}x")
        if self.stoichiometry:
            lines.append(f"  Capacity: {self.stoichiometry.capacity_per_scaffold} "
                         f"targets/scaffold (limited by {self.stoichiometry.limiting_module})")
        lines.append(f"  System score: {self.system_score:.4f}")
        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Complete pipeline output: all ranked system designs."""
    target_formula: str
    matrix_species: dict[str, float]
    n_products_enumerated: int
    n_viable_pathways: int
    designs: list[SystemDesign]

    # Top recommendation
    recommended: Optional[SystemDesign] = None
    recommendation_rationale: str = ""

    def summary(self) -> str:
        lines = [
            f"Capture-Transform Pipeline: {self.target_formula}",
            f"Products enumerated: {self.n_products_enumerated}",
            f"Viable pathways: {self.n_viable_pathways}",
            f"System designs: {len(self.designs)}",
            "",
        ]
        if self.recommended:
            lines.append(f"RECOMMENDED:")
            lines.append(self.recommended.summary())
            lines.append(f"  Rationale: {self.recommendation_rationale}")
            lines.append("")

        if len(self.designs) > 1:
            lines.append(f"ALL DESIGNS (ranked):")
            for i, d in enumerate(self.designs, 1):
                tier = "GOLD" if d.pathway_score.is_gold else \
                       "SILVER" if d.pathway_score.orthogonality_score >= 0.4 else "BRONZE"
                lines.append(f"  {i}. [{tier}] {d.pathway_score.name} "
                             f"(system: {d.system_score:.4f})")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Mapping helpers
# ═══════════════════════════════════════════════════════════════════════════

def _map_capture_element_type(product: TransformationProduct) -> CaptureElementType:
    """Map a transformation product's capture site to a CaptureElementType."""
    if not product.capture_sites:
        return CaptureElementType.MOLECULAR_BINDER

    site_name = product.capture_sites[0].name.lower()

    if "zn" in site_name and ("ca" in site_name or "mimic" in site_name):
        return CaptureElementType.ENZYME_MIMIC
    elif "tio" in site_name or "photocatal" in site_name or "g-c3n4" in site_name.replace("₃", "3").replace("₄", "4"):
        return CaptureElementType.PHOTOCATALYST
    elif "mof" in site_name:
        return CaptureElementType.MOF_PARTICLE
    elif "zro" in site_name or "zirconia" in site_name or "la" in site_name or "fe" in site_name.split()[0]:
        return CaptureElementType.METAL_OXIDE_NP
    elif "thiol" in site_name or "sulfide" in site_name:
        return CaptureElementType.SULFIDE_NP
    elif "zvi" in site_name or "zerovalent" in site_name:
        return CaptureElementType.ZVI_NP
    elif "amine" in site_name or "pei" in site_name:
        return CaptureElementType.MOLECULAR_BINDER
    elif "sulfonic" in site_name or "acid" in site_name:
        return CaptureElementType.MOLECULAR_BINDER
    else:
        return CaptureElementType.MOLECULAR_BINDER


def _cu_tolerant(product: TransformationProduct) -> bool:
    """Determine if the capture site tolerates Cu."""
    if not product.capture_sites:
        return True
    return product.capture_sites[0].click_compatibility != ClickCompatibility.SPAAC_ONLY


def _build_cascade_spec_from_product(
    product: TransformationProduct,
    target_scale: ScaleClass = ScaleClass.LAB,
) -> Optional[CascadeSpec]:
    """Build a cascade spec from a product that benefits from confinement.

    Returns None if product doesn't benefit from confinement.
    """
    if not product.benefits_from_confinement:
        return None

    # Build modules from capture site + co-reactants
    modules = []

    # Capture module
    if product.capture_sites:
        site = product.capture_sites[0]
        modules.append(CascadeModule(
            module_id="capture_main",
            name=site.name,
            role=ModuleRole.CAPTURE,
            functional_group=", ".join(site.functional_groups) if site.functional_groups else site.name,
            description=site.description,
            click_compatibility=site.click_compatibility,
            stoichiometric_ratio=3.0,
        ))

    # Co-reactant modules
    for i, cr in enumerate(product.co_reactants):
        if cr.source in (cr.source.NONE, cr.source.MATRIX_NATIVE, cr.source.SOLAR_PHOTOCATALYTIC):
            continue  # not a scaffold-positioned module
        if cr.source in (cr.source.SUBSTRATE_PRELOADED, cr.source.SELF_CONTAINED):
            modules.append(CascadeModule(
                module_id=f"coreactant_{i}",
                name=f"{cr.identity} source",
                role=ModuleRole.CO_REACTANT,
                functional_group=cr.formula,
                description=cr.notes or f"Provides {cr.identity} for transformation",
                stoichiometric_ratio=2.0,
            ))

    # Nucleation module (for solid products)
    if product.product_phase.value == "solid_precipitate":
        modules.append(CascadeModule(
            module_id="nucleation",
            name="Nucleation template",
            role=ModuleRole.NUCLEATION,
            functional_group="-COOH cluster",
            description="Seeds product crystal growth",
            stoichiometric_ratio=1.0,
        ))

    if len(modules) < 2:
        # Single module — no cascade benefit, flat surface suffices
        return None

    # Extract cascade pattern from notes
    pattern = "generic"
    if product.cascade_notes:
        for p in ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Pattern 5"]:
            if p in product.cascade_notes:
                pattern = p
                break

    return CascadeSpec(
        name=f"Cascade: {product.name}",
        description=product.cascade_notes[:200] if product.cascade_notes else "",
        cascade_pattern=pattern,
        target_formula=product.target_formula,
        product=product,
        modules=modules,
        min_interior_volume_nm3=20.0,
        target_scale=target_scale,
    )


def _scale_from_string(scale: str) -> ScaleClass:
    """Convert string scale to ScaleClass."""
    mapping = {
        "diagnostic": ScaleClass.DIAGNOSTIC,
        "lab": ScaleClass.LAB,
        "pilot": ScaleClass.PILOT,
        "field": ScaleClass.FIELD,
        "industrial": ScaleClass.INDUSTRIAL,
    }
    return mapping.get(scale.lower(), ScaleClass.LAB)


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def design_capture_transform_system(
    target_formula: str,
    matrix_species: Optional[dict[str, float]] = None,
    substrate_type: SubstrateType = SubstrateType.SILICA_BEADS,
    dg_bind_kj: float = -20.0,
    temperature_c: float = 25.0,
    target_scale: str = "lab",
    max_designs: int = 5,
) -> PipelineResult:
    """End-to-end capture-transform system design.

    Args:
        target_formula: Chemical formula of target (e.g., "CO2", "PO4_3-", "Pb2+")
        matrix_species: {formula: concentration_mM} for the source matrix
        substrate_type: Macroscopic deployment substrate
        dg_bind_kj: Estimated ΔG of initial target binding
        temperature_c: Operating temperature
        target_scale: "diagnostic", "lab", "pilot", "field", "industrial"
        max_designs: Maximum number of designs to return

    Returns:
        PipelineResult with ranked system designs.
    """
    if matrix_species is None:
        matrix_species = {}

    scale_class = _scale_from_string(target_scale)

    # ── Step 1: Enumerate transformation products ──
    products = enumerate_transformations(
        target_formula=target_formula,
        matrix_species=matrix_species,
        temperature_c=temperature_c,
    )

    if not products:
        return PipelineResult(
            target_formula=target_formula,
            matrix_species=matrix_species,
            n_products_enumerated=0,
            n_viable_pathways=0,
            designs=[],
        )

    # ── Step 2: Score all pathways ──
    pathway_scores = score_all_products(
        products=products,
        dg_bind_kj=dg_bind_kj,
        matrix_species=matrix_species,
        temperature_c=temperature_c,
    )

    viable = [s for s in pathway_scores if s.is_viable]

    # ── Step 3: Build system designs for top pathways ──
    designs = []
    for ps in pathway_scores[:max_designs]:
        product = ps.product
        element_type = _map_capture_element_type(product)
        cu_ok = _cu_tolerant(product)

        # Generate tether protocol
        try:
            tether = generate_tether_protocol(
                substrate_type=substrate_type,
                capture_element_type=element_type,
                cu_tolerant=cu_ok,
            )
        except (ValueError, KeyError):
            tether = None

        # Build cascade if applicable
        cascade_spec = None
        scaffold_rec = None
        scaffold_rationale = ""
        stoichiometry = None

        if product.benefits_from_confinement:
            cascade_spec = _build_cascade_spec_from_product(product, scale_class)
            if cascade_spec:
                scaffold_rec, scaffold_rationale = recommend_scaffold(cascade_spec)
                if scaffold_rec.feasible:
                    positions = scaffold_rec.scaffold.module_positions[1]
                    stoichiometry = optimize_stoichiometry(cascade_spec, positions)

        # Compute system score
        # Pathway score dominates (60%), substrate compatibility (15%),
        # cascade benefit (15%), scale match (10%)
        sys_score = 0.60 * ps.composite_score
        if tether:
            sys_score += 0.15 * (1.0 if tether.cu_safe or cu_ok else 0.5)
        if scaffold_rec and scaffold_rec.feasible:
            sys_score += 0.15 * scaffold_rec.composite
        elif not product.benefits_from_confinement:
            sys_score += 0.15 * 0.7  # flat surface OK — no penalty
        if ps.deployment_scale == target_scale:
            sys_score += 0.10
        elif ps.deployment_scale in ("field", "industrial") and target_scale in ("field", "industrial"):
            sys_score += 0.08

        design = SystemDesign(
            pathway_score=ps,
            tether_protocol=tether,
            cascade_spec=cascade_spec,
            scaffold_recommendation=scaffold_rec,
            scaffold_rationale=scaffold_rationale,
            stoichiometry=stoichiometry,
            system_score=round(sys_score, 4),
        )
        designs.append(design)

    # Sort by system score
    designs.sort(key=lambda d: d.system_score, reverse=True)

    # Recommendation
    recommended = designs[0] if designs else None
    rationale = ""
    if recommended:
        parts = [f"Best pathway: {recommended.pathway_score.name}"]
        if recommended.pathway_score.is_gold:
            parts.append("Gold-standard orthogonality (fully passive)")
        if recommended.scaffold_recommendation and recommended.scaffold_recommendation.feasible:
            parts.append(f"Scaffold: {recommended.scaffold_recommendation.scaffold.name}")
        if recommended.tether_protocol:
            parts.append(f"on {recommended.tether_protocol.substrate.name}")
        rationale = " | ".join(parts)

    return PipelineResult(
        target_formula=target_formula,
        matrix_species=matrix_species,
        n_products_enumerated=len(products),
        n_viable_pathways=len(viable),
        designs=designs,
        recommended=recommended,
        recommendation_rationale=rationale,
    )
